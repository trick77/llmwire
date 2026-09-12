package llmwire

import (
	_ "embed"
	"fmt"
	"sort"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// The profile registry.
//
// Profiles are hardcoded and embedded, never fetched at runtime. Published
// catalogues are an AUTHORING source — useful for cross-checking a field while
// writing a profile — but a runtime dependency on one buys a network call, a
// cache, a staleness window and a fallback table, and the fallback table is this
// file. It also means a catalogue's mistake becomes ours silently, and at least
// one of them has the reasoning options for a model in this registry wrong.

//go:embed profiles.yaml
var embeddedProfiles []byte

// Registry holds fully-resolved profiles, keyed by exact model id.
type Registry struct {
	byID map[string]*Profile
}

var (
	defaultOnce sync.Once
	defaultReg  *Registry
	defaultErr  error
)

// Default returns the built-in registry.
//
// It panics if the embedded profiles are malformed, because that is a
// programming error in this package rather than anything a caller can handle:
// the data is compiled in, so if it is broken here it is broken everywhere, and
// the tests would have caught it.
func Default() *Registry {
	defaultOnce.Do(func() {
		defaultReg, defaultErr = NewRegistry(embeddedProfiles)
	})
	if defaultErr != nil {
		panic("llmwire: embedded profiles.yaml is invalid: " + defaultErr.Error())
	}
	return defaultReg
}

// profileFile is the top level of profiles.yaml.
type profileFile struct {
	Profiles []yaml.Node `yaml:"profiles"`
}

// NewRegistry parses and validates a profile document.
//
// Everything is resolved here: bases merged, defaults applied, cross-field rules
// checked. A profile that survives this is complete, so no call site ever has to
// ask "what if this field is unset". A profile that does not survive fails the
// load rather than the request.
func NewRegistry(doc []byte) (*Registry, error) {
	var file profileFile
	if err := yaml.Unmarshal(doc, &file); err != nil {
		return nil, fmt.Errorf("llmwire: parsing profiles: %w", err)
	}
	if len(file.Profiles) == 0 {
		return nil, fmt.Errorf("llmwire: profiles document has no entries")
	}

	// Two passes. The first decodes every entry and records which YAML keys each
	// one actually set, because distinguishing "said nothing" from "said false"
	// is impossible after decoding into a struct and is exactly what the
	// derived-key check needs.
	entries := make([]decodedEntry, 0, len(file.Profiles))
	seen := map[string]bool{}

	for i, node := range file.Profiles {
		var p Profile
		if err := node.Decode(&p); err != nil {
			return nil, fmt.Errorf("llmwire: profile #%d: %w", i, err)
		}
		if p.ID == "" {
			return nil, fmt.Errorf("llmwire: profile #%d has no id", i)
		}
		if seen[p.ID] {
			return nil, fmt.Errorf("llmwire: duplicate profile id %q", p.ID)
		}
		seen[p.ID] = true
		entries = append(entries, decodedEntry{profile: p, keys: mappingKeys(node)})
	}

	// Index the unbased profiles, which are the only legal bases.
	bases := map[string]Profile{}
	for _, e := range entries {
		if e.profile.Base == "" {
			bases[e.profile.ID] = e.profile
		}
	}

	reg := &Registry{byID: make(map[string]*Profile, len(entries))}
	for _, e := range entries {
		p := e.profile
		if p.Base != "" {
			base, ok := bases[p.Base]
			if !ok {
				// Either the base does not exist, or it is itself based. Both
				// are refused, and the message says which.
				if other := findByID(entries, p.Base); other != nil {
					return nil, fmt.Errorf("llmwire: profile %q bases on %q, which itself has a base. "+
						"Single base, no chains: a chain makes \"which rule applied\" unanswerable",
						p.ID, p.Base)
				}
				return nil, fmt.Errorf("llmwire: profile %q bases on %q, which does not exist",
					p.ID, p.Base)
			}
			if err := checkDerivedKeys(p.ID, e.keys); err != nil {
				return nil, fmt.Errorf("llmwire: %w", err)
			}
			merged, err := p.resolve(base)
			if err != nil {
				return nil, fmt.Errorf("llmwire: resolving %q: %w", p.ID, err)
			}
			p = merged
		}
		applyDefaults(&p)
		if err := p.validate(); err != nil {
			return nil, fmt.Errorf("llmwire: %w", err)
		}
		stored := p
		reg.byID[p.ID] = &stored
	}
	return reg, nil
}

// decodedEntry pairs a decoded profile with the YAML keys the entry actually
// set. The key list is kept because distinguishing "said nothing" from "said
// false" is impossible once a value has been decoded into a struct, and that
// distinction is what the derived-key check rests on.
type decodedEntry struct {
	profile Profile
	keys    []string
}

// findByID locates a decoded entry, for producing a better error message.
func findByID(entries []decodedEntry, id string) *Profile {
	for i := range entries {
		if entries[i].profile.ID == id {
			return &entries[i].profile
		}
	}
	return nil
}

// applyDefaults fills the few fields that have a safe default, so nothing
// downstream applies one.
//
// The list is deliberately short. A default is only safe where every endpoint
// agrees; everywhere else an absent value means "not measured", and inventing
// one is how a profile comes to assert something nobody checked.
func applyDefaults(p *Profile) {
	if p.Endpoint == "" {
		p.Endpoint = EndpointChat
	}
	if p.WireModelID == "" {
		p.WireModelID = p.ID
	}
	if p.Verified == "" {
		p.Verified = VerifiedSource
	}
	if p.Endpoint == EndpointChat && p.Streaming.Supported && p.Reasoning.Supported && p.Reasoning.StreamField == "" {
		// Every reasoning model measured streams on this field, and an endpoint
		// that uses the other spelling is handled by the parser reading both.
		p.Reasoning.StreamField = "reasoning_content"
	}
	if p.Tools.Supported && p.Tools.Format == "" {
		p.Tools.Format = FormatNative
	}
}

// Lookup returns the profile for an exact model id.
//
// An unknown id is an ERROR, never a zero profile. The alternative — returning
// something empty and letting capability checks read false — turns a typo into
// "this model supports nothing", which is indistinguishable from a real answer
// and fails much later, somewhere unrelated. One widely-used library does
// exactly that, and it is why a missing model there presents as a mysteriously
// featureless one.
func (r *Registry) Lookup(id string) (*Profile, error) {
	if p, ok := r.byID[id]; ok {
		return p, nil
	}
	return nil, &UnknownModelError{ID: id, Known: sortedIDs(r.byID)}
}

// UnknownModelError names what was asked for and what is available, because an
// error that does not say which ids exist leaves the reader to guess at a typo.
type UnknownModelError struct {
	ID    string
	Known []string
}

func (e *UnknownModelError) Error() string {
	return fmt.Sprintf("llmwire: unknown model %q; the registry has %s",
		e.ID, strings.Join(e.Known, ", "))
}

// Models lists every registered id, sorted.
func (r *Registry) Models() []string { return sortedIDs(r.byID) }

// mappingKeys returns the keys of a YAML mapping node, which is how a caller
// tells "the field was omitted" from "the field was set to its zero value".
func mappingKeys(node yaml.Node) []string {
	if node.Kind != yaml.MappingNode {
		return nil
	}
	// Content alternates key, value, key, value.
	keys := make([]string, 0, len(node.Content)/2)
	for i := 0; i+1 < len(node.Content); i += 2 {
		keys = append(keys, node.Content[i].Value)
	}
	sort.Strings(keys)
	return keys
}
