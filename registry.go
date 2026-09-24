package llmwire

import (
	"bytes"
	_ "embed"
	"fmt"
	"net/url"
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

// Registry holds fully-resolved profiles, keyed by exact model id, and the
// providers they are reached through.
type Registry struct {
	byID      map[string]*Profile
	providers map[string]Provider
}

// Provider is one host and account a profile is reached through: the thing
// `provider:` names. It carries the endpoint root so an application supplies a
// key and nothing else. The URL is a property of the provider, not of the
// model, which is why it lives here rather than on every profile.
type Provider struct {
	// BaseURL is the OpenAI-compatible root the client appends routes to.
	// Empty means the library ships no host for this provider (a self-hosted
	// gateway), and LLMWIRE_<PROVIDER>_BASE_URL is then required.
	BaseURL string `yaml:"base_url"`
	// EmulateOpenCode: this host is sold as opencode's backend and refuses a
	// neutral User-Agent as a bot, so FromEnv presents as that client. A
	// property of the host, which is why it is here and not in every
	// application's configuration. See Config.EmulateOpenCode.
	EmulateOpenCode bool `yaml:"emulate_opencode"`
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
	Providers map[string]yaml.Node `yaml:"providers"`
	Profiles  []yaml.Node          `yaml:"profiles"`
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
	providers := make(map[string]Provider, len(file.Providers))
	// In name order, so a document with two bad providers fails on the same
	// one every run; a map walk would name either.
	for _, name := range sortedKeys(file.Providers) {
		node := file.Providers[name]
		if !providerName.MatchString(name) {
			return nil, fmt.Errorf("llmwire: provider %q must be lowercase letters and digits", name)
		}
		var pv Provider
		if err := strictDecode(node, &pv); err != nil {
			return nil, fmt.Errorf("llmwire: provider %q: %w", name, err)
		}
		if err := pv.validate(); err != nil {
			return nil, fmt.Errorf("llmwire: provider %q: %w", name, err)
		}
		providers[name] = pv
	}

	// Two passes. The first decodes every entry and records which YAML keys each
	// one actually set, because distinguishing "said nothing" from "said false"
	// is impossible after decoding into a struct and is exactly what the
	// derived-key check needs.
	entries := make([]decodedEntry, 0, len(file.Profiles))
	seen := map[string]bool{}

	for i, node := range file.Profiles {
		var p Profile
		if err := strictDecode(node, &p); err != nil {
			return nil, fmt.Errorf("llmwire: profile #%d: %w", i, err)
		}
		if p.ID == "" {
			return nil, fmt.Errorf("llmwire: profile #%d has no id", i)
		}
		if seen[p.ID] {
			return nil, fmt.Errorf("llmwire: duplicate profile id %q", p.ID)
		}
		seen[p.ID] = true
		entries = append(entries, decodedEntry{profile: p, node: node})
	}

	// Index the unbased profiles, which are the only legal bases.
	//
	// Defaults are applied HERE, before anything inherits from them. Resolving
	// first and defaulting afterwards looks equivalent and is not: a base that
	// omits wire_model_id would hand the derived entry an empty one, which the
	// derived entry then fills from its OWN id — so a gateway's registry key
	// would go on the wire as the model name, with no error anywhere.
	bases := map[string]Profile{}
	for _, e := range entries {
		if e.profile.Base == "" {
			base := e.profile
			applyDefaults(&base)
			bases[base.ID] = base
		}
	}

	reg := &Registry{byID: make(map[string]*Profile, len(entries)), providers: providers}
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
			if err := checkDerivedKeys(p.ID, e.node); err != nil {
				return nil, fmt.Errorf("llmwire: %w", err)
			}
			// A different provider is a different host, and a different host
			// is a different party pricing the call. Naming the gateway is
			// what drops the base's cost block; without it the route would
			// keep the vendor's list rate for calls the vendor never sees,
			// and be priced from the table behind a proxy.
			if p.Provider != "" && p.Provider != base.Provider && p.Gateway == "" {
				return nil, fmt.Errorf("llmwire: profile %q changes provider from %q to %q but names no gateway; "+
					"a route through another host must say so, so it is priced by that host and never from the table",
					p.ID, base.Provider, p.Provider)
			}
			p = p.resolve(base)
		}
		applyDefaults(&p)
		if err := p.validate(); err != nil {
			return nil, fmt.Errorf("llmwire: %w", err)
		}
		// Resolved here, after the base merge, because provider REPLACES on a
		// derived profile and the URL follows the provider. A document with no
		// providers: section at all is allowed (every host from the
		// environment); one that HAS the section must name every provider a
		// profile uses, or a typo in `provider:` would surface as a missing
		// environment variable and send the operator to the wrong file.
		if p.Provider != "" && len(providers) > 0 {
			if _, ok := providers[p.Provider]; !ok {
				return nil, fmt.Errorf("llmwire: profile %q names provider %q, which is not in providers: (known: %v)",
					p.ID, p.Provider, sortedKeys(providers))
			}
		}
		p.BaseURL = providers[p.Provider].BaseURL
		p.EmulateOpenCode = providers[p.Provider].EmulateOpenCode
		stored := p
		reg.byID[p.ID] = &stored
	}
	return reg, nil
}

// decodedEntry pairs a decoded profile with the YAML node it came from. The
// node is kept because distinguishing "said nothing" from "said false" is
// impossible once a value has been decoded into a struct, and that distinction
// is what the derived-key check rests on.
type decodedEntry struct {
	profile Profile
	node    yaml.Node
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

// Provider returns the host a provider name resolves to. An unknown name is
// an error rather than an empty Provider, for the same reason Lookup errors:
// a typo must not read as "no host shipped".
func (r *Registry) Provider(name string) (Provider, error) {
	if pv, ok := r.providers[name]; ok {
		return pv, nil
	}
	return Provider{}, fmt.Errorf("llmwire: unknown provider %q; known: %v", name, sortedKeys(r.providers))
}

// sortedKeys orders a map's keys, so every listing and every error names
// things in the same order on every run.
func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// validate checks a provider's host. A trailing slash is trimmed at use, so
// the check is on things that would go wrong silently: a scheme other than
// https sends the key in clear, and a query string is where a key ends up
// echoed in every error body (see the redaction notes in errors.go).
func (pv Provider) validate() error {
	if pv.BaseURL == "" {
		return nil
	}
	return validateBaseURL(pv.BaseURL, false)
}

// validateBaseURL is the host check, shared by the shipped hosts and the one
// read from the environment. allowLoopbackHTTP admits http:// for a host on
// this machine only: a self-hosted gateway on localhost has no TLS and no
// wire for the key to cross, while the same scheme on any other host sends
// the key in clear.
func validateBaseURL(raw string, allowLoopbackHTTP bool) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("base_url %q: %w", raw, err)
	}
	switch {
	case u.Scheme == "http" && allowLoopbackHTTP && isLoopbackHost(u.Hostname()):
	case u.Scheme != "https":
		return fmt.Errorf("base_url %q must be https", raw)
	}
	switch {
	case u.Host == "":
		return fmt.Errorf("base_url %q has no host", raw)
	case u.RawQuery != "" || u.Fragment != "" || u.User != nil:
		return fmt.Errorf("base_url %q must be a bare root: no query, fragment or credentials", raw)
	case strings.HasSuffix(strings.TrimRight(u.Path, "/"), "/chat/completions"),
		strings.HasSuffix(strings.TrimRight(u.Path, "/"), "/embeddings"):
		return fmt.Errorf("base_url %q ends in a route; the client appends /chat/completions and /embeddings itself", raw)
	}
	return nil
}

// isLoopbackHost is the literal loopback names only; a hostname that merely
// resolves to loopback is not trusted, since resolution is not this
// package's to inspect.
func isLoopbackHost(host string) bool {
	switch strings.ToLower(host) {
	case "localhost", "127.0.0.1", "::1":
		return true
	}
	return false
}

// Lookup returns the profile for an exact model id.
//
// An unknown id is an ERROR, never a zero profile. The alternative — returning
// something empty and letting capability checks read false — turns a typo into
// "this model supports nothing", which is indistinguishable from a real answer
// and fails much later, somewhere unrelated. One widely-used library does
// exactly that, and it is why a missing model there presents as a mysteriously
// featureless one.
// The returned profile is a COPY. The registry is cached for the life of the
// process and shared by every caller, so returning the stored pointer would make
// one caller's experiment everyone else's configuration.
func (r *Registry) Lookup(id string) (*Profile, error) {
	if p, ok := r.byID[id]; ok {
		return p.clone(), nil
	}
	return nil, &UnknownModelError{ID: id, Known: sortedKeys(r.byID)}
}

// LookupEmbedding is Lookup for a model that must be an embeddings model. An
// embeddings client that is handed a chat id fails at its first request with
// a body about messages; this fails at construction, naming the id, which is
// where a build-time constant is checked. The width is guaranteed by the
// profile: an embeddings profile without default_dimensions does not load.
func (r *Registry) LookupEmbedding(id string) (*Profile, error) {
	p, err := r.Lookup(id)
	if err != nil {
		return nil, err
	}
	if p.Endpoint != EndpointEmbeddings {
		return nil, fmt.Errorf("llmwire: model %q is a %s model, not embeddings", id, p.Endpoint)
	}
	return p, nil
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
func (r *Registry) Models() []string { return sortedKeys(r.byID) }

// strictDecode decodes one entry, refusing unknown keys.
//
// yaml.Node.Decode has no strict mode, so the node is re-encoded and read back
// through a decoder that does. The round trip is worth it: without it a typo
// silently becomes a zero value, so "visoin: true" loads clean and leaves
// vision false. For a package whose entire purpose is that a wrong assumption
// fails loudly, a mistyped capability turning into an assumed-absent one is the
// worst available failure.
func strictDecode(node yaml.Node, into any) error {
	raw, err := yaml.Marshal(&node)
	if err != nil {
		return fmt.Errorf("re-encoding entry: %w", err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	return dec.Decode(into)
}

// isYAMLFalse asks the decoder itself whether a scalar reads as false, so
// every spelling yaml.v3 accepts ("no", "off", "False") is caught and no
// second table of them lives here.
func isYAMLFalse(node *yaml.Node) bool {
	var b bool
	return strictDecode(*node, &b) == nil && !b
}

// checkDerivedKeys rejects a based profile that sets a capability field.
//
// It walks the YAML node rather than the decoded struct, because that is the
// only way to tell "said nothing" from "said false". Nested keys are walked too:
// admitting a whole mapping and trusting resolve to copy the fields it knows
// about would silently discard the rest.
func checkDerivedKeys(id string, node yaml.Node) error {
	if node.Kind != yaml.MappingNode {
		return nil
	}
	var offending, loosened []string
	for i := 0; i+1 < len(node.Content); i += 2 {
		key, value := node.Content[i].Value, node.Content[i+1]
		if !derivedAllowedKeys[key] {
			offending = append(offending, key)
			continue
		}
		allowedSub, nested := derivedAllowedNestedKeys[key]
		if !nested || value.Kind != yaml.MappingNode {
			continue
		}
		for j := 0; j+1 < len(value.Content); j += 2 {
			sub, subValue := value.Content[j].Value, value.Content[j+1]
			switch {
			case !allowedSub[sub]:
				offending = append(offending, key+"."+sub)
			case subValue.Kind == yaml.ScalarNode && isYAMLFalse(subValue):
				// Tighten-only means ON is the only value a route can say.
				// resolve would ignore a false, so it is refused rather than
				// loaded as a no-op the author thinks took effect.
				loosened = append(loosened, key+"."+sub)
			}
		}
	}
	if len(loosened) > 0 {
		sort.Strings(loosened)
		return fmt.Errorf("profile %q sets %v to false, but those keys are tighten-only: "+
			"a route can switch them on, never off, so a false would load and change nothing", id, loosened)
	}
	if len(offending) == 0 {
		return nil
	}
	sort.Strings(offending)
	return fmt.Errorf("profile %q sets %v, but a profile with a base inherits every "+
		"capability from it and may only override routing. A capability belongs to the "+
		"model, not to the route: if this deployment really differs, give it its own "+
		"profile with no base", id, offending)
}
