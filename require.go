package llmwire

import (
	"fmt"
	"strings"
)

// Role checks. An application gives each lane a role (the vision lane takes
// images, the agent lane calls tools) and a model id from config. Checking the
// pair at boot is a registry question, answered here once, so no application
// carries its own list of which models can do what.

// Needs is what a role requires of a chat model. The zero value requires
// nothing beyond being a chat model.
type Needs struct {
	// Tools: native or recovered tool calling (Profile.Tools.Supported).
	Tools bool
	// Vision: image parts (Profile.Vision).
	Vision bool
	// JSONObject: response_format json_object.
	JSONObject bool
	// JSONSchema: response_format json_schema, enforced rather than downgraded.
	JSONSchema bool
	// Streaming: ChatStream.
	Streaming bool
}

// missing lists what p lacks of n, in field order.
func (n Needs) missing(p *Profile) []string {
	var out []string
	for _, c := range []struct {
		want, have bool
		name       string
	}{
		{n.Tools, p.Tools.Supported, "tools"},
		{n.Vision, p.Vision, "vision"},
		{n.JSONObject, p.Output.JSONObject, "json_object"},
		{n.JSONSchema, p.Output.JSONSchema, "json_schema"},
		{n.Streaming, p.Streaming.Supported, "streaming"},
	} {
		if c.want && !c.have {
			out = append(out, c.name)
		}
	}
	return out
}

// CapabilityError is a model that cannot fill a role: which model, what it
// lacks, and which registered chat models would do, so a boot error reads as
// the fix.
type CapabilityError struct {
	Model   string
	Missing []string
	// Satisfying is every chat model in the registry that meets the same
	// Needs, sorted.
	Satisfying []string
}

func (e *CapabilityError) Error() string {
	msg := fmt.Sprintf("llmwire: model %q lacks %s", e.Model, strings.Join(e.Missing, ", "))
	if len(e.Satisfying) == 0 {
		return msg + "; no registered chat model has all of it"
	}
	return msg + "; valid choices are " + strings.Join(e.Satisfying, ", ")
}

// Require returns the profile for id if it is a chat model meeting needs. An
// unknown id is an *UnknownModelError, a model short of a need a
// *CapabilityError naming every gap and every model that has none.
func (r *Registry) Require(id string, needs Needs) (*Profile, error) {
	p, err := r.Lookup(id)
	if err != nil {
		return nil, err
	}
	if p.Endpoint != EndpointChat {
		return nil, fmt.Errorf("llmwire: model %q is a %s model, not chat; valid choices are %s",
			id, p.Endpoint, strings.Join(r.ChatModels(needs), ", "))
	}
	if gaps := needs.missing(p); len(gaps) > 0 {
		return nil, &CapabilityError{Model: id, Missing: gaps, Satisfying: r.ChatModels(needs)}
	}
	return p, nil
}

// ChatModels lists the chat model ids that meet needs, sorted: the "valid
// choices are ..." of a configuration error.
func (r *Registry) ChatModels(needs Needs) []string {
	var out []string
	for _, id := range sortedKeys(r.byID) {
		p := r.byID[id]
		if p.Endpoint == EndpointChat && len(needs.missing(p)) == 0 {
			out = append(out, id)
		}
	}
	return out
}
