package llmwire

import (
	"fmt"
	"sort"
	"strings"
)

// Request validation, and the policy behind it.
//
// The question every adapter has to answer is what to do when a caller asks for
// something the model cannot do. Surveying how others answer it: one gateway
// raises on one code path, silently drops on a second and passes through
// untouched on a third; one SDK never throws and returns warnings; one library
// errors in the constructor. The inconsistency is the lesson — pick one policy
// and apply it everywhere.
//
// This package sorts every case by how LOSSY the fix is and by WHO asked:
//
//  1. HARD ERROR, before any socket opens, when honouring the request would
//     change its meaning and the caller asked explicitly. Disabling thinking on
//     a model that cannot, an effort level outside the accepted set, a
//     non-default temperature where only the default is taken, an image for a
//     text-only model.
//
//  2. SILENT COERCION when the transformation is lossless. Choosing which
//     output-cap parameter to send. Adding stream_options where the endpoint
//     needs it. Nobody wants a warning for a field rename.
//
//  3. COERCION PLUS A WARNING when the request can be honoured approximately.
//     json_schema downgraded to json_object; a forced tool choice relaxed to
//     auto where the endpoint only takes auto; a sampling parameter that the
//     model will ignore while it reasons.
//
// Warnings ride back with a SUCCESSFUL response rather than being logged,
// because Go has no warnings mechanism: a log line is invisible to the caller
// and an error is too blunt for something that worked.
//
// BestEffort demotes tier 1 to tier 3 for a single call. It is per-request on
// purpose. A process-wide equivalent exists in at least one popular library and
// is a landmine: the call whose behaviour silently changes is never the call
// that set the flag.

// WarningKind classifies a warning. Three cases, deliberately: a longer
// taxonomy invites callers to switch on distinctions that do not change what
// they should do.
type WarningKind string

const (
	// WarnUnsupported: the request asked for something the model does not
	// support, and it was dropped.
	WarnUnsupported WarningKind = "unsupported"
	// WarnCompatibility: the request was honoured approximately. The call
	// worked; the result may differ from what was asked for.
	WarnCompatibility WarningKind = "compatibility"
	// WarnOther is anything that fits neither.
	WarnOther WarningKind = "other"
)

// Warning is something the caller should know about a request that otherwise
// succeeded.
type Warning struct {
	Kind WarningKind
	// Feature names the request field concerned.
	Feature string
	// Details says what happened to it, and why.
	Details string
}

func (w Warning) String() string {
	return fmt.Sprintf("%s: %s (%s)", w.Kind, w.Feature, w.Details)
}

// UnsupportedError is a tier-1 refusal: the model cannot do what was asked, and
// doing something else instead would change what the caller meant.
//
// The message always names what IS accepted. An error that only says "no"
// leaves the reader to go and find the accepted set, which for these models is
// not reliably documented anywhere.
type UnsupportedError struct {
	Model   string
	Feature string
	Reason  string
	// Accepted, when non-empty, is the set the model does take.
	Accepted []string
}

func (e *UnsupportedError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "llmwire: model %q: %s", e.Model, e.Reason)
	if len(e.Accepted) > 0 {
		fmt.Fprintf(&b, "; accepted: %s", strings.Join(e.Accepted, ", "))
	}
	b.WriteString("; or pass BestEffort to send the nearest supported request")
	return b.String()
}

// Validate checks a request against its model's profile without touching the
// network, returning the warnings the request would produce.
//
// Exported and network-free on purpose. It is what makes "fails fast" testable,
// and it lets a service check its configuration at boot rather than on the first
// user request. Chat and ChatStream call it internally, so a caller that skips
// it loses nothing but the early answer.
func (c *Client) Validate(req ChatRequest) ([]Warning, error) {
	_, warnings, err := c.plan(req)
	return warnings, err
}

// plan resolves a request against its profile: the profile it will use, the
// warnings it produces, and any refusal.
func (c *Client) plan(req ChatRequest) (*Profile, []Warning, error) {
	p, err := c.registry.Lookup(req.Model)
	if err != nil {
		return nil, nil, err
	}
	if p.Endpoint != EndpointChat {
		return nil, nil, &UnsupportedError{
			Model:   p.ID,
			Feature: "chat",
			Reason:  fmt.Sprintf("this is an %s model and cannot take a chat request", p.Endpoint),
		}
	}

	v := &validation{profile: p, bestEffort: req.BestEffort}
	v.checkMessages(req)
	v.checkReasoning(req)
	v.checkSampling("temperature", req.Temperature, p.Temperature)
	v.checkSampling("top_p", req.TopP, p.TopP)
	v.checkTools(req)
	v.checkResponseFormat(req)

	if v.err != nil {
		return nil, nil, v.err
	}
	return p, v.warnings, nil
}

// validation accumulates warnings and the first refusal.
//
// The FIRST refusal, not all of them: a caller fixes one thing and runs again,
// and a list of complaints about a request that was rejected on its first
// problem is mostly noise about parameters that were never really evaluated.
type validation struct {
	profile    *Profile
	bestEffort bool
	warnings   []Warning
	err        error
}

// refuse records a tier-1 refusal — unless BestEffort demotes it to a tier-3
// warning, which is the whole of what that flag does.
func (v *validation) refuse(feature, reason string, accepted []string) {
	if v.bestEffort {
		v.warn(WarnUnsupported, feature, reason+"; dropped because BestEffort was set")
		return
	}
	if v.err == nil {
		v.err = &UnsupportedError{
			Model:    v.profile.ID,
			Feature:  feature,
			Reason:   reason,
			Accepted: accepted,
		}
	}
}

func (v *validation) warn(kind WarningKind, feature, details string) {
	v.warnings = append(v.warnings, Warning{Kind: kind, Feature: feature, Details: details})
}

func (v *validation) checkMessages(req ChatRequest) {
	if len(req.Messages) == 0 {
		v.refuse("messages", "a chat request needs at least one message", nil)
		return
	}
	for i, m := range req.Messages {
		if m.HasImage() && !v.profile.Vision {
			// Deliberately a refusal rather than a silent reroute to a
			// vision-capable sibling. Answering with a model the caller did not
			// name misattributes both the answer and its cost; choosing the
			// model is the application's job, and it has the context to do it.
			v.refuse("vision",
				fmt.Sprintf("message %d carries an image and this model is text-only", i),
				nil)
			return
		}
		if m.Role == RoleTool && m.ToolCallID == "" {
			v.refuse("messages",
				fmt.Sprintf("message %d has role %q but no ToolCallID to tie it to a call", i, RoleTool),
				nil)
			return
		}
	}
}

func (v *validation) checkReasoning(req ChatRequest) {
	if req.Reasoning == nil {
		return
	}
	r := v.profile.Reasoning
	if !r.Supported {
		v.refuse("reasoning", "this model does not reason", nil)
		return
	}

	switch want := req.Reasoning.(type) {
	case reasoningOff:
		if !r.CanBeDisabled {
			// The motivating case for this whole package. One model in the
			// registry refuses the disable toggle with an error code it also
			// uses for unrelated failures, while its vendor's own reference
			// documents the toggle as valid.
			v.refuse("reasoning",
				"thinking cannot be disabled on this model",
				r.EffortValues)
			return
		}
		if r.Control == ControlEffort && !r.Accepts("none") {
			v.refuse("reasoning",
				`this model disables thinking by a means other than reasoning_effort "none"`,
				r.EffortValues)
		}

	case reasoningEffort:
		if r.Control != ControlEffort {
			v.refuse("reasoning",
				fmt.Sprintf("this model takes reasoning as %q, not as an effort level", r.Control),
				nil)
			return
		}
		if !r.Accepts(want.level) {
			// The accepted set is per-model and narrower than the vendor's
			// global enum, so naming it is the difference between a fixable
			// error and a guess.
			v.refuse("reasoning_effort",
				fmt.Sprintf("effort %q is not accepted by this model", want.level),
				r.EffortValues)
		}

	case reasoningBudget:
		if r.Control != ControlBudget {
			v.refuse("reasoning",
				fmt.Sprintf("this model takes reasoning as %q, not as a token budget", r.Control),
				nil)
		}
	}
}

// checkSampling covers the three ways a sampling parameter can fail to mean what
// the caller thinks, which a single "supported" bool cannot express.
func (v *validation) checkSampling(name string, want *float64, s Sampling) {
	if want == nil {
		return
	}
	if !s.Supported {
		v.refuse(name, fmt.Sprintf("%s is not supported by this model", name), nil)
		return
	}
	if s.ForcedValue != nil && *want != *s.ForcedValue {
		v.refuse(name,
			fmt.Sprintf("%s must be %g on this model; any other value is rejected", name, *s.ForcedValue),
			[]string{fmt.Sprintf("%g", *s.ForcedValue)})
		return
	}
	if s.InertWhileReasoning && v.reasoningWillBeOn() {
		// Accepted by the endpoint and then overridden. Not an error — the
		// request succeeds — but believing it took effect is how a caller
		// spends a week tuning a parameter the model discarded.
		v.warn(WarnCompatibility, name,
			fmt.Sprintf("%s is accepted but overridden by the model while it is thinking, "+
				"so this value will not take effect", name))
	}
}

// reasoningWillBeOn reports whether this request will reason, which decides
// whether an inert-while-reasoning parameter matters.
func (v *validation) reasoningWillBeOn() bool {
	return v.profile.Reasoning.Supported && v.profile.Reasoning.EnabledByDefault
}

func (v *validation) checkTools(req ChatRequest) {
	if len(req.Tools) == 0 {
		if req.ToolChoice.Mode != ToolChoiceUnset {
			v.warn(WarnUnsupported, "tool_choice",
				"a tool choice was set but no tools were offered, so it was dropped")
		}
		return
	}
	t := v.profile.Tools
	if !t.Supported {
		v.refuse("tools", "this model does not support tool calling", nil)
		return
	}
	for i, tool := range req.Tools {
		if tool.Name == "" {
			v.refuse("tools", fmt.Sprintf("tool %d has no name", i), nil)
			return
		}
		if tool.Strict && !v.profile.Output.StrictSchema {
			v.warn(WarnCompatibility, "tools",
				fmt.Sprintf("tool %q asked for a strict schema, which this model does not "+
					"accept; the flag was dropped", tool.Name))
		}
	}

	if req.ToolChoice.Mode == ToolChoiceUnset {
		return
	}
	if req.ToolChoice.Mode == ToolChoiceFunction && req.ToolChoice.Name == "" {
		v.refuse("tool_choice", "a function tool choice needs the function's name", nil)
		return
	}
	if !toolChoiceAccepted(t.ToolChoiceValues, req.ToolChoice.Mode) {
		// Tier 3 rather than tier 1: the request still runs, just less
		// constrained. Warned rather than dropped silently because several of
		// these endpoints DROP the field themselves without saying so, which is
		// how a caller comes to believe a call was forced when it was not.
		v.warn(WarnCompatibility, "tool_choice",
			fmt.Sprintf("this model accepts tool_choice %v; %q was relaxed to %q",
				t.ToolChoiceValues, req.ToolChoice.Mode, ToolChoiceAuto))
	}
}

func toolChoiceAccepted(accepted []string, mode ToolChoiceMode) bool {
	for _, a := range accepted {
		if a == string(mode) {
			return true
		}
	}
	return false
}

func (v *validation) checkResponseFormat(req ChatRequest) {
	f := req.ResponseFormat
	switch f.Kind {
	case FormatUnset, FormatText:
		return
	case FormatJSONObject:
		if !v.profile.Output.JSONObject {
			v.refuse("response_format", "this model does not support JSON object output", nil)
		}
	case FormatJSONSchema:
		if f.Name == "" {
			v.refuse("response_format", "a JSON schema response format needs a name", nil)
			return
		}
		if len(f.Schema) == 0 {
			v.refuse("response_format", "a JSON schema response format needs a schema", nil)
			return
		}
		if !v.profile.Output.JSONSchema {
			if !v.profile.Output.JSONObject {
				v.refuse("response_format", "this model supports no structured output", nil)
				return
			}
			// Approximate but honourable: the reply is still constrained to
			// JSON, just not to this shape. Exactly the case warnings exist for.
			v.warn(WarnCompatibility, "response_format",
				"this model does not support json_schema; downgraded to json_object, "+
					"so the schema is not enforced")
		}
	default:
		v.refuse("response_format",
			fmt.Sprintf("unknown response format %q", f.Kind),
			[]string{string(FormatText), string(FormatJSONObject), string(FormatJSONSchema)})
	}
}

// Warnings renders a list for a log line, sorted so the output is stable.
func Warnings(ws []Warning) string {
	if len(ws) == 0 {
		return ""
	}
	out := make([]string, len(ws))
	for i, w := range ws {
		out[i] = w.String()
	}
	sort.Strings(out)
	return strings.Join(out, "; ")
}
