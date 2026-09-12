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
//     json_schema downgraded to json_object; a sampling parameter the model
//     will ignore while it reasons.
//
// Warnings ride back with a SUCCESSFUL response rather than being logged,
// because Go has no warnings mechanism: a log line is invisible to the caller
// and an error is too blunt for something that worked.
//
// BestEffort demotes tier 1 to tier 3 for a single call. It is per-request on
// purpose. A process-wide equivalent exists in at least one popular library and
// is a landmine: the call whose behaviour silently changes is never the call
// that set the flag.
//
// WHAT BESTEFFORT DOES NOT TOUCH: a malformed request. A message list that is
// empty, a tool with no name, a tool result with nothing to tie it to — these
// are not capability mismatches, there is nothing to "send instead", and no
// endpoint accepts them. Demoting those would hand back a nil error and put the
// malformed request on the wire, turning a local failure into exactly the 400
// this package exists to catch first.

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

// UnsupportedError is a refusal: the request cannot be sent as written.
//
// The message always names what IS accepted where there is such a set. An error
// that only says "no" leaves the reader to go and find it, which for these
// models is not reliably documented anywhere.
type UnsupportedError struct {
	Model   string
	Feature string
	Reason  string
	// Accepted, when non-empty, is the set the model does take.
	Accepted []string
	// Demotable says whether BestEffort would have let this request through.
	// False for a malformed request and for a model/endpoint mismatch, where
	// suggesting the flag would be advice that does not work — or worse, advice
	// that suppresses the error and sends something broken.
	Demotable bool
}

func (e *UnsupportedError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "llmwire: model %q: %s", e.Model, e.Reason)
	if len(e.Accepted) > 0 {
		fmt.Fprintf(&b, "; accepted: %s", strings.Join(e.Accepted, ", "))
	}
	if e.Demotable {
		b.WriteString("; or pass BestEffort to send the nearest supported request")
	}
	return b.String()
}

// Validate checks a request against its model's profile without touching the
// network, returning the warnings the request would produce.
//
// Exported and network-free on purpose. It is what makes "fails fast" testable,
// and it lets a service check its configuration at boot rather than on the first
// user request.
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
		// Not demotable: there is no "nearest supported request" for a model
		// that does not serve this route at all.
		return nil, nil, &UnsupportedError{
			Model:   p.ID,
			Feature: "chat",
			Reason:  fmt.Sprintf("this is an %s model and cannot take a chat request", p.Endpoint),
		}
	}

	v := &validation{profile: p, bestEffort: req.BestEffort}
	// Whether this REQUEST will reason, not merely whether the model reasons by
	// default. The two differ exactly when the caller said something, which is
	// the case the inert-parameter warning is about.
	v.reasoningActive = reasoningActive(req.Reasoning, p.Reasoning)

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

// reasoningActive reports whether a request will actually make the model think.
//
// Reading this off the profile alone gets it backwards in both directions: a
// caller who switched thinking OFF would be warned that their temperature is
// inert, and a caller who switched it ON for a model that is off by default
// would not be warned at all — which is the case the warning exists for.
func reasoningActive(want ReasoningRequest, r Reasoning) bool {
	if !r.Supported {
		return false
	}
	switch req := want.(type) {
	case nil:
		return r.EnabledByDefault
	case reasoningOff:
		return false
	case reasoningEffort:
		return req.level != "none"
	case reasoningBudget:
		return req.tokens > 0
	default:
		return r.EnabledByDefault
	}
}

// validation accumulates warnings and the first refusal.
//
// The FIRST refusal, not all of them: a caller fixes one thing and runs again,
// and a list of complaints about a request that was rejected on its first
// problem is mostly noise about parameters that were never really evaluated.
type validation struct {
	profile         *Profile
	bestEffort      bool
	reasoningActive bool
	warnings        []Warning
	err             error
}

// reject records a MALFORMED request: something no endpoint would accept and
// that BestEffort cannot rescue, because there is nothing to send instead.
//
// Always hard. Always stops the check.
func (v *validation) reject(feature, reason string) {
	if v.err == nil {
		v.err = &UnsupportedError{
			Model:   v.profile.ID,
			Feature: feature,
			Reason:  reason,
		}
	}
}

// refuse records a capability mismatch, and reports whether it was a HARD
// refusal.
//
// Returns false when BestEffort demoted it to a warning, so the caller keeps
// checking the rest of the request. Returning early on a demoted refusal would
// silently skip every later check, which is how a BestEffort request reaches the
// wire carrying a second problem nobody looked for.
func (v *validation) refuse(feature, reason string, accepted []string) bool {
	if v.bestEffort {
		v.warn(WarnUnsupported, feature, reason+"; dropped because BestEffort was set")
		return false
	}
	if v.err == nil {
		v.err = &UnsupportedError{
			Model:     v.profile.ID,
			Feature:   feature,
			Reason:    reason,
			Accepted:  accepted,
			Demotable: true,
		}
	}
	return true
}

func (v *validation) warn(kind WarningKind, feature, details string) {
	v.warnings = append(v.warnings, Warning{Kind: kind, Feature: feature, Details: details})
}

func (v *validation) checkMessages(req ChatRequest) {
	if len(req.Messages) == 0 {
		v.reject("messages", "a chat request needs at least one message")
		return
	}
	for i, m := range req.Messages {
		// Structural: a tool result with nothing to tie it to is malformed
		// whatever the model supports.
		if m.Role == RoleTool && m.ToolCallID == "" {
			v.reject("messages",
				fmt.Sprintf("message %d has role %q but no ToolCallID to tie it to a call", i, RoleTool))
			return
		}
		if m.HasImage() && !v.profile.Vision {
			// Deliberately a refusal rather than a silent reroute to a
			// vision-capable sibling. Answering with a model the caller did not
			// name misattributes both the answer and its cost; choosing the
			// model is the application's job, and it has the context to do it.
			if v.refuse("vision",
				fmt.Sprintf("message %d carries an image and this model is text-only", i),
				nil) {
				return
			}
		}
	}
}

func (v *validation) checkReasoning(req ChatRequest) {
	if req.Reasoning == nil {
		return
	}
	r := v.profile.Reasoning

	switch want := req.Reasoning.(type) {
	case reasoningOff:
		// A model that never reasons already satisfies "do not reason". Refusing
		// would force model-agnostic callers to branch per model to ask for the
		// thing they are already getting.
		if !r.Supported {
			return
		}
		if !r.CanBeDisabled {
			// The motivating case for this whole package. One model in the
			// registry refuses the disable toggle with an error code it also
			// uses for unrelated failures, while its vendor's own reference
			// documents the toggle as valid.
			v.refuse("reasoning", "thinking cannot be disabled on this model", r.EffortValues)
			return
		}
		if r.Control == ControlEffort && !r.Accepts("none") {
			v.refuse("reasoning",
				`this model disables thinking by a means other than reasoning_effort "none"`,
				r.EffortValues)
		}

	case reasoningEffort:
		if !r.Supported {
			v.refuse("reasoning", "this model does not reason", nil)
			return
		}
		if r.Control != ControlEffort {
			v.refuse("reasoning", reasoningControlMismatch(r.Control, "an effort level"),
				reasoningControlHint(r.Control))
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
		if !r.Supported {
			v.refuse("reasoning", "this model does not reason", nil)
			return
		}
		if r.Control != ControlBudget {
			v.refuse("reasoning", reasoningControlMismatch(r.Control, "a token budget"),
				reasoningControlHint(r.Control))
		}
	}
}

// reasoningControlMismatch phrases a control mismatch in terms of what a CALLER
// would write, rather than leaking the profile's internal constant name.
func reasoningControlMismatch(have ReasoningControl, wanted string) string {
	return fmt.Sprintf("this model takes reasoning as %s, not as %s",
		reasoningControlPhrase(have), wanted)
}

func reasoningControlPhrase(c ReasoningControl) string {
	switch c {
	case ControlToggleObject:
		return "an on/off switch"
	case ControlEffort:
		return "an effort level"
	case ControlBudget:
		return "a token budget"
	default:
		return string(c)
	}
}

// reasoningControlHint names the constructor that would work, so a refusal is
// actionable rather than merely correct.
func reasoningControlHint(c ReasoningControl) []string {
	switch c {
	case ControlToggleObject:
		return []string{"ReasoningOff()"}
	case ControlEffort:
		return []string{"ReasoningEffort(level)"}
	case ControlBudget:
		return []string{"ReasoningBudget(tokens)"}
	default:
		return nil
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
	if s.InertWhileReasoning && v.reasoningActive {
		// Accepted by the endpoint and then overridden. Not an error — the
		// request succeeds — but believing it took effect is how a caller
		// spends a week tuning a parameter the model discarded.
		v.warn(WarnCompatibility, name,
			fmt.Sprintf("%s is accepted but overridden by the model while it is thinking, "+
				"so this value will not take effect", name))
	}
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
		if v.refuse("tools", "this model does not support tool calling", nil) {
			return
		}
	}
	for i, tool := range req.Tools {
		// Structural: a nameless tool cannot be called by any endpoint.
		if tool.Name == "" {
			v.reject("tools", fmt.Sprintf("tool %d has no name", i))
			return
		}
	}

	if req.ToolChoice.Mode == ToolChoiceUnset {
		return
	}
	if req.ToolChoice.Mode == ToolChoiceFunction && req.ToolChoice.Name == "" {
		v.reject("tool_choice", "a function tool choice needs the function's name")
		return
	}
	if toolChoiceAccepted(t.ToolChoiceValues, req.ToolChoice.Mode) {
		return
	}

	// "none" is not a relaxable constraint, and this is where the tier rule bites
	// rather than being a formality. Every other mode asks the model to PREFER a
	// tool, so relaxing it costs the caller some control. "none" asks the model
	// not to call one AT ALL, and relaxing that to "auto" permits exactly the
	// thing the caller ruled out — which is a change of meaning, hence tier 1.
	//
	// Dropping the tool definitions would achieve it, but that is not this
	// package's call to make silently: withholding the tools changes what the
	// model is told exists, which can change the answer even when no tool would
	// have been called. The caller is told to do it, and keeps the decision.
	if req.ToolChoice.Mode == ToolChoiceNone {
		v.refuse("tool_choice",
			fmt.Sprintf("this model accepts tool_choice %v and cannot be told %q; "+
				"omit Tools from the request to guarantee no tool call",
				t.ToolChoiceValues, ToolChoiceNone),
			t.ToolChoiceValues)
		return
	}

	// Tier 3: the request still runs, just less constrained. Warned rather than
	// dropped silently because several of these endpoints DROP the field
	// themselves without saying so, which is how a caller comes to believe a
	// call was forced when it was not.
	v.warn(WarnCompatibility, "tool_choice",
		fmt.Sprintf("this model accepts tool_choice %v; %q was relaxed to %q",
			t.ToolChoiceValues, req.ToolChoice.Mode, ToolChoiceAuto))
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
		// Structural: a schema format with no schema is malformed, and no
		// endpoint takes it.
		if f.Name == "" {
			v.reject("response_format", "a JSON schema response format needs a name")
			return
		}
		if len(f.Schema) == 0 {
			v.reject("response_format", "a JSON schema response format needs a schema")
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
			return
		}
		// StrictSchema describes exactly this field — whether "strict": true is
		// accepted inside a json_schema block. Tool-level strictness is a
		// different property that no profile measures, so a tool's Strict flag
		// is passed through unjudged rather than gated on an unrelated flag.
		if f.Strict && !v.profile.Output.StrictSchema {
			v.warn(WarnCompatibility, "response_format",
				"this model does not accept strict schema adherence; the flag was dropped, "+
					"so the schema is requested but not enforced")
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
