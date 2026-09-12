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
// and an error is too blunt for something that worked. Chat, ChatStream and Embed
// each return them beside their result; on a stream the pricing warnings can only
// exist once usage has arrived, so Stream.Warnings() carries the complete set.
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
	_, warnings, err := c.plan(req, false)
	return warnings, err
}

// wirePlan is what plan produces and the renderer consumes: the request with
// every capability decision ALREADY APPLIED, plus the two wire-level knobs that
// are not request fields at all.
//
// A coerced request rather than a list of decisions, deliberately. A parallel
// decision struct duplicates the request's shape and lets the two drift, and
// worse, it leaves the renderer free to re-derive a decision from the profile —
// which is how a policy comes to live in two places and disagree with itself.
// Here there is nothing else to read.
//
// The invariant, which a reviewer can check by grepping render.go for
// ".Supported": the renderer reads WIRE-NAME profile fields only — WireModelID,
// MaxTokensParam, Reasoning.Control, Reasoning.BudgetParam, Streaming — and never
// a capability field. Capabilities were decided here.
type wirePlan struct {
	profile *Profile
	req     ChatRequest
	stream  bool
	// capParam is which output-cap parameter this endpoint honours. Copied so
	// the renderer never reaches back into the profile for it: one endpoint
	// accepts the wrong name and silently ignores it, returning thirty times the
	// requested tokens, so this is the single most expensive field to get wrong.
	capParam string
	// includeUsage says to send stream_options.include_usage.
	includeUsage bool
}

// plan resolves a request against its profile and returns it coerced, ready to
// render.
//
// stream is a parameter rather than a ChatRequest field on purpose: a request
// that could carry Stream: true and be handed to Chat would be a contradiction
// with no correct resolution, and the method the caller chose is never ambiguous.
func (c *Client) plan(req ChatRequest, stream bool) (*wirePlan, []Warning, error) {
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
	if stream && !p.Streaming.Supported {
		// Also not demotable: the caller chose a method, and there is nothing to
		// demote a method to. (No profile in the registry reaches this branch
		// today; it exists so adding a non-streaming model is data, not code.)
		return nil, nil, &UnsupportedError{
			Model:   p.ID,
			Feature: "streaming",
			Reason:  "this model does not stream; use Chat instead",
		}
	}

	v := &validation{profile: p, bestEffort: req.BestEffort, out: req.clone()}
	// Whether this REQUEST will reason, not merely whether the model reasons by
	// default. The two differ exactly when the caller said something, which is
	// the case the inert-parameter warning is about.
	v.reasoningActive = reasoningActive(req.Reasoning, p.Reasoning)

	v.checkMessages(req)
	v.checkReasoning(req)
	// Recomputed from the COERCED request, because checkReasoning may just have
	// dropped the caller's request under BestEffort. A caller who asked a
	// glm-5.3-flash to stop thinking and was demoted still gets a thinking
	// model, so their temperature really is inert — and computing this once, up
	// front, from the original request would say the opposite.
	v.reasoningActive = reasoningActive(v.out.Reasoning, p.Reasoning)
	v.checkSampling("temperature", req.Temperature, p.Temperature, &v.out.Temperature)
	v.checkSampling("top_p", req.TopP, p.TopP, &v.out.TopP)
	v.checkTools(req)
	v.checkResponseFormat(req)
	v.checkMaxTokens(req)

	if v.err != nil {
		return nil, nil, v.err
	}
	return &wirePlan{
		profile:  p,
		req:      v.out,
		stream:   stream,
		capParam: p.MaxTokensParam,
		// Sent only where the endpoint needs it AND accepts it. A gateway that
		// strips the usage chunk from a client which did not ask forces
		// needs_include_usage on for its own route, which is what that
		// tighten-only profile key exists for.
		includeUsage: stream && p.Streaming.NeedsIncludeUsage && p.Streaming.AcceptsStreamOptions,
	}, v.warnings, nil
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
	profile    *Profile
	bestEffort bool
	// out is the request as it will be sent: a deep copy, coerced in place as
	// each check decides. Deep because Validate is exported — a validation call
	// that stripped an image from the caller's own slice would be a data-loss
	// bug in a function documented as read-only.
	out             ChatRequest
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
			// Demoted: send the turn without its image. Dropping the parts
			// wholesale would take the text with it, so only image parts go.
			v.dropImages(i)
		}
	}
}

// dropImages strips the image parts from one coerced message.
//
// When nothing but images remains, Parts is emptied and the renderer falls back
// to Text — which may be empty. Whether these endpoints accept an empty content
// string is UNMEASURED; no probe has sent one. The fallback ships because the
// alternative is inventing a refusal for a 400 nobody has seen, and it is a
// probe candidate.
func (v *validation) dropImages(i int) {
	m := &v.out.Messages[i]
	kept := m.Parts[:0]
	for _, part := range m.Parts {
		if part.Kind != PartImage {
			kept = append(kept, part)
		}
	}
	m.Parts = kept
	if len(m.Parts) == 0 {
		m.Parts = nil
	}
}

func (v *validation) checkReasoning(req ChatRequest) {
	if req.Reasoning == nil {
		return
	}
	r := v.profile.Reasoning

	// Every refusal below is followed by dropReasoning on the demoted path. A
	// bare refuse() whose return is ignored would leave the coerced request
	// carrying the very knob that was just refused, and the renderer would send
	// it — turning a caught error into the 400 this package exists to prevent.
	switch want := req.Reasoning.(type) {
	case reasoningOff:
		// A model that never reasons already satisfies "do not reason". Refusing
		// would force model-agnostic callers to branch per model to ask for the
		// thing they are already getting. The knob is still dropped: there is no
		// wire spelling for disabling what does not exist.
		if !r.Supported {
			v.dropReasoning()
			return
		}
		if !r.CanBeDisabled {
			// The motivating case for this whole package. One model in the
			// registry refuses the disable toggle with an error code it also
			// uses for unrelated failures, while its vendor's own reference
			// documents the toggle as valid.
			if !v.refuse("reasoning", "thinking cannot be disabled on this model", r.EffortValues) {
				v.dropReasoning()
			}
			return
		}
		if r.Control == ControlEffort && !r.Accepts("none") {
			if !v.refuse("reasoning",
				`this model disables thinking by a means other than reasoning_effort "none"`,
				r.EffortValues) {
				v.dropReasoning()
			}
		}

	case reasoningEffort:
		if !r.Supported {
			if !v.refuse("reasoning", "this model does not reason", nil) {
				v.dropReasoning()
			}
			return
		}
		if r.Control != ControlEffort {
			if !v.refuse("reasoning", reasoningControlMismatch(r.Control, "an effort level"),
				reasoningControlHint(r.Control)) {
				v.dropReasoning()
			}
			return
		}
		if !r.Accepts(want.level) {
			// The accepted set is per-model and narrower than the vendor's
			// global enum, so naming it is the difference between a fixable
			// error and a guess.
			if !v.refuse("reasoning_effort",
				fmt.Sprintf("effort %q is not accepted by this model", want.level),
				r.EffortValues) {
				v.dropReasoning()
			}
		}

	case reasoningBudget:
		if !r.Supported {
			if !v.refuse("reasoning", "this model does not reason", nil) {
				v.dropReasoning()
			}
			return
		}
		if r.Control != ControlBudget {
			if !v.refuse("reasoning", reasoningControlMismatch(r.Control, "a token budget"),
				reasoningControlHint(r.Control)) {
				v.dropReasoning()
			}
		}
	}
}

// dropReasoning removes the reasoning knob from the coerced request, leaving the
// model at its own default.
func (v *validation) dropReasoning() { v.out.Reasoning = nil }

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
// out points at the coerced request's field for this parameter, so the resolved
// value and any drop land where the renderer will read them.
func (v *validation) checkSampling(name string, want *float64, s Sampling, out **float64) {
	if want == nil {
		// The caller expressed no preference, so the vendor's tuned operating
		// point is sent instead of nothing. Silent, and lossless relative to
		// what the caller said: omitting the parameter does NOT mean "the
		// model's default" on every endpoint — at least one falls back
		// materially below the value the model was tuned for, so an omission
		// would quietly run the model off its recommended point.
		if s.Supported && s.RecommendedValue != nil {
			*out = copyFloat(s.RecommendedValue)
		}
		return
	}
	if !s.Supported {
		if !v.refuse(name, fmt.Sprintf("%s is not supported by this model", name), nil) {
			*out = nil
		}
		return
	}
	if s.ForcedValue != nil && *want != *s.ForcedValue {
		if !v.refuse(name,
			fmt.Sprintf("%s must be %g on this model; any other value is rejected", name, *s.ForcedValue),
			[]string{fmt.Sprintf("%g", *s.ForcedValue)}) {
			// Demoted to the one value the endpoint takes, not dropped: the
			// caller asked for a specific parameter, and the forced value is the
			// nearest thing the model will accept.
			*out = copyFloat(s.ForcedValue)
		}
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

// checkMaxTokens warns when the cap exceeds what the model can produce.
//
// A warning, not a refusal: every endpoint measured clamps silently rather than
// erroring, so the request works and only the caller's expectation is wrong. The
// cap is still sent verbatim — rewriting it would hide the profile's limit behind
// a number the caller never chose.
func (v *validation) checkMaxTokens(req ChatRequest) {
	max := v.profile.Limits.MaxOutput
	if req.MaxTokens == nil || max <= 0 || int64(*req.MaxTokens) <= max {
		return
	}
	v.warn(WarnCompatibility, "max_tokens",
		fmt.Sprintf("%d exceeds this model's output limit of %d, which the endpoint clamps silently",
			*req.MaxTokens, max))
}

func (v *validation) checkTools(req ChatRequest) {
	if len(req.Tools) == 0 {
		if req.ToolChoice.Mode != ToolChoiceUnset {
			v.warn(WarnUnsupported, "tool_choice",
				"a tool choice was set but no tools were offered, so it was dropped")
			v.out.ToolChoice = ToolChoice{}
		}
		return
	}
	t := v.profile.Tools
	if !t.Supported {
		if v.refuse("tools", "this model does not support tool calling", nil) {
			return
		}
		// Demoted: the tools go with the choice. Sending a tool_choice for tools
		// the endpoint never received is a 400 on every implementation.
		v.out.Tools = nil
		v.out.ToolChoice = ToolChoice{}
		return
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
		if !v.refuse("tool_choice",
			fmt.Sprintf("this model accepts tool_choice %v and cannot be told %q; "+
				"omit Tools from the request to guarantee no tool call",
				t.ToolChoiceValues, ToolChoiceNone),
			t.ToolChoiceValues) {
			// Demoted: the field is dropped, not relaxed to "auto". The tools
			// stay offered, because withholding them changes what the model is
			// told exists — and that is the caller's decision, not this
			// package's, even under BestEffort.
			v.out.ToolChoice = ToolChoice{}
		}
		return
	}

	// Relaxing to "auto" is only a coercion if this model takes "auto". Every chat
	// profile in the registry does, but substituting one refused mode for another
	// would send the 400 this tier policy exists to catch locally — so the
	// substitute is checked against the same accepted set.
	if !toolChoiceAccepted(t.ToolChoiceValues, ToolChoiceAuto) {
		if !v.refuse("tool_choice",
			fmt.Sprintf("this model accepts tool_choice %v, and %q is not among them, so there is "+
				"nothing to relax %q to", t.ToolChoiceValues, ToolChoiceAuto, req.ToolChoice.Mode),
			t.ToolChoiceValues) {
			v.out.ToolChoice = ToolChoice{}
		}
		return
	}

	// Tier 3: the request still runs, just less constrained. Warned rather than
	// dropped silently because several of these endpoints DROP the field
	// themselves without saying so, which is how a caller comes to believe a
	// call was forced when it was not.
	v.warn(WarnCompatibility, "tool_choice",
		fmt.Sprintf("this model accepts tool_choice %v; %q was relaxed to %q",
			t.ToolChoiceValues, req.ToolChoice.Mode, ToolChoiceAuto))
	v.out.ToolChoice = ToolChoice{Mode: ToolChoiceAuto}
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
			if !v.refuse("response_format", "this model does not support JSON object output", nil) {
				v.out.ResponseFormat = ResponseFormat{}
			}
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
				if !v.refuse("response_format", "this model supports no structured output", nil) {
					v.out.ResponseFormat = ResponseFormat{}
				}
				return
			}
			// Approximate but honourable: the reply is still constrained to
			// JSON, just not to this shape. Exactly the case warnings exist for.
			v.warn(WarnCompatibility, "response_format",
				"this model does not support json_schema; downgraded to json_object, "+
					"so the schema is not enforced")
			// The downgrade is recorded as DATA, not only as prose in the
			// warning: the renderer emits whatever Kind says and never re-derives
			// the decision from the profile.
			v.out.ResponseFormat = ResponseFormat{Kind: FormatJSONObject}
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
			v.out.ResponseFormat.Strict = false
		}
	default:
		if !v.refuse("response_format",
			fmt.Sprintf("unknown response format %q", f.Kind),
			[]string{string(FormatText), string(FormatJSONObject), string(FormatJSONSchema)}) {
			// Nothing to coerce an unrecognised kind TO, so it is dropped
			// entirely rather than guessed at.
			v.out.ResponseFormat = ResponseFormat{}
		}
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
