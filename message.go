package llmwire

import "encoding/json"

// The request vocabulary.
//
// These types are what every consumer binds to, so their shape was settled by
// measurement rather than guessed: Phase 0 probed whether reasoning has to be
// replayed through tool history before deciding whether Message carries it.
// See FINDINGS.md.

// Role is a message author.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is one turn.
//
// Content is either plain text or a sequence of parts. Both spellings exist on
// the wire and the difference is not cosmetic: a text-only model rejects the
// parts form outright once it contains an image, so the encoder emits whichever
// the message actually needs rather than always using the richer one.
type Message struct {
	Role Role
	// Text is the whole content, for the common single-string case.
	Text string
	// Parts, when non-empty, replaces Text. Used for multimodal turns.
	Parts []Part

	// ToolCalls are the calls an assistant turn requested.
	ToolCalls []ToolCall
	// ToolCallID ties a tool result back to the call that asked for it.
	// Required on a RoleTool message.
	ToolCallID string

	// ReasoningContent is the model's thinking from a previous assistant turn.
	//
	// OPTIONAL, and measured to be so. A vendor issue reports a 400 with "The
	// reasoning_content in the thinking mode must be passed back to the API"
	// when an assistant turn carrying tool_calls is replayed without it. That
	// does not reproduce on the deployments in the registry — a replay omitting
	// it was accepted — and a production caller has been omitting it for
	// months. The field exists so a deployment that does demand it can be fed,
	// not because one currently does.
	ReasoningContent string
}

// PartKind distinguishes the content parts a message can carry.
type PartKind string

const (
	PartText  PartKind = "text"
	PartImage PartKind = "image_url"
)

// Part is one element of a multimodal message.
type Part struct {
	Kind PartKind
	// Text is set when Kind is PartText.
	Text string
	// URL is set when Kind is PartImage. A data: URI is accepted and is how an
	// inline image is sent.
	URL string
}

// TextMessage builds a plain-text turn.
func TextMessage(role Role, text string) Message {
	return Message{Role: role, Text: text}
}

// System, User and Assistant are shorthands for the three ordinary roles.
func System(text string) Message    { return TextMessage(RoleSystem, text) }
func User(text string) Message      { return TextMessage(RoleUser, text) }
func Assistant(text string) Message { return TextMessage(RoleAssistant, text) }

// ToolResult builds the reply to a tool call.
func ToolResult(callID, content string) Message {
	return Message{Role: RoleTool, ToolCallID: callID, Text: content}
}

// HasImage reports whether the message carries an image part, which is what
// validation checks against a model's vision capability.
func (m Message) HasImage() bool {
	for _, p := range m.Parts {
		if p.Kind == PartImage {
			return true
		}
	}
	return false
}

// Tool is a function offered to the model.
type Tool struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object. Held as a decoded value rather than a
	// string so a caller can build it however it likes.
	Parameters map[string]any
	// Strict asks for strict schema adherence. Profile-gated: some compat
	// servers reject the field outright.
	Strict bool
}

// ToolChoiceMode is how the model should decide about calling a tool.
type ToolChoiceMode string

const (
	// ToolChoiceUnset leaves the field off the wire entirely.
	ToolChoiceUnset ToolChoiceMode = ""
	ToolChoiceAuto  ToolChoiceMode = "auto"
	ToolChoiceNone  ToolChoiceMode = "none"
	// ToolChoiceRequired compels some tool call.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceFunction compels one named tool; set ToolChoice.Name too.
	ToolChoiceFunction ToolChoiceMode = "function"
)

// ToolChoice constrains tool selection.
type ToolChoice struct {
	Mode ToolChoiceMode
	// Name is the tool to compel, when Mode is ToolChoiceFunction.
	Name string
}

// ResponseFormatKind is the structured-output mode.
type ResponseFormatKind string

const (
	FormatUnset ResponseFormatKind = ""
	FormatText  ResponseFormatKind = "text"
	// FormatJSONObject asks for a single JSON object, with no schema.
	FormatJSONObject ResponseFormatKind = "json_object"
	// FormatJSONSchema asks for output conforming to a named schema.
	FormatJSONSchema ResponseFormatKind = "json_schema"
)

// ResponseFormat constrains the shape of the reply.
//
// Worth reaching for rather than trusting a prompt: on one model in the registry
// a prompt saying "reply as JSON" produced a raw reply that parsed 0 times out
// of 8, while the same prompt with a response format parsed 8 out of 8.
type ResponseFormat struct {
	Kind ResponseFormatKind
	// Name identifies the schema. Required by FormatJSONSchema.
	Name string
	// Schema is a JSON Schema object, for FormatJSONSchema.
	Schema map[string]any
	// Strict asks for exact schema adherence.
	Strict bool
}

// Reasoning is how much thinking to ask for.
//
// A sum type rather than a string, so validation is "does this variant match one
// the profile allows" rather than a string comparison against a set the caller
// had to know. The three variants mirror the three controls models actually
// expose; a request variant that does not match the profile's control is
// refused by name rather than sent and rejected.
type ReasoningRequest interface{ isReasoning() }

type reasoningOff struct{}
type reasoningEffort struct{ level string }
type reasoningBudget struct{ tokens int }

func (reasoningOff) isReasoning()    {}
func (reasoningEffort) isReasoning() {}
func (reasoningBudget) isReasoning() {}

// ReasoningOff asks the model not to reason.
//
// Refused before any request is sent when the profile says the model cannot
// comply — which is the case for at least one model in the registry, whose
// vendor documentation claims otherwise.
func ReasoningOff() ReasoningRequest { return reasoningOff{} }

// ReasoningEffort asks for a named depth. The accepted levels are per-model and
// narrower than the vendor's global enum.
func ReasoningEffort(level string) ReasoningRequest { return reasoningEffort{level: level} }

// ReasoningBudget asks for a token budget.
func ReasoningBudget(tokens int) ReasoningRequest { return reasoningBudget{tokens: tokens} }

// ChatRequest is one completion request, before profile resolution.
//
// Pointers where "unset" and "zero" differ: a temperature of 0 is a real
// request, and so is a MaxTokens the caller deliberately left alone.
type ChatRequest struct {
	// Model is a registry id, matched exactly.
	Model    string
	Messages []Message

	Reasoning   ReasoningRequest
	Temperature *float64
	TopP        *float64
	// MaxTokens caps the completion. Written once here; the profile decides
	// which wire parameter carries it, because the two spellings are not
	// interchangeable and one endpoint accepts the wrong one and ignores it.
	MaxTokens *int

	Tools      []Tool
	ToolChoice ToolChoice

	ResponseFormat ResponseFormat

	// Stop sequences, when the endpoint supports them.
	Stop []string

	// ExtraBody is merged into the request body verbatim, for the long tail of
	// provider-specific parameters this package does not model. Keys collide
	// with generated ones at the caller's risk: they win, deliberately, since
	// the point is to reach something the package does not know about.
	ExtraBody map[string]any

	// BestEffort demotes what would be a hard error into a coercion plus a
	// Warning, for one call. Deliberately per-request rather than a package
	// global: a process-wide switch is a landmine, because the call that
	// silently changes behaviour is never the one that set it.
	BestEffort bool
}

// clone copies a request deeply enough that coercing the copy cannot touch the
// caller's own slices and maps.
//
// Validation coerces: it drops an image part, relaxes a tool choice, fills in a
// recommended temperature. Doing that in place would mean a call documented as
// read-only quietly editing the caller's request — and with Validate exported,
// checking a request at boot would mutate the very thing being checked.
//
// Tool.Parameters and the values inside ExtraBody are NOT deep-copied: nothing in
// this package writes through them, and copying an arbitrary caller-supplied
// JSON tree would be both expensive and lossy.
func (r ChatRequest) clone() ChatRequest {
	out := r
	if r.Messages != nil {
		out.Messages = make([]Message, len(r.Messages))
		copy(out.Messages, r.Messages)
		for i := range out.Messages {
			out.Messages[i].Parts = append([]Part(nil), r.Messages[i].Parts...)
			out.Messages[i].ToolCalls = append([]ToolCall(nil), r.Messages[i].ToolCalls...)
		}
	}
	out.Tools = append([]Tool(nil), r.Tools...)
	out.Stop = append([]string(nil), r.Stop...)
	out.Temperature = copyFloat(r.Temperature)
	out.TopP = copyFloat(r.TopP)
	if r.MaxTokens != nil {
		v := *r.MaxTokens
		out.MaxTokens = &v
	}
	if r.ExtraBody != nil {
		out.ExtraBody = make(map[string]any, len(r.ExtraBody))
		for k, v := range r.ExtraBody {
			out.ExtraBody[k] = v
		}
	}
	return out
}

// EmbedRequest is one embeddings request.
type EmbedRequest struct {
	Model  string
	Inputs []string
	// Dimensions shortens the returned vectors. Only the -3 generation accepts
	// it; older models reject it outright.
	Dimensions *int

	// BestEffort demotes a capability refusal into a coercion plus a Warning, as
	// on a chat request. It cannot rescue a malformed request: an empty input list
	// or an empty string has nothing to send instead.
	BestEffort bool
}

// EmbedResponse is a batch of vectors, in the order of the inputs that produced
// them.
type EmbedResponse struct {
	Vectors [][]float32
	Usage   Usage
	Model   string
}

// ChatResponse is a completed non-streaming turn.
type ChatResponse struct {
	Content   string
	Reasoning string
	ToolCalls []ToolCall
	// FinishReason is the endpoint's own string, treated as an open set.
	FinishReason string
	Usage        Usage
	// Model is what actually ran, which behind a gateway is not what was asked
	// for: the response body carries the alias, and the real deployment is only
	// in a header.
	Model string
	// Raw is the undecoded response, kept for fields this package does not model.
	Raw json.RawMessage
}
