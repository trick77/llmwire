package llmwire

import (
	"encoding/json"
	"fmt"
)

// Wire rendering: a coerced request becomes the JSON body.
//
// THE INVARIANT, and a reviewer can check it by grepping this file for
// ".Supported": nothing here reads a CAPABILITY field off the profile. Every
// capability decision was made in plan and is already applied to wirePlan.req.
// What this file reads from the profile is wire NAMES only — which model id goes
// on the wire, which output-cap parameter this endpoint honours, which spelling
// the reasoning knob takes. Deciding a capability here as well would put one
// policy in two places, and the two would eventually disagree.
//
// The body is built as a map rather than marshalled from a tagged struct. Three
// reasons, in order of weight: ExtraBody keys must be able to overwrite generated
// ones, which a struct can only do by marshalling and merging afterwards anyway;
// a struct would need a pointer per field to tell "unset" from a real zero
// (temperature: 0 is a legitimate request); and encoding/json sorts map keys, so
// the output is deterministic for free.

// renderChatBody builds the body for a planned chat request.
func renderChatBody(pl *wirePlan) ([]byte, error) {
	req := pl.req
	body := map[string]any{
		// Never req.Model: behind a gateway the registry id and the wire id are
		// different strings, and the proxy only answers to its own alias.
		"model":    pl.profile.WireModelID,
		"messages": renderMessages(req.Messages),
	}

	// The single most expensive line in this package. One endpoint accepts
	// max_completion_tokens and silently IGNORES it — a cap of 16 returned 481
	// completion tokens with finish_reason "stop" — so the name comes from the
	// profile and exactly one of the two spellings is ever emitted.
	if req.MaxTokens != nil {
		body[pl.capParam] = *req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		body["stop"] = req.Stop
	}
	if err := renderReasoning(body, req.Reasoning, pl.profile.Reasoning); err != nil {
		return nil, err
	}
	if len(req.Tools) > 0 {
		body["tools"] = renderTools(req.Tools)
	}
	if tc := renderToolChoice(req.ToolChoice); tc != nil {
		body["tool_choice"] = tc
	}
	if rf := renderResponseFormat(req.ResponseFormat); rf != nil {
		body["response_format"] = rf
	}
	if pl.stream {
		// RawStream does not inject this: the body it is handed must already say
		// so, because the probe suite needs to send a non-streaming body down the
		// same path.
		body["stream"] = true
		if pl.includeUsage {
			body["stream_options"] = map[string]any{"include_usage": true}
		}
	}

	// Last, and by replacement. The whole point of ExtraBody is reaching a
	// parameter this package does not model, so a caller key that collides with a
	// generated one wins — deliberately, and documented on the field. Top-level
	// only: a deep merge would leave a half-generated nested object that neither
	// side asked for. The four keys that may not be overridden (stream,
	// stream_options, model, the other cap spelling) were refused in plan, so
	// nothing here has to check.
	for k, v := range req.ExtraBody {
		body[k] = v
	}
	return json.Marshal(body)
}

// renderMessages encodes the conversation.
func renderMessages(ms []Message) []any {
	out := make([]any, 0, len(ms))
	for _, m := range ms {
		msg := map[string]any{"role": string(m.Role)}
		if len(m.Parts) > 0 {
			msg["content"] = renderParts(m.Parts)
		} else {
			// The plain string form, not a one-element parts array. A text-only
			// model rejects the parts form outright once it contains an image,
			// and several compat servers are stricter about the richer shape in
			// general, so the encoder sends whichever the message actually needs.
			msg["content"] = m.Text
		}
		if len(m.ToolCalls) > 0 {
			msg["tool_calls"] = renderToolCalls(m.ToolCalls)
			if len(m.Parts) == 0 && m.Text == "" {
				// UNMEASURED: whether an assistant turn carrying only tool calls
				// should send "", null, or omit content entirely. The empty
				// string is the widest-accepted spelling of the three and is what
				// the OpenAI reference shows; the alternatives stay a probe
				// candidate rather than a guess dressed up as a decision.
				msg["content"] = ""
			}
		}
		if m.ToolCallID != "" {
			msg["tool_call_id"] = m.ToolCallID
		}
		if m.ReasoningContent != "" {
			// Only when non-empty. MEASURED: MiMo does not require reasoning to
			// be replayed alongside tool_calls history, contra a vendor issue
			// reporting a 400, so an empty field is never sent to find out.
			msg["reasoning_content"] = m.ReasoningContent
		}
		out = append(out, msg)
	}
	return out
}

func renderParts(parts []Part) []any {
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		switch p.Kind {
		case PartImage:
			// image_url is a nested OBJECT, not a bare string. This is the exact
			// shape that returns 404 "No endpoints found that support image
			// input" on mimo-v2.5-pro and is accepted by its sibling.
			out = append(out, map[string]any{
				"type":      string(PartImage),
				"image_url": map[string]any{"url": p.URL},
			})
		default:
			// PartText. Any other kind was refused in plan, so this arm is
			// text only.
			out = append(out, map[string]any{"type": string(PartText), "text": p.Text})
		}
	}
	return out
}

func renderToolCalls(calls []ToolCall) []any {
	out := make([]any, 0, len(calls))
	for _, tc := range calls {
		typ := tc.Type
		if typ == "" {
			typ = "function"
		}
		out = append(out, map[string]any{
			"id":   tc.ID,
			"type": typ,
			"function": map[string]any{
				"name": tc.Name,
				// Arguments stay a STRING. The wire carries a JSON document
				// encoded as text; re-encoding it as an object here would change
				// the shape every server parses.
				"arguments": tc.Arguments,
			},
		})
	}
	return out
}

func renderTools(ts []Tool) []any {
	out := make([]any, 0, len(ts))
	for _, t := range ts {
		fn := map[string]any{"name": t.Name}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		if len(t.Parameters) > 0 {
			fn["parameters"] = t.Parameters
		}
		if t.Strict {
			// Passed through unjudged. Output.StrictSchema describes the
			// response_format field, which is a different property; no profile
			// measures tool-level strictness, so gating this on an unrelated flag
			// would be a guess.
			fn["strict"] = true
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

// renderToolChoice returns nil when there is nothing to send. The mode arriving
// here has already been relaxed to something the model accepts.
func renderToolChoice(tc ToolChoice) any {
	switch tc.Mode {
	case ToolChoiceUnset:
		return nil
	case ToolChoiceFunction:
		return map[string]any{"type": "function", "function": map[string]any{"name": tc.Name}}
	default:
		return string(tc.Mode)
	}
}

// renderResponseFormat returns nil when there is nothing to send. The kind
// arriving here has already been downgraded if the model needed it.
func renderResponseFormat(f ResponseFormat) any {
	switch f.Kind {
	case FormatUnset:
		return nil
	case FormatJSONSchema:
		schema := map[string]any{"name": f.Name, "schema": f.Schema}
		if f.Strict {
			schema["strict"] = true
		}
		return map[string]any{"type": string(FormatJSONSchema), "json_schema": schema}
	default:
		return map[string]any{"type": string(f.Kind)}
	}
}

// renderReasoning writes the thinking knob in the spelling this model takes.
//
// A nil request writes NOTHING: the caller said nothing, so the model runs at
// its own default, on or off. Restating a default would add a key for no
// effect — and on the model that cannot be switched off, sending the knob at
// all is how you learn its error code is overloaded.
func renderReasoning(body map[string]any, want ReasoningRequest, r Reasoning) error {
	if want == nil {
		return nil
	}
	switch w := want.(type) {
	case reasoningOff:
		// Reached only where the profile says the model can be switched off:
		// plan refuses or drops it otherwise.
		switch r.Control {
		case ControlToggleObject:
			body["thinking"] = map[string]any{"type": "disabled"}
		case ControlEffort:
			body["reasoning_effort"] = "none"
		case ControlBudget:
			body[r.BudgetParam] = 0
		}
	case reasoningEffort:
		body["reasoning_effort"] = w.level
	case reasoningBudget:
		if r.BudgetParam == "" {
			// Unreachable through plan, which refuses a budget on any other
			// control, and the profile loader requires the name for this one. An
			// error rather than a silent omission: a dropped budget is a 10x cost
			// surprise, which is exactly the failure this package exists to catch.
			return fmt.Errorf("llmwire: a token budget was requested but the profile names no " +
				"parameter to send it under")
		}
		body[r.BudgetParam] = w.tokens
	}
	return nil
}

// renderEmbedBody builds one embeddings request. Inputs is a single batch, not
// the caller's whole list.
func renderEmbedBody(p *Profile, inputs []string, dimensions *int) ([]byte, error) {
	body := map[string]any{
		"model": p.WireModelID,
		"input": inputs,
	}
	if dimensions != nil {
		body["dimensions"] = *dimensions
	}
	return json.Marshal(body)
}
