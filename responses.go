package llmwire

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
	"unicode/utf8"
)

// The Responses protocol: POST /responses, for a profile with wire_api:
// responses. Same intents, same plan, same bounds, same pricing as Chat
// Completions; only the body, the reply and the stream are spelled
// differently. Every wire fact below was MEASURED on gpt-6.1-sol through
// LiteLLM 1.105.1 → Azure Switzerland North (`gpt-6.1-sol-2026-09-29`,
// upstream `/openai/v1/responses?api-version=preview`), 2026-10-19, unless it
// says otherwise.
//
// STATELESS, ALWAYS
//
//   - store:false and include:["reasoning.encrypted_content"] on every call.
//     The model's reasoning comes back as an encrypted item and goes back in
//     on the next round, so no turn depends on the server remembering one.
//     previous_response_id is never sent.
//   - Round two of a tool loop replaying the reasoning item, the
//     function_call and its function_call_output was accepted, input_tokens
//     97 → 177, and it returned a NEW reasoning item: a turn's items
//     accumulate, and every one goes back each round (Message.ResponseItems).
//   - Replayed trimmed to what the API takes as input: a reasoning item as
//     {type,id,summary,encrypted_content} (no `content`), a call as
//     {type,id,call_id,name,arguments} (no `status`, `namespace`). That is the
//     exact shape round two sent.
//
// TOOLS
//
//   - Function tools come back `"strict": true` although none was sent: the
//     protocol defaults them strict, which rejects any schema that is not
//     closed. So `strict` is always sent, false unless the caller set it.
//   - tool_choice auto, required, none and the forced {"type":"function",
//     "name":…} were each honoured; two tools in one turn gave two
//     function_call items.
//
// STREAMING
//
//   - Dispatched on the JSON `type` inside each `data:` frame. The gateway
//     sends no `event:` lines at all.
//   - Completion is asserted by response.completed (or response.incomplete,
//     a clean truncation), never inferred. LiteLLM appends `data: [DONE]`,
//     which the protocol does not define: tolerated, never relied on.
//   - Behind LiteLLM with include_cost_in_streaming_usage the stream is
//     BUFFERED: time to first token equalled the whole call (17.0s for 1310
//     tokens), response.created already carried the final usage, and
//     output_item.added carried the complete arguments and text before the
//     deltas replayed them in 5-character pieces. So a call's arguments are
//     taken from the deltas (or from .done when no delta came), NEVER also
//     from output_item.added, which would double them; and the header bound
//     on such a route has to cover the whole call.
//   - The assembled result is the response object on response.completed, the
//     authority; the deltas are what the iterator sees on the way.

// routeResponses is the Responses route, appended to the configured base URL.
const routeResponses = "/responses"

// responses says this plan goes out on the Responses protocol.
func (pl *wirePlan) responses() bool { return pl.profile.WireAPI == WireResponses }

// route is where this plan's request goes.
func (pl *wirePlan) route() string {
	if pl.responses() {
		return routeResponses
	}
	return routeChat
}

// render builds this plan's body in its protocol's spelling.
func (pl *wirePlan) render() ([]byte, error) {
	if pl.responses() {
		return renderResponsesBody(pl)
	}
	return renderChatBody(pl)
}

// parse decodes this plan's non-streaming reply.
func (pl *wirePlan) parse(redact redactor, raw json.RawMessage) (*ChatResponse, error) {
	if pl.responses() {
		return parseResponsesResponseWith(redact, raw)
	}
	return parseChatResponseWith(redact, raw)
}

// renderResponsesBody builds the body for a planned Responses request. The
// render.go invariant holds here too: every capability was decided in plan,
// and this reads wire names only.
func renderResponsesBody(pl *wirePlan) ([]byte, error) {
	req := pl.req
	body := map[string]any{
		"model": pl.profile.WireModelID,
		"input": renderInput(req.Messages),
		"store": false,
	}
	if pl.encryptedReasoning {
		body["include"] = []string{"reasoning.encrypted_content"}
	}
	if req.MaxTokens != nil {
		body[pl.capParam] = *req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	renderResponsesReasoning(body, req.Reasoning)
	if len(req.Tools) > 0 {
		body["tools"] = renderResponsesTools(req.Tools)
	}
	if tc := renderResponsesToolChoice(req.ToolChoice); tc != nil {
		body["tool_choice"] = tc
	}
	if f := renderTextFormat(req.ResponseFormat); f != nil {
		body["text"] = map[string]any{"format": f}
	}
	if pl.stream {
		body["stream"] = true
	}
	// Last, by replacement, as on Chat Completions; the keys that may not be
	// overridden were refused in plan.
	for k, v := range req.ExtraBody {
		body[k] = v
	}
	return json.Marshal(body)
}

// renderInput encodes the conversation as input items. An assistant turn that
// carries ResponseItems goes out as those items, verbatim and in their order:
// they are the same turn as Text and ToolCalls (plan checked the calls agree)
// plus the reasoning neither of those holds.
func renderInput(ms []Message) []any {
	out := make([]any, 0, len(ms))
	for _, m := range ms {
		switch {
		case len(m.ResponseItems) > 0:
			for _, item := range m.ResponseItems {
				out = append(out, item)
			}
		case m.Role == RoleTool:
			out = append(out, map[string]any{
				"type":    "function_call_output",
				"call_id": m.ToolCallID,
				"output":  m.Text,
			})
		default:
			if len(m.Parts) > 0 {
				out = append(out, map[string]any{"role": string(m.Role), "content": renderResponsesParts(m.Parts)})
			} else if m.Text != "" || len(m.ToolCalls) == 0 {
				out = append(out, map[string]any{"role": string(m.Role), "content": m.Text})
			}
			// A turn built by hand, or from a Chat Completions model: its
			// calls as items, with no reasoning to go before them and no fc_
			// item id. UNMEASURED, and likely REFUSED by a reasoning model
			// under store:false: OpenAI documents "Item 'fc_…' of type
			// 'function_call' was provided without its required 'reasoning'
			// item". Measured accepted is only the Assistant() turn.
			for _, tc := range m.ToolCalls {
				out = append(out, map[string]any{
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      tc.Name,
					"arguments": tc.Arguments,
				})
			}
		}
	}
	return out
}

// renderResponsesParts spells a multimodal turn. UNMEASURED for images: the
// protocol's input_image takes image_url as a bare STRING, where Chat
// Completions nests it in an object.
func renderResponsesParts(parts []Part) []any {
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		if p.Kind == PartImage {
			out = append(out, map[string]any{"type": "input_image", "image_url": p.URL})
			continue
		}
		out = append(out, map[string]any{"type": "input_text", "text": p.Text})
	}
	return out
}

// renderResponsesTools spells the tools flat: no nested "function" object.
func renderResponsesTools(ts []Tool) []any {
	out := make([]any, 0, len(ts))
	for _, t := range ts {
		tool := map[string]any{
			"type": "function",
			"name": t.Name,
			// Always sent: the protocol defaults a function tool to strict
			// (measured, see the head comment), which rejects an open schema.
			"strict": t.Strict,
		}
		if t.Description != "" {
			tool["description"] = t.Description
		}
		if len(t.Parameters) > 0 {
			tool["parameters"] = t.Parameters
		}
		out = append(out, tool)
	}
	return out
}

// renderResponsesToolChoice returns nil when there is nothing to send.
func renderResponsesToolChoice(tc ToolChoice) any {
	switch tc.Mode {
	case ToolChoiceUnset:
		return nil
	case ToolChoiceFunction:
		return map[string]any{"type": "function", "name": tc.Name}
	default:
		return string(tc.Mode)
	}
}

// renderTextFormat is response_format in this protocol's spelling, the schema
// flat beside its name. nil when there is nothing to send.
func renderTextFormat(f ResponseFormat) any {
	switch f.Kind {
	case FormatUnset:
		return nil
	case FormatJSONSchema:
		out := map[string]any{"type": string(FormatJSONSchema), "name": f.Name, "schema": f.Schema}
		if f.Strict {
			out["strict"] = true
		}
		return out
	default:
		return map[string]any{"type": string(f.Kind)}
	}
}

// renderResponsesReasoning writes reasoning.effort. The loader allows only the
// effort control on this protocol, so the knob is a level or the off switch.
func renderResponsesReasoning(body map[string]any, want ReasoningRequest) {
	switch w := want.(type) {
	case reasoningOff:
		body["reasoning"] = map[string]any{"effort": "none"}
	case reasoningEffort:
		body["reasoning"] = map[string]any{"effort": w.level}
	}
}

// wireResponse mirrors the response object, leniently, as wireChatResponse
// does: Azure adds content_filters, phase and more nobody has modelled.
type wireResponse struct {
	Model             string            `json:"model"`
	Status            string            `json:"status"`
	Output            []json.RawMessage `json:"output"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Usage json.RawMessage `json:"usage"`
	Error json.RawMessage `json:"error"`
}

// wireOutputItem is one entry of output: a reasoning item, a function call or
// an assistant message.
type wireOutputItem struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Role      string          `json:"role"`
	Content   []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Summary          json.RawMessage `json:"summary"`
	EncryptedContent string          `json:"encrypted_content"`
}

// turn is one assistant turn decoded from a response object.
type turn struct {
	content, reasoning, finish string
	calls                      []ToolCall
	items                      []json.RawMessage
	usage                      Usage
	model                      string
}

// parseResponsesResponseWith decodes a non-streamed response object.
func parseResponsesResponseWith(redact redactor, raw json.RawMessage) (*ChatResponse, error) {
	t, err := decodeResponse(redact, raw)
	if err != nil {
		return nil, err
	}
	return &ChatResponse{
		Content:       t.content,
		Reasoning:     t.reasoning,
		ToolCalls:     t.calls,
		FinishReason:  t.finish,
		Usage:         t.usage,
		Model:         t.model,
		Raw:           raw,
		ResponseItems: t.items,
	}, nil
}

// decodeResponse reads a response object, the body of a non-streamed call or
// the `response` of a stream's completed event.
func decodeResponse(redact redactor, raw json.RawMessage) (turn, error) {
	var w wireResponse
	if err := json.Unmarshal(raw, &w); err != nil {
		return turn{}, fmt.Errorf("llmwire: %w: decoding response: %w (body: %s)",
			ErrMalformedResponse, err, bodySnippet(redact, raw))
	}
	// A failure under a 200, the gateway's {"error":…} or a failed response.
	if len(w.Error) > 0 && !isJSONNull(w.Error) {
		return turn{}, parseAPIErrorWith(redact, 0, w.Error)
	}

	t := turn{model: w.Model, usage: parseUsage(w.Usage)}
	var content, reasoning strings.Builder
	for _, rawItem := range w.Output {
		var it wireOutputItem
		if err := json.Unmarshal(rawItem, &it); err != nil {
			continue
		}
		switch it.Type {
		case "reasoning":
			reasoning.WriteString(summaryText(it.Summary))
		case "function_call":
			t.calls = append(t.calls, ToolCall{ID: it.CallID, Type: "function", Name: it.Name,
				Arguments: argumentsText(it.Arguments)})
		case "message":
			for _, c := range it.Content {
				if c.Type == "output_text" {
					content.WriteString(c.Text)
				}
			}
		}
		if item, ok := replayItem(it); ok {
			t.items = append(t.items, item)
		}
	}
	t.content, t.reasoning = content.String(), reasoning.String()

	switch w.Status {
	case "completed":
		t.finish = "stop"
		if len(t.calls) > 0 {
			t.finish = "tool_calls"
		}
	case "incomplete":
		// Copied in Chat Completions' vocabulary where one exists, so a caller
		// checking for "length" need not know which protocol answered; any
		// other reason verbatim, as an open set.
		reason := "incomplete"
		if w.IncompleteDetails != nil && w.IncompleteDetails.Reason != "" {
			reason = w.IncompleteDetails.Reason
		}
		if reason == "max_output_tokens" {
			reason = "length"
		}
		t.finish = reason
	default:
		// failed without an error object, queued, in_progress: nothing a
		// stateless foreground call should ever see, and never an answer.
		return turn{}, fmt.Errorf("llmwire: %w: response status %q (body: %s)",
			ErrResponseShape, w.Status, bodySnippet(redact, raw))
	}
	// A reasoning item alone is no answer: an empty success in all but name.
	if len(t.calls) == 0 && t.content == "" && t.finish == "stop" {
		return turn{}, fmt.Errorf("llmwire: %w: response carried no answer and no error (body: %s)",
			ErrResponseShape, bodySnippet(redact, raw))
	}
	return t, nil
}

// summaryText joins a reasoning item's readable summary, which is empty unless
// a summary was asked for (none is).
func summaryText(raw json.RawMessage) string {
	var parts []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		b.WriteString(p.Text)
	}
	return b.String()
}

// replayItem is an output item in the shape the next request's input takes,
// or false for one that cannot go back: a reasoning item without its
// encrypted content would need the server-side state store:false forgoes, and
// an item type this package does not model has no known input spelling.
func replayItem(it wireOutputItem) (json.RawMessage, bool) {
	var v any
	switch it.Type {
	case "reasoning":
		if it.EncryptedContent == "" {
			return nil, false
		}
		summary := it.Summary
		if len(summary) == 0 || isJSONNull(summary) {
			summary = json.RawMessage("[]")
		}
		v = map[string]any{"type": "reasoning", "id": it.ID, "summary": summary,
			"encrypted_content": it.EncryptedContent}
	case "function_call":
		v = map[string]any{"type": "function_call", "id": it.ID, "call_id": it.CallID,
			"name": it.Name, "arguments": argumentsText(it.Arguments)}
	case "message":
		// As the plain role form a hand-built turn uses; the output item's
		// own shape (phase, annotations) is not measured as input.
		var b strings.Builder
		for _, c := range it.Content {
			if c.Type == "output_text" {
				b.WriteString(c.Text)
			}
		}
		v = map[string]any{"role": string(RoleAssistant), "content": b.String()}
	default:
		return nil, false
	}
	raw, err := json.Marshal(v)
	return raw, err == nil
}

// responseItemCallIDs lists the call_ids of the function calls among items,
// in order, for plan's agreement check.
func responseItemCallIDs(items []json.RawMessage) ([]string, error) {
	ids := []string{}
	for i, raw := range items {
		var it struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
		}
		if err := json.Unmarshal(raw, &it); err != nil {
			return nil, fmt.Errorf("ResponseItems[%d] is not a JSON object: %w", i, err)
		}
		if it.Type == "function_call" {
			ids = append(ids, it.CallID)
		}
	}
	return ids, nil
}

// responsesEvent is one `data:` frame of a Responses stream. Only what the
// reader dispatches on; the rest stays in the raw frame.
type responsesEvent struct {
	Type        string          `json:"type"`
	OutputIndex int             `json:"output_index"`
	Delta       string          `json:"delta"`
	Arguments   json.RawMessage `json:"arguments"`
	Item        wireOutputItem  `json:"item"`
	Response    json.RawMessage `json:"response"`
	// Error is the gateway's {"error":…} frame inside a 200.
	Error json.RawMessage `json:"error"`
}

// readResponsesStream consumes a Responses SSE body. The transport rules are
// readStream's, through the same frameMeter: the idle guard re-arms on `data:`
// frames only, Timing shares the caller's clock, and Total is written on every
// return.
func readResponsesStream(body io.Reader, guard *stallGuard, bounds streamBounds, sink func(streamEvent), redact redactor, now func() time.Time, start time.Time) (res StreamResult, err error) {
	defer func() { res.Timing.Total = now().Sub(start) }()
	idle := bounds.idle
	emit := func(ev streamEvent) {
		if sink != nil {
			sink(ev)
		}
	}
	var (
		content, reasoning strings.Builder
		// calls maps output_index to the call's ordinal among function
		// calls, which is the Index a consumer sees.
		calls     = map[int]int{}
		gotDeltas = map[int]bool{}
		// partial is the calls as the deltas built them, returned only when
		// the stream ends without its completed event.
		partial   []ToolCall
		completed bool
		frameErr  error
	)

	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64<<10), maxStreamLine)
	meter := newFrameMeter(now, start)
	for sc.Scan() {
		res.Bytes += int64(len(sc.Bytes()))
		line := strings.TrimRight(sc.Text(), "\r")
		if !strings.HasPrefix(line, dataPrefix) {
			continue
		}
		meter.frame(&res)
		guard.arm(idle, stallIdle)

		payload := strings.TrimSpace(strings.TrimPrefix(line, dataPrefix))
		if payload == doneMarker {
			// Not part of the protocol; the gateway's. Completion was
			// already asserted or it was not.
			res.Done = true
			break
		}
		var ev responsesEvent
		if json.Unmarshal([]byte(payload), &ev) != nil {
			continue
		}

		switch ev.Type {
		case "response.output_text.delta":
			if ev.Delta != "" {
				content.WriteString(ev.Delta)
				res.Chars += int64(utf8.RuneCountInString(ev.Delta))
				emit(streamEvent{kind: evContent, text: ev.Delta})
			}
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			if ev.Delta != "" {
				reasoning.WriteString(ev.Delta)
				emit(streamEvent{kind: evReasoning, text: ev.Delta})
			}
		case "response.output_item.added":
			if ev.Item.Type != "function_call" {
				continue
			}
			// The name and id only. A buffering gateway puts the complete
			// arguments here too, and the deltas then repeat them.
			if bounds.toolIdle > 0 && idle != bounds.toolIdle {
				idle = bounds.toolIdle
				guard.arm(idle, stallIdle)
			}
			ord := len(calls)
			calls[ev.OutputIndex] = ord
			partial = append(partial, ToolCall{ID: ev.Item.CallID, Type: "function", Name: ev.Item.Name})
			var tc toolCallDelta
			tc.Index, tc.ID, tc.Type, tc.Function.Name = ord, ev.Item.CallID, "function", ev.Item.Name
			emit(streamEvent{kind: evToolCall, toolCall: tc})
		case "response.function_call_arguments.delta":
			ord, ok := calls[ev.OutputIndex]
			if !ok || ev.Delta == "" {
				continue
			}
			gotDeltas[ev.OutputIndex] = true
			partial[ord].Arguments += ev.Delta
			emit(streamEvent{kind: evToolCall, toolCall: argsDelta(ord, ev.Delta)})
		case "response.function_call_arguments.done":
			// The whole arguments in one piece, for a stream that sent no
			// deltas; after deltas it would be the second copy.
			ord, ok := calls[ev.OutputIndex]
			if !ok || gotDeltas[ev.OutputIndex] {
				continue
			}
			if args := argumentsText(ev.Arguments); args != "" {
				partial[ord].Arguments = args
				emit(streamEvent{kind: evToolCall, toolCall: argsDelta(ord, args)})
			}
		case "response.completed", "response.incomplete":
			t, derr := decodeResponse(redact, ev.Response)
			if derr != nil {
				frameErr = derr
				break
			}
			completed = true
			res.Content, res.Reasoning, res.ToolCalls = t.content, t.reasoning, t.calls
			res.ResponseItems, res.FinishReason, res.Usage = t.items, t.finish, t.usage
			emit(streamEvent{kind: evFinish, finishReason: t.finish})
		case "response.failed":
			var r wireResponse
			if json.Unmarshal(ev.Response, &r) == nil && len(r.Error) > 0 && !isJSONNull(r.Error) {
				frameErr = parseAPIErrorWith(redact, 0, r.Error)
			} else {
				frameErr = fmt.Errorf("llmwire: %w: response.failed with no error object", ErrResponseShape)
			}
		case "error":
			frameErr = parseAPIErrorWith(redact, 0, []byte(payload))
		case "":
			if len(ev.Error) > 0 && !isJSONNull(ev.Error) {
				frameErr = parseAPIErrorWith(redact, 0, ev.Error)
			}
		}
		if frameErr != nil || completed {
			break
		}
	}

	if !completed {
		// What arrived before the failure, as readStream returns it: the
		// paid-for partial rides beside the error.
		res.Content, res.Reasoning, res.ToolCalls = content.String(), reasoning.String(), partial
	}
	switch {
	case frameErr != nil:
		return res, frameErr
	case completed:
		return res, nil
	case errors.Is(sc.Err(), context.Canceled), errors.Is(sc.Err(), context.DeadlineExceeded):
		return res, fmt.Errorf("llmwire: reading stream: %w", sc.Err())
	case sc.Err() != nil:
		return res, fmt.Errorf("llmwire: %w: reading stream: %w", ErrMalformedResponse, sc.Err())
	default:
		return res, fmt.Errorf("llmwire: %w: stream ended after %d events (%d chars) without response.completed",
			ErrMalformedResponse, res.Events, res.Chars)
	}
}

// argsDelta is an argument fragment for call ord, encoded as the JSON string
// a Chat Completions fragment carries so consumers read both alike.
func argsDelta(ord int, text string) toolCallDelta {
	var tc toolCallDelta
	tc.Index = ord
	tc.Function.Arguments, _ = json.Marshal(text)
	return tc
}
