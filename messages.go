package llmwire

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"time"
	"unicode/utf8"
)

// The Messages protocol: Anthropic's POST /messages, for a profile with
// wire_api: messages. Same intents, same plan, same bounds, same pricing as the
// OpenAI protocols; the caller's ChatRequest, ChatResponse and Stream do not
// change. Source: the API reference and the claude-api skill reference
// (anthropic-version 2026-10-21, claude-opus-5-5), read 2026-10-26. Every wire
// fact below is NOT measured unless it says MEASURED.
//
// THE BODY
//
//   - max_tokens is REQUIRED. plan sends limits.max_output when the caller set
//     no cap, which is what "no cap" means on the other protocols.
//   - System turns are one top-level `system` string. A system message after
//     the first turn is hoisted there too, in order: the protocol's own
//     mid-conversation system role has placement rules (after a user turn, last
//     or before an assistant turn) that a converted OpenAI history breaks.
//   - A tool result is a tool_result block in a USER turn, and consecutive
//     user turns are merged into one, so a round of parallel tool results
//     goes back as one user turn, as the protocol asks. Assistant turns are
//     never merged here; the endpoint joins consecutive ones itself
//     (MEASURED), so plan refuses a replayed or calling turn placed between
//     a call and its results, where joining would edit a signed turn.
//   - Thinking is sent as {"type":"adaptive","display":"summarized"} on a
//     model that reasons. The display is visibility only, billed the same; the
//     default ("omitted") streams thinking blocks with no text, so a long think
//     would be nothing but pings, which never re-arm the idle guard (below),
//     and Reasoning would always be empty. Depth is output_config.effort.
//
// REPLAY
//
//   - The assistant turn goes back as its content blocks, verbatim: thinking
//     with its signature, text, tool_use. A thinking block is signed and must
//     go back unchanged; Opus 5.5 runs "preserved thinking", which invalidates
//     the blocks of a history that was edited. So a tool loop appends
//     resp.Assistant(), never a turn rebuilt from Content and ToolCalls, the
//     same rule as on Responses. ResponseItems carries the blocks.
//
// STREAMING
//
//   - Dispatched on the JSON `type` inside each `data:` frame; `event:` lines
//     repeat it and are ignored.
//   - `ping` arrives as a DATA frame ({"type":"ping"}). It never re-arms the
//     idle guard and is not counted as an event: a heartbeat must not mask a
//     stalled model (AGENTS.md).
//   - A streamed block of a type not spelled here (server_tool_use and its
//     result, reachable only through tools in ExtraBody) is its start frame
//     alone, so the whole turn is left unreplayable: it goes back as Text
//     and ToolCalls, its thinking with it. A non-streamed reply keeps such
//     blocks verbatim. Server tools are unmodelled, on purpose: a pause_turn
//     resume ends on the paused assistant turn, which plan refuses as a
//     prefill on a model without assistant_prefill.
//   - Completion is asserted by message_stop, never inferred. A tool call's
//     input comes from its input_json_delta fragments only; the input on
//     content_block_start is the empty placeholder and is never also read.
//   - Usage arrives twice: message_start carries the input side, message_delta
//     the CUMULATIVE totals. Fields of the later object replace the earlier;
//     nothing is summed.

// routeMessages is the Messages route, appended to the configured base URL.
const routeMessages = "/messages"

// messages says this plan goes out on the Messages protocol.
func (pl *wirePlan) messages() bool { return pl.profile.WireAPI == WireMessages }

// renderMessagesBody builds the body for a planned Messages request. The
// render.go invariant holds here too: every capability was decided in plan.
func renderMessagesBody(pl *wirePlan) ([]byte, error) {
	req := pl.req
	turns, err := renderMessagesTurns(req.Messages)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"model":    pl.profile.WireModelID,
		"messages": turns,
	}
	if req.MaxTokens != nil {
		body[pl.capParam] = *req.MaxTokens
	}
	if s := systemText(req.Messages); s != "" {
		body["system"] = s
	}
	if pl.summarizedThinking {
		body["thinking"] = map[string]any{"type": "adaptive", "display": "summarized"}
	}
	outputConfig := map[string]any{}
	if w, ok := req.Reasoning.(reasoningEffort); ok {
		outputConfig["effort"] = w.level
	}
	if req.ResponseFormat.Kind == FormatJSONSchema {
		// Constrained decoding: the schema is enforced whatever Strict says,
		// so the flag has no spelling and none is sent.
		outputConfig["format"] = map[string]any{"type": "json_schema", "schema": req.ResponseFormat.Schema}
	}
	if len(outputConfig) > 0 {
		body["output_config"] = outputConfig
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		body["stop_sequences"] = req.Stop
	}
	if len(req.Tools) > 0 {
		body["tools"] = renderMessagesTools(req.Tools)
	}
	if req.ToolChoice.Mode != ToolChoiceUnset {
		// plan admits auto and none only on this protocol.
		body["tool_choice"] = map[string]any{"type": string(req.ToolChoice.Mode)}
	}
	if pl.stream {
		body["stream"] = true
	}
	for k, v := range req.ExtraBody {
		body[k] = v
	}
	return json.Marshal(body)
}

// systemText is every system turn's text, in order, as the one top-level
// system string.
func systemText(ms []Message) string {
	var parts []string
	for _, m := range ms {
		if m.Role == RoleSystem {
			if t := toolOutputText(m); hasText(t) {
				parts = append(parts, t)
			}
		}
	}
	return strings.Join(parts, "\n\n")
}

// messagesTurn is one turn on the wire, before merging.
type messagesTurn struct {
	Role    string `json:"role"`
	Content []any  `json:"content"`
}

// renderMessagesTurns encodes the conversation, system turns excluded. Only
// consecutive USER turns are merged: a round of tool results goes back as one
// user turn, as the protocol asks, the results first (plan has refused
// anything between a call and its results). Assistant turns are never merged,
// so a replayed turn's signed blocks go back exactly as they came. plan
// refuses every turn turnBlocks leaves empty (checkMessagesTurns); one
// reaching here is plan's bug and an error, never an empty content array on
// the wire.
func renderMessagesTurns(ms []Message) ([]messagesTurn, error) {
	var out []messagesTurn
	for i, m := range ms {
		if m.Role == RoleSystem {
			continue
		}
		role, blocks := turnBlocks(m)
		if len(blocks) == 0 {
			return nil, fmt.Errorf("llmwire: message %d has nothing to send and plan did not refuse it", i)
		}
		if n := len(out); n > 0 && role == RoleUser && out[n-1].Role == string(RoleUser) {
			out[n-1].Content = append(out[n-1].Content, blocks...)
			continue
		}
		out = append(out, messagesTurn{Role: string(role), Content: blocks})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("llmwire: the Messages protocol needs a user or assistant turn with content; the request carries none")
	}
	return out, nil
}

// turnBlocks is one non-system message as the role it goes out under and its
// content blocks. The one rule for what a turn sends: plan's emptiness check
// and render both read it.
func turnBlocks(m Message) (Role, []any) {
	switch m.Role {
	case RoleTool:
		return RoleUser, []any{toolResultBlock(m)}
	case RoleAssistant:
		if len(m.ResponseItems) > 0 {
			blocks := make([]any, len(m.ResponseItems))
			for i, item := range m.ResponseItems {
				blocks[i] = item
			}
			return RoleAssistant, blocks
		}
		var blocks []any
		if t := toolOutputText(m); hasText(t) {
			blocks = append(blocks, map[string]any{"type": "text", "text": t})
		}
		for _, tc := range m.ToolCalls {
			blocks = append(blocks, map[string]any{"type": "tool_use", "id": tc.ID, "name": tc.Name,
				"input": toolInput(tc.Arguments)})
		}
		return RoleAssistant, blocks
	default:
		return m.Role, contentBlocks(m)
	}
}

// hasText says a text block would be accepted. MEASURED: whitespace alone is
// a 400, "text content blocks must contain non-whitespace text".
func hasText(s string) bool { return strings.TrimSpace(s) != "" }

// toolInput is a call's arguments as the object tool_use.input takes. plan
// has checked they are empty or a JSON object.
func toolInput(args string) json.RawMessage {
	if strings.TrimSpace(args) == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(args)
}

// toolResultBlock is a tool turn as a tool_result block: its text, or its
// parts when it carries an image, which this protocol takes there.
func toolResultBlock(m Message) map[string]any {
	b := map[string]any{"type": "tool_result", "tool_use_id": m.ToolCallID}
	if m.HasImage() {
		b["content"] = contentBlocks(m)
	} else {
		b["content"] = toolOutputText(m)
	}
	return b
}

// contentBlocks spells a user turn's content. An empty or whitespace text
// block is left out: the protocol refuses one, where Chat Completions takes
// it without a word, so an OpenAI history carries them. A turn left with
// nothing is refused in plan (checkMessagesTurns).
func contentBlocks(m Message) []any {
	if len(m.Parts) == 0 {
		if !hasText(m.Text) {
			return nil
		}
		return []any{map[string]any{"type": "text", "text": m.Text}}
	}
	out := make([]any, 0, len(m.Parts))
	for _, p := range m.Parts {
		switch {
		case p.Kind == PartImage:
			out = append(out, imageBlock(p.URL))
		case hasText(p.Text):
			out = append(out, map[string]any{"type": "text", "text": p.Text})
		}
	}
	return out
}

// imageBlock spells an image. A data URL, which is how an OpenAI caller inlines
// one, becomes a base64 source; anything else goes as a URL source.
func imageBlock(u string) map[string]any {
	if mediaType, data, ok := parseDataURL(u); ok {
		return map[string]any{"type": "image", "source": map[string]any{
			"type": "base64", "media_type": mediaType, "data": data}}
	}
	return map[string]any{"type": "image", "source": map[string]any{"type": "url", "url": u}}
}

// parseDataURL splits data:<type>;base64,<data>. Only the base64 form: it is
// the one an image is inlined as, and the protocol takes no other.
func parseDataURL(u string) (mediaType, data string, ok bool) {
	rest, found := strings.CutPrefix(u, "data:")
	if !found {
		return "", "", false
	}
	head, data, found := strings.Cut(rest, ",")
	if !found {
		return "", "", false
	}
	mediaType, found = strings.CutSuffix(head, ";base64")
	if !found || mediaType == "" {
		return "", "", false
	}
	return mediaType, data, true
}

// renderMessagesTools spells the tools. input_schema is required, so a tool
// with no parameters takes the empty object schema.
func renderMessagesTools(ts []Tool) []any {
	out := make([]any, 0, len(ts))
	for _, t := range ts {
		schema := any(t.Parameters)
		if len(t.Parameters) == 0 {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		tool := map[string]any{"name": t.Name, "input_schema": schema}
		if t.Description != "" {
			tool["description"] = t.Description
		}
		if t.Strict {
			// Off by default here, unlike Responses, so sent only when asked.
			tool["strict"] = true
		}
		out = append(out, tool)
	}
	return out
}

// wireMessage mirrors the message object, leniently.
type wireMessage struct {
	Type       string            `json:"type"`
	Model      string            `json:"model"`
	Content    []json.RawMessage `json:"content"`
	StopReason string            `json:"stop_reason"`
	Usage      json.RawMessage   `json:"usage"`
	Error      json.RawMessage   `json:"error"`
}

// wireBlock is one content block, response side.
type wireBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	Data      string          `json:"data"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

// parseMessagesResponseWith decodes a non-streamed message.
func parseMessagesResponseWith(redact redactor, raw json.RawMessage) (*ChatResponse, error) {
	t, err := decodeMessage(redact, raw)
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

// decodeMessage reads a message object into a turn.
func decodeMessage(redact redactor, raw json.RawMessage) (turn, error) {
	var w wireMessage
	if err := json.Unmarshal(raw, &w); err != nil {
		return turn{}, fmt.Errorf("llmwire: %w: decoding message: %w (body: %s)",
			ErrMalformedResponse, err, bodySnippet(redact, raw))
	}
	// {"type":"error","error":{…}} under a 200, or a gateway's {"error":…}.
	if len(w.Error) > 0 && !isJSONNull(w.Error) {
		return turn{}, parseAPIErrorWith(redact, 0, raw)
	}
	t := turn{model: w.Model, usage: parseMessagesUsage(w.Usage)}
	var b assembler
	for _, rawBlock := range w.Content {
		var blk wireBlock
		if json.Unmarshal(rawBlock, &blk) != nil {
			continue
		}
		b.block(blk, rawBlock)
	}
	b.into(&t)
	t.finish = messagesFinish(w.StopReason, len(t.calls))
	return t, checkAnswered(redact, t, raw)
}

// assembler builds a turn's text, reasoning, calls and replay blocks from its
// content blocks, in order. Shared by the reply and the stream.
type assembler struct {
	content, reasoning strings.Builder
	calls              []ToolCall
	items              []json.RawMessage
	unreplayable       bool
	// answered is set by any block that is not thinking: a turn of thinking
	// alone (a reply cut at max_tokens mid-think) has nothing to replay.
	answered bool
}

// block takes one finished content block. raw is the block as the endpoint
// sent it, replayed verbatim when its type is not one this package spells;
// nil from the stream, where a block of such a type is only its start frame
// and its deltas were never joined, so the turn cannot go back as blocks.
func (a *assembler) block(blk wireBlock, raw json.RawMessage) {
	var item any
	switch blk.Type {
	case "text":
		a.content.WriteString(blk.Text)
		if !hasText(blk.Text) {
			// Replayed, an empty or whitespace text block is a 400.
			return
		}
		item = map[string]any{"type": "text", "text": blk.Text}
	case "thinking":
		a.reasoning.WriteString(blk.Thinking)
		item = map[string]any{"type": "thinking", "thinking": blk.Thinking, "signature": blk.Signature}
	case "redacted_thinking":
		item = map[string]any{"type": "redacted_thinking", "data": blk.Data}
	case "tool_use":
		args := compactJSON(blk.Input)
		a.calls = append(a.calls, ToolCall{ID: blk.ID, Type: "function", Name: blk.Name, Arguments: args})
		if !json.Valid([]byte(args)) {
			// A call whose input did not arrive whole cannot go back as a
			// block; the turn then replays as Text and ToolCalls.
			a.unreplayable = true
			return
		}
		item = map[string]any{"type": "tool_use", "id": blk.ID, "name": blk.Name, "input": json.RawMessage(args)}
	default:
		if len(raw) == 0 {
			a.unreplayable = true
			return
		}
		a.answered = true
		a.items = append(a.items, raw)
		return
	}
	if blk.Type == "text" || blk.Type == "tool_use" {
		a.answered = true
	}
	if enc, err := json.Marshal(item); err == nil {
		a.items = append(a.items, enc)
	}
}

// into writes the assembled turn. One of thinking alone (a reply cut at
// max_tokens mid-think) carries no items: an assistant turn of nothing but a
// thinking block has nothing to answer from, and replays as the empty turn it
// is. Server-tool blocks, unmodelled, still count as an answer and replay.
func (a *assembler) into(t *turn) {
	t.content, t.reasoning, t.calls = a.content.String(), a.reasoning.String(), a.calls
	if !a.unreplayable && a.answered {
		t.items = a.items
	}
}

// compactJSON is a tool input as the arguments string consumers read: the
// object's text, "{}" for an absent one.
func compactJSON(raw json.RawMessage) string {
	if len(raw) == 0 || isJSONNull(raw) {
		return "{}"
	}
	var buf bytes.Buffer
	if json.Compact(&buf, raw) != nil {
		return string(raw)
	}
	return buf.String()
}

// messagesFinish maps stop_reason onto the Chat Completions vocabulary where one
// exists, so a caller checking "length" or "tool_calls" need not know which
// protocol answered; anything else verbatim, as an open set. A refusal is the
// model declining under its safety classifiers, OpenAI's content_filter.
func messagesFinish(reason string, calls int) string {
	switch reason {
	case "end_turn", "stop_sequence":
		if calls > 0 {
			return "tool_calls"
		}
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	default:
		return reason
	}
}

// checkAnswered refuses a turn with no stop_reason, which no finished message
// lacks (a proxy's bare {} under a 200 reads exactly so), and one with no
// text, no call and a clean stop: an empty success in all but name.
func checkAnswered(redact redactor, t turn, raw []byte) error {
	if t.finish == "" {
		return fmt.Errorf("llmwire: %w: message carried no stop_reason (body: %s)",
			ErrResponseShape, bodySnippet(redact, raw))
	}
	if len(t.calls) == 0 && t.content == "" && t.finish == "stop" {
		return fmt.Errorf("llmwire: %w: message carried no answer and no error (body: %s)",
			ErrResponseShape, bodySnippet(redact, raw))
	}
	return nil
}

// wireMessagesUsage mirrors the Messages usage object. Unlike the OpenAI
// shape, input_tokens is the UNCACHED remainder only: cache reads and cache
// writes sit BESIDE it, and the prompt is the sum of the three.
type wireMessagesUsage struct {
	InputTokens              *int64 `json:"input_tokens"`
	OutputTokens             *int64 `json:"output_tokens"`
	CacheCreationInputTokens *int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int64 `json:"cache_read_input_tokens"`
	CacheCreation            *struct {
		Ephemeral5m *int64 `json:"ephemeral_5m_input_tokens"`
		Ephemeral1h *int64 `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation,omitempty"`
	// OutputTokensDetails.ThinkingTokens is the thinking INSIDE
	// output_tokens. MEASURED: output 16000 with thinking_tokens 16000 on a
	// stream that stopped at max_tokens mid-think.
	OutputTokensDetails *struct {
		ThinkingTokens *int64 `json:"thinking_tokens"`
	} `json:"output_tokens_details,omitempty"`
}

// merge lays a later usage object over this one, field by field: the stream's
// message_delta totals are cumulative and replace message_start's.
func (u *wireMessagesUsage) merge(later wireMessagesUsage) {
	if later.InputTokens != nil {
		u.InputTokens = later.InputTokens
	}
	if later.OutputTokens != nil {
		u.OutputTokens = later.OutputTokens
	}
	if later.CacheCreationInputTokens != nil {
		u.CacheCreationInputTokens = later.CacheCreationInputTokens
	}
	if later.CacheReadInputTokens != nil {
		u.CacheReadInputTokens = later.CacheReadInputTokens
	}
	if later.CacheCreation != nil {
		u.CacheCreation = later.CacheCreation
	}
	if later.OutputTokensDetails != nil {
		u.OutputTokensDetails = later.OutputTokensDetails
	}
}

// parseMessagesUsage decodes a usage object; absent or malformed is the zero
// Usage, as parseUsage.
func parseMessagesUsage(raw json.RawMessage) Usage {
	if len(raw) == 0 || isJSONNull(raw) {
		return Usage{}
	}
	var w wireMessagesUsage
	if json.Unmarshal(raw, &w) != nil {
		return Usage{Raw: raw}
	}
	return w.usage(raw)
}

// usage normalises the lanes. NoCache is input_tokens itself, never a
// subtraction; Total is built by ADDING the three input lanes, the one place
// this package adds rather than subtracts, because here they are disjoint.
// MEASURED 2026-10-26 on claude-opus-5-5: an 11215-token system prompt under
// cache_control reported input_tokens 4 with cache_creation 11215, then
// input_tokens 4 with cache_read 11215 on the identical second call.
// Thinking is inside output_tokens, as reasoning is on the other protocols.
func (w wireMessagesUsage) usage(raw json.RawMessage) Usage {
	u := Usage{Raw: raw}
	u.Input.NoCache = nonNegative(w.InputTokens)
	u.Input.CacheRead = nonNegative(w.CacheReadInputTokens)
	u.Input.CacheWrite = nonNegative(w.CacheCreationInputTokens)
	u.Output.Total = nonNegative(w.OutputTokens)
	if w.CacheCreation != nil {
		u.longCacheWrite = nonNegative(w.CacheCreation.Ephemeral1h)
	}
	if w.OutputTokensDetails != nil {
		u.Output.Reasoning = nonNegative(w.OutputTokensDetails.ThinkingTokens)
	}
	if u.Input.NoCache != nil {
		total := *u.Input.NoCache + Tokens(u.Input.CacheRead) + Tokens(u.Input.CacheWrite)
		u.Input.Total = &total
	}
	u.Output.Text = subClamped(u.Output.Total, u.Output.Reasoning)
	u.reported = u.Input.NoCache != nil || u.Output.Total != nil ||
		u.Input.CacheRead != nil || u.Input.CacheWrite != nil || u.Output.Reasoning != nil
	return u
}

// messagesItemCallIDs lists the tool_use ids among replay blocks, in order, for
// plan's agreement check.
func messagesItemCallIDs(items []json.RawMessage) ([]string, error) {
	ids := []string{}
	for i, raw := range items {
		var it struct {
			Type string `json:"type"`
			ID   string `json:"id"`
		}
		if !strings.HasPrefix(strings.TrimSpace(string(raw)), "{") || json.Unmarshal(raw, &it) != nil {
			return nil, fmt.Errorf("ResponseItems[%d] is not a JSON object; append Assistant() unmodified", i)
		}
		if it.Type == "tool_use" {
			ids = append(ids, it.ID)
		}
	}
	return ids, nil
}

// itemsWire says which protocol a turn's ResponseItems came from. A Responses
// item is typed reasoning, function_call or message, or is a role object; a
// Messages block is any other typed object. A conversation switched between
// models carries both kinds, and an item replayed into the other protocol is a
// 400 at best.
func itemsWire(items []json.RawMessage) (WireAPI, error) {
	var wire WireAPI
	for i, raw := range items {
		var it struct {
			Type string `json:"type"`
			Role string `json:"role"`
		}
		if !strings.HasPrefix(strings.TrimSpace(string(raw)), "{") || json.Unmarshal(raw, &it) != nil {
			return "", fmt.Errorf("ResponseItems[%d] is not a JSON object; append Assistant() unmodified", i)
		}
		this := WireMessages
		if it.Role != "" || slices.Contains([]string{"reasoning", "function_call", "message"}, it.Type) {
			this = WireResponses
		}
		if it.Type == "" && it.Role == "" {
			return "", fmt.Errorf("ResponseItems[%d] has neither type nor role; append Assistant() unmodified", i)
		}
		if wire != "" && wire != this {
			return "", fmt.Errorf("ResponseItems mix the %s and %s protocols; append Assistant() unmodified", wire, this)
		}
		wire = this
	}
	return wire, nil
}

// messagesEvent is one `data:` frame of a Messages stream.
type messagesEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	Message      json.RawMessage `json:"message"`
	ContentBlock json.RawMessage `json:"content_block"`
	Delta        struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage json.RawMessage `json:"usage"`
	Error json.RawMessage `json:"error"`
}

// streamBlock is one content block being assembled from its deltas.
type streamBlock struct {
	blk                             wireBlock
	text, thinking, signature, args strings.Builder
	// ord is a tool_use block's ordinal among calls, the Index a consumer sees.
	ord int
}

// readMessagesStream consumes a Messages SSE body. The transport rules are
// readStream's, through the same frameMeter, with one difference the head
// comment explains: a ping frame neither re-arms the guard nor counts.
func readMessagesStream(body io.Reader, guard *stallGuard, bounds streamBounds, sink func(streamEvent), redact redactor, now func() time.Time, start time.Time) (res StreamResult, err error) {
	defer func() { res.Timing.Total = now().Sub(start) }()
	idle := bounds.idle
	emit := func(ev streamEvent) {
		if sink != nil {
			sink(ev)
		}
	}
	var (
		blocks = map[int]*streamBlock{}
		calls  int
		usage  wireMessagesUsage
		// usageRaw is the last usage object as sent, kept as Raw: the merged
		// lanes are the accounting, the endpoint's own object the evidence.
		usageRaw   json.RawMessage
		stopReason string
		completed  bool
		frameErr   error
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
		payload := strings.TrimSpace(strings.TrimPrefix(line, dataPrefix))
		var ev messagesEvent
		parsed := json.Unmarshal([]byte(payload), &ev) == nil
		if parsed && ev.Type == "ping" {
			continue
		}
		meter.frame(&res)
		guard.arm(idle, stallIdle)
		if !parsed {
			continue
		}

		switch ev.Type {
		case "message_start":
			var m wireMessage
			if json.Unmarshal(ev.Message, &m) == nil {
				var u wireMessagesUsage
				if len(m.Usage) > 0 && json.Unmarshal(m.Usage, &u) == nil {
					usage.merge(u)
					usageRaw = m.Usage
				}
			}
		case "content_block_start":
			sb := &streamBlock{}
			if json.Unmarshal(ev.ContentBlock, &sb.blk) != nil {
				continue
			}
			blocks[ev.Index] = sb
			if sb.blk.Type != "tool_use" {
				continue
			}
			if bounds.toolIdle > 0 && idle != bounds.toolIdle {
				idle = bounds.toolIdle
				guard.arm(idle, stallIdle)
			}
			sb.ord = calls
			calls++
			var tc toolCallDelta
			tc.Index, tc.ID, tc.Type, tc.Function.Name = sb.ord, sb.blk.ID, "function", sb.blk.Name
			emit(streamEvent{kind: evToolCall, toolCall: tc})
		case "content_block_delta":
			sb, ok := blocks[ev.Index]
			if !ok {
				continue
			}
			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text != "" {
					sb.text.WriteString(ev.Delta.Text)
					res.Chars += int64(utf8.RuneCountInString(ev.Delta.Text))
					emit(streamEvent{kind: evContent, text: ev.Delta.Text})
				}
			case "thinking_delta":
				if ev.Delta.Thinking != "" {
					sb.thinking.WriteString(ev.Delta.Thinking)
					emit(streamEvent{kind: evReasoning, text: ev.Delta.Thinking})
				}
			case "signature_delta":
				sb.signature.WriteString(ev.Delta.Signature)
			case "input_json_delta":
				if ev.Delta.PartialJSON != "" {
					sb.args.WriteString(ev.Delta.PartialJSON)
					emit(streamEvent{kind: evToolCall, toolCall: argsDelta(sb.ord, ev.Delta.PartialJSON)})
				}
			}
		case "message_delta":
			if ev.Delta.StopReason != "" {
				stopReason = ev.Delta.StopReason
			}
			var u wireMessagesUsage
			if len(ev.Usage) > 0 && json.Unmarshal(ev.Usage, &u) == nil {
				usage.merge(u)
				usageRaw = ev.Usage
			}
		case "message_stop":
			completed = true
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

	var t turn
	var a assembler
	for _, i := range slices.Sorted(maps.Keys(blocks)) {
		sb := blocks[i]
		// The deltas build the block. A start may carry a text or thinking
		// prefix, kept ahead of them; a tool_use start's input is the {}
		// placeholder and is replaced, never joined.
		sb.blk.Text += sb.text.String()
		sb.blk.Thinking += sb.thinking.String()
		sb.blk.Signature += sb.signature.String()
		if sb.blk.Type == "tool_use" {
			sb.blk.Input = json.RawMessage(sb.args.String())
		}
		a.block(sb.blk, nil)
	}
	a.into(&t)
	res.Content, res.Reasoning, res.ToolCalls = t.content, t.reasoning, t.calls
	if usageRaw != nil {
		res.Usage = usage.usage(usageRaw)
	}

	switch {
	case frameErr != nil:
		return res, frameErr
	case completed:
		res.ResponseItems = t.items
		res.FinishReason = messagesFinish(stopReason, len(t.calls))
		t.finish = res.FinishReason
		if err := checkAnswered(redact, t, nil); err != nil {
			return res, err
		}
		emit(streamEvent{kind: evFinish, finishReason: res.FinishReason})
		return res, nil
	case sc.Err() != nil:
		return res, scanEndErr(sc.Err())
	default:
		return res, fmt.Errorf("llmwire: %w: stream ended after %d events (%d chars) without message_stop",
			ErrMalformedResponse, res.Events, res.Chars)
	}
}
