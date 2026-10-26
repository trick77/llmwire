package llmwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Fixtures are shaped on the probes of 2026-10-26 (claude-opus-5-5 on
// api.anthropic.com, eval_messages_test.go), the signature shortened; it is
// opaque either way.

// opusRound1 is a thinking block, then the call, as round 1 of the tool loop
// probe returned them.
const opusRound1 = `{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5",
"content":[
 {"type":"thinking","thinking":"26 x 4 - 70 = 34, even, so Bern.","signature":"CAQSrwUKEAgT"},
 {"type":"tool_use","id":"toolu_01AM","name":"get_weather","input":{"city": "Bern"},"caller":null}],
"stop_reason":"tool_use","stop_sequence":null,
"usage":{"input_tokens":455,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,
 "cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},"output_tokens":101,
 "output_tokens_details":{"thinking_tokens":50},"service_tier":"standard","inference_geo":"global","speed":"standard"}}`

const opusRound2 = `{"id":"msg_02","type":"message","role":"assistant","model":"claude-opus-5-5",
"content":[{"type":"text","text":"Bern: 14°C, overcast.","citations":null}],
"stop_reason":"end_turn","usage":{"input_tokens":587,"output_tokens":33}}`

func opusRequest(msgs ...Message) ChatRequest {
	return ChatRequest{Model: "claude-opus-5-5", Messages: msgs, Tools: []Tool{weatherTool},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto}, Reasoning: ReasoningEffort("high")}
}

func TestRenderMessages_Body(t *testing.T) {
	c := testClient(t, nil)
	req := opusRequest(System("be brief"), User("weather in Bern?"), System("in English"))
	req.ToolChoice = ToolChoice{Mode: ToolChoiceRequired}
	req.Stop = []string{"END"}
	req.ResponseFormat = ResponseFormat{Kind: FormatJSONSchema, Name: "w", Schema: map[string]any{"type": "object"}, Strict: true}
	pl, warnings, err := c.plan(req, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 1 || warnings[0].Feature != "tool_choice" {
		t.Errorf("warnings = %v, want the required->auto relaxation", warnings)
	}
	if pl.route() != "/messages" {
		t.Errorf("route = %s", pl.route())
	}
	body := renderedBody(t, pl)
	for key, want := range map[string]any{
		"model":  "claude-opus-5-5",
		"stream": true,
		// No cap asked: the required field goes out at the model's limit.
		"max_tokens":     float64(128000),
		"system":         "be brief\n\nin English",
		"thinking":       map[string]any{"type": "adaptive", "display": "summarized"},
		"tool_choice":    map[string]any{"type": "auto"},
		"stop_sequences": []any{"END"},
		"output_config": map[string]any{"effort": "high",
			"format": map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}}},
		"messages": []any{map[string]any{"role": "user",
			"content": []any{map[string]any{"type": "text", "text": "weather in Bern?"}}}},
	} {
		if !reflect.DeepEqual(body[key], want) {
			t.Errorf("%s = %#v, want %#v", key, body[key], want)
		}
	}
	for _, key := range []string{"stream_options", "max_completion_tokens", "reasoning_effort", "response_format", "temperature"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s sent on the Messages protocol", key)
		}
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	if tool["name"] != "get_weather" || tool["input_schema"] == nil || tool["strict"] != nil {
		t.Errorf("tool = %#v", tool)
	}
}

// The tool loop as a consumer runs it: Assistant() and the results appended,
// the signed thinking block replayed verbatim before its call, the parallel
// results merged into one user turn.
func TestMessages_ToolLoopReplaysThinking(t *testing.T) {
	r1, err := parseMessagesResponseWith(Redact, json.RawMessage(opusRound1))
	if err != nil {
		t.Fatal(err)
	}
	want := ToolCall{ID: "toolu_01AM", Type: "function", Name: "get_weather", Arguments: `{"city":"Bern"}`}
	if r1.FinishReason != "tool_calls" || len(r1.ToolCalls) != 1 || r1.ToolCalls[0] != want ||
		r1.Reasoning != "26 x 4 - 70 = 34, even, so Bern." || len(r1.ResponseItems) != 2 {
		t.Fatalf("round 1 = %+v", r1)
	}
	u := r1.Usage
	if !u.Reported() || Tokens(u.Input.Total) != 455 || Tokens(u.Input.NoCache) != 455 || Tokens(u.Output.Total) != 101 ||
		Tokens(u.Output.Reasoning) != 50 || Tokens(u.Output.Text) != 51 {
		t.Errorf("usage = %+v", u)
	}

	c := testClient(t, nil)
	// Two results for one call id is not a real loop; it is the shape of a
	// parallel round, which must reach the wire as ONE user turn.
	pl := mustPlan(t, c, opusRequest(User("weather?"), r1.Assistant(),
		ToolResult("toolu_01AM", `{"temp_c":14}`), ToolResult("toolu_01AM", "again"), User("and?")), false)
	got, _ := json.Marshal(renderedBody(t, pl)["messages"])
	wantWire := `[{"content":[{"text":"weather?","type":"text"}],"role":"user"},` +
		`{"content":[{"signature":"CAQSrwUKEAgT","thinking":"26 x 4 - 70 = 34, even, so Bern.","type":"thinking"},` +
		`{"id":"toolu_01AM","input":{"city":"Bern"},"name":"get_weather","type":"tool_use"}],"role":"assistant"},` +
		`{"content":[{"content":"{\"temp_c\":14}","tool_use_id":"toolu_01AM","type":"tool_result"},` +
		`{"content":"again","tool_use_id":"toolu_01AM","type":"tool_result"},{"text":"and?","type":"text"}],"role":"user"}]`
	if string(got) != wantWire {
		t.Errorf("messages =\n%s\nwant\n%s", got, wantWire)
	}

	r2, err := parseMessagesResponseWith(Redact, json.RawMessage(opusRound2))
	if err != nil {
		t.Fatal(err)
	}
	if r2.Content != "Bern: 14°C, overcast." || r2.FinishReason != "stop" ||
		string(r2.ResponseItems[0]) != `{"text":"Bern: 14°C, overcast.","type":"text"}` {
		t.Errorf("round 2 = %+v", r2)
	}
}

// A hand-built tool turn, measured accepted with no thinking block before it.
func TestRenderMessages_HandBuiltToolTurn(t *testing.T) {
	c := testClient(t, nil)
	turn := Message{Role: RoleAssistant, Text: "checking", ToolCalls: []ToolCall{
		{ID: "t1", Name: "get_weather", Arguments: `{"city":"Bern"}`}, {ID: "t2", Name: "get_weather"}}}
	pl := mustPlan(t, c, opusRequest(User("weather?"), turn, ToolResult("t1", "14"), ToolResult("t2", "15")), false)
	got, _ := json.Marshal(renderedBody(t, pl)["messages"].([]any)[1])
	want := `{"content":[{"text":"checking","type":"text"},` +
		`{"id":"t1","input":{"city":"Bern"},"name":"get_weather","type":"tool_use"},` +
		`{"id":"t2","input":{},"name":"get_weather","type":"tool_use"}],"role":"assistant"}`
	if string(got) != want {
		t.Errorf("assistant turn =\n%s\nwant\n%s", got, want)
	}

	bad := Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "t1", Name: "get_weather", Arguments: `["Bern"]`}}}
	if _, _, err := c.plan(opusRequest(User("x"), bad, ToolResult("t1", "14")), false); err == nil ||
		!strings.Contains(err.Error(), "not a JSON object") {
		t.Errorf("array arguments: err = %v", err)
	}
}

func TestRenderMessages_Images(t *testing.T) {
	c := testClient(t, nil)
	user := Message{Role: RoleUser, Parts: []Part{{Kind: PartText, Text: "what is this?"},
		{Kind: PartImage, URL: "data:image/png;base64,iVBORw0K"}, {Kind: PartImage, URL: "https://example.com/a.png"}}}
	result := Message{Role: RoleTool, ToolCallID: "t1", Parts: []Part{{Kind: PartImage, URL: "data:,plain"}}}
	turn := Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "t1", Name: "get_weather"}}}
	pl := mustPlan(t, c, opusRequest(user, turn, result), false)
	got, _ := json.Marshal(renderedBody(t, pl)["messages"])
	want := `[{"content":[{"text":"what is this?","type":"text"},` +
		`{"source":{"data":"iVBORw0K","media_type":"image/png","type":"base64"},"type":"image"},` +
		`{"source":{"type":"url","url":"https://example.com/a.png"},"type":"image"}],"role":"user"},` +
		`{"content":[{"id":"t1","input":{},"name":"get_weather","type":"tool_use"}],"role":"assistant"},` +
		`{"content":[{"content":[{"source":{"type":"url","url":"data:,plain"},"type":"image"}],"tool_use_id":"t1","type":"tool_result"}],"role":"user"}]`
	if string(got) != want {
		t.Errorf("messages =\n%s\nwant\n%s", got, want)
	}

	assistantImage := Message{Role: RoleAssistant, Parts: []Part{{Kind: PartImage, URL: "https://example.com/a.png"}}}
	if _, _, err := c.plan(opusRequest(User("x"), assistantImage), false); err == nil {
		t.Error("an image on an assistant turn was planned")
	}
}

// Replay state from the other protocol is dropped with a warning, never sent;
// a Messages turn whose items and calls disagree is refused.
func TestMessages_ReplayAcrossProtocols(t *testing.T) {
	c := testClient(t, nil)
	sol, err := parseResponsesResponseWith(Redact, json.RawMessage(solRound1))
	if err != nil {
		t.Fatal(err)
	}
	thinking := Message{Role: RoleAssistant, Text: "ok", ReasoningContent: "glm thought"}
	pl, warnings, err := c.plan(opusRequest(User("weather?"), sol.Assistant(), ToolResult("call_FaRG", "14"), thinking, User("go")), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(pl.req.Messages[1].ResponseItems) != 0 || pl.req.Messages[3].ReasoningContent != "" || len(warnings) != 2 {
		t.Errorf("foreign state kept: items %d, reasoning %q, warnings %v",
			len(pl.req.Messages[1].ResponseItems), pl.req.Messages[3].ReasoningContent, warnings)
	}

	// The other way round: Opus blocks sent to the Responses model.
	r1, _ := parseMessagesResponseWith(Redact, json.RawMessage(opusRound1))
	pl = mustPlan(t, c, solRequest(User("weather?"), r1.Assistant(), ToolResult("toolu_01AM", "14")), false)
	if len(pl.req.Messages[1].ResponseItems) != 0 {
		t.Error("Messages blocks kept for a Responses model")
	}

	edited := r1.Assistant()
	edited.ToolCalls[0].ID = "other"
	if _, _, err := c.plan(opusRequest(User("x"), edited, ToolResult("other", "14")), false); err == nil ||
		!strings.Contains(err.Error(), "ResponseItems carry calls") {
		t.Errorf("edited turn: err = %v", err)
	}
	mixed := r1.Assistant()
	mixed.ResponseItems = append(mixed.ResponseItems, sol.ResponseItems[0])
	if _, _, err := c.plan(opusRequest(User("x"), mixed), false); err == nil || !strings.Contains(err.Error(), "mix") {
		t.Errorf("mixed items: err = %v", err)
	}
	if _, err := itemsWire([]json.RawMessage{json.RawMessage(`{"id":"x"}`)}); err == nil {
		t.Error("an untyped item was accepted")
	}
	if _, err := itemsWire([]json.RawMessage{json.RawMessage(`null`)}); err == nil {
		t.Error("a null item was accepted")
	}
}

func TestPlanMessages_Refusals(t *testing.T) {
	c := testClient(t, nil)
	for name, req := range map[string]ChatRequest{
		"thinking off":   {Reasoning: ReasoningOff()},
		"temperature":    {Temperature: f64(0.5)},
		"top_p":          {TopP: f64(0.9)},
		"effort none":    {Reasoning: ReasoningEffort("none")},
		"json_object":    {ResponseFormat: ResponseFormat{Kind: FormatJSONObject}},
		"budget":         {Reasoning: ReasoningBudget(1024)},
		"extra thinking": {ExtraBody: map[string]any{"thinking": map[string]any{"type": "disabled"}}},
		"extra speed":    {ExtraBody: map[string]any{"speed": "fast"}},
		"extra geo":      {ExtraBody: map[string]any{"inference_geo": "us"}},
		"extra system":   {ExtraBody: map[string]any{"system": "x"}},
		"extra cap":      {ExtraBody: map[string]any{"max_completion_tokens": 5}},
	} {
		req.Model, req.Messages = "claude-opus-5-5", []Message{User("hi")}
		if _, _, err := c.plan(req, false); err == nil {
			t.Errorf("%s: planned", name)
		}
	}
	// ReasoningMinimal is the floor, never off.
	pl := mustPlan(t, c, ChatRequest{Model: "claude-opus-5-5", Messages: []Message{User("hi")},
		Reasoning: ReasoningMinimal(), MaxAnswerTokens: ptr(100)}, false)
	if pl.req.Reasoning != (reasoningEffort{level: "low"}) || *pl.req.MaxTokens != 100+DefaultReasoningOverhead {
		t.Errorf("minimal: reasoning %v, cap %d", pl.req.Reasoning, *pl.req.MaxTokens)
	}
	// cache_control rides through: the opt-in to caching.
	pl = mustPlan(t, c, ChatRequest{Model: "claude-opus-5-5", Messages: []Message{User("hi")},
		ExtraBody: map[string]any{"cache_control": map[string]any{"type": "ephemeral"}}}, false)
	if renderedBody(t, pl)["cache_control"] == nil {
		t.Error("cache_control dropped")
	}
	// A conversation of system turns only has nothing to send, nor one whose
	// only turn is empty: refused in plan, so Validate says so.
	for _, ms := range [][]Message{{System("x")}, {System("x"), User("")}} {
		if _, err := c.Validate(ChatRequest{Model: "claude-opus-5-5", Messages: ms}); err == nil {
			t.Errorf("%v validated", ms)
		}
	}
}

func TestParseMessages_Shapes(t *testing.T) {
	for name, tc := range map[string]struct {
		body   string
		finish string
		err    error
	}{
		"max_tokens mid-think": {`{"content":[{"type":"thinking","thinking":"hmm","signature":"s"}],"stop_reason":"max_tokens"}`, "length", nil},
		"refusal":              {`{"content":[],"stop_reason":"refusal"}`, "content_filter", nil},
		"stop sequence":        {`{"content":[{"type":"text","text":"a"}],"stop_reason":"stop_sequence"}`, "stop", nil},
		"open set":             {`{"content":[{"type":"text","text":"a"}],"stop_reason":"pause_turn"}`, "pause_turn", nil},
		"empty success":        {`{"content":[{"type":"thinking","thinking":"x"}],"stop_reason":"end_turn"}`, "", ErrResponseShape},
		"not json":             {`<html>`, "", ErrMalformedResponse},
		"error under 200":      {`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, "", ErrUpstream},
	} {
		resp, err := parseMessagesResponseWith(Redact, json.RawMessage(tc.body))
		switch {
		case tc.err != nil && !errors.Is(err, tc.err):
			t.Errorf("%s: err = %v, want %v", name, err, tc.err)
		case tc.err == nil && err != nil:
			t.Errorf("%s: %v", name, err)
		case tc.err == nil && resp.FinishReason != tc.finish:
			t.Errorf("%s: finish %q, want %q", name, resp.FinishReason, tc.finish)
		}
	}
	// An unknown block type is replayed verbatim; a cut tool input makes the
	// turn unreplayable rather than sending half a call.
	resp, err := parseMessagesResponseWith(Redact, json.RawMessage(
		`{"content":[{"type":"server_tool_use","id":"s1"},{"type":"text","text":"a"}],"stop_reason":"end_turn"}`))
	if err != nil || string(resp.ResponseItems[0]) != `{"type":"server_tool_use","id":"s1"}` {
		t.Errorf("unknown block: %v %s", err, resp.ResponseItems)
	}
	if got := compactJSON(json.RawMessage(`{"a":`)); got != `{"a":` {
		t.Errorf("compactJSON(cut) = %q", got)
	}
}

func TestMessagesUsage_Lanes(t *testing.T) {
	// The measured cache pair: 4 uncached beside an 11215-token write, then
	// beside the read.
	write := parseMessagesUsage(json.RawMessage(`{"input_tokens":4,"cache_creation_input_tokens":11215,` +
		`"cache_read_input_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":11215,"ephemeral_1h_input_tokens":0},"output_tokens":4}`))
	if Tokens(write.Input.Total) != 11219 || Tokens(write.Input.NoCache) != 4 || Tokens(write.Input.CacheWrite) != 11215 {
		t.Errorf("write = %+v", write.Input)
	}
	p, _ := Default().Lookup("claude-opus-5-5")
	cost, warnings := priceCall(p, write, nil, 200, testStart)
	// 4 x 4.00 + 11215 x 5.00 + 4 x 20.00 per million = 56171 micro-USD,
	// the figure the probe priced.
	if cost.Provenance != FromTable || cost.NanoUSD != 56_171_000 || len(warnings) != 0 {
		t.Errorf("write cost = %+v %v", cost, warnings)
	}

	long := parseMessagesUsage(json.RawMessage(`{"input_tokens":4,"cache_creation_input_tokens":100,` +
		`"cache_creation":{"ephemeral_1h_input_tokens":100},"output_tokens":4}`))
	cost, warnings = priceCall(p, long, nil, 200, testStart)
	if cost.Provenance != Unpriced || len(warnings) != 1 || !strings.Contains(warnings[0].Details, "one-hour") {
		t.Errorf("1h write = %+v %v", cost, warnings)
	}

	for _, raw := range []string{``, `null`, `{"input_tokens":"x"}`} {
		if u := parseMessagesUsage(json.RawMessage(raw)); u.Reported() {
			t.Errorf("%q reported", raw)
		}
	}
	if u := parseMessagesUsage(json.RawMessage(`{"output_tokens":3}`)); !u.Reported() || u.Input.Total != nil {
		t.Errorf("output only = %+v", u)
	}
}

// opusStream is the frame sequence the stream probe read, shortened, with a
// ping in the middle of the think.
var opusStream = []string{
	"event: message_start\n" + `data: {"type":"message_start","message":{"model":"claude-opus-5-5","content":[],` +
		`"usage":{"input_tokens":455,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}}`,
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"26 x 4"}}`,
	"event: ping\n" + `data: {"type":"ping"}`,
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" = 104"}}`,
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"CAQS"}}`,
	`data: {"type":"content_block_stop","index":0}`,
	`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
	`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking."}}`,
	`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
	`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":""}}`,
	`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\": "}}`,
	`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Bern\"}"}}`,
	`data: {"type":"content_block_stop","index":2}`,
	`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":455,"output_tokens":109,` +
		`"output_tokens_details":{"thinking_tokens":58}}}`,
	`data: {"type":"message_stop"}`,
}

func TestChatStreamMessages_ToolCall(t *testing.T) {
	srv := sseServer(t, opusStream)
	c := streamClient(t, srv.URL, 5*time.Second)
	s, _, err := c.ChatStream(context.Background(), opusRequest(User("weather?")))
	if err != nil {
		t.Fatal(err)
	}
	var kinds []EventKind
	var args string
	for s.Next() {
		ev := s.Event()
		kinds = append(kinds, ev.Kind)
		args += ev.ArgumentsDelta
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	res := s.Result()
	if srv.last().URL.Path != "/messages" {
		t.Errorf("path = %s", srv.last().URL.Path)
	}
	wantCall := ToolCall{ID: "toolu_1", Type: "function", Name: "get_weather", Arguments: `{"city":"Bern"}`}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0] != wantCall || args != `{"city": "Bern"}` ||
		res.Content != "Checking." || res.Reasoning != "26 x 4 = 104" || res.FinishReason != "tool_calls" {
		t.Errorf("result = %+v, args %q", res, args)
	}
	// The ping is neither an event nor a reason to stay alive.
	if res.Events != int64(len(opusStream)-1) {
		t.Errorf("events = %d, want %d (ping not counted)", res.Events, len(opusStream)-1)
	}
	if Tokens(res.Usage.Input.Total) != 455 || Tokens(res.Usage.Output.Total) != 109 || Tokens(res.Usage.Output.Reasoning) != 58 ||
		res.Usage.Cost.Provenance != FromTable || !strings.Contains(string(res.Usage.Raw), "thinking_tokens") {
		t.Errorf("usage = %+v", res.Usage)
	}
	if len(res.ResponseItems) != 3 || string(res.ResponseItems[0]) != `{"signature":"CAQS","thinking":"26 x 4 = 104","type":"thinking"}` {
		t.Errorf("items = %s", res.ResponseItems)
	}
	if kinds[0] != EventReasoning || kinds[len(kinds)-1] != EventFinish {
		t.Errorf("kinds = %v", kinds)
	}
}

func TestChatStreamMessages_Failures(t *testing.T) {
	start := opusStream[:3]
	for name, tc := range map[string]struct {
		lines []string
		check func(error) bool
	}{
		"cut": {start, func(err error) bool { return errors.Is(err, ErrMalformedResponse) }},
		"error frame": {append(append([]string(nil), start...),
			"event: error\n"+`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
			func(err error) bool {
				var apiErr *APIError
				return errors.As(err, &apiErr) && apiErr.Code == "overloaded_error"
			}},
		"gateway error": {append(append([]string(nil), start...), `data: {"error":{"message":"boom","code":"500"}}`),
			func(err error) bool { var apiErr *APIError; return errors.As(err, &apiErr) && apiErr.Code == "500" }},
		"empty success": {[]string{opusStream[0], `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			`data: {"type":"message_stop"}`}, func(err error) bool { return errors.Is(err, ErrResponseShape) }},
	} {
		srv := sseServer(t, tc.lines)
		c := streamClient(t, srv.URL, 5*time.Second)
		s, _, err := c.ChatStream(context.Background(), opusRequest(User("weather?")))
		if err != nil {
			t.Fatal(err)
		}
		_, err = s.Collect(nil)
		if !tc.check(err) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

// Pings alone must let the idle guard fire: a heartbeat is not a model.
func TestChatStreamMessages_PingDoesNotRearm(t *testing.T) {
	pings := []string{opusStream[0]}
	for range 20 {
		pings = append(pings, `data: {"type":"ping"}`)
	}
	srv := sseServer(t, pings, reply{gap: 30 * time.Millisecond})
	c := streamClient(t, srv.URL, 200*time.Millisecond)
	s, _, err := c.ChatStream(context.Background(), opusRequest(User("weather?")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Collect(nil); !errors.Is(err, ErrStreamIdle) {
		t.Errorf("err = %v, want the idle guard", err)
	}
}

// Chat end to end: the route, the key in x-api-key and never a bearer, the
// host's required header, and the reply parsed.
func TestChatMessages_Transport(t *testing.T) {
	srv := jsonServer(t, http.StatusOK, opusRound2)
	c := clientOn(t, srv.URL, Config{APIKey: "fixture-anthropic-key", APIKeyHeader: "x-api-key",
		Headers: map[string]string{"anthropic-version": "2026-10-21"}})
	resp, _, err := c.Chat(context.Background(), opusRequest(User("weather?")))
	if err != nil {
		t.Fatal(err)
	}
	r := srv.last()
	if r.URL.Path != "/messages" || r.Header.Get("x-api-key") != "fixture-anthropic-key" ||
		r.Header.Get("Authorization") != "" || r.Header.Get("anthropic-version") != "2026-10-21" {
		t.Errorf("request %s headers %v", r.URL.Path, r.Header)
	}
	if resp.Content != "Bern: 14°C, overcast." || resp.Usage.Cost.Provenance != FromTable {
		t.Errorf("resp = %+v", resp)
	}

	srv = jsonServer(t, 529, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_1"}`)
	c = clientOn(t, srv.URL)
	_, _, err = c.Chat(context.Background(), opusRequest(User("weather?")))
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "overloaded_error" || !errors.Is(err, ErrUpstream) {
		t.Errorf("529: %v", err)
	}
}

// FromEnv takes the key header and the required headers from the provider;
// the caller's own Headers win on a clash.
func TestFromEnv_AnthropicProvider(t *testing.T) {
	ep, err := envEndpoint("claude-opus-5-5", Config{Registry: Default(), Headers: map[string]string{"x-extra": "1"}},
		envGetter(mapLookup(map[string]string{"LLMWIRE_ANTHROPIC_API_KEY": "sk-ant-k"})), false)
	if err != nil {
		t.Fatal(err)
	}
	cfg := ep.cfg
	if cfg.BaseURL != "https://api.anthropic.com/v1" || cfg.APIKeyHeader != "x-api-key" || cfg.APIKey != "sk-ant-k" ||
		cfg.Headers["anthropic-version"] != "2026-10-21" || cfg.Headers["x-extra"] != "1" {
		t.Errorf("cfg = %+v", cfg)
	}
	ep, _ = envEndpoint("claude-opus-5-5", Config{Registry: Default(), Headers: map[string]string{"anthropic-version": "2023-06-01"}},
		envGetter(mapLookup(map[string]string{"LLMWIRE_ANTHROPIC_API_KEY": "k"})), false)
	if ep.cfg.Headers["anthropic-version"] != "2023-06-01" {
		t.Errorf("caller header lost: %v", ep.cfg.Headers)
	}
	// Through the gateway the provider's host fields are the gateway's.
	reg, err := Default().viaGateway(map[string]string{"claude-opus-5-5": "opus"})
	if err != nil {
		t.Fatal(err)
	}
	p, _ := reg.Lookup("claude-opus-5-5")
	if p.APIKeyHeader != "" || len(p.Headers) != 0 || p.WireAPI != WireMessages {
		t.Errorf("gateway route = %+v", p)
	}
}

func TestProvider_HeaderValidation(t *testing.T) {
	for name, pv := range map[string]Provider{
		"key header case":  {APIKeyHeader: "X-Api-Key"},
		"header name":      {Headers: map[string]string{"Anthropic Version": "1"}},
		"empty value":      {Headers: map[string]string{"anthropic-version": ""}},
		"key in headers":   {Headers: map[string]string{"authorization": "Bearer x"}},
		"key header twice": {APIKeyHeader: "x-api-key", Headers: map[string]string{"x-api-key": "x"}},
		"undeclared key":   {Headers: map[string]string{"x-api-key": "x"}},
		"token header":     {Headers: map[string]string{"x-auth-token": "x"}},
		"kong apikey":      {Headers: map[string]string{"apikey": "x"}},
		"session id":       {Headers: map[string]string{"x-session-id": "x"}},
	} {
		if pv.validate() == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	ok := Provider{Headers: map[string]string{"anthropic-version": "v", "anthropic-beta": "b"}}
	if err := ok.validate(); err != nil {
		t.Errorf("plain headers refused: %v", err)
	}
}

func TestProfileMessages_Validation(t *testing.T) {
	base, _ := Default().Lookup("claude-opus-5-5")
	for name, mutate := range map[string]func(*Profile){
		"cap param":       func(p *Profile) { p.MaxTokensParam = ParamMaxCompletionTokens },
		"no max_output":   func(p *Profile) { p.Limits.MaxOutput = 0 },
		"budget control":  func(p *Profile) { p.Reasoning.Control = ControlBudget },
		"can disable":     func(p *Profile) { p.Reasoning.CanBeDisabled = true },
		"stream field":    func(p *Profile) { p.Reasoning.StreamField = "reasoning_content" },
		"xml tools":       func(p *Profile) { p.Tools.RecoverInlineMarkup = true },
		"forced choice":   func(p *Profile) { p.Tools.SupportsForcedChoice = true },
		"required choice": func(p *Profile) { p.Tools.ToolChoiceValues = []string{"auto", "required"} },
		"json_object":     func(p *Profile) { p.Output.JSONObject = true },
		"stream_options":  func(p *Profile) { p.Streaming.AcceptsStreamOptions = true },
	} {
		p := base.clone()
		mutate(p)
		if err := p.validateMessages(func(f string, _ ...any) error { return errors.New(f) }); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// The listing is paged on Anthropic: followed by last_id while has_more, and
// max_tokens read as the output cap.
func TestListModels_AnthropicPages(t *testing.T) {
	srv := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("after_id") == "" {
			_, _ = w.Write([]byte(`{"data":[{"type":"model","id":"claude-opus-5-5","max_input_tokens":1000000,` +
				`"max_tokens":128000}],"has_more":true,"last_id":"claude-opus-5-5"}`))
			return
		}
		_, _ = w.Write([]byte(`{"data":[{"type":"model","id":"claude-haiku-5-5"}],"has_more":false}`))
	})
	entries, warnings, err := clientOn(t, srv.URL).ListModels(context.Background())
	if err != nil || len(warnings) != 0 || len(entries) != 2 {
		t.Fatalf("entries %+v warnings %v err %v", entries, warnings, err)
	}
	if Tokens(entries[0].MaxOutputTokens) != 128000 || Tokens(entries[0].MaxInputTokens) != 1000000 {
		t.Errorf("entry = %+v", entries[0])
	}
	if q := srv.last().URL.Query(); q.Get("after_id") != "claude-opus-5-5" || q.Get("limit") != "1000" {
		t.Errorf("second page query = %v", q)
	}

	// A server that never stops paging is cut off, with a warning; last_id
	// absent continues after the page's last row.
	n := 0
	endless := serve(t, func(w http.ResponseWriter, _ *http.Request) {
		n++
		_, _ = fmt.Fprintf(w, `{"data":[{"id":"m%d"}],"has_more":true}`, n)
	})
	entries, warnings, err = clientOn(t, endless.URL).ListModels(context.Background())
	if err != nil || len(entries) != maxModelPages || len(warnings) != 1 || len(endless.requests()) != maxModelPages {
		t.Errorf("endless: %d entries, warnings %v, %d requests, err %v", len(entries), warnings, len(endless.requests()), err)
	}

	// One that ignores after_id answers the same page, one cycles between two:
	// each id kept once, each cursor followed once. One with has_more and no
	// cursor stops at once.
	for name, tc := range map[string]struct {
		pages    []string
		entries  int
		requests int
	}{
		"stuck": {[]string{`{"data":[{"id":"m"}],"has_more":true,"last_id":"m"}`}, 1, 2},
		"cycle": {[]string{`{"data":[{"id":"a"}],"has_more":true,"last_id":"a"}`,
			`{"data":[{"id":"b"}],"has_more":true,"last_id":"b"}`}, 2, 3},
		"no cursor": {[]string{`{"data":[{"bad":1}],"has_more":true}`}, 0, 1},
	} {
		i := 0
		srv := serve(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(tc.pages[i%len(tc.pages)]))
			i++
		})
		entries, warnings, err := clientOn(t, srv.URL).ListModels(context.Background())
		if err != nil || len(entries) != tc.entries || len(warnings) == 0 || len(srv.requests()) != tc.requests {
			t.Errorf("%s: %d entries, warnings %v, %d requests, err %v", name, len(entries), warnings, len(srv.requests()), err)
		}
	}
}

// What the review found: arguments that go out as tool_use only because the
// foreign items were dropped are checked too; a system image is refused; an
// empty text block is left out; a reply with no stop_reason is no success.
func TestMessages_ReviewEdges(t *testing.T) {
	c := testClient(t, nil)
	sol, _ := parseResponsesResponseWith(Redact, json.RawMessage(solRound1))
	turn := sol.Assistant()
	turn.ToolCalls[0].Arguments = `[1]`
	if _, _, err := c.plan(opusRequest(User("x"), turn, ToolResult("call_FaRG", "14")), false); err == nil ||
		!strings.Contains(err.Error(), "not a JSON object") {
		t.Errorf("foreign turn with array arguments: err = %v", err)
	}
	sysImage := Message{Role: RoleSystem, Parts: []Part{{Kind: PartImage, URL: "https://example.com/a.png"}}}
	if _, _, err := c.plan(opusRequest(sysImage, User("x")), false); err == nil {
		t.Error("a system image was planned")
	}
	pl := mustPlan(t, c, opusRequest(User("q"), Assistant("a"),
		Message{Role: RoleUser, Parts: []Part{{Kind: PartText}, {Kind: PartText, Text: "b"}}}), false)
	got, _ := json.Marshal(renderedBody(t, pl)["messages"].([]any)[2])
	if want := `{"content":[{"text":"b","type":"text"}],"role":"user"}`; string(got) != want {
		t.Errorf("last turn = %s, want %s", got, want)
	}
	// An empty user turn is dropped mid-conversation, and refused where
	// dropping it would end the request on the assistant turn, a prefill.
	pl = mustPlan(t, c, opusRequest(User(""), User("q")), false)
	if got := renderedBody(t, pl)["messages"].([]any); len(got) != 1 {
		t.Errorf("empty first turn: %v", got)
	}
	// The coerced request is what goes out: the empty turn is gone from it,
	// and the warning names it by index.
	if w := pl.req; len(w.Messages) != 1 {
		t.Errorf("the coerced request kept the empty turn: %v", w.Messages)
	}
	if _, warnings, _ := c.plan(opusRequest(User(""), User("q")), false); len(warnings) != 1 ||
		!strings.Contains(warnings[0].Details, "messages [0]") {
		t.Errorf("dropped turn not warned by index: %v", warnings)
	}
	// Refused only where the drop makes the request end on an assistant
	// turn the caller did not end on.
	for name, ms := range map[string][]Message{
		"ends on assistant":  {User("q"), Assistant("a"), User(""), System("s")},
		"two empty trailing": {User("q"), Assistant("a"), User(""), Assistant("")},
	} {
		if _, _, err := c.plan(opusRequest(ms...), false); err == nil || !strings.Contains(err.Error(), "prefill") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
	// What the caller sent stays the caller's: a prefill, an assistant-first
	// history once the empty turn is out, a pause_turn resume, and two empty
	// trailing turns after a real user turn.
	for name, ms := range map[string][]Message{
		"prefill as sent":    {User("q"), Assistant("{")},
		"opens on assistant": {User(""), Assistant("a"), User("q")},
		"pause_turn resume": {User("q"), {Role: RoleAssistant,
			ResponseItems: []json.RawMessage{json.RawMessage(`{"type":"server_tool_use","id":"s","name":"web_search","input":{}}`)}}},
		"empty after q": {User("q"), Assistant(""), User("")},
	} {
		if _, _, err := c.plan(opusRequest(ms...), false); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	// An empty text block in a reply is not replayed; a reply of thinking
	// alone carries no items at all.
	r, err := parseMessagesResponseWith(Redact, json.RawMessage(
		`{"content":[{"type":"text","text":""},{"type":"tool_use","id":"t","name":"f","input":{}}],"stop_reason":"tool_use"}`))
	if err != nil || len(r.ResponseItems) != 1 {
		t.Errorf("empty reply text replayed: %s %v", r.ResponseItems, err)
	}
	r, err = parseMessagesResponseWith(Redact, json.RawMessage(
		`{"content":[{"type":"thinking","thinking":"x","signature":"s"},{"type":"text","text":""}],"stop_reason":"max_tokens"}`))
	if err != nil || r.ResponseItems != nil {
		t.Errorf("thinking-only reply replayed: %s %v", r.ResponseItems, err)
	}
	// A server-tool turn is an answer: it replays, so pause_turn can resume.
	r, err = parseMessagesResponseWith(Redact, json.RawMessage(
		`{"content":[{"type":"server_tool_use","id":"s","name":"web_search","input":{}}],"stop_reason":"pause_turn"}`))
	if err != nil || len(r.ResponseItems) != 1 {
		t.Errorf("server-tool reply not replayed: %s %v", r.ResponseItems, err)
	}
	for _, body := range []string{`{}`, `{"type":"message","content":[]}`} {
		if _, err := parseMessagesResponseWith(Redact, json.RawMessage(body)); !errors.Is(err, ErrResponseShape) {
			t.Errorf("%s: err = %v", body, err)
		}
	}
	// A streamed block of a type this package does not spell cannot replay.
	res, err := readMessagesFixture(t, []string{opusStream[0],
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"s1","input":{}}}`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":"a"}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`, `data: {"type":"message_stop"}`})
	if err != nil || res.Content != "a" || res.ResponseItems != nil {
		t.Errorf("unknown streamed block: %+v %v", res, err)
	}
}

func readMessagesFixture(t *testing.T, lines []string) (StreamResult, error) {
	t.Helper()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	return readMessagesStream(strings.NewReader(strings.Join(lines, "\n\n")+"\n\n"), guard,
		streamBounds{idle: time.Hour}, nil, Redact, time.Now, time.Now())
}

// The caller's header wins over the provider's whatever its case, and a
// caller's own key header over the provider's.
func TestMergeHeaders_CaseInsensitive(t *testing.T) {
	got := mergeHeaders(map[string]string{"anthropic-version": "2026-10-21", "x-other": "p"},
		map[string]string{"Anthropic-Version": "2023-06-01"})
	if len(got) != 2 || got["Anthropic-Version"] != "2023-06-01" || got["x-other"] != "p" {
		t.Errorf("merged = %v", got)
	}
	ep, err := envEndpoint("claude-opus-5-5", Config{Registry: Default(), APIKeyHeader: "authorization-token"},
		envGetter(mapLookup(map[string]string{"LLMWIRE_ANTHROPIC_API_KEY": "k"})), false)
	if err != nil || ep.cfg.APIKeyHeader != "authorization-token" {
		t.Errorf("caller key header lost: %q %v", ep.cfg.APIKeyHeader, err)
	}
	// Across providers one header cannot be right for all of them.
	_, err = FromEnvModels(Config{APIKeyHeader: "x-api-key", Lookup: mapLookup(map[string]string{
		"LLMWIRE_ANTHROPIC_API_KEY": "a", "LLMWIRE_OPENAI_API_KEY": "o"})}, "claude-opus-5-5", "gpt-5.5")
	if err == nil || !strings.Contains(err.Error(), "APIKeyHeader") {
		t.Errorf("FromEnvModels across providers: err = %v", err)
	}
}

func TestParseDataURL(t *testing.T) {
	for in, ok := range map[string]bool{
		"data:image/png;base64,AAAA": true,
		"data:,plain":                false,
		"data:image/png,AAAA":        false,
		"data:;base64,AAAA":          false,
		"https://x/a.png":            false,
	} {
		if _, _, got := parseDataURL(in); got != ok {
			t.Errorf("%q: ok = %v", in, got)
		}
	}
}
