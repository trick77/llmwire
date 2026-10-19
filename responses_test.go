package llmwire

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Fixtures are the measured gateway bodies of 2026-10-19 (gpt-6.1-sol via
// LiteLLM 1.105.1 → Azure), cut to the fields that matter and with the
// encrypted reasoning shortened; it is opaque either way.

// solRound1 is call 58975efa: a reasoning item, then the call.
const solRound1 = `{"id":"resp_x","object":"response","model":"ai-gateway-gpt-6.1-sol","status":"completed",
"error":null,"incomplete_details":null,
"output":[
 {"id":"rs_0ef3","summary":[],"type":"reasoning","content":[],"encrypted_content":"gAAAAABq1gmV"},
 {"arguments":"{\"city\":\"Bern\"}","call_id":"call_FaRG","name":"get_weather","type":"function_call","id":"fc_0ef3","namespace":null,"status":"completed"}],
"tools":[{"name":"get_weather","strict":true,"type":"function"}],
"usage":{"input_tokens":97,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
 "output_tokens":54,"output_tokens_details":{"reasoning_tokens":34},"total_tokens":151,"cost":null},
"content_filters":[{"blocked":false}]}`

// solRound2 is call 32b33ec8: a new reasoning item, then the answer.
const solRound2 = `{"object":"response","model":"ai-gateway-gpt-6.1-sol","status":"completed","error":null,
"output":[
 {"id":"rs_0ef4","summary":[],"type":"reasoning","content":[],"encrypted_content":"gAAAAABq1gnc"},
 {"id":"msg_0ef3","content":[{"annotations":[],"text":"14 degrees, overcast.","type":"output_text","logprobs":[]}],
  "role":"assistant","status":"completed","type":"message","phase":"final_answer"}],
"usage":{"input_tokens":177,"input_tokens_details":{"cached_tokens":0},"output_tokens":34,"output_tokens_details":{"reasoning_tokens":13},"total_tokens":211}}`

var weatherTool = Tool{Name: "get_weather", Description: "Current weather for a city",
	Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}}

func solRequest(msgs ...Message) ChatRequest {
	return ChatRequest{Model: "gpt-6.1-sol", Messages: msgs, Tools: []Tool{weatherTool},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto}, Reasoning: ReasoningEffort("high")}
}

func renderedBody(t *testing.T, pl *wirePlan) map[string]any {
	t.Helper()
	raw, err := pl.render()
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	return body
}

func TestRenderResponses_Body(t *testing.T) {
	c := testClient(t, nil)
	req := solRequest(System("be brief"), User("weather in Bern?"))
	req.MaxTokens = ptr(4000)
	req.ToolChoice = ToolChoice{Mode: ToolChoiceFunction, Name: "get_weather"}
	req.ResponseFormat = ResponseFormat{Kind: FormatJSONSchema, Name: "w", Schema: map[string]any{"type": "object"}, Strict: true}
	body := renderedBody(t, mustPlan(t, c, req, true))

	for key, want := range map[string]any{
		"model":             "gpt-6.1-sol",
		"store":             false,
		"stream":            true,
		"max_output_tokens": float64(4000),
		"include":           []any{"reasoning.encrypted_content"},
		"reasoning":         map[string]any{"effort": "high"},
		"tool_choice":       map[string]any{"type": "function", "name": "get_weather"},
		"text": map[string]any{"format": map[string]any{"type": "json_schema", "name": "w",
			"schema": map[string]any{"type": "object"}, "strict": true}},
		"input": []any{
			map[string]any{"role": "system", "content": "be brief"},
			map[string]any{"role": "user", "content": "weather in Bern?"},
		},
	} {
		if !reflect.DeepEqual(body[key], want) {
			t.Errorf("%s = %#v, want %#v", key, body[key], want)
		}
	}
	for _, key := range []string{"messages", "stream_options", "max_completion_tokens", "reasoning_effort", "response_format"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s sent on the Responses protocol", key)
		}
	}
	// Flat, and strict always stated: the protocol defaults a tool to strict.
	tool := body["tools"].([]any)[0].(map[string]any)
	if tool["name"] != "get_weather" || tool["strict"] != false || tool["function"] != nil {
		t.Errorf("tool = %#v", tool)
	}
}

// The tool loop as noergler runs it: round two appends Assistant() and the
// result, and the reasoning item goes back verbatim, before its call.
func TestResponses_ToolLoopReplaysReasoning(t *testing.T) {
	r1, err := parseResponsesResponseWith(Redact, json.RawMessage(solRound1))
	if err != nil {
		t.Fatal(err)
	}
	if r1.FinishReason != "tool_calls" || len(r1.ToolCalls) != 1 ||
		r1.ToolCalls[0] != (ToolCall{ID: "call_FaRG", Type: "function", Name: "get_weather", Arguments: `{"city":"Bern"}`}) {
		t.Fatalf("round 1 = %+v", r1)
	}
	if Tokens(r1.Usage.Input.Total) != 97 || Tokens(r1.Usage.Output.Total) != 54 ||
		Tokens(r1.Usage.Output.Reasoning) != 34 || Tokens(r1.Usage.Output.Text) != 20 {
		t.Errorf("usage = %+v", r1.Usage)
	}

	c := testClient(t, nil)
	pl := mustPlan(t, c, solRequest(User("weather?"), r1.Assistant(),
		ToolResult("call_FaRG", `{"temp_c":14}`)), false)
	input, _ := json.Marshal(renderedBody(t, pl)["input"])
	// The trimmed shapes round two sent and Azure accepted.
	want := `[{"content":"weather?","role":"user"},` +
		`{"encrypted_content":"gAAAAABq1gmV","id":"rs_0ef3","summary":[],"type":"reasoning"},` +
		`{"arguments":"{\"city\":\"Bern\"}","call_id":"call_FaRG","id":"fc_0ef3","name":"get_weather","type":"function_call"},` +
		`{"call_id":"call_FaRG","output":"{\"temp_c\":14}","type":"function_call_output"}]`
	if string(input) != want {
		t.Errorf("input =\n%s\nwant\n%s", input, want)
	}

	r2, err := parseResponsesResponseWith(Redact, json.RawMessage(solRound2))
	if err != nil {
		t.Fatal(err)
	}
	if r2.Content != "14 degrees, overcast." || r2.FinishReason != "stop" || len(r2.ResponseItems) != 2 {
		t.Errorf("round 2 = %+v", r2)
	}
	// The answer replays as the plain role form, phase dropped.
	if got := string(r2.ResponseItems[1]); got != `{"content":"14 degrees, overcast.","role":"assistant"}` {
		t.Errorf("message item = %s", got)
	}
}

// A hand-built turn (or one from a Chat Completions model) still renders its
// calls as items.
func TestRenderInput_HandBuiltToolTurn(t *testing.T) {
	got, _ := json.Marshal(renderInput([]Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "f", Arguments: "{}"}}},
		{Role: RoleUser, Parts: []Part{{Kind: PartText, Text: "see"}, {Kind: PartImage, URL: "data:x"}}},
	}))
	want := `[{"arguments":"{}","call_id":"c1","name":"f","type":"function_call"},` +
		`{"content":[{"text":"see","type":"input_text"},{"image_url":"data:x","type":"input_image"}],"role":"user"}]`
	if string(got) != want {
		t.Errorf("input =\n%s\nwant\n%s", got, want)
	}
}

func TestPlanResponses_ProtocolChecks(t *testing.T) {
	c := testClient(t, nil)
	r1, _ := parseResponsesResponseWith(Redact, json.RawMessage(solRound1))

	tampered := r1.Assistant()
	tampered.ToolCalls[0].ID = "other"
	var unsup *UnsupportedError
	if _, err := c.Validate(solRequest(User("x"), tampered, ToolResult("other", "y"))); !errors.As(err, &unsup) || unsup.Demotable {
		t.Errorf("ResponseItems disagreeing with ToolCalls: err %v", err)
	}
	misplaced := User("x")
	misplaced.ResponseItems = r1.ResponseItems
	if _, err := c.Validate(solRequest(misplaced)); err == nil {
		t.Error("ResponseItems on a user turn were accepted")
	}

	req := solRequest(User("x"))
	req.Stop = []string{"END"}
	if _, err := c.Validate(req); !errors.As(err, &unsup) || !unsup.Demotable {
		t.Errorf("stop: err %v, want a demotable refusal", err)
	}
	req.BestEffort = true
	if pl := mustPlan(t, c, req, false); pl.req.Stop != nil {
		t.Errorf("BestEffort kept stop %v", pl.req.Stop)
	}

	for _, key := range []string{"store", "include", "input", "max_completion_tokens", "max_tokens", "stream"} {
		req := solRequest(User("x"))
		req.ExtraBody = map[string]any{key: true}
		if _, err := c.Validate(req); err == nil {
			t.Errorf("ExtraBody %q accepted", key)
		}
	}
}

// Another model's reasoning state means nothing to a Chat Completions model:
// dropped, said so, and the turn goes out as text and calls.
func TestPlan_ResponseItemsDroppedOnChatCompletions(t *testing.T) {
	c := testClient(t, nil)
	r1, _ := parseResponsesResponseWith(Redact, json.RawMessage(solRound1))
	pl, ws, err := c.plan(ChatRequest{Model: "gpt-5.5", Messages: []Message{User("x"), r1.Assistant(),
		ToolResult("call_FaRG", "y")}}, false)
	if err != nil {
		t.Fatal(err)
	}
	if pl.req.Messages[1].ResponseItems != nil || len(pl.req.Messages[1].ToolCalls) != 1 {
		t.Errorf("coerced turn = %+v", pl.req.Messages[1])
	}
	if !strings.Contains(Warnings(ws), "ResponseItems were dropped") {
		t.Errorf("warnings = %s", Warnings(ws))
	}
	if r1.Assistant().ResponseItems == nil {
		t.Error("plan's drop reached the caller's response")
	}
}

func TestDecodeResponse_Endings(t *testing.T) {
	for name, tc := range map[string]struct {
		body, finish string
		sentinel     error
		apiErr       bool
	}{
		"truncated by the cap": {body: `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
			"output":[{"type":"message","content":[{"type":"output_text","text":"par"}]}]}`, finish: "length"},
		"other incomplete reason verbatim": {body: `{"status":"incomplete","incomplete_details":{"reason":"content_filter"},
			"output":[{"type":"message","content":[{"type":"output_text","text":"x"}]}]}`, finish: "content_filter"},
		"error under a 200":      {body: `{"error":{"message":"boom","code":"500"}}`, apiErr: true},
		"completed, no output":   {body: `{"status":"completed","output":[]}`, sentinel: ErrResponseShape},
		"background status":      {body: `{"status":"queued","output":[]}`, sentinel: ErrResponseShape},
		"not JSON":               {body: `<html>`, sentinel: ErrMalformedResponse},
		"unreplayable reasoning": {body: `{"status":"completed","output":[{"type":"reasoning","id":"rs"}]}`, sentinel: ErrResponseShape},
		"reasoning and no answer": {body: `{"status":"completed","output":[{"type":"reasoning","id":"rs","encrypted_content":"x"}]}`,
			sentinel: ErrResponseShape},
		// The cap ran out inside the thinking: an empty answer, cut, is the
		// same report Chat Completions gives, not an error.
		"cut while thinking": {body: `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
			"output":[{"type":"reasoning","id":"rs","encrypted_content":"x"}]}`, finish: "length"},
	} {
		resp, err := parseResponsesResponseWith(Redact, json.RawMessage(tc.body))
		var apiErr *APIError
		switch {
		case tc.apiErr:
			if !errors.As(err, &apiErr) {
				t.Errorf("%s: err %v, want *APIError", name, err)
			}
		case tc.sentinel != nil:
			if !errors.Is(err, tc.sentinel) {
				t.Errorf("%s: err %v, want %v", name, err, tc.sentinel)
			}
		case err != nil || resp.FinishReason != tc.finish:
			t.Errorf("%s: %+v, %v; want finish %q", name, resp, err, tc.finish)
		}
	}
}

// solStream is call 94fcdf8c as LiteLLM sent it before the gateway's catalog
// alias: a faked stream, so output_item.added already
// holds the complete arguments the deltas then repeat, cost on the body of
// response.completed, and a [DONE] the protocol does not define.
var solStream = []string{
	`data: {"type":"response.created","response":{"status":"in_progress","output":[]}}`,
	`data: {"type":"response.output_item.added","output_index":0,"item":{"arguments":"{\"city\":\"Bern\"}","call_id":"call_IVP6","name":"get_weather","type":"function_call","id":"fc_0dd5","status":"completed"}}`,
	`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"cit"}`,
	`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"y\":\"B"}`,
	`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"ern\"}"}`,
	`data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"city\":\"Bern\"}"}`,
	`data: {"type":"response.output_item.done","output_index":0,"item":{"arguments":"{\"city\":\"Bern\"}","call_id":"call_IVP6","name":"get_weather","type":"function_call","id":"fc_0dd5"}}`,
	`data: {"type":"response.completed","response":{"status":"completed","output":[{"arguments":"{\"city\":\"Bern\"}","call_id":"call_IVP6","name":"get_weather","type":"function_call","id":"fc_0dd5","status":"completed"}],` +
		`"usage":{"input_tokens":53,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":18,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":71,"cost":0.0003432}}}`,
	doneFrame,
}

func readResponsesSSE(t *testing.T, lines []string) (StreamResult, []streamEvent, error) {
	t.Helper()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var events []streamEvent
	res, err := readResponsesStream(strings.NewReader(strings.Join(lines, "\n\n")+"\n\n"), guard,
		streamBounds{idle: time.Hour}, func(ev streamEvent) { events = append(events, ev) }, Redact, time.Now, time.Now())
	return res, events, err
}

func TestReadResponsesStream_BufferedGatewayStream(t *testing.T) {
	res, events, err := readResponsesSSE(t, solStream)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != `{"city":"Bern"}` || res.FinishReason != "tool_calls" {
		t.Fatalf("result = %+v", res)
	}
	// The iterator's view: the name once, the arguments ONCE, from the deltas.
	var args strings.Builder
	var names []string
	for _, ev := range events {
		if ev.kind == evToolCall {
			args.WriteString(ev.toolCall.argumentsDelta())
			if ev.toolCall.Function.Name != "" {
				names = append(names, ev.toolCall.Function.Name)
			}
		}
	}
	if args.String() != `{"city":"Bern"}` || !reflect.DeepEqual(names, []string{"get_weather"}) {
		t.Errorf("events carried arguments %q, names %v", args.String(), names)
	}
	// Completion is response.completed; the gateway's [DONE] after it is
	// never read.
	if events[len(events)-1].kind != evFinish || res.Events != 8 || res.Done {
		t.Errorf("last event %+v, events %d, done %v", events[len(events)-1], res.Events, res.Done)
	}
	if len(res.ResponseItems) != 1 || Tokens(res.Usage.Input.Total) != 53 {
		t.Errorf("items %d, usage %+v", len(res.ResponseItems), res.Usage)
	}
}

func TestReadResponsesStream_ArgumentsFromDoneWithoutDeltas(t *testing.T) {
	_, events, err := readResponsesSSE(t, []string{
		`data: {"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","call_id":"c","name":"f"}}`,
		`data: {"type":"response.function_call_arguments.done","output_index":3,"arguments":"{}"}`,
		`data: {"type":"response.output_text.delta","delta":"ok"}`,
		`data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]}}`,
	})
	if err != nil {
		t.Fatal(err)
	}
	var args string
	for _, ev := range events {
		if ev.kind == evToolCall {
			args += ev.toolCall.argumentsDelta()
			if ev.toolCall.Index != 0 {
				t.Errorf("call index %d, want the ordinal 0", ev.toolCall.Index)
			}
		}
	}
	if args != "{}" || events[2].kind != evContent || events[2].text != "ok" {
		t.Errorf("args %q, events %+v", args, events)
	}
}

func TestReadResponsesStream_Failures(t *testing.T) {
	for name, tc := range map[string]struct {
		lines    []string
		sentinel error
		apiErr   bool
		partial  string
	}{
		"cut before completed": {lines: []string{`data: {"type":"response.output_text.delta","delta":"par"}`, doneFrame},
			sentinel: ErrMalformedResponse, partial: "par"},
		"response.failed": {lines: []string{`data: {"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"boom"}}}`},
			apiErr: true},
		"failed without error": {lines: []string{`data: {"type":"response.failed","response":{"status":"failed"}}`},
			sentinel: ErrResponseShape},
		"error event":   {lines: []string{`data: {"type":"error","code":"rate_limit","message":"slow down"}`}, apiErr: true},
		"gateway frame": {lines: []string{`data: {"error":{"message":"upstream gone","code":"502"}}`}, apiErr: true},
		"completed but empty": {lines: []string{`data: {"type":"response.completed","response":{"status":"completed","output":[]}}`},
			sentinel: ErrResponseShape},
	} {
		res, _, err := readResponsesSSE(t, tc.lines)
		var apiErr *APIError
		if tc.apiErr && !errors.As(err, &apiErr) {
			t.Errorf("%s: err %v, want *APIError", name, err)
		}
		if tc.sentinel != nil && !errors.Is(err, tc.sentinel) {
			t.Errorf("%s: err %v, want %v", name, err, tc.sentinel)
		}
		if res.Content != tc.partial {
			t.Errorf("%s: partial %q, want %q", name, res.Content, tc.partial)
		}
	}

	// A call cut mid-arguments comes back as far as it got, as readStream
	// returns one: Collect promises the paid-for partial.
	res, _, err := readResponsesSSE(t, []string{
		`data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c","name":"f"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"a\":"}`,
	})
	if !errors.Is(err, ErrMalformedResponse) || len(res.ToolCalls) != 1 ||
		res.ToolCalls[0] != (ToolCall{ID: "c", Type: "function", Name: "f", Arguments: `{"a":`}) {
		t.Errorf("cut call: %+v, %v", res.ToolCalls, err)
	}
}

// End to end through the gateway route a deployment declares: the request
// leaves for /responses under the alias, the non-streamed call is priced from
// the header, and the stream from usage.cost on response.completed.
func TestFromEnv_SolOverResponsesThroughTheGateway(t *testing.T) {
	env := gatewayEnv("", "gpt-6.1-sol=ai-gateway-gpt-6.1-sol")
	srv := jsonServer(t, http.StatusOK, solRound1, reply{header: map[string]string{
		"x-litellm-response-cost":           "0.0008808",
		"x-litellm-response-cost-input":     "0.0002328",
		"x-litellm-response-cost-output":    "0.000648",
		"x-litellm-response-cost-reasoning": "0.000408",
	}})
	env["LLMWIRE_LITELLM_BASE_URL"] = srv.URL
	c, err := FromEnv("gpt-6.1-sol", Config{Lookup: mapLookup(env)})
	if err != nil {
		t.Fatal(err)
	}
	resp, _, err := c.Chat(context.Background(), solRequest(User("weather?")))
	if err != nil {
		t.Fatal(err)
	}
	var sent map[string]any
	_ = json.Unmarshal(srv.lastBody(), &sent)
	if srv.last().URL.Path != "/responses" || sent["model"] != "ai-gateway-gpt-6.1-sol" {
		t.Errorf("sent to %s as %v", srv.last().URL.Path, sent["model"])
	}
	if resp.Usage.Cost.Provenance != Reported || resp.Usage.Cost.NanoUSD != 880_800 {
		t.Errorf("cost = %+v", resp.Usage.Cost)
	}
	// Measured: the reasoning lane sits INSIDE output (34 of 54 tokens at the
	// same rate); input + output is the total.
	g := resp.Gateway.Cost
	if *g.InputNanoUSD+*g.OutputNanoUSD != 880_800 || *g.ReasoningNanoUSD != 408_000 {
		t.Errorf("split = %+v", g)
	}

	sse := sseServer(t, solStream)
	env["LLMWIRE_LITELLM_BASE_URL"] = sse.URL
	c, err = FromEnv("gpt-6.1-sol", Config{Lookup: mapLookup(env)})
	if err != nil {
		t.Fatal(err)
	}
	s, _, err := c.ChatStream(context.Background(), solRequest(User("weather?")))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	res, err := s.Collect(nil)
	if err != nil {
		t.Fatal(err)
	}
	if sse.last().URL.Path != "/responses" || res.Usage.Cost.Provenance != Reported || res.Usage.Cost.NanoUSD != 343_200 {
		t.Errorf("stream path %s, cost %+v", sse.last().URL.Path, res.Usage.Cost)
	}
	if len(res.Assistant().ResponseItems) != 1 {
		t.Errorf("stream Assistant() lost the items: %+v", res.Assistant())
	}
}

func TestProfile_ResponsesValidation(t *testing.T) {
	const head = "profiles:\n  - id: m\n    wire_model_id: m\n    verified: measured\n"
	ok := head + "    wire_api: responses\n    max_tokens_param: max_output_tokens\n" +
		"    reasoning: {supported: true, enabled_by_default: true, control: effort, effort_values: [low]}\n" +
		"    streaming: {supported: true}\n"
	if _, err := NewRegistry([]byte(ok)); err != nil {
		t.Fatalf("a valid responses profile: %v", err)
	}
	for name, doc := range map[string]string{
		"chat cap spelling":      strings.Replace(ok, "max_output_tokens", "max_completion_tokens", 1),
		"stream_options":         strings.Replace(ok, "{supported: true}\n", "{supported: true, accepts_stream_options: true}\n", 1),
		"toggle control":         strings.Replace(ok, "control: effort, effort_values: [low]", "control: toggle_object, can_be_disabled: true", 1),
		"inline recovery":        ok + "    tools: {supported: true, tool_choice_values: [auto], recover_inline_markup: true}\n",
		"reasoning stream field": strings.Replace(ok, "effort_values: [low]", "effort_values: [low], stream_field: reasoning", 1),
		"unknown protocol":       strings.Replace(ok, "wire_api: responses", "wire_api: grpc", 1),
		"output cap on chat":     head + "    max_tokens_param: max_output_tokens\n    streaming: {supported: true}\n",
		"embeddings":             "profiles:\n  - id: e\n    endpoint: embeddings\n    wire_api: responses\n    verified: measured\n    embedding: {default_dimensions: 8}\n",
	} {
		if _, err := NewRegistry([]byte(doc)); err == nil {
			t.Errorf("%s: loaded", name)
		}
	}
	if err := validateBaseURL("https://gw.example/v1/responses", false); err == nil {
		t.Error("a base URL ending in /responses was accepted")
	}
}
