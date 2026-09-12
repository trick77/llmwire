package llmwire

import (
	"encoding/json"
	"testing"
)

// Bodies are asserted by DECODING them, never against golden bytes. A golden
// comparison turns a harmless added key into a failing diff, and the fix someone
// reaches for is to stop building the body as a map — which is the one decision
// ExtraBody's "caller keys win" rule depends on.

func renderFor(t *testing.T, c *Client, req ChatRequest, stream bool) map[string]any {
	t.Helper()
	pl, _, err := c.plan(req, stream)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	raw, err := renderChatBody(pl)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("rendered body is not JSON: %v\n%s", err, raw)
	}
	return body
}

func absent(t *testing.T, body map[string]any, key string) {
	t.Helper()
	if v, ok := body[key]; ok {
		t.Errorf("%s should be absent, got %v", key, v)
	}
}

// THE test this whole phase exists for. glm-5.3-flash accepts
// max_completion_tokens and silently ignores it: a cap of 16 returned 481
// completion tokens with finish_reason "stop" and no error anywhere. The cap is
// written once by the caller and spelled by the profile, and exactly one spelling
// is ever sent.
func TestRender_OutputCapUsesTheProfilesSpelling(t *testing.T) {
	c := testClient(t, Default())
	cap16 := 16
	for _, tc := range []struct{ model, want, mustNotSend string }{
		{"glm-5.3-flash", "max_tokens", "max_completion_tokens"},
		{"mimo-v2.5-pro", "max_completion_tokens", "max_tokens"},
		{"mimo-v2.5", "max_completion_tokens", "max_tokens"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			body := renderFor(t, c, ChatRequest{
				Model:     tc.model,
				Messages:  []Message{User("hi")},
				MaxTokens: &cap16,
			}, false)
			if body[tc.want] != float64(16) {
				t.Errorf("%s = %v, want 16", tc.want, body[tc.want])
			}
			absent(t, body, tc.mustNotSend)
		})
	}
}

// The wire id, not the registry id: behind a gateway they are different strings
// and the proxy answers only to its own alias.
func TestRender_ModelIsTheWireID(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: base-model
    wire_model_id: vendor/base-model
    max_tokens_param: max_tokens
    verified: measured
    streaming: {supported: true, accepts_stream_options: true}
`)
	body := renderFor(t, testClient(t, reg), ChatRequest{
		Model:    "base-model",
		Messages: []Message{User("hi")},
	}, false)
	if body["model"] != "vendor/base-model" {
		t.Errorf("model = %v, want the wire id", body["model"])
	}
}

// Sampling is sent from the profile's recommended value when the caller says
// nothing, because omitting it on this endpoint does not mean "the model's
// default" — its fallback sits materially below the tuned operating point.
func TestRender_SamplingDefaultsAndOverrides(t *testing.T) {
	c := testClient(t, Default())

	body := renderFor(t, c, ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}}, false)
	if body["temperature"] != 1.0 {
		t.Errorf("temperature = %v, want 1.0", body["temperature"])
	}
	if body["top_p"] != 0.95 {
		t.Errorf("top_p = %v, want 0.95", body["top_p"])
	}

	body = renderFor(t, c, ChatRequest{
		Model:       "glm-5.3-flash",
		Messages:    []Message{User("hi")},
		Temperature: f64(0.2),
	}, false)
	if body["temperature"] != 0.2 {
		t.Errorf("temperature = %v, want the caller's 0.2", body["temperature"])
	}

	// A model with no recommendation gets no key at all.
	body = renderFor(t, c, ChatRequest{Model: "mimo-v2.5-pro", Messages: []Message{User("hi")}}, false)
	absent(t, body, "temperature")
	absent(t, body, "top_p")
}

// A zero temperature is a real request, not an unset field. This is why the body
// is built from pointers rather than from a struct's zero values.
func TestRender_ZeroTemperatureIsSent(t *testing.T) {
	body := renderFor(t, testClient(t, Default()), ChatRequest{
		Model:       "mimo-v2.5",
		Messages:    []Message{User("hi")},
		Temperature: f64(0),
	}, false)
	v, ok := body["temperature"]
	if !ok || v != float64(0) {
		t.Errorf("temperature = %v (present=%v), want 0 to be sent", v, ok)
	}
}

// Each model's reasoning knob takes a different spelling, which is the whole
// reason Control is a profile field rather than an assumption.
func TestRender_ReasoningSpellingPerControl(t *testing.T) {
	c := testClient(t, Default())

	body := renderFor(t, c, ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningEffort("high"),
	}, false)
	if body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v", body["reasoning_effort"])
	}
	absent(t, body, "thinking")

	body = renderFor(t, c, ChatRequest{
		Model:     "mimo-v2.5",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningOff(),
	}, false)
	thinking, ok := body["thinking"].(map[string]any)
	if !ok || thinking["type"] != "disabled" {
		t.Errorf("thinking = %v, want {type: disabled}", body["thinking"])
	}
	absent(t, body, "reasoning_effort")

	// Nothing said, nothing sent: every reasoning model here is on by default,
	// so restating the default would add a key for no effect.
	body = renderFor(t, c, ChatRequest{Model: "mimo-v2.5", Messages: []Message{User("hi")}}, false)
	absent(t, body, "thinking")
	absent(t, body, "reasoning_effort")
}

// A budget model sends the key its profile names. No model in the registry uses
// this control, so the profile is synthetic — which is the point: adding one must
// be data, not code.
func TestRender_BudgetUsesTheProfilesParameterName(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: budgeted
    wire_model_id: budgeted
    max_tokens_param: max_tokens
    verified: source-derived
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: budget_tokens
      budget_param: thinking_budget
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)

	body := renderFor(t, c, ChatRequest{
		Model:     "budgeted",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningBudget(2048),
	}, false)
	if body["thinking_budget"] != float64(2048) {
		t.Errorf("thinking_budget = %v, want 2048", body["thinking_budget"])
	}

	body = renderFor(t, c, ChatRequest{
		Model:     "budgeted",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningOff(),
	}, false)
	if body["thinking_budget"] != float64(0) {
		t.Errorf("a budget model switched off should send 0, got %v", body["thinking_budget"])
	}
}

// A demoted refusal must leave no trace of the knob in the body. This is the
// assertion that would have caught a bare refuse() whose return was ignored.
func TestRender_BestEffortSendsNoRefusedKnob(t *testing.T) {
	c := testClient(t, Default())
	body := renderFor(t, c, ChatRequest{
		Model:      "glm-5.3-flash",
		Messages:   []Message{User("hi")},
		Reasoning:  ReasoningOff(),
		BestEffort: true,
	}, false)
	absent(t, body, "reasoning_effort")
	absent(t, body, "thinking")
	// The sampling values still go, since the request is otherwise honoured.
	if body["temperature"] != 1.0 {
		t.Errorf("temperature = %v, want the recommended 1.0", body["temperature"])
	}
}

// stream and stream_options.
func TestRender_StreamKeys(t *testing.T) {
	c := testClient(t, Default())

	body := renderFor(t, c, ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}}, false)
	absent(t, body, "stream")
	absent(t, body, "stream_options")

	// RawStream does not inject stream:true, so the rendered body must carry it.
	body = renderFor(t, c, ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}}, true)
	if body["stream"] != true {
		t.Errorf("stream = %v, want true", body["stream"])
	}
	// This endpoint reports usage unasked, and an endpoint that rejects the
	// parameter must never receive it.
	absent(t, body, "stream_options")

	reg := registryFrom(t, `profiles:
  - id: needs
    wire_model_id: needs
    max_tokens_param: max_tokens
    verified: source-derived
    streaming: {supported: true, needs_include_usage: true, accepts_stream_options: true}
`)
	body = renderFor(t, testClient(t, reg), ChatRequest{Model: "needs", Messages: []Message{User("hi")}}, true)
	opts, ok := body["stream_options"].(map[string]any)
	if !ok || opts["include_usage"] != true {
		t.Errorf("stream_options = %v, want {include_usage: true}", body["stream_options"])
	}
}

// Messages: the plain string form unless the turn really needs parts.
func TestRender_MessageShapes(t *testing.T) {
	c := testClient(t, Default())
	body := renderFor(t, c, ChatRequest{
		Model: "mimo-v2.5",
		Messages: []Message{
			System("be brief"),
			{Role: RoleUser, Parts: []Part{
				{Kind: PartText, Text: "what is this"},
				{Kind: PartImage, URL: "data:image/png;base64,AAAA"},
			}},
			{Role: RoleAssistant, ReasoningContent: "thought", ToolCalls: []ToolCall{
				{ID: "call_1", Name: "search", Arguments: `{"q":"x"}`},
			}},
			ToolResult("call_1", "found"),
		},
	}, false)

	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) != 4 {
		t.Fatalf("messages = %v", body["messages"])
	}

	sys := msgs[0].(map[string]any)
	if sys["content"] != "be brief" {
		t.Errorf("a text-only turn should send a plain string, got %T %v", sys["content"], sys["content"])
	}

	parts, ok := msgs[1].(map[string]any)["content"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("multimodal content = %v", msgs[1])
	}
	img := parts[1].(map[string]any)
	if img["type"] != "image_url" {
		t.Errorf("image part type = %v", img["type"])
	}
	// A nested object, not a bare string: this exact shape is what 404s on
	// mimo-v2.5-pro and is accepted here.
	nested, ok := img["image_url"].(map[string]any)
	if !ok || nested["url"] != "data:image/png;base64,AAAA" {
		t.Errorf("image_url = %v, want a nested {url: ...}", img["image_url"])
	}

	asst := msgs[2].(map[string]any)
	if asst["reasoning_content"] != "thought" {
		t.Errorf("reasoning_content = %v", asst["reasoning_content"])
	}
	calls, ok := asst["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %v", asst["tool_calls"])
	}
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "search" || fn["arguments"] != `{"q":"x"}` {
		t.Errorf("function = %v; arguments must stay a JSON STRING", fn)
	}
	if calls[0].(map[string]any)["type"] != "function" {
		t.Errorf("tool call type = %v, want function", calls[0])
	}
	// An assistant turn with only tool calls still carries a content key: the
	// widest-accepted spelling, and marked unmeasured in render.go.
	if _, ok := asst["content"]; !ok {
		t.Error("an assistant turn with only tool calls sent no content key")
	}

	tool := msgs[3].(map[string]any)
	if tool["tool_call_id"] != "call_1" {
		t.Errorf("tool_call_id = %v", tool["tool_call_id"])
	}
	// reasoning_content is sent only when non-empty: MEASURED as not required on
	// a tool_calls history, so an empty one is never sent to find out.
	absent(t, tool, "reasoning_content")
}

// Tools, tool_choice and response_format.
func TestRender_ToolsAndFormats(t *testing.T) {
	c := testClient(t, Default())
	body := renderFor(t, c, ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
		Tools: []Tool{{
			Name:        "search",
			Description: "look it up",
			Parameters:  map[string]any{"type": "object"},
			Strict:      true,
		}},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto},
		ResponseFormat: ResponseFormat{
			Kind:   FormatJSONSchema,
			Name:   "answer",
			Schema: map[string]any{"type": "object"},
			Strict: true,
		},
		Stop: []string{"\n\n"},
	}, false)

	tools := body["tools"].([]any)
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type = %v", tool["type"])
	}
	fn := tool["function"].(map[string]any)
	if fn["name"] != "search" || fn["description"] != "look it up" || fn["strict"] != true {
		t.Errorf("function = %v", fn)
	}
	if body["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v, want the bare string auto", body["tool_choice"])
	}
	rf := body["response_format"].(map[string]any)
	if rf["type"] != "json_schema" {
		t.Errorf("response_format type = %v", rf["type"])
	}
	schema := rf["json_schema"].(map[string]any)
	if schema["name"] != "answer" || schema["strict"] != true || schema["schema"] == nil {
		t.Errorf("json_schema = %v", schema)
	}
	stop := body["stop"].([]any)
	if len(stop) != 1 || stop[0] != "\n\n" {
		t.Errorf("stop = %v", body["stop"])
	}
}

// A forced function choice is an object, not a string. Sent here through a
// synthetic profile that accepts it, since no registry model does.
func TestRender_ForcedFunctionChoice(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: forcer
    wire_model_id: forcer
    max_tokens_param: max_tokens
    verified: source-derived
    tools:
      supported: true
      format: native
      tool_choice_values: [auto, required, function]
      supports_forced_choice: true
    streaming: {supported: true, accepts_stream_options: true}
`)
	body := renderFor(t, testClient(t, reg), ChatRequest{
		Model:      "forcer",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceFunction, Name: "search"},
	}, false)
	tc, ok := body["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "function" {
		t.Fatalf("tool_choice = %v", body["tool_choice"])
	}
	if tc["function"].(map[string]any)["name"] != "search" {
		t.Errorf("forced function = %v", tc["function"])
	}
}

// The json_schema downgrade reaches the body as json_object, decided once in plan
// and merely emitted here.
func TestRender_DowngradedFormatIsEmittedNotRederived(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: jsonobj
    wire_model_id: jsonobj
    max_tokens_param: max_tokens
    verified: measured
    output: {json_object: true}
    streaming: {supported: true, accepts_stream_options: true}
`)
	body := renderFor(t, testClient(t, reg), ChatRequest{
		Model:    "jsonobj",
		Messages: []Message{User("hi")},
		ResponseFormat: ResponseFormat{
			Kind:   FormatJSONSchema,
			Name:   "answer",
			Schema: map[string]any{"type": "object"},
			Strict: true,
		},
	}, false)
	rf := body["response_format"].(map[string]any)
	if rf["type"] != "json_object" {
		t.Errorf("response_format = %v, want the downgrade", rf)
	}
	if _, ok := rf["json_schema"]; ok {
		t.Error("a downgraded format still carried its schema")
	}
}

// ExtraBody wins, deliberately: its whole purpose is reaching a parameter this
// package does not model, and half-honouring that would be worse than not
// offering it.
func TestRender_ExtraBodyOverwritesGeneratedKeys(t *testing.T) {
	body := renderFor(t, testClient(t, Default()), ChatRequest{
		Model:       "glm-5.3-flash",
		Messages:    []Message{User("hi")},
		Temperature: f64(0.2),
		ExtraBody: map[string]any{
			"temperature":   0.9,
			"do_sample":     true,
			"unknown_thing": "passed through",
		},
	}, false)
	if body["temperature"] != 0.9 {
		t.Errorf("temperature = %v, want ExtraBody's 0.9", body["temperature"])
	}
	if body["do_sample"] != true || body["unknown_thing"] != "passed through" {
		t.Errorf("ExtraBody keys missing from the body: %v", body)
	}
}

// Embeddings bodies: the wire id, the batch, and dimensions only when asked.
func TestRender_EmbedBody(t *testing.T) {
	p := mustLookup(t, Default(), "text-embedding-3-small")
	raw, err := renderEmbedBody(p, []string{"a", "b"}, nil)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "text-embedding-3-small" {
		t.Errorf("model = %v", body["model"])
	}
	if in := body["input"].([]any); len(in) != 2 || in[0] != "a" {
		t.Errorf("input = %v", body["input"])
	}
	absent(t, body, "dimensions")

	dims := 256
	raw, _ = renderEmbedBody(p, []string{"a"}, &dims)
	body = nil
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if body["dimensions"] != float64(256) {
		t.Errorf("dimensions = %v", body["dimensions"])
	}
}
