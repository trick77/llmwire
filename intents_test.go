package llmwire

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Intents: a caller says "as shallow as this model goes" or "fast but not
// shallow", and the profile decides what that is. Swapping the model is then a
// config change, never an edit to every call site that named a level.

// intentsDoc is one model per resolution path, so each test names the path it
// pins rather than borrowing a shipped model that may move.
const intentsDoc = `profiles:
  - id: offable
    display_name: Offable
    wire_model_id: offable
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: effort
      effort_values: [none, low, medium, high]
      balanced: medium
      overhead: {low: 100, medium: 300}
    limits: {context: 100000, max_output: 4000}
  - id: always
    display_name: Always
    wire_model_id: always
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      control: effort
      effort_values: [low, high, max]
      default_effort: max
  - id: toggle
    display_name: Toggle
    wire_model_id: toggle
    max_tokens_param: max_completion_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: toggle_object
  - id: budget
    display_name: Budget
    wire_model_id: budget
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      control: budget_tokens
      budget_param: thinking_budget
      min_budget: 256
  - id: plain
    display_name: Plain
    wire_model_id: plain
    max_tokens_param: max_tokens
`

func intentsClient(t *testing.T) *Client {
	t.Helper()
	return testClient(t, registryFrom(t, intentsDoc))
}

func TestReasoningMinimal_ResolvesPerProfile(t *testing.T) {
	c := intentsClient(t)
	for _, tc := range []struct {
		model string
		want  ReasoningRequest
		label string
	}{
		// Switchable: off is the shallowest there is.
		{"offable", ReasoningOff(), "off"},
		{"toggle", ReasoningOff(), "off"},
		// Cannot be switched off: the lowest level, which is why effort_values
		// must be listed shallowest first.
		{"always", ReasoningEffort("low"), "low"},
		// Budget control: the profile's own floor, never a guessed number.
		{"budget", ReasoningBudget(256), "budget:256"},
		// No reasoning at all: already minimal, nothing to send, nothing to warn.
		{"plain", nil, ""},
	} {
		pl, warnings, err := c.plan(ChatRequest{Model: tc.model, Messages: []Message{User("hi")}, Reasoning: ReasoningMinimal()}, false)
		if err != nil {
			t.Fatalf("%s: %v", tc.model, err)
		}
		if len(warnings) != 0 {
			t.Errorf("%s: warnings %v", tc.model, warnings)
		}
		if pl.req.Reasoning != tc.want {
			t.Errorf("%s: resolved to %#v, want %#v", tc.model, pl.req.Reasoning, tc.want)
		}
		if got := reasoningLabel(pl.req.Reasoning); got != tc.label {
			t.Errorf("%s: label %q, want %q", tc.model, got, tc.label)
		}
	}
}

func TestReasoningBalanced_ResolvesPerProfile(t *testing.T) {
	c := intentsClient(t)
	for _, tc := range []struct {
		model string
		want  ReasoningRequest
	}{
		{"offable", ReasoningEffort("medium")},
		// No balanced level declared: the model's own default, so nothing goes
		// on the wire.
		{"always", nil},
		{"toggle", nil},
		{"plain", nil},
	} {
		pl, warnings, err := c.plan(ChatRequest{Model: tc.model, Messages: []Message{User("hi")}, Reasoning: ReasoningBalanced()}, false)
		if err != nil {
			t.Fatalf("%s: %v", tc.model, err)
		}
		if len(warnings) != 0 {
			t.Errorf("%s: warnings %v", tc.model, warnings)
		}
		if pl.req.Reasoning != tc.want {
			t.Errorf("%s: resolved to %#v, want %#v", tc.model, pl.req.Reasoning, tc.want)
		}
	}
}

// An intent renders exactly as the concrete request it resolves to.
func TestReasoningIntents_RenderAsTheirResolution(t *testing.T) {
	c := intentsClient(t)
	for _, tc := range []struct {
		model            string
		intent, concrete ReasoningRequest
	}{
		{"offable", ReasoningMinimal(), ReasoningOff()},
		{"offable", ReasoningBalanced(), ReasoningEffort("medium")},
		{"toggle", ReasoningMinimal(), ReasoningOff()},
		{"always", ReasoningMinimal(), ReasoningEffort("low")},
		{"budget", ReasoningMinimal(), ReasoningBudget(256)},
	} {
		got := renderFor(t, c, ChatRequest{Model: tc.model, Messages: []Message{User("hi")}, Reasoning: tc.intent}, false)
		want := renderFor(t, c, ChatRequest{Model: tc.model, Messages: []Message{User("hi")}, Reasoning: tc.concrete}, false)
		if mustJSON(t, got) != mustJSON(t, want) {
			t.Errorf("%s: intent rendered %v, concrete %v", tc.model, got, want)
		}
	}
}

// A budget model that cannot be switched off and states no floor: a document
// from before min_budget still loads, and the intent is refused by name rather
// than sent with a guessed budget.
func TestReasoningMinimal_BudgetModelWithoutAFloor(t *testing.T) {
	c := testClient(t, registryFrom(t, chatHead+
		"    reasoning: {supported: true, control: budget_tokens, budget_param: b, enabled_by_default: true}\n"))
	req := ChatRequest{Model: "m", Messages: []Message{User("hi")}, Reasoning: ReasoningMinimal()}
	if _, err := c.Validate(req); err == nil || !strings.Contains(err.Error(), "min_budget") {
		t.Errorf("err = %v, want a refusal naming min_budget", err)
	}
	req.BestEffort = true
	pl, warnings, err := c.plan(req, false)
	if err != nil || pl.req.Reasoning != nil || len(warnings) != 1 {
		t.Errorf("BestEffort: reasoning %v, warnings %v, err %v; want the knob dropped with a warning", pl, warnings, err)
	}
}

// The shipped profiles: the values a caller gets without naming a level.
func TestReasoningIntents_ShippedProfiles(t *testing.T) {
	c := testClient(t, Default())
	for _, tc := range []struct{ model, minimal, balanced string }{
		// Cannot be disabled (measured 400/1210): the lowest accepted level.
		{"glm-5.3-flash", "low", "high"},
		{"mimo-v2.6-pro", "off", "medium"},
		{"mimo-v2.6-flash", "off", "medium"},
		{"mimo-v2.5-pro", "off", "medium"},
		{"mimo-v2.5", "off", "medium"},
		{"gpt-5.5", "off", "medium"},
	} {
		for intent, want := range map[string]string{"minimal": tc.minimal, "balanced": tc.balanced} {
			r := ReasoningMinimal()
			if intent == "balanced" {
				r = ReasoningBalanced()
			}
			pl := mustPlan(t, c, ChatRequest{Model: tc.model, Messages: []Message{User("hi")}, Reasoning: r}, false)
			if got := reasoningLabel(pl.req.Reasoning); got != want {
				t.Errorf("%s %s: resolved to %q, want %q", tc.model, intent, got, want)
			}
		}
	}
}

// The resolved request is what a caller persists and logs, so it rides back on
// both results.
func TestReasoningSent_OnChatAndStream(t *testing.T) {
	srv, _ := jsonServer(t, 200, goodCompletion)
	resp, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model: "glm-5.3-flash", Messages: []Message{User("hi")}, Reasoning: ReasoningMinimal(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ReasoningSent != "low" {
		t.Errorf("Chat ReasoningSent = %q, want low", resp.ReasoningSent)
	}

	srv2 := sseServer(t, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	s, _, err := chatClient(t, srv2).ChatStream(context.Background(), ChatRequest{
		Model: "mimo-v2.6-pro", Messages: []Message{User("hi")}, Reasoning: ReasoningMinimal(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	res, err := s.Collect(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.ReasoningSent != "off" {
		t.Errorf("stream ReasoningSent = %q, want off", res.ReasoningSent)
	}

	// Nothing sent reads as empty: the model ran at its own default.
	resp, _, err = chatClient(t, srv).Chat(context.Background(), ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ReasoningSent != "" {
		t.Errorf("no knob: ReasoningSent = %q, want empty", resp.ReasoningSent)
	}
}

// --- answer budget ---------------------------------------------------------

func TestMaxAnswerTokens_AddsTheResolvedOverhead(t *testing.T) {
	c := intentsClient(t)
	for _, tc := range []struct {
		name  string
		model string
		r     ReasoningRequest
		want  int
	}{
		{"measured level", "offable", ReasoningEffort("medium"), 500 + 300},
		{"balanced resolves to medium", "offable", ReasoningBalanced(), 500 + 300},
		{"level without a measurement takes the package default", "offable", ReasoningEffort("high"), 500 + DefaultReasoningOverhead},
		{"off costs nothing", "offable", ReasoningOff(), 500},
		{"minimal resolves to off", "offable", ReasoningMinimal(), 500},
		{"effort none is off", "offable", ReasoningEffort("none"), 500},
		{"no knob on a thinking model", "offable", nil, 500 + DefaultReasoningOverhead},
		{"a budget is its own overhead", "budget", ReasoningBudget(700), 500 + 700},
		{"no reasoning at all", "plain", nil, 500},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pl, _, err := c.plan(ChatRequest{Model: tc.model, Messages: []Message{User("hi")}, Reasoning: tc.r, MaxAnswerTokens: ptr(500)}, false)
			if err != nil {
				t.Fatal(err)
			}
			if pl.req.MaxTokens == nil || *pl.req.MaxTokens != tc.want {
				t.Errorf("wire cap = %v, want %d", pl.req.MaxTokens, tc.want)
			}
			body := renderFor(t, c, ChatRequest{Model: tc.model, Messages: []Message{User("hi")}, Reasoning: tc.r, MaxAnswerTokens: ptr(500)}, false)
			if body[pl.capParam] != float64(tc.want) {
				t.Errorf("rendered %s = %v, want %d", pl.capParam, body[pl.capParam], tc.want)
			}
		})
	}
}

func TestMaxAnswerTokens_DefaultEffortKeysTheNoKnobCase(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: d
    display_name: D
    wire_model_id: d
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      control: effort
      effort_values: [low, high]
      default_effort: high
      overhead: {high: 900}
`)
	pl := mustPlan(t, testClient(t, reg), ChatRequest{Model: "d", Messages: []Message{User("hi")}, MaxAnswerTokens: ptr(100)}, false)
	if *pl.req.MaxTokens != 1000 {
		t.Errorf("wire cap = %d, want 100 + the default level's 900", *pl.req.MaxTokens)
	}
}

func TestMaxAnswerTokens_ClampsToMaxOutput(t *testing.T) {
	c := intentsClient(t)
	pl, warnings, err := c.plan(ChatRequest{Model: "offable", Messages: []Message{User("hi")},
		Reasoning: ReasoningEffort("medium"), MaxAnswerTokens: ptr(3900)}, false)
	if err != nil {
		t.Fatal(err)
	}
	if *pl.req.MaxTokens != 4000 {
		t.Errorf("wire cap = %d, want the 4000 max_output", *pl.req.MaxTokens)
	}
	if len(warnings) != 0 {
		t.Errorf("a clamp that still fits the answer warned: %v", warnings)
	}
	// An answer the model cannot produce at all is the caller's mistake to hear
	// about, as with MaxTokens.
	_, warnings, err = c.plan(ChatRequest{Model: "offable", Messages: []Message{User("hi")}, MaxAnswerTokens: ptr(5000)}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 1 || warnings[0].Feature != "max_answer_tokens" {
		t.Errorf("warnings = %v, want one on max_answer_tokens", warnings)
	}
}

func TestMaxAnswerTokens_StructuralRejections(t *testing.T) {
	c := intentsClient(t)
	for _, tc := range []struct {
		name string
		req  ChatRequest
		want string
	}{
		{"both caps", ChatRequest{MaxTokens: ptr(10), MaxAnswerTokens: ptr(10)}, "not both"},
		{"zero", ChatRequest{MaxAnswerTokens: ptr(0)}, "not positive"},
		{"negative", ChatRequest{MaxAnswerTokens: ptr(-5)}, "not positive"},
	} {
		tc.req.Model, tc.req.Messages, tc.req.BestEffort = "offable", []Message{User("hi")}, true
		_, err := c.Validate(tc.req)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want %q", tc.name, err, tc.want)
		}
	}
}

// A budget endpoint refuses a completion cap at or below the budget, so the
// clamp to max_output must never cut into it.
func TestMaxAnswerTokens_ClampNeverCutsIntoABudget(t *testing.T) {
	c := testClient(t, registryFrom(t, `profiles:
  - id: b
    display_name: B
    wire_model_id: b
    max_tokens_param: max_tokens
    reasoning: {supported: true, enabled_by_default: true, can_be_disabled: true, control: budget_tokens, budget_param: thinking_budget}
    limits: {context: 100000, max_output: 1000}
`))
	req := func(budget int, bestEffort bool) ChatRequest {
		return ChatRequest{Model: "b", Messages: []Message{User("hi")}, Reasoning: ReasoningBudget(budget),
			MaxAnswerTokens: ptr(500), BestEffort: bestEffort}
	}

	// Room left, but less than asked: clamped above the budget, and said so.
	pl, warnings, err := c.plan(req(700, false), false)
	if err != nil {
		t.Fatal(err)
	}
	if *pl.req.MaxTokens != 1000 {
		t.Errorf("wire cap = %d, want max_output 1000", *pl.req.MaxTokens)
	}
	if len(warnings) != 1 || warnings[0].Feature != "max_answer_tokens" || !strings.Contains(warnings[0].Details, "300") {
		t.Errorf("warnings = %v, want one saying the answer gets 300", warnings)
	}

	// No room at all: any cap under max_output is at or below the budget.
	if _, err := c.Validate(req(1000, false)); err == nil || !strings.Contains(err.Error(), "no room") {
		t.Errorf("err = %v, want a refusal saying the budget leaves no room", err)
	}
	pl, warnings, err = c.plan(req(1000, true), false)
	if err != nil {
		t.Fatal(err)
	}
	if pl.req.Reasoning != nil || *pl.req.MaxTokens != 1000 || len(warnings) == 0 {
		t.Errorf("BestEffort: reasoning %v cap %d warnings %v; want the budget dropped and the cap at max_output",
			pl.req.Reasoning, *pl.req.MaxTokens, warnings)
	}
}

// A budget dropped for leaving no room is dropped before anything reads
// whether the request thinks: the sampling warning must describe the request
// that is sent, not the one that was asked for.
func TestMaxAnswerTokens_BudgetDropPrecedesTheThinkingChecks(t *testing.T) {
	c := testClient(t, registryFrom(t, `profiles:
  - id: b
    display_name: B
    wire_model_id: b
    max_tokens_param: max_tokens
    reasoning: {supported: true, enabled_by_default: false, can_be_disabled: true, control: budget_tokens, budget_param: thinking_budget}
    temperature: {supported: true, inert_while_reasoning: true}
    limits: {context: 200000, max_output: 16384}
`))
	pl, warnings, err := c.plan(ChatRequest{Model: "b", Messages: []Message{User("hi")},
		Reasoning: ReasoningBudget(20000), MaxAnswerTokens: ptr(500), Temperature: f64(0.3), BestEffort: true}, false)
	if err != nil {
		t.Fatal(err)
	}
	if pl.req.Reasoning != nil {
		t.Errorf("reasoning = %#v, want the budget dropped", pl.req.Reasoning)
	}
	if pl.req.Temperature == nil || *pl.req.Temperature != 0.3 {
		t.Errorf("temperature = %v, want 0.3 sent", pl.req.Temperature)
	}
	var dropped bool
	for _, w := range warnings {
		if w.Feature == "temperature" {
			t.Errorf("warned %v, but the request sent does not think", w)
		}
		if w.Feature == "reasoning" && strings.Contains(w.Details, "no room") {
			dropped = true
		}
	}
	if !dropped {
		t.Errorf("warnings = %v, want the budget drop named", warnings)
	}
	if *pl.req.MaxTokens != 500 {
		t.Errorf("wire cap = %d, want 500: thinking is off by default once the budget is gone", *pl.req.MaxTokens)
	}
}

// ReasoningSent names what went on the wire, so two requests that render the
// same body read the same.
func TestReasoningLabel_IsTheWireMeaning(t *testing.T) {
	for _, tc := range []struct {
		r    ReasoningRequest
		want string
	}{
		{ReasoningOff(), "off"},
		{ReasoningEffort("none"), "off"},
		{ReasoningBudget(0), "off"},
		{ReasoningEffort("low"), "low"},
		{ReasoningBudget(256), "budget:256"},
		{nil, ""},
	} {
		if got := reasoningLabel(tc.r); got != tc.want {
			t.Errorf("%#v: %q, want %q", tc.r, got, tc.want)
		}
	}
}

// Validate is read-only: resolving an answer budget must not write a cap into
// the caller's request.
func TestMaxAnswerTokens_DoesNotTouchTheCallersRequest(t *testing.T) {
	c := intentsClient(t)
	req := ChatRequest{Model: "offable", Messages: []Message{User("hi")}, Reasoning: ReasoningBalanced(), MaxAnswerTokens: ptr(10)}
	if _, err := c.Validate(req); err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != nil || *req.MaxAnswerTokens != 10 || req.Reasoning != ReasoningBalanced() {
		t.Errorf("request mutated: %+v", req)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// --- load rules for the new profile fields ----------------------------------

func TestNewRegistry_IntentFieldRules(t *testing.T) {
	head := `profiles:
  - id: m
    display_name: M
    wire_model_id: m
    max_tokens_param: max_tokens
`
	for _, tc := range []struct{ name, doc, want string }{
		{"levels out of depth order", head + `    reasoning: {supported: true, enabled_by_default: true, control: effort, effort_values: [high, low]}
`, "shallowest first"},
		{"a level the depth ladder does not know", head + `    reasoning: {supported: true, enabled_by_default: true, control: effort, effort_values: [low, turbo]}
`, "turbo"},
		{"balanced outside the set", head + `    reasoning: {supported: true, enabled_by_default: true, control: effort, effort_values: [low, high], balanced: medium}
`, "balanced"},
		{"balanced none is off, not balanced", head + `    reasoning: {supported: true, enabled_by_default: true, can_be_disabled: true, control: effort, effort_values: [none, low], balanced: none}
`, "balanced"},
		{"overhead for a level the model does not take", head + `    reasoning: {supported: true, enabled_by_default: true, control: effort, effort_values: [low, high], overhead: {medium: 10}}
`, "overhead"},
		{"negative overhead", head + `    reasoning: {supported: true, enabled_by_default: true, control: effort, effort_values: [low, high], overhead: {low: -1}}
`, "overhead"},
		{"overhead on a model that does not reason", head + `    reasoning: {overhead: {off: 0}}
`, "unsupported"},
		{"negative min_budget", head + `    reasoning: {supported: true, enabled_by_default: true, control: budget_tokens, budget_param: b, min_budget: -1}
`, "min_budget"},
		{"min_budget on another control", head + `    reasoning: {supported: true, enabled_by_default: true, control: effort, effort_values: [low], min_budget: 5}
`, "min_budget"},
		{"embeddings declaring tool-arg buffering", `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    embedding: {default_dimensions: 8}
    streaming: {buffers_tool_args: true}
`, "streaming"},
		{"tool-arg buffering on a model that does not stream", head + `    streaming: {buffers_tool_args: true}
`, "buffers_tool_args"},
	} {
		_, err := NewRegistry([]byte(tc.doc))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want it to mention %q", tc.name, err, tc.want)
		}
	}
}

func TestNewRegistry_DisplayName(t *testing.T) {
	// A document written before the field existed still loads: the id stands
	// in, resolved at load like wire_model_id.
	reg := registryFrom(t, chatHead)
	if p := mustLookup(t, reg, "m"); p.DisplayName != "m" {
		t.Errorf("display_name = %q, want the id", p.DisplayName)
	}
	// A route keeps its model's label.
	reg = registryFrom(t, `providers:
  gw: {}
  v: {base_url: https://v.example}
profiles:
  - id: base
    display_name: Base Model
    wire_model_id: base
    provider: v
    max_tokens_param: max_tokens
  - id: routed
    base: base
    gateway: gw
    provider: gw
`)
	if p := mustLookup(t, reg, "routed"); p.DisplayName != "Base Model" {
		t.Errorf("route display_name = %q, want the base's", p.DisplayName)
	}
}

// Every shipped profile names itself; the id fallback is for documents written
// before the field, never for the embedded one.
func TestDefault_EveryProfileHasADisplayName(t *testing.T) {
	var file struct {
		Profiles []map[string]any `yaml:"profiles"`
	}
	if err := yaml.Unmarshal(embeddedProfiles, &file); err != nil {
		t.Fatal(err)
	}
	for _, p := range file.Profiles {
		if name, _ := p["display_name"].(string); name == "" {
			t.Errorf("%v: no display_name", p["id"])
		}
	}
	if p := mustLookup(t, Default(), "glm-5.3-flash"); p.DisplayName != "GLM 5.3 Flash" {
		t.Errorf("glm-5.3-flash display_name = %q", p.DisplayName)
	}
}

// Observed in production on MiMo: the whole argument is serialized server-side
// before it is flushed, ~82s of silence for a ~10KB spec. Data, so a caller
// sizes ToolCallIdleTimeout from the profile rather than from a model name.
func TestDefault_MiMoBuffersToolArgs(t *testing.T) {
	reg := Default()
	for _, id := range []string{"mimo-v2.5-pro", "mimo-v2.5", "mimo-v2.6-pro", "mimo-v2.6-flash"} {
		if !mustLookup(t, reg, id).Streaming.BuffersToolArgs {
			t.Errorf("%s: buffers_tool_args is false", id)
		}
	}
	for _, id := range []string{"glm-5.3-flash", "gpt-5.5"} {
		if mustLookup(t, reg, id).Streaming.BuffersToolArgs {
			t.Errorf("%s: buffers_tool_args is true with no observation behind it", id)
		}
	}
}
