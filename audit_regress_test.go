package llmwire

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// Regression tests for the findings of the 2026-09-24 audit of the loader,
// the gateway route, the environment and the request path. Each one failed
// before its fix.

// --- tool choice ------------------------------------------------------------------

// A forced function is an OBJECT on the wire and is admitted by
// supports_forced_choice, not by a "function" entry in the string list. Before
// the fix every shipped model that declares the bit still relaxed the call to
// "auto" with a warning, so forcing a named tool never worked anywhere.
func TestValidate_ForcedFunctionChoiceUsesTheBit(t *testing.T) {
	c := testClient(t, nil)
	req := ChatRequest{
		Model:      "gpt-5.4",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceFunction, Name: "search"},
	}
	warnings, err := c.Validate(req)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	for _, w := range warnings {
		if w.Feature == "tool_choice" {
			t.Fatalf("a model with supports_forced_choice relaxed the forced call: %v", w)
		}
	}

	// And a model without the bit still relaxes it, with the warning.
	req.Model = "mimo-v2.5-pro"
	warnings, err = c.Validate(req)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	var relaxed bool
	for _, w := range warnings {
		if w.Feature == "tool_choice" && strings.Contains(w.Details, "relaxed") {
			relaxed = true
		}
	}
	if !relaxed {
		t.Errorf("warnings = %v, want the relaxation warning on a model that cannot force", warnings)
	}
}

// A mode this package does not know is a typo, not a request for "auto".
func TestValidate_UnknownToolChoiceModeIsRejected(t *testing.T) {
	_, err := testClient(t, nil).Validate(ChatRequest{
		Model:      "gpt-5.4",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceMode("any")},
		BestEffort: true,
	})
	if err == nil || !strings.Contains(err.Error(), "not a tool choice mode") {
		t.Fatalf("err = %v, want a structural rejection even under BestEffort", err)
	}
}

// --- loader rules --------------------------------------------------------------------

const embedHead = `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
    embedding: {default_dimensions: 8}
`

// Every rule the loader's comments promised and the code lacked. Each case
// loaded clean before the fix.
func TestNewRegistry_RulesThePromisedButDidNotEnforce(t *testing.T) {
	for _, tc := range []struct{ name, doc, want string }{
		{"tool_choice_values with function", chatHead +
			"    tools: {supported: true, tool_choice_values: [auto, function], supports_forced_choice: true}\n",
			"is not one of"},
		{"tool_choice_values typo", chatHead +
			"    tools: {supported: true, tool_choice_values: [auto, requird]}\n",
			"is not one of"},
		{"tool_choice_values duplicate", chatHead +
			"    tools: {supported: true, tool_choice_values: [auto, auto]}\n",
			"twice"},
		{"tools unsupported with parallel", chatHead +
			"    tools: {supported: false, supports_parallel: true}\n",
			"tool settings are declared"},
		{"tools unsupported with recovery", chatHead +
			"    tools: {supported: false, recover_inline_markup: true}\n",
			"tool settings are declared"},
		{"temperature unsupported with forced_value", chatHead +
			"    temperature: {supported: false, forced_value: 1.0}\n",
			"temperature is unsupported"},
		{"top_p unsupported with inert flag", chatHead +
			"    top_p: {supported: false, inert_while_reasoning: true}\n",
			"top_p is unsupported"},
		{"recommended differs from forced", chatHead +
			"    temperature: {supported: true, forced_value: 1.0, recommended_value: 0.7}\n",
			"differs from forced_value"},
		{"default_effort on a budget model", chatHead +
			"    reasoning: {supported: true, control: budget_tokens, budget_param: b, enabled_by_default: true, default_effort: low}\n",
			"takes no levels"},
		{"stream_field without reasoning", chatHead +
			"    reasoning: {supported: false, stream_field: reasoning_content}\n",
			"reasoning is unsupported but"},
		{"leaks_close_tag without reasoning", chatHead +
			"    reasoning: {supported: false, leaks_close_tag: true}\n",
			"reasoning is unsupported but"},
		{"duplicate effort_values", chatHead +
			"    reasoning: {supported: true, control: effort, effort_values: [low, low], enabled_by_default: true}\n",
			"twice"},
		{"negative limits", chatHead +
			"    limits: {context: -1}\n",
			"must not be negative"},
		{"embeddings with max_tokens_param", embedHead +
			"    max_tokens_param: max_tokens\n",
			"max_tokens_param"},
		{"embeddings with streaming", embedHead +
			"    streaming: {supported: true}\n",
			"streaming"},
		{"embeddings with a reasoning control", embedHead +
			"    reasoning: {supported: false, control: effort}\n",
			"reasoning"},
		{"embeddings with a tools format", embedHead +
			"    tools: {supported: false, format: native}\n",
			"tools"},
		{"embeddings with a recommended temperature", embedHead +
			"    temperature: {supported: false, recommended_value: 1.0}\n",
			"temperature"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(tc.doc))
			if err == nil {
				t.Fatal("loaded clean; the rule is not enforced")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

// A derived entry that changes provider is reached through another host, and
// another host is another party pricing the call: it must name a gateway, or
// it would keep the base's list rate for calls the vendor never sees.
func TestNewRegistry_ProviderChangeNeedsAGateway(t *testing.T) {
	const providers = `providers:
  vendor: {base_url: https://vendor.example/v1}
  proxy: {base_url: https://proxy.example/v1}
`
	const base = `profiles:
  - id: m
    provider: vendor
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance
	_, err := NewRegistry([]byte(providers + base + "  - id: r\n    base: m\n    provider: proxy\n    wire_model_id: alias\n"))
	if err == nil || !strings.Contains(err.Error(), "names no gateway") {
		t.Fatalf("err = %v, want a refusal naming the missing gateway", err)
	}
	reg := registryFrom(t, providers+base+"  - id: r\n    base: m\n    gateway: proxy\n    provider: proxy\n    wire_model_id: alias\n")
	if p := mustLookup(t, reg, "r"); p.Cost != nil {
		t.Errorf("route carries the base's cost block %+v; a gateway route is never priced from the table", p.Cost)
	}
	// Same provider, no gateway: an alias on the vendor's own host keeps the
	// vendor's rate.
	reg = registryFrom(t, providers+base+"  - id: r\n    base: m\n    wire_model_id: alias\n")
	if p := mustLookup(t, reg, "r"); p.Cost == nil {
		t.Error("an alias on the same host lost its cost block")
	}
}

// Tighten-only means ON is the only value a route can say: an explicit false
// would load and change nothing, which is the silent-config failure the
// allowlist exists to prevent.
func TestNewRegistry_DerivedFalseOnATightenOnlyKeyIsRefused(t *testing.T) {
	doc := `profiles:
  - id: base-model
    wire_model_id: base-model
    max_tokens_param: max_tokens
    verified: measured
    tools: {supported: true, format: native, tool_choice_values: [auto], recover_inline_markup: true}
    streaming: {supported: true, accepts_stream_options: true, needs_include_usage: true}
  - id: gw/base-model
    base: base-model
    streaming: {needs_include_usage: false}
`
	_, err := NewRegistry([]byte(doc))
	if err == nil || !strings.Contains(err.Error(), "tighten-only") {
		t.Fatalf("err = %v, want a refusal of the false", err)
	}
}

// endpoint is a property of the model: a route cannot move a chat model to
// the embeddings route.
func TestNewRegistry_DerivedCannotChangeEndpoint(t *testing.T) {
	_, err := NewRegistry([]byte(chatHead + "  - id: r\n    base: m\n    endpoint: embeddings\n"))
	if err == nil || !strings.Contains(err.Error(), "endpoint") {
		t.Fatalf("err = %v, want endpoint refused on a derived entry", err)
	}
}

// Two bad providers name the same one on every run.
func TestNewRegistry_BadProvidersAreNamedInOrder(t *testing.T) {
	doc := "providers:\n  zeta: {base_url: http://z.example}\n  alpha: {base_url: http://a.example}\n" + chatHead
	for i := 0; i < 5; i++ {
		_, err := NewRegistry([]byte(doc))
		if err == nil || !strings.Contains(err.Error(), `provider "alpha"`) {
			t.Fatalf("run %d: err = %v, want the alphabetically first provider named", i, err)
		}
	}
}

// A lone "." is a stray character, not a free lane.
func TestRate_LoneDotIsRefused(t *testing.T) {
	for _, v := range []string{".", "-."} {
		doc := chatHead + "    cost:\n      input: '" + v + "'\n      cache_read: 0\n      cache_write: 0\n      output: 0\n" + goodProvenance
		if _, err := NewRegistry([]byte(doc)); err == nil || !strings.Contains(err.Error(), "not a decimal number") {
			t.Errorf("rate %q: err = %v, want a refusal", v, err)
		}
	}
}

// --- reported figures from a gateway ---------------------------------------------

// big.Rat.SetString accepts fractions, hex floats and an exponent of any
// size; a gateway's text is not trusted to pick the size of an allocation.
func TestParseReportedCost_RejectsNonDecimalForms(t *testing.T) {
	for _, s := range []string{"1/3", "0x1p-3", "1e99999", "1e9999999", "1e", ""} {
		if _, err := parseReportedCost(s); err == nil {
			t.Errorf("%q parsed; want a refusal", s)
		}
	}
	for _, s := range []string{"1.23e-05", "0.5", ".5", "5.", "+1E2"} {
		if _, err := parseReportedCost(s); err != nil {
			t.Errorf("%q: %v; want it accepted", s, err)
		}
	}
}

func TestLimitField_BoundsTheExponent(t *testing.T) {
	var warnings []Warning
	if got := limitField("m", "max_input_tokens", json.RawMessage(`1e9999999`), &warnings); got != nil {
		t.Errorf("got %d, want nil", *got)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0].Details, "not a finite number") {
		t.Errorf("warnings = %v", warnings)
	}
}

// --- token counts ------------------------------------------------------------------

// A negative count means nothing on the wire, and carrying it into pricing
// would credit the caller. It is read as not reported, so the call is
// Unpriced with a warning rather than priced below zero.
func TestParseUsage_NegativeCountIsNotReported(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"prompt_tokens":10,"completion_tokens":-7,"completion_tokens_details":{"reasoning_tokens":-1}}`))
	if u.Output.Total != nil || u.Output.Reasoning != nil {
		t.Fatalf("output = %+v, want the negative lanes unreported", u.Output)
	}
	p := mustLookup(t, Default(), "glm-5.3-flash")
	cost, warnings := priceCall(p, u, http.Header{}, 200, time.Unix(0, 0).UTC())
	if cost.Provenance != Unpriced || cost.NanoUSD != 0 || len(warnings) != 1 {
		t.Errorf("cost = %+v warnings = %v, want Unpriced with one warning", cost, warnings)
	}
}

// --- gateway routes and the environment ---------------------------------------

// A proxy forwards the usage chunk only to a client that asked for it, so an
// env-declared route streams with stream_options.include_usage. Before the
// fix the route inherited the base's needs_include_usage: false and every
// stream through the gateway was Unpriced.
func TestFromEnv_gatewayStreamAsksForUsage(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()
	env := map[string]string{
		"LLMWIRE_LITELLM_BASE_URL": srv.URL,
		"LLMWIRE_LITELLM_API_KEY":  "gw-key",
		GatewayModelsEnv:           "glm-5.3-flash=proxy-glm",
	}
	c, err := FromEnv("glm-5.3-flash", Config{Lookup: func(k string) (string, bool) { v, ok := env[k]; return v, ok }})
	if err != nil {
		t.Fatal(err)
	}
	s, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Collect(nil); err != nil {
		t.Fatal(err)
	}
	var sent map[string]any
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatal(err)
	}
	opts, _ := sent["stream_options"].(map[string]any)
	if opts["include_usage"] != true {
		t.Errorf("body = %s, want stream_options.include_usage on a gateway route", body)
	}
}

// One wire name is one deployment; two profiles claiming it means at least
// one capability set is wrong for it.
func TestParseGatewayModels_RefusesOneNameForTwoModels(t *testing.T) {
	if _, err := parseGatewayModels("a=x,b=x"); err == nil || !strings.Contains(err.Error(), "one deployment is one model") {
		t.Fatalf("err = %v", err)
	}
}

// A base that is already reached through a gateway cannot be routed again.
func TestRegistry_viaGatewayRefusesAGatewayBase(t *testing.T) {
	reg := registryFrom(t, `providers:
  litellm: {}
profiles:
  - id: g
    gateway: mystery
    wire_model_id: g
    max_tokens_param: max_tokens
    verified: source-derived
`)
	if _, err := reg.viaGateway(map[string]string{"g": "x"}); err == nil || !strings.Contains(err.Error(), "already reached through gateway") {
		t.Fatalf("err = %v", err)
	}
}

// The host read from the environment passes the checks a shipped host passes
// at load, with one allowance: plain http for a host on this machine, which
// is where a self-hosted gateway usually is.
func TestFromEnv_baseURLFromTheEnvironmentIsValidated(t *testing.T) {
	reg := registryFrom(t, "providers:\n  gw: {}\n"+chatHead+"    provider: gw\n")
	for _, tc := range []struct {
		url  string
		want string // "" means accepted
	}{
		{"http://localhost:4000/v1", ""},
		{"http://127.0.0.1:4000", ""},
		{"http://[::1]:4000", ""},
		{"https://gw.example/v1/", ""},
		{"http://gw.example/v1", "must be https"},
		{"https://gw.example/v1?key=x", "bare root"},
		{"https://user:pw@gw.example/v1", "bare root"},
		{"https://gw.example/v1/chat/completions", "ends in a route"},
	} {
		t.Run(tc.url, func(t *testing.T) {
			env := map[string]string{"LLMWIRE_GW_BASE_URL": tc.url, "LLMWIRE_GW_API_KEY": "k"}
			_, err := FromEnv("m", Config{Registry: reg, Lookup: func(k string) (string, bool) { v, ok := env[k]; return v, ok }})
			switch {
			case tc.want == "" && err != nil:
				t.Fatalf("refused: %v", err)
			case tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "LLMWIRE_GW_BASE_URL")):
				t.Fatalf("err = %v, want a refusal naming the variable and %q", err, tc.want)
			}
		})
	}
}

// --- request path --------------------------------------------------------------------

// ExtraBody wins over every generated key but the ones that would make the
// body lie about itself: a caller who switches `stream` swaps the parser out
// from under the call, and a second cap spelling breaks the one rule the
// profile exists to enforce.
func TestValidate_ExtraBodyReservedKeysAreRejected(t *testing.T) {
	c := testClient(t, nil)
	for key, value := range map[string]any{
		"stream":                true,
		"stream_options":        map[string]any{"include_usage": true},
		"model":                 "other",
		"max_completion_tokens": 5, // glm takes max_tokens; the other spelling is refused
	} {
		t.Run(key, func(t *testing.T) {
			_, err := c.Validate(ChatRequest{
				Model:      "glm-5.3-flash",
				Messages:   []Message{User("hi")},
				ExtraBody:  map[string]any{key: value},
				BestEffort: true,
			})
			var ue *UnsupportedError
			if !errors.As(err, &ue) || ue.Feature != "extra_body" || ue.Demotable {
				t.Fatalf("err = %v, want a hard extra_body refusal", err)
			}
		})
	}
	// The profile's own spelling, and any other key, still pass through.
	if _, err := c.Validate(ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hi")},
		ExtraBody: map[string]any{"max_tokens": 5, "thinking_budget": 3},
	}); err != nil {
		t.Fatalf("an ordinary ExtraBody was refused: %v", err)
	}
}

// A part with no Kind is not text: rendering it as an empty text part loses
// the image and skips the vision check, so it is refused by index.
func TestValidate_PartKindIsChecked(t *testing.T) {
	c := testClient(t, nil)
	_, err := c.Validate(ChatRequest{
		Model:    "mimo-v2.5",
		Messages: []Message{{Role: RoleUser, Parts: []Part{{Kind: PartText, Text: "look"}, {URL: "data:image/png;base64,AAAA"}}}},
	})
	if err == nil || !strings.Contains(err.Error(), "part 1") {
		t.Fatalf("err = %v, want the kind-less part refused by index", err)
	}
	_, err = c.Validate(ChatRequest{
		Model:    "mimo-v2.5",
		Messages: []Message{{Role: RoleUser, Parts: []Part{{Kind: PartImage}}}},
	})
	if err == nil || !strings.Contains(err.Error(), "no URL") {
		t.Fatalf("err = %v, want an image part without a URL refused", err)
	}
}

// A zero or negative cap has no meaning any endpoint shares, and a negative
// budget none at all. A zero budget is the disable switch in another
// spelling, so it is refused where ReasoningOff would be.
func TestValidate_CapAndBudgetBounds(t *testing.T) {
	c := testClient(t, nil)
	zero := 0
	if _, err := c.Validate(ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}, MaxTokens: &zero, BestEffort: true}); err == nil ||
		!strings.Contains(err.Error(), "not positive") {
		t.Errorf("zero cap: err = %v, want a rejection", err)
	}
	reg := registryFrom(t, chatHead+
		"    reasoning: {supported: true, control: budget_tokens, budget_param: thinking_budget, enabled_by_default: true}\n")
	bc := testClient(t, reg)
	if _, err := bc.Validate(ChatRequest{Model: "m", Messages: []Message{User("hi")}, Reasoning: ReasoningBudget(-1)}); err == nil ||
		!strings.Contains(err.Error(), "negative") {
		t.Errorf("negative budget: err = %v, want a rejection", err)
	}
	_, err := bc.Validate(ChatRequest{Model: "m", Messages: []Message{User("hi")}, Reasoning: ReasoningBudget(0)})
	if err == nil || !strings.Contains(err.Error(), "cannot be disabled") {
		t.Errorf("zero budget on a model that cannot be disabled: err = %v, want the disable refusal", err)
	}
}

// ValidateStream checks what ChatStream would refuse; Validate alone plans a
// non-streaming call and so never sees the streaming refusal.
func TestValidateStream_SeesTheStreamingRefusal(t *testing.T) {
	reg := registryFrom(t, chatHead+"    streaming: {supported: false}\n")
	c := testClient(t, reg)
	req := ChatRequest{Model: "m", Messages: []Message{User("hi")}}
	if _, err := c.Validate(req); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if _, err := c.ValidateStream(req); err == nil || !strings.Contains(err.Error(), "does not stream") {
		t.Fatalf("ValidateStream: err = %v, want the streaming refusal", err)
	}
}

// --- review findings on the sweep itself ---------------------------------------

// A proxy writes its cost header before the body, and on a stream has
// shipped it as a literal 0. A stream that was cut cannot be described by
// that header at all, so pricing it would record a confident zero for
// exactly the call that failed: the header lane is withheld unless the
// stream ended cleanly.
func TestChatStream_CutGatewayStreamIgnoresTheCostHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set(litellmCostHeader, "0")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"half\"}}]}\n\n"))
	}))
	t.Cleanup(srv.Close)
	reg := registryFrom(t, `profiles:
  - id: m
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
    streaming: {supported: true, accepts_stream_options: true}
  - id: m-via
    base: m
    gateway: litellm
    wire_model_id: proxy/m
`)
	c := New(Config{BaseURL: srv.URL, Registry: reg})
	s, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "m-via", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	res, err := s.Collect(nil)
	if !errors.Is(err, ErrMalformedResponse) {
		t.Fatalf("err = %v, want the cut-stream error", err)
	}
	if res.Usage.Cost.Provenance != Unpriced {
		t.Errorf("cost = %+v, want Unpriced: the header cannot describe a cut stream", res.Usage.Cost)
	}
}

// With recovery on, an error frame ends the scan but not the recovery tail:
// the markup that arrived before it is still turned into calls, and text the
// gate was holding still reaches the sink.
func TestReadStream_ErrorFrameStillRecoversInline(t *testing.T) {
	body := "data: " + contentDelta("<tool_call><function=search><parameter=q>x</parameter></function></tool_call>") + "\n\n" +
		"data: {\"error\":{\"message\":\"upstream exploded\",\"code\":\"1210\"}}\n\n"
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var calls int
	res, _, err := readStream(strings.NewReader(body), guard, streamBounds{idle: time.Hour}, func(ev streamEvent) {
		if ev.kind == evToolCall {
			calls++
		}
	}, Redact, true, time.Now, time.Now())
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "1210" {
		t.Fatalf("err = %v, want the frame's *APIError", err)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "search" || strings.Contains(res.Content, "<tool_call>") {
		t.Errorf("res = %+v, want the markup recovered into a call despite the error", res)
	}
	if calls == 0 {
		t.Error("no tool-call event reached the sink")
	}
}

// A usage object whose every figure was negative has nothing countable in
// it and reads as unreported, so the call is flagged, not merely unpriced.
func TestParseUsage_AllNegativeIsUnreported(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"prompt_tokens":-1,"completion_tokens":-1}`))
	if u.Reported() {
		t.Error("Reported() is true for a usage object with nothing countable in it")
	}
	if parseUsage(json.RawMessage(`{"prompt_tokens":-1,"completion_tokens":3}`)).Reported() != true {
		t.Error("one surviving lane must still count as reported")
	}
}

// Arguments that arrive as an object instead of a string are kept as their
// JSON text rather than failing the whole response.
func TestParseChatResponse_ObjectArgumentsAreKeptAsText(t *testing.T) {
	resp, err := parseChatResponseWith(Redact, json.RawMessage(`{"choices":[{"finish_reason":"tool_calls","message":{"content":"",
	  "tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":{"q":"x"}}}]}}]}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Arguments != `{"q":"x"}` {
		t.Errorf("tool calls = %+v, want the object kept as its JSON text", resp.ToolCalls)
	}
}
