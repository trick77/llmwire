package llmwire

import (
	"errors"
	"strings"
	"testing"
)

// testClient builds a client with no base URL: Validate must not need one,
// because being network-free is what lets a service check its configuration at
// boot rather than on the first user request.
func testClient(t *testing.T, reg *Registry) *Client {
	t.Helper()
	if reg == nil {
		reg = Default()
	}
	return New(Config{Registry: reg})
}

// registryFrom builds a one-off registry, for the shapes the shipped profiles
// do not cover.
func registryFrom(t *testing.T, doc string) *Registry {
	t.Helper()
	reg, err := NewRegistry([]byte(doc))
	if err != nil {
		t.Fatalf("building test registry: %v", err)
	}
	return reg
}

// --- the motivating case ---------------------------------------------------

// Asking a model that always thinks to stop thinking is refused before a socket
// opens, and the refusal names what the model DOES take. This is the case the
// package was built around: the vendor's own reference documents the disable
// toggle as valid, and the endpoint answers 400 with a code it also uses for
// unrelated failures.
func TestValidate_ReasoningOffOnAModelThatCannot(t *testing.T) {
	c := testClient(t, nil)
	_, err := c.Validate(ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hello")},
		Reasoning: ReasoningOff(),
	})
	if err == nil {
		t.Fatal("expected a refusal")
	}
	var unsup *UnsupportedError
	if !errors.As(err, &unsup) {
		t.Fatalf("error is %T, want *UnsupportedError", err)
	}
	msg := err.Error()
	for _, want := range []string{"glm-5.3-flash", "cannot be disabled", "low", "high", "max", "BestEffort"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error message is missing %q: %v", want, msg)
		}
	}
}

// BestEffort demotes that refusal to a warning, for this call only.
func TestValidate_BestEffortDemotesARefusalToAWarning(t *testing.T) {
	c := testClient(t, nil)
	req := ChatRequest{
		Model:      "glm-5.3-flash",
		Messages:   []Message{User("hello")},
		Reasoning:  ReasoningOff(),
		BestEffort: true,
	}
	warnings, err := c.Validate(req)
	if err != nil {
		t.Fatalf("BestEffort should have demoted the refusal: %v", err)
	}
	if len(warnings) != 1 {
		t.Fatalf("got %d warnings, want 1: %v", len(warnings), warnings)
	}
	if warnings[0].Kind != WarnUnsupported {
		t.Errorf("warning kind = %q, want %q", warnings[0].Kind, WarnUnsupported)
	}
	if !strings.Contains(warnings[0].Details, "BestEffort") {
		t.Errorf("warning should say why it was tolerated: %v", warnings[0])
	}

	// And it is per-request: the same client refuses the next call.
	req.BestEffort = false
	if _, err := c.Validate(req); err == nil {
		t.Error("BestEffort leaked beyond the request that set it")
	}
}

// The accepted effort set is per-model and narrower than the vendor's global
// enum, so a refusal has to name it or the caller is left guessing.
func TestValidate_EffortLevels(t *testing.T) {
	c := testClient(t, nil)
	for _, tc := range []struct {
		level     string
		wantError bool
	}{
		{"low", false},
		{"high", false},
		{"max", false},
		{"medium", true},
		{"minimal", true},
		{"xhigh", true},
		{"none", true},
		{"", true},
	} {
		t.Run(tc.level, func(t *testing.T) {
			_, err := c.Validate(ChatRequest{
				Model:     "glm-5.3-flash",
				Messages:  []Message{User("hi")},
				Reasoning: ReasoningEffort(tc.level),
			})
			if tc.wantError && err == nil {
				t.Fatalf("effort %q should be refused: the endpoint rejects it", tc.level)
			}
			if !tc.wantError && err != nil {
				t.Fatalf("effort %q should be accepted: %v", tc.level, err)
			}
			if tc.wantError && !strings.Contains(err.Error(), "low, high, max") {
				t.Errorf("refusal does not name the accepted set: %v", err)
			}
		})
	}
}

// A request variant that does not match the model's control is refused by name,
// rather than sent and rejected on the wire.
func TestValidate_ReasoningVariantMustMatchTheModelsControl(t *testing.T) {
	c := testClient(t, nil)

	// mimo takes a toggle object AND a level beside it (measured), so an
	// accepted level passes and one outside the set is refused by name.
	_, err := c.Validate(ChatRequest{
		Model:     "mimo-v2.5-pro",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningEffort("high"),
	})
	if err != nil {
		t.Errorf("error = %v, want a level the profile lists to pass", err)
	}
	_, err = c.Validate(ChatRequest{
		Model:     "mimo-v2.5-pro",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningEffort("xhigh"),
	})
	if err == nil || !strings.Contains(err.Error(), "not accepted by this model") {
		t.Errorf("error = %v, want a refusal naming the set", err)
	}

	// glm takes an effort level, not a budget.
	_, err = c.Validate(ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningBudget(1000),
	})
	if err == nil || !strings.Contains(err.Error(), "not as a token budget") {
		t.Errorf("error = %v, want a refusal naming the mismatch", err)
	}
}

// A model that CAN stop thinking accepts the request.
func TestValidate_ReasoningOffOnAModelThatCan(t *testing.T) {
	c := testClient(t, nil)
	if _, err := c.Validate(ChatRequest{
		Model:     "mimo-v2.5-pro",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningOff(),
	}); err != nil {
		t.Fatalf("mimo-v2.5-pro can disable thinking: %v", err)
	}
}

// --- vision ------------------------------------------------------------------

// An image for a text-only model is refused rather than rerouted to the
// vision-capable sibling. Answering with a model the caller did not name
// misattributes the answer and its cost; picking the model is the application's
// job.
func TestValidate_ImageAgainstATextOnlyModel(t *testing.T) {
	c := testClient(t, nil)
	msg := Message{Role: RoleUser, Parts: []Part{
		{Kind: PartText, Text: "what is this?"},
		{Kind: PartImage, URL: "data:image/png;base64,iVBORw0KGgo="},
	}}

	_, err := c.Validate(ChatRequest{Model: "mimo-v2.5-pro", Messages: []Message{msg}})
	if err == nil {
		t.Fatal("mimo-v2.5-pro is text-only and should refuse an image")
	}
	if !strings.Contains(err.Error(), "text-only") {
		t.Errorf("error = %v", err)
	}

	// The sibling takes it.
	if _, err := c.Validate(ChatRequest{Model: "mimo-v2.5", Messages: []Message{msg}}); err != nil {
		t.Fatalf("mimo-v2.5 supports vision: %v", err)
	}
}

// --- sampling -----------------------------------------------------------------

// A parameter the model accepts and then overrides is not an error — the call
// succeeds — but believing it took effect is how a caller spends a week tuning
// something the model discarded.
func TestValidate_InertSamplingParameterWarns(t *testing.T) {
	c := testClient(t, nil)
	temp := 0.2
	warnings, err := c.Validate(ChatRequest{
		Model:       "mimo-v2.5-pro",
		Messages:    []Message{User("hi")},
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("an inert parameter must not be an error: %v", err)
	}
	if len(warnings) == 0 {
		t.Fatal("expected a warning that the value will not take effect")
	}
	w := warnings[0]
	if w.Kind != WarnCompatibility || w.Feature != "temperature" {
		t.Errorf("warning = %+v", w)
	}
	if !strings.Contains(w.Details, "not take effect") {
		t.Errorf("warning should say the value is discarded: %v", w)
	}
}

// Only the default is accepted on some models: any other value is a 400, which
// is distinct from the parameter being unsupported.
func TestValidate_ForcedSamplingValue(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: forced
    wire_model_id: forced
    max_tokens_param: max_completion_tokens
    verified: measured
    temperature: {supported: true, forced_value: 1.0}
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)

	off := 0.3
	_, err := c.Validate(ChatRequest{Model: "forced", Messages: []Message{User("hi")}, Temperature: &off})
	if err == nil {
		t.Fatal("a non-default temperature should be refused")
	}
	if !strings.Contains(err.Error(), "must be 1") {
		t.Errorf("error = %v, want it to name the only accepted value", err)
	}

	// The forced value itself is fine, as is omitting the parameter.
	ok := 1.0
	if _, err := c.Validate(ChatRequest{Model: "forced", Messages: []Message{User("hi")}, Temperature: &ok}); err != nil {
		t.Errorf("the forced value must be accepted: %v", err)
	}
	if _, err := c.Validate(ChatRequest{Model: "forced", Messages: []Message{User("hi")}}); err != nil {
		t.Errorf("omitting the parameter must be accepted: %v", err)
	}
}

func TestValidate_UnsupportedSamplingParameter(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: nosample
    wire_model_id: nosample
    max_tokens_param: max_completion_tokens
    verified: measured
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)
	temp := 0.5
	_, err := c.Validate(ChatRequest{Model: "nosample", Messages: []Message{User("hi")}, Temperature: &temp})
	if err == nil || !strings.Contains(err.Error(), "temperature is not supported") {
		t.Errorf("error = %v", err)
	}
}

// --- structured output ---------------------------------------------------------

// A schema downgraded to a bare JSON object is approximate but honourable: the
// reply is still constrained to JSON, just not to this shape. Exactly what
// warnings exist for.
func TestValidate_JSONSchemaDowngradesWithAWarning(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: objectonly
    wire_model_id: objectonly
    max_tokens_param: max_completion_tokens
    verified: measured
    output: {json_object: true}
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)
	warnings, err := c.Validate(ChatRequest{
		Model:    "objectonly",
		Messages: []Message{User("hi")},
		ResponseFormat: ResponseFormat{
			Kind:   FormatJSONSchema,
			Name:   "answer",
			Schema: map[string]any{"type": "object"},
		},
	})
	if err != nil {
		t.Fatalf("a downgrade must not be an error: %v", err)
	}
	if len(warnings) != 1 || warnings[0].Kind != WarnCompatibility {
		t.Fatalf("warnings = %v, want one compatibility warning", warnings)
	}
	if !strings.Contains(warnings[0].Details, "not enforced") {
		t.Errorf("warning should say the schema is not enforced: %v", warnings[0])
	}
}

// Both models measured in Phase 0 do support json_schema, despite their vendor
// docs. No warning should appear for them.
func TestValidate_JSONSchemaIsCleanOnModelsThatSupportIt(t *testing.T) {
	c := testClient(t, nil)
	for _, model := range []string{"glm-5.3-flash", "mimo-v2.5-pro"} {
		warnings, err := c.Validate(ChatRequest{
			Model:    model,
			Messages: []Message{User("hi")},
			ResponseFormat: ResponseFormat{
				Kind:   FormatJSONSchema,
				Name:   "answer",
				Schema: map[string]any{"type": "object"},
			},
		})
		if err != nil {
			t.Errorf("%s: %v", model, err)
		}
		if len(warnings) != 0 {
			t.Errorf("%s: unexpected warnings %v", model, warnings)
		}
	}
}

func TestValidate_MalformedResponseFormat(t *testing.T) {
	c := testClient(t, nil)
	for _, tc := range []struct {
		name       string
		format     ResponseFormat
		wantSubstr string
	}{
		{"schema without a name", ResponseFormat{Kind: FormatJSONSchema, Schema: map[string]any{"type": "object"}}, "needs a name"},
		{"schema without a schema", ResponseFormat{Kind: FormatJSONSchema, Name: "x"}, "needs a schema"},
		{"unknown kind", ResponseFormat{Kind: "yaml_object"}, "unknown response format"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := c.Validate(ChatRequest{
				Model:          "glm-5.3-flash",
				Messages:       []Message{User("hi")},
				ResponseFormat: tc.format,
			})
			if err == nil || !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error = %v, want it to mention %q", err, tc.wantSubstr)
			}
		})
	}
}

// --- tools ----------------------------------------------------------------------

// A forced choice on an endpoint that takes only "auto" is relaxed with a
// warning rather than dropped silently — several of these endpoints drop the
// field themselves without saying so, which is how a caller comes to believe a
// call was forced when it was not.
func TestValidate_ForcedToolChoiceIsRelaxedWithAWarning(t *testing.T) {
	c := testClient(t, nil)
	warnings, err := c.Validate(ChatRequest{
		Model:      "mimo-v2.5-pro",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceRequired},
	})
	if err != nil {
		t.Fatalf("a relaxable tool choice must not be an error: %v", err)
	}
	if len(warnings) != 1 || warnings[0].Feature != "tool_choice" {
		t.Fatalf("warnings = %v", warnings)
	}
	if !strings.Contains(warnings[0].Details, "relaxed") {
		t.Errorf("warning = %v", warnings[0])
	}
}

func TestValidate_AutoToolChoiceIsClean(t *testing.T) {
	c := testClient(t, nil)
	warnings, err := c.Validate(ChatRequest{
		Model:      "mimo-v2.5-pro",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto},
	})
	if err != nil || len(warnings) != 0 {
		t.Errorf("err = %v, warnings = %v", err, warnings)
	}
}

func TestValidate_ToolChoiceWithoutTools(t *testing.T) {
	c := testClient(t, nil)
	warnings, err := c.Validate(ChatRequest{
		Model:      "mimo-v2.5-pro",
		Messages:   []Message{User("hi")},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0].Details, "no tools were offered") {
		t.Errorf("warnings = %v", warnings)
	}
}

func TestValidate_MalformedTools(t *testing.T) {
	c := testClient(t, nil)
	_, err := c.Validate(ChatRequest{
		Model:    "mimo-v2.5-pro",
		Messages: []Message{User("hi")},
		Tools:    []Tool{{Name: ""}},
	})
	if err == nil || !strings.Contains(err.Error(), "no name") {
		t.Errorf("error = %v", err)
	}

	_, err = c.Validate(ChatRequest{
		Model:      "mimo-v2.5-pro",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "x"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceFunction},
	})
	if err == nil || !strings.Contains(err.Error(), "needs the function's name") {
		t.Errorf("error = %v", err)
	}
}

// --- messages and routing ---------------------------------------------------------

func TestValidate_MessageProblems(t *testing.T) {
	c := testClient(t, nil)

	if _, err := c.Validate(ChatRequest{Model: "glm-5.3-flash"}); err == nil ||
		!strings.Contains(err.Error(), "at least one message") {
		t.Errorf("empty messages: err = %v", err)
	}

	_, err := c.Validate(ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{{Role: RoleTool, Text: "result"}},
	})
	if err == nil || !strings.Contains(err.Error(), "ToolCallID") {
		t.Errorf("tool message without an id: err = %v", err)
	}
}

// A chat request aimed at an embeddings model is caught by the schema, not by a
// confusing 400 from the wrong route.
func TestValidate_ChatRequestAgainstAnEmbeddingsModel(t *testing.T) {
	c := testClient(t, nil)
	_, err := c.Validate(ChatRequest{
		Model:    "text-embedding-3-small",
		Messages: []Message{User("hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "cannot take a chat request") {
		t.Errorf("error = %v", err)
	}
}

func TestValidate_UnknownModel(t *testing.T) {
	c := testClient(t, nil)
	_, err := c.Validate(ChatRequest{Model: "gpt-5.9-imaginary", Messages: []Message{User("hi")}})
	var unknown *UnknownModelError
	if !errors.As(err, &unknown) {
		t.Fatalf("error is %T, want *UnknownModelError: %v", err, err)
	}
}

// A clean request produces no warnings at all. Worth asserting: a validator that
// warns about everything trains callers to ignore warnings.
func TestValidate_CleanRequestIsSilent(t *testing.T) {
	c := testClient(t, nil)
	warnings, err := c.Validate(ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{System("be brief"), User("hi")},
		Reasoning: ReasoningEffort("high"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("clean request produced warnings: %v", warnings)
	}
}

// Only the FIRST refusal is reported. A caller fixes one thing and runs again,
// and a list of complaints about a request rejected on its first problem is
// mostly noise about parameters that were never really evaluated.
func TestValidate_ReportsTheFirstRefusalOnly(t *testing.T) {
	c := testClient(t, nil)
	temp := 0.5
	_, err := c.Validate(ChatRequest{
		Model:       "glm-5.3-flash",
		Messages:    nil, // first problem
		Reasoning:   ReasoningOff(),
		Temperature: &temp,
	})
	if err == nil {
		t.Fatal("expected a refusal")
	}
	if !strings.Contains(err.Error(), "at least one message") {
		t.Errorf("error = %v, want the first problem encountered", err)
	}
}

func TestWarnings_RendersStably(t *testing.T) {
	if got := Warnings(nil); got != "" {
		t.Errorf("Warnings(nil) = %q, want empty", got)
	}
	ws := []Warning{
		{Kind: WarnUnsupported, Feature: "b", Details: "second"},
		{Kind: WarnCompatibility, Feature: "a", Details: "first"},
	}
	got := Warnings(ws)
	if !strings.Contains(got, "a") || !strings.Contains(got, "b") {
		t.Errorf("Warnings = %q", got)
	}
	if got != Warnings(ws) {
		t.Error("Warnings is not stable across calls")
	}
}
