package llmwire

import (
	"errors"
	"testing"
)

// What plan DECIDES has to be readable as data, not only as prose in a warning.
// Before this, every coercion reached the renderer as a sentence, so the renderer
// had to re-derive the same decision from the profile — two copies of one policy,
// free to disagree. These tests pin the coerced request instead.

func f64(v float64) *float64 { return &v }

func mustPlan(t *testing.T, c *Client, req ChatRequest, stream bool) *wirePlan {
	t.Helper()
	pl, _, err := c.plan(req, stream)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	return pl
}

// The cap parameter is chosen from the profile and carried on the plan. This is
// the phase's reason to exist: one endpoint ACCEPTS the wrong spelling and
// ignores it, so a request capped at 16 came back with 481 completion tokens and
// finish_reason "stop".
func TestPlan_CapParamComesFromTheProfile(t *testing.T) {
	c := testClient(t, Default())
	for _, tc := range []struct{ model, want string }{
		{"glm-5.3-flash", ParamMaxTokens},
		{"mimo-v2.5-pro", ParamMaxCompletionTokens},
	} {
		pl := mustPlan(t, c, ChatRequest{
			Model:    tc.model,
			Messages: []Message{User("hi")},
		}, false)
		if pl.capParam != tc.want {
			t.Errorf("%s: capParam = %q, want %q", tc.model, pl.capParam, tc.want)
		}
	}
}

// A caller who says nothing about sampling gets the vendor's tuned operating
// point, because omitting the parameter does not mean "the model's default" on
// this endpoint — its fallback is materially lower than the point it was tuned
// for.
func TestPlan_RecommendedSamplingFillsInSilently(t *testing.T) {
	c := testClient(t, Default())
	pl, warnings, err := c.plan(ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	}, false)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if pl.req.Temperature == nil || *pl.req.Temperature != 1.0 {
		t.Errorf("temperature = %v, want the profile's 1.0", pl.req.Temperature)
	}
	if pl.req.TopP == nil || *pl.req.TopP != 0.95 {
		t.Errorf("top_p = %v, want the profile's 0.95", pl.req.TopP)
	}
	// Tier 2: lossless relative to what the caller expressed, so no warning.
	if len(warnings) != 0 {
		t.Errorf("filling a recommended default warned: %v", warnings)
	}

	// A caller value wins over the recommendation.
	pl = mustPlan(t, c, ChatRequest{
		Model:       "glm-5.3-flash",
		Messages:    []Message{User("hi")},
		Temperature: f64(0.2),
	}, false)
	if *pl.req.Temperature != 0.2 {
		t.Errorf("temperature = %v, want the caller's 0.2", *pl.req.Temperature)
	}
}

// A model with no recommended value gets nothing: absent means "not measured",
// and inventing a value here would assert something nobody checked.
func TestPlan_NoRecommendationSendsNothing(t *testing.T) {
	pl := mustPlan(t, testClient(t, Default()), ChatRequest{
		Model:    "mimo-v2.5-pro",
		Messages: []Message{User("hi")},
	}, false)
	if pl.req.Temperature != nil || pl.req.TopP != nil {
		t.Errorf("temperature=%v top_p=%v, want both absent", pl.req.Temperature, pl.req.TopP)
	}
}

// Every BestEffort demotion has to leave the coerced request without the thing
// that was refused. A refusal that warned and then sent the knob anyway would
// turn a caught error into the 400 this package exists to prevent.
func TestPlan_BestEffortDropsWhatItRefused(t *testing.T) {
	c := testClient(t, Default())

	t.Run("reasoning off on a model that always thinks", func(t *testing.T) {
		pl := mustPlan(t, c, ChatRequest{
			Model:      "glm-5.3-flash",
			Messages:   []Message{User("hi")},
			Reasoning:  ReasoningOff(),
			BestEffort: true,
		}, false)
		if pl.req.Reasoning != nil {
			t.Error("the refused reasoning request survived into the plan")
		}
	})

	t.Run("effort outside the accepted set", func(t *testing.T) {
		pl := mustPlan(t, c, ChatRequest{
			Model:      "glm-5.3-flash",
			Messages:   []Message{User("hi")},
			Reasoning:  ReasoningEffort("medium"),
			BestEffort: true,
		}, false)
		if pl.req.Reasoning != nil {
			t.Error("the refused effort survived into the plan")
		}
	})

	t.Run("image on a text-only model keeps the text", func(t *testing.T) {
		req := ChatRequest{
			Model: "mimo-v2.5-pro",
			Messages: []Message{{
				Role: RoleUser,
				Parts: []Part{
					{Kind: PartText, Text: "what is this"},
					{Kind: PartImage, URL: "data:image/png;base64,AAAA"},
				},
			}},
			BestEffort: true,
		}
		pl := mustPlan(t, c, req, false)
		parts := pl.req.Messages[0].Parts
		if len(parts) != 1 || parts[0].Kind != PartText {
			t.Fatalf("parts = %+v, want the text part only", parts)
		}
		// The caller's own slice is untouched: Validate is documented as
		// network-free and read-only, and it is exported.
		if len(req.Messages[0].Parts) != 2 {
			t.Error("planning mutated the caller's message")
		}
	})

	t.Run("tools on a model without them", func(t *testing.T) {
		reg := registryFrom(t, `profiles:
  - id: plain
    wire_model_id: plain
    max_tokens_param: max_tokens
    verified: measured
    streaming: {supported: true, accepts_stream_options: true}
`)
		pl := mustPlan(t, testClient(t, reg), ChatRequest{
			Model:      "plain",
			Messages:   []Message{User("hi")},
			Tools:      []Tool{{Name: "search"}},
			ToolChoice: ToolChoice{Mode: ToolChoiceAuto},
			BestEffort: true,
		}, false)
		if pl.req.Tools != nil || pl.req.ToolChoice.Mode != ToolChoiceUnset {
			t.Errorf("tools=%v choice=%v, want both dropped", pl.req.Tools, pl.req.ToolChoice)
		}
	})
}

// The tier-3 coercions are recorded on the request too, so the renderer emits
// what was decided rather than deciding again.
func TestPlan_Tier3CoercionsAreRecorded(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: jsonobj
    wire_model_id: jsonobj
    max_tokens_param: max_tokens
    verified: measured
    output: {json_object: true}
    tools:
      supported: true
      format: native
      tool_choice_values: [auto]
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)

	pl := mustPlan(t, c, ChatRequest{
		Model:    "jsonobj",
		Messages: []Message{User("hi")},
		ResponseFormat: ResponseFormat{
			Kind:   FormatJSONSchema,
			Name:   "answer",
			Schema: map[string]any{"type": "object"},
			Strict: true,
		},
	}, false)
	if pl.req.ResponseFormat.Kind != FormatJSONObject {
		t.Errorf("response format = %q, want the recorded downgrade to %q",
			pl.req.ResponseFormat.Kind, FormatJSONObject)
	}
	if pl.req.ResponseFormat.Strict {
		t.Error("strict survived a downgrade to json_object")
	}

	pl = mustPlan(t, c, ChatRequest{
		Model:      "jsonobj",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceRequired},
	}, false)
	if pl.req.ToolChoice.Mode != ToolChoiceAuto {
		t.Errorf("tool choice = %q, want the recorded relaxation to %q",
			pl.req.ToolChoice.Mode, ToolChoiceAuto)
	}

	pl = mustPlan(t, c, ChatRequest{
		Model:      "jsonobj",
		Messages:   []Message{User("hi")},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto},
	}, false)
	if pl.req.ToolChoice.Mode != ToolChoiceUnset {
		t.Error("a tool choice with no tools survived into the plan")
	}
}

// A strict schema the model DOES support keeps its flag. The drop above must be
// the coercion, not a blanket rule.
func TestPlan_StrictSchemaSurvivesWhereSupported(t *testing.T) {
	pl := mustPlan(t, testClient(t, Default()), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
		ResponseFormat: ResponseFormat{
			Kind:   FormatJSONSchema,
			Name:   "answer",
			Schema: map[string]any{"type": "object"},
			Strict: true,
		},
	}, false)
	if !pl.req.ResponseFormat.Strict || pl.req.ResponseFormat.Kind != FormatJSONSchema {
		t.Errorf("response format = %+v, want json_schema with strict kept", pl.req.ResponseFormat)
	}
}

// Whether the caller's temperature is inert depends on whether the request will
// actually reason, which BestEffort can change after the fact: a demoted
// ReasoningOff leaves a thinking model, so the warning must still fire.
func TestPlan_InertSamplingWarnsAfterADemotedReasoningOff(t *testing.T) {
	_, warnings, err := testClient(t, Default()).plan(ChatRequest{
		Model:       "mimo-v2.5-pro",
		Messages:    []Message{User("hi")},
		Reasoning:   ReasoningOff(),
		Temperature: f64(0.2),
	}, false)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	// mimo CAN be disabled, so this request really does switch thinking off and
	// the temperature is honoured: no inert warning.
	for _, w := range warnings {
		if w.Feature == "temperature" {
			t.Errorf("temperature warned while thinking is off: %v", w)
		}
	}

	// Now the same shape against a model that cannot be disabled AND overrides
	// sampling while thinking. BestEffort drops the off-request, so the model
	// thinks and the temperature is inert after all.
	reg := registryFrom(t, `profiles:
  - id: always
    wire_model_id: always
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: false
      control: effort
      effort_values: [low, high]
    temperature: {supported: true, inert_while_reasoning: true}
    streaming: {supported: true, accepts_stream_options: true}
`)
	_, warnings, err = testClient(t, reg).plan(ChatRequest{
		Model:       "always",
		Messages:    []Message{User("hi")},
		Reasoning:   ReasoningOff(),
		Temperature: f64(0.2),
		BestEffort:  true,
	}, false)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	var sawInert bool
	for _, w := range warnings {
		if w.Feature == "temperature" {
			sawInert = true
		}
	}
	if !sawInert {
		t.Errorf("a demoted reasoning-off left the model thinking, so temperature is inert; "+
			"warnings = %v", warnings)
	}
}

// stream_options is sent only where the endpoint needs it AND accepts it. A
// gateway that strips the usage chunk from a client which did not ask forces
// needs_include_usage on for its own route.
func TestPlan_IncludeUsageFollowsTheProfile(t *testing.T) {
	pl := mustPlan(t, testClient(t, Default()), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	}, true)
	if pl.includeUsage {
		t.Error("include_usage set for an endpoint that reports usage unasked")
	}

	reg := registryFrom(t, `profiles:
  - id: needs
    wire_model_id: needs
    max_tokens_param: max_tokens
    verified: source-derived
    streaming: {supported: true, needs_include_usage: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)
	if pl = mustPlan(t, c, ChatRequest{Model: "needs", Messages: []Message{User("hi")}}, true); !pl.includeUsage {
		t.Error("include_usage not set for an endpoint that reports nothing without it")
	}
	// Not a streaming call: there is no usage chunk to ask for.
	if pl = mustPlan(t, c, ChatRequest{Model: "needs", Messages: []Message{User("hi")}}, false); pl.includeUsage {
		t.Error("include_usage set on a non-streaming call")
	}
}

// A model that does not stream refuses the streaming METHOD, and the refusal is
// not demotable: the caller chose a method, and there is nothing to demote a
// method to.
func TestPlan_NonStreamingModelRefusesTheStreamMethod(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: sync
    wire_model_id: sync
    max_tokens_param: max_tokens
    verified: source-derived
`)
	_, _, err := testClient(t, reg).plan(ChatRequest{
		Model:      "sync",
		Messages:   []Message{User("hi")},
		BestEffort: true,
	}, true)
	if err == nil {
		t.Fatal("streaming a non-streaming model should be refused")
	}
	var ue *UnsupportedError
	if !errors.As(err, &ue) {
		t.Fatalf("error = %T %v, want *UnsupportedError", err, err)
	}
	if ue.Demotable {
		t.Error("the streaming refusal is marked demotable; BestEffort cannot rescue a method choice")
	}
}

// A cap above the model's output limit is a warning, not a refusal: every
// endpoint measured clamps silently, so the call works and only the caller's
// expectation is wrong.
func TestPlan_CapAboveTheModelLimitWarns(t *testing.T) {
	huge := 1 << 20
	pl, warnings, err := testClient(t, Default()).plan(ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hi")},
		MaxTokens: &huge,
	}, false)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	var saw bool
	for _, w := range warnings {
		if w.Feature == "max_tokens" {
			saw = true
		}
	}
	if !saw {
		t.Errorf("no warning for a cap above the model's limit: %v", warnings)
	}
	// Still sent verbatim: rewriting it would hide the profile's limit behind a
	// number the caller never chose.
	if pl.req.MaxTokens == nil || *pl.req.MaxTokens != huge {
		t.Errorf("MaxTokens = %v, want it passed through", pl.req.MaxTokens)
	}
}
