package llmwire

import (
	"errors"
	"strings"
	"testing"
)

// Regressions from the Phase 1b review. Each of these passed a naive
// implementation and failed the real one.

// --- BestEffort must not rescue a malformed request -----------------------------

// A malformed request is not a capability mismatch: there is nothing to "send
// instead", and no endpoint accepts it. Demoting these would hand back a nil
// error and put the broken request on the wire — turning a local failure into
// exactly the 400 this package exists to catch first.
func TestValidate_BestEffortDoesNotRescueAMalformedRequest(t *testing.T) {
	c := testClient(t, nil)
	for _, tc := range []struct {
		name       string
		req        ChatRequest
		wantSubstr string
	}{
		{
			name:       "no messages",
			req:        ChatRequest{Model: "mimo-v2.5-pro", BestEffort: true},
			wantSubstr: "at least one message",
		},
		{
			name: "tool result with nothing to tie it to",
			req: ChatRequest{
				Model:      "mimo-v2.5-pro",
				Messages:   []Message{{Role: RoleTool, Text: "result"}},
				BestEffort: true,
			},
			wantSubstr: "ToolCallID",
		},
		{
			name: "nameless tool",
			req: ChatRequest{
				Model:      "mimo-v2.5-pro",
				Messages:   []Message{User("hi")},
				Tools:      []Tool{{Name: ""}},
				BestEffort: true,
			},
			wantSubstr: "no name",
		},
		{
			name: "function choice without a name",
			req: ChatRequest{
				Model:      "mimo-v2.5-pro",
				Messages:   []Message{User("hi")},
				Tools:      []Tool{{Name: "search"}},
				ToolChoice: ToolChoice{Mode: ToolChoiceFunction},
				BestEffort: true,
			},
			wantSubstr: "needs the function's name",
		},
		{
			name: "schema format without a schema",
			req: ChatRequest{
				Model:          "mimo-v2.5-pro",
				Messages:       []Message{User("hi")},
				ResponseFormat: ResponseFormat{Kind: FormatJSONSchema, Name: "x"},
				BestEffort:     true,
			},
			wantSubstr: "needs a schema",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := c.Validate(tc.req)
			if err == nil {
				t.Fatal("BestEffort must not suppress a malformed request")
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error = %v, want it to mention %q", err, tc.wantSubstr)
			}
			// And the advice must not appear, since following it would suppress
			// the error and send something broken.
			if strings.Contains(err.Error(), "BestEffort") {
				t.Errorf("a malformed request must not advise BestEffort: %v", err)
			}
		})
	}
}

// A demoted refusal must not stop the remaining checks, or a BestEffort request
// reaches the wire carrying a second problem nobody looked for.
func TestValidate_DemotedRefusalDoesNotSkipLaterChecks(t *testing.T) {
	c := testClient(t, nil)
	image := Message{Role: RoleUser, Parts: []Part{{Kind: PartImage, URL: "data:image/png;base64,AA"}}}

	_, err := c.Validate(ChatRequest{
		Model:      "mimo-v2.5-pro", // text-only: the image is a demotable refusal
		Messages:   []Message{image, {Role: RoleTool, Text: "orphan"}},
		BestEffort: true,
	})
	if err == nil {
		t.Fatal("the malformed second message must still be caught after the demoted refusal")
	}
	if !strings.Contains(err.Error(), "ToolCallID") {
		t.Errorf("error = %v, want the structural problem in message 1", err)
	}
}

// --- the advice must be true -----------------------------------------------------

// An endpoint mismatch is not demotable: there is no nearest supported request
// for a model that does not serve this route.
func TestValidate_EndpointMismatchDoesNotAdviseBestEffort(t *testing.T) {
	c := testClient(t, nil)
	_, err := c.Validate(ChatRequest{
		Model:      "text-embedding-3-small",
		Messages:   []Message{User("hi")},
		BestEffort: true,
	})
	if err == nil {
		t.Fatal("BestEffort must not make an embeddings model take a chat request")
	}
	if strings.Contains(err.Error(), "BestEffort") {
		t.Errorf("the advice does not work here and must not be offered: %v", err)
	}
}

// Where the advice IS offered, it must actually work.
func TestValidate_AdviceIsOfferedOnlyWhereItWorks(t *testing.T) {
	c := testClient(t, nil)
	req := ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningOff(),
	}
	_, err := c.Validate(req)
	if err == nil {
		t.Fatal("expected a refusal")
	}
	var unsup *UnsupportedError
	if !errors.As(err, &unsup) || !unsup.Demotable {
		t.Fatalf("error should be marked demotable: %+v", unsup)
	}
	if !strings.Contains(err.Error(), "BestEffort") {
		t.Errorf("a demotable refusal should offer the advice: %v", err)
	}
	// Following it must work.
	req.BestEffort = true
	if _, err := c.Validate(req); err != nil {
		t.Errorf("the advice did not work: %v", err)
	}
}

// --- the inert-parameter warning follows the REQUEST ------------------------------

// Reading the reasoning state off the profile alone gets it wrong in both
// directions. This is the direction that produces a false warning.
func TestValidate_NoInertWarningWhenThinkingWasSwitchedOff(t *testing.T) {
	c := testClient(t, nil)
	temp := 0.2
	warnings, err := c.Validate(ChatRequest{
		Model:       "mimo-v2.5-pro",
		Messages:    []Message{User("hi")},
		Reasoning:   ReasoningOff(),
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, w := range warnings {
		if w.Feature == "temperature" {
			t.Errorf("thinking was switched off, so the temperature WILL take effect; "+
				"warning is wrong: %v", w)
		}
	}
}

// And this is the direction that misses a real one: a model off by default,
// switched on by the caller, whose sampling parameters are then discarded.
func TestValidate_InertWarningAppearsWhenThinkingWasSwitchedOn(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: offbydefault
    wire_model_id: offbydefault
    max_tokens_param: max_completion_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: false
      can_be_disabled: true
      control: effort
      effort_values: [none, low, high]
    temperature: {supported: true, inert_while_reasoning: true}
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)
	temp := 0.2

	// Switched on: the warning must appear.
	warnings, err := c.Validate(ChatRequest{
		Model:       "offbydefault",
		Messages:    []Message{User("hi")},
		Reasoning:   ReasoningEffort("high"),
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) == 0 {
		t.Fatal("thinking was switched on, so the temperature is discarded: expected a warning")
	}

	// Left at its default (off): no warning.
	warnings, err = c.Validate(ChatRequest{
		Model:       "offbydefault",
		Messages:    []Message{User("hi")},
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("this model does not think by default, so nothing is discarded: %v", warnings)
	}

	// Explicitly "none" counts as off too.
	warnings, err = c.Validate(ChatRequest{
		Model:       "offbydefault",
		Messages:    []Message{User("hi")},
		Reasoning:   ReasoningEffort("none"),
		Temperature: &temp,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf(`effort "none" means no thinking, so nothing is discarded: %v`, warnings)
	}
}

// --- tool_choice none is a change of meaning ---------------------------------------

// Every other mode asks the model to PREFER a tool; "none" asks it not to call
// one at all. Relaxing that to "auto" permits exactly what the caller ruled out.
func TestValidate_ToolChoiceNoneIsRefusedNotRelaxed(t *testing.T) {
	c := testClient(t, nil)
	_, err := c.Validate(ChatRequest{
		Model:      "mimo-v2.5-pro",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceNone},
	})
	if err == nil {
		t.Fatal(`tool_choice "none" must not be silently relaxed to "auto"`)
	}
	if !strings.Contains(err.Error(), "omit Tools") {
		t.Errorf("the refusal should name the workaround: %v", err)
	}
}

// --- reasoning ---------------------------------------------------------------------

// A model that never reasons already satisfies "do not reason". Refusing would
// force model-agnostic callers to branch per model to ask for what they are
// already getting.
func TestValidate_ReasoningOffOnANonReasoningModelIsANoOp(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: noreasoning
    wire_model_id: noreasoning
    max_tokens_param: max_completion_tokens
    verified: measured
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)
	warnings, err := c.Validate(ChatRequest{
		Model:     "noreasoning",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningOff(),
	})
	if err != nil {
		t.Fatalf("asking a non-reasoning model not to reason should be a no-op: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("and should be silent: %v", warnings)
	}

	// Asking such a model TO reason is still a refusal.
	if _, err := c.Validate(ChatRequest{
		Model:     "noreasoning",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningEffort("high"),
	}); err == nil {
		t.Error("asking a non-reasoning model to reason should be refused")
	}
}

// A control mismatch must name the constructor that would work and must not leak
// the profile's internal constant name into a caller-facing message.
func TestValidate_ControlMismatchIsActionable(t *testing.T) {
	// A toggle model WITHOUT effort_values: the switch is its only knob. The
	// shipped MiMo profiles take levels too, so the shape is spelled out here.
	c := testClient(t, registryFrom(t, `
profiles:
  - id: toggle-only
    wire_model_id: toggle-only
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: toggle_object
`))
	_, err := c.Validate(ChatRequest{
		Model:     "toggle-only",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningEffort("high"),
	})
	if err == nil {
		t.Fatal("expected a refusal")
	}
	msg := err.Error()
	if strings.Contains(msg, "toggle_object") {
		t.Errorf("the internal constant leaked into a caller-facing message: %v", msg)
	}
	if !strings.Contains(msg, "on/off switch") {
		t.Errorf("the message should describe the control in caller terms: %v", msg)
	}
	if !strings.Contains(msg, "ReasoningOff()") {
		t.Errorf("the message should name the constructor that works: %v", msg)
	}
}

// --- strictness is checked against the field it describes -----------------------------

// StrictSchema describes whether "strict": true is accepted inside a json_schema
// block, so it gates ResponseFormat.Strict — not a tool's Strict flag, which is a
// different property no profile measures.
func TestValidate_StrictSchemaGatesTheResponseFormatNotTools(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: nostrict
    wire_model_id: nostrict
    max_tokens_param: max_completion_tokens
    verified: measured
    output: {json_object: true, json_schema: true, strict_schema: false}
    tools: {supported: true, format: native, tool_choice_values: [auto]}
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)

	// A strict RESPONSE FORMAT warns.
	warnings, err := c.Validate(ChatRequest{
		Model:    "nostrict",
		Messages: []Message{User("hi")},
		ResponseFormat: ResponseFormat{
			Kind: FormatJSONSchema, Name: "a",
			Schema: map[string]any{"type": "object"}, Strict: true,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) != 1 || warnings[0].Feature != "response_format" {
		t.Fatalf("warnings = %v, want one about response_format", warnings)
	}

	// A strict TOOL does not, because the flag describes something else.
	warnings, err = c.Validate(ChatRequest{
		Model:    "nostrict",
		Messages: []Message{User("hi")},
		Tools:    []Tool{{Name: "search", Strict: true}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("tool strictness is a different, unmeasured property and must not be "+
			"judged by the response-format flag: %v", warnings)
	}
}
