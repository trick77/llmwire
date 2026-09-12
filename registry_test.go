package llmwire

import (
	"errors"
	"strings"
	"testing"
)

// The embedded registry must load. This is the test that makes shipping a
// malformed profile impossible: the data is compiled in, so a failure here is a
// failure everywhere.
func TestDefault_EmbeddedProfilesLoad(t *testing.T) {
	reg := Default()
	models := reg.Models()
	if len(models) == 0 {
		t.Fatal("embedded registry is empty")
	}
	for _, id := range models {
		p, err := reg.Lookup(id)
		if err != nil {
			t.Fatalf("Lookup(%q) failed on a registered id: %v", id, err)
		}
		if err := p.validate(); err != nil {
			t.Errorf("registered profile fails its own validation: %v", err)
		}
		if p.WireModelID == "" {
			t.Errorf("%s: wire_model_id is empty after resolution", id)
		}
	}
	t.Logf("registry: %s", strings.Join(models, ", "))
}

// The findings are the source of truth for these bits. If a future edit
// contradicts a measurement, this test says so — naming the probe that
// established it, so the disagreement can be re-measured rather than argued.
func TestDefault_MeasuredFactsMatchFindings(t *testing.T) {
	reg := Default()

	t.Run("glm-5.3-flash cannot disable thinking", func(t *testing.T) {
		p := mustLookup(t, reg, "glm-5.3-flash")
		if p.Reasoning.CanBeDisabled {
			t.Error("can_be_disabled is true; ZaiThinkingCanBeDisabled measured a " +
				"400/1210 refusal. The vendor doc says otherwise and is wrong.")
		}
		if !p.Reasoning.EnabledByDefault {
			t.Error("enabled_by_default should be true: the model always reasons")
		}
	})

	t.Run("glm-5.3-flash accepts exactly low high max", func(t *testing.T) {
		p := mustLookup(t, reg, "glm-5.3-flash")
		want := []string{"low", "high", "max"}
		if strings.Join(p.Reasoning.EffortValues, ",") != strings.Join(want, ",") {
			t.Errorf("effort_values = %v, want %v (measured by ZaiReasoningEffortValues; "+
				"none, minimal, medium and xhigh were all refused)", p.Reasoning.EffortValues, want)
		}
		for _, refused := range []string{"none", "minimal", "medium", "xhigh"} {
			if p.Reasoning.Accepts(refused) {
				t.Errorf("effort_values contains %q, which the endpoint refuses", refused)
			}
		}
	})

	t.Run("glm-5.3-flash uses max_tokens", func(t *testing.T) {
		p := mustLookup(t, reg, "glm-5.3-flash")
		if p.MaxTokensParam != ParamMaxTokens {
			t.Errorf("max_tokens_param = %q, want %q. The other name is ACCEPTED and "+
				"IGNORED here: a cap of 16 returned 481 tokens with finish_reason stop.",
				p.MaxTokensParam, ParamMaxTokens)
		}
	})

	t.Run("json_schema is supported despite the docs", func(t *testing.T) {
		for _, id := range []string{"glm-5.3-flash", "mimo-v2.5-pro"} {
			p := mustLookup(t, reg, id)
			if !p.Output.JSONSchema {
				t.Errorf("%s: json_schema should be true; ResponseFormatSupport measured "+
					"it accepted and honoured, though the vendor docs omit it", id)
			}
		}
	})

	t.Run("neither endpoint needs stream_options", func(t *testing.T) {
		for _, id := range []string{"glm-5.3-flash", "mimo-v2.5-pro"} {
			p := mustLookup(t, reg, id)
			if p.Streaming.NeedsIncludeUsage {
				t.Errorf("%s: needs_include_usage should be false; usage arrives unasked", id)
			}
			if !p.Streaming.AcceptsStreamOptions {
				t.Errorf("%s: accepts_stream_options should be true; it was accepted", id)
			}
		}
	})

	t.Run("the MiMo vision split", func(t *testing.T) {
		pro := mustLookup(t, reg, "mimo-v2.5-pro")
		if pro.Vision {
			t.Error("mimo-v2.5-pro: vision should be false; an image part returns 404")
		}
		sibling := mustLookup(t, reg, "mimo-v2.5")
		if !sibling.Vision {
			t.Error("mimo-v2.5: vision should be true; it accepts the same part")
		}
	})

	t.Run("MiMo returns native tool calls", func(t *testing.T) {
		for _, id := range []string{"mimo-v2.5-pro", "mimo-v2.5"} {
			p := mustLookup(t, reg, id)
			if p.Tools.Format != FormatNative {
				t.Errorf("%s: tools.format = %q, want %q (MiMoToolCallFormat)",
					id, p.Tools.Format, FormatNative)
			}
		}
	})
}

// Every profile must say how its facts were established, so a reader can tell a
// measurement from something read off a web page.
func TestDefault_EveryProfileDeclaresItsProvenance(t *testing.T) {
	reg := Default()
	for _, id := range reg.Models() {
		p := mustLookup(t, reg, id)
		if p.Verified != VerifiedMeasured && p.Verified != VerifiedSource {
			t.Errorf("%s: verified = %q", id, p.Verified)
		}
	}
}

// --- Lookup -------------------------------------------------------------------

// An unknown id must ERROR. Returning a zero profile would let every capability
// check read false, turning a typo into "this model supports nothing" — a
// plausible-looking answer that fails much later, somewhere unrelated.
func TestLookup_UnknownIDIsAnError(t *testing.T) {
	reg := Default()
	_, err := reg.Lookup("gpt-4-turbo-preview")
	if err == nil {
		t.Fatal("expected an error for an unregistered id")
	}
	var unknown *UnknownModelError
	if !errors.As(err, &unknown) {
		t.Fatalf("error is %T, want *UnknownModelError", err)
	}
	// The message has to name what IS available, or the reader is left guessing
	// at a typo.
	if !strings.Contains(err.Error(), "glm-5.3-flash") {
		t.Errorf("error does not list the known ids: %v", err)
	}
}

// Ids are matched exactly. Prefix matching is how a new model silently inherits
// an older family's quirks.
func TestLookup_IsExactNotPrefix(t *testing.T) {
	reg := Default()
	for _, near := range []string{"glm-5.3", "glm-5.3-flash-preview", "mimo-v2.5-pro-2026", "MIMO-V2.5"} {
		if _, err := reg.Lookup(near); err == nil {
			t.Errorf("Lookup(%q) resolved; ids must match exactly", near)
		}
	}
}

// --- loading and validation ----------------------------------------------------

func TestNewRegistry_RejectsMalformedDocuments(t *testing.T) {
	for _, tc := range []struct {
		name, doc, wantSubstr string
	}{
		{
			name:       "no entries",
			doc:        "profiles: []\n",
			wantSubstr: "no entries",
		},
		{
			name: "duplicate id",
			doc: `profiles:
  - {id: a, wire_model_id: a, max_tokens_param: max_tokens, verified: measured}
  - {id: a, wire_model_id: a, max_tokens_param: max_tokens, verified: measured}
`,
			wantSubstr: "duplicate profile id",
		},
		{
			name: "missing base",
			doc: `profiles:
  - {id: derived, base: nowhere}
`,
			wantSubstr: "does not exist",
		},
		{
			name: "unknown endpoint",
			doc: `profiles:
  - {id: a, endpoint: images, wire_model_id: a, verified: measured}
`,
			wantSubstr: "endpoint",
		},
		{
			name: "bad max_tokens_param",
			doc: `profiles:
  - {id: a, wire_model_id: a, max_tokens_param: maxTokens, verified: measured}
`,
			wantSubstr: "max_tokens_param",
		},
		{
			name: "unknown provenance",
			doc: `profiles:
  - {id: a, wire_model_id: a, max_tokens_param: max_tokens, verified: probably}
`,
			wantSubstr: "verified",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(tc.doc))
			if err == nil {
				t.Fatal("expected a load error")
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error = %v, want it to mention %q", err, tc.wantSubstr)
			}
		})
	}
}

// The inference this schema exists to prevent, checked both ways. Deriving
// "cannot be disabled" from the absence of "none" is what put a wrong value in a
// published catalogue for a model in this registry.
func TestNewRegistry_RejectsContradictoryReasoning(t *testing.T) {
	for _, tc := range []struct{ name, doc, wantSubstr string }{
		{
			name: "cannot disable yet offers none",
			doc: `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: false
      control: effort
      effort_values: [none, low, high]
`,
			wantSubstr: `contains "none"`,
		},
		{
			name: "can disable but offers no way to",
			doc: `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: effort
      effort_values: [low, high]
`,
			wantSubstr: `no "none"`,
		},
		{
			name: "effort control with no values",
			doc: `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
    reasoning: {supported: true, enabled_by_default: true, control: effort}
`,
			wantSubstr: "effort_values is empty",
		},
		{
			name: "unsupported yet configured",
			doc: `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
    reasoning: {supported: false, control: effort, effort_values: [low]}
`,
			wantSubstr: "unsupported",
		},
		{
			name: "default effort outside the accepted set",
			doc: `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      control: effort
      effort_values: [low, high]
      default_effort: max
`,
			wantSubstr: "default_effort",
		},
		{
			name: "reasons neither by default nor on request",
			doc: `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: false
      can_be_disabled: false
      control: toggle_object
`,
			wantSubstr: "never reasons",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(tc.doc))
			if err == nil {
				t.Fatal("expected a validation error")
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error = %v, want it to mention %q", err, tc.wantSubstr)
			}
		})
	}
}

func TestNewRegistry_RejectsContradictoryCapabilities(t *testing.T) {
	base := `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
`
	for _, tc := range []struct{ name, extra, wantSubstr string }{
		{
			name:       "xml tools claiming forced choice",
			extra:      "    tools: {supported: true, format: xml, tool_choice_values: [auto], supports_forced_choice: true}\n",
			wantSubstr: "supports_forced_choice",
		},
		{
			name:       "tools with no accepted tool_choice",
			extra:      "    tools: {supported: true, format: native}\n",
			wantSubstr: "tool_choice_values is empty",
		},
		{
			name:       "tool settings without tools",
			extra:      "    tools: {supported: false, tool_choice_values: [auto]}\n",
			wantSubstr: "tools are unsupported",
		},
		{
			name:       "strict schema without json_schema",
			extra:      "    output: {json_object: true, strict_schema: true}\n",
			wantSubstr: "strict_schema",
		},
		{
			name:       "needs a parameter the endpoint rejects",
			extra:      "    streaming: {supported: true, needs_include_usage: true, accepts_stream_options: false}\n",
			wantSubstr: "needs_include_usage",
		},
		{
			name:       "output larger than context",
			extra:      "    limits: {context: 1000, max_output: 2000}\n",
			wantSubstr: "exceeds context",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(base + tc.extra))
			if err == nil {
				t.Fatal("expected a validation error")
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error = %v, want it to mention %q", err, tc.wantSubstr)
			}
		})
	}
}

// An embeddings profile carrying chat capabilities is a copy-paste error that
// would otherwise surface as a confusing 400 from the wrong route.
func TestNewRegistry_EmbeddingsProfileRejectsChatCapabilities(t *testing.T) {
	doc := `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
    tools: {supported: true, format: native, tool_choice_values: [auto]}
`
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("expected a validation error")
	}
	// "must not declare", not merely "embeddings profile": every refusal an
	// embeddings profile can earn mentions the latter, including the missing
	// default_dimensions this fixture also lacks, so the looser check would pass
	// with the guard under test deleted.
	if !strings.Contains(err.Error(), "must not declare") {
		t.Errorf("error = %v, want the capability guard, not a later check", err)
	}
}

// --- composition ----------------------------------------------------------------

func TestNewRegistry_DerivedProfileInheritsAndOverridesRouting(t *testing.T) {
	doc := `profiles:
  - id: base-model
    wire_model_id: base-model
    max_tokens_param: max_completion_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: toggle_object
    tools: {supported: true, format: native, tool_choice_values: [auto]}
    output: {json_object: true, json_schema: true}
    streaming: {supported: true, accepts_stream_options: true}
    vision: true
  - id: gw/base-model
    base: base-model
    gateway: litellm
    wire_model_id: alias/base-model
    base_url_env: GW_BASE_URL
    api_key_env: GW_API_KEY
    streaming: {needs_include_usage: true}
`
	reg, err := NewRegistry([]byte(doc))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	d := mustLookup(t, reg, "gw/base-model")

	// Routing replaced.
	if d.WireModelID != "alias/base-model" {
		t.Errorf("wire_model_id = %q, want the gateway alias", d.WireModelID)
	}
	if d.BaseURLEnv != "GW_BASE_URL" || d.APIKeyEnv != "GW_API_KEY" {
		t.Errorf("routing not overridden: %+v", d)
	}
	if d.Gateway != "litellm" {
		t.Errorf("gateway = %q", d.Gateway)
	}
	// Capabilities inherited whole — this is what keeps fail-fast validation
	// working behind a proxy whose wire id says nothing about the model.
	if !d.Vision || !d.Tools.Supported || !d.Output.JSONSchema {
		t.Errorf("capabilities were not inherited: %+v", d)
	}
	if d.MaxTokensParam != ParamMaxCompletionTokens {
		t.Errorf("max_tokens_param = %q, want it inherited", d.MaxTokensParam)
	}
	if !d.Reasoning.CanBeDisabled || d.Reasoning.Control != ControlToggleObject {
		t.Errorf("reasoning was not inherited: %+v", d.Reasoning)
	}
	// The one capability a route may tighten.
	if !d.Streaming.NeedsIncludeUsage {
		t.Error("needs_include_usage should be forced on by the gateway")
	}
}

// A derived profile restating a capability is refused rather than merged,
// because an omitted YAML bool and an explicit false decode identically and a
// merge would have to guess which was meant.
func TestNewRegistry_DerivedProfileCannotRestateCapabilities(t *testing.T) {
	doc := `profiles:
  - id: base-model
    wire_model_id: base-model
    max_tokens_param: max_tokens
    verified: measured
    vision: true
  - id: gw/base-model
    base: base-model
    vision: false
`
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("expected the derived profile to be refused")
	}
	if !strings.Contains(err.Error(), "vision") {
		t.Errorf("error = %v, want it to name the offending key", err)
	}
	if !strings.Contains(err.Error(), "own profile") {
		t.Errorf("error = %v, want it to say what to do instead", err)
	}
}

// Single base, no chains: a chain makes "which rule applied" unanswerable.
func TestNewRegistry_RejectsBaseChains(t *testing.T) {
	doc := `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
  - id: b
    base: a
    wire_model_id: b
  - id: c
    base: b
    wire_model_id: c
`
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("expected a chained base to be refused")
	}
	if !strings.Contains(err.Error(), "no chains") {
		t.Errorf("error = %v", err)
	}
}

func mustLookup(t *testing.T, reg *Registry, id string) *Profile {
	t.Helper()
	p, err := reg.Lookup(id)
	if err != nil {
		t.Fatalf("Lookup(%q): %v", id, err)
	}
	return p
}

// String is what names a deployment in an error message, so it has to
// distinguish a model reached directly from the same model reached through a
// proxy under an alias.
func TestProfile_String(t *testing.T) {
	for _, tc := range []struct {
		name string
		p    Profile
		want string
	}{
		{
			name: "direct model",
			p:    Profile{ID: "glm-5.3-flash", WireModelID: "glm-5.3-flash"},
			want: "glm-5.3-flash",
		},
		{
			name: "alias behind a gateway",
			p:    Profile{ID: "gw/gpt", WireModelID: "alias/gpt", Gateway: "litellm"},
			want: "gw/gpt (wire: alias/gpt) via litellm",
		},
		{
			name: "alias with no gateway named",
			p:    Profile{ID: "a", WireModelID: "b"},
			want: "a (wire: b)",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.p.String(); got != tc.want {
				t.Errorf("String() = %q, want %q", got, tc.want)
			}
		})
	}
}

// The defaults applied at load are deliberately few: a default is only safe
// where every endpoint agrees, and elsewhere an absent value means "not
// measured". These are the ones that are safe.
func TestNewRegistry_AppliesOnlyTheSafeDefaults(t *testing.T) {
	doc := `profiles:
  - id: minimal
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: toggle_object
    streaming: {supported: true}
    tools: {supported: true, tool_choice_values: [auto]}
`
	reg, err := NewRegistry([]byte(doc))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	p := mustLookup(t, reg, "minimal")

	if p.Endpoint != EndpointChat {
		t.Errorf("endpoint = %q, want the chat default", p.Endpoint)
	}
	// wire_model_id defaults to the id, which is right for a direct model and
	// overridden for anything behind an alias.
	if p.WireModelID != "minimal" {
		t.Errorf("wire_model_id = %q, want it defaulted from the id", p.WireModelID)
	}
	// Every reasoning model measured streams on this field, and the parser reads
	// the other spelling too, so defaulting it is safe.
	if p.Reasoning.StreamField != "reasoning_content" {
		t.Errorf("stream_field = %q, want reasoning_content", p.Reasoning.StreamField)
	}
	if p.Tools.Format != FormatNative {
		t.Errorf("tools.format = %q, want the native default", p.Tools.Format)
	}
}

// Provenance defaults to the CAUTIOUS value: a profile that does not say how its
// facts were established has not been measured.
func TestNewRegistry_UnstatedProvenanceDefaultsToSourceDerived(t *testing.T) {
	doc := `profiles:
  - {id: a, wire_model_id: a, max_tokens_param: max_tokens}
`
	reg, err := NewRegistry([]byte(doc))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := mustLookup(t, reg, "a").Verified; got != VerifiedSource {
		t.Errorf("verified = %q, want %q — silence must never read as measured",
			got, VerifiedSource)
	}
}

// --- regressions from the Phase 1a review -------------------------------------

// A derived profile may set streaming and tools ONLY for the two tighten-only
// sub-keys. Admitting the whole mapping and trusting resolve to copy the fields
// it knows about silently DISCARDS the rest: a gateway saying it does not stream
// would load clean and resolve to streaming.supported=true — a profile
// asserting a capability the route does not have, which is the exact failure
// this package exists to prevent.
func TestNewRegistry_DerivedProfileCannotRestateNestedCapabilities(t *testing.T) {
	base := `profiles:
  - id: base-model
    wire_model_id: base-model
    max_tokens_param: max_tokens
    verified: measured
    tools: {supported: true, format: native, tool_choice_values: [auto]}
    streaming: {supported: true, accepts_stream_options: true}
  - id: gw/base-model
    base: base-model
`
	for _, tc := range []struct{ name, extra, wantKey string }{
		{"turning streaming off", "    streaming: {supported: false}\n", "streaming.supported"},
		{"withdrawing stream_options", "    streaming: {accepts_stream_options: false}\n", "streaming.accepts_stream_options"},
		{"turning tools off", "    tools: {supported: false}\n", "tools.supported"},
		{"changing the tool format", "    tools: {format: xml}\n", "tools.format"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(base + tc.extra))
			if err == nil {
				t.Fatal("expected the derived profile to be refused")
			}
			if !strings.Contains(err.Error(), tc.wantKey) {
				t.Errorf("error = %v, want it to name %q", err, tc.wantKey)
			}
		})
	}
}

// The two sub-keys that ARE allowed must still work.
func TestNewRegistry_DerivedProfileMayTightenTheTwoAllowedSubKeys(t *testing.T) {
	doc := `profiles:
  - id: base-model
    wire_model_id: base-model
    max_tokens_param: max_tokens
    verified: measured
    tools: {supported: true, format: native, tool_choice_values: [auto]}
    streaming: {supported: true, accepts_stream_options: true}
  - id: gw/base-model
    base: base-model
    streaming: {needs_include_usage: true}
    tools: {recover_inline_markup: true}
`
	reg, err := NewRegistry([]byte(doc))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	d := mustLookup(t, reg, "gw/base-model")
	if !d.Streaming.NeedsIncludeUsage {
		t.Error("needs_include_usage should have been tightened on")
	}
	if !d.Tools.RecoverInlineMarkup {
		t.Error("recover_inline_markup should have been tightened on")
	}
	// And the rest of both mappings still comes from the base.
	if !d.Streaming.Supported || !d.Streaming.AcceptsStreamOptions || !d.Tools.Supported {
		t.Errorf("base capabilities were lost: %+v %+v", d.Streaming, d.Tools)
	}
}

// Defaults must be applied to a base BEFORE anything inherits from it.
// Resolving first and defaulting afterwards looks equivalent and is not: the
// derived entry would inherit an empty wire_model_id and then fill it from its
// OWN id, putting the gateway's registry key on the wire as the model name.
func TestNewRegistry_BaseDefaultsApplyBeforeInheritance(t *testing.T) {
	doc := `profiles:
  - id: real-model
    max_tokens_param: max_tokens
    verified: measured
  - id: gw/real-model
    base: real-model
    gateway: litellm
`
	reg, err := NewRegistry([]byte(doc))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	d := mustLookup(t, reg, "gw/real-model")
	if d.WireModelID != "real-model" {
		t.Errorf("wire_model_id = %q, want %q inherited from the base; "+
			"defaulting it from the derived id puts the registry key on the wire",
			d.WireModelID, "real-model")
	}
}

// An unknown key must fail the load. Without strict decoding a typo silently
// becomes a zero value, so a mistyped capability reads as absent — the worst
// available failure for a package whose purpose is that wrong assumptions fail
// loudly.
func TestNewRegistry_UnknownKeyIsRejected(t *testing.T) {
	for _, tc := range []struct{ name, doc string }{
		{
			name: "misspelled capability",
			doc: `profiles:
  - {id: a, wire_model_id: a, max_tokens_param: max_tokens, verified: measured, visoin: true}
`,
		},
		{
			name: "misspelled nested key",
			doc: `profiles:
  - id: a
    wire_model_id: a
    max_tokens_param: max_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: effort
      effort_valuez: [low, none]
`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(tc.doc))
			if err == nil {
				t.Fatal("expected an unknown key to fail the load")
			}
		})
	}
}

// The embeddings guard covers every generation knob, not a sample of them: a
// guard that covers most of the set reads as complete while leaving a hole.
func TestNewRegistry_EmbeddingsGuardIsComplete(t *testing.T) {
	base := `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
`
	for _, tc := range []struct{ name, extra string }{
		{"top_p", "    top_p: {supported: true}\n"},
		{"temperature", "    temperature: {supported: true}\n"},
		{"vision", "    vision: true\n"},
		{"structured output", "    output: {json_object: true}\n"},
		{"reasoning", "    reasoning: {supported: true, enabled_by_default: true, can_be_disabled: true, control: toggle_object}\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(base + tc.extra))
			if err == nil {
				t.Fatalf("an embeddings profile declaring %s should be refused", tc.name)
			}
			// The guard's own wording, so a deleted case cannot be masked by the
			// missing-default_dimensions refusal this fixture would otherwise
			// fall through to — which also says "embeddings profile".
			if !strings.Contains(err.Error(), "must not declare") {
				t.Errorf("error = %v, want the %s guard itself", err, tc.name)
			}
		})
	}
}

// Limits are checked on embeddings profiles too, which an early return skipped.
func TestNewRegistry_EmbeddingsLimitsAreStillChecked(t *testing.T) {
	// A valid embedding block, so the limits check is what this fixture reaches:
	// without it the profile is refused earlier, for a missing default_dimensions,
	// and the test would pass while proving nothing about limits.
	doc := `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
    embedding: {default_dimensions: 1536}
    limits: {context: 100, max_output: 200}
`
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("expected the limits check to run on an embeddings profile")
	}
	if !strings.Contains(err.Error(), "exceeds context") {
		t.Errorf("error = %v", err)
	}
}

// The registry is cached for the life of the process and shared by every
// caller, so a returned profile must be a copy: otherwise one caller's
// experiment becomes everyone else's configuration.
func TestLookup_ReturnsACopyNotTheSharedProfile(t *testing.T) {
	reg := Default()

	first := mustLookup(t, reg, "mimo-v2.5")
	if !first.Vision {
		t.Fatal("fixture assumption: mimo-v2.5 has vision")
	}
	first.Vision = false
	first.Reasoning.EffortValues = append(first.Reasoning.EffortValues, "invented")
	first.Tools.ToolChoiceValues = append(first.Tools.ToolChoiceValues, "required")

	second := mustLookup(t, reg, "mimo-v2.5")
	if !second.Vision {
		t.Error("mutating a returned profile changed the registry")
	}
	for _, v := range second.Reasoning.EffortValues {
		if v == "invented" {
			t.Error("slice backing storage is shared with the registry")
		}
	}
	for _, v := range second.Tools.ToolChoiceValues {
		if v == "required" {
			t.Error("tool_choice slice backing storage is shared with the registry")
		}
	}
}

// Provenance is not inherited. A derived profile describes a different ROUTE,
// which no probe of the base ever exercised, so carrying "measured" across would
// have it claim evidence that does not exist for it.
func TestNewRegistry_ProvenanceIsNotInherited(t *testing.T) {
	doc := `profiles:
  - id: real-model
    wire_model_id: real-model
    max_tokens_param: max_tokens
    verified: measured
  - id: gw/real-model
    base: real-model
    gateway: litellm
    wire_model_id: alias/real-model
`
	reg, err := NewRegistry([]byte(doc))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := mustLookup(t, reg, "gw/real-model").Verified; got != VerifiedSource {
		t.Errorf("derived verified = %q, want %q: the route was never probed", got, VerifiedSource)
	}
	if got := mustLookup(t, reg, "real-model").Verified; got != VerifiedMeasured {
		t.Errorf("base verified = %q, want it unchanged", got)
	}
}
