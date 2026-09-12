package llmwire

import (
	"strings"
	"testing"
)

// A price nobody can audit is worse than no price, so every rule below fails the
// LOAD. The embedded table is compiled in: a rule that fired at call time would
// fire in production, on a figure someone was about to bill against.

const chatHead = `profiles:
  - id: m
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
`

const goodProvenance = `      source_url: https://vendor.example/pricing
      verified_on: 2026-09-12
`

func TestCostBlock_LoadRefusals(t *testing.T) {
	for _, tc := range []struct{ name, doc, want string }{
		{
			name: "no source_url",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      verified_on: 2026-09-12
`,
			want: "source_url is required",
		},
		{
			name: "source_url is not https",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      source_url: http://vendor.example/pricing
      verified_on: 2026-09-12
`,
			want: "must be an https URL",
		},
		{
			name: "no verified_on",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      source_url: https://vendor.example/pricing
`,
			want: "verified_on is required",
		},
		{
			name: "verified_on is not a date",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      source_url: https://vendor.example/pricing
      verified_on: last Tuesday
`,
			want: "not a YYYY-MM-DD date",
		},
		{
			name: "a chat lane is missing",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      output: 0.50
` + goodProvenance,
			want: "cache_write rate is required",
		},
		{
			name: "a lane is negative",
			doc: chatHead + `    cost:
      input: -0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance,
			want: "input rate is negative",
		},
		{
			// The transposition this catches is real: cache_read and input are
			// adjacent in every vendor table and differ by two orders of
			// magnitude, so swapping them looks plausible and overcharges every
			// cached call.
			name: "cache_read is dearer than input",
			doc: chatHead + `    cost:
      input: 0.03
      cache_read: 0.15
      cache_write: 0
      output: 0.50
` + goodProvenance,
			want: "dearer than input",
		},
		{
			name: "scientific notation",
			doc: chatHead + `    cost:
      input: 1.5e-01
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance,
			want: "not scientific notation",
		},
		{
			name: "finer than nano",
			doc: chatHead + `    cost:
      input: 0.0000000001
      cache_read: 0.0000000001
      cache_write: 0
      output: 0.50
` + goodProvenance,
			want: "finer than nano-USD",
		},
		{
			name: "not a number",
			doc: chatHead + `    cost:
      input: cheap
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance,
			want: "not a decimal number",
		},
		{
			name: "unknown cost sub-key",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      discount: 0.5
` + goodProvenance,
			want: "field discount not found",
		},
		{
			name: "tier with no name",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      tiers:
        - {min_input_tokens: 272000, multiplier_permille: 2000}
` + goodProvenance,
			want: "has no name",
		},
		{
			name: "tier cheaper than list",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      tiers:
        - {name: long, min_input_tokens: 272000, multiplier_permille: 900}
` + goodProvenance,
			want: "under-report spend",
		},
		{
			name: "two tiers share a threshold",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      tiers:
        - {name: long, min_input_tokens: 272000, multiplier_permille: 2000}
        - {name: longer, min_input_tokens: 272000, multiplier_permille: 3000}
` + goodProvenance,
			want: "share min_input_tokens",
		},
		{
			name: "window with an unknown zone",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      windows:
        - {name: peak, zone: Mars/Olympus, from: "14:00", to: "18:00", multiplier_permille: 500}
` + goodProvenance,
			want: "zone \"Mars/Olympus\"",
		},
		{
			name: "window crossing midnight",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      windows:
        - {name: night, zone: Asia/Shanghai, from: "22:00", to: "06:00", multiplier_permille: 800}
` + goodProvenance,
			want: "written as two windows",
		},
		{
			name: "window with a bad day",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      windows:
        - {name: peak, zone: Asia/Singapore, days: [Wensday], from: "14:00", to: "18:00", multiplier_permille: 500}
` + goodProvenance,
			want: "is not Mon..Sun",
		},
		{
			name: "overlapping windows",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      windows:
        - {name: a, zone: Asia/Singapore, from: "14:00", to: "18:00", multiplier_permille: 500}
        - {name: b, zone: Asia/Singapore, from: "17:00", to: "20:00", multiplier_permille: 800}
` + goodProvenance,
			want: "overlap",
		},
		{
			name: "outside multiplier with no window",
			doc: chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      outside_windows_permille: 500
` + goodProvenance,
			want: "unconditional multiplier",
		},
		{
			name: "embeddings profile with an output lane",
			doc: `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
    embedding: {default_dimensions: 1536}
    cost:
      input: 0.02
      output: 0.10
` + goodProvenance,
			want: "must not declare a output rate",
		},
		{
			name: "embeddings profile with no input lane",
			doc: `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
    embedding: {default_dimensions: 1536}
    cost:
` + goodProvenance,
			want: "input rate is required",
		},
		{
			name: "gateway profile stating its own rate",
			doc: `profiles:
  - id: g
    gateway: litellm
    wire_model_id: g
    max_tokens_param: max_tokens
    verified: source-derived
    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance,
			want: "priced from its response header",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRegistry([]byte(tc.doc))
			if err == nil {
				t.Fatalf("expected the load to fail")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v\nwant it to mention %q", err, tc.want)
			}
		})
	}
}

// A derived profile may not restate a price. A rate belongs to the model and the
// vendor, never to the route, and the derived-key allowlist is what enforces it.
func TestCostBlock_DerivedProfileCannotRestateCost(t *testing.T) {
	doc := chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance + `  - id: m-via
    base: m
    gateway: litellm
    cost:
      input: 0.01
      cache_read: 0.001
      cache_write: 0
      output: 0.02
` + goodProvenance
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("a derived profile setting cost should be refused")
	}
	if !strings.Contains(err.Error(), "cost") {
		t.Errorf("error = %v", err)
	}
}

// A gateway route inherits no price either: the proxy reports real spend, and a
// list rate carried across would be a confident figure for a call nobody priced.
func TestCostBlock_GatewayInheritsNoCost(t *testing.T) {
	doc := chatHead + `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance + `  - id: m-via
    base: m
    gateway: litellm
    wire_model_id: proxy/m
`
	reg := registryFrom(t, doc)
	p := mustLookup(t, reg, "m-via")
	if p.Cost != nil {
		t.Fatalf("gateway route inherited a cost block: %+v", p.Cost)
	}
	direct := mustLookup(t, reg, "m")
	if direct.Cost == nil {
		t.Fatal("the base lost its own cost block")
	}
}

// The decimal spelling in the file is converted from its literal TEXT, never
// through float64: 0.0036 has no exact binary form, so a float path would make
// the stored integer depend on the rounding of the value under test.
func TestRate_DecimalTextConvertsExactly(t *testing.T) {
	doc := chatHead + `    cost:
      input: 0.435
      cache_read: 0.0036
      cache_write: 0
      output: 0.87
` + goodProvenance
	p := mustLookup(t, registryFrom(t, doc), "m")
	for _, tc := range []struct {
		lane string
		got  *rate
		want rate
	}{
		{"input", p.Cost.Input, 435_000_000},
		{"cache_read", p.Cost.CacheRead, 3_600_000},
		{"cache_write", p.Cost.CacheWrite, 0},
		{"output", p.Cost.Output, 870_000_000},
	} {
		if tc.got == nil {
			t.Fatalf("%s lane is nil", tc.lane)
		}
		if *tc.got != tc.want {
			t.Errorf("%s = %d, want %d nano-USD per 1M tokens", tc.lane, *tc.got, tc.want)
		}
	}
}

// Every shipped rate is checked against the vendor page it cites, in the unit
// the file claims. A transcription slip here is a billing error nothing else
// catches.
func TestDefault_ShippedRatesMatchTheVendorPages(t *testing.T) {
	reg := Default()
	for _, tc := range []struct {
		id                                string
		input, cacheRead, cacheWrite, out rate
		hasOutput                         bool
	}{
		{id: "glm-5.3-flash", input: 150_000_000, cacheRead: 30_000_000, cacheWrite: 0, out: 500_000_000, hasOutput: true},
		{id: "mimo-v2.5-pro", input: 435_000_000, cacheRead: 3_600_000, cacheWrite: 0, out: 870_000_000, hasOutput: true},
		{id: "mimo-v2.5", input: 140_000_000, cacheRead: 2_800_000, cacheWrite: 0, out: 280_000_000, hasOutput: true},
		{id: "text-embedding-3-small", input: 20_000_000},
		{id: "text-embedding-3-large", input: 130_000_000},
	} {
		t.Run(tc.id, func(t *testing.T) {
			p := mustLookup(t, reg, tc.id)
			if p.Cost == nil {
				t.Fatal("no cost block")
			}
			if *p.Cost.Input != tc.input {
				t.Errorf("input = %d, want %d", *p.Cost.Input, tc.input)
			}
			if p.Cost.SourceURL == "" || p.Cost.VerifiedOn == "" {
				t.Error("provenance is incomplete")
			}
			if !tc.hasOutput {
				if p.Cost.Output != nil || p.Cost.CacheRead != nil {
					t.Error("an embeddings block declares a lane that does not exist")
				}
				return
			}
			if *p.Cost.CacheRead != tc.cacheRead {
				t.Errorf("cache_read = %d, want %d", *p.Cost.CacheRead, tc.cacheRead)
			}
			if *p.Cost.CacheWrite != tc.cacheWrite {
				t.Errorf("cache_write = %d, want %d", *p.Cost.CacheWrite, tc.cacheWrite)
			}
			if *p.Cost.Output != tc.out {
				t.Errorf("output = %d, want %d", *p.Cost.Output, tc.out)
			}
			// No window ships until it is verified that the vendor's off-peak
			// coefficient touches the USD lane and not only credit burn. A
			// window appearing here would halve reported spend.
			if len(p.Cost.Windows) != 0 {
				t.Errorf("a window is attached: %+v", p.Cost.Windows)
			}
		})
	}
}

// A looked-up profile's cost block is a copy. The registry is process-wide, so a
// caller adjusting a rate for one experiment must not reprice everyone else.
func TestClone_CostIsDeepCopied(t *testing.T) {
	reg := Default()
	a := mustLookup(t, reg, "glm-5.3-flash")
	*a.Cost.Input = 1
	a.Cost.Tiers = append(a.Cost.Tiers, CostTier{Name: "x", MinInputTokens: 1, MultiplierPermille: 2000})

	b := mustLookup(t, reg, "glm-5.3-flash")
	if *b.Cost.Input == 1 {
		t.Error("mutating a looked-up rate changed the registry")
	}
	if len(b.Cost.Tiers) != 0 {
		t.Error("appending a tier to a looked-up profile changed the registry")
	}
}

// budget_param is the wire NAME of the budget knob, so it is required for that
// one control and refused everywhere else: missing, the renderer has no key to
// send; present elsewhere, the profile describes a knob the model lacks.
func TestReasoning_BudgetParamIsControlSpecific(t *testing.T) {
	budget := `profiles:
  - id: b
    wire_model_id: b
    max_tokens_param: max_tokens
    verified: source-derived
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: budget_tokens
`
	if _, err := NewRegistry([]byte(budget)); err == nil {
		t.Error("budget control without budget_param should be refused")
	} else if !strings.Contains(err.Error(), "budget_param is empty") {
		t.Errorf("error = %v", err)
	}

	if _, err := NewRegistry([]byte(budget + "      budget_param: thinking_budget\n")); err != nil {
		t.Errorf("budget control with budget_param should load: %v", err)
	}

	effort := `profiles:
  - id: e
    wire_model_id: e
    max_tokens_param: max_tokens
    verified: source-derived
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: false
      control: effort
      effort_values: [low, high]
      budget_param: thinking_budget
`
	if _, err := NewRegistry([]byte(effort)); err == nil {
		t.Error("budget_param on an effort model should be refused")
	} else if !strings.Contains(err.Error(), "takes no budget") {
		t.Errorf("error = %v", err)
	}
}

// The embedding block is embeddings-only, the mirror of the guard that keeps
// chat knobs off an embeddings profile. Both sub-keys are checked: a guard that
// covers one reads as complete while leaving a hole.
func TestEmbedding_BlockIsRefusedOnAChatProfile(t *testing.T) {
	for _, block := range []string{
		"    embedding: {dimensions: true}\n",
		"    embedding: {default_dimensions: 1536}\n",
	} {
		_, err := NewRegistry([]byte(chatHead + block))
		if err == nil {
			t.Fatalf("a chat profile declaring %s should be refused", strings.TrimSpace(block))
		}
		if !strings.Contains(err.Error(), "must not declare embedding") {
			t.Errorf("error = %v", err)
		}
	}
}

// default_dimensions is required, because a caller sizes its vector column and
// index to it BEFORE the first call. Absent, the number goes back to living in
// every application that uses the model — two sources of truth for one fact,
// with nothing to compare them against.
func TestEmbedding_DefaultDimensionsIsRequired(t *testing.T) {
	doc := `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
    limits: {context: 8191}
`
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("an embeddings profile without default_dimensions should be refused")
	}
	if !strings.Contains(err.Error(), "default_dimensions") {
		t.Errorf("error = %v", err)
	}

	if _, err := NewRegistry([]byte(doc + "    embedding: {default_dimensions: 768}\n")); err != nil {
		t.Errorf("a stated default_dimensions should load: %v", err)
	}
	if _, err := NewRegistry([]byte(doc + "    embedding: {default_dimensions: -1}\n")); err == nil {
		t.Error("a negative default_dimensions should be refused")
	}
}

// The shipped embedding profiles carry the width their vendor documents. A
// caller builds storage to this, so a wrong number here is a corpus that has to
// be rebuilt rather than an error anyone sees at the time.
func TestDefault_EmbeddingDimensionsMatchTheVendor(t *testing.T) {
	reg := Default()
	for id, want := range map[string]int{
		"text-embedding-3-small": 1536,
		"text-embedding-3-large": 3072,
	} {
		p := mustLookup(t, reg, id)
		if p.Embedding.DefaultDimensions != want {
			t.Errorf("%s default_dimensions = %d, want %d", id, p.Embedding.DefaultDimensions, want)
		}
		if !p.Embedding.Dimensions {
			t.Errorf("%s should accept the dimensions parameter", id)
		}
	}
}
