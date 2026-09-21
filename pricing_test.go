package llmwire

import (
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Every instant here is PINNED. A pricing test that only passes between 14:00 and
// 18:00 Singapore time is worse than no test, which is why Config.Now is
// injectable in the first place.

func i64(v int64) *int64 { return &v }

// costHeader builds the response header the way net/http will read it. A map
// literal keyed by the lowercase name would store a key Header.Get never finds,
// since Get canonicalises and direct map writes do not — a test that passes
// against nothing.
func costHeader(v string) http.Header {
	h := http.Header{}
	h.Set(litellmCostHeader, v)
	return h
}

// usageOf builds the lanes parseUsage would have derived, so the tests price the
// same shape the parser produces: NoCache is Total minus CacheRead.
func usageOf(promptTotal, cacheRead, outputTotal int64) Usage {
	noCache := promptTotal - cacheRead
	return Usage{
		Input:  InputTokens{Total: i64(promptTotal), NoCache: i64(noCache), CacheRead: i64(cacheRead)},
		Output: OutputTokens{Total: i64(outputTotal)},
	}
}

func priceWith(t *testing.T, doc, model string, u Usage, at time.Time) (Cost, []Warning) {
	t.Helper()
	p := mustLookup(t, registryFrom(t, doc), model)
	return priceCall(p, u, nil, 200, at)
}

const pricedChat = `profiles:
  - id: m
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
    limits: {context: 1000000, max_output: 100000}
    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      source_url: https://vendor.example/pricing
      verified_on: 2026-09-12
`

// 1000 fresh prompt tokens at $0.15/1M plus 500 output at $0.50/1M:
// 1000*150 + 500*500 = 400_000 nano-USD.
func TestPrice_LanesSumExactly(t *testing.T) {
	cost, warnings := priceWith(t, pricedChat, "m", usageOf(1000, 0, 500), time.Unix(0, 0).UTC())
	if len(warnings) != 0 {
		t.Errorf("unexpected warnings: %v", warnings)
	}
	if cost.Provenance != FromTable {
		t.Errorf("provenance = %v, want from-table", cost.Provenance)
	}
	if cost.NanoUSD != 400_000 {
		t.Errorf("cost = %d nano, want 400000", cost.NanoUSD)
	}
}

// The measured MiMo shape: a 258-token prompt reporting 192 cached. Cache reads
// are a DISCOUNT, so the total must come out below the all-fresh price — the
// assertion that catches the lanes being added instead of subtracted.
func TestPrice_CacheReadsAreSubtractedNotAdded(t *testing.T) {
	at := time.Unix(0, 0).UTC()
	cached, _ := priceWith(t, pricedChat, "m", usageOf(258, 192, 100), at)
	fresh, _ := priceWith(t, pricedChat, "m", usageOf(258, 0, 100), at)

	// (258-192)*150 + 192*30 + 100*500 = 9900 + 5760 + 50000 = 65_660
	if cached.NanoUSD != 65_660 {
		t.Errorf("cost = %d nano, want 65660", cached.NanoUSD)
	}
	if cached.NanoUSD >= fresh.NanoUSD {
		t.Errorf("a cached call (%d) is not cheaper than the same call uncached (%d); the cache "+
			"lane is being added rather than subtracted", cached.NanoUSD, fresh.NanoUSD)
	}
}

// completion_tokens already CONTAINS reasoning_tokens. Pricing the reasoning lane
// as well double-counts every call a reasoning model makes, which is all of them
// on the models this library targets.
func TestPrice_ReasoningIsNotASeparateLane(t *testing.T) {
	at := time.Unix(0, 0).UTC()
	u := usageOf(1000, 0, 500)
	plain, _ := priceWith(t, pricedChat, "m", u, at)

	u.Output.Reasoning = i64(400)
	u.Output.Text = i64(100)
	thinking, _ := priceWith(t, pricedChat, "m", u, at)

	if thinking.NanoUSD != plain.NanoUSD {
		t.Errorf("reasoning changed the price: %d vs %d; completion_tokens already includes it",
			thinking.NanoUSD, plain.NanoUSD)
	}
}

// A total nobody reported makes the call unpriceable. Charging zero for it would
// under-report, and the warning is what tells the caller the number is missing
// rather than free.
func TestPrice_MissingTotalsAreUnpriced(t *testing.T) {
	at := time.Unix(0, 0).UTC()
	for _, tc := range []struct {
		name string
		u    Usage
		want string
	}{
		{"no prompt tokens", Usage{Output: OutputTokens{Total: i64(10)}}, "no prompt tokens"},
		{"no completion tokens", Usage{Input: InputTokens{Total: i64(10), NoCache: i64(10)}}, "no completion tokens"},
		{"nothing at all", Usage{}, "no prompt tokens"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cost, warnings := priceWith(t, pricedChat, "m", tc.u, at)
			if cost.Provenance != Unpriced || cost.NanoUSD != 0 {
				t.Errorf("cost = %+v, want unpriced 0", cost)
			}
			if len(warnings) != 1 || !strings.Contains(warnings[0].Details, tc.want) {
				t.Errorf("warnings = %v, want one mentioning %q", warnings, tc.want)
			}
		})
	}
}

// A nil cache count is nothing to charge for, not an error.
func TestPrice_NilCacheLaneIsZero(t *testing.T) {
	u := Usage{
		Input:  InputTokens{Total: i64(1000), NoCache: i64(1000)},
		Output: OutputTokens{Total: i64(0)},
	}
	cost, warnings := priceWith(t, pricedChat, "m", u, time.Unix(0, 0).UTC())
	if len(warnings) != 0 {
		t.Errorf("warnings = %v", warnings)
	}
	if cost.NanoUSD != 150_000 {
		t.Errorf("cost = %d, want 150000", cost.NanoUSD)
	}
}

// A model with no rate reports 0 WITH a warning, because 0 means unknown and
// never free. Per call, not once per process: the return channel is a per-call
// []Warning, and a once-per-process warning would make the set depend on call
// order and vanish for whichever caller went second.
func TestPrice_UnpricedModelWarnsEveryCall(t *testing.T) {
	doc := `profiles:
  - id: free
    wire_model_id: free
    max_tokens_param: max_tokens
    verified: measured
`
	p := mustLookup(t, registryFrom(t, doc), "free")
	for i := 0; i < 3; i++ {
		cost, warnings := priceCall(p, usageOf(10, 0, 10), nil, 200, time.Unix(0, 0).UTC())
		if cost.Provenance != Unpriced || cost.NanoUSD != 0 {
			t.Fatalf("cost = %+v, want unpriced", cost)
		}
		if len(warnings) != 1 {
			t.Fatalf("call %d produced %d warnings, want exactly 1 every time", i, len(warnings))
		}
		if !strings.Contains(warnings[0].Details, "unknown, not zero") {
			t.Errorf("warning = %q", warnings[0].Details)
		}
	}
}

// The zero Cost means Unpriced, so anything that forgets to price reads as
// unknown rather than as a free call.
func TestCost_ZeroValueIsUnpriced(t *testing.T) {
	var c Cost
	if c.Provenance != Unpriced {
		t.Fatalf("zero provenance = %v, want unpriced", c.Provenance)
	}
	if c.Provenance.String() != "unpriced" {
		t.Errorf("String() = %q", c.Provenance.String())
	}
	var u Usage
	if u.Cost.Provenance != Unpriced {
		t.Error("a usage object built below profile resolution is not unpriced by default")
	}
}

// Windows are half-open [from, to) in the provider's OWN zone. No profile ships
// one, so the machinery is exercised on a synthetic block — see the shipped-rate
// test for why none is attached.
func TestPrice_WindowBoundaries(t *testing.T) {
	doc := `profiles:
  - id: w
    wire_model_id: w
    max_tokens_param: max_tokens
    verified: source-derived
    cost:
      input: 1.0
      cache_read: 0
      cache_write: 0
      output: 0
      windows:
        - name: peak
          zone: Asia/Singapore
          days: [Mon, Tue, Wed, Thu, Fri]
          from: "14:00"
          to: "18:00"
          multiplier_permille: 1000
      outside_windows_permille: 500
      source_url: https://vendor.example/pricing
      verified_on: 2026-09-12
`
	sg, err := time.LoadLocation("Asia/Singapore")
	if err != nil {
		t.Fatalf("no tzdata: %v", err)
	}
	// 2026-09-11 is a Friday, 2026-09-12 a Saturday.
	for _, tc := range []struct {
		name       string
		at         time.Time
		wantWindow string
		wantNano   int64
	}{
		{"Friday 17:59 is inside", time.Date(2026, 9, 11, 17, 59, 0, 0, sg), "peak", 1_000_000_000},
		{"Friday 18:00 is outside", time.Date(2026, 9, 11, 18, 0, 0, 0, sg), "", 500_000_000},
		{"Friday 14:00 is inside", time.Date(2026, 9, 11, 14, 0, 0, 0, sg), "peak", 1_000_000_000},
		{"Saturday afternoon is outside", time.Date(2026, 9, 12, 15, 0, 0, 0, sg), "", 500_000_000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u := Usage{
				Input:  InputTokens{Total: i64(1_000_000), NoCache: i64(1_000_000)},
				Output: OutputTokens{Total: i64(0)},
			}
			cost, _ := priceWith(t, doc, "w", u, tc.at)
			if cost.Window != tc.wantWindow {
				t.Errorf("window = %q, want %q", cost.Window, tc.wantWindow)
			}
			if cost.NanoUSD != tc.wantNano {
				t.Errorf("cost = %d, want %d", cost.NanoUSD, tc.wantNano)
			}
		})
	}

	// The same instant, expressed in UTC, must select the same window: the rule is
	// evaluated in the provider's zone, not the caller's.
	utc := time.Date(2026, 9, 11, 17, 59, 0, 0, sg).UTC()
	u := Usage{Input: InputTokens{Total: i64(1_000_000), NoCache: i64(1_000_000)}, Output: OutputTokens{Total: i64(0)}}
	if cost, _ := priceWith(t, doc, "w", u, utc); cost.Window != "peak" {
		t.Errorf("the same instant in UTC selected %q, want peak", cost.Window)
	}
}

// A tier is selected by the prompt's TOTAL, cached tokens included, and applies to
// the whole call.
func TestPrice_TierThreshold(t *testing.T) {
	doc := `profiles:
  - id: t
    wire_model_id: t
    max_tokens_param: max_tokens
    verified: source-derived
    cost:
      input: 1.0
      cache_read: 0
      cache_write: 0
      output: 1.0
      tiers:
        - {name: long-context, min_input_tokens: 272000, multiplier_permille: 2000}
      source_url: https://vendor.example/pricing
      verified_on: 2026-09-12
`
	at := time.Unix(0, 0).UTC()
	under := Usage{
		Input:  InputTokens{Total: i64(271_999), NoCache: i64(271_999)},
		Output: OutputTokens{Total: i64(0)},
	}
	over := Usage{
		Input:  InputTokens{Total: i64(272_000), NoCache: i64(272_000)},
		Output: OutputTokens{Total: i64(0)},
	}

	cost, _ := priceWith(t, doc, "t", under, at)
	if cost.Tier != "" || cost.NanoUSD != 271_999_000 {
		t.Errorf("under the threshold: tier=%q cost=%d", cost.Tier, cost.NanoUSD)
	}
	cost, _ = priceWith(t, doc, "t", over, at)
	if cost.Tier != "long-context" || cost.NanoUSD != 544_000_000 {
		t.Errorf("at the threshold: tier=%q cost=%d, want long-context and 2x", cost.Tier, cost.NanoUSD)
	}

	// The threshold counts cached tokens too: the vendor prices by how much
	// context it had to hold.
	cached := Usage{
		Input:  InputTokens{Total: i64(272_000), NoCache: i64(1_000), CacheRead: i64(271_000)},
		Output: OutputTokens{Total: i64(0)},
	}
	if cost, _ = priceWith(t, doc, "t", cached, at); cost.Tier != "long-context" {
		t.Errorf("a mostly-cached prompt over the threshold selected %q", cost.Tier)
	}
}

// Rounding is UP, once, on the summed total. Rounding per lane would over-report
// by a nano per lane; rounding down would under-report, which is the direction
// that silently raises what a budget cap allows.
func TestPrice_RoundsUpOnceOnTheTotal(t *testing.T) {
	doc := `profiles:
  - id: r
    wire_model_id: r
    max_tokens_param: max_tokens
    verified: source-derived
    cost:
      input: 0.000000001
      cache_read: 0.000000001
      cache_write: 0
      output: 0.000000001
      source_url: https://vendor.example/pricing
      verified_on: 2026-09-12
`
	// One token in each of two lanes at 1 nano per million: the exact total is
	// 2/1_000_000 nano, which must report as 1, not 0.
	u := Usage{
		Input:  InputTokens{Total: i64(1), NoCache: i64(1)},
		Output: OutputTokens{Total: i64(1)},
	}
	cost, _ := priceWith(t, doc, "r", u, time.Unix(0, 0).UTC())
	if cost.NanoUSD != 1 {
		t.Errorf("cost = %d, want 1: a sub-nano charge rounds up, never away", cost.NanoUSD)
	}
}

// A zero-token call at a real rate costs nothing, and that zero is FromTable —
// known to be zero, not unknown.
func TestPrice_ZeroTokensIsAKnownZero(t *testing.T) {
	u := Usage{
		Input:  InputTokens{Total: i64(0), NoCache: i64(0)},
		Output: OutputTokens{Total: i64(0)},
	}
	cost, warnings := priceWith(t, pricedChat, "m", u, time.Unix(0, 0).UTC())
	if cost.NanoUSD != 0 || cost.Provenance != FromTable {
		t.Errorf("cost = %+v, want a from-table 0", cost)
	}
	if len(warnings) != 0 {
		t.Errorf("warnings = %v", warnings)
	}
}

// An embeddings model has one lane and no completion side, so a missing output
// count must not make it unpriceable.
func TestPrice_EmbeddingsInputLaneOnly(t *testing.T) {
	p := mustLookup(t, Default(), "text-embedding-3-small")
	u := Usage{Input: InputTokens{Total: i64(1_000_000), NoCache: i64(1_000_000)}}
	cost, warnings := priceCall(p, u, nil, 200, time.Unix(0, 0).UTC())
	if len(warnings) != 0 {
		t.Errorf("warnings = %v", warnings)
	}
	if cost.NanoUSD != 20_000_000 {
		t.Errorf("cost = %d nano, want 20000000 ($0.02 for 1M tokens)", cost.NanoUSD)
	}
}

// --- the gateway lane ------------------------------------------------------

const gatewayDoc = `profiles:
  - id: m
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
      source_url: https://vendor.example/pricing
      verified_on: 2026-09-12
  - id: m-via
    base: m
    gateway: litellm
    wire_model_id: proxy/m
`

// The gateway's own figure, in the scientific notation it actually sends.
func TestPrice_ReportedHeader(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	for _, tc := range []struct {
		header string
		want   int64
	}{
		{"1.23e-05", 12_300},
		{"0.000012300", 12_300},
		{" 0.0000123 ", 12_300},
		// Below a nano: rounds up to 1 rather than vanishing.
		{"0.0000000001", 1},
		{"0", 0},
	} {
		t.Run(tc.header, func(t *testing.T) {
			hdr := costHeader(tc.header)
			cost, warnings := priceCall(p, usageOf(100, 0, 100), hdr, 200, time.Unix(0, 0).UTC())
			if len(warnings) != 0 {
				t.Errorf("warnings = %v", warnings)
			}
			if cost.Provenance != Reported {
				t.Errorf("provenance = %v, want reported", cost.Provenance)
			}
			if cost.NanoUSD != tc.want {
				t.Errorf("cost = %d, want %d", cost.NanoUSD, tc.want)
			}
		})
	}
}

// NaN and ±Inf are the reason this parses with big.Rat rather than ParseFloat: a
// NaN cost makes every `total >= limit` comparison FALSE, which silently disables
// the budget cap it feeds.
func TestPrice_ReportedHeaderRejectsUnusableValues(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	for _, header := range []string{"NaN", "Inf", "+Inf", "-Inf", "-0.01", "abc", "None"} {
		t.Run(header, func(t *testing.T) {
			hdr := costHeader(header)
			cost, warnings := priceCall(p, usageOf(100, 0, 100), hdr, 200, time.Unix(0, 0).UTC())
			if cost.Provenance != Unpriced || cost.NanoUSD != 0 {
				t.Fatalf("cost = %+v, want unpriced", cost)
			}
			if len(warnings) != 1 {
				t.Fatalf("warnings = %v, want one", warnings)
			}
			// The budget check a NaN would have broken, asserted directly.
			if cost.NanoUSD < 0 {
				t.Error("the cost failed a >= comparison, which is what a NaN does")
			}
		})
	}
}

// A gateway route is NEVER priced from the table, in either direction: the block
// is dropped on inheritance, and a missing header degrades to Unpriced rather
// than to a list rate.
func TestPrice_GatewayNeverFallsBackToTheTable(t *testing.T) {
	reg := registryFrom(t, gatewayDoc)
	via := mustLookup(t, reg, "m-via")

	cost, warnings := priceCall(via, usageOf(1000, 0, 500), nil, 200, time.Unix(0, 0).UTC())
	if cost.Provenance != Unpriced || cost.NanoUSD != 0 {
		t.Errorf("cost = %+v, want unpriced: a stream behind a gateway sends no cost header", cost)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0].Details, "unknown, not zero") {
		t.Errorf("warnings = %v", warnings)
	}

	// The direct route still prices, so the rule is about the gateway and not
	// about the model.
	direct := mustLookup(t, reg, "m")
	if cost, _ = priceCall(direct, usageOf(1000, 0, 500), nil, 200, time.Unix(0, 0).UTC()); cost.Provenance != FromTable {
		t.Errorf("the direct route stopped pricing: %+v", cost)
	}
}

// "0" under a non-2xx means ERRORED, not free.
func TestPrice_NonSuccessStatusIsUnpriced(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	hdr := costHeader("0")
	cost, warnings := priceCall(p, usageOf(10, 0, 10), hdr, 502, time.Unix(0, 0).UTC())
	if cost.Provenance != Unpriced {
		t.Errorf("cost = %+v, want unpriced on a 502", cost)
	}
	if len(warnings) != 0 {
		t.Errorf("an error path should not warn about cost too: %v", warnings)
	}
}

// A gateway nobody has modelled is Unpriced with a warning, not a load failure:
// the cost header's name is per-gateway, and a new proxy must be addable as data.
func TestPrice_UnknownGatewayLoadsAndIsUnpriced(t *testing.T) {
	doc := `profiles:
  - id: g
    gateway: mystery
    wire_model_id: g
    max_tokens_param: max_tokens
    verified: source-derived
`
	p := mustLookup(t, registryFrom(t, doc), "g")
	cost, warnings := priceCall(p, usageOf(10, 0, 10), http.Header{}, 200, time.Unix(0, 0).UTC())
	if cost.Provenance != Unpriced {
		t.Errorf("cost = %+v, want unpriced", cost)
	}
	if len(warnings) != 1 {
		t.Errorf("warnings = %v", warnings)
	}
}

// AppliedAt records the instant that chose the window, even when no window
// applied and even when the call could not be priced at all.
func TestPrice_AppliedAtIsAlwaysRecorded(t *testing.T) {
	at := time.Date(2026, 9, 12, 8, 30, 0, 0, time.UTC)
	cost, _ := priceWith(t, pricedChat, "m", usageOf(10, 0, 10), at)
	if !cost.AppliedAt.Equal(at) {
		t.Errorf("AppliedAt = %v, want %v", cost.AppliedAt, at)
	}
	cost, _ = priceWith(t, pricedChat, "m", Usage{}, at)
	if !cost.AppliedAt.Equal(at) {
		t.Errorf("an unpriced call lost its instant: %v", cost.AppliedAt)
	}
}

// usageWithCost builds the usage a gateway sends on the final stream chunk when
// include_cost_in_streaming_usage is on: the token lanes plus the gateway's own
// figure on usage.cost. Raw is what priceFromGateway reads, so it carries the
// literal bytes rather than a float that has already been rounded.
func usageWithCost(promptTotal, outputTotal int64, cost string) Usage {
	u := usageOf(promptTotal, 0, outputTotal)
	u.Raw = []byte(`{"prompt_tokens":` + itoa(promptTotal) +
		`,"completion_tokens":` + itoa(outputTotal) +
		`,"cost":` + cost + `}`)
	return u
}

func itoa(v int64) string { return strconv.FormatInt(v, 10) }

// A stream cannot be priced from the header — it is written before the body is
// consumed — so the gateway puts the figure on usage.cost instead. This is the
// lane that makes a streamed answer cost anything at all.
func TestPrice_ReportedReadsCostFromTheUsageBody(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	for _, tc := range []struct {
		cost string
		want int64
	}{
		{"0.0000123", 12_300},
		{"1.23e-05", 12_300},
		// Below a nano rounds up to 1, the same direction as every other lane.
		{"0.0000000001", 1},
		{"0", 0},
	} {
		t.Run(tc.cost, func(t *testing.T) {
			u := usageWithCost(100, 100, tc.cost)
			cost, warnings := priceCall(p, u, nil, 200, time.Unix(0, 0).UTC())
			if len(warnings) != 0 {
				t.Errorf("warnings = %v", warnings)
			}
			if cost.Provenance != Reported {
				t.Errorf("provenance = %v, want reported", cost.Provenance)
			}
			if cost.NanoUSD != tc.want {
				t.Errorf("cost = %d, want %d", cost.NanoUSD, tc.want)
			}
		})
	}
}

// The body is the only figure that can be right on a stream, so it wins. A
// gateway that sends BOTH — LiteLLM has shipped the header as "0" on streams —
// must not have its zero recorded as a confident price.
func TestPrice_ReportedBodyCostBeatsTheHeader(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	u := usageWithCost(100, 100, "0.0000123")
	cost, warnings := priceCall(p, u, costHeader("0"), 200, time.Unix(0, 0).UTC())
	if len(warnings) != 0 {
		t.Fatalf("warnings = %v", warnings)
	}
	if cost.Provenance != Reported || cost.NanoUSD != 12_300 {
		t.Fatalf("cost = %+v, want 12300 reported from the body", cost)
	}
}

// Neither lane reporting is the honest unknown, and it must stay Unpriced rather
// than falling through to the table: a list rate here would be a confident figure
// for a call nobody priced.
func TestPrice_ReportedWithNeitherLaneStaysUnpriced(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	u := usageOf(100, 0, 100)
	u.Raw = []byte(`{"prompt_tokens":100,"completion_tokens":100}`)
	cost, warnings := priceCall(p, u, http.Header{}, 200, time.Unix(0, 0).UTC())
	if cost.Provenance != Unpriced || cost.NanoUSD != 0 {
		t.Fatalf("cost = %+v, want unpriced", cost)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %v, want one", warnings)
	}
}

// A cost field this package cannot read is not a reason to price the call from
// somewhere else, and not a reason to record zero.
//
// The header is set to "0" throughout, which is what LiteLLM sends on a stream:
// a body value wrongly judged ABSENT falls through to that header and records a
// confident zero with no warning. With nil headers this test passed against a
// decoder that rejected nothing, because the warning it counted came from the
// header lane. The assertion on the warning text is what nails the lane.
func TestPrice_ReportedRejectsAnUnusableBodyCost(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	// A STRING cost is the case that fails a json.Number decode, which is how an
	// unusable value once passed for an absent one.
	for _, raw := range []string{`"NaN"`, `"abc"`, `"Inf"`, `-0.01`, `"-0.01"`} {
		t.Run(raw, func(t *testing.T) {
			u := usageWithCost(100, 100, raw)
			cost, warnings := priceCall(p, u, costHeader("0"), 200, time.Unix(0, 0).UTC())
			if cost.Provenance != Unpriced || cost.NanoUSD != 0 {
				t.Fatalf("cost = %+v, want unpriced", cost)
			}
			if len(warnings) != 1 {
				t.Fatalf("warnings = %v, want one", warnings)
			}
			if !strings.Contains(warnings[0].Details, "usage.cost") {
				t.Errorf("warning came from the header lane, not the body: %q", warnings[0].Details)
			}
		})
	}
}

// null is "not reported", the JSON analog of the None literal the header lane
// already accepts — so it falls through to the header on purpose, and a real
// header still prices the call.
func TestPrice_ReportedTreatsANullBodyCostAsAbsent(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	u := usageWithCost(100, 100, `null`)
	cost, warnings := priceCall(p, u, costHeader("1.23e-05"), 200, time.Unix(0, 0).UTC())
	if len(warnings) != 0 {
		t.Fatalf("warnings = %v", warnings)
	}
	if cost.Provenance != Reported || cost.NanoUSD != 12_300 {
		t.Fatalf("cost = %+v, want the header's 12300", cost)
	}
}

// A gateway sending the cost as a decimal STRING is priced, not discarded: the
// quotes are packaging, and the text inside is the same decimal.
func TestPrice_ReportedAcceptsAStringBodyCost(t *testing.T) {
	p := mustLookup(t, registryFrom(t, gatewayDoc), "m-via")
	u := usageWithCost(100, 100, `"1.23e-05"`)
	cost, warnings := priceCall(p, u, costHeader("0"), 200, time.Unix(0, 0).UTC())
	if len(warnings) != 0 {
		t.Fatalf("warnings = %v", warnings)
	}
	if cost.Provenance != Reported || cost.NanoUSD != 12_300 {
		t.Fatalf("cost = %+v, want 12300 from the body", cost)
	}
}
