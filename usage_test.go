package llmwire

import (
	"encoding/json"
	"testing"
)

// "Not reported" and "reported as zero" are different facts. Collapsing them
// makes a missing field indistinguishable from a genuinely free call, which is
// how a cost column fills with zeros nobody can explain later.
func TestParseUsage_AbsentIsNotZero(t *testing.T) {
	u := parseUsage(nil)
	if u.Input.Total != nil {
		t.Errorf("input total = %v, want nil for an absent usage object", *u.Input.Total)
	}
	if u.Output.Total != nil {
		t.Errorf("output total = %v, want nil", *u.Output.Total)
	}
	if u.Raw != nil {
		t.Errorf("raw = %s, want nil", u.Raw)
	}
}

func TestParseUsage_ReportedZeroIsDistinctFromAbsent(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`))
	if u.Input.Total == nil {
		t.Fatal("input total is nil, want a reported 0")
	}
	if *u.Input.Total != 0 {
		t.Errorf("input total = %d, want 0", *u.Input.Total)
	}
}

// prompt_tokens INCLUDES cached_tokens, so the full-rate lane is the difference.
// Adding instead of subtracting overcharges every cached call.
func TestParseUsage_CacheLaneIsSubtractedNotAdded(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":30}}`))
	if got := *u.Input.CacheRead; got != 30 {
		t.Errorf("cache read = %d, want 30", got)
	}
	if got := *u.Input.NoCache; got != 70 {
		t.Errorf("no-cache lane = %d, want 70 (100 total includes the 30 cached)", got)
	}
}

// completion_tokens INCLUDES reasoning_tokens. Pricing reasoning separately
// double-counts every call these models make.
func TestParseUsage_ReasoningIsInsideCompletion(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"completion_tokens":200,"completion_tokens_details":{"reasoning_tokens":150}}`))
	if got := *u.Output.Reasoning; got != 150 {
		t.Errorf("reasoning = %d, want 150", got)
	}
	if got := *u.Output.Text; got != 50 {
		t.Errorf("text lane = %d, want 50 (200 total includes the 150 reasoning)", got)
	}
}

// An endpoint that reports text_tokens itself must win over the derived value.
func TestParseUsage_ReportedTextLaneWinsOverDerived(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"completion_tokens":200,"completion_tokens_details":{"reasoning_tokens":150,"text_tokens":45}}`))
	if got := *u.Output.Text; got != 45 {
		t.Errorf("text lane = %d, want the reported 45, not the derived 50", got)
	}
}

// Both operands arrive from the wire. A cached count exceeding the prompt count
// would produce a negative input lane, which at pricing time CREDITS the caller
// for tokens actually spent.
func TestParseUsage_NegativeLaneIsClampedToZero(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":9999}}`))
	if got := *u.Input.NoCache; got != 0 {
		t.Errorf("no-cache lane = %d, want 0 — a negative lane would credit the caller", got)
	}
}

// A lane derived from an unknown total is itself unknown, not zero.
func TestParseUsage_DerivedLaneIsNilWhenTotalIsAbsent(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"prompt_tokens_details":{"cached_tokens":5}}`))
	if u.Input.NoCache != nil {
		t.Errorf("no-cache lane = %d, want nil when the total was never reported", *u.Input.NoCache)
	}
}

// The raw bytes are the evidence for whatever shape an endpoint really sent,
// and the only way to settle whether a zero is an answer or an unread field.
func TestParseUsage_RawIsPreserved(t *testing.T) {
	raw := json.RawMessage(`{"prompt_tokens":1,"image_tokens":42}`)
	u := parseUsage(raw)
	if string(u.Raw) != string(raw) {
		t.Errorf("raw = %s, want it preserved verbatim", u.Raw)
	}
}

// A usage field this package cannot read must not fail a completion the caller
// already received.
func TestParseUsage_MalformedObjectKeepsBytesAndDoesNotPanic(t *testing.T) {
	raw := json.RawMessage(`{"prompt_tokens":"not a number"}`)
	u := parseUsage(raw)
	if u.Input.Total != nil {
		t.Error("input total should be nil for an undecodable object")
	}
	if string(u.Raw) != string(raw) {
		t.Errorf("raw = %s, want the bytes kept as evidence", u.Raw)
	}
}

// The finish_reason chunk commonly carries "usage": null, and the real usage
// arrives later. "Present" must not be confused with "reported", or the parser
// keeps the null and discards the accounting.
func TestWireUsage_ReportedDistinguishesNullFromPopulated(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want bool
	}{
		{"null", `null`, false},
		{"empty object", `{}`, false},
		{"populated", `{"prompt_tokens":1}`, true},
		{"only details", `{"prompt_tokens_details":{"cached_tokens":0}}`, true},
		{"explicit zeros", `{"total_tokens":0}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var w wireUsage
			if err := json.Unmarshal([]byte(tc.body), &w); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if got := w.reported(); got != tc.want {
				t.Errorf("reported() = %v, want %v", got, tc.want)
			}
		})
	}
}
