package llmwire

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// Regression tests for the findings of the review on PR #4. Each one failed before
// its fix.

// Warnings() is documented as readable while the stream is still running, so the
// reader goroutine appending pricing warnings outside the mutex was a real race
// against a caller polling it. Caught by -race.
func TestStream_WarningsAreRaceFreeWhilePolled(t *testing.T) {
	lines := make([]string, 0, 30)
	for i := 0; i < 25; i++ {
		lines = append(lines, `data: {"choices":[{"delta":{"content":"x"}}]}`)
	}
	lines = append(lines,
		`data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}`,
		"data: [DONE]")
	srv := flushingServer(t, lines, time.Millisecond)

	s, _, err := streamClient(t, srv, 5*time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = s.Warnings()
			}
		}
	}()

	for s.Next() {
		_ = s.Event()
	}
	close(stop)
	wg.Wait()

	if err := s.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
}

// A partial sum is not a smaller price, it is an unknown one. When one batch cannot
// be priced the whole total is Unpriced, and Cost's contract is that Unpriced
// carries 0 — otherwise a budget check is handed three batches' worth of a
// four-batch call.
func TestEmbed_UnpricedTotalCarriesZero(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		raw, _ := readAllBody(r)
		var req struct {
			Input []string `json:"input"`
		}
		_ = json.Unmarshal(raw, &req)
		rows := make([]map[string]any, 0, len(req.Input))
		for i := range req.Input {
			rows = append(rows, map[string]any{"index": i, "embedding": []float32{1}})
		}
		body := map[string]any{"model": "m", "data": rows}
		// Batch 2 reports nothing, so it cannot be priced; the others can.
		if calls != 2 {
			body["usage"] = map[string]any{"prompt_tokens": len(req.Input)}
		}
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(srv.Close)

	resp, _, err := embedClient(t, srv).Embed(context.Background(), EmbedRequest{
		Model:  "text-embedding-3-small",
		Inputs: inputs(200),
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Usage.Cost.Provenance != Unpriced {
		t.Fatalf("provenance = %v, want unpriced", resp.Usage.Cost.Provenance)
	}
	if resp.Usage.Cost.NanoUSD != 0 {
		t.Errorf("cost = %d nano with provenance unpriced; an understated total is worse than "+
			"an absent one", resp.Usage.Cost.NanoUSD)
	}
}

// Cost has to survive being written down and read back: the type exists to be
// recorded, and a provenance that marshals but will not unmarshal turns that into
// an error at read time.
func TestCost_RoundTripsThroughJSON(t *testing.T) {
	for _, want := range []CostProvenance{Unpriced, FromTable, Reported} {
		in := Cost{NanoUSD: 42, Provenance: want, AppliedAt: time.Unix(0, 0).UTC(), Window: "peak"}
		raw, err := json.Marshal(in)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var out Cost
		if err := json.Unmarshal(raw, &out); err != nil {
			t.Fatalf("unmarshal %s: %v", raw, err)
		}
		if out.Provenance != want || out.NanoUSD != 42 || out.Window != "peak" {
			t.Errorf("round trip of %v gave %+v", want, out)
		}
	}

	// A name written by a later version reads as "unknown", which is true, rather
	// than making the whole record unreadable.
	var out Cost
	if err := json.Unmarshal([]byte(`{"nano_usd":1,"provenance":"from-the-future"}`), &out); err != nil {
		t.Fatalf("an unknown provenance should decode, not fail: %v", err)
	}
	if out.Provenance != Unpriced {
		t.Errorf("unknown provenance = %v, want unpriced", out.Provenance)
	}
}

// The overflow guard has to cover the scaling multiply, not only the digit loop:
// some wrapped values come back POSITIVE, which the negative-rate check never sees,
// leaving a silently wrong rate in the table.
func TestRate_HugeIntegerPartIsRefused(t *testing.T) {
	doc := chatHead + `    cost:
      input: 20000000000
      cache_read: 0
      cache_write: 0
      output: 0
` + goodProvenance
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("a rate too large for nano-USD should fail the load")
	}
	if !strings.Contains(err.Error(), "too large") {
		t.Errorf("error = %v", err)
	}
}

// Two windows in different zones can both match the same instant, so "the first
// matching window" would be a document-order accident. One block is one vendor's
// schedule, so mixed zones are refused.
func TestCostBlock_MixedZonesAreRefused(t *testing.T) {
	doc := chatHead + `    cost:
      input: 1.0
      cache_read: 0
      cache_write: 0
      output: 0
      windows:
        - {name: sg, zone: Asia/Singapore, from: "14:00", to: "18:00", multiplier_permille: 500}
        - {name: cn, zone: Asia/Shanghai, from: "00:00", to: "08:00", multiplier_permille: 800}
` + goodProvenance
	_, err := NewRegistry([]byte(doc))
	if err == nil {
		t.Fatal("windows in two zones should fail the load")
	}
	if !strings.Contains(err.Error(), "different zones") {
		t.Errorf("error = %v", err)
	}
}

// Relaxing a tool choice to "auto" is only a coercion if the model takes "auto".
// Substituting one refused mode for another would send the 400 the tier policy
// exists to catch locally.
func TestPlan_RelaxationChecksThatAutoIsAccepted(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: required-only
    wire_model_id: required-only
    max_tokens_param: max_tokens
    verified: source-derived
    tools:
      supported: true
      format: native
      tool_choice_values: [required]
    streaming: {supported: true, accepts_stream_options: true}
`)
	c := testClient(t, reg)
	req := ChatRequest{
		Model:      "required-only",
		Messages:   []Message{User("hi")},
		Tools:      []Tool{{Name: "search"}},
		ToolChoice: ToolChoice{Mode: ToolChoiceFunction, Name: "search"},
	}

	_, _, err := c.plan(req, false)
	if err == nil {
		t.Fatal("expected a refusal: there is no accepted mode to relax to")
	}
	if !strings.Contains(err.Error(), "nothing to relax") {
		t.Errorf("error = %v", err)
	}

	// Under BestEffort the field is dropped rather than replaced with a mode the
	// model also refuses.
	req.BestEffort = true
	pl, warnings, err := c.plan(req, false)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if pl.req.ToolChoice.Mode != ToolChoiceUnset {
		t.Errorf("tool choice = %q, want it dropped", pl.req.ToolChoice.Mode)
	}
	if len(warnings) == 0 {
		t.Error("the drop was not reported")
	}
}
