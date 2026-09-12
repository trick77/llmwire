package llmwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// embedServer answers each batch with one vector per input, echoing the requested
// order back in whatever sequence the caller of newBody chooses.
func embedServer(t *testing.T, reorder bool) (*httptest.Server, *[]int) {
	t.Helper()
	var batchSizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := readAllBody(r)
		var req struct {
			Input      []string `json:"input"`
			Dimensions *int     `json:"dimensions"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Errorf("request body: %v", err)
		}
		batchSizes = append(batchSizes, len(req.Input))

		type row struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		rows := make([]row, 0, len(req.Input))
		for i := range req.Input {
			// The vector encodes its own index, so a misplaced row is detectable.
			rows = append(rows, row{Index: i, Embedding: []float32{float32(i)}})
		}
		if reorder {
			// The spec does not promise ordered data. Reversing it here is what
			// catches a reader that trusts the order.
			for i, j := 0, len(rows)-1; i < j; i, j = i+1, j-1 {
				rows[i], rows[j] = rows[j], rows[i]
			}
		}
		resp := map[string]any{
			"model": "text-embedding-3-small",
			"data":  rows,
			"usage": map[string]any{"prompt_tokens": len(req.Input)},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(srv.Close)
	return srv, &batchSizes
}

func embedClient(t *testing.T, srv *httptest.Server) *Client {
	t.Helper()
	return New(Config{
		BaseURL:  srv.URL,
		Registry: Default(),
		Now:      func() time.Time { return time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC) },
	})
}

func inputs(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("text %d", i)
	}
	return out
}

// 150 inputs go out as 64 + 64 + 22, and every vector comes back at its own index.
func TestEmbed_BatchesAtSixtyFour(t *testing.T) {
	srv, sizes := embedServer(t, false)
	resp, warnings, err := embedClient(t, srv).Embed(context.Background(), EmbedRequest{
		Model:  "text-embedding-3-small",
		Inputs: inputs(150),
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("warnings = %v", warnings)
	}
	if fmt.Sprint(*sizes) != "[64 64 22]" {
		t.Errorf("batch sizes = %v, want [64 64 22]", *sizes)
	}
	if len(resp.Vectors) != 150 {
		t.Fatalf("got %d vectors, want 150", len(resp.Vectors))
	}
	// Each batch numbers its rows from 0, so row i of batch b carries i.
	for i, v := range resp.Vectors {
		if len(v) != 1 {
			t.Fatalf("vector %d = %v", i, v)
		}
		want := float32(i % embedBatchSize)
		if v[0] != want {
			t.Errorf("vector %d = %v, want %v: a row landed in the wrong slot", i, v[0], want)
		}
	}
	// Usage sums across batches, and the price is the sum of per-batch prices.
	if resp.Usage.Input.Total == nil || *resp.Usage.Input.Total != 150 {
		t.Errorf("input total = %v, want 150", resp.Usage.Input.Total)
	}
	if resp.Usage.Cost.Provenance != FromTable {
		t.Errorf("cost = %+v, want from-table", resp.Usage.Cost)
	}
}

// THE assertion that matters for reassembly: out-of-order data must still land in
// input order, because a misplaced vector produces perfectly shaped output that is
// silently wrong.
func TestEmbed_ReassemblesByIndex(t *testing.T) {
	srv, _ := embedServer(t, true)
	resp, _, err := embedClient(t, srv).Embed(context.Background(), EmbedRequest{
		Model:  "text-embedding-3-small",
		Inputs: inputs(5),
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	for i, v := range resp.Vectors {
		if v[0] != float32(i) {
			t.Errorf("vector %d = %v, want %v", i, v[0], float32(i))
		}
	}
}

// Every way the index can be wrong is an error rather than a plausible-looking
// slice.
func TestEmbed_MalformedBatchesAreRefused(t *testing.T) {
	for _, tc := range []struct{ name, body, want string }{
		{
			name: "too few rows",
			body: `{"model":"m","data":[{"index":0,"embedding":[1]}]}`,
			want: "asked for 2 embeddings and got 1",
		},
		{
			name: "index outside the batch",
			body: `{"model":"m","data":[{"index":0,"embedding":[1]},{"index":9,"embedding":[2]}]}`,
			want: "outside the batch",
		},
		{
			name: "duplicate index",
			body: `{"model":"m","data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}`,
			want: "repeats index",
		},
		{
			name: "error object under a 200",
			body: `{"error":{"message":"too long","code":"400"}}`,
			want: "too long",
		},
		{
			name: "not JSON at all",
			body: `<html>502</html>`,
			want: "502",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := jsonServer(t, 200, tc.body)
			resp, _, err := embedClient(t, srv).Embed(context.Background(), EmbedRequest{
				Model:  "text-embedding-3-small",
				Inputs: []string{"a", "b"},
			})
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want it to mention %q", err, tc.want)
			}
			if resp != nil {
				t.Error("a failed call returned a partial response")
			}
		})
	}
}

// A batch failing part-way through returns nothing. A half-filled slice is worse
// than none: the caller cannot tell which rows are real, and a zero vector looks
// like a valid one.
func TestEmbed_NoPartialResults(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"boom","code":"500"}}`))
			return
		}
		raw, _ := readAllBody(r)
		var req struct {
			Input []string `json:"input"`
		}
		_ = json.Unmarshal(raw, &req)
		rows := make([]map[string]any, 0, len(req.Input))
		for i := range req.Input {
			rows = append(rows, map[string]any{"index": i, "embedding": []float32{1}})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "m", "data": rows})
	}))
	t.Cleanup(srv.Close)

	resp, _, err := embedClient(t, srv).Embed(context.Background(), EmbedRequest{
		Model:  "text-embedding-3-small",
		Inputs: inputs(100),
	})
	if err == nil {
		t.Fatal("expected the failing batch to fail the call")
	}
	if resp != nil {
		t.Fatal("a failed call returned vectors")
	}
}

// Usage is summed only while every batch reported it: a partial sum understates and
// is indistinguishable from a real total.
func TestEmbed_PartialUsageIsNotSummed(t *testing.T) {
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
		if calls == 1 {
			body["usage"] = map[string]any{"prompt_tokens": len(req.Input)}
		}
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(srv.Close)

	resp, warnings, err := embedClient(t, srv).Embed(context.Background(), EmbedRequest{
		Model:  "text-embedding-3-small",
		Inputs: inputs(100),
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Usage.Input.Total != nil {
		t.Errorf("input total = %v, want nil when a batch reported nothing", resp.Usage.Input.Total)
	}
	// And the price is unknown rather than a sum missing a batch.
	if resp.Usage.Cost.Provenance != Unpriced {
		t.Errorf("cost = %+v, want unpriced when a batch could not be priced", resp.Usage.Cost)
	}
	if len(warnings) == 0 {
		t.Error("no warning for the batch that could not be priced")
	}
}

// dimensions is sent only where the profile says the model takes it; elsewhere it
// is a refusal, and under BestEffort a drop with a warning.
func TestEmbed_DimensionsFollowsTheProfile(t *testing.T) {
	dims := 256

	t.Run("supported", func(t *testing.T) {
		var seen []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			seen, _ = readAllBody(r)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": "m",
				"data":  []map[string]any{{"index": 0, "embedding": []float32{1}}},
			})
		}))
		t.Cleanup(srv.Close)
		if _, _, err := embedClient(t, srv).Embed(context.Background(), EmbedRequest{
			Model:      "text-embedding-3-small",
			Inputs:     []string{"a"},
			Dimensions: &dims,
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if !strings.Contains(string(seen), `"dimensions":256`) {
			t.Errorf("body = %s", seen)
		}
	})

	reg := registryFrom(t, `profiles:
  - id: old-embed
    endpoint: embeddings
    wire_model_id: old-embed
    verified: source-derived
    embedding: {default_dimensions: 1536}
    limits: {context: 8191}
`)

	t.Run("unsupported is refused", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			t.Error("a refused request was sent anyway")
		}))
		t.Cleanup(srv.Close)
		c := New(Config{BaseURL: srv.URL, Registry: reg})
		_, _, err := c.Embed(context.Background(), EmbedRequest{
			Model:      "old-embed",
			Inputs:     []string{"a"},
			Dimensions: &dims,
		})
		var unsup *UnsupportedError
		if !errors.As(err, &unsup) {
			t.Fatalf("error = %T %v", err, err)
		}
		if !unsup.Demotable {
			t.Error("the refusal should say BestEffort would have worked")
		}
	})

	t.Run("BestEffort drops it", func(t *testing.T) {
		var seen []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			seen, _ = readAllBody(r)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": "m",
				"data":  []map[string]any{{"index": 0, "embedding": []float32{1}}},
			})
		}))
		t.Cleanup(srv.Close)
		c := New(Config{BaseURL: srv.URL, Registry: reg})
		_, warnings, err := c.Embed(context.Background(), EmbedRequest{
			Model:      "old-embed",
			Inputs:     []string{"a"},
			Dimensions: &dims,
			BestEffort: true,
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if strings.Contains(string(seen), "dimensions") {
			t.Errorf("dimensions reached an endpoint that rejects it: %s", seen)
		}
		var saw bool
		for _, w := range warnings {
			if w.Feature == "dimensions" {
				saw = true
			}
		}
		if !saw {
			t.Errorf("warnings = %v, want one about dimensions", warnings)
		}
	})
}

// Malformed requests are refused locally, and BestEffort cannot rescue them: there
// is nothing to send instead of no inputs.
func TestEmbed_MalformedRequestsNeverReachTheNetwork(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("a malformed request was sent anyway")
	}))
	t.Cleanup(srv.Close)
	c := embedClient(t, srv)
	zero := 0

	for _, tc := range []struct {
		name string
		req  EmbedRequest
		want string
	}{
		{"no inputs", EmbedRequest{Model: "text-embedding-3-small", BestEffort: true}, "at least one input"},
		{"empty input", EmbedRequest{Model: "text-embedding-3-small", Inputs: []string{"a", ""}, BestEffort: true}, "input 1 is empty"},
		{"zero dimensions", EmbedRequest{Model: "text-embedding-3-small", Inputs: []string{"a"}, Dimensions: &zero, BestEffort: true}, "must be positive"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := c.Embed(context.Background(), tc.req)
			if err == nil {
				t.Fatal("expected a refusal")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want %q", err, tc.want)
			}
			var unsup *UnsupportedError
			if errors.As(err, &unsup) && unsup.Demotable {
				t.Error("a malformed request was marked demotable")
			}
		})
	}
}

// The endpoints are not interchangeable, in either direction.
func TestEmbed_WrongEndpointIsRefused(t *testing.T) {
	c := New(Config{BaseURL: "https://example.invalid", Registry: Default()})

	_, _, err := c.Embed(context.Background(), EmbedRequest{Model: "glm-5.3-flash", Inputs: []string{"a"}})
	var unsup *UnsupportedError
	if !errors.As(err, &unsup) || unsup.Demotable {
		t.Fatalf("error = %T %v, want a non-demotable refusal", err, err)
	}

	_, _, err = c.Chat(context.Background(), ChatRequest{
		Model:    "text-embedding-3-small",
		Messages: []Message{User("hi")},
	})
	if !errors.As(err, &unsup) || unsup.Demotable {
		t.Fatalf("error = %T %v, want a non-demotable refusal", err, err)
	}
}
