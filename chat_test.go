package llmwire

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// jsonServer answers every request with one fixed status and body, and records
// the request body so a test can assert what went on the wire.
func jsonServer(t *testing.T, status int, body string) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = readAllBody(r)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

func readAllBody(r *http.Request) ([]byte, error) {
	defer r.Body.Close()
	buf := make([]byte, 0, 1024)
	tmp := make([]byte, 512)
	for {
		n, err := r.Body.Read(tmp)
		buf = append(buf, tmp[:n]...)
		if err != nil {
			return buf, nil
		}
	}
}

func chatClient(t *testing.T, srv *httptest.Server) *Client {
	t.Helper()
	return New(Config{
		BaseURL:  srv.URL,
		Registry: Default(),
		Now:      func() time.Time { return time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC) },
	})
}

const goodCompletion = `{
  "model": "glm-5.3-flash",
  "choices": [{
    "finish_reason": "stop",
    "message": {"role": "assistant", "content": "hello", "reasoning_content": "thought"}
  }],
  "usage": {"prompt_tokens": 258, "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 192},
            "completion_tokens_details": {"reasoning_tokens": 40}}
}`

func TestChat_HappyPath(t *testing.T) {
	srv, sent := jsonServer(t, 200, goodCompletion)
	resp, warnings, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("warnings = %v", warnings)
	}
	if resp.Content != "hello" || resp.Reasoning != "thought" || resp.FinishReason != "stop" {
		t.Errorf("response = %+v", resp)
	}
	if resp.Model != "glm-5.3-flash" {
		t.Errorf("model = %q", resp.Model)
	}
	if len(resp.Raw) == 0 {
		t.Error("Raw is empty; an unmodelled field would be unrecoverable")
	}
	// Lanes, and the cost priced from them: (258-192)*150 + 192*30 + 100*500.
	if resp.Usage.Input.CacheRead == nil || *resp.Usage.Input.CacheRead != 192 {
		t.Errorf("cache read = %v", resp.Usage.Input.CacheRead)
	}
	if resp.Usage.Cost.Provenance != FromTable || resp.Usage.Cost.NanoUSD != 65_660 {
		t.Errorf("cost = %+v, want a from-table 65660", resp.Usage.Cost)
	}
	// The body carried the profile's spelling of the cap-free request.
	var body map[string]any
	if err := json.Unmarshal(*sent, &body); err != nil {
		t.Fatalf("request body: %v", err)
	}
	if body["stream"] != nil {
		t.Error("a non-streaming call sent stream")
	}
}

// The live measurement this phase exists to defuse, asserted on the wire: a cap
// of 16 must arrive as max_tokens on this model. Sent as max_completion_tokens it
// is ACCEPTED and ignored, and the endpoint returned 481 tokens.
func TestChat_CapReachesTheWireUnderTheRightName(t *testing.T) {
	srv, sent := jsonServer(t, 200, goodCompletion)
	cap16 := 16
	if _, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hi")},
		MaxTokens: &cap16,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(*sent, &body); err != nil {
		t.Fatal(err)
	}
	if body["max_tokens"] != float64(16) {
		t.Errorf("max_tokens = %v, want 16", body["max_tokens"])
	}
	if _, ok := body["max_completion_tokens"]; ok {
		t.Error("max_completion_tokens went to an endpoint that accepts and IGNORES it")
	}
}

// A refusal returns before any socket opens. The handler failing the test is the
// assertion.
func TestChat_RefusalNeverReachesTheNetwork(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("a refused request was sent anyway")
	}))
	t.Cleanup(srv.Close)

	_, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningOff(),
	})
	var unsup *UnsupportedError
	if !errors.As(err, &unsup) {
		t.Fatalf("error = %T %v, want *UnsupportedError", err, err)
	}
}

// An error object delivered under a 200. Read as an empty success it would look
// like a model that answered with nothing.
func TestChat_ErrorInsideA200IsSurfaced(t *testing.T) {
	srv, _ := jsonServer(t, 200, `{"error":{"message":"thinking cannot be disabled","code":"1210"}}`)
	_, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %T %v, want *APIError", err, err)
	}
	if apiErr.Code != "1210" {
		t.Errorf("code = %q, want the dispatchable code", apiErr.Code)
	}
}

// A 200 with neither choices nor an error object. Indexing choices[0] here would
// panic in production on a body some endpoint really sends.
func TestChat_NoChoicesIsAnErrorNotAPanic(t *testing.T) {
	srv, _ := jsonServer(t, 200, `{"model":"glm-5.3-flash","choices":[]}`)
	_, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "no choices") {
		t.Fatalf("error = %v, want it to name the missing choices", err)
	}
}

// Parsed leniently: nulls where the spec says objects, an unknown finish reason,
// an unmodelled field. None of these may fail a completion the caller paid for.
func TestChat_LenientDecoding(t *testing.T) {
	body := `{
      "model": "mimo-v2.5-pro",
      "id": "chatcmpl-x",
      "some_new_field": {"nested": true},
      "choices": [{"finish_reason": "repetition_truncation",
                   "message": {"content": null, "reasoning": "thought", "tool_calls": null}}],
      "usage": null
    }`
	srv, _ := jsonServer(t, 200, body)
	resp, warnings, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "mimo-v2.5-pro",
		Messages: []Message{User("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.FinishReason != "repetition_truncation" {
		t.Errorf("finish_reason = %q; the set is open and vendor values pass through",
			resp.FinishReason)
	}
	// The second spelling of the reasoning field, as the stream parser also reads.
	if resp.Reasoning != "thought" {
		t.Errorf("reasoning = %q", resp.Reasoning)
	}
	// No usage reported: not zero, unknown — so the call is unpriced and says so.
	if resp.Usage.Input.Total != nil {
		t.Errorf("usage was invented: %+v", resp.Usage.Input)
	}
	if resp.Usage.Cost.Provenance != Unpriced {
		t.Errorf("cost = %+v, want unpriced", resp.Usage.Cost)
	}
	if len(warnings) != 1 {
		t.Errorf("warnings = %v, want one saying the cost is unknown", warnings)
	}
}

// Tool calls are assembled from the nested wire shape.
func TestChat_ToolCalls(t *testing.T) {
	body := `{
      "model": "mimo-v2.5-pro",
      "choices": [{"finish_reason": "tool_calls", "message": {"content": "",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}}]}}],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }`
	srv, _ := jsonServer(t, 200, body)
	resp, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "mimo-v2.5-pro",
		Messages: []Message{User("hi")},
		Tools:    []Tool{{Name: "search"}},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v", resp.ToolCalls)
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_1" || tc.Name != "search" || tc.Arguments != `{"q":"x"}` {
		t.Errorf("tool call = %+v", tc)
	}
}

// A body that is not JSON at all — a proxy's HTML error page under a 200 — names
// what answered instead of producing a bare "invalid character '<'".
func TestChat_UnparseableBodyCarriesTheEvidence(t *testing.T) {
	srv, _ := jsonServer(t, 200, "<html><body>502 Bad Gateway</body></html>")
	_, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "502 Bad Gateway") {
		t.Errorf("error = %v, want the body quoted", err)
	}
}

// Status codes keep their existing classification, which Chat must not swallow.
func TestChat_StatusErrors(t *testing.T) {
	t.Run("400", func(t *testing.T) {
		srv, _ := jsonServer(t, 400, `{"error":{"message":"bad","code":"1210"}}`)
		_, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
			Model:    "glm-5.3-flash",
			Messages: []Message{User("hi")},
		})
		var apiErr *APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
			t.Fatalf("error = %T %v", err, err)
		}
	})

	t.Run("429 keeps Retry-After", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", "17")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down","code":"rate_limit"}}`))
		}))
		t.Cleanup(srv.Close)
		_, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
			Model:    "glm-5.3-flash",
			Messages: []Message{User("hi")},
		})
		var rl *RateLimitError
		if !errors.As(err, &rl) {
			t.Fatalf("error = %T %v, want *RateLimitError", err, err)
		}
		if rl.RetryAfter != 17*time.Second {
			t.Errorf("RetryAfter = %v", rl.RetryAfter)
		}
	})
}

// A non-streaming call that never gets headers names the bound that fired. Before
// RawPost armed the guard this was an unnamed context deadline, which is the one
// thing the split bounds exist to avoid.
//
// The bound here is the WHOLE-CALL cap, not the header timeout: on this route the
// endpoint withholds headers until the answer is complete, so the header bound
// would be a latency limit in disguise — and profiles.yaml records 25-64s single
// calls against a 60s default.
func TestChat_HeaderStallIsNamed(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})

	c := New(Config{
		BaseURL:     srv.URL,
		Registry:    Default(),
		CallTimeout: 60 * time.Millisecond,
	})
	_, _, err := c.Chat(context.Background(), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	})
	if err == nil {
		t.Fatal("expected a timeout")
	}
	if !strings.Contains(err.Error(), stallHeaders) {
		t.Errorf("error = %v, want it to name %q", err, stallHeaders)
	}
}

// A non-streaming answer slower than the HEADER bound but inside the call cap must
// still succeed. mimo-v2.5-pro measured 25-64 seconds per call against a 60s
// default header timeout, so binding this route to c.header would abort calls the
// endpoint was about to answer.
func TestChat_SlowAnswerIsNotAHeaderStall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Deliberately longer than HeaderTimeout below, shorter than CallTimeout.
		select {
		case <-time.After(120 * time.Millisecond):
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(goodCompletion))
	}))
	t.Cleanup(srv.Close)

	c := New(Config{
		BaseURL:       srv.URL,
		Registry:      Default(),
		HeaderTimeout: 30 * time.Millisecond,
		IdleTimeout:   5 * time.Second,
		CallTimeout:   5 * time.Second,
	})
	resp, _, err := c.Chat(context.Background(), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	})
	if err != nil {
		t.Fatalf("a slow but successful answer failed: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("content = %q", resp.Content)
	}
}

// A cancelled parent context stays a cancellation, not a misreported call cap.
func TestChat_ParentCancellationIsNotACallCap(t *testing.T) {
	srv, _ := jsonServer(t, 200, goodCompletion)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := chatClient(t, srv).Chat(ctx, ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), "call cap") {
		t.Errorf("error = %v, want the caller's cancellation rather than a cap", err)
	}
}
