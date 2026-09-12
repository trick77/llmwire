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

// sseServer serves a fixed SSE body.
func sseServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestRawStream_HappyPath(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}

data: [DONE]

`)
	c := New(Config{BaseURL: srv.URL})
	res, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Content != "hello" {
		t.Errorf("content = %q, want hello", res.Content)
	}
	if got := res.Usage.Input.Total; got == nil || *got != 3 {
		t.Errorf("usage not captured: %v", deref(got))
	}
}

// An empty API key must send NO Authorization header rather than an empty one,
// which some gateways reject outright.
func TestRawStream_EmptyAPIKeySendsNoAuthorizationHeader(t *testing.T) {
	var sawAuth, hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hadAuth = r.Header["Authorization"]
		sawAuth = true
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
	}))
	defer srv.Close()

	c := New(Config{BaseURL: srv.URL})
	if _, err := c.RawStream(context.Background(), []byte(`{}`), nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !sawAuth {
		t.Fatal("handler never ran")
	}
	if hadAuth {
		t.Error("an empty APIKey must not produce an Authorization header")
	}
}

func TestRawStream_SendsConfiguredHeaders(t *testing.T) {
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
	}))
	defer srv.Close()

	c := New(Config{
		BaseURL:   srv.URL,
		APIKey:    "secret",
		UserAgent: "custom/1.0",
		Headers:   map[string]string{"X-Session-Id": "ses_abc"},
	})
	if _, err := c.RawStream(context.Background(), []byte(`{}`), nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Get("Authorization") != "Bearer secret" {
		t.Errorf("Authorization = %q", got.Get("Authorization"))
	}
	if got.Get("User-Agent") != "custom/1.0" {
		t.Errorf("User-Agent = %q", got.Get("User-Agent"))
	}
	if got.Get("X-Session-Id") != "ses_abc" {
		t.Errorf("X-Session-Id = %q", got.Get("X-Session-Id"))
	}
	if got.Get("Accept") != "text/event-stream" {
		t.Errorf("Accept = %q", got.Get("Accept"))
	}
	// Accept-Encoding is deliberately left unset so net/http keeps negotiating
	// and decompressing gzip itself. Setting it by hand hands us a compressed
	// body to decode mid-stream.
	if _, set := got["Accept-Encoding"]; set && got.Get("Accept-Encoding") != "gzip" {
		t.Errorf("Accept-Encoding = %q, want unset or net/http's own gzip", got.Get("Accept-Encoding"))
	}
}

// The base URL must not already end in the route.
func TestRawStream_AppendsRouteToBaseURL(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
	}))
	defer srv.Close()

	// A trailing slash on the base must not double up.
	c := New(Config{BaseURL: srv.URL + "/"})
	if _, err := c.RawStream(context.Background(), []byte(`{}`), nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", path)
	}
}

func TestRawStream_ErrorStatusIsDecoded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"thinking cannot be disabled","code":"1210"}}`))
	}))
	defer srv.Close()

	c := New(Config{BaseURL: srv.URL})
	_, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error is %T, want *APIError", err)
	}
	if apiErr.Code != "1210" {
		t.Errorf("code = %q, want 1210", apiErr.Code)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("status = %d, want 400", apiErr.StatusCode)
	}
}

// A 429 carries Retry-After when the endpoint sends one. The library parses it
// and stops there: it never retries on its own, because a library-level retry
// turns a transient outage into a permanent failure for a whole job queue.
func TestRawStream_RateLimitCarriesRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "17")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down","code":"1302"}}`))
	}))
	defer srv.Close()

	c := New(Config{BaseURL: srv.URL})
	_, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	var rl *RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("error is %T, want *RateLimitError", err)
	}
	if rl.RetryAfter != 17*time.Second {
		t.Errorf("RetryAfter = %v, want 17s", rl.RetryAfter)
	}
	if !errors.Is(err, ErrRateLimited) {
		t.Error("should match ErrRateLimited")
	}
}

// An error body carrying a credential must not reach the caller's log.
func TestRawStream_ErrorBodyIsRedacted(t *testing.T) {
	// Assembled, not a literal — see fakeKey in errors_test.go.
	key := fakeKey("sk", "abcdefghijklmnopqrstuvwxyz01")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid key ` + key + `"}}`))
	}))
	defer srv.Close()

	c := New(Config{BaseURL: srv.URL})
	_, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), key) {
		t.Errorf("error leaked the key: %v", err)
	}
}

// The header bound and the idle bound mean different things, and a failure has
// to say which one gave up — that naming is the deliverable of splitting them.
func TestRawStream_HeaderStallNamesItsBound(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	c := New(Config{BaseURL: srv.URL, HeaderTimeout: 50 * time.Millisecond})
	_, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected a stall error")
	}
	if !strings.Contains(err.Error(), stallHeaders) {
		t.Errorf("error = %v, want it to name %q", err, stallHeaders)
	}
}

// Headers arrive promptly, then the stream goes silent. That is a different
// failure from the endpoint never answering, and must be named differently.
func TestRawStream_IdleStallNamesItsBound(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"start\"},\"finish_reason\":null}]}\n\n"))
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	c := New(Config{BaseURL: srv.URL, HeaderTimeout: 5 * time.Second, IdleTimeout: 60 * time.Millisecond})
	_, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected a stall error")
	}
	if !strings.Contains(err.Error(), stallIdle) {
		t.Errorf("error = %v, want it to name %q", err, stallIdle)
	}
}

// Comment lines prove the connection is alive but say nothing about the model
// making progress. They deliberately do NOT re-arm the idle guard, or a
// heartbeat-emitting upstream masks a model that has stalled completely.
func TestRawStream_KeepaliveCommentsDoNotReArmTheIdleGuard(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		fl.Flush()
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n"))
		fl.Flush()
		// Ping forever, send no further data frame.
		for {
			select {
			case <-release:
				return
			case <-r.Context().Done():
				return
			case <-time.After(10 * time.Millisecond):
				if _, err := w.Write([]byte(": ping\n\n")); err != nil {
					return
				}
				fl.Flush()
			}
		}
	}))
	defer srv.Close()
	defer close(release)

	c := New(Config{BaseURL: srv.URL, HeaderTimeout: 5 * time.Second, IdleTimeout: 80 * time.Millisecond})
	start := time.Now()
	_, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected the idle guard to fire despite the keepalives")
	}
	if !strings.Contains(err.Error(), stallIdle) {
		t.Errorf("error = %v, want it to name %q", err, stallIdle)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("took %v — the keepalives appear to be re-arming the guard", elapsed)
	}
}

// A stream that stays alive forever without finishing still has to end.
func TestRawStream_CallCapNamesItself(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl := w.(http.Flusher)
		fl.Flush()
		for {
			select {
			case <-release:
				return
			case <-r.Context().Done():
				return
			case <-time.After(5 * time.Millisecond):
				if _, err := w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\".\"},\"finish_reason\":null}]}\n\n")); err != nil {
					return
				}
				fl.Flush()
			}
		}
	}))
	defer srv.Close()
	defer close(release)

	c := New(Config{
		BaseURL:       srv.URL,
		HeaderTimeout: 5 * time.Second,
		IdleTimeout:   5 * time.Second,
		CallTimeout:   120 * time.Millisecond,
	})
	_, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected the call cap to fire")
	}
	if !strings.Contains(err.Error(), "call cap") {
		t.Errorf("error = %v, want it to name the call cap", err)
	}
}

// A caller cancelling is not a stall, and must not be reported as one.
func TestRawStream_ParentCancellationIsNotReportedAsAStall(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()

	c := New(Config{BaseURL: srv.URL, HeaderTimeout: 10 * time.Second, CallTimeout: 10 * time.Second})
	_, err := c.RawStream(ctx, []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), stallHeaders) || strings.Contains(err.Error(), "call cap") {
		t.Errorf("error = %v, want the caller's own cancellation, not a bound", err)
	}
}

// --- RawPost -----------------------------------------------------------------

func TestRawPost_ReturnsBodyAndHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A gateway reports per-call spend and the real deployment name in
		// headers and nowhere else, which is why RawPost hands them back.
		w.Header().Set("x-litellm-response-cost", "1.23e-05")
		w.Header().Set("x-litellm-model-name", "azure/gpt-5.4")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2]}]}`))
	}))
	defer srv.Close()

	c := New(Config{BaseURL: srv.URL})
	raw, hdr, err := c.RawPost(context.Background(), "/embeddings", []byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var decoded struct {
		Data []struct {
			Index int `json:"index"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(decoded.Data) != 1 {
		t.Errorf("got %d embeddings, want 1", len(decoded.Data))
	}
	if hdr.Get("x-litellm-model-name") != "azure/gpt-5.4" {
		t.Error("response headers were not returned")
	}
}

func TestRawPost_ErrorStatusIsDecodedAndHeadersStillReturned(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-litellm-call-id", "abc123")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad model","code":"1211"}}`))
	}))
	defer srv.Close()

	c := New(Config{BaseURL: srv.URL})
	_, hdr, err := c.RawPost(context.Background(), "/embeddings", []byte(`{}`))
	if err == nil {
		t.Fatal("expected an error")
	}
	// The call id is how a streamed call's cost is looked up afterwards, so it
	// must survive a failure.
	if hdr.Get("x-litellm-call-id") != "abc123" {
		t.Error("headers must still be returned on an error response")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "1211" {
		t.Errorf("error = %v", err)
	}
}

// --- config ------------------------------------------------------------------

func TestNew_AppliesDefaults(t *testing.T) {
	c := New(Config{BaseURL: "https://example.test"})
	if c.header != DefaultHeaderTimeout || c.idle != DefaultIdleTimeout || c.cap != DefaultCallTimeout {
		t.Errorf("bounds = %v/%v/%v, want the documented defaults", c.header, c.idle, c.cap)
	}
	if c.userAgent != DefaultUserAgent {
		t.Errorf("user agent = %q", c.userAgent)
	}
	if c.Now().IsZero() {
		t.Error("clock was not defaulted")
	}
}

// The clock is injectable because cost depends on wall-clock time on endpoints
// with off-peak pricing, and a test that only passes during a particular local
// window is worse than no test.
func TestNew_ClockIsInjectable(t *testing.T) {
	fixed := time.Date(2026, 9, 12, 15, 0, 0, 0, time.UTC)
	c := New(Config{BaseURL: "https://example.test", Now: func() time.Time { return fixed }})
	if !c.Now().Equal(fixed) {
		t.Errorf("Now() = %v, want the injected %v", c.Now(), fixed)
	}
}

// http.Client.Timeout must never be set: it caps body reads too, so it would
// cut a legitimately long stream mid-answer.
func TestNew_DefaultHTTPClientHasNoWholeRequestTimeout(t *testing.T) {
	c := New(Config{BaseURL: "https://example.test"})
	if c.http.Timeout != 0 {
		t.Errorf("http.Client.Timeout = %v, want 0 — it would truncate a long stream", c.http.Timeout)
	}
	tr, ok := c.http.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport is %T, want *http.Transport", c.http.Transport)
	}
	// Set as a backstop only; the stall guard is the bound actually relied on,
	// because ResponseHeaderTimeout does not apply over HTTP/2.
	if tr.ResponseHeaderTimeout != DefaultHeaderTimeout {
		t.Errorf("ResponseHeaderTimeout = %v, want the configured header bound", tr.ResponseHeaderTimeout)
	}
}

// Mutating the caller's header map after New must not change what is sent.
func TestNew_CopiesTheHeaderMap(t *testing.T) {
	h := map[string]string{"X-A": "1"}
	c := New(Config{BaseURL: "https://example.test", Headers: h})
	h["X-A"] = "mutated"
	if c.headers["X-A"] != "1" {
		t.Error("New must copy the header map")
	}
}
