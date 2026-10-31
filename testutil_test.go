package llmwire

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// Shared scaffolding: the servers, clients, clocks, lookups and YAML heads the
// tests compose. Nothing here asserts; where two tests wanted the same helper
// with one detail changed, that detail is a parameter.

// --- clocks -----------------------------------------------------------------

// testStart is the instant every pinned clock starts from. A rate depends on
// wall-clock day and hour, so a test that prices anything pins it here.
var testStart = time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC)

func fixedNow() time.Time { return testStart }

// stepClock advances one second per reading, so a test can assert Timing to the
// second without depending on how fast the machine ran. The second value is
// the start instant, which the readings count from.
func stepClock() (func() time.Time, time.Time) {
	n := 0
	return func() time.Time {
		n++
		return testStart.Add(time.Duration(n) * time.Second)
	}, testStart
}

// --- servers ----------------------------------------------------------------

// testServer is an httptest server that remembers every request it saw, so a
// test can assert what went on the wire: method, header, body.
type testServer struct {
	*httptest.Server
	mu     sync.Mutex
	reqs   []*http.Request
	bodies [][]byte
}

func (s *testServer) record(r *http.Request) {
	body, _ := io.ReadAll(r.Body) // before Clone: the clone shares the Body
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reqs = append(s.reqs, r.Clone(context.Background()))
	s.bodies = append(s.bodies, body)
}

// last is the most recent request, nil before the first.
func (s *testServer) last() *http.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.reqs) == 0 {
		return nil
	}
	return s.reqs[len(s.reqs)-1]
}

// lastBody is the most recent request body, nil before the first.
func (s *testServer) lastBody() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.bodies) == 0 {
		return nil
	}
	return s.bodies[len(s.bodies)-1]
}

// requests is every request so far, in order.
func (s *testServer) requests() []*http.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*http.Request(nil), s.reqs...)
}

// serve starts a recording server around handler and closes it with the test.
func serve(t *testing.T, handler http.HandlerFunc) *testServer {
	t.Helper()
	s := &testServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.record(r)
		handler(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

// reply tunes jsonServer and sseServer beyond their body: extra response
// headers (via Set, so the name canonicalises the way Header.Get reads it), a
// delay before the status line for the timing tests, and for sseServer a gap
// after every line or one pause after the line at pauseAfter. pause > 0 arms
// the pause, because index 0 is a line like any other.
type reply struct {
	header     map[string]string
	delay      time.Duration
	gap        time.Duration
	pauseAfter int
	pause      time.Duration
}

// first is the one optional value a variadic helper takes. A second one is a
// test author's mistake, failed rather than silently dropped.
func first[T any](t *testing.T, opts []T) T {
	t.Helper()
	var zero T
	switch len(opts) {
	case 0:
		return zero
	case 1:
		return opts[0]
	default:
		t.Fatalf("one option at most, got %d", len(opts))
		return zero
	}
}

// preamble is the response head jsonServer and sseServer share: the delay,
// the content type, the extra headers, the status.
func (o reply) preamble(w http.ResponseWriter, contentType string, status int) {
	if o.delay > 0 {
		time.Sleep(o.delay)
	}
	w.Header().Set("Content-Type", contentType)
	for k, v := range o.header {
		w.Header().Set(k, v)
	}
	w.WriteHeader(status)
}

// jsonServer answers every request with one status and body as JSON.
func jsonServer(t *testing.T, status int, body string, opts ...reply) *testServer {
	t.Helper()
	o := first(t, opts)
	return serve(t, func(w http.ResponseWriter, _ *http.Request) {
		o.preamble(w, "application/json", status)
		_, _ = w.Write([]byte(body))
	})
}

// sseServer streams lines as SSE frames, flushing after every one, so the
// client sees a real incremental stream rather than one buffered write. A gap
// or pause in reply holds the next line back, the shape of a model that names
// a tool and then goes silent while it serializes the argument.
func sseServer(t *testing.T, lines []string, opts ...reply) *testServer {
	t.Helper()
	o := first(t, opts)
	return serve(t, func(w http.ResponseWriter, r *http.Request) {
		o.preamble(w, "text/event-stream", http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for i, line := range lines {
			if _, err := w.Write([]byte(line + "\n\n")); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			wait := o.gap
			if o.pause > 0 && i == o.pauseAfter {
				wait = o.pause
			}
			if wait > 0 {
				select {
				case <-time.After(wait):
				case <-r.Context().Done():
					return
				}
			}
		}
	})
}

// Frames the stream tests share.
const (
	stopFrame = `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`
	doneFrame = "data: [DONE]"
)

// hiStream is the smallest complete, accounted stream: content, finish_reason
// and usage on one frame, then [DONE].
var hiStream = []string{
	`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
	doneFrame,
}

// --- clients ----------------------------------------------------------------

// clientOn builds a client against url on the pinned clock, with the shipped
// registry unless cfg brings one; cfg carries whatever else the test sets.
func clientOn(t *testing.T, url string, cfg ...Config) *Client {
	t.Helper()
	c := first(t, cfg)
	c.BaseURL = url
	if c.Registry == nil {
		c.Registry = Default()
	}
	if c.Now == nil {
		c.Now = fixedNow
	}
	return New(c)
}

// streamClient is clientOn with the bounds every streaming test wants: idle as
// given, and a 20s call cap so a hung stream fails the test before go test does.
func streamClient(t *testing.T, url string, idle time.Duration) *Client {
	t.Helper()
	return clientOn(t, url, Config{IdleTimeout: idle, CallTimeout: 20 * time.Second})
}

// capture builds a client whose log lines land in the returned buffer, at Debug
// so the success line is visible. The clock is cfg's, wall unless pinned: the
// lines are asserted for their keys, not their figures.
func capture(t *testing.T, cfg Config) (*Client, *bytes.Buffer) {
	t.Helper()
	var buf bytes.Buffer
	if cfg.Registry == nil {
		cfg.Registry = Default()
	}
	cfg.Logger = slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	return New(cfg), &buf
}

// --- the environment --------------------------------------------------------

// mapLookup is a Config.Lookup over a map: nothing from the process.
func mapLookup(m map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := m[k]; return v, ok }
}

// gatewayEnv is the litellm trio a routed FromEnv reads: host, key and the
// route list. A map rather than a Lookup, so a test can drop or add a key
// before handing it to mapLookup.
func gatewayEnv(host, models string) map[string]string {
	return map[string]string{
		"LLMWIRE_LITELLM_BASE_URL": host,
		"LLMWIRE_LITELLM_API_KEY":  "gw-key",
		GatewayModelsEnv:           models,
	}
}

// --- the parser -------------------------------------------------------------

// readOpts tunes readSSE. The zero value parses on the wall clock with
// recovery off and fails the test on any error; keepErr hands the error back
// instead, for the fixtures that are cut streams. clock and start feed Timing;
// the helper never reads clock itself, so a stepping clock's readings are
// exactly the parser's.
type readOpts struct {
	recovery bool
	keepErr  bool
	clock    func() time.Time
	start    time.Time
}

// readSSE runs body through readStream under a guard that never fires, which
// is what lets a test assert parsing alone, and collects every event.
func readSSE(t *testing.T, body string, o readOpts) (StreamResult, []streamEvent, []Warning, error) {
	t.Helper()
	if o.clock == nil {
		o.clock, o.start = time.Now, time.Now()
	}
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var events []streamEvent
	res, warnings, err := readStream(strings.NewReader(body), guard, streamBounds{idle: time.Hour}, func(ev streamEvent) {
		events = append(events, ev)
	}, Redact, o.recovery, o.clock, o.start)
	if err != nil && !o.keepErr {
		t.Fatalf("readStream: %v", err)
	}
	return res, events, warnings, err
}

// readFixture is readSSE for the parser tests that want the result and the
// error and nothing else: cut streams are fixtures there, so the error comes
// back rather than failing the test.
func readFixture(t *testing.T, body string) (StreamResult, error) {
	t.Helper()
	res, _, _, err := readSSE(t, body, readOpts{keepErr: true})
	return res, err
}

// --- pointers ---------------------------------------------------------------

func ptr[T any](v T) *T { return &v }

// f64 and i64 spell the typed pointer where ptr(int64(5)) would read worse.
func f64(v float64) *float64 { return &v }
func i64(v int64) *int64     { return &v }

// --- fixtures ---------------------------------------------------------------

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

// The YAML heads the loader tests compose. gatewayDoc and gatewayStreamDoc are
// kept as separate values because a dozen pricing tests read the former and
// must not start streaming.
const chatHead = `profiles:
  - id: m
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
`

// textOnlyDoc keeps the shape of the retired mimo-v2.5-pro / mimo-v2.5 pair,
// the last shipped text-only model and the last with inert sampling: no shipped
// profile exercises the vision refusal or the inert-temperature warning now.
// textonly thinks by default, can be switched off, and overrides temperature
// and top_p while thinking; seeing is the same model with vision.
const textOnlyDoc = `profiles:
  - id: textonly
    wire_model_id: textonly
    max_tokens_param: max_completion_tokens
    verified: measured
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: toggle_object
      effort_values: [low, medium, high]
      balanced: medium
    temperature: {supported: true, inert_while_reasoning: true}
    top_p: {supported: true, inert_while_reasoning: true}
    tools:
      supported: true
      format: native
      tool_choice_values: [auto]
    output: {json_object: true}
    streaming: {supported: true, accepts_stream_options: true}
    vision: false
  - id: seeing
    wire_model_id: seeing
    max_tokens_param: max_completion_tokens
    verified: measured
    tools:
      supported: true
      format: native
      tool_choice_values: [auto]
    output: {json_object: true}
    streaming: {supported: true, accepts_stream_options: true}
    vision: true
`

const goodProvenance = `      source_url: https://vendor.example/pricing
      verified_on: 2026-09-12
`

const costBlock = `    cost:
      input: 0.15
      cache_read: 0.03
      cache_write: 0
      output: 0.50
` + goodProvenance

const streamingBlock = `    streaming:
      supported: true
      accepts_stream_options: true
`

// viaTail routes m through litellm under the gateway's own name.
const viaTail = `  - id: m-via
    base: m
    gateway: litellm
    wire_model_id: proxy/m
`

const gatewayDoc = chatHead + costBlock + viaTail

const gatewayStreamDoc = chatHead + streamingBlock + costBlock + viaTail

const embedHead = `profiles:
  - id: e
    endpoint: embeddings
    wire_model_id: e
    verified: measured
    embedding: {default_dimensions: 8}
`

// syncDoc is a model whose profile says it cannot stream.
const syncDoc = `profiles:
  - id: sync
    wire_model_id: sync
    max_tokens_param: max_tokens
    verified: source-derived
`
