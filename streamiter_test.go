package llmwire

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// flushingServer writes each line, flushing after every one, so the client sees a
// real incremental stream rather than one buffered write.
func flushingServer(t *testing.T, lines []string, gap time.Duration) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		for _, line := range lines {
			if _, err := w.Write([]byte(line + "\n\n")); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			if gap > 0 {
				select {
				case <-time.After(gap):
				case <-r.Context().Done():
					return
				}
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func streamClient(t *testing.T, srv *httptest.Server, idle time.Duration) *Client {
	t.Helper()
	return New(Config{
		BaseURL:     srv.URL,
		Registry:    Default(),
		IdleTimeout: idle,
		CallTimeout: 20 * time.Second,
		Now:         func() time.Time { return time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC) },
	})
}

func hiRequest() ChatRequest {
	return ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}}
}

func TestChatStream_EventsThenResult(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}`,
		`data: {"choices":[{"delta":{"content":"hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":258,"completion_tokens":100,` +
			`"prompt_tokens_details":{"cached_tokens":192}}}`,
		"data: [DONE]",
	}, 0)

	s, warnings, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("validation warnings = %v", warnings)
	}
	defer s.Close()

	var kinds []EventKind
	var content string
	for s.Next() {
		ev := s.Event()
		kinds = append(kinds, ev.Kind)
		if ev.Kind == EventContent {
			content += ev.Text
		}
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if content != "hello" {
		t.Errorf("content = %q", content)
	}
	want := []EventKind{EventReasoning, EventContent, EventContent, EventFinish}
	if fmt.Sprint(kinds) != fmt.Sprint(want) {
		t.Errorf("kinds = %v, want %v", kinds, want)
	}

	res := s.Result()
	if res.Content != "hello" || res.Reasoning != "thinking" || res.FinishReason != "stop" {
		t.Errorf("result = %+v", res)
	}
	if !res.Done {
		t.Error("[DONE] was not recorded")
	}
	// Usage arrives in the trailing chunk, so it is only valid here — and it is
	// priced, with the warnings that came with it.
	u := s.Usage()
	if u.Input.CacheRead == nil || *u.Input.CacheRead != 192 {
		t.Errorf("cache read = %v", u.Input.CacheRead)
	}
	if u.Cost.Provenance != FromTable || u.Cost.NanoUSD != 65_660 {
		t.Errorf("cost = %+v, want a from-table 65660", u.Cost)
	}
}

// Tool-call fragments reach the caller as fragments AND are assembled on the
// result, so a caller can either stream them or wait.
func TestChatStream_ToolCallFragments(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",` +
			`"function":{"name":"search","arguments":"{\"q\":"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},` +
			`"finish_reason":"tool_calls"}]}`,
		"data: [DONE]",
	}, 0)

	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []Message{User("hi")},
		Tools:    []Tool{{Name: "search"}},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	var args string
	var sawID bool
	for s.Next() {
		if ev := s.Event(); ev.Kind == EventToolCall {
			args += ev.ArgumentsDelta
			if ev.ID == "call_1" && ev.Name == "search" {
				sawID = true
			}
		}
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if !sawID {
		t.Error("the first fragment carried neither id nor name")
	}
	if args != `{"q":"x"}` {
		t.Errorf("streamed arguments = %q", args)
	}
	calls := s.Result().ToolCalls
	if len(calls) != 1 || calls[0].Arguments != `{"q":"x"}` {
		t.Errorf("assembled calls = %+v", calls)
	}
}

// THE test for the queue's design. A consumer far slower than the idle bound must
// not kill the call: the idle guard measures the ENDPOINT's silence, and charging
// consumer latency to it would abort a stream whose model was producing perfectly.
func TestChatStream_SlowConsumerIsNotAStall(t *testing.T) {
	lines := make([]string, 0, 12)
	for i := 0; i < 10; i++ {
		lines = append(lines, fmt.Sprintf(`data: {"choices":[{"delta":{"content":"%d"}}]}`, i))
	}
	lines = append(lines,
		`data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}`,
		"data: [DONE]")
	srv := flushingServer(t, lines, 0)

	// An idle bound far shorter than the consumer's own pace.
	s, _, err := streamClient(t, srv, 60*time.Millisecond).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	var got string
	for s.Next() {
		if ev := s.Event(); ev.Kind == EventContent {
			got += ev.Text
			time.Sleep(25 * time.Millisecond)
		}
	}
	if err := s.Err(); err != nil {
		t.Fatalf("a slow consumer was reported as a stream failure: %v", err)
	}
	if got != "0123456789!" {
		t.Errorf("content = %q", got)
	}
}

// A genuinely silent endpoint is still caught, and the error names the bound.
func TestChatStream_StalledStreamIsNamed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	s, _, err := streamClient(t, srv, 50*time.Millisecond).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	for s.Next() {
	}
	if err := s.Err(); err == nil || !strings.Contains(err.Error(), stallIdle) {
		t.Fatalf("Err = %v, want it to name %q", err, stallIdle)
	}
}

// An early Close is not a failure. Before this, explain() reported the
// cancellation Close causes as "exceeded the call cap" — a timeout that never
// happened — and the truncation guard called the deliberate stop a truncated
// answer.
func TestChatStream_EarlyCloseIsNotAnError(t *testing.T) {
	lines := make([]string, 0, 50)
	for i := 0; i < 50; i++ {
		lines = append(lines, fmt.Sprintf(`data: {"choices":[{"delta":{"content":"%d"}}]}`, i))
	}
	srv := flushingServer(t, lines, 10*time.Millisecond)

	s, _, err := streamClient(t, srv, 5*time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if !s.Next() {
		t.Fatalf("no first event: %v", s.Err())
	}
	if err := s.Close(); err != nil {
		t.Errorf("Close reported %v, want nil for a deliberate stop", err)
	}
	// Idempotent, so a defer after an explicit close is safe.
	if err := s.Close(); err != nil {
		t.Errorf("second Close reported %v", err)
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err after a deliberate close = %v, want nil", err)
	}
	// The goroutine is gone: Close waits for it, so nothing leaks into the next
	// test or into a caller's process.
	select {
	case <-s.done:
	default:
		t.Error("the reading goroutine is still running after Close returned")
	}
}

// An error frame inside a 200 reaches the caller as an API error rather than as a
// short, clean answer.
func TestChatStream_ErrorFrameInsideA200(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"partial"}}]}`,
		`data: {"error":{"message":"upstream exploded","code":"1210"}}`,
	}, 0)

	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()
	for s.Next() {
	}
	var apiErr *APIError
	if !errors.As(s.Err(), &apiErr) {
		t.Fatalf("Err = %T %v, want *APIError", s.Err(), s.Err())
	}
	if apiErr.Code != "1210" {
		t.Errorf("code = %q", apiErr.Code)
	}
}

// A stream that just stops is an error, not a short answer: a dropped connection
// ends the scan exactly like a finished one.
func TestChatStream_TruncatedStreamIsAnError(t *testing.T) {
	srv := flushingServer(t, []string{`data: {"choices":[{"delta":{"content":"half"}}]}`}, 0)

	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()
	for s.Next() {
	}
	if err := s.Err(); err == nil || !strings.Contains(err.Error(), "without finish_reason") {
		t.Fatalf("Err = %v, want a truncation error", err)
	}
}

// Everything that can fail before the first byte fails up front, on the third
// return value, rather than being buried in Err() after a pointless Next().
func TestChatStream_PreFlightFailuresReturnImmediately(t *testing.T) {
	t.Run("validation refusal sends nothing", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			t.Error("a refused request was sent anyway")
		}))
		t.Cleanup(srv.Close)
		s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), ChatRequest{
			Model:     "glm-5.3-flash",
			Messages:  []Message{User("hi")},
			Reasoning: ReasoningOff(),
		})
		if s != nil {
			t.Error("a refused request returned a stream")
		}
		var unsup *UnsupportedError
		if !errors.As(err, &unsup) {
			t.Fatalf("err = %T %v", err, err)
		}
	})

	t.Run("401 arrives as an error, not as an empty stream", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"bad key","code":"401"}}`))
		}))
		t.Cleanup(srv.Close)
		s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
		if s != nil {
			t.Error("a 401 returned a stream")
		}
		if !errors.Is(err, ErrAuth) {
			t.Fatalf("err = %v, want an auth error", err)
		}
	})
}

// The rendered body carries stream:true — RawStream does not inject it, and
// neither does ChatStream: the renderer does, from the plan.
func TestChatStream_BodyRequestsStreaming(t *testing.T) {
	var seen []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = readAllBody(r)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for s.Next() {
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !strings.Contains(string(seen), `"stream":true`) {
		t.Errorf("request body did not ask for a stream: %s", seen)
	}
}

// A model whose profile says it cannot stream is refused by the METHOD, and
// BestEffort cannot rescue it: there is nothing to demote a method choice to.
func TestChatStream_NonStreamingModelIsRefused(t *testing.T) {
	reg := registryFrom(t, `profiles:
  - id: sync
    wire_model_id: sync
    max_tokens_param: max_tokens
    verified: source-derived
`)
	c := New(Config{BaseURL: "https://example.invalid", Registry: reg})
	_, _, err := c.ChatStream(context.Background(), ChatRequest{
		Model:      "sync",
		Messages:   []Message{User("hi")},
		BestEffort: true,
	})
	if err == nil {
		t.Fatal("expected a refusal")
	}
	if !strings.Contains(err.Error(), "does not stream") {
		t.Errorf("err = %v", err)
	}
}

// Collect is the whole stream at once: every content delta to the callback in
// order, then the result the caller would otherwise assemble by hand.
func TestChatStream_CollectDrainsToTheResult(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}`,
		`data: {"choices":[{"delta":{"content":"hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"length"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}`,
		"data: [DONE]",
	}, 0)
	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	var deltas []string
	res, err := s.Collect(func(text string) { deltas = append(deltas, text) })
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if fmt.Sprint(deltas) != fmt.Sprint([]string{"hel", "lo"}) {
		t.Errorf("deltas = %q", deltas)
	}
	if res.Content != "hello" || res.FinishReason != "length" || res.Reasoning != "thinking" {
		t.Errorf("result = %+v", res)
	}
	if total, ok := res.Usage.Total(); !ok || total != 12 {
		t.Errorf("usage total = %d, %v; want 12", total, ok)
	}
}

// A stream that breaks still hands back what arrived: the deltas were
// delivered as they came, and the result carries whatever usage was read, so
// a caller can account for a call that was paid for and then cut.
func TestChatStream_CollectReturnsTheErrorWithThePartialResult(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"partial"}}]}`,
	}, 0)
	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	var got string
	res, err := s.Collect(func(text string) { got += text })
	if !errors.Is(err, ErrMalformedResponse) {
		t.Fatalf("err = %v, want ErrMalformedResponse for a stream cut before finish_reason", err)
	}
	if got != "partial" || res.Content != "partial" {
		t.Errorf("delivered %q, result %q; want the partial content in both", got, res.Content)
	}
	if res.Usage.Reported() {
		t.Error("usage reported on a stream that carried none")
	}
}

func TestChatStream_CollectWithoutACallbackStillCollects(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		"data: [DONE]",
	}, 0)
	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()
	res, err := s.Collect(nil)
	if err != nil || res.Content != "ok" {
		t.Errorf("result = %+v, err = %v", res, err)
	}
}

// gatewayStreamDoc is gatewayDoc's shape with streaming declared, so the
// gateway route can be driven through ChatStream. Kept separate rather than
// adding streaming to gatewayDoc, which a dozen pricing tests read.
const gatewayStreamDoc = `profiles:
  - id: m
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
    streaming:
      supported: true
      accepts_stream_options: true
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

// The whole point of the usage.cost lane, end to end: a STREAM behind a gateway
// carries no cost header (it is written before the body is consumed), so without
// the field on the final chunk the most expensive call in a turn is unpriced.
func TestChatStream_PricesAGatewayStreamFromTheUsageBody(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,"cost":0.0000123}}`,
		"data: [DONE]",
	}, 0)
	c := New(Config{
		BaseURL:     srv.URL,
		Registry:    registryFrom(t, gatewayStreamDoc),
		CallTimeout: 20 * time.Second,
		Now:         func() time.Time { return time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC) },
	})
	s, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "m-via", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	res, err := s.Collect(nil)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if res.Usage.Cost.Provenance != Reported {
		t.Errorf("provenance = %v, want reported", res.Usage.Cost.Provenance)
	}
	if res.Usage.Cost.NanoUSD != 12_300 {
		t.Errorf("cost = %d nano, want 12300", res.Usage.Cost.NanoUSD)
	}
}

// Without the field the call stays unpriced — never zero, and never the table
// rate, which behind a gateway would be a confident figure for a call nobody
// priced. This is intg's state until the gateway sets
// include_cost_in_streaming_usage.
func TestChatStream_AGatewayStreamWithoutTheFieldStaysUnpriced(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10}}`,
		"data: [DONE]",
	}, 0)
	c := New(Config{
		BaseURL:     srv.URL,
		Registry:    registryFrom(t, gatewayStreamDoc),
		CallTimeout: 20 * time.Second,
		Now:         func() time.Time { return time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC) },
	})
	s, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "m-via", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()

	res, err := s.Collect(nil)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if res.Usage.Cost.Provenance != Unpriced || res.Usage.Cost.NanoUSD != 0 {
		t.Fatalf("cost = %+v, want unpriced", res.Usage.Cost)
	}
}

// A caller who Closes after EventFinish has, on an endpoint that sends usage
// in the finish chunk, already been handed the figures. The call was paid for,
// so it is priced: skipping the pricing on a deliberate close left the cost at
// the zero value Unpriced with no warning, against the rule that Unpriced
// always warns.
func TestChatStream_EarlyCloseStillPricesReportedUsage(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`,
		`data: {"choices":[{"delta":{"content":"late"}}]}`,
		`data: {"choices":[{"delta":{"content":"later"}}]}`,
	}, 20*time.Millisecond)

	s, _, err := streamClient(t, srv, 5*time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for s.Next() {
		if s.Event().Kind == EventFinish {
			break
		}
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	res := s.Result()
	if !res.Usage.Reported() {
		t.Fatal("usage that arrived with the finish chunk was lost on Close")
	}
	if res.Usage.Cost.Provenance != FromTable || res.Usage.Cost.NanoUSD == 0 {
		t.Errorf("cost = %+v, want a table price for usage the stream had already reported", res.Usage.Cost)
	}
}

// A closed stream that carried no usage at all is still reported as Unpriced
// WITH the warning, on Warnings(): the zero cost is never silent.
func TestChatStream_EarlyCloseWithoutUsageWarnsUnpriced(t *testing.T) {
	lines := make([]string, 0, 20)
	for i := 0; i < 20; i++ {
		lines = append(lines, `data: {"choices":[{"delta":{"content":"x"}}]}`)
	}
	srv := flushingServer(t, lines, 20*time.Millisecond)
	s, _, err := streamClient(t, srv, 5*time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if !s.Next() {
		t.Fatalf("no first event: %v", s.Err())
	}
	_ = s.Close()
	var found bool
	for _, w := range s.Warnings() {
		if w.Feature == "cost" {
			found = true
		}
	}
	if !found {
		t.Errorf("warnings = %v, want the Unpriced warning on a closed stream that reported no usage", s.Warnings())
	}
}

// The error frame ends the scan, not the assembly: what arrived before it is
// returned beside the error, as Collect promises for every other cut.
func TestChatStream_ErrorFrameKeepsThePartialResult(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"partial"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{}"}}]}}]}`,
		`data: {"error":{"message":"upstream exploded","code":"1210"}}`,
	}, 0)
	s, _, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()
	res, err := s.Collect(nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "1210" {
		t.Fatalf("err = %v, want the frame's *APIError", err)
	}
	if res.Content != "partial" || len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "f" {
		t.Errorf("result = %+v, want the content and the tool call that arrived before the error", res)
	}
}

// The warnings ChatStream returns are the caller's own: appending to them
// must not race the reader appending pricing warnings to the stream's copy.
// Caught by -race when the two shared one backing array.
func TestChatStream_ReturnedWarningsAreTheCallersOwn(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		"data: [DONE]",
	}, 0)
	// A request that produces a validation warning, so the slice has content
	// and spare capacity.
	req := hiRequest()
	req.ToolChoice = ToolChoice{Mode: ToolChoiceAuto}
	s, warnings, err := streamClient(t, srv, time.Second).ChatStream(context.Background(), req)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer s.Close()
	if len(warnings) == 0 {
		t.Fatal("expected a validation warning to start from")
	}
	for i := 0; i < 8; i++ {
		warnings = append(warnings, Warning{Kind: WarnOther, Feature: "caller"})
	}
	for s.Next() {
	}
	got := s.Warnings()
	for _, w := range got {
		if w.Feature == "caller" {
			t.Fatal("the caller's append reached the stream's own warnings")
		}
	}
	// And the other way: the caller's copy of Warnings() is theirs too. The
	// append is to the returned slice's backing array, and the check is
	// whether the stream saw it.
	_ = append(got, Warning{Feature: "caller"})
	for _, w := range s.Warnings() {
		if w.Feature == "caller" {
			t.Fatal("appending to Warnings() wrote into the stream")
		}
	}
}
