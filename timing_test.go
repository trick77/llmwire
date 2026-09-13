package llmwire

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// stepClock advances one second per reading, so a test can assert Timing to the
// second without depending on how fast the machine ran.
func stepClock() (func() time.Time, time.Time) {
	t0 := time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC)
	n := 0
	return func() time.Time {
		n++
		return t0.Add(time.Duration(n) * time.Second)
	}, t0
}

// Readings: lastData t1, frame 1 at t2, frame 2 at t3, [DONE] at t4, the
// deferred Total at t5. FirstData counts from start, not from the first reading.
func TestReadStream_TimingFromClock(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"a"}}]}

data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}

data: [DONE]

`
	now, start := stepClock()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var counters streamCounters
	res, _, err := readStream(strings.NewReader(body), guard, &counters, time.Hour, nil, Redact, false, now, start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Timing.FirstData != 2*time.Second {
		t.Errorf("FirstData = %v, want 2s", res.Timing.FirstData)
	}
	if res.Timing.Total != 5*time.Second {
		t.Errorf("Total = %v, want 5s", res.Timing.Total)
	}
	// Headers is the caller's to set; the parser never saw them.
	if res.Timing.Headers != 0 {
		t.Errorf("Headers = %v, want 0 from the parser", res.Timing.Headers)
	}
}

// A stream that dies without finish_reason is an error, and the partial result
// that rides back with it still says how long the call took: that figure beside
// the named bound is what settles whether the bound or the endpoint was wrong.
func TestReadStream_TimingSetOnErrorPath(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"a"}}]}

`
	now, start := stepClock()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var counters streamCounters
	res, _, err := readStream(strings.NewReader(body), guard, &counters, time.Hour, nil, Redact, false, now, start)
	if err == nil {
		t.Fatal("expected a truncation error")
	}
	if res.Timing.FirstData != 2*time.Second || res.Timing.Total != 3*time.Second {
		t.Errorf("Timing = %+v, want FirstData 2s, Total 3s", res.Timing)
	}
}

// Through the real transport the figures must be ordered and bounded below by
// the delays the server imposed: Headers covers the pause before the status
// line, FirstData the pause before the first frame, Total the whole body.
func TestRawStream_TimingOrderedAndBounded(t *testing.T) {
	const headerDelay, frameGap = 30 * time.Millisecond, 20 * time.Millisecond
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(headerDelay)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		flusher.Flush()
		time.Sleep(frameGap)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"))
		flusher.Flush()
		time.Sleep(frameGap)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	c := New(Config{BaseURL: srv.URL})
	res, err := c.RawStream(context.Background(), []byte(`{}`), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tm := res.Timing
	if tm.Headers < headerDelay {
		t.Errorf("Headers = %v, want >= %v", tm.Headers, headerDelay)
	}
	if tm.FirstData < tm.Headers+frameGap {
		t.Errorf("FirstData = %v, want >= Headers %v + %v", tm.FirstData, tm.Headers, frameGap)
	}
	if tm.Total < tm.FirstData+frameGap {
		t.Errorf("Total = %v, want >= FirstData %v + %v", tm.Total, tm.FirstData, frameGap)
	}
}

func TestChatStream_CollectCarriesTiming(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}, 10*time.Millisecond)
	c := New(Config{BaseURL: srv.URL, Registry: Default()})
	s, _, err := c.ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer s.Close()
	res, err := s.Collect(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tm := res.Timing
	if tm.Headers <= 0 || tm.FirstData < tm.Headers || tm.Total < tm.FirstData {
		t.Errorf("Timing = %+v, want 0 < Headers <= FirstData <= Total", tm)
	}
}

// A non-streaming route withholds headers until the answer is ready, so Headers
// is the model's whole latency and FirstData has nothing to measure.
func TestChat_TimingHeadersIsTheAnswerLatency(t *testing.T) {
	const delay = 30 * time.Millisecond
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(delay)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)

	c := New(Config{BaseURL: srv.URL, Registry: Default()})
	resp, _, err := c.Chat(context.Background(), hiRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tm := resp.Timing
	if tm.Headers < delay {
		t.Errorf("Headers = %v, want >= %v", tm.Headers, delay)
	}
	if tm.Total < tm.Headers {
		t.Errorf("Total = %v, want >= Headers %v", tm.Total, tm.Headers)
	}
	if tm.FirstData != 0 {
		t.Errorf("FirstData = %v, want 0 on a non-streaming route", tm.FirstData)
	}
}

// Two batches, each delayed: the call's Timing is the sum, not the last batch.
func TestEmbed_TimingSumsOverBatches(t *testing.T) {
	const delay = 20 * time.Millisecond
	inner, _ := embedServer(t, false)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(delay)
		inner.Config.Handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	c := New(Config{BaseURL: srv.URL, Registry: Default()})
	resp, _, err := c.Embed(context.Background(), EmbedRequest{Model: "text-embedding-3-small", Inputs: inputs(embedBatchSize + 1)})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Timing.Headers < 2*delay {
		t.Errorf("Headers = %v, want >= %v summed over two batches", resp.Timing.Headers, 2*delay)
	}
	if resp.Timing.Total < resp.Timing.Headers {
		t.Errorf("Total = %v, want >= Headers %v", resp.Timing.Total, resp.Timing.Headers)
	}
}
