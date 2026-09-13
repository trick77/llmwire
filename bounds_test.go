package llmwire

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// pausingServer streams lines, flushing each, and sleeps `pause` right after the
// line at index pauseAfter — the shape of a model that names a tool and then
// goes silent while it serializes the argument.
func pausingServer(t *testing.T, lines []string, pauseAfter int, pause time.Duration) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		for i, line := range lines {
			if _, err := w.Write([]byte(line + "\n\n")); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			if i == pauseAfter {
				select {
				case <-time.After(pause):
				case <-r.Context().Done():
					return
				}
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

var toolThenSilence = []string{
	`data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}`,
	`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_pdf_file","arguments":""}}]}}]}`,
	`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"blocks\":[]}"}}]},"finish_reason":"tool_calls"}]}`,
	`data: [DONE]`,
}

func drain(t *testing.T, c *Client, req ChatRequest) (StreamResult, error) {
	t.Helper()
	stream, _, err := c.ChatStream(context.Background(), req)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	defer stream.Close()
	for stream.Next() {
	}
	return stream.Result(), stream.Err()
}

// Without the per-request bound the silence after the tool name trips the idle
// guard; with it, the same stream completes.
func TestToolCallIdleTimeout_WidensAfterFirstFragment(t *testing.T) {
	srv := pausingServer(t, toolThenSilence, 1, 300*time.Millisecond)
	c := streamClient(t, srv, 100*time.Millisecond)

	_, err := drain(t, c, hiRequest())
	if !errors.Is(err, ErrStreamIdle) {
		t.Fatalf("without the bound: err = %v, want ErrStreamIdle", err)
	}
	if !strings.Contains(err.Error(), "stream idle for 100ms") {
		t.Errorf("message = %q, want the bound named", err)
	}

	req := hiRequest()
	req.ToolCallIdleTimeout = 2 * time.Second
	res, err := drain(t, c, req)
	if err != nil {
		t.Fatalf("with the bound: %v", err)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != `{"blocks":[]}` {
		t.Fatalf("tool calls = %#v", res.ToolCalls)
	}
}

// The narrow bound still guards the phase BEFORE a tool call: silence during
// reasoning is a stall whatever the per-request bound says.
func TestToolCallIdleTimeout_DoesNotWidenBeforeAToolCall(t *testing.T) {
	srv := pausingServer(t, toolThenSilence, 0, 300*time.Millisecond)
	c := streamClient(t, srv, 100*time.Millisecond)
	req := hiRequest()
	req.ToolCallIdleTimeout = 2 * time.Second
	if _, err := drain(t, c, req); !errors.Is(err, ErrStreamIdle) {
		t.Fatalf("err = %v, want ErrStreamIdle", err)
	}
}

// Inline markup means the same silence on the model that emits it, so the gate
// seeing a marker widens too.
func TestToolCallIdleTimeout_WidensOnInlineMarker(t *testing.T) {
	lines := []string{
		`data: {"choices":[{"delta":{"content":"<tool_call>\n<function=create_pdf_file>"}}]}`,
		`data: {"choices":[{"delta":{"content":"<parameter=blocks>[]</parameter></function></tool_call>"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}
	srv := pausingServer(t, lines, 0, 300*time.Millisecond)
	c := streamClient(t, srv, 100*time.Millisecond)
	req := ChatRequest{Model: "mimo-v2.5-pro", Messages: []Message{User("hi")}, ToolCallIdleTimeout: 2 * time.Second}
	res, err := drain(t, c, req)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "create_pdf_file" {
		t.Fatalf("tool calls = %#v", res.ToolCalls)
	}
}

func TestBoundSentinels(t *testing.T) {
	// Headers never arrive.
	release := make(chan struct{})
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(slow.Close)
	t.Cleanup(func() { close(release) })
	c := New(Config{BaseURL: slow.URL, HeaderTimeout: 50 * time.Millisecond, IdleTimeout: time.Second, CallTimeout: 5 * time.Second})
	_, _, err := c.ChatStream(context.Background(), hiRequest())
	if !errors.Is(err, ErrNoResponseHeaders) || !strings.Contains(err.Error(), "no response headers within 50ms") {
		t.Fatalf("headers: err = %v", err)
	}

	// The whole call outruns its cap while frames keep arriving.
	lines := make([]string, 0, 200)
	for i := 0; i < 200; i++ {
		lines = append(lines, `data: {"choices":[{"delta":{"content":"."}}]}`)
	}
	busy := flushingServer(t, lines, 5*time.Millisecond)
	c = New(Config{BaseURL: busy.URL, HeaderTimeout: time.Second, IdleTimeout: time.Second, CallTimeout: 80 * time.Millisecond})
	_, err = drain(t, c, hiRequest())
	if !errors.Is(err, ErrCallCap) || !strings.Contains(err.Error(), "exceeded the 80ms call cap") {
		t.Fatalf("cap: err = %v", err)
	}
}

func TestStreamResult_TimingMetrics(t *testing.T) {
	lines := []string{
		`: keepalive`,
		`data: {"choices":[{"delta":{"content":"a"}}]}`,
		`data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}
	srv := pausingServer(t, lines, 1, 120*time.Millisecond)
	// Not streamClient: that pins the clock for pricing, and the timing figures
	// share it.
	c := New(Config{BaseURL: srv.URL, IdleTimeout: 5 * time.Second})
	res, err := drain(t, c, hiRequest())
	if err != nil {
		t.Fatal(err)
	}
	if res.Timing.FirstData <= 0 || res.Timing.FirstData > 100*time.Millisecond {
		t.Errorf("Timing.FirstData = %v, want a small positive figure", res.Timing.FirstData)
	}
	if res.MaxDataGap < 120*time.Millisecond {
		t.Errorf("MaxDataGap = %v, want at least the 120ms pause", res.MaxDataGap)
	}
	if res.MaxCommentGap < 120*time.Millisecond {
		t.Errorf("MaxCommentGap = %v, want at least the 120ms pause", res.MaxCommentGap)
	}
	if want := int64(len(strings.Join(lines, ""))); res.Bytes != want {
		t.Errorf("Bytes = %d, want %d", res.Bytes, want)
	}
}
