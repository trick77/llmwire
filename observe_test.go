package llmwire

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// capture builds a client on srv whose log lines land in the returned buffer,
// at Debug so the success line is visible.
func capture(t *testing.T, srv *httptest.Server, idle time.Duration) (*Client, *bytes.Buffer) {
	t.Helper()
	var buf bytes.Buffer
	c := New(Config{
		BaseURL:     srv.URL,
		Registry:    Default(),
		IdleTimeout: idle,
		CallTimeout: 20 * time.Second,
		Logger:      slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})),
	})
	return c, &buf
}

func wantAll(t *testing.T, got string, wants ...string) {
	t.Helper()
	for _, w := range wants {
		if !strings.Contains(got, w) {
			t.Errorf("missing %q in:\n%s", w, got)
		}
	}
}

// The success line carries the figures and never the text.
func TestChat_logsTheCallWithoutItsText(t *testing.T) {
	srv, _ := jsonServer(t, 200, goodCompletion)
	c, buf := capture(t, srv, time.Second)
	if _, _, err := c.Chat(context.Background(), ChatRequest{
		Model: "glm-5.3-flash", Messages: []Message{User("the secret prompt")}, Reasoning: ReasoningEffort("low"),
	}); err != nil {
		t.Fatal(err)
	}
	got := buf.String()
	if strings.Count(got, "\n") != 1 {
		t.Fatalf("want one line, got:\n%s", got)
	}
	wantAll(t, got, `level=DEBUG msg="llmwire call"`, "kind=chat", "model=glm-5.3-flash",
		"messages=1", "reasoning=low", "finish_reason=stop", "content_chars=5", "reasoning_chars=7",
		"usage_reported=true", "input_tokens=258", "cache_read_tokens=192", "output_tokens=100",
		"reasoning_tokens=40", "cost_usd=0.000066", "cost_provenance=from-table", "total_ms=")
	for _, leak := range []string{"secret", "hello", "thought"} {
		if strings.Contains(got, leak) {
			t.Errorf("%q reached the log: %s", leak, got)
		}
	}
}

// An answer cut by the output cap is a Warn with a name; a tool-call-only
// turn, which is also content-less, is not.
func TestChat_anomaliesAreNamedWarnings(t *testing.T) {
	cases := []struct {
		name, body, want string
	}{
		{"length", `{"choices":[{"finish_reason":"length","message":{"content":"partial"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
			`level=WARN msg="llmwire answer truncated by the output cap"`},
		{"empty", `{"choices":[{"finish_reason":"stop","message":{"content":""}}],
			"usage":{"prompt_tokens":1,"completion_tokens":0}}`,
			`level=WARN msg="llmwire answer empty"`},
		{"tool call only", `{"choices":[{"finish_reason":"tool_calls","message":{"content":"",
			"tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{}"}}]}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
			`level=DEBUG msg="llmwire call"`},
		{"no usage", `{"choices":[{"finish_reason":"stop","message":{"content":"x"}}]}`,
			`level=WARN msg="llmwire call unaccounted, no usage reported"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := jsonServer(t, 200, tc.body)
			c, buf := capture(t, srv, time.Second)
			if _, _, err := c.Chat(context.Background(), hiRequest()); err != nil {
				t.Fatal(err)
			}
			wantAll(t, buf.String(), tc.want)
		})
	}
}

// A failure is a Warn naming the bound, with the timing it fired at.
func TestChatStream_logsTheBoundThatFired(t *testing.T) {
	srv := flushingServer(t, []string{
		`data: {"choices":[{"delta":{"content":"hel"}}]}`,
		`data: {"choices":[{"delta":{"content":"lo"}}]}`,
	}, 2*time.Second)
	c, buf := capture(t, srv, 50*time.Millisecond)
	s, _, err := c.ChatStream(context.Background(), hiRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.Collect(nil); !errors.Is(err, ErrStreamIdle) {
		t.Fatalf("err = %v, want the idle bound", err)
	}
	wantAll(t, buf.String(), `level=WARN msg="llmwire call failed"`, "kind=chat_stream",
		"error=\"llmwire: stream idle for 50ms\"", "total_ms=", "content_chars=3")
}

// A non-2xx is a Warn with the status; a 429 adds the retry hint.
func TestChat_logsTheStatus(t *testing.T) {
	srv, _ := jsonServer(t, 429, `{"error":{"message":"slow down"}}`)
	c, buf := capture(t, srv, time.Second)
	if _, _, err := c.Chat(context.Background(), hiRequest()); err == nil {
		t.Fatal("want an error")
	}
	wantAll(t, buf.String(), `level=WARN msg="llmwire call failed"`, "status=429", "retry_after_ms=")
}

// Stats sum every finished call, failures counted apart, and the copy handed
// out is the caller's.
func TestStats_sumPerModel(t *testing.T) {
	srv, _ := jsonServer(t, 200, goodCompletion)
	c, _ := capture(t, srv, time.Second)
	for range 2 {
		if _, _, err := c.Chat(context.Background(), hiRequest()); err != nil {
			t.Fatal(err)
		}
	}
	bad, _ := jsonServer(t, 500, `{"error":{"message":"down"}}`)
	c.baseURL = bad.URL
	if _, _, err := c.Chat(context.Background(), hiRequest()); err == nil {
		t.Fatal("want an error")
	}

	st := c.Stats()
	m := st.Models["glm-5.3-flash"]
	if m.Calls != 3 || m.Errors != 1 || m.Closed != 0 {
		t.Errorf("calls/errors/closed = %d/%d/%d, want 3/1/0", m.Calls, m.Errors, m.Closed)
	}
	if m.InputTokens != 516 || m.CacheReadTokens != 384 || m.OutputTokens != 200 || m.ReasoningTokens != 80 {
		t.Errorf("tokens = %+v", m)
	}
	if m.CostNanoUSD != 2*65_660 || m.UnpricedCalls != 1 || m.UnreportedCalls != 1 {
		t.Errorf("cost = %d unpriced = %d unreported = %d, want 131320/1/1", m.CostNanoUSD, m.UnpricedCalls, m.UnreportedCalls)
	}
	st.Models["glm-5.3-flash"] = ModelStats{}
	if c.Stats().Models["glm-5.3-flash"].Calls != 3 {
		t.Error("Stats handed out its own map")
	}
	var buf bytes.Buffer
	slog.New(slog.NewTextHandler(&buf, nil)).Info("summary", "llmwire", c.Stats())
	wantAll(t, buf.String(), "llmwire.glm-5.3-flash.calls=3", "llmwire.glm-5.3-flash.cost_usd=0.000131")
}
