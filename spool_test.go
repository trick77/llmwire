package llmwire

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func spoolFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

func TestSpoolTransport_WritesStatusHeadersAndBody(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Request-Id", "req-1")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)

	dir := filepath.Join(t.TempDir(), "spool")
	hc := &http.Client{Transport: NewSpoolTransport(dir, nil, nil)}
	c := New(Config{BaseURL: srv.URL, HTTPClient: hc})
	stream, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	for stream.Next() {
	}
	//nolint:govet // shadow: statement-scoped err, distinct from the outer one
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}

	files := spoolFiles(t, dir)
	if len(files) != 1 || !strings.HasSuffix(files[0], "-000001.http") {
		t.Fatalf("files = %v, want one numbered .http file", files)
	}
	raw, err := os.ReadFile(filepath.Join(dir, files[0]))
	if err != nil {
		t.Fatal(err)
	}
	got := string(raw)
	for _, want := range []string{"HTTP/1.1 200 OK\n", "X-Request-Id: req-1\r\n", "Content-Type: text/event-stream\r\n", "\r\n\n" + body} {
		if !strings.Contains(got, want) {
			t.Errorf("spool file lacks %q:\n%s", want, got)
		}
	}
	info, err := os.Stat(filepath.Join(dir, files[0]))
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("file mode = %o, want 0600", perm)
	}
}

func TestSpoolTransport_SkipLeavesNothingBehind(t *testing.T) {
	srv, _ := jsonServer(t, 200, `{"model":"glm-5.3-flash","choices":[{"finish_reason":"stop","message":{"content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	dir := filepath.Join(t.TempDir(), "spool")
	skipped := 0
	hc := &http.Client{Transport: NewSpoolTransport(dir, nil, func(r *http.Request) bool {
		skipped++
		return r.Header.Get("X-Ephemeral") == "1"
	})}
	c := New(Config{BaseURL: srv.URL, HTTPClient: hc, Headers: map[string]string{"X-Ephemeral": "1"}})
	if _, _, err := c.Chat(context.Background(), ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}}); err != nil {
		t.Fatal(err)
	}
	if skipped != 1 {
		t.Fatalf("skip asked %d times, want once", skipped)
	}
	if files := spoolFiles(t, dir); len(files) != 0 {
		t.Fatalf("files = %v, want none", files)
	}
}

func TestSpoolTransport_SequenceCountsPerTransport(t *testing.T) {
	srv, _ := jsonServer(t, 200, `{"model":"glm-5.3-flash","choices":[{"finish_reason":"stop","message":{"content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	dir := filepath.Join(t.TempDir(), "spool")
	c := New(Config{BaseURL: srv.URL, HTTPClient: &http.Client{Transport: NewSpoolTransport(dir, nil, nil)}})
	for i := 0; i < 2; i++ {
		if _, _, err := c.Chat(context.Background(), ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}}); err != nil {
			t.Fatal(err)
		}
	}
	files := spoolFiles(t, dir)
	if len(files) != 2 || !strings.HasSuffix(files[0], "-000001.http") || !strings.HasSuffix(files[1], "-000002.http") {
		t.Fatalf("files = %v", files)
	}
}

// A transport error passes through untouched, and nothing is written for it.
func TestSpoolTransport_TransportErrorPassesThrough(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "spool")
	c := New(Config{BaseURL: "http://127.0.0.1:1", HTTPClient: &http.Client{Transport: NewSpoolTransport(dir, nil, nil)}})
	if _, _, err := c.Chat(context.Background(), ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}}); err == nil {
		t.Fatal("expected a dial error")
	}
	if files := spoolFiles(t, dir); len(files) != 0 {
		t.Fatalf("files = %v, want none", files)
	}
}
