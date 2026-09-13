package llmwire

import (
	"context"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"
)

// The id shape the upstream issues, and the one the emulation must reproduce:
// "ses_", 12 lowercase hex, 14 base62.
var sessionIDShape = regexp.MustCompile(`^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`)

// The client string is four name/version tokens in the order the SDK composes
// them. A future bump that drops or reorders one would still pass every test
// that compares against the constant, so the shape is pinned here.
func TestOpenCodeUserAgentShape(t *testing.T) {
	want := []string{"opencode", "ai-sdk/openai-compatible", "ai-sdk/provider-utils", "runtime/bun"}
	tokens := strings.Split(OpenCodeUserAgent, " ")
	if len(tokens) != len(want) {
		t.Fatalf("OpenCodeUserAgent = %q, want %d tokens", OpenCodeUserAgent, len(want))
	}
	version := regexp.MustCompile(`^\d+\.\d+\.\d+$`)
	for i, tok := range tokens {
		// The name may itself contain a slash; the version is after the last one.
		idx := strings.LastIndex(tok, "/")
		if idx < 0 || tok[:idx] != want[i] || !version.MatchString(tok[idx+1:]) {
			t.Errorf("token %d = %q, want %s/<semver>", i, tok, want[i])
		}
	}
}

func TestNewSessionID_hasTheUpstreamShapeAndIsUnique(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 1000; i++ {
		id := newSessionID()
		if !sessionIDShape.MatchString(id) {
			t.Fatalf("id %q does not match %s", id, sessionIDShape)
		}
		if seen[id] {
			t.Fatalf("duplicate id %q within one millisecond; the counter is not doing its job", id)
		}
		seen[id] = true
	}
}

// Off by default, and the default identity says llmwire.
func TestEmulateOpenCode_offSendsNothingOfIt(t *testing.T) {
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop","message":{"content":"x"}}]}`))
	}))
	defer srv.Close()

	_, _, err := New(Config{BaseURL: srv.URL}).Chat(context.Background(), ChatRequest{
		Model: "glm-5.3-flash", Messages: []Message{User("hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if ua := got.Get("User-Agent"); !strings.HasPrefix(ua, "llmwire/") {
		t.Errorf("User-Agent = %q, want the library's own", ua)
	}
	if got.Get(HeaderSessionID) != "" || got.Get(HeaderSessionAffinity) != "" {
		t.Errorf("session headers sent without the flag: %v", got)
	}
}

// On: the client string, and the pair carrying one id — the same id across
// calls, because one Client is one session.
func TestEmulateOpenCode_sendsTheIdentityAndKeepsOneSession(t *testing.T) {
	var seen []http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Clone())
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop","message":{"content":"x"}}]}`))
	}))
	defer srv.Close()

	c := New(Config{BaseURL: srv.URL, EmulateOpenCode: true})
	for i := 0; i < 3; i++ {
		if _, _, err := c.Chat(context.Background(), ChatRequest{
			Model: "glm-5.3-flash", Messages: []Message{User("hi")},
		}); err != nil {
			t.Fatal(err)
		}
	}

	first := seen[0]
	if first.Get("User-Agent") != OpenCodeUserAgent {
		t.Errorf("User-Agent = %q, want %q", first.Get("User-Agent"), OpenCodeUserAgent)
	}
	id := first.Get(HeaderSessionID)
	if !sessionIDShape.MatchString(id) {
		t.Fatalf("%s = %q, not a session id", HeaderSessionID, id)
	}
	if first.Get(HeaderSessionAffinity) != id {
		t.Errorf("affinity %q != session %q; the pair must carry one value", first.Get(HeaderSessionAffinity), id)
	}
	for i, h := range seen[1:] {
		if h.Get(HeaderSessionID) != id {
			t.Errorf("call %d changed session to %q without an idle gap", i+1, h.Get(HeaderSessionID))
		}
	}
}

// The emulation overrides a UserAgent however it was supplied — the string IS the
// flag, and a User-Agent smuggled through the generic Headers map would
// otherwise leave the flag on and the emulation off. A session header the
// caller set by hand survives, and is mirrored into its twin: opencode never
// sends the pair apart, so a hand-pinned id must not produce the one shape the
// upstream never sees.
func TestEmulateOpenCode_precedence(t *testing.T) {
	capture := func(t *testing.T, cfg Config) http.Header {
		t.Helper()
		var got http.Header
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			got = r.Header.Clone()
			_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop","message":{"content":"x"}}]}`))
		}))
		defer srv.Close()
		cfg.BaseURL, cfg.EmulateOpenCode = srv.URL, true
		if _, _, err := New(cfg).Chat(context.Background(), ChatRequest{
			Model: "glm-5.3-flash", Messages: []Message{User("hi")},
		}); err != nil {
			t.Fatal(err)
		}
		return got
	}

	t.Run("flag beats the UserAgent field", func(t *testing.T) {
		got := capture(t, Config{UserAgent: "something-else/1.0"})
		if got.Get("User-Agent") != OpenCodeUserAgent {
			t.Errorf("User-Agent = %q", got.Get("User-Agent"))
		}
	})

	t.Run("flag beats a User-Agent in Headers too", func(t *testing.T) {
		got := capture(t, Config{Headers: map[string]string{"User-Agent": "something-else/1.0"}})
		if got.Get("User-Agent") != OpenCodeUserAgent {
			t.Errorf("User-Agent = %q; the generic map must not switch the emulation off", got.Get("User-Agent"))
		}
	})

	t.Run("a hand-pinned id survives and is mirrored", func(t *testing.T) {
		got := capture(t, Config{Headers: map[string]string{HeaderSessionID: "ses_pinned_by_hand"}})
		if got.Get(HeaderSessionID) != "ses_pinned_by_hand" {
			t.Errorf("%s = %q; a hand-set header must survive", HeaderSessionID, got.Get(HeaderSessionID))
		}
		if got.Get(HeaderSessionAffinity) != "ses_pinned_by_hand" {
			t.Errorf("%s = %q; the pair must carry one value", HeaderSessionAffinity, got.Get(HeaderSessionAffinity))
		}
	})

	t.Run("pinning the affinity alone mirrors the other way", func(t *testing.T) {
		got := capture(t, Config{Headers: map[string]string{HeaderSessionAffinity: "ses_pinned_by_hand"}})
		if got.Get(HeaderSessionID) != "ses_pinned_by_hand" || got.Get(HeaderSessionAffinity) != "ses_pinned_by_hand" {
			t.Errorf("pair = %q / %q, want both pinned", got.Get(HeaderSessionID), got.Get(HeaderSessionAffinity))
		}
	})
}

// A session ends when its user steps away. Driven by the injected clock, so the
// half-hour is asserted rather than waited for.
func TestEmulateOpenCode_rotatesAfterAnIdleGap(t *testing.T) {
	now := time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC)
	s := newSession(func() time.Time { return now })

	first, rotated := s.current()
	if rotated {
		t.Fatal("the first call reported a rotation; the id was minted at construction")
	}
	now = now.Add(sessionIdleRotation - time.Second)
	if id, rotated := s.current(); id != first || rotated {
		t.Fatal("rotated inside the idle window: a burst of related calls must stay one session")
	}
	// The window is measured from the LAST call, not the first: activity keeps
	// a session alive indefinitely, the way it does for a person who keeps
	// working.
	now = now.Add(sessionIdleRotation - time.Second)
	if id, rotated := s.current(); id != first || rotated {
		t.Fatal("rotated while still active; the gap is measured from the last call")
	}
	now = now.Add(sessionIdleRotation + time.Second)
	second, rotated := s.current()
	if second == first || !rotated {
		t.Fatal("did not rotate, or did not say so, after an idle gap longer than the window")
	}
	if !sessionIDShape.MatchString(second) {
		t.Errorf("rotated id %q has the wrong shape", second)
	}
	if id, rotated := s.current(); id != second || rotated {
		t.Error("a fresh session did not persist across the next call")
	}
}
