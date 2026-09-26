// Package llmwiretest lets a module that uses llmwire test what it ASKS for
// without pinning what any real model is called, how its knobs are spelled on
// the wire, or what it costs.
//
// It has two parts: a registry holding synthetic profiles (ChatModel,
// EmbedModel), and Server, an httptest fake of the OpenAI-compatible routes
// that records every request. A test builds its client on both and asserts on
// intent:
//
//	srv := llmwiretest.NewServer(t)
//	c := srv.Client()
//	resp, _, err := c.Chat(ctx, llmwire.ChatRequest{
//		Model:           llmwiretest.ChatModel,
//		Messages:        []llmwire.Message{llmwire.User("title this")},
//		Reasoning:       llmwire.ReasoningMinimal(),
//		MaxAnswerTokens: &n,
//	})
//	// resp.ReasoningSent and srv.Last().Reasoning() are llmwiretest.MinimalSent;
//	// srv.Last().MaxTokens() is n + llmwiretest.MinimalOverhead.
//
// Code that builds its client with llmwire.FromEnv or FromEnvModels passes
// Config{Registry: llmwiretest.Registry(), Lookup: srv.Lookup} instead.
package llmwiretest

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/trick77/llmwire"
)

// The synthetic models and the facts a test asserts against. Constants rather
// than a profile lookup so an assertion reads as the intent it checks.
const (
	// ChatModel reasons by effort level, can be switched off, and takes tools,
	// images, both JSON formats and streaming.
	ChatModel = "llmwiretest-chat"
	// BudgetModel takes reasoning as a token budget and can be switched off.
	BudgetModel = "llmwiretest-budget"
	// EmbedModel returns EmbedDimensions-long vectors.
	EmbedModel = "llmwiretest-embed"

	// MinimalSent and BalancedSent are ChatModel's ReasoningSent for
	// ReasoningMinimal and ReasoningBalanced; the Overhead constants are what
	// MaxAnswerTokens adds for each.
	MinimalSent      = "off"
	MinimalOverhead  = 0
	BalancedSent     = "medium"
	BalancedOverhead = 256

	// EmbedDimensions is EmbedModel's default vector length.
	EmbedDimensions = 8

	// Reply is the content every chat call answers with until SetReply.
	Reply = "ok"
)

//go:embed profiles.yaml
var profiles []byte

var (
	regOnce sync.Once
	reg     *llmwire.Registry
)

// Registry returns the registry holding the synthetic models. Shared and
// read-only, like llmwire.Default.
func Registry() *llmwire.Registry {
	regOnce.Do(func() {
		r, err := llmwire.NewRegistry(profiles)
		if err != nil {
			panic("llmwiretest: embedded profiles.yaml is invalid: " + err.Error())
		}
		reg = r
	})
	return reg
}

// Profiles returns the synthetic profile document, for a test that builds a
// registry of its own around it.
func Profiles() []byte { return append([]byte(nil), profiles...) }

// Request is one call the fake received.
type Request struct {
	Method, Path string
	Header       http.Header
	// Body is the decoded JSON body; nil for a GET.
	Body map[string]any
}

// Model is the body's "model".
func (r Request) Model() string { s, _ := r.Body["model"].(string); return s }

// Stream reports whether the call asked for SSE.
func (r Request) Stream() bool { b, _ := r.Body["stream"].(bool); return b }

// MaxTokens is the output cap the call carried, in whichever spelling.
func (r Request) MaxTokens() (int, bool) {
	for _, k := range []string{"max_tokens", "max_completion_tokens"} {
		if v, ok := r.Body[k].(float64); ok {
			return int(v), true
		}
	}
	return 0, false
}

// Reasoning is the reasoning knob the call carried, in exactly ReasoningSent's
// vocabulary, so srv.Last().Reasoning() == resp.ReasoningSent for every
// variant: "off" (the toggle, effort "none", a zero budget), a level,
// "budget:<n>", or "" when none was sent. The budget's wire key is read off the
// model's profile in Registry(), since vendors disagree on its name.
func (r Request) Reasoning() string {
	if t, ok := r.Body["thinking"].(map[string]any); ok && t["type"] == "disabled" {
		return "off"
	}
	if e, ok := r.Body["reasoning_effort"].(string); ok {
		if e == "none" {
			return "off"
		}
		return e
	}
	if p, err := Registry().Lookup(r.Model()); err == nil && p.Reasoning.BudgetParam != "" {
		if n, ok := r.Body[p.Reasoning.BudgetParam].(float64); ok {
			if n == 0 {
				return "off"
			}
			return fmt.Sprintf("budget:%d", int(n))
		}
	}
	return ""
}

// Server is a fake OpenAI-compatible endpoint: /chat/completions (plain and
// streamed), /embeddings and /models. Every request is recorded.
type Server struct {
	*httptest.Server

	mu       sync.Mutex
	requests []Request
	reply    string
}

// NewServer starts a fake, closed when the test ends.
func NewServer(t testing.TB) *Server {
	t.Helper()
	s := &Server{reply: Reply}
	s.Server = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.Close)
	return s
}

// SetReply sets the content later chat calls answer with.
func (s *Server) SetReply(content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reply = content
}

// Requests returns every request received so far, oldest first.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Request(nil), s.requests...)
}

// Last returns the latest request, or the zero Request if none arrived.
func (s *Server) Last() Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.requests) == 0 {
		return Request{}
	}
	return s.requests[len(s.requests)-1]
}

// Lookup is a Config.Lookup naming this server as the llmwiretest provider's
// host and key, for code that builds its client with FromEnv or
// FromEnvModels.
func (s *Server) Lookup(name string) (string, bool) {
	switch name {
	case "LLMWIRE_LLMWIRETEST_BASE_URL":
		return s.URL, true
	case "LLMWIRE_LLMWIRETEST_API_KEY":
		return "llmwiretest-key", true
	}
	return "", false
}

// Client returns a client on this server and Registry(), logging nowhere.
func (s *Server) Client() *llmwire.Client {
	return llmwire.New(llmwire.Config{ //nolint:gosec // G101: fake key, only ever sent to this fake
		BaseURL:  s.URL,
		APIKey:   "llmwiretest-key",
		Registry: Registry(),
		Logger:   slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	req := Request{Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone()}
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req.Body); err != nil {
			http.Error(w, `{"error":{"message":"body is not JSON"}}`, http.StatusBadRequest)
			return
		}
	}
	s.mu.Lock()
	s.requests = append(s.requests, req)
	reply := s.reply
	s.mu.Unlock()

	switch {
	case strings.HasSuffix(req.Path, "/chat/completions"):
		s.chat(w, req, reply)
	case strings.HasSuffix(req.Path, "/embeddings"):
		embeddings(w, req)
	case strings.HasSuffix(req.Path, "/models"):
		writeJSON(w, map[string]any{"object": "list", "data": []any{
			map[string]any{"id": ChatModel}, map[string]any{"id": EmbedModel},
		}})
	default:
		http.Error(w, `{"error":{"message":"no such route"}}`, http.StatusNotFound)
	}
}

// usage is fixed: a test asserting on cost reads it off the response and the
// synthetic rates, never off a vendor's.
var usage = map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

func (s *Server) chat(w http.ResponseWriter, req Request, reply string) {
	if !req.Stream() {
		writeJSON(w, map[string]any{
			"model": req.Model(),
			"choices": []any{map[string]any{
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": reply},
			}},
			"usage": usage,
		})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	frame := func(v any) {
		raw, _ := json.Marshal(v)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", raw)
	}
	frame(map[string]any{"model": req.Model(), "choices": []any{map[string]any{"delta": map[string]any{"content": reply}}}})
	frame(map[string]any{"model": req.Model(), "choices": []any{map[string]any{"delta": map[string]any{}, "finish_reason": "stop"}}, "usage": usage})
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
}

func embeddings(w http.ResponseWriter, req Request) {
	dims := EmbedDimensions
	if d, ok := req.Body["dimensions"].(float64); ok {
		dims = int(d)
	}
	inputs, _ := req.Body["input"].([]any)
	data := make([]any, len(inputs))
	for i := range inputs {
		vec := make([]float64, dims)
		vec[i%dims] = 1
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": vec}
	}
	writeJSON(w, map[string]any{"object": "list", "model": req.Model(), "data": data,
		"usage": map[string]any{"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
