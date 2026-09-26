package llmwire

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// One client, several models on several hosts: each request goes to its own
// model's host with its own provider's key, so a lane swap is a config change.

const multiDoc = `providers:
  a: {emulate_opencode: true}
  b: {}
profiles:
  - id: ma1
    display_name: MA1
    wire_model_id: ma1
    provider: a
    max_tokens_param: max_tokens
    streaming: {supported: true, accepts_stream_options: true}
  - id: ma2
    display_name: MA2
    wire_model_id: ma2
    provider: a
    max_tokens_param: max_tokens
  - id: mb
    display_name: MB
    wire_model_id: mb
    provider: b
    max_tokens_param: max_tokens
  - id: other
    display_name: Other
    wire_model_id: other
    provider: b
    max_tokens_param: max_tokens
`

type seenCall struct {
	path, auth, session string
}

// recordingServer answers every chat request with goodCompletion (or an SSE
// equivalent when asked to stream) and records who was asked.
func recordingServer(t *testing.T) (*httptest.Server, func() []seenCall) {
	t.Helper()
	var mu sync.Mutex
	var calls []seenCall
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := readAllBody(r)
		mu.Lock()
		calls = append(calls, seenCall{path: r.URL.Path, auth: r.Header.Get("Authorization"), session: r.Header.Get(HeaderSessionID)})
		mu.Unlock()
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(goodCompletion))
	}))
	t.Cleanup(srv.Close)
	return srv, func() []seenCall { mu.Lock(); defer mu.Unlock(); return append([]seenCall(nil), calls...) }
}

func mapLookup(m map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := m[k]; return v, ok }
}

func TestFromEnvModels_RoutesEachModelToItsProvider(t *testing.T) {
	srvA, seenA := recordingServer(t)
	srvB, seenB := recordingServer(t)
	reg := registryFrom(t, multiDoc)
	c, err := FromEnvModels(Config{Registry: reg, Lookup: mapLookup(map[string]string{
		"LLMWIRE_A_BASE_URL": srvA.URL, "LLMWIRE_A_API_KEY": "ka",
		"LLMWIRE_B_BASE_URL": srvB.URL, "LLMWIRE_B_API_KEY": "kb",
	})}, "ma1", "ma2", "mb")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	for _, m := range []string{"ma1", "mb", "ma2"} {
		if _, _, err := c.Chat(ctx, ChatRequest{Model: m, Messages: []Message{User("hi")}}); err != nil {
			t.Fatalf("%s: %v", m, err)
		}
	}
	s, _, err := c.ChatStream(ctx, ChatRequest{Model: "ma1", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Collect(nil); err != nil {
		t.Fatal(err)
	}
	_ = s.Close()

	a, b := seenA(), seenB()
	if len(a) != 3 || len(b) != 1 {
		t.Fatalf("host a saw %d calls, b saw %d; want 3 and 1", len(a), len(b))
	}
	for _, call := range a {
		if call.auth != "Bearer ka" {
			t.Errorf("host a got %q", call.auth)
		}
		// One opencode identity per provider: every call to a shares it.
		if call.session == "" || call.session != a[0].session {
			t.Errorf("host a session %q, want one id for the provider (%q)", call.session, a[0].session)
		}
	}
	if b[0].auth != "Bearer kb" || b[0].session != "" {
		t.Errorf("host b got %+v; its provider does not emulate opencode", b[0])
	}

	// One client, one set of counters, whichever host ran the call.
	st := c.Stats()
	if st.Models["ma1"].Calls != 2 || st.Models["mb"].Calls != 1 {
		t.Errorf("stats = %+v", st.Models)
	}
}

func TestFromEnvModels_RefusesAModelItWasNotGiven(t *testing.T) {
	srvA, _ := recordingServer(t)
	srvB, seenB := recordingServer(t)
	reg := registryFrom(t, multiDoc)
	c, err := FromEnvModels(Config{Registry: reg, Lookup: mapLookup(map[string]string{
		"LLMWIRE_A_BASE_URL": srvA.URL, "LLMWIRE_A_API_KEY": "ka",
		"LLMWIRE_B_BASE_URL": srvB.URL, "LLMWIRE_B_API_KEY": "kb",
	})}, "ma1", "mb")
	if err != nil {
		t.Fatal(err)
	}
	var nc *ModelNotConfiguredError
	// "other" shares mb's host and key, and is still refused: the list is the
	// contract, not the host.
	_, _, err = c.Chat(context.Background(), ChatRequest{Model: "other", Messages: []Message{User("hi")}})
	if !errors.As(err, &nc) || nc.Model != "other" || !strings.Contains(err.Error(), "ma1, mb") {
		t.Fatalf("err = %v, want a ModelNotConfiguredError naming the configured set", err)
	}
	if _, err := c.Validate(ChatRequest{Model: "other", Messages: []Message{User("hi")}}); !errors.As(err, &nc) {
		t.Errorf("Validate: err = %v", err)
	}
	if _, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "other", Messages: []Message{User("hi")}}); !errors.As(err, &nc) {
		t.Errorf("ChatStream: err = %v", err)
	}
	if _, _, err := c.Embed(context.Background(), EmbedRequest{Model: "other", Inputs: []string{"x"}}); !errors.As(err, &nc) {
		t.Errorf("Embed: err = %v", err)
	}
	if len(seenB()) != 0 {
		t.Error("a refused model reached the wire")
	}

	// The per-model client for listing and raw calls is scoped the same way:
	// it cannot carry b's key to a's models.
	sub, err := c.ForModel("mb")
	if err != nil {
		t.Fatal(err)
	}
	if sub.baseURL != srvB.URL || sub.apiKey != "kb" {
		t.Errorf("ForModel(mb) = %q %q", sub.baseURL, sub.apiKey)
	}
	if _, _, err := sub.Chat(context.Background(), ChatRequest{Model: "ma1", Messages: []Message{User("hi")}}); !errors.As(err, &nc) {
		t.Errorf("sub-client serving another provider's model: err = %v", err)
	}
	if _, err := c.ForModel("other"); !errors.As(err, &nc) {
		t.Errorf("ForModel(other): err = %v", err)
	}
	// Host-level calls need a host: the multi-model client has several.
	if _, _, err := c.ListModels(context.Background()); err == nil || !strings.Contains(err.Error(), "ForModel") {
		t.Errorf("ListModels on the multi-model client: err = %v", err)
	}
	if _, err := c.RawStream(context.Background(), []byte(`{}`), nil); err == nil || !strings.Contains(err.Error(), "ForModel") {
		t.Errorf("RawStream on the multi-model client: err = %v", err)
	}
	if _, _, err := c.RawPost(context.Background(), routeChat, []byte(`{}`)); err == nil || !strings.Contains(err.Error(), "ForModel") {
		t.Errorf("RawPost on the multi-model client: err = %v", err)
	}
}

func TestFromEnvModels_NamesEveryMissingVariable(t *testing.T) {
	reg := registryFrom(t, multiDoc)
	_, err := FromEnvModels(Config{Registry: reg, Lookup: mapLookup(map[string]string{
		"LLMWIRE_A_BASE_URL": "https://a.example", "LLMWIRE_B_BASE_URL": "https://b.example",
	})}, "ma1", "ma2", "mb")
	if err == nil {
		t.Fatal("no error with both keys missing")
	}
	for _, v := range []string{"LLMWIRE_A_API_KEY", "LLMWIRE_B_API_KEY"} {
		if !strings.Contains(err.Error(), v) {
			t.Errorf("error does not name %s: %v", v, err)
		}
	}
	// Named once, although two models need it.
	if strings.Count(err.Error(), "LLMWIRE_A_API_KEY") != 1 {
		t.Errorf("LLMWIRE_A_API_KEY named more than once: %v", err)
	}
	var me *MissingEnvError
	if !errors.As(err, &me) {
		t.Errorf("err = %v, want it to unwrap to *MissingEnvError", err)
	}

	var ue *UnknownModelError
	if _, err := FromEnvModels(Config{Registry: reg}, "ma1", "nope"); !errors.As(err, &ue) {
		t.Errorf("unknown id: err = %v", err)
	}
	if _, err := FromEnvModels(Config{Registry: reg}); err == nil {
		t.Error("no models: want an error")
	}
}

func TestFromEnvModels_ExplicitEndpoint(t *testing.T) {
	srv, seen := recordingServer(t)
	reg := registryFrom(t, multiDoc)
	// Wiring the endpoint by hand, as with FromEnv: no variables, one host.
	c, err := FromEnvModels(Config{Registry: reg, BaseURL: srv.URL, APIKey: "k", Lookup: mapLookup(nil)}, "ma1", "mb")
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"ma1", "mb"} {
		if _, _, err := c.Chat(context.Background(), ChatRequest{Model: m, Messages: []Message{User("hi")}}); err != nil {
			t.Fatal(err)
		}
	}
	if got := seen(); len(got) != 2 || got[0].auth != "Bearer k" {
		t.Errorf("calls = %+v", got)
	}

	// One explicit key across two providers would hand one vendor's key to
	// the other's host.
	_, err = FromEnvModels(Config{Registry: reg, APIKey: "k", Lookup: mapLookup(map[string]string{
		"LLMWIRE_A_BASE_URL": "https://a.example", "LLMWIRE_B_BASE_URL": "https://b.example",
	})}, "ma1", "mb")
	if err == nil || !strings.Contains(err.Error(), "APIKey") {
		t.Errorf("err = %v, want the shared key refused", err)
	}
}

// The single-model constructor is untouched: a FromEnv client still serves any
// model on its host.
func TestFromEnv_StillServesAnyModel(t *testing.T) {
	srv, _ := recordingServer(t)
	reg := registryFrom(t, multiDoc)
	c, err := FromEnv("mb", Config{Registry: reg, BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Chat(context.Background(), ChatRequest{Model: "other", Messages: []Message{User("hi")}}); err != nil {
		t.Fatalf("FromEnv client refused a sibling model: %v", err)
	}
}
