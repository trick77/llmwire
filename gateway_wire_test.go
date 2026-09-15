package llmwire

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The end-to-end shape of a routed call: the request leaves for the gateway's
// host with the gateway's key, and its body names the alias, not the profile.
func TestFromEnv_gatewayModelsOnTheWire(t *testing.T) {
	var gotAuth, gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		gotModel = req.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	env := map[string]string{
		"LLMWIRE_LITELLM_BASE_URL": srv.URL,
		"LLMWIRE_LITELLM_API_KEY":  "gw-key",
		GatewayModelsEnv:           "gpt-5.4-mini=ai-gateway-gpt-5.4-mini",
	}
	c, err := FromEnv("gpt-5.4-mini", Config{Lookup: func(k string) (string, bool) { v, ok := env[k]; return v, ok }})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Chat(context.Background(), ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}}); err != nil {
		t.Fatal(err)
	}
	if gotModel != "ai-gateway-gpt-5.4-mini" || gotAuth != "Bearer gw-key" {
		t.Fatalf("wire model %q, auth %q", gotModel, gotAuth)
	}
}
