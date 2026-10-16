package llmwire

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
)

// The end-to-end shape of a routed call: the request leaves for the gateway's
// host with the gateway's key, and its body names the alias, not the profile.
func TestFromEnv_gatewayModelsOnTheWire(t *testing.T) {
	srv := jsonServer(t, http.StatusOK, `{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)

	c, err := FromEnv("gpt-5.4-mini", Config{Lookup: mapLookup(gatewayEnv(srv.URL, "gpt-5.4-mini=ai-gateway-gpt-5.4-mini"))})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Chat(context.Background(), ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}}); err != nil {
		t.Fatal(err)
	}
	var req struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal(srv.lastBody(), &req)
	if gotAuth := srv.last().Header.Get("Authorization"); req.Model != "ai-gateway-gpt-5.4-mini" || gotAuth != "Bearer gw-key" {
		t.Fatalf("wire model %q, auth %q", req.Model, gotAuth)
	}
}
