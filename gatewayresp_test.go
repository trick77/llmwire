package llmwire

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func gatewayClient(t *testing.T, srv *httptest.Server) *Client {
	t.Helper()
	env := map[string]string{
		"LLMWIRE_LITELLM_BASE_URL": srv.URL,
		"LLMWIRE_LITELLM_API_KEY":  "gw-key",
		GatewayModelsEnv:           "gpt-5.4-mini=ai-gateway-gpt-5.4-mini",
	}
	c, err := FromEnv("gpt-5.4-mini", Config{Lookup: func(k string) (string, bool) { v, ok := env[k]; return v, ok }})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

const gatewayChatBody = `{"model":"ai-gateway-gpt-5.4-mini","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`

func TestChat_GatewayHeadersRideOnTheResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-litellm-call-id", "call-42")
		w.Header().Set("x-litellm-model-name", "azure/gpt-5.4-mini")
		w.Header().Set("x-litellm-key-spend", "12.5")
		w.Header().Set("x-litellm-response-cost", "1.23e-05")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(gatewayChatBody))
	}))
	defer srv.Close()

	resp, warnings, err := gatewayClient(t, srv).Chat(context.Background(),
		ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Errorf("warnings = %v, want none", warnings)
	}
	g := resp.Gateway
	if !g.Reported() || g.CallID != "call-42" || g.ModelName != "azure/gpt-5.4-mini" || g.ResponseCost != "1.23e-05" {
		t.Errorf("gateway = %+v", g)
	}
	if g.KeySpendNanoUSD == nil || *g.KeySpendNanoUSD != 12_500_000_000 {
		t.Errorf("key spend = %v, want 12.5 USD in nano", g.KeySpendNanoUSD)
	}
	if resp.Usage.Cost.Provenance != Reported || resp.Usage.Cost.NanoUSD != 12300 {
		t.Errorf("cost = %+v, want 12300 nano reported", resp.Usage.Cost)
	}
}

func TestChat_DirectRouteHasNoGatewayBlock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(gatewayChatBody))
	}))
	defer srv.Close()
	resp, _, err := New(Config{BaseURL: srv.URL}).Chat(context.Background(),
		ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Gateway.Reported() {
		t.Errorf("gateway = %+v, want empty", resp.Gateway)
	}
}

func TestChat_KeySpendVariants(t *testing.T) {
	cases := []struct {
		name, value string
		want        *int64
		warn        bool
	}{
		{"absent", "", nil, false},
		{"none literal", "None", nil, false},
		{"null literal", "null", nil, false},
		{"zero is a fresh key", "0", ptr(int64(0)), false},
		{"scientific", "9.25e-05", ptr(int64(92500)), false},
		{"garbage", "lots", nil, true},
		{"negative", "-1", nil, true},
		{"nan", "NaN", nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.value != "" {
					w.Header().Set("x-litellm-key-spend", tc.value)
				}
				w.Header().Set("x-litellm-response-cost", "0.001")
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(gatewayChatBody))
			}))
			defer srv.Close()
			resp, warnings, err := gatewayClient(t, srv).Chat(context.Background(),
				ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
			if err != nil {
				t.Fatal(err)
			}
			got := resp.Gateway.KeySpendNanoUSD
			switch {
			case tc.want == nil && got != nil:
				t.Errorf("key spend = %d, want nil", *got)
			case tc.want != nil && (got == nil || *got != *tc.want):
				t.Errorf("key spend = %v, want %d", got, *tc.want)
			}
			var seen bool
			for _, w := range warnings {
				if w.Feature == "key_spend" {
					seen = true
				}
			}
			if seen != tc.warn {
				t.Errorf("key_spend warning = %v, want %v (warnings %v)", seen, tc.warn, warnings)
			}
			// The cost lane is untouched by the key spend header.
			if resp.Usage.Cost.Provenance != Reported {
				t.Errorf("cost provenance = %v, want Reported", resp.Usage.Cost.Provenance)
			}
		})
	}
}

func TestChat_NoneCostLiteralIsUnpricedNotCorrupt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-litellm-call-id", "call-7")
		w.Header().Set("x-litellm-response-cost", "None")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(gatewayChatBody))
	}))
	defer srv.Close()
	resp, warnings, err := gatewayClient(t, srv).Chat(context.Background(),
		ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Cost.Provenance != Unpriced || resp.Usage.Cost.NanoUSD != 0 {
		t.Errorf("cost = %+v, want unpriced zero", resp.Usage.Cost)
	}
	if len(warnings) != 1 || warnings[0].Feature != "cost" || !strings.Contains(warnings[0].Details, "no cost") {
		t.Errorf("warnings = %v, want the plain no-cost warning, not an unusable-cost one", warnings)
	}
	if resp.Gateway.ResponseCost != "None" || resp.Gateway.CallID != "call-7" {
		t.Errorf("gateway = %+v, want the raw None text and the call id kept", resp.Gateway)
	}
}

func TestStream_GatewayHeadersRideOnTheResult(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-litellm-call-id", "call-stream")
		w.Header().Set("x-litellm-model-name", "azure/gpt-5.4-mini")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()
	st, _, err := gatewayClient(t, srv).ChatStream(context.Background(),
		ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := st.Collect(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Gateway.CallID != "call-stream" || res.Gateway.ModelName != "azure/gpt-5.4-mini" {
		t.Errorf("gateway = %+v", res.Gateway)
	}
	if res.Usage.Cost.Provenance != Unpriced {
		t.Errorf("a stream is never priced from headers, got %v", res.Usage.Cost.Provenance)
	}
}

func ptr[T any](v T) *T { return &v }
