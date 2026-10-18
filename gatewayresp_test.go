package llmwire

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"strings"
	"testing"
)

// gatewayClient routes gpt-5.4-mini through a litellm at url; log is the
// client's logger, nil for none.
func gatewayClient(t *testing.T, url string, log *slog.Logger) *Client {
	t.Helper()
	c, err := FromEnv("gpt-5.4-mini", Config{Lookup: mapLookup(gatewayEnv(url, "gpt-5.4-mini=ai-gateway-gpt-5.4-mini")), Logger: log})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestChat_GatewayCallIDIsLoggedOnARejection(t *testing.T) {
	srv := jsonServer(t, http.StatusTooManyRequests, `{"error":{"message":"slow down","code":"429"}}`,
		reply{header: map[string]string{"x-litellm-call-id": "call-rejected"}})
	var logs bytes.Buffer
	c := gatewayClient(t, srv.URL, slog.New(slog.NewTextHandler(&logs, nil)))
	if _, _, err := c.Chat(context.Background(), ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}}); err == nil {
		t.Fatal("expected the 429")
	}
	if !strings.Contains(logs.String(), "gateway_call_id=call-rejected") {
		t.Errorf("the failure line must carry the call id: %s", logs.String())
	}
}

func TestStream_GatewayHeadersSurviveACutStream(t *testing.T) {
	// One frame, then EOF without [DONE] or finish_reason: a cut stream.
	srv := sseServer(t, []string{`data: {"choices":[{"delta":{"content":"hi"}}]}`},
		reply{header: map[string]string{"x-litellm-call-id": "call-cut"}})
	st, _, err := gatewayClient(t, srv.URL, nil).ChatStream(context.Background(),
		ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := st.Collect(nil)
	if err == nil {
		t.Fatal("a cut stream is an error")
	}
	if res.Gateway.CallID != "call-cut" {
		t.Errorf("gateway = %+v, want the call id kept on the error path", res.Gateway)
	}
}

func TestStream_GatewayCallIDIsLoggedOnARejectedOpen(t *testing.T) {
	srv := jsonServer(t, http.StatusBadGateway, `{"error":{"message":"upstream down","code":"502"}}`,
		reply{header: map[string]string{"x-litellm-call-id": "call-stream-rejected"}})
	var logs bytes.Buffer
	c := gatewayClient(t, srv.URL, slog.New(slog.NewTextHandler(&logs, nil)))
	if _, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}}); err == nil {
		t.Fatal("expected the 502")
	}
	if !strings.Contains(logs.String(), "gateway_call_id=call-stream-rejected") {
		t.Errorf("the failure line must carry the call id: %s", logs.String())
	}
}

const gatewayChatBody = `{"model":"ai-gateway-gpt-5.4-mini","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`

func TestChat_GatewayHeadersRideOnTheResponse(t *testing.T) {
	srv := jsonServer(t, http.StatusOK, gatewayChatBody, reply{header: map[string]string{
		"x-litellm-call-id":       "call-42",
		"x-litellm-model-name":    "azure/gpt-5.4-mini",
		"x-litellm-key-spend":     "12.5",
		"x-litellm-response-cost": "1.23e-05",
	}})

	resp, warnings, err := gatewayClient(t, srv.URL, nil).Chat(context.Background(),
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
	srv := jsonServer(t, http.StatusOK, gatewayChatBody)
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
			header := map[string]string{"x-litellm-response-cost": "0.001"}
			if tc.value != "" {
				header["x-litellm-key-spend"] = tc.value
			}
			srv := jsonServer(t, http.StatusOK, gatewayChatBody, reply{header: header})
			resp, warnings, err := gatewayClient(t, srv.URL, nil).Chat(context.Background(),
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

// Header values as LiteLLM's cost_headers() sends them for a prompt-cached
// call: input already excludes both cache lanes, reasoning sits inside output.
func TestChat_GatewayCostBreakdownRidesOnTheResponse(t *testing.T) {
	var logs bytes.Buffer
	srv := jsonServer(t, http.StatusOK, gatewayChatBody, reply{header: map[string]string{
		"x-litellm-version":                      "1.107.0",
		"x-litellm-response-cost":                "0.0042",
		"x-litellm-response-cost-input":          "0.001",
		"x-litellm-response-cost-cache-read":     "0.0001",
		"x-litellm-response-cost-cache-creation": "0.0011",
		"x-litellm-response-cost-output":         "0.002",
		"x-litellm-response-cost-reasoning":      "0.0015",
		"x-litellm-response-cost-original":       "0.0042",
		"x-litellm-response-cost-margin-percent": "0.0",
	}})
	resp, warnings, err := gatewayClient(t, srv.URL, slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))).
		Chat(context.Background(), ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Errorf("warnings = %v, want none", warnings)
	}
	g := resp.Gateway
	if g.Version != "1.107.0" || !g.Cost.Reported() {
		t.Errorf("gateway = %+v", g)
	}
	c := g.Cost
	for name, tc := range map[string]struct {
		got  *int64
		want int64
	}{
		"input": {c.InputNanoUSD, 1_000_000}, "cache read": {c.CacheReadNanoUSD, 100_000},
		"cache write": {c.CacheWriteNanoUSD, 1_100_000}, "output": {c.OutputNanoUSD, 2_000_000},
		"reasoning": {c.ReasoningNanoUSD, 1_500_000},
	} {
		if tc.got == nil || *tc.got != tc.want {
			t.Errorf("%s = %v, want %d", name, tc.got, tc.want)
		}
	}
	// Not sent means nil, never a zero lane.
	if c.ToolUsageNanoUSD != nil {
		t.Errorf("tool usage = %d, want nil", *c.ToolUsageNanoUSD)
	}
	if c.Original != "0.0042" || c.MarginPercent != "0.0" || c.DiscountAmount != "" {
		t.Errorf("raw = %+v", c)
	}
	// The total stays the cost header, never a sum of lanes.
	if resp.Usage.Cost.NanoUSD != 4_200_000 {
		t.Errorf("cost = %d, want the header total", resp.Usage.Cost.NanoUSD)
	}
	for _, want := range []string{"gateway_version=1.107.0", "gateway_cache_read_usd=0.000100", "gateway_cache_write_usd=0.001100"} {
		if !strings.Contains(logs.String(), want) {
			t.Errorf("log line lacks %s: %s", want, logs.String())
		}
	}
}

// A margin makes the reported total the marked-up figure: kept as reported,
// but never passed off silently as the list rate.
func TestChat_GatewayMarginIsFlagged(t *testing.T) {
	cases := []struct {
		name   string
		header map[string]string
		warn   bool
	}{
		{"margin", map[string]string{"x-litellm-response-cost-margin-amount": "0.0007"}, true},
		{"margin percent", map[string]string{"x-litellm-response-cost-margin-percent": "0.2"}, true},
		{"discount", map[string]string{"x-litellm-response-cost-discount-amount": "0.001"}, true},
		{"negative margin", map[string]string{"x-litellm-response-cost-margin-percent": "-0.1"}, true},
		{"zero margin", map[string]string{"x-litellm-response-cost-margin-amount": "0.0"}, false},
		{"none literal", map[string]string{"x-litellm-response-cost-margin-amount": "None"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.header["x-litellm-response-cost"] = "0.0042"
			tc.header["x-litellm-response-cost-original"] = "0.0035"
			srv := jsonServer(t, http.StatusOK, gatewayChatBody, reply{header: tc.header})
			resp, warnings, err := gatewayClient(t, srv.URL, nil).Chat(context.Background(),
				ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
			if err != nil {
				t.Fatal(err)
			}
			var seen bool
			for _, w := range warnings {
				if w.Feature == "gateway_cost" && strings.Contains(w.Details, "not the list rate") {
					seen = true
				}
			}
			if seen != tc.warn {
				t.Errorf("margin warning = %v, want %v (warnings %v)", seen, tc.warn, warnings)
			}
			if resp.Usage.Cost.NanoUSD != 4_200_000 || resp.Usage.Cost.Provenance != Reported {
				t.Errorf("cost = %+v, want the reported total kept", resp.Usage.Cost)
			}
		})
	}
}

// An adjustment beside a total that was never used says nothing true.
func TestChat_GatewayMarginOnAnUnpricedCallIsSilent(t *testing.T) {
	srv := jsonServer(t, http.StatusOK, gatewayChatBody, reply{header: map[string]string{
		"x-litellm-call-id":                     "call-9",
		"x-litellm-response-cost":               "None",
		"x-litellm-response-cost-margin-amount": "0.0007",
	}})
	_, warnings, err := gatewayClient(t, srv.URL, nil).Chat(context.Background(),
		ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range warnings {
		if w.Feature == "gateway_cost" {
			t.Errorf("warning %v on an unpriced call", w)
		}
	}
}

// A stream's headers precede its body, so a split there prices nothing:
// dropped, never a confident zero.
func TestStream_GatewayCostSplitIsDropped(t *testing.T) {
	srv := sseServer(t, hiStream, reply{header: map[string]string{
		"x-litellm-call-id":                  "call-stream",
		"x-litellm-response-cost-input":      "0",
		"x-litellm-response-cost-cache-read": "0",
	}})
	st, _, err := gatewayClient(t, srv.URL, nil).ChatStream(context.Background(),
		ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := st.Collect(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Gateway.CallID != "call-stream" || res.Gateway.Cost.Reported() {
		t.Errorf("gateway = %+v, want the call id and no cost split", res.Gateway)
	}
}

func TestChat_GatewayCostLaneVariants(t *testing.T) {
	cases := []struct {
		name, value string
		want        *int64
		warn        bool
	}{
		{"absent", "", nil, false},
		{"none literal", "None", nil, false},
		{"response cache hit", "0.0", ptr(int64(0)), false},
		{"nan", "NaN", nil, true},
		{"negative", "-0.1", nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			header := map[string]string{"x-litellm-response-cost": "0.001"}
			if tc.value != "" {
				header["x-litellm-response-cost-cache-read"] = tc.value
			}
			srv := jsonServer(t, http.StatusOK, gatewayChatBody, reply{header: header})
			resp, warnings, err := gatewayClient(t, srv.URL, nil).Chat(context.Background(),
				ChatRequest{Model: "gpt-5.4-mini", Messages: []Message{User("hi")}})
			if err != nil {
				t.Fatal(err)
			}
			got := resp.Gateway.Cost.CacheReadNanoUSD
			switch {
			case tc.want == nil && got != nil:
				t.Errorf("cache read = %d, want nil", *got)
			case tc.want != nil && (got == nil || *got != *tc.want):
				t.Errorf("cache read = %v, want %d", got, *tc.want)
			}
			var seen bool
			for _, w := range warnings {
				if w.Feature == "gateway_cost" {
					seen = true
				}
			}
			if seen != tc.warn {
				t.Errorf("gateway_cost warning = %v, want %v (warnings %v)", seen, tc.warn, warnings)
			}
			if resp.Usage.Cost.Provenance != Reported {
				t.Errorf("a bad lane must not touch the total: %v", resp.Usage.Cost.Provenance)
			}
		})
	}
}

func TestChat_NoneCostLiteralIsUnpricedNotCorrupt(t *testing.T) {
	srv := jsonServer(t, http.StatusOK, gatewayChatBody, reply{header: map[string]string{
		"x-litellm-call-id":       "call-7",
		"x-litellm-response-cost": "None",
	}})
	resp, warnings, err := gatewayClient(t, srv.URL, nil).Chat(context.Background(),
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
	srv := sseServer(t, hiStream, reply{header: map[string]string{
		"x-litellm-call-id":    "call-stream",
		"x-litellm-model-name": "azure/gpt-5.4-mini",
	}})
	st, _, err := gatewayClient(t, srv.URL, nil).ChatStream(context.Background(),
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
