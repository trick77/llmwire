package llmwire

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

// The provider's error code arrives as a JSON string on MiMo and LiteLLM, and
// Z.ai's schema declares it an integer while its live bodies send a string.
// Both must reach a call site as the same comparable value, or dispatching on
// "1210" works against one endpoint and silently fails against the other.
func TestParseAPIError_CodeIsAStringWhateverTheWireTypeIs(t *testing.T) {
	for _, tc := range []struct {
		name, body, want string
	}{
		{"string code", `{"error":{"message":"nope","code":"1210"}}`, "1210"},
		{"numeric code", `{"error":{"message":"nope","code":1210}}`, "1210"},
		{"numeric code as float", `{"error":{"message":"nope","code":1210.0}}`, "1210"},
		{"absent code", `{"error":{"message":"nope"}}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := parseAPIError(400, []byte(tc.body))
			if got.Code != tc.want {
				t.Errorf("code = %q, want %q", got.Code, tc.want)
			}
		})
	}
}

// The 1210 body is Chinese on the live endpoint while the published table
// renders it in English, so only the code is a safe thing to dispatch on. This
// test exists to keep that decodable.
func TestParseAPIError_NonASCIIMessageSurvivesDecoding(t *testing.T) {
	body := `{"error":{"code":"1210","message":"该模型始终思考，不支持关闭思考；请使用 low、high 或 max。"}}`
	got := parseAPIError(400, []byte(body))
	if got.Code != "1210" {
		t.Fatalf("code = %q, want 1210", got.Code)
	}
	if !strings.Contains(got.Message, "该模型始终思考") {
		t.Errorf("message did not survive: %q", got.Message)
	}
}

func TestParseAPIError_FullEnvelopeFields(t *testing.T) {
	body := `{"error":{"message":"Unsupported value","type":"invalid_request_error","param":"temperature","code":"unsupported_value"}}`
	got := parseAPIError(400, []byte(body))
	if got.Type != "invalid_request_error" {
		t.Errorf("type = %q", got.Type)
	}
	if got.Param != "temperature" {
		t.Errorf("param = %q, want temperature", got.Param)
	}
	if got.Code != "unsupported_value" {
		t.Errorf("code = %q", got.Code)
	}
}

// A LiteLLM proxy falls back to FastAPI's "detail" for validation failures and
// for its catch-all handler, in three different shapes.
func TestParseAPIError_DetailShapes(t *testing.T) {
	for _, tc := range []struct {
		name, body, wantSubstr string
	}{
		{"string detail", `{"detail":"Invalid model name passed in"}`, "Invalid model name"},
		{"object wrapping error", `{"detail":{"error":"no healthy deployments"}}`, "no healthy deployments"},
		{"array detail", `{"detail":[{"loc":["body","model"],"msg":"field required"}]}`, "field required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := parseAPIError(400, []byte(tc.body))
			if !strings.Contains(got.Message, tc.wantSubstr) {
				t.Errorf("message = %q, want it to contain %q", got.Message, tc.wantSubstr)
			}
		})
	}
}

// An error frame inside an SSE stream carries a bare error object with no
// {"error": …} wrapper.
func TestParseAPIError_BareErrorObject(t *testing.T) {
	got := parseAPIError(0, []byte(`{"message":"stream died","type":"server_error","code":"500"}`))
	if got.Code != "500" || !strings.Contains(got.Message, "stream died") {
		t.Errorf("got %+v", got)
	}
}

// A body this package cannot parse must still produce an error carrying the
// text: returning nil would report a failure as a success, and dropping the
// body would discard the only evidence of what an undocumented endpoint
// objected to.
func TestParseAPIError_UnparseableBodyKeepsItsText(t *testing.T) {
	got := parseAPIError(502, []byte("<html><body>Bad Gateway</body></html>"))
	if got == nil {
		t.Fatal("parseAPIError returned nil")
	}
	if !strings.Contains(got.Message, "Bad Gateway") {
		t.Errorf("message = %q, want the raw text", got.Message)
	}
	if !errors.Is(got, ErrUpstream) {
		t.Error("a 502 should classify as ErrUpstream")
	}
}

// A LiteLLM budget failure is 429 and an unknown model is 400, not 402 and 404.
// The mapping has to hold for those without a special case.
func TestClassify(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   error
	}{
		{429, ErrRateLimited},
		{401, ErrAuth},
		{403, ErrAuth},
		{400, ErrBadRequest},
		{404, ErrBadRequest},
		{413, ErrBadRequest},
		{500, ErrUpstream},
		{502, ErrUpstream},
		{0, ErrUpstream},
	} {
		if got := classify(tc.status); got != tc.want {
			t.Errorf("classify(%d) = %v, want %v", tc.status, got, tc.want)
		}
	}
}

func TestAPIError_IsMatchesItsClass(t *testing.T) {
	err := parseAPIError(429, []byte(`{"error":{"message":"slow down"}}`))
	if !errors.Is(err, ErrRateLimited) {
		t.Error("429 should match ErrRateLimited")
	}
	if errors.Is(err, ErrAuth) {
		t.Error("429 should not match ErrAuth")
	}
}

// RFC 9110 allows Retry-After as a delay in seconds or an HTTP-date, and both
// appear in the wild.
func TestRetryAfter(t *testing.T) {
	now := time.Date(2026, 9, 12, 12, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		name, value string
		want        time.Duration
	}{
		{"seconds", "30", 30 * time.Second},
		{"http date in the future", now.Add(45 * time.Second).Format(http.TimeFormat), 45 * time.Second},
		{"http date in the past", now.Add(-time.Hour).Format(http.TimeFormat), 0},
		{"zero seconds", "0", 0},
		{"negative seconds", "-5", 0},
		{"garbage", "soon", 0},
		{"absent", "", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := http.Header{}
			if tc.value != "" {
				h.Set("Retry-After", tc.value)
			}
			if got := retryAfter(h, now); got != tc.want {
				t.Errorf("retryAfter(%q) = %v, want %v", tc.value, got, tc.want)
			}
		})
	}
}

// --- redaction ---------------------------------------------------------------

// fakeKey builds a credential-shaped string at runtime.
//
// It is assembled rather than written as a literal on purpose:
// hack/secret-scan.sh refuses anything key-shaped entering the tree, and these
// tests would otherwise be the one place in the repo needing an exemption from
// the very guard they exercise. Composing the fixture keeps the scanner
// absolute — no allowlist, no marker comment, nothing that could be copied into
// a file which really is leaking — while still exercising the real pattern.
func fakeKey(prefix, body string) string { return prefix + "-" + body }

// An upstream error body is echoed back verbatim and can carry a credential.
// Redaction runs at construction time, not print time: a value redacted only
// when printed is one Printf away from a leak.
func TestRedact_KeyShapes(t *testing.T) {
	openai := fakeKey("sk", "abcdefghijklmnopqrstuvwxyz012345")
	tokenPlan := fakeKey("tp", "ABCDEFGHIJKLMNOPQRSTUV")
	shortish := fakeKey("sk", "0123456789abcdefghij")

	for _, tc := range []struct {
		name, in string
		leaked   string
	}{
		{"openai style", "bad key " + openai + " supplied", openai},
		{"token plan style", "auth failed for " + tokenPlan, tokenPlan},
		{"in a json blob", `{"error":{"message":"invalid key ` + shortish + `"}}`, shortish},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Redact(tc.in)
			if strings.Contains(got, tc.leaked) {
				t.Errorf("Redact left the key in: %q", got)
			}
			if !strings.Contains(got, "[REDACTED]") {
				t.Errorf("Redact did not mark the removal: %q", got)
			}
		})
	}
}

// Some deployments carry the key in the query string, which is how a bare
// fmt.Errorf("%s: %w", url, err) leaks one.
func TestRedact_QueryParameters(t *testing.T) {
	for _, in := range []string{
		"https://example.test/v1/chat?api_key=supersecretvalue",
		"https://example.test/v1/chat?foo=1&key=supersecretvalue",
		"https://example.test/v1/chat?access_token=supersecretvalue&x=2",
		"https://example.test/v1/chat?API-KEY=supersecretvalue",
	} {
		got := Redact(in)
		if strings.Contains(got, "supersecretvalue") {
			t.Errorf("Redact(%q) leaked the value: %q", in, got)
		}
	}
}

// Prose mentioning a prefix must not be mangled, or every comment about key
// formats becomes unreadable.
func TestRedact_LeavesProseAlone(t *testing.T) {
	in := "keys use the sk- prefix, or tp- on the token plan"
	if got := Redact(in); got != in {
		t.Errorf("Redact altered prose: %q", got)
	}
}

func TestRedactURL_DropsQueryAndUserinfo(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"https://example.test/v1/chat?api_key=secretvalue123", "https://example.test/v1/chat"},
		{"https://user:pass@example.test/v1", "https://example.test/v1"},
		{"https://example.test/v1#frag", "https://example.test/v1"},
		{"https://example.test/v1", "https://example.test/v1"},
	} {
		if got := RedactURL(tc.in); got != tc.want {
			t.Errorf("RedactURL(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// These endpoints return non-ASCII error text routinely, so a byte-offset cut
// would put invalid UTF-8 in a log.
func TestTruncate_CutsOnARuneBoundary(t *testing.T) {
	s := strings.Repeat("日", 50)
	got := truncate(s, 10)
	if !strings.HasSuffix(got, "…(truncated)") {
		t.Errorf("truncate did not mark the cut: %q", got)
	}
	body := strings.TrimSuffix(got, "…(truncated)")
	for i, r := range body {
		if r == '�' {
			t.Fatalf("truncate produced invalid UTF-8 at byte %d", i)
		}
	}
}

func TestTruncate_ShortStringIsUntouched(t *testing.T) {
	if got := truncate("short", 100); got != "short" {
		t.Errorf("truncate altered a short string: %q", got)
	}
}
