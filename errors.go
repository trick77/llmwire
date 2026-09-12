package llmwire

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Error handling for OpenAI-compatible endpoints, which agree on the shape of a
// successful response and on almost nothing about the shape of a failure.
//
// Two rules drive everything below, and both were learned the expensive way:
//
//   - Dispatch on the error CODE, never on the message text. Z.ai returns code
//     1210 for "this model always thinks", and the live body for it is in
//     Chinese ("该模型始终思考，不支持关闭思考；请使用 low、high 或 max"), while
//     the published table renders the same code as the generic English "Invalid
//     API parameter, please check the documentation." The same code is also
//     overloaded onto unavailable-model errors. Any matcher keyed on the prose
//     is wrong in at least one of those cases.
//   - Redact before anything is stored. An upstream error body is echoed back
//     verbatim, and some OpenAI-compatible deployments carry the API key in the
//     query string, so an unredacted error reaches a log file or a committed
//     findings note with a live credential inside it.

// Sentinel errors for the failure classes a caller acts on differently. Compare
// with errors.Is; the concrete *APIError is still available via errors.As when
// the status, code or param matter.
var (
	// ErrRateLimited is a 429 from any layer — the model, the gateway's own
	// key limit, or a budget cap. A typed *RateLimitError carries Retry-After
	// when the endpoint sent one.
	ErrRateLimited = errors.New("rate limited")
	// ErrAuth is a 401/403.
	ErrAuth = errors.New("authentication failed")
	// ErrBadRequest is a 4xx that is neither of the above: a rejected
	// parameter, an unknown model, a context overflow.
	ErrBadRequest = errors.New("bad request")
	// ErrUpstream is a 5xx, or a transport failure with no status at all.
	ErrUpstream = errors.New("upstream failure")
)

// APIError is a decoded error response. Every field is best-effort: endpoints
// populate different subsets, and an absent field must read as "not reported"
// rather than as an empty value that means something.
type APIError struct {
	// StatusCode is the HTTP status, or 0 when the failure happened before a
	// response (dial, TLS, a stall guard firing).
	StatusCode int
	// Code is the provider's own error code, ALWAYS held as a string. MiMo and
	// LiteLLM send it as a JSON string; Z.ai's schema declares it an integer
	// and its live bodies send a string. Keeping one Go type for both means
	// call sites compare against "1210", never against two spellings.
	Code string
	// Type is OpenAI's error type ("invalid_request_error", …) when present.
	Type string
	// Param names the request field the endpoint objected to, when it says.
	Param string
	// Message is the human-readable text, already redacted.
	Message string
	// Class is the sentinel this error matches, for errors.Is.
	Class error
}

func (e *APIError) Error() string {
	var b strings.Builder
	b.WriteString("llmwire: ")
	if e.StatusCode != 0 {
		fmt.Fprintf(&b, "status %d", e.StatusCode)
	} else {
		b.WriteString("no response")
	}
	if e.Code != "" {
		fmt.Fprintf(&b, " code %s", e.Code)
	}
	if e.Param != "" {
		fmt.Fprintf(&b, " param %s", e.Param)
	}
	if e.Message != "" {
		fmt.Fprintf(&b, ": %s", e.Message)
	}
	return b.String()
}

// Is lets errors.Is(err, ErrRateLimited) and friends work against the class the
// status mapped to.
func (e *APIError) Is(target error) bool { return target == e.Class }

// RateLimitError is a 429 that carried a Retry-After header.
type RateLimitError struct {
	*APIError
	// RetryAfter is how long the endpoint asked us to wait. Zero means it sent
	// no usable hint — NOT "retry immediately".
	RetryAfter time.Duration
}

// classify maps an HTTP status onto a sentinel. Deliberately coarse: the
// interesting distinctions (which param, which code) live in the struct, and a
// finer taxonomy here would only invite call sites to switch on the wrong thing.
//
// Note the asymmetry a LiteLLM proxy introduces and which this mapping handles
// without a special case: a budget-exceeded failure arrives as 429, not 402,
// and an unknown model name arrives as 400, not 404.
func classify(status int) error {
	switch {
	case status == http.StatusTooManyRequests:
		return ErrRateLimited
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return ErrAuth
	case status >= 400 && status < 500:
		return ErrBadRequest
	default:
		return ErrUpstream
	}
}

// errorEnvelope is the dominant shape: {"error": {...}}.
//
// Code is json.RawMessage because the wire type genuinely varies (see
// APIError.Code); decoding it into a string field directly fails on Z.ai, and
// into a number fails on MiMo.
type errorEnvelope struct {
	Error struct {
		Message string          `json:"message"`
		Type    string          `json:"type"`
		Param   *string         `json:"param"`
		Code    json.RawMessage `json:"code"`
	} `json:"error"`
	// Detail is the FastAPI shape a LiteLLM proxy falls back to for validation
	// failures and for its catch-all handler. It is a string, an object or an
	// array depending on which handler produced it, so it is kept raw and
	// rendered rather than decoded into a fixed type.
	Detail json.RawMessage `json:"detail"`
}

// parseAPIError decodes an error response body. It never returns nil: a body it
// cannot parse still yields an APIError carrying the (redacted, truncated) text,
// because a caller that gets nil here would report a failure as a success.
func parseAPIError(status int, body []byte) *APIError {
	e := &APIError{StatusCode: status, Class: classify(status)}

	var env errorEnvelope
	if err := json.Unmarshal(body, &env); err == nil {
		if env.Error.Message != "" || len(env.Error.Code) > 0 || env.Error.Type != "" {
			e.Message = Redact(env.Error.Message)
			e.Type = env.Error.Type
			if env.Error.Param != nil {
				e.Param = *env.Error.Param
			}
			e.Code = decodeCode(env.Error.Code)
			return e
		}
		if len(env.Detail) > 0 {
			e.Message = Redact(renderDetail(env.Detail))
			return e
		}
	}

	// A BARE error object, with no {"error": …} wrapper. This is the shape an
	// error frame carries inside an SSE stream, and some endpoints use it for
	// ordinary error responses too.
	var bare struct {
		Message string          `json:"message"`
		Type    string          `json:"type"`
		Param   *string         `json:"param"`
		Code    json.RawMessage `json:"code"`
	}
	if err := json.Unmarshal(body, &bare); err == nil && (bare.Message != "" || len(bare.Code) > 0) {
		e.Message = Redact(bare.Message)
		e.Type = bare.Type
		if bare.Param != nil {
			e.Param = *bare.Param
		}
		e.Code = decodeCode(bare.Code)
		return e
	}

	// Unparseable: keep the text, bounded and redacted. A truncated body is far
	// better than none — this is the only evidence of what an undocumented
	// endpoint objected to.
	e.Message = Redact(truncate(strings.TrimSpace(string(body)), maxErrorBody))
	return e
}

// decodeCode renders the provider's code as a string whatever its JSON type.
// A float is formatted without an exponent or trailing ".0", so a numeric 1210
// and a string "1210" compare equal at the call site.
func decodeCode(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err == nil {
		return strconv.FormatFloat(f, 'f', -1, 64)
	}
	return strings.Trim(string(raw), `"`)
}

// renderDetail flattens the FastAPI "detail" field to one line. An object with a
// single "error" key — the shape a LiteLLM model-not-found produces — is
// unwrapped to just its value, since the wrapper carries no information.
func renderDetail(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		if inner, ok := obj["error"]; ok && len(obj) == 1 {
			return renderDetail(inner)
		}
	}
	return string(raw)
}

// newRateLimitError attaches a parsed Retry-After to a 429.
func newRateLimitError(e *APIError, h http.Header) *RateLimitError {
	return &RateLimitError{APIError: e, RetryAfter: retryAfter(h, time.Now())}
}

// retryAfter parses the Retry-After header, which RFC 9110 allows in two forms:
// a delay in seconds, or an HTTP-date. Both appear in the wild, so both are
// handled; anything else yields 0, meaning "no usable hint".
//
// A date in the past yields 0 rather than a negative duration, so a caller can
// treat the value as a sleep length without guarding the sign.
func retryAfter(h http.Header, now time.Time) time.Duration {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs <= 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// --- redaction ---------------------------------------------------------------

// Key shapes the endpoints this library talks to actually issue. The length
// bound is what stops prose ("the sk- prefix") from being mangled; a real key is
// far longer than sixteen characters.
var keyPattern = regexp.MustCompile(`(sk|tp)-[A-Za-z0-9_-]{16,}`)

// Query parameters that carry a credential. Some OpenAI-compatible deployments
// accept the key in the URL rather than a header, which is how a bare
// fmt.Errorf("%s: %w", url, err) leaks one into a log.
var keyParamPattern = regexp.MustCompile(`(?i)([?&](?:api[-_]?key|key|token|access[-_]?token)=)[^&\s]*`)

// Redact removes anything shaped like a credential from a string bound for a
// log, an error, or a committed findings note. It is applied at construction
// time rather than at print time: a value that is never redacted until it is
// printed is one Printf away from a leak.
func Redact(s string) string {
	if s == "" {
		return s
	}
	s = keyPattern.ReplaceAllString(s, "[REDACTED]")
	s = keyParamPattern.ReplaceAllString(s, "${1}[REDACTED]")
	return s
}

// RedactURL renders a URL as scheme://host/path, dropping the query and any
// userinfo. Used wherever an endpoint is named in an error or an observation:
// the host and path are what identify a deployment, and the query is where a
// credential hides.
func RedactURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return Redact(raw)
	}
	u.RawQuery = ""
	u.Fragment = ""
	u.User = nil
	return u.String()
}

// truncate caps a string at max bytes, cutting on a rune boundary and marking
// the cut. Byte-offset truncation would split a multi-byte character and put
// invalid UTF-8 in the log, which is not hypothetical: these endpoints return
// non-ASCII error text routinely (see the Chinese 1210 body above).
func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…(truncated)"
}

// utf8RuneStart reports whether b is the first byte of a UTF-8 sequence.
func utf8RuneStart(b byte) bool { return b&0xC0 != 0x80 }
