package llmwire

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
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

	// ErrMalformedResponse is a 2xx whose body could not be read as an answer
	// — an HTML error page from a proxy, a stream that ended without
	// finish_reason or [DONE]. The wrapped error carries the decoder's own
	// complaint and, for a body, a redacted and bounded slice of it.
	ErrMalformedResponse = errors.New("malformed response")
	// ErrResponseShape is a 2xx that decoded but does not answer the request:
	// no choices and no error object, the wrong number of embeddings, an
	// embedding index outside or repeated within the batch. Distinct from
	// ErrMalformedResponse because a caller retries the two differently — a
	// proxy page is transient, a shape the endpoint chose to send is not.
	ErrResponseShape = errors.New("response shape")

	// The three bounds this package enforces on a call, so a caller can act
	// on which one gave up without matching the message. Each is wrapped by
	// the error that names the duration: "llmwire: stream idle for 1m30s".
	//
	// ErrNoResponseHeaders: the endpoint sent nothing within the header bound
	// (HeaderTimeout on a stream; the call cap on a non-streaming call, which
	// withholds headers until the whole answer is ready).
	ErrNoResponseHeaders = errors.New("no response headers")
	// ErrStreamIdle: a started stream sent no data frame for IdleTimeout (or
	// the request's ToolCallIdleTimeout once a tool call was underway).
	ErrStreamIdle = errors.New("stream idle")
	// ErrCallCap: the whole call outran CallTimeout.
	ErrCallCap = errors.New("exceeded the call cap")
)

// APIError is a decoded error response. Every field is best-effort: endpoints
// populate different subsets, and an absent field must read as "not reported"
// rather than as an empty value that means something.
type APIError struct {
	// StatusCode is the HTTP status, or 0 for an error object inside a
	// SUCCESSFUL response: a 2xx whose body is {"error":…}, or an error frame
	// in a stream. A dial or TLS failure and a stall guard firing are not
	// APIErrors; they carry no body to decode.
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
		b.WriteString("error in a 2xx response")
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

// RateLimitError is every 429, whether or not it carried a Retry-After
// header: the type is what a caller dispatches on, and the hint is optional.
type RateLimitError struct {
	*APIError
	// RetryAfter is how long the endpoint asked us to wait. Zero means it sent
	// no usable hint — NOT "retry immediately".
	RetryAfter time.Duration
}

// Unwrap exposes the embedded *APIError to errors.As.
//
// Embedding alone is not enough and the difference is silent: errors.As matches
// on the dynamic type or on what Unwrap returns, and *RateLimitError is neither
// an *APIError nor, without this, wrapping one. So
//
//	var apiErr *APIError
//	errors.As(err, &apiErr)
//
// returned FALSE for exactly one status — 429 — while succeeding for every other
// refusal. A caller rendering "failed with status %d" from that check therefore
// dropped the one status worth retrying on, and the only visible symptom was a
// differently-worded log line. The first consumer to migrate hit it, and it would
// have been a per-repo rediscovery otherwise.
func (e *RateLimitError) Unwrap() error { return e.APIError }

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

// errorObject is the error record itself, as it appears under {"error": …}
// and, on some endpoints and in every SSE error frame, bare.
type errorObject struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Param   *string         `json:"param"`
	Code    json.RawMessage `json:"code"`
}

// errorEnvelope is the dominant shape: {"error": {...}}.
//
// Code is json.RawMessage because the wire type genuinely varies (see
// APIError.Code); decoding it into a string field directly fails on Z.ai, and
// into a number fails on MiMo.
type errorEnvelope struct {
	Error errorObject `json:"error"`
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
	return parseAPIErrorWith(Redact, status, body)
}

// cleanText is THE rule for upstream text an error keeps: redacted, then
// bounded, in that order. A cut that lands inside a credential leaves its
// head behind for a pass that only knows the whole value.
func cleanText(redact redactor, s string) string {
	return Truncate(redact(s), maxErrorBody)
}

// bodySnippet is cleanText over a body that an error may quote: enough to
// name what answered, never enough to leak what it said.
func bodySnippet(redact redactor, raw []byte) string {
	return cleanText(redact, string(raw))
}

// redactor is what turns upstream text into loggable text. Redact is the
// shape-only default; a Client supplies one that also strips its own key by
// value, and runs that FIRST — the shape pass can eat the middle of a key
// whose value contains an sk-/tp- run, after which the value pass would find
// nothing to strip.
type redactor func(string) string

func parseAPIErrorWith(redact redactor, status int, body []byte) *APIError {
	e := &APIError{StatusCode: status, Class: classify(status)}
	body = unframeSSE(body)

	// Every field kept from the body goes through cleanText. Bounded because
	// an in-band error in a 200 body or a stream frame is not read through
	// httpError's LimitReader, so nothing upstream has capped it.
	clean := func(s string) string { return cleanText(redact, s) }

	var env errorEnvelope
	if err := json.Unmarshal(body, &env); err == nil {
		if env.Error.Message != "" || len(env.Error.Code) > 0 || env.Error.Type != "" {
			e.fill(clean, env.Error)
			return e
		}
		if len(env.Detail) > 0 {
			e.Message = clean(renderDetail(env.Detail))
			return e
		}
	}

	// A BARE error object, with no {"error": …} wrapper. This is the shape an
	// error frame carries inside an SSE stream, and some endpoints use it for
	// ordinary error responses too.
	var bare errorObject
	if err := json.Unmarshal(body, &bare); err == nil && (bare.Message != "" || len(bare.Code) > 0) {
		e.fill(clean, bare)
		return e
	}

	// Unparseable: keep the text. A truncated body is far better than none —
	// this is the only evidence of what an undocumented endpoint objected to.
	e.Message = clean(strings.TrimSpace(string(body)))
	return e
}

// fill copies a decoded error object into e, every string through clean. The
// code and param are cleaned too: they are upstream text like the message,
// and an endpoint that echoes the request into `param` echoes its key.
func (e *APIError) fill(clean func(string) string, o errorObject) {
	e.Message = clean(o.Message)
	e.Type = clean(o.Type)
	if o.Param != nil {
		e.Param = clean(*o.Param)
	}
	e.Code = clean(decodeCode(o.Code))
}

// unframeSSE strips an SSE "data:" wrapper from an error body.
//
// An endpoint that has already decided to answer in text/event-stream can send
// its ERROR that way too, even under a non-2xx status. Measured against MiMo:
// asking mimo-v2.5-pro for image input returns HTTP 404 whose body is the single
// line
//
//	data:{"error":{"code":"404","message":"No endpoints found that support image input",...}}
//
// Without this the JSON never parses and the whole envelope degrades to raw
// text, losing the code that callers are supposed to dispatch on — which is the
// one field this package insists on.
//
// Only a single leading frame is unwrapped, and only when what follows parses as
// JSON, so an ordinary body that happens to begin with "data:" is left alone.
func unframeSSE(body []byte) []byte {
	trimmed := bytes.TrimSpace(body)
	if !bytes.HasPrefix(trimmed, []byte(dataPrefix)) {
		return body
	}
	// Take the first frame only; an error body may still be newline-terminated.
	if i := bytes.IndexByte(trimmed, '\n'); i >= 0 {
		trimmed = trimmed[:i]
	}
	payload := bytes.TrimSpace(trimmed[len(dataPrefix):])
	if !json.Valid(payload) {
		return body
	}
	return payload
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
	// An object or array where a scalar belongs: kept as text so a caller
	// still sees what was sent. fill bounds it with every other field.
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

// newRateLimitError attaches a parsed Retry-After to a 429. now is the
// client's clock, so an HTTP-date form measures against the same time source
// every other figure in the package uses.
func newRateLimitError(e *APIError, h http.Header, now time.Time) *RateLimitError {
	return &RateLimitError{APIError: e, RetryAfter: retryAfter(h, now)}
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
	if secs, err := strconv.ParseInt(v, 10, 64); err == nil {
		if secs <= 0 {
			return 0
		}
		// Clamped: a delay past what a Duration holds would wrap negative on
		// the multiply, and a negative value here is documented never to
		// happen. Nobody sleeps 292 years either way.
		if secs > math.MaxInt64/int64(time.Second) {
			return time.Duration(math.MaxInt64)
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

// Truncate caps a string at max bytes, cutting on a rune boundary and marking
// the cut. Byte-offset truncation would split a multi-byte character and put
// invalid UTF-8 in the log, which is not hypothetical: these endpoints return
// non-ASCII error text routinely (see the Chinese 1210 body above). Exported
// for the same reason Redact is: every consumer logging a raw usage object or
// an error body needs exactly this and was carrying its own copy.
func Truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…(truncated)"
}

// JSONObject returns the first JSON object in s, or false when there is none.
// It is the salvage step before json.Unmarshal on a reply that was asked for
// JSON: models fence it in ```json, lead with "Sure, here it is:", or trail a
// sentence after the closing brace, and every consumer was carrying its own
// strip-the-fence helper — three of them in one program, each slightly
// different.
//
// Each '{' is tried as a start, the candidate is cut where its braces balance
// (string-aware: a brace inside a JSON string does not count, an escaped
// quote does not close the string), and the first candidate that is valid
// JSON wins. Trying every start is what survives prose that mentions a brace
// before the object — `wrap it in {braces}: {"a":1}` yields the second. It is
// the FIRST object, not first-brace-to-last-brace: a reply offering two
// parses on the first, where the widest cut includes the prose and fails.
// What it does not do is repair: an unbalanced object is reported as none,
// and the caller's parse failure path is the right place for that.
//
// Consider ChatRequest.ResponseFormat first. On one model a prompt asking for
// JSON parsed 0 times out of 8 and the response format 8 out of 8 — but a
// fenced reply still arrives with the format set on some endpoints, so the
// salvage stays useful behind it.
func JSONObject(s string) (string, bool) {
	for start := strings.IndexByte(s, '{'); start >= 0; {
		if end := balancedObjectEnd(s, start); end > start {
			if candidate := s[start:end]; json.Valid([]byte(candidate)) {
				return candidate, true
			}
		}
		next := strings.IndexByte(s[start+1:], '{')
		if next < 0 {
			break
		}
		start += 1 + next
	}
	return "", false
}

// balancedObjectEnd returns the index one past the brace that closes the
// object opening at start, or -1 when it never closes.
func balancedObjectEnd(s string, start int) int {
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}
