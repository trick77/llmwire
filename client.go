// Package llmwire is a dependency-light client for OpenAI-compatible inference
// endpoints — chat, tools, vision and embeddings — with each model's quirks held
// as data rather than scattered through call sites.
//
// It speaks exactly one wire protocol: POST {base}/chat/completions and
// POST {base}/embeddings. Anthropic, Gemini-native and Bedrock are out of scope
// permanently; supporting a second protocol would require the provider
// abstraction this package exists to avoid.
//
// This file is the transport. It knows nothing about model profiles: it takes a
// request body already built, sends it, and returns what came back. Profile
// lookup, validation and pricing sit above it.
package llmwire

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

// Bounds.
//
// Three separate timeouts replace the single whole-request timeout these
// clients usually carry. That one timeout has to cover both "the endpoint never
// answered" and "the model is thinking hard", so it can only ever be wrong for
// one of them — and when it fires, minutes in, the log cannot say which
// happened. Splitting them makes each failure nameable.
//
// http.Client.Timeout is deliberately NOT set anywhere in this package: it caps
// body reads too, so it would cut a legitimately long stream mid-answer.
const (
	// DefaultHeaderTimeout is how long the endpoint may take to send response
	// headers. Generous, because it competes with nothing — a stall costs a
	// minute rather than the whole call budget.
	DefaultHeaderTimeout = 60 * time.Second
	// DefaultIdleTimeout is how long a started stream may go without a data
	// frame. Note that comment and blank lines do NOT re-arm this; see
	// readStream.
	DefaultIdleTimeout = 90 * time.Second
	// DefaultCallTimeout is the backstop for a stream that stays alive forever
	// without finishing — dribbling frames past any sane answer length.
	DefaultCallTimeout = 15 * time.Minute
	// headerBackstopHeadroom is how much later than the stall guard the
	// transport's own header timeout fires. Large enough that the guard always
	// wins the race and names the bound; small enough that a request still ends
	// if the guard somehow never fires at all.
	headerBackstopHeadroom = 30 * time.Second
)

// Config configures the transport.
type Config struct {
	// BaseURL is the OpenAI-compatible root. The client appends the route, so
	// this must NOT already end in /chat/completions.
	BaseURL string
	// APIKey is sent as a bearer token. Optional: some self-hosted endpoints
	// take none, and an empty string sends no Authorization header at all
	// rather than an empty one, which some gateways reject.
	APIKey string
	// UserAgent overrides the default. Go's own "Go-http-client/1.1" names the
	// HTTP library and says nothing about the protocol being spoken; several
	// of these upstreams behave better when told they are talking to an
	// ordinary OpenAI-compatible SDK.
	UserAgent string
	// Headers are added to every request. Used for session-affinity pairs and
	// for gateway-specific headers.
	Headers map[string]string

	HeaderTimeout time.Duration
	IdleTimeout   time.Duration
	CallTimeout   time.Duration

	// HTTPClient is optional. The default has no whole-request timeout (see
	// above) and carries ResponseHeaderTimeout as a backstop under the guard.
	HTTPClient *http.Client

	// Now is the clock, injectable because cost depends on wall-clock time on
	// endpoints with off-peak pricing, and a test that only passes during a
	// particular local window is worse than no test. Defaults to time.Now.
	// It is also the clock behind Timing and MaxCommentGap, so a fixed clock
	// pinned for pricing reports every duration as zero.
	Now func() time.Time

	// Lookup is where FromEnv reads a profile's endpoint variables from. Nil
	// means the process environment. An application whose own config loader
	// is the single source of settings passes its getter; a test passes a
	// map.
	Lookup func(name string) (string, bool)
	// Logger receives one Info line from FromEnv naming the settings the
	// client was built with: model, provider, host, which key variable was
	// read, the identity it presents and the bounds. Never the key itself.
	// Nil means slog.Default(). A client built by New logs nothing: the
	// caller chose every value by hand and has nothing to learn.
	Logger *slog.Logger
	// Registry supplies the model profiles this client validates against.
	// Defaults to the built-in one. Injectable so a caller can add a deployment
	// without waiting for it to be upstreamed here, and so tests can drive
	// validation with a profile that does not exist in the real world.
	Registry *Registry

	// EmulateOpenCode presents every request as the opencode client: its
	// User-Agent, and the session header pair it sends. Headers only, never a
	// request body. It overrides UserAgent, since that string is the whole point;
	// it does not override a session header set by hand in Headers.
	//
	// Some endpoints are sold as one client's backend and treat a neutral
	// User-Agent as a bot. The session id is the Client's own — minted at
	// construction, rotated after an idle gap — and there is deliberately no way
	// to supply one. See identity.go.
	EmulateOpenCode bool
}

// DefaultUserAgent is what this package sends when Config.UserAgent is empty.
const DefaultUserAgent = "llmwire/0.1 (+https://github.com/trick77/llmwire)"

// EmulateOpenCodeEnv is the variable FromEnv reads to present as opencode on a
// provider profiles.yaml does not mark. Global, not per provider; boolean per
// strconv.ParseBool; unset or false leaves the provider's own setting in
// place. It adds the identity, never removes it.
const EmulateOpenCodeEnv = "LLMWIRE_EMULATE_OPENCODE"

// Client is a transport for one endpoint.
type Client struct {
	baseURL   string
	apiKey    string
	userAgent string
	headers   map[string]string
	http      *http.Client
	header    time.Duration
	idle      time.Duration
	cap       time.Duration
	now       func() time.Time
	registry  *Registry
	// session is non-nil only under EmulateOpenCode.
	session *session
}

// New builds a Client.
func New(cfg Config) *Client {
	if cfg.HeaderTimeout <= 0 {
		cfg.HeaderTimeout = DefaultHeaderTimeout
	}
	if cfg.IdleTimeout <= 0 {
		cfg.IdleTimeout = DefaultIdleTimeout
	}
	if cfg.CallTimeout <= 0 {
		cfg.CallTimeout = DefaultCallTimeout
	}
	if cfg.UserAgent == "" {
		cfg.UserAgent = DefaultUserAgent
	}
	if cfg.EmulateOpenCode {
		// After the default, so the emulation is what wins.
		cfg.UserAgent = OpenCodeUserAgent
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Registry == nil {
		cfg.Registry = Default()
	}
	hc := cfg.HTTPClient
	if hc == nil {
		// Clone the stdlib default rather than build a bare Transport, so
		// proxy support, dial timeouts and connection pooling keep their tuned
		// values instead of being silently dropped. Built after the defaults
		// resolve, so the backstop matches the bound actually configured.
		tr := http.DefaultTransport.(*http.Transport).Clone()
		// Deliberately LONGER than the guard's own header bound.
		//
		// Transport.ResponseHeaderTimeout is a backstop for the case the guard
		// cannot cover, not a second bound of equal standing: it is an HTTP/1.1
		// feature and does not apply once a connection is negotiated as HTTP/2,
		// which is what these endpoints do. Setting the two to the same value
		// makes them race, and when the transport wins the error is its generic
		// "timeout awaiting response headers" rather than the guard's named
		// bound — losing exactly the classification the split bounds exist to
		// provide. The headroom makes the guard reliably first.
		tr.ResponseHeaderTimeout = cfg.HeaderTimeout + headerBackstopHeadroom
		hc = &http.Client{Transport: tr}
	}
	headers := make(map[string]string, len(cfg.Headers))
	for k, v := range cfg.Headers {
		headers[k] = v
	}
	c := &Client{
		baseURL:   strings.TrimRight(cfg.BaseURL, "/"),
		apiKey:    cfg.APIKey,
		userAgent: cfg.UserAgent,
		headers:   headers,
		http:      hc,
		header:    cfg.HeaderTimeout,
		idle:      cfg.IdleTimeout,
		cap:       cfg.CallTimeout,
		now:       cfg.Now,
		registry:  cfg.Registry,
	}
	if cfg.EmulateOpenCode {
		c.session = newSession(cfg.Now)
	}
	return c
}

// FromEnv builds a Client for one model. The host comes from the profile's
// provider (profiles.yaml providers:), the key from the environment variable
// that provider implies, LLMWIRE_<PROVIDER>_API_KEY. An application supplies a
// key and nothing else; the URL is the library's to know, so it is not
// repeated in every application's configuration. cfg.Lookup replaces the
// process environment as the source.
//
// LLMWIRE_<PROVIDER>_BASE_URL exists only for a provider that ships no host
// (a self-hosted gateway): there it is required, and unset is a
// MissingEnvError naming it. For a provider that ships a host the variable is
// REFUSED with a StaleEnvError: the host is the library's, and an application
// still carrying one is on the old contract, where every deployment copied
// the URL. Accepting it as an override would let that copy live on unnoticed
// and diverge; a different host is a different provider entry.
//
// A cfg.BaseURL already set means the caller is wiring the endpoint itself,
// and no variable is consulted at all: cfg.APIKey goes as given, empty
// included. The provider's key belongs to the provider's host, and a test
// fake or a stand-in endpoint must not be handed it just because the model is
// the same. With cfg.BaseURL empty, a cfg.APIKey already set still wins over
// the variable.
//
// A key variable that is unset or empty is a MissingEnvError, never a
// fallback. A profile with no_api_key sends no key at all, which is the
// self-hosted case.
//
// A provider marked emulate_opencode in profiles.yaml gets Config.EmulateOpenCode
// set here, which replaces a cfg.UserAgent the caller supplied: on such a host
// the caller's own string is exactly what gets refused. EmulateOpenCodeEnv set
// true does the same for any provider, marked or not: a gateway fronting such
// a host is not in profiles.yaml, and the operator running it knows. A value
// that is not a boolean is an error, never ignored.
func FromEnv(model string, cfg Config) (*Client, error) {
	reg := cfg.Registry
	if reg == nil {
		reg = Default()
	}
	p, err := reg.Lookup(model)
	if err != nil {
		return nil, err
	}
	if cfg.BaseURL != "" {
		return New(cfg), nil
	}
	if p.Provider == "" {
		return nil, &MissingEnvError{Model: model}
	}
	lookup := cfg.Lookup
	if lookup == nil {
		lookup = os.LookupEnv
	}
	// Trimmed: a whitespace-only value (a stray `export X= ` in a .env) would
	// otherwise build a client that fails later with an opaque dial or 401
	// instead of the named error here.
	get := func(name string) string {
		v, _ := lookup(name)
		return strings.TrimSpace(v)
	}
	cfg.BaseURL = p.BaseURL
	if v := get(p.BaseURLEnv()); v != "" {
		if p.BaseURL != "" {
			return nil, &StaleEnvError{Model: model, Provider: p.Provider, Var: p.BaseURLEnv()}
		}
		cfg.BaseURL = v
	}
	if cfg.BaseURL == "" {
		return nil, &MissingEnvError{Model: model, Provider: p.Provider, Var: p.BaseURLEnv()}
	}
	// The identity follows the host: a provider sold as opencode's backend
	// gets that client string without every application knowing to ask. An
	// explicit cfg.BaseURL returned above and gets nothing: the caller is
	// wiring the endpoint itself, identity included.
	if p.EmulateOpenCode {
		cfg.EmulateOpenCode = true
	}
	if v := get(EmulateOpenCodeEnv); v != "" {
		on, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("llmwire: %s=%q is not a boolean", EmulateOpenCodeEnv, v)
		}
		if on {
			cfg.EmulateOpenCode = true
		}
	}
	keyVar := ""
	if cfg.APIKey == "" && !p.NoAPIKey {
		v := get(p.APIKeyEnv())
		if v == "" {
			return nil, &MissingEnvError{Model: model, Provider: p.Provider, Var: p.APIKeyEnv()}
		}
		cfg.APIKey = v
		keyVar = p.APIKeyEnv()
	}
	c := New(cfg)
	logSettings(cfg.Logger, model, p, keyVar, c)
	return c, nil
}

// logSettings is the one line FromEnv writes: everything an operator asks
// "what is it actually talking to" about, at Info, with the key named by its
// variable and never by its value.
func logSettings(log *slog.Logger, model string, p *Profile, keyVar string, c *Client) {
	if log == nil {
		log = slog.Default()
	}
	key := "none"
	switch {
	case keyVar != "":
		key = keyVar
	case c.apiKey != "":
		key = "config"
	}
	log.Info("llmwire client",
		"model", model,
		"provider", p.Provider,
		"base_url", RedactURL(c.baseURL),
		"api_key", key,
		"emulate_opencode", c.session != nil,
		"user_agent", c.userAgent,
		"header_timeout", c.header,
		"idle_timeout", c.idle,
		"call_timeout", c.cap,
	)
}

// MissingEnvError names the variable a profile's provider expects, so the fix
// is one export away rather than a search. Var is empty when the profile
// names no provider at all, which is a profile bug rather than a deployment
// one.
type MissingEnvError struct {
	Model    string
	Provider string
	Var      string
}

func (e *MissingEnvError) Error() string {
	if e.Var == "" {
		return fmt.Sprintf("llmwire: model %q names no provider in its profile, so FromEnv cannot find its endpoint", e.Model)
	}
	return fmt.Sprintf("llmwire: model %q needs %s set in the environment (provider %s)", e.Model, e.Var, e.Provider)
}

// StaleEnvError names a variable from the old contract that is still set:
// LLMWIRE_<PROVIDER>_BASE_URL for a provider whose host the library ships.
// The fix is to delete the line, and the error says so, because a deployment
// carrying a URL the library already knows is one that nobody will notice
// drifting.
type StaleEnvError struct {
	Model    string
	Provider string
	Var      string
}

func (e *StaleEnvError) Error() string {
	return fmt.Sprintf("llmwire: %s is set, but the host for provider %s is the library's (profiles.yaml providers:); "+
		"delete the variable. A different host is a different provider entry, not an override (model %q)",
		e.Var, e.Provider, e.Model)
}

// Now exposes the client's clock, so pricing and tests share one source of time.
func (c *Client) Now() time.Time { return c.now() }

// RawStream POSTs an already-built request body to /chat/completions with
// stream:true and consumes the SSE response.
//
// It takes raw JSON rather than a typed request because it sits below profile
// resolution: the probe suite needs to send deliberately malformed and
// endpoint-specific bodies that no validated type would permit.
//
// onDelta, when non-nil, is called for each content fragment as it arrives. It
// runs on the reading goroutine, so it must not block.
func (c *Client) RawStream(ctx context.Context, body []byte, onDelta func(string)) (StreamResult, error) {
	// Two nested contexts, because their failures mean different things and the
	// error has to say which: callCtx is the overall cap, reqCtx is what the
	// stallGuard cancels. Cancelling reqCtx leaves callCtx's deadline intact so
	// the two stay distinguishable afterwards.
	callCtx, cancelCall := context.WithTimeout(ctx, c.cap)
	defer cancelCall()
	reqCtx, cancelReq := context.WithCancel(callCtx)
	defer cancelReq()

	req, err := c.newRequest(reqCtx, "/chat/completions", body)
	if err != nil {
		return StreamResult{}, err
	}
	req.Header.Set("Accept", "text/event-stream")

	var counters streamCounters
	guard := newStallGuard(cancelReq, c.header, stallHeaders)
	defer guard.stop()

	start := c.now()
	resp, err := c.http.Do(req)
	if err != nil {
		return StreamResult{}, c.explain(ctx, callCtx, guard, c.dialError("/chat/completions", err))
	}
	defer resp.Body.Close()
	headers := c.now().Sub(start)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// The headers figure survives the failure: a 503 after 45s against a
		// 60s header bound is exactly the margin the probe suite records.
		return StreamResult{Timing: Timing{Headers: headers, Total: headers}}, c.httpError(resp)
	}

	// Headers are in, so the bound that matters from here is silence, not
	// arrival.
	guard.arm(c.idle, stallIdle)

	// Content only, which is what this entry point promises. The parser's sink
	// carries every channel now, and the iterator in streamiter.go consumes all of
	// them; RawStream keeps its narrow callback so the probe suite is untouched.
	var sink func(streamEvent)
	if onDelta != nil {
		sink = func(ev streamEvent) {
			if ev.kind == evContent {
				onDelta(ev.text)
			}
		}
	}

	// Below profile resolution, so no inline recovery: the caller owns the body.
	res, _, err := readStream(resp.Body, guard, &counters, streamBounds{idle: c.idle}, sink, c.redact, false, c.now, start)
	res.Timing.Headers = headers
	if err != nil {
		return res, c.explain(ctx, callCtx, guard, err)
	}
	return res, nil
}

// RawPost sends an already-built body to an arbitrary route and returns the
// decoded JSON response, the response headers and the status. Used for
// non-streaming completions, for embeddings, and by the probe suite when the
// interesting part of a response is its headers.
//
// The headers are returned because a gateway reports per-call spend and the
// real deployment name there and nowhere else.
func (c *Client) RawPost(ctx context.Context, route string, body []byte) (json.RawMessage, http.Header, error) {
	raw, hdr, _, err := c.rawPost(ctx, route, body)
	return raw, hdr, err
}

// rawPost is RawPost with the call's Timing. Separate so the exported signature
// the probe suite calls stays put while Chat and Embed get the figures.
func (c *Client) rawPost(ctx context.Context, route string, body []byte) (json.RawMessage, http.Header, Timing, error) {
	var t Timing
	callCtx, cancelCall := context.WithTimeout(ctx, c.cap)
	defer cancelCall()
	reqCtx, cancelReq := context.WithCancel(callCtx)
	defer cancelReq()

	req, err := c.newRequest(reqCtx, route, body)
	if err != nil {
		return nil, nil, t, err
	}
	req.Header.Set("Accept", "application/json")

	// Bounded by the WHOLE-CALL cap, not by c.header, and that difference is the
	// point.
	//
	// On a non-streaming route the endpoint withholds headers until the entire
	// answer is ready, so "time until headers" is not "time until the endpoint
	// starts talking" — it is the model's total latency. profiles.yaml records
	// single calls on mimo-v2.5-pro at 25-64 seconds against a 60s default header
	// bound, so arming this with c.header would abort a call the endpoint was about
	// to answer. The guard is still worth arming: when nothing arrives at all, the
	// failure says "no response headers" rather than a bare context deadline.
	//
	// (The transport's own ResponseHeaderTimeout, set in New to c.header + 30s,
	// remains the unnamed backstop it has always been on this path.)
	guard := newStallGuard(cancelReq, c.cap, stallHeaders)
	defer guard.stop()

	start := c.now()
	resp, err := c.http.Do(req)
	if err != nil {
		// Redaction first, then classification: the URL can carry a key in its
		// query string, and explain returns the error as given when no bound
		// fired.
		return nil, nil, t, c.explain(ctx, callCtx, guard, c.dialError(route, err))
	}
	defer resp.Body.Close()
	t.Headers = c.now().Sub(start)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		t.Total = t.Headers
		return nil, resp.Header, t, c.httpError(resp)
	}
	// Headers are in, so the bound that matters from here is silence on the body.
	guard.arm(c.idle, stallIdle)

	raw, err := io.ReadAll(resp.Body)
	t.Total = c.now().Sub(start)
	if err != nil {
		return nil, resp.Header, t, c.explain(ctx, callCtx, guard,
			fmt.Errorf("llmwire: reading response: %w", err))
	}
	return raw, resp.Header, t, nil
}

// newRequest builds a POST with the standard headers.
func (c *Client) newRequest(ctx context.Context, route string, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+route, bytes.NewReader(body))
	if err != nil {
		// The URL is in this error, so it goes through redaction: a base URL
		// can carry a key in its query string.
		return nil, fmt.Errorf("llmwire: building request for %s: %s", RedactURL(c.baseURL+route), Redact(err.Error()))
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	// Accept-Encoding is left unset on purpose so net/http keeps negotiating
	// and decompressing gzip transparently. Setting it by hand hands us a
	// compressed body to decode ourselves, mid-stream.
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	// After the caller's headers. Read per request rather than fixed at New,
	// because the id rotates after an idle gap.
	if c.session != nil {
		// The client string is the flag's whole meaning, so it wins even over a
		// User-Agent smuggled in through the generic Headers map — otherwise the
		// flag would be on and the emulation off, with nothing to say so.
		req.Header.Set("User-Agent", OpenCodeUserAgent)

		// The session pair is one value under two names; opencode never sends
		// them apart. A caller who pinned either one by hand keeps their value
		// — Headers is the escape hatch for what this package does not model —
		// and it is mirrored into the other, so the pair stays a pair. Only when
		// neither was set does the client's own session id go out.
		id := c.session.current()
		switch pinnedID, pinnedAff := req.Header.Get(HeaderSessionID), req.Header.Get(HeaderSessionAffinity); {
		case pinnedID != "":
			id = pinnedID
		case pinnedAff != "":
			id = pinnedAff
		}
		req.Header.Set(HeaderSessionID, id)
		req.Header.Set(HeaderSessionAffinity, id)
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	return req, nil
}

// httpError reads a bounded error body and decodes it. The body is bounded
// because an endpoint returning HTML for a 502 would otherwise pull an entire
// error page into memory and into the log.
func (c *Client) httpError(resp *http.Response) error {
	// Read past the cap by the key's length: a key straddling the cut would
	// leave its head in the message, and the value pass needs it whole. The
	// parser truncates to the cap afterwards, so the extra bytes never reach a
	// log.
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, int64(maxErrorBody+len(c.apiKey))))
	apiErr := parseAPIErrorWith(c.redact, resp.StatusCode, raw)
	if resp.StatusCode == http.StatusTooManyRequests {
		return newRateLimitError(apiErr, resp.Header)
	}
	return apiErr
}

// explain replaces the bare "context canceled" a cancelled request returns with
// the bound that actually gave up. Without it all three failures look identical
// in a log, which is the exact problem the split bounds exist to solve — so the
// classification, not the split, is the deliverable.
//
// Order matters: the guard is checked first because it cancels reqCtx directly,
// and a parent that is also done would otherwise mask it.
func (c *Client) explain(parent, call context.Context, guard *stallGuard, err error) error {
	if reason, bound := guard.firedReason(); reason != "" {
		if reason == stallHeaders {
			return fmt.Errorf("llmwire: %w within %s", ErrNoResponseHeaders, bound)
		}
		return fmt.Errorf("llmwire: %w for %s", ErrStreamIdle, bound)
	}
	if call.Err() != nil && parent.Err() == nil {
		// Phrased around the sentinel so the message still reads "exceeded
		// the 15m0s call cap" and errors.Is(err, ErrCallCap) holds.
		return fmt.Errorf("llmwire: %w", &callCapError{cap: c.cap})
	}
	return c.scrub(err)
}

// dialError is a transport failure phrased without the URL net/http put in it.
//
// Every failure out of http.Client.Do is a *url.Error whose Error() prints the
// FULL request URL, query string included, and some deployments carry their
// key there. Wrapping that error with a redacted prefix does not help: the
// inner text still prints. So the url.Error is taken apart — its operation,
// the URL through RedactURL, and its cause wrapped with %w so errors.Is on
// context.DeadlineExceeded or a *net.OpError still holds. What a caller loses
// is errors.As(*url.Error), which nobody should be reading the URL out of
// anyway, and with it the net.Error view of a bare context.Canceled cause.
func (c *Client) dialError(route string, err error) error {
	var uerr *url.Error
	if errors.As(err, &uerr) {
		return fmt.Errorf("llmwire: %s %s: %w", uerr.Op, RedactURL(c.baseURL+route), uerr.Err)
	}
	return fmt.Errorf("llmwire: %s: %w", RedactURL(c.baseURL+route), err)
}

// redactKey strips the configured key BY VALUE. Redact knows credentials by
// shape, and the shapes it knows are the ones the documented endpoints issue;
// a token-plan host or a self-hosted gateway can issue any string, and an
// upstream that echoes the Authorization header would put that string into
// every log line that touches the failure. The client is the one party that
// knows the key, so it is the one that can remove it whatever it looks like.
func (c *Client) redactKey(s string) string {
	if c.apiKey == "" || s == "" {
		return s
	}
	return strings.ReplaceAll(s, c.apiKey, "[REDACTED]")
}

// redact is the redactor this client hands the parsers: the key by value
// first, then everything Redact knows by shape. Value first, because the
// shape pass can eat the middle of a key whose value happens to contain an
// sk-/tp- run, and the value pass would then find nothing to strip.
func (c *Client) redact(s string) string {
	return Redact(c.redactKey(s))
}

// scrub is the last line: an error built somewhere that had no redactor —
// a transport failure quoting a response, a decode error — whose text carries
// the key is wrapped so the text is clean and the chain is kept.
func (c *Client) scrub(err error) error {
	if err == nil || c.apiKey == "" {
		return err
	}
	if text := err.Error(); strings.Contains(text, c.apiKey) {
		return &scrubbedError{msg: c.redactKey(text), err: err}
	}
	return err
}

// scrubbedError is an error whose text was redacted after the fact. It keeps
// the original as its cause so errors.Is and errors.As see through it.
type scrubbedError struct {
	msg string
	err error
}

func (e *scrubbedError) Error() string { return e.msg }
func (e *scrubbedError) Unwrap() error { return e.err }

// Registry exposes the profiles this client validates against, so a caller can
// ask what a model supports without making a request.
func (c *Client) Registry() *Registry { return c.registry }

// callCapError names the cap that elapsed and unwraps to ErrCallCap.
type callCapError struct{ cap time.Duration }

func (e *callCapError) Error() string { return fmt.Sprintf("exceeded the %s call cap", e.cap) }
func (e *callCapError) Unwrap() error { return ErrCallCap }
