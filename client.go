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
	"fmt"
	"io"
	"net/http"
	"os"
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
	Now func() time.Time

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

// FromEnv builds a Client for one model, taking BaseURL and APIKey from the
// environment variables that model's profile names (base_url_env and
// api_key_env in profiles.yaml). A BaseURL or APIKey already set in cfg wins
// over the environment. The profile carries the names; this is what reads them,
// so the wiring does not have to be repeated in every application.
//
// A named variable that is unset or empty is a MissingEnvError, never a
// fallback. A profile that names no api_key_env sends no key at all, which is
// the self-hosted case.
func FromEnv(model string, cfg Config) (*Client, error) {
	reg := cfg.Registry
	if reg == nil {
		reg = Default()
	}
	p, err := reg.Lookup(model)
	if err != nil {
		return nil, err
	}
	// Trimmed: a whitespace-only value (a stray `export X= ` in a .env) would
	// otherwise build a client that fails later with an opaque dial or 401
	// instead of the named error here.
	if cfg.BaseURL == "" {
		v := strings.TrimSpace(os.Getenv(p.BaseURLEnv))
		if v == "" {
			return nil, &MissingEnvError{Model: model, Var: p.BaseURLEnv, Field: "base_url_env"}
		}
		cfg.BaseURL = v
	}
	if cfg.APIKey == "" && p.APIKeyEnv != "" {
		v := strings.TrimSpace(os.Getenv(p.APIKeyEnv))
		if v == "" {
			return nil, &MissingEnvError{Model: model, Var: p.APIKeyEnv, Field: "api_key_env"}
		}
		cfg.APIKey = v
	}
	return New(cfg), nil
}

// MissingEnvError names the variable a profile expects and the profile field
// that expects it, so the fix is one export away rather than a search.
type MissingEnvError struct {
	Model string
	// Var is the variable name, or empty when the profile names none.
	Var   string
	Field string
}

func (e *MissingEnvError) Error() string {
	if e.Var == "" {
		return fmt.Sprintf("llmwire: model %q has no %s in its profile", e.Model, e.Field)
	}
	return fmt.Sprintf("llmwire: model %q needs %s (%s) set in the environment", e.Model, e.Var, e.Field)
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

	resp, err := c.http.Do(req)
	if err != nil {
		return StreamResult{}, c.explain(ctx, callCtx, guard, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return StreamResult{}, c.httpError(resp)
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

	res, err := readStream(resp.Body, guard, &counters, c.idle, sink)
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
	callCtx, cancelCall := context.WithTimeout(ctx, c.cap)
	defer cancelCall()
	reqCtx, cancelReq := context.WithCancel(callCtx)
	defer cancelReq()

	req, err := c.newRequest(reqCtx, route, body)
	if err != nil {
		return nil, nil, err
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

	resp, err := c.http.Do(req)
	if err != nil {
		// Redaction first, then classification: the URL can carry a key in its
		// query string, and explain returns the original error untouched when no
		// bound fired.
		return nil, nil, c.explain(ctx, callCtx, guard,
			fmt.Errorf("llmwire: %s: %w", RedactURL(c.baseURL+route), err))
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, resp.Header, c.httpError(resp)
	}
	// Headers are in, so the bound that matters from here is silence on the body.
	guard.arm(c.idle, stallIdle)

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.Header, c.explain(ctx, callCtx, guard,
			fmt.Errorf("llmwire: reading response: %w", err))
	}
	return raw, resp.Header, nil
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
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
	apiErr := parseAPIError(resp.StatusCode, raw)
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
			return fmt.Errorf("llmwire: %s within %s", reason, bound)
		}
		return fmt.Errorf("llmwire: %s for %s", reason, bound)
	}
	if call.Err() != nil && parent.Err() == nil {
		return fmt.Errorf("llmwire: exceeded the %s call cap", c.cap)
	}
	return err
}

// Registry exposes the profiles this client validates against, so a caller can
// ask what a model supports without making a request.
func (c *Client) Registry() *Registry { return c.registry }
