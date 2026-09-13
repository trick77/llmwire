package llmwire

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// Harness for the live probes.
//
// These tests call real, paid endpoints. They exist because published model
// catalogues disagree with each other and with the endpoints themselves, so the
// only trustworthy source for a profile bit is a measurement.
//
// Everything here is gated on LLMWIRE_EVAL=1 and skips otherwise, which is what
// keeps CI from ever calling out. The gate is an env check with t.Skip rather
// than a build tag on purpose: a build tag hides this file from gofmt, go vet
// and the compiler, so it would rot unnoticed between the rare runs.
//
// SECRETS. Keys are read from the environment only. There is no literal, no
// fixture and no fallback: a missing key skips with a named reason. Nothing here
// prints a key, a full URL, or a raw request body — observations go through
// Redact and URLs through RedactURL, because the findings note these probes
// produce is a committed artefact.

const evalEnvVar = "LLMWIRE_EVAL"

// --- spend guard --------------------------------------------------------------

// The probes deliberately provoke error paths and unknown behaviour, so an
// unbounded or looping run is a live cost risk. Four independent limits, all
// fail-closed, all raisable but none disableable.
const (
	// defaultMaxCalls trips independently of cost. It is the one that catches a
	// retry loop or a test that never terminates, where each call is cheap but
	// the count is not.
	defaultMaxCalls = 60
	// defaultMaxUSD is a circuit breaker, not a budget. A full run at the
	// per-call caps below costs well under a cent.
	defaultMaxUSD = 0.50
	// defaultSuiteDeadline bounds the whole run, so an abort still reaches the
	// spend summary instead of being killed by the outer go test timeout.
	defaultSuiteDeadline = 10 * time.Minute
	// probeMaxTokens is the per-call completion cap. Every probe asks a
	// one-sentence question whose ANSWER does not matter — what matters is
	// whether the endpoint accepted the request — so this only has to be large
	// enough that a rejection is distinguishable from a truncation.
	probeMaxTokens = 64
)

// evalRate is a conservative upper-bound price, used ONLY by the spend guard.
//
// It is deliberately not a pricing table and must never become one. Published
// rates for these models conflict by 2x or more between models.dev and the
// vendors' own pages, and the Token Plan hosts bill subscription credits rather
// than dollars at all. So each figure below is the HIGHEST published rate found,
// which makes the guard over-estimate spend and trip early. Over-estimating is
// the correct failure direction for a circuit breaker; the real accounting table
// arrives with verified rates and provenance.
type evalRate struct{ inputPerM, outputPerM float64 }

var evalRates = map[string]evalRate{
	// docs.z.ai publishes 0.15/0.50; models.dev has 0.075/0.25. Higher wins.
	"glm-5.3-flash": {0.15, 0.50},
	// ismimodown vendors 1.00/3.00; models.dev has 0.435/0.87. Higher wins.
	// Reached over a Token Plan host this is credits, not dollars — the USD
	// figure is a proxy bound, which is all a circuit breaker needs.
	"mimo-v2.5-pro": {1.00, 3.00},
	"mimo-v2.5":     {0.40, 2.00},
}

// meter accumulates spend across the whole run. Shared by every probe, because
// a per-test budget would let N tests each spend the maximum.
type meter struct {
	mu       sync.Mutex
	calls    int
	usd      float64
	tokens   int64
	perModel map[string]int64
	deadline time.Time

	maxCalls int
	maxUSD   float64
}

// Built in init, AFTER .env is loaded: a package-level initialiser would read
// the budget variables before the file could supply them, and a ceiling
// written there would silently run under the default.
var theMeter *meter

func init() {
	if os.Getenv(evalEnvVar) == "1" {
		loadDotEnv(".env")
	}
	theMeter = newMeter()
}

func newMeter() *meter {
	return &meter{
		perModel: map[string]int64{},
		deadline: time.Now().Add(envDuration("LLMWIRE_EVAL_DEADLINE", defaultSuiteDeadline)),
		maxCalls: int(envInt("LLMWIRE_EVAL_MAX_CALLS", defaultMaxCalls)),
		maxUSD:   envFloat("LLMWIRE_EVAL_MAX_USD", defaultMaxUSD),
	}
}

// reserve is called BEFORE every request. Checking up front is what bounds the
// overshoot to a single call: a check that only ran afterwards would let the
// last call be arbitrarily large.
//
// It fails the calling test rather than returning an error, because there is no
// sensible way for a probe to continue once the run is over budget.
func (m *meter) reserve(t *testing.T, model string) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()

	if time.Now().After(m.deadline) {
		t.Fatalf("eval suite deadline exceeded after %d calls — aborting", m.calls)
	}
	if m.calls >= m.maxCalls {
		t.Fatalf("eval call ceiling reached (%d) — aborting before calling %s", m.maxCalls, model)
	}
	if m.usd >= m.maxUSD {
		t.Fatalf("eval spend ceiling reached ($%.4f of $%.2f) — aborting before calling %s",
			m.usd, m.maxUSD, model)
	}
	m.calls++
}

// record adds one call's usage. A model with no rate is charged the WHOLE
// remaining budget, never zero: an unpriced model must not be able to spend
// unmetered, and tripping early on an unknown model is the safe direction.
func (m *meter) record(model string, u Usage) {
	m.mu.Lock()
	defer m.mu.Unlock()

	in, out := valueOr(u.Input.Total, 0), valueOr(u.Output.Total, 0)
	m.tokens += in + out
	m.perModel[model] += in + out

	rate, known := evalRates[model]
	if !known {
		m.usd = m.maxUSD
		return
	}
	m.usd += float64(in)/1e6*rate.inputPerM + float64(out)/1e6*rate.outputPerM
}

func (m *meter) summary() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var b strings.Builder
	fmt.Fprintf(&b, "eval spend: %d calls, %d tokens, ~$%.5f (ceiling $%.2f)",
		m.calls, m.tokens, m.usd, m.maxUSD)
	for model, tok := range m.perModel {
		fmt.Fprintf(&b, "\n  %-16s %d tokens", model, tok)
	}
	return b.String()
}

// TestMain prints the spend summary after the run, so the cost of a full pass is
// a known number rather than a surprise on an invoice.
func TestMain(m *testing.M) {
	code := m.Run()
	if os.Getenv(evalEnvVar) == "1" && theMeter.calls > 0 {
		fmt.Fprintln(os.Stderr, theMeter.summary())
	}
	os.Exit(code)
}

// loadDotEnv exports KEY=value lines from a gitignored .env into the process,
// only under the eval gate and only for names not already set, so a value in
// the real environment always wins. It is the file .env.example describes;
// sourcing it by hand before every run is what people get wrong. A missing file
// is nothing: the probes then skip with the usual named reason. CI has no .env.
//
// Accepts the common dotenv spellings: an "export " prefix and a value wrapped
// in matching quotes. Anything else is taken literally, which for a key or a
// URL means a 401 rather than a named skip, so the file is kept bare.
func loadDotEnv(path string) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "export "))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		if n := len(v); n >= 2 && (v[0] == '"' || v[0] == '\'') && v[n-1] == v[0] {
			v = v[1 : n-1]
		}
		if !ok || v == "" {
			continue
		}
		// Empty counts as unset, matching how endpoint.client reads it: an
		// exported-but-empty name in a shell profile must not shadow the file.
		if os.Getenv(k) == "" {
			os.Setenv(k, v)
		}
	}
}

// --- gating -------------------------------------------------------------------

// evalGate skips unless the run was asked for explicitly.
func evalGate(t *testing.T) {
	t.Helper()
	if os.Getenv(evalEnvVar) != "1" {
		t.Skipf("live probe: set %s=1 to run (calls a real, paid endpoint)", evalEnvVar)
	}
}

// endpoint names one upstream under test: the provider as profiles.yaml names
// it. The host is the provider's shipped base_url, the same contract FromEnv
// enforces: a LLMWIRE_<PROVIDER>_BASE_URL beside a shipped host fails the run
// rather than silently measuring some other host. The key comes from the
// environment only and is never defaulted to a literal.
type endpoint struct {
	name       string
	baseURLVar string
	apiKeyVar  string
}

var (
	mimoEndpoint = endpoint{"mimo", "LLMWIRE_MIMO_BASE_URL", "LLMWIRE_MIMO_API_KEY"}
	zaiEndpoint  = endpoint{"zai", "LLMWIRE_ZAI_BASE_URL", "LLMWIRE_ZAI_API_KEY"}
)

// client builds a Client for an endpoint, or skips with a named reason.
//
// A missing key SKIPS rather than failing: an incomplete local setup should not
// look like a broken library.
func (e endpoint) client(t *testing.T) *Client {
	t.Helper()
	evalGate(t)
	base, key := strings.TrimSpace(os.Getenv(e.baseURLVar)), strings.TrimSpace(os.Getenv(e.apiKeyVar))
	pv, err := Default().Provider(e.name)
	if err != nil {
		t.Fatal(err)
	}
	switch {
	case pv.BaseURL != "" && base != "":
		t.Fatalf("live probe: %s is set but the host for %s is the library's; delete the variable, "+
			"or the findings describe a host the profile does not name", e.baseURLVar, e.name)
	case pv.BaseURL != "":
		base = pv.BaseURL
	case base == "":
		t.Skipf("live probe: %s is not set and the provider ships no host", e.baseURLVar)
	}
	if key == "" {
		t.Skipf("live probe: %s is not set", e.apiKeyVar)
	}
	t.Logf("probing %s at %s", e.name, RedactURL(base))
	return New(Config{
		BaseURL: base,
		APIKey:  key,
		// Deliberately shorter than production: a probe that hangs should fail
		// the run quickly, and none of these questions needs a long answer.
		HeaderTimeout: 60 * time.Second,
		IdleTimeout:   90 * time.Second,
		CallTimeout:   3 * time.Minute,
	})
}

// --- request helpers -----------------------------------------------------------

// body is a chat-completions request under construction. Probes build raw maps
// rather than a typed request because the whole point is to send shapes a
// validated type would refuse.
type body map[string]any

// probeBody returns a minimal streaming request for a model. Callers add or
// override fields to isolate the one parameter under test.
func probeBody(model, prompt string) body {
	return body{
		"model":    model,
		"messages": []any{map[string]any{"role": "user", "content": prompt}},
		"stream":   true,
	}
}

// stream sends a request and returns the result, metering the call. The model
// is read back out of the body so the meter always attributes spend to what was
// actually sent.
func stream(t *testing.T, c *Client, b body) (StreamResult, error) {
	t.Helper()
	model, _ := b["model"].(string)
	theMeter.reserve(t, model)

	raw, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal probe body: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	res, err := c.RawStream(ctx, raw, nil)
	theMeter.record(model, res.Usage)
	return res, err
}

// accepted reports whether the endpoint took the request, and renders the
// rejection for a findings note when it did not.
//
// A truncation (finish_reason "length") counts as ACCEPTED: the parameter under
// test was honoured, and the answer being cut off is this harness's own tiny
// token cap doing its job.
func accepted(res StreamResult, err error) (bool, string) {
	if err == nil {
		return true, ""
	}
	var apiErr *APIError
	if ok := asAPIError(err, &apiErr); ok {
		return false, fmt.Sprintf("status=%d code=%s param=%s message=%s",
			apiErr.StatusCode, apiErr.Code, apiErr.Param, Redact(apiErr.Message))
	}
	return false, Redact(err.Error())
}

// asAPIError was a hand-rolled errors.As, and its second branch existed only
// because *RateLimitError embedded an *APIError without unwrapping to one. It
// does now, so this is errors.As and nothing else — kept as a named function
// purely so the probes read the same as before.
func asAPIError(err error, target **APIError) bool {
	return errors.As(err, target)
}

// --- env helpers ---------------------------------------------------------------

func envInt(name string, def int64) int64 {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func envFloat(name string, def float64) float64 {
	if v := os.Getenv(name); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			return f
		}
	}
	return def
}

func envDuration(name string, def time.Duration) time.Duration {
	if v := os.Getenv(name); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}
	return def
}

func valueOr(p *int64, def int64) int64 {
	if p == nil {
		return def
	}
	return *p
}
