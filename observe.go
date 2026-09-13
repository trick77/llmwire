package llmwire

import (
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"
)

// Every call ends in one log line and one stats update, from the same
// summary, so the two never disagree about what happened.
//
// The line is Debug on success, Warn on failure and on a named anomaly. Keys
// are flat and durations are integer milliseconds: a JSON handler prints a
// time.Duration as nanoseconds and a text handler as "1.2s", which would be
// two different fields in two deployments of the same binary. Never the
// prompt, the answer or the key: text lengths and the key's variable name.

// callSummary is what one finished call amounts to, whichever route ran it.
type callSummary struct {
	kind  string // chat, chat_stream, embed
	model string
	// Request side, from the plan (what went on the wire), nil for embed.
	plan *wirePlan
	// inputs is the embed input count.
	inputs int

	content, reasoning int // lengths in bytes
	toolCalls          int
	finishReason       string
	usage              Usage
	timing             Timing
	warnings           []Warning
	err                error
	// closed is a caller-initiated Stream.Close: not a failure, not a success.
	closed bool
}

// finish logs the call and folds it into the stats.
func (c *Client) finish(s callSummary) {
	c.stats.record(s)
	c.logCall(s)
}

func (c *Client) logCall(s callSummary) {
	attrs := make([]any, 0, 40)
	attrs = append(attrs, "kind", s.kind, "model", s.model)
	if pl := s.plan; pl != nil {
		req := pl.req
		attrs = append(attrs, "messages", len(req.Messages), "tools", len(req.Tools))
		if req.MaxTokens != nil {
			attrs = append(attrs, "max_tokens", *req.MaxTokens, "cap_param", pl.capParam)
		}
		if req.Temperature != nil {
			attrs = append(attrs, "temperature", *req.Temperature)
		}
		if req.TopP != nil {
			attrs = append(attrs, "top_p", *req.TopP)
		}
		if r := reasoningLabel(req.Reasoning); r != "" {
			attrs = append(attrs, "reasoning", r)
		}
		if len(req.ExtraBody) > 0 {
			keys := make([]string, 0, len(req.ExtraBody))
			for k := range req.ExtraBody {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			attrs = append(attrs, "extra_body_keys", strings.Join(keys, ","))
		}
	}
	if s.kind == "embed" {
		attrs = append(attrs, "inputs", s.inputs)
	}
	if s.kind != "embed" {
		attrs = append(attrs, "finish_reason", s.finishReason,
			"content_chars", s.content, "reasoning_chars", s.reasoning, "tool_calls", s.toolCalls)
	}
	u := s.usage
	attrs = append(attrs,
		"usage_reported", u.Reported(),
		"input_tokens", Tokens(u.Input.Total),
		"cache_read_tokens", Tokens(u.Input.CacheRead),
		"cache_write_tokens", Tokens(u.Input.CacheWrite),
		"output_tokens", Tokens(u.Output.Total),
		"reasoning_tokens", Tokens(u.Output.Reasoning),
		"cost_usd", fmt.Sprintf("%.6f", float64(u.Cost.NanoUSD)/1e9),
		"cost_provenance", u.Cost.Provenance.String(),
	)
	if u.Cost.Window != "" {
		attrs = append(attrs, "cost_window", u.Cost.Window)
	}
	if u.Cost.Tier != "" {
		attrs = append(attrs, "cost_tier", u.Cost.Tier)
	}
	t := s.timing
	attrs = append(attrs, "headers_ms", t.Headers.Milliseconds(),
		"first_data_ms", t.FirstData.Milliseconds(), "total_ms", t.Total.Milliseconds())
	if tps := tokensPerSecond(s); tps > 0 {
		attrs = append(attrs, "output_tokens_per_s", fmt.Sprintf("%.1f", tps))
	}
	if n := len(s.warnings); n > 0 {
		names := make([]string, n)
		for i, w := range s.warnings {
			names[i] = string(w.Kind) + ":" + w.Feature
		}
		attrs = append(attrs, "warnings", strings.Join(names, ","))
	}

	switch {
	case s.closed:
		c.log.Debug("llmwire call closed by caller", attrs...)
		return
	case s.err != nil:
		attrs = append(attrs, "error", c.redact(s.err.Error()))
		var rl *RateLimitError
		var api *APIError
		switch {
		case errors.As(s.err, &rl):
			attrs = append(attrs, "status", rl.StatusCode, "retry_after_ms", rl.RetryAfter.Milliseconds())
		case errors.As(s.err, &api):
			attrs = append(attrs, "status", api.StatusCode)
		}
		c.log.Warn("llmwire call failed", attrs...)
		return
	}
	// Anomalies: a call that worked on the wire and is still worth a look.
	// Each is a Warn with a name, because "the model gave nothing" has three
	// different causes and a reader should not have to diff attribute sets.
	switch {
	case s.finishReason == "length":
		c.log.Warn("llmwire answer truncated by the output cap", attrs...)
	case s.kind != "embed" && s.content == 0 && s.toolCalls == 0 && s.finishReason == "stop":
		c.log.Warn("llmwire answer empty", attrs...)
	case !u.Reported():
		c.log.Warn("llmwire call unaccounted, no usage reported", attrs...)
	default:
		c.log.Debug("llmwire call", attrs...)
	}
}

func reasoningLabel(r ReasoningRequest) string {
	switch v := r.(type) {
	case reasoningOff:
		return "off"
	case reasoningEffort:
		return v.level
	case reasoningBudget:
		return fmt.Sprintf("budget:%d", v.tokens)
	}
	return ""
}

// tokensPerSecond is output tokens over the time the model spent producing
// them: after the first data frame on a stream, the whole wait for a
// non-streaming call, where the headers arrive with the answer.
func tokensPerSecond(s callSummary) float64 {
	out := Tokens(s.usage.Output.Total)
	if out == 0 {
		return 0
	}
	var window time.Duration
	if s.timing.FirstData > 0 {
		window = s.timing.Total - s.timing.FirstData
	} else {
		window = s.timing.Total
	}
	if window <= 0 {
		return 0
	}
	return float64(out) / window.Seconds()
}

// ModelStats is what one model has done since the client was built.
type ModelStats struct {
	// Calls is every call that reached finish, Errors those that failed,
	// Closed those the caller stopped reading.
	Calls, Errors, Closed int64
	// Token lanes, summed over calls that reported them. UnreportedCalls
	// counts the ones that did not, so a low total is read beside it.
	InputTokens, CacheReadTokens, CacheWriteTokens, OutputTokens, ReasoningTokens int64
	UnreportedCalls                                                               int64
	// CostNanoUSD sums priced calls only; UnpricedCalls says how many are
	// missing from it. 0 with UnpricedCalls > 0 is unknown, never free.
	CostNanoUSD   int64
	UnpricedCalls int64
}

// Stats is the client's counters, per model, as of the call. A copy: mutate
// it freely.
type Stats struct {
	Models map[string]ModelStats
}

type stats struct {
	mu     sync.Mutex
	models map[string]*ModelStats
}

func (st *stats) record(s callSummary) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.models == nil {
		st.models = make(map[string]*ModelStats)
	}
	m := st.models[s.model]
	if m == nil {
		m = &ModelStats{}
		st.models[s.model] = m
	}
	m.Calls++
	switch {
	case s.closed:
		m.Closed++
	case s.err != nil:
		m.Errors++
	}
	u := s.usage
	if !u.Reported() {
		m.UnreportedCalls++
	}
	m.InputTokens += Tokens(u.Input.Total)
	m.CacheReadTokens += Tokens(u.Input.CacheRead)
	m.CacheWriteTokens += Tokens(u.Input.CacheWrite)
	m.OutputTokens += Tokens(u.Output.Total)
	m.ReasoningTokens += Tokens(u.Output.Reasoning)
	if u.Cost.Provenance == Unpriced {
		m.UnpricedCalls++
	} else {
		m.CostNanoUSD += u.Cost.NanoUSD
	}
}

// Stats returns the per-model counters accumulated since New. Every call
// that reached its end is in here, failures included; a stream the caller
// closed early counts as Closed, with whatever usage had arrived by then.
func (c *Client) Stats() Stats {
	c.stats.mu.Lock()
	defer c.stats.mu.Unlock()
	out := Stats{Models: make(map[string]ModelStats, len(c.stats.models))}
	for k, v := range c.stats.models {
		out.Models[k] = *v
	}
	return out
}

// LogValue renders the stats as one line of flat attributes per model, for a
// shutdown or periodic summary from the application's own logger.
func (s Stats) LogValue() slog.Value {
	names := make([]string, 0, len(s.Models))
	for k := range s.Models {
		names = append(names, k)
	}
	sort.Strings(names)
	attrs := make([]slog.Attr, 0, len(names))
	for _, name := range names {
		m := s.Models[name]
		attrs = append(attrs, slog.Group(name,
			"calls", m.Calls, "errors", m.Errors, "closed", m.Closed,
			"input_tokens", m.InputTokens, "cache_read_tokens", m.CacheReadTokens,
			"output_tokens", m.OutputTokens, "reasoning_tokens", m.ReasoningTokens,
			"cost_usd", fmt.Sprintf("%.6f", float64(m.CostNanoUSD)/1e9),
			"unpriced_calls", m.UnpricedCalls, "unreported_calls", m.UnreportedCalls,
		))
	}
	return slog.GroupValue(attrs...)
}
