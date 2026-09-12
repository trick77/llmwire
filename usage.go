package llmwire

import "encoding/json"

// Token accounting.
//
// Every count is a POINTER. "The endpoint did not report this" and "the endpoint
// reported zero" are different facts, and collapsing them into a plain int makes
// a missing field indistinguishable from a genuinely free call — which is how a
// cost column quietly fills with zeros that nobody can explain later.
//
// The lane breakdown matters as much as the totals, because of two containment
// rules that OpenAI-shaped usage objects follow and that are easy to get
// backwards, expensively:
//
//   - prompt_tokens INCLUDES cached_tokens. Cached tokens are billed at a
//     fraction of the input rate, so they must be SUBTRACTED out of the input
//     lane, never added on top.
//   - completion_tokens INCLUDES reasoning_tokens. Reasoning is never a separate
//     billing lane; pricing it again double-counts every call a reasoning model
//     makes, which is all of them on the models this library targets.

// Usage is one call's token accounting, normalised across endpoints.
type Usage struct {
	Input  InputTokens  `json:"input"`
	Output OutputTokens `json:"output"`
	// Cost is what this call cost, priced at the call site with the model that
	// actually ran. Its zero value means Unpriced, which is why it is a field
	// rather than a second return: parseUsage and RawStream sit BELOW profile
	// resolution, so a usage object from either already reads as "not priced"
	// with nobody having to remember to say so.
	Cost Cost `json:"cost"`
	// Raw is the endpoint's own usage object, verbatim and undecoded. Kept
	// because these endpoints carry fields nobody has modelled yet (image and
	// video token counts, web-search request counts), and because it is the
	// only way to settle whether a zero is the endpoint's answer or a field
	// name this package does not know about.
	Raw json.RawMessage `json:"raw,omitempty"`

	// reported is whether the wire object carried any accounting at all, by
	// the same rule the stream parser uses to pick which usage chunk to keep
	// (wireUsage.reported). Held here rather than re-derived from the lanes
	// because the two rules differ on one shape: a bare {"total_tokens":N}
	// is reported on the wire and yields no lane, since this type has no
	// total lane. Unexported so a caller-built Usage{} reads as not reported.
	reported bool
}

// Reported says the endpoint sent a usage object with something in it, so a
// zero lane is the endpoint's answer rather than silence. A malformed object
// keeps its bytes in Raw and is NOT reported. Every consumer that counts
// "how many of my calls were accounted for" needs exactly this and was
// re-deriving it from the lane pointers.
func (u Usage) Reported() bool { return u.reported }

// Total is prompt plus completion — the OpenAI-compatible total_tokens, which
// parseUsage does not keep as its own lane because it is by definition the sum
// of two lanes that are. ok is false when neither side was reported; a side
// that is absent counts as zero, so a completion-only object still totals.
func (u Usage) Total() (total int64, ok bool) {
	if u.Input.Total == nil && u.Output.Total == nil {
		return 0, false
	}
	if u.Input.Total != nil {
		total += *u.Input.Total
	}
	if u.Output.Total != nil {
		total += *u.Output.Total
	}
	return total, true
}

// InputTokens is the prompt side. Total is what the endpoint billed as prompt
// tokens; the rest break that total down where the endpoint says.
type InputTokens struct {
	Total *int64 `json:"total,omitempty"`
	// NoCache is Total minus CacheRead — the tokens billed at the full input
	// rate. Derived, not reported: no endpoint sends it.
	NoCache *int64 `json:"no_cache,omitempty"`
	// CacheRead is prompt_tokens_details.cached_tokens: prompt tokens the
	// endpoint served from its own cache, billed at a reduced rate.
	CacheRead *int64 `json:"cache_read,omitempty"`
	// CacheWrite is tokens written INTO the cache. Free on every model this
	// library currently targets, and billed at a premium on newer OpenAI
	// models, which is why it is a lane of its own rather than folded into
	// NoCache.
	CacheWrite *int64 `json:"cache_write,omitempty"`
}

// OutputTokens is the completion side.
type OutputTokens struct {
	Total *int64 `json:"total,omitempty"`
	// Text is Total minus Reasoning — the part the caller actually received.
	// Derived when the endpoint reports reasoning but not text, which is the
	// usual case.
	Text *int64 `json:"text,omitempty"`
	// Reasoning is completion_tokens_details.reasoning_tokens: tokens spent
	// thinking, already counted inside Total.
	Reasoning *int64 `json:"reasoning,omitempty"`
}

// wireUsage mirrors the OpenAI-compatible `usage` object.
//
// Every field is a pointer and every nested object is optional, deliberately:
// endpoints vary in how much of the breakdown they report, several send
// "usage": null on a non-final chunk, and a missing field must decode as "not
// reported" rather than failing the whole response.
type wireUsage struct {
	PromptTokens        *int64 `json:"prompt_tokens"`
	CompletionTokens    *int64 `json:"completion_tokens"`
	TotalTokens         *int64 `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens     *int64 `json:"cached_tokens"`
		CacheWriteTokens *int64 `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens *int64 `json:"reasoning_tokens"`
		TextTokens      *int64 `json:"text_tokens"`
	} `json:"completion_tokens_details"`
}

// reported says the endpoint sent a usage object with something in it, so its
// zeros are answers rather than silence.
//
// This is load-bearing during streaming: the chunk carrying finish_reason also
// carries "usage": null on several endpoints, and the real usage arrives in a
// later chunk. Keeping "the last usage SEEN" would therefore discard the
// accounting entirely; the parser keeps the last usage that satisfies this.
func (u wireUsage) reported() bool {
	return u.PromptTokens != nil || u.CompletionTokens != nil || u.TotalTokens != nil ||
		u.PromptTokensDetails != nil || u.CompletionTokensDetails != nil
}

// parseUsage decodes a raw usage object. An absent, null or malformed one yields
// the zero Usage — "not reported" — and never an error: a usage field this
// package cannot read must not fail a completion the caller already received.
func parseUsage(raw json.RawMessage) Usage {
	var w wireUsage
	if len(raw) == 0 {
		return Usage{}
	}
	if err := json.Unmarshal(raw, &w); err != nil {
		// Keep the bytes anyway. They are the evidence for whatever shape this
		// endpoint actually sent.
		return Usage{Raw: raw}
	}

	u := Usage{Raw: raw, reported: w.reported()}
	u.Input.Total = w.PromptTokens
	u.Output.Total = w.CompletionTokens

	if d := w.PromptTokensDetails; d != nil {
		u.Input.CacheRead = d.CachedTokens
		u.Input.CacheWrite = d.CacheWriteTokens
	}
	if d := w.CompletionTokensDetails; d != nil {
		u.Output.Reasoning = d.ReasoningTokens
		u.Output.Text = d.TextTokens
	}

	// Derive the lanes nobody reports. Clamped rather than trusted: both
	// operands arrive from the wire, and a cached count exceeding the prompt
	// count would otherwise produce a negative input lane, which at pricing
	// time CREDITS the caller for tokens they actually spent.
	u.Input.NoCache = subClamped(u.Input.Total, u.Input.CacheRead)
	if u.Output.Text == nil {
		u.Output.Text = subClamped(u.Output.Total, u.Output.Reasoning)
	}
	return u
}

// subClamped returns total-part, floored at zero. It returns nil when total is
// unreported, because a lane derived from an unknown total is itself unknown —
// not zero. A nil part means "nothing to subtract", so the answer is total.
func subClamped(total, part *int64) *int64 {
	if total == nil {
		return nil
	}
	v := *total
	if part != nil {
		v -= *part
		if v < 0 {
			v = 0
		}
	}
	return &v
}
