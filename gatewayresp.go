package llmwire

import (
	"fmt"
	"net/http"
	"strings"
)

// What a gateway says about a call in its response headers.
//
// The body of a proxied completion is the vendor's; everything the proxy itself
// knows lands in headers: which deployment really ran, what the call cost, what
// the key has spent so far, and the id under which the proxy logged it. Chat
// drops the header map once the cost is read, so this block carries the rest to
// the caller. Direct routes leave it empty.

// Header names LiteLLM sets on every proxied response.
const (
	litellmCallIDHeader   = "x-litellm-call-id"
	litellmModelHeader    = "x-litellm-model-name"
	litellmKeySpendHeader = "x-litellm-key-spend"
	litellmVersionHeader  = "x-litellm-version"
)

// The per-component split of x-litellm-response-cost, documented 2026-08-15
// (litellm-docs #898), set by cost_headers() in
// litellm/proxy/common_request_processing.py. Non-streaming only: a header is
// written before the stream it would price is consumed (LiteLLM #12689). A
// priced call sends the cache, reasoning and tool lanes only when nonzero; a
// hit on LiteLLM's own response cache, and a non-inference call priced at
// zero, send input, output and tool usage as 0.0.
const (
	litellmCostInputHeader      = "x-litellm-response-cost-input"
	litellmCostCacheReadHeader  = "x-litellm-response-cost-cache-read"
	litellmCostCacheWriteHeader = "x-litellm-response-cost-cache-creation"
	litellmCostOutputHeader     = "x-litellm-response-cost-output"
	litellmCostReasoningHeader  = "x-litellm-response-cost-reasoning"
	litellmCostToolUsageHeader  = "x-litellm-response-cost-tool-usage"
	// Sent by cost_headers(), absent from the docs.
	litellmCostOriginalHeader      = "x-litellm-response-cost-original"
	litellmCostDiscountHeader      = "x-litellm-response-cost-discount-amount"
	litellmCostMarginHeader        = "x-litellm-response-cost-margin-amount"
	litellmCostMarginPercentHeader = "x-litellm-response-cost-margin-percent"
)

// Gateway is the proxy's own account of one call, read from its headers.
type Gateway struct {
	// CallID is the id the proxy logged the call under: the handle for matching
	// an unpriced or failed review to the gateway's own log.
	CallID string
	// ModelName is the deployment that ran (`azure/gpt-5.5`), which the
	// response body's model field, carrying the alias, never says.
	ModelName string
	// KeySpendNanoUSD is what the key has spent in total, as of this call. A
	// GAUGE the proxy maintains across every call made with the key, not a
	// per-call amount: show it, never add it up. nil when absent or unusable.
	KeySpendNanoUSD *int64
	// ResponseCost is the per-call cost header verbatim, for a log line. The
	// parsed figure is Usage.Cost; this is the text that produced it, or the
	// literal "None" an older proxy sends for a deployment it cannot price,
	// which is exactly what an operator needs to see.
	ResponseCost string
	// Cost is the proxy's split of that figure. Always empty on a stream,
	// whatever the proxy sent: LiteLLM has shipped the stream's total header
	// as "0", and a lane read there would be a confident zero for a paid call.
	Cost GatewayCost
	// Version is the proxy's own version: what explains a Cost an older
	// proxy never sends.
	Version string
}

// Reported says whether any gateway header arrived at all. An endpoint that
// sends none is not a LiteLLM proxy and never prices; one that sends some and
// no cost priced this deployment on another worker, and that difference is
// what a caller warns about.
func (g Gateway) Reported() bool {
	return g != Gateway{}
}

// GatewayCost is the proxy's split of its per-call cost, by component.
// Nano-USD; nil is "not sent", never zero.
//
// Informational only: Usage.Cost stays the one total. Each lane is rounded up
// on its own, so the lanes can sum past the total by a nano each — never
// rebuild the total from them.
type GatewayCost struct {
	// InputNanoUSD, CacheReadNanoUSD, CacheWriteNanoUSD, OutputNanoUSD and
	// ToolUsageNanoUSD are ADDITIVE: together they make the total. Input is
	// the UNCACHED part only — the proxy subtracts both cache lanes before
	// sending it, the reverse of the token lanes, where prompt_tokens includes
	// cached_tokens.
	InputNanoUSD      *int64
	CacheReadNanoUSD  *int64
	CacheWriteNanoUSD *int64
	OutputNanoUSD     *int64
	ToolUsageNanoUSD  *int64
	// ReasoningNanoUSD is a SUBSET of OutputNanoUSD. Never add it.
	ReasoningNanoUSD *int64
	// Sent but undocumented, so kept verbatim. Under a discount or margin the
	// operator configured, the total is that adjusted figure rather than the
	// vendor's list rate, and a warning says so; Original is the figure
	// before either.
	Original, DiscountAmount, MarginAmount, MarginPercent string
}

// Reported says whether any cost component arrived. Never on a stream.
func (c GatewayCost) Reported() bool {
	return c != GatewayCost{}
}

// parseGatewayHeaders reads the block, Cost aside: only a non-streamed call's
// path adds that, through parseGatewayCost, so a new call path that forgets it
// gets no split rather than a stream's confident zeros. Never fails: a bad
// header must not fail a completion the caller already paid for, so an
// unusable figure is nil with a warning.
func parseGatewayHeaders(hdr http.Header) (Gateway, []Warning) {
	var warnings []Warning
	g := Gateway{
		CallID:          hdr.Get(litellmCallIDHeader),
		ModelName:       hdr.Get(litellmModelHeader),
		ResponseCost:    hdr.Get(litellmCostHeader),
		Version:         hdr.Get(litellmVersionHeader),
		KeySpendNanoUSD: headerUSD(hdr, litellmKeySpendHeader, "key_spend", "key spend", &warnings),
	}
	return g, warnings
}

// parseGatewayCost reads the per-component split of a NON-STREAMED call.
func parseGatewayCost(hdr http.Header) (GatewayCost, []Warning) {
	var w []Warning
	return GatewayCost{
		InputNanoUSD:      headerUSD(hdr, litellmCostInputHeader, "gateway_cost", "input cost", &w),
		CacheReadNanoUSD:  headerUSD(hdr, litellmCostCacheReadHeader, "gateway_cost", "cache read cost", &w),
		CacheWriteNanoUSD: headerUSD(hdr, litellmCostCacheWriteHeader, "gateway_cost", "cache write cost", &w),
		OutputNanoUSD:     headerUSD(hdr, litellmCostOutputHeader, "gateway_cost", "output cost", &w),
		ReasoningNanoUSD:  headerUSD(hdr, litellmCostReasoningHeader, "gateway_cost", "reasoning cost", &w),
		ToolUsageNanoUSD:  headerUSD(hdr, litellmCostToolUsageHeader, "gateway_cost", "tool usage cost", &w),
		Original:          hdr.Get(litellmCostOriginalHeader),
		DiscountAmount:    hdr.Get(litellmCostDiscountHeader),
		MarginAmount:      hdr.Get(litellmCostMarginHeader),
		MarginPercent:     hdr.Get(litellmCostMarginPercentHeader),
	}, w
}

// adjustmentWarning flags a Reported total the operator's discount or margin
// moved off the list rate. Kept as reported, since it is what the key is
// charged, but every other figure here is list rate, so it says so. The text
// is fixed, so a multi-batch call names it once. Callers pass it only when
// the cost really came from the gateway.
func adjustmentWarning(c GatewayCost) []Warning {
	if !nonzero(c.DiscountAmount) && !nonzero(c.MarginAmount) && !nonzero(c.MarginPercent) {
		return nil
	}
	return []Warning{{Kind: WarnOther, Feature: "gateway_cost",
		Details: "gateway cost includes an operator discount or margin, so it is not the list rate; " +
			"Gateway.Cost.Original is the figure before them"}}
}

// headerUSD parses one nano-USD header: nil when absent or "None", nil and a
// warning when unusable.
func headerUSD(hdr http.Header, header, feature, what string, warnings *[]Warning) *int64 {
	raw := hdr.Get(header)
	if raw == "" || isNoneLiteral(raw) {
		return nil
	}
	nano, err := parseReportedCost(raw)
	if err != nil {
		*warnings = append(*warnings, Warning{Kind: WarnOther, Feature: feature,
			Details: fmt.Sprintf("gateway reported an unusable %s %q: %v", what, raw, err)})
		return nil
	}
	return &nano
}

// nonzero says whether a raw decimal header is a usable number other than
// zero, either sign: a discount spelled as a negative margin is an
// adjustment too.
func nonzero(raw string) bool {
	nano, err := parseReportedCost(strings.TrimPrefix(strings.TrimSpace(raw), "-"))
	return err == nil && nano > 0
}

// isNoneLiteral spots the text an older LiteLLM sends for a figure it does not
// have: the header is set via str(value), so a None becomes the literal "None"
// rather than an absent header (newer versions drop it). Not reported, not
// corrupt.
func isNoneLiteral(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "none", "null":
		return true
	}
	return false
}
