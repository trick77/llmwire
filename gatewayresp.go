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
}

// Reported says whether any gateway header arrived at all. An endpoint that
// sends none is not a LiteLLM proxy and never prices; one that sends some and
// no cost priced this deployment on another worker, and that difference is
// what a caller warns about.
func (g Gateway) Reported() bool {
	return g.CallID != "" || g.ModelName != "" || g.KeySpendNanoUSD != nil || g.ResponseCost != ""
}

// parseGatewayHeaders reads the block. Never fails: a bad header must not
// fail a completion the caller already paid for, so an unusable key spend is
// nil with a warning.
func parseGatewayHeaders(hdr http.Header) (Gateway, []Warning) {
	g := Gateway{
		CallID:       hdr.Get(litellmCallIDHeader),
		ModelName:    hdr.Get(litellmModelHeader),
		ResponseCost: hdr.Get(litellmCostHeader),
	}
	var warnings []Warning
	if raw := hdr.Get(litellmKeySpendHeader); raw != "" && !isNoneLiteral(raw) {
		nano, err := parseReportedCost(raw)
		if err != nil {
			warnings = append(warnings, Warning{Kind: WarnOther, Feature: "key_spend",
				Details: fmt.Sprintf("gateway reported an unusable key spend %q: %v", raw, err)})
		} else {
			g.KeySpendNanoUSD = &nano
		}
	}
	return g, warnings
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
