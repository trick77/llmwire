package llmwire

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
)

// The models listing.
//
// GET {base}/models is the one read-only route this package speaks. Behind a
// gateway it does two jobs the completions route cannot: it says which models
// THIS key may use (LiteLLM lists exactly the key's own set), and it carries the
// gateway's per-deployment limits, which the chat response never does. A service
// that wants to size its prompts to a gateway-capped window reads it once at
// boot.

// routeModels is the listing route, appended to the configured base URL.
const routeModels = "/models"

// ModelEntry is one row of the listing.
//
// Only the id is promised by the OpenAI shape. The limits are LiteLLM
// extensions (`max_input_tokens`, `max_output_tokens`) and are nil when the
// entry does not carry them; a caller that needs one has to say so itself. Raw
// keeps the whole object for fields this package does not model.
type ModelEntry struct {
	ID string
	// MaxInputTokens is the gateway's context window for this alias. A gateway
	// may cap below the vendor's own limit, so this is the figure to size
	// prompts against, not the profile's.
	MaxInputTokens *int64
	// MaxOutputTokens is the gateway's completion cap for this alias.
	MaxOutputTokens *int64
	Raw             json.RawMessage
}

// wireModelEntry mirrors one listing row. Decoded leniently: an entry from a
// gateway carries dozens of fields nobody has modelled, and the two limits are
// read as raw JSON because they arrive in more than one shape (see limitField).
type wireModelEntry struct {
	ID              json.RawMessage `json:"id"`
	MaxInputTokens  json.RawMessage `json:"max_input_tokens"`
	MaxOutputTokens json.RawMessage `json:"max_output_tokens"`
}

// ListModels fetches the listing behind the client's base URL and key.
//
// The three returns follow Chat: an entry whose limit is present but unusable
// (a bool, a string, a fraction, zero) keeps a nil limit and adds a Warning
// naming it, rather than failing the whole listing over one row the caller may
// never look at. Only the listing's own shape is an error: `data` must be a
// list. A bad status is the same typed error a completion would return.
//
// Bounded by the caller's context and the whole-call cap, like every other
// route. A listing is quick, so a caller wanting a tighter bound for a boot
// check passes a context with one.
func (c *Client) ListModels(ctx context.Context) ([]ModelEntry, []Warning, error) {
	raw, _, timing, err := c.rawCall(ctx, http.MethodGet, routeModels, nil)
	if err != nil {
		c.log.Warn("llmwire models listing failed", "error", Truncate(c.redact(err.Error()), maxLoggedError),
			"headers_ms", timing.Headers.Milliseconds(), "total_ms", timing.Total.Milliseconds())
		return nil, nil, err
	}
	entries, warnings, err := parseModelsResponseWith(c.redact, raw)
	if err != nil {
		c.log.Warn("llmwire models listing failed", "error", Truncate(c.redact(err.Error()), maxLoggedError),
			"headers_ms", timing.Headers.Milliseconds(), "total_ms", timing.Total.Milliseconds())
		return nil, nil, err
	}
	// Not a model call: it goes through no plan, carries no usage and would
	// only pollute the per-model stats, so it reports on its own line rather
	// than through finish.
	c.log.Debug("llmwire models listed", "models", len(entries), "warnings", len(warnings),
		"headers_ms", timing.Headers.Milliseconds(), "total_ms", timing.Total.Milliseconds())
	return entries, warnings, nil
}

func parseModelsResponseWith(redact redactor, raw json.RawMessage) ([]ModelEntry, []Warning, error) {
	var w struct {
		Data  json.RawMessage `json:"data"`
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, nil, fmt.Errorf("llmwire: %w: decoding models listing: %w (body: %s)",
			ErrMalformedResponse, err, Truncate(redact(string(raw)), maxErrorBody))
	}
	if len(w.Error) > 0 && !isJSONNull(w.Error) {
		return nil, nil, parseAPIErrorWith(redact, 0, raw)
	}
	var rows []json.RawMessage
	if len(w.Data) == 0 || isJSONNull(w.Data) {
		return nil, nil, fmt.Errorf("llmwire: %w: models listing carries no data list (body: %s)",
			ErrResponseShape, Truncate(redact(string(raw)), maxErrorBody))
	}
	if err := json.Unmarshal(w.Data, &rows); err != nil {
		return nil, nil, fmt.Errorf("llmwire: %w: models listing data is not a list (body: %s)",
			ErrResponseShape, Truncate(redact(string(raw)), maxErrorBody))
	}

	var warnings []Warning
	out := make([]ModelEntry, 0, len(rows))
	for i, row := range rows {
		var e wireModelEntry
		if err := json.Unmarshal(row, &e); err != nil {
			// Not an object. The row is dropped with a warning rather than
			// failing the listing: it is not the row the caller is looking
			// for, or the caller will find it missing and say so.
			warnings = append(warnings, Warning{Kind: WarnOther, Feature: "models",
				Details: fmt.Sprintf("entry %d is not an object and was skipped", i)})
			continue
		}
		var id string
		if err := json.Unmarshal(e.ID, &id); err != nil || id == "" {
			warnings = append(warnings, Warning{Kind: WarnOther, Feature: "models",
				Details: fmt.Sprintf("entry %d has no string id and was skipped", i)})
			continue
		}
		entry := ModelEntry{ID: id, Raw: row}
		entry.MaxInputTokens = limitField(id, "max_input_tokens", e.MaxInputTokens, &warnings)
		entry.MaxOutputTokens = limitField(id, "max_output_tokens", e.MaxOutputTokens, &warnings)
		out = append(out, entry)
	}
	return out, warnings, nil
}

// limitField reads one of the gateway's token limits.
//
// LiteLLM emits an int, but its info endpoints have been seen with integral
// floats (16385.0), so those are accepted. Anything else that is present is a
// warning and nil, never a number: a bool decodes as 1 under a lenient cast, a
// fraction rounds, and a zero or negative window would size every prompt to
// nothing. Absent and null are plain nil, no warning, because a plain OpenAI
// listing never carries the field.
func limitField(model, name string, raw json.RawMessage, warnings *[]Warning) *int64 {
	if len(raw) == 0 || isJSONNull(raw) {
		return nil
	}
	unusable := func(why string) *int64 {
		*warnings = append(*warnings, Warning{Kind: WarnOther, Feature: name,
			Details: fmt.Sprintf("model %q lists %s as %s, %s", model, name, string(raw), why)})
		return nil
	}
	var num json.Number
	if err := json.Unmarshal(raw, &num); err != nil {
		return unusable("not a number")
	}
	// big.Rat rather than ParseFloat: it says exactly whether the value is an
	// integer, where a float comparison would pass 2^53+1 as one.
	r, ok := new(big.Rat).SetString(num.String())
	if !ok || !r.IsInt() {
		return unusable("not an integer")
	}
	if !r.Num().IsInt64() {
		return unusable("out of range")
	}
	v := r.Num().Int64()
	if v <= 0 {
		return unusable("not positive")
	}
	return &v
}
