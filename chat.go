package llmwire

import (
	"context"
	"encoding/json"
	"fmt"
)

// The non-streaming call.
//
// Warnings ride back beside a SUCCESSFUL response rather than being logged,
// because Go has no warnings mechanism: a log line is invisible to the caller and
// an error is too blunt for a request that worked. A refusal returns before any
// socket opens.

// routeChat is the completions route, appended to the configured base URL.
const routeChat = "/chat/completions"

// Chat sends one request and returns the completed turn.
//
// The three returns are (response, warnings, error) in that order: a coercion
// that succeeded is not an error, and a caller who ignores the middle value still
// gets a working call — which is the point of the tier-3 policy.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, []Warning, error) {
	pl, warnings, err := c.plan(req, false)
	if err != nil {
		return nil, nil, err
	}
	body, err := renderChatBody(pl)
	if err != nil {
		return nil, warnings, err
	}
	// Everything past the plan ends in one line and one stats update, on
	// success and on every failure alike.
	sum := callSummary{kind: "chat", model: req.Model, plan: pl}
	defer func() { c.finish(sum) }()

	// Captured BEFORE the request, and carried into pricing: which instant bills
	// is unspecified by both vendors, so llmwire uses request start and records
	// what it used. Taking it afterwards would attribute a call that straddles an
	// off-peak boundary to the wrong window.
	at := c.Now()

	raw, hdr, timing, err := c.rawPost(ctx, routeChat, body)
	sum.timing = timing
	// Read before the error checks: a proxy 429/5xx returns its headers, and
	// its call id is what matches the failure to the gateway's own log. A
	// nil header map (dial failure) reads as empty.
	gateway, gwWarnings := parseGatewayHeaders(hdr)
	sum.gateway = gateway
	if err != nil {
		sum.err = err
		return nil, warnings, err
	}
	resp, err := parseChatResponseWith(c.redact, raw)
	if err != nil {
		sum.err = err
		return nil, warnings, err
	}
	resp.Timing = timing
	if pl.profile.Tools.recoversInline() {
		rec := recoverInline(resp.Content, resp.Reasoning, len(resp.ToolCalls))
		resp.Content, resp.Reasoning = rec.content, rec.reasoning
		resp.ToolCalls = append(resp.ToolCalls, rec.calls...)
		warnings = append(warnings, rec.warnings()...)
	}
	if pl.profile.Reasoning.LeaksCloseTag {
		var cut bool
		resp.Content, resp.Reasoning, cut = cutStrayCloseTag(resp.Content, resp.Reasoning)
		if cut {
			warnings = append(warnings, Warning{
				Kind:    WarnOther,
				Feature: "content",
				Details: "a stray </think> in content was cut; the text before it moved to reasoning",
			})
		}
	}
	cost, priceWarnings := priceCall(pl.profile, resp.Usage, hdr, 200, at)
	resp.Usage.Cost = cost
	warnings = append(warnings, priceWarnings...)
	resp.Gateway = gateway
	warnings = append(warnings, gwWarnings...)
	sum.content, sum.reasoning, sum.toolCalls = len(resp.Content), len(resp.Reasoning), len(resp.ToolCalls)
	sum.finishReason, sum.usage, sum.warnings = resp.FinishReason, resp.Usage, warnings
	return resp, warnings, nil
}

// wireChatResponse mirrors the OpenAI-compatible response object.
//
// Decoded LENIENTLY, never with DisallowUnknownFields: these endpoints carry
// fields nobody has modelled, and failing a completion the caller already paid
// for because of an unrecognised key is the worst available trade.
type wireChatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			// Two spellings, same field. Mirrors the stream parser, which reads
			// both because servers disagree about the name.
			ReasoningContent string `json:"reasoning_content"`
			Reasoning        string `json:"reasoning"`
		} `json:"message"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
	// Error catches a failure delivered under a 200, which is what several of
	// these endpoints do once they have committed a status.
	Error json.RawMessage `json:"error"`
}

// parseChatResponseWith decodes a completion.
func parseChatResponseWith(redact redactor, raw json.RawMessage) (*ChatResponse, error) {
	var w wireChatResponse
	if err := json.Unmarshal(raw, &w); err != nil {
		// The body goes into the error, redacted and bounded: a compat server
		// answering with an HTML error page would otherwise produce a bare
		// "invalid character '<'" and nothing to identify what answered.
		return nil, fmt.Errorf("llmwire: %w: decoding response: %w (body: %s)",
			ErrMalformedResponse, err, Truncate(redact(string(raw)), maxErrorBody))
	}

	// A 200 carrying an error object. Surfaced rather than read as an empty
	// answer: the same shape inside a stream frame is already handled this way,
	// and dropping it here would report success with no content.
	if len(w.Error) > 0 && !isJSONNull(w.Error) {
		return nil, parseAPIErrorWith(redact, 0, raw)
	}
	// Never index Choices[0]: a choices-less 200 is exactly what the error case
	// above looks like when the error object is absent too, and an index would
	// panic instead of saying so.
	if len(w.Choices) == 0 {
		return nil, fmt.Errorf("llmwire: %w: response carried no choices and no error (body: %s)",
			ErrResponseShape, Truncate(redact(string(raw)), maxErrorBody))
	}

	ch := w.Choices[0]
	resp := &ChatResponse{
		Content: ch.Message.Content,
		// finish_reason is copied verbatim and never switched on: vendors extend
		// the vocabulary ("repetition_truncation", "sensitive"), and an unknown
		// value must not be fatal.
		FinishReason: ch.FinishReason,
		Model:        w.Model,
		Usage:        parseUsage(w.Usage),
		Raw:          raw,
	}
	resp.Reasoning = ch.Message.ReasoningContent
	if resp.Reasoning == "" {
		resp.Reasoning = ch.Message.Reasoning
	}
	for _, tc := range ch.Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, ToolCall{
			ID:        tc.ID,
			Type:      tc.Type,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}
	return resp, nil
}
