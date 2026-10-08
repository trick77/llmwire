package llmwire

import (
	"context"
	"encoding/json"
	"fmt"
)

// Embeddings.
//
// The one place this package batches, because the endpoint takes a list and the
// caller's list has no natural bound. Everything else about it is conservative: no
// partial results, no truncation, no reordering trusted.

// routeEmbeddings is the embeddings route, appended to the configured base URL.
const routeEmbeddings = "/embeddings"

// embedBatchSize caps one request's inputs.
//
// The spec allows 2048. Sixty-four bounds the blast radius of a single rejected
// batch and one response's memory, at the cost of more round trips on a large
// corpus — and a caller embedding a corpus is already spending more on tokens than
// on requests.
const embedBatchSize = 64

// Embed returns one vector per input, in the order of the inputs.
func (c *Client) Embed(ctx context.Context, req EmbedRequest) (_ *EmbedResponse, warnings []Warning, err error) {
	if c.routes != nil {
		return c.routedEmbed(ctx, req)
	}
	p, warnings, dimensions, err := c.planEmbed(req)
	if err != nil {
		return nil, nil, err
	}

	// CallTimeout bounds the whole call, every batch together; each exchange
	// also caps itself, which is the same bound for a single batch.
	callCtx, cancelCall := context.WithTimeout(ctx, c.cap)
	defer cancelCall()

	out := &EmbedResponse{Vectors: make([][]float32, len(req.Inputs))}
	sum := callSummary{kind: "embed", model: req.Model, inputs: len(req.Inputs)}
	defer func() {
		// rawCall reads an outer deadline as the caller's own; this one is not.
		if err != nil && callCtx.Err() != nil && ctx.Err() == nil {
			err = fmt.Errorf("llmwire: %w", &callCapError{cap: c.cap})
		}
		// Usage on failure too: the batches before the failing one were paid
		// for, and a failed call returns no response to carry them.
		sum.timing, sum.warnings, sum.usage, sum.err = out.Timing, warnings, out.Usage, err
		c.finish(sum)
	}()
	// Summed only while every batch has reported: a partial sum understates and is
	// indistinguishable from a real total, which is worse than admitting the
	// number is unknown.
	var inputTotal int64
	allReported := true

	for start := 0; start < len(req.Inputs); start += embedBatchSize {
		// Checked between batches so a cancelled context stops the loop rather
		// than finishing a long corpus nobody is waiting for.
		if err := callCtx.Err(); err != nil {
			return nil, warnings, err
		}
		end := min(start+embedBatchSize, len(req.Inputs))
		batch := req.Inputs[start:end]

		body, err := renderEmbedBody(p, batch, dimensions)
		if err != nil {
			return nil, warnings, err
		}
		at := c.now()
		raw, hdr, timing, err := c.rawPost(callCtx, routeEmbeddings, body)
		// Summed before the error check, so the log line for a failed corpus
		// includes the batch that failed: that is the one whose wait explains
		// the failure.
		out.Timing.Headers += timing.Headers
		out.Timing.Total += timing.Total
		// Read before the error check, as on Chat: on a failure the failing
		// batch's call id is the one that matches the gateway's own log.
		var gwWarnings []Warning
		out.Gateway, gwWarnings = parseGatewayHeaders(hdr)
		sum.gateway = out.Gateway
		warnings = append(warnings, gwWarnings...)
		if err != nil {
			// No partial result. A half-filled [][]float32 is worse than none,
			// because the caller cannot tell which rows are real, and a row of
			// zeros is a valid-looking vector that means nothing.
			return nil, warnings, err
		}
		batchResp, err := parseEmbedResponseWith(c.redact, raw, len(batch))
		if err != nil {
			return nil, warnings, err
		}
		copy(out.Vectors[start:end], batchResp.vectors)
		out.Model = batchResp.model

		// Priced per batch, with the model that ran it: a total is a sum of
		// per-call prices, never re-derived from summed tokens.
		cost, priceWarnings := priceCall(p, batchResp.usage, hdr, 200, at)
		warnings = append(warnings, priceWarnings...)
		out.Usage.Cost.NanoUSD += cost.NanoUSD
		out.Usage.Cost.Provenance = mergeProvenance(out.Usage.Cost.Provenance, cost.Provenance, start == 0)
		out.Usage.Cost.AppliedAt = cost.AppliedAt
		if out.Usage.Cost.Provenance == Unpriced {
			// A partial sum is not a smaller price, it is an unknown one — and
			// Cost's own contract is that Unpriced carries 0, because a caller
			// comparing a total against a budget must not be handed three
			// batches' worth of a four-batch call.
			out.Usage.Cost.NanoUSD = 0
		}

		// Kept current per batch, so a later failure still accounts what was
		// paid. Built by hand rather than through parseUsage, so the flag it
		// would have set is set here: a sum of reported batches is reported.
		if t := batchResp.usage.Input.Total; t != nil && allReported {
			inputTotal += *t
			noCache := inputTotal
			out.Usage.Input.Total, out.Usage.Input.NoCache = &inputTotal, &noCache
			out.Usage.reported = true
		} else {
			allReported = false
			out.Usage.Input.Total, out.Usage.Input.NoCache = nil, nil
			out.Usage.reported = false
		}
	}
	return out, warnings, nil
}

// mergeProvenance combines per-batch provenance into one answer for the whole
// call. Any unpriced batch makes the total unpriced: a sum missing one batch is
// not a smaller price, it is an unknown one.
func mergeProvenance(acc, next CostProvenance, first bool) CostProvenance {
	if first {
		return next
	}
	if acc == next {
		return acc
	}
	return Unpriced
}

// planEmbed validates an embeddings request.
//
// Its own path: the chat plan refuses an embeddings profile outright, and the two
// requests share no field but the model id.
//
// WHAT IS DELIBERATELY NOT CHECKED: input length. The per-input ceiling is 8191
// tokens, and this package has no tokenizer — a character-count heuristic would be
// wrong in both directions, rejecting valid CJK text whose characters outnumber its
// tokens and waving through English prose that is over the line. So an over-length
// input reaches the endpoint and its 400 is surfaced as a typed error. Input is
// never truncated to fit: a silently shortened embedding is a wrong answer that
// looks right.
func (c *Client) planEmbed(req EmbedRequest) (*Profile, []Warning, *int, error) {
	if err := c.checkServes(req.Model); err != nil {
		return nil, nil, nil, err
	}
	p, err := c.registry.Lookup(req.Model)
	if err != nil {
		return nil, nil, nil, err
	}
	if p.Endpoint != EndpointEmbeddings {
		// The mirror of the chat check, and equally not demotable: a chat model
		// has no vectors to return.
		return nil, nil, nil, &UnsupportedError{
			Model:   p.ID,
			Feature: "embeddings",
			Reason:  fmt.Sprintf("this is a %s model and cannot take an embeddings request", p.Endpoint),
		}
	}

	v := &validation{profile: p, bestEffort: req.BestEffort}
	if len(req.Inputs) == 0 {
		v.reject("inputs", "an embeddings request needs at least one input")
		return nil, nil, nil, v.err
	}
	for i, in := range req.Inputs {
		// Structural: every endpoint 400s on an empty string, and catching it here
		// turns a whole rejected batch into a named local error.
		if in == "" {
			v.reject("inputs", fmt.Sprintf("input %d is empty", i))
			return nil, nil, nil, v.err
		}
	}

	dimensions := req.Dimensions
	if dimensions != nil {
		if *dimensions <= 0 {
			v.reject("dimensions", fmt.Sprintf("dimensions must be positive, got %d", *dimensions))
			return nil, nil, nil, v.err
		}
		if !p.Embedding.Dimensions {
			// Only the -3 generation takes this parameter; older models reject it
			// outright, so it cannot be sent hopefully.
			if v.refuse("dimensions", "this model does not accept the dimensions parameter", nil) {
				return nil, nil, nil, v.err
			}
			// Demoted: full-length vectors. Truncating them here instead would be
			// wrong twice — a full-length vector is unit length, so a hand-cut one
			// needs renormalising, which is exactly why the parameter exists.
			dimensions = nil
		}
	}
	return p, v.warnings, dimensions, nil
}

// wireEmbedResponse mirrors the embeddings response object.
type wireEmbedResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Usage json.RawMessage `json:"usage"`
	Error json.RawMessage `json:"error"`
}

type embedBatch struct {
	vectors [][]float32
	usage   Usage
	model   string
}

// parseEmbedResponseWith decodes one batch and places its vectors BY INDEX.
//
// The spec does not promise ordered data, and trusting the order would
// misattribute every vector to the wrong input — a failure that produces perfectly
// shaped output and that no downstream check would ever catch, unlike a missing
// row. So the index is used, and every way it can be wrong is an error.
func parseEmbedResponseWith(redact redactor, raw json.RawMessage, want int) (*embedBatch, error) {
	var w wireEmbedResponse
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("llmwire: %w: decoding embeddings response: %w (body: %s)",
			ErrMalformedResponse, err, bodySnippet(redact, raw))
	}
	if len(w.Error) > 0 && !isJSONNull(w.Error) {
		return nil, parseAPIErrorWith(redact, 0, raw)
	}
	if len(w.Data) != want {
		return nil, fmt.Errorf("llmwire: %w: asked for %d embeddings and got %d", ErrResponseShape, want, len(w.Data))
	}

	out := &embedBatch{vectors: make([][]float32, want), usage: parseUsage(w.Usage), model: w.Model}
	for _, d := range w.Data {
		if d.Index < 0 || d.Index >= want {
			return nil, fmt.Errorf("llmwire: %w: embeddings response has index %d outside the batch of %d",
				ErrResponseShape, d.Index, want)
		}
		if out.vectors[d.Index] != nil {
			return nil, fmt.Errorf("llmwire: %w: embeddings response repeats index %d", ErrResponseShape, d.Index)
		}
		out.vectors[d.Index] = d.Embedding
	}
	for i, v := range out.vectors {
		if v == nil {
			return nil, fmt.Errorf("llmwire: %w: embeddings response has no vector for index %d", ErrResponseShape, i)
		}
	}
	return out, nil
}
