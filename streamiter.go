package llmwire

import (
	"context"
	"net/http"
	"sync"
	"time"
)

// The streaming call, as an iterator.
//
// An iterator rather than a pair of channels. A channel pair cannot be closed
// cleanly when the caller returns early, and it leaves Usage() — which only exists
// once the final chunk has arrived — with nowhere to live.
//
// Everything that can fail BEFORE the first byte fails synchronously, on the
// calling goroutine: the refusal, the render, the dial, the status. So a validation
// error, a 401 and a rate limit all arrive as the third return value rather than
// being buried in Err() after a pointless Next().

// EventKind says which channel an event came from.
type EventKind string

const (
	EventContent   EventKind = "content"
	EventReasoning EventKind = "reasoning"
	EventToolCall  EventKind = "tool_call"
	EventFinish    EventKind = "finish"
)

// StreamEvent is one increment of a streamed answer.
//
// Tool calls arrive as FRAGMENTS keyed by Index: the id and name come on the first
// one and the arguments build up as text across the rest. Assembled calls are on
// Result() once the stream ends, so a caller who only wants the finished calls
// need not reassemble them.
type StreamEvent struct {
	Kind EventKind
	// Text is the content or reasoning delta.
	Text string
	// Index, ID, Name and ArgumentsDelta describe a tool-call fragment. ID and
	// Name are present on the first fragment only.
	Index          int
	ID             string
	Name           string
	ArgumentsDelta string
	// FinishReason is the endpoint's own string, treated as an open set.
	FinishReason string
}

// Stream is a live streamed completion. Close it when done; a deferred Close is
// always correct, including after reading to the end.
type Stream struct {
	client *Client
	// ctx is the caller's, kept so a genuine parent cancellation stays
	// distinguishable from this package's own bounds.
	ctx     context.Context
	callCtx context.Context
	// cancelReq cancels the request; cancelCall releases the whole-call bound.
	cancelReq  context.CancelFunc
	cancelCall context.CancelFunc
	guard      *stallGuard

	// queue is UNBOUNDED, and that is the load-bearing decision in this file.
	//
	// The reader must never block. It re-arms the idle guard on each data frame,
	// so a hook that waited for a slow consumer would stop re-arming it and the
	// call would die reporting "stream idle" while the model was producing
	// perfectly — the exact misattribution the named bounds exist to prevent. An
	// unbounded queue costs nothing here: readStream already accumulates the whole
	// content and reasoning in memory regardless of the consumer, so the deltas
	// are held either way.
	mu     sync.Mutex
	cond   *sync.Cond
	queue  []StreamEvent
	closed bool // the reader has finished; no more events will arrive
	// stopped records a caller-initiated Close, so the cancellation it causes is
	// not reported as a timeout.
	stopped bool

	cur      StreamEvent
	res      StreamResult
	err      error
	warnings []Warning
	done     chan struct{}
	closeOne sync.Once
}

// ChatStream starts a streaming completion.
//
// The returned warnings are the VALIDATION warnings, available before any byte is
// consumed so a caller who wants to abort on a coercion can. Pricing cannot run
// until usage arrives, so the complete set — validation plus pricing — comes from
// Warnings() after Next() returns false.
func (c *Client) ChatStream(ctx context.Context, req ChatRequest) (*Stream, []Warning, error) {
	pl, warnings, err := c.plan(req, true)
	if err != nil {
		return nil, nil, err
	}
	body, err := renderChatBody(pl)
	if err != nil {
		c.finish(callSummary{kind: "chat_stream", model: req.Model, plan: pl, warnings: warnings, err: err})
		return nil, warnings, err
	}
	at := c.Now()

	// Two nested contexts, as RawStream uses: callCtx is the whole-call bound and
	// reqCtx is what the stall guard cancels. Cancelling reqCtx leaves callCtx's
	// deadline intact, so afterwards the two failures are still distinguishable.
	callCtx, cancelCall := context.WithTimeout(ctx, c.cap)
	reqCtx, cancelReq := context.WithCancel(callCtx)

	httpReq, err := c.newRequest(reqCtx, http.MethodPost, routeChat, body)
	if err != nil {
		cancelReq()
		cancelCall()
		c.finish(callSummary{kind: "chat_stream", model: req.Model, plan: pl, warnings: warnings, err: err})
		return nil, warnings, err
	}
	httpReq.Header.Set("Accept", "text/event-stream")

	guard := newStallGuard(cancelReq, c.header, stallHeaders)
	start := c.now()
	resp, err := c.http.Do(httpReq)
	headers := c.now().Sub(start)
	if err != nil {
		err = c.explain(ctx, callCtx, guard, c.dialError(routeChat, err))
		guard.stop()
		cancelReq()
		cancelCall()
		c.finish(callSummary{kind: "chat_stream", model: req.Model, plan: pl, warnings: warnings, err: err,
			timing: Timing{Headers: headers, Total: c.now().Sub(start)}})
		return nil, warnings, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		err = c.httpError(resp)
		resp.Body.Close()
		guard.stop()
		cancelReq()
		cancelCall()
		c.finish(callSummary{kind: "chat_stream", model: req.Model, plan: pl, warnings: warnings, err: err,
			timing: Timing{Headers: headers, Total: c.now().Sub(start)}})
		return nil, warnings, err
	}
	// Headers are in, so the bound that matters from here is silence.
	guard.arm(c.idle, stallIdle)

	s := &Stream{
		client:     c,
		ctx:        ctx,
		callCtx:    callCtx,
		cancelReq:  cancelReq,
		cancelCall: cancelCall,
		guard:      guard,
		warnings:   warnings,
		done:       make(chan struct{}),
	}
	s.cond = sync.NewCond(&s.mu)

	go s.read(resp, pl, at, start, headers)
	return s, warnings, nil
}

// read consumes the stream on its own goroutine and publishes the result.
func (s *Stream) read(resp *http.Response, pl *wirePlan, at, start time.Time, headers time.Duration) {
	var counters streamCounters
	res, inlineWarnings, err := readStream(resp.Body, s.guard, &counters,
		streamBounds{idle: s.client.idle, toolIdle: pl.req.ToolCallIdleTimeout}, s.push, s.client.redact,
		pl.profile.Tools.recoversInline(), s.client.now, start)
	res.Timing.Headers = headers

	resp.Body.Close()
	s.guard.stop()

	s.mu.Lock()
	stopped := s.stopped
	s.mu.Unlock()

	var priceWarnings []Warning
	switch {
	case stopped:
		// A caller-initiated Close cancelled the request, and explain() would
		// report that as "exceeded the call cap" — a timeout that never happened.
		// The truncation guard's "ended without finish_reason" is equally wrong
		// here: the caller stopped reading on purpose.
		err = nil
	case err != nil:
		err = s.client.explain(s.ctx, s.callCtx, s.guard, err)
	default:
		var cost Cost
		cost, priceWarnings = priceCall(pl.profile, res.Usage, resp.Header, resp.StatusCode, at)
		res.Usage.Cost = cost
	}

	// Reported from the locals: s.res and s.err are written under the mutex
	// below, and the caller may already be reading them.
	warnings := append(append(append([]Warning(nil), s.warnings...), inlineWarnings...), priceWarnings...)
	s.client.finish(callSummary{kind: "chat_stream", model: pl.req.Model, plan: pl,
		content: len(res.Content), reasoning: len(res.Reasoning), toolCalls: len(res.ToolCalls),
		finishReason: res.FinishReason, usage: res.Usage, timing: res.Timing,
		warnings: warnings, err: err, closed: stopped})

	s.mu.Lock()
	// Written under the mutex, all of it. Warnings() is documented as readable
	// while the stream is still running — it returns the validation warnings until
	// the pricing ones arrive — so appending outside the lock is a real race
	// against a caller polling it, not a theoretical one.
	//
	// Written before the reader is marked finished, and read only after Next has
	// observed that flag, so the mutex also carries the happens-before edge for
	// res and err.
	s.warnings = append(s.warnings, inlineWarnings...)
	s.warnings = append(s.warnings, priceWarnings...)
	s.res, s.err = res, err
	s.closed = true
	s.cond.Broadcast()
	s.mu.Unlock()
	close(s.done)
}

// push queues one event. Called on the reading goroutine and must never block.
func (s *Stream) push(ev streamEvent) {
	out := StreamEvent{}
	switch ev.kind {
	case evContent:
		out.Kind, out.Text = EventContent, ev.text
	case evReasoning:
		out.Kind, out.Text = EventReasoning, ev.text
	case evToolCall:
		out.Kind = EventToolCall
		out.Index = ev.toolCall.Index
		out.ID = ev.toolCall.ID
		out.Name = ev.toolCall.Function.Name
		out.ArgumentsDelta = ev.toolCall.Function.Arguments
	case evFinish:
		out.Kind, out.FinishReason = EventFinish, ev.finishReason
	}

	s.mu.Lock()
	s.queue = append(s.queue, out)
	s.cond.Signal()
	s.mu.Unlock()
}

// Next advances to the next event, blocking until one arrives. It returns false
// when the stream is finished or has failed; check Err afterwards.
func (s *Stream) Next() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.queue) == 0 && !s.closed {
		s.cond.Wait()
	}
	if len(s.queue) == 0 {
		return false
	}
	s.cur = s.queue[0]
	s.queue = s.queue[1:]
	return true
}

// Event returns the event Next advanced to.
func (s *Stream) Event() StreamEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cur
}

// Err returns why the stream ended, or nil. Valid once Next has returned false.
//
// A failure names the bound that gave up — "stream idle for 1m30s", "exceeded the
// 15m0s call cap" — rather than surfacing a bare context error, and a stream that
// ended without [DONE] or a finish_reason is an error rather than a short answer: a
// dropped connection looks exactly like a finished one otherwise.
func (s *Stream) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// Collect drains the stream: every content delta goes to onDelta as it
// arrives, in order, and the assembled result comes back once the stream
// ends. It is the loop every consumer that does not need the event kinds was
// writing by hand — Next, switch on Kind, keep the finish reason, then read
// Usage — and getting subtly different: one forgot the usage frame follows
// the finish, another read Usage before Next returned false.
//
// The result is returned WITH the error. A stream cut after its usage frame,
// or after content the reader has already seen, was paid for; a caller that
// accounts per call needs the figures whether or not the read completed, and
// gates on Usage.Total() — Reported() is true for an object with nothing
// countable in it. onDelta may be nil.
func (s *Stream) Collect(onDelta func(string)) (StreamResult, error) {
	for s.Next() {
		if ev := s.Event(); ev.Kind == EventContent && ev.Text != "" && onDelta != nil {
			onDelta(ev.Text)
		}
	}
	return s.Result(), s.Err()
}

// Result returns the accumulated answer: content, reasoning, assembled tool calls,
// finish reason and usage. Valid once Next has returned false.
func (s *Stream) Result() StreamResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.res
}

// Usage returns the token accounting and its cost. Valid once Next has returned
// false — usage arrives in the final or trailing chunk, so mid-stream it is the
// zero Usage, which reads as "not reported" rather than as zero tokens.
func (s *Stream) Usage() Usage {
	return s.Result().Usage
}

// Warnings returns the validation warnings plus anything the stream produced as it
// finished, which in practice means pricing. It is the union, not only the late
// ones, so a caller who checks one place gets everything.
func (s *Stream) Warnings() []Warning {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.warnings
}

// Close stops the stream and releases its goroutine and connection.
//
// Idempotent, and correct to defer: a caller who read to the end gets a no-op, and
// one who returns early gets a cancelled request. A deliberate close is not an
// error, so Err stays nil for it.
func (s *Stream) Close() error {
	s.closeOne.Do(func() {
		s.mu.Lock()
		s.stopped = true
		s.mu.Unlock()

		// Cancelling reqCtx unblocks the body read the goroutine is sitting in.
		s.cancelReq()
		<-s.done
		s.cancelCall()
	})
	return s.Err()
}
