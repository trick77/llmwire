package llmwire

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// SSE parsing for OpenAI-compatible chat completions.
//
// The wire details below are load-bearing and each cost a debugging session
// somewhere. They are recorded here rather than rediscovered.
//
// FRAMING
//
//   - "data:" has NO trailing space in the prefix match. SSE permits both
//     "data:{...}" and "data: {...}", and matching only the spaced form makes
//     every event from an endpoint that omits it fall through as a non-data
//     line — silently, with no error anywhere.
//   - A malformed data line is skipped, not fatal. One unparseable keepalive
//     must not discard an answer that otherwise completed.
//   - Completion is ASSERTED, never inferred. Only "[DONE]" or a finish_reason
//     ends a stream cleanly. A connection dropped mid-answer ends the scan
//     exactly as a finished one does — cleanly, with no read error — so
//     accepting whatever arrived would hand back a truncated answer as success.
//
// REASONING
//
//   - Reasoning arrives on its own delta field, reasoning_content, with content
//     null throughout. A long thinking phase therefore produces a steady flow
//     of events carrying no output at all, which is why liveness is measured in
//     events rather than in characters.
//   - Some endpoints spell the same field "reasoning". Both are read; content
//     wins if an endpoint somehow sends both.
//
// USAGE
//
//   - Its position differs by endpoint, and both shapes must work. MiMo-style:
//     a trailing chunk AFTER the finish_reason chunk, with an EMPTY choices
//     array — stopping at finish_reason loses all accounting, and indexing
//     choices[0] on it panics. Z.ai-style: finish_reason, a non-empty delta and
//     usage all in the SAME chunk.
//   - The finish_reason chunk commonly carries "usage": null. A usage field
//     being PRESENT is therefore not the same as usage being REPORTED, so the
//     last usage that wireUsage.reported() accepts is kept, not the last seen.
//
// FAILURE INSIDE A SUCCESS
//
//   - An error can arrive as a frame inside a 200 stream: one frame, no
//     choices, an "error" object. This is what an OpenAI-style upstream sends
//     when it fails AFTER the status line is committed, and what a LiteLLM
//     proxy sends when an SSE keepalive ping has already forced the 200.
//     Dropping that frame reads as a clean, empty stream.

const (
	dataPrefix = "data:"
	doneMarker = "[DONE]"
	// maxStreamLine caps one SSE line. A delta is ordinarily a few words, but
	// the bound is generous because the cost of guessing low is a failed
	// completion on a legitimately large chunk — a tool call whose arguments
	// are a whole file arrives as one line. bufio's 64 KiB default is a guess
	// nobody made deliberately.
	maxStreamLine = 1 << 20
	maxErrorBody  = 4 << 10
)

// streamDelta is one chunk's incremental payload.
type streamDelta struct {
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content"`
	// Reasoning is the alternate spelling some OpenAI-compatible servers use
	// for the same channel.
	Reasoning string          `json:"reasoning"`
	ToolCalls []toolCallDelta `json:"tool_calls"`
}

// reasoningText returns whichever reasoning spelling this endpoint used.
func (d streamDelta) reasoningText() string {
	if d.ReasoningContent != "" {
		return d.ReasoningContent
	}
	return d.Reasoning
}

// toolCallDelta is a fragment of a native tool call. Arguments stream as a
// string built up across chunks, keyed by Index — Id and Name arrive once, on
// the first fragment, and are empty on the rest.
type toolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ToolCall is a completed tool call assembled from its fragments.
type ToolCall struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// streamChunk is one `data:` event.
type streamChunk struct {
	Choices []struct {
		Delta        streamDelta `json:"delta"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	// Usage is kept raw so the identical bytes can feed both the decoder and a
	// debug line showing what the endpoint really sent — which is what settles
	// whether a zero is an answer or an unrecognised field name.
	Usage json.RawMessage `json:"usage"`
	// Error catches the failure-inside-a-200 case described in the head
	// comment.
	Error json.RawMessage `json:"error"`
}

// StreamResult is what a completed stream yielded.
type StreamResult struct {
	Content   string
	Reasoning string
	ToolCalls []ToolCall
	// FinishReason is the endpoint's own string and is treated as an OPEN set.
	// Vendors add their own beyond the OpenAI vocabulary — MiMo sends
	// "repetition_truncation", Z.ai sends "sensitive",
	// "model_context_window_exceeded" and "network_error" — so this is never
	// switched on exhaustively.
	FinishReason string
	Usage        Usage
	// Events counts `data:` frames received, Chars counts runes of content.
	Events int64
	Chars  int64
	// Done records that the endpoint sent [DONE].
	Done bool
	// MaxCommentGap is the longest interval during which the endpoint sent
	// only comment or blank lines and no `data:` frame. Recorded because the
	// idle guard deliberately does NOT re-arm on those lines, so this is the
	// margin between that policy and a false abort — the number the transport
	// probe exists to measure.
	MaxCommentGap time.Duration
}

// streamCounters are the live counts a heartbeat can read while the stream is
// still being consumed. Atomic because the reader goroutine writes them while
// another goroutine reads.
type streamCounters struct {
	events atomic.Int64
	chars  atomic.Int64
}

// readStream consumes an SSE body.
//
// The idle guard is re-armed on `data:` frames ONLY — never on blank separators
// or on SSE comment lines (": ping"). Those prove the connection is alive but
// say nothing about the model making progress, and an upstream that emits them
// on a timer would otherwise mask a model that has stalled completely. Reasoning
// deltas are real data frames, so a legitimately long thinking phase still
// re-arms the guard and is unaffected by this rule.
// streamEvent is one increment the reader hands onward. Every channel the parser
// understands is represented, not only content: an iterator that could not see
// reasoning deltas or tool-call fragments would be strictly weaker than the
// accumulated result it sits in front of, and callers would go back to waiting for
// the whole answer.
type streamEvent struct {
	kind         eventKind
	text         string
	toolCall     toolCallDelta
	finishReason string
}

type eventKind uint8

const (
	evContent eventKind = iota
	evReasoning
	evToolCall
	evFinish
)

func readStream(body io.Reader, guard *stallGuard, counters *streamCounters, idle time.Duration, sink func(streamEvent)) (StreamResult, error) {
	var (
		content   strings.Builder
		reasoning strings.Builder
		res       StreamResult
		tools     = map[int]*ToolCall{}
		order     []int
	)

	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64<<10), maxStreamLine)

	lastData := time.Now()
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		if line == "" || !strings.HasPrefix(line, dataPrefix) {
			// Deliberately no guard.arm here. See the doc comment.
			continue
		}

		// Measure the gap this frame closes before re-arming, so the recorded
		// figure is the real silence the guard had to tolerate.
		if gap := time.Since(lastData); gap > res.MaxCommentGap {
			res.MaxCommentGap = gap
		}
		lastData = time.Now()
		guard.arm(idle, stallIdle)

		// Counted per event rather than per line: an event is "data:…" plus a
		// blank separator, so counting lines reports double, and a log saying
		// 824 for a 412-chunk stream is one nobody can reconcile.
		res.Events = counters.events.Add(1)

		payload := strings.TrimSpace(strings.TrimPrefix(line, dataPrefix))
		if payload == doneMarker {
			res.Done = true
			break
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}

		// A failure that arrived after the 200. Surfaced, not skipped: this
		// frame carries no choices, so ignoring it yields a clean empty stream
		// and the caller reports success with no answer.
		if len(chunk.Error) > 0 && !isJSONNull(chunk.Error) {
			return res, parseAPIError(0, chunk.Error)
		}

		emit := func(ev streamEvent) {
			if sink != nil {
				sink(ev)
			}
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				content.WriteString(ch.Delta.Content)
				emit(streamEvent{kind: evContent, text: ch.Delta.Content})
				// Runes, not bytes: these endpoints return non-ASCII routinely,
				// and a count inflated by UTF-8 encoding misreads as more
				// output than the model produced.
				res.Chars = counters.chars.Add(int64(utf8.RuneCountInString(ch.Delta.Content)))
			}
			if r := ch.Delta.reasoningText(); r != "" {
				reasoning.WriteString(r)
				emit(streamEvent{kind: evReasoning, text: r})
			}
			for _, tc := range ch.Delta.ToolCalls {
				emit(streamEvent{kind: evToolCall, toolCall: tc})
				acc, seen := tools[tc.Index]
				if !seen {
					acc = &ToolCall{}
					tools[tc.Index] = acc
					order = append(order, tc.Index)
				}
				// Id, Type and Name arrive on the first fragment only; the
				// rest carry argument text. Assigning unconditionally would
				// blank them on every later fragment.
				if tc.ID != "" {
					acc.ID = tc.ID
				}
				if tc.Type != "" {
					acc.Type = tc.Type
				}
				if tc.Function.Name != "" {
					acc.Name = tc.Function.Name
				}
				acc.Arguments += tc.Function.Arguments
			}
			if ch.FinishReason != "" {
				res.FinishReason = ch.FinishReason
				emit(streamEvent{kind: evFinish, finishReason: ch.FinishReason})
			}
		}

		// Keep the last usage actually REPORTED, not the last one seen — the
		// finish_reason chunk commonly rides along with "usage": null.
		if len(chunk.Usage) > 0 {
			var w wireUsage
			if err := json.Unmarshal(chunk.Usage, &w); err == nil && w.reported() {
				res.Usage = parseUsage(chunk.Usage)
			}
		}
	}

	res.Content = content.String()
	res.Reasoning = reasoning.String()
	for _, i := range order {
		res.ToolCalls = append(res.ToolCalls, *tools[i])
	}

	if err := sc.Err(); err != nil {
		return res, err
	}
	if !res.Done && res.FinishReason == "" {
		return res, fmt.Errorf("stream ended after %d events (%d chars) without finish_reason or %s",
			res.Events, res.Chars, doneMarker)
	}
	return res, nil
}

// isJSONNull reports whether raw is the literal null, so a `"error": null` field
// is not mistaken for a reported error.
func isJSONNull(raw json.RawMessage) bool {
	return strings.TrimSpace(string(raw)) == "null"
}

// Reasons a request was cancelled from our side, used verbatim in the returned
// error so a failure names which bound gave up. The whole point is that "the
// endpoint sent nothing at all" and "the model thought for a long time" stop
// being the same log line.
const (
	stallHeaders = "no response headers"
	stallIdle    = "stream idle"
)

// stallGuard cancels a request when nothing has arrived within the current
// bound, and remembers why. It starts armed for headers and is re-armed for
// idleness on every data frame, so one mechanism covers both phases while each
// keeps its own name.
//
// Doing this rather than leaning on Transport.ResponseHeaderTimeout alone is
// deliberate: that setting is an HTTP/1.1 transport feature and does not apply
// once a connection is negotiated as HTTP/2, which is the case for these
// endpoints. It is still set, as a backstop; this is the bound actually relied
// on.
type stallGuard struct {
	cancel context.CancelFunc

	mu       sync.Mutex
	timer    *time.Timer
	deadline time.Time     // when the current arming expires
	pending  string        // reason the currently-armed deadline would report
	bound    time.Duration // length of the current arming, reported beside pending
	reason   string        // reason it actually fired with, "" while healthy
	firedFor time.Duration // the bound that elapsed when it fired
	fired    bool
}

// newStallGuard arms the guard for its first deadline. Callers must stop() it.
func newStallGuard(cancel context.CancelFunc, d time.Duration, reason string) *stallGuard {
	g := &stallGuard{cancel: cancel, pending: reason, bound: d, deadline: time.Now().Add(d)}
	g.timer = time.AfterFunc(d, g.fire)
	return g
}

// arm resets the deadline and the reason it would report. It is a no-op once
// the guard has fired: the request is already being cancelled, and re-arming
// would leave a cancelled call looking healthy.
//
// The deadline is recorded as a timestamp, not merely handed to Reset, because
// Reset alone is not enough — see fire.
func (g *stallGuard) arm(d time.Duration, reason string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.fired {
		return
	}
	g.pending = reason
	g.bound = d
	g.deadline = time.Now().Add(d)
	g.timer.Reset(d)
}

// fire cancels the request, unless the deadline moved while this firing was
// already on its way.
//
// Reset cannot recall a callback the runtime has ALREADY scheduled. An event
// arriving in the window between expiry and this function taking the mutex
// therefore leaves arm() believing it disarmed the guard — it saw fired == false
// — while this firing proceeds anyway. Two things went wrong without the check
// below: a stream that had just revived was cancelled regardless, and the
// failure was labelled with the reason arm() had just written rather than the
// bound that actually elapsed. The second is the worse one, since naming the
// right bound is this type's entire purpose. Comparing against the recorded
// deadline makes a stale firing detectable and re-arms for the remaining time.
func (g *stallGuard) fire() {
	g.mu.Lock()
	if g.fired {
		g.mu.Unlock()
		return
	}
	if remaining := time.Until(g.deadline); remaining > 0 {
		g.timer.Reset(remaining)
		g.mu.Unlock()
		return
	}
	g.fired = true
	g.reason, g.firedFor = g.pending, g.bound
	g.mu.Unlock()
	g.cancel()
}

// stop disarms the guard. It does not cancel the request; the caller's own
// defer does that.
func (g *stallGuard) stop() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.timer.Stop()
}

// firedReason returns why the guard cancelled and the bound that elapsed, or ""
// if it did not fire. Read after a request error to turn a bare "context
// canceled" into the bound that caused it. The duration comes from the guard,
// not from the client's configured bounds: RawPost arms the header phase with the
// whole-call cap, so the client's header bound would name a number that never
// applied.
func (g *stallGuard) firedReason() (string, time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.reason, g.firedFor
}
