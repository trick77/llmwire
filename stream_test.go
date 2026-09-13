package llmwire

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// readFixture runs a recorded SSE body through the parser with a guard that
// never fires, which is what lets these tests assert parsing alone.
func readFixture(t *testing.T, body string) (StreamResult, error) {
	t.Helper()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var counters streamCounters
	res, _, err := readStream(strings.NewReader(body), guard, &counters, streamBounds{idle: time.Hour}, nil, Redact, false, time.Now, time.Now())
	return res, err
}

// The usage object arrives in a trailing chunk with an EMPTY choices array,
// after the chunk carrying finish_reason — and that earlier chunk carries
// "usage": null. Stopping at finish_reason loses all accounting; indexing
// choices[0] on the trailing chunk panics; keeping the last usage SEEN rather
// than last REPORTED keeps the null.
func TestReadStream_TrailingUsageChunkWithEmptyChoices(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"Hello","role":"assistant"},"finish_reason":null,"index":0}]}

data: {"choices":[{"delta":{"content":null},"finish_reason":"stop","index":0}],"usage":null}

data: {"choices":[],"usage":{"prompt_tokens":15,"completion_tokens":16,"total_tokens":31,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":12}}}

data: [DONE]

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Content != "Hello" {
		t.Errorf("content = %q, want %q", res.Content, "Hello")
	}
	if res.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", res.FinishReason)
	}
	if !res.Done {
		t.Error("Done = false, want true")
	}
	if got := res.Usage.Input.Total; got == nil || *got != 15 {
		t.Errorf("input total = %v, want 15", deref(got))
	}
	if got := res.Usage.Input.CacheRead; got == nil || *got != 4 {
		t.Errorf("cache read = %v, want 4", deref(got))
	}
	// prompt_tokens includes cached_tokens, so the full-rate lane is 15-4.
	if got := res.Usage.Input.NoCache; got == nil || *got != 11 {
		t.Errorf("no-cache lane = %v, want 11", deref(got))
	}
	// completion_tokens includes reasoning_tokens, so text is 16-12.
	if got := res.Usage.Output.Text; got == nil || *got != 4 {
		t.Errorf("text lane = %v, want 4", deref(got))
	}
}

// The other shape: finish_reason, a non-empty delta and usage all in the SAME
// chunk. Both must work from one parser.
func TestReadStream_UsageInlineWithFinishReason(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}

data: [DONE]

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Content != "hi" {
		t.Errorf("content = %q, want hi", res.Content)
	}
	if got := res.Usage.Output.Total; got == nil || *got != 2 {
		t.Errorf("output total = %v, want 2", deref(got))
	}
}

// SSE permits "data:{...}" as well as "data: {...}". Matching only the spaced
// form makes every event from an endpoint that omits the space fall through as
// a non-data line, silently.
func TestReadStream_UnspacedDataPrefix(t *testing.T) {
	body := "data:{\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata:[DONE]\n\n"
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Content != "x" {
		t.Errorf("content = %q, want x", res.Content)
	}
}

// Reasoning streams on its own field with content null throughout, and some
// endpoints spell it "reasoning" rather than "reasoning_content".
func TestReadStream_ReasoningChannels(t *testing.T) {
	for _, tc := range []struct{ name, field string }{
		{"reasoning_content", "reasoning_content"},
		{"reasoning", "reasoning"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := "data: {\"choices\":[{\"delta\":{\"content\":null,\"" + tc.field + "\":\"think \"},\"finish_reason\":null}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"content\":null,\"" + tc.field + "\":\"harder\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n"
			res, err := readFixture(t, body)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if res.Reasoning != "think harder" {
				t.Errorf("reasoning = %q, want %q", res.Reasoning, "think harder")
			}
			if res.Content != "done" {
				t.Errorf("content = %q, want done", res.Content)
			}
		})
	}
}

// Native tool calls stream as fragments keyed by index: id/name arrive once on
// the first fragment and are empty afterwards, while arguments accumulate.
// Assigning unconditionally would blank the name on every later fragment.
func TestReadStream_ToolCallFragmentsAccumulate(t *testing.T) {
	body := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":"\"go\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(res.ToolCalls))
	}
	tc := res.ToolCalls[0]
	if tc.ID != "call_1" || tc.Name != "search" {
		t.Errorf("id/name = %q/%q, want call_1/search", tc.ID, tc.Name)
	}
	if tc.Arguments != `{"q":"go"}` {
		t.Errorf("arguments = %q, want %q", tc.Arguments, `{"q":"go"}`)
	}
}

// A failure discovered after the status line is committed arrives as a frame
// inside a 200 stream: one frame, no choices. Dropping it reads as a clean,
// empty stream and the caller reports success with no answer.
func TestReadStream_ErrorFrameInsideSuccessfulStream(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"par"},"finish_reason":null}]}

data: {"error":{"message":"upstream exploded","type":"server_error","code":"500"}}

`
	res, err := readFixture(t, body)
	if err == nil {
		t.Fatalf("expected an error, got content %q", res.Content)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error is %T, want *APIError", err)
	}
	if apiErr.Code != "500" || !strings.Contains(apiErr.Message, "exploded") {
		t.Errorf("decoded error = %+v", apiErr)
	}
}

// "error": null must not be mistaken for a reported error.
func TestReadStream_NullErrorFieldIsNotAnError(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"error":null}

data: [DONE]

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Content != "ok" {
		t.Errorf("content = %q, want ok", res.Content)
	}
}

// A connection dropped mid-answer ends the scan exactly as a finished one does:
// cleanly, with no read error. Accepting that would store a truncated answer as
// a success, so completion must be asserted by [DONE] or a finish_reason.
func TestReadStream_TruncatedStreamIsAnError(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"half an ans"},"finish_reason":null}]}

`
	res, err := readFixture(t, body)
	if err == nil {
		t.Fatal("expected an error for a stream with no finish_reason and no [DONE]")
	}
	if !strings.Contains(err.Error(), "without finish_reason") {
		t.Errorf("error = %v, want it to name the missing terminator", err)
	}
	if !errors.Is(err, ErrMalformedResponse) {
		t.Errorf("errors.Is(%v, ErrMalformedResponse) = false: a cut stream is a body that could not be read to the end", err)
	}
	// The partial content is still returned, so a caller can log what arrived.
	if res.Content != "half an ans" {
		t.Errorf("content = %q, want the partial text", res.Content)
	}
}

// A finish_reason with no [DONE] is a complete answer: several endpoints never
// send the marker.
func TestReadStream_FinishReasonWithoutDoneMarker(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"complete"},"finish_reason":"stop"}]}

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Done {
		t.Error("Done = true, want false: no [DONE] was sent")
	}
	if res.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", res.FinishReason)
	}
}

// One unparseable keepalive must not discard an answer that otherwise
// completed.
func TestReadStream_MalformedLineIsSkipped(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}

data: not json at all

: ping

data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}

data: [DONE]

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Content != "ab" {
		t.Errorf("content = %q, want ab", res.Content)
	}
}

// finish_reason is an open set: vendors add their own beyond the OpenAI
// vocabulary, and an unknown one must pass through rather than be rejected.
func TestReadStream_VendorFinishReasonsPassThrough(t *testing.T) {
	for _, reason := range []string{"repetition_truncation", "sensitive", "model_context_window_exceeded", "network_error"} {
		t.Run(reason, func(t *testing.T) {
			body := `data: {"choices":[{"delta":{"content":"x"},"finish_reason":"` + reason + `"}]}

`
			res, err := readFixture(t, body)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if res.FinishReason != reason {
				t.Errorf("finish_reason = %q, want %q", res.FinishReason, reason)
			}
		})
	}
}

// Events count data frames, not lines: an event is "data:…" plus a blank
// separator, so counting lines reports double.
func TestReadStream_EventsCountFramesNotLines(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}

: keepalive

data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}

data: [DONE]

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Two content frames plus the [DONE] frame.
	if res.Events != 3 {
		t.Errorf("events = %d, want 3", res.Events)
	}
	// Chars counts runes of content only.
	if res.Chars != 2 {
		t.Errorf("chars = %d, want 2", res.Chars)
	}
}

// Chars counts runes, not bytes: these endpoints return non-ASCII routinely and
// a byte count misreads as more output than the model produced.
func TestReadStream_CharsCountsRunesNotBytes(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"日本語"},"finish_reason":"stop"}]}

`
	res, err := readFixture(t, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Chars != 3 {
		t.Errorf("chars = %d, want 3 runes (not 9 bytes)", res.Chars)
	}
}

func TestReadStream_OnDeltaReceivesEveryContentFragment(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"one "},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"two"},"finish_reason":"stop"}]}

`
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var counters streamCounters
	var got []string
	res, _, err := readStream(strings.NewReader(body), guard, &counters, streamBounds{idle: time.Hour}, func(ev streamEvent) {
		if ev.kind == evContent {
			got = append(got, ev.text)
		}
	}, Redact, false, time.Now, time.Now())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Join(got, "") != res.Content {
		t.Errorf("streamed %q but returned %q; they must agree", strings.Join(got, ""), res.Content)
	}
	if len(got) != 2 {
		t.Errorf("got %d fragments, want 2", len(got))
	}
}

// --- stall guard --------------------------------------------------------------

func TestStallGuard_FiresAndNamesItsReason(t *testing.T) {
	fired := make(chan struct{})
	g := newStallGuard(func() { close(fired) }, 10*time.Millisecond, stallHeaders)
	defer g.stop()

	select {
	case <-fired:
	case <-time.After(2 * time.Second):
		t.Fatal("guard did not fire")
	}
	if got, _ := g.firedReason(); got != stallHeaders {
		t.Errorf("reason = %q, want %q", got, stallHeaders)
	}
}

func TestStallGuard_ArmChangesTheReasonReported(t *testing.T) {
	fired := make(chan struct{})
	g := newStallGuard(func() { close(fired) }, time.Hour, stallHeaders)
	defer g.stop()
	g.arm(10*time.Millisecond, stallIdle)

	select {
	case <-fired:
	case <-time.After(2 * time.Second):
		t.Fatal("guard did not fire after re-arming")
	}
	if got, _ := g.firedReason(); got != stallIdle {
		t.Errorf("reason = %q, want %q — naming the bound that actually elapsed is the point", got, stallIdle)
	}
}

// Re-arming must actually postpone the deadline, or every long thinking phase
// would be cancelled.
func TestStallGuard_ArmPostponesFiring(t *testing.T) {
	fired := make(chan struct{})
	g := newStallGuard(func() { close(fired) }, 40*time.Millisecond, stallHeaders)
	defer g.stop()

	for i := 0; i < 5; i++ {
		time.Sleep(10 * time.Millisecond)
		g.arm(40*time.Millisecond, stallIdle)
	}
	select {
	case <-fired:
		t.Fatal("guard fired despite being re-armed inside its bound")
	default:
	}
	if got, _ := g.firedReason(); got != "" {
		t.Errorf("reason = %q, want empty while healthy", got)
	}
}

// arm is a no-op once the guard has fired: the request is already being
// cancelled, and re-arming would leave a cancelled call looking healthy.
func TestStallGuard_ArmAfterFiringIsIgnored(t *testing.T) {
	fired := make(chan struct{})
	g := newStallGuard(func() { close(fired) }, 5*time.Millisecond, stallHeaders)
	defer g.stop()
	<-fired

	g.arm(time.Hour, stallIdle)
	if got, _ := g.firedReason(); got != stallHeaders {
		t.Errorf("reason = %q, want it to stay %q after firing", got, stallHeaders)
	}
}

func TestStallGuard_StopPreventsFiring(t *testing.T) {
	fired := make(chan struct{})
	g := newStallGuard(func() { close(fired) }, 20*time.Millisecond, stallHeaders)
	g.stop()

	select {
	case <-fired:
		t.Fatal("guard fired after stop")
	case <-time.After(100 * time.Millisecond):
	}
}

func deref(p *int64) any {
	if p == nil {
		return "<nil>"
	}
	return *p
}

// time.Timer.Reset cannot recall a callback the runtime has ALREADY scheduled.
// A data frame arriving in the window between expiry and fire() taking the
// mutex therefore leaves arm() believing it disarmed the guard, while the
// firing proceeds anyway. Without the deadline check, a stream that had just
// revived was cancelled regardless, and — worse — the failure was labelled with
// the reason arm() had just written rather than the bound that actually
// elapsed. Naming the right bound is this type's entire purpose.
//
// Calling fire() directly is the only way to reproduce that interleaving
// deterministically.
func TestStallGuard_StaleFiringReArmsInsteadOfCancelling(t *testing.T) {
	var cancels atomic.Int64
	g := newStallGuard(func() { cancels.Add(1) }, time.Hour, stallHeaders)
	defer g.stop()

	// Simulate the race: the deadline has just been pushed into the future by
	// arm(), but a firing scheduled before that is already on its way.
	g.arm(time.Hour, stallIdle)
	g.fire()

	if got := cancels.Load(); got != 0 {
		t.Errorf("cancel called %d times, want 0 — the deadline had moved", got)
	}
	if got, _ := g.firedReason(); got != "" {
		t.Errorf("reason = %q, want empty: the guard must still read as healthy", got)
	}
}

// The converse: once the deadline really has passed, fire() cancels and records
// the reason that was pending at that moment.
func TestStallGuard_FiringAfterTheDeadlineCancelsOnce(t *testing.T) {
	// Atomic because the timer's own goroutine may reach the callback first;
	// the assertion is that it happens exactly once however the firings race.
	var cancels atomic.Int64
	g := newStallGuard(func() { cancels.Add(1) }, time.Nanosecond, stallIdle)
	defer g.stop()
	time.Sleep(5 * time.Millisecond)

	g.fire()
	g.fire() // a second firing must be a no-op

	if got := cancels.Load(); got != 1 {
		t.Errorf("cancel called %d times, want exactly 1", got)
	}
	if got, _ := g.firedReason(); got != stallIdle {
		t.Errorf("reason = %q, want %q", got, stallIdle)
	}
}
