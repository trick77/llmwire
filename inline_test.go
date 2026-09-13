package llmwire

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestFirstInlineToolName(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Sure.<tool_call>\n", ""},
		{"<tool_call>\n<function=create_pdf_fi", ""},
		{"<tool_call>\n<function=create_pdf_file>\n<parameter=content>partial", "create_pdf_file"},
		{`<tool_invocation name="generate_image" arguments={"prompt": "partial`, "generate_image"},
		{`<tool_invocation display_name="Friendly Label" name="generate_image" arguments={"prompt": "a fox"} />`, "generate_image"},
	}
	for _, tc := range cases {
		if got := firstInlineToolName(tc.in); got != tc.want {
			t.Errorf("firstInlineToolName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestInlineToolCallID(t *testing.T) {
	if got := inlineToolCallID(0); got != "inline_call_1" {
		t.Fatalf("inlineToolCallID(0) = %q", got)
	}
	if got := inlineToolCallID(2); got != "inline_call_3" {
		t.Fatalf("inlineToolCallID(2) = %q", got)
	}
}

func TestParseInlineToolCalls_SingleBlock(t *testing.T) {
	calls, cleaned := parseInlineToolCalls("<tool_call>\n<function=tavily__tavily_search>\n<parameter=q>colossus forbin project</parameter>\n</function>\n</tool_call>")
	if len(calls) != 1 {
		t.Fatalf("calls = %#v, want 1", calls)
	}
	if calls[0].Type != "function" || calls[0].Name != "tavily__tavily_search" || calls[0].ID != "inline_call_1" {
		t.Fatalf("call = %#v", calls[0])
	}
	if calls[0].Arguments != `{"q":"colossus forbin project"}` {
		t.Fatalf("arguments = %q", calls[0].Arguments)
	}
	if cleaned != "" {
		t.Fatalf("cleaned = %q, want empty", cleaned)
	}
}

// Verbatim assistant content captured in production, where MiMo emitted two
// adjacent inline calls instead of native tool_calls.
func TestParseInlineToolCalls_ProductionCapture(t *testing.T) {
	content := "<tool_call>\n<function=tavily__tavily_search>\n<parameter=max_results>8</parameter>\n<parameter=q>Colossus Forbin Project 1970 Eric Braeden Hans Gudegast casting production history visual effects matte paintings</parameter>\n</function>\n</tool_call>" +
		"<tool_call>\n<function=tavily__tavily_search>\n<parameter=max_results>8</parameter>\n<parameter=q>Colossus Forbin Project James Bridges screenplay production design John Lloyd budget Universal 1970</parameter>\n</function>\n</tool_call>"
	calls, cleaned := parseInlineToolCalls(content)
	if len(calls) != 2 {
		t.Fatalf("calls = %#v, want 2", calls)
	}
	if calls[0].ID != "inline_call_1" || calls[1].ID != "inline_call_2" {
		t.Fatalf("ids = %q, %q", calls[0].ID, calls[1].ID)
	}
	if calls[0].Arguments != `{"max_results":"8","q":"Colossus Forbin Project 1970 Eric Braeden Hans Gudegast casting production history visual effects matte paintings"}` {
		t.Fatalf("call[0] arguments = %q", calls[0].Arguments)
	}
	if calls[1].Arguments != `{"max_results":"8","q":"Colossus Forbin Project James Bridges screenplay production design John Lloyd budget Universal 1970"}` {
		t.Fatalf("call[1] arguments = %q", calls[1].Arguments)
	}
	if cleaned != "" {
		t.Fatalf("cleaned = %q, want empty", cleaned)
	}
}

func TestParseInlineToolCalls_KeepsProseAndPassesThroughPlainText(t *testing.T) {
	calls, cleaned := parseInlineToolCalls("Let me search for that.\n<tool_call>\n<function=tavily__tavily_search>\n<parameter=q>colossus</parameter>\n</function>\n</tool_call>")
	if len(calls) != 1 || cleaned != "Let me search for that." {
		t.Fatalf("calls = %#v cleaned = %q", calls, cleaned)
	}
	plain := "Colossus: The Forbin Project is a 1970 film."
	if calls, cleaned := parseInlineToolCalls(plain); calls != nil || cleaned != plain {
		t.Fatalf("plain text: calls = %#v cleaned = %q", calls, cleaned)
	}
}

func TestParseInlineToolCalls_EscapesArgumentText(t *testing.T) {
	calls, _ := parseInlineToolCalls(`<tool_call><function=tavily__tavily_search><parameter=q>"quoted" & line
break</parameter></function></tool_call>`)
	if len(calls) != 1 || calls[0].Arguments != `{"q":"\"quoted\" & line\nbreak"}` {
		t.Fatalf("calls = %#v", calls)
	}
}

// The <tool_invocation …/> variant, verbatim from a production thread where the
// raw markup reached the UI instead of generating an image.
func TestParseInlineToolCalls_InvocationVariant(t *testing.T) {
	wantArgs := `{"filename": "rethinked-logo", "height": 1024, "prompt": "A modern, minimalist logo on a solid black background, icon in burnt orange (#c15f3c)."}`
	calls, cleaned := parseInlineToolCalls(`<tool_invocation name="generate_image" arguments=` + wantArgs + ` />`)
	if len(calls) != 1 {
		t.Fatalf("calls = %#v, want 1", calls)
	}
	if calls[0].ID != "inline_call_1" || calls[0].Type != "function" || calls[0].Name != "generate_image" {
		t.Fatalf("call = %#v", calls[0])
	}
	if calls[0].Arguments != wantArgs {
		t.Fatalf("arguments = %q", calls[0].Arguments)
	}
	if cleaned != "" {
		t.Fatalf("cleaned = %q, want empty", cleaned)
	}
}

func TestParseInlineToolCalls_InvocationCases(t *testing.T) {
	cases := []struct {
		name      string
		in        string
		wantArgs  string // "" means no call is expected
		wantClean string
	}{
		{"keeps prose", `Sure, here it is. <tool_invocation name="generate_image" arguments={"prompt": "a red fox"} /> Done.`, `{"prompt": "a red fox"}`, "Sure, here it is.  Done."},
		{"nested braces and angle", `<tool_invocation name="generate_image" arguments={"prompt": "a sign reading {SALE} 50% off >> here", "meta": {"w": 1024}} />`, `{"prompt": "a sign reading {SALE} 50% off >> here", "meta": {"w": 1024}}`, ""},
		{"marker inside args", `<tool_invocation name="generate_image" arguments={"prompt": "a diagram of a <tool_invocation> tag"} />`, `{"prompt": "a diagram of a <tool_invocation> tag"}`, ""},
		{"name-suffixed attribute", `<tool_invocation display_name="Friendly Label" name="generate_image" arguments={"prompt": "a fox"} />`, `{"prompt": "a fox"}`, ""},
		{"no arguments", `<tool_invocation name="ping" />`, `{}`, ""},
		// Malformed values are left in place, never degraded to {}: that would
		// dispatch a call with its arguments dropped.
		{"truncated", `<tool_invocation name="generate_image" arguments={"prompt": "truncated`, "", ""},
		{"truncated with inner >", `<tool_invocation name="generate_image" arguments={"prompt": "a > b"`, "", ""},
		{"single-quoted", `<tool_invocation name="generate_image" arguments={'prompt': 'fox'} />`, "", ""},
		{"non-object value", `<tool_invocation name="generate_image" arguments=oops />`, "", ""},
		{"no name", `<tool_invocation arguments={} />`, "", ""},
		{"no name before a named tag", `<tool_invocation arguments={"a":1} /> <tool_invocation name="ping" />`, "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls, cleaned := parseInlineToolCalls(tc.in)
			if tc.name == "no name before a named tag" {
				// The nameless tag must not borrow "ping"; only the named one parses.
				if len(calls) != 1 || calls[0].Name != "ping" || calls[0].Arguments != "{}" || !strings.HasPrefix(cleaned, `<tool_invocation arguments={"a":1} />`) {
					t.Fatalf("calls = %#v cleaned = %q", calls, cleaned)
				}
				return
			}
			if tc.wantArgs == "" {
				if calls != nil || cleaned != tc.in {
					t.Fatalf("calls = %#v cleaned = %q, want none and unchanged", calls, cleaned)
				}
				return
			}
			if len(calls) != 1 || calls[0].Name != "generate_image" && calls[0].Name != "ping" {
				t.Fatalf("calls = %#v", calls)
			}
			if calls[0].Arguments != tc.wantArgs {
				t.Fatalf("arguments = %q, want %q", calls[0].Arguments, tc.wantArgs)
			}
			if cleaned != tc.wantClean {
				t.Fatalf("cleaned = %q, want %q", cleaned, tc.wantClean)
			}
		})
	}
}

func TestCutAtFirstInlineMarker(t *testing.T) {
	cases := []struct{ in, want string }{
		{"a normal answer with no markup", "a normal answer with no markup"},
		{`Here you go. <tool_invocation name="generate_image" arguments={"prompt": "truncated`, "Here you go."},
		{"prose <tool_call> then <tool_invocation", "prose"},
	}
	for _, tc := range cases {
		if got := cutAtFirstInlineMarker(tc.in); got != tc.want {
			t.Errorf("cutAtFirstInlineMarker(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// gateCollect feeds deltas through a gate and returns everything it let out,
// including the final flush.
func gateCollect(deltas ...string) string {
	g := &inlineGate{}
	var out string
	for _, d := range deltas {
		out += g.push(d)
	}
	return out + g.flush()
}

func TestInlineGate(t *testing.T) {
	cases := []struct {
		name   string
		deltas []string
		want   string
	}{
		{"withholds a call at the start", []string{"<tool_call>", "<function=x>", "</function></tool_call>"}, ""},
		{"streams prose before the call", []string{"Let me search.\n", "<tool_call><function=x></function></tool_call>"}, "Let me search.\n"},
		{"marker split across deltas", []string{"<too", "l_ca", "ll><function=x></function></tool_call>"}, ""},
		{"invocation variant", []string{"Sure. ", `<tool_invocation name="generate_image" `, `arguments={"prompt": "a red fox"} />`}, "Sure. "},
		{"invocation split across deltas", []string{"<tool_inv", "ocation name=\"x\" arguments={} />"}, ""},
		{"plain content passes", []string{"Colossus ", "is a ", "1970 film."}, "Colossus is a 1970 film."},
		{"trailing lone bracket is flushed", []string{"a < b is 1 <"}, "a < b is 1 <"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := gateCollect(tc.deltas...); got != tc.want {
				t.Fatalf("emitted = %q, want %q", got, tc.want)
			}
		})
	}
}

// readInline runs readStream with recovery on and collects every event.
func readInline(t *testing.T, body string) (StreamResult, []streamEvent, []Warning) {
	t.Helper()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var counters streamCounters
	var events []streamEvent
	res, warnings, err := readStream(strings.NewReader(body), guard, &counters, time.Hour, func(ev streamEvent) {
		events = append(events, ev)
	}, Redact, true, time.Now, time.Now())
	if err != nil {
		t.Fatalf("readStream: %v", err)
	}
	return res, events, warnings
}

func frames(deltas ...string) string {
	var b strings.Builder
	for _, d := range deltas {
		b.WriteString("data: " + d + "\n\n")
	}
	b.WriteString(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n" + "data: [DONE]\n\n")
	return b.String()
}

func contentDelta(s string) string {
	return `{"choices":[{"delta":{"content":` + jsonString(s) + `}}]}`
}

func reasoningDelta(s string) string {
	return `{"choices":[{"delta":{"reasoning_content":` + jsonString(s) + `}}]}`
}

func jsonString(s string) string {
	var b strings.Builder
	writeJSONString(&b, s)
	return b.String()
}

func TestReadStream_RecoversInlineCallFromContent(t *testing.T) {
	body := frames(
		contentDelta("Let me look. "),
		contentDelta("<tool_call>\n<function=tavily__"),
		contentDelta("tavily_search>\n<parameter=q>colossus</parameter>\n</function>\n</tool_call>"),
	)
	res, events, warnings := readInline(t, body)

	if res.Content != "Let me look." {
		t.Fatalf("content = %q, want the prose only", res.Content)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "inline_call_1" || res.ToolCalls[0].Name != "tavily__tavily_search" {
		t.Fatalf("tool calls = %#v", res.ToolCalls)
	}
	if res.ToolCalls[0].Arguments != `{"q":"colossus"}` {
		t.Fatalf("arguments = %q", res.ToolCalls[0].Arguments)
	}

	var text []string
	var calls []toolCallDelta
	for _, ev := range events {
		switch ev.kind {
		case evContent:
			text = append(text, ev.text)
		case evToolCall:
			calls = append(calls, ev.toolCall)
		}
	}
	if got := strings.Join(text, ""); got != "Let me look. " {
		t.Fatalf("streamed content = %q, markup must never reach the sink", got)
	}
	// The name goes out as soon as it parses, before the arguments exist; the
	// final fragment carries the arguments under the same id and no name.
	if len(calls) != 2 {
		t.Fatalf("tool-call events = %#v, want name first then arguments", calls)
	}
	if calls[0].ID != "inline_call_1" || calls[0].Function.Name != "tavily__tavily_search" || calls[0].Function.Arguments != "" {
		t.Fatalf("early event = %#v", calls[0])
	}
	if calls[1].ID != "inline_call_1" || calls[1].Function.Name != "" || calls[1].Function.Arguments != `{"q":"colossus"}` {
		t.Fatalf("final event = %#v", calls[1])
	}
	if len(warnings) != 1 || warnings[0].Feature != "tool_calls" || !strings.Contains(warnings[0].Details, "content channel") {
		t.Fatalf("warnings = %v", warnings)
	}
}

func TestReadStream_RecoversInlineCallFromReasoning(t *testing.T) {
	body := frames(
		reasoningDelta("I should search. "),
		reasoningDelta(`<tool_invocation name="generate_image" arguments={"prompt": "a fox"} />`),
		contentDelta("Here is the image."),
	)
	res, events, warnings := readInline(t, body)

	if res.Reasoning != "I should search." {
		t.Fatalf("reasoning = %q", res.Reasoning)
	}
	if res.Content != "Here is the image." {
		t.Fatalf("content = %q", res.Content)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "generate_image" {
		t.Fatalf("tool calls = %#v", res.ToolCalls)
	}
	for _, ev := range events {
		if ev.kind == evReasoning && strings.Contains(ev.text, "<tool_invocation") {
			t.Fatalf("markup reached the reasoning sink: %q", ev.text)
		}
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0].Details, "reasoning channel") {
		t.Fatalf("warnings = %v", warnings)
	}
}

func TestReadStream_SecondInlineCallCarriesItsName(t *testing.T) {
	body := frames(contentDelta(
		"<tool_call><function=a><parameter=x>1</parameter></function></tool_call>" +
			"<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"))
	res, events, _ := readInline(t, body)

	if len(res.ToolCalls) != 2 || res.ToolCalls[1].ID != "inline_call_2" || res.ToolCalls[1].Name != "b" {
		t.Fatalf("tool calls = %#v", res.ToolCalls)
	}
	var final []toolCallDelta
	for _, ev := range events {
		if ev.kind == evToolCall && ev.toolCall.Function.Arguments != "" {
			final = append(final, ev.toolCall)
		}
	}
	if len(final) != 2 || final[1].Index != 1 || final[1].Function.Name != "b" || final[1].Function.Arguments != `{"y":"2"}` {
		t.Fatalf("final fragments = %#v", final)
	}
}

func TestReadStream_UnparsedMarkupIsCutAndWarned(t *testing.T) {
	body := frames(contentDelta(`Here you go. <tool_invocation name="generate_image" arguments={"prompt": "trunc`))
	res, _, warnings := readInline(t, body)

	if res.Content != "Here you go." {
		t.Fatalf("content = %q, the broken markup must be cut", res.Content)
	}
	if len(res.ToolCalls) != 0 {
		t.Fatalf("tool calls = %#v, want none", res.ToolCalls)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0].Details, "without a call parsing out of it") {
		t.Fatalf("warnings = %v", warnings)
	}
}

func TestReadStream_NativeCallsWinOverInline(t *testing.T) {
	body := frames(
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"a","arguments":"{}"}}]}}]}`,
		contentDelta("<tool_call><function=b></function></tool_call>"),
	)
	res, _, warnings := readInline(t, body)

	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "call_1" {
		t.Fatalf("tool calls = %#v, want the native call only", res.ToolCalls)
	}
	if res.Content != "" {
		t.Fatalf("content = %q, markup still cut", res.Content)
	}
	// The cut is reported: the model may only have quoted the markup.
	if len(warnings) != 1 || !strings.Contains(warnings[0].Details, "alongside a native call") {
		t.Fatalf("warnings = %v", warnings)
	}
}

func TestReadStream_PlainAnswerUntouchedWithRecoveryOn(t *testing.T) {
	body := frames(contentDelta("a < b is 1 "), contentDelta("<"))
	res, events, warnings := readInline(t, body)

	if res.Content != "a < b is 1 <" {
		t.Fatalf("content = %q", res.Content)
	}
	var text string
	for _, ev := range events {
		if ev.kind == evContent {
			text += ev.text
		}
	}
	if text != "a < b is 1 <" {
		t.Fatalf("streamed = %q, a held suffix must be flushed at the end", text)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings = %v", warnings)
	}
}

func TestReadStream_NoRecoveryLeavesMarkupAlone(t *testing.T) {
	body := frames(contentDelta("<tool_call><function=a></function></tool_call>"))
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	guard := newStallGuard(cancel, time.Hour, stallHeaders)
	defer guard.stop()
	var counters streamCounters
	res, warnings, err := readStream(strings.NewReader(body), guard, &counters, time.Hour, nil, Redact, false, time.Now, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Content, "<tool_call>") || len(res.ToolCalls) != 0 || len(warnings) != 0 {
		t.Fatalf("res = %#v warnings = %v", res, warnings)
	}
}

// The MiMo profiles opt in, so a call through the public API recovers too — on
// the streaming route and on Chat.
func TestChatStream_MiMoRecoversInlineCall(t *testing.T) {
	srv := sseServer(t, frames(contentDelta("<tool_call><function=search><parameter=q>x</parameter></function></tool_call>")))
	c := New(Config{BaseURL: srv.URL})
	stream, _, err := c.ChatStream(context.Background(), ChatRequest{Model: "mimo-v2.5-pro", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var calls []StreamEvent
	for stream.Next() {
		if ev := stream.Event(); ev.Kind == EventToolCall {
			calls = append(calls, ev)
		} else if ev.Kind == EventContent {
			t.Fatalf("content event %q, markup must be withheld", ev.Text)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || calls[0].Name != "search" || calls[1].ArgumentsDelta != `{"q":"x"}` {
		t.Fatalf("events = %#v", calls)
	}
	res := stream.Result()
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "search" || res.Content != "" {
		t.Fatalf("result = %#v", res)
	}
	if ws := stream.Warnings(); len(ws) == 0 || ws[0].Feature != "tool_calls" {
		t.Fatalf("warnings = %v", ws)
	}
}

func TestChat_MiMoRecoversInlineCall(t *testing.T) {
	srv, _ := jsonServer(t, 200, `{"model":"mimo-v2.5-pro","choices":[{"finish_reason":"stop","message":{"content":"Sure. <tool_invocation name=\"generate_image\" arguments={\"prompt\": \"a fox\"} />"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	c := New(Config{BaseURL: srv.URL})
	resp, warnings, err := c.Chat(context.Background(), ChatRequest{Model: "mimo-v2.5-pro", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "Sure." {
		t.Fatalf("content = %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "generate_image" || resp.ToolCalls[0].Arguments != `{"prompt": "a fox"}` {
		t.Fatalf("tool calls = %#v", resp.ToolCalls)
	}
	found := false
	for _, w := range warnings {
		found = found || w.Feature == "tool_calls"
	}
	if !found {
		t.Fatalf("warnings = %v, want the recovery named", warnings)
	}
}

func TestChat_ProfileWithoutRecoveryKeepsMarkup(t *testing.T) {
	srv, _ := jsonServer(t, 200, `{"model":"glm-5.3-flash","choices":[{"finish_reason":"stop","message":{"content":"<tool_call><function=a></function></tool_call>"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	c := New(Config{BaseURL: srv.URL})
	resp, _, err := c.Chat(context.Background(), ChatRequest{Model: "glm-5.3-flash", Messages: []Message{User("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(resp.Content, "<tool_call>") || len(resp.ToolCalls) != 0 {
		t.Fatalf("resp = %#v", resp)
	}
}

// Result text is what the gate let through: everything from the first marker
// on is cut, prose after a block included, so a delta-accumulating consumer
// and a Result reader hold the same text.
func TestReadStream_ResultMatchesStreamedText(t *testing.T) {
	body := frames(contentDelta(`Sure, here it is. <tool_invocation name="generate_image" arguments={"prompt": "a fox"} /> Done.`))
	res, events, warnings := readInline(t, body)
	var text string
	for _, ev := range events {
		if ev.kind == evContent {
			text += ev.text
		}
	}
	if res.Content != "Sure, here it is." || strings.TrimSpace(text) != res.Content {
		t.Fatalf("result = %q streamed = %q", res.Content, text)
	}
	if len(res.ToolCalls) != 1 || len(warnings) != 1 {
		t.Fatalf("calls = %#v warnings = %v", res.ToolCalls, warnings)
	}
}

func TestReadStream_FinishIsTheLastEventWithRecoveryOn(t *testing.T) {
	body := frames(contentDelta("<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"))
	_, events, _ := readInline(t, body)
	if last := events[len(events)-1]; last.kind != evFinish || last.finishReason != "stop" {
		t.Fatalf("last event = %#v, want the held finish", last)
	}
	for _, ev := range events[:len(events)-1] {
		if ev.kind == evFinish {
			t.Fatal("finish emitted before the recovered calls")
		}
	}
}

func TestReadStream_PartialBlocksAreCounted(t *testing.T) {
	body := frames(contentDelta("<tool_call><function=a><parameter=x>1</parameter></function></tool_call><tool_call><function=b><parameter=y>2"))
	res, _, warnings := readInline(t, body)
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "a" {
		t.Fatalf("calls = %#v", res.ToolCalls)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0].Details, "1 of 2 blocks did not parse") {
		t.Fatalf("warnings = %v", warnings)
	}
}

// The early name came from reasoning; the call was then recovered from content,
// so the final fragment names it again rather than trusting the early guess.
func TestReadStream_EarlyNameFromOtherChannelIsRenamed(t *testing.T) {
	body := frames(
		reasoningDelta("<tool_call><function=guess>"),
		contentDelta("<tool_call><function=real><parameter=x>1</parameter></function></tool_call>"),
	)
	res, events, _ := readInline(t, body)
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "real" {
		t.Fatalf("calls = %#v", res.ToolCalls)
	}
	var final []toolCallDelta
	for _, ev := range events {
		if ev.kind == evToolCall && ev.toolCall.Function.Arguments != "" {
			final = append(final, ev.toolCall)
		}
	}
	if len(final) != 1 || final[0].Function.Name != "real" || final[0].ID != "inline_call_1" {
		t.Fatalf("final = %#v, want the name carried", final)
	}
}
