package llmwire

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// mimo-v2.6-flash, asked with thinking disabled and no tools, answered with a
// draft, a stray closing tag and the answer, all in content:
//
//	FOUND: A.java:67 draft...),</think>FOUND: svc/A.java:67 the answer
//
// With reasoning.leaks_close_tag the text up to the FIRST </think> moves to
// Reasoning and Content keeps the answer.
func TestChat_StrayCloseTagIsCutWhereTheProfileSaysItLeaks(t *testing.T) {
	body := `{"model":"mimo-v2.6-flash","choices":[{"finish_reason":"stop",
	  "message":{"content":"draft line),</think>FOUND: svc/A.java:67 the answer"}}],
	  "usage":{"prompt_tokens":10,"completion_tokens":5}}`
	srv, _ := jsonServer(t, 200, body)
	resp, warnings, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:     "mimo-v2.6-flash",
		Messages:  []Message{User("hi")},
		Reasoning: ReasoningOff(),
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "FOUND: svc/A.java:67 the answer" {
		t.Errorf("content = %q, want the text after the stray tag", resp.Content)
	}
	if !strings.Contains(resp.Reasoning, "draft line") {
		t.Errorf("reasoning = %q, want the cut draft kept there", resp.Reasoning)
	}
	found := false
	for _, w := range warnings {
		if strings.Contains(w.Details, "</think>") {
			found = true
		}
	}
	if !found {
		t.Errorf("warnings = %v, want one saying a stray tag was cut", warnings)
	}
}

func TestChat_StrayCloseTagIsLeftAloneWithoutTheBit(t *testing.T) {
	body := `{"model":"mimo-v2.6-pro","choices":[{"finish_reason":"stop",
	  "message":{"content":"a</think>b"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	srv, _ := jsonServer(t, 200, body)
	resp, _, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "mimo-v2.6-pro",
		Messages: []Message{User("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "a</think>b" {
		t.Errorf("content = %q, want it untouched: the bit is off", resp.Content)
	}
}

// cutCase runs one flash reply through Chat and returns what the caller gets.
func cutCase(t *testing.T, content string, req ChatRequest) *ChatResponse {
	t.Helper()
	b, _ := json.Marshal(content)
	body := `{"model":"mimo-v2.6-flash","choices":[{"finish_reason":"stop","message":{"content":` +
		string(b) + `}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	srv, _ := jsonServer(t, 200, body)
	req.Model, req.Messages = "mimo-v2.6-flash", []Message{User("hi")}
	resp, _, err := chatClient(t, srv).Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	return resp
}

func TestChat_StrayCloseTagCutsOnlyALeak(t *testing.T) {
	off := ChatRequest{Reasoning: ReasoningOff()}
	cases := []struct {
		name, content string
		req           ChatRequest
		want          string
	}{
		{"an answer quoting the tag after the leak keeps its start",
			"draft</think>the answer quotes `</think>` here", off,
			"the answer quotes `</think>` here"},
		{"a quoted pair inside an answer is not a leak",
			"the tags are `<think>` and `</think>`", off,
			"the tags are `<think>` and `</think>`"},
		{"a leading think block is reasoning",
			"<think>draft</think>answer", off, "answer"},
		{"thinking not switched off: untouched",
			"a</think>b", ChatRequest{}, "a</think>b"},
		{"a JSON reply: untouched",
			`{"k":"a</think>b"}`, ChatRequest{Reasoning: ReasoningOff(),
				ResponseFormat: ResponseFormat{Kind: FormatJSONObject}}, `{"k":"a</think>b"}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := cutCase(t, c.content, c.req).Content; got != c.want {
				t.Errorf("content = %q, want %q", got, c.want)
			}
		})
	}
}

// Inline tool-call recovery runs on the answer, never on the draft: markup in
// a discarded draft must not become a call, and a broken marker there must not
// cut the answer away.
func TestChat_StrayCloseTagIsCutBeforeInlineRecovery(t *testing.T) {
	resp := cutCase(t, "draft <tool_call><function=x>broken</think>the answer",
		ChatRequest{Reasoning: ReasoningOff()})
	if resp.Content != "the answer" {
		t.Errorf("content = %q, want the answer", resp.Content)
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("tool calls = %v, want none from the draft", resp.ToolCalls)
	}
}
