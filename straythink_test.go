package llmwire

import (
	"context"
	"strings"
	"testing"
)

// mimo-v2.6-flash, asked with thinking disabled and no tools, answered with a
// draft, a stray closing tag and the answer, all in content:
//
//	FOUND: A.java:67 draft...),</think>FOUND: svc/A.java:67 the answer
//
// With reasoning.leaks_close_tag the text up to the last </think> moves to
// Reasoning and Content keeps the answer.
func TestChat_StrayCloseTagIsCutWhereTheProfileSaysItLeaks(t *testing.T) {
	body := `{"model":"mimo-v2.6-flash","choices":[{"finish_reason":"stop",
	  "message":{"content":"draft line),</think>FOUND: svc/A.java:67 the answer"}}],
	  "usage":{"prompt_tokens":10,"completion_tokens":5}}`
	srv, _ := jsonServer(t, 200, body)
	resp, warnings, err := chatClient(t, srv).Chat(context.Background(), ChatRequest{
		Model:    "mimo-v2.6-flash",
		Messages: []Message{User("hi")},
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
