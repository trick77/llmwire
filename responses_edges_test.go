package llmwire

import (
	"encoding/json"
	"strings"
	"testing"
)

// Edge cases of the Responses protocol, each as the wire would show it.

func TestResponses_ReplayEdges(t *testing.T) {
	// A reasoning item that cannot go back takes its calls' replay with it:
	// a call without its reasoning item is refused.
	resp, err := parseResponsesResponseWith(Redact, json.RawMessage(`{"status":"completed","output":[
		{"type":"reasoning","id":"rs"},
		{"type":"function_call","id":"fc","call_id":"c","name":"f","arguments":"{}"}]}`))
	if err != nil || resp.ResponseItems != nil || len(resp.ToolCalls) != 1 {
		t.Fatalf("unreplayable reasoning: %+v, %v", resp, err)
	}
	// An upstream that omits the item id: no empty id on the wire.
	resp, _ = parseResponsesResponseWith(Redact, json.RawMessage(`{"status":"completed","output":[
		{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]}`))
	if got := string(resp.ResponseItems[0]); strings.Contains(got, `"id"`) {
		t.Errorf("call item = %s", got)
	}
	// A refusal is an answer.
	resp, err = parseResponsesResponseWith(Redact, json.RawMessage(`{"status":"completed","output":[
		{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}]}`))
	if err != nil || resp.Content != "I can't help with that." || resp.FinishReason != "stop" {
		t.Errorf("refusal: %+v, %v", resp, err)
	}
	_, events, err := readResponsesSSE(t, []string{
		`data: {"type":"response.refusal.delta","delta":"no"}`,
		`data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"no"}]}]}}`,
	})
	if err != nil || events[0].kind != evContent || events[0].text != "no" {
		t.Errorf("refusal stream: %+v, %v", events, err)
	}
}

func TestRenderInput_RolesAndParts(t *testing.T) {
	got, _ := json.Marshal(renderInput([]Message{
		{Role: RoleAssistant, Parts: []Part{{Kind: PartText, Text: "earlier"}}},
		{Role: RoleTool, ToolCallID: "c", Parts: []Part{{Kind: PartText, Text: "a"}, {Kind: PartText, Text: "b"}}},
	}))
	want := `[{"content":[{"text":"earlier","type":"output_text"}],"role":"assistant"},` +
		`{"call_id":"c","output":"ab","type":"function_call_output"}]`
	if string(got) != want {
		t.Errorf("input =\n%s\nwant\n%s", got, want)
	}
}

func TestPlanResponses_HistoryFromOtherModels(t *testing.T) {
	c := testClient(t, nil)
	mimoTurn := Message{Role: RoleAssistant, Text: "hi", ReasoningContent: "thinking"}
	pl, ws, err := c.plan(solRequest(User("x"), mimoTurn, User("y")), false)
	if err != nil {
		t.Fatal(err)
	}
	if pl.req.Messages[1].ReasoningContent != "" || !strings.Contains(Warnings(ws), "ReasoningContent was dropped") {
		t.Errorf("ReasoningContent kept or unwarned: %+v, %s", pl.req.Messages[1], Warnings(ws))
	}
	imageTurn := Message{Role: RoleAssistant, Parts: []Part{{Kind: PartImage, URL: "data:x"}}}
	if _, err := c.Validate(solRequest(User("x"), imageTurn)); err == nil {
		t.Error("an image on an assistant turn was accepted")
	}
	nullItem := Message{Role: RoleAssistant, ResponseItems: []json.RawMessage{json.RawMessage("null")}}
	if _, err := c.Validate(solRequest(User("x"), nullItem)); err == nil {
		t.Error("a null ResponseItem was accepted")
	}
	// The Responses cap spelling is not a Chat Completions key either.
	if _, err := c.Validate(ChatRequest{Model: "gpt-5.5", Messages: []Message{User("x")},
		ExtraBody: map[string]any{"max_output_tokens": 500}}); err == nil {
		t.Error("ExtraBody max_output_tokens accepted on a chat_completions model")
	}
}
