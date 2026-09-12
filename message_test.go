package llmwire

import (
	"reflect"
	"testing"
)

func TestMessageConstructors(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  Message
		want Message
	}{
		{"system", System("be brief"), Message{Role: RoleSystem, Text: "be brief"}},
		{"user", User("hello"), Message{Role: RoleUser, Text: "hello"}},
		{"assistant", Assistant("hi"), Message{Role: RoleAssistant, Text: "hi"}},
		{
			"tool result",
			ToolResult("call_1", `{"temp":14}`),
			Message{Role: RoleTool, ToolCallID: "call_1", Text: `{"temp":14}`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// DeepEqual rather than !=: Message carries slices, so it is
			// not comparable.
			if !reflect.DeepEqual(tc.got, tc.want) {
				t.Errorf("got %+v, want %+v", tc.got, tc.want)
			}
		})
	}
}

// HasImage is what validation checks against a model's vision capability, so it
// has to be exact: a text part that merely mentions an image is not one.
func TestMessage_HasImage(t *testing.T) {
	for _, tc := range []struct {
		name string
		msg  Message
		want bool
	}{
		{"plain text", User("a picture of a cat"), false},
		{
			"text parts only",
			Message{Role: RoleUser, Parts: []Part{{Kind: PartText, Text: "image_url"}}},
			false,
		},
		{
			"an image part",
			Message{Role: RoleUser, Parts: []Part{
				{Kind: PartText, Text: "what is this?"},
				{Kind: PartImage, URL: "data:image/png;base64,AAAA"},
			}},
			true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.msg.HasImage(); got != tc.want {
				t.Errorf("HasImage() = %v, want %v", got, tc.want)
			}
		})
	}
}

// The reasoning request variants are a closed set: only this package can
// implement the interface, so a caller cannot invent a fourth control the
// profiles know nothing about.
func TestReasoningRequestVariantsAreDistinct(t *testing.T) {
	off, effort, budget := ReasoningOff(), ReasoningEffort("high"), ReasoningBudget(512)

	if _, ok := off.(reasoningOff); !ok {
		t.Errorf("ReasoningOff() = %T", off)
	}
	e, ok := effort.(reasoningEffort)
	if !ok || e.level != "high" {
		t.Errorf("ReasoningEffort() = %#v", effort)
	}
	b, ok := budget.(reasoningBudget)
	if !ok || b.tokens != 512 {
		t.Errorf("ReasoningBudget() = %#v", budget)
	}
}
