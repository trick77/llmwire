package llmwire

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

// A role check at boot: "the vision lane needs a model that takes images" fails
// with the model, what it lacks and what would do, before the first request.

func TestRegistry_Require(t *testing.T) {
	reg := Default()

	p, err := reg.Require("mimo-v2.6-flash", Needs{Tools: true, Vision: true, JSONObject: true})
	if err != nil || p.ID != "mimo-v2.6-flash" {
		t.Fatalf("got %v, %v", p, err)
	}

	_, err = reg.Require("mimo-v2.5-pro", Needs{Vision: true, Tools: true})
	var ce *CapabilityError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v, want *CapabilityError", err)
	}
	if ce.Model != "mimo-v2.5-pro" || !slices.Equal(ce.Missing, []string{"vision"}) {
		t.Errorf("error = %+v", ce)
	}
	if !slices.Contains(ce.Satisfying, "mimo-v2.6-pro") || slices.Contains(ce.Satisfying, "mimo-v2.5-pro") {
		t.Errorf("satisfying = %v", ce.Satisfying)
	}
	for _, want := range []string{`"mimo-v2.5-pro"`, "vision", "mimo-v2.6-pro"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("message %q does not name %s", err, want)
		}
	}

	// An embeddings model has no chat role to play.
	if _, err := reg.Require("text-embedding-3-small", Needs{}); err == nil || !strings.Contains(err.Error(), "chat") {
		t.Errorf("embeddings model: err = %v", err)
	}
	var unknown *UnknownModelError
	if _, err := reg.Require("nope", Needs{}); !errors.As(err, &unknown) {
		t.Errorf("unknown model: err = %v", err)
	}
}

func TestRegistry_ChatModels(t *testing.T) {
	reg := Default()
	all := reg.ChatModels(Needs{})
	if slices.Contains(all, "text-embedding-3-small") || !slices.Contains(all, "glm-5.3-flash") {
		t.Errorf("ChatModels = %v", all)
	}
	if !slices.IsSorted(all) {
		t.Errorf("not sorted: %v", all)
	}
	vision := reg.ChatModels(Needs{Vision: true})
	if slices.Contains(vision, "mimo-v2.5-pro") {
		t.Errorf("mimo-v2.5-pro 404s on images: %v", vision)
	}

	strict := registryFrom(t, chatHead)
	for _, n := range []Needs{{Tools: true}, {Vision: true}, {JSONObject: true}, {JSONSchema: true}, {Streaming: true}} {
		if got := strict.ChatModels(n); len(got) != 0 {
			t.Errorf("%+v: a bare profile satisfies it: %v", n, got)
		}
	}
}
