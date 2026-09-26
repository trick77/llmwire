package llmwiretest_test

import (
	"context"
	"testing"

	"github.com/trick77/llmwire"
	"github.com/trick77/llmwire/llmwiretest"
)

// What a consuming module's test looks like: intents in, intents asserted, no
// real model id, wire spelling or rate anywhere.

func TestIntentsAreObservable(t *testing.T) {
	srv := llmwiretest.NewServer(t)
	c := srv.Client()
	ctx := context.Background()
	answer := 64

	resp, warnings, err := c.Chat(ctx, llmwire.ChatRequest{
		Model:           llmwiretest.ChatModel,
		Messages:        []llmwire.Message{llmwire.User("title this")},
		Reasoning:       llmwire.ReasoningMinimal(),
		MaxAnswerTokens: &answer,
	})
	if err != nil || len(warnings) != 0 {
		t.Fatalf("Chat: %v %v", err, warnings)
	}
	if resp.Content != llmwiretest.Reply || resp.ReasoningSent != llmwiretest.MinimalSent {
		t.Errorf("response %q, reasoning sent %q", resp.Content, resp.ReasoningSent)
	}
	last := srv.Last()
	if got := last.Reasoning(); got != llmwiretest.MinimalSent {
		t.Errorf("wire reasoning %q, want %q", got, llmwiretest.MinimalSent)
	}
	if got, ok := last.MaxTokens(); !ok || got != answer+llmwiretest.MinimalOverhead {
		t.Errorf("wire cap %d, want %d", got, answer+llmwiretest.MinimalOverhead)
	}
	if !resp.Usage.Reported() || resp.Usage.Cost.Provenance == llmwire.Unpriced {
		t.Errorf("usage %+v: the fake reports usage and the synthetic model is priced", resp.Usage)
	}

	srv.SetReply("streamed")
	s, _, err := c.ChatStream(ctx, llmwire.ChatRequest{
		Model:           llmwiretest.ChatModel,
		Messages:        []llmwire.Message{llmwire.User("hi")},
		Reasoning:       llmwire.ReasoningBalanced(),
		MaxAnswerTokens: &answer,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	res, err := s.Collect(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "streamed" || res.ReasoningSent != llmwiretest.BalancedSent {
		t.Errorf("stream %q, reasoning sent %q", res.Content, res.ReasoningSent)
	}
	last = srv.Last()
	if !last.Stream() || last.Reasoning() != llmwiretest.BalancedSent || last.Model() != llmwiretest.ChatModel {
		t.Errorf("recorded %+v", last.Body)
	}
	if got, _ := last.MaxTokens(); got != answer+llmwiretest.BalancedOverhead {
		t.Errorf("wire cap %d, want %d", got, answer+llmwiretest.BalancedOverhead)
	}
}

func TestEmbeddingsAndListing(t *testing.T) {
	srv := llmwiretest.NewServer(t)
	c := srv.Client()
	resp, _, err := c.Embed(context.Background(), llmwire.EmbedRequest{Model: llmwiretest.EmbedModel, Inputs: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Vectors) != 2 || len(resp.Vectors[0]) != llmwiretest.EmbedDimensions {
		t.Errorf("vectors %v", resp.Vectors)
	}
	models, _, err := c.ListModels(context.Background())
	if err != nil || len(models) != 2 {
		t.Errorf("models %v, %v", models, err)
	}
	if n := len(srv.Requests()); n != 2 {
		t.Errorf("recorded %d requests, want 2", n)
	}
}

// Code that builds its own client from the environment runs unchanged against
// the fake.
func TestFromEnvModelsAgainstTheFake(t *testing.T) {
	srv := llmwiretest.NewServer(t)
	c, err := llmwire.FromEnvModels(llmwire.Config{Registry: llmwiretest.Registry(), Lookup: srv.Lookup},
		llmwiretest.ChatModel, llmwiretest.EmbedModel)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Chat(context.Background(), llmwire.ChatRequest{
		Model: llmwiretest.ChatModel, Messages: []llmwire.Message{llmwire.User("hi")},
	}); err != nil {
		t.Fatal(err)
	}
	if got := srv.Last().Header.Get("Authorization"); got == "" {
		t.Error("no key sent")
	}
	// The document composes into a registry of the caller's own.
	if _, err := llmwire.NewRegistry(llmwiretest.Profiles()); err != nil {
		t.Errorf("Profiles(): %v", err)
	}
	p, err := llmwiretest.Registry().Require(llmwiretest.ChatModel, llmwire.Needs{Tools: true, Vision: true, JSONSchema: true, Streaming: true})
	if err != nil || p.DisplayName == "" {
		t.Errorf("ChatModel should fill every role: %v", err)
	}
}
