package llmwiretest_test

import (
	"context"
	"testing"

	"github.com/trick77/llmwire"
	"github.com/trick77/llmwire/llmwiretest"
)

// What a consuming module's test looks like: intents in, intents asserted, no
// real model id, wire spelling or rate anywhere.

// The recorded request reads back in ReasoningSent's vocabulary, every
// variant, on both reasoning controls the synthetic models use.
func TestRecordedReasoningMatchesReasoningSent(t *testing.T) {
	srv := llmwiretest.NewServer(t)
	c := srv.Client()
	for _, tc := range []struct {
		model string
		r     llmwire.ReasoningRequest
	}{
		{llmwiretest.ChatModel, nil},
		{llmwiretest.ChatModel, llmwire.ReasoningOff()},
		{llmwiretest.ChatModel, llmwire.ReasoningEffort("none")},
		{llmwiretest.ChatModel, llmwire.ReasoningEffort("high")},
		{llmwiretest.ChatModel, llmwire.ReasoningMinimal()},
		{llmwiretest.ChatModel, llmwire.ReasoningBalanced()},
		{llmwiretest.BudgetModel, llmwire.ReasoningBudget(300)},
		{llmwiretest.BudgetModel, llmwire.ReasoningBudget(0)},
		{llmwiretest.BudgetModel, llmwire.ReasoningOff()},
		{llmwiretest.BudgetModel, nil},
	} {
		resp, _, err := c.Chat(context.Background(), llmwire.ChatRequest{
			Model: tc.model, Messages: []llmwire.Message{llmwire.User("hi")}, Reasoning: tc.r,
		})
		if err != nil {
			t.Fatalf("%s %#v: %v", tc.model, tc.r, err)
		}
		if got := srv.Last().Reasoning(); got != resp.ReasoningSent {
			t.Errorf("%s %#v: recorded %q, ReasoningSent %q", tc.model, tc.r, got, resp.ReasoningSent)
		}
	}
}

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
	if err != nil || len(models) != 4 {
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

// A loop that appends resp.Assistant() replays the reasoning on the Responses
// protocol; one that rebuilds the turn from Content does not, and the fake
// tells the two apart.
func TestResponsesModelReplay(t *testing.T) {
	srv := llmwiretest.NewServer(t)
	c := srv.Client()
	ctx := context.Background()
	req := llmwire.ChatRequest{Model: llmwiretest.ResponsesModel, Messages: []llmwire.Message{llmwire.User("hi")},
		Reasoning: llmwire.ReasoningEffort("high")}
	resp, _, err := c.Chat(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if srv.Last().Path != "/responses" || srv.Last().Reasoning() != resp.ReasoningSent || resp.Content != llmwiretest.Reply {
		t.Fatalf("path %s, reasoning %q vs %q, content %q", srv.Last().Path, srv.Last().Reasoning(), resp.ReasoningSent, resp.Content)
	}

	req.Messages = append(req.Messages, resp.Assistant(), llmwire.User("again"))
	if _, _, err := c.Chat(ctx, req); err != nil || !srv.Last().ReplayedReasoning() {
		t.Fatalf("Assistant() turn: err %v, replayed %v", err, srv.Last().ReplayedReasoning())
	}
	req.Messages[1] = llmwire.Message{Role: llmwire.RoleAssistant, Text: resp.Content}
	if _, _, err := c.Chat(ctx, req); err != nil || srv.Last().ReplayedReasoning() {
		t.Fatalf("rebuilt turn: err %v, replayed %v", err, srv.Last().ReplayedReasoning())
	}

	s, _, err := c.ChatStream(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	res, err := s.Collect(nil)
	if err != nil || res.Content != llmwiretest.Reply || len(res.Assistant().ResponseItems) != 2 {
		t.Fatalf("stream: %+v, %v", res, err)
	}
}

// The same on the Messages protocol: the signed thinking block goes back with
// Assistant(), and not with a rebuilt turn.
func TestMessagesModelReplay(t *testing.T) {
	srv := llmwiretest.NewServer(t)
	c := srv.Client()
	ctx := context.Background()
	answer := 100
	req := llmwire.ChatRequest{Model: llmwiretest.MessagesModel, Messages: []llmwire.Message{llmwire.User("hi")},
		Reasoning: llmwire.ReasoningBalanced(), MaxAnswerTokens: &answer}
	resp, _, err := c.Chat(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := srv.Last().MaxTokens(); srv.Last().Path != "/messages" || srv.Last().Reasoning() != resp.ReasoningSent ||
		resp.Content != llmwiretest.Reply || n != 100+llmwiretest.BalancedOverhead {
		t.Fatalf("path %s, reasoning %q vs %q, content %q, cap %d", srv.Last().Path, srv.Last().Reasoning(),
			resp.ReasoningSent, resp.Content, n)
	}

	req.Messages = append(req.Messages, resp.Assistant(), llmwire.User("again"))
	if _, _, err := c.Chat(ctx, req); err != nil || !srv.Last().ReplayedThinking() {
		t.Fatalf("Assistant() turn: err %v, replayed %v", err, srv.Last().ReplayedThinking())
	}
	req.Messages[1] = llmwire.Message{Role: llmwire.RoleAssistant, Text: resp.Content}
	if _, _, err := c.Chat(ctx, req); err != nil || srv.Last().ReplayedThinking() {
		t.Fatalf("rebuilt turn: err %v, replayed %v", err, srv.Last().ReplayedThinking())
	}

	s, _, err := c.ChatStream(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	res, err := s.Collect(nil)
	if err != nil || res.Content != llmwiretest.Reply || len(res.Assistant().ResponseItems) != 2 {
		t.Fatalf("stream: %+v, %v", res, err)
	}
}
