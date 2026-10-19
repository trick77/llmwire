package llmwire

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

// TestProbe_SolResponsesToolLoop is the Responses protocol end to end on a
// deployment's gateway route: a tool round that makes the model think, the
// round that replays its reasoning, and the same round streamed. It runs
// through FromEnv, so it needs exactly what an application supplies:
// LLMWIRE_LITELLM_BASE_URL, LLMWIRE_LITELLM_API_KEY, and
// LLMWIRE_LITELLM_MODELS listing gpt-6.1-sol. The figures it checks are the
// ones measured by hand on 2026-10-19 (responses.go head comment).
func TestProbe_SolResponsesToolLoop(t *testing.T) {
	evalGate(t)
	const model = "gpt-6.1-sol"
	c, err := FromEnv(model, Config{
		// Behind LiteLLM with include_cost_in_streaming_usage the stream is
		// buffered: the first byte is the whole answer, so the header bound
		// covers the call.
		HeaderTimeout: 3 * time.Minute,
		IdleTimeout:   90 * time.Second,
		CallTimeout:   3 * time.Minute,
	})
	if err != nil {
		t.Skipf("live probe: %v", err)
	}
	ctx := context.Background()
	req := ChatRequest{
		Model: model,
		// A system turn first, as noergler's mention loop opens: the role is
		// UNMEASURED on /responses until this runs. Then arithmetic, so the
		// model reasons: a trivial prompt reported reasoning_tokens 0 even at
		// effort high.
		Messages: []Message{System("You are a terse assistant. Use tools when asked."), User("Think carefully first: take the number of cantons in Switzerland, " +
			"multiply by the number of letters in its capital, subtract 70. If the result is even, the city " +
			"is Bern, otherwise Basel. Then call get_weather for that city. Do not answer in prose.")},
		Tools: []Tool{{Name: "get_weather", Description: "Current weather for a city",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required": []any{"city"}}}},
		ToolChoice: ToolChoice{Mode: ToolChoiceAuto},
		Reasoning:  ReasoningEffort("high"),
		MaxTokens:  ptr(4000),
	}

	theMeter.reserve(t, model)
	r1, _, err := c.Chat(ctx, req)
	if err != nil {
		t.Fatalf("FINDING round 1 REJECTED: %v", err)
	}
	theMeter.record(model, r1.Usage)
	t.Logf("round 1: calls %+v, items %d, usage in %d out %d reasoning %d, cost %+v, gateway %s",
		r1.ToolCalls, len(r1.ResponseItems), Tokens(r1.Usage.Input.Total), Tokens(r1.Usage.Output.Total),
		Tokens(r1.Usage.Output.Reasoning), r1.Usage.Cost, r1.Gateway.ModelName)
	if len(r1.ToolCalls) != 1 || r1.FinishReason != "tool_calls" {
		t.Fatalf("FINDING round 1: no single tool call, finish %q", r1.FinishReason)
	}
	if Tokens(r1.Usage.Output.Reasoning) == 0 || len(r1.ResponseItems) < 2 {
		t.Errorf("FINDING round 1: no reasoning to replay (reasoning_tokens %d, items %d)",
			Tokens(r1.Usage.Output.Reasoning), len(r1.ResponseItems))
	}

	req.Messages = append(req.Messages, r1.Assistant(),
		ToolResult(r1.ToolCalls[0].ID, `{"city":"Bern","temp_c":14,"sky":"overcast"}`))
	theMeter.reserve(t, model)
	r2, _, err := c.Chat(ctx, req)
	if err != nil {
		t.Fatalf("FINDING round 2 (reasoning replay) REJECTED: %v", err)
	}
	theMeter.record(model, r2.Usage)
	t.Logf("round 2: content %q, finish %q, usage in %d out %d reasoning %d, cost %+v",
		r2.Content, r2.FinishReason, Tokens(r2.Usage.Input.Total), Tokens(r2.Usage.Output.Total),
		Tokens(r2.Usage.Output.Reasoning), r2.Usage.Cost)
	if r2.Content == "" || Tokens(r2.Usage.Input.Total) <= Tokens(r1.Usage.Input.Total) {
		t.Errorf("FINDING round 2: content %q, input tokens %d after %d; the replay did not land",
			r2.Content, Tokens(r2.Usage.Input.Total), Tokens(r1.Usage.Input.Total))
	}
	if r1.Usage.Cost.Provenance != Reported || r2.Usage.Cost.Provenance != Reported {
		t.Errorf("FINDING non-streamed calls unpriced: %v / %v", r1.Usage.Cost.Provenance, r2.Usage.Cost.Provenance)
	}

	req.Messages = req.Messages[:2]
	theMeter.reserve(t, model)
	s, _, err := c.ChatStream(ctx, req)
	if err != nil {
		t.Fatalf("FINDING stream REJECTED: %v", err)
	}
	defer func() { _ = s.Close() }()
	res, err := s.Collect(nil)
	theMeter.record(model, res.Usage)
	if err != nil {
		t.Fatalf("FINDING stream failed: %v", err)
	}
	// FirstData close to Total is the buffered gateway; far below it is a
	// stream passed through.
	t.Logf("stream: calls %+v, items %d, first data %s of %s total, events %d, cost %+v",
		res.ToolCalls, len(res.ResponseItems), res.Timing.FirstData, res.Timing.Total, res.Events, res.Usage.Cost)
	var args map[string]any
	if len(res.ToolCalls) != 1 || json.Unmarshal([]byte(res.ToolCalls[0].Arguments), &args) != nil {
		t.Errorf("FINDING stream: tool call %+v does not carry one JSON argument object", res.ToolCalls)
	}
	if res.Usage.Cost.Provenance != Reported {
		t.Errorf("FINDING stream unpriced: is include_cost_in_streaming_usage still set gateway-side?")
	}
}
