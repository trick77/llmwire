package llmwire

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

// Live probes.
//
// Each asserts one thing about a real endpoint and LOGS what it found, because
// the finding is the deliverable: these answers become the profile data and the
// comment text that explains it. A probe fails only when it cannot reach a
// conclusion at all.
//
// Run: LLMWIRE_EVAL=1 go test -run Probe -v ./...
//
// The three marked ARCHITECTURE decide type shapes rather than field values. A
// wrong guess on those is expensive — Message is the one type every consumer
// binds to — which is why they run before any of it is designed.

// --- ARCHITECTURE 1: does MiMo emit native tool calls, or inline XML? ---------
//
// loom and trift both carry ~270 lines of inline-XML recovery plus dual-channel
// stream gating, written because MiMo emitted tool calls as XML inside the
// content stream instead of populating tool_calls. The vendor's own chat
// template documents that XML as the model's native format — but SGLang runs a
// "mimo" tool-call parser server-side, whose job is to convert it into standard
// OpenAI tool_calls before it ever reaches a client.
//
// If the server parses it, that machinery is vestigial and the module carries a
// fraction of it. If it does not, it is core. Nothing short of asking settles
// it.
func TestProbe_MiMoToolCallFormat(t *testing.T) {
	c := mimoEndpoint.client(t)

	for _, model := range []string{"mimo-v2.5-pro", "mimo-v2.5"} {
		t.Run(model, func(t *testing.T) {
			b := probeBody(model, "What is the weather in Zurich? Use the tool.")
			b["tools"] = []any{map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "get_weather",
					"description": "Get the current weather for a city.",
					"parameters": map[string]any{
						"type":       "object",
						"properties": map[string]any{"city": map[string]any{"type": "string"}},
						"required":   []any{"city"},
					},
				},
			}}
			b["max_completion_tokens"] = 256

			res, err := stream(t, c, b)
			if ok, why := accepted(res, err); !ok {
				t.Fatalf("FINDING %s: tool offer REJECTED: %s", model, why)
			}

			native := len(res.ToolCalls) > 0
			inline := containsInlineToolMarkup(res.Content) || containsInlineToolMarkup(res.Reasoning)

			switch {
			case native && !inline:
				t.Logf("FINDING %s: NATIVE tool_calls (name=%q). The server-side parser converts; "+
					"inline-XML recovery is vestigial for this deployment.",
					model, res.ToolCalls[0].Name)
			case inline && !native:
				t.Logf("FINDING %s: INLINE XML in %s. Recovery is CORE; port loom's parser.",
					model, channelOf(res))
			case native && inline:
				t.Logf("FINDING %s: BOTH native tool_calls and inline markup present — "+
					"recovery must run even when tool_calls is populated.", model)
			default:
				t.Logf("FINDING %s: no tool call of either kind. finish_reason=%q content=%q "+
					"(model declined the tool; not evidence either way)",
					model, res.FinishReason, Truncate(res.Content, 200))
			}
		})
	}
}

// inlineToolMarkers are the two syntaxes loom observed in production. The second
// is undocumented and no prompt teaches it.
var inlineToolMarkers = []string{"<tool_call>", "<tool_invocation", "<function="}

func containsInlineToolMarkup(s string) bool {
	for _, m := range inlineToolMarkers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

func channelOf(res StreamResult) string {
	switch {
	case containsInlineToolMarkup(res.Content) && containsInlineToolMarkup(res.Reasoning):
		return "BOTH content and reasoning_content"
	case containsInlineToolMarkup(res.Reasoning):
		return "reasoning_content (not content)"
	default:
		return "content"
	}
}

// --- ARCHITECTURE 1b: the forced tool-free call --------------------------------
//
// TestProbe_MiMoToolCallFormat showed native tool_calls when tools ARE offered,
// which makes the inline-XML parser look vestigial. But loom's comment describes
// a narrower case: "the forced tool-free final-answer call regularly gets an
// inline tool call back instead of an answer, and leaving it ungated leaked the
// raw markup as the reply."
//
// That call offers NO tools at all. With no tools in the request there is no
// tool_calls field for a server-side parser to populate, so if the model still
// decides to call something, the markup has nowhere to go but the content
// stream. This is what decides whether stream GATING is still needed even when
// the parser is not.
func TestProbe_MiMoToolFreeCall(t *testing.T) {
	c := mimoEndpoint.client(t)

	for _, model := range []string{"mimo-v2.5-pro", "mimo-v2.5"} {
		t.Run(model, func(t *testing.T) {
			// Prompt shaped like the tail of a research turn: the model has
			// "gathered" material and is told to answer without tools. No tools
			// are offered.
			b := probeBody(model,
				"You previously searched the web and found that the Zurich tram network "+
					"has 15 lines. Now write the final answer for the user in one sentence. "+
					"Do not call any tool; there are no tools available.")
			b["max_completion_tokens"] = 256

			res, err := stream(t, c, b)
			if ok, why := accepted(res, err); !ok {
				t.Fatalf("FINDING %s: tool-free call REJECTED: %s", model, why)
			}

			inline := containsInlineToolMarkup(res.Content) || containsInlineToolMarkup(res.Reasoning)
			if inline {
				t.Logf("FINDING %s: inline tool markup LEAKED on a tool-free call, in %s. "+
					"=> stream gating is CORE; without it this reaches the user as raw XML.",
					model, channelOf(res))
				t.Logf("  content=%q", Truncate(res.Content, 300))
				return
			}
			t.Logf("FINDING %s: clean prose on a tool-free call, no markup "+
				"(finish_reason=%q, %d chars). => gating not reproduced here.",
				model, res.FinishReason, res.Chars)
		})
	}
}

// --- ARCHITECTURE 2: must reasoning_content be replayed in history? ----------
//
// A vendor issue reports HTTP 400 with "The reasoning_content in the thinking
// mode must be passed back to the API" when an assistant turn carrying
// tool_calls is replayed without its reasoning. loom builds exactly that
// history — Content plus ToolCalls, no reasoning — with thinking on, across
// multi-round tool loops, in production, and works.
//
// Both cannot be true of the same deployment. If replay IS required, Message has
// to plumb reasoning through history and every consumer inherits that; if it is
// not, the field stays optional.
func TestProbe_MiMoReasoningReplayRequirement(t *testing.T) {
	c := mimoEndpoint.client(t)
	const model = "mimo-v2.5-pro"

	tool := map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "get_weather",
			"description": "Get the current weather for a city.",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []any{"city"},
			},
		},
	}

	// Round one: get the model to call the tool.
	first := probeBody(model, "What is the weather in Zurich? Use the tool.")
	first["tools"] = []any{tool}
	first["max_completion_tokens"] = 256
	res1, err1 := stream(t, c, first)
	if ok, why := accepted(res1, err1); !ok {
		t.Fatalf("FINDING: first tool round REJECTED: %s", why)
	}
	if len(res1.ToolCalls) == 0 {
		t.Skipf("model returned no native tool call (content=%q); "+
			"this probe needs one to build the second turn", Truncate(res1.Content, 160))
	}
	call := res1.ToolCalls[0]

	// Round two: replay that turn WITHOUT reasoning_content, exactly as loom
	// builds it.
	assistant := map[string]any{
		"role":    "assistant",
		"content": res1.Content,
		"tool_calls": []any{map[string]any{
			"id":       call.ID,
			"type":     "function",
			"function": map[string]any{"name": call.Name, "arguments": call.Arguments},
		}},
	}
	second := body{
		"model": model,
		"messages": []any{
			map[string]any{"role": "user", "content": "What is the weather in Zurich? Use the tool."},
			assistant,
			map[string]any{"role": "tool", "tool_call_id": call.ID, "content": `{"temp_c":14,"sky":"overcast"}`},
		},
		"stream":                true,
		"tools":                 []any{tool},
		"max_completion_tokens": probeMaxTokens,
	}

	res2, err2 := stream(t, c, second)
	if ok, why := accepted(res2, err2); !ok {
		t.Logf("FINDING: replay WITHOUT reasoning_content was REJECTED: %s", why)
		t.Logf("  => Message MUST carry ReasoningContent through history, and the " +
			"assistant-turn builder must replay it. loom's history shape would be " +
			"broken on this deployment.")
		return
	}
	t.Logf("FINDING: replay WITHOUT reasoning_content ACCEPTED (finish_reason=%q). "+
		"=> No replay invariant on this deployment; the vendor issue does not apply here. "+
		"ReasoningContent stays optional on Message.", res2.FinishReason)
}

// --- ARCHITECTURE 3: how long does an endpoint go silent between data frames? -
//
// The idle guard deliberately does NOT re-arm on blank lines or ": ping"
// comments, because an upstream emitting them on a timer would otherwise mask a
// model that has stalled completely. The cost of that strictness is a false
// abort if an endpoint legitimately sends only comments for longer than the idle
// bound.
//
// peeq measured a deep call taking 69.9s against a 90s bound, which is not a
// wide margin. This measures the real gap on a deliberately slow call.
func TestProbe_LongestCommentOnlyGap(t *testing.T) {
	for _, tc := range []struct {
		ep    endpoint
		model string
		deep  body
	}{
		{mimoEndpoint, "mimo-v2.5-pro", body{"max_completion_tokens": 1024}},
		{zaiEndpoint, "glm-5.3-flash", body{"max_tokens": 1024, "reasoning_effort": "max"}},
	} {
		t.Run(tc.model, func(t *testing.T) {
			c := tc.ep.client(t)
			b := probeBody(tc.model,
				"Think carefully and at length about how a stream parser should distinguish "+
					"a stalled model from a slow one, then answer in one sentence.")
			for k, v := range tc.deep {
				b[k] = v
			}
			res, err := stream(t, c, b)
			if ok, why := accepted(res, err); !ok {
				t.Fatalf("FINDING %s: probe REJECTED: %s", tc.model, why)
			}
			t.Logf("FINDING %s: longest gap with no data frame = %v (over %d frames, %d chars). "+
				"Default idle bound is %v.",
				tc.model, res.MaxCommentGap.Round(time.Millisecond), res.Events, res.Chars, DefaultIdleTimeout)
			if res.MaxCommentGap > DefaultIdleTimeout/2 {
				t.Logf("  => WARNING: gap exceeds half the idle bound. The data-frames-only "+
					"re-arm rule needs a per-profile escape hatch for %s.", tc.model)
			}
		})
	}
}

// --- data sweep: one request each, same run ------------------------------------

// Z.ai returns code 1210 for "this model always thinks" on glm-5.3-flash, with
// the live body in Chinese. Its own API reference documents
// thinking:{"type":"disabled"} as valid, and models.dev lists effort values with
// no "none". Three sources, three answers.
func TestProbe_ZaiThinkingCanBeDisabled(t *testing.T) {
	c := zaiEndpoint.client(t)
	b := probeBody("glm-5.3-flash", "Reply with the single word: ok")
	b["thinking"] = map[string]any{"type": "disabled"}
	b["max_tokens"] = probeMaxTokens

	res, err := stream(t, c, b)
	if ok, why := accepted(res, err); !ok {
		t.Logf("FINDING: thinking:disabled REJECTED on glm-5.3-flash: %s", why)
		t.Logf("  => can_be_disabled: false. ReasoningOff() must fail fast for this model.")
		return
	}
	t.Logf("FINDING: thinking:disabled ACCEPTED on glm-5.3-flash "+
		"(reasoning tokens reported: %v). => can_be_disabled: true; peeq's 1210 no longer reproduces.",
		valueOr(res.Usage.Output.Reasoning, -1))
}

// The accepted set is the profile's effort_values, and it is per-model: the
// global enum is wider than any single model takes.
func TestProbe_ZaiReasoningEffortValues(t *testing.T) {
	c := zaiEndpoint.client(t)
	for _, effort := range []string{"none", "minimal", "low", "medium", "high", "max", "xhigh"} {
		t.Run(effort, func(t *testing.T) {
			b := probeBody("glm-5.3-flash", "Reply with the single word: ok")
			b["reasoning_effort"] = effort
			b["max_tokens"] = probeMaxTokens
			res, err := stream(t, c, b)
			ok, why := accepted(res, err)
			if !ok {
				t.Logf("FINDING: reasoning_effort=%q REJECTED: %s", effort, why)
				return
			}
			t.Logf("FINDING: reasoning_effort=%q ACCEPTED (reasoning tokens: %v)",
				effort, valueOr(res.Usage.Output.Reasoning, -1))
		})
	}
}

// The vendor doc puts reasoning_effort on the Responses API only, yet loom sends
// it to chat/completions on every main turn. Either it is accepted-and-ignored,
// or the doc is incomplete.
func TestProbe_MiMoAcceptsReasoningEffort(t *testing.T) {
	c := mimoEndpoint.client(t)
	b := probeBody("mimo-v2.5-pro", "Reply with the single word: ok")
	b["reasoning_effort"] = "high"
	b["max_completion_tokens"] = probeMaxTokens

	res, err := stream(t, c, b)
	if ok, why := accepted(res, err); !ok {
		t.Logf("FINDING: reasoning_effort REJECTED by mimo-v2.5-pro on chat/completions: %s", why)
		t.Logf("  => loom has been sending a parameter this endpoint refuses.")
		return
	}
	t.Logf("FINDING: reasoning_effort ACCEPTED by mimo-v2.5-pro (reasoning tokens: %v). "+
		"Accepted-and-possibly-ignored; the vendor doc is incomplete.",
		valueOr(res.Usage.Output.Reasoning, -1))
}

// Which levels MiMo takes beside its thinking toggle, and whether they move
// anything. loom offers low/medium/high to its users; the profile's
// effort_values must be the set the endpoint accepts, and the reasoning-token
// counts per level are the evidence for whether the knob does anything. A
// prompt that needs some thinking, so a level that is honoured has room to
// show: at a trivial prompt every level reasons the same handful of tokens.
func TestProbe_MiMoReasoningEffortValues(t *testing.T) {
	c := mimoEndpoint.client(t)
	// "" is the control: the same prompt with no level sent, so a model that
	// stops thinking when a level arrives is told apart from one that never
	// thought about this prompt.
	for _, model := range []string{"mimo-v2.5-pro", "mimo-v2.5"} {
		for _, effort := range []string{"", "low", "medium", "high", "xhigh"} {
			t.Run(model+"/"+effort, func(t *testing.T) {
				b := probeBody(model, "A train leaves at 09:40 and arrives at 13:05 the same day. "+
					"How many minutes is the journey? Reply with the number only.")
				if effort != "" {
					b["reasoning_effort"] = effort
				}
				b["max_completion_tokens"] = 4096
				res, err := stream(t, c, b)
				ok, why := accepted(res, err)
				if !ok {
					t.Logf("FINDING: %s reasoning_effort=%q REJECTED: %s", model, effort, why)
					return
				}
				t.Logf("FINDING: %s reasoning_effort=%q ACCEPTED (reasoning tokens: %v, completion: %v, finish: %s, content: %q)",
					model, effort, valueOr(res.Usage.Output.Reasoning, -1), valueOr(res.Usage.Output.Total, -1),
					res.FinishReason, Truncate(res.Content, 40))
			})
		}
	}
}

// Which output-cap parameter each endpoint honours. Sending the wrong one is
// silently unbounded on some endpoints and a 400 on others.
func TestProbe_OutputCapParameter(t *testing.T) {
	for _, tc := range []struct {
		ep    endpoint
		model string
	}{
		{mimoEndpoint, "mimo-v2.5-pro"},
		{zaiEndpoint, "glm-5.3-flash"},
	} {
		for _, param := range []string{"max_tokens", "max_completion_tokens"} {
			t.Run(tc.model+"/"+param, func(t *testing.T) {
				c := tc.ep.client(t)
				b := probeBody(tc.model, "Count slowly from one to fifty in words.")
				b[param] = 16
				res, err := stream(t, c, b)
				ok, why := accepted(res, err)
				if !ok {
					t.Logf("FINDING: %s REJECTS %s: %s", tc.model, param, why)
					return
				}
				honoured := res.FinishReason == "length" || valueOr(res.Usage.Output.Total, 0) <= 32
				t.Logf("FINDING: %s accepts %s; honoured=%v (finish_reason=%q, completion tokens=%v)",
					tc.model, param, honoured, res.FinishReason, valueOr(res.Usage.Output.Total, -1))
			})
		}
	}
}

// Whether a usage frame arrives at all without asking, and where in the sequence
// it lands. The parser handles both positions; this records which each endpoint
// uses, and whether stream_options is needed.
func TestProbe_UsageReportingWithoutStreamOptions(t *testing.T) {
	for _, tc := range []struct {
		ep    endpoint
		model string
		cap   string
	}{
		{mimoEndpoint, "mimo-v2.5-pro", "max_completion_tokens"},
		{zaiEndpoint, "glm-5.3-flash", "max_tokens"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			c := tc.ep.client(t)

			b := probeBody(tc.model, "Reply with the single word: ok")
			b[tc.cap] = probeMaxTokens
			res, err := stream(t, c, b)
			if ok, why := accepted(res, err); !ok {
				t.Fatalf("FINDING %s: REJECTED: %s", tc.model, why)
			}
			without := res.Usage.Input.Total != nil
			t.Logf("FINDING %s: usage WITHOUT stream_options: reported=%v (raw=%s)",
				tc.model, without, Truncate(string(res.Usage.Raw), 240))

			b2 := probeBody(tc.model, "Reply with the single word: ok")
			b2[tc.cap] = probeMaxTokens
			b2["stream_options"] = map[string]any{"include_usage": true}
			res2, err2 := stream(t, c, b2)
			ok2, why2 := accepted(res2, err2)
			if !ok2 {
				t.Logf("FINDING %s: stream_options REJECTED: %s => must not be sent", tc.model, why2)
				return
			}
			t.Logf("FINDING %s: stream_options accepted; usage reported=%v => needs_include_usage=%v",
				tc.model, res2.Usage.Input.Total != nil, !without)
		})
	}
}

// json_schema is absent from Z.ai's OpenAPI enum and from its docs, while
// OpenRouter claims support. Nobody has tested it.
func TestProbe_ResponseFormatSupport(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"answer": map[string]any{"type": "string"}},
		"required":             []any{"answer"},
		"additionalProperties": false,
	}
	for _, tc := range []struct {
		ep    endpoint
		model string
		cap   string
	}{
		{zaiEndpoint, "glm-5.3-flash", "max_tokens"},
		{mimoEndpoint, "mimo-v2.5-pro", "max_completion_tokens"},
	} {
		for _, format := range []struct {
			name string
			val  map[string]any
		}{
			{"json_object", map[string]any{"type": "json_object"}},
			{"json_schema", map[string]any{
				"type":        "json_schema",
				"json_schema": map[string]any{"name": "probe", "strict": true, "schema": schema},
			}},
		} {
			t.Run(tc.model+"/"+format.name, func(t *testing.T) {
				c := tc.ep.client(t)
				b := probeBody(tc.model, `Answer as JSON: {"answer":"ok"}`)
				b["response_format"] = format.val
				b[tc.cap] = 128
				res, err := stream(t, c, b)
				ok, why := accepted(res, err)
				if !ok {
					t.Logf("FINDING: %s REJECTS response_format %s: %s", tc.model, format.name, why)
					return
				}
				parses := json.Valid([]byte(strings.TrimSpace(res.Content)))
				t.Logf("FINDING: %s accepts response_format %s; raw reply parses as JSON=%v",
					tc.model, format.name, parses)
			})
		}
	}
}

// mimo-v2.5-pro is text-only and 404s on any image_url part, while mimo-v2.5
// takes images. loom routes around this; the module errors instead, so the
// profile bit has to be right.
func TestProbe_MiMoProRejectsImageInput(t *testing.T) {
	c := mimoEndpoint.client(t)
	// A 1x1 transparent PNG, small enough to be inline and large enough to be a
	// real image part.
	const onePixelPNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

	for _, model := range []string{"mimo-v2.5-pro", "mimo-v2.5"} {
		t.Run(model, func(t *testing.T) {
			b := body{
				"model": model,
				"messages": []any{map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "What colour is this pixel?"},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": onePixelPNG}},
					},
				}},
				"stream":                true,
				"max_completion_tokens": probeMaxTokens,
			}
			res, err := stream(t, c, b)
			if ok, why := accepted(res, err); !ok {
				t.Logf("FINDING: %s REJECTS image input: %s => vision: false", model, why)
				return
			}
			t.Logf("FINDING: %s ACCEPTS image input (finish_reason=%q) => vision: true",
				model, res.FinishReason)
		})
	}
}

// The regression test for the trap this library exists to defuse, driven through
// the PUBLIC call surface rather than a hand-built body.
//
// glm-5.3-flash accepts max_completion_tokens and silently ignores it: the probe
// above measured a cap of 16 returning 481 completion tokens with finish_reason
// "stop". Chat writes the cap once and the profile chooses the spelling, so this
// asserts the OUTCOME rather than the parameter name — if the profile, the plan or
// the renderer regresses, the count comes back over the cap and this fails.
func TestProbe_ChatHonoursTheOutputCap(t *testing.T) {
	for _, tc := range []struct {
		ep    endpoint
		model string
	}{
		{zaiEndpoint, "glm-5.3-flash"},
		{mimoEndpoint, "mimo-v2.5-pro"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			c := tc.ep.client(t)
			theMeter.reserve(t, tc.model)

			capTokens := probeMaxTokens
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()

			resp, warnings, err := c.Chat(ctx, ChatRequest{
				Model:     tc.model,
				Messages:  []Message{User("Count slowly from one to fifty in words.")},
				MaxTokens: &capTokens,
			})
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			theMeter.record(tc.model, resp.Usage)
			for _, w := range warnings {
				t.Logf("warning: %s", w)
			}

			got := valueOr(resp.Usage.Output.Total, -1)
			t.Logf("FINDING: %s capped at %d returned %d completion tokens "+
				"(finish_reason=%q, cost=%d nano-USD via %s)",
				tc.model, capTokens, got, resp.FinishReason,
				resp.Usage.Cost.NanoUSD, resp.Usage.Cost.Provenance)
			if got < 0 {
				t.Fatalf("no completion tokens reported, so the cap cannot be verified")
			}
			// Some allowance over the cap: endpoints round to a token boundary and
			// one of them counts reasoning against the same budget. Thirty times
			// over, which is what the wrong parameter name produced, is not
			// rounding.
			if got > int64(capTokens)*2 {
				t.Errorf("cap of %d was IGNORED: %d completion tokens came back. The output-cap "+
					"parameter for this model is wrong in profiles.yaml, or the renderer stopped "+
					"reading it", capTokens, got)
			}
		})
	}
}

// Cached prompt tokens: are they INSIDE prompt_tokens, or beside it?
//
// Pricing subtracts cached_tokens from prompt_tokens to get the full-rate lane,
// on the strength of the OpenAI shape, where prompt_tokens contains them. The
// measured MiMo figures (258 prompt, 192 cached) are consistent with that and do
// not prove it: a vendor reporting cached tokens BESIDE prompt_tokens, the way
// Anthropic does, would make that subtraction under-count the full-rate lane —
// which is the one direction a budget cap cannot tolerate. Z.ai has never shown
// a non-zero cached_tokens at all, so its field name is unconfirmed too.
//
// Two identical calls with a prompt past any cache minimum settle it. The prompt
// is the same bytes both times, so if the second call reports more cached tokens
// than the first and prompt_tokens does NOT move, cached tokens are inside it.
// If prompt_tokens drops by the cached delta, they are beside it.
func TestProbe_CachedTokensAreInsidePromptTokens(t *testing.T) {
	for _, tc := range []struct {
		ep    endpoint
		model string
		cap   string
	}{
		{mimoEndpoint, "mimo-v2.5-pro", "max_completion_tokens"},
		{zaiEndpoint, "glm-5.3-flash", "max_tokens"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			c := tc.ep.client(t)

			call := func() (StreamResult, int64, int64) {
				b := probeBody(tc.model, cacheProbePrompt())
				b[tc.cap] = probeMaxTokens
				res, err := stream(t, c, b)
				if ok, why := accepted(res, err); !ok {
					t.Fatalf("FINDING %s: REJECTED: %s", tc.model, why)
				}
				if res.Usage.Input.Total == nil {
					t.Fatalf("FINDING %s: no prompt_tokens reported (raw=%s); nothing to compare",
						tc.model, Truncate(string(res.Usage.Raw), 240))
				}
				return res, *res.Usage.Input.Total, valueOr(res.Usage.Input.CacheRead, 0)
			}

			first, p1, c1 := call()
			t.Logf("call 1: prompt=%d cached=%d raw=%s", p1, c1, Truncate(string(first.Usage.Raw), 240))
			// Caches populate asynchronously on some endpoints; a few seconds is
			// what their docs suggest and costs nothing.
			time.Sleep(5 * time.Second)
			second, p2, c2 := call()
			t.Logf("call 2: prompt=%d cached=%d raw=%s", p2, c2, Truncate(string(second.Usage.Raw), 240))

			delta := c2 - c1
			switch {
			// The two inconclusive outcomes SKIP rather than pass: a green result
			// must mean the containment rule was actually read off a delta.
			case c2 == 0:
				t.Skipf("FINDING %s: no cached tokens on the second call. Either the prompt (%d tokens) "+
					"is under the cache minimum, the cache is off for this route, or the field is "+
					"spelled differently — read the raw usage above for the vendor's own name. "+
					"Inconclusive; the profile keeps the OpenAI-shape assumption.", tc.model, p2)
			case delta <= 0:
				t.Skipf("FINDING %s: cached did not grow between calls (%d -> %d); the first call was "+
					"already served from cache, so containment cannot be read off a delta. prompt "+
					"%d -> %d. Inconclusive.", tc.model, c1, c2, p1, p2)
			case p2 == p1:
				t.Logf("FINDING %s: INSIDE. cached grew %d -> %d and prompt_tokens stayed at %d. "+
					"prompt_tokens contains cached_tokens; subtracting is correct.", tc.model, c1, c2, p1)
			case p1-p2 == delta:
				t.Errorf("FINDING %s: BESIDE. cached grew %d -> %d and prompt_tokens fell by exactly "+
					"that (%d -> %d). This vendor reports cached tokens outside prompt_tokens, so "+
					"NoCache = prompt - cached UNDER-COUNTS the full-rate lane here. Pricing needs a "+
					"per-profile containment flag before this model's cost figure can be trusted.",
					tc.model, c1, c2, p1, p2)
			default:
				t.Errorf("FINDING %s: AMBIGUOUS. cached grew %d -> %d, prompt_tokens moved %d -> %d, "+
					"which matches neither containment rule. Record both raw usages above and "+
					"look for a third field before touching pricing.", tc.model, c1, c2, p1, p2)
			}
		})
	}
}

// cacheProbeNonce makes the prompt unique to one process, so a rerun inside the
// vendor's cache TTL starts cold again instead of finding call 1 already cached
// and skipping. Fixed for the process, so the two calls within a run still share
// a cache key.
var cacheProbeNonce = time.Now().UnixNano()

// cacheProbePrompt is a fixed prompt comfortably past the documented cache
// minimums (1024 tokens on the OpenAI shape both vendors mirror). The same bytes
// on both calls of a run, so they share a cache key; numbered lines, so the
// tokenizer cannot collapse it into a repeat.
func cacheProbePrompt() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Measurement %d. The lines below are filler for a caching measurement. "+
		"Ignore them and reply with the single word: ok\n\n", cacheProbeNonce)
	for i := 1; i <= 400; i++ {
		fmt.Fprintf(&b, "Line %d: the quick brown fox number %d jumps over the lazy dog number %d.\n", i, i*7, i*13)
	}
	b.WriteString("\nReply with the single word: ok")
	return b.String()
}
