package llmwire

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
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

	for _, model := range []string{"mimo-v2.6-pro", "mimo-v2.6-flash"} {
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

	for _, model := range []string{"mimo-v2.6-pro", "mimo-v2.6-flash"} {
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
	const model = "mimo-v2.6-pro"

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
		{mimoEndpoint, "mimo-v2.6-pro", body{"max_completion_tokens": 1024}},
		{mimoEndpoint, "mimo-v2.6-flash", body{"max_completion_tokens": 1024}},
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

// Whether MiMo's thinking toggle actually works, in either direction.
//
// This is the oldest unmeasured bit in the MiMo entries: both V2.5 profiles
// carry can_be_disabled: true on vendor documentation alone, and no probe has
// ever sent the toggle to this vendor. The V2.6 model cards document the shape
// verbatim in their sample code, as extra_body {"thinking": {"type":
// "disabled"}} — extra_body is the OpenAI SDK's passthrough, so on the wire
// that is a top-level "thinking" object, the same shape Z.ai takes.
//
// Z.ai is the cautionary case: its own API reference documents the disable
// toggle as valid and the endpoint refuses it outright with code 1210, "This
// model always engages in thinking and cannot be disabled". A documented
// toggle is not a working toggle.
//
// Reasoning tokens are the evidence, not acceptance: an accepted-and-ignored
// toggle still reasons. The prompt therefore has to be one this family
// actually thinks about — a one-word prompt reports 0 reasoning tokens with or
// without the toggle, which is how the V2.5 effort probe came out
// inconclusive on -pro.
func TestProbe_MiMoThinkingCanBeDisabled(t *testing.T) {
	c := mimoEndpoint.client(t)
	for _, model := range []string{"mimo-v2.6-pro", "mimo-v2.6-flash"} {
		t.Run(model, func(t *testing.T) {
			// The same arithmetic prompt MiMoReasoningEffortValues uses, so the
			// two logs are comparable: there, the retired mimo-v2.5 spent 225-396
			// completion tokens on it.
			const prompt = "A train leaves at 09:40 and arrives at 13:05 the same day. " +
				"How many minutes is the journey? Reply with the number only."

			b := probeBody(model, prompt)
			b["thinking"] = map[string]any{"type": "disabled"}
			b["max_completion_tokens"] = 4096
			res, err := stream(t, c, b)
			if ok, why := accepted(res, err); !ok {
				t.Logf("FINDING: %s thinking:disabled REJECTED: %s", model, why)
				t.Logf("  => can_be_disabled: false, contradicting the model card. " +
					"ReasoningOff() must fail fast for this model.")
				return
			}
			off := valueOr(res.Usage.Output.Reasoning, -1)
			offTotal := valueOr(res.Usage.Output.Total, -1)

			// The control: the identical prompt with no knob at all. This is
			// also what enabled_by_default is read off — reasoning on a request
			// that asked for nothing means thinking is on by default.
			ctl := probeBody(model, prompt)
			ctl["max_completion_tokens"] = 4096
			cres, cerr := stream(t, c, ctl)
			if ok, why := accepted(cres, cerr); !ok {
				t.Logf("FINDING: %s thinking:disabled ACCEPTED (reasoning: %v, completion: %v) "+
					"but the no-knob control was REJECTED: %s. Inconclusive.",
					model, off, offTotal, why)
				return
			}
			on := valueOr(cres.Usage.Output.Reasoning, -1)
			onTotal := valueOr(cres.Usage.Output.Total, -1)

			t.Logf("FINDING: %s thinking:disabled ACCEPTED. "+
				"disabled: reasoning=%v completion=%v; no knob: reasoning=%v completion=%v",
				model, off, offTotal, on, onTotal)
			t.Logf("  content disabled=%q, no knob=%q",
				Truncate(res.Content, 40), Truncate(cres.Content, 40))

			// "Not reported" is not zero (AGENTS.md, and Usage lanes are
			// pointers for exactly this reason). A response that omits
			// completion_tokens_details says NOTHING about whether the model
			// thought, so it must not reach the honoured branch below: that
			// branch is what sets can_be_disabled, and concluding it from a
			// missing field is concluding it from no evidence.
			offReported := res.Usage.Output.Reasoning != nil
			onReported := cres.Usage.Output.Reasoning != nil

			switch {
			case !offReported || !onReported:
				// The completion-token fallback, and the only case where the
				// silent lane is still informative: this family has been seen
				// counting thinking inside completion_tokens (the retired
				// mimo-v2.5 reported the lane as 0 throughout), so a large drop
				// is the signal.
				if onTotal > 0 && offTotal > 0 && onTotal > offTotal*2 {
					t.Logf("  => reasoning lane NOT REPORTED on at least one call "+
						"(disabled reported=%v, control reported=%v), but completion tokens "+
						"fell by more than half with the toggle (%v -> %v). Reads as HONOURED "+
						"on the completion-token lane; record can_be_disabled: true and say "+
						"in the comment that the reasoning lane was silent.",
						offReported, onReported, onTotal, offTotal)
					return
				}
				t.Skipf("inconclusive: the reasoning lane was not reported "+
					"(disabled reported=%v, control reported=%v) and completion tokens did "+
					"not move materially (%v vs %v). Not reported is not zero.",
					offReported, onReported, offTotal, onTotal)
			case off > 0:
				t.Logf("  => ACCEPTED AND IGNORED: it still reasoned with the toggle set. " +
					"can_be_disabled must NOT be set true on acceptance alone.")
			case on > 0 && off == 0:
				t.Logf("  => HONOURED: reasoning on the control, none with the toggle. " +
					"can_be_disabled: true, enabled_by_default: true.")
			default:
				t.Skipf("inconclusive: neither request reported reasoning and completion "+
					"tokens did not move materially (%v vs %v). This prompt does not make "+
					"%s think, so it cannot show the toggle working either.",
					offTotal, onTotal, model)
			}
		})
	}
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
	b := probeBody("mimo-v2.6-pro", "Reply with the single word: ok")
	b["reasoning_effort"] = "high"
	b["max_completion_tokens"] = probeMaxTokens

	res, err := stream(t, c, b)
	if ok, why := accepted(res, err); !ok {
		t.Logf("FINDING: reasoning_effort REJECTED by mimo-v2.6-pro on chat/completions: %s", why)
		t.Logf("  => loom has been sending a parameter this endpoint refuses.")
		return
	}
	t.Logf("FINDING: reasoning_effort ACCEPTED by mimo-v2.6-pro (reasoning tokens: %v). "+
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
	for _, model := range []string{"mimo-v2.6-pro", "mimo-v2.6-flash"} {
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

// Does reasoning_effort actually form a ladder on the V2.6 pair, or is the
// toggle the only real control?
//
// MiMoReasoningEffortValues established the ACCEPTED SET and nothing more. Its
// per-level token counts were one sample each on a one-step arithmetic prompt,
// and they came out flat: mimo-v2.6-pro read 67 / 69 / 74 for low / medium /
// high against 197 for no level at all, and mimo-v2.6-flash read 27 / 55 / 43
// against 43. Two ways to read that — the levels do nothing, or the prompt had
// too little thinking in it for a level to vary — and one sample per cell
// cannot tell them apart.
//
// This probe fixes both weaknesses: a prompt with genuine variable depth, and
// five samples per cell so a range can be compared instead of a point. It emits
// a VERDICT, not a table, because the verdict is what goes in the profile
// comment.
//
// Deliberately not included: xhigh (already measured as a 400, would waste ten
// calls) and the old arithmetic prompt (a different prompt is the entire point).
func TestProbe_MiMoEffortLadderIsReal(t *testing.T) {
	c := mimoEndpoint.client(t)

	// Variable-depth and cheaply checkable. Listing primes in a range is work
	// the model must actually do rather than recall, it scales with how much
	// checking the model chooses to do, and the answer is short — so the cap
	// below bounds nothing and finish_reason "length" means the sample is
	// invalid rather than merely truncated.
	const prompt = "List every prime number between 100 and 200. " +
		"Reply with the numbers only, comma-separated, nothing else."

	// The number of samples per cell. Five is what makes min/max ranges mean
	// anything; the levels differed by single tokens at n=1.
	const samples = 5

	type cell struct {
		vals       []int64
		mean       float64
		min, max   int64
		invalidate string
	}

	for _, model := range []string{"mimo-v2.6-pro", "mimo-v2.6-flash"} {
		t.Run(model, func(t *testing.T) {
			// "" is the control: the same prompt with no level sent at all. On
			// -pro at n=1 this reasoned MORE than any level, which is the claim
			// most worth rechecking.
			levels := []string{"", "low", "medium", "high"}
			cells := map[string]*cell{}

			for _, effort := range levels {
				cl := &cell{min: 1 << 62}
				for i := 0; i < samples; i++ {
					b := probeBody(model, prompt)
					if effort != "" {
						b["reasoning_effort"] = effort
					}
					// Large enough that the ANSWER never hits it; a sample that
					// does is thrown away rather than recorded short.
					b["max_completion_tokens"] = 8192
					res, err := stream(t, c, b)
					if ok, why := accepted(res, err); !ok {
						cl.invalidate = fmt.Sprintf("sample %d REJECTED: %s", i, why)
						break
					}
					if res.FinishReason == "length" {
						cl.invalidate = fmt.Sprintf("sample %d hit the output cap; "+
							"the cap is binding and the reasoning figure is not comparable", i)
						break
					}
					r := valueOr(res.Usage.Output.Reasoning, -1)
					if r < 0 {
						cl.invalidate = fmt.Sprintf("sample %d reported no reasoning lane", i)
						break
					}
					cl.vals = append(cl.vals, r)
					if r < cl.min {
						cl.min = r
					}
					if r > cl.max {
						cl.max = r
					}
				}
				if cl.invalidate == "" && len(cl.vals) > 0 {
					var sum int64
					for _, v := range cl.vals {
						sum += v
					}
					cl.mean = float64(sum) / float64(len(cl.vals))
				}
				cells[effort] = cl
				label := effort
				if label == "" {
					label = "(none)"
				}
				if cl.invalidate != "" {
					t.Logf("FINDING %s effort=%s: INVALID, %s", model, label, cl.invalidate)
					continue
				}
				t.Logf("FINDING %s effort=%s: reasoning tokens n=%d mean=%.1f min=%d max=%d %v",
					model, label, len(cl.vals), cl.mean, cl.min, cl.max, cl.vals)
			}

			for _, effort := range levels {
				if cells[effort].invalidate != "" {
					t.Skipf("inconclusive: the %q cell did not produce %d clean samples", effort, samples)
				}
			}

			lo, mid, hi, none := cells["low"], cells["medium"], cells["high"], cells[""]

			// A ladder is only real if the ranges do not overlap. Comparing
			// means alone would call a 67/69/74 spread a ladder, which is
			// exactly the mistake this probe exists to avoid.
			separated := lo.max < mid.min && mid.max < hi.min

			// The middle verdict is deliberately gated on RANGES, never on a
			// mean ratio. An earlier draft asked whether hi.mean exceeded
			// lo.mean by half, and on the -pro data that fires (337.8 against
			// 198.2) purely because `high` drew one 1088-token sample — it
			// would have printed "the knob moves something" over the same
			// numbers this file reads as no ladder at all. A mean over five
			// draws is one outlier away from any verdict you like, which is
			// the whole reason this probe compares ranges.
			//
			// So: low and high must not overlap EACH OTHER for the knob to be
			// credited with moving anything, even where medium sits astride
			// them.
			endsSeparated := lo.max < hi.min || hi.max < lo.min

			switch {
			case separated:
				t.Logf("FINDING %s: LADDER IS REAL. low/medium/high ranges do not overlap "+
					"(%d-%d, %d-%d, %d-%d). reasoning_effort is a working control; "+
					"record the means in the profile comment.",
					model, lo.min, lo.max, mid.min, mid.max, hi.min, hi.max)
			case endsSeparated:
				t.Logf("FINDING %s: PARTIAL. low and high do not overlap (%d-%d vs %d-%d) "+
					"but medium (%d-%d) does not sit cleanly between them. The knob moves "+
					"something; it does not give a caller three distinguishable settings.",
					model, lo.min, lo.max, hi.min, hi.max, mid.min, mid.max)
			default:
				t.Logf("FINDING %s: LADDER IS NOT REAL. low/medium/high overlap "+
					"(%d-%d, %d-%d, %d-%d), means %.1f / %.1f / %.1f. "+
					"The three levels are not three settings.",
					model, lo.min, lo.max, mid.min, mid.max, hi.min, hi.max,
					lo.mean, mid.mean, hi.mean)
			}

			// The separate question, and the surprising one at n=1: sending ANY
			// level reasoned less than sending none. If that holds at n=5,
			// reasoning_effort on this model is a second "think less" switch
			// rather than a ladder, which is a cost and quality surprise in the
			// direction no caller expects.
			maxLevel := lo.max
			for _, cl := range []*cell{mid, hi} {
				if cl.max > maxLevel {
					maxLevel = cl.max
				}
			}
			switch {
			case maxLevel < none.min:
				t.Logf("FINDING %s: ANY LEVEL SUPPRESSES THINKING. no level reasoned %d-%d "+
					"(mean %.1f); every level stayed at or below %d. Sending a level is "+
					"closer to a second disable switch than to a ladder.",
					model, none.min, none.max, none.mean, maxLevel)
			case none.max < lo.min && none.max < hi.min:
				t.Logf("FINDING %s: any level INCREASES thinking over sending none "+
					"(none %d-%d).", model, none.min, none.max)
			default:
				t.Logf("FINDING %s: no level (%d-%d, mean %.1f) overlaps the levels; "+
					"the n=1 suppression reading does not reproduce.",
					model, none.min, none.max, none.mean)
			}
		})
	}
}

// What happens when a caller disables thinking AND sends an effort level in the
// same request. ReasoningOff() has to be safe against a caller that also set an
// effort, and plan() needs to know which one the endpoint honours: if the level
// wins, a caller who asked for no thinking silently gets thinking.
func TestProbe_MiMoThinkingToggleBeatsEffort(t *testing.T) {
	c := mimoEndpoint.client(t)
	for _, model := range []string{"mimo-v2.6-pro", "mimo-v2.6-flash"} {
		t.Run(model, func(t *testing.T) {
			b := probeBody(model, "A train leaves at 09:40 and arrives at 13:05 the same day. "+
				"How many minutes is the journey? Reply with the number only.")
			b["thinking"] = map[string]any{"type": "disabled"}
			b["reasoning_effort"] = "high"
			b["max_completion_tokens"] = 4096
			res, err := stream(t, c, b)
			if ok, why := accepted(res, err); !ok {
				t.Logf("FINDING %s: thinking:disabled + reasoning_effort:high REJECTED: %s", model, why)
				t.Logf("  => the two parameters conflict at the endpoint; plan() must not send both.")
				return
			}
			// Not reported is not zero. A response that omits
			// completion_tokens_details says nothing about whether the model
			// thought, so it must not land in either verdict below — the same
			// guard MiMoThinkingCanBeDisabled needs, for the same reason.
			if res.Usage.Output.Reasoning == nil {
				t.Skipf("inconclusive: the reasoning lane was not reported at all, so "+
					"whether the toggle or the effort won cannot be read off this call "+
					"(finish=%s, content=%q)", res.FinishReason, Truncate(res.Content, 40))
			}
			r := *res.Usage.Output.Reasoning
			if r == 0 {
				t.Logf("FINDING %s: thinking:disabled WINS over reasoning_effort:high "+
					"(reasoning tokens 0, content %q). ReasoningOff() is safe against a "+
					"caller that also set an effort.", model, Truncate(res.Content, 40))
				return
			}
			t.Logf("FINDING %s: reasoning_effort:high WINS or both are ignored — reasoning "+
				"tokens %d with thinking disabled. A caller asking for no thinking would "+
				"silently get thinking; plan() must drop the effort when reasoning is off.",
				model, r)
		})
	}
}

// Which output-cap parameter each endpoint honours. Sending the wrong one is
// silently unbounded on some endpoints and a 400 on others.
func TestProbe_OutputCapParameter(t *testing.T) {
	for _, tc := range []struct {
		ep    endpoint
		model string
	}{
		{mimoEndpoint, "mimo-v2.6-pro"},
		{mimoEndpoint, "mimo-v2.6-flash"},
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
		{mimoEndpoint, "mimo-v2.6-pro", "max_completion_tokens"},
		{mimoEndpoint, "mimo-v2.6-flash", "max_completion_tokens"},
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
		{mimoEndpoint, "mimo-v2.6-pro", "max_completion_tokens"},
		{mimoEndpoint, "mimo-v2.6-flash", "max_completion_tokens"},
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

// The retired mimo-v2.5-pro was text-only and 404'd on any image_url part,
// while mimo-v2.5 took images; V2.6 takes them on both. The module errors on an
// image for a text-only model instead of rerouting, so the profile bit has to
// be right. No served MiMo id exercises the REJECTS branch now.
func TestProbe_MiMoImageInput(t *testing.T) {
	c := mimoEndpoint.client(t)
	// A 1x1 transparent PNG, small enough to be inline and large enough to be a
	// real image part.
	const onePixelPNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

	for _, model := range []string{"mimo-v2.6-pro", "mimo-v2.6-flash"} {
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
		{mimoEndpoint, "mimo-v2.6-pro"},
		{mimoEndpoint, "mimo-v2.6-flash"},
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
		{mimoEndpoint, "mimo-v2.6-pro", "max_completion_tokens"},
		{mimoEndpoint, "mimo-v2.6-flash", "max_completion_tokens"},
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

// --- reasoning overhead: how far does glm-5.3-flash think on a long input? ---
//
// reasoning.overhead is the allowance MaxAnswerTokens adds beside the answer,
// and only a measured count may set it. glm-5.3-flash had none, so
// DefaultReasoningOverhead (1024) applied at every level. A consumer's memory
// refresh (MaxAnswerTokens 1900, ReasoningBalanced = high, wire cap 2924) saw
// reasoning_tokens 459, 474, 1600, >=2834 and >=2923 in production, the last
// two cut with finish_reason length and the answer starved, and the count grew
// with the prompt (5k-17.5k tokens). A one-word prompt (ZaiReasoningEffortValues:
// low 0, high 11) says nothing about that.
//
// So this sends that shape: a system prompt maintaining a sectioned user
// memory, an existing memory, a multi-turn transcript, at three input sizes
// (about 3k, 8k and 20k prompt tokens, the last past production's 17.5k),
// five samples per level and size. Every sample is a different transcript, so
// the spread covers content and not only sampling; both levels of one sample
// get the identical prompt. The cap is 16384 on max_tokens, the name this
// endpoint honours (it ACCEPTS max_completion_tokens and ignores it), far
// above the answer asked for. A draw cut by it anyway (finish_reason length)
// is only a floor: it is counted and logged apart, never fed into the
// figures. Temperature and top_p are the profile's recommended values, what
// Chat sends.
//
// The FINDING lines are min/p50/p95/max per cell and per level, nearest-rank,
// so at n < 20 the p95 IS the max: five or fifteen draws carry no tail
// estimate beyond that. The "this run's max rounded up to 512" line covers one
// run only; the profile takes the max over every run recorded in FINDINGS.md.
//
// OPT-IN, beyond LLMWIRE_EVAL: 30 calls of 7-90 seconds each, 13-15
// minutes, past the suite's default deadline and go test's default -timeout,
// so a `-run Probe` sweep skips it. Run it alone:
//
//	LLMWIRE_EVAL=1 LLMWIRE_EVAL_OVERHEAD=1 LLMWIRE_EVAL_MAX_CALLS=30 \
//	LLMWIRE_EVAL_DEADLINE=60m go test -run Probe_ZaiReasoningOverhead -v -timeout 70m .
func TestProbe_ZaiReasoningOverhead(t *testing.T) {
	const (
		model     = "glm-5.3-flash"
		samples   = 5
		capTokens = 16384
	)
	evalGate(t)
	if os.Getenv("LLMWIRE_EVAL_OVERHEAD") != "1" {
		t.Skip("live probe: 30 long calls, about 15 minutes; set LLMWIRE_EVAL_OVERHEAD=1 and see the doc comment")
	}
	// A 16k-token call on a slow tail must not turn a count into a timeout
	// floor.
	c := zaiEndpoint.clientWithin(t, 10*time.Minute)

	sizes := []struct {
		name   string
		tokens int
	}{{"3k", 3000}, {"8k", 8000}, {"20k", 20000}}
	levels := []string{"low", "high"}

	type draw struct{ prompt, reasoning int64 }
	cells := map[string][]draw{}
	cut := map[string][]draw{}
	var promptSum, completionSum int64
	var calls, failed int

	report := func() {
		t.Logf("calls=%d failed=%d prompt_tokens=%d completion_tokens=%d",
			calls, failed, promptSum, completionSum)
		for _, level := range levels {
			var pooled []int64
			var floors []int64
			for _, sz := range sizes {
				key := level + "/" + sz.name
				var rs, ps, fs []int64
				for _, d := range cells[key] {
					rs = append(rs, d.reasoning)
					// An unreported prompt total says nothing about size; its
					// reasoning count still stands.
					if d.prompt >= 0 {
						ps = append(ps, d.prompt)
					}
				}
				for _, d := range cut[key] {
					fs = append(fs, d.reasoning)
				}
				pooled = append(pooled, rs...)
				floors = append(floors, fs...)
				if len(rs) == 0 {
					t.Logf("FINDING %s effort=%s size=%s: no clean sample; cut (floors) %v", model, level, sz.name, fs)
					continue
				}
				prompt := "unreported"
				if len(ps) > 0 {
					p := quantiles(ps)
					prompt = fmt.Sprintf("%d-%d", p.min, p.max)
				}
				s := quantiles(rs)
				t.Logf("FINDING %s effort=%s size=%s prompt %s: reasoning n=%d min=%d p50=%d p95=%d max=%d raw=%v; "+
					"cut at the cap (floors, not counted) %v", model, level, sz.name, prompt, len(rs),
					s.min, s.p50, s.p95, s.max, rs, fs)
			}
			if len(pooled) == 0 {
				continue
			}
			s := quantiles(pooled)
			t.Logf("FINDING %s effort=%s all sizes: reasoning n=%d min=%d p50=%d p95=%d max=%d; "+
				"this run's max rounded up to 512 = %d (the profile takes the max over all runs in "+
				"FINDINGS.md); cut (floors) %v", model, level, len(pooled),
				s.min, s.p50, s.p95, s.max, (s.max+511)/512*512, floors)
		}
	}
	defer report()

	// Sample-major, so a run cut short by the spend guard still holds a draw
	// in every cell rather than all of some and none of others.
	for i := 0; i < samples; i++ {
		for j, sz := range sizes {
			// A seed per (sample, size), so the three sizes are independent
			// transcripts rather than prefixes of one another.
			system, user := memoryRefreshPrompt(sz.tokens, i*len(sizes)+j)
			for _, level := range levels {
				b := body{
					"model": model,
					"messages": []any{
						map[string]any{"role": "system", "content": system},
						map[string]any{"role": "user", "content": user},
					},
					"stream":           true,
					"reasoning_effort": level,
					"max_tokens":       capTokens,
					"temperature":      1.0,
					"top_p":            0.95,
				}
				start := time.Now()
				res, err := stream(t, c, b)
				calls++
				promptSum += valueOr(res.Usage.Input.Total, 0)
				completionSum += valueOr(res.Usage.Output.Total, 0)
				// A 4xx other than 429 is the request shape, and every later
				// call would repeat it. A 429, a 5xx or a cut connection
				// costs one draw, not the run; never retried.
				var apiErr *APIError
				if asAPIError(err, &apiErr) && apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 &&
					apiErr.StatusCode != 429 {
					_, why := accepted(res, err)
					t.Fatalf("FINDING %s effort=%s: REJECTED: %s", model, level, why)
				}
				d := draw{
					prompt:    valueOr(res.Usage.Input.Total, -1),
					reasoning: valueOr(res.Usage.Output.Reasoning, -1),
				}
				t.Logf("call %d effort=%s size=%s sample=%d: prompt=%d completion=%d reasoning=%d finish=%q "+
					"content_chars=%d took=%s", calls, level, sz.name, i, d.prompt,
					valueOr(res.Usage.Output.Total, -1), d.reasoning, res.FinishReason, len(res.Content),
					time.Since(start).Round(time.Second))
				key := level + "/" + sz.name
				switch {
				case err != nil || d.reasoning < 0:
					failed++
					t.Logf("  not counted: err=%v reasoning lane reported=%v",
						Redact(fmt.Sprint(err)), res.Usage.Output.Reasoning != nil)
				case res.FinishReason == "length":
					cut[key] = append(cut[key], d)
				default:
					cells[key] = append(cells[key], d)
				}
			}
		}
	}

	for _, level := range levels {
		n := 0
		for _, sz := range sizes {
			n += len(cells[level+"/"+sz.name])
		}
		if n == 0 {
			t.Errorf("effort=%s produced no clean sample: no figure to read", level)
		}
	}
}

type tokenQuantiles struct{ min, p50, p95, max int64 }

// quantiles is nearest-rank: the ceil(q*n)-th smallest value, so every figure
// is a draw that happened, never an interpolation between two.
func quantiles(vs []int64) tokenQuantiles {
	s := slices.Clone(vs)
	slices.Sort(s)
	rank := func(q float64) int64 {
		k := int(math.Ceil(q*float64(len(s)))) - 1
		return s[max(0, min(k, len(s)-1))]
	}
	return tokenQuantiles{min: s[0], p50: rank(0.50), p95: rank(0.95), max: s[len(s)-1]}
}

// memoryRefreshCharsPerToken is glm-5.3-flash's prompt_tokens against the
// characters of this prompt, measured 2026-11-01: 12495 chars -> 2809 tokens,
// 32364 -> 7241, 72071 -> 16085, 4.45-4.48 throughout.
const memoryRefreshCharsPerToken = 4.47

// memoryRefreshPrompt builds a synthetic memory-refresh request of about
// `tokens` prompt tokens (the call logs the real count). seed varies the
// transcript, so the samples of one cell are different conversations about
// the same person.
func memoryRefreshPrompt(tokens, seed int) (system, user string) {
	system = memoryRefreshSystem
	var tr strings.Builder
	budget := int(float64(tokens)*memoryRefreshCharsPerToken) - len(system) - len(memoryRefreshExisting) - 200
	for k := 0; tr.Len() < budget; k++ {
		turn := memoryRefreshTurns[(k+seed*5)%len(memoryRefreshTurns)]
		u, a := turn(k*7 + seed*11)
		fmt.Fprintf(&tr, "User: %s\n\nAssistant: %s\n\n", u, a)
	}
	user = "EXISTING MEMORY\n\n" + memoryRefreshExisting + "\n\nNEW CONVERSATION\n\n" + tr.String() +
		"Return the complete updated memory."
	return system, user
}

const memoryRefreshSystem = `You maintain the long-term memory a personal assistant keeps about one user.
You receive the current memory and a new conversation between the user and the assistant.
Return the complete updated memory in exactly these sections, in this order:
## Identity, ## Household, ## Work, ## Health, ## Interests, ## Preferences, ## Ongoing, ## Recent.

Rules:
- Keep only durable facts the assistant would need in a future conversation.
- Merge each new fact into the section it belongs to; never duplicate a fact.
- When a new fact contradicts an old one, keep the new one and drop the old one.
- Drop anything stale, one-off or already resolved.
- No speculation, no advice, nothing the user did not say or clearly confirm.
- One short bullet per fact. The whole memory stays under 600 words.
- Output the memory only, no preamble and no closing remark.`

const memoryRefreshExisting = `## Identity
- Name: Marco. 41, software engineer, lives in Zurich (Wipkingen) since 2019.
- Grew up in Ticino; Italian native, German and English fluent, learning Japanese.
- Prefers to be addressed informally.

## Household
- Partner Lena, a physiotherapist; daughter Sofia, 9, in third grade.
- Dog Pixel, a four-year-old border collie mix, 21 kg at the last check.
- Rented four-room flat, third floor, no lift; small balcony with herbs.
- One car, a 2015 Skoda Octavia estate, used mostly on weekends.

## Work
- Backend engineer at a mid-size insurance company, team of six.
- Main stack Java and Kotlin; pushing to move new services to Go.
- Owns the billing integration and the nightly reconciliation jobs.
- On call one week in six; dislikes the pager rota.
- Works from home Mondays and Fridays.

## Health
- Runs three times a week, usually along the Limmat; long run on Sundays.
- Training for the Zurich half marathon in April; target time under 1:50.
- Mild lactose intolerance.
- Sleeps badly when on call; trying to stop coffee after lunch.

## Interests
- Home lab: Proxmox host, a Synology NAS, Home Assistant, Pi-hole.
- Cooks most evenings; likes Japanese and northern Italian food.
- Book club once a month with former colleagues; mostly literary fiction.
- Learning Japanese with Genki II and a weekly tutor on Thursday evenings.
- Follows FC Lugano.

## Preferences
- Wants concrete steps over background explanations.
- Metric units, CHF for money, 24-hour time.
- Code examples in Go unless he asks otherwise.
- Does not want product recommendations unless he asks for them.

## Ongoing
- Planning a family trip to Japan next year, three weeks, dates not fixed.
- Kitchen renovation: collecting quotes, budget around CHF 28000.
- Considering selling the car.
- Evaluating whether to replace the NAS drives (four years old).

## Recent
- Sofia started piano lessons.
- Pixel had a stomach upset after eating something in the park; recovered.
- Finished reading the book club pick about a family in Trieste; liked it.`

var (
	refreshCities = []string{"Zurich", "Lucerne", "Basel", "Bern", "Lugano", "Winterthur", "St. Gallen"}
	refreshMonths = []string{"March", "April", "May", "September", "October", "November"}
	refreshDays   = []string{"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
	refreshDishes = []string{"risotto alla milanese", "a Japanese curry", "polenta with mushrooms",
		"okonomiyaki", "a vegetable lasagne", "pizzoccheri", "miso ramen"}
	refreshServices = []string{"invoice-export", "grafana", "paperless", "nextcloud", "jellyfin", "gitea"}
	refreshBooks    = []string{"a novel about a lighthouse keeper", "a memoir set in post-war Milan",
		"a thriller set in Hokkaido", "a family saga in Sicily", "a short-story collection from Korea"}
)

func pick(xs []string, k int) string { return xs[k%len(xs)] }

// memoryRefreshTurns are user/assistant pairs with slots, so a long
// transcript repeats a topic without repeating its text.
var memoryRefreshTurns = []func(k int) (string, string){
	func(k int) (string, string) {
		return fmt.Sprintf("Did a %d km run along the lake in %s this morning, pace around 5:%02d per km. "+
				"My left knee felt off after km %d. Should I worry with the half marathon in %s?",
				8+k%9, pick(refreshCities, k), 10+k%40, 3+k%5, pick(refreshMonths, k)),
			"A knee that feels off after a few kilometres is usually load, not injury, but it is worth " +
				"watching. Take two easy days, then do a short flat run and note when it starts. If it is " +
				"the outside of the knee and gets worse on descents, that points at the IT band: foam roll " +
				"the outer thigh, add side planks and clamshells three times a week, and keep the long run " +
				"at the same distance for a week instead of adding to it. If there is swelling, sharp pain, " +
				"or it hurts walking down stairs, stop running and have it looked at. You have enough weeks " +
				"before race day that one cautious week costs you nothing on the target time."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("I moved the Proxmox box to the basement rack. The %s VM keeps losing its NFS "+
				"mount after a reboot. The NAS is at 192.168.%d.%d now, I changed the subnet last week.",
				pick(refreshServices, k), 10+k%40, 2+k%200),
			"The mount most likely races the network on boot, and the subnet change may have left an old " +
				"address in the export rules. First, on the NAS, check that the NFS permission entry names " +
				"the new subnet; a stale rule fails silently. Then in the VM replace the plain fstab line " +
				"with a systemd automount: add noauto,x-systemd.automount,_netdev to the options so the " +
				"share mounts on first access instead of at boot. Run systemctl daemon-reload and reboot " +
				"once to confirm. If it still fails, journalctl -u the mount unit shows whether it is a " +
				"timeout or a permission error, which tells you which side to look at."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("Making %s for %d people on %s. Lena has gone fully vegetarian since last month, "+
				"and one guest is gluten-free. What should I swap?",
				pick(refreshDishes, k), 4+k%6, pick(refreshDays, k)),
			"Most of it adapts cleanly. Use a vegetable stock you have tasted on its own, since it carries " +
				"the dish once meat is gone; a spoon of white miso or a parmesan rind adds the depth the meat " +
				"stock would have given (rind is fine for a vegetarian who eats cheese, check with Lena on " +
				"rennet). For the gluten-free guest, check the stock cube and any soy sauce, swap to tamari, " +
				"and keep a separate pot and spoon if there is pasta or flour anywhere else in the meal. Make " +
				"the gluten-free portion first and set it aside before anything else touches the pan. Plan " +
				"a little more of the vegetarian side, it tends to go first."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("We fixed the Japan dates: %d %s to %d %s, flying into %s. I want two days in "+
				"Nara with Sofia. Is the JR pass still worth it after the price increase?",
				1+k%20, pick(refreshMonths, k), 18+k%10, pick(refreshMonths, k+1),
				[]string{"Tokyo Haneda", "Osaka Kansai", "Tokyo Narita"}[k%3]),
			"After the increase the national pass only pays off if you do several long Shinkansen legs " +
				"inside its validity window. Price your actual route with single tickets: Tokyo to Kyoto and " +
				"back, plus a side trip, is usually below the seven-day pass. Regional passes such as the " +
				"Kansai area pass are often the better deal for Kyoto, Nara and Osaka. Children under twelve " +
				"pay half on most tickets, so Sofia's fare is cheap either way. For Nara, stay one night " +
				"rather than day-tripping: the deer park and Todai-ji are much calmer before nine in the " +
				"morning, and that is the part a nine-year-old remembers."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("At work we are moving the billing service from %s to Go. About %d endpoints. "+
				"I'm worried there is no maintained equivalent of the PDF library we use.",
				[]string{"Java 11", "Kotlin", "Spring Boot 2"}[k%3], 20+k%60),
			"Do the migration endpoint by endpoint behind the existing routing, not as a rewrite: put " +
				"the Go service next to the old one and move traffic per route, so each step can be rolled " +
				"back. For the PDF part, keep it where it is for now and call it over a small internal API; " +
				"rendering is the least valuable thing to port first and the riskiest. Start with the " +
				"read-only endpoints, compare responses from both services on mirrored traffic for a week, " +
				"and only then move writes. Write the reconciliation job's checks as the acceptance test, " +
				"since billing errors show up there first."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("The contractor quoted CHF %d for the kitchen, %d%% over our budget. The "+
				"induction hob alone is CHF %d. Push back on cabinets or countertop?",
				30000+k*350%9000, 5+k%20, 1800+k*90%1500),
			"Ask for the quote itemised first: labour, cabinets, countertop, appliances and installation " +
				"separately, so you are negotiating line items rather than a total. Cabinets are usually " +
				"where the margin is, and a standard carcass with custom fronts gets most of the look for " +
				"less. A countertop is hard to change later, so it is the worse place to save. Appliances " +
				"you can often buy yourself at a lower price; check whether the contractor will install " +
				"them without a surcharge. A second quote with the same itemised spec gives you leverage " +
				"and a reality check on the labour figure."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("I'm on lesson %d of Genki II and I keep mixing up %s. Can you give me a short "+
				"drill? My tutor session is on %s.",
				13+k%11, []string{"ageru and kureru", "the passive and the causative", "you ni naru and koto ni naru",
					"ba and tara"}[k%4], pick(refreshDays, k+3)),
			"Here is a five-minute drill. Say each sentence out loud, then check it. First, decide who " +
				"is the giver and who receives, and whose side the speaker is on; that choice decides the " +
				"verb. Second, translate five sentences in each direction, changing only the people: my " +
				"friend to me, me to my friend, my friend to my sister, my teacher to me, and me to my " +
				"teacher. Third, write two sentences about your own week with each form. Bring the ones " +
				"you hesitated on to the tutor; the hesitation is the useful part, not the right answers."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("Pixel was at the vet on %s, she's %d kg and the vet wants her at %d. She gets "+
				"bored on the usual route along the river.", pick(refreshDays, k), 20+k%4, 18+k%2),
			"A slow loss is the goal, about one percent of body weight a month. Weigh her food rather " +
				"than using a cup, cut the daily amount by around ten percent, and count treats against it: " +
				"use part of the kibble ration as training rewards. For boredom, sniffing tires a border " +
				"collie mix more than distance. Vary the route twice a week, scatter part of her food on a " +
				"lawn to search for, and add five minutes of trick training. A weekly weigh-in at the same " +
				"time of day tells you whether the amount is right within a month."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("Doing the Zurich tax return. I paid CHF %d into pillar 3a this year and had "+
				"%d home-office days. Can I deduct the standing desk?", 5000+k*137%2300, 60+k%60),
			"The 3a contribution is deductible up to the annual maximum for employees with a pension " +
				"fund; enter it with the bank certificate. Home office: Zurich generally does not allow a " +
				"separate home-office deduction for employees if the employer provides a workplace, and " +
				"the standing desk falls under that unless your employer requires home work and states it " +
				"in writing. What does count is the flat-rate professional expense deduction, and travel " +
				"costs for the days you commuted. Keep the receipt anyway; if your employer confirms the " +
				"home-office requirement, you can claim actual costs instead of the flat rate."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("Our book club picked %s for %s. I found the first half slow. How do I lead "+
				"the discussion without spoiling it for the two who are behind?",
				pick(refreshBooks, k), pick(refreshMonths, k+2)),
			"Structure the evening in two halves and say so up front: the first hour on the first half " +
				"of the book only, then a break, and the rest open for whoever finished. Prepare three " +
				"questions for the first half that do not depend on the ending: why the author starts where " +
				"they do, which character you trusted early and why, and what the slow pace is doing. That " +
				"last one turns your complaint into the discussion. Ask the two who are behind to say where " +
				"they stopped, so everyone knows the line."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("By the way, keep answers shorter from now on, bullet points are fine. And stop "+
				"suggesting %s, I tried it and it did nothing.",
				[]string{"foam rolling", "meditation apps", "a sleep tracker", "blue-light glasses"}[k%4]),
			"Understood. Shorter answers, bullets where they help, and I will not suggest that again."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("Sofia starts %s in %s and she is anxious about the new teacher. Any tips "+
				"for the first week?", []string{"fourth grade", "the after-school club", "swimming lessons",
				"a new piano teacher"}[k%4], pick(refreshMonths, k+4)),
			"Make the unknown smaller. If you can, walk past the school or the venue with her before the " +
				"first day and look at where she will go in. Ask her what she is worried about specifically " +
				"and take each worry seriously rather than reassuring it away. Agree on a small routine for " +
				"after school, for example ten minutes where she tells you one good and one hard thing. Most " +
				"of the anxiety is about the first two days, so plan something she looks forward to on the " +
				"first evening. If it has not eased after three weeks, a word with the teacher is reasonable."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("Started taking magnesium in the evening and I'm sleeping better, around %d "+
				"hours now. I also stopped coffee after %d o'clock.", 6+k%3, 13+k%4),
			"That is a good change, and the coffee cut-off probably does as much as the magnesium; " +
				"caffeine's half-life is around five hours, so an afternoon cup is still active at bedtime. " +
				"Keep the two changes stable for a few weeks before adding anything else, so you know what " +
				"works. On-call weeks will disturb it regardless; on those, a fixed wake-up time matters " +
				"more than the time you go to bed."
	},
	func(k int) (string, string) {
		return fmt.Sprintf("Thinking about selling the car and going e-bike only. Commute is %d km each "+
				"way with %d m of climb, and winters in %s are wet.", 6+k%12, 80+k*13%250, pick(refreshCities, k)),
			"At that distance an e-bike is realistic all year if you set it up for winter: mudguards, " +
				"lights that are visible in daylight, and studded or at least winter tyres for the icy days. " +
				"Rain gear decides whether you keep doing it, so budget for a good jacket and overtrousers. " +
				"For weekends, a car-sharing subscription and the occasional rental cover the trips the car " +
				"was for. Try it for a month before selling: park the car and only use it if there is no " +
				"other way, and count how often that happens."
	},
}
