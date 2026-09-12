# Phase 0 findings

Measured against live endpoints on 2026-09-12 with `LLMWIRE_EVAL=1 go test -run Probe`.
Reproduce with the probes in `eval_probe_test.go`; each finding below names the
probe that produced it.

These measurements are the source of truth for the profile schema. Where a vendor
doc or a published catalogue disagrees with a line here, the line here wins — and
three of them do disagree.

Endpoints: a MiMo Token Plan host serving `mimo-v2.5-pro` and `mimo-v2.5`, and
Z.ai's general (non-Coding-Plan) host serving `glm-5.3-flash`. No LiteLLM
instance was reachable, so nothing about that gateway is measured.

## The three architecture questions

### 1. MiMo returns native `tool_calls`, not inline XML

`TestProbe_MiMoToolCallFormat`

Both `mimo-v2.5-pro` and `mimo-v2.5` populated standard OpenAI `tool_calls` when
offered a tool. No `<tool_call>`, `<function=` or `<tool_invocation` markup
appeared in either `content` or `reasoning_content`.

The model's own chat template does emit that XML, but the serving layer parses it
before it reaches a client, which is what the vendor's SGLang `--tool-call-parser
mimo` setting exists to do.

`TestProbe_MiMoToolFreeCall` covered the narrower case loom's comment describes:
a *forced tool-free* call, made with no tools offered at all, where a model that
still decides to call something has nowhere to put the markup but the content
stream. Both models returned clean prose, no markup, `finish_reason: stop`.

**Consequence.** The ~270-line inline-XML parser and the dual-channel stream
gating that loom and trift carry did not fire in either scenario tested.

**But do not delete them on this evidence.** The tool-free probe used a synthetic
prompt describing a prior search, not a genuine multi-round tool history, and
loom's observation is of production traffic after real tool rounds. Two
explanations fit equally well: the serving layer gained the `mimo` parser after
loom's code was written, or the leak needs a longer, tool-saturated history than
a probe reproduces. Inline recovery therefore ships as an **opt-in profile
capability**, defaulting off for these deployments, rather than as dead code
removed or live code always running. That keeps the machinery one profile flag
away if loom's case resurfaces during its migration.

### 2. `reasoning_content` need not be replayed in history

`TestProbe_MiMoReasoningReplayRequirement`

A second tool round whose assistant turn carried `content` and `tool_calls` but
**no** `reasoning_content` was accepted by `mimo-v2.5-pro`, with thinking on.

A vendor issue reports HTTP 400 and *"The reasoning_content in the thinking mode
must be passed back to the API"* for exactly this shape. It does not reproduce
here, which matches loom building its history this way in production.

**Consequence.** `Message.ReasoningContent` stays optional. No replay invariant,
no forced plumbing through every consumer's history builder.

### 3. The `data:`-only idle rule has a 60x margin

`TestProbe_LongestCommentOnlyGap`

Longest interval with no `data:` frame, on a deliberately deep call:

| model | longest gap | frames | idle bound |
|---|---|---|---|
| `mimo-v2.5-pro` | 1.129s | 274 | 90s |
| `glm-5.3-flash` | 1.396s | 1022 | 90s |

**Consequence.** Re-arming the idle guard on `data:` frames only — never on blank
lines or `: ping` comments — is safe on both endpoints. No per-profile
`idle_rearm_on_keepalive` escape hatch is needed. Revisit if an endpoint is added
whose measured gap exceeds half the bound.

Incidentally measured: `glm-5.3-flash` at `reasoning_effort: max` with a 1024
token cap produced **1022 frames and zero content characters** — the entire budget
went to reasoning. That is the "cap counts reasoning, so a deep call returns
empty" failure, reproduced.

## Per-model profile data

### `glm-5.3-flash`

| bit | value | probe |
|---|---|---|
| `can_be_disabled` | **false** | `ZaiThinkingCanBeDisabled` |
| `effort_values` | **`[low, high, max]`** | `ZaiReasoningEffortValues` |
| `max_tokens_param` | **`max_tokens`** | `OutputCapParameter` |
| `needs_include_usage` | **false** | `UsageReportingWithoutStreamOptions` |
| `json_object` | **true** | `ResponseFormatSupport` |
| `json_schema` | **true** | `ResponseFormatSupport` |

`thinking: {"type":"disabled"}` is refused:

```
status 400, code 1210
"This model always engages in thinking and cannot be disabled; please use low, high, or max"
```

`reasoning_effort` values `none`, `minimal`, `medium` and `xhigh` are refused with
**the same code 1210 and the same message**. The code is therefore overloaded
across two unrelated conditions, and the message names only a subset of what it
rejects. Reasoning tokens scale with the accepted tiers: `low` 0, `high` 11,
`max` 31 on a one-word prompt.

**`max_completion_tokens` is accepted and silently ignored.** A request capping at
16 returned 481 completion tokens with `finish_reason: stop`. Nothing is rejected
and nothing signals the cap was dropped, so sending the wrong parameter name here
means unbounded output rather than an error. This is the most expensive silent
failure found.

### `mimo-v2.5-pro`

| bit | value | probe |
|---|---|---|
| `vision` | **false** | `MiMoProRejectsImageInput` |
| `max_tokens_param` | **both accepted and honoured** | `OutputCapParameter` |
| `reasoning_effort` | **accepted** on chat/completions | `MiMoAcceptsReasoningEffort` |
| `needs_include_usage` | **false** | `UsageReportingWithoutStreamOptions` |
| `json_object` | **true** | `ResponseFormatSupport` |
| `json_schema` | **true** | `ResponseFormatSupport` |

An `image_url` part returns HTTP 404, *"No endpoints found that support image
input"*.

Both `max_tokens` and `max_completion_tokens` cap the completion: each returned
exactly 16 tokens with `finish_reason: length`. The belief that only
`max_completion_tokens` works is too strong — the alias is honoured too.

`reasoning_effort` is accepted on chat/completions despite the vendor documenting
it as a Responses-API parameter. Whether it changes anything is not established;
it reported 0 reasoning tokens on a one-word prompt, as did the request without
it.

Latency is the thing to plan around, not tokens: single calls took **25–64
seconds**, against 1–5 seconds for `glm-5.3-flash` on comparable prompts.

### `mimo-v2.5`

| bit | value | probe |
|---|---|---|
| `vision` | **true** | `MiMoProRejectsImageInput` |
| tool-call format | **native** | `MiMoToolCallFormat` |

The same image part that 404s on `-pro` is accepted here. This is the split loom
routes around, and the reason the module errors rather than rerouting: the two
are different models with different capabilities, and a silent swap would
misattribute cost.

## Where the documentation is wrong

| claim | source | measured |
|---|---|---|
| `thinking: {"type":"disabled"}` is valid on `glm-5.3-flash` | Z.ai API reference | **refused, code 1210** |
| `json_schema` unsupported on Z.ai | Z.ai OpenAPI enum and docs | **accepted and honoured** |
| `json_schema` unsupported on MiMo | vendor documents `json_object` only | **accepted and honoured** |
| Z.ai does not accept `stream_options` | absent from its request schema | **accepted** (and unnecessary) |
| MiMo takes `max_completion_tokens`, not `max_tokens` | inferred from a prior measurement | **both honoured** |
| code 1210 means "invalid API parameter" | Z.ai error table | correct but useless: it is returned for at least two unrelated conditions |

Two further notes on error handling, both of which changed the code:

- **The 1210 message arrived in English.** Prior evidence had it in Chinese. The
  text varies; only the code is stable. This is the strongest argument yet for
  dispatching on `code` and never on message prose.
- **MiMo returns SSE-framed error bodies under non-2xx statuses.** The image 404
  body was the single line `data:{"error":{...}}`. The envelope parser now
  unwraps a leading `data:` frame, or the code would have been lost to raw text
  on exactly the failures that matter most.

## Cost observations

Prompt caching is live and visible on MiMo: a 258-token prompt reported
`cached_tokens: 192`. That matches the documented behaviour where the endpoint
injects its own system message when none is supplied, most of it served from
cache. It also means the cache-read lane is not theoretical on this endpoint and
must be priced separately from the first call onward.

`glm-5.3-flash` reported `cached_tokens: 0` throughout, on prompts too short to
reach any cache minimum.

The whole run — every probe in this document — cost **30 calls and roughly
$0.008** at conservative upper-bound rates, well inside the $0.50 circuit
breaker.

## Still open

- **Everything LiteLLM.** No instance was reachable. Header names, the cost-header
  parsing and the error shapes remain source-derived at v1.100.1 and unverified.
- **Token Plan credit rates.** Not published per token in any form verified here,
  and a probe cannot read a credit balance. The MiMo profiles reached over a
  Token Plan host therefore ship `Unpriced` rather than a guessed conversion.
- **Whether off-peak multipliers apply to the USD lane** as well as to credit
  burn. Both vendors document them as credit coefficients.
- **Whether inline tool markup can still appear at all.** Neither probe
  reproduced it, but neither reproduced loom's exact conditions either: a long,
  tool-saturated history in production. The capability stays in the profile
  schema, defaulted off, and loom's migration is where it gets a real test.
