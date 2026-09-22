# Phase 0 findings

Measured against live endpoints on 2026-09-12 with `LLMWIRE_EVAL=1 go test -run Probe`.
Reproduce with the probes in `eval_probe_test.go`; each finding below names the
probe that produced it.

These measurements are the source of truth for the profile schema. Where a vendor
doc or a published catalogue disagrees with a line here, the line here wins — and
three of them do disagree.

Endpoints: a MiMo Token Plan host serving `mimo-v2.5-pro`, `mimo-v2.5`,
`mimo-v2.6-pro` and `mimo-v2.6-flash`, and Z.ai's general (non-Coding-Plan) host
serving `glm-5.3-flash`.

The V2.6 pair was added on 2026-09-22 (`mimo-v2.6-pro`, `mimo-v2.6-flash`
sections below); the Z.ai probes skipped on that run for want of a key, so every
`glm-5.3-flash` line still dates from 2026-09-12.

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
a probe reproduces. Inline recovery therefore ships as a **profile capability**
(`tools.recover_inline_markup`), implemented in `inline.go` from loom's parser
and its production captures, and switched ON for both MiMo profiles: it acts
only when the markup appears, so a clean answer costs nothing, and a leak that
does resurface is recovered instead of reaching a user.

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

### `mimo-v2.6-pro`

Measured on 2026-09-22. Vendor deprecates the V2.5 pair on 2026-10-21; the
host's own `/models` listed `mimo-v2.6-pro` and `mimo-v2.6-flash` beside them.

| bit | value | probe |
|---|---|---|
| `vision` | **true** | `MiMoProRejectsImageInput` |
| `max_tokens_param` | **both accepted and honoured** | `OutputCapParameter` |
| `reasoning.can_be_disabled` | **true** | `MiMoThinkingCanBeDisabled` |
| `reasoning.enabled_by_default` | **true** | `MiMoThinkingCanBeDisabled` |
| `reasoning.effort_values` | **low, medium, high** (`xhigh` a 400) | `MiMoReasoningEffortValues` |
| `needs_include_usage` | **false** | `UsageReportingWithoutStreamOptions` |
| `json_object` / `json_schema` / `strict_schema` | **true** | `ResponseFormatSupport` |
| tool-call format | **native** | `MiMoToolCallFormat` |
| usage containment | **cached inside prompt** | `CachedTokensAreInsidePromptTokens` |

**It accepts image input, and that reverses the V2.5 split.** The same
`image_url` part that returns HTTP 404 *"No endpoints found that support image
input"* on `mimo-v2.5-pro` is accepted here — with `mimo-v2.5-pro` re-probed as a
control in the same run, still 404. The vendor's own model card is
self-contradictory: the prose sells *"Omni-Modal Understanding"* and *"joint
input and understanding of images, video, audio, and text"*, while the
specification table on the same page lists **Input Modality: Text**, and
OpenRouter documents text only. The prose is right.

**First direct measurement of MiMo's thinking toggle.** Both V2.5 entries carry
`can_be_disabled` on vendor documentation alone; no probe had ever sent the
toggle to this vendor in either direction. Sent as a top-level `thinking:
{"type": "disabled"}` (the card documents it inside the OpenAI SDK's
`extra_body`, which is the same thing on the wire) it is accepted and honoured:

| request | reasoning tokens | completion tokens | answer |
|---|---|---|---|
| `thinking: disabled` | 0 | 4 | `205` |
| no knob at all | 54 | 59 | `205` |

That second row is also what `enabled_by_default: true` rests on. Worth
contrasting with Z.ai, where the vendor documents the same toggle and the
endpoint refuses it outright with code 1210.

**The reasoning lane is populated, where V2.5 reported zero.** On one arithmetic
prompt, 197 of 202 completion tokens came back as `reasoning_tokens`;
`mimo-v2.5` reported `0` on the identical prompt and counted its thinking inside
`completion_tokens`. Code that read the lane as "this model does not think" will
now see the real figure.

**`reasoning_effort` is accepted with an enforced set, and the levels are flat.**
`low` / `medium` / `high` return 200, `xhigh` is a 400 *"Invalid request
parameters"* — the same set as V2.5, and still in no vendor document for this
model. What the levels do is NOT established. One sample each:

| level | reasoning tokens |
|---|---|
| none sent | 197 |
| `low` | 67 |
| `medium` | 69 |
| `high` | 74 |

The three levels do not separate from each other; they separate only from
sending nothing, and in the direction a caller would not expect — every level
thought *less* than no level at all. Not reproduced on `-flash` (43 / 27 / 55 /
43), so this is one model, one prompt, one sample. Recorded as accepted-and-flat.

**Cache starts cold and is contained.** `cached_tokens` went 0 then 10880 against
a `prompt_tokens` of 10909 on the identical prompt, so pricing subtracts. Unlike
`mimo-v2.5-pro` — which reported 192 of 258 cached on a tiny prompt because it
injects its own system message — this endpoint reported 0 on a fresh 10.9k prompt.

**Latency is 1-5 seconds**, and the longest gap with no `data:` frame was 1.03s
over 221 frames, against the 90s default idle bound.

### `mimo-v2.6-flash`

Measured on 2026-09-22.

| bit | value | probe |
|---|---|---|
| `vision` | **true** | `MiMoProRejectsImageInput` |
| `max_tokens_param` | **both accepted and honoured** | `OutputCapParameter` |
| `reasoning.can_be_disabled` | **true** | `MiMoThinkingCanBeDisabled` |
| `reasoning.enabled_by_default` | **true** | `MiMoThinkingCanBeDisabled` |
| `reasoning.effort_values` | **low, medium, high** (`xhigh` a 400) | `MiMoReasoningEffortValues` |
| `needs_include_usage` | **false** | `UsageReportingWithoutStreamOptions` |
| `json_object` / `json_schema` / `strict_schema` | **true** | `ResponseFormatSupport` |
| tool-call format | **native** | `MiMoToolCallFormat` |
| usage containment | **cached inside prompt** | `CachedTokensAreInsidePromptTokens` |

Every bit matches `-pro` except the ones below. The thinking toggle is honoured
independently here too: reasoning tokens 42 with no knob, 0 with the toggle.

**Disabling thinking made it answer incorrectly.** With `thinking: disabled` it
returned `155` for a one-step arithmetic prompt whose answer is `205`, which it
gives correctly with thinking on. One sample, so not a law — but on this model
disabling thinking is a quality decision, not only a latency one. `-pro` kept the
correct answer both ways.

**`max_output` moved 4x across the version bump.** `mimo-v2.5` shipped 32768;
this model's card documents 128K, the same ceiling as `-pro`. Not measured — no
probe sent a cap above the ceiling, and every endpoint measured so far clamps
silently.

**Fastest endpoint in this file**: 0.9-2.6 seconds per call, longest
comment-only gap 522ms over 353 frames.

### `reasoning_effort` on the V2.6 pair: the ladder is not a ladder

Measured 2026-09-22 by `MiMoEffortLadderIsReal`, five samples per level on a
variable-depth prompt ("list every prime number between 100 and 200"), which
replaced the one-step arithmetic prompt that produced the earlier n=1 reading.
Reasoning tokens:

| level | `mimo-v2.6-pro` | `mimo-v2.6-flash` |
|---|---|---|
| none sent | 104-656, mean 254.6 | 117-1784, mean 451.0 |
| `low` | 120-507, mean 198.2 | 104-121, mean 115.2 |
| `medium` | 120-148, mean 125.8 | 117-121, mean 118.6 |
| `high` | 104-1088, mean 337.8 | 104-117, mean 111.8 |

**Every range overlaps every other, on both models.** On `-flash` the three
levels span 17 tokens between them and `high` has the *lowest* mean of the
three — there is no trend to be noisy around.

**Read the samples, not the means.** `-pro`'s means look directional (338 for
high against 198 for low) and that is an artefact. The raw draws are
`[121 255 121 1088 104]` for high and `[121 507 120 122 121]` for low: four of
five samples at every level sit in a ~104-150 band, and the mean is decided by
whether that level happened to draw one outlier. What varies is not the level.
It is whether the deployment takes a deep pass on a given call. The same
applies to `-flash`'s no-level mean of 451, which is one 1784-token draw in
five whose other four samples sit at 117-120.

So a caller gets **a floor with occasional deep thinks, not three settings**.
`effort_values` stays `[low, medium, high]` because the endpoint enforces that
set and `xhigh` is a 400 — the profile describes the wire — but the thinking
toggle is the control that works.

**The earlier suppression reading does not reproduce.** At n=1 every level on
`-pro` appeared to reason *less* than sending no level (67/69/74 against 197),
which suggested `reasoning_effort` was a second disable switch. At n=5 the
no-level range (104-656) overlaps every level. This is the reason the profile
comments now carry ranges and raw draws rather than point figures.

### The thinking toggle wins over an effort level

`MiMoThinkingToggleBeatsEffort`, one call per model: `thinking: {"type":
"disabled"}` and `reasoning_effort: "high"` sent in the same request are not
refused, and the toggle wins — reasoning tokens 0 on both. A caller that
disables thinking while also passing an effort gets what it asked for, so
`ReasoningOff()` needs no special handling to strip the level.

## Where the documentation is wrong

| claim | source | measured |
|---|---|---|
| `thinking: {"type":"disabled"}` is valid on `glm-5.3-flash` | Z.ai API reference | **refused, code 1210** |
| `json_schema` unsupported on Z.ai | Z.ai OpenAPI enum and docs | **accepted and honoured** |
| `json_schema` unsupported on MiMo | vendor documents `json_object` only | **accepted and honoured** |
| Z.ai does not accept `stream_options` | absent from its request schema | **accepted** (and unnecessary) |
| MiMo takes `max_completion_tokens`, not `max_tokens` | inferred from a prior measurement | **both honoured** |
| code 1210 means "invalid API parameter" | Z.ai error table | correct but useless: it is returned for at least two unrelated conditions |
| `mimo-v2.6-*` input modality is text | vendor model card **specification table**, and OpenRouter | **image input accepted** — the prose on the same card ("Omni-Modal Understanding") is the correct half |
| `mimo-v2.6-flash` inherits V2.5's 32768 output ceiling | inferred from the V2.5 entry | **131072**, per the card; the ceiling moved 4x at the version bump |
| a published catalogue can price `mimo-v2.6-*` | models.dev | **no v2.6 entry exists at all** (2026-09-22); the vendor page is the only source, with no cross-check |
| the vendor page carries USD rates for `mimo-v2.5*` | the `source_url` both V2.5 entries cite | **it no longer does**: the Overseas (USD) table lists only V2.6 ids, V2.5 survives only in the domestic RMB table marked "(deprecated)" |
| `reasoning_effort` is a Responses-API parameter | vendor documentation, all MiMo models | **accepted on chat/completions with an enforced set**; `xhigh` is a 400 |

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

**Cached tokens are INSIDE `prompt_tokens` on both vendors** (measured
2026-09-13, `TestProbe_CachedTokensAreInsidePromptTokens`). The same ~10k-token
prompt sent twice, five seconds apart:

| model | call 1 prompt / cached | call 2 prompt / cached |
|---|---|---|
| mimo-v2.5-pro | 11131 / 192 | 11131 / 11072 |
| glm-5.3-flash | 9208 / 0 | 9208 / 9152 |

`prompt_tokens` did not move while `cached_tokens` grew to nearly all of it, so
the full-rate lane is `prompt_tokens - cached_tokens` and pricing subtracts
correctly. Had a vendor reported cached tokens beside `prompt_tokens`, the
subtraction would under-count the full-rate lane, which is the one direction a
budget cap cannot tolerate; that is now ruled out for both. Z.ai's field is
confirmed as `prompt_tokens_details.cached_tokens`, and its cache is live: this
was the first non-zero value seen from it.

The original run — every probe above the containment table — cost **30 calls
and roughly $0.008** at conservative upper-bound rates. The containment probe
adds **4 calls and roughly $0.025** per run, nearly all of it the ~10k-token
prompt sent twice per vendor. Both well inside the $0.50 circuit breaker.

The 2026-09-22 V2.6 onboarding run added the two new ids to every MiMo probe
table, so it re-ran the V2.5 rows as controls: **about 63 calls and roughly
$0.065** across nine separate invocations, of which the containment probe alone
was $0.053. The Z.ai rows skipped for want of a key. Well inside the breaker
again, and the whole run fit the default 10-minute deadline per invocation
because V2.6 answers in 1-5 seconds where V2.5-pro took 25-64.

The same run showed `mimo-v2.5-pro` answering in **3.5-18 seconds**, not the
25-64 recorded on 2026-09-12. Its `notes` still carry the old figure; the
latency class of that endpoint has evidently improved and has not been re-probed
deliberately.

Settling the effort ladder cost a further **46 calls and roughly $0.034**:
40 for `MiMoEffortLadderIsReal` (four levels, five samples, two models, on a
prompt that produces real reasoning rather than the ~10 tokens the arithmetic
prompt drew), 4 to re-confirm the thinking toggle after fixing a nil-versus-zero
bug in its own probe, and 2 for `MiMoThinkingToggleBeatsEffort`.

`defaultMaxCalls` moved 60 -> 160 at the same time. With four MiMo models in the
per-model tables a full `-run Probe` sweep is ~75 calls before the ladder probe's
40, so the documented command was dying partway at the old ceiling and reading as
a harness failure rather than as the guard working. The USD breaker is unchanged
and is what bounds real spend: a full sweep is about $0.10 against $0.50.

## Still open

- **Everything LiteLLM.** No instance was reachable. Header names, the cost-header
  parsing and the error shapes remain source-derived at v1.100.1 and unverified.
- ~~**Token Plan credit rates.**~~ **Closed by policy, not by measurement.** Credits
  are not converted at all: llmwire reports the vendor's published pay-as-you-go USD
  rate for every model, including one reached over a Token Plan host, and labels it
  a list-rate equivalent rather than an invoice. The MiMo rates in `profiles.yaml`
  come from the vendor's own page (0.435/0.87 for `mimo-v2.5-pro`, contradicting a
  third-party doc's 1.00/3.00). What a credit invoice actually charges is still
  unknown and is now deliberately out of scope.
- **Whether off-peak multipliers apply to the USD lane** as well as to credit
  burn. Both vendors document them as credit coefficients.
- **Whether inline tool markup can still appear at all.** Neither probe
  reproduced it, but neither reproduced loom's exact conditions either: a long,
  tool-saturated history in production. Recovery is on for the MiMo profiles;
  the `tool_calls` warning it emits is how a recurrence gets noticed.
- ~~**What `reasoning_effort` actually does on the V2.6 pair.**~~ **Closed
  2026-09-22 by measurement: the ladder is not a ladder.** See the
  `reasoning_effort` section below.
- **`stream_field` on the V2.6 pair.** Left at the load default. `stream.go` reads
  both `reasoning_content` and `reasoning` with content winning, so which key the
  delta carried cannot be read back out of a `StreamResult`; confirming it needs a
  spooled raw stream.
- **`temperature.inert_while_reasoning` on the V2.6 pair.** Absent rather than
  carried over from V2.5. That bit rests on a vendor statement that the model
  forcibly overrides both sampling parameters while thinking, and no V2.6 source
  repeating it has been found; the vendor's API docs are JS-rendered and were not
  machine-readable. `supported: true` is kept so a caller sending either is not
  refused.
- **`tool_choice` and parallel tool calls on the V2.6 pair.** No probe offered a
  forced choice or two tools in one turn to any MiMo endpoint. Both entries carry
  the family's documented values (`auto` only, no forced choice, parallel
  supported); `tool_choice_values` cannot be empty while tools are supported, so
  these are the doc's values rather than an absence.
- **Whether the V2.6 thinking toggle behaves the same on the V2.5 pair.**
  `MiMoThinkingCanBeDisabled` covers the V2.6 ids only. Adding the two V2.5 rows
  would close a bit those entries have carried on documentation since they
  shipped — about four calls — but they are deprecated on 2026-10-21.
- **Whether disabling thinking systematically degrades `-flash`.** Now two
  samples, both wrong and in the same direction: the same one-step arithmetic
  prompt (answer 205) returned **155** and **145** with thinking disabled, and
  the correct answer with it on. `-pro` kept the correct answer both ways. Two
  draws is not a study, but it is enough that `ReasoningOff()` against `-flash`
  should be treated as a quality decision. A real accuracy comparison over a
  prompt set would settle it.
