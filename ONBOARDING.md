# Onboarding a model

The procedure for adding a model, a model version, a provider or a gateway route
to `profiles.yaml`. FINDINGS.md is what the probes *found*; this is how to
*run* them and where each answer goes. Follow it top to bottom. The output is
one profile entry, one FINDINGS.md section, one pinning subtest, and every
`verified: measured` bit backed by a named `TestProbe_*`.

Rules this file assumes: AGENTS.md (quirks are data, evals are the source of
truth, comment standard) and the head comment of `profiles.yaml`.

## 0. Which shape is this

| shape | first step |
|---|---|
| New model, existing provider | §1 |
| New **version** of an existing model | §0.1, then §1 |
| New **provider** (new host / key) | §0.2, then §1 |
| **Gateway** route (LiteLLM) | §0.3 |
| Same id, vendor swapped the weights | §0.4 |

### 0.1 New version

- **New entry, new exact id.** Ids are matched exactly (`registry.go`
  `Lookup`, `TestLookup_IsExactNotPrefix`): `glm-5.3-flash` says nothing about
  `glm-5.4-flash`. Never edit the old entry in place; it describes a model that
  may still be served. It and its `TestDefault_MeasuredFactsMatchFindings`
  subtests stay until the endpoint stops serving the old id.
- **Copy nothing as measured.** A copied entry that still says
  `verified: measured` is a lie until re-probed. Either every probed bit is
  re-measured, or the field is absent (absent = assume nothing, never "assume
  the previous version").
- Diff old findings against new and record what moved in the new FINDINGS.md
  section. A version bump that changes `effort_values` or `max_tokens_param` is
  exactly what this file exists to catch.

### 0.2 New provider

- `provider:` matches `^[a-z][a-z0-9]*$` (`profile.go` `providerName`). It
  derives `LLMWIRE_<PROVIDER>_API_KEY` (required) and, only for a provider
  that ships no host, `LLMWIRE_<PROVIDER>_BASE_URL` (refused beside a shipped
  host); two profiles on one host share them.
- **Add the host to `providers:` in `profiles.yaml`** with a comment naming
  which of the vendor's hosts it is (general vs plan-specific, and what the
  other ones refuse). Root only, https, no query string; `registry.go`
  `Provider.validate` refuses the rest. A self-hosted gateway gets `{}` and
  its URL becomes required from the environment. Apps never carry the URL.
- Add the `_API_KEY` line to `.env.example` (name only, empty value). A host
  that authenticates nothing gets `no_api_key: true` and no `_API_KEY`
  variable: an empty bearer is not sent, some gateways reject it.
- Add an `endpoint` var next to `mimoEndpoint` / `zaiEndpoint` in
  `eval_harness_test.go`; it reads the shipped host and fails the run if the
  URL variable is set beside it. A missing key skips that provider's probes
  with a named reason; it never falls back.
- **New key prefix?** `hack/secret-scan.sh` (`PATTERN='(sk|tp)-...'`) and
  `Redact` by shape (`errors.go`, the `sk-`/`tp-` run) only know those two.
  A key with another prefix is invisible to the scan and to redaction: extend
  both, with a test, before the first probe logs anything.
- **User-Agent.** A host sold as one client's backend may refuse a neutral UA
  as a bot (`identity.go` head comment; MiMo's Token Plan host does). Probe
  with the default UA first; if refused, `emulate_opencode: true` on the
  `providers:` entry, so `FromEnv` presents as that client and no application
  has to know. `LLMWIRE_EMULATE_OPENCODE=1` forces it for any provider, e.g. a
  gateway in front of such a host; default off.
- **Key in the query string?** Some deployments carry it there
  (`api_key`, `key`, `token`, `access_token`) and their error bodies echo it.
  `Redact`/`RedactURL` handle the known names; a new name needs adding
  (`errors.go` head comment on query keys).

### 0.3 Gateway route

Usually not a profile entry at all: a deployment's gateway is declared in the
environment, `LLMWIRE_LITELLM_MODELS=<profile id>[=<alias>],...` beside the
litellm URL and key (`gateway.go`, README "Behind a gateway"). What gets
onboarded is the MODEL behind the alias, as a plain vendor profile; the route
is derived from it at `FromEnv` through the same `resolve`/`validate` path.
A hand-written route entry is for a gateway the library itself ships, and is
shaped as follows.

`base` + `gateway` + `provider` + `wire_model_id`, nothing else but the two
tighten-only sub-keys (`profile.go` `derivedAllowedNestedKeys`:
`streaming.needs_include_usage`, `tools.recover_inline_markup`). No `cost`
block: "a gateway route is priced from its response header, never from the
table" (`pricing.go` `CostBlock.validate`). `verified` is not inherited
(`TestNewRegistry_ProvenanceIsNotInherited`): the route was never probed.

Everything LiteLLM is **source-derived at v1.100.1 and unverified**
(FINDINGS.md "Still open"). The first real gateway onboarding must measure all
of it:

- `x-litellm-response-cost`: scientific notation (`1.23e-05`), absent on
  streams, `"0"` under a non-2xx means ERRORED not free (`pricing.go`
  `litellmCostHeader` and the parser below it).
- `x-litellm-model-name` (real deployment, e.g. `azure/...`) and
  `x-litellm-call-id`, returned on errors too (`client_test.go`).
- Budget exceeded arrives as 429 not 402; unknown model as 400 not 404;
  FastAPI `detail` as a string, an `{"error":…}` object, or an array
  (`errors.go`, the LiteLLM paragraphs).
- A keepalive ping forces the 200 before the upstream fails, so the error
  arrives as a frame inside the stream (`stream.go` head comment, FAILURE
  INSIDE A SUCCESS).
- Whether the proxy strips the usage chunk from a client that did not ask
  (`needs_include_usage`, the tighten-only override).

### 0.4 Same id, new weights

Re-run `-run Probe` for that provider, diff the log against the model's
FINDINGS.md section, and add a dated paragraph to it. Each section carries its
own "measured on" line; the file header date is the date of the last full run.

## 1. Before the first call

1. `.env` from `.env.example`, values bare. `LLMWIRE_EVAL=1` goes in
   the REAL environment, never in `.env`: it is the flag that makes the
   harness load the file (`eval_harness_test.go` `init`).
2. **Add the model to `evalRates` in `eval_harness_test.go` at the HIGHEST
   published rate.** `meter.record` charges an unknown model the whole
   remaining budget, so the breaker trips after call one and the failure reads
   as "spend ceiling reached", not "unknown model". No published rate (free
   preview, credits only)? Still an entry: a deliberate over-bound, the highest
   rate in the family or 10x a sibling. Never omitted.
3. Cost the run before it starts. Defaults: 60 calls, $0.50, 10 min
   (`LLMWIRE_EVAL_MAX_CALLS`, `_MAX_USD`, `_DEADLINE`; raise, never disable).
   The three-model run is 30 calls at about $0.008 plus the containment probe
   at 4 calls and about $0.025; a new chat model adds roughly the same again.
   `TestProbe_CachedTokensAreInsidePromptTokens` is the expensive one: two
   ~10k-token prompts per model. Compute it from the `evalRates` entry; a model
   priced in dollars per million rather than cents needs `_MAX_USD` raised.
4. Gather sources. The vendor's own pricing page (that URL is `source_url`,
   today is `verified_on`), API reference, model card. models.dev and
   LiteLLM's `model_prices_and_context_window.json` are a CROSS-CHECK, never a
   source: models.dev carried `glm-5.3-flash` at exactly half the vendor's
   rate from release day on, and a third-party doc carried `mimo-v2.5-pro` at
   double (`profiles.yaml`, both cost comments). Note every disagreement now;
   they go into the FINDINGS.md table in §6.
5. Draft the entry with `verified: source-derived` and **every unmeasured
   field absent**. An unknown key fails the load (`registry.go` `KnownFields`,
   `TestNewRegistry_UnknownKeyIsRejected`). The defaults applied at load are
   exactly these and no others: `endpoint: chat`, `wire_model_id: <id>`,
   `verified: source-derived`, `reasoning.stream_field: reasoning_content`,
   `tools.format: native` (`registry.go` `applyDefaults`). Run
   `go test -run 'Default_|NewRegistry_' ./...` before the first probe; the
   cross-field rules in §6 fail here, not in production.
6. Add the model to the tables of every probe you intend to run (§3, column
   "table"). The probes are hardcoded per model on purpose; there is no
   model-list env var.
7. For the question no log line answers (what did the endpoint actually
   send: raw usage field names, frame order, an error body's real shape), hand
   the probe client an `http.Client` whose transport is
   `NewSpoolTransport(dir, nil, nil)` (`spool.go`). One `.http` file per
   response, status line plus headers plus body as read. The directory holds
   prompts and answers: keep it out of the tree and out of the findings note.

## 2. Run order

```
LLMWIRE_EVAL=1 go test -run 'Probe_<Name>' -v ./...   # one probe
LLMWIRE_EVAL=1 go test -run Probe -v ./...            # all of them
```

`-run Eval` matches nothing; every live test is `TestProbe_*`.

1. **`TestProbe_OutputCapParameter` first.** Every other probe hardcodes a cap
   name. On a glm-shaped endpoint the wrong name is ACCEPTED and IGNORED: a
   cap of 16 returned 481 tokens with `finish_reason: stop`. A probe sending
   the wrong name is therefore uncapped, whatever `probeMaxTokens` says. Read
   the token COUNT off the log, never acceptance. Write `max_tokens_param` into the profile before running
   anything else.
2. **`TestProbe_UsageReportingWithoutStreamOptions` second.** The meter needs
   usage to meter. Note where the usage chunk lands: trailing after
   `finish_reason` with an empty `choices` array (MiMo), or in the same chunk
   (Z.ai); and whether the finish chunk carries `"usage": null` (`stream.go`
   head comment, USAGE).
3. Everything else, in any order. Read the spend summary `TestMain` prints; it
   goes into FINDINGS.md.
4. **Inconclusive is a `t.Skip`, never a pass.** `CachedTokensAreInsidePromptTokens`
   is the pattern: two of its four outcomes skip with the raw usage in the
   message. `accepted()` counts `finish_reason: length` as accepted (the
   harness's own cap fired). A probe fails only when it cannot reach a
   conclusion at all.

## 3. Probe matrix: existing probes

One row per profile bit an existing probe establishes. "Table" is the
per-model list inside the probe to extend. Values in the profile follow the
log line, never the vendor page.

| profile bit | probe | table | log → value | trap |
|---|---|---|---|---|
| `max_tokens_param` | `OutputCapParameter` | the `tc` list, both params | honoured=true for exactly one name → that name. Both honoured (MiMo) → the vendor-documented one, say so in the comment | accepted+ignored is silent (glm); a body carrying both names is uncapped |
| (regression) | `ChatHonoursTheOutputCap` | the `tc` list | runs through `Chat`, so needs the registry entry with `max_tokens_param` set. Fails when count > 2x cap | run AFTER the profile entry exists |
| `reasoning.can_be_disabled` | `ZaiThinkingCanBeDisabled` (glm) or `MiMoThinkingCanBeDisabled` (MiMo, per-model table) | copy the nearer probe, sending the control's OFF shape: `thinking:{"type":"disabled"}` for `toggle_object`, `reasoning_effort: none` for `effort` | REJECTED → `false`; ACCEPTED with the lane REPORTED and 0 → `true` | "not reported" is NOT zero: guard `Usage.Output.Reasoning != nil` before reading it, or a missing `completion_tokens_details` reads as proof the toggle worked. The V2.5 MiMo entries are still doc-derived; the V2.6 pair is measured. `enabled_by_default` is a SEPARATE bool, read it off a request with no knob at all |
| `reasoning.effort_values`, `default_effort` | `ZaiReasoningEffortValues` (pattern) | copy for the new model | the exact ACCEPTED set, nothing wider; token count per tier goes in the comment | one overloaded code (`1210`) rejects both bad tiers and the disable toggle; the message names a subset of what it rejects. `default_effort` is what the vendor says, not measured |
| `reasoning.control` accepted on chat | `MiMoAcceptsReasoningEffort` | copy for the new model | accepted → note "accepted, effect not established" unless reasoning tokens moved | accepted-and-ignored is indistinguishable from honoured on a one-word prompt |
| `tools.format`, `tools.recover_inline_markup` | `MiMoToolCallFormat` + `MiMoToolFreeCall` | the model list in each | native, no markup → `native`. Markup in content/reasoning → `xml`. `recover_inline_markup` is defensive, not a capability: ON wherever the serving family has been seen leaking (both MiMo profiles), since it acts only when markup appears | "not reproduced" is not "refuted": the production leak needed a long tool-saturated history the probe does not build (FINDINGS.md, question 1). Spool the raw stream (§1) if in doubt |
| `Message.ReasoningContent` replay | `MiMoReasoningReplayRequirement` | `const model` | rejected → the history builder must replay reasoning; record in FINDINGS, it is a type-shape finding not a profile bit | needs a native tool call in round one; skips otherwise |
| idle bound | `LongestCommentOnlyGap` | the `tc` list, with the model's cap name | `MaxCommentGap` vs `DefaultIdleTimeout` (90s). Above 45s → warn; the data-only re-arm rule needs a per-profile escape hatch | reasoning at max with a small cap produced 1022 frames and ZERO content: "the cap counts reasoning" |
| `streaming.needs_include_usage`, `streaming.accepts_stream_options` | `UsageReportingWithoutStreamOptions` | the `tc` list | usage without the option → `needs_include_usage: false`; option rejected → `accepts_stream_options: false` | `needs_include_usage: true` with `accepts_stream_options: false` fails the load |
| `output.json_object`, `output.json_schema`, `output.strict_schema` | `ResponseFormatSupport` | the `tc` list | accepted and `parses=true` → `true`. Strict is sent inside the schema block, so acceptance there is `strict_schema` | both vendors accept `json_schema` though neither documents it. A fenced reply can still arrive with the format set (`errors.go` `JSONObject` comment) |
| `vision` | `MiMoProRejectsImageInput` | the model list | rejected → `false`; accepted → `true` | the 404 body is SSE-framed (`data:{"error":…}`); `image_url` is a nested object, never a bare string |
| `cost.cache_read` lane, usage containment | `CachedTokensAreInsidePromptTokens` | the `tc` list | INSIDE → pricing subtracts, nothing to set. BESIDE → stop: pricing needs a containment flag before this model can be priced | read the vendor's raw usage field name off the `raw=` log; the first non-zero `cached_tokens` is what confirmed Z.ai's spelling |

Wire-level observations to take from ANY of the runs above and put in the
profile's `notes`: latency class (MiMo 25-64s per call vs glm 1-5s), an
injected system prompt showing up as cache hits on a tiny prompt (192 of 258
on MiMo), vendor `finish_reason` strings seen (`finish_reasons_extra`).

## 4. No probe exists: write one, never guess

Each of these is in the schema and in no probe. Writing the probe is part of
onboarding when the field matters for the model; leaving the field absent is
the alternative. Guessing is not.

| field | send | outcome → value | inconclusive |
|---|---|---|---|
| `tools.tool_choice_values`, `tools.supports_forced_choice` | a tool plus `tool_choice: {"type":"function","function":{"name":…}}`, then `"required"`, then `"none"` | a call to the named tool → forced supported; prose back with no error → DROPPED silently, list only what forced a change | the model calling the tool anyway under `auto` (`profiles.yaml` glm entry calls this "the weakest line here"; `validate.go` documents endpoints that DROP the field) |
| `tools.supports_parallel` | two independent tool questions in one prompt, two tools offered | two `tool_calls` in one turn → `true` | one call then stop is not evidence either way |
| `temperature.forced_value`, `top_p.forced_value` | a non-default value | 400 with `"param":"temperature"` → `forced_value` set to the one accepted value (gpt-5.x shape) | |
| `temperature.inert_while_reasoning` | temperature 0 vs 2 with thinking on, same prompt, compare outputs and any vendor statement | identical outputs plus a vendor doc saying "overridden" → `true` | outputs differ by chance; this bit is mostly doc-derived, say so |
| `temperature.recommended_value`, `top_p.recommended_value` | nothing; read the vendor's recommended operating point | vendor states it → set. Omission does NOT mean "the model's default": Z.ai falls back materially below its tuned point (`validate.go`, sampling) | |
| `reasoning.enabled_by_default` | a request with NO reasoning knob | reasoning tokens > 0 or a `reasoning_content` delta → `true` | |
| `reasoning.stream_field` | any streamed reasoning call; read the delta key | `reasoning_content` or `reasoning` (`stream.go` reads both, content wins) | |
| `reasoning.budget_param` | only for `control: budget_tokens`; the vendor's key name, then a tiny budget and read reasoning tokens | honoured → that name. Required for this control, forbidden elsewhere; there is no default because vendors disagree and a guessed key is ignored silently | |
| `limits.context`, `limits.max_output` | a cap above the documented `max_output` | every endpoint measured clamps silently (`validate.go`, "clamps silently"); the doc value stands, note the clamp | |
| `finish_reasons_extra` | nothing extra; collect every non-OpenAI string seen across the run | list them; the parser treats the set as open regardless | |
| tool-level `strict` | a tool with `"strict": true` | rejected → note it; no profile bit yet, `strict_schema` covers only the response_format block (`render.go`, `validate.go` "tool-level strictness") | |
| assistant turn carrying only `tool_calls` | history with `content: ""` vs `null` vs omitted | which shapes are accepted; `""` is sent today (`render.go`, unmeasured) | |
| empty content after an image drop | a text-only model, message whose only part was an image | accepted or 400 (`validate.go` `dropImages`, "whether these endpoints accept an empty content string") | |
| `embedding.dimensions`, `embedding.default_dimensions` | a call with no `dimensions`, read the vector length; then `dimensions: 256` | length → `default_dimensions`; shorter vector back → `dimensions: true`; 400 → `false` | zero embeddings probes exist today; `text-embedding-3-*` are source-derived |
| embeddings over-length, batch order | an input past the per-input ceiling; a batch of 3 | rejected not truncated; `data[].index` order (`embed.go`, "WHAT IS DELIBERATELY NOT CHECKED") | |
| `cost.cache_write` | the containment probe's second call, read `cache_write_tokens` if the vendor has it | 0 because "Limited-time Free" is a promotion, comment says so with the date | most vendors report no write lane at all |
| `cost.tiers` | a prompt past the documented tier threshold, compare the LiteLLM/vendor-reported cost | only with a vendor doc naming the threshold; `min_input_tokens` unique and positive, `multiplier_permille` >= 1000 | |
| `cost.windows`, `cost.outside_windows_permille` | nothing ships attached; both vendors document off-peak as a coefficient on CREDIT burn, not the USD lane (`pricing.go` `Windows` comment, FINDINGS "Still open") | attach only after a measurement shows the USD invoice moving; `zone` IANA, `from`/`to` `HH:MM` no midnight wrap, `days` Mon..Sun, no overlap | |

Probe-writing rules (`eval_probe_test.go` head comment and the harness):

- One question per probe. `t.Logf("FINDING ...")` IS the deliverable; the
  line becomes profile data and comment text.
- Fail only when no conclusion is reachable. Rejected is a finding. Skip when
  the answer cannot be read off the result.
- Raw `body` maps via `probeBody`, never a typed request: the point is to
  send shapes `Validate` would refuse.
- Every call through `stream()` so the meter sees it. No retries: a failed
  probe is data.
- `Redact` / `RedactURL` on everything logged. The findings note is a
  committed artefact, and vendor error bodies echo query-string keys.
- A cache probe carries a per-process nonce so a rerun inside the vendor's
  TTL starts cold (`cacheProbeNonce`).

## 5. Wire traps: check on every new endpoint

Each was learned once and is enforced at the site named. A new endpoint can
exhibit any of them; read the probe logs for these shapes and record what you
see in the profile comment, whether or not it matched.

| trap | enforcement site |
|---|---|
| error body SSE-framed under a non-2xx: `data:{"error":…}` | `errors.go` `unframeSSE` |
| `{"error":…}` under a 200; `"error": null` must NOT trip it | `chat.go` `parseChatResponseWith`; `stream.go` FAILURE INSIDE A SUCCESS |
| 200 with no choices, or an HTML proxy page | `chat.go` `parseChatResponseWith` |
| `code` as string (MiMo, LiteLLM), int declared but string sent (Z.ai), float | `errors.go` `decodeCode` |
| one code for unrelated conditions; message language varies between runs | dispatch on `code` only (`errors.go` head, Z.ai `1210`) |
| `[DONE]` never sent; clean EOF without `finish_reason` is truncation | `stream.go` head, PARSING |
| `data:` with no space after the colon | `stream.go` head, PARSING |
| usage chunk trailing with `choices: []`, or inline; `"usage": null` on the finish chunk | `stream.go` head, USAGE |
| tool-call fragments: `id`/`name` on the first only, `""` after; arguments accumulate by `index` | `stream.go` `toolCallDelta` |
| one SSE line over 1 MiB (a tool call carrying a whole file) | `stream.go` `maxStreamLine` |
| `: ping` keepalives: they do NOT re-arm the idle guard, so measure the longest comment-only gap | `stream.go` "re-armed on `data:` frames ONLY" |
| `reasoning_tokens` inside `completion_tokens`; `cached_tokens` inside `prompt_tokens`; `cached > prompt` clamped | `usage.go` head comment and `parseUsage` |
| a bare `{"total_tokens":N}` with no lanes, or an empty details object: reported but unpriceable | `usage.go` `Usage.Reported` comment |
| `completion_tokens_details.text_tokens` present: the reported figure wins over prompt minus reasoning (fixture only, never seen live) | `usage.go` `parseUsage` |
| unmodelled usage fields (image/video tokens, web-search counts) | `Usage.Raw` |
| non-streaming route withholds headers until the whole answer is ready (MiMo 25-64s) | `client.go` `RawPost`, "Bounded by the WHOLE-CALL cap" |
| HTTP/2 negotiated, so `Transport.ResponseHeaderTimeout` is inert | `client.go` `New`, `stream.go` guard comment |
| `Retry-After` as seconds or HTTP-date; past date is 0 | `errors.go` `retryAfter` |
| empty `Authorization` header rejected; send none instead | `client.go` `Config.APIKey` |
| key echoed back inside error bodies or `url.Error` text | `errors.go` `Redact`, `Client.scrub` |
| parts-array content rejected by a text-only model; plain string when there are no parts | `render.go` `renderMessages` |
| reasoning consumes the output cap (1022 frames, zero content, `finish_reason: length`) | `profiles.yaml` glm `notes` |

## 6. Record

### profiles.yaml

- `verified: measured` is profile-level: this profile was probed at all. The
  per-bit truth is a **MEASURED** / **NOT measured** marker in each bit's
  comment (pattern: the glm entry's `tool_choice_values`). Mark every new bit
  either way. The shipped entries predate this rule and carry a marker mostly
  where a measurement contradicted a doc; do not read their unmarked bits as
  measured.
- Comment standard (AGENTS.md): why, with the measurement. Vendor strings and
  codes verbatim, latency/token tables inline, a dead workaround labelled dead
  not deleted.
- Cross-field rules that fail the load (`profile.go` `validate`,
  `TestNewRegistry_RejectsContradictory*`): `can_be_disabled: false` with
  `none` in `effort_values`; `can_be_disabled: true` under `control: effort`
  with no `none`; effort control with empty `effort_values`; `default_effort`
  outside the set; supported but neither on by default nor switchable;
  `budget_param` present iff `control: budget_tokens`; `format: xml` with
  `supports_forced_choice`; tools supported with empty `tool_choice_values`;
  `strict_schema` without `json_schema`; `needs_include_usage` without
  `accepts_stream_options`; `max_output` > `context`; an embeddings profile
  declaring any chat capability, or missing `default_dimensions`.
- `cost`: all four chat lanes (`input`, `cache_read`, `cache_write`, `output`;
  explicit `0` for free, absent fails), `cache_read` <= `input`, `source_url`
  https, `verified_on` `YYYY-MM-DD`, decimal text (no exponent). Embeddings:
  `input` only, the other lanes ABSENT not 0. No rate at all → omit the block;
  the model prices as `Unpriced` with a warning.
- `notes`: the caveats a caller needs before the first call (latency class,
  the cap-counts-reasoning trap, injected system prompt).

### FINDINGS.md

A `### <model>` section: the bit / value / probe table, the verbatim
rejections, one "measured on" line, and the run's call count and USD from the
spend summary. New rows in "Where the documentation is wrong" for every
disagreement from §1 step 4. New bullets in "Still open" for every §4 field
left absent.

### Tests the model must join, or it is silently untested

| test | what to add |
|---|---|
| `registry_test.go` `TestDefault_MeasuredFactsMatchFindings` | one subtest per measured bit, naming the probe in the failure message |
| `pricing_load_test.go` `TestDefault_ShippedRatesMatchTheVendorPages` | the four lanes in nano-USD, and the no-window assertion |
| `pricing_load_test.go` `TestDefault_EmbeddingDimensionsMatchTheVendor` | embeddings only |
| `plan_test.go` `TestPlan_CapParamComesFromTheProfile` | the cap spelling |
| `render_test.go` `TestRender_OutputCapUsesTheProfilesSpelling` | the cap spelling |
| `eval_harness_test.go` `evalRates` | done in §1 |
| `eval_probe_test.go` `TestProbe_ChatHonoursTheOutputCap` | done in §3 |

Adding those Go rows is what makes the merge mint a release tag
(`release.yaml` triggers on `**.go`, `go.mod`, `go.sum`); a `profiles.yaml`-only
change ships no tag.

## 7. Ship

CI order: `gofmt -l .`, `./hack/secret-scan.sh`, `go vet ./...`,
`go build ./...`, `go test -race ./...`. Patch coverage 75% on non-comment
lines. Test fixtures assemble keys at runtime (`fakeKey`), never as literals.
No key, no `.env`, no full URL in anything committed, the findings note
included.

PR body: what was measured, what disagreed with the docs, what was left absent
and why, cost of the run.
