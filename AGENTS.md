# AGENTS.md

## What this is

Dependency-light Go **library**, one package at repo root. **One** wire protocol:
OpenAI-compatible `/chat/completions` + `/embeddings`. Anthropic, Gemini-native,
Bedrock: permanently out of scope.

## Commands

CI order: `gofmt -l .`, `./hack/secret-scan.sh`, `go vet ./...`, `go build ./...`,
`go test -race ./...`. Go 1.25. No Makefile, no golangci-lint.

## Quirks are data

`profiles.yaml`, keyed by **exact** model id. Never prefix-match: `gpt-5.5` vs
`gpt-5.5-pro` differ.

- `Lookup` **errors on unknown id**: a typo must not become a capability downgrade.
- **Fully resolved at load.** No optionals past init, no call-site defaults;
  cross-field rules fail `go test`.
- `EnabledByDefault` and `CanBeDisabled` are **separate bools**. Never infer
  "cannot disable" from a missing `none` (models.dev has `glm-5.3-flash` wrong this way).
- `budget_param` required for `control: budget_tokens`, forbidden elsewhere: a
  guessed key is ignored silently.
- **Composition**: gateway entry names a `base`, single level. Capabilities only
  **narrow**; `wire_model_id` **replaces**; `cost` never restated. Validate against
  base even where the gateway would drop the param: a silently dropped
  `reasoning_effort` is a 10x cost surprise.

## Request path

`plan(req, stream)` → coerced `ChatRequest` → `render.go` → wire JSON.

- Capability decisions in `plan` **only**. `grep render.go .Supported` must find
  nothing.
- A demoted (`BestEffort`) refusal must DROP the knob on the coerced request.
- `Validate` is exported and read-only → the coerced copy is deep.
- Streaming is a `plan` parameter, never a request field.
- `ExtraBody` merges last and **wins**; top level only.

## Transport

- **`http.Client.Timeout` never set**: it caps body reads. Bounds are header /
  stream-idle / whole-call; a failure names which fired. Every path arms the guard.
- **Idle guard re-arms on `data:` frames only**, never blank lines or `: ping`,
  else a heartbeat masks a stalled model.
- **Completion asserted, never inferred**: only `[DONE]` or `finish_reason`.
- `Stream` reader goroutine **never blocks** (unbounded queue): a blocked sink stops
  re-arming the guard. Caller `Close` is not an error.
- `Timing` measured with `Config.Now`; set on error paths too.
- Logging is `observe.go`, one line per call via `Client.finish` (Debug ok, Warn
  failed/truncated/empty/unaccounted) + `Stats()`. Flat keys, `_ms` ints, never
  prompt/answer/key text. Every new call path ends in `finish`, every error path.
  `ListModels` is the one exception: no plan, no usage, no model, so it logs its
  own line instead of polluting per-model stats.
- `rawCall(method)` is the only non-streaming exchange; a GET (models listing)
  sends no `Content-Type`. Gateway limits in the listing decode via `limitField`:
  integral float ok, anything else present → nil + `Warning`, never a cast.
- Wire mechanics live at their enforcement site. Read `stream.go`'s head comment
  before touching the parser.

## Errors

- **Dispatch on `code`, never message text.** Z.ai 1210 body is Chinese live and
  overloaded onto unrelated errors.
- `code` is a **string** on MiMo and LiteLLM; Z.ai declares int, sends string.
- **Parse leniently.** Never `DisallowUnknownFields`; `null` where spec says
  number; `finish_reason` open set. `{"error":…}` under 200, or 200 with no
  choices: errors, never empty success.
- **Redaction is an invariant**: by shape (`Redact`) AND by value (`Client.scrub`);
  a dial failure's `url.Error` text never prints the URL.
- **Never retry.** Parse `Retry-After` into a typed error and stop.
- Non-answer 2xx wraps a sentinel: `ErrMalformedResponse` (undecodable, stream cut
  early) or `ErrResponseShape` (decoded, wrong shape). Callers `errors.Is`.

## Usage and cost

- Usage fields are **pointers**: "not reported" is not zero.
- `prompt_tokens` **includes** `cached_tokens`; `completion_tokens` **includes**
  `reasoning_tokens`. Subtract, never add.
- Priced **per call** with the model that ran; totals sum prices, never tokens.
  Integer nano-USD end to end.
- **Always USD at the vendor's list rate.** Plan credits still report list rate;
  `FromTable` is an equivalent, **not an invoice**. No rate → `Unpriced` → 0 **with
  a warning**.
- **Behind a gateway: `usage.cost` first, then `x-litellm-response-cost`, else
  nothing.** The HEADER is written before the body is consumed, so a STREAM
  never carries it (LiteLLM #12689) — the figure rides on `usage.cost` in the
  final chunk, and only with `include_cost_in_streaming_usage` set gateway-side.
  Body wins: LiteLLM has shipped the header as `0` on streams, and recording
  that is a confident zero for the most expensive call in a turn. Neither lane
  → `Unpriced`, never the table.
  Literal `None`/`null` = not reported (LiteLLM sets headers via `str()`), never
  corrupt. Other proxy headers → `Gateway` block on the response
  (`gatewayresp.go`): call id, deployment, key spend (**gauge**, never summed).
- A `cost` block needs `source_url` + `verified_on` and all four chat lanes.
- **No off-peak window ships attached**: vendors document them against credit burn,
  not USD. Round **up**, once, on the total.
- Rate depends on size and wall-clock → **clock is injectable**.

## Secrets

Keys from the **environment only**, never a literal anywhere; missing key skips
with a named reason. **Hosts live in `profiles.yaml` `providers:`**, never in an
app's config: an app supplies `LLMWIRE_<PROVIDER>_API_KEY` and nothing else.
`LLMWIRE_<PROVIDER>_BASE_URL` exists only for a provider that ships no host
(litellm); beside a shipped host it is **refused** (`StaleEnvError`): no
override, a different host is a different provider entry. **A gateway is
env, not profiles**: `LLMWIRE_LITELLM_MODELS=<id>[=<alias>],...` routes listed
profiles through litellm (`gateway.go` `viaGateway`, same resolve/validate as a
YAML route); aliases are the operator's, never shipped. Read by `FromEnv` and the evals. `.env` gitignored, loaded under
`LLMWIRE_EVAL=1` (real env wins). `hack/secret-scan.sh` runs pre-commit **and** CI.
CI holds no key.

## Evals

`LLMWIRE_EVAL=1 go test -run Probe ./...`. Gated via `t.Skip`, **never a build tag**
(hides code from vet and the compiler).

**Evals are the source of truth for profile bits**; vendor doc loses to a
measurement, recorded in the profile comment.

New model, version, provider or gateway route → `ONBOARDING.md`, top to bottom.

Spend guard fail-closed: budget checked *before* each call, call ceiling, deadline,
per-call caps. `Unpriced` counts as the whole remaining budget. **No retries**: a
failed eval is data. Guard keeps its own over-estimating rate table.

## Comment standard

Why, with the measurement. Vendor strings verbatim, latency/token tables inline,
dead workaround labelled dead rather than deleted.

## Git & CI

- Default branch `master`; branch + PR, never push to `master`.
- A merge touching `**.go` / `go.mod` / `go.sum` auto-mints a semver tag.
- `.yaml`, never `.yml`.
