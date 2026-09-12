# AGENTS.md

## What this is

Dependency-light Go **library**: one package at repo root, no CLI, no server.
**One** wire protocol — OpenAI-compatible `/chat/completions` + `/embeddings`.
Anthropic, Gemini-native, Bedrock: permanently out of scope; a second protocol
needs the provider abstraction this module exists to avoid.

## Commands

CI order: `gofmt -l .` (prints nothing), `./hack/secret-scan.sh`, `go vet ./...`,
`go build ./...`, `go test -race ./...`. Go 1.25. No Makefile, no golangci-lint.

## Quirks are data

`profiles.yaml`, keyed by **exact** model id. Never prefix-match: `gpt-5.5` vs
`gpt-5.5-pro` differ, and a prefix makes a new model inherit an old family's
quirks.

- `Lookup` **errors on unknown id**: a missing flag reading as "unsupported" turns
  a typo into a capability downgrade.
- **Fully resolved at load.** No optionals past init, no call-site defaults;
  cross-field rules fail `go test`, not production.
- `EnabledByDefault` and `CanBeDisabled` are **separate bools**. Never infer
  "cannot disable" from a missing `none` — that inference is why models.dev has
  `glm-5.3-flash` wrong.
- `budget_param` required for `control: budget_tokens`, forbidden elsewhere: the
  vendors disagree on the name and a guessed key is ignored silently.

**Composition.** A gateway entry names a `base` and inherits it. Single base, no
chains. Capabilities only **narrow**; `wire_model_id` **replaces**; `cost` may not
be restated and is dropped for a gateway route. `base` is the real upstream
deployment, not the vendor flagship. Validate against base even where the gateway
would drop the param itself — stricter than the wire on purpose: a silently dropped
`reasoning_effort` is a 10x cost surprise.

## Request path

`plan(req, stream)` → coerced `ChatRequest` → `render.go` → wire JSON.

- Capability decisions live in `plan` **only**. The renderer reads wire names
  (`WireModelID`, `MaxTokensParam`, `Reasoning.Control`), never a capability field —
  grep `render.go` for `.Supported`, it must find nothing. Deciding twice gives two
  policies that drift.
- Every demoted (`BestEffort`) refusal must DROP what it refused on the coerced
  request, or the renderer sends the knob just refused.
- `Validate` is exported and read-only → the coerced copy is deep.
- Streaming is a `plan` parameter, never a request field: a request able to carry
  `Stream: true` into `Chat` is a contradiction with no correct resolution.
- `ExtraBody` merges last and **wins**; top level only.

## Transport

- **`http.Client.Timeout` never set** — it caps body reads, cutting long streams
  mid-answer. Bounds are header / stream-idle / whole-call and a failure names which
  fired. Every call path arms the guard, streaming or not.
- **Idle guard re-arms on `data:` frames only**, never blank lines or `: ping`,
  else a heartbeat masks a stalled model. (peeq re-armed on everything, loom on
  nothing — merged rule.)
- **Completion asserted, never inferred**: only `[DONE]` or `finish_reason`. A
  dropped connection ends the scan exactly like a finished one.
- The `Stream` reader goroutine must **never block**, so its queue is unbounded: a
  blocked sink stops re-arming the idle guard, and a slow consumer then reports as
  a stalled model. A caller-initiated `Close` is not an error.
- Wire mechanics that bit us once are documented at their enforcement site, not
  here. Read `stream.go`'s head comment before touching the parser.

## Errors

- **Dispatch on `code`, never message text.** Z.ai's 1210 body is Chinese live and
  that code is overloaded onto unrelated errors.
- `code` is a **string** on MiMo and LiteLLM; Z.ai declares int, sends string.
- **Parse leniently.** Never `DisallowUnknownFields`; accept `null` where the spec
  says number; `finish_reason` is an open set. An `{"error":…}` under a 200, and a
  200 with no choices, are errors — never an empty success.
- **Redaction is an invariant**: some deployments carry the key in a query string.
- **Never retry.** Parse `Retry-After` into a typed error and stop; a library-level
  retry turns a transient outage into a permanent failure for a job queue.

## Usage and cost

- Usage fields are **pointers**: "not reported" is not zero.
- `prompt_tokens` **includes** `cached_tokens`; `completion_tokens` **includes**
  `reasoning_tokens`. Subtract, never add — either way round overcharges.
- Priced **per call** with the model that ran; a total is a sum of per-call prices,
  never re-derived from summed tokens. Integer end to end.
- **Always USD, at the vendor's list pay-as-you-go rate.** Flat rates are not
  modelled: a call billed as plan credits still reports the list rate, since a credit
  figure compares between nothing. `FromTable` is an equivalent, **not an invoice**.
  No rate → `Unpriced` → 0 **with a warning**: 0 is unknown, not free.
- **Behind a gateway: `x-litellm-response-cost` or nothing** — never the table,
  either direction. A stream carries no such header.
- A `cost` block needs `source_url` + `verified_on` and all four chat lanes: rates
  move, two of our models have conflicting published prices, and a lane falling
  back to another lane's rate is a call-site default.
- **No window ships attached.** Both vendors document off-peak multipliers against
  credit burn, not the USD lane; an inapplicable 0.5x halves reported spend. Round
  **up**, once, on the total: under-reporting raises what a budget cap allows.
- Rate depends on request size and wall-clock time → **the clock is injectable**. A
  test passing only 14:00–18:00 Singapore is worse than no test.

## Secrets

Keys from the **environment only** — never a literal in a test, fixture or profile;
a missing key skips with a named reason, never falls back. `.env` is gitignored and
the eval harness loads it under `LLMWIRE_EVAL=1` (real environment wins);
`hack/secret-scan.sh` runs in the pre-commit hook **and** CI, since a hook protects
one machine. `.env.example` holds names with empty values. CI holds no key.

## Evals

`LLMWIRE_EVAL=1 go test -run Eval ./...`. Env-gated via `t.Skip`, **never a build
tag** — a tag hides code from `gofmt`, `go vet` and the compiler, and it rots.

**Evals are the source of truth for profile bits**: a vendor doc that disagrees
with a measurement loses. Record the measurement in the profile comment.

Spend guard mandatory, fail-closed: budget checked *before* each call, call ceiling
independent of cost, wall-clock deadline, minimal per-call caps. `Unpriced` counts as
the whole remaining budget. Budgets raise, never disable. **No retries** — a failed
eval is data. The guard keeps its own over-estimating rate table: a breaker sharing
the accounting table stops being conservative.

## Comment standard

Why, with the measurement. Quote error codes and vendor strings verbatim, keep
latency/token tables inline, label a dead workaround dead rather than deleting it.

## Git & CI

- Default branch `master`. Never push to `master` — branch + PR.
- A merge touching `**.go` / `go.mod` / `go.sum` auto-mints a semver tag: a merge
  is a release.
- Config files use `.yaml`, never `.yml`.
