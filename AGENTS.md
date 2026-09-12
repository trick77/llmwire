# AGENTS.md

## What this is

Dependency-light Go **library**: one package at repo root, no CLI, no server,
in-process. Speaks **one** wire protocol — OpenAI-compatible `/chat/completions`
+ `/embeddings`. Anthropic, Gemini-native, Bedrock, any non-OpenAI shape:
permanently out of scope. A second protocol needs a provider abstraction — the
design this module avoids.

## Commands

CI order:

```
gofmt -l .          # must print nothing
./hack/secret-scan.sh
go vet ./...
go build ./...
go test -race ./...
```

Go 1.25. No Makefile, no golangci-lint — `gofmt` + `go vet` are the only style gates.

## Quirks are data

`profiles.yaml`, keyed by **exact** model id. Never glob or prefix-match:
`gpt-5.5` vs `gpt-5.5-pro` differ, and prefixes make a new model inherit an old
family's quirks.

- `Lookup` **errors on unknown id**; a missing flag must never read as
  "unsupported", or a typo becomes a capability downgrade.
- **Fully resolved at load**: no optionals past init, no call-site defaults.
- Cross-field rules validated at load → malformed profile fails `go test`, not
  production.
- `EnabledByDefault` and `CanBeDisabled` are **separate bools**. Never infer
  "cannot disable" from `none` missing in an effort list — that inference is why
  models.dev has `glm-5.3-flash` wrong.

**Composition.** Gateway entry names a `base` and inherits it. Single base, no
chains. Capability fields only **narrow** (remove, never add); `wire_model_id`,
`cost.source`, `cost.unit` **replace**. `base` is the real upstream deployment, not
the vendor flagship. Validate against base even where the gateway would itself drop
the param — deliberately stricter than the wire: a silently dropped
`reasoning_effort` is a 10x cost surprise.

## Transport

- **`http.Client.Timeout` never set** — it caps body reads, cutting long streams
  mid-answer. Bounds are header / stream-idle / whole-call; a failure names which
  fired.
- **Idle guard re-arms on `data:` frames only**, never blank lines or `: ping`,
  else a heartbeat-emitting upstream masks a stalled model. (peeq re-armed on
  everything, loom on nothing — merged rule.)
- **Completion asserted, never inferred**: only `[DONE]` or `finish_reason`. A
  dropped connection ends the scan exactly like a finished one, so accepting what
  arrived persists a truncated answer as success.
- Wire mechanics that bit us once are documented at their enforcement site, not
  here — the `data:` prefix form, the error frame inside a 200 stream, the unset
  `Accept-Encoding`. Read `stream.go`'s head comment before touching the parser.

## Errors

- **Dispatch on `code`, never message text.** Z.ai's 1210 body is Chinese live,
  and that code is overloaded onto unrelated errors.
- `code` is a **string** on MiMo and LiteLLM; Z.ai declares int, sends string.
  Accept both.
- **Parse leniently.** Never `DisallowUnknownFields`; accept `null` where spec says
  number; `finish_reason` is an open set.
- **Redaction is an invariant** — every error, log line and observation passes
  through it; some deployments carry the key in a query string. URLs kept
  host-and-path only.
- **Never retry.** Parse `Retry-After` into a typed error and stop; library-level
  retry turns a transient outage into a permanent failure for a whole job queue.

## Usage and cost

- Usage fields are **pointers**: "not reported" is not zero.
- `prompt_tokens` **includes** `cached_tokens`; `completion_tokens` **includes**
  `reasoning_tokens`. Subtract, never add — either backwards overcharges.
- Priced **per call at the call site** with the model that ran; a total is a sum
  of per-call prices, never re-derived from summed tokens.
- Integer arithmetic end to end — a float total drifts in the digits displayed.
- **Credits are never rendered as USD.** Token-plan hosts bill subscription
  credits. Unverified rate → `Unpriced` → 0 **with a warning**: 0 means unknown,
  never free.
- Every `cost` block needs `unit`, `source_url`, `verified_on`: rates move, and
  two of our models have conflicting published prices.
- Rate depends on request size and wall-clock time → **the clock is injectable**.
  A test passing only 14:00–18:00 Singapore is worse than no test.

## Secrets

Keys from the **environment only** — never a literal in a test, fixture, profile
or default; a missing key skips with a named reason, never falls back. `.env` is
gitignored; `hack/secret-scan.sh` runs in the pre-commit hook **and** CI, because a
hook protects one machine only. `.env.example` carries names with empty values. CI
holds no key.

## Evals

`LLMWIRE_EVAL=1 go test -run Eval ./...`. Env-gated via `t.Skip`, **never a build
tag** — a tag hides code from `gofmt`, `go vet` and the compiler, and it rots.

**Evals are the source of truth for profile bits**: a vendor doc that disagrees
with a measurement loses. Record the measurement in the profile comment.

Spend guard mandatory, fail-closed: budget checked *before* each call, call
ceiling independent of cost, wall-clock deadline, minimal per-call caps.
`Unpriced` counts as the whole remaining budget, never zero. Budgets raise, never
disable. **No retries** — a failed eval is data.

## Comment standard

Why, with the measurement. Quote error codes and vendor strings verbatim, keep
latency/token tables inline, label a dead workaround dead rather than deleting it —
inert and explained beats rediscovered.

## Git & CI

- Default branch `master`. Never push to `master` — branch + PR.
- Merge to `master` touching `**.go` / `go.mod` / `go.sum` auto-mints a semver tag:
  a merge is a release.
- Workflow and config files use `.yaml`, never `.yml`.
