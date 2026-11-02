# AGENTS.md

## Quirks are data

`profiles.yaml`, keyed by **exact** model id. Never prefix-match: `gpt-5.5` vs
`gpt-5.5-pro` differ. Unknown id errors: a typo must not become a capability
downgrade. Fully resolved at load: no call-site defaults.

- `EnabledByDefault` and `CanBeDisabled` are **separate bools**. Never infer
  "cannot disable" from a missing `none` (models.dev has `glm-5.3-flash` wrong this way).
- `budget_param` exists because a guessed key is ignored silently.
- New vendor effort level → place it on `effortLadder` (`profile.go`) first:
  `ReasoningMinimal` reads `effort_values` entry 0 as the floor.
- `overhead` absent unless measured; `DefaultReasoningOverhead` (1024) is known
  short at glm max.
- Clients express **intents** (`ReasoningMinimal/Balanced`, `MaxAnswerTokens`,
  `Needs`), never a model's level names or caps. A model fact a client needs
  goes in the profile, not in the client.
- Gateway entries validate against `base` even where the gateway would drop the
  param: a silently dropped `reasoning_effort` is a 10x cost surprise.

## Request path

- Capability decisions in `plan` **only**; `render.go` never reads `.Supported`.
- A demoted (`BestEffort`) refusal must DROP the knob on the coerced request.
- `Validate` is exported and read-only → the coerced copy is deep.
- Tool loops append `resp.Assistant()`, never a rebuilt turn (encrypted
  reasoning on Responses, signed thinking on Messages). Read the head comment of
  `responses.go`, `messages.go`, `stream.go` before touching them.

## Transport

- **`http.Client.Timeout` never set**: it caps body reads. Every path arms the guard.
- **Idle guard re-arms on `data:` frames only**, never blank lines, `: ping` or
  Messages `ping`, else a heartbeat masks a stalled model.
- **Completion asserted, never inferred** from a clean EOF.
- `Stream` reader goroutine **never blocks**: a blocked sink stops re-arming the guard.
- Every new call path and every error path ends in `Client.finish` (one log
  line + `Stats()`). Never prompt/answer/key text. `ListModels` logs its own
  line; `RawStream`/`RawPost` log nothing (probe entry points).

## Errors

- **Dispatch on `code`, never message text.** Z.ai 1210 body is Chinese live and
  overloaded onto unrelated errors.
- `code` is a **string** on MiMo and LiteLLM; Z.ai declares int, sends string.
- **Parse leniently.** `{"error":…}` under 200, or 200 with no choices: errors,
  never empty success.
- **Redaction is an invariant**: by shape (`Redact`) AND by value (`Client.scrub`);
  a dial failure's `url.Error` text never prints the URL.
- **Never retry.** Parse `Retry-After` into a typed error and stop.

## Usage and cost

- Usage fields are **pointers**: "not reported" is not zero.
- `prompt_tokens` **includes** `cached_tokens`; `completion_tokens` **includes**
  `reasoning_tokens`. Subtract, never add.
- Priced **per call** with the model that ran; totals sum prices, never tokens.
- **Always USD at the vendor's list rate**, plan credits too: an equivalent,
  **not an invoice**. No rate → `Unpriced` → 0 **with a warning**.
- Behind a gateway: proxy's figure or nothing, never the table (`priceFromGateway`).
  **Still unguarded**: header `0` with `include_cost_in_streaming_usage` OFF (the
  default) records `Reported 0`; nothing distinguishes it from a free call.
- **No off-peak window ships attached**: vendors document them against credit burn,
  not USD.

## Secrets

Keys from the **environment only**, never a literal anywhere. **Hosts and
provider headers live in `profiles.yaml` `providers:`**, never in an app's
config: an app supplies `LLMWIRE_<PROVIDER>_API_KEY` and nothing else. A gateway
is env, not profiles; aliases are the operator's, never shipped. Secret scan
runs in `ship check`; pushes straight to a branch without a PR (GitHub-side
edits) pass unscanned. CI holds no key.

## Evals

**Evals are the source of truth for profile bits**; vendor doc loses to a
measurement, recorded in the profile comment.

New model, version, provider or gateway route → `ONBOARDING.md`, top to bottom.

## Comment standard

Why, with the measurement. Vendor strings verbatim, latency/token tables inline,
dead workaround labelled dead rather than deleted.

## Git & CI

- Gate: `ship check` (`ship` is the `ship()` from ship's README). Fresh clone:
  `ship hooks` before the first push; without the pre-push hook a push skips the gate.
- Branch + PR, never push to `master`.
- Commit subjects + PR titles: Go style `area: what` (`profiles: …`, `stream: …`),
  gated by ship; release notes list them by area.
- A merge ships nothing: release by hand only (`gh workflow run Release -f bump=patch`).
- `.yaml`, never `.yml`.
