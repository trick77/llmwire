# llmwire

A dependency-light Go library for talking to **OpenAI-compatible** inference
endpoints — chat, tools, vision and embeddings — with each model's quirks kept as
data rather than scattered through call sites.

> **Status: under construction.** Phase 0 (transport + live probes) is in
> progress. The API below is the target shape and is not stable yet.

```go
resp, warnings, err := client.Chat(ctx, llmwire.ChatRequest{
    Model:     "glm-5.3-flash",
    Messages:  []llmwire.Message{{Role: "system", Content: "..."}, {Role: "user", Content: q}},
    Reasoning: llmwire.ReasoningEffort("high"),
})
```

## Why

Every OpenAI-compatible endpoint bends the protocol somewhere, and the bends are
not discoverable from the spec:

- `mimo-v2.5-pro` takes `max_completion_tokens`; `glm-5.3-flash` takes `max_tokens`.
- `mimo-v2.5-pro` returns 404 for any `image_url` part. Its sibling `mimo-v2.5`
  does not.
- On `glm-5.3-flash`, a prompt that says "reply as JSON" parses 0 times out of 8;
  the same prompt with `response_format` parses 8 out of 8.
- gpt-5.x rejects a non-default `temperature` outright, with a 400.
- Some models cannot turn reasoning off at all.

Hard-coding those into one application means the next application relearns them —
usually in production. llmwire keeps them in a profile registry so a wrong
assumption fails locally, before the request is sent.

## What it does

**Fails fast on an impossible request.** Ask a model to disable thinking when it
cannot, and you get an error before a socket is opened, naming what *is* accepted:

```
llmwire: model "glm-5.3-flash": thinking cannot be disabled on this model;
         use ReasoningEffort with one of [low high max], or pass BestEffort()
```

`Validate(req)` is exported and needs no network, so a service can check its
configuration at boot.

**Three tiers, by how lossy the fix is.** A hard error when semantics would change
and you asked explicitly. A silent coercion when it is lossless (`max_tokens` →
`max_completion_tokens`). A coercion plus a returned `Warning` when the request can
be honoured approximately (`json_schema` → `json_object`). `BestEffort()` demotes
the first tier to the third for a single call.

**Reports what a call cost.** Usage is captured with its cache and reasoning lanes
broken out, and priced per call with the model that actually ran. The rate is a
function, not a constant — it varies by request size (long-context surcharges), by
wall-clock time (off-peak windows), and by unit, because subscription endpoints
bill credits rather than dollars. A rate we have not verified reports `Unpriced`
and warns, because zero means unknown, not free.

**Handles the stream cases that bite.** Separate header / idle / whole-call
bounds, and a failure says which one fired. Completion is asserted by `[DONE]` or
`finish_reason`, never inferred from a clean EOF — a connection dropping mid-answer
looks exactly like a finished one. An `error` frame arriving inside a `200` stream
is surfaced rather than read as an empty answer.

## Scope

One wire protocol: `/chat/completions` and `/embeddings`. Anthropic, Gemini-native
and Bedrock are permanently out of scope — supporting a second protocol would
require the provider abstraction this library exists to avoid.

Also out of scope, by design: vector stores, chunking, embedding caches, retry
policy and prompt content. llmwire never retries; it parses `Retry-After` into a
typed error and leaves the decision to you, because a library-level retry turns a
transient outage into a permanent failure for a whole job queue.

## Adding a model

Add an entry to `profiles.yaml` and run the evals against it. A model's behaviour
is never inferred from its name — ids are matched exactly, and an unknown id is an
error rather than a silent set of defaults.

```yaml
- id: mimo-v2.5-pro
  endpoint: chat
  max_tokens_param: max_completion_tokens  # verified live: a cap of 8 returned
                                           # exactly 8, finish_reason=length
  reasoning:
    supported: true
    enabled_by_default: true               # thinking defaults ON
    can_be_disabled: true
    control: toggle_object                 # thinking:{type}, not reasoning_effort
    stream_field: reasoning_content
  vision: false                            # 404s on any image_url part
  cost:
    unit: usd
    source_url: https://mimo.mi.com/docs/en-US/price/pay-as-you-go
    verified_on: 2026-09-12
    input: 0.435
    cache_read: 0.0036
    output: 0.87
```

`enabled_by_default` and `can_be_disabled` are deliberately separate flags. "Can
this model turn reasoning off" is not answerable from whether `none` appears in a
list of effort levels — inferring it is how published model catalogues get it
wrong.

### Behind a gateway

A LiteLLM entry is not a new model. It names the model it really is, inherits
every capability, and adds only what the gateway changes:

```yaml
- id: litellm/openai-gpt-5.4
  base: azure/gpt-5.4          # inherits capabilities and parameters
  gateway: litellm
  wire_model_id: ai-gateway/gpt-5.4   # the proxy's alias
  cost:
    source: reported           # the gateway reports real spend per call
```

Because the entry names its base, validation still works behind a proxy whose wire
id is opaque: `Temperature(0.3)` is rejected locally, since gpt-5.4 rejects it.

## Evals

Profile data is verified against live endpoints, not taken from documentation —
published catalogues disagree with each other and with the endpoints.

```
LLMWIRE_EVAL=1 go test -run Eval ./...
```

Evals skip unless that variable is set, so they never run in CI. They call real,
paid endpoints, so the suite is bounded by a spend budget, a call ceiling, a
wall-clock deadline and minimal per-call caps, all fail-closed. It does not retry:
a failed eval is a finding, not a flake.

## Development

```
gofmt -l .          # must print nothing
./hack/secret-scan.sh
go vet ./...
go build ./...
go test -race ./...
```

Keys are read from the environment only. `.env` is gitignored, and
`hack/secret-scan.sh` refuses keys and `.env` files entering the tree — from the
pre-commit hook and from CI, since a hook only protects one machine.

## Licence

MIT.
