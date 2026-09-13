# llmwire

A dependency-light Go library for talking to **OpenAI-compatible** inference
endpoints — chat, tools, vision and embeddings — with each model's quirks kept as
data rather than scattered through call sites.

> **Status: under construction.** `Chat`, `ChatStream` and `Embed` work against
> the profiled endpoints; pricing and the LiteLLM gateway are still landing. The
> API is not stable yet.

```go
// Base URL and key come from LLMWIRE_<PROVIDER>_BASE_URL and _API_KEY, with
// the provider named by the model's profile (zai here, mimo and openai for
// the others), so one .env serves every application and none repeats the
// wiring. New(Config{...}) is there for a custom transport or an explicit URL.
client, err := llmwire.FromEnv("glm-5.3-flash", llmwire.Config{})
if err != nil {
    return err // names the missing variable
}

resp, warnings, err := client.Chat(ctx, llmwire.ChatRequest{
    Model:     "glm-5.3-flash",
    Messages:  []llmwire.Message{llmwire.System("..."), llmwire.User(q)},
    Reasoning: llmwire.ReasoningEffort("high"),
    MaxTokens: &cap,
})
```

Streaming is an iterator, so an early return closes cleanly and the usage that
only arrives in the final chunk has somewhere to live:

```go
stream, warnings, err := client.ChatStream(ctx, req)
if err != nil {
    return err
}
defer stream.Close()

for stream.Next() {
    if ev := stream.Event(); ev.Kind == llmwire.EventContent {
        fmt.Print(ev.Text)
    }
}
if err := stream.Err(); err != nil {
    return err
}
cost := stream.Usage().Cost
ttft := stream.Result().Timing.FirstData // also Headers and Total, on every response
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
broken out, and priced per call with the model that actually ran, in integer
nano-USD. The rate is a function, not a constant: it varies by request size
(long-context surcharges) and by wall-clock time (off-peak windows, evaluated in
the provider's own timezone). Figures are always the vendor's published
pay-as-you-go rate — a comparable equivalent, **not an invoice**, since a call
routed over a subscription host is billed in plan credits nobody can compare
across models. Behind a gateway the cost comes from the proxy's own header or not
at all. A rate we have not verified reports `Unpriced` and warns, because zero
means unknown, not free.

**Recovers tool calls a model wrote as markup.** Some deployments answer with
`<tool_call>…</tool_call>` blocks (or an invented `<tool_invocation …/>`) in the
content instead of the native `tool_calls` field, on tool-free calls too. A
profile with `tools.recover_inline_markup` has that markup withheld from streamed
deltas, the calls surfaced as tool-call events (name first, arguments when they
land, `EventFinish` last) and cut from the returned text, with a `Warning` saying
what was recovered or that text was cut with nothing coming out of it. The
returned text is what streamed: everything from the first marker on is gone.

**Shows what the endpoint really sent.** `NewSpoolTransport(dir, next, skip)` is
an `http.RoundTripper` that writes every response — status line, headers, body
as read, a truncated stream included — to one `.http` file per call. Hand it to
`Config.HTTPClient` while diagnosing; `skip` keeps a request the caller promised
to keep ephemeral off disk.

**Handles the stream cases that bite.** Separate header / idle / whole-call
bounds, and a failure says which one fired. Completion is asserted by `[DONE]` or
`finish_reason`, never inferred from a clean EOF — a connection dropping mid-answer
looks exactly like a finished one. An `error` frame arriving inside a `200` stream
is surfaced rather than read as an empty answer.

**Can present as opencode.** Some endpoints are sold as one client's backend and
treat a neutral `User-Agent` as a bot. `Config{EmulateOpenCode: true}` sends
opencode's client string and its session header pair; headers only, never a
request body. The session id is the client's own — minted at construction,
rotated after a 30-minute idle gap, the way a person's session starts and ends —
and there is deliberately no way to supply one.

**Carries the small helpers every consumer was copying.** `Stream.Collect(onDelta)`
drains a stream into its result. `JSONObject(s)` cuts the first brace-balanced
object out of a reply that was asked for JSON and came back fenced or wrapped in
prose. `Registry.LookupEmbedding(id)` is `Lookup` that refuses a chat model, for
an embeddings client built from a constant. `Redact` and `Truncate` are exported
for the same reason.

## Scope

One wire protocol: `/chat/completions` and `/embeddings`. Anthropic, Gemini-native
and Bedrock are permanently out of scope — supporting a second protocol would
require the provider abstraction this library exists to avoid.

Also out of scope, by design: vector stores, chunking, embedding caches, retry
policy and prompt content. llmwire never retries; it parses `Retry-After` into a
typed error and leaves the decision to you, because a library-level retry turns a
transient outage into a permanent failure for a whole job queue.

## Adding a model

Add an entry to `profiles.yaml` and run the evals against it; `ONBOARDING.md`
is the step-by-step procedure, including which probe establishes which field
and what to record where. A model's behaviour is never inferred from its name —
ids are matched exactly, and an unknown id is an error rather than a silent set
of defaults.

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
  cost:                                    # USD per 1M tokens, the vendor's own
    input: 0.435                           # decimals, converted exactly from text
    cache_read: 0.0036
    cache_write: 0                         # "limited-time free" as of verified_on
    output: 0.87
    source_url: https://mimo.mi.com/docs/en-US/price/pay-as-you-go
    verified_on: 2026-09-12
```

All four chat lanes are required, or the block is left out entirely and the model
reports `Unpriced`: a lane falling back to another lane's rate would be a
call-site default, and this registry resolves everything at load.

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
  provider: litellm            # LLMWIRE_LITELLM_BASE_URL and _API_KEY
  wire_model_id: ai-gateway/gpt-5.4   # the proxy's alias
```

A gateway entry carries no `cost` block and inherits none: the proxy reports real
per-call spend in a response header, and it knows which deployment ran and at what
discount, where this table does not. No header, no price — never a list rate
standing in for one.

Because the entry names its base, validation still works behind a proxy whose wire
id is opaque: `Temperature(0.3)` is rejected locally, since gpt-5.4 rejects it.

## Evals

Profile data is verified against live endpoints, not taken from documentation —
published catalogues disagree with each other and with the endpoints.

```
LLMWIRE_EVAL=1 go test -run Probe ./...
```

Evals skip unless that variable is set, so they never run in CI. With it set, the
harness loads a gitignored `.env` (see `.env.example` for the names; a value
already in the environment wins). They call real,
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
