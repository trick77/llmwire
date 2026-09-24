# llmwire

A dependency-light Go library for talking to **OpenAI-compatible** inference
endpoints — chat, tools, vision and embeddings — with each model's quirks kept as
data rather than scattered through call sites. It is an adapter over model
dialects, not over wire protocols: one request contract, one protocol, and the
per-model bends live in `profiles.yaml`.

```go
// The host comes from the model's profile (profiles.yaml carries each
// vendor's base URL); the key from LLMWIRE_<PROVIDER>_API_KEY, the provider
// named by the profile (zai here, mimo and openai for the others). An
// application configures a key and nothing else; a LLMWIRE_<PROVIDER>_BASE_URL
// beside a shipped host is refused as the old contract. New(Config{...}) is
// there for a custom transport or an explicit URL.
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
         accepted: low, high, max; or pass BestEffort to send the nearest supported request
```

`Validate(req)` is exported and needs no network, so a service can check its
configuration at boot.

**Three tiers, by how lossy the fix is.** A hard error when semantics would change
and you asked explicitly. A silent coercion when it is lossless (`max_tokens` →
`max_completion_tokens`). A coercion plus a returned `Warning` when the request can
be honoured approximately (`json_schema` → `json_object`). Setting `BestEffort`
on the request demotes the first tier to the third for a single call.

**Reports what a call cost.** Usage is captured with its cache and reasoning lanes
broken out, and priced per call with the model that actually ran, in integer
nano-USD. The rate is a function, not a constant: it varies by request size
(long-context surcharges) and by wall-clock time (off-peak windows, evaluated in
the provider's own timezone). Figures are always the vendor's published
pay-as-you-go rate — a comparable equivalent, **not an invoice**, since a call
routed over a subscription host is billed in plan credits nobody can compare
across models. Behind a gateway the cost is what the proxy reports, on
`usage.cost` in the body first and in its response header second, or nothing
at all. A rate we have not verified reports `Unpriced` and warns, because zero
means unknown, not free. What else the proxy said rides on `ChatResponse.Gateway`
/ `StreamResult.Gateway`: the call id it logged under, the deployment that
really ran, the key's cumulative spend (a gauge, never summed) and the cost
header verbatim, so an unpriced call can be matched to the gateway's own log.

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
treat a neutral `User-Agent` as a bot. A provider marked `emulate_opencode` in
`profiles.yaml` gets opencode's client string and its session header pair from
`FromEnv`; `LLMWIRE_EMULATE_OPENCODE=1` (global, default off) does the same for
a provider the file does not mark; a client built with an explicit
`Config.BaseURL` sets `EmulateOpenCode: true` by hand. Headers only, never a
request body. The session id is the client's own — minted at construction,
rotated after a 30-minute idle gap, the way a person's session starts and ends —
and there is deliberately no way to supply one.

**Reports every call, never its text.** `Config.Logger` (slog; `slog.Default()`
when nil, so the line is JSON or text as the process already logs) gets one
line per call: what went on the wire (message and tool counts, cap, sampling,
reasoning), what came back (`finish_reason`, content and reasoning lengths,
tool calls), tokens per lane (input, cache read, cache write, output,
reasoning), `cost_usd` with its provenance, `headers_ms`/`first_data_ms`/
`total_ms`, output tokens per second, warnings. Debug when it worked; Warn
when it failed (with the bound or status that ended it) or when the answer
needs a look: cut by the output cap, empty on a `stop`, no usage reported.
`Client.Stats()` returns the same figures summed per model since `New`, with
unpriced and unreported calls counted apart so a low total is never read as
cheap; it is a `slog.LogValuer`, for a shutdown summary. Prompt, answer and
key never appear.

**Lists what the key may use.** `ListModels(ctx)` is `GET {base}/models`,
one row per id with the gateway's own `max_input_tokens` / `max_output_tokens`
as `*int64` (nil when the endpoint does not carry them) and the raw object for
the rest. Behind LiteLLM the listing is the key's own model set, so it doubles
as the access check, and a gateway that caps a deployment below the vendor's
window says so here and nowhere else. A limit present but unusable is a
`Warning`, not a failed listing.

**Carries the small helpers every consumer was copying.** `Stream.Collect(onDelta)`
drains a stream into its result. `JSONObject(s)` cuts the first brace-balanced
object out of a reply that was asked for JSON and came back fenced or wrapped in
prose. `Registry.LookupEmbedding(id)` is `Lookup` that refuses a chat model, for
an embeddings client built from a constant. `Redact` and `Truncate` are exported
for the same reason.

## Scope

One wire protocol: `/chat/completions` and `/embeddings`. llmwire adapts models
within that protocol; it does not abstract over protocols. Anthropic, Gemini-native
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

A toggle model may also list `effort_values`: MiMo takes `reasoning_effort`
beside its `thinking` switch, so `ReasoningEffort("high")` renders the level and
`ReasoningOff()` renders the toggle.

`enabled_by_default` and `can_be_disabled` are deliberately separate flags. "Can
this model turn reasoning off" is not answerable from whether `none` appears in a
list of effort levels — inferring it is how published model catalogues get it
wrong.

### Behind a gateway

A gateway is a deployment, not a model: its URL, its key and the names it serves
models under belong to whoever runs it, so they are environment, never
`profiles.yaml`. Three variables, read by `FromEnv`:

```
LLMWIRE_LITELLM_BASE_URL=https://ai-gateway.example/v1
LLMWIRE_LITELLM_API_KEY=<the gateway's key>
LLMWIRE_LITELLM_MODELS=gpt-5.4-mini=ai-gateway-gpt-5.4-mini,text-embedding-3-small
```

The URL passes the same checks a shipped host does: `https`, a bare root
with no query string or credentials, no route suffix. Plain `http://` is
accepted for `localhost`, `127.0.0.1` and `::1` only, where the key never
crosses a wire; a gateway on any other host needs TLS in front of it.

`LLMWIRE_LITELLM_MODELS` lists which profiles the gateway serves, as
`<profile id>[=<name on the gateway>]`; a bare id means the gateway takes the
public name. An application keeps asking for `gpt-5.4-mini`; a listed model is
built as a route to the gateway under its alias, an unlisted one reaches its
vendor as before. The left-hand side must be a profile id, because that is the
model the route inherits from: every capability, every parameter rule.

A route carries no `cost` block and inherits none: the proxy reports real
per-call spend (on `usage.cost`, or in a response header for a non-streamed
call), and it knows which deployment ran and at what discount, where this table
does not. Nothing reported, no price — never a list rate standing in for one.

Because the route names its model, validation still works behind a proxy whose
wire id is opaque: a `Temperature` of 0.3 is rejected locally, since
gpt-5.4-mini rejects it. The same shape can be written into a profile document by hand
(`base` + `gateway` + `provider` + `wire_model_id`); the variable builds it in
memory.

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
