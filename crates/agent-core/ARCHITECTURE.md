# agent-core Architecture

`agent_core` (package `beyond-ai-agent-core`) takes a [`Session`](src/session.rs) (message history)
and an [`Agent`](src/agent.rs) config (model, tools, transport) and drives the model-turn / tool-call
loop to completion, mutating the session in place and emitting an [`AgentEvent`](src/agent.rs) per
streamed token, tool boundary, and turn boundary. It contains no HTTP server, no provider SDK, and no
executor — its only network dependency is a `ModelTransport` trait it never implements itself in
production except via the included `GatewayClient`, an HTTP client that speaks OpenAI/Anthropic wire to
a Beyond gateway base URL.

## Capabilities at a glance

The loop has grown well past a single-shot tool-caller; these are the load-bearing seams (most have a
builder on `Agent` or `ModelRequest`, and each is exercised by unit tests):

- **Cancellation** — `run_events_cancellable(.., cancel)` races the model stream and the tool-dispatch
  phase against a `CancellationToken`; a trip returns `Error::Cancelled`, drops the stream (reqwest
  aborts the request), and drops in-flight tool futures (a `bash` subprocess dies via `kill_on_drop`).
- **Steering** — `run_events_steered(.., steering)` drains a `Steering` handle's *two* lanes:
  *follow-ups* (`push`) injected as fresh user turns at each would-stop boundary, and *steers*
  (`push_steer`) folded onto the in-flight tool-results turn mid-run — so a client can either queue the
  next task or redirect a busy agent between tool turns without waiting for it to stop.
- **Hooks** — `AgentHooks` (`with_hooks`) gates (`before_tool_call` → block reason) and rewrites
  (`after_tool_call`) tool calls; the permission/redaction seam. Defaults to `NoHooks`.
- **Compaction** — when the live prompt crosses `context_window − reserve` (or the provider rejects an
  overflow), [`compaction`](src/compaction.rs) summarizes the prefix via one model call and splices a
  summary in, keeping recent turns verbatim (`Agent::compact`, auto-trigger, compact-and-retry). A
  *second* compaction is incremental: a prior summary (tagged `SUMMARY_MARKER`) is fed forward and
  updated rather than re-summarized, so early context isn't lost (and re-billed) each cycle.
- **Thinking** — `ContentBlock::Thinking`/`RedactedThinking` + `ThinkingDelta`/`SignatureDelta` stream
  events; signatures replay verbatim (Anthropic requires it with tools). `with_thinking(budget)`; the
  thinking *shape* (Anthropic enabled-budget vs adaptive) is chosen per model from the capability
  table, and `with_reasoning_effort` drives OpenAI reasoning models / Anthropic adaptive thinking.
- **Model capabilities** — [`models::capabilities`](src/models.rs) maps a model id (by prefix) to a
  minimal `ModelCaps` table (context window, max output, `max_tokens` vs `max_completion_tokens` field,
  long-cache support, vision, thinking shape, reasoning-effort). The dialects and `Agent::new` consult
  it, so adding a model rarely needs new request-shape plumbing.
- **Tool output & multimodal** — `Tool::run` returns `ToolOutput { text, images, terminate }`: a tool
  can attach images the multimodal model sees (a screenshot, `read` on an image), and `terminate` ends
  the run when *every* call in the batch agrees. `ContentBlock::ToolResult` and both dialects carry the
  images through to the wire.
- **Streaming tool progress** — a tool may override `Tool::run_streaming(input, &ToolProgress)` to
  `emit` incremental output while it runs (pi's `tool_execution_update`); the dispatch forwards each
  chunk as `AgentEvent::ToolProgress` over a `futures::mpsc` channel, ahead of the tool's `ToolEnd`.
  Default `run_streaming` delegates to `run`, so non-streaming tools are untouched.
- **Tool choice** — `ModelRequest::tool_choice` (`Auto`/`None`/`Required`/`Tool(name)`) maps to each
  dialect's vocabulary; unset emits nothing (provider default), so the common request shape is intact.
- **Transport resilience** — `GatewayClient` retries transient failures (429/5xx/connection, honoring
  `Retry-After`) up to the first byte; a mid-stream `event: error` or truncated stream surfaces as
  `Error::Transport` (the SSE decoder's `finish` returns `Result`).
- **Cache observability** — `StreamEvent::Usage` carries `TokenUsage` (input/output + cache-read/write
  + reasoning); both decoders populate it, and `Session` folds the cumulative totals + `last_input_tokens`.
- **Lifecycle events** — `AgentEvent` adds `AgentStart`/`TurnStart`/`Steered`/`AgentEnd`/`Compacted`/`Error`.

## What this crate is not

It ships **zero concrete `Tool` implementations** — `Read`/`Write`/`Edit`/`Bash`/`fork`/`sync`/`logs`
all live in `crates/agent`. It has **no dependency on the gateway crate** — `GatewayClient`'s entire
contract is "POST dialect JSON to a base URL, get SSE back"; routing, provider auth, and metering are
the gateway's job. This split is what lets the loop, the dialect adapters, and the tool dispatch logic
run as pure unit tests with `MockTransport` — no network, no live model, no gateway binary.

## Data Flow

### The turn loop (`Agent::run_events`)

```
Session.messages (Arc, shared) ──► ModelRequest ──► ModelTransport::stream() ──► EventStream<StreamEvent>
       ▲                                                                                  │
       │                                                                     Accumulator::apply (fold)
       │                                                                                  │
       │                                                                  Turn{blocks, stop_reason, usage}
       │                                                                                  │
       │                                                       session.push(Message::assistant(blocks))
       │                                                              session.steps += 1; sink(TurnEnd)
       │                                                                                  │
       │                                              stop_reason != ToolUse, or no tool_use blocks?
       │                                                              │                              │
       │                                                             Yes                             No
       │                                                              │                              │
       │                                                          return Ok(())          sink(ToolStart) × N (call order)
       │                                                                                              │
       │                                            group by write_target → buffer_unordered(≤8 groups)  ◄── bounded; same-path serial
       │                                                                                              │
       │                                                              sink(ToolEnd) × N (call order, post-join)
       │                                                                                              │
       └──────────────────────────────── session.push(Message::tool_results([ToolResult; N])) ◄──── one `user` turn
                                                          (loop back to ModelRequest)

Error exits (no session mutation for the failed turn):
  cancel tripped              ──► Err(Cancelled)                [user abort; mid-stream or mid-tool]
  session.steps >= max_steps  ──► Err(MaxSteps)                [checked before the request is built]
  stream() / stream item Err  ──► Err(Transport(..))            [network/non-2xx (after retries), bad UTF-8/SSE,
                                                                 mid-stream `event: error`, truncated stream]

Recoverable (no error): a streamed tool call whose JSON args never parse keeps its `tool_use` block
(with an empty `{}` input) and is fed back as an error `tool_result` the model can correct — the run
continues rather than aborting.
```

### Bytes to `StreamEvent` (`GatewayClient` + `dialect`)

```
TCP chunks (Bytes) ──► Vec<u8> byte buffer ──split on '\n'──► whole UTF-8 line ──► push_sse_line()
                                                                                          │
                                                                     strip "data:" prefix; skip blank/
                                                                     comment/`event:`/`[DONE]` lines
                                                                                          │
                                                                    serde_json::from_str(payload) → Value
                                                                                          │
                                                                    Dialect::Decoder::push(&Value)
                                                                                          │
                                                                          0..N StreamEvent
```

Anthropic emits an explicit `content_block_stop` per block; OpenAI doesn't, so its decoder synthesizes
`ContentBlockStop` when a tool call's `id` arrives (closing the prior block) or `finish_reason` shows
up, and defers `MessageStop` to `Decoder::finish()` so it lands after the trailing usage-only chunk.
Both decoders produce the identical `StreamEvent` sequence shape — the loop's `Accumulator` is
dialect-blind.

## Concepts & Terminology

| Term              | What It Controls                                                                 | NOT                                                          |
| ----------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `Dialect`         | Which wire shape (`/v1/messages` vs `/v1/chat/completions`) a model id maps to    | Not which *provider* serves the request — that's gateway routing on the virtual key |
| `ModelTransport`  | The loop's only network seam; turns a `ModelRequest` into an `EventStream`        | Not the gateway client specifically — `MockTransport` is the other implementor |
| `Session`         | One run's message history + step/token counters, Arc-shared, serde round-trips   | Not multi-session storage — one `Session` is one conversation |
| `ToolRegistry`    | Name → `Arc<dyn Tool>` lookup the loop dispatches against                          | Not a permission system itself — gating is the `AgentHooks::before_tool_call` seam |
| `AgentHooks`      | Interception around each tool call: `before_tool_call` (block) / `after_tool_call` (rewrite) | Not a sandbox — it decides per call; defaults to `NoHooks` |
| `ToolError`       | A tool's own failure → an error `tool_result` fed back to the model                | Not a loop-aborting error — the run continues             |
| `ToolOutput`      | A tool's success value: `text` + `images` (multimodal) + a `terminate` hint        | Not just a string — `String`/`&str` convert in, and `terminate` ends the run only when every call in the batch agrees |
| `ModelCaps`       | Per-model wire knobs from `capabilities(model)`: max-tokens field, long-cache, vision, thinking shape, reasoning-effort, context window | Not a model catalog or pricing/routing table — the gateway routes and meters; this is the smallest table the wire decisions need |
| `ToolChoice` / `ReasoningEffort` | How the model may use tools this turn / its effort level — optional `ModelRequest` fields mapped per dialect | Unset emits nothing on the wire (provider default), so the default request shape is unchanged |
| `Error`           | A loop/transport failure → `run`/`run_events` returns `Err`, the in-flight turn is discarded | `Cancelled` is a user abort, not a fault; malformed tool args are recoverable, not an error |
| `StreamEvent`      | The normalized unit both dialect decoders emit; what `Accumulator` folds (text/thinking/tool/usage) | Not the wire format — it's the post-translation internal shape |
| `ContentBlock`     | One piece of a `Message` (`Text`/`Thinking`/`RedactedThinking`/`ToolUse`/`ToolResult`/`Image`) | Not a streaming unit — it's the assembled, turn-final form  |
| `AgentEvent`       | The full observation surface (`AgentStart`/`TurnStart`/`Stream`/`ToolStart`/`ToolEnd`/`TurnEnd`/`Steered`/`Compacted`/`AgentEnd`/`Error`) | Not exposed by `Agent::run` — that filters to `Stream` only |
| `max_steps`        | Loop-iteration ceiling; one step = one model turn (tool dispatch doesn't increment it again) | Not a token or wall-clock budget                |

## Core Mechanism

### Accumulating a turn (`agent.rs::Accumulator`)

`Accumulator` folds a `StreamEvent` sequence into `Vec<ContentBlock>` + stop reason + token counts:
text deltas accrue into a `String` buffer; a `ToolUseStart` flushes any open text run and opens a
`(id, name, json-buffer)` tuple; `InputJsonDelta` fragments append to that buffer; `ContentBlockStop`
finalizes whichever is open (parsing the buffered JSON, or `{}` if it was empty). A thinking block
accrues `(text, signature)` from `ThinkingDelta`/`SignatureDelta` and flushes to a `Thinking` block.
If the buffered tool arguments never parse as JSON, the tool call is recorded in `Turn::malformed`
and its `ToolUse` block keeps a wire-valid empty `{}` input; the loop then feeds an error
`tool_result` ("arguments were not valid JSON") back to the model so it can correct, rather than
aborting the run.

Edge case: `stop_reason` defaults to `StopReason::EndTurn` and usage defaults to `0`/`0` if the stream
never delivers a `MessageStop`/`Usage` event before closing cleanly (e.g. a non-conformant upstream) —
the turn completes as if the model ended normally, with no token accounting, rather than erroring.

### Concurrent tool dispatch

Once an assistant turn's `tool_uses()` are collected, the calls are dispatched concurrently — but with
two guards, so concurrency never costs correctness:

- **Same-path calls are serialized.** Each call is grouped by `Tool::write_target(input)` (the path it
  would mutate, or a unique `solo:<i>` key for read-only/path-less calls). Calls sharing a target run
  *sequentially in call order* within their group, because two tools that read-modify-write the same
  file (the model batching two `edit`s on one source) would otherwise race and drop or interleave a
  write. Distinct groups still run concurrently against each other.
- **Concurrency is bounded.** Groups are run through `futures::stream::buffer_unordered` with a cap of
  `MAX_CONCURRENT_TOOL_GROUPS` (8), so a turn requesting dozens of `bash`/`grep` calls can't spawn that
  many subprocesses / parallel walks at once (`grep` itself fans out over CPU cores, which would
  compound). `buffer_unordered` is safe despite its name: each group yields results tagged with their
  original call index `i`, scattered into a pre-sized `results[i]` — cross-group completion order never
  reaches the transcript.

The transcript stays deterministic regardless of which tool finishes first: every `ToolStart` is sunk
*before* dispatch (in call order), and every `ToolEnd` + `ToolResult` block is rebuilt in call order
*after* the groups resolve — so the wall-clock savings never leak into transcript ordering.

All of a turn's results are folded into **one** `Message::tool_results([...])` user turn, not one
message per result (`message.rs:85-94`). Anthropic carries a turn's tool results as multiple blocks on
a single `user` turn and rejects two consecutive same-role messages, so N separate `user` messages
would 400 the next request whenever the model batched more than one tool call.

Each call resolves to `(text, images, is_error, terminate)`: a tool's `ToolOutput` images ride onto
its `ContentBlock::ToolResult` so the multimodal model sees them, and if *every* call in the batch set
`terminate` the loop ends the run after recording the results — an `attempt_completion`/`exit`-style
tool, gated so one tool can't cut off the others dispatched alongside it. Any **steer** messages a
client queued mid-run (`Steering::push_steer`) are folded onto this same tool-results user turn as
trailing text blocks, letting a client redirect a busy agent between tool turns while keeping role
alternation valid; **follow-ups** (`push`) are a separate lane, injected only at the stop boundary.

### Session history sharing

`Session.messages` is `Arc<Vec<Message>>`. `Session::push` mutates via `Arc::make_mut` — in place when
the session solely owns the `Arc` (the steady state between turns), cloning only if a still-live
`ModelRequest` snapshot holds the same pointer. `ModelRequest::messages` and `Agent::tool_defs` are
both `Arc`-shared for the same reason: building a request clones a pointer, not a deep copy of a
history that grows every step (an O(n²) cost over a long run otherwise) or a tool-definition list with
embedded JSON Schemas. `tool_defs` specifically is computed once in `Agent::with_tools`, not rebuilt
per turn. See `agent.rs:tests::request_snapshots_are_isolated_across_turns` for the isolation guarantee
this depends on: an in-flight request's message snapshot must not retroactively see a later turn's
appends.

### SSE byte framing

`GatewayClient::stream` buffers raw `Vec<u8>`, not a per-chunk lossy-decoded `String`. A TCP/HTTP chunk
boundary can land inside a multi-byte UTF-8 character; `from_utf8_lossy` per chunk would replace each
half with `U+FFFD`, silently corrupting non-ASCII tool arguments or prose. Since `\n` (0x0A) never
appears inside a UTF-8 multi-byte sequence, every newline-terminated line is guaranteed whole UTF-8 —
only the unterminated tail is buffered across chunks (`client.rs:83-113`). Verified against a real
socket that splits a write inside a 4-byte emoji (`tests/client_socket.rs`).

## State Machine

### Loop-level (`Agent::run_events`)

```
        ┌────────────────────────────────────────────────────────────────────┐
        │                                                                    │
        ▼                                                                    │
  [check steps]──steps≥max_steps──► Err(MaxSteps)                            │
        │ steps<max_steps                                                    │
        ▼                                                                    │
  [request built] ──stream()/stream item Err──► Err(Transport) (after retries) │
        │ stream exhausts cleanly                                            │
        ▼                                                                    │
  [turn assembled] (malformed tool args → recoverable error result, not fatal) │
        │                                                                    │
        ▼                                                                    │
  [pushed to session, steps+=1, TurnEnd sunk]                                │
        │                                                                    │
        ├──stop_reason≠ToolUse OR no tool_use blocks──► Ok(()) [done]        │
        │                                                                    │
        └──ToolUse + calls present──► [dispatch tools: grouped, bounded] ────┘
              ToolStart×N → group by write_target → buffer_unordered(≤8) → ToolEnd×N + tool_results pushed
```

| From             | Event                              | To                     | Guard                  | What Actually Happens                                              |
| ---------------- | ----------------------------------- | ---------------------- | ----------------------- | -------------------------------------------------------------------- |
| (loop top)       | iteration begins                    | Err(MaxSteps)          | `steps >= max_steps`    | No request sent; session unchanged                                  |
| (loop top)       | iteration begins                    | request built          | `steps < max_steps`     | `ModelRequest` cloned (Arc pointers) from session + cached tool defs |
| request built    | `transport.stream()` / stream item  | Err(Transport)         | network/HTTP/decode err (after retries), mid-stream `event: error`, truncated stream | Turn discarded; an `Error` event is sunk; error returned from `run`/`run_events` |
| request built    | stream exhausts                     | turn assembled         | always                  | `Accumulator::finish()` returns `Turn`; malformed tool args become recoverable error `tool_result`s |
| any await point  | `cancel` tripped                    | Err(Cancelled)         | client abort            | Stream/tool futures dropped (HTTP + subprocess killed); no `Error` event (not a fault) |
| turn assembled   | —                                    | turn pushed            | —                        | `session.push(assistant)`, `record_usage`, `steps += 1`, `TurnEnd` sunk |
| turn pushed      | no tool_use blocks / `stop_reason != ToolUse` | done (`Ok`)    | —                        | Returns to caller; session ends on the assistant turn                |
| turn pushed      | `tool_use` blocks present, `ToolUse` | dispatching tools      | —                        | `ToolStart` sunk per call, in call order                            |
| dispatching tools | all groups resolve (`buffer_unordered`) | (loop top, next iter)  | —                        | `ToolEnd` sunk + one `tool_results` user message pushed, in call order (carrying any tool images + mid-run steer text); ends the run early instead if every call set `terminate` |

### Per-block accumulation (`Accumulator`)

| From        | Event                | To          | What Actually Happens                                  |
| ----------- | --------------------- | ----------- | --------------------------------------------------------- |
| none open   | `TextDelta`           | text open   | Appends to the text buffer                               |
| text open   | `TextDelta`           | text open   | Appends                                                    |
| text open   | `ToolUseStart`        | tool open   | Flushes the text buffer as a `ContentBlock::Text`         |
| none open   | `ToolUseStart`        | tool open   | Opens `(id, name, "")`                                    |
| tool open   | `InputJsonDelta`      | tool open   | Appends to the JSON argument buffer                        |
| text open   | `ContentBlockStop`    | none open   | Flushes text block                                          |
| tool open   | `ContentBlockStop`    | none open   | Parses the JSON buffer; on parse failure, records the call in `Turn::malformed` and pushes `ToolUse` with a wire-valid empty `{}` input (the loop then feeds back a recoverable error result) |
| thinking open | `ContentBlockStop`  | none open   | Flushes a `Thinking { text, signature }` block (from `ThinkingDelta`/`SignatureDelta`) |

## Why It Behaves This Way

### Why tool calls run concurrently but the transcript stays ordered

Tools are I/O-bound (file reads, shell commands, the `beyond` CLI) and a model routinely batches
independent calls in one turn. Running them serially makes the tool phase the sum of their latencies;
overlapping them (up to `MAX_CONCURRENT_TOOL_GROUPS`) collapses it toward the slowest member.
Determinism is preserved by separating *when results become available* (whichever order, via
`buffer_unordered`) from *when they're observed* (each group's results carry their original call index
`i` and are scattered into a pre-sized `results` vec, then the transcript is rebuilt in `calls` order)
— see `agent.rs:tests::independent_tool_calls_run_concurrently`, which deadlocks under serial dispatch
by design to prove the concurrency, and `same_write_target_calls_run_sequentially`, which proves
same-path calls *don't* overlap.

### Why tool results batch into one user message

Both the internal model and Anthropic's wire carry a turn's tool results as multiple blocks on a
single `user` message; Anthropic additionally rejects two consecutive same-role messages. A turn that
calls N>1 tools, fed back as N separate `user` messages, would 400 on the very next request. Folding
them into one message isn't a style choice — it's required by the wire contract on the dialect this
crate's vocabulary is modeled on.

### Why session history and tool defs are `Arc`-shared with copy-on-write

A naive `Vec<Message>` cloned into every `ModelRequest` is an O(n²) cost over a long-running session
(each of n turns deep-copies a history of average size n/2). Sharing via `Arc` makes building a request
a pointer clone; `Arc::make_mut` on `push` keeps appends in-place in the common case (no live request
still holds the old snapshot) and falls back to a real clone only when one does — which is also what
*must* happen for request-snapshot isolation: a request built from turn 3's history must not silently
see turn 4's appended tool results.

### Why the SSE client buffers bytes instead of decoding each chunk independently

A chunk boundary from the underlying TCP stream can fall inside a multi-byte UTF-8 character. Decoding
each chunk independently with `from_utf8_lossy` would replace the split character's bytes with
`U+FFFD` on both sides — a silent, undetectable corruption of tool arguments or assistant prose. Lines
are the unit of decoding instead of chunks because `\n` is guaranteed never to occur inside a UTF-8
multi-byte sequence, so a whole line is always whole UTF-8.

### Why the read timeout is 600s and only an idle timeout

This client sits downstream of the Beyond gateway, which applies its own 600s idle read timeout to the
*provider* connection — the gateway can legitimately hold this connection open with no bytes for up to
600s while waiting on a slow provider (e.g. an extended-thinking gap). A downstream hop's patience must
be at least its upstream's, so `READ_TIMEOUT` mirrors the gateway's `read_timeout_secs` exactly, and is
applied between reads (not as a `Client::timeout` over the whole response) so a long-but-healthy stream
is never killed mid-flight.

### Why the Anthropic body stamps three prompt-cache breakpoints

An agent loop re-sends an ever-growing prefix every turn — tools, then system, then the entire prior
conversation — in request order. Without prompt caching each turn re-bills that whole prefix at full
input-token price: an O(n²) token cost over a `max_steps`-deep run, on the very history this crate
already keeps O(1) *in memory* via `Arc`/COW (the wire cost was the half left unoptimized).
`anthropic::build_body` marks `cache_control` (ephemeral; `ttl: "1h"` only when `cache_long` *and*
`capabilities(model).supports_long_cache` — an unsupported model silently falls back to the 5-minute
TTL instead of 400-ing) at three points (Anthropic caches the request prefix up to each mark; cache
reads cost ~10% of input tokens):

- an **anchor** on the last tool definition — the JSON schemas are byte-identical every turn and sit
  first in the cache order, so this entry stays warm independently of the rolling one;
- a **system** breakpoint on the (array-wrapped) system prompt — a stable anchor that survives
  Anthropic's ~20-block breakpoint lookback on tool-heavy turns, when the rolling mark falls outside it;
- a **rolling** breakpoint on the last message block — caches `tools + system + conversation-so-far`, so
  next turn the whole accumulated transcript is a cache read instead of a re-bill.

Cache hits require a byte-identical prefix, so `ToolRegistry::definitions()` sorts by name (`tool.rs`):
`HashMap` iteration order would otherwise cold-miss the anchor after every process restart / `serve`
reattach. The decoder reads `cache_read_input_tokens`/`cache_creation_input_tokens` back into
`TokenUsage`, so hits are observable. The OpenAI dialect needs no breakpoints — its provider caches
prefixes automatically — but sets `prompt_cache_key` (from `cache_key`) for cache-node affinity.

### Why malformed streamed tool arguments are recoverable, not fatal

A streamed tool call whose JSON args never parse is a protocol glitch, not a model mistake the model
can't recover from. Rather than abort the run, the loop keeps the `tool_use` block (with a wire-valid
empty `{}` input so the next request doesn't 400) and feeds back an error `tool_result` naming the
problem, which the model corrects on the next turn — the same shape as any other recoverable tool error.

### Why dialect selection is a model-id prefix check, not configuration

The gateway relays bytes verbatim; it doesn't translate dialects. Per the "Beyond twist," provider
*selection* (which upstream serves a request) is the gateway's job via the virtual key, but wire
*shape* (`/v1/messages` vs `/v1/chat/completions`) is this crate's job, because the gateway has no
opinion on it. `Dialect::for_model` (`claude*`/`*anthropic*` → Anthropic, else OpenAI) is the entire
selection mechanism — adding a model never requires touching the gateway.

## Trust Boundaries

**What the system verifies (rejects if invalid):**

- HTTP status: a non-2xx response from the gateway is read as text and surfaced as
  `Error::Transport`, never silently treated as a successful empty stream.
- UTF-8 validity of each framed SSE line (`Error::Transport("invalid UTF-8 in SSE stream")`).
- JSON well-formedness of each SSE `data:` payload (`Error::Transport("malformed SSE json")`).
- JSON well-formedness of a streamed tool call's assembled arguments — a parse failure is *not* fatal;
  it becomes a recoverable error `tool_result` (`Turn::malformed`).
- in-band provider errors (`event: error`) and streams truncated before their terminal marker
  (`Error::Transport`).

**What passes through unchecked:**

- The model-supplied tool `name`/`input` pair: the loop looks the name up in the `ToolRegistry` and,
  if present, hands `input` to that tool's `run()` with **no schema validation against the advertised
  `input_schema`** at the loop layer — each `Tool` impl is solely responsible for validating its own
  arguments (see `EchoTool` checking `input.get("text")` itself).
- The remote gateway's identity beyond TLS trust-store validation via `rustls` defaults — there's no
  certificate pinning, no response signing, nothing that distinguishes "the real gateway" from "anything
  that answers on `base_url` with a 2xx and parseable SSE."
- `ToolError`'s `Display` text (including a wrapped `std::io::Error`'s message, which can contain
  filesystem paths) is fed back to the model verbatim as `ToolResult.content` — no redaction.
- The `api_key` is sent as a bearer token over whatever scheme `base_url` specifies; the client itself
  doesn't require or enforce `https://`.

**Why these boundaries are where they are:**

- Provider auth, key-swap, and routing are explicitly the gateway's job (the "Beyond twist" in the
  crate's design) — this crate's contract ends at "valid HTTP wire to a base URL," so it has no basis
  to verify gateway identity beyond what `rustls`'s trust store already does.
- Tool input validation is pushed to each `Tool` because the registry is generic over an open-ended,
  pluggable set of capabilities (`crates/agent` registers 10; nothing stops a caller registering more)
  — there's no single schema the loop could enforce centrally beyond "is this valid JSON," which the
  `Accumulator` already does.

## Package Structure

| File                          | What It Does                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `lib.rs`                       | Crate root: module list, public re-exports, and the crate-wide `#![cfg_attr(test, allow(...))]` panic-free gate |
| `agent.rs`                     | `Agent` config + `run`/`run_events`/`run_events_cancellable`/`run_events_steered` loop, `Accumulator`, concurrent tool dispatch (threading text/images/`terminate`), tool-driven termination, mid-run + stop-boundary steering, `with_reasoning_effort`, model-aware `new` defaults, hooks, auto-compaction + overflow retry |
| `message.rs`                   | `Role`/`ContentBlock`(+ `Thinking`/`RedactedThinking`/`Image`; `ToolResult` carries optional `images`)/`Message`/`ToolDef`/`StopReason`(+ `Refusal`)/`StreamEvent`/`TokenUsage`(+ `reasoning_tokens`) — the internal model |
| `compaction.rs`                | Context compaction: trigger, cut-point search, summary-prompt build, file-op extraction, and incremental update (`previous_summary`/`SUMMARY_MARKER` fold-forward) — the network-free half of `Agent::compact` |
| `models.rs`                    | `capabilities(model) -> ModelCaps`: minimal per-model wire table (max-tokens field, long-cache, vision, thinking shape, reasoning-effort, context window), matched by id prefix; consumed by the dialects and `Agent::new` |
| `hooks.rs`                     | `AgentHooks` interception trait (`before_tool_call`/`after_tool_call`) + `NoHooks` default                       |
| `steering.rs`                  | `Steering` — two shared queues: `push_steer` (mid-run, folded onto the tool-results turn) and `push`/follow-up (injected at would-stop boundaries) |
| `tool.rs`                      | `Tool` trait (`run -> ToolOutput`, optional streaming `run_streaming` + `ToolProgress` sink) + `ToolOutput { text, images, terminate }` + `ToolRegistry` (name-keyed `Arc<dyn Tool>` map, last-registration-wins, name-sorted `definitions`) |
| `transport.rs`                 | `ModelRequest` (system/tools/thinking/`reasoning_effort`/`tool_choice`/cache_key/cache_long), `ReasoningEffort` + `ToolChoice` enums, `ModelTransport` trait, `EventStream` alias |
| `client.rs`                    | `GatewayClient`: production `ModelTransport`; retry-with-backoff (`Retry-After` delta-seconds *or* HTTP-date), chunked-SSE byte framing into whole UTF-8 lines |
| `dialect/mod.rs`                | `Dialect` enum (model-id → wire selection), `StreamDecoder` trait, SSE line-splitting (`push_sse_line`/`decode_sse`) |
| `dialect/anthropic.rs`         | `/v1/messages` body builder (three prompt-cache breakpoints, capability-gated 1h TTL, per-model thinking shape, `tool_choice`, tool-result image rewrite) + decoder (text/thinking/tool/cache-usage, reasoning-token breakout, `pause_turn`/refusal-explanation, in-band error + truncation detection) |
| `dialect/openai.rs`            | `/v1/chat/completions` body builder + decoder — real translation: flattened messages, string-encoded tool args, `image_url` data-URIs (user + fanned-out tool-result images), `max_completion_tokens` vs `max_tokens`, `reasoning_effort`, `tool_choice`, synthesized block-stop events |
| `session.rs`                   | `Session`: Arc-shared copy-on-write message history + step/token counters, serde round-trippable                |
| `error.rs`                     | `Error` (loop/transport, aborts the run) and `ToolError` (tool failure, becomes an error `tool_result`)         |
| `mock.rs`                      | `MockTransport` + `turn::{text, tool_call}` builders — scripted, no-network loop testing                        |
| `tests/client_socket.rs`       | `GatewayClient` over a real TCP socket: SSE decode, UTF-8 chunk-split reassembly, HTTP-error surfacing            |

## Configuration

| Setting                          | Default | What It Controls                                                                                     |
| ---------------------------------- | ------- | --------------------------------------------------------------------------------------------------------- |
| `DEFAULT_MAX_TOKENS` / `Agent::with_max_tokens` | model-aware (≥ 4096) | Per-turn output ceiling (`max_tokens`/`max_completion_tokens` per dialect). `Agent::new` seeds it from `capabilities(model).max_output` (floored at 4096); the compaction `context_window` is likewise seeded from `capabilities`. Both still overridable via the builders |
| `DEFAULT_MAX_STEPS` / `Agent::with_max_steps`   | 24      | Loop-iteration ceiling; once `session.steps` reaches it, the *next* iteration returns `Error::MaxSteps` before sending a request — a runaway-tool-call backstop |
| `Agent::with_system`              | `None`  | System prompt; hoisted to each dialect's native system field (Anthropic top-level `system`, OpenAI leading `system` message) |
| `Agent::with_tools`               | empty   | The tool set advertised to the model; definitions + JSON Schemas computed once here, shared via `Arc<[ToolDef]>` for the agent's lifetime |
| `CONNECT_TIMEOUT` (`client.rs`)   | 10s     | TCP+TLS handshake cap to the gateway; mirrors the gateway's own upstream `connect_timeout_secs`        |
| `READ_TIMEOUT` (`client.rs`)      | 600s    | Idle timeout *between* reads on the streaming body (not total stream duration); sized to the gateway's own upstream `read_timeout_secs` |

## Failure Modes

| Failure                                                  | What Actually Happens                                                                                              | Recovery                                                                 |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `session.steps` reaches `max_steps`                       | `run_events` returns `Error::MaxSteps(n)` before another request is sent; session retains every completed turn        | Caller persists/inspects `Session`; can raise `max_steps` and resume     |
| Gateway returns non-2xx                                   | Body read as text; `Error::Transport("gateway returned {status}: {detail}")` returned from `stream()`                  | First `run`/`run_events` call errors; session has no partial turn       |
| SSE chunk splits a multi-byte UTF-8 char                  | Raw bytes buffered across chunks; decoding waits for the newline, so the split is invisible                            | Transparent — no error (regression-tested over a real socket)            |
| SSE-framed line isn't valid UTF-8                          | `Error::Transport("invalid UTF-8 in SSE stream: …")` from inside the event stream                                      | Stream item `Err`; in-progress turn discarded, error returned             |
| SSE `data:` payload isn't valid JSON                        | `Error::Transport("malformed SSE json: …")`                                                                              | Same as above                                                              |
| Streamed tool-call JSON never completes/parses              | The `tool_use` block keeps an empty `{}` input; the loop feeds back an error `tool_result` naming the bad buffer | Run continues; model corrects on the next turn (recoverable, not fatal)   |
| Model calls an unregistered tool name                      | That call's result becomes `("unknown tool: {name}", is_error: true)`                                                   | Not fatal — fed back as an error `tool_result` the model sees next turn   |
| A registered tool's `run()` returns `Err`                  | The error's `Display` text becomes `ToolResult.content` with `is_error: true`                                          | Not fatal — same as above                                                  |
| Stream ends cleanly without a `MessageStop`/`Usage` event   | `stop_reason` defaults to `EndTurn`, token counts default to `0` — the turn looks like a normal completion             | Silent — no error surfaced; usage accounting is simply incomplete for that turn |
| Gateway holds the connection open with no bytes for >600s   | `reqwest`'s idle read timeout fires; the in-flight request errors                                                       | Surfaces as a transport `Error`; no automatic retry in this crate          |
| A `Mutex` in `MockTransport` is poisoned by a panicked test thread | `.unwrap_or_else(|e| e.into_inner())` recovers the data instead of returning an empty/misleading state           | Test-only path — this crate holds no `Mutex` outside `mock.rs`            |
