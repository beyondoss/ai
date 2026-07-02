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
  The race extends to `run_turn`'s mid-stream retry backoff sleep and `run_turn_once`'s initial
  `stream()` call itself (not just the resulting event stream) — cancellation is observed even before
  the first byte, without depending on `async_stream`'s laziness as an implementation detail. Cancelling
  mid-tool-dispatch still leaves the session resumable: any call with no result yet is synthesized as an
  error `tool_result` (`repair_cancelled_dispatch`) rather than leaving an orphaned `tool_use` block with
  no matching result.
- **Steering** — `run_events_steered(.., steering)` drains a `Steering` handle's _two_ lanes:
  _follow-ups_ (`push`) injected as fresh user turns at each would-stop boundary, and _steers_
  (`push_steer`) folded onto the in-flight tool-results turn mid-run — so a client can either queue the
  next task or redirect a busy agent between tool turns without waiting for it to stop. A **refusal**
  (`StopReason::Refusal`) is a distinct terminal condition: the run ends immediately without draining
  either lane, since injecting a new turn right after a refusal would likely just be refused again — the
  queue is left intact for whatever run reads the same `Steering` handle next. How much of a lane one
  drain call consumes is `QueueMode`, one **independent** setting per lane (`Steering::set_steering_mode`/
  `steering_mode` for the steer lane, `set_follow_up_mode`/`follow_up_mode` for the follow-up-plus-
  stranded-steer stop-boundary drain — matching pi's own separate `steeringMode`/`followUpMode`, not one
  shared toggle): `OneAtATime` (the default, matching pi's `PendingMessageQueue`) takes only the oldest
  queued message, leaving the rest queued for the *next* drain point, so several messages queued in
  quick succession land as separate turns; `All` folds everything queued into one injection (this
  crate's original behavior, still available).
  `Steering::pending_count` peeks the combined depth of both lanes without draining either — pi's
  `pendingMessageCount` — for a host surface (e.g. `serve`'s `get_state`) that wants to report queue
  depth to a client.
- **Graceful stop** — `Steering::request_stop` (pi's `shouldStopAfterTurn` equivalent) sets a flag
  checked at every turn boundary, *after* that turn's tool calls (if any) have already run and their
  results are committed to the session, but before the next model call would start. It wins over both
  continuing a tool-call turn and draining queued follow-up/steer messages (mirroring the refusal case,
  those are left queued, not dropped). Unlike `cancel`, it never drops an in-flight future or leaves an
  orphaned `tool_use` — the difference between a graceful stop and a hard abort. Whatever a run does with
  a pending request, `run_events_steered` always clears it before returning (a `Drop` guard, so an early
  `?`/error/cancellation return can't skip it), so a request can never bind to a later, unrelated call
  that reuses the same `Steering` handle.
- **Hooks** — `AgentHooks` (`with_hooks`) gates (`before_tool_call` → block reason) and rewrites
  (`after_tool_call`) tool calls; the permission/redaction seam. Both methods also receive the run's
  `&CancellationToken`, so a hook can observe (or react to) an in-flight cancellation. Defaults to
  `NoHooks`.
- **Compaction** — when the live prompt crosses `context_window − reserve` (or the provider rejects an
  overflow), [`compaction`](src/compaction.rs) summarizes the prefix via one model call and splices a
  summary in, keeping recent turns verbatim (`Agent::compact`, auto-trigger, compact-and-retry).
  `Agent::compact`'s `custom_instructions: Option<&str>` steers what the summary emphasizes — a manual
  (client-triggered) compaction's own focus, matching pi's `compact(customInstructions)`; every
  automatic trigger passes `None`, having no client in the loop to ask. A _second_ compaction is
  incremental: a prior summary (tagged `SUMMARY_MARKER`) is fed forward and updated rather than
  re-summarized, so early context isn't lost (and re-billed) each cycle. A cut landing mid-turn splits
  into a **closed-history** call (gets `custom_instructions`) and a **turn-prefix** call (does not —
  matches pi's own split — run *sequentially*, not concurrently: pi originally ran them via
  `Promise.all`, then fixed exactly this itself so a single-concurrency local provider doesn't reject
  the second of two simultaneous completions; `SPLIT_TURN_PREFIX_SCALE` × the summary token budget for
  the smaller one), merged under a "Turn Context (split turn)" header — closer to pi's two-template
  approach than summarizing the whole prefix with one minimal-context call. `summary_max_tokens` scales
  from the model's own `max_output` rather
  than a flat constant (`agent::scaled_summary_max_tokens`) — seeded by `Agent::new` and *rescaled* by
  `with_compaction` whenever the incoming config still carries the struct's own flat default, so a
  caller replacing the whole config wholesale (e.g. `serve.rs`'s `build_agent`, to override just
  `reserve_tokens`) doesn't silently discard the model-aware value in favor of that default. The trigger
  adds `compaction::trailing_tokens` (messages appended since the last
  usage snapshot, estimated rather than assumed already reflected in `last_input_tokens`) so it doesn't
  compare a stale, undercounted size against the window. Overflow detection
  (`agent.rs::is_context_overflow`) is table-driven (`OVERFLOW_PATTERNS`) with a `THROTTLE_EXCLUSIONS`
  allowlist checked first, so a provider's rate-limit/service-unavailable message (which can happen to
  contain "too many tokens") is never misread as a context overflow worth compacting for. A second,
  non-error-based detector (`compaction::is_hard_overflow`) fires regardless of whether auto-compaction
  is enabled: if the live prompt already meets the *raw* `context_window` (a silent overflow — the
  request that got there still succeeded, no error was ever raised), or a `StopReason::MaxTokens` stop
  lands where the live prompt plus the next turn's full output budget would meet it (the window, not
  `max_tokens`, is plausibly the real constraint), the run compacts anyway — disabling proactive
  compaction is a preference about timing with headroom to spare, not license to keep sending requests
  already guaranteed to overflow. Complementary and proactive, on the Anthropic dialect specifically:
  `dialect/anthropic::clamp_max_tokens_to_context` estimates the live prompt from `req.messages`/
  `req.system` directly and clamps the *request's own* `max_tokens` down to whatever headroom is left
  under `context_window` (minus a margin absorbing estimation slop) — so a long-running session doesn't
  keep sending its static output ceiling on every turn regardless of how much of the window the prompt
  has already consumed, paying for a wasted round-trip each time before the reactive path above ever
  gets a chance to compact. Never clamps below a configured thinking budget, and never raises
  `max_tokens` above what was asked.
- **Thinking** — `ContentBlock::Thinking`/`RedactedThinking` + `ThinkingDelta`/`SignatureDelta` stream
  events; signatures replay verbatim (Anthropic requires it with tools). `with_thinking(budget)`; the
  thinking _shape_ (Anthropic enabled-budget vs adaptive) is chosen per model from the capability
  table, and `with_reasoning_effort` drives OpenAI reasoning models / Anthropic adaptive thinking.
  `models::ThinkingLevel` (`Off`/`Minimal`/`Low`/`Medium`/`High`/`XHigh`) is the portable vocabulary a
  caller (`serve`'s `cycle_thinking_level`/`set_reasoning_effort`) uses instead of a raw budget;
  `models::thinking_for_level(caps, level)` translates a level into the `(thinking_budget,
  reasoning_effort)` pair `with_thinking`/`with_reasoning_effort` need for a *specific* model — setting
  both together for `Adaptive` shape (the budget is a pure on/off gate there; `output_config.effort`
  carries the real depth), only `reasoning_effort` for OpenAI reasoning models, only a scaled budget for
  `Budget` shape. Whatever `ReasoningEffort` reaches a dialect — via `thinking_for_level` or a raw
  `with_reasoning_effort`/`--reasoning-effort` call — is clamped there to what the specific model's wire
  actually accepts (`models::clamp_reasoning_effort`): several OpenAI reasoning models (o-series, bare
  gpt-5, every gpt-5.1 variant) and two Anthropic adaptive ids (sonnet-4-6, sonnet-5) have no `xhigh`
  tier and clamp down to `high`; `gpt-5.5`/`gpt-5.5-pro` reject `minimal`(+`low`) and clamp up. Anthropic
  adaptive additionally has no `minimal` wire tier at all (always sent as `"low"`) and remaps `xhigh` per
  model (`"max"` on `claude-opus-4-6` uniquely; `"xhigh"` elsewhere) via `models::anthropic_adaptive_effort_wire`.
- **Model capabilities** — [`models::capabilities`](src/models.rs) maps a model id (by prefix) to a
  minimal `ModelCaps` table (context window, max output, `max_tokens` vs `max_completion_tokens` field,
  long-cache support, vision, thinking shape, reasoning-effort, explicit-disable capability, eager
  tool-streaming). The dialects and `Agent::new` consult it, so adding a model rarely needs new
  request-shape plumbing.
- **Vision downgrade** — a model whose `ModelCaps::supports_vision` is `false` never sees a raw image
  block: each dialect replaces a user/tool-result image with a text placeholder
  (`"(image omitted: model does not support images)"` / the tool-result variant) instead of sending
  bytes the model can't decode.
- **Eager tool-input streaming** — every Anthropic tool definition is marked
  `eager_input_streaming: true` when `ModelCaps::supports_eager_tool_streaming` is set (true for every
  current Anthropic id); `client.rs` only sends the mutually-exclusive
  `fine-grained-tool-streaming-2025-05-14` beta header for a model that lacks the per-tool marker —
  never both.
- **Tool output & multimodal** — `Tool::run` returns `ToolOutput { text, images, terminate }`: a tool
  can attach images the multimodal model sees (a screenshot, `read` on an image), and `terminate` ends
  the run when _every_ call in the batch agrees. `ContentBlock::ToolResult` and both dialects carry the
  images through to the wire.
- **Streaming tool progress** — a tool may override `Tool::run_streaming(input, &ToolProgress)` to
  `emit` incremental output while it runs (pi's `tool_execution_update`); the dispatch forwards each
  chunk as `AgentEvent::ToolProgress` the instant it arrives (not batched after the group joins), and
  `AgentEvent::ToolEnd` fires per call the moment its own result is known — a client watching the event
  stream sees completions in actual finish order, not batched after the slowest call in the group.
  Default `run_streaming` delegates to `run`, so non-streaming tools are untouched.
- **Cross-run file-mutation exclusivity** — `Agent::with_write_locks` shares a `WriteLockRegistry`
  (`write_lock.rs`) across every `Agent` rebuild for a process's lifetime (not just one turn's grouping):
  a tool's `write_target(input)` path acquires the registry's per-path async lock for the tool's whole
  serial run, so two `Agent`s built back-to-back (a `set_model` rebuild, or two sessions sharing one
  registry) can't race a same-path `edit`/`write` against each other. Layered on top of, not a
  replacement for, the intra-turn write-target grouping described below. Documented limitation: this
  only serializes within one process — cross-process locking would need a filesystem advisory lock,
  out of scope until a real multi-process use case needs it.
- **Explicit reasoning/thinking disable** — `ModelCaps::reasoning_disableable` (per exact model id, not
  per `ThinkingShape`) drives an explicit "off" signal (Anthropic `{"type":"disabled"}`, OpenAI
  `{"effort":"none"}`) on a turn that isn't requesting thinking, for a model capable of one — instead of
  omitting the field and trusting the provider's undocumented default.
- **Cross-model state scrubbing** — `Message::model_id` (stamped on every assistant turn from
  `Agent`'s own `model`) records which model produced it; `Session::scrub_cross_model_state(new_model)`
  downgrades a non-empty signed `Thinking` block to a plain `Text` block (preserving the visible
  reasoning trace as context rather than erasing it — only the block's *replayability as thinking* is
  model-specific, not the prose itself), drops an empty `Thinking` block or any `RedactedThinking`
  block outright (opaque ciphertext, nothing to preserve), and truncates a combined OpenAI-Responses
  tool-call id (`"call_id|item_id"` → `"call_id"`) — all from any message not stamped with
  `new_model`. `model_id: None` (a message from before this field existed) is always treated as
  foreign. `anthropic::build_body`'s own `downgrade_unsigned_thinking` (an unsigned — not
  cross-model — thinking block, e.g. from an aborted stream) follows the same empty-drops-instead-of-
  degrades rule: a block whose `thinking` text is also empty is dropped rather than downgraded to
  `{"type": "text", "text": ""}`, which Anthropic's non-empty-text requirement would just as readily
  reject.
- **Tool choice** — `ModelRequest::tool_choice` (`Auto`/`None`/`Required`/`Tool(name)`) maps to each
  dialect's vocabulary; unset emits nothing (provider default), so the common request shape is intact.
- **Transport resilience** — `GatewayClient` retries transient failures (429/5xx/connection, honoring
  `Retry-After`) up to the first byte; a mid-stream `event: error` or truncated stream surfaces as
  `Error::Transport` (the SSE decoder's `finish` returns `Result`), which `Agent::run_turn` retries with
  backoff (`is_retryable_mid_stream`, capped at `MAX_MID_STREAM_RETRIES`) from a fresh connection and
  a fresh `Accumulator`, rather than resuming a dead attempt's partial blocks. Beyond the decoder's own
  truncation rejection and a tagged network failure (`MID_STREAM_NETWORK_ERROR`),
  `is_retryable_mid_stream` recognizes a table of named in-band provider error *types*
  (`MID_STREAM_RETRYABLE_ERROR_TYPES` — Anthropic's `rate_limit_error`/`api_error`/`timeout_error`,
  OpenAI's `rate_limit_exceeded`/`server_error`/`internal_error`/`service_unavailable`) and explicit
  provider retry-guidance phrases (`MID_STREAM_RETRY_GUIDANCE_PHRASES`), deliberately keyed on names
  rather than raw HTTP status-code substrings ("500" et al.) — a mid-stream failure never carries a
  fresh status code to key on, and a bare digit substring risks matching an unrelated number in the
  message. `Agent::with_auto_retry` (default enabled) disables this specific layer — a normally-retried
  failure surfaces on the very first attempt instead — for debugging a flaky connection without several
  silent attempts first; `GatewayClient`'s own pre-first-byte retry above is unaffected either way.
- **Cache observability** — `StreamEvent::Usage` carries `TokenUsage` (input/output + cache-read/write
  - reasoning); both decoders populate it, and `Session` folds the cumulative totals + `last_input_tokens`.
- **Lifecycle events** — `AgentEvent` adds `AgentStart`/`TurnStart`/`Steered`/`AgentEnd`/`Compacted`/`Error`.

## What this crate is not

It ships **zero concrete `Tool` implementations** — `Read`/`Write`/`Edit`/`Bash`/`fork`/`sync`/`logs`
all live in `crates/agent`. It has **no dependency on the gateway crate** — `GatewayClient`'s entire
contract is "POST dialect JSON to a base URL, get SSE back"; routing, provider auth, and metering are
the gateway's job. This split is what lets the loop, the dialect adapters, and the tool dispatch logic
run as pure unit tests with `MockTransport` — no network, no live model, no gateway binary.

It is **not a plugin host**. pi has a real extension system (`core/extensions/`): dynamically loaded
third-party code that subscribes to typed lifecycle events (including `session_compact`) and can call
back into session actions (`navigateTree`, `fork`, `newSession`, …). This crate's equivalent seams —
`AgentHooks` (tool-call gate/rewrite) and `CheckpointHook` (mid-run persistence points) — are Rust
traits with exactly one implementation chosen by the embedder at compile time, not a runtime-loadable,
subscribable, third-party extension surface. Evaluated and deliberately **not** extended to cover
compaction or branch-navigation as hookable lifecycle events: nothing in this codebase dynamically
loads untrusted code, there is no extension marketplace or third-party-author story to serve, and
adding one would mean taking on loading/sandboxing/versioning machinery with no concrete consumer —
directly against this project's minimum-effective-abstraction bias. If a genuine need for third-party
extensibility ever appears, revisit this as a deliberate, scoped addition rather than bolting hooks on
piecemeal.

Two specific extension points pi's harness has and this crate deliberately doesn't: an ephemeral
per-request context transform (rewrite/prune the outbound message list for one call without persisting
the change — pi's `transformContext`) and a before/after-provider-request seam (patch headers/timeout
per call, mutate the raw wire payload, observe raw response status/headers). Both are real seams in
pi's *extension* system specifically — they exist to let a runtime-loaded, third-party plugin reach
into the request pipeline. Since this crate has no such plugin system (see above) and no first-party
caller has ever needed either seam, adding them now would be exactly the same mistake: machinery with
no concrete consumer. If a real need appears — a caller that wants to inject transient context, or
observe/rewrite the wire below `ModelRequest`/`StreamEvent` — add the narrowest seam that call actually
needs then, not a general-purpose hook ahead of time.

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

| Term                             | What It Controls                                                                                                                          | NOT                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `Dialect`                        | Which wire shape (`/v1/messages`, `/v1/chat/completions`, or `/v1/responses`) a model id maps to, via `ApiKind`                           | Not which _provider_ serves the request — that's gateway routing on the virtual key                                              |
| `ModelTransport`                 | The loop's only network seam; turns a `ModelRequest` into an `EventStream`                                                                | Not the gateway client specifically — `MockTransport` is the other implementor                                                   |
| `Session`                        | One run's message history + step/token counters, Arc-shared, serde round-trips                                                            | Not multi-session storage — one `Session` is one conversation                                                                    |
| `ToolRegistry`                   | Name → `Arc<dyn Tool>` lookup the loop dispatches against                                                                                 | Not a permission system itself — gating is the `AgentHooks::before_tool_call` seam                                               |
| `AgentHooks`                     | Interception around each tool call: `before_tool_call` (block) / `after_tool_call` (rewrite)                                              | Not a sandbox — it decides per call; defaults to `NoHooks`                                                                       |
| `ToolError`                      | A tool's own failure → an error `tool_result` fed back to the model                                                                       | Not a loop-aborting error — the run continues                                                                                    |
| `ToolOutput`                     | A tool's success value: `text` + `images` (multimodal) + a `terminate` hint                                                               | Not just a string — `String`/`&str` convert in, and `terminate` ends the run only when every call in the batch agrees            |
| `ModelCaps`                      | Per-model wire knobs from `capabilities(model)`: max-tokens field, long-cache, vision, thinking shape, reasoning-effort, context window   | Not a model catalog or pricing/routing table — the gateway routes and meters; this is the smallest table the wire decisions need |
| `ToolChoice` / `ReasoningEffort` | How the model may use tools this turn / its effort level — optional `ModelRequest` fields mapped per dialect                              | Unset emits nothing on the wire (provider default), so the default request shape is unchanged                                    |
| `Error`                          | A loop/transport failure → `run`/`run_events` returns `Err`, the in-flight turn is discarded                                              | `Cancelled` is a user abort, not a fault; malformed tool args are recoverable, not an error                                      |
| `StreamEvent`                    | The normalized unit both dialect decoders emit; what `Accumulator` folds (text/thinking/tool/usage)                                       | Not the wire format — it's the post-translation internal shape                                                                   |
| `ContentBlock`                   | One piece of a `Message` (`Text`/`Thinking`/`RedactedThinking`/`ToolUse`/`ToolResult`/`Image`)                                            | Not a streaming unit — it's the assembled, turn-final form                                                                       |
| `AgentEvent`                     | The full observation surface (`AgentStart`/`TurnStart`/`Stream`/`ToolStart`/`ToolEnd`/`TurnEnd`/`Steered`/`Compacted`/`AgentEnd`/`Error`) | Not exposed by `Agent::run` — that filters to `Stream` only                                                                      |
| `max_steps`                      | Loop-iteration ceiling; one step = one model turn (tool dispatch doesn't increment it again)                                              | Not a token or wall-clock budget                                                                                                 |

## Core Mechanism

### Accumulating a turn (`agent.rs::Accumulator`)

`Accumulator` folds a `StreamEvent` sequence into `Vec<ContentBlock>` + stop reason + token counts:
text deltas accrue into a `String` buffer; a `ToolUseStart` flushes any open text run and opens a
`(id, name, json-buffer)` tuple; `InputJsonDelta` fragments append to that buffer; `ContentBlockStop`
finalizes whichever is open (parsing the buffered JSON, or `{}` if it was empty). A thinking block
accrues `(text, signature)` from `ThinkingDelta`/`SignatureDelta` and flushes to a `Thinking` block.
Parsing a non-empty buffer tries three tiers in order before giving up: the raw buffer as-is; then
`repair_json` (fixes mis-escaping — a raw control byte or stray backslash inside a string, not a
structural problem); then `close_incomplete_json` (closes whatever string/`{`/`[` were still open when
the buffer ended — a genuinely *incomplete* stream, e.g. a long `write`/`edit` value cut off by an
output-token ceiling, recovering a partial object instead of discarding it to `{}`). If the buffered
tool arguments still never parse as JSON after all three, the tool call is recorded in
`Turn::malformed` and its `ToolUse` block keeps a wire-valid empty `{}` input; the loop then feeds an
error `tool_result` ("arguments were not valid JSON") back to the model so it can correct, rather than
aborting the run.

Edge case: `stop_reason` defaults to `StopReason::EndTurn` and usage defaults to `0`/`0` if the stream
never delivers a `MessageStop`/`Usage` event before closing cleanly (e.g. a non-conformant upstream) —
the turn completes as if the model ended normally, with no token accounting, rather than erroring.

### Concurrent tool dispatch

Once an assistant turn's `tool_uses()` are collected, the calls are dispatched concurrently — but with
two guards, so concurrency never costs correctness:

- **Same-path calls are serialized.** Each call is grouped by `Tool::write_target(input)` (the path it
  would mutate, or a unique `solo:<i>` key for read-only/path-less calls). Calls sharing a target run
  _sequentially in call order_ within their group, because two tools that read-modify-write the same
  file (the model batching two `edit`s on one source) would otherwise race and drop or interleave a
  write. Distinct groups still run concurrently against each other.
- **Concurrency is bounded.** Groups are run through `futures::stream::buffer_unordered` with a cap of
  `MAX_CONCURRENT_TOOL_GROUPS` (8), so a turn requesting dozens of `bash`/`grep` calls can't spawn that
  many subprocesses / parallel walks at once (`grep` itself fans out over CPU cores, which would
  compound). `buffer_unordered` is safe despite its name: each group yields results tagged with their
  original call index `i`, scattered into a pre-sized `results[i]` — cross-group completion order never
  reaches the transcript.

The transcript stays deterministic regardless of which tool finishes first: every `ToolStart` is sunk
_before_ dispatch (in call order), and every `ToolEnd` + `ToolResult` block is rebuilt in call order
_after_ the groups resolve — so the wall-clock savings never leak into transcript ordering.

All of a turn's results are folded into **one** `Message::tool_results([...])` user turn, not one
message per result (`message.rs:85-94`). Anthropic carries a turn's tool results as multiple blocks on
a single `user` turn and rejects two consecutive same-role messages, so N separate `user` messages
would 400 the next request whenever the model batched more than one tool call.

Each call resolves to `(text, images, is_error, terminate)`: a tool's `ToolOutput` images ride onto
its `ContentBlock::ToolResult` so the multimodal model sees them, and if _every_ call in the batch set
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

`GatewayClient::stream` frames the chunked body through `LineFramer`, which buffers raw *bytes* (a
`BytesMut`), not a per-chunk lossy-decoded `String`. A TCP/HTTP chunk boundary can land inside a
multi-byte UTF-8 character; `from_utf8_lossy` per chunk would replace each half with `U+FFFD`, silently
corrupting non-ASCII tool arguments or prose. Since `\n` (0x0A) never appears inside a UTF-8 multi-byte
sequence, every newline-terminated line is guaranteed whole UTF-8 — only the unterminated tail is
buffered across chunks. The newline is found with SIMD `memchr` and the line handed back via
`BytesMut::split_to` — an O(1) pointer split sharing the backing allocation, so a line costs neither a
per-line heap allocation nor a memmove of the buffer remainder (a `Vec<u8>` framer paid both on every
line — O(lines × remaining bytes) when a burst coalesces many `data:` lines into one read). Verified
against a real socket that splits a write inside a 4-byte emoji (`tests/client_socket.rs`), and benched
in `benches/decode.rs` (framing is ~4× faster on coalesced chunks: ~2 allocations/turn vs ~2,500).
`LineFramer` is `pub` so the framing hot path is benchable in isolation.

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

| From              | Event                                         | To                    | Guard                                                                                | What Actually Happens                                                                                                                                                            |
| ----------------- | --------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (loop top)        | iteration begins                              | Err(MaxSteps)         | `steps >= max_steps`                                                                 | No request sent; session unchanged                                                                                                                                               |
| (loop top)        | iteration begins                              | request built         | `steps < max_steps`                                                                  | `ModelRequest` cloned (Arc pointers) from session + cached tool defs                                                                                                             |
| request built     | `transport.stream()` / stream item            | Err(Transport)        | network/HTTP/decode err (after retries), mid-stream `event: error`, truncated stream | Turn discarded; an `Error` event is sunk; error returned from `run`/`run_events`                                                                                                 |
| request built     | stream exhausts                               | turn assembled        | always                                                                               | `Accumulator::finish()` returns `Turn`; malformed tool args become recoverable error `tool_result`s                                                                              |
| any await point   | `cancel` tripped                              | Err(Cancelled)        | client abort                                                                         | Stream/tool futures dropped (HTTP + subprocess killed); no `Error` event (not a fault)                                                                                           |
| turn assembled    | —                                             | turn pushed           | —                                                                                    | `session.push(assistant)`, `record_usage`, `steps += 1`, `TurnEnd` sunk                                                                                                          |
| turn pushed       | no tool_use blocks / `stop_reason != ToolUse` | done (`Ok`)           | —                                                                                    | Returns to caller; session ends on the assistant turn                                                                                                                            |
| turn pushed       | `tool_use` blocks present, `ToolUse`          | dispatching tools     | —                                                                                    | `ToolStart` sunk per call, in call order                                                                                                                                         |
| dispatching tools | all groups resolve (`buffer_unordered`)       | (loop top, next iter) | —                                                                                    | `ToolEnd` sunk + one `tool_results` user message pushed, in call order (carrying any tool images + mid-run steer text); ends the run early instead if every call set `terminate` |

### Per-block accumulation (`Accumulator`)

| From          | Event              | To        | What Actually Happens                                                                                                                                                                         |
| ------------- | ------------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| none open     | `TextDelta`        | text open | Appends to the text buffer                                                                                                                                                                    |
| text open     | `TextDelta`        | text open | Appends                                                                                                                                                                                       |
| text open     | `ToolUseStart`     | tool open | Flushes the text buffer as a `ContentBlock::Text`                                                                                                                                             |
| none open     | `ToolUseStart`     | tool open | Opens `(id, name, "")`                                                                                                                                                                        |
| tool open     | `InputJsonDelta`   | tool open | Appends to the JSON argument buffer                                                                                                                                                           |
| text open     | `ContentBlockStop` | none open | Flushes text block                                                                                                                                                                            |
| tool open     | `ContentBlockStop` | none open | Parses the JSON buffer (raw, then `repair_json`, then `close_incomplete_json`); on parse failure even after both repairs, records the call in `Turn::malformed` and pushes `ToolUse` with a wire-valid empty `{}` input (the loop then feeds back a recoverable error result) |
| thinking open | `ContentBlockStop` | none open | Flushes a `Thinking { text, signature }` block (from `ThinkingDelta`/`SignatureDelta`)                                                                                                        |

## Why It Behaves This Way

### Why tool calls run concurrently but the transcript stays ordered

Tools are I/O-bound (file reads, shell commands, the `beyond` CLI) and a model routinely batches
independent calls in one turn. Running them serially makes the tool phase the sum of their latencies;
overlapping them (up to `MAX_CONCURRENT_TOOL_GROUPS`) collapses it toward the slowest member.
Determinism is preserved by separating _when results become available_ (whichever order, via
`buffer_unordered`) from _when they're observed_ (each group's results carry their original call index
`i` and are scattered into a pre-sized `results` vec, then the transcript is rebuilt in `calls` order)
— see `agent.rs:tests::independent_tool_calls_run_concurrently`, which deadlocks under serial dispatch
by design to prove the concurrency, and `same_write_target_calls_run_sequentially`, which proves
same-path calls _don't_ overlap.

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
_must_ happen for request-snapshot isolation: a request built from turn 3's history must not silently
see turn 4's appended tool results.

### Why the SSE client buffers bytes instead of decoding each chunk independently

A chunk boundary from the underlying TCP stream can fall inside a multi-byte UTF-8 character. Decoding
each chunk independently with `from_utf8_lossy` would replace the split character's bytes with
`U+FFFD` on both sides — a silent, undetectable corruption of tool arguments or assistant prose. Lines
are the unit of decoding instead of chunks because `\n` is guaranteed never to occur inside a UTF-8
multi-byte sequence, so a whole line is always whole UTF-8.

### Why the read timeout is 600s and only an idle timeout

This client sits downstream of the Beyond gateway, which applies its own 600s idle read timeout to the
_provider_ connection — the gateway can legitimately hold this connection open with no bytes for up to
600s while waiting on a slow provider (e.g. an extended-thinking gap). A downstream hop's patience must
be at least its upstream's, so `READ_TIMEOUT` mirrors the gateway's `read_timeout_secs` exactly, and is
applied between reads (not as a `Client::timeout` over the whole response) so a long-but-healthy stream
is never killed mid-flight.

### Why the Anthropic body stamps three prompt-cache breakpoints

An agent loop re-sends an ever-growing prefix every turn — tools, then system, then the entire prior
conversation — in request order. Without prompt caching each turn re-bills that whole prefix at full
input-token price: an O(n²) token cost over a `max_steps`-deep run, on the very history this crate
already keeps O(1) _in memory_ via `Arc`/COW (the wire cost was the half left unoptimized).
`anthropic::build_body` marks `cache_control` (ephemeral; `ttl: "1h"` only when `cache_long` _and_
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
prefixes automatically — but sets `prompt_cache_key` (from `cache_key`) for cache-node affinity, gated
on `capabilities(model).supports_long_cache` (the same flag `cache_long`/24h-retention reuses just
below it): `openai.rs`'s Chat Completions dialect is the fallback for every third-party
OpenAI-compatible provider (`Dialect::for_model` routes native OpenAI ids to the Responses dialect
instead), and a strict-schema third-party endpoint can 400 the whole request over an unrecognized
field — `ModelCaps::unknown()`'s conservative default omits it unless a capability-table entry opts in.

### Why malformed streamed tool arguments are recoverable, not fatal

A streamed tool call whose JSON args never parse is a protocol glitch, not a model mistake the model
can't recover from. Rather than abort the run, the loop keeps the `tool_use` block (with a wire-valid
empty `{}` input so the next request doesn't 400) and feeds back an error `tool_result` naming the
problem, which the model corrects on the next turn — the same shape as any other recoverable tool error.

### Why dialect selection is a model-id prefix check, not configuration

The gateway relays bytes verbatim; it doesn't translate dialects. Per the "Beyond twist," provider
_selection_ (which upstream serves a request) is the gateway's job via the virtual key, but wire
_shape_ (`/v1/messages` vs `/v1/chat/completions` vs `/v1/responses`) is this crate's job, because the
gateway has no opinion on it. `Dialect::for_model` picks Anthropic for `claude*`/`*anthropic*`; for
everything else it consults `models::capabilities(model).api` (an `ApiKind`) — every native OpenAI id
(gpt-4/4.1/4o, the gpt-5 family, o-series) is `ApiKind::Responses` and gets `dialect/openai_responses.rs`
(`/v1/responses`); every third-party OpenAI-compatible id (OpenRouter, DeepSeek, Together, Cerebras,
xAI, Groq, Fireworks, Mistral) and any unrecognized id stays `ApiKind::ChatCompletions`. Adding a model
never requires touching the gateway. The Responses dialect reuses `ContentBlock::Thinking`'s `signature`
field for a reasoning item's _entire JSON-stringified content_ (not just a cryptographic signature the
way Anthropic's is) — the only way to replay OpenAI's encrypted reasoning across turns without a new
content-block variant; a signature that fails to parse as JSON (e.g. after `set_model` scrubbed a
cross-provider thinking history) degrades to plain text instead of erroring.

## Trust Boundaries

**What the system verifies (rejects if invalid):**

- HTTP status: a non-2xx response from the gateway is read as text and surfaced as
  `Error::Transport`, never silently treated as a successful empty stream.
- UTF-8 validity of each framed SSE line (`Error::Transport("invalid UTF-8 in SSE stream")`).
- JSON well-formedness of each SSE `data:` payload (`Error::Transport("malformed SSE json")`).
- JSON well-formedness of a streamed tool call's assembled arguments — a parse failure is _not_ fatal;
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

| File                          | What It Does                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lib.rs`                      | Crate root: module list, public re-exports, and the crate-wide `#![cfg_attr(test, allow(...))]` panic-free gate                                                                                                                                                                                                                                                                            |
| `agent.rs`                    | `Agent` config + `run`/`run_events`/`run_events_cancellable`/`run_events_steered` loop, `Accumulator`, concurrent tool dispatch (threading text/images/`terminate`), tool-driven termination, mid-run + stop-boundary steering, `with_reasoning_effort`, model-aware `new` defaults, hooks, auto-compaction + overflow retry                                                               |
| `message.rs`                  | `Role`/`ContentBlock`(+ `Thinking`/`RedactedThinking`/`Image`; `ToolResult` carries optional `images`)/`Message`/`ToolDef`/`StopReason`(+ `Refusal`)/`StreamEvent`/`TokenUsage`(+ `reasoning_tokens`) — the internal model                                                                                                                                                                 |
| `compaction.rs`               | Context compaction: trigger, cut-point search, summary-prompt build (`summary_request` takes an optional `custom_instructions`, appended as "Additional focus: …" — a manual compaction's client-supplied steering, matching pi's `generateSummary`; never applied to the split-turn prefix call, matching pi's `generateTurnPrefixSummary` not accepting one at all), file-op extraction, and incremental update (`previous_summary`/`SUMMARY_MARKER` fold-forward) — the network-free half of `Agent::compact`                                                                                                                                                                             |
| `models.rs`                   | `capabilities(model) -> ModelCaps`: minimal per-model wire table (max-tokens field, long-cache, vision, thinking shape, reasoning-effort, context window), matched by id prefix; consumed by the dialects and `Agent::new`                                                                                                                                                                 |
| `hooks.rs`                    | `AgentHooks` interception trait (`before_tool_call`/`after_tool_call`, both cancellation-aware) + `NoHooks` default                                                                                                                                                                                                                                                                        |
| `steering.rs`                 | `Steering` — two shared queues: `push_steer` (mid-run, folded onto the tool-results turn) and `push`/follow-up (injected at would-stop boundaries), each with its own independent `QueueMode` (`set_steering_mode`/`set_follow_up_mode`); `pending_count` peeks the combined depth of both lanes without draining; plus `request_stop`/`take_stop_requested`, a graceful-stop flag checked at turn boundaries; `clear()` drops both lanes and the stop flag without returning them, for a caller about to swap in a different session's conversation     |
| `write_lock.rs`               | `WriteLockRegistry` — a process-scoped, path-keyed async-mutex map (`Agent::with_write_locks`) extending same-path write exclusivity across `Agent` rebuilds (a `set_model`/`set_thinking` rebuild, or multiple sessions sharing one registry), layered on top of the per-turn write-target grouping below                                                                                |
| `tool.rs`                     | `Tool` trait (`run -> ToolOutput`, optional streaming `run_streaming` + `ToolProgress` sink) + `ToolOutput { text, images, terminate }` + `ToolRegistry` (name-keyed `Arc<dyn Tool>` map, last-registration-wins, name-sorted `definitions`)                                                                                                                                               |
| `transport.rs`                | `ModelRequest` (system/tools/thinking/`reasoning_effort`/`tool_choice`/cache_key/cache_long/`user_id` — Anthropic's `metadata.user_id` abuse-detection hint, unset by default), `ReasoningEffort` + `ToolChoice` enums, `ModelTransport` trait, `EventStream` alias                                                                                                                                                                                                          |
| `client.rs`                   | `GatewayClient`: production `ModelTransport`; retry-with-backoff (`Retry-After` delta-seconds _or_ HTTP-date), chunked-SSE byte framing into whole UTF-8 lines; sends `x-client-request-id: <cache_key>` for the OpenAI Responses dialect only, when a `cache_key` is set — connection-level session-affinity routing, distinct from `prompt_cache_key`'s cache-node affinity in the body, matching pi's `openai-responses.ts`                                                                                                                                                                                                                            |
| `dialect/mod.rs`              | `Dialect` enum (model-id → wire selection), `StreamDecoder` trait, SSE line-splitting (`push_sse_line`/`decode_sse`)                                                                                                                                                                                                                                                                       |
| `dialect/anthropic.rs`        | `/v1/messages` body builder (three prompt-cache breakpoints, capability-gated 1h TTL, per-model thinking shape, `tool_choice`, tool-result image rewrite) + decoder (text/thinking/tool/cache-usage, reasoning-token breakout, `pause_turn`/refusal-explanation, in-band error + truncation detection)                                                                                     |
| `dialect/openai.rs`           | `/v1/chat/completions` body builder + decoder — real translation: flattened messages, string-encoded tool args, `image_url` data-URIs (user + fanned-out tool-result images), `max_completion_tokens` vs `max_tokens`, `reasoning_effort`, `tool_choice`, synthesized block-stop events                                                                                                    |
| `dialect/openai_responses.rs` | `/v1/responses` body builder (flat `input` array of typed items, flat tool defs, `max_output_tokens`, `reasoning.effort` + `include:["reasoning.encrypted_content"]`, `store:false`) + decoder — genuine item-boundary events (`output_item.added`/`.done`), not implicit index-keyed deltas; a reasoning item's whole JSON becomes the `Thinking` block's `signature` for verbatim replay; genuinely interleaved items (concurrent tool calls) are buffered per-index and replayed as one fully-formed block once the focus item closes, rather than force-closed and truncated, since the shared `StreamEvent`/`Accumulator` contract only ever tracks one open block at a time; `function_call_arguments.done`/`output_item.done`'s own `arguments`/`content` resync (replace, not append) whatever the streamed deltas produced, so a single dropped/duplicated delta can't silently corrupt the final block |
| `branch_summary.rs`           | `branch_summary_request`: the (network-free) prompt builder for summarizing an abandoned tree branch on navigation — reuses `compaction`'s `render_prefix`/`SUMMARY_SYSTEM`/`extract_file_ops` unchanged, framed by its own instruction (no incremental-update path; a branch is summarized once); `windowed_by_budget` trims the rendered tail to fit the summarization call's own context, privileging a nested compaction/branch-summary entry (a dense recap, not raw conversation) past the ordinary cutoff as long as the accumulated tail is still under 90% of budget, matching pi's `prepareBranchEntries` |
| `session.rs`                  | `Session`: Arc-shared copy-on-write message history + step/token counters + `last_usage_message_count`, serde round-trippable; `scrub_cross_model_state(new_model)` downgrades a non-empty signed `Thinking` block to `Text` (drops it only if empty), drops `RedactedThinking`, and truncates a combined tool-call id — all from any message not stamped with `new_model`                                                                                                                |
| `error.rs`                    | `Error` (loop/transport, aborts the run) and `ToolError` (tool failure, becomes an error `tool_result`)                                                                                                                                                                                                                                                                                    |
| `mock.rs`                     | `MockTransport` + `turn::{text, tool_call}` builders — scripted, no-network loop testing                                                                                                                                                                                                                                                                                                   |
| `tests/client_socket.rs`      | `GatewayClient` over a real TCP socket: SSE decode, UTF-8 chunk-split reassembly, HTTP-error surfacing                                                                                                                                                                                                                                                                                     |

## Configuration

| Setting                                         | Default              | What It Controls                                                                                                                                                                                                                                                           |
| ----------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DEFAULT_MAX_TOKENS` / `Agent::with_max_tokens` | model-aware (≥ 4096) | Per-turn output ceiling (`max_tokens`/`max_completion_tokens` per dialect). `Agent::new` seeds it from `capabilities(model).max_output` (floored at 4096); the compaction `context_window` is likewise seeded from `capabilities`. Both still overridable via the builders |
| `DEFAULT_MAX_STEPS` / `Agent::with_max_steps`   | 50                   | Loop-iteration ceiling; once `session.steps` reaches it, the _next_ iteration returns `Error::MaxSteps` before sending a request — a runaway-tool-call backstop. `Error::MaxSteps` is resumable: the check runs before any per-turn state is touched, so a fresh `run`/`run_events_steered` call against the same session simply continues past it with a new per-call step budget |
| `Agent::with_system`                            | `None`               | System prompt; hoisted to each dialect's native system field (Anthropic top-level `system`, OpenAI leading `system` message)                                                                                                                                               |
| `Agent::with_tools`                             | empty                | The tool set advertised to the model; definitions + JSON Schemas computed once here, shared via `Arc<[ToolDef]>` for the agent's lifetime                                                                                                                                  |
| `CONNECT_TIMEOUT` (`client.rs`)                 | 10s                  | TCP+TLS handshake cap to the gateway; mirrors the gateway's own upstream `connect_timeout_secs`                                                                                                                                                                            |
| `READ_TIMEOUT` (`client.rs`)                    | 600s                 | Idle timeout _between_ reads on the streaming body (not total stream duration); sized to the gateway's own upstream `read_timeout_secs`                                                                                                                                    |

## Failure Modes

| Failure                                                            | What Actually Happens                                                                                            | Recovery                                                                        |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `session.steps` reaches `max_steps`                                | `run_events` returns `Error::MaxSteps(n)` before another request is sent; session retains every completed turn   | Caller persists/inspects `Session`; can raise `max_steps` and resume            |
| Gateway returns non-2xx                                            | Body read as text; `Error::Transport("gateway returned {status}: {detail}")` returned from `stream()`            | First `run`/`run_events` call errors; session has no partial turn               |
| SSE chunk splits a multi-byte UTF-8 char                           | Raw bytes buffered across chunks; decoding waits for the newline, so the split is invisible                      | Transparent — no error (regression-tested over a real socket)                   |
| SSE-framed line isn't valid UTF-8                                  | `Error::Transport("invalid UTF-8 in SSE stream: …")` from inside the event stream                                | Stream item `Err`; in-progress turn discarded, error returned                   |
| SSE `data:` payload isn't valid JSON                               | `Error::Transport("malformed SSE json: …")`                                                                      | Same as above                                                                   |
| Streamed tool-call JSON never completes/parses                     | The `tool_use` block keeps an empty `{}` input; the loop feeds back an error `tool_result` naming the bad buffer | Run continues; model corrects on the next turn (recoverable, not fatal)         |
| Model calls an unregistered tool name                              | That call's result becomes `("unknown tool: {name}", is_error: true)`                                            | Not fatal — fed back as an error `tool_result` the model sees next turn         |
| A registered tool's `run()` returns `Err`                          | The error's `Display` text becomes `ToolResult.content` with `is_error: true`                                    | Not fatal — same as above                                                       |
| Stream ends cleanly without a `MessageStop`/`Usage` event          | `stop_reason` defaults to `EndTurn`, token counts default to `0` — the turn looks like a normal completion       | Silent — no error surfaced; usage accounting is simply incomplete for that turn |
| Gateway holds the connection open with no bytes for >600s          | `reqwest`'s idle read timeout fires; the in-flight request errors                                                | Surfaces as a transport `Error`; no automatic retry in this crate               |
| A `Mutex` in `MockTransport` is poisoned by a panicked test thread | `.unwrap_or_else(                                                                                                | e                                                                               |
