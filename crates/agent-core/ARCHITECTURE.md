# agent-core Architecture

`agent_core` (package `beyond-ai-agent-core`) takes a [`Session`](src/session.rs) (message history)
and an [`Agent`](src/agent.rs) config (model, tools, transport) and drives the model-turn / tool-call
loop to completion, mutating the session in place and emitting an [`AgentEvent`](src/agent.rs) per
streamed token, tool boundary, and turn boundary. It contains no HTTP server and no provider SDK — its
only network dependency is a `ModelTransport` trait it never implements itself in production except via
the included `GatewayClient`, an HTTP client that speaks OpenAI/Anthropic wire to a Beyond gateway base
URL. The loop itself is still executor-agnostic (no `tokio`/executor dependency in `agent.rs`, `session.rs`,
`tool.rs`, `compaction.rs`, …); `GatewayClient`'s one Codex-specific live WebSocket connection
([`codex_websocket`](src/codex_websocket.rs), see the `Codex live WebSocket transport` capability below)
is this crate's sole exception — a persistent socket genuinely needs a bound async runtime the way a
per-request `reqwest` call never did, so this crate now depends on `tokio` directly rather than only
through `dev-dependencies`.

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
  no matching result. "No result yet" means genuinely unfinished, not merely un-joined: each call
  publishes its own result on an mpsc channel the instant that result exists, and the harvest into
  `results` happens _after_ the cancellation race resolves, so it serves both outcomes. Results used to
  be returned per _group_ — one `Vec` handed back only once the group's whole serial run finished —
  which meant cancellation dropped the group future along with every already-completed call's result in
  it, and the synthesized "cancelled: tool call aborted before it finished" then overwrote the real
  outcome of a tool that had run to completion. Its writes were already on disk and its `ToolEnd` had
  already reached the client, so the transcript contradicted both, and a resuming model would redo the
  work it was told never happened.
- **Steering** — `run_events_steered(.., steering)` drains a `Steering` handle's _two_ lanes:
  _follow-ups_ (`push`) injected as fresh user turns at each would-stop boundary, and _steers_
  (`push_steer`) folded onto the in-flight tool-results turn mid-run — so a client can either queue the
  next task or redirect a busy agent between tool turns without waiting for it to stop. A **refusal**
  (`StopReason::Refusal`) is a distinct terminal condition: the run ends immediately without draining
  either lane, since injecting a new turn right after a refusal would likely just be refused again — the
  queue is left intact for whatever run reads the same `Steering` handle next. Checked unconditionally,
  before `calls` is even collected (pi-parity fix) — a turn that streamed one or more complete `tool_use`
  blocks _before_ the model was cut off with a refusal (a real wire shape: a refusal explanation arriving
  as trailing content after a tool call already closed; OpenAI's `content_filter` maps to this same stop
  reason) used to have a non-empty `calls`, since the old check lived only inside the tool-less
  `calls.is_empty()` branch — dispatching tools the model was ultimately blocked from continuing. Matches
  pi's `agent-loop.ts`, which returns unconditionally on an "error"/"aborted" stop before ever looking at
  `message.content` for tool calls, dialect-agnostic. How much of a lane one
  drain call consumes is `QueueMode`, one **independent** setting per lane (`Steering::set_steering_mode`/
  `steering_mode` for the steer lane, `set_follow_up_mode`/`follow_up_mode` for the follow-up-plus-
  stranded-steer stop-boundary drain — matching pi's own separate `steeringMode`/`followUpMode`, not one
  shared toggle): `OneAtATime` (the default, matching pi's `PendingMessageQueue`) takes only the oldest
  queued message, leaving the rest queued for the _next_ drain point, so several messages queued in
  quick succession land as separate turns; `All` folds everything queued into one injection (this
  crate's original behavior, still available).
  `Steering::pending_count` peeks the combined depth of both lanes without draining either — pi's
  `pendingMessageCount` — for a host surface (e.g. `serve`'s `get_state`) that wants to report queue
  depth to a client.
  The steer lane is drained once more **after a compaction**, just before the next request is built:
  a host reacting to `Compacted` (the `/session` recall reminder) queues its steer after that turn's
  tool results already drained the lane, so it is folded onto the trailing user turn (prompt or tool
  results) of the request about to go out — the first one the model sees post-compaction. Without
  that, the model acted on the lossy summary first, and when it answered without tools the stranded
  steer cost a whole extra request. Noticed by `CompactionProvenance::compactions` moving, so all
  three compaction sites (top-of-loop threshold/overflow, overflow-error recovery, truncation
  recovery) are covered; checkpointed, then reported by `Steered` like every other delivery.
  A queued message is a `SteeringMessage` (text plus optional image attachments), not a bare `String` —
  the same shape a fresh `prompt` accepts, so `steer`/`follow_up` aren't a lesser channel that silently
  drops attachments a `prompt` would have kept. `From<&str>`/`From<String>` build a text-only message,
  so every plain-text `push`/`push_steer` call site keeps compiling unchanged. A message may carry an
  opaque `tag` (`SteeringMessage::with_tag`); the `Steered` event that reports its injection lists the
  tags injected (`AgentEvent::Steered::tags`, not serialized), emitted after the transcript holding
  them is checkpointed on both paths — so a host learns exactly which of its queued messages reached
  the model, whatever later happens to the transcript (`serve` records MCP Events delivery this way).
  On drain, a follow-up
  with images becomes a real `Message::user_with_images` turn instead of a plain-text one; a mid-run
  steer's images are appended as `ContentBlock::Image` blocks after its text block, onto the same
  tool-results turn its text rides on.
- **Host-held turn context** — `Steering::set_turn_context(source)` installs a host callback the loop
  pulls at every boundary that starts a user turn: after a tool-results turn mid-run, and after the
  follow-up/steer turns injected at a stop boundary (never at a plain stop with nothing queued, so the
  context alone never makes the model answer again). The source _takes_ what it returns, so each piece
  reaches the model once. What it returns is attached with `Session::attach_request_block` — a
  **request-only block**: appended, as its own content block, to that turn in every later request
  (`Session::request_messages`, the one place requests read history) but never to `Session::messages`,
  so it is not persisted, not part of the user's text, and not read by anything else (a title, an
  export). Each block is pinned to its message by value, so it rides the same turn in every request —
  keeping the request history byte-stable for the prompt cache — follows it through compaction
  (`apply_summary` re-indexes kept blocks), and lapses if that message is rewritten or dropped.
  `serve` uses this for MCP App views' `ui/update-model-context`.
- **Mid-run model switching** — `Steering::request_model_switch(model, thinking?)` (pi's
  `prepareNextTurn`/`nextTurnSnapshot` equivalent) retargets every subsequent turn of a run already in
  flight, applied at the same turn boundary a graceful stop is checked — never mid-turn, so the request
  already in flight when a switch is requested is unaffected. A `ModelSwitched` event fires when it's
  applied, after the same cross-model scrub the run start applies (`needs_cross_model_scrub` →
  `scrub_cross_model_state`, below): turns the previous model produced, this run's included, carry
  thinking signatures and combined tool-call ids the new vendor rejects (Anthropic: 400 ``Invalid
  `signature` in `thinking` block``). Deliberately narrower than pi's full snapshot: only the main conversational turns' own model/
  thinking budget are affected — `compact`/`compact_or_report`/`summarize_branch` keep using the `Agent`'s
  original, as-configured model, matching pi's own summarization path (no `nextTurnSnapshot` awareness
  there either). `serve`'s `switch_model` RPC command is the concrete surface for this; `set_model`/
  `set_thinking` are a separate, idle-only mechanism that only takes effect on the _next_ `prompt`.
- **Mid-run tool-set switching** — `Steering::request_tool_set(tools)` (mirrors the mechanism behind
  pi's real shipped `setActiveToolsByName`, `packages/coding-agent/src/core/agent-session.ts:840`, wired
  to the extension runtime's `setActiveTools` handler at line 2283) reconfigures a run's advertised
  tools — replacing both the wire definitions and the actual `Arc<dyn Tool>` lookup dispatch uses —
  applied at the exact same turn boundary a mid-run model switch is, so it takes effect starting the
  very next turn, never mid-turn. A `ToolsUpdated { tool_names }` event fires when it's applied — pi's
  own `setActiveToolsByName` has no discrete event of its own; this is this crate's own addition for a
  streaming client to observe the change. Before this, the tool set really was fixed for the whole run,
  not just the `Agent`'s own configured baseline.
- **Per-turn system prompt** — `Agent::with_system_fn(|| String)` installs a callback re-evaluated
  fresh every turn, at the same point the static `with_system` string would otherwise be read — takes
  priority over `with_system` when both are set (mirrors pi's function-valued `systemPrompt`,
  `harness/types.ts:817-826`, one field, either a string or a callback, re-evaluated via
  `createTurnState()`). `set_system(&mut self, ..)` needs exclusive access, unavailable while
  `run_events_steered` holds `&self` for a whole in-flight run — `with_system_fn`'s callback works from
  _inside_ one, through `&self` alone, so a long run can keep a time-varying prompt (a date stamp)
  current turn-to-turn without a full `Agent` rebuild.
- **Graceful stop** — two independent, OR-combined ways to end a run at a turn boundary rather than a
  hard abort: `Steering::request_stop` — an external flag a host sets from outside the loop (`serve`'s
  `stop_after_turn` RPC command) — and `AgentHooks::should_stop_after_turn` — an in-process,
  content-aware hook the loop itself calls with the turn's own assistant message and tool results (the
  external flag has no such content access; see the `Hooks` bullet below for why this needed its own
  seam). A beyond-only capability, not a literal port of anything reachable in pi's real shipped
  product: pi's own `shouldStopAfterTurn` config field is real (`agent-loop.ts`/`types.ts`), but it's
  dropped at the `Agent` wrapper class `packages/coding-agent` actually constructs (never forwarded by
  that class's `AgentOptions`/`createLoopConfig`), so pi's real product has no post-turn stop hook to
  mirror — the only place it's ever wired end-to-end is the unused
  `packages/agent/src/harness/agent-harness.ts`. Either wanting to stop ends the run. Checked at every
  turn boundary, _after_ that turn's tool calls (if any) have already run and their results are committed
  to the session, but before the next model call would start. Wins over both continuing a tool-call turn
  and draining queued follow-up/steer messages (mirroring the refusal case, those are left queued, not
  dropped). Unlike `cancel`, neither drops an in-flight future or leaves an orphaned `tool_use` — the
  difference between a graceful stop and a hard abort. Whatever a run does with a pending `request_stop`,
  `run_events_steered` always clears it before returning (a `Drop` guard, so an early `?`/error/
  cancellation return can't skip it), so a request can never bind to a later, unrelated call that reuses
  the same `Steering` handle.
- **Hooks** — `AgentHooks` (`with_hooks`) gates (`before_tool_call` → block reason), rewrites
  (`after_tool_call`), can end a run early (`should_stop_after_turn`, see the `Graceful stop` bullet
  above), and can rewrite the model's own generated content (`on_assistant_message`) — the permission/
  redaction/stop-decision seam. `before_tool_call` receives the _coerced_ input (pi-parity fix — matches
  pi's `prepareToolCall`, which calls `validateToolArguments` before `config.beforeToolCall`), not the
  model's raw, possibly stringified wire arguments: `coerce_tool_arguments` now runs in both dispatch
  paths (the default gate-the-batch loop and Task #28's fully-interleaved
  `run_tool_calls_interleaved`) _before_ the hook is called, not just before the tool itself runs, so a
  permission hook sees the same typed value (`"42"` → `42`) `Tool::run_streaming` is about to receive.
  `after_tool_call` sees (and can rewrite) a tool's `images: Vec<ImageSource>`
  alongside its text — not just the text, which used to be all a redaction hook could see or touch, so an
  image a tool returned (a screenshot `read` pulled off disk, say) passed through completely invisibly to
  any hook wanting to redact/replace it. Kept as parallel `(text, images)` fields rather than pi's own
  unified `content: (TextContent | ImageContent)[]` array — a bigger structural change to `ToolOutput`'s
  public shape this fix didn't need, since every existing hook only cares about 2-3 of the (still
  positionally clear) 8 parameters anyway. Every method also receives `&Session` (the live conversation as of the
  call — the requesting/just-completed assistant turn is already `session.messages.last()` or one before
  it, since the pre-dispatch checkpoint below establishes that invariant before any hook ever runs) and
  the run's `&CancellationToken`, so a hook can condition its decision on the surrounding conversation —
  not just the one call's own name/args — and observe (or react to) an in-flight cancellation.
  `on_assistant_message` fires once per turn, right after the assistant message is finalized (the
  `Aborted` marker, if any, already applied) but before it's pushed to `session.messages`, checkpointed,
  or surfaced in `AgentEvent::TurnEnd` — pi's `message_end` extension event, narrowed to the
  assistant-authored case (`before_tool_call`/`after_tool_call` cover the analogous tool-call/tool-result
  seams already). `should_stop_after_turn`/`on_assistant_message` both fail _open_ on a panic (don't stop
  / keep the original message) — unlike `before_tool_call`'s fail-_closed_ (blocks the call): neither is a
  security boundary, and losing content or halting an otherwise-healthy run over a buggy hook would be
  more disruptive than just continuing it; `on_assistant_message` additionally discards a replacement
  whose role isn't `Role::Assistant` (a caller bug), for the same reason. Defaults to `NoHooks`.
  `crates/agent`'s own `ToolPolicy` (`--deny-tool`/`--deny-bash-pattern`) is the concrete implementation
  this workspace actually installs; see its `ARCHITECTURE.md`.
- **Provider request/response hooks** — `AgentHooks::before_provider_request(&mut ModelRequest)` is a
  beyond-only capability with no equivalent in pi's real shipped product: pi's real
  `packages/coding-agent/src/core/sdk.ts:332-338` has exactly one pre-send hook, `onPayload`, and it
  only ever sees the literal dialect-specific wire JSON (the layer the wire-payload hook below operates
  at), never an earlier, abstract, dialect-agnostic request — the only place a `ModelRequest`-level
  pre-send hook ever existed in pi is the unused `packages/agent/src/harness/agent-harness.ts:251-320`
  (zero references from `packages/coding-agent`). `after_provider_response(status, headers)` _does_ have
  a real counterpart — pi's `onResponse` (`sdk.ts:340-346`), wired to the extension event
  `"after_provider_response"`. The request-hook half is called from `Agent::run_turn_once` (the one
  place already holding both the configured hooks and the not-yet-sent request) right before
  `self.transport.stream(req)`, so a host can inject a request-shape tweak that reaches `client.rs`'s
  dialect/body construction; a panicking hook's partial mutation is discarded, falling back to the
  request as it was before the call. The response half is instead called from
  `client.rs::GatewayClient::stream` itself, the instant a response's status/headers are known (before
  its body starts streaming) — `headers` is a plain `&[(String, String)]`, not a `reqwest`-specific
  type, so the hook trait stays implementable/testable without depending on `reqwest`, matching
  `ModelTransport`'s own "the loop never depends on reqwest" contract; a `MockTransport`-driven test
  never fires it, since there's no real response to report. Both default to no-ops, so an existing
  `AgentHooks` implementor is unaffected.
- **Wire-payload hook** — `AgentHooks::before_provider_payload(&mut serde_json::Value)` fills the gap
  between the two hooks above: `before_provider_request` only ever sees the abstract, dialect-agnostic
  `ModelRequest`, and `after_provider_response` is read-only, so nothing previously exposed the
  _literal_ dialect-specific wire JSON a host might need to inspect or rewrite. Called from
  `client.rs::GatewayClient::stream`, once per attempt, immediately after `dialect.build_body(..)` builds
  the wire body and before that tree is serialized to compact JSON bytes for `send_with_retry` — mirrors pi's real (single-layer) `onPayload`
  (`sdk.ts:332-338`, wired to the extension event confusingly also named `"before_provider_request"` —
  despite the name collision, it operates on the literal wire JSON, matching this hook, not the
  `ModelRequest`-level hook above). The unused `agent-harness.ts` additionally exposes its own,
  differently-shaped `beforeProviderPayload` (`emitBeforeProviderPayload`, which can replace the payload
  object wholesale), but that harness is dead code, not a second layer pi's real product actually has.
  This hook mutates `payload` in place rather than returning a wholesale replacement, matching
  `before_provider_request`'s own convention. Same "fails open" panic handling as the other two: a
  panicking hook's partial mutation is discarded and the payload falls back to exactly what `build_body`
  produced. Defaults to a no-op, so an existing `AgentHooks` implementor is unaffected.
- **Panic isolation** — every `before_tool_call`/`Tool::run_streaming`/`after_tool_call` invocation (in
  `agent.rs`) and `before_provider_payload` invocation (in `client.rs`) runs behind a `catch_unwind`
  boundary (`catch_tool_panic`, `pub(crate)` so both modules share it): a panic in any of them degrades
  to one failed tool call (an error `tool_result`, or a fail-closed block for `before_tool_call`) or a
  discarded payload mutation, instead of unwinding the whole run. None of these hook traits return a
  `Result` a panic could be redirected through instead, so this is the mechanism that actually needs to
  catch the unwind.
- **Compaction** — when the live prompt strictly exceeds `input_budget − reserve`, or a lower per-model
  `compact_at` (below) (landing exactly on
  the threshold still leaves the full reserve intact, so `compaction::should_compact` requires `>`, not
  pi-mismatched `>=`; `input_budget` is `min(context_window, input_limit)` —
  `ModelCaps::input_limit` records a vendor input cap below the window, as OpenAI's GPT-5/GPT-6 API
  enforces: the window less the max output, 272K of 400K and 922K of 1.05M. Triggering off the raw
  window put GPT-5.6's line at ~1.03M, past the 922K the API already rejects, so only overflow recovery
  ran. **A per-model cap can fire it earlier**: `ModelCaps::compact_at` (set in one place,
  `models::default_compact_at`) is a quality or price line, not a hard limit, and the trigger is
  `CompactionConfig::trigger_tokens` = `min(input_budget − reserve, compact_at)`. The cap _is_ the
  trigger rather than one more limit the reserve comes off: the reserve is headroom below a limit the
  vendor rejects past, and a cap already sits far under that, so `compact_at = 200K` compacts at 200K
  (subtracting the reserve too would put Haiku 5.5's 90K line at 73.6K). **Measured** by the
  long-context eval's 2026-10 threshold sweep (`crates/agent/eval/longctx`; 100K / 200K / `window −
  reserve`, n=3–4 per cell, quality neutral everywhere): Opus/Fable 5.x 200K (Opus 5.5 −15% vs the
  uncapped default) and Sonnet 5.x 200K (−12%; at 100K no cheaper, −10%, with about 3× the compactions
  and more turns, so the research's provisional 128K moved up). **Provisional**, with their rationale:
  GPT-6 and GPT-5.6 240K (under OpenAI's 272K surcharge, past which the whole request bills 2× input
  and 1.5× output; 200K measured −50% on GPT-6.1 Sol with neutral quality). Codex defaults GPT-6.1 Sol
  to 500K; we don't: our 200K-vs-500K experiment (Sonnet 5.5) found no quality gain, utils-text came
  out identical at both and 500K cost 24% more, and on pallets the 500K run hit its cost cap after
  spending 56% more. Haiku 5.5 90K (under its 100K tier), Grok, Gemini and Claude Sonnet 4.5 180K (under their 200K
  long-context tier), open-weight families 128K. **Everything else falls back to 200K** (an owner
  decision: no model runs to its full window by default; a window under 200K is unaffected, and
  `--compact-at 0` restores `window − reserve`). `crates/agent`'s `--compact-at`
  overrides any of them. `CompactionConfig::for_model`
  scales the reserve against the input budget and keep-recent against the trigger (half of it, so a
  compaction lands well under the line that fired it); the max-tokens leg of `is_hard_overflow` still measures prompt + output against the whole
  window; `models::tests::every_catalog_input_limit_bounds_the_input_budget` holds the table to
  `verify/catalog_truth.toml`'s `input_limit` rows) (or the provider rejects an
  overflow), [`compaction`](src/compaction.rs) summarizes the prefix via one model call and splices a
  summary in, keeping recent turns verbatim (`Agent::compact`, auto-trigger, compact-and-retry).
  `Agent::compact` returns a `CompactOutcome` (`Compacted`/`TooSmall`/`AlreadyCompacted`), not a bare
  `bool` — pi throws two distinct errors ("Nothing to compact (session too small)" / "Already
  compacted") for what this crate used to collapse into the same undifferentiated `false`; `serve`'s
  `compact` RPC surfaces the distinction via a `reason` field (`CompactOutcome::reason()`), while every
  overflow-retry call site still only branches on `CompactOutcome::compacted()` (the bare-bool question
  they actually care about). A completed compaction embeds `"Compacted from {tokens_before} tokens\n\n"`
  as a leading line in the spliced summary body (`compaction::apply_summary`'s new `tokens_before: u32`
  param) — read back by `crates/agent`'s `export.rs` to show a token count in an exported transcript's
  compaction block (matching pi's `entry.tokensBefore`), and stripped back out by `previous_summary`
  before an incremental re-summarization prompt ever sees it (so the line can't recursively accumulate
  turn over turn). `Agent::compact`'s `custom_instructions: Option<&str>` steers what the summary
  emphasizes — a manual
  (client-triggered) compaction's own focus, matching pi's `compact(customInstructions)`; every
  automatic trigger passes `None`, having no client in the loop to ask. A _second_ compaction is
  incremental: a prior summary (tagged `SUMMARY_MARKER`) is fed forward and updated rather than
  re-summarized, so early context isn't lost (and re-billed) each cycle. The fresh (non-incremental)
  render is wrapped in `<conversation>` tags, matching pi's `generateSummary`/`generateTurnPrefixSummary`
  and this crate's own incremental path's pre-existing `<new-activity>` wrapper; `SUMMARY_SYSTEM` also
  now ends with an explicit anti-continuation guardrail ("Do NOT continue the conversation. Do NOT
  respond to any questions in the conversation.") — the rendered transcript embeds real past user
  questions, and without it the model can slip into answering one instead of emitting the structured
  checkpoint format the call actually needs. A cut landing mid-turn splits
  into a **closed-history** call (gets `custom_instructions`) and a **turn-prefix** call (does not —
  matches pi's own split — run _sequentially_, not concurrently: pi originally ran them via
  `Promise.all`, then fixed exactly this itself so a single-concurrency local provider doesn't reject
  the second of two simultaneous completions; `SPLIT_TURN_PREFIX_SCALE` × `reserve_tokens` **directly**
  for the smaller one — matching pi's `generateTurnPrefixSummary` (`Math.floor(0.5 * reserveTokens)`),
  not a fraction of the already-scaled `summary_max_tokens`, which would instead compound the two scale
  factors into an effective ~0.4× `reserve_tokens`), merged under a "Turn Context (split turn)" header —
  closer to pi's two-template
  approach than summarizing the whole prefix with one minimal-context call. **Both token budgets scale
  with the model's real context window** (`CompactionConfig::for_window`; `Agent::new` uses
  `for_model`, the same caps plus the model's input limit, `compact_at`, family and price tier): the
  struct's `Default` states `reserve_tokens`/`keep_recent_tokens` as 200k-model _absolutes_ (16_384 /
  20_000; `for_model` takes keep-recent from the model's token family instead, see the next bullet),
  and seeding only `context_window` from the model's capabilities while leaving those two at
  their 200k values breaks the invariant the pair has to satisfy between them — _what a compaction keeps
  must stay comfortably under the threshold that triggers one_. The trigger fires at
  `context_window - reserve_tokens`, so on the catalogue's smallest window (32_768) the un-scaled numbers
  give a trigger budget of 16_384 while `keep_recent_tokens` retains 20_000: a compaction that keeps
  _more_ than the line it just crossed, so every turn re-triggers, each spending a real summarization
  call, and the context never gets back under the threshold. `for_window` caps the reserve at ¼ of the
  window and `keep_recent_tokens` at half the resulting trigger budget. The caps are one-sided (`min`,
  not a proportion) so every window at or above ~64k keeps byte-for-byte the tuning it had before — only
  the genuinely small windows, the ones that were broken, see different numbers.
  `summary_max_tokens` scales
  from the model's own `max_output` rather
  than a flat constant (`agent::scaled_summary_max_tokens`) — seeded by `Agent::new` and _rescaled_ by
  `with_compaction` whenever the incoming config still carries the struct's own flat default, so a
  caller replacing the whole config wholesale (e.g. `serve.rs`'s `build_agent`, to override just
  `reserve_tokens`) doesn't silently discard the model-aware value in favor of that default. The trigger
  adds `compaction::trailing_tokens` (messages appended since the last
  usage snapshot, estimated rather than assumed already reflected in `last_input_tokens`) so it doesn't
  compare a stale, undercounted size against the window; those messages are sized by the family's
  calibrated estimate (next bullet). Overflow detection
  (`agent.rs::is_context_overflow`) is table-driven (`OVERFLOW_PATTERNS`) with a `THROTTLE_EXCLUSIONS`
  allowlist checked first, so a provider's rate-limit/service-unavailable message (which can happen to
  contain "too many tokens") is never misread as a context overflow worth compacting for. A second,
  non-error-based detector (`compaction::is_hard_overflow`) fires regardless of whether auto-compaction
  is enabled: if the live prompt already meets the raw input budget (a silent overflow — the
  request that got there still succeeded, no error was ever raised), or a `StopReason::MaxTokens` stop
  lands where the live prompt plus the next turn's full output budget would meet it (the window, not
  `max_tokens`, is plausibly the real constraint), the run compacts anyway — disabling proactive
  compaction is a preference about timing with headroom to spare, not license to keep sending requests
  already guaranteed to overflow. The `MaxTokens` branch's `stop_reason` is sourced from
  `Message::stop_reason` (pi-parity fix — pi's own `AssistantMessage.stopReason`/`usage`, required
  fields there, read fresh by `isContextOverflow` on every `prompt()` call): `Message` persists the stop reason that
  produced it, and `run_events_steered` seeds its per-call `last_stop_reason` local from the session's
  own last assistant message rather than hardcoding `EndTurn`, so a `MaxTokens` turn a _prior_ run (or
  process, after a session reload) left behind is still visible to the very first top-of-loop check of
  a brand new call — matching pi's own re-derive-fresh-every-time behavior instead of trusting only an
  in-flight run's local state. Complementary and proactive, shared across **all three dialects**:
  `dialect::clamp_max_tokens_to_context` estimates the live prompt from `req.messages`/`req.system`
  directly and clamps the _request's own_ `max_tokens` down to whatever headroom is left under
  `context_window` (minus a margin absorbing estimation slop) — so a long-running session doesn't keep
  sending its static output ceiling on every turn regardless of how much of the window the prompt has
  already consumed, paying for a wasted round-trip each time before the reactive path above ever gets a
  chance to compact. A third call site, distinct from the two above (which both run at the _top_ of the
  loop, before a request is built): a turn that already completed successfully but got cut off by
  `max_tokens` alone, with no tool calls — a silent truncation, not an error — is checked against
  `is_hard_overflow` too, right before the run would otherwise end and hand back the hard-truncated
  non-answer. If compacting frees real room, the truncated response is discarded (never shown to the
  user, and through `Session::cut_back`, so the cut is recorded in `Session::cuts` for a persistence
  layer whose compaction checkpoint already wrote it to rewind past) and the same turn is retried
  fresh; if there's nothing left to compact, the truncated answer is reported as-is, unchanged from
  prior behavior. Every in-place shrink of the transcript (`pop_error_record` too) goes through
  `cut_back`, and an in-place rewrite of an already-pushed message (folding next-turn messages into the
  prompt) through `rewriting_from`; cut ids come from one process-wide counter (`cut_watermark`), and a
  compaction clears the list. Anthropic writes `max_tokens`, OpenAI Chat Completions
  writes `max_tokens`/
  `max_completion_tokens` (per `ModelCaps::max_tokens_field`), and OpenAI Responses writes
  `max_output_tokens` — all three funnel through the one shared clamp rather than each reimplementing
  it (a HIGH pi-parity gap, fixed: the clamp originally existed only on the Anthropic dialect, so the
  two OpenAI wire formats had no proactive protection at all against this exact failure mode). Never
  clamps below a configured thinking budget, and never raises
  `max_tokens` above what was asked.

- **Sizes in real tokens: anchors, then a calibrated estimate** — every size that spans messages (the
  kept tail, masking's window and what a pass frees, `tokens_after`, the cached summary request) is in
  the real prompt tokens the provider bills, from [`token_size`](src/token_size.rs). chars/4 was 1.9×
  low on Sonnet and Opus and 2.3–2.5× low on Haiku for a whole conversation (Claude 5.x stores a
  summary of its thinking but the prompt carries all of it, 5.7–6.4× as long; tool output runs at
  2.3–2.7 chars/token; each tool result costs ~80 tokens of framing), and 1.11–1.14× low on GPT.
  - **Anchors.** `Session::record_usage` keeps the request's real prompt size
    (`TokenUsage::prompt_tokens`: input + cache read + cache write, each counted once on every dialect)
    keyed to the boundary it saw; when the previous request's reply landed two messages earlier, the
    difference is the exact size of the turn between them, and `Session::push` stamps it on the reply
    as `Message::prior_turn_tokens`. Persisted with the message (`#[serde(default)]`: older sessions
    read `None`), so a reopened session keeps every measured turn; the in-memory anchor itself is not,
    so a reopened session measures its first turn afresh. Nothing is recorded across a rewrite:
    `apply_summary`, a masking pass and `scrub_cross_model_state` call `Session::forget_anchor` (the
    scrub also drops measurements on the turns it rewrites), and a reply from a different model than
    the turn's earlier reply, an error turn in between, or a prompt that shrank measure nothing.
    `token_size::message_sizes` uses a measured turn while it still describes the prompt: both its
    messages from the sizing model, no masked tool output in it. A suffix starting at an assistant
    boundary is then exact; the split inside a turn (assistant: its estimate; user: the rest) is the
    only estimated part.
  - **The calibrated estimate**, for the first message, the trailing turn no request has measured
    yet, rewritten turns and pre-anchor sessions: `TokenFamily` (from the model id) — Claude: tool
    output 2.5 chars/token + 80 per result, prose 2.3, an assistant message its recorded output tokens
    (the full thinking); GPT: tool output 3.5, prose 4.0, output less 70% of reasoning (only ~30% comes
    back). `Other` (unmeasured families) keeps chars/4. Whole-conversation real/est 0.96–1.02 at the
    median (p10–p90 0.93–1.08) on every family, against 1.83–2.20 (Claude) and 1.12–1.14 (GPT) for
    chars/4. The trigger's trailing step uses it too: a crossing fired one request late 20–22% of the
    time on Claude (step-size weighted), now 0.6–2.1% late and 3.3–5.0% early, by at most ~1.2K.
  - **The tail keeps the validated size.** `keep_recent_tokens` defaults per family
    (`TokenFamily::keep_recent_tokens`): Claude 40.5K, GPT 22K, other 20K. The threshold sweep that
    validated the caps ran on the chars/4 cut with a nominal 20K, which really kept 45.0K (Haiku),
    37.0K (Sonnet), 35.5K (Opus) and 20.7K (GPT) at the median; the anchored cut reproduces the pooled
    medians (37.8K Claude, 20.7K GPT) at those budgets. Masking's defaults scale by the same measured
    ratios. Which tail is best is a later A/B.
  - **`find_cut` keeps at most the budget**: the earliest assistant boundary whose suffix fits, or the
    last turn whole when even it does not. **`plan_cut` makes declining while over the trigger
    impossible**: an automatic or overflow compaction that would leave the prompt over the trigger
    (projected: live − the summarized prefix + `summary_max_tokens`) cuts deeper — into the tail, past
    a turn too big to keep, or past everything — and on a compacted session a cut at 1 (the old
    summary alone) is no progress. Before, a Haiku turn with 56K of hidden thinking sat in the tail
    sized at ~1K, the cut found nothing, `compact` returned `AlreadyCompacted`, and the prompt grew
    89K → 111K over 13 requests (two eval runs; all 10 Haiku requests over its 100K price tier).
    Replayed through `plan_cut`, both runs compact at the first request over the trigger, landing at
    35–36K. When nothing can get under (system prompt and tools alone fill the trigger), the
    configured cut stands and it is logged.
  - `Session::restore_usage_snapshot` rebuilds the live size on reopen (`crates/agent`'s
    `SessionStore::open`): the last usage, unless it equals the size the leading summary says it
    compacted from — a reopen between a compaction and the next reply, whose kept usage measured the
    pre-compaction prompt — then the summed sizes.
  - `crates/agent/eval/longctx/token_sizing.py` reproduces every number above from eval output.
- **Masking old tool output before summarizing** — when the prompt crosses the trigger,
  `Agent::mask_tool_output` first plans `compaction::mask_tool_results`: every tool result older than
  the last `mask_keep_turns` turns (assistant messages, default 10), or older than the last
  `mask_keep_tokens` of transcript when those turns hold more (default a quarter of the trigger, in
  real tokens scaled by the family's measured ratio: the quarter was tuned on the chars/4 walk, which
  kept 2.20× it on Haiku and 1.80–1.84× on Sonnet and Opus, 1.06× on GPT), becomes
  a one-line placeholder
  naming the call (`[tool output elided to save context: read(path=src/lib.rs), 48213 chars. Run the
  tool again if you need it.]`, images dropped). Kept: results of files being edited (the provenance's
  modified files plus this transcript's own `write`/`edit` targets), errors within `2 × mask_keep_turns`
  turns, results under 1,000 chars, and results already masked. The pass is applied only when it frees
  at least `mask_min_free_tokens` (real tokens, from the calibrated estimate of the elided output
  scaled by its message's measured size; default 25K, capped at a tenth of the trigger: 9K on Haiku's 90K
  in chars/4 terms, × 1.6 on Claude and × 1.14 on GPT so the same passes qualify —
  our `clear_at_least`). Both budgets scale with the trigger because a fixed ten turns could be most of
  a small trigger's context: on Haiku 5.5 masking never qualified in 21 compactions of the sweep, its
  older output 1–17K against a 22.5K bar **and** brings the prompt back under the trigger; then no summary
  runs (`AgentEvent::ToolOutputMasked`, no `CompactionStart`). A pass that would leave the prompt over
  the trigger is not applied: the summary runs on the unmasked conversation, because the summary
  replaces the old turns anyway and the cached summary call reads the live conversation from cache — a
  mask first rewrites it, and on Sonnet 5.5 those summaries read 88–105K uncached ($0.21–0.23) against
  $0.06 at 98% cache read. **Batched, never incremental**: it only runs past the trigger and a
  too-small pass is not applied, so history is rewritten at most once per fill cycle and every turn in
  between reads a byte-identical prefix from the cache (masking a little every turn invalidates the
  cache each turn and costs 2–3× more). No recall tool: in the evidence one was rarely called and
  bought nothing; the placeholder says to re-run the call. Only `ToolResult` blocks change, in place,
  so count, roles and every `tool_use_id` pairing hold by construction (`find_cut`'s invariant and
  role alternation), and request blocks riding a masked message move to its new content. The live-size
  snapshot drops by the estimate freed. Whether a pass gets under the trigger is now judged on real
  tokens (it used to be judged on chars/4 of the freed output, which Claude under-counts 1.5×). `Session::edited` lists the rewritten indices for persistence
  (`crates/agent`'s `SessionStore::record_edits`), cleared by `apply_summary`, whose rewrite carries the
  content. Off with `mask_tool_output: false`.

- **The summary call reads the conversation from cache** — `Agent::compact` first tries
  `conversation_summary_request`: the request a turn would send (same model, system prompt, tools,
  deferred tools, cache key and TTL, reasoning; one helper, `conversation_request`, builds both) with
  the trailing user message replaced by `compaction::conversation_summary_instruction` (the
  per-wire opener (`SummaryFraming`), "do not call any tools", the `SUMMARY_INSTRUCTION` format, incremental
  rules when the conversation opens with a summary, the file tags and any custom focus). Everything up
  to the previous turn's rolling breakpoint is a cache read, and the summarizer sees full tool results
  instead of 2,000-char excerpts. Only the trailing message's tool results are kept beside the
  instruction: a pending user prompt there wins over it (measured on gpt-6-luna, a pending "reply with
  just OK" made the summary "OK"), and it is in the kept suffix anyway. One call covers a split turn,
  since the model sees the whole turn. It falls back to the `summary_request` re-render when the
  conversation plus the instruction exceeds the input budget, or leaves the window room for fewer than
  2,048 output tokens (short of `summary_max_tokens`, the output budget shrinks to the room left: a
  trigger at `window − reserve`, on a small window or a user-set `--context-window`, overshoots by a few
  thousand tokens, and a shorter cached summary beats a full-price re-render), when it would cross the
  model's price tier (`CompactionConfig::price_tier`, from the rate card's long-context threshold via
  `models::price_tier`: Haiku 5.5 bills 5× over 100K) and the re-render fits under it (13 of 94 cached
  Haiku summaries crossed the tier; re-renders measured 16–21K), for an overflow (the provider already refused this conversation), when the model answers
  with a tool call, nothing, or anything that is not a checkpoint, when the cached call errors, or with
  `cached_summary: false`. The re-render itself is now sent
  `no_cache`: nothing re-reads that one-off prompt, so a breakpoint only bought a cache-write premium on
  all of it. The provenance carry is unchanged: host code appends it to whichever summary comes back.
  Measured on real models (summary call only, ~80–100K-token sessions; see the PR): cache-read share
  0% → 98–99%; input cost about equal when tool results are long (the re-render truncates them to
  ~10K tokens), and −84% (Haiku 5.5) / −87% (gpt-6-luna) when they are ~2K chars each and the re-render
  is nearly the whole conversation. `AgentEvent::Compacted.summary_source` says which call wrote the
  summary: `cached`, `rerender` (no cached call ran), or `cached_rejected` (it ran and its reply was
  turned down, so the re-render wrote it).

  **The opener depends on the wire, and the reply must be a checkpoint.** Claude Haiku 5.5 refused the
  cached summary in 30 of 33 eval compactions ("I'm not going to produce that checkpoint summary; the
  transcript stays with me"), and the refusal became the summary; 0 of 10 refused on the re-render.
  Replaying ten of those compaction points from their persisted sessions reproduced it (24 of 31
  refused) under the "stop working, reply with nothing but a summary" opener plus `SUMMARY_SYSTEM`.
  On the Anthropic wire (`Dialect::for_model` of the agent's model) the instruction opens with
  `HANDOFF_FRAMING` instead (`SummaryFraming::Handoff`): a checkpoint the model writes for itself from
  what the transcript shows, not its reasoning, with the format unchanged. That gave 0 of 30 refusals on
  Haiku and 0 of 5 on Sonnet 5.5. Swapping only "another LLM" for "you" did not help (15 of 20). Every
  other wire keeps the old opener (`SummaryFraming::Compact`). GPT-6 Luna, replayed as production sends
  it (the Responses dialect emits the instruction ahead of the trailing tool output), wrote a clean
  checkpoint 10 of 10 times under it. Under the handoff text it called `todo` instead 4 of 15 times, and
  2 of 15 times it looped `-` lines until the output limit.

  Whatever the wording, a cached reply is accepted only if it is a checkpoint. It must have a `## Goal`
  heading (`checkpoint_body`; a preamble before it is dropped). It must not be cut off at the output
  limit (`StopReason::MaxTokens`). And it must not be a runaway loop (`is_runaway`: 10 identical
  non-blank lines in a row). Luna's two loops opened with a real checkpoint, so the heading check alone
  accepted them. No other reply in the replays (220) and no summary the eval kept (55) repeated a line
  even twice in a row. Any other reply re-renders, as does a cached call that errors, instead of failing
  the compaction. The pre-merge replay ran 5 compaction points each on Haiku (handoff) and Luna
  (compact), in production ordering and at the production output budget. All 10 replies were accepted
  checkpoints: no refusals, no tool calls and no loops. Two other fixes lost. A mid-conversation
  `system`-role message also stopped the refusals (0 of 31), but Claude Sonnet 5 rejects that role with
  a 400. The top-level `system` prompt invalidates the whole cached history and still refused 2 of 10.
  Assistant prefill is a 400 on Haiku 5.5 and every 4.6+ Claude model.

- **Candidates under evaluation** — `CompactionConfig::candidates` (`CompactionCandidates`, the CLI's
  `--compaction-candidates`) holds compaction changes being A/B tested, each off by default so
  `Default` stays the shipped behavior. Two answer what the summaries said, across all 87 eval
  compactions:
  - `state_template` (`state`): every summary is one current-state document. A single-prompt
    agentic run is one long turn and `find_cut` always leaves the prefix ending on a tool result, so
    55 of 87 summaries took the split-turn path (51 opening "No prior history.") and each later round
    reused the previous summary verbatim (`turn_start == 1`) and appended another section: 30 summaries
    stack request sections, and re-split sizes grow 6.4K → 13K → 22.9K → 26.7K chars by round. Under
    this candidate `compact` never splits; the re-render uses `STATE_INTRO` (or `STATE_UPDATE_INTRO`,
    which rewrites the previous summary instead of appending to it), `STATE_RULES` (state as of the
    newest messages, in-flight step included; one document; no pasted code; about 1,500 words; the
    user credited only with the user's words) and `STATE_FORMAT` (the same headings, so `## Goal`
    still marks a checkpoint); the cached call uses the same rules and format, with `STATE_FOLD_IN`.
  - `sees_tail` (`tail`): the re-render also gets the kept suffix, read-only, in `<recent-context>`
    (`TAIL_NOTE`), so its summary is current rather than as of the cut. A Haiku judge rated 93% of the
    split-template summaries stale against the newest messages; the cached call, which sees the whole
    conversation, 40%.

  Two masking candidates were deleted. `mask-breakeven` applied a pass only when it paid for its
  cache rewrite before the freed room refilled (`0.1 · freed · (freed / growth) ≥ (w − 0.1) ·
  rewrite`). Its counterfactual was wrong: a pass runs only at the trigger, so skipping it means
  summarizing now, which rewrites the whole prefix and drops verbatim context. On Haiku (sqlparse and
  marshmallow, 4 runs a side) it made 27 compactions against 21 and 2 masks against 10, and the mean
  fixed fraction fell from 0.971 to 0.898; one run's post-compaction context collapsed and the run
  quit at 20%. `mask-best-start` (the same rule from the start that maximizes the saving) changed the
  start of 1 of 29 passes offline.

- **The deterministic-carry channel** — `apply_summary` is deliberately destructive: it _replaces_
  `session.messages` with `[summary, …kept_suffix]`, and `SessionStore::open` reloads that
  post-compaction list, so anything living only inside a folded-away message is gone for good. A
  summarizing model asked to "merge the previous summary" will paraphrase or drop specifics, so nothing
  that _must_ survive a cut may depend on it doing so. `CompactionProvenance` (`Session::compaction`) is
  the channel that does: `merge_provenance` folds it forward every round from the doomed prefix,
  `crates/agent`'s `session_store.rs` persists it on the `Entry::Compaction` record and restores it on
  reopen, and `Agent::compact` re-renders it into every new summary by appending host-generated blocks
  to the model's prose — never trusting the prose itself.

  Three things ride it, for the same reason:
  - **File awareness** — `extract_file_ops` (read vs. modified paths from `read`/`write`/`edit` calls;
    the modified side is `tool::written_paths`, so every file a multi-file `edit` names counts — less
    the files its result reports `FAILED` (`tool::edit_failed_paths`), since each file of such a call
    applies on its own),
    plus `extract_bash_writes` (files a `bash` command wrote), accumulated and deduped across rounds,
    rendered by `format_file_operations` as `<read-files>`/`<modified-files>`. The bash side parses
    the command text (`bash_writes::written_paths`: redirects, `tee`, `sed -i`, `perl -i`, `mv`/`cp`
    destinations, and Python heredocs or `-c` that `open(…, 'w')` or `write_text` a literal or a name
    bound to one, joined onto the last `cd`) rather than diffing the filesystem, so the provenance
    stays derivable from the transcript. Claude Sonnet 5.5 edited almost only through `bash` (2
    `write`/`edit` calls in 17 eval runs), so 19 of its 23 summaries missed changed files (102); over
    the eval's 4,623 bash commands the parser recovers all of them, and 25 of 25 sampled hits are real
    edits.
  - **The model's plan** — `extract_todos` (the last `todo` call's list), rendered by `format_todo_list`
    as `<todo_list>`. _Last-wins_, not accumulated: the `todo` tool's contract is a full replace, so
    folding two rounds' lists together would resurrect steps the model deliberately dropped — and an
    explicitly cleared list (`[]`) must beat the older one rather than letting a finished plan reappear.
    This is why `tools::todo` holds no state of its own (it couldn't: its registry is rebuilt on every
    `set_model`), and it is what `serve`'s `get_todos` falls back to once the `tool_use` block that
    carried the list has been compacted away.
  - **The user's requests** — `merge_user_requests`, rendered by `format_user_requests` as
    `<user-requests>`, so no later round has to re-infer them: without the carry, 23 of 52 later-round
    eval summaries did ("The user's exact request is not visible in this prefix…"), and only 34–40%
    kept the "don't edit tests" constraint. Each genuine user turn's text, deduped, capped at 4,000
    chars; past four, the first and the latest. Handed to the re-render beside the file tags too (its
    `<previous-summary>` has carry blocks stripped, so it would otherwise never see the user's words
    again). Harness notes riding user turns (background notices and reminders, the step-budget
    wrap-up — `background::is_note`) are not the user's words and are skipped.
  - **Work still running** — the tools' `Background::carry`, as `<background-jobs>`, so a compacted
    model does not start a job again that it forgot. The one block **not** in the summary text or
    `CompactionProvenance`: it is rendered from live state at each compaction and attached to the
    summary message as a request-only block (`Session::attach_request_block_at`), because it
    describes processes, and a persisted copy would list dead jobs as running after a restart (it
    once was, until a restart test caught it). A host that saves request blocks across restarts
    skips it (`background::is_carry_block`; `serve`'s MCP App context sidecar). A copy the model
    echoes is stripped like the rest. With nothing running it says so (`background::NO_JOBS`), and
    a run on a summarized session whose block is gone — every reopen, since request blocks are never
    persisted — attaches it afresh (`Agent::attach_background_carry`, a no-op while compaction's
    block still rides, so the cached prefix is undisturbed). The same call first asks each source to
    `Background::resume` — the first run it sees on a transcript, summarized or not — and attaches
    any note (the work that transcript left running, stopped since) to the run's prompt, also
    request-only; hosts that save request blocks skip both (`background::is_process_state`). That is what corrects a summary whose
    own prose says "the server is running as job 1": the summarizer is told not to record job state
    (`compaction::LIVENESS_NOTE`, on every summary instruction), but it still reads the bash results
    that started the job. The summarizer's input keeps the block rather than stripping it: it sits
    on message 0, so stripping it would turn the cached summary call's prefix into a full-price
    re-read; the instruction tells it to ignore the block instead.

  `compact` first strips any blocks the model echoed at the end of its own reply (`strip_echoed_carry`):
  the cached call sees the previous summary, blocks included, and a model that mirrors its shape ends
  the same way — live on Haiku 5.5 they doubled every round (1, 2, 2, 4, 4 over five) before this.
  `previous_summary` strips these blocks back off the body before it is fed forward — both into the
  incremental summarization prompt (or the model echoes a stale copy into its prose alongside the fresh
  ones) and into `compact`'s own `turn_start == 1` verbatim-reuse path (or every split-turn round appends
  another copy of every block, unboundedly). Adding a third rider should clear the same bar: it is the
  run's working state, and losing it degrades the model silently rather than failing loudly.
- **Thinking** — `ContentBlock::Thinking`/`RedactedThinking` + `ThinkingDelta`/`SignatureDelta` stream
  events; signatures replay verbatim (Anthropic requires it with tools). `with_thinking(budget)`; the
  thinking _shape_ (Anthropic enabled-budget vs adaptive) is chosen per model from the capability
  table, and `with_reasoning_effort` drives OpenAI reasoning models / Anthropic adaptive thinking.
  `models::ThinkingLevel` (`Off`/`Minimal`/`Low`/`Medium`/`High`/`XHigh`) is the portable vocabulary a
  caller (`serve`'s `cycle_thinking_level`/`set_reasoning_effort`) uses instead of a raw budget;
  `models::thinking_for_level(caps, level)` translates a level into the `(thinking_budget,
  reasoning_effort)` pair `with_thinking`/`with_reasoning_effort` need for a _specific_ model — setting
  both together for `Adaptive` shape (the budget is a pure on/off gate there; `output_config.effort`
  carries the real depth), only `reasoning_effort` for OpenAI reasoning models, only a scaled budget for
  `Budget` shape. Whatever `ReasoningEffort` reaches a dialect — via `thinking_for_level` or a raw
  `with_reasoning_effort`/`--reasoning-effort` call — is clamped there to what the specific model's wire
  actually accepts (`models::clamp_reasoning_effort`): several OpenAI reasoning models (o-series, bare
  gpt-5, every gpt-5.1 variant) and Anthropic `claude-sonnet-4-6` have no `xhigh` tier and clamp down
  to `high`; Sonnet 5 / Opus 5 / Fable 5.1 accept `xhigh`. `gpt-5.5`/`gpt-5.5-pro` and the GPT-5.6 /
  GPT-6 family reject `minimal`(+`low` on 5.5-pro) and clamp up. Anthropic
  adaptive additionally has no `minimal` wire tier at all (always sent as `"low"`) and remaps `xhigh` per
  model (`"max"` on `claude-opus-4-6` uniquely; `"xhigh"` elsewhere) via `models::anthropic_adaptive_effort_wire`.
- **Model capabilities** — [`models::capabilities`](src/models.rs) maps a model id (by prefix) to a
  minimal `ModelCaps` table (context window, max output, `max_tokens` vs `max_completion_tokens` field,
  long-cache support, vision, thinking shape, reasoning-effort, explicit-disable capability, eager
  tool-streaming). The dialects and `Agent::new` consult it, so adding a model rarely needs new
  request-shape plumbing. Coverage extends past Anthropic/native-OpenAI ids to the third-party
  OpenAI-wire families the gateway actually routes to — DeepSeek, Z.ai/GLM, Moonshot/Kimi, Qwen,
  MiniMax, xAI, Groq, Cerebras, Together, Mistral, and a generic vendor/model-shaped fallback
  (OpenRouter, or any uncatalogued addition) — matched by id prefix/substring rather than an exhaustive
  per-exact-id table the way the native Anthropic/OpenAI branches are (those providers' catalogues are
  large and change often; a reasonably-accurate family default closes the truncation/no-thinking gap
  without needing to track every id). Mistral is the one exception: its ~30-id catalogue has enough
  per-id variance (and no shared prefix across its codestral/devstral/ministral/magistral/pixtral/
  open-mistral/open-mixtral sub-families) that it's matched id-for-id instead — see
  `models::is_mistral_model` (also shared with `dialect::openai`'s Mistral tool-call-id reshaping,
  below) and `models::is_deepseek_model` (narrower than `OpenAiReasoningFormat::DeepSeek`, which
  Moonshot/Kimi shares too; used by DeepSeek's own assistant-replay quirk, below). `ModelCaps::openai_reasoning_format`
  (an `OpenAiReasoningFormat`, only read by `dialect::openai::build_body`) selects which of these
  families' real reasoning-toggle shape to emit — `Standard`'s bare `reasoning_effort` string (o-series,
  gpt-5, xAI, Groq, Cerebras, Mistral, and anything unrecognized), or a third-party shape (`DeepSeek`'s/
  `Zai`'s nested `thinking:{}` toggle, `Together`'s/`OpenRouter`'s nested `reasoning:{}` object) —
  mirroring pi's own `compat.thinkingFormat` tag. Several of these families (Kimi K2; GLM below 5.2) have a
  real on/off toggle but no graduated effort vocabulary at all (`reasoning_effort: false`). Kimi K3 is
  the break: always-on thinking, a real `low`/`high`/`max` effort vocabulary (`xhigh` remaps to `"max"`),
  OpenAI `reasoning_effort` on a bare `kimi-k3` id and OpenRouter's nested `reasoning:{effort}` on a
  vendor-slug id such as `moonshotai/kimi-k3`. Putting K3 in the K2 bucket omits effort and the
  provider defaults to `max` — same session thinking level as Pi, ~40× the reasoning tokens.
  GLM-5.3 is the same class of miss: always-on (`thinking.disabled` is rejected), `low`/`high`/`max`
  (portable `medium` → `"high"`, `xhigh` → `"max"`), Z.ai thinking toggle on a bare `glm-5.3` id and
  OpenRouter nested `reasoning:{effort}` on `z-ai/glm-5.3`. It must not hit `nvidia_caps` (that table
  owns `z-ai/glm-5.2` and strips reasoning).
  `models::has_reasoning_mechanism` (consulted by
  `available_thinking_levels`/`clamp_thinking_level`/`thinking_for_level`) treats
  `openai_reasoning_format != Standard` as its own "has a mechanism" signal so such a model isn't
  reported as `Off`-locked. Once a level clears `clamp_reasoning_effort`, `models::reasoning_wire_override(model,
  effort)` gets one more say over _how it's spelled_ on the wire before `dialect::openai`'s
  `apply_reasoning_wire` falls back to the effort's own literal name — mirroring pi's per-model
  `thinkingLevelMap` remaps (DeepSeek's `xhigh` → `"max"`; Kimi K3's `xhigh` → `"max"`; GLM-5.3's
  `medium` → `"high"` and `xhigh` → `"max"`; GLM-5.2's `low`/`medium`/`high` → `"high"` and `xhigh` →
  `"max"`; Groq's one qwen id's `high` → `"default"`;
  every Mistral reasoning id's any active level → `"high"`, matching its real API's bare `"none"|"high"`
  vocabulary). Deliberately a
  standalone lookup, not a `ModelCaps` field — `ModelCaps::adaptive_xhigh_effort_wire` already covers
  the analogous single-value need on the Anthropic adaptive shape, but a per-model _map_ would need
  touching every one of this table's ~25 `ModelCaps` struct-literal construction sites for a table only
  four families use. A model id genuinely reachable through more than one host (a Together-hosted
  "deepseek-ai/deepseek-r1" also matching the native-DeepSeek prefix, say) can still land on the wrong
  sub-branch — a known limitation of a table keyed on model id alone, with no route/provider context to
  disambiguate by. That same limitation caps `models::is_non_standard_store_provider` (which providers
  withhold `store:false` — see `dialect/openai.rs` below): it recognizes DeepSeek/Zai/Kimi/Grok/
  Together-Qwen/Ant-Ling/Cerebras-native by id, but can't recognize NVIDIA/Cloudflare at all (no id
  shape of their own — `nvidia.models.ts`'s ids are ordinary `org/model` vendor slugs a NIM proxy can
  serve arbitrarily, not a fixed prefix, though the _capability numbers_ for its current catalogue are
  ported id-for-id regardless, `models::nvidia_caps`). MiniMax/MiniMax-CN, OpenCode/OpenCode-Go, and
  Fireworks were all the identical dialect-routing gap as each other: pi serves several of their ids over
  the Anthropic wire, but `Dialect::for_model` only recognized a `claude`/`anthropic`-named id as
  Anthropic by default, silently misrouting the rest through Chat Completions. Fixed (not just
  documented) for these three: `dialect::routes_to_anthropic_by_default` now recognizes MiniMax's/
  MiniMax-CN's/OpenCode's/OpenCode-Go's own known bare (unprefixed) Anthropic-wire ids, and Fireworks'
  own distinctive `"accounts/fireworks/"`-prefixed ids (all but its two genuinely-`openai-completions`
  `glm-5p2` ids), by default — no per-model `models.json` override required for any of them. Kimi-Coding
  keeps the config-override path instead (few ids, already user-configured); Mistral has the identical
  gap for a different reason (no bespoke wire client in this crate at all, unlike pi's own
  `mistral-conversations.ts`): every Mistral id is routed through the generic Chat Completions dialect,
  which can approximate its reasoning toggle (via `reasoning_wire_override`, above) and its
  tool-call-id shape (below) but not its real `promptMode` field — that one remains an accepted,
  documented gap, not a default-routing candidate the way the other three were.
- **Thinking-budget overrides** — [`models::budget_for_effort_with_override`](src/models.rs) is the
  same fixed effort→token-budget ladder [`models::budget_for_effort`] uses (itself unchanged, and still
  the only path `thinking_for_level` calls), but takes an optional
  `&HashMap<ReasoningEffort, u32>` consulted before the hardcoded ladder — the extension point a
  settings/CLI layer can wire an operator-supplied override through (e.g. `--thinking-budget
  high=40000`) without this crate taking any opinion on where that table comes from. The `max_output`
  clamp still applies to an overridden value the same as the built-in one.
- **Vision downgrade** — a model whose `ModelCaps::supports_vision` is `false` never sees a raw image
  block: each dialect replaces a user/tool-result image with a text placeholder
  (`"(image omitted: model does not support images)"` / the tool-result variant) instead of sending
  bytes the model can't decode. Dispatch also injects a schema-undocumented `_model_supports_vision`
  boolean (keyed off the active model's own `ModelCaps::supports_vision`) into every tool call's
  `input` object right before `run`/`run_streaming` — invisible to the model (it never appears in the
  advertised `input_schema`) and ignored by every tool except `crates/agent`'s `read`, which reads it
  back to append a "current model doesn't support images" note when the file it read is an image; see
  `agent.rs::tests::dispatch_tells_a_tool_whether_the_active_model_supports_vision`.
  `Agent::with_block_images(true)` is an operator-facing override that forces this same
  `_model_supports_vision: false` dispatch regardless of the active model's real capability — the
  mechanism a host's own `--block-images`-style flag wires into (a cost/bandwidth policy, a text-only
  audit log), not just a fallback for a genuinely non-vision model. Defaults to `false` (report the
  model's real capability untouched).
- **Eager tool-input streaming** — every Anthropic tool definition is marked
  `eager_input_streaming: true` when `ModelCaps::supports_eager_tool_streaming` is set (true for every
  current Anthropic id); `client.rs` only sends the mutually-exclusive
  `fine-grained-tool-streaming-2025-05-14` beta header for a model that lacks the per-tool marker —
  never both.
- **Tool output & multimodal** — `Tool::run` returns `ToolOutput { text, images, terminate }`: a tool
  can attach images the multimodal model sees (a screenshot, `read` on an image), and `terminate` ends
  the run when _every_ call in the batch agrees. `ContentBlock::ToolResult` and both dialects carry the
  images through to the wire.
- **Delegated spend** — `ToolProgress::report_usage` is the progress channel's other use: a tool that makes model
  requests of its own (`subagent`'s child runs) reports each request's `TokenUsage`. The dispatch sinks
  it as `AgentEvent::ToolUsage`, adds the batch's sum to the session's cumulative counters
  (`Session::record_delegated_usage`: never `last_input_tokens`, since the child's prompt is not this
  conversation's), and persists it as `Message::delegated_usage` on the batch's tool-results message,
  so a reopened session's message-derived totals still include it. A cancelled dispatch keeps what was
  spent.
- **Streaming tool progress** — a tool may override `Tool::run_streaming(input, &ToolProgress)` to
  `emit` incremental output while it runs (pi's `tool_execution_update`); the dispatch forwards each
  chunk as `AgentEvent::ToolProgress` the instant it arrives (not batched after the group joins), and
  `AgentEvent::ToolEnd` fires per call the moment its own result is known — a client watching the event
  stream sees completions in actual finish order, not batched after the slowest call in the group.
  Default `run_streaming` delegates to `run`, so non-streaming tools are untouched. `ToolProgress` also
  carries the run's `CancellationToken`: `is_cancelled()` for a poll check, `cancelled()` for a future a
  tool's own execution loop can race against (`tokio::select!`) so it notices a cancellation promptly and
  gets a chance to flush partial state itself, rather than relying solely on the dispatch dropping its
  whole future out from under it with no chance to finalize — `tools::bash::Bash::exec` is the first real
  user, racing its subprocess runner against this to return already-captured output on cancellation
  instead of discarding it.
- **Cross-run file-mutation exclusivity** — `Agent::with_write_locks` shares a `WriteLockRegistry`
  (`write_lock.rs`) across every `Agent` rebuild for a process's lifetime (not just one turn's grouping):
  a group's targets acquire the registry's per-path async locks for the group's whole serial run
  (`WriteLockRegistry::lock_all`, which takes them in sorted order so two multi-path holders cannot
  deadlock), so two `Agent`s built back-to-back (a `set_model` rebuild, or two sessions sharing one
  registry) can't race a same-path `edit`/`write` against each other. Layered on top of, not a
  replacement for, the intra-turn write-target grouping described below. `WriteLockRegistry`'s map entry
  for a key is evicted (pi-parity fix) once its last holder releases it — the returned `WriteLockGuard`
  checks, on drop, whether the map is the _only_ remaining `Arc` reference for that key
  (`Arc::strong_count == 1`) and removes the entry if so, race-free because the check-and-remove runs
  while holding the same map lock every concurrent `lock()` call must also acquire to clone the `Arc`
  (mirrors pi's `file-mutation-queue.ts`'s identity-checked `finally`-block delete). Before this fix the
  map instead accumulated one entry per distinct path _ever_ locked for the registry's whole lifetime,
  unbounded for a long-running `serve` process. The same eviction check also runs on the _acquire_ path
  (a `PendingLock` drop guard, defused into the `WriteLockGuard` on success): a `lock()` future dropped
  while still parked — a cancelled run — never constructs a guard, so without it the last reference to a
  contended key could die with no one left to evict the entry, reintroducing exactly the leak above.
  The guard is acquired at the dispatch seam and normally held for the tool's whole run, releasing when
  that run's future resolves. One case needs more than that: `edit` performs its atomic write on a
  `spawn_blocking` thread, which tokio cannot cancel — on a cancelled run the dispatch future is dropped
  (abandoning the `.await`) while that write runs on regardless. Releasing the guard when the dispatch
  future drops would then free the lock with the write still physically in flight, letting another
  turn/session acquire the same path and interleave (a lost update). So the guard is shared as an `Arc`
  and a clone rides _into_ the `spawn_blocking` closure via `ToolProgress::write_lock_keepalive`: the
  registry lock releases only once every clone is gone — i.e. once the write has landed — regardless of
  when the dispatch future was dropped. (`write` needs no such move: it mutates synchronously within its
  own future, which can't be preempted mid-write.) Pinned by
  `agent::tests::cancelling_a_write_holds_its_lock_until_the_in_flight_blocking_write_lands`.
  Two documented limitations: this only serializes within one process (cross-process locking would need
  a filesystem advisory lock, out of scope until a real multi-process use case needs it); and
  `futures::lock::Mutex` is _barging_, not FIFO — the registry guarantees mutual exclusion but no
  fairness, so sustained contention on one path can starve a waiter. The only ordering that actually
  matters (a turn's own same-target calls) is already guaranteed upstream, where they run serially in
  call order as one group.
- **Explicit reasoning/thinking disable** — `ModelCaps::reasoning_disableable` (per exact model id, not
  per `ThinkingShape`) drives an explicit "off" signal (Anthropic `{"type":"disabled"}`, OpenAI
  `{"effort":"none"}`) on a turn that isn't requesting thinking, for a model capable of one — instead of
  omitting the field and trusting the provider's undocumented default. Claude Sonnet 5.5 and later
  Sonnets reject `disabled`; their "off" is `{"type":"between_tools"}`
  (`models::thinking_off_is_between_tools`, parsed like the gateway's `ClaudeModel::of`).
- **`ThinkingLevel::Off` is not a legal state for every model** — a model with a reasoning mechanism it
  can't explicitly disable (`reasoning_disableable == false`; most of the OpenAI gpt-5 codex/pro line,
  `claude-fable-5`) has no way to actually turn reasoning off: sending `Off` there doesn't disable
  anything, it just omits the reasoning field entirely and lets the provider apply its own hidden
  default effort — silent cost/latency divergence from what a client believes is happening.
  `models::clamp_thinking_level(caps, level)` bumps `Off` up to the model's own floor
  (`ModelCaps::min_reasoning_effort`) for exactly these models (every other rung delegates to
  `clamp_reasoning_effort` unchanged); `models::next_available_thinking_level` is the cycling
  counterpart, advancing only through `models::available_thinking_levels(caps)` rather than the raw
  6-rung ladder — a plain `level.next()` then re-clamp would bounce forever between `High` and a
  re-clamped `XHigh` for any model lacking `xhigh` support. `serve`'s startup level, `set_model`/
  `cycle_model` (re-clamping the _existing_ level against the _new_ model), `set_reasoning_effort`, and
  `switch_branch`'s restored level all clamp through these before the level is ever stored or built into
  an `Agent` — mirrors pi's `clampThinkingLevel`, applied on every model switch and level change.
- **Cross-model state scrubbing** — `Message::model_id` (stamped on every assistant turn from
  `Agent`'s own `model`) records which model produced it; `Session::scrub_cross_model_state(new_model)`
  downgrades a non-empty signed `Thinking` block to a plain `Text` block (preserving the visible
  reasoning trace as context rather than erasing it — only the block's _replayability as thinking_ is
  model-specific, not the prose itself), drops an empty `Thinking` block or any `RedactedThinking`
  block outright (opaque ciphertext, nothing to preserve), and truncates a combined OpenAI-Responses
  tool-call id (`"call_id|item_id"` → `"call_id"`) — all from any message not stamped with
  `new_model`. `model_id: None` (a message from before this field existed) is always treated as
  foreign. `anthropic::build_body`'s own `downgrade_unsigned_thinking` (an unsigned — not
  cross-model — thinking block, e.g. from an aborted stream) follows the same empty-drops-instead-of-
  degrades rule: a block whose `thinking` text is also empty is dropped rather than downgraded to
  `{"type": "text", "text": ""}`, which Anthropic's non-empty-text requirement would just as readily
  reject. `anthropic::build_body`'s own `normalize_cross_model_tool_ids` is the belt-and-suspenders
  counterpart to the id-truncation above, applied at the point a request actually reaches the wire
  rather than only at an explicit model-switch: any `tool_use.id`/paired `tool_result.tool_use_id` from
  a message not stamped `model_id == req.model` gets every character outside `[a-zA-Z0-9_-]` replaced
  with `_` and truncated to 64 chars — Anthropic's own id-shape requirement — covering a same-turn
  multi-model fan-out, a hand-edited/externally-loaded session, or a non-Anthropic-native id shape
  (OpenAI Responses' `"call_id|item_id"` combined form, or a non-standard OpenAI-compatible provider's
  own 450+-char blobs) that reaches an Anthropic request some path other than `scrub_cross_model_state`
  didn't already normalize. Matches pi's `normalizeToolCallId` (`anthropic-messages.ts:1006-1009`).
- **Tool choice** — `ModelRequest::tool_choice` (`Auto`/`None`/`Required`/`Tool(name)`) maps to each
  dialect's vocabulary; unset emits nothing (provider default), so the common request shape is intact.
- **Transport resilience** — `GatewayClient` retries transient failures (429/5xx/connection, honoring
  `Retry-After`) up to the first byte; a mid-stream `event: error` or truncated stream surfaces as
  `Error::Transport` (the SSE decoder's `finish` returns `Result`), which `Agent::run_turn` retries with
  backoff (`is_retryable_mid_stream`, capped at `MAX_MID_STREAM_RETRIES`) from a fresh connection and
  a fresh `Accumulator`, rather than resuming a dead attempt's partial blocks. Beyond the decoder's own
  truncation rejection and a tagged network failure (`MID_STREAM_NETWORK_ERROR`),
  `is_retryable_mid_stream` recognizes a table of named in-band provider error _types_
  (`MID_STREAM_RETRYABLE_ERROR_TYPES` — Anthropic's `rate_limit_error`/`api_error`/`timeout_error`,
  OpenAI's `rate_limit_exceeded`/`server_error`/`internal_error`/`service_unavailable`), a free-text
  prose fallback for a provider error with no recognized `error.type` at all
  (`MID_STREAM_RETRYABLE_FREE_TEXT_PATTERNS` — pi's `RETRYABLE_PROVIDER_ERROR_PATTERN`, narrowed to
  this crate's two dialects), and explicit provider retry-guidance phrases
  (`MID_STREAM_RETRY_GUIDANCE_PHRASES`), deliberately keyed on names/prose rather than raw HTTP
  status-code substrings ("500" et al.) — a mid-stream failure never carries a fresh status code to key
  on, and a bare digit substring risks matching an unrelated number in the message (raw status digits
  are instead a whole-run-only concern — see the `agent` crate's `retry.rs`). `Agent::with_auto_retry`
  (default enabled) disables this specific layer — a normally-retried
  failure surfaces on the very first attempt instead — for debugging a flaky connection without several
  silent attempts first; `GatewayClient`'s own pre-first-byte retry above is unaffected either way.
- **Rate limits get a time budget, not a count** (`rate_limit.rs`) — inside that same `run_turn` loop, a
  failure `rate_limit::is_rate_limited` recognizes (a 429, either dialect's in-stream rate-limit error,
  or Anthropic's 529 `overloaded_error`; never a quota/billing 429, a context overflow, or OpenAI's
  "Request too large … must be reduced") is re-sent until `RateLimitRetry::budget` (default 300s) would
  be overrun, instead of counting against `MAX_MID_STREAM_RETRIES`. Each wait is the larger of the
  server's hint and `base_backoff · 2^(n-1)` ± 20% (default 500ms base, 60s cap per wait). The hint is
  read from the error text by `rate_limit::retry_hint`: `retry-after-ms`, then `retry-after`, then
  OpenAI's "try again in 243ms", then (only as a last resort) the longest `x-ratelimit-reset-*` — the
  reset is when the bucket is _full_, not when this request fits (59s vs 243ms in the TPM error that
  motivated this). An in-stream OpenAI error carries those headers in its body; for an HTTP 429 the
  client writes the wait it derived from the response headers into its final error as
  `(retry-after-ms: N)`, so both paths reach the same parser. A run that outlasts the budget fails with
  `rate_limit::BUDGET_EXHAUSTED` leading the message, which `is_retryable_mid_stream` (and so the
  `agent` crate's whole-run retry) refuses, so the wait is never restarted. Every retry in this loop,
  either class, is announced as `AgentEvent::Retry { attempt, delay_ms, rate_limited, message }` before
  its wait, and the wait races cancellation. Retrying after partial output is safe by construction: a
  retried attempt starts a fresh `Accumulator`, and tools only run on a `Turn` that completed, so a
  tool call streamed before the failure is neither executed nor kept. `Agent::with_rate_limit_retry`
  sets the policy. Defaults, for comparison: Claude Code retries 10 times (500ms base) by default,
  with `CLAUDE_CODE_RETRY_WATCHDOG` to wait 429/529 out indefinitely; Codex retries a stream 5 and a
  request 4 times at a 200ms base, honouring `Retry-After` from headers or the in-stream error body.
  `is_retryable_status` also matches `524` (Cloudflare's "origin didn't respond in time") alongside the
  other 5xx entries — matching pi's `packages/ai/src/utils/retry.ts:36`, which treats it as its only
  retry signal for that status; the coarser whole-run retry layer (`crates/agent::retry`) already covered
  it, but that restarts the entire turn from scratch instead of this cheap pre-first-byte retry.
  `MAX_BACKOFF` (the ceiling on both exponential backoff and a server-supplied `Retry-After` hint) is
  raised from an earlier 10s to 60s, toward pi's own `openai-codex-responses.ts` default
  (`DEFAULT_MAX_RETRY_DELAY_MS`) — at 10s, a 429 with a `Retry-After: 30` hint got retried back into the
  very rate-limit window it named, capable of exhausting the whole retry budget before that window
  closed; `GatewayClient::with_max_backoff` overrides it per client, independent of
  `with_retry`'s `max_retries`/`base_backoff` pair. `GatewayClient::with_extra_headers(HashMap<String,
  String>)` merges an operator-configured header map onto every outgoing request, applied last (via
  `HeaderMap::insert`, not `RequestBuilder::header`'s append semantics) so an operator's value always
  wins on a name collision — the plumbing a self-hosted/proxied endpoint's custom auth/routing header
  needs (pi's `model-registry.ts` supports the same per-deployment concept); the CLI/settings surface
  that actually lets an operator configure this lives in `crates/agent/src/settings.rs`, not here.
- **Cache observability** — `StreamEvent::Usage` carries `TokenUsage` (input/output + cache-read/write
  - reasoning); both decoders populate it, and `Session` folds the cumulative totals + `last_input_tokens`.
    `Message::usage: Option<TokenUsage>` additionally stamps the _exact_ per-turn figure directly onto the
    assistant message that turn produced (pi's own `AssistantMessage.usage`, a required field there) —
    `None` for every `user`/tool-result turn and for a synthetic closing record with no real model call
    behind it (`Message::error`'s bare case), `Some` even on an aborted/error-tagged turn that still
    carries real content. Session-level totals answer "how much has this session cost so far"; this field
    answers "how much did _this_ turn cost" without needing to diff two running totals — the data a
    session-persistence layer (`crates/agent`, a later change) needs to actually exist before it can save
    per-message cost.
- **Lifecycle events** — `AgentEvent` adds `AgentStart`/`TurnStart`/`Steered`/`AgentEnd`/`CompactionStart`/
  `Compacted`/`Retry`/`Error`. `CompactionStart` (carrying the same `CompactionReason` `Compacted` does) fires
  once `Agent::compact` confirms a worthwhile prefix exists, right before its first summarization model
  call — a caller tracking "is a compaction currently in flight" (pi's `isCompacting`, surfaced by
  `crates/agent`'s `serve.rs`) sets a flag here and clears it on `Compacted`, since `compact`'s own
  summarization call passes a discarding inner sink and so nothing else can arrive in between.
  `Compacted` also carries `summary` (the generated text actually spliced in, file-operations list
  already appended — pi's own `CompactionResult.summary`) and `tokens_after` (pi's own
  `estimatedTokensAfter`: the post-compaction prompt in real tokens, system prompt and tools included —
  the live size, less the real size of the summarized prefix, plus the summary message's estimate;
  0.99–1.01× the next request's real prompt at the median, where the old chars/4 sum of the messages was
  2.0–2.6× low on Claude and left out the ~3.3K system prompt and tools); `crates/agent`'s `serve.rs`
  threads both straight into the `compact` RPC's own response, which previously exposed neither.
- **Codex live WebSocket transport** — [`codex_websocket`](src/codex_websocket.rs) gives a
  `RouteOverride::Prefixed`-routed (OpenAI-Codex-Responses/ChatGPT-subscription) request a live,
  persistent WebSocket alternative to the ordinary HTTP/SSE path, mirroring pi's real shipped
  `openai-codex-responses.ts`: one connection is reused across turns of the same session (keyed by
  `ModelRequest::cache_key`, the same stable id already sent as `prompt_cache_key`), and every turn
  after the first sends only the _delta_ of new conversation items via `previous_response_id` instead
  of resending the whole transcript. Always attempted first for an eligible request (matching pi's
  `"auto"` transport default — no `"websocket"`/`"websocket-cached"`/`"sse"` mode selector, since
  nothing in this codebase needs to pin one explicitly yet), with a transparent, silent fallback to the
  existing HTTP/SSE path — unchanged, still the correctness-preserving default — on any connect
  failure, send failure, or pre-first-event transport error; a session sticks to SSE for the rest of
  the process once its WebSocket has genuinely failed once (pi's own `websocketSseFallbackSessions`).
  `GatewayClient::with_codex_websocket(false)` is the (currently CLI-unwired) escape hatch back to
  HTTP/SSE-only behavior. Reuses `dialect::openai_responses::build_body`/`Decoder` entirely rather than
  duplicating the wire shape — this module only adds the connection lifecycle and the delta-diffing
  logic on top. See that module's own doc comment for the full design, including where it deliberately
  diverges from pi (no proactive idle-connection sweep timer; the delta baseline is built by harvesting
  each turn's own `response.output_item.done` wire events directly, not by reconstructing them from a
  live `AssistantMessage` object the transport layer here doesn't hold).

## What this crate is not

It ships **zero concrete `Tool` implementations** — `read`/`write`/`edit`/`bash` and the rest of the
coding tools all live in `crates/agent`. It has **no dependency on the gateway crate** — `GatewayClient`'s entire
contract is "POST dialect JSON to a base URL, get SSE back"; routing, provider auth, and metering are
the gateway's job. (It does depend on the `providers` crate, which sits _below_ both: a table of static
per-provider facts — host, wire format, auth header, env var — that the gateway routes on and this crate
uses to pick a request dialect. That is a shared table, not a dependency on the gateway.) This split is what lets the loop, the dialect adapters, and the tool dispatch logic
run as pure unit tests with `MockTransport` — no network, no live model, no gateway binary.

It is **not a plugin host**. pi has a real extension system (`core/extensions/`): dynamically loaded
third-party code that subscribes to typed lifecycle events (including `session_compact`) and can call
back into session actions (`navigateTree`, `fork`, `newSession`, …). This crate's equivalent seams —
`AgentHooks` (tool-call gate/rewrite) and `CheckpointHook` (mid-run persistence points; `checkpoint` returns whether the snapshot is durable, and the one before a tool dispatch **fails closed** — a failed write there ends the run with `Error::Checkpoint`, every call of that turn answered in memory as not run, and no tool runs; every other checkpoint is best-effort, logged and superseded by the host's end-of-run persist) — are Rust traits with exactly one implementation chosen by the embedder at compile time, not a runtime-loadable,
subscribable, third-party extension surface. Evaluated and deliberately **not** extended to cover
compaction or branch-navigation as hookable lifecycle events: nothing in this codebase dynamically
loads untrusted code, there is no extension marketplace or third-party-author story to serve, and
adding one would mean taking on loading/sandboxing/versioning machinery with no concrete consumer —
directly against this project's minimum-effective-abstraction bias. If a genuine need for third-party
extensibility ever appears, revisit this as a deliberate, scoped addition rather than bolting hooks on
piecemeal.

Two specific extension points pi's real product has and this crate deliberately doesn't: an ephemeral
per-request context transform (rewrite/prune the outbound message list for one call without persisting
the change — pi's `transformContext`, `packages/coding-agent/src/core/sdk.ts:351`), and a once-per-prompt
pre-flight hook fired after the user submits a fresh prompt but before the run starts, letting an
extension inspect/rewrite the raw prompt text, attached images, and assembled system prompt (pi's
`before_agent_start`, `packages/coding-agent/src/core/extensions/types.ts:675`). Both are real seams in
pi's _extension_ system specifically — they exist to let a runtime-loaded, third-party plugin reach into
the request pipeline before it's built. Since this crate has no such plugin system (see above) and no
first-party caller has ever needed either, adding them now would be exactly the same mistake: machinery
with no concrete consumer. If a real need appears — a caller that wants to inject transient context for
one call, or rewrite a prompt/system-prompt before a run starts, without persisting it — add the
narrowest seam that call actually needs then, not a general-purpose hook ahead of time.

(The before/after-provider-request seam — patch headers/timeout per call, mutate the raw wire payload,
observe raw response status/headers — used to belong in this "doesn't have" list too, but this crate
has since grown exactly that seam: see `AgentHooks::before_provider_request`/`before_provider_payload`/
`after_provider_response` in the "Hooks"/"Provider request/response hooks"/"Wire-payload hook" bullets
above and the `hooks.rs` row in the Package Structure table below.)

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
       │                                                        any complete tool_use blocks present?
       │                                                              │                              │
       │                                                             No                             Yes
       │                                                              │                              │
       │                                                          return Ok(())          sink(ToolStart) × N (call order)
       │                                                                                              │
       │                                            group by write_targets → buffer_unordered(≤8 groups) ◄── bounded; same-path serial
       │                                                                                              │
       │                                                    sink(ToolEnd) × N (finish order, as each call resolves)
       │                                                                                              │
       └──────────────────────────────── session.push(Message::tool_results([ToolResult; N])) ◄──── one `user` turn
                                                          (loop back to ModelRequest)

Error exits (no session mutation for the failed turn):
  cancel tripped              ──► Err(Cancelled)                [user abort; mid-stream or mid-tool]
  call's steps >= limit       ──► Err(MaxSteps)                [opt-in ceiling only — unset by default;
                                                                 per call, not `session.steps`; limit is
                                                                 `max_steps` (+ `WRAP_UP_STEPS` once the
                                                                 wrap-up fired); checked before the request]
  stream() / stream item Err  ──► Err(Transport(..))            [network/non-2xx (after retries), bad UTF-8/SSE,
                                                                 mid-stream `event: error`, truncated stream]
  pre-dispatch checkpoint Err ──► Err(Checkpoint(..))           [the `tool_use` turn could not be made durable;
                                                                 no tool runs, each call answered as not run]

**Wrap-up at the step ceiling (`step_budget.rs`).** With a `max_steps` ceiling, when a tool-results
turn uses the last allowed step, the per-call `limit` grows by `WRAP_UP_STEPS` (5) and the same turn
gets a block telling the model to stop cleanly: no new code and no finishing a change in progress, fix
or revert only a file that no longer parses or imports, run the test suite once, summarize. It fires
at most once per call and is reported as `AgentEvent::WrapUp { steps }`. Folding into the tool-results
turn keeps role alternation valid, the same way the steer lane does. A model that stops within the grant
ends the run normally; one that uses it up gets `Error::MaxSteps(max_steps)`, still the _configured_
ceiling. Without a ceiling the loop is unchanged.

Why: in the longctx eval, gpt-6-luna hit `MaxSteps(150)` in 7 of 22 code runs, each within two steps of
an untested edit, and two of those scored 0% because an edit had left the tree broken. With the wrap-up
(Luna, sqlparse and marshmallow at `--max-steps 60`, 3 runs a side) all 6 runs ended with an honest
summary against 6 silent cut-offs, no tree ended broken on either side, and cost was equal. The grant
made no source edit in 5 of the 6 runs and one targeted fix in the sixth, so the fixed fraction it can
move is its own; the arms' means (0.634 against 0.679) differ by what happened before the ceiling. The
first wording ("fix or revert anything broken, start no new work") let one run keep implementing during
the grant and end with 171 regressions; the current text forbids continuing the task outright.

Two sibling arms were deleted. A one-time "N of M steps remain" notice near the ceiling made Luna rush
more, and one run broke mid-edit (220 regressions). A continue nudge when the model stopped with 20%+ of
its steps left made runs 274% costlier and ended with broken trees.

**Background work (`background.rs`).** A tool whose work outlives its call (the `agent` crate's `bash`
jobs) returns a `Background` from `Tool::background`; the loop collects them from the registry at each
use (deduped by pointer, so two tools over one table report it once) and asks three things. After a
tool batch, `take_notices` (finish notices for work that ended unattended, each delivered once) folds
into the tool-results turn after any steer, as the wrap-up does. At a would-stop, pending notices are
delivered first — pushed as a user turn after any drained follow-ups, checkpointed, and the run
continues — so a run never ends without a result that already arrived; with none, `stop_reminder`
lists work still running that no earlier reminder covered, once per piece of work and at most once
per run, so a model that ends again ends. A pending stop request still wins over both, and so does the step limit: once
the run has used its steps (or the wrap-up's extended ones), neither is delivered — the step they
would start is one the run cannot take, and a clean final answer used to become a `MaxSteps`
failure. The run ends `Ok`; notices stay unconsumed for the next run. Each delivery is reported as
`AgentEvent::Background { reason: finished | still_running, text, wake }` (`wake` is for a host that
starts a turn itself to deliver a notice). Every text starts with one of `background::NOTE_PREFIXES`
(`[job `, `[background jobs]`, `[step budget]`), which `merge_user_requests` skips: a run of harness
notes would otherwise push the user's own words out of the four carried requests. `carry` is the third
question: see the `<background-jobs>` block under the deterministic-carry channel.

Recoverable (no error): a streamed tool call whose JSON args never parse keeps its `tool_use` block
(with `close_incomplete_json`'s partial object, or `{}`), never runs, and is fed back as an error `tool_result` the model can correct — the run
continues rather than aborting.

**Sink `Send` bound.** The event sink is bounded `FnMut(AgentEvent) + Send` (and `FnMut(&StreamEvent) +
Send` for the stream-only entry points). The `+ Send` is what lets a *nested* agent run inside a
`Tool::run` future — the `agent` crate's `subagent` tool drives a child `Agent` and awaits it directly,
which requires the run future to be `Send`, which requires the sink to be. The loop's internal helpers
hold the sink as `&mut (dyn FnMut(AgentEvent) + Send)` across `.await` for the same reason. Every
existing sink (stdout writers, `serve`'s mpsc push, test collectors) is already `Send`. The `serve`
daemon relies on that bound: `serve_session` is itself `Send`, so the WebSocket supervisor
`tokio::spawn`s each session onto the process-wide runtime instead of dedicating an OS thread and a
current-thread executor per tenant.

Dispatch gates on the presence of complete `tool_use` blocks alone (pi-parity fix) — never on
`stop_reason` for any *other* value, which the diagram above no longer branches on. A `MaxTokens`-
truncated turn that already emitted one or more complete tool calls before running out of room (the
model calls two tools, then starts trailing commentary that gets cut off) used to be silently treated
as "done, no tools to run" whenever `stop_reason != ToolUse`, dropping the calls the model actually
made. See `agent.rs::tests::tool_call_dispatches_even_when_stop_reason_is_max_tokens`.

The call the length stop cut off never runs. Every dialect's length stop (Anthropic `max_tokens`, Chat
Completions `length`, Responses `incomplete` for `max_output_tokens`, all `StopReason::MaxTokens`)
closes whatever block was open when the output limit hit, so a `write`/`edit` cut off mid-`content`
arrives as a closed block with an incomplete input. `Accumulator::finish` puts every call in a
`MaxTokens` turn whose input didn't parse as one whole JSON value (or that has no input bytes and is
the turn's last block) in `Turn::not_run` with `CUT_OFF_TOOL_CALL` as its error result: "not run:
the response hit the output token limit while this tool call was being written, so its input was cut
off. Re-issue it with less input per call — for example, split a large write or edit into several
smaller ones." Running a closed-up truncated input would write the truncated content with
`is_error: false`, so the model never learns and the file is silently wrong. Whole calls earlier in
the same turn still run: a whole input parsed as one JSON value, which needs the model's own closing
`}`, so nothing the cut drops afterwards can belong to it. pi's `agent-loop.ts` refuses every call on
a length stop because it can't tell whole from cut; we can, so refusing the whole ones would only throw
away work the model then repeats. Responses' `incomplete` for `content_filter` maps to `Refusal`, not
`MaxTokens`, the same as Chat Completions' `content_filter` — "resend smaller" is the wrong advice
there. See `agent.rs::tests::every_dialect_s_length_stop_refuses_the_cut_off_call_and_runs_the_whole_one`.

The one exception to dispatching whole calls is `StopReason::Refusal`, checked unconditionally *before*
`calls` is even collected (a
separate pi-parity fix — a refusal blocking dispatch is dialect-agnostic, matching pi's own
unconditional "error"/"aborted" return in `agent-loop.ts` ahead of any look at `message.content`): a
turn that streamed one or more complete `tool_use` blocks before being cut off with a refusal must not
dispatch them, since checking `calls.is_empty()` first would see a non-empty batch and run tools the
model was ultimately blocked from continuing. See `agent.rs::tests::
refusal_blocks_dispatch_even_when_a_tool_call_already_streamed`.
```

### Bytes to `StreamEvent` (`GatewayClient` + `dialect`)

```
TCP chunks (Bytes) ──► LineFramer (BytesMut) ──split on '\n'──► whole UTF-8 line ──► push_sse_line()
                                                                                          │
                                                                     strip "data:" prefix, buffer payload
                                                                     in SseEventBuffer; skip comment/
                                                                     `event:`/`[DONE]` lines outright
                                                                                          │
                                                              blank line (or end-of-stream) flushes the
                                                              buffer: join buffered payloads with "\n"
                                                                                          │
                                                                    serde_json::from_str(payload) → Value
                                                                                          │
                                                                    Dialect::Decoder::push(&Value)
                                                                                          │
                                                                          0..N StreamEvent
```

A `data:` line is buffered, not parsed immediately — the SSE spec's own event boundary is a blank
line, not a per-`data:`-line one, so a logical event can (rarely, and not from real Anthropic/OpenAI in
practice, but legitimately from a spec-conformant intermediary) span multiple consecutive `data:` lines
that must be joined with `"\n"` before the first JSON-parse attempt. `SseEventBuffer` accumulates them;
a blank line or end-of-stream (the streaming client and `decode_sse` both flush once after their last
line) triggers the join-and-parse. See `dialect/mod.rs`'s `SseEventBuffer`/`push_sse_line` doc comments.

Anthropic emits an explicit `content_block_stop` per block; OpenAI doesn't, so its decoder synthesizes
`ContentBlockStop` when a tool call's `id` arrives (closing the prior block) or `finish_reason` shows
up, and defers `MessageStop` to `Decoder::finish()` so it lands after the trailing usage-only chunk.
Its tool-call deltas route to an open call by wire `index`, then by `id`; a delta carrying a new `id`
together with a `name` opens a new call even on an index already in use (a provider that numbers
every parallel call 0), while a changed `id` on an argument-only fragment (Kimi) stays with its call.
All three dialect decoders produce the identical `StreamEvent` sequence shape — the loop's
`Accumulator` is dialect-blind. Anthropic's own wire already carries a real per-block `index` on
`content_block_start`/`_delta`/`_stop`, read straight through rather than discarded (defensive, since
Anthropic's documented behavior never actually interleaves blocks in practice — but the decoder no
longer has to _assume_ strict sequential delivery to stay correct if that ever changed).

## Concepts & Terminology

| Term                             | What It Controls                                                                                                                                                                                                                                                                          | NOT                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `Dialect`                        | Which wire shape (`/v1/messages`, `/v1/chat/completions`, or `/v1/responses`) a model id maps to, via `ApiKind`                                                                                                                                                                           | Not which _provider_ serves the request — that's gateway routing on the virtual key                                              |
| `ModelTransport`                 | The loop's only network seam; turns a `ModelRequest` into an `EventStream`                                                                                                                                                                                                                | Not the gateway client specifically — `MockTransport` is the other implementor                                                   |
| `Session`                        | One run's message history + step/token counters, Arc-shared, serde round-trips                                                                                                                                                                                                            | Not multi-session storage — one `Session` is one conversation                                                                    |
| `ToolRegistry`                   | Name → `Arc<dyn Tool>` lookup the loop dispatches against                                                                                                                                                                                                                                 | Not a permission system itself — gating is the `AgentHooks::before_tool_call` seam                                               |
| `AgentHooks`                     | Interception around each tool call and turn: `before_tool_call` (block) / `after_tool_call` (rewrite) / `on_assistant_message` (rewrite the model's own generated content)                                                                                                                | Not a sandbox — it decides per call; defaults to `NoHooks`                                                                       |
| `ToolError`                      | A tool's own failure → an error `tool_result` fed back to the model                                                                                                                                                                                                                       | Not a loop-aborting error — the run continues                                                                                    |
| `ToolOutput`                     | A tool's success value: `text` + `images` (multimodal) + a `terminate` hint                                                                                                                                                                                                               | Not just a string — `String`/`&str` convert in, and `terminate` ends the run only when every call in the batch agrees            |
| `ModelCaps`                      | Per-model wire knobs from `capabilities(model)`: max-tokens field, long-cache, vision, thinking shape, reasoning-effort, context window                                                                                                                                                   | Not a model catalog or pricing/routing table — the gateway routes and meters; this is the smallest table the wire decisions need |
| `ToolChoice` / `ReasoningEffort` | How the model may use tools this turn / its effort level — optional `ModelRequest` fields mapped per dialect                                                                                                                                                                              | Unset emits nothing on the wire (provider default), so the default request shape is unchanged                                    |
| `Error`                          | A loop/transport failure → `run`/`run_events` returns `Err`, the in-flight turn is discarded                                                                                                                                                                                              | `Cancelled` is a user abort, not a fault; malformed tool args are recoverable, not an error                                      |
| `StreamEvent`                    | The normalized unit all three dialect decoders emit; every block-scoped variant carries an `index`, what `Accumulator` folds (text/thinking/tool/usage) — more than one index can be concurrently open                                                                                    | Not the wire format — it's the post-translation internal shape                                                                   |
| `ContentBlock`                   | One piece of a `Message` (`Text`/`Thinking`/`RedactedThinking`/`ToolUse`/`ToolResult`/`Image`)                                                                                                                                                                                            | Not a streaming unit — it's the assembled, turn-final form                                                                       |
| `AgentEvent`                     | The full observation surface (`AgentStart`/`TurnStart`/`Stream`/`ToolStart`/`ToolProgress`/`ToolUsage`/`ToolEnd`/`TurnEnd`/`Steered`/`WrapUp`/`Background`/`ToolOutputMasked`/`CompactionStart`/`Compacted`/`CompactionFailed`/`ModelSwitched`/`ToolsUpdated`/`Retry`/`AgentEnd`/`Error`) | Not exposed by `Agent::run` — that filters to `Stream` only                                                                      |
| `max_steps`                      | Opt-in loop-iteration ceiling (`None`/unbounded by default); one step = one model turn (tool dispatch doesn't increment it again)                                                                                                                                                         | Not a token or wall-clock budget; not set unless a caller opts in                                                                |

## Core Mechanism

### Accumulating a turn (`agent.rs::Accumulator`)

`Accumulator` folds a `StreamEvent` sequence into `Vec<ContentBlock>` + stop reason + token counts.
Every block-scoped event carries an `index`, and **more than one index can be open at once** — a
dialect whose wire genuinely interleaves multiple blocks (OpenAI Responses, when the model streams two
tool calls' arguments concurrently) reports them as such: `Accumulator.open: HashMap<usize, OpenBlock>`
tracks each index's own accruing state (`OpenBlock::Text(text, id, phase)` — `id`/`phase` are OpenAI
Responses' replay metadata, threaded through from a `TextFinal` resync, `None` from every other dialect
— / `Thinking(text, signature)` / `Tool(id, name, json-buffer)`) completely independently — a
`ToolUseStart` at one index no longer
force-closes whatever's open at a different index. `ContentBlockStop{index}` finalizes just that one
index (parsing its buffered JSON for a tool call, or `{}` if empty) into `Accumulator.done: HashMap<usize,
Option<ContentBlock>>`; a finalized index sits there until every _earlier-declared_ index (tracked in
declaration order by `Accumulator.order`) has also finalized, at which point the run of
consecutively-ready entries flushes into `blocks` in declaration order (`try_flush`) — so the assembled
message never reorders blocks relative to when the model announced them, even though a later-declared
block may finish streaming its content first. A dialect whose wire never interleaves (Anthropic in
practice, OpenAI Chat Completions' text) just always uses index 0, so exactly one index is ever open and
this degenerates to the single-current-block behavior every dialect used before this design. Parsing a
non-empty tool-argument buffer tries the raw buffer as-is, then `repair_json` (fixes mis-escaping — a
raw control byte or stray backslash inside a string, not a structural problem). A buffer that parses
neither way never runs. Its `ToolUse` block records what `close_incomplete_json` recovers (the buffer
with its open string/`{`/`[` closed — the partial object the model actually wrote) or a wire-valid `{}`
when even that doesn't parse, and `finish` settles the call once the stop reason is known: in a
`MaxTokens` turn it is a cut-off call (see the dispatch section above); otherwise the model itself
emitted JSON that doesn't parse. Either way it lands in `Turn::not_run` with its error `tool_result`
(`CUT_OFF_TOOL_CALL`, or `invalid_tool_input_error`'s diagnosis below), which the gate feeds back so
the model can correct it, rather than aborting the run. An empty buffer is `{}`, the whole input of a
tool that takes none, unless it is the last block of a `MaxTokens` turn.

`invalid_tool_input_error` says what was wrong, where, and how to resend — never the whole input (a
broken `write` can be tens of kilobytes, and the model has it in its own `tool_use` block). It names
serde's error, then quotes at most 60 bytes before and 40 bytes from the failure point, cut from the
input after `repair_json` (the text the parse actually failed on), single-line. A failure that is the
`>` right after a quoted name in key position — Haiku's XML parameter syntax leaking into JSON as
`"new_string">RAW`, all 89 recorded Haiku edit failures — gets its own hint:

> not run: the tool call's input is not valid JSON (expected `:` at line 1 column 25). Parsing failed
> after `{"edits": [{"new_string"`, at `>        try:\n            value = getter…`. "new_string"> is
> XML parameter syntax; JSON needs "new_string": "…", with quotes, backslashes and newlines inside the
> value escaped (\", \\, \n). Resend the whole call as JSON.

An input that ends early is told "It ends with `…` before the JSON is complete. Resend the whole call
as one complete JSON object."; anything else gets the excerpt and "Resend the whole call as one JSON
object, with quotes, backslashes and newlines inside string values escaped (\", \\, \n)." The input is
never repaired into a call that runs: a guessed `new_string` can be silently wrong, while a resend
costs one turn (retries already recovered 72 of 88 recorded failures with the old, bare message).

Edge case: a stream that started but closed before its terminal event (Anthropic `message_stop`, a
Chat Completions `finish_reason`, a Responses terminal `response.*` event) is not a completion: each
decoder's `finish` returns `Error::Transport("… stream ended before …")`, which `run_turn` retries as
truncation (`is_retryable_mid_stream`). Only a body that never started at all (no Anthropic
`message_start`, no JSON event on the OpenAI dialects) still finishes cleanly: `stop_reason` keeps its
`StopReason::EndTurn` default, the turn has no blocks, and usage is `0`/`0`.

### Concurrent tool dispatch

Once an assistant turn's `tool_uses()` are collected, the calls are dispatched concurrently — but with
two guards, so concurrency never costs correctness:

- **Same-path calls are serialized.** Each call is grouped by `Tool::write_targets(input)` (the paths
  it would mutate — `write_target`'s one path by default, every file of a multi-file `edit` — or a
  group of its own for read-only/path-less calls). Calls sharing **any** target run _sequentially in
  call order_ within their group (`group_by_write_targets`; a multi-file call joins every group it
  touches), because two tools that read-modify-write the same file (the model batching two `edit`s on
  one source) would otherwise race and drop or interleave a write. Distinct groups still run
  concurrently against each other.
- **Concurrency is bounded.** Groups are run through `futures::stream::buffer_unordered` with a cap of
  `MAX_CONCURRENT_TOOL_GROUPS` (8), so a turn requesting dozens of `bash`/`grep` calls can't spawn that
  many subprocesses / parallel walks at once (`grep` itself fans out over CPU cores, which would
  compound). `buffer_unordered` is safe despite its name: each group yields results tagged with their
  original call index `i`, scattered into a pre-sized `results[i]` — cross-group completion order never
  reaches the transcript. A single `conservative_exclusive` call anywhere in the turn (`bash`'s own
  override — its mutation scope can't be named by `write_target`) forces the whole turn's concurrency to
  1 regardless of the cap; `Agent::with_sequential_tools(true)` forces that same concurrency-1 dispatch
  unconditionally, host-selected rather than inferred from the calls themselves (a deterministic-repro
  debugging session, or a host policy that never wants two tool calls actually overlapping) — matching
  pi's own `AgentOptions.toolExecution: "sequential" | "parallel"`.

This default split is always **gate the whole batch, then execute** — every call's argument coercion
runs, then its `before_tool_call` resolves, in call order, before any call's actual tool execution
begins (`Agent::run_events_steered`'s phase 1/phase 2, matching pi's `prepareToolCall` loop ahead of
`executeToolCallsParallel`'s `Promise.all`). The tool lookup itself runs _first_, ahead of both
coercion and the hook (pi-parity fix — matches pi's own `prepareToolCall`, which returns its
`Tool ${name} not found` immediate outcome before `prepareToolCallArguments`/`validateToolArguments`/
`config.beforeToolCall` ever run): a name naming no registered tool short-circuits straight to the
same "unknown tool: {name}" error result the execution phase would otherwise only discover later,
without ever invoking `before_tool_call` for a call that was never going to run anyway. Coercion runs
_before_ the hook, not just before the tool itself (pi-parity fix — matches pi's own `prepareToolCall`,
which calls `validateToolArguments` ahead of `config.beforeToolCall`): a permission hook must see the
same coerced/typed arguments the tool is about to run with, not the model's raw, possibly stringified
wire values. Phase 1 re-checks
cancellation twice per call, not once: at the top of its loop iteration, _and_ again right after that
call's own `before_tool_call` hook returns — a slow permission-check hook can observe cancellation
firing mid-await, and without the second check a single-call (or last-in-batch) turn had no _later_
iteration to catch it, so the call was marked ready and phase 2 dispatched it for real despite the run
already having been cancelled (pi-parity fix). A per-tool override changes this default entirely: any
call in the batch naming a tool
whose `Tool::execution_mode()` returns `Some(ToolExecutionMode::Sequential)` routes the turn's _whole_
batch through `Agent::run_tool_calls_interleaved` instead — a fully-interleaved
gate→execute→finalize-per-call path (pi's `executeToolCallsSequential`) where call 1 is completely
resolved, including its `after_tool_call` rewrite, before call 2's own gate even starts. This is the
seam a concurrency-aware policy (a permission hook reasoning about "what's already run", a rate
limiter) needs and the default split structurally can't offer; `ToolStart` is still emitted upfront for
the whole batch either way (not duplicated by the interleaved path), so the two paths differ only in
when gating/execution/`after_tool_call` happen relative to each other, not in the observable
`ToolStart` ordering.

Because that up-front `ToolStart` covers the whole batch, both paths owe every call a matching
`ToolEnd` — including a call resolved at its _gate_ without ever running (malformed streamed arguments,
an unregistered tool name, a `before_tool_call` block). The default path gets this structurally (its
`ToolUpdate::End` send sits below the gate/execute match, so an `Immediate` outcome passes through it);
the interleaved path emits one explicitly in each of its three gate-resolution arms. A missing end
there is not cosmetic — a client pairing the two events renders the call as still running forever, even
though the run has moved on and the model already received its error result.

The transcript stays deterministic regardless of which tool finishes first: every `ToolStart` is sunk
_before_ dispatch (in call order), and every `ToolResult` block is rebuilt in call order _after_ the
groups resolve — so the wall-clock savings never leak into transcript ordering. Only `ToolEnd` events
follow finish order, sent the moment each call's result is known (see `Streaming tool progress`).

All of a turn's results are folded into **one** `Message::tool_results([...])` user turn, not one
message per result (`Message::tool_results`, `message.rs`). Anthropic carries a turn's tool results as multiple blocks on
a single `user` turn and rejects two consecutive same-role messages, so N separate `user` messages
would 400 the next request whenever the model batched more than one tool call.

Each call resolves to `(text, images, is_error, terminate)`: a tool's `ToolOutput` images ride onto
its `ContentBlock::ToolResult` so the multimodal model sees them, and if _every_ call in the batch set
`terminate` the loop ends the run after recording the results — an `attempt_completion`/`exit`-style
tool, gated so one tool can't cut off the others dispatched alongside it.

`crates/agent`'s `structured_output` is that mechanism's one production consumer, and its shape is what
callers have to reason about. The error path can never terminate (a `ToolError` resolves to
`terminate: false`), so an invalid payload becomes an error `tool_result` the model retries against —
that _is_ the retry loop, with no extra machinery. And because the unanimous-agreement rule lets a
mixed batch continue, a `structured_output` call dispatched alongside an `edit` stages its value and the
run goes on; a host must therefore read its result only once the run has fully drained, since a later
call may revise it. Neither behavior is special-cased for the tool — both fall out of the fold.

Dispatch also stamps one schema-undocumented key onto every call's coerced input:
`tool::MODEL_SUPPORTS_VISION_KEY` (`_model_supports_vision`), which `read` uses to decide whether to
downgrade an image to a text placeholder. Every other tool ignores extra keys — except one that cannot:
a tool validating its input against a caller-supplied JSON Schema must strip the key first, or a schema
with `"additionalProperties": false` would reject every call the loop ever makes to it.

Any **steer** messages a
client queued mid-run (`Steering::push_steer`) are folded onto this same tool-results user turn as
trailing text blocks, letting a client redirect a busy agent between tool turns while keeping role
alternation valid; **follow-ups** (`push`) are a separate lane, injected only at the stop boundary.

### Session history sharing

`Session.messages` is `Arc<Vec<Message>>`. `Session::push` mutates via `Arc::make_mut` — in place when
the session solely owns the `Arc` (the steady state between turns), cloning only if another snapshot
still holds the same pointer (an in-flight `ModelRequest`, a serve catch-up `OutFanout` history, or a
queued checkpoint `Arc`). When that clone does happen, it's cheap by construction: the large immutable
`ContentBlock` payloads (`Text`/`Thinking` text, `RedactedThinking`/`ToolResult`/`ImageSource` bodies —
the multi-KB tool results and base64 images) are `Arc<str>`, so `make_mut`'s per-message deep clone is a
refcount bump per payload, not a byte copy — only the small `String` fields (ids, names, signatures) and
`ToolUse.input` (`serde_json::Value`) are actually re-allocated. `ModelRequest::messages` and `Agent::tool_defs` are
both `Arc`-shared for the same reason: building a request clones a pointer, not a deep copy of a
history that grows every step (an O(n²) cost over a long run otherwise) or a tool-definition list with
embedded JSON Schemas. `tool_defs` specifically is computed once in `Agent::with_tools`, not rebuilt
per turn. See `agent.rs:tests::request_snapshots_are_isolated_across_turns` for the isolation guarantee
this depends on: an in-flight request's message snapshot must not retroactively see a later turn's
appends.

The copy that _does_ scale with the transcript is the provider wire body: every dialect's `build_body`
walks the history into a `serde_json::Value` tree (owned `String`s, `BTreeMap` objects).
`GatewayClient::stream` serializes that tree to compact JSON `Bytes` immediately after the payload hook
and drops it before the SSE stream is built, so the tree does not sit alongside the response for the
whole turn. Retries reuse those bytes. The Codex WebSocket path still needs the `Value` long enough to
delta-diff (and parks a `Vec<Value>` baseline on a cached connection between turns). Serve catch-up
history is the same `Arc<Vec<Message>>` — it is serialized to JSON only once per attach, not per turn.

### SSE byte framing

`GatewayClient::stream` frames the chunked body through `LineFramer`, which buffers raw _bytes_ (a
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
[check steps]──steps≥max_steps──► Err(MaxSteps)  [only when a ceiling was opted in]
      │ steps<max_steps (or no ceiling set — the default)                  │
      ▼                                                                    │
[request built] ──stream()/stream item Err──► Err(Transport) (after retries) │
      │ stream exhausts cleanly                                            │
      ▼                                                                    │
[turn assembled] (malformed tool args → recoverable error result, not fatal) │
      │                                                                    │
      ▼                                                                    │
[pushed to session, steps+=1, TurnEnd sunk]                                │
      │                                                                    │
      ├──no complete tool_use blocks──► Ok(()) [done]                     │
      │                                                                    │
      └──complete tool_use blocks present──► [dispatch tools: grouped, bounded] ┘
            ToolStart×N → group by write_targets → buffer_unordered(≤8) → ToolEnd×N (finish order) + tool_results pushed
```

| From              | Event                                   | To                    | Guard                                                                                                                       | What Actually Happens                                                                                                                                                                                                                                                                                                             |
| ----------------- | --------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (loop top)        | iteration begins                        | Err(MaxSteps)         | a `max_steps` ceiling was opted in and this call's steps `>=` it (plus `WRAP_UP_STEPS` once the wrap-up fired)              | No request sent; session unchanged; an `Error` event is sunk                                                                                                                                                                                                                                                                      |
| (loop top)        | iteration begins                        | request built         | no ceiling set (default), or `steps < max_steps`                                                                            | `ModelRequest` cloned (Arc pointers) from session + cached tool defs                                                                                                                                                                                                                                                              |
| request built     | `transport.stream()` / stream item      | Err(Transport)        | network/HTTP/decode err (after retries), mid-stream `event: error`, truncated stream                                        | An `Error` event is sunk; if real content had already streamed before the failure, it's kept — pushed as an assistant turn tagged `error_message` (not discarded) — otherwise a bare closing record; error returned from `run`/`run_events`                                                                                       |
| request built     | stream exhausts                         | turn assembled        | always                                                                                                                      | `Accumulator::finish()` returns `Turn`; malformed tool args and a call the length stop cut off become recoverable error `tool_result`s (`Turn::not_run`), never run                                                                                                                                                               |
| any await point   | `cancel` tripped                        | Err(Cancelled)        | client abort                                                                                                                | Stream/tool futures dropped (HTTP + subprocess killed); no `Error` event (not a fault). If nothing had streamed yet (pre-connect, or the top-of-loop check), the pending `user` turn is still closed out with an aborted assistant record — not left dangling — so a later prompt can't stack a second consecutive `user` message |
| turn assembled    | —                                       | turn pushed           | —                                                                                                                           | `session.push(assistant)`, `record_usage`, `steps += 1`, `TurnEnd` sunk                                                                                                                                                                                                                                                           |
| turn pushed       | no complete `tool_use` blocks           | done (`Ok`)           | `calls.is_empty()` — `stop_reason` is not consulted (pi-parity fix; see the turn-loop diagram above)                        | Returns to caller; session ends on the assistant turn                                                                                                                                                                                                                                                                             |
| turn pushed       | complete `tool_use` blocks present      | dispatching tools     | `!calls.is_empty()`, regardless of `stop_reason` — e.g. still dispatches under `MaxTokens` if the calls themselves finished | `ToolStart` sunk per call, in call order; the `tool_use` turn is checkpointed here too — _before_ any tool runs — so a crash mid-dispatch never loses the record of which calls the model asked for. A cut-off call is answered with an error at its gate, never run                                                              |
| dispatching tools | all groups resolve (`buffer_unordered`) | (loop top, next iter) | —                                                                                                                           | `ToolEnd` sunk per call as each resolves (finish order); one `tool_results` user message pushed, in call order (carrying any tool images + mid-run steer text); checkpointed again here; ends the run early instead if every call set `terminate`                                                                                 |

### Per-block accumulation (`Accumulator`)

| From          | Event              | To        | What Actually Happens                                                                                                                                                                                                                                                                                |
| ------------- | ------------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| none open     | `TextDelta`        | text open | Appends to the text buffer                                                                                                                                                                                                                                                                           |
| text open     | `TextDelta`        | text open | Appends                                                                                                                                                                                                                                                                                              |
| text open     | `ToolUseStart`     | tool open | Opens the tool at its own `index`; text open at another index stays open until its own `ContentBlockStop`                                                                                                                                                                                            |
| none open     | `ToolUseStart`     | tool open | Opens `(id, name, "")`                                                                                                                                                                                                                                                                               |
| tool open     | `InputJsonDelta`   | tool open | Appends to the JSON argument buffer                                                                                                                                                                                                                                                                  |
| text open     | `ContentBlockStop` | none open | Flushes text block                                                                                                                                                                                                                                                                                   |
| tool open     | `ContentBlockStop` | none open | Parses the JSON buffer (raw, then `repair_json`); on failure records the call as unsettled — `finish` puts it in `Turn::not_run`, as cut off under a `MaxTokens` stop, else as invalid JSON — and pushes `ToolUse` with `close_incomplete_json`'s partial object or a wire-valid `{}`; it never runs |
| thinking open | `ContentBlockStop` | none open | Flushes a `Thinking { text, signature }` block (from `ThinkingDelta`/`SignatureDelta`)                                                                                                                                                                                                               |

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
a pointer clone; `Arc::make_mut` on `push` keeps appends in-place in the common case (no live snapshot
still holds the old pointer) and falls back to a real clone only when one does — which is also what
_must_ happen for request-snapshot isolation: a request built from turn 3's history must not silently
see turn 4's appended tool results. The remaining per-turn copy is the dialect wire body: a
`serde_json::Value` of the transcript, serialized to bytes and dropped before the stream waits on the
model, rather than held as a tree for the whole turn.

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
is never killed mid-flight. When it does fire mid-body, the read error is tagged
`MID_STREAM_NETWORK_ERROR`, so `run_turn` retries the turn from a fresh connection like any other
dropped stream. `GatewayClient::with_idle_timeout` overrides it for a deployment that sits
behind a different upstream (rebuilds the underlying `reqwest::Client`, since its timeouts are fixed at
construction); outbound proxy config needs no client-side option at all — `reqwest` already reads
`HTTP_PROXY`/`HTTPS_PROXY` from the environment at the library level.

### Why the Anthropic body stamps three prompt-cache breakpoints

An agent loop re-sends an ever-growing prefix every turn — tools, then system, then the entire prior
conversation — in request order. Without prompt caching each turn re-bills that whole prefix at full
input-token price: an O(n²) token cost over an n-step run, on the very history this crate
already keeps O(1) _in memory_ via `Arc`/COW (the wire cost was the half left unoptimized).
`anthropic::build_body` marks `cache_control` (ephemeral; `ttl: "1h"` only when `cache_long` _and_
`capabilities(model).supports_long_cache` — an unsupported model silently falls back to the 5-minute
TTL instead of 400-ing; `GatewayClient::stream` has already cleared `cache_long` on a route not
documented to take the long TTL, see `client::long_cache_reaches` and the agent crate's "Prompt-cache
TTL by mode") at three points (Anthropic caches the request prefix up to each mark; cache
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
below it) _and_ on `!req.no_cache`: `openai.rs`'s Chat Completions dialect is the fallback for every
third-party OpenAI-compatible provider (`Dialect::for_model` routes native OpenAI ids to the Responses
dialect instead), and a strict-schema third-party endpoint can 400 the whole request over an
unrecognized field — `ModelCaps::unknown()`'s conservative default omits it unless a capability-table
entry opts in. `no_cache` skips `prompt_cache_key`/`prompt_cache_retention` on both OpenAI dialects the
same way it skips Anthropic's `cache_control` breakpoints — a one-off request has no follow-up turn to
route back to the same cache node, so the affinity hint is pointless.

### Why the Responses `prompt_cache_key` is a hash of the prefix, not the session id

OpenAI serves a cached prefix only under the `prompt_cache_key` that wrote it. Measured with the
same 3.5K-token prefix:

- **gpt-6-luna:** each new key wrote the whole prefix (`cache_write_tokens: 3523`, billed at 1.25×
  input on GPT-5.6+). A repeated key read it (`cached_tokens: 3523`).
- **gpt-5.4-nano:** two keys never shared, and a repeated key hit (`cached_tokens: 2816`).

A session-id key therefore re-wrote the instructions and tools on every new session's first request,
even when the setup was identical.

`openai_responses::prefix_cache_key` derives the key instead. It is a 64-bit FNV-1a hash of what
every session with the same setup sends first: the system prompt and the tool definitions. FNV-1a is
used because it is stable across builds, which `DefaultHasher` is not. The hash is seeded with
`ModelRequest::cache_scope`, a hash of the credential that `GatewayClient::stream` sets. Sessions under
one key share the prefix. Two keys never do, which means two tenants behind one gateway org cannot
probe each other's cache.

How the key is shaped depends on the model:

- **GPT-5.6 and later:** the key only partitions the cache, so it is `pfx-<hash>`.
- **Earlier models:** the key also routes, and OpenAI asks for about 15 requests a minute per key
  before overflow scatters the cache. One agent session can reach that rate alone. So the session id
  picks one of `ROUTING_BUCKETS` (4) keys per prefix: `pfx-<hash>-<0..3>`. A session keeps its bucket,
  and a fleet's load is split four ways.

Measured with the agent binary, fresh one-turn `run` sessions in a row, 3,149-token prefix:

| Model                    | Session-id key (before)            | Prefix key (after)                                  |
| ------------------------ | ---------------------------------- | --------------------------------------------------- |
| gpt-6-luna               | every session's first request cold | session 1 wrote 3,146; sessions 2–3 read 3,146 each |
| gpt-5.4-nano (4 buckets) | every session cold                 | sessions 1–3 cold; sessions 4–8 read 2,560 each     |

Codex and Copilot keep the session id. Both are OAuth routes into an org other users share, so a
prefix key would pool every user's cache, and their tokens rotate, so a credential scope would not
hold still. The decoder now also reads `input_tokens_details.cache_write_tokens` into
`TokenUsage::cache_write_tokens`. Before this, GPT-5.6+ cache writes were reported as plain input.

### Why malformed streamed tool arguments are recoverable, not fatal

A streamed tool call whose JSON args never parse is a protocol glitch, not a model mistake the model
can't recover from. Rather than abort the run, the loop keeps the `tool_use` block (with a wire-valid
object input so the next request doesn't 400) and feeds back an error `tool_result` naming the problem,
which the model corrects on the next turn — the same shape as any other recoverable tool error. It
never runs the call on a guessed repair: closing up an unclosed input is a guess about where a value
ended, and for a `write`/`edit` a wrong guess is a silently wrong file.

### Why dialect selection is a model-id prefix check, not configuration

The gateway relays bytes verbatim; it doesn't translate dialects. Per the "Beyond twist," provider
_selection_ (which upstream serves a request) is the gateway's job via the virtual key, but wire
_shape_ (`/v1/messages` vs `/v1/chat/completions` vs `/v1/responses`) is this crate's job, because the
gateway has no opinion on it. `Dialect::for_model` picks Anthropic for `claude*`/`*anthropic*`; for
everything else it consults `models::capabilities(model).api` (an `ApiKind`) — every native OpenAI id
(gpt-4/4.1/4o, the gpt-5 family, o-series) is `ApiKind::Responses` and gets `dialect/openai_responses.rs`
(`/v1/responses`); every third-party OpenAI-compatible id (OpenRouter, DeepSeek, Together, Cerebras,
xAI, Groq, Fireworks, Mistral) and any unrecognized id stays `ApiKind::ChatCompletions`. Adding a model
never requires touching the gateway. The name check is the fallback, not the first question: when
the serving provider is known, `Dialect::for_model_via_provider` asks its `providers::WireFormat`
first, so an OpenAI-wire aggregator (OpenRouter's `anthropic/claude-…`) gets the OpenAI wire, with
the model's `ApiKind` choosing Responses vs Chat Completions; `provider: None` is identical to
`for_model`. The Responses dialect reuses `ContentBlock::Thinking`'s `signature`
field for a reasoning item's _entire JSON-stringified content_ (not just a cryptographic signature the
way Anthropic's is) — the only way to replay OpenAI's encrypted reasoning across turns without a new
content-block variant; a signature that fails to parse as JSON (e.g. after `set_model` scrubbed a
cross-provider thinking history) degrades to plain text instead of erroring. A reasoning or message
item opens on its `output_item.added` (as an empty `ThinkingDelta`/`TextDelta`, declaring the block in
its true position), never on its first delta: a reasoning item with an empty summary — routine with
`summary:"auto"` — streams no delta at all, yet its `output_item.done` carries the `encrypted_content`
the next request must replay ahead of the function call it led to. An item that streamed no text gets
its text from `done` as one delta before the resync, so clients rendering deltas see it; a thinking
block that ends with neither text nor signature (a stream cut before the item's `done`) is dropped.

`ContentBlock::Text` similarly carries two OpenAI-Responses-only, otherwise-`None` fields: `id` (the
wire message id) and `phase` (`"commentary"` vs `"final_answer"` — a channel label OpenAI's docs say
must be preserved on replay for gpt-5.3-codex and later, or those models' quality measurably degrades).
Both are captured off `output_item.done`'s `item.id`/`item.phase` alongside the block's authoritative
text (`StreamEvent::TextFinal`'s own `id`/`phase` fields) and restamped verbatim by
`push_assistant_content`, which also always sets `status: "completed"` (the only status a _replayed_
block can have) and, when nothing was ever captured (a locally-authored block — a compaction/branch
summary — or a session persisted before this field existed), a deterministic `msg_{msg_index}` /
`msg_{msg_index}_{text_block_index}` fallback id via `fallback_message_id` — stable across rebuilds of
the same request so it doesn't churn the prompt-cache prefix turn to turn. Every other dialect leaves
both fields `None` and never reads them.

## Trust Boundaries

**What the system verifies (rejects if invalid):**

- HTTP status: a non-2xx response from the gateway is read as text and surfaced as
  `Error::Transport`, never silently treated as a successful empty stream.
- UTF-8 validity of each framed SSE line (`Error::Transport("invalid UTF-8 in SSE stream")`).
- JSON well-formedness of each SSE `data:` payload (`Error::Transport("malformed SSE json")`).
- JSON well-formedness of a streamed tool call's assembled arguments — a parse failure is _not_ fatal;
  it becomes a recoverable error `tool_result` (`Turn::not_run`).
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

| File                          | What It Does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lib.rs`                      | Crate root: module list, public re-exports, and the crate-wide `#![cfg_attr(test, allow(...))]` panic-free gate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `agent.rs`                    | `Agent` config + `run`/`run_cancellable`/`run_events`/`run_events_cancellable`/`run_events_steered` loop, `Accumulator`, concurrent tool dispatch (threading text/images/`terminate`), tool-driven termination, mid-run + stop-boundary steering, `with_reasoning_effort`, model-aware `new` defaults, hooks, auto-compaction + overflow retry                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `message.rs`                  | `Role`/`ContentBlock`(+ `Thinking`/`RedactedThinking`/`Image`; `ToolResult` carries optional `images`; `Text` carries optional `id`/`phase`, OpenAI Responses' replay metadata)/`Message`(+ `usage: Option<TokenUsage>`, the per-turn figure stamped on an assistant message; + `stop_reason: Option<StopReason>`, why that turn ended, persisted so it survives across runs/processes; + `prior_turn_tokens: Option<u32>`, the measured real size of the turn before an assistant message, persisted)/`ToolDef`/`StopReason`(+ `Refusal`)/`StreamEvent`/`TokenUsage`(+ `reasoning_tokens`) — the internal model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `token_size.rs`               | Real-token sizing: `TokenFamily` (Claude / GPT / Other, from the model id) and its measured constants, the calibrated per-message estimate (`message_tokens`), and `message_sizes` (measured turns exact, the rest estimated)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `compaction.rs`               | Context compaction: trigger, cut-point search (`find_cut`; `plan_cut`, which never declines while the prompt is over the trigger), summary-prompt build (`summary_request` takes an optional `custom_instructions`, appended as "Additional focus: …" — a manual compaction's client-supplied steering, matching pi's `generateSummary`; never applied to the split-turn prefix call, matching pi's `generateTurnPrefixSummary` not accepting one at all), file-op extraction, and incremental update (`previous_summary`/`SUMMARY_MARKER` fold-forward) — the network-free half of `Agent::compact`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `bash_writes.rs`              | `written_paths(command)`: the files a `bash` command writes, parsed from the command text alone (redirects, `tee`, `sed -i`, `perl -i`, `mv`/`cp` destinations, Python heredoc/`-c` `open(…, 'w')`/`write_text`), relative paths joined onto the last `cd`, `/dev`/`/proc`/`/tmp` left out — feeds compaction's `<modified-files>` provenance (`extract_bash_writes`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `rate_limit.rs`               | Rate-limit retry: `is_rate_limited`, `retry_hint` (reads `retry-after-ms`/`retry-after`/"try again in"/`x-ratelimit-reset-*` out of an error), `RateLimitRetry` (the per-turn time budget `Agent::run_turn` applies)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `step_budget.rs`              | The wrap-up at a `max_steps` ceiling: `WRAP_UP_STEPS` (5), the `[step budget]` note text, and the per-call `BudgetState` that grants the extra steps once and raises the loop's limit (see "Wrap-up at the step ceiling")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `background.rs`               | `Background` — what the loop asks a tool's work that outlives its call (`take_notices`, `stop_reminder`, `carry`, `resume`); collection from the registry, the `<background-jobs>` carry block (`CARRY_OPEN`/`NO_JOBS`), and the harness-note prefixes (`NOTE_PREFIXES`, `is_note`, `is_carry_block`, `is_process_state`) (see "Background work")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `models.rs`                   | `capabilities(model) -> ModelCaps`: minimal per-model wire table (max-tokens field, long-cache, vision, thinking shape, reasoning-effort, context window), matched by id prefix; consumed by the dialects and `Agent::new`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `hooks.rs`                    | `AgentHooks` interception trait (`before_tool_call`/`after_tool_call`/`should_stop_after_turn`/`on_assistant_message`/`before_provider_request`/`before_provider_payload`/`after_provider_response`, all cancellation-aware where a run is in flight) + `NoHooks` default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `steering.rs`                 | `Steering` — three shared queues of `SteeringMessage` (text + optional images): `push_steer` (mid-run, folded onto the tool-results turn), `push`/follow-up (injected at would-stop boundaries), `push_next_turn` (folded onto a fresh run's first prompt turn), each independently `QueueMode`-tunable (`set_steering_mode`/`set_follow_up_mode`); `pending_count` peeks the combined depth without draining; `request_stop`/`take_stop_requested` (graceful-stop flag), `request_model_switch`/`take_model_switch` (mid-run model/thinking retarget), `request_tool_set`/`take_tool_switch` (mid-run tool-set retarget — see the `Mid-run tool-set switching` capability), all consumed at the same turn boundary; `clear()`/`clear_run_scoped()` drop the relevant lanes/flags without returning them, for a session swap vs. a mere cancellation respectively                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `write_lock.rs`               | `WriteLockRegistry` — a process-scoped, path-keyed async-mutex map (`Agent::with_write_locks`) extending same-path write exclusivity across `Agent` rebuilds (a `set_model`/`set_thinking` rebuild, or multiple sessions sharing one registry), layered on top of the per-turn write-target grouping below; `WriteLockGuard`'s `Drop` evicts a key's entry once its refcount shows no other holder/waiter left (see the `Cross-run file-mutation exclusivity` capability above) so the map stays bounded to currently-contended paths, not every path ever locked                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `tool.rs`                     | `Tool` trait (`run -> ToolOutput`, optional streaming `run_streaming` + `ToolProgress` sink, optional `execution_mode() -> Option<ToolExecutionMode>` — `Some(Sequential)` routes a turn's whole batch through `Agent::run_tool_calls_interleaved`) + `ToolOutput { text, images, terminate }` + `ToolRegistry` (name-keyed `Arc<dyn Tool>` map, last-registration-wins, name-sorted `definitions`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `deferred.rs`                 | Deferred tools: callable by name but advertised only once a successful `tool_search` (`TOOL_SEARCH`) result names them — the transcript is the record of what is loaded. Each dialect shows a loaded tool the cheapest way it can: Anthropic Claude 4.5+ sends every deferred definition with `defer_loading: true` and expands `tool_reference` blocks (`native`), OpenAI Responses GPT-5.4+ sends an `additional_tools` item after the search output, everything else appends loaded definitions to `tools` (`with_loaded`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `validation.rs`               | `coerce_tool_arguments(schema, input)` — best-effort JSON-Schema-shaped type coercion (numeric string → number, `"true"`/`"false"` → bool, etc., never lossy int/float or non-canonical bool text) run on every dispatched call's args before `Tool::run`/`run_streaming` sees them (`agent.rs`'s dispatch, `.unwrap_or_else(\|_\| input.clone())` on failure — a tool that can't be coerced still reaches the tool's own, clearer validation error rather than a new failure path here); recurses into `properties`, schema-typed `additionalProperties` (a bare `true`/`false` is not a schema and is left alone), array `items` (both the tuple-per-index form and the single-schema-for-every-element form), and `allOf`/`anyOf`/`oneOf` composition (best-effort — a member that fails to coerce is skipped/falls back to the value untouched, never fails the surrounding call, unlike every other coercion path here, which is all-or-nothing per AJV's own `coerceTypes` — matches pi's `coerceWithJsonSchema`/`coerceWithUnionSchema`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `transport.rs`                | `ModelRequest` (system/tools/thinking/`reasoning_effort`/`tool_choice`/cache_key/cache_long/`user_id` — Anthropic's `metadata.user_id` abuse-detection hint, unset by default; `is_azure`/`is_copilot`, both `false` by default — `GatewayClient::stream` sets them from the credential's route (an `api-key` auth header or an Azure deployment name; a Copilot route), and the dialects, `deferred` and the per-route capability lookup read them), `ReasoningEffort` + `ToolChoice` enums, `ModelTransport` trait, `EventStream` alias. `tools` is fixed for a given _request_, but not necessarily the whole _run_ it belongs to — see `Mid-run tool-set switching` above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `client.rs`                   | `GatewayClient`: production `ModelTransport`; retry-with-backoff (`Retry-After` delta-seconds _or_ HTTP-date, `MAX_BACKOFF` 60s default, `with_max_backoff` override, `is_retryable_status` matching 429/500/502/503/504/524/529), chunked-SSE byte framing into whole UTF-8 lines; `with_extra_headers`/`with_hooks` (see `Transport resilience`/`Provider request/response hooks` above); sends both `session_id: <cache_key>` and `x-client-request-id: <cache_key>` for the OpenAI Responses dialect only, when a `cache_key` is set — connection-level session-affinity routing, distinct from `prompt_cache_key`'s cache-node affinity in the body, matching pi's `openai-responses.ts` (which sends both headers, not just one); `needs_interleaved_thinking_beta(caps)` gates the `interleaved-thinking-2025-05-14` beta header on the model's own thinking _shape_ alone (any non-`Adaptive` model, `Adaptive` skipped since it interleaves by default) — pi-parity fix: previously also required _this turn's own_ `req.thinking.is_some()`, so a request with thinking off mid-session dropped the header even though a `Budget`-shape model always benefits from it being present, matching pi's own default-on `interleavedThinking ?? true` gate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `tls.rs`                      | `ensure_provider()`: installs the process-wide rustls crypto provider (`ring`) once; called at every client-construction site, since reqwest's `rustls-no-provider` feature panics in `Client::builder().build()` with no process default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `dialect/mod.rs`              | `Dialect` enum (model-id → wire selection), `StreamDecoder` trait, SSE line-splitting (`push_sse_line`/`decode_sse`), `clamp_max_tokens_to_context` (shared by all three dialects), `ensure_non_empty_content` — pads a message's outbound copy (never the persisted session record) to a placeholder text block when it's whitespace-only/empty, _or_ (pi-parity fix) when the whole message is a `Message::aborted`/`with_error` assistant turn: replaying a dangling `Thinking` block or other partial content from a cancelled/errored turn is exactly Anthropic/OpenAI's own "reasoning with nothing following it"/incomplete-turn rejection, and once one turn hits it every later turn resends the identical dangling block and fails identically — matches pi's `transform-messages.ts:186-194`, which skips such a message outright when building the next request. Paired with `repair_orphaned_tool_use`, which gives any `tool_use` lacking an immediately-following `tool_result` a synthetic error one. **`GatewayClient::stream` must run `ensure_non_empty_content` FIRST**: it _removes_ content, so an aborted/errored turn carrying a fully-streamed `ToolUse` block (the ordinary shape after a cancel, or a mid-stream failure right as a tool call closed) would otherwise be paired with a synthetic `tool_result` and then have its `tool_use` deleted out from under it, leaving a dangling `tool_result` no provider accepts — and, since nothing pops an _aborted_ record from the session, re-derived from history on every later request, wedging the session permanently. The reverse dependency does not exist: the orphan repair only ever adds a `tool_result` to a user turn, which is never empty/aborted/errored                                                                                                                                                                                                                                                                                            |
| `dialect/anthropic.rs`        | `/v1/messages` body builder (three prompt-cache breakpoints, capability-gated 1h TTL, per-model thinking shape, `tool_choice`, tool-result image rewrite, `normalize_cross_model_tool_ids` — disallowed-character/length normalization of a foreign-model tool-call id at the point a request reaches the wire) + decoder (text/thinking/tool/cache-usage, reasoning-token breakout, `pause_turn`/refusal-explanation, `model_context_window_exceeded` read as the `max_tokens` length stop, in-band error + truncation detection)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `dialect/openai.rs`           | `/v1/chat/completions` body builder + decoder — real translation: flattened messages, string-encoded tool args, `image_url` data-URIs (user + fanned-out tool-result images), `max_completion_tokens` vs `max_tokens`, `reasoning_effort`, `tool_choice`, synthesized block-stop events. System-prompt role is `"developer"`/`"system"` per `dialect::openai_responses::instruction_role` (shared, not duplicated); `store:false` sent except for `models::is_non_standard_store_provider`'s denylist; Mistral ids get their `tool_calls[].id`/`tool_call_id` reshaped to exactly 9 alphanumeric characters (`MistralToolCallIdNormalizer`, request-scoped) since Mistral's real API rejects any other shape; `finish_reason:"end"` is a "stop" synonym and `model_length`/`max_tokens` are `length` synonyms (all `StopReason::MaxTokens`); `network_error` and `error` (a router's upstream failing mid-generation) end the stream as a retryable `Err`, never a clean finish; DeepSeek assistant replay backfills an empty `reasoning_content` when a turn carried no thinking. Content is frozen at the terminal event: once `finish_reason` lands, both `push` and the `try_fast_path` fast lane ignore any further text/reasoning/`tool_calls` payload, because decoding does _not_ stop at the terminal chunk (`run_turn_once` reads until the byte stream ends, and `push_sse_line` consults `is_terminal()` only to tolerate non-JSON trailing noise). A gateway that replays the assembled `tool_calls` array after the terminal chunk would otherwise match no open call (`close_tools` drained `open_tools` while `next_tool_index` kept counting), open at a fresh index, and reach the session as a _second_ `tool_use` block carrying the same wire `id` — which the loop then dispatches and physically executes twice, since nothing downstream dedupes by id. The trailing `stream_options.include_usage` chunk (`choices: []`, by design after `finish_reason`) is deliberately outside the guard, so accounting still lands |
| `dialect/openai_responses.rs` | `/v1/responses` body builder (flat `input` array of typed items, flat tool defs, `max_output_tokens`, `reasoning.effort` + `reasoning.summary` (`ModelRequest::reasoning_summary`, defaults to `ReasoningSummary::Auto`) + `include:["reasoning.encrypted_content"]`, `service_tier` (`ModelRequest::service_tier`, omitted unless set — a queueing/latency class, not just pricing), `store:false`; the explicit `reasoning:{"effort":"none"}` disable is withheld when `ModelRequest::is_copilot` (GitHub Copilot-hosted gpt-5.x ids have no "off" wire shape at all, unlike the same id direct from OpenAI); `prompt_cache_retention` is withheld entirely when `ModelRequest::is_azure` (pi's Azure dialect never sends it)) + decoder — genuine item-boundary events (`output_item.added`/`.done`), not implicit index-keyed deltas; every event carries its own true `output_index` and is emitted immediately, live, in real arrival order — no buffering, no "focus" item — since `Accumulator` natively tracks as many concurrently-open indices as the wire actually has (genuinely interleaved items, e.g. concurrent tool calls, stream live rather than one being buffered and replayed as a burst once the other closes); `function_call_arguments.done`/`output_item.done`'s own `arguments`/`content` resync (replace, not append) whatever the streamed deltas produced, so a single dropped/duplicated delta can't silently corrupt the final block; a `message`-type item's `id`/`phase` are captured off that same `output_item.done` and restamped verbatim on replay (`push_assistant_content`/`fallback_message_id`)                                                                                                                                                                                                                                                                                                                                                                                                     |
| `codex_websocket.rs`          | The live Codex WebSocket transport (see the `Codex live WebSocket transport` capability above): `CodexWebSocketCache` (per-`GatewayClient` connection cache, keyed by `cache_key`), `Continuation`/`build_wire_body` (the delta-diffing logic, pure and unit-tested standalone), `try_stream`/`Attempt` (the connect-or-reuse/send/first-event orchestration `client.rs::GatewayClient::stream` calls into)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `branch_summary.rs`           | `branch_summary_request`: the (network-free) prompt builder for summarizing an abandoned tree branch on navigation — reuses `compaction`'s `SUMMARY_SYSTEM`/`extract_file_ops` unchanged and its `render_prefix` in spirit but not verbatim (`render_prefix_without_tool_results` drops tool-result content entirely rather than truncating it, matching pi's `getMessageFromEntry` returning `undefined` for a tool-result-role entry — a branch summary is read once on return, not carried forward as live context, so terseness matters more than detail here), rendered transcript wrapped in `<conversation>` tags (matching pi's `generateBranchSummary` and this crate's own compaction summary-request path), framed by its own instruction (no incremental-update path; a branch is summarized once), fixed `BRANCH_SUMMARY_MAX_TOKENS` (2048) output budget — matching pi's own hardcoded `generateBranchSummary` constant, deliberately independent of `CompactionConfig::summary_max_tokens`'s `reserve_tokens`-scaled budget, since a branch recap has no incremental-update path to size for; `windowed_by_budget` trims the rendered tail to fit the summarization call's own context, privileging a nested compaction/branch-summary entry (a dense recap, not raw conversation) past the ordinary cutoff as long as the accumulated tail is still under 90% of budget, matching pi's `prepareBranchEntries`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `session_title.rs`            | `session_title_request(model, messages)`: a one-off request (`SESSION_TITLE_SYSTEM`, `SESSION_TITLE_MAX_TOKENS` 32) naming a session from its first user turn and first assistant reply, text only; `clean_title` strips quotes/preamble and discards a `none` reply or anything over `SESSION_TITLE_MAX_CHARS` (60). Generated once, never regenerated                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `session.rs`                  | `Session`: Arc-shared copy-on-write message history + step/token counters + `last_usage_message_count`, serde round-trippable; the prompt-size anchor `record_usage` keeps and `push` turns into `prior_turn_tokens` (`forget_anchor` after a rewrite); `restore_usage_snapshot` for a reopened session; `scrub_cross_model_state(new_model)` downgrades a non-empty signed `Thinking` block to `Text` (drops it only if empty), drops `RedactedThinking`, and truncates a combined tool-call id — all from any message not stamped with `new_model`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `error.rs`                    | `Error` (loop/transport, aborts the run) and `ToolError` (tool failure, becomes an error `tool_result`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `mock.rs`                     | `MockTransport` + `turn::{text, tool_call}` builders — scripted, no-network loop testing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `tests/client_socket.rs`      | `GatewayClient` over a real TCP socket: SSE decode, UTF-8 chunk-split reassembly, HTTP-error surfacing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## Configuration

| Setting                                           | Default              | What It Controls                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DEFAULT_MAX_TOKENS` / `Agent::with_max_tokens`   | model-aware (≥ 4096) | Per-turn output ceiling (`max_tokens`/`max_completion_tokens` per dialect). `Agent::new` seeds it from `capabilities(model).max_output` (floored at 4096); the compaction `context_window` is likewise seeded from `capabilities`. Both still overridable via the builders                                                                                                                                                                                                                                                                                                                                                                                                               |
| `Agent::with_max_steps`                           | `None` (unbounded)   | Opt-in loop-iteration ceiling; at the ceiling the model gets `WRAP_UP_STEPS` more to stop cleanly (see "Wrap-up at the step ceiling"), and the iteration after those returns `Error::MaxSteps` before sending a request. Unbounded by default (matching pi's own loop): the run is already bounded by cancellation and token spend, and every finite default this crate shipped (24, then 50) got hit by legitimate deep tasks without a recorded runaway to its name. `Error::MaxSteps` is resumable: the check runs before any per-turn state is touched, so a fresh `run`/`run_events_steered` call against the same session simply continues past it with a new per-call step budget |
| `Agent::with_system`                              | `None`               | System prompt; hoisted to each dialect's native system field (Anthropic top-level `system`; OpenAI a leading `developer`/`system`-role message per `instruction_role`, or top-level `instructions` on the Codex route)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `Agent::with_system_fn`                           | `None`               | Per-turn system-prompt callback, re-evaluated every turn; takes priority over `with_system` when both are set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `Agent::with_tools`                               | empty                | The tool set advertised to the model; definitions + JSON Schemas computed once here, shared via `Arc<[ToolDef]>` for the agent's lifetime (mid-run overridable — see `Steering::request_tool_set`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `Agent::with_block_images`                        | `false`              | Force the vision-downgrade path regardless of the active model's real `supports_vision` capability                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `CONNECT_TIMEOUT` (`client.rs`)                   | 10s                  | TCP+TLS handshake cap to the gateway; mirrors the gateway's own upstream `connect_timeout_secs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `READ_TIMEOUT` (`client.rs`)                      | 600s                 | Idle timeout _between_ reads on the streaming body (not total stream duration); sized to the gateway's own upstream `read_timeout_secs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `Agent::with_rate_limit_retry`                    | 300s budget          | How long one turn re-sends a rate-limited/overloaded request; waits honour the server's hint, else 500ms doubling to 60s                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MAX_BACKOFF` / `GatewayClient::with_max_backoff` | 60s                  | Ceiling on a single backoff wait (exponential or `Retry-After`-derived), independent of `with_retry`'s `max_retries`/`base_backoff`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `GatewayClient::with_extra_headers`               | empty                | Operator-configured headers merged onto every outgoing request, applied last (wins on a name collision)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

## Failure Modes

| Failure                                                                                           | What Actually Happens                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Recovery                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A run's own steps reach an opted-in `max_steps`                                                   | The model first gets `WRAP_UP_STEPS` (5) more to stop cleanly; if it uses them, `run_events` returns `Error::MaxSteps(n)` (the configured ceiling) before another request is sent; session retains every completed turn (no ceiling exists unless the caller set one)                                                                                                                                                                                                                                                                                                                                                                              | Caller persists/inspects `Session`; can raise/clear `max_steps` and resume                                                                                                                                                                                               |
| Gateway returns non-2xx                                                                           | Body read as text (capped at 4,000 chars); `Error::Transport("{host} returned {status}: {detail}")` returned from `stream()` once `send_with_retry` gives up — `{host}` is whichever host answered (a direct provider route has no gateway), and a response with a `Retry-After` hint carries the derived wait as `(retry-after-ms: N)` for the turn-level rate-limit retry — a 401 specifically gets a pointed, actionable message naming the API key as the likely cause instead of a bare status code (pi-parity fix: neither this crate nor pi itself used to say anything more specific than the raw upstream body for a live auth rejection) | A rate limit/overload (429, 529) is re-sent by `run_turn` within `RateLimitRetry::budget`; any other error text `is_retryable_mid_stream` accepts gets `MAX_MID_STREAM_RETRIES`; otherwise the run errors with no partial turn                                           |
| SSE chunk splits a multi-byte UTF-8 char                                                          | Raw bytes buffered across chunks; decoding waits for the newline, so the split is invisible                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Transparent — no error (regression-tested over a real socket)                                                                                                                                                                                                            |
| SSE-framed line isn't valid UTF-8                                                                 | `Error::Transport("invalid UTF-8 in SSE stream: …")` from inside the event stream                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Stream item `Err`; in-progress turn discarded, error returned                                                                                                                                                                                                            |
| SSE `data:` payload isn't valid JSON                                                              | `Error::Transport("malformed SSE json: …")`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Same as above                                                                                                                                                                                                                                                            |
| Streamed tool-call JSON never completes/parses                                                    | The call never runs; its `tool_use` block keeps `close_incomplete_json`'s partial object (or `{}`), and the loop feeds back an error `tool_result` (`CUT_OFF_TOOL_CALL` under a `MaxTokens` stop, else `invalid_tool_input_error`'s diagnosis with a short excerpt, not the whole input)                                                                                                                                                                                                                                                                                                                                                           | Run continues; model corrects on the next turn (recoverable, not fatal)                                                                                                                                                                                                  |
| Model calls an unregistered tool name                                                             | That call's result becomes `("unknown tool: {name}", is_error: true)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Not fatal — fed back as an error `tool_result` the model sees next turn                                                                                                                                                                                                  |
| A registered tool's `run()` returns `Err`                                                         | The error's `Display` text becomes `ToolResult.content` with `is_error: true`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Not fatal — same as above                                                                                                                                                                                                                                                |
| Stream ends without its terminal event (`message_stop` / `finish_reason` / terminal `response.*`) | The decoder's `finish` returns `Error::Transport("… stream ended before …")`; a body that never started at all (no event) is the one case that still finishes as an empty `EndTurn` turn with `0` usage                                                                                                                                                                                                                                                                                                                                                                                                                                            | `run_turn` retries it from a fresh connection (`is_retryable_mid_stream`, up to `MAX_MID_STREAM_RETRIES` = 3, unless `with_auto_retry(false)`); then `Err(Transport)`                                                                                                    |
| Gateway holds the connection open with no bytes for >600s                                         | `reqwest`'s idle read timeout fires. Mid-body it surfaces as `Error::Transport` tagged `MID_STREAM_NETWORK_ERROR`; before the response arrives it is a send error                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Mid-body: `run_turn` retries from a fresh connection (up to `MAX_MID_STREAM_RETRIES` = 3, 250ms doubling to 5s, unless `with_auto_retry(false)`); before the response: `send_with_retry` retries it like a connection failure (`MAX_RETRIES` = 3). Then `Err(Transport)` |
| A `Mutex` in `MockTransport` is poisoned by a panicked test thread                                | `lock().unwrap_or_else(\|e\| e.into_inner())` takes the guard back out of the poisoned lock instead of panicking again                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Transparent — later calls on the same mock keep working                                                                                                                                                                                                                  |
