//! Headless `serve` — a newline-delimited JSON control protocol, over stdio, a WebSocket, or HTTP POST.
//!
//! The server is the source of truth; any client (a TUI, an editor, an `ssh` pipe, a script, or the Beyond
//! iPhone app) drives it by sending one JSON command per message and reading one JSON frame per
//! message. The shape mirrors pi's `rpc` mode and opencode's session server: commands get a
//! `response` frame, and a `prompt` streams `event` frames (the agent's `AgentEvent`s) before its
//! response. A `tool_start` event carries `not_run:true` (absent otherwise) for a call that will not
//! run because the output limit cut its input off or it never parsed: `input` is partial and an
//! error `tool_end` follows at once, so a client can show it as refused from the start.
//!
//! **Three transports, one protocol.** The command/frame protocol below is byte-identical regardless of
//! transport. [`serve`] is the default stdio transport (one line per command on stdin, one frame per
//! line on stdout — built for an `ssh` pipe). [`serve_ws`](crate::serve_ws), enabled by
//! `--listen <addr>`, offers the same protocol over a WebSocket (one command per inbound text message,
//! one frame per outbound text message) **and** over HTTP POST to the same path (one command in the
//! body; the matching `ack`/`response` is the HTTP body). Both socket transports feed the
//! transport-agnostic [`serve_session`] core, which
//! reads commands from an `mpsc` channel and emits frames to whichever connection is currently
//! attached — so a WebSocket session **outlives its connection**: a dropped mobile client reconnects
//! (same `?session_id=`) and re-attaches to the same still-running run. An HTTP POST is a *view* in
//! the same sense: it injects one command and returns, and the run keeps going (a `prompt` is 202
//! Accepted on the ack; events live on a WebSocket attach or on [`crate::lifecycle`]). A reconnecting
//! WebSocket is caught up on whatever committed while it was gone by a `catchup` frame the server
//! seeds on attach, *before* the connection starts receiving live frames — catch-up is not something
//! the client asks for and hopes arrives first, because frames carry no sequence number and delivery
//! order is the only order a client has (see [`OutFanout::add_with_catchup`]). A POST is not seeded
//! with catch-up: it is waiting for one command's `ack`/`response`, not reconstructing a stream.
//!
//! Sessions persist as append-only JSONL: `--session-file` for one session, or `--session-dir` for a
//! [`SessionRepo`](crate::session_store::SessionRepo) of many (WebSocket sessions get one file each,
//! named by session id). A turn appends only its new messages (compaction rewrites atomically). A
//! reattaching client sees a **stable** session id and metadata.
//!
//! Commands (stdin): `{id?, type, …}`
//!   - `{type:"prompt", message, streaming_behavior?, output_schema?, output_description?}` run a turn:
//!     an immediate lightweight `ack`
//!     frame (the turn is queued and starting), then `event` frames, then a `response` whose `data`
//!     includes `refused: bool` — whether the run's last turn ended in a refusal rather than an
//!     ordinary stop (a refusal doesn't drain queued `steer`/`follow_up` messages; see `agent-core`).
//!     Sent while another `prompt` is already in flight, it's rejected as busy *unless*
//!     `streaming_behavior` is `"steer"` or `"follow_up"`, in which case `message` is queued through
//!     the same `Steering` lane an explicit `steer`/`follow_up` command would use.
//!
//!     `output_schema` (a JSON Schema object) makes the run a callable function: the model must return
//!     a conforming payload through the `structured_output` tool instead of answering in prose, and
//!     `data.structured_output` on the terminal response carries it — `null` if the model never
//!     produced one, and **absent entirely** when the prompt didn't ask for typed output, so a client
//!     can tell the two apart. `output_description` overrides what the model is told the payload is for.
//!     Both are **per-prompt**: the schema is installed for this run and removed by the next prompt that
//!     omits it. A malformed schema is rejected with `success:false` *before* the `ack`, so no model call
//!     is billed for a contract that can never be satisfied.
//!   - `{type:"abort"}`                  cancel the in-flight `prompt` (if any), else a no-op ack —
//!     the ack for a *mid-run* abort isn't sent until the cancelled run has actually gone idle (matches
//!     pi's own `agent-session.ts` awaiting `waitForIdle()` first), so a client that treats the ack as
//!     "safe to send the next command" doesn't get rejected as busy in the gap
//!   - `{type:"abort_retry"}`            interrupt a pending whole-run auto-retry backoff (see below),
//!     surfacing the real underlying error that triggered it rather than waiting the delay out; a
//!     no-op ack when no retry is pending (a run is actively streaming, or none is in flight at all)
//!   - `{type:"stop_after_turn"}`        request a graceful stop at the next turn boundary — the
//!     current turn's tool calls (if any) still finish and their results are committed, unlike
//!     `abort`. A no-op ack when no `prompt` is in flight (never affects a *future* prompt).
//!   - `{type:"switch_model", model, thinking?}` retarget the *in-flight* `prompt` — pi's own
//!     `prepareNextTurn`: unlike `set_model` (only takes effect on the next `prompt`), this changes
//!     what every subsequent turn of the *current* run targets, without stopping and restarting it.
//!     Applied at the next turn boundary; the turn already in flight when this arrives is unaffected.
//!     A no-op ack (pointing the client at `set_model` instead) when no `prompt` is in flight.
//!   - `{type:"get_state"}`              → `data: {session_id, model, steps, message_count, title, cwd,
//!     git_branch, thinking_level, auto_compaction, auto_retry, steering_mode, follow_up_mode,
//!     pending_messages, steer_queue, follow_up_queue, …}` — the middle eight are the runtime-mutable
//!     settings and current steer/follow-up queue state, so a client can render current settings
//!     without a second round trip. `pending_messages` is the bare depth (`Steering::pending_count`);
//!     `steer_queue`/`follow_up_queue` (Fix 5, pi-parity gap) are each lane's actual queued message
//!     text (`Steering::steer_texts`/`follow_up_texts`) — previously a client could see *how much* was
//!     queued but never *what*. `cwd`/`git_branch` (Task #25, pi-parity fix) matter more here than for
//!     pi: a headless RPC client may have no shared filesystem with this process at all, so `get_state`
//!     is its only way to learn which directory (and branch) the agent's tools are actually operating
//!     against. `git_branch` is a lazily-resolved, best-effort `git symbolic-ref --short HEAD` — `null`
//!     (never a failed call) outside a git repo, on detached `HEAD`, or if the lookup fails for any
//!     other reason. In service mode it runs **in the sandbox**, through that session's own runner,
//!     since the workspace is not on this machine.
//!   - `{type:"get_messages", since?}`   → `data: {messages: [...], leaf_id}` (each message tagged with
//!     its tree `id` when persistence is configured, so a client can fork from any point via
//!     `switch_branch`; `leaf_id` is the same active-tip id `get_tree`'s own response carries — pi's own
//!     `get_entries({since})` returns `{entries, leafId}` in one round trip too, so a client doesn't need
//!     a second `get_tree` call just to learn the current tip); `since` (a tree id the client already
//!     has) returns only the messages appended after it — an error, not a silent full re-fetch, when
//!     `since` names no known id (or persistence isn't configured, so nothing is tagged at all)
//!   - `{type:"new_session", parent_session?}` start a fresh session → `data: {session_id, parent}`
//!     (repo mode: `parent` is `parent_session` when given, else whatever session id was active
//!     immediately before this call — pi's own `parentSession` lineage marker, provenance only, not a
//!     fork; `null` in single-file/in-memory mode, where there's no *new* id to link). An *addressed*
//!     session — `--session-id`, the daemon's `?session_id=` — keeps its own id instead of minting one,
//!     archiving the outgoing conversation into a sibling session; see [`Persistence::new_session`]
//!   - `{type:"list_sessions", query?}`  (repo mode) → `data: {sessions: [SessionMeta + updated_at/
//!     message_count/preview…]}` (via `SessionMeta::to_listing_json` — those derived fields are
//!     `#[serde(skip)]` on the struct itself), this project's sessions only (matched by the default
//!     per-cwd directory, or whatever `--session-dir` points at). An optional `query` string filters and
//!     ranks the result (`session_store::search_sessions` — case-insensitive substring match against
//!     `title`/`id`/`preview`/`cwd`, in that priority order); omitted (or blank) returns
//!     every session, recency-sorted, unchanged from before this existed. The underlying scan
//!     (`SessionRepo::list_with_progress`) runs across a small worker pool rather than one file at a
//!     time, and streams `list_progress` frames while it's in flight.
//!   - `{type:"list_all_sessions", query?}` (repo mode) → same shape, same `query` handling, and same
//!     `list_progress` streaming as `list_sessions`, across every project's own session directory, not
//!     just this one's — each entry's own `cwd` field says which project it belongs to (pi's
//!     cross-project `listAll`)
//!   - `{type:"switch_session", session_id}` (repo mode) load another session
//!   - `{type:"delete_session", session_id}` (repo mode) soft-delete another session (moved to
//!     `.trash`, not the currently active one — see `Persistence::delete`'s doc comment); idempotent
//!   - `{type:"list_trash"}` (repo mode) → `data: {trash: [{id, deleted_at, original_path}...]}` every
//!     session sitting in `.trash/`, most-recently-deleted first — see `session_store::TrashEntry`
//!   - `{type:"restore_session", session_id}` (repo mode) move a trashed session back to its original
//!     location; fails clearly (not a silent overwrite) if something already occupies that path, and
//!     reports failure (not idempotent success, unlike `delete_session`) when `session_id` isn't in
//!     `.trash/` at all
//!   - `{type:"fork", upto?, target_id?, before?}` (repo mode) copy a prefix into a new session, switch
//!     to it. `target_id` (any tree entry, on or off the active path) wins over `upto` (a message-count
//!     prefix of just the active path) when given; `before:true` excludes `target_id` itself from the
//!     copied prefix (fork right before it), the default `true` excludes it. **In service mode it
//!     does not switch**: the new `<shard>.<opaque>` id comes back with `switched:false`, and the
//!     client opens it with a grant of its own (this session's id is what its slot, lock and grant
//!     are keyed on). A `fork` arriving mid-run is answered "busy" rather than aborting the run.
//!   - `{type:"clone"}` (repo mode) `fork` with no arguments — the current session's active path in
//!     full, at its current tip. pi's own `clone`: a thin, argument-free alias, not a separate code
//!     path (pi needs it because pi's own `fork` requires an explicit entry id; this crate's `fork`
//!     already defaults to the same behavior when called bare).
//!   - `{type:"get_fork_messages"}` list this session's own candidate fork points — every user-turn
//!     entry anywhere in the tree, `{entry_id, text}` — matching pi's own same-named command (which
//!     takes no parameters, is scoped to the current session only, and walks its own `getEntries()`'s
//!     *every* entry ever appended, not just the active path — Track (pi-parity fix): this used to be
//!     scoped to `active_ids()` only, so a message on a branch already navigated away from never
//!     appeared as a candidate); feed one `entry_id` to `fork`'s `target_id` to actually fork there. Not
//!     a preview of `fork`'s output — see `preview_fork` for that (this crate's own extension, previously
//!     misnamed `get_fork_messages`, which broke a pi-compatible client expecting the shape above)
//!   - `{type:"preview_fork", session_id?, upto?, target_id?, before?}` (repo mode) preview what `fork`
//!     would produce for *any* session at an arbitrary point — no new session, no switch
//!   - `{type:"set_session_name", title}` set the session's title → `data: {title}` (the final,
//!     sanitized value — see `session_store::sanitize_title` — `null` if it sanitized to empty); also
//!     pushes an unsolicited `session_info_changed` frame carrying the same `title`, so a client doesn't
//!     need a follow-up `get_state` to learn what was actually recorded (pi's own `session_info_changed`)
//!   - `{type:"set_label", target_id, label?}` set (or, with `label` omitted/`null`, clear) a
//!     user-defined bookmark on any tree entry (not just the active tip) → `data: {target_id, label}`;
//!     an error when `target_id` names no known entry, or persistence isn't configured (there's no tree
//!     to label at all in pure in-memory mode) — `SessionStore::set_label`, fully built but previously
//!     unreachable from any RPC command
//!   - `{type:"get_label", target_id}`   the label currently set on `target_id`, if any → `data:
//!     {target_id, label}` (`label: null` if never labeled or last cleared); same persistence-required
//!     error as `set_label`
//!   - `{type:"append_custom", kind, data?}` append an opaque, caller-defined entry as a child of the
//!     active tip and advance the tip to it → `data: {id}` (the new entry's id); `data` defaults to `{}`
//!     when omitted, `kind` identifies the shape of `data` to whatever produced it (this crate never
//!     interprets either) — `SessionStore::append_custom`, fully built and tested but previously
//!     unreachable from any RPC command; same persistence-required error as `set_label`. A kind the
//!     agent writes itself (`session_store::HOST_CUSTOM_KINDS`) is refused, so a client cannot forge
//!     one
//!   - `{type:"export_html", output_path?}` render the active session's transcript as a single
//!     self-contained HTML file → `data: {path}`. `output_path` defaults to a timestamped
//!     `session-<unix-seconds>.html` in the current directory; parent directories are created as
//!     needed.
//!   - `{type:"compact", custom_instructions?}` summarize the prefix now (while it runs, `abort`
//!     cancels it, `get_state`/`get_session_stats`/`get_messages` are answered, and anything else
//!     is refused as busy) → `data: {compacted: bool,
//!     reason?, summary?, tokens_before?, tokens_after?, first_kept_entry_id?}` — `reason` (Task #26
//!     pi-parity fix) is `null` on a real compaction, else `"too_small"`/`"already_compacted"`
//!     (`agent_core::CompactOutcome`), matching pi's own two distinct thrown errors instead of
//!     collapsing both no-op cases into the same bare `compacted:false`; `first_kept_entry_id` (Fix 2,
//!     pi-parity gap) is the tree entry beginning the retained post-compaction portion of history —
//!     pi's own `firstKeptEntryId` — `null` when persistence isn't configured or no compaction fired
//!   - `{type:"get_last_assistant_text"}` → `data: {text}` (the latest assistant reply)
//!   - `{type:"get_todos"}`              → `data: {todos}` — the model's current `todo` list (`null` if
//!     it never made one). The `todo` tool is stateless and its live `tool_progress` frames are
//!     ephemeral, so this is how a client attaching mid-run learns the plan without replaying history.
//!     Idle, it reads the last `todo` call still in the transcript, falling back to the list a past
//!     compaction folded into `Session::compaction`; mid-run, it reads the live mirror (`LiveStats`).
//!   - `{type:"get_session_stats"}`      → token/step accounting + message-type breakdown
//!     (`user_messages`/`assistant_messages`/`tool_calls`/`tool_results`/`total_messages`), plus
//!     `context_usage: {tokens, context_window, compact_at, percent}` — `compact_at` is where
//!     compaction fires, at or below `context_window` (the input budget `percent` is against)
//!   - `{type:"get_commands"}`           → discoverable skills + prompt templates, each entry carrying
//!     `{name, source, description, scope, path}` — `scope` (Task #39 pi-parity fix, previously
//!     omitted) is `"user"`/`"project"`/`"temporary"` (which discovery root it actually came from),
//!     matching pi's own `get_commands` `sourceInfo.scope`/`sourceInfo.path`
//!   - `{type:"reload"}`                 re-run project-instruction/skill/prompt-template discovery and
//!     re-check trust, refreshing the static half of the system prompt (the cheap date/cwd footer is
//!     already refreshed every turn regardless)
//!   - `{type:"set_model", model}`       switch the model for subsequent prompts → `data:` the same
//!     capability shape `get_available_models`' own entries carry (`model`, `provider`,
//!     `context_window`, `max_output`, `reasoning`, `supports_vision`), plus `reasoning_effort` — so a
//!     client learns the new model's capabilities without a separate round trip; `cycle_model` below
//!     carries the identical shape (plus its own `scoped`). `model` is trimmed and rejected if
//!     empty/whitespace-only — the one mistake this process can catch on its own; an
//!     unrecognized-but-well-formed id is *not* rejected (every id is forwarded verbatim through the
//!     gateway, with no local registry here to validate a real one against — `available_models()` is a
//!     non-exhaustive picker hint, not an allowlist). `model` may carry a trailing `:<level>` suffix
//!     (Fix 2, pi-parity gap — e.g. `"sonnet:high"`, pi's own `--model <pattern>:<thinking-level>`
//!     shorthand) to set the reasoning effort in the same call, winning over whatever level was
//!     already active
//!   - `{type:"set_thinking", budget}`   set/clear an explicit raw thinking-budget override (integer,
//!     or `null` to disable it and defer back to the portable level below) → `data: {thinking}`
//!   - `{type:"set_reasoning_effort", effort}` set the portable thinking-depth level directly — one of
//!     off/minimal/low/medium/high/xhigh (or `null`, an alias for `"off"`) — correctly for whichever
//!     mechanism the active model actually uses (an Anthropic token budget, an Anthropic adaptive
//!     effort, or an OpenAI `reasoning_effort`); see `agent_core::ThinkingLevel` → `data: {level,
//!     thinking, reasoning_effort}`
//!   - `{type:"cycle_model"}`            advance through the `--models` scope when given (else the
//!     full known-model list `get_available_models` reports), wrapping — a `pattern:<level>` entry in
//!     `--models` pins that model's thinking level for as long as cycling stays on it
//!   - `{type:"cycle_thinking_level"}`   advance the same portable level one rung, wrapping from
//!     `xhigh` back to `off` → `data: {level, thinking, reasoning_effort}`
//!   - `{type:"set_auto_compaction", enabled}` toggle threshold-triggered compaction (manual `compact`
//!     is unaffected either way)
//!   - `{type:"set_auto_retry", enabled}` toggle transport-failure retry (on by default) — off surfaces
//!     a normally-retried transient failure immediately, for debugging a flaky connection rather than
//!     waiting through several silent attempts. Gates *two* layers as one user-facing concept: the
//!     mid-stream retry inside a single model turn (`agent_core::Agent::with_auto_retry`; each of its
//!     retries streams an `event` with `kind:"retry"` (`rate_limited` is true for the time-budgeted class; an
//!     overload adds `overloaded:true` beside it, absent otherwise), and a rate limit is waited out there for up to
//!     `--retry-rate-limit-secs`), and, once that's exhausted, automatically re-invoking the *whole*
//!     `prompt` run against the same session up to 3 more times without progress, with backoff
//!     (2/4/8s) — pi's `agent-session.ts` auto-retry. Each whole-run retry
//!     attempt emits an unsolicited `auto_retry_start` frame (`{type:"auto_retry_start", id, command:"prompt",
//!     attempt, max_attempts, delay_ms, error}`) before its backoff sleep. A sequence that made at
//!     least one attempt ends with exactly one `auto_retry_end` frame (`{type:"auto_retry_end", id,
//!     command:"prompt", success, attempt, final_error?}`) — `success:true` when a retried attempt
//!     recovers, `success:false` (with `final_error`) when retries are exhausted, the failure turns out
//!     non-retryable, or the pending backoff itself is interrupted via `abort`/`abort_retry`
//!     (`final_error:"retry cancelled"`) — mirroring pi's own `auto_retry_end` event.
//!   - `{type:"set_block_images", enabled}` toggle forcing every image down the vision-downgrade path
//!     regardless of the active model's real `supports_vision` capability (pass 20, pi-parity fix: same
//!     mutate-persist-rebuild shape as `set_auto_compaction` above — previously only settable at process
//!     startup via `--block-images`/a persisted `agent settings --block-images` default, with no live
//!     toggle for an already-running `serve`).
//!   - `{type:"set_image_auto_resize", enabled}` toggle `read`'s resize/downscale path for an oversized
//!     image (pass 20, pi-parity fix: same shape as `set_block_images` just above).
//!   - `{type:"set_steering_mode", mode}`/`{type:"set_follow_up_mode", mode}` toggle how much of the
//!     steer lane (mid-run drain) / follow-up lane (stop-boundary drain, including any stranded steer
//!     messages swept in there) a single drain point consumes (`agent_core::QueueMode`) —
//!     `"one_at_a_time"` (the default, matching pi): only the oldest queued message is injected per
//!     drain, the rest stays queued for the next one; `"all"`: everything queued is folded into one
//!     injection. The two lanes have independent settings, matching pi's own `steeringMode`/
//!     `followUpMode`. Takes effect immediately (owned by the `Steering` handle itself, no `Agent`
//!     rebuild), including mid-run.
//!   - `{type:"get_available_models"}`   → `data: {models: [{id, provider, context_window, reasoning}
//!     …]}` — always the full shown-model list (`listed_models`: the resolution hint list plus every natively served catalog chat model), never narrowed by `--models`
//!     (that only scopes `cycle_model`'s own candidate list — see its entry above), each entry's
//!     capability fields read from the same `agent_core::capabilities` table every wire decision
//!     already consults (pi's own structured `Model<any>`, minus pricing — gateway-owned, out of scope
//!     here)
//!   - `{type:"list_branches"}`          → `data: {branches: [BranchInfo…]}` (the session's *leaves*)
//!   - `{type:"get_tree", since?}`       → `data: {nodes: [TreeNode…], leaf_id}` (every message on
//!     every branch, not just the leaves `list_branches` reports; `leaf_id` is the active path's own
//!     tip — `null` in pure in-memory mode — pi's own `get_tree`'s `leafId`). `since` (Task #48,
//!     pi-parity gap — a tree id the client already has, same contract as `get_messages`'s own `since`
//!     above) returns only the entries appended after it, across every entry type (message,
//!     branch_summary, compaction, custom — unlike `get_messages`'s `since`, which only ever sees plain
//!     LLM messages on the active path) — pi's own `SessionManager.getEntries({since})` backs both its
//!     `get_tree` and its dedicated `get_entries` RPC command the same way; an error, not a silent full
//!     re-fetch, when `since` names no known id (see `nodes_since`)
//!   - `{type:"switch_branch", target_id, before?, summarize?, custom_instructions?,
//!     replace_instructions?}` navigate to another point in the tree — or, when `before:true`, to
//!     `target_id`'s own *parent* instead, which is the tree's own root (before any message) when
//!     `target_id` is the very first message, letting a client redo it in place (pi's own
//!     `SessionManager::resetLeaf`) — summarizing the abandoned branch's activity first unless
//!     `summarize:false` (an optional `custom_instructions` string steers what that recap emphasizes,
//!     the same "Additional focus" framing `compact`'s own `custom_instructions` supports;
//!     `replace_instructions:true` (Task #17, pi-parity fix) uses `custom_instructions` as the *entire*
//!     instruction section instead of appending it after the default template — a no-op without
//!     `custom_instructions`; both ignored when `summarize:false`, since no summarization call happens
//!     at all) → `data: {target_id, model, reasoning_effort}` — also restores whichever
//!     model/thinking-level was actually active at wherever the session actually landed (a
//!     `set_model`/`cycle_model`/`set_reasoning_effort`/`cycle_thinking_level` made *after* leaving
//!     that point doesn't leak backward into it), rebuilding the `Agent` if either differs from what's
//!     currently active
//!   - `{type:"bash", command, cwd?, timeout_ms?, exclude_from_context?}` run a shell command directly
//!     — independent of the model's own tool-call loop; `timeout_ms` cancels it at that deadline
//!     ("Command timed out after 1.5 seconds"), and without one it runs until it exits or `abort_bash`
//!     — streaming `tool_progress`/`tool_end` events
//!     exactly like a model-invoked `bash` call, then a terminal `response` whose `data` carries
//!     `{result, is_error, exit_code, cancelled, truncated, full_output_path}` (Fix 7, pi-parity gap: the
//!     last four previously only ever reached a client via the interim `tool_progress` event, or embedded
//!     as a status line inside `result` itself — pi's own structured `BashResult`). Always recorded as a
//!     plain informational message in the session/persisted storage/an eventual `export` (pi's
//!     `recordBashResult`); `exclude_from_context: true` (Fix 9, pi-parity gap: previously this instead
//!     skipped recording entirely, losing the command/output outright) hides it from the model on its
//!     *next* turn only, mirroring pi's separate `convertToLlm` transform. Rejected if `bash` isn't
//!     registered for this process (`--exclude-tools bash` / `--no-tools`). While it runs, only
//!     `abort_bash`/`abort` (cancel it) are accepted; everything else is rejected as busy.
//!   - `{type:"abort_bash"}`             cancel an in-flight host `bash` command, else a no-op ack
//!   - `{type:"login", provider}`        log into `anthropic`/`github-copilot`/`openai-codex` as a
//!     subscription instead of a metered API key (see `crate::oauth`/`crate::auth_store`) — an
//!     immediate `ack`, then zero or more unsolicited `login_progress` frames (a URL to open, a
//!     device code, or narration), then a terminal `response` once the flow resolves, however long
//!     that takes; runs as a detached background task, not inline, so every other command (including
//!     a `login` for a *different* provider — a second one for the same provider in flight is
//!     rejected) keeps working while it's pending
//!   - `{type:"submit_code", code}`      only meaningful after a `login_progress{step:"manual_code"}`
//!     (the local OAuth callback listener couldn't complete, e.g. no loopback access over SSH) →
//!     `data: {accepted: bool}`; the originating `login` request's own `id`-correlated `response`
//!     remains the authoritative completion signal, this only confirms the code was delivered
//!   - `{type:"approve", request_id, decision, scope?}` answer an outstanding `approval_request` (see
//!     `--approve`). `decision` is `"allow"`/`"deny"`; `scope` is `"once"` (default) or `"session"`,
//!     which remembers the decision for the rest of the session against the request's own `scope_key`.
//!     → `data: {accepted: bool}`. `accepted:false` means the `request_id` names no outstanding
//!     question — another attached client already answered it, it timed out, or the run was aborted;
//!     that is a race a client can lose, not an error. Accepted **while a run is in flight**, which is
//!     the only time a question can be outstanding at all.
//!   - `{type:"abort_login"}`            cancel an in-flight `login`, else a no-op ack
//!   - `{type:"logout", provider}`       remove `provider`'s stored subscription credential, if any,
//!     idempotently → `data: {provider, was_logged_in}`
//!   - `{type:"auth_status", provider?}` read-only, never touches the network — subscription-login
//!     status (`logged_in`/`logged_out`/`needs_reauth`, the last meaning the most recent refresh
//!     attempt failed but the credential is still on disk) for `provider`, or every known provider
//!     when omitted → `data: {provider, status}` or `data: {providers: [{provider, status}…]}`
//!   - `{type:"mcp_events_list", server?}` / `{type:"mcp_events_subscribe", server, name,
//!     arguments?, delivery?, action?, instructions?}` / `{type:"mcp_events_unsubscribe", server,
//!     name, arguments?}` — MCP Events (a **draft** extension; see [`crate::tools::mcp_events`]):
//!     list what each server offers and this session's subscriptions, subscribe (an idempotent
//!     upsert), unsubscribe (idempotent, `data: {removed}`). Run as spawned tasks, accepted mid-run too; refused in service mode.
//!
//! While a `prompt` runs, the loop keeps reading stdin so an `abort` can cancel it, or `steer` /
//! `follow_up` (with a `message`) can queue input: a `steer` is injected mid-run at the next tool
//! turn (to redirect a busy agent), a `follow_up` waits for the model to next stop. A handful of
//! read-only commands that don't need the run's exclusively-borrowed session — `get_state` (with
//! `message_count: null`, the one field that genuinely needs it), `get_session_stats` (from a live
//! mirror of the session's own counters, updated as the run streams), `get_commands`, `list_branches`,
//! `get_todos` (from that same live mirror), `get_tree`, `list_sessions`, `list_all_sessions`,
//! `get_available_models` — are answered live too,
//! rather than rejected, so a client can poll for progress during a long tool-heavy turn. Everything
//! else issued during a run is rejected as busy. `steer`/`follow_up` are also accepted while idle (no
//! `prompt` in flight) — they
//! queue against a persistent handle and are picked up by whichever `prompt` runs next;
//! `new_session`/`switch_session`/`fork`/`switch_branch` clear that queue so a message meant for one
//! session's next turn can't leak into another.
//!
//! Frames (stdout) are `{type:"ack", id?, command}`, `{type:"response", id?, command, success, data?,
//! error?}`, `{type:"event", event: <AgentEvent>}`, `{type:"list_progress", id?, command, scanned,
//! total}` — zero or more unsolicited progress updates a `list_sessions`/`list_all_sessions` scan emits
//! while in flight, correlated to the request via the same `id` its eventual `response` carries — or
//! `{type:"auto_retry_start", id?, command:"prompt", attempt, max_attempts, delay_ms, error}`, one per
//! whole-run auto-retry attempt (see `set_auto_retry` above), also correlated via `id` — or
//! `{type:"auto_retry_end", id?, command:"prompt", success, attempt, final_error?}`, the terminal
//! notice for a retry sequence that made at least one attempt — or `{type:"session_info_changed", id?,
//! command:"set_session_name", title}`, pushed once per successful `set_session_name` (pi's own
//! `session_info_changed`; see `set_session_name` above) — or `{type:"login_progress", id?,
//! command:"login", provider, step, url?, user_code?, verification_uri?, expires_in?, message?}`,
//! zero or more unsolicited updates for an in-flight `login`, correlated via `id` (see `login` above)
//! — or `{type:"session_superseded", session_id, tenant?}`, the last frame of a session whose storage
//! another owner has taken over (service mode's shared mounts): reconnect to the same id, which lands
//! on whoever owns it now. — or `{type:"mcp_event", server, name, arguments, delivery, action,
//! event}`, one per MCP Events occurrence a subscription of this session received (after `eventId`
//! dedup), and `{type:"mcp_event_status", kind, …}` for its lifecycle (`subscribed`, `resubscribed`,
//! `error`, `gap`, `terminated`, `delivery_status`). Subscriptions configured in settings belong to
//! one session only — this stdio session, or the daemon's `--mcp-events-session` — so only it sees
//! their events. An event whose action is not `notify` is also injected (durably — it is re-injected
//! after a restart until its run completes) as a `prompt` command with `id: "mcp_events:<n>"`,
//! `mcp_events: <n>` and `streaming_behavior` `follow_up`/`steer`, so its `ack`/`response` frames
//! carry that id.
//!
//! `{type:"catchup", data:{messages, leaf_id}, turn_in_flight, turn_truncated}` is pushed **once,
//! unsolicited, on attach** over the WebSocket/UDS transports (nothing is sent to a fresh session with
//! no history and no run — there is no backlog). Its `data` is byte-for-byte the payload `get_messages`
//! answers with, so a client parses one shape either way. It is the *first* frame the connection
//! receives, queued ahead of the sink going live.
//!
//! Catch-up is **base plus frames**: `data.messages` is the transcript as of the current run's start,
//! and — when `turn_in_flight` is `true` — every frame that run has emitted so far is *replayed*
//! immediately after it, before the live stream resumes. The two together reconstruct exactly what a
//! client that never dropped has seen, with nothing duplicated (which is why the base deliberately does
//! not advance at mid-run checkpoints). Committed history alone would not do: the turn in flight hasn't
//! committed, so a client would pick it up from the middle — a `tool_end` for a `tool_start` it never
//! got. `turn_truncated: true` means that turn outgrew the replay budget
//! ([`TURN_REPLAY_MAX_BYTES`]) and the client's view of it starts partway in, so it should reconcile
//! with `get_messages` once the run ends.
//!
//! The ordering here is the only thing a client can rely on — frames carry no sequence number — and it
//! is enforced structurally: catch-up, replay, and sink registration all happen inside one critical
//! section on the lock `broadcast` itself takes ([`OutFanout::add_with_catchup`]), so nothing can
//! interleave or be lost between them. It is not a race the client is expected to win. The stdio
//! transport has one connection for the life of the process and no re-attach, so it is never sent one.
//!
//! Two more frames exist only when `--approve` installed an interactive gate (see [`crate::approval`]).
//! `{type:"approval_request", request_id, tool, summary, scope_key, origin, options}` is broadcast to
//! **every** attached client when a gated tool call is about to run, and the call blocks until one of
//! them answers with an `approve` command. `summary` is a truncated preview of the call's arguments (a
//! megabyte file body is not fanned out to a phone); `scope_key` is what an `"session"`-scoped answer
//! would be remembered against, shown so the user can see exactly what they are agreeing to; `origin` is
//! `"main"` or `{agent, spawn_id}` naming the subagent child that is asking.
//! A question from an MCP-served skill (SEP-2640 — activating a model-chosen skill, code execution
//! while one is active, or a read from another server while one is active; asked whatever `--approve`
//! says, unless `--approve-mcp-skills`) carries an extra `mcp_skill` object: `{purpose:"activate",
//! server, uri, name, description, frontmatter, manifest, files, manifest_files, nested_in}`,
//! `{purpose:"execute", active_skills:[{server, uri, name, manifest}]}` or
//! `{purpose:"cross_origin_read", from_servers, to_server}`.
//! `{type:"approval_resolved", request_id, decision, scope?, reason?}` follows, broadcast to everyone, so
//! the clients that *didn't* answer can dismiss the prompt — a login is a single-client RPC and needs no
//! such frame, but an approval fans out to N clients of which exactly one wins. `reason` is
//! `timed_out`/`cancelled`/`no_client` when nobody answered; **all three deny the call**, because a gate
//! that fails open on absence provides no security (the model then gets an ordinary error `tool_result`
//! and the run continues, rather than hanging on a question nobody will ever see).
//!
//! **Structural stdout guard:** every byte on stdout must be a protocol frame — a stray line (a
//! debug `println!`, a misconfigured logger) would corrupt the NDJSON stream a remote client is
//! reading line-by-line. pi's equivalent (`output-guard.ts`'s `takeOverStdout`) monkey-patches
//! `process.stdout.write` globally so *any* code, including a dependency's own stray `console.log`,
//! is transparently redirected to stderr — the only sanctioned writer left is its own
//! `writeRawStdout`. That specific mechanism has no honest Rust equivalent here: it requires
//! reopening/redirecting the raw fd, and this workspace forbids `unsafe_code` outright (see the
//! workspace `Cargo.toml`), so fd-level interception is off the table on principle, not oversight.
//! The right *Rust*-appropriate backstop is a static one instead: `#![deny(clippy::print_stdout)]`
//! below makes any `println!`/`print!` in this module (including a future one, accidentally left in
//! from local debugging) a hard compile error — the only sanctioned stdout writer is already the
//! single `tokio::io::stdout()` task below, and nothing in this module has ever needed `println!`/
//! `print!` directly, so this costs nothing today and closes the most realistic version of this gap
//! (our own code, not a dependency's) permanently. It doesn't
//! reach into a third-party dependency the way pi's runtime patch does, but idiomatic Rust crates
//! overwhelmingly log through `tracing`/`log`, not raw stdout writes — a materially smaller residual
//! risk than in the Node ecosystem pi targets, where ad hoc `console.log` debug output is common.
#![deny(clippy::print_stdout)]

mod build;
mod commands;
mod config;
mod control;
mod frames;
mod gates;
mod git;
mod models;
mod out;
mod persistence;
mod prompt;
mod session_setup;
mod stats;
mod stdin;
mod steering;
pub(crate) use build::*;
use commands::{
    Activity, StateView, get_available_models_frame, get_commands_frame, get_messages_frame,
    get_session_stats_frame, get_state_frame, get_tree_frame, list_all_sessions_frame,
    list_branches_frame, list_sessions_frame,
};
pub use config::*;
pub(crate) use frames::*;
pub(crate) use gates::*;
pub(crate) use git::*;
pub use models::*;
pub(crate) use out::*;
use persistence::*;
pub use stats::*;
pub(crate) use stdin::*;
use steering::*;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::approval::{ApprovalDecision, ApprovalError, ApprovalScope};

use bytes::Bytes;

use agent_core::{
    Agent, AgentEvent, AgentHooks, CancellationToken, GatewayClient, Session, StopReason,
    StreamEvent, ToolUpdate,
};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::gateway_credential::{GatewayCredential, resolve_gateway_credential};
use crate::mode::Mode;
use crate::session_store::{
    BranchInfo, BranchSummaryDetails, SessionMeta, SessionRepo, SessionStore, search_sessions,
};
use crate::tools;

/// Waits for an OS shutdown request (SIGTERM, SIGHUP, or SIGINT/Ctrl-C) so `serve` (and `run` — see
/// `commands/run.rs::run_task`, which reuses this same type) can drain in-flight work and persist before
/// exiting — the same graceful-shutdown path stdin closing already takes for `serve` (cancel the run,
/// persist, break), just with the process's terminate/hangup signals as additional triggers. Without
/// this, a `systemctl restart`/`docker stop`/pod eviction mid-turn (or a plain Ctrl-C on a `run`) hits
/// Rust's default disposition — immediate termination, no destructors run — losing that turn's
/// unpersisted messages and orphaning any backgrounded child process `exec`'s `kill_on_drop` cleanup
/// depends on `Drop` running to reap. SIGHUP's own default disposition is the same immediate
/// termination — a controlling terminal closing (a bare `serve` backgrounded without `nohup`/`setsid`)
/// would otherwise kill the process outright, same failure mode as an unhandled SIGTERM. Matches pi's
/// own `rpc-mode.ts`, which treats SIGHUP identically to SIGTERM (graceful shutdown, not a reload
/// trigger — unlike this crate's own `reload` RPC command, which is unrelated and client-driven).
pub struct ShutdownSignal {
    /// Set for a **supervised** session: one spawned by `serve_ws`, where the supervisor — not this
    /// session — decides when to stop, so [`wait`](ShutdownSignal::wait) must never resolve.
    ///
    /// Without this, a daemon session installs its own handler and cancels its in-flight run the
    /// instant SIGTERM lands, which is precisely the work a drain exists to protect: the supervisor
    /// would still be counting down its grace while the run it was protecting had already been
    /// aborted out from under it. Inert here, authoritative there.
    inert: bool,
    #[cfg(unix)]
    sigterm: Option<tokio::signal::unix::Signal>,
    #[cfg(unix)]
    sighup: Option<tokio::signal::unix::Signal>,
}

/// Which OS signal [`ShutdownSignal::wait`] actually observed — Task #41 (pi-parity fix): previously
/// every trigger (SIGTERM, SIGHUP, Ctrl-C/SIGINT) was merged into one undifferentiated `()`, so `serve`/
/// `run` always exited 0 (clean stdin-EOF path) or 1 (`run`'s cancelled-turn path), with no way to tell
/// *which* signal (if any) actually caused a shutdown. `serve`/`run` use this to exit with the matching
/// POSIX `128 + signal` code (see [`Self::exit_code`]) instead — matching pi's own `rpc-mode.ts`/
/// `print-mode.ts`, which register SIGTERM/SIGHUP separately and propagate 143/129 respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// `SIGTERM` — pi's own 143 (`128 + 15`).
    Term,
    /// `SIGHUP` — pi's own 129 (`128 + 1`); Unix-only (see [`ShutdownSignal`]'s own doc comment on why
    /// this crate treats it identically to `SIGTERM`).
    Hup,
    /// Ctrl-C/`SIGINT` — `130` (`128 + 2`), extending the same POSIX convention pi's own two hardcoded
    /// values derive from. pi's headless `rpc-mode.ts`/`print-mode.ts` don't register a `SIGINT` handler
    /// of their own at all (Node's own default disposition already exits 130 for it), so this is this
    /// crate's own explicit counterpart of that same outcome, not a divergence from it — this crate
    /// already treats Ctrl-C as a graceful-shutdown trigger equal to SIGTERM/SIGHUP (see
    /// [`ShutdownSignal::wait`]), so it gets the same treatment here for consistency.
    Int,
}

impl Signal {
    /// The POSIX `128 + signal-number` convention a shell reports for a process killed by this signal.
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Term => 143,
            Self::Hup => 129,
            Self::Int => 130,
        }
    }
}

impl ShutdownSignal {
    /// A signal source that never fires, for a session whose supervisor owns its lifecycle.
    pub fn inert() -> Self {
        Self {
            inert: true,
            #[cfg(unix)]
            sigterm: None,
            #[cfg(unix)]
            sighup: None,
        }
    }

    pub fn new() -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            Ok(Self {
                inert: false,
                sigterm: Some(tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::terminate(),
                )?),
                sighup: Some(tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::hangup(),
                )?),
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self { inert: false })
        }
    }

    /// Resolves once a shutdown signal arrives, naming which one. Safe to call fresh on every loop
    /// iteration — every `Signal::recv` and `tokio::signal::ctrl_c` are re-armable, not one-shot.
    pub async fn wait(&mut self) -> Signal {
        if self.inert {
            std::future::pending::<()>().await;
        }
        #[cfg(unix)]
        {
            let (Some(sigterm), Some(sighup)) = (self.sigterm.as_mut(), self.sighup.as_mut())
            else {
                std::future::pending::<()>().await;
                unreachable!("an inert signal source never resolves")
            };
            tokio::select! {
                _ = sigterm.recv() => Signal::Term,
                _ = sighup.recv() => Signal::Hup,
                _ = tokio::signal::ctrl_c() => Signal::Int,
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            Signal::Int
        }
    }
}

fn not_in_repo_mode() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "not in repo mode (start serve with --session-dir)",
    )
}

/// Listing metadata for every session under `dirs`, newest first — the same `scan_listings` walk a
/// `SessionRepo` does, over several directories at once and **without creating any of them** (a
/// listing must not mint a tenant directory on every mounted shard just by being asked). A directory
/// that doesn't exist contributes nothing.
///
/// Runs on `spawn_blocking` for the same reason `Persistence::list_with_progress` does: the scan is
/// synchronous file I/O reachable from the busy loop while a turn is in flight — and, through the
/// supervisor's `list_daemon_sessions`, from a connection task that has not attached to any session
/// at all. **Both** halves of the walk are inside the closure: a `spawn_blocking` that wraps only the
/// per-file listing parse leaves the `read_dir` on the runtime thread, and on a network mount that is
/// the half with the latency.
pub(crate) async fn scan_session_dirs(
    dirs: Vec<std::path::PathBuf>,
    layout: crate::session_store::Layout,
    on_progress: impl Fn(usize, usize) + Send + Sync + 'static,
) -> Vec<SessionMeta> {
    match crate::blocking::spawn(crate::blocking::Site::SessionList, move || {
        let paths: Vec<std::path::PathBuf> = dirs
            .iter()
            .flat_map(|dir| crate::session_store::scan_session_dir_in(dir, &layout))
            .collect();
        let mut metas = crate::session_store::scan_listings_in(paths, &layout, &on_progress);
        metas.sort_by(crate::session_store::by_recency);
        metas
    })
    .await
    {
        Ok(metas) => metas,
        // Neither `scan_session_dir_in` nor `scan_listings` panics, and this task is never
        // cancelled — same re-raise as every other `spawn_blocking` in this file.
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// Whether a session's recorded `cwd` no longer reflects reality for this process: the directory has
/// been moved or deleted since the session was created, or — since `switch_session`/`fork`/reattaching
/// to an existing session never change the process's actual working directory — it simply isn't where
/// this process is running (a shared `--session-dir` spanning multiple projects, or a session forked
/// from one created elsewhere; `SessionRepo::fork` copies the *source* session's `cwd`, not the current
/// one). Surfaced (as `cwd_stale` on the relevant responses) so a client can warn before the model's
/// tools proceed against a mismatched or nonexistent directory instead of silently producing confusing
/// results.
pub fn cwd_is_stale(meta_cwd: &str, actual_cwd: &std::path::Path) -> bool {
    !std::path::Path::new(meta_cwd).is_dir() || meta_cwd != actual_cwd.to_string_lossy()
}

/// The model's current `todo` list for this session, or `None` if it never made one.
///
/// Two sources, in priority order, because the `todo` tool holds no state of its own (its registry is
/// rebuilt on every `set_model`, so it couldn't):
/// 1. The last `todo` `tool_use` block still in the live transcript.
/// 2. `Session::compaction.todos` — where a compaction folded the plan when it dropped the block that
///    carried it. Persisted with the session, so this is also what a resumed session recovers from.
///
/// Checked in that order: a call in the live suffix is always newer than anything a past compaction
/// carried forward.
fn current_todos(session: &Session) -> Option<Value> {
    agent_core::compaction::extract_todos(&session.messages)
        .or_else(|| session.compaction.todos.clone())
}

/// Rebuild the exec target a session recorded on its own header (`meta.exec_endpoint`, written by
/// `set_exec_endpoint`). `None` if it recorded none, or if what it recorded no longer resolves.
///
/// One function, two call sites, on purpose: a **warm** re-point (`switch_session`/`new_session`/
/// fork/clone, via `reset_exec_endpoint!`) and a **cold** one (process start, below). They used to
/// differ — only the warm path restored the record — so a daemon restart followed by a reconnect to
/// the same `?session_id=` silently ran that session's tools on the host instead of in its sandbox,
/// which is precisely the "moves a tenant's work onto the server" failure the record exists to
/// prevent.
///
/// A target that no longer resolves leaves the session on the host with the record still on its
/// header, rather than failing the start: the sandbox may simply be gone, and a session that cannot
/// be opened at all is worse than one that has to be re-pointed.
///
/// Never used in service mode — the grant names the sandbox there, and `set_exec_endpoint` is refused
/// outright (see [`crate::service::refused_command`]), so no service session can have a record of its
/// own to restore. Credentials are not part of the record either: only the header *names* are
/// persisted, never their values.
async fn restore_persisted_exec_endpoint(
    spec: &Value,
    max_response_bytes: usize,
) -> Option<crate::exec_endpoint::ExecTarget> {
    let url = spec.get("url").and_then(Value::as_str);
    let cmd = spec.get("command").and_then(Value::as_str);
    let headers: Vec<String> = spec
        .get("headers")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    match (url, cmd) {
        (Some(u), _) => crate::exec_endpoint::ExecTarget::http(u, &headers, max_response_bytes)
            .await
            .ok(),
        (None, Some(c)) => crate::exec_endpoint::ExecTarget::template(c).await.ok(),
        (None, None) => None,
    }
}

/// Resolve the model/thinking-level `serve` actually starts with (Task #5, pi-parity fix). An explicit
/// `--model`/`--reasoning-effort` for *this* invocation (`model_explicit`/`reasoning_effort_explicit` —
/// see [`ServeConfig::model_explicit`]'s doc comment) always wins for its own half; otherwise the
/// reattached session's own last-recorded value (`session_model`/`session_level`, from
/// [`Persistence::model_and_level_at_active`]) wins instead of the CLI-resolved default
/// (`cfg_model`/`cfg_level`) — the same "bleed" `switch_session`/`switch_branch` are already hardened
/// against, previously left unguarded at ordinary process restart. Each half is resolved independently,
/// matching `--model`/`--reasoning-effort` being independent flags. `session_model`/`session_level` are
/// expected to already carry `model_and_level_at_active`'s own fallback-to-`cfg`-when-nothing-recorded
/// behavior (see that method's doc comment), so this is a no-op for a genuinely fresh session or pure
/// in-memory mode regardless of which flags were explicit. Pulled out as a pure function — no
/// `Persistence`/gateway needed — so the precedence itself is unit-testable on its own.
fn resolve_startup_model_and_level(
    cfg_model: &str,
    cfg_level: agent_core::ThinkingLevel,
    model_explicit: bool,
    reasoning_effort_explicit: bool,
    session_model: String,
    session_level: agent_core::ThinkingLevel,
) -> (String, agent_core::ThinkingLevel) {
    (
        if model_explicit {
            cfg_model.to_string()
        } else {
            session_model
        },
        if reasoning_effort_explicit {
            cfg_level
        } else {
            session_level
        },
    )
}

/// Run the control loop until stdin closes.
/// Headless `serve` over **stdio** — the default transport. A thin wrapper over the transport-
/// agnostic [`serve_session`]: pump stdin lines into its input channel (EOF drops the sender, which
/// closes the channel and shuts the session down — byte-identical to the pre-refactor loop), and
/// drain its single "connection" to stdout. One session, one permanent connection. Every existing
/// `serve` behavior and test rides this path unchanged; [`crate::serve_ws`] is the parallel WebSocket
/// entry point that reuses the same [`serve_session`] core.
pub async fn serve(cfg: ServeConfig) -> Result<Option<Signal>, Box<dyn std::error::Error>> {
    let (input_tx, input_rx) = mpsc::channel::<String>(IN_CHANNEL_BOUND);
    let (conn_tx, mut conn_rx) = mpsc::unbounded_channel::<OutFrame>();
    // stdio has exactly one permanent "connection": stdout. Register it in the fanout as the sole sink.
    let out_conn: SharedOutConn =
        Arc::new(std::sync::Mutex::new(crate::serve::OutFanout::default()));
    lock_ignoring_poison(&out_conn).add(crate::serve::OutSink::Unbounded(conn_tx));

    // Weak: the stdout task ends when the last strong handle to its sink goes, at session end, and a
    // reader parked on a stdin that never closes (a signal exit) must not hold it open.
    let reader_out = Arc::downgrade(&out_conn);
    tokio::spawn(async move {
        let mut stdin = BufReader::new(tokio::io::stdin());
        let mut buf = Vec::new();
        // Only end of input (or a read error, which leaves nothing more to read) ends the session. A
        // line that can't be taken as a command is answered and skipped, like invalid JSON is: one
        // bad byte from a client must not end a session silently, with exit 0 and nothing said.
        while let Ok(Some(line)) = next_stdin_line(&mut stdin, &mut buf, MAX_COMMAND_BYTES).await {
            let error = match line {
                StdinLine::Text(line) => {
                    // Awaits rather than dropping when the queue is full: a command the session never
                    // sees is a correctness bug (the client waits on a response that never comes), so
                    // the right answer to a producer outrunning the loop is to stop reading it, not to
                    // shed it. Here that simply pauses the stdin reader.
                    if input_tx.send(line).await.is_err() {
                        break; // the session ended; nothing left to feed
                    }
                    continue;
                }
                StdinLine::NotUtf8(e) => format!("stdin line is not valid UTF-8: {e}"),
                StdinLine::TooLong(n) => format!(
                    "stdin line of {n} bytes exceeds the {MAX_COMMAND_BYTES}-byte limit; discarded"
                ),
            };
            // Answered from here, as the session answers a line that isn't JSON (`command: "parse"`,
            // no id — there is no command to take one from). Sent straight to stdout's sink rather
            // than through the session, so it can land ahead of replies to commands still queued.
            let Some(out) = reader_out.upgrade() else {
                break; // the session ended
            };
            lock_ignoring_poison(&out).broadcast(response(
                None,
                "parse",
                false,
                None,
                Some(&error),
            ));
        }
        // Dropping `input_tx` closes `input_rx` → the session observes EOF and shuts down.
    });
    let stdout_task = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        // Reused across frames: the `Bytes` line plus its `\n` terminator, assembled once so the frame
        // still goes out as a single write. The one small copy here (a single local sink) is well worth
        // the full per-event line clone the fan-out path no longer pays now that `Raw` is `Bytes`.
        let mut scratch: Vec<u8> = Vec::new();
        while let Some(frame) = conn_rx.recv().await {
            let Some(line) = frame_to_line(frame) else {
                continue;
            };
            scratch.clear();
            scratch.extend_from_slice(&line);
            scratch.push(b'\n');
            if let Err(e) = write_frame(&mut out, &scratch).await {
                eprintln!("serve: stdout write failed, shutting down writer: {e}");
                break;
            }
        }
    });

    // Stdio has no supervisor and no reaper, so the `running` flag is inert here — a throwaway.
    let running = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let keep_alive = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sig = serve_session(cfg, input_rx, out_conn, running, keep_alive)
        .await
        .map_err(|e| e.to_string())?;
    // The session has ended (and dropped its `out_conn` clones), so `conn_rx` is now closed — this
    // just reaps the stdout task after it flushes the last frame.
    let _ = stdout_task.await;
    Ok(sig)
}

/// The transport-agnostic control loop: one session, driven by a stream of command lines
/// (`input_rx`) and emitting frames to whatever connection is currently attached (`out_conn`). The
/// stdio [`serve`] wrapper feeds it (stdin lines / a stdout task); [`crate::serve_ws`] feeds it from a
/// WebSocket or a one-shot HTTP POST, where the connection can detach and a later one re-attach to
/// this same still-running
/// task — because `input_rx`'s `Sender` is held by the supervisor (not the socket), a dropped
/// connection is *not* an EOF: `input_rx.recv()` simply pends until the next command, and the run
/// keeps going. The command protocol below is byte-identical across all three transports.
///
/// The future is `Send` (`Box<dyn Error + Send + Sync>` plus a `Send` event sink) so the WebSocket
/// daemon can `tokio::spawn` it onto the process-wide runtime instead of pinning a thread and a
/// current-thread executor per session.
pub(crate) async fn serve_session(
    cfg: ServeConfig,
    input_rx: mpsc::Receiver<String>,
    out_conn: SharedOutConn,
    // Set `true` for the duration of a `prompt` run, `false` otherwise. The daemon supervisor's idle
    // reaper reads this to never reap a session with an in-flight background run (see
    // [`crate::serve_ws`]); the stdio wrapper passes a throwaway it never observes.
    running: Arc<std::sync::atomic::AtomicBool>,
    // Set by the MCP Events hub while this session holds a live subscription; the daemon's idle
    // reaper leaves such a session alone (unless `--mcp-events-reapable`). Inert over stdio.
    keep_alive: Arc<std::sync::atomic::AtomicBool>,
) -> Result<Option<Signal>, Box<dyn std::error::Error + Send + Sync>> {
    let mut session_loop = SessionLoop::new(cfg, input_rx, out_conn, running, keep_alive).await?;
    // If this already fails the writer never started; there is nothing to serve.
    if !session_loop.announce_ready() {
        let _ = session_loop.writer.await;
        return Ok(None);
    }
    // A supervised session never acts on a signal itself: `serve_ws` drives the drain, and a session
    // that cancelled its own run on SIGTERM would defeat the grace the supervisor is counting down.
    session_loop.shutdown = if session_loop.cfg.supervised {
        ShutdownSignal::inert()
    } else {
        ShutdownSignal::new()?
    };
    session_loop.run().await;
    session_loop.finish().await
}

/// The state of one `serve` session's control loop: everything the command arms read and mutate,
/// built once by [`SessionLoop::new`], driven by [`SessionLoop::run`] and ended by
/// [`SessionLoop::finish`]. The commands that used to be inline helpers of one giant function
/// (`new_session`, `switch_session`, `fork`, `clone`, `compact`, …) are methods on it.
struct SessionLoop {
    cfg: ServeConfig,
    persistence: Persistence,
    session: Session,
    title_attempted_for: Option<String>,
    mode: Mode,
    cwd: std::path::PathBuf,
    disk_overrides: bool,
    project_trusted: bool,
    exec_cell: crate::exec_endpoint::ExecCell,
    bash_jobs: Arc<crate::tools::bash_jobs::BashJobs>,
    bash_jobs_wake: tokio::sync::watch::Receiver<u64>,
    host_bash_jobs: Arc<crate::tools::bash_jobs::BashJobs>,
    bash_jobs_transcript: u64,
    mcp_enabled: crate::tools::mcp::McpEnabledSet,
    mcp_host: Arc<crate::tools::mcp_host::McpHost>,
    mcp_events: Option<crate::tools::mcp_events::McpEventsHub>,
    mcp_skills: crate::tools::mcp_skills::McpSkills,
    job_notify: Option<Arc<crate::tools::job_notify::SessionNotify>>,
    sandbox_refresh: bool,
    sandbox_snapshot_dirty: bool,
    sandbox_resources: Option<crate::service::Resources>,
    skills: Vec<crate::skills::Skill>,
    skill_collisions: Vec<crate::skills::Collision>,
    prompt_templates: Vec<crate::prompts::PromptTemplate>,
    prompt_collisions: Vec<crate::skills::Collision>,
    context_files: Vec<(String, String)>,
    sandbox_system: Option<String>,
    sandbox_append: Option<String>,
    has_read: bool,
    has_todo: bool,
    has_memory: bool,
    session_dir_cell: crate::memory::file::SessionDir,
    mounts: Vec<crate::memory::Mount>,
    memory_tool: Option<Arc<dyn agent_core::Tool>>,
    session_memory_active: bool,
    memory_sections: Vec<(crate::memory::MountKind, String)>,
    apps_epoch: Arc<std::sync::Mutex<u64>>,
    pending_approvals: PendingApprovals,
    approval: Option<crate::approval::ApprovalRuntime>,
    pending_elicitations: PendingElicitations,
    pending_samplings: serve_sampling::PendingSamplings,
    app_sink: crate::tools::mcp_apps::AppSink,
    app_redact: crate::tools::mcp_apps::Redactor,
    app_detached: crate::tools::mcp_apps::Detached,
    app_filter: crate::tools::mcp_apps::ToolFilter,
    apps_generation: Option<u64>,
    mcp_tools_generation: u64,
    mcp_filter: crate::tools::McpFilter,
    output_slot: crate::tools::structured_output::OutputSlot,
    structured_output: Option<Arc<crate::tools::structured_output::StructuredOutput>>,
    current_output_spec: Option<(Value, Option<String>)>,
    static_system: String,
    run_retry_policy: crate::retry::RunRetryPolicy,
    current_model: String,
    scoped_models: Vec<ScopedModel>,
    cycle_models: Vec<String>,
    current_thinking: Option<u32>,
    starting_level: agent_core::ThinkingLevel,
    current_level: agent_core::ThinkingLevel,
    settings_store: Option<crate::settings::SettingsStore>,
    current_auto_compaction: bool,
    current_auto_retry: bool,
    current_block_images: bool,
    current_image_auto_resize: bool,
    write_locks: Arc<agent_core::WriteLockRegistry>,
    subagent_ctx: Option<Arc<crate::tools::subagent::SubagentCtx>>,
    checkpoint_rx: mpsc::UnboundedReceiver<Checkpoint>,
    checkpoint: Arc<dyn agent_core::CheckpointHook>,
    mcp_task_tx: mpsc::UnboundedSender<(String, Value)>,
    mcp_task_rx: mpsc::UnboundedReceiver<(String, Value)>,
    client: Arc<GatewayClient>,
    agent: Agent,
    steering: agent_core::Steering,
    steer_done_rx: mpsc::UnboundedReceiver<SteerExpanded>,
    pending_steers: PendingSteers,
    bash_tool: Option<Arc<dyn agent_core::Tool>>,
    bash_policy: crate::policy::ToolPolicy,
    host_bash_seq: u32,
    out_tx: OutTx,
    writer: tokio::task::JoinHandle<()>,
    mcp_resumer: crate::mcp_resume::Resumer,
    last_history: Arc<Vec<agent_core::Message>>,
    app_state_for: Option<std::path::PathBuf>,
    saved_app_context: String,
    saved_app_views: u64,
    shutdown: ShutdownSignal,
    shutdown_cause: Option<Signal>,
    deferred_events: std::collections::VecDeque<String>,
    events_state_file: Option<std::path::PathBuf>,
    events_cmds: Option<crate::tools::mcp_events::McpEventsCommands>,
    pending_login: Arc<std::sync::Mutex<Option<PendingLogin>>>,
    login_generation: u64,
    input_rx: mpsc::Receiver<String>,
    out_conn: SharedOutConn,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl SessionLoop {
    /// Sends a frame through the writer; `false` if the writer has shut down (stdout closed), which
    /// the caller answers by stopping the control loop — there is no way to deliver any further
    /// response, so continuing would only swallow output.
    ///
    /// Every frame passes through `redact_frame` on the way out — inside [`OutTx::send`], which
    /// is the only way to reach the writer.
    fn emit(&self, frame: OutFrame) -> bool {
        self.out_tx.send(frame).is_ok()
    }

    /// Announce readiness so a client can sync before issuing commands. `false` if the writer is
    /// already gone.
    fn announce_ready(&self) -> bool {
        self.out_tx
            .send(
                json!({
                    "type": "ready",
                    "session_id": self.persistence.session_id(),
                    "model": self.current_model,
                    "cwd_stale": self.mode.cwd_is_stale(&self.persistence.meta.cwd, &self.cwd),
                })
                .into(),
            )
            .is_ok()
    }

    /// Starts the resumer for the session now active (see [`start_mcp_resumer`]).
    fn start_mcp_resumer(
        &self,
        carried: &[crate::mcp_resume::Resolved],
    ) -> crate::mcp_resume::Resumer {
        start_mcp_resumer(
            &self.persistence,
            &self.session,
            &self.cfg,
            &self.out_tx,
            &self.mcp_host,
            &self.mcp_task_tx,
            carried,
        )
    }

    /// Re-point the `/session` working-memory mount at whatever session is now active, after a
    /// `switch_session`/`new_session`/`fork`/`clone` has swapped `persistence` onto it. One cell write
    /// reaches the parent's `memory` tool and every subagent sharing the backend by `Arc`, so neither the
    /// tool nor the agent has to be rebuilt. Harmless when the session mount is disabled (nothing reads the
    /// cell). The injected `/session` index in the system prompt refreshes at the next prompt/`set_model`
    /// rebuild; the tool's own live `view /session` is authoritative in the meantime.
    /// A session's exec target must NOT survive a switch to a different session. Inheriting it would
    /// hand the incoming session — in a multi-tenant server, a different tenant — the previous one's
    /// sandbox, which is a cross-tenant breach rather than stale config. Cleared to `None` (this host)
    /// and re-established by `set_exec_endpoint`, so the failure mode of forgetting to re-point is
    /// "runs locally", not "runs in somebody else's box".
    async fn reset_exec_endpoint(&self) {
        // Clear first, unconditionally: whatever the incoming session turns out to want, it must
        // never keep the outgoing session's target even briefly.
        self.exec_cell.set(None);
        // Service mode re-applies the **grant's** target, never a persisted one: the recorded
        // spec belongs to whichever grant last ran this session, and its credentials are not
        // persisted at all (see below). Lazily, as at startup: the incoming session probes it
        // again on its first use, and nothing is sent for the switch itself.
        if let Some(connect) = self.mode.lazy_exec(self.cfg.exec_max_response_bytes) {
            self.exec_cell.set_lazy(connect, self.mode.sandbox_home());
        } else if let Some(spec) = self.persistence.meta.exec_endpoint.clone() {
            // A target that no longer resolves leaves the session on the host with the header
            // still recorded, rather than failing the switch — the sandbox may simply be gone.
            self.exec_cell.set(
                restore_persisted_exec_endpoint(&spec, self.cfg.exec_max_response_bytes).await,
            );
        }
    }

    fn reset_mcp_enabled(&self) {
        // Same cross-tenant rule as exec: an incoming session must not inherit the previous
        // session's MCP kit — its enablement, nor the skills it loaded or approved. Default back
        // to all configured servers enabled — and to no MCP Apps: the incoming session's client
        // re-declares (`set_mcp_apps`) if it renders them.
        self.mcp_enabled.reset_session();
        *lock_ignoring_poison(&self.apps_epoch) += 1;
        crate::tools::mcp_apps::close(&self.mcp_enabled, "session_switched");
        lock_ignoring_poison(&self.out_conn).clear_app_views();
    }

    fn repoint_session_memory(&self) {
        self.session_dir_cell.set(self.persistence.memory_dir());
    }

    /// The approval memory is session-scoped (see [`crate::approval::SessionMemory`]), so an
    /// "always allow" granted in the outgoing conversation must not carry into the incoming one —
    /// the gate would never ask, and the operator who granted it was answering about a different
    /// session's commands and paths. Cleared through the shared `Arc`, never swapped: the
    /// `ApprovalRuntime` is already cloned by value into every subagent, and a same-model
    /// `switch_session` skips `build_agent` entirely.
    ///
    /// `fork`/`clone` deliberately do **not** call this, unlike the exec and MCP resets beside it.
    /// Those two are cross-tenant rules — an incoming session's *machine* and *tools* must never be
    /// inherited — and a fork cannot cross a tenant, since it derives from the session already open.
    /// It also carries this same conversation forward, so its remembered decisions still describe
    /// exactly the commands and paths on screen; re-asking for them would be noise, not safety.
    fn reset_approval_memory(&self) {
        if let Some(runtime) = self.approval.as_ref() {
            runtime.memory.clear();
        }
    }

    fn sync_history(&mut self) {
        let changed = !Arc::ptr_eq(&self.session.messages, &self.last_history);
        if changed {
            self.last_history = self.session.messages.clone();
        }
        if let Some(file) = self
            .persistence
            .session_file()
            .map(std::path::Path::to_path_buf)
        {
            let seal = self.mode.sidecar_seal(self.persistence.session_id());
            let ctx = crate::tools::mcp_apps::Sidecar::new(
                &file,
                crate::tools::mcp_apps::Sidecar::CONTEXT,
                seal.clone(),
            );
            let views = crate::tools::mcp_apps::Sidecar::new(
                &file,
                crate::tools::mcp_apps::Sidecar::VIEWS,
                seal,
            );
            if self.app_state_for.as_deref() != Some(file.as_path()) {
                self.app_state_for = Some(file.clone());
                if let Some(records) = ctx.read() {
                    crate::tools::mcp_apps::attach_context_records(&records, &mut self.session);
                }
                // Forces one write, so context carried in by a fork/clone lands in its sidecar.
                self.saved_app_context = String::new();
                let mut fanout = lock_ignoring_poison(&self.out_conn);
                if let Some(bytes) = views.read() {
                    fanout.restore_app_views(&bytes);
                }
                self.saved_app_views = fanout.app_views_version();
            } else if changed && self.session.request_blocks.is_empty() {
                // The same session's transcript was reloaded (a branch switch): re-attach.
                crate::tools::mcp_apps::attach_context_records(
                    self.saved_app_context.as_bytes(),
                    &mut self.session,
                );
            }
            // A store another owner took over is not ours to write.
            if !self.persistence.superseded() {
                let records = crate::tools::mcp_apps::context_records(&self.session);
                if records != self.saved_app_context {
                    let body = (records != "[]").then_some(records.as_bytes());
                    match ctx.write(body) {
                        Ok(()) => self.saved_app_context = records,
                        Err(e) => eprintln!("serve: could not save MCP App view context: {e}"),
                    }
                }
                let (version, snapshot) = {
                    let fanout = lock_ignoring_poison(&self.out_conn);
                    let version = fanout.app_views_version();
                    (
                        version,
                        (version != self.saved_app_views).then(|| fanout.app_views_snapshot()),
                    )
                };
                if let Some(snapshot) = snapshot {
                    match views.write(snapshot.as_deref()) {
                        Ok(()) => self.saved_app_views = version,
                        Err(e) => eprintln!("serve: could not save MCP App views: {e}"),
                    }
                }
            }
        }
        let mut fanout = lock_ignoring_poison(&self.out_conn);
        if changed {
            fanout.set_history(self.last_history.clone(), self.persistence.active_ids());
        }
        // Idle, so any run that was recording has finished and its messages are in the history just
        // published. Drop the recording under this *same* lock: a connection attaching must never see
        // the new history and then have the finished turn's frames replayed on top of it.
        fanout.end_turn();
    }

    /// Install a freshly discovered resource set: the lists `get_commands` and prompt expansion read,
    /// the subagent context, the static system prompt, and the agent built over them. `reload` and
    /// the post-reopen refresh share it, so the two cannot drift.
    async fn apply_discovered(&mut self, found: Discovered) {
        let mut discovered: Discovered = found;
        self.skills = std::mem::take(&mut discovered.skills);
        self.skill_collisions = std::mem::take(&mut discovered.skill_collisions);
        self.prompt_templates = std::mem::take(&mut discovered.prompt_templates);
        self.prompt_collisions = std::mem::take(&mut discovered.prompt_collisions);
        self.context_files = std::mem::take(&mut discovered.context_files);
        self.sandbox_system = discovered.system.take();
        self.sandbox_append = discovered.append_system.take();
        // Agent definitions are trust-gated like skills, so a `reload` after a trust change (or
        // an edit to `.claude/agents/`) must re-discover them and rebuild the subagent context —
        // otherwise `<available_agents>` and the `subagent` tool would advertise a stale set until
        // restart. The `set_model` arm below then rebuilds the agent with the refreshed ctx.
        self.cfg.agents = std::mem::take(&mut discovered.agents);
        self.subagent_ctx = if self.cfg.agents.is_empty() {
            None
        } else {
            Some(build_subagent_ctx(
                &self.exec_cell,
                &self.cfg,
                &self.cwd,
                self.project_trusted,
                &self.current_model,
                &self.write_locks,
                &self.skills,
                self.mode.context_files(self.context_files.as_slice()),
                self.mounts.clone(),
                self.approval.as_ref(),
                &self.mcp_enabled,
            ))
        };
        self.memory_sections = crate::memory::mount_sections(&self.mounts).await;
        self.static_system = crate::resources::build_static_system_prompt_with_context(
            &crate::resources::PromptOptions {
                base: self.sandbox_system.as_deref(),
                default_base: &self.cfg.system,
                append: self
                    .cfg
                    .append_system
                    .as_deref()
                    .or(self.sandbox_append.as_deref()),
                cwd: &self.cwd,
                include_context_files: self.cfg.context_files,
                skills: &self.skills,
                context_window: agent_core::capabilities(&self.cfg.model).context_window,
                has_read: self.has_read,
                has_todo: self.has_todo,
                has_structured_output: self.structured_output.is_some(),
                has_memory: self.has_memory,
                memory_sections: &self.memory_sections,
                project_trusted: self.project_trusted,
                agents: &self.cfg.agents,
                disk_overrides: self.disk_overrides,
            },
            &self.context_files,
        );
        // A full rebuild, not just `agent.set_system(...)`: `reload` may have changed the agent
        // *definitions* (a new `.claude/agents/*.md`, or trust newly granted), and the `subagent`
        // tool's registration lives in the registry, which only a rebuild refreshes. Without this,
        // a mid-session-added agent wouldn't become delegable until the next `set_model`. Mirrors
        // `set_model`'s own rebuild; the session is untouched.
        self.agent = build_agent(
            &self.exec_cell,
            self.client.clone(),
            &full_system(&self.static_system, &self.cwd, &self.mcp_skills),
            &self.cfg,
            &self.current_model,
            self.current_thinking,
            self.current_level,
            self.current_auto_compaction,
            self.current_auto_retry,
            self.current_block_images,
            self.current_image_auto_resize,
            self.persistence.session_id(),
            &self.write_locks,
            &self.checkpoint,
            self.subagent_ctx.as_ref(),
            self.structured_output.as_ref(),
            self.memory_tool.as_ref(),
            self.approval.as_ref(),
            &self.mcp_enabled,
            &self.skills,
        );
    }

    /// The session is over, however it ended (EOF, a reap, a supersede, shutdown): stop what it owns.
    async fn finish(mut self) -> Result<Option<Signal>, Box<dyn std::error::Error + Send + Sync>> {
        // The session is over, however it ended (EOF, a reap, a supersede, shutdown): its `bash` jobs
        // stop with it. Explicit rather than left to the table's drop: in a daemon the process lives
        // on, and nothing else would ever reach these jobs again. Off the runtime thread, because each
        // kill blocks until its processes are gone (on the daemon's current-thread runtime that would
        // stall every other session).
        let stopped_jobs = {
            let jobs = self.bash_jobs.clone();
            let host_jobs = self.host_bash_jobs.clone();
            crate::blocking::spawn(crate::blocking::Site::Serve, move || {
                // The client's are stopped too, but not reported as the model's jobs.
                host_jobs.stop_all();
                // Not ended, only closed: the jobs the sandbox owns run on for the next replica to
                // reattach. Everything this process owns stops.
                jobs.release()
            })
            .await
            .unwrap_or_default()
        };
        if !stopped_jobs.is_empty() {
            // Best effort: the client may already be gone.
            let _ = self.out_tx.send(OutFrame::Value(json!({
                "type": "bash_jobs_stopped",
                "jobs": stopped_jobs,
            })));
        }
        // Unsubscribe (webhook) / cancel (push) every MCP Events subscription before the session goes:
        // best effort and bounded, so a dead server cannot hold the teardown.
        if let Some(hub) = self.mcp_events.take() {
            hub.shutdown().await;
        }

        // A `login` still in flight owns a detached task holding its own clone of `out_tx` — and the writer
        // below ends *only* once every sender is gone. Without this cancel, a client that starts a login and
        // then simply disconnects (or a SIGTERM arriving mid-flow) parks that task forever, `writer.await`
        // never returns, and `serve_session` never completes: the OS thread, its runtime, the `Agent`, the
        // `GatewayClient` and its pool, the session history, and the OAuth callback server's bound — and
        // *fixed* — port all leak for the daemon's lifetime, which also makes every subsequent login on that
        // port fail outright. In stdio `serve` the same hang means the process can never exit gracefully.
        //
        // `abort_login` is not a substitute: it only fires when a client explicitly sends it, which is
        // exactly what an abandoned login never does. Teardown is the path that has to be unconditional.
        if let Some(p) = lock_ignoring_poison(&self.pending_login).take() {
            p.cancel.cancel();
        }
        // The same hazard: a resumed MCP task still polling holds an `out_tx` clone for its progress.
        // Dropping the resumer aborts its tasks (never cancelling the MCP tasks themselves, which stay
        // journaled for the next start), so the writer can end.
        drop(self.mcp_resumer);
        drop(self.out_tx);
        let _ = self.writer.await;
        Ok(self.shutdown_cause)
    }
}

/// A runtime toggle writes through to the persisted defaults — except in service mode, where the
/// in-memory change is the whole effect (it applies for this session's life and nothing else sees
/// it, and `settings_store` is `None`).
fn persist_setting(
    settings_store: &mut Option<crate::settings::SettingsStore>,
    what: &str,
    write: impl FnOnce(&mut crate::settings::SettingsStore) -> std::io::Result<()>,
) {
    if let Some(store) = settings_store.as_mut()
        && let Err(e) = write(store)
    {
        eprintln!("serve: failed to persist {what} setting: {e}");
    }
}

/// Appends an MCP task-journal entry to the session it names (see `crate::mcp_resume`).
fn journal_mcp(persistence: &mut Persistence, kind: &str, data: Value) {
    if data["sessionId"].as_str() == Some(persistence.session_id()) {
        match persistence.append_journal(kind, data) {
            Err(e) if e.kind() != std::io::ErrorKind::Unsupported => {
                tracing::warn!(error = %e, "MCP task journal entry failed to persist");
            }
            _ => {}
        }
    }
}

/// MCP tasks a previous process left in flight resume now, in the background, not on the next
/// prompt (see `crate::mcp_resume`). Restarted for a session switched to later.
///
/// `carried`: results the previous resumer already resolved, which count as answers whether or
/// not their journal writes have landed yet (they travel through `mcp_task_rx`, which a prompt
/// can overtake), so a restart never polls a finished task again.
fn start_mcp_resumer(
    persistence: &Persistence,
    session: &Session,
    cfg: &ServeConfig,
    out_tx: &OutTx,
    mcp_host: &Arc<crate::tools::mcp_host::McpHost>,
    mcp_task_tx: &mpsc::UnboundedSender<(String, Value)>,
    carried: &[crate::mcp_resume::Resolved],
) -> crate::mcp_resume::Resumer {
    let journal = persistence.active_journal(crate::mcp_resume::TASK_ENTRY_KIND);
    let mut results = crate::mcp_resume::results(
        &persistence.active_journal(crate::mcp_resume::RESULT_ENTRY_KIND),
    );
    results.extend_from_slice(carried);
    let placeholders = crate::mcp_resume::placeholder_ids(
        &persistence.active_journal(crate::mcp_resume::PLACEHOLDER_ENTRY_KIND),
    );
    let mut pending = crate::mcp_resume::pending(
        &session.messages,
        &journal,
        &results,
        &placeholders,
        persistence.session_id(),
    );
    // Only a configured server is ever polled.
    pending.retain(|p| cfg.mcp_catalog.has_server(&p.record.server));
    if pending.is_empty() {
        crate::mcp_resume::Resumer::idle(persistence.session_id())
    } else {
        let out = out_tx.clone();
        crate::mcp_resume::Resumer::start(
            persistence.session_id(),
            pending,
            cfg.mcp_catalog.clone(),
            mcp_host.clone(),
            Arc::new(move |ev| {
                if let Some(frame) = event_frame(ev) {
                    let _ = out.send(frame);
                }
            }),
            mcp_task_tx.clone(),
        )
    }
}

/// `mcp_app_request` proxies a view's call to its server, which may take as long as any tool
/// call. Spawned, so neither command loop (busy or idle) waits on it.
#[allow(clippy::too_many_arguments)]
fn spawn_app_request(
    id: Option<String>,
    cmd: Value,
    out_tx: &OutTx,
    enabled: &crate::tools::mcp::McpEnabledSet,
    filter: &crate::tools::mcp_apps::ToolFilter,
    approval: &Option<crate::approval::ApprovalRuntime>,
    detached: &crate::tools::mcp_apps::Detached,
    session_host: &Arc<crate::tools::mcp_host::McpHost>,
) {
    let enabled = enabled.clone();
    let filter = filter.clone();
    let approval = approval.clone();
    let detached = detached.clone();
    let out_tx = out_tx.clone();
    let session_host = session_host.clone();
    tokio::spawn(async move {
        // Scoped to this session's host, like a run: a question the server asks during a
        // view's call is answered by this session (or refused when ambiguous).
        let request =
            crate::tools::mcp_apps::handle_request(enabled, filter, approval, detached, cmd);
        let frame = match crate::tools::mcp::with_session_host(session_host, request).await {
            Ok(result) => response(
                id,
                "mcp_app_request",
                true,
                Some(json!({ "result": result })),
                None,
            ),
            Err(e) => response(id, "mcp_app_request", false, None, Some(&e)),
        };
        let _ = out_tx.send(frame);
    });
}

/// Routes an `mcp_events_*` command: to the hub's dispatcher, or — with no MCP server connected —
/// straight to an error response.
fn dispatch_events(
    events_cmds: &Option<crate::tools::mcp_events::McpEventsCommands>,
    out_tx: &OutTx,
    id: Option<String>,
    ctype: &str,
    c: &Value,
) {
    match events_cmds {
        Some(d) => d.dispatch(id, ctype, c.clone()),
        None => {
            let _ = out_tx.send(response(
                id,
                ctype,
                false,
                None,
                Some("no MCP server is connected, so there are no MCP events to subscribe to"),
            ));
        }
    }
}

/// Join the cached static system prompt with a freshly-computed dynamic footer (current date/cwd) —
/// the full prompt text an `Agent` should carry. Cheap: `static_system` is already-computed text and
/// `dynamic_footer` does no filesystem discovery, so this is safe to call every turn (see the `prompt`
/// arm's per-turn refresh) as well as at every `build_agent` rebuild.
fn full_system(
    static_system: &str,
    cwd: &std::path::Path,
    mcp_skills: &crate::tools::mcp_skills::McpSkills,
) -> String {
    // MCP-served skills are rendered here rather than baked into `static_system`: which of them are
    // visible follows the session's live `set_mcp_enabled` gate, and this runs at every rebuild and
    // every turn, so a gate change (or a session switch resetting it) can never leave the listing
    // stale.
    let mcp = mcp_skills.format_available();
    let sep = if mcp.is_empty() { "" } else { "\n\n" };
    format!(
        "{static_system}{sep}{mcp}{}",
        crate::resources::dynamic_footer(cwd)
    )
}

/// The `output_schema`/`output_description` a `prompt` command asks for, as the pair the session
/// compares against what it already has installed. `None` means "answer in prose", the default.
///
/// An explicit `output_schema: null` is the same as omitting it — that is how a client *removes* a
/// schema installed by an earlier prompt on the same session. Anything else non-object is a client bug
/// and is rejected here rather than reaching `StructuredOutput::new`, which would then complain about
/// the wrong thing.
type OutputSpec = Option<(Value, Option<String>)>;

fn parse_output_spec(cmd: &Value) -> Result<OutputSpec, String> {
    let schema = match cmd.get("output_schema") {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    if !schema.is_object() {
        return Err("`output_schema` must be a JSON Schema object".to_string());
    }
    let description = match cmd.get("output_description") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err("`output_description` must be a string".to_string()),
    };
    Ok(Some((schema.clone(), description)))
}

/// Text prefix stamped on a host `bash` RPC result recorded with `exclude_from_context: true` — this
/// crate's stand-in for pi's dedicated `BashExecutionMessage.excludeFromContext` field
/// (`packages/coding-agent/src/core/messages.ts`): `agent_core::Message` is wire-shaped 1:1 (see its own
/// module doc comment), with no side channel for a per-message "hide from the model, but not from
/// history" flag the way pi's own richer `AgentMessage` union has. The label is never user-influenced —
/// it precedes `$ {command}` entirely, so a command's own text (or its output) can't forge it — and is
/// consulted only by [`ServeHooks::before_provider_request`], the one place this crate's own "what's
/// actually sent to the model" transform lives.
const HOST_BASH_EXCLUDED_LABEL: &str = "[Host bash command, excluded from model context]";

/// Returned on a `steer`/`follow_up` (or a `prompt` routed into one of those lanes) whose queue is at
/// its cap. The lanes are bounded because they're fed straight from remote RPC and drain at most one
/// message per turn boundary — see `agent_core::steering`'s own cap. The client is told so explicitly:
/// a lane that refuses the newest message while its ack still claims `success: true` would leave a
/// client with no signal at all that its instruction never reached the model, which is the one outcome
/// it cannot recover from on its own. The accompanying `queue_content` payload shows what *is* queued,
/// so a client can see the depth it's up against and retry once the agent drains.
const STEERING_QUEUE_FULL: &str = "steering queue full; retry once the agent drains a message";

/// The ordinary (not excluded) counterpart to [`HOST_BASH_EXCLUDED_LABEL`] — unchanged from before Fix 9.
const HOST_BASH_LABEL: &str = "[Host bash command, run outside the model's own turn]";
/// Prefix for the structured `exit_code`/`cancelled`/`truncated`/`full_output_path` status line the
/// `bash` RPC command now writes into the persisted host-bash message, right after the blank-line
/// separator and before the legacy `"(error)\n"` marker (Fix 3, pi-parity gap). Same "ride on the
/// self-describing marker text, don't widen `agent_core::Message`" reasoning as
/// [`HOST_BASH_EXCLUDED_LABEL`] above — this RPC's own response has always carried these four fields
/// live (a few lines below), but nothing threaded them into what actually gets persisted/exported until
/// now. `export.rs`'s own copy of this constant (`~export.rs:HOST_BASH_STATUS_LINE_PREFIX`) is what
/// parses it back out at export time; a session written before this line existed simply has none, which
/// `export.rs`'s parser treats exactly like any other line that doesn't start with this prefix — falls
/// back to the old `"(error)\n"`-marker-only detection, no migration needed.
const HOST_BASH_STATUS_LINE_PREFIX: &str = "[Host bash status] ";

/// The closing record for a reopened session's prompt that its run never answered (see
/// [`close_interrupted_prompt`]).
pub const INTERRUPTED_PROMPT: &str =
    "interrupted: the previous run ended before answering this request";

/// Close the prompt a killed process left unanswered, as a session is reopened — by `run`, and by
/// `serve` on open and on `switch_session`. A process killed outright (OOM, `kill -9`, a deploy's
/// SIGKILL) leaves the session ending on its prompt (checkpointed before the model call) or on a
/// tool-results turn, with no closing record: no `Err` path ever ran. Left open, the next prompt
/// stacks a second `user` turn and the model is never told the first was cut off. A host bash record
/// is the exception: a `user` message that legitimately ends a session between prompts. Returns
/// whether a record was added; it is persisted with the next write.
pub fn close_interrupted_prompt(session: &mut agent_core::Session) -> bool {
    if session.messages.last().is_some_and(is_host_bash_record) {
        return false;
    }
    session.close_unanswered_prompt(INTERRUPTED_PROMPT)
}

/// Whether `m` is a host bash record (`serve`'s `bash` RPC, either label): a `user` message that
/// legitimately ends a session between prompts, so a reopened session ending on one was not
/// interrupted mid-turn (see [`close_interrupted_prompt`]).
pub fn is_host_bash_record(m: &agent_core::Message) -> bool {
    m.role == agent_core::Role::User
        && m.content.iter().any(|c| {
            matches!(c, agent_core::ContentBlock::Text { text, .. }
                if text.starts_with(HOST_BASH_LABEL) || text.starts_with(HOST_BASH_EXCLUDED_LABEL))
        })
}

/// Whether `m` is a [`HOST_BASH_EXCLUDED_LABEL`]-tagged message — see that constant's own doc comment.
fn is_excluded_from_model_context(m: &agent_core::Message) -> bool {
    m.role == agent_core::Role::User
        && m.content.iter().any(|c| {
            matches!(c, agent_core::ContentBlock::Text { text, .. } if text.starts_with(HOST_BASH_EXCLUDED_LABEL))
        })
}

/// The concatenated text of any message's plain-text blocks, regardless of role — pi's own
/// `extractUserMessageText` (despite the name, role-agnostic there too: it just joins every `text` part
/// of whatever content array it's given). `None` for a message with no plain-text block at all (a pure
/// tool-use/tool-result/thinking/image turn), which has nothing meaningful to report as "the message's
/// text". Shared by [`user_message_text`] (which adds the role restriction its own callers need) and
/// `fork`'s response (Fix 3, pi-parity gap — see the `"fork"` command handler).
fn message_text(msg: &agent_core::Message) -> Option<String> {
    let text: String = msg
        .content
        .iter()
        .filter_map(|b| match b {
            agent_core::ContentBlock::Text { text, .. } => Some(&**text),
            _ => None,
        })
        .collect();
    (!text.is_empty()).then_some(text)
}

/// The concatenated text of a `User`-role message, for `get_fork_messages`'s candidate list — pi's own
/// `getUserMessagesForForking`, which only ever lists user turns as fork candidates. `None` for a
/// non-`User` message too, not just a textless one — see [`message_text`] for the shared extraction.
fn user_message_text(msg: &agent_core::Message) -> Option<String> {
    (msg.role == agent_core::Role::User)
        .then(|| message_text(msg))
        .flatten()
}

/// The concatenated text of the most recent assistant message, for scripting clients that just want
/// the answer. Empty if there's no assistant turn yet.
fn last_assistant_text(session: &Session) -> String {
    session
        .messages
        .iter()
        .rev()
        .find(|m| m.role == agent_core::Role::Assistant)
        .map(|m| {
            m.content
                .iter()
                .filter_map(|b| match b {
                    agent_core::ContentBlock::Text { text, .. } => Some(&**text),
                    _ => None,
                })
                .collect::<String>()
        })
        .unwrap_or_default()
}

/// Whether a `list_sessions`/`list_all_sessions` scan's progress at `scanned`/`total` is worth putting
/// on the wire: the first file, the last, and roughly every 10% step in between — enough for a client's
/// "scanning…" indicator to move without a frame per individual file when there are thousands of them.
/// A pure function of `scanned`'s value, not of arrival order, so it stays deterministic even though
/// `list_with_progress`'s underlying scan runs in parallel across several files at once.
fn should_report_scan_progress(scanned: usize, total: usize) -> bool {
    scanned <= 1 || scanned >= total || scanned.is_multiple_of((total / 10).max(1))
}

/// Lock `m`, tolerating a poisoned mutex (a prior holder panicked mid-critical-section) by recovering
/// its last-written contents rather than propagating the poison — matches this workspace's
/// panic-free-production-code convention (no bare `.unwrap()` on a lock), and there's no invariant
/// here a partial write could actually violate (the guarded value is always a plain, independently
/// valid `Option`/replace-whole-value).
pub(crate) fn lock_ignoring_poison<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Wrap an `AgentEvent` in an `event` frame, or `None` if it can't be serialized. Returning `None`
/// (and skipping the frame) rather than emitting `{type:"event"}` with no `event` field keeps a
/// serialization bug from putting a malformed frame on the wire that a client would silently mis-read.
///
/// Serializes the envelope straight to its final JSON text in one pass (`OutFrame::Raw`) instead of
/// going through `serde_json::to_value` first — this runs once per streamed model delta (`AgentEvent::
/// Stream`), the highest-frequency frame this process ever emits, far more often than any RPC
/// `response`/`ack`. `to_value` would build a full owned `Value` tree from `ev` only for the writer
/// task to immediately walk that same tree again via its own `to_string` — twice the allocation and
/// serialization work for the one frame kind that can least afford it.
/// How long an idle session waits, once a `bash` job's notice is due, before it wakes: jobs that end
/// together (a `make -j` fan-out, a server and its watcher) then fold into one turn rather than one
/// turn each.
const BASH_JOBS_WAKE_SETTLE: std::time::Duration = std::time::Duration::from_millis(200);

/// Stop the local `bash` jobs a killed process left running on the session `persistence` just
/// opened — see [`crate::tools::bash_jobs::BashJobs::stop_survivors`]. Off the runtime thread: each
/// stop blocks until its processes are gone.
async fn stop_survivors(jobs: &Arc<crate::tools::bash_jobs::BashJobs>, persistence: &Persistence) {
    let records = persistence.active_journal(crate::tools::bash_jobs::PROCS_ENTRY_KIND);
    if records.is_empty() {
        return;
    }
    let session = persistence.session_id().to_string();
    let jobs = jobs.clone();
    let fates = crate::blocking::spawn(crate::blocking::Site::Serve, move || {
        jobs.stop_survivors(&records)
    })
    .await
    .unwrap_or_default();
    if !fates.is_empty() {
        tracing::info!(%session, ?fates, "stopped bash jobs a killed process left running");
    }
}

/// Where a session's `bash` jobs journal their processes: the session loop's journal channel,
/// stamped with `session` (the loop drops an entry for any other session).
fn procs_recorder(
    tx: &mpsc::UnboundedSender<(String, Value)>,
    session: &str,
) -> crate::tools::bash_jobs::ProcRecorder {
    let tx = tx.clone();
    let session = session.to_string();
    Arc::new(move |mut data: Value| {
        data["sessionId"] = Value::String(session.clone());
        let _ = tx.send((crate::tools::bash_jobs::PROCS_ENTRY_KIND.to_string(), data));
    })
}

/// Resolves once a parked `bash` job's finish notice is due, after [`BASH_JOBS_WAKE_SETTLE`].
/// Level-triggered: the state is re-checked every time this future is created and after every
/// change, so a change that landed while the session was busy is never missed.
async fn bash_job_notice_due(
    jobs: &crate::tools::bash_jobs::BashJobs,
    wake: &mut tokio::sync::watch::Receiver<u64>,
) {
    loop {
        // Seen before checking, so a job that ends between the check and the wait still wakes it.
        wake.borrow_and_update();
        if jobs.has_notice() {
            tokio::time::sleep(BASH_JOBS_WAKE_SETTLE).await;
            return;
        }
        if wake.changed().await.is_err() {
            // The table is gone (it never is while the session runs): never wake.
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_bash_record_is_recognized_under_either_label() {
        for label in [HOST_BASH_LABEL, HOST_BASH_EXCLUDED_LABEL] {
            let m = agent_core::Message::user(format!("{label}\n$ ls\n\nout"));
            assert!(is_host_bash_record(&m));
        }
        assert!(!is_host_bash_record(&agent_core::Message::user("a prompt")));
    }

    #[test]
    fn reopening_closes_an_unanswered_prompt_but_not_a_host_bash_record() {
        let mut session = agent_core::Session::new();
        session.user("cut off mid-turn");
        assert!(close_interrupted_prompt(&mut session));
        assert_eq!(
            session.messages.last().unwrap().error_message.as_deref(),
            Some(INTERRUPTED_PROMPT)
        );
        assert!(!close_interrupted_prompt(&mut session), "already closed");

        let mut session = agent_core::Session::new();
        session.push(agent_core::Message::user(format!(
            "{HOST_BASH_LABEL}\n$ ls\n\nout"
        )));
        assert!(
            !close_interrupted_prompt(&mut session),
            "a host bash record legitimately ends a session"
        );
    }

    #[test]
    fn resolve_startup_model_and_level_prefers_the_session_only_when_not_explicit() {
        // Task #5 (pi-parity fix): the precedence matrix `serve`'s startup relies on — each half
        // (model vs level) resolved independently, matching `--model`/`--reasoning-effort` being
        // independent flags.
        let cfg_level = agent_core::ThinkingLevel::Off;
        let session_level = agent_core::ThinkingLevel::High;

        // Neither flag explicit: both halves come from the session.
        assert_eq!(
            resolve_startup_model_and_level(
                "cfg-model",
                cfg_level,
                false,
                false,
                "session-model".to_string(),
                session_level,
            ),
            ("session-model".to_string(), session_level)
        );

        // Both explicit: both halves come from cfg, regardless of what the session recorded.
        assert_eq!(
            resolve_startup_model_and_level(
                "cfg-model",
                cfg_level,
                true,
                true,
                "session-model".to_string(),
                session_level,
            ),
            ("cfg-model".to_string(), cfg_level)
        );

        // Only `--model` explicit: model from cfg, level still from the session — an explicit model
        // override must not also silently reset the thinking level back to the process's bare default.
        assert_eq!(
            resolve_startup_model_and_level(
                "cfg-model",
                cfg_level,
                true,
                false,
                "session-model".to_string(),
                session_level,
            ),
            ("cfg-model".to_string(), session_level)
        );

        // Only `--reasoning-effort` explicit: the reverse split.
        assert_eq!(
            resolve_startup_model_and_level(
                "cfg-model",
                cfg_level,
                false,
                true,
                "session-model".to_string(),
                session_level,
            ),
            ("session-model".to_string(), cfg_level)
        );
    }

    #[test]
    fn serve_startup_restores_a_reattached_sessions_own_model_over_the_cli_default() {
        // Task #5 (pi-parity fix): a plain `serve` reattaching to an existing session — no `--model`
        // passed — must continue on whatever that session was actually last driven on
        // (`set_model gpt-5` mid-session, then a restart), not silently revert to the process's
        // CLI-resolved default. This drives the exact pieces `serve`'s own startup composes
        // (`Persistence::open_repo` to "restart", then `model_and_level_at_active` +
        // `resolve_startup_model_and_level`) without needing a live gateway/stdin to exercise `serve`
        // itself end to end.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-original", SessionSelect::Latest)
                .unwrap();
        let mut session = Session::new();
        {
            let store = persistence.store.as_mut().unwrap();
            session.user("hello");
            store.append_new(&session.messages).unwrap();
        }
        persistence
            .store
            .as_mut()
            .unwrap()
            .record_model_change("gpt-5")
            .unwrap();
        {
            // A change is anchored at the tip it's recorded against and takes effect for what comes
            // *after* it, not the tip itself (see `SessionStore::record_model_change`'s own doc comment
            // and `model_and_thinking_level_changes_are_branch_scoped`) — so a follow-up turn actually
            // has to land *after* the `set_model` for the active tip to observe the new model, matching
            // every real `set_model`-then-continue session shape.
            let store = persistence.store.as_mut().unwrap();
            session.push(agent_core::Message::assistant(vec![
                agent_core::ContentBlock::text("hi there"),
            ]));
            store.append_new(&session.messages).unwrap();
        }
        drop(persistence);

        // "Restart": a fresh `Persistence::open_repo` against the same directory/cwd, exactly like a
        // brand-new `serve --continue` process's `Persistence::open` would do — reattaching to the most
        // recent session for this cwd rather than creating a new one. (A bare `serve` now selects
        // `Fresh` instead; reattach is what this test is about, so it asks for it explicitly.)
        let (restarted, _session) = Persistence::open_repo(
            dir.path(),
            "/w",
            "claude-cli-default",
            SessionSelect::Latest,
        )
        .unwrap();

        let cfg_level = agent_core::ThinkingLevel::Off;
        let (session_model, session_level) = restarted.model_and_level_at_active(cfg_level);
        // No `--model`/`--reasoning-effort` passed on this "restart" (`model_explicit: false`,
        // `reasoning_effort_explicit: false`) — the session's own recorded model must win over
        // "claude-cli-default", the value a bare `serve` with no flag would otherwise have resolved to.
        let (current_model, _starting_level) = resolve_startup_model_and_level(
            "claude-cli-default",
            cfg_level,
            false,
            false,
            session_model,
            session_level,
        );
        assert_eq!(
            current_model, "gpt-5",
            "must continue on the session's own last-set model, not the CLI/global default"
        );

        // An explicit `--model` on the same restart must still win outright.
        let (session_model, session_level) = restarted.model_and_level_at_active(cfg_level);
        let (current_model, _starting_level) = resolve_startup_model_and_level(
            "claude-explicit-override",
            cfg_level,
            true,
            false,
            session_model,
            session_level,
        );
        assert_eq!(current_model, "claude-explicit-override");
    }
}
