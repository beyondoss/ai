//! Headless `serve` — a newline-delimited JSON control protocol, over stdio or a WebSocket.
//!
//! The server is the source of truth; any client (a TUI, an editor, an `ssh` pipe, or the Beyond
//! iPhone app) drives it by sending one JSON command per message and reading one JSON frame per
//! message. The shape mirrors pi's `rpc` mode and opencode's session server: commands get a
//! `response` frame, and a `prompt` streams `event` frames (the agent's `AgentEvent`s) before its
//! response.
//!
//! **Two transports, one protocol.** The command/frame protocol below is byte-identical regardless of
//! transport. [`serve`] is the default stdio transport (one line per command on stdin, one frame per
//! line on stdout — built for an `ssh` pipe). [`serve_ws`](crate::serve_ws), enabled by
//! `--listen <addr>`, offers the same protocol over a WebSocket: one command per inbound text message,
//! one frame per outbound text message. Both feed the transport-agnostic [`serve_session`] core, which
//! reads commands from an `mpsc` channel and emits frames to whichever connection is currently
//! attached — so a WebSocket session **outlives its connection**: a dropped mobile client reconnects
//! (same `?session_id=`) and re-attaches to the same still-running run, catching up on anything
//! committed while it was gone via `get_messages {since}`.
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
//!     other reason.
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
//!     fork; `null` in single-file/in-memory mode, where there's no *new* id to link)
//!   - `{type:"list_sessions", query?}`  (repo mode) → `data: {sessions: [SessionMeta + updated_at/
//!     message_count/preview/search_text…]}` (via `SessionMeta::to_listing_json` — those four fields are
//!     `#[serde(skip)]` on the struct itself), this project's sessions only (matched by the default
//!     per-cwd directory, or whatever `--session-dir` points at). An optional `query` string filters and
//!     ranks the result (`session_store::search_sessions` — case-insensitive substring match against
//!     `title`/`id`/`preview`/`cwd`/`search_text`, in that priority order); omitted (or blank) returns
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
//!     copied prefix (fork right before it), the default `true` excludes it.
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
//!     unreachable from any RPC command; same persistence-required error as `set_label`
//!   - `{type:"export_html", output_path?}` render the active session's transcript as a single
//!     self-contained HTML file → `data: {path}`. `output_path` defaults to a timestamped
//!     `session-<unix-seconds>.html` in the current directory; parent directories are created as
//!     needed.
//!   - `{type:"compact", custom_instructions?}` summarize the prefix now → `data: {compacted: bool,
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
//!     (`user_messages`/`assistant_messages`/`tool_calls`/`tool_results`/`total_messages`)
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
//!     mid-stream retry inside a single model turn (`agent_core::Agent::with_auto_retry`), and, once
//!     that's exhausted, automatically re-invoking the *whole* `prompt` run against the same session up
//!     to 3 more times with backoff (2/4/8s) — pi's `agent-session.ts` auto-retry. Each whole-run retry
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
//!     …]}` — always the full, non-exhaustive known-model hint list, never narrowed by `--models`
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
//!     — independent of the model's own tool-call loop — streaming `tool_progress`/`tool_end` events
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
//! zero or more unsolicited updates for an in-flight `login`, correlated via `id` (see `login` above).
//!
//! Two more frames exist only when `--approve` installed an interactive gate (see [`crate::approval`]).
//! `{type:"approval_request", request_id, tool, summary, scope_key, origin, options}` is broadcast to
//! **every** attached client when a gated tool call is about to run, and the call blocks until one of
//! them answers with an `approve` command. `summary` is a truncated preview of the call's arguments (a
//! megabyte file body is not fanned out to a phone); `scope_key` is what an `"session"`-scoped answer
//! would be remembered against, shown so the user can see exactly what they are agreeing to; `origin` is
//! `"main"` or `{agent, spawn_id}` naming the subagent child that is asking.
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

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::approval::{ApprovalDecision, ApprovalError, ApprovalScope};

use agent_core::{
    Agent, AgentEvent, AgentHooks, CancellationToken, GatewayClient, Session, StopReason,
    StreamEvent, ToolUpdate,
};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::gateway_credential::{GatewayCredential, resolve_gateway_credential};
use crate::session_store::{
    BranchInfo, BranchSummaryDetails, CompactionMeta, SessionMeta, SessionRepo, SessionStore,
    search_sessions,
};
use crate::tools;

/// How the daemon pools its upstream (agent→gateway) connections across sessions — the mode selected
/// by `--upstream-http2` (see [`crate::serve_ws::build_shared_h2_client`]).
///
/// The agent→gateway hop is plaintext, so `H2c` (HTTP/2 cleartext) is what actually collapses N
/// sessions onto ~one multiplexed connection — but it hard-requires a gateway that speaks h2c, so it
/// is **not** the default (see the variant docs). `FromStr` backs the CLI flag's `value_parser`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpstreamHttp2 {
    /// No shared client — every session (and every model switch) builds its own `reqwest::Client`, as
    /// before this feature. The default until the gateway's h2c support is verified end-to-end.
    #[default]
    Off,
    /// One shared client with **no** prior-knowledge: HTTP/1.1 connection pooling across all sessions
    /// today, transparently negotiating h2 if the hop later moves to `https://` with ALPN. Safe against
    /// an h1-only gateway (unlike [`Self::H2c`]).
    Auto,
    /// One shared client pinned to HTTP/2 cleartext (`http2_prior_knowledge`) — multiplexes every
    /// session over ~one TCP connection to the gateway. **Requires a gateway that accepts h2c**: against
    /// an h1-only gateway *every* request fails, which is why this is never the default.
    H2c,
}

impl std::str::FromStr for UpstreamHttp2 {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "h2c" => Ok(Self::H2c),
            other => Err(format!(
                "invalid --upstream-http2 {other:?} (expected one of: off, auto, h2c)"
            )),
        }
    }
}

/// Options for the headless server (mirrors `run`, plus persistence).
///
/// `Clone` so a WebSocket supervisor ([`crate::serve_ws`]) can hand each per-session task its own
/// copy with a distinct [`Self::session_id`] pinned in — `mcp_tools` clones the `Arc`s (shared live
/// connections), every other field is a plain owned value.
#[derive(Clone)]
pub struct ServeConfig {
    pub gateway: String,
    /// The raw `--key`/`AI_AGENT_KEY` value, if the operator gave one explicitly. Deliberately *not*
    /// pre-resolved into a [`GatewayCredential`] by `main.rs` before `serve` starts, unlike every other
    /// `ServeConfig` field: [`resolve_gateway_credential`] is keyed on the model, and this process's
    /// active model isn't fixed for its lifetime — a reattached `--session`/`--continue` session may
    /// override `model` with its own last-recorded one before the first turn even runs, and
    /// `set_model`/`cycle_model`/`switch_session`/`fork`/`clone`/`switch_branch` can all change it again
    /// at any point afterward. `crate::serve::build_gateway_client` calls `resolve_gateway_credential`
    /// fresh with this value every time the active model is (re)determined, rather than resolving once
    /// here and freezing whichever provider happened to be active at that moment.
    pub key: Option<String>,
    pub model: String,
    /// Whether the operator explicitly passed `--model` for *this* invocation, as opposed to `model`
    /// having fallen back to a stored `agent settings` default or this crate's own built-in default
    /// (`main.rs`'s `resolved_model` resolution) — Task #5 (pi-parity fix). `serve`'s own startup uses
    /// this to decide whether to prefer a reattached session's own last-recorded model
    /// (`Persistence::model_and_level_at_active`) over `model` when reopening an existing session: a
    /// merely-stored default must not override what that session was actually last running on, the same
    /// "bleed" `switch_session`/`switch_branch` are already hardened against (see their own doc
    /// comments) — previously unguarded at ordinary process restart. A no-op for a genuinely fresh
    /// session, since its own recorded model already equals `model` either way. Matches `run
    /// --continue`'s own identical `model_explicit` distinction (`main.rs::run_task`).
    pub model_explicit: bool,
    /// Same idea as [`Self::model_explicit`], for `--reasoning-effort`/`starting_level`.
    pub reasoning_effort_explicit: bool,
    pub max_steps: u32,
    /// Base system prompt (agent identity). Project instructions, skills, and env are layered on top.
    pub system: String,
    /// Extra instructions appended after the base prompt (`--append-system-prompt`).
    pub append_system: Option<String>,
    /// Whether to discover and inject `AGENTS.md`/`CLAUDE.md` project-instruction files.
    pub context_files: bool,
    /// Persist a single session to this JSONL file (append-per-turn). Mutually exclusive with
    /// `session_dir`.
    pub session_file: Option<String>,
    /// Persist many sessions under this directory (a [`SessionRepo`]). Enables the multi-session
    /// commands (`list_sessions`, `switch_session`, `fork`, `set_session_name`).
    pub session_dir: Option<String>,
    /// The persistent-memory backend DSN (`--memory`/`AI_AGENT_MEMORY_URL`, with the stored
    /// `default_memory_backend` folded in by `main.rs`). `None` resolves to a per-project local-file
    /// store. Resolved into a backend by `serve_session` itself — after project trust — not here, so it
    /// shares the same resolved `cwd` (mirroring `agents`). See [`crate::memory`].
    pub memory: Option<String>,
    /// Disable persistent memory entirely (`--no-memory`): no `memory` tool, no injected index.
    pub no_memory: bool,
    /// Use this exact session id instead of a freshly generated one, wherever a *new* `SessionMeta` is
    /// actually minted by [`Persistence::open`] — already-validated by `main.rs` (embedded directly into
    /// a persisted filename, so it must be sanitized before it ever reaches here). Matches `run`'s
    /// identical `--session-id` flag/contract: ignored when reattaching to an existing session (already
    /// has a fixed id from disk), whether that's an existing `session_file` or a repo-mode match on the
    /// current `cwd`.
    pub session_id: Option<String>,
    /// Skip persistence entirely, even though neither `session_file` nor `session_dir` was set —
    /// without this, `Persistence::open` defaults to a per-cwd directory under
    /// `~/.claude/sessions/<encoded-cwd>/` rather than silently running in-memory-only (an operator
    /// who simply forgot the flag would otherwise lose all history on restart with no indication why).
    /// An explicit opt-out for the cases that genuinely want pure in-memory operation (e.g. a
    /// short-lived test harness).
    pub no_session_persistence: bool,
    /// The model's context window in tokens — the budget the loop compacts against to stay below.
    /// `None` defers to the model's own capability-table window (`Agent::new`'s default), so switching
    /// to a model with a different real window (via `set_model`) gets that model's true budget instead
    /// of a stale operator-supplied number. `Some` pins a fixed budget that survives model switches.
    pub context_window: Option<u32>,
    /// The per-turn output token ceiling (`Agent::with_max_tokens`). `None` defers to `Agent::new`'s
    /// own model-aware default (the model's own capability-table `max_output`, floored at
    /// `DEFAULT_MAX_TOKENS`) — matches `context_window`'s own "unset means model-derived" convention.
    /// Re-applied on every `build_agent` rebuild (`set_model`/`set_thinking`/…), same as every other
    /// `cfg`-sourced override, so it survives a model switch rather than resetting to that new model's
    /// own default.
    pub max_tokens: Option<u32>,
    /// Use the 1-hour prompt-cache TTL (vs the default 5 minutes) — useful when turns are spaced out.
    pub cache_long: bool,
    /// Extended-thinking token budget, when enabled (`None` leaves thinking off). Must be below the
    /// per-turn `max_tokens`.
    pub thinking: Option<u32>,
    /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
    /// reasoning models, Anthropic adaptive-thinking models). `None` leaves the provider default.
    /// Fixed for the process — unlike `thinking`, there's no `set_reasoning_effort` RPC command.
    pub reasoning_effort: Option<agent_core::ReasoningEffort>,
    /// Sampling temperature. `None` leaves the provider default. Fixed for the process — no RPC
    /// command to change it mid-run. See `agent_core::ModelRequest::temperature`'s doc comment for
    /// per-dialect gating (Anthropic omits it while thinking is enabled).
    pub temperature: Option<f64>,
    /// Trust the working directory for this run only, so a project-local `.claude/SYSTEM.md` is
    /// honored even if the directory isn't in the persisted allowlist (`agent trust <path>`). See
    /// `crate::trust_store`.
    pub trust_project: bool,
    /// Force the working directory *untrusted* for this run only, overriding both `trust_project` and
    /// the persisted allowlist — pi's own `--no-approve`/`-na`. For a directory the operator has
    /// permanently `agent trust`ed but wants to run against once as if it weren't (testing untrusted
    /// behavior, or extra caution on a checkout that happens to live under an already-trusted parent).
    /// Wins over `trust_project` if both are somehow set.
    pub force_untrusted: bool,
    /// Compaction headroom (tokens) reserved below the context window. `None` keeps
    /// `CompactionConfig::default()`'s value.
    pub compaction_reserve_tokens: Option<u32>,
    /// Roughly how many tokens of recent conversation compaction keeps verbatim. `None` keeps
    /// `CompactionConfig::default()`'s value.
    pub compaction_keep_recent_tokens: Option<u32>,
    /// Token budget reserved below the context window when summarizing an abandoned tree branch —
    /// independent of `compaction_reserve_tokens`. `None` keeps `Agent`'s prior hard-tied behavior
    /// (falls back to whatever `compaction.reserve_tokens` resolves to — see
    /// `agent_core::Agent::summarize_branch`'s doc comment). Task #31 (pi-parity feature):
    /// `agent_core::Agent::with_branch_summary_reserve_tokens` previously had no caller in either
    /// binary; `run`'s identical `--branch-summary-reserve-tokens` flag.
    pub branch_summary_reserve_tokens: Option<u32>,
    /// Disable automatic (threshold-triggered) compaction entirely for this process — `run`'s
    /// identical flag. Only ever a one-way "definitely off" here: when `false` (the flag/env's own
    /// default, indistinguishable from "not given" for a bare disable-only flag), `serve`'s startup
    /// falls back to the persisted `agent settings` `compaction_enabled` override (see
    /// `settings::Settings::compaction_enabled`'s doc comment) before finally defaulting to enabled —
    /// see [`serve`]'s own resolution of `current_auto_compaction`.
    pub no_compaction: bool,
    /// How many times to retry a gateway request that fails before the first response byte arrives.
    /// `None` keeps the client's built-in default.
    pub retry_max_retries: Option<u32>,
    /// Base of the exponential backoff between those retries. `None` keeps the client's built-in
    /// default.
    pub retry_base_delay_ms: Option<std::time::Duration>,
    /// Ceiling on that exponential backoff. `None` keeps the client's built-in default
    /// (`agent_core::client::MAX_BACKOFF`, 60s). Task #30 (pi-parity feature): the retry cluster's
    /// third knob, `agent_core::client::GatewayClient::with_max_backoff`, previously had no CLI flag or
    /// persisted override at all.
    pub retry_max_backoff_ms: Option<std::time::Duration>,
    /// Idle-read timeout between response chunks on the gateway HTTP client. `None` keeps the client's
    /// built-in default. Task #38 (pi-parity fix): `run`'s identical `--idle-timeout-ms`/
    /// `AI_AGENT_IDLE_TIMEOUT_MS`/persisted `default_provider_timeout_ms` setting (see `main.rs::
    /// run_task`'s own `with_idle_timeout` call site) previously had no `serve` counterpart at all —
    /// `build_gateway_client` never called `GatewayClient::with_idle_timeout`.
    pub idle_timeout_ms: Option<u64>,
    /// Force every image down the same downgrade-to-text-placeholder path a vision-incapable model
    /// already gets, regardless of the active model's real `supports_vision` capability — Task #34
    /// (pi-parity fix): `run`'s identical `--block-images`/`agent settings --block-images` (see
    /// `main.rs::run_task`'s own `Agent::with_block_images` call site) previously had no `serve`
    /// counterpart at all. Threaded to `Agent::with_block_images` in `build_agent`.
    pub block_images: bool,
    /// Whether `read` downscales/re-encodes an oversized image to fit the inline size budget — Task #34
    /// (pi-parity fix): `run`'s identical `--image-auto-resize`/`agent settings --image-auto-resize`
    /// (see `main.rs::run_task`'s own `default_registry_with_prefix_and_image_auto_resize` call site)
    /// previously had no `serve` counterpart at all — `build_tools` always called the `image_auto_resize`
    /// registry constructor with a hardcoded `true`, so `--no-image-auto-resize`/`agent settings
    /// --image-auto-resize false` had no effect on a `serve` process. Threaded to
    /// `tools::default_registry_with_prefix_and_image_auto_resize` in `build_tools`.
    pub image_auto_resize: bool,
    /// Default `bash` command timeout when the model omits `timeout_ms`. `None` keeps the tool's
    /// built-in default.
    pub bash_timeout_ms: Option<u64>,
    /// Run `bash` commands through this shell instead of the auto-resolved one (`/bin/bash`, else
    /// `bash` on `$PATH`, else `sh`). Already validated to exist by the time it reaches here (see
    /// `--bash-shell-path` in `main.rs`) — `build_tools` trusts it rather than re-checking on every
    /// rebuild.
    pub bash_shell_path: Option<String>,
    /// Prepend this line to every `bash` command, in the same shell invocation — matches pi's own
    /// `shellCommandPrefix` setting. Fixed for the process, like `bash_shell_path`.
    pub bash_command_prefix: Option<String>,
    /// `--web-allow-private`: let the `web` tool reach loopback/private/link-local addresses. Off by
    /// default. See `tools::web`.
    pub web_allow_private: bool,
    /// `--web-allow-host`: hostnames the `web` tool may reach even with private egress off.
    pub web_allow_hosts: Vec<String>,
    /// `--web-timeout-ms`: the `web` tool's per-request timeout (default 30 s).
    pub web_timeout_ms: Option<u64>,
    /// Restrict the tool set to exactly these names, dropping everything else. Combine with
    /// `exclude_tools` to carve one back out of the allow-list. Fixed for the process — like `system`,
    /// there's no runtime RPC to change it, but it does survive a `set_model`/`set_thinking` rebuild
    /// (`build_agent` reapplies it every time).
    pub tools: Option<Vec<String>>,
    /// Drop these tools from the default set — e.g. `["bash", "write"]` for a read-only reviewer.
    pub exclude_tools: Option<Vec<String>>,
    /// Register no tools at all. Wins over `tools`/`exclude_tools`.
    pub no_tools: bool,
    /// Tools discovered from every configured MCP server, already connected and namespaced
    /// (`mcp__<server>__<tool>`) — see `tools::mcp`'s module doc comment. Resolved exactly once, by
    /// `main.rs`'s `serve`-dispatch branch (`tools::mcp::connect_all`), *before* this `ServeConfig` is
    /// constructed — connecting is inherently async and this struct is a plain, synchronous value.
    /// `build_tools` merges these into every registry rebuild (`set_model`/`set_thinking`/...) without
    /// reconnecting to any server: the live connection (and its spawned child process, for a stdio
    /// server) lives as long as this `Vec`'s `Arc<dyn Tool>` entries do, i.e. for the whole `serve`
    /// session.
    pub mcp_tools: Vec<Arc<dyn agent_core::Tool>>,
    /// Agent definitions discovered at startup (see [`crate::agents`]) — the delegable personas the
    /// `subagent` tool accepts, advertised in `<available_agents>`. Discovered once, like `mcp_tools`,
    /// rather than re-walked on every registry rebuild. Empty when subagents aren't configured.
    pub agents: Vec<crate::agents::AgentDef>,
    /// Force every batch of tool calls in a turn to run one at a time instead of the default
    /// bounded-concurrent dispatch (`agent_core::Agent::with_sequential_tools`). Fixed for the process,
    /// like `tools`/`no_tools` — `build_agent` reapplies it every rebuild.
    pub sequential_tools: bool,
    /// Block every call to a tool named here, even though it stays registered/visible to the model —
    /// unlike `exclude_tools` (the model never learns it exists), a denied call still surfaces as a
    /// normal error `tool_result` explaining it was blocked by policy. Installs an `AgentHooks` gate
    /// (`crate::policy::ToolPolicy`) on every `build_agent` rebuild; empty means no hook at all (the
    /// `NoHooks` default), same zero-cost as before this flag existed.
    pub deny_tool: Vec<String>,
    /// Block a `bash` call whenever its command contains this substring, case-insensitively. Combines
    /// with `deny_tool` under the same policy hook.
    pub deny_bash_pattern: Vec<String>,
    /// Block a `write`/`edit` call whenever its `path` argument matches this glob. Combines with
    /// `deny_tool`/`deny_bash_pattern` under the same policy hook.
    pub deny_path: Vec<String>,
    /// Which tools require a human's approval before they run (`--approve off|writes|all|tools:a,b`). The
    /// static deny-lists above still win first — they need no round trip, and asking a person to approve
    /// what the operator already forbade is a way to social-engineer past them. See [`crate::approval`].
    pub approve: crate::approval::GatedSet,
    /// How long an unanswered `approval_request` waits before it is denied (`--approval-timeout`).
    /// `None` waits forever, which is only safe with a reliably-attached client: `running` stays `true`
    /// for the whole prompt, so a question nobody answers pins the session until an `abort`.
    pub approval_timeout: Option<std::time::Duration>,
    /// Disable *standard-root* skills discovery/loading (`~/.claude/skills`, `<cwd>/.claude/skills`) —
    /// no `<available_skills>` listing in the system prompt from either, and a `/skill:name` invocation
    /// is sent through unexpanded unless it resolves against a `--skill` path instead. Matches pi's own
    /// `--no-skills`, and `run`'s identical flag (`main.rs`'s `Run::no_skills`) — `serve` previously had
    /// no equivalent, so an operator wanting a hardened, no-custom-content deployment had no way to
    /// refuse project-supplied skills the way a one-shot `run` could. Skips standard-root discovery
    /// outright (like `run`) rather than discovering and then discarding, applied on every `reload` too
    /// — but an explicit `--skill <path>` is still honored even so, matching pi's own `--no-skills` (a
    /// documented, tested combination: `--no-skills` means "nothing auto-discovered", not "no skills at
    /// all" — see `skills::discover_extra_only`'s doc comment).
    pub no_skills: bool,
    /// Disable *standard-root* prompt-template discovery/loading (`~/.claude/prompts`,
    /// `<cwd>/.claude/prompts`) — a `/name` invocation is sent through unexpanded unless it resolves
    /// against a `--prompt-template` path instead. Matches pi's own `--no-prompt-templates` and `run`'s
    /// identical flag; see `no_skills`'s doc comment for why `serve` needed this too, and for the
    /// explicit-extra-path carve-out that applies here identically.
    pub no_prompt_templates: bool,
    /// Additional, ad-hoc skill-discovery roots beyond the two standard ones — pi's own
    /// `--skill <path>` (repeatable). Applied on every `reload` too, same as the standard roots.
    pub extra_skill_paths: Vec<String>,
    /// Additional, ad-hoc prompt-template-discovery roots — pi's own `--prompt-template <path>`
    /// (repeatable). See `extra_skill_paths`'s doc comment; applies identically.
    pub extra_prompt_template_paths: Vec<String>,
    /// Set the session's title up front, if a *new* session is minted at startup (persistence
    /// configured, no existing session reattached) — pi's own `--name`. Never renames an existing
    /// session just because the process happened to be started with this flag again; already validated
    /// non-whitespace by `main.rs` before this struct is even constructed.
    pub name: Option<String>,
    /// Restrict `cycle_model`'s candidate list to exactly these patterns, resolved in this order
    /// (`--models`, comma-separated; see `resolve_model_scope` for the exact grammar — literal ids,
    /// globs against [`available_models`], and an optional `:<thinking-level>` suffix pinning that
    /// entry's depth) — pi's own `--models` flag (`resolveModelScopeWithDiagnostics`/`_scopedModels`).
    /// `set_model` is unaffected: it still accepts any id, scoped or not. `get_available_models` is
    /// unaffected too — unlike pi's own scope-defaulted `/model` picker, this RPC has no secondary
    /// "show everything" toggle, so it always reports the full list and leaves any scoped-vs-all UI
    /// distinction to the client. Empty or absent falls back to the full [`available_models`] list for
    /// cycling as well.
    pub models: Vec<String>,
    /// A persisted, blanket project-trust policy (`agent settings --default-project-trust`) — Fix 1
    /// (pi-parity gap): this field didn't exist at all before, so a stored policy had zero effect on
    /// `serve` sessions even though `run` already partially honored it (`main.rs::run_task`). Consulted
    /// by [`resolve_project_trust`] at both startup and `reload`, with the same precedence `run` now
    /// uses: `trust_project`/`force_untrusted` win outright when set; otherwise an explicit per-path
    /// `TrustStore` entry (`agent trust`/`agent untrust <path>`) wins next; only when neither applies
    /// does this blanket policy take effect.
    pub default_project_trust: Option<crate::settings::TrustPolicy>,
    /// The initial steer-lane drain mode (`agent_core::QueueMode`) — pi's own `steeringMode`. Applied
    /// once, at startup, via `steering.set_steering_mode` right after `Steering::new()`; the
    /// `set_steering_mode` RPC command can still change it at runtime afterward. `main.rs` resolves
    /// this the same "explicit flag/env, then stored `agent settings --steering-mode`, then
    /// `QueueMode::default()`" precedence `run_task` uses for its own one-shot `Steering` (see that
    /// call site's doc comment) before ever constructing this config.
    pub steering_mode: agent_core::QueueMode,
    /// Same idea as [`Self::steering_mode`], for the follow-up lane (`set_follow_up_mode`) — pi's own
    /// `followUpMode`.
    pub follow_up_mode: agent_core::QueueMode,
    /// When set, `serve` offers its (byte-identical) control protocol over a WebSocket on this address
    /// instead of stdio — each accepted connection drives a session, and a session outlives its
    /// connection so a dropped mobile client can reconnect and re-attach to a still-running run (see
    /// [`crate::serve_ws`]). Bind loopback/internal only: the agent authenticates no caller, it trusts
    /// whatever the front door forwarded. `None` (the default) keeps the stdio transport.
    pub listen: Option<std::net::SocketAddr>,
    /// When set, `serve` also (or instead) offers its control protocol over a Unix-domain socket at
    /// this path — a same-VM client gets kernel-enforced local authz via filesystem permissions
    /// (unlike loopback TCP, which authenticates nothing). Bound on the *same* shared supervisor as
    /// [`Self::listen`], so a session created over either transport is reachable over the other by the
    /// same `?session_id=`. `None` (the default) means no UDS listener. See [`crate::serve_ws`].
    pub listen_uds: Option<std::path::PathBuf>,
    /// The octal permission mode to `chmod` the [`Self::listen_uds`] socket to after binding (default
    /// `0o600` — owner-only; use `0o660` for a shared group). Ignored when `listen_uds` is `None`.
    pub listen_uds_mode: Option<u32>,
    /// When set (daemon mode only), the supervisor reaps a session that has had **no attached
    /// connection** for at least this long and isn't mid-run — dropping its retained `input_tx` so it
    /// persists and exits, exactly like graceful shutdown does per-session (see [`crate::serve_ws`]).
    /// `None` (the default) keeps today's forever-lived behavior: a detached session lives until the
    /// daemon shuts down. Ignored outside the WebSocket/UDS daemon path.
    pub session_idle_timeout: Option<std::time::Duration>,
    /// How the daemon pools upstream (agent→gateway) connections across sessions — the mode from
    /// `--upstream-http2`. Only consulted in the WebSocket daemon path ([`crate::serve_ws::serve_ws`],
    /// which reads it once to build [`Self::shared_http`]); the stdio/`run` path leaves it at
    /// [`UpstreamHttp2::Off`] and never pools.
    pub upstream_http2: UpstreamHttp2,
    /// The one `reqwest::Client` every session in this daemon shares, so N concurrent sessions collapse
    /// onto one connection pool (HTTP/2-multiplexed under [`UpstreamHttp2::H2c`]) instead of N. Built
    /// once by [`crate::serve_ws::serve_ws`] from [`Self::upstream_http2`] and injected per session via
    /// [`agent_core::GatewayClient::with_http_client`] in `build_gateway_client`; **kept** (not nulled)
    /// by `session_cfg` so every session inherits it. `reqwest::Client` is `Arc`-backed, so cloning it
    /// into each session is a cheap pointer bump. `None` (the default, and always on the stdio/`run`
    /// path) preserves the per-session-client behavior.
    pub shared_http: Option<reqwest::Client>,
}

/// Resolve whether a project is trusted for this session, from already-gathered inputs — shared by
/// `run` (`main.rs::run_task`) and `serve` (both its startup and `reload` paths) so the two binaries
/// agree on trust for the same directory under the same settings, matching pi's own
/// `trust-manager.ts:46-96` precedence:
///
/// 1. `force_untrusted` (`--force-untrusted`/`-na`) always wins outright: untrusted, full stop.
/// 2. `trust_project` (`--trust-project`/`-a`) wins next: trusted, full stop.
/// 3. An explicit **per-path** entry in the persisted [`crate::trust_store::TrustStore`] allowlist
///    (`agent trust`/`agent untrust <path>`, already resolved into `trust_lookup` by the caller) — Fix 1
///    (pi-parity bug): this used to be checked *after* the blanket policy below, so an operator's
///    specific exception for one directory could be silently overridden by a coarser `never`/`always`
///    default; pi's own resolution always checks the nearest explicit per-path decision before falling
///    back to any blanket policy, in either direction (an explicit `never` policy must not override a
///    specific `agent trust <path>`, and vice versa for `always`/an explicit `agent untrust <path>`).
/// 4. Only when `trust_lookup` is [`crate::trust_store::Trust::Unknown`] does the persisted blanket
///    `default_project_trust` policy apply — `always`/`never` decide outright; `ask` (or no policy at
///    all) falls back to this crate's original heuristic: trusted whenever the project has nothing
///    project trust actually gates (`has_gated_resources`, from
///    `crate::trust_store::has_trust_gated_resources`). This binary is headless, so unlike pi's own
///    interactive "trust this folder?" prompt, `ask` can't actually ask anything — see
///    [`crate::settings::TrustPolicy::Ask`]'s own doc comment.
///
/// Takes `trust_lookup`/`has_gated_resources` as plain data (rather than resolving them itself from
/// `cwd`) so the decision logic is unit-testable with no filesystem/global-trust-store state to
/// sandbox — the same "split I/O gathering from pure decision logic" shape this crate already uses
/// elsewhere (e.g. `resources::tz_string_offset` vs `tz_env_offset`).
pub fn resolve_project_trust(
    trust_project: bool,
    force_untrusted: bool,
    default_project_trust: Option<crate::settings::TrustPolicy>,
    trust_lookup: crate::trust_store::Trust,
    has_gated_resources: bool,
) -> bool {
    use crate::settings::TrustPolicy;
    use crate::trust_store::Trust;
    if force_untrusted {
        return false;
    }
    if trust_project {
        return true;
    }
    // Task #35 (pi-parity fix): pi's own `resolveProjectTrusted` (`project-trust.ts`) runs this
    // "nothing here for project trust to actually gate" fast path *immediately* after the
    // override check above — before even consulting the persisted trust store, let alone a blanket
    // `default_project_trust` policy. Previously this crate only consulted `has_gated_resources` deep
    // inside the `Unknown` arm below, as the final fallback — meaning an explicit per-path
    // `Trust::Trusted`/`Trust::Untrusted` entry, or an explicit `always`/`never` blanket policy, would
    // win even when the project has nothing trust-gated at all to protect. Currently inert in practice
    // (no trust-gated resource type today actually depends on this ordering), but would silently
    // matter the moment a new one is added without this precedence already matching pi's.
    if !has_gated_resources {
        return true;
    }
    match trust_lookup {
        Trust::Trusted => true,
        Trust::Untrusted => false,
        Trust::Unknown => match default_project_trust {
            Some(TrustPolicy::Always) => true,
            Some(TrustPolicy::Never) => false,
            // `has_gated_resources` is always `true` by the time this arm is reached (the fast path
            // above already returned otherwise), so this is effectively always `false` now — kept as
            // `!has_gated_resources` rather than a bare `false` literal so the fallback stays correct
            // and self-documenting on its own terms if the fast path above is ever refactored away.
            Some(TrustPolicy::Ask) | None => !has_gated_resources,
        },
    }
}

/// Waits for an OS shutdown request (SIGTERM, SIGHUP, or SIGINT/Ctrl-C) so `serve` (and `run` — see
/// `main.rs::run_task`, which reuses this same type) can drain in-flight work and persist before
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
    #[cfg(unix)]
    sigterm: tokio::signal::unix::Signal,
    #[cfg(unix)]
    sighup: tokio::signal::unix::Signal,
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
    pub fn new() -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            Ok(Self {
                sigterm: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
                sighup: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())?,
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self {})
        }
    }

    /// Resolves once a shutdown signal arrives, naming which one. Safe to call fresh on every loop
    /// iteration — every `Signal::recv` and `tokio::signal::ctrl_c` are re-armable, not one-shot.
    pub async fn wait(&mut self) -> Signal {
        #[cfg(unix)]
        {
            tokio::select! {
                _ = self.sigterm.recv() => Signal::Term,
                _ = self.sighup.recv() => Signal::Hup,
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

/// Runs [`Persistence::persist`]'s blocking file I/O (`sync_all`, directory `fsync`) on tokio's
/// blocking thread pool instead of `serve`'s single async control-loop task. `persist` runs after
/// every turn (and every manual `compact`), so on a slow or network-backed session directory, a
/// multi-ms-to-100ms `sync_all` would otherwise stall that task directly — delaying its stdin
/// `select!` loop, and so `abort`/`steer` responsiveness, for the duration. `persistence` and `session`
/// are moved in and handed back so the caller can keep using them (`Session` is cheap to clone: its
/// `messages` field is an `Arc`, so this is a pointer clone, not a deep copy of the transcript).
async fn persist_blocking(
    mut persistence: Persistence,
    session: Session,
    tokens_before: Option<u32>,
) -> (Persistence, std::io::Result<()>) {
    // `persist` never panics itself (it handles its own I/O errors internally — see the `Err` arm
    // below for what a caller does with the returned `Result`), and this task is never cancelled
    // (always awaited directly, never `.abort()`ed) — so a `JoinError` here can only mean the closure
    // panicked. Re-raise that panic rather than `.expect()`ing (denied by the workspace's
    // panic-surface lints) on what would otherwise look like an ordinary recoverable error.
    match tokio::task::spawn_blocking(move || {
        let r = persistence.persist(&session, tokens_before);
        (persistence, r)
    })
    .await
    {
        Ok(result) => result,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// Like [`persist_blocking`], but for one incremental mid-run checkpoint (see [`ChannelCheckpoint`]):
/// takes just the message snapshot a checkpoint carries rather than a whole `Session`, and always
/// appends (never a compacted rewrite — see [`Persistence::persist_messages`]).
async fn persist_messages_blocking(
    mut persistence: Persistence,
    messages: Arc<Vec<agent_core::Message>>,
) -> (Persistence, std::io::Result<()>) {
    match tokio::task::spawn_blocking(move || {
        let r = persistence.persist_messages(&messages);
        (persistence, r)
    })
    .await
    {
        Ok(result) => result,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// A [`agent_core::CheckpointHook`] that forwards each mid-run checkpoint's message snapshot through an
/// unbounded channel rather than persisting it directly — sending is instant (never blocks on I/O), so
/// it's safe to call from deep inside `Agent::run_events_steered`'s hot loop without stalling it. The
/// `"prompt"` arm's busy-loop (the only place a checkpoint can fire from) drains the receiving half and
/// does the actual (blocking) append.
struct ChannelCheckpoint(mpsc::UnboundedSender<Arc<Vec<agent_core::Message>>>);

#[async_trait::async_trait]
impl agent_core::CheckpointHook for ChannelCheckpoint {
    async fn checkpoint(&self, session: &Session) {
        // Best-effort: if the receiving end is gone (a prior run already dropped it — shouldn't happen
        // while a run is in flight, since the receiver outlives every `"prompt"` arm) there's nothing to
        // recover, and the run itself must not fail just because incremental persistence couldn't.
        let _ = self.0.send(session.messages.clone());
    }
}

/// Mint a brand-new `SessionMeta`, using `session_id` in place of a freshly generated one when given
/// (`ServeConfig::session_id`, already validated by `main.rs` before this is ever called).
fn fresh_meta(cwd: &str, model: &str, session_id: Option<&str>) -> SessionMeta {
    match session_id {
        Some(id) => SessionMeta::with_id(id.to_string(), cwd, model),
        None => SessionMeta::new(cwd, model),
    }
}

/// Where the server persists sessions: a multi-session [`SessionRepo`] (`--session-dir`), a single
/// JSONL file (`--session-file`), or nowhere (in-memory). It always carries the current session's
/// [`SessionMeta`] so the session id is stable across reattaches.
struct Persistence {
    repo: Option<SessionRepo>,
    store: Option<SessionStore>,
    meta: SessionMeta,
}

impl Persistence {
    /// Open persistence and restore (or create) the active session. In repo mode, reopens the most
    /// recent session or creates a fresh one; in file mode, opens the file or creates it.
    fn open(cfg: &ServeConfig) -> std::io::Result<(Self, Session)> {
        let cwd = crate::session_store::canonical_cwd(&std::env::current_dir().unwrap_or_default())
            .to_string_lossy()
            .into_owned();
        if let Some(dir) = &cfg.session_dir {
            return Self::open_repo(dir, &cwd, &cfg.model, cfg.session_id.as_deref());
        }
        if let Some(path) = &cfg.session_file {
            let path = std::path::PathBuf::from(path);
            // A zero-byte file at `path` (e.g. `touch`'d ahead of time, or left over from a crash
            // before the header write landed) has nothing to open — route it through `create`, which
            // now initializes an empty file in place rather than failing (see its own doc comment).
            let has_content = path.metadata().is_ok_and(|m| m.len() > 0);
            let (store, session) = if has_content {
                SessionStore::open(path)?
            } else {
                let meta = fresh_meta(&cwd, &cfg.model, cfg.session_id.as_deref());
                let store = SessionStore::create(path, meta)?;
                (store, Session::new())
            };
            let meta = store.meta().clone();
            return Ok((
                Self {
                    repo: None,
                    store: Some(store),
                    meta,
                },
                session,
            ));
        }
        if cfg.no_session_persistence {
            return Ok((
                Self {
                    repo: None,
                    store: None,
                    meta: fresh_meta(&cwd, &cfg.model, cfg.session_id.as_deref()),
                },
                Session::new(),
            ));
        }
        // Neither flag was given and persistence wasn't explicitly opted out: default to a per-cwd
        // repo directory rather than silently running in-memory-only, so an operator who simply
        // forgot the flag doesn't lose all history on the next restart with no indication why.
        Self::open_repo(
            crate::session_store::default_session_dir(&cwd),
            &cwd,
            &cfg.model,
            cfg.session_id.as_deref(),
        )
    }

    /// Open (creating if needed) a multi-session repo at `dir` and reattach to the most recent session
    /// whose recorded cwd matches `cwd` — not just the globally newest one, so a shared `--session-dir`
    /// spanning multiple projects (or the shared default directory before cwd-encoding existed) doesn't
    /// resume a stranger's unrelated session. No match (a fresh directory, or one with no session for
    /// this cwd yet) creates a new one, using `session_id` in place of a freshly generated one when
    /// given (see `ServeConfig::session_id`'s doc comment).
    fn open_repo(
        dir: impl Into<std::path::PathBuf>,
        cwd: &str,
        model: &str,
        session_id: Option<&str>,
    ) -> std::io::Result<(Self, Session)> {
        let repo = SessionRepo::open(dir)?;
        let (store, session) = repo.resume_or_create(cwd, model, session_id)?;
        let meta = store.meta().clone();
        Ok((
            Self {
                repo: Some(repo),
                store: Some(store),
                meta,
            },
            session,
        ))
    }

    fn session_id(&self) -> &str {
        &self.meta.id
    }

    /// The active session's on-disk JSONL path — pi's `sessionFile` — or `None` when persistence is
    /// disabled entirely (`--no-session-persistence`, an in-memory-only run). Repo and single-file
    /// modes both populate `store`, so this covers either without needing to distinguish them.
    fn session_file(&self) -> Option<&Path> {
        self.store.as_ref().map(SessionStore::path)
    }

    /// Persist the session after a turn: non-destructively rewrite the transcript (see
    /// `SessionStore::rewrite_compacted`) if compaction fired this round, otherwise append just the
    /// new messages. `tokens_before` is `Some` (carrying `AgentEvent::Compacted`'s own value) exactly
    /// when a compaction fired.
    fn persist(&mut self, session: &Session, tokens_before: Option<u32>) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = match tokens_before {
            Some(tokens_before) => store.rewrite_compacted(
                &session.messages,
                CompactionMeta {
                    tokens_before,
                    // Fix 2 (pi-parity fix): persist this round's folded-forward file-provenance onto
                    // the new `Entry::Compaction` record — otherwise it never reaches disk at all, and
                    // a restart/reattach after this round forgets every file this session has ever
                    // touched (see `Entry::Compaction::read_files`'s own doc comment).
                    provenance: session.compaction.clone(),
                },
            ),
            None => store.append_new(&session.messages),
        };
        if let Err(e) = &r {
            eprintln!("serve: failed to persist session: {e}");
        }
        r
    }

    /// Incremental persist for a mid-run checkpoint (see [`ChannelCheckpoint`]) — always a plain
    /// append, never a compacted rewrite: compaction only ever runs at a turn's *start*, strictly
    /// before that turn's own checkpoint(s), so the session a checkpoint carries is never mid-compaction.
    fn persist_messages(&mut self, messages: &[agent_core::Message]) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = store.append_new(messages);
        if let Err(e) = &r {
            eprintln!("serve: failed to persist checkpoint: {e}");
        }
        r
    }

    /// The working directory, for new session metadata — canonicalized, so a session created via
    /// `new_session` matches the same [`canonical_cwd`](crate::session_store::canonical_cwd) form
    /// every other session-cwd is recorded in.
    fn cwd() -> String {
        crate::session_store::canonical_cwd(&std::env::current_dir().unwrap_or_default())
            .to_string_lossy()
            .into_owned()
    }

    /// Start a fresh session. In repo mode this creates a new file (new id); in single-file mode it
    /// resets the existing file (keeping its id); in-memory it just mints new metadata.
    ///
    /// In repo mode, the fresh session's `parent` records whatever session id was active immediately
    /// before this call, unless `parent_session` explicitly names a different one — pi's own
    /// `parentSession` lineage marker on a `/new`-equivalent reset (pi's own default, absent an
    /// explicit override, is no parent at all; this crate's default deliberately differs, always
    /// linking to the previously-active session, since that provenance is used elsewhere — e.g. tree
    /// navigation in the HTML export). It's provenance only (no shared history, unlike an actual
    /// `fork`): a client browsing sessions can still trace "this fresh one followed that one" in the
    /// same directory/conversation, or — with an explicit `parent_session` — "this one continues a
    /// *different* lineage than whatever happened to be active." Not recorded in single-file mode
    /// (there's no *new* id to link — the file/id just gets cleared in place) or in-memory mode
    /// (nothing persists a "previous" id anywhere a client could look it up later), matching pi's own
    /// `this.persist ? previousSessionFile : undefined`.
    ///
    /// On failure, nothing is mutated (no partial state) — the caller keeps whatever session was
    /// already active, matching the source of truth still on disk. This matters: silently returning a
    /// fresh in-memory `Session` while the on-disk reset actually failed would desync `SessionStore`'s
    /// persisted-message-count bookkeeping from what the caller believes is live, so a *subsequent*
    /// successful `persist` could look like a no-op append and silently drop every message of the
    /// "new" session (`SessionStore::append_new`'s own dedup guard, `messages.len() <= self.persisted`).
    fn new_session(
        &mut self,
        model: &str,
        parent_session: Option<&str>,
    ) -> std::io::Result<Session> {
        let cwd = Self::cwd();
        if let Some(repo) = &self.repo {
            let mut meta = SessionMeta::new(&cwd, model);
            meta.parent = Some(parent_session.unwrap_or(&self.meta.id).to_string());
            match repo.create(meta) {
                Ok(store) => {
                    self.meta = store.meta().clone();
                    self.store = Some(store);
                }
                Err(e) => {
                    eprintln!("serve: failed to create session: {e}");
                    return Err(e);
                }
            }
        } else if let Some(store) = &mut self.store {
            if let Err(e) = store.reset_for_new_session() {
                eprintln!("serve: failed to reset session: {e}");
                return Err(e);
            }
        } else {
            self.meta = SessionMeta::new(&cwd, model);
        }
        Ok(Session::new())
    }

    /// Switch to another session by id (repo mode only).
    fn switch(&mut self, id: &str) -> std::io::Result<Session> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (store, session) = repo.open_id(id)?;
        self.meta = store.meta().clone();
        self.store = Some(store);
        Ok(session)
    }

    /// Fork the current session into a new session and switch to it (repo mode). `entry_id`, when
    /// given, forks at that specific tree entry — anywhere in the whole tree, on or off the active
    /// path (`before` excludes the entry itself, matching pi's `position:"before"`/`"at"`) — otherwise
    /// falls back to `upto`, a message-count prefix of the active path (the original, narrower form).
    ///
    /// Also returns the model/thinking-level that was actually active at the forked-from point (Task #2,
    /// pi-parity fix) — resolved against *this* session's own tree (see
    /// [`Self::fork_target_model_and_level`]) *before* the swap below, since `SessionRepo::fork`/
    /// `fork_at_entry` deliberately don't carry `ModelChange`/`ThinkingLevelChange` bookkeeping into the
    /// freshly forked session's own file (see their own doc comments — forking only ever preserves
    /// message content, never the surrounding per-branch bookkeeping). Concretely: a session that
    /// switched models mid-run, then forked back to an entry on the *old* model's branch, previously kept
    /// running on whatever the process was last set to — silently replaying that old provider's signed
    /// `Thinking` blocks to a foreign model on the fork's very next turn. The caller (the `fork`/`clone`
    /// RPC handlers) applies this exactly like `switch_session`/`switch_branch` already do: clamp,
    /// `scrub_cross_model_state` when the model actually changes, and rebuild `agent` only if needed.
    fn fork(
        &mut self,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
        starting_level: agent_core::ThinkingLevel,
    ) -> std::io::Result<(Session, String, agent_core::ThinkingLevel)> {
        let id = self.meta.id.clone();
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (restored_model, restored_level) =
            self.fork_target_model_and_level(upto, entry_id, before, starting_level);
        let (store, session) = match entry_id {
            Some(entry_id) => repo.fork_at_entry(&id, entry_id, before)?,
            None => repo.fork(&id, upto)?,
        };
        self.meta = store.meta().clone();
        self.store = Some(store);
        Ok((session, restored_model, restored_level))
    }

    /// The model/thinking-level actually active at what a would-be [`Self::fork`] targets — see that
    /// method's own doc comment for why this must be computed against the *source* tree before the fork
    /// swaps `self.store` to the freshly created one. Mirrors `fork`'s own `entry_id`/`before`-vs-`upto`
    /// resolution exactly, so the point this reads model/level at is the same one the copied prefix
    /// actually ends on.
    fn fork_target_model_and_level(
        &self,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        match entry_id {
            Some(entry_id) => {
                // Same `before` resolution as `switch_branch`: `parent_of` returns `None` only for an
                // unknown id, in which case there's nothing meaningful to resolve here anyway — the
                // actual fork call just below fails with a clear `NotFound` and this fallback value is
                // never returned to a caller.
                let target: Option<String> = if before {
                    self.store
                        .as_ref()
                        .and_then(|s| s.parent_of(entry_id))
                        .unwrap_or(None)
                } else {
                    Some(entry_id.to_string())
                };
                self.model_and_level_at_opt(target.as_deref(), process_starting_level)
            }
            None => {
                let active = self.active_ids();
                let upto = upto.min(active.len());
                match upto {
                    0 => self.model_and_level_at_opt(None, process_starting_level),
                    n => self.model_and_level_at(&active[n - 1], process_starting_level),
                }
            }
        }
    }

    /// Preview what forking `session_id` would produce — the exact prefix `fork` would copy — without
    /// creating a new session file or switching to it (repo mode only, like `switch`/`fork`). Same
    /// `entry_id`/`before`-vs-`upto` choice as `fork`. A client browsing `list_sessions` uses this to
    /// preview a fork point before committing to it.
    fn fork_messages(
        &self,
        session_id: &str,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
    ) -> std::io::Result<Vec<agent_core::Message>> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        if let Some(entry_id) = entry_id {
            return repo.fork_at_entry_messages(session_id, entry_id, before);
        }
        let (_, session) = repo.open_id(session_id)?;
        let upto = upto.min(session.messages.len());
        Ok(session.messages[..upto].to_vec())
    }

    /// Delete a session by id (repo mode only) — [`SessionRepo::delete`](crate::session_store::SessionRepo::delete)'s
    /// soft-delete-to-`.trash`, idempotent whether or not `id` still exists. Refuses to delete the
    /// *currently active* session (the one this `Persistence` itself is pointed at): that would move the
    /// file out from under an in-memory `Session` a client is still actively using — with no session
    /// file left to persist the next turn into — a footgun no legitimate caller needs, since `list_
    /// sessions`/`list_all_sessions` responses report which id is current. A client that genuinely wants
    /// that must `new_session`/`switch_session` away first.
    fn delete(&self, id: &str) -> std::io::Result<()> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        if id == self.session_id() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot delete the currently active session — switch to another session first",
            ));
        }
        repo.delete(id)
    }

    /// List every session sitting in this repo's `.trash/` subdirectory (repo mode only) — see
    /// [`crate::session_store::SessionRepo::list_trash`].
    fn list_trash(&self) -> std::io::Result<Vec<crate::session_store::TrashEntry>> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        repo.list_trash()
    }

    /// Restore a session by id out of `.trash/` back to its original location (repo mode only) — see
    /// [`crate::session_store::SessionRepo::restore_session`].
    fn restore_session(&self, id: &str) -> std::io::Result<bool> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        repo.restore_session(id)
    }

    /// This process's own cwd's sessions, newest first (empty unless in repo mode).
    /// `on_progress(scanned, total)` is invoked once per file as the scan completes it — see
    /// [`SessionRepo::list_with_progress`](crate::session_store::SessionRepo::list_with_progress).
    ///
    /// Filtered to `self.meta.cwd` — the exact match [`SessionRepo::resume_or_create`] already applies
    /// when reattaching at startup. A no-op under the default per-cwd repo directory (every session
    /// there already shares this cwd, by construction — see `default_session_dir`), but load-bearing
    /// under an explicit `--session-dir` shared across projects: Track L28 (pi-parity fix) — this used
    /// to return every session in the directory unfiltered, so a shared `--session-dir` leaked another
    /// project's sessions into this one's `list_sessions` response.
    ///
    /// Runs the scan on `spawn_blocking` (like [`persist_blocking`]) rather than directly on the
    /// caller's own task — `SessionRepo::list_with_progress`'s underlying `scan_listings` does
    /// synchronous file I/O across a worker pool, and this command is reachable from the busy-mode
    /// loop while a model turn is in flight (unlike `persist_blocking`'s dedicated task, this one
    /// would otherwise stall that same task's turn-event delivery, `abort` handling, and checkpoint
    /// persistence for however long a large, unpruned session directory takes to scan).
    async fn list_with_progress(
        &self,
        on_progress: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> Vec<SessionMeta> {
        let Some(repo) = self.repo.clone() else {
            return Vec::new();
        };
        let cwd = self.meta.cwd.clone();
        let sessions =
            match tokio::task::spawn_blocking(move || repo.list_with_progress(on_progress)).await {
                Ok(Ok(sessions)) => sessions,
                Ok(Err(e)) => {
                    eprintln!("serve: failed to list sessions: {e}");
                    Vec::new()
                }
                // `list_with_progress` never panics itself and this task is never cancelled — same
                // reasoning as `persist_blocking`'s identical re-raise.
                Err(e) => std::panic::resume_unwind(e.into_panic()),
            };
        sessions.into_iter().filter(|m| m.cwd == cwd).collect()
    }

    /// Every session across every project's own repo directory (pi's cross-project `listAll`), not
    /// just this process's own — the parent of this repo's directory is treated as the shared sessions
    /// root, with one subdirectory per project (the convention
    /// [`session_store::default_session_dir`](crate::session_store::default_session_dir) follows).
    /// `Err` when not in repo mode, or the repo directory has no parent to scan siblings of.
    /// `on_progress(scanned, total)` is invoked once per file across every project combined — see
    /// [`SessionRepo::list_all_with_progress`](crate::session_store::SessionRepo::list_all_with_progress).
    ///
    /// Runs on `spawn_blocking` — see [`Self::list_with_progress`]'s identical doc comment for why a
    /// cross-project scan (bigger than a single repo's own) must not run directly on the caller's own
    /// task.
    async fn list_all_with_progress(
        &self,
        on_progress: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> std::io::Result<Vec<SessionMeta>> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let root = repo
            .dir()
            .parent()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "session directory has no parent to list other projects from",
                )
            })?
            .to_path_buf();
        match tokio::task::spawn_blocking(move || {
            SessionRepo::list_all_with_progress(&root, on_progress)
        })
        .await
        {
            Ok(result) => result,
            // `list_all_with_progress` never panics itself and this task is never cancelled — same
            // reasoning as `persist_blocking`'s identical re-raise.
            Err(e) => std::panic::resume_unwind(e.into_panic()),
        }
    }

    /// Set the current session's title.
    fn set_title(&mut self, title: &str) -> std::io::Result<()> {
        if let Some(store) = &mut self.store {
            store.set_title(title)?;
            self.meta = store.meta().clone();
        }
        Ok(())
    }

    /// Set (`label: Some`) or clear (`label: None`) a bookmark on `target_id` — see
    /// `SessionStore::set_label`. Unlike `set_title` (meaningful even without persistence, since
    /// `self.meta` always exists), a label only makes sense against a persisted tree entry, so this
    /// errors rather than silently no-oping when persistence isn't configured — the same contract
    /// `switch_branch` already uses for a `target_id`-based command.
    fn set_label(&mut self, target_id: &str, label: Option<&str>) -> std::io::Result<()> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;
        store.set_label(target_id, label)
    }

    /// Append an opaque, caller-defined tree entry as a child of the active tip and advance the tip to
    /// it — see `SessionStore::append_custom`. Same persistence-required contract as `set_label` (a
    /// custom entry only makes sense against a persisted tree). Returns the new entry's id.
    fn append_custom(
        &mut self,
        kind: impl Into<String>,
        data: serde_json::Value,
    ) -> std::io::Result<String> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;
        store.append_custom(kind, data)
    }

    /// The label currently set on `target_id`, if any — see `SessionStore::get_label`. Same
    /// persistence-required contract as `set_label`.
    fn get_label(&self, target_id: &str) -> std::io::Result<Option<&str>> {
        let store = self.store.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;
        Ok(store.get_label(target_id))
    }

    /// Every branch in the current session's tree (empty unless persistence is configured — a
    /// pure in-memory session, with no `--session-file`/`--session-dir`, has no tree to report).
    fn list_branches(&self) -> Vec<BranchInfo> {
        self.store
            .as_ref()
            .map(SessionStore::list_branches)
            .unwrap_or_default()
    }

    /// Every node in the current session's tree — every message on every branch, unlike
    /// `list_branches`' leaves-only view (empty unless persistence is configured).
    fn tree(&self) -> Vec<crate::session_store::TreeNode> {
        self.store
            .as_ref()
            .map(SessionStore::tree)
            .unwrap_or_default()
    }

    /// Every abandoned branch's full message chain, for `export_html` — see
    /// `SessionStore::abandoned_branches` (empty unless persistence is configured).
    fn abandoned_branches(&self) -> Vec<(usize, Vec<agent_core::Message>)> {
        self.store
            .as_ref()
            .map(SessionStore::abandoned_branches)
            .unwrap_or_default()
    }

    /// Every non-message session event recorded so far, for `export_html` — see
    /// `SessionStore::export_events` (Track L36; empty unless persistence is configured).
    fn export_events(&self) -> &[crate::session_store::ExportEvent] {
        self.store
            .as_ref()
            .map(SessionStore::export_events)
            .unwrap_or_default()
    }

    /// Ids of the active path's messages, root-first — parallel to the live `Session.messages` when
    /// persistence is configured; empty in pure in-memory mode (nothing to id). What `get_messages`
    /// tags each message with, and what a client names as `switch_branch`'s `target_id`.
    fn active_ids(&self) -> &[String] {
        self.store
            .as_ref()
            .map(SessionStore::active_ids)
            .unwrap_or(&[])
    }

    /// Every user-turn message anywhere in the session's tree — every branch, not just the active path
    /// — as `(id, Message)` pairs (empty unless persistence is configured). What `get_fork_messages`
    /// surfaces, matching pi's own whole-tree `getUserMessagesForForking` rather than only the active
    /// path's own chain.
    fn all_user_messages(&self) -> Vec<(String, agent_core::Message)> {
        self.store
            .as_ref()
            .map(SessionStore::all_user_messages)
            .unwrap_or_default()
    }

    /// Switch the active branch to `target_id` — or, when `before` is set, to `target_id`'s own
    /// *parent* instead, which is the tree's root (before any message) when `target_id` is the very
    /// first message. That lets a client redo the first message in place (pi's own
    /// `SessionManager::resetLeaf`) without a separate root sentinel on the wire: the client already
    /// has real entry ids from `get_fork_messages`, so `{target_id: <first message>, before: true}`
    /// means exactly that, mirroring `fork`'s identical `before` semantics.
    ///
    /// When `summarize` and the branch being left behind has unsummarized activity (see
    /// `SessionStore::abandoned_by_switch`/`abandoned_to_root`), generates a summary via `agent` and
    /// persists it *before* switching — mirroring pi's `navigateTree`. A client-requested abort
    /// (`Error::Cancelled`) leaves the session completely unchanged (see the `Err` match arm below); any
    /// *other* summarization failure (a network error, the model returning an error) is fatal to the
    /// whole navigation too — Fix 3 (pi-parity gap): this used to log the failure and switch anyway
    /// (`eprintln!("serve: branch summarization failed, switching anyway: ...")`), matching neither pi's
    /// own `packages/coding-agent/src/core/agent-session.ts`'s `navigateTree` (which `throw`s on any
    /// non-abort summarization error — see `generateBranchSummary` in
    /// `packages/coding-agent/src/core/compaction/branch-summarization.ts`, whose `{error}` result
    /// `navigateTree` turns into that throw, aborting the whole navigation before ever switching the
    /// active leaf) nor this RPC's own error-handling convention elsewhere (`switch_active_with_summary`
    /// failing to *persist* an already-generated summary, just below, still falls back to a plain switch —
    /// a different, later failure mode this fix does not touch: the summary text already exists there,
    /// only the durable record of it is what's missing, so blocking navigation over that would strand the
    /// client on the old branch to save a recap it already generated once and could just regenerate).
    /// Silently proceeding as if nothing happened gave a caller no way to know the recap never got made
    /// and no way to retry — returning `Err` here makes the RPC dispatch's own generic `Err` handling
    /// surface it as a normal failed response, so the client sees the failure and can choose to retry
    /// (with `summarize:false` if it wants to switch without one) rather than silently getting a switch
    /// it didn't ask for.
    ///
    /// `custom_instructions`, when given, steers *what* the branch recap emphasizes — the same
    /// "Additional focus" framing manual `compact` already supports — threaded straight through to
    /// [`Agent::summarize_branch`]; ignored (no summarization call happens at all) when `summarize` is
    /// `false`. `replace_instructions` (Task #17, pi-parity fix) forwards straight through to that same
    /// call's own parameter of the same name — `true` uses `custom_instructions` as the *entire*
    /// instruction section instead of appending it after the default structured template; a no-op when
    /// `custom_instructions` is `None`, same as `summarize_branch`'s own doc comment describes.
    ///
    /// Returns the resolved target alongside the switched-to `Session` — `None` when `before` resolved
    /// to the tree's own root — so the caller can restore the correct model/thinking-level for
    /// wherever the session actually landed (see [`Self::model_and_level_at_opt`]) instead of querying
    /// against the raw, pre-resolution `target_id` argument.
    #[allow(clippy::too_many_arguments)]
    async fn switch_branch(
        &mut self,
        agent: &Agent,
        target_id: &str,
        before: bool,
        summarize: bool,
        custom_instructions: Option<&str>,
        replace_instructions: bool,
        cancel: &CancellationToken,
    ) -> std::io::Result<(Session, Option<String>)> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;

        // Resolve `before` up front, rejecting an unknown `target_id` here rather than wasting a
        // summarization call on a switch that's about to fail anyway. `resolved: None` means "reset to
        // root" — everything else (including a normal, non-`before` switch) names a real node.
        let resolved: Option<String> = if before {
            match store.parent_of(target_id) {
                Some(parent) => parent,
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("no message with id {target_id} in this session"),
                    ));
                }
            }
        } else {
            Some(target_id.to_string())
        };

        // A summary, once generated, is applied via `switch_active_with_summary`/
        // `switch_active_to_root_with_summary` — it both persists the recap *and* installs it as the
        // new active tip in one step, so it actually reaches the model on the next turn. Anything else
        // (nothing abandoned, `summarize` off, the model call failed or returned nothing) falls through
        // to a plain `switch_active`/`switch_active_to_root`.
        let mut summary_to_apply: Option<(String, String, BranchSummaryDetails)> = None;
        if summarize {
            let abandoned = match &resolved {
                Some(real_target) => store.abandoned_by_switch(real_target),
                None => store.abandoned_to_root(),
            };
            if !abandoned.is_empty() {
                if let Some(from_id) = store.active_ids().last().cloned() {
                    let (ids, messages): (Vec<String>, Vec<agent_core::Message>) =
                        abandoned.into_iter().unzip();
                    match agent
                        .summarize_branch(
                            &messages,
                            cancel,
                            custom_instructions,
                            replace_instructions,
                        )
                        .await
                    {
                        Ok(summary) if !summary.trim().is_empty() => {
                            let (mut read_files, mut modified_files) =
                                agent_core::compaction::extract_file_ops(&messages);
                            // Fold forward any nested branch summary's own file-tracking within this
                            // same abandoned range — otherwise a detour-off-a-detour loses the earlier
                            // summary's file awareness the moment only its prose survives to be scanned
                            // (see `SessionStore::branch_summary_details_within`).
                            let prior = store.branch_summary_details_within(&ids);
                            for f in prior.read_files {
                                if !read_files.contains(&f) {
                                    read_files.push(f);
                                }
                            }
                            for f in prior.modified_files {
                                if !modified_files.contains(&f) {
                                    modified_files.push(f);
                                }
                            }
                            let details = BranchSummaryDetails {
                                read_files,
                                modified_files,
                                summarized_messages: messages.len() as u64,
                            };
                            summary_to_apply = Some((summary, from_id, details));
                        }
                        Ok(_) => {} // empty summary — nothing worth recording
                        // A client-requested abort mid-summarization: unlike a genuine failure below,
                        // this must not fall through to switching anyway — pi's own contract
                        // (`navigateTree`'s `abortBranchSummary`) leaves the session completely
                        // unchanged on cancel, distinct from `cancelled: true` in the response the
                        // caller builds from this `Err`.
                        Err(agent_core::Error::Cancelled) => {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Interrupted,
                                "branch summarization cancelled",
                            ));
                        }
                        // Fix 3: any other summarization failure is fatal to the whole navigation — the
                        // switch does not happen, and the RPC dispatch's generic `Err` handling reports
                        // this as a failed `switch_branch` response the client can see and retry.
                        Err(e) => {
                            return Err(std::io::Error::other(format!(
                                "branch summarization failed: {e}"
                            )));
                        }
                    }
                }
            }
        }

        let messages = match (&resolved, summary_to_apply) {
            (Some(real_target), Some((summary, from_id, details))) => {
                match store.switch_active_with_summary(real_target, summary, from_id, details) {
                    Ok(messages) => messages,
                    Err(e) => {
                        // Recording the summary failed (a disk error mid-rewrite) — the switch itself
                        // must still succeed rather than leaving the client stuck on the old branch.
                        eprintln!("serve: failed to persist branch summary, switching anyway: {e}");
                        store.switch_active(real_target)?
                    }
                }
            }
            (None, Some((summary, from_id, details))) => {
                match store.switch_active_to_root_with_summary(summary, from_id, details) {
                    Ok(messages) => messages,
                    Err(e) => {
                        eprintln!("serve: failed to persist branch summary, switching anyway: {e}");
                        store.switch_active_to_root()?
                    }
                }
            }
            (Some(real_target), None) => store.switch_active(real_target)?,
            (None, None) => store.switch_active_to_root()?,
        };
        self.meta = store.meta().clone();
        let mut session = Session::new();
        session.messages = Arc::new(messages);
        Ok((session, resolved))
    }

    /// Record that the active model changed — a no-op when persistence isn't configured (in-memory
    /// mode has nothing to append to). The caller (`set_model`/`cycle_model`) only calls this once it
    /// has already confirmed the model actually changed.
    fn record_model_change(&mut self, model: &str) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = store.record_model_change(model);
        if let Err(e) = &r {
            eprintln!("serve: failed to record model change: {e}");
        }
        r
    }

    /// Same idea as [`Self::record_model_change`], for the portable thinking level.
    fn record_thinking_level_change(&mut self, level: &str) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = store.record_thinking_level_change(level);
        if let Err(e) = &r {
            eprintln!("serve: failed to record thinking-level change: {e}");
        }
        r
    }

    /// The model/thinking-level to restore when switching to `target_id` (see
    /// `SessionStore::model_at`/`thinking_level_at`). Both always resolve to *something* concrete, never
    /// "leave whatever's currently active alone" — that would let a sibling branch's `set_model`/
    /// `cycle_thinking_level` bleed across a switch to a branch that never touched it, instead of
    /// landing on that branch's own actual state. The model falls back to this session's own
    /// creation-time model (`self.meta.model`) if nothing was ever recorded reaching that point — every
    /// session has a real starting model. The thinking level has no equivalent creation-time baseline
    /// in `SessionMeta`, so the caller passes `process_starting_level` (the level the process itself
    /// started at, from `--reasoning-effort`/`ThinkingLevel::Off`) as its fallback instead — the same
    /// "no change recorded yet" logic, just sourced from the process's own starting config rather than
    /// the session's.
    fn model_and_level_at(
        &self,
        target_id: &str,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        let Some(store) = &self.store else {
            return (self.meta.model.clone(), process_starting_level);
        };
        (
            store
                .model_at(target_id)
                .map(str::to_string)
                .unwrap_or_else(|| self.meta.model.clone()),
            store
                .thinking_level_at(target_id)
                .and_then(agent_core::ThinkingLevel::parse)
                .unwrap_or(process_starting_level),
        )
    }

    /// [`Self::model_and_level_at`], but for a `switch_branch` target that may have resolved to the
    /// tree's own root (`before: true` reaching the very first message) instead of a real node —
    /// `model_at`/`thinking_level_at` require an existing id and can't express "root" directly, so
    /// `None` here reads `SessionStore::model_at_root`/`thinking_level_at_root` instead.
    fn model_and_level_at_opt(
        &self,
        target_id: Option<&str>,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        let Some(target_id) = target_id else {
            let Some(store) = &self.store else {
                return (self.meta.model.clone(), process_starting_level);
            };
            return (
                store
                    .model_at_root()
                    .map(str::to_string)
                    .unwrap_or_else(|| self.meta.model.clone()),
                store
                    .thinking_level_at_root()
                    .and_then(agent_core::ThinkingLevel::parse)
                    .unwrap_or(process_starting_level),
            );
        };
        self.model_and_level_at(target_id, process_starting_level)
    }

    /// [`Self::model_and_level_at`] resolved at the *currently active* session's own tip — what a
    /// caller that just opened a different session entirely (`switch`), rather than a different branch
    /// within the same one (`switch_branch`), needs: `switch`'s `self.store = Some(store)` swap means
    /// `model_and_level_at`'s own tree lookups already operate on the newly-opened session, so only the
    /// target id needs to come from *this* session's own active tip instead of an RPC-supplied one.
    fn model_and_level_at_active(
        &self,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        let Some(target_id) = self
            .store
            .as_ref()
            .and_then(|s| s.active_ids().last())
            .cloned()
        else {
            return (self.meta.model.clone(), process_starting_level);
        };
        self.model_and_level_at(&target_id, process_starting_level)
    }
}

fn not_in_repo_mode() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "not in repo mode (start serve with --session-dir)",
    )
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

/// The current short branch name at `cwd` (Task #25, pi-parity fix), for `get_state`'s `git_branch`
/// field — matters more for this crate than for pi, since the whole point of the RPC protocol is
/// letting a client with no shared filesystem drive the agent remotely; without this, such a client has
/// no way at all to learn which branch the agent's tools are actually operating against.
///
/// `None` — never an error surfaced to the caller — when `cwd` isn't inside a git repo, `HEAD` is
/// detached (matching `git symbolic-ref`'s own behavior: it only ever resolves a real branch ref, not a
/// raw commit), `git` itself isn't installed, or the lookup fails for any other reason: a client polling
/// `get_state` shouldn't have an unrelated git hiccup fail the whole call. Spawned via
/// `tokio::process::Command` (not a blocking `std::process::Command`) so this never stalls the control
/// loop's own task the way a blocking subprocess wait would — see `persist_blocking`'s doc comment for
/// the same "don't block this task" reasoning applied to disk I/O instead of a subprocess. No
/// filesystem watcher or caching: a plain request/response poll has nothing to invalidate, and a `git`
/// invocation is cheap enough to just redo every call.
async fn git_branch(cwd: &std::path::Path) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(["symbolic-ref", "--short", "HEAD"])
        .current_dir(cwd)
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8(output.stdout).ok()?;
    let branch = branch.trim();
    (!branch.is_empty()).then(|| branch.to_string())
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
    let (input_tx, input_rx) = mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        let mut lines = BufReader::new(tokio::io::stdin()).lines();
        // A malformed-UTF-8 read error is treated as a clean EOF, matching the pre-refactor idle/busy
        // read loops (killing a long-running process over one bad stdin byte is far more disruptive).
        while let Ok(Some(line)) = lines.next_line().await {
            if input_tx.send(line).is_err() {
                break; // the session ended; nothing left to feed
            }
        }
        // Dropping `input_tx` closes `input_rx` → the session observes EOF and shuts down.
    });

    let (conn_tx, mut conn_rx) = mpsc::unbounded_channel::<OutFrame>();
    // stdio has exactly one permanent "connection": stdout. Register it in the fanout as the sole sink.
    let out_conn: SharedOutConn =
        Arc::new(std::sync::Mutex::new(crate::serve::OutFanout::default()));
    lock_ignoring_poison(&out_conn).add(conn_tx);
    let stdout_task = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(frame) = conn_rx.recv().await {
            let Some(mut line) = frame_to_line(frame) else {
                continue;
            };
            // The frame's bytes and its terminator go out as one write.
            line.push('\n');
            if let Err(e) = write_frame(&mut out, &line).await {
                eprintln!("serve: stdout write failed, shutting down writer: {e}");
                break;
            }
        }
    });

    // Stdio has no supervisor and no reaper, so the `running` flag is inert here — a throwaway.
    let running = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sig = serve_session(cfg, input_rx, out_conn, running).await?;
    // The session has ended (and dropped its `out_conn` clones), so `conn_rx` is now closed — this
    // just reaps the stdout task after it flushes the last frame.
    let _ = stdout_task.await;
    Ok(sig)
}

/// The transport-agnostic control loop: one session, driven by a stream of command lines
/// (`input_rx`) and emitting frames to whatever connection is currently attached (`out_conn`). The
/// stdio [`serve`] wrapper feeds it (stdin lines / a stdout task); [`crate::serve_ws`] feeds it from a
/// WebSocket, where the connection can detach and a later one re-attach to this same still-running
/// task — because `input_rx`'s `Sender` is held by the supervisor (not the socket), a dropped
/// connection is *not* an EOF: `input_rx.recv()` simply pends until the next command, and the run
/// keeps going. The command protocol below is byte-identical across both transports.
pub(crate) async fn serve_session(
    mut cfg: ServeConfig,
    mut input_rx: mpsc::UnboundedReceiver<String>,
    out_conn: SharedOutConn,
    // Set `true` for the duration of a `prompt` run, `false` otherwise. The daemon supervisor's idle
    // reaper reads this to never reap a session with an in-flight background run (see
    // [`crate::serve_ws`]); the stdio wrapper passes a throwaway it never observes.
    running: Arc<std::sync::atomic::AtomicBool>,
) -> Result<Option<Signal>, Box<dyn std::error::Error>> {
    let mut timing = crate::timing::StartupTiming::new();
    let (mut persistence, mut session) = Persistence::open(&cfg)?;
    timing.mark("open persistence");

    // `--name`: only for a genuinely fresh session (no messages, no title yet) — a resumed session
    // (repo mode reattaching to an existing cwd session, or a `--session-file` that already has
    // content) must not be silently renamed just because the process happened to be started with this
    // flag again. Deliberately diverges from pi, whose `--name` renames unconditionally on every
    // invocation (last-write-wins) — a script that resumes the same session on a schedule would
    // otherwise get its title rewritten to the same value every single run, and there's no way for an
    // operator to tell "rename this" from "just start it the way I always do" apart on pi's side.
    if let Some(name) = &cfg.name {
        if session.messages.is_empty() && persistence.meta.title.is_none() {
            if let Err(e) = persistence.set_title(name) {
                eprintln!("serve: failed to set initial session name: {e}");
            }
        }
    }

    // Assemble the system prompt from the base identity + this repo's project instructions + skills +
    // environment, so the agent behaves like it belongs in the working directory. Split into a static
    // half (this discovery-based block — expensive, rebuilt only on `set_model`/`set_thinking`/`reload`)
    // and a cheap dynamic footer (current date/cwd, recomputed before every `prompt` via `full_system`)
    // so a long-running `serve` process doesn't re-walk the filesystem every turn just for the date.
    let cwd = crate::session_store::canonical_cwd(&std::env::current_dir().unwrap_or_default());
    let has_gated_resources = crate::trust_store::has_trust_gated_resources(&cwd);
    let mut project_trusted = resolve_project_trust(
        cfg.trust_project,
        cfg.force_untrusted,
        cfg.default_project_trust,
        crate::trust_store::TrustStore::open_default().lookup(&cwd),
        has_gated_resources,
    );
    // Agent definitions are trust-gated exactly like skills (a project-local `.claude/agents/*.md` body
    // is injected verbatim as a child's system prompt), so they're discovered here — after trust is
    // resolved — not by `main.rs` at `ServeConfig`-construction time, where the interactive trust grant
    // hasn't happened yet. Stored on `cfg` so `build_agent`/`build_tools` (which take only `&cfg`) can
    // reach them on every rebuild without re-walking. The `reload` arm re-discovers below, since trust
    // (and the on-disk definitions) can change mid-process.
    cfg.agents = crate::agents::discover(&cwd, project_trusted);
    // Reap any subagent worktree orphaned by a previous crash of a process against this repo.
    crate::worktree::sweep(&cwd);

    // Track L32 (pi-parity fix): mirrors `main.rs`'s identical warning for `run` — an untrusted
    // project with a `SYSTEM.md`/skills/prompts on disk silently skipped all of them with no signal at
    // all that anything was there. Re-checked (not just at startup) on every `reload`, below, since
    // trust can change mid-process (`agent trust`/`--trust-project` since startup) without a restart.
    if !project_trusted && has_gated_resources {
        eprintln!(
            "warning: {} has a project-local SYSTEM.md/APPEND_SYSTEM.md, skills, prompt templates, or a \
             settings.json on disk, but the project isn't trusted, so they were skipped — pass \
             --trust-project or run `agent trust {}` to enable them (a project's own settings.json \
             additionally requires a *persisted* `agent trust`, not just a one-off --trust-project — see \
             `settings::effective_settings_for_cwd`'s doc comment)",
            cwd.display(),
            cwd.display()
        );
    }

    // Slash-command prompt templates (`/name args`) and discoverable skills, for `get_commands`, for
    // expanding a `/name`/`/skill:name` prompt before it reaches the model, and — for skills — to
    // advertise in the system prompt below. Discovered *before* building the system prompt (rather
    // than after, as this and the `reload` arm's own discovery used to be ordered) so
    // `build_static_system_prompt` can take the already-discovered list instead of re-walking the same
    // skills directories a second time itself.
    //
    // Prompt templates are gated on trust wholesale: an untrusted repo's `.claude/prompts` is
    // attacker-controlled instructions, so it's neither advertised nor invocable until the directory is
    // trusted — otherwise `/name` would inject arbitrary content into context regardless of trust.
    //
    // Skills are *not* gated wholesale — only the project-local root is (`skills::discover_with_diagnostics`'s
    // own `project_trusted` param): the user-global root (`~/.claude/skills`) is the operator's own
    // machine, not something the current project checkout controls, so an untrusted project must not
    // blank out the user's own skills along with its own.
    //
    // The `_with_diagnostics` variant also reports name collisions (the same `/name` or skill name
    // shadowed across roots), surfaced via `get_commands`'s `collisions` field rather than silently
    // resolved with no way for a client to notice.
    //
    // `--no-skills`/`--no-prompt-templates` skip *standard-root* discovery outright rather than
    // discovering and then discarding — matching `run`'s identical flags (`main.rs`), and avoiding a
    // needless filesystem walk when the operator has already said neither standard root is wanted. An
    // explicit `--skill`/`--prompt-template` extra path is still honored even so — pi's own
    // `noSkills`/`noPromptTemplates` do the same (a documented, tested combination; see
    // `skills::discover_extra_only`'s doc comment — pi-parity fix, M2).
    let (mut prompt_templates, mut prompt_collisions) = if cfg.no_prompt_templates {
        crate::prompts::discover_extra_only(&cfg.extra_prompt_template_paths)
    } else {
        crate::prompts::discover_with_diagnostics(
            &cwd,
            project_trusted,
            &cfg.extra_prompt_template_paths,
        )
    };
    let (mut skills, mut skill_collisions) = if cfg.no_skills {
        crate::skills::discover_extra_only(&cfg.extra_skill_paths)
    } else {
        crate::skills::discover_with_diagnostics(&cwd, project_trusted, &cfg.extra_skill_paths)
    };
    timing.mark("discover prompt templates/skills");

    // Skills are discovered by path, not inlined into the prompt — invoking one relies on the model
    // being able to open its `SKILL.md` itself, so advertising them at all when `read` isn't registered
    // (a restricted `--tools`/`--exclude-tools` invocation) just adds dead weight (pi-parity fix).
    // Tools are fixed for the whole process (see `build_agent`'s doc comment), so this one check is
    // reused verbatim by `reload`'s own rebuild below rather than re-deriving it.
    let startup_tools = build_tools(&cfg, cfg.image_auto_resize);
    let has_read = startup_tools.get("read").is_some();
    let has_todo = startup_tools.get(crate::tools::todo::NAME).is_some();

    // Persistent memory (mirrors `run`): resolve the backend once here where `cwd` is known, then wrap it
    // as a shared `Arc<dyn Tool>` that every `build_agent` rebuild registers (so a mid-run `set_model`
    // doesn't drop it). `--no-memory` disables it; a bad backend DSN is fatal at startup, before any
    // session runs. The `MEMORY.md` index is read once and injected into the system prompt — a session's
    // memory is surfaced from its start (Claude Code's auto-memory model).
    let memory_backend: Option<Arc<dyn crate::memory::MemoryBackend>> =
        if cfg.no_memory || cfg.no_tools {
            None
        } else {
            Some(
                crate::memory::open(cfg.memory.as_deref(), &cwd)
                    .map_err(Box::<dyn std::error::Error>::from)?,
            )
        };
    let has_memory = memory_backend.is_some();
    let memory_tool: Option<Arc<dyn agent_core::Tool>> = memory_backend
        .clone()
        .map(|b| Arc::new(crate::tools::memory::Memory::new(b)) as Arc<dyn agent_core::Tool>);
    // Re-read at each `static_system` rebuild (a `set_model`/`set_thinking` switch, or a `prompt` whose
    // output-schema changed — see `current_memory_index`) so a session that has written new memories
    // injects the *current* index, not a stale startup snapshot. Steady-state within one model, the
    // `memory` tool's own `view` is always live, so this need only refresh where the prompt is already
    // being rebuilt — one small file read, never per turn.
    let mut memory_index: Option<String> = current_memory_index(&memory_backend).await;

    // The interactive approval gate. One per session, shared by every `build_agent` rebuild and by every
    // subagent child, so a remembered decision isn't re-asked and eight parallel children can't spam
    // eight simultaneous questions. `pending_approvals` is the map the `approve` command resolves
    // against — held here rather than inside the gate so both the busy and idle command loops can reach
    // it while a run is blocked inside the gate itself.
    let (approval, pending_approvals) = if cfg.approve.is_off() {
        (None, None)
    } else {
        let (gate, pending) = ServeApprovalGate::new(
            out_conn.clone(),
            cfg.approval_timeout,
            persistence.session_id(),
        );
        let runtime = crate::approval::ApprovalRuntime::new(gate, cfg.approve.clone());
        (Some(runtime), Some(pending))
    };

    // `structured_output` is installed per-`prompt` (see that arm's `output_schema` handling), not at
    // startup: one session can answer one request in prose and the next as typed JSON. The `OutputSlot`
    // outlives every rebuild — it is this session's return channel, and a `set_model` mid-run must not
    // drop the answer that landed in it. `current_output_spec` is what an incoming `prompt` is compared
    // against to decide whether anything actually has to be rebuilt.
    let output_slot = crate::tools::structured_output::OutputSlot::new();
    let mut structured_output: Option<Arc<crate::tools::structured_output::StructuredOutput>> =
        None;
    let mut current_output_spec: Option<(Value, Option<String>)> = None;

    let mut static_system =
        crate::resources::build_static_system_prompt(&crate::resources::PromptOptions {
            base: None,
            default_base: &cfg.system,
            append: cfg.append_system.as_deref(),
            cwd: &cwd,
            include_context_files: cfg.context_files,
            skills: &skills,
            has_read,
            has_todo,
            has_structured_output: structured_output.is_some(),
            has_memory,
            memory_index: memory_index.as_deref(),
            project_trusted,
            agents: &cfg.agents,
        });
    timing.mark("build static system prompt");

    // Task #50: the same two operator-supplied overrides also drive the *whole-run* retry loop
    // (`crate::retry::RunRetryPolicy`) below, not just `client`'s own pre-connect/mid-stream layer just
    // above — previously `--retry-max-retries`/`--retry-base-delay-ms` had no effect on the `"prompt"`
    // command's own auto-retry loop.
    let run_retry_policy = crate::retry::RunRetryPolicy::from_overrides(
        cfg.retry_max_retries,
        cfg.retry_base_delay_ms,
    );

    // The model, thinking budget, and auto-compaction flag are runtime-switchable; everything else
    // (transport, tools, system prompt, loop bounds, cache settings) is fixed for the process.
    // `build_agent` folds the mutable trio into a fresh `Agent` whenever any of them changes.
    //
    // Task #5 (pi-parity fix): when the operator didn't explicitly pass `--model`/`--reasoning-effort`
    // for *this* invocation (`cfg.model_explicit`/`cfg.reasoning_effort_explicit` — see their own doc
    // comments), prefer whatever this reattached session's own active tip was actually last running on
    // over the CLI-resolved default, via the same `model_and_level_at_active` lookup `switch_session`
    // already uses. Without this, `set_model gpt-5` in a live session, disconnect, then a plain `serve`
    // in the same directory with no `--model` re-passed silently reverted to the global default for the
    // rest of the process — the exact "bleed" class `switch_session`/`switch_branch` are hardened
    // against (see their own doc comments), just left unguarded at ordinary process restart.
    // `model_and_level_at_active` already falls back to `cfg.model`/the CLI-resolved level (via
    // `self.meta.model`, which a fresh session's own `SessionMeta` is seeded from — see
    // `Persistence::open`) when nothing was ever recorded reaching the active tip, so this is a no-op
    // for a genuinely fresh session or pure in-memory mode. Model and level are resolved independently,
    // matching `--model`/`--reasoning-effort` being independent flags — an explicit `--model` with no
    // `--reasoning-effort` must still pick up the session's own last thinking level, not the process's
    // bare `Off` default, and vice versa.
    // Fix 1 (pi-parity gap): `cfg.reasoning_effort` is `None` both when the operator explicitly
    // requested no reasoning effort be set and when they simply never passed `--reasoning-effort`/had
    // no stored default — previously both collapsed to a bare `ThinkingLevel::Off` here, so a bare
    // `serve`/`run` invocation with no flags on a reasoning-capable model wire-disabled thinking
    // outright instead of picking any default depth. `default_reasoning_effort_for_model` supplies pi's
    // own "medium" default in that case (see its own doc comment); a model with no reasoning mechanism
    // at all still falls through to `Off`, same as before.
    //
    // Task #29 (pi-parity fix): that "medium" fallback must never fire when `cfg.reasoning_effort_explicit`
    // is `true` — an explicit `--model <pattern>:off` (or `--reasoning-effort` with no portable "off"
    // value of its own) resolves `cfg.reasoning_effort` to this same bare `None`, previously
    // indistinguishable here from "the operator never said anything", so it was silently promoted back
    // to the default depth instead of actually starting off. `main.rs` already computes
    // `reasoning_effort_explicit` correctly for both `run` and `serve` (see `ServeConfig::
    // reasoning_effort_explicit`'s own doc comment) — this is the one place that value wasn't consulted.
    let cfg_level = cfg
        .reasoning_effort
        .or_else(|| {
            if cfg.reasoning_effort_explicit {
                None
            } else {
                default_reasoning_effort_for_model(&cfg.model)
            }
        })
        .map(agent_core::ThinkingLevel::from)
        .unwrap_or(agent_core::ThinkingLevel::Off);
    let (session_model, session_level) = persistence.model_and_level_at_active(cfg_level);
    let (mut current_model, restored_level) = resolve_startup_model_and_level(
        &cfg.model,
        cfg_level,
        cfg.model_explicit,
        cfg.reasoning_effort_explicit,
        session_model,
        session_level,
    );
    // `cycle_model`'s candidate list: the operator-scoped `--models` set when given (parsed by
    // `resolve_model_scope` — comma-separated patterns, each an exact id, a glob against
    // `available_models()`, or either suffixed with `:<thinking-level>` to pin that entry's depth —
    // pi's own `--models`/`resolveModelScopeWithDiagnostics`), else the full known-model hint list.
    // Fixed for the process, like `tools` — there's no runtime RPC to change it. `get_available_models`
    // is deliberately NOT scoped by this — see that handler's own comment.
    let scoped_models: Vec<ScopedModel> = resolve_model_scope(&cfg.models, available_models());
    let cycle_models: Vec<String> = if scoped_models.is_empty() {
        available_models().iter().map(|s| s.to_string()).collect()
    } else {
        scoped_models.iter().map(|m| m.id.clone()).collect()
    };
    let mut current_thinking = cfg.thinking;
    // The portable thinking-depth level (see `agent_core::ThinkingLevel`) — the runtime-mutable
    // counterpart to `cfg.reasoning_effort` (or, per the Task #5 restoration above, whatever this
    // reattached session's own active tip last recorded), seeded from it so a process started with
    // `--reasoning-effort` keeps that depth until `cycle_thinking_level`/`set_reasoning_effort` change
    // it. Unlike `current_thinking` (an explicit raw-budget override that wins when present), this
    // always takes effect: `build_agent` derives both the thinking budget *and* the reasoning effort
    // from it via `agent_core::thinking_for_level`, correctly for whichever mechanism `current_model`
    // actually uses.
    // Clamped against `current_model`'s capabilities immediately: a model that has a reasoning
    // mechanism but can't explicitly disable it (`reasoning_disableable == false`, e.g. most of the
    // OpenAI gpt-5 codex/pro line) has no legal `Off` state at all — leaving it there would silently
    // omit the reasoning field on every request and let the provider apply its own hidden default
    // effort, with the operator believing reasoning is off. See `agent_core::clamp_thinking_level`.
    let starting_level =
        agent_core::clamp_thinking_level(&agent_core::capabilities(&current_model), restored_level);
    // The runtime-mutable level starts at `starting_level`, but `switch_branch` needs the original
    // starting value too — as the fallback for a branch that never recorded its own thinking-level
    // change (see `Persistence::model_and_level_at`), so switching to it lands on the process's real
    // starting depth instead of silently keeping whatever a *different* branch last set.
    let mut current_level = starting_level;
    // Track L26 (pi-parity fix): `--no-compaction`/`AI_AGENT_NO_COMPACTION` (`cfg.no_compaction`) wins
    // outright when given; otherwise a persisted `agent settings` override survives across restarts
    // where the flag/env is absent (matching `default_project_trust`'s identical precedence tier —
    // `settings::Settings::compaction_enabled`'s own doc comment), finally defaulting to enabled.
    // `settings_store` is kept open for the rest of this process so `set_auto_compaction`, below, can
    // write a later runtime toggle straight back through to the same file.
    let mut settings_store = crate::settings::SettingsStore::open_default();
    let mut current_auto_compaction = if cfg.no_compaction {
        false
    } else {
        settings_store.get().compaction_enabled.unwrap_or(true)
    };
    // Mid-stream transport-failure retry (`agent_core::Agent::with_auto_retry`) — on by default;
    // `set_auto_retry` lets an operator debugging a flaky network hop disable it to see the raw failure
    // on the very first hiccup instead of after several silent retries.
    let mut current_auto_retry = true;
    // Pi-parity fix (pass 20): `cfg.block_images`/`cfg.image_auto_resize` are already fully resolved by
    // `main.rs` (explicit flag, then a persisted `agent settings --block-images`/`--image-auto-resize`
    // default), but — unlike `current_auto_compaction`/`current_auto_retry` just above — `build_agent`
    // used to read them straight back off `cfg` on every rebuild instead of a runtime-mutable local,
    // so there was no `set_block_images`/`set_image_auto_resize` RPC arm that could ever change them for
    // a live process. These start at whatever `cfg` already resolved, and from here on are the sole
    // source `build_agent` consults for both.
    let mut current_block_images = cfg.block_images;
    let mut current_image_auto_resize = cfg.image_auto_resize;
    // Shared across every `build_agent` rebuild for this process's lifetime, so file-mutation
    // exclusivity (same-path `edit`/`write` calls) survives a `set_model`/`set_thinking` rebuild.
    let write_locks = Arc::new(agent_core::WriteLockRegistry::new());
    // The subagent context, built once and reused across every rebuild — its transport factory closes
    // over `cfg` and re-resolves per child model, so nothing here changes when the *parent's* model does.
    // `None` when no agent definitions were discovered: no delegable agents, no `subagent` tool.
    // `mut` because `reload` rebuilds it when trust or the on-disk definitions change and then rebuilds
    // the agent, so a mid-session-added agent becomes delegable immediately.
    let mut subagent_ctx: Option<Arc<crate::tools::subagent::SubagentCtx>> =
        if cfg.agents.is_empty() {
            None
        } else {
            Some(build_subagent_ctx(
                &cfg,
                &cwd,
                project_trusted,
                &current_model,
                &write_locks,
                &skills,
                memory_backend.clone(),
                approval.as_ref(),
            ))
        };
    // A multi-step run (several tool round-trips) is otherwise only ever durable once it *fully*
    // completes — a crash, OOM-kill, or panic mid-run loses everything back to the turn's start,
    // including the user's own prompt. `ChannelCheckpoint` streams a cheap `Arc` snapshot of
    // `session.messages` through this channel at each durable mid-run point (see
    // `agent_core::CheckpointHook`'s doc comment for exactly which points those are); the `"prompt"`
    // arm's own busy-loop below drains it and persists incrementally, off the async task via
    // `spawn_blocking` like every other write in this file. The channel — not a `Mutex` around
    // `Persistence` itself — is what lets the checkpoint hook (called deep inside `agent.
    // run_events_steered`, which holds `&mut session` for the run's whole duration) reach persistence
    // without ever needing to borrow `session` a second time.
    let (checkpoint_tx, mut checkpoint_rx) =
        mpsc::unbounded_channel::<Arc<Vec<agent_core::Message>>>();
    let checkpoint: Arc<dyn agent_core::CheckpointHook> =
        Arc::new(ChannelCheckpoint(checkpoint_tx));
    // Resolved *here*, against `current_model` (the actual starting model, after the anti-bleed
    // session-restore check above may have overridden `cfg.model`) rather than earlier against
    // `cfg.model` directly — a credential/routing resolved for the wrong (pre-restore) model would
    // silently misroute the very first turn. Kept in an `Arc` we can clone into `build_agent`, and
    // swapped out (not just the `Agent` rebuilt) by `set_model`/`cycle_model` and any other command
    // that changes `current_model` at runtime — see `build_gateway_client`'s own doc comment.
    let mut client = Arc::new(build_gateway_client(&cfg, &current_model)?);
    let mut agent = build_agent(
        client.clone(),
        &full_system(&static_system, &cwd),
        &cfg,
        &current_model,
        current_thinking,
        current_level,
        current_auto_compaction,
        current_auto_retry,
        current_block_images,
        current_image_auto_resize,
        persistence.session_id(),
        &write_locks,
        &checkpoint,
        subagent_ctx.as_ref(),
        structured_output.as_ref(),
        memory_tool.as_ref(),
        approval.as_ref(),
    );
    timing.mark("build agent");
    // Persistent across every `prompt` call (not just the one currently in flight), so `steer`/
    // `follow_up` sent while idle queue for the *next* `prompt` instead of being rejected as an unknown
    // command. Cleared on every session switch (`new_session`/`switch_session`/`fork`/`switch_branch`)
    // so a message meant for the old session's next turn can't leak into the newly switched-to one.
    let steering = agent_core::Steering::new();
    // Task 1 (pi-parity fix, pass 19): the persisted `agent settings --steering-mode`/
    // `--follow-up-mode` defaults (or an explicit `serve --steering-mode`/`--follow-up-mode` flag),
    // already resolved by `main.rs` into `cfg.steering_mode`/`cfg.follow_up_mode` — previously only
    // `run_task` applied its own equivalent resolution to its one-shot `Steering`; a `serve` process
    // silently ignored both and always started at `QueueMode::default()` regardless of what was
    // configured, until `set_steering_mode`/`set_follow_up_mode` was called at runtime. The `set_*` RPC
    // commands still change this afterward, same as before.
    steering.set_steering_mode(cfg.steering_mode);
    steering.set_follow_up_mode(cfg.follow_up_mode);
    // The `bash` tool, looked up once from the same (possibly filtered) registry `build_agent` builds —
    // `None` when `bash` was excluded (`--exclude-tools bash` / `--no-tools`), which the `bash` RPC
    // command below reports as a clean error rather than a side door around that restriction. A direct
    // `Arc<dyn Tool>` handle, not routed through `agent`/the model loop: the host `bash` RPC command runs
    // independent of any conversation turn.
    let bash_tool = build_tools(&cfg, cfg.image_auto_resize).get("bash");
    // The same deny-list `build_agent` installs as an `AgentHooks` gate on the model's own tool calls —
    // built once here rather than inline in the `bash` RPC arm below, since `--deny-tool`/
    // `--deny-bash-pattern`/`--deny-path` are fixed for the whole process (`build_agent`'s doc comment).
    // Track L23 (pi-parity fix): the host `bash` RPC command used to call `tool.run_streaming` directly,
    // never consulting this policy at all — `--exclude-tools bash` already gated it (`bash_tool` above
    // is `None`), but `--deny-tool`/`--deny-bash-pattern`/`--deny-path` didn't, a side door around an
    // operator's own restriction for any client speaking the wire protocol directly instead of through
    // the model.
    let bash_policy = crate::policy::ToolPolicy::from_lists(
        &cfg.deny_tool,
        &cfg.deny_bash_pattern,
        &cfg.deny_path,
    );
    // Distinguishes successive host `bash` calls in their `tool_start`/`tool_progress`/`tool_end` event
    // ids — only ever one in flight at a time (see the `bash` command arm), but a stable, incrementing
    // id per call still lets a client correlate a run's own three events without ambiguity.
    let mut host_bash_seq: u32 = 0;

    // One writer task drains the FIFO and forwards each frame to whatever connection is currently
    // attached (`out_conn`); every frame (events + responses) passes through it in order, so output
    // never interleaves. The actual serialize + socket/stdout write happens in the connection's own
    // consumer (the stdio stdout task, or a WebSocket send task) — this task only preserves ordering
    // and bridges to the swappable tail.
    //
    // The channel is intentionally unbounded. The event `sink` (see `Agent::run_events`) is a
    // synchronous `FnMut`, so the producer cannot `.await` to apply backpressure; a bounded channel
    // would force `try_send`, which silently drops frames and corrupts the event stream — unacceptable
    // for a protocol. In practice the backlog is bounded by one in-flight turn's events (capped by
    // `max_steps`), drained as fast as the attached connection accepts.
    //
    // Unlike the old stdout-owning writer, this task does NOT tear down when the connection breaks: a
    // detached session (no connection, or a dead one) simply drops frames and keeps running, which is
    // exactly what lets a WebSocket client reconnect and re-attach to a still-live run. The task ends
    // only when every `out_tx` clone is dropped at teardown (`drop(out_tx)` + the joined run).
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<OutFrame>();
    let writer = {
        let out_conn = out_conn.clone();
        tokio::spawn(async move {
            while let Some(frame) = out_rx.recv().await {
                // Broadcast to every attached connection (see `OutFanout`); if none are attached the
                // frame is dropped by design. Lock is held only across the non-`await` sends.
                lock_ignoring_poison(&out_conn).broadcast(frame);
            }
        })
    };

    // Sends a frame through the writer; if the writer has shut down (stdout closed), stop the control
    // loop — there is no way to deliver any further response, so continuing would only swallow output.
    macro_rules! emit {
        ($frame:expr) => {
            if out_tx.send($frame).is_err() {
                break;
            }
        };
    }

    // pi-parity (Task 4, serve pass 19): the ordinary idle-path logic for `new_session`/
    // `switch_session`/`fork`/`clone`/`compact`, each extracted into a macro rather than a function —
    // `session`/`persistence`/`agent`/`client`/`current_model`/etc. are all plain local variables of
    // this one giant `serve` function body, and a macro lets each expansion reference them directly
    // (exactly like `emit!` already references `out_tx`) with no 15-parameter function signature. Each
    // is callable from two places: its ordinary top-level command-match arm below, and the `"prompt"`
    // arm's busy-loop deferred-until-idle path (`pending_deferred`, see its own doc comment) once a run
    // one of these commands interrupted has actually gone idle — see the module's own `serve` doc
    // comment / Task 4 for why these five self-abort-and-proceed instead of being rejected as busy.
    // `$cmd`/`$id` stand in for whichever command `Value`/response-id are in play: the outer `cmd`/`id`
    // when idle, or a deferred command's own stored `Value`/id once its abort has resolved.
    //
    // Responses are sent via a bare `out_tx.send` rather than `emit!`: `emit!`'s bare `break` breaks
    // whichever loop *textually* encloses the macro invocation, which differs between the idle call
    // site (the outer per-command loop, where breaking on a dead writer is exactly `emit!`'s intended
    // behavior) and the deferred call site (a `for` loop over `pending_deferred`, where the same bare
    // `break` would only end that short-lived drain loop instead). A bare send whose failure is
    // ignored is the same idiom `pending_abort_acks`'s own drain loop (and every other busy-mode
    // response in the `"prompt"` arm) already uses for exactly this reason.
    macro_rules! do_new_session {
        ($cmd:expr, $id:expr) => {{
            match persistence.new_session(
                &current_model,
                $cmd.get("parent_session").and_then(Value::as_str),
            ) {
                Ok(s) => {
                    session = s;
                    steering.clear();
                    let _ = out_tx.send(response(
                        $id,
                        "new_session",
                        true,
                        Some(json!({
                            "session_id": persistence.session_id(),
                            "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
                            "parent": persistence.meta.parent,
                        })),
                        None,
                    ));
                }
                Err(e) => {
                    let _ = out_tx.send(response(
                        $id,
                        "new_session",
                        false,
                        None,
                        Some(&e.to_string()),
                    ));
                }
            }
        }};
    }
    macro_rules! do_switch_session {
        ($cmd:expr, $id:expr) => {{
            match $cmd.get("session_id").and_then(Value::as_str) {
                Some(target) => match persistence.switch(target) {
                    Ok(s) => {
                        session = s;
                        steering.clear();
                        // Restore whichever model/thinking-level this session was actually last
                        // running on, the same way `switch_branch` already does — without this, the
                        // process's current global model/level (possibly set on a *different* session
                        // entirely) silently bled into the reattached session. See
                        // `model_and_level_at_active`'s doc comment.
                        let (restored_model, restored_level) =
                            persistence.model_and_level_at_active(starting_level);
                        let restored_level = agent_core::clamp_thinking_level(
                            &agent_core::capabilities(&restored_model),
                            restored_level,
                        );
                        let mut rebuild_needed = false;
                        if restored_model != current_model {
                            session.scrub_cross_model_state(&restored_model);
                            match build_gateway_client(&cfg, &restored_model) {
                                Ok(new_client) => client = Arc::new(new_client),
                                Err(e) => eprintln!(
                                    "serve: could not resolve a gateway credential for restored model {restored_model:?}: \
                                     {e} — keeping the previous client"
                                ),
                            }
                            current_model = restored_model;
                            rebuild_needed = true;
                        }
                        if restored_level != current_level {
                            current_level = restored_level;
                            current_thinking = None;
                            rebuild_needed = true;
                        }
                        if rebuild_needed {
                            agent = build_agent(
                                client.clone(),
                                &full_system(&static_system, &cwd),
                                &cfg,
                                &current_model,
                                current_thinking,
                                current_level,
                                current_auto_compaction,
                                current_auto_retry,
                                current_block_images,
                                current_image_auto_resize,
                                persistence.session_id(),
                                &write_locks,
                                &checkpoint,
                                subagent_ctx.as_ref(),
                                structured_output.as_ref(),
                                memory_tool.as_ref(),
                                approval.as_ref(),
                                );
                        }
                        let _ = out_tx.send(response(
                            $id,
                            "switch_session",
                            true,
                            Some(json!({
                                "session_id": persistence.session_id(),
                                "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
                                "model": current_model,
                                // Task 3 (pi-parity fix, serve pass 19): matches `switch_branch`/`fork`/
                                // `clone`'s own response shape — previously omitted here, forcing a
                                // client to make a separate `get_state` round trip after every
                                // `switch_session` its 3 sibling commands don't need.
                                "reasoning_effort": current_level.as_str(),
                            })),
                            None,
                        ));
                    }
                    Err(e) => {
                        let _ = out_tx.send(response(
                            $id,
                            "switch_session",
                            false,
                            None,
                            Some(&e.to_string()),
                        ));
                    }
                },
                None => {
                    let _ = out_tx.send(response(
                        $id,
                        "switch_session",
                        false,
                        None,
                        Some("missing `session_id`"),
                    ));
                }
            }
        }};
    }
    macro_rules! do_fork {
        ($cmd:expr, $id:expr) => {{
            // `upto` messages to copy into the new session; absent = clone the whole session.
            // `target_id`, when given, forks at that specific tree entry instead — anywhere in the
            // whole tree, not just a message-count prefix of the active path (`before` excludes the
            // entry itself); wins over `upto` if both are present.
            let upto = $cmd
                .get("upto")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(usize::MAX);
            let target_id = $cmd.get("target_id").and_then(Value::as_str);
            // Defaults to `true` (excluding the target entry itself), matching pi's real production
            // client convention.
            let before = $cmd.get("before").and_then(Value::as_bool).unwrap_or(true);
            // Resolved *before* `persistence.fork` below swaps `self.store` onto the freshly created
            // session — `target_id` can name an entry that isn't even on the copied prefix (an
            // off-active-path fork point, or one `before` excludes), so it wouldn't survive into the
            // new tree to look up afterward.
            let fork_text = match target_id {
                Some(tid) => persistence
                    .store
                    .as_ref()
                    .and_then(|s| s.message_at(tid))
                    .and_then(message_text),
                None => upto
                    .min(session.messages.len())
                    .checked_sub(1)
                    .and_then(|i| session.messages.get(i))
                    .and_then(message_text),
            };
            match persistence.fork(upto, target_id, before, starting_level) {
                Ok((s, restored_model, restored_level)) => {
                    session = s;
                    steering.clear();
                    // Restore whichever model/thinking-level was actually active at the forked-from
                    // point, the same way `switch_session`/`switch_branch` already do.
                    let restored_level = agent_core::clamp_thinking_level(
                        &agent_core::capabilities(&restored_model),
                        restored_level,
                    );
                    let mut rebuild_needed = false;
                    if restored_model != current_model {
                        session.scrub_cross_model_state(&restored_model);
                        match build_gateway_client(&cfg, &restored_model) {
                            Ok(new_client) => client = Arc::new(new_client),
                            Err(e) => eprintln!(
                                "serve: could not resolve a gateway credential for restored model {restored_model:?}: \
                                 {e} — keeping the previous client"
                            ),
                        }
                        current_model = restored_model;
                        rebuild_needed = true;
                    }
                    if restored_level != current_level {
                        current_level = restored_level;
                        current_thinking = None;
                        rebuild_needed = true;
                    }
                    if rebuild_needed {
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_level,
                            current_auto_compaction,
                            current_auto_retry,
                            current_block_images,
                            current_image_auto_resize,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                            subagent_ctx.as_ref(),
                                structured_output.as_ref(),
                                memory_tool.as_ref(),
                                approval.as_ref(),
                            );
                    }
                    let _ = out_tx.send(response(
                        $id,
                        "fork",
                        true,
                        Some(json!({
                            "session_id": persistence.session_id(),
                            "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
                            "model": current_model,
                            "reasoning_effort": current_level.as_str(),
                            "text": fork_text,
                        })),
                        None,
                    ));
                }
                Err(e) => {
                    let _ = out_tx.send(response($id, "fork", false, None, Some(&e.to_string())));
                }
            }
        }};
    }
    macro_rules! do_clone {
        ($cmd:expr, $id:expr) => {{
            // pi's own `clone` — fork the current session at its current tip, with no arguments.
            match persistence.fork(usize::MAX, None, false, starting_level) {
                Ok((s, restored_model, restored_level)) => {
                    session = s;
                    steering.clear();
                    let restored_level = agent_core::clamp_thinking_level(
                        &agent_core::capabilities(&restored_model),
                        restored_level,
                    );
                    let mut rebuild_needed = false;
                    if restored_model != current_model {
                        session.scrub_cross_model_state(&restored_model);
                        match build_gateway_client(&cfg, &restored_model) {
                            Ok(new_client) => client = Arc::new(new_client),
                            Err(e) => eprintln!(
                                "serve: could not resolve a gateway credential for restored model {restored_model:?}: \
                                 {e} — keeping the previous client"
                            ),
                        }
                        current_model = restored_model;
                        rebuild_needed = true;
                    }
                    if restored_level != current_level {
                        current_level = restored_level;
                        current_thinking = None;
                        rebuild_needed = true;
                    }
                    if rebuild_needed {
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_level,
                            current_auto_compaction,
                            current_auto_retry,
                            current_block_images,
                            current_image_auto_resize,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                            subagent_ctx.as_ref(),
                                structured_output.as_ref(),
                                memory_tool.as_ref(),
                                approval.as_ref(),
                            );
                    }
                    let _ = out_tx.send(response(
                        $id,
                        "clone",
                        true,
                        Some(json!({
                            "session_id": persistence.session_id(),
                            "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
                            "model": current_model,
                            "reasoning_effort": current_level.as_str(),
                        })),
                        None,
                    ));
                }
                Err(e) => {
                    let _ = out_tx.send(response($id, "clone", false, None, Some(&e.to_string())));
                }
            }
        }};
    }
    macro_rules! do_compact {
        ($cmd:expr, $id:expr) => {{
            // Manual compaction (no run in flight here). Streams a `compacted` event if it cuts.
            // `custom_instructions`, when given, steers what the summary emphasizes.
            let custom_instructions = $cmd.get("custom_instructions").and_then(Value::as_str);
            let tx = out_tx.clone();
            let mut compacted_tokens_before: Option<u32> = None;
            let mut compacted_summary: Option<String> = None;
            let mut compacted_tokens_after: Option<u32> = None;
            let mut compacted_first_kept: Option<usize> = None;
            let result = agent
                .compact(
                    &mut session,
                    agent_core::CompactionReason::Manual,
                    &CancellationToken::new(),
                    &mut |ev| {
                        if let AgentEvent::Compacted {
                            tokens_before,
                            summary,
                            tokens_after,
                            first_kept,
                            ..
                        } = &ev
                        {
                            compacted_tokens_before = Some(*tokens_before);
                            compacted_summary = Some(summary.clone());
                            compacted_tokens_after = Some(*tokens_after);
                            compacted_first_kept = Some(*first_kept);
                        }
                        if let Some(frame) = event_frame(ev) {
                            let _ = tx.send(frame);
                        }
                    },
                    custom_instructions,
                )
                .await;
            match result {
                Ok(outcome) => {
                    // The tree entry that begins the retained (post-compaction) portion of history —
                    // resolved *before* `persist_blocking` below rewrites `persistence`'s own store.
                    let first_kept_entry_id = compacted_first_kept.and_then(|first_kept| {
                        persistence
                            .store
                            .as_ref()
                            .and_then(|s| s.active_ids().get(first_kept).cloned())
                    });
                    while checkpoint_rx.try_recv().is_ok() {}
                    let (p, persist_result) =
                        persist_blocking(persistence, session.clone(), compacted_tokens_before)
                            .await;
                    persistence = p;
                    match persist_result {
                        Ok(()) => {
                            let _ = out_tx.send(response(
                                $id,
                                "compact",
                                true,
                                Some(json!({
                                    "compacted": outcome.compacted(),
                                    "reason": outcome.reason(),
                                    "summary": compacted_summary,
                                    "tokens_before": compacted_tokens_before,
                                    "tokens_after": compacted_tokens_after,
                                    "first_kept_entry_id": first_kept_entry_id,
                                })),
                                None,
                            ));
                        }
                        Err(e) => {
                            let _ = out_tx.send(response(
                                $id,
                                "compact",
                                false,
                                None,
                                Some(&format!("compacted but failed to persist: {e}")),
                            ));
                        }
                    }
                }
                Err(e) => {
                    let _ = out_tx.send(response($id, "compact", false, None, Some(&e.to_string())));
                }
            }
        }};
    }

    timing.print();

    // Announce readiness so a client can sync before issuing commands. If this already fails the
    // writer never started; there is nothing to serve.
    if out_tx
        .send(
            json!({
                "type": "ready",
                "session_id": persistence.session_id(),
                "model": current_model,
                "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
            })
            .into(),
        )
        .is_err()
    {
        let _ = writer.await;
        return Ok(None);
    }

    let mut shutdown = ShutdownSignal::new()?;
    // Task #41 (pi-parity fix): which signal (if any) actually triggered shutdown, so the caller can
    // exit with the matching POSIX code (`Signal::exit_code`) instead of always `0` — every graceful
    // path previously returned bare `Ok(())` with no way to tell a clean stdin-EOF apart from a real
    // SIGTERM/SIGHUP/Ctrl-C. Set from whichever of this function's several `shutdown.wait()` sites
    // actually fires (there's no concurrent access to guard against — this is all one async task's own
    // sequential control flow, just interleaved via `select!`, so a bare local suffices).
    let mut shutdown_cause: Option<Signal> = None;
    // At most one `login` in flight at a time, tracked here so a concurrent second `login` is
    // rejected and `submit_code`/`abort_login` know what to reach. Shared with the detached task
    // `login` spawns (see that arm below) — cleared back to `None` by that task itself once the
    // flow resolves, success or failure, so a later `login` is accepted again.
    let pending_login: Arc<std::sync::Mutex<Option<PendingLogin>>> =
        Arc::new(std::sync::Mutex::new(None));
    loop {
        let line = tokio::select! {
            biased;
            // Idle between commands: nothing is in flight, so a shutdown request needs no drain —
            // just stop reading and fall out to the writer join below.
            sig = shutdown.wait() => {
                shutdown_cause = Some(sig);
                break;
            }
            // A closed `input_rx` (the stdio reader hit EOF and dropped its sender, or a WebSocket
            // supervisor deliberately dropped this session's retained sender to tear it down) ends the
            // session gracefully — the same clean-shutdown path stdin EOF always drove. A merely
            // *detached* WebSocket connection does NOT close `input_rx` (the supervisor keeps the
            // sender), so this arm simply pends across a reconnect rather than firing.
            maybe_line = input_rx.recv() => match maybe_line {
                Some(l) => l,
                None => break,
            },
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cmd: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                emit!(response(
                    None,
                    "parse",
                    false,
                    None,
                    Some(&format!("invalid JSON: {e}")),
                ));
                continue;
            }
        };
        let id = cmd.get("id").and_then(Value::as_str).map(str::to_string);
        let ctype = cmd
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        match ctype.as_str() {
            "prompt" => {
                // `output_schema` makes this prompt a callable function: the model must fill the schema
                // in via `structured_output` rather than answering in prose, and the validated payload
                // rides the terminal response's `data.structured_output`. Handled before the ack, so a
                // malformed schema is rejected outright rather than acknowledged and then failed.
                //
                // Installing (or removing, or changing) a schema changes both the *tool set* and the
                // system-prompt section that tells the model to use it, so both are rebuilt — through
                // the very same `build_agent` path `set_model` already uses. A prompt repeating the
                // schema it already had rebuilds nothing.
                match parse_output_spec(&cmd) {
                    Err(e) => {
                        emit!(response(id, "prompt", false, None, Some(&e)));
                        continue;
                    }
                    Ok(spec) if spec != current_output_spec => {
                        structured_output = match &spec {
                            Some((schema, description)) => {
                                match crate::tools::structured_output::StructuredOutput::new(
                                    schema.clone(),
                                    description.clone(),
                                    output_slot.clone(),
                                ) {
                                    Ok(tool) => Some(Arc::new(tool)),
                                    Err(e) => {
                                        emit!(response(id, "prompt", false, None, Some(&e)));
                                        continue;
                                    }
                                }
                            }
                            None => None,
                        };
                        current_output_spec = spec;
                        memory_index = current_memory_index(&memory_backend).await;
                        static_system = crate::resources::build_static_system_prompt(
                            &crate::resources::PromptOptions {
                                base: None,
                                default_base: &cfg.system,
                                append: cfg.append_system.as_deref(),
                                cwd: &cwd,
                                include_context_files: cfg.context_files,
                                skills: &skills,
                                has_read,
                                has_todo,
                                has_structured_output: structured_output.is_some(),
                                has_memory,
                                memory_index: memory_index.as_deref(),
                                project_trusted,
                                agents: &cfg.agents,
                            },
                        );
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_level,
                            current_auto_compaction,
                            current_auto_retry,
                            current_block_images,
                            current_image_auto_resize,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                            subagent_ctx.as_ref(),
                            structured_output.as_ref(),
                            memory_tool.as_ref(),
                            approval.as_ref(),
                        );
                    }
                    Ok(_) => {}
                }
                // A previous prompt's answer must never be mistaken for this one's.
                output_slot.clear();
                // Refresh the cheap, time-varying part of the system prompt (the date) every turn —
                // the expensive discovery-based static half only changes on `set_model`/`set_thinking`/
                // `reload`, so it's cached rather than recomputed here.
                agent.set_system(full_system(&static_system, &cwd));
                // A `prompt` with no (or non-string) `message` is a malformed command: running an
                // empty user turn would spend a model call on nothing and still report success.
                let Some(message) = cmd.get("message").and_then(Value::as_str) else {
                    emit!(response(
                        id,
                        "prompt",
                        false,
                        None,
                        Some("missing `message`"),
                    ));
                    continue;
                };
                let message = expand_message(message, &skills, &prompt_templates);
                // Optional image attachments: `images: [{media_type, data}]` (base64). Builds a
                // multimodal user turn; absent or empty → a plain text turn.
                let images = parse_images(cmd.get("images"));
                if images.is_empty() {
                    session.user(message);
                } else {
                    session.push(agent_core::Message::user_with_images(message, images));
                }
                // Acknowledge immediately — the turn is queued and about to start — rather than
                // leaving a client with no signal until the (possibly much later) terminal response.
                emit!(ack(id.clone(), "prompt"));
                let cancel = CancellationToken::new();
                // The sink sets this to the compaction's `tokens_before` when the loop compacts
                // mid-run, so we know to non-destructively rewrite (not append) the persisted
                // transcript afterwards. 0 doubles as "no compaction fired this run" — see
                // `Persistence::persist`; in practice a real compaction's `tokens_before` is never
                // legitimately 0 (`should_compact`/`compact` only ever fire once real usage has been
                // recorded). Reset (not recreated) at the top of each retry attempt below, so the value
                // read after the loop always reflects only the most recent attempt.
                let tokens_before = Arc::new(AtomicU32::new(0));
                // Whether the run's *last* turn ended in a refusal — set from every `TurnEnd`, so by
                // the time the run returns it reflects only the final one (any refusal earlier in a
                // multi-turn run is superseded once the model goes on to call tools or answer plainly).
                // Also reset per retry attempt, for the same reason as `tokens_before`.
                let refused = Arc::new(std::sync::atomic::AtomicBool::new(false));
                // Whether an auto-triggered compaction is *currently* in flight this instant — pi's
                // `isCompacting`, surfaced from `get_state`. Set on `AgentEvent::CompactionStart`,
                // cleared on `Compacted`/`CompactionFailed` *or* the very next event of any other kind
                // (`TurnStart` for the run's next turn, ordinarily) — `Agent::compact` can end without
                // ever emitting `Compacted` (an empty-summary no-op, or a failure reported via
                // `CompactionFailed` rather than propagated — see that event's doc comment), and its own
                // model call uses a discarding inner sink, so nothing else can arrive between
                // `CompactionStart` and whatever naturally follows it; clearing on "anything else" is
                // exact, not a guess.
                // Manual `compact`/branch-summary-on-`switch_branch` don't touch this: both only ever
                // run from the idle main loop, which processes one command to completion before reading
                // the next — no concurrent `get_state` could ever observe them in flight anyway.
                let is_compacting = Arc::new(std::sync::atomic::AtomicBool::new(false));

                // Drive the run while staying responsive to stdin: `abort` cancels it, `steer` queues a
                // mid-run injection and `follow_up` a stop-boundary one; any other command is
                // rejected as busy (the session is borrowed by the in-flight run). If stdin closes
                // mid-run, cancel and drain. The block scopes the run's `&mut session` borrow so we can
                // persist after.
                let mut stdin_open = true;
                // A run that ends in a transient-looking `Err` (`crate::retry::is_retryable_whole_run`
                // — a superset of `run_turn`'s own mid-stream retry's `is_retryable_mid_stream`, plus
                // raw HTTP status digits appropriate only at this outer layer) is automatically
                // re-invoked against the *same* session — resuming from wherever it left off, not
                // restarting the turn — up to `retry::MAX_RUN_RETRIES` times with backoff. pi's
                // `agent-session.ts` equivalent (`_prepareRetry`/`isRetryableAssistantError`); ours is
                // gated by the same `current_auto_retry` toggle as the mid-stream layer below it (one
                // user-facing "retries on/off" concept, even though internally it's two layers), and
                // never retries past a client disconnect or an explicit `abort`
                // (`is_retryable_whole_run` already excludes `Error::Cancelled` transitively through
                // `is_retryable_mid_stream`, and `stdin_open`/`cancel.is_cancelled()` below are
                // redundant, cheap belt-and-suspenders on top of that). The backoff sleep itself *is*
                // raced against `abort`/`abort_retry`/shutdown (see the `select!` around the `sleep`
                // below) — a client abort during backoff takes effect immediately, not just once the
                // next attempt starts.
                let mut retry_attempt: u32 = 0;
                // Whether the final post-attempt persist (below) failed to write to disk — surfaced in
                // the terminal `prompt` response instead of being silently swallowed, since an agent run
                // that appears to succeed while its transcript failed to persist is exactly the failure
                // mode this exists to catch. Always set from *that* persist's own outcome alone, once
                // per attempt: it re-persists the whole current session state fresh, so it's a strict
                // superset of any mid-run checkpoint along the way (see the `checkpoint_rx.recv()` arm's
                // comment below) — a checkpoint that failed but was then followed by a successful final
                // persist must report success, not a stale failure the superseding write already fixed.
                let mut persist_error: Option<String>;
                // pi-parity (Task 4, serve pass 19): `compact`/`switch_session`/`fork`/`clone`/
                // `new_session` received while this run (or its auto-retry backoff wait) is in flight
                // self-abort-and-proceed instead of being rejected as busy — cancelled via the exact
                // same `cancel.cancel()`/deferred-until-idle discipline `abort` itself already uses
                // (see `pending_abort_acks` just below), then queued here (raw command + its own `id`)
                // to run their ordinary idle-path logic once this attempt has actually gone idle, in
                // the order received. Declared outside the loop (not inside, like `pending_abort_acks`)
                // and `clear()`-reset every attempt, not re-declared: a deferred command always forces
                // this attempt to end via cancellation (a cancelled/aborted result is never itself
                // retryable — see the match on `attempt_result` below), so it's always populated and
                // drained within the very same iteration, but the drain happens *after* this loop
                // returns, once `session`/`persistence` are no longer borrowed by `run` — a variable
                // declared inside the loop body would be dropped at `break` and unreachable there.
                let mut pending_deferred: Vec<(Option<String>, Value)> = Vec::new();
                let result = 'retry: loop {
                    tokens_before.store(0, Ordering::Relaxed);
                    refused.store(false, Ordering::Relaxed);
                    is_compacting.store(false, Ordering::Relaxed);
                    pending_deferred.clear();
                    // `abort` command ids received while this run is still unwinding — acked only once
                    // the run has actually gone idle (see the flush right after `attempt_result` below,
                    // and that ack's own doc comment for why it's deferred rather than sent the instant
                    // `cancel.cancel()` is called).
                    let mut pending_abort_acks: Vec<Option<String>> = Vec::new();
                    let tx = out_tx.clone();
                    let tokens_before_sink = tokens_before.clone();
                    let refused_sink = refused.clone();
                    let is_compacting_sink = is_compacting.clone();
                    // Live token/step counters a `get_state`/`get_session_stats` sent while this run is in
                    // flight can answer from (see the busy-loop's own arms for those types below) — seeded
                    // fresh from the session's *current* totals every attempt (a retry may follow a partial,
                    // already-persisted success from an earlier attempt in this same loop), then kept current
                    // from the same events a streaming client already observes.
                    let live_stats = Arc::new(LiveStats::from_session(&session));
                    let live_stats_sink = live_stats.clone();

                    // Mark this session busy for the reaper's benefit (daemon mode) — it never reaps a
                    // session with a run in flight. Cleared once the attempt's future resolves, below.
                    running.store(true, Ordering::Relaxed);
                    let attempt_result = {
                        let run = agent.run_events_steered(
                            &mut session,
                            move |ev| {
                                // Set on `CompactionStart`, cleared on literally anything else — see
                                // `is_compacting`'s own declaration above for why that's exact, not a
                                // conservative approximation.
                                is_compacting_sink.store(
                                    matches!(ev, AgentEvent::CompactionStart { .. }),
                                    Ordering::Relaxed,
                                );
                                if let AgentEvent::Compacted { tokens_before, .. } = ev {
                                    tokens_before_sink.store(tokens_before, Ordering::Relaxed);
                                }
                                if let AgentEvent::TurnEnd { stop_reason, step } = &ev {
                                    refused_sink.store(
                                        *stop_reason == StopReason::Refusal,
                                        Ordering::Relaxed,
                                    );
                                    live_stats_sink.set_steps(*step);
                                }
                                if let AgentEvent::Stream(StreamEvent::Usage(usage)) = &ev {
                                    live_stats_sink.record_usage(usage);
                                }
                                // B-L1: mirrors the same events into `pending_tool_ids`, so `get_state`
                                // can answer "which calls are still running" mid-turn — see that
                                // field's own doc comment.
                                if let AgentEvent::ToolStart { id, .. } = &ev {
                                    live_stats_sink.tool_started(id.clone());
                                }
                                if let AgentEvent::ToolEnd { id, .. } = &ev {
                                    live_stats_sink.tool_ended(id);
                                }
                                // Same mirroring, for the `todo` list — see `LiveStats::todos`. The tool
                                // validates before it emits, so a rejected call never lands here and the
                                // mirror only ever holds a list the model actually committed.
                                if let AgentEvent::ToolProgress {
                                    name,
                                    details: Some(d),
                                    ..
                                } = &ev
                                    && name == crate::tools::todo::NAME
                                    && let Some(todos) = d.get("todos")
                                {
                                    live_stats_sink.todos_updated(todos.clone());
                                }
                                // Best-effort: a sync sink can't break the control loop. If the writer is
                                // gone the send fails here and the terminal response send below detects it
                                // via `emit!` and stops the loop. An unserializable event is skipped rather
                                // than emitted as a malformed frame (see `event_frame`).
                                if let Some(frame) = event_frame(ev) {
                                    let _ = tx.send(frame);
                                }
                            },
                            cancel.clone(),
                            steering.clone(),
                        );
                        tokio::pin!(run);
                        loop {
                            tokio::select! {
                                biased;
                                r = &mut run => break r,
                                // A shutdown request mid-run gets the same treatment as stdin closing:
                                // cancel and let the run unwind so the code below can persist before we
                                // fall out of the outer loop (`!stdin_open` breaks it, further down).
                                sig = shutdown.wait() => {
                                    shutdown_cause = Some(sig);
                                    stdin_open = false;
                                    cancel.cancel();
                                }
                                // Drain and persist each mid-run checkpoint as it arrives (see
                                // `ChannelCheckpoint`). `persistence` isn't touched by `run` itself (only
                                // `session` is borrowed there), so reassigning it here is safe exactly like
                                // `stdin_open`/`cancel` being mutated from a sibling branch above. Any
                                // checkpoint still left undrained once the run ends is explicitly swept up
                                // right after this loop, below, before it can leak into some later, unrelated
                                // turn — see that drain's own comment for why "harmless, the final persist
                                // is a superset" isn't quite true on its own.
                                Some(messages) = checkpoint_rx.recv() => {
                                    let (p, r) = persist_messages_blocking(persistence, messages).await;
                                    persistence = p;
                                    // Logged, not tracked in `persist_error`: the unconditional final
                                    // persist below re-persists the whole session fresh regardless, so
                                    // this checkpoint's own outcome never affects what's reported to the
                                    // client — only worth knowing about here for operational visibility.
                                    if let Err(e) = r {
                                        tracing::warn!(error = %e, "mid-run checkpoint failed to persist");
                                    }
                                }
                                maybe_line = input_rx.recv(), if stdin_open => match maybe_line {
                                    Some(l) => {
                                        let l = l.trim();
                                        if l.is_empty() {
                                            continue;
                                        }
                                        let c: Value = match serde_json::from_str(l) {
                                            Ok(v) => v,
                                            Err(e) => {
                                                let _ = out_tx.send(response(None, "parse", false, None, Some(&format!("invalid JSON: {e}"))));
                                                continue;
                                            }
                                        };
                                        let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                                        match c.get("type").and_then(Value::as_str).unwrap_or("") {
                                            // Fix 4 (pi-parity gap): the ack is *not* sent here — see
                                            // the flush right after this busy-loop exits, below. Sending
                                            // it immediately (the instant `cancel.cancel()` is called)
                                            // let a client that — correctly, per this RPC's own contract
                                            // — treats the ack as "safe to send the next command" get
                                            // rejected as busy: this same busy-loop keeps rejecting every
                                            // non-abort command as busy until `run` itself actually
                                            // resolves, which can lag well behind the cancellation
                                            // request while in-flight tool calls/HTTP streams unwind.
                                            // Matches pi's `agent-session.ts`, which awaits
                                            // `waitForIdle()` before acking `abort`.
                                            "abort" => {
                                                cancel.cancel();
                                                pending_abort_acks.push(cid);
                                            }
                                            "stop_after_turn" => {
                                                // Graceful, not a cancel: the current turn's tool calls (if
                                                // any) still finish and their results are committed; the run
                                                // just doesn't start another model call afterward. See
                                                // `agent_core::Steering::request_stop`.
                                                steering.request_stop();
                                                let _ = out_tx.send(response(cid, "stop_after_turn", true, None, None));
                                            }
                                            "switch_model" => {
                                                // Pi-parity gap (pi's `prepareNextTurn`): unlike `set_model`
                                                // (which only takes effect on the *next* `prompt`), this
                                                // retargets the run already in flight — the current turn's
                                                // own request is unaffected, but every subsequent turn of
                                                // this same run targets the new model. See
                                                // `agent_core::Steering::request_model_switch`.
                                                match c.get("model").and_then(Value::as_str) {
                                                    Some(model) => {
                                                        let thinking = c
                                                            .get("thinking")
                                                            .and_then(Value::as_u64)
                                                            .map(|t| t as u32);
                                                        steering.request_model_switch(model, thinking);
                                                        let _ = out_tx.send(response(cid, "switch_model", true, None, None));
                                                    }
                                                    None => {
                                                        let _ = out_tx.send(response(cid, "switch_model", false, None, Some("missing `model`")));
                                                    }
                                                }
                                            }
                                            cmd @ ("steer" | "follow_up") => {
                                                match c.get("message").and_then(Value::as_str) {
                                                    Some(m) => {
                                                        // Same expansion (and the same optional
                                                        // `images` attachments) a fresh `prompt` gets —
                                                        // a `/skill:name`/`/name` invocation, or an
                                                        // image, steered or queued this way must not
                                                        // reach the model unexpanded/dropped just
                                                        // because it arrived on a different command
                                                        // type.
                                                        let m = expand_message(m, &skills, &prompt_templates);
                                                        let m = agent_core::SteeringMessage::new(
                                                            m,
                                                            parse_images(c.get("images")),
                                                        );
                                                        // `steer` redirects mid-run (injected at the next
                                                        // tool turn); `follow_up` waits for the stop
                                                        // boundary. Two separate lanes.
                                                        if cmd == "steer" {
                                                            steering.push_steer(m);
                                                        } else {
                                                            steering.push(m);
                                                        }
                                                        // Fix 5 (pi-parity gap): the pushing client
                                                        // learns what's actually queued from its own
                                                        // ack, same round trip — see `queue_content`'s
                                                        // own doc comment for why this doesn't also
                                                        // need a separate unsolicited event.
                                                        let _ = out_tx.send(response(cid, cmd, true, Some(queue_content(&steering)), None));
                                                    }
                                                    None => {
                                                        let _ = out_tx.send(response(cid, cmd, false, None, Some("missing `message`")));
                                                    }
                                                }
                                            }
                                            // A `prompt` sent while busy is rejected by default (the
                                            // session is borrowed by the in-flight run) — *unless* it
                                            // carries `streaming_behavior: "steer"|"follow_up"`, in which
                                            // case its `message` is routed through the same `Steering`
                                            // queue as an explicit `steer`/`follow_up` command, rather than
                                            // forcing the client to re-encode it as a different command type.
                                            "prompt" => {
                                                match (
                                                    c.get("streaming_behavior").and_then(Value::as_str),
                                                    c.get("message").and_then(Value::as_str),
                                                ) {
                                                    (Some("steer"), Some(m)) => {
                                                        let m = expand_message(m, &skills, &prompt_templates);
                                                        let m = agent_core::SteeringMessage::new(
                                                            m,
                                                            parse_images(c.get("images")),
                                                        );
                                                        steering.push_steer(m);
                                                        // Fix 5 (pi-parity gap): same queue-content
                                                        // visibility the dedicated `steer`/`follow_up`
                                                        // commands' own acks now carry.
                                                        let mut data = queue_content(&steering);
                                                        data["queued_as"] = json!("steer");
                                                        let _ = out_tx.send(response(cid, "prompt", true, Some(data), None));
                                                    }
                                                    (Some("follow_up"), Some(m)) => {
                                                        let m = expand_message(m, &skills, &prompt_templates);
                                                        let m = agent_core::SteeringMessage::new(
                                                            m,
                                                            parse_images(c.get("images")),
                                                        );
                                                        steering.push(m);
                                                        let mut data = queue_content(&steering);
                                                        data["queued_as"] = json!("follow_up");
                                                        let _ = out_tx.send(response(cid, "prompt", true, Some(data), None));
                                                    }
                                                    _ => {
                                                        let _ = out_tx.send(response(cid, "prompt", false, None, Some("busy: a prompt is running; only `abort`/`steer`/`follow_up`, or a `prompt` with `streaming_behavior: \"steer\"|\"follow_up\"`, are accepted")));
                                                    }
                                                }
                                            }
                                            // Read-only commands that don't need the `&mut Session` this
                                            // run holds — answered live instead of rejected as busy, so a
                                            // client polling for progress (tokens/steps so far) during a
                                            // long tool-heavy turn isn't left with nothing until it ends.
                                            // `get_state`/`get_session_stats` read `live_stats` (mirrored
                                            // from this run's own events, see above) instead of `session`
                                            // directly; `message_count` is `null` here specifically — it's
                                            // the one `get_state` field that genuinely needs `&session` to
                                            // report exactly, and a stale/guessed number would be worse
                                            // than an honest "not available mid-run".
                                            "get_state" => {
                                                let mut data = live_stats.snapshot();
                                                if let Value::Object(m) = &mut data {
                                                    insert_session_identity(m, &persistence);
                                                    m.insert("model".into(), json!(current_model));
                                                    m.insert("message_count".into(), Value::Null);
                                                    m.insert("title".into(), json!(persistence.meta.title));
                                                    m.insert("cwd_stale".into(), json!(cwd_is_stale(&persistence.meta.cwd, &cwd)));
                                                    // Task #25 (pi-parity fix): same fields, same reasoning, as the idle `get_state` arm below.
                                                    m.insert("cwd".into(), json!(cwd.display().to_string()));
                                                    m.insert("git_branch".into(), json!(git_branch(&cwd).await));
                                                    m.insert("is_streaming".into(), json!(true));
                                                    m.insert("is_compacting".into(), json!(is_compacting.load(Ordering::Relaxed)));
                                                    if let Value::Object(rt) = runtime_settings(current_level, current_auto_compaction, current_auto_retry, current_block_images, current_image_auto_resize, &steering) {
                                                        m.extend(rt);
                                                    }
                                                }
                                                let _ = out_tx.send(response(cid, "get_state", true, Some(data), None));
                                            }
                                            "get_session_stats" => {
                                                // Fix 8 (pi-parity gap): previously the raw
                                                // `LiveStats::snapshot()` verbatim — dropping
                                                // session_id/session_file (this process's own live
                                                // sources, same as `get_state`'s busy arm just above)
                                                // and user_messages/assistant_messages/tool_calls/
                                                // tool_results/total_messages/context_usage (genuinely
                                                // unavailable mid-run: `&session` is exclusively
                                                // borrowed by the in-flight turn, the same reason
                                                // `get_state`'s own `message_count` is `null` here) —
                                                // present as `null` rather than silently absent, so a
                                                // client sees the same field *set* idle or busy, values
                                                // aside.
                                                let mut data = live_stats.snapshot();
                                                if let Value::Object(m) = &mut data {
                                                    insert_session_identity(m, &persistence);
                                                    for field in [
                                                        "context_usage",
                                                        "user_messages",
                                                        "assistant_messages",
                                                        "tool_calls",
                                                        "tool_results",
                                                        "total_messages",
                                                    ] {
                                                        m.insert(field.to_string(), Value::Null);
                                                    }
                                                }
                                                let _ = out_tx.send(response(cid, "get_session_stats", true, Some(data), None));
                                            }
                                            "get_commands" => {
                                                let mut commands: Vec<Value> = skills.iter().map(|s| {
                                                    json!({ "name": format!("skill:{}", s.name), "source": "skill", "description": s.description, "scope": s.scope, "path": s.path })
                                                }).collect();
                                                commands.extend(prompt_templates.iter().map(|t| {
                                                    json!({ "name": t.name, "source": "prompt", "description": t.description, "scope": t.scope, "path": t.path })
                                                }));
                                                let collisions: Vec<&crate::skills::Collision> = skill_collisions.iter().chain(prompt_collisions.iter()).collect();
                                                let _ = out_tx.send(response(cid, "get_commands", true, Some(json!({ "commands": commands, "collisions": collisions })), None));
                                            }
                                            "list_branches" => {
                                                let _ = out_tx.send(response(cid, "list_branches", true, Some(json!({ "branches": persistence.list_branches() })), None));
                                            }
                                            "get_todos" => {
                                                // From the live mirror: `&session` is exclusively borrowed by
                                                // the in-flight turn. See `LiveStats::todos`.
                                                let _ = out_tx.send(response(cid, "get_todos", true, Some(json!({ "todos": live_stats.todos() })), None));
                                            }
                                            // The arm that matters: an approval question is only ever
                                            // outstanding *while a run is blocked inside the gate*, so this
                                            // must be answerable mid-run. `accepted:false` tells the losing
                                            // client in a multi-attach race that its answer arrived too late.
                                            "approve" => {
                                                let _ = out_tx.send(handle_approve(cid, &c, pending_approvals.as_ref()));
                                            }
                                            "get_tree" => {
                                                // Same `since` handling as the idle-loop arm below — see
                                                // `nodes_since`'s own doc comment (Task #48, pi-parity gap).
                                                match c.get("since").and_then(Value::as_str) {
                                                    Some(since) => match nodes_since(persistence.tree(), since) {
                                                        Ok(nodes) => {
                                                            let _ = out_tx.send(response(cid, "get_tree", true, Some(json!({ "nodes": nodes, "leaf_id": persistence.active_ids().last() })), None));
                                                        }
                                                        Err(e) => {
                                                            let _ = out_tx.send(response(cid, "get_tree", false, None, Some(&e)));
                                                        }
                                                    },
                                                    None => {
                                                        let _ = out_tx.send(response(cid, "get_tree", true, Some(json!({ "nodes": persistence.tree(), "leaf_id": persistence.active_ids().last() })), None));
                                                    }
                                                }
                                            }
                                            "list_sessions" => {
                                                let progress_id = cid.clone();
                                                let progress_tx = out_tx.clone();
                                                let query = c.get("query").and_then(Value::as_str);
                                                let sessions = persistence
                                                    .list_with_progress(move |scanned, total| {
                                                        if should_report_scan_progress(scanned, total) {
                                                            let _ = progress_tx.send(list_progress_frame(progress_id.clone(), "list_sessions", scanned, total));
                                                        }
                                                    })
                                                    .await;
                                                let sessions: Vec<Value> = search_sessions(sessions, query)
                                                    .iter()
                                                    .map(SessionMeta::to_listing_json)
                                                    .collect();
                                                let _ = out_tx.send(response(cid, "list_sessions", true, Some(json!({ "sessions": sessions })), None));
                                            }
                                            "list_all_sessions" => {
                                                let progress_id = cid.clone();
                                                let progress_tx = out_tx.clone();
                                                let query = c.get("query").and_then(Value::as_str);
                                                match persistence.list_all_with_progress(move |scanned, total| {
                                                    if should_report_scan_progress(scanned, total) {
                                                        let _ = progress_tx.send(list_progress_frame(progress_id.clone(), "list_all_sessions", scanned, total));
                                                    }
                                                }).await {
                                                    Ok(sessions) => {
                                                        let sessions: Vec<Value> = search_sessions(sessions, query).iter().map(SessionMeta::to_listing_json).collect();
                                                        let _ = out_tx.send(response(cid, "list_all_sessions", true, Some(json!({ "sessions": sessions })), None));
                                                    }
                                                    Err(e) => {
                                                        let _ = out_tx.send(response(cid, "list_all_sessions", false, None, Some(&e.to_string())));
                                                    }
                                                }
                                            }
                                            "get_available_models" => {
                                                // Always the full known-model hint list, never scoped by `--models` — see the
                                                // idle-mode handler's identical comment for why.
                                                let models: Vec<Value> = available_models().iter().map(|m| model_info(m)).collect();
                                                let _ = out_tx.send(response(cid, "get_available_models", true, Some(json!({ "models": models })), None));
                                            }
                                            // A run is actively in flight — no retry backoff is pending yet
                                            // (that only happens after an attempt fails, outside this loop),
                                            // so there's nothing to cancel. Acknowledge idempotently, same
                                            // as `abort`'s own idle-mode no-op below.
                                            "abort_retry" => {
                                                let _ = out_tx.send(response(cid, "abort_retry", true, None, None));
                                            }
                                            // pi-parity (Task 4): pi's real product has no busy/idle
                                            // gating at all for these — `compact()`/`switchSession`/
                                            // `newSession`/`fork` all call `abort()` unconditionally
                                            // before replacing the session, with no check for whether a
                                            // prompt is currently streaming. Mirrored here: cancel this
                                            // run exactly like an explicit `abort` (same token), then
                                            // queue the raw command to run its ordinary idle-path logic
                                            // once the run has actually gone idle — see
                                            // `pending_deferred`'s own doc comment above. No ack is sent
                                            // here; the command's own idle-path response (sent once
                                            // idle, below) doubles as it, same as `abort`'s own deferred
                                            // ack just above.
                                            "compact" | "switch_session" | "fork" | "clone"
                                            | "new_session" => {
                                                cancel.cancel();
                                                pending_deferred.push((cid, c.clone()));
                                            }
                                            other => {
                                                let _ = out_tx.send(response(cid, other, false, None, Some("busy: a prompt is running; only `abort`/`abort_retry`/`steer`/`follow_up`, `compact`/`switch_session`/`fork`/`clone`/`new_session` (which self-abort-and-proceed), or a handful of read-only commands (get_state/get_session_stats/get_commands/list_branches/get_tree/list_sessions/list_all_sessions/get_available_models), are accepted")));
                                            }
                                        }
                                    }
                                    // stdin closed (or errored) mid-run: cancel and let the run unwind, then
                                    // we'll fall out of the outer loop below.
                                    None => {
                                        stdin_open = false;
                                        cancel.cancel();
                                    }
                                }
                            }
                        }
                    };
                    // The run's future has resolved (idle again) — clear the busy flag so the reaper may
                    // reclaim this session once it's also been detached long enough. A subsequent retry
                    // attempt re-sets it at the top of the next loop iteration.
                    running.store(false, Ordering::Relaxed);

                    // Fix 4: the run has now actually gone idle — `run_events_steered`'s future only
                    // resolves once the whole run (including tool cleanup) has stopped touching
                    // `session` — so it's safe to tell a client-issued `abort` it can now send its next
                    // command without racing this busy-loop's own "still draining" rejection above.
                    for cid in pending_abort_acks.drain(..) {
                        let _ = out_tx.send(response(cid, "abort", true, None, None));
                    }

                    // A checkpoint sent right as the run above ended (see `agent_core::Agent`'s final,
                    // unconditional post-run `checkpoint.checkpoint(session)`) races the `r = &mut run`
                    // arm above under `biased` select: whichever is ready first wins, so a checkpoint
                    // that arrived in the same instant the run itself completed can be left sitting in
                    // `checkpoint_rx`'s buffer, undrained. Track (pi-parity fix, found via this crate's
                    // own test suite going red): left alone, that stale message would only ever get
                    // drained during some *later*, unrelated prompt's own inner loop — by which point an
                    // intervening `new_session`/`switch_session`/`fork`/`switch_branch` may have already
                    // swapped `persistence` onto a *different* session's store, so draining it there
                    // would silently write *this* turn's content into that other session's file (and,
                    // via `append_new`'s `messages.len() <= self.persisted` dedup guard, could mask that
                    // later turn's own real content entirely). Discarded here, not persisted: the
                    // unconditional `persist_blocking` right below is a strict superset of it for this
                    // turn's own final state, and this must run *before* any later command gets a chance
                    // to swap `persistence` out from under it.
                    while checkpoint_rx.try_recv().is_ok() {}

                    // Persist whatever this attempt produced (even a failed one may have committed a tool
                    // round-trip or two before the error) before deciding whether to retry.
                    let compacted_tokens_before = match tokens_before.load(Ordering::Relaxed) {
                        0 => None,
                        n => Some(n),
                    };
                    {
                        let (p, r) =
                            persist_blocking(persistence, session.clone(), compacted_tokens_before)
                                .await;
                        persistence = p;
                        persist_error = r.err().map(|e| e.to_string());
                    }

                    match &attempt_result {
                        Err(e)
                            if current_auto_retry
                                && stdin_open
                                && !cancel.is_cancelled()
                                && retry_attempt < run_retry_policy.max_retries
                                && crate::retry::is_retryable_whole_run(e) =>
                        {
                            retry_attempt += 1;
                            let delay = run_retry_policy.backoff(retry_attempt);
                            let _ = out_tx.send(auto_retry_frame(
                                id.clone(),
                                retry_attempt,
                                run_retry_policy.max_retries,
                                delay.as_millis() as u64,
                                &e.to_string(),
                            ));
                            // Race the backoff itself against `abort`/`abort_retry`/shutdown, instead
                            // of a bare `sleep` no command can interrupt — no run is in flight during
                            // this wait, so only those two commands (plus stdin closing) are accepted;
                            // anything else is rejected as busy, the same shape the live-run busy-loop
                            // uses. A cancelled retry surfaces the *real* underlying error that
                            // triggered it (`attempt_result`, set below), not a synthetic
                            // `Error::Cancelled` — nothing was actually cancelled mid-flight, the
                            // automatic retry was just declined.
                            let mut retry_cancelled = false;
                            let sleep = tokio::time::sleep(delay);
                            tokio::pin!(sleep);
                            loop {
                                tokio::select! {
                                    biased;
                                    () = &mut sleep => break,
                                    sig = shutdown.wait() => {
                                        shutdown_cause = Some(sig);
                                        stdin_open = false;
                                        retry_cancelled = true;
                                        break;
                                    }
                                    maybe_line = input_rx.recv(), if stdin_open => match maybe_line {
                                        Some(l) => {
                                            let l = l.trim();
                                            if l.is_empty() {
                                                continue;
                                            }
                                            let c: Value = match serde_json::from_str(l) {
                                                Ok(v) => v,
                                                Err(e) => {
                                                    let _ = out_tx.send(response(None, "parse", false, None, Some(&format!("invalid JSON: {e}"))));
                                                    continue;
                                                }
                                            };
                                            let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                                            let cmd_type = c.get("type").and_then(Value::as_str).unwrap_or("");
                                            match cmd_type {
                                                "abort" | "abort_retry" => {
                                                    retry_cancelled = true;
                                                    let _ = out_tx.send(response(cid, cmd_type, true, None, None));
                                                    break;
                                                }
                                                // pi-parity (Task 4): same self-abort-and-proceed
                                                // treatment as the live-run busy-loop's own arm above —
                                                // this backoff wait is still "a prompt is running" from
                                                // the client's point of view. Ends the retry sequence
                                                // (like `abort_retry`) and queues the command to run
                                                // once idle, via the same `pending_deferred` drained
                                                // right after this whole retry loop.
                                                "compact" | "switch_session" | "fork" | "clone"
                                                | "new_session" => {
                                                    retry_cancelled = true;
                                                    pending_deferred.push((cid, c.clone()));
                                                    break;
                                                }
                                                other => {
                                                    let _ = out_tx.send(response(cid, other, false, None, Some("busy: retrying after a transient error; only `abort`/`abort_retry`/`compact`/`switch_session`/`fork`/`clone`/`new_session` are accepted")));
                                                }
                                            }
                                        }
                                        None => {
                                            stdin_open = false;
                                            retry_cancelled = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            if retry_cancelled {
                                let _ = out_tx.send(auto_retry_end_frame(
                                    id.clone(),
                                    false,
                                    retry_attempt,
                                    Some("retry cancelled"),
                                ));
                                break 'retry attempt_result;
                            }
                            // The failed attempt's closing error record (`run_events_steered` always
                            // persists one on an `Err`-ending run — see `Message::error`'s doc comment)
                            // must not survive into the retry: this is the *same* run resuming from
                            // scratch, not a client's own fresh prompt, so it needs to be genuinely
                            // invisible in the transcript, not stacked under the retry's real response.
                            session.pop_error_record();
                            continue 'retry;
                        }
                        _ => {
                            // A terminal notice only for a sequence that actually retried at least
                            // once — a first-attempt success or an immediately-non-retryable failure
                            // never emitted `auto_retry`, so it needs no "the retries ended" notice
                            // either. Mirrors pi's own `this._retryAttempt > 0` guard.
                            if retry_attempt > 0 {
                                let (success, final_error) = match &attempt_result {
                                    Ok(_) => (true, None),
                                    Err(e) => (false, Some(e.to_string())),
                                };
                                let _ = out_tx.send(auto_retry_end_frame(
                                    id.clone(),
                                    success,
                                    retry_attempt,
                                    final_error.as_deref(),
                                ));
                            }
                            break 'retry attempt_result;
                        }
                    }
                };

                let frame = match result {
                    Ok(()) => {
                        let mut data = session_stats(&session, &current_model);
                        if let Value::Object(m) = &mut data {
                            m.insert("refused".into(), json!(refused.load(Ordering::Relaxed)));
                            // Present exactly when this prompt asked for typed output, so a client can
                            // tell "the model never produced a payload" (`null`) apart from "this run
                            // wasn't asked for one" (field absent) — the same reason `refused` is always
                            // present rather than only when true. Read after the run has fully drained,
                            // so a mixed batch (`structured_output` + `edit` in one turn, which the
                            // loop's unanimous-terminate rule lets continue) reports the final value.
                            if current_output_spec.is_some() {
                                m.insert(
                                    "structured_output".into(),
                                    output_slot.get().unwrap_or(Value::Null),
                                );
                            }
                        }
                        match &persist_error {
                            // The run itself succeeded, but its transcript failed to durably persist —
                            // report failure rather than a false success (still with `data`: the run
                            // did produce a real result, just not one safely on disk yet).
                            Some(e) => response(
                                id.clone(),
                                "prompt",
                                false,
                                Some(data),
                                Some(&format!("run completed but failed to persist: {e}")),
                            ),
                            None => response(id.clone(), "prompt", true, Some(data), None),
                        }
                    }
                    Err(e) => response(id.clone(), "prompt", false, None, Some(&e.to_string())),
                };
                emit!(frame);
                // pi-parity (Task 4): this run has now actually gone idle (same guarantee
                // `pending_abort_acks` relies on above) and its own terminal response has just been
                // sent — run each command that arrived mid-run and self-aborted-and-proceeded through
                // its ordinary idle-path logic now, in the order received. Exactly what a client
                // manually sending `abort`, waiting for the response, then retrying the command would
                // have produced, minus the extra round trip.
                for (dcid, dcmd) in pending_deferred.drain(..) {
                    match dcmd.get("type").and_then(Value::as_str).unwrap_or("") {
                        "new_session" => do_new_session!(dcmd, dcid),
                        "switch_session" => do_switch_session!(dcmd, dcid),
                        "fork" => do_fork!(dcmd, dcid),
                        "clone" => do_clone!(dcmd, dcid),
                        "compact" => do_compact!(dcmd, dcid),
                        _ => {}
                    }
                }
                if !stdin_open {
                    break;
                }
            }
            // Reachable while idle (a mid-run `steer`/`follow_up` is handled inside the `prompt` arm's
            // own busy-loop above, which shares this same persistent `steering` handle) — queues for
            // whichever `prompt` runs next, rather than being rejected as an unknown command just
            // because nothing is in flight *yet*.
            cmd_type @ ("steer" | "follow_up") => {
                match cmd.get("message").and_then(Value::as_str) {
                    Some(m) => {
                        let m = expand_message(m, &skills, &prompt_templates);
                        let m =
                            agent_core::SteeringMessage::new(m, parse_images(cmd.get("images")));
                        if cmd_type == "steer" {
                            steering.push_steer(m);
                        } else {
                            steering.push(m);
                        }
                        // Fix 5 (pi-parity gap): see `queue_content`'s own doc comment.
                        emit!(response(
                            id,
                            cmd_type,
                            true,
                            Some(queue_content(&steering)),
                            None
                        ));
                    }
                    None => emit!(response(
                        id,
                        cmd_type,
                        false,
                        None,
                        Some("missing `message`")
                    )),
                }
            }
            "abort" => {
                // No run is in flight (a mid-run abort is handled inside the `prompt` arm above), so
                // there is nothing to cancel — acknowledge idempotently.
                emit!(response(id, "abort", true, None, None));
            }
            "abort_retry" => {
                // Same idea as `abort` above: interrupting a pending whole-run-retry backoff only
                // means something while a `prompt` is between attempts (handled inside that arm's own
                // retry loop) — idle, there's no backoff in progress to cancel.
                emit!(response(id, "abort_retry", true, None, None));
            }
            "stop_after_turn" => {
                // No run is in flight, so there is no turn boundary to stop at. Unlike `steer`/
                // `follow_up`, deliberately *not* forwarded to `steering.request_stop()` here: that
                // would silently cut the *next* `prompt` off after its first turn, surprising a client
                // that sent this expecting to affect a run that no longer exists. Acknowledge as a
                // no-op instead, matching `abort`'s idle behavior.
                emit!(response(id, "stop_after_turn", true, None, None));
            }
            "switch_model" => {
                // No run is in flight, so there is no next turn to retarget — changing the model while
                // idle is `set_model`'s job (it takes effect immediately, on the very next `prompt`).
                // Acknowledge as a no-op rather than silently queuing a switch that would surprise a
                // later, unrelated run, matching `stop_after_turn`'s idle behavior.
                emit!(response(
                    id,
                    "switch_model",
                    true,
                    Some(json!({ "note": "no run in flight; use set_model instead" })),
                    None
                ));
            }
            "get_state" => {
                let mut data = session_stats(&session, &current_model);
                if let Value::Object(m) = &mut data {
                    insert_session_identity(m, &persistence);
                    m.insert("model".into(), json!(current_model));
                    m.insert("message_count".into(), json!(session.messages.len()));
                    m.insert("title".into(), json!(persistence.meta.title));
                    m.insert(
                        "cwd_stale".into(),
                        json!(cwd_is_stale(&persistence.meta.cwd, &cwd)),
                    );
                    // Task #25 (pi-parity fix): the directory (and, best-effort, branch) the agent's
                    // tools are actually operating against — the live process `cwd`, not
                    // `persistence.meta.cwd` (already separately surfaced via `cwd_stale` when the two
                    // disagree), since that's what a remote client with no shared filesystem needs to
                    // know. See `git_branch`'s own doc comment for why a lookup failure is `null`, not
                    // an error.
                    m.insert("cwd".into(), json!(cwd.display().to_string()));
                    m.insert("git_branch".into(), json!(git_branch(&cwd).await));
                    // Both hardcoded, not stale placeholders: no `prompt`/compaction can possibly be
                    // in flight here at all — this arm only ever runs from the idle main loop, which
                    // processes one command to completion before reading the next, so there is no
                    // concurrently-running turn or compaction this response could be racing.
                    m.insert("is_streaming".into(), json!(false));
                    m.insert("is_compacting".into(), json!(false));
                    // Same reasoning as `is_streaming`/`is_compacting` above: idle means nothing is
                    // dispatching a tool call either.
                    m.insert("pending_tool_ids".into(), json!(Vec::<String>::new()));
                    if let Value::Object(rt) = runtime_settings(
                        current_level,
                        current_auto_compaction,
                        current_auto_retry,
                        current_block_images,
                        current_image_auto_resize,
                        &steering,
                    ) {
                        m.extend(rt);
                    }
                }
                emit!(response(id, "get_state", true, Some(data), None));
            }
            "get_messages" => {
                // Tag each message with its tree id when persistence tracks one (parallel to
                // `session.messages` — see `Persistence::active_ids`), so a client can pick a
                // `target_id` for `switch_branch` from anywhere in the visible transcript, not only
                // from a branch's leaf (which is all `list_branches` alone can offer). A length
                // mismatch (in-memory mode, or a transient inconsistency) leaves messages untagged
                // rather than mistagging them.
                let msg_ids = persistence.active_ids();
                let mut messages =
                    serde_json::to_value(session.messages.as_ref()).unwrap_or(Value::Null);
                if let Value::Array(arr) = &mut messages {
                    if arr.len() == msg_ids.len() {
                        for (m, mid) in arr.iter_mut().zip(msg_ids) {
                            if let Value::Object(obj) = m {
                                obj.insert("id".into(), json!(mid));
                            }
                        }
                    }
                }
                // `since`: return only what's new since a tree id the client already has — pi's own
                // `get_entries({since})` — so a client polling for updates doesn't have to re-transfer
                // the whole transcript every time. Unmatched, or given while nothing is tagged (a
                // length mismatch, or in-memory-only mode with no ids at all), is an error: silently
                // falling back to "everything" would look like a working incremental fetch that's
                // actually just re-sending the full history, masking the bug instead of surfacing it.
                if let Some(since) = cmd.get("since").and_then(Value::as_str) {
                    match msg_ids.iter().position(|mid| mid == since) {
                        Some(idx) => {
                            if let Value::Array(arr) = &mut messages {
                                *arr = arr.split_off(idx + 1);
                            }
                        }
                        None => {
                            emit!(response(
                                id,
                                "get_messages",
                                false,
                                None,
                                Some(&format!("no message with id {since} in this session"))
                            ));
                            continue;
                        }
                    }
                }
                // `leaf_id`: the same active-tip id `get_tree`'s response already carries (Fix 6,
                // pi-parity gap — pi's own `get_entries({since})` returns `{entries, leafId}` in one
                // round trip). Without this, a client fetching the transcript still had to issue a
                // second `get_tree` call just to learn the current tip to pass as a future
                // `switch_branch` target.
                emit!(response(
                    id,
                    "get_messages",
                    true,
                    Some(json!({ "messages": messages, "leaf_id": msg_ids.last() })),
                    None,
                ));
            }
            "new_session" => do_new_session!(cmd, id),
            "list_sessions" => {
                let progress_id = id.clone();
                let progress_tx = out_tx.clone();
                let query = cmd.get("query").and_then(Value::as_str);
                let sessions = persistence
                    .list_with_progress(move |scanned, total| {
                        if should_report_scan_progress(scanned, total) {
                            let _ = progress_tx.send(list_progress_frame(
                                progress_id.clone(),
                                "list_sessions",
                                scanned,
                                total,
                            ));
                        }
                    })
                    .await;
                let sessions: Vec<Value> = search_sessions(sessions, query)
                    .iter()
                    .map(SessionMeta::to_listing_json)
                    .collect();
                emit!(response(
                    id,
                    "list_sessions",
                    true,
                    Some(json!({ "sessions": sessions })),
                    None,
                ));
            }
            "list_all_sessions" => {
                let progress_id = id.clone();
                let progress_tx = out_tx.clone();
                let query = cmd.get("query").and_then(Value::as_str);
                match persistence
                    .list_all_with_progress(move |scanned, total| {
                        if should_report_scan_progress(scanned, total) {
                            let _ = progress_tx.send(list_progress_frame(
                                progress_id.clone(),
                                "list_all_sessions",
                                scanned,
                                total,
                            ));
                        }
                    })
                    .await
                {
                    Ok(sessions) => {
                        let sessions: Vec<Value> = search_sessions(sessions, query)
                            .iter()
                            .map(SessionMeta::to_listing_json)
                            .collect();
                        emit!(response(
                            id,
                            "list_all_sessions",
                            true,
                            Some(json!({ "sessions": sessions })),
                            None,
                        ));
                    }
                    Err(e) => emit!(response(
                        id,
                        "list_all_sessions",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                }
            }
            "switch_session" => do_switch_session!(cmd, id),
            // Soft-deletes (moves to `.trash`) another session — never the one currently active, see
            // `Persistence::delete`'s doc comment. Idempotent: deleting an absent (or already-deleted)
            // session id is a successful no-op.
            "delete_session" => match cmd.get("session_id").and_then(Value::as_str) {
                Some(target) => match persistence.delete(target) {
                    Ok(()) => emit!(response(id, "delete_session", true, None, None)),
                    Err(e) => emit!(response(
                        id,
                        "delete_session",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                },
                None => emit!(response(
                    id,
                    "delete_session",
                    false,
                    None,
                    Some("missing `session_id`")
                )),
            },
            // Minimal surface (Fix 7, pi-parity gap) over `delete_session`'s own `.trash/` soft-delete —
            // nothing previously read, listed, restored from, or pruned it, so a mistaken delete was
            // recoverable only by reaching for a shell. `list_trash` reports basic metadata
            // (id/deleted_at/original_path — see `session_store::TrashEntry`); `restore_session` moves an
            // entry back out. Deliberately not a full trash-management UI (no bulk purge, no age-based
            // pruning) — a low-priority nice-to-have, not a core session-lifecycle feature.
            "list_trash" => match persistence.list_trash() {
                Ok(trash) => emit!(response(
                    id,
                    "list_trash",
                    true,
                    Some(json!({ "trash": trash })),
                    None,
                )),
                Err(e) => emit!(response(
                    id,
                    "list_trash",
                    false,
                    None,
                    Some(&e.to_string())
                )),
            },
            "restore_session" => match cmd.get("session_id").and_then(Value::as_str) {
                Some(target) => match persistence.restore_session(target) {
                    Ok(true) => emit!(response(id, "restore_session", true, None, None)),
                    Ok(false) => emit!(response(
                        id,
                        "restore_session",
                        false,
                        None,
                        Some(&format!("no trashed session {target}"))
                    )),
                    Err(e) => emit!(response(
                        id,
                        "restore_session",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                },
                None => emit!(response(
                    id,
                    "restore_session",
                    false,
                    None,
                    Some("missing `session_id`")
                )),
            },
            "fork" => do_fork!(cmd, id),
            // pi's own `clone` — fork the current session at its current tip, with no arguments —
            // exists there because pi's `fork` *requires* an explicit `entryId`; this crate's `fork`
            // already defaults to exactly that (no `upto`/`target_id` given), so `clone` is a thin,
            // deliberately-argument-free alias over the same call for a client speaking pi's protocol
            // shape, not a second code path.
            "clone" => do_clone!(cmd, id),
            "get_fork_messages" => {
                // pi-compatible contract: no parameters, every user-turn entry across the WHOLE session
                // tree — every branch, not just the active path — as a flat `{entry_id, text}` candidate
                // list (pi's own `getUserMessagesForForking`), for a client to build a fork-point picker
                // from and then feed one `entry_id` to `fork`'s own `target_id`/`fork_at_entry`. This is
                // a listing, not a preview of any one fork's output — see `preview_fork` for that.
                // Track (pi-parity fix): previously scoped to `persistence.active_ids()` only, so a
                // message on a branch the session had already navigated away from (via `switch_branch`)
                // never appeared as a fork candidate at all — empty (not an error) in pure in-memory mode,
                // same as every other persistence-dependent listing here.
                let candidates: Vec<Value> = persistence
                    .all_user_messages()
                    .iter()
                    .filter_map(|(entry_id, m)| {
                        user_message_text(m)
                            .map(|text| json!({ "entry_id": entry_id, "text": text }))
                    })
                    .collect();
                emit!(response(
                    id,
                    "get_fork_messages",
                    true,
                    Some(json!({ "messages": candidates })),
                    None,
                ));
            }
            "preview_fork" => {
                // A read-only preview of what `fork` would produce for `session_id` (default: the
                // current session) at `upto` messages — no new session file, no switch. Lets a client
                // browsing `list_sessions` show a fork point before committing to it. This crate's own
                // extension beyond pi's protocol (previously misnamed `get_fork_messages`, colliding
                // with pi's own same-named but differently-shaped command — see the module doc comment).
                let target_id = cmd
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| persistence.session_id().to_string());
                let upto = cmd
                    .get("upto")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .unwrap_or(usize::MAX);
                let entry_id = cmd.get("target_id").and_then(Value::as_str);
                // Same default as `fork` itself, above — a preview must match what `fork` would
                // actually produce.
                let before = cmd.get("before").and_then(Value::as_bool).unwrap_or(true);
                match persistence.fork_messages(&target_id, upto, entry_id, before) {
                    Ok(messages) => emit!(response(
                        id,
                        "preview_fork",
                        true,
                        Some(json!({ "messages": messages })),
                        None,
                    )),
                    Err(e) => emit!(response(
                        id,
                        "preview_fork",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                }
            }
            "export_html" => {
                let output_path = cmd.get("output_path").and_then(Value::as_str);
                let branches = persistence.abandoned_branches();
                let events = persistence.export_events();
                // `export_html_full` (Task #44 integration): this session's actual system prompt (the
                // same static-plus-dynamic-footer join every `prompt` sends — see `full_system`) and
                // tool set (`build_tools(&cfg)`, the same on-demand rebuild `has_read`/`bash_tool`
                // already call elsewhere in this function rather than keeping a long-lived registry
                // variable around), plus `session`'s own running token totals — previously omitted
                // entirely via the plainer `export_html_with_entries`.
                let system_prompt = full_system(&static_system, &cwd);
                let tool_defs = build_tools(&cfg, cfg.image_auto_resize).definitions();
                let usage = crate::export::UsageTotals {
                    input_tokens: session.input_tokens,
                    output_tokens: session.output_tokens,
                    cache_read_tokens: session.cache_read_tokens,
                    cache_write_tokens: session.cache_write_tokens,
                };
                match crate::export::export_html_full(
                    &persistence.meta,
                    &session.messages,
                    &branches,
                    Some(usage),
                    events,
                    Some(&system_prompt),
                    Some(&tool_defs),
                    output_path,
                ) {
                    Ok(path) => emit!(response(
                        id,
                        "export_html",
                        true,
                        Some(json!({ "path": path.to_string_lossy() })),
                        None,
                    )),
                    Err(e) => emit!(response(
                        id,
                        "export_html",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                }
            }
            "set_session_name" => match cmd.get("title").and_then(Value::as_str) {
                Some(title) => match persistence.set_title(title) {
                    Ok(()) => {
                        // pi: `session_info_changed` (`rpc-mode.ts:632-639`) — a push notification so a
                        // client learns the final *sanitized* name (newlines collapsed, or cleared
                        // entirely if the sanitized result is empty — see `sanitize_title`) without a
                        // second `get_state` round trip. Sent as its own unsolicited frame (like
                        // `list_progress`/`auto_retry` above) *and* echoed on the response's own `data`,
                        // since unlike pi's multi-subscriber extension model this protocol has exactly
                        // one client and the response is already the more direct notification path.
                        let title = persistence.meta.title.clone();
                        let _ = out_tx.send(session_info_changed_frame(id.clone(), title.clone()));
                        emit!(response(
                            id,
                            "set_session_name",
                            true,
                            Some(json!({ "title": title })),
                            None,
                        ));
                    }
                    Err(e) => emit!(response(
                        id,
                        "set_session_name",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                },
                None => emit!(response(
                    id,
                    "set_session_name",
                    false,
                    None,
                    Some("missing `title`")
                )),
            },
            "set_label" => match cmd.get("target_id").and_then(Value::as_str) {
                Some(target_id) => match cmd.get("label") {
                    // Present-but-null is the explicit "clear it" signal; a missing key is an error
                    // (so a typo can't silently no-op) — matches `set_thinking`'s own `budget` contract.
                    Some(Value::Null) => match persistence.set_label(target_id, None) {
                        Ok(()) => emit!(response(
                            id,
                            "set_label",
                            true,
                            Some(json!({ "target_id": target_id, "label": Value::Null })),
                            None,
                        )),
                        Err(e) => {
                            emit!(response(id, "set_label", false, None, Some(&e.to_string())))
                        }
                    },
                    Some(Value::String(label)) => {
                        match persistence.set_label(target_id, Some(label)) {
                            Ok(()) => emit!(response(
                                id,
                                "set_label",
                                true,
                                Some(json!({ "target_id": target_id, "label": label })),
                                None,
                            )),
                            Err(e) => {
                                emit!(response(id, "set_label", false, None, Some(&e.to_string())))
                            }
                        }
                    }
                    Some(_) => emit!(response(
                        id,
                        "set_label",
                        false,
                        None,
                        Some("`label` must be a string or null")
                    )),
                    None => emit!(response(
                        id,
                        "set_label",
                        false,
                        None,
                        Some("missing `label` — pass a string to set it or null to clear it")
                    )),
                },
                None => emit!(response(
                    id,
                    "set_label",
                    false,
                    None,
                    Some("missing `target_id`")
                )),
            },
            "get_label" => match cmd.get("target_id").and_then(Value::as_str) {
                Some(target_id) => match persistence.get_label(target_id) {
                    Ok(label) => emit!(response(
                        id,
                        "get_label",
                        true,
                        Some(json!({ "target_id": target_id, "label": label })),
                        None,
                    )),
                    Err(e) => emit!(response(id, "get_label", false, None, Some(&e.to_string()))),
                },
                None => emit!(response(
                    id,
                    "get_label",
                    false,
                    None,
                    Some("missing `target_id`")
                )),
            },
            "append_custom" => match cmd.get("kind").and_then(Value::as_str) {
                Some(kind) => {
                    let data = cmd.get("data").cloned().unwrap_or_else(|| json!({}));
                    match persistence.append_custom(kind, data) {
                        Ok(new_id) => emit!(response(
                            id,
                            "append_custom",
                            true,
                            Some(json!({ "id": new_id })),
                            None,
                        )),
                        Err(e) => {
                            emit!(response(
                                id,
                                "append_custom",
                                false,
                                None,
                                Some(&e.to_string())
                            ))
                        }
                    }
                }
                None => emit!(response(
                    id,
                    "append_custom",
                    false,
                    None,
                    Some("missing `kind`")
                )),
            },
            "compact" => do_compact!(cmd, id),
            "get_last_assistant_text" => {
                let text = last_assistant_text(&session);
                emit!(response(
                    id,
                    "get_last_assistant_text",
                    true,
                    Some(json!({ "text": text })),
                    None,
                ));
            }
            // Reachable while idle only for a stale/duplicate answer (`accepted:false`): a real question
            // can only be outstanding while a run is in flight, which is the busy arm above.
            "approve" => {
                emit!(handle_approve(id, &cmd, pending_approvals.as_ref()));
            }
            "get_todos" => {
                // Straight from the session while idle — no mirror needed, and no chance of one going
                // stale against a `switch_branch`/`fork`/`compact` that happened since the last turn.
                emit!(response(
                    id,
                    "get_todos",
                    true,
                    Some(json!({ "todos": current_todos(&session) })),
                    None,
                ));
            }
            "get_session_stats" => {
                // Fix 8 (pi-parity gap): backfills session_id/session_file (mirroring get_state's own
                // idle arm, above) and pending_tool_ids (mirroring get_state's own "nothing pending
                // while idle" convention) — previously omitted, unlike the busy-mode arm's
                // `LiveStats::snapshot()`-derived `pending_tool_ids`, so idle and busy reported
                // different field sets for the same command.
                let mut data = session_stats(&session, &current_model);
                if let Value::Object(m) = &mut data {
                    insert_session_identity(m, &persistence);
                    m.insert("pending_tool_ids".into(), json!(Vec::<String>::new()));
                }
                emit!(response(id, "get_session_stats", true, Some(data), None));
            }
            "get_commands" => {
                // Skills (read-on-demand) and prompt templates (`/name`), for client autocomplete.
                // `scope`/`path` (Task #39 pi-parity fix — previously omitted entirely) mirror pi's own
                // `get_commands` `sourceInfo.scope`/`sourceInfo.path`: which discovery root a command
                // actually came from (`"user"`/`"project"`/`"temporary"`) and its on-disk location.
                let mut commands: Vec<Value> = skills
                    .iter()
                    .map(|s| {
                        json!({
                            "name": format!("skill:{}", s.name),
                            "source": "skill",
                            "description": s.description,
                            "scope": s.scope,
                            "path": s.path,
                        })
                    })
                    .collect();
                commands.extend(prompt_templates.iter().map(|t| {
                    json!({
                        "name": t.name,
                        "source": "prompt",
                        "description": t.description,
                        "scope": t.scope,
                        "path": t.path,
                    })
                }));
                // Every shadowed name (a skill or template defined at more than one path) is otherwise
                // silently resolved with no way for a client to notice — surfaced here instead.
                let collisions: Vec<&crate::skills::Collision> = skill_collisions
                    .iter()
                    .chain(prompt_collisions.iter())
                    .collect();
                emit!(response(
                    id,
                    "get_commands",
                    true,
                    Some(json!({ "commands": commands, "collisions": collisions })),
                    None,
                ));
            }
            "reload" => {
                // Re-run the full (expensive) discovery pi's `/reload` triggers: trust may have
                // changed (`agent trust`/`--trust-project` since startup), and project instructions,
                // `SYSTEM.md`, and skills/prompt templates may have changed on disk. The per-turn
                // `full_system` refresh alone only ever picks up the cheap date/cwd footer.
                let has_gated_resources = crate::trust_store::has_trust_gated_resources(&cwd);
                project_trusted = resolve_project_trust(
                    cfg.trust_project,
                    cfg.force_untrusted,
                    cfg.default_project_trust,
                    crate::trust_store::TrustStore::open_default().lookup(&cwd),
                    has_gated_resources,
                );
                // Track L32 (pi-parity fix): same warning as startup, above — trust may have just
                // changed to untrusted (or gated resources may have just appeared on disk) as of this
                // very `reload`, and an operator watching stderr deserves the same signal they'd have
                // gotten from a fresh `serve` invocation instead of silence.
                if !project_trusted && has_gated_resources {
                    eprintln!(
                        "warning: {} has a project-local SYSTEM.md/APPEND_SYSTEM.md, skills, prompt \
                         templates, or a settings.json on disk, but the project isn't trusted, so they \
                         were skipped — pass --trust-project or run `agent trust {}` to enable them (a \
                         project's own settings.json additionally requires a *persisted* `agent trust`, \
                         not just a one-off --trust-project)",
                        cwd.display(),
                        cwd.display()
                    );
                }
                // `--no-skills`/`--no-prompt-templates` still honor an explicit `--skill`/
                // `--prompt-template` extra path (pi-parity fix, M2) — see the identical reasoning at
                // this function's startup discovery, above.
                (prompt_templates, prompt_collisions) = if cfg.no_prompt_templates {
                    crate::prompts::discover_extra_only(&cfg.extra_prompt_template_paths)
                } else {
                    crate::prompts::discover_with_diagnostics(
                        &cwd,
                        project_trusted,
                        &cfg.extra_prompt_template_paths,
                    )
                };
                (skills, skill_collisions) = if cfg.no_skills {
                    crate::skills::discover_extra_only(&cfg.extra_skill_paths)
                } else {
                    crate::skills::discover_with_diagnostics(
                        &cwd,
                        project_trusted,
                        &cfg.extra_skill_paths,
                    )
                };
                // Agent definitions are trust-gated like skills, so a `reload` after a trust change (or
                // an edit to `.claude/agents/`) must re-discover them and rebuild the subagent context —
                // otherwise `<available_agents>` and the `subagent` tool would advertise a stale set until
                // restart. The `set_model` arm below then rebuilds the agent with the refreshed ctx.
                cfg.agents = crate::agents::discover(&cwd, project_trusted);
                subagent_ctx = if cfg.agents.is_empty() {
                    None
                } else {
                    Some(build_subagent_ctx(
                        &cfg,
                        &cwd,
                        project_trusted,
                        &current_model,
                        &write_locks,
                        &skills,
                        memory_backend.clone(),
                        approval.as_ref(),
                    ))
                };
                memory_index = current_memory_index(&memory_backend).await;
                static_system = crate::resources::build_static_system_prompt(
                    &crate::resources::PromptOptions {
                        base: None,
                        default_base: &cfg.system,
                        append: cfg.append_system.as_deref(),
                        cwd: &cwd,
                        include_context_files: cfg.context_files,
                        skills: &skills,
                        has_read,
                        has_todo,
                        has_structured_output: structured_output.is_some(),
                        has_memory,
                        memory_index: memory_index.as_deref(),
                        project_trusted,
                        agents: &cfg.agents,
                    },
                );
                // A full rebuild, not just `agent.set_system(...)`: `reload` may have changed the agent
                // *definitions* (a new `.claude/agents/*.md`, or trust newly granted), and the `subagent`
                // tool's registration lives in the registry, which only a rebuild refreshes. Without this,
                // a mid-session-added agent wouldn't become delegable until the next `set_model`. Mirrors
                // `set_model`'s own rebuild; the session is untouched.
                agent = build_agent(
                    client.clone(),
                    &full_system(&static_system, &cwd),
                    &cfg,
                    &current_model,
                    current_thinking,
                    current_level,
                    current_auto_compaction,
                    current_auto_retry,
                    current_block_images,
                    current_image_auto_resize,
                    persistence.session_id(),
                    &write_locks,
                    &checkpoint,
                    subagent_ctx.as_ref(),
                    structured_output.as_ref(),
                    memory_tool.as_ref(),
                    approval.as_ref(),
                );
                emit!(response(id, "reload", true, None, None));
            }
            // Rejects an empty/whitespace-only id, and — Fix 10 (pi-parity feature) — resolves a
            // partial/fuzzy id against the known-model hint list first (`resolve_model_id`, mirroring
            // `--model`'s identical resolution in `main.rs`): a genuinely unrecognized id (no partial
            // match at all) is still forwarded verbatim, unlike pi (which talks directly to each
            // provider and can validate against a live, authoritative registry of what it's actually
            // configured to reach) — this process has no local source of truth to validate a real id
            // against, and `available_models()` is explicitly documented as a non-exhaustive picker
            // hint, not an allowlist (see its own doc comment). An empty string sneaking through
            // `Value::as_str` and getting durably recorded via `record_model_change` is caught here too,
            // the same class of fix `set_session_name` already got (reject empty, don't pretend to
            // validate against a list this process can't actually authoritatively check). `model` may
            // also carry a `:<level>` suffix (Fix 2, pi-parity gap — e.g. `"sonnet:high"`); when present,
            // `resolve_model_id` returns it alongside the resolved id and it wins outright over whatever
            // level was already active.
            "set_model" => match cmd
                .get("model")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|raw| resolve_model_id(raw, available_models()))
            {
                Some(Ok((model, thinking_level))) => {
                    let model = model.as_str();
                    // Re-derive the gateway credential/routing for `model` before touching any other
                    // state — a switch can cross OAuth providers (or `models.json` overrides), and
                    // this must fail closed (leaving `current_model`/the persisted lineage untouched)
                    // rather than silently keep using whichever client happened to be active before.
                    // Skipped when `model` is already the active one, for the same reason
                    // `record_result` below skips a redundant persist: a no-op `set_model <current>`
                    // shouldn't discard `OAuthCredentialSource`'s in-memory token cache for nothing.
                    let new_client = if model == current_model {
                        Ok(None)
                    } else {
                        build_gateway_client(&cfg, model).map(|c| Some(Arc::new(c)))
                    };
                    match new_client {
                        Ok(new_client) => {
                            // Persist the lineage marker *before* applying the switch in memory: if it
                            // fails to write, leave the live model unchanged too, rather than forking
                            // live state away from what's durably recorded (an `Err` here aborts the
                            // whole switch).
                            let record_result = if model != current_model {
                                persistence.record_model_change(model)
                            } else {
                                Ok(())
                            };
                            match record_result {
                                Ok(()) => {
                                    // A signed thinking block is only valid for replay to the model
                                    // that produced it, and a combined OpenAI-Responses tool-call id
                                    // only means anything back to that same model — scrub both from
                                    // any message not already stamped with the model we're switching
                                    // to.
                                    session.scrub_cross_model_state(model);
                                    current_model = model.to_string();
                                    // Fix 2 (pi-parity gap): a `:<level>` suffix on `model` (e.g.
                                    // `sonnet:high`) wins outright over whatever level was already
                                    // active — the operator explicitly asked for this depth on the new
                                    // model. No suffix falls back to re-clamping the *existing* level
                                    // against the *new* model instead, same as before this fix: e.g. a
                                    // session sitting at `Off` on a disable-capable model must not
                                    // silently carry that `Off` over to a model that can't actually
                                    // disable reasoning.
                                    current_level = agent_core::clamp_thinking_level(
                                        &agent_core::capabilities(&current_model),
                                        thinking_level.unwrap_or(current_level),
                                    );
                                    if let Some(new_client) = new_client {
                                        client = new_client;
                                    }
                                    agent = build_agent(
                                        client.clone(),
                                        &full_system(&static_system, &cwd),
                                        &cfg,
                                        &current_model,
                                        current_thinking,
                                        current_level,
                                        current_auto_compaction,
                                        current_auto_retry,
                                        current_block_images,
                                        current_image_auto_resize,
                                        persistence.session_id(),
                                        &write_locks,
                                        &checkpoint,
                                        subagent_ctx.as_ref(),
                                        structured_output.as_ref(),
                                        memory_tool.as_ref(),
                                        approval.as_ref(),
                                    );
                                    emit!(response(
                                        id,
                                        "set_model",
                                        true,
                                        Some(model_switch_response(&current_model, current_level)),
                                        None,
                                    ));
                                }
                                Err(e) => {
                                    emit!(response(
                                        id,
                                        "set_model",
                                        false,
                                        None,
                                        Some(&e.to_string())
                                    ))
                                }
                            }
                        }
                        Err(e) => emit!(response(id, "set_model", false, None, Some(&e))),
                    }
                }
                Some(Err(e)) => emit!(response(id, "set_model", false, None, Some(&e))),
                None => emit!(response(
                    id,
                    "set_model",
                    false,
                    None,
                    Some("missing or empty `model`")
                )),
            },
            "set_thinking" => {
                // `budget` is a positive integer to enable extended thinking, or `null` to disable it.
                // A present-but-null value is the explicit "turn it off" signal; a missing key is an
                // error (so a typo can't silently no-op).
                match cmd.get("budget") {
                    Some(Value::Null) => {
                        current_thinking = None;
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_level,
                            current_auto_compaction,
                            current_auto_retry,
                            current_block_images,
                            current_image_auto_resize,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                            subagent_ctx.as_ref(),
                            structured_output.as_ref(),
                            memory_tool.as_ref(),
                            approval.as_ref(),
                        );
                        emit!(response(
                            id,
                            "set_thinking",
                            true,
                            Some(json!({ "thinking": Value::Null })),
                            None,
                        ));
                    }
                    Some(v) if v.as_u64().is_some() => {
                        current_thinking = v.as_u64().map(|n| n as u32);
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_level,
                            current_auto_compaction,
                            current_auto_retry,
                            current_block_images,
                            current_image_auto_resize,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                            subagent_ctx.as_ref(),
                            structured_output.as_ref(),
                            memory_tool.as_ref(),
                            approval.as_ref(),
                        );
                        emit!(response(
                            id,
                            "set_thinking",
                            true,
                            Some(json!({ "thinking": current_thinking })),
                            None,
                        ));
                    }
                    _ => emit!(response(
                        id,
                        "set_thinking",
                        false,
                        None,
                        Some("`budget` must be a non-negative integer or null"),
                    )),
                }
            }
            "set_reasoning_effort" => {
                // `effort` is one of the portable level names ("off"/"minimal"/"low"/"medium"/"high"/
                // "xhigh"), or `null` as an alias for `"off"`. Unlike `set_thinking`, this sets
                // `current_level` directly (no raw-override concept — a named effort already *is* a
                // ladder rung) and clears any pending `set_thinking` override, for the same reason
                // `cycle_thinking_level` does: the newly-requested level should take visible effect
                // immediately, not be silently masked by a stale raw budget.
                let parsed = match cmd.get("effort") {
                    Some(Value::Null) => Some(agent_core::ThinkingLevel::Off),
                    Some(Value::String(s)) => agent_core::ThinkingLevel::parse(s),
                    _ => None,
                };
                match parsed {
                    Some(level) => {
                        // Clamp against `current_model`'s capabilities before storing or recording:
                        // an explicit `"off"` request is only honored if the model can actually
                        // disable reasoning; otherwise it's bumped to the model's own floor rather
                        // than silently persisted as a level the wire can't represent.
                        let level = agent_core::clamp_thinking_level(
                            &agent_core::capabilities(&current_model),
                            level,
                        );
                        let record_result = if level != current_level {
                            persistence.record_thinking_level_change(level.as_str())
                        } else {
                            Ok(())
                        };
                        match record_result {
                            Ok(()) => {
                                current_level = level;
                                current_thinking = None;
                                agent = build_agent(
                                    client.clone(),
                                    &full_system(&static_system, &cwd),
                                    &cfg,
                                    &current_model,
                                    current_thinking,
                                    current_level,
                                    current_auto_compaction,
                                    current_auto_retry,
                                    current_block_images,
                                    current_image_auto_resize,
                                    persistence.session_id(),
                                    &write_locks,
                                    &checkpoint,
                                    subagent_ctx.as_ref(),
                                    structured_output.as_ref(),
                                    memory_tool.as_ref(),
                                    approval.as_ref(),
                                );
                                let (thinking, reasoning_effort) = agent_core::thinking_for_level(
                                    &agent_core::capabilities(&current_model),
                                    current_level,
                                );
                                emit!(response(
                                    id,
                                    "set_reasoning_effort",
                                    true,
                                    Some(json!({
                                        "level": current_level.as_str(),
                                        "thinking": thinking,
                                        "reasoning_effort": reasoning_effort.map(|e| e.as_str()),
                                    })),
                                    None,
                                ));
                            }
                            Err(e) => emit!(response(
                                id,
                                "set_reasoning_effort",
                                false,
                                None,
                                Some(&e.to_string())
                            )),
                        }
                    }
                    None => emit!(response(
                        id,
                        "set_reasoning_effort",
                        false,
                        None,
                        Some("`effort` must be one of off/minimal/low/medium/high/xhigh, or null"),
                    )),
                }
            }
            "cycle_model" => {
                // Advance through `cycle_models` (the `--models`-scoped list, or the full known-model
                // hint list when unscoped), wrapping — a quick way for a client to step through the
                // candidate list without needing its own copy of it. An id outside the list (a custom
                // `set_model` the client issued directly) wraps to the first entry, same as "not found".
                let next_idx = cycle_models
                    .iter()
                    .position(|m| m == &current_model)
                    .map_or(0, |i| (i + 1) % cycle_models.len());
                let next_model = cycle_models[next_idx].clone();
                // A `pattern:level` pin from `--models` (e.g. `gpt-4o:high`) rides along whenever
                // cycling lands on that model — pi's own `ScopedModel.thinkingLevel`/
                // `_cycleScopedModel`. Landing on an unpinned model just keeps whatever level was
                // already active, same as pi.
                let pinned_level = scoped_models
                    .iter()
                    .find(|m| m.id == next_model)
                    .and_then(|m| m.thinking_level);
                // See `set_model`'s identical re-derivation for why this must happen before anything
                // else, and why it's skipped when `next_model` (wrapping back around) is already the
                // active one.
                let new_client = if next_model == current_model {
                    Ok(None)
                } else {
                    build_gateway_client(&cfg, &next_model).map(|c| Some(Arc::new(c)))
                };
                match new_client {
                    Ok(new_client) => {
                        let record_result = if next_model != current_model {
                            persistence.record_model_change(&next_model)
                        } else {
                            Ok(())
                        };
                        match record_result {
                            Ok(()) => {
                                session.scrub_cross_model_state(&next_model);
                                current_model = next_model;
                                if let Some(level) = pinned_level {
                                    // Same staleness hazard `cycle_thinking_level` already guards
                                    // against: a stale raw-budget override must not silently outlive
                                    // the level it was pinned over.
                                    current_thinking = None;
                                    current_level = level;
                                }
                                // See `set_model`'s identical re-clamp for why this can't be skipped.
                                current_level = agent_core::clamp_thinking_level(
                                    &agent_core::capabilities(&current_model),
                                    current_level,
                                );
                                if let Some(new_client) = new_client {
                                    client = new_client;
                                }
                                agent = build_agent(
                                    client.clone(),
                                    &full_system(&static_system, &cwd),
                                    &cfg,
                                    &current_model,
                                    current_thinking,
                                    current_level,
                                    current_auto_compaction,
                                    current_auto_retry,
                                    current_block_images,
                                    current_image_auto_resize,
                                    persistence.session_id(),
                                    &write_locks,
                                    &checkpoint,
                                    subagent_ctx.as_ref(),
                                    structured_output.as_ref(),
                                    memory_tool.as_ref(),
                                    approval.as_ref(),
                                );
                                let mut resp_data =
                                    model_switch_response(&current_model, current_level);
                                if let Value::Object(map) = &mut resp_data {
                                    map.insert("scoped".to_string(), json!(!cfg.models.is_empty()));
                                }
                                emit!(response(id, "cycle_model", true, Some(resp_data), None));
                            }
                            Err(e) => emit!(response(
                                id,
                                "cycle_model",
                                false,
                                None,
                                Some(&e.to_string())
                            )),
                        }
                    }
                    Err(e) => emit!(response(id, "cycle_model", false, None, Some(&e))),
                }
            }
            "cycle_thinking_level" => {
                // Advance the portable Off/Minimal/Low/Medium/High/XHigh ladder, wrapping, and clear
                // any explicit raw-budget override (`set_thinking`) — otherwise a stale override could
                // silently mask the level having just changed. `thinking_for_level` reports what the
                // new level actually resolves to for `current_model` specifically, so the response
                // reflects reality rather than a number that may be meaningless for this model's shape.
                // Advances through only the levels `current_model` actually supports (see
                // `next_available_thinking_level`'s doc comment) rather than the raw 6-rung ladder —
                // a blind `.next()` would let the cycle land on (and durably record) an `Off` or
                // `xhigh` state the active model can't represent on the wire.
                let next_level = agent_core::next_available_thinking_level(
                    &agent_core::capabilities(&current_model),
                    current_level,
                );
                match persistence.record_thinking_level_change(next_level.as_str()) {
                    Ok(()) => {
                        current_level = next_level;
                        current_thinking = None;
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_level,
                            current_auto_compaction,
                            current_auto_retry,
                            current_block_images,
                            current_image_auto_resize,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                            subagent_ctx.as_ref(),
                            structured_output.as_ref(),
                            memory_tool.as_ref(),
                            approval.as_ref(),
                        );
                        let (thinking, reasoning_effort) = agent_core::thinking_for_level(
                            &agent_core::capabilities(&current_model),
                            current_level,
                        );
                        emit!(response(
                            id,
                            "cycle_thinking_level",
                            true,
                            Some(json!({
                                "level": current_level.as_str(),
                                "thinking": thinking,
                                "reasoning_effort": reasoning_effort.map(|e| e.as_str()),
                            })),
                            None,
                        ));
                    }
                    Err(e) => emit!(response(
                        id,
                        "cycle_thinking_level",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                }
            }
            "set_auto_compaction" => match cmd.get("enabled").and_then(Value::as_bool) {
                Some(enabled) => {
                    current_auto_compaction = enabled;
                    // Track L26 (pi-parity fix): previously mutated only this in-process local, so a
                    // restarted `serve` (with neither `--no-compaction` nor its env var given) silently
                    // reverted to the built-in enabled-by-default behavior — surviving a restart is the
                    // whole point of `agent settings`' persisted-default tier (same as
                    // `default_project_trust`). Best-effort: a failed write still applies for the rest
                    // of *this* process (the in-memory toggle above already took effect), it just won't
                    // survive a restart.
                    if let Err(e) = settings_store.set_compaction_enabled(Some(enabled)) {
                        eprintln!("serve: failed to persist auto-compaction setting: {e}");
                    }
                    agent = build_agent(
                        client.clone(),
                        &full_system(&static_system, &cwd),
                        &cfg,
                        &current_model,
                        current_thinking,
                        current_level,
                        current_auto_compaction,
                        current_auto_retry,
                        current_block_images,
                        current_image_auto_resize,
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
                        subagent_ctx.as_ref(),
                        structured_output.as_ref(),
                        memory_tool.as_ref(),
                        approval.as_ref(),
                    );
                    emit!(response(
                        id,
                        "set_auto_compaction",
                        true,
                        Some(json!({ "auto_compaction": current_auto_compaction })),
                        None,
                    ));
                }
                None => emit!(response(
                    id,
                    "set_auto_compaction",
                    false,
                    None,
                    Some("missing boolean `enabled`")
                )),
            },
            "set_auto_retry" => match cmd.get("enabled").and_then(Value::as_bool) {
                Some(enabled) => {
                    current_auto_retry = enabled;
                    agent = build_agent(
                        client.clone(),
                        &full_system(&static_system, &cwd),
                        &cfg,
                        &current_model,
                        current_thinking,
                        current_level,
                        current_auto_compaction,
                        current_auto_retry,
                        current_block_images,
                        current_image_auto_resize,
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
                        subagent_ctx.as_ref(),
                        structured_output.as_ref(),
                        memory_tool.as_ref(),
                        approval.as_ref(),
                    );
                    emit!(response(
                        id,
                        "set_auto_retry",
                        true,
                        Some(json!({ "auto_retry": current_auto_retry })),
                        None,
                    ));
                }
                None => emit!(response(
                    id,
                    "set_auto_retry",
                    false,
                    None,
                    Some("missing boolean `enabled`")
                )),
            },
            // Pass 20 (pi-parity fix): `build_agent` already threaded `cfg.block_images` into
            // `Agent::with_block_images` (Task #34) and `cfg.image_auto_resize` into `build_tools`'s
            // registry (Task #34), but neither had a live RPC toggle the way `auto_compaction`/
            // `auto_retry` just above do — an operator could only change either by restarting `serve`
            // with a different `--block-images`/`--no-image-auto-resize` flag or persisted `agent
            // settings` default. Mirrors `set_auto_compaction`'s exact shape: mutate the runtime-mutable
            // local, persist best-effort (survives a restart, same as `set_auto_compaction`), rebuild
            // `agent` so the very next turn actually sees it, then ack over RPC.
            "set_block_images" => match cmd.get("enabled").and_then(Value::as_bool) {
                Some(enabled) => {
                    current_block_images = enabled;
                    if let Err(e) = settings_store.set_block_images(Some(enabled)) {
                        eprintln!("serve: failed to persist block-images setting: {e}");
                    }
                    agent = build_agent(
                        client.clone(),
                        &full_system(&static_system, &cwd),
                        &cfg,
                        &current_model,
                        current_thinking,
                        current_level,
                        current_auto_compaction,
                        current_auto_retry,
                        current_block_images,
                        current_image_auto_resize,
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
                        subagent_ctx.as_ref(),
                        structured_output.as_ref(),
                        memory_tool.as_ref(),
                        approval.as_ref(),
                    );
                    emit!(response(
                        id,
                        "set_block_images",
                        true,
                        Some(json!({ "block_images": current_block_images })),
                        None,
                    ));
                }
                None => emit!(response(
                    id,
                    "set_block_images",
                    false,
                    None,
                    Some("missing boolean `enabled`")
                )),
            },
            // Same shape as `set_block_images` just above, for `image_auto_resize`.
            "set_image_auto_resize" => match cmd.get("enabled").and_then(Value::as_bool) {
                Some(enabled) => {
                    current_image_auto_resize = enabled;
                    if let Err(e) = settings_store.set_image_auto_resize(Some(enabled)) {
                        eprintln!("serve: failed to persist image-auto-resize setting: {e}");
                    }
                    agent = build_agent(
                        client.clone(),
                        &full_system(&static_system, &cwd),
                        &cfg,
                        &current_model,
                        current_thinking,
                        current_level,
                        current_auto_compaction,
                        current_auto_retry,
                        current_block_images,
                        current_image_auto_resize,
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
                        subagent_ctx.as_ref(),
                        structured_output.as_ref(),
                        memory_tool.as_ref(),
                        approval.as_ref(),
                    );
                    emit!(response(
                        id,
                        "set_image_auto_resize",
                        true,
                        Some(json!({ "image_auto_resize": current_image_auto_resize })),
                        None,
                    ));
                }
                None => emit!(response(
                    id,
                    "set_image_auto_resize",
                    false,
                    None,
                    Some("missing boolean `enabled`")
                )),
            },
            // Toggle how much of the steer lane a single mid-run drain point consumes (see
            // `agent_core::QueueMode`) — pi's default is `"one_at_a_time"`, one queued message per
            // drain, leaving the rest queued for the next one; `"all"` folds everything queued into a
            // single injection (this crate's original behavior). Independent of `set_follow_up_mode`
            // below — pi carries the same split (`steeringMode`/`followUpMode`). Owned entirely by the
            // `Steering` handle itself (shared with the loop, no `Agent` rebuild needed), so this takes
            // effect immediately, including mid-run.
            "set_steering_mode" => match cmd.get("mode").and_then(Value::as_str) {
                Some(mode @ ("one_at_a_time" | "all")) => {
                    steering.set_steering_mode(if mode == "all" {
                        agent_core::QueueMode::All
                    } else {
                        agent_core::QueueMode::OneAtATime
                    });
                    // Same "best-effort, in-memory toggle already took effect either way" persistence
                    // `set_auto_compaction` above uses — a failed write just won't survive a restart.
                    // See `settings::Settings::steering_mode`'s own doc comment: this RPC handler is
                    // exactly where that persistence is expected to happen.
                    if let Err(e) = settings_store.set_steering_mode(Some(mode.to_string())) {
                        eprintln!("serve: failed to persist steering_mode setting: {e}");
                    }
                    emit!(response(
                        id,
                        "set_steering_mode",
                        true,
                        Some(json!({ "mode": mode })),
                        None,
                    ));
                }
                _ => emit!(response(
                    id,
                    "set_steering_mode",
                    false,
                    None,
                    Some("missing `mode`; expected \"one_at_a_time\" or \"all\"")
                )),
            },
            // Same idea, for the follow-up lane drained at a stop boundary (plus any stranded steer
            // messages swept in there — see `Steering::drain_at_stop`).
            "set_follow_up_mode" => match cmd.get("mode").and_then(Value::as_str) {
                Some(mode @ ("one_at_a_time" | "all")) => {
                    steering.set_follow_up_mode(if mode == "all" {
                        agent_core::QueueMode::All
                    } else {
                        agent_core::QueueMode::OneAtATime
                    });
                    if let Err(e) = settings_store.set_follow_up_mode(Some(mode.to_string())) {
                        eprintln!("serve: failed to persist follow_up_mode setting: {e}");
                    }
                    emit!(response(
                        id,
                        "set_follow_up_mode",
                        true,
                        Some(json!({ "mode": mode })),
                        None,
                    ));
                }
                _ => emit!(response(
                    id,
                    "set_follow_up_mode",
                    false,
                    None,
                    Some("missing `mode`; expected \"one_at_a_time\" or \"all\"")
                )),
            },
            "get_available_models" => {
                // Deliberately the full known-model hint list, not `cycle_models` — `--models` scopes
                // what `cycle_model` steps through, not what a client's model *picker* can see. pi's
                // own `/model` selector defaults its view to the scope but still lets the operator Tab
                // to the full catalog; since this RPC has no such secondary toggle, it always answers
                // with the unscoped list and leaves any scoped-vs-all UI distinction to the client.
                let models: Vec<Value> = available_models().iter().map(|m| model_info(m)).collect();
                emit!(response(
                    id,
                    "get_available_models",
                    true,
                    Some(json!({ "models": models })),
                    None,
                ));
            }
            "list_branches" => {
                emit!(response(
                    id,
                    "list_branches",
                    true,
                    Some(json!({ "branches": persistence.list_branches() })),
                    None,
                ));
            }
            "get_tree" => {
                // `since` (Task #48, pi-parity gap): return only entries appended after a tree id the
                // client already has, across every entry type — see [`nodes_since`]. Unmatched is an
                // error, not a silent full re-fetch, mirroring `get_messages`'s own `since` above.
                let nodes = match cmd.get("since").and_then(Value::as_str) {
                    Some(since) => match nodes_since(persistence.tree(), since) {
                        Ok(nodes) => nodes,
                        Err(e) => {
                            emit!(response(id, "get_tree", false, None, Some(&e)));
                            continue;
                        }
                    },
                    None => persistence.tree(),
                };
                emit!(response(
                    id,
                    "get_tree",
                    true,
                    Some(json!({ "nodes": nodes, "leaf_id": persistence.active_ids().last() })),
                    None,
                ));
            }
            "switch_branch" => match cmd.get("target_id").and_then(Value::as_str) {
                Some(target_id) => {
                    // When set, switches to `target_id`'s own parent instead — the tree's root (before
                    // any message) when `target_id` is the very first one, letting a client redo it in
                    // place. Mirrors `fork`'s identical `before` flag.
                    let before = cmd.get("before").and_then(Value::as_bool).unwrap_or(false);
                    // Defaults to summarizing the abandoned branch's activity (mirroring pi's
                    // `navigateTree`); a client can pass `summarize:false` for a quick, cheap switch.
                    let summarize = cmd
                        .get("summarize")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                    // Steers what the abandoned branch's recap emphasizes — the same "Additional
                    // focus" framing the `compact` command already supports (see its own
                    // `custom_instructions` handling above). Ignored when `summarize` is false, since
                    // no summarization call happens at all in that case.
                    let custom_instructions =
                        cmd.get("custom_instructions").and_then(Value::as_str);
                    // Task #17 (pi-parity fix): `true` uses `custom_instructions` as the *entire*
                    // instruction section instead of appending it after the default structured
                    // template — see `Agent::summarize_branch`'s own doc comment for the exact
                    // semantics. Read the same way `custom_instructions` itself is, just above; a no-op
                    // when `custom_instructions` is absent, or when `summarize` is false (no
                    // summarization call happens at all in that case).
                    let replace_instructions = cmd
                        .get("replace_instructions")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let target_id = target_id.to_string();
                    // Branch summarization is one LLM call, but it's still a real network round trip
                    // — potentially the slowest single operation this loop ever awaits outside a
                    // `prompt` run. Racing it against stdin (mirroring `prompt`'s own busy-select
                    // below, scaled down to this operation's much narrower surface — pi exposes only
                    // `abortBranchSummary` here, nothing like steer/follow_up) keeps an `abort` able to
                    // interrupt it instead of the whole RPC loop going unresponsive until the call
                    // either lands or times out. A stdin close/shutdown mid-call also cancels, so the
                    // loop can still unwind and exit cleanly instead of hanging on this one `.await`.
                    let branch_cancel = CancellationToken::new();
                    let mut branch_stdin_open = true;
                    let switch_result = {
                        let fut = persistence.switch_branch(
                            &agent,
                            &target_id,
                            before,
                            summarize,
                            custom_instructions,
                            replace_instructions,
                            &branch_cancel,
                        );
                        tokio::pin!(fut);
                        loop {
                            tokio::select! {
                                biased;
                                r = &mut fut => break r,
                                sig = shutdown.wait() => {
                                    shutdown_cause = Some(sig);
                                    branch_stdin_open = false;
                                    branch_cancel.cancel();
                                }
                                maybe_line = input_rx.recv(), if branch_stdin_open => match maybe_line {
                                    Some(l) => {
                                        let l = l.trim();
                                        if l.is_empty() {
                                            continue;
                                        }
                                        let c: Value = match serde_json::from_str(l) {
                                            Ok(v) => v,
                                            Err(e) => {
                                                let _ = out_tx.send(response(None, "parse", false, None, Some(&format!("invalid JSON: {e}"))));
                                                continue;
                                            }
                                        };
                                        let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                                        match c.get("type").and_then(Value::as_str).unwrap_or("") {
                                            "abort" => {
                                                branch_cancel.cancel();
                                                let _ = out_tx.send(response(cid, "abort", true, None, None));
                                            }
                                            other => {
                                                let _ = out_tx.send(response(cid, other, false, None, Some("busy: a branch switch is summarizing; only `abort` is accepted until it settles")));
                                            }
                                        }
                                    }
                                    None => {
                                        branch_stdin_open = false;
                                        branch_cancel.cancel();
                                    }
                                }
                            }
                        }
                    };
                    match switch_result {
                        Ok((s, resolved_target)) => {
                            session = s;
                            steering.clear();
                            // Restore whichever model/thinking-level was actually active on this
                            // branch, rather than silently continuing with the process's current
                            // global setting (which may have since moved on via a `set_model`/
                            // `cycle_thinking_level` made on a *different* branch — that's the bleed
                            // this guards against). Both always resolve to something real: the model
                            // falls back to the session's own creation-time model, the level to the
                            // process's own starting level (`starting_level`) — see
                            // `model_and_level_at`'s doc comment. Queried against `resolved_target`
                            // (`None` when `before` landed on the tree's own root), not the raw
                            // `target_id` argument, which names the entry navigated *relative to*, not
                            // necessarily where the session actually ended up.
                            let (restored_model, restored_level) = persistence
                                .model_and_level_at_opt(resolved_target.as_deref(), starting_level);
                            // Clamp against the *restored* model's capabilities, not the model that
                            // was active before the switch — a branch recorded at `Off` on a model
                            // that's since been superseded by a non-disable-capable one must not
                            // resurrect an illegal `Off` state.
                            let restored_level = agent_core::clamp_thinking_level(
                                &agent_core::capabilities(&restored_model),
                                restored_level,
                            );
                            let mut rebuild_needed = false;
                            if restored_model != current_model {
                                session.scrub_cross_model_state(&restored_model);
                                // The restored model can be on a different OAuth provider than whatever was
                                // active before this switch — best-effort, matching this restore path's existing
                                // "recorded state wins" philosophy: a failure here leaves `client` on the previous
                                // (now-wrong-for-`current_model`) credential rather than aborting the whole switch,
                                // so the next `prompt` surfaces its own clear transport-level error instead.
                                match build_gateway_client(&cfg, &restored_model) {
                                    Ok(new_client) => client = Arc::new(new_client),
                                    Err(e) => eprintln!(
                                        "serve: could not resolve a gateway credential for restored model {restored_model:?}: \
                                         {e} — keeping the previous client"
                                    ),
                                }
                                current_model = restored_model;
                                rebuild_needed = true;
                            }
                            if restored_level != current_level {
                                current_level = restored_level;
                                current_thinking = None;
                                rebuild_needed = true;
                            }
                            if rebuild_needed {
                                agent = build_agent(
                                    client.clone(),
                                    &full_system(&static_system, &cwd),
                                    &cfg,
                                    &current_model,
                                    current_thinking,
                                    current_level,
                                    current_auto_compaction,
                                    current_auto_retry,
                                    current_block_images,
                                    current_image_auto_resize,
                                    persistence.session_id(),
                                    &write_locks,
                                    &checkpoint,
                                    subagent_ctx.as_ref(),
                                    structured_output.as_ref(),
                                    memory_tool.as_ref(),
                                    approval.as_ref(),
                                );
                            }
                            emit!(response(
                                id,
                                "switch_branch",
                                true,
                                Some(json!({
                                    "target_id": target_id,
                                    "model": current_model,
                                    "reasoning_effort": current_level.as_str(),
                                })),
                                None,
                            ));
                        }
                        // Not a failure — the client asked to abort, and the session is left exactly
                        // as it was (no partial summary, no switch): mirrors pi's `navigateTree`
                        // result shape (`{cancelled:true, aborted:true}`) rather than surfacing this as
                        // an RPC-level error.
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                            emit!(response(
                                id,
                                "switch_branch",
                                true,
                                Some(json!({
                                    "target_id": target_id,
                                    "cancelled": true,
                                    "aborted": true,
                                })),
                                None,
                            ));
                        }
                        Err(e) => emit!(response(
                            id,
                            "switch_branch",
                            false,
                            None,
                            Some(&e.to_string())
                        )),
                    }
                }
                None => emit!(response(
                    id,
                    "switch_branch",
                    false,
                    None,
                    Some("missing `target_id`")
                )),
            },
            "bash" => {
                let Some(command) = cmd.get("command").and_then(Value::as_str) else {
                    emit!(response(id, "bash", false, None, Some("missing `command`")));
                    continue;
                };
                let Some(tool) = bash_tool.clone() else {
                    emit!(response(
                        id,
                        "bash",
                        false,
                        None,
                        Some(
                            "the `bash` tool is not registered for this process \
                             (excluded via --exclude-tools/--no-tools)"
                        ),
                    ));
                    continue;
                };
                let mut input = json!({ "command": command });
                if let Some(cwd) = cmd.get("cwd").and_then(Value::as_str) {
                    input["cwd"] = json!(cwd);
                }
                if let Some(timeout_ms) = cmd.get("timeout_ms").and_then(Value::as_u64) {
                    input["timeout_ms"] = json!(timeout_ms);
                }
                // Track L23 (pi-parity fix): this host command used to call `tool.run_streaming`
                // directly, bypassing `--deny-tool`/`--deny-bash-pattern`/`--deny-path` entirely (only
                // `--exclude-tools bash` — the `bash_tool` check above — actually gated it). Checked
                // before the `ToolStart` event fires, matching the "not registered" early-return just
                // above: a blocked call never starts, so it never needs a `tool_end` to close it out.
                if let Some(reason) = bash_policy
                    .before_tool_call("bash", &input, &session, &CancellationToken::new())
                    .await
                {
                    emit!(response(id, "bash", false, None, Some(&reason)));
                    continue;
                }
                // Always recorded into `session`/persisted storage/`export.rs`'s rendered output — pi's
                // `recordBashResult` (`agent-session.ts`) unconditionally records too. `exclude_from_context`
                // (Fix 9, pi-parity gap: previously misdescribed here as gating *recording* itself, and
                // implemented that way too — see the `session.push`/`persist` call below, now unconditional)
                // instead gates only the separate transform that builds what's actually sent to the model
                // on its *next* turn (`ServeHooks::before_provider_request`) — pi's identical two-layer split
                // (`recordBashResult` vs. `convertToLlm`, `messages.ts`). So a diagnostic command run outside
                // the model's own turn is always visible in history; only whether the model itself sees it
                // next turn is what this flag controls.
                let exclude_from_context = cmd
                    .get("exclude_from_context")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let command = command.to_string();
                host_bash_seq += 1;
                let call_id = format!("host-bash-{host_bash_seq}");
                if let Some(frame) = event_frame(AgentEvent::ToolStart {
                    id: call_id.clone(),
                    name: "bash".to_string(),
                    input: input.clone(),
                }) {
                    emit!(frame);
                }

                // Independent of the model loop and the session — runs directly against the
                // registered `bash` tool, reusing its own streaming/truncation/cancellation
                // machinery. Races the command against continued stdin reads so `abort_bash`/`abort`
                // (or shutdown) can cancel it; any other command is rejected as busy, the same shape
                // as the `prompt` arm's own busy-loop, simplified with no session/steering involved.
                let cancel = CancellationToken::new();
                let (prog_tx, mut prog_rx) = futures::channel::mpsc::unbounded::<ToolUpdate>();
                let progress = agent_core::ToolProgress::new(
                    prog_tx,
                    call_id.clone(),
                    "bash".to_string(),
                    cancel.clone(),
                );
                let mut stdin_open = true;
                // Fix 7 (pi-parity gap): `tools::bash`'s own `truncation_details` (the `{truncation,
                // full_output_path}` payload — see that module's doc comment) only ever streams on an
                // interim `ToolUpdate::Progress`, never on the terminal outcome — captured here so the
                // response built below can carry it as its own structured field instead of only ever
                // reaching a client via a progress event that a client polling just for the final
                // result never sees at all.
                let mut last_bash_details: Option<Value> = None;
                let outcome = {
                    let run = tool.run_streaming(input, &progress);
                    tokio::pin!(run);
                    loop {
                        tokio::select! {
                            biased;
                            r = &mut run => break r,
                            // `cancel.cancel()` alone (set by `abort_bash`/`abort` below, or by
                            // shutdown) doesn't, by itself, interrupt an in-flight `run` — but `bash`'s
                            // own `exec` races the same token internally (see `tools::bash`) and returns
                            // a graceful "Command cancelled" error carrying whatever output had already
                            // accumulated, so in practice `r = &mut run` above resolves first and wins
                            // this biased select on a real cancellation. This arm stays as a fallback for
                            // a future tool reachable from this same host-command path that *isn't*
                            // internally cancellation-aware — breaking here drops the pinned `run`
                            // future, which still kills a `bash` subprocess via its
                            // `kill_on_drop`/process-group guard even if that tool never noticed.
                            () = cancel.cancelled() => {
                                break Err(agent_core::ToolError::Execution("cancelled".to_string()));
                            }
                            sig = shutdown.wait() => {
                                shutdown_cause = Some(sig);
                                stdin_open = false;
                                cancel.cancel();
                            }
                            update = prog_rx.next() => {
                                if let Some(ToolUpdate::Progress { id, name, snapshot, details }) = update {
                                    if details.is_some() {
                                        last_bash_details = details.clone();
                                    }
                                    if let Some(frame) = event_frame(AgentEvent::ToolProgress { id, name, snapshot, details }) {
                                        let _ = out_tx.send(frame);
                                    }
                                }
                            }
                            maybe_line = input_rx.recv(), if stdin_open => match maybe_line {
                                Some(l) => {
                                    let l = l.trim();
                                    if l.is_empty() {
                                        continue;
                                    }
                                    let c: Value = match serde_json::from_str(l) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            let _ = out_tx.send(response(None, "parse", false, None, Some(&format!("invalid JSON: {e}"))));
                                            continue;
                                        }
                                    };
                                    let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                                    match c.get("type").and_then(Value::as_str).unwrap_or("") {
                                        "abort_bash" | "abort" => {
                                            cancel.cancel();
                                            let _ = out_tx.send(response(cid, "abort_bash", true, None, None));
                                        }
                                        other => {
                                            let _ = out_tx.send(response(cid, other, false, None, Some("busy: a host bash command is running; only `abort_bash`/`abort` are accepted")));
                                        }
                                    }
                                }
                                None => {
                                    stdin_open = false;
                                    cancel.cancel();
                                }
                            }
                        }
                    }
                };
                // Flush anything buffered between the final poll above and the tool's own return.
                while let Ok(update) = prog_rx.try_recv() {
                    if let ToolUpdate::Progress {
                        id,
                        name,
                        snapshot,
                        details,
                    } = update
                    {
                        if details.is_some() {
                            last_bash_details = details.clone();
                        }
                        if let Some(frame) = event_frame(AgentEvent::ToolProgress {
                            id,
                            name,
                            snapshot,
                            details,
                        }) {
                            let _ = out_tx.send(frame);
                        }
                    }
                }

                let (result_text, is_error) = match outcome {
                    Ok(output) => (output.text, false),
                    Err(e) => (e.to_string(), true),
                };
                // Fix 7 (pi-parity gap): pi's own `BashResult{output, exitCode, cancelled, truncated,
                // fullOutputPath}` (`bash-executor.ts`) alongside the text, not flattened into it — see
                // `bash_exit_code_from_status_line`/`bash_result_was_cancelled`'s own doc comments for
                // how each is recovered from what `tools::bash` already reports (its own status-line
                // text, and the last progress `details` captured above).
                let exit_code = bash_exit_code_from_status_line(&result_text, is_error);
                let cancelled = bash_result_was_cancelled(&result_text, is_error);
                let truncated = last_bash_details
                    .as_ref()
                    .and_then(|d| d.get("truncation"))
                    .and_then(|t| t.get("truncated"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let full_output_path = last_bash_details
                    .as_ref()
                    .and_then(|d| d.get("full_output_path"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                // Fix 3 (pi-parity gap): thread these same four fields into the *persisted* message
                // too, as a leading status line `export.rs`'s host-bash parser now understands — see
                // `HOST_BASH_STATUS_LINE_PREFIX`'s own doc comment for why this rides on the marker
                // text rather than a new `agent_core::Message` field. Previously only a bare `is_error`
                // bool (the `"(error)\n"` marker below) ever reached the persisted message, even though
                // this RPC response has always reported the real exit code/cancelled/truncated/
                // full-output-path live, a few lines down.
                let status_line = format!(
                    "{HOST_BASH_STATUS_LINE_PREFIX}{}\n",
                    json!({
                        "exit_code": exit_code,
                        "cancelled": cancelled,
                        "truncated": truncated,
                        "full_output_path": full_output_path,
                    })
                );
                if let Some(frame) = event_frame(AgentEvent::ToolEnd {
                    id: call_id,
                    name: "bash".to_string(),
                    result: result_text.clone(),
                    is_error,
                }) {
                    emit!(frame);
                }
                // This command has no `tool_use` to pair a `tool_result` against (it never went
                // through the model), so it's recorded as a plainly-labeled informational user message
                // instead — structurally impossible to run this arm while a turn is streaming (the
                // busy-loop above rejects everything but `abort_bash`/`abort` while a host bash command
                // is in flight, and this command is only reachable from the idle loop to begin with), so
                // there's no tool_use/tool_result ordering to protect against, unlike pi's own
                // stream-in-flight deferral.
                //
                // Fix 9 (pi-parity gap): always pushed and persisted now, regardless of
                // `exclude_from_context` — pi's own `recordBashResult` (`agent-session.ts`) unconditionally
                // records into `agent.state.messages` too; `excludeFromContext` is consulted only later, in
                // the separate `convertToLlm` transform that builds what's actually sent to the model
                // (`messages.ts`). This used to skip recording entirely when the flag was set, losing the
                // command/output outright instead of merely hiding it from the model — it never showed up
                // in `session.messages`, persisted storage, or `export.rs`'s rendered output either.
                // `HOST_BASH_EXCLUDED_LABEL` is this crate's stand-in for pi's dedicated
                // `BashExecutionMessage.excludeFromContext` field (`agent_core::Message` has no such field —
                // see the label's own doc comment); `ServeHooks::before_provider_request`, installed on
                // every `build_agent` agent, is the "what's actually sent to the model" transform that
                // consults it.
                let label = if exclude_from_context {
                    HOST_BASH_EXCLUDED_LABEL
                } else {
                    HOST_BASH_LABEL
                };
                session.push(agent_core::Message::user(format!(
                    "{label}\n$ {command}\n\n{status_line}{}{result_text}",
                    if is_error { "(error)\n" } else { "" }
                )));
                if let Err(e) = persistence.persist(&session, None) {
                    eprintln!("serve: failed to persist host bash result: {e}");
                }
                emit!(response(
                    id,
                    "bash",
                    true,
                    Some(json!({
                        "result": result_text,
                        "is_error": is_error,
                        "exit_code": exit_code,
                        "cancelled": cancelled,
                        "truncated": truncated,
                        "full_output_path": full_output_path,
                    })),
                    None,
                ));
                if !stdin_open {
                    break;
                }
            }
            "abort_bash" => {
                // No host bash is in flight (a running one is handled inside the `bash` arm above), so
                // there is nothing to cancel — acknowledge idempotently, matching `abort`'s idle shape.
                emit!(response(id, "abort_bash", true, None, None));
            }
            "login" => {
                let provider_id = cmd
                    .get("provider")
                    .and_then(Value::as_str)
                    .and_then(crate::oauth::OAuthProviderId::parse);
                match provider_id {
                    None => {
                        emit!(response(
                            id,
                            "login",
                            false,
                            None,
                            Some("missing or unknown `provider`"),
                        ));
                    }
                    Some(_) if lock_ignoring_poison(&pending_login).is_some() => {
                        emit!(response(
                            id,
                            "login",
                            false,
                            None,
                            Some(
                                "busy: a login is already in flight; use `abort_login` to cancel it \
                                 first"
                            ),
                        ));
                    }
                    Some(provider_id) => {
                        emit!(ack(id.clone(), "login"));
                        // Runs as a detached task, not inline like `prompt`'s own busy loop: unlike a
                        // model turn, a login has nothing to do with `session`/`agent` state, so there's
                        // no reason every *other* idle command (`list_sessions`, `logout` of a different
                        // provider, …) should have to wait out however long a user takes to finish a
                        // browser flow. Only a second concurrent `login` is rejected (above).
                        let login_cancel = agent_core::CancellationToken::new();
                        let pending_code: PendingCodeSlot = Arc::new(std::sync::Mutex::new(None));
                        *lock_ignoring_poison(&pending_login) = Some(PendingLogin {
                            cancel: login_cancel.clone(),
                            pending_code: pending_code.clone(),
                        });
                        let out_tx_bg = out_tx.clone();
                        let pending_login_bg = pending_login.clone();
                        let login_id = id.clone();
                        tokio::spawn(async move {
                            // Always clears `pending_login` when this task ends, even on panic — see
                            // `PendingLoginGuard`'s own doc comment.
                            let _reset_pending_login_on_exit =
                                PendingLoginGuard(pending_login_bg.clone());
                            let callbacks = ServeLoginCallbacks {
                                out_tx: out_tx_bg.clone(),
                                id: login_id.clone(),
                                provider: provider_id,
                                pending_code,
                            };
                            let result =
                                crate::oauth::login(provider_id, &callbacks, &login_cancel).await;
                            let frame = match result {
                                Ok(credential) => {
                                    // `AuthStore::set`'s cross-process `FileLock` can block briefly under
                                    // contention — kept off this task's own async context the same way
                                    // `persist_blocking` keeps session persistence off the caller's.
                                    let saved = tokio::task::spawn_blocking(move || {
                                        crate::auth_store::AuthStore::open_default()
                                            .set(provider_id.store_key(), credential)
                                    })
                                    .await;
                                    match saved {
                                        Ok(Ok(())) => response(
                                            login_id,
                                            "login",
                                            true,
                                            Some(json!({
                                                "provider": provider_id.to_string(),
                                                "status": "logged_in",
                                            })),
                                            None,
                                        ),
                                        Ok(Err(e)) => response(
                                            login_id,
                                            "login",
                                            false,
                                            None,
                                            Some(&format!(
                                                "logged in but failed to save credential: {e}"
                                            )),
                                        ),
                                        Err(e) => response(
                                            login_id,
                                            "login",
                                            false,
                                            None,
                                            Some(&format!(
                                                "logged in but failed to save credential: {e}"
                                            )),
                                        ),
                                    }
                                }
                                Err(e) => {
                                    response(login_id, "login", false, None, Some(&e.to_string()))
                                }
                            };
                            let _ = out_tx_bg.send(frame);
                            // `_reset_pending_login_on_exit`'s `Drop` clears `pending_login` here.
                        });
                    }
                }
            }
            "submit_code" => {
                let code = cmd.get("code").and_then(Value::as_str).map(str::to_string);
                let accepted = match (code, lock_ignoring_poison(&pending_login).as_ref()) {
                    (Some(code), Some(p)) => match lock_ignoring_poison(&p.pending_code).take() {
                        Some(tx) => tx.send(code).is_ok(),
                        None => false,
                    },
                    _ => false,
                };
                emit!(response(
                    id,
                    "submit_code",
                    true,
                    Some(json!({ "accepted": accepted })),
                    None,
                ));
            }
            "abort_login" => {
                // Idempotent no-op if none is in flight — matches `abort`'s own idle-mode convention.
                if let Some(p) = lock_ignoring_poison(&pending_login).as_ref() {
                    p.cancel.cancel();
                }
                emit!(response(id, "abort_login", true, None, None));
            }
            "logout" => {
                let provider_id = cmd
                    .get("provider")
                    .and_then(Value::as_str)
                    .and_then(crate::oauth::OAuthProviderId::parse);
                match provider_id {
                    None => emit!(response(
                        id,
                        "logout",
                        false,
                        None,
                        Some("missing or unknown `provider`"),
                    )),
                    Some(provider_id) => {
                        let mut store = crate::auth_store::AuthStore::open_default();
                        match store.remove(provider_id.store_key()) {
                            Ok(was_logged_in) => emit!(response(
                                id,
                                "logout",
                                true,
                                Some(json!({
                                    "provider": provider_id.to_string(),
                                    "was_logged_in": was_logged_in,
                                })),
                                None,
                            )),
                            Err(e) => {
                                emit!(response(id, "logout", false, None, Some(&e.to_string())))
                            }
                        }
                    }
                }
            }
            "auth_status" => {
                let store = crate::auth_store::AuthStore::open_default();
                let status_of = |pid: crate::oauth::OAuthProviderId| {
                    let status = match store.get(pid.store_key()) {
                        None => "logged_out",
                        Some(stored) if stored.last_refresh_error.is_some() => "needs_reauth",
                        Some(_) => "logged_in",
                    };
                    json!({ "provider": pid.to_string(), "status": status })
                };
                match cmd.get("provider").and_then(Value::as_str) {
                    Some(p) => match crate::oauth::OAuthProviderId::parse(p) {
                        Some(pid) => {
                            emit!(response(
                                id,
                                "auth_status",
                                true,
                                Some(status_of(pid)),
                                None
                            ))
                        }
                        None => emit!(response(
                            id,
                            "auth_status",
                            false,
                            None,
                            Some("unknown `provider`")
                        )),
                    },
                    None => {
                        let providers: Vec<Value> = crate::oauth::OAuthProviderId::all()
                            .iter()
                            .map(|&pid| status_of(pid))
                            .collect();
                        emit!(response(
                            id,
                            "auth_status",
                            true,
                            Some(json!({ "providers": providers })),
                            None,
                        ));
                    }
                }
            }
            other => {
                emit!(response(id, other, false, None, Some("unknown command")));
            }
        }
    }

    drop(out_tx);
    let _ = writer.await;
    Ok(shutdown_cause)
}

/// Join the cached static system prompt with a freshly-computed dynamic footer (current date/cwd) —
/// the full prompt text an `Agent` should carry. Cheap: `static_system` is already-computed text and
/// `dynamic_footer` does no filesystem discovery, so this is safe to call every turn (see the `prompt`
/// arm's per-turn refresh) as well as at every `build_agent` rebuild.
fn full_system(static_system: &str, cwd: &std::path::Path) -> String {
    format!("{static_system}{}", crate::resources::dynamic_footer(cwd))
}

/// The current `MEMORY.md` index for prompt injection, re-read from the backend so a rebuilt system
/// prompt reflects memories written since startup. `None` when memory is disabled.
async fn current_memory_index(
    backend: &Option<Arc<dyn crate::memory::MemoryBackend>>,
) -> Option<String> {
    match backend {
        Some(b) => Some(b.index().await.unwrap_or_default()),
        None => None,
    }
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

/// The extra per-request headers a `models.json` override configures for this model id, if any (Task
/// #11 pi-parity feature) — the lib-crate-side twin of `main.rs::model_override_extra_headers`. Not
/// literally shared with it: `main.rs` is a separate binary crate over this library (see
/// `beyond-ai-agent`'s `Cargo.toml` `[[bin]]` target), so a `pub(crate)` there still wouldn't be
/// visible here, and vice versa — the two must each call through to
/// `settings::ModelOverrides::open_default()`/`ModelOverride::resolved_headers` (the one real shared
/// primitive both live in this same library crate and already depend on) rather than one calling the
/// other. Kept to the same trivial one-lookup shape as `main.rs`'s copy so the two can't drift on
/// anything but this wrapper itself.
fn model_override_extra_headers(model: &str) -> std::collections::HashMap<String, String> {
    crate::settings::ModelOverrides::open_default()
        .get(model)
        .map(|over| over.resolved_headers())
        .unwrap_or_default()
}

/// Resolve `model`'s gateway credential/routing ([`resolve_gateway_credential`]) and build a
/// ready-to-use [`GatewayClient`] for it, applying the process's fixed retry policy. Called once at
/// `serve` startup and again every time the active model changes at runtime (`set_model`,
/// `cycle_model`, and `switch_session`/`fork`/`clone`/`switch_branch` restoring a session's own
/// recorded model) — never cached across a model switch, since the resolved provider/routing can be
/// completely different from one model to the next (an OAuth login resolved for an Anthropic model has
/// no bearing on a GitHub Copilot one). This is the fix for the credential/routing having previously
/// been resolved exactly once, before `serve` even started, and then silently reused — stale provider,
/// stale routing — by every later model switch for the rest of the process's life.
fn build_gateway_client(cfg: &ServeConfig, model: &str) -> Result<GatewayClient, String> {
    let credential = resolve_gateway_credential(cfg.key.clone(), model)?;
    let client = match credential {
        GatewayCredential::Static(key) => {
            GatewayClient::new(cfg.gateway.clone(), key).map_err(|e| e.to_string())?
        }
        GatewayCredential::Oauth(source) => {
            GatewayClient::with_credential_source(cfg.gateway.clone(), source)
                .map_err(|e| e.to_string())?
        }
    }
    .with_retry(
        cfg.retry_max_retries
            .unwrap_or(agent_core::client::MAX_RETRIES),
        cfg.retry_base_delay_ms
            .unwrap_or(agent_core::client::BASE_BACKOFF),
    )
    // pi-parity (Task 2, serve pass 19): `main.rs::run_task`'s identical `with_extra_headers` wiring
    // (Task #11) had no `serve` counterpart — every `serve` entrypoint that (re)builds a gateway client
    // (startup, `set_model`, `cycle_model`, `fork`, `clone`, `switch_session`, `switch_branch`) went
    // through this one function, so a `models.json` `ModelOverride.headers` override (and the
    // auto-seeded NVIDIA `NVCF-POLL-SECONDS` / Kimi-Coding `User-Agent` defaults —
    // `ModelOverride::resolved_headers`'s own doc comment) silently never reached the wire under
    // `serve`, even though it did under `run`. Fixed once, here, rather than at each of those 8+ call
    // sites: chained right after `with_retry`, matching `run_task`'s own ordering — harmless (a no-op)
    // when no override configures any headers, since an empty map is also `GatewayClient::new`'s own
    // default.
    .with_extra_headers(model_override_extra_headers(model));
    // Task #30 (pi-parity feature): `run`'s identical `--retry-max-backoff-ms` wiring (`main.rs::
    // run_task`'s own `with_max_backoff` call site) previously had no `serve` counterpart — called on
    // every model switch, same as `with_retry` above, so the override survives a mid-run switch.
    let client = if let Some(max_backoff) = cfg.retry_max_backoff_ms {
        client.with_max_backoff(max_backoff)
    } else {
        client
    };
    // W3 (shared upstream pool): inject the daemon-wide `reqwest::Client` so every session multiplexes
    // over one connection pool instead of each opening its own. **Must** precede `with_idle_timeout`
    // below: `with_http_client` sets the `http_shared` guard, which makes that later call a no-op rather
    // than rebuilding (and thereby tearing down) the shared pool on a model switch or idle override. The
    // shared client's own read timeout is fixed when the supervisor builds it (see
    // `serve_ws::build_shared_h2_client`). `None` (stdio/`run`, or `--upstream-http2 off`) is unchanged.
    let client = match &cfg.shared_http {
        Some(h) => client.with_http_client(h.clone()),
        None => client,
    };
    // Task #38 (pi-parity fix): `run`'s identical `--idle-timeout-ms` wiring (`main.rs::run_task`'s own
    // `with_idle_timeout` call site) previously had no `serve` counterpart — this is called on every
    // model switch (`set_model`/`cycle_model` rebuild the whole client), same as `with_retry` above, so
    // the override survives a mid-run switch rather than only applying at startup.
    let client = if let Some(ms) = cfg.idle_timeout_ms {
        client
            .with_idle_timeout(std::time::Duration::from_millis(ms))
            .map_err(|e| e.to_string())?
    } else {
        client
    };
    Ok(client)
}

/// Build the [`Agent`] for the current model + thinking budget + auto-compaction/auto-retry flags.
/// Called once at startup and again on every `set_model`/`set_thinking`/`cycle_model`/
/// `cycle_thinking_level`/`set_auto_compaction`/`set_auto_retry`, so a client can re-tune the run
/// without restarting `serve`. The transport, tools, system prompt, loop bounds, and cache settings are
/// the same each time; only the model id, thinking budget, and the two flags vary. Compaction's
/// `context_window` defaults to `model`'s own capabilities; an explicit `--context-window` overrides
/// that and stays pinned across a model switch (the operator's compaction budget, not the dialect's) —
/// left unset, each switch picks up the *new* model's real window instead of a stale operator number.
/// `reserve_tokens`/`keep_recent_tokens` default to `CompactionConfig::default()`, overridable
/// Build the subagent context once, at startup. Its transport factory closes over a clone of `cfg` and
/// calls [`build_gateway_client`] per child model — the same model-switch-safe recipe the parent uses —
/// so the ctx is valid for the whole session regardless of parent model switches.
///
/// `skills` are the parent session's own discovered skills, so a worktree/read-only child sees the same
/// `<available_skills>` the parent does. `prompt_guidelines` is empty: `serve` renders its base prompt
/// once into `cfg.system` and doesn't carry the raw guideline list past that.
#[allow(clippy::too_many_arguments)]
fn build_subagent_ctx(
    cfg: &ServeConfig,
    cwd: &Path,
    project_trusted: bool,
    parent_model: &str,
    write_locks: &Arc<agent_core::WriteLockRegistry>,
    skills: &[crate::skills::Skill],
    // The parent's memory backend, shared by reference so the whole subagent tree reads/writes one
    // durable store — see `SubagentCtx::memory_backend`.
    memory_backend: Option<Arc<dyn crate::memory::MemoryBackend>>,
    // Shared with the parent, not rebuilt — see `SubagentCtx::approval`.
    approval: Option<&crate::approval::ApprovalRuntime>,
) -> Arc<crate::tools::subagent::SubagentCtx> {
    use crate::tools::subagent;
    let cfg_for_factory = cfg.clone();
    let factory: subagent::TransportFactory = Arc::new(move |m: &str| {
        build_gateway_client(&cfg_for_factory, m)
            .map(|c| Arc::new(c) as Arc<dyn agent_core::ModelTransport>)
    });
    // The parent's effective set (after `--tools`/`--exclude-tools`) minus `subagent`, so a child with
    // no `tools:` of its own inherits exactly what the parent may do — no more.
    let mut probe = build_tools(cfg, cfg.image_auto_resize);
    crate::tools::apply_filter(
        &mut probe,
        cfg.tools.as_deref(),
        cfg.exclude_tools.as_deref(),
        cfg.no_tools,
    );
    let parent_tools: Vec<String> = probe
        .definitions()
        .into_iter()
        .map(|d| d.name)
        .filter(|n| n != subagent::NAME)
        .collect();
    Arc::new(subagent::SubagentCtx {
        factory,
        agents: Arc::new(cfg.agents.clone()),
        skills: Arc::new(skills.to_vec()),
        write_locks: write_locks.clone(),
        mcp_tools: cfg.mcp_tools.clone(),
        memory_backend,
        tool_cfg: subagent::ChildToolConfig {
            bash_timeout_ms: cfg.bash_timeout_ms,
            bash_shell_path: cfg.bash_shell_path.clone(),
            bash_command_prefix: cfg.bash_command_prefix.clone(),
            web_allow_private: cfg.web_allow_private,
            web_allow_hosts: cfg.web_allow_hosts.clone(),
            web_timeout_ms: cfg.web_timeout_ms,
            image_auto_resize: cfg.image_auto_resize,
        },
        cwd: cwd.to_path_buf(),
        project_trusted,
        prompt_guidelines: Vec::new(),
        parent_model: parent_model.to_string(),
        parent_cache_key: parent_model.to_string(),
        parent_tools,
        deny_tool: cfg.deny_tool.clone(),
        deny_bash_pattern: cfg.deny_bash_pattern.clone(),
        deny_path: cfg.deny_path.clone(),
        child_max_steps: subagent::DEFAULT_CHILD_MAX_STEPS,
        max_depth: subagent::DEFAULT_MAX_DEPTH,
        approval: approval.cloned(),
    })
}

/// independently of `context_window`.
// 11 arguments, all independent inputs every call site already has on hand from `cfg`/local
// runtime-switchable state — bundling them into a struct would just be a second place those same
// fields live, not a reduction in what the function needs to know (see `client.rs::send_with_retry`
// for the same tradeoff). Private, single-purpose helper, not a public API shape.
#[allow(clippy::too_many_arguments)]
fn build_agent(
    transport: Arc<GatewayClient>,
    system: &str,
    cfg: &ServeConfig,
    model: &str,
    thinking: Option<u32>,
    level: agent_core::ThinkingLevel,
    auto_compaction: bool,
    auto_retry: bool,
    // Pi-parity fix (pass 20): live counterparts of `cfg.block_images`/`cfg.image_auto_resize` — see
    // `current_block_images`/`current_image_auto_resize`'s own doc comment in `serve` for why these are
    // now explicit, runtime-switchable parameters (mirroring `auto_compaction`/`auto_retry` just above)
    // instead of this function reading `cfg.block_images`/`cfg.image_auto_resize` straight off the
    // static startup config on every rebuild.
    block_images: bool,
    image_auto_resize: bool,
    cache_key: &str,
    write_locks: &Arc<agent_core::WriteLockRegistry>,
    checkpoint: &Arc<dyn agent_core::CheckpointHook>,
    subagent_ctx: Option<&Arc<crate::tools::subagent::SubagentCtx>>,
    // The `structured_output` tool for the schema currently installed on this session, if any (see the
    // `prompt` command's `output_schema`). An `Arc` shared across every rebuild rather than
    // reconstructed here: it owns the compiled validator, and — more importantly — the `OutputSlot` the
    // model's answer lands in, which must survive a mid-run `set_model` rebuild.
    structured_output: Option<&Arc<crate::tools::structured_output::StructuredOutput>>,
    // The `memory` tool for this session, when memory is enabled. An `Arc<dyn Tool>` shared across every
    // rebuild (like `structured_output`) so a mid-run `set_model` doesn't drop it — it wraps the
    // host-owned `MemoryBackend`, whose store must persist across a model switch.
    memory_tool: Option<&Arc<dyn agent_core::Tool>>,
    // The interactive approval gate, when `--approve` asked for one. Shared (not rebuilt) across every
    // `set_model` rebuild so the session's remembered "always allow" decisions — and the one
    // prompt-serializing lock the whole subagent tree queues behind — survive a model switch.
    approval: Option<&crate::approval::ApprovalRuntime>,
) -> Agent {
    let mut compaction = agent_core::CompactionConfig {
        context_window: cfg
            .context_window
            .unwrap_or_else(|| agent_core::capabilities(model).context_window),
        enabled: auto_compaction,
        ..agent_core::CompactionConfig::default()
    };
    if let Some(reserve) = cfg.compaction_reserve_tokens {
        compaction.reserve_tokens = reserve;
    }
    if let Some(keep_recent) = cfg.compaction_keep_recent_tokens {
        compaction.keep_recent_tokens = keep_recent;
    }

    let mut registry = build_tools(cfg, image_auto_resize);
    // Registered here, not in `build_tools`, because this is where `write_locks` is in scope — the ctx
    // needs the same registry every child shares. Reused across rebuilds; the ctx's factory handles the
    // per-child model itself, so a `set_model` rebuild doesn't invalidate it.
    if let Some(ctx) = subagent_ctx {
        registry.register(Arc::new(crate::tools::subagent::Subagent::new(ctx.clone())));
    }
    // After `build_tools`'s `--tools`/`--exclude-tools` filter, like `subagent` above: an allow-list
    // scopes what the agent may *do*, and must not strip the one tool a `prompt {output_schema}` exists
    // to add — leaving a run that can never satisfy the contract it was started with.
    if let Some(tool) = structured_output {
        registry.register(tool.clone());
    }
    // Persistent memory, registered after the filter for the same reason as `structured_output`/
    // `subagent`: a `--tools` allow-list scopes what the agent may *do* and must not strip its memory.
    if let Some(tool) = memory_tool {
        registry.register(tool.clone());
    }
    let mut agent = Agent::new(transport, model.to_string())
        .with_tools(registry)
        .with_system(system.to_string())
        .with_max_steps(cfg.max_steps)
        .with_compaction(compaction)
        .with_auto_retry(auto_retry)
        .with_sequential_tools(cfg.sequential_tools)
        // Pin this session to a warm prompt-cache node via its stable id.
        .with_cache_key(cache_key.to_string())
        .with_cache_long(cfg.cache_long)
        // Shared across every `build_agent` call for this process (including `set_model`/
        // `set_thinking` rebuilds), so file-mutation exclusivity survives a model switch and extends
        // to any other session sharing this same registry.
        .with_write_locks(write_locks.clone())
        // Streams a snapshot of `session.messages` through `checkpoint`'s channel at each durable
        // mid-run point (see `ChannelCheckpoint`), so a long multi-step turn is persisted incrementally
        // instead of only once it fully completes.
        .with_checkpoint_hook(checkpoint.clone());
    // Task #31 (pi-parity feature): `run`'s identical `--branch-summary-reserve-tokens` wiring
    // (`main.rs::run_task`'s own `with_branch_summary_reserve_tokens` call site) previously had no
    // `serve` counterpart — `Agent::with_branch_summary_reserve_tokens` had zero callers in either
    // binary. Applied on every `build_agent` rebuild, like every other `cfg`-sourced override, so it
    // survives a model switch.
    if let Some(reserve) = cfg.branch_summary_reserve_tokens {
        agent = agent.with_branch_summary_reserve_tokens(reserve);
    }
    // `level` (the portable ladder) supplies the default thinking/reasoning-effort pair for whichever
    // mechanism `model`'s capabilities actually call for; `thinking`, when present, is an explicit raw
    // budget override that wins over the level's own derived budget (but never touches reasoning
    // effort, which always comes from `level` — see `current_thinking`'s doc comment in `serve`).
    let (level_thinking, reasoning_effort) =
        agent_core::thinking_for_level(&agent_core::capabilities(model), level);
    if let Some(budget) = thinking.or(level_thinking) {
        agent = agent.with_thinking(budget);
    }
    if let Some(effort) = reasoning_effort {
        agent = agent.with_reasoning_effort(effort);
    }
    if let Some(temperature) = cfg.temperature {
        agent = agent.with_temperature(temperature);
    }
    if let Some(max_tokens) = cfg.max_tokens {
        agent = agent.with_max_tokens(max_tokens);
    }
    // Task #34 (pi-parity fix): forces every image down the vision-downgrade path regardless of the
    // active model's real `supports_vision` capability — `run`'s identical `--block-images` handling
    // (`main.rs::run_task`'s own `with_block_images` call site), previously never threaded through here
    // at all. Pass 20 (pi-parity fix): reads the live `block_images` parameter (settable at runtime via
    // `set_block_images`), not `cfg.block_images` — see this function's own doc comment.
    if block_images {
        agent = agent.with_block_images(true);
    }
    let policy = crate::policy::ToolPolicy::from_lists(
        &cfg.deny_tool,
        &cfg.deny_bash_pattern,
        &cfg.deny_path,
    );
    // Fix 9 (pi-parity gap): always installed now — unlike the bare `ToolPolicy` this used to install
    // only when non-empty, `ServeHooks::before_provider_request` (the `bash` RPC's
    // `exclude_from_context` filter) is needed unconditionally, and `Agent::with_hooks` only ever holds
    // one hook object. See `ServeHooks`'s own doc comment.
    agent = agent.with_hooks(Arc::new(ServeHooks {
        policy,
        approval: approval.cloned(),
    }));
    agent
}

/// The one [`AgentHooks`] implementation `build_agent` installs — composes the optional
/// `--deny-tool`/`--deny-bash-pattern`/`--deny-path` gate ([`crate::policy::ToolPolicy`]) with
/// [`before_provider_request`](AgentHooks::before_provider_request)'s own, unconditional job: filtering
/// [`HOST_BASH_EXCLUDED_LABEL`]-tagged messages out of the outgoing model request (Fix 9, pi-parity
/// gap). `session.messages`/persisted storage/`export.rs`'s rendered output all still carry the
/// message unchanged — only the wire payload a turn actually sends omits it, mirroring pi's
/// `convertToLlm` (`packages/coding-agent/src/core/messages.ts`), which performs the identical
/// late-and-separate filter rather than gating what `recordBashResult` commits to history in the first
/// place.
///
/// It also composes in the interactive approval gate ([`crate::approval`]) — `Agent::with_hooks` only
/// ever holds one hook object, so a second seam has to fold in here rather than replace this one.
struct ServeHooks {
    policy: crate::policy::ToolPolicy,
    /// `None` unless `--approve` asked for one. Static deny still wins first; see
    /// [`crate::approval::gated_before_tool_call`], whose decision order is load-bearing.
    approval: Option<crate::approval::ApprovalRuntime>,
}

#[async_trait::async_trait]
impl AgentHooks for ServeHooks {
    async fn before_tool_call(
        &self,
        name: &str,
        input: &Value,
        session: &Session,
        cancel: &CancellationToken,
    ) -> Option<String> {
        crate::approval::gated_before_tool_call(
            &self.policy,
            self.approval.as_ref(),
            &crate::approval::ApprovalOrigin::Main,
            name,
            input,
            session,
            cancel,
        )
        .await
    }

    async fn before_provider_request(&self, req: &mut agent_core::transport::ModelRequest) {
        if !req.messages.iter().any(is_excluded_from_model_context) {
            return;
        }
        let kept: Vec<agent_core::Message> = req
            .messages
            .iter()
            .filter(|m| !is_excluded_from_model_context(m))
            .cloned()
            .collect();
        req.messages = std::sync::Arc::new(kept);
    }
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

/// Whether `m` is a [`HOST_BASH_EXCLUDED_LABEL`]-tagged message — see that constant's own doc comment.
fn is_excluded_from_model_context(m: &agent_core::Message) -> bool {
    m.role == agent_core::Role::User
        && m.content.iter().any(|c| {
            matches!(c, agent_core::ContentBlock::Text { text, .. } if text.starts_with(HOST_BASH_EXCLUDED_LABEL))
        })
}

/// The tool registry after `--tools`/`--exclude-tools`/`--no-tools` filtering — shared by every
/// `build_agent` rebuild and by the host-level `bash` RPC command (see [`serve`]), so excluding `bash`
/// from the model's own tool set also disables the host command rather than leaving a side door open
/// around an operator's explicit restriction.
///
/// `image_auto_resize` is an explicit parameter, not read off `cfg` (pass 20, pi-parity fix): the only
/// call site where its value is actually load-bearing is `build_agent`'s own live-rebuildable registry
/// (see `set_image_auto_resize`), which passes the runtime-mutable `current_image_auto_resize` rather
/// than `cfg`'s frozen startup value; the other call sites here only ever check tool
/// presence/definitions, which don't depend on this flag either way, so they just pass
/// `cfg.image_auto_resize` straight through.
fn build_tools(cfg: &ServeConfig, image_auto_resize: bool) -> agent_core::ToolRegistry {
    // Task #34 (pi-parity fix): previously always called the plain `default_registry_with_prefix`,
    // hardcoding `image_auto_resize: true` regardless of `cfg.image_auto_resize` — `run`'s identical
    // `default_registry_with_prefix_and_image_auto_resize` call site (`main.rs::run_task`) is the one
    // this should have matched all along.
    //
    // `cfg.mcp_tools` (already-connected, namespaced `mcp__<server>__<tool>` entries — see
    // `tools::mcp`'s module doc comment) is merged in *before* `apply_filter` below, not after: an
    // operator's `--tools`/`--exclude-tools` allow/deny-list is meant to scope the *whole* tool set the
    // model sees, MCP-discovered tools included (e.g. excluding a misbehaving MCP tool by name), not
    // just the built-ins.
    let mut registry = tools::default_registry_with_config(&tools::ToolConfig {
        bash_timeout_ms: cfg.bash_timeout_ms,
        bash_shell_path: cfg.bash_shell_path.as_deref(),
        bash_command_prefix: cfg.bash_command_prefix.as_deref(),
        web_allow_private: cfg.web_allow_private,
        web_allow_hosts: &cfg.web_allow_hosts,
        web_timeout_ms: cfg.web_timeout_ms,
        image_auto_resize,
        mcp_tools: &cfg.mcp_tools,
        ..tools::ToolConfig::new()
    });
    tools::apply_filter(
        &mut registry,
        cfg.tools.as_deref(),
        cfg.exclude_tools.as_deref(),
        cfg.no_tools,
    );
    registry
}

/// The `--reasoning-effort` default (Fix 1, pi-parity gap) both `main.rs`'s plain `run` path and this
/// module's own `serve` startup fall back to when the operator passed neither an explicit
/// `--reasoning-effort` flag nor a stored `agent settings --default-reasoning-effort` override: pi's own
/// `DEFAULT_THINKING_LEVEL` ("medium", `packages/coding-agent/src/core/defaults.ts`) applies whenever
/// `model` supports reasoning/thinking at all — a bare `pi "task"` invocation runs with medium
/// adaptive-thinking effort on pi's own default model, not with thinking disabled outright. Previously
/// this crate left the effort `None` in that case, which — for a model whose thinking mechanism can be
/// explicitly disabled (`reasoning_disableable`) — wire-sends an explicit `{"type":"disabled"}"`/
/// `{"effort":"none"}` rather than picking any default depth at all.
///
/// `None` for a model with no thinking/reasoning mechanism whatsoever — see
/// [`agent_core::models::has_reasoning_mechanism`] — nothing to default there. Deliberately a
/// CLI/serve-layer default, not new `agent_core` plumbing: `agent_core::Agent::new`/
/// `with_reasoning_effort` are unchanged, still doing exactly nothing when the caller never calls the
/// latter — this only changes what `main.rs`/`serve` pass in when the operator didn't specify anything
/// themselves.
///
/// Calls agent-core's own [`has_reasoning_mechanism`](agent_core::models::has_reasoning_mechanism)
/// rather than reimplementing the check here: a prior version tested only `caps.thinking !=
/// ThinkingShape::None || caps.reasoning_effort`, narrower than agent-core's own internal gate by
/// exactly one arm (`caps.openai_reasoning_format != OpenAiReasoningFormat::Standard`) — so
/// Kimi-thinking and pre-5.2 GLM models, which signal their reasoning toggle only via that third arm,
/// silently fell through to "no default" here even though pi's own reference defaults them to medium
/// effort too.
pub fn default_reasoning_effort_for_model(model: &str) -> Option<agent_core::ReasoningEffort> {
    let caps = agent_core::capabilities(model);
    agent_core::models::has_reasoning_mechanism(&caps)
        .then_some(agent_core::ReasoningEffort::Medium)
}

/// A small, non-exhaustive list of model ids the [`capabilities`](agent_core::capabilities) table
/// recognizes, for a client's model picker. The gateway forwards any id verbatim, so this is a
/// convenience hint — not an allowlist; `set_model` accepts ids outside this list. Shared by `serve`'s
/// own `get_available_models`/`cycle_model` and `main`'s `list-models` CLI subcommand.
pub fn available_models() -> &'static [&'static str] {
    &[
        "claude-opus-4-8",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gpt-5",
        "gpt-5-mini",
        "gpt-4o",
        "gpt-4.1",
        "o3",
        "o4-mini",
    ]
}

/// One resolved entry in a `--models` scope: a model id plus an optional thinking level pinned via a
/// trailing `:<level>` suffix on the pattern that produced it. Mirrors pi's own `ScopedModel`
/// (`model-resolver.ts`), minus the `provider/id` canonical form pi's real multi-provider registry
/// carries — this crate has no such registry (`model_info`'s doc comment), so a glob only matches
/// against the bare id.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ScopedModel {
    id: String,
    thinking_level: Option<agent_core::ThinkingLevel>,
}

/// Parses `--models`'s comma-separated pattern list (already split by clap) into a resolved candidate
/// set — pi's own `resolveModelScopeWithDiagnostics`. Each pattern, independently:
///
/// - has a trailing `:<level>` suffix stripped off and remembered as that entry's pinned thinking
///   level (one of off/minimal/low/medium/high/xhigh — `agent_core::ThinkingLevel::parse`), applied
///   whenever `cycle_model` lands on it; a suffix that isn't a valid level is left alone and treated
///   as part of the id/pattern itself, same as pi's `parseModelPattern` recursion;
/// - if what's left contains a glob metacharacter (`*`, `?`, `[`), is matched case-insensitively
///   against every id in `catalog`, expanding to every match in `catalog`'s order — e.g. `claude-*`
///   or `*sonnet*` — each getting the same pinned level;
/// - otherwise is kept as a literal id verbatim, whether or not it's in `catalog` — matching
///   `available_models`'s own "hint, not an allowlist" contract, since the gateway forwards any id
///   verbatim regardless of what this process happens to know about.
///
/// A glob that matches nothing is dropped silently — pi's own resolver only ever warns, never
/// hard-fails, and cycling still works from whatever else resolved. A duplicate id keeps the first
/// occurrence's pin, same as pi's `modelsAreEqual` dedup.
///
/// Splits a trailing `:<level>` suffix (one of `agent_core::ThinkingLevel::parse`'s vocabulary) off
/// `pattern`, shared by [`resolve_model_scope`] (a `--models` entry) and [`resolve_model_id`] (a bare
/// `--model`/`set_model` id) — Fix 2 (pi-parity gap): `resolve_model_id` previously had no colon-suffix
/// handling at all, unlike pi's own `model-resolver.ts::parseModelPattern`, which applies it to
/// `--model`/`--provider` too (its own help text example: `pi --model sonnet:high`). A suffix that isn't
/// a valid level is left attached to `pattern` untouched — the whole string is then resolved (or kept
/// literal) as one id, same as if no colon were present at all.
fn split_thinking_level_suffix(pattern: &str) -> (&str, Option<agent_core::ThinkingLevel>) {
    match pattern.rfind(':') {
        Some(idx) => {
            let suffix = &pattern[idx + 1..];
            match agent_core::ThinkingLevel::parse(suffix) {
                Some(level) => (&pattern[..idx], Some(level)),
                None => (pattern, None),
            }
        }
        None => (pattern, None),
    }
}

fn resolve_model_scope(patterns: &[String], catalog: &[&str]) -> Vec<ScopedModel> {
    let mut resolved: Vec<ScopedModel> = Vec::new();
    for raw in patterns {
        let pattern = raw.trim();
        if pattern.is_empty() {
            continue;
        }
        let (base, thinking_level) = split_thinking_level_suffix(pattern);
        let is_glob = base.chars().any(|c| matches!(c, '*' | '?' | '['));
        if is_glob {
            let matcher = match globset::GlobBuilder::new(base)
                .case_insensitive(true)
                .build()
            {
                Ok(g) => g.compile_matcher(),
                Err(_) => continue,
            };
            for &id in catalog {
                if matcher.is_match(id) && !resolved.iter().any(|m| m.id == id) {
                    resolved.push(ScopedModel {
                        id: id.to_string(),
                        thinking_level,
                    });
                }
            }
        } else {
            // Fix 10 (pi-parity feature): a literal, non-glob pattern that doesn't exactly match a
            // catalog id now also gets the same partial/substring resolution `--model`/`set_model` get
            // — e.g. `--models sonnet` resolves to `claude-sonnet-4-5` instead of cycling to the
            // literal, almost-certainly-wrong string "sonnet". An ambiguous partial match can't fail
            // the whole `serve` startup the way an ambiguous `--model` does (this is a background
            // candidate-list build, not the one model actively in use) — it's warned about and kept
            // literal instead, same graceful-degrade this crate already applies to a glob matching
            // nothing.
            let resolved_id = match resolve_model_id_base(base, catalog) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("warning: --models {base:?}: {e}; using it literally");
                    base.to_string()
                }
            };
            if !resolved.iter().any(|m| m.id == resolved_id) {
                resolved.push(ScopedModel {
                    id: resolved_id,
                    thinking_level,
                });
            }
        }
    }
    resolved
}

/// Resolve a possibly-partial `--model`/`set_model` id against the known-model hint list
/// ([`available_models`]) — pi's own `model-resolver.ts` partial/substring matching, scoped down: this
/// crate has no per-provider catalog to rank un-dated aliases against dated snapshots with (pi's own
/// alias-preference rule), and — unlike pi's own `resolveCliModel`, which silently picks whichever
/// candidate sorts first — never silently guesses on an ambiguous match. Instead mirrors
/// `SessionRepo::find_path`'s own "ambiguous is a real error naming every candidate, not a guess"
/// philosophy (see `session_store.rs`), applied consistently to `--model`, `set_model`, and (via
/// `resolve_model_scope`, above) the literal entries in a `--models` scope.
///
/// - An exact, case-insensitive match against `catalog` resolves to the catalog's own canonical
///   spelling (so `--model Claude-Opus-4-8` still lands on `claude-opus-4-8`).
/// - Otherwise, every catalog id that *contains* `input` as a substring (case-insensitive) is a
///   candidate — e.g. `opus` matches `claude-opus-4-8`. Exactly one candidate resolves to it; more than
///   one is `Err`, naming every candidate so the caller can be specific instead of silently guessing
///   which one was meant (e.g. `gpt` matches all of `gpt-5`/`gpt-5-mini`/`gpt-4o`/`gpt-4.1`).
/// - No candidates at all returns `Ok(input)` unchanged — `available_models` is explicitly documented as
///   a non-exhaustive hint, not an allowlist, so an id genuinely outside it (a brand-new model, a
///   provider-specific id the gateway still understands) must still reach the gateway verbatim rather
///   than being rejected outright.
///
/// Fix 2 (pi-parity gap): `input` may also carry a trailing `:<level>` suffix (e.g. `sonnet:high`,
/// pi's own `--model <pattern>:<thinking-level>` shorthand) — split off via
/// [`split_thinking_level_suffix`] *before* resolution, so both `--model`'s own startup resolution and
/// the `set_model` RPC handler get it for free, the same way [`resolve_model_scope`]'s `--models`
/// entries already do. The returned level is the caller's to apply (setting the corresponding
/// reasoning effort); a suffix that isn't a valid level is left attached to the id itself, and resolved
/// (or kept literal) as one string, exactly as if no colon were present.
pub fn resolve_model_id(
    input: &str,
    catalog: &[&str],
) -> Result<(String, Option<agent_core::ThinkingLevel>), String> {
    let (base, thinking_level) = split_thinking_level_suffix(input.trim());
    resolve_model_id_base(base, catalog).map(|id| (id, thinking_level))
}

/// [`resolve_model_id`]'s resolution, without the colon-suffix split — the id-only lookup
/// [`resolve_model_scope`]'s own literal-pattern branch calls directly (it already extracted any
/// `:<level>` suffix itself, via the same [`split_thinking_level_suffix`] helper, so re-splitting here
/// would be a no-op at best and a double-strip of a legitimately colon-containing id at worst).
fn resolve_model_id_base(input: &str, catalog: &[&str]) -> Result<String, String> {
    let trimmed = input.trim();
    if let Some(exact) = catalog.iter().find(|m| m.eq_ignore_ascii_case(trimmed)) {
        return Ok((*exact).to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    let candidates: Vec<&str> = catalog
        .iter()
        .copied()
        .filter(|m| m.to_ascii_lowercase().contains(&lower))
        .collect();
    match candidates.as_slice() {
        [] => Ok(trimmed.to_string()),
        [one] => Ok((*one).to_string()),
        many => Err(format!(
            "{trimmed:?} matches more than one known model id: {} — pass one of these exactly, or a \
             full id outside this hint list to bypass resolution entirely",
            many.join(", ")
        )),
    }
}

/// A `get_available_models` entry: enough per-model capability info (from the same
/// [`agent_core::capabilities`] table every wire decision already consults) for a client to build a
/// capability-aware picker without shipping its own hardcoded model registry. pi: `Model<any>`
/// (`rpc-types.ts:143-146`) — `id`/`contextWindow`/`reasoning` carry the same *kind* of information
/// here under this codebase's own field-naming convention (snake_case, matching `get_state`/
/// `get_session_stats`); `provider` is a coarse wire-family hint (`agent_core::dialect::Dialect::
/// for_model`, "anthropic"/"openai"), not a real routing provider id — this crate has no such registry,
/// the gateway owns routing. Pricing (pi's `cost`) is deliberately omitted: gateway-owned, out of scope
/// for this crate (see `ServeConfig`'s module doc).
fn model_info(model: &str) -> Value {
    let caps = agent_core::capabilities(model);
    let provider = match agent_core::dialect::Dialect::for_model(model) {
        agent_core::dialect::Dialect::Anthropic => "anthropic",
        agent_core::dialect::Dialect::OpenAi | agent_core::dialect::Dialect::OpenAiResponses => {
            "openai"
        }
    };
    json!({
        "id": model,
        "provider": provider,
        "context_window": caps.context_window,
        "max_output": caps.max_output,
        // Shares `default_reasoning_effort_for_model`'s own `has_reasoning_mechanism` gate rather than a
        // separate `reasoning_effort || thinking != None` check — that narrower pair previously missed
        // the `openai_reasoning_format` toggle shapes (Kimi-thinking, pre-5.2 GLM), reporting `reasoning:
        // false` for a model that genuinely does support a client-steerable reasoning toggle.
        "reasoning": agent_core::models::has_reasoning_mechanism(&caps),
        "supports_vision": caps.supports_vision,
    })
}

/// [`model_info`] plus the thinking-level state a model *switch* also carries — the shared response
/// shape for `set_model`/`cycle_model`, so a client gets the same capability info
/// `get_available_models` already provides (context window, max output, reasoning/vision support)
/// instead of just a bare model id, matching pi's own `set_model`/`cycle_model` (`rpc-mode.ts`), which
/// embed the identical per-model object `get_available_models` uses. `model`/`reasoning_effort` are
/// kept as top-level fields (not folded into `model_info`'s own `id`) for wire-compatibility with a
/// client already reading this shape.
fn model_switch_response(model: &str, level: agent_core::ThinkingLevel) -> Value {
    let mut info = model_info(model);
    if let Value::Object(map) = &mut info {
        map.insert("model".to_string(), json!(model));
        map.insert("reasoning_effort".to_string(), json!(level.as_str()));
    }
    info
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
            agent_core::ContentBlock::Text { text, .. } => Some(text.as_str()),
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
                    agent_core::ContentBlock::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>()
        })
        .unwrap_or_default()
}

/// Expand an explicit `/skill:name` invocation first (its own prefix, so it can't collide with a
/// `/name` prompt template), then fall through to prompt-template expansion — a no-op on whichever
/// message reaches it unmatched. Shared by every command that queues raw user-authored text as a turn
/// or steering message (`prompt`, `steer`, `follow_up`, and `prompt`'s `streaming_behavior` variant) —
/// a message queued through `steer`/`follow_up` must get the same expansion a fresh `prompt` would, or
/// a `/skill:name`/`/name` invocation sent through one of those paths silently reaches the model
/// unexpanded instead of triggering the skill/template it names.
fn expand_message(
    message: &str,
    skills: &[crate::skills::Skill],
    prompt_templates: &[crate::prompts::PromptTemplate],
) -> String {
    let message = crate::skills::expand_if_skill_invocation(message, skills);
    crate::prompts::expand_if_slash(&message, prompt_templates)
}

/// Parse a `prompt`'s optional `images` array into base64 image sources. Each entry is
/// `{media_type, data}`; malformed entries are skipped.
fn parse_images(images: Option<&Value>) -> Vec<agent_core::ImageSource> {
    images
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|img| {
                    let media_type = img.get("media_type").and_then(Value::as_str)?;
                    let data = img.get("data").and_then(Value::as_str)?;
                    Some(agent_core::ImageSource::base64(media_type, data))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Token + step accounting for a session, including the prompt-cache reads/writes that make the
/// Anthropic cache breakpoints observable. Shared by the `prompt` response and `get_state`.
/// `model` is the session's *currently active* model, not necessarily the one that produced
/// `last_input_tokens` — a `set_model`/`cycle_model` mid-session means the context-window figure is
/// always reported against what a client would actually be constrained by on the *next* turn, matching
/// how a client would use this value (to decide whether compaction is coming soon), not as a historical
/// record of what the previous turn's provider reported.
fn session_stats(session: &Session, model: &str) -> Value {
    let context_window = agent_core::models::capabilities(model).context_window;
    // `last_input_tokens == 0` covers both "no turn has run yet" and pi's "immediately after
    // compaction, before a new reply" case (compaction never sets this field itself — only a real
    // provider response does) — both genuinely have no current-context figure to report, so `null`
    // rather than a misleading `0%`. Matches pi's `contextUsage` being `null` in the same situations.
    // Track L37 (pi-parity fix): `last_input_tokens` alone is only ever as fresh as the last turn's
    // own provider-reported usage — every message appended *since* then (this turn's own user prompt,
    // a mid-run tool round-trip) was missing from the reported figure entirely. `trailing_tokens` is
    // the exact same delta `should_compact`/`is_hard_overflow` already fold in before comparing against
    // the context window, so a client reading `context_usage` sees the same live-prompt size the
    // compaction trigger itself is actually watching, not a stale undercount.
    let context_usage = (session.last_input_tokens > 0).then(|| {
        let tokens = session
            .last_input_tokens
            .saturating_add(agent_core::compaction::trailing_tokens(session));
        json!({
            "tokens": tokens,
            "context_window": context_window,
            "percent": (tokens as f64 / context_window as f64 * 100.0),
        })
    });
    let breakdown = message_type_breakdown(session);
    let usage_totals = message_usage_totals(session);
    json!({
        "steps": session.steps,
        "input_tokens": usage_totals.input_tokens,
        "output_tokens": usage_totals.output_tokens,
        "cache_read_tokens": usage_totals.cache_read_tokens,
        "cache_write_tokens": usage_totals.cache_write_tokens,
        "cache_write_1h_tokens": usage_totals.cache_write_1h_tokens,
        "reasoning_tokens": usage_totals.reasoning_tokens,
        "last_input_tokens": session.last_input_tokens,
        "context_usage": context_usage,
        "user_messages": breakdown.user_messages,
        "assistant_messages": breakdown.assistant_messages,
        "tool_calls": breakdown.tool_calls,
        "tool_results": breakdown.tool_results,
        "total_messages": session.messages.len(),
    })
}

/// Cumulative token usage across every message currently in `session.messages` (Task #6, pi-parity
/// fix) — computed fresh on every `session_stats` call, the same way `message_type_breakdown`, just
/// below, already derives its counts fresh from `session.messages` rather than a running counter.
///
/// Previously `session_stats` read `Session::input_tokens`/`output_tokens`/... directly — those fields
/// are a *process-lifetime* running total (`Session::record_usage` only ever adds to them) that resets
/// to zero every time the process restarts, since `SessionStore::open` only restores `session.messages`
/// from disk, never those counters. A resumed session's `get_session_stats` (and the `prompt`
/// response/`get_state`, both of which share this same function) therefore reported totals covering
/// only activity since the *current* process started, not the session's full history — silently
/// resetting on every restart with no indication anything was lost.
///
/// Same caveat `message_type_breakdown` already carries: a compacted-away message's `usage` is gone
/// along with the rest of its content, so this reports what's still visible on the active path, not a
/// truly unbounded historical total — an accepted, pre-existing limitation of deriving these figures
/// from `session.messages` at all, not something this fix introduces.
struct MessageUsageTotals {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    cache_write_1h_tokens: u64,
    reasoning_tokens: u64,
}

fn message_usage_totals(session: &Session) -> MessageUsageTotals {
    let mut totals = MessageUsageTotals {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cache_write_1h_tokens: 0,
        reasoning_tokens: 0,
    };
    for usage in session.messages.iter().filter_map(|m| m.usage) {
        totals.input_tokens += u64::from(usage.input_tokens);
        totals.output_tokens += u64::from(usage.output_tokens);
        totals.cache_read_tokens += u64::from(usage.cache_read_tokens);
        totals.cache_write_tokens += u64::from(usage.cache_write_tokens);
        totals.cache_write_1h_tokens += u64::from(usage.cache_write_1h_tokens);
        totals.reasoning_tokens += u64::from(usage.reasoning_tokens);
    }
    totals
}

/// [`message_usage_totals`], reshaped into [`crate::export::UsageTotals`] — Fix 6 (pi-parity gap): the
/// standalone `export` subcommand (`main.rs`) has no live `Agent`/`Session` with running counters to
/// pull from (a freshly `SessionStore::open`ed session never restores those — see
/// `message_usage_totals`'s own doc comment for why), but it does have `sess.messages` right there,
/// from which this is fully reconstructable — no live agent needed. Previously that entry point passed
/// `usage: None` unconditionally, the one of the three export entry points (alongside `serve`'s
/// `export_html` RPC and `run --export`, both of which already report real totals from a live
/// `Session`'s running counters) that silently omitted the stats section's usage line entirely.
pub fn message_export_usage_totals(session: &Session) -> crate::export::UsageTotals {
    let totals = message_usage_totals(session);
    crate::export::UsageTotals {
        input_tokens: totals.input_tokens,
        output_tokens: totals.output_tokens,
        cache_read_tokens: totals.cache_read_tokens,
        cache_write_tokens: totals.cache_write_tokens,
    }
}

/// pi's `userMessages`/`assistantMessages`/`toolCalls`/`toolResults` (`getSessionStats`) — a client
/// wanting these previously had to fetch `get_messages` and count client-side.
///
/// Adapted, not ported 1:1: pi's message model has a dedicated `toolResult` role, separate from `user`,
/// so its `userMessages`/`toolResults` split cleanly on role alone. Ours carries a tool result as a
/// `ContentBlock::ToolResult` *within* a `User`-role message (Anthropic's own convention) — and a
/// steered mid-run text injection can land in that very same message alongside the tool results (see
/// `steering.rs`'s module doc: "folded onto the same tool-results user turn"). So a `User` message here
/// counts as a "user message" when it carries anything *other than* a tool result (real authored
/// content), and `tool_calls`/`tool_results` count individual content blocks rather than whole messages
/// — matching pi's own `toolCalls` definition, which already counts blocks within an assistant message,
/// not messages themselves.
struct MessageTypeBreakdown {
    user_messages: u64,
    assistant_messages: u64,
    tool_calls: u64,
    tool_results: u64,
}

fn message_type_breakdown(session: &Session) -> MessageTypeBreakdown {
    let mut breakdown = MessageTypeBreakdown {
        user_messages: 0,
        assistant_messages: 0,
        tool_calls: 0,
        tool_results: 0,
    };
    for m in session.messages.iter() {
        match m.role {
            agent_core::Role::User => {
                let has_non_tool_result = m
                    .content
                    .iter()
                    .any(|c| !matches!(c, agent_core::ContentBlock::ToolResult { .. }));
                if has_non_tool_result {
                    breakdown.user_messages += 1;
                }
                breakdown.tool_results += m
                    .content
                    .iter()
                    .filter(|c| matches!(c, agent_core::ContentBlock::ToolResult { .. }))
                    .count() as u64;
            }
            agent_core::Role::Assistant => {
                breakdown.assistant_messages += 1;
                breakdown.tool_calls += m
                    .content
                    .iter()
                    .filter(|c| matches!(c, agent_core::ContentBlock::ToolUse { .. }))
                    .count() as u64;
            }
            agent_core::Role::System => {}
        }
    }
    breakdown
}

/// Insert `session_id`/`session_file` into `m` — the on-disk identity every `get_state`/
/// `get_session_stats` response carries, idle or busy (Fix 8, pi-parity gap: `get_session_stats`
/// previously omitted both in its idle response, and its entire busy response, unlike `get_state`'s own
/// sibling arms, which already backfilled them from these same two sources). Shared so the two RPC
/// types can't drift out of shape with each other again.
fn insert_session_identity(m: &mut Map<String, Value>, persistence: &Persistence) {
    m.insert("session_id".into(), json!(persistence.session_id()));
    m.insert(
        "session_file".into(),
        json!(persistence.session_file().map(|p| p.display().to_string())),
    );
}

/// The runtime-mutable settings and queue depth `get_state` reports — pi's `get_state` carries the
/// same shape (`thinkingLevel`/`autoCompactionEnabled`/`steeringMode`/`followUpMode`/
/// `pendingMessageCount`), so a client can render current settings and "N queued" without a second
/// round trip. Answerable from the process's own mutable state, not `&Session`, so — like
/// `session_stats` — it's available even mid-run (see the `prompt` arm's read-only-command handling).
///
/// Fix 5 (pi-parity gap): `steer_queue`/`follow_up_queue` carry the queued lanes' actual message text
/// (`Steering::steer_texts`/`follow_up_texts`), not just `pending_messages`' bare count — previously a
/// client had no way to see *what* was queued short of guessing from its own prior `steer`/`follow_up`
/// calls, unlike pi's own `queue_update` event (`steering`/`followUp` string arrays). No unsolicited
/// push-time event is added alongside it (that gap is deliberate — see `steering.rs`'s own module doc
/// comment); instead every place this function's result already lands (`get_state`, and the `steer`/
/// `follow_up` handlers' own ack responses, below) picks the content up for free.
fn runtime_settings(
    current_level: agent_core::ThinkingLevel,
    current_auto_compaction: bool,
    current_auto_retry: bool,
    current_block_images: bool,
    current_image_auto_resize: bool,
    steering: &agent_core::Steering,
) -> Value {
    json!({
        "thinking_level": current_level.as_str(),
        "auto_compaction": current_auto_compaction,
        "auto_retry": current_auto_retry,
        // Pass 20 (pi-parity fix): reported here for the same reason `auto_compaction`/`auto_retry`
        // are — both are now live-toggleable (`set_block_images`/`set_image_auto_resize`), so `get_state`
        // needs to reflect whichever value is actually in effect, not just what `serve` started with.
        "block_images": current_block_images,
        "image_auto_resize": current_image_auto_resize,
        "steering_mode": match steering.steering_mode() {
            agent_core::QueueMode::OneAtATime => "one_at_a_time",
            agent_core::QueueMode::All => "all",
        },
        "follow_up_mode": match steering.follow_up_mode() {
            agent_core::QueueMode::OneAtATime => "one_at_a_time",
            agent_core::QueueMode::All => "all",
        },
        "pending_messages": steering.pending_count(),
        "steer_queue": steering.steer_texts(),
        "follow_up_queue": steering.follow_up_texts(),
    })
}

/// `{steer_queue, follow_up_queue}` alone — the same pair [`runtime_settings`] carries, factored out
/// so the `steer`/`follow_up` handlers' own ack responses (Fix 5, pi-parity gap) can attach it without
/// pulling in that function's wider signature (`thinking_level`/`auto_compaction`/… — irrelevant to a
/// bare queue push/ack).
fn queue_content(steering: &agent_core::Steering) -> Value {
    json!({
        "steer_queue": steering.steer_texts(),
        "follow_up_queue": steering.follow_up_texts(),
    })
}

/// Recover the exit code the host `bash` RPC command's terminal response reports (Fix 7, pi-parity
/// gap) — `tools::bash` never returns one structurally; a non-zero exit is folded into `result_text`
/// itself as a trailing `"Command exited with code <N>"` status line (`tools::bash::append_status`),
/// the only place it's recorded at all. `Some(0)` for an ordinary success: `tools::bash` itself treats
/// exit `0` and "no code reported" (`result.code: None`) as the identical `Ok` outcome, so neither is
/// distinguishable from the other here either. `None` for a cancelled/timed-out run — pi's own
/// `BashResult.exitCode` is `undefined` for both (see `bash_result_was_cancelled`), matching the
/// dedicated "cancelled" status text `tools::bash` appends instead of an exit-code one for either case.
fn bash_exit_code_from_status_line(result_text: &str, is_error: bool) -> Option<i64> {
    if !is_error {
        return Some(0);
    }
    let marker = "Command exited with code ";
    let start = result_text.rfind(marker)? + marker.len();
    result_text[start..]
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .filter(|digits| !digits.is_empty())
        .and_then(|digits| digits.parse().ok())
}

/// Whether the host `bash` RPC command's result represents a cancelled or timed-out run rather than an
/// ordinary completion (Fix 7, pi-parity gap) — pi's own `BashResult.cancelled` conflates both under one
/// signal (`executeBashWithOperations`'s `options?.signal?.aborted` fires for either). Recovered from
/// `tools::bash`'s own status-line text (`append_status`'s `"Command cancelled"`/`"Command timed out
/// after <N> seconds"`), or this RPC handler's own fallback synthetic `"cancelled"` error for a tool
/// that isn't internally cancellation-aware (see the `bash` handler's own `select!` comment on why, in
/// practice, `tools::bash` always wins that race first).
fn bash_result_was_cancelled(result_text: &str, is_error: bool) -> bool {
    is_error
        && (result_text.contains("Command cancelled")
            || result_text.contains("Command timed out after")
            || result_text == "cancelled")
}

/// A live mirror of [`Session`]'s token/step counters, updated from the event sink as a `prompt` runs —
/// so `get_state`/`get_session_stats` can answer with real progress from *inside* the busy-loop
/// (see the `prompt` arm), without the `&mut Session` the run itself holds exclusively for the turn's
/// duration. Seeded from the session's own values right before the run starts, then kept current by
/// mirroring exactly the events a client watching the stream already sees: `record_usage` matches
/// `Session::record_usage`'s accumulation, and `set_steps` takes `AgentEvent::TurnEnd`'s `step` field,
/// which is already `session.steps` post-increment at the moment that event fires. Lock-free (plain
/// atomics): the sink runs synchronously inside the run's own task, and a command handler reads it from
/// the control loop's task — never contended enough to need more than `Relaxed` ordering, since these
/// are independent counters with no cross-field invariant a reader depends on.
#[derive(Default)]
struct LiveStats {
    steps: AtomicU32,
    input_tokens: std::sync::atomic::AtomicU64,
    output_tokens: std::sync::atomic::AtomicU64,
    cache_read_tokens: std::sync::atomic::AtomicU64,
    cache_write_tokens: std::sync::atomic::AtomicU64,
    cache_write_1h_tokens: std::sync::atomic::AtomicU64,
    reasoning_tokens: std::sync::atomic::AtomicU64,
    last_input_tokens: AtomicU32,
    /// B-L1 pi-parity gap (fixed): pi's `agent.state.pendingToolCalls` (a live, in-process reactive
    /// set) has no RPC equivalent — a client watching a long tool-heavy turn had to reconstruct "which
    /// calls are still running" itself from the raw `tool_start`/`tool_end` event stream. Mirrored from
    /// those same events (see the `prompt` command's busy-loop sink), so `get_state`/
    /// `get_session_stats` can answer it directly mid-run, the same way every other field here already
    /// does. A `Mutex<BTreeSet>` rather than another atomic: ids are strings, and insert/remove need to
    /// be atomic *with each other*, not just individually.
    pending_tool_ids: std::sync::Mutex<std::collections::BTreeSet<String>>,
    /// The model's current `todo` list, mirrored off the `todo` tool's own `ToolProgress` events for
    /// exactly the same reason as `pending_tool_ids` above: `&session` is exclusively borrowed by the
    /// in-flight turn, so `get_todos` could otherwise only be answered while idle — and the client that
    /// most needs it is a phone attaching *mid-run*, which has already missed every `tool_progress`
    /// frame streamed before it connected. Seeded from the session (see [`current_todos`]) so a
    /// `get_todos` answered one event into a fresh turn still reports the plan the last turn left.
    todos: std::sync::Mutex<Option<Value>>,
}

impl LiveStats {
    /// Seed from a session's current cumulative totals, so a `get_state`/`get_session_stats` answered
    /// one event into a brand-new turn still reflects everything before it, not a reset-to-zero count.
    ///
    /// Task #6 (pi-parity fix): seeded from [`message_usage_totals`] — the same per-message sum
    /// `session_stats` itself now uses — rather than `Session::input_tokens`/`output_tokens`/... those
    /// fields are only ever a *process-lifetime* running total (`Session::record_usage`'s own doc
    /// comment), reset to zero on every restart since `SessionStore::open` never restores them from
    /// disk. Seeding from the stale counters here meant the mid-run "busy" snapshot a client polls via
    /// `get_state`/`get_session_stats` while a `prompt` is in flight (see the `prompt` arm's own
    /// busy-loop) reported only the *current process's* activity on a resumed session, same bug, same
    /// fix, different call site.
    fn from_session(session: &Session) -> Self {
        let totals = message_usage_totals(session);
        Self {
            steps: AtomicU32::new(session.steps),
            input_tokens: totals.input_tokens.into(),
            output_tokens: totals.output_tokens.into(),
            cache_read_tokens: totals.cache_read_tokens.into(),
            cache_write_tokens: totals.cache_write_tokens.into(),
            cache_write_1h_tokens: totals.cache_write_1h_tokens.into(),
            reasoning_tokens: totals.reasoning_tokens.into(),
            last_input_tokens: AtomicU32::new(session.last_input_tokens),
            pending_tool_ids: std::sync::Mutex::new(std::collections::BTreeSet::new()),
            todos: std::sync::Mutex::new(current_todos(session)),
        }
    }

    /// Record the list a `todo` call just committed (the tool's own `ToolProgress` payload).
    fn todos_updated(&self, todos: Value) {
        *self.todos.lock().unwrap_or_else(|e| e.into_inner()) = Some(todos);
    }

    fn todos(&self) -> Option<Value> {
        self.todos.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn tool_started(&self, id: String) {
        self.pending_tool_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id);
    }

    fn tool_ended(&self, id: &str) {
        self.pending_tool_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id);
    }

    /// Fold one turn's usage into the running totals — the same arithmetic as
    /// [`Session::record_usage`], applied to atomics instead of plain fields.
    fn record_usage(&self, usage: &agent_core::TokenUsage) {
        self.input_tokens
            .fetch_add(usage.input_tokens.into(), Ordering::Relaxed);
        self.output_tokens
            .fetch_add(usage.output_tokens.into(), Ordering::Relaxed);
        self.cache_read_tokens
            .fetch_add(usage.cache_read_tokens.into(), Ordering::Relaxed);
        self.cache_write_tokens
            .fetch_add(usage.cache_write_tokens.into(), Ordering::Relaxed);
        self.cache_write_1h_tokens
            .fetch_add(usage.cache_write_1h_tokens.into(), Ordering::Relaxed);
        self.reasoning_tokens
            .fetch_add(usage.reasoning_tokens.into(), Ordering::Relaxed);
        let last = usage
            .input_tokens
            .saturating_add(usage.cache_read_tokens)
            .saturating_add(usage.cache_write_tokens);
        self.last_input_tokens.store(last, Ordering::Relaxed);
    }

    fn set_steps(&self, step: u32) {
        self.steps.store(step, Ordering::Relaxed);
    }

    /// A `session_stats`-shaped snapshot of the current values.
    fn snapshot(&self) -> Value {
        json!({
            "steps": self.steps.load(Ordering::Relaxed),
            "input_tokens": self.input_tokens.load(Ordering::Relaxed),
            "output_tokens": self.output_tokens.load(Ordering::Relaxed),
            "cache_read_tokens": self.cache_read_tokens.load(Ordering::Relaxed),
            "cache_write_tokens": self.cache_write_tokens.load(Ordering::Relaxed),
            "cache_write_1h_tokens": self.cache_write_1h_tokens.load(Ordering::Relaxed),
            "reasoning_tokens": self.reasoning_tokens.load(Ordering::Relaxed),
            "last_input_tokens": self.last_input_tokens.load(Ordering::Relaxed),
            "pending_tool_ids": self
                .pending_tool_ids
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        })
    }
}

/// Whether a `list_sessions`/`list_all_sessions` scan's progress at `scanned`/`total` is worth putting
/// on the wire: the first file, the last, and roughly every 10% step in between — enough for a client's
/// "scanning…" indicator to move without a frame per individual file when there are thousands of them.
/// A pure function of `scanned`'s value, not of arrival order, so it stays deterministic even though
/// `list_with_progress`'s underlying scan runs in parallel across several files at once.
fn should_report_scan_progress(scanned: usize, total: usize) -> bool {
    scanned <= 1 || scanned >= total || scanned.is_multiple_of((total / 10).max(1))
}

/// The manual-code-paste channel for an in-flight `login`: `ServeLoginCallbacks::prompt_text` parks a
/// `oneshot::Sender` here and awaits its receiver; the `"submit_code"` dispatch arm (running
/// concurrently in the same process's main idle loop — see the `"login"` arm) takes it and sends the
/// pasted code, waking the parked `login`.
type PendingCodeSlot = Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<String>>>>;

/// Lock `m`, tolerating a poisoned mutex (a prior holder panicked mid-critical-section) by recovering
/// its last-written contents rather than propagating the poison — matches this workspace's
/// panic-free-production-code convention (no bare `.unwrap()` on a lock), and there's no invariant
/// here a partial write could actually violate (the guarded value is always a plain, independently
/// valid `Option`/replace-whole-value).
pub(crate) fn lock_ignoring_poison<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Tracks the one `login` allowed in flight at a time — see `pending_login`'s own declaration.
struct PendingLogin {
    cancel: agent_core::CancellationToken,
    pending_code: PendingCodeSlot,
}

/// Resets `pending_login` back to `None` when the detached login task's future ends, for any
/// reason — including a panic unwinding through it. The login task's own JoinHandle is discarded
/// (nothing awaits it), so without this guard a panic anywhere in `crate::oauth::login` or the
/// credential save would skip the task's normal end-of-body reset and leave `pending_login`
/// permanently `Some`: every subsequent `login` call would return "busy: a login is already in
/// flight" forever, with no way to recover short of restarting the whole `serve` process (even
/// `abort_login` can't help — it just cancels a token nothing is left alive to observe). A `Drop`
/// impl runs during unwinding, unlike a plain line of cleanup code placed after the panicking call,
/// so this holds regardless of where in the task body things go wrong.
struct PendingLoginGuard(Arc<std::sync::Mutex<Option<PendingLogin>>>);

impl Drop for PendingLoginGuard {
    fn drop(&mut self) {
        *lock_ignoring_poison(&self.0) = None;
    }
}

/// Outstanding approval questions, keyed by request id.
///
/// A **map**, not the single slot `pending_login` gets away with: the loop dispatches tool groups
/// `buffer_unordered`, so several `before_tool_call` hooks can await concurrently, and a `subagent`
/// fan-out multiplies that again. One `oneshot` per question.
type PendingApprovals =
    Arc<std::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalDecision>>>>;

/// Clears a pending approval's slot when its `request` call returns — normally, on timeout, on abort, or
/// through a panic. The `PendingLoginGuard` idea, keyed.
struct PendingApprovalGuard {
    pending: PendingApprovals,
    id: String,
}

impl Drop for PendingApprovalGuard {
    fn drop(&mut self) {
        lock_ignoring_poison(&self.pending).remove(&self.id);
    }
}

/// `serve`'s [`ApprovalGate`]: broadcast the question to every attached client, park a `oneshot`, and
/// let whichever client answers first resolve it — the same ack-now/respond-later shape
/// [`ServeLoginCallbacks`] uses for a pasted OAuth code, generalized to N concurrent questions and N
/// concurrent answerers.
///
/// Broadcasts through `out` (the fan-out) rather than the session's `out_tx` writer channel, because the
/// gate is constructed before that channel exists — and because `broadcast` *is* the multi-attach
/// primitive. A control frame may therefore interleave with streamed `event` frames; that is fine, and
/// the client correlates by `request_id` regardless.
pub(crate) struct ServeApprovalGate {
    out: SharedOutConn,
    pending: PendingApprovals,
    /// Width-1: the user is asked **one question at a time**, even when eight tool calls (or eight
    /// subagent children sharing this gate) block at once.
    ///
    /// Deliberately not `ToolExecutionMode::Sequential`: one sequential-requesting call reroutes the
    /// loop's *entire* batch through its interleaved dispatch path, which is a cross-cutting change to
    /// execution semantics in service of a UI concern. Approved calls still run concurrently afterward.
    serialize: tokio::sync::Mutex<()>,
    /// `None` waits forever. A finite default exists because `running` stays `true` for the whole
    /// prompt, so an unanswered question on a session whose clients all detached mid-wait would
    /// otherwise pin it indefinitely (an `abort` still frees it either way).
    timeout: Option<std::time::Duration>,
    seq: AtomicU64,
    session_id: String,
}

impl ServeApprovalGate {
    pub(crate) fn new(
        out: SharedOutConn,
        timeout: Option<std::time::Duration>,
        session_id: impl Into<String>,
    ) -> (Arc<Self>, PendingApprovals) {
        let pending: PendingApprovals = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let gate = Arc::new(Self {
            out,
            pending: pending.clone(),
            serialize: tokio::sync::Mutex::new(()),
            timeout,
            seq: AtomicU64::new(0),
            session_id: session_id.into(),
        });
        (gate, pending)
    }

    fn broadcast(&self, frame: OutFrame) {
        lock_ignoring_poison(&self.out).broadcast(frame);
    }

    fn no_client(&self) -> bool {
        lock_ignoring_poison(&self.out).is_empty()
    }
}

/// Resolve a parked approval. Returns `false` when the id is unknown or already answered — which is how
/// the *losing* client in a multi-attach race learns its answer arrived too late.
pub(crate) fn resolve_approval(
    pending: &PendingApprovals,
    request_id: &str,
    decision: ApprovalDecision,
) -> bool {
    let sender = lock_ignoring_poison(pending).remove(request_id);
    match sender {
        Some(tx) => tx.send(decision).is_ok(),
        None => false,
    }
}

#[async_trait::async_trait]
impl crate::approval::ApprovalGate for ServeApprovalGate {
    async fn request(
        &self,
        req: crate::approval::ApprovalRequest,
        cancel: &CancellationToken,
    ) -> Result<ApprovalDecision, ApprovalError> {
        // One question at a time. An abort while queued behind another question must not wait for it.
        let _one_at_a_time = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(ApprovalError::Cancelled),
            guard = self.serialize.lock() => guard,
        };
        if cancel.is_cancelled() {
            return Err(ApprovalError::Cancelled);
        }
        // Fail closed rather than hang: a gate that silently executes privileged operations with nobody
        // watching is not a gate, and a detached background session has nobody watching.
        if self.no_client() {
            return Err(ApprovalError::NoClient);
        }

        let request_id = format!(
            "{}:{}",
            self.session_id,
            self.seq.fetch_add(1, Ordering::Relaxed)
        );
        let (tx, rx) = tokio::sync::oneshot::channel();
        lock_ignoring_poison(&self.pending).insert(request_id.clone(), tx);
        let _slot = PendingApprovalGuard {
            pending: self.pending.clone(),
            id: request_id.clone(),
        };

        self.broadcast(approval_request_frame(&request_id, &req));

        // `biased`, so a photo-finish resolves the only way that is defensible in both directions: an
        // abort outranks a human's answer (never run a tool after the run was cancelled), and a human's
        // answer outranks the timeout (they *did* answer, and `resolve_approval` already told them it was
        // accepted). Left unbiased, `select!` picks at random and a client could be told `accepted:true`
        // for a call the timeout then denied.
        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(ApprovalError::Cancelled),
            answer = rx => answer.map_err(|_| ApprovalError::Cancelled),
            _ = sleep_opt(self.timeout) => Err(ApprovalError::TimedOut),
        };
        // Every other attached client is still showing this question. Tell them it's answered.
        self.broadcast(approval_resolved_frame(&request_id, outcome));
        outcome
    }
}

/// `sleep(d)`, or a future that never resolves when `d` is `None`.
async fn sleep_opt(duration: Option<std::time::Duration>) {
    match duration {
        Some(d) => tokio::time::sleep(d).await,
        None => std::future::pending().await,
    }
}

/// `{type:"approval_request", …}` — a tool call waiting on a human. Broadcast to every attached client.
fn approval_request_frame(request_id: &str, req: &crate::approval::ApprovalRequest) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("approval_request"));
    m.insert("request_id".into(), json!(request_id));
    m.insert("tool".into(), json!(req.tool));
    m.insert("summary".into(), req.summary.clone());
    m.insert("scope_key".into(), json!(req.scope_key));
    m.insert("origin".into(), req.origin.to_json());
    m.insert(
        "options".into(),
        json!(["allow_once", "allow_session", "deny_once", "deny_session"]),
    );
    Value::Object(m).into()
}

/// `{type:"approval_resolved", …}` — this question is settled, however it settled.
///
/// No OAuth analogue: a login is a single-client RPC, but an approval fans out to every attached client
/// and only one of them answers. Without this the others keep a dead prompt on screen forever.
fn approval_resolved_frame(
    request_id: &str,
    outcome: Result<ApprovalDecision, ApprovalError>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("approval_resolved"));
    m.insert("request_id".into(), json!(request_id));
    let (decision, reason) = match outcome {
        Ok(d) if d.allow => ("allow", None),
        Ok(_) => ("deny", None),
        Err(ApprovalError::TimedOut) => ("deny", Some("timed_out")),
        Err(ApprovalError::Cancelled) => ("deny", Some("cancelled")),
        Err(ApprovalError::NoClient) => ("deny", Some("no_client")),
    };
    m.insert("decision".into(), json!(decision));
    if let Ok(d) = outcome {
        m.insert(
            "scope".into(),
            json!(match d.scope {
                ApprovalScope::Once => "once",
                ApprovalScope::Session => "session",
            }),
        );
    }
    if let Some(reason) = reason {
        m.insert("reason".into(), json!(reason));
    }
    Value::Object(m).into()
}

/// The `approve` command, shared by the busy and idle dispatch loops.
///
/// `accepted` is `false` when the `request_id` names no outstanding question — it was already answered
/// by another attached client, it timed out, or the run was aborted. That is not an error: it is the
/// answer a client races and loses.
fn handle_approve(id: Option<String>, cmd: &Value, pending: Option<&PendingApprovals>) -> OutFrame {
    let Some(request_id) = cmd.get("request_id").and_then(Value::as_str) else {
        return response(id, "approve", false, None, Some("missing `request_id`"));
    };
    let decision = match parse_approval_decision(cmd) {
        Ok(d) => d,
        Err(e) => return response(id, "approve", false, None, Some(&e)),
    };
    let Some(pending) = pending else {
        return response(
            id,
            "approve",
            false,
            None,
            Some("this session has no approval gate (see `--approve`)"),
        );
    };
    let accepted = resolve_approval(pending, request_id, decision);
    response(
        id,
        "approve",
        true,
        Some(json!({ "accepted": accepted })),
        None,
    )
}

/// Parse an inbound `approve` command's `decision`/`scope` fields.
fn parse_approval_decision(cmd: &Value) -> Result<ApprovalDecision, String> {
    let allow = match cmd.get("decision").and_then(Value::as_str) {
        Some("allow") => true,
        Some("deny") => false,
        _ => return Err("`decision` must be \"allow\" or \"deny\"".to_string()),
    };
    let scope = match cmd.get("scope").and_then(Value::as_str) {
        None | Some("once") => ApprovalScope::Once,
        Some("session") => ApprovalScope::Session,
        Some(other) => {
            return Err(format!(
                "`scope` must be \"once\" or \"session\" (got {other:?})"
            ));
        }
    };
    Ok(ApprovalDecision { allow, scope })
}

/// Drives an in-flight `login` RPC command: pushes `login_progress` frames for whatever the flow
/// needs to show (a URL, a device code, narration), and for the manual-code-paste path, parks on
/// `pending_code` until a `"submit_code"` command (processed by the same main loop, concurrently —
/// see the `"login"` dispatch arm) wakes it with the pasted value.
struct ServeLoginCallbacks {
    out_tx: mpsc::UnboundedSender<OutFrame>,
    id: Option<String>,
    provider: crate::oauth::OAuthProviderId,
    pending_code: PendingCodeSlot,
}

#[async_trait::async_trait]
impl crate::oauth::LoginCallbacks for ServeLoginCallbacks {
    async fn show_auth_url(&self, url: &str, _instructions: Option<&str>) {
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "browser",
            Some(url),
            None,
            None,
            None,
            None,
        ));
    }

    async fn show_device_code(&self, info: &crate::oauth::DeviceCodeInfo) {
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "device_code",
            None,
            Some(&info.user_code),
            Some(&info.verification_uri),
            info.expires_in.map(|d| d.as_secs()),
            None,
        ));
    }

    async fn progress(&self, message: &str) {
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "progress",
            None,
            None,
            None,
            None,
            Some(message),
        ));
    }

    async fn prompt_text(
        &self,
        _prompt: &crate::oauth::TextPrompt<'_>,
    ) -> std::result::Result<String, crate::oauth::OAuthError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        *lock_ignoring_poison(&self.pending_code) = Some(tx);
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "manual_code",
            None,
            None,
            None,
            None,
            None,
        ));
        rx.await
            .map_err(|_| crate::oauth::OAuthError::LoginCancelled)
    }

    async fn select(
        &self,
        prompt: &crate::oauth::SelectPrompt<'_>,
    ) -> std::result::Result<Option<String>, crate::oauth::OAuthError> {
        // Headless: always take the first (recommended) option rather than interactively prompting —
        // matches this trait's own documented contract for a non-interactive caller.
        Ok(prompt.options.first().map(|o| o.id.clone()))
    }
}

/// A frame queued for the single stdout writer task (see its own comment, above). Every RPC
/// `response`/`ack`/progress frame below is low-frequency and already built as a shallow `Map` of
/// scalar `json!` leaves — for those, `Value` costs nothing extra: the writer's own `serde_json::
/// to_string` is the only serialize they ever pay. [`event_frame`] is the one high-frequency producer
/// (once per streamed model delta, far more often than any RPC frame) and carries its own pre-built
/// `String` instead, serialized straight from the `AgentEvent` in one pass — see its doc comment for
/// why going through `Value` first would cost a second full serialize on the hottest path this process
/// has.
#[derive(Clone)]
pub(crate) enum OutFrame {
    Value(Value),
    Raw(String),
}

impl From<Value> for OutFrame {
    fn from(v: Value) -> Self {
        OutFrame::Value(v)
    }
}

/// The set of connections currently attached to a session, each an unbounded sink the session's single
/// writer task **broadcasts** every [`OutFrame`] to — so a phone and a TUI (or any N of the user's own
/// devices) on one session all render the same live stream, in the same order. Empty ⇒ detached, frames
/// dropped (a reconnecting client replays committed state via `get_messages {since}` — see
/// [`crate::serve_ws`]). The supervisor [`add`](OutFanout::add)s a sink on attach and
/// [`remove`](OutFanout::remove)s it on disconnect. An `std::sync::Mutex` (not tokio's) is deliberate:
/// it is only ever held across non-`await` `send`s into unbounded channels.
#[derive(Default)]
pub(crate) struct OutFanout {
    next_id: u64,
    sinks: Vec<(u64, mpsc::UnboundedSender<OutFrame>)>,
}

impl OutFanout {
    /// Register a connection's sink; returns an id to [`remove`](Self::remove) it by on disconnect.
    pub(crate) fn add(&mut self, tx: mpsc::UnboundedSender<OutFrame>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.sinks.push((id, tx));
        id
    }

    /// Drop the sink registered under `id` (a disconnected connection). Idempotent.
    pub(crate) fn remove(&mut self, id: u64) {
        self.sinks.retain(|(i, _)| *i != id);
    }

    /// Whether no client is currently attached. The stdio transport registers exactly one sink for the
    /// life of the process, so this is only ever `false` there; a WebSocket/UDS session whose clients
    /// have all detached reports `true`. [`crate::approval`] checks this before posing a question that
    /// nobody would ever see.
    pub(crate) fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }

    /// Send `frame` to every attached sink, pruning any whose receiver has gone away. The single-sink
    /// case (one connection — the common case) moves the frame with **no clone**, preserving the hot
    /// event path's zero-copy behavior; only 2+ attached connections pay a per-sink clone.
    pub(crate) fn broadcast(&mut self, frame: OutFrame) {
        match self.sinks.as_slice() {
            [] => {}
            [(_, tx)] => {
                if tx.send(frame).is_err() {
                    self.sinks.clear();
                }
            }
            _ => self.sinks.retain(|(_, tx)| tx.send(frame.clone()).is_ok()),
        }
    }
}

/// Shared handle to a session's attached-connection set (see [`OutFanout`]).
pub(crate) type SharedOutConn = Arc<std::sync::Mutex<OutFanout>>;

/// Serialize one [`OutFrame`] to its final JSON line (no trailing newline). `Raw` (the hot
/// `event`-frame path) is already the final text; only `Value` pays a `serde_json::to_string`. Returns
/// `None` only if a frame we built ourselves fails to serialize — a bug, skipped rather than tearing
/// down the stream. Shared by both transports so stdio and WebSocket emit byte-identical frame text.
pub(crate) fn frame_to_line(frame: OutFrame) -> Option<String> {
    match frame {
        OutFrame::Raw(line) => Some(line),
        OutFrame::Value(v) => match serde_json::to_string(&v) {
            Ok(line) => Some(line),
            Err(e) => {
                eprintln!("serve: failed to serialize output frame: {e}");
                None
            }
        },
    }
}

/// Build a `login_progress` frame — an unsolicited push update for an in-flight `login`, correlated
/// to the eventual terminal `response` via the same `id`. `step` is `"browser"` (open `url`),
/// `"device_code"` (show `user_code`/`verification_uri`), `"manual_code"` (the local callback
/// listener couldn't complete — paste a code via `submit_code`), or `"progress"` (a bare narration
/// `message`).
#[allow(clippy::too_many_arguments)]
fn login_progress_frame(
    id: Option<String>,
    provider: &str,
    step: &str,
    url: Option<&str>,
    user_code: Option<&str>,
    verification_uri: Option<&str>,
    expires_in_secs: Option<u64>,
    message: Option<&str>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("login_progress"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("login"));
    m.insert("provider".into(), json!(provider));
    m.insert("step".into(), json!(step));
    if let Some(url) = url {
        m.insert("url".into(), json!(url));
    }
    if let Some(user_code) = user_code {
        m.insert("user_code".into(), json!(user_code));
    }
    if let Some(verification_uri) = verification_uri {
        m.insert("verification_uri".into(), json!(verification_uri));
    }
    if let Some(expires_in) = expires_in_secs {
        m.insert("expires_in".into(), json!(expires_in));
    }
    if let Some(message) = message {
        m.insert("message".into(), json!(message));
    }
    Value::Object(m).into()
}

/// Build a `list_progress` frame — an unsolicited progress update for an in-flight `list_sessions`/
/// `list_all_sessions` scan, correlated to the eventual `response` frame via the same request `id`.
fn list_progress_frame(
    id: Option<String>,
    command: &str,
    scanned: usize,
    total: usize,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("list_progress"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    m.insert("scanned".into(), json!(scanned));
    m.insert("total".into(), json!(total));
    Value::Object(m).into()
}

/// Build an `auto_retry_start` frame — an unsolicited notice that a `prompt`'s run failed with what
/// looks like a transient error and is about to be automatically retried, correlated to the eventual
/// `response` frame via the same request `id`. Sent once per attempt, immediately before the backoff
/// sleep (not after — a client watching for retry activity shouldn't have to infer it from a gap).
/// Named `auto_retry_start` (not the bare `auto_retry` this crate used before) to match pi's own wire
/// discriminator (`agent-session.ts`'s `AgentSessionEvent`), the counterpart to `auto_retry_end` below.
fn auto_retry_frame(
    id: Option<String>,
    attempt: u32,
    max_attempts: u32,
    delay_ms: u64,
    error: &str,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("auto_retry_start"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("prompt"));
    m.insert("attempt".into(), json!(attempt));
    m.insert("max_attempts".into(), json!(max_attempts));
    m.insert("delay_ms".into(), json!(delay_ms));
    m.insert("error".into(), json!(error));
    Value::Object(m).into()
}

/// Build an `auto_retry_end` frame — the terminal notice for a whole-run retry sequence that made at
/// least one attempt: either a later attempt succeeded (`success: true`), or the sequence gave up
/// (`success: false`, `final_error` set) — including the backoff itself being interrupted by
/// `abort`/`abort_retry` (`final_error: "retry cancelled"`). Mirrors pi's own `auto_retry_end` event.
/// Never sent when no retry ever started (a first-attempt success, or a failure that was never
/// retryable in the first place, needs no "the retries ended" notice — nothing began to end).
fn auto_retry_end_frame(
    id: Option<String>,
    success: bool,
    attempt: u32,
    final_error: Option<&str>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("auto_retry_end"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("prompt"));
    m.insert("success".into(), json!(success));
    m.insert("attempt".into(), json!(attempt));
    if let Some(err) = final_error {
        m.insert("final_error".into(), json!(err));
    }
    Value::Object(m).into()
}

/// Build a `session_info_changed` frame — an unsolicited push notice that the session's title changed,
/// correlated to the triggering `set_session_name` request via the same `id`. pi's own
/// `session_info_changed` event (`rpc-mode.ts:632-639`) lets a client learn the final *sanitized* title
/// without a follow-up `get_state`; `title` is `None` when the sanitized result was empty (a caller can
/// explicitly clear a title — see `sanitize_title`/`title_or_clear`).
fn session_info_changed_frame(id: Option<String>, title: Option<String>) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("session_info_changed"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("set_session_name"));
    m.insert("title".into(), json!(title));
    Value::Object(m).into()
}

/// Filter `nodes` (the full, chronologically ordered [`crate::session_store::TreeNode`] list — see
/// `SessionStore::tree`'s own doc comment: every node, every branch, sorted by `(timestamp, id)`) down
/// to only the entries appended after `since`, a tree id the caller already has. Task #48 (pi-parity
/// gap): pi's own `SessionManager.getEntries({since})` backs both its `get_tree` and its dedicated
/// `get_entries` RPC command (`rpc-types.ts:63`, `rpc-mode.ts:609-620`), letting a client fetch only
/// what's new across *every* entry type (message/branch_summary/compaction/custom — `tree()` already
/// folds all of these into one list, unlike `get_messages`'s own `since`, which only ever sees the
/// active path's plain LLM messages). Since `tree()` is already in append order, finding `since`'s own
/// position in that same list and returning everything after it is enough — no separate incremental
/// index needed. `Err` (not a silent full re-fetch) when `since` matches no node in `nodes` — the same
/// "surface the bug, don't mask it" contract `get_messages`'s own `since` already established, rather
/// than silently returning everything a naive fallback would.
fn nodes_since(
    mut nodes: Vec<crate::session_store::TreeNode>,
    since: &str,
) -> Result<Vec<crate::session_store::TreeNode>, String> {
    match nodes.iter().position(|n| n.id == since) {
        Some(idx) => Ok(nodes.split_off(idx + 1)),
        None => Err(format!("no entry with id {since} in this session")),
    }
}

/// Build a `response` frame.
fn response(
    id: Option<String>,
    command: &str,
    success: bool,
    data: Option<Value>,
    error: Option<&str>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("response"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    m.insert("success".into(), json!(success));
    if let Some(d) = data {
        m.insert("data".into(), d);
    }
    if let Some(e) = error {
        m.insert("error".into(), json!(e));
    }
    Value::Object(m).into()
}

/// A lightweight acknowledgement frame, emitted the moment a `prompt` is queued — before the model
/// turn(s) actually run — so a client can distinguish "received and starting" from the eventual
/// terminal `response` (which may be seconds away on a long tool-heavy run).
fn ack(id: Option<String>, command: &str) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("ack"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    Value::Object(m).into()
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
fn event_frame(ev: AgentEvent) -> Option<OutFrame> {
    #[derive(serde::Serialize)]
    struct EventFrame<'a> {
        r#type: &'static str,
        event: &'a AgentEvent,
    }
    let line = serde_json::to_string(&EventFrame {
        r#type: "event",
        event: &ev,
    })
    .inspect_err(|e| eprintln!("serve: failed to serialize agent event: {e}"))
    .ok()?;
    Some(OutFrame::Raw(line))
}

/// Write one newline-delimited frame to stdout and flush it.
/// `line` must already carry its own trailing `\n` — folded into the one caller's buffer rather than a
/// second `write_all` here, so a frame goes out as a single write instead of two.
async fn write_frame(out: &mut tokio::io::Stdout, line: &str) -> std::io::Result<()> {
    out.write_all(line.as_bytes()).await?;
    out.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_login_guard_resets_the_slot_even_when_the_task_panics() {
        // Regression: the detached `login` task's `JoinHandle` is discarded (see the `"login"`
        // dispatch arm), so nothing else can react to a panic inside it — only a `Drop`-based guard
        // can guarantee `pending_login` gets reset, closing the "every subsequent login hangs behind
        // 'busy' forever" failure mode. Exercises `PendingLoginGuard` directly rather than the real
        // `oauth::login` flow, since forcing a genuine panic through that path would need a fake
        // provider seam this module doesn't otherwise need.
        let pending_login: Arc<std::sync::Mutex<Option<PendingLogin>>> =
            Arc::new(std::sync::Mutex::new(Some(PendingLogin {
                cancel: CancellationToken::new(),
                pending_code: Arc::new(std::sync::Mutex::new(None)),
            })));
        let slot = pending_login.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = PendingLoginGuard(slot.clone());
            panic!("simulated failure inside the login task");
        }));
        assert!(
            result.is_err(),
            "the simulated panic should propagate out of catch_unwind"
        );
        assert!(
            lock_ignoring_poison(&pending_login).is_none(),
            "pending_login must be cleared even though the guarded scope panicked"
        );
    }

    #[test]
    fn session_stats_reports_no_context_usage_before_any_turn_has_run() {
        // pi: agent-session-stats.test.ts — `contextUsage` is `null` when nothing has run yet (and,
        // by the same field, immediately after a compaction that hasn't been followed by a real reply
        // — compaction never sets `last_input_tokens` itself, only a real provider response does).
        let session = Session::new();
        let stats = session_stats(&session, "claude-opus-4-8");
        assert_eq!(stats["context_usage"], Value::Null, "got: {stats}");
    }

    #[test]
    fn session_stats_reports_the_message_type_breakdown() {
        // pi-parity fix (M14): `get_session_stats` had no `userMessages`/`assistantMessages`/
        // `toolCalls`/`toolResults`/`totalMessages` at all — a client wanting these had to fetch
        // `get_messages` and count client-side.
        let mut session = Session::new();
        session.user("first request");
        session.push(agent_core::Message::assistant(vec![
            agent_core::ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "a.rs" }),
                thought_signature: None,
            },
            agent_core::ContentBlock::ToolUse {
                id: "2".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "b.rs" }),
                thought_signature: None,
            },
        ]));
        // A pure tool-results turn — no user-authored content of its own.
        session.push(agent_core::Message {
            role: agent_core::Role::User,
            content: vec![
                agent_core::ContentBlock::ToolResult {
                    tool_use_id: "1".into(),
                    content: "contents of a.rs".into(),
                    is_error: false,
                    images: vec![],
                },
                agent_core::ContentBlock::ToolResult {
                    tool_use_id: "2".into(),
                    content: "contents of b.rs".into(),
                    is_error: false,
                    images: vec![],
                },
            ],
            model_id: None,
            error_message: None,
            aborted: false,
            usage: None,
            stop_reason: None,
        });
        session.push(agent_core::Message::assistant(vec![
            agent_core::ContentBlock::text("done reading both files"),
        ]));
        session.user("second request");

        let stats = session_stats(&session, "claude-opus-4-8");
        assert_eq!(stats["total_messages"], 5, "got: {stats}");
        assert_eq!(
            stats["user_messages"], 2,
            "the pure tool-results turn must not count as a user message: {stats}"
        );
        assert_eq!(stats["assistant_messages"], 2, "got: {stats}");
        assert_eq!(stats["tool_calls"], 2, "got: {stats}");
        assert_eq!(stats["tool_results"], 2, "got: {stats}");
    }

    #[test]
    fn session_stats_a_steered_tool_results_turn_still_counts_as_a_user_message() {
        // The one case a `User` message mixes a tool result with genuine authored content: a steer
        // injected mid-run lands on the same turn as that turn's own tool results (see `steering.rs`'s
        // module doc). That message must count as a user message — it does carry real content.
        let mut session = Session::new();
        session.push(agent_core::Message {
            role: agent_core::Role::User,
            content: vec![
                agent_core::ContentBlock::ToolResult {
                    tool_use_id: "1".into(),
                    content: "result".into(),
                    is_error: false,
                    images: vec![],
                },
                agent_core::ContentBlock::text("actually, also check the tests"),
            ],
            model_id: None,
            error_message: None,
            aborted: false,
            usage: None,
            stop_reason: None,
        });

        let stats = session_stats(&session, "claude-opus-4-8");
        assert_eq!(stats["user_messages"], 1, "got: {stats}");
        assert_eq!(stats["tool_results"], 1, "got: {stats}");
    }

    #[test]
    fn session_stats_reports_context_usage_against_the_currently_active_model() {
        let mut session = Session::new();
        session.last_input_tokens = 50_000;
        let stats = session_stats(&session, "claude-opus-4-8");
        let usage = &stats["context_usage"];
        assert_eq!(usage["tokens"], 50_000);
        let context_window = agent_core::models::capabilities("claude-opus-4-8").context_window;
        assert_eq!(usage["context_window"], context_window);
        let expected_percent = 50_000.0 / context_window as f64 * 100.0;
        assert!(
            (usage["percent"].as_f64().unwrap() - expected_percent).abs() < f64::EPSILON,
            "got: {usage}"
        );
    }

    #[test]
    fn session_stats_context_usage_folds_in_trailing_tokens_since_the_last_usage_snapshot() {
        // Track L37 (pi-parity fix): `context_usage` used to report `last_input_tokens` alone, which
        // is only ever as fresh as the last turn's own provider-reported usage — every message
        // appended *since* (this turn's own prompt, a mid-run tool round-trip) was invisible to a
        // client reading this field, even though `should_compact`/`is_hard_overflow` already fold the
        // same `agent_core::compaction::trailing_tokens` delta in before comparing against the window.
        let mut session = Session::new();
        session.last_input_tokens = 50_000;
        session.last_usage_message_count = session.messages.len(); // snapshot taken with no messages yet
        let stats_before = session_stats(&session, "claude-opus-4-8");
        assert_eq!(
            stats_before["context_usage"]["tokens"], 50_000,
            "no trailing messages yet: {stats_before}"
        );

        // ~100 estimated tokens' worth of message appended since the snapshot (400 chars / 4).
        session.push(agent_core::Message::user("x".repeat(400)));
        let stats_after = session_stats(&session, "claude-opus-4-8");
        let tokens_after = stats_after["context_usage"]["tokens"].as_u64().unwrap();
        assert!(
            tokens_after > 50_000,
            "trailing tokens since the last usage snapshot must be folded into `context_usage`: {stats_after}"
        );
        let context_window = agent_core::models::capabilities("claude-opus-4-8").context_window;
        let expected_percent = tokens_after as f64 / context_window as f64 * 100.0;
        assert!(
            (stats_after["context_usage"]["percent"].as_f64().unwrap() - expected_percent).abs()
                < f64::EPSILON,
            "percent must be derived from the same trailing-inclusive figure, not just last_input_tokens: {stats_after}"
        );
    }

    #[test]
    fn session_stats_context_window_reflects_a_model_switch_not_the_original_model() {
        // The window reported is always for the model the *next* turn would actually run against —
        // a session that ran under one model and then switched (`set_model`/`cycle_model`) must report
        // the new model's window, not the one that produced `last_input_tokens`.
        let mut session = Session::new();
        session.last_input_tokens = 10_000;
        let stats = session_stats(&session, "gpt-5-mini");
        let expected_window = agent_core::models::capabilities("gpt-5-mini").context_window;
        assert_eq!(stats["context_usage"]["context_window"], expected_window);
    }

    #[test]
    fn session_stats_sums_usage_from_message_history_not_the_process_lifetime_counter() {
        // Task #6 (pi-parity fix): a resumed session's `Session::input_tokens`/`output_tokens`/...
        // reset to zero every process restart (`SessionStore::open` only restores `session.messages`,
        // never those running counters) — simulated here by setting them directly while leaving
        // `session.messages` carrying the real historical `usage` a persisted session would actually
        // have. `session_stats` must report the totals from the message history, not the
        // process-lifetime counters (which this test deliberately sets to an obviously-wrong sentinel
        // to prove they're ignored).
        let mut session = Session::new();
        session.user("go");
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("ok")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 100,
                    output_tokens: 20,
                    cache_read_tokens: 5,
                    cache_write_tokens: 3,
                    cache_write_1h_tokens: 1,
                    reasoning_tokens: 2,
                }),
        );
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("more")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 50,
                    output_tokens: 10,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    cache_write_1h_tokens: 0,
                    reasoning_tokens: 0,
                }),
        );
        // A "reset on restart" process-lifetime counter simulated as a wrong, stale value that must
        // not leak into the reported totals.
        session.input_tokens = 999_999;
        session.output_tokens = 999_999;

        let stats = session_stats(&session, "claude-opus-4-8");
        assert_eq!(stats["input_tokens"], 150, "got: {stats}");
        assert_eq!(stats["output_tokens"], 30, "got: {stats}");
        assert_eq!(stats["cache_read_tokens"], 5, "got: {stats}");
        assert_eq!(stats["cache_write_tokens"], 3, "got: {stats}");
        assert_eq!(stats["cache_write_1h_tokens"], 1, "got: {stats}");
        assert_eq!(stats["reasoning_tokens"], 2, "got: {stats}");
    }

    #[test]
    fn session_stats_reports_full_history_after_a_simulated_restart() {
        // The exact scenario Task #6 names: persist a session with several messages carrying usage,
        // reload it fresh (simulating a process restart via a real `SessionStore`/`Persistence` round
        // trip, not just constructing a `Session` by hand), then confirm `session_stats` reflects the
        // full history rather than zero/current-process-only.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, mut session) =
            Persistence::open_repo(dir.path(), "/w", "claude-opus-4-8", None).unwrap();
        session.user("go");
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("ok")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 1_000,
                    output_tokens: 200,
                    ..Default::default()
                }),
        );
        persistence.persist(&session, None).unwrap();
        drop(persistence);
        drop(session);

        // "Restart": a fresh `Persistence::open_repo` — a brand-new `Session::new()` with none of the
        // in-memory counters the original process accumulated (matching a real process restart exactly,
        // since those counters never persist regardless).
        let (_restarted, reloaded) =
            Persistence::open_repo(dir.path(), "/w", "claude-opus-4-8", None).unwrap();
        assert_eq!(
            reloaded.input_tokens, 0,
            "sanity: the running counter itself stayed at zero"
        );

        let stats = session_stats(&reloaded, "claude-opus-4-8");
        assert_eq!(
            stats["input_tokens"], 1_000,
            "must reflect the full persisted history, not the fresh process's zeroed counter: {stats}"
        );
        assert_eq!(stats["output_tokens"], 200, "got: {stats}");
    }

    #[test]
    fn live_stats_from_session_seeds_from_message_history_not_the_process_lifetime_counter() {
        // Same Task #6 fix, for the mid-run "busy" snapshot path (`LiveStats::from_session`, consulted
        // by `get_state`/`get_session_stats` while a `prompt` is in flight) — it must seed from the same
        // per-message totals `session_stats` uses, not the process-lifetime `Session` counters.
        let mut session = Session::new();
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("ok")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 42,
                    output_tokens: 7,
                    ..Default::default()
                }),
        );
        session.input_tokens = 999_999; // stale process-lifetime counter, must be ignored

        let live = LiveStats::from_session(&session);
        let snapshot = live.snapshot();
        assert_eq!(snapshot["input_tokens"], 42, "got: {snapshot}");
        assert_eq!(snapshot["output_tokens"], 7, "got: {snapshot}");
    }

    // Fix 1 (pi-parity bug): `default_project_trust` used to be checked *before* an explicit per-path
    // `TrustStore` entry, so an operator's specific `agent trust`/`agent untrust <path>` exception could
    // be silently overridden by a coarser blanket policy. `resolve_project_trust` is the shared
    // precedence both `run` and `serve` now consult, taking the already-resolved `Trust` lookup as a
    // plain parameter rather than reading the real trust store, so these tests don't need to sandbox any
    // filesystem/global state.

    #[test]
    fn resolve_project_trust_force_untrusted_wins_over_everything() {
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(!resolve_project_trust(
            true, // trust_project also set
            true, // force_untrusted
            Some(TrustPolicy::Always),
            Trust::Trusted,
            false,
        ));
    }

    #[test]
    fn resolve_project_trust_trust_project_wins_when_not_force_untrusted() {
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(resolve_project_trust(
            true,
            false,
            Some(TrustPolicy::Never),
            Trust::Untrusted,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_an_explicit_trusted_entry_wins_over_a_never_policy() {
        // The core Fix 1 regression: an operator's specific `agent trust <path>` exception must win
        // over a coarser `never` blanket default, not be overridden by it.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Never),
            Trust::Trusted,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_an_explicit_untrusted_entry_wins_over_an_always_policy() {
        // The mirror case: `agent untrust <path>` must win over a blanket `always` default too.
        // `has_gated_resources: true` — Task #35's own fast path (below) short-circuits to trusted
        // whenever there's nothing to gate at all, which would defeat the very precedence this test
        // means to isolate if it were `false` here instead.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(!resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Always),
            Trust::Untrusted,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_falls_back_to_the_blanket_policy_only_when_the_lookup_is_unknown() {
        // Both cases use `has_gated_resources: true` for the same reason as the test above — isolating
        // the `Unknown`-branch blanket-policy precedence from Task #35's separate "nothing to gate"
        // fast path, which is covered by its own dedicated test below.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Always),
            Trust::Unknown,
            true,
        ));
        assert!(!resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Never),
            Trust::Unknown,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_nothing_to_gate_wins_outright_over_the_trust_store_and_any_policy() {
        // Task #35 (pi-parity fix): pi's own `resolveProjectTrusted` runs the "nothing here for
        // project trust to actually gate" fast path *before* consulting the trust store at all — so it
        // wins even over an explicit per-path `Trust::Untrusted` entry or a blanket `never` policy, not
        // just the `Unknown`+no-policy fallback this crate previously limited it to. Currently inert
        // (no gated resource type hits this today), but this pins the correct precedence so a future
        // gated resource type added without re-checking this ordering doesn't silently regress it.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(
            resolve_project_trust(false, false, None, Trust::Untrusted, false),
            "an explicit untrusted entry must not matter when there's nothing to gate"
        );
        assert!(
            resolve_project_trust(
                false,
                false,
                Some(TrustPolicy::Never),
                Trust::Unknown,
                false
            ),
            "a blanket never policy must not matter either, for the same reason"
        );
    }

    #[test]
    fn resolve_project_trust_ask_or_no_policy_falls_back_to_has_gated_resources() {
        use crate::trust_store::Trust;
        // No gated resources at all: nothing meaningfully differs between trusted/untrusted, so this
        // crate's headless "ask" fallback treats it as trusted.
        assert!(resolve_project_trust(
            false,
            false,
            None,
            Trust::Unknown,
            false
        ));
        assert!(!resolve_project_trust(
            false,
            false,
            None,
            Trust::Unknown,
            true
        ));
    }

    // pi-parity fix (L10): `--models` only ever accepted flat literal ids — no glob against the known
    // catalog, no `pattern:<level>` suffix to pin a scoped entry's thinking depth. `resolve_model_scope`
    // is the resolver pi calls `resolveModelScopeWithDiagnostics`.

    #[test]
    fn resolve_model_scope_keeps_a_literal_id_verbatim_even_outside_the_catalog() {
        // `available_models` is a hint, not an allowlist (its own doc comment) — the gateway forwards
        // any id verbatim, so a literal the operator typed must survive resolution unchanged.
        let scoped = resolve_model_scope(&["totally-custom-id".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "totally-custom-id".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_fuzzy_resolves_a_literal_partial_match_against_the_catalog() {
        // Fix 10: a literal, non-glob `--models` entry that partially matches exactly one catalog id
        // now resolves to it, the same as `--model`/`set_model` — `--models sonnet` must cycle to
        // `claude-sonnet-4-5`, not the literal, almost-certainly-wrong string "sonnet".
        let scoped = resolve_model_scope(&["sonnet".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "claude-sonnet-4-5".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_an_ambiguous_literal_partial_match_warns_and_keeps_it_literal() {
        // Unlike `--model`, an ambiguous `--models` entry must not fail the whole `serve` startup —
        // this is a background candidate-list build, not the one model actively in use — so it falls
        // back to the literal string (with a warning), same graceful-degrade already applied to a glob
        // matching nothing.
        let scoped = resolve_model_scope(&["gpt".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_expands_a_glob_against_the_catalog_in_catalog_order() {
        let scoped = resolve_model_scope(&["claude-*".to_string()], available_models());
        let ids: Vec<&str> = scoped.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["claude-opus-4-8", "claude-sonnet-4-5", "claude-haiku-4-5"]
        );
    }

    #[test]
    fn resolve_model_scope_glob_matching_is_case_insensitive() {
        let scoped = resolve_model_scope(&["CLAUDE-*".to_string()], available_models());
        assert_eq!(scoped.len(), 3, "got: {scoped:?}");
    }

    #[test]
    fn resolve_model_scope_a_glob_matching_nothing_is_dropped_silently() {
        let scoped = resolve_model_scope(
            &["no-such-provider/*".to_string(), "gpt-4o".to_string()],
            available_models(),
        );
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_pattern_level_suffix_pins_a_literal_entry() {
        let scoped = resolve_model_scope(&["gpt-4o:high".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o".to_string(),
                thinking_level: Some(agent_core::ThinkingLevel::High),
            }]
        );
    }

    #[test]
    fn resolve_model_scope_pattern_level_suffix_pins_every_glob_match_the_same_way() {
        let scoped = resolve_model_scope(&["claude-*:low".to_string()], available_models());
        assert!(
            scoped
                .iter()
                .all(|m| m.thinking_level == Some(agent_core::ThinkingLevel::Low)),
            "got: {scoped:?}"
        );
        assert_eq!(scoped.len(), 3);
    }

    #[test]
    fn resolve_model_scope_an_invalid_level_suffix_is_kept_as_part_of_the_literal_id() {
        // pi's own scope-mode fallback: a trailing `:bogus` isn't a real thinking level, so the whole
        // string — colon included — is treated as the pattern/id rather than silently truncated.
        let scoped = resolve_model_scope(&["gpt-4o:bogus".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o:bogus".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_dedupes_a_glob_overlapping_a_literal_keeping_the_first_occurrences_pin()
    {
        let scoped = resolve_model_scope(
            &["gpt-4o:high".to_string(), "gpt-*".to_string()],
            available_models(),
        );
        let gpt_4o = scoped.iter().find(|m| m.id == "gpt-4o").unwrap();
        assert_eq!(gpt_4o.thinking_level, Some(agent_core::ThinkingLevel::High));
        // The rest of the `gpt-*` glob still resolved past the one duplicate.
        assert!(scoped.iter().any(|m| m.id == "gpt-5"));
    }

    #[test]
    fn resolve_model_scope_trims_whitespace_and_skips_empty_patterns() {
        let scoped = resolve_model_scope(
            &[" gpt-4o ".to_string(), "".to_string(), "  ".to_string()],
            available_models(),
        );
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o".to_string(),
                thinking_level: None,
            }]
        );
    }

    // Fix 1 (pi-parity gap): a bare invocation with no `--reasoning-effort` previously left the
    // effort/level `None`/`Off`, wire-disabling reasoning outright for a model that can explicitly
    // disable it, instead of picking pi's own "medium" default.

    #[test]
    fn default_reasoning_effort_for_model_is_medium_on_the_built_in_default_model() {
        assert_eq!(
            default_reasoning_effort_for_model("claude-opus-4-8"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    #[test]
    fn default_reasoning_effort_for_model_is_medium_for_any_reasoning_capable_model() {
        // Budget-shape (extended thinking, not an OpenAI `reasoning_effort` toggle) counts too.
        assert_eq!(
            default_reasoning_effort_for_model("claude-test"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    #[test]
    fn default_reasoning_effort_for_model_is_none_for_a_model_with_no_reasoning_mechanism_at_all() {
        assert_eq!(
            default_reasoning_effort_for_model("claude-3-5-sonnet-20241022"),
            None
        );
        assert_eq!(default_reasoning_effort_for_model("gpt-4o"), None);
    }

    // Fix 1 (pi-parity remediation, Round 2): `default_reasoning_effort_for_model` previously
    // reimplemented `has_reasoning_mechanism`'s own check narrower than the original — missing the
    // `openai_reasoning_format` arm — so a Kimi-thinking or pre-5.2 GLM model (both toggle reasoning
    // only via that third-party wire format, with `reasoning_effort: false` and
    // `thinking: ThinkingShape::None`) silently got no default reasoning effort at all, even though pi's
    // own reference defaults every reasoning-capable model to medium effort.
    #[test]
    fn default_reasoning_effort_for_model_is_medium_for_a_kimi_thinking_model() {
        assert_eq!(
            default_reasoning_effort_for_model("kimi-k2-thinking"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    #[test]
    fn default_reasoning_effort_for_model_is_medium_for_a_pre_5_2_glm_model() {
        assert_eq!(
            default_reasoning_effort_for_model("glm-4.5"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    // Fix 1's sibling gap: `model_info`'s own `"reasoning"` field (surfaced via `get_available_models`/
    // `set_model`/`cycle_model`) reimplemented the identical narrower check independently, so it too
    // reported `false` for these families even though they have a genuine client-steerable toggle.
    #[test]
    fn model_info_reasoning_is_true_for_a_kimi_thinking_model() {
        assert_eq!(model_info("kimi-k2-thinking")["reasoning"], true);
    }

    #[test]
    fn model_info_reasoning_is_true_for_a_pre_5_2_glm_model() {
        assert_eq!(model_info("glm-4.5")["reasoning"], true);
    }

    // Fix 10 (pi-parity feature): `--model`/`set_model` previously forwarded any id verbatim with no
    // resolution at all. `resolve_model_id` ports a scoped-down version of pi's own
    // `model-resolver.ts` partial/substring matching, but never silently guesses on an ambiguous
    // match — mirrors `SessionRepo::find_path`'s own "list every candidate, don't guess" philosophy
    // instead of pi's own silent-pick-first behavior.

    #[test]
    fn resolve_model_id_an_exact_match_is_returned_in_the_catalogs_own_casing() {
        assert_eq!(
            resolve_model_id("Claude-Opus-4-8", available_models())
                .unwrap()
                .0,
            "claude-opus-4-8"
        );
    }

    #[test]
    fn resolve_model_id_an_unambiguous_partial_match_resolves_to_the_full_id() {
        assert_eq!(
            resolve_model_id("opus", available_models()).unwrap().0,
            "claude-opus-4-8"
        );
        assert_eq!(
            resolve_model_id("SONNET", available_models()).unwrap().0,
            "claude-sonnet-4-5"
        );
    }

    #[test]
    fn resolve_model_id_an_ambiguous_partial_match_errors_naming_every_candidate() {
        let err = resolve_model_id("gpt", available_models()).unwrap_err();
        assert!(err.contains("gpt-5"), "got: {err}");
        assert!(err.contains("gpt-5-mini"), "got: {err}");
        assert!(err.contains("gpt-4o"), "got: {err}");
        assert!(err.contains("gpt-4.1"), "got: {err}");
    }

    #[test]
    fn resolve_model_id_no_match_at_all_is_forwarded_verbatim() {
        // `available_models` is a hint, not an allowlist — a brand-new or provider-specific id the
        // gateway still understands must reach it unchanged, not be rejected.
        assert_eq!(
            resolve_model_id("totally-custom-id", available_models())
                .unwrap()
                .0,
            "totally-custom-id"
        );
    }

    #[test]
    fn resolve_model_id_an_exact_match_short_circuits_before_any_ambiguity_check() {
        // "gpt-5" is itself a substring of "gpt-5-mini" too, but an exact match must win outright
        // rather than ever reaching the ambiguous-candidates error path.
        assert_eq!(
            resolve_model_id("gpt-5", available_models()).unwrap().0,
            "gpt-5"
        );
    }

    #[test]
    fn resolve_model_id_a_colon_level_suffix_resolves_the_id_and_returns_the_level() {
        // Fix 2 (pi-parity gap): `--model sonnet:high` must resolve to the sonnet model id *and*
        // report `high` as the level to apply — pi's own `--model <pattern>:<thinking-level>`
        // shorthand, previously only honored by `--models`/`resolve_model_scope`.
        let (id, level) = resolve_model_id("sonnet:high", available_models()).unwrap();
        assert_eq!(id, "claude-sonnet-4-5");
        assert_eq!(level, Some(agent_core::ThinkingLevel::High));
    }

    #[test]
    fn resolve_model_id_no_colon_suffix_returns_no_level() {
        let (id, level) = resolve_model_id("opus", available_models()).unwrap();
        assert_eq!(id, "claude-opus-4-8");
        assert_eq!(level, None);
    }

    #[test]
    fn resolve_model_id_an_invalid_colon_suffix_is_kept_as_part_of_the_literal_id() {
        // Matches `resolve_model_scope`'s identical treatment of an invalid suffix: not a valid
        // thinking level, so it's left attached and the whole string is resolved (or kept literal) as
        // one id rather than silently dropped.
        let (id, level) =
            resolve_model_id("totally-custom-id:notalevel", available_models()).unwrap();
        assert_eq!(id, "totally-custom-id:notalevel");
        assert_eq!(level, None);
    }

    #[tokio::test]
    async fn switch_branch_returns_a_clear_error_when_no_session_persistence_is_configured() {
        // B-M7 pi-parity gap: in `--no-session-persistence` (pure in-memory) mode there's no tree to
        // navigate at all — `self.store` is `None` by construction. `switch_branch` must fail with a
        // clear, documented error rather than panicking or returning a confusing failure; this
        // codebase has no in-memory tree fallback the way pi does (a deliberate architectural
        // difference, not a gap to close), so the contract this test protects is simply "fails
        // cleanly, doesn't crash."
        let mut persistence = Persistence {
            repo: None,
            store: None,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let agent = Agent::new(
            Arc::new(agent_core::MockTransport::new(vec![])),
            "claude-test",
        );
        let cancel = CancellationToken::new();

        let err = persistence
            .switch_branch(&agent, "some-id", false, true, None, false, &cancel)
            .await
            .expect_err("must fail clearly, not panic, when there's no session tree to navigate");

        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no session persistence configured"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn switch_branch_propagates_a_non_cancelled_summarization_failure_instead_of_switching_anyway()
     {
        // Fix 3 (pi-parity gap): a genuine (non-abort) summarization failure used to be logged and the
        // switch proceeded anyway with no summary attached — this test pins the corrected behavior:
        // the switch must not happen at all, and the caller must see it failed, matching pi's own
        // `packages/coding-agent/src/core/agent-session.ts`'s `navigateTree`, which `throw`s on any
        // non-abort summarization error (any non-abort summarization error is fatal to the whole
        // navigation).
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", None).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            session.user("c");
            session.user("d");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };

        // No scripted turns at all: the one model call `summarize_branch` makes fails outright — a
        // stand-in for a real network error, not a client-requested abort.
        let agent = Agent::new(
            Arc::new(agent_core::MockTransport::new(vec![])),
            "claude-test",
        );
        let cancel = CancellationToken::new();

        // Switching to `ids[1]` ("b") abandons "c"/"d" (non-empty), so `summarize:true` actually
        // attempts a summarization call rather than skipping it as a no-op.
        let err = persistence
            .switch_branch(&agent, &ids[1], false, true, None, false, &cancel)
            .await
            .expect_err("a genuine summarization failure must fail the whole switch");
        assert_ne!(
            err.kind(),
            std::io::ErrorKind::Interrupted,
            "must not be reported as a client-requested abort — that's a distinct, non-error outcome"
        );

        // The switch must not have happened: the active tip is still "d", not "b".
        assert_eq!(
            persistence.store.as_ref().unwrap().active_ids(),
            ids.as_slice(),
            "a failed summarization must leave the session on its original branch"
        );
    }

    #[tokio::test]
    async fn switch_branch_threads_replace_instructions_through_to_the_summarization_request() {
        // Task #17 (pi-parity fix): `replace_instructions:true` must actually reach
        // `Agent::summarize_branch`/`branch_summary::branch_summary_request` — previously `switch_branch`
        // had no such parameter at all and always passed a hardcoded `false`, so a caller asking to
        // fully replace the default structured template with its own instructions had no way to do so.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", None).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            session.user("c");
            session.user("d");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };

        let transport = Arc::new(agent_core::MockTransport::new(vec![
            agent_core::mock::turn::text("recap"),
        ]));
        let agent = Agent::new(transport.clone(), "claude-test");
        let cancel = CancellationToken::new();

        // Switching to "b" abandons "c"/"d" (non-empty), so `summarize:true` actually attempts a
        // summarization call rather than skipping it as a no-op.
        let (_session, _resolved) = persistence
            .switch_branch(
                &agent,
                &ids[1],
                false,
                true,
                Some("only mention the file names"),
                true,
                &cancel,
            )
            .await
            .expect("summarization is scripted to succeed");

        let requests = transport.requests();
        assert_eq!(
            requests.len(),
            1,
            "exactly one summarization call must fire"
        );
        let prompt = match requests[0].messages.first().map(|m| &m.content[0]) {
            Some(agent_core::ContentBlock::Text { text, .. }) => text.clone(),
            other => panic!("expected a single text block, got {other:?}"),
        };
        assert!(
            prompt.contains("only mention the file names"),
            "custom_instructions must reach the request: {prompt}"
        );
        assert!(
            !prompt.contains(agent_core::branch_summary::BRANCH_SUMMARY_INSTRUCTION),
            "replace_instructions:true must drop the default structured template entirely: {prompt}"
        );
        assert!(
            !prompt.contains("Additional focus:"),
            "replace_instructions:true must not use the append-style framing: {prompt}"
        );
    }

    #[test]
    fn set_label_and_get_label_return_a_clear_error_when_no_session_persistence_is_configured() {
        // Pi-parity audit H3: `SessionStore::set_label`/`get_label` were fully built, persisted, and
        // even carried across forks, but had no RPC surface at all — `Persistence::set_label`/
        // `get_label` are the RPC handlers' entry point. Same "no tree, no label" contract as
        // `switch_branch` above.
        let mut persistence = Persistence {
            repo: None,
            store: None,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let err = persistence
            .set_label("some-id", Some("checkpoint"))
            .expect_err("must fail clearly when there's no session tree to label");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no session persistence configured"),
            "got: {err}"
        );

        let err = persistence
            .get_label("some-id")
            .expect_err("must fail clearly when there's no session tree to query");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn list_trash_and_restore_session_return_a_clear_error_outside_repo_mode() {
        // Fix 7: `.trash/` is a `SessionRepo`-level concept (multiple sessions sharing one directory),
        // same as `delete`'s own repo-mode requirement — neither single-file nor in-memory-only mode has
        // a repo directory to consult.
        let persistence = Persistence {
            repo: None,
            store: None,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let err = persistence
            .list_trash()
            .expect_err("must fail clearly outside repo mode");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(err.to_string().contains("not in repo mode"), "got: {err}");

        let err = persistence
            .restore_session("some-id")
            .expect_err("must fail clearly outside repo mode");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn list_trash_and_restore_session_round_trip_through_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let (persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", None).unwrap();
        let repo = persistence.repo.as_ref().unwrap();
        let other = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let other_id = other.meta().id.clone();

        assert!(persistence.list_trash().unwrap().is_empty());
        persistence.delete(&other_id).unwrap();
        let trash = persistence.list_trash().unwrap();
        assert_eq!(trash.len(), 1);
        assert_eq!(trash[0].id, other_id);

        assert!(persistence.restore_session(&other_id).unwrap());
        assert!(persistence.list_trash().unwrap().is_empty());
        assert!(!persistence.restore_session("never-existed").unwrap());
    }

    #[test]
    fn append_custom_returns_a_clear_error_when_no_session_persistence_is_configured() {
        // Pi-parity audit: `SessionStore::append_custom` was fully built and tested but had no RPC
        // surface at all — `Persistence::append_custom` is the RPC handler's entry point. Same
        // "no tree, nothing to append to" contract as `set_label`/`get_label` above.
        let mut persistence = Persistence {
            repo: None,
            store: None,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let err = persistence
            .append_custom("marker", json!({"k": "v"}))
            .expect_err("must fail clearly when there's no session tree to append to");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no session persistence configured"),
            "got: {err}"
        );
    }

    #[test]
    fn fork_returns_a_clear_error_when_no_session_persistence_is_configured() {
        // Same contract as `switch_branch` above, for `Persistence::fork` — in-memory mode has no
        // `repo` to fork within either (`--session-file`/`--no-session-persistence` both leave `repo`
        // `None`; only `--session-dir` sets it).
        let mut persistence = Persistence {
            repo: None,
            store: None,
            meta: SessionMeta::new("/w", "claude-test"),
        };

        let err = persistence
            .fork(usize::MAX, None, false, agent_core::ThinkingLevel::Off)
            .expect_err("must fail clearly, not panic, without a repo to fork within");

        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn fork_restores_the_model_and_level_recorded_at_the_forked_from_point() {
        // Task #2 (pi-parity fix): `fork`/`clone` previously swapped `session`/`store` but never
        // resolved model/level at all — the process's current global setting silently bled into the
        // forked session regardless of what was actually recorded on the branch being forked. Since
        // `SessionRepo::fork_at_entry` deliberately doesn't carry `ModelChange` bookkeeping into the new
        // session's own file (see its doc comment), this must be resolved against the *source* tree,
        // before the fork swaps `self.store`.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-a", None).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };
        // Record a model change anchored at "a" (ids[0]) — everything from there on (including "b")
        // should resolve to "claude-b", even though the *source* session's `meta.model` is still
        // "claude-a" (its creation-time model) and the live process may be running neither.
        {
            let store = persistence.store.as_mut().unwrap();
            store.switch_active(&ids[0]).unwrap();
            store.record_model_change("claude-b").unwrap();
            store.switch_active(&ids[1]).unwrap();
        }

        let (_session, restored_model, restored_level) = persistence
            .fork(usize::MAX, None, false, agent_core::ThinkingLevel::Off)
            .expect("fork must succeed");

        assert_eq!(
            restored_model, "claude-b",
            "must resolve against the source tree's own recorded model change, not meta.model \
             (\"claude-a\") or a hardcoded default"
        );
        // No thinking-level change was ever recorded, so this falls back to the process's own starting
        // level, exactly like `model_and_level_at`'s doc comment describes.
        assert_eq!(restored_level, agent_core::ThinkingLevel::Off);

        // The new session's own header must NOT have silently inherited a stale model either — Task
        // #18 fixes `record_model_change` to also keep `meta.model` current, so the source's `meta`
        // (copied verbatim into the fork's own header by `SessionRepo::fork`) already reflects the
        // latest recorded model by the time the fork runs.
        assert_eq!(persistence.meta.model, "claude-b");
    }

    #[test]
    fn fork_at_entry_restores_the_model_recorded_at_the_target_not_the_active_tip() {
        // Same as the test above, but for the `target_id`-based `fork_at_entry` path — the resolved
        // point is `target_id` itself (or its parent, when `before` is set), not wherever the active
        // path currently ends.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-a", None).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };
        {
            let store = persistence.store.as_mut().unwrap();
            store.record_model_change("claude-b").unwrap(); // anchored at the active tip, "b"
        }

        // Forking at "a" (before the model change ever took effect) must resolve to "claude-a", not
        // "claude-b" — the change was anchored at "b", not "a".
        let (_session, restored_model, _restored_level) = persistence
            .fork(
                usize::MAX,
                Some(&ids[0]),
                false,
                agent_core::ThinkingLevel::Off,
            )
            .expect("fork_at_entry must succeed");
        assert_eq!(restored_model, "claude-a");
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
            Persistence::open_repo(dir.path(), "/w", "claude-original", None).unwrap();
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
        // brand-new `serve` process's `Persistence::open` would do — reattaching to the most recent
        // session for this cwd rather than creating a new one.
        let (restarted, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-cli-default", None).unwrap();

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

    /// Runs `git` with `args` in `cwd`, synchronously — test setup only (the production `git_branch`
    /// helper uses the async `tokio::process::Command` instead; see its own doc comment for why).
    fn run_git(cwd: &std::path::Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .status()
            .expect("git must be on PATH for this test");
        assert!(status.success(), "git {args:?} failed in {}", cwd.display());
    }

    #[tokio::test]
    async fn git_branch_reports_none_outside_a_git_repository() {
        // Task #25 (pi-parity fix): a lookup failure (no repo here at all) must report `None`, never
        // an error surfaced to the RPC caller.
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(git_branch(dir.path()).await, None);
    }

    #[tokio::test]
    async fn git_branch_reports_the_current_branch_inside_a_repository() {
        let dir = tempfile::tempdir().unwrap();
        run_git(dir.path(), &["init", "--quiet", "--initial-branch=main"]);
        run_git(
            dir.path(),
            &["commit", "--quiet", "--allow-empty", "-m", "init"],
        );
        run_git(
            dir.path(),
            &["checkout", "--quiet", "-b", "feature/pi-parity"],
        );

        assert_eq!(
            git_branch(dir.path()).await.as_deref(),
            Some("feature/pi-parity")
        );
    }

    #[tokio::test]
    async fn git_branch_reports_none_on_a_detached_head() {
        // `git symbolic-ref --short HEAD` (and this helper) only ever resolves a real branch ref —
        // detached HEAD has none, matching `symbolic-ref`'s own failure in that state rather than
        // falling back to a raw commit SHA.
        let dir = tempfile::tempdir().unwrap();
        run_git(dir.path(), &["init", "--quiet", "--initial-branch=main"]);
        run_git(
            dir.path(),
            &["commit", "--quiet", "--allow-empty", "-m", "init"],
        );
        run_git(dir.path(), &["checkout", "--quiet", "HEAD~0"]);

        assert_eq!(git_branch(dir.path()).await, None);
    }
}
