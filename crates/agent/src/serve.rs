//! Headless `serve` — a newline-delimited JSON control protocol over stdio.
//!
//! The server is the source of truth; any client (a TUI, an editor, or an `ssh` pipe) drives it by
//! writing one JSON command per line to stdin and reading one JSON frame per line from stdout. The
//! shape mirrors pi's `rpc` mode and opencode's session server: commands get a `response` frame,
//! and a `prompt` streams `event` frames (the agent's `AgentEvent`s) before its response.
//!
//! Sessions persist as append-only JSONL: `--session-file` for one session, or `--session-dir` for a
//! [`SessionRepo`](crate::session_store::SessionRepo) of many. A turn appends only its new messages
//! (compaction rewrites atomically). A reattaching client sees a **stable** session id and metadata.
//!
//! Commands (stdin): `{id?, type, …}`
//!   - `{type:"prompt", message, streaming_behavior?}` run a turn: an immediate lightweight `ack`
//!     frame (the turn is queued and starting), then `event` frames, then a `response` whose `data`
//!     includes `refused: bool` — whether the run's last turn ended in a refusal rather than an
//!     ordinary stop (a refusal doesn't drain queued `steer`/`follow_up` messages; see `agent-core`).
//!     Sent while another `prompt` is already in flight, it's rejected as busy *unless*
//!     `streaming_behavior` is `"steer"` or `"follow_up"`, in which case `message` is queued through
//!     the same `Steering` lane an explicit `steer`/`follow_up` command would use.
//!   - `{type:"abort"}`                  cancel the in-flight `prompt` (if any), else a no-op ack
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
//!   - `{type:"get_state"}`              → `data: {session_id, model, steps, message_count, title,
//!     thinking_level, auto_compaction, auto_retry, steering_mode, follow_up_mode, pending_messages, …}`
//!     — the last six are the runtime-mutable settings and current steer/follow-up queue depth
//!     (`Steering::pending_count`), so a client can render current settings without a second round trip
//!   - `{type:"get_messages", since?}`   → `data: {messages: [...]}` (each tagged with its tree `id`
//!     when persistence is configured, so a client can fork from any point via `switch_branch`);
//!     `since` (a tree id the client already has) returns only the messages appended after it — pi's
//!     own `get_entries({since})` — an error, not a silent full re-fetch, when `since` names no known
//!     id (or persistence isn't configured, so nothing is tagged at all)
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
//!   - `{type:"fork", upto?, target_id?, before?}` (repo mode) copy a prefix into a new session, switch
//!     to it. `target_id` (any tree entry, on or off the active path) wins over `upto` (a message-count
//!     prefix of just the active path) when given; `before:true` excludes `target_id` itself from the
//!     copied prefix (fork right before it), the default `false` includes it.
//!   - `{type:"clone"}` (repo mode) `fork` with no arguments — the current session's active path in
//!     full, at its current tip. pi's own `clone`: a thin, argument-free alias, not a separate code
//!     path (pi needs it because pi's own `fork` requires an explicit entry id; this crate's `fork`
//!     already defaults to the same behavior when called bare).
//!   - `{type:"get_fork_messages"}` list this session's own candidate fork points — every user-turn
//!     entry on the active path, `{entry_id, text}` — matching pi's own same-named command (which takes
//!     no parameters and is scoped to the current session only); feed one `entry_id` to `fork`'s
//!     `target_id` to actually fork there. Not a preview of `fork`'s output — see `preview_fork` for
//!     that (this crate's own extension, previously misnamed `get_fork_messages`, which broke a
//!     pi-compatible client expecting the shape above)
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
//!   - `{type:"compact"}`                summarize the prefix now → `data: {compacted: bool}`
//!   - `{type:"get_last_assistant_text"}` → `data: {text}` (the latest assistant reply)
//!   - `{type:"get_session_stats"}`      → token/step accounting + message-type breakdown
//!     (`user_messages`/`assistant_messages`/`tool_calls`/`tool_results`/`total_messages`)
//!   - `{type:"get_commands"}`           → discoverable skills + prompt templates
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
//!     non-exhaustive picker hint, not an allowlist)
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
//!     attempt emits an unsolicited `auto_retry` frame (`{type:"auto_retry", id, command:"prompt",
//!     attempt, max_attempts, delay_ms, error}`) before its backoff sleep. A sequence that made at
//!     least one attempt ends with exactly one `auto_retry_end` frame (`{type:"auto_retry_end", id,
//!     command:"prompt", success, attempt, final_error?}`) — `success:true` when a retried attempt
//!     recovers, `success:false` (with `final_error`) when retries are exhausted, the failure turns out
//!     non-retryable, or the pending backoff itself is interrupted via `abort`/`abort_retry`
//!     (`final_error:"retry cancelled"`) — mirroring pi's own `auto_retry_end` event.
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
//!   - `{type:"get_tree"}`               → `data: {nodes: [TreeNode…], leaf_id}` (every message on
//!     every branch, not just the leaves `list_branches` reports; `leaf_id` is the active path's own
//!     tip — `null` in pure in-memory mode — pi's own `get_tree`'s `leafId`)
//!   - `{type:"switch_branch", target_id, before?, summarize?, custom_instructions?}` navigate to
//!     another point in the tree — or, when `before:true`, to `target_id`'s own *parent* instead,
//!     which is the tree's own root (before any message) when `target_id` is the very first message,
//!     letting a client redo it in place (pi's own `SessionManager::resetLeaf`) — summarizing the
//!     abandoned branch's activity first unless `summarize:false` (an optional `custom_instructions`
//!     string steers what that recap emphasizes, the same "Additional focus" framing `compact`'s own
//!     `custom_instructions` supports; ignored when `summarize:false`, since no summarization call
//!     happens at all) → `data: {target_id, model, reasoning_effort}` — also restores whichever
//!     model/thinking-level was actually active at wherever the session actually landed (a
//!     `set_model`/`cycle_model`/`set_reasoning_effort`/`cycle_thinking_level` made *after* leaving
//!     that point doesn't leak backward into it), rebuilding the `Agent` if either differs from what's
//!     currently active
//!   - `{type:"bash", command, cwd?, timeout_ms?, exclude_from_context?}` run a shell command directly
//!     — independent of the model's own tool-call loop — streaming `tool_progress`/`tool_end` events
//!     exactly like a model-invoked `bash` call. Recorded into the session as a plain informational
//!     message by default (so the model sees it on its next turn — pi's `recordBashResult`), unless
//!     `exclude_from_context: true`. Rejected if `bash` isn't registered for this process
//!     (`--exclude-tools bash` / `--no-tools`). While it runs, only `abort_bash`/`abort` (cancel it) are
//!     accepted; everything else is rejected as busy.
//!   - `{type:"abort_bash"}`             cancel an in-flight host `bash` command, else a no-op ack
//!
//! While a `prompt` runs, the loop keeps reading stdin so an `abort` can cancel it, or `steer` /
//! `follow_up` (with a `message`) can queue input: a `steer` is injected mid-run at the next tool
//! turn (to redirect a busy agent), a `follow_up` waits for the model to next stop. A handful of
//! read-only commands that don't need the run's exclusively-borrowed session — `get_state` (with
//! `message_count: null`, the one field that genuinely needs it), `get_session_stats` (from a live
//! mirror of the session's own counters, updated as the run streams), `get_commands`, `list_branches`,
//! `get_tree`, `list_sessions`, `list_all_sessions`, `get_available_models` — are answered live too,
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
//! `{type:"auto_retry", id?, command:"prompt", attempt, max_attempts, delay_ms, error}`, one per
//! whole-run auto-retry attempt (see `set_auto_retry` above), also correlated via `id` — or
//! `{type:"auto_retry_end", id?, command:"prompt", success, attempt, final_error?}`, the terminal
//! notice for a retry sequence that made at least one attempt — or `{type:"session_info_changed", id?,
//! command:"set_session_name", title}`, pushed once per successful `set_session_name` (pi's own
//! `session_info_changed`; see `set_session_name` above).
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

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use agent_core::{
    Agent, AgentEvent, CancellationToken, GatewayClient, Session, StopReason, StreamEvent,
    ToolUpdate,
};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::session_store::{
    BranchInfo, BranchSummaryDetails, CompactionMeta, SessionMeta, SessionRepo, SessionStore,
    search_sessions,
};
use crate::tools;

/// Options for the headless server (mirrors `run`, plus persistence).
pub struct ServeConfig {
    pub gateway: String,
    pub key: String,
    pub model: String,
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
    /// How many times to retry a gateway request that fails before the first response byte arrives.
    /// `None` keeps the client's built-in default.
    pub retry_max_retries: Option<u32>,
    /// Base of the exponential backoff between those retries. `None` keeps the client's built-in
    /// default.
    pub retry_base_delay_ms: Option<std::time::Duration>,
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
    /// Restrict the tool set to exactly these names, dropping everything else. Combine with
    /// `exclude_tools` to carve one back out of the allow-list. Fixed for the process — like `system`,
    /// there's no runtime RPC to change it, but it does survive a `set_model`/`set_thinking` rebuild
    /// (`build_agent` reapplies it every time).
    pub tools: Option<Vec<String>>,
    /// Drop these tools from the default set — e.g. `["bash", "write"]` for a read-only reviewer.
    pub exclude_tools: Option<Vec<String>>,
    /// Register no tools at all. Wins over `tools`/`exclude_tools`.
    pub no_tools: bool,
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

    /// Resolves once a shutdown signal arrives. Safe to call fresh on every loop iteration — every
    /// `Signal::recv` and `tokio::signal::ctrl_c` are re-armable, not one-shot.
    pub async fn wait(&mut self) {
        #[cfg(unix)]
        {
            tokio::select! {
                _ = self.sigterm.recv() => {}
                _ = self.sighup.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
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
            Some(tokens_before) => {
                store.rewrite_compacted(&session.messages, CompactionMeta { tokens_before })
            }
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
            if let Err(e) = store.rewrite(&[]) {
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
    fn fork(
        &mut self,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
    ) -> std::io::Result<Session> {
        let id = self.meta.id.clone();
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (store, session) = match entry_id {
            Some(entry_id) => repo.fork_at_entry(&id, entry_id, before)?,
            None => repo.fork(&id, upto)?,
        };
        self.meta = store.meta().clone();
        self.store = Some(store);
        Ok(session)
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

    /// All sessions' metadata, newest first (empty unless in repo mode). `on_progress(scanned, total)`
    /// is invoked once per file as the scan completes it — see
    /// [`SessionRepo::list_with_progress`](crate::session_store::SessionRepo::list_with_progress).
    fn list_with_progress(
        &self,
        on_progress: impl Fn(usize, usize) + Send + Sync,
    ) -> Vec<SessionMeta> {
        self.repo
            .as_ref()
            .and_then(|r| match r.list_with_progress(on_progress) {
                Ok(sessions) => Some(sessions),
                Err(e) => {
                    eprintln!("serve: failed to list sessions: {e}");
                    None
                }
            })
            .unwrap_or_default()
    }

    /// Every session across every project's own repo directory (pi's cross-project `listAll`), not
    /// just this process's own — the parent of this repo's directory is treated as the shared sessions
    /// root, with one subdirectory per project (the convention
    /// [`session_store::default_session_dir`](crate::session_store::default_session_dir) follows).
    /// `Err` when not in repo mode, or the repo directory has no parent to scan siblings of.
    /// `on_progress(scanned, total)` is invoked once per file across every project combined — see
    /// [`SessionRepo::list_all_with_progress`](crate::session_store::SessionRepo::list_all_with_progress).
    fn list_all_with_progress(
        &self,
        on_progress: impl Fn(usize, usize) + Send + Sync,
    ) -> std::io::Result<Vec<SessionMeta>> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let root = repo.dir().parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "session directory has no parent to list other projects from",
            )
        })?;
        SessionRepo::list_all_with_progress(root, on_progress)
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

    /// Ids of the active path's messages, root-first — parallel to the live `Session.messages` when
    /// persistence is configured; empty in pure in-memory mode (nothing to id). What `get_messages`
    /// tags each message with, and what a client names as `switch_branch`'s `target_id`.
    fn active_ids(&self) -> &[String] {
        self.store
            .as_ref()
            .map(SessionStore::active_ids)
            .unwrap_or(&[])
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
    /// persists it *before* switching — mirroring pi's `navigateTree`. A summarization failure (a
    /// network error, or the model returning nothing) is logged and the switch proceeds anyway: losing
    /// the recap is far better than being unable to navigate away from a branch at all.
    ///
    /// `custom_instructions`, when given, steers *what* the branch recap emphasizes — the same
    /// "Additional focus" framing manual `compact` already supports — threaded straight through to
    /// [`Agent::summarize_branch`]; ignored (no summarization call happens at all) when `summarize` is
    /// `false`.
    ///
    /// Returns the resolved target alongside the switched-to `Session` — `None` when `before` resolved
    /// to the tree's own root — so the caller can restore the correct model/thinking-level for
    /// wherever the session actually landed (see [`Self::model_and_level_at_opt`]) instead of querying
    /// against the raw, pre-resolution `target_id` argument.
    async fn switch_branch(
        &mut self,
        agent: &Agent,
        target_id: &str,
        before: bool,
        summarize: bool,
        custom_instructions: Option<&str>,
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
                        .summarize_branch(&messages, cancel, custom_instructions)
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
                        Err(e) => {
                            eprintln!("serve: branch summarization failed, switching anyway: {e}")
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

/// Run the control loop until stdin closes.
pub async fn serve(cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
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
    let mut project_trusted = !cfg.force_untrusted
        && (cfg.trust_project
            || crate::trust_store::TrustStore::open_default().is_trusted(&cwd)
            || !crate::trust_store::has_trust_gated_resources(&cwd));

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

    let mut static_system =
        crate::resources::build_static_system_prompt(&crate::resources::PromptOptions {
            base: &cfg.system,
            append: cfg.append_system.as_deref(),
            cwd: &cwd,
            include_context_files: cfg.context_files,
            skills: &skills,
            project_trusted,
        });
    timing.mark("build static system prompt");

    // Keep the transport in an `Arc` we can clone: `set_model`/`set_thinking` rebuild the `Agent` at
    // runtime (a new model id picks a new dialect), and each rebuild reuses this one HTTP client.
    let client = GatewayClient::new(cfg.gateway.clone(), cfg.key.clone())?.with_retry(
        cfg.retry_max_retries
            .unwrap_or(agent_core::client::MAX_RETRIES),
        cfg.retry_base_delay_ms
            .unwrap_or(agent_core::client::BASE_BACKOFF),
    );
    let client = Arc::new(client);

    // The model, thinking budget, and auto-compaction flag are runtime-switchable; everything else
    // (transport, tools, system prompt, loop bounds, cache settings) is fixed for the process.
    // `build_agent` folds the mutable trio into a fresh `Agent` whenever any of them changes.
    let mut current_model = cfg.model.clone();
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
    // counterpart to `cfg.reasoning_effort`, seeded from it so a process started with
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
    let starting_level = agent_core::clamp_thinking_level(
        &agent_core::capabilities(&current_model),
        cfg.reasoning_effort
            .map(agent_core::ThinkingLevel::from)
            .unwrap_or(agent_core::ThinkingLevel::Off),
    );
    // The runtime-mutable level starts at `starting_level`, but `switch_branch` needs the original
    // starting value too — as the fallback for a branch that never recorded its own thinking-level
    // change (see `Persistence::model_and_level_at`), so switching to it lands on the process's real
    // starting depth instead of silently keeping whatever a *different* branch last set.
    let mut current_level = starting_level;
    let mut current_auto_compaction = true;
    // Mid-stream transport-failure retry (`agent_core::Agent::with_auto_retry`) — on by default;
    // `set_auto_retry` lets an operator debugging a flaky network hop disable it to see the raw failure
    // on the very first hiccup instead of after several silent retries.
    let mut current_auto_retry = true;
    // Shared across every `build_agent` rebuild for this process's lifetime, so file-mutation
    // exclusivity (same-path `edit`/`write` calls) survives a `set_model`/`set_thinking` rebuild.
    let write_locks = Arc::new(agent_core::WriteLockRegistry::new());
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
    let mut agent = build_agent(
        client.clone(),
        &full_system(&static_system, &cwd),
        &cfg,
        &current_model,
        current_thinking,
        current_level,
        current_auto_compaction,
        current_auto_retry,
        persistence.session_id(),
        &write_locks,
        &checkpoint,
    );
    timing.mark("build agent");
    // Persistent across every `prompt` call (not just the one currently in flight), so `steer`/
    // `follow_up` sent while idle queue for the *next* `prompt` instead of being rejected as an unknown
    // command. Cleared on every session switch (`new_session`/`switch_session`/`fork`/`switch_branch`)
    // so a message meant for the old session's next turn can't leak into the newly switched-to one.
    let steering = agent_core::Steering::new();
    // The `bash` tool, looked up once from the same (possibly filtered) registry `build_agent` builds —
    // `None` when `bash` was excluded (`--exclude-tools bash` / `--no-tools`), which the `bash` RPC
    // command below reports as a clean error rather than a side door around that restriction. A direct
    // `Arc<dyn Tool>` handle, not routed through `agent`/the model loop: the host `bash` RPC command runs
    // independent of any conversation turn.
    let bash_tool = build_tools(&cfg).get("bash");
    // Distinguishes successive host `bash` calls in their `tool_start`/`tool_progress`/`tool_end` event
    // ids — only ever one in flight at a time (see the `bash` command arm), but a stable, incrementing
    // id per call still lets a client correlate a run's own three events without ambiguity.
    let mut host_bash_seq: u32 = 0;

    // One writer task owns stdout; every frame (events + responses) is serialized through it in FIFO
    // order, so output never interleaves.
    //
    // The channel is intentionally unbounded. The event `sink` (see `Agent::run_events`) is a
    // synchronous `FnMut`, so the producer cannot `.await` to apply backpressure; a bounded channel
    // would force `try_send`, which silently drops frames and corrupts the event stream — unacceptable
    // for a protocol. In practice the backlog is bounded by one in-flight turn's events (capped by
    // `max_steps`), drained concurrently by the writer as fast as stdout accepts. If a client stops
    // reading, stdout's write eventually fails and the writer tears down (below), surfacing the stall
    // rather than masking it.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(frame) = out_rx.recv().await {
            // A frame we built ourselves failing to serialize is a bug, not bad input — skip it
            // rather than tearing down the whole stream.
            let line = match serde_json::to_string(&frame) {
                Ok(line) => line,
                Err(e) => {
                    eprintln!("serve: failed to serialize output frame: {e}");
                    continue;
                }
            };
            // stdout is the only sink. If it breaks (client hung up, broken pipe) there is nothing
            // left to do but stop; dropping `out_rx` here makes every sender observe the closure and
            // halt the control loop instead of writing into a dead pipe forever.
            if let Err(e) = write_frame(&mut out, &line).await {
                eprintln!("serve: stdout write failed, shutting down writer: {e}");
                break;
            }
        }
    });

    // Sends a frame through the writer; if the writer has shut down (stdout closed), stop the control
    // loop — there is no way to deliver any further response, so continuing would only swallow output.
    macro_rules! emit {
        ($frame:expr) => {
            if out_tx.send($frame).is_err() {
                break;
            }
        };
    }

    timing.print();

    // Announce readiness so a client can sync before issuing commands. If this already fails the
    // writer never started; there is nothing to serve.
    if out_tx
        .send(json!({
            "type": "ready",
            "session_id": persistence.session_id(),
            "model": current_model,
            "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
        }))
        .is_err()
    {
        let _ = writer.await;
        return Ok(());
    }

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut shutdown = ShutdownSignal::new()?;
    loop {
        let line = tokio::select! {
            biased;
            // Idle between commands: nothing is in flight, so a shutdown request needs no drain —
            // just stop reading and fall out to the writer join below.
            () = shutdown.wait() => break,
            maybe_line = lines.next_line() => match maybe_line? {
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
                    "?",
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
                let result = 'retry: loop {
                    tokens_before.store(0, Ordering::Relaxed);
                    refused.store(false, Ordering::Relaxed);
                    is_compacting.store(false, Ordering::Relaxed);
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
                                () = shutdown.wait() => {
                                    stdin_open = false;
                                    cancel.cancel();
                                }
                                // Drain and persist each mid-run checkpoint as it arrives (see
                                // `ChannelCheckpoint`). `persistence` isn't touched by `run` itself (only
                                // `session` is borrowed there), so reassigning it here is safe exactly like
                                // `stdin_open`/`cancel` being mutated from a sibling branch above. Any
                                // checkpoint left undrained when the run ends is harmless — the unconditional
                                // full persist right after this loop (below) is a strict superset of it.
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
                                maybe_line = lines.next_line(), if stdin_open => match maybe_line {
                                    Ok(Some(l)) => {
                                        let l = l.trim();
                                        if l.is_empty() {
                                            continue;
                                        }
                                        let c: Value = match serde_json::from_str(l) {
                                            Ok(v) => v,
                                            Err(e) => {
                                                let _ = out_tx.send(response(None, "?", false, None, Some(&format!("invalid JSON: {e}"))));
                                                continue;
                                            }
                                        };
                                        let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                                        match c.get("type").and_then(Value::as_str).unwrap_or("") {
                                            "abort" => {
                                                cancel.cancel();
                                                let _ = out_tx.send(response(cid, "abort", true, None, None));
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
                                                        let _ = out_tx.send(response(cid, cmd, true, None, None));
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
                                                        let _ = out_tx.send(response(cid, "prompt", true, Some(json!({ "queued_as": "steer" })), None));
                                                    }
                                                    (Some("follow_up"), Some(m)) => {
                                                        let m = expand_message(m, &skills, &prompt_templates);
                                                        let m = agent_core::SteeringMessage::new(
                                                            m,
                                                            parse_images(c.get("images")),
                                                        );
                                                        steering.push(m);
                                                        let _ = out_tx.send(response(cid, "prompt", true, Some(json!({ "queued_as": "follow_up" })), None));
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
                                                    m.insert("session_id".into(), json!(persistence.session_id()));
                                                    m.insert("model".into(), json!(current_model));
                                                    m.insert("message_count".into(), Value::Null);
                                                    m.insert("title".into(), json!(persistence.meta.title));
                                                    m.insert("cwd_stale".into(), json!(cwd_is_stale(&persistence.meta.cwd, &cwd)));
                                                    m.insert("session_file".into(), json!(persistence.session_file().map(|p| p.display().to_string())));
                                                    m.insert("is_streaming".into(), json!(true));
                                                    m.insert("is_compacting".into(), json!(is_compacting.load(Ordering::Relaxed)));
                                                    if let Value::Object(rt) = runtime_settings(current_level, current_auto_compaction, current_auto_retry, &steering) {
                                                        m.extend(rt);
                                                    }
                                                }
                                                let _ = out_tx.send(response(cid, "get_state", true, Some(data), None));
                                            }
                                            "get_session_stats" => {
                                                let _ = out_tx.send(response(cid, "get_session_stats", true, Some(live_stats.snapshot()), None));
                                            }
                                            "get_commands" => {
                                                let mut commands: Vec<Value> = skills.iter().map(|s| {
                                                    json!({ "name": format!("skill:{}", s.name), "source": "skill", "description": s.description })
                                                }).collect();
                                                commands.extend(prompt_templates.iter().map(|t| {
                                                    json!({ "name": t.name, "source": "prompt", "description": t.description })
                                                }));
                                                let collisions: Vec<&str> = skill_collisions.iter().chain(prompt_collisions.iter()).map(String::as_str).collect();
                                                let _ = out_tx.send(response(cid, "get_commands", true, Some(json!({ "commands": commands, "collisions": collisions })), None));
                                            }
                                            "list_branches" => {
                                                let _ = out_tx.send(response(cid, "list_branches", true, Some(json!({ "branches": persistence.list_branches() })), None));
                                            }
                                            "get_tree" => {
                                                let _ = out_tx.send(response(cid, "get_tree", true, Some(json!({ "nodes": persistence.tree(), "leaf_id": persistence.active_ids().last() })), None));
                                            }
                                            "list_sessions" => {
                                                let progress_id = cid.clone();
                                                let query = c.get("query").and_then(Value::as_str);
                                                let sessions = persistence
                                                    .list_with_progress(|scanned, total| {
                                                        if should_report_scan_progress(scanned, total) {
                                                            let _ = out_tx.send(list_progress_frame(progress_id.clone(), "list_sessions", scanned, total));
                                                        }
                                                    });
                                                let sessions: Vec<Value> = search_sessions(sessions, query)
                                                    .iter()
                                                    .map(SessionMeta::to_listing_json)
                                                    .collect();
                                                let _ = out_tx.send(response(cid, "list_sessions", true, Some(json!({ "sessions": sessions })), None));
                                            }
                                            "list_all_sessions" => {
                                                let progress_id = cid.clone();
                                                let query = c.get("query").and_then(Value::as_str);
                                                match persistence.list_all_with_progress(|scanned, total| {
                                                    if should_report_scan_progress(scanned, total) {
                                                        let _ = out_tx.send(list_progress_frame(progress_id.clone(), "list_all_sessions", scanned, total));
                                                    }
                                                }) {
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
                                            other => {
                                                let _ = out_tx.send(response(cid, other, false, None, Some("busy: a prompt is running; only `abort`/`abort_retry`/`steer`/`follow_up`, or a handful of read-only commands (get_state/get_session_stats/get_commands/list_branches/get_tree/list_sessions/list_all_sessions/get_available_models), are accepted")));
                                            }
                                        }
                                    }
                                    // stdin closed (or errored) mid-run: cancel and let the run unwind, then
                                    // we'll fall out of the outer loop below.
                                    Ok(None) | Err(_) => {
                                        stdin_open = false;
                                        cancel.cancel();
                                    }
                                }
                            }
                        }
                    };

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
                                && retry_attempt < crate::retry::MAX_RUN_RETRIES
                                && crate::retry::is_retryable_whole_run(e) =>
                        {
                            retry_attempt += 1;
                            let delay = crate::retry::backoff(retry_attempt);
                            let _ = out_tx.send(auto_retry_frame(
                                id.clone(),
                                retry_attempt,
                                crate::retry::MAX_RUN_RETRIES,
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
                                    () = shutdown.wait() => {
                                        stdin_open = false;
                                        retry_cancelled = true;
                                        break;
                                    }
                                    maybe_line = lines.next_line(), if stdin_open => match maybe_line {
                                        Ok(Some(l)) => {
                                            let l = l.trim();
                                            if l.is_empty() {
                                                continue;
                                            }
                                            let c: Value = match serde_json::from_str(l) {
                                                Ok(v) => v,
                                                Err(e) => {
                                                    let _ = out_tx.send(response(None, "?", false, None, Some(&format!("invalid JSON: {e}"))));
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
                                                other => {
                                                    let _ = out_tx.send(response(cid, other, false, None, Some("busy: retrying after a transient error; only `abort`/`abort_retry` are accepted until the retry starts")));
                                                }
                                            }
                                        }
                                        Ok(None) | Err(_) => {
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
                        emit!(response(id, cmd_type, true, None, None));
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
                    m.insert("session_id".into(), json!(persistence.session_id()));
                    m.insert("model".into(), json!(current_model));
                    m.insert("message_count".into(), json!(session.messages.len()));
                    m.insert("title".into(), json!(persistence.meta.title));
                    m.insert(
                        "cwd_stale".into(),
                        json!(cwd_is_stale(&persistence.meta.cwd, &cwd)),
                    );
                    m.insert(
                        "session_file".into(),
                        json!(persistence.session_file().map(|p| p.display().to_string())),
                    );
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
                emit!(response(
                    id,
                    "get_messages",
                    true,
                    Some(json!({ "messages": messages })),
                    None,
                ));
            }
            "new_session" => match persistence.new_session(
                &current_model,
                cmd.get("parent_session").and_then(Value::as_str),
            ) {
                Ok(s) => {
                    session = s;
                    steering.clear();
                    emit!(response(
                        id,
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
                Err(e) => emit!(response(
                    id,
                    "new_session",
                    false,
                    None,
                    Some(&e.to_string())
                )),
            },
            "list_sessions" => {
                let progress_id = id.clone();
                let query = cmd.get("query").and_then(Value::as_str);
                let sessions = persistence.list_with_progress(|scanned, total| {
                    if should_report_scan_progress(scanned, total) {
                        let _ = out_tx.send(list_progress_frame(
                            progress_id.clone(),
                            "list_sessions",
                            scanned,
                            total,
                        ));
                    }
                });
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
                let query = cmd.get("query").and_then(Value::as_str);
                match persistence.list_all_with_progress(|scanned, total| {
                    if should_report_scan_progress(scanned, total) {
                        let _ = out_tx.send(list_progress_frame(
                            progress_id.clone(),
                            "list_all_sessions",
                            scanned,
                            total,
                        ));
                    }
                }) {
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
            "switch_session" => match cmd.get("session_id").and_then(Value::as_str) {
                Some(target) => match persistence.switch(target) {
                    Ok(s) => {
                        session = s;
                        steering.clear();
                        // Restore whichever model/thinking-level this session was actually last
                        // running on, the same way `switch_branch` already does — without this, the
                        // process's current global model/level (possibly set on a *different* session
                        // entirely) silently bled into the reattached session: reopening a session last
                        // driven on `gpt-5` without re-passing `--model` would continue it on whatever
                        // the process happened to be running, with no warning. See
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
                                persistence.session_id(),
                                &write_locks,
                                &checkpoint,
                            );
                        }
                        emit!(response(
                            id,
                            "switch_session",
                            true,
                            Some(json!({
                                "session_id": persistence.session_id(),
                                "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
                                "model": current_model,
                            })),
                            None,
                        ));
                    }
                    Err(e) => emit!(response(
                        id,
                        "switch_session",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                },
                None => emit!(response(
                    id,
                    "switch_session",
                    false,
                    None,
                    Some("missing `session_id`")
                )),
            },
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
            "fork" => {
                // `upto` messages to copy into the new session; absent = clone the whole session.
                // `target_id`, when given, forks at that specific tree entry instead — anywhere in the
                // whole tree, not just a message-count prefix of the active path (`before` excludes the
                // entry itself); wins over `upto` if both are present.
                let upto = cmd
                    .get("upto")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .unwrap_or(usize::MAX);
                let target_id = cmd.get("target_id").and_then(Value::as_str);
                let before = cmd.get("before").and_then(Value::as_bool).unwrap_or(false);
                match persistence.fork(upto, target_id, before) {
                    Ok(s) => {
                        session = s;
                        steering.clear();
                        emit!(response(
                            id,
                            "fork",
                            true,
                            Some(json!({
                                "session_id": persistence.session_id(),
                                "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
                            })),
                            None,
                        ));
                    }
                    Err(e) => emit!(response(id, "fork", false, None, Some(&e.to_string()))),
                }
            }
            // pi's own `clone` — fork the current session at its current tip, with no arguments —
            // exists there because pi's `fork` *requires* an explicit `entryId`; this crate's `fork`
            // already defaults to exactly that (no `upto`/`target_id` given), so `clone` is a thin,
            // deliberately-argument-free alias over the same call for a client speaking pi's protocol
            // shape, not a second code path.
            "clone" => match persistence.fork(usize::MAX, None, false) {
                Ok(s) => {
                    session = s;
                    steering.clear();
                    emit!(response(
                        id,
                        "clone",
                        true,
                        Some(json!({
                            "session_id": persistence.session_id(),
                            "cwd_stale": cwd_is_stale(&persistence.meta.cwd, &cwd),
                        })),
                        None,
                    ));
                }
                Err(e) => emit!(response(id, "clone", false, None, Some(&e.to_string()))),
            },
            "get_fork_messages" => {
                // pi-compatible contract: no parameters, scoped to *this* session only — every
                // user-turn entry on the active path, as a flat `{entry_id, text}` candidate list (pi's
                // own `getUserMessagesForForking`), for a client to build a fork-point picker from and
                // then feed one `entry_id` to `fork`'s own `target_id`. This is a listing, not a preview
                // of any one fork's output — see `preview_fork` for that.
                let msg_ids = persistence.active_ids();
                let candidates: Vec<Value> = if msg_ids.len() == session.messages.len() {
                    session
                        .messages
                        .iter()
                        .zip(msg_ids)
                        .filter(|(m, _)| m.role == agent_core::Role::User)
                        .filter_map(|(m, entry_id)| {
                            user_message_text(m)
                                .map(|text| json!({ "entry_id": entry_id, "text": text }))
                        })
                        .collect()
                } else {
                    // In-memory-only mode (no persistence configured), or some other length mismatch —
                    // there are no stable entry ids to offer, so there is nothing a client could
                    // meaningfully pass back to `fork`'s `target_id`. Matches `get_messages`'s own
                    // precedent for this same mismatch: degrade gracefully, don't error.
                    Vec::new()
                };
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
                let before = cmd.get("before").and_then(Value::as_bool).unwrap_or(false);
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
                match crate::export::export_html(
                    &persistence.meta,
                    &session.messages,
                    &branches,
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
            "compact" => {
                // Manual compaction (no run in flight here). Streams a `compacted` event if it cuts.
                // `custom_instructions`, when given, steers what the summary emphasizes — pi's own
                // `compact(customInstructions)` — rather than replacing the structured template.
                let custom_instructions = cmd.get("custom_instructions").and_then(Value::as_str);
                let tx = out_tx.clone();
                let mut compacted_tokens_before: Option<u32> = None;
                let mut compacted_summary: Option<String> = None;
                let mut compacted_tokens_after: Option<u32> = None;
                let result = agent
                    .compact(
                        &mut session,
                        agent_core::CompactionReason::Manual,
                        &CancellationToken::new(),
                        &mut |ev| {
                            // Matched by reference: `summary` is owned (`String`), and binding it by
                            // value here would partially move `ev` out from under the `event_frame(ev)`
                            // call just below, which still needs the whole event intact to forward.
                            if let AgentEvent::Compacted {
                                tokens_before,
                                summary,
                                tokens_after,
                                ..
                            } = &ev
                            {
                                compacted_tokens_before = Some(*tokens_before);
                                compacted_summary = Some(summary.clone());
                                compacted_tokens_after = Some(*tokens_after);
                            }
                            if let Some(frame) = event_frame(ev) {
                                let _ = tx.send(frame);
                            }
                        },
                        custom_instructions,
                    )
                    .await;
                match result {
                    Ok(did) => {
                        let (p, persist_result) =
                            persist_blocking(persistence, session.clone(), compacted_tokens_before)
                                .await;
                        persistence = p;
                        match persist_result {
                            Ok(()) => emit!(response(
                                id,
                                "compact",
                                true,
                                Some(json!({
                                    "compacted": did,
                                    "summary": compacted_summary,
                                    "tokens_before": compacted_tokens_before,
                                    "tokens_after": compacted_tokens_after,
                                })),
                                None
                            )),
                            Err(e) => emit!(response(
                                id,
                                "compact",
                                false,
                                None,
                                Some(&format!("compacted but failed to persist: {e}"))
                            )),
                        }
                    }
                    Err(e) => emit!(response(id, "compact", false, None, Some(&e.to_string()))),
                }
            }
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
            "get_session_stats" => {
                emit!(response(
                    id,
                    "get_session_stats",
                    true,
                    Some(session_stats(&session, &current_model)),
                    None
                ));
            }
            "get_commands" => {
                // Skills (read-on-demand) and prompt templates (`/name`), for client autocomplete.
                let mut commands: Vec<Value> = skills
                    .iter()
                    .map(|s| {
                        json!({ "name": format!("skill:{}", s.name), "source": "skill", "description": s.description })
                    })
                    .collect();
                commands.extend(prompt_templates.iter().map(
                    |t| json!({ "name": t.name, "source": "prompt", "description": t.description }),
                ));
                // Every shadowed name (a skill or template defined at more than one path) is otherwise
                // silently resolved with no way for a client to notice — surfaced here instead.
                let collisions: Vec<&str> = skill_collisions
                    .iter()
                    .chain(prompt_collisions.iter())
                    .map(String::as_str)
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
                project_trusted = !cfg.force_untrusted
                    && (cfg.trust_project
                        || crate::trust_store::TrustStore::open_default().is_trusted(&cwd)
                        || !crate::trust_store::has_trust_gated_resources(&cwd));
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
                static_system = crate::resources::build_static_system_prompt(
                    &crate::resources::PromptOptions {
                        base: &cfg.system,
                        append: cfg.append_system.as_deref(),
                        cwd: &cwd,
                        include_context_files: cfg.context_files,
                        skills: &skills,
                        project_trusted,
                    },
                );
                agent.set_system(full_system(&static_system, &cwd));
                emit!(response(id, "reload", true, None, None));
            }
            // Rejects only an empty/whitespace-only id, not an unrecognized one: unlike pi (which talks
            // directly to each provider and can validate against a live, authoritative registry of what
            // it's actually configured to reach), every model id here is forwarded verbatim through the
            // gateway (`AI_GATEWAY_URL`) — this process has no local source of truth to validate a real
            // id against, and `available_models()` is explicitly documented as a non-exhaustive picker
            // hint, not an allowlist (see its own doc comment). What IS a genuine, unambiguous mistake
            // regardless of the gateway's own model set — an empty string sneaking through `Value::as_str`
            // and getting durably recorded via `record_model_change` — is caught here, the same class of
            // fix `set_session_name` already got (reject empty, don't pretend to validate against a list
            // this process can't actually authoritatively check).
            "set_model" => match cmd
                .get("model")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(model) => {
                    // Persist the lineage marker *before* applying the switch in memory: if it fails
                    // to write, leave the live model unchanged too, rather than forking live state
                    // away from what's durably recorded (an `Err` here aborts the whole switch).
                    let record_result = if model != current_model {
                        persistence.record_model_change(model)
                    } else {
                        Ok(())
                    };
                    match record_result {
                        Ok(()) => {
                            // A signed thinking block is only valid for replay to the model that
                            // produced it, and a combined OpenAI-Responses tool-call id only means
                            // anything back to that same model — scrub both from any message not
                            // already stamped with the model we're switching to.
                            session.scrub_cross_model_state(model);
                            current_model = model.to_string();
                            // Re-clamp the *existing* level against the *new* model: e.g. a session
                            // sitting at `Off` on a disable-capable model must not silently carry that
                            // `Off` over to a model that can't actually disable reasoning.
                            current_level = agent_core::clamp_thinking_level(
                                &agent_core::capabilities(&current_model),
                                current_level,
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
                                persistence.session_id(),
                                &write_locks,
                                &checkpoint,
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
                            emit!(response(id, "set_model", false, None, Some(&e.to_string())))
                        }
                    }
                }
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
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
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
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
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
                                    persistence.session_id(),
                                    &write_locks,
                                    &checkpoint,
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
                            // Same staleness hazard `cycle_thinking_level` already guards against:
                            // a stale raw-budget override must not silently outlive the level it
                            // was pinned over.
                            current_thinking = None;
                            current_level = level;
                        }
                        // See `set_model`'s identical re-clamp for why this can't be skipped.
                        current_level = agent_core::clamp_thinking_level(
                            &agent_core::capabilities(&current_model),
                            current_level,
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
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                        );
                        let mut resp_data = model_switch_response(&current_model, current_level);
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
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
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
                    agent = build_agent(
                        client.clone(),
                        &full_system(&static_system, &cwd),
                        &cfg,
                        &current_model,
                        current_thinking,
                        current_level,
                        current_auto_compaction,
                        current_auto_retry,
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
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
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
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
            // Toggle how much of the steer lane a single mid-run drain point consumes (see
            // `agent_core::QueueMode`) — pi's default is `"one_at_a_time"`, one queued message per
            // drain, leaving the rest queued for the next one; `"all"` folds everything queued into a
            // single injection (this crate's original behavior). Independent of `set_follow_up_mode`
            // below — pi carries the same split (`steeringMode`/`followUpMode`). Owned entirely by the
            // `Steering` handle itself (shared with the loop, no `Agent` rebuild needed), so this takes
            // effect immediately, including mid-run.
            "set_steering_mode" => match cmd.get("mode").and_then(Value::as_str) {
                Some("one_at_a_time") => {
                    steering.set_steering_mode(agent_core::QueueMode::OneAtATime);
                    emit!(response(
                        id,
                        "set_steering_mode",
                        true,
                        Some(json!({ "mode": "one_at_a_time" })),
                        None,
                    ));
                }
                Some("all") => {
                    steering.set_steering_mode(agent_core::QueueMode::All);
                    emit!(response(
                        id,
                        "set_steering_mode",
                        true,
                        Some(json!({ "mode": "all" })),
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
                Some("one_at_a_time") => {
                    steering.set_follow_up_mode(agent_core::QueueMode::OneAtATime);
                    emit!(response(
                        id,
                        "set_follow_up_mode",
                        true,
                        Some(json!({ "mode": "one_at_a_time" })),
                        None,
                    ));
                }
                Some("all") => {
                    steering.set_follow_up_mode(agent_core::QueueMode::All);
                    emit!(response(
                        id,
                        "set_follow_up_mode",
                        true,
                        Some(json!({ "mode": "all" })),
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
                emit!(response(
                    id,
                    "get_tree",
                    true,
                    Some(
                        json!({ "nodes": persistence.tree(), "leaf_id": persistence.active_ids().last() })
                    ),
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
                            &branch_cancel,
                        );
                        tokio::pin!(fut);
                        loop {
                            tokio::select! {
                                biased;
                                r = &mut fut => break r,
                                () = shutdown.wait() => {
                                    branch_stdin_open = false;
                                    branch_cancel.cancel();
                                }
                                maybe_line = lines.next_line(), if branch_stdin_open => match maybe_line {
                                    Ok(Some(l)) => {
                                        let l = l.trim();
                                        if l.is_empty() {
                                            continue;
                                        }
                                        let c: Value = match serde_json::from_str(l) {
                                            Ok(v) => v,
                                            Err(e) => {
                                                let _ = out_tx.send(response(None, "?", false, None, Some(&format!("invalid JSON: {e}"))));
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
                                    Ok(None) => {
                                        branch_stdin_open = false;
                                        branch_cancel.cancel();
                                    }
                                    Err(_) => {
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
                                    persistence.session_id(),
                                    &write_locks,
                                    &checkpoint,
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
                // Recorded into session context by default — pi's `recordBashResult`, opted out of via
                // `excludeFromContext` — so a diagnostic command run outside the model's own turn is
                // still visible to it on the *next* turn. Previously this command never touched
                // `session` at all: the calling client saw the result, but the model never did.
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
                            () = shutdown.wait() => {
                                stdin_open = false;
                                cancel.cancel();
                            }
                            update = prog_rx.next() => {
                                if let Some(ToolUpdate::Progress { id, name, snapshot, details }) = update {
                                    if let Some(frame) = event_frame(AgentEvent::ToolProgress { id, name, snapshot, details }) {
                                        let _ = out_tx.send(frame);
                                    }
                                }
                            }
                            maybe_line = lines.next_line(), if stdin_open => match maybe_line {
                                Ok(Some(l)) => {
                                    let l = l.trim();
                                    if l.is_empty() {
                                        continue;
                                    }
                                    let c: Value = match serde_json::from_str(l) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            let _ = out_tx.send(response(None, "?", false, None, Some(&format!("invalid JSON: {e}"))));
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
                                Ok(None) | Err(_) => {
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
                if !exclude_from_context {
                    session.push(agent_core::Message::user(format!(
                        "[Host bash command, run outside the model's own turn]\n$ {command}\n\n{}{result_text}",
                        if is_error { "(error)\n" } else { "" }
                    )));
                    if let Err(e) = persistence.persist(&session, None) {
                        eprintln!("serve: failed to persist host bash result: {e}");
                    }
                }
                emit!(response(
                    id,
                    "bash",
                    true,
                    Some(json!({ "result": result_text, "is_error": is_error })),
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
            other => {
                emit!(response(id, other, false, None, Some("unknown command")));
            }
        }
    }

    drop(out_tx);
    let _ = writer.await;
    Ok(())
}

/// Join the cached static system prompt with a freshly-computed dynamic footer (current date/cwd) —
/// the full prompt text an `Agent` should carry. Cheap: `static_system` is already-computed text and
/// `dynamic_footer` does no filesystem discovery, so this is safe to call every turn (see the `prompt`
/// arm's per-turn refresh) as well as at every `build_agent` rebuild.
fn full_system(static_system: &str, cwd: &std::path::Path) -> String {
    format!("{static_system}{}", crate::resources::dynamic_footer(cwd))
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
/// independently of `context_window`.
// 9 arguments, all independent inputs every call site already has on hand from `cfg`/local
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
    cache_key: &str,
    write_locks: &Arc<agent_core::WriteLockRegistry>,
    checkpoint: &Arc<dyn agent_core::CheckpointHook>,
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

    let mut agent = Agent::new(transport, model.to_string())
        .with_tools(build_tools(cfg))
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
    let policy = crate::policy::ToolPolicy::from_lists(&cfg.deny_tool, &cfg.deny_bash_pattern);
    if !policy.is_empty() {
        agent = agent.with_hooks(Arc::new(policy));
    }
    agent
}

/// The tool registry after `--tools`/`--exclude-tools`/`--no-tools` filtering — shared by every
/// `build_agent` rebuild and by the host-level `bash` RPC command (see [`serve`]), so excluding `bash`
/// from the model's own tool set also disables the host command rather than leaving a side door open
/// around an operator's explicit restriction.
fn build_tools(cfg: &ServeConfig) -> agent_core::ToolRegistry {
    let mut registry = tools::default_registry_with_prefix(
        cfg.bash_timeout_ms,
        cfg.bash_shell_path.as_deref(),
        cfg.bash_command_prefix.as_deref(),
    );
    tools::apply_filter(
        &mut registry,
        cfg.tools.as_deref(),
        cfg.exclude_tools.as_deref(),
        cfg.no_tools,
    );
    registry
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
fn resolve_model_scope(patterns: &[String], catalog: &[&str]) -> Vec<ScopedModel> {
    let mut resolved: Vec<ScopedModel> = Vec::new();
    for raw in patterns {
        let pattern = raw.trim();
        if pattern.is_empty() {
            continue;
        }
        let (base, thinking_level) = match pattern.rfind(':') {
            Some(idx) => {
                let suffix = &pattern[idx + 1..];
                match agent_core::ThinkingLevel::parse(suffix) {
                    Some(level) => (&pattern[..idx], Some(level)),
                    None => (pattern, None),
                }
            }
            None => (pattern, None),
        };
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
        } else if !resolved.iter().any(|m| m.id == base) {
            resolved.push(ScopedModel {
                id: base.to_string(),
                thinking_level,
            });
        }
    }
    resolved
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
        "reasoning": caps.reasoning_effort || caps.thinking != agent_core::models::ThinkingShape::None,
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

/// The concatenated text of a `User`-role message, for `get_fork_messages`'s candidate list — pi's own
/// `getUserMessagesForForking`. `None` for a message with no plain-text block at all (a pure
/// tool-result turn), which isn't a meaningful fork-point candidate to offer.
fn user_message_text(msg: &agent_core::Message) -> Option<String> {
    if msg.role != agent_core::Role::User {
        return None;
    }
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
    let context_usage = (session.last_input_tokens > 0).then(|| {
        json!({
            "tokens": session.last_input_tokens,
            "context_window": context_window,
            "percent": (session.last_input_tokens as f64 / context_window as f64 * 100.0),
        })
    });
    let breakdown = message_type_breakdown(session);
    json!({
        "steps": session.steps,
        "input_tokens": session.input_tokens,
        "output_tokens": session.output_tokens,
        "cache_read_tokens": session.cache_read_tokens,
        "cache_write_tokens": session.cache_write_tokens,
        "cache_write_1h_tokens": session.cache_write_1h_tokens,
        "reasoning_tokens": session.reasoning_tokens,
        "last_input_tokens": session.last_input_tokens,
        "context_usage": context_usage,
        "user_messages": breakdown.user_messages,
        "assistant_messages": breakdown.assistant_messages,
        "tool_calls": breakdown.tool_calls,
        "tool_results": breakdown.tool_results,
        "total_messages": session.messages.len(),
    })
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

/// The runtime-mutable settings and queue depth `get_state` reports — pi's `get_state` carries the
/// same shape (`thinkingLevel`/`autoCompactionEnabled`/`steeringMode`/`followUpMode`/
/// `pendingMessageCount`), so a client can render current settings and "N queued" without a second
/// round trip. Answerable from the process's own mutable state, not `&Session`, so — like
/// `session_stats` — it's available even mid-run (see the `prompt` arm's read-only-command handling).
fn runtime_settings(
    current_level: agent_core::ThinkingLevel,
    current_auto_compaction: bool,
    current_auto_retry: bool,
    steering: &agent_core::Steering,
) -> Value {
    json!({
        "thinking_level": current_level.as_str(),
        "auto_compaction": current_auto_compaction,
        "auto_retry": current_auto_retry,
        "steering_mode": match steering.steering_mode() {
            agent_core::QueueMode::OneAtATime => "one_at_a_time",
            agent_core::QueueMode::All => "all",
        },
        "follow_up_mode": match steering.follow_up_mode() {
            agent_core::QueueMode::OneAtATime => "one_at_a_time",
            agent_core::QueueMode::All => "all",
        },
        "pending_messages": steering.pending_count(),
    })
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
}

impl LiveStats {
    /// Seed from a session's current cumulative totals, so a `get_state`/`get_session_stats` answered
    /// one event into a brand-new turn still reflects everything before it, not a reset-to-zero count.
    fn from_session(session: &Session) -> Self {
        Self {
            steps: AtomicU32::new(session.steps),
            input_tokens: session.input_tokens.into(),
            output_tokens: session.output_tokens.into(),
            cache_read_tokens: session.cache_read_tokens.into(),
            cache_write_tokens: session.cache_write_tokens.into(),
            cache_write_1h_tokens: session.cache_write_1h_tokens.into(),
            reasoning_tokens: session.reasoning_tokens.into(),
            last_input_tokens: AtomicU32::new(session.last_input_tokens),
            pending_tool_ids: std::sync::Mutex::new(std::collections::BTreeSet::new()),
        }
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
    scanned <= 1 || scanned >= total || scanned % (total / 10).max(1) == 0
}

/// Build a `list_progress` frame — an unsolicited progress update for an in-flight `list_sessions`/
/// `list_all_sessions` scan, correlated to the eventual `response` frame via the same request `id`.
fn list_progress_frame(id: Option<String>, command: &str, scanned: usize, total: usize) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("list_progress"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    m.insert("scanned".into(), json!(scanned));
    m.insert("total".into(), json!(total));
    Value::Object(m)
}

/// Build an `auto_retry` frame — an unsolicited notice that a `prompt`'s run failed with what looks
/// like a transient error and is about to be automatically retried, correlated to the eventual
/// `response` frame via the same request `id`. Sent once per attempt, immediately before the backoff
/// sleep (not after — a client watching for retry activity shouldn't have to infer it from a gap).
fn auto_retry_frame(
    id: Option<String>,
    attempt: u32,
    max_attempts: u32,
    delay_ms: u64,
    error: &str,
) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("auto_retry"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("prompt"));
    m.insert("attempt".into(), json!(attempt));
    m.insert("max_attempts".into(), json!(max_attempts));
    m.insert("delay_ms".into(), json!(delay_ms));
    m.insert("error".into(), json!(error));
    Value::Object(m)
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
) -> Value {
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
    Value::Object(m)
}

/// Build a `session_info_changed` frame — an unsolicited push notice that the session's title changed,
/// correlated to the triggering `set_session_name` request via the same `id`. pi's own
/// `session_info_changed` event (`rpc-mode.ts:632-639`) lets a client learn the final *sanitized* title
/// without a follow-up `get_state`; `title` is `None` when the sanitized result was empty (a caller can
/// explicitly clear a title — see `sanitize_title`/`title_or_clear`).
fn session_info_changed_frame(id: Option<String>, title: Option<String>) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("session_info_changed"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("set_session_name"));
    m.insert("title".into(), json!(title));
    Value::Object(m)
}

/// Build a `response` frame.
fn response(
    id: Option<String>,
    command: &str,
    success: bool,
    data: Option<Value>,
    error: Option<&str>,
) -> Value {
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
    Value::Object(m)
}

/// Wrap an `AgentEvent` in an `event` frame, or `None` if it can't be serialized. Returning `None`
/// (and skipping the frame) rather than emitting `{type:"event"}` with no `event` field keeps a
/// serialization bug from putting a malformed frame on the wire that a client would silently mis-read.
/// A lightweight acknowledgement frame, emitted the moment a `prompt` is queued — before the model
/// turn(s) actually run — so a client can distinguish "received and starting" from the eventual
/// terminal `response` (which may be seconds away on a long tool-heavy run).
fn ack(id: Option<String>, command: &str) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("ack"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    Value::Object(m)
}

fn event_frame(ev: AgentEvent) -> Option<Value> {
    let event = serde_json::to_value(&ev)
        .inspect_err(|e| eprintln!("serve: failed to serialize agent event: {e}"))
        .ok()?;
    let mut m = Map::new();
    m.insert("type".into(), json!("event"));
    m.insert("event".into(), event);
    Some(Value::Object(m))
}

/// Write one newline-delimited frame to stdout and flush it.
async fn write_frame(out: &mut tokio::io::Stdout, line: &str) -> std::io::Result<()> {
    out.write_all(line.as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;

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
            .switch_branch(&agent, "some-id", false, true, None, &cancel)
            .await
            .expect_err("must fail clearly, not panic, when there's no session tree to navigate");

        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no session persistence configured"),
            "got: {err}"
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
            .fork(usize::MAX, None, false)
            .expect_err("must fail clearly, not panic, without a repo to fork within");

        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }
}
