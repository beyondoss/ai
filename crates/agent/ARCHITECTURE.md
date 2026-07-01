# Beyond Agent Harness — CLI & Tools Architecture

`beyond-ai-agent` (lib `beyond_ai_agent`, bin `beyond-ai-agent`) takes a task prompt or a stream of
NDJSON commands on stdin and turns them into a running coding agent: it drives
[`agent_core::Agent`](../agent-core/ARCHITECTURE.md) through a fixed set of ten tools (file I/O,
search, shell, Beyond-platform) and streams the model's text and tool activity back out — to stdout
for a one-shot `run`, or as NDJSON event frames for a headless `serve` session. It holds no provider
keys and makes no provider-specific decisions; all model traffic is one HTTP POST per turn to a
Beyond gateway, authenticated with a `bai_v1` virtual key (or a BYO key the gateway forwards as-is).

This crate is the "everything above the wire" layer. `agent_core` owns the message model, the loop,
and the two seams (`Tool`, `ModelTransport`); this crate is the concrete `Tool` implementations, the
CLI, and the `serve` control protocol built on top of them.

## Beyond the basics

The harness layers several capabilities over the bare tools + loop:

- **Trust** ([`trust_store`](src/trust_store.rs)) — a tri-state, ancestor-inheriting allowlist
  (`~/.claude/trusted-projects.json`: `{trusted: [...], untrusted: [...]}`, most-specific directory
  wins, untrusted checked first at each level) gates the project-local `SYSTEM.md`/`APPEND_SYSTEM.md`
  overrides and discovered skills/prompt templates (both `discover` calls take `project_trusted` as a
  required parameter — `skills::discover`/`prompts::discover` — precisely so a call site can't forget the gate;
  the `get_commands` listing and `/skill:name`/`/name` invocation inherit it from there) — all only
  honored once the working directory is trusted, via `agent trust <path>` (persistent) or
  `--trust-project`/RPC (session-scoped). `agent clear-trust <path>` removes a directory's own
  trust/untrust entry without recording a new one (`TrustStore::clear`) — unlike `trust`/`untrust`,
  which always leave it pinned to its own explicit grant or denial, this reverts it to inheriting
  whatever its nearest ancestor decides, matching pi's own `ProjectTrustStore::setMany` accepting a
  `null` decision to delete an entry. A legacy bare-array trust file still parses (trusted-only),
  migrated to the tri-state shape on the next `trust`/`distrust`/`clear-trust` call.
  **Not** gated by trust: `AGENTS.md`/`CLAUDE.md` project-instruction files
  (`resources::load_context_files` takes no `project_trusted` parameter at all) — matching pi's own
  `resource-loader.ts::loadProjectContextFiles`, which has no trust check either. Whether project
  context files *should* be trust-gated is a separate design question both codebases currently answer
  the same way (no); this is a factual note about current behavior, not a parity gap.
- **System-prompt assembly** ([`resources`](src/resources.rs)) — split into a **static** half (base
  identity, overridable by an on-disk `SYSTEM.md`: project `.claude/` when trusted, else user; then an
  on-disk `APPEND_SYSTEM.md` — same project-then-user discovery/trust order, but *additive* rather than
  a replacement, appended right after the base/override — consulted only when the caller didn't already
  supply an explicit `append` (e.g. `--append-system-prompt` wins outright over the on-disk file rather
  than combining with it, matching pi's `resource-loader.ts`); project instruction files, global then
  cwd→root, **one file per directory**, `AGENTS.md` winning over `CLAUDE.md`, matched case-insensitively;
  discovered [`skills`](src/skills.rs) via `<available_skills>`, read-on-demand) and a **dynamic** footer
  (current **local** date + cwd). `serve` caches the static half — rebuilt only at startup and on
  `set_model`/`set_thinking`/an explicit `reload` — and refreshes just the cheap dynamic footer before
  every `prompt`, so a long-running process doesn't re-walk the filesystem every turn just for the date.
  CLI flags: `--system-prompt` (replace), `--append-system-prompt`, `--no-context-files`. Skills are
  discovered recursively (`SKILL.md` at any depth, following symlinked directories — matching pi's own
  `skills.ts`, so a shared skills library symlinked into `.claude/skills` isn't invisible; a symlink
  loop is caught by `walkdir`'s own detection, not an unbounded walk), honoring `.gitignore`/`.ignore`
  (`WalkBuilder`'s own defaults) and `.fdignore` (registered explicitly via
  `add_custom_ignore_filename` — the third file `skills.ts::IGNORE_FILE_NAMES` lists, which the
  `ignore` crate doesn't honor by default); a `disable-model-invocation`
  skill is omitted from the listing but still
  reachable by an explicit `/skill:name`, which strips the raw YAML frontmatter and wraps the body in a
  `<skill name="..." location="...">` tag rather than leaking the frontmatter verbatim. Both
  skill/prompt-template discovery report name collisions (`discover_with_diagnostics`) — the same name
  shadowed across roots or within one root — surfaced via `get_commands`'s `collisions` field *and*
  `tracing::warn!`-logged at the point of detection, so an operator watching server logs notices a
  shadowed skill/template without a client having to proactively call `get_commands` and inspect it.
- **Prompt templates** ([`prompts`](src/prompts.rs)) — a `/name args` prompt is expanded from a
  `.claude/prompts/*.md` template with bash-style substitution before it reaches the model: quote-aware
  arg splitting, `$N` for any positional, `${@:N}`/`${@:N:L}` slices, and `${N:-default}`. A template's
  `description` (frontmatter, else first body line) feeds autocomplete.
- **Session persistence** ([`session_store`](src/session_store.rs)) — append-only JSONL with a header
  carrying a collision-resistant id + metadata; a turn appends only its new messages (compaction
  rewrites atomically, folding the compacted prefix into a durable `Entry::Compaction` record —
  `tokens_before`/`folded_ids`/`summary` — rather than only the flat `compactions`/`dropped_messages`
  counters), every write is `fsync`ed for durability, a corrupt/torn line anywhere in the file (not just
  the last one) is skipped rather than aborting the whole read, and a header whose `version` is newer
  than the build understands is refused (migration hook). `rewrite`'s write-to-`.jsonl.tmp`-then-rename
  is safe either way a failure lands: a hard crash leaves the `.tmp` behind harmlessly (never read back
  as a session; the next `rewrite` reuses the same deterministic path anyway), while a genuine
  in-process error (disk full, a permission error) mid-write is caught by a drop guard that removes the
  half-written `.tmp` right then, since the process is still alive to do it. `--session-dir` opens a
  multi-session
  `SessionRepo` (list-with-metadata/create/open/soft-delete-to-`.trash`/fork); `--session-file` is the
  single-session form; neither flag defaults to `~/.claude/sessions/<encoded-cwd>/` rather than silent
  in-memory-only (`--no-session-persistence` opts out explicitly). Every cwd this module ever records
  into or matches against a session — `serve`'s own startup reattach, `run --continue` — is passed
  through `canonical_cwd` first (resolves symlinks/`.`/`..`, drops a trailing separator), so a project
  reached through a symlink one time and its real path another still matches the same session instead
  of silently fragmenting into two; a path that can't be resolved (removed out from under the process)
  falls back to matching by the string as given, same as before this existed. `list` carries derived
  `updated_at`(the newest stamped `Entry::Message::timestamp` found in the file, not the file's OS
  mtime — a copy/restore/sync that doesn't preserve mtime exactly, or a wrong one, no longer makes a
  session look stale or falsely fresh; mtime is only a fallback for a legacy file with no stamped
  message at all)/`message_count`/`preview`(first user message, truncated)/`search_text`(every user *and*
  assistant message in the session — matching pi's own `allMessagesText`, so a session is findable by
  something only the assistant said — space-joined and capped at 2,000 chars — a broader substring-match
  surface than `preview` alone) without opening each transcript fully beyond the single streaming scan
  `list` already does, and resuming without an explicit session id picks the newest session matching the
  **current cwd**, not just the globally newest. These four fields are `#[serde(skip)]` on `SessionMeta`
  itself (so a stale scan value can never leak into the on-disk header, which reuses the same
  `Serialize` impl) — `SessionMeta::to_listing_json` re-inserts them for a listing response; `serve`'s
  `list_sessions`/`list_all_sessions` use it instead of a bare `serde_json::to_value`, which would
  otherwise silently omit all four. `list`/`list_all`'s per-file scan (`read_listing`, called once per
  session file) is pure I/O with no cross-file dependency, so `list_with_progress`/
  `list_all_with_progress` fan it out across a small `std::thread::scope` worker pool
  (`available_parallelism`, capped at one worker per file; a one- or two-file listing just runs inline —
  no pool to justify the setup) rather than scanning one file at a time, and invoke `on_progress(scanned,
  total)` once per file so `serve` can put a live "scanning…" indicator on the wire for a listing large
  enough to take a moment; `list`/`list_all` are the same scan with a no-op progress callback.
- **Tree-shaped history** — every message line also carries an `id`/`parent_id` (additive,
  `#[serde(default)]`; a pre-tree file's absent fields are migrated to synthesized, chained ids in
  memory only, never persisted back). The "active path" (`Session.messages`) is the `parent_id` chain
  from root to tip; a `Leaf` entry (`SessionStore::switch_active`) redirects the tip append-only —
  navigating never deletes anything. Compaction (`rewrite`) stays destructive to the active path's own
  entries but now preserves every node on some _other_ branch, writing them before the fresh
  compacted-path entries so "the last message in the file" still resolves to the tip without needing a
  `Leaf` marker. `abandoned_by_switch` computes what a hypothetical switch would abandon (via a
  common-ancestor walk); `switch_active_with_summary` attaches the summary as a real message at the
  branch's new tip and updates in-memory state immediately, so the summary actually reaches the model on
  the next turn (not just a persisted-but-inert record). `list_branches` reports every **leaf** plus the
  active tip; `get_tree`/`SessionStore::tree()` reports **every node** on every branch (id, parent,
  role, a short preview of its own text) for a client that wants the full picture, not just the leaves.
  `SessionRepo::fork_at_entry` forks from any of those nodes directly — on or off the *current* active
  path — unlike `fork`'s own `upto`, a message-count prefix of just the active path; a client that wants
  to fork an abandoned branch no longer has to `switch_active`/`switch_branch` to it first just to make
  it forkable. `before: bool` excludes the target entry itself from the copied prefix (fork right before
  it) instead of including it (the default).
- **Branch-local model/thinking-level** — `set_model`/`cycle_model`/`set_reasoning_effort`/
  `cycle_thinking_level` each append an O(1) `Entry::ModelChange`/`Entry::ThinkingLevelChange` record
  anchored to the *current tip* (`SessionStore::record_model_change`/`record_thinking_level_change`) —
  "everything appended after this message uses X" — rather than only mutating in-process state.
  `switch_branch`'s RPC handler queries `model_at`/`thinking_level_at` for whatever it's switching *to*
  and restores that (rebuilding the `Agent` if it differs from what's currently active), so a
  `set_model` made on one branch doesn't leak backward into an earlier point being switched back to; the
  model always resolves to something real (falling back to the session's own creation-time model if
  nothing was ever recorded), the thinking level has no such baseline and is left alone when absent.
  Deliberately *not* a full tree participant the way messages are (a change record never becomes the
  active tip, and doesn't redirect what a new message's `parent_id` chains off) — a known, documented
  edge case: two sibling branches that both grow from the same anchor point share one lookup entry, so a
  query against a descendant on the second branch can see a stale change actually made on the first.
  `switch_branch`'s own restoration is unaffected (it only ever queries the target being switched to,
  never one of its descendants); the fuller fix (threading these through the same per-branch chain
  messages use) is real but not warranted by any concrete case found so far.
- **Expanded `serve` surface** — beyond `prompt`/`get_state`/`get_messages`/`new_session`: `abort`,
  `stop_after_turn` (graceful — the current turn's tool calls still finish and commit; only the *next*
  model call is skipped — see `agent_core::Steering::request_stop`), `steer` (mid-run, folded onto the
  next tool turn) / `follow_up` (queued for the next stop boundary, and — unlike a bare
  `steer`/`follow_up` — also accepted while idle, queuing against a persistent handle for whichever
  `prompt` runs next; `stop_after_turn` sent while idle is instead a no-op ack, since there is no future
  run it could safely bind to), `compact`,
  `list_sessions`/`list_all_sessions`(cross-project — every project's own session directory under the
  shared root, not just this one's; each streams unsolicited `list_progress` frames, correlated to the
  request's own `id`, while its parallel scan is in flight — throttled to roughly ten frames regardless
  of how many sessions exist, deterministically by `scanned`'s value so it stays stable despite the
  concurrent scan)/`switch_session`/`fork`/`get_fork_messages`(read-only fork
  preview)/`set_session_name`(an O(1) `Entry::TitleChange` append, not a rewrite of the whole
  file — a rename used to cost rewriting every message already on disk just to update the header's
  `title` field; whole-session-scoped rather than branch-scoped like `ModelChange`/
  `ThinkingLevelChange`, so the most recent rename anywhere in the file wins regardless of which
  branch is active, matching pi's own `session_info` entries)/`export_html`(render the active session's transcript as one
  self-contained HTML file, no server or client involved once written — plus every abandoned branch
  (`SessionStore::abandoned_branches`), each as its own labeled section after the main transcript,
  rendering only the part that diverges from the active path so the shared prefix isn't duplicated),
  `get_last_assistant_text`/`get_session_stats`/`get_commands`(+ collision diagnostics)/`reload`,
  `set_model`/`set_thinking`(raw budget override)/`set_reasoning_effort`(the portable
  `agent_core::ThinkingLevel`, correct for whichever mechanism the active model actually
  uses)/`cycle_model`(steps through `--models`'s scoped list when given — comma-separated ids, in order
  — else the full `get_available_models` hint list; reports `scoped: bool` so a client knows which)/
  `cycle_thinking_level`(advances the same portable level)/`set_auto_compaction`/
  `set_auto_retry`(toggle `agent_core::Agent::with_auto_retry` — off surfaces an otherwise-retried
  mid-stream failure on the first attempt, for debugging a flaky connection)
  (rebuild the `Agent` for subsequent prompts) / `get_available_models`, `list_branches`/`get_tree`/`switch_branch`
  (navigate the session's tree, optionally summarizing the abandoned branch first — `get_messages` tags
  each message with its tree `id` so a client can name any point as a `switch_branch` target, not only a
  branch's leaf; `get_messages`'s optional `since` — a tree id the client already has — returns only
  what was appended after it, pi's own `get_entries({since})`, so a polling client doesn't have to
  re-transfer the whole transcript every time; an unmatched `since` is an error, not a silent full
  re-fetch), and `bash`/`abort_bash` (run a host shell command directly, independent of the
  model's own tool loop). A `prompt` emits an immediate `ack` frame the moment it's queued, and its
  terminal `response` reports `refused: bool` (a refusal is a distinct terminal condition — it doesn't
  drain queued steering); a `prompt` sent while another is in flight can carry
  `streaming_behavior: "steer"|"follow_up"` to be accepted and queued instead of rejected as busy. A
  `prompt` runs concurrently with stdin so `abort`/`stop_after_turn`/`steer`/`follow_up` land mid-turn.
- **Whole-run auto-retry** — once `agent_core`'s own mid-stream retry (inside one model turn) is
  exhausted, a `prompt` that still ends in a transient-looking `Err`
  (`agent_core::agent::is_retryable_mid_stream`, exposed `pub` for exactly this — the classification is
  identical one layer up) is automatically re-invoked against the *same* session — resuming from
  wherever it left off, not restarting the turn or re-appending the user message — up to 3 more times
  with exponential backoff (2/4/8s, capped at 30s; pi's `agent-session.ts` equivalent). Gated by the
  same `current_auto_retry`/`set_auto_retry` flag as the mid-stream layer (one user-facing "retries
  on/off" concept spanning two internal layers); never retries past a client disconnect or an explicit
  `abort` (redundant with `is_retryable_mid_stream` already excluding `Error::Cancelled`, but checked
  explicitly too). Each attempt persists whatever it produced (even a failed one may have committed a
  tool round-trip before erroring) before deciding whether to retry, and emits an unsolicited
  `auto_retry` frame just before its backoff sleep so a client can distinguish "still working, retrying
  a hiccup" from silence. The backoff sleep itself isn't raced against `abort`/shutdown (both are only
  read inside an attempt's own event loop) — bounded by the retry cap and backoff ceiling, so the
  unresponsive window is short, not gapless.
- **Stale-cwd detection** — `cwd_is_stale` compares a session's recorded `cwd` against reality: the
  directory no longer exists, or (since `switch_session`/`fork`/reattaching never change the process's
  actual working directory) it simply isn't where this process is running. Surfaced as `cwd_stale` on
  `ready`, `get_state`, `new_session`, `switch_session`, and `fork` — the points a client picks up a
  session — rather than silently letting the model's tools run against a mismatched or missing
  directory.
- **Tool set filtering** — `--tools`/`--exclude-tools`/`--no-tools` (both `run` and `serve`) restrict
  or drop from the default registry before it's advertised to the model; the auto-generated system
  prompt's tool list (`default_system_prompt`) reflects the filtered set, not the full default one, so a
  restricted agent never sees itself advertised a tool it doesn't actually have.
- **Multimodal** — `prompt` accepts `images: [{media_type, data}]` (base64), built into a multimodal
  user turn. `read` on an oversized image file downscales/re-encodes it (Lanczos3, PNG-then-JPEG
  re-encode, Exif orientation correction via the `image` crate's own generic
  `ImageDecoder::orientation`/`DynamicImage::apply_orientation` — correct for every format that can
  carry Exif, JPEG and WebP included, with no per-format parser of our own to maintain) to fit a
  4.5 MB base64 budget rather than refusing outright, and sniffs the real magic bytes to recover from
  a mislabeled extension.

---

## Data Flow

### `run` — one-shot CLI

```
beyond-ai-agent run "<task>"
   │
   ▼
expand_if_skill_invocation → expand_if_slash (same expansion `serve`'s "prompt" handler applies —
   │                          a `/skill:name` or `/name args` first message/arg is expanded before
   │                          the model ever sees it; project-local skills/templates gated by
   │                          `project_trusted` same as `serve`)
   ▼
Session::new() + user(expanded task)
   │
   ▼
Agent::run[_events] (agent_core loop; see agent-core/ARCHITECTURE.md for the loop itself)
   │  each step, text mode (default) — Agent::run, StreamEvent only:
   │   ├─ POST one model turn → gateway → provider ──── network/4xx/5xx ──► Error::Transport, exit ≠0
   │   ├─ StreamEvent::TextDelta   ───────────────────────────────────────► stdout (printed live)
   │   ├─ StreamEvent::ToolUseStart ──────────────────────────────────────► stdout "\n[tool: name]"
   │   └─ assistant turn carries tool_use blocks → ToolRegistry.get(name)
   │        │
   │        ├─ found    → tool.run(input) → Ok(text) / Err(ToolError) ──► tool_result (is_error?)
   │        └─ not found → "unknown tool: <name>" ───────────────────────► tool_result, is_error=true
   │  each step, `--json` — Agent::run_events, the full AgentEvent surface (same shape `serve` streams):
   │   └─ every AgentEvent (AgentStart/TurnStart/Stream(..)/ToolStart/ToolProgress/ToolEnd/TurnEnd/
   │        Compacted/AgentEnd/Error) ──► stdout, one `serde_json::to_string`-d object per line, flushed
   │        immediately; a `{"kind":"session", id, model, cwd}` header line precedes the first event
   │
   ▼ (model ends its turn without a tool_use, or session.steps == max_steps)
stdout: trailing newline (text mode only — `--json` has no extra trailing line, just the event stream)
stderr: "[done in N step(s); X in / Y out tokens]"   (or the propagated Error::MaxSteps / Error::Transport)
```

### `serve` — headless NDJSON control protocol

```
stdin (one JSON command per line)              stdout (one JSON frame per line, single writer task)
  │                                                   ▲
  ▼                                                   │
serve() boot: open persistence (file/dir/none)        │
  │            + build static system prompt            │
  ▼                                                    ├── {"type":"ready", session_id, model, cwd_stale}
loop over stdin lines ──────────────────────────────► │
  │                                                    │
  ├─ {"type":"prompt", message, streaming_behavior?} ─► {"type":"ack", command:"prompt"}   (immediate)
  │     agent.set_system(static + fresh dynamic footer)
  │     session.user(message)                          event* (Stream / ToolStart / ToolProgress /
  │     Agent::run_events_steered(session, sink, ..)             ToolEnd / TurnEnd / Steered)
  │     persist (rewrite/rewrite_compacted as needed)   response{command:"prompt", success,
  │                                                              data:{steps, tokens…, refused: bool}}
  │     (busy + streaming_behavior set → queued via Steering, acked, no ack/event/response above)
  │
  ├─ {"type":"get_state"} ───────────────────────────► response{data:{session_id, model, steps,
  │                                                              message_count, input/output_tokens,
  │                                                              thinking_level, auto_compaction,
  │                                                              auto_retry, queue_mode,
  │                                                              pending_messages}}
  ├─ {"type":"get_messages", since?} / {"type":"get_tree"} ──► response{data:{messages:[...]}} /
  │     (since: only what's new)                                 response{data:{nodes:[TreeNode…]}}
  ├─ {"type":"new_session"} ─────────────────────────► response{data:{session_id, parent}}  (fresh
  │     steering.clear()                                        Session; persisted per the open mode;
  │                                                              `parent` — repo mode only — is the
  │                                                              lineage marker below)
  ├─ {"type":"bash", command, cwd?, timeout_ms?} ─────► tool_progress*/tool_end event frames, then
  │                                                              response{command:"bash", success, data}
  └─ invalid JSON / unknown "type" ──────────────────► response{success:false, error}

See `serve.rs`'s own module doc for the full command list (session/branch nav, model/thinking/
auto-compaction tuning, skill/prompt discovery, `reload`, `abort`/`abort_bash`) — this diagram shows
the shape, not every command.

stdin EOF  →  out_tx dropped → writer drains queued frames → process exits Ok(())
stdout write fails (broken pipe) → writer task exits its loop → next emit! observes a closed
                                     channel → main loop breaks → process exits Ok(())
```

A `prompt`'s tool calls are NOT shown above as a separate fan-out: `agent_core::Agent::run_events_steered`
runs the tools the model batched in one turn **concurrently but bounded** (grouped by write-target so
same-path calls serialize, then `buffer_unordered` over ≤8 groups — see `agent-core/ARCHITECTURE.md`),
emits `ToolStart` for all of them up front (call order), then streams each call's own `ToolProgress`/
`ToolEnd` the instant it's known — a client sees completions in actual finish order, not batched after
the slowest call in the group — while the *persisted transcript* is still rebuilt in call order
afterward, so on-disk determinism is unaffected by which tool actually finished first. Same-path write
exclusivity extends across the whole `serve` process via `write_locks: Arc<WriteLockRegistry>`, shared
by every `build_agent` rebuild (`set_model`/`set_thinking`/`cycle_model`/`cycle_thinking_level`/
`set_auto_compaction`/`set_auto_retry`) — not just re-derived per `Agent`.

---

## Concepts & Terminology

| Term                        | What It Controls                                                                                 | NOT                                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Tool**                    | A registered capability the model invokes by name + JSON input (`agent_core::Tool` impl)         | Not necessarily a subprocess — only `bash`/`fork`/`sync`/`logs` shell out; the rest touch the filesystem in-process             |
| **`ToolRegistry`**          | The name → `Tool` map advertised to the model every turn (`default_registry_with(..)`, then `tools::apply_filter`'d by `--tools`/`--exclude-tools`/`--no-tools`) | Not per-session or hot-reloadable at runtime — filtering happens once when the process's registry is built, not per turn |
| **`CommandRunner`**         | The seam between a tool and real process execution (`exec.rs`)                                   | Not a sandbox — `RealRunner` execs a resolved real `bash` (falling back to `sh`) / `beyond …` with the host process's full ambient privilege |
| **`TrustStore`**            | Tri-state (trusted/untrusted/unknown), ancestor-inheriting allowlist gating `SYSTEM.md`/skill/prompt-template access | Not a sandbox or a permission system for tools (only gates *which instructions the model sees*, not what it can do) — and not `AGENTS.md`/`CLAUDE.md` project-instruction files either, which have no trust gate at all (matching pi) |
| **`Session`**               | Message history + step/token counters; optionally `serde`-persisted (single file or a `SessionRepo`) | Not multi-session by itself — one `serve` process holds exactly one *active* `Session`, though a repo can hold many on disk |
| **`Steering`**               | The two-lane queue (`steer`/`follow_up`) a client injects mid-run or at a stop boundary, plus a graceful stop-request flag (`request_stop`/`stop_after_turn`); persistent across a whole `serve` process, not just one `prompt` call | Not drained on a refusal (a distinct terminal condition) — queues cleared only on `new_session`/`switch_session`/`fork`/`switch_branch`; the stop flag is *also* always cleared when the run it was set on returns, however it ends, so it can never bind to a later, unrelated run |
| **`AgentEvent`**            | `Stream`/`ToolStart`/`ToolProgress`/`ToolEnd`/`TurnEnd`/`Steered` boundaries streamed as `event` frames during a `prompt` | Not the terminal answer — the `response` frame (success/data/error) is separate and always comes last, after an immediate `ack` frame |
| **Virtual key** (`bai_v1…`) | The bearer token this crate forwards to the gateway on every request                             | Not verified or interpreted here — Ed25519 signature check and deny-set live entirely in the gateway                            |
| **`max_steps`**             | Ceiling on loop iterations per `run` invocation, or per `prompt` in `serve` (default `agent_core::DEFAULT_MAX_STEPS`, 50) | Not a token budget — `max_tokens` (seeded model-aware by `agent_core`'s `Agent::new`, ≥4096/turn) has no CLI flag in this crate; also not a permanent dead end — `Error::MaxSteps` is resumable with a fresh call |
| **`HARD_CAP`** (grep/find)  | OOM guard: walk quits once 10,000 matches/paths are collected, before `limit` truncation runs    | Not the reported limit — `limit` (default 100/1000) is the user-facing cap; `HARD_CAP` is a backstop far above it               |

---

## Core Mechanism

### Tool dispatch

`tools::default_registry_with(bash_timeout_ms)` assembles the fixed, hard-coded set of ten tools, then
`tools::apply_filter(&mut registry, tools, exclude_tools, no_tools)` optionally restricts it — an
allow-list (`--tools`), a deny-list (`--exclude-tools`), or `--no-tools` (wins outright, a
pure-conversation run) — before it's ever advertised to the model; there is still no *per-turn* or
hot-reloadable configuration, filtering happens once when the process's registry is built. Each `Tool`
is a stateless (or `Arc<dyn CommandRunner>`-holding) value; `agent_core::Agent::run_events_steered`
looks tools up by the name the model used and calls `run(input)`, converting `Err(ToolError)` into an
error `tool_result` rather than aborting the run (see `agent_core/ARCHITECTURE.md`). That error/success
split is what each tool actually produces:

| Tool                 | `ToolError::InvalidInput` when…                                                                                                      | `ToolError::Execution` when…                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `read`               | missing `path`; offset past EOF; image cannot be downscaled to fit the 4.5 MB base64 budget                                          | file unreadable (missing, permission denied) — non-UTF-8 bytes decode lossily, not an error |
| `write`              | missing `path`/`content`                                                                                                             | `mkdir`/`write` syscall fails                                                               |
| `edit`               | `old_string` matches 0 or >1 times after exact+fuzzy (without single-edit `replace_all`); overlapping/no-op edits; malformed `edits` | file unreadable/unwritable (writability is checked *before* any match/diff work — pi's own `access(path, W_OK)` pre-check — so a read-only file fails fast rather than after paying for fuzzy-matching's NFKC normalization) |
| `ls`                 | —                                                                                                                                    | `read_dir` fails                                                                            |
| `grep`/`find`        | bad regex / bad glob                                                                                                                 | the `spawn_blocking` task itself panics (walk failures are swallowed per-entry)             |
| `bash`               | missing `command`                                                                                                                    | spawn failure, or timeout (`timed_out`)                                                     |
| `fork`/`sync`/`logs` | missing `app` (fork only)                                                                                                            | `beyond` spawn failure, timeout, or non-zero exit                                           |

### File tools — direct, synchronous, in-process

`read`, `write`, `edit`, and `ls` call `std::fs` directly inside their `async fn run` — no
`spawn_blocking`, no async I/O crate. A single stat/read/write syscall is sub-millisecond, so the
cost of a thread hop would dominate the work itself; these tools accept blocking the async task
briefly rather than pay that overhead, consistent with "do less work" over reflexive parallelism.

- **`read`** numbers every line (`{lineno:>6}\t{line}`), 1-based `offset`, default `limit` 2000,
  appends a bracketed `[showing lines A-B; use offset=N to continue]` marker (the same
  `output::marker` bracket convention every truncating tool now shares) if more lines remain. It streams
  line-by-line and caps each line at `MAX_LINE_BYTES` (4000) — bytes past the cap are drained but not
  stored, so one pathological single line (a minified bundle) can't balloon memory; a capped line gets
  a `"… [line truncated]"` marker — a **mid-line** clip, deliberately not pi's own `truncateHead`
  (`truncate.ts`), which "never returns partial lines": there, a line that alone exceeds the byte budget
  is either dropped from the output entirely, or — if it's the very first line — the *whole read* comes
  back empty. Showing a clipped prefix instead means a model reading a single-giant-line file always
  sees *something*, rather than nothing it can't distinguish from an empty file. Line bytes decode
  lossily (`from_utf8_lossy`), so a non-UTF-8 file reads with replacement chars rather than erroring.
  Every "N lines total" this tool reports (e.g. an
  offset-past-EOF error) counts real content lines from its own line-at-a-time scan, independent of
  whether the file ends with a trailing newline — a deliberate divergence from pi's own `read.ts`
  (`textContent.split("\n").length`, which in JS yields a trailing empty-string element whenever the
  file ends with `\n`, over-counting by one only in that case); matching that inconsistency isn't a
  parity goal worth having. An **image** file skips text decoding entirely: sniffing the real magic
  bytes (`image::guess_format`) always decides routing (an extension is only the fallback when
  sniffing can't identify a format at all), and the reported format comes from that same sniff — a
  mislabeled extension still reports its true type. It's returned as a
  base64 `ImageSource` attachment the multimodal model can see; one already under a 4.5 MB base64
  budget goes out as its original bytes/format unmodified, an oversized one has its Exif orientation
  applied first (`image::ImageDecoder::orientation`/`DynamicImage::apply_orientation` — generic across
  every format that can carry it, JPEG and WebP included), is downscaled (Lanczos3, max 2000px), and
  re-encoded — a lossless PNG is tried first (a downscaled screenshot/diagram/text-heavy image often
  already fits that way), falling back to JPEG (quality 80) only if the PNG doesn't fit, stepping
  dimensions/quality down further if still over budget — only refused outright if even the smallest
  re-encode can't fit.
- **`write`** creates parent directories (`create_dir_all`), then writes **atomically** (sibling temp
  file + `rename`, shared with `edit` via `tools::write_atomic`) so a kill mid-write can't leave a
  half-written file; always overwrites.
- **`edit`** accepts either an `edits: [{old_string,new_string}]` array (applied in order) or a
  single `old_string`/`new_string` pair. Each `old_string` must match **exactly once** in the current
  content unless it's a single-edit call with `replace_all: true` — uniqueness is the only safety
  check; there is no diff/dry-run, the file is rewritten on success. Writability is checked before any of
  that match/diff work runs — a read-only file fails fast rather than after paying for fuzzy-matching's
  NFKC normalization (pi's own `access(path, W_OK)` pre-check). Matching tries an **exact** hit
  first, then a normalized **fuzzy fallback** (NFKC + folding smart quotes/dash family/unicode spaces +
  per-line trailing-whitespace), with hits mapped back to original byte offsets — so a model's
  `old_string` carrying a curly quote, em-dash, nbsp, or stray trailing space still lands instead of
  failing "not found". The fold table (`fold_char`) is a deliberate **superset** of pi's own
  `normalizeForFuzzyMatch`, not a port of it — it additionally folds prime marks (`′`/`″`) toward
  quotes, small/fullwidth dash forms, and a couple more unicode space code points — every addition the
  same kind of narrowly-scoped, single-family confusable pi's own set already folds, so it only helps a
  slightly-off `old_string` land and can never conflate two genuinely different characters. Edits
  resolve to byte ranges against the _original_ text (order-independent, overlap-rejected).
- **`ls`** sorts directories first, then alphabetically; dot-entries are hidden unless `all: true` —
  both deliberate divergences from the reference agent, kept rather than "fixed" to match it (dotfiles
  hidden by default cuts real noise like `.git`/editor swapfiles; directories-first is a UX improvement
  independent of parity). Caps the listing at `limit` (default 500, overridable) so a
  `node_modules`-sized directory can't flood context, appending a bracketed `[N more entries; M total —
  narrow with a subpath or use find/grep]` marker when it truncates.

### Search tools — gitignore-aware tree walks, deterministic output

`grep` and `find` both walk with `ignore::WalkBuilder` (the crate behind ripgrep/fd — respects
`.gitignore`, `hidden(false)` includes dotfiles) and run their walk inside `tokio::task::spawn_blocking`
because, unlike the file tools, their cost scales with the size of the tree, not a single syscall.

They diverge on parallelism, and the divergence is **measured, not assumed**:

| Tool   | Walk strategy                                       | 5,000-file benchmark (`benches/search.rs`)                                                                             |
| ------ | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `grep` | `ignore::WalkParallel` (`threads: 0` = ≈ CPU count) | single-threaded: **29.1 ms** (171 Kelem/s) → auto-threaded: **7.68 ms** (651 Kelem/s) — **3.8× faster**                |
| `find` | sequential `WalkBuilder::build()`                   | **3.10 ms** (1.61 Melem/s) sequential only — the module doc records that a parallel version regressed ~2× on this tree |

`grep`'s per-file work (read the file, regex-scan every line) is expensive enough that overlapping
files across threads wins outright. `find`'s per-file work is a single glob test against a path —
cheap enough that `WalkParallel`'s thread-coordination overhead costs more than it saves, so it stays
sequential. This is the Theory-of-Constraints rule applied literally: parallelize the walk only where
the walked work, not the traversal, is the bottleneck.

`find` also disambiguates what it's matching against based on the pattern: a pattern with no `/`
(e.g. `*.rs`) matches against the basename only; a pattern containing `/` matches the full path, with
a `**/` prefix implicitly added unless the pattern already starts with `**/` or `/` — so
`src/*.rs` actually matches `**/src/*.rs` (any `src/` directory at any depth), not just a top-level one.

Both tools collect into a `Vec`, **sort by path** (`grep` then by line), and `truncate(limit)` —
so which results survive a `limit` cutoff is the lexicographically-smallest set, not whatever order
threads happened to finish in (`grep.rs:Lab — output_is_path_sorted_and_deterministic` and the
matching `find` test assert this byte-for-byte across repeated runs). A `HARD_CAP` of 10,000
collected items is an OOM guard for pathological patterns; if it trips, the _output_ still gets
sorted+truncated deterministically, but which items entered the collected set before the cap tripped
can vary — the tool flags this in its trailing `"… limit reached"` line either way.

`grep` additionally clips any single match line to 500 bytes at a UTF-8 char boundary
(`clip()` in `grep.rs`) so one absurdly long line can't blow the model's context. It also takes
context-line params — `context` (both sides, like ripgrep's `-C`) or `before`/`after` per side, each
clamped to `MAX_CONTEXT` (100) — emitting surrounding lines flagged as context vs match; only matches
count toward `limit` and `HARD_CAP`. A large `before`/`after` window on a match-dense file can still
blow past any sane size well before `limit` matches are reached, so `grep` additionally caps the whole
rendered output at `MAX_OUTPUT_BYTES` (50 KiB), truncating at a char boundary and appending its own
marker; the byte cap is checked *before* the match-count marker is appended and wins outright when both
would fire (appending the count marker first and then truncating could otherwise slice through, and
silently corrupt, the marker it just added).

### Shell tools — a shared `CommandRunner` seam

`bash` and the Beyond tools (`fork`/`sync`/`logs`) don't touch the filesystem directly; they both
go through `tools::exec::CommandRunner`, implemented for production by `RealRunner`
(`tokio::process::Command` with `kill_on_drop(true)`, a process-group leader — via a `GroupKillGuard`
RAII drop guard so a backgrounded grandchild is killed on *either* a timeout *or* the caller dropping
the future (cancellation) — and **bounded streaming capture** for the non-`bash` runner path). This is
the same seam pattern as `agent_core`'s `Tool`/`ModelTransport` traits — it's what lets
`fork_builds_argv`/`sync_builds_argv`/etc. assert the exact argv without a live Beyond control plane,
and `serve`'s own `bash`/`abort_bash` RPC commands reuse `RealRunner` directly, outside the model loop.

- **`bash`**: runs through a **resolved real `bash`** (`/bin/bash`, else `bash` on `$PATH`, else `sh` —
  `resolve_shell()`, cached), not a hardcoded `sh -c`, since a model-generated command routinely uses
  associative arrays/`[[`/`pipefail`/process substitution that a POSIX `sh` (often `dash`) rejects.
  Default timeout is 30 minutes (`timeout_ms` overridable) — deliberately longer than the reference
  agent's no-default, since this runs unattended on a homelab node with no one watching a hung shell.
  Output streams through one `OutputAccumulator` (`tools/output.rs`): a rolling byte tail
  (`DEFAULT_MAX_BYTES`, 50 KiB) is kept for display while the *complete* stream spills to a temp file
  once it outgrows that window (`"Full output: <path>"` marker), so the model can go read the whole
  thing if the tail isn't enough; while the command runs, `ToolProgress` streams an initial empty
  update then throttled snapshots (`UPDATE_THROTTLE`, 100 ms) of the output so far, plus truncation
  details, ahead of the final result. The combined text is run through output hygiene — ANSI/OSC escape
  stripping and C0 control-char sanitizing so terminal/binary noise can't corrupt the model's context —
  before either an `(no output)` placeholder (success, genuinely silent) or an appended status line
  (`"Command exited with code N"` / `"Command timed out after Ns"`, the latter with no placeholder glued
  in front of it). A timeout or a dropped future (cancellation) both route through the same
  `GroupKillGuard`, so a backgrounded grandchild process can't outlive either.
- **`fork`/`sync`/`logs`**: each builds a `beyond <subcommand> [args…]` argv (e.g.
  `["fork", app, "--name", name]`) and runs it through the same runner with a fixed 120 s timeout.
  Non-zero exit becomes `ToolError::Execution` (unlike `bash`, which reports a non-zero exit as text
  rather than failing the tool call). The `beyond` CLI itself lives outside this repo — these tools
  are tested only at the argv level.

---

## State Machine — `serve` session lifecycle

```
spawn ──► Booting ──writer task up + "ready" frame sent──► Ready
                                                               │
                                 stdin "prompt" line           │ stdin "get_state"/"get_messages" line
                                                               ▼                        │
                                                         RunningTurn                     │ (no transition)
                                                   (Agent::run_events;                  │
                                                    ≤ max_steps iterations)              │
                                                               │                          │
                                       turn ends / MaxSteps / transport error             │
                                                               ▼                          ▼
                                                             Ready ◄─────────────────────┘
                                                               │
                                                   stdin "new_session" line
                                                               ▼
                                                 Ready (fresh Session, new session_id)
                                                               │
                                         stdin EOF  or  stdout write fails
                                                               ▼
                                                            Closed
```

| From              | Event                                                 | To          | Guard                  | What Actually Happens                                                                                                          |
| ----------------- | ------------------------------------------------------ | ----------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Booting           | writer task spawned, `ready` frame sent               | Ready       | `out_tx.send` succeeds | `Session` restored from persistence (file/dir/none); `session_id` minted; static system prompt built                             |
| Ready             | `{"type":"prompt"}`                                   | RunningTurn | —                      | `ack` frame sent immediately; message pushed as a user turn; `Agent::run_events_steered` streams `event` frames live             |
| RunningTurn       | `{"type":"abort"}` / `"stop_after_turn"` / `"steer"` / `"follow_up"` | RunningTurn | —              | cancels the run / requests a graceful stop at the next turn boundary / queues a mid-run steer / queues a stop-boundary follow-up — the run keeps going until its current turn's tool calls (if any) finish |
| RunningTurn       | `{"type":"prompt", streaming_behavior:"steer"\|"follow_up"}` | RunningTurn | —                | accepted (not rejected as busy) and routed through the same `Steering` queue as an explicit `steer`/`follow_up`                  |
| RunningTurn       | any other command                                     | RunningTurn | —                      | rejected: `response{success:false, error:"busy…"}` — the session is borrowed by the in-flight run                                |
| RunningTurn       | model ends turn (no more `tool_use`, not a refusal)    | Ready       | —                      | session persisted (rewrite/rewrite_compacted if compaction fired); `response{success:true, data:{steps,…, refused:false}}`       |
| RunningTurn       | model refuses (`StopReason::Refusal`)                  | Ready       | —                      | run ends immediately *without* draining queued steering; `response{success:true, data:{…, refused:true}}`                        |
| RunningTurn       | `Error::MaxSteps` / transport error                    | Ready       | —                      | session still persisted; `response{success:false, error}` — the process keeps serving                                           |
| Ready             | `{"type":"new_session"}` / `"switch_session"` / `"fork"` / `"switch_branch"` | Ready | —              | history replaced/switched; `steering.clear()` — a message queued for the old session's next turn can't leak into the new one     |
| Ready             | invalid JSON / unknown `type`                         | Ready       | —                      | `response{success:false, error}`; loop continues, no state change                                                                |
| Ready/RunningTurn | stdin EOF                                             | Closed      | —                      | `out_tx` dropped → writer drains its queue → awaited → process returns `Ok(())`                                                  |
| any               | stdout write fails (broken pipe)                      | Closed      | —                      | writer task `break`s its receive loop; the next `emit!` send fails → main loop `break`s                                          |

---

## Why It Behaves This Way

### Why a single writer task with an unbounded channel

`Agent::run_events`'s event sink is a synchronous `FnMut` — the producer can't `.await` mid-callback
to apply backpressure. A bounded channel would force `try_send`, which silently drops frames on
backpressure and corrupts the NDJSON stream for a protocol where every frame matters. The channel is
unbounded instead, with the backlog naturally bounded by one in-flight turn's events (capped by
`max_steps`); if a client stops reading, stdout's write eventually fails and the writer tears down,
surfacing the stall as a closed session rather than masking it with dropped frames or unbounded
buffering. Every frame — events and responses — flows through this one task, so output is FIFO and
never interleaves even though tool execution itself is concurrent.

### Why tool results batch into one user message, not one per tool

(Enforced in `agent_core`, but it constrains how this crate's tools must behave.) Anthropic rejects
consecutive same-role messages; if a model batches N tool calls in one turn and the loop emitted N
separate `user` messages, the next request would 400 whenever N > 1. All of a turn's `tool_result`
blocks are gathered onto a single `user` message instead — which is also why `ToolStart` events are
emitted for the whole batch up front, then `ToolEnd`s after the concurrent join: the transcript order
must stay deterministic regardless of which tool actually finishes first.

### Why `edit` demands a unique match

A non-unique `old_string` means the model under-specified the change — `edit` refuses rather than
guessing which occurrence was meant, forcing the model to add surrounding context (or pass
`replace_all` for the explicit bulk case) before any byte of the file is touched. This trades one
extra round-trip for never silently editing the wrong occurrence.

### Why grep is parallel and find is sequential

See the Core Mechanism numbers above — this isn't a stylistic choice, it's the benchmark result.
`grep`'s bottleneck is per-file regex scanning (parallel walk gave 3.8×); `find`'s bottleneck is the
directory traversal itself, and its per-file cost (one glob test) is too cheap to amortize
`WalkParallel`'s thread-coordination overhead. Applying the same "parallelize it" instinct to both
would have made `find` slower, not faster — the Theory-of-Constraints discipline of profiling before
optimizing, then only investing in the part that's actually the bottleneck.

### Why this crate never holds a provider key

Model traffic always flows through the Beyond gateway; this crate's only network code is
`GatewayClient` (in `agent_core`), which sends a Bearer token and lets the gateway decide
authentication, routing, and key-swapping. Centralizing that in the gateway means this crate (and
every other agent harness) doesn't duplicate Ed25519 verification, deny-set checks, or per-provider
auth schemes — it just forwards a token and trusts the gateway's response.

---

## Trust Boundaries

**What this crate checks before acting:**

- `edit`'s `old_string` must match exactly once (or be an explicit single-edit `replace_all`).
- Each tool's required JSON fields (`path`, `command`, `pattern`, …) — checked ad hoc per tool, not
  against `input_schema()`; the schema is advisory to the model, not enforced before `run()`.
- Regex/glob patterns are compiled before use; a bad pattern is `ToolError::InvalidInput`, not a panic.
- **Project trust gates *some* instruction sources, not tool execution.** An untrusted working
  directory's own `.claude/SYSTEM.md`/`.claude/APPEND_SYSTEM.md`, project-local skills
  (`<cwd>/.claude/skills`), and prompt
  templates are none of: honored as a system-prompt override, injected into context, advertised via
  `get_commands`, or invocable via `/skill:name`/`/name` — closing the obvious prompt-injection path a
  hostile checkout could otherwise use (planting a `SKILL.md` a user innocently triggers). **Not**
  gated: `AGENTS.md`/`CLAUDE.md` project-instruction files (`resources::load_context_files` takes no
  `project_trusted` parameter at all) — these are injected into `<project_context>` regardless of
  trust, matching pi's own `resource-loader.ts::loadProjectContextFiles` (which has no trust check
  either). An untrusted repo's files can still be freely read/written/executed by the tools above once
  the model decides to, exactly as trusted-repo files can — trust never gates *that*, only which
  instruction sources are honored, and even there, only some of them. Trust gates *only* the
  project-local root, never the user-global one (`~/.claude/skills`, `~/.claude/SYSTEM.md`,
  `~/.claude/APPEND_SYSTEM.md`) — those are
  the operator's own machine, not something the current (possibly untrusted) project checkout controls,
  so `skills::discover`/`prompts::discover` (both `discover`/`discover_with_diagnostics` pairs) take
  `project_trusted` as a *required* parameter that adds or omits *only* the project root, and always
  scan the user root regardless. Trust alone doesn't make a `SKILL.md`'s frontmatter cooperative,
  though — "trusted" means the operator opted the directory in, not that every file in it is benign —
  so `skills::format_available` XML-escapes `name`/`description`/`path` before writing them into the
  `<available_skills>` block, closing the narrower residual path of a crafted description closing the
  tag early and forging a fake instruction block after it.

**What passes through unchecked:**

- **File paths.** No workspace-root containment check anywhere in `read`/`write`/`edit`/`ls`/`grep`/
  `find` — the agent can touch any path the OS-level process can reach (`../../etc/passwd` works if
  the model asks for it).
- **Shell commands.** `bash` execs whatever string the model supplies via a resolved real `bash` (or
  `sh`) with no allowlist, no sandboxing, and no resource limits beyond a wall-clock timeout — full
  ambient privilege of the host process. `--tools`/`--exclude-tools`/`--no-tools` can remove `bash`
  from the registry entirely for a given process, but that's an operator opt-in, not a default.
- **The gateway key.** `--key`/`AI_AGENT_KEY` is forwarded as a Bearer token without inspection;
  signature verification and deny-set checks happen entirely in the gateway (see
  `agent-core/ARCHITECTURE.md` and the gateway's own `ARCHITECTURE.md`).
- **The session file/repo.** Persistence distinguishes a missing file (the expected first-run case,
  silent) from one that exists but fails to read or parse (corrupt JSON, a schema-version mismatch, a
  permissions error): the latter is **logged to stderr** before falling back to an empty `Session`. A
  corrupt line *mid*-file (not just a torn final line) is skipped rather than discarding everything
  after it, but a corrupt/tampered file can still lose the prior transcript — it's no longer silent,
  matching the write-side error reporting.
- **`serve`'s stdin.** Any process that can write to this process's stdin has full control — there is
  no per-command authentication; trust is established once, by whoever spawned the process (e.g. an
  authenticated SSH pipe). This includes the trust-gated instruction sources above: once a directory
  *is* trusted, nothing further authenticates individual commands against it.

**Why these boundaries are where they are:** the agent is built to act as a fully trusted local
actor with the same authority as whoever launched it (`main.rs`'s system prompt: "You operate inside
a real working directory"). Containment, when it's needed, happens one layer up — e.g. the `fork`
tool's isolated Beyond branch, or running the whole process inside a constrained
container/VM — not by this crate restricting its own tools.

---

## Package Structure

| File                   | What It Does                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/main.rs`          | CLI entry point (`run`/`serve`/`tools`/`list-models`/`trust`/`untrust`/`clear-trust`/`export` subcommands — `list-models` prints `serve::available_models()`, no gateway/key needed, the same shape as `tools`); `DEFAULT_MODEL`, `DEFAULT_GATEWAY`, `default_system_prompt(&registry)` (generated from the actually-registered, filtered tool set); renders streamed text + `[tool: name]` markers (followed live by each `InputJsonDelta` fragment — a
  growing preview of the call's arguments as they stream in, not just its name) to stdout for `run` in its default text mode, or (`run --json`) one `AgentEvent` object per line via `run_events`/`serde_json::to_string` — the same full observation surface (tool calls/results, turn boundaries) `serve`'s NDJSON protocol streams, preceded by a `{"kind":"session", id, model, cwd}` header line — for a scripting caller that wants structured output without spawning `serve`; `run` composes its first message from piped stdin + `@file` references (`partition_tasks`/`read_file_refs`) + the first positional message, runs any further positional messages as separate sequential turns, and — via `--session <path>`/`--continue` (reusing `SessionStore`/`SessionRepo::resume_or_create`) — can persist and resume a transcript across invocations exactly like `serve`'s own repo/file modes; `run --export <path>` renders the finished transcript via `export::export_html` after a live run completes, while the standalone `export <session.jsonl> [output.html]` subcommand renders an *already-persisted* session file straight off disk — no gateway, key, or model involved at all (`SessionStore::open` + `export::export_html`)                                                                          |
| `src/lib.rs`           | Library root; re-exports `serve`/`tools`/`resources`/`skills`/`prompts`/`session_store`/`trust_store`/`export` for tests/benches                                                                                                                                                                                                                                                                                                         |
| `src/serve.rs`         | NDJSON control protocol: single stdout-writer task, `Persistence` (file/dir/none, default-per-cwd directory), a large command set (session/branch nav, `reload`, model/thinking/tool/auto-compaction tuning, `bash`/`abort_bash`, `export_html`) — see the module's own doc comment for the exhaustive list; prompt runs concurrently with stdin routing `steer`/`follow_up` (also accepted while idle, via a persistent `Steering` handle) |
| `src/export.rs`        | `export_html`/`render_html` — renders a session's transcript as one self-contained, dependency-free HTML file (inline CSS, no JS, images inlined as data URIs); also renders every abandoned branch (passed in as `SessionStore::abandoned_branches`'s output) as its own labeled section after the main transcript — only the divergent suffix, not the shared prefix already shown above; message text is rendered as markdown (`render_markdown`, `pulldown-cmark`, server-side at export time — not pi's client-side `marked`/`highlight.js`) with raw HTML defused to visible text and link/image URLs scheme-allow-listed (`sanitize_url`); a fenced ` ```diff ` block or diff-shaped tool-result content (`looks_like_diff`) gets per-line +/- coloring (`diff_html`) instead of real syntax highlighting, which is deliberately not implemented (would need a heavy crate like `syntect`, bloating every build of this CLI for a nice-to-have); reuses `skills::xml_escape` for HTML-text escaping; shared by `serve`'s `export_html` RPC command, `run --export`, and the standalone `export` subcommand                                                                                                                                                    |
| `src/trust_store.rs`   | Tri-state (`Trust::{Trusted,Untrusted,Unknown}`), ancestor-inheriting trust allowlist (`~/.claude/trusted-projects.json`); `trust`/`distrust` record an explicit grant/denial, `clear` removes a directory's own entry (trusted *or* untrusted) without recording a new one, reverting it to inheriting its nearest ancestor's decision; legacy bare-array files still parse (trusted-only)                                                                                                                                                                                                                                                            |
| `src/session_store.rs` | JSONL `SessionStore` (fsync'd append/atomic-rewrite/mid-file-corruption recovery, header metadata + durable `Entry::Compaction` provenance, version-migration guard, collision-safe ids) + multi-session `SessionRepo` (list-with-metadata, soft-delete-to-`.trash`, fork + read-only fork preview, `resume_or_create` — reopen the most recent session matching a `cwd` or make a fresh one, shared by `serve`'s startup reattach and `run --continue`); `default_session_dir`/`encode_cwd`/`canonical_cwd` — the `~/.claude/sessions/<encoded-cwd>/` convention and the symlink/trailing-separator-safe form every recorded `cwd` is passed through first, likewise shared; tree-shaped history (`id`/`parent_id` per message, `Leaf`/`BranchSummary`/`Compaction`/`ModelChange`/`ThinkingLevelChange`/`TitleChange` entries, `switch_active_with_summary`/`list_branches`/`tree`/`abandoned_by_switch`/`abandoned_branches`(every non-active leaf's full root-to-leaf message chain, plus how much of it is shared with the active path — for HTML export), legacy migration, off-branch-preserving compaction); `set_title` — an O(1) `TitleChange` append, not a rewrite, whole-session-scoped (most-recent-wins across the whole file) unlike the branch-scoped model/thinking-level changes; `SessionMeta::to_listing_json` — the derived listing fields (`updated_at`/`message_count`/`preview`/`search_text`) are `#[serde(skip)]` on the struct itself, so this is the only path that actually surfaces them as JSON |
| `src/resources.rs`     | System-prompt assembly split into `build_static_system_prompt` (on-disk `SYSTEM.md` override + additive `APPEND_SYSTEM.md`, one-file-per-dir `AGENTS.md`>`CLAUDE.md` discovery, skill injection — expensive, cached) and `dynamic_footer` (local date/cwd — cheap, refreshed every turn); `build_system_prompt` composes both for a one-shot caller                                                                                                                   |
| `src/skills.rs`        | Recursive skill discovery (`SKILL.md` frontmatter at any depth, `disable-model-invocation`, `/skill:` lookup — expands into a `<skill name=".." location="..">` tag with the frontmatter stripped, not the raw file) + `<available_skills>` rendering + `discover_with_diagnostics` (name-collision reporting); `project_trusted` gates only the project-local root, the user-global root is always scanned; `validate_skill_name`/`validate_skill_description` — non-fatal, `warn!`-logged shape/length checks (a bad `name`, or a `description` past 1024 chars) that never block discovery                |
| `src/prompts.rs`       | `/name args` prompt-template discovery + bash-style expansion (quote-aware args, `$N`, `${@:N:L}` slices, `${N:-default}`, `description` frontmatter) + `discover_with_diagnostics` (name-collision reporting)                                                                                                                                                                                                                           |
| `src/timing.rs`        | `StartupTiming` — `AI_AGENT_TIMING=1`-gated startup profiling (pi's own `PI_TIMING=1`/`timings.ts`); `mark(label)`/`print()` are no-ops (don't even read the clock) when unset, so it's safe to sprinkle through `run`/`serve`'s startup path unconditionally; prints to stderr only, never stdout                                                                                                                                      |
| `src/tools/mod.rs`     | `default_registry_with(bash_timeout_ms)` — assembles the base 10-tool `ToolRegistry`; `apply_filter(&mut registry, tools, exclude_tools, no_tools)` — allow/deny-list/no-tools filtering applied once at process build time                                                                                                                                                                                                             |
| `src/tools/read.rs`    | `read` — line-numbered read with `offset`/`limit`, byte budget, offset-past-EOF error, continuation hints; image files sniffed by magic bytes and downscaled/re-encoded (Lanczos3, PNG-then-JPEG, Exif orientation via `image`'s own generic decoder API — JPEG and WebP both) to fit a 4.5 MB base64 budget                                                                                                                            |
| `src/tools/write.rs`   | `write` — create/overwrite a file, creating parent directories                                                                                                                                                                                                                                                                                                                                                                           |
| `src/tools/edit.rs`    | `edit` — exact-then-fuzzy (NFKC/quote/dash/space/trailing-ws) replacement matched in LF space (CRLF/BOM restored), against the original, overlap/no-op checks, `replace_all`                                                                                                                                                                                                                                                             |
| `src/tools/ls.rs`      | `ls` — directory listing, directories-first sort, dotfile filtering, `limit` entry cap                                                                                                                                                                                                                                                                                                                                                   |
| `src/tools/grep.rs`    | `grep` — parallel, gitignore-aware regex (or `literal`) search with `context`/`before`/`after` lines, a whole-output byte cap (`MAX_OUTPUT_BYTES`), deterministic sort+truncate                                                                                                                                                                                                                                                          |
| `src/tools/find.rs`    | `find` — sequential, gitignore-aware glob search over files **and** dirs; deterministic sort+truncate                                                                                                                                                                                                                                                                                                                                    |
| `src/tools/bash.rs`    | `bash` — resolved real-`bash` (falling back to `sh`) execution with a 30-minute default timeout, streaming `OutputAccumulator` (tail-truncated display + full-output temp-file spill), output hygiene (ANSI strip/control sanitize)                                                                                                                                                                                                     |
| `src/tools/output.rs`  | Shared `OutputAccumulator`/`format_size`/`marker` — the bounded, spill-to-disk output-truncation machinery every truncating tool (`bash`/`read`/`grep`/`ls`/`find`) now shares                                                                                                                                                                                                                                                            |
| `src/tools/beyond.rs`  | `fork`/`sync`/`logs` — shell out to the `beyond` platform CLI                                                                                                                                                                                                                                                                                                                                                                            |
| `src/tools/exec.rs`    | `CommandRunner` trait + `RealRunner` (`GroupKillGuard` process-group kill on timeout *or* a dropped/cancelled future, bounded head+tail streaming capture, `ExecResult.truncated`) — the process-execution seam shared by `bash`/`beyond` tools and `serve`'s own `bash`/`abort_bash` RPC                                                                                                                                                |
| `benches/search.rs`    | Criterion macro-bench: `grep` (1 vs auto threads) and `find` (sequential) over a 5,000-file tree                                                                                                                                                                                                                                                                                                                                         |
| `tests/common/mod.rs`  | Shared test harness: mock Anthropic-SSE model server, gateway binary locator, port/connection helpers                                                                                                                                                                                                                                                                                                                                    |
| `tests/run_e2e.rs`     | `run` binary against a mock model server (no gateway in the loop)                                                                                                                                                                                                                                                                                                                                                                        |
| `tests/serve_e2e.rs`   | `serve` binary: NDJSON protocol round-trip, tool-call event streaming, session reattach, trust gating, model/thinking/tool cycling, refusal handling, idle steering, `stop_after_turn`, bash/abort_bash                                                                                                                                                                                                                                  |
| `tests/gateway_e2e.rs` | `run` binary → real gateway binary → mock upstream (proves key-swap + the virtual key never reaches upstream)                                                                                                                                                                                                                                                                                                                            |
| `tests/smoke.rs`       | Ignored-by-default live test: real gateway → real Anthropic/OpenAI across both providers (`mise run test:smoke:agent`)                                                                                                                                                                                                                                                                                                                   |

---

## Configuration

| Variable / Flag                                                | Default                    | What It Controls                                                                                                                 |
| -------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--model` / `AI_AGENT_MODEL`                                   | `claude-opus-4-8`           | Model id sent in each `ModelRequest`; selects the wire dialect (`agent_core::Dialect::for_model`); `serve`'s `set_model`/`cycle_model` switch it at runtime |
| `--gateway-url` / `AI_GATEWAY_URL`                             | `http://ai.internal`        | Base URL `GatewayClient` posts completions to                                                                                     |
| `--key` / `AI_AGENT_KEY`                                       | none (required)             | Bearer token sent to the gateway — a `bai_v1…` virtual key, or a BYO provider key forwarded as-is                                |
| `--max-steps`                                                  | `agent_core::DEFAULT_MAX_STEPS` (50) | Ceiling on loop iterations (`run`) or per-`prompt` iterations (`serve`) before `Error::MaxSteps` (resumable with a fresh call)     |
| `--tools` / `--exclude-tools` / `--no-tools` (+ `AI_AGENT_TOOLS`/`AI_AGENT_EXCLUDE_TOOLS` for `serve`) | none (full default registry) | Restrict/drop from the advertised tool set before the process's `Agent`/system prompt are built; `--no-tools` wins outright        |
| `--trust-project`                                              | `false`                     | Trust the cwd for this run only (session-scoped), independent of `agent trust <path>`'s persistent allowlist                     |
| `--force-untrusted`                                             | `false`                     | Force the cwd *untrusted* for this run only, overriding both `--trust-project` and a persisted `agent trust <path>` grant — pi's own `--no-approve`/`-na`; wins over `--trust-project` if both are given |
| `--session-file` / `--session-dir` / `AI_AGENT_SESSION_FILE`   | per-cwd directory under `~/.claude/sessions/` | `serve`-only: where the `Session`/`SessionRepo` persists; `--no-session-persistence` opts out to pure in-memory                   |
| `timeout_ms` (per-call, `bash` tool input)                     | `1_800_000` (30 min)        | Wall-clock ceiling for one `bash` invocation — deliberately long; this runs unattended with no one watching a hung shell           |
| `--bash-timeout-ms` / `AI_AGENT_BASH_TIMEOUT_MS`                | same as above               | Overrides `bash`'s own default when the model omits `timeout_ms`                                                                  |
| `RUST_LOG` (`tracing_subscriber::EnvFilter::from_default_env`) | unset (no logs)             | Verbosity of `tracing` spans/events emitted by the binary's subscriber                                                            |
| `AI_AGENT_TIMING`                                              | unset (no timing output)    | `=1` prints a startup-timing breakdown (resource discovery, system-prompt build, session open, agent construction) to stderr just before the first turn/`ready` frame — pi's own `PI_TIMING=1`; every checkpoint is a zero-cost no-op when unset |

Not configurable from this crate via a CLI flag: `max_tokens` (seeded model-aware by `agent_core`'s
`Agent::new`, ≥4096/turn; `serve`'s `set_thinking`/`cycle_thinking_level` tune the thinking budget at
runtime, `set_auto_compaction`/`set_auto_retry` toggle threshold-triggered compaction / mid-stream
retry at runtime).

---

## Failure Modes

| Failure                                                        | What Actually Happens                                                                                                                               | Recovery                                                             |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Gateway unreachable / non-2xx response                         | `Error::Transport`; `run` exits non-zero printing the error; `serve`'s `prompt` returns `response{success:false}`, process stays up                 | Caller retries the command — no built-in retry in this crate         |
| Tool execution fails (bad path, non-zero exit, regex error)    | `ToolError` → an error `tool_result`, fed back to the model next turn; the run does **not** abort                                                   | Model sees the error text and can adjust its next call               |
| `bash` command exceeds `timeout_ms`; `beyond` exceeds 120 s    | `RealRunner` returns `timed_out:true`, killing the whole process group (`GroupKillGuard`); tool turns it into `ToolError::Execution`                | Surfaced to the model as a tool error, same as any other failure     |
| A `prompt`/`bash` run is cancelled (`abort`/`abort_bash`, or the future dropped) | `GroupKillGuard`'s drop cascades into killing any backgrounded grandchild process either way — a timeout and a cancellation take the same kill path | The command is gone; `serve` reports `Error::Cancelled`, not a fault |
| Model never stops requesting tools                             | `session.steps >= max_steps` → `Error::MaxSteps`; `run` exits non-zero; `serve` returns a failed `response` but keeps serving — resumable with a fresh `prompt` | Caller issues a fresh `prompt`, or `new_session`                     |
| Model refuses the request (`StopReason::Refusal`)              | The run ends immediately without draining queued `steer`/`follow_up`; `serve`'s `prompt` response reports `refused: true`, still `success: true`   | Queued steering survives intact for a later `prompt` on the same session |
| Malformed JSON on `serve`'s stdin                              | Parsed as `Value`, fails → `response{success:false, error:"invalid JSON: …"}`; loop continues                                                       | Client resends a valid command                                       |
| `serve`'s stdout write fails (broken pipe)                     | Writer task exits its loop; the next `emit!` send observes the closed channel and `break`s the main loop; process exits `Ok(())`                    | None by design — a client must reconnect via a new process           |
| Session file/repo unreadable/corrupt at `serve` startup        | Persistence logs the read/parse failure to stderr, then falls back to a fresh empty `Session` (a missing file stays silent — expected first run); a corrupt line *mid*-file is skipped, not fatal | None automatic for a truly unreadable file — prior transcript is lost, but the failure is logged |
| Session file write fails after a turn                          | Logged to stderr only (`eprintln!`); the in-memory session and the turn's `response` still report success                                           | None automatic — operator must notice the stderr line                |
| `edit`'s `old_string` matches 0 or >1 times (no `replace_all`) | `ToolError::InvalidInput`; the file is **not** modified                                                                                             | Model adds more surrounding context and retries                      |
| `grep`/`find` hit `HARD_CAP` (10,000 collected items)          | Walk quits early; output is still sorted+truncated deterministically, but which items entered the set before the cap can vary                       | Model narrows the `pattern`/`glob` and retries                       |
| `grep`'s rendered output exceeds `MAX_OUTPUT_BYTES` (50 KiB)   | Truncated at a char boundary with its own marker; wins outright over the match-count marker if both would otherwise fire in the same response       | Model narrows the pattern/path/context and retries                   |
| `read` on an image that can't fit the 4.5 MB base64 budget even after downscaling | `ToolError::InvalidInput` — refused rather than silently sending a payload the provider would reject                                                | Model resizes the source image itself, or narrows what it asks for   |
| A working directory is untrusted                               | `SYSTEM.md`/skill/prompt-template sources from it are silently absent from the system prompt and `get_commands` — not an error (`AGENTS.md`/`CLAUDE.md` are unaffected either way — they're never trust-gated) | `agent trust <path>` (persistent) or `--trust-project`/RPC (session) |
