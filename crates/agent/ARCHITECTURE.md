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
  wins, untrusted checked first at each level) gates everything an untrusted repo's own files could use
  to inject instructions: a project-local `SYSTEM.md` override, `AGENTS.md`/`CLAUDE.md` project
  instructions, and discovered skills/prompt templates (including the `get_commands` listing and
  `/skill:name`/`/name` invocation) are all only honored once the working directory is trusted —
  `agent trust <path>` (persistent) or `--trust-project`/RPC (session-scoped). A legacy bare-array trust
  file still parses (trusted-only), migrated to the tri-state shape on the next `trust`/`distrust` call.
- **System-prompt assembly** ([`resources`](src/resources.rs)) — split into a **static** half (base
  identity, overridable by an on-disk `SYSTEM.md`: project `.claude/` when trusted, else user; project
  instruction files, global then cwd→root, **one file per directory**, `AGENTS.md` winning over
  `CLAUDE.md`, matched case-insensitively; discovered [`skills`](src/skills.rs) via
  `<available_skills>`, read-on-demand) and a **dynamic** footer (current **local** date + cwd). `serve`
  caches the static half — rebuilt only at startup and on `set_model`/`set_thinking`/an explicit
  `reload` — and refreshes just the cheap dynamic footer before every `prompt`, so a long-running
  process doesn't re-walk the filesystem every turn just for the date. CLI flags: `--system-prompt`
  (replace), `--append-system-prompt`, `--no-context-files`. Skills are discovered recursively
  (`SKILL.md` at any depth); a `disable-model-invocation` skill is omitted from the listing but still
  reachable by an explicit `/skill:name`, which strips the raw YAML frontmatter and wraps the body in a
  `<skill name="..." location="...">` tag rather than leaking the frontmatter verbatim. Both
  skill/prompt-template discovery report name collisions (`discover_with_diagnostics`) — the same name
  shadowed across roots or within one root — surfaced via `get_commands`'s `collisions` field.
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
  than the build understands is refused (migration hook). `--session-dir` opens a multi-session
  `SessionRepo` (list-with-metadata/create/open/soft-delete-to-`.trash`/fork); `--session-file` is the
  single-session form; neither flag defaults to `~/.claude/sessions/<encoded-cwd>/` rather than silent
  in-memory-only (`--no-session-persistence` opts out explicitly). `list` carries derived
  `updated_at`/`message_count`/`preview` without opening each transcript fully, and resuming without an
  explicit session id picks the newest session matching the **current cwd**, not just the globally
  newest.
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
- **Expanded `serve` surface** — beyond `prompt`/`get_state`/`get_messages`/`new_session`: `abort`,
  `stop_after_turn` (graceful — the current turn's tool calls still finish and commit; only the *next*
  model call is skipped — see `agent_core::Steering::request_stop`), `steer` (mid-run, folded onto the
  next tool turn) / `follow_up` (queued for the next stop boundary, and — unlike a bare
  `steer`/`follow_up` — also accepted while idle, queuing against a persistent handle for whichever
  `prompt` runs next; `stop_after_turn` sent while idle is instead a no-op ack, since there is no future
  run it could safely bind to), `compact`,
  `list_sessions`/`switch_session`/`fork`/`get_fork_messages`(read-only fork preview)/`set_session_name`,
  `get_last_assistant_text`/`get_session_stats`/`get_commands`(+ collision diagnostics)/`reload`,
  `set_model`/`set_thinking`/`cycle_model`/`cycle_thinking_level`/`set_auto_compaction` (rebuild the
  `Agent` for subsequent prompts) / `get_available_models`, `list_branches`/`get_tree`/`switch_branch`
  (navigate the session's tree, optionally summarizing the abandoned branch first — `get_messages` tags
  each message with its tree `id` so a client can name any point as a `switch_branch` target, not only a
  branch's leaf), and `bash`/`abort_bash` (run a host shell command directly, independent of the
  model's own tool loop). A `prompt` emits an immediate `ack` frame the moment it's queued, and its
  terminal `response` reports `refused: bool` (a refusal is a distinct terminal condition — it doesn't
  drain queued steering); a `prompt` sent while another is in flight can carry
  `streaming_behavior: "steer"|"follow_up"` to be accepted and queued instead of rejected as busy. A
  `prompt` runs concurrently with stdin so `abort`/`stop_after_turn`/`steer`/`follow_up` land mid-turn.
- **Tool set filtering** — `--tools`/`--exclude-tools`/`--no-tools` (both `run` and `serve`) restrict
  or drop from the default registry before it's advertised to the model; the auto-generated system
  prompt's tool list (`default_system_prompt`) reflects the filtered set, not the full default one, so a
  restricted agent never sees itself advertised a tool it doesn't actually have.
- **Multimodal** — `prompt` accepts `images: [{media_type, data}]` (base64), built into a multimodal
  user turn. `read` on an oversized image file downscales/re-encodes it (Lanczos3, JPEG re-encode,
  hand-rolled Exif orientation correction) to fit a 4.5 MB base64 budget rather than refusing outright,
  and sniffs the real magic bytes to recover from a mislabeled extension.

---

## Data Flow

### `run` — one-shot CLI

```
beyond-ai-agent run "<task>"
   │
   ▼
Session::new() + user(task)
   │
   ▼
Agent::run (agent_core loop; see agent-core/ARCHITECTURE.md for the loop itself)
   │  each step:
   │   ├─ POST one model turn → gateway → provider ──── network/4xx/5xx ──► Error::Transport, exit ≠0
   │   ├─ StreamEvent::TextDelta   ───────────────────────────────────────► stdout (printed live)
   │   ├─ StreamEvent::ToolUseStart ──────────────────────────────────────► stdout "\n[tool: name]"
   │   └─ assistant turn carries tool_use blocks → ToolRegistry.get(name)
   │        │
   │        ├─ found    → tool.run(input) → Ok(text) / Err(ToolError) ──► tool_result (is_error?)
   │        └─ not found → "unknown tool: <name>" ───────────────────────► tool_result, is_error=true
   │
   ▼ (model ends its turn without a tool_use, or session.steps == max_steps)
stdout: trailing newline
stderr: "[done in N step(s); X in / Y out tokens]"   (or the propagated Error::MaxSteps / Error::Transport)
```

### `serve` — headless NDJSON control protocol

```
stdin (one JSON command per line)              stdout (one JSON frame per line, single writer task)
  │                                                   ▲
  ▼                                                   │
serve() boot: open persistence (file/dir/none)        │
  │            + build static system prompt            │
  ▼                                                    ├── {"type":"ready", session_id, model}
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
  │                                                              message_count, input/output_tokens}}
  ├─ {"type":"get_messages"} / {"type":"get_tree"} ──► response{data:{messages:[...]}} /
  │                                                              response{data:{nodes:[TreeNode…]}}
  ├─ {"type":"new_session"} ─────────────────────────► response{data:{session_id}}  (fresh Session;
  │     steering.clear()                                        persisted per the open persistence mode)
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
`set_auto_compaction`) — not just re-derived per `Agent`.

---

## Concepts & Terminology

| Term                        | What It Controls                                                                                 | NOT                                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Tool**                    | A registered capability the model invokes by name + JSON input (`agent_core::Tool` impl)         | Not necessarily a subprocess — only `bash`/`fork`/`sync`/`logs` shell out; the rest touch the filesystem in-process             |
| **`ToolRegistry`**          | The name → `Tool` map advertised to the model every turn (`default_registry_with(..)`, then `tools::apply_filter`'d by `--tools`/`--exclude-tools`/`--no-tools`) | Not per-session or hot-reloadable at runtime — filtering happens once when the process's registry is built, not per turn |
| **`CommandRunner`**         | The seam between a tool and real process execution (`exec.rs`)                                   | Not a sandbox — `RealRunner` execs a resolved real `bash` (falling back to `sh`) / `beyond …` with the host process's full ambient privilege |
| **`TrustStore`**            | Tri-state (trusted/untrusted/unknown), ancestor-inheriting allowlist gating `SYSTEM.md`/project-instruction/skill/prompt-template access | Not a sandbox or a permission system for tools — it only gates *which instructions the model sees*, not what it can do   |
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
| `edit`               | `old_string` matches 0 or >1 times after exact+fuzzy (without single-edit `replace_all`); overlapping/no-op edits; malformed `edits` | file unreadable/unwritable                                                                  |
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
  a `"… [line truncated]"` marker. Line bytes decode lossily (`from_utf8_lossy`), so a non-UTF-8 file
  reads with replacement chars rather than erroring. An **image** file skips text decoding entirely:
  the extension gate decides *whether* to route into the image path (so a plain text read never pays
  for an image-format probe), but the reported format comes from sniffing the real magic bytes
  (`image::guess_format`) — a mislabeled extension still reports its true type. It's returned as a
  base64 `ImageSource` attachment the multimodal model can see; one already under a 4.5 MB base64
  budget goes out as its original bytes/format unmodified, an oversized one is downscaled (Lanczos3,
  max 2000px) and re-encoded as JPEG (quality 80, stepping dimensions/quality down further if still over
  budget), applying a hand-rolled Exif orientation correction first — only refused outright if even the
  smallest re-encode can't fit.
- **`write`** creates parent directories (`create_dir_all`), then writes **atomically** (sibling temp
  file + `rename`, shared with `edit` via `tools::write_atomic`) so a kill mid-write can't leave a
  half-written file; always overwrites.
- **`edit`** accepts either an `edits: [{old_string,new_string}]` array (applied in order) or a
  single `old_string`/`new_string` pair. Each `old_string` must match **exactly once** in the current
  content unless it's a single-edit call with `replace_all: true` — uniqueness is the only safety
  check; there is no diff/dry-run, the file is rewritten on success. Matching tries an **exact** hit
  first, then a normalized **fuzzy fallback** (NFKC + folding smart quotes/dash family/unicode spaces +
  per-line trailing-whitespace), with hits mapped back to original byte offsets — so a model's
  `old_string` carrying a curly quote, em-dash, nbsp, or stray trailing space still lands instead of
  failing "not found". Edits resolve to byte ranges against the _original_ text (order-independent,
  overlap-rejected).
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
- **Project trust gates instruction sources, not tool execution.** An untrusted working directory's
  own `.claude/SYSTEM.md`, `AGENTS.md`/`CLAUDE.md`, skills, and prompt templates are none of: honored
  as a system-prompt override, injected into context, advertised via `get_commands`, or invocable via
  `/skill:name`/`/name` — closing the obvious prompt-injection path a hostile checkout could otherwise
  use (planting a `SKILL.md` a user innocently triggers). This is the *only* thing trust gates: an
  untrusted repo's files can still be freely read/written/executed by the tools above once the model
  decides to, exactly as trusted-repo files can.

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
| `src/main.rs`          | CLI entry point (`run`/`serve`/`tools`/`trust`/`untrust` subcommands); `DEFAULT_MODEL`, `DEFAULT_GATEWAY`, `default_system_prompt(&registry)` (generated from the actually-registered, filtered tool set); renders streamed text + `[tool: name]` markers to stdout for `run`                                                                                                                                                            |
| `src/lib.rs`           | Library root; re-exports `serve`/`tools`/`resources`/`skills`/`prompts`/`session_store`/`trust_store` for tests/benches                                                                                                                                                                                                                                                                                                                  |
| `src/serve.rs`         | NDJSON control protocol: single stdout-writer task, `Persistence` (file/dir/none, default-per-cwd directory), a large command set (session/branch nav, `reload`, model/thinking/tool/auto-compaction tuning, `bash`/`abort_bash`) — see the module's own doc comment for the exhaustive list; prompt runs concurrently with stdin routing `steer`/`follow_up` (also accepted while idle, via a persistent `Steering` handle)           |
| `src/trust_store.rs`   | Tri-state (`Trust::{Trusted,Untrusted,Unknown}`), ancestor-inheriting trust allowlist (`~/.claude/trusted-projects.json`); legacy bare-array files still parse (trusted-only)                                                                                                                                                                                                                                                            |
| `src/session_store.rs` | JSONL `SessionStore` (fsync'd append/atomic-rewrite/mid-file-corruption recovery, header metadata + durable `Entry::Compaction` provenance, version-migration guard, collision-safe ids) + multi-session `SessionRepo` (list-with-metadata, soft-delete-to-`.trash`, fork + read-only fork preview); tree-shaped history (`id`/`parent_id` per message, `Leaf`/`BranchSummary`/`Compaction` entries, `switch_active_with_summary`/`list_branches`/`tree`/`abandoned_by_switch`, legacy migration, off-branch-preserving compaction) |
| `src/resources.rs`     | System-prompt assembly split into `build_static_system_prompt` (on-disk `SYSTEM.md` override, one-file-per-dir `AGENTS.md`>`CLAUDE.md` discovery, skill injection — expensive, cached) and `dynamic_footer` (local date/cwd — cheap, refreshed every turn); `build_system_prompt` composes both for a one-shot caller                                                                                                                   |
| `src/skills.rs`        | Recursive skill discovery (`SKILL.md` frontmatter at any depth, `disable-model-invocation`, `/skill:` lookup — expands into a `<skill name=".." location="..">` tag with the frontmatter stripped, not the raw file) + `<available_skills>` rendering + `discover_with_diagnostics` (name-collision reporting)                                                                                                                          |
| `src/prompts.rs`       | `/name args` prompt-template discovery + bash-style expansion (quote-aware args, `$N`, `${@:N:L}` slices, `${N:-default}`, `description` frontmatter) + `discover_with_diagnostics` (name-collision reporting)                                                                                                                                                                                                                           |
| `src/tools/mod.rs`     | `default_registry_with(bash_timeout_ms)` — assembles the base 10-tool `ToolRegistry`; `apply_filter(&mut registry, tools, exclude_tools, no_tools)` — allow/deny-list/no-tools filtering applied once at process build time                                                                                                                                                                                                             |
| `src/tools/read.rs`    | `read` — line-numbered read with `offset`/`limit`, byte budget, offset-past-EOF error, continuation hints; image files sniffed by magic bytes and downscaled/re-encoded (Lanczos3 + Exif-orientation-aware) to fit a 4.5 MB base64 budget                                                                                                                                                                                                |
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
| `--session-file` / `--session-dir` / `AI_AGENT_SESSION_FILE`   | per-cwd directory under `~/.claude/sessions/` | `serve`-only: where the `Session`/`SessionRepo` persists; `--no-session-persistence` opts out to pure in-memory                   |
| `timeout_ms` (per-call, `bash` tool input)                     | `1_800_000` (30 min)        | Wall-clock ceiling for one `bash` invocation — deliberately long; this runs unattended with no one watching a hung shell           |
| `--bash-timeout-ms` / `AI_AGENT_BASH_TIMEOUT_MS`                | same as above               | Overrides `bash`'s own default when the model omits `timeout_ms`                                                                  |
| `RUST_LOG` (`tracing_subscriber::EnvFilter::from_default_env`) | unset (no logs)             | Verbosity of `tracing` spans/events emitted by the binary's subscriber                                                            |

Not configurable from this crate via a CLI flag: `max_tokens` (seeded model-aware by `agent_core`'s
`Agent::new`, ≥4096/turn; `serve`'s `set_thinking`/`cycle_thinking_level` tune the thinking budget at
runtime, `set_auto_compaction` toggles threshold-triggered compaction at runtime).

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
| A working directory is untrusted                               | `SYSTEM.md`/project-instruction/skill/prompt-template sources from it are silently absent from the system prompt and `get_commands` — not an error  | `agent trust <path>` (persistent) or `--trust-project`/RPC (session) |
