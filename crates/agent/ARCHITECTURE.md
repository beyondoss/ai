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

- **System-prompt assembly** ([`resources`](src/resources.rs)) — the system prompt is built per
  session from a base identity (overridable by an on-disk `SYSTEM.md`: project `.claude/`, then user) +
  project instruction files (global, then cwd→root; **one file per directory**, `AGENTS.md` winning
  over `CLAUDE.md`, matched case-insensitively) + discovered [`skills`](src/skills.rs)
  (`<available_skills>`, read-on-demand) + the current **local** date and cwd. Flags:
  `--system-prompt` (replace), `--append-system-prompt`, `--no-context-files`. Skills are discovered
  recursively (`SKILL.md` at any depth); a `disable-model-invocation` skill is omitted from the listing
  but still reachable by an explicit `/skill:name`.
- **Prompt templates** ([`prompts`](src/prompts.rs)) — a `/name args` prompt is expanded from a
  `.claude/prompts/*.md` template with bash-style substitution before it reaches the model: quote-aware
  arg splitting, `$N` for any positional, `${@:N}`/`${@:N:L}` slices, and `${N:-default}`. A template's
  `description` (frontmatter, else first body line) feeds autocomplete.
- **Session persistence** ([`session_store`](src/session_store.rs)) — append-only JSONL with a header
  carrying a collision-resistant id + metadata; a turn appends only its new messages (compaction
  rewrites atomically and records its provenance — `compactions`/`dropped_messages`), every write is
  `fsync`ed for durability, a torn final line is dropped on load, and a header whose `version` is newer
  than the build understands is refused (migration hook). `--session-dir` opens a multi-session
  `SessionRepo` (list-with-metadata/create/open/idempotent-delete/fork); `--session-file` is the
  single-session form. `list` carries derived `updated_at`/`message_count`/`preview` without opening
  each transcript fully.
- **Tree-shaped history** — every message line also carries an `id`/`parent_id` (additive,
  `#[serde(default)]`; a pre-tree file's absent fields are migrated to synthesized, chained ids in
  memory only, never persisted back). The "active path" (`Session.messages`) is the `parent_id` chain
  from root to tip; a `Leaf` entry (`SessionStore::switch_active`) redirects the tip append-only —
  navigating never deletes anything. Compaction (`rewrite`) stays destructive to the active path's own
  entries but now preserves every node on some _other_ branch, writing them before the fresh
  compacted-path entries so "the last message in the file" still resolves to the tip without needing a
  `Leaf` marker. `abandoned_by_switch` computes what a hypothetical switch would abandon (via a
  common-ancestor walk) so a caller can summarize it first (`agent_core::branch_summary_request` +
  `record_branch_summary`, a `branch_summaries`/`summarized_branch_messages` counter pair alongside
  `compactions`/`dropped_messages`). `list_branches` reports every leaf plus the active tip (even when
  it isn't a leaf — navigating to an ancestor without yet forking a new line from it).
- **Expanded `serve` surface** — beyond `prompt`/`get_state`/`get_messages`/`new_session`: `abort`,
  `steer` (mid-run, folded onto the next tool turn) / `follow_up` (queued for the next stop boundary),
  `compact`, `list_sessions`/`switch_session`/`fork`/`set_session_name`,
  `get_last_assistant_text`/`get_session_stats`/`get_commands`, `set_model`/`set_thinking`
  (rebuild the `Agent` for subsequent prompts) / `get_available_models`, and `list_branches`/
  `switch_branch` (navigate the session's tree, optionally summarizing the abandoned branch first —
  `get_messages` tags each message with its tree `id` so a client can name any point as a
  `switch_branch` target, not only a branch's leaf). A `prompt` runs concurrently with stdin so
  `abort`/`steer`/`follow_up` land mid-turn.
- **Multimodal** — `prompt` accepts `images: [{media_type, data}]` (base64), built into a multimodal
  user turn.

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
serve() boot: load --session-file or Session::new()   │
  │                                                    ├── {"type":"ready", session_id, model}
  ▼                                                    │
loop over stdin lines ──────────────────────────────► │
  │                                                    │
  ├─ {"type":"prompt", message} ─────────────────────► event* (Stream / ToolStart / ToolEnd / TurnEnd)
  │     session.user(message)                          response{command:"prompt", success, data:{steps,
  │     Agent::run_events(session, |ev| tx.send(ev))            input_tokens, output_tokens}}
  │     save_session(--session-file)  [write error → stderr only, turn still reports success]
  │
  ├─ {"type":"get_state"} ───────────────────────────► response{data:{session_id, model, steps,
  │                                                              message_count, input/output_tokens}}
  ├─ {"type":"get_messages"} ────────────────────────► response{data:{messages:[...]}}
  ├─ {"type":"new_session"} ─────────────────────────► response{data:{session_id}}  (fresh Session,
  │                                                              persisted if --session-file is set)
  └─ invalid JSON / unknown "type" ──────────────────► response{success:false, error}

stdin EOF  →  out_tx dropped → writer drains queued frames → process exits Ok(())
stdout write fails (broken pipe) → writer task exits its loop → next emit! observes a closed
                                     channel → main loop breaks → process exits Ok(())
```

A `prompt`'s tool calls are NOT shown above as a separate fan-out: `agent_core::Agent::run_events`
runs the tools the model batched in one turn **concurrently but bounded** (grouped by write-target so
same-path calls serialize, then `buffer_unordered` over ≤8 groups — see `agent-core/ARCHITECTURE.md`),
yet always emits `ToolStart` for all of them up front, then `ToolEnd`s in original call order — so the
NDJSON stream a client sees is deterministic regardless of which tool finishes first.

---

## Concepts & Terminology

| Term                        | What It Controls                                                                                 | NOT                                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Tool**                    | A registered capability the model invokes by name + JSON input (`agent_core::Tool` impl)         | Not necessarily a subprocess — only `bash`/`fork`/`sync`/`logs` shell out; the rest touch the filesystem in-process             |
| **`ToolRegistry`**          | The name → `Tool` map advertised to the model every turn (`default_registry()`, fixed at 10)     | Not per-session or hot-reloadable — one registry is built once at `Agent` construction and reused for the process's life        |
| **`CommandRunner`**         | The seam between a tool and real process execution (`exec.rs`)                                   | Not a sandbox — `RealRunner` execs `sh -c …` / `beyond …` with the host process's full ambient privilege                        |
| **`Session`**               | Message history + step/token counters; optionally `serde`-persisted to `--session-file`          | Not multi-session — one `serve` process holds exactly one active `Session` at a time                                            |
| **`AgentEvent`**            | `Stream`/`ToolStart`/`ToolEnd`/`TurnEnd` boundaries streamed as `event` frames during a `prompt` | Not the terminal answer — the `response` frame (success/data/error) is separate and always comes last                           |
| **Virtual key** (`bai_v1…`) | The bearer token this crate forwards to the gateway on every request                             | Not verified or interpreted here — Ed25519 signature check and deny-set live entirely in the gateway                            |
| **`max_steps`**             | Ceiling on loop iterations per `run` invocation, or per `prompt` in `serve` (CLI default 24)     | Not a token budget — `max_tokens` (seeded model-aware by `agent_core`'s `Agent::new`, ≥4096/turn) has no CLI flag in this crate |
| **`HARD_CAP`** (grep/find)  | OOM guard: walk quits once 10,000 matches/paths are collected, before `limit` truncation runs    | Not the reported limit — `limit` (default 100/1000) is the user-facing cap; `HARD_CAP` is a backstop far above it               |

---

## Core Mechanism

### Tool dispatch

`tools::default_registry()` assembles a fixed, hard-coded set of ten tools — there is no
configuration surface in this crate to add, remove, or reorder them. Each `Tool` is a stateless
(or `Arc<dyn CommandRunner>`-holding) value; `agent_core::Agent::run_events` looks tools up by the
name the model used and calls `run(input)`, converting `Err(ToolError)` into an error `tool_result`
rather than aborting the run (see `agent_core/ARCHITECTURE.md`). That error/success split is what
each tool actually produces:

| Tool                 | `ToolError::InvalidInput` when…                                                                                                      | `ToolError::Execution` when…                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `read`               | missing `path`; offset past EOF; image file over the 5 MB inline cap                                                                 | file unreadable (missing, permission denied) — non-UTF-8 bytes decode lossily, not an error |
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
  appends a `"… (truncated …)"` marker if more lines remain. It streams line-by-line and caps each
  line at `MAX_LINE_BYTES` (4000) — bytes past the cap are drained but not stored, so one pathological
  single line (a minified bundle) can't balloon memory; a capped line gets a `"… [line truncated]"`
  marker. Line bytes decode lossily (`from_utf8_lossy`), so a non-UTF-8 file reads with replacement
  chars rather than erroring. An **image** file (`.png`/`.jpg`/`.gif`/`.webp`, by extension) skips text
  decoding entirely: it's returned as a base64 `ImageSource` attachment the multimodal model can see,
  refused above a 5 MB inline cap rather than ballooning the request.
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
- **`ls`** sorts directories first, then alphabetically; dot-entries are hidden unless `all: true`.
  Caps the listing at `limit` (default 500, overridable) so a `node_modules`-sized directory can't
  flood context, appending a `"… (N more entries) …"` marker when it truncates.

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
count toward `limit` and `HARD_CAP`.

### Shell tools — a shared `CommandRunner` seam

`bash` and the Beyond tools (`fork`/`sync`/`logs`) don't touch the filesystem directly; they both
go through `tools::exec::CommandRunner`, implemented for production by `RealRunner`
(`tokio::process::Command` with `kill_on_drop(true)`, a process-group leader so a timeout kills the
whole tree, and **bounded streaming capture**: each of stdout/stderr keeps only its head+tail
(~128 KB each) as it drains, so a `yes`-style firehose holds bounded memory instead of OOMing; the
dropped middle is reported via `ExecResult.truncated`; `CommandRunner::run_streaming` additionally
hands each chunk to a `ChunkSink` as it arrives, which `bash`'s `Tool::run_streaming` forwards to its
`ToolProgress` sink so a client sees live command output) and by recording test doubles in each tool's
`#[cfg(test)]` module. This is the same seam pattern as `agent_core`'s `Tool`/`ModelTransport` traits
— it's what lets `fork_builds_argv`/`sync_builds_argv`/etc. assert the exact argv without a live
Beyond control plane.

- **`bash`**: `sh -c "<command>"`, default 120 s timeout (`timeout_ms` overridable). A timeout returns
  `ToolError::Execution` immediately (the command's partial output is discarded). Otherwise stdout+
  stderr are combined and `[exit code N]` is appended for a non-zero exit, or `[killed]` if the
  process died with no exit code (killed by a signal). The combined text is then run through output
  hygiene — ANSI/OSC escape stripping and C0 control-char sanitizing so terminal/binary noise can't
  corrupt the model's context — then capped to `MAX_LINES` (2000) and ~30,000 bytes, each keeping the
  **head and tail** (the middle of a long log is least useful; the start shows what ran and the end the
  result). A `… (output truncated at source) …` marker surfaces an exec-layer capture drop even when
  the display caps don't fire.
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

| From              | Event                                   | To          | Guard                  | What Actually Happens                                                                     |
| ----------------- | --------------------------------------- | ----------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| Booting           | writer task spawned, `ready` frame sent | Ready       | `out_tx.send` succeeds | `Session` loaded from `--session-file` (or `Session::new()`); `session_id` minted         |
| Ready             | `{"type":"prompt"}`                     | RunningTurn | —                      | message pushed as a user turn; `Agent::run_events` streams `event` frames live            |
| RunningTurn       | model ends turn (no more `tool_use`)    | Ready       | —                      | session saved to `--session-file` (if set); `response{success:true, data:{steps,…}}`      |
| RunningTurn       | `Error::MaxSteps` / transport error     | Ready       | —                      | session still saved; `response{success:false, error}` — the process keeps serving         |
| Ready             | `{"type":"new_session"}`                | Ready       | —                      | `Session::new()` replaces history; new `session_id`; persisted if `--session-file` is set |
| Ready             | invalid JSON / unknown `type`           | Ready       | —                      | `response{success:false, error}`; loop continues, no state change                         |
| Ready/RunningTurn | stdin EOF                               | Closed      | —                      | `out_tx` dropped → writer drains its queue → awaited → process returns `Ok(())`           |
| any               | stdout write fails (broken pipe)        | Closed      | —                      | writer task `break`s its receive loop; the next `emit!` send fails → main loop `break`s   |

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

**What passes through unchecked:**

- **File paths.** No workspace-root containment check anywhere in `read`/`write`/`edit`/`ls`/`grep`/
  `find` — the agent can touch any path the OS-level process can reach (`../../etc/passwd` works if
  the model asks for it).
- **Shell commands.** `bash` execs whatever string the model supplies via `sh -c` with no allowlist,
  no sandboxing, and no resource limits beyond a wall-clock timeout — full ambient privilege of the
  host process.
- **The gateway key.** `--key`/`AI_AGENT_KEY` is forwarded as a Bearer token without inspection;
  signature verification and deny-set checks happen entirely in the gateway (see
  `agent-core/ARCHITECTURE.md` and the gateway's own `ARCHITECTURE.md`).
- **The session file.** `load_session` distinguishes a missing file (the expected first-run case,
  silent) from a file that exists but fails to read or parse (corrupt JSON, a schema change across a
  deploy, a permissions error): the latter is **logged to stderr** before falling back to an empty
  `Session::default()`. The fallback still means a corrupt/tampered file loses the prior transcript —
  but it's no longer silent, matching `save_session`'s error reporting.
- **`serve`'s stdin.** Any process that can write to this process's stdin has full control — there is
  no per-command authentication; trust is established once, by whoever spawned the process (e.g. an
  authenticated SSH pipe).

**Why these boundaries are where they are:** the agent is built to act as a fully trusted local
actor with the same authority as whoever launched it (`main.rs`'s system prompt: "You operate inside
a real working directory"). Containment, when it's needed, happens one layer up — e.g. the `fork`
tool's isolated Beyond branch, or running the whole process inside a constrained
container/VM — not by this crate restricting its own tools.

---

## Package Structure

| File                   | What It Does                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/main.rs`          | CLI entry point (`run`/`serve`/`tools` subcommands); `DEFAULT_MODEL`, `DEFAULT_GATEWAY`, `SYSTEM_PROMPT`; renders streamed text + `[tool: name]` markers to stdout for `run`                                                                                                                                                                                                                                                             |
| `src/lib.rs`           | Library root; re-exports `serve`/`tools`/`resources`/`skills`/`prompts`/`session_store` for tests/benches                                                                                                                                                                                                                                                                                                                                |
| `src/serve.rs`         | NDJSON control protocol: single stdout-writer task, `Persistence` (file or repo), expanded command set (incl. `set_model`/`set_thinking`/`get_available_models`), prompt runs concurrently with stdin routing `steer` (mid-run) vs `follow_up` (stop-boundary)                                                                                                                                                                           |
| `src/session_store.rs` | JSONL `SessionStore` (fsync'd append/atomic-rewrite/torn-line recovery, header metadata + compaction provenance, version-migration guard, collision-safe ids) + multi-session `SessionRepo` (list-with-metadata, idempotent delete, fork); tree-shaped history (`id`/`parent_id` per message, `Leaf`/`BranchSummary` entries, `switch_active`/`list_branches`/`abandoned_by_switch`, legacy migration, off-branch-preserving compaction) |
| `src/resources.rs`     | System-prompt assembly: on-disk `SYSTEM.md` base override, one-file-per-dir `AGENTS.md`>`CLAUDE.md` discovery (case-insensitive), skill injection, local date/cwd                                                                                                                                                                                                                                                                        |
| `src/skills.rs`        | Recursive skill discovery (`SKILL.md` frontmatter at any depth, `disable-model-invocation`, `/skill:` lookup) + `<available_skills>` rendering                                                                                                                                                                                                                                                                                           |
| `src/prompts.rs`       | `/name args` prompt-template discovery + bash-style expansion (quote-aware args, `$N`, `${@:N:L}` slices, `${N:-default}`, `description` frontmatter)                                                                                                                                                                                                                                                                                    |
| `src/tools/mod.rs`     | `default_registry()` — assembles the fixed 10-tool `ToolRegistry`                                                                                                                                                                                                                                                                                                                                                                        |
| `src/tools/read.rs`    | `read` — line-numbered read with `offset`/`limit`, byte budget, offset-past-EOF error, continuation hints; image files as base64 attachments                                                                                                                                                                                                                                                                                             |
| `src/tools/write.rs`   | `write` — create/overwrite a file, creating parent directories                                                                                                                                                                                                                                                                                                                                                                           |
| `src/tools/edit.rs`    | `edit` — exact-then-fuzzy (NFKC/quote/dash/space/trailing-ws) replacement matched in LF space (CRLF/BOM restored), against the original, overlap/no-op checks, `replace_all`                                                                                                                                                                                                                                                             |
| `src/tools/ls.rs`      | `ls` — directory listing, directories-first sort, dotfile filtering, `limit` entry cap                                                                                                                                                                                                                                                                                                                                                   |
| `src/tools/grep.rs`    | `grep` — parallel, gitignore-aware regex (or `literal`) search with `context`/`before`/`after` lines; deterministic sort+truncate                                                                                                                                                                                                                                                                                                        |
| `src/tools/find.rs`    | `find` — sequential, gitignore-aware glob search over files **and** dirs; deterministic sort+truncate                                                                                                                                                                                                                                                                                                                                    |
| `src/tools/bash.rs`    | `bash` — `sh -c` execution with timeout, output hygiene (ANSI strip/control sanitize) and head/tail line+byte truncation (process-group kill on timeout)                                                                                                                                                                                                                                                                                 |
| `src/tools/beyond.rs`  | `fork`/`sync`/`logs` — shell out to the `beyond` platform CLI                                                                                                                                                                                                                                                                                                                                                                            |
| `src/tools/exec.rs`    | `CommandRunner` trait + `RealRunner` (bounded head+tail streaming capture, `ExecResult.truncated`) — the process-execution seam shared by `bash`/`beyond` tools                                                                                                                                                                                                                                                                          |
| `benches/search.rs`    | Criterion macro-bench: `grep` (1 vs auto threads) and `find` (sequential) over a 5,000-file tree                                                                                                                                                                                                                                                                                                                                         |
| `tests/common/mod.rs`  | Shared test harness: mock Anthropic-SSE model server, gateway binary locator, port/connection helpers                                                                                                                                                                                                                                                                                                                                    |
| `tests/run_e2e.rs`     | `run` binary against a mock model server (no gateway in the loop)                                                                                                                                                                                                                                                                                                                                                                        |
| `tests/serve_e2e.rs`   | `serve` binary: NDJSON protocol round-trip, tool-call event streaming, session reattach                                                                                                                                                                                                                                                                                                                                                  |
| `tests/gateway_e2e.rs` | `run` binary → real gateway binary → mock upstream (proves key-swap + the virtual key never reaches upstream)                                                                                                                                                                                                                                                                                                                            |
| `tests/smoke.rs`       | Ignored-by-default live test: real gateway → real Anthropic (`mise run test:smoke:agent`)                                                                                                                                                                                                                                                                                                                                                |

---

## Configuration

| Variable / Flag                                                | Default               | What It Controls                                                                                  |
| -------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------- |
| `--model` / `AI_AGENT_MODEL`                                   | `claude-opus-4-8`     | Model id sent in each `ModelRequest`; selects the wire dialect (`agent_core::Dialect::for_model`) |
| `--gateway-url` / `AI_GATEWAY_URL`                             | `http://ai.internal`  | Base URL `GatewayClient` posts completions to                                                     |
| `--key` / `AI_AGENT_KEY`                                       | none (required)       | Bearer token sent to the gateway — a `bai_v1…` virtual key, or a BYO provider key forwarded as-is |
| `--max-steps`                                                  | `24`                  | Ceiling on loop iterations (`run`) or per-`prompt` iterations (`serve`) before `Error::MaxSteps`  |
| `--session-file` / `AI_AGENT_SESSION_FILE`                     | none (in-memory only) | `serve`-only: path the `Session` is serialized to after every turn; required for reattach         |
| `timeout_ms` (per-call, `bash` tool input)                     | `120000`              | Wall-clock ceiling for one `bash` invocation                                                      |
| `RUST_LOG` (`tracing_subscriber::EnvFilter::from_default_env`) | unset (no logs)       | Verbosity of `tracing` spans/events emitted by the binary's subscriber                            |

Not configurable from this crate via a CLI flag: `max_tokens` (seeded model-aware by `agent_core`'s
`Agent::new`, ≥4096/turn; `serve`'s `set_thinking` tunes the thinking budget at runtime) and the
tool set itself (always the same 10 tools from `default_registry()` — no enable/disable flag).

---

## Failure Modes

| Failure                                                        | What Actually Happens                                                                                                                               | Recovery                                                             |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Gateway unreachable / non-2xx response                         | `Error::Transport`; `run` exits non-zero printing the error; `serve`'s `prompt` returns `response{success:false}`, process stays up                 | Caller retries the command — no built-in retry in this crate         |
| Tool execution fails (bad path, non-zero exit, regex error)    | `ToolError` → an error `tool_result`, fed back to the model next turn; the run does **not** abort                                                   | Model sees the error text and can adjust its next call               |
| `bash` command exceeds `timeout_ms`; `beyond` exceeds 120 s    | `RealRunner` returns `timed_out:true`; tool turns it into `ToolError::Execution`                                                                    | Surfaced to the model as a tool error, same as any other failure     |
| Model never stops requesting tools                             | `session.steps >= max_steps` → `Error::MaxSteps`; `run` exits non-zero; `serve` returns a failed `response` but keeps serving                       | Caller issues a fresh `prompt`, or `new_session`                     |
| Malformed JSON on `serve`'s stdin                              | Parsed as `Value`, fails → `response{success:false, error:"invalid JSON: …"}`; loop continues                                                       | Client resends a valid command                                       |
| `serve`'s stdout write fails (broken pipe)                     | Writer task exits its loop; the next `emit!` send observes the closed channel and `break`s the main loop; process exits `Ok(())`                    | None by design — a client must reconnect via a new process           |
| Session file unreadable/corrupt at `serve` startup             | `load_session` logs the read/parse failure to stderr, then falls back to a fresh empty `Session` (a missing file stays silent — expected first run) | None automatic — prior transcript is lost, but the failure is logged |
| Session file write fails after a turn                          | Logged to stderr only (`eprintln!`); the in-memory session and the turn's `response` still report success                                           | None automatic — operator must notice the stderr line                |
| `edit`'s `old_string` matches 0 or >1 times (no `replace_all`) | `ToolError::InvalidInput`; the file is **not** modified                                                                                             | Model adds more surrounding context and retries                      |
| `grep`/`find` hit `HARD_CAP` (10,000 collected items)          | Walk quits early; output is still sorted+truncated deterministically, but which items entered the set before the cap can vary                       | Model narrows the `pattern`/`glob` and retries                       |
