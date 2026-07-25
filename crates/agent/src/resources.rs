//! System-prompt assembly: base prompt + project instructions + skills + environment.
//!
//! The model's effective instructions are built here rather than baked into a constant, so it sees
//! the project's own `AGENTS.md`/`CLAUDE.md` files, any discovered skills, and the current date/cwd —
//! the context a coding agent needs to behave like it belongs in *this* repo.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::path_utils::resolved_path;
use crate::skills::{self, Skill};

/// The agent's base identity/instructions. The tool list is generated from `registry` — the tools this
/// process actually registered, after any `--tools`/`--exclude-tools`/`--no-tools` filtering — rather
/// than hand-listed as a static string or assumed to be the full default set. A prior hardcoded version
/// silently omitted the Beyond platform tools (fork/sync/logs) entirely, and a version that always
/// listed `default_registry()` regardless of filtering would claim tools a restricted agent doesn't
/// actually have, inviting the model to call one that gets rejected.
///
/// `extra_guidelines` are operator-supplied bullets (`--prompt-guideline`, repeatable) appended after
/// the built-in ones — pi's own `promptGuidelines` (deduplicated and trimmed, matching pi's
/// `buildSystemPrompt`). Deliberately *not* a full port of pi's system prompt: pi also renders a
/// redundant per-tool text snippet list ("Available tools:\n- bash: Execute bash commands...")
/// alongside the native tool-call JSON schema already describing each tool to the model — the same
/// information twice, in two different places the model reads. This function's own dynamic tool-name
/// listing (`Use them to accomplish...with tools: {names}`) already avoids that duplication, so only
/// the genuinely useful, non-redundant half of pi's feature is ported here: the guideline-bullet
/// mechanism itself, including its one built-in conditional (`bash` registered but none of its usual
/// companions).
///
/// Lives here, not in `main.rs`, because a **subagent** must recompute it against its *own* (usually
/// restricted) registry: a child given `tools: read,grep` must not be told it has `bash` and `edit`.
pub fn default_system_prompt(
    registry: &agent_core::ToolRegistry,
    extra_guidelines: &[String],
) -> String {
    let names: Vec<String> = registry.definitions().into_iter().map(|d| d.name).collect();
    let has = |n: &str| names.iter().any(|x| x == n);

    let mut guidelines: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    let add = |g: String, guidelines: &mut Vec<String>, seen: &mut HashSet<String>| {
        if seen.insert(g.clone()) {
            guidelines.push(g);
        }
    };
    // Matches pi's own conditional exactly: when `bash` is the only exploration tool registered (none
    // of `grep`/`find`/`ls`), the model needs to be told it's the fallback for those operations too.
    if has("bash") && !has("grep") && !has("find") && !has("ls") {
        add(
            "Use bash for file operations like ls, rg, find".to_string(),
            &mut guidelines,
            &mut seen,
        );
    }
    // pi's own per-tool `promptGuidelines` (`read.ts`/`edit.ts`/`write.ts`) — declared on the tool
    // definition itself and collected from whatever's actually registered. Adapted, not ported
    // verbatim: pi's edit tool takes an `edits[].oldText`/`newText` array, ours takes `edits[].old_string`/
    // `new_string` (see `tools/edit.rs`'s own schema) — porting pi's exact field names would tell the
    // model to look for parameters that don't exist on our tool. `bash`/`grep`/`find`/`ls` carry no
    // `promptGuidelines` on pi's side, so there's nothing to port for those.
    if has("read") {
        add(
            "Use read to examine files instead of cat or sed.".to_string(),
            &mut guidelines,
            &mut seen,
        );
    }
    if has("edit") {
        for g in [
            "Use edit for precise changes (edits[].old_string must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple \
             entries in edits[] instead of multiple edit calls",
            "Each edits[].old_string is matched against the original file, not after earlier edits are \
             applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].old_string as small as possible while still being unique in the file. Do not \
             pad with large unchanged regions.",
        ] {
            add(g.to_string(), &mut guidelines, &mut seen);
        }
    }
    if has("write") {
        add(
            "Use write only for new files or complete rewrites.".to_string(),
            &mut guidelines,
            &mut seen,
        );
    }
    for g in extra_guidelines {
        let g = g.trim();
        if !g.is_empty() {
            add(g.to_string(), &mut guidelines, &mut seen);
        }
    }
    add(
        "Show file paths clearly when working with files".to_string(),
        &mut guidelines,
        &mut seen,
    );
    let guidelines = guidelines
        .into_iter()
        .map(|g| format!("- {g}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are the Beyond coding agent. You operate inside a real working directory with tools: {}. \
         Use them to accomplish the user's task directly — inspect before you change, make minimal \
         edits, and verify your work. Be concise.\n\nGuidelines:\n{guidelines}",
        names.join(", ")
    )
}

/// Options controlling how the system prompt is assembled.
pub struct PromptOptions<'a> {
    /// An explicit override for the base agent identity/instructions (`--system-prompt`), if the caller
    /// was given one. `None` means no explicit override was passed — a trusted project's on-disk
    /// `SYSTEM.md` still gets a chance to apply in that case, falling through to `default_base` only
    /// when that's also absent. Distinguishing "explicitly asked for X" from "nothing asked, use the
    /// built-in default" is what lets `SYSTEM.md` apply in exactly the cases pi's own resource-loader
    /// does (no explicit flag) without ever silently outranking an operator's own explicit
    /// `--system-prompt` — previously this distinction was collapsed away before ever reaching here
    /// (`base` was always a plain, already-defaulted `&str`), so a project's `SYSTEM.md` always won even
    /// over an explicit flag.
    pub base: Option<&'a str>,
    /// The computed built-in default base prompt ([`default_system_prompt`]), used only when
    /// neither `base` nor an on-disk `SYSTEM.md` override applies. Cheap and pure to compute (no I/O),
    /// so a caller builds it eagerly even on the common path where it goes unused.
    pub default_base: &'a str,
    /// Extra text appended after the base (e.g. `--append-system-prompt`). When `None`, an on-disk
    /// `APPEND_SYSTEM.md` (project, then user — same discovery/trust order as `SYSTEM.md`) is used
    /// instead, if one exists; an explicit `append` here wins outright rather than combining with it.
    pub append: Option<&'a str>,
    /// The working directory whose project-instruction files to load.
    pub cwd: &'a Path,
    /// Whether to discover and inject `AGENTS.md`/`CLAUDE.md` project-instruction files.
    pub include_context_files: bool,
    /// Already-discovered skills to advertise. Renders no `<available_skills>` section at all when
    /// empty, *or* when every skill in it is `disable-model-invocation` (see `skills::format_available`'s
    /// doc comment) — a non-empty list with nothing actually model-visible must not still produce an
    /// empty wrapper.
    /// Discovery itself — `skills::discover`/`discover_with_diagnostics` — is the caller's job, not
    /// this function's: every real caller (`main.rs`'s `run`, `serve.rs`'s startup/`reload`) already
    /// discovers skills separately for its own purposes (expanding a `/skill:name` invocation,
    /// `get_commands`'s collision diagnostics), so re-discovering here as well would walk the exact
    /// same directories twice per startup/reload for no reason. Pass `&[]` (and skip discovery
    /// upstream, matching `--no-skills`'s existing "skip outright rather than discover-then-discard"
    /// pattern) to build a prompt with no skills at all.
    pub skills: &'a [Skill],
    /// Whether the registered tool set includes `read`. A model with no way to open a file it doesn't
    /// already have inline can't act on a skill's own `SKILL.md` contents (skills are discovered by path,
    /// not inlined into the prompt — invoking one relies on the model reading the file itself), so
    /// advertising `<available_skills>` when `read` isn't registered just adds dead weight to the prompt:
    /// entries the model has no way to actually use. `false` skips the whole section regardless of
    /// `skills`'s own contents.
    pub has_read: bool,
    /// Whether the registered tool set includes `todo`. The tool is useless to a model that hasn't been
    /// told the protocol it expects — full replacement of the list on every call, exactly one item
    /// `in_progress` — and the guidance is dead weight in a prompt for a process (or a subagent child)
    /// whose registry never advertised the tool. Same gate-on-what's-registered discipline as
    /// [`has_read`](Self::has_read) and [`agents`](Self::agents).
    pub has_todo: bool,
    /// Whether the registered tool set includes `structured_output`. With no `tool_choice` forcing (it
    /// is per-request, so pinning it would stop the model doing any real work first, and the OpenAI Chat
    /// Completions dialect ignores it outright), *this prompt section is the forcing mechanism*: the tool
    /// itself ends the run, but nothing else tells the model to call it rather than answer in prose.
    pub has_structured_output: bool,
    /// Whether the registered tool set includes `memory`. Gates the `## Memory` guidance block (how to
    /// drive the tool and the curation discipline) — dead weight in a prompt for a process whose registry
    /// never advertised the tool, same gate-on-what's-registered discipline as [`has_todo`](Self::has_todo).
    pub has_memory: bool,
    /// The mounted memory stores to surface, each paired with its current, already-bounded `MEMORY.md`
    /// index (from [`crate::memory::MemoryBackend::index`]), in display order. Typically the durable
    /// `/memories` mount plus — when a session mount is active — the `/session` working-memory mount, so
    /// both are auto-surfaced at session start (Claude Code's auto-memory model). Each entry renders its
    /// own guidance + index (an empty index becomes an "index is empty" note). Only consulted when
    /// [`has_memory`](Self::has_memory) is set; an empty slice there falls back to a bare durable section.
    pub memory_sections: &'a [(crate::memory::MountKind, String)],
    /// Whether `cwd` is a trusted project (an explicit `--trust-project`/RPC override, or recorded in
    /// `TrustStore`). Gates the *project-local* `SYSTEM.md`/`APPEND_SYSTEM.md` overrides and the
    /// project-local skills root (`<cwd>/.claude/skills`, see `skills::discover`) — an untrusted
    /// checkout can't replace or extend the agent's identity or inject its own skills just by shipping
    /// the files (see `crate::trust_store`). The user-global `~/.claude/SYSTEM.md`/`APPEND_SYSTEM.md`
    /// overrides and `~/.claude/skills` are unaffected either way: they're the operator's own machine,
    /// not something a repo checkout controls.
    pub project_trusted: bool,
    /// Agent definitions to advertise in an `<available_agents>` block — the delegable personas the
    /// `subagent` tool accepts (see [`crate::agents`]). Renders nothing when empty, so a prompt built for
    /// a process (or a child) that has no `subagent` tool passes `&[]` and the section simply doesn't
    /// appear. Discovery is the caller's job, mirroring `skills` above.
    pub agents: &'a [crate::agents::AgentDef],
}

/// Build the full system prompt for a session: the static base (see [`build_static_system_prompt`])
/// plus the current dynamic footer (see [`dynamic_footer`]). A caller that rebuilds the prompt every
/// turn just to pick up the current date (e.g. `serve`'s per-turn refresh) should call the two pieces
/// separately instead — see `serve::build_agent`'s callers.
pub fn build_system_prompt(opts: &PromptOptions) -> String {
    let mut s = build_static_system_prompt(opts);
    s.push_str(&dynamic_footer(opts.cwd));
    s
}

/// [`build_system_prompt`] with the project-instruction files already resolved — the
/// [`build_static_system_prompt_with_context`] counterpart, plus the per-turn dynamic footer. Used by
/// the subagent fan-out to reuse one cwd→root walk across siblings sharing a root (T8-F9).
pub(crate) fn build_system_prompt_with_context(
    opts: &PromptOptions,
    context_files: &[(String, String)],
) -> String {
    let mut s = build_static_system_prompt_with_context(opts, context_files);
    s.push_str(&dynamic_footer(opts.cwd));
    s
}

/// Everything in the system prompt except the per-turn dynamic footer (current date/cwd, see
/// [`dynamic_footer`]): the base identity or its `SYSTEM.md` override, project instructions, and
/// discovered skills. Walks the filesystem (project-instruction files, skill directories), so it's
/// meant to be rebuilt only at startup and on a model/thinking-triggered `Agent` rebuild — not on every
/// turn, unlike the cheap dynamic footer.
pub fn build_static_system_prompt(opts: &PromptOptions) -> String {
    let context_files = if opts.include_context_files {
        load_context_files(opts.cwd)
    } else {
        Vec::new()
    };
    build_static_system_prompt_with_context(opts, &context_files)
}

/// [`build_static_system_prompt`] with the project-instruction files already resolved, rather than
/// walked from `opts.cwd` here. Lets a subagent fan-out (see [`crate::tools::subagent`]) walk cwd→root
/// ONCE and reuse the `(path, body)` result across sibling children that share a root, instead of every
/// child re-reading the same `AGENTS.md`/`CLAUDE.md` chain (T8-F9). `opts.include_context_files` is not
/// consulted here — the caller decides by what it passes (an empty slice renders no block, exactly as a
/// `false` flag would); the public wrapper above preserves the flag's meaning for every other caller.
pub(crate) fn build_static_system_prompt_with_context(
    opts: &PromptOptions,
    context_files: &[(String, String)],
) -> String {
    // An explicit `--system-prompt` (`opts.base`) wins outright, exactly like `opts.append` below —
    // never even consulting the on-disk `SYSTEM.md`. Only when no explicit override was given does a
    // trusted project's on-disk `SYSTEM.md` (project `<cwd>/.claude/`, else user `~/.claude/`) get a
    // chance to replace the built-in base — that's how a project pins its own agent identity (pi's
    // resource-loader does the same) — and only when *that's* absent too does `default_base` apply.
    let mut s = opts
        .base
        .map(str::to_string)
        .or_else(|| system_prompt_override(opts.cwd, opts.project_trusted))
        .unwrap_or_else(|| opts.default_base.to_string());
    let append = opts
        .append
        .map(str::to_string)
        .or_else(|| append_system_prompt_override(opts.cwd, opts.project_trusted));
    if let Some(extra) = append {
        s.push_str("\n\n");
        s.push_str(&extra);
    }

    if !context_files.is_empty() {
        // Wrapped in an outer `<project_context>` element (matching the reference agent) so the
        // model sees these as a distinct, labeled section rather than bare instruction blocks
        // floating in the prompt.
        s.push_str("\n\n<project_context>\n\n");
        s.push_str("Project-specific instructions and guidelines:\n\n");
        for (path, body) in context_files {
            s.push_str(&format!(
                "<project_instructions path=\"{path}\">\n{}\n</project_instructions>\n\n",
                body.trim()
            ));
        }
        s.push_str("</project_context>");
    }

    // pi-parity fix (M1): checking `!opts.skills.is_empty()` guards on the *unfiltered* list — a
    // non-empty list where every skill is `disable-model-invocation` still built an empty
    // `<available_skills>…</available_skills>` shell. `format_available` itself now returns `""` in
    // that case (see its doc comment); check its actual output instead of the raw skill count.
    //
    // pi-parity fix: also gated on `has_read` — advertising skills to a model with no way to open the
    // file a skill invocation points at (skills are discovered by path, not inlined; see `has_read`'s
    // own doc comment) just adds dead weight the model can never act on.
    if opts.has_read {
        let available = skills::format_available(opts.skills);
        if !available.is_empty() {
            s.push_str("\n\n");
            s.push_str(&available);
        }
    }

    // The delegable agents, when this prompt is for a process that actually has the `subagent` tool. A
    // caller without it passes `&[]`; `format_available` returns "" for that, so no empty shell appears.
    let agents = crate::agents::format_available(opts.agents);
    if !agents.is_empty() {
        s.push_str("\n\n");
        s.push_str(&agents);
    }

    // The `todo` protocol, when the tool is actually registered. Its schema can state the shape of a
    // call but not *when* to make one, nor that the list is a full replacement rather than a delta —
    // the two things a model gets wrong without being told.
    if opts.has_todo {
        s.push_str("\n\n");
        s.push_str(TODO_GUIDANCE);
    }

    // The `structured_output` contract. See `PromptOptions::has_structured_output`: with no
    // `tool_choice` forcing, this is what actually makes the model return typed data instead of prose.
    if opts.has_structured_output {
        s.push_str("\n\n");
        s.push_str(STRUCTURED_OUTPUT_GUIDANCE);
    }

    // Memory: each mounted store's guidance block plus its current (already-bounded) MEMORY.md index,
    // auto-injected so a memory is never silently forgotten. Gated on the tool being registered; an
    // empty section list still surfaces the durable store's guidance so the model knows it exists.
    if opts.has_memory {
        let durable_fallback = [(crate::memory::MountKind::Durable, String::new())];
        let sections = if opts.memory_sections.is_empty() {
            &durable_fallback[..]
        } else {
            opts.memory_sections
        };
        s.push_str("\n\n");
        s.push_str(&crate::memory::render_sections(sections));
    }

    s
}

/// How to drive the `structured_output` tool. Gated on [`PromptOptions::has_structured_output`].
const STRUCTURED_OUTPUT_GUIDANCE: &str = "\
<structured_output_protocol>
This run must end by calling the `structured_output` tool exactly once, with the task's result \
conforming to that tool's schema. Do the work first, using whatever tools you need; then call \
`structured_output`. Do not describe the result in prose — return it through the tool, which ends the \
run. If the payload you send does not match the schema, you will be told what was wrong and must call \
the tool again with a corrected one.
</structured_output_protocol>";

/// How to drive the `todo` tool. Gated on [`PromptOptions::has_todo`].
const TODO_GUIDANCE: &str = "\
<todo_protocol>
You have a `todo` tool for planning multi-step work. Use it for any task that takes more than a couple \
of steps, or when the user gives you several things to do. Skip it for trivial single-step requests \
where a plan adds nothing.

Call `todo` with the COMPLETE list every time — it fully replaces the previous list, so always include \
every item, not just the one that changed. Give each item a `content` (imperative: \"Add the retry \
loop\"), an `activeForm` (present continuous: \"Adding the retry loop\"), and a `status` of `pending`, \
`in_progress`, or `completed`.

Keep exactly one item `in_progress` at a time: mark an item `in_progress` right before you start it, \
and `completed` the moment it is done — don't batch completions. Send an empty list to clear the plan \
once the work is finished.
</todo_protocol>";

/// The cheap, time-varying tail of the system prompt: the current date and working directory. Does no
/// filesystem discovery (unlike [`build_static_system_prompt`]), so it's cheap enough to recompute
/// before every turn — the one part of the prompt that's actually time-varying.
///
/// pi-parity fix (Task #43, cosmetic): pi's own `system-prompt.ts` (~168-170) appends these two lines
/// with a single `\n` each (`prompt += "\nCurrent date: ..."; prompt += "\nCurrent working directory:
/// ..."`), not a blank-line separator — this used to unconditionally prefix `"\n\n"`, guaranteeing a
/// blank line before "Current date" no matter what the preceding content ended with. Matching pi exactly
/// means a blank line only appears here when the preceding content already ends in its own trailing
/// newline (the same as pi), rather than being forced every time.
pub fn dynamic_footer(cwd: &Path) -> String {
    format!(
        "\nCurrent date: {}\nCurrent working directory: {}",
        today(),
        cwd.display()
    )
}

/// A `SYSTEM.md` override: project-local (`<cwd>/.claude/SYSTEM.md`, only when `project_trusted`) takes
/// precedence over the user one (`~/.claude/SYSTEM.md`, always eligible — it's the operator's own
/// machine). Returns its raw contents, or `None` when neither exists / is blank / the project one
/// exists but isn't trusted (falls through to the global candidate exactly as if the project file were
/// simply absent — no different fallback than the untrusted-file-doesn't-exist case).
fn system_prompt_override(cwd: &Path, project_trusted: bool) -> Option<String> {
    discover_claude_file(cwd, project_trusted, "SYSTEM.md")
}

/// An `APPEND_SYSTEM.md` on disk (same project-then-user discovery/trust order as `SYSTEM.md`) is
/// additive rather than a replacement: its contents are appended after the base/override system prompt.
/// Only consulted when the caller didn't already supply an explicit `append` (e.g.
/// `--append-system-prompt`) — an explicit override wins outright rather than combining with the
/// on-disk file, matching pi's `resource-loader.ts` (`appendSystemPromptSource ?? discovered`).
fn append_system_prompt_override(cwd: &Path, project_trusted: bool) -> Option<String> {
    discover_claude_file(cwd, project_trusted, "APPEND_SYSTEM.md")
}

/// Shared project-then-user `.claude/<filename>` discovery: project-local (only when `project_trusted`)
/// takes precedence over the user-global one (always eligible — it's the operator's own machine).
/// Returns `None` when neither exists / is blank / the project one exists but isn't trusted (falls
/// through to the global candidate exactly as if the project file were simply absent).
fn discover_claude_file(cwd: &Path, project_trusted: bool, filename: &str) -> Option<String> {
    let mut candidates = Vec::new();
    if project_trusted {
        candidates.push(cwd.join(".claude").join(filename));
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join(".claude").join(filename));
    }
    for path in candidates {
        if let Ok(body) = fs::read_to_string(&path) {
            if !body.trim().is_empty() {
                return Some(body);
            }
        }
    }
    None
}

/// Collect project-instruction files as `(path, body)`, nearest-last so the model reads the most
/// specific instructions (the cwd's) after the broader ones: first the global `~/.claude` file, then
/// one file per directory from the filesystem root down to `cwd`.
///
/// At most one file is taken per directory: when both `AGENTS.md` and `CLAUDE.md` are present,
/// `AGENTS.md` wins (matching pi's resource-loader). Filename matching is case-insensitive.
pub fn load_context_files(cwd: &Path) -> Vec<(String, String)> {
    load_context_files_with_home(cwd, std::env::var_os("HOME").map(PathBuf::from).as_deref())
}

/// [`load_context_files`], with the global home directory injected rather than read from `$HOME` —
/// lets a test exercise the `cwd`-is-under-`~/.claude` dedup case deterministically, without racing
/// real env-var state across parallel test threads.
///
/// `home`'s `.claude` subdirectory is deduped against the ancestor walk below by canonical path
/// (`path_utils::resolved_path`), not skipped by construction: when `cwd` is `~/.claude` itself or a
/// descendant of it, the ancestor walk would otherwise revisit the exact same directory the global
/// lookup already read, injecting its contents into the system prompt twice.
fn load_context_files_with_home(cwd: &Path, home: Option<&Path>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    // Global instruction file (one, AGENTS-wins).
    if let Some(home) = home {
        let global_dir = home.join(".claude");
        seen.insert(resolved_path(&global_dir));
        if let Some(file) = load_context_file_from_dir(&global_dir) {
            out.push(file);
        }
    }

    // Walk cwd → root, collecting each ancestor's file; reverse so the deepest (cwd) lands last.
    let mut ancestors: Vec<&Path> = cwd.ancestors().collect();
    ancestors.reverse(); // root-most first
    for dir in ancestors {
        if !seen.insert(resolved_path(dir)) {
            continue;
        }
        if let Some(file) = load_context_file_from_dir(dir) {
            out.push(file);
        }
    }
    out
}

/// Pick this directory's context file: `AGENTS.md` if it reads cleanly and is non-empty, else
/// `CLAUDE.md` under the same conditions (matched case-insensitively). A directory we can't even list
/// yields nothing.
///
/// Matches pi's own `loadContextFileFromDir`: a read failure on the first candidate (permission
/// denied, a dangling symlink) logs a warning and falls through to the next one, rather than
/// abandoning the whole directory — the prior version committed to `AGENTS.md` the moment it was
/// *seen*, so `fs::read_to_string` failing on it returned `None` outright and a perfectly good sibling
/// `CLAUDE.md` was silently never tried. An empty `AGENTS.md` falls through the same way (Beyond
/// deliberately treats an empty file as no file at all, unlike pi, which stops there and includes the
/// empty content as-is — a real, intentional divergence, not a gap to close), so this also fixes the
/// same class of bug for that trigger: "absent" must mean "try the next candidate," not "give up on
/// the directory."
///
/// **Deliberate divergence, awareness-only: filename matching here is genuinely case-insensitive
/// (`to_ascii_lowercase() == "agents.md"`), not just tolerant of a few specific spellings.** pi's own
/// `resource-loader.ts` checks an exact 4-candidate list — `["AGENTS.md", "AGENTS.MD", "CLAUDE.md",
/// "CLAUDE.MD"]` via `existsSync` — which is really a filesystem-case-sensitivity accident, not a
/// deliberate case-insensitive spec: on a case-insensitive filesystem (macOS/Windows default) any of
/// those 4 literal strings resolves to the same file regardless of its actual on-disk casing, so the
/// list works there for any casing at all; on a case-sensitive one (most Linux setups, this codebase's
/// primary target), a real `agents.md` or `Agents.MD` matches none of the 4 and is silently invisible to
/// pi. Beyond's lowercase-compare is a strict superset of pi's 4-string list on every filesystem, so
/// this isn't something to narrow down to match — it can only recognize a project-instruction file pi
/// would miss, never the reverse.
fn load_context_file_from_dir(dir: &Path) -> Option<(String, String)> {
    let mut agents: Option<PathBuf> = None;
    let mut claude: Option<PathBuf> = None;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let lower = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if lower == "agents.md" && agents.is_none() {
            agents = Some(entry.path());
        } else if lower == "claude.md" && claude.is_none() {
            claude = Some(entry.path());
        }
    }
    for candidate in [agents, claude].into_iter().flatten() {
        match fs::read_to_string(&candidate) {
            Ok(body) if !body.trim().is_empty() => {
                return Some((candidate.display().to_string(), body));
            }
            Ok(_) => {} // empty file: treated as absent, try the next candidate
            Err(e) => {
                tracing::warn!(
                    path = %candidate.display(),
                    error = %e,
                    "could not read project-instruction file, trying the next candidate"
                );
            }
        }
    }
    None
}

/// Today's date as `YYYY-MM-DD` in the host's local timezone, without pulling in a date crate. Local
/// (not UTC) so the injected date never reads a day behind/ahead of the user near midnight.
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // The date itself is recomputed every call (it must stay current across midnight), but the local
    // UTC offset is process-constant in practice, so it's cached rather than re-read+re-parsed from
    // `/etc/localtime` (or the `TZ`-named zoneinfo) on every turn — see [`cached_local_utc_offset`].
    let local = secs + cached_local_utc_offset(secs);
    format_civil_datetime(local, false)
}

/// The host's local UTC offset, memoized for the hot dynamic-footer path ([`today`], rebuilt before
/// every turn). Resolving it means reading and TZif-parsing `/etc/localtime` (or the zoneinfo file the
/// `TZ` env var names) — pure syscalls+parsing to recover a value that is constant for the life of the
/// process barring a `TZ` change or a DST transition. The cache is keyed on the current `TZ` env value,
/// so a process that changes `TZ` mid-run still refreshes; a DST transition mid-session is the one
/// tolerated staleness (an at-most-one-hour skew in the *displayed* date near the boundary), the same
/// bound the audit accepted for this per-turn recompute.
///
/// Deliberately NOT used by [`format_local_datetime`] (export rendering): that formats arbitrary
/// historical `created_at` timestamps, whose correct offset genuinely depends on the timestamp (a
/// winter session exported in summer must render with the winter offset), so it keeps resolving the
/// per-timestamp offset directly.
fn cached_local_utc_offset(now: i64) -> i64 {
    static CACHE: Mutex<Option<(Option<String>, i64)>> = Mutex::new(None);
    let tz = std::env::var("TZ").ok();
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cached_tz, offset)) = guard.as_ref() {
        if *cached_tz == tz {
            return *offset;
        }
    }
    let offset = local_utc_offset(now);
    *guard = Some((tz, offset));
    offset
}

/// Format a unix-seconds timestamp as a human-readable local date/time string, without pulling in a
/// date crate — reuses the exact same [`local_utc_offset`]/[`civil_from_days`] machinery [`today`]
/// already relies on for the system prompt's dynamic footer (same rationale: local, not UTC, so the
/// rendered moment never reads a day behind/ahead of the viewer near midnight), just also keeping the
/// time-of-day component `today` itself discards. Used by `export::render_stats_section` for a
/// session's `created_at` (pi-parity fix: pi's own export always renders a `Date:` line, `new
/// Date(...).toLocaleString()`) — `export.rs` has no date-formatting convention of its own to diverge
/// from, so this is it.
pub(crate) fn format_local_datetime(secs: u64) -> String {
    format_local_datetime_at(secs as i64, true)
}

/// Shared implementation behind [`today`]/[`format_local_datetime`] — `with_time` selects whether the
/// `HH:MM:SS` suffix is included, so `today`'s existing `YYYY-MM-DD`-only output stays byte-for-byte
/// unchanged. Resolves the local offset (the one host-dependent, not-purely-deterministic step) and
/// hands off to [`format_civil_datetime`] for the actual formatting, so that part stays independently
/// testable against fixed inputs regardless of the host's own timezone.
fn format_local_datetime_at(secs: i64, with_time: bool) -> String {
    let local = secs + local_utc_offset(secs);
    format_civil_datetime(local, with_time)
}

/// Format `local_secs` (unix seconds already shifted into whatever timezone the caller wants shown) as
/// `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS` — pure date/time-of-day math over [`civil_from_days`], no
/// timezone lookup of its own, so it's deterministic for a fixed input regardless of the host running
/// it (unlike [`format_local_datetime_at`], which resolves the host's own local offset first).
fn format_civil_datetime(local_secs: i64, with_time: bool) -> String {
    let (y, m, d) = civil_from_days(local_secs.div_euclid(86_400));
    if !with_time {
        return format!("{y:04}-{m:02}-{d:02}");
    }
    let time_of_day = local_secs.rem_euclid(86_400);
    let (hh, mm, ss) = (
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60,
    );
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02}")
}

/// The host's UTC offset in seconds at `now`. We have no date crate and `unsafe_code` is forbidden (so
/// no libc `localtime`); parsing a TZif file is the dependency-free, safe way to get the local offset.
/// `TZ` takes precedence over the system zoneinfo (`/etc/localtime`) when it names a resolvable zone —
/// the precedence every libc/ICU implementation uses, and the one that matters in a container: `TZ` is
/// routinely set to override a base image whose `/etc/localtime` is still UTC. Any problem — `TZ`
/// unset/unresolvable, missing file, unknown format — degrades to `0` (UTC), which is correct, just not
/// local.
fn local_utc_offset(now: i64) -> i64 {
    tz_env_offset(now)
        .or_else(|| {
            fs::read("/etc/localtime")
                .ok()
                .and_then(|data| tzif_offset(&data, now))
        })
        .unwrap_or(0)
}

/// The offset from the `TZ` environment variable, if set to a resolvable zone. `TZ` unset falls
/// through to `None` so the caller tries `/etc/localtime` next.
fn tz_env_offset(now: i64) -> Option<i64> {
    let tz = std::env::var("TZ").ok()?;
    tz_string_offset(&tz, now)
}

/// Resolve a `TZ`-style value — an IANA zone name (e.g. `America/New_York`, optionally `:`-prefixed per
/// POSIX convention) or an absolute zoneinfo path — to its UTC offset at `now`. The form containers/CI
/// set almost universally. A raw POSIX offset-rule string (`EST5EDT,M3.2.0,M11.1.0`) isn't parsed (no
/// rule-transition logic here, just TZif files); empty-but-for-the-`:`, or naming something we can't
/// resolve to a real zoneinfo file, falls through to `None`.
///
/// Split from [`tz_env_offset`] so the zone-resolution logic is unit-testable without mutating the
/// process's environment — `std::env::set_var` is unsafe to call from a test that may run in parallel
/// with others reading `TZ`.
fn tz_string_offset(tz: &str, now: i64) -> Option<i64> {
    let zone = tz.strip_prefix(':').unwrap_or(tz);
    if zone.is_empty() {
        return Some(0); // POSIX: an empty TZ value means UTC
    }
    let path = if zone.starts_with('/') {
        PathBuf::from(zone)
    } else {
        Path::new("/usr/share/zoneinfo").join(zone)
    };
    let data = fs::read(path).ok()?;
    tzif_offset(&data, now)
}

/// Extract the UTC offset applicable at `now` from a TZif (RFC 8536) blob. A v2/v3 file carries a
/// second block with 64-bit transition times after the legacy 32-bit one; that block is authoritative.
fn tzif_offset(data: &[u8], now: i64) -> Option<i64> {
    if data.get(0..4)? != b"TZif" {
        return None;
    }
    let version = *data.get(4)?;
    let (offset32, consumed) = tzif_block(data, 0, 4, now)?;
    if version == b'2' || version == b'3' {
        let (offset64, _) = tzif_block(data, consumed, 8, now)?;
        Some(offset64)
    } else {
        Some(offset32)
    }
}

/// Parse one TZif header+data block starting at `start`, with transition times `time_size` bytes wide
/// (4 for the v1 block, 8 for the v2/v3 block). Returns `(offset_seconds_at_now, end_index)`.
fn tzif_block(data: &[u8], start: usize, time_size: usize, now: i64) -> Option<(i64, usize)> {
    let header = data.get(start..start + 44)?;
    if header.get(0..4)? != b"TZif" {
        return None;
    }
    // Six 4-byte counts begin at header offset 20 (after magic + version + 15 reserved bytes).
    let isutcnt = be32(header, 20)? as usize;
    let isstdcnt = be32(header, 24)? as usize;
    let leapcnt = be32(header, 28)? as usize;
    let timecnt = be32(header, 32)? as usize;
    let typecnt = be32(header, 36)? as usize;
    let charcnt = be32(header, 40)? as usize;

    let trans = start + 44;
    let type_idx = trans + timecnt * time_size;
    let ttinfo = type_idx + timecnt;
    let names = ttinfo + typecnt * 6;
    let leaps = names + charcnt;
    let end = leaps + leapcnt * (time_size + 4) + isstdcnt + isutcnt;
    if end > data.len() {
        return None;
    }

    // The active type is that of the latest transition at or before `now` (transitions are ascending).
    let mut active: Option<usize> = None;
    for k in 0..timecnt {
        if read_int(data, trans + k * time_size, time_size)? <= now {
            active = Some(*data.get(type_idx + k)? as usize);
        } else {
            break;
        }
    }
    // Before the first transition, fall back to the first non-DST type, else type 0.
    let ti = active.unwrap_or_else(|| {
        (0..typecnt)
            .find(|&t| data.get(ttinfo + t * 6 + 4) == Some(&0))
            .unwrap_or(0)
    });
    let ti = if ti < typecnt { ti } else { 0 };
    // A ttinfo record is 6 bytes: i32 gmtoff, u8 isdst, u8 abbrind.
    let gmtoff = read_int(data, ttinfo + ti * 6, 4)?;
    Some((gmtoff, end))
}

/// Read a big-endian signed integer of `size` (4 or 8) bytes at offset `o`.
fn read_int(b: &[u8], o: usize, size: usize) -> Option<i64> {
    let s = b.get(o..o + size)?;
    if size == 8 {
        Some(i64::from_be_bytes(s.try_into().ok()?))
    } else {
        Some(i32::from_be_bytes(s.try_into().ok()?) as i64)
    }
}

/// Read a big-endian `u32` at offset `o`.
fn be32(b: &[u8], o: usize) -> Option<u32> {
    let s = b.get(o..o + 4)?;
    Some(u32::from_be_bytes(s.try_into().ok()?))
}

/// Convert a count of days since the Unix epoch to a `(year, month, day)` civil date. Howard
/// Hinnant's `civil_from_days` algorithm — exact, branch-light, no leap-year special cases.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_local_utc_offset_matches_the_uncached_resolution_at_the_current_time() {
        // The cache resolves the offset for whatever `now` first populated it (keyed on `TZ`), which is
        // sound precisely because its one hot caller, `today()`, always passes ~the current time — so a
        // query at the current time must equal the uncached resolution at that same time. (A query at an
        // arbitrary *historical* `now` deliberately would NOT match: that's the documented DST-tolerance
        // trade-off, and why `format_local_datetime`'s export path resolves the offset directly instead.)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        assert_eq!(cached_local_utc_offset(now), local_utc_offset(now));
    }

    // Micro-bench (T8-F1): run with `cargo test -p beyond-ai-agent --lib -- --ignored --nocapture
    // offset_micro_bench` to see ns/call for the uncached disk+TZif path vs the cached path.
    #[test]
    #[ignore = "micro-bench, prints timings; not an assertion"]
    fn offset_micro_bench() {
        let now = 1_700_000_000;
        let iters = 200_000;
        // Warm the cache so we measure steady-state, not the first miss.
        let _ = cached_local_utc_offset(now);

        let t0 = std::time::Instant::now();
        let mut acc = 0i64;
        for _ in 0..iters {
            acc = acc.wrapping_add(local_utc_offset(std::hint::black_box(now)));
        }
        let uncached = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            acc = acc.wrapping_add(cached_local_utc_offset(std::hint::black_box(now)));
        }
        let cached = t1.elapsed().as_nanos() as f64 / iters as f64;

        std::hint::black_box(acc);
        println!("local_utc_offset: uncached {uncached:.1} ns/call, cached {cached:.1} ns/call");
    }

    #[test]
    fn default_system_prompt_lists_every_registered_tool() {
        // The whole point of generating this dynamically: it can't silently omit a tool the way the
        // prior hardcoded string did (it never mentioned the Beyond platform tools at all).
        let registry = crate::tools::default_registry();
        let prompt = default_system_prompt(&registry, &[]);
        for def in crate::tools::default_registry().definitions() {
            assert!(
                prompt.contains(&def.name),
                "system prompt is missing registered tool {:?}: {prompt}",
                def.name
            );
        }
    }

    #[test]
    fn default_system_prompt_reflects_a_restricted_registry() {
        // A tool-restricted agent's own system prompt must not claim tools it doesn't actually have —
        // otherwise the model is invited to call one that's guaranteed to be rejected.
        let mut registry = crate::tools::default_registry();
        crate::tools::apply_filter(&mut registry, None, Some(&["bash".to_string()]), false);
        let prompt = default_system_prompt(&registry, &[]);
        assert!(!prompt.contains("bash"));
        assert!(prompt.contains("read"));
    }

    #[test]
    fn default_system_prompt_always_shows_the_file_paths_guideline() {
        // pi: system-prompt.test.ts, "shows file paths guideline even with no tools" — a built-in
        // guideline, always present regardless of the tool set (unlike the conditional bash one below).
        let registry = crate::tools::default_registry();
        let prompt = default_system_prompt(&registry, &[]);
        assert!(prompt.contains("Show file paths clearly when working with files"));
    }

    #[test]
    fn default_system_prompt_tells_the_model_to_use_bash_for_exploration_without_grep_find_ls() {
        // pi: the one built-in conditional guideline — only fires when `bash` is registered but none
        // of its usual companions are, since the model then has no other way to explore the filesystem.
        let mut only_bash = agent_core::tool::ToolRegistry::new();
        only_bash.register(std::sync::Arc::new(crate::tools::bash::Bash::real()));
        let prompt = default_system_prompt(&only_bash, &[]);
        assert!(prompt.contains("Use bash for file operations like ls, rg, find"));

        // The guideline must not fire when grep/find/ls are also registered — bash isn't the only
        // exploration tool anymore.
        let full = crate::tools::default_registry();
        let prompt = default_system_prompt(&full, &[]);
        assert!(!prompt.contains("Use bash for file operations like ls, rg, find"));
    }

    #[test]
    fn default_system_prompt_includes_pis_per_tool_guidelines_for_read_edit_write() {
        // pi-parity fix: pi declares real default guidance on its read/edit/write tool definitions
        // (`promptGuidelines`), collected automatically whenever the tool is registered — we ported
        // only the operator-typed `--prompt-guideline` mechanism, not this content, so a model never
        // got told (for example) edit's exact-match/non-overlapping-edit semantics unless an operator
        // happened to type the same guidance in by hand.
        let registry = crate::tools::default_registry();
        let prompt = default_system_prompt(&registry, &[]);
        assert!(
            prompt.contains("Use read to examine files instead of cat or sed."),
            "got: {prompt}"
        );
        assert!(
            prompt.contains("Use edit for precise changes (edits[].old_string must match exactly)"),
            "got: {prompt}"
        );
        assert!(
            prompt.contains("Keep edits[].old_string as small as possible"),
            "got: {prompt}"
        );
        assert!(
            prompt.contains("Use write only for new files or complete rewrites."),
            "got: {prompt}"
        );
    }

    #[test]
    fn default_system_prompt_omits_a_tools_guidelines_when_the_tool_is_not_registered() {
        let mut only_bash = agent_core::tool::ToolRegistry::new();
        only_bash.register(std::sync::Arc::new(crate::tools::bash::Bash::real()));
        let prompt = default_system_prompt(&only_bash, &[]);
        assert!(!prompt.contains("Use read to examine files"));
        assert!(!prompt.contains("Use edit for precise changes"));
        assert!(!prompt.contains("Use write only for new files"));
    }

    #[test]
    fn default_system_prompt_appends_and_dedupes_extra_guidelines() {
        // pi: system-prompt.test.ts, "appends promptGuidelines to default guidelines" /
        // "deduplicates and trims promptGuidelines".
        let registry = crate::tools::default_registry();
        let prompt = default_system_prompt(
            &registry,
            &[
                "Use dynamic_tool for project summaries.".to_string(),
                "  Use dynamic_tool for project summaries.  ".to_string(),
                "   ".to_string(),
            ],
        );
        assert_eq!(
            prompt
                .matches("- Use dynamic_tool for project summaries.")
                .count(),
            1,
            "got: {prompt}"
        );
    }
    use std::fs;

    #[test]
    fn civil_date_matches_known_epochs() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(18_993), (2022, 1, 1));
        // 2020-02-29 (a leap day) is day 18321.
        assert_eq!(civil_from_days(18_321), (2020, 2, 29));
    }

    #[test]
    fn format_civil_datetime_renders_the_epoch_and_a_known_date() {
        // Pure date/time-of-day math, no timezone lookup involved — deterministic regardless of the
        // host running the test (unlike `format_local_datetime`/`today`, which resolve the host's own
        // local offset first).
        assert_eq!(format_civil_datetime(0, true), "1970-01-01 00:00:00");
        assert_eq!(format_civil_datetime(0, false), "1970-01-01");
        // One second before the epoch rolls back to the last second of 1969.
        assert_eq!(format_civil_datetime(-1, true), "1969-12-31 23:59:59");
        // 2022-01-01T00:00:00Z is day 18993 (see `civil_date_matches_known_epochs`) plus a mid-day
        // time-of-day component, exercising the HH:MM:SS math too.
        assert_eq!(
            format_civil_datetime(18_993 * 86_400 + 12 * 3600 + 34 * 60 + 56, true),
            "2022-01-01 12:34:56"
        );
    }

    #[test]
    fn context_files_are_nearest_last() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let nested = root.join("a/b");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("CLAUDE.md"), "root rules").unwrap();
        fs::write(nested.join("AGENTS.md"), "nested rules").unwrap();

        let files = load_context_files(&nested);
        let joined: Vec<&str> = files.iter().map(|(_, b)| b.as_str()).collect();
        // Both present, root-most before nearest.
        let root_pos = joined
            .iter()
            .position(|b| b.contains("root rules"))
            .unwrap();
        let nested_pos = joined
            .iter()
            .position(|b| b.contains("nested rules"))
            .unwrap();
        assert!(
            root_pos < nested_pos,
            "nearest file must come last: {joined:?}"
        );
    }

    #[test]
    fn dedupes_the_global_home_claude_file_when_cwd_is_under_it() {
        // Pi-parity fix: when `cwd` is `~/.claude` itself or a descendant of it, the ancestor walk
        // would revisit the exact same directory the global `~/.claude` lookup already read — the same
        // file's content injected into the system prompt twice.
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        let claude_dir = home.join(".claude");
        let nested = claude_dir.join("some-project");
        fs::create_dir_all(&nested).unwrap();
        fs::write(claude_dir.join("AGENTS.md"), "global-instructions-marker").unwrap();

        let files = load_context_files_with_home(&nested, Some(home));
        let occurrences = files
            .iter()
            .filter(|(_, body)| body.contains("global-instructions-marker"))
            .count();
        assert_eq!(
            occurrences, 1,
            "the global ~/.claude file must appear exactly once, not once per ancestor-walk revisit: \
             {files:#?}"
        );
    }

    #[test]
    fn one_file_per_dir_prefers_agents() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("proj");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("AGENTS.md"), "agents-wins").unwrap();
        fs::write(dir.join("CLAUDE.md"), "claude-loses").unwrap();

        let joined: String = load_context_files(&dir)
            .iter()
            .map(|(_, b)| b.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("agents-wins"));
        assert!(
            !joined.contains("claude-loses"),
            "CLAUDE.md must be skipped when AGENTS.md is present"
        );
    }

    #[test]
    fn an_empty_agents_md_falls_back_to_a_readable_claude_md() {
        // Pi-parity audit H67: the prior version resolved `chosen = agents.or(claude)` and read
        // *that one name only* — an empty `AGENTS.md` returned `None` for the whole directory,
        // discarding a perfectly good sibling `CLAUDE.md` instead of trying it.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("proj");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("AGENTS.md"), "   \n\n").unwrap();
        fs::write(dir.join("CLAUDE.md"), "claude-still-wins").unwrap();

        let joined: String = load_context_files(&dir)
            .iter()
            .map(|(_, b)| b.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("claude-still-wins"),
            "an empty AGENTS.md must fall back to a readable CLAUDE.md: {joined:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_agents_md_falls_back_to_a_readable_claude_md() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("proj");
        fs::create_dir_all(&dir).unwrap();
        let agents_path = dir.join("AGENTS.md");
        fs::write(&agents_path, "unreadable").unwrap();
        fs::set_permissions(&agents_path, fs::Permissions::from_mode(0o000)).unwrap();
        // Some environments (root, certain sandboxes) don't actually enforce permission bits — skip
        // rather than assert a false failure if the mode change didn't block the read at all.
        let mode_actually_blocks_reads = fs::read_to_string(&agents_path).is_err();
        fs::write(dir.join("CLAUDE.md"), "claude-still-wins").unwrap();

        let result = load_context_files(&dir);
        // Restore permissions before any assertion can panic, so the tempdir cleans up.
        let _ = fs::set_permissions(&agents_path, fs::Permissions::from_mode(0o644));

        if !mode_actually_blocks_reads {
            return;
        }
        let joined: String = result
            .iter()
            .map(|(_, b)| b.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("claude-still-wins"),
            "an unreadable AGENTS.md must fall back to a readable CLAUDE.md: {joined:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_broken_symlink_named_agents_md_falls_back_to_a_readable_claude_md() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("proj");
        fs::create_dir_all(&dir).unwrap();
        std::os::unix::fs::symlink(dir.join("does-not-exist"), dir.join("AGENTS.md")).unwrap();
        fs::write(dir.join("CLAUDE.md"), "claude-still-wins").unwrap();

        let joined: String = load_context_files(&dir)
            .iter()
            .map(|(_, b)| b.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("claude-still-wins"),
            "a broken symlink named AGENTS.md must fall back to a readable CLAUDE.md: {joined:?}"
        );
    }

    #[test]
    fn context_filenames_are_case_insensitive() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("proj");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("Claude.MD"), "mixed-case rules").unwrap();

        let joined: String = load_context_files(&dir)
            .iter()
            .map(|(_, b)| b.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("mixed-case rules"));
    }

    #[test]
    fn system_md_overrides_the_computed_default_base_when_trusted_and_no_explicit_flag_was_given() {
        // No explicit `--system-prompt` here (`base: None`) — a trusted project's on-disk `SYSTEM.md`
        // still gets to replace the computed built-in default in that case. See the sibling test below
        // for the case this one used to get wrong: an *explicit* `--system-prompt` must win outright
        // instead of being silently overridden the same way.
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("SYSTEM.md"), "OVERRIDE IDENTITY").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: None,
            default_base: "DEFAULT IDENTITY",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: true,
            agents: &[],
        });
        assert!(prompt.contains("OVERRIDE IDENTITY"));
        assert!(
            !prompt.contains("DEFAULT IDENTITY"),
            "a trusted project's on-disk SYSTEM.md must replace the computed default base when no \
             explicit --system-prompt was given"
        );
    }

    #[test]
    fn an_explicit_system_prompt_wins_outright_over_a_trusted_on_disk_system_md() {
        // pi-parity fix: previously `PromptOptions::base` was always a plain, already-defaulted `&str`
        // with no way to tell "this is an explicit --system-prompt" apart from "this is just the
        // built-in default" — so a trusted project's on-disk SYSTEM.md silently overrode an operator's
        // own explicit flag too, exactly like it does the computed default. An explicit override must
        // win outright instead, matching `append`'s own (always-correct) precedence.
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("SYSTEM.md"), "ON-DISK IDENTITY").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: Some("EXPLICIT IDENTITY"),
            default_base: "DEFAULT IDENTITY",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: true,
            agents: &[],
        });
        assert!(prompt.contains("EXPLICIT IDENTITY"));
        assert!(
            !prompt.contains("ON-DISK IDENTITY"),
            "an explicit --system-prompt must win outright over a trusted project's on-disk SYSTEM.md"
        );
    }

    #[test]
    fn system_md_is_ignored_when_project_is_untrusted() {
        // The whole point of the trust gate: an untrusted checkout can't hijack the agent's identity
        // just by shipping a `.claude/SYSTEM.md` — it falls through exactly as if the file were
        // absent, not a different (or erroring) fallback.
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("SYSTEM.md"), "MALICIOUS OVERRIDE").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: None,
            default_base: "DEFAULT IDENTITY",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: false,
            agents: &[],
        });
        assert!(prompt.contains("DEFAULT IDENTITY"));
        assert!(!prompt.contains("MALICIOUS OVERRIDE"));
    }

    #[test]
    fn build_static_system_prompt_advertises_the_passed_in_skills_without_discovering_its_own() {
        // LOW pi-parity gap (fixed): this used to call `skills::discover` itself, re-walking the exact
        // same directories every caller had already walked for its own purposes (expanding a
        // `/skill:name` invocation, `get_commands`'s collision diagnostics) — a second filesystem walk
        // per startup/reload for no reason. `opts.cwd` here has no `.claude/skills` directory at all, so
        // a real discovery would find nothing; the skill below only appears in the rendered prompt
        // because `build_static_system_prompt` trusts the list it was given instead.
        let tmp = tempfile::tempdir().unwrap();
        let skill = Skill {
            name: "already-discovered".into(),
            description: "found by the caller, not by this function".into(),
            path: tmp.path().join("SKILL.md"),
            disable_model_invocation: false,
            scope: "user",
        };
        let prompt = build_system_prompt(&PromptOptions {
            base: Some("DEFAULT IDENTITY"),
            default_base: "",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: std::slice::from_ref(&skill),
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: true,
            agents: &[],
        });
        assert!(
            prompt.contains("already-discovered") && prompt.contains("found by the caller"),
            "got: {prompt}"
        );
    }

    #[test]
    fn all_hidden_skills_produce_no_available_skills_wrapper_at_all() {
        // pi-parity fix (M1): the call site here used to gate on `!opts.skills.is_empty()` — the
        // *unfiltered* list — so a non-empty list where every skill is `disable-model-invocation` still
        // produced an empty `<available_skills>\n…\n</available_skills>` shell with no actual entries.
        // `skills::format_available` itself now returns `""` in that case; this pins the fix at the
        // level a caller actually observes it: the assembled system prompt.
        let tmp = tempfile::tempdir().unwrap();
        let skill = Skill {
            name: "hidden".into(),
            description: "Explicit only".into(),
            path: tmp.path().join("SKILL.md"),
            disable_model_invocation: true,
            scope: "user",
        };
        let prompt = build_system_prompt(&PromptOptions {
            base: Some("DEFAULT IDENTITY"),
            default_base: "",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: std::slice::from_ref(&skill),
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: true,
            agents: &[],
        });
        assert!(
            !prompt.contains("available_skills"),
            "no wrapper at all when nothing is model-visible: {prompt}"
        );
    }

    #[test]
    fn skills_are_not_advertised_at_all_when_the_read_tool_is_not_registered() {
        // pi-parity fix: a skill is discovered by path, not inlined into the prompt — invoking one
        // relies on the model being able to open its `SKILL.md` itself. Advertising
        // `<available_skills>` to a model with no `read` tool at all just adds dead weight it can never
        // act on.
        let tmp = tempfile::tempdir().unwrap();
        let skill = Skill {
            name: "visible-but-unusable".into(),
            description: "would be model-visible if read were registered".into(),
            path: tmp.path().join("SKILL.md"),
            disable_model_invocation: false,
            scope: "user",
        };
        let prompt = build_system_prompt(&PromptOptions {
            base: Some("DEFAULT IDENTITY"),
            default_base: "",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: std::slice::from_ref(&skill),
            has_read: false,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: true,
            agents: &[],
        });
        assert!(
            !prompt.contains("available_skills") && !prompt.contains("visible-but-unusable"),
            "no skills section at all without the read tool: {prompt}"
        );
    }

    #[test]
    fn append_system_md_is_appended_when_trusted_and_no_explicit_override() {
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("APPEND_SYSTEM.md"), "EXTRA HOUSE RULES").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: Some("DEFAULT IDENTITY"),
            default_base: "",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: true,
            agents: &[],
        });
        assert!(prompt.contains("DEFAULT IDENTITY"));
        assert!(
            prompt.contains("EXTRA HOUSE RULES"),
            "a trusted project's on-disk APPEND_SYSTEM.md must be appended to the system prompt"
        );
    }

    #[test]
    fn append_system_md_is_ignored_when_project_is_untrusted() {
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("APPEND_SYSTEM.md"), "MALICIOUS EXTRA RULES").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: Some("DEFAULT IDENTITY"),
            default_base: "",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: false,
            agents: &[],
        });
        assert!(!prompt.contains("MALICIOUS EXTRA RULES"));
    }

    #[test]
    fn explicit_append_wins_outright_over_on_disk_append_system_md() {
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("APPEND_SYSTEM.md"), "ON-DISK APPEND").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: Some("DEFAULT IDENTITY"),
            default_base: "",
            append: Some("CLI APPEND"),
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: true,
            agents: &[],
        });
        assert!(prompt.contains("CLI APPEND"));
        assert!(
            !prompt.contains("ON-DISK APPEND"),
            "an explicit --append-system-prompt must win outright, not combine with the on-disk file"
        );
    }

    #[test]
    fn tzif_offset_reads_fixed_offset_zone() {
        // A minimal v1 TZif blob with one type (gmtoff = +3600s) and no transitions: the parser must
        // fall back to that type and return +3600 regardless of `now`.
        let mut blob: Vec<u8> = Vec::new();
        blob.extend_from_slice(b"TZif"); // magic
        blob.push(0); // version '\0' → 32-bit block only
        blob.extend_from_slice(&[0u8; 15]); // reserved
        // counts: isut, isstd, leap, time, type, char
        for c in [0u32, 0, 0, 0, 1, 1] {
            blob.extend_from_slice(&c.to_be_bytes());
        }
        // ttinfo[0]: gmtoff=3600, isdst=0, abbrind=0
        blob.extend_from_slice(&3600i32.to_be_bytes());
        blob.push(0);
        blob.push(0);
        blob.push(0); // abbreviation chars (charcnt = 1)

        assert_eq!(tzif_offset(&blob, 1_700_000_000), Some(3600));
    }

    #[test]
    fn tz_string_offset_resolves_a_real_iana_zone() {
        // UTC's zoneinfo file exists on essentially every Linux distro (including this test machine's)
        // and has a fixed 0 offset — a reliable resolvable-zone case without hardcoding a path.
        assert_eq!(tz_string_offset("UTC", 1_700_000_000), Some(0));
        // POSIX `:`-prefix convention (explicitly "this is a file reference") resolves the same way.
        assert_eq!(tz_string_offset(":UTC", 1_700_000_000), Some(0));
    }

    #[test]
    fn tz_string_offset_empty_value_is_utc() {
        // POSIX: an empty `TZ` value means UTC.
        assert_eq!(tz_string_offset("", 1_700_000_000), Some(0));
        assert_eq!(tz_string_offset(":", 1_700_000_000), Some(0));
    }

    #[test]
    fn tz_string_offset_falls_through_on_unresolvable_value() {
        // A raw POSIX offset-rule string isn't parsed (no rule-transition logic here) — falls through
        // to `None` so the caller tries `/etc/localtime` next, rather than erroring the date entirely.
        assert_eq!(
            tz_string_offset("EST5EDT,M3.2.0,M11.1.0", 1_700_000_000),
            None
        );
        assert_eq!(tz_string_offset("Not/A/Real/Zone", 1_700_000_000), None);
    }

    /// Options whose only interesting axes are the two tool-gated prompt sections.
    fn section_opts(cwd: &Path, has_todo: bool, has_structured_output: bool) -> PromptOptions<'_> {
        PromptOptions {
            base: Some("You are an agent."),
            default_base: "",
            append: None,
            cwd,
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo,
            has_structured_output,
            has_memory: false,
            memory_sections: &[],
            project_trusted: false,
            agents: &[],
        }
    }

    #[test]
    fn the_todo_protocol_is_advertised_only_when_the_tool_is_registered() {
        // Same gate-on-what's-registered discipline as `<available_skills>`/`<available_agents>`: a
        // process (or subagent child) whose registry never advertised `todo` must not carry the
        // protocol's dead weight in its prompt.
        let tmp = tempfile::tempdir().unwrap();
        let with = build_static_system_prompt(&section_opts(tmp.path(), true, false));
        assert!(with.contains("<todo_protocol>"));
        assert!(with.contains("fully replaces the previous list"));
        assert!(with.contains("exactly one item `in_progress`"));

        let without = build_static_system_prompt(&section_opts(tmp.path(), false, false));
        assert!(!without.contains("todo_protocol"));
        assert!(!without.contains("in_progress"));
    }

    #[test]
    fn the_structured_output_protocol_is_advertised_only_when_the_tool_is_registered() {
        // This section *is* the forcing mechanism — there is no `tool_choice` pinning — so its presence
        // when the tool is registered is a correctness property, not a nicety.
        let tmp = tempfile::tempdir().unwrap();
        let with = build_static_system_prompt(&section_opts(tmp.path(), false, true));
        assert!(with.contains("<structured_output_protocol>"));
        assert!(with.contains("exactly once"));
        assert!(with.contains("Do not describe the result in prose"));

        let without = build_static_system_prompt(&section_opts(tmp.path(), false, false));
        assert!(!without.contains("structured_output"));
    }

    #[test]
    fn the_memory_section_and_index_are_injected_only_when_the_tool_is_registered() {
        let tmp = tempfile::tempdir().unwrap();
        let sections = [(
            crate::memory::MountKind::Durable,
            "- [notes](notes.md) — build & test".to_string(),
        )];
        let mut opts = section_opts(tmp.path(), false, false);
        opts.has_memory = true;
        opts.memory_sections = &sections;
        let with = build_static_system_prompt(&opts);
        assert!(with.contains("## Memory"), "the guidance block must appear");
        assert!(
            with.contains("[notes](notes.md)"),
            "the index must be injected verbatim"
        );

        // No tool → no section, even if an index string is present.
        let mut off = section_opts(tmp.path(), false, false);
        off.memory_sections = &sections;
        let without = build_static_system_prompt(&off);
        assert!(!without.contains("## Memory"));
        assert!(!without.contains("[notes](notes.md)"));
    }

    #[test]
    fn both_memory_mounts_render_with_distinct_roots_when_a_session_is_active() {
        let tmp = tempfile::tempdir().unwrap();
        let sections = [
            (
                crate::memory::MountKind::Durable,
                "- [proj](proj.md) — durable".to_string(),
            ),
            (
                crate::memory::MountKind::Session,
                "- [scratch](scratch.md) — this task".to_string(),
            ),
        ];
        let mut opts = section_opts(tmp.path(), false, false);
        opts.has_memory = true;
        opts.memory_sections = &sections;
        let out = build_static_system_prompt(&opts);
        assert!(out.contains("Memory (durable, cross-session)"));
        assert!(out.contains("Working memory (this session)"));
        assert!(out.contains("/memories/MEMORY.md"));
        assert!(out.contains("/session/MEMORY.md"));
        assert!(out.contains("[proj](proj.md)") && out.contains("[scratch](scratch.md)"));
    }

    #[test]
    fn system_prompt_includes_project_instructions_and_env() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("CLAUDE.md"), "Be excellent.").unwrap();
        let prompt = build_system_prompt(&PromptOptions {
            base: Some("You are an agent."),
            default_base: "",
            append: Some("Stay terse."),
            cwd: tmp.path(),
            include_context_files: true,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: false,
            agents: &[],
        });
        assert!(prompt.contains("You are an agent."));
        assert!(prompt.contains("Stay terse."));
        assert!(prompt.contains("<project_context>"));
        assert!(prompt.contains("</project_context>"));
        assert!(prompt.contains("<project_instructions"));
        assert!(prompt.contains("Be excellent."));
        assert!(prompt.contains("Current date:"));
        assert!(prompt.contains("Current working directory:"));
        // The instructions block must be nested *inside* the wrapper, not just present somewhere.
        let wrapper_start = prompt.find("<project_context>").unwrap();
        let wrapper_end = prompt.find("</project_context>").unwrap();
        let instructions_pos = prompt.find("<project_instructions").unwrap();
        assert!(instructions_pos > wrapper_start && instructions_pos < wrapper_end);
    }

    #[test]
    fn static_prompt_plus_dynamic_footer_equals_the_full_prompt() {
        let tmp = tempfile::tempdir().unwrap();
        let opts = PromptOptions {
            base: Some("You are an agent."),
            default_base: "",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: false,
            agents: &[],
        };
        let full = build_system_prompt(&opts);
        let static_part = build_static_system_prompt(&opts);
        let footer = dynamic_footer(tmp.path());
        assert_eq!(full, format!("{static_part}{footer}"));
        // The static half must carry no date/cwd at all — that's the whole point of the split.
        assert!(!static_part.contains("Current date:"));
        assert!(!static_part.contains("Current working directory:"));
        assert!(footer.contains("Current date:"));
        assert!(footer.contains("Current working directory:"));
    }

    #[test]
    fn project_context_wrapper_is_absent_when_context_files_are_disabled() {
        // `include_context_files: false` skips `load_context_files` entirely — the one way to prove
        // the wrapper is conditional without depending on `$HOME/.claude` being empty (which this test
        // can't safely control: mutating `HOME` via `std::env::set_var` is unsafe to do from a test
        // that may run in parallel with others, and this repo's own `~/.claude/CLAUDE.md` genuinely
        // exists on a real dev machine, so an empty-tempdir-cwd variant of this test would be flaky).
        let tmp = tempfile::tempdir().unwrap();
        let prompt = build_system_prompt(&PromptOptions {
            base: Some("You are an agent."),
            default_base: "",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            skills: &[],
            has_read: true,
            has_todo: false,
            has_structured_output: false,
            has_memory: false,
            memory_sections: &[],
            project_trusted: false,
            agents: &[],
        });
        assert!(!prompt.contains("<project_context>"));
    }
}
