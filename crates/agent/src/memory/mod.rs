//! Persistent, cross-session agent memory — a durable knowledge store the model curates itself.
//!
//! # Why this is its own thing, not the filesystem tools
//!
//! `read`/`write`/`edit`/`ls`/`grep` operate on the **working tree**: real filesystem paths, rooted at
//! the process cwd, with staleness checks, image/binary handling, and git awareness. They exist to edit
//! *code*. Memory is a different subject with different needs:
//!
//! - It addresses a **logical namespace** (`/memories/foo.md`), not a filesystem path — so the model
//!   never confuses a note-to-self with the checkout it's editing.
//! - Its backend is **pluggable**: local `*.md` files today (see [`file::FileBackend`]), a networked
//!   `redis://` / `postgres://` store later. Nothing above [`MemoryBackend`] may assume a filesystem —
//!   pointing `read`/`write` at a memory dir would weld memory to local disk forever.
//! - It carries an **index** ([`MemoryBackend::index`], the `MEMORY.md` file) that the host injects into
//!   the system prompt at session start, so a durable memory is never silently forgotten.
//!
//! What we *do* reuse is deliberate: the tool's command verbs (`view`≈read/ls, `create`≈write,
//! `str_replace`≈edit, `insert`) mirror Anthropic's canonical `memory_20250818` tool surface — which the
//! model is already trained on — so its existing skills transfer. See [`crate::tools::memory`].
//!
//! # Scope
//!
//! Per-project, keyed by cwd, under `~/.claude/projects/<encoded-cwd>/memory/` — the same scoping
//! sessions use ([`crate::session_store::encode_cwd`]), so all worktrees of one repo share one memory.
//! A backend is selected by [`open`], which dispatches on a DSN scheme (bare path / `file://` → files;
//! `redis://` / `postgres://` are recognized but not yet implemented, so the seam exists without the
//! impl).

pub mod file;

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

/// The durable, cross-session store's logical root, matching Anthropic's `memory_20250818` tool. A
/// `view` of exactly this path lists the store; everything else is a document beneath it.
pub const MEMORY_ROOT: &str = "/memories";

/// The per-session **working memory** root — a private scratchpad for the current task that survives
/// the agent's own compaction and is cleaned up when the session ends. Mounted alongside
/// [`MEMORY_ROOT`] as a distinct namespace so the model never conflates durable knowledge with
/// task-local scratch. See [`MountKind`].
pub const SESSION_ROOT: &str = "/session";

/// The index document. Its contents are what [`MemoryBackend::index`] returns for system-prompt
/// injection; the model curates it as an ordinary document (a one-line pointer per memory).
pub const INDEX_FILE: &str = "MEMORY.md";

/// Cap on the injected index, mirroring Claude Code's auto-memory (first ~200 lines / ~25 KB of
/// `MEMORY.md`). Bounds the always-present prefix so a sprawling index can't blow the context window or
/// the prompt cache.
pub const INDEX_MAX_LINES: usize = 200;
/// Byte cap companion to [`INDEX_MAX_LINES`] — whichever bites first wins.
pub const INDEX_MAX_BYTES: usize = 25 * 1024;

/// A failure inside a [`MemoryBackend`]. The tool maps the model-correctable variants to
/// [`agent_core::ToolError::InvalidInput`] (the model can fix its own call) and [`MemoryError::Backend`]
/// to [`agent_core::ToolError::Execution`] (a store/IO failure it can't).
#[derive(Debug, Error)]
pub enum MemoryError {
    /// The path names nothing in the store.
    #[error("no memory at `{0}`")]
    NotFound(String),
    /// A create/rename target is already occupied.
    #[error("`{0}` already exists; edit it with `str_replace`/`insert`, or `delete` it first")]
    AlreadyExists(String),
    /// A `str_replace` whose `old_str` matched a number of times other than exactly one — the model must
    /// disambiguate. `count == 0` is "not found in the document"; `> 1` is "ambiguous".
    #[error("`{old}` matched {count} times in `{path}` (need exactly 1)")]
    NotUnique {
        path: String,
        old: String,
        count: usize,
    },
    /// The logical path was malformed (outside `/memories`, contained `..`, etc.).
    #[error("invalid memory path: {0}")]
    InvalidPath(String),
    /// The backend itself failed (disk IO, a network store, a lock timeout).
    #[error("memory backend error: {0}")]
    Backend(String),
}

/// One entry in a directory [`View::Listing`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The entry's full logical path (`/memories/...`).
    pub path: String,
    /// Whether it's a sub-directory (a document otherwise).
    pub is_dir: bool,
    /// Size in bytes (0 for a directory).
    pub size: u64,
}

/// A single `search` match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// Full logical path of the document the match is in.
    pub path: String,
    /// 1-indexed line number.
    pub line: usize,
    /// The matching line's text (trimmed of the trailing newline).
    pub text: String,
}

/// What a [`MemoryBackend::view`] returned: a document's contents or a directory listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum View {
    /// A document's text (possibly a `view_range` slice of it).
    Document(String),
    /// A directory's entries, sorted by path.
    Listing(Vec<Entry>),
}

/// A validated logical path within the store — the contract every backend addresses, independent of how
/// that backend actually stores things. Guaranteed to sit under [`MEMORY_ROOT`] with no `.`/`..`
/// traversal, so a file backend can join it onto its real directory and a key/value backend can use it
/// as a key, both safely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemPath {
    /// Which mounted root this path lives under (e.g. [`MEMORY_ROOT`] or [`SESSION_ROOT`]). Carried so
    /// error messages and listings display the right namespace; a backend still addresses only [`rel`].
    root: &'static str,
    /// Clean relative path beneath the root; `""` is the root itself. Never has a leading/trailing `/`,
    /// never an empty/`.`/`..` component.
    rel: String,
}

impl MemPath {
    /// Parse under the default durable root ([`MEMORY_ROOT`]) — the single-root shorthand.
    pub fn parse(raw: &str) -> Result<Self, MemoryError> {
        Self::parse_in(raw, MEMORY_ROOT)
    }

    /// Parse and validate a model-supplied path **under a specific `root`**. Accepts `<root>`,
    /// `<root>/<rel>`, `<root-without-leading-slash>/<rel>`, or a bare relative `<rel>` (treated as
    /// under `root`); rejects any other absolute path and any `.`/`..`/empty component.
    pub fn parse_in(raw: &str, root: &'static str) -> Result<Self, MemoryError> {
        let trimmed = raw.trim();
        let bad = || MemoryError::InvalidPath(raw.to_string());
        let bare = root.trim_start_matches('/'); // e.g. "memories" / "session"

        // Peel the logical root off the front, however the model spelled it.
        let remainder = if trimmed == root || trimmed == bare {
            ""
        } else if let Some(r) = trimmed.strip_prefix(&format!("{root}/")) {
            r
        } else if let Some(r) = trimmed.strip_prefix(&format!("{bare}/")) {
            r
        } else if trimmed.starts_with('/') {
            // An absolute path that isn't under `root` — never allowed to escape the store.
            return Err(bad());
        } else if trimmed.is_empty() {
            return Err(bad());
        } else {
            // A bare relative path — treat it as living under the root.
            trimmed
        };

        // Validate and re-clean each component so the joined path can never traverse out.
        let mut clean: Vec<&str> = Vec::new();
        for comp in remainder.split('/') {
            match comp {
                "" | "." | ".." => {
                    // Empty (`a//b`, leading/trailing slash), current-dir, or parent-dir — all rejected
                    // outright rather than normalized, so nothing can climb out of the store.
                    if comp.is_empty() && clean.is_empty() && remainder.is_empty() {
                        // remainder == "" already handled as root above; this guards the split's single
                        // empty element for an empty remainder — unreachable in practice, kept explicit.
                        continue;
                    }
                    return Err(bad());
                }
                _ if comp.contains('\0') => return Err(bad()),
                _ => clean.push(comp),
            }
        }
        Ok(Self {
            root,
            rel: clean.join("/"),
        })
    }

    /// Classify a model-supplied path across several mounted `roots`: pick whichever root the path is
    /// spelled under and parse beneath it. A bare relative path (no known root prefix) defaults to the
    /// **first** root; any other absolute path (`/etc/...`, an unknown root) is rejected with a message
    /// listing the valid roots.
    pub fn classify(raw: &str, roots: &[&'static str]) -> Result<Self, MemoryError> {
        let trimmed = raw.trim();
        for &root in roots {
            let bare = root.trim_start_matches('/');
            if trimmed == root
                || trimmed == bare
                || trimmed.starts_with(&format!("{root}/"))
                || trimmed.starts_with(&format!("{bare}/"))
            {
                return Self::parse_in(trimmed, root);
            }
        }
        if trimmed.starts_with('/') || trimmed.is_empty() {
            return Err(MemoryError::InvalidPath(format!(
                "`{raw}` — memory paths must be under one of: {}",
                roots.join(", ")
            )));
        }
        // Bare relative path → the default (first) root.
        Self::parse_in(trimmed, roots.first().copied().unwrap_or(MEMORY_ROOT))
    }

    /// The mounted root this path lives under.
    pub fn root(&self) -> &'static str {
        self.root
    }

    /// Whether this points at the store root (a `view` of it lists the store).
    pub fn is_root(&self) -> bool {
        self.rel.is_empty()
    }

    /// The clean relative path beneath the root (`""` for the root). A file backend joins this onto its
    /// directory; a key/value backend uses it as a key.
    pub fn rel(&self) -> &str {
        &self.rel
    }

    /// The full logical path (`<root>/...`) for display back to the model.
    pub fn display(&self) -> String {
        if self.rel.is_empty() {
            self.root.to_string()
        } else {
            format!("{}/{}", self.root, self.rel)
        }
    }
}

/// A pluggable backing store for agent memory. Implementors address a logical `/memories` namespace via
/// [`MemPath`]; the concrete store (files, redis, postgres) is an implementation detail. Async because a
/// networked backend needs it; the file backend does blocking IO inline (matching the other filesystem
/// tools in this crate).
#[async_trait]
pub trait MemoryBackend: Send + Sync {
    /// The index document ([`INDEX_FILE`]) rendered for system-prompt injection, already bounded to
    /// [`INDEX_MAX_LINES`] / [`INDEX_MAX_BYTES`]. Empty string when there is no index yet.
    async fn index(&self) -> Result<String, MemoryError>;

    /// View a document (its text, optionally a 1-indexed inclusive `range` of lines) or, for the root or
    /// a sub-directory, a listing of what's inside.
    async fn view(
        &self,
        path: &MemPath,
        range: Option<(usize, usize)>,
    ) -> Result<View, MemoryError>;

    /// Create or overwrite a document with `text` (matching `memory_20250818`'s `create`, which
    /// overwrites — the model edits with `str_replace`/`insert` and replaces wholesale with `create`).
    async fn create(&self, path: &MemPath, text: &str) -> Result<(), MemoryError>;

    /// Replace the single occurrence of `old` with `new` in a document. Errors ([`MemoryError::NotUnique`])
    /// unless `old` matched exactly once.
    async fn str_replace(&self, path: &MemPath, old: &str, new: &str) -> Result<(), MemoryError>;

    /// Insert `text` after 1-indexed line `line` (`0` = the start of the document).
    async fn insert(&self, path: &MemPath, line: usize, text: &str) -> Result<(), MemoryError>;

    /// Delete a document, or an empty directory. A non-empty directory is refused.
    async fn delete(&self, path: &MemPath) -> Result<(), MemoryError>;

    /// Move `from` to `to`. Refused if `to` is already occupied.
    async fn rename(&self, from: &MemPath, to: &MemPath) -> Result<(), MemoryError>;

    /// Case-insensitive substring search across every document, newest-relevant first. A file backend
    /// scans; a SQL backend can push this down to full-text search.
    async fn search(&self, query: &str) -> Result<Vec<Hit>, MemoryError>;
}

/// Open the memory backend named by `dsn` for the project rooted at `cwd`.
///
/// - `None` (the default) → a per-project [`file::FileBackend`] under `~/.claude/projects/<cwd>/memory/`.
/// - a bare path or `file://<path>` → a [`file::FileBackend`] at that directory.
/// - `redis://…` / `postgres://…` → recognized, but **not yet implemented**: returns a clear error so
///   the seam is real without the impl (the trait makes them a drop-in later).
///
/// `Err` is meant to be printed to the operator who passed `--memory`.
pub fn open(dsn: Option<&str>, cwd: &Path) -> Result<Arc<dyn MemoryBackend>, String> {
    match dsn.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(Arc::new(file::FileBackend::for_project(cwd))),
        Some(spec) => {
            if let Some(path) = spec.strip_prefix("file://") {
                Ok(Arc::new(file::FileBackend::at(std::path::PathBuf::from(
                    path,
                ))))
            } else if spec.starts_with("redis://") || spec.starts_with("rediss://") {
                Err(
                    "redis:// memory backend is not yet supported (only local files today)"
                        .to_string(),
                )
            } else if spec.starts_with("postgres://") || spec.starts_with("postgresql://") {
                Err(
                    "postgres:// memory backend is not yet supported (only local files today)"
                        .to_string(),
                )
            } else if spec.contains("://") {
                let scheme = spec.split("://").next().unwrap_or(spec);
                Err(format!(
                    "unsupported memory backend `{scheme}://` (supported: a local path, file://, \
                     and — soon — redis://, postgres://)"
                ))
            } else {
                // A bare filesystem path.
                Ok(Arc::new(file::FileBackend::at(std::path::PathBuf::from(
                    spec,
                ))))
            }
        }
    }
}

/// A single mounted store: a [`MemoryBackend`] surfaced at the root its [`MountKind`] dictates. The
/// `memory` tool routes each path to the mount whose root it names; hosts build the list (durable, plus
/// a session mount when a session is active) and share it — by `Arc` clone of the backend — with
/// subagents so a child sees the very same `/session` scratch as its parent (it is *the same task*).
#[derive(Clone)]
pub struct Mount {
    pub kind: MountKind,
    pub backend: Arc<dyn MemoryBackend>,
}

impl Mount {
    /// The logical root this mount is addressed under.
    pub fn root(&self) -> &'static str {
        self.kind.root()
    }
}

/// Read each mount's current, already-bounded index, paired with its [`MountKind`], for system-prompt
/// injection (see [`render_sections`]). A backend read error degrades to an empty index rather than
/// failing the prompt build — a memory store hiccup must not take down the session.
pub async fn mount_sections(mounts: &[Mount]) -> Vec<(MountKind, String)> {
    let mut out = Vec::with_capacity(mounts.len());
    for m in mounts {
        let index = m.backend.index().await.unwrap_or_default();
        out.push((m.kind, index));
    }
    out
}

/// The lifecycle a mounted store has, which decides its root, its guidance, and (for the host) where it
/// lives on disk. The `memory` tool can mount one of each — durable knowledge and session-local working
/// memory — at distinct roots at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountKind {
    /// Durable, cross-session knowledge at [`MEMORY_ROOT`] (`~/.claude/projects/<cwd>/memory/`).
    Durable,
    /// Per-session working memory at [`SESSION_ROOT`], cleaned up with the session — the scratchpad that
    /// survives compaction.
    Session,
}

impl MountKind {
    /// The logical root this kind mounts at.
    pub fn root(self) -> &'static str {
        match self {
            MountKind::Durable => MEMORY_ROOT,
            MountKind::Session => SESSION_ROOT,
        }
    }

    /// The system-prompt guidance for this kind (heading + how/when to use it).
    fn guidance(self) -> &'static str {
        match self {
            MountKind::Durable => MEMORY_GUIDANCE,
            MountKind::Session => SESSION_GUIDANCE,
        }
    }
}

/// The `## Memory` guidance for the durable, cross-session store. Tells the model the store exists, how
/// to reach it, and the curation discipline that keeps it useful instead of a junk drawer.
pub const MEMORY_GUIDANCE: &str = "\
## Memory (durable, cross-session)

You have a persistent, cross-session memory store at `/memories`, reached through the `memory` tool \
(commands: `view`, `create`, `str_replace`, `insert`, `delete`, `rename`, `search`). It survives across \
sessions on this project — use it for durable facts that would help *future* sessions: build and test \
commands, non-obvious architecture decisions, debugging lessons, the user's stated preferences.

- `MEMORY.md` under `/memories` is the index. Keep it a lean list of one-line pointers to topic files — \
do not put memory content in it. Store one fact per topic file, then add a pointer in `MEMORY.md`.
- `create` makes a *new* file and fails if one already exists; edit an existing memory with \
`str_replace`/`insert`. Update rather than duplicate; delete a memory once you learn it is wrong.
- Save only what is genuinely useful to a future session and not already obvious from the repo, its \
docs, or version history. Convert relative dates to absolute ones.";

/// Guidance for the per-session working-memory store — the scratchpad that survives compaction. Framed
/// deliberately in contrast to [`MEMORY_GUIDANCE`]: precise, task-local, ephemeral.
pub const SESSION_GUIDANCE: &str = "\
## Working memory (this session)

You also have a private working-memory store at `/session`, reached through the same `memory` tool by \
using a `/session/...` path. It is scoped to *this task* and is cleared when the session ends — it is \
**not** durable project knowledge (that goes in `/memories`).

Its purpose is to survive your own context compaction. When your conversation grows long it is \
compacted into a lossy summary that drops specifics; anything you saved to `/session` stays intact. So \
as you work, record the exact facts you would not want summarized away — precise values, file paths and \
line numbers, command outputs, decisions and their rationale, your current plan and progress. After a \
compaction, read them back with `memory` `view /session` before continuing. Keep `/session/MEMORY.md` \
as a lean index of pointers, same as `/memories`.";

/// Render the injected `Memory` sections for the given mounts, each as its guidance followed by its
/// current (already-bounded) index. `entries` pairs a [`MountKind`] with what its backend's
/// [`MemoryBackend::index`] returned. Used by [`crate::resources`] when assembling the system prompt.
pub fn render_sections(entries: &[(MountKind, String)]) -> String {
    entries
        .iter()
        .map(|(kind, index)| {
            let guidance = kind.guidance();
            let index = index.trim();
            let root = kind.root();
            if index.is_empty() {
                format!("{guidance}\n\nYour `{root}/MEMORY.md` index is currently empty.")
            } else {
                format!("{guidance}\n\nCurrent `{root}/MEMORY.md` index:\n\n{index}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The active nudge pushed as a mid-run steer on each compaction, when a `/session` working-memory mount
/// is present. Compaction rewrites the message history into a lossy summary but leaves the system prompt
/// (and its `/session` guidance) intact; this reminder rides the next post-compaction request so the model
/// actively reads back the specifics it saved before continuing. See [`SESSION_GUIDANCE`].
pub const COMPACTION_REMINDER: &str = "<system-reminder>Your earlier context was just compacted to a \
    summary and may have lost specifics. Before continuing, consult your working memory (the `memory` \
    tool, `view /session`) to recover exact details — values, paths, decisions — you saved there.\
    </system-reminder>";

/// The pre-compaction nudge, pushed as a mid-run steer when the live prompt first crosses
/// [`compaction_pressure_point`] and a `/session` working-memory mount is present. The pre-emptive
/// complement to [`COMPACTION_REMINDER`]: rather than helping the model recover *after* the cut, it warns
/// it *before*, while it still has full detail, to checkpoint what it will want to keep. Fired at most
/// once per fill cycle (re-armed on each `Compacted`).
pub const PRESSURE_NUDGE: &str = "<system-reminder>Your context is getting large and will soon be \
    compacted into a summary that drops specifics. While you still have the full detail, save anything \
    you'll need afterward to your working memory now — use the `memory` tool to record it under \
    `/session`: exact values, file paths and line numbers, key decisions and their rationale, and your \
    current plan and progress. Then carry on.</system-reminder>";

/// The live-prompt size at which to warn of an approaching compaction: 80% of the compaction threshold
/// (`context_window - reserve_tokens`, the point [`agent_core::compaction::should_compact`] fires at).
/// This leaves runway — typically a few turns — for the model to checkpoint to `/session` before the cut.
/// Integer math (÷5 ×4) so there is no float cast and no overflow even at a 1M-token window.
pub fn compaction_pressure_point(context_window: u32, reserve_tokens: u32) -> u32 {
    context_window.saturating_sub(reserve_tokens) / 5 * 4
}

/// The full prompt size a [`agent_core::message::TokenUsage`] report implies — uncached input plus
/// everything served from or written to the prompt cache. Matches what `Session::record_usage` folds
/// into `last_input_tokens` (the figure the compaction trigger compares), so a host-side pressure check
/// measures the same thing the trigger does.
pub fn live_prompt_tokens(usage: &agent_core::message::TokenUsage) -> u32 {
    usage
        .input_tokens
        .saturating_add(usage.cache_read_tokens)
        .saturating_add(usage.cache_write_tokens)
}

/// Resolve the on-disk directory for a session's working memory.
///
/// - With a persisted session file, a sibling of the `<created_at>_<id>.jsonl` — `<...>.memory/` — in
///   the same repo directory, so it is trashed/restored alongside the session (see
///   [`crate::session_store`]).
/// - Without one (`--no-session-persistence`), an ephemeral tempdir keyed by the session id, reaped at
///   teardown.
pub fn session_dir(session_file: Option<&Path>, session_id: &str) -> std::path::PathBuf {
    match session_file {
        Some(file) => file.with_extension("memory"),
        None => std::env::temp_dir()
            .join("beyond-ai-session-memory")
            .join(session_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_the_canonical_and_lenient_spellings() {
        assert!(MemPath::parse("/memories").unwrap().is_root());
        assert!(MemPath::parse("/memories/").unwrap().is_root());
        assert!(MemPath::parse("memories").unwrap().is_root());
        assert_eq!(MemPath::parse("/memories/a.md").unwrap().rel(), "a.md");
        assert_eq!(MemPath::parse("memories/a.md").unwrap().rel(), "a.md");
        assert_eq!(MemPath::parse("a.md").unwrap().rel(), "a.md");
        assert_eq!(
            MemPath::parse("/memories/sub/b.md").unwrap().rel(),
            "sub/b.md"
        );
    }

    #[test]
    fn parse_rejects_traversal_and_foreign_absolutes() {
        for bad in [
            "/memories/../etc/passwd",
            "/memories/a/../../b",
            "../secrets",
            "/etc/passwd",
            "memories/./x",
            "/memories/a//b",
        ] {
            assert!(
                MemPath::parse(bad).is_err(),
                "{bad} must be rejected as an invalid path"
            );
        }
    }

    #[test]
    fn display_round_trips_the_root_and_children() {
        assert_eq!(MemPath::parse("/memories").unwrap().display(), "/memories");
        assert_eq!(
            MemPath::parse("/memories/x.md").unwrap().display(),
            "/memories/x.md"
        );
    }

    #[test]
    fn open_recognizes_but_defers_network_backends() {
        let cwd = std::path::Path::new("/tmp");
        assert!(open(Some("redis://localhost"), cwd).is_err());
        assert!(open(Some("postgres://localhost/db"), cwd).is_err());
        assert!(open(Some("mysql://x"), cwd).is_err());
        // A bare path and file:// both resolve to a file backend.
        assert!(open(Some("/tmp/mem"), cwd).is_ok());
        assert!(open(Some("file:///tmp/mem"), cwd).is_ok());
        assert!(open(None, cwd).is_ok());
    }

    #[test]
    fn render_sections_handles_empty_populated_and_multi_mount() {
        // A single empty durable mount.
        let empty = render_sections(&[(MountKind::Durable, String::new())]);
        assert!(empty.contains("currently empty"));
        assert!(empty.contains("/memories/MEMORY.md"));

        // A populated durable mount.
        let s = render_sections(&[(MountKind::Durable, "- [x](x.md) — a thing".to_string())]);
        assert!(s.contains("Current `/memories/MEMORY.md` index"));
        assert!(s.contains("[x](x.md)"));

        // Both mounts render with distinct roots and distinct guidance.
        let both = render_sections(&[
            (MountKind::Durable, "- durable".to_string()),
            (MountKind::Session, "- scratch".to_string()),
        ]);
        assert!(both.contains("Memory (durable, cross-session)"));
        assert!(both.contains("Working memory (this session)"));
        assert!(both.contains("Current `/memories/MEMORY.md` index"));
        assert!(both.contains("Current `/session/MEMORY.md` index"));
    }

    #[test]
    fn classify_routes_across_roots_and_rejects_the_unknown() {
        let roots = [MEMORY_ROOT, SESSION_ROOT];

        // Each spelled root routes to its own mount, rel intact.
        let d = MemPath::classify("/memories/a.md", &roots).unwrap();
        assert_eq!(d.root(), MEMORY_ROOT);
        assert_eq!(d.rel(), "a.md");
        let s = MemPath::classify("/session/notes/b.md", &roots).unwrap();
        assert_eq!(s.root(), SESSION_ROOT);
        assert_eq!(s.rel(), "notes/b.md");
        // Bare (no-slash) root spelling routes too.
        assert_eq!(
            MemPath::classify("session/c.md", &roots).unwrap().root(),
            SESSION_ROOT
        );
        // A bare relative path defaults to the first root.
        assert_eq!(
            MemPath::classify("loose.md", &roots).unwrap().root(),
            MEMORY_ROOT
        );
        // An unknown absolute root is rejected, and the message names the valid roots.
        let err = MemPath::classify("/etc/passwd", &roots).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("/memories") && msg.contains("/session"));
        // Traversal is still rejected even under a known root.
        assert!(MemPath::classify("/session/../etc", &roots).is_err());
    }

    #[test]
    fn pressure_point_is_below_the_compaction_threshold_with_runway() {
        // threshold = 200_000 - 16_384 = 183_616; 80% = 146_892 (÷5×4 rounds the 183_616/5).
        let p = compaction_pressure_point(200_000, 16_384);
        assert!(p < 200_000 - 16_384, "must fire before the compaction cut");
        assert!(p > (200_000 - 16_384) / 2, "but not absurdly early");
        assert_eq!(p, 183_616 / 5 * 4);
        // Degenerate windows don't panic or overflow.
        assert_eq!(compaction_pressure_point(0, 16_384), 0);
        assert_eq!(compaction_pressure_point(100, 200), 0);
        assert_eq!(compaction_pressure_point(u32::MAX, 0), u32::MAX / 5 * 4);
    }

    #[test]
    fn live_prompt_tokens_sums_uncached_and_cached_like_the_trigger() {
        let usage = agent_core::message::TokenUsage {
            input_tokens: 1000,
            cache_read_tokens: 500,
            cache_write_tokens: 200,
            output_tokens: 9999, // must NOT be counted — it isn't part of the prompt
            ..Default::default()
        };
        assert_eq!(live_prompt_tokens(&usage), 1700);
    }

    #[test]
    fn session_dir_is_a_sibling_when_persisted_else_a_tempdir() {
        // Persisted: a `.memory` sibling of the session `.jsonl`.
        let file = std::path::Path::new("/home/u/.claude/sessions/enc/1700_abc.jsonl");
        let dir = session_dir(Some(file), "abc");
        assert_eq!(
            dir,
            std::path::Path::new("/home/u/.claude/sessions/enc/1700_abc.memory")
        );
        // Ephemeral: keyed by session id under the system temp dir.
        let eph = session_dir(None, "abc");
        assert!(eph.ends_with("beyond-ai-session-memory/abc"));
        assert!(eph.starts_with(std::env::temp_dir()));
    }
}
