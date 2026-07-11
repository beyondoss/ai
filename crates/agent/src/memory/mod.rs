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

/// The fixed logical root every memory path lives under, matching Anthropic's `memory_20250818` tool.
/// A `view` of exactly this path lists the store; everything else is a document beneath it.
pub const MEMORY_ROOT: &str = "/memories";

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
    /// Clean relative path beneath the root; `""` is the root itself. Never has a leading/trailing `/`,
    /// never an empty/`.`/`..` component.
    rel: String,
}

impl MemPath {
    /// Parse and validate a model-supplied path. Accepts `/memories`, `/memories/<rel>`,
    /// `memories/<rel>`, or a bare relative `<rel>` (treated as under the root); rejects any other
    /// absolute path and any `.`/`..`/empty component.
    pub fn parse(raw: &str) -> Result<Self, MemoryError> {
        let trimmed = raw.trim();
        let bad = || MemoryError::InvalidPath(raw.to_string());

        // Peel the logical root off the front, however the model spelled it.
        let remainder = if trimmed == MEMORY_ROOT || trimmed == "memories" {
            ""
        } else if let Some(r) = trimmed.strip_prefix("/memories/") {
            r
        } else if let Some(r) = trimmed.strip_prefix("memories/") {
            r
        } else if trimmed.starts_with('/') {
            // An absolute path that isn't under /memories — never allowed to escape the store.
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
            rel: clean.join("/"),
        })
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

    /// The full logical path (`/memories/...`) for display back to the model.
    pub fn display(&self) -> String {
        if self.rel.is_empty() {
            MEMORY_ROOT.to_string()
        } else {
            format!("{MEMORY_ROOT}/{}", self.rel)
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

/// The `## Memory` guidance block injected into the system prompt whenever the `memory` tool is present.
/// Tells the model the store exists, how to reach it, and — critically — the curation discipline that
/// keeps a durable memory useful instead of a junk drawer. Mirrors the conventions in Claude Code's own
/// auto-memory. The bounded [`INDEX_FILE`] contents are injected separately, right after this.
pub const MEMORY_GUIDANCE: &str = "\
## Memory

You have a persistent, cross-session memory store at `/memories`, reached through the `memory` tool \
(commands: `view`, `create`, `str_replace`, `insert`, `delete`, `rename`, `search`). It survives across \
sessions on this project — use it to remember durable facts that would help future sessions: build and \
test commands, non-obvious architecture decisions, debugging lessons, the user's stated preferences.

- `MEMORY.md` is the index. Its current contents are shown below and reloaded at the start of every \
session. Keep it a lean list of one-line pointers to topic files — do not put memory content in it.
- Store one fact (or one tightly-related cluster) per topic file, then add a one-line pointer to it in \
`MEMORY.md`. Read a topic file with `memory` `view` only when its pointer looks relevant.
- `create` makes a *new* file and fails if one already exists; to change an existing memory use \
`str_replace`/`insert` (e.g. append a new pointer line to `MEMORY.md`), not `create`.
- Before saving, check whether an existing memory already covers it and update that instead of creating \
a duplicate. Delete a memory once you learn it is wrong.
- Save only what is genuinely useful to a future session and not already obvious from the repo, its \
docs, or version history. Convert relative dates to absolute ones.";

/// Render the injected memory section: the [`MEMORY_GUIDANCE`] block, followed by the current
/// (already-bounded) index contents when there are any. `index` is what [`MemoryBackend::index`]
/// returned. Used by [`crate::resources`] when assembling the system prompt.
pub fn render_section(index: &str) -> String {
    let index = index.trim();
    if index.is_empty() {
        format!("{MEMORY_GUIDANCE}\n\nYour `MEMORY.md` index is currently empty.")
    } else {
        format!("{MEMORY_GUIDANCE}\n\nCurrent `MEMORY.md` index:\n\n{index}")
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
    fn render_section_handles_an_empty_and_a_populated_index() {
        assert!(render_section("").contains("currently empty"));
        let s = render_section("- [x](x.md) — a thing");
        assert!(s.contains("Current `MEMORY.md` index"));
        assert!(s.contains("[x](x.md)"));
    }
}
