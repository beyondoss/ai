use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// Basic metadata for one entry sitting in a [`SessionRepo`]'s `.trash/` subdirectory — see
/// [`SessionRepo::list_trash`]. Deliberately minimal: enough to identify and restore an entry, not a
/// full trash-management surface (title/preview/message-count are all still reachable by restoring the
/// session and reading it normally).
#[derive(Debug, Clone, Serialize)]
pub struct TrashEntry {
    /// The session id (the same `<id>` component [`SessionRepo::find_path`]/`path_for` key off).
    pub id: String,
    /// When the entry was moved into `.trash/` (its file's own mtime, Unix seconds) — `None` if the
    /// filesystem couldn't report one.
    pub deleted_at: Option<u64>,
    /// Where [`SessionRepo::restore_session`] would move this entry back to — this repo's own
    /// directory, under the file's original name (not `.trash/`'s copy).
    pub original_path: String,
}

/// A session's sibling `/session` working-memory directory is `<created_at>_<id>.memory/` next to its
/// `<created_at>_<id>.jsonl` (see [`crate::memory::session_dir`]). This derives the memory dir's name for
/// a given session file, so trash/restore can move the two together.
pub(super) fn sibling_memory_name(session_jsonl: &Path) -> Option<PathBuf> {
    session_jsonl
        .file_name()
        .map(|n| Path::new(n).with_extension("memory"))
}

/// Best-effort move of a session's sibling working-memory dir into `dst_dir`, alongside a `.jsonl` that
/// was just moved there (trash on delete, or back out on restore). Never errors: a missing dir (the
/// session had no working memory, or predates this feature) or a failed rename must not fail the session
/// operation — matching [`SessionRepo::delete`]'s best-effort contract for the soft-delete itself.
pub(super) fn move_sibling_memory(session_jsonl: &Path, dst_dir: &Path) {
    let src_mem = session_jsonl.with_extension("memory");
    if let Some(name) = sibling_memory_name(session_jsonl)
        && src_mem.is_dir()
    {
        let _ = fs::rename(&src_mem, dst_dir.join(name));
    }
    // The session's sidecar files (MCP Events state, MCP App view state, the journal key) travel
    // with it too.
    let sidecars = SESSION_SIDECARS
        .iter()
        .map(|ext| session_jsonl.with_extension(ext))
        .chain([
            journal_key_beside(session_jsonl),
            // A key under the old name not yet carried over: without it a restored session would
            // read as keyless.
            session_jsonl.with_extension(crate::mcp_resume::JournalAuth::SIDECAR),
        ]);
    for sidecar in sidecars {
        if let Some(name) = sidecar.file_name()
            && sidecar.is_file()
        {
            let _ = fs::rename(&sidecar, dst_dir.join(name));
        }
    }
}

/// The extensions of every per-session sidecar file kept beside a single-file session's `.jsonl`:
/// the MCP Events state (`tools::mcp_events::state_path_for` — the snapshot of cursors and keys, and
/// the append-only log of undelivered events) and the MCP App view state (`tools::mcp_apps::Sidecar`
/// — the attached view context, and the view replay store holding each kept view's HTML and full
/// tool result). Each trashes, restores and is removed with its session: left behind, it would keep
/// a deleted session's content on disk. The MCP task journal's key ([`journal_key_beside`]) goes
/// with them too, so its journal resumes wherever the session goes. (A segmented session keeps its
/// MCP App sidecars and its journal key inside its own directory, which moves whole.)
pub(super) const SESSION_SIDECARS: [&str; 4] = [
    "mcp-events.json",
    "mcp-events.log",
    "mcp-app-context.json",
    "mcp-app-views.json",
];

/// Where a single-file session's MCP task journal key lives: its whole file name plus a suffix
/// (`<session>.jsonl.mcp-task-journal.json`), not `with_extension`, so sessions whose names differ
/// only after their last dot (`work.1`, `work.2`) never share one key.
pub(super) fn journal_key_beside(session: &Path) -> PathBuf {
    let mut name = session.as_os_str().to_owned();
    name.push(".");
    name.push(crate::mcp_resume::JournalAuth::SIDECAR);
    PathBuf::from(name)
}

/// The MCP Events subset — the only sidecars a segmented session keeps *beside* its directory.
pub(super) const MCP_EVENTS_SIDECARS: [&str; 2] = ["mcp-events.json", "mcp-events.log"];

/// Best-effort removal of a session's sibling working-memory dir, for the hard-delete fallback path where
/// there is no `.trash/` to move it into. Same no-error contract as [`move_sibling_memory`].
pub(super) fn remove_sibling_memory(session_jsonl: &Path) {
    let src_mem = session_jsonl.with_extension("memory");
    if src_mem.is_dir() {
        let _ = fs::remove_dir_all(&src_mem);
    }
    for ext in SESSION_SIDECARS {
        let _ = fs::remove_file(session_jsonl.with_extension(ext));
    }
    let _ = fs::remove_file(journal_key_beside(session_jsonl));
    let _ = fs::remove_file(session_jsonl.with_extension(crate::mcp_resume::JournalAuth::SIDECAR));
}
