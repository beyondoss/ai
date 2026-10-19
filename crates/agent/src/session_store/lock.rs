use std::path::Path;

// ---- The session lock --------------------------------------------------------------------------

/// A held advisory lock on one session ([`crate::file_lock::FileLock`]). Dropping it releases the
/// lock.
pub type SessionLock = crate::file_lock::FileLock;

/// Take the advisory lock for the session at `session_path` — the session directory for the segmented
/// layout, the `.jsonl` file otherwise. `Ok(None)` means someone else holds it *right now*.
///
/// The lock is **liveness only**: it says a process is alive and working on this session, nothing
/// more. Correctness comes from the epoch fence, which needs no lock at all — so a lock lost to a
/// crash, a network partition or a stuck NFS client costs a retry, never history.
///
/// It is an open file description lock ([`crate::file_lock`]): no other descriptor of the lock file
/// releases it, a second acquire in this process is refused, and it is released with an explicit
/// unlock — a forked child inherits the description, but cannot keep the lock once its holder lets
/// go. After locking, `fstat` on the held descriptor is compared with `stat` of the path, so a
/// directory renamed into `.trash/` between the open and the lock is caught rather than silently
/// "locked". This only says which lock file is the session's.
pub fn acquire_session_lock(session_path: &Path) -> std::io::Result<Option<SessionLock>> {
    crate::file_lock::try_lock(if session_path.is_dir() {
        crate::file_lock::Target::Dir(session_path)
    } else {
        crate::file_lock::Target::File(session_path)
    })
}

/// The single-writer lock a process holds on a single-file session for as long as it may write it —
/// `run`, and `serve` outside service mode, take it right after opening (before any write; see
/// `Log::File::torn_at`) and keep it with the store. A second process asking for a session already
/// held is **refused**, with an error naming the file ([`std::io::ErrorKind::WouldBlock`]), rather
/// than opened read-only: a turn on a read-only session can't be saved, and the operator asked for one
/// that is. Two writers can't share a single-file session at all — `append_new` is count-keyed, so
/// neither sees the other's turns and each chains its next one onto its own stale tip.
///
/// `Ok(None)` for the segmented layout: its writers are fenced by epoch, and service mode takes the
/// session lock itself (`serve_ws`), which a second acquire in the same process would be refused.
pub fn lock_session_writer(session_path: &Path) -> std::io::Result<Option<SessionLock>> {
    if session_path.is_dir() {
        return Ok(None);
    }
    match acquire_session_lock(session_path)? {
        Some(lock) => Ok(Some(lock)),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            format!(
                "session {} is open in another agent process; wait for it to finish, or fork it",
                session_path.display()
            ),
        )),
    }
}
