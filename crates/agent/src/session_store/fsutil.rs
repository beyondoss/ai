use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::*;

/// Create (or truncate) `path` for exclusive access: `0600` on Unix, set atomically at creation
/// rather than via a `set_permissions` call afterward (which would leave a window where the file is
/// world/group-readable). Session files carry the full transcript — including whatever `read` has
/// pulled off disk — so they should never be readable by anyone but the owner, independent of the
/// process umask.
pub(super) fn create_private(path: &Path) -> std::io::Result<File> {
    // Remove any leftover at this (deterministic `.tmp`) path first, then create it fresh with
    // `create_new`. `.mode()` only takes effect when the file is actually *created*, so a plain
    // `create(true).truncate(true)` on a pre-existing path would open-and-truncate a **pre-planted**
    // file — keeping its existing, possibly world-readable, mode — or follow a pre-planted symlink to
    // some other target, writing the transcript (which can hold secrets `read` pulled off disk)
    // through it. A stale `.tmp` from a prior crash is the normal reason something is already here;
    // unlink it (which removes a symlink itself, not its target) then create exclusively. If an
    // attacker races a fresh plant in between, `create_new` fails outright — fail closed, never
    // truncating or following it.
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

/// Create `path` holding exactly `bytes`, all at once: they go to a private temp file beside it
/// first, which is then hard-linked into place. `link` refuses a name that exists, so this is the same
/// exclusive-create fence `create_new` is — two creators racing for one id, exactly one wins — but the
/// name never exists holding less than all of `bytes`, so a crash mid-create leaves no half-made
/// session behind (at most a hidden temp file, which nothing reads). `replace_empty` initializes a
/// zero-byte file already at `path` (`touch`ed ahead of time) instead of refusing it.
pub(super) fn create_whole(path: &Path, bytes: &[u8], replace_empty: bool) -> std::io::Result<()> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("session");
    let tmp = path.with_file_name(format!(".{name}.{}.creating", new_id()));
    let placed = (|| -> std::io::Result<()> {
        let mut f = create_private(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        match fs::hard_link(&tmp, path) {
            Err(e)
                if e.kind() == std::io::ErrorKind::AlreadyExists
                    && replace_empty
                    && fs::metadata(path).is_ok_and(|m| m.len() == 0) =>
            {
                fs::rename(&tmp, path)
            }
            other => other,
        }
    })();
    // Gone already after a rename; a link leaves it as a second name to drop.
    let _ = fs::remove_file(&tmp);
    placed?;
    fsync_dir(path)
}

/// Whether the session file (or first segment) at `path` holds nothing but the start of its first
/// line — a crash between creating it and writing it, under a build that did not create files whole
/// ([`create_whole`]): empty, or no newline yet and its bytes a prefix of what that first line
/// begins with (`expected`). Nothing else counts, so a file holding anything real is never taken for
/// one.
pub(super) fn is_headerless(path: &Path, expected: &[u8]) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    if bytes.contains(&b'\n') {
        return false;
    }
    let n = bytes.len().min(expected.len());
    bytes[..n] == expected[..n]
}

/// Move a headerless first file ([`is_headerless`]) out of the way so its id can be created again.
/// Renamed aside before it is judged, so a creator that put a whole file there in the meantime is
/// never deleted: what was moved is checked again, and put back if it turns out to be real. `true`
/// when the name is free.
pub(super) fn clear_headerless(path: &Path, expected: &[u8]) -> std::io::Result<bool> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("session");
    let aside = path.with_file_name(format!(".{name}.{}.headerless", new_id()));
    fs::rename(path, &aside)?;
    if is_headerless(&aside, expected) {
        tracing::warn!(path = %path.display(), "re-creating a session whose create never wrote its header");
        fs::remove_file(&aside)?;
        return Ok(true);
    }
    match fs::hard_link(&aside, path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    fs::remove_file(&aside)?;
    Ok(false)
}

/// Replace `path` with `bytes` the way a transcript rewrite does: a `0600` temp file beside it
/// ([`create_private`]), `fsync`ed, renamed over `path`, then the directory `fsync`ed — so a reader
/// (or a crash) sees the old content or the new, never a torn file, and never a group/world-readable
/// one. For a session's sidecar files, which can carry as much as the transcript does.
pub(crate) fn write_private_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut f = create_private(&tmp)?;
    let write = (|| -> std::io::Result<()> {
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        fsync_dir(path)
    })();
    if write.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    write
}

/// Remove `path` durably (the directory `fsync`ed after); absent already is success.
pub(crate) fn remove_durably(path: &Path) -> std::io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => fsync_dir(path),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Tighten an existing session file to `0600` before writing more into it. `create_private` sets
/// the mode when a file is created, but a file that already exists keeps whatever mode it has (an
/// older version's, a restore's, or a `chmod`), and every append adds transcript, MCP task ids
/// (which a server may treat as bearer tokens, see `tools::mcp::McpTaskRecord`) included. One
/// `fstat` per append, beside the append's own `fsync`.
pub(super) fn keep_private(file: &File) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if file.metadata()?.permissions().mode() & 0o077 != 0 {
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
    }
    #[cfg(not(unix))]
    let _ = file;
    Ok(())
}

pub(super) fn fsync_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        // Opening the directory read-only and `sync_all`ing it flushes its updated entries. Linux/macOS
        // allow fsync on a directory handle; this is the standard atomic-rename durability step.
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}
