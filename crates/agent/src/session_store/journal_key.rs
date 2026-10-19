use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use super::*;

/// The journal auth for a session whose log is `log` (see [`crate::mcp_resume::JournalAuth`]):
/// sealed storage needs none; otherwise the key in the session's own sidecar, if it has one yet. A
/// session without one (it never journaled a task, or predates per-session keys) is unkeyed until
/// its first journal write makes one ([`make_journal_key`]). Reading never writes.
pub(super) fn journal_auth(log: &Log) -> crate::mcp_resume::JournalAuth {
    use crate::mcp_resume::JournalAuth;
    let Some(path) = log.journal_key_path() else {
        return JournalAuth::Storage;
    };
    read_journal_key(&path)
        .or_else(|| {
            log.legacy_journal_key_path()
                .and_then(|p| read_journal_key(&p))
        })
        .unwrap_or(JournalAuth::Unkeyed)
}

pub(super) fn read_journal_key(path: &Path) -> Option<crate::mcp_resume::JournalAuth> {
    crate::mcp_resume::JournalAuth::from_sidecar(&fs::read(path).ok()?)
}

/// How long a first journal write waits for another writer making the same session's key.
pub(super) const JOURNAL_KEY_LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Make the journal key of a session that has none yet, at its first journal write: the entries it
/// already holds (`nodes`, any branch; from before per-session keys) stay accepted. `None` when no
/// key belongs in a sidecar or none could be written (the write then goes out unsealed, as before).
///
/// Every first writer (threads of this process, other processes) must end up with the *same* key,
/// or the loser's entries would be refused on replay. So the key is made under an exclusive lock on
/// the sidecar itself (`file_lock`), and re-read once the lock is held: whoever got there first has
/// written it, and everyone after uses theirs. The key is written through the locked descriptor, in
/// place: no temp file for two writers to share, and none left beside the session. A sidecar that
/// does not parse (torn, or overwritten) is replaced the same way. A key file under the
/// pre-suffix name (`legacy_journal_key_path`) is carried over rather than replaced.
pub(super) fn make_journal_key(
    log: &Log,
    nodes: &HashMap<String, Node>,
) -> Option<crate::mcp_resume::JournalAuth> {
    use crate::mcp_resume::JournalAuth;
    use std::io::{Seek, SeekFrom};
    let path = log.journal_key_path()?;
    let failed = |e: &dyn std::fmt::Display| {
        tracing::warn!(path = %path.display(), error = %e, "MCP task journal key not written");
    };
    // Never write a key through a link someone planted at the sidecar's name.
    if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        failed(&"the sidecar is a symlink");
        return None;
    }
    let deadline = std::time::Instant::now() + JOURNAL_KEY_LOCK_WAIT;
    let lock = loop {
        match crate::file_lock::try_lock(crate::file_lock::Target::Itself(&path)) {
            Ok(Some(lock)) => break lock,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            Ok(None) => {
                failed(&"another writer held its lock too long");
                return None;
            }
            Err(e) => {
                failed(&e);
                return None;
            }
        }
    };
    // Re-read through the locked descriptor: a second open and close would drop the lock on NFS.
    let mut on_disk = Vec::new();
    if lock.file().read_to_end(&mut on_disk).is_ok()
        && let Some(theirs) = JournalAuth::from_sidecar(&on_disk)
    {
        return Some(theirs);
    }
    let (auth, bytes) = match log.legacy_journal_key_path().and_then(|p| {
        let bytes = fs::read(&p).ok()?;
        Some((JournalAuth::from_sidecar(&bytes)?, bytes))
    }) {
        Some(carried) => carried,
        None => {
            let entries = nodes.values().filter_map(|node| match &*node.content {
                NodeContent::Custom { kind, data } => Some((kind.as_str(), data)),
                NodeContent::Message(_) => None,
            });
            JournalAuth::fresh(entries)?
        }
    };
    let written = (|| -> std::io::Result<()> {
        let mut file = lock.file();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fsync_dir(&path)
    })();
    match written {
        Ok(()) => Some(auth),
        Err(e) => {
            failed(&e);
            None
        }
    }
}
