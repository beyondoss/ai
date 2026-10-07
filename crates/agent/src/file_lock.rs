//! An advisory lock on one file: a kernel lock (`flock`, emulated with POSIX record locks over NFS)
//! plus a process-wide registry of the paths this process holds.
//!
//! The kernel releases the lock when its holder exits, however it exits, so there is no staleness to
//! judge and no lockfile to break: a crashed holder's leftover file is simply unlocked. The registry
//! is there because closing *any* descriptor to a POSIX-locked file drops the lock — a second open of
//! the same path in this process would quietly release the first one's hold — so a second attempt in
//! the same process reports "held" instead of opening it.
//!
//! Used for a session's liveness lock (`session_store::acquire_session_lock`) and the MCP manifest
//! cache's write lock (`tools::mcp_manifest`). Non-blocking: [`try_lock`] answers at once, and a
//! caller that wants to wait polls it off the async runtime.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// How many times [`try_lock`] re-opens after finding it locked a file that is no longer the one at
/// the path (the file, or a directory holding it, renamed away mid-acquire).
const RETRIES: usize = 5;

/// A held lock. Dropping it closes the descriptor (releasing the kernel lock) and frees the
/// registration.
pub struct FileLock {
    _file: File,
    _registration: Registration,
}

/// "This process holds the lock at this path."
struct Registration(PathBuf);

impl Drop for Registration {
    fn drop(&mut self) {
        held()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.0);
    }
}

fn held() -> &'static Mutex<HashSet<PathBuf>> {
    static HELD: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    HELD.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Take the lock on `lock_path`, creating the file (mode 0600) if needed. `Ok(None)` means someone —
/// another process, or this one — holds it *right now*.
///
/// The descriptor is opened read+write, because NFS emulates `flock` with POSIX record locks and those
/// need a writable descriptor. After locking, `fstat` on the held descriptor is compared with `stat`
/// of the path, so a file renamed away between the open and the lock is caught rather than silently
/// "locked".
pub fn try_lock(lock_path: &Path) -> std::io::Result<Option<FileLock>> {
    {
        let mut held = held()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !held.insert(lock_path.to_path_buf()) {
            return Ok(None);
        }
    }
    // Registered from here on, so every path out of this function frees the entry.
    let registration = Registration(lock_path.to_path_buf());
    for _ in 0..RETRIES {
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let file = opts.open(lock_path)?;
        #[cfg(test)]
        tests::between_open_and_lock(lock_path);
        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Ok(None),
            Err(std::fs::TryLockError::Error(e)) => return Err(e),
        }
        if same_file(&file, lock_path)? {
            return Ok(Some(FileLock {
                _file: file,
                _registration: registration,
            }));
        }
        // The file we locked is no longer the file at that path. Drop it and look again.
        drop(file);
    }
    Ok(None)
}

/// Whether the open descriptor and `path` still name the same inode.
fn same_file(file: &File, path: &Path) -> std::io::Result<bool> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let a = file.metadata()?;
        let Ok(b) = std::fs::metadata(path) else {
            return Ok(false);
        };
        Ok(a.dev() == b.dev() && a.ino() == b.ino())
    }
    #[cfg(not(unix))]
    {
        let _ = (file, path);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Run (on this thread) after `try_lock` opens the file and before it locks it.
    type Hook = Box<dyn FnMut(&Path)>;

    thread_local! {
        static BETWEEN: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    pub(super) fn between_open_and_lock(path: &Path) {
        BETWEEN.with(|hook| {
            if let Some(hook) = hook.borrow_mut().as_mut() {
                hook(path);
            }
        });
    }

    /// The file at the path replaced between the open and the lock — a directory holding it renamed
    /// away, a crashed holder's file swapped for a fresh one — must not leave the caller "holding" a
    /// lock on a file nobody else will ever open. `try_lock` sees the inode moved and starts over, and
    /// the lock it returns is on the file now at the path.
    #[cfg(unix)]
    #[test]
    fn a_lock_file_replaced_between_open_and_lock_is_not_the_one_held() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.lock");
        let opens = std::rc::Rc::new(std::cell::Cell::new(0));
        let counted = opens.clone();
        let fresh = dir.path().join("fresh");
        BETWEEN.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |path: &Path| {
                counted.set(counted.get() + 1);
                if counted.get() == 1 {
                    std::fs::write(&fresh, b"").unwrap();
                    std::fs::rename(&fresh, path).unwrap();
                }
            }));
        });
        let held = try_lock(&path).unwrap().expect("taken on the second look");
        BETWEEN.with(|hook| *hook.borrow_mut() = None);
        assert_eq!(opens.get(), 2, "the moved inode sent it round again");
        // The file at the path is the locked one: another open file description cannot lock it.
        let other = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(
            matches!(other.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
            "the lock held is on the file now at the path"
        );
        drop(held);
    }

    /// One holder at a time within the process (the registry), and the lock is free again once the
    /// holder drops it.
    #[test]
    fn one_holder_at_a_time_and_free_again_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.lock");
        let held = try_lock(&path).unwrap().expect("free at first");
        assert!(try_lock(&path).unwrap().is_none(), "held by this process");
        drop(held);
        assert!(try_lock(&path).unwrap().is_some(), "free once dropped");
    }

    /// A path this process holds is refused to it a second time without opening the file — so even
    /// when the file at the path is replaced underneath the holder, this process never ends up
    /// holding the path twice. (On NFS the kernel lock is a POSIX record lock, which closing *any*
    /// descriptor drops; the registry is what keeps a second attempt from opening one.)
    #[test]
    fn a_path_held_here_is_refused_here_even_if_its_file_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.lock");
        let _held = try_lock(&path).unwrap().expect("free at first");
        std::fs::remove_file(&path).unwrap();
        assert!(
            try_lock(&path).unwrap().is_none(),
            "this process already holds the path"
        );
        assert!(
            !path.exists(),
            "the refused attempt never opened (and so created) the file"
        );
    }

    /// Different paths are different locks.
    #[test]
    fn locks_on_different_paths_are_independent() {
        let dir = tempfile::tempdir().unwrap();
        let _a = try_lock(&dir.path().join("a.lock")).unwrap().unwrap();
        assert!(try_lock(&dir.path().join("b.lock")).unwrap().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn same_file_catches_a_path_whose_inode_moved() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lock");
        std::fs::write(&path, b"").unwrap();
        let held = File::open(&path).unwrap();
        assert!(same_file(&held, &path).unwrap());

        // Another file takes the path — what a session directory renamed into `.trash/` looks like
        // from the lock's point of view.
        let other = dir.path().join("other");
        std::fs::write(&other, b"").unwrap();
        std::fs::rename(&other, &path).unwrap();
        assert!(
            !same_file(&held, &path).unwrap(),
            "the descriptor no longer names the file at that path"
        );

        // And a path that is simply gone is not the same file either.
        std::fs::remove_file(&path).unwrap();
        assert!(!same_file(&held, &path).unwrap());
    }
}
