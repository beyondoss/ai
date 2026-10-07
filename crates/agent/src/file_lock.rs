//! An advisory lock on one resource (a session directory, a session file, the MCP manifest).
//!
//! The kernel releases the lock when its holder exits, however it exits, so there is no staleness to
//! judge and no lockfile to break: a crashed holder's leftover file is simply unlocked. Used for a
//! session's liveness lock (`session_store::acquire_session_lock`), the MCP manifest cache's write
//! lock (`tools::mcp_manifest`) and the memory store's lock (`memory::file`). Non-blocking: [`try_lock`] answers at once (bar a short retry, below),
//! and a caller that wants to wait polls it off the async runtime.
//!
//! **The lock is an open file description (OFD) lock** on Linux (`fcntl(F_OFD_SETLK)`, whole file,
//! through `nix`), on a lock file only this module names. What it belongs to is the one `open` this
//! module made, which settles the two ways a lock used to be lost:
//!
//! - **Another descriptor of the same file** — a file tool or the memory tool reading it through a
//!   symlink or a hard link, anything — is a different open file description: opening and closing it
//!   neither releases this lock nor shares it. (A POSIX record lock belongs to the *process*, so
//!   closing *any* of its descriptors of the file dropped it.) Two descriptions conflict even inside
//!   one process, so a second acquire here — under any spelling of the path — is refused by the
//!   kernel itself; there is no in-process registry to keep in step.
//! - **A forked child.** A child forked by any thread (a `bash` call, an MCP stdio server) shares the
//!   description until it execs, and with it the lock. So the lock is released with an explicit
//!   `F_UNLCK` on this descriptor before it closes, which drops it whatever copies a child holds.
//!   (`flock` without that explicit unlock is what made a just-released lock look held 180 times in
//!   3000 while other threads spawned processes.)
//!
//! Off Linux there are no OFD locks: there it is `flock`, which also belongs to the open file
//! description, released the same way with an explicit `LOCK_UN`.
//!
//! **On NFS/EFS** the Linux client sends an OFD lock to the server as a POSIX byte-range lock whose
//! owner is the open file description, not the process — so another description's open/close does not
//! release it there either, and a second description on the same client conflicts with it (verified
//! against a loopback NFSv4.2 mount; see ARCHITECTURE.md). `flock` on NFS, by contrast, is emulated
//! with a process-owned POSIX lock, which is exactly why it is not the authoritative half.
//!
//! **Mixed rollout.** Binaries before this one took an `flock` on `<dir>/lock` / `<file>.lock` and
//! nothing else. So, for now, [`try_lock`] also takes that `flock` (on its own descriptor,
//! close-on-exec, released with an explicit `LOCK_UN`), and a held one means "held": an old binary
//! still excludes this one, and this one still excludes an old one. It is taken with a short bounded
//! retry ([`LEGACY_RETRY`]), since an old binary's own fork→exec window can briefly hold it. The OFD
//! lock is the authoritative one; **the legacy `flock` half is removed in a later release, once no
//! binary older than this one can be running against the same files.**

use std::ffi::{OsStr, OsString};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// How many times [`try_lock`] re-opens after finding it locked a file that is no longer the one at
/// the path (the file, or a directory holding it, renamed away mid-acquire).
const RETRIES: usize = 5;

/// The lock file's name: `<dir>/.beyond-lock` for a directory, `<file>.beyond-lock` for a file.
const RECORD_SUFFIX: &str = ".beyond-lock";

/// How long the legacy `flock` is retried before it counts as held.
const LEGACY_RETRY: Duration = Duration::from_millis(250);
const LEGACY_RETRY_INTERVAL: Duration = Duration::from_millis(2);

/// What is being locked: a directory (its lock files live inside it), a file (they sit beside it), or
/// a file locked **itself** — a small file whose holder writes its content through the locked
/// descriptor ([`FileLock::file`]), as the MCP task journal key is made: lock it, re-read it, write it.
#[derive(Clone, Copy, Debug)]
pub enum Target<'a> {
    Dir(&'a Path),
    File(&'a Path),
    Itself(&'a Path),
}

impl Target<'_> {
    /// The authoritative lock file.
    fn record_path(&self) -> PathBuf {
        match self {
            Target::Dir(dir) => dir.join(RECORD_SUFFIX),
            Target::File(file) => with_suffix(file, RECORD_SUFFIX),
            Target::Itself(file) => file.to_path_buf(),
        }
    }

    /// The legacy `flock` file older binaries lock. None for a file locked itself: no released
    /// binary locked one (the journal key's lock is newer than any of them), and an `flock` on the
    /// same file would conflict with this process's own OFD lock on NFS, where `flock` is emulated
    /// with a process-owned POSIX lock.
    fn legacy_path(&self) -> Option<PathBuf> {
        match self {
            Target::Dir(dir) => Some(dir.join("lock")),
            Target::File(file) => Some(with_suffix(file, ".lock")),
            Target::Itself(_) => None,
        }
    }
}

fn with_suffix(file: &Path, suffix: &str) -> PathBuf {
    let mut name: OsString = file.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

/// Whether `path` is named like a record lock file — compared without regard to ASCII case, since on
/// a case-insensitive filesystem (macOS) `.BEYOND-LOCK` is the same file. For copying code that
/// should leave locks behind (a worktree seed: a copy of a lock file is not a lock).
pub fn is_record_lock_file(path: &Path) -> bool {
    path.file_name().and_then(OsStr::to_str).is_some_and(|n| {
        n.len() >= RECORD_SUFFIX.len()
            && n.as_bytes()[n.len() - RECORD_SUFFIX.len()..]
                .eq_ignore_ascii_case(RECORD_SUFFIX.as_bytes())
    })
}

/// Whether an atomic write must not rename a fresh file over `path`. Replacing a lock file leaves the
/// lock on the old inode, which the path no longer names, so the next owner would lock the new file.
///
/// **This never opens `path`, and never locks it** — it only `stat`s. Opening a FIFO or a device
/// blocks or acts, and a test lock makes another program's own non-blocking lock fail while it is
/// held. So a file is a lock file only when one of these says so:
///
/// - **Its name**, without regard to ASCII case (a case-insensitive filesystem folds `.BEYOND-LOCK`
///   onto `.beyond-lock`): a record lock file, or a legacy `lock` / `<f>.lock` beside one.
/// - **It is a lock this process holds**: its (device, inode) is in the held set [`try_lock`] keeps.
///   That covers a [`Target::Itself`] key (any name) and any spelling or hard link of this process's
///   own lock files.
/// - **An old binary's legacy lock with no record lock file beside it**, by where it sits: a `lock`
///   in a session directory (one holding `000001.jsonl`, which is never deleted), or a `<f>.lock`
///   beside a session file (`*.jsonl`), the MCP manifest or the memory store's `.memory`. Whether
///   anyone holds it is not asked.
///
/// Anything else — a SQLite database another program holds open and locked, a daemon's pid file — is
/// an ordinary file, and an edit of it goes through.
pub fn is_lock_file(path: &Path) -> bool {
    is_lock_file_by_name(path) || is_held_here(path)
}

fn is_lock_file_by_name(path: &Path) -> bool {
    if is_record_lock_file(path) {
        return true;
    }
    let Some(name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    if lower == "lock" {
        return path.with_file_name(RECORD_SUFFIX).exists()
            || path.with_file_name("000001.jsonl").is_file();
    }
    let Some(stem) = lower.strip_suffix(".lock") else {
        return false;
    };
    stem.ends_with(".jsonl")
        || stem == "mcp-manifest.json"
        || stem == ".memory"
        || path
            .with_file_name(format!("{}{RECORD_SUFFIX}", &name[..stem.len()]))
            .exists()
}

type FileId = (u64, u64);

/// The (device, inode) of every lock file this process holds or is acquiring, counted (see
/// [`Marks`]).
#[cfg(unix)]
fn held() -> std::sync::MutexGuard<'static, std::collections::HashMap<FileId, usize>> {
    static HELD: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<FileId, usize>>> =
        std::sync::OnceLock::new();
    HELD.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

#[cfg(unix)]
fn file_id(meta: &std::fs::Metadata) -> FileId {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

/// The held-set entries one acquire made, taken out again when it is dropped. An acquire marks each
/// lock file **before** it tries to lock it, and an acquire that fails drops its marks — so there is
/// no moment when a lock is held but its file reads as not held. (A brief false "held" while an
/// acquire is in flight is the safe direction.) A [`FileLock`] keeps its marks until it is dropped,
/// which is after it has unlocked.
#[derive(Default)]
struct Marks(Vec<FileId>);

impl Marks {
    fn mark(&mut self, file: &File) {
        #[cfg(unix)]
        if let Ok(meta) = file.metadata() {
            let id = file_id(&meta);
            *held().entry(id).or_default() += 1;
            self.0.push(id);
        }
        #[cfg(not(unix))]
        let _ = file;
    }
}

impl Drop for Marks {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.0.is_empty() {
            let mut held = held();
            for id in self.0.drain(..) {
                if let std::collections::hash_map::Entry::Occupied(mut e) = held.entry(id) {
                    *e.get_mut() -= 1;
                    if *e.get() == 0 {
                        e.remove();
                    }
                }
            }
        }
    }
}

/// Whether the file at `path` is one this process holds a lock on — by `stat`, nothing opened.
fn is_held_here(path: &Path) -> bool {
    #[cfg(unix)]
    {
        std::fs::metadata(path).is_ok_and(|m| held().contains_key(&file_id(&m)))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// Whether `dir` holds nothing but its own lock files — a lock directory that never became anything
/// (a session directory whose start failed before its first segment).
pub fn dir_holds_only_lock_files(dir: &Path) -> bool {
    let record = Target::Dir(dir).record_path();
    let legacy = Target::Dir(dir).legacy_path().unwrap_or_default();
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().all(|e| {
            let p = e.path();
            p == record || p == legacy
        })
    })
}

/// A held lock. Dropping it unlocks each descriptor explicitly (so a forked child's copy of the
/// description cannot keep it), then closes them.
pub struct FileLock {
    record: File,
    legacy: Option<File>,
    record_path: PathBuf,
    legacy_path: Option<PathBuf>,
    /// Dropped after [`FileLock`]'s own `Drop` has unlocked, so the set never misses a held lock.
    _marks: Marks,
}

impl FileLock {
    /// The locked file, for a holder of a [`Target::Itself`] lock that writes the file's own content.
    /// (Another descriptor would not release the lock — it belongs to this one's description — but
    /// writing through the descriptor that holds it keeps "re-read, then write" one step.)
    pub fn file(&self) -> &File {
        &self.record
    }

    /// Release the lock and unlink its files — for a lock directory being taken back. What matters is
    /// that the descriptors are closed before the directory is removed: an NFS client turns the unlink
    /// of a file it still has open into a rename to `.nfs*`, which stays until the last close and keeps
    /// `remove_dir` failing (ENOTEMPTY). Closing first means the unlinks are real ones.
    pub fn release_and_remove_files(self) {
        let (record_path, legacy_path) = (self.record_path.clone(), self.legacy_path.clone());
        drop(self);
        let _ = std::fs::remove_file(&record_path);
        if let Some(legacy_path) = &legacy_path {
            let _ = std::fs::remove_file(legacy_path);
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        unlock(&self.record);
        if let Some(legacy) = &self.legacy {
            let _ = legacy.unlock();
        }
    }
}

/// Take the lock on `target`, creating its lock files (mode 0600) if needed; the directory a lock
/// file goes in must exist. `Ok(None)` means someone — another process, or another acquire in this
/// one — holds it *right now*.
///
/// Each descriptor is opened read+write, because a byte-range lock needs a writable one. After
/// locking, `fstat` on the held descriptor is compared with `stat` of the path, so a file renamed away
/// between the open and the lock is caught rather than silently "locked".
pub fn try_lock(target: Target<'_>) -> std::io::Result<Option<FileLock>> {
    let record_path = target.record_path();
    let legacy_path = target.legacy_path();
    for _ in 0..RETRIES {
        let mut marks = Marks::default();
        let record = open_lock_file(&record_path)?;
        marks.mark(&record);
        #[cfg(test)]
        tests::between_open_and_lock(&record_path);
        if !try_lock_description(&record)? {
            return Ok(None);
        }
        if !same_file(&record, &record_path)? {
            // The file we locked is no longer the file at that path. Let it go and look again.
            unlock(&record);
            continue;
        }
        let legacy = match legacy_path
            .as_deref()
            .map(|path| take_legacy_flock(path, &mut marks))
            .transpose()?
        {
            None => None,
            Some(Legacy::Taken(file)) => Some(file),
            // An old binary holds it: held.
            Some(Legacy::Held) => {
                unlock(&record);
                return Ok(None);
            }
            Some(Legacy::Moved) => {
                unlock(&record);
                continue;
            }
        };
        return Ok(Some(FileLock {
            record,
            legacy,
            record_path,
            legacy_path,
            _marks: marks,
        }));
    }
    Ok(None)
}

fn open_lock_file(path: &Path) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn whole_file(l_type: libc::c_int) -> libc::flock {
    libc::flock {
        l_type: l_type as libc::c_short,
        l_whence: libc::SEEK_SET as libc::c_short,
        l_start: 0,
        l_len: 0,
        // Must be 0 for an OFD lock.
        l_pid: 0,
    }
}

/// An exclusive, non-blocking, whole-file lock owned by `file`'s open file description; `Ok(false)`
/// when another description (in any process, this one included) holds it.
fn try_lock_description(file: &File) -> std::io::Result<bool> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let lock = whole_file(libc::F_WRLCK);
        match nix::fcntl::fcntl(file, nix::fcntl::FcntlArg::F_OFD_SETLK(&lock)) {
            Ok(_) => Ok(true),
            Err(nix::errno::Errno::EAGAIN | nix::errno::Errno::EACCES) => Ok(false),
            Err(e) => Err(std::io::Error::from(e)),
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        match file.try_lock() {
            Ok(()) => Ok(true),
            Err(std::fs::TryLockError::WouldBlock) => Ok(false),
            Err(std::fs::TryLockError::Error(e)) => Err(e),
        }
    }
}

/// Release `file`'s lock explicitly — not by closing, which a forked child's copy of the description
/// would outlive.
fn unlock(file: &File) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let lock = whole_file(libc::F_UNLCK);
        let _ = nix::fcntl::fcntl(file, nix::fcntl::FcntlArg::F_OFD_SETLK(&lock));
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = file.unlock();
    }
}

enum Legacy {
    Taken(File),
    Held,
    /// The file at the path was replaced mid-acquire: look again from the top.
    Moved,
}

/// The legacy `flock` older binaries take (see the module doc), retried for [`LEGACY_RETRY`].
fn take_legacy_flock(path: &Path, marks: &mut Marks) -> std::io::Result<Legacy> {
    let file = open_lock_file(path)?;
    marks.mark(&file);
    let deadline = Instant::now() + LEGACY_RETRY;
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(LEGACY_RETRY_INTERVAL);
            }
            Err(std::fs::TryLockError::WouldBlock) => return Ok(Legacy::Held),
            Err(std::fs::TryLockError::Error(e)) => return Err(e),
        }
    }
    if !same_file(&file, path)? {
        let _ = file.unlock();
        return Ok(Legacy::Moved);
    }
    Ok(Legacy::Taken(file))
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
pub(crate) mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    /// Run (on this thread) after `try_lock` opens the lock file and before it locks it.
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

    /// The lock file a target's lock lives on (for a test that links to it).
    pub(crate) fn record_path(target: Target<'_>) -> PathBuf {
        target.record_path()
    }

    /// What another process sees: `Some(true)` if it cannot take a POSIX record lock on `target`'s
    /// lock file right now (which an OFD lock held anywhere else conflicts with), `Some(false)` if it
    /// can (and lets go at once). `None` without `python3`.
    pub(crate) fn held_for_another_process(target: Target<'_>) -> Option<bool> {
        let out = Command::new("python3")
            .args([
                "-c",
                "import fcntl,sys\nf=open(sys.argv[1],'r+')\ntry:\n fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n print('free')\nexcept OSError:\n print('held')",
            ])
            .arg(target.record_path())
            .output()
            .ok()?;
        match String::from_utf8_lossy(&out.stdout).trim() {
            "held" => Some(true),
            "free" => Some(false),
            _ => None,
        }
    }

    /// The file at the path replaced between the open and the lock must not leave the caller
    /// "holding" a lock on a file nobody else will ever open: `try_lock` sees the inode moved and
    /// starts over, and the lock it returns is on the file now at the path.
    #[cfg(unix)]
    #[test]
    fn a_lock_file_replaced_between_open_and_lock_is_not_the_one_held() {
        let dir = tempfile::tempdir().unwrap();
        let target_file = dir.path().join("x");
        let target = Target::File(&target_file);
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
        let held = try_lock(target).unwrap().expect("taken on the second look");
        BETWEEN.with(|hook| *hook.borrow_mut() = None);
        assert_eq!(opens.get(), 2, "the moved inode sent it round again");
        if let Some(seen) = held_for_another_process(target) {
            assert!(seen, "the lock held is on the file now at the path");
        }
        drop(held);
    }

    /// One holder at a time — a second acquire in the same process is refused by the kernel (two open
    /// file descriptions conflict) — and the lock is free again once the holder drops it.
    #[test]
    fn one_holder_at_a_time_in_this_process_and_free_again_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("x");
        let held = try_lock(Target::File(&file))
            .unwrap()
            .expect("free at first");
        assert!(
            try_lock(Target::File(&file)).unwrap().is_none(),
            "a second acquire in this process is refused"
        );
        drop(held);
        assert!(
            try_lock(Target::File(&file)).unwrap().is_some(),
            "free once dropped"
        );
    }

    /// Different targets are different locks.
    #[test]
    fn locks_on_different_targets_are_independent() {
        let dir = tempfile::tempdir().unwrap();
        let _a = try_lock(Target::File(&dir.path().join("a")))
            .unwrap()
            .unwrap();
        assert!(
            try_lock(Target::File(&dir.path().join("b")))
                .unwrap()
                .is_some()
        );
    }

    /// Opening and closing the lock file through another descriptor — by its own name, a symlink or a
    /// hard link — neither releases the lock nor takes it: it belongs to the description that locked
    /// it. (With a process-owned record lock, any such close dropped it.)
    #[cfg(unix)]
    #[test]
    fn another_descriptor_of_the_lock_file_neither_releases_nor_shares_it() {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("s1");
        std::fs::create_dir_all(&session).unwrap();
        let target = Target::Dir(&session);
        let held = try_lock(target).unwrap().unwrap();
        let symlink = dir.path().join("innocent.txt");
        std::os::unix::fs::symlink(target.record_path(), &symlink).unwrap();
        let hardlink = dir.path().join("also-innocent.txt");
        std::fs::hard_link(target.record_path(), &hardlink).unwrap();
        for path in [&target.record_path(), &symlink, &hardlink] {
            drop(std::fs::read(path).unwrap());
            drop(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .unwrap(),
            );
        }
        assert!(
            try_lock(target).unwrap().is_none(),
            "still held against this process"
        );
        if let Some(seen) = held_for_another_process(target) {
            assert!(seen, "and against another");
        }
        drop(held);
        if let Some(seen) = held_for_another_process(target) {
            assert!(!seen, "released when the holder lets go");
        }
    }

    /// The unlock is explicit, so a forked child that still holds a copy of the description (between
    /// its fork and its exec) cannot keep the lock alive after its holder lets go.
    #[cfg(unix)]
    #[test]
    fn a_released_lock_is_free_while_a_child_still_holds_the_description() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("x");
        let target = Target::File(&file);
        let held = try_lock(target).unwrap().unwrap();
        // A child holding a copy of the lock's description for the whole test — the fork→exec
        // window, made long: a `dup` (which drops close-on-exec) survives into it.
        let inherited = nix::unistd::dup(&held.record).unwrap();
        let Ok(mut child) = Command::new("sh")
            .args(["-c", "read _"])
            .stdin(Stdio::piped())
            .spawn()
        else {
            return;
        };
        drop(inherited);
        drop(held);
        let retaken = try_lock(target).unwrap();
        drop(child.stdin.take());
        let _ = child.wait();
        assert!(
            retaken.is_some(),
            "the explicit unlock released it despite the child's copy"
        );
    }

    /// A file locked *itself* (the MCP task journal key): one holder at a time, the holder writes
    /// through the locked descriptor, and no lock file appears beside it.
    #[cfg(unix)]
    #[test]
    fn a_file_locked_itself_is_written_through_its_lock() {
        use std::io::{Seek as _, Write as _};
        let dir = tempfile::tempdir().unwrap();
        let key = dir.path().join("journal.key");
        let held = try_lock(Target::Itself(&key)).unwrap().unwrap();
        assert!(
            try_lock(Target::Itself(&key)).unwrap().is_none(),
            "one holder at a time"
        );
        let mut file = held.file();
        file.set_len(0).unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        file.write_all(b"secret").unwrap();
        assert_eq!(std::fs::read(&key).unwrap(), b"secret");
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(entries, ["journal.key"], "no lock file beside it");
        if let Some(seen) = held_for_another_process(Target::Itself(&key)) {
            assert!(seen, "held for another process");
        }
        drop(held);
        assert!(try_lock(Target::Itself(&key)).unwrap().is_some());
    }

    /// No moment exists when a lock is held but its file reads as not held: an acquire marks the lock
    /// file before it tries to lock it. A failed attempt takes its mark out again, so a key a
    /// competing acquire could not get is ordinary once its holder lets go.
    #[cfg(unix)]
    #[test]
    fn a_lock_file_reads_as_held_from_before_it_is_locked_until_after_it_is_released() {
        let dir = tempfile::tempdir().unwrap();
        let key = dir.path().join("journal.key");
        let seen = std::rc::Rc::new(std::cell::Cell::new(None));
        let saw = seen.clone();
        BETWEEN.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |path: &Path| {
                saw.set(Some(is_lock_file(path)));
            }));
        });
        let held = try_lock(Target::Itself(&key)).unwrap().unwrap();
        assert_eq!(seen.get(), Some(true), "held before the lock is taken");
        seen.set(None);
        assert!(
            try_lock(Target::Itself(&key)).unwrap().is_none(),
            "a competing acquire fails"
        );
        BETWEEN.with(|hook| *hook.borrow_mut() = None);
        assert_eq!(
            seen.get(),
            Some(true),
            "and was held throughout its attempt"
        );
        assert!(is_lock_file(&key));
        drop(held);
        assert!(
            !is_lock_file(&key),
            "the failed attempt took its mark out again"
        );
    }

    /// A lock file is recognized by its name in any case — a record lock file, a legacy one beside a
    /// record lock file, or an old binary's legacy one where a session or the MCP manifest keeps it —
    /// and nothing else is: an unrelated `Cargo.lock`, or a `lock` in a directory that holds no
    /// session, is just a file.
    #[test]
    fn a_lock_file_is_one_by_its_name_in_any_case_or_by_where_it_sits() {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("s1");
        std::fs::create_dir_all(&session).unwrap();
        drop(try_lock(Target::Dir(&session)).unwrap().unwrap());
        assert!(
            is_lock_file(&session.join("lock")),
            "beside a record lock file"
        );
        assert!(is_lock_file(&session.join(".BEYOND-LOCK")), "any case");
        // An old binary's session directory: a segment and its legacy `lock`, no record lock file.
        let old = dir.path().join("old");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("000001.jsonl"), "").unwrap();
        std::fs::write(old.join("lock"), "").unwrap();
        assert!(
            is_lock_file(&old.join("lock")),
            "a lock in a session directory"
        );
        assert!(is_lock_file(&old.join("Lock")), "in any case");
        for beside in [
            "1700000000_abc.jsonl.lock",
            "mcp-manifest.json.lock",
            ".memory.lock",
        ] {
            assert!(is_lock_file(&dir.path().join(beside)), "{beside}");
        }
        std::fs::write(dir.path().join("Cargo.lock"), "").unwrap();
        assert!(!is_lock_file(&dir.path().join("Cargo.lock")));
        let plain = dir.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("lock"), "").unwrap();
        assert!(!is_lock_file(&plain.join("lock")), "no session here");
    }

    /// A lock this process holds is recognized under any name — a hard link, a key locked itself —
    /// by its inode, and stops being one once released.
    #[cfg(unix)]
    #[test]
    fn a_held_lock_file_is_recognized_under_any_name() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("x");
        let target = Target::File(&file);
        let held = try_lock(target).unwrap().unwrap();
        let alias = dir.path().join("INNOCENT.TXT");
        std::fs::hard_link(target.record_path(), &alias).unwrap();
        assert!(is_lock_file(&alias));
        let key = dir.path().join("journal.key");
        let held_key = try_lock(Target::Itself(&key)).unwrap().unwrap();
        assert!(is_lock_file(&key), "a file locked itself is one while held");
        drop(held);
        drop(held_key);
        assert!(!is_lock_file(&alias), "released");
        assert!(!is_lock_file(&key), "released");
    }

    /// The legacy lock file's inode is in the held set too, not only the record lock file's: a hard
    /// link to it under an innocent name reads as a lock file while it is held, and as an ordinary file
    /// once released.
    #[cfg(unix)]
    #[test]
    fn a_hard_link_to_a_held_legacy_lock_file_is_not_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("s1");
        std::fs::create_dir_all(&session).unwrap();
        let target = Target::Dir(&session);
        let held = try_lock(target).unwrap().unwrap();
        let alias = dir.path().join("notes.txt");
        std::fs::hard_link(target.legacy_path().unwrap(), &alias).unwrap();
        assert!(
            crate::tools::write_atomic(alias.to_str().unwrap(), b"x").is_err(),
            "a link to the held legacy lock file is refused"
        );
        drop(held);
        crate::tools::write_atomic(alias.to_str().unwrap(), b"x").unwrap();
    }

    /// Asking whether a file is a lock file takes no lock on it, so another program's own
    /// non-blocking `flock` on that file never fails because of us. (A test lock taken to ask made
    /// 2071 of 20000 of its attempts fail.)
    #[cfg(unix)]
    #[test]
    fn asking_never_makes_another_programs_lock_fail() {
        let dir = tempfile::tempdir().unwrap();
        let theirs = dir.path().join("daemon.pid");
        std::fs::write(&theirs, "").unwrap();
        let Ok(mut other) = Command::new("python3")
            .args([
                "-c",
                "import fcntl,sys\nf=open(sys.argv[1],'a+')\nprint('ready',flush=True)\nfailed=0\nfor _ in range(20000):\n    try:\n        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n        fcntl.flock(f,fcntl.LOCK_UN)\n    except BlockingIOError:\n        failed+=1\nprint(failed,flush=True)",
            ])
            .arg(&theirs)
            .stdout(Stdio::piped())
            .spawn()
        else {
            eprintln!("no python3: skipping the other-program check");
            return;
        };
        let mut out = BufReader::new(other.stdout.take().unwrap());
        let mut line = String::new();
        out.read_line(&mut line).unwrap();
        assert_eq!(line.trim(), "ready");
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let asker = {
            let (stop, theirs) = (stop.clone(), theirs.clone());
            std::thread::spawn(move || {
                let (mut asked, mut called_lock) = (0u64, 0u64);
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    called_lock += u64::from(is_lock_file(&theirs));
                    asked += 1;
                }
                (asked, called_lock)
            })
        };
        line.clear();
        out.read_line(&mut line).unwrap();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let (asked, called_lock) = asker.join().unwrap();
        other.wait().unwrap();
        assert!(asked > 0);
        assert_eq!(
            line.trim(),
            "0",
            "its own lock attempts failed while we asked {asked} times"
        );
        assert_eq!(called_lock, 0, "its file is not a lock file of ours");
    }

    /// An old binary held only the legacy `flock`. While one runs, it must still exclude this binary.
    #[cfg(unix)]
    #[test]
    fn an_old_binary_holding_only_the_legacy_flock_excludes_this_one() {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("s1");
        std::fs::create_dir_all(&session).unwrap();
        let target = Target::Dir(&session);
        let Ok(mut old) = Command::new("python3")
            .args([
                "-c",
                "import fcntl,sys\nf=open(sys.argv[1],'a+')\nfcntl.flock(f,fcntl.LOCK_EX)\nprint('locked',flush=True)\nsys.stdin.read()",
            ])
            .arg(target.legacy_path().unwrap())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
        else {
            eprintln!("no python3: skipping the mixed-rollout check");
            return;
        };
        let mut line = String::new();
        BufReader::new(old.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert_eq!(line.trim(), "locked");
        let started = Instant::now();
        assert!(
            try_lock(target).unwrap().is_none(),
            "an old binary's flock still means held"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "and it says so within the bounded retry"
        );
        drop(old.stdin.take());
        old.wait().unwrap();
        assert!(
            try_lock(target).unwrap().is_some(),
            "free once the old binary lets go"
        );
    }
}
