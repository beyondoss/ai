//! Where a job's retained output lives: chunk files under a directory only execd can enter.
//!
//! A kept job's last 8 MiB of output would be anonymous memory, which cannot be reclaimed without
//! swap; as files in a disk-backed directory it is page cache, which can. The layout is
//! `<base>/execd-jobs-<uid>-<pid>-<random>/<job number>/<chunk index>`, where a chunk is the 64 KiB
//! at offset `index * 64 KiB` of the job's output.
//!
//! **Bounds.** On disk, at most the job's retention plus one chunk per job; each chunk file is
//! deleted when its bytes are evicted, a job's directory when the job is dropped, and the whole
//! directory when the server shuts down or is dropped. In memory, a job holds a path and at most
//! one unfinished chunk.
//!
//! **Safety.** The directories are created `0700` and the files `0600`, each with `create_new`
//! (`O_EXCL`) and `O_NOFOLLOW`, and a job's directory is named by a counter, never by anything a
//! client sends, so a command cannot plant a symlink that redirects a write nor name its way out.
//! Without `--run-as` a command runs as execd's own uid and can read what execd can; with it the
//! commands run as another uid and the `0700` directory is closed to them.
//!
//! **Stale directories.** At startup, a directory in the base that is named like this, is a real
//! directory (not a link), belongs to the same uid, and names a pid that no longer exists is
//! removed. Nothing else is touched.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io;
use std::os::unix::fs::{DirBuilderExt as _, FileExt as _, MetadataExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const PREFIX: &str = "execd-jobs-";

/// execd's directory for job output.
pub(crate) struct JobsDir {
    path: PathBuf,
    next: AtomicU64,
}

impl JobsDir {
    /// Create this server's directory under `base` (default `$TMPDIR`, else `/tmp`).
    pub fn open(base: Option<&Path>) -> io::Result<Self> {
        let base = base_dir(base);
        // The owner of /proc/self is the process's effective uid.
        let uid = fs::metadata("/proc/self")?.uid();
        let me = std::process::id();
        let mut rnd = [0u8; 6];
        getrandom::fill(&mut rnd).map_err(io::Error::other)?;
        let tag: String = rnd.iter().map(|b| format!("{b:02x}")).collect();
        let path = base.join(format!("{PREFIX}{uid}-{me}-{tag}"));
        DirBuilder::new().mode(0o700).create(&path)?;
        let dir = Self {
            path,
            next: AtomicU64::new(0),
        };
        dir.log_filesystem();
        Ok(dir)
    }

    /// The path a new job will use (nothing is created until it writes a full chunk).
    pub fn new_job(&self) -> PathBuf {
        self.path
            .join(self.next.fetch_add(1, Ordering::Relaxed).to_string())
    }

    /// Say what the directory is on, and warn where retained output would be RAM.
    #[allow(clippy::unnecessary_cast)] // `f_type` is signed on glibc and unsigned on musl
    fn log_filesystem(&self) {
        use nix::sys::statfs::{TMPFS_MAGIC, statfs};
        const RAMFS_MAGIC: u64 = 0x8584_58f6;
        let Ok(st) = statfs(&self.path) else { return };
        let kind = st.filesystem_type();
        if kind == TMPFS_MAGIC || (kind.0 as u64) & 0xffff_ffff == RAMFS_MAGIC {
            tracing::warn!(
                dir = %self.path.display(),
                "the jobs directory is on tmpfs, so retained job output (up to 8 MiB per job) \
                 is RAM here; point --jobs-dir (EXECD_JOBS_DIR) at a disk-backed directory"
            );
        } else {
            tracing::info!(dir = %self.path.display(), fs_type = format!("{:#x}", kind.0), "jobs directory");
        }
    }

    /// Remove the directory and everything under it.
    pub fn clean(&self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

impl Drop for JobsDir {
    fn drop(&mut self) {
        self.clean();
    }
}

fn base_dir(base: Option<&Path>) -> PathBuf {
    match base {
        Some(b) => b.to_path_buf(),
        None => std::env::var_os("TMPDIR")
            .filter(|v| !v.is_empty())
            .map_or_else(|| PathBuf::from("/tmp"), PathBuf::from),
    }
}

/// Remove the jobs directories of execds that are gone (see the module doc). A binary calls this
/// once, before it starts serving; it reads the whole of the base directory.
pub fn remove_stale_jobs_dirs(base: Option<&Path>) {
    if let Ok(meta) = fs::metadata("/proc/self") {
        remove_stale(&base_dir(base), meta.uid(), std::process::id());
    }
}

/// Remove the directories of execds that are gone: `execd-jobs-<our uid>-<pid>-<hex>`, a real
/// directory owned by us, whose pid is not running and is not ours.
fn remove_stale(base: &Path, uid: u32, me: u32) {
    let Ok(entries) = fs::read_dir(base) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let Some(rest) = name.to_str().and_then(|n| n.strip_prefix(PREFIX)) else {
            continue;
        };
        let mut parts = rest.splitn(3, '-');
        let (Some(u), Some(p), Some(_)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let (Ok(u), Ok(p)) = (u.parse::<u32>(), p.parse::<u32>()) else {
            continue;
        };
        if u != uid || p == me {
            continue;
        }
        let Ok(meta) = fs::symlink_metadata(e.path()) else {
            continue;
        };
        if !meta.is_dir() || meta.uid() != uid {
            continue;
        }
        // Gone means `kill(pid, 0)` says there is no such process; a process we may not signal
        // exists, and so does a pid that is alive for any other reason.
        let gone = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(p as i32), None),
            Err(nix::errno::Errno::ESRCH)
        );
        if gone {
            let _ = fs::remove_dir_all(e.path());
        }
    }
}

/// Create the directory `dir` (mode `0700`) unless it exists.
pub(crate) fn make_job_dir(dir: &Path) -> io::Result<()> {
    match DirBuilder::new().mode(0o700).create(dir) {
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        r => r,
    }
}

/// Write a new chunk file: `0600`, never through a link, never over an existing file.
pub(crate) fn write_chunk(path: &Path, data: &[u8]) -> io::Result<()> {
    use std::io::Write as _;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits())
        .open(path)?;
    f.write_all(data)
}

/// Read up to `buf.len()` bytes of a chunk file from `pos`.
pub(crate) fn read_chunk(path: &Path, pos: u64, buf: &mut [u8]) -> io::Result<usize> {
    let f: File = OpenOptions::new()
        .read(true)
        .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits())
        .open(path)?;
    f.read_at(buf, pos)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn dead_pid() -> u32 {
        let mut c = std::process::Command::new("true").spawn().unwrap();
        let pid = c.id();
        c.wait().unwrap();
        pid
    }

    #[test]
    fn stale_removal_takes_only_its_own_dead_directories() {
        let base = tempfile::tempdir().unwrap();
        let uid = fs::metadata("/proc/self").unwrap().uid();
        let me = std::process::id();
        let dead = dead_pid();
        let mk = |name: String| {
            let p = base.path().join(name);
            fs::create_dir(&p).unwrap();
            fs::write(p.join("f"), b"x").unwrap();
            p
        };
        let stale = mk(format!("{PREFIX}{uid}-{dead}-aa"));
        let alive = mk(format!("{PREFIX}{uid}-{me}-bb"));
        let other_uid = mk(format!("{PREFIX}{}-{dead}-cc", uid + 1));
        let other_name = mk(format!("not-{PREFIX}{uid}-{dead}-dd"));
        let malformed = mk(format!("{PREFIX}x-y"));
        // A link with the right name is not a directory of ours, and its target is never followed.
        let target = mk("target".into());
        let link = base.path().join(format!("{PREFIX}{uid}-{dead}-ee"));
        std::os::unix::fs::symlink(&target, &link).unwrap();
        remove_stale(base.path(), uid, me);
        assert!(!stale.exists());
        for kept in [&alive, &other_uid, &other_name, &malformed, &target] {
            assert!(kept.join("f").exists(), "{kept:?} was removed");
        }
        assert!(link.symlink_metadata().is_ok());
    }

    #[test]
    fn a_jobs_directory_is_private_and_goes_away_with_its_server() {
        use std::os::unix::fs::PermissionsExt as _;
        let base = tempfile::tempdir().unwrap();
        let dir = JobsDir::open(Some(base.path())).unwrap();
        let path = dir.path.clone();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        fs::write(path.join("x"), b"y").unwrap();
        drop(dir);
        assert!(!path.exists());
    }
}
