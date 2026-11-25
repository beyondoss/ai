//! The jobs ledger: the one typed value execd keeps across a restart of its own process.
//!
//! A job's processes run on when execd is killed, crashes or is restarted; what dies is the table
//! that knew them. With [`crate::Options::state_dir`] set, the table's jobs are written here as one
//! file, `execd-jobs.json`, replaced whole (written aside and renamed, so a reader sees the old
//! file or the new, never half of one) on every change of a job's life: its spawn, its shell's exit,
//! its end, its keep and its removal. A restarted execd reads it back and asks
//! [`crate::job_kernel::recover`] what each job becomes (`jobs::recover`).
//!
//! Only what a restart cannot learn from the OS is kept: the job's address, its label, its mark (how
//! its processes are found again, wherever they went), the kernel's [`State`] (phase, claim, what
//! the model has seen) and, once it has them, the shell's status and the end's. Its output is not in it: the entry names the job's spool directory
//! (`Entry::spool`, see `tail.rs`), where the job's files, the journal and the ring's chunks are,
//! and a restarted execd rebuilds the ring from there.
//!
//! The file is `0600` in a `0700` directory. A file that does not parse, or names a version this
//! execd does not know, is set aside (read as empty) and never trusted.

use std::fs;
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::job_kernel::{Claim, Persist, Phase, State};

const FILE: &str = "execd-jobs.json";
const VERSION: u32 = 1;

/// One saved job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Entry {
    pub scope: String,
    pub id: String,
    pub label: String,
    pub mark: String,
    pub phase: Saved,
    pub pgid: i32,
    /// The shell's exit code, once it has one (`exited`, `ended`).
    pub code: Option<i32>,
    pub signal: Option<i32>,
    /// When it ended (`ended`).
    pub at: Option<u64>,
    /// The shell's exit was reported before the end (`ended`).
    pub told: Option<bool>,
    /// `None`: a held start nobody kept.
    pub kept: Option<Kept>,
    pub seen: u64,
    pub started: u64,
    /// The directory of the job's output (`tail.rs`): its two files, its journal and its ring's
    /// chunks. A restarted execd, and whoever clears stale directories at its start, keep what
    /// this names. `None` for a job with no spool.
    #[serde(default)]
    pub spool: Option<String>,
    /// The start's idempotency key and the digest of the spec it started with (`jobs::spec_digest`): the
    /// job's record in the key ledger. Never the spec itself, which can hold secrets in its env. A restarted
    /// execd puts it back (K6), so a start retried across the restart replays the job. `None`: an unkeyed start.
    #[serde(default)]
    pub idem: Option<Idem>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Idem {
    pub key: String,
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Kept {
    pub num: Option<u64>,
    pub kind: Option<String>,
}

/// How far the job's life had gone: the kernel's [`Phase`] past its spawn (a job with no group is never saved). The fields a
/// phase carries are flat on the entry, present as far as the phase has them: this is a record of the kernel's value, not a
/// second state machine; every decision about it is `job_kernel`'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Saved {
    /// A keyed start's claim, written before its spawn (`idem` is set, `pgid` is 0).
    Spawning,
    Running,
    Exited,
    Ended,
}

#[derive(Serialize, Deserialize)]
struct File {
    version: u32,
    jobs: Vec<Entry>,
}

impl Entry {
    /// What to save of a job, or `None` for one not worth saving: no group yet, or removed.
    pub fn of(
        scope: &str,
        id: &str,
        label: &str,
        mark: &str,
        k: &State,
        spool: Option<&Path>,
        idem: Option<&Idem>,
    ) -> Option<Self> {
        if k.removed {
            return None;
        }
        let (phase, pgid, code, signal, at, told) = match k.phase {
            // A keyed start's intent is saved before the spawn, so a crash before the group's save leaves a claim a
            // restart can answer (K9). An unkeyed start with no group is not saved (R4).
            Phase::Spawning if idem.is_some() => (Saved::Spawning, 0, None, None, None, None),
            Phase::Spawning => return None,
            Phase::Running { pgid } => (Saved::Running, pgid, None, None, None, None),
            Phase::Exited { pgid, code, signal } => {
                (Saved::Exited, pgid, Some(code), signal, None, None)
            }
            Phase::Ended {
                pgid,
                code,
                signal,
                at,
                shell_told,
            } => (
                Saved::Ended,
                pgid,
                Some(code),
                signal,
                Some(at),
                Some(shell_told),
            ),
        };
        let kept = match &k.claim {
            Claim::Held => None,
            Claim::Kept { num, kind } => Some(Kept {
                num: *num,
                kind: kind.clone(),
            }),
        };
        Some(Self {
            scope: scope.to_string(),
            id: id.to_string(),
            label: label.to_string(),
            mark: mark.to_string(),
            phase,
            pgid,
            code,
            signal,
            at,
            told,
            kept,
            seen: k.seen,
            started: k.started,
            spool: spool.map(|p| p.to_string_lossy().into_owned()),
            idem: idem.cloned(),
        })
    }

    /// The kernel state this entry saved.
    pub fn state(&self) -> State {
        let (pgid, code, signal) = (self.pgid, self.code.unwrap_or(0), self.signal);
        let phase = match self.phase {
            Saved::Spawning => Phase::Spawning,
            Saved::Running => Phase::Running { pgid },
            Saved::Exited => Phase::Exited { pgid, code, signal },
            Saved::Ended => Phase::Ended {
                pgid,
                code,
                signal,
                at: self.at.unwrap_or(0),
                shell_told: self.told.unwrap_or(false),
            },
        };
        let claim = match &self.kept {
            None => Claim::Held,
            Some(k) => Claim::Kept {
                num: k.num,
                kind: k.kind.clone(),
            },
        };
        State {
            phase,
            claim,
            seen: self.seen,
            removed: false,
            started: self.started,
        }
    }
}

/// Where the ledger lives; `None` keeps nothing, as execd always did.
pub(crate) struct Ledger {
    dir: Option<PathBuf>,
    /// One writer at a time, so the last file written is the last state snapshotted.
    write: std::sync::Mutex<()>,
}

impl Ledger {
    pub fn disabled() -> Self {
        Self {
            dir: None,
            write: std::sync::Mutex::new(()),
        }
    }

    /// The ledger in `dir`, made (`0700`) if need be. A directory that cannot be made disables it,
    /// said once: jobs are then not remembered across a restart.
    pub fn at(dir: &Path) -> Self {
        match fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
        {
            Ok(()) => Self {
                dir: Some(dir.to_path_buf()),
                write: std::sync::Mutex::new(()),
            },
            Err(e) => {
                crate::log_warn!(
                    "no state directory: jobs are not remembered across a restart dir={} error={e}",
                    dir.display()
                );
                Self::disabled()
            }
        }
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    /// Replace the file with `snapshot()`, taken under the writer lock. Best effort: a write that
    /// fails is logged, and the previous file stands.
    pub fn save(&self, snapshot: impl FnOnce() -> Vec<Entry>) {
        let Some(dir) = &self.dir else { return };
        let _one = crate::proc::locked(&self.write);
        let file = File {
            version: VERSION,
            jobs: snapshot(),
        };
        if let Err(e) = write_atomic(dir, &file, false) {
            crate::log_warn!(
                "cannot save the jobs ledger dir={} error={e}",
                dir.display()
            );
        }
    }

    /// [`Self::save`] for a keyed start's claim (K10): the file is written, synced, renamed and its directory synced,
    /// and how that came out is returned, so the start can refuse when the claim is not on disk. A ledger that is not
    /// enabled keeps nothing across a restart, so there is nothing to lose: `Durable`.
    pub fn claim(&self, snapshot: impl FnOnce() -> Vec<Entry>) -> Persist {
        let Some(dir) = &self.dir else {
            return Persist::Durable;
        };
        let _one = crate::proc::locked(&self.write);
        let file = File {
            version: VERSION,
            jobs: snapshot(),
        };
        let written = match crate::fault::at("claim_save").as_deref() {
            Some("full") => Err(std::io::ErrorKind::StorageFull.into()),
            Some(_) => Err(std::io::Error::other("injected claim fault")),
            None => write_atomic(dir, &file, true),
        };
        match written {
            Ok(()) => Persist::Durable,
            Err(e) => {
                crate::log_warn!(
                    "cannot save a start's claim dir={} error={e}",
                    dir.display()
                );
                if e.kind() == std::io::ErrorKind::StorageFull {
                    Persist::DiskFull
                } else {
                    Persist::Failed
                }
            }
        }
    }

    /// The saved jobs: none if there is no file, or it is not one of ours.
    pub fn load(&self) -> Vec<Entry> {
        let Some(dir) = &self.dir else {
            return Vec::new();
        };
        let Ok(bytes) = fs::read(dir.join(FILE)) else {
            return Vec::new();
        };
        match serde_json::from_slice::<File>(&bytes) {
            Ok(f) if f.version == VERSION => f.jobs,
            _ => {
                crate::log_warn!("the jobs ledger is not readable; starting with none");
                Vec::new()
            }
        }
    }
}

/// Replace the ledger file with `file`: write a temporary beside it, rename it over. With `durable`, the temporary is
/// synced before the rename and the directory after, so a crash of the host leaves the old file or the new one, and
/// a failure at any step is the caller's to see (the temporary is removed).
fn write_atomic(dir: &Path, file: &File, durable: bool) -> std::io::Result<()> {
    let tmp = dir.join(format!("{FILE}.tmp"));
    let _ = fs::remove_file(&tmp);
    let put = || -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        f.write_all(&serde_json::to_vec(file).map_err(std::io::Error::other)?)?;
        if durable {
            f.sync_all()?;
        }
        drop(f);
        fs::rename(&tmp, dir.join(FILE))?;
        if durable {
            fs::File::open(dir)?.sync_all()?;
        }
        Ok(())
    };
    let r = put();
    if r.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    r
}
