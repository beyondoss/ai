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
//! the model has seen) and, once it has them, the shell's status and the end's. Output is not kept:
//! a job's retained output is the previous process's spool, which dies with it, so a recovered job
//! reads back empty.
//!
//! The file is `0600` in a `0700` directory. A file that does not parse, or names a version this
//! execd does not know, is set aside (read as empty) and never trusted.

use std::fs;
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::job_kernel::{Claim, Phase, State};

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
    /// `None`: a held start nobody kept.
    pub kept: Option<Kept>,
    pub seen: u64,
    pub started: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Kept {
    pub num: Option<u64>,
    pub kind: Option<String>,
}

/// The kernel's [`Phase`] the job had, past its spawn (a job with no group is never saved).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub(crate) enum Saved {
    Running {
        pgid: i32,
    },
    Exited {
        pgid: i32,
        code: i32,
        signal: Option<i32>,
    },
    Ended {
        pgid: i32,
        code: i32,
        signal: Option<i32>,
        at: u64,
        shell_told: bool,
    },
}

#[derive(Serialize, Deserialize)]
struct File {
    version: u32,
    jobs: Vec<Entry>,
}

impl Entry {
    /// What to save of a job, or `None` for one not worth saving: no group yet, or removed.
    pub fn of(scope: &str, id: &str, label: &str, mark: &str, k: &State) -> Option<Self> {
        if k.removed {
            return None;
        }
        let phase = match k.phase {
            Phase::Spawning => return None,
            Phase::Running { pgid } => Saved::Running { pgid },
            Phase::Exited { pgid, code, signal } => Saved::Exited { pgid, code, signal },
            Phase::Ended {
                pgid,
                code,
                signal,
                at,
                shell_told,
            } => Saved::Ended {
                pgid,
                code,
                signal,
                at,
                shell_told,
            },
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
            kept,
            seen: k.seen,
            started: k.started,
        })
    }

    /// The kernel state this entry saved.
    pub fn state(&self) -> State {
        let phase = match self.phase {
            Saved::Running { pgid } => Phase::Running { pgid },
            Saved::Exited { pgid, code, signal } => Phase::Exited { pgid, code, signal },
            Saved::Ended {
                pgid,
                code,
                signal,
                at,
                shell_told,
            } => Phase::Ended {
                pgid,
                code,
                signal,
                at,
                shell_told,
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
        if let Err(e) = write_atomic(dir, &file) {
            crate::log_warn!(
                "cannot save the jobs ledger dir={} error={e}",
                dir.display()
            );
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

fn write_atomic(dir: &Path, file: &File) -> std::io::Result<()> {
    let tmp = dir.join(format!("{FILE}.tmp"));
    let _ = fs::remove_file(&tmp);
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp)?;
    f.write_all(&serde_json::to_vec(file).map_err(std::io::Error::other)?)?;
    drop(f);
    fs::rename(&tmp, dir.join(FILE))
}
