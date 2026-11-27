//! The jobs ledger: the one typed value execd keeps across a restart of its own process.
//!
//! A job's processes run on when execd is killed, crashes or is restarted; what dies is the table
//! that knew them. With [`crate::Options::state_dir`] set, the table's jobs are kept in two files, and
//! a restarted execd reads them back and asks [`crate::job_kernel::recover`] what each job becomes
//! (`jobs::recover`).
//!
//! **A file of the table, and a log of the changes since it.** `execd-jobs.json` is the table as one
//! file: `{"version":1,"jobs":[...]}`, written aside and renamed, so a reader sees the old file or the
//! new one, never half of one. `execd-jobs.log` is what has happened to it since: one framed record per
//! change of a job's life, each the job's whole row (or the fact that its row is gone), a length and a
//! checksum in front of it so a record cut short by a crash is dropped instead of believed. A life
//! change appends its record; nothing rewrites the table. The log is folded back into a new
//! `execd-jobs.json` when it outgrows [`LOG_MIN`] or the size of the table it sits on, and its own
//! header names the file it was written against by length and checksum, so a log of a compaction that
//! did not finish is dropped rather than applied over a file it does not describe.
//!
//! Why: the file used to be replaced whole on **every** change of a job's life, 1.16 MB written per
//! start→end→remove cycle at 1024 ended entries and 4.6 MB at the 4096 ceiling — 26/46/134 ms a cycle,
//! serialized, on a table that only grows. An append is one 350-byte record (`reports/CAPY-9.md`).
//!
//! Only what a restart cannot learn from the OS is kept: the job's address, its label, its mark (how
//! its processes are found again, wherever they went), the kernel's [`State`] (phase, claim, what
//! the model has seen) and, once it has them, the shell's status and the end's. Its output is not in it: the entry names the job's spool directory
//! (`Entry::spool`, see `tail.rs`), where the job's files, the journal and the ring's chunks are,
//! and a restarted execd rebuilds the ring from there.
//!
//! The files are `0600` in a `0700` directory. A file that does not parse, or names a version this
//! execd does not know, is read as empty and never trusted; so is a log whose records stop making
//! sense, from there on.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::job_kernel::{Claim, Persist, Phase, Size, State, Term};

/// The table, whole: the file execd wrote on every life change before this crate appended, and the one
/// a rollback to that execd still reads. Unchanged in shape, so an `execd-jobs.json` written by this
/// execd and one written by the execd before it are the same file.
const FILE: &str = "execd-jobs.json";
/// The changes since [`FILE`], one framed record each.
const LOG: &str = "execd-jobs.log";
const VERSION: u32 = 1;
/// [`LOG`]'s own format: its header and its framing.
const LOG_VERSION: u32 = 1;
/// The most a record may claim to be. A row of a full table is ~350 bytes and the longest argv is
/// bounded; a longer length is a corrupt one, and the record is not read.
const RECORD_MAX: u64 = 1 << 20;
/// The log is folded into a new [`FILE`] once the bytes appended to it pass this much — or the size the
/// table itself takes as a file, if that is more. So the two files together hold at most two tables, and
/// a small table never rewrites itself on every change. At the 4096-entry ceiling (1.5 MB) a fold is one
/// write every ~3800 life changes: 4 cases in 10 000 of what the whole-file write cost per change.
const LOG_MIN: u64 = 64 * 1024;

/// One saved job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Entry {
    pub scope: String,
    pub id: String,
    pub label: String,
    /// What the job runs, for `list`. Never its environment.
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
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
    /// The offset every byte of the job's output ends at, when the retention dropped the output and
    /// kept the entry (a tombstone, `jobs::sweep`): nothing is held, and a restart rebuilds the ring
    /// at this offset so a client's reads never go backwards. `None`: the output is the spool's, if
    /// any. Serde-default, so a file written before it still parses, and absent when `None`, so a
    /// ledger with no tombstone is byte for byte the file execd wrote before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<u64>,
    /// A terminal's: its size, its keeper's pid and whether it was open. The keeper's socket is in the spool.
    #[serde(default)]
    pub term: Option<SavedTerm>,
    /// The start's idempotency key and the digest of the spec it started with (`jobs::spec_digest`): the
    /// job's record in the key ledger. Never the spec itself, which can hold secrets in its env. A restarted
    /// execd puts it back (K6), so a start retried across the restart replays the job. `None`: an unkeyed start.
    #[serde(default)]
    pub idem: Option<Idem>,
}

/// What a restart needs of a terminal: the keeper to ask (its pid is what `SO_PEERCRED` must say) and
/// the size the job had.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SavedTerm {
    pub cols: u16,
    pub rows: u16,
    pub keeper_pid: i32,
    pub open: bool,
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

#[derive(Serialize)]
struct FileRef<'a> {
    version: u32,
    jobs: &'a [Entry],
}

#[derive(Deserialize)]
struct File {
    version: u32,
    jobs: Vec<Entry>,
}

/// One change of a job's life, as [`LOG`] records it: the job's row as it now is, or the fact that the
/// table no longer has it.
#[allow(clippy::large_enum_variant)] // one row, built to be written and dropped: a box would allocate for nothing
#[derive(Serialize, Deserialize)]
enum Record {
    Put(Entry),
    Del { scope: String, id: String },
}

/// [`LOG`]'s header: which [`FILE`] its records belong to. A file is named by what it is (its length and
/// its checksum), so a log written against a table that is no longer there — a compaction that renamed
/// its file and crashed before its log — is a log of nothing, and is dropped.
#[derive(Serialize, Deserialize)]
struct Head {
    version: u32,
    len: u64,
    crc: u32,
}

impl Entry {
    /// What to save of a job, or `None` for one not worth saving: no group yet, or removed.
    #[allow(clippy::too_many_arguments)] // one saved row, each field from a different owner
    pub fn of(
        (scope, id): (&str, &str),
        label: &str,
        argv: &crate::jobs::Argv,
        mark: &str,
        k: &State,
        spool: Option<&Path>,
        // Where the job's output ends, when its output was dropped and only its entry kept
        // (`crate::ring::Ring::evict_all`): a restart answers from there.
        out: Option<u64>,
        keeper_pid: Option<i32>,
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
            command: argv.command.clone(),
            args: argv.args.clone(),
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
            out,
            term: match k.term {
                Term::No => None,
                Term::Open { size, .. } => Some(SavedTerm {
                    cols: size.cols,
                    rows: size.rows,
                    keeper_pid: keeper_pid.unwrap_or(0),
                    open: true,
                }),
                Term::Gone { size } => Some(SavedTerm {
                    cols: size.cols,
                    rows: size.rows,
                    keeper_pid: keeper_pid.unwrap_or(0),
                    open: false,
                }),
            },
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
            term: match self.term {
                None => Term::No,
                Some(t) => {
                    let size = Size {
                        cols: t.cols,
                        rows: t.rows,
                    };
                    if t.open {
                        Term::Open {
                            size,
                            applied: true,
                        }
                    } else {
                        Term::Gone { size }
                    }
                }
            },
        }
    }
}

/// Where the ledger lives; `None` keeps nothing, as execd always did.
pub(crate) struct Ledger {
    dir: Option<PathBuf>,
    /// Append each change of a job's life and fold the log when it outgrows the table
    /// ([`crate::Options::ledger_append`]). `false` writes the whole file on every change, as execd did
    /// before this existed — the same file, and no log.
    append: bool,
    /// One writer at a time: a record is built from the table *and* written under this, so records for
    /// one job are appended in the order their states were read, and a fold replaces the file and the
    /// log as one step no append can land inside.
    state: std::sync::Mutex<Log>,
}

/// What the ledger knows of its two files.
struct Log {
    /// The log, opened for appending: `None` until the first append, and after a fold (the log is a new
    /// file then, and the old handle one that is gone).
    file: Option<fs::File>,
    /// What the log says: the rows of [`FILE`] plus every record applied over them, so the next change
    /// can append only what differs.
    base: HashMap<(String, String), Entry>,
    /// The bytes of records in the log.
    bytes: u64,
    /// The length and checksum of the [`FILE`] the log on disk names.
    len: u64,
    crc: u32,
    /// The bytes [`FILE`] takes: the log may grow to as much before it is folded.
    table: u64,
    /// The log on disk is not to be appended to until a fold rewrites it: a record was cut short, a
    /// write failed, or the log names a file that is not the one there.
    repair: bool,
}

impl Log {
    fn none() -> Self {
        Self {
            file: None,
            base: HashMap::new(),
            bytes: 0,
            len: 0,
            crc: 0,
            table: 0,
            repair: true,
        }
    }
}

impl Ledger {
    pub fn disabled() -> Self {
        Self {
            dir: None,
            append: true,
            state: std::sync::Mutex::new(Log::none()),
        }
    }

    /// The ledger in `dir`, made (`0700`) if need be. A directory that cannot be made disables it,
    /// said once: jobs are then not remembered across a restart.
    pub fn at(dir: &Path, append: bool) -> Self {
        match fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
        {
            Ok(()) => Self {
                dir: Some(dir.to_path_buf()),
                append,
                state: std::sync::Mutex::new(Log::none()),
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

    /// A change of a job's life: append the rows that differ from what the log says, and fold the log
    /// into a new file once it has outgrown the table. The table is read under the writer lock, so what
    /// is appended is the table as of the moment of the write, whatever the caller changed — the
    /// guarantee the whole-file write gave, which is why two writers (a life change, a retention sweep)
    /// cannot lose each other's fact.
    ///
    /// Best effort, as the whole-file write was: a write that fails is logged and the previous state
    /// stands, and the next write folds the log, so a record cut short is never appended past.
    pub fn save(&self, snapshot: impl FnOnce() -> Vec<Entry>) {
        let Some(dir) = &self.dir else { return };
        let mut st = crate::proc::locked(&self.state);
        if !self.append {
            let rows = snapshot();
            self.whole(dir, &mut st, false, &rows);
            return;
        }
        let rows = snapshot();
        // The rows the log does not have as they now are, and the ones it has that the table does not.
        let live: HashSet<(&str, &str)> = rows
            .iter()
            .map(|e| (e.scope.as_str(), e.id.as_str()))
            .collect();
        let mut buf: Vec<u8> = Vec::new();
        let mut framed = true;
        for e in &rows {
            if st.base.get(&(e.scope.clone(), e.id.clone())) != Some(e) {
                framed &= frame(&mut buf, &Record::Put(e.clone())).is_ok();
            }
        }
        for (scope, id) in st.base.keys() {
            if !live.contains(&(scope.as_str(), id.as_str())) {
                let del = Record::Del {
                    scope: scope.clone(),
                    id: id.clone(),
                };
                framed &= frame(&mut buf, &del).is_ok();
            }
        }
        drop(live);
        // A test's fold (`fault.rs`): the same write, taken down the fold's path whatever the log's size.
        let forced = crate::fault::at("ledger_fold").is_some();
        let full = !cfg!(feature = "mut_ledger_never_folds")
            && st.bytes + buf.len() as u64 > LOG_MIN.max(st.table);
        if !framed || forced || st.repair || full {
            self.whole(dir, &mut st, false, &rows);
            return;
        }
        if buf.is_empty() || cfg!(feature = "mut_ledger_append_skipped") {
            return;
        }
        if let Err(e) = append(dir, &mut st, &buf) {
            crate::log_warn!(
                "cannot append to the jobs ledger dir={} error={e}",
                dir.display()
            );
            st.repair = true;
            return;
        }
        st.bytes += buf.len() as u64;
        st.base = rows
            .into_iter()
            .map(|e| ((e.scope.clone(), e.id.clone()), e))
            .collect();
    }

    /// [`Self::save`] for a write that must leave the table whole in [`FILE`]: a retention sweep's (one
    /// pass, off the runtime, whenever it acted). The log is folded, so the file is the table and the log
    /// is empty — a whole table after every sweep, which is what an execd that knows nothing of the log,
    /// one this deployment rolled back to, reads. Best effort, as [`Self::save`].
    pub fn save_whole(&self, snapshot: impl FnOnce() -> Vec<Entry>) {
        let Some(dir) = &self.dir else { return };
        let mut st = crate::proc::locked(&self.state);
        let rows = snapshot();
        self.whole(dir, &mut st, false, &rows);
    }

    /// The server is going away and kills every group: the table is empty, and both files say so.
    pub fn forget(&self) {
        let Some(dir) = &self.dir else { return };
        let mut st = crate::proc::locked(&self.state);
        self.whole(dir, &mut st, false, &[]);
    }

    /// [`Self::save`] for a keyed start's claim (K10): the file is written, synced, renamed and its directory synced,
    /// and how that came out is returned, so the start can refuse when the claim is not on disk. The claim is a whole
    /// write, as it always was, so the file a rollback reads names the claim too. A ledger that is not enabled keeps
    /// nothing across a restart, so there is nothing to lose: `Durable`.
    pub fn claim(&self, snapshot: impl FnOnce() -> Vec<Entry>) -> Persist {
        let Some(dir) = &self.dir else {
            return Persist::Durable;
        };
        let mut st = crate::proc::locked(&self.state);
        let rows = snapshot();
        let written = match crate::fault::at("claim_save").as_deref() {
            Some("full") => Err(std::io::ErrorKind::StorageFull.into()),
            Some(_) => Err(std::io::Error::other("injected claim fault")),
            None => self.fold(dir, &mut st, true, &rows),
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

    /// The saved jobs: none if there is no file, or it is not one of ours. The rows of [`FILE`] with the
    /// records of [`LOG`] applied over them, in the order the table had them — which is the order `list`
    /// answers in, so a restart never reorders a client's answer.
    pub fn load(&self) -> Vec<Entry> {
        let Some(dir) = &self.dir else {
            return Vec::new();
        };
        let bytes = fs::read(dir.join(FILE)).unwrap_or_default();
        // A file that is absent or unreadable is not a table a log can sit on: the next write folds
        // instead. (A directory nothing has written yet takes its first change as a fold of a one-row
        // table, so from then on a log always sits on a file — which is what a rollback reads.)
        let mut unreadable = !dir.join(FILE).is_file();
        let rows = match serde_json::from_slice::<File>(&bytes) {
            Ok(f) if f.version == VERSION => f.jobs,
            _ if bytes.is_empty() => Vec::new(),
            _ => {
                crate::log_warn!("the jobs ledger is not readable; starting with none");
                unreadable = true;
                Vec::new()
            }
        };
        let mut st = Log {
            len: bytes.len() as u64,
            crc: crc32(&bytes),
            table: bytes.len() as u64,
            ..Log::none()
        };
        let mut rows = Rows::of(rows);
        // The records since it. Read even with `append` off: a log a previous run of this execd left
        // names changes this file does not have, and the first whole write puts them in it.
        if let Ok(raw) = fs::read(dir.join(LOG)) {
            st.bytes = replay(&raw, (st.len, st.crc), &mut rows);
        }
        // A log whose records stop holding together (a cut-off last one) is folded at the next write:
        // appending after the fragment would leave every later record unreadable behind it.
        st.repair = if cfg!(feature = "mut_ledger_appends_past_tail") {
            unreadable
        } else {
            rows.repair || unreadable
        };
        let rows = rows.entries();
        st.base = rows
            .iter()
            .map(|e| ((e.scope.clone(), e.id.clone()), e.clone()))
            .collect();
        *crate::proc::locked(&self.state) = st;
        rows
    }

    /// [`Self::fold`], best effort: a failure is logged and the previous state stands.
    fn whole(&self, dir: &Path, st: &mut Log, durable: bool, rows: &[Entry]) {
        if let Err(e) = self.fold(dir, st, durable, rows) {
            crate::log_warn!(
                "cannot save the jobs ledger dir={} error={e}",
                dir.display()
            );
            st.file = None;
            st.repair = true;
        }
    }

    /// Write the table whole, and fold the log into a new one that names it: [`FILE`] aside, synced,
    /// renamed over the old one; then [`LOG`] aside, renamed over the old one. What a crash can leave:
    ///
    /// - before [`FILE`]'s rename: the old file, and the old log, which names it — the state before;
    /// - between the two renames: the new file, and a log that names the old one, and is dropped — the
    ///   state after;
    /// - after both: the new file and an empty log naming it — the state after.
    ///
    /// So the pair is the old state or the new one, never a mix, and the log in force always names the
    /// file it was written against. The table's bytes are synced before the log's rename, so no crash
    /// leaves a log naming a file whose bytes are not on disk.
    fn fold(&self, dir: &Path, st: &mut Log, durable: bool, rows: &[Entry]) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(&FileRef {
            version: VERSION,
            jobs: rows,
        })
        .map_err(std::io::Error::other)?;
        let crc = crc32(&bytes);
        // Nothing to write: the file holds exactly these bytes and the log holds nothing, so the pair on
        // disk is the state that was just read — a restart that adopted nothing new, on a clean log. (Never
        // for a durable write: "the same bytes as last time" is not "on disk".)
        if !durable && !st.repair && st.bytes == 0 && st.len == bytes.len() as u64 && st.crc == crc
        {
            return Ok(());
        }
        // With a log to rewrite, the table's bytes are on disk before the log that names them; without
        // one (the whole-file write, `append` off) the write costs what it always did — no sync.
        replace(dir, FILE, &bytes, self.append || durable, durable)?;
        // A test's crash between the two files (`fault.rs`): the table is renamed and the process is
        // gone before its log is, which is the one window the pair is not written as one step.
        if crate::fault::at("ledger_fold_crash").as_deref() == Some("exit") {
            std::process::exit(88);
        }
        st.file = None;
        st.bytes = 0;
        st.len = bytes.len() as u64;
        st.crc = crc;
        st.table = st.len;
        st.repair = false;
        st.base = rows
            .iter()
            .map(|e| ((e.scope.clone(), e.id.clone()), e.clone()))
            .collect();
        if !self.append {
            return Ok(());
        }
        replace(dir, LOG, &header(st.len, crc)?, durable, durable)
    }
}

/// The rows of a ledger, in the order the table had them: [`Ledger::load`]'s output and, through it,
/// the order `list` answers in after a restart.
struct Rows {
    v: Vec<Entry>,
    /// A row a record deleted: kept in `v` (its place in the order taken by nothing) and dropped at the
    /// end, so deleting one row never shifts another.
    gone: Vec<bool>,
    at: HashMap<(String, String), usize>,
    /// A record could not be read, so the log is not trusted from there on.
    repair: bool,
}

impl Rows {
    fn of(v: Vec<Entry>) -> Self {
        let at = v
            .iter()
            .enumerate()
            .map(|(i, e)| ((e.scope.clone(), e.id.clone()), i))
            .collect();
        let gone = vec![false; v.len()];
        Self {
            v,
            gone,
            at,
            repair: false,
        }
    }

    fn put(&mut self, e: Entry) {
        let key = (e.scope.clone(), e.id.clone());
        match self.at.get(&key) {
            Some(&i) => self.v[i] = e,
            None => {
                self.at.insert(key, self.v.len());
                self.v.push(e);
                self.gone.push(false);
            }
        }
    }

    fn del(&mut self, scope: &str, id: &str) {
        if let Some(i) = self.at.remove(&(scope.to_string(), id.to_string())) {
            self.gone[i] = true;
        }
    }

    fn entries(self) -> Vec<Entry> {
        self.v
            .into_iter()
            .zip(self.gone)
            .filter_map(|(e, gone)| (!gone).then_some(e))
            .collect()
    }
}

/// Apply the log's records to `rows`, and return the bytes of the log after its header (what the fold's
/// limit counts). Everything from the first record that does not hold together — a length that runs
/// past the end, a checksum that does not match, bytes that are not a record — is left alone, and
/// `rows.repair` says so, so the next write folds instead of appending past it.
fn replay(raw: &[u8], (len, crc): (u64, u32), rows: &mut Rows) -> u64 {
    let Some(nl) = raw.iter().position(|&b| b == b'\n') else {
        // No header: a log that cannot be read at all (an empty file is one that was made and never
        // used, which is not a repair).
        rows.repair = !raw.is_empty();
        return raw.len() as u64;
    };
    match serde_json::from_slice::<Head>(&raw[..nl]) {
        Ok(h)
            if h.version == LOG_VERSION
                && (cfg!(feature = "mut_ledger_ignores_base")
                    || (h.len == len && h.crc == crc)) => {}
        _ => {
            // A log of another file: a fold renamed its file and died before its log, so its records
            // are in that file already and dropping them costs nothing.
            rows.repair = true;
            return (raw.len() - nl - 1) as u64;
        }
    }
    let mut i = nl + 1;
    while i < raw.len() {
        if raw.len() - i < 8 {
            rows.repair = true;
            break;
        }
        let n = u32_at(raw, i);
        let want = u32_at(raw, i + 4) as u32;
        let start = i + 8;
        if n > RECORD_MAX || n > (raw.len() - start) as u64 {
            rows.repair = true;
            break;
        }
        let body = &raw[start..start + n as usize];
        if crc32(body) != want && !cfg!(feature = "mut_ledger_trusts_checksum") {
            rows.repair = true;
            break;
        }
        match serde_json::from_slice::<Record>(body) {
            Ok(Record::Put(e)) => rows.put(e),
            Ok(Record::Del { scope, id }) => rows.del(&scope, &id),
            Err(_) => {
                rows.repair = true;
                break;
            }
        }
        i = start + n as usize;
    }
    (raw.len() - nl - 1) as u64
}

fn u32_at(b: &[u8], o: usize) -> u64 {
    u64::from(u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]))
}

/// One framed record: a length, a checksum, the record.
fn frame(buf: &mut Vec<u8>, r: &Record) -> Result<(), ()> {
    let body = serde_json::to_vec(r).map_err(|_| ())?;
    let len = u32::try_from(body.len()).map_err(|_| ())?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&crc32(&body).to_le_bytes());
    buf.extend_from_slice(&body);
    Ok(())
}

/// [`LOG`]'s header: the file its records belong to, named by length and checksum, and a newline so the
/// records read on from it.
fn header(len: u64, crc: u32) -> std::io::Result<Vec<u8>> {
    let mut head = serde_json::to_vec(&Head {
        version: LOG_VERSION,
        len,
        crc,
    })
    .map_err(std::io::Error::other)?;
    head.push(b'\n');
    Ok(head)
}

/// Open the log for appending (once) and append `buf`. A log file that is not there is not recreated: a
/// log with no header is one nothing can read, so the caller folds instead (`st.repair`).
fn append(dir: &Path, st: &mut Log, buf: &[u8]) -> std::io::Result<()> {
    if let Some(ms) = slow() {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    if st.file.is_none() {
        // A log that is not there is made, header first: a state directory nothing has written yet
        // (a first job, or one that ran with `append` off) gets its log at its first change.
        if !dir.join(LOG).exists() {
            replace(dir, LOG, &header(st.len, st.crc)?, false, false)?;
        }
        let opened = fs::OpenOptions::new()
            .append(true)
            .mode(0o600)
            .open(dir.join(LOG))?;
        st.file = Some(opened);
    }
    let Some(f) = st.file.as_mut() else {
        return Err(std::io::Error::other("the jobs log is not open"));
    };
    // A test's torn tail (`fault.rs`): this record's first `n` bytes reach the file and the process is
    // gone, as a crash in the middle of the write leaves it.
    if let Some(n) = crate::fault::at("ledger_append_crash").and_then(|v| v.parse::<usize>().ok()) {
        let _ = f.write_all(&buf[..n.min(buf.len())]);
        let _ = f.sync_all();
        std::process::exit(87);
    }
    f.write_all(buf)
}

/// A test's slow disk (`fault.rs`): every ledger write costs this long, on the thread that wrote it.
fn slow() -> Option<u64> {
    crate::fault::at("ledger_slow").and_then(|v| v.parse::<u64>().ok())
}

/// Replace `dir/name` with `bytes`: written aside, renamed over. `sync_file` syncs the bytes before the
/// rename and `sync_dir` the rename itself, so a crash of the host leaves the old file or a whole new
/// one; a failure at any step is the caller's to see (the temporary is removed).
fn replace(
    dir: &Path,
    name: &str,
    bytes: &[u8],
    sync_file: bool,
    sync_dir: bool,
) -> std::io::Result<()> {
    if let Some(ms) = slow() {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    let tmp = dir.join(format!("{name}.tmp"));
    let _ = fs::remove_file(&tmp);
    let put = || -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        f.write_all(bytes)?;
        // The bytes before the name: a crash leaves the old file or a whole new one, never a new name
        // over half a file.
        if sync_file {
            f.sync_all()?;
        }
        drop(f);
        fs::rename(&tmp, dir.join(name))?;
        if sync_dir {
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

/// CRC-32 (the checksum gzip uses): each record and each table file is named by it, so a record cut
/// short, or a log whose table is not the one there, is not mistaken for a whole one.
fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    };
    let mut c = u32::MAX;
    for &b in bytes {
        c = TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checksum is CRC-32: the vectors every implementation agrees on, so a change that keeps "some"
    /// checksum but not this one is caught.
    #[test]
    fn the_checksum_is_crc_32() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }
}
