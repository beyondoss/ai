//! Processes: spawning into their own group, reaping every child ourselves, group liveness and kills
//! by syscall.
//!
//! Everything here is a syscall; nothing spawns `kill`, and nothing reads `/proc` except the one
//! scan that decides a group's emptiness on the slow tick (see [`group_gone`]).
//!
//! **Reaping.** execd reaps its children itself, like an init: on each `SIGCHLD` the reaper task
//! ([`start_reaper`]) calls `waitpid` until nothing is left to collect. tokio's own child
//! handling is never used, so nothing else reaps behind it: a command is spawned with std's
//! `Command` (std never waits for a child, not even on drop), its pipes are handed to tokio as
//! plain async descriptors, and its exit status reaches the one waiting for it through a oneshot
//! channel the reaper fills. Where execd is the subreaper (the binary) the reaper takes
//! `waitpid(-1)`, so an orphaned grandchild is collected the moment it exits and no zombie
//! outlives the instant between its death and the next pass; elsewhere (a host that embeds
//! `execd::serve` and owns other children) it collects only the pids execd spawned, so it never
//! steals another's exit status. Either way the cost is a few `waitpid` calls per pass, whatever
//! the host's process count, and no pass reads `/proc`.
//!
//! **Bounds.** [`CHILDREN`] holds one entry per child execd spawned and has not yet seen exit:
//! freed by the reaper when the child is collected. N concurrent cold callers means N entries,
//! plus the children themselves, which the OS bounds (`pids.max`, `RLIMIT_NPROC`). [`Groups`] is
//! the same idea per server, for the shutdown sweep.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::os::fd::AsFd;
use std::os::unix::process::ExitStatusExt as _;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Duration;

use nix::errno::Errno;
use nix::sys::signal::{Signal, killpg};
use nix::sys::wait::{Id, WaitPidFlag, WaitStatus, waitid, waitpid};
use nix::unistd::Pid;
use tokio::sync::{oneshot, watch};

use crate::job_kernel;
use crate::wire_kernel as wire;

/// The children execd spawned and has not yet collected, each with the channel its exit status
/// goes down.
static CHILDREN: Mutex<Option<HashMap<u32, oneshot::Sender<ExitStatus>>>> = Mutex::new(None);

/// Held shared across every spawn, from before the child exists until it is in [`CHILDREN`]; held
/// exclusively by the reaper for the whole of one pass. Spawns do not exclude each other; the pass
/// excludes them all, and lasts only as long as its `waitpid` calls.
///
/// **Why a child's status is never lost.** The reaper collects a pid only while it holds this lock
/// exclusively, and looks the pid up in [`CHILDREN`] before it lets go. A spawn holds the lock
/// shared from before it creates the child to after the insert. So at any pass, every spawn is
/// entirely before it (its child is in `CHILDREN`, and its status goes to its waiter) or entirely
/// after it (no such child exists yet, so the pass cannot collect it). The lookup also happens
/// inside the pass because a pid is reused as soon as it is collected: a later spawn may register
/// the same number, and must not be handed this status.
static SPAWNING: RwLock<()> = RwLock::new(());

/// The groups execd holds (J11): inserted by a spawn under [`SPAWNING`] shared, taken out by the
/// reaper in the pass that collected the group's last member (under [`SPAWNING`] exclusive). A
/// signal by number reads it under [`SPAWNING`] shared, so the group cannot be freed between the
/// check and the `kill`, and a freed number (any process may carry it now) is never signalled.
static OWNED_GROUPS: Mutex<Option<job_kernel::Held>> = Mutex::new(None);

fn held_groups() -> std::sync::MutexGuard<'static, Option<job_kernel::Held>> {
    let mut g = locked(&OWNED_GROUPS);
    g.get_or_insert_with(job_kernel::Held::new);
    g
}

/// The spawns execd has made, counted: each group's identity in [`OWNED_GROUPS`], so a number the
/// OS hands to a later spawn is not taken for the earlier one's.
static SPAWNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One spawn's process group: its number and which spawn it belongs to (J11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Pg {
    pub id: i32,
    pub sp: u64,
}

/// Whether this process is its descendants' subreaper, so every orphaned group member becomes its
/// own child (see [`group_gone`]). Set by [`become_subreaper`].
static SUBREAPER: AtomicBool = AtomicBool::new(false);

fn children() -> std::sync::MutexGuard<'static, Option<HashMap<u32, oneshot::Sender<ExitStatus>>>> {
    let mut g = locked(&CHILDREN);
    g.get_or_insert_with(HashMap::new);
    g
}

/// The signals whose dispositions [`release_signals`] takes out of a command's hands to execd's.
const RELEASED: [tokio::signal::unix::SignalKind; 7] = [
    tokio::signal::unix::SignalKind::hangup(),
    tokio::signal::unix::SignalKind::interrupt(),
    tokio::signal::unix::SignalKind::quit(),
    tokio::signal::unix::SignalKind::terminate(),
    tokio::signal::unix::SignalKind::user_defined1(),
    tokio::signal::unix::SignalKind::user_defined2(),
    tokio::signal::unix::SignalKind::alarm(),
];

static HELD: Mutex<Vec<tokio::signal::unix::Signal>> = Mutex::new(Vec::new());

/// Give every command the default disposition of the signals a user sends a job (`Ctrl-C` is
/// `SIGINT`). A process the launcher started with `cmd &` or `nohup` has `SIGINT`, `SIGQUIT` or
/// `SIGHUP` set to `SIG_IGN`, and an ignored disposition survives `exec`: every command and job
/// would inherit it, a shell could not even `trap` it, and a signal sent to the job would do nothing.
/// A disposition that is a *handler* is reset to the default by `exec`, so installing one here (the
/// safe way: tokio's signal driver) is what hands the children `SIG_DFL`. The handlers do nothing:
/// execd itself no longer dies of a stray `SIGHUP`/`SIGUSR1`/`SIGQUIT` (it stops on `SIGTERM` and
/// `SIGINT` where it waits for them, as before). Call inside a runtime; idempotent.
pub(crate) fn release_signals() {
    let mut held = locked(&HELD);
    if !held.is_empty() {
        return;
    }
    for kind in RELEASED {
        if let Ok(sig) = tokio::signal::unix::signal(kind) {
            held.push(sig);
        }
    }
}

/// A spawned command, leading a process group of its own.
pub(crate) struct Proc {
    pub stdin: Option<tokio::process::ChildStdin>,
    pub stdout: Option<tokio::process::ChildStdout>,
    pub stderr: Option<tokio::process::ChildStderr>,
    /// The group's id, which is also the leader's pid.
    pub pgid: i32,
    /// Which spawn it was.
    pub sp: u64,
    exit: oneshot::Receiver<ExitStatus>,
    /// What `wait` answered, kept so that it can answer again: a oneshot receiver must not be
    /// polled after it completes.
    done: Option<Result<ExitStatus, ()>>,
}

impl Proc {
    /// The group, as a signal names it.
    pub fn pg(&self) -> Pg {
        Pg {
            id: self.pgid,
            sp: self.sp,
        }
    }

    /// The command's exit status, once the reaper has collected it. Cancel-safe, and callable again
    /// after it has answered.
    pub async fn wait(&mut self) -> io::Result<ExitStatus> {
        let answer = match self.done {
            Some(a) => a,
            None => {
                let a = (&mut self.exit).await.map_err(|_| ());
                self.done = Some(a);
                a
            }
        };
        answer.map_err(|()| io::Error::other("the command's exit status was lost"))
    }
}

/// Spawn `cmd` as the leader of a new process group.
///
/// [`SPAWNING`] is held shared across the spawn, so the reaper cannot collect the child before it
/// is registered, and concurrent spawns do not wait for each other.
pub(crate) fn spawn(cmd: &mut std::process::Command) -> io::Result<Proc> {
    use std::os::unix::process::CommandExt as _;
    cmd.process_group(0);
    // Poison is recovered, as for every lock here: a panic elsewhere must not stop a spawn.
    let _spawning = SPAWNING
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut child = cmd.spawn()?;
    let pid = child.id();
    // A pid that is not an i32 cannot name a group; 0 would mean our own. Not registered, so
    // nothing else will reap it: kill and collect it here.
    let Ok(pgid) = i32::try_from(pid) else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(io::Error::other(format!("pid {pid} is out of range")));
    };
    let (tx, exit) = oneshot::channel();
    children().as_mut().map(|m| m.insert(pid, tx));
    let sp = SPAWNS.fetch_add(1, Ordering::Relaxed);
    if let Some(h) = held_groups().as_mut() {
        h.insert(pgid, sp);
    }
    let pipes = (|| -> io::Result<_> {
        Ok((
            child
                .stdin
                .take()
                .map(tokio::process::ChildStdin::from_std)
                .transpose()?,
            child
                .stdout
                .take()
                .map(tokio::process::ChildStdout::from_std)
                .transpose()?,
            child
                .stderr
                .take()
                .map(tokio::process::ChildStderr::from_std)
                .transpose()?,
        ))
    })();
    match pipes {
        Ok((stdin, stdout, stderr)) => Ok(Proc {
            stdin,
            stdout,
            stderr,
            pgid,
            sp,
            exit,
            done: None,
        }),
        Err(e) => {
            // Registered, so the reaper collects it once it is dead.
            let _ = child.kill();
            Err(e)
        }
    }
}

/// Take hold of a group a previous execd started (a restart's adoption), as a spawn of this one's:
/// a signal by number reaches it while it is held, and the reaper lets it go once it is gone (J11).
pub(crate) fn adopt_group(pgid: i32) -> Pg {
    let sp = SPAWNS.fetch_add(1, Ordering::Relaxed);
    if let Some(h) = held_groups().as_mut() {
        h.insert(pgid, sp);
    }
    Pg { id: pgid, sp }
}

/// SIGKILL every member of group `pgid`. A group that is already gone is not an error.
pub(crate) fn kill_group(pg: Pg) {
    let _no_reaping = no_reaping();
    signal_held_group(pg, Signal::SIGKILL);
}

/// Keep the reaper from collecting anything until the guard drops (a pass takes [`SPAWNING`]
/// exclusively): a group that is held now cannot be freed, and a pid found now cannot be reused.
fn no_reaping() -> std::sync::RwLockReadGuard<'static, ()> {
    SPAWNING
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `sig` to group `pgid` if execd still holds it (J11). The caller holds [`no_reaping`]. A group
/// that was freed is not ours any more: nothing is sent, and there is nothing of ours to send to.
fn signal_held_group(pg: Pg, sig: Signal) {
    let ours = held_groups()
        .as_ref()
        .is_some_and(|h| job_kernel::may_deliver(h, pg.id, pg.sp));
    if ours && pg.id > 1 {
        let _ = killpg(Pid::from_raw(pg.id), sig);
    }
}

/// Whether any process, zombie or not, is in group `pgid`: `kill(-pgid, 0)`.
pub(crate) fn group_exists(pgid: i32) -> bool {
    pgid > 1 && !matches!(killpg(Pid::from_raw(pgid), None), Err(Errno::ESRCH))
}

/// The live (non-zombie) members of group `pgid`, by reading `/proc`: the pids, for an operation
/// that reports them. A zombie never counts: the spec says a group of zombies is empty.
#[cfg(test)]
pub(crate) fn live_members(pgid: i32) -> Vec<u32> {
    if !group_exists(pgid) {
        return Vec::new();
    }
    live_members_scan(pgid)
}

fn is_live(state: char) -> bool {
    state != 'Z' && state != 'X'
}

/// [`live_members`] by reading every process in `/proc`.
fn live_members_scan(pgid: i32) -> Vec<u32> {
    scan(pgid, None, false)
}

/// The one `/proc` scan: every live process that belongs to the job — for `mark` its group's members
/// and everything carrying it, for `None` the group's members alone. `strays_only` keeps just the
/// marked processes the group signal does not reach.
///
/// One [`Scratch`] serves the whole scan, so the work here is one `openat` and one `read` per
/// process and nothing per process on the heap: a box with two processes allocates what a box with
/// two thousand does (`benches/proc_scan.py` counts both).
fn scan(pgid: i32, mark: Option<&str>, strays_only: bool) -> Vec<u32> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return out;
    };
    let me = std::process::id();
    let mut scratch = Scratch::new(mark);
    for e in dir.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if pid == me {
            continue;
        }
        // Membership is the kernel's (W3). The mark is read from `/proc` only for a process the
        // group does not already reach: for one it reaches the answer does not depend on it.
        if let Some((state, _, pg)) = scratch.stat(pid)
            && is_live(state)
            && wire::is_member(
                pgid,
                pg,
                mark.is_some() && !wire::group_reaches(pgid, pg) && scratch.has_mark(pid),
                strays_only,
            )
        {
            out.push(pid);
        }
    }
    out
}

/// The buffers one `/proc` scan reuses: the path it opened last, the mark it looks for, and the file
/// it read last. None of them is per process, so a scan costs the same handful of allocations
/// whatever the box's process table holds — where a `format!` and an `fs::read` per process made the
/// cycle's allocations grow with it (`benches/proc_scan.py --assert-flat` measures that they do not).
struct Scratch {
    path: String,
    /// `EXECD_JOB_MARK=<mark>`, built once per scan; empty for a scan that only reads `stat`.
    want: Vec<u8>,
    buf: Vec<u8>,
}

impl Scratch {
    fn new(mark: Option<&str>) -> Self {
        let mut want = Vec::with_capacity(mark.map_or(0, |m| JOB_MARK_ENV.len() + m.len() + 1));
        if let Some(mark) = mark {
            want.extend_from_slice(JOB_MARK_ENV.as_bytes());
            want.push(b'=');
            want.extend_from_slice(mark.as_bytes());
        }
        Self {
            path: String::with_capacity(32),
            want,
            // A process's `environ` is a few KiB: the first read grows the buffer, and every later
            // process's fits in it.
            buf: Vec::with_capacity(4 << 10),
        }
    }

    /// Read `/proc/<pid>/<file>` into [`Self::buf`], which is reused. `false` when it cannot be
    /// read: the process exited between the directory read and this open, or its `environ` belongs
    /// to another uid (execd reads it only as root).
    fn read(&mut self, pid: u32, file: &str) -> bool {
        use std::fmt::Write as _;
        use std::io::Read as _;
        self.path.clear();
        // Writing to a `String` cannot fail.
        let _ = write!(self.path, "/proc/{pid}/{file}");
        let Ok(mut f) = std::fs::File::open(&self.path) else {
            return false;
        };
        self.buf.clear();
        f.read_to_end(&mut self.buf).is_ok()
    }

    /// `(state, ppid, pgrp)` of `pid`, from `/proc/<pid>/stat`.
    fn stat(&mut self, pid: u32) -> Option<(char, i32, i32)> {
        // Bytes, not a `String`: the command name inside the parentheses is a raw `comm` and need
        // not be UTF-8. A name that failed to decode would otherwise vanish from the scan, and a
        // precise check would call a still-live group empty.
        self.read(pid, "stat")
            .then(|| parse_stat(&self.buf))
            .flatten()
    }

    /// Whether `pid`'s environment, as it was at exec, holds `EXECD_JOB_MARK=<mark>`.
    fn has_mark(&mut self, pid: u32) -> bool {
        self.read(pid, "environ")
            && self
                .buf
                .split(|&b| b == 0)
                .any(|kv| kv == self.want.as_slice())
    }
}

/// Whether group `pgid` has no live member, as cheaply as is right.
///
/// `kill(-pgid, 0)` answers a gone group at once. A group that exists is alive, unless every
/// member is a zombie. Where execd is the subreaper it collects every zombie at once, so a zombie
/// can outlast an instant only while its parent is alive; if that parent is in the group the
/// group is alive, and the rare case left is a parent outside it. That case, and any host without
/// the subreaper (whose orphans go to init, and may sit as zombies there for good), is decided by
/// reading `/proc`, which `precise` asks for: it is asked on the slow tick, never per `SIGCHLD`.
pub(crate) async fn group_gone(pgid: i32, precise: bool) -> bool {
    if !group_exists(pgid) {
        return true;
    }
    if SUBREAPER.load(Ordering::Relaxed) && !precise {
        return false;
    }
    off_runtime(move || live_members_scan(pgid).is_empty()).await
}

/// Run `f`, a blocking `/proc` scan, on a thread of its own so the runtime is not held up. Where
/// the process cannot make a thread (its process limit is reached), `f` runs here instead: a few
/// milliseconds of stalled runtime, where `tokio::task::spawn_blocking` would panic the request.
/// `T::default()` if the scan itself panics.
pub(crate) async fn off_runtime<T: Default + Send + 'static>(
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let slot = std::sync::Arc::new(Mutex::new(Some(f)));
    let (tx, rx) = oneshot::channel();
    let theirs = slot.clone();
    let spawned = std::thread::Builder::new()
        .name("execd-scan".into())
        .stack_size(256 << 10)
        .spawn(move || {
            let f = locked(&theirs).take();
            if let Some(f) = f {
                let _ = tx.send(f());
            }
        });
    if spawned.is_ok() {
        return rx.await.unwrap_or_default();
    }
    let f = locked(&slot).take();
    f.map_or_else(T::default, |f| f())
}

/// The variable every job's processes carry (a job's `mark`, unique to it), so that what leaves
/// the job's process group still belongs to the job: `timeout` and `setsid` put themselves and
/// their child in groups and sessions of their own, and a daemon double-forks away. The
/// environment is inherited by all of them. What clears it (`env -i`, a wrapper that sanitizes)
/// or is another uid's (`/proc/<pid>/environ` is unreadable to execd unless it is root) is not
/// found.
pub(crate) const JOB_MARK_ENV: &str = "EXECD_JOB_MARK";

/// A fresh mark: unguessable enough that one job cannot name another's by accident.
pub(crate) fn new_job_mark(seq: u64) -> String {
    use std::hash::{BuildHasher as _, Hasher as _};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(seq);
    format!("{:016x}{seq:x}", h.finish())
}

/// The live (non-zombie) processes of a job: its process group's members and everything that
/// carries its mark, wherever it has gone. By reading `/proc`.
pub(crate) fn job_members(pgid: i32, mark: &str) -> Vec<u32> {
    members(pgid, mark, false)
}

/// The scan behind [`job_members`]; `strays_only` keeps just the marked processes that have left
/// the job's group.
fn members(pgid: i32, mark: &str, strays_only: bool) -> Vec<u32> {
    scan(pgid, Some(mark), strays_only)
}

/// [`job_members`] without holding up the runtime for the `/proc` scan.
pub(crate) async fn job_members_async(pgid: i32, mark: &str) -> Vec<u32> {
    let mark = mark.to_string();
    off_runtime(move || job_members(pgid, &mark)).await
}

/// Whether a job has no live process, for a caller that is already off the runtime. Where execd
/// is the subreaper a group that exists is alive (a zombie is collected at once).
pub(crate) fn job_empty(pgid: i32, mark: &str) -> bool {
    if group_exists(pgid) && SUBREAPER.load(Ordering::Relaxed) {
        return false;
    }
    job_members(pgid, mark).is_empty()
}

/// Whether execd has no child at all, as the kernel answers it: `waitid` with `WNOWAIT` reports a
/// zombie without collecting it (a `waitpid` with `WNOWAIT` is `EINVAL`), so nothing is taken out of
/// the reaper's hands, and `ECHILD` — no child to wait for — is the answer this asks for. A child
/// that is merely running, or stopped, is not `ECHILD`.
fn no_children() -> bool {
    matches!(
        waitid(
            Id::All,
            WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT
        ),
        Err(Errno::ECHILD)
    )
}

/// Whether the job's group's own number already says it is alive. `kill(-pgid, 0)` finds a group
/// that exists, zombies included, and where execd is the subreaper every zombie is collected at
/// once: so a group that exists is alive until its last member is gone. `precise` asks for the slow
/// tick, whose `/proc` read settles the one case left (a zombie whose live parent is outside the
/// group).
fn group_alive(pgid: i32, precise: bool) -> bool {
    group_exists(pgid) && SUBREAPER.load(Ordering::Relaxed) && !precise
}

/// Whether a job has no live process, as cheaply as is right: [`group_alive`], plus whatever left
/// the group with the job's mark (found by a scan, which a group that is already empty still needs).
///
/// This is the answer for a job this server did **not** spawn — a restart's adoption — whose
/// processes are not its descendants. A job this server spawned asks [`job_gone_owned`] instead.
pub(crate) async fn job_gone(pgid: i32, mark: &str, precise: bool) -> bool {
    if group_alive(pgid, precise) {
        return false;
    }
    job_gone_by_scan(pgid, mark).await
}

/// [`job_gone`] for a job this server spawned: its processes are this process's descendants, so a
/// child list the kernel reports empty answers the common end without reading `/proc` — one
/// `waitid` instead of an `openat` and a `read` per process on the box, which was the whole cost of
/// a job-life cycle on a busy sandbox (`benches/proc_scan.py`).
///
/// That is the whole argument, and it needs all three of its parts:
///
/// 1. every process carrying the job's mark inherited it from the child this server spawned, so it
///    descends from execd;
/// 2. a live process's ancestors are live, so the topmost of that chain is a live child of execd —
///    or one of them died, and the survivors were adopted by the nearest subreaper ancestor, which
///    is execd again (it is the subreaper for everything it spawned);
/// 3. so `ECHILD` — no child of execd at all, a zombie included — means nothing of this job (or of
///    any other) is running, and the group's own number has already said its group is gone.
///
/// The converse is what keeps this a fast path and not a replacement: another job's child, a
/// terminal's keeper, or a process execd adopted keeps a child list non-empty, and then the answer
/// is the scan's, as before. A job a restart adopted is not this process's descendant at all, so
/// it never takes this path (`jobs::supervise_adopted` calls [`job_gone`]).
///
pub(crate) async fn job_gone_owned(pgid: i32, mark: &str, precise: bool) -> bool {
    if group_alive(pgid, precise) {
        return false;
    }
    if !cfg!(feature = "mut_end_scan_always") && SUBREAPER.load(Ordering::Relaxed) && no_children()
    {
        crate::log_debug!("a job has no process left: execd has no child at all pgid={pgid}");
        return true;
    }
    job_gone_by_scan(pgid, mark).await
}

/// The scan's answer, off the runtime: no live process of the job is left.
async fn job_gone_by_scan(pgid: i32, mark: &str) -> bool {
    if cfg!(feature = "mut_end_group_only") {
        return true;
    }
    let mark = mark.to_string();
    off_runtime(move || job_members(pgid, &mark).is_empty()).await
}

/// `sig` to the job's group, and to each process that carries its mark and has left the group:
/// every process of the job hears it exactly once (a `SIGINT` handler counts deliveries).
pub(crate) fn signal_job(pg: Pg, mark: &str, sig: Signal) {
    let _no_reaping = no_reaping();
    signal_held_group(pg, sig);
    for pid in members(pg.id, mark, true) {
        if let Ok(raw) = i32::try_from(pid) {
            let _ = nix::sys::signal::kill(Pid::from_raw(raw), sig);
        }
    }
}

/// `(state, ppid, pgrp)` of `pid`, from `/proc/<pid>/stat`, for the tests below.
#[cfg(test)]
fn stat(pid: u32) -> Option<(char, i32, i32)> {
    Scratch::new(None).stat(pid)
}

/// The fields after the last `)` of a `/proc/<pid>/stat` line. The command name may itself hold
/// spaces, parentheses, and bytes that are not UTF-8, so the split is on that last parenthesis.
fn parse_stat(text: &[u8]) -> Option<(char, i32, i32)> {
    let close = text.iter().rposition(|&b| b == b')')?;
    let mut fields = text[close + 1..]
        .split(|b| b.is_ascii_whitespace())
        .filter(|f| !f.is_empty());
    let state = *fields.next()?.first()?;
    if !state.is_ascii() {
        return None;
    }
    let ppid = std::str::from_utf8(fields.next()?).ok()?.parse().ok()?;
    let pgrp = std::str::from_utf8(fields.next()?).ok()?.parse().ok()?;
    Some((state as char, ppid, pgrp))
}

/// How many bytes a pipe holds before its writer blocks: the pipe's own capacity
/// (`F_GETPIPE_SZ`, 64 KiB unless the command raised it), read from the kernel.
pub(crate) fn pipe_capacity<F: AsFd>(pipe: &F) -> usize {
    use nix::fcntl::{FcntlArg, fcntl};
    fcntl(pipe, FcntlArg::F_GETPIPE_SZ)
        .ok()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(64 * 1024)
}

/// Lock `m`, whatever a panicking holder left: the data these locks guard stays valid after a
/// panic (plain inserts and removes), and a poisoned lock must never stop a kill or a shutdown.
pub(crate) fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The process groups a server has running, so a shutdown can decide what to do with each.
///
/// Bounded by the number of running commands and jobs; each entry is freed when its command or
/// job ends ([`GroupSlot`] drops, or the job's supervisor finishes). Once [`Self::shutdown`] has
/// run the set is closed: a group registered after it (a spawn that raced the shutdown) gets the
/// same decision at once instead of escaping it.
pub(crate) struct Groups {
    set: Mutex<GroupSet>,
    /// Whether a ledger keeps the jobs across this server ([`crate::Options::state_dir`]): the
    /// kernel's `recorded`.
    recorded: bool,
}

#[derive(Default)]
struct GroupSet {
    live: BTreeMap<Pg, job_kernel::Kind>,
    /// Set once the shutdown has begun: why.
    closed: Option<job_kernel::Stop>,
}

impl Groups {
    pub fn new(recorded: bool) -> Self {
        Self {
            set: Mutex::default(),
            recorded,
        }
    }

    /// What a shutdown for `stop` does to a group of this kind (S3 to S6).
    pub fn at_shutdown(
        &self,
        kind: job_kernel::Kind,
        stop: job_kernel::Stop,
    ) -> job_kernel::Shutdown {
        job_kernel::shutdown_of(kind, self.recorded, stop)
    }

    pub fn insert(&self, pg: Pg, kind: job_kernel::Kind) {
        let mut g = locked(&self.set);
        if let Some(stop) = g.closed {
            drop(g);
            if self.at_shutdown(kind, stop).signal {
                kill_group(pg);
            }
        } else {
            g.live.insert(pg, kind);
        }
    }

    pub fn remove(&self, pg: Pg) {
        locked(&self.set).live.remove(&pg);
    }

    /// Decide each registered group's fate ([`job_kernel::shutdown_of`]): SIGKILL the ones it says
    /// to, leave the rest running (a recorded job: the next execd adopts it), forget them all, and
    /// close the set. How many it killed.
    pub fn shutdown(&self, stop: job_kernel::Stop) -> usize {
        let all = {
            let mut g = locked(&self.set);
            g.closed = Some(stop);
            std::mem::take(&mut g.live)
        };
        let mut killed = 0;
        for (&pg, &kind) in &all {
            if self.at_shutdown(kind, stop).signal {
                kill_group(pg);
                killed += 1;
            }
        }
        killed
    }
}

/// Make this process the subreaper of its descendants: a process that loses its parent is adopted
/// by execd rather than by the host's init, and the reaper collects it. Meant for the binary; an
/// embedding test harness has no business changing its process's reaping.
pub fn become_subreaper() -> io::Result<()> {
    nix::sys::prctl::set_child_subreaper(true).map_err(io::Error::from)?;
    SUBREAPER.store(true, Ordering::Relaxed);
    Ok(())
}

/// Tells waiters that a pass collected something. A counter, so a waiter that subscribed before the
/// pass sees it whenever it next looks.
fn reaped() -> &'static watch::Sender<u64> {
    static TX: OnceLock<watch::Sender<u64>> = OnceLock::new();
    TX.get_or_init(|| watch::channel(0).0)
}

/// One pass of the reaper: collect what has exited and hand each status to its waiter. How many
/// children it collected.
fn sweep() -> usize {
    let mut delivered = Vec::new();
    let mut n = 0;
    {
        let _exclusive = SPAWNING
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut table = children();
        let table = table.as_mut();
        let Some(table) = table else { return 0 };
        let owned: Vec<u32> = table.keys().copied().collect();
        let mut take = |status: WaitStatus| {
            let (pid, st) = match status {
                WaitStatus::Exited(pid, code) => (pid, ExitStatus::from_raw((code & 0xff) << 8)),
                WaitStatus::Signaled(pid, sig, core) => (
                    pid,
                    ExitStatus::from_raw(sig as i32 | if core { 0x80 } else { 0 }),
                ),
                _ => return false,
            };
            n += 1;
            if let Some(tx) = u32::try_from(pid.as_raw())
                .ok()
                .and_then(|p| table.remove(&p))
            {
                delivered.push((tx, st));
            }
            true
        };
        if SUBREAPER.load(Ordering::Relaxed) {
            // Every child, execd's commands and the orphans it adopted alike.
            loop {
                match waitpid(None, Some(WaitPidFlag::WNOHANG)) {
                    Ok(WaitStatus::StillAlive) | Err(Errno::ECHILD) => break,
                    Err(Errno::EINTR) => {}
                    Ok(st) => {
                        take(st);
                    }
                    Err(_) => break,
                }
            }
        } else {
            // Only what execd spawned: the host may own other children.
            for pid in owned {
                // A pid that is not an i32 cannot be one of ours; 0 would wait on any child.
                let Ok(raw) = i32::try_from(pid) else {
                    continue;
                };
                if let Ok(st) = waitpid(Pid::from_raw(raw), Some(WaitPidFlag::WNOHANG)) {
                    take(st);
                }
            }
        }
        if n > 0 {
            // A group's number is free once its last member is collected: stop holding every group
            // that is gone, before the exclusion ends, so no signal can name it afterwards (J11).
            if let Some(h) = held_groups().as_mut() {
                for g in h.numbers() {
                    if !group_exists(g) {
                        h.freed(g);
                    }
                }
            }
        }
    }
    for (tx, st) in delivered {
        let _ = tx.send(st);
    }
    if n > 0 {
        reaped().send_modify(|v| *v += 1);
    }
    n
}

/// Start the reaper: one pass now and one after every `SIGCHLD`, until `stop`. The signal is
/// registered before this returns, so no exit that follows it is missed; signals that arrive during
/// a pass coalesce into the next one.
pub(crate) fn start_reaper(
    stop: tokio_util::sync::CancellationToken,
) -> io::Result<tokio::task::JoinHandle<()>> {
    let mut sigchld = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::child())?;
    Ok(tokio::spawn(async move {
        loop {
            sweep();
            tokio::select! {
                _ = sigchld.recv() => {}
                () = stop.cancelled() => return,
            }
        }
    }))
}

/// Why a waiter woke.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wake {
    /// The reaper collected something.
    Exit,
    /// The slow tick: look even though nothing was collected.
    Tick,
}

/// Wake-ups for a task that waits for a process group to empty.
///
/// When the last member of a group exits, execd collects it (it is its child, or its adopted
/// orphan), and the reaper says so; a waiter sleeps until then and re-checks, with no timer running.
/// The case no collection covers is a last member that is a zombie whose live parent is neither
/// execd nor in the group; for that there is the slow tick, [`GROUP_FALLBACK`], at which the
/// waiter reads `/proc`. A host without the subreaper gets the tick every 50 ms, because its
/// orphans go to init and nothing tells execd that they exited. Create one before the first check,
/// so an exit that races the check is kept.
pub(crate) struct ChildEvents(watch::Receiver<u64>);

/// How long a waiter sleeps without a collection before it looks anyway: as long as a heartbeat,
/// the delay the protocol already tolerates for a silent stream, and rare enough to cost nothing.
pub(crate) const GROUP_FALLBACK: Duration = Duration::from_secs(5);

impl ChildEvents {
    pub fn new() -> Self {
        Self(reaped().subscribe())
    }

    pub async fn next(&mut self) -> Wake {
        if !SUBREAPER.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(50)).await;
            return Wake::Tick;
        }
        tokio::select! {
            r = self.0.changed() => if r.is_ok() { Wake::Exit } else { Wake::Tick },
            () = tokio::time::sleep(GROUP_FALLBACK) => Wake::Tick,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    /// One test forks at a time: a script written for another test's exec is still open for writing
    /// in a forked child until it execs (`ETXTBSY`).
    static FORKS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn forking() -> std::sync::MutexGuard<'static, ()> {
        FORKS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    use super::*;
    use std::os::unix::process::CommandExt as _;

    /// A zombie is in the group for `kill(-pgid, 0)` and not for the spec: its group is empty.
    #[test]
    fn a_zombie_is_not_a_live_member() {
        let _forking = forking();
        let mut leader = std::process::Command::new("true");
        leader.process_group(0);
        let mut child = leader.spawn().unwrap();
        let pgid = child.id() as i32;
        // Exited, not yet waited for: a zombie.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while stat(child.id()).map(|(s, _, _)| s) != Some('Z') {
            assert!(
                std::time::Instant::now() < deadline,
                "never became a zombie"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            group_exists(pgid),
            "the zombie still holds the group for kill(2)"
        );
        assert!(live_members(pgid).is_empty());
        child.wait().unwrap();
        assert!(!group_exists(pgid));
    }

    #[test]
    fn a_live_process_is_a_member_and_a_kill_removes_it() {
        let _forking = forking();
        let mut sleeper = std::process::Command::new("sleep");
        sleeper.arg("30").process_group(0);
        let mut child = sleeper.spawn().unwrap();
        let pgid = child.id() as i32;
        // Not through `spawn`, so held by hand, as a spawn would.
        let pg = Pg { id: pgid, sp: 0 };
        if let Some(h) = held_groups().as_mut() {
            h.insert(pgid, 0);
        }
        assert_eq!(live_members(pgid), vec![child.id()]);
        kill_group(pg);
        child.wait().unwrap();
        assert!(live_members(pgid).is_empty());
        // A group that is gone is not an error, and nothing outside a real group is ever signalled.
        // The reaper has not run in this test, so the number is still held: free it as a pass does.
        if let Some(h) = held_groups().as_mut() {
            h.freed(pgid);
        }
        kill_group(pg);
        for id in [0, 1] {
            kill_group(Pg { id, sp: 0 });
        }
    }

    /// J11 on the real reaper: a group is held from its spawn until the pass that collects its last
    /// member, and a signal by number after that pass reaches nothing (its number is any process's).
    #[test]
    fn a_group_is_held_until_the_reaper_collects_it_and_never_signalled_after() {
        let _forking = forking();
        let held_now = |pg: Pg| {
            // Under the exclusion that keeps the reaper out, as a signal checks it.
            let _no_reaping = no_reaping();
            held_groups()
                .as_ref()
                .is_some_and(|h| job_kernel::may_deliver(h, pg.id, pg.sp))
        };
        let mut cmd = std::process::Command::new("sleep");
        cmd.arg("30");
        let proc = spawn(&mut cmd).unwrap();
        let pgid = proc.pgid;
        let pg = proc.pg();
        assert!(held_now(pg), "a spawned group is held");
        kill_group(pg);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while held_now(pg) {
            sweep();
            assert!(
                std::time::Instant::now() < deadline,
                "the group was never freed"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!group_exists(pgid), "freed means the number is free");
        // A signal now names a number that is not ours: refused, whatever the number holds.
        signal_job(pg, "no-such-mark", Signal::SIGKILL);
        kill_group(pg);
        // The number taken by a later spawn is not the earlier one's group: its signal is refused.
        if let Some(h) = held_groups().as_mut() {
            h.insert(pgid, pg.sp + 1_000_000);
        }
        assert!(!held_now(pg));
        if let Some(h) = held_groups().as_mut() {
            h.freed(pgid);
        }
    }

    #[test]
    fn the_pipe_capacity_is_the_kernels_number() {
        let _forking = forking();
        let mut c = std::process::Command::new("true");
        c.stdout(std::process::Stdio::piped());
        let mut child = c.spawn().unwrap();
        let out = child.stdout.take().unwrap();
        // Whatever the kernel gave this pipe (64 KiB, or one page once the user's pipes exceed
        // `pipe-user-pages-soft`), never a constant.
        let given = nix::fcntl::fcntl(&out, nix::fcntl::FcntlArg::F_GETPIPE_SZ).unwrap();
        assert_eq!(pipe_capacity(&out), usize::try_from(given).unwrap());
        // And after the size is changed, the new number.
        if nix::fcntl::fcntl(&out, nix::fcntl::FcntlArg::F_SETPIPE_SZ(128 * 1024)).is_ok() {
            assert_eq!(pipe_capacity(&out), 128 * 1024);
        }
        child.wait().unwrap();
    }

    #[test]
    fn a_comm_that_is_not_text_still_parses() {
        // `pid (comm) state ppid pgrp`. The name holds a raw byte, a space, and a parenthesis;
        // the split is the last `)`.
        let mut text = b"12 (".to_vec();
        text.push(0xff);
        text.extend_from_slice(b" a)b) R 7 99 1");
        assert!(std::str::from_utf8(&text).is_err());
        assert_eq!(parse_stat(&text), Some(('R', 7, 99)));
        assert_eq!(parse_stat(b"no parenthesis here"), None);
    }

    /// A live member whose `comm` is a raw byte still shows up in the scan. A parent cannot write
    /// another process's `comm`, so the byte is the executable's own name.
    #[test]
    fn a_comm_that_is_not_text_is_still_a_live_member() {
        let _forking = forking();
        use std::os::unix::ffi::OsStringExt as _;
        use std::os::unix::fs::PermissionsExt as _;
        // A script named by the raw byte. The kernel takes `comm` from the file name, and the shell
        // stays alive as the process: it runs `sleep` as a child rather than in its place. Copying
        // `/bin/sleep` would not do: a multi-call binary (busybox, uutils) picks its applet from
        // this name and exits.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(std::ffi::OsString::from_vec(vec![0xff]));
        std::fs::write(&path, "#!/bin/sh\nsleep 30\nexit 0\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut cmd = std::process::Command::new(&path);
        cmd.process_group(0);
        let mut child = cmd.spawn().unwrap();
        let pid = child.id();
        let kill = KillOnDrop(&mut child);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let stat_path = format!("/proc/{pid}/stat");
        loop {
            let text = std::fs::read(&stat_path).unwrap_or_default();
            if text.contains(&0xff) {
                assert!(std::str::from_utf8(&text).is_err());
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the raw comm never appeared: {text:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            live_members(pid as i32).contains(&pid),
            "a non-text comm was skipped"
        );
        drop(kill);
    }

    /// Kill the group and wait, including when the assertion above fails.
    struct KillOnDrop<'a>(&'a mut std::process::Child);

    impl Drop for KillOnDrop<'_> {
        fn drop(&mut self) {
            // The group, not only the child: the script's `sleep` is still running in it.
            let _ = killpg(Pid::from_raw(self.0.id() as i32), Signal::SIGKILL);
            let _ = self.0.wait();
        }
    }
}
