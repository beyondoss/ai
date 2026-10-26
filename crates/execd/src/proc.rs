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

use std::collections::{BTreeSet, HashMap};
use std::io;
use std::os::fd::AsFd;
use std::os::unix::process::ExitStatusExt as _;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Duration;

use nix::errno::Errno;
use nix::sys::signal::{Signal, killpg};
use nix::sys::wait::{WaitPidFlag, WaitStatus, waitpid};
use nix::unistd::Pid;
use tokio::sync::{oneshot, watch};

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

/// Whether this process is its descendants' subreaper, so every orphaned group member becomes its
/// own child (see [`group_gone`]). Set by [`become_subreaper`].
static SUBREAPER: AtomicBool = AtomicBool::new(false);

fn children() -> std::sync::MutexGuard<'static, Option<HashMap<u32, oneshot::Sender<ExitStatus>>>> {
    let mut g = locked(&CHILDREN);
    g.get_or_insert_with(HashMap::new);
    g
}

/// A spawned command, leading a process group of its own.
pub(crate) struct Proc {
    pub stdin: Option<tokio::process::ChildStdin>,
    pub stdout: Option<tokio::process::ChildStdout>,
    pub stderr: Option<tokio::process::ChildStderr>,
    /// The group's id, which is also the leader's pid.
    pub pgid: i32,
    exit: oneshot::Receiver<ExitStatus>,
    /// What `wait` answered, kept so that it can answer again: a oneshot receiver must not be
    /// polled after it completes.
    done: Option<Result<ExitStatus, ()>>,
}

impl Proc {
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

/// SIGKILL every member of group `pgid`. A group that is already gone is not an error.
pub(crate) fn kill_group(pgid: i32) {
    signal_group(pgid, Signal::SIGKILL);
}

pub(crate) fn signal_group(pgid: i32, sig: Signal) {
    if pgid > 1 {
        let _ = killpg(Pid::from_raw(pgid), sig);
    }
}

/// Whether any process, zombie or not, is in group `pgid`: `kill(-pgid, 0)`.
pub(crate) fn group_exists(pgid: i32) -> bool {
    pgid > 1 && !matches!(killpg(Pid::from_raw(pgid), None), Err(Errno::ESRCH))
}

/// The live (non-zombie) members of group `pgid`, by reading `/proc`: the pids, for an operation
/// that reports them. A zombie never counts: the spec says a group of zombies is empty.
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
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return out;
    };
    for e in dir.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if let Some((state, _, pg)) = stat(pid)
            && is_live(state)
            && pg == pgid
        {
            out.push(pid);
        }
    }
    out
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

/// [`live_members`] without holding up the runtime for the `/proc` scan.
pub(crate) async fn live_members_async(pgid: i32) -> Vec<u32> {
    if !group_exists(pgid) {
        return Vec::new();
    }
    off_runtime(move || live_members(pgid)).await
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

/// Whether group `pgid` has no live member, for a caller that is already off the runtime.
pub(crate) fn group_empty(pgid: i32) -> bool {
    if !group_exists(pgid) {
        return true;
    }
    !SUBREAPER.load(Ordering::Relaxed) && live_members_scan(pgid).is_empty()
}

/// `(state, ppid, pgrp)` of `pid`, from `/proc/<pid>/stat`.
fn stat(pid: u32) -> Option<(char, i32, i32)> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name is parenthesised and may itself hold spaces and parentheses: split after
    // the last one.
    let (_, rest) = text.rsplit_once(')')?;
    let mut f = rest.split_whitespace();
    let state = f.next()?.chars().next()?;
    let ppid = f.next()?.parse().ok()?;
    let pgrp = f.next()?.parse().ok()?;
    Some((state, ppid, pgrp))
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

/// The process groups a server has running, so a shutdown can kill them all.
///
/// Bounded by the number of running commands and jobs; each entry is freed when its command or
/// job ends ([`GroupSlot`] drops, or the job's supervisor finishes). Once [`Self::kill_all`] has
/// run the set is closed: a group registered after it (a spawn that raced the shutdown) is killed
/// at once instead of surviving it.
#[derive(Default)]
pub(crate) struct Groups(Mutex<GroupSet>);

#[derive(Default)]
struct GroupSet {
    live: BTreeSet<i32>,
    closed: bool,
}

impl Groups {
    pub fn insert(&self, pgid: i32) {
        let mut g = locked(&self.0);
        if g.closed {
            drop(g);
            kill_group(pgid);
        } else {
            g.live.insert(pgid);
        }
    }

    pub fn remove(&self, pgid: i32) {
        locked(&self.0).live.remove(&pgid);
    }

    /// SIGKILL every registered group, forget them, and close the set. How many it killed.
    pub fn kill_all(&self) -> usize {
        let all = {
            let mut g = locked(&self.0);
            g.closed = true;
            std::mem::take(&mut g.live)
        };
        for &g in &all {
            kill_group(g);
        }
        all.len()
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
    use super::*;
    use std::os::unix::process::CommandExt as _;

    /// A zombie is in the group for `kill(-pgid, 0)` and not for the spec: its group is empty.
    #[test]
    fn a_zombie_is_not_a_live_member() {
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
        let mut sleeper = std::process::Command::new("sleep");
        sleeper.arg("30").process_group(0);
        let mut child = sleeper.spawn().unwrap();
        let pgid = child.id() as i32;
        assert_eq!(live_members(pgid), vec![child.id()]);
        kill_group(pgid);
        child.wait().unwrap();
        assert!(live_members(pgid).is_empty());
        // A group that is gone is not an error, and nothing outside a real group is ever signalled.
        kill_group(pgid);
        kill_group(0);
        kill_group(1);
    }

    #[test]
    fn the_pipe_capacity_is_the_kernels_number() {
        let mut c = std::process::Command::new("true");
        c.stdout(std::process::Stdio::piped());
        let mut child = c.spawn().unwrap();
        let out = child.stdout.take().unwrap();
        assert_eq!(pipe_capacity(&out), 64 * 1024);
        child.wait().unwrap();
    }
}
