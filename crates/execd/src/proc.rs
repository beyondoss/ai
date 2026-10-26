//! Processes: spawning into their own group, group liveness and kills by syscall, and reaping the
//! orphans a PID 1 inherits.
//!
//! Everything here is a syscall or a read of `/proc`; nothing spawns `kill`.
//!
//! **Bounds.** [`OWNED`] holds one pid per child execd spawned and has not yet finished with: freed
//! when its [`Proc`] drops. N concurrent cold callers means N entries, 4 bytes each, plus the
//! children themselves, which the OS bounds (`pids.max`, `RLIMIT_NPROC`). [`Groups`] is the same
//! idea per server, for the shutdown sweep.

use std::collections::BTreeSet;
use std::io;
use std::os::fd::AsFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

use nix::errno::Errno;
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;

/// The pids of children execd spawned and still awaits. The orphan reaper must leave these alone:
/// reaping one would steal the exit status its owner is about to ask for.
static OWNED: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());

/// Held shared across every spawn, from before the child exists until it is in [`OWNED`]; held
/// exclusively by the reaper around the check and reap of one zombie. Spawns do not exclude each
/// other; the reaper excludes them all, and only for as long as one `waitpid`.
///
/// **Why a child is never reaped between its creation and its registration.** The reaper takes
/// this lock exclusively before it asks whether a pid is owned, and a spawn holds it shared from
/// before it creates the child to after the insert. So when the reaper holds it, every spawn is
/// either entirely before it (its child is in `OWNED`, and is skipped) or entirely after it (no
/// such child exists yet). There is no moment at which a new child is both visible to the reaper
/// and unregistered.
static SPAWNING: RwLock<()> = RwLock::new(());

/// Whether this process is its descendants' subreaper, so every orphaned group member is one of its
/// own descendants (see [`live_members`]). Set by [`become_subreaper`].
static SUBREAPER: AtomicBool = AtomicBool::new(false);

/// A spawned command, leading a process group of its own.
pub(crate) struct Proc {
    pub child: tokio::process::Child,
    /// The group's id, which is also the leader's pid.
    pub pgid: i32,
    pid: u32,
}

impl Drop for Proc {
    fn drop(&mut self) {
        if let Ok(mut owned) = OWNED.lock() {
            owned.remove(&self.pid);
        }
    }
}

/// Spawn `cmd` as the leader of a new process group.
///
/// [`SPAWNING`] is held shared across the spawn, so the reaper can never see the child as an
/// unowned zombie before it is registered, and concurrent spawns do not wait for each other.
pub(crate) fn spawn(cmd: &mut tokio::process::Command) -> io::Result<Proc> {
    cmd.process_group(0);
    let _spawning = SPAWNING
        .read()
        .map_err(|_| io::Error::other("process registry poisoned"))?;
    let child = cmd.spawn()?;
    let pid = child.id().unwrap_or(0);
    if let Ok(mut owned) = OWNED.lock() {
        owned.insert(pid);
    }
    Ok(Proc {
        child,
        pgid: i32::try_from(pid).unwrap_or(0),
        pid,
    })
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

/// The live (non-zombie) members of group `pgid`. A zombie never counts: a sandbox whose pid 1 is
/// not an init never reaps what it inherits, and the spec says such a group is empty.
///
/// The syscall answers the common case, an empty group, without touching `/proc`. For a group that
/// exists, the members are found in execd's own process tree: as the subreaper, every orphan of a
/// command is a descendant of execd, and the kernel lists a process's children
/// (`/proc/<pid>/task/*/children`), so the cost follows the size of that tree and not the number of
/// processes on the host. Without a subreaper, or on a kernel that does not list children, the
/// whole of `/proc` is scanned. So is it when the tree shows no live member of a group that still
/// exists: a member whose parent exited during the walk moves to execd unseen, and "empty" ends a
/// stream, so that answer is confirmed. It is reached only while a zombie holds the group.
pub(crate) fn live_members(pgid: i32) -> Vec<u32> {
    if !group_exists(pgid) {
        return Vec::new();
    }
    if SUBREAPER.load(Ordering::Relaxed)
        && let Some(found) = live_members_in_tree(pgid)
        && !found.is_empty()
    {
        return found;
    }
    live_members_scan(pgid)
}

fn is_live(state: char) -> bool {
    state != 'Z' && state != 'X'
}

/// The children of `pid` (`"self"` for this process), from every thread's `children` file; `None`
/// if the kernel has no such file.
fn children_of(pid: &str) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    for t in std::fs::read_dir(format!("/proc/{pid}/task"))
        .ok()?
        .flatten()
    {
        match std::fs::read_to_string(t.path().join("children")) {
            Ok(text) => out.extend(
                text.split_whitespace()
                    .filter_map(|p| p.parse::<u32>().ok()),
            ),
            // The file is missing from a kernel without it, and so is a thread that just exited.
            Err(e) if e.kind() == io::ErrorKind::NotFound && t.path().exists() => return None,
            Err(_) => {}
        }
    }
    Some(out)
}

fn live_members_in_tree(pgid: i32) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    let mut todo = children_of("self")?;
    while let Some(pid) = todo.pop() {
        if let Some((state, _, pg)) = stat(pid)
            && is_live(state)
            && pg == pgid
        {
            out.push(pid);
        }
        // A process that is gone has no children to find.
        if let Some(kids) = children_of(&pid.to_string()) {
            todo.extend(kids);
        }
    }
    Some(out)
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

/// Whether group `pgid` has no live member.
pub(crate) fn group_empty(pgid: i32) -> bool {
    live_members(pgid).is_empty()
}

/// Whether the kernel lists a process's children (`CONFIG_PROC_CHILDREN`), probed once. The reads
/// that use them take microseconds, so they run on the calling thread; a full scan of `/proc`
/// does not, and goes to the blocking pool.
fn has_children_files() -> bool {
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROBE.get_or_init(|| children_of("self").is_some())
}

/// [`live_members`] without holding up the runtime: the walk of execd's own tree runs inline, and
/// only a scan of `/proc` is sent to the blocking pool.
pub(crate) async fn live_members_async(pgid: i32) -> Vec<u32> {
    if !group_exists(pgid) {
        return Vec::new();
    }
    if SUBREAPER.load(Ordering::Relaxed)
        && has_children_files()
        && let Some(found) = live_members_in_tree(pgid)
        && !found.is_empty()
    {
        return found;
    }
    tokio::task::spawn_blocking(move || live_members(pgid))
        .await
        .unwrap_or_default()
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

/// The process groups a server has running, so a shutdown can kill them all.
///
/// Bounded by the number of running commands and jobs; each entry is freed when its command or
/// job ends ([`GroupSlot`] drops, or the job's supervisor finishes).
#[derive(Default)]
pub(crate) struct Groups(Mutex<BTreeSet<i32>>);

impl Groups {
    pub fn insert(&self, pgid: i32) {
        if let Ok(mut g) = self.0.lock() {
            g.insert(pgid);
        }
    }

    pub fn remove(&self, pgid: i32) {
        if let Ok(mut g) = self.0.lock() {
            g.remove(&pgid);
        }
    }

    /// SIGKILL every registered group, and forget them.
    pub fn kill_all(&self) -> usize {
        let all = self
            .0
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();
        for &g in &all {
            kill_group(g);
        }
        all.len()
    }
}

/// Make this process the subreaper of its descendants: a process that loses its parent is adopted
/// by execd rather than by the host's init, so [`reap_orphans`] can reap it. Meant for the binary;
/// an embedding test harness has no business changing its process's reaping.
pub fn become_subreaper() -> io::Result<()> {
    nix::sys::prctl::set_child_subreaper(true).map_err(io::Error::from)?;
    SUBREAPER.store(true, Ordering::Relaxed);
    Ok(())
}

/// Reap every zombie child of this process that execd did not spawn itself, on each `SIGCHLD`.
///
/// A PID 1 (a container's, E2B's) inherits every orphan of every command, and unless it reaps them
/// they sit as zombies for the life of the sandbox. This waits on exactly those pids: the ones
/// execd spawned are skipped (their owner is about to collect the exit status, and a bulk
/// `waitpid(-1)` here would steal it), and a zombie is found by its parent being this process.
///
/// Bounded: one scan of `/proc` per wakeup, and `SIGCHLD`s that arrive during a scan coalesce into
/// the next wakeup. Runs until `stop` is cancelled.
pub async fn reap_orphans(stop: tokio_util::sync::CancellationToken) {
    use tokio::signal::unix::{SignalKind, signal};
    let Ok(mut sigchld) = signal(SignalKind::child()) else {
        tracing::warn!("cannot listen for SIGCHLD; orphans will not be reaped");
        return;
    };
    loop {
        if has_children_files() {
            reap_once();
        } else {
            let _ = tokio::task::spawn_blocking(reap_once).await;
        }
        tokio::select! {
            _ = sigchld.recv() => {}
            () = stop.cancelled() => return,
        }
    }
}

/// Child exits, as a wake-up for a task that waits for a process group to empty.
///
/// When the last member of a group exits, its parent is nearly always execd (the subreaper adopts
/// every orphan) or the member's own parent in the group, and execd gets `SIGCHLD` for the first
/// and, a moment later, for the second. So a waiter sleeps until a `SIGCHLD` and re-checks, with no
/// timer running. The one case no signal covers is a last member whose live parent is neither
/// execd nor in the group (a process that left the group and still has a child in it); for that
/// there is a slow fallback, [`GROUP_FALLBACK`], which only decides how late the end of such a
/// group is noticed. Create one before the first check, so an exit that races the check is kept.
pub(crate) struct ChildEvents(Option<tokio::signal::unix::Signal>);

/// How long a waiter sleeps without a `SIGCHLD` before it looks anyway: as long as a heartbeat, the
/// delay the protocol already tolerates for a silent stream, and rare enough to cost nothing.
pub(crate) const GROUP_FALLBACK: Duration = Duration::from_secs(5);

impl ChildEvents {
    pub fn new() -> Self {
        Self(tokio::signal::unix::signal(tokio::signal::unix::SignalKind::child()).ok())
    }

    /// Wait for a child to exit, or for [`GROUP_FALLBACK`].
    pub async fn next(&mut self) {
        match &mut self.0 {
            Some(sig) => {
                tokio::select! {
                    _ = sig.recv() => {}
                    () = tokio::time::sleep(GROUP_FALLBACK) => {}
                }
            }
            // No signal to wait on: look often.
            None => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
}

/// One pass of [`reap_orphans`]; how many zombies it collected.
///
/// The candidates are this process's own children (listed by the kernel, so the pass costs the
/// number of execd's threads and children, not of processes on the host; a full `/proc` scan is the
/// fallback). Each zombie is checked and reaped under [`SPAWNING`] held exclusively, which is what
/// keeps a child that is mid-spawn safe; nothing is held while looking.
pub fn reap_once() -> usize {
    let me = i32::try_from(std::process::id()).unwrap_or(0);
    let candidates = match children_of("self") {
        Some(c) => c,
        None => match std::fs::read_dir("/proc") {
            Ok(dir) => dir
                .flatten()
                .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()))
                .collect(),
            Err(_) => return 0,
        },
    };
    candidates
        .into_iter()
        .filter(|&pid| matches!(stat(pid), Some(('Z', ppid, _)) if ppid == me))
        .filter(|&pid| reap_unowned(pid))
        .count()
}

/// Reap zombie `pid` unless execd spawned it.
fn reap_unowned(pid: u32) -> bool {
    use nix::sys::wait::{WaitPidFlag, waitpid};
    let Ok(_exclusive) = SPAWNING.write() else {
        return false;
    };
    let Ok(owned) = OWNED.lock() else {
        return false;
    };
    !owned.contains(&pid) && waitpid(Pid::from_raw(pid as i32), Some(WaitPidFlag::WNOHANG)).is_ok()
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
