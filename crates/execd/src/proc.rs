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
use std::sync::Mutex;
use std::time::Duration;

use nix::errno::Errno;
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;

/// The pids of children execd spawned and still awaits. The orphan reaper must leave these alone:
/// reaping one would steal the exit status its owner is about to ask for.
static OWNED: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());

/// A spawned command, leading a process group of its own.
pub(crate) struct Proc {
    pub child: tokio::process::Child,
    /// The group's id, which is also the leader's pid.
    pub pgid: i32,
    pid: u32,
}

impl Drop for Proc {
    fn drop(&mut self) {
        locked(&OWNED).remove(&self.pid);
    }
}

/// Spawn `cmd` as the leader of a new process group.
///
/// The registry lock is held across the spawn: the reaper takes it to scan, so it can never see the
/// child as an unowned zombie in the gap between `fork` and registration.
pub(crate) fn spawn(cmd: &mut tokio::process::Command) -> io::Result<Proc> {
    cmd.process_group(0);
    let mut owned = locked(&OWNED);
    let child = cmd.spawn()?;
    let pid = child.id().unwrap_or(0);
    owned.insert(pid);
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
/// The syscall answers the common case, an empty group, without touching `/proc`; only a group
/// that exists is scanned (one `stat` read per process on the host).
pub(crate) fn live_members(pgid: i32) -> Vec<u32> {
    let mut out = Vec::new();
    if !group_exists(pgid) {
        return out;
    }
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return out;
    };
    for e in dir.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if let Some((state, _, pg)) = stat(pid)
            && state != 'Z'
            && state != 'X'
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

/// [`live_members`] without holding up the runtime for the `/proc` scan.
pub(crate) async fn live_members_async(pgid: i32) -> Vec<u32> {
    if !group_exists(pgid) {
        return Vec::new();
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
/// by execd rather than by the host's init, so [`reap_orphans`] can reap it. Meant for the binary;
/// an embedding test harness has no business changing its process's reaping.
pub fn become_subreaper() -> io::Result<()> {
    nix::sys::prctl::set_child_subreaper(true).map_err(io::Error::from)
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
        let _ = tokio::task::spawn_blocking(reap_once).await;
        tokio::select! {
            _ = sigchld.recv() => {}
            () = stop.cancelled() => return,
            // A zombie created between a scan and the next signal registration is still caught: the
            // signal is level-coalesced by tokio, and this is the backstop for the one it can miss.
            () = tokio::time::sleep(Duration::from_secs(1)) => {}
        }
    }
}

/// One pass of [`reap_orphans`]; how many zombies it collected.
pub fn reap_once() -> usize {
    use nix::sys::wait::{WaitPidFlag, waitpid};
    let me = i32::try_from(std::process::id()).unwrap_or(0);
    // Held for the whole pass, so a child being spawned cannot be seen unregistered.
    let owned = locked(&OWNED);
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return 0;
    };
    let mut reaped = 0;
    for e in dir.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if owned.contains(&pid) {
            continue;
        }
        if let Some(('Z', ppid, _)) = stat(pid)
            && ppid == me
            && waitpid(Pid::from_raw(pid as i32), Some(WaitPidFlag::WNOHANG)).is_ok()
        {
            reaped += 1;
        }
    }
    reaped
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
