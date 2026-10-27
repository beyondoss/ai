//! Killing a local process group, and keeping the kill from being lost.
//!
//! Everything that spawns a local child as its own process-group leader (the host command runner,
//! an MCP stdio server, [`TemplateRunner`](crate::exec_endpoint::TemplateRunner)'s local transport)
//! needs the same few things: an identity for the group that survives pid recycling ([`GroupId`]), a
//! guard that kills the group when the owning future is dropped ([`GroupKillGuard`]), and a way to
//! run a blocking kill on its own thread that the exit path can wait for
//! ([`spawn_tracked_cleanup`], [`wait_for_pending_group_kills`]).

use std::time::Duration;

/// Owns a spawned process-group leader; kills the whole group on drop unless [`disarm`](Self::disarm)d
/// (the normal, already-reaped exit path). This is what actually closes the cancellation gap
/// `kill_on_drop` leaves open: that flag only reaps the *direct* child on drop, but a future dropped
/// mid-`await` — cancellation racing the whole dispatch, not just this call's own timeout — unwinds
/// through this stack frame the same way, so arming the kill here (rather than only in the timeout
/// branch) covers both.
pub struct GroupKillGuard {
    pub group: Option<GroupId>,
}

impl GroupKillGuard {
    pub fn disarm(&mut self) {
        self.group = None;
    }
}

/// One per in-flight `GroupKillGuard`-triggered cleanup thread, `recv`d by
/// [`wait_for_pending_group_kills`] — see that function's doc comment for why this registry exists
/// at all (in short: `std::process::exit` doesn't wait for threads, so a caller that's about to call
/// it needs an explicit way to wait for a just-dropped guard's cleanup itself).
#[cfg(unix)]
static PENDING_GROUP_KILLS: std::sync::Mutex<Vec<std::sync::mpsc::Receiver<()>>> =
    std::sync::Mutex::new(Vec::new());

impl Drop for GroupKillGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(group) = self.group.take() {
            // `Drop` can't `.await`, so this can't be made fully synchronous with the caller — but a
            // real OS thread (not a `tokio::spawn`ed task) is scheduled by the kernel directly, rather
            // than depending on the ambient tokio runtime finding a moment to poll it. That matters
            // specifically when the runtime this drop happens on is a single-threaded `#[tokio::test]`
            // runtime under real contention from many other concurrently-running OS threads (one per
            // test) — a queued task on a busy single-threaded runtime can be delayed well past a
            // `tokio::spawn`'s usual near-instant turnaround; a plain OS thread isn't multiplexed onto
            // that runtime's own poll loop at all.
            //
            // Registered in `PENDING_GROUP_KILLS` (not truly fire-and-forget): a cancellation
            // (SIGTERM/SIGINT/SIGHUP) can reach `std::process::exit` moments after this guard drops,
            // and `process::exit` tears down every thread immediately with no chance for this one to
            // finish — see `wait_for_pending_group_kills`, which such a caller must call first.
            spawn_tracked_cleanup(move || kill_process_group(group));
        }
    }
}

/// Run `cleanup` on its own OS thread, registered so [`wait_for_pending_group_kills`] waits for it
/// before `std::process::exit` tears it down. The one way a `Drop` here hands off process cleanup.
#[cfg(unix)]
pub fn spawn_tracked_cleanup(cleanup: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        cleanup();
        let _ = tx.send(());
    });
    if let Ok(mut pending) = PENDING_GROUP_KILLS.lock() {
        // Opportunistically reclaim entries whose kill thread has already finished (`Ok`) or
        // whose sender was dropped (`Disconnected`), so this registry — otherwise drained in
        // full only just before `process::exit` — can't grow for the whole lifetime of a
        // long-lived `serve` daemon that cancels one bash after another. A still-running kill
        // thread's receiver (`Err(Empty)`) is kept, so `wait_for_pending_group_kills` can still
        // block on it.
        pending.retain(|rx| matches!(rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)));
        pending.push(rx);
    }
}

/// Block until every `GroupKillGuard` cleanup thread registered so far has actually finished its
/// `kill_process_group` work, or `timeout` elapses overall, whichever comes first.
///
/// Necessary, not just defensive: `GroupKillGuard::drop` can't `.await` (it spawns a detached OS
/// thread instead — see that impl's doc comment), and `std::process::exit` — which every shutdown-
/// signal cancellation path in this crate calls, to set a precise POSIX `128+signal` exit code —
/// terminates every thread immediately with no chance to finish whatever it was doing. Without this,
/// a `run` process that gets SIGTERM'd mid-bash-tool-call could exit before its own cleanup thread
/// ever got to run `kill`, silently orphaning the exact backgrounded grandchild `GroupKillGuard`
/// exists to reap — confirmed live via `run_signal_handling.rs`'s e2e tests, which spawn the real
/// compiled binary and send it a real OS signal. Call this immediately before any
/// `std::process::exit` that follows a turn ending in `agent_core::Error::Cancelled`.
#[cfg(unix)]
pub fn wait_for_pending_group_kills(timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    let Ok(mut pending) = PENDING_GROUP_KILLS.lock() else {
        return;
    };
    let receivers = std::mem::take(&mut *pending);
    drop(pending);
    for rx in receivers {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        // A zero/negative remaining budget still gets one non-blocking `try_recv`-equivalent poll
        // (`recv_timeout(ZERO)`) rather than being skipped outright, so a kill that finished just as
        // the deadline passed is still observed instead of assumed incomplete.
        let _ = rx.recv_timeout(remaining);
    }
}

/// SIGKILL an entire process group: `kill -KILL -<pgid>` (falling back to `kill -KILL <pgid>`, no
/// leading `-`, if that doesn't succeed — matching pi's own `killProcessTree`,
/// `process.kill(-pid, "SIGKILL")` falling back to `process.kill(pid, "SIGKILL")`), *then* an explicit
/// sweep that individually signals every process still reporting this pgid.
///
/// The sweep isn't defensive padding — it's load-bearing. Confirmed live on a real, resource-contended
/// GitHub Actions runner (a fast dedicated dev box never reproduces this): a backgrounded grandchild,
/// independently confirmed alive and already a member of the target pgid *before* the group-kill was
/// even issued, survived it completely — through its full multi-second sleep — while the group
/// leader died promptly. `kill -KILL -pgid` returning success only means the signal was handed to the
/// kernel for group-wide delivery; it does not appear to guarantee every existing member actually
/// receives it on every environment. Enumerating `ps`'s own pid/pgid columns and individually
/// `kill -KILL <pid>`-ing each match closes that gap without depending on group-signal semantics at
/// all. `pgid` doubles as the original child's own pid: the child is spawned as its own process-group
/// leader (`process_group(0)`), so pid and pgid are the same number. Shelling out (not `libc`/`nix`)
/// keeps this free of `unsafe`, which the workspace forbids.
///
/// Blocking, not async: called from places that each need a *synchronous* guarantee rather than a
/// detached future — [`GroupKillGuard`]'s `Drop` impl (which can't `.await` at all, so this runs on a
/// dedicated OS thread via `std::thread::spawn`, scheduled by the kernel independent of whatever the
/// ambient tokio runtime is doing) and `exec`'s timeout branch (via `spawn_blocking`, so the caller
/// actually waits for the kill to complete before `run()` returns, rather than firing it and hoping it
/// lands before a caller re-checks side effects).
///
/// Never once the group's number has been recycled: see [`GroupId`].
#[cfg(unix)]
pub fn kill_process_group(group: GroupId) {
    kill_group(group, true);
}

/// [`kill_process_group`] without its kill-by-pid fallback: only processes that are *in* group
/// `pgid` are signalled. For a caller whose group leader has already been reaped — its pid is free
/// to be reused by an unrelated process, which a fallback `kill -KILL <pid>` would then hit. The
/// non-Linux `bash_proc` sweep (Linux tracks processes through `/proc` instead).
#[cfg(unix)]
#[cfg_attr(all(target_os = "linux", not(test)), allow(dead_code))]
pub fn kill_group_members(group: GroupId) {
    kill_group(group, false);
}

/// A process group, as the process that leads it was started: its number, and the leader's start
/// time — which tells our group's number apart from the same number recycled for someone else's.
///
/// A group id is its leader's pid, and the kernel hands a pid out again only once nothing uses it —
/// no process, and no group or session — so while our leader is unreaped (alive or a zombie), or
/// any member of our group lives, the number is ours. Once the group is empty it is free: a later
/// process may get that pid and lead a group of its own with it, and a `kill -<pgid>` aimed at ours
/// would then hit the stranger's. [`Self::recycled`] is that case: a live process holds the pid
/// but is not our leader (another start time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupId {
    pub pgid: u32,
    /// The leader's start time (`/proc/<pid>/stat` field 22), read while it was ours; `None` where
    /// `/proc` is unavailable, which leaves the old behavior (no recycling check).
    start: Option<u64>,
}

impl GroupId {
    /// The group `leader` leads, read now — call it while `leader` is still unreaped (right after
    /// spawning it).
    pub fn leader(leader: u32) -> Self {
        Self {
            pgid: leader,
            start: start_time(leader),
        }
    }

    /// The group as a session recorded it — see [`super::bash_proc::ProcGroup::recorded`].
    pub fn recorded(pgid: u32, start: Option<u64>) -> Self {
        Self { pgid, start }
    }

    /// The leader's start time, read while it was ours.
    pub fn start(&self) -> Option<u64> {
        self.start
    }

    /// Whether our leader still holds the number (alive or a zombie): a group signal can reach only
    /// our group.
    pub fn leader_holds(&self) -> bool {
        self.start.is_some() && start_time(self.pgid) == self.start
    }

    /// Whether the number now belongs to someone else: a live process has the pid and it is not our
    /// leader. Our group is then necessarily empty (the kernel would not have reused the number
    /// otherwise), so there is nothing of ours left to signal by group.
    pub fn recycled(&self) -> bool {
        match (self.start, start_time(self.pgid)) {
            (Some(ours), Some(now)) => ours != now,
            _ => false,
        }
    }
}

/// A process's start time in clock ticks since boot (`/proc/<pid>/stat` field 22), or `None` if it
/// does not exist (or this is not Linux). With the pid, it names one process for all time.
pub fn start_time(pid: u32) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read(format!("/proc/{pid}/stat")).ok()?;
        // `pid (comm) state …`: `comm` may hold spaces or parens, so count from after the last `)`.
        let close = stat.iter().rposition(|&b| b == b')')?;
        let rest = std::str::from_utf8(&stat[close + 1..]).ok()?;
        // Field 3 (`state`) is the first after `)`, so field 22 is the 20th.
        rest.split_whitespace().nth(19)?.parse().ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

#[cfg(unix)]
fn kill_group(group: GroupId, pid_fallback: bool) {
    let pgid = group.pgid;
    if group.recycled() {
        // Our group is gone and its number is a stranger's now: signalling it would kill them.
        return;
    }
    // The pid is the leader's only while the leader holds it; once reaped, a fallback could hit
    // whoever got the pid next.
    let pid_fallback = pid_fallback && group.leader_holds();
    let group_result = std::process::Command::new("kill")
        .arg("-KILL")
        // `--` first, so the exit status means what the code below assumes. Without it procps-ng
        // `kill` still signals a live group, but exits 1 doing so (and 0 for a group that is already
        // gone) — inverted, so the kill-by-pid fallback below ran after every successful group kill.
        .arg("--")
        .arg(format!("-{pgid}"))
        // `kill` prints "No such process" on a group that already exited, which is the *expected*
        // case here — and since an MCP connection sweeps its group on every drop, inheriting that
        // would put a meaningless line on the operator's console routinely. The status is what this
        // function acts on; the text adds nothing.
        .stderr(std::process::Stdio::null())
        .status();
    // A non-success here (including `kill`'s own exit code, e.g. ESRCH because the group already
    // exited on its own between the timeout firing and this running) isn't worth logging on its own —
    // the group kill covers the overwhelmingly common case, so try the direct fallback next regardless
    // of *why* it didn't succeed; a no-op fallback against an already-gone process is harmless.
    if pid_fallback && !matches!(&group_result, Ok(status) if status.success()) {
        match std::process::Command::new("kill")
            .arg("-KILL")
            .arg(pgid.to_string())
            .stderr(std::process::Stdio::null())
            .status()
        {
            // Still only the "couldn't even run `kill`" case is worth a warning — a non-zero exit
            // here most likely just means the process (or its whole group) was already gone.
            Ok(_) => {}
            Err(e) => {
                // If `kill` itself couldn't run (missing binary, restrictive sandboxing), a
                // backgrounded grandchild from the timed-out command may be left running with
                // nothing else to reap it — surface that instead of silently losing the signal.
                tracing::warn!(pgid, error = %e, "failed to run `kill` to reap a timed-out process");
            }
        }
    }
    // Two passes with a short gap: the first catches whatever the group-kill above missed; the
    // second catches anything that was itself mid-fork (and so invisible to `ps`) during the first.
    sweep_kill_remaining_group_members(pgid);
    std::thread::sleep(Duration::from_millis(50));
    if !group.recycled() {
        sweep_kill_remaining_group_members(pgid);
    }
}

/// Enumerate every process currently reporting `pgid` via `ps -eo pid=,pgid=` and SIGKILL each one
/// individually. See [`kill_process_group`]'s doc comment for why this exists — a bulk
/// `kill -KILL -<pgid>` was empirically observed to miss a live, already-a-member process on a real
/// CI runner. Best-effort: a `ps`/`kill` invocation failing outright is silently ignored (mirrors the
/// group-kill's own "no-op fallback against an already-gone process is harmless" stance) since the
/// caller already tried the cheaper group-kill/direct-kill paths first.
#[cfg(unix)]
fn sweep_kill_remaining_group_members(pgid: u32) {
    let Ok(out) = std::process::Command::new("ps")
        .args(["-eo", "pid=,pgid="])
        .output()
    else {
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let (Some(pid_str), Some(pgid_str)) = (fields.next(), fields.next()) else {
            continue;
        };
        if pgid_str.parse::<u32>() != Ok(pgid) {
            continue;
        }
        // Best-effort per-pid kill: an already-dead pid (ESRCH) is the overwhelmingly common,
        // harmless case (the group-kill above likely already got it) — nothing further to do either
        // way, so the result isn't checked.
        let _ = std::process::Command::new("kill")
            .arg("-KILL")
            .arg(pid_str)
            .status();
    }
}
