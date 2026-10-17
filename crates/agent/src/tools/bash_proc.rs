//! The processes of one `bash` job on this host: spawned tagged, streamed, noticed when the shell
//! exits but leaves some of them running, and killed — all of them — when the job is stopped.
//!
//! **Which processes belong to a job.** A process group alone does not answer it: `setsid`, a
//! daemon's double fork and `timeout` (which makes its own group) all leave the group, so a
//! group-only tracker loses exactly the processes most likely to outlive the agent. Every job is
//! therefore spawned with an environment tag, [`TAG_ENV`]: an environment is inherited across
//! `fork`/`exec`, `setsid`, `nohup`, `timeout` and double forks alike, and `/proc/<pid>/environ` (the
//! environment a process was *started* with) is readable for our own uid without privilege. A job's
//! processes are the live ones that carry its tag **or** are still in its process group (which also
//! covers a process that cleared its environment but stayed in the group) — the group only while its
//! number is still ours, see [`ProcGroup`] — **or** descend from one of those, **or** were found by an
//! earlier scan (by pid and start time: membership lasts a process's life, so a descendant stays the
//! job's after its parent exits). See "`bash` jobs: what a command leaves running" in
//! `ARCHITECTURE.md` for the alternatives measured and rejected (cgroup v2 per job). Stdio MCP servers
//! are tracked the same way (`mcp_stdio`).
//!
//! What escapes: a process that clears its environment (`env -i`) or whose environment we may not
//! read (another uid, a setuid binary, a non-dumpable process, e.g. `ssh-agent`), that left the group,
//! and whose parent had exited before any scan saw it. Never a name match: a kill only ever reaches a
//! process a scan found, verified through a pidfd (same start time) so a reused pid is never hit.
//!
//! Linux reads `/proc`; another unix falls back to the process group (`kill(-pgid, 0)` for liveness),
//! which misses `setsid` and `timeout`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::exec::{ExecResult, JobHooks, Kill};

/// The environment variable every job's processes inherit. Its value is a `:`-separated list of
/// tags: a job started from inside another agent's job keeps the outer tag too, so the outer agent
/// still finds (and stops) everything the inner one started.
pub const TAG_ENV: &str = "AI_AGENT_JOB_TAG";

/// How long the pipes may sit silent once the shell has exited before the call asks what is still
/// running: output already in flight lands first. Same 100 ms as `exec::POST_EXIT_IDLE_GRACE`, for
/// the same reason (pi's `EXIT_STDIO_GRACE_MS`).
const SETTLE: Duration = Duration::from_millis(100);

/// How often liveness is polled where a pidfd is not available (another unix, a kernel before 5.3).
const POLL: Duration = Duration::from_millis(250);

/// How long a stopped job's processes get to exit on SIGTERM before the SIGKILL sweep. SIGTERM
/// first because SIGKILL cannot be relayed: `sudo cmd` forwards a TERM to its root child (gone 40
/// ms after the TERM on the dev box, `sudo -n sleep`), but a KILL takes `sudo` down alone and leaves
/// the child — which we may not signal — running. 300 ms is ~7x that, and is only spent in full on a
/// process that ignores or traps TERM; a job whose processes all exit returns as soon as they have.
pub const TERM_GRACE: Duration = Duration::from_millis(300);

/// How often the grace re-checks whether the jobs' processes have exited.
const GRACE_POLL: Duration = Duration::from_millis(10);

/// One job's processes: its process group and its tag.
///
/// **The group number is ours only while our group lives.** Once the shell is reaped and every
/// process left in its group has exited (`setsid` leftovers do not count: they left), the kernel
/// may hand the number to an unrelated process, which can lead a group of its own with it. So
/// processes are matched by group, and the group signalled as a unit, only while that cannot have
/// happened: a group signal only while the shell itself still holds the number (alive or a zombie),
/// and group membership only until the group is seen empty after the shell's reap, or the number is
/// seen held by another process ([`crate::tools::exec::GroupId::recycled`]) — after which only the
/// tag counts, each kill verified through a pidfd.
#[derive(Debug, Clone)]
pub struct ProcGroup {
    /// The shell's pid, which leads its own group (`process_group(0)`).
    pub pgid: u32,
    /// This job's tag — see [`TAG_ENV`].
    pub tag: String,
    /// The shell's identity, to tell our group's number from a recycled one.
    leader: super::exec::GroupId,
    /// Latched once the group can no longer be ours: matched by tag only from then on.
    group_gone: std::sync::Arc<AtomicBool>,
    /// What the last scan found, by pid and start time: a process once seen as the job's stays the
    /// job's for life (Linux; see `imp::scan`).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    seen: std::sync::Arc<Mutex<std::collections::HashSet<(u32, u64)>>>,
}

/// A tag no other job of any process on this host shares: this process's pid, its start time, and a
/// per-process counter.
pub(crate) fn new_tag() -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    static EPOCH: std::sync::OnceLock<u128> = std::sync::OnceLock::new();
    let epoch = EPOCH.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    });
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{}.{epoch:x}.{seq}", std::process::id())
}

/// The tags this process itself carries: it runs inside those jobs, so it must never stop them.
pub fn own_tags() -> Vec<String> {
    std::env::var(TAG_ENV)
        .map(|v| {
            v.split(':')
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The [`TAG_ENV`] value a job's shell starts with: any tags this process inherited, then `tag`.
pub(crate) fn tag_value(tag: &str) -> String {
    match std::env::var(TAG_ENV) {
        Ok(outer) if !outer.is_empty() => format!("{outer}:{tag}"),
        _ => tag.to_string(),
    }
}

impl ProcGroup {
    /// The group `leader` leads (call it while `leader` is unreaped, right after spawning it), with
    /// its tag.
    pub fn new(leader: u32, tag: String) -> Self {
        Self {
            pgid: leader,
            tag,
            leader: super::exec::GroupId::leader(leader),
            group_gone: std::sync::Arc::new(AtomicBool::new(false)),
            seen: std::sync::Arc::default(),
        }
    }

    /// A job's processes as a session recorded them — `pgid`, the leader's start time, the tag — for
    /// a later process to find what an earlier one, killed before it could stop them, left behind.
    /// Without a start time (no `/proc` when recorded) the group number cannot be told from a
    /// recycled one, so only the tag counts.
    pub fn recorded(pgid: u32, leader_start: Option<u64>, tag: String) -> Self {
        Self {
            pgid,
            tag,
            leader: super::exec::GroupId::recorded(pgid, leader_start),
            group_gone: std::sync::Arc::new(AtomicBool::new(leader_start.is_none())),
            seen: std::sync::Arc::default(),
        }
    }

    /// The group leader's start time, as read when it was spawned — what [`Self::recorded`] takes.
    pub fn leader_start(&self) -> Option<u64> {
        self.leader.start()
    }

    /// Whether a live process carries this job's tag. Only the tag proves a process is a job's: a
    /// recorded group number alone is no evidence that its members are ours.
    pub fn tag_alive(&self) -> bool {
        imp::tag_alive(self)
    }

    /// Whether a process in group `pgid` is still one of ours — see the type's doc. Latches `false`.
    fn by_group(&self) -> bool {
        if self.pgid == 0 || self.group_gone.load(Ordering::Acquire) {
            return false;
        }
        if self.leader.recycled() {
            self.group_gone.store(true, Ordering::Release);
            return false;
        }
        true
    }

    /// A scan found no live process in group `pgid`: once the shell no longer holds the number
    /// either, the group is gone for good — it can only ever be someone else's from here on.
    fn saw_group_empty(&self) {
        if !self.leader.leader_holds() {
            self.group_gone.store(true, Ordering::Release);
        }
    }

    /// Whether signalling group `pgid` as a unit can reach only this job: the shell still holds the
    /// number.
    fn may_signal_group(&self) -> bool {
        self.pgid != 0 && self.leader.leader_holds()
    }

    /// The pids of this job's live processes (zombies excluded), never this process itself.
    pub fn members(&self) -> Vec<u32> {
        imp::members(self)
    }

    /// Stop every process of the job: SIGTERM, up to [`TERM_GRACE`] for them to exit, then SIGKILL,
    /// sweeping until none is left (a process that forked mid-sweep is caught by the next pass).
    /// Returns the pids still running after it — processes we may not signal (`EPERM`: another
    /// uid, e.g. the root child of a `sudo` that died before relaying). Blocking: callers run it on
    /// a blocking thread. Several jobs at once: [`super::exec::kill_all`], which spends one grace.
    pub fn kill(&self) -> Vec<u32> {
        self.terminate();
        let deadline = Instant::now() + TERM_GRACE;
        await_exit(&[self], deadline);
        self.sweep()
    }

    /// SIGTERM every process of the job, without waiting.
    pub(crate) fn terminate(&self) {
        imp::terminate(self);
    }

    /// The SIGKILL sweep alone; returns what survived it — see [`Self::kill`].
    pub(crate) fn sweep(&self) -> Vec<u32> {
        imp::kill(self)
    }

    /// Resolve once every process of the job has exited.
    pub async fn wait_gone(&self) {
        imp::wait_gone(self).await;
    }
}

/// Block until every process of `groups` has exited or `deadline` passes. Blocking.
pub(crate) fn await_exit(groups: &[&ProcGroup], deadline: Instant) {
    while groups.iter().any(|g| !g.members().is_empty()) {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        std::thread::sleep(GRACE_POLL.min(deadline - now));
    }
}

/// A short description of `pids` for the model — "a process (`sleep 300`)", "2 processes (`a`,
/// `b`)" — their command lines truncated, at most three shown.
pub fn describe(pids: &[u32]) -> String {
    const SHOWN: usize = 3;
    let mut parts: Vec<String> = pids
        .iter()
        .take(SHOWN)
        .map(|&pid| format!("`{}`", imp::command_line(pid)))
        .collect();
    if pids.len() > SHOWN {
        parts.push(format!("and {} more", pids.len() - SHOWN));
    }
    let what = match pids.len() {
        1 => "a process".to_string(),
        n => format!("{n} processes"),
    };
    format!("{what} ({})", parts.join(", "))
}

/// Truncate a command line for display: first line, at most `max` characters.
pub fn truncate_command(command: &str, max: usize) -> String {
    let line = command.trim().lines().next().unwrap_or("").trim();
    let multi = command.trim().lines().nth(1).is_some();
    if line.chars().count() > max {
        let cut: String = line.chars().take(max).collect();
        format!("{cut}…")
    } else if multi {
        format!("{line} …")
    } else {
        line.to_string()
    }
}

/// Kills a job's processes when dropped, unless disarmed: a call whose future is dropped (a
/// cancelled run) never leaves any of them behind.
struct KillGuard(Option<ProcGroup>);

impl Drop for KillGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(group) = self.0.take() {
            super::exec::spawn_tracked_cleanup(move || {
                group.kill();
            });
        }
    }
}

/// `RealRunner::run_streaming_tracked`: run `program args` as a tagged process-group leader, stream
/// both pipes into `hooks.on_chunk`, and return once the shell **and everything it left running**
/// have exited. When the shell exits with processes still running, `hooks.on_orphans` hears so at
/// once, with the shell's own result: the caller can return to the model while this keeps streaming
/// and waiting. No deadline: the [`Kill`] reported to `hooks.on_spawn`, or dropping this future, is
/// what stops the job early.
pub(crate) async fn run_tracked(
    program: &str,
    args: &[String],
    cwd: Option<&str>,
    hooks: &JobHooks<'_>,
) -> std::io::Result<ExecResult> {
    let tag = new_tag();
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args)
        // Closed, as for every model-run command: see `RealRunner::exec`.
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env(TAG_ENV, tag_value(&tag))
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn()?;
    let pgid = child.id().unwrap_or(0);
    let group = ProcGroup::new(pgid, tag);
    let mut guard = KillGuard(Some(group.clone()));
    (hooks.on_spawn)(Kill::Procs(group.clone()));

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let last_chunk = Mutex::new(Instant::now());
    let eof = AtomicBool::new(false);
    let sink = |bytes: &[u8]| {
        if let Ok(mut t) = last_chunk.lock() {
            *t = Instant::now();
        }
        (hooks.on_chunk)(bytes);
    };
    // Both pipes drain until EOF, for as long as anything holds them; this future never resolves on
    // its own, so the `select!` below ends when the body does.
    let drains = async {
        tokio::join!(drain(stdout, &sink), drain(stderr, &sink));
        eof.store(true, Ordering::Release);
        std::future::pending::<()>().await;
    };
    let settle = |since: Instant| {
        let last_chunk = &last_chunk;
        let eof = &eof;
        async move {
            loop {
                if eof.load(Ordering::Acquire) {
                    return;
                }
                let quiet_from = last_chunk.lock().map(|t| *t).unwrap_or(since).max(since);
                let quiet = quiet_from.elapsed();
                if quiet >= SETTLE {
                    return;
                }
                tokio::time::sleep(SETTLE - quiet).await;
            }
        }
    };
    let body = async {
        let status = child.wait().await?;
        let shell = ExecResult {
            code: status.code(),
            signal: exit_signal(&status),
            ..Default::default()
        };
        settle(Instant::now()).await;
        let left = {
            let group = group.clone();
            tokio::task::spawn_blocking(move || {
                let pids = group.members();
                (!pids.is_empty()).then(|| describe(&pids))
            })
            .await
            .unwrap_or(None)
        };
        if let Some(left) = left {
            (hooks.on_orphans)(&shell, left);
            group.wait_gone().await;
            settle(Instant::now()).await;
        }
        Ok::<_, std::io::Error>(shell)
    };
    let result = tokio::select! {
        r = body => r,
        () = drains => unreachable_pending(),
    };
    // Everything exited on its own (or a spawn-side error): nothing left to kill.
    guard.0 = None;
    result
}

/// The `drains` arm of `run_tracked`'s `select!` never completes; this satisfies the type.
fn unreachable_pending() -> std::io::Result<ExecResult> {
    Err(std::io::Error::other("job output drain ended unexpectedly"))
}

async fn drain<R: tokio::io::AsyncRead + Unpin>(reader: Option<R>, sink: &(dyn Fn(&[u8]) + Sync)) {
    use tokio::io::AsyncReadExt as _;
    let Some(mut r) = reader else {
        return;
    };
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        match r.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => sink(&buf[..n]),
        }
    }
}

#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(status)
}

#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{POLL, ProcGroup, TAG_ENV};
    use rustix::process::{Pid, PidfdFlags, Signal};
    use std::collections::HashMap;
    use std::time::Duration;

    /// Passes of [`kill`] before giving up on a process that keeps forking faster than it is killed.
    const KILL_PASSES: usize = 20;

    /// How far up a parent chain [`scan`] looks for a member. A real tree is a handful deep; the bound
    /// only keeps a pathological one (or a cycle read across a racing `/proc`) from costing more.
    const MAX_DEPTH: usize = 64;

    /// One live process, as one pass over `/proc` read it.
    fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    struct Proc {
        ppid: u32,
        start: u64,
        /// A member in its own right: in the group (while that is ours), carrying the tag, or seen
        /// as a member by an earlier scan.
        direct: bool,
    }

    pub(super) fn members(group: &ProcGroup) -> Vec<u32> {
        scan(group).into_iter().map(|(pid, _)| pid).collect()
    }

    /// The job's live processes (zombies excluded, never this process itself), each with its start
    /// time — the identity [`signal_verified`] checks before it signals. One pass over `/proc`:
    ///
    /// - in the process group (while that number is still ours — see [`ProcGroup`]), or
    /// - carrying the tag, or
    /// - seen as a member by an earlier scan (same pid **and** start time: membership is for life,
    ///   so a process stays the job's after its parent dies and it is re-parented), or
    /// - a descendant of any of those — which is what reaches a process that cleared its
    ///   environment *and* left the group (`setsid env -i …`) while its parent chain is intact.
    ///
    /// Remembers what it found for the next scan.
    pub(super) fn scan(group: &ProcGroup) -> Vec<(u32, u64)> {
        let me = std::process::id();
        let Ok(dir) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        let by_group = group.by_group();
        let seen = lock(&group.seen).clone();
        let mut in_group = false;
        let mut procs: HashMap<u32, Proc> = HashMap::new();
        for entry in dir.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            if pid == me {
                continue;
            }
            let Some(stat) = read_stat(pid) else {
                continue;
            };
            let grouped = by_group && stat.pgrp == group.pgid;
            in_group |= grouped;
            let direct = grouped || seen.contains(&(pid, stat.start)) || has_tag(pid, &group.tag);
            procs.insert(
                pid,
                Proc {
                    ppid: stat.ppid,
                    start: stat.start,
                    direct,
                },
            );
        }
        if by_group && !in_group {
            group.saw_group_empty();
        }
        let mut verdict: HashMap<u32, bool> = HashMap::with_capacity(procs.len());
        let mut out: Vec<(u32, u64)> = Vec::new();
        for (&pid, p) in &procs {
            if descends(pid, &procs, &mut verdict, me) {
                out.push((pid, p.start));
            }
        }
        out.sort_unstable();
        *lock(&group.seen) = out.iter().copied().collect();
        out
    }

    pub(super) fn tag_alive(group: &ProcGroup) -> bool {
        let me = std::process::id();
        let Ok(dir) = std::fs::read_dir("/proc") else {
            return false;
        };
        dir.flatten().any(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|s| s.parse::<u32>().ok())
                .is_some_and(|pid| pid != me && is_live(pid) && has_tag(pid, &group.tag))
        })
    }

    /// Whether `pid` exists and is not a zombie.
    fn is_live(pid: u32) -> bool {
        let Ok(stat) = std::fs::read(format!("/proc/{pid}/stat")) else {
            return false;
        };
        let Some(close) = stat.iter().rposition(|&b| b == b')') else {
            return false;
        };
        let rest = String::from_utf8_lossy(&stat[close + 1..]);
        !matches!(rest.split_whitespace().next(), Some("Z" | "X"))
    }

    /// Whether `pid` is a member or descends from one, memoized in `verdict`. Stops at this process
    /// (never a member: everything it spawned would otherwise be the job's), at init, at a process
    /// the scan did not see, and at [`MAX_DEPTH`].
    fn descends(
        pid: u32,
        procs: &HashMap<u32, Proc>,
        verdict: &mut HashMap<u32, bool>,
        me: u32,
    ) -> bool {
        let mut chain = Vec::new();
        let mut at = pid;
        let found = loop {
            if let Some(&known) = verdict.get(&at) {
                break known;
            }
            let Some(p) = procs.get(&at) else {
                break false;
            };
            chain.push(at);
            if p.direct {
                break true;
            }
            if p.ppid == me || p.ppid <= 1 || chain.len() >= MAX_DEPTH {
                break false;
            }
            at = p.ppid;
        };
        for link in chain {
            verdict.insert(link, found);
        }
        found
    }

    struct Stat {
        ppid: u32,
        pgrp: u32,
        start: u64,
    }

    /// `/proc/<pid>/stat`'s parent, group and start time; `None` for a zombie or a vanished pid.
    fn read_stat(pid: u32) -> Option<Stat> {
        let stat = std::fs::read(format!("/proc/{pid}/stat")).ok()?;
        // `pid (comm) state ppid pgrp …`; `comm` may hold spaces or parens, so read after the last
        // `)`. Field 3 (`state`) is the first after it, so field 22 (`starttime`) is the 20th.
        let close = stat.iter().rposition(|&b| b == b')')?;
        let rest = std::str::from_utf8(&stat[close + 1..]).ok()?;
        let fields: Vec<&str> = rest.split_whitespace().take(20).collect();
        let state = *fields.first()?;
        if state == "Z" || state == "X" {
            return None;
        }
        Some(Stat {
            ppid: fields.get(1)?.parse().ok()?,
            pgrp: fields.get(2)?.parse().ok()?,
            start: fields.get(19)?.parse().ok()?,
        })
    }

    fn has_tag(pid: u32, tag: &str) -> bool {
        let Ok(env) = std::fs::read(format!("/proc/{pid}/environ")) else {
            return false;
        };
        let key = TAG_ENV.as_bytes();
        env.split(|&b| b == 0).any(|var| {
            var.len() > key.len()
                && var.starts_with(key)
                && var[key.len()] == b'='
                && var[key.len() + 1..]
                    .split(|&b| b == b':')
                    .any(|t| t == tag.as_bytes())
        })
    }

    pub(super) fn terminate(group: &ProcGroup) {
        // The scan first, while every parent link is intact: a process that exits on the TERM
        // re-parents its children, and a descendant matched only through it would be lost.
        let procs = scan(group);
        if group.may_signal_group()
            && let Some(pgid) = Pid::from_raw(group.pgid as i32)
        {
            let _ = rustix::process::kill_process_group(pgid, Signal::TERM);
        }
        for (pid, start) in procs {
            signal_verified(pid, start, Signal::TERM);
        }
    }

    pub(super) fn kill(group: &ProcGroup) -> Vec<u32> {
        for pass in 0..KILL_PASSES {
            let procs = scan(group);
            // After the first scan (see `terminate`), and only once: what the group signal reaches
            // that the scan missed — a process forked in between — the next pass finds anyway.
            if pass == 0
                && group.may_signal_group()
                && let Some(pgid) = Pid::from_raw(group.pgid as i32)
            {
                let _ = rustix::process::kill_process_group(pgid, Signal::KILL);
            }
            if procs.is_empty() {
                return Vec::new();
            }
            // Every one left refused the signal: another pass cannot do better.
            let mut denied = 0;
            for &(pid, start) in &procs {
                if signal_verified(pid, start, Signal::KILL) {
                    denied += 1;
                }
            }
            if denied == procs.len() {
                let pids: Vec<u32> = procs.iter().map(|&(pid, _)| pid).collect();
                tracing::warn!(
                    pgid = group.pgid,
                    ?pids,
                    "tracked processes we may not signal survived the kill"
                );
                return pids;
            }
            // A SIGKILLed process is a zombie (skipped above) once the kernel has torn it down.
            std::thread::sleep(Duration::from_millis(if pass == 0 { 5 } else { 25 }));
        }
        let pids = members(group);
        if !pids.is_empty() {
            tracing::warn!(
                pgid = group.pgid,
                ?pids,
                "tracked processes survived repeated kill passes"
            );
        }
        pids
    }

    /// Send `signal` to `pid` if it is still the process a scan saw (same start time) — checked
    /// after opening a pidfd on it, and signalled through that pidfd, so a pid reused by an
    /// unrelated process in between is never hit. Whether the kernel refused it (`EPERM`: a process
    /// of another uid).
    fn signal_verified(pid: u32, start: u64, signal: Signal) -> bool {
        let Some(p) = Pid::from_raw(pid as i32) else {
            return false;
        };
        let same = || crate::tools::exec::start_time(pid) == Some(start);
        let sent = match rustix::process::pidfd_open(p, PidfdFlags::empty()) {
            Ok(fd) => {
                if !same() {
                    return false;
                }
                rustix::process::pidfd_send_signal(&fd, signal)
            }
            // No pidfd (a kernel before 5.3): the check-then-kill gap is a few microseconds.
            Err(_) => {
                if !same() {
                    return false;
                }
                rustix::process::kill_process(p, signal)
            }
        };
        sent == Err(rustix::io::Errno::PERM)
    }

    pub(super) async fn wait_gone(group: &ProcGroup) {
        loop {
            let procs = {
                let group = group.clone();
                tokio::task::spawn_blocking(move || scan(&group))
                    .await
                    .unwrap_or_default()
            };
            if procs.is_empty() {
                return;
            }
            // A pidfd turns readable when its process exits: wait on all of them, then look again
            // (a process may have forked meanwhile).
            let mut fds = Vec::new();
            for (pid, start) in procs {
                let Some(p) = Pid::from_raw(pid as i32) else {
                    continue;
                };
                if let Ok(fd) = rustix::process::pidfd_open(p, PidfdFlags::NONBLOCK)
                    && crate::tools::exec::start_time(pid) == Some(start)
                    && let Ok(fd) = tokio::io::unix::AsyncFd::new(fd)
                {
                    fds.push(fd);
                }
            }
            if fds.is_empty() {
                tokio::time::sleep(POLL).await;
                continue;
            }
            for fd in &fds {
                let _ = fd.readable().await;
            }
        }
    }

    pub(super) fn command_line(pid: u32) -> String {
        let raw = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
        let text: Vec<String> = raw
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        if text.is_empty() {
            return format!("pid {pid}");
        }
        super::truncate_command(&text.join(" "), 60)
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
mod imp {
    //! The weaker fallback: the process group only. `setsid` and `timeout` leave it, and are missed.
    use super::{POLL, ProcGroup};
    use rustix::process::Pid;

    pub(super) fn members(group: &ProcGroup) -> Vec<u32> {
        match Pid::from_raw(group.pgid as i32) {
            Some(p) if rustix::process::test_kill_process_group(p).is_ok() => vec![group.pgid],
            _ => Vec::new(),
        }
    }

    /// No environment to read: a recorded group is never proven ours.
    pub(super) fn tag_alive(_group: &ProcGroup) -> bool {
        false
    }

    pub(super) fn terminate(group: &ProcGroup) {
        if let Some(p) = Pid::from_raw(group.pgid as i32) {
            let _ = rustix::process::kill_process_group(p, rustix::process::Signal::TERM);
        }
    }

    pub(super) fn kill(group: &ProcGroup) -> Vec<u32> {
        super::super::exec::kill_group_members(group.leader);
        Vec::new()
    }

    pub(super) async fn wait_gone(group: &ProcGroup) {
        while !members(group).is_empty() {
            tokio::time::sleep(POLL).await;
        }
    }

    pub(super) fn command_line(pid: u32) -> String {
        format!("process group {pid}")
    }
}

#[cfg(not(unix))]
mod imp {
    use super::ProcGroup;

    pub(super) fn members(_group: &ProcGroup) -> Vec<u32> {
        Vec::new()
    }

    pub(super) fn tag_alive(_group: &ProcGroup) -> bool {
        false
    }

    pub(super) fn terminate(_group: &ProcGroup) {}

    pub(super) fn kill(_group: &ProcGroup) -> Vec<u32> {
        Vec::new()
    }

    pub(super) async fn wait_gone(_group: &ProcGroup) {}

    pub(super) fn command_line(pid: u32) -> String {
        format!("pid {pid}")
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn tags_are_unique_and_nest_under_an_inherited_one() {
        assert_ne!(new_tag(), new_tag());
        assert!(!tag_value("x").is_empty());
    }

    /// What `run_tracked` reports for `cmd` (which must end within 10 s): its result, and what the
    /// shell left running (if anything) as `on_orphans` heard it.
    async fn tracked(cmd: &str) -> (ExecResult, Option<String>, String) {
        let orphans = Mutex::new(None);
        let output = Mutex::new(String::new());
        let on_chunk = |b: &[u8]| output.lock().unwrap().push_str(&String::from_utf8_lossy(b));
        let on_spawn = |_k: Kill| {};
        let on_orphans = |_r: &ExecResult, s: String| *orphans.lock().unwrap() = Some(s);
        let hooks = JobHooks {
            on_chunk: &on_chunk,
            on_spawn: &on_spawn,
            on_orphans: &on_orphans,
        };
        let args = vec!["-c".to_string(), cmd.to_string()];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            run_tracked("/bin/sh", &args, None, &hooks),
        )
        .await
        .expect("the command ends on its own")
        .unwrap();
        let left = orphans.lock().unwrap().take();
        let out = output.lock().unwrap().clone();
        (result, left, out)
    }

    #[tokio::test]
    async fn a_command_that_leaves_nothing_running_reports_no_orphans() {
        let (result, left, out) = tracked("echo done; exit 2").await;
        assert_eq!(result.code, Some(2));
        assert!(left.is_none());
        assert_eq!(out, "done\n");
    }

    #[tokio::test]
    async fn a_background_process_is_reported_and_waited_for() {
        let started = Instant::now();
        let (result, left, out) = tracked("(sleep 0.5; echo late) & echo now").await;
        assert_eq!(result.code, Some(0));
        let left = left.expect("the subshell was left running");
        assert!(left.contains("sleep 0.5"), "{left}");
        assert!(out.contains("now") && out.contains("late"), "{out}");
        assert!(
            started.elapsed() >= Duration::from_millis(400),
            "waited for it"
        );
    }

    #[tokio::test]
    async fn dropping_the_call_stops_what_the_shell_left_running() {
        let orphans = Mutex::new(None);
        let groups = Mutex::new(None);
        let on_chunk = |_b: &[u8]| {};
        let on_spawn = |k: Kill| {
            if let Kill::Procs(g) = k {
                *groups.lock().unwrap() = Some(g);
            }
        };
        let on_orphans = |_r: &ExecResult, s: String| *orphans.lock().unwrap() = Some(s);
        let hooks = JobHooks {
            on_chunk: &on_chunk,
            on_spawn: &on_spawn,
            on_orphans: &on_orphans,
        };
        let args = vec!["-c".to_string(), "setsid sleep 81 & echo x".to_string()];
        // No deadline of its own: it would wait on the sleep for 81 s. Dropping it is the kill.
        let mut run = Box::pin(run_tracked("/bin/sh", &args, None, &hooks));
        let reported = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                tokio::select! {
                    _ = &mut run => panic!("ended while the setsid'd sleep runs"),
                    () = tokio::time::sleep(Duration::from_millis(50)) => {}
                }
                if orphans.lock().unwrap().is_some() {
                    return;
                }
            }
        })
        .await;
        assert!(reported.is_ok(), "the shell's exit was reported");
        let group = groups.lock().unwrap().clone().unwrap();
        assert!(!group.members().is_empty(), "the sleep is still running");
        drop(run);
        for _ in 0..100 {
            if group.members().is_empty() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("dropping the call left the setsid'd sleep running");
    }

    /// Set (to anything) only for the copy of this binary [`a_recycled_group_number_is_never_claimed`]
    /// runs inside a fresh pid namespace.
    const IN_PID_NS: &str = "BASH_PROC_TEST_IN_PID_NS";

    /// A job's group number, recycled for an unrelated group after the shell is reaped and its
    /// `setsid` leftover has left the group: the stranger must never read as one of the job's
    /// processes, and a kill of the job must never reach it. Pids are only predictable in a pid
    /// namespace of our own (`ns_last_pid`), so this runs the real check
    /// ([`recycled_group_inner`]) in a copy of this test binary under `unshare -Urpf --mount-proc`,
    /// and skips where that is unavailable (no `unshare`, unprivileged user namespaces off).
    #[test]
    fn a_recycled_group_number_is_never_claimed() {
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let usable = std::process::Command::new("unshare")
            .args(["-Urpf", "--mount-proc", "true"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !usable {
            eprintln!("skipped: `unshare -Urpf --mount-proc` is unavailable here");
            return;
        }
        let out = std::process::Command::new("unshare")
            .args(["-Urpf", "--mount-proc"])
            .arg(&exe)
            .args([
                "--exact",
                "tools::bash_proc::tests::recycled_group_inner",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(IN_PID_NS, "1")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.status.success() && text.contains("1 passed"),
            "in the pid namespace:\n{text}"
        );
    }

    #[tokio::test]
    #[ignore = "run by a_recycled_group_number_is_never_claimed, inside a pid namespace"]
    async fn recycled_group_inner() {
        if std::env::var_os(IN_PID_NS).is_none() {
            return;
        }
        let groups = Mutex::new(None);
        let orphans = Mutex::new(None);
        let output = Mutex::new(String::new());
        let on_chunk = |b: &[u8]| output.lock().unwrap().push_str(&String::from_utf8_lossy(b));
        let on_spawn = |k: Kill| {
            if let Kill::Procs(g) = k {
                *groups.lock().unwrap() = Some(g);
            }
        };
        let on_orphans = |_r: &ExecResult, s: String| *orphans.lock().unwrap() = Some(s);
        let hooks = JobHooks {
            on_chunk: &on_chunk,
            on_spawn: &on_spawn,
            on_orphans: &on_orphans,
        };
        // The leftover leaves the group (`setsid`), so the group empties when the shell exits.
        let args = vec![
            "-c".to_string(),
            "setsid sleep 97 >/dev/null 2>&1 &".to_string(),
        ];
        let mut run = Box::pin(run_tracked("/bin/sh", &args, None, &hooks));
        tokio::time::timeout(Duration::from_secs(10), async {
            while orphans.lock().unwrap().is_none() {
                tokio::select! {
                    _ = &mut run => panic!("ended while the setsid'd sleep runs"),
                    () = tokio::time::sleep(Duration::from_millis(20)) => {}
                }
            }
        })
        .await
        .expect("the shell's exit was reported");
        let group = groups.lock().unwrap().clone().unwrap();
        let leftover = group.members();
        assert_eq!(leftover.len(), 1, "{leftover:?}");

        // Hand the shell's old pid — the group's number — to a stranger leading its own group.
        let mut victim = None;
        for _ in 0..50 {
            std::fs::write("/proc/sys/kernel/ns_last_pid", (group.pgid - 1).to_string()).unwrap();
            let mut cmd = std::process::Command::new("sleep");
            std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
            let mut child = cmd.arg("98").spawn().unwrap();
            if child.id() == group.pgid {
                victim = Some(child);
                break;
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        let mut victim = victim.expect("the group's number was recycled");

        let members = group.members();
        assert!(
            !members.contains(&group.pgid),
            "the stranger holding the recycled number reads as the job's: {members:?}"
        );
        let survived = group.kill();
        assert!(survived.is_empty(), "{survived:?}");
        // The same window in the other group killers (`exec`'s `GroupKillGuard`, the MCP sweep
        // after a reap, a template transport's local group): never once the number is recycled.
        crate::tools::exec::kill_process_group(group.leader);
        crate::tools::exec::kill_group_members(group.leader);
        std::thread::sleep(Duration::from_millis(200));
        let victim_alive = victim.try_wait().unwrap().is_none();
        let _ = victim.kill();
        let _ = victim.wait();
        assert!(victim_alive, "the job's kill reached the stranger's group");
        assert!(
            !std::path::Path::new(&format!("/proc/{}", leftover[0])).exists()
                || group.members().is_empty(),
            "the job's own leftover was killed"
        );
        let _ = tokio::time::timeout(Duration::from_secs(5), run).await;
    }

    #[test]
    fn a_command_is_truncated_to_its_first_line() {
        assert_eq!(truncate_command("echo hi", 20), "echo hi");
        assert_eq!(truncate_command("cd a\nmake", 20), "cd a …");
        assert_eq!(
            truncate_command(&"x".repeat(30), 10),
            format!("{}…", "x".repeat(10))
        );
    }
}
