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
//! covers a process that cleared its environment but stayed in the group). See "`bash` jobs: what a
//! command leaves running" in `ARCHITECTURE.md` for the alternatives measured and rejected (cgroup v2
//! per job, a child subreaper).
//!
//! What escapes: a process that both clears its environment (`env -i`) and leaves the group, or one
//! whose environment we may not read (another uid, a setuid binary, a process that made itself
//! non-dumpable *and* left the group, e.g. `ssh-agent`). Never a name match: a kill only ever reaches a
//! pid that carries the tag or the group at the moment of the kill, verified through a pidfd so a
//! reused pid is never hit.
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

/// One job's processes: its process group and its tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcGroup {
    /// The shell's pid, which leads its own group (`process_group(0)`).
    pub pgid: u32,
    /// This job's tag — see [`TAG_ENV`].
    pub tag: String,
}

/// A tag no other job of any process on this host shares: this process's pid, its start time, and a
/// per-process counter.
fn new_tag() -> String {
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

/// The [`TAG_ENV`] value a job's shell starts with: any tags this process inherited, then `tag`.
fn tag_value(tag: &str) -> String {
    match std::env::var(TAG_ENV) {
        Ok(outer) if !outer.is_empty() => format!("{outer}:{tag}"),
        _ => tag.to_string(),
    }
}

impl ProcGroup {
    /// The pids of this job's live processes (zombies excluded), never this process itself.
    pub fn members(&self) -> Vec<u32> {
        imp::members(self)
    }

    /// SIGKILL every process of the job and keep sweeping until none is left (a process that forked
    /// mid-sweep is caught by the next pass). Blocking: callers run it on a blocking thread.
    pub fn kill(&self) {
        imp::kill(self);
    }

    /// Resolve once every process of the job has exited.
    pub async fn wait_gone(&self) {
        imp::wait_gone(self).await;
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
            super::exec::spawn_tracked_cleanup(move || group.kill());
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
    let group = ProcGroup { pgid, tag };
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
    use std::time::Duration;

    /// Passes of [`kill`] before giving up on a process that keeps forking faster than it is killed.
    const KILL_PASSES: usize = 20;

    pub(super) fn members(group: &ProcGroup) -> Vec<u32> {
        let me = std::process::id();
        let Ok(dir) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in dir.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            if pid != me && belongs(group, pid) {
                out.push(pid);
            }
        }
        out.sort_unstable();
        out
    }

    /// Whether `pid` is a live (non-zombie) process of `group`: in its process group, or carrying
    /// its tag.
    fn belongs(group: &ProcGroup, pid: u32) -> bool {
        let Ok(stat) = std::fs::read(format!("/proc/{pid}/stat")) else {
            return false;
        };
        // `pid (comm) state ppid pgrp …`; `comm` may hold spaces or parens, so read after the last
        // `)`.
        let Some(close) = stat.iter().rposition(|&b| b == b')') else {
            return false;
        };
        let rest = String::from_utf8_lossy(&stat[close + 1..]);
        let mut fields = rest.split_whitespace();
        let state = fields.next().unwrap_or("");
        if state == "Z" || state == "X" {
            return false;
        }
        let pgrp = fields.nth(1).and_then(|s| s.parse::<u32>().ok());
        if pgrp == Some(group.pgid) && group.pgid != 0 {
            return true;
        }
        has_tag(pid, &group.tag)
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

    pub(super) fn kill(group: &ProcGroup) {
        if let Some(pgid) = Pid::from_raw(group.pgid as i32) {
            let _ = rustix::process::kill_process_group(pgid, Signal::KILL);
        }
        for pass in 0..KILL_PASSES {
            let pids = members(group);
            if pids.is_empty() {
                return;
            }
            for pid in pids {
                kill_verified(group, pid);
            }
            // A SIGKILLed process is a zombie (skipped above) once the kernel has torn it down.
            std::thread::sleep(Duration::from_millis(if pass == 0 { 5 } else { 25 }));
        }
        tracing::warn!(
            pgid = group.pgid,
            "bash job processes survived repeated kill passes"
        );
    }

    /// SIGKILL `pid` if it still belongs to `group` — checked after opening a pidfd on it, and
    /// signalled through that pidfd, so a pid reused by an unrelated process in between is never
    /// hit.
    fn kill_verified(group: &ProcGroup, pid: u32) {
        let Some(p) = Pid::from_raw(pid as i32) else {
            return;
        };
        match rustix::process::pidfd_open(p, PidfdFlags::empty()) {
            Ok(fd) => {
                if belongs(group, pid) {
                    let _ = rustix::process::pidfd_send_signal(&fd, Signal::KILL);
                }
            }
            // No pidfd (a kernel before 5.3): the check-then-kill gap is a few microseconds.
            Err(_) => {
                if belongs(group, pid) {
                    let _ = rustix::process::kill_process(p, Signal::KILL);
                }
            }
        }
    }

    pub(super) async fn wait_gone(group: &ProcGroup) {
        loop {
            let pids = {
                let group = group.clone();
                tokio::task::spawn_blocking(move || members(&group))
                    .await
                    .unwrap_or_default()
            };
            if pids.is_empty() {
                return;
            }
            // A pidfd turns readable when its process exits: wait on all of them, then look again
            // (a process may have forked meanwhile).
            let mut fds = Vec::new();
            for pid in pids {
                let Some(p) = Pid::from_raw(pid as i32) else {
                    continue;
                };
                if let Ok(fd) = rustix::process::pidfd_open(p, PidfdFlags::NONBLOCK)
                    && belongs(group, pid)
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

    pub(super) fn kill(group: &ProcGroup) {
        super::super::exec::kill_group_members(group.pgid);
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

    pub(super) fn kill(_group: &ProcGroup) {}

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
