//! `bash` jobs the sandbox owns (`tools::bash_sandbox`): a remote session's intended jobs outlive
//! the replica that started them, and the next replica to open the session finds them again.
//!
//! "A replica" here is a job table: dropping it is what a reaped, redeployed or crashed replica does
//! to its jobs, and a fresh table over the same sandbox scope, reattached, is the next replica. The
//! sandbox is real processes on this host behind the mock.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_core::{Background, ContentBlock, Message};
use beyond_agent::exec_endpoint::HttpExecRunner;
use beyond_agent::exec_endpoint::jobs::JobScope;
use beyond_agent::tools::bash::{Bash, BashJob};
use beyond_agent::tools::bash_jobs::{BashJobs, EndPolicy};
use beyond_agent::tools::bash_sandbox::{Reattached, reattach, reattach_later};
use beyond_agent::tools::exec::CommandRunner;
use beyond_agent::tools::fs::FsBackend;
use beyond_agent::tools::fs::shell::{Capabilities, ShellFs};
use common::exec_mock::{ExecMock, MockOptions};
use common::remote_bash::{call, job_number, kill_job};
use serde_json::json;

/// A sandbox, and the scope one session's jobs live under in it.
struct Sandbox {
    runner: Arc<dyn CommandRunner>,
    scope: JobScope,
    _dir: tempfile::TempDir,
    _mock: ExecMock,
}

impl Sandbox {
    async fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mock = ExecMock::start_with(
            dir.path(),
            MockOptions {
                heartbeat: Duration::from_millis(500),
                ..MockOptions::default()
            },
        )
        .await;
        let runner: Arc<dyn CommandRunner> = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
        let scope = JobScope::under(dir.path().join("jobs").to_str().unwrap(), "session");
        Self {
            runner,
            scope,
            _dir: dir,
            _mock: mock,
        }
    }

    /// A replica serving the session: a job table over the session's scope, and its tools.
    fn replica(&self, policy: EndPolicy) -> Replica {
        let jobs = Arc::new(BashJobs::new(policy));
        jobs.set_scope(Some(self.scope.clone()));
        let backend: Arc<dyn FsBackend> = Arc::new(ShellFs::with_capabilities(
            self.runner.clone(),
            Capabilities::default(),
        ));
        let bash = Arc::new(
            Bash::with_runner(self.runner.clone())
                .with_backend(backend)
                .with_jobs(jobs.clone())
                .with_idle_yield(Some(Duration::from_secs(30))),
        );
        let job = BashJob::new(bash.clone());
        Replica { jobs, bash, job }
    }

    /// How many jobs the sandbox still has for the session.
    async fn jobs_left(&self) -> usize {
        self.runner
            .clone()
            .sandbox_jobs()
            .unwrap()
            .list(&self.scope)
            .await
            .unwrap()
            .len()
    }

    /// Kill whatever a failed assertion left behind.
    async fn wipe(&self) {
        let _ = self
            .runner
            .clone()
            .sandbox_jobs()
            .unwrap()
            .wipe(&self.scope)
            .await;
    }
}

struct Replica {
    jobs: Arc<BashJobs>,
    bash: Arc<Bash>,
    job: BashJob,
}

impl Replica {
    /// The replica goes away without ending the session: a reap, a deploy, a crash.
    fn vanish(self) {
        drop(self);
    }
}

/// The pid a command printed as `pid N`.
fn printed_pid(out: &str) -> u32 {
    out.lines()
        .find_map(|l| l.strip_prefix("pid "))
        .unwrap_or_else(|| panic!("no pid in {out}"))
        .trim()
        .parse()
        .unwrap()
}

fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| s.rsplit_once(')').map(|(_, r)| r.trim_start().to_string()))
        .is_some_and(|r| !r.starts_with('Z'))
}

async fn wait_dead(pid: u32) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(pid) {
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    true
}

async fn notice(jobs: &BashJobs) -> String {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(n) = jobs.take_notices() {
            return n;
        }
        assert!(Instant::now() < deadline, "no notice");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn a_background_job_outlives_its_replica_and_the_next_one_takes_it_over() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let out = call(
        a.bash.as_ref(),
        json!({
            "command": "echo \"pid $$\"; echo before; sleep 2; echo after; sleep 300",
            "run_in_background": true,
        }),
    )
    .await
    .unwrap();
    assert!(out.contains("Running in the background as job 1"), "{out}");
    assert!(
        out.contains(
            "it keeps running in the sandbox until it exits, you stop it, or the session ends"
        ),
        "the lifetime is the sandbox's: {out}"
    );
    let pid = printed_pid(&out);
    a.vanish();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(alive(pid), "the job died with its replica");

    let b = sb.replica(EndPolicy::KeepRunning);
    let found = reattach(&b.jobs, sb.runner.clone()).await;
    assert_eq!(found.running, vec![1], "{found:?}");
    // Its output picks up after what the model saw at the start.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let peek = call(&b.job, json!({ "job": 1, "timeout_ms": 0 }))
        .await
        .unwrap();
    assert!(peek.contains("after"), "{peek}");
    assert!(!peek.contains("before"), "already seen: {peek}");
    // A new job is numbered past it.
    let next = call(
        b.bash.as_ref(),
        json!({ "command": "sleep 300", "run_in_background": true }),
    )
    .await
    .unwrap();
    assert_eq!(job_number(&next), Some(2), "{next}");
    let killed = kill_job(&b.job, 1).await.unwrap();
    assert!(killed.contains("Job 1 (`"), "{killed}");
    assert!(wait_dead(pid).await, "the kill reached the sandbox");
    tokio::task::spawn_blocking({
        let jobs = b.jobs.clone();
        move || jobs.stop_all()
    })
    .await
    .unwrap();
    assert_eq!(sb.jobs_left().await, 0, "killed jobs leave no directory");
}

/// A reattach deferred to the first use (`reattach_later`): nothing is read until a job is asked
/// about, and then the answer is already the truth — a peek straight away shows what the job
/// printed while nobody was attached, not an empty follow that has not caught up yet.
#[tokio::test]
async fn a_deferred_reattach_reads_the_jobs_back_when_one_is_asked_about() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let out = call(
        a.bash.as_ref(),
        json!({
            "command": "echo \"pid $$\"; echo before; sleep 2; echo after; sleep 300",
            "run_in_background": true,
        }),
    )
    .await
    .unwrap();
    let pid = printed_pid(&out);
    a.vanish();
    tokio::time::sleep(Duration::from_millis(1000)).await;

    let b = sb.replica(EndPolicy::KeepRunning);
    reattach_later(&b.jobs, sb.runner.clone());
    assert!(b.jobs.reattach_pending());
    // `after` is printed while the reattach is still deferred.
    tokio::time::sleep(Duration::from_millis(2000)).await;
    assert!(b.jobs.reattach_pending(), "nothing reads it back unasked");
    let peek = call(&b.job, json!({ "job": 1, "timeout_ms": 0 }))
        .await
        .unwrap();
    assert!(!b.jobs.reattach_pending(), "read back by the ask");
    assert!(peek.contains("after"), "{peek}");
    assert!(!peek.contains("before"), "already seen: {peek}");
    let killed = kill_job(&b.job, 1).await.unwrap();
    assert!(killed.contains("Job 1 (`"), "{killed}");
    assert!(wait_dead(pid).await, "the kill reached the sandbox");
    sb.wipe().await;
}

#[tokio::test]
async fn a_foreground_command_dies_with_its_replica() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let out = call(
        a.bash.as_ref(),
        json!({ "command": "echo \"pid $$\"; sleep 300", "timeout_ms": 500 }),
    )
    .await
    .unwrap();
    assert!(out.contains("still running as job 1"), "{out}");
    let pid = printed_pid(&out);
    a.vanish();
    assert!(
        wait_dead(pid).await,
        "a yielded command outlived its replica"
    );
    let b = sb.replica(EndPolicy::KeepRunning);
    assert_eq!(
        reattach(&b.jobs, sb.runner.clone()).await,
        Reattached::default()
    );
    assert_eq!(sb.jobs_left().await, 0, "the stale directory is gone");
}

#[tokio::test]
async fn an_adopted_leftover_outlives_its_replica() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let out = call(
        a.bash.as_ref(),
        json!({ "command": "sleep 300 & echo \"pid $!\"" }),
    )
    .await
    .unwrap();
    assert!(out.contains("tracked as job 1"), "{out}");
    assert!(out.contains("in the sandbox"), "{out}");
    let pid = printed_pid(&out);
    a.vanish();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(alive(pid), "the adopted leftover died with its replica");
    let b = sb.replica(EndPolicy::KeepRunning);
    let found = reattach(&b.jobs, sb.runner.clone()).await;
    assert_eq!(found.running, vec![1]);
    let carry = b.jobs.carry().unwrap();
    assert!(carry.contains("left running by its command"), "{carry}");
    kill_job(&b.job, 1).await.unwrap();
    assert!(wait_dead(pid).await);
}

#[tokio::test]
async fn a_job_that_ends_while_detached_is_noticed_once_on_the_next_attach() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let out = call(
        a.bash.as_ref(),
        json!({
            "command": "echo started; sleep 2; echo finished-while-away; exit 3",
            "run_in_background": true,
        }),
    )
    .await
    .unwrap();
    assert!(out.contains("job 1"), "{out}");
    a.vanish();
    tokio::time::sleep(Duration::from_secs(3)).await;

    let b = sb.replica(EndPolicy::KeepRunning);
    let found = reattach(&b.jobs, sb.runner.clone()).await;
    assert_eq!(found.ended, vec![1], "{found:?}");
    let n = notice(&b.jobs).await;
    assert!(n.starts_with("[job 1 exited with code 3 after "), "{n}");
    assert!(n.contains("finished-while-away"), "{n}");
    assert!(!n.lines().any(|l| l == "started"), "shown already: {n}");
    // Its directory goes once the report is persisted (the host collects then).
    b.jobs.collect_garbage().await;
    assert_eq!(sb.jobs_left().await, 0);
    b.vanish();
    // So the replica after that has nothing to say about it.
    let c = sb.replica(EndPolicy::KeepRunning);
    assert_eq!(
        reattach(&c.jobs, sb.runner.clone()).await,
        Reattached::default()
    );
    assert!(c.jobs.take_notices().is_none(), "delivered once");
}

#[tokio::test]
async fn a_replica_that_dies_before_persisting_a_notice_leaves_it_for_the_next() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    call(
        a.bash.as_ref(),
        // Past the start's settle, so it ends as a job.
        json!({ "command": "sleep 2.5; echo done", "run_in_background": true }),
    )
    .await
    .unwrap();
    // Noticed by this replica, which dies before its run is persisted (no garbage collection).
    assert!(
        notice(&a.jobs)
            .await
            .starts_with("[job 1 exited with code 0")
    );
    a.vanish();
    let b = sb.replica(EndPolicy::KeepRunning);
    assert_eq!(reattach(&b.jobs, sb.runner.clone()).await.ended, vec![1]);
    assert!(
        notice(&b.jobs)
            .await
            .starts_with("[job 1 exited with code 0")
    );
    b.jobs.collect_garbage().await;
    assert_eq!(sb.jobs_left().await, 0);
}

#[tokio::test]
async fn a_run_stops_what_an_earlier_run_of_its_session_left_in_the_sandbox() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::StopAtEnd);
    let out = call(
        a.bash.as_ref(),
        json!({ "command": "echo \"pid $$\"; sleep 300", "run_in_background": true }),
    )
    .await
    .unwrap();
    assert!(
        out.contains("it is stopped for you when the run ends"),
        "{out}"
    );
    let pid = printed_pid(&out);
    // Killed before it could stop its jobs: no exit path, nothing dropped.
    std::mem::forget(a);
    let b = sb.replica(EndPolicy::StopAtEnd);
    let found = reattach(&b.jobs, sb.runner.clone()).await;
    assert_eq!(found.stopped, vec![1], "{found:?}");
    assert!(
        wait_dead(pid).await,
        "a run's reopen stops the old run's jobs"
    );
    assert_eq!(sb.jobs_left().await, 0);
}

#[tokio::test]
async fn ending_the_transcript_stops_kept_jobs_and_wipes_the_scope() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let out = call(
        a.bash.as_ref(),
        json!({ "command": "echo \"pid $$\"; sleep 300", "run_in_background": true }),
    )
    .await
    .unwrap();
    let pid = printed_pid(&out);
    let stopped = tokio::task::spawn_blocking({
        let jobs = a.jobs.clone();
        move || jobs.stop_all()
    })
    .await
    .unwrap();
    assert_eq!(stopped.len(), 1);
    assert!(wait_dead(pid).await, "an explicit end kills a kept job");
    assert_eq!(sb.jobs_left().await, 0);
}

#[tokio::test]
async fn release_lets_go_of_kept_jobs_and_stops_the_rest() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let kept = call(
        a.bash.as_ref(),
        json!({ "command": "echo \"pid $$\"; sleep 300", "run_in_background": true }),
    )
    .await
    .unwrap();
    let yielded = call(
        a.bash.as_ref(),
        json!({ "command": "echo \"pid $$\"; sleep 301", "timeout_ms": 300 }),
    )
    .await
    .unwrap();
    let stopped = tokio::task::spawn_blocking({
        let jobs = a.jobs.clone();
        move || jobs.release()
    })
    .await
    .unwrap();
    assert_eq!(stopped.len(), 1, "only the yielded one: {stopped:?}");
    assert_eq!(stopped[0].job, 2);
    assert!(wait_dead(printed_pid(&yielded)).await);
    assert!(alive(printed_pid(&kept)), "the kept one runs on");
    sb.wipe().await;
    assert!(wait_dead(printed_pid(&kept)).await);
}

#[tokio::test]
async fn the_reopen_note_says_a_job_the_sandbox_kept_is_still_running() {
    let sb = Sandbox::new().await;
    let a = sb.replica(EndPolicy::KeepRunning);
    let started = call(
        a.bash.as_ref(),
        json!({ "command": "npm run dev; sleep 300", "run_in_background": true }),
    )
    .await
    .unwrap();
    a.vanish();
    let transcript = vec![
        Message::user("start the dev server"),
        Message::assistant(vec![ContentBlock::tool_use(
            "a",
            "bash",
            json!({ "command": "npm run dev; sleep 300", "run_in_background": true }),
        )]),
        Message::tool_results(vec![ContentBlock::ToolResult {
            tool_use_id: "a".into(),
            content: started.into(),
            is_error: false,
            images: Vec::new(),
        }]),
    ];
    let b = sb.replica(EndPolicy::KeepRunning);
    reattach(&b.jobs, sb.runner.clone()).await;
    let note = b.jobs.resume(&transcript).unwrap();
    assert!(
        note.contains("- job 1: `npm run dev; sleep 300` (still running)"),
        "{note}"
    );
    assert!(!note.contains("(not running)"), "{note}");
    assert!(note.contains("New jobs are numbered from 2."), "{note}");
    sb.wipe().await;
}
