//! Sandbox-owned jobs through the exec endpoint's native operations: start held or kept, follow the
//! output, keep, kill, list, remove, wipe. Each runs against real processes on this host, behind the
//! endpoint the shared mock runs (`execd`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_agent::exec_endpoint::HttpExecRunner;
use beyond_agent::exec_endpoint::jobs::{
    JobEvent, JobId, JobScope, JobSpec, JobStatus, SandboxJobs,
};
use beyond_agent::tools::exec::CommandRunner;
use common::exec_mock::{ExecMock, MockOptions, group_alive, wait_group_gone};

/// A sandbox, its jobs rooted in a fresh directory.
struct Sandbox {
    jobs: Arc<dyn SandboxJobs>,
    scope: JobScope,
    _dir: tempfile::TempDir,
    _mock: ExecMock,
}

impl Sandbox {
    async fn new() -> Self {
        Self::with_cap(None).await
    }

    async fn with_cap(cap: Option<u64>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mock = ExecMock::start_with(
            dir.path(),
            MockOptions {
                heartbeat: Duration::from_millis(500),
                job_output_cap: cap.map_or(8 * 1024 * 1024, |c| c as usize),
                ..MockOptions::default()
            },
        )
        .await;
        let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
        let jobs = runner.sandbox_jobs().unwrap();
        let scope = JobScope::under(dir.path().join("jobs").to_str().unwrap(), "scope");
        Self {
            jobs,
            scope,
            _dir: dir,
            _mock: mock,
        }
    }

    /// The sandbox no longer has the job.
    async fn gone(&self, id: &JobId) -> bool {
        self.jobs.read(&self.scope, id, 0, 0).await.unwrap().status == JobStatus::Gone
    }

    async fn read_all(&self, id: &JobId, from: u64) -> (JobStatus, Vec<u8>, u64) {
        let mut out = Vec::new();
        let mut offset = from;
        loop {
            let c = self
                .jobs
                .read(&self.scope, id, offset, 64 * 1024)
                .await
                .unwrap();
            offset = c.offset + c.data.len() as u64;
            out.extend_from_slice(&c.data);
            if c.data.is_empty() || offset >= c.end {
                return (c.status, out, offset);
            }
        }
    }

    /// Wait (bounded) for the job to reach a status `want` accepts.
    async fn wait_status(&self, id: &JobId, want: impl Fn(JobStatus) -> bool) -> JobStatus {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let s = self.jobs.read(&self.scope, id, 0, 0).await.unwrap().status;
            if want(s) {
                return s;
            }
            assert!(Instant::now() < deadline, "still {s:?}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

fn sh(cmd: &str) -> JobSpec {
    JobSpec {
        program: "sh".into(),
        args: vec!["-c".into(), cmd.into()],
        cwd: None,
        label: cmd.into(),
    }
}

/// Everything a job reported, in order.
#[derive(Default)]
struct Heard {
    started: Option<u32>,
    output: Vec<u8>,
    shell_exit: Option<i32>,
    gaps: u64,
}

fn recorder(heard: &Mutex<Heard>) -> impl Fn(JobEvent<'_>) + Send + Sync + '_ {
    move |e| {
        let mut h = heard.lock().unwrap();
        match e {
            JobEvent::Started(pgid) => h.started = Some(pgid),
            JobEvent::Output(b) => h.output.extend_from_slice(b),
            JobEvent::Gap(n) => h.gaps += n,
            JobEvent::ShellExit(c) => h.shell_exit = Some(c),
        }
    }
}

async fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_held_job_streams_its_output_live_and_reports_its_end() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let code = sb
        .jobs
        .run_held(
            &sb.scope,
            &id,
            &sh("echo first; sleep 0.4; echo second >&2; exit 3"),
            &events,
        )
        .await
        .unwrap();
    assert_eq!(code, 3);
    {
        let h = heard.lock().unwrap();
        assert!(h.started.is_some());
        // stdout and stderr are one combined stream, in order.
        assert_eq!(h.output, b"first\nsecond\n");
        assert_eq!(h.shell_exit, None);
    }
    // The sandbox has it too, from offset 0.
    let (status, out, end) = sb.read_all(&id, 0).await;
    assert_eq!(status, JobStatus::Ended(3));
    assert_eq!(out, b"first\nsecond\n");
    assert_eq!(end, 13);
}

#[tokio::test]
async fn a_held_job_dies_with_its_hold() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let spec = sh("echo up; sleep 60");
    let run = sb.jobs.run_held(&sb.scope, &id, &spec, &events);
    tokio::select! {
        r = run => panic!("ended early: {r:?}"),
        () = wait_for("output", || !heard.lock().unwrap().output.is_empty()) => {}
    }
    // The future is dropped here: the hold is gone.
    let pgid = heard.lock().unwrap().started.unwrap();
    assert!(
        wait_group_gone(pgid, Duration::from_secs(10)).await,
        "the held job outlived its hold"
    );
    let s = sb.wait_status(&id, JobStatus::ended).await;
    assert_eq!(s, JobStatus::Ended(137), "killed, not exited");
}

#[tokio::test]
async fn a_kept_job_outlives_its_hold_and_is_read_back_from_an_offset() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let spec = sh("echo one; sleep 1; echo two; sleep 60");
    let mut run = Box::pin(sb.jobs.run_held(&sb.scope, &id, &spec, &events));
    // Boxed, not stack-pinned: dropping it below must really close the hold.
    tokio::select! {
        r = &mut run => panic!("ended early: {r:?}"),
        () = wait_for("output", || heard.lock().unwrap().output == b"one\n") => {}
    }
    // Kept while the hold is still up — as `bash` keeps a job before it lets go of anything.
    sb.jobs
        .keep(&sb.scope, &id, 7, "background", 4)
        .await
        .unwrap();
    drop(run);
    let pgid = heard.lock().unwrap().started.unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(group_alive(pgid), "a kept job died with its hold");
    let listed = sb.jobs.list(&sb.scope).await.unwrap();
    assert_eq!(listed.len(), 1);
    let j = &listed[0];
    assert_eq!(j.id, id);
    assert!(j.kept);
    assert_eq!(
        (j.num, j.kind.as_deref(), j.seen),
        (Some(7), Some("background"), 4)
    );
    assert_eq!(j.status, JobStatus::Running);
    assert_eq!(j.label, "echo one; sleep 1; echo two; sleep 60");
    assert!(j.started > 1_700_000_000);
    // From the offset the model had seen: only what came after.
    let (_, out, _) = sb.read_all(&id, j.seen).await;
    assert_eq!(out, b"two\n");
    let survivors = sb
        .jobs
        .kill(&sb.scope, std::slice::from_ref(&id), true)
        .await
        .unwrap();
    assert!(survivors.is_empty(), "{survivors:?}");
    assert!(!group_alive(pgid));
    assert!(sb.gone(&id).await, "killed and removed");
    assert_eq!(
        sb.jobs.read(&sb.scope, &id, 0, 10).await.unwrap().status,
        JobStatus::Gone
    );
}

#[tokio::test]
async fn leftovers_are_reported_at_the_shells_exit_and_the_job_ends_with_them() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let started = Instant::now();
    let code = sb
        .jobs
        .run_held(
            &sb.scope,
            &id,
            &sh("(sleep 1; echo leftover-done) & echo shell-done; exit 4"),
            &events,
        )
        .await
        .unwrap();
    assert_eq!(code, 4, "the shell's own status");
    assert!(started.elapsed() >= Duration::from_millis(900));
    let h = heard.lock().unwrap();
    assert_eq!(h.shell_exit, Some(4));
    assert_eq!(h.output, b"shell-done\nleftover-done\n");
}

#[tokio::test]
async fn a_job_started_kept_is_followed_by_polling() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    let pgid = sb
        .jobs
        .start_kept(&sb.scope, &id, &sh("echo a; sleep 0.6; echo b; exit 5"))
        .await
        .unwrap();
    assert!(pgid > 1);
    let listed = sb.jobs.list(&sb.scope).await.unwrap();
    assert!(listed[0].kept && listed[0].num.is_none());
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let code = sb.jobs.follow(&sb.scope, &id, 0, &events).await.unwrap();
    assert_eq!(code, 5);
    assert_eq!(heard.lock().unwrap().output, b"a\nb\n");
}

#[tokio::test]
async fn polling_reports_leftovers_and_follows_them_to_the_end() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    sb.jobs
        .start_kept(&sb.scope, &id, &sh("(sleep 1.5; echo late) & echo early"))
        .await
        .unwrap();
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let code = sb.jobs.follow(&sb.scope, &id, 0, &events).await.unwrap();
    assert_eq!(code, 0);
    let h = heard.lock().unwrap();
    assert_eq!(h.shell_exit, Some(0));
    assert_eq!(h.output, b"early\nlate\n");
}

#[tokio::test]
async fn kill_stops_a_job_that_ignores_sigterm_and_removes_it() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    let pgid = sb
        .jobs
        .start_kept(
            &sb.scope,
            &id,
            &sh("trap '' TERM; sleep 60 & sleep 60 & wait"),
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(group_alive(pgid));
    let survivors = sb
        .jobs
        .kill(&sb.scope, std::slice::from_ref(&id), true)
        .await
        .unwrap();
    assert!(survivors.is_empty(), "{survivors:?}");
    assert!(!group_alive(pgid));
    assert!(sb.gone(&id).await);
    // Idempotent: killing it again is not an error.
    assert!(
        sb.jobs
            .kill(&sb.scope, &[id], true)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn output_past_the_cap_is_deleted_and_read_back_as_a_gap_of_known_size() {
    let sb = Sandbox::with_cap(Some(64 * 1024)).await;
    let id = JobId::fresh();
    // 3 MiB at once, then quiet.
    sb.jobs
        .start_kept(
            &sb.scope,
            &id,
            &sh("head -c 3500000 /dev/zero | tr '\\0' a; echo; sleep 60"),
        )
        .await
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    let first = loop {
        let c = sb.jobs.read(&sb.scope, &id, 0, 16).await.unwrap();
        if c.offset > 0 && c.end == 3_500_001 {
            break c;
        }
        assert!(Instant::now() < deadline, "nothing was pruned");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    // What is kept is at least the cap, and the reader is told where it starts.
    assert!(first.offset >= 1024 * 1024, "{}", first.offset);
    assert!(first.end - first.offset >= 64 * 1024);
    assert_eq!(first.data, b"aaaaaaaaaaaaaaaa");
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let follow = sb.jobs.follow(&sb.scope, &id, 0, &events);
    tokio::select! {
        r = follow => panic!("ended early: {r:?}"),
        () = wait_for("the tail", || heard.lock().unwrap().output.ends_with(b"a\n")) => {}
    }
    {
        let h = heard.lock().unwrap();
        assert_eq!(h.gaps, first.offset);
        assert_eq!(h.gaps + h.output.len() as u64, 3_500_001);
    }
    sb.jobs.wipe(&sb.scope).await.unwrap();
}

#[tokio::test]
async fn wipe_kills_every_job_of_the_scope_and_no_other() {
    let sb = Sandbox::new().await;
    let other = JobScope::under(sb.scope.root(), "other");
    let mine = [JobId::fresh(), JobId::fresh()];
    let mut groups = Vec::new();
    for id in &mine {
        groups.push(
            sb.jobs
                .start_kept(&sb.scope, id, &sh("sleep 60"))
                .await
                .unwrap(),
        );
    }
    let theirs = JobId::fresh();
    let their_group = sb
        .jobs
        .start_kept(&other, &theirs, &sh("sleep 60"))
        .await
        .unwrap();
    assert_eq!(sb.jobs.list(&sb.scope).await.unwrap().len(), 2);
    assert_eq!(
        sb.jobs.list(&other).await.unwrap().len(),
        1,
        "scopes are separate"
    );
    assert!(sb.jobs.wipe(&sb.scope).await.unwrap().is_empty());
    for g in groups {
        assert!(!group_alive(g));
    }
    assert!(sb.jobs.list(&sb.scope).await.unwrap().is_empty());
    assert!(group_alive(their_group), "another scope's job was touched");
    sb.jobs.wipe(&other).await.unwrap();
    assert!(!group_alive(their_group));
}

#[tokio::test]
async fn remove_deletes_only_jobs_whose_processes_are_gone() {
    let sb = Sandbox::new().await;
    let running = JobId::fresh();
    let done = JobId::fresh();
    let g = sb
        .jobs
        .start_kept(&sb.scope, &running, &sh("sleep 60"))
        .await
        .unwrap();
    sb.jobs
        .start_kept(&sb.scope, &done, &sh("true"))
        .await
        .unwrap();
    sb.wait_status(&done, JobStatus::ended).await;
    sb.jobs
        .remove(&sb.scope, &[running.clone(), done.clone()])
        .await
        .unwrap();
    let left: Vec<JobId> = sb
        .jobs
        .list(&sb.scope)
        .await
        .unwrap()
        .into_iter()
        .map(|j| j.id)
        .collect();
    assert_eq!(
        left,
        vec![running.clone()],
        "a running job is never removed"
    );
    assert!(group_alive(g));
    sb.jobs.kill(&sb.scope, &[running], true).await.unwrap();
}

#[tokio::test]
async fn a_job_runs_in_its_directory_and_a_bad_one_fails_inside_the_job() {
    let sb = Sandbox::new().await;
    let work = tempfile::tempdir().unwrap();
    std::fs::write(work.path().join("marker.txt"), "x").unwrap();
    let id = JobId::fresh();
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let mut spec = sh("ls");
    spec.cwd = Some(work.path().to_str().unwrap().to_string());
    assert_eq!(
        sb.jobs
            .run_held(&sb.scope, &id, &spec, &events)
            .await
            .unwrap(),
        0
    );
    assert_eq!(heard.lock().unwrap().output, b"marker.txt\n");
    let id = JobId::fresh();
    spec.cwd = Some("/definitely/not/here".into());
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    assert_eq!(
        sb.jobs
            .run_held(&sb.scope, &id, &spec, &events)
            .await
            .unwrap(),
        1
    );
    assert!(
        String::from_utf8_lossy(&heard.lock().unwrap().output).contains("/definitely/not/here"),
        "the shell says why"
    );
}

#[tokio::test]
async fn a_job_id_that_already_exists_is_not_started_twice() {
    let sb = Sandbox::new().await;
    let id = JobId::fresh();
    sb.jobs
        .start_kept(&sb.scope, &id, &sh("sleep 60"))
        .await
        .unwrap();
    assert!(
        sb.jobs
            .start_kept(&sb.scope, &id, &sh("echo no"))
            .await
            .is_err()
    );
    let heard = Mutex::new(Heard::default());
    let events = recorder(&heard);
    let err = sb
        .jobs
        .run_held(&sb.scope, &id, &sh("echo no"), &events)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("did not start"), "{err}");
    assert!(heard.lock().unwrap().started.is_none());
    sb.jobs.wipe(&sb.scope).await.unwrap();
}
