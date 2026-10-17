//! Sandbox-owned jobs over plain exec commands (`exec_endpoint::jobs::ShellJobs`): start held or
//! kept, stream or poll the output, keep, kill, list, remove, wipe — over every transport the
//! baseline has to work on. Each runs the real scripts against real processes on this host: this
//! host's `sh` directly, the v1.2 mock, the v1.1 mock (no hold: jobs start kept and are polled), and a
//! template transport whose far side, like `docker exec`'s, lives in its own session. The live suite
//! (`exec_endpoint_live`) repeats the essentials inside CI's busybox container.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_ai_agent::exec_endpoint::jobs::{
    JobEvent, JobId, JobScope, JobSpec, JobStatus, SandboxJobs, ShellJobs,
};
use beyond_ai_agent::exec_endpoint::{HttpExecRunner, TemplateRunner};
use beyond_ai_agent::tools::exec::{CommandRunner, RealRunner};
use common::exec_mock::{ExecMock, MockOptions, Protocol, group_alive, wait_group_gone};

#[derive(Clone, Copy, Debug)]
enum Transport {
    /// This host's `sh`, no transport at all.
    Local,
    /// The conforming v1.2 mock.
    Stream,
    /// A v1.1 endpoint: nothing streams, so nothing can be held.
    V11,
    /// A template whose far side runs in its own session (`docker exec`'s shape).
    Template,
}

const HOLDING: [Transport; 3] = [Transport::Local, Transport::Stream, Transport::Template];
const ALL: [Transport; 4] = [
    Transport::Local,
    Transport::Stream,
    Transport::V11,
    Transport::Template,
];

/// A sandbox over `t`, its jobs rooted in a fresh directory.
struct Sandbox {
    jobs: Arc<dyn SandboxJobs>,
    scope: JobScope,
    dir: tempfile::TempDir,
    _mock: Option<ExecMock>,
}

impl Sandbox {
    async fn new(t: Transport) -> Self {
        Self::with_cap(t, None).await
    }

    async fn with_cap(t: Transport, cap: Option<u64>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (runner, mock): (Arc<dyn CommandRunner>, _) = match t {
            Transport::Local => (Arc::new(RealRunner), None),
            Transport::Stream | Transport::V11 => {
                let mock = ExecMock::start_with(
                    dir.path(),
                    MockOptions {
                        protocol: if matches!(t, Transport::V11) {
                            Protocol::V11
                        } else {
                            Protocol::Streaming
                        },
                        heartbeat: Duration::from_millis(500),
                        ..MockOptions::default()
                    },
                )
                .await;
                (
                    Arc::new(HttpExecRunner::new(&mock.url).unwrap()),
                    Some(mock),
                )
            }
            Transport::Template => (
                Arc::new(
                    TemplateRunner::new(vec![
                        "sh".into(),
                        "-c".into(),
                        r#"setsid "$@" & wait $!"#.into(),
                        "sh".into(),
                        "{}".into(),
                    ])
                    .unwrap(),
                ),
                None,
            ),
        };
        let mut shell = ShellJobs::new(runner);
        if let Some(cap) = cap {
            shell = shell.with_output_cap(cap);
        }
        let scope = JobScope::under(dir.path().join("jobs").to_str().unwrap(), "scope");
        Self {
            jobs: Arc::new(shell),
            scope,
            dir,
            _mock: mock,
        }
    }

    fn job_dir(&self, id: &JobId) -> std::path::PathBuf {
        self.dir.path().join("jobs/scope").join(id.as_str())
    }

    fn pgid(&self, id: &JobId) -> u32 {
        std::fs::read_to_string(self.job_dir(id).join("pgid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap()
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
async fn every_transport_can_run_jobs_and_says_whether_it_can_hold_them() {
    for t in ALL {
        let sb = Sandbox::new(t).await;
        assert!(sb.jobs.available().await, "{t:?}");
        assert_eq!(sb.jobs.holds().await, !matches!(t, Transport::V11), "{t:?}");
    }
}

#[tokio::test]
async fn a_held_job_streams_its_output_live_and_reports_its_end() {
    for t in HOLDING {
        let sb = Sandbox::new(t).await;
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
        assert_eq!(code, 3, "{t:?}");
        {
            let h = heard.lock().unwrap();
            assert!(h.started.is_some(), "{t:?}");
            // stdout and stderr are one combined stream, in order.
            assert_eq!(h.output, b"first\nsecond\n", "{t:?}");
            assert_eq!(h.shell_exit, None, "{t:?}");
        }
        // The sandbox has it too, from offset 0.
        let (status, out, end) = sb.read_all(&id, 0).await;
        assert_eq!(status, JobStatus::Ended(3), "{t:?}");
        assert_eq!(out, b"first\nsecond\n", "{t:?}");
        assert_eq!(end, 13);
    }
}

#[tokio::test]
async fn a_held_job_dies_with_its_hold() {
    for t in HOLDING {
        let sb = Sandbox::new(t).await;
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
            "{t:?}: the held job outlived its hold"
        );
        let s = sb.wait_status(&id, JobStatus::ended).await;
        assert_eq!(s, JobStatus::Ended(137), "{t:?}: killed, not exited");
    }
}

#[tokio::test]
async fn a_kept_job_outlives_its_hold_and_is_read_back_from_an_offset() {
    for t in HOLDING {
        let sb = Sandbox::new(t).await;
        let id = JobId::fresh();
        let heard = Mutex::new(Heard::default());
        let events = recorder(&heard);
        let spec = sh("echo one; sleep 1; echo two; sleep 60");
        let run = sb.jobs.run_held(&sb.scope, &id, &spec, &events);
        tokio::select! {
            r = run => panic!("ended early: {r:?}"),
            () = wait_for("output", || heard.lock().unwrap().output == b"one\n") => {}
        }
        sb.jobs
            .keep(&sb.scope, &id, 7, "background", 4)
            .await
            .unwrap();
        // ...and the hold is dropped as the select ends.
        let pgid = heard.lock().unwrap().started.unwrap();
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(group_alive(pgid), "{t:?}: a kept job died with its hold");
        let listed = sb.jobs.list(&sb.scope).await.unwrap();
        assert_eq!(listed.len(), 1, "{t:?}");
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
        assert_eq!(out, b"two\n", "{t:?}");
        let survivors = sb
            .jobs
            .kill(&sb.scope, std::slice::from_ref(&id), true)
            .await
            .unwrap();
        assert!(survivors.is_empty(), "{t:?}: {survivors:?}");
        assert!(!group_alive(pgid), "{t:?}");
        assert!(!sb.job_dir(&id).exists(), "{t:?}: killed and removed");
        assert_eq!(
            sb.jobs.read(&sb.scope, &id, 0, 10).await.unwrap().status,
            JobStatus::Gone
        );
    }
}

#[tokio::test]
async fn leftovers_are_reported_at_the_shells_exit_and_the_job_ends_with_them() {
    for t in HOLDING {
        let sb = Sandbox::new(t).await;
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
        assert_eq!(code, 4, "{t:?}: the shell's own status");
        assert!(started.elapsed() >= Duration::from_millis(900), "{t:?}");
        let h = heard.lock().unwrap();
        assert_eq!(h.shell_exit, Some(4), "{t:?}");
        assert_eq!(h.output, b"shell-done\nleftover-done\n", "{t:?}");
    }
}

#[tokio::test]
async fn a_job_started_kept_is_followed_by_polling() {
    for t in ALL {
        let sb = Sandbox::new(t).await;
        let id = JobId::fresh();
        let pgid = sb
            .jobs
            .start_kept(&sb.scope, &id, &sh("echo a; sleep 0.6; echo b; exit 5"))
            .await
            .unwrap();
        assert_eq!(pgid, sb.pgid(&id), "{t:?}");
        let listed = sb.jobs.list(&sb.scope).await.unwrap();
        assert!(listed[0].kept && listed[0].num.is_none(), "{t:?}");
        let heard = Mutex::new(Heard::default());
        let events = recorder(&heard);
        let code = sb.jobs.follow(&sb.scope, &id, 0, &events).await.unwrap();
        assert_eq!(code, 5, "{t:?}");
        assert_eq!(heard.lock().unwrap().output, b"a\nb\n", "{t:?}");
    }
}

#[tokio::test]
async fn polling_reports_leftovers_and_follows_them_to_the_end() {
    let sb = Sandbox::new(Transport::V11).await;
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
    for t in ALL {
        let sb = Sandbox::new(t).await;
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
        assert!(survivors.is_empty(), "{t:?}: {survivors:?}");
        assert!(!group_alive(pgid), "{t:?}");
        assert!(!sb.job_dir(&id).exists(), "{t:?}");
        // Idempotent: killing it again is not an error.
        assert!(
            sb.jobs
                .kill(&sb.scope, &[id], true)
                .await
                .unwrap()
                .is_empty()
        );
    }
}

#[tokio::test]
async fn output_past_the_cap_is_deleted_and_read_back_as_a_gap_of_known_size() {
    let sb = Sandbox::with_cap(Transport::Local, Some(64 * 1024)).await;
    let id = JobId::fresh();
    // 3 MiB at once, then quiet: three full 1 MiB chunks and a partial one.
    sb.jobs
        .start_kept(
            &sb.scope,
            &id,
            &sh("head -c 3500000 /dev/zero | tr '\\0' a; echo; sleep 60"),
        )
        .await
        .unwrap();
    // The keeper prunes every four seconds.
    let deadline = Instant::now() + Duration::from_secs(15);
    let first = loop {
        let c = sb.jobs.read(&sb.scope, &id, 0, 16).await.unwrap();
        if c.offset > 0 {
            break c;
        }
        assert!(Instant::now() < deadline, "nothing was pruned");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    assert_eq!(first.end, 3_500_001);
    // What is kept starts at a chunk boundary, and the reader is told where.
    assert!(first.offset >= 1024 * 1024, "{}", first.offset);
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
    let sb = Sandbox::new(Transport::Local).await;
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
    let sb = Sandbox::new(Transport::Local).await;
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
    let sb = Sandbox::new(Transport::Local).await;
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
    let sb = Sandbox::new(Transport::Local).await;
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
