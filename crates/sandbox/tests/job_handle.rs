//! The Rust SDK's job handle against a scripted endpoint that sends the mark frames
//! (`started{mark}`, `attached{mark,from}`): what it asks for after a cut, what it hands the sink, and
//! that it never gives up on a front that is down. Every decision is the client kernel's
//! (`job_client::step`); these tests drive the shell around it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_sandbox::exec_endpoint::jobs::{
    JobEnd, JobError, JobEvent, JobId, JobScope, JobSpec, SandboxJobs,
};
use beyond_sandbox::exec_endpoint::jobs_native::NativeJobs;
use beyond_test_support::job_front::{JobFront, Reply, Script};
use serde_json::Value;

/// E2B's answer for a sandbox it no longer has (captured from a deleted VM), for this endpoint's sandbox.
const E2B_GONE: &str = r#"{"sandboxId":"abc123","message":"The sandbox was not found","code":502}"#;

const MARK: &str = "00ff00ff00ff00ff1a";

fn b64(s: &str) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(s)
}

fn out(s: &str) -> String {
    format!(r#"{{"stdout":"{}"}}"#, b64(s))
}

fn started() -> String {
    format!(r#"{{"started":4242,"mark":"{MARK}","stdin":false}}"#)
}

fn attached(from: u64) -> String {
    format!(r#"{{"attached":{{"mark":"{MARK}","from":{from}}}}}"#)
}

fn spec() -> JobSpec {
    JobSpec {
        program: "sh".into(),
        args: vec!["-c".into(), "true".into()],
        cwd: None,
        label: "t".into(),
        env: Vec::new(),
        stdin_open: false,
    }
}

#[derive(Default)]
struct Sink(Mutex<Vec<String>>);

impl Sink {
    fn on(&self, e: JobEvent<'_>) {
        let line = match e {
            JobEvent::Started(..) => "started".to_string(),
            JobEvent::Attached { .. } => "attached".to_string(),
            JobEvent::Output(b, _) => format!("out:{}", String::from_utf8_lossy(b)),
            JobEvent::Gap(n) => format!("gap:{n}"),
            JobEvent::ShellExit(c) => format!("shell:{c}"),
        };
        self.0.lock().unwrap().push(line);
    }
    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

async fn run(front: &JobFront, div: u64) -> (std::io::Result<JobEnd>, Vec<String>) {
    // The endpoint's URL names the sandbox, as E2B's does (`https://8081-<id>.e2b.app/exec`).
    let runner = HttpExecRunner::new(&front.url)
        .unwrap()
        .with_sandbox_id("abc123")
        .with_time_divisor(div);
    let jobs = NativeJobs::over(&runner);
    let sink = Sink::default();
    let r = jobs
        .run_held(
            &JobScope::named("s1").unwrap(),
            &JobId::parse("j1").unwrap(),
            &spec(),
            &|e| sink.on(e),
        )
        .await;
    (r, sink.lines())
}

fn script(f: impl Fn(usize, &Value) -> Reply + Send + Sync + 'static) -> Script {
    Arc::new(f)
}

#[tokio::test]
async fn a_cut_stream_is_reattached_at_its_offset_with_its_mark() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        _ => Reply::Stream(vec![attached(5), out("world"), r#"{"exit_code":0}"#.into()]),
    }))
    .await;
    let (r, lines) = run(&front, 1000).await;
    assert_eq!(r.unwrap(), JobEnd::Exit(0));
    assert_eq!(lines, ["started", "out:hello", "attached", "out:world"]);
    let seen = front.seen.lock().unwrap();
    assert_eq!(seen[1]["op"], "attach");
    assert_eq!(seen[1]["offset"], 5);
    assert_eq!(seen[1]["mark"], MARK);
}

#[tokio::test]
async fn bytes_the_endpoint_no_longer_holds_are_a_gap_before_the_next_output() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        _ => Reply::Stream(vec![attached(8), out("x"), r#"{"exit_code":3}"#.into()]),
    }))
    .await;
    let (r, lines) = run(&front, 1000).await;
    assert_eq!(r.unwrap(), JobEnd::Exit(3));
    assert_eq!(
        lines,
        ["started", "out:hello", "attached", "gap:3", "out:x"]
    );
}

#[tokio::test]
async fn a_start_lost_before_its_echo_is_sent_again_with_the_same_key() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![]),
        _ => Reply::Stream(vec![started(), out("a"), r#"{"exit_code":0}"#.into()]),
    }))
    .await;
    let (r, lines) = run(&front, 1000).await;
    assert_eq!(r.unwrap(), JobEnd::Exit(0));
    assert_eq!(lines, ["started", "out:a"]);
    let seen = front.seen.lock().unwrap();
    assert_eq!(seen[0]["op"], "start");
    assert_eq!(seen[1]["op"], "start");
    assert_eq!(seen[0]["idempotency_key"], seen[1]["idempotency_key"]);
    assert!(seen[0]["idempotency_key"].as_str().unwrap().len() >= 16);
}

#[tokio::test]
async fn an_endpoint_that_sends_no_mark_is_refused_not_trusted() {
    let front = JobFront::start(script(|_, _| {
        Reply::Stream(vec![r#"{"started":4242,"stdin":false}"#.into(), out("x")])
    }))
    .await;
    let (r, lines) = run(&front, 1000).await;
    let e = r.unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{e}");
    assert!(lines.is_empty());
}

#[tokio::test]
async fn a_job_that_is_gone_is_not_retried() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        _ => Reply::Status(404, "no such job".into()),
    }))
    .await;
    let (r, _) = run(&front, 1000).await;
    let e = r.unwrap_err();
    assert!(
        matches!(JobError::of(&e), Some(JobError::Gone { .. })),
        "{e}"
    );
    assert_eq!(front.seen.lock().unwrap().len(), 2);
}

/// The front is down for far longer than three minutes (about twenty on the handle's clock, which a
/// test runs 2000 times faster) and the command's output and end still arrive.
#[tokio::test]
async fn a_front_that_is_down_for_minutes_does_not_end_the_handle() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        1..=60 => Reply::Down,
        _ => Reply::Stream(vec![attached(5), out("world"), r#"{"exit_code":0}"#.into()]),
    }))
    .await;
    let (r, lines) = run(&front, 2000).await;
    assert_eq!(r.unwrap(), JobEnd::Exit(0));
    assert_eq!(lines, ["started", "out:hello", "attached", "out:world"]);
}

/// The backend, which is up, says the sandbox is not there (E2B: a 502 whose body says so): that is
/// the one answer that ends a handle for an unreachable sandbox, on its first sight.
#[tokio::test]
async fn an_answer_that_the_sandbox_is_gone_ends_the_handle_at_once() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        _ => Reply::Status(502, E2B_GONE.into()),
    }))
    .await;
    let (r, lines) = run(&front, 1000).await;
    let e = r.unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "{e}");
    assert!(e.to_string().contains("sandbox no longer exists"), "{e}");
    assert_eq!(lines, ["started", "out:hello"]);
    assert_eq!(front.seen.lock().unwrap().len(), 2, "never asked again");
}

/// A 502 that does not say so is a busy front, however many times it comes: the handle waits.
#[tokio::test]
async fn a_bare_502_is_a_busy_front_not_a_gone_sandbox() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        1..=30 => Reply::Status(502, "bad gateway".into()),
        _ => Reply::Stream(vec![attached(5), out("world"), r#"{"exit_code":0}"#.into()]),
    }))
    .await;
    let (r, lines) = run(&front, 2000).await;
    assert_eq!(r.unwrap(), JobEnd::Exit(0));
    assert_eq!(lines, ["started", "out:hello", "attached", "out:world"]);
}

/// The front is down but the sandbox is alive: a proxy answers 502 with some other body, or with E2B's
/// shape about another sandbox. Either is a busy front, however many times it comes.
#[tokio::test]
async fn a_502_that_is_not_the_backends_gone_answer_keeps_the_handle_waiting() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        1..=20 => Reply::Status(502, r#"{"message":"upstream connect error"}"#.into()),
        21..=40 => Reply::Status(
            502,
            r#"{"sandboxId":"other","message":"The sandbox was not found","code":502}"#.into(),
        ),
        _ => Reply::Stream(vec![attached(5), out("world"), r#"{"exit_code":0}"#.into()]),
    }))
    .await;
    let (r, lines) = run(&front, 2000).await;
    assert_eq!(r.unwrap(), JobEnd::Exit(0));
    assert_eq!(lines, ["started", "out:hello", "attached", "out:world"]);
}

/// Beyond's front says a session has ended with a 401 whose code says so: the job's sandbox is gone. A
/// 401 without it is a refused credential, which also ends the handle, as `Auth`.
#[tokio::test]
async fn a_session_that_has_ended_ends_the_handle_as_gone() {
    let front = JobFront::start(script(|i, _| match i {
        0 => Reply::Stream(vec![started(), out("hello")]),
        _ => Reply::Status(
            401,
            r#"{"code":"session_ended","message":"session ended"}"#.into(),
        ),
    }))
    .await;
    let (r, _) = run(&front, 1000).await;
    let e = r.unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "{e}");
    assert!(e.to_string().contains("session has ended"), "{e}");
}
