//! The Rust SDK's job handle against a scripted endpoint that sends the mark frames
//! (`started{mark}`, `attached{mark,from}`): what it asks for after a cut, what it hands the sink, and
//! that it never gives up on a front that is down. Every decision is the client kernel's
//! (`job_client::step`); these tests drive the shell around it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_sandbox::exec_endpoint::jobs::{JobEnd, JobError, JobEvent, JobId, JobScope, JobSpec, SandboxJobs};
use beyond_sandbox::exec_endpoint::jobs_native::NativeJobs;
use beyond_test_support::job_front::{JobFront, Reply, Script};
use serde_json::Value;

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
            JobEvent::Started(_) => "started".to_string(),
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
    let runner = HttpExecRunner::new(&front.url)
        .unwrap()
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
    assert_eq!(lines, ["started", "out:hello", "out:world"]);
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
    assert_eq!(lines, ["started", "out:hello", "gap:3", "out:x"]);
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
    assert!(matches!(JobError::of(&e), Some(JobError::Gone { .. })), "{e}");
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
    assert_eq!(lines, ["started", "out:hello", "out:world"]);
}
