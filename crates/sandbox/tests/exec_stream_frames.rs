//! Exec protocol streaming at the runner: frames reach the sink in order, arbitrary bytes survive, output
//! is bounded, every final frame maps onto the result — and an endpoint that has not *proved* it
//! streams is driven exactly as one-shot always was.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_sandbox::exec::CommandRunner;
use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_test_support::exec_mock::{ExecMock, MockOptions, Protocol, StreamEnd};

fn sh(script: &str) -> Vec<String> {
    vec!["-c".to_string(), script.to_string()]
}

/// One sink call: when it arrived, and its bytes.
type Call = (Instant, Vec<u8>);

/// Every sink call, in order.
#[derive(Default, Clone)]
struct Sink(Arc<Mutex<Vec<Call>>>);

impl Sink {
    fn bytes(&self) -> Vec<u8> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .flat_map(|(_, b)| b.clone())
            .collect()
    }
    fn calls(&self) -> Vec<Call> {
        self.0.lock().unwrap().clone()
    }
}

async fn stream(
    runner: &HttpExecRunner,
    script: &str,
    timeout: Duration,
) -> (beyond_sandbox::exec::ExecResult, Sink) {
    let sink = Sink::default();
    let s = sink.clone();
    let on_chunk = move |b: &[u8]| s.0.lock().unwrap().push((Instant::now(), b.to_vec()));
    let res = runner
        .run_streaming("sh", &sh(script), None, timeout, &on_chunk)
        .await
        .unwrap();
    (res, sink)
}

async fn mock(protocol: Protocol) -> (ExecMock, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(
        dir.path(),
        MockOptions {
            protocol,
            ..MockOptions::default()
        },
    )
    .await;
    (mock, dir)
}

#[tokio::test]
async fn frames_reach_the_sink_in_order_as_the_command_produces_them() {
    let (mock, _dir) = mock(Protocol::Streaming).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let started = Instant::now();
    let (res, sink) = stream(
        &runner,
        "printf one; sleep 0.4; printf two >&2; sleep 0.4; printf three",
        Duration::from_secs(10),
    )
    .await;
    let finished = Instant::now();

    assert_eq!(res.code, Some(0), "{res:?}");
    assert_eq!(sink.bytes(), b"onetwothree", "both streams, in order");
    assert!(
        res.stdout.is_empty() && res.stderr.is_empty(),
        "streamed output is the sink's, never repeated in the result: {res:?}"
    );
    // Live, not replayed: the first chunk landed well before the command finished.
    let first = sink.calls()[0].0;
    assert!(
        finished.duration_since(first) >= Duration::from_millis(500),
        "the first frame arrived {:?} into a {:?} command — it was not streamed",
        first.duration_since(started),
        finished.duration_since(started)
    );
}

#[tokio::test]
async fn arbitrary_bytes_survive_the_wire() {
    let (mock, _dir) = mock(Protocol::Streaming).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let (res, sink) = stream(
        &runner,
        r"printf '\377\000\001\033['",
        Duration::from_secs(10),
    )
    .await;
    assert_eq!(res.code, Some(0));
    assert_eq!(sink.bytes(), [0xff, 0x00, 0x01, 0x1b, b'[']);
}

#[tokio::test]
async fn large_output_arrives_whole_in_bounded_frames() {
    let (mock, _dir) = mock(Protocol::Streaming).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let total = 5 * 1024 * 1024;
    let (res, sink) = stream(
        &runner,
        &format!("head -c {total} /dev/zero | tr '\\0' a"),
        Duration::from_secs(60),
    )
    .await;
    assert_eq!(res.code, Some(0));
    let calls = sink.calls();
    let biggest = calls.iter().map(|(_, b)| b.len()).max().unwrap();
    assert!(
        biggest <= 64 * 1024,
        "a frame carried {biggest} bytes; the spec caps it at 64 KiB"
    );
    assert_eq!(
        calls.iter().map(|(_, b)| b.len()).sum::<usize>(),
        total,
        "every byte reaches the sink — bounding memory is the sink's job (bash keeps a tail), not \
         a reason for the runner to drop any"
    );
}

#[tokio::test]
async fn every_final_frame_maps_onto_the_result() {
    let (mock, _dir) = mock(Protocol::Streaming).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();

    let (res, _) = stream(&runner, "exit 3", Duration::from_secs(10)).await;
    assert_eq!(
        (res.code, res.signal, res.timed_out),
        (Some(3), None, false)
    );

    let (res, _) = stream(&runner, "kill -KILL $$", Duration::from_secs(10)).await;
    assert_eq!(
        (res.code, res.signal),
        (None, Some(9)),
        "a signal is a signal, not exit 137: {res:?}"
    );

    let started = Instant::now();
    let (res, _) = stream(
        &runner,
        "echo timeout-probe; sleep 30",
        Duration::from_millis(500),
    )
    .await;
    assert!(res.timed_out, "{res:?}");
    assert!(started.elapsed() < Duration::from_secs(10));
    let s = mock.streams_running("timeout-probe").pop().unwrap();
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::TimedOut),
        "the endpoint enforced the timeout, not a client-side abort"
    );
    assert!(beyond_test_support::exec_mock::wait_group_gone(s.pgid, Duration::from_secs(5)).await);
}

#[tokio::test]
async fn a_stream_asks_for_the_stream_and_a_plain_run_does_not() {
    let (mock, _dir) = mock(Protocol::Streaming).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    stream(&runner, "true", Duration::from_secs(10)).await;
    assert_eq!(mock.bodies().pop().unwrap()["stream"], true);
    runner
        .run("true", &[], None, Duration::from_secs(10))
        .await
        .unwrap();
    let last = mock.bodies().pop().unwrap();
    assert!(
        last.get("stream").is_none(),
        "`run` (the filesystem backend's path) must not ask to stream: {last}"
    );
    // No request but the two: nothing is probed first.
    assert_eq!(mock.bodies().len(), 2);
}
