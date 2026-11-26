//! Ended-job retention against the real binary (`jobs::sweep`, `Retention`): what a late client sees once an
//! ended job's output is dropped (the status and a gap) or its entry reaped, and **one test per veto that is red
//! when its protection is deleted**. Each protection has a `mut_*` feature that deletes it, and each test
//! names the feature that turns it red:
//!
//! ```sh
//! cargo test -p beyond-execd --test retention                                   # green
//! cargo test -p beyond-execd --test retention --features mut_retention_ignores_streams  # red: held_stream
//! ```
//!
//! The deadlines here are 1 ms, so every job is due from the second after it ended (`Phase::Ended.at` is
//! epoch seconds). No test waits on a hoped-for interleaving: every race is forced through a fault seam
//! (`EXECD_TEST_FAULT`), and every "a sweep had its chance" is read from execd's own debug log.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use base64::Engine as _;
use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::{Value, json};
use support::Execd;
use support::http::post;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

/// Every knob on, at its sharpest: output and entry due a second after the end, one byte and one entry of
/// ceiling. Whatever a veto protects, this would take.
const SHARP: [(&str, &str); 4] = [
    ("EXECD_JOB_OUTPUT_MS", "1"),
    ("EXECD_JOB_ENTRY_MS", "1"),
    ("EXECD_JOBS_ENDED_BYTES", "1"),
    ("EXECD_JOBS_ENDED_MAX", "1"),
];

fn addr(e: &Execd) -> SocketAddr {
    SocketAddr::V4(e.addr)
}

fn b64(v: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(v.as_str().unwrap_or(""))
        .unwrap()
}

async fn op(a: SocketAddr, job: Value) -> support::http::Answer {
    post(a, &[], &json!({ "command": "true", "job": job })).await
}

/// Start an unheld job running `sh -c script` as `id`; its first answer.
async fn start(a: SocketAddr, id: &str, script: &str) -> Value {
    let r = post(
        a,
        &[],
        &json!({ "command": "sh", "args": ["-c", script],
                 "job": { "op": "start", "scope": "s", "id": id, "hold": false } }),
    )
    .await;
    assert_eq!(
        r.status,
        200,
        "start {id}: {}",
        String::from_utf8_lossy(&r.body)
    );
    r.json()
}

async fn read(a: SocketAddr, id: &str) -> Value {
    op(
        a,
        json!({ "op": "read", "scope": "s", "id": id, "offset": 0, "max": 1 << 24 }),
    )
    .await
    .json()
}

async fn list(a: SocketAddr) -> Vec<Value> {
    op(a, json!({ "op": "list", "scope": "s" })).await.json()["jobs"]
        .as_array()
        .unwrap()
        .clone()
}

async fn until_state(a: SocketAddr, id: &str, state: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let r = read(a, id).await;
        if r["state"] == state {
            return r;
        }
        assert!(Instant::now() < deadline, "{id} never became {state}: {r}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Wait for a line of execd's log that holds every one of `parts` (and is past `from` bytes of it).
async fn until_log(e: &Execd, from: usize, any: &[&[&str]]) -> String {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let log = e.log();
        if let Some(l) = log
            .get(from..)
            .unwrap_or("")
            .lines()
            .find(|l| any.iter().any(|parts| parts.iter().all(|p| l.contains(p))))
        {
            return l.to_string();
        }
        assert!(Instant::now() < deadline, "no log line {any:?} in:\n{log}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn id_is(id: &str) -> String {
    format!("id={id:?}")
}

/// The log lines that say the retention acted on `id`, or kept it for `reason`.
fn acted_or_kept<'a>(id: &'a str, reason: &'a str) -> [Vec<&'a str>; 3] {
    [
        vec!["job output dropped", id],
        vec!["job entry reaped", id],
        vec!["job retained", id, reason],
    ]
}

async fn wait_acted_or_kept(e: &Execd, from: usize, id: &str, reason: &str) -> String {
    let idq = id_is(id);
    let r = format!("reason={reason}");
    let a = acted_or_kept(&idq, &r);
    let refs: Vec<&[&str]> = a.iter().map(|x| &x[..]).collect();
    until_log(e, from, &refs).await
}

/// A shell script that writes `n` bytes of a known pattern, then waits for `flag` to exist, then exits 0.
fn writes_then_waits(n: usize, flag: &Path) -> String {
    format!(
        "head -c {n} /dev/zero | tr '\\0' 'x'; while [ ! -e {f} ]; do sleep 0.02; done",
        f = flag.display()
    )
}

/// A job stream over HTTP/2 whose client reads only what it asks for: the stream's flow-control window is
/// 64 KiB, so a server with more to send than that is held (its `Watching` lives) for as long as the client
/// does not read. Deterministic: no socket buffer size is involved.
struct H2Stream {
    body: hyper::body::Incoming,
    buf: Vec<u8>,
    _send: hyper::client::conn::http2::SendRequest<Full<Bytes>>,
    conn: tokio::task::JoinHandle<()>,
}

impl Drop for H2Stream {
    fn drop(&mut self) {
        self.conn.abort();
    }
}

impl H2Stream {
    async fn open(a: SocketAddr, body: &Value) -> (u16, H2Stream) {
        let sock = tokio::net::TcpStream::connect(a).await.unwrap();
        let (mut send, conn) = hyper::client::conn::http2::Builder::new(TokioExecutor::new())
            .initial_stream_window_size(65_535)
            .initial_connection_window_size(65_535)
            .adaptive_window(false)
            .handshake(TokioIo::new(sock))
            .await
            .unwrap();
        let conn = tokio::spawn(async move {
            let _ = conn.await;
        });
        let req = hyper::Request::builder()
            .method("POST")
            .uri("http://execd/exec")
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap();
        let resp = send.send_request(req).await.unwrap();
        let status = resp.status().as_u16();
        (
            status,
            H2Stream {
                body: resp.into_body(),
                buf: Vec::new(),
                _send: send,
                conn,
            },
        )
    }

    /// The next frame (heartbeats skipped), or `None` at the end.
    async fn next(&mut self) -> Option<Value> {
        loop {
            if let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=nl).collect();
                let v: Value = serde_json::from_slice(&line).unwrap();
                if v.as_object().is_some_and(serde_json::Map::is_empty) {
                    continue;
                }
                return Some(v);
            }
            match tokio::time::timeout(Duration::from_secs(20), self.body.frame()).await {
                Ok(Some(Ok(f))) => {
                    if let Ok(d) = f.into_data() {
                        self.buf.extend_from_slice(&d);
                    }
                }
                Ok(_) => return None,
                Err(_) => panic!("the stream went silent for 20 s"),
            }
        }
    }

    /// Every frame to the end: the output bytes, the gaps, and the last frame.
    async fn drain(&mut self) -> (Vec<u8>, Vec<Value>, Value) {
        let (mut out, mut gaps, mut last) = (Vec::new(), Vec::new(), Value::Null);
        while let Some(f) = self.next().await {
            if let Some(d) = f.get("stdout").or_else(|| f.get("stderr")) {
                out.extend(b64(d));
            } else if f.get("gap").is_some() {
                gaps.push(f);
            } else {
                last = f;
            }
        }
        (out, gaps, last)
    }
}

fn attach_body(id: &str, offset: u64) -> Value {
    json!({ "command": "true", "stream": true,
            "job": { "op": "attach", "scope": "s", "id": id, "offset": offset } })
}

fn fault_env(dir: &Path, lines: &str) -> String {
    let f = dir.join("fault");
    std::fs::write(&f, lines).unwrap();
    f.to_str().unwrap().to_string()
}

fn debug_settings<'a>(extra: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut s = vec![("EXECD_LOG", "debug")];
    s.extend_from_slice(extra);
    s
}

// ---------------------------------------------------------------------------------------------
// The vetoes
// ---------------------------------------------------------------------------------------------

/// **Attached stream.** A job ends while a client holds an `attach` it is not reading (HTTP/2 flow control holds the
/// server at 64 KiB of 4 MiB). Past every deadline and over both ceilings, its output is all still there: `read`
/// answers the bytes, and the stream, once read, carries all 4 MiB with no gap and the exit. After the stream
/// closes the same job *is* reaped (the sweeper is alive in this configuration: the test is not vacuous).
///
/// Red under `mut_retention_ignores_streams`: the `read` during the hold answers no bytes (or `gone`), and the stream
/// carries a gap.
#[tokio::test]
async fn an_attached_stream_holds_its_job_past_every_deadline() {
    let tmp = tempfile::tempdir().unwrap();
    let flag = tmp.path().join("go");
    let e = Execd::start(&debug_settings(&SHARP), &[]);
    let a = addr(&e);
    const N: usize = 4 << 20;
    start(a, "h", &writes_then_waits(N, &flag)).await;
    let (status, mut st) = H2Stream::open(a, &attach_body("h", 0)).await;
    assert_eq!(status, 200);
    let first = st.next().await.unwrap();
    assert!(first.get("attached").is_some(), "{first}");
    // The job ends with the stream open and unread.
    let mark = e.log().len();
    std::fs::write(&flag, "").unwrap();
    until_state(a, "h", "ended").await;
    // The end wakes the sweeper: it saw this job, and either kept it for its stream or acted on it.
    let line = wait_acted_or_kept(&e, mark, "h", "attached").await;
    // Past every deadline (1 ms, second-resolution): a sweep that ignored the stream would have acted by now.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let r = read(a, "h").await;
    assert_eq!(
        r["state"], "ended",
        "while attached: {r} (sweep said: {line})"
    );
    assert_eq!(
        b64(&r["data"]).len(),
        N,
        "while attached, the output went: {r}"
    );
    let (out, gaps, last) = st.drain().await;
    assert!(gaps.is_empty(), "the held stream got a gap: {gaps:?}");
    assert_eq!(out.len(), N, "the held stream lost output");
    assert!(out.iter().all(|&b| b == b'x'));
    assert_eq!(last, json!({ "exit_code": 0 }));
    drop(st);
    // The stream is gone, so is the veto: the job is reaped (entry deadline 1 ms).
    until_state(a, "h", "gone").await;
}

/// **Attached stream, the window.** An attach that has decided to stream (its offset chosen, its stream counted)
/// is parked by the fault seam `attach_after_decide` *before* it answers. The job ends while it is parked, and the
/// sweeper runs over it. When the park is released the stream carries the job's bytes and its exit, never
/// `{"error":"the job is gone"}`.
///
/// Deterministic: the test waits for execd's own log to say the attach is parked, then for the sweep's verdict on the
/// job, and only then releases it. Red under `mut_attach_registers_late` (the old order, counting the stream after
/// the decision): the sweep reaps the job inside the park and the stream ends with the error frame.
#[tokio::test]
async fn an_attach_mid_decision_is_never_reaped() {
    let tmp = tempfile::tempdir().unwrap();
    let flag = tmp.path().join("go");
    let hold = tmp.path().join("hold");
    std::fs::write(&hold, "").unwrap();
    let fault = fault_env(
        tmp.path(),
        &format!("attach_after_decide={}\n", hold.display()),
    );
    let e = Execd::start(&debug_settings(&SHARP), &[("EXECD_TEST_FAULT", &fault)]);
    let a = addr(&e);
    start(
        a,
        "w",
        &format!(
            "printf hi; while [ ! -e {} ]; do sleep 0.02; done",
            flag.display()
        ),
    )
    .await;
    let mark = e.log().len();
    let attach = tokio::spawn(async move {
        let (status, mut st) = H2Stream::open(a, &attach_body("w", 0)).await;
        (status, st.drain().await)
    });
    until_log(&e, mark, &[&["fault parked", "attach_after_decide"]]).await;
    let mark = e.log().len();
    std::fs::write(&flag, "").unwrap();
    until_state(a, "w", "ended").await;
    let line = wait_acted_or_kept(&e, mark, "w", "attached").await;
    tokio::time::sleep(Duration::from_millis(2500)).await;
    std::fs::remove_file(&hold).unwrap();
    let (status, (out, gaps, last)) = attach.await.unwrap();
    assert_eq!(status, 200);
    assert_eq!(
        last,
        json!({ "exit_code": 0 }),
        "the parked attach ended with {last} (sweep said: {line})"
    );
    assert!(gaps.is_empty(), "{gaps:?}");
    assert_eq!(out, b"hi");
    // Released, the veto lifts and the job goes.
    until_state(a, "w", "gone").await;
}

/// **The other half of the window.** An attach that has *looked up* its job and not yet locked it is parked
/// (`attach_before_decide`); the job ends and is reaped while it is parked (no stream is counted yet: the take is
/// legitimate). Released, the attach answers the defined `404 no such job`, never a `200` stream of a job that is gone.
///
/// Red under `mut_attach_ignores_removed`: a `200` whose stream ends `{"error":"the job is gone"}`.
#[tokio::test]
async fn an_attach_that_loses_to_a_take_is_404() {
    let tmp = tempfile::tempdir().unwrap();
    let flag = tmp.path().join("go");
    let hold = tmp.path().join("hold");
    std::fs::write(&hold, "").unwrap();
    let fault = fault_env(
        tmp.path(),
        &format!("attach_before_decide={}\n", hold.display()),
    );
    let e = Execd::start(&debug_settings(&SHARP), &[("EXECD_TEST_FAULT", &fault)]);
    let a = addr(&e);
    start(
        a,
        "r",
        &format!(
            "printf hi; while [ ! -e {} ]; do sleep 0.02; done",
            flag.display()
        ),
    )
    .await;
    let mark = e.log().len();
    let attach = tokio::spawn(async move {
        let (status, mut st) = H2Stream::open(a, &attach_body("r", 0)).await;
        let frames = if status == 200 {
            Some(st.drain().await)
        } else {
            None
        };
        (status, frames)
    });
    until_log(&e, mark, &[&["fault parked", "attach_before_decide"]]).await;
    std::fs::write(&flag, "").unwrap();
    until_state(a, "r", "gone").await;
    std::fs::remove_file(&hold).unwrap();
    let (status, frames) = attach.await.unwrap();
    assert_eq!(
        status, 404,
        "an attach of a reaped job answered {status} {frames:?}"
    );
}

/// A callback receiver on loopback: `None` never answers (the attempt stays in flight), `Some(s)` answers `s`.
async fn receiver(status: Option<u16>) -> (String, Arc<AtomicUsize>) {
    let listener = beyond_test_support::ports::tokio_listener().await;
    let url = format!("http://{}/cb", listener.local_addr().unwrap());
    let seen = Arc::new(AtomicUsize::new(0));
    let count = seen.clone();
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let count = count.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                count.fetch_add(1, Ordering::SeqCst);
                match status {
                    None => {
                        // Hold the connection, answer nothing.
                        tokio::time::sleep(Duration::from_secs(120)).await;
                    }
                    Some(s) => {
                        let _ = sock
                            .write_all(
                                format!("HTTP/1.1 {s} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                                    .as_bytes(),
                            )
                            .await;
                    }
                }
            });
        }
    });
    (url, seen)
}

/// **Owed callback.** Two jobs kept with `notify`, one at a receiver that never answers (an attempt in flight for 30 s)
/// and one at a receiver that answers `503` (a retry waiting on its backoff). Both end; past every deadline and over
/// both ceilings, both are still listed, still `ended`, and still answer their bytes.
///
/// Red under `mut_retention_ignores_owed`: both are reaped (`read` says `gone`), and the fleet never hears their end.
#[tokio::test]
async fn an_owed_callback_holds_its_job() {
    let tmp = tempfile::tempdir().unwrap();
    let flag = tmp.path().join("go");
    let e = Execd::start(&debug_settings(&SHARP), &[]);
    let a = addr(&e);
    let (silent, silent_seen) = receiver(None).await;
    let (busy, busy_seen) = receiver(Some(503)).await;
    let script = format!(
        "printf hi; while [ ! -e {} ]; do sleep 0.02; done",
        flag.display()
    );
    for (id, url) in [("n", &silent), ("f", &busy)] {
        start(a, id, &script).await;
        let keep = op(
            a,
            json!({ "op": "keep", "scope": "s", "id": id, "num": 1, "kind": "background", "seen": 0,
                    "notify": { "url": url, "token": "tok" } }),
        )
        .await;
        assert_eq!(keep.status, 200, "{}", String::from_utf8_lossy(&keep.body));
    }
    let mark = e.log().len();
    std::fs::write(&flag, "").unwrap();
    until_state(a, "n", "ended").await;
    until_state(a, "f", "ended").await;
    let lines = [
        wait_acted_or_kept(&e, mark, "n", "owed_callback").await,
        wait_acted_or_kept(&e, mark, "f", "owed_callback").await,
    ];
    tokio::time::sleep(Duration::from_millis(2500)).await;
    for id in ["n", "f"] {
        let r = read(a, id).await;
        assert_eq!(r["state"], "ended", "{id}: {r} (sweeps said: {lines:?})");
        assert_eq!(b64(&r["data"]), b"hi", "{id}: {r}");
    }
    let ids: Vec<String> = list(a)
        .await
        .iter()
        .map(|j| j["id"].as_str().unwrap().to_string())
        .collect();
    assert!(
        ids.contains(&"n".into()) && ids.contains(&"f".into()),
        "{ids:?}"
    );
    assert!(
        silent_seen.load(Ordering::SeqCst) >= 1,
        "the silent receiver was never called"
    );
    assert!(
        busy_seen.load(Ordering::SeqCst) >= 1,
        "the 503 receiver was never called"
    );
}

/// **Not ended.** A `sleep 60` job lives through every deadline and both ceilings, with its bytes readable and its
/// group signalable; only once a signal ends it does the retention touch it. The sweeper's pass over it is
/// witnessed by a second job that ended after it started and *was* tombstoned in the same configuration.
///
/// Red under `mut_retention_any_phase` (a job aged from its start): its output is dropped while it runs. (The entry
/// itself is saved even then by the kernel's gate, J7: that is why this test runs with the entry clock off, so
/// the one protection it measures is the phase check.)
#[tokio::test]
async fn a_running_job_is_never_touched() {
    let e = Execd::start(
        &debug_settings(&[
            ("EXECD_JOB_OUTPUT_MS", "1"),
            ("EXECD_JOBS_ENDED_BYTES", "1"),
            ("EXECD_JOBS_ENDED_MAX", "1"),
            ("EXECD_JOB_ENTRY_MS", "0"),
        ]),
        &[],
    );
    let a = addr(&e);
    start(a, "run", "printf ready; exec sleep 60").await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while b64(&read(a, "run").await["data"]) != b"ready" {
        assert!(Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let mark = e.log().len();
    start(a, "x", "printf x").await;
    until_state(a, "x", "ended").await;
    until_log(&e, mark, &[&["job output dropped", &id_is("x")]]).await;
    let r = read(a, "run").await;
    assert_eq!(r["state"], "running", "{r}");
    assert_eq!(
        b64(&r["data"]),
        b"ready",
        "a running job's output was dropped: {r}"
    );
    let sig = op(
        a,
        json!({ "op": "signal", "scope": "s", "ids": ["run"], "signal": "SIGTERM" }),
    )
    .await;
    assert_eq!(sig.json()["sent"], json!(["run"]), "{}", sig.json());
    until_state(a, "run", "ended").await;
    // Ended now, it is the retention's like any other.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let r = read(a, "run").await;
        if r["data"] == "" {
            assert_eq!(r["state"], "ended");
            assert_eq!(r["end"], 5);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the ended job was never tombstoned: {r}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// **Keyed.** Two keyed starts, each appending one line to a file, under the count ceiling (1) and the entry
/// clock (1 ms): the ceiling and the clock both say reap, and neither may, because a keyed job's key record leaves
/// with it (K8). Past every deadline the replay of the first start answers the same `mark`, its file holds exactly
/// one line, and both entries are listed. Its output, like anyone's, is dropped.
///
/// Red under `mut_retention_reaps_keyed`: the replay starts the command again (a new mark, two lines).
#[tokio::test]
async fn a_keyed_job_replays_after_every_deadline() {
    let tmp = tempfile::tempdir().unwrap();
    let e = Execd::start(&debug_settings(&SHARP), &[]);
    let a = addr(&e);
    let body = |id: &str| {
        let f = tmp.path().join(id);
        json!({ "command": "sh", "args": ["-c", format!("echo line >> {}; printf out", f.display())],
                "job": { "op": "start", "scope": "s", "id": id, "hold": false, "idempotency_key": format!("key-{id}") } })
    };
    let mut marks = Vec::new();
    let from = e.log().len();
    for id in ["ka", "kb"] {
        let mark = e.log().len();
        let r = post(a, &[], &body(id)).await;
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        marks.push(r.json()["mark"].clone());
        until_state(a, id, "ended").await;
        // The sweeper ran over it after it was due: its output went (or, were the veto gone, its entry).
        let idq = id_is(id);
        until_log(
            &e,
            mark,
            &[&["job output dropped", &idq], &["job entry reaped", &idq]],
        )
        .await;
    }
    // One more end, a second past both: every deadline is past, and the ceiling is over.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    start(a, "tick", "true").await;
    until_state(a, "tick", "gone").await;
    let keyed = wait_acted_or_kept(&e, from, "ka", "keyed").await;
    for (i, id) in ["ka", "kb"].iter().enumerate() {
        let replay = post(a, &[], &body(id)).await;
        assert_eq!(replay.status, 200);
        assert_eq!(
            replay.json()["mark"],
            marks[i],
            "{id} ran again (sweep said: {keyed})"
        );
        let lines = std::fs::read_to_string(tmp.path().join(id)).unwrap();
        assert_eq!(lines, "line\n", "{id} ran {} times", lines.lines().count());
        let r = read(a, id).await;
        assert_eq!(r["state"], "ended");
        assert_eq!(
            r["data"], "",
            "a keyed job's output is the retention's like anyone's"
        );
    }
    let ids: Vec<String> = list(a)
        .await
        .iter()
        .map(|j| j["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids.len(), 2, "{ids:?}");
}

// ---------------------------------------------------------------------------------------------
// What a late client sees, the ceilings, and restart
// ---------------------------------------------------------------------------------------------

/// **Tier 1, the tombstone.** After the output deadline an ended job keeps its entry: `read` answers `ended`, the code,
/// the same `end`, and no bytes from the end; `attach` from 0 answers one `{"gap": n}` and the exit, exactly the gap past the 8
/// MiB floor today; `list` keeps the row with `outputStart == outputEnd`; its spool directory is gone. A second
/// sweep (another job ending) leaves its ledger row byte for byte as it was, and a `SIGHUP` restart answers from the same
/// offset, not from 0.
#[tokio::test]
async fn a_tombstone_keeps_the_status_and_answers_a_gap_across_a_restart() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let settings = debug_settings(&[
        ("EXECD_STATE_DIR", state.to_str().unwrap()),
        ("EXECD_JOBS_DIR", jobs.to_str().unwrap()),
        ("EXECD_JOB_OUTPUT_MS", "1"),
        ("EXECD_JOB_ENTRY_MS", "0"),
        ("EXECD_JOBS_ENDED_BYTES", "0"),
        ("EXECD_JOBS_ENDED_MAX", "0"),
    ]);
    let mut first = Execd::start(&settings, &[]);
    let a = addr(&first);
    const N: u64 = 100_000;
    let m = start(
        a,
        "t",
        &format!("head -c {N} /dev/zero | tr '\\0' 'x'; exit 3"),
    )
    .await["mark"]
        .as_str()
        .unwrap()
        .to_string();
    let r = until_state(a, "t", "ended").await;
    assert_eq!(r["end"], N);
    let spools = |m: &str| -> usize {
        let mut n = 0;
        for d in std::fs::read_dir(&jobs).unwrap().flatten() {
            if d.path().join(m).exists() {
                n += 1;
            }
        }
        n
    };
    assert_eq!(spools(&m), 1, "the job has a spool before the deadline");
    let deadline = Instant::now() + Duration::from_secs(10);
    let r = loop {
        let r = read(a, "t").await;
        if r["data"] == "" {
            break r;
        }
        assert!(Instant::now() < deadline, "never tombstoned: {r}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(r["state"], "ended");
    assert_eq!(r["code"], 3);
    assert_eq!(r["offset"], N);
    assert_eq!(r["end"], N);
    let frames = post(a, &[], &attach_body("t", 0)).await.frames();
    assert_eq!(frames.len(), 3, "{frames:?}");
    assert_eq!(frames[0]["attached"]["from"], 0);
    assert_eq!(frames[1], json!({ "gap": N }));
    assert_eq!(frames[2], json!({ "exit_code": 3 }));
    let row = list(a).await.into_iter().find(|j| j["id"] == "t").unwrap();
    assert_eq!(
        (row["outputStart"].clone(), row["outputEnd"].clone()),
        (json!(N), json!(N))
    );
    // The spool is deleted off the runtime thread after the ledger write: wait for it.
    let deadline = Instant::now() + Duration::from_secs(5);
    while spools(&m) != 0 {
        assert!(
            Instant::now() < deadline,
            "the tombstone's spool is still on disk"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let ledger = || -> Value {
        let text = std::fs::read_to_string(state.join("execd-jobs.json")).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        v["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|j| j["id"] == "t")
            .unwrap()
            .clone()
    };
    let row = ledger();
    assert_eq!(row["out"], N, "{row}");
    assert!(row["spool"].is_null(), "{row}");
    // Idempotent: another sweep (another job's end and tombstone) leaves this row as it was.
    let mark = first.log().len();
    start(a, "u", "printf u").await;
    until_state(a, "u", "ended").await;
    until_log(&first, mark, &[&["job output dropped", &id_is("u")]]).await;
    assert_eq!(ledger(), row, "a second sweep changed the tombstone");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(i32::try_from(first.child.id()).unwrap()),
        nix::sys::signal::Signal::SIGHUP,
    )
    .unwrap();
    first.child.wait().unwrap();
    let second = Execd::start(&settings, &[]);
    let r = read(addr(&second), "t").await;
    assert_eq!(
        (
            r["state"].clone(),
            r["code"].clone(),
            r["offset"].clone(),
            r["end"].clone()
        ),
        (json!("ended"), json!(3), json!(N), json!(N)),
        "after the restart: {r}"
    );
}

/// **Crash between the marker and the unlinks.** The sweep's off-thread pass writes the ledger, then execd dies
/// (`sweep_after_save=exit`) before it deletes the spool. The next execd answers the tombstone from the same offset,
/// and the spool the ledger no longer names is gone (`Spool::retain`): the crash leaves the same state the sweep would
/// have.
#[tokio::test]
async fn a_crash_after_the_ledger_write_leaves_no_spool_behind() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault_env(root.path(), "sweep_after_save=exit\n");
    let settings = [
        ("EXECD_STATE_DIR", state.to_str().unwrap()),
        ("EXECD_JOBS_DIR", jobs.to_str().unwrap()),
        ("EXECD_JOB_OUTPUT_MS", "1"),
        ("EXECD_JOB_ENTRY_MS", "0"),
        ("EXECD_JOBS_ENDED_BYTES", "0"),
        ("EXECD_JOBS_ENDED_MAX", "0"),
    ];
    let mut first = Execd::start(&settings, &[("EXECD_TEST_FAULT", &fault)]);
    let a = addr(&first);
    let m = start(a, "c", "head -c 100000 /dev/zero | tr '\\0' 'x'").await["mark"]
        .as_str()
        .unwrap()
        .to_string();
    until_state(a, "c", "ended").await;
    let status = first.child.wait().unwrap();
    assert_eq!(status.code(), Some(86), "the sweep never reached its fault");
    let has_spool = || {
        std::fs::read_dir(&jobs)
            .unwrap()
            .flatten()
            .any(|d| d.path().join(&m).exists())
    };
    assert!(
        has_spool(),
        "the crash came after the unlinks: the test proves nothing"
    );
    let second = Execd::start(&settings, &[]);
    let r = read(addr(&second), "c").await;
    assert_eq!(
        (
            r["state"].clone(),
            r["offset"].clone(),
            r["end"].clone(),
            r["data"].clone()
        ),
        (json!("ended"), json!(100_000), json!(100_000), json!("")),
        "{r}"
    );
    assert!(
        !has_spool(),
        "the spool the ledger no longer names survived the restart"
    );
}

/// **Tier 2a, the count ceiling.** `EXECD_JOBS_ENDED_MAX=2` and nothing else: five jobs end a second apart; the three
/// oldest are reaped (`read` says `gone`, `attach` is `404`, a second `remove` names nothing), the two newest keep
/// their entries *and* their output. No clock is involved.
#[tokio::test]
async fn the_count_ceiling_reaps_the_oldest_ended_entries() {
    let e = Execd::start(
        &[
            ("EXECD_JOB_OUTPUT_MS", "0"),
            ("EXECD_JOB_ENTRY_MS", "0"),
            ("EXECD_JOBS_ENDED_BYTES", "0"),
            ("EXECD_JOBS_ENDED_MAX", "2"),
        ],
        &[],
    );
    let a = addr(&e);
    let ids = ["c0", "c1", "c2", "c3", "c4"];
    for id in ids {
        start(a, id, &format!("printf {id}")).await;
        until_state(a, id, "ended").await;
        tokio::time::sleep(Duration::from_millis(1100)).await;
    }
    for id in &ids[..3] {
        until_state(a, id, "gone").await;
        assert_eq!(post(a, &[], &attach_body(id, 0)).await.status, 404);
    }
    for id in &ids[3..] {
        let r = read(a, id).await;
        assert_eq!(r["state"], "ended");
        assert_eq!(b64(&r["data"]), id.as_bytes());
    }
    let rm = op(a, json!({ "op": "remove", "scope": "s", "ids": ["c0"] })).await;
    assert_eq!(rm.json(), json!({ "running": [], "mismatch": [] }));
    assert_eq!(list(a).await.len(), 2);
}

/// **Tier 1, the byte budget.** `EXECD_JOBS_ENDED_BYTES=150000` and nothing else: three jobs of 100,000 bytes end a second
/// apart; the two oldest are tombstoned (in that order: the newest keeps its output), and all three entries stay —
/// byte pressure alone never reaps.
#[tokio::test]
async fn the_byte_budget_tombstones_the_oldest_and_reaps_nothing() {
    let e = Execd::start(
        &[
            ("EXECD_JOB_OUTPUT_MS", "0"),
            ("EXECD_JOB_ENTRY_MS", "0"),
            ("EXECD_JOBS_ENDED_BYTES", "150000"),
            ("EXECD_JOBS_ENDED_MAX", "0"),
        ],
        &[],
    );
    let a = addr(&e);
    for id in ["b0", "b1", "b2"] {
        start(a, id, "head -c 100000 /dev/zero | tr '\\0' 'x'").await;
        until_state(a, id, "ended").await;
        tokio::time::sleep(Duration::from_millis(1100)).await;
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let held: Vec<usize> = futures_len(a, &["b0", "b1", "b2"]).await;
        if held == [0, 0, 100_000] {
            break;
        }
        assert!(Instant::now() < deadline, "held {held:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    for id in ["b0", "b1", "b2"] {
        let r = read(a, id).await;
        assert_eq!(
            (r["state"].clone(), r["end"].clone()),
            (json!("ended"), json!(100_000))
        );
    }
}

async fn futures_len(a: SocketAddr, ids: &[&str]) -> Vec<usize> {
    let mut out = Vec::new();
    for id in ids {
        out.push(b64(&read(a, id).await["data"]).len());
    }
    out
}

/// **Every knob off is today.** All four at `0`: past any deadline the retention never started (no line of its in the
/// log), every job keeps its entry and every byte, and the spool is still on disk.
#[tokio::test]
async fn every_knob_off_is_todays_behaviour() {
    let root = tempfile::tempdir().unwrap();
    let e = Execd::start(
        &debug_settings(&[
            ("EXECD_JOBS_DIR", root.path().to_str().unwrap()),
            ("EXECD_JOB_OUTPUT_MS", "0"),
            ("EXECD_JOB_ENTRY_MS", "0"),
            ("EXECD_JOBS_ENDED_BYTES", "0"),
            ("EXECD_JOBS_ENDED_MAX", "0"),
        ]),
        &[],
    );
    let a = addr(&e);
    // In RAM only, one chunk on disk, many chunks on disk.
    const SIZES: [usize; 3] = [1000, 100_000, 1_000_000];
    let mut marks = Vec::new();
    for (i, id) in ["o0", "o1", "o2"].iter().enumerate() {
        let n = SIZES[i];
        marks.push(
            start(a, id, &format!("head -c {n} /dev/zero | tr '\\0' 'x'")).await["mark"].clone(),
        );
        until_state(a, id, "ended").await;
    }
    tokio::time::sleep(Duration::from_millis(2500)).await;
    for (i, id) in ["o0", "o1", "o2"].iter().enumerate() {
        let r = read(a, id).await;
        assert_eq!(r["state"], "ended");
        assert_eq!(b64(&r["data"]).len(), SIZES[i]);
    }
    assert_eq!(list(a).await.len(), 3);
    let log = e.log();
    assert!(
        !log.contains("retention")
            && !log.contains("job output dropped")
            && !log.contains("job entry reaped"),
        "{log}"
    );
    let m = marks[2].as_str().unwrap();
    assert!(
        std::fs::read_dir(root.path())
            .unwrap()
            .flatten()
            .any(|d| d.path().join(m).exists())
    );
}

/// **Concurrent with clients.** Rounds of 24 jobs ending while a client removes them and starts the same ids again,
/// with every knob at its sharpest: every answer is a defined one (no `5xx`, no panic), a `remove` never reports a
/// mismatch, and once everything is removed the table is empty and nothing is left in the jobs directory.
#[tokio::test]
async fn a_sweep_concurrent_with_remove_and_start_leaves_one_state() {
    let root = tempfile::tempdir().unwrap();
    let e = Execd::start(
        &[
            ("EXECD_JOBS_DIR", root.path().to_str().unwrap()),
            ("EXECD_JOB_OUTPUT_MS", "1"),
            ("EXECD_JOB_ENTRY_MS", "1"),
            ("EXECD_JOBS_ENDED_BYTES", "1"),
            ("EXECD_JOBS_ENDED_MAX", "4"),
        ],
        &[],
    );
    let a = addr(&e);
    let ids: Vec<String> = (0..24).map(|i| format!("j{i}")).collect();
    for round in 0u64..4 {
        let mut tasks = Vec::new();
        for id in ids.clone() {
            tasks.push(tokio::spawn(async move {
                let s = post(
                    a,
                    &[],
                    &json!({ "command": "sh", "args": ["-c", "head -c 70000 /dev/zero"],
                    "job": { "op": "start", "scope": "s", "id": id, "hold": false } }),
                )
                .await
                .status;
                assert!(s == 200 || s == 409, "start answered {s}");
                tokio::time::sleep(Duration::from_millis(round * 300)).await;
                let rm = op(a, json!({ "op": "remove", "scope": "s", "ids": [id] })).await;
                assert_eq!(rm.status, 200);
                assert_eq!(rm.json()["mismatch"], json!([]));
                let r = read(a, &id).await;
                assert!(
                    ["running", "ended", "gone"].contains(&r["state"].as_str().unwrap()),
                    "{r}"
                );
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let left = list(a).await;
        if left.is_empty() {
            break;
        }
        let names: Vec<Value> = left.iter().map(|j| j["id"].clone()).collect();
        op(a, json!({ "op": "remove", "scope": "s", "ids": names })).await;
        assert!(Instant::now() < deadline, "{left:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(!e.log().contains("panicked"), "{}", e.log());
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let mut files = Vec::new();
        let mut stack = vec![root.path().to_path_buf()];
        while let Some(d) = stack.pop() {
            for f in std::fs::read_dir(&d).unwrap().flatten() {
                if f.file_type().unwrap().is_dir() {
                    stack.push(f.path());
                } else {
                    files.push(f.path());
                }
            }
        }
        if files.is_empty() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "left in the jobs directory: {files:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
