//! A job's chunks in one file (the pack), against the real binary: one inode for a job's whole
//! output instead of one a chunk, and the invariants that do not move with them — the offsets, the
//! retention floor, the stderr labels, the gap a refused or torn chunk costs, and a restart.
//!
//! Each protection has a `mut_*` feature that deletes it, and each test names the feature that
//! turns it red:
//!
//! ```sh
//! cargo test -p beyond-execd --test pack                                          # green
//! cargo test -p beyond-execd --test pack --features mut_pack_keeps_short_tail     # red
//! cargo test -p beyond-execd --test pack --features mut_pack_ignores_punch_probe  # red
//! cargo test -p beyond-execd --test pack --features mut_pack_stride_overlap       # red
//! ```
//!
//! The two crash tests force their crash through `EXECD_TEST_FAULT` (`pack_torn`,
//! `pack_torn_after_journal`): a pack whose last stride the disk cut short, and one whose chunk the
//! journal had advanced past. Neither waits on an interleaving.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use base64::Engine as _;
use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::{Value, json};
use support::Execd;
use support::http::post;

const FRAME: u64 = 64 * 1024;

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

/// Start an unheld job running `sh -c script` as `id`; its mark.
async fn start(a: SocketAddr, id: &str, script: &str) -> String {
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
    r.json()["mark"].as_str().unwrap().to_string()
}

async fn read(a: SocketAddr, id: &str, offset: u64, max: u64) -> Value {
    op(
        a,
        json!({ "op": "read", "scope": "s", "id": id, "offset": offset, "max": max }),
    )
    .await
    .json()
}

/// Every byte of `id`'s output from `offset`, read a frame at a time, with its stderr ranges. A
/// declared gap moves the next ask past it: the answer's own `offset` is where its bytes start.
async fn read_all(a: SocketAddr, id: &str, offset: u64) -> (Vec<u8>, Vec<(u64, u64)>) {
    let mut out = Vec::new();
    let mut ranges = Vec::new();
    let mut at = offset;
    loop {
        let r = read(a, id, at, FRAME).await;
        let data = b64(&r["data"]);
        if data.is_empty() {
            return (out, ranges);
        }
        for (a, n) in r["stderr_ranges"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap()))
        {
            ranges.push((a, n));
        }
        at = r["offset"].as_u64().unwrap_or(at) + data.len() as u64;
        out.extend(data);
    }
}

async fn until_state(a: SocketAddr, id: &str, state: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let r = read(a, id, 0, 1 << 24).await;
        if r["state"] == state {
            return r;
        }
        assert!(Instant::now() < deadline, "{id} never became {state}: {r}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn fault_env(dir: &Path, lines: &str) -> String {
    let f = dir.join("fault");
    std::fs::write(&f, lines).unwrap();
    f.to_str().unwrap().to_string()
}

/// The job's spool under `jobs`: the stable directory's, or this server's own.
fn spool_of(jobs: &Path, mark: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(jobs).ok()?.flatten() {
        let p = e.path().join(mark);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Every byte in a stderr range is `marker`, and no other byte is: the labels name the bytes they
/// came with, whatever order the two files' bytes arrived in.
fn labelled(data: &[u8], ranges: &[(u64, u64)], marker: u8) {
    let mut want = vec![false; data.len()];
    for &(a, n) in ranges {
        for off in a..a + n {
            want[off as usize] = true;
        }
    }
    for (i, &b) in data.iter().enumerate() {
        assert_eq!(
            b == marker,
            want[i],
            "byte {i} is {b:?}, its label says stderr={}",
            want[i]
        );
    }
    assert!(!ranges.is_empty(), "no labels at all");
}

/// Wait for execd to exit on its own (a fault kill), with a deadline: a fault that never fires is a
/// failed test, not a hung one.
fn exited(e: &mut Execd, want: i32) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = e.child.try_wait().unwrap() {
            assert_eq!(status.code(), Some(want), "execd exited on its own");
            return;
        }
        assert!(
            Instant::now() < deadline,
            "execd never reached the fault (expected exit {want})"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn names(d: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(d)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// **One file a job.** 8 MiB of output, 128 chunks: the spool is `out`, `err`, `journal` and
/// `ring/chunks` — four names, not 128 files and 128 sidecars — and the bytes read back are the
/// bytes written, with their labels.
///
/// Red under `mut_pack_ignores_punch_probe`? No: this is the pack's own shape. It goes red if a
/// per-chunk file comes back where the filesystem can punch (`a_punchless_filesystem_keeps_one_file_a_chunk`
/// is the other side of the same probe).
#[tokio::test]
async fn a_jobs_chunks_are_one_file() {
    const N: usize = 8 << 20;
    let jobs = tempfile::tempdir().unwrap();
    let e = Execd::start(&[("EXECD_JOBS_DIR", jobs.path().to_str().unwrap())], &[]);
    let a = addr(&e);
    // A pattern with no zero byte in it, so a byte the pack does not hold is unmistakable.
    let mark = start(a, "p", &format!("head -c {N} /dev/zero | tr '\\0' 'x'")).await;
    let r = until_state(a, "p", "ended").await;
    assert_eq!(r["end"], N as u64, "{r}");
    let (data, _) = read_all(a, "p", 0).await;
    assert_eq!(data.len(), N);
    assert!(data.iter().all(|&b| b == b'x'), "the bytes came back");
    let spool = spool_of(jobs.path(), &mark).expect("the job's spool");
    let mut entries = names(&spool);
    entries.sort();
    assert_eq!(
        entries,
        vec!["err", "journal", "out", "ring"],
        "a job is four files, whatever it wrote"
    );
    assert_eq!(
        names(&spool.join("ring")),
        vec!["chunks"],
        "one file, 128 chunks"
    );
}

/// **A filesystem that cannot punch keeps one file a chunk.** With the capability probe answered
/// "no" (the fault seam stands in for NFS/SMB, which cannot be mounted here), the ring writes the
/// per-chunk layout: the same bytes, the same labels, and a restart of execd answers at the same
/// offset from the same files.
///
/// Red under `mut_pack_ignores_punch_probe`: the answer to the probe is thrown away, the pack is
/// used on a filesystem whose eviction frees nothing, and no chunk file is there to find.
#[tokio::test]
async fn a_punchless_filesystem_keeps_one_file_a_chunk() {
    // Four chunks, each one stderr byte then 64 KiB - 1 stdout bytes, in two halves the restart
    // falls between.
    const CHUNKS: u64 = 4;
    const HALF: u64 = 2 * 64 * 1024;
    const N: u64 = CHUNKS * 64 * 1024;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    let flag = root.path().join("go");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault_env(root.path(), "punch_probe=unsupported\n");
    let settings = [
        ("EXECD_STATE_DIR", state.to_str().unwrap()),
        ("EXECD_JOBS_DIR", jobs.to_str().unwrap()),
        ("EXECD_LOG", "debug"),
    ];
    let mut first = Execd::start(&settings, &[("EXECD_TEST_FAULT", &fault)]);
    let a = addr(&first);
    let chunk = 64 * 1024 - 1;
    let script = format!(
        "i=0; while [ $i -lt 2 ]; do printf e >&2; head -c {chunk} /dev/zero | tr '\\0' 'x'; \
         i=$((i+1)); done; \
         while [ ! -e {f} ]; do sleep 0.02; done; \
         i=0; while [ $i -lt 2 ]; do printf e >&2; head -c {chunk} /dev/zero | tr '\\0' 'x'; \
         i=$((i+1)); done",
        f = flag.display()
    );
    let mark = start(a, "f", &script).await;
    // Past the first half, still running.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let r = read(a, "f", 0, 1 << 24).await;
        if r["end"].as_u64().unwrap_or(0) >= HALF {
            break;
        }
        assert!(Instant::now() < deadline, "never wrote the first half: {r}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let ring = spool_of(&jobs, &mark)
        .expect("the job's spool")
        .join("ring");
    let wrote = names(&ring);
    assert!(
        !wrote.iter().any(|n| n == "chunks"),
        "the pack was used where the filesystem cannot punch: {wrote:?}"
    );
    assert_eq!(
        wrote.iter().filter(|n| !n.ends_with(".e")).count(),
        2,
        "{wrote:?}"
    );
    // A restart reads the same files, and the job writes on through it.
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(i32::try_from(first.child.id()).unwrap()),
        nix::sys::signal::Signal::SIGHUP,
    )
    .unwrap();
    first.child.wait().unwrap();
    let second = Execd::start(&settings, &[("EXECD_TEST_FAULT", &fault)]);
    let a2 = addr(&second);
    std::fs::write(&flag, b"go").unwrap();
    until_state(a2, "f", "ended").await;
    let (data, ranges) = read_all(a2, "f", 0).await;
    assert_eq!(data.len() as u64, N, "every byte, across the restart");
    assert_eq!(ranges.iter().map(|r| r.1).sum::<u64>(), CHUNKS);
    labelled(&data, &ranges, b'e');
    let after = names(&spool_of(&jobs, &mark).unwrap().join("ring"));
    assert!(
        after.iter().filter(|n| !n.ends_with(".e")).count() == CHUNKS as usize,
        "the restart changed the layout: {after:?}"
    );
}

/// **A torn last chunk is a declared gap of at most one chunk, and never fabricated bytes.** The
/// fault cuts the pack short under the chunk a push wrote (`pack_torn`), exactly what a write whose
/// tail never reached the disk leaves, and execd dies at once. The journal has not advanced past
/// that chunk, so what it holds rebuilds it: after the restart the answer is every byte the job
/// wrote, in order, and no zero from the stride that was cut short.
///
/// Red under `mut_pack_keeps_short_tail`: recovery counts the torn stride as a whole chunk, its
/// length reaches past the bytes that are there, and the ring no longer agrees with the journal.
#[tokio::test]
async fn a_torn_pack_tail_is_recovered_from_the_journal() {
    const N: usize = 300 * 1024;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    // Half a chunk of the second stride never landed.
    let fault = fault_env(root.path(), &format!("pack_torn={}\n", FRAME / 2));
    let settings = [
        ("EXECD_STATE_DIR", state.to_str().unwrap()),
        ("EXECD_JOBS_DIR", jobs.to_str().unwrap()),
        ("EXECD_LOG", "debug"),
    ];
    let mut first = Execd::start(&settings, &[("EXECD_TEST_FAULT", &fault)]);
    let a = addr(&first);
    let mark = start(
        a,
        "t",
        &format!(
            "head -c {N} /dev/zero | tr '\\0' 'x'; sleep 1; head -c {N} /dev/zero | tr '\\0' 'y'"
        ),
    )
    .await;
    exited(&mut first, 86);
    let second = Execd::start(&settings, &[]);
    let a2 = addr(&second);
    // The job kept writing while no execd ran; the new one reads the files and the ring — and it
    // rebuilt them, rather than starting the output over, which is what a gap it cannot account
    // for looks like (logged, and a moved offset).
    let r = until_state(a2, "t", "ended").await;
    assert_eq!(r["offset"], 0, "the answer starts over: {r}");
    assert!(
        !second.log().contains("restarts with a gap"),
        "the pack did not agree with the journal:\n{}",
        second.log()
    );
    let (data, _) = read_all(a2, "t", 0).await;
    assert_eq!(r["end"], 2 * N as u64, "{r}");
    assert_eq!(data.len(), 2 * N, "every byte, {mark}");
    let (first_half, second_half) = data.split_at(N);
    assert!(
        first_half.iter().all(|&b| b == b'x'),
        "the bytes before the tear"
    );
    assert!(second_half.iter().all(|&b| b == b'y'), "the bytes after it");
}

/// **A pack the journal had outrun is a declared gap, and no byte of it is served as zeros.** The
/// fault cuts the pack short after the journal records the push (`pack_torn_after_journal`): the
/// journal's advance is on disk and the chunk it names is not, which a machine crash can produce and
/// a process crash cannot. The chunk is gone from the pack and from the journal both, so the answer
/// starts after it — a gap of exactly the bytes that window held, one chunk here — and every byte it
/// does serve is the byte the job wrote.
///
/// Red under `mut_pack_ignores_journal_floor`: recovery keeps the pack as it is, so the chunk the
/// journal had advanced past reads back as the hole it is — zeros at an offset the answer still
/// names — and every byte after it moves. (`mut_pack_keeps_short_tail` does not reach this case:
/// the torn pack ends below the journal's first byte, so the floor trip, not the whole-chunk count,
/// is what declares this gap.)
#[tokio::test]
async fn a_pack_the_journal_outran_is_a_declared_gap() {
    const N: usize = 300 * 1024;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    // Everything the journal's first advance covers, and more than one stride's worth.
    let fault = fault_env(
        root.path(),
        &format!("pack_torn_after_journal={}\n", 200_000),
    );
    let settings = [
        ("EXECD_STATE_DIR", state.to_str().unwrap()),
        ("EXECD_JOBS_DIR", jobs.to_str().unwrap()),
        ("EXECD_LOG", "debug"),
    ];
    let mut first = Execd::start(&settings, &[("EXECD_TEST_FAULT", &fault)]);
    let a = addr(&first);
    start(
        a,
        "g",
        &format!(
            "head -c {N} /dev/zero | tr '\\0' 'x'; sleep 1; head -c {N} /dev/zero | tr '\\0' 'y'"
        ),
    )
    .await;
    let status = first.child.wait().unwrap();
    assert_eq!(
        status.code(),
        Some(86),
        "the fault never reached the journal"
    );
    let second = Execd::start(&settings, &[]);
    let a2 = addr(&second);
    let r = until_state(a2, "g", "ended").await;

    let (data, _) = read_all(a2, "g", 0).await;
    // The answer starts after the gap, and says so: a read from 0 that answers from a later offset
    // is the gap, on the wire.
    let from = r["offset"].as_u64().unwrap();
    assert!(from > 0, "the gap was not declared: {r}");
    assert!(from <= FRAME, "the gap is more than one chunk: {from}");
    assert_eq!(data.len() as u64, 2 * N as u64 - from);
    assert!(
        data.iter().all(|&b| b == b'x' || b == b'y'),
        "a zero was served as data"
    );
}

/// **The stride is the format.** A chunk with stderr in every chunk reads back with exactly its
/// labels, and the bytes of every chunk are its own: the bits of one chunk never land on another's
/// bytes.
///
/// Red under `mut_pack_stride_overlap`: the bits start where the next chunk's bytes do, so a
/// chunk's labels overwrite its successor's payload.
#[tokio::test]
async fn the_stride_gives_every_chunk_its_own_bytes_and_labels() {
    const N: usize = 4 * 64 * 1024;
    let jobs = tempfile::tempdir().unwrap();
    let e = Execd::start(&[("EXECD_JOBS_DIR", jobs.path().to_str().unwrap())], &[]);
    let a = addr(&e);
    // One stderr byte at the start of each chunk, and stdout filling the rest of it.
    let script = format!(
        "i=0; while [ $i -lt 4 ]; do printf E >&2; head -c {} /dev/zero | tr '\\0' 'x'; i=$((i+1)); done",
        64 * 1024 - 1
    );
    start(a, "s", &script).await;
    until_state(a, "s", "ended").await;
    let (data, ranges) = read_all(a, "s", 0).await;
    assert_eq!(data.len(), N, "every byte of the four chunks");
    assert_eq!(ranges.iter().map(|r| r.1).sum::<u64>(), 4, "{ranges:?}");
    labelled(&data, &ranges, b'E');
}

// -- read and stream, the same job ---------------------------------------------------------------

/// A stream client over HTTP/2 that reads the frames as they come.
struct Stream {
    body: hyper::body::Incoming,
    buf: Vec<u8>,
    conn: tokio::task::JoinHandle<()>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.conn.abort();
    }
}

impl Stream {
    async fn open(a: SocketAddr, body: &Value) -> Self {
        let sock = tokio::net::TcpStream::connect(a).await.unwrap();
        let (mut send, conn) = hyper::client::conn::http2::Builder::new(TokioExecutor::new())
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
        assert_eq!(resp.status().as_u16(), 200);
        drop(send);
        Self {
            body: resp.into_body(),
            buf: Vec::new(),
            conn,
        }
    }

    /// Every stdout and stderr byte, the gaps, and the last frame.
    async fn drain(&mut self) -> (Vec<u8>, Vec<u64>, Value) {
        let (mut out, mut gaps, mut last) = (Vec::new(), Vec::new(), Value::Null);
        loop {
            while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=nl).collect();
                let v: Value = serde_json::from_slice(&line).unwrap();
                if v.as_object().is_some_and(serde_json::Map::is_empty) {
                    continue;
                }
                if let Some(d) = v.get("stdout").or_else(|| v.get("stderr")) {
                    out.extend(b64(d));
                } else if let Some(n) = v.get("gap").and_then(Value::as_u64) {
                    gaps.push(n);
                } else {
                    last = v;
                }
            }
            match tokio::time::timeout(Duration::from_secs(30), self.body.frame()).await {
                Ok(Some(Ok(f))) => {
                    if let Ok(d) = f.into_data() {
                        self.buf.extend_from_slice(&d);
                    }
                }
                Ok(_) => return (out, gaps, last),
                Err(_) => panic!("the stream went silent"),
            }
        }
    }
}

/// **`read` and `attach` answer the same bytes.** For one ended job's output — 300 KiB with stderr
/// at both ends of every chunk — the stream from offset 0 carries exactly what `read` gives, in the
/// same order, with the same labels. They share `Ring::read`, and this is what says so end to end.
#[tokio::test]
async fn read_and_stream_answer_the_same_bytes() {
    const N: usize = 300 * 1024;
    let jobs = tempfile::tempdir().unwrap();
    let e = Execd::start(&[("EXECD_JOBS_DIR", jobs.path().to_str().unwrap())], &[]);
    let a = addr(&e);
    start(
        a,
        "r",
        &format!(
            "printf 'E' >&2; head -c {} /dev/zero | tr '\\0' 'x'; printf 'E' >&2; head -c {N} /dev/zero | tr '\\0' 'z'",
            N - 2
        ),
    )
    .await;
    until_state(a, "r", "ended").await;
    let (read_bytes, read_ranges) = read_all(a, "r", 0).await;
    let mut s = Stream::open(
        a,
        &json!({ "command": "true", "stream": true,
                 "job": { "op": "attach", "scope": "s", "id": "r", "offset": 0 } }),
    )
    .await;
    let (stream_bytes, gaps, last) = s.drain().await;
    assert!(gaps.is_empty(), "{gaps:?}");
    assert_eq!(last["exit_code"], 0, "{last}");
    assert_eq!(stream_bytes.len(), read_bytes.len(), "the same length");
    assert!(stream_bytes == read_bytes, "the same bytes");
    // The labels: `read` answers ranges, the stream tags each frame; both cover the same bytes.
    let labelled: u64 = read_ranges.iter().map(|r| r.1).sum();
    assert_eq!(labelled, 2, "{read_ranges:?}");
    assert_eq!(
        read_bytes.iter().filter(|&&b| b == b'E').count(),
        2,
        "the stderr bytes are where the ranges say"
    );
}
