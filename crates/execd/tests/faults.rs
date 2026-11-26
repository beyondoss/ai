//! What execd does when the machine under it fails: a full disk under the jobs directory, no free
//! file descriptors, no room for another process. Each fault must give an honest error (never a
//! panic, never a hang), leave what already runs alone, and be forgotten when it clears. The
//! limits are applied to the running binary with `prlimit`, and the disk is a small tmpfs mounted
//! through `sudo -n` (a test that cannot mount says `SKIPPED`).
//!
//! The spec's rules these encode (`crates/sandbox/src/exec_endpoint.rs`): a command that cannot be
//! spawned is an ordinary result (one-shot `exit_code` 126, streaming a `stderr` frame then 126), a job that
//! cannot be started is a non-2xx answer, offsets never move (dropped output is a gap, reported in
//! `offset`), and a failure never ends a job that is running.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};
use support::Execd;
use support::http::{Answer, post, send};

const SCOPE: &str = "faults";

fn addr(e: &Execd) -> SocketAddr {
    SocketAddr::V4(e.addr)
}

/// One request, which must be answered in good time: a hang is a failure here.
async fn call(e: &Execd, body: Value) -> Answer {
    try_call(e, body, Duration::from_secs(15))
        .await
        .expect("execd did not answer (a hang)")
}

async fn try_call(e: &Execd, body: Value, within: Duration) -> Option<Answer> {
    tokio::time::timeout(within, post(addr(e), &[], &body))
        .await
        .ok()
}

fn sh(script: &str) -> Value {
    json!({ "command": "sh", "args": ["-c", script], "timeout_ms": 20_000 })
}

fn stream(script: &str) -> Value {
    let mut r = sh(script);
    r["stream"] = json!(true);
    r
}

fn job_op(job: Value) -> Value {
    let mut job = job;
    job["scope"] = json!(SCOPE);
    json!({ "command": "true", "job": job })
}

async fn start_job(e: &Execd, id: &str, script: &str) -> Answer {
    let mut r = sh(script);
    r["job"] = json!({ "op": "start", "scope": SCOPE, "id": id, "hold": false, "label": "f" });
    call(e, r).await
}

struct Read {
    state: String,
    from: u64,
    end: u64,
    data: Vec<u8>,
}

async fn read(e: &Execd, id: &str, offset: u64) -> Read {
    let a = call(
        e,
        job_op(json!({ "op": "read", "id": id, "offset": offset, "max": 4 << 20 })),
    )
    .await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    let v = a.json();
    Read {
        state: v["state"].as_str().unwrap().to_string(),
        from: v["offset"].as_u64().unwrap(),
        end: v["end"].as_u64().unwrap(),
        data: base64::engine::general_purpose::STANDARD
            .decode(v["data"].as_str().unwrap())
            .unwrap(),
    }
}

async fn wipe(e: &Execd) {
    call(e, job_op(json!({ "op": "wipe", "grace_ms": 100 }))).await;
}

/// The final frame of a stream.
fn final_frame(frames: &[Value]) -> &Value {
    frames.last().expect("a stream with no frames")
}

fn assert_no_panic(e: &Execd) {
    let log = e.log();
    assert!(!log.contains("panicked"), "execd panicked:\n{log}");
}

/// A command that cannot be spawned is an ordinary result, on every surface that runs one.
async fn assert_spawn_fails_cleanly(e: &Execd, why: &str) {
    let a = call(e, sh("echo never")).await;
    assert_eq!(
        a.status,
        200,
        "one-shot ({why}): {}",
        String::from_utf8_lossy(&a.body)
    );
    let v = a.json();
    assert_eq!(v["exit_code"], 126, "one-shot ({why}): {v}");
    assert!(
        !v["stderr"].as_str().unwrap_or("").is_empty(),
        "one-shot ({why}): 126 with no reason: {v}"
    );

    let a = call(e, stream("echo never")).await;
    assert_eq!(a.status, 200, "streaming ({why})");
    let frames = a.frames();
    assert_eq!(
        final_frame(&frames)["exit_code"],
        126,
        "streaming ({why}): {frames:?}"
    );
    assert!(
        frames.iter().any(|f| f.get("stderr").is_some()),
        "streaming ({why}): no reason given: {frames:?}"
    );
    assert!(
        !frames.iter().any(|f| f.get("stdout").is_some()),
        "streaming ({why}): the command ran: {frames:?}"
    );

    let a = start_job(e, "refused", "echo never").await;
    assert_eq!(
        a.status,
        500,
        "job start ({why}): {}",
        String::from_utf8_lossy(&a.body)
    );
    assert!(!a.body.is_empty());
    // The refused start left no job behind to hold its id.
    assert_eq!(
        read(e, "refused", 0).await.state,
        "gone",
        "job start ({why})"
    );
}

async fn assert_spawns_work(e: &Execd, why: &str) {
    assert_spawns_work_with(e, why, true).await;
}

/// `jobs_start`: a job can be started too. A job's output is a file in the jobs directory, so on a
/// full disk it cannot be: the start is refused (a non-2xx answer, as the spec has it), not hung,
/// and leaves nothing in the table.
async fn assert_spawns_work_with(e: &Execd, why: &str, jobs_start: bool) {
    let a = call(e, sh("printf ok")).await;
    assert_eq!(
        a.json()["exit_code"],
        0,
        "one-shot ({why}): {}",
        String::from_utf8_lossy(&a.body)
    );
    assert_eq!(a.json()["stdout"], "b2s=", "one-shot ({why})");
    let a = call(e, stream("printf ok")).await;
    let frames = a.frames();
    assert_eq!(
        final_frame(&frames)["exit_code"],
        0,
        "streaming ({why}): {frames:?}"
    );
    let id = format!("after-{}", why.replace(' ', "-"));
    let a = start_job(e, &id, "printf ok").await;
    if jobs_start {
        assert_eq!(
            a.status,
            200,
            "job start ({why}): {}",
            String::from_utf8_lossy(&a.body)
        );
    } else {
        // A full disk is `507` with a reason, as E2B's envd answers one; the start is not a bare `500`.
        assert_eq!(a.status, 507, "job start ({why})");
        assert!(
            a.json()["error"]
                .as_str()
                .is_some_and(|e| e.contains("disk is full")),
            "job start ({why}): {}",
            String::from_utf8_lossy(&a.body)
        );
        assert_eq!(read(e, &id, 0).await.state, "gone", "job start ({why})");
    }
}

async fn livez(e: &Execd) -> u16 {
    tokio::time::timeout(
        Duration::from_secs(5),
        send(addr(e), "GET", "/livez", &[], b""),
    )
    .await
    .expect("/livez did not answer")
    .status
}

fn prlimit(pid: u32, resource: &str) -> String {
    let out = Command::new("prlimit")
        .args([
            "--pid",
            &pid.to_string(),
            resource,
            "--noheadings",
            "--raw",
            "-o",
            "SOFT",
        ])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn set_prlimit(pid: u32, resource: &str, soft: &str) {
    let ok = Command::new("prlimit")
        .args(["--pid", &pid.to_string(), &format!("{resource}={soft}:")])
        .status()
        .unwrap()
        .success();
    assert!(ok, "prlimit {resource}={soft}: failed");
}

fn open_fds(pid: u32) -> usize {
    std::fs::read_dir(format!("/proc/{pid}/fd"))
        .unwrap()
        .count()
}

async fn until<F: std::future::Future<Output = bool>>(what: &str, mut f: impl FnMut() -> F) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !f().await {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ---- the disk under the jobs directory is full ----

/// A small tmpfs, and a thread that keeps it full on demand.
struct SmallDisk {
    dir: tempfile::TempDir,
    stop: Arc<AtomicBool>,
    filler: Option<std::thread::JoinHandle<()>>,
}

impl SmallDisk {
    fn mount(kib: u32) -> Option<Self> {
        if !Execd::can_be_root() {
            eprintln!("SKIPPED: this test needs a tmpfs mount, and the host gives no root");
            return None;
        }
        let dir = tempfile::tempdir().unwrap();
        let ok = Command::new("sudo")
            .args(["-n", "mount", "-t", "tmpfs", "-o"])
            .arg(format!("size={kib}k,mode=1777"))
            .arg("tmpfs")
            .arg(dir.path())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            eprintln!("SKIPPED: could not mount a tmpfs");
            return None;
        }
        Some(Self {
            dir,
            stop: Arc::new(AtomicBool::new(true)),
            filler: None,
        })
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Keep every free byte taken until [`Self::free`].
    fn fill(&mut self) {
        use std::io::Write as _;
        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();
        let file = self.path().join("fill");
        self.filler = Some(std::thread::spawn(move || {
            let block = [0u8; 4096];
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&file)
                .unwrap();
            while !stop.load(Ordering::SeqCst) {
                while f.write_all(&block).is_ok() {}
                std::thread::sleep(Duration::from_millis(2));
            }
        }));
    }

    fn free(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.filler.take() {
            t.join().unwrap();
        }
        let _ = std::fs::remove_file(self.path().join("fill"));
    }

    /// Every file execd keeps under the mount, the filler and each job's two spool files excluded.
    fn job_files(&self) -> usize {
        fn walk(p: &Path, n: &mut usize) {
            for e in std::fs::read_dir(p).into_iter().flatten().flatten() {
                let path: PathBuf = e.path();
                if path.is_dir() {
                    walk(&path, n);
                } else if path
                    .file_name()
                    .is_some_and(|f| f != "fill" && f != "out" && f != "err" && f != "journal")
                {
                    *n += 1;
                }
            }
        }
        let mut n = 0;
        walk(self.path(), &mut n);
        n
    }
}

impl Drop for SmallDisk {
    fn drop(&mut self) {
        self.free();
        let _ = Command::new("sudo")
            .args(["-n", "umount", "-l"])
            .arg(self.path())
            .stderr(Stdio::null())
            .status();
    }
}

/// 64 KiB of `y\n` at a time, so the byte at offset `i` is `y` for even `i` and a newline for odd.
const WRITER: &str = "while :; do yes | head -c 65536; sleep 0.1; done";

fn assert_parity(from: u64, data: &[u8]) {
    for (i, b) in data.iter().enumerate() {
        let want = if (from + i as u64).is_multiple_of(2) {
            b'y'
        } else {
            b'\n'
        };
        assert_eq!(*b, want, "offset {}: output is corrupt", from + i as u64);
    }
}

#[tokio::test]
async fn a_full_jobs_disk_costs_the_output_and_nothing_else() {
    let Some(mut disk) = SmallDisk::mount(1024) else {
        return;
    };
    let jobs_dir = disk.path().to_string_lossy().into_owned();
    let e = Execd::start(&[], &[("EXECD_JOBS_DIR", &jobs_dir)]);

    // Healthy first: output is retained in full from offset 0.
    assert_eq!(
        start_job(&e, "calm", "echo calm; sleep 120").await.status,
        200
    );
    assert_eq!(start_job(&e, "w", WRITER).await.status, 200);
    until("128 KiB retained", || async {
        read(&e, "w", 0).await.end >= 128 << 10
    })
    .await;
    let r = read(&e, "w", 0).await;
    assert_eq!((r.from, r.state.as_str()), (0, "running"));
    assert_eq!(r.data.len() as u64, r.end, "healthy output is whole");
    assert_parity(0, &r.data);

    // The disk fills (and stays full). A job's output is a file on that disk, so its own writes
    // fail as any write to a full disk does (`ENOSPC`); the job itself runs on, and what execd
    // already holds reads back honestly: offsets never move backwards, bytes are the bytes.
    disk.fill();
    let last = std::cell::Cell::new((r.from, r.end));
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        let (last_from, last_end) = last.get();
        let r = read(&e, "w", 0).await;
        assert_eq!(r.state, "running", "the writer must outlive a full disk");
        assert!(
            r.from >= last_from,
            "offset moved back: {} < {last_from}",
            r.from
        );
        assert!(r.end >= last_end, "end moved back: {} < {last_end}", r.end);
        assert!(r.end >= r.from + r.data.len() as u64);
        assert_parity(r.from, &r.data);
        last.set((r.from, r.end));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Asking for the lost bytes is answered from what is held, never with an error.
    let r = read(&e, "w", 0).await;
    assert!(r.from <= r.end);

    // Everything else still works: another job, one-shot, streaming, a new job, /livez.
    let calm = read(&e, "calm", 0).await;
    assert_eq!(
        (calm.state.as_str(), calm.data.as_slice()),
        ("running", &b"calm\n"[..])
    );
    assert_spawns_work_with(&e, "full disk", false).await;
    assert_eq!(livez(&e).await, 200);

    // A refused chunk leaves nothing on the disk behind it.
    assert!(
        disk.job_files() <= 2,
        "{} files pile up on a full disk",
        disk.job_files()
    );

    // Space comes back: new output is retained again, from the first byte written after.
    disk.free();
    assert_eq!(
        start_job(&e, "later", "printf later").await.status,
        200,
        "a job starts again once there is room"
    );
    until("a job started after the disk freed to end", || async {
        read(&e, "later", 0).await.state == "ended"
    })
    .await;
    assert_eq!(read(&e, "later", 0).await.data, b"later");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let e0 = read(&e, "w", 0).await.end;
    until("192 KiB written after the disk freed", || async {
        read(&e, "w", 0).await.end >= e0 + (192 << 10)
    })
    .await;
    let r = read(&e, "w", e0).await;
    assert_eq!(
        r.from, e0,
        "output written after the disk freed was not retained"
    );
    assert!(
        r.data.len() >= 128 << 10,
        "only {} bytes came back",
        r.data.len()
    );
    assert_parity(r.from, &r.data);
    assert_eq!(r.state, "running");

    wipe(&e).await;
    assert_no_panic(&e);
}

// ---- no free file descriptors ----

#[tokio::test]
async fn running_out_of_descriptors_fails_spawns_cleanly_and_recovers() {
    let mut e = Execd::start(&[], &[]);
    let pid = e.pid();
    assert_eq!(
        start_job(&e, "keep", "echo up; sleep 120").await.status,
        200
    );
    until("the job's first line", || async {
        read(&e, "keep", 0).await.end >= 3
    })
    .await;
    // Let the requests so far finish closing before the baseline is taken.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let before = open_fds(pid);
    let original = prlimit(pid, "--nofile");
    set_prlimit(pid, "--nofile", &(before + 14).to_string());

    // Idle connections eat the descriptors one at a time. While one is left for the request
    // itself but not for the pipes of a spawn, the spawn must fail as a result, not a hang.
    let mut idle = Vec::new();
    let mut refused = false;
    for _ in 0..40 {
        idle.push(tokio::net::TcpStream::connect(addr(&e)).await.unwrap());
        tokio::time::sleep(Duration::from_millis(40)).await;
        let Some(a) = try_call(&e, sh("true"), Duration::from_secs(2)).await else {
            break; // the listener has no descriptor for even the request: the next phase
        };
        assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
        let v = a.json();
        if v["exit_code"] == 126 {
            let why = String::from_utf8_lossy(
                &base64::engine::general_purpose::STANDARD
                    .decode(v["stderr"].as_str().unwrap_or(""))
                    .unwrap_or_default(),
            )
            .into_owned();
            assert!(
                why.contains("Too many open files"),
                "unexpected reason: {v}"
            );
            refused = true;
            break;
        }
        assert_eq!(v["exit_code"], 0, "{v}");
    }
    assert!(refused, "no spawn ever failed for lack of descriptors");

    // In that state: spawns fail on every surface, nothing that is running is disturbed, and the
    // listener still answers.
    assert_eq!(livez(&e).await, 200);
    assert_spawn_fails_cleanly(&e, "no descriptors").await;
    let keep = read(&e, "keep", 0).await;
    assert_eq!(
        (keep.state.as_str(), keep.data.as_slice()),
        ("running", &b"up\n"[..])
    );
    assert_no_panic(&e);

    // Further in, there is no descriptor even to accept a connection. That is a request that waits
    // (the kernel holds it), not a crash; closing the idle connections frees the listener.
    for _ in 0..40 {
        match tokio::net::TcpStream::connect(addr(&e)).await {
            Ok(s) => idle.push(s),
            Err(_) => break,
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(e.child.try_wait().unwrap().is_none(), "execd died");
    drop(idle);

    // Capacity recovers when the connections close, and nothing leaked.
    until("spawns to work again", || async {
        try_call(&e, sh("printf ok"), Duration::from_secs(3))
            .await
            .is_some_and(|a| a.status == 200 && a.json()["exit_code"] == 0)
    })
    .await;
    assert_spawns_work(&e, "descriptors back").await;
    assert_eq!(livez(&e).await, 200);
    set_prlimit(pid, "--nofile", &original);
    until("descriptors to be released", || async {
        open_fds(pid) <= before + 2
    })
    .await;
    assert_eq!(read(&e, "keep", 0).await.state, "running");
    wipe(&e).await;
    assert_no_panic(&e);
}

/// Out of descriptors, a new connection is answered `503` and closed, not left waiting in the
/// kernel's queue with no answer; the connections already open are unaffected, and the listener
/// serves again once descriptors are free.
#[tokio::test]
async fn a_connection_with_no_descriptor_left_is_refused_503_not_left_waiting() {
    let mut e = Execd::start(&[], &[]);
    let pid = e.pid();
    tokio::time::sleep(Duration::from_millis(300)).await;
    set_prlimit(pid, "--nofile", &(open_fds(pid) + 6).to_string());

    let mut idle = Vec::new();
    let mut refused = None;
    for _ in 0..40 {
        idle.push(tokio::net::TcpStream::connect(addr(&e)).await.unwrap());
        tokio::time::sleep(Duration::from_millis(40)).await;
        let answer = tokio::time::timeout(
            Duration::from_secs(3),
            send(addr(&e), "GET", "/livez", &[], b""),
        )
        .await
        .expect("a connection past the descriptor limit was left waiting with no answer");
        if answer.status == 503 {
            refused = Some(answer);
            break;
        }
        assert_eq!(answer.status, 200);
    }
    let refused = refused.expect("no connection was ever refused");
    assert_eq!(refused.json()["error"], "out of file descriptors");
    assert!(e.child.try_wait().unwrap().is_none(), "execd died");
    assert!(e.log().contains("refused"), "{}", e.log());

    drop(idle);
    until("the listener to serve again", || async {
        try_call(&e, sh("printf ok"), Duration::from_secs(3))
            .await
            .is_some_and(|a| a.status == 200)
    })
    .await;
}

/// A stock shell's 256 descriptors are raised to the hard limit at start, and the log says so.
#[test]
fn the_soft_descriptor_limit_is_raised_to_the_hard_limit_at_start() {
    let e = Execd::start_with_soft_nofile(256);
    let pid = e.pid();
    let hard = Command::new("prlimit")
        .args([
            "--pid",
            &pid.to_string(),
            "--nofile",
            "--noheadings",
            "--raw",
            "-o",
            "HARD",
        ])
        .output()
        .unwrap();
    let hard = String::from_utf8_lossy(&hard.stdout).trim().to_string();
    let soft = prlimit(pid, "--nofile");
    assert_eq!(soft, hard, "{}", e.log());
    assert!(e.log().contains("open file limit raised"), "{}", e.log());
}

// ---- a slow disk under job output ----

/// The `tail_gate` seam (`jobs.rs` `tail_gate`): job output batches hold, as on a disk that never
/// answers, until [`Gate::open`]. Each batch held names itself with a `held-<n>` file.
struct Gate {
    dir: tempfile::TempDir,
    fault: PathBuf,
}

impl Gate {
    fn closed() -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("closed"), b"").unwrap();
        let fault = dir.path().join("fault");
        std::fs::write(&fault, format!("tail_gate={}\n", dir.path().display())).unwrap();
        Self { dir, fault }
    }

    fn env(&self) -> (&'static str, &str) {
        ("EXECD_TEST_FAULT", self.fault.to_str().unwrap())
    }

    fn held(&self) -> usize {
        std::fs::read_dir(self.dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("held-"))
            .count()
    }

    fn open(&self) {
        let _ = std::fs::remove_file(self.dir.path().join("closed"));
    }
}

/// Starts writing once its start has been answered, then a line every 50 ms until stopped.
const CHATTER: &str = "sleep 0.3; while :; do echo chatter; sleep 0.05; done";

async fn livez_within(e: &Execd, within: Duration) -> Option<u16> {
    tokio::time::timeout(within, send(addr(e), "GET", "/livez", &[], b""))
        .await
        .ok()
        .map(|a| a.status)
}

/// While a job's output write is stuck on the disk (held at the gate, indefinitely, not for a
/// time), an unrelated caller is served: `/livez`, and a command run to completion. Red: with the
/// batch on the runtime (`EXECD_DISK_WRITERS=0`, or `off_runtime` removed) the same stuck write
/// leaves `/livez` unanswered (next test).
#[tokio::test]
async fn a_stuck_output_write_does_not_stall_an_unrelated_caller() {
    let gate = Gate::closed();
    let e = Execd::start(&[], &[gate.env()]);
    assert_eq!(start_job(&e, "slow", CHATTER).await.status, 200);
    until("a batch stuck on the disk", || async { gate.held() == 1 }).await;
    assert_eq!(livez_within(&e, Duration::from_secs(5)).await, Some(200));
    let a = call(&e, sh("printf ok")).await;
    assert_eq!((a.status, a.json()["exit_code"].as_i64()), (200, Some(0)));
    assert_eq!(
        gate.held(),
        1,
        "the write is still stuck while the caller was served"
    );
    gate.open();
    until("the job's output", || async {
        read(&e, "slow", 0).await.end >= 8
    })
    .await;
    wipe(&e).await;
}

/// The off value: `EXECD_DISK_WRITERS=0` writes on the runtime, as before, so the same stuck write
/// stalls every caller until it clears. This is the behaviour the fix removes, shown on the seam.
#[tokio::test]
async fn with_disk_writers_zero_a_stuck_output_write_stalls_every_caller_as_before() {
    let gate = Gate::closed();
    let e = Execd::start(&[("EXECD_DISK_WRITERS", "0")], &[gate.env()]);
    assert_eq!(start_job(&e, "slow", CHATTER).await.status, 200);
    until("a batch stuck on the disk", || async { gate.held() == 1 }).await;
    assert_eq!(livez_within(&e, Duration::from_secs(2)).await, None);
    gate.open();
    assert_eq!(livez_within(&e, Duration::from_secs(5)).await, Some(200));
    wipe(&e).await;
}

/// Ten jobs write through two writer slots with the disk stuck: exactly two batches are on the
/// disk and the other eight have each logged that they wait for a slot (so all ten are accounted
/// for, and none is on the disk unseen). Opening the disk serves every one. Red: without the
/// semaphore all ten reach the disk and none waits.
#[tokio::test]
async fn writer_threads_are_bounded_by_disk_writers_and_the_rest_wait() {
    let gate = Gate::closed();
    let e = Execd::start(
        &[("EXECD_DISK_WRITERS", "2"), ("EXECD_LOG", "debug")],
        &[gate.env()],
    );
    for i in 0..10 {
        assert_eq!(start_job(&e, &format!("w{i}"), CHATTER).await.status, 200);
    }
    let waiting = || e.log().matches("tail waits for a writer slot").count();
    until("eight tails waiting for a slot", || async {
        waiting() >= 8
    })
    .await;
    assert_eq!(gate.held(), 2, "batches on the disk");
    assert_eq!(waiting(), 8, "tails waiting");
    gate.open();
    for i in 0..10 {
        until("every job's output", || async {
            read(&e, &format!("w{i}"), 0).await.end >= 8
        })
        .await;
    }
    wipe(&e).await;
}

/// About 4.9 MB of output (more than three batches of 16 rounds of 64 KiB, within the 8 MiB
/// retention) piles up in the spool while the disk is stuck, then drains in order with every byte
/// once and contiguous offsets. A regression guard for ordering through the batches: it goes red
/// when a batch hands over bytes out of place (an offset advanced short of what was read), not
/// when the batch loop alone is removed, because punching the spool also wakes the tail.
#[tokio::test]
async fn output_past_one_batch_keeps_its_order_and_every_byte() {
    let gate = Gate::closed();
    let e = Execd::start(&[], &[gate.env()]);
    let done = gate.dir.path().join("written");
    let script = format!(
        "sleep 0.3; seq 1 700000; touch {}; sleep 30",
        done.display()
    );
    assert_eq!(start_job(&e, "ord", &script).await.status, 200);
    until("the job to write everything", || async { done.exists() }).await;
    assert_eq!(gate.held(), 1);
    gate.open();
    let want: String = (1..=700_000).map(|i| format!("{i}\n")).collect();
    assert!(want.len() > 3 * 16 * 64 * 1024, "more than three batches");
    let mut got = Vec::new();
    let mut next = 0u64;
    let deadline = Instant::now() + Duration::from_secs(30);
    while got.len() < want.len() {
        let r = read(&e, "ord", next).await;
        assert_eq!(r.from, next, "offsets are contiguous");
        next = r.from + r.data.len() as u64;
        got.extend_from_slice(&r.data);
        assert!(
            Instant::now() < deadline,
            "only {} of {} bytes",
            got.len(),
            want.len()
        );
        if r.data.is_empty() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    assert!(
        got == want.as_bytes(),
        "the bytes differ from what was written"
    );
    wipe(&e).await;
}

/// `EXECD_RAISE_NOFILE=0` leaves the inherited soft limit, as before.
#[test]
fn with_raise_nofile_zero_the_soft_limit_is_left_as_inherited() {
    let e = Execd::start_launched_with("ulimit -Sn 256", &[("EXECD_RAISE_NOFILE", "0")]);
    assert_eq!(prlimit(e.pid(), "--nofile"), "256", "{}", e.log());
    assert!(
        e.log().contains("open file limit soft=256 (unchanged)"),
        "{}",
        e.log()
    );
}

/// `EXECD_SHED_WHEN_OUT_OF_FDS=0` (with the cap off too): a connection past the descriptor limit
/// waits in the kernel's queue with no answer, as before, and is served once descriptors free.
#[tokio::test]
async fn with_shedding_off_a_connection_past_the_descriptor_limit_waits_as_before() {
    let e = Execd::start(
        &[
            ("EXECD_SHED_WHEN_OUT_OF_FDS", "0"),
            ("EXECD_MAX_STREAMS", "0"),
        ],
        &[],
    );
    let pid = e.pid();
    tokio::time::sleep(Duration::from_millis(300)).await;
    set_prlimit(pid, "--nofile", &(open_fds(pid) + 6).to_string());
    let mut idle = Vec::new();
    let mut waited = false;
    for _ in 0..40 {
        idle.push(tokio::net::TcpStream::connect(addr(&e)).await.unwrap());
        tokio::time::sleep(Duration::from_millis(40)).await;
        match livez_within(&e, Duration::from_secs(2)).await {
            Some(s) => assert_eq!(s, 200),
            None => {
                waited = true;
                break;
            }
        }
    }
    assert!(waited, "no connection ever waited");
    assert!(!e.log().contains("refused"), "{}", e.log());
    drop(idle);
    assert_eq!(livez_within(&e, Duration::from_secs(10)).await, Some(200));
}

// ---- a cap on requests in flight ----

/// Open `n` streaming requests that stay open, and return each one's first status line.
async fn hold_streams(e: &Execd, n: usize) -> (Vec<tokio::net::TcpStream>, Vec<String>) {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    let body = serde_json::to_vec(&stream("sleep 8")).unwrap();
    let (mut socks, mut heads) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let mut s = tokio::net::TcpStream::connect(addr(e)).await.unwrap();
        s.write_all(&support::http::request("POST", "/exec", &[], &body))
            .await
            .unwrap();
        let mut buf = vec![0u8; 512];
        let got = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf))
            .await
            .expect("a request over the limit was left with no answer")
            .unwrap();
        heads.push(String::from_utf8_lossy(&buf[..got]).into_owned());
        socks.push(s);
    }
    (socks, heads)
}

/// The `EXECD_MAX_STREAMS` gate (`http.rs`, `InFlight::enter_within`): the request past the cap is
/// answered `503` with a reason and `retry-after`, health still answers, and a freed slot serves.
/// Red: without the gate the fourth is served (`200`).
#[tokio::test]
async fn a_request_past_the_cap_is_answered_503_with_a_reason_and_recovers() {
    let e = Execd::start(&[("EXECD_MAX_STREAMS", "3")], &[]);
    let (mut held, heads) = hold_streams(&e, 3).await;
    assert!(
        heads.iter().all(|h| h.starts_with("HTTP/1.1 200")),
        "{heads:?}"
    );
    let over = call(&e, sh("true")).await;
    assert_eq!(over.status, 503, "{}", String::from_utf8_lossy(&over.body));
    assert_eq!(over.header("retry-after"), Some("1"));
    assert!(over.json()["error"].as_str().unwrap().contains("limit 3"));
    assert_eq!(livez(&e).await, 200, "health is not capped");
    drop(held.pop());
    until("a slot to free", || async {
        call(&e, sh("true")).await.status == 200
    })
    .await;
}

/// `EXECD_MAX_STREAMS=0` is no cap, as before the cap existed.
#[tokio::test]
async fn max_streams_zero_is_no_cap() {
    let e = Execd::start(&[("EXECD_MAX_STREAMS", "0")], &[]);
    assert!(
        e.log().contains("no cap on requests in flight"),
        "{}",
        e.log()
    );
    let (_held, heads) = hold_streams(&e, 6).await;
    assert!(
        heads.iter().all(|h| h.starts_with("HTTP/1.1 200")),
        "{heads:?}"
    );
}

/// A machine whose hard limit is 128 (soft 64): the soft limit is raised to 128 (the most there
/// is), the cap is sized from it (16 requests: `stream_limit`), and 40 concurrent streams are each
/// either served or answered `503`: none hangs, execd never runs out of descriptors, and health
/// answers throughout. Red: without the cap, 3-4 descriptors each exhaust the limit.
#[tokio::test]
async fn a_stingy_descriptor_limit_sizes_the_cap_and_every_stream_is_answered() {
    let e = Execd::start_with_nofile(64, 128);
    let log = e.log();
    assert!(
        log.contains("open file limit raised soft=64 -> 128"),
        "{log}"
    );
    assert!(log.contains("at most 16 requests in flight"), "{log}");
    let (_held, heads) = hold_streams(&e, 40).await;
    let served = heads
        .iter()
        .filter(|h| h.starts_with("HTTP/1.1 200"))
        .count();
    let refused = heads
        .iter()
        .filter(|h| h.starts_with("HTTP/1.1 503"))
        .count();
    assert_eq!((served, refused), (16, 24), "{heads:?}");
    assert!(!e.log().contains("accept failed"), "{}", e.log());
    assert!(e.log().contains("request kind=overload"));
    assert_eq!(livez(&e).await, 200);
}

// ---- no room for another process ----

#[tokio::test]
async fn a_failing_fork_fails_spawns_cleanly_and_recovers() {
    let e = Execd::start(&[], &[]);
    let pid = e.pid();
    assert_eq!(
        start_job(&e, "keep", "echo up; sleep 120").await.status,
        200
    );
    until("the job's first line", || async {
        read(&e, "keep", 0).await.end >= 3
    })
    .await;
    // Every fork by this user fails once the limit is below the processes it already has.
    let original = prlimit(pid, "--nproc");
    set_prlimit(pid, "--nproc", "1");
    let seen = prlimit(pid, "--nproc");
    assert_eq!(seen, "1", "the limit did not apply");

    assert_spawn_fails_cleanly(&e, "no processes").await;
    assert_eq!(livez(&e).await, 200);
    // The job that was running before is untouched, can still be read, and can still be killed
    // (a signal is not a fork).
    let keep = read(&e, "keep", 0).await;
    assert_eq!(
        (keep.state.as_str(), keep.data.as_slice()),
        ("running", &b"up\n"[..])
    );
    let a = call(
        &e,
        job_op(json!({ "op": "kill", "ids": ["keep"], "remove": false, "grace_ms": 200 })),
    )
    .await;
    assert_eq!(a.status, 200);
    assert_eq!(a.json()["survivors"], json!([]));
    until("the killed job to end", || async {
        read(&e, "keep", 0).await.state == "ended"
    })
    .await;
    assert_no_panic(&e);

    set_prlimit(pid, "--nproc", &original);
    assert_spawns_work(&e, "forks back").await;
    assert_ne!(read(&e, "after-forks-back", 0).await.state, "gone");
    wipe(&e).await;
    assert_no_panic(&e);
}

// ---- found by the above: the supervisor of a job whose leftovers had closed their pipes ----

#[tokio::test]
async fn a_job_whose_leftover_closed_its_pipes_still_ends() {
    let e = Execd::start(&[], &[]);
    // The shell exits at once; a descendant outlives it with no pipe open.
    let script = "sleep 2 </dev/null >/dev/null 2>&1 & echo hi";
    assert_eq!(start_job(&e, "left", script).await.status, 200);
    until("the job to report its leftover", || async {
        read(&e, "left", 0).await.state != "running"
    })
    .await;
    until("the job to end once the leftover is gone", || async {
        read(&e, "left", 0).await.state == "ended"
    })
    .await;
    assert_eq!(read(&e, "left", 0).await.data, b"hi\n");
    assert_no_panic(&e);
}
