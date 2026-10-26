//! What execd does when the machine under it fails: a full disk under the jobs directory, no free
//! file descriptors, no room for another process. Each fault must give an honest error (never a
//! panic, never a hang), leave what already runs alone, and be forgotten when it clears. The
//! limits are applied to the running binary with `prlimit`, and the disk is a small tmpfs mounted
//! through `sudo -n` (a test that cannot mount says `SKIPPED`).
//!
//! The spec's rules these encode (`crates/agent/src/exec_endpoint.rs`): a command that cannot be
//! spawned is an ordinary result (v1.1 `exit_code` 127, v1.2 a `stderr` frame then 127), a job that
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

/// The final frame of a v1.2 stream.
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
        "v1.1 ({why}): {}",
        String::from_utf8_lossy(&a.body)
    );
    let v = a.json();
    assert_eq!(v["exit_code"], 127, "v1.1 ({why}): {v}");
    assert!(
        !v["stderr"].as_str().unwrap_or("").is_empty(),
        "v1.1 ({why}): 127 with no reason: {v}"
    );

    let a = call(e, stream("echo never")).await;
    assert_eq!(a.status, 200, "v1.2 ({why})");
    let frames = a.frames();
    assert_eq!(
        final_frame(&frames)["exit_code"],
        127,
        "v1.2 ({why}): {frames:?}"
    );
    assert!(
        frames.iter().any(|f| f.get("stderr").is_some()),
        "v1.2 ({why}): no reason given: {frames:?}"
    );
    assert!(
        !frames.iter().any(|f| f.get("stdout").is_some()),
        "v1.2 ({why}): the command ran: {frames:?}"
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
    let a = call(e, sh("printf ok")).await;
    assert_eq!(
        a.json()["exit_code"],
        0,
        "v1.1 ({why}): {}",
        String::from_utf8_lossy(&a.body)
    );
    assert_eq!(a.json()["stdout"], "ok", "v1.1 ({why})");
    let a = call(e, stream("printf ok")).await;
    let frames = a.frames();
    assert_eq!(
        final_frame(&frames)["exit_code"],
        0,
        "v1.2 ({why}): {frames:?}"
    );
    let id = format!("after-{}", why.replace(' ', "-"));
    let a = start_job(e, &id, "printf ok").await;
    assert_eq!(
        a.status,
        200,
        "job start ({why}): {}",
        String::from_utf8_lossy(&a.body)
    );
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

    /// Every file execd keeps under the mount, the filler excluded.
    fn job_files(&self) -> usize {
        fn walk(p: &Path, n: &mut usize) {
            for e in std::fs::read_dir(p).into_iter().flatten().flatten() {
                let path: PathBuf = e.path();
                if path.is_dir() {
                    walk(&path, n);
                } else if path.file_name().is_some_and(|f| f != "fill") {
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
    let Some(mut disk) = SmallDisk::mount(512) else {
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

    // The disk fills (and stays full). The writer keeps running, and what it wrote reads back
    // honestly: a gap at the front, offsets that never move backwards, bytes that are the bytes.
    disk.fill();
    let last = std::cell::Cell::new((r.from, r.end));
    let started = r.end;
    until("600 KiB written while full", || async {
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
        r.end >= started + (600 << 10)
    })
    .await;
    assert!(
        last.get().0 > 0,
        "a full disk must read back as a gap, not as the old bytes"
    );
    // Asking for the lost bytes is answered from what is held, never with an error.
    let r = read(&e, "w", 0).await;
    assert!(r.from > 0 && r.from <= r.end);

    // Everything else still works: another job, v1.1, v1.2, a new job, /livez.
    let calm = read(&e, "calm", 0).await;
    assert_eq!(
        (calm.state.as_str(), calm.data.as_slice()),
        ("running", &b"calm\n"[..])
    );
    assert_spawns_work(&e, "full disk").await;
    assert_eq!(livez(&e).await, 200);
    let a = start_job(&e, "small", "printf small").await;
    assert_eq!(a.status, 200);
    until("a small job to end", || async {
        read(&e, "small", 0).await.state == "ended"
    })
    .await;
    assert_eq!(read(&e, "small", 0).await.data, b"small");

    // A refused chunk leaves nothing on the disk behind it.
    assert!(
        disk.job_files() <= 2,
        "{} files pile up on a full disk",
        disk.job_files()
    );

    // Space comes back: new output is retained again, from the first byte written after.
    disk.free();
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
        if v["exit_code"] == 127 {
            let why = v["stderr"].as_str().unwrap_or("");
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
