//! The jobs ledger: where its write runs, and what a restart reads of it.
//!
//! `execd-jobs.json` is one file, replaced whole on every change of a job's life. Two costs scaled with
//! a table of thousands of ended entries (`EXECD_JOBS_ENDED_MAX = 4096`): the write's 1.5 MB sat on the
//! runtime's thread, and recovery scanned `/proc` once per row before it learned the row had ended.
//! Each has a test here that goes red when it is taken away.
//!
//! ```sh
//! cargo test -p beyond-execd --test ledger
//! cargo test -p beyond-execd --test ledger --features mut_ledger_write_inline   # red
//! cargo test -p beyond-execd --test ledger --features mut_recover_scans_ended   # red
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::Execd;
use support::http::{post, send};

fn addr(e: &Execd) -> SocketAddr {
    SocketAddr::V4(e.addr)
}

/// A fault file with these lines (`fault.rs`), for a debug build.
fn fault(dir: &Path, lines: &str) -> String {
    let f = dir.join("fault");
    std::fs::write(&f, lines).unwrap();
    f.to_str().unwrap().to_string()
}

/// Settings with a state directory that survives a restart, and a fault when one is asked for.
fn settings<'a>(
    state: &'a Path,
    jobs: &'a Path,
    fault: Option<&'a str>,
) -> Vec<(&'a str, &'a str)> {
    let mut s = vec![
        ("EXECD_STATE_DIR", state.to_str().unwrap()),
        ("EXECD_JOBS_DIR", jobs.to_str().unwrap()),
        ("EXECD_JOB_OUTPUT_MS", "0"),
        ("EXECD_JOB_ENTRY_MS", "0"),
        ("EXECD_JOBS_ENDED_BYTES", "0"),
        ("EXECD_JOBS_ENDED_MAX", "0"),
    ];
    if let Some(f) = fault {
        s.push(("EXECD_TEST_FAULT", f));
    }
    s
}

async fn start(a: SocketAddr, id: &str, script: &str) {
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
}

async fn op(a: SocketAddr, job: Value) -> Value {
    post(a, &[], &json!({ "command": "true", "job": job }))
        .await
        .json()
}

async fn keep(a: SocketAddr, id: &str) {
    let r = post(
        a,
        &[],
        &json!({ "command": "true",
                 "job": { "op": "keep", "scope": "s", "id": id, "num": 1, "kind": "background", "seen": 1 } }),
    )
    .await;
    assert_eq!(r.status, 200, "keep {id}: {}", r.json());
}

async fn until_state(a: SocketAddr, id: &str, state: &str) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let r = op(
            a,
            json!({ "op": "read", "scope": "s", "id": id, "offset": 0, "max": 16 }),
        )
        .await;
        if r["state"] == state {
            return;
        }
        assert!(Instant::now() < deadline, "{id} never became {state}: {r}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The ledger's row for `id`, as it is on disk right now: no polling, no sleep.
fn row(state: &Path, id: &str) -> Value {
    let v: Value =
        serde_json::from_slice(&std::fs::read(state.join("execd-jobs.json")).unwrap()).unwrap();
    v["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == id)
        .cloned()
        .unwrap_or(Value::Null)
}

/// **A life change's ledger write is off the runtime thread.** Every ledger write is held for 400 ms
/// (`ledger_slow`). A `keep` waits for its own write — the file is the promise its answer makes — but
/// nothing else does: `GET /livez` and a `read` answer while it is in flight. Written inline, both wait
/// out the 400 ms with it, which is what a 1.5 MB file at a full table does to every request.
#[tokio::test]
async fn a_life_changes_ledger_write_holds_nobody_else() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault(root.path(), "ledger_slow=400\n");
    let e = Execd::start(&settings(&state, &jobs, Some(&fault)), &[]);
    let a = addr(&e);
    start(a, "j", "sleep 30").await;
    until_state(a, "j", "running").await;

    let keeping = tokio::spawn(async move {
        let t = Instant::now();
        keep(a, "j").await;
        t.elapsed()
    });
    // The write is under way (the keep's answer is behind it): everything else must not care.
    tokio::time::sleep(Duration::from_millis(120)).await;
    let t = Instant::now();
    let live = send(a, "GET", "/livez", &[], b"").await;
    let live_ms = t.elapsed().as_millis();
    let t = Instant::now();
    op(
        a,
        json!({ "op": "read", "scope": "s", "id": "j", "offset": 0, "max": 16 }),
    )
    .await;
    let read_ms = t.elapsed().as_millis();
    let keep_ms = keeping.await.unwrap().as_millis();

    assert!(live.status == 200, "livez: {}", live.status);
    assert!(
        keep_ms >= 380,
        "the fault never held the write: {keep_ms} ms"
    );
    assert!(
        live_ms < 200 && read_ms < 200,
        "a request waited on the ledger write: livez {live_ms} ms, read {read_ms} ms, while a keep \
         was writing for {keep_ms} ms"
    );
}

/// **The ledger is current when the answer that promised it arrives.** With every write held 250 ms, a
/// `keep` answers only after its write landed: the row on disk already carries the keep the moment the
/// answer is read, no polling. A write that is fired and forgotten would answer first and land later.
#[tokio::test]
async fn a_life_change_is_on_disk_when_its_answer_arrives() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault(root.path(), "ledger_slow=250\n");
    let e = Execd::start(&settings(&state, &jobs, Some(&fault)), &[]);
    let a = addr(&e);
    start(a, "j", "true").await;
    until_state(a, "j", "ended").await;
    assert!(
        row(&state, "j")["kept"]["num"].is_null(),
        "the job carries no number before the keep: {}",
        row(&state, "j")
    );
    let t = Instant::now();
    keep(a, "j").await;
    let took = t.elapsed();
    assert!(took >= Duration::from_millis(230), "{took:?}");
    assert_eq!(
        row(&state, "j")["kept"]["num"],
        json!(1),
        "the answer came before the ledger had the keep"
    );
}

/// **Overlapping writes lose nothing.** A life change's write and a retention sweep's write are the same
/// `Ledger::save`, serialized by the file's writer lock with the snapshot taken under it, so whichever
/// lands second carries the whole table: neither can overwrite the other with a state that predates it.
/// Held at 300 ms a write, a `keep` is in flight while a sweep tombstones another job's output; both
/// facts are in the file afterwards.
#[tokio::test]
async fn a_write_that_overlaps_a_sweep_loses_nothing() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault(root.path(), "ledger_slow=300\n");
    let mut settings = settings(&state, &jobs, Some(&fault));
    settings.retain(|(k, _)| *k != "EXECD_JOB_OUTPUT_MS");
    settings.push(("EXECD_JOB_OUTPUT_MS", "1"));
    let e = Execd::start(&settings, &[]);
    let a = addr(&e);
    start(a, "k", "sleep 30").await;
    until_state(a, "k", "running").await;
    const N: usize = 100_000;
    start(a, "o", &format!("head -c {N} /dev/zero | tr '\\0' 'x'")).await;
    until_state(a, "o", "ended").await;

    // The keep's write is in flight when the sweep's write is asked for.
    let keeping = tokio::spawn(async move { keep(a, "k").await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if row(&state, "o")["out"] == json!(N) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the sweep never recorded its tombstone: {}",
            row(&state, "o")
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    keeping.await.unwrap();
    assert_eq!(row(&state, "o")["out"], json!(N), "the tombstone was lost");
    assert_eq!(
        row(&state, "k")["kept"],
        json!({"num": 1, "kind": "background"}),
        "the keep was lost"
    );
}

/// One ledger row for an ended job, as `ledger.rs` writes one: a realistic id and argv, a mark, a dead
/// group's number, no spool.
fn ended_row(i: usize, at: u64) -> Value {
    json!({
        "scope": "s",
        "id": format!("job_01J8Z9K3M4N5P6Q7R8S9T0V1W{i:05}"),
        "label": "",
        "command": "bash",
        "args": ["-lc", "cargo test --workspace --release"],
        "mark": format!("{:016x}", 0x9F3A_0000u64 + i as u64),
        "phase": "ended",
        "pgid": 400_000 + i as i64,
        "code": 0,
        "signal": null,
        "at": at,
        "told": true,
        "kept": {"num": null, "kind": "background"},
        "seen": 1,
        "started": at - 1,
        "spool": null,
        "term": null,
        "idem": null,
    })
}

/// **Recovery does not scan `/proc` for rows already saved as `ended`.** 4000 such rows: asked for its
/// group, each costs one scan of every process on the host (1.8 ms with 86 here, and more on a busier
/// one — 7.0 s at 4096 rows), and the answer is one `execd_jobs::recover` discards for that phase.
/// Skipped, the start is the parse and the table insert. The bound is generous on the side that cannot
/// flake green: without the skip the scans alone are several times it here.
#[tokio::test]
async fn a_restart_does_not_scan_the_proc_table_once_per_ended_row() {
    let root = tempfile::tempdir().unwrap();
    let state: PathBuf = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    std::fs::create_dir(&state).unwrap();
    const N: usize = 4000;
    let at = now() - 60;
    let rows: Vec<Value> = (0..N).map(|i| ended_row(i, at)).collect();
    std::fs::write(
        state.join("execd-jobs.json"),
        serde_json::to_vec(&json!({ "version": 1, "jobs": rows })).unwrap(),
    )
    .unwrap();

    let t = Instant::now();
    let e = Execd::start(&settings(&state, &jobs, None), &[]);
    let a = addr(&e);
    let answer = op(a, json!({ "op": "list", "scope": "s" })).await;
    let start = t.elapsed();
    assert_eq!(answer["jobs"].as_array().unwrap().len(), 512);
    assert_eq!(answer["more"], json!(N - 512));
    assert!(
        start < Duration::from_millis(2500),
        "a restart of {N} ended rows took {start:?}: recovery is asking `/proc` for rows the ledger \
         already says have ended"
    );
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
