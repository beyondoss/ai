//! The jobs ledger: where its write runs, what a life change costs, and what a restart reads of it.
//!
//! The ledger is two files: `execd-jobs.json`, the table whole, and `execd-jobs.log`, one framed record
//! per change of a job's life since (`ledger.rs`). A change used to replace the file whole — 1.5 MB at
//! the 4096-entry ceiling, on the runtime's thread — and recovery scanned `/proc` once per row before it
//! learned the row had ended. Each property here has a test that goes red when it is taken away, and each
//! mutation has a `# red` line:
//!
//! ```sh
//! cargo test -p beyond-execd --test ledger
//! cargo test -p beyond-execd --test ledger --features mut_ledger_write_inline        # red
//! cargo test -p beyond-execd --test ledger --features mut_recover_scans_ended        # red
//! cargo test -p beyond-execd --test ledger --features mut_ledger_append_skipped      # red
//! cargo test -p beyond-execd --test ledger --features mut_ledger_appends_past_tail   # red
//! cargo test -p beyond-execd --test ledger --features mut_ledger_trusts_checksum     # red
//! cargo test -p beyond-execd --test ledger --features mut_ledger_ignores_base        # red
//! cargo test -p beyond-execd --test ledger --features mut_ledger_never_folds         # red
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::Execd;
use support::http::{post, send, try_post};

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

/// Wait for the execd under test to exit on its own, with the request that kills it polled alongside: a
/// blocking `wait` would hold the very runtime that request needs.
async fn died(e: &mut Execd, req: impl std::future::Future) -> Option<i32> {
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut req = Box::pin(req);
    let mut done = false;
    loop {
        if let Some(s) = e.child.try_wait().unwrap() {
            return s.code();
        }
        assert!(Instant::now() < deadline, "execd never exited");
        if done {
            tokio::time::sleep(Duration::from_millis(20)).await;
        } else {
            // The answer is never read: the request's only job is to be in flight when execd dies.
            done = tokio::time::timeout(Duration::from_millis(20), &mut req)
                .await
                .is_ok();
        }
    }
}

/// The `list` answer's row for `id` — what the server, or the execd that recovered from it, says the job is
/// — or `null` for a job it does not have. Never the file: what a restart reads is the contract, so the tests
/// here ask the server, not the bytes.
async fn listed(a: SocketAddr, id: &str) -> Value {
    let rows = op(a, json!({ "op": "list", "scope": "s" })).await;
    rows["jobs"]
        .as_array()
        .and_then(|js| js.iter().find(|j| j["id"] == id).cloned())
        .unwrap_or(Value::Null)
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

/// **The ledger has the change when the answer that promised it arrives.** With every write held 250 ms,
/// a `keep` answers only after its record reached the log — and the server is killed the instant the answer
/// is in hand, as a crash at the worst moment is. The next execd recovers the job with the keep, so the
/// record was on disk before the answer, not fired and forgotten.
#[tokio::test]
async fn a_life_change_is_on_disk_when_its_answer_arrives() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault(root.path(), "ledger_slow=250\n");
    let settings = settings(&state, &jobs, Some(&fault));
    let mut e = Execd::start(&settings, &[]);
    let a = addr(&e);
    start(a, "j", "true").await;
    until_state(a, "j", "ended").await;
    assert!(
        listed(a, "j").await["num"].is_null(),
        "the job carries no number before the keep"
    );
    let t = Instant::now();
    keep(a, "j").await;
    let took = t.elapsed();
    assert!(took >= Duration::from_millis(230), "{took:?}");
    e.child.kill().unwrap();
    e.child.wait().unwrap();
    let e = Execd::start(&settings, &[]);
    let row = listed(addr(&e), "j").await;
    assert_eq!(
        row["num"],
        json!(1),
        "the answer came before the ledger had the keep: {row}"
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

/// **A record cut short is dropped, not believed — and the log is folded, not appended past.** The next
/// record reaches the file down to its 24th byte and the process dies there (`ledger_append_crash`), the way
/// a crash in the middle of an append leaves it: four bytes of length, four of checksum, and part of a row.
/// The complete record before it survives, the cut one leaves no job behind, and a change made after the
/// restart is in the ledger a restart later — which holds only because recovery folded the torn log instead
/// of writing after the fragment, where every later record would be unreadable behind it.
#[tokio::test]
async fn a_record_cut_short_is_dropped_and_the_log_folded() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault(root.path(), "");
    let settings = settings(&state, &jobs, Some(&fault));
    let mut e = Execd::start(&settings, &[]);
    let a = addr(&e);
    start(a, "a", "sleep 20").await;
    until_state(a, "a", "running").await;
    std::fs::write(&fault, "ledger_append_crash=24\n").unwrap();
    let body = json!({ "command": "sh", "args": ["-c", "sleep 20"],
                       "job": { "op": "start", "scope": "s", "id": "b", "hold": false } });
    let code = died(&mut e, try_post(a, &[], &body)).await;
    assert_eq!(code, Some(87), "the append never reached its fault");
    std::fs::remove_file(&fault).unwrap();

    let mut e = Execd::start(&settings, &[]);
    let a = addr(&e);
    assert_eq!(
        listed(a, "a").await["state"],
        json!("running"),
        "the whole record before the cut was dropped too"
    );
    assert!(
        listed(a, "b").await.is_null(),
        "the half-written record became a job"
    );
    start(a, "c", "true").await;
    until_state(a, "c", "ended").await;
    e.child.kill().unwrap();
    e.child.wait().unwrap();
    let e = Execd::start(&settings, &[]);
    assert_eq!(
        listed(addr(&e), "c").await["state"],
        json!("ended"),
        "a change made after the torn tail was lost: the log was appended to, not folded"
    );
}

/// **A record whose checksum does not match is not applied.** One byte of the newest record's row is changed
/// for another — the record still parses, and now says the job exited 1 where it exited 0 — and the answer
/// after the restart is what the record before it left: the exit code, and no keep. Believed, the changed row
/// would show both.
#[tokio::test]
async fn a_record_that_does_not_checksum_is_not_applied() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let settings = settings(&state, &jobs, None);
    let mut e = Execd::start(&settings, &[]);
    let a = addr(&e);
    start(a, "c", "exit 0").await;
    until_state(a, "c", "ended").await;
    keep(a, "c").await;
    e.child.kill().unwrap();
    e.child.wait().unwrap();

    let log = std::fs::read(state.join("execd-jobs.log")).unwrap();
    let at = log
        .windows(8)
        .rposition(|w| w == b"\"code\":0")
        .expect("the newest record's exit code");
    let mut bad = log;
    bad[at + 7] = b'1';
    std::fs::write(state.join("execd-jobs.log"), &bad).unwrap();

    let e = Execd::start(&settings, &[]);
    let row = listed(addr(&e), "c").await;
    assert_eq!(
        row["code"],
        json!(0),
        "a record the checksum rejects was applied: {row}"
    );
    assert!(
        row["num"].is_null(),
        "the rejected record's keep was applied: {row}"
    );
}

/// **A crash between the fold's two files is the old state or the new one, never a mix.** The fold renames
/// the table and the process dies before its log is renamed (`ledger_fold_crash`), so the log on disk still
/// names the *old* table — and is dropped. The change that triggered the fold (a `keep`) is there, because the
/// new table has it; applied anyway, the stale log would put the row back to the state before the keep.
#[tokio::test]
async fn a_crash_between_the_fold_and_its_log_leaves_the_new_table() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let fault = fault(root.path(), "");
    let settings = settings(&state, &jobs, Some(&fault));
    let mut e = Execd::start(&settings, &[]);
    let a = addr(&e);
    start(a, "f", "sleep 20").await;
    until_state(a, "f", "running").await;
    // The keep folds — the log's one record goes into the table — and the crash comes between the two files.
    std::fs::write(&fault, "ledger_fold=1\nledger_fold_crash=exit\n").unwrap();
    let body = json!({ "command": "true",
                       "job": { "op": "keep", "scope": "s", "id": "f", "num": 1,
                                "kind": "background", "seen": 1 } });
    let code = died(&mut e, try_post(a, &[], &body)).await;
    assert_eq!(code, Some(88), "the fold never reached its fault");
    std::fs::remove_file(&fault).unwrap();

    let e = Execd::start(&settings, &[]);
    let row = listed(addr(&e), "f").await;
    assert_eq!(row["state"], json!("running"), "{row}");
    assert_eq!(
        row["num"],
        json!(1),
        "the log of the fold that did not finish was applied over its table: {row}"
    );
}

/// Seeded rows for the two size tests: `n` ended jobs of a realistic size, the shape `ledger.rs` writes.
fn seed(state: &Path, n: usize, at: u64) {
    std::fs::create_dir_all(state).unwrap();
    let rows: Vec<Value> = (0..n).map(|i| ended_row(i, at)).collect();
    std::fs::write(
        state.join("execd-jobs.json"),
        serde_json::to_vec(&json!({ "version": 1, "jobs": rows })).unwrap(),
    )
    .unwrap();
}

fn size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// **A life change writes one record, not the table.** With 1024 ended rows adopted, one `keep` leaves
/// `execd-jobs.json` byte for byte as it was and adds under a kilobyte to the log — where replacing the table
/// wrote 386 KB for the same change. At the ceiling that is 1.16 MB a cycle against about a kilobyte.
#[tokio::test]
async fn a_life_change_appends_a_record_and_leaves_the_table_alone() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    seed(&state, 1024, now() - 60);
    let e = Execd::start(&settings(&state, &jobs, None), &[]);
    let a = addr(&e);
    // A restart adopts the table and folds it, so both files are where the last write left them.
    listed(a, "none").await;
    let (table, _) = (size(&state.join("execd-jobs.json")), ());
    let log = size(&state.join("execd-jobs.log"));
    keep(a, "job_01J8Z9K3M4N5P6Q7R8S9T0V1W00000").await;
    assert_eq!(
        size(&state.join("execd-jobs.json")),
        table,
        "the change rewrote the table"
    );
    let grew = size(&state.join("execd-jobs.log")) - log;
    assert!(
        (1..1024).contains(&grew),
        "the change wrote {grew} bytes, not one record"
    );
}

/// **A restart answers `list` in the order the table had.** The file names the rows in order, and the log's records are
/// applied in place — a row it changes keeps its place, a row it adds goes last, a row it deletes leaves its place empty
/// rather than shifting the rest — so a client that reattaches after a restart reads the same order it read before.
#[tokio::test]
async fn a_restart_keeps_the_tables_order() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    let ids: Vec<String> = (0..5).map(|i| format!("job_z{i}")).collect();
    let rows: Vec<Value> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let mut r = ended_row(i, now() - 60);
            r["id"] = json!(id);
            r
        })
        .collect();
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("execd-jobs.json"),
        serde_json::to_vec(&json!({ "version": 1, "jobs": rows })).unwrap(),
    )
    .unwrap();
    let mut e = Execd::start(&settings(&state, &jobs, None), &[]);
    let a = addr(&e);
    // A change of a middle row (a log record) and a new one (appended after it), then a crash.
    keep(a, &ids[2]).await;
    start(a, "job_last", "true").await;
    until_state(a, "job_last", "ended").await;
    e.child.kill().unwrap();
    e.child.wait().unwrap();

    let e = Execd::start(&settings(&state, &jobs, None), &[]);
    let answer = op(addr(&e), json!({ "op": "list", "scope": "s" })).await;
    let order: Vec<String> = answer["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["id"].as_str().unwrap().to_string())
        .collect();
    let mut want = ids.clone();
    want.push("job_last".into());
    assert_eq!(order, want, "the restart reordered the table: {answer}");
}

/// **The log is folded before it outgrows the table.** With 256 ended rows (a ~90 KB file) and 400 life
/// changes, the log never passes the documented bound — the size of the table, with a 64 KiB floor — because
/// the write that would pass it folds instead; and the fold happens. Without it the log reaches 150 KB and
/// keeps going: the files are unbounded again, which is the whole cost this change exists to remove.
#[tokio::test]
async fn the_log_is_folded_before_it_outgrows_the_table() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let jobs = root.path().join("jobs");
    std::fs::create_dir(&jobs).unwrap();
    const N: usize = 256;
    seed(&state, N, now() - 60);
    let e = Execd::start(&settings(&state, &jobs, None), &[]);
    let a = addr(&e);
    listed(a, "none").await;
    let table = size(&state.join("execd-jobs.json"));
    let bound = table.max(64 * 1024) + 4096;
    let mut folds = 0;
    let mut before = size(&state.join("execd-jobs.log"));
    for i in 0..400 {
        let id = format!("job_01J8Z9K3M4N5P6Q7R8S9T0V1W{:05}", i % N);
        let r = post(
            a,
            &[],
            &json!({ "command": "true",
                     "job": { "op": "keep", "scope": "s", "id": id, "num": i + 1,
                              "kind": "background", "seen": 1 } }),
        )
        .await;
        assert_eq!(r.status, 200, "keep {id}: {}", r.json());
        let log = size(&state.join("execd-jobs.log"));
        assert!(
            log <= bound,
            "after {} changes the log is {log} bytes, past the {bound} of the table it sits on",
            i + 1
        );
        assert!(
            log + size(&state.join("execd-jobs.json")) <= 2 * bound,
            "the two files passed twice the table's bound after {} changes",
            i + 1
        );
        if log < before {
            folds += 1;
        }
        before = log;
    }
    assert!(folds >= 1, "the log was never folded: {folds}");
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
