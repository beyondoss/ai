//! The start's idempotency key against the real binary (K1-K8 of `job_kernel::Table::begin`): a start
//! retried with the key of the first send answers the first job and runs nothing, across a restart of
//! execd too; the same key with another command is `409`; the key's record leaves only with its job.
//! Each test is red on a binary that ignores the key (the retry is `409`, "the job exists"), and under
//! the `mut_key_*` feature the comment names.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::Execd;
use support::http::post;

/// A job that appends a line to `mark` each time it runs, then sleeps.
fn script(mark: &std::path::Path) -> String {
    format!("echo run >> {}; sleep 300", mark.display())
}

fn runs(mark: &std::path::Path) -> usize {
    std::fs::read_to_string(mark).map_or(0, |s| s.lines().count())
}

async fn start(
    addr: SocketAddr,
    id: &str,
    key: Option<&str>,
    script: &str,
    extra: Value,
) -> (u16, Value) {
    let mut job = json!({ "op": "start", "scope": "s", "id": id, "hold": false });
    if let Some(k) = key {
        job["idempotency_key"] = json!(k);
    }
    let mut body = json!({ "command": "sh", "args": ["-c", script], "job": job });
    for (k, v) in extra.as_object().unwrap() {
        if k == "hold" || k == "stdin_open" {
            body["job"][k] = v.clone();
        } else {
            body[k] = v.clone();
        }
    }
    let a = post(addr, &[], &body).await;
    (a.status, a.json())
}

async fn job(addr: SocketAddr, op: Value) -> Value {
    let a = post(addr, &[], &json!({ "command": "true", "job": op })).await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    a.json()
}

async fn ids(addr: SocketAddr) -> Vec<String> {
    let l = job(addr, json!({"op":"list","scope":"s"})).await;
    let mut v: Vec<String> = l["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["id"].as_str().unwrap().to_string())
        .collect();
    v.sort();
    v
}

async fn until(what: &str, mut f: impl AsyncFnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !f().await {
        assert!(Instant::now() < deadline, "never: {what}");
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

async fn kill(addr: SocketAddr, ids: &[&str]) {
    job(
        addr,
        json!({"op":"kill","scope":"s","ids":ids,"grace_ms":200,"remove":true}),
    )
    .await;
}

/// K1: the retry of a start whose answer was lost answers the first job, and nothing runs twice. The same key under
/// another id is `409` `key_conflict` (K2). Red under `mut_key_replay_starts` (a second job starts),
/// `mut_key_id_ignored` and on a binary that ignores the key.
#[tokio::test]
async fn a_start_retried_with_its_key_answers_the_first_job_and_runs_nothing() {
    let execd = Execd::start(&[], &[]);
    let a = SocketAddr::V4(execd.addr);
    let dir = tempfile::tempdir().unwrap();
    let mark = dir.path().join("mark");
    let sc = script(&mark);
    let (s1, first) = start(a, "j1", Some("deploy-1"), &sc, json!({})).await;
    assert_eq!(s1, 200, "{first}");
    let (s2, again) = start(a, "j1", Some("deploy-1"), &sc, json!({})).await;
    assert_eq!((s2, &again), (200, &first), "the same id and key replays");
    // The id is part of what the client asked for: another id under the key is a conflict, not a job named otherwise.
    let (s3, other_id) = start(a, "j2", Some("deploy-1"), &sc, json!({})).await;
    assert_eq!(s3, 409, "{other_id}");
    assert_eq!(other_id["code"], "key_conflict");
    until("the job ran", async || runs(&mark) >= 1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(runs(&mark), 1, "the command ran once");
    assert_eq!(ids(a).await, vec!["j1".to_string()], "one job");
    kill(a, &["j1"]).await;
}

/// K2: the same key with another command is refused, and says why; nothing starts. Every field of the
/// spec is in the digest. Red under `mut_key_digest_ignored`.
#[tokio::test]
async fn a_key_used_for_another_command_is_a_409_for_every_field_of_the_spec() {
    let execd = Execd::start(&[], &[]);
    let a = SocketAddr::V4(execd.addr);
    let sc = "sleep 300";
    let base = json!({ "env": { "A": "1", "B": "2" }, "cwd": "/" });
    let (s, v) = start(a, "j1", Some("k"), sc, base.clone()).await;
    assert_eq!(s, 200, "{v}");
    // The env's order is not part of the spec.
    let (s, _) = start(
        a,
        "j1",
        Some("k"),
        sc,
        json!({ "env": { "B": "2", "A": "1" }, "cwd": "/" }),
    )
    .await;
    assert_eq!(s, 200, "the same env in another order is the same spec");
    for (what, script, extra) in [
        ("command", "sleep 301", base.clone()),
        (
            "env value",
            sc,
            json!({ "env": { "A": "1", "B": "3" }, "cwd": "/" }),
        ),
        (
            "env name",
            sc,
            json!({ "env": { "A": "1", "C": "2" }, "cwd": "/" }),
        ),
        ("env gone", sc, json!({ "cwd": "/" })),
        (
            "cwd",
            sc,
            json!({ "env": { "A": "1", "B": "2" }, "cwd": "/tmp" }),
        ),
        ("cwd gone", sc, json!({ "env": { "A": "1", "B": "2" } })),
        (
            "stdin_open",
            sc,
            json!({ "env": { "A": "1", "B": "2" }, "cwd": "/", "stdin_open": true }),
        ),
        (
            "hold",
            sc,
            json!({ "env": { "A": "1", "B": "2" }, "cwd": "/", "hold": true }),
        ),
    ] {
        let (s, v) = start(a, "j9", Some("k"), script, extra).await;
        assert_eq!(s, 409, "{what}: {v}");
        assert_eq!(
            v["error"], "the key was used for a different command or id",
            "{what}"
        );
    }
    // An arg is a field too.
    let a2 = post(
        a,
        &[],
        &json!({"command":"sh","args":["-c", sc, "x"],"env":{"A":"1","B":"2"},"cwd":"/",
                "job":{"op":"start","scope":"s","id":"j1","hold":false,"idempotency_key":"k"}}),
    )
    .await;
    assert_eq!(a2.status, 409, "args");
    assert_eq!(ids(a).await, vec!["j1".to_string()], "nothing else started");
    kill(a, &["j1"]).await;
}

/// K4, K5: an id in the table is the claim, with no key or under another key; a start with no key never
/// replays. A malformed key is a `400`. Red under `mut_unkeyed_replays`.
#[tokio::test]
async fn an_id_in_the_table_is_refused_without_its_key_and_a_bad_key_is_400() {
    let execd = Execd::start(&[], &[]);
    let a = SocketAddr::V4(execd.addr);
    let (s, _) = start(a, "j1", Some("k1"), "sleep 300", json!({})).await;
    assert_eq!(s, 200);
    for key in [None, Some("k2")] {
        let (s, v) = start(a, "j1", key, "sleep 300", json!({})).await;
        assert_eq!(
            (s, v["error"].as_str()),
            (409, Some("the job exists")),
            "{key:?}"
        );
    }
    for bad in ["", "a b", "k/1", &"k".repeat(129)] {
        let (s, v) = start(a, "j3", Some(bad), "sleep 300", json!({})).await;
        assert_eq!(
            (s, v["error"].as_str()),
            (400, Some("bad idempotency key")),
            "{bad:?}"
        );
    }
    // An unkeyed job's id replays nothing either.
    let (s, _) = start(a, "j4", None, "sleep 300", json!({})).await;
    assert_eq!(s, 200);
    let (s, _) = start(a, "j4", None, "sleep 300", json!({})).await;
    assert_eq!(s, 409);
    kill(a, &["j1", "j4"]).await;
}

/// K3 (the refusal clears): a fresh key on a free id starts; the refused key's spec, started under a fresh
/// key, runs; and once the job is removed its key is free (the record left with it, K8), so the key starts a
/// new job. Red under `mut_key_fresh_refused` and `mut_key_take_forgets_none` (a stale record is not what makes
/// the next start run, but it must not make it replay either).
#[tokio::test]
async fn a_refusal_clears_and_a_removed_jobs_key_is_free() {
    let execd = Execd::start(&[], &[]);
    let a = SocketAddr::V4(execd.addr);
    let dir = tempfile::tempdir().unwrap();
    let mark = dir.path().join("mark");
    let sc = script(&mark);
    let (s, _) = start(a, "j1", Some("k"), &sc, json!({})).await;
    assert_eq!(s, 200);
    let (s, _) = start(a, "j1", Some("k"), "sleep 301", json!({})).await;
    assert_eq!(s, 409, "refused");
    let (s, _) = start(a, "j1", Some("k2"), "sleep 301", json!({})).await;
    assert_eq!(s, 409, "the id is taken whatever the key");
    let (s, _) = start(a, "j2", Some("k2"), "sleep 301", json!({})).await;
    assert_eq!(s, 200, "a fresh key on a free id starts");
    // Writing j2's record did not drop j1's (K7): the first key still replays.
    let (s, _) = start(a, "j1", Some("k"), &sc, json!({})).await;
    assert_eq!(s, 200, "the first key's record stayed");
    assert_eq!(runs(&mark), 1);
    kill(a, &["j1"]).await;
    // j1 is gone and so is its key's record (K8), j2 and its record stay.
    assert_eq!(ids(a).await, vec!["j2".to_string()]);
    let (s, _) = start(a, "j2", Some("k2"), "sleep 301", json!({})).await;
    assert_eq!(s, 200, "the other key's record stayed (K7, K8)");
    // The id is taken again, by a job under another key: the removed job's key must not replay it.
    let (s, third) = start(a, "j1", Some("k3"), "sleep 302", json!({})).await;
    assert_eq!(s, 200);
    let (s, fresh) = start(a, "j3", Some("k"), &sc, json!({})).await;
    assert_eq!(s, 200, "{fresh}");
    assert_ne!(fresh, third, "the removed job's key replayed another job");
    assert_eq!(
        ids(a).await,
        vec!["j1".to_string(), "j2".to_string(), "j3".to_string()],
        "the key started a new job"
    );
    kill(a, &["j1", "j2"]).await;
    until("the second run", async || runs(&mark) >= 2).await;
    kill(a, &["j3"]).await;
}

/// K1 and K6: the answer is lost and execd restarts (a `SIGHUP`, which leaves the job running, and a crash),
/// and the retry replays the job: no second job, no second run. The record survives a job that ended too:
/// a keyed record is never evicted before its output. Red under `mut_key_forgotten_on_restart` (an ended job's
/// record) and on a binary whose ledger keeps no key.
#[tokio::test]
async fn a_keyed_start_retried_after_a_restart_of_execd_replays() {
    let state = tempfile::tempdir().unwrap();
    let ledger = state.path().join("ledger");
    let settings = [("EXECD_STATE_DIR", ledger.to_str().unwrap())];
    let dir = tempfile::tempdir().unwrap();
    let mark = dir.path().join("mark");
    let sc = script(&mark);
    let mut first = Execd::start(&settings, &[]);
    let a1 = SocketAddr::V4(first.addr);
    let (s, live) = start(
        a1,
        "live",
        Some("deploy"),
        &sc,
        json!({ "env": { "TOKEN": "s3cr3t-value" } }),
    )
    .await;
    assert_eq!(s, 200);
    let (s, done) = start(a1, "done", Some("build"), "echo built", json!({})).await;
    assert_eq!(s, 200);
    until("done ended", async || {
        let l = job(a1, json!({"op":"list","scope":"s"})).await;
        l["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|j| j["id"] == "done" && j["state"] == "ended")
    })
    .await;
    until("the run", async || runs(&mark) == 1).await;
    // Only the digest is kept: the env's value is in no file of the ledger's directory.
    let saved = std::fs::read_to_string(ledger.join("execd-jobs.json")).unwrap();
    assert!(saved.contains("deploy"), "the key is kept");
    assert!(!saved.contains("s3cr3t-value"), "the env is not");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(i32::try_from(first.child.id()).unwrap()),
        nix::sys::signal::Signal::SIGHUP,
    )
    .unwrap();
    first.child.wait().unwrap();

    let mut second = Execd::start(&settings, &[]);
    let a2 = SocketAddr::V4(second.addr);
    let spec = json!({ "env": { "TOKEN": "s3cr3t-value" } });
    assert_eq!(
        start(a2, "live", Some("deploy"), &sc, spec.clone()).await,
        (200, live.clone())
    );
    assert_eq!(
        start(a2, "done", Some("build"), "echo built", json!({})).await,
        (200, done.clone())
    );
    let (s, v) = start(a2, "live", Some("deploy"), &sc, json!({})).await;
    assert_eq!(
        (s, v["error"].as_str()),
        (409, Some("the key was used for a different command or id"))
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(runs(&mark), 1, "the command ran once, across the restart");
    // A crash leaves the same.
    second.child.kill().unwrap();
    second.child.wait().unwrap();
    let third = Execd::start(&settings, &[]);
    let a3 = SocketAddr::V4(third.addr);
    assert_eq!(
        start(a3, "live", Some("deploy"), &sc, spec).await,
        (200, live)
    );
    assert_eq!(
        start(a3, "done", Some("build"), "echo built", json!({})).await,
        (200, done)
    );
    assert_eq!(ids(a3).await, vec!["done".to_string(), "live".to_string()]);
    kill(a3, &["live", "done"]).await;
}

/// A replayed held start streams the original job from its first byte, as a held start does.
#[tokio::test]
async fn a_held_start_retried_with_its_key_streams_the_first_job() {
    let execd = Execd::start(&[], &[]);
    let a = SocketAddr::V4(execd.addr);
    let dir = tempfile::tempdir().unwrap();
    let mark = dir.path().join("mark");
    let sc = format!("echo run >> {}; echo hi", mark.display());
    let first = post(
        a,
        &[],
        &json!({"command":"sh","args":["-c", &sc],"job":{"op":"start","scope":"s","id":"h1","hold":true,"idempotency_key":"hk"}}),
    )
    .await;
    assert_eq!(first.status, 200);
    let again = post(
        a,
        &[],
        &json!({"command":"sh","args":["-c", &sc],"job":{"op":"start","scope":"s","id":"h1","hold":true,"idempotency_key":"hk"}}),
    )
    .await;
    assert_eq!(
        again.status,
        200,
        "{}",
        String::from_utf8_lossy(&again.body)
    );
    let text = String::from_utf8_lossy(&again.body);
    assert!(text.contains("started"), "{text}");
    assert_eq!(runs(&mark), 1);
}

/// K9: a keyed start whose claim was written and whose spawn was never saved (execd crashed in between) comes back
/// as a job that ended lost; its retry answers it and runs nothing. A ledger holding only that intent stands for
/// the crash. Red under `mut_intent_respawns` (the intent is dropped, so the retry spawns).
#[tokio::test]
async fn a_keyed_start_claimed_and_never_saved_is_lost_and_its_retry_never_spawns() {
    let state = tempfile::tempdir().unwrap();
    let ledger = state.path().join("ledger");
    std::fs::create_dir_all(&ledger).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mark = dir.path().join("mark");
    let sc = script(&mark);
    // What execd's own digest of this spec is: the digest of a first send, read from a live execd's ledger.
    let digest = {
        let probe = tempfile::tempdir().unwrap();
        let pl = probe.path().join("ledger");
        let settings = [("EXECD_STATE_DIR", pl.to_str().unwrap())];
        let e = Execd::start(&settings, &[]);
        let a = SocketAddr::V4(e.addr);
        let (s, _) = start(a, "lost", Some("intent"), &sc, json!({})).await;
        assert_eq!(s, 200);
        let saved: Value =
            serde_json::from_slice(&std::fs::read(pl.join("execd-jobs.json")).unwrap()).unwrap();
        let d = saved["jobs"][0]["idem"]["digest"]
            .as_str()
            .unwrap()
            .to_string();
        kill(a, &["lost"]).await;
        d
    };
    std::fs::remove_file(&mark).ok();
    let file = json!({ "version": 1, "jobs": [{
        "scope": "s", "id": "lost", "label": "", "mark": "EXECD_JOB_MARK_intent", "phase": "spawning",
        "pgid": 0, "code": null, "signal": null, "at": null, "told": null, "kept": { "num": null, "kind": null },
        "seen": 0, "started": 1, "spool": null,
        "idem": { "key": "intent", "digest": digest },
    }]});
    std::fs::write(
        ledger.join("execd-jobs.json"),
        serde_json::to_vec(&file).unwrap(),
    )
    .unwrap();
    let settings = [("EXECD_STATE_DIR", ledger.to_str().unwrap())];
    let e = Execd::start(&settings, &[]);
    let a = SocketAddr::V4(e.addr);
    // The claim is a job that ended lost, not nothing.
    let l = job(a, json!({"op":"list","scope":"s"})).await;
    assert_eq!(l["jobs"][0]["id"], "lost", "{l}");
    assert_eq!(l["jobs"][0]["state"], "ended");
    assert_eq!(l["jobs"][0]["code"], -1);
    // The retry is answered by it, however often it comes, and nothing runs.
    for _ in 0..2 {
        let (s, v) = start(a, "lost", Some("intent"), &sc, json!({})).await;
        assert_eq!((s, &v["started"]), (200, &json!(0)), "{v}");
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(runs(&mark), 0, "the command never ran");
}
