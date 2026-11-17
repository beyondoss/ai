//! The job protocol's additions against the real binary: a command's own `env`, a job's stderr
//! told from its stdout on one clock, positional writes to a job's stdin, and the `signal` op.
//! Each test is red on a binary without the feature (the op is unknown or the field ignored).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};
use support::Execd;
use support::http::post;

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

fn unb64(v: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(v.as_str().unwrap_or(""))
        .unwrap()
}

async fn job(addr: SocketAddr, op: Value) -> (u16, Value) {
    let a = post(addr, &[], &json!({ "command": "true", "job": op })).await;
    (a.status, a.json())
}

async fn start(addr: SocketAddr, scope: &str, id: &str, cmd: &[&str], extra: Value) -> Value {
    let mut j = json!({ "op": "start", "scope": scope, "id": id, "hold": false });
    // `env` belongs to the request, everything else to the job.
    let mut extra = extra.as_object().unwrap().clone();
    let env = extra.remove("env");
    j.as_object_mut().unwrap().extend(extra);
    let mut body = json!({ "command": cmd[0], "args": &cmd[1..], "job": j });
    if let Some(env) = env {
        body["env"] = env;
    }
    let a = post(addr, &[], &body).await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    a.json()
}

async fn read_all(addr: SocketAddr, scope: &str, id: &str) -> (Vec<u8>, Vec<(u64, u64)>, Value) {
    let (mut out, mut ranges, mut at) = (Vec::new(), Vec::new(), 0u64);
    loop {
        let (s, v) = job(
            addr,
            json!({"op":"read","scope":scope,"id":id,"offset":at,"max":1 << 20}),
        )
        .await;
        assert_eq!(s, 200);
        let data = unb64(&v["data"]);
        for r in v["stderr_ranges"].as_array().cloned().unwrap_or_default() {
            ranges.push((r[0].as_u64().unwrap(), r[1].as_u64().unwrap()));
        }
        at = v["offset"].as_u64().unwrap() + data.len() as u64;
        out.extend(data);
        if at >= v["end"].as_u64().unwrap() {
            return (out, ranges, v);
        }
    }
}

async fn wait_ended(addr: SocketAddr, scope: &str, id: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (_, v) = job(
            addr,
            json!({"op":"read","scope":scope,"id":id,"offset":0,"max":1}),
        )
        .await;
        if v["state"] == "ended" {
            return v;
        }
        assert!(Instant::now() < deadline, "job never ended: {v}");
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

// ---- env -------------------------------------------------------------------------------------

#[tokio::test]
async fn env_is_applied_refused_when_hostile_and_never_reaches_execd_names() {
    let execd = Execd::start(&[], &[("EXECD_FOO", "secret")]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    let run = |script: &str, env: Value| {
        let body = json!({"command":"sh","args":["-c",script],"env":env,"timeout_ms":10000});
        async move { post(addr, &[], &body).await }
    };
    let a = run("printf %s \"$A:$HOME\"", json!({"A":"x y","HOME":"/h"})).await;
    assert_eq!(a.status, 200);
    assert_eq!(unb64(&a.json()["stdout"]), b"x y:/h");
    // execd's own variables are scrubbed, and a request cannot set one.
    let a = run("printf %s \"${EXECD_FOO-unset}\"", json!({})).await;
    assert_eq!(unb64(&a.json()["stdout"]), b"unset");
    let marker = execd.dir.path().join("ran");
    let touch = format!("touch {}", marker.display());
    let too_many: serde_json::Map<String, Value> =
        (0..129).map(|i| (format!("V{i}"), json!("1"))).collect();
    let huge = json!({"BIG": "x".repeat(33 * 1024)});
    for env in [
        json!({"EXECD_FOO":"y"}),
        json!({"EXECD_JOB_MARK":"x"}),
        json!({"EXECD_CMD_ENV":"x"}),
        json!({"1A":"x"}),
        json!({"A-B":"x"}),
        json!({"":"x"}),
        json!({"A":"nul\u{0}"}),
        json!({"A":1}),
        json!(["A"]),
        Value::Object(too_many),
        huge,
    ] {
        let a = run(&touch, env.clone()).await;
        assert_eq!(a.status, 400, "{env:.80}");
        assert!(!String::from_utf8_lossy(&a.body).contains("secret"));
    }
    assert!(!marker.exists(), "a refused request ran its command");
    // A job start is held to the same rules, and ignores env on ops that run nothing.
    let a = post(
        addr,
        &[],
        &json!({"command":"sh","args":["-c",&touch],"env":{"EXECD_X":"1"},
                "job":{"op":"start","scope":"s","id":"a1","hold":false}}),
    )
    .await;
    assert_eq!(a.status, 400);
    let a = post(
        addr,
        &[],
        &json!({"command":"true","env":{"EXECD_X":"1"},"job":{"op":"list","scope":"s"}}),
    )
    .await;
    assert_eq!(a.status, 200);
    assert!(!marker.exists());
    // Env values are not logged.
    run("true", json!({"A":"value-that-must-not-be-logged"})).await;
    assert!(!execd.log().contains("value-that-must-not-be-logged"));
}

#[tokio::test]
async fn a_jobs_env_is_its_own_and_its_mark_cannot_be_overridden() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    start(
        addr,
        "s1",
        "a1",
        &["sh", "-c", "printf %s \"$VFY\""],
        json!({"env":{"VFY":"one"}}),
    )
    .await;
    start(
        addr,
        "s2",
        "a1",
        &["sh", "-c", "printf %s \"${VFY-none}\""],
        json!({}),
    )
    .await;
    wait_ended(addr, "s1", "a1").await;
    wait_ended(addr, "s2", "a1").await;
    assert_eq!(read_all(addr, "s1", "a1").await.0, b"one");
    assert_eq!(read_all(addr, "s2", "a1").await.0, b"none");
}

// ---- streams ---------------------------------------------------------------------------------

#[tokio::test]
async fn stderr_is_told_from_stdout_on_one_clock() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    // 300 alternations: more than a window of ranges is not reached, but frames must split.
    let script = "i=0; while [ $i -lt 300 ]; do printf o; printf e >&2; i=$((i+1)); sleep 0.001 2>/dev/null; done";
    start(addr, "st", "a1", &["sh", "-c", script], json!({})).await;
    wait_ended(addr, "st", "a1").await;
    let (data, ranges, _) = read_all(addr, "st", "a1").await;
    assert_eq!(data.len(), 600);
    let mut prev = 0;
    for &(a, n) in &ranges {
        assert!(n > 0 && a >= prev);
        prev = a + n;
    }
    for (i, b) in data.iter().enumerate() {
        let in_range = ranges
            .iter()
            .any(|&(a, n)| (i as u64) >= a && (i as u64) < a + n);
        assert_eq!(in_range, *b == b'e', "byte {i}");
    }
    // An attach: frames re-concatenate to the same bytes, each keyed by what it holds.
    let a = post(
        addr,
        &[],
        &json!({"command":"true","stream":true,
                "job":{"op":"attach","scope":"st","id":"a1","offset":0}}),
    )
    .await;
    let mut joined = Vec::new();
    for f in a.frames() {
        for (key, want) in [("stdout", b'o'), ("stderr", b'e')] {
            if f[key].is_string() {
                let p = unb64(&f[key]);
                assert!(p.iter().all(|&b| b == want), "{key} frame holds {p:?}");
                joined.extend(p);
            }
        }
    }
    assert_eq!(joined, data);
}

#[tokio::test]
async fn a_read_answers_at_most_1024_ranges_and_the_client_reads_on() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    // One byte at a time, strictly alternating, from one pipe at a time (a pause between each).
    let script = "i=0; while [ $i -lt 1500 ]; do printf o; sleep 0.002 2>/dev/null; printf e >&2; sleep 0.002 2>/dev/null; i=$((i+1)); done";
    start(addr, "rg", "a1", &["sh", "-c", script], json!({})).await;
    wait_ended(addr, "rg", "a1").await;
    let (_, v) = job(
        addr,
        json!({"op":"read","scope":"rg","id":"a1","offset":0,"max":1 << 20}),
    )
    .await;
    let n = v["stderr_ranges"].as_array().map_or(0, Vec::len);
    assert!(n <= 1024, "{n} ranges in one answer");
    let (data, ranges, _) = read_all(addr, "rg", "a1").await;
    assert_eq!(data.len(), 3000);
    assert_eq!(ranges.len(), 1500, "every stderr byte is named once");
}

// ---- stdin -----------------------------------------------------------------------------------

async fn write(addr: SocketAddr, id: &str, at: u64, data: &[u8], eof: bool) -> (u16, Value) {
    job(
        addr,
        json!({"op":"write","scope":"sw","id":id,"at":at,"data":b64(data),"eof":eof}),
    )
    .await
}

#[tokio::test]
async fn stdin_writes_are_positional_ordered_idempotent_and_close_on_eof() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    let a = start(addr, "sw", "a1", &["cat"], json!({"stdin_open":true})).await;
    assert_eq!(a["stdin"], true);
    assert_eq!(write(addr, "a1", 0, b"hello ", false).await.1["written"], 6);
    assert_eq!(
        write(addr, "a1", 6, b"world\n", false).await.1["written"],
        12
    );
    assert_eq!(
        write(addr, "a1", 0, b"hello ", false).await.1["written"],
        12
    );
    assert_eq!(
        write(addr, "a1", 6, b"world\nmore", false).await.1["written"],
        16
    );
    // Out of order, over several connections: 16.. arrives last, the stream is still in order.
    let (a, b, c) = tokio::join!(
        write(addr, "a1", 18, b"C", false),
        write(addr, "a1", 17, b"B", false),
        async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            write(addr, "a1", 16, b"A", false).await
        }
    );
    assert_eq!((a.0, b.0, c.0), (200, 200, 200));
    let end = write(addr, "a1", 19, b"", true).await;
    assert_eq!(
        (end.0, end.1["written"].clone(), end.1["closed"].clone()),
        (200, json!(19), json!(true))
    );
    let again = write(addr, "a1", 19, b"", true).await;
    assert_eq!(again.1["closed"], true);
    wait_ended(addr, "sw", "a1").await;
    assert_eq!(read_all(addr, "sw", "a1").await.0, b"hello world\nmoreABC");
    // After the end: bytes are refused, a repeat of applied bytes is not.
    assert_eq!(write(addr, "a1", 19, b"late", false).await.0, 409);
    assert_eq!(write(addr, "a1", 0, b"hello ", false).await.0, 200);
    // Another scope's job is not found.
    let (s, _) = job(
        addr,
        json!({"op":"write","scope":"other","id":"a1","at":0,"data":b64(b"x")}),
    )
    .await;
    assert_eq!(s, 404);
}

#[tokio::test]
async fn a_job_without_stdin_open_reads_eof_and_has_no_stdin_to_write() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    let a = start(addr, "sw", "a1", &["cat"], json!({})).await;
    assert!(a.get("stdin").is_none());
    let v = wait_ended(addr, "sw", "a1").await;
    assert_eq!(v["code"], 0);
    start(addr, "sw", "a2", &["sleep", "30"], json!({})).await;
    let (s, v) = write(addr, "a2", 0, b"x", false).await;
    assert_eq!(s, 409, "{v}");
    assert!(v["error"].as_str().unwrap().contains("no stdin"));
    // A write larger than the cap is refused whole.
    start(addr, "sw", "a3", &["cat"], json!({"stdin_open":true})).await;
    let big = vec![b'x'; 128 * 1024 + 1];
    assert_eq!(write(addr, "a3", 0, &big, false).await.0, 413);
    // Kill ends the job and releases its stdin.
    job(
        addr,
        json!({"op":"kill","scope":"sw","ids":["a3","a2"],"remove":false,"grace_ms":100}),
    )
    .await;
    assert_eq!(write(addr, "a3", 0, b"x", false).await.0, 409);
}

#[tokio::test]
async fn a_stalled_stdin_does_not_freeze_the_table_and_bounds_its_writers() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    // A job that never reads: the pipe fills and writes wait.
    start(
        addr,
        "sw",
        "a1",
        &["sleep", "60"],
        json!({"stdin_open":true}),
    )
    .await;
    let chunk = vec![b'x'; 128 * 1024];
    let mut writers = Vec::new();
    for i in 0..20u64 {
        let chunk = chunk.clone();
        writers.push(tokio::spawn(async move {
            // timeout_ms bounds the wait: a short write answers 200 with `written` short.
            post(
                addr,
                &[],
                &json!({"command":"true","timeout_ms":1500,"job":{"op":"write","scope":"sw","id":"a1",
                        "at":i * 128 * 1024,"data":b64(&chunk),"eof":false}}),
            )
            .await
            .status
        }));
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let t = Instant::now();
    let (s, _) = job(addr, json!({"op":"list","scope":"sw"})).await;
    assert_eq!(s, 200);
    assert!(
        t.elapsed() < Duration::from_secs(1),
        "list waited on a stalled write"
    );
    let mut statuses = Vec::new();
    for w in writers {
        statuses.push(w.await.unwrap());
    }
    assert!(statuses.contains(&429), "{statuses:?}: waiters are bounded");
    job(
        addr,
        json!({"op":"kill","scope":"sw","ids":["a1"],"remove":true,"grace_ms":100}),
    )
    .await;
}

// ---- signal ----------------------------------------------------------------------------------

#[tokio::test]
async fn signal_reaches_the_group_once_inside_the_scope_and_never_an_ended_job() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    let script = "trap 'printf I' INT; trap 'printf U' USR1; printf ready; while :; do (trap '' INT USR1; sleep 0.1); done";
    start(addr, "sg", "a1", &["sh", "-c", script], json!({})).await;
    start(addr, "sg", "a2", &["sh", "-c", "sleep 30"], json!({})).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while read_all(addr, "sg", "a1").await.0 != b"ready" {
        assert!(Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    let sig = |scope: &'static str, ids: Value, s: &'static str| async move {
        job(
            addr,
            json!({"op":"signal","scope":scope,"ids":ids,"signal":s}),
        )
        .await
    };
    let (st, v) = sig("sg", json!(["a1", "nope"]), "SIGINT").await;
    assert_eq!((st, v), (200, json!({"sent":["a1"]})));
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(read_all(addr, "sg", "a1").await.0, b"readyI");
    // Another scope reaches nothing; a name off the list is a 400 that signals nothing.
    assert_eq!(
        sig("other", json!(["a1"]), "SIGUSR1").await.1,
        json!({"sent":[]})
    );
    assert_eq!(sig("sg", json!(["a1"]), "SIGSTOP").await.0, 400);
    assert_eq!(sig("sg", json!(["a1"]), "2").await.0, 400);
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(read_all(addr, "sg", "a1").await.0, b"readyI");
    // SIGINT ends a plain sleep job as signal 2, code 130.
    assert_eq!(
        sig("sg", json!(["a2"]), "SIGINT").await.1,
        json!({"sent":["a2"]})
    );
    let v = wait_ended(addr, "sg", "a2").await;
    assert_eq!(
        (v["code"].clone(), v["signal"].clone()),
        (json!(130), json!(2))
    );
    assert_eq!(
        sig("sg", json!(["a2"]), "SIGTERM").await.1,
        json!({"sent":[]})
    );
    sig("sg", json!(["a1"]), "SIGKILL").await;
    wait_ended(addr, "sg", "a1").await;
}

// ---- run-as ----------------------------------------------------------------------------------

/// Under `EXECD_RUN_AS` a request's env reaches the command as the target user and never the
/// root-privileged helper: a `LD_PRELOAD` it carries runs once, as the target uid. Needs root (or
/// `sudo -n`) and `cc`; a machine without either fails the test.
#[tokio::test]
async fn env_under_run_as_applies_after_the_switch_not_to_the_helper() {
    assert!(Execd::can_be_root(), "this test needs root or sudo -n");
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        dir.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o777),
    )
    .unwrap();
    let src = dir.path().join("pre.c");
    let so = dir.path().join("pre.so");
    let log = dir.path().join("uids");
    std::fs::write(
        &src,
        format!(
            "#include <stdio.h>\n#include <unistd.h>\n__attribute__((constructor)) static void f(void){{FILE*x=fopen(\"{}\",\"a\");if(x){{fprintf(x,\"%d\\n\",(int)getuid());fclose(x);}}}}\n",
            log.display()
        ),
    )
    .unwrap();
    let cc = std::process::Command::new("cc")
        .args(["-shared", "-fPIC", "-o"])
        .arg(&so)
        .arg(&src)
        .status()
        .expect("cc is needed");
    assert!(cc.success());
    std::fs::set_permissions(&so, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    let execd = Execd::start_as_root(&[("EXECD_RUN_AS", "nobody")], &[]).unwrap();
    let addr = std::net::SocketAddr::V4(execd.addr);
    let a = post(
        addr,
        &[],
        &json!({"command":"sh","args":["-c","id -u; printf %s \"$VFY:${EXECD_CMD_ENV-gone}\""],
                "env":{"VFY":"v","LD_PRELOAD":so.to_str().unwrap()},"timeout_ms":10000}),
    )
    .await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    let out = String::from_utf8(unb64(&a.json()["stdout"])).unwrap();
    let mut lines = out.lines();
    let uid = lines.next().unwrap();
    assert_ne!(uid, "0");
    assert_eq!(lines.next().unwrap(), "v:gone");
    let uids = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        uids.lines().all(|l| l == uid) && !uids.is_empty(),
        "the preload ran as {uids:?}, the target is {uid}: it must never run as root"
    );
}
