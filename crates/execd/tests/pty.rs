//! Terminals (T1 to T12, K1, K2), driven through the real binary: a job started with `pty` is a
//! terminal kept by a keeper process, so it has a size, takes input positionally, outlives a restart
//! of execd, and is hung up (not lost) when its keeper dies.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};
use support::Execd;
use support::http::post;

async fn raw(addr: SocketAddr, op: Value) -> (u16, Value) {
    let a = post(addr, &[], &json!({ "command": "true", "job": op })).await;
    let status = a.status;
    let body = serde_json::from_slice(&a.body).unwrap_or(Value::Null);
    (status, body)
}

async fn job(addr: SocketAddr, op: Value) -> Value {
    let (status, body) = raw(addr, op).await;
    assert_eq!(status, 200, "{body}");
    body
}

static LOGS: std::sync::Mutex<Vec<(SocketAddr, std::path::PathBuf)>> =
    std::sync::Mutex::new(Vec::new());

async fn term(addr: SocketAddr, id: &str, script: &str, cols: u64, rows: u64) -> Value {
    let body = json!({
        "command": "sh", "args": ["-c", script],
        "job": { "op": "start", "scope": "s", "id": id, "hold": false, "pty": { "cols": cols, "rows": rows } },
    });
    let a = post(addr, &[], &body).await;
    if a.status != 200 {
        let log = LOGS
            .lock()
            .unwrap()
            .iter()
            .find(|(x, _)| *x == addr)
            .and_then(|(_, p)| std::fs::read_to_string(p).ok());
        panic!("{}: execd said {log:?}", String::from_utf8_lossy(&a.body));
    }
    a.json()
}

async fn read(addr: SocketAddr, id: &str) -> (String, Value) {
    let v = job(
        addr,
        json!({"op":"read","scope":"s","id":id,"offset":0,"max":1 << 20}),
    )
    .await;
    let data = base64::engine::general_purpose::STANDARD
        .decode(v["data"].as_str().unwrap_or(""))
        .unwrap();
    (String::from_utf8_lossy(&data).to_string(), v)
}

async fn read_until(addr: SocketAddr, id: &str, want: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (text, _) = read(addr, id).await;
        if text.contains(want) {
            return text;
        }
        assert!(
            Instant::now() < deadline,
            "{id} never wrote {want:?}: {text:?}"
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

async fn ended(addr: SocketAddr, id: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (_, v) = read(addr, id).await;
        if v["state"] == "ended" {
            return v;
        }
        assert!(Instant::now() < deadline, "{id} never ended: {v}");
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

async fn write(addr: SocketAddr, id: &str, at: u64, data: &[u8], eof: bool) -> (u16, Value) {
    raw(
        addr,
        json!({"op":"write","scope":"s","id":id,"at":at,
               "data": base64::engine::general_purpose::STANDARD.encode(data), "eof": eof}),
    )
    .await
}

fn start_execd(extra: &[(&str, &str)]) -> (Execd, tempfile::TempDir) {
    let jobs = tempfile::tempdir().unwrap();
    let mut settings = vec![("EXECD_JOBS_DIR", jobs.path().to_str().unwrap().to_string())];
    settings.extend(extra.iter().map(|(k, v)| (*k, (*v).to_string())));
    let s: Vec<(&str, &str)> = settings.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let e = Execd::start(&s, &[]);
    LOGS.lock()
        .unwrap()
        .push((SocketAddr::V4(e.addr), e.dir.path().join("execd.log")));
    (e, jobs)
}

/// The keeper processes of terminals whose spool is under `jobs`.
fn keepers(jobs: &std::path::Path) -> Vec<i32> {
    let needle = jobs.to_string_lossy().to_string();
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(raw) = std::fs::read(e.path().join("cmdline")) else {
            continue;
        };
        let text = String::from_utf8_lossy(&raw).replace('\0', " ");
        if text.contains("--execd-run-as-helper keep ")
            && text.contains(&needle)
            && let Ok(pid) = e.file_name().to_string_lossy().parse::<i32>()
        {
            out.push(pid);
        }
    }
    out
}

/// T2: a terminal start echoes its size, its command has a terminal on 0, 1 and 2 at that size, and
/// everything it prints is stdout (T3): no stderr range.
#[tokio::test]
async fn a_start_with_pty_is_a_terminal_of_that_size_and_its_output_is_one_stream() {
    let (e, _jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    let started = term(
        a,
        "t",
        "test -t 0 && test -t 1 && test -t 2 && stty size; echo err >&2",
        80,
        24,
    )
    .await;
    assert_eq!(started["pty"], json!({"cols": 80, "rows": 24}), "{started}");
    assert_eq!(started["stdin"], json!(true));
    let v = ended(a, "t").await;
    assert_eq!(v["code"], json!(0), "{v}");
    let (text, v) = read(a, "t").await;
    assert_eq!(text, "24 80\r\nerr\r\n", "{text:?}");
    assert!(v.get("stderr_ranges").is_none(), "{v}");
    assert_eq!(v["pty"], json!({"cols": 80, "rows": 24, "open": false}));
}

/// The command is a session leader and its terminal is its controlling one (`/dev/tty` opens).
#[tokio::test]
async fn the_terminal_is_the_commands_controlling_terminal() {
    let (e, _jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    term(
        a,
        "t",
        "exec 3</dev/tty && echo ctty; ps -o sid=,pid= -p $$",
        80,
        24,
    )
    .await;
    let v = ended(a, "t").await;
    assert_eq!(v["code"], json!(0), "{v}");
    let (text, _) = read(a, "t").await;
    assert!(text.starts_with("ctty\r\n"), "{text:?}");
}

/// T5 (input lands once), T10 (a terminal has no end of input) and the echo of a line discipline.
#[tokio::test]
async fn input_lands_once_in_order_and_eof_is_refused() {
    let (e, _jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    term(a, "c", "exec cat", 80, 24).await;
    let (st, b) = write(a, "c", 0, b"ab", false).await;
    assert_eq!((st, &b["written"]), (200, &json!(2)), "{b}");
    // A replay of the same bytes lands nothing.
    let (st, b) = write(a, "c", 0, b"ab", false).await;
    assert_eq!((st, &b["written"]), (200, &json!(2)), "{b}");
    // A write that starts below `written` lands its tail.
    let (st, b) = write(a, "c", 1, b"bc\n", false).await;
    assert_eq!((st, &b["written"]), (200, &json!(4)), "{b}");
    let text = read_until(a, "c", "abc\r\nabc\r\n").await;
    assert_eq!(text.matches("abc").count(), 2, "{text:?}");
    // eof is refused and writes nothing (T10) ...
    let (st, b) = write(a, "c", 4, b"", true).await;
    assert_eq!(st, 400, "{b}");
    // ... and the same write without eof is taken (the partner).
    let (st, b) = write(a, "c", 4, b"", false).await;
    assert_eq!(st, 200, "{b}");
    // Ctrl-D at the start of a line is data that ends cat's input.
    let (st, _) = write(a, "c", 4, b"\x04", false).await;
    assert_eq!(st, 200);
    let v = ended(a, "c").await;
    assert_eq!(v["code"], json!(0), "{v}");
    // After the end a write is refused with the exact count.
    let (st, b) = write(a, "c", 5, b"x", false).await;
    assert_eq!(st, 409, "{b}");
    assert_eq!(b["written"], json!(5), "{b}");
}

/// T1: a resize reaches the program (it gets SIGWINCH and reads the new size), its bounds are
/// 1 to 1024, and a refusal clears: the same resize inside the bounds is taken (T4).
#[tokio::test]
async fn a_resize_reaches_the_program_and_its_bounds_are_enforced() {
    let (e, _jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    term(
        a,
        "w",
        "trap 'stty size' WINCH; echo ready; while :; do sleep 0.05; done",
        80,
        24,
    )
    .await;
    read_until(a, "w", "ready").await;
    for (c, r) in [(0, 24), (1025, 24), (80, 0)] {
        let (st, b) = raw(
            a,
            json!({"op":"resize","scope":"s","id":"w","cols":c,"rows":r}),
        )
        .await;
        assert_eq!(st, 400, "{c}x{r}: {b}");
    }
    let (st, b) = raw(
        a,
        json!({"op":"resize","scope":"s","id":"w","cols":120,"rows":40}),
    )
    .await;
    assert_eq!(
        (st, &b["cols"], &b["rows"]),
        (200, &json!(120), &json!(40)),
        "{b}"
    );
    read_until(a, "w", "40 120").await;
    // The limits themselves are taken.
    let (st, _) = raw(
        a,
        json!({"op":"resize","scope":"s","id":"w","cols":1024,"rows":1024}),
    )
    .await;
    assert_eq!(st, 200);
    let (_, v) = read(a, "w").await;
    assert_eq!(
        v["pty"],
        json!({"cols": 1024, "rows": 1024, "open": true}),
        "{v}"
    );
    // Refusals: unknown id, a job with no terminal, an ended terminal.
    let (st, _) = raw(
        a,
        json!({"op":"resize","scope":"s","id":"nope","cols":80,"rows":24}),
    )
    .await;
    assert_eq!(st, 404);
    let body = json!({"command":"sleep","args":["300"],
        "job":{"op":"start","scope":"s","id":"plain","hold":false}});
    assert_eq!(post(a, &[], &body).await.status, 200);
    let (st, b) = raw(
        a,
        json!({"op":"resize","scope":"s","id":"plain","cols":80,"rows":24}),
    )
    .await;
    assert_eq!(
        (st, b["error"].as_str()),
        (409, Some("the job has no terminal")),
        "{b}"
    );
    job(
        a,
        json!({"op":"kill","scope":"s","ids":["w","plain"],"grace_ms":200}),
    )
    .await;
    ended(a, "w").await;
    let (st, b) = raw(
        a,
        json!({"op":"resize","scope":"s","id":"w","cols":80,"rows":24}),
    )
    .await;
    assert_eq!(
        (st, b["error"].as_str()),
        (409, Some("the job has ended")),
        "{b}"
    );
    // A job that is not a terminal has no `pty` entry.
    let l = job(a, json!({"op":"list","scope":"s"})).await;
    let plain = l["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == "plain")
        .unwrap();
    assert!(plain.get("pty").is_none(), "{plain}");
    // The start itself checks the bounds, and runs nothing when it refuses.
    let body = json!({"command":"true","job":{"op":"start","scope":"s","id":"bad","pty":{"cols":0,"rows":24}}});
    assert_eq!(post(a, &[], &body).await.status, 400);
    let l = job(a, json!({"op":"list","scope":"s"})).await;
    assert!(
        l["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|j| j["id"] != "bad")
    );
}

/// T6, K1: a terminal survives a restart of execd (its keeper holds the master): the same shell
/// answers, what was written before is counted (a replay lands nothing), its size is the same, and
/// the output continues at the same offsets.
#[tokio::test]
async fn a_terminal_survives_a_restart_of_execd_with_its_offsets_and_its_shell() {
    let state = tempfile::tempdir().unwrap();
    let ledger = state.path().join("ledger");
    let (mut first, jobs) = start_execd(&[("EXECD_STATE_DIR", ledger.to_str().unwrap())]);
    let a1 = SocketAddr::V4(first.addr);
    term(a1, "sh", "exec sh", 100, 30).await;
    let (st, b) = write(a1, "sh", 0, b"echo one\n", false).await;
    assert_eq!((st, &b["written"]), (200, &json!(9)), "{b}");
    let before = read_until(a1, "sh", "one\r\n").await;
    let (_, v) = read(a1, "sh").await;
    let end_before = v["end"].as_u64().unwrap();
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(i32::try_from(first.child.id()).unwrap()),
        nix::sys::signal::Signal::SIGHUP,
    )
    .unwrap();
    first.child.wait().unwrap();
    assert_eq!(keepers(jobs.path()).len(), 1, "the keeper outlives execd");

    // Input written while execd was down waits in the tty; the new execd finds the terminal.
    let second = Execd::start(
        &[
            ("EXECD_STATE_DIR", ledger.to_str().unwrap()),
            ("EXECD_JOBS_DIR", jobs.path().to_str().unwrap()),
        ],
        &[],
    );
    let a2 = SocketAddr::V4(second.addr);
    let (_, v) = read(a2, "sh").await;
    assert_eq!(v["state"], json!("running"), "{v}");
    assert_eq!(
        v["pty"],
        json!({"cols": 100, "rows": 30, "open": true}),
        "{v}"
    );
    assert!(v["end"].as_u64().unwrap() >= end_before, "{v}");
    let (text, _) = read(a2, "sh").await;
    assert!(
        text.starts_with(&before),
        "the offsets moved: {text:?} vs {before:?}"
    );
    // A write retried across the restart lands once: the keeper counted it.
    let (st, b) = write(a2, "sh", 0, b"echo one\n", false).await;
    assert_eq!((st, &b["written"]), (200, &json!(9)), "{b}");
    let (st, b) = write(a2, "sh", 9, b"stty size; exit 3\n", false).await;
    assert_eq!((st, &b["written"]), (200, &json!(27)), "{b}");
    let v = ended(a2, "sh").await;
    assert_eq!(
        v["code"],
        json!(3),
        "the shell's real status, not the lost one: {v}"
    );
    let (text, _) = read(a2, "sh").await;
    assert!(text.contains("30 100\r\n"), "{text:?}");
    assert_eq!(text.matches("one\r\n").count(), 2, "{text:?}");
    // The keeper ends with the job (K2): no process is left.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !keepers(jobs.path()).is_empty() {
        assert!(Instant::now() < deadline, "a keeper outlived its job");
        std::thread::sleep(Duration::from_millis(30));
    }
}

/// T7, T12: a terminal whose keeper died is hung up, not lost: input and resize are refused with
/// 409, `pty.open` is false, and its session (which ignores the hangup) is still a job that can end.
#[tokio::test]
async fn a_terminal_whose_keeper_died_is_hung_up() {
    let (e, jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    term(a, "h", "trap '' HUP; echo up; exec sleep 300", 80, 24).await;
    read_until(a, "h", "up").await;
    let ks = keepers(jobs.path());
    assert_eq!(ks.len(), 1, "{ks:?}");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(ks[0]),
        nix::sys::signal::Signal::SIGKILL,
    )
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let (_, v) = read(a, "h").await;
        if v["pty"]["open"] == json!(false) {
            assert_eq!(v["state"], json!("running"), "{v}");
            break;
        }
        assert!(Instant::now() < deadline, "never hung up: {v}");
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    let (st, b) = write(a, "h", 0, b"x", false).await;
    assert_eq!(
        (st, b["error"].as_str()),
        (409, Some("the terminal hung up")),
        "{b}"
    );
    assert_eq!(b["written"], json!(0), "{b}");
    let (st, b) = raw(
        a,
        json!({"op":"resize","scope":"s","id":"h","cols":90,"rows":30}),
    )
    .await;
    assert_eq!(
        (st, b["error"].as_str()),
        (409, Some("the terminal hung up")),
        "{b}"
    );
    // eof is still the client's error, not a hangup.
    let (st, _) = write(a, "h", 0, b"", true).await;
    assert_eq!(st, 400);
    job(
        a,
        json!({"op":"kill","scope":"s","ids":["h"],"grace_ms":200}),
    )
    .await;
    ended(a, "h").await;
    // The partner: a new terminal opens.
    term(a, "h2", "echo again", 80, 24).await;
    read_until(a, "h2", "again").await;
}

fn rss_kib(pid: i32, field: &str) -> u64 {
    let text = std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")).unwrap();
    text.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|n| n.parse().ok())
        .unwrap()
}

/// What a keeper costs: 30 idle terminals, the resident and the proportional size of each keeper (a
/// keeper is execd's own binary, so most of its resident pages are shared with execd's text).
#[tokio::test]
async fn a_keeper_is_small_and_idle_terminals_cost_no_wakeups() {
    let (e, jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    for n in 0..30 {
        term(a, &format!("k{n}"), "exec sleep 300", 80, 24).await;
    }
    let ks = keepers(jobs.path());
    assert_eq!(ks.len(), 30);
    let mut rss: Vec<u64> = ks.iter().map(|p| rss_kib(*p, "Rss:")).collect();
    let mut pss: Vec<u64> = ks.iter().map(|p| rss_kib(*p, "Pss:")).collect();
    rss.sort_unstable();
    pss.sort_unstable();
    let at = |v: &[u64], q: usize| v[(v.len() * q / 100).min(v.len() - 1)];
    println!(
        "keeper n=30 Rss KiB p50 {} p95 {} max {}; Pss KiB p50 {} p95 {} max {}",
        at(&rss, 50),
        at(&rss, 95),
        rss[rss.len() - 1],
        at(&pss, 50),
        at(&pss, 95),
        pss[pss.len() - 1]
    );
    // Idle means asleep: no keeper runs on its own (voluntary context switches do not move).
    let switches = |p: i32| -> u64 {
        std::fs::read_to_string(format!("/proc/{p}/status"))
            .unwrap()
            .lines()
            .filter(|l| l.contains("ctxt_switches"))
            .filter_map(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .sum()
    };
    let before: Vec<u64> = ks.iter().map(|p| switches(*p)).collect();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let moved: u64 = ks.iter().zip(&before).map(|(p, b)| switches(*p) - b).sum();
    println!("context switches of 30 idle keepers in 1.5 s: {moved}");
    assert_eq!(moved, 0, "an idle keeper woke up");
    job(a, json!({"op":"wipe","scope":"s","grace_ms":200})).await;
}

/// T9: a held terminal nobody kept dies with its client's connection, every process of its session
/// included: the shell puts its background command in a group of its own (`set -m`), so a kill of the
/// first group alone would leave it. Red under `mut_pty_kill_group_only`.
#[tokio::test]
async fn a_held_terminal_nobody_kept_dies_whole_with_its_client() {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    let (e, _jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    let body = json!({
        "command": "sh", "args": ["-c", "set -m; sleep 300 & echo up; wait"],
        "job": { "op": "start", "scope": "s", "id": "h", "hold": true, "pty": { "cols": 80, "rows": 24 } },
    })
    .to_string();
    let mut sock = tokio::net::TcpStream::connect(a).await.unwrap();
    sock.write_all(
        format!(
            "POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    // The stream is alive: its first frame says it started a terminal.
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    while !String::from_utf8_lossy(&seen).contains("\"pty\"") {
        let n = tokio::time::timeout(Duration::from_secs(10), sock.read(&mut buf))
            .await
            .unwrap()
            .unwrap();
        assert!(
            n > 0,
            "the stream ended early: {:?}",
            String::from_utf8_lossy(&seen)
        );
        seen.extend_from_slice(&buf[..n]);
    }
    read_until(a, "h", "up").await;
    drop(sock);
    let v = ended(a, "h").await;
    assert_eq!(v["state"], json!("ended"), "{v}");
    // A job ends only when its group and every process that carries its mark are gone, so the
    // background `sleep` (in a group of its own) is dead too.
}

fn fds(pid: u32) -> usize {
    std::fs::read_dir(format!("/proc/{pid}/fd"))
        .unwrap()
        .count()
}

/// T8: a terminal's descriptors and processes do not outlive its job: after a mixed workload of
/// terminals (ended, killed, removed) execd holds the same number of fds and threads as before, and
/// no keeper is left.
#[tokio::test]
async fn terminals_leave_nothing_behind() {
    let (e, jobs) = start_execd(&[]);
    let a = SocketAddr::V4(e.addr);
    let threads = |pid: u32| {
        std::fs::read_dir(format!("/proc/{pid}/task"))
            .unwrap()
            .count()
    };
    // Warm up: the first job opens the long-lived things (the reaper's, the tail hub's).
    term(a, "warm", "true", 80, 24).await;
    ended(a, "warm").await;
    job(a, json!({"op":"remove","scope":"s","ids":["warm"]})).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let (f0, t0) = (fds(e.child.id()), threads(e.child.id()));
    for n in 0..10 {
        let id = format!("t{n}");
        term(
            a,
            &id,
            if n % 2 == 0 {
                "echo hi"
            } else {
                "exec sleep 300"
            },
            80,
            24,
        )
        .await;
        if n % 2 == 1 {
            job(
                a,
                json!({"op":"kill","scope":"s","ids":[id],"grace_ms":100}),
            )
            .await;
        }
        ended(a, &id).await;
        job(a, json!({"op":"remove","scope":"s","ids":[id]})).await;
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while (fds(e.child.id()) != f0 || !keepers(jobs.path()).is_empty()) && Instant::now() < deadline
    {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(fds(e.child.id()), f0, "execd leaked descriptors");
    assert!(keepers(jobs.path()).is_empty(), "a keeper outlived its job");
    assert!(threads(e.child.id()) <= t0 + 1, "execd leaked threads");
}

/// T-503: with no pseudo-terminal free a start is `503`, nothing runs, and the refusal clears: once
/// a terminal ends the same start is taken (the partner, retried once). The cap is the test-only
/// `EXECD_TERMINALS_MAX`; the kernel's table is the real limit.
#[tokio::test]
async fn a_start_with_no_pseudo_terminal_free_is_503_and_a_retry_after_one_ends_opens() {
    let (e, jobs) = start_execd(&[("EXECD_TERMINALS_MAX", "2")]);
    let a = SocketAddr::V4(e.addr);
    term(a, "a", "exec sleep 300", 80, 24).await;
    term(a, "b", "exec sleep 300", 80, 24).await;
    let body = json!({
        "command": "true",
        "job": { "op": "start", "scope": "s", "id": "c", "hold": false, "pty": { "cols": 80, "rows": 24 } },
    });
    for _ in 0..2 {
        let r = post(a, &[], &body).await;
        assert_eq!(r.status, 503, "{}", String::from_utf8_lossy(&r.body));
        assert!(String::from_utf8_lossy(&r.body).contains("no pseudo-terminal is free"));
    }
    // Nothing was left of the refused start: no entry, no keeper.
    let l = job(a, json!({"op":"list","scope":"s"})).await;
    assert!(l["jobs"].as_array().unwrap().iter().all(|j| j["id"] != "c"));
    assert_eq!(keepers(jobs.path()).len(), 2);
    // A job with no terminal is not counted.
    let plain = json!({"command":"true","job":{"op":"start","scope":"s","id":"p","hold":false}});
    assert_eq!(post(a, &[], &plain).await.status, 200);
    // One ends: the same start now opens.
    job(
        a,
        json!({"op":"kill","scope":"s","ids":["a"],"grace_ms":100}),
    )
    .await;
    ended(a, "a").await;
    assert_eq!(post(a, &[], &body).await.status, 200);
}

/// `EXECD_RUN_AS`: the shell of a terminal runs as the target user, which owns its tty and has it as
/// its controlling terminal; the keeper (root) is not the shell's uid. As E2B runs execd.
#[tokio::test]
async fn under_run_as_the_shell_is_the_target_and_owns_its_terminal() {
    let jobs = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        jobs.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o755),
    )
    .unwrap();
    let e = Execd::start_as_root(
        &[
            ("EXECD_RUN_AS", "nobody"),
            ("EXECD_HOME", "/tmp"),
            ("EXECD_JOBS_DIR", jobs.path().to_str().unwrap()),
        ],
        &[],
    )
    .expect("this test needs root (or passwordless sudo)");
    let a = SocketAddr::V4(e.addr);
    term(
        a,
        "u",
        "id -u; stat -c %U $(tty); exec 3</dev/tty && echo ctty; stty size",
        90,
        33,
    )
    .await;
    let v = ended(a, "u").await;
    assert_eq!(v["code"], json!(0), "{v}");
    let (text, _) = read(a, "u").await;
    assert!(text.starts_with("65534\r\nnobody\r\n"), "{text:?}");
    assert!(text.contains("ctty\r\n33 90\r\n"), "{text:?}");
    // The keeper runs as root, out of the user's reach: the user cannot signal it.
    term(a, "k", "exec sleep 300", 80, 24).await;
    let ks = keepers(jobs.path());
    assert_eq!(ks.len(), 1);
    let uid = std::fs::read_to_string(format!("/proc/{}/status", ks[0]))
        .unwrap()
        .lines()
        .find(|l| l.starts_with("Uid:"))
        .map(str::to_string)
        .unwrap();
    assert!(uid.split_whitespace().nth(1) == Some("0"), "{uid}");
}
