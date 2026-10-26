//! What the bearer and the sandbox are protected from, against the real binary: the token never
//! reaches a command, execd refuses to start open on a network, and `--run-as` puts execd's memory,
//! environment and files out of a command's reach.
//!
//! The root-only tests run execd through `sudo -n` (or as root) and say `SKIPPED` on a host that
//! gives them neither.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::os::unix::fs::MetadataExt as _;
use std::process::{Command, Stdio};

use beyond_ai_test_support::ports;
use serde_json::{Value, json};
use support::{Execd, http::post};

/// A bearer no other test shares: the tests walk every readable `/proc/*/environ`, and run side by
/// side with each other.
fn token(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("bearer-{tag}-{}-{nanos}", std::process::id())
}

/// Run `script` under `sh -c` through execd and give back its answer.
async fn sh(execd: &Execd, bearer: Option<&String>, script: &str) -> Value {
    let addr = std::net::SocketAddr::V4(execd.addr);
    let auth = bearer.map(|b| format!("Bearer {b}"));
    let headers: Vec<(&str, &str)> = auth.iter().map(|a| ("Authorization", a.as_str())).collect();
    let answer = post(
        addr,
        &headers,
        &json!({ "command": "sh", "args": ["-c", script], "timeout_ms": 20_000 }),
    )
    .await;
    assert_eq!(
        answer.status,
        200,
        "{}",
        String::from_utf8_lossy(&answer.body)
    );
    answer.json()
}

fn out(v: &Value) -> String {
    format!(
        "{}{}",
        v["stdout"].as_str().unwrap_or(""),
        v["stderr"].as_str().unwrap_or("")
    )
}

/// Everything a command can learn about the environment of execd and of every process it can see:
/// its own `env`, its parent's `/proc/$PPID/environ`, and every readable `/proc/*/environ`.
const LOOK: &str = "env; echo ---ppid; cat /proc/$PPID/environ | tr '\\0' '\\n'; \
                    echo ---walk; for f in /proc/[0-9]*/environ; do cat $f 2>/dev/null | tr '\\0' '\\n'; done; true";

/// A process the test starts with the bearer in its environment: proof that the walk can see an
/// environ of another process and would find the bearer if it were there.
struct Decoy(std::process::Child);

impl Decoy {
    fn start(bearer: &str) -> Self {
        Self(
            Command::new("sleep")
                .arg("60")
                .env("DECOY_BEARER", bearer)
                .stdin(Stdio::null())
                .spawn()
                .unwrap(),
        )
    }
}

impl Drop for Decoy {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn a_command_never_sees_the_bearer_delivered_in_the_environment() {
    let bearer = token("env");
    let decoy = Decoy::start(&bearer);
    let execd = Execd::start(&[], &[("EXECD_BEARER", &bearer)]);
    let seen = out(&sh(&execd, Some(&bearer), LOOK).await);
    // The walk reaches other processes' environments (the decoy's holds the bearer) and execd's own.
    assert!(
        seen.contains(&format!("DECOY_BEARER={bearer}")),
        "the walk is blind"
    );
    assert!(
        seen.contains("EXECD_BEARER_FD"),
        "the walk never read execd's environ"
    );
    // ...and the decoy is the only place it finds the bearer: not in the command's env, not in its
    // parent (execd), not in any other process.
    assert_eq!(
        seen.matches(bearer.as_str()).count(),
        1,
        "the bearer leaked:\n{}",
        seen.replace(bearer.as_str(), "<BEARER>")
    );
    assert!(!seen.contains("EXECD_BEARER="));
    drop(decoy);
}

#[tokio::test]
async fn a_command_never_sees_the_bearer_delivered_in_a_file() {
    let bearer = token("file");
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bearer");
    std::fs::write(&file, format!("{bearer}\n")).unwrap();
    std::fs::set_permissions(&file, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
    let execd = Execd::start(&[("EXECD_BEARER_FILE", file.to_str().unwrap())], &[]);
    let seen = out(&sh(&execd, Some(&bearer), LOOK).await);
    assert!(!seen.contains(bearer.as_str()), "the bearer leaked");
    // No `EXECD_*` variable reaches a command's environment. execd's own environ (its parent's, to
    // the command) still lists the settings it was started with, which are not secrets, but never
    // the bearer. (The walk is every process of every test running beside this one, so it is not
    // read for that.)
    let own = seen.split("---walk").next().unwrap();
    let (env, ppid) = own.split_once("---ppid").unwrap();
    assert!(
        !env.lines().any(|l| l.starts_with("EXECD_")),
        "an EXECD_ variable reached a command"
    );
    assert!(
        !ppid.lines().any(|l| l.starts_with("EXECD_BEARER=")),
        "the bearer is in execd's environ"
    );
}

/// The same for every way a command is started: a stream and a job, not only v1.1.
#[tokio::test]
async fn a_streamed_command_and_a_job_do_not_see_the_bearer_either() {
    let bearer = token("a_streamed_c");
    let execd = Execd::start(&[], &[("EXECD_BEARER", &bearer)]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    let auth = format!("Bearer {bearer}");
    let h = [("Authorization", auth.as_str())];
    let stream = post(
        addr,
        &h,
        &json!({ "command": "sh", "args": ["-c", "env; cat /proc/$PPID/environ | tr '\\0' '\\n'"],
                 "stream": true }),
    )
    .await;
    let text: String = stream
        .frames()
        .iter()
        .filter_map(|f| f["stdout"].as_str())
        .map(|b| {
            use base64::Engine as _;
            String::from_utf8_lossy(&base64::engine::general_purpose::STANDARD.decode(b).unwrap())
                .into_owned()
        })
        .collect();
    assert!(text.contains("PATH="), "{text}");
    assert!(
        !text.contains(bearer.as_str()) && !text.contains("EXECD_BEARER="),
        "{text}"
    );

    let started = post(
        addr,
        &h,
        &json!({ "command": "sh", "args": ["-c", "env; cat /proc/$PPID/environ | tr '\\0' '\\n'"],
                 "job": { "op": "start", "scope": "s", "id": "a1", "hold": false } }),
    )
    .await;
    assert_eq!(started.status, 200);
    let mut data = String::new();
    for _ in 0..100 {
        let r = post(
            addr,
            &h,
            &json!({ "command": "true",
                     "job": { "op": "read", "scope": "s", "id": "a1", "offset": 0, "max": 1_000_000 } }),
        )
        .await
        .json();
        if r["state"] == "ended" {
            use base64::Engine as _;
            data = String::from_utf8_lossy(
                &base64::engine::general_purpose::STANDARD
                    .decode(r["data"].as_str().unwrap())
                    .unwrap(),
            )
            .into_owned();
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(data.contains("PATH="), "the job never ended: {data}");
    assert!(
        !data.contains(bearer.as_str()) && !data.contains("EXECD_BEARER="),
        "{data}"
    );
}

// --- fail closed ------------------------------------------------------------------------------

#[test]
fn it_refuses_to_start_open_on_a_network() {
    let (status, log) = Execd::run_to_exit(&[("EXECD_LISTEN", ports::CHILD_LISTEN_ALL)], &[], false);
    assert!(!status.success(), "it started: {log}");
    assert!(log.contains("refusing to start"), "{log}");
    assert!(log.contains("EXECD_INSECURE_NO_AUTH"), "{log}");
    // An empty bearer is no bearer.
    let (status, log) = Execd::run_to_exit(
        &[("EXECD_LISTEN", ports::CHILD_LISTEN_ALL)],
        &[("EXECD_BEARER", "")],
        false,
    );
    assert!(
        !status.success() && log.contains("refusing to start"),
        "{log}"
    );
    // A named bearer file that holds nothing is an error of its own, even on loopback.
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty");
    std::fs::write(&empty, "\n").unwrap();
    let (status, log) = Execd::run_to_exit(
        &[("EXECD_LISTEN", ports::CHILD_LISTEN), ("EXECD_BEARER_FILE", empty.to_str().unwrap())],
        &[],
        false,
    );
    assert!(!status.success() && log.contains("is empty"), "{log}");
}

#[tokio::test]
async fn it_starts_open_on_loopback_and_on_a_network_only_when_told_or_secured() {
    let bearer = token("it_starts_op");
    // Loopback, no bearer: allowed, and loud.
    let open = Execd::start_on(ports::CHILD_LISTEN, &[], &[]);
    assert!(open.log().contains("NO BEARER CONFIGURED"));
    // All interfaces with a bearer: allowed, and it answers 401 without it.
    let secured = Execd::start_on(ports::CHILD_LISTEN_ALL, &[], &[("EXECD_BEARER", &bearer)]);
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], secured.addr.port()));
    let req = json!({ "command": "true" });
    assert_eq!(post(addr, &[], &req).await.status, 401);
    // All interfaces with nothing, on purpose: allowed, and louder.
    let deliberate = Execd::start_on(ports::CHILD_LISTEN_ALL, &[("EXECD_INSECURE_NO_AUTH", "1")], &[]);
    assert!(deliberate.log().contains("NO BEARER CONFIGURED"));
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], deliberate.addr.port()));
    assert_eq!(post(addr, &[], &req).await.status, 200);
}

// --- --run-as, without root -------------------------------------------------------------------

fn my_uid() -> u32 {
    std::fs::metadata("/proc/self").unwrap().uid()
}

fn my_gid() -> u32 {
    std::fs::metadata("/proc/self").unwrap().gid()
}

#[test]
fn run_as_refuses_the_uid_execd_already_has() {
    let me = format!("{}:{}", my_uid(), my_gid());
    let (status, log) = Execd::run_to_exit(&[("EXECD_RUN_AS", &me)], &[], false);
    assert!(!status.success(), "{log}");
    assert!(log.contains("execd's own uid"), "{log}");
}

#[test]
fn run_as_refuses_when_execd_cannot_switch() {
    if my_uid() == 0 {
        return;
    }
    // `nobody` exists on every host this runs on, and an unprivileged execd cannot become it.
    let (status, log) = Execd::run_to_exit(&[("EXECD_RUN_AS", "nobody")], &[], false);
    assert!(!status.success(), "it started: {log}");
    assert!(log.contains("cannot switch"), "{log}");
    let (status, log) = Execd::run_to_exit(&[("EXECD_RUN_AS", "no-such-user-xyz")], &[], false);
    assert!(!status.success() && log.contains("no such user"), "{log}");
    let (status, log) = Execd::run_to_exit(&[("EXECD_RUN_AS", "4242")], &[], false);
    assert!(!status.success() && log.contains("name its gid"), "{log}");
}

// --- --run-as, as root ------------------------------------------------------------------------

fn sudo(args: &[&str]) {
    let ok = Command::new("sudo")
        .arg("-n")
        .args(args)
        .status()
        .unwrap()
        .success();
    assert!(ok, "sudo {args:?} failed");
}

#[tokio::test]
async fn run_as_refuses_root_when_execd_is_root() {
    if !Execd::can_be_root() {
        eprintln!("SKIPPED: needs root");
        return;
    }
    let (status, log) = Execd::run_to_exit(
        &[("EXECD_LISTEN", ports::CHILD_LISTEN), ("EXECD_RUN_AS", "root")],
        &[],
        true,
    );
    assert!(
        !status.success() && log.contains("execd's own uid"),
        "{log}"
    );
}

/// Under `--run-as`, a command is another user and cannot read execd's memory, its environment or a
/// 0600 bearer file execd owns; a request still works end to end, from the right directory and home.
#[tokio::test]
async fn under_run_as_a_command_cannot_reach_execd() {
    let bearer = token("under_run_as");
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bearer");
    std::fs::write(&file, format!("{bearer}\n")).unwrap();
    std::fs::set_permissions(&file, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
    // A file only execd's user can read: root owns it.
    sudo(&["chown", "root:root", file.to_str().unwrap()]);
    let decoy = Decoy::start(&bearer);
    let Some(execd) = Execd::start_as_root(
        &[("EXECD_RUN_AS", "nobody"), ("EXECD_BEARER_FILE", file.to_str().unwrap())],
        &[],
    ) else {
        return;
    };
    let pid = execd.pid();
    let nobody = String::from_utf8(
        Command::new("id")
            .args(["-u", "nobody"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_string();

    let script = format!(
        "id -u; id -g; cat /proc/self/status | grep ^Groups; pwd; echo HOME=$HOME; \
         echo ---file; cat {file} 2>&1; \
         echo ---environ; cat /proc/{pid}/environ 2>&1 | head -c 200; \
         echo ---mem; head -c 16 /proc/{pid}/mem 2>&1; \
         echo ---ppid; cat /proc/$PPID/environ 2>&1 | head -c 200; \
         echo ---walk; for f in /proc/[0-9]*/environ; do cat $f 2>/dev/null | tr '\\0' '\\n'; done; \
         echo ---write; touch made && stat -c %u made",
        file = file.display(),
    );
    let answer = sh(&execd, Some(&bearer), &script).await;
    let text = out(&answer);
    println!("{text}");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], nobody, "not running as nobody: {text}");
    assert!(
        !text.contains(bearer.as_str()),
        "the bearer leaked:\n{}",
        text.replace(bearer.as_str(), "<BEARER>")
    );
    assert!(
        text.contains("Permission denied"),
        "execd's files and memory were readable:\n{text}"
    );
    for section in ["---file", "---environ", "---mem", "---ppid"] {
        let body = text
            .split(section)
            .nth(1)
            .unwrap()
            .lines()
            .nth(1)
            .unwrap_or("");
        assert!(body.contains("Permission denied"), "{section}: {body:?}");
    }
    // The walk reads what `nobody` may, which includes the commands of other tests (also `nobody`),
    // so it is not searched for execd's variables: execd's own environ is shown unreadable above,
    // and this test's bearer (unique) is checked absent from all of it.
    // The command ran in the root directory, as a user who could write there.
    let root = execd.dir.path().display().to_string();
    assert!(
        text.lines().any(|l| l == root),
        "cwd is not the root {root}:\n{text}"
    );
    assert!(
        text.trim_end().ends_with(&nobody),
        "the file nobody made is not nobody's:\n{text}"
    );
    // HOME is the user's own (its passwd entry), not root's.
    let passwd = String::from_utf8(
        Command::new("getent")
            .args(["passwd", "nobody"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    let home = passwd.trim().split(':').nth(5).unwrap().to_string();
    assert!(
        text.lines().any(|l| l == format!("HOME={home}")),
        "HOME is not {home}:\n{text}"
    );
    drop(decoy);
}

/// `--home` still wins over the user's own home, and a `SIGTERM` to execd (root) kills the jobs and
/// commands of the user it switched to.
#[tokio::test]
async fn under_run_as_home_is_configurable_and_shutdown_kills_the_users_jobs() {
    let Some(mut execd) = Execd::start_as_root(&[("EXECD_RUN_AS", "nobody"), ("EXECD_HOME", "/tmp")], &[])
    else {
        return;
    };
    let addr = std::net::SocketAddr::V4(execd.addr);
    let pids = execd.dir.path().join("pids");
    let script = format!(
        "echo $$ >> {p}; sleep 300 & echo $! >> {p}; echo HOME=$HOME; wait",
        p = pids.display()
    );
    let started = post(
        addr,
        &[],
        &json!({ "command": "sh", "args": ["-c", script],
                 "job": { "op": "start", "scope": "s", "id": "j", "hold": false } }),
    )
    .await;
    assert_eq!(started.status, 200);
    let mut listed = Vec::new();
    for _ in 0..200 {
        listed = std::fs::read_to_string(&pids)
            .unwrap_or_default()
            .lines()
            .map(|l| l.trim().parse::<u32>().unwrap())
            .collect();
        if listed.len() == 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert_eq!(listed.len(), 2, "the job never started");
    let read = post(addr, &[], &json!({ "command": "true", "job": { "op": "read", "scope": "s", "id": "j", "offset": 0, "max": 100 } })).await.json();
    use base64::Engine as _;
    let out = String::from_utf8(
        base64::engine::general_purpose::STANDARD
            .decode(read["data"].as_str().unwrap())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(out, "HOME=/tmp\n");
    let alive = |p: u32| std::path::Path::new(&format!("/proc/{p}")).exists();
    assert!(listed.iter().all(|&p| alive(p)));
    sudo(&["kill", "-TERM", &execd.pid().to_string()]);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while execd.child.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "execd did not exit");
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while listed.iter().any(|&p| alive(p)) {
        assert!(
            std::time::Instant::now() < deadline,
            "the user's job survived the shutdown"
        );
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}

/// Everything the spec lets a command depend on holds under `--run-as`: stdin, a `cwd` the user
/// can enter, an exit status, a missing program, a directory the user may not enter.
#[tokio::test]
async fn under_run_as_the_envelope_behaves() {
    let Some(execd) = Execd::start_as_root(&[("EXECD_RUN_AS", "nobody:daemon")], &[]) else {
        return;
    };
    let addr = std::net::SocketAddr::V4(execd.addr);
    let run = |v: Value| async move { post(addr, &[], &v).await };
    // gid is the named group.
    let a = run(json!({ "command": "id", "args": ["-g"] })).await.json();
    assert_eq!(a["stdout"], "1\n", "{a}");
    // stdin, and an exit status.
    use base64::Engine as _;
    let a = run(json!({ "command": "sh", "args": ["-c", "cat; exit 7"],
                        "stdin_base64": base64::engine::general_purpose::STANDARD.encode("hi") }))
    .await
    .json();
    assert_eq!(
        (a["exit_code"].as_i64(), a["stdout"].as_str()),
        (Some(7), Some("hi")),
        "{a}"
    );
    // A missing program is exit 127 with the reason.
    let a = run(json!({ "command": "no-such-program-xyz" }))
        .await
        .json();
    assert_eq!(a["exit_code"], 127, "{a}");
    assert!(
        a["stderr"].as_str().unwrap().contains("No such file"),
        "{a}"
    );
    // A directory the user cannot enter is refused as that user, not entered as root.
    let a = run(json!({ "command": "pwd", "cwd": "/root" }))
        .await
        .json();
    assert_eq!(a["exit_code"], 127, "{a}");
    assert!(
        a["stderr"].as_str().unwrap().contains("Permission denied"),
        "{a}"
    );
    // A stream ends with its status too, and the group is the command's.
    let s =
        run(json!({ "command": "sh", "args": ["-c", "echo hi; exit 3"], "stream": true })).await;
    let frames = s.frames();
    assert_eq!(frames.last().unwrap()["exit_code"], 3, "{frames:?}");
}

/// A `--root` the user cannot use is said so at startup, and a command then fails as the user.
#[tokio::test]
async fn under_run_as_an_unusable_root_is_reported_at_startup() {
    let Some(execd) = Execd::start_as_root_in(
        Some(std::path::Path::new("/root")),
        &[("EXECD_RUN_AS", "nobody")],
        &[],
    ) else {
        return;
    };
    assert!(
        execd.log().contains("is not writable by uid"),
        "{}",
        execd.log()
    );
    let a = post(
        std::net::SocketAddr::V4(execd.addr),
        &[],
        &json!({ "command": "true" }),
    )
    .await
    .json();
    assert_eq!(a["exit_code"], 127, "{a}");
}

// --- logs -------------------------------------------------------------------------------------

/// The log, at trace level, holds neither a notify token nor the secret half of a callback URL:
/// not when a callback succeeds, nor when the receiver refuses it, nor when it cannot be reached.
#[tokio::test]
async fn notify_tokens_and_callback_urls_never_reach_the_log() {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    let bearer = token("log");
    let token_secret = format!("TOKSECRET{}", std::process::id());
    let url_secret = format!("URLSECRET{}", std::process::id());
    let listener = beyond_ai_test_support::ports::tokio_listener().await;
    let port = listener.local_addr().unwrap().port();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        let mut n = 0;
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            n += 1;
            let tx = tx.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let got = sock.read(&mut buf).await.unwrap_or(0);
                let _ = tx.send(String::from_utf8_lossy(&buf[..got]).into_owned());
                // The first attempts are refused, so the error path is walked; then it answers.
                let status = if n <= 2 {
                    "500 Internal Server Error"
                } else {
                    "200 OK"
                };
                let _ = sock
                    .write_all(
                        format!(
                            "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .await;
            });
        }
    });
    let execd = Execd::start(
        &[("EXECD_KEPT_JOBS_SURVIVE_IDLE", "1"), ("EXECD_LOG", "trace")],
        &[("EXECD_BEARER", &bearer)],
    );
    let addr = std::net::SocketAddr::V4(execd.addr);
    let auth = format!("Bearer {bearer}");
    let h = [("Authorization", auth.as_str())];
    let url = format!("http://127.0.0.1:{port}/hook?sig={url_secret}");
    let dead = format!("http://127.0.0.1:1/dead?sig={url_secret}");
    for (id, target) in [("j1", url), ("j2", dead)] {
        let started = post(
            addr,
            &h,
            &json!({ "command": "sh", "args": ["-c", "sleep 0.2"],
                     "job": { "op": "start", "scope": "sc", "id": id, "hold": false } }),
        )
        .await;
        assert_eq!(started.status, 200);
        let kept = post(
            addr,
            &h,
            &json!({ "command": "true",
                     "job": { "op": "keep", "scope": "sc", "id": id, "num": 1, "kind": "background",
                              "notify": { "url": target, "token": token_secret } } }),
        )
        .await;
        assert_eq!(kept.status, 200, "{}", String::from_utf8_lossy(&kept.body));
    }
    // The receiver sees the token (it is the receiver's), so the run did reach the callback code.
    let first = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
        .await
        .expect("a callback arrived")
        .unwrap();
    assert!(
        first.contains(&token_secret) && first.contains(&url_secret),
        "{first}"
    );
    // Let the retries and the job ends run.
    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
    let log = execd.log();
    assert!(
        log.contains("callback"),
        "the callbacks left no trace line: {log}"
    );
    // The most verbose level: execd has no events below debug, so this is every line it can write.
    assert!(log.contains(" INFO "), "no log lines: {log}");
    for secret in [&token_secret, &url_secret, &bearer] {
        assert!(!log.contains(secret.as_str()), "a secret is in the log");
    }
}

/// Under `--run-as`, the jobs directory is execd's: a command (another user) can neither list it
/// nor read a chunk file of a job's retained output, while execd itself still reads the output back.
#[tokio::test]
async fn under_run_as_a_command_cannot_list_or_read_the_jobs_dir() {
    let base = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        base.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o755),
    )
    .unwrap();
    let Some(execd) = Execd::start_as_root(
        &[("EXECD_RUN_AS", "nobody"), ("EXECD_JOBS_DIR", base.path().to_str().unwrap())],
        &[],
    ) else {
        return;
    };
    let addr = std::net::SocketAddr::V4(execd.addr);
    // A job that writes three chunks.
    let started = post(
        addr,
        &[],
        &json!({ "command": "head", "args": ["-c", "200000", "/dev/zero"],
                 "job": { "op": "start", "scope": "s", "id": "j", "hold": false } }),
    )
    .await;
    assert_eq!(started.status, 200);
    let mut ended = json!(null);
    for _ in 0..200 {
        let r = post(
            addr,
            &[],
            &json!({ "command": "true",
                     "job": { "op": "read", "scope": "s", "id": "j", "offset": 0, "max": 10 } }),
        )
        .await
        .json();
        if r["state"] == "ended" {
            ended = r;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(ended["end"], 200_000, "the job did not finish: {ended}");
    // Chunk files exist (root sees them) ...
    let listing = String::from_utf8(
        Command::new("sudo")
            .args(["-n", "find", base.path().to_str().unwrap(), "-type", "f"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    assert!(listing.lines().count() >= 3, "no chunk files:\n{listing}");
    // ... and the command, as nobody, can reach none of it.
    let script = format!(
        "for d in {b}/execd-jobs-*; do echo \"dir $d\"; ls -A $d 2>&1; \
         for f in $d/*/*; do cat $f 2>&1 | head -c 20; done; done; \
         ls -A {b}/execd-jobs-*/0 2>&1; cat {b}/execd-jobs-*/0/0000000000 2>&1 | head -c 20",
        b = base.path().display()
    );
    let answer = sh(&execd, None, &script).await;
    let text = out(&answer);
    println!("{text}");
    assert!(text.contains("Permission denied"), "{text}");
    assert!(
        !text.contains("0000000000\n") && !text.contains('\0'),
        "a chunk name or its bytes were visible:\n{text}"
    );
    // execd itself still reads the output back.
    let read = post(
        addr,
        &[],
        &json!({ "command": "true",
                 "job": { "op": "read", "scope": "s", "id": "j", "offset": 70_000, "max": 100 } }),
    )
    .await
    .json();
    assert_eq!(read["offset"], 70_000);
    assert!(!read["data"].as_str().unwrap().is_empty(), "{read}");
}
