//! The binary as an operator meets it: its startup lines, its bearer sources, its signals.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::{Duration, Instant};

use serde_json::json;
use support::{Execd, http::post};

fn log(execd: &Execd) -> String {
    std::fs::read_to_string(execd.dir.path().join("execd.log")).unwrap_or_default()
}

#[tokio::test]
async fn it_says_so_loudly_when_nothing_protects_it() {
    let open = Execd::start(&[], &[]);
    assert!(
        log(&open).contains("NO BEARER CONFIGURED"),
        "{}",
        log(&open)
    );
    let secured = Execd::start(&[], &[("EXECD_BEARER", "x")]);
    assert!(!log(&secured).contains("NO BEARER"), "{}", log(&secured));
}

#[tokio::test]
async fn the_bearer_comes_from_a_file_and_is_never_logged() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bearer");
    std::fs::write(&file, "from-a-file\n").unwrap();
    let execd = Execd::start(&[("EXECD_BEARER_FILE", file.to_str().unwrap())], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    let req = json!({ "command": "true", "args": [], "timeout_ms": 5_000 });
    assert_eq!(post(addr, &[], &req).await.status, 401);
    let auth = [("Authorization", "Bearer from-a-file")];
    assert_eq!(post(addr, &auth, &req).await.status, 200);
    assert!(!log(&execd).contains("from-a-file"));
    // One line per request, with the command's name and no arguments.
    let lines = log(&execd);
    assert!(
        lines.contains("kind=\"exec\"") || lines.contains("kind=exec"),
        "{lines}"
    );
}

#[tokio::test]
async fn sigterm_kills_every_running_group_and_exits_cleanly() {
    let mut execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    let pidfile = execd.dir.path().join("job.pid");
    let script = format!("echo $$ > {}; exec sleep 300", pidfile.display());
    let req = json!({
        "command": "sh", "args": ["-c", script], "timeout_ms": 30_000,
        "job": { "op": "start", "scope": "s", "id": "ab", "hold": false, "label": "" },
    });
    assert_eq!(post(addr, &[], &req).await.status, 200);
    let pid: u32 = loop {
        if let Some(p) = std::fs::read_to_string(&pidfile)
            .ok()
            .and_then(|s| s.trim().parse().ok())
        {
            break p;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    let proc = format!("/proc/{pid}");
    assert!(std::path::Path::new(&proc).exists());
    std::process::Command::new("kill")
        .args(["-TERM", &execd.pid().to_string()])
        .status()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(s) = execd.child.try_wait().unwrap() {
            break s;
        }
        assert!(Instant::now() < deadline, "execd did not exit on SIGTERM");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(status.success(), "{status:?}");
    // The job went with it (adopted by init and reaped, or still a zombie for a moment).
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::fs::read_to_string(format!("{proc}/stat")).is_ok_and(|s| !s.contains(") Z")) {
        assert!(Instant::now() < deadline, "the job outlived execd");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(log(&execd).contains("killed_groups=1"), "{}", log(&execd));
}
