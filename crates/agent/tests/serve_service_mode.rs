//! `serve --service`: what it demands at startup, and what a session looks like once it is up.
//!
//! Startup validation is half the security property. Service mode's guarantee is "nothing of the
//! replica's reaches a tenant", and most of the ways to break it are a flag: `--key` would hand
//! every tenant the replica's credential, `--session-dir` would collapse them onto one directory,
//! `--exec-url` would point them all at one sandbox. Each is **refused**, not ignored — an operator
//! who passes one must find out at startup, not from an audit months later.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use common::service::Service;
use common::{
    SpawnGuarded, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers,
    ws_next_frame, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Run `serve` to completion and return its stderr, asserting it refused to start.
fn startup_error(cmd: &mut Command) -> String {
    let output = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn_guarded()
        .wait_with_output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        !output.status.success(),
        "expected a startup failure; stderr: {stderr}"
    );
    stderr
}

struct Keys {
    dir: tempfile::TempDir,
    flag: String,
    seal: std::path::PathBuf,
    shard: std::path::PathBuf,
}

fn keys() -> Keys {
    let dir = tempfile::tempdir().unwrap();
    let minter = common::grant::Minter::new(dir.path());
    let shard = dir.path().join("shard");
    std::fs::create_dir_all(&shard).unwrap();
    Keys {
        flag: minter.grant_key_flag(),
        seal: minter.seal_key().to_path_buf(),
        shard,
        dir,
    }
}

/// The minimal valid `--service` command line, for a test to spoil one piece of.
/// Every use is a startup that must fail; were one to succeed, port 0 is still the kernel's to pick.
fn service_cmd(k: &Keys) -> Command {
    let mut c = crate::common::agent_cmd();
    c.args([
        "serve",
        "--service",
        "--gateway-url",
        "http://127.0.0.1:1",
        "--model",
        "claude-test",
        "--listen",
        // port-0: the child binds it and keeps it (and every use is a startup meant to fail).
        "127.0.0.1:0",
        "--grant-key",
        &k.flag,
        "--seal-key",
        &k.seal.to_string_lossy(),
        "--shard",
        &format!("a={}", k.shard.display()),
    ]);
    c.env("HOME", common::ISOLATED_HOME);
    c
}

#[test]
fn service_mode_demands_a_verifier_a_shard_and_a_listener() {
    let k = keys();

    let base: Vec<String> = vec![
        "serve".into(),
        "--service".into(),
        "--model".into(),
        "claude-test".into(),
    ];
    // port-0: the child binds it and keeps it (and every use is a startup meant to fail).
    let listen = ["--listen".to_string(), "127.0.0.1:0".to_string()];
    let verifier = [
        "--grant-key".to_string(),
        k.flag.clone(),
        "--seal-key".to_string(),
        k.seal.to_string_lossy().into_owned(),
    ];
    let shard = ["--shard".to_string(), format!("a={}", k.shard.display())];

    let run = |extra: Vec<&[String]>| {
        let mut c = crate::common::agent_cmd();
        c.args(&base);
        for group in extra {
            c.args(group);
        }
        c.env("HOME", common::ISOLATED_HOME);
        startup_error(&mut c)
    };

    let err = run(vec![&listen, &shard]);
    assert!(err.contains("--grant-key"), "{err}");
    let err = run(vec![&listen, &verifier]);
    assert!(err.contains("--shard"), "{err}");
    // No listener at all: stdio `serve --service` has no per-connection grant to carry, so it is
    // refused rather than quietly serving one anonymous session.
    let err = run(vec![&verifier, &shard]);
    assert!(err.contains("--listen"), "{err}");
}

#[test]
fn a_shard_must_be_a_name_and_an_absolute_path() {
    let k = keys();
    for bad in ["a", "a=relative/path", "a.b=/tmp", "=/tmp"] {
        let mut c = service_cmd(&k);
        // Append a second `--shard`; the bad one is what fails.
        c.arg("--shard").arg(bad);
        let err = startup_error(&mut c);
        assert!(err.contains("--shard"), "{bad}: {err}");
    }
}

#[test]
fn every_refused_flag_fails_at_startup() {
    let k = keys();
    let scratch = k.dir.path().join("scratch");
    std::fs::create_dir_all(&scratch).unwrap();
    let scratch = scratch.to_string_lossy().into_owned();
    let refused: Vec<Vec<String>> = vec![
        vec!["--key".into(), "bai_v1.test".into()],
        vec!["--session-file".into(), format!("{scratch}/s.jsonl")],
        vec!["--session-dir".into(), scratch.clone()],
        vec!["--session-id".into(), "abc".into()],
        vec!["--continue".into()],
        vec!["--name".into(), "a name".into()],
        vec!["--no-session-persistence".into()],
        vec!["--memory".into(), format!("file://{scratch}")],
        vec!["--trust-project".into()],
        vec!["--force-untrusted".into()],
        vec!["--exec-url".into(), "http://127.0.0.1:1/exec".into()],
        vec!["--exec-cmd".into(), "sh -c {}".into()],
        vec!["--exec-header".into(), "A: b".into()],
        vec!["--web-allow-private".into()],
        vec!["--web-allow-host".into(), "example.com".into()],
        vec!["--skill".into(), scratch.clone()],
        vec!["--prompt-template".into(), scratch.clone()],
        vec!["--bash-shell-path".into(), "/bin/sh".into()],
    ];
    for flag in &refused {
        let mut c = service_cmd(&k);
        c.args(flag);
        let err = startup_error(&mut c);
        assert!(
            err.contains(&flag[0]) && err.contains("--service"),
            "{flag:?} must be refused by name; got: {err}"
        );
    }
}

/// A refused flag's **env var** is refused too: on a replica the ambient environment is exactly
/// where an operator credential would come from, and `--key`/`AI_AGENT_KEY` are one flag.
#[test]
fn an_ambient_agent_key_refuses_startup() {
    let k = keys();
    let mut c = service_cmd(&k);
    c.env("AI_AGENT_KEY", "bai_v1.the-replicas-own-key");
    let err = startup_error(&mut c);
    assert!(err.contains("--key") && err.contains("--service"), "{err}");
}

/// `--code-mode` is refused too, but only a binary built with the feature can even parse it — on the
/// default build it fails for its own reason. Asserted separately so the table above stays exact.
#[test]
#[cfg(feature = "code-mode")]
fn code_mode_is_refused() {
    let k = keys();
    let mut c = service_cmd(&k);
    c.arg("--code-mode");
    let err = startup_error(&mut c);
    assert!(err.contains("--code-mode"), "{err}");
}

/// The state a tenant sees is the tenant's: its sandbox workspace, no branch lookup against the
/// replica's checkout, and no path into the replica's filesystem.
#[tokio::test]
async fn get_state_reports_the_sandbox_workspace_and_no_host_paths() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("00tenant1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("a valid grant opens its session");

    ws_send(&mut ws, json!({"type":"get_state","id":"g1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    let state = &frames.last().unwrap()["data"];
    assert_eq!(state["session_id"], "s1.alpha");
    assert_eq!(
        state["cwd"],
        svc.workspace.to_string_lossy().into_owned(),
        "cwd must be the grant's workspace_root"
    );
    assert_eq!(state["git_branch"], Value::Null, "no host git lookup");
    assert_eq!(state["cwd_stale"], false);
    assert_eq!(
        state["session_file"],
        Value::Null,
        "the transcript's mount path is not the tenant's business"
    );

    // And the session really did land in the tenant's own directory.
    let dir = svc.sessions_dir("s1", "00tenant1");
    assert!(dir.is_dir(), "{} must exist", dir.display());
    assert!(
        std::fs::read_dir(&dir).unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("s1.alpha")),
        "the transcript must be named for the session id under {}",
        dir.display()
    );
}

/// The same holds while a prompt is running: the busy loop answers `get_state` itself, and it must
/// not judge the tenant's sandbox path against the replica's disk. The workspace here exists only in
/// the sandbox, so a host-side check would call it stale.
#[tokio::test]
async fn get_state_mid_run_is_never_cwd_stale() {
    const WORKSPACE: &str = "/sandbox/workspace-not-on-the-replica";
    // The sandbox has no such directory either, so the tool fails at once; the run stays busy
    // because the model's answer to that takes two seconds.
    let turn1 = turn_tool_use(
        "toolu_gs",
        "bash",
        &json!({ "command": "sleep 2" }).to_string(),
    );
    let (base, _requests) = beyond_test_support::spawn_model_server_fn(move |req: &str| {
        if req.contains("tool_result") {
            std::thread::sleep(std::time::Duration::from_secs(2));
            turn_text("done")
        } else {
            turn1.clone()
        }
    });
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token_with("00tenant1", "s1.alpha", |c| {
        c.workspace_root = WORKSPACE.into()
    });
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("a valid grant opens its session");

    ws_send(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    // Land mid-run by waiting for the run to say it got there: `tool_start` means the tool is
    // executing right now.
    while let Some(f) = ws_next_frame(&mut ws).await {
        if f["type"] == "event" && f["event"]["kind"] == "tool_start" {
            break;
        }
    }
    ws_send(&mut ws, json!({"type":"get_state","id":"g1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    let state = &frames.last().unwrap()["data"];
    assert_eq!(
        state["is_streaming"], true,
        "must be the busy loop's answer: {state:#?}"
    );
    assert_eq!(state["cwd"], WORKSPACE);
    assert_eq!(state["cwd_stale"], false, "got: {state:#?}");
}

/// A failed reply sent while a run is busy is redacted exactly like an idle one. `get_tree` with an
/// unknown `since` echoes the id back in its error, so the shard root the client names comes back
/// inside an error string: the stand-in for any storage error that names the file it failed on.
#[tokio::test]
async fn failed_reply_mid_run_is_redacted() {
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_rd",
            "bash",
            &json!({ "command": "sleep 2" }).to_string(),
        ),
        turn_text("done"),
    ]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("00tenant1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("a valid grant opens its session");

    ws_send(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    while let Some(f) = ws_next_frame(&mut ws).await {
        if f["type"] == "event" && f["event"]["kind"] == "tool_start" {
            break;
        }
    }
    let root = svc.shard("s1").to_string_lossy().into_owned();
    ws_send(
        &mut ws,
        json!({"type":"get_tree","id":"t1","since":format!("{root}/entry")}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "get_tree").await;
    let reply = frames.last().unwrap();
    assert_eq!(reply["success"], false, "{reply:#?}");
    let error = reply["error"].as_str().expect("an error string");
    assert!(error.contains("no entry with id"), "got: {error}");
    assert!(
        !error.contains(&root),
        "a host path reached the tenant mid-run: {error}"
    );
    assert!(error.contains("<store>"), "got: {error}");
}

/// The same rule as `get_state`, on the other replies that report `cwd_stale`: the `ready` frame a
/// session sends on opening and the `new_session` answer. `switch_session` is refused in service
/// mode, and `fork`/`clone` derive a detached session whose reply carries only the new id, so the
/// last three never report the field there; they are checked so that stays true.
#[tokio::test]
async fn every_reply_that_reports_cwd_stale_says_false_in_service_mode() {
    const WORKSPACE: &str = "/sandbox/workspace-not-on-the-replica";
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token_with("00tenant1", "s1.alpha", |c| {
        c.workspace_root = WORKSPACE.into()
    });
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("a valid grant opens its session");

    // The `ready` frame is the session's first frame; the connect helper leaves it unread.
    let ready = loop {
        let f = ws_next_frame(&mut ws).await.expect("a ready frame");
        if f["type"] == "ready" {
            break f;
        }
    };
    assert_eq!(ready["cwd_stale"], false, "ready: {ready:#?}");

    // `fork`/`clone` copy what is on disk, so give the session a turn first.
    ws_send(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    ws_read_until_response(&mut ws, "prompt").await;

    for command in ["fork", "clone", "new_session"] {
        ws_send(&mut ws, json!({"type": command, "id": command})).await;
        let frames = ws_read_until_response(&mut ws, command).await;
        let reply = frames.last().unwrap();
        assert_eq!(reply["success"], true, "{command}: {reply:#?}");
        assert_ne!(
            reply["data"]["cwd_stale"], true,
            "{command} must not judge the sandbox path against the replica's disk: {reply:#?}"
        );
        if command == "new_session" {
            assert_eq!(reply["data"]["cwd_stale"], false, "{reply:#?}");
        }
    }
}

/// Every command that could reach the replica answers with a refusal rather than doing something
/// surprising. (`fork`/`clone`/`new_session` are *not* here: they mint ids on the session's own
/// shard — see `serve_service_derived.rs`.)
#[tokio::test]
async fn the_refused_commands_answer_with_an_error() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();

    for command in [
        "set_exec_endpoint",
        "login",
        "submit_code",
        "abort_login",
        "logout",
        "auth_status",
        "switch_session",
    ] {
        ws_send(&mut ws, json!({"type": command, "id": command})).await;
        let frames = ws_read_until_response(&mut ws, command).await;
        let response = frames.last().unwrap();
        assert_eq!(response["success"], false, "{command}: {response}");
        assert!(
            response["error"]
                .as_str()
                .unwrap_or_default()
                .contains("service mode"),
            "{command}: {response}"
        );
    }

    // `reload` is *not* among them: sandbox discovery re-aimed it at the tenant's own filesystem,
    // which is the whole reason it could come back (see `serve_service_discovery.rs`).
    ws_send(&mut ws, json!({"type":"reload","id":"reload"})).await;
    let frames = ws_read_until_response(&mut ws, "reload").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    // The session is still perfectly usable afterwards.
    ws_send(&mut ws, json!({"type":"get_state","id":"after"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true);
}

/// `export_html` returns the document inline: writing it would put a whole transcript on the
/// replica's disk and hand the caller a host path to fetch it from.
#[tokio::test]
async fn export_html_is_inline_and_refuses_an_output_path() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    ws_send(&mut ws, json!({"type":"export_html","id":"e1"})).await;
    let frames = ws_read_until_response(&mut ws, "export_html").await;
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "{response}");
    let html = response["data"]["html"].as_str().expect("inline html");
    assert!(
        html.starts_with("<!DOCTYPE html>"),
        "{}",
        &html[..80.min(html.len())]
    );
    assert!(
        response["data"]["path"].is_null(),
        "no host path: {response}"
    );

    ws_send(
        &mut ws,
        json!({"type":"export_html","id":"e2","output_path":"/tmp/leak.html"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "export_html").await;
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "{response}");
    assert!(!Path::new("/tmp/leak.html").exists());
}

/// A runtime toggle applies to this session and is never written to the replica's settings file —
/// which is the operator's, and shared with every other tenant on the box.
#[tokio::test]
async fn a_settings_toggle_stays_session_local() {
    let (base, _requests) = spawn_model_server(vec![]);
    let host_home = tempfile::tempdir().unwrap();
    let svc = Service::start_with(
        &base,
        &["s1"],
        common::service::Options {
            host_home: Some(host_home.path().to_path_buf()),
            ..Default::default()
        },
    )
    .await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();

    ws_send(
        &mut ws,
        json!({"type":"set_auto_compaction","id":"c1","enabled":false}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "set_auto_compaction").await;
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["data"]["auto_compaction"], false);

    // …and it took effect for this session.
    ws_send(&mut ws, json!({"type":"get_state","id":"g1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["data"]["auto_compaction"], false);

    // Nothing was written under the replica's own HOME.
    let settings = host_home.path().join(".beyond");
    assert!(
        !settings.exists(),
        "service mode must not write the replica's settings: {}",
        settings.display()
    );
}

/// A session whose sandbox never answers must not fall back to the replica, and must *say so*: the
/// old behaviour was a line on the replica's stderr and a socket that accepted commands forever.
#[tokio::test]
async fn a_session_whose_sandbox_is_unreachable_fails_with_an_error_frame() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    // A port nothing listens on: the probe cannot succeed.
    let token = svc.token_with("t1", "s1.alpha", |c| {
        c.exec_url = "http://127.0.0.1:1/exec".into()
    });
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("the grant is valid, so the socket is accepted");
    ws_send(&mut ws, json!({"type":"get_state","id":"g1"})).await;

    let mut saw_error = false;
    while let Some(frame) = common::ws_next_frame(&mut ws).await {
        if frame["type"] == "error" {
            saw_error = true;
            let text = frame["error"].as_str().unwrap_or_default();
            assert!(text.contains("sandbox"), "{frame}");
            break;
        }
    }
    assert!(saw_error, "a session that fails to start must say so");
}

/// The replica's stderr is the operator's channel; assert the mode is announced there so a
/// misconfigured deployment is obvious in a log.
#[tokio::test]
async fn the_replica_announces_service_mode_and_its_shards() {
    let (base, _requests) = spawn_model_server(vec![]);
    let mut svc = Service::start(&base, &["s1", "s2"]).await;
    // Give the daemon a moment's work so its startup lines are certainly flushed.
    let token = svc.token("t1", "s1.alpha");
    drop(
        ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
            .await
            .unwrap(),
    );
    let _ = svc.child.kill();
    let _ = svc.child.wait();
    // Copied to the log line by line as the replica writes it; give the copy a moment to catch up.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let stderr = loop {
        let stderr = std::fs::read_to_string(&svc.stderr_log).unwrap_or_default();
        let complete =
            stderr.contains("service mode") && stderr.contains("s1") && stderr.contains("s2");
        if complete || std::time::Instant::now() > deadline {
            break stderr;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(stderr.contains("service mode"), "{stderr}");
    assert!(stderr.contains("s1") && stderr.contains("s2"), "{stderr}");
}
