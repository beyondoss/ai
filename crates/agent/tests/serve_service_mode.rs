//! `serve --service`: what it demands at startup, and what a session looks like once it is up.
//!
//! Startup validation is half the security property. Service mode's guarantee is "nothing of the
//! replica's reaches a tenant", and most of the ways to break it are a flag: `--key` would hand
//! every tenant the replica's credential, `--session-dir` would collapse them onto one directory,
//! `--exec-url` would point them all at one sandbox. Each is **refused**, not ignored — an operator
//! who passes one must find out at startup, not from an audit months later.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use common::service::Service;
use common::{
    BIN, SpawnGuarded, free_port, spawn_model_server, turn_text, ws_connect_with_headers,
    ws_read_until_response, ws_send,
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
fn service_cmd(k: &Keys, port: u16) -> Command {
    let mut c = Command::new(BIN);
    c.args([
        "serve",
        "--service",
        "--gateway-url",
        "http://127.0.0.1:1",
        "--model",
        "claude-test",
        "--listen",
        &format!("127.0.0.1:{port}"),
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
    let port = free_port();

    let base: Vec<String> = vec![
        "serve".into(),
        "--service".into(),
        "--model".into(),
        "claude-test".into(),
    ];
    let listen = ["--listen".to_string(), format!("127.0.0.1:{port}")];
    let verifier = [
        "--grant-key".to_string(),
        k.flag.clone(),
        "--seal-key".to_string(),
        k.seal.to_string_lossy().into_owned(),
    ];
    let shard = ["--shard".to_string(), format!("a={}", k.shard.display())];

    let run = |extra: Vec<&[String]>| {
        let mut c = Command::new(BIN);
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
        let mut c = service_cmd(&k, free_port());
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
        let mut c = service_cmd(&k, free_port());
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
    let mut c = service_cmd(&k, free_port());
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
    let mut c = service_cmd(&k, free_port());
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

/// Every command that could reach the replica — or that needs a prefixed derived id — answers with a
/// refusal rather than doing something surprising.
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
        "reload",
        "fork",
        "clone",
        "new_session",
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
    let settings = host_home.path().join(".claude");
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
    let mut stderr = String::new();
    if let Some(mut pipe) = svc.child.stderr.take() {
        let _ = pipe.read_to_string(&mut stderr);
    }
    assert!(stderr.contains("service mode"), "{stderr}");
    assert!(stderr.contains("s1") && stderr.contains("s2"), "{stderr}");
}
