//! `run`/`serve` e2e: a mistyped tool name, a missing `--skill` path, or a contradictory pair of tool
//! flags fails at startup, naming the problem, before any model call.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{Read, Write};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{run_cmd, serve_cmd};

/// A gateway that answers every request with a 400 and counts them, so a run that gets past its
/// startup checks fails fast instead of waiting on a model.
struct Gateway {
    base: String,
    hits: Arc<AtomicUsize>,
}

fn gateway() -> Gateway {
    let listener = beyond_ai_test_support::ports::listener();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let hits = Arc::new(AtomicUsize::new(0));
    let seen = hits.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            seen.fetch_add(1, Ordering::SeqCst);
            let mut buf = [0u8; 65536];
            let _ = stream.read(&mut buf);
            let body = r#"{"error":{"type":"invalid_request_error","message":"test gateway"}}"#;
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            );
        }
    });
    Gateway { base, hits }
}

fn assert_no_model_call(gateway: &Gateway) {
    assert_eq!(
        gateway.hits.load(Ordering::SeqCst),
        0,
        "the model was called despite the bad flag"
    );
}

fn run(gateway: &Gateway, args: &[&str]) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "--gateway-url",
            &gateway.base,
            "--key",
            "bai_v1.test",
        ])
        .args(["--model", "claude-test", "--no-session-persistence"])
        .args(["--retry-max-retries", "0", "--no-retry"])
        .args(args)
        .arg("hi")
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

/// Run `run` with `args`; return its stderr after asserting it failed without calling the model.
fn run_fails(args: &[&str]) -> String {
    let gateway = gateway();
    let out = run(&gateway, args);
    assert!(!out.status.success(), "run {args:?} should fail");
    assert_no_model_call(&gateway);
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn run_refuses_an_unknown_tool_name_with_the_closest_match() {
    let err = run_fails(&["--tools", "raed,bash"]);
    assert!(
        err.contains("--tools: unknown tool `raed` (did you mean `read`?)"),
        "{err}"
    );
    let err = run_fails(&["--exclude-tools", "bsh"]);
    assert!(
        err.contains("--exclude-tools: unknown tool `bsh` (did you mean `bash`?)"),
        "{err}"
    );
    let err = run_fails(&["--deny-tool", "wirte"]);
    assert!(
        err.contains("--deny-tool: unknown tool `wirte` (did you mean `write`?)"),
        "{err}"
    );
}

#[test]
fn run_refuses_no_tools_with_a_tool_list() {
    let err = run_fails(&["--no-tools", "--tools", "read"]);
    assert!(err.contains("--no-tools and --tools contradict"), "{err}");
}

#[test]
fn run_refuses_a_skill_path_that_does_not_exist() {
    let err = run_fails(&["--skill", "/nope/skill"]);
    assert!(err.contains("--skill /nope/skill: does not exist"), "{err}");
}

#[test]
fn serve_refuses_the_same_flags_at_startup() {
    for (args, want) in [
        (
            &["--tools", "raed"][..],
            "--tools: unknown tool `raed` (did you mean `read`?)",
        ),
        (
            &["--no-tools", "--tools", "read"][..],
            "--no-tools and --tools contradict",
        ),
        (
            &["--skill", "/nope/skill"][..],
            "--skill /nope/skill: does not exist",
        ),
    ] {
        let gateway = gateway();
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("s.jsonl");
        let out = serve_cmd(
            env!("CARGO_BIN_EXE_beyond-ai-agent"),
            &gateway.base,
            session.to_str().unwrap(),
        )
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "serve {args:?} should fail: {err}");
        assert!(err.contains(want), "serve {args:?}: {err}");
        assert_no_model_call(&gateway);
    }
}

#[test]
fn mcp_tool_names_are_left_for_the_server_to_supply() {
    // An `mcp__` name can't be checked before its server connects, so it passes the startup check and
    // the run goes on to call the model.
    let gateway = gateway();
    let out = run(&gateway, &["--tools", "read,mcp__github__get_issue"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!err.contains("unknown tool"), "{err}");
    assert_eq!(gateway.hits.load(Ordering::SeqCst), 1, "{err}");
}
