//! e2e: a subagent's tokens count toward its parent's totals. The child runs its own loop on its own
//! `Session`, and its usage used to stop there, so `run`'s `[done in …]` line and `serve`'s
//! `get_session_stats` reported only the parent's own turns — delegated work looked free.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::process::Stdio;

use common::{
    SpawnGuarded, read_until_response, run_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

/// A parent that delegates once to a child that reads a file: four model requests, two of them the
/// child's.
fn script() -> Vec<String> {
    vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({"agent": "scout", "task": "read it"}).to_string(),
        ),
        turn_tool_use("call-2", "read", &json!({"path": "target.txt"}).to_string()),
        turn_text("read it"),
        turn_text("Done."),
    ]
}

/// The (input, output) tokens a scripted SSE response reports: `message_start`'s input, and the last
/// output figure (the closing `message_delta` carries the turn's total).
fn usage_of(sse: &str) -> (u64, u64) {
    let (mut input, mut output) = (0, 0);
    for data in sse.lines().filter_map(|l| l.strip_prefix("data: ")) {
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let usage = if v["type"] == "message_start" {
            &v["message"]["usage"]
        } else {
            &v["usage"]
        };
        if let Some(i) = usage["input_tokens"].as_u64() {
            input = i;
        }
        if let Some(o) = usage["output_tokens"].as_u64() {
            output = o;
        }
    }
    (input, output)
}

/// What the mock server billed across every request it answered.
fn billed() -> (u64, u64) {
    script()
        .iter()
        .map(|r| usage_of(r))
        .fold((0, 0), |(i, o), (ri, ro)| (i + ri, o + ro))
}

fn project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("target.txt"), "hello\n").unwrap();
    let agents = dir.path().join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("scout.md"),
        "---\nname: scout\ndescription: recon\ntools: read\n---\nYou are scout.\n",
    )
    .unwrap();
    dir
}

#[test]
fn runs_summary_counts_the_childs_tokens() {
    let dir = project();
    let (base, bodies) = spawn_model_server(script());
    let out = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "go",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(
        bodies
            .lock()
            .unwrap()
            .iter()
            .filter(|b| b.contains("\"messages\""))
            .count(),
        4,
        "two parent requests and two child requests"
    );
    let (input, output) = billed();
    let want = format!("{input} in / {output} out tokens");
    let line = stderr
        .lines()
        .find(|l| l.contains("[done in"))
        .unwrap_or("<none>");
    assert!(
        line.contains(&want),
        "the summary must equal what the server billed ({want}): {line}"
    );
}

#[test]
fn serves_session_stats_count_the_childs_tokens() {
    let dir = project();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(script());
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = common::serve_cmd(bin, &base, &session_file)
        .arg("--trust-project")
        .current_dir(dir.path())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(stdin, "{}", json!({"type": "prompt", "message": "go"})).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(stdin, "{}", json!({"type": "get_session_stats"})).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_session_stats");
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let stats = &frames.last().unwrap()["data"];
    let (input, output) = billed();
    assert_eq!(stats["input_tokens"], input, "{stats}");
    assert_eq!(stats["output_tokens"], output, "{stats}");
}
