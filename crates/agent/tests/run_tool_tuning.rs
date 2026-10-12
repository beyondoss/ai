//! `run` e2e: `--tool-tuning` reaches the advertised tool set on the wire, and a bad entry fails
//! startup rather than silently running the control arm.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text};

fn run_with(tuning: &[&str]) -> (std::process::Output, Vec<String>) {
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut args = vec![
        "run",
        "hi",
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--no-session-persistence",
    ];
    for t in tuning {
        args.extend(["--tool-tuning", t]);
    }
    let output = run_cmd(bin)
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    let bodies = bodies.lock().unwrap().clone();
    (output, bodies)
}

/// The tools in a recorded raw request (headers + body) whose name mentions `edit`.
fn edit_tools(request: &str) -> Vec<serde_json::Value> {
    let body = request.split_once("\r\n\r\n").map_or(request, |(_, b)| b);
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    v["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["name"].as_str().is_some_and(|n| n.contains("edit")))
        .cloned()
        .collect()
}

#[test]
fn edit_schema_tuning_changes_the_advertised_edit_tools() {
    let (out, bodies) = run_with(&[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let default = edit_tools(&bodies[0]);
    assert_eq!(default.len(), 1);
    assert!(default[0]["input_schema"]["properties"]["replace_all"].is_object());

    let (out, bodies) = run_with(&["edit-schema=edits"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let edits = edit_tools(&bodies[0]);
    assert_eq!(edits.len(), 1);
    assert!(
        edits[0]["input_schema"]["properties"]
            .get("old_string")
            .is_none()
    );

    let (out, bodies) = run_with(&["edit-schema=split,edit-pair=lenient"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let names: Vec<String> = edit_tools(&bodies[0])
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, ["edit", "multi_edit"]);
}

#[test]
fn an_unknown_tool_tuning_entry_fails_startup() {
    let (out, bodies) = run_with(&["edit-schema=bogus"]);
    assert!(!out.status.success(), "a typo must not run the control arm");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("edit-schema"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(bodies.is_empty(), "no request may be sent");
}

#[test]
fn serve_builds_its_default_system_prompt_from_the_tuned_registry() {
    // `serve` renders its default system prompt from a registry of its own; under
    // `edit-schema=split` the edit bullets must describe `edit` + `multi_edit`, not edits[].
    use std::io::Write as _;

    use common::{SpawnGuarded as _, read_until_response, serve_cmd};

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args(["--tool-tuning", "edit-schema=split"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        serde_json::json!({ "type": "prompt", "message": "hi" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let request = bodies.lock().unwrap()[0].clone();
    assert!(
        request.contains("Use edit for one precise change"),
        "{request}"
    );
    assert!(
        !request.contains("Use edit for precise changes (edits[]"),
        "{request}"
    );
    let names: Vec<String> = edit_tools(&request)
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, ["edit", "multi_edit"]);
}
