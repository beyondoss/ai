//! `run` e2e: `--tool-tuning` reaches the advertised tool set on the wire, and a bad entry fails
//! startup rather than silently running the control arm.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};

fn run_with(tuning: &[&str]) -> (std::process::Output, Vec<String>) {
    run_script(vec![turn_text("ok")], tuning)
}

fn run_script(responses: Vec<String>, tuning: &[&str]) -> (std::process::Output, Vec<String>) {
    let (base, bodies) = spawn_model_server(responses);
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

/// A failing pytest run big enough to digest (> 16 KiB), as `cat` will print it.
fn pytest_log(dir: &std::path::Path) -> std::path::PathBuf {
    let mut s = String::from(
        "============================= test session starts ==============================\n",
    );
    s.push_str(&format!("tests/test_a.py {}\n", ".".repeat(300)));
    s.push_str(
        "=================================== FAILURES ===================================\n",
    );
    for i in 0..30 {
        s.push_str(&format!(
            "_________________________________ test_case_{i} _________________________________\n"
        ));
        for j in 0..20 {
            s.push_str(&format!(
                "    assert value_{i}_{j} == expected  # traceback line {j}\n"
            ));
        }
    }
    s.push_str(
        "=========================== short test summary info ============================\n",
    );
    for i in 0..30 {
        s.push_str(&format!(
            "FAILED tests/test_a.py::test_case_{i} - AssertionError\n"
        ));
    }
    s.push_str("======================= 30 failed, 270 passed in 3.21s ========================\n");
    let path = dir.join("pytest.log");
    std::fs::write(&path, s).unwrap();
    path
}

fn tool_result_after_cat(tuning: &[&str]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let log = pytest_log(dir.path());
    let cmd =
        serde_json::json!({ "command": format!("cat {}; exit 1", log.display()) }).to_string();
    let (out, bodies) = run_script(
        vec![turn_tool_use("toolu_1", "bash", &cmd), turn_text("done")],
        tuning,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    bodies[1].clone()
}

#[test]
fn bash_test_digest_condenses_a_large_pytest_run_and_the_default_does_not() {
    let digested = tool_result_after_cat(&["bash-test-digest=pytest"]);
    assert!(digested.contains("pytest output condensed"), "{digested}");
    assert!(digested.contains("short test summary info"), "{digested}");
    assert!(digested.contains("30 failed, 270 passed"), "{digested}");
    assert!(
        digested.contains("Command exited with code 1"),
        "{digested}"
    );
    assert!(!digested.contains("test session starts"), "{digested}");

    let plain = tool_result_after_cat(&[]);
    assert!(!plain.contains("pytest output condensed"));
    assert!(plain.contains("test session starts"));
}

#[test]
fn bash_max_bytes_is_what_the_bash_description_promises() {
    let (out, bodies) = run_with(&["bash-max-bytes=16k"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(bodies[0].contains("or 16.0KB, whichever"), "{}", bodies[0]);
    let (_, bodies) = run_with(&[]);
    assert!(bodies[0].contains("or 50.0KB, whichever"), "{}", bodies[0]);
}
