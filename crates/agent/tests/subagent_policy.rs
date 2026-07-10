//! THE security regression test for `subagent`. `Agent::new` installs `NoHooks`, so a child built
//! without inheriting the parent's `ToolPolicy` would run whatever the parent's `--deny-*` flags forbid
//! — making `subagent` a sandbox escape. This drives the real binary end to end and proves a child is
//! actually blocked, by having the forbidden command try to create a file and asserting it never appears.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

fn write_agent(dir: &std::path::Path, name: &str, tools: &str, body: &str) {
    let agents = dir.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join(format!("{name}.md")),
        format!("---\nname: {name}\ndescription: test {name}\ntools: {tools}\n---\n{body}\n"),
    )
    .unwrap();
}

#[test]
fn a_child_cannot_run_a_bash_command_the_parents_deny_pattern_forbids() {
    let dir = tempfile::tempdir().unwrap();
    write_agent(dir.path(), "runner", "read,bash", "You are runner.");

    // If the policy is NOT inherited, this command runs and creates the sentinel. If it IS inherited,
    // the `FORBIDDEN` substring trips `--deny-bash-pattern` and the command never executes.
    let sentinel = dir.path().join("SHOULD-NOT-EXIST");
    let forbidden_command = format!("touch '{}' # FORBIDDEN", sentinel.display());

    let (base, bodies) = spawn_model_server(vec![
        // Parent delegates to the runner.
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "runner", "task": "run the command" }).to_string(),
        ),
        // Child tries the forbidden bash command.
        turn_tool_use(
            "call-2",
            "bash",
            &json!({ "command": forbidden_command }).to_string(),
        ),
        // Child, after the (blocked) tool result, gives up.
        turn_text("I was not allowed to run that."),
        // Parent finishes.
        turn_text("The runner could not complete the task."),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run the forbidden thing",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
            "--deny-bash-pattern",
            "FORBIDDEN",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // The load-bearing assertion: the forbidden command never ran.
    assert!(
        !sentinel.exists(),
        "SECURITY REGRESSION: a subagent ran a bash command the parent's --deny-bash-pattern forbids"
    );

    // And the child's follow-up turn should have been handed a policy-denial tool result.
    let bodies = bodies.lock().unwrap();
    let child_followup = &bodies[2];
    assert!(
        child_followup
            .to_ascii_lowercase()
            .contains("denied by policy")
            || child_followup
                .to_ascii_lowercase()
                .contains("blocked by policy"),
        "the child should have received a policy denial as its tool result: {child_followup}"
    );
}

#[test]
fn a_child_can_run_a_bash_command_the_policy_permits() {
    // The control: without a matching deny pattern, the same delegation runs the command for real. This
    // proves the block above is the policy doing its job, not the plumbing simply never running bash.
    let dir = tempfile::tempdir().unwrap();
    write_agent(dir.path(), "runner", "read,bash", "You are runner.");
    let sentinel = dir.path().join("SHOULD-EXIST");
    let command = format!("touch '{}'", sentinel.display());

    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "runner", "task": "run the command" }).to_string(),
        ),
        turn_tool_use("call-2", "bash", &json!({ "command": command }).to_string()),
        turn_text("Done."),
        turn_text("The runner completed the task."),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run the allowed thing",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
            "--deny-bash-pattern",
            "SOMETHING-ELSE",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        sentinel.exists(),
        "a permitted bash command must actually run inside the child"
    );
}
