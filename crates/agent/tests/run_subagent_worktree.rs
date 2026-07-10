//! `run` e2e: a worktree-isolated `subagent` child edits a file, and merge-back lands the change in the
//! real tree. This exercises the whole isolation path through the actual tool — `Worktree::create`,
//! rooted child tools, `child_delta`, `git apply` — which the `worktree.rs` unit tests only cover
//! piece by piece.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

fn git(dir: &std::path::Path, args: &[&str]) {
    let ok = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap()
        .status
        .success();
    assert!(ok, "git {args:?} failed");
}

/// A git repo with one commit containing `target.txt`, plus a worktree-isolated writer agent.
fn repo_with_writer() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    git(p, &["init", "--quiet", "-b", "main"]);
    git(p, &["config", "user.name", "t"]);
    git(p, &["config", "user.email", "t@t"]);
    std::fs::write(p.join("target.txt"), "original contents\n").unwrap();

    let agents = p.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("writer.md"),
        "---\nname: writer\ndescription: edits files\ntools: read,edit,write\nisolation: worktree\n---\nYou are writer.\n",
    )
    .unwrap();

    git(p, &["add", "-A"]);
    git(p, &["commit", "--quiet", "-m", "init"]);
    dir
}

#[test]
fn a_worktree_child_edit_is_merged_back_into_the_real_tree() {
    let dir = repo_with_writer();

    let (base, _bodies) = spawn_model_server(vec![
        // Parent delegates a single edit to the worktree-isolated writer.
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "writer", "task": "rewrite target.txt" }).to_string(),
        ),
        // Child writes target.txt using a RELATIVE path — it must resolve against the worktree root, and
        // the write must survive merge-back into the main tree.
        turn_tool_use(
            "call-2",
            "write",
            &json!({ "path": "target.txt", "content": "rewritten by the subagent\n" }).to_string(),
        ),
        // Child, after the successful write, finishes.
        turn_text("I rewrote target.txt."),
        // Parent finishes.
        turn_text("Done."),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "rewrite the file",
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

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Merge-back must have landed the child's edit in the real tree.
    let merged = std::fs::read_to_string(dir.path().join("target.txt")).unwrap();
    assert_eq!(
        merged, "rewritten by the subagent\n",
        "the worktree child's edit must be merged into the main tree"
    );

    // And no worktree may be left behind: `git worktree list` should show only the main checkout.
    let wt_list = Command::new("git")
        .arg("-C")
        .arg(dir.path())
        .args(["worktree", "list"])
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&wt_list.stdout);
    assert_eq!(
        listing.lines().count(),
        1,
        "a clean-merged worktree must be removed; git worktree list:\n{listing}"
    );
}
