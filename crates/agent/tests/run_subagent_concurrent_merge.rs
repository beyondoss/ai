//! `run` e2e: two parallel worktree-isolated children whose edits both merge back into one repo. This
//! is the end-to-end coverage for **concurrent merge-back**: each child applies its patch to the shared
//! repo, which must serialize (git holds `index.lock` per repo) rather than one child's `git apply`
//! spuriously failing because another's is mid-flight. The conflict-marker path (two children editing the
//! *same* line) is covered deterministically by `worktree.rs`'s unit tests, where apply order is
//! controlled — forcing a line-level conflict between two independently-scheduled children here would be
//! timing-dependent.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{run_cmd, spawn_model_server_routed, turn_text, turn_tool_use};
use serde_json::json;

fn git(dir: &std::path::Path, args: &[&str]) {
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .status()
            .unwrap()
            .success(),
        "git {args:?} failed"
    );
}

#[test]
fn two_parallel_worktree_children_both_merge_back_into_one_repo() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path();
    git(repo, &["init", "--quiet", "-b", "main"]);
    git(repo, &["config", "user.name", "t"]);
    git(repo, &["config", "user.email", "t@t"]);
    // Two distinct files, one per child — disjoint edits that must both land.
    std::fs::write(repo.join("a.txt"), "a original\n").unwrap();
    std::fs::write(repo.join("b.txt"), "b original\n").unwrap();

    let agents = repo.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("writer.md"),
        "---\nname: writer\ndescription: edits files\ntools: read,edit,write,ls\nisolation: worktree\n---\nYou are writer.\n",
    )
    .unwrap();
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "--quiet", "-m", "init"]);

    let (base, _bodies) = spawn_model_server_routed(
        vec![
            // Parent's final turn (tool result carries both children's outputs).
            ("succeeded".to_string(), turn_text("both writers finished")),
            // Either child's post-write turn.
            ("wrote".to_string(), turn_text("child finished its write")),
            // Child A: edit a.txt.
            (
                "target A".to_string(),
                turn_tool_use(
                    "w-a",
                    "write",
                    &json!({ "path": "a.txt", "content": "a rewritten by child\n" }).to_string(),
                ),
            ),
            // Child B: edit b.txt.
            (
                "target B".to_string(),
                turn_tool_use(
                    "w-b",
                    "write",
                    &json!({ "path": "b.txt", "content": "b rewritten by child\n" }).to_string(),
                ),
            ),
            // Parent's first turn: fan out.
            (
                "PARENT-FANOUT".to_string(),
                turn_tool_use(
                    "call-1",
                    "subagent",
                    &json!({ "tasks": [
                        { "agent": "writer", "task": "edit target A" },
                        { "agent": "writer", "task": "edit target B" }
                    ] })
                    .to_string(),
                ),
            ),
        ],
        turn_text("fallback"),
    );

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "PARENT-FANOUT: edit both files",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
        ])
        .current_dir(repo)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Both children's edits must have merged into the real tree — proving concurrent merge-back applied
    // both patches without one clobbering or spuriously failing the other.
    assert_eq!(
        std::fs::read_to_string(repo.join("a.txt")).unwrap(),
        "a rewritten by child\n"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("b.txt")).unwrap(),
        "b rewritten by child\n"
    );

    // Both clean merges ⇒ both worktrees removed; only the main checkout remains.
    let listing = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["worktree", "list"])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&listing.stdout).lines().count(),
        1,
        "clean-merged worktrees must be removed; git worktree list:\n{}",
        String::from_utf8_lossy(&listing.stdout)
    );
}
