//! `run` e2e: merge-back into a parent whose working tree is not clean, and from a child that
//! commits. Both used to lose the child's work: `git apply --3way` (which implies `--index`)
//! refused any file the parent had edited without staging, and a delta diffed against the child's
//! moved `HEAD` came back empty.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::Path;
use std::process::{Command, Stdio};

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A repo with an eight-line `target.txt` and a worktree-isolated `writer` holding `tools`.
fn repo(tools: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    git(p, &["init", "--quiet", "-b", "main"]);
    git(p, &["config", "user.name", "t"]);
    git(p, &["config", "user.email", "t@t"]);
    std::fs::write(p.join("target.txt"), "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\n").unwrap();
    let agents = p.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("writer.md"),
        format!(
            "---\nname: writer\ndescription: edits files\ntools: {tools}\nisolation: worktree\n---\nYou are writer.\n"
        ),
    )
    .unwrap();
    git(p, &["add", "-A"]);
    git(p, &["commit", "--quiet", "-m", "init"]);
    dir
}

fn run(dir: &Path, cache: &Path, base: &str) {
    let out = run_cmd(env!("CARGO_BIN_EXE_agent"))
        .args([
            "run",
            "go",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
        ])
        .env("XDG_CACHE_HOME", cache)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn a_child_edit_merges_into_a_file_the_parent_edited_without_staging() {
    let dir = repo("read,edit,write");
    let cache = tempfile::tempdir().unwrap();
    // The parent's own uncommitted edit, on line 1, and a file it created and never added.
    std::fs::write(
        dir.path().join("target.txt"),
        "PARENT-WIP\nl2\nl3\nl4\nl5\nl6\nl7\nl8\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("fresh.rs"), "fn main() { bug(); }\n").unwrap();
    let index_before = std::fs::read(dir.path().join(".git/index")).unwrap();

    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({"agent": "writer", "task": "change line 8 and fix fresh.rs"}).to_string(),
        ),
        turn_tool_use(
            "call-2",
            "edit",
            &json!({"path": "target.txt", "old_string": "l8", "new_string": "CHILD-EDIT"})
                .to_string(),
        ),
        turn_tool_use(
            "call-3",
            "edit",
            &json!({"path": "fresh.rs", "old_string": "bug()", "new_string": "fixed()"})
                .to_string(),
        ),
        turn_text("Changed line 8 and fixed fresh.rs."),
        turn_text("Done."),
    ]);
    run(dir.path(), cache.path(), &base);

    assert_eq!(
        std::fs::read_to_string(dir.path().join("target.txt")).unwrap(),
        "PARENT-WIP\nl2\nl3\nl4\nl5\nl6\nl7\nCHILD-EDIT\n",
        "both the parent's WIP and the child's edit must be in the tree"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("fresh.rs")).unwrap(),
        "fn main() { fixed(); }\n"
    );
    assert_eq!(
        std::fs::read(dir.path().join(".git/index")).unwrap(),
        index_before,
        "merge-back must never stage anything in the user's index"
    );
    assert_eq!(git(dir.path(), &["worktree", "list"]).lines().count(), 1);
}

#[test]
fn a_child_that_commits_its_work_still_merges_it_back() {
    let dir = repo("read,edit,write,bash");
    let cache = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({"agent": "writer", "task": "add a file and commit"}).to_string(),
        ),
        turn_tool_use(
            "call-2",
            "bash",
            &json!({"command": "echo hello > new.txt && git add -A && git -c user.name=c -c user.email=c@c commit -qm child-work"}).to_string(),
        ),
        turn_text("Added new.txt and committed."),
        turn_text("Done."),
    ]);
    run(dir.path(), cache.path(), &base);

    assert_eq!(
        std::fs::read_to_string(dir.path().join("new.txt")).unwrap(),
        "hello\n",
        "a committed change must merge back, not be dropped as an empty delta"
    );
    assert_eq!(git(dir.path(), &["worktree", "list"]).lines().count(), 1);
}
