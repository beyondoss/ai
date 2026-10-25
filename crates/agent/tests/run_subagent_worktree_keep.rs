//! `run` e2e: a preserved worktree — a failed child's only copy of its work — survives both a retry
//! of the same agent in this process and the startup sweep of the next one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

fn git(dir: &Path, args: &[&str]) {
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

fn repo() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    git(p, &["init", "--quiet", "-b", "main"]);
    git(p, &["config", "user.name", "t"]);
    git(p, &["config", "user.email", "t@t"]);
    std::fs::write(p.join("target.txt"), "original\n").unwrap();
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

fn run(dir: &Path, cache: &Path, script: Vec<String>) {
    let (base, _bodies) = spawn_model_server(script);
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

/// Every checkout under the cache whose `half.txt` holds the failed child's work.
fn kept_checkouts(cache: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let root = cache.join("beyond-agent/worktrees");
    for repo in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        for wt in std::fs::read_dir(repo.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            if wt.path().join("half.txt").exists() {
                found.push(wt.path());
            }
        }
    }
    found
}

#[test]
fn a_failed_childs_checkout_survives_a_retry_and_the_next_processs_sweep() {
    let dir = repo();
    let cache = tempfile::tempdir().unwrap();
    let schema = json!({"type": "object", "properties": {"done": {"type": "boolean"}}});
    run(
        dir.path(),
        cache.path(),
        vec![
            // A child asked for structured output that ends in prose has failed, and its worktree is
            // preserved rather than merged.
            turn_tool_use(
                "call-1",
                "subagent",
                &json!({"agent": "writer", "task": "step one", "output_schema": schema})
                    .to_string(),
            ),
            turn_tool_use(
                "call-2",
                "write",
                &json!({"path": "half.txt", "content": "half done\n"}).to_string(),
            ),
            turn_text("I wrote half of it."),
            // The parent retries the same agent: single mode, index 0 again.
            turn_tool_use(
                "call-3",
                "subagent",
                &json!({"agent": "writer", "task": "try again"}).to_string(),
            ),
            turn_text("nothing to do"),
            turn_text("Done."),
        ],
    );
    assert!(
        !dir.path().join("half.txt").exists(),
        "a failed child is not merged"
    );
    let kept = kept_checkouts(cache.path());
    assert_eq!(
        kept.len(),
        1,
        "the retry must not delete the preserved checkout"
    );

    // The next process sweeps at startup; the first one's PID is dead by now.
    run(dir.path(), cache.path(), vec![turn_text("hello")]);
    assert_eq!(
        kept_checkouts(cache.path()),
        kept,
        "the next process's sweep must keep a preserved checkout"
    );
}
