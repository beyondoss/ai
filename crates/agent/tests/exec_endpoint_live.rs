//! **Live** end-to-end: the real binary driving its whole toolset against a real remote target.
//!
//! Everything else tests the protocol against a mock that runs commands on this machine. That proves
//! wiring; it proves nothing about a genuinely separate, isolated filesystem. This does.
//!
//! ## Vendor-neutral by construction
//!
//! The target is whatever you point it at — there is no provider knowledge here, only the two flags
//! the agent actually ships:
//!
//! ```sh
//! # any transparent transport (docker, ssh, kubectl, podman …)
//! AGENT_LIVE_EXEC_CMD='docker exec my-container {}' \
//!   cargo test -p beyond-ai-agent --test exec_endpoint_live -- --nocapture
//!
//! # any HTTP endpoint speaking the exec protocol
//! AGENT_LIVE_EXEC_URL=https://exec.example/run AGENT_LIVE_EXEC_HEADER='Authorization: Bearer …' \
//!   cargo test -p beyond-ai-agent --test exec_endpoint_live -- --nocapture
//! ```
//!
//! `AGENT_LIVE_WORKDIR` names a directory that exists **on the target and not on this host** (default
//! `/work`) — that asymmetry is what makes the result meaningful rather than a local no-op.
//!
//! **This runs in CI.** The `exec-live` shard stands up an Alpine container and points the tests at
//! it. Alpine is deliberate: it is busybox, whose `grep` has no `-Z`/`-I`/`--include` and whose
//! `find` has no `-printf`, so it exercises the hardest capability rung — and it is what a large
//! share of real sandboxes actually are. Running against a GNU-flavoured image would quietly skip the
//! code path most likely to be wrong.
//!
//! Skipped when neither variable is set, so a local `cargo test` needs no infrastructure. When one
//! *is* set it fails rather than skips: at that point a target has been asserted to exist, and a
//! silent pass would be worse than no test.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use serde_json::json;

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

/// The exec flags to pass through, or `None` to skip.
fn target() -> Option<Vec<String>> {
    if let Ok(url) = std::env::var("AGENT_LIVE_EXEC_URL")
        && !url.is_empty()
    {
        let mut v = vec!["--exec-url".to_string(), url];
        if let Ok(h) = std::env::var("AGENT_LIVE_EXEC_HEADER")
            && !h.is_empty()
        {
            v.push("--exec-header".to_string());
            v.push(h);
        }
        return Some(v);
    }
    if let Ok(cmd) = std::env::var("AGENT_LIVE_EXEC_CMD")
        && !cmd.is_empty()
    {
        return Some(vec!["--exec-cmd".to_string(), cmd]);
    }
    None
}

fn workdir() -> String {
    std::env::var("AGENT_LIVE_WORKDIR").unwrap_or_else(|_| "/work".to_string())
}

macro_rules! target_or_skip {
    () => {
        match target() {
            Some(t) => t,
            None => {
                eprintln!(
                    "skipping: set AGENT_LIVE_EXEC_CMD or AGENT_LIVE_EXEC_URL to run live tests"
                );
                return;
            }
        }
    };
}

/// Drive the real binary for one scripted turn and return `(stdout, stderr)`.
fn run_agent(flags: &[String], turns: Vec<String>) -> (String, String) {
    let (base, _bodies) = common::spawn_model_server(turns);
    let mut cmd = common::run_cmd(BIN);
    cmd.args(["run", "do the task"]);
    cmd.args(flags);
    cmd.args([
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--max-steps",
        "6",
        "--no-session-persistence",
        "--json",
    ]);
    let out = cmd.output().expect("run the agent");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn the_tools_read_a_file_that_exists_only_on_the_target() {
    // The whole point. `AGENT_LIVE_WORKDIR` must not exist on this host, so a passing read cannot
    // have been satisfied locally.
    let flags = target_or_skip!();
    let dir = workdir();
    assert!(
        !std::path::Path::new(&dir).exists(),
        "{dir} exists on this host too, so this test would prove nothing — point \
         AGENT_LIVE_WORKDIR at a path that only exists on the target"
    );

    let path = format!("{dir}/marker.txt");
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use("t1", "read", &json!({ "path": &path }).to_string()),
            common::turn_text("done"),
        ],
    );
    assert!(
        stderr.contains("remote exec endpoint"),
        "the agent should report that it attached; stderr: {stderr}"
    );
    assert!(
        stdout.contains("ONLY"),
        "expected the target-only marker in the tool result.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

#[test]
fn grep_searches_the_targets_filesystem() {
    let flags = target_or_skip!();
    let dir = workdir();
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "grep",
                &json!({ "pattern": "NEEDLE", "path": &dir }).to_string(),
            ),
            common::turn_text("done"),
        ],
    );
    assert!(
        stdout.contains("a.rs"),
        "grep should find the seeded match on the target.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

#[test]
fn bash_runs_on_the_target_not_this_host() {
    // The hole #41 shipped. `bash` must reach the same machine as the filesystem tools, proven by a
    // hostname the host cannot produce.
    let flags = target_or_skip!();
    let (stdout, _stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "bash",
                &json!({ "command": "uname -n; cat /etc/hostname 2>/dev/null || true" })
                    .to_string(),
            ),
            common::turn_text("done"),
        ],
    );
    let host = String::from_utf8_lossy(
        &std::process::Command::new("uname")
            .arg("-n")
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .to_string();
    assert!(
        !stdout.contains(&host),
        "`bash` reported this host's name ({host}) — it ran locally, not on the target.\n{stdout}"
    );
}

#[test]
fn a_write_past_the_argv_limit_round_trips_through_the_target() {
    // A remote write used to ship the whole file as one argv entry, which Linux caps at 128 KiB, so
    // anything over ~96 KiB failed. CI's `docker exec` has no `-i` — the far side gets no stdin, the
    // probe sees that — so this runs the chunked fallback on busybox, the path most likely to trip on
    // a shell difference. Behind `ssh` or `docker exec -i` it runs the stdin path instead.
    use sha2::Digest as _;
    let flags = target_or_skip!();
    let dir = workdir();
    let path = format!("{dir}/big-written-by-agent.txt");
    let line = "chunk boundary check — ünïcödé, a $dollar and a 'quote'\n";
    // Unique per run, so a file left by an earlier run cannot satisfy the digest check.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let content = format!("run {nonce}\n{}", line.repeat(200 * 1024 / line.len() + 1));
    let digest = hex::encode(sha2::Sha256::digest(content.as_bytes()));
    // The check runs on the target, whose shell is POSIX `sh` — Alpine has no `/bin/bash`.
    let flags = [flags, vec!["--bash-shell-path".into(), "/bin/sh".into()]].concat();
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "write",
                &json!({ "path": &path, "content": &content }).to_string(),
            ),
            common::turn_tool_use(
                "t2",
                "bash",
                &json!({ "command": format!("sha256sum < {path}; ls {dir}") }).to_string(),
            ),
            common::turn_text("done"),
        ],
    );
    // The tail only: the head of stdout echoes the 200 KiB `write` input back.
    let tail: String = {
        let chars: Vec<char> = stdout.chars().collect();
        chars[chars.len().saturating_sub(2000)..].iter().collect()
    };
    assert!(
        stdout.contains(&digest),
        "the {} bytes written must be the bytes on the target.\n--- stdout (tail) ---\n{tail}\n\
         --- stderr ---\n{stderr}",
        content.len()
    );
    assert!(
        !stdout.contains(".tmp."),
        "a completed write must not leave its temp file behind.\n--- stdout (tail) ---\n{tail}"
    );
}

#[test]
fn write_then_read_round_trips_through_the_target() {
    let flags = target_or_skip!();
    let dir = workdir();
    let path = format!("{dir}/written-by-agent.txt");
    let body = "round trip — ünïcödé and a $dollar";
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "write",
                &json!({ "path": &path, "content": format!("{body}\n") }).to_string(),
            ),
            common::turn_tool_use("t2", "read", &json!({ "path": &path }).to_string()),
            common::turn_text("done"),
        ],
    );
    assert!(
        stdout.contains("ünïcödé") && stdout.contains("$dollar"),
        "content must survive the round trip verbatim.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        !std::path::Path::new(&path).exists(),
        "the file must not appear on this host — that would mean the write never left"
    );
}
