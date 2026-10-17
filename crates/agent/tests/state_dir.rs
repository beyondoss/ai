//! The agent's own state lives in `~/.beyond`, and `~/.claude` — Claude Code's directory — is never
//! written. Every command that persists something runs against a `HOME` whose `~/.claude` is stocked
//! the way a Claude Code user's is (a `settings.json` with permissions, hooks and env, credentials,
//! `sessions/<pid>.json`, per-project memory, a skill), and the whole `~/.claude` tree — every path,
//! its bytes, its mode and its mtime — must be identical afterwards.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

use common::mcp_oauth_fixture::{OAuthFixture, mcp_login};
use common::{BIN, run_cmd, spawn_model_server, turn_text};
use serde_json::{Value, json};

/// One entry of a tree snapshot: file bytes (`None` for a directory), mode and mtime.
type Snapshot = BTreeMap<PathBuf, (Option<Vec<u8>>, u32, SystemTime)>;

fn snapshot(root: &Path) -> Snapshot {
    fn walk(root: &Path, dir: &Path, out: &mut Snapshot) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let meta = std::fs::symlink_metadata(&path).unwrap();
            let rel = path.strip_prefix(root).unwrap().to_path_buf();
            let bytes = meta.is_file().then(|| std::fs::read(&path).unwrap());
            out.insert(
                rel,
                (bytes, meta.permissions().mode(), meta.modified().unwrap()),
            );
            if meta.is_dir() {
                walk(root, &path, out);
            }
        }
    }
    let mut out = Snapshot::new();
    walk(root, root, &mut out);
    out
}

fn write(path: &Path, body: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

/// A `~/.claude` as a Claude Code user (who also ran an older release of this agent) has it.
fn stock_claude_dir(claude: &Path) {
    write(
        &claude.join("settings.json"),
        &json!({
            "permissions": { "allow": ["Bash(ls:*)"], "deny": ["Read(.env)"] },
            "hooks": { "Stop": [{ "hooks": [{ "type": "command", "command": "notify-send done" }] }] },
            "env": { "CLAUDE_CODE_FLAG": "1" },
            "default_model": "legacy-model",
        })
        .to_string(),
    );
    write(
        &claude.join("auth.json"),
        r#"{"anthropic":{"type":"oauth","access":"legacy","refresh":"legacy","expires":4000000000000}}"#,
    );
    write(
        &claude.join("mcp_auth.json"),
        r#"{"legacy":{"client_id":"x"}}"#,
    );
    write(
        &claude.join("trusted-projects.json"),
        r#"{"trusted":["/legacy/project"],"untrusted":[]}"#,
    );
    write(&claude.join("models.json"), r#"{"legacy-model":{}}"#);
    write(&claude.join("sessions/41234.json"), r#"{"pid":41234}"#);
    write(&claude.join("sessions/41234.ab12.key"), "k");
    write(
        &claude.join("sessions/-old-project/1700000000000_old.jsonl"),
        "{}\n",
    );
    write(
        &claude.join("projects/-old-project/memory/MEMORY.md"),
        "- claude code memory\n",
    );
    write(
        &claude.join("skills/greet/SKILL.md"),
        "---\nname: greet\ndescription: say hi\n---\nbody\n",
    );
    write(&claude.join("CLAUDE.md"), "be kind\n");
}

/// A command against `home` with no override of where state lives.
fn agent(home: &Path) -> Command {
    let mut cmd = run_cmd(BIN);
    cmd.env("HOME", home)
        .env_remove("AI_AGENT_CONFIG_DIR")
        .env_remove("AI_AGENT_SESSION_DIR")
        .env_remove("AI_AGENT_MEMORY_URL")
        .stdin(Stdio::null());
    cmd
}

fn ok(cmd: &mut Command) -> String {
    let out = cmd.output().expect("spawn binary");
    assert!(
        out.status.success(),
        "command failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn every_state_write_lands_in_dot_beyond_and_dot_claude_is_never_touched() {
    let home = tempfile::tempdir().unwrap();
    let claude = home.path().join(".claude");
    let beyond = home.path().join(".beyond");
    stock_claude_dir(&claude);
    let before = snapshot(&claude);

    // `settings`: our typed store, in our own file — Claude Code's keys are nowhere near it.
    ok(agent(home.path()).args(["settings", "--model", "new-model"]));
    let settings = read_json(&beyond.join("settings.json"));
    assert_eq!(settings["default_model"], "new-model");
    for key in ["permissions", "hooks", "env"] {
        assert!(settings.get(key).is_none(), "{key} leaked: {settings}");
    }

    // `trust`.
    let project = tempfile::tempdir().unwrap();
    let project_dir = project.path().canonicalize().unwrap();
    ok(agent(home.path()).args(["trust", project_dir.to_str().unwrap()]));
    let trust = read_json(&beyond.join("trusted-projects.json"));
    assert!(
        trust["trusted"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p == project_dir.to_str().unwrap()),
        "{trust}"
    );

    // The credential store: the legacy `~/.claude/auth.json` is not read (a clean break), and a write
    // goes to `~/.beyond/auth.json` with owner-only permissions.
    let status = ok(agent(home.path()).args(["auth-status", "anthropic"]));
    assert!(status.starts_with("anthropic: logged_out"), "{status}");
    write(
        &beyond.join("auth.json"),
        r#"{"anthropic":{"type":"oauth","access":"a","refresh":"r","expires":4000000000000}}"#,
    );
    let out = ok(agent(home.path()).args(["logout", "anthropic"]));
    assert!(out.starts_with("logged out:"), "{out}");
    let auth = read_json(&beyond.join("auth.json"));
    assert!(auth.get("anthropic").is_none(), "{auth}");
    let mode = std::fs::metadata(beyond.join("auth.json"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o077, 0, "auth.json must be owner-only: {mode:o}");

    // A `run` creates its session under `~/.beyond/sessions/<encoded-cwd>/`.
    let (base, _bodies) = spawn_model_server(vec![turn_text("hello")]);
    ok(agent(home.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(&project_dir));
    let encoded: String = project_dir
        .to_str()
        .unwrap()
        .chars()
        .map(|c| if c == '/' { '-' } else { c })
        .collect();
    let repo = beyond.join("sessions").join(&encoded);
    let sessions: Vec<_> = std::fs::read_dir(&repo)
        .unwrap_or_else(|e| panic!("{}: {e}", repo.display()))
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .collect();
    assert_eq!(sessions.len(), 1, "one new session in {}", repo.display());

    // An MCP OAuth login, through the real flow, stores its credential in `~/.beyond/mcp_auth.json`.
    let fixture = OAuthFixture::spawn(3600);
    let mut settings = read_json(&beyond.join("settings.json"));
    settings["mcp_servers"] =
        json!([{ "name": "protected", "transport": "http", "url": fixture.url, "headers": {} }]);
    write(&beyond.join("settings.json"), &settings.to_string());
    mcp_login(home.path(), "protected");
    let mcp_auth = read_json(&beyond.join("mcp_auth.json"));
    assert!(mcp_auth.get("protected").is_some(), "{mcp_auth}");
    assert!(
        mcp_auth.get("legacy").is_none(),
        "nothing is copied from ~/.claude"
    );
    let mode = std::fs::metadata(beyond.join("mcp_auth.json"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(
        mode & 0o077,
        0,
        "mcp_auth.json must be owner-only: {mode:o}"
    );

    // And through all of it, not one byte, mode or mtime under `~/.claude` changed, and nothing was
    // created or removed there.
    let after = snapshot(&claude);
    assert_eq!(
        before.keys().collect::<Vec<_>>(),
        after.keys().collect::<Vec<_>>(),
        "a path under ~/.claude was created or removed"
    );
    for (path, entry) in &before {
        assert!(
            after.get(path) == Some(entry),
            "{} under ~/.claude was modified",
            path.display()
        );
    }
}

#[test]
fn ai_agent_config_dir_moves_sessions_too() {
    let home = tempfile::tempdir().unwrap();
    let config = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hello")]);
    ok(agent(home.path())
        .env("AI_AGENT_CONFIG_DIR", config.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(project.path()));
    assert!(config.path().join("sessions").is_dir());
    assert!(!home.path().join(".beyond").exists());
    assert!(!home.path().join(".claude").exists());
}
