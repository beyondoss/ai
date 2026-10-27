//! Tests never touch the developer's real home directory.
//!
//! A test that spawned the agent binary with a temp cwd but the developer's own `HOME` left a
//! session behind under `~/.<state-dir>/sessions/-tmp-.tmpXXXXXX` on every run — over a thousand of
//! them accumulated on one machine before anyone looked. Two guards keep that from coming back:
//!
//! - **One way to spawn the binary.** Every test builds the command with `common::agent_cmd()` (or
//!   `common::isolated_cmd(path)`, which the `run_cmd`/`serve_cmd` helpers use), which points
//!   `HOME` at a directory that doesn't exist and strips the variables that move state elsewhere.
//!   [`no_test_spawns_the_agent_binary_with_a_bare_command_new`] fails on any
//!   `Command::new(<agent binary>)` outside that helper, so a new test can't forget.
//! - **The helper really isolates.** [`a_spawned_run_never_writes_under_the_real_home`] runs a
//!   session-persisting `run` through the helper and checks that nothing for its cwd appeared under
//!   the real `$HOME/.beyond` or `$HOME/.claude`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

use common::{ISOLATED_HOME, agent_cmd, spawn_model_server, turn_text};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `Command::new(<agent binary>)` anywhere in the test tree except the helper itself. "The agent
/// binary" is the `env!` path, the shared `BIN` constant, or any name a file binds to either.
#[test]
fn no_test_spawns_the_agent_binary_with_a_bare_command_new() {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut files = Vec::new();
    rust_files(&tests, &mut files);
    let binding = regex::Regex::new(
        r#"(?:let|const|static)\s+(?:mut\s+)?(\w+)\s*(?::\s*&(?:'static\s+)?str\s*)?=\s*(?:env!\(\s*"CARGO_BIN_EXE_agent"\s*\)|(?:[\w:]*::)?BIN)\s*;"#,
    )
    .unwrap();
    let mut offenders = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        let code: String = text
            .lines()
            .map(|l| {
                if l.trim_start().starts_with("//") {
                    ""
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut names = vec!["bin".to_string(), "BIN".to_string()];
        names.extend(binding.captures_iter(&code).map(|c| c[1].to_string()));
        let alternatives = names
            .iter()
            .map(|n| format!(r"(?:[\w:]*::)?{}", regex::escape(n)))
            .chain([r#"env!\(\s*"CARGO_BIN_EXE_agent"\s*\)"#.to_string()])
            .collect::<Vec<_>>()
            .join("|");
        let spawn =
            regex::Regex::new(&format!(r"Command::new\(\s*(?:{alternatives})\s*\)")).unwrap();
        for m in spawn.find_iter(&code) {
            let line = code[..m.start()].matches('\n').count() + 1;
            let rel = file.strip_prefix(&tests).unwrap().display().to_string();
            offenders.push(format!("tests/{rel}:{line}: {}", m.as_str()));
        }
    }
    assert!(
        offenders.is_empty(),
        "spawn the agent binary with `common::agent_cmd()` (or `common::isolated_cmd(path)`), \
         which keeps it out of the real home directory:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn a_spawned_run_never_writes_under_the_real_home() {
    let Some(real_home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return;
    };
    assert_ne!(real_home, Path::new(ISOLATED_HOME));
    let cwd = tempfile::tempdir().unwrap();
    let canonical = cwd.path().canonicalize().unwrap();
    let encoded: String = canonical
        .to_str()
        .unwrap()
        .chars()
        .map(|c| if c == '/' { '-' } else { c })
        .collect();

    // A default-persistence `run`: with the developer's `HOME` it would create
    // `$HOME/.beyond/sessions/<encoded-cwd>/`.
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);
    let out = agent_cmd()
        .args(["run", "hi", "--gateway-url", &base, "--key", "bai_v1.test"])
        .args(["--model", "claude-test"])
        .current_dir(cwd.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() || stderr.contains(ISOLATED_HOME),
        "a forgotten HOME must fail at the isolated home, not anywhere else: {stderr}"
    );

    for root in [".beyond", ".claude"] {
        for sub in ["sessions", "projects"] {
            let leaked = real_home.join(root).join(sub).join(&encoded);
            assert!(
                !leaked.exists(),
                "a test wrote under the real home: {}",
                leaked.display()
            );
        }
    }
}
