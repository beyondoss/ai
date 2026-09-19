//! What the four *search-and-list* tools must never ask this host once their backend is remote.
//!
//! `fs_backend_parity.rs` proves the two backends agree about a tree they can both see. This file
//! proves the opposite thing: that when the tree is **somewhere else**, nothing about the answer is
//! decided here. Every host-side question in this path — expanding `~`, "is the root a directory",
//! "is the root inside a git repository" — has the same shape of failure, and it is the worst one:
//! the syscall *succeeds* and returns a confident, plausible answer about the wrong machine.
//!
//! So the assertions are deliberately adversarial. The host and the target are made to disagree (a
//! `$HOME` that isn't the sandbox's, a root that is inside a git repo here and not there, a directory
//! that exists there and not here), and the test asserts the **target's** answer won.
//!
//! The target is a recorder: every command the backend would have run is captured with its argv and
//! `cwd`, and canned output is fed back. That is the only way to see the decision — it is expressed
//! entirely in the flags and paths the backend chooses.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use beyond_ai_agent::tools::exec::{CommandRunner, ExecResult};
use beyond_ai_agent::tools::fs::FsBackend;
use beyond_ai_agent::tools::fs::shell::{Capabilities, ShellFs};
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::{Value, json};

/// The sandbox's home. Nothing like any real `$HOME`, so an expansion against the wrong one is
/// unmistakable in an assertion failure.
const SANDBOX_HOME: &str = "/sbx";

/// One command the backend issued.
#[derive(Clone, Debug)]
struct Call {
    program: String,
    args: Vec<String>,
    cwd: Option<String>,
}

impl Call {
    /// Whether any part of the invocation mentions `needle` — the question every assertion here asks,
    /// since a path can ride in argv or in `cwd` depending on the tool.
    fn mentions(&self, needle: &str) -> bool {
        self.args.iter().any(|a| a.contains(needle))
            || self.cwd.as_deref().is_some_and(|c| c.contains(needle))
    }
}

/// A `CommandRunner` that runs nothing: it records the invocation and replays a canned answer.
///
/// Canned rather than real because the point is *which command the backend decided to run*, and a
/// real run on this host would answer with this host's filesystem — the very thing under test.
struct Recorder {
    calls: Mutex<Vec<Call>>,
    #[allow(clippy::type_complexity)]
    reply: Box<dyn Fn(&str, &[String]) -> ExecResult + Send + Sync>,
}

impl Recorder {
    fn new(reply: impl Fn(&str, &[String]) -> ExecResult + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            reply: Box::new(reply),
        })
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl CommandRunner for Recorder {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        _timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.calls.lock().unwrap().push(Call {
            program: program.to_string(),
            args: args.to_vec(),
            cwd: cwd.map(str::to_string),
        });
        Ok((self.reply)(program, args))
    }
}

fn ok(stdout: &str) -> ExecResult {
    ExecResult {
        code: Some(0),
        signal: None,
        stdout: stdout.to_string(),
        stderr: String::new(),
        timed_out: false,
        truncated: false,
    }
}

fn exit(code: i32) -> ExecResult {
    ExecResult {
        code: Some(code),
        ..ok("")
    }
}

/// Everything present, so the backend takes the ripgrep rung — the only one whose flags carry a
/// git-repo decision at all.
fn full_caps() -> Capabilities {
    Capabilities {
        rg: true,
        grep_null: true,
        find_printf: true,
        stdin: true,
    }
}

/// A registry whose six filesystem tools *and* `bash` point at `runner`, with the target's home
/// declared — the production pairing, built through the production constructor.
fn registry_over(
    runner: Arc<Recorder>,
    root: &str,
) -> (agent_core::ToolRegistry, Arc<dyn FsBackend>) {
    let backend: Arc<dyn FsBackend> =
        Arc::new(ShellFs::with_capabilities(runner.clone(), full_caps()).with_home(SANDBOX_HOME));
    let reg = default_registry_with_config(&ToolConfig {
        root: root.into(),
        fs_backend: Some(backend.clone()),
        command_runner: Some(runner),
        ..ToolConfig::new()
    });
    (reg, backend)
}

/// Run one tool against a recorder and hand back everything the target was asked to do.
async fn calls_for(
    tool: &str,
    input: Value,
    reply: impl Fn(&str, &[String]) -> ExecResult + Send + Sync + 'static,
) -> Vec<Call> {
    let runner = Recorder::new(reply);
    let (reg, _backend) = registry_over(runner.clone(), "");
    let _ = reg.get(tool).expect("tool registered").run(input).await;
    runner.calls()
}

/// This host's `$HOME`, which must never appear in anything sent to the target.
fn host_home() -> String {
    std::env::var("HOME").unwrap_or_default()
}

// ---- `~` expands against the target's home, in all four tools ----------------------------------

/// The one assertion, applied four times: the sandbox's home reached the target and this host's did
/// not. A tool that skipped `resolve_against_in` would send `$HOME/proj` — a real directory here,
/// owned by whoever is running the replica.
async fn tilde_expands_for(tool: &str, input: Value) {
    let calls = calls_for(tool, input, |program, _| match program {
        // `rg --files` and `rg` search output: one hit so `find`/`grep` reach their rendering.
        "rg" => ok(""),
        _ => ok(""),
    })
    .await;
    assert!(
        !calls.is_empty(),
        "{tool} issued no command at all, so nothing was resolved"
    );
    assert!(
        calls.iter().any(|c| c.mentions("/sbx/proj")),
        "{tool} never mentioned the target-home-expanded path.\n{calls:#?}"
    );
    let host = format!("{}/proj", host_home());
    assert!(
        host_home().is_empty() || !calls.iter().any(|c| c.mentions(&host)),
        "{tool} expanded `~` against this host's home ({host}).\n{calls:#?}"
    );
}

#[tokio::test]
async fn grep_expands_a_tilde_path_against_the_targets_home() {
    tilde_expands_for("grep", json!({ "pattern": "x", "path": "~/proj" })).await;
}

#[tokio::test]
async fn find_expands_a_tilde_path_against_the_targets_home() {
    tilde_expands_for("find", json!({ "pattern": "*.rs", "path": "~/proj" })).await;
}

#[tokio::test]
async fn ls_expands_a_tilde_path_against_the_targets_home() {
    tilde_expands_for("ls", json!({ "path": "~/proj" })).await;
}

#[tokio::test]
async fn bash_expands_a_tilde_cwd_against_the_targets_home() {
    tilde_expands_for("bash", json!({ "command": "true", "cwd": "~/proj" })).await;
}

#[tokio::test]
async fn a_tilde_path_is_left_alone_when_the_targets_home_is_unknown() {
    // No `with_home`: guessing would produce a plausible path for the wrong user, so the `~` survives
    // and the target's own shell (or its absence of expansion) decides.
    let runner = Recorder::new(|_, _| ok(""));
    let backend: Arc<dyn FsBackend> =
        Arc::new(ShellFs::with_capabilities(runner.clone(), full_caps()));
    let reg = default_registry_with_config(&ToolConfig {
        fs_backend: Some(backend),
        command_runner: Some(runner.clone()),
        ..ToolConfig::new()
    });
    let _ = reg
        .get("ls")
        .unwrap()
        .run(json!({ "path": "~/proj" }))
        .await;
    let calls = runner.calls();
    assert!(
        calls.iter().any(|c| c.mentions("~/proj")),
        "an unknown home must leave `~` untouched, not expand it against this host.\n{calls:#?}"
    );
    assert!(
        host_home().is_empty()
            || !calls
                .iter()
                .any(|c| c.mentions(&format!("{}/proj", host_home()))),
        "`~` was expanded against this host's home.\n{calls:#?}"
    );
}

// ---- the git-repo decision is the target's -----------------------------------------------------

/// `--no-require-git` decides whether `rg` honors `.gitignore` at all, and the host and the target
/// can easily disagree: this checkout *is* a git repository, and a sandbox mounted at the same path
/// need not be. The host's answer here would be "inside a repo" (no flag); the target says otherwise.
#[tokio::test]
async fn the_git_repo_answer_comes_from_the_target_not_this_host() {
    let repo_root = env!("CARGO_MANIFEST_DIR"); // inside a git repo on *this* host
    let calls = calls_for(
        "grep",
        json!({ "pattern": "x", "path": repo_root }),
        // The target says "no .git anywhere" (exit 1) for the same absolute path.
        |program, args| {
            if program == "sh" && args.iter().any(|a| a.contains(".git")) {
                exit(1)
            } else {
                ok("")
            }
        },
    )
    .await;
    let rg = calls
        .iter()
        .find(|c| c.program == "rg")
        .expect("the ripgrep rung must have run a search");
    assert!(
        rg.args.iter().any(|a| a == "--no-require-git"),
        "the target said this root is not in a git repo; the flag must follow that, not this \
         host's `.git`.\n{rg:#?}"
    );
}

#[tokio::test]
async fn a_target_that_reports_a_git_repo_drops_the_no_require_git_flag() {
    let dir = tempfile::tempdir().unwrap(); // *not* a git repo on this host
    let calls = calls_for(
        "grep",
        json!({ "pattern": "x", "path": dir.path().to_str().unwrap() }),
        |program, args| {
            if program == "sh" && args.iter().any(|a| a.contains(".git")) {
                exit(0) // the target *is* in a repo
            } else {
                ok("")
            }
        },
    )
    .await;
    let rg = calls.iter().find(|c| c.program == "rg").unwrap();
    assert!(
        !rg.args.iter().any(|a| a == "--no-require-git"),
        "the target reported a git repo, so the flag must be absent.\n{rg:#?}"
    );
}

#[tokio::test]
async fn the_git_repo_answer_is_asked_once_per_root_not_once_per_search() {
    // A round trip per `grep` would be a permanent tax on the most-used search tool, for an answer
    // that changes only if someone runs `git init` mid-session.
    let runner = Recorder::new(|_, _| ok(""));
    let (reg, _backend) = registry_over(runner.clone(), "");
    let grep = reg.get("grep").unwrap();
    for _ in 0..3 {
        let _ = grep
            .run(json!({ "pattern": "x", "path": "/sbx/proj" }))
            .await;
    }
    let probes = runner
        .calls()
        .into_iter()
        .filter(|c| c.program == "sh" && c.args.iter().any(|a| a.contains(".git")))
        .count();
    assert_eq!(probes, 1, "the git-repo probe must be cached per root");
}

// ---- `format_path` renders against a root this host has never seen -----------------------------

/// A hit's path is rendered relative to the search root — which requires knowing the root is a
/// directory. Asking `is_dir()` here answers `false` for any sandbox-only path, and every hit would
/// silently collapse to a bare basename: `sub/a.rs` and `other/a.rs` both become `a.rs`, which is not
/// a cosmetic difference — the model uses these paths to `read` and `edit`.
#[tokio::test]
async fn grep_renders_hits_relative_to_a_root_that_exists_only_on_the_target() {
    let runner = Recorder::new(|program, _| match program {
        "rg" => ok("/sbx/proj/sub/a.rs\u{0}12:hit here\n"),
        _ => ok(""),
    });
    let (reg, _backend) = registry_over(runner, "");
    let out = reg
        .get("grep")
        .unwrap()
        .run(json!({ "pattern": "hit", "path": "/sbx/proj" }))
        .await
        .expect("grep succeeds")
        .text;
    assert!(
        out.contains("sub/a.rs:12: hit here"),
        "expected a root-relative path, got:\n{out}"
    );
    assert!(
        !out.contains("/sbx/proj/sub/a.rs"),
        "the root prefix must not be repeated on every line:\n{out}"
    );
}

#[tokio::test]
async fn grep_on_a_single_target_file_still_renders_just_the_basename() {
    // The other half of the same decision: when the root *is* the file being searched, stripping it
    // would leave an empty string, so the basename is what the caller sees. Derived from the hits
    // rather than from a host `stat`, and this pins that the derivation agrees.
    let runner = Recorder::new(|program, _| match program {
        "rg" => ok("/sbx/proj/a.rs\u{0}3:only file\n"),
        _ => ok(""),
    });
    let (reg, _backend) = registry_over(runner, "");
    let out = reg
        .get("grep")
        .unwrap()
        .run(json!({ "pattern": "only", "path": "/sbx/proj/a.rs" }))
        .await
        .expect("grep succeeds")
        .text;
    assert!(out.starts_with("a.rs:3: "), "got:\n{out}");
}

#[tokio::test]
async fn find_renders_matches_relative_to_a_root_that_exists_only_on_the_target() {
    let runner = Recorder::new(|program, args| match program {
        // The one-shot stat: writable, directory, size, mtime — the root is a directory *there*.
        "sh" if args.iter().any(|a| a.contains("/sbx/proj")) => ok("w\td\t0\t0\n"),
        "rg" => ok("/sbx/proj/sub/a.rs\u{0}"),
        _ => ok(""),
    });
    let (reg, _backend) = registry_over(runner, "");
    let out = reg
        .get("find")
        .unwrap()
        .run(json!({ "pattern": "*.rs", "path": "/sbx/proj" }))
        .await
        .expect("find succeeds")
        .text;
    assert!(
        out.lines().any(|l| l == "sub/a.rs"),
        "expected a root-relative match, got:\n{out}"
    );
}
