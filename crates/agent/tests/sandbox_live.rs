//! **Live** end-to-end: the agent's filesystem tools driven against a real Firecracker VM.
//!
//! Everything else in this repo tests the seam on the host. This tests the thing the seam exists for:
//! `read`/`write`/`edit`/`ls`/`grep`/`find` acting on a filesystem that is genuinely not this
//! machine's, reached only by executing commands inside a microVM over vsock.
//!
//! ## Opt-in, and why
//!
//! Skipped unless `BEYOND_TEST_SANDBOX=<instance-id>` names a *running* instance. It needs a live
//! `instd`, a booted VM, and privilege to reach the admin channel — none of which a CI box or a
//! laptop has, and a test that silently passes when its subject is absent is worse than no test. When
//! the variable is set and the instance is unusable, this **fails** rather than skipping: at that
//! point the operator has asserted a sandbox exists.
//!
//! ```sh
//! BEYOND_TEST_SANDBOX=22f5evbtx520 cargo test -p beyond-ai-agent --test sandbox_live -- --nocapture
//! ```
//!
//! ## What it establishes
//!
//! That the six tools produce correct, model-visible results against a real guest — including the
//! cases the host-side differential suite can only simulate: a genuinely foreign filesystem, a
//! different `grep`/`find` implementation (the guest has GNU grep 3.11 and findutils 4.9.0, where
//! this host has ugrep and bfs), and no `rg` at all, so the POSIX fallback rung is the one under test.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use beyond_ai_agent::sandbox::InstdRunner;
use beyond_ai_agent::tools::exec::CommandRunner;
use beyond_ai_agent::tools::fs::FsBackend;
use beyond_ai_agent::tools::fs::shell::ShellFs;
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::{Value, json};

/// The instance to drive, or `None` to skip.
fn instance() -> Option<String> {
    std::env::var("BEYOND_TEST_SANDBOX")
        .ok()
        .filter(|s| !s.is_empty())
}

macro_rules! sandbox_or_skip {
    () => {
        match instance() {
            Some(id) => id,
            None => {
                eprintln!(
                    "skipping: set BEYOND_TEST_SANDBOX=<instance-id> to run live sandbox tests"
                );
                return;
            }
        }
    };
}

fn runner(id: &str) -> Arc<dyn CommandRunner> {
    Arc::new(InstdRunner::new(id).with_sudo())
}

/// A tool registry whose filesystem tools act inside the VM.
async fn sandbox_registry(id: &str) -> agent_core::ToolRegistry {
    let backend = ShellFs::connect(runner(id)).await;
    default_registry_with_config(&ToolConfig {
        fs_backend: Some(Arc::new(backend) as Arc<dyn FsBackend>),
        ..ToolConfig::new()
    })
}

async fn call(reg: &agent_core::ToolRegistry, tool: &str, input: Value) -> String {
    reg.get(tool)
        .expect("tool registered")
        .run(input)
        .await
        .unwrap_or_else(|e| panic!("{tool} failed inside the sandbox: {e}"))
        .text
}

/// Build a scratch tree **inside the guest**, using the tools themselves where possible.
async fn seed(reg: &agent_core::ToolRegistry, dir: &str) {
    // `write` creates parent directories, so this exercises `create_dir_all` in the guest too.
    call(
        reg,
        "write",
        json!({ "path": format!("{dir}/src/alpha.rs"), "content": "fn one() { NEEDLE }\nfn two() {}\n" }),
    )
    .await;
    call(
        reg,
        "write",
        json!({ "path": format!("{dir}/src/beta.rs"), "content": "// nothing here\n" }),
    )
    .await;
    call(
        reg,
        "write",
        json!({ "path": format!("{dir}/notes.txt"), "content": "NEEDLE in text\nand NEEDLE again\n" }),
    )
    .await;
}

#[tokio::test]
async fn the_sandbox_is_reachable_and_reports_what_it_has() {
    let id = sandbox_or_skip!();
    let backend = ShellFs::connect(runner(&id)).await;
    let caps = backend.capabilities();
    eprintln!("sandbox {id}: search engine = {:?}", caps.search_engine());
    // Not asserting *which* engine: the point of the capability probe is that the answer is whatever
    // the box actually has. What must hold is that the probe reached the box at all and the backend
    // reports itself as a non-local filesystem.
    assert!(matches!(
        backend.world(),
        beyond_ai_agent::tools::fs::PathWorld::Remote { .. }
    ));
    let _ = caps;
}

#[tokio::test]
async fn write_then_read_round_trips_through_the_vm() {
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;
    let dir = "/tmp/beyond-agent-live-rt";
    let body = "line one\nline two — em dash, ünïcödé, and a $dollar\n";

    call(
        &reg,
        "write",
        json!({ "path": format!("{dir}/deep/nested/out.txt"), "content": body }),
    )
    .await;
    let out = call(
        &reg,
        "read",
        json!({ "path": format!("{dir}/deep/nested/out.txt") }),
    )
    .await;

    // `read` renders `<lineno>\t<text>`, so assert on content rather than exact framing.
    assert!(
        out.contains("em dash, ünïcödé"),
        "content must survive the VM round trip: {out}"
    );
    assert!(
        out.contains("$dollar"),
        "a literal $ must not be expanded by any shell: {out}"
    );

    // And the file really is in the guest, not on this host.
    assert!(
        !std::path::Path::new(&format!("{dir}/deep/nested/out.txt")).exists(),
        "the file must NOT exist on the host — that would mean the tools never left this machine"
    );
}

#[tokio::test]
async fn a_path_of_shell_metacharacters_is_inert_inside_the_vm() {
    // The invariant that matters most: a model-supplied path is an argument, never syntax. If any
    // layer ever builds a command string, this test deletes files in a VM instead of failing.
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;
    let dir = "/tmp/beyond-agent-live-meta";
    let canary = format!("{dir}/canary.txt");
    call(
        &reg,
        "write",
        json!({ "path": &canary, "content": "alive\n" }),
    )
    .await;

    let nasty = format!("{dir}/'; rm -rf {dir} $(id) * #.txt");
    call(
        &reg,
        "write",
        json!({ "path": &nasty, "content": "NEEDLE quoted\n" }),
    )
    .await;

    let back = call(&reg, "read", json!({ "path": &nasty })).await;
    assert!(back.contains("NEEDLE quoted"), "{back}");
    let still = call(&reg, "read", json!({ "path": &canary })).await;
    assert!(
        still.contains("alive"),
        "the canary must survive — a deleted canary means the path was executed, not passed: {still}"
    );
}

#[tokio::test]
async fn grep_finds_matches_inside_the_vm() {
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;
    let dir = "/tmp/beyond-agent-live-grep";
    seed(&reg, dir).await;

    let out = call(&reg, "grep", json!({ "pattern": "NEEDLE", "path": dir })).await;
    assert!(out.contains("alpha.rs"), "{out}");
    assert!(out.contains("notes.txt"), "{out}");
    assert!(
        !out.contains("beta.rs"),
        "a non-matching file must not appear: {out}"
    );

    // Glob restriction, case folding and literal mode all go through the guest's own grep.
    let rs_only = call(
        &reg,
        "grep",
        json!({ "pattern": "NEEDLE", "path": dir, "glob": "*.rs" }),
    )
    .await;
    assert!(
        rs_only.contains("alpha.rs") && !rs_only.contains("notes.txt"),
        "{rs_only}"
    );

    let folded = call(
        &reg,
        "grep",
        json!({ "pattern": "needle", "path": dir, "ignore_case": true }),
    )
    .await;
    assert!(folded.contains("alpha.rs"), "{folded}");

    let none = call(
        &reg,
        "grep",
        json!({ "pattern": "zzz-absent", "path": dir }),
    )
    .await;
    assert!(none.starts_with("no matches for"), "{none}");
}

#[tokio::test]
async fn grep_context_lines_work_inside_the_vm() {
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;
    let dir = "/tmp/beyond-agent-live-ctx";
    call(
        &reg,
        "write",
        json!({ "path": format!("{dir}/ctx.txt"), "content": "before-line\nNEEDLE mid\nafter-line\n" }),
    )
    .await;
    let out = call(
        &reg,
        "grep",
        json!({ "pattern": "NEEDLE mid", "path": dir, "context": 1 }),
    )
    .await;
    assert!(out.contains("before-line"), "{out}");
    assert!(out.contains("after-line"), "{out}");
    // Context lines use the `-` separator, matches use `:` — the distinction must survive the guest's
    // grep, whose output format the parser reads.
    assert!(
        out.contains("- before-line") || out.contains("-1- "),
        "{out}"
    );
}

#[tokio::test]
async fn ls_and_find_work_inside_the_vm() {
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;
    let dir = "/tmp/beyond-agent-live-ls";
    seed(&reg, dir).await;

    let listing = call(&reg, "ls", json!({ "path": dir })).await;
    assert!(
        listing.contains("src/"),
        "a directory must carry its suffix: {listing}"
    );
    assert!(listing.contains("notes.txt"), "{listing}");

    let found = call(&reg, "find", json!({ "pattern": "*.rs", "path": dir })).await;
    assert!(
        found.contains("alpha.rs") && found.contains("beta.rs"),
        "{found}"
    );

    let dirs = call(&reg, "find", json!({ "pattern": "src", "path": dir })).await;
    assert!(
        dirs.contains("src/"),
        "find must match directories too — this is the ancestor-derivation path: {dirs}"
    );
}

#[tokio::test]
async fn edit_applies_inside_the_vm_and_the_guard_holds() {
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;
    let dir = "/tmp/beyond-agent-live-edit";
    let path = format!("{dir}/src.rs");
    call(
        &reg,
        "write",
        json!({ "path": &path, "content": "fn main() {\n    let quick = 1;\n}\n" }),
    )
    .await;

    let out = call(
        &reg,
        "edit",
        json!({ "path": &path, "old_string": "quick", "new_string": "slow" }),
    )
    .await;
    assert!(out.contains("1 replacement"), "{out}");

    let back = call(&reg, "read", json!({ "path": &path })).await;
    assert!(back.contains("let slow = 1;"), "{back}");
    assert!(!back.contains("quick"), "{back}");
}

#[tokio::test]
async fn errors_from_inside_the_vm_are_reported_not_swallowed() {
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;

    // A missing file must error. This is the regression that the host suite caught: a shell pipeline
    // returns its *last* command's status, so a failing `dd` piped into a succeeding `base64` made a
    // missing file read back as an empty one. Against a real guest it matters just as much.
    let err = reg
        .get("read")
        .unwrap()
        .run(json!({ "path": "/tmp/definitely-not-here-9f3a.txt" }))
        .await
        .expect_err("reading a missing file must fail, not return empty");
    let msg = err.to_string();
    assert!(
        msg.contains("definitely-not-here"),
        "the error must name the path: {msg}"
    );

    // A path that is not a directory must be distinguished from one that does not exist.
    let dir = "/tmp/beyond-agent-live-err";
    let f = format!("{dir}/plain.txt");
    call(&reg, "write", json!({ "path": &f, "content": "x\n" })).await;
    let e1 = reg
        .get("ls")
        .unwrap()
        .run(json!({ "path": &f }))
        .await
        .unwrap_err()
        .to_string();
    assert!(e1.contains("Not a directory"), "{e1}");
    let e2 = reg
        .get("ls")
        .unwrap()
        .run(json!({ "path": format!("{dir}/nope") }))
        .await
        .unwrap_err()
        .to_string();
    assert!(e2.contains("Path not found"), "{e2}");
}

#[tokio::test]
async fn the_model_sees_the_same_toolset_against_a_vm_as_it_does_locally() {
    // The contract, checked against the real thing rather than a stand-in: attaching a sandbox must
    // not change one byte of what is advertised, or every turn cold-misses the prompt cache and the
    // model is looking at a different toolset than the one it was tuned against.
    let id = sandbox_or_skip!();
    let remote = sandbox_registry(&id).await.definitions();
    let local = default_registry_with_config(&ToolConfig::new()).definitions();
    assert_eq!(
        serde_json::to_string(&remote).unwrap(),
        serde_json::to_string(&local).unwrap(),
        "the advertised toolset must be identical whether the tools act here or in a VM"
    );
}

#[tokio::test]
async fn bash_still_runs_on_the_host_not_in_the_vm() {
    // A deliberate, documented boundary of this phase: the `FsBackend` seam moves the *filesystem*
    // tools, and `bash` goes through `CommandRunner`, which the registry does not yet repoint. This
    // pins the asymmetry so it is a known state rather than a surprise — and so the test fails the
    // day someone wires `bash` through too, prompting them to update the docs with it.
    let id = sandbox_or_skip!();
    let reg = sandbox_registry(&id).await;
    let out = call(&reg, "bash", json!({ "command": "uname -n" })).await;
    let host = std::process::Command::new("uname")
        .arg("-n")
        .output()
        .unwrap();
    let host = String::from_utf8_lossy(&host.stdout).trim().to_string();
    assert!(
        out.contains(&host),
        "`bash` is expected to still run on the host in this phase; got {out:?}, host is {host:?}"
    );
    let _ = id;
}

// ---------------------------------------------------------------- the CLI

mod common;

/// Drive the **real binary** with `--sandbox`, against a scripted model, against a real VM.
///
/// The tool-level tests above prove the backend works; this proves the flag actually wires it in —
/// the difference between "the library can do this" and "a person can use this". It is also the exact
/// invocation the PR documents, so if the documented command stops working, this fails.
#[test]
fn the_sandbox_flag_makes_the_real_binary_read_files_from_inside_the_vm() {
    let Some(id) = instance() else {
        eprintln!("skipping: set BEYOND_TEST_SANDBOX=<instance-id>");
        return;
    };
    const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Seed a file inside the guest that does **not** exist on this host. If the agent can read it,
    // the tools genuinely ran in the VM; nothing on the host could have satisfied the call.
    let marker = "SANDBOX-ONLY-MARKER-7f2a";
    let guest_path = "/tmp/beyond-agent-cli-e2e/only-in-vm.txt";
    let seed = std::process::Command::new("sudo")
        .args([
            "-n", "instd", "instance", "exec", &id, "--", "/bin/sh", "-c",
        ])
        .arg(format!(
            "mkdir -p /tmp/beyond-agent-cli-e2e && printf '%s\\n' {marker} > {guest_path}"
        ))
        .output()
        .expect("seed the guest");
    assert!(seed.status.success(), "seeding failed: {seed:?}");
    assert!(
        !std::path::Path::new(guest_path).exists(),
        "the fixture must not exist on the host, or this test proves nothing"
    );

    // One scripted turn: call `read` on the guest-only path, then report what came back.
    let (base, _bodies) = common::spawn_model_server(vec![
        common::turn_tool_use("t1", "read", &json!({ "path": guest_path }).to_string()),
        common::turn_text("done"),
    ]);

    let out = common::run_cmd(BIN)
        .args([
            "run",
            "read the file",
            "--sandbox",
            &id,
            "--sandbox-sudo",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
            "--json",
        ])
        .output()
        .expect("run the agent");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&format!("sandbox {id}")),
        "the attach must be reported to the operator; stderr was: {stderr}"
    );
    assert!(
        stdout.contains(marker),
        "the tool result must carry the guest-only file's contents.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}
