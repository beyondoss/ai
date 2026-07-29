//! Multi-tenancy: two sessions, two sandboxes, no leakage between them.
//!
//! A server that multiplexes tenants must give each one its own exec target, and **must not** let one
//! tenant's tools reach another's filesystem. This file is the proof, and it is deliberately adversarial
//! about the specific way that could go wrong here.
//!
//! ## The hazard being tested
//!
//! `serve` rebuilds its tool registry only when the model or thinking level changes. A `switch_session`
//! between two sessions on the *same model* does **not** rebuild it. So a target bound at
//! registry-construction time would leave the newly-switched-to session still talking to the previous
//! session's machine — one tenant's `read`, `write` and `bash` operating inside another tenant's
//! sandbox. That is a cross-tenant data breach, not a stale-config bug.
//!
//! [`ExecCell`] closes it by construction: the tools hold the cell and read it on every call, so a
//! re-point takes effect immediately and a skipped rebuild cannot miss it. The tests below assert that
//! with a registry that is built **exactly once** and never rebuilt — the pessimal case.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use beyond_ai_agent::exec_endpoint::{ExecCell, ExecTarget, TemplateRunner};
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::{Value, json};

/// A stand-in for one tenant's sandbox: a directory only that tenant's target can reach.
///
/// `env {}` is the transport — it runs whatever follows, exactly as `docker exec ctr` or `ssh host --`
/// do — and the *cwd* is what makes each target distinct, so a call landing on the wrong target reads
/// the wrong tenant's files. That is the failure this file is looking for.
async fn tenant_target(dir: &std::path::Path) -> ExecTarget {
    let template = format!("env -C {} {{}}", dir.display());
    ExecTarget::over(Arc::new(TemplateRunner::parse(&template).unwrap())).await
}

fn tenant_dir(name: &str, secret: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("secret.txt"), format!("{secret}\n")).unwrap();
    std::fs::write(dir.path().join("who.txt"), format!("{name}\n")).unwrap();
    dir
}

/// One registry over the cell, built **once**. Nothing below ever rebuilds it — that is the point.
fn registry_over(cell: &ExecCell) -> agent_core::ToolRegistry {
    default_registry_with_config(&ToolConfig {
        fs_backend: Some(cell.backend()),
        command_runner: Some(cell.runner()),
        ..ToolConfig::new()
    })
}

async fn call(reg: &agent_core::ToolRegistry, tool: &str, input: Value) -> String {
    reg.get(tool)
        .expect("tool registered")
        .run(input)
        .await
        .unwrap_or_else(|e| panic!("{tool}: {e}"))
        .text
}

#[tokio::test]
async fn switching_sessions_repoints_every_tool_without_rebuilding_the_registry() {
    let a = tenant_dir("tenant-a", "SECRET-AAAA");
    let b = tenant_dir("tenant-b", "SECRET-BBBB");
    let cell = ExecCell::new();
    // Built once, before either target exists. If any tool captured its backend at construction time,
    // every assertion below would read the wrong tenant.
    let reg = registry_over(&cell);

    cell.set(Some(tenant_target(a.path()).await));
    let a_read = call(&reg, "read", json!({ "path": "secret.txt" })).await;
    let a_bash = call(&reg, "bash", json!({ "command": "cat secret.txt" })).await;
    assert!(a_read.contains("SECRET-AAAA"), "{a_read}");
    assert!(a_bash.contains("SECRET-AAAA"), "{a_bash}");
    assert!(!a_read.contains("SECRET-BBBB") && !a_bash.contains("SECRET-BBBB"));

    // The switch a same-model `switch_session` performs — and nothing else.
    cell.set(Some(tenant_target(b.path()).await));
    let b_read = call(&reg, "read", json!({ "path": "secret.txt" })).await;
    let b_bash = call(&reg, "bash", json!({ "command": "cat secret.txt" })).await;
    assert!(
        b_read.contains("SECRET-BBBB"),
        "after the switch, `read` still reached the previous tenant: {b_read}"
    );
    assert!(
        b_bash.contains("SECRET-BBBB"),
        "after the switch, `bash` still reached the previous tenant: {b_bash}"
    );
    assert!(
        !b_read.contains("SECRET-AAAA") && !b_bash.contains("SECRET-AAAA"),
        "tenant A's secret leaked into tenant B's session"
    );
}

#[tokio::test]
async fn every_tool_follows_the_switch_not_just_the_ones_that_are_easy() {
    // A partial re-point is the dangerous outcome: if `grep` followed the switch but `write` did not,
    // one tenant would be *writing into* another's sandbox. Each tool is checked individually so a
    // failure names the one that stayed behind.
    let a = tenant_dir("tenant-a", "SECRET-AAAA");
    let b = tenant_dir("tenant-b", "SECRET-BBBB");
    let cell = ExecCell::new();
    let reg = registry_over(&cell);

    cell.set(Some(tenant_target(b.path()).await));

    let ls = call(&reg, "ls", json!({ "path": "." })).await;
    assert!(ls.contains("secret.txt"), "ls: {ls}");

    let grep = call(&reg, "grep", json!({ "pattern": "SECRET", "path": "." })).await;
    assert!(
        grep.contains("SECRET-BBBB"),
        "grep reached the wrong tenant: {grep}"
    );
    assert!(
        !grep.contains("SECRET-AAAA"),
        "grep leaked tenant A: {grep}"
    );

    let find = call(&reg, "find", json!({ "pattern": "*.txt", "path": "." })).await;
    assert!(find.contains("secret.txt"), "find: {find}");

    call(
        &reg,
        "write",
        json!({ "path": "written.txt", "content": "from-tenant-b\n" }),
    )
    .await;
    assert!(
        b.path().join("written.txt").exists(),
        "`write` did not land in tenant B's sandbox"
    );
    assert!(
        !a.path().join("written.txt").exists(),
        "`write` landed in tenant A's sandbox — cross-tenant write"
    );

    call(
        &reg,
        "edit",
        json!({ "path": "written.txt", "old_string": "tenant-b", "new_string": "edited" }),
    )
    .await;
    assert_eq!(
        std::fs::read_to_string(b.path().join("written.txt")).unwrap(),
        "from-edited\n",
        "`edit` did not land in tenant B's sandbox"
    );
}

#[tokio::test]
async fn concurrent_sessions_each_keep_their_own_target() {
    // Two sessions live at once, each with its own cell and its own registry — the shape `serve`'s
    // daemon mode has, where every session runs on its own thread with its own registry. Interleaved
    // on purpose: a target held in any process-global would surface here as one session reading the
    // other's secret.
    let a = tenant_dir("tenant-a", "SECRET-AAAA");
    let b = tenant_dir("tenant-b", "SECRET-BBBB");

    let cell_a = ExecCell::new();
    let cell_b = ExecCell::new();
    cell_a.set(Some(tenant_target(a.path()).await));
    cell_b.set(Some(tenant_target(b.path()).await));
    let reg_a = registry_over(&cell_a);
    let reg_b = registry_over(&cell_b);

    for _ in 0..3 {
        let (ra, rb) = tokio::join!(
            call(&reg_a, "read", json!({ "path": "secret.txt" })),
            call(&reg_b, "read", json!({ "path": "secret.txt" })),
        );
        assert!(
            ra.contains("SECRET-AAAA") && !ra.contains("SECRET-BBBB"),
            "A saw: {ra}"
        );
        assert!(
            rb.contains("SECRET-BBBB") && !rb.contains("SECRET-AAAA"),
            "B saw: {rb}"
        );
    }
}

#[tokio::test]
async fn a_session_with_no_target_uses_the_host_and_does_not_inherit_a_neighbours() {
    // Mixed deployment: some sessions sandboxed, some not. An empty cell must mean *this host*, not
    // "whatever the last session was using" — the latter would silently hand a local session another
    // tenant's filesystem.
    let b = tenant_dir("tenant-b", "SECRET-BBBB");
    let cell = ExecCell::new();
    let reg = registry_over(&cell);

    cell.set(Some(tenant_target(b.path()).await));
    let remote = call(&reg, "read", json!({ "path": "secret.txt" })).await;
    assert!(remote.contains("SECRET-BBBB"), "{remote}");

    cell.set(None);
    let local_dir = tempfile::tempdir().unwrap();
    std::fs::write(local_dir.path().join("host.txt"), "ON-THE-HOST\n").unwrap();
    let host = call(
        &reg,
        "read",
        json!({ "path": local_dir.path().join("host.txt").to_str().unwrap() }),
    )
    .await;
    assert!(host.contains("ON-THE-HOST"), "{host}");
    assert!(
        !host.contains("SECRET-BBBB"),
        "clearing the target left the previous tenant's sandbox attached"
    );
}

#[tokio::test]
async fn the_advertised_toolset_never_changes_when_a_session_is_repointed() {
    // Re-pointing must be invisible to the model: the tool block is a prompt-cache breakpoint, so a
    // change here would cold-miss the cache on every tenant switch *and* mean the model is looking at
    // a different toolset than it was tuned against.
    let a = tenant_dir("tenant-a", "SECRET-AAAA");
    let cell = ExecCell::new();
    let reg = registry_over(&cell);
    let before = serde_json::to_string(&reg.definitions()).unwrap();

    cell.set(Some(tenant_target(a.path()).await));
    let attached = serde_json::to_string(&reg.definitions()).unwrap();
    cell.set(None);
    let cleared = serde_json::to_string(&reg.definitions()).unwrap();

    assert_eq!(before, attached);
    assert_eq!(before, cleared);
    let plain =
        serde_json::to_string(&default_registry_with_config(&ToolConfig::new()).definitions())
            .unwrap();
    assert_eq!(
        before, plain,
        "the cell must not change what the model is offered"
    );
}
