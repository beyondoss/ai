//! Sandbox discovery: everything a service session's prompt is made of comes from the tenant's own
//! box, reached through the exec endpoint — and the replica's own `~/.claude` reaches none of it.
//!
//! The two halves are asserted together on purpose. "The sandbox's skill is advertised" alone would
//! still pass if discovery were walking *both* filesystems, and "the host's skill is absent" alone
//! would still pass if discovery had simply been left off. Each test here checks a marker that must
//! be present next to one that must not.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::service::{Options, Service, host_claude_home};
use common::{
    spawn_model_server, turn_text, ws_connect_with_headers, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Start a replica whose own `HOME` is stocked with a full `~/.claude`, so every test in this file
/// gets the "must not leak" half for free.
async fn service_with_a_stocked_host_home(
    base: &str,
) -> (Service, tempfile::TempDir, Vec<&'static str>) {
    let (host_home, markers) = host_claude_home();
    let svc = Service::start_with(
        base,
        &["s1"],
        Options {
            host_home: Some(host_home.path().to_path_buf()),
            ..Default::default()
        },
    )
    .await;
    (svc, host_home, markers)
}

fn assert_no_host_markers(raw: &str, markers: &[&str]) {
    for marker in markers {
        assert!(
            !raw.contains(marker),
            "the replica's own {marker} reached a tenant: {raw}"
        );
    }
}

/// The whole feature in one pass: a workspace `AGENTS.md`, a workspace `SKILL.md`, a workspace agent
/// definition and a workspace `SYSTEM.md` all reach the system prompt, and none of the replica's own
/// equivalents do.
#[tokio::test]
async fn the_sandboxs_own_context_skill_agent_and_system_prompt_reach_the_model() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let (svc, _host_home, markers) = service_with_a_stocked_host_home(&base).await;

    svc.write_in_workspace(
        "AGENTS.md",
        "SANDBOX-CONTEXT-MARKER: follow the house style.",
    );
    svc.write_sandbox_skill(
        ".claude/skills",
        "sandbox-skill",
        "SANDBOX-SKILL-DESCRIPTION",
        "the skill body",
    );
    svc.write_in_workspace(
        ".claude/agents/sandbox-agent.md",
        "---\nname: sandbox-agent\ndescription: SANDBOX-AGENT-DESCRIPTION\n---\nChild prompt.",
    );
    svc.write_in_workspace(".claude/SYSTEM.md", "You are SANDBOX-SYSTEM-MARKER.");

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let raw = requests.lock().unwrap()[0].clone();
    for marker in [
        "SANDBOX-CONTEXT-MARKER",
        "SANDBOX-SKILL-DESCRIPTION",
        "SANDBOX-AGENT-DESCRIPTION",
        "SANDBOX-SYSTEM-MARKER",
    ] {
        assert!(raw.contains(marker), "{marker} is missing from the prompt");
    }
    // The advertised skill location is a sandbox path the model's own `read` can open.
    assert!(
        raw.contains(
            &svc.workspace
                .join(".claude/skills/sandbox-skill/SKILL.md")
                .to_string_lossy()
                .into_owned()
        ),
        "the advertised skill location must be the sandbox path: {raw}"
    );
    assert_no_host_markers(&raw, &markers);
}

/// A tenant's `~/.claude` inside its *own* sandbox counts too — that home is the tenant's, not the
/// replica's — and the workspace shadows it on a name collision.
#[tokio::test]
async fn the_sandbox_home_contributes_and_the_workspace_shadows_it() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let (svc, _host_home, markers) = service_with_a_stocked_host_home(&base).await;

    svc.write_in_sandbox_home(
        ".claude/skills/home-only/SKILL.md",
        "---\nname: home-only\ndescription: SANDBOX-HOME-ONLY\n---\nbody\n",
    );
    svc.write_in_sandbox_home(
        ".claude/skills/shared/SKILL.md",
        "---\nname: shared\ndescription: SANDBOX-HOME-SHARED\n---\nbody\n",
    );
    svc.write_sandbox_skill(
        ".claude/skills",
        "shared",
        "SANDBOX-WORKSPACE-SHARED",
        "body",
    );

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let raw = requests.lock().unwrap()[0].clone();
    assert!(raw.contains("SANDBOX-HOME-ONLY"), "{raw}");
    assert!(raw.contains("SANDBOX-WORKSPACE-SHARED"), "{raw}");
    assert!(
        !raw.contains("SANDBOX-HOME-SHARED"),
        "the workspace's own definition must shadow the sandbox home's: {raw}"
    );
    assert_no_host_markers(&raw, &markers);
}

/// `get_commands` lists what the sandbox holds — a skill, a prompt template — and nothing of the
/// replica's. This is the surface a client builds its slash-command menu from.
#[tokio::test]
async fn get_commands_lists_the_sandboxs_skills_and_templates_only() {
    let (base, _requests) = spawn_model_server(vec![]);
    let (svc, _host_home, markers) = service_with_a_stocked_host_home(&base).await;

    svc.write_sandbox_skill(".agents/skills", "agents-skill", "from .agents/skills", "b");
    svc.write_in_workspace(
        ".claude/prompts/ship.md",
        "---\ndescription: SANDBOX-TEMPLATE\n---\nShip it.",
    );

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(&mut ws, json!({"type":"get_commands","id":"c1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_commands").await;
    let commands = frames.last().unwrap()["data"]["commands"].clone();
    let raw = serde_json::to_string(&commands).unwrap();

    let names: Vec<&str> = commands
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["name"].as_str())
        .collect();
    assert!(names.contains(&"skill:agents-skill"), "{raw}");
    assert!(names.contains(&"ship"), "{raw}");
    assert!(raw.contains("SANDBOX-TEMPLATE"), "{raw}");
    assert_no_host_markers(&raw, &markers);
}

/// `/skill:name` expands into the skill's body before the message reaches the model — which is the
/// property the prefetch exists for, since the manifest lives on the far side of the exec endpoint
/// and expansion happens on the synchronous prompt path.
#[tokio::test]
async fn a_slash_skill_invocation_expands_from_the_sandboxs_manifest() {
    let (base, requests) = spawn_model_server(vec![turn_text("done")]);
    let (svc, _host_home, markers) = service_with_a_stocked_host_home(&base).await;
    svc.write_sandbox_skill(
        ".claude/skills",
        "deploy",
        "how to deploy",
        "SANDBOX-SKILL-BODY-MARKER: run the deploy script.",
    );

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"/skill:deploy to staging"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let raw = requests.lock().unwrap()[0].clone();
    assert!(
        raw.contains("SANDBOX-SKILL-BODY-MARKER"),
        "the skill body must have been expanded into the message: {raw}"
    );
    assert!(raw.contains("to staging"), "trailing text is kept: {raw}");
    assert_no_host_markers(&raw, &markers);
}

/// A prompt template invoked as `/name` likewise expands from the sandbox.
#[tokio::test]
async fn a_slash_template_invocation_expands_from_the_sandbox() {
    let (base, requests) = spawn_model_server(vec![turn_text("done")]);
    let (svc, _host_home, markers) = service_with_a_stocked_host_home(&base).await;
    svc.write_in_workspace(
        ".claude/prompts/audit.md",
        "SANDBOX-TEMPLATE-BODY: audit $ARGUMENTS.",
    );

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"/audit the parser"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let raw = requests.lock().unwrap()[0].clone();
    assert!(raw.contains("SANDBOX-TEMPLATE-BODY"), "{raw}");
    assert!(raw.contains("the parser"), "arguments substituted: {raw}");
    assert_no_host_markers(&raw, &markers);
}

/// `reload` is allowed in service mode precisely because it re-walks the *tenant's* sandbox. A skill
/// added after the session started must show up, and the replica's own must still not.
#[tokio::test]
async fn reload_rediscovers_from_the_sandbox() {
    let (base, requests) = spawn_model_server(vec![turn_text("one"), turn_text("two")]);
    let (svc, _host_home, markers) = service_with_a_stocked_host_home(&base).await;

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert!(
        !requests.lock().unwrap()[0].contains("ADDED-AFTER-STARTUP"),
        "nothing to find yet"
    );

    // Appears only now, inside the sandbox.
    svc.write_sandbox_skill(
        ".claude/skills",
        "late-skill",
        "ADDED-AFTER-STARTUP",
        "body",
    );
    svc.write_in_workspace("AGENTS.md", "ADDED-CONTEXT-AFTER-STARTUP");

    ws_send(&mut ws, json!({"type":"reload","id":"r1"})).await;
    let frames = ws_read_until_response(&mut ws, "reload").await;
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "reload must be allowed in service mode: {frames:#?}"
    );

    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p2","message":"again"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let raw = requests.lock().unwrap()[1].clone();
    assert!(raw.contains("ADDED-AFTER-STARTUP"), "{raw}");
    assert!(raw.contains("ADDED-CONTEXT-AFTER-STARTUP"), "{raw}");
    assert_no_host_markers(&raw, &markers);
}

/// `get_state` reports the branch of the repository **in the sandbox**, not the replica's own
/// checkout — and `null` when there is no repository there, rather than the replica's answer.
#[tokio::test]
async fn get_state_reports_the_sandboxs_branch() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(&mut ws, json!({"type":"get_state","id":"g1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(
        frames.last().unwrap()["data"]["git_branch"],
        Value::Null,
        "no repository in the sandbox, and the replica's own must never be reported"
    );

    // Make the sandbox workspace a repository on a distinctly-named branch.
    for args in [
        vec!["init", "-q", "-b", "sandbox-branch"],
        vec!["config", "user.email", "t@example.com"],
        vec!["config", "user.name", "t"],
    ] {
        let status = std::process::Command::new("git")
            .args(&args)
            .current_dir(&svc.workspace)
            .status()
            .unwrap();
        assert!(status.success(), "{args:?}");
    }

    ws_send(&mut ws, json!({"type":"get_state","id":"g2"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    let data = frames.last().unwrap()["data"].clone();
    assert_eq!(data["git_branch"], "sandbox-branch", "{data}");
    assert_eq!(
        data["cwd"],
        svc.workspace.to_string_lossy().as_ref(),
        "cwd is the tenant's workspace"
    );
}
