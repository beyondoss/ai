//! MCP Skills extension (SEP-2640) end to end: the real `run`/`serve` binary against a real
//! subprocess (`mcp_skills_fixture_server`) speaking the wire protocol, with a mock model recording
//! exactly what the agent sent it.
//!
//! The fixture logs every request it receives, so "the host did not fetch a skill file ahead of
//! need" and "the server was never started" are asserted against what reached the server, not
//! against the client's opinion of itself.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;
use std::process::Stdio;

use common::skills_env::{
    BIN, Env, FIXTURE, SKILL_TOOL, last_message_text, read_skill, system_text, turn_bodies,
};
use common::{SpawnGuarded, advertised_tools, run_cmd, spawn_model_server, turn_text};
use serde_json::json;

#[test]
fn run_lists_mcp_skills_without_fetching_any_skill_file() {
    let env = Env::new(json!({}));
    let bodies = env.run("hello", vec![turn_text("ok")]);
    let system = system_text(&bodies[0]);

    assert!(
        system.contains("<available_skills origin=\"mcp\">"),
        "{system}"
    );
    for expected in [
        "<name>docs:git-workflow</name>",
        "<location>skill://git-workflow/SKILL.md</location>",
        "<server>docs</server>",
        "<tool>mcp__docs__skill__read</tool>",
        "Follow the team&apos;s git conventions",
        // Two listed skills named `refunds`: both kept, each qualified by its skill path.
        "<name>docs:acme/billing/refunds</name>",
        "<name>docs:acme/support/refunds</name>",
        "untrusted",
    ] {
        assert!(
            system.contains(expected),
            "missing {expected:?} in:\n{system}"
        );
    }
    // `disable-model-invocation`, a dynamic skill (declined) and an unlisted one stay out.
    // Nor do skills over the per-skill limits (declined), nor one not yet published.
    for absent in [
        "manual-only",
        "skill://dyn/",
        "hidden",
        "skill://huge/",
        "skill://heavy/",
        "skill://late/",
    ] {
        assert!(
            !system.contains(absent),
            "{absent:?} must not be listed:\n{system}"
        );
    }

    let tools = advertised_tools(&bodies[0]);
    assert!(tools.contains(&SKILL_TOOL.to_string()), "{tools:?}");
    assert!(tools.contains(&"mcp__docs__echo".to_string()), "{tools:?}");
    // The ordinary resource is still a resource tool; the skill files are not duplicated as ones.
    assert!(
        tools.contains(&"mcp__docs__resource__notes".to_string()),
        "{tools:?}"
    );
    assert!(
        !tools
            .iter()
            .any(|t| t.contains("git_workflow") || t.contains("acme") || t.contains("tampered")),
        "skill files must not also be generic resource tools: {tools:?}"
    );
    // The MCP Apps `ui://` filter composes with the skills one in the same pass: the app's HTML is
    // no resource tool either, while the ordinary resource above still is.
    assert!(
        !tools.iter().any(|t| t.contains("widget_app")),
        "a `ui://` resource must not be a generic resource tool: {tools:?}"
    );
    // Nor are a declined skill's files (dynamic, over the limits), nor anything under `skill://`: a
    // generic resource read would open no acting window and pass no gate.
    assert!(
        !tools
            .iter()
            .any(|t| t.contains("_SKILL_md") || t.contains("dyn") || t.contains("huge")),
        "declined skill files must not be generic resource tools: {tools:?}"
    );

    // Listing only: no skill file, and no per-skill lookup, was requested.
    let log = env.log();
    assert!(log.contains(&"skills/list -".to_string()), "{log:?}");
    assert!(
        !log.iter()
            .any(|l| l.starts_with("resources/read") || l.starts_with("skills/get")),
        "nothing may be fetched ahead of need: {log:?}"
    );
}

#[test]
fn run_loads_a_skill_then_reads_its_files_through_the_skill_tool() {
    let env = Env::new(json!({}));
    let bodies = env.run(
        "use the git workflow",
        vec![
            read_skill("t1", "skill://git-workflow/SKILL.md"),
            read_skill("t2", "skill://git-workflow/references/GUIDE.md"),
            read_skill("t3", "skill://git-workflow/references"),
            turn_text("done"),
        ],
    );
    assert_eq!(bodies.len(), 4);

    let loaded = last_message_text(&bodies[1]);
    assert!(loaded.contains("GIT-WORKFLOW-BODY-v1"), "{loaded}");
    assert!(
        loaded.contains(
            "<skill name=\"git-workflow\" server=\"docs\" location=\"skill://git-workflow/SKILL.md\" manifest=\""
        ),
        "loaded content must carry its origin: {loaded}"
    );
    assert!(
        loaded.contains("skill root skill://git-workflow"),
        "{loaded}"
    );
    assert!(loaded.contains("Files: references/GUIDE.md"), "{loaded}");
    assert!(
        !loaded.contains("description: Follow"),
        "frontmatter is metadata, not body: {loaded}"
    );

    let guide = last_message_text(&bodies[2]);
    assert!(
        guide.contains("GUIDE-BODY: squash before merge."),
        "{guide}"
    );
    assert!(guide.contains("from MCP server `docs`"), "{guide}");

    let dir = last_message_text(&bodies[3]);
    assert!(dir.contains("GUIDE.md"), "{dir}");

    let log = env.log();
    let reads: Vec<&String> = log
        .iter()
        .filter(|l| l.starts_with("resources/read"))
        .collect();
    assert_eq!(
        reads,
        [
            "resources/read skill://git-workflow/SKILL.md",
            "resources/read skill://git-workflow/references/GUIDE.md",
        ],
        "the directory listing is answered from the manifest, without a read: {log:?}"
    );
}

#[test]
fn run_refuses_tampered_unloaded_and_unknown_skill_content() {
    let env = Env::new(json!({}));
    let bodies = env.run(
        "try things",
        vec![
            read_skill("t1", "skill://tampered/SKILL.md"),
            read_skill("t2", "skill://git-workflow/references/GUIDE.md"),
            read_skill("t3", "skill://hidden/SKILL.md"),
            read_skill("t4", "skill://nope/SKILL.md"),
            turn_text("done"),
        ],
    );
    assert_eq!(bodies.len(), 5);

    let tampered = last_message_text(&bodies[1]);
    assert!(
        tampered.contains("does not match its manifest digest"),
        "{tampered}"
    );
    assert!(
        !tampered.contains("EVIL"),
        "unverified bytes must never reach the model: {tampered}"
    );

    let unloaded = last_message_text(&bodies[2]);
    assert!(unloaded.contains("load the skill first"), "{unloaded}");

    // Never listed, but the server answers `skills/get` for it, so it loads by URI alone.
    let hidden = last_message_text(&bodies[3]);
    assert!(hidden.contains("HIDDEN-BODY"), "{hidden}");

    let unknown = last_message_text(&bodies[4]);
    assert!(unknown.contains("not a skill"), "{unknown}");

    let log = env.log();
    // The tampered read was retried once against a refreshed entry, then refused.
    assert!(
        log.contains(&"skills/get skill://tampered/SKILL.md".to_string()),
        "{log:?}"
    );
    // A file of a skill that was never loaded is not fetched at all.
    assert!(
        !log.contains(&"resources/read skill://git-workflow/references/GUIDE.md".to_string()),
        "{log:?}"
    );
}

#[test]
fn run_skill_invocation_loads_the_mcp_skill_at_the_users_request() {
    let env = Env::new(json!({}));
    // `manual-only` is `disable-model-invocation`: not listed for the model, but the user may name it.
    let bodies = env.run(
        "/skill:docs:manual-only do the thing",
        vec![turn_text("ok")],
    );
    let user = last_message_text(&bodies[0]);
    assert!(user.contains("MANUAL-ONLY-BODY"), "{user}");
    assert!(user.contains("server=\"docs\""), "{user}");
    assert!(user.ends_with("do the thing\n"), "{user:?}");
    assert!(!system_text(&bodies[0]).contains("manual-only"));
    assert!(
        env.log()
            .contains(&"resources/read skill://manual-only/SKILL.md".to_string())
    );

    // A colliding name is reached by its qualified form.
    let bodies = env.run("/skill:docs:acme/support/refunds", vec![turn_text("ok")]);
    let user = last_message_text(&bodies[0]);
    assert!(user.contains("SUPPORT-REFUNDS-BODY"), "{user}");
}

#[test]
fn a_server_without_the_extension_is_unchanged() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_NO_EXTENSION": "1" }));
    let bodies = env.run("/skill:docs:git-workflow", vec![turn_text("ok")]);
    assert!(!system_text(&bodies[0]).contains("origin=\"mcp\""));
    let tools = advertised_tools(&bodies[0]);
    assert!(
        !tools.iter().any(|t| t.contains("skill__read")),
        "{tools:?}"
    );
    // Its `skill://` resources are ordinary resources, wrapped like any other.
    assert!(
        tools.contains(&"mcp__docs__resource__git_workflow_SKILL_md".to_string()),
        "{tools:?}"
    );
    // The invocation names nothing this host knows, so it reaches the model as typed.
    assert_eq!(last_message_text(&bodies[0]), "/skill:docs:git-workflow\n");
    let log = env.log();
    assert!(!log.iter().any(|l| l.starts_with("skills/")), "{log:?}");
}

#[test]
fn an_untrusted_projects_skill_server_is_never_started() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = dir.path().join("project");
    let log = dir.path().join("fixture.log");
    std::fs::create_dir_all(project.join(".claude")).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        project.join(".claude/settings.json"),
        json!({ "mcp_servers": [{
            "name": "docs",
            "transport": "stdio",
            "command": FIXTURE,
            "env": { "MCP_SKILLS_FIXTURE_LOG": log },
        }] })
        .to_string(),
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let output = run_cmd(BIN)
        .env("HOME", &home)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(&project)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded()
        .wait_with_output()
        .unwrap();
    assert!(output.status.success());
    let bodies = bodies.lock().unwrap().clone();
    assert!(!system_text(&bodies[0]).contains("origin=\"mcp\""));
    assert!(
        !Path::new(&log).exists(),
        "an untrusted project's server must not start"
    );
}

#[test]
fn a_cached_manifest_lists_skills_without_a_server_and_a_stale_one_is_refreshed() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_VERSION": "1" }));
    // First boot discovers, and records the manifest under `$HOME/.claude`.
    env.run("hello", vec![turn_text("ok")]);
    env.clear_log();

    // Second boot: same invocation, so the cache answers — the server is never started, and the
    // skills are listed anyway.
    let bodies = env.run("hello", vec![turn_text("ok")]);
    assert!(
        system_text(&bodies[0]).contains("<name>docs:git-workflow</name>"),
        "{}",
        system_text(&bodies[0])
    );
    assert!(env.log().is_empty(), "no server may start: {:?}", env.log());

    // The server changes its skill without changing its invocation (env values are not part of the
    // cache key). A cached entry's freshness is unknown, so loading re-fetches it through
    // `skills/get` first and serves the current body, never the cached promise of the old one.
    env.configure(json!({ "MCP_SKILLS_FIXTURE_VERSION": "2" }));
    let bodies = env.run(
        "use it",
        vec![
            read_skill("t1", "skill://git-workflow/SKILL.md"),
            turn_text("done"),
        ],
    );
    let loaded = last_message_text(&bodies[1]);
    assert!(loaded.contains("GIT-WORKFLOW-BODY-v2"), "{loaded}");
    let log: Vec<String> = env
        .log()
        .into_iter()
        .filter(|l| !l.starts_with("server/discover") && !l.starts_with("initialize"))
        .collect();
    assert_eq!(
        log,
        [
            "start -",
            "skills/get skill://git-workflow/SKILL.md",
            "resources/read skill://git-workflow/SKILL.md",
        ],
        "a dormant server starts on first use, and the stale entry is refreshed: {log:?}"
    );
}

/// `serve`: `get_commands` lists MCP skills, and `set_mcp_enabled` hides them everywhere — the
/// listing, the tool, `/skill:` expansion and `get_commands` — and brings them back.
#[test]
fn serve_get_commands_and_the_mcp_gate_cover_skills() {
    let env = Env::new(json!({}));
    // Three answers: the two prompts, and the session-title request the first one triggers.
    let (base, bodies) =
        spawn_model_server(vec![turn_text("one"), turn_text("two"), turn_text("three")]);
    let mut serve = env.serve(&base, &[]);
    let response = |frames: Vec<serde_json::Value>| frames.last().unwrap().clone();

    let commands = response(serve.call(json!({ "type": "get_commands" }), "get_commands"));
    let list = commands["data"]["commands"].as_array().unwrap();
    let git = list
        .iter()
        .find(|c| c["name"] == "skill:docs:git-workflow")
        .unwrap_or_else(|| panic!("{commands}"));
    assert_eq!(git["scope"], "mcp");
    assert_eq!(git["server"], "docs");
    assert_eq!(git["path"], "skill://git-workflow/SKILL.md");
    // `get_commands` is the user's menu: a user-only skill belongs on it.
    assert!(list.iter().any(|c| c["name"] == "skill:docs:manual-only"));

    let gated = response(serve.call(
        json!({ "type": "set_mcp_enabled", "servers": [] }),
        "set_mcp_enabled",
    ));
    assert_eq!(gated["success"], true, "{gated}");
    let commands = response(serve.call(json!({ "type": "get_commands" }), "get_commands"));
    assert!(
        !commands.to_string().contains("skill:docs:"),
        "a disabled server's skills leave the menu: {commands}"
    );
    serve.call(
        json!({ "type": "prompt", "message": "/skill:docs:git-workflow" }),
        "prompt",
    );

    let enabled = response(serve.call(
        json!({ "type": "set_mcp_enabled", "servers": null }),
        "set_mcp_enabled",
    ));
    assert_eq!(enabled["success"], true, "{enabled}");
    serve.call(
        json!({ "type": "prompt", "message": "/skill:docs:git-workflow now" }),
        "prompt",
    );
    serve.finish();

    let bodies = turn_bodies(&bodies.lock().unwrap());
    assert_eq!(bodies.len(), 2);
    // Gated: no listing, no tool, and the invocation is not expanded (nor fetched).
    assert!(!system_text(&bodies[0]).contains("origin=\"mcp\""));
    assert!(!advertised_tools(&bodies[0]).contains(&SKILL_TOOL.to_string()));
    assert_eq!(last_message_text(&bodies[0]), "/skill:docs:git-workflow\n");
    // Re-enabled: all of it is back, and the invocation loads the skill.
    assert!(
        system_text(&bodies[1]).contains("<name>docs:git-workflow</name>"),
        "{}",
        system_text(&bodies[1])
    );
    assert!(advertised_tools(&bodies[1]).contains(&SKILL_TOOL.to_string()));
    let user = last_message_text(&bodies[1]);
    assert!(user.contains("GIT-WORKFLOW-BODY-v1"), "{user}");
    assert!(user.trim_end().ends_with("now"), "{user:?}");
    let reads: Vec<String> = env
        .log()
        .into_iter()
        .filter(|l| l.starts_with("resources/read"))
        .collect();
    assert_eq!(reads, ["resources/read skill://git-workflow/SKILL.md"]);
}

/// The same server over streamable HTTP. Its results carry `_meta`, as every `2026-07-28` server's
/// do — which is exactly what `rmcp` mistakes for a tool result unless `mcp_wire` intervenes — so
/// this is the HTTP half of that regression, the stdio half being every test above.
#[test]
fn skills_work_over_streamable_http() {
    use std::io::BufRead;
    let env = Env::new(json!({}));
    let mut server = std::process::Command::new(FIXTURE)
        .arg("--http")
        .env("MCP_SKILLS_FIXTURE_LOG", &env.log)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn_guarded();
    let mut url = String::new();
    common::child_frames(&mut server)
        .read_line(&mut url)
        .unwrap();
    std::fs::write(
        env.home.join(".claude/settings.json"),
        json!({ "mcp_servers": [{ "name": "docs", "transport": "http", "url": url.trim() }] })
            .to_string(),
    )
    .unwrap();

    let bodies = env.run(
        "use it",
        vec![
            read_skill("t1", "skill://git-workflow/SKILL.md"),
            read_skill("t2", "skill://git-workflow/references/GUIDE.md"),
            turn_text("done"),
        ],
    );
    assert!(system_text(&bodies[0]).contains("<name>docs:git-workflow</name>"));
    assert!(last_message_text(&bodies[1]).contains("GIT-WORKFLOW-BODY-v1"));
    assert!(last_message_text(&bodies[2]).contains("GUIDE-BODY"));

    let bodies = env.run("/skill:docs:acme/billing/refunds", vec![turn_text("ok")]);
    assert!(last_message_text(&bodies[0]).contains("BILLING-REFUNDS-BODY"));
    drop(server);
}
