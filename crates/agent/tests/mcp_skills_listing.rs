//! SEP-2640 listings over time, and what is left out of them.
//!
//! - A skill published after connect gets listed: on a list-changed notification, or once the
//!   listing's `ttlMs` runs out — and not before (`ttlMs` is honoured, not polled past).
//! - A `cacheScope: "private"` listing is never written to the on-disk manifest cache.
//! - A listing re-fetched after connect is written back to that cache at once (or, now private,
//!   forgotten), so a restart does not advertise the listing the server replaced.
//! - Skills over the spec's per-skill limits (512 files / 16 MiB) are declined, and every entry left
//!   out — over-limit, dynamic — is surfaced through the skills diagnostics (`get_commands`'
//!   `collisions`, `run`'s warnings), as is a name one server lists twice.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::skills_env::{Env, Serve, is_response, system_text, turn_bodies};
use common::{spawn_model_server_routed, turn_text, turn_tool_use};
use serde_json::{Value, json};

fn routed_serve(env: &Env, routes: Vec<(&str, String)>) -> (Serve, common::skills_env::Bodies) {
    let (base, bodies) = spawn_model_server_routed(
        routes
            .into_iter()
            .map(|(n, r)| (n.to_string(), r))
            .collect(),
        turn_text("ok"),
    );
    (env.serve(&base, &[]), bodies)
}

fn prompt(serve: &mut Serve, message: &str) {
    serve.send(json!({ "type": "prompt", "message": message }));
    serve.read_until(|f| is_response(f, "prompt"));
}

/// The system prompt of the first real turn whose conversation holds `marker`.
fn system_of(bodies: &[String], marker: &str) -> String {
    turn_bodies(bodies)
        .iter()
        .find(|b| b.contains(marker))
        .map(|b| system_text(b))
        .unwrap_or_else(|| panic!("no turn carried {marker:?}"))
}

#[test]
fn an_expired_listing_is_refetched_and_a_skill_published_meanwhile_is_listed() {
    let env = Env::new(json!({
        "MCP_SKILLS_FIXTURE_LATE_FLAG": "LATE",
        "MCP_SKILLS_FIXTURE_TTL_MS": "0",
    }));
    // A relative flag path resolves against the fixture's cwd — the agent's.
    let flag = env.cwd.join("LATE");
    let (mut serve, bodies) = routed_serve(&env, vec![]);
    prompt(&mut serve, "first-turn");
    std::fs::write(&flag, "published").unwrap();
    prompt(&mut serve, "second-turn");
    serve.finish();

    let bodies = bodies.lock().unwrap().clone();
    assert!(!system_of(&bodies, "first-turn").contains("<name>docs:late</name>"));
    // `ttlMs: 0` — stale at once — so the next turn that needs the listing fetches it again.
    assert!(
        system_of(&bodies, "second-turn").contains("<name>docs:late</name>"),
        "{}",
        system_of(&bodies, "second-turn")
    );
}

#[test]
fn a_fresh_listing_is_not_refetched_but_a_list_changed_notification_invalidates_it() {
    let env = Env::new(json!({
        "MCP_SKILLS_FIXTURE_LATE_FLAG": "LATE",
        "MCP_SKILLS_FIXTURE_TTL_MS": "3600000",
    }));
    let flag = env.cwd.join("LATE");
    let (mut serve, bodies) = routed_serve(
        &env,
        vec![
            ("third-turn", turn_text("third done")),
            ("LATE-PUBLISHED", turn_text("published it")),
            (
                "publish-now",
                turn_tool_use("p1", "mcp__docs__publish_late", "{}"),
            ),
        ],
    );
    prompt(&mut serve, "first-turn");
    // Published behind the listing's back: no notification, and the listing is fresh for an hour —
    // so it is not asked for again, and the late skill is not listed yet.
    std::fs::write(&flag, "published").unwrap();
    prompt(&mut serve, "second-turn");
    let lists_before = env.log_of("skills/list").len();
    std::fs::remove_file(&flag).unwrap();

    // Now the server publishes it itself, and says the listing changed.
    prompt(&mut serve, "publish-now");
    prompt(&mut serve, "third-turn");
    serve.finish();

    assert_eq!(
        lists_before,
        1,
        "a listing fresh by its ttlMs is not fetched again: {:?}",
        env.log()
    );
    let bodies = bodies.lock().unwrap().clone();
    assert!(!system_of(&bodies, "second-turn").contains("<name>docs:late</name>"));
    let third = system_of(&bodies, "third-turn");
    assert!(
        third.contains("<name>docs:late</name>"),
        "a list-changed notification invalidates a fresh listing: {:?}",
        env.log()
    );
}

#[test]
fn a_private_listing_is_never_written_to_the_manifest_cache() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_CACHE_SCOPE": "private" }));
    env.run("hello", vec![turn_text("ok")]);
    let manifest =
        std::fs::read_to_string(env.home.join(".claude/mcp-manifest.json")).unwrap_or_default();
    assert!(
        !manifest.contains("skill://"),
        "a private listing must not outlive its authorization context on disk: {manifest}"
    );
    // So the next boot asks the server again rather than answering from the cache.
    env.clear_log();
    let bodies = env.run("hello", vec![turn_text("ok")]);
    assert!(
        env.log().contains(&"start -".to_string()),
        "{:?}",
        env.log()
    );
    assert!(system_text(&bodies[0]).contains("<name>docs:git-workflow</name>"));

    // A public one is cached (the control).
    let public = Env::new(json!({}));
    public.run("hello", vec![turn_text("ok")]);
    let manifest = std::fs::read_to_string(public.home.join(".claude/mcp-manifest.json")).unwrap();
    assert!(manifest.contains("skill://git-workflow/SKILL.md"));
}

#[test]
fn entries_left_out_are_surfaced_in_get_commands_and_run_warnings() {
    let env = Env::new(json!({}));
    let (mut serve, _bodies) = routed_serve(&env, vec![]);
    let frames = serve.call(json!({ "type": "get_commands" }), "get_commands");
    serve.finish();
    let response = frames.last().unwrap();
    let commands = response["data"]["commands"].as_array().unwrap();
    for over in ["huge", "heavy", "dyn"] {
        assert!(
            !commands
                .iter()
                .any(|c| c["name"] == format!("skill:docs:{over}")),
            "{over} must not be offered: {response}"
        );
    }
    let messages: Vec<String> = response["data"]["collisions"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c: &Value| c["message"].as_str().map(str::to_string))
        .collect();
    let has = |needle: &str| messages.iter().any(|m| m.contains(needle));
    assert!(
        has("skill://huge/SKILL.md") && has("513 files exceeds the 512-file per-skill limit"),
        "{messages:#?}"
    );
    assert!(
        has("skill://heavy/SKILL.md") && has("16 MiB"),
        "{messages:#?}"
    );
    assert!(
        has("skill://dyn/SKILL.md") && has("dynamic"),
        "{messages:#?}"
    );
    assert!(has("lists 2 skills named `refunds`"), "{messages:#?}");

    // `run` has no `get_commands`; it says so on stderr.
    let out = env.run_with("hello", vec![turn_text("ok")], &[]);
    assert!(
        out.stderr
            .contains("warning: mcp server `docs`: `skill://huge/SKILL.md`: declined"),
        "{}",
        out.stderr
    );
}

fn cached_manifest(env: &Env) -> String {
    std::fs::read_to_string(env.home.join(".claude/mcp-manifest.json")).unwrap_or_default()
}

/// A listing re-fetched mid-session reaches the on-disk cache then, not at the next live connect:
/// a restart that answers from the cache offers the skill published meanwhile.
#[test]
fn a_refreshed_listing_is_written_back_to_the_manifest_cache() {
    let env = Env::new(json!({
        "MCP_SKILLS_FIXTURE_LATE_FLAG": "LATE",
        "MCP_SKILLS_FIXTURE_TTL_MS": "0",
    }));
    let flag = env.cwd.join("LATE");
    let (mut serve, _bodies) = routed_serve(&env, vec![]);
    prompt(&mut serve, "first-turn");
    assert!(cached_manifest(&env).contains("skill://git-workflow/SKILL.md"));
    assert!(!cached_manifest(&env).contains("skill://late/SKILL.md"));
    std::fs::write(&flag, "published").unwrap();
    prompt(&mut serve, "second-turn");
    // Written while the session is still up, by the re-list itself.
    assert!(
        cached_manifest(&env).contains("skill://late/SKILL.md"),
        "the refreshed listing must be written back: {}",
        cached_manifest(&env)
    );
    serve.finish();

    // And a restart answering from the cache (it starts no server) offers it.
    env.clear_log();
    let (mut serve, bodies) = routed_serve(&env, vec![]);
    prompt(&mut serve, "after-restart");
    serve.finish();
    assert!(
        !env.log().contains(&"start -".to_string()),
        "the restart answered from the cache: {:?}",
        env.log()
    );
    let bodies = bodies.lock().unwrap().clone();
    assert!(system_of(&bodies, "after-restart").contains("<name>docs:late</name>"));
}

/// A listing that turns private on a re-fetch is dropped from the cache, as one private at connect
/// is never written.
#[test]
fn a_listing_that_turns_private_on_refresh_is_forgotten_by_the_cache() {
    let env = Env::new(json!({
        "MCP_SKILLS_FIXTURE_PRIVATE_FLAG": "PRIVATE",
        "MCP_SKILLS_FIXTURE_TTL_MS": "0",
    }));
    let (mut serve, _bodies) = routed_serve(&env, vec![]);
    prompt(&mut serve, "first-turn");
    assert!(cached_manifest(&env).contains("skill://git-workflow/SKILL.md"));
    std::fs::write(env.cwd.join("PRIVATE"), "now").unwrap();
    prompt(&mut serve, "second-turn");
    serve.finish();
    assert!(
        !cached_manifest(&env).contains("skill://"),
        "a now-private listing must not stay on disk: {}",
        cached_manifest(&env)
    );
}
