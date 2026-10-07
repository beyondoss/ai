//! SEP-2640 approvals, end to end: MCP-served skill content may not drive host-side code execution
//! without the user's explicit, per-skill approval.
//!
//! - Activating a skill the model chose needs the user's approval, bound to the skill's manifest; a
//!   `/skill:` the user typed is that consent. A nested skill needs its own — approving the enclosing
//!   skill never covers it.
//! - While a session is acting on an MCP skill, every `bash`/`execute` call needs approval too.
//! - `serve` asks through `approval_request` frames (with an `mcp_skill` object); `run` has no one to
//!   ask and denies, unless `--approve-mcp-skills` approved in advance.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::skills_env::{
    Env, bash, is_response, last_message_text, read_skill, system_text, turn_bodies,
};
use common::{spawn_model_server_routed, turn_text};
use serde_json::{Value, json};

const GIT: &str = "skill://git-workflow/SKILL.md";
const NESTED: &str = "skill://git-workflow/nested-helper/SKILL.md";

/// The tool result the model was handed for its `id` call, from the request that carried it.
fn tool_result(bodies: &[String], needle: &str) -> String {
    bodies
        .iter()
        .map(|b| last_message_text(b))
        .find(|t| t.contains(needle))
        .unwrap_or_else(|| panic!("no tool result containing {needle:?}"))
}

#[test]
fn run_denies_a_model_initiated_load_unless_skills_were_approved_in_advance() {
    let env = Env::new(json!({}));
    let out = env.run_with(
        "use the workflow",
        vec![read_skill("t1", GIT), turn_text("done")],
        &[],
    );
    assert!(out.ok, "{}", out.stderr);
    let result = last_message_text(&out.bodies[1]);
    assert!(
        result.contains("needs the user's approval") && result.contains("--approve-mcp-skills"),
        "{result}"
    );
    assert!(!result.contains("GIT-WORKFLOW-BODY"), "{result}");
    assert!(
        env.log_of("resources/read").is_empty(),
        "nothing is fetched for a denied load: {:?}",
        env.log()
    );

    // With the operator's advance approval, the same load goes through.
    let bodies = env.run(
        "use the workflow",
        vec![read_skill("t1", GIT), turn_text("done")],
    );
    assert!(last_message_text(&bodies[1]).contains("GIT-WORKFLOW-BODY-v1"));
}

#[test]
fn run_denies_code_execution_while_an_mcp_skill_is_active() {
    let env = Env::new(json!({}));
    // Without an active MCP skill, `bash` is ungated — the gate is about acting on a skill.
    let out = env.run_with(
        "hello",
        vec![bash("t1", "echo RAN-$((40+2))"), turn_text("done")],
        &[],
    );
    assert!(last_message_text(&out.bodies[1]).contains("RAN-42"));

    // A `/skill:` the user typed loads (their consent), but code execution under it is not theirs
    // to have implied: denied without `--approve-mcp-skills`.
    let out = env.run_with(
        "/skill:docs:git-workflow commit it",
        vec![bash("t1", "echo RAN-$((40+2))"), turn_text("done")],
        &[],
    );
    assert!(out.ok, "{}", out.stderr);
    assert!(last_message_text(&out.bodies[0]).contains("GIT-WORKFLOW-BODY-v1"));
    let result = last_message_text(&out.bodies[1]);
    assert!(
        result.contains("'bash' was denied") && result.contains("docs:git-workflow"),
        "{result}"
    );
    assert!(!result.contains("RAN-42"), "{result}");

    // Approved in advance, it runs.
    let bodies = env.run(
        "/skill:docs:git-workflow commit it",
        vec![bash("t1", "echo RAN-$((40+2))"), turn_text("done")],
    );
    assert!(last_message_text(&bodies[1]).contains("RAN-42"));
}

/// The responses for one `serve` prompt keyed on what the conversation already holds, so the run
/// is deterministic whatever else (a session-title request) shares the mock.
fn serve_with(env: &Env, routes: Vec<(&str, String)>) -> (common::skills_env::Serve, Bodies) {
    let (base, bodies) = spawn_model_server_routed(
        routes
            .into_iter()
            .map(|(needle, resp)| (needle.to_string(), resp))
            .collect(),
        turn_text("fallback"),
    );
    (env.serve(&base, &[]), bodies)
}

type Bodies = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

/// Read to the next `approval_request` (returned), failing if the prompt ends first.
fn next_question(serve: &mut common::skills_env::Serve) -> Value {
    let frames = serve.read_until(|f| f["type"] == "approval_request" || is_response(f, "prompt"));
    let last = frames.last().cloned().unwrap_or(Value::Null);
    assert_eq!(
        last["type"], "approval_request",
        "the prompt ended without asking: {frames:#?}"
    );
    last
}

#[test]
fn serve_asks_to_activate_a_skill_and_to_run_code_while_it_is_active() {
    let env = Env::new(json!({}));
    // Each route fires on the newest thing in the conversation; earlier routes win.
    let (mut serve, bodies) = serve_with(
        &env,
        vec![
            ("did not approve loading", turn_text("not loaded")),
            ("RAN-THIRD", turn_text("all done")),
            ("RAN-SECOND", bash("t4", "echo RAN-THIRD")),
            ("was denied", bash("t3", "echo RAN-SECOND")),
            ("GIT-WORKFLOW-BODY-v1", bash("t2", "echo RAN-FIRST")),
            ("activate-and-run", read_skill("t1", GIT)),
        ],
    );
    serve.send(json!({ "type": "prompt", "message": "activate-and-run" }));

    // 1. Activation: the model chose the skill, so the user is asked — about this skill, its server
    //    and its manifest.
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["purpose"], "activate", "{q}");
    assert_eq!(q["mcp_skill"]["server"], "docs");
    assert_eq!(q["mcp_skill"]["uri"], GIT);
    assert!(q["mcp_skill"]["nested_in"].is_null(), "{q}");
    // Inspect before load: the listing entry's frontmatter and its file manifest are in the question.
    assert_eq!(
        q["mcp_skill"]["frontmatter"]["description"], "Follow the team's git conventions",
        "{q}"
    );
    let files = q["mcp_skill"]["manifest_files"].as_array().unwrap();
    assert!(
        files
            .iter()
            .any(|f| f["uri"] == "skill://git-workflow/references/GUIDE.md" && f["size"].is_u64()),
        "{q}"
    );
    assert!(
        q["scope_key"].as_str().unwrap().contains(GIT),
        "the remembered approval names the skill and its manifest: {q}"
    );
    assert!(
        env.log_of("resources/read").is_empty(),
        "SKILL.md is fetched only after approval: {:?}",
        env.log()
    );
    serve.approve(&q, "allow", "session");

    // 2. Code execution while acting on it: asked per call, naming the active skill. Denied once.
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["purpose"], "execute", "{q}");
    assert_eq!(q["tool"], "bash");
    assert_eq!(q["mcp_skill"]["active_skills"][0]["uri"], GIT, "{q}");
    serve.approve(&q, "deny", "once");

    // 3. A "once" denial is not remembered: the next command is asked about again. Allowed for the
    //    session...
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["purpose"], "execute", "{q}");
    assert_eq!(q["summary"]["command"], "echo RAN-SECOND");
    serve.approve(&q, "allow", "session");

    // 4. ...which covers that command only; a different one is asked about.
    let q = next_question(&mut serve);
    assert_eq!(q["summary"]["command"], "echo RAN-THIRD");
    serve.approve(&q, "allow", "once");
    serve.read_until(|f| is_response(f, "prompt"));

    // A new session forgets both the loaded skill and the approvals.
    let frames = serve.call(json!({ "type": "new_session" }), "new_session");
    assert_eq!(frames.last().unwrap()["success"], true);
    serve.send(json!({ "type": "prompt", "message": "activate-and-run" }));
    let q = next_question(&mut serve);
    assert_eq!(
        q["mcp_skill"]["purpose"], "activate",
        "a new session is asked again: {q}"
    );
    serve.approve(&q, "deny", "once");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();

    let bodies = turn_bodies(&bodies.lock().unwrap());
    // The denied call never ran: its result is the refusal, naming the active skill.
    let first = tool_result(&bodies, "'bash' was denied");
    assert!(first.contains("docs:git-workflow"), "{first}");
    assert!(!first.contains("RAN-FIRST"), "{first}");
    // The approved ones did.
    assert!(tool_result(&bodies, "RAN-SECOND").starts_with("RAN-SECOND"));
    assert!(tool_result(&bodies, "RAN-THIRD").starts_with("RAN-THIRD"));
    let refused = tool_result(&bodies, "did not approve loading");
    assert!(refused.contains(GIT), "{refused}");
    assert!(!system_text(&bodies[0]).is_empty());
}

#[test]
fn a_nested_skill_needs_its_own_consent() {
    let env = Env::new(json!({}));
    let (mut serve, bodies) = serve_with(
        &env,
        vec![
            ("did not approve loading", turn_text("ok, not loading it")),
            ("GIT-WORKFLOW-BODY-v1", read_skill("t2", NESTED)),
            ("use-nested", read_skill("t1", GIT)),
        ],
    );
    serve.send(json!({ "type": "prompt", "message": "use-nested" }));
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["uri"], GIT);
    serve.approve(&q, "allow", "session");

    // The enclosing skill is approved for the session, and yet activating the skill nested inside it
    // is a fresh question — one that says it is nested, and in what.
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["purpose"], "activate", "{q}");
    assert_eq!(q["mcp_skill"]["uri"], NESTED);
    assert_eq!(q["mcp_skill"]["nested_in"], GIT, "{q}");
    serve.approve(&q, "deny", "once");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();

    let bodies = turn_bodies(&bodies.lock().unwrap());
    let refused = tool_result(&bodies, "did not approve loading");
    assert!(refused.contains(NESTED), "{refused}");
    assert!(
        !env.log()
            .iter()
            .any(|l| l == &format!("resources/read {NESTED}")),
        "a refused nested skill is never fetched: {:?}",
        env.log()
    );
}

#[test]
fn a_skill_whose_manifest_changed_is_asked_about_again() {
    // Approved for the session under one manifest; the server then changes the skill. Loading it again
    // refreshes the entry — a different manifest — so the approval bound to the old one does not
    // cover it.
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_VERSION_FILE": "V2" }));
    let (mut serve, _bodies) = serve_with(
        &env,
        vec![
            ("GIT-WORKFLOW-BODY-v2", turn_text("loaded v2")),
            ("load-2", read_skill("t2", GIT)),
            ("GIT-WORKFLOW-BODY-v1", turn_text("loaded v1")),
            ("load-1", read_skill("t1", GIT)),
        ],
    );
    serve.send(json!({ "type": "prompt", "message": "load-1" }));
    let first = next_question(&mut serve);
    serve.approve(&first, "allow", "session");
    serve.read_until(|f| is_response(f, "prompt"));

    std::fs::write(env.cwd.join("V2"), "flip").unwrap();
    serve.send(json!({ "type": "prompt", "message": "load-2" }));
    let second = next_question(&mut serve);
    assert_eq!(second["mcp_skill"]["purpose"], "activate", "{second}");
    assert_ne!(
        second["mcp_skill"]["manifest"], first["mcp_skill"]["manifest"],
        "asked again because the manifest changed: {second}"
    );
    serve.approve(&second, "allow", "once");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();
}

#[test]
fn run_denies_a_cross_origin_read_while_acting_on_another_servers_skill() {
    let env = Env::new(json!({}));
    env.configure_two();
    // Acting on `docs`' skill (the user loaded it), the model reads `other`'s resource: denied,
    // naming both servers. Reading `docs`' own resource is not cross-origin.
    let out = env.run_with(
        "/skill:docs:git-workflow look around",
        vec![
            common::turn_tool_use("r1", "mcp__other__resource__notes", "{}"),
            common::turn_tool_use("r2", "mcp__docs__resource__notes", "{}"),
            turn_text("done"),
        ],
        &[],
    );
    assert!(out.ok, "{}", out.stderr);
    let cross = last_message_text(&out.bodies[1]);
    assert!(
        cross.contains("was denied") && cross.contains("`other`") && cross.contains("`docs`"),
        "{cross}"
    );
    assert!(!cross.contains("NOTES-BODY"), "{cross}");
    assert!(last_message_text(&out.bodies[2]).contains("NOTES-BODY"));
}

#[test]
fn serve_asks_per_call_before_a_cross_origin_read() {
    let env = Env::new(json!({}));
    env.configure_two();
    let (mut serve, bodies) = serve_with(
        &env,
        vec![
            ("SECOND-READ", turn_text("done")),
            (
                "NOTES-BODY",
                common::turn_tool_use("r2", "mcp__other__resource__notes", "{\"SECOND-READ\":1}"),
            ),
            (
                "cross-read",
                common::turn_tool_use("r1", "mcp__other__resource__notes", "{}"),
            ),
        ],
    );
    serve.call(
        json!({ "type": "prompt", "message": "/skill:docs:git-workflow load" }),
        "prompt",
    );
    serve.send(json!({ "type": "prompt", "message": "cross-read" }));
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["purpose"], "cross_origin_read", "{q}");
    assert_eq!(q["mcp_skill"]["to_server"], "other");
    assert_eq!(q["mcp_skill"]["from_servers"][0], "docs");
    serve.approve(&q, "allow", "session");
    // Per call: even a "session" answer does not cover the next read.
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["purpose"], "cross_origin_read", "{q}");
    serve.approve(&q, "deny", "once");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();
    let bodies = turn_bodies(&bodies.lock().unwrap());
    assert!(tool_result(&bodies, "NOTES-BODY").contains("NOTES-BODY"));
}

/// A skill a subagent chooses is asked about in that subagent's name — the same provenance its
/// code-execution questions carry — not as the session's own agent.
#[test]
fn a_subagents_skill_load_is_asked_about_in_its_own_name() {
    let env = Env::new(json!({}));
    let agents = env.home.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("helper.md"),
        "---\nname: helper\ndescription: helps\n---\nYou are HELPER-MARKER.\n",
    )
    .unwrap();
    let (mut serve, _bodies) = serve_with(
        &env,
        vec![
            // The parent, once the child has answered.
            ("CHILD-LOADED", turn_text("parent done")),
            // The child, after its load (allowed or not).
            ("GIT-WORKFLOW-BODY", turn_text("CHILD-LOADED")),
            ("did not approve loading", turn_text("CHILD-LOADED")),
            // The child's first turn: it chooses the skill.
            ("HELPER-MARKER", read_skill("c1", GIT)),
            (
                "delegate-load",
                common::turn_tool_use(
                    "d1",
                    "subagent",
                    &json!({ "agent": "helper", "task": "load the git skill" }).to_string(),
                ),
            ),
        ],
    );
    serve.send(json!({ "type": "prompt", "message": "delegate-load" }));
    let q = next_question(&mut serve);
    assert_eq!(q["mcp_skill"]["purpose"], "activate", "{q}");
    assert_eq!(
        q["origin"]["agent"], "helper",
        "the question must name the subagent that chose the skill: {q}"
    );
    assert!(q["origin"]["spawn_id"].is_string(), "{q}");
    serve.approve(&q, "allow", "once");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();
}
