//! `serve` e2e: Project trust gating for skills/prompt-templates, and the system prompt.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Read, Write};
use std::process::{Command, Stdio};

use common::{SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text};
use serde_json::json;

#[test]
fn serve_injects_project_instructions_into_system_prompt() {
    // A `CLAUDE.md` in the working directory must reach the model: the agent assembles it into the
    // system prompt. The mock server records the request body, so we assert the marker is present.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("CLAUDE.md"), "PROJECT-MARKER-9182").unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.current_dir(dir.path()); // CLAUDE.md is discovered relative to cwd
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("PROJECT-MARKER-9182")),
        "project CLAUDE.md must be injected into the system prompt; bodies: {recorded:#?}"
    );
}

#[test]
fn serve_caches_the_static_system_prompt_until_reload() {
    // The static half of the system prompt (project instructions, skills) is expensive (filesystem
    // discovery) and is meant to be cached across ordinary turns, refreshed only by `set_model`/
    // `set_thinking`/an explicit `reload` — never by the per-turn dynamic-footer refresh alone.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("CLAUDE.md"), "MARKER-BEFORE-RELOAD").unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) =
        spawn_model_server(vec![turn_text("ok"), turn_text("ok"), turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.current_dir(dir.path());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Turn 1: the marker as it existed at startup.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "one" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Change the file on disk *while the process keeps running*, then prompt again with no reload.
    std::fs::write(dir.path().join("CLAUDE.md"), "MARKER-AFTER-EDIT").unwrap();
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "two" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Now explicitly reload, then prompt a third time.
    writeln!(stdin, "{}", json!({ "type": "reload" })).unwrap();
    stdin.flush().unwrap();
    let reload_frames = read_until_response(&mut stdout, "reload");
    assert_eq!(
        reload_frames.last().unwrap()["success"],
        true,
        "reload must succeed: {reload_frames:#?}"
    );
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "three" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert_eq!(recorded.len(), 3, "expected exactly three model calls");
    assert!(
        recorded[0].contains("MARKER-BEFORE-RELOAD"),
        "turn 1 must see the file as it was at startup"
    );
    assert!(
        recorded[1].contains("MARKER-BEFORE-RELOAD") && !recorded[1].contains("MARKER-AFTER-EDIT"),
        "turn 2 must still see the cached static prompt, not the on-disk edit: {}",
        recorded[1]
    );
    assert!(
        recorded[2].contains("MARKER-AFTER-EDIT"),
        "turn 3, after an explicit reload, must see the on-disk edit: {}",
        recorded[2]
    );
}

#[test]
fn serve_gates_skills_and_prompts_on_project_trust() {
    // An untrusted project's `.claude/skills`/`.claude/prompts` are attacker-controlled instructions:
    // they must not be advertised via `get_commands`, nor invocable via `/skill:name`/`/name`, until
    // the directory is trusted. Isolate HOME so this doesn't see (or pollute) the real developer's
    // `~/.claude/skills`/`~/.claude/trusted-projects.json`.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nDo the foo thing.",
    )
    .unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(prompt_dir.join("bar.md"), "Bar template body: $1").unwrap();

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Untrusted: no --trust-project.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
        let mut cmd = serve_cmd(bin, &base, &session_file);
        cmd.current_dir(dir.path()).env("HOME", home.path());
        let mut child = cmd.spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_commands");
        let commands = frames.last().unwrap()["data"]["commands"]
            .as_array()
            .unwrap();
        assert!(
            commands.is_empty(),
            "an untrusted project must advertise no skills/prompts, got: {commands:?}"
        );

        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "/skill:foo" })
        )
        .unwrap();
        stdin.flush().unwrap();
        drop(stdin);
        child.wait().unwrap();

        let recorded = _bodies.lock().unwrap();
        assert!(
            recorded.iter().all(|b| !b.contains("Do the foo thing")),
            "an untrusted project's skill body must never reach the model: {recorded:#?}"
        );
    }

    // Trusted: with --trust-project, both the skill and the prompt template are discoverable.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
        let session_file_2 = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
        let mut cmd = serve_cmd(bin, &base, &session_file_2);
        cmd.arg("--trust-project")
            .current_dir(dir.path())
            .env("HOME", home.path());
        let mut child = cmd.spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_commands");
        let commands = frames.last().unwrap()["data"]["commands"]
            .as_array()
            .unwrap();
        let names: Vec<&str> = commands.iter().filter_map(|c| c["name"].as_str()).collect();
        assert!(
            names.contains(&"skill:foo"),
            "trusted project should list the skill: {names:?}"
        );
        assert!(
            names.contains(&"bar"),
            "trusted project should list the prompt template: {names:?}"
        );

        drop(stdin);
        child.wait().unwrap();
    }
}

#[test]
fn serve_force_untrusted_overrides_a_persisted_trust_grant() {
    // Track L8: `--force-untrusted` must win even over a *persisted* `agent trust <path>` grant, not
    // just the absence of `--trust-project` — the whole point is overriding an operator's standing
    // trust decision for one run (testing untrusted behavior, or extra caution on a checkout that
    // happens to live under an already-trusted parent).
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nDo the foo thing.",
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Persist trust for `dir` via the real `trust` subcommand — the same allowlist `serve` itself
    // reads from, not a hand-rolled fixture.
    let trust_output = Command::new(bin)
        .args(["trust", dir.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust_output.status.success());

    // Control: without --force-untrusted, the persisted trust is honored — the skill is visible.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let session_file = dir.path().join("s1.jsonl").to_string_lossy().into_owned();
        let mut cmd = serve_cmd(bin, &base, &session_file);
        cmd.current_dir(dir.path()).env("HOME", home.path());
        let mut child = cmd.spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_commands");
        let names: Vec<&str> = frames.last().unwrap()["data"]["commands"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c["name"].as_str())
            .collect();
        assert!(
            names.contains(&"skill:foo"),
            "the persisted trust grant should be honored: {names:?}"
        );

        drop(stdin);
        child.wait().unwrap();
    }

    // With --force-untrusted: the same persisted trust grant must be overridden.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let session_file = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
        let mut cmd = serve_cmd(bin, &base, &session_file);
        cmd.arg("--force-untrusted")
            .current_dir(dir.path())
            .env("HOME", home.path());
        let mut child = cmd.spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_commands");
        let commands = frames.last().unwrap()["data"]["commands"]
            .as_array()
            .unwrap();
        assert!(
            commands.is_empty(),
            "--force-untrusted must override the persisted trust grant: {commands:?}"
        );

        drop(stdin);
        child.wait().unwrap();
    }
}

#[test]
fn serve_untrusted_project_still_advertises_and_invokes_a_user_global_skill() {
    // The bug this guards: `.claude/skills` under the *project* is attacker-controlled and rightly
    // gated on trust, but `~/.claude/skills` is the operator's own machine — an untrusted project must
    // not blank that out too.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let user_skill_dir = home.path().join(".claude/skills/mine");
    std::fs::create_dir_all(&user_skill_dir).unwrap();
    std::fs::write(
        user_skill_dir.join("SKILL.md"),
        "---\nname: mine\ndescription: a user-global skill\n---\nDo the user thing.",
    )
    .unwrap();
    // Also seed an untrusted *project* skill, to confirm it stays gated even while the user skill
    // isn't — the split, not just a blanket toggle.
    let project_skill_dir = dir.path().join(".claude/skills/theirs");
    std::fs::create_dir_all(&project_skill_dir).unwrap();
    std::fs::write(
        project_skill_dir.join("SKILL.md"),
        "---\nname: theirs\ndescription: an untrusted project skill\n---\nBody.",
    )
    .unwrap();

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let mut cmd = serve_cmd(bin, &base, &session_file);
    // No --trust-project: the project is untrusted.
    cmd.current_dir(dir.path()).env("HOME", home.path());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap();
    let names: Vec<&str> = commands.iter().filter_map(|c| c["name"].as_str()).collect();
    assert!(
        names.contains(&"skill:mine"),
        "an untrusted project must still advertise the user-global skill: {names:?}"
    );
    assert!(
        !names.contains(&"skill:theirs"),
        "the untrusted project's own skill must stay gated: {names:?}"
    );

    // And it's actually invocable, not merely listed.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "/skill:mine" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = _bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("Do the user thing")),
        "the user-global skill's body must reach the model when invoked: {recorded:#?}"
    );
}

#[test]
fn serve_get_commands_reports_a_cross_root_skill_collision() {
    // A skill named "dup" declared at both the user (`~/.claude/skills`) and project
    // (`<cwd>/.claude/skills`) roots: `get_commands` must list it once (project wins) but also report
    // the shadowing via `collisions`, rather than silently resolving it with no client-visible trace.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    for (root, description) in [(home.path(), "user copy"), (dir.path(), "project copy")] {
        let skill_dir = root.join(".claude/skills/dup");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: dup\ndescription: {description}\n---\nBody."),
        )
        .unwrap();
    }
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project")
        .current_dir(dir.path())
        .env("HOME", home.path());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let data = &frames.last().unwrap()["data"];
    let commands = data["commands"].as_array().unwrap();
    let dup_count = commands
        .iter()
        .filter(|c| c["name"].as_str() == Some("skill:dup"))
        .count();
    assert_eq!(
        dup_count, 1,
        "the shadowed name must appear once: {commands:?}"
    );
    // `collisions` is a structured `Collision` list (skills.rs's compile-fix change), not a flat
    // string array — `name`/`resource_type`/`winner_path`/`loser_path` are each their own field, so a
    // client can build real tooling on top of a collision instead of scraping a sentence apart.
    let collisions = data["collisions"].as_array().unwrap();
    let dup_collision = collisions
        .iter()
        .find(|c| c["name"] == "dup")
        .unwrap_or_else(|| panic!("collisions must report the shadowed skill: {collisions:?}"));
    assert_eq!(dup_collision["resource_type"], "skill");
    assert!(
        dup_collision["winner_path"]
            .as_str()
            .is_some_and(|p| p.contains(&dir.path().to_string_lossy().to_string())),
        "the project copy must win: {dup_collision:?}"
    );
    assert!(
        dup_collision["loser_path"]
            .as_str()
            .is_some_and(|p| p.contains(&home.path().to_string_lossy().to_string())),
        "the user copy must be reported as shadowed: {dup_collision:?}"
    );
    assert!(
        dup_collision["message"]
            .as_str()
            .is_some_and(|s| s.contains("dup")),
        "the human-readable message must still be populated: {dup_collision:?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_commands_reports_scope_and_path_per_entry() {
    // Task #39 (pi-parity fix): `get_commands` used to report only `{name, source, description}` per
    // entry — pi's own real headless RPC protocol also carries `sourceInfo.scope`/`sourceInfo.path`, so
    // a client can tell a user-global skill apart from a project-local one (and where either actually
    // lives on disk), not just its name.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();

    let user_skill_dir = home.path().join(".claude/skills/mine");
    std::fs::create_dir_all(&user_skill_dir).unwrap();
    std::fs::write(
        user_skill_dir.join("SKILL.md"),
        "---\nname: mine\ndescription: a user-global skill\n---\nBody.",
    )
    .unwrap();
    let project_skill_dir = dir.path().join(".claude/skills/theirs");
    std::fs::create_dir_all(&project_skill_dir).unwrap();
    std::fs::write(
        project_skill_dir.join("SKILL.md"),
        "---\nname: theirs\ndescription: a project-local skill\n---\nBody.",
    )
    .unwrap();

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let (base, _bodies) = spawn_model_server(vec![]);
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project")
        .current_dir(dir.path())
        .env("HOME", home.path());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap();

    let mine = commands
        .iter()
        .find(|c| c["name"] == "skill:mine")
        .unwrap_or_else(|| panic!("expected the user-global skill: {commands:#?}"));
    assert_eq!(mine["scope"], "user", "{mine:#?}");
    assert!(
        mine["path"]
            .as_str()
            .is_some_and(|p| p.contains(&home.path().to_string_lossy().to_string())),
        "the user skill's path must point under HOME: {mine:#?}"
    );

    let theirs = commands
        .iter()
        .find(|c| c["name"] == "skill:theirs")
        .unwrap_or_else(|| panic!("expected the project-local skill: {commands:#?}"));
    assert_eq!(theirs["scope"], "project", "{theirs:#?}");
    assert!(
        theirs["path"]
            .as_str()
            .is_some_and(|p| p.contains(&dir.path().to_string_lossy().to_string())),
        "the project skill's path must point under the project dir: {theirs:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_commands_reports_the_prompt_templates_own_description_not_its_argument_hint() {
    // A prompt template's `get_commands` entry must surface its `description:` frontmatter, not the
    // separate `argument-hint:` field — the two are deliberately distinct (a hint like "<file>" isn't
    // a human-readable summary a client's command palette should display as one).
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(
        prompt_dir.join("fix.md"),
        "---\nargument-hint: <file>\ndescription: Fix a bug in the given file\n---\nFix $1.",
    )
    .unwrap();

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project")
        .current_dir(dir.path())
        .env("HOME", home.path());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap();
    let fix = commands
        .iter()
        .find(|c| c["name"].as_str() == Some("fix"))
        .unwrap_or_else(|| panic!("prompt template \"fix\" not listed: {commands:?}"));
    assert_eq!(
        fix["description"].as_str(),
        Some("Fix a bug in the given file"),
        "description field must be the template's own description, not its argument hint: {fix:?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_warns_on_stderr_when_an_untrusted_projects_gated_resources_are_skipped() {
    // Track L32 (pi-parity fix): mirrors `run`'s identical fix
    // (`run_binary_warns_on_stderr_when_an_untrusted_projects_gated_resources_are_skipped`,
    // `run_skills_prompts.rs`) — an untrusted project with a `SYSTEM.md`/skills/prompts on disk
    // previously skipped all of them with no signal at all that anything was there. No
    // `--trust-project` here at all: the project-local `SYSTEM.md` exists but is never honored, and
    // the warning must fire on `serve`'s startup path too, not just `run`'s.
    let dir = tempfile::tempdir().unwrap();
    let claude_dir = dir.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(claude_dir.join("SYSTEM.md"), "SHOULD-NOT-APPLY").unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.current_dir(dir.path());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(
        stderr.contains("warning:") && stderr.contains("isn't trusted"),
        "must warn on stderr that gated resources were found but skipped: {stderr}"
    );
}
