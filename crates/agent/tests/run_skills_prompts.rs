//! `run` e2e: Skill/prompt-template expansion in a one-shot `run`, trust gating, ad-hoc discovery roots, project instructions.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text};

#[test]
fn run_binary_expands_a_skill_invocation_in_the_first_message() {
    // `serve` has always expanded `/skill:name`/`/name` invocations before sending them to the model;
    // `run` (this one-shot binary) previously did not — a message starting with either was sent as a
    // literal, unexpanded string. `--trust-project` is required for the project-local skill to be
    // discovered at all (skill discovery is trust-gated; the untrusted-by-default case is covered by
    // `serve_e2e.rs`'s own trust tests, not duplicated here).
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("SKILL-BODY-MARKER-123"),
        "the skill's body must be expanded into the first message: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("/skill:greet"),
        "the raw, unexpanded invocation must not reach the model: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_force_untrusted_overrides_trust_project() {
    // Track L8: `--force-untrusted` must win even when `--trust-project` is *also* given — the whole
    // point is a way to force the untrusted codepath for one run regardless of anything else asking
    // for trust.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--force-untrusted",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("SKILL-BODY-MARKER-123"),
        "--force-untrusted must override --trust-project, so the skill must not expand: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_expands_a_prompt_template_in_the_first_message() {
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(
        prompt_dir.join("fix.md"),
        "Fix the bug in $1 — TEMPLATE-BODY-MARKER-456",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/fix foo.rs",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("TEMPLATE-BODY-MARKER-456") && bodies[0].contains("foo.rs"),
        "the template's body, with its argument substituted, must reach the model: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("/fix foo.rs"),
        "the raw, unexpanded invocation must not reach the model: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_no_skills_leaves_a_skill_invocation_unexpanded() {
    // Same fixture as `run_binary_expands_a_skill_invocation_in_the_first_message`, but with
    // `--no-skills` — the skill must not be discovered at all, so `/skill:greet` reaches the model
    // as a literal, unexpanded string instead of the skill's body.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-skills",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("SKILL-BODY-MARKER-123"),
        "--no-skills must prevent the skill from being discovered/expanded at all: {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("/skill:greet"),
        "the raw invocation must reach the model unexpanded: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_no_skills_still_honors_an_explicit_skill_path() {
    // pi-parity fix (M2): pi's own `noSkills` still loads an explicit `--skill <path>` passed
    // alongside it (`resource-loader.test.ts`, "should still load additional skill paths when
    // noSkills is true" — a documented, tested combination). `--no-skills` only suppresses the two
    // *standard* roots (`~/.claude/skills`, `<cwd>/.claude/skills`); it must not also discard an
    // operator-supplied `--skill` extra root, unlike the plain `--no-skills` case above (that fixture
    // lives under the *standard* project root, `.claude/skills`, not an ad-hoc `--skill` path).
    let dir = tempfile::tempdir().unwrap();
    let extra_skill_dir = dir.path().join("shared-skills/greet");
    std::fs::create_dir_all(&extra_skill_dir).unwrap();
    std::fs::write(
        extra_skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: an ad-hoc skill\n---\nAD-HOC-SKILL-MARKER-789",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-skills",
            "--skill",
            dir.path().join("shared-skills").to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("AD-HOC-SKILL-MARKER-789"),
        "--no-skills must still honor an explicit --skill path: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_no_context_files_skips_project_instructions() {
    // MEDIUM pi-parity gap (fixed): `run` hardcoded `include_context_files: true` with no way to opt
    // out, unlike `serve`'s `--no-context-files`. A `CLAUDE.md` in cwd must be injected by default, but
    // must be skipped entirely when `--no-context-files` is passed.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("CLAUDE.md"), "PROJECT-MARKER-777").unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-context-files",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("PROJECT-MARKER-777"),
        "--no-context-files must prevent CLAUDE.md from being injected at all: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_injects_project_instructions_by_default() {
    // The other half of the previous test: without the flag, CLAUDE.md must still be injected —
    // proving `--no-context-files` genuinely toggles behavior rather than the marker just never having
    // been reachable.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("CLAUDE.md"), "PROJECT-MARKER-778").unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("PROJECT-MARKER-778"),
        "CLAUDE.md must be injected by default: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_no_prompt_templates_leaves_a_template_invocation_unexpanded() {
    // Same fixture as `run_binary_expands_a_prompt_template_in_the_first_message`, but with
    // `--no-prompt-templates` — the template must not be discovered at all, so `/fix foo.rs` reaches
    // the model as a literal, unexpanded string instead of the template's body.
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(
        prompt_dir.join("fix.md"),
        "Fix the bug in $1 — TEMPLATE-BODY-MARKER-456",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/fix foo.rs",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-prompt-templates",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("TEMPLATE-BODY-MARKER-456"),
        "--no-prompt-templates must prevent the template from being discovered/expanded at all: {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("/fix foo.rs"),
        "the raw invocation must reach the model unexpanded: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_discovers_skills_from_an_ad_hoc_skill_path() {
    // pi: coding-agent/skills.test.ts, "should load from explicit skillPaths" — `--skill <path>`
    // (repeatable), a discovery root beyond the two standard ones. Deliberately no `--trust-project`:
    // an operator-supplied CLI path is an explicit, deliberate choice, unlike auto-discovered
    // project-local content, so it isn't trust-gated the way `.claude/skills` in the repo itself is.
    let dir = tempfile::tempdir().unwrap();
    let extra_skill_dir = dir.path().join("shared-skills/greet");
    std::fs::create_dir_all(&extra_skill_dir).unwrap();
    std::fs::write(
        extra_skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: an ad-hoc skill\n---\nAD-HOC-SKILL-MARKER-456",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--skill",
            dir.path().join("shared-skills").to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("AD-HOC-SKILL-MARKER-456"),
        "the ad-hoc skill's body must be expanded into the first message: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_prompt_guideline_flag_reaches_the_system_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--prompt-guideline",
            "Always run tests before finishing.",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("Always run tests before finishing."),
        "the extra guideline must reach the system prompt: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_system_prompt_flag_replaces_the_base_prompt() {
    // Pi-parity audit H58: `run` had no way to override the agent's identity at all — `serve` already
    // had `--system-prompt`, but a one-shot invocation (e.g. a specialized reviewer/persona for
    // automation) had no equivalent.
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--system-prompt",
            "CUSTOM-PERSONA-MARKER: you are a terse code reviewer.",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("CUSTOM-PERSONA-MARKER"),
        "--system-prompt must replace the system prompt sent on the wire: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("You are the Beyond coding agent"),
        "the built-in base prompt must be replaced entirely, not merely appended to: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_append_system_prompt_flag_appends_after_the_base_prompt() {
    // Pi-parity audit M60: `run` had `--system-prompt` (replace) but no way to layer *extra*
    // instructions on top of the built-in base without discarding it entirely.
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--append-system-prompt",
            "APPENDED-INSTRUCTION-MARKER: always cite line numbers.",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("APPENDED-INSTRUCTION-MARKER"),
        "--append-system-prompt must reach the wire system prompt: {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("You are the Beyond coding agent"),
        "the built-in base prompt must still be present, not replaced: {}",
        bodies[0]
    );
}
