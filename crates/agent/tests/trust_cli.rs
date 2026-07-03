//! `agent trust`/`untrust`/`clear-trust`/`trust-status` CLI subcommands: the out-of-band,
//! `~/.claude/trusted-projects.json`-backed allowlist management surface, exercised directly (not just
//! indirectly via `serve`'s own trust-gating tests in `serve_trust_prompts.rs`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::Command;

#[test]
fn trust_status_reports_unknown_for_a_never_mentioned_directory() {
    // Pi-parity audit: `TrustStore::lookup`'s tri-state result was fully built and used internally,
    // but had no CLI/RPC surface at all to actually query it.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["trust-status", dir.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("unknown:"),
        "a never-mentioned directory must report unknown: {stdout}"
    );
    assert!(stdout.contains(dir.path().to_str().unwrap()), "{stdout}");
}

#[test]
fn trust_status_reflects_trust_and_untrust_round_trip() {
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let path = dir.path().to_str().unwrap();

    let status = |args: &[&str]| -> String {
        let output = Command::new(bin)
            .args(args)
            .env("HOME", home.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).into_owned()
    };

    assert!(status(&["trust-status", path]).starts_with("unknown:"));

    let trust_output = Command::new(bin)
        .args(["trust", path])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust_output.status.success());
    assert!(status(&["trust-status", path]).starts_with("trusted:"));

    let untrust_output = Command::new(bin)
        .args(["untrust", path])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(untrust_output.status.success());
    assert!(status(&["trust-status", path]).starts_with("untrusted:"));

    let clear_output = Command::new(bin)
        .args(["clear-trust", path])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(clear_output.status.success());
    assert!(status(&["trust-status", path]).starts_with("unknown:"));
}

#[test]
fn trust_status_inherits_a_trusted_ancestors_grant_for_an_unmentioned_child() {
    // `TrustStore::lookup` walks ancestors, returning the nearest explicit entry — this proves
    // `trust-status` surfaces that inheritance, not just an exact-path lookup.
    let home = tempfile::tempdir().unwrap();
    let parent = tempfile::tempdir().unwrap();
    let child = parent.path().join("child-project");
    std::fs::create_dir_all(&child).unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let trust_output = Command::new(bin)
        .args(["trust", parent.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust_output.status.success());

    let output = Command::new(bin)
        .args(["trust-status", child.to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("trusted:"),
        "a child of a trusted directory must inherit that trust: {stdout}"
    );
}

#[test]
fn trust_status_defaults_to_the_current_directory_when_no_path_is_given() {
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .arg("trust-status")
        .current_dir(dir.path())
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("unknown:"), "{stdout}");
}
