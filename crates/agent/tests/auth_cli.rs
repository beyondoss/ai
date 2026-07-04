//! `agent login`/`logout`/`auth-status` CLI subcommands: the out-of-band, `~/.claude/auth.json`-backed
//! OAuth/subscription-credential store, exercised directly (not just indirectly via `serve`'s own RPC
//! surface in `serve_auth.rs`). Follows `trust_cli.rs`'s exact shape — `login` itself needs a live
//! OAuth provider (or a mock one) to actually complete, so these tests cover everything reachable
//! without a network: unknown-provider handling, `logout`/`auth-status` idempotency, and status
//! reporting against a hand-seeded `auth.json` fixture.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::Command;

fn write_auth_json(home: &std::path::Path, body: &str) {
    let dir = home.join(".claude");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("auth.json"), body).unwrap();
}

#[test]
fn auth_status_with_no_file_reports_every_provider_logged_out() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .arg("auth-status")
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("anthropic: logged_out"), "{stdout}");
    assert!(stdout.contains("github-copilot: logged_out"), "{stdout}");
    assert!(stdout.contains("openai-codex: logged_out"), "{stdout}");
}

#[test]
fn auth_status_for_an_unknown_provider_is_an_error() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["auth-status", "not-a-real-provider"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown provider"), "{stderr}");
}

#[test]
fn login_for_an_unknown_provider_is_an_error() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["login", "not-a-real-provider"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown provider"), "{stderr}");
}

#[test]
fn logout_of_a_never_logged_in_provider_is_idempotent() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["logout", "anthropic"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("not logged in:"), "{stdout}");
}

#[test]
fn auth_status_reports_logged_in_for_a_non_expired_stored_credential() {
    let home = tempfile::tempdir().unwrap();
    let far_future_ms = 4_000_000_000_000i64; // year ~2096, always "not expired" for this test
    write_auth_json(
        home.path(),
        &format!(
            r#"{{"anthropic":{{"type":"oauth","access":"tok","refresh":"ref","expires":{far_future_ms}}}}}"#
        ),
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["auth-status", "anthropic"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("anthropic: logged_in"), "{stdout}");
}

#[test]
fn auth_status_reports_needs_reauth_when_the_last_refresh_failed() {
    let home = tempfile::tempdir().unwrap();
    write_auth_json(
        home.path(),
        r#"{"anthropic":{"type":"oauth","access":"tok","refresh":"ref","expires":1,
            "last_refresh_error":"network unreachable"}}"#,
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["auth-status", "anthropic"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("anthropic: needs_reauth"), "{stdout}");
}

#[test]
fn logout_removes_a_stored_credential_and_reports_it_was_present() {
    let home = tempfile::tempdir().unwrap();
    write_auth_json(
        home.path(),
        r#"{"anthropic":{"type":"oauth","access":"tok","refresh":"ref","expires":4000000000000}}"#,
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["logout", "anthropic"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("logged out:"), "{stdout}");

    // A second logout is now a no-op idempotent success, not an error.
    let second = Command::new(bin)
        .args(["logout", "anthropic"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(second.status.success());
    assert!(
        String::from_utf8_lossy(&second.stdout).starts_with("not logged in:")
    );
}

#[test]
fn ai_agent_config_dir_env_var_redirects_the_auth_store_off_of_home() {
    // Matches `settings_store.rs`'s identical coverage for `SettingsStore`/`TrustStore` — `AuthStore`
    // shares the same `config_dir_root()` helper and must honor the same override.
    let home = tempfile::tempdir().unwrap();
    let config_dir = tempfile::tempdir().unwrap();
    write_auth_json(
        config_dir.path(),
        r#"{"anthropic":{"type":"oauth","access":"tok","refresh":"ref","expires":4000000000000}}"#,
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = Command::new(bin)
        .args(["auth-status", "anthropic"])
        .env("HOME", home.path())
        .env("AI_AGENT_CONFIG_DIR", config_dir.path().join(".claude"))
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("anthropic: logged_in"), "{stdout}");

    // And HOME's own `.claude/auth.json` (never written) must NOT have been consulted.
    assert!(!home.path().join(".claude/auth.json").exists());
}
