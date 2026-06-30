//! Live smoke: the real `beyond-ai-agent` binary → the real `beyond-ai` gateway → **real Anthropic**.
//!
//! Ignored by default (bills a tiny real request). Run with a key present:
//!
//!   ANTHROPIC_API_KEY=sk-ant-… mise run test:smoke:agent
//!   ANTHROPIC_API_KEY=sk-ant-… cargo test -p beyond-ai-agent --test smoke -- --ignored --nocapture
//!
//! This is the one test that validates the dialect decoder against a *real* provider's SSE: the
//! gateway boots with the caller's Anthropic key as the managed pool key (and the dev signing key),
//! then the agent — holding only a `bai_v1` virtual key — drives a real tool round-trip through it.
//! A model-not-found is a stale model id, not a harness bug (adjust `MODEL`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{DEV_PUBKEY_B64, DEV_TOKEN, free_port, gateway_bin, wait_for_port};

/// Cheapest small Anthropic model (matches the gateway's own smoke test).
const MODEL: &str = "claude-haiku-4-5";

fn env_key(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with ANTHROPIC_API_KEY set"]
fn smoke_agent_through_gateway_to_real_anthropic() {
    let Some(anthropic_key) = env_key("ANTHROPIC_API_KEY") else {
        eprintln!("smoke[agent]: ANTHROPIC_API_KEY unset — skipping");
        return;
    };

    // A file the agent must read with its `read` tool, then echo back — proves a live tool round-trip.
    let dir = tempfile::tempdir().unwrap();
    let token = "PINEAPPLE-7493";
    std::fs::write(dir.path().join("marker.txt"), format!("{token}\n")).unwrap();

    // Gateway → REAL Anthropic (no authority override, real TLS upstream). Managed pool key = the
    // caller's real key; the agent presents only the dev virtual key.
    let gw_port = free_port();
    let metrics_port = free_port();
    let config = format!(
        "listen = \"127.0.0.1:{gw_port}\"\n\
         metrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:59321\"\n\
         config_bucket = \"ai-gateway\"\n\
         upstream_tls = true\n\
         \n[pool_keys]\nanthropic = \"{anthropic_key}\"\n\
         \n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"
    );
    let config_path = dir.path().join("gateway.toml");
    std::fs::write(&config_path, config).unwrap();

    let mut gateway = Command::new(gateway_bin())
        .arg("run")
        .arg("-c")
        .arg(&config_path)
        .env("AI_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn gateway");
    wait_for_port(gw_port);

    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "Use the read tool to read the file marker.txt in the current directory, then reply with ONLY the exact token it contains.",
            "--gateway-url",
            &format!("http://127.0.0.1:{gw_port}"),
            "--key",
            DEV_TOKEN,
            "--model",
            MODEL,
            "--max-steps",
            "6",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn agent");

    let _ = gateway.kill();
    let _ = gateway.wait();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("--- agent stdout ---\n{stdout}\n--- agent stderr ---\n{stderr}");
    assert!(
        output.status.success(),
        "agent failed against real Anthropic.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains(token),
        "real Claude should have read the file via the tool and echoed `{token}`.\nstdout: {stdout}"
    );
}
