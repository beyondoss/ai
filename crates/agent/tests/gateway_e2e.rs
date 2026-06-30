//! The full stack: the real `beyond-ai-agent` binary → the real `beyond-ai` gateway binary →
//! a scripted mock upstream.
//!
//! This is the end-to-end proof that the harness works *through the actual gateway*: the agent
//! authenticates with a `bai_v1` virtual key, the gateway verifies it, swaps in the pool key, routes
//! to the (mock) provider, and relays the streamed tool round-trip back. No NATS (the gateway fails
//! open), no real provider, no TLS (plaintext upstream) — but every byte flows through the gateway.
//!
//! The signing key is the gateway's deterministic dev key (`mise run ai:mint-dev-key`), so the
//! public key + token are fixed constants — no dependency on the gateway crate.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::PathBuf;
use std::process::{Command, Stdio};

use common::{free_port, spawn_model_server, turn_text, turn_tool_use, wait_for_port};
use serde_json::json;

/// Deterministic dev signing public key (standard base64), for `[signing_keys] 1 = …`.
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
/// The matching dev `bai_v1` token (tenant 1 / vpc 1, kid 1).
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// Locate the gateway binary (built beside the agent binary); build it on demand if absent.
fn gateway_bin() -> PathBuf {
    let agent = PathBuf::from(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    let dir = agent.parent().unwrap();
    let gw = dir.join("beyond-ai");
    if !gw.exists() {
        let mut args = vec!["build", "-q", "-p", "beyond-ai", "--bin", "beyond-ai"];
        if agent.to_string_lossy().contains("/release/") {
            args.push("--release");
        }
        let status = Command::new(env!("CARGO"))
            .args(&args)
            .status()
            .expect("build gateway");
        assert!(status.success(), "failed to build the gateway binary");
    }
    gw
}

#[test]
fn agent_through_real_gateway_to_mock_upstream() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-99\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();

    // Scripted upstream: the model calls `read`, then replies and ends.
    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("Read complete.");
    let (mock_base, requests) = spawn_model_server(vec![turn1, turn2]);
    let mock_authority = mock_base.strip_prefix("http://").unwrap().to_string();

    // Gateway config: Anthropic provider → the mock (plaintext), pool key to swap in, dev signing key.
    let gw_port = free_port();
    let metrics_port = free_port();
    let config = format!(
        "listen = \"127.0.0.1:{gw_port}\"\n\
         metrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:59321\"\n\
         config_bucket = \"ai-gateway\"\n\
         upstream_tls = false\n\
         \n[provider_authorities]\nanthropic = \"{mock_authority}\"\n\
         \n[pool_keys]\nanthropic = \"sk-pool-secret\"\n\
         \n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"
    );
    let config_path = dir.path().join("gateway.toml");
    std::fs::write(&config_path, config).unwrap();

    // Boot the real gateway binary.
    let mut gateway = Command::new(gateway_bin())
        .arg("run")
        .arg("-c")
        .arg(&config_path)
        .env("AI_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn gateway");
    wait_for_port(gw_port);

    // Drive the agent through the gateway.
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "read hello.txt and report it",
            "--gateway-url",
            &format!("http://127.0.0.1:{gw_port}"),
            "--key",
            DEV_TOKEN,
            "--model",
            "claude-test",
            "--max-steps",
            "4",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn agent");

    let _ = gateway.kill();
    let _ = gateway.wait(); // reap the child so it doesn't linger as a zombie

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "agent failed through gateway.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("[tool: read]"),
        "expected the tool round-trip.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains("Read complete."),
        "expected the final turn.\nstdout: {stdout}"
    );

    // The gateway relayed two upstream requests, swapped the key, and never leaked the virtual key.
    let reqs = requests.lock().unwrap();
    assert_eq!(reqs.len(), 2, "gateway should relay two upstream requests");
    let all = reqs.join("\n---\n");
    assert!(
        all.contains("sk-pool-secret"),
        "gateway must swap in the pool key.\n{all}"
    );
    assert!(
        !all.contains("bai_v1.1."),
        "the bai_v1 virtual key must NOT reach the upstream.\n{all}"
    );
    assert!(
        reqs[1].contains("secret-marker-99"),
        "the tool result must be fed back to the model through the gateway.\n{}",
        reqs[1]
    );
}
