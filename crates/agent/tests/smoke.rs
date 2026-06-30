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

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

use common::{DEV_PUBKEY_B64, DEV_TOKEN, free_port, gateway_bin, wait_for_port};
use serde_json::{Value, json};

/// Cheapest small Anthropic model (matches the gateway's own smoke test).
const MODEL: &str = "claude-haiku-4-5";
/// A model whose prompt-cache activates at a ~3k-token prefix — `claude-haiku-4-5` has a higher
/// cache-activation threshold, so the cache test uses this to validate hits end-to-end.
const CACHING_MODEL: &str = "claude-sonnet-4-5";

fn env_key(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Boot the real gateway in `dir`, fronting REAL Anthropic with `anthropic_key` as the managed pool
/// key (and the dev signing key). Returns its port and the child handle (kill it when done).
fn boot_gateway(dir: &Path, anthropic_key: &str) -> (u16, Child) {
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
    let config_path = dir.join("gateway.toml");
    std::fs::write(&config_path, config).unwrap();
    let gateway = Command::new(gateway_bin())
        .arg("run")
        .arg("-c")
        .arg(&config_path)
        .env("AI_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn gateway");
    wait_for_port(gw_port);
    (gw_port, gateway)
}

/// Read stdout frames until the `response` for `command` arrives; return all frames seen.
fn read_until_response(reader: &mut impl BufRead, command: &str) -> Vec<Value> {
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some(command);
        frames.push(v);
        if done {
            break;
        }
    }
    frames
}

/// Spawn the agent `serve` binary against the gateway, with extra args appended.
fn serve_child(gw_port: u16, cwd: &Path, model: &str, extra: &[&str]) -> Child {
    let mut args = vec![
        "serve".to_string(),
        "--gateway-url".into(),
        format!("http://127.0.0.1:{gw_port}"),
        "--key".into(),
        DEV_TOKEN.into(),
        "--model".into(),
        model.into(),
        "--max-steps".into(),
        "6".into(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(&args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn agent serve")
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

/// Multimodal end-to-end: the `read` tool returns a real image as an attachment, the dialect encodes
/// it into Anthropic's `tool_result` content-array shape, and **real Claude vision** sees it. The agent
/// reads a solid-red PNG and must name the color — exercising the whole new image path live (read
/// image detection → `ToolOutput.images` → `ContentBlock::ToolResult.images` → wire encoding → vision).
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with ANTHROPIC_API_KEY set"]
fn smoke_reads_an_image_and_describes_it() {
    use base64::Engine as _;

    let Some(key) = env_key("ANTHROPIC_API_KEY") else {
        eprintln!("smoke[image]: ANTHROPIC_API_KEY unset — skipping");
        return;
    };
    // A 48x48 solid-red PNG (generated deterministically; no image crate needed).
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAANklEQVR42u3OQQ0AAAgAoetfWls4H2wEoKlXEhISEhISEhISEhISEhISEhISEhISEhISEhK6s98T93mKDkyKAAAAAElFTkSuQmCC";
    let dir = tempfile::tempdir().unwrap();
    let png = base64::engine::general_purpose::STANDARD
        .decode(RED_PNG_B64)
        .unwrap();
    std::fs::write(dir.path().join("swatch.png"), png).unwrap();

    let (gw_port, mut gateway) = boot_gateway(dir.path(), &key);
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "Use the read tool to read the image file swatch.png in the current directory, then reply with ONLY the single dominant color word you see.",
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
    eprintln!("--- image stdout ---\n{stdout}\n--- image stderr ---\n{stderr}");
    assert!(
        output.status.success(),
        "agent failed reading an image through the multimodal path.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.to_lowercase().contains("red"),
        "real Claude vision should have seen a red swatch via the read-tool image path.\nstdout: {stdout}"
    );
}

/// Prompt caching actually *hits* against real Anthropic — not just that the body is accepted. A
/// read-tool round-trip is two model turns: turn 1 writes the prefix cache, turn 2 (re-sending that
/// prefix) reads it. The `prompt` response's usage must show both a cache write and a cache read.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with ANTHROPIC_API_KEY set"]
fn smoke_prompt_cache_produces_hits() {
    let Some(key) = env_key("ANTHROPIC_API_KEY") else {
        eprintln!("smoke[cache]: ANTHROPIC_API_KEY unset — skipping");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), "PINEAPPLE-7493\n").unwrap();
    let (gw_port, mut gateway) = boot_gateway(dir.path(), &key);
    let session_file = dir.path().join("s.jsonl");

    let mut child = serve_child(
        gw_port,
        dir.path(),
        CACHING_MODEL,
        &["--session-file", session_file.to_str().unwrap()],
    );
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "Use the read tool to read marker.txt in the current directory, then reply with ONLY the token it contains." })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    let _ = gateway.kill();
    let _ = gateway.wait();
    let _ = child.wait();

    let resp = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("a prompt response");
    let data = &resp["data"];
    eprintln!("--- cache usage ---\n{data}");
    assert_eq!(resp["success"], true, "the run should succeed: {resp}");
    let cache_write = data["cache_write_tokens"].as_u64().unwrap_or(0);
    let cache_read = data["cache_read_tokens"].as_u64().unwrap_or(0);
    assert!(
        cache_write > 0,
        "turn 1 should have *written* the prompt cache (cache_write_tokens > 0): {data}"
    );
    assert!(
        cache_read > 0,
        "turn 2 should have *read* the prompt cache — a real hit (cache_read_tokens > 0): {data}"
    );
}

/// Extended thinking + tools against real Anthropic. This is the correctness landmine: turn 2's
/// request must replay turn 1's *signed* thinking block, or Anthropic 400s. A successful multi-turn
/// tool round-trip with `--thinking` on proves the signature round-trips intact.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with ANTHROPIC_API_KEY set"]
fn smoke_thinking_with_tools_replays_signature() {
    let Some(key) = env_key("ANTHROPIC_API_KEY") else {
        eprintln!("smoke[thinking]: ANTHROPIC_API_KEY unset — skipping");
        return;
    };
    let token = "PINEAPPLE-7493";
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), format!("{token}\n")).unwrap();
    let (gw_port, mut gateway) = boot_gateway(dir.path(), &key);

    let mut child = serve_child(gw_port, dir.path(), MODEL, &["--thinking", "2000"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "Use the read tool to read marker.txt in the current directory, then reply with ONLY the token it contains." })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    // Pull the transcript to confirm the token was echoed (and a thinking block was produced).
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let msg_frames = read_until_response(&mut stdout, "get_messages");
    drop(stdin);
    let _ = gateway.kill();
    let _ = gateway.wait();
    let _ = child.wait();

    let resp = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("a prompt response");
    let dump = msg_frames.last().unwrap()["data"]["messages"].to_string();
    eprintln!("--- thinking transcript ---\n{dump}");
    // Success here means the turn-2 request (carrying the replayed signed thinking block) was accepted.
    assert_eq!(
        resp["success"], true,
        "thinking+tools must not 400 on signature replay: {resp}"
    );
    assert!(
        dump.contains(token),
        "the agent should have read the file and echoed `{token}` with thinking on: {dump}"
    );
    assert!(
        dump.contains("\"thinking\""),
        "the transcript should contain a thinking block (proves replay was exercised): {dump}"
    );
}
