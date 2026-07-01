//! Live smoke: the real `beyond-ai-agent` binary → the real `beyond-ai` gateway → a **real provider**.
//!
//! Ignored by default (bills tiny real requests). The gateway routes a bare `/v1/messages` to
//! Anthropic and `/v1/chat/completions` to OpenAI by dialect, so one gateway with the right pool key
//! serves either — which lets the provider-agnostic tests run the *same* body against **every provider
//! whose key is present** (see [`PROVIDERS`] / [`available`]). Each test skips a provider whose key is
//! unset rather than failing. Run with whatever keys you have:
//!
//!   mise run test:smoke:agent          # auto-loads .env (ANTHROPIC_API_KEY, OPENAI_API_KEY, …)
//!   ANTHROPIC_API_KEY=… OPENAI_API_KEY=… cargo test -p beyond-ai-agent --test smoke -- --ignored --nocapture
//!
//! These validate the dialect decoders against *real* provider SSE: the gateway boots with the
//! caller's key as the managed pool key (and the dev signing key), then the agent — holding only a
//! `bai_v1` virtual key — drives real round-trips through it. A model-not-found is a stale model id,
//! not a harness bug (adjust the provider's `model`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};

use common::{DEV_PUBKEY_B64, DEV_TOKEN, free_port, gateway_bin, wait_for_port};
use serde_json::{Value, json};

/// A live provider profile: the env var holding its key, the gateway pool-key name (= the provider
/// path segment the dialect routes to), and a cheap small model that is both tool- and vision-capable.
struct Provider {
    /// Human label for skip/assert messages.
    name: &'static str,
    /// Env var holding the real upstream key.
    env: &'static str,
    /// Gateway pool-key name = provider path segment (`anthropic` → `/v1/messages`, `openai` →
    /// `/v1/chat/completions`).
    pool: &'static str,
    /// A cheap small chat model on this provider (tool- and vision-capable).
    model: &'static str,
}

/// The providers the shared (provider-agnostic) smokes run against. Add a row to cover a new provider.
const PROVIDERS: &[Provider] = &[
    Provider {
        name: "anthropic",
        env: "ANTHROPIC_API_KEY",
        pool: "anthropic",
        model: "claude-haiku-4-5",
    },
    Provider {
        name: "openai",
        env: "OPENAI_API_KEY",
        pool: "openai",
        model: "gpt-4o-mini",
    },
];

/// A model whose prompt-cache activates at a ~3k-token prefix — `claude-haiku-4-5` has a higher
/// cache-activation threshold, so the (Anthropic-specific) cache test uses this to validate hits.
const CACHING_MODEL: &str = "claude-sonnet-4-5";

fn env_key(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// The providers whose key is set; the rest are skipped (not failed) by the shared tests.
fn available() -> Vec<&'static Provider> {
    PROVIDERS
        .iter()
        .filter(|p| env_key(p.env).is_some())
        .collect()
}

/// Boot the real gateway in `dir`, fronting a REAL upstream with `key` as the managed pool key for
/// `pool` (and the dev signing key). Returns its port and the child handle (kill it when done).
fn boot_gateway(dir: &Path, pool: &str, key: &str) -> (u16, Child) {
    let gw_port = free_port();
    let metrics_port = free_port();
    let config = format!(
        "listen = \"127.0.0.1:{gw_port}\"\n\
         metrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:59321\"\n\
         config_bucket = \"ai-gateway\"\n\
         upstream_tls = true\n\
         \n[pool_keys]\n{pool} = \"{key}\"\n\
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

/// Boot a gateway for `provider` and run the agent `run` one-shot with `prompt` in `cwd`. Returns the
/// process output (the gateway is killed before returning). The caller must have confirmed the key is
/// set (via [`available`]).
fn run_oneshot(provider: &Provider, cwd: &Path, prompt: &str) -> Output {
    let key = env_key(provider.env).expect("provider key present");
    let (gw_port, mut gateway) = boot_gateway(cwd, provider.pool, &key);
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            prompt,
            "--gateway-url",
            &format!("http://127.0.0.1:{gw_port}"),
            "--key",
            DEV_TOKEN,
            "--model",
            provider.model,
            "--max-steps",
            "6",
        ])
        .current_dir(cwd)
        .output()
        .expect("spawn agent");
    let _ = gateway.kill();
    let _ = gateway.wait();
    output
}

/// Shared across providers: a live tool round-trip. The agent reads a file with its `read` tool and
/// echoes the token back — proving tool-calling and the dialect decoder against each real provider.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_tool_round_trip() {
    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[tool]: no provider key set — skipping");
        return;
    }
    for p in providers {
        let dir = tempfile::tempdir().unwrap();
        let token = "PINEAPPLE-7493";
        std::fs::write(dir.path().join("marker.txt"), format!("{token}\n")).unwrap();

        let output = run_oneshot(
            p,
            dir.path(),
            "Use the read tool to read the file marker.txt in the current directory, then reply with ONLY the exact token it contains.",
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "--- [{}] tool stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            p.name
        );
        assert!(
            output.status.success(),
            "[{}] agent run failed.\nstdout: {stdout}\nstderr: {stderr}",
            p.name
        );
        assert!(
            stdout.contains(token),
            "[{}] should have read the file via the tool and echoed `{token}`.\nstdout: {stdout}",
            p.name
        );
    }
}

/// Shared across providers: multimodal end-to-end. The `read` tool returns a real image as an
/// attachment; the dialect encodes it (Anthropic content-array / OpenAI `image_url` fan-out) and the
/// real provider's **vision** sees it. The agent reads a solid-red PNG and must name the colour —
/// exercising the whole image path live (read image detection → `ToolOutput.images` →
/// `ContentBlock::ToolResult.images` → wire encoding → vision). This is the only live coverage of the
/// OpenAI-wire image path, which previously dropped images on the floor.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_reads_an_image_and_describes_it() {
    use base64::Engine as _;

    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[image]: no provider key set — skipping");
        return;
    }
    // A 48x48 solid-red PNG (generated deterministically; no image crate needed).
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAANklEQVR42u3OQQ0AAAgAoetfWls4H2wEoKlXEhISEhISEhISEhISEhISEhISEhISEhISEhK6s98T93mKDkyKAAAAAElFTkSuQmCC";
    let png = base64::engine::general_purpose::STANDARD
        .decode(RED_PNG_B64)
        .unwrap();

    for p in providers {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("swatch.png"), &png).unwrap();

        let output = run_oneshot(
            p,
            dir.path(),
            "Use the read tool to read the image file swatch.png in the current directory, then reply with ONLY the single dominant color word you see.",
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "--- [{}] image stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            p.name
        );
        assert!(
            output.status.success(),
            "[{}] agent failed reading an image through the multimodal path.\nstdout: {stdout}\nstderr: {stderr}",
            p.name
        );
        assert!(
            stdout.to_lowercase().contains("red"),
            "[{}] real vision should have seen a red swatch via the read-tool image path.\nstdout: {stdout}",
            p.name
        );
    }
}

/// Anthropic-specific: prompt caching actually *hits* — not just that the body is accepted. (Anthropic
/// caches via the explicit breakpoints we stamp and reports cache_write/cache_read; OpenAI's automatic
/// prefix caching only kicks in past ~1k-token prefixes and reports only cached reads, so it isn't
/// asserted here.) A read-tool round-trip is two turns: turn 1 writes the prefix cache, turn 2
/// re-sends it and reads it.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with ANTHROPIC_API_KEY set"]
fn smoke_prompt_cache_produces_hits() {
    let Some(key) = env_key("ANTHROPIC_API_KEY") else {
        eprintln!("smoke[cache]: ANTHROPIC_API_KEY unset — skipping");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), "PINEAPPLE-7493\n").unwrap();
    let (gw_port, mut gateway) = boot_gateway(dir.path(), "anthropic", &key);
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

/// Anthropic-specific: extended thinking + tools. The correctness landmine — turn 2's request must
/// replay turn 1's *signed* thinking block, or Anthropic 400s. A successful multi-turn tool round-trip
/// with `--thinking` on proves the signature round-trips intact. (OpenAI reasoning models use
/// `reasoning_effort` and emit no replayable signature, so this is Anthropic-only.)
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
    let (gw_port, mut gateway) = boot_gateway(dir.path(), "anthropic", &key);

    let mut child = serve_child(
        gw_port,
        dir.path(),
        "claude-haiku-4-5",
        &["--thinking", "2000"],
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

/// Anthropic-specific: the `adaptive` thinking shape (generation-6+ models — `claude-opus-4-8` is our
/// own default). Distinct from [`smoke_thinking_with_tools_replays_signature`], which only exercises
/// the older `Budget`/`enabled` shape via `claude-haiku-4-5` — neither model id proves the other's wire
/// shape is correct, so both need live coverage. A successful multi-turn tool round-trip with thinking
/// on proves the `{type:"adaptive", display}` + sibling `output_config`/`thinking` body shape (see
/// `dialect::anthropic::build_body`) is accepted, and that a signed adaptive thinking block replays.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with ANTHROPIC_API_KEY set"]
fn smoke_adaptive_thinking_with_tools_replays_signature() {
    let Some(key) = env_key("ANTHROPIC_API_KEY") else {
        eprintln!("smoke[adaptive-thinking]: ANTHROPIC_API_KEY unset — skipping");
        return;
    };
    let token = "MANGOSTEEN-2081";
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), format!("{token}\n")).unwrap();
    let (gw_port, mut gateway) = boot_gateway(dir.path(), "anthropic", &key);

    let mut child = serve_child(
        gw_port,
        dir.path(),
        "claude-opus-4-8",
        &["--thinking", "2000"],
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
    eprintln!("--- adaptive thinking transcript ---\n{dump}");
    // Success here means the adaptive-shaped request was accepted — the regression this test guards
    // against ("Adaptive" resolving to the wrong shape) would 400 on turn 1 before any tool call.
    assert_eq!(
        resp["success"], true,
        "adaptive thinking+tools must not error on the request shape: {resp}"
    );
    assert!(
        dump.contains(token),
        "the agent should have read the file and echoed `{token}` with adaptive thinking on: {dump}"
    );
    // Unlike `Budget`-shape thinking (always emits a block when enabled), `adaptive` lets the model
    // itself decide *whether* to think at all — a trivial single-tool-call task may legitimately
    // produce none. When it does think, that's the more interesting case (it proves the signed block
    // replayed cleanly into turn 2 without a 400), so still check it — just don't fail the run over the
    // model's own judgment call on a task this simple.
    if dump.contains("\"thinking\"") {
        eprintln!(
            "adaptive thinking: model produced a visible thinking block; signature replay exercised"
        );
    } else {
        eprintln!(
            "adaptive thinking: model chose not to think for this trivial task (adaptive's own call) — shape acceptance still verified"
        );
    }
}

/// OpenAI-specific: the Responses API dialect (`dialect::openai_responses`), live. Every native
/// OpenAI id now routes through `/v1/responses` instead of `/v1/chat/completions` (see
/// `models::ApiKind`) — this is the only live coverage of that path. `--reasoning-effort` requests
/// `include:["reasoning.encrypted_content"]`, so turn 2's request carries turn 1's *replayed*
/// reasoning item; a successful multi-turn tool round-trip proves both the request shape (flat tools,
/// `input` array, `max_output_tokens`) and the reasoning-item replay are accepted by the real API.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with OPENAI_API_KEY set"]
fn smoke_openai_responses_reasoning_replays_signature() {
    let Some(key) = env_key("OPENAI_API_KEY") else {
        eprintln!("smoke[responses-reasoning]: OPENAI_API_KEY unset — skipping");
        return;
    };
    let token = "STARFRUIT-6614";
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), format!("{token}\n")).unwrap();
    let (gw_port, mut gateway) = boot_gateway(dir.path(), "openai", &key);

    let mut child = serve_child(
        gw_port,
        dir.path(),
        "gpt-5-mini",
        &["--reasoning-effort", "low"],
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

    // A second turn forces a follow-up request that must replay turn 1's reasoning item (if any) —
    // the correctness landmine this test guards: a malformed replay would 400 here, not on turn 1.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "Now reply with ONLY the word done." })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames2 = read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let msg_frames = read_until_response(&mut stdout, "get_messages");
    drop(stdin);
    let _ = gateway.kill();
    let _ = gateway.wait();
    let _ = child.wait();

    let resp1 = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("a turn-1 prompt response");
    let resp2 = frames2
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("a turn-2 prompt response");
    let dump = msg_frames.last().unwrap()["data"]["messages"].to_string();
    eprintln!("--- openai responses reasoning transcript ---\n{dump}");
    assert_eq!(
        resp1["success"], true,
        "turn 1 (Responses API request shape) must not error: {resp1}"
    );
    assert!(
        dump.contains(token),
        "the agent should have read the file and echoed `{token}`: {dump}"
    );
    // Success on turn 2 means any reasoning item captured on turn 1 replayed cleanly — the whole
    // point of this test (a malformed replay 400s here, not on turn 1).
    assert_eq!(
        resp2["success"], true,
        "turn 2 must not 400 on reasoning-item replay: {resp2}"
    );
    if dump.contains("\"thinking\"") {
        eprintln!("openai responses: a reasoning block was captured and replayed across turns");
    } else {
        eprintln!(
            "openai responses: model produced no visible reasoning summary for this trivial task — shape acceptance still verified"
        );
    }
}
