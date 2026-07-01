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
    boot_gateway_pools(dir, &[(pool, key)])
}

/// Like [`boot_gateway`] but registers *several* pool keys, so a single gateway can front more than
/// one real provider at once — needed to exercise a cross-provider `set_model` switch (Anthropic →
/// OpenAI) through one gateway, where each dialect's request routes to its own pool.
fn boot_gateway_pools(dir: &Path, pools: &[(&str, &str)]) -> (u16, Child) {
    let gw_port = free_port();
    let metrics_port = free_port();
    let pool_keys: String = pools
        .iter()
        .map(|(pool, key)| format!("{pool} = \"{key}\"\n"))
        .collect();
    let config = format!(
        "listen = \"127.0.0.1:{gw_port}\"\n\
         metrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:59321\"\n\
         config_bucket = \"ai-gateway\"\n\
         upstream_tls = true\n\
         \n[pool_keys]\n{pool_keys}\
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

/// OpenAI-compatible **Chat Completions** dialect (`dialect::openai`), live. This is the wire shape
/// *every third-party OpenAI-compatible provider* speaks (Groq, DeepSeek, OpenRouter, Together,
/// Cerebras, xAI, Fireworks, Mistral) — but the two shared-suite providers never touch it: a native
/// OpenAI id like `gpt-4o-mini` routes through `/v1/responses` (Responses dialect), and Claude routes
/// through `/v1/messages`. `ChatCompletions` is only the fallback for an *unrecognized* id (see
/// `Dialect::for_model` / `models::ApiKind`), so without a dedicated test the entire dialect — its
/// `build_body` and SSE decoder — has zero live coverage, and a broken decoder here would silently
/// break every non-Anthropic, non-native-OpenAI provider in production.
///
/// `gpt-3.5-turbo` is the lever that needs no new key: it matches no prefix branch in
/// `models::capabilities`, so it falls to `ApiKind::ChatCompletions` by construction, yet it's still a
/// live, callable model on OpenAI's real `/v1/chat/completions` — the existing `OPENAI_API_KEY`
/// exercises the third-party wire shape against a real endpoint.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with OPENAI_API_KEY set"]
fn smoke_chat_completions_dialect_tool_round_trip() {
    use agent_core::dialect::Dialect;

    let Some(_key) = env_key("OPENAI_API_KEY") else {
        eprintln!("smoke[chat-completions]: OPENAI_API_KEY unset — skipping");
        return;
    };
    // Guard the premise: if a future capability-table change routes this id elsewhere, this test
    // would silently stop covering the ChatCompletions wire — fail loudly instead.
    assert_eq!(
        Dialect::for_model("gpt-3.5-turbo"),
        Dialect::OpenAi,
        "gpt-3.5-turbo must route through the Chat Completions dialect for this test to cover it"
    );

    // A one-off provider profile that forces the ChatCompletions dialect while reusing the OpenAI
    // pool key (a native OpenAI id would take the Responses path instead).
    let provider = Provider {
        name: "openai-chat-completions",
        env: "OPENAI_API_KEY",
        pool: "openai",
        model: "gpt-3.5-turbo",
    };

    let dir = tempfile::tempdir().unwrap();
    let token = "KUMQUAT-3157";
    std::fs::write(dir.path().join("marker.txt"), format!("{token}\n")).unwrap();

    let output = run_oneshot(
        &provider,
        dir.path(),
        "Use the read tool to read the file marker.txt in the current directory, then reply with ONLY the exact token it contains.",
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("--- [chat-completions] stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    assert!(
        output.status.success(),
        "chat-completions dialect tool round-trip failed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains(token),
        "the ChatCompletions decoder should have driven a real tool call and echoed `{token}`.\nstdout: {stdout}"
    );
}

/// Auto-compaction actually fires against a **real** model. Compaction is the single largest section
/// of `agent-core/ARCHITECTURE.md`, yet every one of its unit tests uses `MockTransport` — so the real
/// "cross the threshold → one model call summarizes the prefix → splice a summary in → the next real
/// request is accepted" round-trip has no live proof. The `serve` compaction flags exist precisely to
/// make this cheap to force: pinning a tiny `--context-window` drives the *proactive* trigger (the
/// model's own 200k window still applies on the wire, so nothing 400s — only our local threshold
/// fires). We drive a few real tool-call turns and assert a `Compacted` event was emitted.
///
/// Compaction lives in `agent_core` above the dialect, so it's provider-agnostic — but the
/// summarization call still round-trips each provider's wire, so run it against every provider with a
/// key (the `--context-window` pin is the same regardless of the model's real window).
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_auto_compaction_fires_live() {
    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[compaction]: no provider key set — skipping");
        return;
    }
    for p in providers {
        let key = env_key(p.env).expect("provider key present");
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "PINEAPPLE-7493\n").unwrap();
        let (gw_port, mut gateway) = boot_gateway(dir.path(), p.pool, &key);

        // A tiny window with almost no reserve/keep-recent: the system prompt + tool schemas alone push
        // `last_input_tokens` past `context_window - reserve` after the first real turn, so the *next*
        // turn compacts its prefix. The model's real window is untouched, so no request 400s. The
        // threshold (window − reserve = 900) sits below *both* providers' real prompt size for a read
        // round-trip — OpenAI's tokenizer counts the same system+tools prompt at ~1.3k tokens vs
        // Anthropic's ~2.5k, so a 1800 threshold would clear on Anthropic but never on OpenAI.
        let mut child = serve_child(
            gw_port,
            dir.path(),
            p.model,
            &[
                "--context-window",
                "1000",
                "--compaction-reserve-tokens",
                "100",
                "--compaction-keep-recent-tokens",
                "100",
            ],
        );
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        // Drive several turns; scan every frame across all of them for a `compacted` event. Turn 1
        // writes the prefix; a later turn crosses the threshold at its start and compacts.
        let mut saw_compaction = false;
        let mut all_frames: Vec<Value> = Vec::new();
        for prompt in [
            "Use the read tool to read marker.txt in the current directory, then reply with ONLY the token it contains.",
            "Use the read tool to read marker.txt again, then reply with ONLY the token.",
            "Read marker.txt one more time with the read tool and reply with ONLY the token.",
        ] {
            writeln!(stdin, "{}", json!({ "type": "prompt", "message": prompt })).unwrap();
            stdin.flush().unwrap();
            let frames = read_until_response(&mut stdout, "prompt");
            if frames
                .iter()
                .any(|f| f["type"] == "event" && f["event"]["kind"] == "compacted")
            {
                saw_compaction = true;
            }
            all_frames.extend(frames);
            if saw_compaction {
                break;
            }
        }
        drop(stdin);
        let _ = gateway.kill();
        let _ = gateway.wait();
        let _ = child.wait();

        let compacted_events: Vec<&Value> = all_frames
            .iter()
            .filter(|f| f["type"] == "event" && f["event"]["kind"] == "compacted")
            .collect();
        eprintln!(
            "--- [{}] compaction events ---\n{compacted_events:?}",
            p.name
        );
        assert!(
            saw_compaction,
            "[{}] auto-compaction should have fired against the real model with a pinned tiny context \
             window — no `Compacted` event was seen across the driven turns",
            p.name
        );
        // It fired via the auto-*threshold* trigger (not an overflow-retry or a manual `compact`), it
        // recorded a real pre-compaction context size, and it never *grew* the history. (Note: the
        // count needn't strictly shrink — compaction replaces the summarized prefix with one summary
        // message, so when `find_cut` picks a minimal 1-message boundary on a short conversation the
        // count is unchanged; the load-bearing proof is that the threshold trigger engaged live.)
        let ev = &compacted_events[0]["event"];
        assert_eq!(
            ev["reason"], "threshold",
            "[{}] the live trigger must be the proactive threshold, not overflow/manual: {ev}",
            p.name
        );
        let tokens_before = ev["tokens_before"].as_u64().unwrap_or(0);
        assert!(
            tokens_before > 0,
            "[{}] compaction should record the real pre-compaction context size: {ev}",
            p.name
        );
        let before = ev["messages_before"].as_u64().unwrap_or(0);
        let after = ev["messages_after"].as_u64().unwrap_or(u64::MAX);
        assert!(
            after <= before,
            "[{}] compaction must never grow the history (before={before}, after={after}): {ev}",
            p.name
        );
    }
}

/// `Error::MaxSteps` is reached against a **real** model, and `run` surfaces it as a non-zero exit —
/// the documented failure mode when the model never stops requesting tools. Pinning `--max-steps 1`
/// against a prompt that inherently needs a tool call *plus* a final answer (≥2 steps) forces the
/// ceiling: turn 1 runs the tool (steps=1), the loop check `steps >= max_steps` then bails with
/// `Error::MaxSteps` before a second request. Proves the ceiling actually stops a live run rather than
/// looping forever, and that the error propagates to the process exit code.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_max_steps_halts_a_live_run() {
    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[max-steps]: no provider key set — skipping");
        return;
    }
    for p in providers {
        let key = env_key(p.env).expect("provider key present");
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "PINEAPPLE-7493\n").unwrap();
        let (gw_port, mut gateway) = boot_gateway(dir.path(), p.pool, &key);

        let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
            .args([
                "run",
                "Use the read tool to read marker.txt in the current directory, then reply with ONLY the token it contains.",
                "--gateway-url",
                &format!("http://127.0.0.1:{gw_port}"),
                "--key",
                DEV_TOKEN,
                "--model",
                p.model,
                "--max-steps",
                "1",
            ])
            .current_dir(dir.path())
            .output()
            .expect("spawn agent");
        let _ = gateway.kill();
        let _ = gateway.wait();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "--- [{}] max-steps stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            p.name
        );
        assert!(
            !output.status.success(),
            "[{}] a run pinned to --max-steps 1 on a tool-requiring prompt must exit non-zero.\nstdout: {stdout}\nstderr: {stderr}",
            p.name
        );
        // `main` returns the error, so the runtime prints its `Debug` form (`MaxSteps(1)`), not the
        // `Display` string — accept either shape so a future switch to `Display` doesn't break this.
        let low = stderr.to_lowercase();
        assert!(
            low.contains("maxsteps") || low.contains("max steps"),
            "[{}] the failure should be the step ceiling (Error::MaxSteps).\nstderr: {stderr}",
            p.name
        );
    }
}

/// Shared across providers: the model batches **two** tool calls in **one** turn, and the concurrent
/// grouped-dispatch path (`agent_core`'s flagship mechanism — bounded `buffer_unordered`, same-write-
/// target serialization, in-call-order transcript rebuild, all N results folded into one
/// `tool_result` user turn) round-trips on the real provider wire. Every other smoke prompt is
/// deliberately single-tool, so this is the only live proof that a real multi-block `tool_result` turn
/// (the shape Anthropic *requires* and the Responses/ChatCompletions dialects each encode differently)
/// is accepted end-to-end. Two independent reads reliably elicit a single batched turn from a capable
/// model; the load-bearing assertion is that some assistant message carries ≥2 `tool_use` blocks.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_concurrent_multi_tool_calls_round_trip() {
    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[multi-tool]: no provider key set — skipping");
        return;
    }
    for p in providers {
        let key = env_key(p.env).expect("provider key present");
        let alpha = "APRICOT-1121";
        let beta = "DAMSON-8420";
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("alpha.txt"), format!("{alpha}\n")).unwrap();
        std::fs::write(dir.path().join("beta.txt"), format!("{beta}\n")).unwrap();
        let (gw_port, mut gateway) = boot_gateway(dir.path(), p.pool, &key);

        let mut child = serve_child(gw_port, dir.path(), p.model, &[]);
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "Using the read tool, read BOTH files alpha.txt and beta.txt in the current directory (issue both reads together), then reply with ONLY the two tokens they contain, separated by a single space." })
        )
        .unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "prompt");

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
        assert_eq!(
            resp["success"], true,
            "[{}] the run should succeed: {resp}",
            p.name
        );

        let messages = &msg_frames.last().unwrap()["data"]["messages"];
        let dump = messages.to_string();
        eprintln!("--- [{}] multi-tool transcript ---\n{dump}", p.name);
        // Both tokens were read and echoed.
        assert!(
            dump.contains(alpha) && dump.contains(beta),
            "[{}] both files should have been read and both tokens echoed.\n{dump}",
            p.name
        );
        // The load-bearing proof: at least one assistant turn carried ≥2 `tool_use` blocks, i.e. the
        // model batched them and the concurrent-dispatch + single-`tool_result`-turn path handled it.
        let max_tool_uses_in_a_turn = messages
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter(|m| m["role"] == "assistant")
            .map(|m| {
                m["content"]
                    .as_array()
                    .map(|blocks| blocks.iter().filter(|b| b["type"] == "tool_use").count())
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(
            max_tool_uses_in_a_turn >= 2,
            "[{}] expected some assistant turn to batch ≥2 tool_use blocks (concurrent dispatch path); \
             the most any single turn had was {max_tool_uses_in_a_turn}.\n{dump}",
            p.name
        );
    }
}

/// `abort` cancels a **live** in-flight run — dropping a real `reqwest` stream and killing a real
/// `bash` subprocess (`kill_on_drop`), then leaving `serve` alive for the next command. The mock test
/// (`serve_abort_cancels_an_in_flight_prompt`) can't prove the real-network + real-subprocess cleanup;
/// this does. A `bash sleep 30` gives a wide mid-run window; the aborted prompt must return failure
/// well under 30s, and a follow-up `get_state` must still answer (the process survived the abort).
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_abort_cancels_a_live_run() {
    use std::time::{Duration, Instant};

    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[abort]: no provider key set — skipping");
        return;
    }
    for p in providers {
        let key = env_key(p.env).expect("provider key present");
        let dir = tempfile::tempdir().unwrap();
        let (gw_port, mut gateway) = boot_gateway(dir.path(), p.pool, &key);

        let mut child = serve_child(gw_port, dir.path(), p.model, &[]);
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "Run this exact shell command with the bash tool: sleep 30. Do not do anything else first." })
        )
        .unwrap();
        stdin.flush().unwrap();

        // Wait for the run to actually reach the bash tool (a real turn: request → stream → tool
        // start), then abort — so this exercises the mid-tool cancellation path against a live
        // subprocess.
        std::thread::sleep(Duration::from_secs(4));
        writeln!(stdin, "{}", json!({ "type": "abort", "id": "a1" })).unwrap();
        stdin.flush().unwrap();

        let start = Instant::now();
        let frames = read_until_response(&mut stdout, "prompt");
        let elapsed = start.elapsed();
        eprintln!("--- [{}] abort frames ---\n{frames:#?}", p.name);
        assert!(
            elapsed < Duration::from_secs(20),
            "[{}] abort must cancel the in-flight prompt promptly (well under the 30s sleep), took {elapsed:?}",
            p.name
        );
        let resp = frames
            .iter()
            .rev()
            .find(|f| f["type"] == "response" && f["command"] == "prompt")
            .expect("a prompt response");
        assert_eq!(
            resp["success"], false,
            "[{}] an aborted prompt reports failure: {resp}",
            p.name
        );
        assert!(
            frames.iter().any(|f| f["type"] == "response"
                && f["command"] == "abort"
                && f["success"] == true),
            "[{}] the abort command should have been acknowledged: {frames:#?}",
            p.name
        );

        // The process must still be serving: a `get_state` after the abort answers.
        writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
        stdin.flush().unwrap();
        let state = read_until_response(&mut stdout, "get_state");
        assert!(
            state.iter().any(|f| f["type"] == "response"
                && f["command"] == "get_state"
                && f["success"] == true),
            "[{}] serve must stay alive and answer after an abort: {state:#?}",
            p.name
        );

        drop(stdin);
        let _ = gateway.kill();
        let _ = gateway.wait();
        let _ = child.wait();
    }
}

/// A `follow_up` queued **mid-run** against a live model is injected at the stop boundary as a fresh
/// user turn and answered — proving the `Steering` two-lane machinery works against a real streaming
/// stream, not just `MockTransport` (`serve_follow_up_steers_an_in_flight_run`). A `bash sleep` holds
/// the first turn open long enough to queue the follow-up; the load-bearing, model-wording-independent
/// assertions are that the follow-up is acknowledged, a `steered` event fires, and the injected text
/// lands in the transcript as a real user turn.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_follow_up_injects_into_a_live_run() {
    use std::time::Duration;

    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[follow-up]: no provider key set — skipping");
        return;
    }
    for p in providers {
        let key = env_key(p.env).expect("provider key present");
        let dir = tempfile::tempdir().unwrap();
        let (gw_port, mut gateway) = boot_gateway(dir.path(), p.pool, &key);

        let mut child = serve_child(gw_port, dir.path(), p.model, &[]);
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        let follow_up_marker = "LINGONBERRY-5567";
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "Run this exact shell command with the bash tool: sleep 5. After it finishes, reply with ONLY the word FIRST." })
        )
        .unwrap();
        stdin.flush().unwrap();

        // Queue the follow-up while the first turn's sleep is running (a wide window in the 5s sleep).
        std::thread::sleep(Duration::from_secs(2));
        writeln!(
            stdin,
            "{}",
            json!({ "type": "follow_up", "id": "f1", "message": format!("Now reply with ONLY this exact token: {follow_up_marker}") })
        )
        .unwrap();
        stdin.flush().unwrap();

        let frames = read_until_response(&mut stdout, "prompt");
        eprintln!("--- [{}] follow-up frames ---\n{frames:#?}", p.name);
        assert!(
            frames.iter().any(|f| f["type"] == "response"
                && f["command"] == "follow_up"
                && f["success"] == true),
            "[{}] the follow_up should be acknowledged: {frames:#?}",
            p.name
        );
        assert!(
            frames
                .iter()
                .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
            "[{}] a `steered` event should fire as the follow-up is injected: {frames:#?}",
            p.name
        );

        writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
        stdin.flush().unwrap();
        let msg_frames = read_until_response(&mut stdout, "get_messages");
        drop(stdin);
        let _ = gateway.kill();
        let _ = gateway.wait();
        let _ = child.wait();

        let dump = msg_frames.last().unwrap()["data"]["messages"].to_string();
        assert!(
            dump.contains(follow_up_marker),
            "[{}] the follow-up text should have been injected as a real user turn: {dump}",
            p.name
        );
    }
}

/// A `switch_branch` with `summarize: true` triggers a **real** model summarization of the abandoned
/// branch (`agent_core::branch_summary_request` → a live model call), which is then persisted. The
/// mock test (`serve_switch_branch_summarizes_abandoned_activity_and_navigates`) proves the tree
/// plumbing; this proves the real summarization round-trip produces non-empty, persisted content and
/// that navigation still lands. Runs against every provider with a key — branch summarization lives in
/// `agent_core` above the dialect, but the summarization request still round-trips each provider's wire.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with a provider key set"]
fn smoke_branch_summary_generated_live() {
    let providers = available();
    if providers.is_empty() {
        eprintln!("smoke[branch-summary]: no provider key set — skipping");
        return;
    }
    for p in providers {
        let key = env_key(p.env).expect("provider key present");
        let dir = tempfile::tempdir().unwrap();
        let session_file = dir.path().join("s.jsonl");
        let (gw_port, mut gateway) = boot_gateway(dir.path(), p.pool, &key);

        let mut child = serve_child(
            gw_port,
            dir.path(),
            p.model,
            &["--session-file", session_file.to_str().unwrap()],
        );
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        // Build a two-turn linear history (no tools — keep it cheap and deterministic).
        for msg in [
            "Reply with ONLY the word ONE.",
            "Reply with ONLY the word TWO.",
        ] {
            writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
            stdin.flush().unwrap();
            read_until_response(&mut stdout, "prompt");
        }

        // Tagged message ids come back on `get_messages`; navigate back to the first assistant reply,
        // summarizing the abandoned second turn.
        writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
        stdin.flush().unwrap();
        let msg_frames = read_until_response(&mut stdout, "get_messages");
        let messages = msg_frames.last().unwrap()["data"]["messages"]
            .as_array()
            .expect("messages array")
            .clone();
        let ids: Vec<String> = messages
            .iter()
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect();
        assert_eq!(
            ids.len(),
            messages.len(),
            "[{}] every message should be tagged with an id for branch navigation: {messages:#?}",
            p.name
        );
        assert!(
            ids.len() >= 4,
            "[{}] expected ≥4 messages of history: {ids:?}",
            p.name
        );
        let rewind_to = ids[1].clone();

        writeln!(
            stdin,
            "{}",
            json!({ "type": "switch_branch", "target_id": rewind_to, "summarize": true })
        )
        .unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "switch_branch");
        drop(stdin);
        let _ = gateway.kill();
        let _ = gateway.wait();
        let _ = child.wait();

        let resp = frames
            .iter()
            .rev()
            .find(|f| f["type"] == "response" && f["command"] == "switch_branch")
            .expect("a switch_branch response");
        eprintln!("--- [{}] switch_branch response ---\n{resp}", p.name);
        assert_eq!(
            resp["success"], true,
            "[{}] switch_branch (with live summarization) should succeed: {resp}",
            p.name
        );
        assert_eq!(resp["data"]["target_id"], rewind_to);

        // The real summary was generated and persisted as a `branch_summary` entry with non-empty text.
        let raw = std::fs::read_to_string(&session_file).unwrap();
        let summary_line = raw
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .find(|v| v["type"] == "branch_summary")
            .expect("a persisted branch_summary entry");
        eprintln!("--- [{}] branch_summary entry ---\n{summary_line}", p.name);
        let summary_text = summary_line["summary"]
            .as_str()
            .or_else(|| summary_line["text"].as_str())
            .unwrap_or("");
        assert!(
            !summary_text.trim().is_empty(),
            "[{}] the live branch summary must be non-empty: {summary_line}",
            p.name
        );
    }
}

/// Cross-provider `set_model` scrubs a signed thinking history so the next request doesn't 400 —
/// live, through one gateway fronting **both** providers. `agent-core/ARCHITECTURE.md` calls this out
/// as a correctness landmine: a signed thinking/reasoning block is only replayable to the model that
/// produced it, so `set_model` calls `strip_thinking_blocks()` before switching. `MockTransport` never
/// produces a *real* signed block, so only a live test proves the scrub actually prevents a 400 when a
/// conversation started on Anthropic (with `--thinking`, which emits a signed block) continues on an
/// OpenAI (Responses-dialect) model. Needs both keys; skips otherwise.
#[test]
#[ignore = "live provider smoke; run via `mise run test:smoke:agent` with BOTH ANTHROPIC_API_KEY and OPENAI_API_KEY set"]
fn smoke_set_model_scrubs_thinking_across_providers() {
    let (Some(akey), Some(okey)) = (env_key("ANTHROPIC_API_KEY"), env_key("OPENAI_API_KEY")) else {
        eprintln!(
            "smoke[set-model]: needs BOTH ANTHROPIC_API_KEY and OPENAI_API_KEY — skipping (this is the only cross-provider smoke)"
        );
        return;
    };
    let token = "TAMARILLO-4408";
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), format!("{token}\n")).unwrap();
    // One gateway, two pools: Anthropic (`/v1/messages`) and OpenAI (`/v1/responses`) route by dialect.
    let (gw_port, mut gateway) =
        boot_gateway_pools(dir.path(), &[("anthropic", &akey), ("openai", &okey)]);

    // Start on Anthropic with thinking on so turn 1 produces a *signed* thinking block.
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
    let turn1 = read_until_response(&mut stdout, "prompt");

    // Capture the signed thinking blocks turn 1 produced. Their Anthropic signatures are the exact
    // strings `set_model` must scrub — asserting *these specific values* disappear is robust against
    // the new provider later producing its own (differently-signed) reasoning block.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let before = read_until_response(&mut stdout, "get_messages");
    let before_msgs = before.last().unwrap()["data"]["messages"].clone();
    let anthropic_sigs: Vec<String> = before_msgs
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|b| b["type"] == "thinking")
        .filter_map(|b| b["signature"].as_str().map(str::to_string))
        .filter(|s| !s.is_empty())
        .collect();

    // Switch to a native OpenAI model (different provider *and* dialect). This scrubs thinking history.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-5-mini" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let switch = read_until_response(&mut stdout, "set_model");

    // Inspect the history *immediately after the switch, before turn 2* — so no new OpenAI reasoning
    // block is present yet and the scrub is observed cleanly.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let mid = read_until_response(&mut stdout, "get_messages");
    let mid_dump = mid.last().unwrap()["data"]["messages"].to_string();

    // A second turn now runs on OpenAI. If an Anthropic-signed thinking block had survived into this
    // request, the real OpenAI endpoint would 400 — so success here *is* the scrub working live.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "Reply with ONLY the word done." })
    )
    .unwrap();
    stdin.flush().unwrap();
    let turn2 = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    let _ = gateway.kill();
    let _ = gateway.wait();
    let _ = child.wait();

    let resp1 = turn1
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("a turn-1 prompt response");
    let sm = switch
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "set_model")
        .expect("a set_model response");
    let resp2 = turn2
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("a turn-2 prompt response");
    eprintln!(
        "--- set_model cross-provider ---\nanthropic_sigs: {anthropic_sigs:?}\nmid: {mid_dump}"
    );

    assert_eq!(
        resp1["success"], true,
        "turn 1 (Anthropic + thinking) should succeed: {resp1}"
    );
    assert!(
        !anthropic_sigs.is_empty(),
        "turn 1 must have produced a signed thinking block for the scrub to matter: {before_msgs}"
    );
    assert_eq!(sm["success"], true, "set_model should succeed: {sm}");
    assert_eq!(sm["data"]["model"], "gpt-5-mini");
    // The exact Anthropic-signed thinking blocks are gone after the switch (checked before turn 2 adds
    // any new provider reasoning), and no thinking block at all remains in the post-switch history.
    for sig in &anthropic_sigs {
        assert!(
            !mid_dump.contains(sig.as_str()),
            "set_model must scrub the Anthropic-signed thinking block before switching providers; \
             signature still present: {sig}\nmid: {mid_dump}"
        );
    }
    assert!(
        !mid_dump.contains("\"thinking\""),
        "no thinking block should remain in the history immediately after the cross-provider switch: {mid_dump}"
    );
    assert_eq!(
        resp2["success"], true,
        "turn 2 on OpenAI must NOT 400 — proving the Anthropic-signed thinking block was scrubbed: {resp2}"
    );
}
