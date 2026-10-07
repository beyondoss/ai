//! Live session trials (SES-1..3): whole conversations through a gateway built from this commit,
//! to real providers, where something changes between turns.
//!
//! - **SES-1** — the primary provider dies mid-session. Every pool provider is reached through
//!   `verify/clients/py/upstream_proxy.py`, a tiny HTTP → HTTPS reverse proxy (the gateway runs with
//!   `upstream_tls = false` and its authorities pointed at the proxies). The primary's proxy exits
//!   after it has served `k` requests, so turns `k+1..` are refused at the primary and must fail
//!   over, with the whole history replayed onto the fallback — without the gateway restarting.
//! - **SES-2** — the client switches models (Claude ↔ GPT, so dialects too) between turns.
//! - **SES-3** — stateful Responses: chains carried by `previous_response_id`, the documented
//!   behavior when the row's only Responses upstream dies, and a stateless Codex session resumed.
//!
//! A trial is named `CLAIMS::client::route::scenario`. It boots its own nats-server and gateway, runs
//! one scenario (`verify/clients/py/session_probe.py` for an SDK, `verify/clients/session_harness.py`
//! for a coding agent), and checks two witnesses, as `live.rs` does: the client's own verdict, and
//! the gateway's ledger. An SDK scenario names the provider every call must land on, so each turn
//! is held to exactly one `ai.usage` row on that provider with the tokens the client saw; a coding
//! agent's calls aren't visible, so its rows are held to the order of its steps (the models it ran,
//! the provider that served before and after the death).
//!
//! Listed only with `VERIFY_LIVE=1`, the trial's pool keys set, its client installed and the
//! gateway built: a missing key means the trial isn't listed, never that it fails.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/live.rs"]
mod common;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::Value;

/// The dev signing key and the tenant-1 token minted from it (as in `live.rs`).
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// A stock SDK conversation in `session_probe.py`.
    Probe,
    /// A coding agent in `session_harness.py`.
    Harness,
}

#[derive(Clone, Copy)]
struct Scenario {
    claims: &'static str,
    client: &'static str,
    route: &'static str,
    scenario: &'static str,
    kind: Kind,
    model: &'static str,
    /// `(provider, env var)` pool keys; every one must be set for the trial to be listed.
    pools: &'static [(&'static str, &'static str)],
    /// `(provider, requests)`: that provider's proxy exits after serving this many. When set, every
    /// pool provider is proxied.
    dies: Option<(&'static str, u32)>,
    /// The provider that must take over once the primary has died.
    fallback: &'static str,
    /// SES-2: `model=provider,...`, the models the client alternates and who serves each.
    models: &'static str,
}

const HAIKU: &str = "claude-haiku-4-5";
const ANTHROPIC: (&str, &str) = ("anthropic", "ANTHROPIC_API_KEY");
const OPENROUTER: (&str, &str) = ("openrouter", "OPENROUTER_API_KEY");
const BEDROCK: (&str, &str) = ("bedrock", "AWS_BEARER_TOKEN_BEDROCK");
const OPENAI: (&str, &str) = ("openai", "OPENAI_API_KEY");
const SWITCH: &str = "claude-haiku-4-5=anthropic,gpt-5.1=openai";

const fn ses(
    claims: &'static str,
    client: &'static str,
    route: &'static str,
    scenario: &'static str,
    kind: Kind,
    model: &'static str,
    pools: &'static [(&'static str, &'static str)],
) -> Scenario {
    Scenario {
        claims,
        client,
        route,
        scenario,
        kind,
        model,
        pools,
        dies: None,
        fallback: "",
        models: "",
    }
}

const fn dies(
    mut s: Scenario,
    provider: &'static str,
    after: u32,
    fallback: &'static str,
) -> Scenario {
    s.dies = Some((provider, after));
    s.fallback = fallback;
    s
}

const fn switching(mut s: Scenario) -> Scenario {
    s.models = SWITCH;
    s
}

#[rustfmt::skip]
const SCENARIOS: &[Scenario] = &[
    // SES-1: Anthropic serves requests 1-3 (the Paris tool loop and Tokyo's tool call) and dies;
    // Tokyo's tool result and the Berlin turn replay everything onto the fallback.
    dies(ses("SES-1", "anthropic-py", "anthropic-dies-openrouter", "failover_messages", Kind::Probe, HAIKU, &[ANTHROPIC, OPENROUTER]), "anthropic", 3, "openrouter"),
    dies(ses("SES-1", "anthropic-py", "anthropic-dies-bedrock",    "failover_messages", Kind::Probe, HAIKU, &[ANTHROPIC, BEDROCK]),    "anthropic", 3, "bedrock"),
    dies(ses("SES-1", "openai-py",    "anthropic-dies-openrouter", "failover_chat",     Kind::Probe, HAIKU, &[ANTHROPIC, OPENROUTER]), "anthropic", 3, "openrouter"),
    dies(ses("SES-1", "openai-py",    "anthropic-dies-bedrock",    "failover_chat",     Kind::Probe, HAIKU, &[ANTHROPIC, BEDROCK]),    "anthropic", 3, "bedrock"),
    // A coding agent loses Anthropic two requests into the fixture task.
    dies(ses("SES-1", "claude-code",  "anthropic-dies-openrouter", "claude-code-failover", Kind::Harness, HAIKU, &[ANTHROPIC, OPENROUTER]), "anthropic", 2, "openrouter"),
    dies(ses("SES-1", "pi",           "anthropic-dies-openrouter", "pi-failover",       Kind::Harness, HAIKU, &[ANTHROPIC, OPENROUTER]), "anthropic", 2, "openrouter"),
    // SES-2: Claude and GPT alternate every user turn.
    switching(ses("SES-2", "openai-py",    "claude-gpt", "switch_chat",     Kind::Probe,   HAIKU, &[ANTHROPIC, OPENAI])),
    switching(ses("SES-2", "anthropic-py", "claude-gpt", "switch_messages", Kind::Probe,   HAIKU, &[ANTHROPIC, OPENAI])),
    switching(ses("SES-2", "pi",           "claude-gpt", "pi-switch",       Kind::Harness, HAIKU, &[ANTHROPIC, OPENAI])),
    switching(ses("SES-2", "opencode",     "claude-gpt", "opencode-switch", Kind::Harness, HAIKU, &[ANTHROPIC, OPENAI])),
    // SES-3: previous_response_id chains on a Chat-first GPT row (its Responses arm) and a
    // Responses-first row (gpt-5.3-codex), the Agents SDK, the arm's only upstream dying, and a
    // stateless Codex session resumed from its own transcript.
    ses("SES-3", "openai-py",     "gpt",   "responses_chain", Kind::Probe, "gpt-5.1",       &[OPENAI]),
    ses("SES-3", "openai-py",     "codex", "responses_chain", Kind::Probe, "gpt-5.3-codex", &[OPENAI]),
    ses("SES-3", "openai-agents", "gpt",   "agents_chain",    Kind::Probe, "gpt-5.1",       &[OPENAI]),
    dies(ses("SES-3", "openai-py", "openai-dies-openrouter", "responses_failover", Kind::Probe, "gpt-5.1", &[OPENAI, OPENROUTER]), "openai", 1, "openrouter"),
    ses("SES-3", "codex",         "codex", "codex-resume",    Kind::Harness, "gpt-5.3-codex", &[OPENAI]),
];

/// The real host behind each provider a proxy stands in for.
fn upstream_host(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "api.anthropic.com",
        "openrouter" => "openrouter.ai",
        "openai" => "api.openai.com",
        "bedrock" => "bedrock-runtime.us-east-1.amazonaws.com",
        other => panic!("no upstream host for {other}"),
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn python() -> PathBuf {
    repo_root().join("verify/clients/py/.venv/bin/python")
}

/// `.env` at the repo root, then the process environment (which wins); a Bedrock key is minted
/// from the AWS credential chain when none is set (as in `live.rs`).
fn env_keys() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Ok(s) = std::fs::read_to_string(repo_root().join(".env")) {
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                out.insert(k.trim().to_owned(), v.trim().trim_matches('"').to_owned());
            }
        }
    }
    for (k, v) in std::env::vars() {
        out.insert(k, v);
    }
    if !out.contains_key("AWS_BEARER_TOKEN_BEDROCK")
        && let Ok(o) = Command::new(python())
            .arg(repo_root().join("verify/clients/py/bedrock_token.py"))
            .output()
        && o.status.success()
    {
        let token = String::from_utf8_lossy(&o.stdout).trim().to_owned();
        if !token.is_empty() {
            out.insert("AWS_BEARER_TOKEN_BEDROCK".to_owned(), token);
        }
    }
    out
}

fn installed(kind: Kind) -> bool {
    python().exists()
        && (kind == Kind::Probe
            || repo_root()
                .join("verify/clients/node/node_modules/.bin/pi")
                .exists())
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

fn main() {
    common::started();
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") {
        let keys = env_keys();
        for s in SCENARIOS {
            let have_keys = s
                .pools
                .iter()
                .all(|(_, var)| keys.get(*var).is_some_and(|v| !v.is_empty()));
            if !have_keys || !installed(s.kind) || !gateway_bin().exists() {
                continue;
            }
            let name = format!("{}::{}::{}::{}", s.claims, s.client, s.route, s.scenario);
            let (s, keys) = (*s, keys.clone());
            trials.push(Trial::test(name, move || run(s, &keys)));
        }
    }
    // Live traffic: no reconciliation window may be open while it runs (see common::live_traffic).
    let _traffic = (!trials.is_empty() && !args.list).then(common::live_traffic);
    libtest_mimic::run(&args, trials).exit();
}

/// Kills its process on drop, so a failing trial never leaks a gateway, nats-server or proxy.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_ready(metrics: std::net::SocketAddr, gw: &mut Child, log: &Path) -> Result<(), Failed> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = gw.try_wait() {
            return Err(format!("gateway exited {status}: {}", tail(log)).into());
        }
        if let Ok(mut s) = std::net::TcpStream::connect(metrics) {
            let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
            let _ = s.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
            let mut body = String::new();
            let _ = s.read_to_string(&mut body);
            if body.lines().any(|l| l.starts_with("ai_allowance_ready 1")) {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(format!("gateway never became ready: {}", tail(log)).into())
}

fn tail(path: &Path) -> String {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    let start = s.floor_char_boundary(s.len().saturating_sub(2000));
    s[start..].to_owned()
}

/// Start one provider's proxy and wait for it to listen. `serve` = requests before it dies.
fn start_proxy(dir: &Path, provider: &str, serve: Option<u32>) -> Result<(Guard, u16), Failed> {
    let mut cmd = Command::new(python());
    cmd.arg(repo_root().join("verify/clients/py/upstream_proxy.py"))
        .arg(upstream_host(provider));
    if let Some(n) = serve {
        cmd.arg(n.to_string());
    }
    let log = std::fs::File::create(dir.join(format!("proxy-{provider}.log")))
        .map_err(|e| e.to_string())?;
    let mut guard = Guard(
        cmd.stdout(Stdio::piped())
            .stderr(log)
            .spawn()
            .map_err(|e| format!("proxy: {e}"))?,
    );
    // The proxy binds an OS-chosen port and prints `READY <port>`: no window for another trial to
    // take the port between choosing and binding it.
    let mut line = String::new();
    BufReader::new(guard.0.stdout.take().unwrap())
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    let port = line
        .trim()
        .strip_prefix("READY ")
        .and_then(|p| p.parse().ok())
        .ok_or_else(|| format!("{provider} proxy didn't start: {line:?}"))?;
    Ok((guard, port))
}

fn run(s: Scenario, keys: &BTreeMap<String, String>) -> Result<(), Failed> {
    let dir = std::env::temp_dir().join(format!("verify-session-{}", common::unique_id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let (nats_child, nats_port) = common::spawn_nats(&dir.join("nats"))?;
    let _nats = Guard(nats_child);

    // A dying primary means every provider goes through a proxy: `upstream_tls` is gateway-wide.
    let mut proxies = Vec::new();
    let mut authorities = String::new();
    if let Some((primary, after)) = s.dies {
        for (provider, _) in s.pools {
            let serve = (*provider == primary).then_some(after);
            let (guard, port) = start_proxy(&dir, provider, serve)?;
            proxies.push(guard);
            authorities.push_str(&format!("{provider} = \"127.0.0.1:{port}\"\n"));
        }
    }

    let listeners = common::GATEWAY_LISTENERS;
    let mut cfg = format!(
        "{listeners}\
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\n\
         upstream_tls = {}\n\n[pool_keys]\n",
        s.dies.is_none()
    );
    for (provider, var) in s.pools {
        cfg.push_str(&format!("{provider} = [{:?}]\n", keys[*var]));
    }
    if !authorities.is_empty() {
        cfg.push_str(&format!("\n[provider_authorities]\n{authorities}"));
    }
    cfg.push_str(&format!("\n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"));
    cfg.push_str(common::DEV_ID_SIGNING_TOML);
    let cfg_path = dir.join("gateway.toml");
    std::fs::write(&cfg_path, cfg).map_err(|e| e.to_string())?;

    let log_path = dir.join("gateway.log");
    let log = std::fs::File::create(&log_path).map_err(|e| e.to_string())?;
    let mut gw = Guard(
        Command::new(gateway_bin())
            .args(["run", "-c"])
            .arg(&cfg_path)
            .env("AI_LOG", "warn,ai.usage=info")
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .spawn()
            .map_err(|e| format!("gateway: {e}"))?,
    );
    let common::GatewayPorts {
        proxy: port,
        metrics,
    } = common::gateway_ports(&mut gw.0, &log_path)?;
    wait_ready(metrics, &mut gw.0, &log_path)?;

    let primary = s.dies.map_or(s.pools[0].0, |(p, _)| p);
    let script = match s.kind {
        Kind::Probe => "verify/clients/py/session_probe.py",
        Kind::Harness => "verify/clients/session_harness.py",
    };
    let mut cmd = Command::new(python());
    cmd.arg(repo_root().join(script))
        .arg(s.scenario)
        .env("VERIFY_BASE", format!("http://127.0.0.1:{port}"))
        .env("VERIFY_KEY", DEV_TOKEN)
        .env("VERIFY_MODEL", s.model)
        .env("VERIFY_PROVIDER", s.pools[0].0)
        .env("VERIFY_CLIENT", s.client)
        .env("VERIFY_PRIMARY", primary)
        .env("VERIFY_FALLBACK", s.fallback)
        .env("VERIFY_MODELS", s.models);
    if let Some((_, after)) = s.dies {
        cmd.env("VERIFY_PRIMARY_TURNS", after.to_string());
    }
    let out = cmd
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("probe: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let verdict: Value = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("VERIFY "))
        .and_then(|j| serde_json::from_str(j).ok())
        .ok_or_else(|| format!("probe printed no VERIFY line:\n{stdout}"))?;
    drop(proxies);
    let proxy_logs = proxy_logs(&dir, s);

    // Witness 1: the client's own verdict.
    if verdict["ok"] != true {
        return Err(format!(
            "client verdict failed: {}\n--- proxies ---\n{proxy_logs}\n--- gateway log ---\n{}",
            verdict["detail"],
            tail(&log_path)
        )
        .into());
    }

    // Witness 2: the ledger.
    let problems = if verdict["calls"].is_null() {
        std::thread::sleep(Duration::from_millis(500));
        harness_problems(s, &verdict["detail"], &usage_rows(&log_path))
    } else {
        let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
        let rowed = calls.iter().filter(|c| c["expect"]["rows"] != 0).count();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut rows = usage_rows(&log_path);
        while rows.len() < rowed && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(100));
            rows = usage_rows(&log_path);
        }
        std::thread::sleep(Duration::from_millis(300));
        let rows = usage_rows(&log_path);
        let mut problems = Vec::new();
        if rows.len() != rowed {
            problems.push(format!(
                "{rowed} client calls want a row, but there are {} ai.usage rows",
                rows.len()
            ));
        }
        for (i, call) in calls.iter().enumerate() {
            problems.extend(call_problems(i + 1, call, &rows));
        }
        problems
    };
    export_rows(&log_path);
    // `VERIFY_SESSION_TRACE=1`: show what a passing trial saw (each row's model and provider, the
    // proxies' request logs, the client's detail), so a pass can be read, not just counted.
    if std::env::var("VERIFY_SESSION_TRACE").as_deref() == Ok("1") {
        let rows: Vec<String> = usage_rows(&log_path)
            .iter()
            .map(|r| {
                format!(
                    "{}@{}:{}",
                    r["requested_model"], r["provider"], r["outcome"]
                )
            })
            .collect();
        eprintln!(
            "--- {} {}\nrows: {}\n{proxy_logs}detail: {}",
            s.client,
            s.scenario,
            rows.join(" "),
            verdict["detail"]
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "ledger disagrees with the session:\n  {}\n--- proxies ---\n{proxy_logs}\n--- detail --- {}",
            problems.join("\n  "),
            verdict["detail"]
        )
        .into())
    }
}

fn proxy_logs(dir: &Path, s: Scenario) -> String {
    s.pools
        .iter()
        .filter_map(|(p, _)| {
            std::fs::read_to_string(dir.join(format!("proxy-{p}.log")))
                .ok()
                .map(|l| format!("[{p}]\n{l}"))
        })
        .collect()
}

/// One SDK call against the ledger: exactly one row with its request id, on the provider the
/// scenario says must serve it, billing the tokens the client was shown. `expect.rows = 0`: no row;
/// `expect.error`: the call failed with nothing generated, so its row bills nothing.
fn call_problems(n: usize, call: &Value, rows: &[Value]) -> Vec<String> {
    let num = |v: &Value| v.as_u64().unwrap_or(0);
    let expect = &call["expect"];
    let Some(id) = call["request_id"].as_str() else {
        return if expect["rows"] == 0 {
            Vec::new()
        } else {
            vec![format!("call {n} carried no x-beyond-request-id: {call}")]
        };
    };
    let matching: Vec<&Value> = rows.iter().filter(|r| r["request_id"] == id).collect();
    if expect["rows"] == 0 {
        return if matching.is_empty() {
            Vec::new()
        } else {
            vec![format!("call {n} ({id}): billed {matching:?}, want no row")]
        };
    }
    let [row] = matching.as_slice() else {
        return vec![format!(
            "call {n} ({id}): {} ai.usage rows, want exactly 1",
            matching.len()
        )];
    };
    let mut problems = Vec::new();
    if let Some(want) = expect["provider"].as_str().filter(|p| !p.is_empty())
        && row["provider"] != want
    {
        problems.push(format!(
            "call {n} ({id}): served by {}, want {want}",
            row["provider"]
        ));
    }
    // A call that failed with nothing generated: its one row bills no tokens and isn't `ok`.
    if expect["error"] == true {
        let billed: u64 = [
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ]
        .iter()
        .map(|k| num(&row[*k]))
        .sum();
        if billed != 0 || row["outcome"] == "ok" {
            problems.push(format!(
                "call {n} ({id}): a failed call billed {billed} tokens, outcome {} ({row})",
                row["outcome"]
            ));
        }
        return problems;
    }
    if row["usage_estimated"] == true {
        problems.push(format!("call {n} ({id}): row is an estimate ({row})"));
    }
    if row["price_model"].as_str().is_none_or(str::is_empty) {
        problems.push(format!(
            "call {n} ({id}): price_model doesn't resolve ({row})"
        ));
    }
    let Some(u) = call["usage"].as_object() else {
        problems.push(format!("call {n} ({id}): the client was shown no usage"));
        return problems;
    };
    let anthropic_wire = match row["usage_wire"].as_str() {
        Some(w) => w == "anthropic",
        None => matches!(row["provider"].as_str(), Some("anthropic" | "bedrock")),
    };
    let input = num(&row["input_tokens"])
        + if anthropic_wire {
            num(&row["cache_read_tokens"]) + num(&row["cache_write_tokens"])
        } else {
            0
        };
    if input != num(&u["input_total"]) {
        problems.push(format!(
            "call {n} ({id}): client saw {} input tokens, row bills {input} ({row})",
            u["input_total"]
        ));
    }
    if num(&row["output_tokens"]) != num(&u["output"]) {
        problems.push(format!(
            "call {n} ({id}): client saw {} output tokens, row bills {} ({row})",
            u["output"], row["output_tokens"]
        ));
    }
    if let Some(cr) = u.get("cache_read")
        && num(cr) != num(&row["cache_read_tokens"])
    {
        problems.push(format!(
            "call {n} ({id}): client saw {cr} cache-read tokens, row bills {} ({row})",
            row["cache_read_tokens"]
        ));
    }
    problems
}

/// Collapse runs of equal values: `[a, a, b, a]` → `[a, b, a]`.
fn runs<'a>(xs: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
    let mut out: Vec<&str> = Vec::new();
    for x in xs {
        if out.last() != Some(&x) {
            out.push(x);
        }
    }
    out
}

/// A coding agent's session against the ledger, in aggregate:
///
/// - always: rows exist, output was billed, every row's price resolves;
/// - a dying primary: the first row is the primary's and the last the fallback's, nobody else
///   served, and only the primary's rows may be estimates (a stream the death cut short);
/// - switching: the rows' models, in order, are the steps' models, each served by its provider;
/// - otherwise: every row is the model's, on the first pool provider, exact.
fn harness_problems(s: Scenario, detail: &Value, rows: &[Value]) -> Vec<String> {
    let mut problems = Vec::new();
    if rows.is_empty() {
        return vec!["the harness finished but the gateway billed nothing".to_owned()];
    }
    if rows
        .iter()
        .map(|r| r["output_tokens"].as_u64().unwrap_or(0))
        .sum::<u64>()
        == 0
    {
        problems.push("no output tokens billed across the session".to_owned());
    }
    for row in rows {
        if row["price_model"].as_str().is_none_or(str::is_empty) {
            problems.push(format!("row's price_model doesn't resolve ({row})"));
        }
    }
    let providers: Vec<&str> = rows
        .iter()
        .map(|r| r["provider"].as_str().unwrap_or("?"))
        .collect();
    if let Some((primary, _)) = s.dies {
        if providers.first() != Some(&primary) || providers.last() != Some(&s.fallback) {
            problems.push(format!(
                "rows served by {:?}: want {primary} first and {} last",
                runs(providers.iter().copied()),
                s.fallback
            ));
        }
        for row in rows {
            let p = row["provider"].as_str().unwrap_or("?");
            if p != primary && p != s.fallback {
                problems.push(format!(
                    "row served by {p}, neither primary nor fallback ({row})"
                ));
            }
            if row["usage_estimated"] == true && p != primary {
                problems.push(format!("a fallback row is an estimate ({row})"));
            }
        }
    } else if !s.models.is_empty() {
        let serves: BTreeMap<&str, &str> = s
            .models
            .split(',')
            .filter_map(|m| m.split_once('='))
            .collect();
        let want: Vec<&str> = detail["steps"]
            .as_array()
            .map(|a| a.iter().filter_map(|st| st["model"].as_str()).collect())
            .unwrap_or_default();
        let got = runs(
            rows.iter()
                .map(|r| r["requested_model"].as_str().unwrap_or("?")),
        );
        if got != runs(want.iter().copied()) {
            problems.push(format!("rows' models in order {got:?}, steps ran {want:?}"));
        }
        for row in rows {
            let m = row["requested_model"].as_str().unwrap_or("?");
            if serves.get(m).is_some_and(|p| row["provider"] != *p) {
                problems.push(format!(
                    "a {m} row served by {}, want {} ({row})",
                    row["provider"], serves[m]
                ));
            }
            if row["usage_estimated"] == true {
                problems.push(format!("row is an estimate ({row})"));
            }
        }
    } else {
        for row in rows {
            if row["provider"] != s.pools[0].0 || row["requested_model"] != s.model {
                problems.push(format!(
                    "row for {} served by {}, want {} on {} ({row})",
                    row["requested_model"], row["provider"], s.model, s.pools[0].0
                ));
            }
            if row["usage_estimated"] == true {
                problems.push(format!("row is an estimate ({row})"));
            }
        }
    }
    problems
}

/// With `VERIFY_ROWS_OUT=<file>`, append the trial's billing rows there as JSON lines.
fn export_rows(log: &Path) {
    let Some(out) = std::env::var_os("VERIFY_ROWS_OUT") else {
        return;
    };
    let lines: String = usage_rows(log).iter().map(|r| format!("{r}\n")).collect();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
    {
        let _ = f.write_all(lines.as_bytes());
    }
}

fn usage_rows(log: &Path) -> Vec<Value> {
    let Ok(f) = std::fs::File::open(log) else {
        return Vec::new();
    };
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| l.contains("\"ai.usage\""))
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .map(|v| v.get("fields").cloned().unwrap_or(v))
        .collect()
}
