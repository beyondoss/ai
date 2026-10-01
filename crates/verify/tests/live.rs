//! Live cells: real clients, through a gateway built from this commit, to real providers.
//!
//! Each cell is one test, named `CLAIMS::client::route::probe` (claims joined by `+`), so nextest
//! filters, isolates and reports them, and `verify status` reads their outcomes from the JUnit
//! report. A cell boots its own gateway and nats-server with the real pool keys from `.env`, runs
//! one probe in a pinned client (`verify/clients/`), and then checks two witnesses:
//!
//! 1. The client's own verdict: the probe parsed everything and the task's structure held.
//! 2. The gateway's ledger: for every HTTP call the client made, exactly one `ai.usage` row with
//!    that `x-beyond-request-id`, whose tokens equal what the client was shown (normalized for the
//!    wire), served by the provider the route expects. A coding agent's calls aren't visible to
//!    us, so its ledger is checked in aggregate.
//!
//! An E7 cell (a coding agent) checks what the harness itself displays instead of its task: the
//! models it lists against /v1/models, and the session cost it shows against the ledger priced at
//! the card ([`e7_problems`]).
//!
//! Cells are listed only with `VERIFY_LIVE=1`, so an ordinary test run never spends money. A cell
//! whose key or client is missing is not listed at all; `verify status` reports that client as
//! missing for the claim (PARTIAL), never as a pass.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::Value;

/// The dev signing key (seed `[7; 32]`, kid 1) and the tenant-1 token minted from it; the same
/// constants `mise run ai:mint-dev-key` prints.
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// Where a cell sends its traffic: a catalog row and the pool keys the gateway holds.
#[derive(Clone, Copy)]
struct Route {
    name: &'static str,
    model: &'static str,
    /// `(provider, env var)` pool keys; every one must be set for the cell to be listed.
    pools: &'static [(&'static str, &'static str)],
    /// Providers pointed at a dead port, to force failover.
    dead: &'static [&'static str],
    /// The provider every billing row must name.
    serves: &'static str,
}

const CLAUDE: Route = Route {
    name: "claude",
    model: "claude-haiku-4-5",
    pools: &[("anthropic", "ANTHROPIC_API_KEY")],
    dead: &[],
    serves: "anthropic",
};
const GPT: Route = Route {
    name: "gpt",
    model: "gpt-5-mini",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};
/// A Codex-native model: Codex's request (hosted and namespace tools, `store: false`) relays
/// unchanged to OpenAI's Responses API, the first candidate of a Responses-first row.
const CODEX: Route = Route {
    name: "codex",
    model: "gpt-5.3-codex",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};
const OPENROUTER: Route = Route {
    name: "openrouter",
    model: "claude-sonnet-4",
    pools: &[("openrouter", "OPENROUTER_API_KEY")],
    dead: &[],
    serves: "openrouter",
};
/// Anthropic unreachable: the Claude row must fail over to OpenRouter before the client notices.
const FAILOVER: Route = Route {
    name: "failover",
    model: "claude-haiku-4-5",
    pools: &[
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
    dead: &["anthropic"],
    serves: "openrouter",
};
/// The Claude row with Anthropic unkeyed: Amazon Bedrock serves (its key is minted per run from
/// the local AWS credential, see `verify/clients/py/bedrock_token.py`).
const BEDROCK: Route = Route {
    name: "bedrock",
    model: "claude-haiku-4-5",
    pools: &[("bedrock", "AWS_BEARER_TOKEN_BEDROCK")],
    dead: &[],
    serves: "bedrock",
};
/// xAI's own API: grok reports reasoning beside completion_tokens (D23).
const XAI: Route = Route {
    name: "xai",
    model: "grok-4.3",
    pools: &[("xai", "XAI_API_KEY")],
    dead: &[],
    serves: "xai",
};
const EMBED: Route = Route {
    name: "embed",
    model: "text-embedding-3-small",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};

#[derive(Clone, Copy)]
enum Runtime {
    Python,
    Node,
    /// A coding agent (Claude Code, Codex, opencode, pi) driven by `verify/clients/harness.py`.
    Harness,
}

/// `(claims, client, runtime, probe, routes, extra claims on the failover route)`. One line per
/// client × probe; the routes expand it into cells.
type Cell = (
    &'static str,
    &'static str,
    Runtime,
    &'static str,
    &'static [Route],
    &'static str,
);

const GEN: &[Route] = &[CLAUDE, GPT, OPENROUTER, FAILOVER, BEDROCK, XAI];
const SESSION: &[Route] = &[CLAUDE, GPT, OPENROUTER];
const ONE: &[Route] = &[GPT];
const CODEX_ROWS: &[Route] = &[CODEX, CLAUDE, OPENROUTER];

#[rustfmt::skip]
const CELLS: &[Cell] = &[
    // Python SDKs and frameworks.
    ("E1+B1+S1",          "openai-py",     Runtime::Python, "chat_basic",       GEN,      "R1"),
    ("E2+B1+S1",          "anthropic-py",  Runtime::Python, "messages_basic",   GEN,      "R1"),
    ("E3+TRN-1+CAT-9+B1", "openai-py",     Runtime::Python, "responses_basic",  SESSION,  ""),
    ("E4",                "openai-py",     Runtime::Python, "models_list",      ONE,      ""),
    ("E4",                "anthropic-py",  Runtime::Python, "models_list",      ONE,      ""),
    ("T1+B1",             "openai-py",     Runtime::Python, "tools_chat",       GEN,      "R1"),
    ("T1+B1",             "anthropic-py",  Runtime::Python, "tools_messages",   GEN,      "R1"),
    ("M1+B1",             "openai-py",     Runtime::Python, "embeddings",       &[EMBED], ""),
    ("E1+T1+B1",          "langchain",     Runtime::Python, "langchain_chat",   GEN,      "R1"),
    ("E3+T1+B1",          "openai-agents", Runtime::Python, "agents_sdk",       SESSION,  ""),
    // Node SDKs.
    ("E1+B1+S1+S2",       "openai-node",   Runtime::Node,   "chat_basic",       GEN,      "R1"),
    ("E2+B1+S1",          "anthropic-ts",  Runtime::Node,   "messages_basic",   GEN,      "R1"),
    ("E3+TRN-1+CAT-9+B1", "openai-node",   Runtime::Node,   "responses_basic",  SESSION,  ""),
    ("E4",                "openai-node",   Runtime::Node,   "models_list",      ONE,      ""),
    ("E4",                "anthropic-ts",  Runtime::Node,   "models_list",      ONE,      ""),
    ("E1+T1+B1+S2",       "ai-sdk",        Runtime::Node,   "ai_sdk_openai",    GEN,      "R1"),
    ("E2+T1+B1",          "ai-sdk",        Runtime::Node,   "ai_sdk_anthropic", GEN,      "R1"),
    // Coding agents fixing a failing test in a fixture repo (W*). pi runs once per API mode; pi
    // and opencode take models only from config, generated from /v1/models. Codex's GPT row is a
    // Codex-native one (D74 keeps it off gpt-5-mini).
    ("W1",                "claude-code",   Runtime::Harness, "claude-code",     SESSION,  ""),
    ("W2",                "codex",         Runtime::Harness, "codex",           CODEX_ROWS, ""),
    ("W3",                "opencode",      Runtime::Harness, "opencode",        SESSION,  ""),
    ("W4",                "pi",            Runtime::Harness, "pi:chat",         SESSION,  ""),
    ("W4",                "pi",            Runtime::Harness, "pi:messages",     SESSION,  ""),
    ("W4",                "pi",            Runtime::Harness, "pi:responses",    SESSION,  ""),
    // The same sessions for E7, in their own cells so a cost mismatch never reads as a failed task
    // (and a failed task never hides one): the harness's model list, and the session cost it
    // displays against the ledger. Claude Code only where its own price table knows the model
    // (not gpt-5-mini); Codex displays no cost and can't list the catalog.
    ("E7",                "claude-code",   Runtime::Harness, "claude-code",     &[CLAUDE, OPENROUTER], ""),
    ("E7",                "opencode",      Runtime::Harness, "opencode",        SESSION,  ""),
    ("E7",                "pi",            Runtime::Harness, "pi:chat",         SESSION,  ""),
    ("E7",                "pi",            Runtime::Harness, "pi:messages",     SESSION,  ""),
    ("E7",                "pi",            Runtime::Harness, "pi:responses",    SESSION,  ""),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `.env` at the repo root, then the process environment (which wins).
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
    // No Bedrock key in the environment: mint a short-term one from the AWS credential chain.
    if !out.contains_key("AWS_BEARER_TOKEN_BEDROCK")
        && let Ok(o) = Command::new(interpreter(Runtime::Python))
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

fn interpreter(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/.venv/bin/python"),
        Runtime::Node => PathBuf::from("node"),
        Runtime::Harness => repo_root().join("verify/clients/py/.venv/bin/python"),
    }
}

fn probe_script(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/probe.py"),
        Runtime::Node => repo_root().join("verify/clients/node/probe.mjs"),
        Runtime::Harness => repo_root().join("verify/clients/harness.py"),
    }
}

/// The runtime's pinned dependencies are installed (the venv / `npm ci`).
fn installed(rt: Runtime) -> bool {
    match rt {
        Runtime::Python => interpreter(rt).exists(),
        Runtime::Node => repo_root()
            .join("verify/clients/node/node_modules/openai")
            .exists(),
        Runtime::Harness => {
            interpreter(rt).exists()
                && repo_root()
                    .join("verify/clients/node/node_modules/.bin/pi")
                    .exists()
        }
    }
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

fn main() {
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") {
        let keys = env_keys();
        for &(claims, client, rt, probe, routes, failover_claims) in CELLS {
            for route in routes {
                let have_keys = route
                    .pools
                    .iter()
                    .all(|(_, var)| keys.get(*var).is_some_and(|v| !v.is_empty()));
                if !have_keys || !installed(rt) || !gateway_bin().exists() {
                    continue;
                }
                let claims = if route.name == "failover" && !failover_claims.is_empty() {
                    format!("{claims}+{failover_claims}")
                } else {
                    claims.to_owned()
                };
                let name = format!("{claims}::{client}::{}::{probe}", route.name);
                let checks = Checks {
                    task: claims.split('+').any(|c| c != "E7"),
                    e7: claims.split('+').any(|c| c == "E7"),
                };
                let (route, keys) = (*route, keys.clone());
                trials.push(Trial::test(name, move || {
                    run_cell(rt, probe, route, &keys, checks)
                }));
            }
        }
    }
    libtest_mimic::run(&args, trials).exit();
}

/// What a cell asserts beyond the ledger, from the claims in its name.
#[derive(Clone, Copy)]
struct Checks {
    /// Any claim but E7: the client's verdict must hold (for a harness, the fixture test passes).
    task: bool,
    /// E7: the harness's model list and displayed session cost (see [`e7_problems`]). An E7-only
    /// cell doesn't need the task to succeed: a session that failed still has a cost to compare.
    e7: bool,
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Kills its process on drop, so a failing cell never leaks a gateway or nats-server.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_ready(metrics_port: u16, gw: &mut Child, log: &Path) -> Result<(), Failed> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = gw.try_wait() {
            return Err(format!("gateway exited {status}: {}", tail(log)).into());
        }
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", metrics_port)) {
            // `Connection: close` and a read timeout: the admin server keeps an idle connection
            // open for a minute, and reading to EOF without them waits that long.
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
    let start = s.len().saturating_sub(2000);
    s[start..].to_owned()
}

fn run_cell(
    rt: Runtime,
    probe: &str,
    route: Route,
    keys: &BTreeMap<String, String>,
    checks: Checks,
) -> Result<(), Failed> {
    let dir = std::env::temp_dir().join(format!(
        "verify-live-{}-{}",
        std::process::id(),
        free_port()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let nats_port = free_port();
    let _nats = Guard(
        Command::new("nats-server")
            .args([
                "-js",
                "-a",
                "127.0.0.1",
                "-p",
                &nats_port.to_string(),
                "-sd",
            ])
            .arg(dir.join("nats"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("nats-server: {e}"))?,
    );

    let (port, metrics_port) = (free_port(), free_port());
    let mut cfg = format!(
        "listen = \"127.0.0.1:{port}\"\nmetrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\nupstream_tls = true\n\n[pool_keys]\n"
    );
    for (provider, var) in route.pools {
        cfg.push_str(&format!("{provider} = [{:?}]\n", keys[*var]));
    }
    if !route.dead.is_empty() {
        cfg.push_str("\n[provider_authorities]\n");
        for p in route.dead {
            cfg.push_str(&format!("{p} = \"127.0.0.1:9\"\n"));
        }
    }
    cfg.push_str(&format!("\n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"));
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
    // nats needs a moment before the gateway's first connect; readiness waits for the scan.
    wait_ready(metrics_port, &mut gw.0, &log_path)?;

    let out = Command::new(interpreter(rt))
        .arg(probe_script(rt))
        .arg(probe)
        .env("VERIFY_BASE", format!("http://127.0.0.1:{port}"))
        .env("VERIFY_KEY", DEV_TOKEN)
        .env("VERIFY_MODEL", route.model)
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

    // Witness 1: the client's own verdict.
    if checks.task && verdict["ok"] != true {
        return Err(format!(
            "client verdict failed: {}\n--- gateway log ---\n{}",
            verdict["detail"],
            tail(&log_path)
        )
        .into());
    }

    // Witness 2: the ledger. Rows can land a beat after the response; give them a moment.
    // A harness's individual HTTP calls aren't visible to us (`calls` is null): check the ledger
    // in aggregate instead.
    if verdict["calls"].is_null() {
        std::thread::sleep(Duration::from_millis(500));
        let rows = usage_rows(&log_path);
        let mut problems = Vec::new();
        if rows.is_empty() {
            problems.push("the harness finished but the gateway billed nothing".to_owned());
        }
        for row in &rows {
            if row["provider"] != route.serves {
                problems.push(format!(
                    "row served by {}, route expects {} ({row})",
                    row["provider"], route.serves
                ));
            }
            if row["usage_estimated"] == true {
                problems.push(format!("row is an estimate ({row})"));
            }
        }
        if rows
            .iter()
            .map(|r| r["output_tokens"].as_u64().unwrap_or(0))
            .sum::<u64>()
            == 0
        {
            problems.push("no output tokens billed across the session".to_owned());
        }
        if checks.e7 {
            problems.extend(e7_problems(&verdict["detail"], &rows, route.model));
        }
        let _ = std::fs::remove_dir_all(&dir);
        return if problems.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "ledger: {}\n--- detail --- {}",
                problems.join("; "),
                verdict["detail"]
            )
            .into())
        };
    }
    let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
    let deadline = Instant::now() + Duration::from_secs(5);
    let rows = loop {
        let rows = usage_rows(&log_path);
        if rows.len() >= calls.len() || Instant::now() >= deadline {
            break rows;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut problems = Vec::new();
    if rows.len() != calls.len() {
        problems.push(format!(
            "{} client calls but {} ai.usage rows",
            calls.len(),
            rows.len()
        ));
    }
    for call in &calls {
        let Some(id) = call["request_id"].as_str() else {
            problems.push(format!("a call carried no x-beyond-request-id: {call}"));
            continue;
        };
        let matching: Vec<&Value> = rows.iter().filter(|r| r["request_id"] == id).collect();
        let [row] = matching.as_slice() else {
            problems.push(format!(
                "{id}: {} ai.usage rows, want exactly 1",
                matching.len()
            ));
            continue;
        };
        if row["provider"] != route.serves {
            problems.push(format!(
                "{id}: served by {}, route expects {}",
                row["provider"], route.serves
            ));
        }
        if let Some(u) = call["usage"].as_object() {
            let n = |v: &Value| v.as_u64().unwrap_or(0);
            let anthropic_wire = matches!(row["provider"].as_str(), Some("anthropic" | "bedrock"));
            let row_input = n(&row["input_tokens"])
                + if anthropic_wire {
                    n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
                } else {
                    0
                };
            if row_input != n(&u["input_total"]) {
                problems.push(format!(
                    "{id}: client saw {} input tokens, row bills {row_input} ({row})",
                    u["input_total"]
                ));
            }
            // `output_with_reasoning`: a client that reports reasoning apart from output without
            // saying which convention the provider used (the AI SDK) — either exact count is right.
            let alt = u.get("output_with_reasoning").map(n);
            if n(&row["output_tokens"]) != n(&u["output"]) && alt != Some(n(&row["output_tokens"]))
            {
                problems.push(format!(
                    "{id}: client saw {} output tokens, row bills {} ({row})",
                    u["output"], row["output_tokens"]
                ));
            }
            if row["usage_estimated"] == true {
                problems.push(format!(
                    "{id}: row is an estimate on a completed call ({row})"
                ));
            }
        } else {
            problems.push(format!("{id}: the client was shown no usage"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "ledger disagrees with the client:\n  {}\n--- detail --- {}",
            problems.join("\n  "),
            verdict["detail"]
        )
        .into())
    }
}

/// How far a harness's displayed session cost may sit from the ledger priced at the card: its own
/// float rounding, never a missing call or a mispriced token class.
const E7_TOLERANCE: f64 = 0.02;

/// Claim E7 on a harness cell: the harness listed the catalog (`detail.listing`), and the session
/// cost it displayed (`detail.cost.displayed`) equals the ledger rows priced at the model's card
/// (`detail.cost.pricing`, USD per million tokens). A harness that displayed no cost fails: E7 is
/// claimed only on cells where one is shown.
fn e7_problems(detail: &Value, rows: &[Value], model: &str) -> Vec<String> {
    let mut problems = Vec::new();
    if detail["listing"]["ok"] != true {
        problems.push(format!(
            "E7: the harness's model list differs from /v1/models: {}",
            detail["listing"]
        ));
    }
    let cost = &detail["cost"];
    let Some(displayed) = cost["displayed"].as_f64() else {
        problems.push(format!(
            "E7: the harness displayed no cost: {}",
            cost["why"]
        ));
        return problems;
    };
    let price = |k: &str| cost["pricing"][k].as_f64().unwrap_or(f64::NAN) / 1e6;
    let mut billed = 0.0;
    for row in rows {
        if row["requested_model"] != model {
            problems.push(format!(
                "E7: a row for {} can't be priced at {model}'s card ({row})",
                row["requested_model"]
            ));
            continue;
        }
        let n = |k: &str| row[k].as_u64().unwrap_or(0) as f64;
        let (input, read, write) = (
            n("input_tokens"),
            n("cache_read_tokens"),
            n("cache_write_tokens"),
        );
        // A row keeps its upstream wire's token semantics: Anthropic's input excludes the cache,
        // OpenAI's (and OpenRouter's) includes reads and writes. Rows don't say which wire
        // (`usage_wire`, when present, does); today the provider decides it — OpenRouter is
        // always reached on Chat Completions.
        let anthropic = match row["usage_wire"].as_str() {
            Some(wire) => wire == "anthropic",
            None => matches!(row["provider"].as_str(), Some("anthropic" | "bedrock")),
        };
        let fresh = if anthropic {
            input
        } else {
            input - read - write
        };
        billed += fresh * price("input")
            + n("output_tokens") * price("output")
            + read * price("cache_read")
            + write * price("cache_write");
    }
    // Written so a NaN (a price missing from the detail) fails rather than passes.
    let within = billed > 0.0 && (displayed - billed).abs() <= billed * E7_TOLERANCE;
    if !within {
        problems.push(format!(
            "E7: the harness displayed ${displayed:.6}, the ledger priced at the card is \
             ${billed:.6} ({:+.1}%); cost {cost}",
            (displayed / billed - 1.0) * 100.0
        ));
    }
    problems
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
