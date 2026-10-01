//! FLT-1: stock SDKs, at their default retry policy, through a gateway built from this commit, to
//! real providers with faults injected in between.
//!
//! Every provider a trial's route uses sits behind its own fault proxy
//! (`tests/common/fault_proxy.rs`): the gateway dials it over TLS (`provider_authorities` points
//! the built-in provider at it; `upstream_verify_cert = false` accepts its throwaway cert, which
//! offers only `http/1.1`), and the proxy re-originates TLS to the real host. So the gateway's
//! dialects, auth, catalog walk, key walk, ranker and breaker are the production ones; only the
//! transport is H1 and the cert is unverified. The fault is scripted on the route's primary
//! provider; the fallback (OpenRouter on a pooled row) is always healthy.
//!
//! Each trial is named `FLT-1+<claims it also proves>::<client>::<route>::<fault>` and checks three
//! witnesses against what `crates/gateway/ARCHITECTURE.md` (Status-based failover, Failure Modes)
//! documents for that fault:
//!
//! 1. The client: its final outcome (success on the fallback, the SDK's own retry succeeding, or a
//!    clean JSON error with `x-beyond-request-id` and the right status), how many HTTP attempts its
//!    default retries made, and that it waited out `Retry-After`.
//! 2. The proxy: how many requests reached each provider, and how many the provider processed. No
//!    amplification: the provider processes at most one generation per client attempt (a body the
//!    provider has is never resent), and every count is within its documented bound.
//! 3. The ledger: exactly one billed `ai.usage` row for every generation the provider processed
//!    whose response reached the gateway, with the tokens the provider reported (read from the
//!    proxied response, drained to its end even after a cut); an estimate, flagged and never above
//!    the provider's count, when the response was cut short; and no billed row for anything the
//!    provider refused or never saw.
//!
//! Listed only with `VERIFY_LIVE=1` and the route's keys; a missing key or client means the trial
//! isn't listed, never that it fails.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/fault_proxy.rs"]
mod fault_proxy;

#[path = "common/live.rs"]
mod common;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::free_port;
use fault_proxy::{Delivery, Fault, FaultProxy, Record, Script, Shape};
use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::Value;

/// The dev signing key (seed `[7; 32]`, kid 1) and the tenant-1 token minted from it.
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// The gateway's `read_timeout_secs` in these trials (production default 600s, the SDKs' own
/// request timeout): the longest a live provider may stay silent, before the head or between body
/// reads. Long enough for a real first token, short enough that a stall trial takes seconds.
const READ_TIMEOUT_SECS: u64 = 20;
/// How long a stall holds the connection silent: well past the read timeout, so the gateway's bound
/// (not the proxy letting go) is what the client sees.
const STALL: Duration = Duration::from_secs(45);
/// The stock SDKs' default `max_retries`: openai-python, anthropic-python, openai-node and
/// anthropic-sdk-typescript all retry twice, so a call is at most three HTTP attempts.
const SDK_ATTEMPTS: usize = 3;

#[derive(Clone, Copy)]
struct Provider {
    name: &'static str,
    env: &'static str,
    host: &'static str,
    shape: Shape,
}

const ANTHROPIC: Provider = Provider {
    name: "anthropic",
    env: "ANTHROPIC_API_KEY",
    host: "api.anthropic.com",
    shape: Shape::Anthropic,
};
const OPENAI: Provider = Provider {
    name: "openai",
    env: "OPENAI_API_KEY",
    host: "api.openai.com",
    shape: Shape::OpenAi,
};
const OPENROUTER: Provider = Provider {
    name: "openrouter",
    env: "OPENROUTER_API_KEY",
    host: "openrouter.ai",
    shape: Shape::OpenRouter,
};

/// A catalog row and the pools behind it. The fault goes on `primary`.
#[derive(Clone, Copy)]
struct Route {
    name: &'static str,
    model: &'static str,
    primary: Provider,
    /// A healthy second candidate (the row's OpenRouter arm), or none: a single-provider row.
    fallback: Option<Provider>,
    /// Pool keys on the primary. A single-provider row holds its key twice, so a 429 has a key to
    /// walk to (and a 5xx demonstrably doesn't walk).
    primary_keys: usize,
}

/// The Claude row (Anthropic first, OpenRouter's Claude arm second; Bedrock unkeyed).
const CLAUDE_POOLED: Route = Route {
    name: "claude-pooled",
    model: "claude-haiku-4-5",
    primary: ANTHROPIC,
    fallback: Some(OPENROUTER),
    primary_keys: 1,
};
/// A GPT row (OpenAI Chat first, OpenRouter second). gpt-4o-mini rather than gpt-5-mini: no hidden
/// reasoning, so a first token lands well inside the stall bound and a stream has many events.
const GPT_POOLED: Route = Route {
    name: "gpt-pooled",
    model: "gpt-4o-mini",
    primary: OPENAI,
    fallback: Some(OPENROUTER),
    primary_keys: 1,
};
const CLAUDE_SINGLE: Route = Route {
    name: "claude-single",
    model: "claude-haiku-4-5",
    primary: ANTHROPIC,
    fallback: None,
    primary_keys: 2,
};
const GPT_SINGLE: Route = Route {
    name: "gpt-single",
    model: "gpt-4o-mini",
    primary: OPENAI,
    fallback: None,
    primary_keys: 2,
};
const ROUTES: &[Route] = &[CLAUDE_POOLED, GPT_POOLED, CLAUDE_SINGLE, GPT_SINGLE];

#[derive(Clone, Copy)]
enum Runtime {
    Python,
    Node,
}

/// `(client, runtime, sdk)`.
const CLIENTS: &[(&str, Runtime, &str)] = &[
    ("openai-py", Runtime::Python, "openai"),
    ("anthropic-py", Runtime::Python, "anthropic"),
    ("openai-node", Runtime::Node, "openai"),
    ("anthropic-ts", Runtime::Node, "anthropic"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    /// A synthetic 500 on every request to the primary.
    S500,
    /// One synthetic overload: Anthropic's 529, OpenAI's 503.
    Overloaded,
    /// One 429 with `Retry-After: 2`.
    S429Once,
    /// A 429 with `Retry-After: 1` on every request.
    S429Always,
    /// Every connection to the primary is reset before the TLS handshake.
    ConnectReset,
    /// One request is read in full and the connection reset; the provider never sees it.
    ResetBeforeForward,
    /// One request reaches the provider (billed); the gateway's connection is reset at its head.
    ResetAfterForward,
    /// One stream is reset after three events.
    MidStreamReset,
    /// Every stream is relayed at 100 ms per event.
    SlowDrip,
    /// One stream goes silent after two events.
    StallMidStream,
    /// One stream request reaches the provider (billed), whose answer is held back: no head
    /// before the gateway's read timeout, on a live connection.
    StallHead,
    /// One stream gets the provider's 200 head (the provider accepted and is billing), then
    /// silence: no body byte reaches the gateway.
    StallAfterHead,
}

const CASES: &[Case] = &[
    Case::S500,
    Case::Overloaded,
    Case::S429Once,
    Case::S429Always,
    Case::ConnectReset,
    Case::ResetBeforeForward,
    Case::ResetAfterForward,
    Case::MidStreamReset,
    Case::SlowDrip,
    Case::StallMidStream,
    Case::StallHead,
    Case::StallAfterHead,
];

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::S500 => "status_500",
            Self::Overloaded => "overloaded_once",
            Self::S429Once => "status_429_once",
            Self::S429Always => "status_429_always",
            Self::ConnectReset => "reset_on_connect",
            Self::ResetBeforeForward => "reset_before_forward",
            Self::ResetAfterForward => "reset_after_forward",
            Self::MidStreamReset => "reset_mid_stream",
            Self::SlowDrip => "slow_drip",
            Self::StallMidStream => "stall_mid_stream",
            Self::StallHead => "stall_before_head",
            Self::StallAfterHead => "stall_after_head",
        }
    }

    fn stream(self) -> bool {
        matches!(
            self,
            Self::MidStreamReset
                | Self::SlowDrip
                | Self::StallMidStream
                | Self::StallHead
                | Self::StallAfterHead
        )
    }

    fn overload_status(shape: Shape) -> u16 {
        if shape == Shape::Anthropic { 529 } else { 503 }
    }

    fn script(self, shape: Shape) -> Script {
        let status = |status, retry_after| Fault::Status {
            status,
            retry_after,
        };
        match self {
            Self::S500 => Script::always(status(500, None)),
            Self::Overloaded => Script::once(status(Self::overload_status(shape), None)),
            Self::S429Once => Script::once(status(429, Some(2))),
            Self::S429Always => Script::always(status(429, Some(1))),
            Self::ConnectReset => Script::always(Fault::ResetOnConnect),
            Self::ResetBeforeForward => Script::once(Fault::ResetBeforeForward),
            Self::ResetAfterForward => Script::once(Fault::ResetAfterForward),
            Self::MidStreamReset => Script::once(Fault::ResetMidStream { events: 3 }),
            Self::SlowDrip => Script::always(Fault::SlowDrip {
                per_event: Duration::from_millis(100),
            }),
            Self::StallMidStream => Script::once(Fault::StallMidStream {
                events: 2,
                hold: STALL,
            }),
            Self::StallHead => Script::once(Fault::StallBeforeHead { hold: STALL }),
            Self::StallAfterHead => Script::once(Fault::StallMidStream {
                events: 0,
                hold: STALL,
            }),
        }
    }

    /// The claims this fault proves beyond FLT-1 on this route.
    fn claims(self, route: Route) -> &'static str {
        let pooled = route.fallback.is_some();
        match self {
            Self::S500 | Self::ConnectReset if pooled => "R1+REL-1",
            Self::S500 | Self::ConnectReset => "REL-1+REL-2",
            Self::Overloaded if pooled => "R1",
            Self::Overloaded => "REL-1",
            Self::S429Once => "R2+REL-19",
            Self::S429Always => "R2+REL-1+REL-19",
            Self::ResetBeforeForward => "REL-1",
            Self::ResetAfterForward => "REL-1+BIL-14",
            Self::MidStreamReset => "REL-3+B2+BIL-20",
            Self::SlowDrip => "REL-7+S1",
            Self::StallMidStream => "REL-3+REL-7+B2+BIL-20",
            Self::StallHead => "REL-1+REL-7",
            Self::StallAfterHead => "REL-7+BIL-3+BIL-14+BIL-20",
        }
    }
}

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
    out.extend(std::env::vars());
    out
}

fn interpreter(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/.venv/bin/python"),
        Runtime::Node => PathBuf::from("node"),
    }
}

fn probe_script(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/fault_probe.py"),
        Runtime::Node => repo_root().join("verify/clients/node/fault_probe.mjs"),
    }
}

fn installed(rt: Runtime) -> bool {
    match rt {
        Runtime::Python => interpreter(rt).exists(),
        Runtime::Node => repo_root()
            .join("verify/clients/node/node_modules/@anthropic-ai/sdk")
            .exists(),
    }
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
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") && gateway_bin().exists() {
        let keys = env_keys();
        let have = |p: &Provider| keys.get(p.env).is_some_and(|v| !v.is_empty());
        for &route in ROUTES {
            if !have(&route.primary) || route.fallback.is_some_and(|f| !have(&f)) {
                continue;
            }
            for &(client, rt, sdk) in CLIENTS {
                if !installed(rt) {
                    continue;
                }
                for &case in CASES {
                    let name = format!(
                        "FLT-1+{}::{client}::{}::{}",
                        case.claims(route),
                        route.name,
                        case.name()
                    );
                    let keys = keys.clone();
                    trials.push(Trial::test(name, move || {
                        run_trial(rt, sdk, route, case, &keys)
                    }));
                }
            }
        }
    }
    // Live traffic: no reconciliation window may be open while it runs (see common::live_traffic).
    let _traffic = (!trials.is_empty() && !args.list).then(common::live_traffic);
    libtest_mimic::run(&args, trials).exit();
}

/// Removes a trial's scratch directory on drop.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Kills its process on drop, so a failing trial never leaks a gateway, nats-server or probe.
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
    let mut start = s.len().saturating_sub(2000);
    while !s.is_char_boundary(start) {
        start += 1;
    }
    s[start..].to_owned()
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

fn run_trial(
    rt: Runtime,
    sdk: &str,
    route: Route,
    case: Case,
    keys: &BTreeMap<String, String>,
) -> Result<(), Failed> {
    let [nats_port, port, metrics_port] = [free_port(), free_port(), free_port()];
    // Under the target dir rather than /tmp: nats' JetStream store needs real disk, and a tmpfs
    // shared with other suites can be full. Removed on every exit path, after the processes.
    let scratch =
        Scratch(repo_root().join(format!("target/fault-live/{}-{port}", std::process::id())));
    let dir = scratch.0.clone();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;

    let primary = tokio
        .block_on(FaultProxy::start(
            route.primary.host,
            route.primary.shape,
            case.script(route.primary.shape),
        ))
        .map_err(|e| format!("fault proxy: {e}"))?;
    let fallback = match route.fallback {
        Some(p) => Some(
            tokio
                .block_on(FaultProxy::start(p.host, p.shape, Script::pass()))
                .map_err(|e| format!("fault proxy: {e}"))?,
        ),
        None => None,
    };

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

    let mut cfg = format!(
        "listen = \"127.0.0.1:{port}\"\nmetrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\n\
         upstream_tls = true\nupstream_verify_cert = false\n\
         read_timeout_secs = {READ_TIMEOUT_SECS}\n\n[pool_keys]\n"
    );
    let key = &keys[route.primary.env];
    let primary_keys = vec![format!("{key:?}"); route.primary_keys].join(", ");
    cfg.push_str(&format!("{} = [{primary_keys}]\n", route.primary.name));
    if let Some(f) = route.fallback {
        cfg.push_str(&format!("{} = [{:?}]\n", f.name, keys[f.env]));
    }
    cfg.push_str(&format!(
        "\n[provider_authorities]\n{} = \"127.0.0.1:{}\"\n",
        route.primary.name, primary.port
    ));
    if let (Some(f), Some(fp)) = (route.fallback, &fallback) {
        cfg.push_str(&format!("{} = \"127.0.0.1:{}\"\n", f.name, fp.port));
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
    wait_ready(metrics_port, &mut gw.0, &log_path)?;

    let mut probe = Guard(
        Command::new(interpreter(rt))
            .arg(probe_script(rt))
            .arg(sdk)
            .arg(if case.stream() { "stream" } else { "nonstream" })
            .env("VERIFY_BASE", format!("http://127.0.0.1:{port}"))
            .env("VERIFY_KEY", DEV_TOKEN)
            .env("VERIFY_MODEL", route.model)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("probe: {e}"))?,
    );
    let deadline = Instant::now() + Duration::from_secs(180);
    while probe.0.try_wait().map_err(|e| e.to_string())?.is_none() {
        if Instant::now() > deadline {
            return Err(format!(
                "the client never finished (180s)\n--- gateway log ---\n{}",
                tail(&log_path)
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let mut stdout = String::new();
    probe
        .0
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .map_err(|e| e.to_string())?;
    let verdict: Value = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("VERIFY "))
        .and_then(|j| serde_json::from_str(j).ok())
        .ok_or_else(|| format!("probe printed no VERIFY line:\n{stdout}"))?;

    // The proxies finish draining cut-short responses (and holding stalls) before the books close.
    let settle = STALL + Duration::from_secs(15);
    let prim = tokio.block_on(primary.settle(settle));
    let fb = fallback
        .as_ref()
        .map(|f| tokio.block_on(f.settle(settle)))
        .unwrap_or_default();

    // Rows land a beat after the response; a cut-short one when the gateway notices the end.
    let want = prim
        .iter()
        .chain(&fb)
        .filter(|r| seen_generation(r))
        .count();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut rows = usage_rows(&log_path);
    while billed(&rows).len() < want && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
        rows = usage_rows(&log_path);
    }
    std::thread::sleep(Duration::from_millis(500));
    rows = usage_rows(&log_path);

    let mut problems = client_problems(case, route, &verdict);
    problems.extend(proxy_problems(case, route, &verdict, &prim, &fb));
    problems.extend(ledger_problems(route, &verdict, &prim, &fb, &rows));

    if let Some(out) = std::env::var_os("VERIFY_ROWS_OUT")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
    {
        let lines: String = rows.iter().map(|r| format!("{r}\n")).collect();
        let _ = f.write_all(lines.as_bytes());
    }
    let evidence = format!(
        "--- client --- {verdict}\n--- {} proxy --- {}\n--- fallback proxy --- {}\n--- rows --- {}",
        route.primary.name,
        summarize(&prim),
        summarize(&fb),
        rows.iter()
            .map(|r| format!(
                "\n  {} {} in={} out={} cr={} cw={} est={} outcome={} wire={}",
                r["request_id"],
                r["provider"],
                r["input_tokens"],
                r["output_tokens"],
                r["cache_read_tokens"],
                r["cache_write_tokens"],
                r["usage_estimated"],
                r["outcome"],
                r["usage_wire"]
            ))
            .collect::<String>()
    );
    drop(gw);
    if std::env::var_os("VERIFY_FAULT_VERBOSE").is_some() {
        eprintln!("{evidence}");
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{}\n{evidence}", problems.join("\n")).into())
    }
}

fn summarize(recs: &[Record]) -> String {
    recs.iter()
        .map(|r| {
            format!(
                "\n  @{}ms {} fwd={} status={:?} delivered={:?} head={:?}ms first_byte={:?}ms events={}/{} usage={:?}{}",
                r.at_ms,
                r.fault,
                r.forwarded,
                r.status,
                r.delivered,
                r.head_ms,
                r.first_byte_ms,
                r.events_sent,
                r.events_upstream,
                r.usage
                    .as_ref()
                    .map(|u| (u.input_total, u.output, u.cache_read)),
                r.error
                    .as_ref()
                    .map(|e| format!(" error={e}"))
                    .unwrap_or_default()
            )
        })
        .collect()
}

/// A generation the provider processed (a 2xx) that the ledger owes a row: the gateway saw at
/// least its head, or waited on it with the request delivered and the connection up until its read
/// timeout gave up (D130: the gateway bills the prompt then, as for a client that gave up). A head
/// the proxy wrote just as that timeout expired is owed one either way: an estimate whether the
/// gateway read the head first (a cut stream) or not (its 504).
///
/// Not owed: a generation whose connection the proxy reset before any head (`ResetAfterForward`).
/// The gateway reads a bare reset as the peer declining to answer and bills nothing (see
/// `gave_up_waiting` in proxy.rs); the real edges answer a forwarded failure with an HTTP error.
fn seen_generation(r: &Record) -> bool {
    r.forwarded
        && r.status.is_some_and(|s| (200..300).contains(&s))
        && (r.delivered >= Delivery::Head || r.withheld)
}

fn n(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}

/// Rows that bill anything. A refused attempt may write a zero-token row; it bills nothing.
fn billed(rows: &[Value]) -> Vec<&Value> {
    rows.iter()
        .filter(|r| {
            [
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            ]
            .iter()
            .any(|k| n(&r[*k]) > 0)
        })
        .collect()
}

/// Input including cache on either wire: Anthropic's input excludes reads and writes.
fn row_input(row: &Value) -> u64 {
    let anthropic = match row["usage_wire"].as_str() {
        Some(w) => w == "anthropic",
        None => matches!(row["provider"].as_str(), Some("anthropic" | "bedrock")),
    };
    n(&row["input_tokens"])
        + if anthropic {
            n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
        } else {
            0
        }
}

fn attempts(verdict: &Value) -> Vec<Value> {
    verdict["attempts"].as_array().cloned().unwrap_or_default()
}

/// Witness 1: what the client ended with, against the documented outcome for this fault.
fn client_problems(case: Case, route: Route, v: &Value) -> Vec<String> {
    let pooled = route.fallback.is_some();
    let atts = attempts(v);
    let mut p = Vec::new();
    let ok = v["ok"] == true;
    let complete = ok && v["terminal"] == true;
    let status_of = |i: usize| atts.get(i).and_then(|a| a["status"].as_u64());
    let want_attempts = |p: &mut Vec<String>, want: usize, why: &str| {
        if atts.len() != want {
            p.push(format!(
                "client made {} HTTP attempt(s), want {want}: {why}",
                atts.len()
            ));
        }
    };
    let want_success = |p: &mut Vec<String>, why: &str| {
        if !complete {
            p.push(format!(
                "client did not succeed ({why}): ok={} terminal={} error={}",
                v["ok"], v["terminal"], v["error"]
            ));
        }
    };
    // A clean JSON error: the documented status, a JSON body, and the gateway's request id.
    let want_error = |p: &mut Vec<String>, statuses: &[u64], why: &str| {
        let e = &v["error"];
        if ok {
            p.push(format!("client succeeded, want an error ({why})"));
            return;
        }
        if !e["status"].as_u64().is_some_and(|s| statuses.contains(&s)) {
            p.push(format!(
                "client's error status {}, want one of {statuses:?} ({why}): {e}",
                e["status"]
            ));
        }
        if !e["body"].is_object() {
            p.push(format!("client's error has no JSON body ({why}): {e}"));
        }
        if e["request_id"].as_str().is_none_or(str::is_empty) {
            p.push(format!(
                "client's error carries no x-beyond-request-id ({why}): {e}"
            ));
        }
    };
    // Retry-After honored: each retry starts no earlier than the previous attempt's Retry-After.
    let want_backoff = |p: &mut Vec<String>, secs: f64| {
        for w in atts.windows(2) {
            let gap = w[1]["t"].as_f64().unwrap_or(0.0) - w[0]["t"].as_f64().unwrap_or(0.0);
            if gap < secs * 0.95 {
                p.push(format!(
                    "client retried {gap:.2}s after a 429 with Retry-After: {secs}"
                ));
            }
            if w[0]["retry_after"].is_null() {
                p.push(format!(
                    "a relayed 429 carried no Retry-After to the client: {}",
                    w[0]
                ));
            }
        }
    };
    // A stream that died mid-way must reach the client as an error, never a clean end.
    let want_broken_stream = |p: &mut Vec<String>, why: &str| {
        if ok {
            p.push(format!(
                "client read the cut stream to a clean end ({why}): terminal={} usage={}",
                v["terminal"], v["usage"]
            ));
        }
    };
    let overload = Case::overload_status(route.primary.shape) as u64;
    match case {
        Case::S500 if pooled => {
            want_success(&mut p, "a 5xx fails over before the client notices");
            want_attempts(&mut p, 1, "the gateway's walk absorbs the 5xx");
        }
        Case::S500 => {
            want_error(&mut p, &[500], "the last provider's own 5xx is relayed");
            want_attempts(&mut p, SDK_ATTEMPTS, "the SDK retries a 5xx twice");
        }
        Case::Overloaded if pooled => {
            want_success(&mut p, "an overload fails over");
            want_attempts(&mut p, 1, "the gateway's walk absorbs the overload");
        }
        Case::Overloaded => {
            want_success(&mut p, "the SDK's retry succeeds");
            want_attempts(&mut p, 2, "one relayed overload, one retry");
            if status_of(0) != Some(overload) {
                p.push(format!(
                    "first attempt got {:?}, want the provider's {overload} relayed",
                    status_of(0)
                ));
            }
        }
        Case::S429Once if pooled => {
            want_success(&mut p, "the SDK's retry after Retry-After succeeds");
            want_attempts(
                &mut p,
                2,
                "a 429 is relayed, not failed over to another vendor",
            );
            if status_of(0) != Some(429) {
                p.push(format!("first attempt got {:?}, want 429", status_of(0)));
            }
            want_backoff(&mut p, 2.0);
        }
        Case::S429Once => {
            want_success(&mut p, "the key walk absorbs the 429");
            want_attempts(&mut p, 1, "the gateway walks to the next pool key");
        }
        Case::S429Always => {
            want_error(
                &mut p,
                &[429],
                "every key throttled: the last 429 is relayed",
            );
            want_attempts(&mut p, SDK_ATTEMPTS, "the SDK retries a 429 twice");
            want_backoff(&mut p, 1.0);
            if v["error"]["retry_after"].is_null() {
                p.push("the final 429 carries no Retry-After".into());
            }
        }
        Case::ConnectReset if pooled => {
            want_success(&mut p, "a refused connection fails over");
            want_attempts(&mut p, 1, "the walk absorbs the connect failure");
        }
        Case::ConnectReset => {
            want_error(&mut p, &[502], "no provider could be reached");
            want_attempts(&mut p, SDK_ATTEMPTS, "the SDK retries a 502 twice");
        }
        Case::ResetBeforeForward | Case::ResetAfterForward => {
            want_success(&mut p, "the SDK's retry is a fresh request");
            want_attempts(
                &mut p,
                2,
                "a delivered body is never resent by the gateway: one 502, one SDK retry",
            );
            if status_of(0) != Some(502) {
                p.push(format!(
                    "first attempt got {:?}, want the gateway's 502 (upstream failed after receiving the request)",
                    status_of(0)
                ));
            }
        }
        Case::MidStreamReset => {
            want_broken_stream(&mut p, "the provider reset the stream mid-way");
            want_attempts(&mut p, 1, "nothing is retried after the first byte");
        }
        Case::SlowDrip => {
            want_success(&mut p, "a slow stream that keeps sending survives");
            want_attempts(&mut p, 1, "nothing to retry");
        }
        Case::StallMidStream => {
            want_broken_stream(&mut p, "the provider went silent mid-stream");
            want_attempts(&mut p, 1, "nothing is retried after the first byte");
            // From the attempt's start, not the process's (an interpreter can take long to load).
            let start = atts.first().and_then(|a| a["t"].as_f64()).unwrap_or(0.0);
            let el = v["elapsed"].as_f64().unwrap_or(0.0) - start;
            if el >= STALL.as_secs_f64() {
                p.push(format!(
                    "the stall reached the client only after {el:.1}s: the gateway's {READ_TIMEOUT_SECS}s read timeout did not fire"
                ));
            }
        }
        Case::StallHead => {
            want_success(&mut p, "the SDK retries the gateway's 504");
            want_attempts(
                &mut p,
                2,
                "a delivered body is not resent: one 504, one SDK retry",
            );
            if status_of(0) != Some(504) {
                p.push(format!(
                    "first attempt got {:?}, want the gateway's 504 after {READ_TIMEOUT_SECS}s",
                    status_of(0)
                ));
            }
            let t = |i: usize| atts.get(i).and_then(|a| a["t"].as_f64()).unwrap_or(0.0);
            let gap = t(1) - t(0);
            if !(READ_TIMEOUT_SECS as f64 * 0.9..STALL.as_secs_f64()).contains(&gap) {
                p.push(format!(
                    "the retry began {gap:.1}s after the stalled attempt; the {READ_TIMEOUT_SECS}s read timeout should end it"
                ));
            }
        }
        // Either shape is documented: the gateway hasn't committed a head, so the stall ends in
        // its 504 and the SDK retries; or it has, and the stream ends in an error.
        Case::StallAfterHead => match status_of(0) {
            Some(504) => {
                want_success(&mut p, "the SDK retries the gateway's 504");
                want_attempts(&mut p, 2, "one 504, one SDK retry");
            }
            Some(200) => {
                want_broken_stream(&mut p, "the provider went silent after its head");
                want_attempts(&mut p, 1, "nothing is retried after the first byte");
            }
            other => p.push(format!(
                "first attempt got {other:?}, want the gateway's 504 (or a broken 200 stream)"
            )),
        },
    }
    p
}

/// Witness 2: request counts at each provider against their documented bounds.
fn proxy_problems(
    case: Case,
    route: Route,
    v: &Value,
    prim: &[Record],
    fb: &[Record],
) -> Vec<String> {
    let pooled = route.fallback.is_some();
    let atts = attempts(v).len();
    let fwd = |r: &[Record]| r.iter().filter(|x| x.forwarded).count();
    let processed = fwd(prim) + fwd(fb);
    let mut p = Vec::new();
    for r in prim.iter().chain(fb) {
        if let Some(e) = &r.error {
            p.push(format!(
                "fault proxy failed on its own (not a scripted fault): {e}"
            ));
        }
    }
    // No amplification: one client attempt is at most one generation, on whichever provider.
    if processed > atts {
        p.push(format!(
            "providers processed {processed} requests for {atts} client attempt(s): a delivered body was resent"
        ));
    }
    let mut want = |what: &str, got: usize, lo: usize, hi: usize, why: &str| {
        if got < lo || got > hi {
            let range = if lo == hi {
                format!("{lo}")
            } else {
                format!("{lo}..={hi}")
            };
            p.push(format!("{what} = {got}, want {range}: {why}"));
        }
    };
    let (pn, fwd_p, fwd_f) = (prim.len(), fwd(prim), fwd(fb));
    let keys = route.primary_keys;
    match case {
        Case::S500 if pooled => {
            want(
                "primary requests",
                pn,
                1,
                1,
                "one 5xx, then the walk moves on",
            );
            want("fallback processed", fwd_f, 1, 1, "the fallback serves it");
        }
        Case::S500 => want(
            "primary requests",
            pn,
            atts,
            atts,
            "a 5xx does not walk keys: one request per client attempt",
        ),
        Case::Overloaded if pooled => {
            want(
                "primary requests",
                pn,
                1,
                1,
                "one overload, then the walk moves on",
            );
            want("fallback processed", fwd_f, 1, 1, "the fallback serves it");
        }
        Case::Overloaded => {
            want(
                "primary requests",
                pn,
                2,
                2,
                "the relayed overload and the SDK retry",
            );
            want(
                "primary processed",
                fwd_p,
                1,
                1,
                "only the retry reaches the provider",
            );
        }
        Case::S429Once if pooled => {
            want("primary requests", pn, 1, 2, "the 429, then the retry");
            want(
                "processed",
                processed,
                1,
                1,
                "only the retry is a generation",
            );
            if fb.len() > 1 {
                p.push(format!(
                    "the fallback saw {} requests: a 429 moved vendor",
                    fb.len()
                ));
            }
        }
        Case::S429Once => {
            want(
                "primary requests",
                pn,
                2,
                2,
                "the 429, then the walk to key 2",
            );
            want("primary processed", fwd_p, 1, 1, "key 2 is served");
        }
        Case::S429Always => {
            want(
                "primary requests",
                pn,
                atts * keys,
                atts * keys,
                "each client attempt walks every pool key once, never another vendor",
            );
            want(
                "fallback requests",
                fb.len(),
                0,
                0,
                "a 429 never fails over to another vendor",
            );
        }
        Case::ConnectReset if pooled => {
            want(
                "primary connections",
                pn,
                1,
                3,
                "a failed connect is retried at most twice",
            );
            want("fallback processed", fwd_f, 1, 1, "the fallback serves it");
        }
        Case::ConnectReset => want(
            "primary connections",
            pn,
            atts,
            3 * atts,
            "at most three connects per client attempt",
        ),
        Case::ResetBeforeForward => {
            want(
                "processed",
                processed,
                1,
                1,
                "only the SDK retry reaches a provider",
            );
            if prim.first().is_none_or(|r| r.forwarded) {
                p.push("the reset request isn't the primary's first".into());
            }
        }
        Case::ResetAfterForward => {
            want(
                "processed",
                processed,
                2,
                2,
                "the reset generation and the SDK retry; the gateway never resends",
            );
        }
        Case::MidStreamReset | Case::StallMidStream => {
            want(
                "processed",
                processed,
                1,
                1,
                "nothing is resent after the first byte",
            );
            want(
                "fallback requests",
                fb.len(),
                0,
                0,
                "nothing fails over after the first byte",
            );
        }
        Case::SlowDrip => want("processed", processed, 1, 1, "one stream"),
        Case::StallHead => {
            want(
                "processed",
                processed,
                2,
                2,
                "the stalled generation and the SDK retry; the gateway never resends",
            );
        }
        Case::StallAfterHead => {
            want(
                "fallback requests",
                fb.len(),
                0,
                atts.saturating_sub(1),
                "the gateway never fails over a request the provider accepted",
            );
            want(
                "primary processed",
                fwd_p,
                1,
                atts,
                "the stalled generation is never resent; only an SDK retry is a new one",
            );
        }
    }
    p
}

/// Witness 3: the ledger holds exactly the generations the gateway saw, at the provider's counts.
fn ledger_problems(
    route: Route,
    v: &Value,
    prim: &[Record],
    fb: &[Record],
    rows: &[Value],
) -> Vec<String> {
    let mut p = Vec::new();
    let mut bill: Vec<&Value> = billed(rows);
    let providers = [Some(route.primary), route.fallback];
    let tagged = providers
        .iter()
        .flatten()
        .zip([prim, fb])
        .collect::<Vec<_>>();
    // Completed generations first (an exact match), then cut-short ones (an estimate).
    let mut gens: Vec<(&str, &Record)> = tagged
        .iter()
        .flat_map(|(pv, recs)| recs.iter().map(move |r| (pv.name, r)))
        .filter(|(_, r)| seen_generation(r))
        .collect();
    gens.sort_by_key(|(_, r)| r.delivered != Delivery::Full);
    for (provider, rec) in gens {
        let Some(truth) = rec.usage.clone() else {
            p.push(format!(
                "{provider} processed a generation but reported no usage the proxy could read: {rec:?}"
            ));
            continue;
        };
        let full = rec.delivered == Delivery::Full;
        let pos = bill.iter().position(|r| {
            r["provider"] == provider
                && if full {
                    row_input(r) == truth.input_total
                        && n(&r["output_tokens"]) == truth.output
                        && r["usage_estimated"] != true
                } else {
                    r["usage_estimated"] == true
                }
        });
        let Some(i) = pos else {
            p.push(format!(
                "no {} row for a {provider} generation the gateway saw (provider reported in={} out={}, delivered {:?})",
                if full { "exact" } else { "estimated" },
                truth.input_total,
                truth.output,
                rec.delivered
            ));
            continue;
        };
        let row = bill.remove(i);
        if !full {
            // BIL-20 / B2: non-zero once accepted, never above the provider's own count.
            let (inp, out) = (row_input(row), n(&row["output_tokens"]));
            if inp == 0 {
                p.push(format!("a cut-short estimate bills no input ({row})"));
            }
            if inp > truth.input_total || out > truth.output {
                p.push(format!(
                    "estimate in={inp} out={out} exceeds what {provider} reported in={} out={} ({row})",
                    truth.input_total, truth.output
                ));
            }
        }
    }
    for row in bill {
        p.push(format!(
            "a billed row matches no generation the provider processed and the gateway saw (duplicate or phantom billing): {row}"
        ));
    }
    // The client's own numbers on a success equal the row for the attempt that served it.
    if v["ok"] == true
        && v["terminal"] == true
        && let Some(u) = v["usage"].as_object()
    {
        let id = attempts(v)
            .last()
            .and_then(|a| a["request_id"].as_str().map(str::to_owned));
        match rows
            .iter()
            .find(|r| r["request_id"].as_str() == id.as_deref())
        {
            None => p.push(format!(
                "no row carries the served attempt's request id {id:?}"
            )),
            Some(row) => {
                if row_input(row) != n(&u["input_total"])
                    || n(&row["output_tokens"]) != n(&u["output"])
                {
                    p.push(format!(
                        "client saw in={} out={}, its row bills in={} out={} ({row})",
                        u["input_total"],
                        u["output"],
                        row_input(row),
                        row["output_tokens"]
                    ));
                }
            }
        }
    }
    p
}
