//! Long live sessions and large tool sets: claims LNG-1, LNG-2 and TOOL-1.
//!
//! Every trial boots its own gateway (and nats-server) from this tree with the real pool keys in
//! `.env`, and drives a real client through `verify/clients/harness_long.py`, which puts a
//! recording proxy between the client and the gateway. So unlike `live.rs`'s harness cells, which
//! check a coding agent's ledger only in aggregate, these trials see every HTTP call a harness
//! made (path, status, `x-beyond-request-id`, whether it was a compaction) and hold each one to
//! the ledger.
//!
//! # LNG-1: long coding sessions
//!
//! A coding agent works through `ledgerlib`, a fixture repo with eight ordered steps (fix a bug,
//! implement five modules against their tests, refactor, add a CLI), each with its own test file.
//! It takes an agent 30-100 model calls (Codex, the most economical, 28-41). Each harness's own
//! compaction knob is turned down (`harness_long.py` says which) so that auto-compaction happens
//! inside the session. A trial passes when:
//!
//! - the task completed: the repo's full test run passes and no test file changed;
//! - the session made at least [`MIN_TURNS`] billed model calls;
//! - compaction happened (except in an `uncompacted` session), witnessed twice: by the harness's
//!   own events and by a compaction request on the wire (its summarization prompt as the last user
//!   message, or Codex's `/v1/responses/compact`), and it didn't loop (at most one compaction
//!   request per [`TURNS_PER_COMPACTION`] billed calls);
//! - the ledger holds every call: each successful billed call has exactly one `ai.usage` row with
//!   its request id, served by the route's provider and not an estimate; a free call
//!   (`count_tokens`) has none; a refused call bills nothing; no row belongs to no call;
//! - where the session reconciles and the provider's admin key is set, the session's token totals
//!   equal the provider's usage report for the pool key, model and minutes (the BIL-5 method,
//!   reused from `reconcile_live.rs`). Two sessions reconcile, on rows no other live suite drives:
//!   pi on `claude-sonnet-5` and Claude Code (translated) on `gpt-5`. A failure lists the minutes
//!   that differ, to tell another suite's traffic (a catalog sweep) from ours.
//!
//! # LNG-2: caching over a long session
//!
//! From the same sessions (and an opencode one), the per-turn cache share
//! `cache_read / input_total` over the session's main turns. A main turn is a successful billed
//! call that offers tools, isn't a compaction request, and whose prompt is at least
//! [`CACHE_ELIGIBLE`] tokens. The first [`WARMUP`] main turns are skipped, as is the first one
//! after any compaction or other rewrite (fewer items than the turn before): its prefix is new.
//!
//! The floor: over the remaining turns, the mean share is at least [`MEAN_FLOOR`] and at least
//! [`TURN_SHARE`] of turns read at least [`TURN_FLOOR`] of their prompt from cache. Why those:
//! every main turn re-sends the previous turn's whole prompt plus one tool round, and a tool round
//! here adds a median of a few hundred to ~1k tokens to a 10k-50k-token prompt, so a working cache
//! reads well over 0.9 of each turn (measured: 0.95-0.99 on Claude Code and opencode). Two things
//! legitimately lower single turns: Anthropic's cache looks back only 20 content blocks from a
//! breakpoint, so a harness that marks only its last message misses part of the prefix every few
//! turns (pi: 0.66-0.8 on those), and a large tool result written on one turn is new on the next.
//! 0.5 per turn and a 0.75 mean leave room for both; a path whose cache collapsed (the translated
//! history's bytes changing each turn, a breakpoint dropped) sits near 0 and fails both.
//! [`CACHE_ELIGIBLE`] keeps turns whose previous prompt might be below Claude Haiku 4.5's
//! 4096-token minimum cacheable prompt out of the sample.
//!
//! # TOOL-1: large tool sets
//!
//! SDK requests (openai-py, anthropic-py) offering N tools, forced to call the last one, on every
//! wire: Chat, Responses, Messages and Codex's `namespace` tool, native and translated, below and
//! above OpenAI Chat Completions' 128-tool limit (measured: OpenAI's Responses API takes 600, and
//! Anthropic, which publishes no count limit, takes 600), plus large schemas (12-deep nesting, a
//! 1000-value enum). Below a provider's limit the call must succeed and call the last tool (no
//! tool silently dropped); above it, a 4xx whose body names the limit (`128`), never a 5xx. A
//! Messages client above 128 on a GPT row walks the row's Responses arm and must succeed (D131). And
//! two harnesses offered ~150 MCP tools from a local stdio server (Codex sends them as one
//! namespace): the agent must call the last one and report its secret.
//!
//! Trials are named `CLAIMS::client::route::scenario` and listed only with `VERIFY_LIVE=1`, the
//! route's pool keys, the clients installed and the gateway built.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[allow(dead_code, unused_imports)]
#[path = "reconcile_live.rs"]
mod recon;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use libtest_mimic::{Arguments, Failed, Trial};
use recon::{Provider, Totals};
use serde_json::Value;

/// The dev signing key (seed `[7; 32]`, kid 1) and the tenant-1 token minted from it.
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// LNG-1: the fewest billed model calls a session may make.
const MIN_TURNS: usize = 25;
/// LNG-1: a session with more than one compaction request per this many billed calls is looping
/// (Claude Code's measured rate is one per 10-14 calls at its knob here; pi sends two per
/// compaction; a loop compacts every 2-3 turns).
const TURNS_PER_COMPACTION: usize = 4;
/// LNG-2: the smallest prompt (tokens) a main turn needs to count toward the cache floor.
const CACHE_ELIGIBLE: u64 = 8192;
/// LNG-2: main turns skipped at the start of a session.
const WARMUP: usize = 3;
/// LNG-2: the floor on the mean cache share, and the per-turn floor most turns must clear.
const MEAN_FLOOR: f64 = 0.75;
const TURN_FLOOR: f64 = 0.5;
const TURN_SHARE: f64 = 0.85;
/// How long the provider's usage report gets to settle on a session, and how often to ask.
const SETTLE_BUDGET: Duration = Duration::from_secs(15 * 60);
const POLL_EVERY: Duration = Duration::from_secs(30);

/// A catalog row and the pool keys the gateway holds for it.
#[derive(Clone, Copy)]
struct Route {
    name: &'static str,
    model: &'static str,
    pools: &'static [(&'static str, &'static str)],
    /// The provider every billing row must name.
    serves: &'static str,
}

const ANTHROPIC: &[(&str, &str)] = &[("anthropic", "ANTHROPIC_API_KEY")];
const OPENAI: &[(&str, &str)] = &[("openai", "OPENAI_API_KEY")];

const CLAUDE: Route = Route {
    name: "claude",
    model: "claude-haiku-4-5",
    pools: ANTHROPIC,
    serves: "anthropic",
};
/// A Claude row no other suite drives, so its session can be reconciled against Anthropic's
/// usage report without anyone else's traffic in the same key + model + minutes.
const SONNET5: Route = Route {
    name: "claude-sonnet-5",
    model: "claude-sonnet-5",
    pools: ANTHROPIC,
    serves: "anthropic",
};
const GPT: Route = Route {
    name: "gpt",
    model: "gpt-5-mini",
    pools: OPENAI,
    serves: "openai",
};
/// A GPT row no other live suite drives, for a reconciled translated session. Cheaper rows fail
/// the task for their own reasons: gpt-4.1-mini stops after step 1, gpt-5-nano edits the tests,
/// gpt-5-mini answers the turn after a compaction with another summary, and gpt-5.4-mini is D114.
const GPT5: Route = Route {
    name: "gpt-5",
    model: "gpt-5",
    pools: OPENAI,
    serves: "openai",
};
/// A GPT-5.4 row: Claude Code's first turn (thinking + tools) is refused on it (D114).
const GPT54MINI: Route = Route {
    name: "gpt-5.4-mini",
    model: "gpt-5.4-mini",
    pools: OPENAI,
    serves: "openai",
};
/// The one Codex-native row our OpenAI key serves (the other codex rows are OpenRouter-only).
const CODEX: Route = Route {
    name: "codex",
    model: "gpt-5.3-codex",
    pools: OPENAI,
    serves: "openai",
};

/// `(claims, client, harness spec, route, reconcile against)`.
type Session = (
    &'static str,
    &'static str,
    &'static str,
    Route,
    Option<Provider>,
);

#[rustfmt::skip]
const SESSIONS: &[Session] = &[
    ("LNG-1+LNG-2", "claude-code", "claude-code", CLAUDE,    None),
    // Translated: Claude Code's Messages onto a GPT row, with Claude Code's own compaction
    // threshold. Forced to compact, GPT models answer the next turn with another summary: their
    // summary quotes Claude Code's compaction prompt verbatim ("CRITICAL: Respond with TEXT ONLY")
    // into the new context, and the session stalls (5 of 6 tries on gpt-5 and gpt-5-mini; the
    // gateway relays both turns intact). Compaction is covered on the Claude rows.
    ("LNG-1",       "claude-code", "claude-code:uncompacted", GPT5, Some(Provider::OpenAi)),
    // D114: refused at the first turn (Messages thinking + tools -> Chat reasoning_effort + tools).
    ("LNG-1",       "claude-code", "claude-code", GPT54MINI, None),
    // Not reconciled: gpt-5.3-codex is the one row Codex runs on (D74), and live.rs,
    // session_live.rs and catalog_live.rs drive it too. Two tries both met a few small requests
    // from other suites in the same minutes (+3 and +4 requests over ~30), which no report
    // filter can separate from ours.
    ("LNG-1",       "codex",       "codex",       CODEX,     None),
    ("LNG-1+LNG-2", "pi",          "pi:messages", SONNET5,   Some(Provider::Anthropic)),
    // Translated: pi's Chat Completions onto a Claude row (the gateway adds the cache breakpoints).
    ("LNG-1+LNG-2", "pi",          "pi:chat",     CLAUDE,    None),
    ("LNG-2",       "opencode",    "opencode",    CLAUDE,    None),
];

#[derive(Clone, Copy)]
enum Expect {
    /// 200, and the model called the last tool offered.
    Works,
    /// A 4xx whose body names the limit.
    Limit(&'static str),
}

/// `(client, route, scenario, expectation)`; scenario as `harness_long.py tools` takes it.
#[rustfmt::skip]
const TOOL_CASES: &[(&str, Route, &str, Expect)] = &[
    // OpenAI Chat Completions: 128 tools, no more.
    ("openai-py",    GPT,    "chat_128",            Expect::Works),
    ("openai-py",    GPT,    "chat_129",            Expect::Limit("128")),
    // OpenAI's Responses API has no 128 cap.
    ("openai-py",    GPT,    "responses_600",       Expect::Works),
    // Translated Messages → a GPT row: up to 128 onto Chat Completions, more onto the row's
    // Responses arm (D131).
    ("anthropic-py", GPT,    "messages_128",        Expect::Works),
    ("anthropic-py", GPT,    "messages_129",        Expect::Works),
    ("anthropic-py", GPT,    "messages_300",        Expect::Works),
    // Anthropic: no count limit (600 tested), natively and from OpenAI-wire clients.
    ("anthropic-py", CLAUDE, "messages_600",        Expect::Works),
    ("openai-py",    CLAUDE, "chat_300",            Expect::Works),
    ("openai-py",    CLAUDE, "responses_300",       Expect::Works),
    // Codex's namespace tool with many members: relayed (Codex row), flattened (Claude row).
    ("openai-py",    CODEX,  "namespace_200",       Expect::Works),
    ("openai-py",    CLAUDE, "namespace_200",       Expect::Works),
    // Large schemas: deep nesting and a big enum, every wire.
    ("openai-py",    GPT,    "chat_8_deep12",       Expect::Works),
    ("openai-py",    CLAUDE, "chat_8_deep12",       Expect::Works),
    ("anthropic-py", GPT,    "messages_8_deep12",   Expect::Works),
    ("openai-py",    GPT,    "chat_8_enum1000",     Expect::Works),
    ("openai-py",    CLAUDE, "responses_8_enum1000", Expect::Works),
    ("anthropic-py", CLAUDE, "messages_8_enum1000", Expect::Works),
];

/// `(client, route, expectation)`: a harness offered 150 MCP tools from a local stdio server.
/// Works: it calls the last one and reports its secret. Limit: the refusal names the limit and the
/// harness shows it. Claude Code's ~20 tools + 150 on a GPT row are more than OpenAI Chat
/// Completions takes, so they walk the row's Responses arm (D131).
#[rustfmt::skip]
const MCP_CASES: &[(&str, Route, Expect)] = &[
    ("claude-code", CLAUDE, Expect::Works),
    ("claude-code", GPT,    Expect::Works),
    ("codex",       CODEX,  Expect::Works),
    // Codex's web_search is turned off for this one: D78 refuses it on any Claude row.
    ("codex",       CLAUDE, Expect::Works),
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
    out.extend(std::env::vars());
    out.retain(|_, v| !v.is_empty());
    out
}

fn python() -> PathBuf {
    repo_root().join("verify/clients/py/.venv/bin/python")
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

fn installed() -> bool {
    python().exists()
        && repo_root()
            .join("verify/clients/node/node_modules/.bin/pi")
            .exists()
        && gateway_bin().exists()
}

fn main() {
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") && installed() {
        let keys = env_keys();
        let keyed = |r: &Route| r.pools.iter().all(|(_, v)| keys.contains_key(*v));
        for &(claims, client, spec, route, recon) in SESSIONS {
            if !keyed(&route) {
                continue;
            }
            let scenario = match spec.split_once(':') {
                Some((_, mode)) => format!("long_task_{mode}"),
                None => "long_task".to_owned(),
            };
            let name = format!("{claims}::{client}::{}::{scenario}", route.name);
            let keys = keys.clone();
            trials.push(Trial::test(name, move || {
                long_session(claims, spec, route, recon, &keys)
            }));
        }
        for &(client, route, scenario, expect) in TOOL_CASES {
            if !keyed(&route) {
                continue;
            }
            let name = format!("TOOL-1::{client}::{}::{scenario}", route.name);
            let keys = keys.clone();
            trials.push(Trial::test(name, move || {
                tool_case(route, scenario, expect, &keys)
            }));
        }
        for &(client, route, expect) in MCP_CASES {
            if !keyed(&route) {
                continue;
            }
            let name = format!("TOOL-1::{client}::{}::mcp_150", route.name);
            let keys = keys.clone();
            trials.push(Trial::test(name, move || {
                mcp_case(client, route, expect, &keys)
            }));
        }
    }
    libtest_mimic::run(&args, trials).exit();
}

// --- gateway -------------------------------------------------------------------------------------

/// Kills its process on drop, so a failing trial never leaks a gateway or nats-server.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A free port below the ephemeral range (as session_live.rs picks them): a port the OS handed
/// out and took back can be handed to another process's outbound connection before the gateway
/// binds it, which many parallel sessions made happen.
fn free_port() -> u16 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let _ = NEXT.compare_exchange(
        0,
        u64::from(std::process::id())
            ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
    loop {
        // splitmix64: consecutive seeds scatter across the range.
        let mut z = NEXT
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let port = 20_000 + ((z ^ (z >> 31)) % 12_000) as u16;
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
}

struct Gateway {
    _nats: Guard,
    _gw: Guard,
    port: u16,
    log: PathBuf,
    dir: PathBuf,
    tmp: PathBuf,
}

impl Gateway {
    fn boot(route: Route, keys: &BTreeMap<String, String>) -> Result<Gateway, Failed> {
        let dir = repo_root().join(format!(
            "target/verify-long/{}-{}",
            std::process::id(),
            free_port()
        ));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        // Harness homes and repos: outside the user's home, so Claude Code's walk up the parent
        // directories for CLAUDE.md finds nothing of this machine's (under target/ it found the
        // repo's and the user's own), and not on /tmp, a tmpfs a dozen parallel homes can fill.
        let tmp = PathBuf::from(format!(
            "/var/tmp/verify-long-{}",
            dir.file_name().unwrap().to_string_lossy()
        ));
        std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
        let nats_port = free_port();
        let nats = Guard(
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
             nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\n\
             upstream_tls = true\n\n[pool_keys]\n"
        );
        for (provider, var) in route.pools {
            cfg.push_str(&format!("{provider} = [{:?}]\n", keys[*var]));
        }
        cfg.push_str(&format!("\n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"));
        let cfg_path = dir.join("gateway.toml");
        std::fs::write(&cfg_path, cfg).map_err(|e| e.to_string())?;
        let log = dir.join("gateway.log");
        let file = std::fs::File::create(&log).map_err(|e| e.to_string())?;
        let mut gw = Guard(
            Command::new(gateway_bin())
                .args(["run", "-c"])
                .arg(&cfg_path)
                .env("AI_LOG", "warn,ai.usage=info")
                .stdout(file.try_clone().map_err(|e| e.to_string())?)
                .stderr(file)
                .spawn()
                .map_err(|e| format!("gateway: {e}"))?,
        );
        wait_ready(metrics_port, &mut gw.0, &log)?;
        Ok(Gateway {
            _nats: nats,
            _gw: gw,
            port,
            log,
            dir,
            tmp,
        })
    }

    /// Run `harness_long.py <kind> <arg>` against this gateway; its `VERIFY` verdict.
    fn client(&self, kind: &str, arg: &str, model: &str) -> Result<Value, Failed> {
        let out = Command::new(python())
            .arg(repo_root().join("verify/clients/harness_long.py"))
            .args([kind, arg])
            .env("VERIFY_BASE", format!("http://127.0.0.1:{}", self.port))
            .env("VERIFY_KEY", DEV_TOKEN)
            .env("VERIFY_MODEL", model)
            .env("TMPDIR", &self.tmp)
            .stderr(Stdio::inherit())
            .output()
            .map_err(|e| format!("harness_long.py: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let verdict: Value = stdout
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix("VERIFY "))
            .and_then(|j| serde_json::from_str(j).ok())
            .ok_or_else(|| format!("harness_long.py printed no VERIFY line:\n{stdout}"))?;
        if let Some(e) = verdict["detail"]["exception"].as_str() {
            return Err(
                format!("harness_long.py raised {e}\n{}", verdict["detail"]["trace"]).into(),
            );
        }
        Ok(verdict)
    }

    /// The ledger once every row has landed: wait until a row exists for each of `ids`.
    fn rows_for(&self, ids: &[&str]) -> Vec<Value> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let rows = usage_rows(&self.log);
            let all = ids
                .iter()
                .all(|id| rows.iter().any(|r| r["request_id"] == *id));
            if all || Instant::now() >= deadline {
                // A beat more for a stray row a free call might (wrongly) have written.
                std::thread::sleep(Duration::from_millis(500));
                return usage_rows(&self.log);
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    fn cleanup(&self) {
        if std::env::var_os("VERIFY_LONG_KEEP").is_none() {
            let _ = std::fs::remove_dir_all(&self.dir);
            let _ = std::fs::remove_dir_all(&self.tmp);
        }
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

/// With `VERIFY_ROWS_OUT=<file>`, append the trial's billing rows there (to total a run's spend).
fn export_rows(rows: &[Value]) {
    let Some(out) = std::env::var_os("VERIFY_ROWS_OUT") else {
        return;
    };
    let lines: String = rows.iter().map(|r| format!("{r}\n")).collect();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
    {
        let _ = f.write_all(lines.as_bytes());
    }
}

// --- the ledger against the calls ----------------------------------------------------------------

fn n(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}

fn billed_path(path: &str) -> bool {
    matches!(
        path,
        "/v1/messages" | "/v1/chat/completions" | "/v1/responses" | "/v1/responses/compact"
    )
}

/// A row's whole prompt, whatever its wire: Anthropic's input excludes the cache, OpenAI's
/// includes it.
fn input_total(row: &Value) -> u64 {
    if row["usage_wire"] == "anthropic" {
        n(&row["input_tokens"]) + n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
    } else {
        n(&row["input_tokens"])
    }
}

/// Every proxied call against the ledger. Returns the problems, and each call's row (in call
/// order; `None` for calls with no row).
fn ledger_problems<'a>(
    calls: &[Value],
    rows: &'a [Value],
    serves: &str,
) -> (Vec<String>, Vec<Option<&'a Value>>) {
    let mut problems = Vec::new();
    let mut matched = Vec::with_capacity(calls.len());
    let mut used = 0;
    for call in calls {
        let path = call["path"].as_str().unwrap_or("");
        let status = n(&call["status"]);
        let label = format!(
            "{} {path} -> {status} ({})",
            call["method"].as_str().unwrap_or("?"),
            call["request_id"].as_str().unwrap_or("no request id")
        );
        let mine: Vec<&Value> = match call["request_id"].as_str() {
            Some(id) => rows.iter().filter(|r| r["request_id"] == id).collect(),
            None => Vec::new(),
        };
        used += mine.len();
        matched.push(mine.first().copied());
        let ok = (200..300).contains(&status);
        if billed_path(path) && ok {
            let [row] = mine.as_slice() else {
                problems.push(format!("{label}: {} ai.usage rows, want 1", mine.len()));
                continue;
            };
            if row["provider"] != serves {
                problems.push(format!(
                    "{label}: served by {}, want {serves}",
                    row["provider"]
                ));
            }
            if row["usage_estimated"] == true {
                problems.push(format!("{label}: a completed call billed as an estimate"));
            }
            if call["client_gone"] != true && n(&row["output_tokens"]) == 0 {
                problems.push(format!(
                    "{label}: a completed call billed no output ({row})"
                ));
            }
        } else if billed_path(path) {
            // Refused (4xx/5xx): at most one row, and it bills nothing.
            let billed: u64 = mine
                .iter()
                .map(|r| input_total(r) + n(&r["output_tokens"]))
                .sum();
            if mine.len() > 1 || billed != 0 {
                problems.push(format!(
                    "{label}: a refused call has {} rows billing {billed} tokens",
                    mine.len()
                ));
            }
        } else if !mine.is_empty() {
            problems.push(format!(
                "{label}: a free call has {} ai.usage rows",
                mine.len()
            ));
        }
    }
    if used != rows.len() {
        problems.push(format!(
            "{} ai.usage rows belong to no call the client made",
            rows.len() - used
        ));
    }
    (problems, matched)
}

/// The rows' token totals, normalized as BIL-5 does: `fresh` is uncached input on both wires.
fn totals(rows: &[&Value]) -> Totals {
    let mut t = Totals {
        requests: Some(0),
        ..Totals::default()
    };
    for row in rows {
        // Writes from breakpoints the gateway added bill as input (D76) but are writes in the
        // provider's report.
        let gw = n(&row["gateway_cache_write_tokens"]);
        let (read, write) = (
            n(&row["cache_read_tokens"]),
            n(&row["cache_write_tokens"]) + gw,
        );
        t.fresh += input_total(row).saturating_sub(read + write);
        t.cache_read += read;
        t.cache_write += write;
        t.output += n(&row["output_tokens"]);
        t.requests = t.requests.map(|r| r + 1);
    }
    t
}

/// What the rows cost at the card's prices (USD per million tokens, from the verdict).
fn cost(t: &Totals, pricing: &Value) -> f64 {
    let p = |k: &str| pricing[k].as_f64().unwrap_or(0.0) / 1e6;
    t.fresh as f64 * p("input")
        + t.cache_read as f64 * p("cache_read")
        + t.cache_write as f64 * p("cache_write")
        + t.output as f64 * p("output")
}

// --- LNG-1 / LNG-2 -------------------------------------------------------------------------------

fn long_session(
    claims: &str,
    spec: &str,
    route: Route,
    recon: Option<Provider>,
    keys: &BTreeMap<String, String>,
) -> Result<(), Failed> {
    let lng1 = claims.split('+').any(|c| c == "LNG-1");
    let lng2 = claims.split('+').any(|c| c == "LNG-2");
    // Reconciliation needs the pool key's id: find it before spending on a session.
    let recon = match recon {
        Some(p) => {
            let (pool_var, admin_var) = match p {
                Provider::OpenAi => ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY"),
                Provider::Anthropic => ("ANTHROPIC_API_KEY", "ANTHROPIC_ADMIN_KEY"),
            };
            match keys.get(admin_var) {
                Some(admin) => {
                    let id = recon::pool_key_id(p, &keys[pool_var], admin)?;
                    Some((p, admin.clone(), id))
                }
                None => None,
            }
        }
        None => None,
    };
    let gw = Gateway::boot(route, keys)?;
    let start = recon::now_secs() / 60 * 60;
    let verdict = gw.client("long", spec, route.model)?;
    let end = recon::now_secs().div_ceil(60) * 60 + 60;
    let detail = &verdict["detail"];
    let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
    let ids: Vec<&str> = calls
        .iter()
        .filter_map(|c| c["request_id"].as_str())
        .collect();
    let rows = gw.rows_for(&ids);
    export_rows(&rows);
    let (mut problems, matched) = ledger_problems(&calls, &rows, route.serves);

    let billed: Vec<usize> = (0..calls.len())
        .filter(|&i| {
            billed_path(calls[i]["path"].as_str().unwrap_or(""))
                && (200..300).contains(&n(&calls[i]["status"]))
        })
        .collect();
    let compactions = calls.iter().filter(|c| c["compaction"] == true).count();
    let harness_compactions = n(&detail["events"]["compactions"]);
    let session_rows: Vec<&Value> = matched.iter().flatten().copied().collect();
    let t = totals(&session_rows);
    let shares = cache_shares(&calls, &matched);
    eprintln!(
        "LNG {spec} on {}: task ok={} calls={} billed={} compaction requests={compactions} \
         harness compactions={harness_compactions} {t} cost=${:.4} cache shares (eligible \
         turns)={:?}",
        route.model,
        verdict["ok"],
        calls.len(),
        billed.len(),
        cost(&t, &detail["pricing"]),
        shares
            .iter()
            .map(|s| (s * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );

    if lng1 {
        if verdict["ok"] != true {
            problems.push(format!(
                "the task didn't complete: steps passing {}, tests changed {}, exit {}, test \
                 output {}\n  harness stdout tail: {}\n  stderr tail: {}",
                detail["steps_passing"],
                detail["tests_changed"],
                detail["exit"],
                detail["test_tail"],
                detail["stdout_tail"],
                detail["stderr_tail"]
            ));
        }
        if billed.len() < MIN_TURNS {
            problems.push(format!(
                "only {} billed model calls, want >= {MIN_TURNS}",
                billed.len()
            ));
        }
        if !spec.ends_with(":uncompacted") && (compactions == 0 || harness_compactions == 0) {
            problems.push(format!(
                "compaction never happened ({compactions} compaction requests on the wire, \
                 {harness_compactions} in the harness's events; knob {})",
                detail["knob"]
            ));
        }
        if compactions * TURNS_PER_COMPACTION > billed.len() {
            problems.push(format!(
                "{compactions} compaction requests in {} billed calls: a compaction loop",
                billed.len()
            ));
        }
    }
    if lng2 {
        problems.extend(cache_floor(&shares));
    }
    if lng1 && let Some((provider, admin, key_id)) = &recon {
        if problems.is_empty() {
            let mut by_minute: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
            for (call, row) in calls.iter().zip(&matched) {
                if let (Some(row), Some(t0)) = (row, call["t0"].as_f64()) {
                    by_minute.entry(t0 as u64 / 60 * 60).or_default().push(row);
                }
            }
            let by_minute = by_minute
                .into_iter()
                .map(|(m, r)| (m, totals(&r)))
                .collect();
            problems.extend(reconcile(
                *provider,
                admin,
                key_id,
                route.model,
                start,
                end,
                t,
                &by_minute,
            ));
        } else {
            problems.push("not reconciled: the session already failed".to_owned());
        }
    }
    gw.cleanup();
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{}\n--- gateway log ---\n{}",
            problems.join("\n"),
            tail(&gw.log)
        )
        .into())
    }
}

/// LNG-2: each eligible main turn's `cache_read / input_total`, in call order.
fn cache_shares(calls: &[Value], matched: &[Option<&Value>]) -> Vec<f64> {
    let mut out = Vec::new();
    let (mut main_turns, mut rewritten, mut last_items) = (0usize, false, 0u64);
    for (call, row) in calls.iter().zip(matched) {
        if call["compaction"] == true {
            rewritten = true;
            continue;
        }
        let Some(row) = row else { continue };
        if !(200..300).contains(&n(&call["status"])) || n(&call["tools"]) == 0 {
            continue;
        }
        let items = n(&call["items"]);
        let fresh_prefix = rewritten || items < last_items;
        last_items = items;
        rewritten = false;
        main_turns += 1;
        let input = input_total(row);
        if main_turns <= WARMUP || fresh_prefix || input < CACHE_ELIGIBLE {
            continue;
        }
        out.push(n(&row["cache_read_tokens"]) as f64 / input as f64);
    }
    out
}

fn cache_floor(shares: &[f64]) -> Vec<String> {
    if shares.len() < 10 {
        return vec![format!(
            "only {} turns eligible for the cache floor, want >= 10",
            shares.len()
        )];
    }
    let mean = shares.iter().sum::<f64>() / shares.len() as f64;
    let above = shares.iter().filter(|s| **s >= TURN_FLOOR).count() as f64 / shares.len() as f64;
    let mut problems = Vec::new();
    if mean < MEAN_FLOOR {
        problems.push(format!(
            "cache share after warm-up averages {mean:.3}, floor {MEAN_FLOOR}"
        ));
    }
    if above < TURN_SHARE {
        problems.push(format!(
            "only {:.0}% of turns read >= {TURN_FLOOR} of their prompt from cache, want >= {:.0}%",
            above * 100.0,
            TURN_SHARE * 100.0
        ));
    }
    problems
}

/// BIL-5's method on one session: poll the provider's usage report for the pool key, model and
/// minutes until it equals the ledger's totals and holds for a second poll. A surplus that holds
/// for four polls fails early. A failure carries a per-minute breakdown so a surplus can be placed:
/// in a minute the session made no call (someone else's traffic on the same key and model), or
/// among its own calls.
#[allow(clippy::too_many_arguments)]
fn reconcile(
    provider: Provider,
    admin: &str,
    key_id: &str,
    model: &str,
    start: u64,
    end: u64,
    ledger: Totals,
    by_minute: &BTreeMap<u64, Totals>,
) -> Vec<String> {
    let report = |from: u64, to: u64| match provider {
        Provider::OpenAi => recon::openai_usage(admin, key_id, model, from, to),
        Provider::Anthropic => recon::anthropic_usage(admin, key_id, model, from, to),
    };
    let deadline = Instant::now() + SETTLE_BUDGET;
    let (mut last, mut same): (Option<Totals>, usize) = (None, 0);
    loop {
        match report(start, end) {
            Ok(t) => {
                eprintln!("LNG-1 reconcile {model}: ledger {ledger}; provider reports {t}");
                if t.agrees(&ledger) && last == Some(t) {
                    return Vec::new();
                }
                // Totals only grow as the report catches up, so a surplus on any class that
                // holds for four polls is final; a shortfall may still be lag.
                let surplus = t.fresh > ledger.fresh
                    || t.cache_read > ledger.cache_read
                    || t.cache_write > ledger.cache_write
                    || t.output > ledger.output;
                same = if last == Some(t) && surplus {
                    same + 1
                } else {
                    0
                };
                last = Some(t);
                if same >= 3 {
                    break;
                }
            }
            Err(e) => eprintln!(
                "LNG-1 reconcile {model}: usage report failed: {}",
                e.message().unwrap_or("")
            ),
        }
        if Instant::now() + POLL_EVERY > deadline {
            break;
        }
        std::thread::sleep(POLL_EVERY);
    }
    let mut minutes = Vec::new();
    for m in (start..end).step_by(60) {
        let mine = by_minute.get(&m).copied().unwrap_or(Totals {
            requests: Some(0),
            ..Totals::default()
        });
        match report(m, m + 60) {
            Ok(t) if t.agrees(&mine) => {}
            Ok(t) => minutes.push(format!("{m}: ledger {mine} / provider {t}")),
            Err(e) => minutes.push(format!("{m}: {}", e.message().unwrap_or(""))),
        }
    }
    vec![format!(
        "the provider's usage report never settled on the session's ledger\n  \
         ledger:   {ledger}\n  provider: {}\n  (key {key_id}, model {model}, window {start}..{end})\n  \
         minutes that differ (a minute with no ledger requests is another suite's traffic on the \
         same key and model):\n    {}",
        last.map_or("no report".to_owned(), |t| t.to_string()),
        minutes.join("\n    ")
    )]
}

// --- TOOL-1 --------------------------------------------------------------------------------------

fn tool_case(
    route: Route,
    scenario: &str,
    expect: Expect,
    keys: &BTreeMap<String, String>,
) -> Result<(), Failed> {
    let gw = Gateway::boot(route, keys)?;
    let verdict = gw.client("tools", scenario, route.model)?;
    let d = &verdict["detail"];
    let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
    let ids: Vec<&str> = calls
        .iter()
        .filter_map(|c| c["request_id"].as_str())
        .collect();
    let rows = gw.rows_for(&ids);
    export_rows(&rows);
    let (mut problems, _) = ledger_problems(&calls, &rows, route.serves);
    let status = d["status"].as_u64();
    let error = d["error"].as_str().unwrap_or("");
    eprintln!(
        "TOOL-1 {scenario} on {}: status {status:?} called {} error {}",
        route.model,
        d["called"],
        &error[..error.len().min(300)]
    );
    match (expect, status) {
        (_, Some(s)) if s >= 500 => {
            problems.push(format!("a 5xx ({s}) for a large tool set: {error}"));
        }
        (_, None) => problems.push(format!("the request never completed: {d}")),
        (Expect::Works, Some(200)) => {
            if d["called_last"] != true {
                problems.push(format!(
                    "the model didn't call {} (called {}): a tool was dropped or renamed",
                    d["last"], d["called"]
                ));
            }
        }
        (Expect::Works, Some(s)) => {
            problems.push(format!("{s} below the provider's limit: {error}"));
        }
        (Expect::Limit(limit), Some(s)) if (400..500).contains(&s) => {
            if !error.contains(limit) {
                problems.push(format!("the {s} doesn't name the limit ({limit}): {error}"));
            }
        }
        (Expect::Limit(limit), Some(s)) => problems.push(format!(
            "{s} above the provider's {limit}-tool limit (want a 4xx naming it); called {}",
            d["called"]
        )),
    }
    gw.cleanup();
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n").into())
    }
}

fn mcp_case(
    client: &str,
    route: Route,
    expect: Expect,
    keys: &BTreeMap<String, String>,
) -> Result<(), Failed> {
    let gw = Gateway::boot(route, keys)?;
    let verdict = gw.client("mcp", client, route.model)?;
    let d = &verdict["detail"];
    let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
    let ids: Vec<&str> = calls
        .iter()
        .filter_map(|c| c["request_id"].as_str())
        .collect();
    let rows = gw.rows_for(&ids);
    export_rows(&rows);
    let (mut problems, _) = ledger_problems(&calls, &rows, route.serves);
    let offered: Vec<(u64, u64)> = calls
        .iter()
        .map(|c| (n(&c["tools"]), n(&c["namespace_members"])))
        .collect();
    eprintln!(
        "TOOL-1 mcp {client} on {}: found secret {} calls {} (tools, namespace members) per \
         call {offered:?}",
        route.model,
        d["found_secret"],
        calls.len()
    );
    for c in &calls {
        if n(&c["status"]) >= 500 {
            problems.push(format!(
                "a {} on {}: {}",
                c["status"], c["path"], c["error_body"]
            ));
        }
    }
    let refusals: Vec<String> = calls
        .iter()
        .filter(|c| (400..500).contains(&n(&c["status"])))
        .map(|c| c["error_body"].as_str().unwrap_or("").to_owned())
        .collect();
    match expect {
        Expect::Works if verdict["ok"] != true => problems.push(format!(
            "the agent never reported the secret of the last MCP tool; 4xx bodies {refusals:?}\n  \
             stdout tail: {}\n  stderr tail: {}",
            d["stdout_tail"], d["stderr_tail"]
        )),
        Expect::Works => {}
        Expect::Limit(limit) => {
            let shown = d["stdout_tail"].as_str().unwrap_or("");
            if verdict["ok"] == true {
                // It worked after all (a harness that defers its MCP tools): fine for the claim.
            } else if refusals.is_empty() || refusals.iter().any(|b| !b.contains(limit)) {
                problems.push(format!(
                    "the session failed without a 4xx naming the limit ({limit}): {refusals:?}\n  \
                     stdout tail: {shown}"
                ));
            } else if !shown.contains(limit) {
                problems.push(format!(
                    "the harness didn't show the refusal naming the limit: {shown}"
                ));
            }
        }
    }
    gw.cleanup();
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n").into())
    }
}
