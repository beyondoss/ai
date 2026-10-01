//! Live catalog sweep: every in-scope catalog candidate, forced through a gateway built from this
//! commit with `x-beyond-only` (or `x-beyond-order`), checked against what its row claims.
//!
//! One trial per (claim group, candidate), named `CLAIMS::raw::ROUTE::ROW` like the live cells in
//! `live.rs`, so `verify status` attributes each outcome to its claims (client `raw`: plain HTTP,
//! no SDK). ROUTE is the candidate's provider (`openai-responses` for a GPT row's Responses arm,
//! `all` for a whole-row trial); for CAT-13 it is the provider and ROW is `managed` / `byo`.
//!
//! | Trial              | Per         | Asserts                                                       |
//! | ------------------ | ----------- | ------------------------------------------------------------- |
//! | `CAT-1`            | candidate   | 200; echoed model is the candidate's id (snapshot/prefix      |
//! |                    |             | aside); `x-beyond-upstream-model` is the id; one ledger row   |
//! | `BIL-13`           | candidate   | its ledger row (served or not) prices at the row; on a 200    |
//! |                    |             | the billed `model` is the echo                                |
//! | `CAT-2`            | row         | every candidate echoes the same model family and snapshot     |
//! | `CAT-3`            | candidate   | one request over `context_window` is a clean 4xx from that    |
//! |                    |             | candidate (no failover); rows <= 200k also take 0.9x once     |
//! | `CAT-4`            | candidate   | `max_tokens` = card `max_output_tokens` is accepted           |
//! | `CAT-5`            | candidate   | advertised image / PDF input is read; unadvertised image 4xx  |
//! | `CAT-6`            | candidate   | forced tool call, json_schema output; no tools: clean 4xx;    |
//! |                    |             | a candidate that can't honor json_schema is skipped           |
//! | `CAT-6+BIL-9`      | candidate   | reasoning reported; ledger output counts it exactly once      |
//! | `CAT-7`            | candidate   | ledger tokens x card = provider-reported cost (OpenRouter,    |
//! |                    |             | xAI) within 2%; listed price (Together, xAI, OpenRouter);     |
//! |                    |             | else card = `verify/catalog_truth.toml` (vendor-doc verified) |
//! | `CAT-8`            | row         | name / created / owner against the provider's model endpoint  |
//! | `CAT-13`           | provider    | `/{provider}/` passthrough, managed (billed) and BYO (not)    |
//!
//! Scope (CAT-1): OpenAI, Anthropic, OpenRouter, xAI, Bedrock and Together candidates. Mistral,
//! Groq, DeepSeek and Fireworks are out of scope by owner decision; `openai-codex` has no key.
//! A trial that can't run (no key, over the per-call cost cap, a window the provider exceeds so an
//! over-limit request would be billed) is not listed — `VERIFY_CATALOG_PLAN=1` prints each skip and
//! its reason, and the estimated cost of what is listed.
//!
//! Cost: every billed call appends a record to `target/catalog-live/<sweep>.jsonl`
//! (`VERIFY_CATALOG_SWEEP`, default `sweep`), priced from the ledger's tokens at the card;
//! `VERIFY_CATALOG_SUMMARY=1` totals it. Listed only with `VERIFY_LIVE=1`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use base64::Engine as _;
use libtest_mimic::{Arguments, Failed, Trial};
use providers::catalog::{
    Candidate, IN_FILE, IN_IMAGE, MODEL_ROUTES, ModelRoute, REASONING, STRUCTURED_OUTPUTS, TOOLS,
    serves_file_input, serves_structured_outputs,
};
use providers::{ProviderId, by_id};
use serde_json::{Map, Value, json};

/// The dev signing key and tenant-1 token (the same constants as `live.rs`).
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// In-scope providers and the env var holding each pool key.
const SCOPE: &[(&str, &str)] = &[
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("xai", "XAI_API_KEY"),
    ("bedrock", "AWS_BEARER_TOKEN_BEDROCK"),
    ("together", "TOGETHER_API_KEY"),
];

/// Most one call may be estimated to cost before its trial is left out (USD). Override with
/// `VERIFY_CATALOG_CALL_CAP`.
const CALL_CAP: f64 = 0.08;
/// Most the one near-limit context call per row may be estimated to cost (USD).
const NEAR_CAP: f64 = 0.25;
/// Rows at or under this window get the near-limit (0.9x) call.
const NEAR_MAX_WINDOW: u32 = 200_000;
/// Provider-reported cost vs ledger tokens priced at the card.
const COST_TOLERANCE: f64 = 0.02;
/// Card `created` vs the provider's, in seconds.
const CREATED_TOLERANCE: i64 = 2 * 86_400;

const IMAGE_WORD: &str = "KESTREL";
const FILE_WORD: &str = "OSPREY";
const PNG: &[u8] = include_bytes!("fixtures/codeword.png");

// ---------------------------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sweep_dir() -> PathBuf {
    let d = repo_root().join("target/catalog-live");
    let _ = std::fs::create_dir_all(&d);
    d
}

fn fresh(path: &Path, max_age: Duration) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| SystemTime::now().duration_since(t).ok())
        .is_some_and(|age| age < max_age)
}

/// `.env` at the repo root, then the process environment (which wins), then a Bedrock key minted
/// from the AWS credential chain (cached for an hour: one nextest process per trial would
/// otherwise mint one each).
fn keys() -> &'static BTreeMap<String, String> {
    static KEYS: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut out = BTreeMap::new();
        if let Ok(s) = std::fs::read_to_string(repo_root().join(".env")) {
            for line in s.lines().map(str::trim) {
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
        if !out.contains_key("AWS_BEARER_TOKEN_BEDROCK") {
            let cache = sweep_dir().join("bedrock.tok");
            let token = if fresh(&cache, Duration::from_secs(3600)) {
                std::fs::read_to_string(&cache).unwrap_or_default()
            } else {
                Command::new(repo_root().join("verify/clients/py/.venv/bin/python"))
                    .arg(repo_root().join("verify/clients/py/bedrock_token.py"))
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
                    .inspect(|t| {
                        if !t.is_empty() {
                            let _ = std::fs::write(&cache, t);
                        }
                    })
                    .unwrap_or_default()
            };
            if !token.trim().is_empty() {
                out.insert("AWS_BEARER_TOKEN_BEDROCK".into(), token.trim().to_owned());
            }
        }
        out
    })
}

fn key_of(provider: &str) -> Option<&'static str> {
    let var = SCOPE.iter().find(|(p, _)| *p == provider)?.1;
    keys().get(var).map(String::as_str)
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

fn http() -> &'static reqwest::blocking::Client {
    static C: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    C.get_or_init(|| {
        reqwest::blocking::Client::builder()
            // Under nextest's 180s terminate-after, so a slow reasoning call fails by name.
            .timeout(Duration::from_secs(170))
            .build()
            .unwrap()
    })
}

fn call_cap() -> f64 {
    std::env::var("VERIFY_CATALOG_CALL_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CALL_CAP)
}

// ---------------------------------------------------------------------------------------------
// Provider model listings (CAT-3 safety, CAT-7 list prices, CAT-8 metadata)
// ---------------------------------------------------------------------------------------------

/// A provider's model listing, fetched once and cached on disk for six hours.
fn listing(provider: &str) -> Option<&'static Value> {
    static CACHE: OnceLock<Mutex<BTreeMap<String, Option<&'static Value>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut cache = cache.lock().unwrap();
    if let Some(v) = cache.get(provider) {
        return *v;
    }
    let path = sweep_dir().join(format!("listing-{provider}.json"));
    let cached = fresh(&path, Duration::from_secs(6 * 3600))
        .then(|| std::fs::read_to_string(&path).ok())
        .flatten()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let v = cached.or_else(|| {
        let req = match provider {
            "openrouter" => http().get("https://openrouter.ai/api/v1/models"),
            "anthropic" => http()
                .get("https://api.anthropic.com/v1/models?limit=1000")
                .header("x-api-key", key_of("anthropic")?)
                .header("anthropic-version", "2023-06-01"),
            "openai" => http()
                .get("https://api.openai.com/v1/models")
                .bearer_auth(key_of("openai")?),
            "xai" => http()
                .get("https://api.x.ai/v1/language-models")
                .bearer_auth(key_of("xai")?),
            "together" => http()
                .get("https://api.together.xyz/v1/models")
                .bearer_auth(key_of("together")?),
            _ => return None,
        };
        let v: Value = req.send().ok()?.error_for_status().ok()?.json().ok()?;
        let _ = std::fs::write(&path, v.to_string());
        Some(v)
    });
    let leaked: Option<&'static Value> = v.map(|v| &*Box::leak(Box::new(v)));
    cache.insert(provider.to_owned(), leaked);
    leaked
}

/// One model's entry in a provider listing, matched on the id (and xAI's aliases).
fn listed(provider: &str, id: &str) -> Option<&'static Value> {
    let l = listing(provider)?;
    let items = l
        .get("data")
        .or_else(|| l.get("models"))
        .and_then(Value::as_array)
        .or_else(|| l.as_array())?;
    items
        .iter()
        .find(|m| {
            m["id"].as_str().is_some_and(|x| x.eq_ignore_ascii_case(id))
                || m["aliases"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(id)))
        })
        // Anthropic lists only dated snapshots (`claude-haiku-4-5-20251001`).
        .or_else(|| {
            items.iter().find(|m| {
                m["id"]
                    .as_str()
                    .is_some_and(|x| x.starts_with(id) && family(x).0 == family(id).0)
            })
        })
}

/// The input window the provider itself advertises for a candidate, where its listing says.
fn provider_window(provider: &str, id: &str) -> Option<u64> {
    let m = listed(provider, id)?;
    match provider {
        "anthropic" => m["max_input_tokens"].as_u64(),
        "bedrock" => provider_window("anthropic", &bedrock_to_anthropic(id)?),
        "openrouter" | "together" => m["context_length"].as_u64(),
        _ => None,
    }
}

/// `us.anthropic.claude-haiku-4-5-20251001-v1:0` → `claude-haiku-4-5-20251001`.
fn bedrock_to_anthropic(id: &str) -> Option<String> {
    let rest = id.split_once("anthropic.")?.1;
    let rest = rest.split_once("-v").map_or(rest, |(a, _)| a);
    Some(rest.to_owned())
}

// ---------------------------------------------------------------------------------------------
// The gateway: one per process, every in-scope pool key, plus two config-added providers.
// ---------------------------------------------------------------------------------------------

struct Gateway {
    base: String,
    log: PathBuf,
}

static CHILDREN: Mutex<Vec<Child>> = Mutex::new(Vec::new());

/// A free port below the kernel's ephemeral range (as `live.rs` picks them): Pingora binds with
/// `SO_REUSEPORT`, so a `bind(0)` port another session's gateway also picked is shared silently.
fn free_port() -> u16 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let _ = NEXT.compare_exchange(
        0,
        u64::from(std::process::id())
            ^ SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
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

fn gateway() -> Result<&'static Gateway, String> {
    static GW: OnceLock<Result<Gateway, String>> = OnceLock::new();
    GW.get_or_init(boot).as_ref().map_err(Clone::clone)
}

fn boot() -> Result<Gateway, String> {
    let dir = sweep_dir().join(format!("gw-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let nats_port = free_port();
    let nats = Command::new("nats-server")
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
        .map_err(|e| format!("nats-server: {e}"))?;
    CHILDREN.lock().unwrap().push(nats);

    let (port, metrics_port) = (free_port(), free_port());
    let mut cfg = format!(
        "listen = \"127.0.0.1:{port}\"\nmetrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\n\
         upstream_tls = true\nsmart_router = false\n\n[pool_keys]\n"
    );
    for (provider, _) in SCOPE {
        if let Some(k) = key_of(provider) {
            cfg.push_str(&format!("{provider} = [{k:?}]\n"));
        }
    }
    // CAT-13's config-added providers: Bedrock's OpenAI-compat surface (OpenAI wire, Bearer) and
    // Anthropic under a second name (Anthropic wire, x-api-key).
    if let Some(k) = key_of("bedrock") {
        cfg.push_str(&format!("bedrock-openai = [{k:?}]\n"));
    }
    if let Some(k) = key_of("anthropic") {
        cfg.push_str(&format!("anthropic-alt = [{k:?}]\n"));
    }
    cfg.push_str(
        "\n[provider_authorities]\nbedrock-openai = \"bedrock-runtime.us-east-1.amazonaws.com:443\"\n\
         anthropic-alt = \"api.anthropic.com:443\"\n\n[provider_dialects]\nbedrock-openai = \"openai\"\n\
         anthropic-alt = \"anthropic\"\n\n[provider_auth_schemes]\nbedrock-openai = \"bearer\"\n\
         anthropic-alt = \"x-api-key\"\n",
    );
    cfg.push_str(&format!("\n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"));
    let cfg_path = dir.join("gateway.toml");
    std::fs::write(&cfg_path, cfg).map_err(|e| e.to_string())?;

    let log = dir.join("gateway.log");
    let f = std::fs::File::create(&log).map_err(|e| e.to_string())?;
    let gw = Command::new(gateway_bin())
        .args(["run", "-c"])
        .arg(&cfg_path)
        .env("AI_LOG", "warn,ai.usage=info")
        .stdout(f.try_clone().map_err(|e| e.to_string())?)
        .stderr(f)
        .spawn()
        .map_err(|e| format!("gateway: {e}"))?;
    CHILDREN.lock().unwrap().push(gw);

    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", metrics_port)) {
            let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
            let _ = s.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
            let mut body = String::new();
            let _ = s.read_to_string(&mut body);
            if body.lines().any(|l| l.starts_with("ai_allowance_ready 1")) {
                return Ok(Gateway {
                    base: format!("http://127.0.0.1:{port}"),
                    log,
                });
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(format!("gateway never became ready: {}", tail(&log)))
}

fn cleanup() {
    for mut c in CHILDREN.lock().unwrap().drain(..) {
        let _ = c.kill();
        let _ = c.wait();
    }
    let _ = std::fs::remove_dir_all(sweep_dir().join(format!("gw-{}", std::process::id())));
}

fn tail(path: &Path) -> String {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    let start = s.len().saturating_sub(1500);
    s[s.floor_char_boundary(start)..].to_owned()
}

/// The `ai.usage` rows with this request id, waiting a moment for them to land.
fn ledger(request_id: &str, want: usize) -> Vec<Value> {
    let Ok(gw) = gateway() else {
        return Vec::new();
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let rows: Vec<Value> = std::fs::File::open(&gw.log)
            .map(|f| {
                BufReader::new(f)
                    .lines()
                    .map_while(Result::ok)
                    .filter(|l| l.contains("\"ai.usage\"") && l.contains(request_id))
                    .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
                    .map(|v| v.get("fields").cloned().unwrap_or(v))
                    .filter(|r| r["request_id"] == request_id)
                    .collect()
            })
            .unwrap_or_default();
        if rows.len() >= want || Instant::now() >= deadline {
            return rows;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

// ---------------------------------------------------------------------------------------------
// Arms and requests
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Wire {
    Chat,
    Responses,
    Embeddings,
}

/// One in-scope candidate of a row, on the inbound endpoint that reaches it.
#[derive(Clone, Copy)]
struct Arm {
    row: &'static ModelRoute,
    cand: Candidate,
    wire: Wire,
}

impl Arm {
    fn provider(&self) -> &'static str {
        by_id(self.cand.provider).name
    }
    fn route(&self) -> &'static str {
        if self.wire == Wire::Responses {
            "openai-responses"
        } else {
            self.provider()
        }
    }
    fn has(&self, bit: u8) -> bool {
        self.row.card.input & bit != 0
    }
    fn can(&self, bit: u8) -> bool {
        self.row.card.features & bit != 0
    }
    fn claude(&self) -> bool {
        self.cand.upstream_model.contains("claude")
    }
    /// Claude 4.6 and later, whose thinking is adaptive; 4.5 and earlier (and Sonnet 4) think
    /// whenever a budget is set.
    fn adaptive_claude(&self) -> bool {
        let id = self.cand.upstream_model;
        self.claude()
            && !["4-5", "4.5", "4-1", "4.1"].iter().any(|v| id.contains(v))
            && !id.ends_with("sonnet-4")
    }
    fn pro(&self) -> bool {
        self.cand.upstream_model.ends_with("-pro")
    }
    /// Reasons before answering even when nobody asked (OpenAI reasoning models, grok, the open
    /// reasoning models); Claude thinks only when asked.
    fn thinks(&self) -> bool {
        self.can(REASONING) && !self.claude()
    }
    fn price(&self) -> (f64, f64) {
        let p = |s: &str| s.parse::<f64>().unwrap_or(0.0);
        (p(self.row.price.input), p(self.row.price.output))
    }
    fn est(&self, input: u64, output: u64) -> f64 {
        let (pi, po) = self.price();
        (input as f64 * pi + output as f64 * po) / 1e6
    }
    /// Output tokens an answer of a word or a tool call is expected to take, reasoning included.
    fn est_out(&self) -> u64 {
        match (self.thinks(), self.pro()) {
            (true, true) => 1500,
            (true, false) => 250,
            _ => 30,
        }
    }
    /// The budget for the minimal serve probe: 16 tokens, or room for a reasoning model to finish
    /// (OpenAI's o-series 400s "could not finish the message" when reasoning eats a tiny budget).
    fn basic_max(&self) -> u32 {
        if self.thinks() && self.wire == Wire::Chat && self.provider() == "openai" && !self.pro() {
            1024
        } else {
            16
        }
    }
    /// The token budget for a short answer: room to reason first on a model that will.
    fn short_max(&self) -> u32 {
        if self.thinks() || self.can(REASONING) {
            self.row.card.max_output_tokens.min(4096)
        } else {
            64
        }
    }
}

fn in_scope(c: &Candidate) -> bool {
    let name = by_id(c.provider).name;
    SCOPE.iter().any(|(p, _)| *p == name) && key_of(name).is_some()
}

fn arms(row: &'static ModelRoute) -> Vec<Arm> {
    let mut out: Vec<Arm> = row
        .candidates
        .iter()
        .filter(|c| in_scope(c))
        .map(|&cand| Arm {
            row,
            cand,
            wire: if cand.path.ends_with("/embeddings") {
                Wire::Embeddings
            } else {
                Wire::Chat
            },
        })
        .collect();
    out.extend(
        row.responses
            .iter()
            .filter(|c| in_scope(c))
            .map(|&cand| Arm {
                row,
                cand,
                wire: Wire::Responses,
            }),
    );
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Walk {
    /// `x-beyond-only`: this candidate and nothing else.
    Only,
    /// `x-beyond-order`: this candidate first, the rest of the row still failover.
    Order,
}

struct Reply {
    status: u16,
    provider: Option<String>,
    upstream: Option<String>,
    request_id: Option<String>,
    json: Value,
    raw: String,
}

impl Reply {
    fn excerpt(&self) -> String {
        let s = self.raw.trim();
        let end = s.floor_char_boundary(s.len().min(400));
        s[..end].to_owned()
    }
}

fn send(req: reqwest::blocking::RequestBuilder) -> Result<Reply, String> {
    let r = req.send().map_err(|e| format!("transport: {e}"))?;
    let h = |n: &str| {
        r.headers()
            .get(n)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    let (status, provider, upstream, request_id) = (
        r.status().as_u16(),
        h("x-beyond-provider"),
        h("x-beyond-upstream-model"),
        h("x-beyond-request-id"),
    );
    let raw = r.text().map_err(|e| format!("body: {e}"))?;
    Ok(Reply {
        status,
        provider,
        upstream,
        request_id,
        json: serde_json::from_str(&raw).unwrap_or(Value::Null),
        raw,
    })
}

/// One of [`UPLOAD_SLOTS`] cross-process locks; released when the file drops.
fn upload_slot() -> std::fs::File {
    const UPLOAD_SLOTS: usize = 3;
    loop {
        for i in 0..UPLOAD_SLOTS {
            if let Ok(f) = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(sweep_dir().join(format!("upload-{i}.lock")))
                && f.try_lock().is_ok()
            {
                return f;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn call(arm: &Arm, body: &Value, walk: Walk) -> Result<Reply, String> {
    let gw = gateway()?;
    let path = match arm.wire {
        Wire::Chat => "/v1/chat/completions",
        Wire::Responses => "/v1/responses",
        Wire::Embeddings => "/v1/embeddings",
    };
    let walk_header = match walk {
        Walk::Only => "x-beyond-only",
        Walk::Order => "x-beyond-order",
    };
    let bytes = serde_json::to_vec(body).map_err(|e| e.to_string())?;
    // Multi-megabyte context probes share a few upload slots across every trial process: sixteen
    // at once saturate the uplink, and an upstream that answers early then stalls the relay.
    let _slot = (bytes.len() > 1 << 20).then(upload_slot);
    // A 502/503/504/529 is retried twice, as every provider SDK does: a stale pooled upstream
    // connection, or a provider's momentary overload, is not a catalog fact.
    let mut attempt: u64 = 0;
    loop {
        let r = send(
            http()
                .post(format!("{}{path}", gw.base))
                .bearer_auth(DEV_TOKEN)
                .header("x-beyond-model", arm.row.model)
                .header(walk_header, arm.provider())
                .header("x-beyond-cache", "off")
                .header("content-type", "application/json")
                .body(bytes.clone()),
        )?;
        let busy = r.status == 429 && r.raw.contains("rate-limited");
        if attempt < 2 && (busy || matches!(r.status, 502 | 503 | 504 | 529)) {
            attempt += 1;
            println!("retry {attempt} after HTTP {}: {}", r.status, r.excerpt());
            std::thread::sleep(Duration::from_secs(if busy { 8 } else { 2 } * attempt));
            continue;
        }
        return Ok(r);
    }
}

/// What a request carries besides its prompt.
#[derive(Default, Clone, Copy)]
struct Opts {
    image: bool,
    file: bool,
    tools: bool,
    schema: bool,
    reason: bool,
    cost: bool,
}

fn data_uri(mime: &str, bytes: &[u8]) -> String {
    format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// A one-page PDF whose text layer says the code word.
fn pdf() -> Vec<u8> {
    let stream = format!("BT /F1 24 Tf 72 700 Td (The code word is {FILE_WORD}.) Tj ET");
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_owned(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
         /Resources << /Font << /F1 5 0 R >> >> >>"
            .to_owned(),
        format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
    ];
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for off in offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objs.len() + 1
        )
        .as_bytes(),
    );
    out
}

const SCHEMA_NAME: &str = "answer";

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": { "answer": { "type": "integer" } },
        "required": ["answer"],
        "additionalProperties": false
    })
}

fn tool_params() -> Value {
    json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
        "additionalProperties": false
    })
}

/// A request body for the arm's inbound endpoint.
fn body(arm: &Arm, prompt: &str, max: Option<u32>, o: Opts) -> Value {
    let mut m = Map::new();
    m.insert("model".into(), json!(arm.row.model));
    // With tools too: that is the request OpenAI's Chat Completions refuses on gpt-5.4 and later,
    // which those rows now reach over Responses (D114).
    let low_effort = arm.provider() == "openai" && arm.can(REASONING) && !arm.pro();
    match arm.wire {
        Wire::Embeddings => {
            m.insert("input".into(), json!(prompt));
        }
        Wire::Chat => {
            let content = if o.image || o.file {
                let mut parts = vec![json!({"type": "text", "text": prompt})];
                if o.image {
                    parts.push(json!({"type": "image_url", "image_url": {"url": data_uri("image/png", PNG)}}));
                }
                if o.file {
                    parts.push(json!({"type": "file", "file": {"filename": "codeword.pdf", "file_data": data_uri("application/pdf", &pdf())}}));
                }
                Value::Array(parts)
            } else {
                json!(prompt)
            };
            m.insert(
                "messages".into(),
                json!([{"role": "user", "content": content}]),
            );
            if let Some(max) = max {
                // OpenAI's reasoning models reject `max_tokens`; every OpenAI chat model takes
                // `max_completion_tokens`. Other hosts keep the classic name.
                let field = if arm.provider() == "openai" {
                    "max_completion_tokens"
                } else {
                    "max_tokens"
                };
                m.insert(field.into(), json!(max));
            }
            if o.tools {
                m.insert(
                    "tools".into(),
                    json!([{"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city.",
                    "parameters": tool_params()}}]),
                );
                m.insert(
                    "tool_choice".into(),
                    json!({"type": "function", "function": {"name": "get_weather"}}),
                );
            }
            if o.schema {
                m.insert(
                    "response_format".into(),
                    json!({"type": "json_schema", "json_schema": {
                    "name": SCHEMA_NAME, "strict": true, "schema": schema()}}),
                );
            }
            if o.reason && arm.provider() != "xai" {
                // grok reasons unasked and 400s `reasoning_effort` on its reasoning models.
                m.insert("reasoning_effort".into(), json!("low"));
            } else if low_effort && !o.reason {
                m.insert("reasoning_effort".into(), json!("low"));
            }
            if o.cost && arm.provider() == "openrouter" {
                m.insert("usage".into(), json!({"include": true}));
            }
        }
        Wire::Responses => {
            let mut parts = vec![json!({"type": "input_text", "text": prompt})];
            if o.image {
                parts.push(json!({"type": "input_image", "image_url": data_uri("image/png", PNG)}));
            }
            if o.file {
                parts.push(json!({"type": "input_file", "filename": "codeword.pdf", "file_data": data_uri("application/pdf", &pdf())}));
            }
            m.insert("input".into(), json!([{"role": "user", "content": parts}]));
            if let Some(max) = max {
                m.insert("max_output_tokens".into(), json!(max.max(16)));
            }
            if o.tools {
                m.insert(
                    "tools".into(),
                    json!([{"type": "function", "name": "get_weather",
                    "description": "Current weather for a city.", "parameters": tool_params(),
                    "strict": true}]),
                );
                m.insert(
                    "tool_choice".into(),
                    json!({"type": "function", "name": "get_weather"}),
                );
            }
            if o.schema {
                m.insert(
                    "text".into(),
                    json!({"format": {"type": "json_schema",
                    "name": SCHEMA_NAME, "strict": true, "schema": schema()}}),
                );
            }
            if (o.reason || low_effort) && !arm.pro() {
                m.insert("reasoning".into(), json!({"effort": "low"}));
            }
        }
    }
    Value::Object(m)
}

/// What a reply says, on any of the three wires.
#[derive(Debug, Default)]
struct Out {
    model: String,
    text: String,
    tool_calls: Vec<(String, String)>,
    input: u64,
    output: u64,
    reasoning: Option<u64>,
    reasoning_text: bool,
    provider_cost: Option<f64>,
}

fn parse(wire: Wire, v: &Value) -> Out {
    let mut o = Out {
        model: v["model"].as_str().unwrap_or_default().to_owned(),
        ..Out::default()
    };
    let u = &v["usage"];
    match wire {
        Wire::Embeddings => {
            o.input = u["prompt_tokens"].as_u64().unwrap_or(0);
            o.provider_cost = u["cost"].as_f64();
        }
        Wire::Chat => {
            let msg = &v["choices"][0]["message"];
            o.text = match &msg["content"] {
                Value::String(s) => s.clone(),
                Value::Array(a) => a.iter().filter_map(|p| p["text"].as_str()).collect(),
                _ => String::new(),
            };
            for tc in msg["tool_calls"].as_array().into_iter().flatten() {
                o.tool_calls.push((
                    tc["function"]["name"].as_str().unwrap_or_default().into(),
                    tc["function"]["arguments"]
                        .as_str()
                        .unwrap_or_default()
                        .into(),
                ));
            }
            o.input = u["prompt_tokens"].as_u64().unwrap_or(0);
            o.output = u["completion_tokens"].as_u64().unwrap_or(0);
            o.reasoning = u["completion_tokens_details"]["reasoning_tokens"].as_u64();
            let nonempty = |x: &Value| match x {
                Value::String(s) => !s.trim().is_empty(),
                Value::Array(a) => !a.is_empty(),
                _ => false,
            };
            o.reasoning_text = nonempty(&msg["reasoning"])
                || nonempty(&msg["reasoning_content"])
                || nonempty(&msg["reasoning_details"])
                || nonempty(&msg["thinking"])
                || o.text.contains("<think>");
            o.provider_cost = u["cost"]
                .as_f64()
                .or_else(|| u["cost_in_usd_ticks"].as_f64().map(|t| t / 1e10));
        }
        Wire::Responses => {
            for item in v["output"].as_array().into_iter().flatten() {
                match item["type"].as_str() {
                    Some("message") => {
                        for c in item["content"].as_array().into_iter().flatten() {
                            if let Some(t) = c["text"].as_str() {
                                o.text.push_str(t);
                            }
                        }
                    }
                    Some("function_call") => o.tool_calls.push((
                        item["name"].as_str().unwrap_or_default().into(),
                        item["arguments"].as_str().unwrap_or_default().into(),
                    )),
                    Some("reasoning") => o.reasoning_text = true,
                    _ => {}
                }
            }
            o.input = u["input_tokens"].as_u64().unwrap_or(0);
            o.output = u["output_tokens"].as_u64().unwrap_or(0);
            o.reasoning = u["output_tokens_details"]["reasoning_tokens"].as_u64();
            // xAI's Responses usage carries its charge.
            o.provider_cost = u["cost_in_usd_ticks"].as_f64().map(|t| t / 1e10);
        }
    }
    o
}

// ---------------------------------------------------------------------------------------------
// Cost records
// ---------------------------------------------------------------------------------------------

fn sweep_file() -> PathBuf {
    let id = std::env::var("VERIFY_CATALOG_SWEEP").unwrap_or_else(|_| "sweep".into());
    sweep_dir().join(format!("{id}.jsonl"))
}

/// Ledger row priced at the row's card: fresh input, output, cache reads and writes.
fn priced(row: &ModelRoute, l: &Value) -> f64 {
    let p = |s: &str| s.parse::<f64>().unwrap_or(f64::NAN) / 1e6;
    let n = |k: &str| l[k].as_u64().unwrap_or(0) as f64;
    let (input, read, write) = (
        n("input_tokens"),
        n("cache_read_tokens"),
        n("cache_write_tokens"),
    );
    let fresh = if l["usage_wire"] == "anthropic" {
        input
    } else {
        input - read - write
    };
    fresh * p(row.price.input)
        + n("output_tokens") * p(row.price.output)
        + read * p(row.price.cache_read)
        + write * p(row.price.cache_write)
}

fn record(trial: &str, arm: &Arm, probe: &str, l: &Value, out: &Out) {
    let rec = json!({
        "trial": trial, "row": arm.row.model, "provider": arm.provider(), "route": arm.route(),
        "probe": probe, "input": l["input_tokens"], "output": l["output_tokens"],
        "cache_read": l["cache_read_tokens"], "cache_write": l["cache_write_tokens"],
        "reasoning": out.reasoning, "usd": priced(arm.row, l), "provider_usd": out.provider_cost,
        "at": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    });
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(sweep_file())
    {
        let _ = f.write_all(format!("{rec}\n").as_bytes());
    }
}

/// Note a non-billed observation (vendor-doc verified, skipped sub-check) in the sweep file.
fn note(trial: &str, text: &str) {
    println!("note: {text}");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(sweep_file())
    {
        let _ = f.write_all(format!("{}\n", json!({"trial": trial, "note": text})).as_bytes());
    }
}

// ---------------------------------------------------------------------------------------------
// Shared checks
// ---------------------------------------------------------------------------------------------

/// A 200 from the forced candidate with exactly one ledger row naming it and pricing at the row.
/// Records the call's cost.
fn served(trial: &str, arm: &Arm, probe: &str, r: &Reply) -> Result<(Out, Value), String> {
    if r.status != 200 {
        return Err(format!("{probe}: HTTP {}: {}", r.status, r.excerpt()));
    }
    if r.provider.as_deref() != Some(arm.provider()) {
        return Err(format!(
            "{probe}: served by {:?}, forced {}",
            r.provider,
            arm.provider()
        ));
    }
    let out = parse(arm.wire, &r.json);
    let id = r
        .request_id
        .as_deref()
        .ok_or(format!("{probe}: no x-beyond-request-id"))?;
    let rows = ledger(id, 1);
    let [row] = rows.as_slice() else {
        return Err(format!(
            "{probe}: {} ai.usage rows for {id}, want 1",
            rows.len()
        ));
    };
    record(trial, arm, probe, row, &out);
    if row["provider"] != arm.provider() {
        return Err(format!(
            "{probe}: ledger names {}, forced {}",
            row["provider"],
            arm.provider()
        ));
    }
    if row["price_model"] != arm.row.model {
        return Err(format!(
            "{probe}: BIL-13 price_model {} does not resolve to the row {} ({row})",
            row["price_model"], arm.row.model
        ));
    }
    Ok((out, row.clone()))
}

/// A model id reduced to its family tokens: no provider prefix, no Bedrock profile wrapping, no
/// dated snapshot suffix; split on `-._:` and sorted, so `anthropic/claude-4.5-haiku-20251001`
/// and `claude-haiku-4-5` agree. Returns `(tokens, snapshot)`.
fn family(id: &str) -> (Vec<String>, Option<String>) {
    let mut s = id.to_ascii_lowercase();
    if let Some((_, rest)) = s.rsplit_once('/') {
        s = rest.to_owned();
    }
    if let Some((_, rest)) = s.split_once("anthropic.") {
        s = rest.to_owned();
    }
    if let Some(i) = s.rfind("-v")
        && s[i + 2..].contains(':')
    {
        s.truncate(i);
    }
    let mut snapshot = None;
    let bytes = s.as_bytes();
    let digits = |from: usize, n: usize| {
        s.len() >= from + n && bytes[from..from + n].iter().all(u8::is_ascii_digit)
    };
    let n = s.len();
    // -YYYY-MM-DD, -YYYYMMDD, -MMDD (gpt-4-0613, DeepSeek-V4-Pro-0813).
    if n > 11
        && bytes[n - 11] == b'-'
        && digits(n - 10, 4)
        && bytes[n - 6] == b'-'
        && digits(n - 5, 2)
        && bytes[n - 3] == b'-'
        && digits(n - 2, 2)
    {
        snapshot = Some(s[n - 10..].replace('-', ""));
        s.truncate(n - 11);
    } else if n > 9 && bytes[n - 9] == b'-' && digits(n - 8, 8) {
        snapshot = Some(s[n - 8..].to_owned());
        s.truncate(n - 9);
    } else if n > 5 && bytes[n - 5] == b'-' && digits(n - 4, 4) {
        snapshot = Some(s[n - 4..].to_owned());
        s.truncate(n - 5);
    }
    let mut toks: Vec<String> = s
        .split(['-', '.', '_', ':'])
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect();
    toks.sort();
    (toks, snapshot)
}

/// Whether an echoed id is the candidate's model: the same family, or an id the provider's own
/// listing gives the candidate's id as an alias of (xAI: `grok-4.20` → `grok-4.20-0309-reasoning`).
fn is_candidate(arm: &Arm, echo: &str) -> bool {
    let id = arm.cand.upstream_model;
    family(echo).0 == family(id).0
        || listed(arm.provider(), echo).is_some_and(|m| {
            m["aliases"]
                .as_array()
                .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(id)))
        })
}

/// The echo, spelled as the candidate when the provider resolved an alias of it.
fn canonical_echo(arm: &Arm, echo: &str) -> String {
    if family(echo).0 != family(arm.cand.upstream_model).0 && is_candidate(arm, echo) {
        arm.cand.upstream_model.to_owned()
    } else {
        echo.to_owned()
    }
}

/// Hosting labels that name a deployment, not a different model.
const HOSTING_NOISE: &[&str] = &["instruct", "turbo", "versatile", "instant", "it", "latest"];

fn core_family(id: &str) -> Vec<String> {
    let (mut t, _) = family(id);
    t.retain(|x| !HOSTING_NOISE.contains(&x.as_str()));
    t
}

// ---------------------------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------------------------

const OK_PROMPT: &str = "Reply with the single word OK.";
/// Enough arithmetic that a reasoning model reasons at low effort, not so much that o1 spends 4k.
const REASON_PROMPT: &str = "What is 1234 times 5678? Reply with only the number.";
/// Claude's adaptive thinking skips arithmetic it can do in its head, even at medium effort.
const CLAUDE_REASON_PROMPT: &str =
    "How many prime numbers are there between 1000 and 1100? Reply with only the number.";

/// CAT-1: the candidate serves and echoes its own id.
fn cat1(trial: &str, arm: Arm) -> Result<(), Failed> {
    let max = (arm.wire != Wire::Embeddings).then_some(arm.basic_max());
    let r = call(
        &arm,
        &body(&arm, OK_PROMPT, max, Opts::default()),
        Walk::Only,
    )?;
    let (out, _) = served(trial, &arm, "basic", &r)?;
    let mut problems = Vec::new();
    if r.upstream.as_deref() != Some(arm.cand.upstream_model) {
        problems.push(format!(
            "x-beyond-upstream-model {:?}, candidate id {}",
            r.upstream, arm.cand.upstream_model
        ));
    }
    if out.model.is_empty() {
        problems.push(format!("no model echoed: {}", r.excerpt()));
    } else if !is_candidate(&arm, &out.model) {
        problems.push(format!(
            "echoed model {} is not the candidate's {}",
            out.model, arm.cand.upstream_model
        ));
    }
    println!("echo {} → {}", arm.cand.upstream_model, out.model);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// BIL-13: whatever the candidate answers, its one ledger row prices at the row; on a 200 the
/// billed model is the echoed snapshot. A candidate that doesn't serve still writes a row, so
/// this holds on every candidate without depending on CAT-1.
fn bil13(trial: &str, arm: Arm) -> Result<(), Failed> {
    let max = (arm.wire != Wire::Embeddings).then_some(arm.basic_max());
    let r = call(
        &arm,
        &body(&arm, OK_PROMPT, max, Opts::default()),
        Walk::Only,
    )?;
    let id = r.request_id.as_deref().ok_or("no x-beyond-request-id")?;
    let rows = ledger(id, 1);
    let [row] = rows.as_slice() else {
        return Err(format!(
            "{} ai.usage rows for {id}, want 1 (HTTP {})",
            rows.len(),
            r.status
        )
        .into());
    };
    let out = parse(arm.wire, &r.json);
    if r.status == 200 {
        record(trial, &arm, "basic", row, &out);
    }
    let mut problems = Vec::new();
    if row["price_model"] != arm.row.model {
        problems.push(format!(
            "price_model {} does not resolve to the row {} (HTTP {}; {row})",
            row["price_model"], arm.row.model, r.status
        ));
    }
    if r.status == 200 && row["model"] != out.model.as_str() {
        problems.push(format!(
            "ledger bills model {}, the provider echoed {}",
            row["model"], out.model
        ));
    }
    if r.status != 200 {
        note(
            trial,
            &format!(
                "candidate answered HTTP {}; its row still prices at the row",
                r.status
            ),
        );
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// CAT-2: every in-scope candidate of the row serves the same model (family and, where two
/// echoes carry one, snapshot).
fn cat2(trial: &str, row: &'static ModelRoute) -> Result<(), Failed> {
    let mut echoes: Vec<(String, String)> = Vec::new();
    let mut problems = Vec::new();
    for arm in arms(row) {
        let max = (arm.wire != Wire::Embeddings).then_some(arm.basic_max());
        let r = call(
            &arm,
            &body(&arm, OK_PROMPT, max, Opts::default()),
            Walk::Only,
        )?;
        match served(trial, &arm, "basic", &r) {
            Ok((out, _)) => echoes.push((arm.route().to_owned(), canonical_echo(&arm, &out.model))),
            Err(e) => problems.push(format!("{}: {e}", arm.route())),
        }
    }
    println!("echoes: {echoes:?}");
    if let Some((first_route, first)) = echoes.first() {
        let (fam, snap) = (core_family(first), family(first).1);
        for (route, echo) in &echoes[1..] {
            if core_family(echo) != fam {
                problems.push(format!(
                    "{route} serves {echo}, {first_route} serves {first}: different models"
                ));
            } else if let (Some(a), Some(b)) = (&snap, family(echo).1)
                && *a != b
            {
                problems.push(format!(
                    "{route} serves snapshot {echo}, {first_route} serves {first}"
                ));
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// A prompt of at least `words` tokens: " the" is one token in every BPE vocabulary here, and
/// pre-tokenization never merges across the space, so the count is a floor.
fn filler(words: u64) -> String {
    let mut s = String::with_capacity(words as usize * 6 + 64);
    s.push_str("Reply OK.");
    for _ in 0..words {
        s.push_str(" the");
    }
    s
}

/// How CAT-3 runs on a candidate, decided when the trial is listed.
#[derive(Clone, Copy)]
struct Context {
    over: Option<Walk>,
    near: bool,
}

const CONTEXT_WORDS: &[&str] = &[
    "context",
    "token",
    "too long",
    "too large",
    "maximum",
    "exceed",
    "length",
    "limit",
    "prompt is",
];

/// The input limit a context-length rejection names, when it names one.
fn stated_limit(msg: &str) -> Option<u64> {
    let lower = msg.to_ascii_lowercase();
    const PATTERNS: &[&str] = &[
        "configured limit of ",
        "maximum context length is ",
        "context length is ",
        "context window of ",
        "maximum prompt length is ",
        "limit of ",
        "maximum of ",
        "at most ",
    ];
    PATTERNS.iter().find_map(|pat| {
        let at = lower.find(pat)? + pat.len();
        let digits: String = lower[at..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ',')
            .filter(char::is_ascii_digit)
            .collect();
        digits.parse::<u64>().ok().filter(|n| *n >= 1000)
    })
}

/// CAT-3: over the window is a clean 4xx from this candidate; 0.9x is accepted (where planned).
fn cat3(trial: &str, arm: Arm, plan: Context) -> Result<(), Failed> {
    let window = u64::from(arm.row.card.context_window);
    let mut problems = Vec::new();
    if let Some(walk) = plan.over {
        let words = window + window / 20 + 2000;
        let max = (arm.wire != Wire::Embeddings).then_some(16);
        let r = call(
            &arm,
            &body(&arm, &filler(words), max, Opts::default()),
            walk,
        )?;
        let lower = r.raw.to_ascii_lowercase();
        println!(
            "over-limit ({words} words): HTTP {} {}",
            r.status,
            r.excerpt()
        );
        if r.status == 200 {
            if let Ok((out, _)) = served(trial, &arm, "over-limit", &r) {
                problems.push(format!(
                    "{words}-word prompt over a {window} window was accepted ({} input tokens billed)",
                    out.input
                ));
            } else {
                problems.push(format!("over-limit prompt was accepted: {}", r.excerpt()));
            }
        } else if !(400..500).contains(&r.status) {
            problems.push(format!(
                "over-limit: HTTP {} (want a clean 4xx): {}",
                r.status,
                r.excerpt()
            ));
        } else if !CONTEXT_WORDS.iter().any(|w| lower.contains(w)) {
            problems.push(format!(
                "over-limit: HTTP {} but not a context error: {}",
                r.status,
                r.excerpt()
            ));
        } else if let Some(limit) = stated_limit(&r.raw) {
            // The rejection names the provider's real limit: a card window above it is one a
            // client can't use.
            if (limit as f64) < window as f64 * 0.98 {
                problems.push(format!(
                    "{} states an input limit of {limit} tokens, under the card's \
                     context_window {window}: {}",
                    arm.provider(),
                    r.excerpt()
                ));
            } else if (limit as f64) > window as f64 * 1.02 {
                note(
                    trial,
                    &format!(
                        "{} states a limit of {limit}, over the card's {window}",
                        arm.provider()
                    ),
                );
            }
        }
        if r.status != 200 && r.provider.as_deref() != Some(arm.provider()) {
            problems.push(format!(
                "over-limit answered by {:?}, not the candidate {} (failover on a client error?)",
                r.provider,
                arm.provider()
            ));
        }
    }
    if plan.near {
        let words = window * 9 / 10 - 500;
        let r = call(
            &arm,
            &body(&arm, &filler(words), Some(16), Opts::default()),
            Walk::Only,
        )?;
        match served(trial, &arm, "near-limit", &r) {
            Ok((out, _)) => println!(
                "near-limit: {words} words → {} input tokens accepted",
                out.input
            ),
            Err(e) => problems.push(format!("{words}-word prompt (0.9x window) rejected: {e}")),
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// CAT-4: `max_tokens` at the card's `max_output_tokens` is accepted.
fn cat4(trial: &str, arm: Arm) -> Result<(), Failed> {
    let max = arm.row.card.max_output_tokens;
    let r = call(
        &arm,
        &body(&arm, OK_PROMPT, Some(max), Opts::default()),
        Walk::Only,
    )?;
    served(trial, &arm, "max-output", &r)
        .map(|_| ())
        .map_err(|e| format!("max_tokens={max} (the card's max_output_tokens): {e}").into())
}

fn mentions(text: &str, word: &str) -> bool {
    text.to_ascii_uppercase().contains(word)
}

/// CAT-5: advertised image and PDF input is read; an image on a row that doesn't advertise it
/// fails cleanly.
fn cat5(trial: &str, arm: Arm) -> Result<(), Failed> {
    let mut problems = Vec::new();
    let prompt = "What word is written in the attached image? Reply with just that word.";
    let r = call(
        &arm,
        &body(
            &arm,
            prompt,
            Some(arm.short_max()),
            Opts {
                image: true,
                ..Opts::default()
            },
        ),
        Walk::Only,
    )?;
    if arm.has(IN_IMAGE) {
        match served(trial, &arm, "image", &r) {
            // "ESTREL": a model that misreads the K still read the image.
            Ok((out, _)) if mentions(&out.text, &IMAGE_WORD[1..]) => {}
            Ok((out, _)) => problems.push(format!(
                "image: answer {:?} does not name {IMAGE_WORD}",
                out.text
            )),
            Err(e) => problems.push(e),
        }
    } else {
        match r.status {
            400..=499 => println!(
                "image on a text-only row: HTTP {} {}",
                r.status,
                r.excerpt()
            ),
            200 => {
                let out = served(trial, &arm, "image-unadvertised", &r)
                    .map(|(o, _)| o)
                    .unwrap_or_default();
                problems.push(if mentions(&out.text, IMAGE_WORD) {
                    format!(
                        "card omits image input, but the candidate read the image ({:?})",
                        out.text
                    )
                } else {
                    format!(
                        "card omits image input; the candidate accepted one with a 200 instead of \
                         a clean 4xx ({:?})",
                        out.text
                    )
                });
            }
            s => problems.push(format!(
                "image on a text-only row: HTTP {s} (want 4xx): {}",
                r.excerpt()
            )),
        }
    }
    // A candidate that reads no PDF (OpenRouter's grok-build-0.1) must be skipped: ordered first,
    // the walk serves the document elsewhere.
    if arm.has(IN_FILE) && !serves_file_input(&arm.cand) {
        let prompt = "What is the code word in the attached document? Reply with just that word.";
        let r = call(
            &arm,
            &body(
                &arm,
                prompt,
                Some(arm.short_max()),
                Opts {
                    file: true,
                    ..Opts::default()
                },
            ),
            Walk::Order,
        )?;
        let out = parse(arm.wire, &r.json);
        if r.status != 200 {
            problems.push(format!(
                "pdf with {} first: HTTP {}: {}",
                arm.provider(),
                r.status,
                r.excerpt()
            ));
        } else if r.provider.as_deref() == Some(arm.provider()) {
            problems.push(format!(
                "pdf was sent to {}, which reads none",
                arm.provider()
            ));
        } else if !mentions(&out.text, FILE_WORD) {
            problems.push(format!(
                "pdf (served by {:?}): answer {:?} does not name {FILE_WORD}",
                r.provider, out.text
            ));
        } else {
            note(
                trial,
                &format!(
                    "pdf skipped {} and was served by {:?}",
                    arm.provider(),
                    r.provider
                ),
            );
        }
    } else if arm.has(IN_FILE) {
        let prompt = "What is the code word in the attached document? Reply with just that word.";
        let r = call(
            &arm,
            &body(
                &arm,
                prompt,
                Some(arm.short_max()),
                Opts {
                    file: true,
                    ..Opts::default()
                },
            ),
            Walk::Only,
        )?;
        match served(trial, &arm, "file", &r) {
            Ok((out, _)) if mentions(&out.text, FILE_WORD) => {}
            Ok((out, _)) => problems.push(format!(
                "pdf: answer {:?} does not name {FILE_WORD}",
                out.text
            )),
            Err(e) => problems.push(e),
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// CAT-6: tools and structured outputs as advertised; tools on a row without them fail cleanly.
fn cat6(trial: &str, arm: Arm) -> Result<(), Failed> {
    let mut problems = Vec::new();
    let max = Some(arm.short_max());
    // Tools. Forced first; a model that refuses forced tool_choice (several do while thinking:
    // Claude Opus/Sonnet 5.5, Qwen on Alibaba) gets `auto` and a prompt that needs the tool.
    let tool_prompt = "What is the weather in Paris right now? Use the tool.";
    let mut tool_body = body(
        &arm,
        tool_prompt,
        max,
        Opts {
            tools: true,
            ..Opts::default()
        },
    );
    let mut r = call(&arm, &tool_body, Walk::Only)?;
    if arm.can(TOOLS)
        && (400..500).contains(&r.status)
        && r.raw.to_ascii_lowercase().contains("tool_choice")
    {
        note(
            trial,
            &format!(
                "forced tool_choice rejected, retrying with auto: {}",
                r.excerpt()
            ),
        );
        tool_body["tool_choice"] = json!("auto");
        r = call(&arm, &tool_body, Walk::Only)?;
    }
    if arm.can(TOOLS) {
        match served(trial, &arm, "tools", &r) {
            Ok((out, _)) => match out.tool_calls.first() {
                Some((name, args)) if name == "get_weather" => {
                    let parsed: Value = serde_json::from_str(args).unwrap_or(Value::Null);
                    if !parsed["city"].is_string() {
                        problems.push(format!("tools: get_weather arguments {args:?} lack city"));
                    }
                }
                other => problems.push(format!(
                    "tools: forced get_weather, got {other:?} / text {:?}",
                    out.text
                )),
            },
            Err(e) => problems.push(format!("tools: {e}")),
        }
    } else {
        match r.status {
            400..=499 => println!(
                "tools on a tool-less row: HTTP {} {}",
                r.status,
                r.excerpt()
            ),
            200 => {
                let out = served(trial, &arm, "tools-unadvertised", &r)
                    .map(|(o, _)| o)
                    .unwrap_or_default();
                problems.push(if out.tool_calls.is_empty() {
                    "card omits tools; the candidate accepted a tool request with a 200 (tools \
                     silently ignored) instead of a clean 4xx"
                        .to_owned()
                } else {
                    format!(
                        "card omits tools, but the candidate called one ({:?})",
                        out.tool_calls
                    )
                });
            }
            s => problems.push(format!(
                "tools on a tool-less row: HTTP {s} (want 4xx): {}",
                r.excerpt()
            )),
        }
    }
    // Structured outputs. A candidate that cannot honor them (Bedrock) must be skipped: ordered
    // first, the walk serves the request elsewhere.
    if arm.can(STRUCTURED_OUTPUTS) && !serves_structured_outputs(&arm.cand) {
        let r = call(
            &arm,
            &body(
                &arm,
                "What is 17 times 23? Answer in the required JSON format.",
                max,
                Opts {
                    schema: true,
                    ..Opts::default()
                },
            ),
            Walk::Order,
        )?;
        let out = parse(arm.wire, &r.json);
        let v: Value = serde_json::from_str(out.text.trim()).unwrap_or(Value::Null);
        if r.status != 200 {
            problems.push(format!(
                "json_schema with {} first: HTTP {}: {}",
                arm.provider(),
                r.status,
                r.excerpt()
            ));
        } else if r.provider.as_deref() == Some(arm.provider()) {
            problems.push(format!(
                "json_schema was sent to {}, which cannot honor it",
                arm.provider()
            ));
        } else if !v["answer"].is_i64() {
            problems.push(format!(
                "json_schema (served by {:?}): {:?} does not validate",
                r.provider, out.text
            ));
        } else {
            note(
                trial,
                &format!(
                    "json_schema skipped {} and was served by {:?}",
                    arm.provider(),
                    r.provider
                ),
            );
        }
    } else if arm.can(STRUCTURED_OUTPUTS) {
        let r = call(
            &arm,
            &body(
                &arm,
                "What is 17 times 23? Answer in the required JSON format.",
                max,
                Opts {
                    schema: true,
                    ..Opts::default()
                },
            ),
            Walk::Only,
        )?;
        match served(trial, &arm, "json_schema", &r) {
            Ok((out, _)) => {
                let v: Value = serde_json::from_str(out.text.trim()).unwrap_or(Value::Null);
                let keys_ok = v.as_object().is_some_and(|o| o.len() == 1);
                if !v["answer"].is_i64() || !keys_ok {
                    problems.push(format!(
                        "json_schema: {:?} does not validate as {{\"answer\": <integer>}}",
                        out.text
                    ));
                }
            }
            Err(e) => problems.push(format!("json_schema: {e}")),
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// CAT-6 + BIL-9 on a reasoning row: reasoning is reported, and the ledger counts it exactly once.
fn cat6_reasoning(trial: &str, arm: Arm) -> Result<(), Failed> {
    let mut problems = Vec::new();
    let max = Some(arm.short_max());
    {
        let opts = Opts {
            reason: true,
            ..Opts::default()
        };
        let prompt = if arm.claude() {
            CLAUDE_REASON_PROMPT
        } else {
            REASON_PROMPT
        };
        let mut r = call(&arm, &body(&arm, prompt, max, opts), Walk::Only)?;
        // A model at low effort may answer the cheap prompt without reasoning (gpt-5.6); the
        // harder one settles it, and is spent only then.
        let quiet = |r: &Reply| {
            let o = parse(arm.wire, &r.json);
            r.status == 200 && o.reasoning.unwrap_or(0) == 0 && !o.reasoning_text
        };
        if !arm.claude() && quiet(&r) {
            let _ = served(trial, &arm, "reasoning-cheap", &r);
            r = call(
                &arm,
                &body(&arm, CLAUDE_REASON_PROMPT, max, opts),
                Walk::Only,
            )?;
        }
        match served(trial, &arm, "reasoning", &r) {
            Ok((out, row)) => {
                let reasoning = out.reasoning.unwrap_or(0);
                if reasoning == 0 && !out.reasoning_text && arm.adaptive_claude() {
                    // Adaptive thinking (Claude 4.6 and later) is the model's call: Opus 4.7
                    // answers these without thinking even at effort high, natively too. The
                    // request with thinking on was accepted and served, which is the capability.
                    note(
                        trial,
                        "adaptive thinking was accepted; the model chose not to think",
                    );
                } else if reasoning == 0 && !out.reasoning_text {
                    problems.push(format!(
                        "reasoning: no reasoning tokens or text reported (usage {})",
                        r.json["usage"]
                    ));
                }
                let billed = row["output_tokens"].as_u64().unwrap_or(0);
                // OpenAI's convention: completion_tokens includes reasoning. xAI's Chat Completions
                // reports it beside completion_tokens (D23), so its total is the sum; its Responses
                // (the multi-agent candidate) counts it inside output_tokens, as OpenAI does.
                let total = if arm.provider() == "xai" && !arm.cand.path.ends_with("/responses") {
                    out.output + reasoning
                } else {
                    out.output
                };
                if billed != total {
                    problems.push(format!(
                        "BIL-9: ledger bills {billed} output tokens, the provider reported {} \
                         (+{reasoning} reasoning) = {total}",
                        out.output
                    ));
                }
            }
            Err(e) => problems.push(format!("reasoning: {e}")),
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

fn truth_file() -> &'static toml::Value {
    static T: OnceLock<toml::Value> = OnceLock::new();
    T.get_or_init(|| {
        std::fs::read_to_string(repo_root().join("verify/catalog_truth.toml"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(toml::Value::Table(Default::default()))
    })
}

/// The `verify/catalog_truth.toml` entry for a row.
fn truth(row: &str) -> Option<&'static toml::Value> {
    truth_file()
        .get("row")?
        .as_array()?
        .iter()
        .find(|r| r.get("model").and_then(|m| m.as_str()) == Some(row))
}

/// A row's vendor-announced promotional (input, output) rate, USD per million, while it lasts.
fn promo(row: &str) -> Option<(f64, f64)> {
    let p = truth_file()
        .get("promo")?
        .as_array()?
        .iter()
        .find(|r| r.get("model").and_then(|m| m.as_str()) == Some(row))?;
    let until = rfc3339(&format!("{}T23:59:59Z", p.get("until")?.as_str()?))?;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_secs();
    if i64::try_from(now).ok()? > until {
        return None;
    }
    let rate = |k: &str| p.get(k)?.as_str()?.parse::<f64>().ok();
    Some((rate("input")?, rate("output")?))
}

fn price_eq(card: &str, listed: f64) -> bool {
    let c: f64 = card.parse().unwrap_or(f64::NAN);
    (c - listed).abs() <= c.abs() * 1e-6 + 1e-9
}

/// CAT-7: list price against the provider's own number where it gives one, else the vendor doc.
fn cat7(trial: &str, arm: Arm) -> Result<(), Failed> {
    let mut problems = Vec::new();
    let p = arm.row.price;
    // Billed cost the provider reports.
    if matches!(arm.provider(), "openrouter" | "xai") {
        let prompt = format!(
            "{} Reply with the single word OK.",
            "The quick brown fox jumps over the lazy dog.".repeat(25)
        );
        let max = (arm.wire != Wire::Embeddings).then(|| arm.short_max());
        // xAI writes `cost_in_usd_ticks` into its own usage. A Chat client translated from xAI's
        // Responses gets OpenAI's Chat usage, which has no cost field, so on an xAI Responses
        // candidate the probe is a Responses client: a one-shot relayed with xAI's usage intact.
        let probe = if arm.provider() == "xai" && arm.cand.path.ends_with("/responses") {
            Arm {
                wire: Wire::Responses,
                ..arm
            }
        } else {
            arm
        };
        let r = call(
            &probe,
            &body(
                &probe,
                &prompt,
                max,
                Opts {
                    cost: true,
                    ..Opts::default()
                },
            ),
            Walk::Only,
        )?;
        match served(trial, &probe, "cost", &r) {
            Ok((out, row)) => {
                let ours = priced(arm.row, &row);
                match out.provider_cost {
                    Some(theirs) if theirs > 0.0 => {
                        let off = (ours - theirs) / theirs;
                        println!(
                            "cost: ledger x card ${ours:.8}, provider ${theirs:.8} ({:+.2}%)",
                            off * 100.0
                        );
                        // OpenRouter's charge is the declared truth only when it routed to the
                        // model's own vendor (OpenAI/Azure, Anthropic/Bedrock/Vertex, xAI): a
                        // third-party host of an open model charges its own rate, which the card
                        // (the primary vendor's list price) does not claim to be.
                        let host = r.json["provider"].as_str().unwrap_or("");
                        let first_party = arm.provider() == "xai"
                            || matches!(
                                (arm.row.card.owned_by, host),
                                ("openai", "OpenAI" | "Azure")
                                    | ("anthropic", "Anthropic" | "Amazon Bedrock" | "Google")
                                    | ("x-ai" | "xai", "xAI")
                            );
                        // A vendor-announced promotion (`[[promo]]` in catalog_truth.toml) that this
                        // route charges while the card keeps the standard rate.
                        let promo = promo(arm.row.model).filter(|(input, output)| {
                            let p = |k: &str| row[k].as_u64().unwrap_or(0) as f64;
                            let at =
                                (p("input_tokens") * input + p("output_tokens") * output) / 1e6;
                            ((at - theirs) / theirs).abs() <= COST_TOLERANCE
                        });
                        if off.abs() > COST_TOLERANCE && promo.is_some() {
                            note(
                                trial,
                                &format!(
                                    "{} charged the documented promotional rate ${theirs:.8} vs \
                                     card ${ours:.8}; the card keeps the standard rate",
                                    arm.provider()
                                ),
                            );
                        } else if off.abs() > COST_TOLERANCE && !first_party {
                            note(
                                trial,
                                &format!(
                                    "third-party host {host} charged ${theirs:.8} vs card ${ours:.8} \
                                     ({:+.1}%); not the declared source of truth",
                                    off * 100.0
                                ),
                            );
                        } else if off.abs() > COST_TOLERANCE {
                            problems.push(format!(
                                "ledger tokens x card = ${ours:.8}, {} charged ${theirs:.8} ({:+.1}%; \
                                 in {} out {} read {} write {}; card {}/{}/{}/{}; upstream {:?})",
                                arm.provider(), off * 100.0, row["input_tokens"], row["output_tokens"],
                                row["cache_read_tokens"], row["cache_write_tokens"], p.input, p.output,
                                p.cache_read, p.cache_write, r.json["provider"]
                            ));
                        }
                    }
                    other => problems.push(format!(
                        "{} reported no cost ({other:?}): usage {}",
                        arm.provider(),
                        r.json["usage"]
                    )),
                }
            }
            Err(e) => problems.push(e),
        }
    }
    // Listed price.
    let primary = by_id(arm.row.candidates[0].provider).name;
    match arm.provider() {
        // An OpenRouter-primary row whose maker publishes its own rate follows the maker (the
        // vendor-truth check below), not OpenRouter's cheapest host.
        "openrouter" if primary == "openrouter" && truth(arm.row.model).is_none() => {
            if let Some(m) = listed("openrouter", arm.cand.upstream_model) {
                let per_m = |k: &str| {
                    m["pricing"][k]
                        .as_str()
                        .and_then(|s| s.parse::<f64>().ok())
                        .map(|x| x * 1e6)
                };
                for (field, card, k) in [
                    ("input", p.input, "prompt"),
                    ("output", p.output, "completion"),
                ] {
                    match per_m(k) {
                        Some(v) if price_eq(card, v) => {}
                        Some(v) => {
                            problems.push(format!("{field}: card {card}, OpenRouter lists {v}"))
                        }
                        None => {}
                    }
                }
                note(
                    trial,
                    "price checked against OpenRouter's /api/v1/models listing",
                );
            } else {
                problems.push(format!(
                    "OpenRouter's listing has no {}",
                    arm.cand.upstream_model
                ));
            }
        }
        "together" if primary == "together" => match listed("together", arm.cand.upstream_model) {
            Some(m) => {
                for (field, card, k) in
                    [("input", p.input, "input"), ("output", p.output, "output")]
                {
                    if let Some(v) = m["pricing"][k].as_f64()
                        && !price_eq(card, v)
                    {
                        problems.push(format!("{field}: card {card}, Together lists {v}"));
                    }
                }
                note(trial, "price checked against Together's /v1/models listing");
            }
            None => problems.push(format!(
                "Together's listing has no {}",
                arm.cand.upstream_model
            )),
        },
        "xai" => match listed("xai", arm.cand.upstream_model) {
            // xAI lists prices in 1e-4 USD per million tokens.
            Some(m) => {
                for (field, card, k) in [
                    ("input", p.input, "prompt_text_token_price"),
                    ("output", p.output, "completion_text_token_price"),
                    ("cache_read", p.cache_read, "cached_prompt_text_token_price"),
                ] {
                    if let Some(v) = m[k].as_f64().map(|x| x / 1e4)
                        && !price_eq(card, v)
                    {
                        problems.push(format!("{field}: card {card}, xAI lists {v}"));
                    }
                }
            }
            None => problems.push(format!("xAI's listing has no {}", arm.cand.upstream_model)),
        },
        _ => {}
    }
    // Vendor doc, for every row with a primary-vendor truth entry.
    if let Some(t) = truth(arm.row.model) {
        for (field, card) in [
            ("input", p.input),
            ("output", p.output),
            ("cache_read", p.cache_read),
            ("cache_write", p.cache_write),
        ] {
            if let Some(v) = t.get(field).and_then(|v| v.as_str())
                && !price_eq(card, v.parse().unwrap_or(f64::NAN))
            {
                problems.push(format!("{field}: card {card}, vendor doc {v}"));
            }
        }
        if !matches!(arm.provider(), "openrouter" | "xai") {
            note(
                trial,
                &format!(
                    "vendor-doc verified: card equals verify/catalog_truth.toml ({} reports no cost)",
                    arm.provider()
                ),
            );
        }
    } else if !matches!(arm.provider(), "openrouter" | "xai" | "together") {
        problems.push(format!(
            "{} reports no cost and the row has no catalog_truth entry",
            arm.provider()
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// A display name's words, without OpenRouter's `Vendor: ` prefix or a host's deployment labels
/// (Together's `FP8`, `Turbo`, `-it`), which name how it is served, not the model.
fn norm_name(s: &str) -> Vec<String> {
    const LABELS: &[&str] = &["fp8", "fp4", "turbo", "it", "meta"];
    let s = s
        .rsplit_once(": ")
        .map_or(s, |(_, n)| n)
        .to_ascii_lowercase();
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty() && !LABELS.contains(w))
        .map(str::to_owned)
        .collect()
}

/// RFC 3339 `YYYY-MM-DDTHH:MM:SSZ` → Unix seconds.
fn rfc3339(s: &str) -> Option<i64> {
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (hh, mm, ss) = (
        n(11..13).unwrap_or(0),
        n(14..16).unwrap_or(0),
        n(17..19).unwrap_or(0),
    );
    // Days from civil (Howard Hinnant).
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// Where CAT-8 reads a row's metadata: the primary's endpoint, or, for an OpenRouter-primary row
/// whose maker (OpenAI, Anthropic, xAI) lists the id, the maker's; else OpenRouter's.
fn meta_source(row: &'static ModelRoute) -> Option<(&'static str, &'static str)> {
    let p = row.candidates[0];
    let name = by_id(p.provider).name;
    if name == "openrouter" {
        let maker = match row.card.owned_by {
            "openai" => Some("openai"),
            "anthropic" => Some("anthropic"),
            "xai" => Some("xai"),
            _ => None,
        };
        if let Some(m) = maker
            && key_of(m).is_some()
            && listed(m, row.model).is_some()
        {
            return Some((m, row.model));
        }
    }
    match name {
        "anthropic" | "openai" | "xai" | "together" | "openrouter" => {
            Some((name, p.upstream_model))
        }
        _ => row
            .candidates
            .iter()
            .find(|c| c.provider == ProviderId::OpenRouter)
            .map(|c| ("openrouter", c.upstream_model)),
    }
}

/// CAT-8: name, created and owner against the provider's model endpoint.
fn cat8(_trial: &str, row: &'static ModelRoute) -> Result<(), Failed> {
    let (provider, id) = meta_source(row).ok_or("no in-scope metadata source")?;
    let m = listed(provider, id).ok_or(format!("{provider}'s model listing has no {id}"))?;
    let card = row.card;
    let mut problems = Vec::new();
    let (name, created, owner): (Option<&str>, Option<i64>, Option<String>) = match provider {
        "anthropic" => (
            m["display_name"].as_str(),
            m["created_at"].as_str().and_then(rfc3339),
            Some("anthropic".into()),
        ),
        "openai" => (
            None,
            m["created"].as_i64(),
            // OpenAI names the account class (`system`, `openai`), not the vendor; the vendor is
            // always OpenAI.
            Some("openai".into()),
        ),
        "xai" => (
            None,
            m["created"].as_i64(),
            m["owned_by"].as_str().map(str::to_owned),
        ),
        "together" => (
            m["display_name"].as_str(),
            None, // Together's `created` is when it was deployed there, not the release.
            None,
        ),
        _ => (
            m["name"].as_str(),
            m["created"].as_i64(),
            id.split_once('/').map(|(o, _)| o.to_owned()),
        ),
    };
    println!(
        "{provider} {id}: name {name:?} created {created:?} owner {owner:?}; card {:?} {} {:?}",
        card.name, card.created, card.owned_by
    );
    if let Some(n) = name
        && norm_name(n) != norm_name(card.name)
    {
        problems.push(format!("name: card {:?}, {provider} {n:?}", card.name));
    }
    if let Some(c) = created
        && (c - card.created as i64).abs() > CREATED_TOLERANCE
    {
        problems.push(format!(
            "created: card {} , {provider} {c} ({:+.1} days)",
            card.created,
            (card.created as i64 - c) as f64 / 86_400.0
        ));
    }
    if let Some(o) = owner
        && !o.eq_ignore_ascii_case(card.owned_by)
    {
        problems.push(format!(
            "owned_by: card {:?}, {provider} {o:?}",
            card.owned_by
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// One CAT-13 passthrough: a provider mount, a path, a body, and how the client authenticates.
#[derive(Clone, Copy)]
struct Mount {
    name: &'static str,
    path: &'static str,
    model: &'static str,
    anthropic_wire: bool,
    /// Which key a BYO caller holds.
    key: &'static str,
}

const MOUNTS: &[Mount] = &[
    Mount {
        name: "openai",
        path: "/openai/v1/chat/completions",
        model: "gpt-4.1-nano",
        anthropic_wire: false,
        key: "openai",
    },
    Mount {
        name: "anthropic",
        path: "/anthropic/v1/messages",
        model: "claude-haiku-4-5",
        anthropic_wire: true,
        key: "anthropic",
    },
    Mount {
        name: "openrouter",
        path: "/openrouter/api/v1/chat/completions",
        model: "openai/gpt-4.1-nano",
        anthropic_wire: false,
        key: "openrouter",
    },
    Mount {
        name: "xai",
        path: "/xai/v1/chat/completions",
        model: "grok-4.20",
        anthropic_wire: false,
        key: "xai",
    },
    Mount {
        name: "bedrock",
        path: "/bedrock/anthropic/v1/messages",
        model: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        anthropic_wire: true,
        key: "bedrock",
    },
    Mount {
        name: "together",
        path: "/together/v1/chat/completions",
        model: "openai/gpt-oss-120b",
        anthropic_wire: false,
        key: "together",
    },
    // Config-added providers (see `boot`).
    Mount {
        name: "bedrock-openai",
        path: "/bedrock-openai/openai/v1/chat/completions",
        model: "openai.gpt-oss-20b-1:0",
        anthropic_wire: false,
        key: "bedrock",
    },
    Mount {
        name: "anthropic-alt",
        path: "/anthropic-alt/v1/messages",
        model: "claude-haiku-4-5",
        anthropic_wire: true,
        key: "anthropic",
    },
];

/// CAT-13: `/{provider}/…` passes through, managed (pool key, billed) and BYO (the caller's key,
/// not billed).
fn cat13(_trial: &str, m: Mount, byo: bool) -> Result<(), Failed> {
    let gw = gateway()?;
    let body = if m.anthropic_wire {
        json!({"model": m.model, "max_tokens": 16, "messages": [{"role": "user", "content": OK_PROMPT}]})
    } else {
        json!({"model": m.model, "max_tokens": 64, "messages": [{"role": "user", "content": OK_PROMPT}]})
    };
    let mut req = http()
        .post(format!("{}{}", gw.base, m.path))
        .header("anthropic-version", "2023-06-01")
        .json(&body);
    if byo {
        let k = key_of(m.key).ok_or("no key")?;
        req = if m.anthropic_wire {
            req.header("x-api-key", k)
        } else {
            req.bearer_auth(k)
        };
    } else {
        req = req.bearer_auth(DEV_TOKEN);
    }
    let r = send(req)?;
    if r.status != 200 {
        return Err(format!("HTTP {}: {}", r.status, r.excerpt()).into());
    }
    let mut problems = Vec::new();
    if r.provider.as_deref() != Some(m.name) {
        problems.push(format!(
            "x-beyond-provider {:?}, want {}",
            r.provider, m.name
        ));
    }
    let id = r.request_id.clone().unwrap_or_default();
    let rows = ledger(&id, usize::from(!byo));
    match (byo, rows.len()) {
        (false, 1) if rows[0]["provider"] == m.name => {
            println!("managed: billed {}", rows[0]);
        }
        (false, n) => problems.push(format!(
            "managed: {n} ledger rows ({rows:?}), want 1 naming {}",
            m.name
        )),
        (true, 0) => {}
        (true, n) => problems.push(format!(
            "BYO billed {n} ledger rows ({rows:?}); BYO is never billed"
        )),
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

// ---------------------------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------------------------

enum Kind {
    Cat1(Arm),
    Cat2(&'static ModelRoute),
    Cat3(Arm, Context),
    Cat4(Arm),
    Cat5(Arm),
    Cat6(Arm),
    Cat6Reasoning(Arm),
    Cat7(Arm),
    Bil13(Arm),
    Cat8(&'static ModelRoute),
    Cat13(Mount, bool),
}

struct Planned {
    name: String,
    kind: Kind,
    est: f64,
}

/// Every trial to list, and every one left out with its reason.
fn plan() -> (Vec<Planned>, Vec<String>) {
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    let cap = call_cap();
    let name = |claims: &str, route: &str, probe: &str| format!("{claims}::raw::{route}::{probe}");
    for row in MODEL_ROUTES {
        for c in row.candidates.iter().chain(row.responses) {
            let p = by_id(c.provider).name;
            if !SCOPE.iter().any(|(s, _)| *s == p) {
                skipped.push(format!("{} on {p}: provider out of scope", row.model));
            } else if key_of(p).is_none() {
                skipped.push(format!("{} on {p}: no key", row.model));
            }
        }
        let arms = arms(row);
        let card = row.card;
        let mut near_done = false;
        for arm in &arms {
            let (route, m) = (arm.route(), row.model);
            let embeddings = arm.wire == Wire::Embeddings;
            let out_tok = arm.est_out();
            let mut push = |claims: &str, est: f64, kind: Kind, calls: u32| {
                let n = name(claims, route, m);
                if est > cap {
                    skipped.push(format!(
                        "{n}: estimated ${est:.3} per call is over the ${cap} cap"
                    ));
                } else {
                    out.push(Planned {
                        name: n,
                        kind,
                        est: est * f64::from(calls),
                    });
                }
            };
            push("CAT-1", arm.est(30, 16), Kind::Cat1(*arm), 1);
            push("BIL-13", arm.est(30, 16), Kind::Bil13(*arm), 1);
            if !embeddings {
                push("CAT-4", arm.est(30, out_tok), Kind::Cat4(*arm), 1);
                let calls = 1 + u32::from(arm.has(IN_FILE));
                push(
                    "CAT-5",
                    arm.est(if arm.claude() { 1800 } else { 400 }, out_tok),
                    Kind::Cat5(*arm),
                    calls,
                );
                let calls = 1 + u32::from(arm.can(STRUCTURED_OUTPUTS));
                push("CAT-6", arm.est(120, out_tok), Kind::Cat6(*arm), calls);
                if arm.can(REASONING) {
                    push(
                        "CAT-6+BIL-9",
                        arm.est(40, out_tok),
                        Kind::Cat6Reasoning(*arm),
                        1,
                    );
                }
            }
            if arm.wire != Wire::Responses {
                let billed = matches!(arm.provider(), "openrouter" | "xai");
                push(
                    "CAT-7",
                    if billed { arm.est(260, out_tok) } else { 0.0 },
                    Kind::Cat7(*arm),
                    1,
                );
            }
            // CAT-3: not on the Responses arm (same model, same window as the chat arm).
            if arm.wire == Wire::Responses {
                continue;
            }
            let window = u64::from(card.context_window);
            let theirs = provider_window(arm.provider(), arm.cand.upstream_model);
            let over = match theirs {
                Some(w) if w > window + window / 20 => {
                    skipped.push(format!(
                        "{} over-limit: {} serves a {w}-token window, over the card's {window}; \
                         an over-card prompt would be accepted and billed",
                        name("CAT-3", route, m),
                        arm.provider()
                    ));
                    None
                }
                _ => {
                    // Order (this candidate first, the rest still failover) proves a client error
                    // does not walk on — unless another keyed candidate would accept the prompt.
                    let others_bigger = arms.iter().any(|o| {
                        o.provider() != arm.provider()
                            && provider_window(o.provider(), o.cand.upstream_model)
                                .is_some_and(|w| w > window + window / 20)
                    });
                    Some(if others_bigger {
                        Walk::Only
                    } else {
                        Walk::Order
                    })
                }
            };
            let mut near = false;
            if !near_done && !embeddings && card.context_window <= NEAR_MAX_WINDOW {
                let est = arm.est(window * 9 / 10, 16);
                if est > NEAR_CAP {
                    skipped.push(format!(
                        "{} near-limit: 0.9x window estimated ${est:.3}, over the ${NEAR_CAP} cap",
                        name("CAT-3", route, m)
                    ));
                } else {
                    near = true;
                }
                near_done = true;
            }
            if over.is_some() || near {
                let est = if near {
                    arm.est(window * 9 / 10, 16)
                } else {
                    0.0
                };
                out.push(Planned {
                    name: name("CAT-3", route, m),
                    kind: Kind::Cat3(*arm, Context { over, near }),
                    est,
                });
            }
        }
        if arms.len() >= 2 {
            let est = arms.iter().map(|a| a.est(30, 16)).sum();
            out.push(Planned {
                name: name("CAT-2", "all", row.model),
                kind: Kind::Cat2(row),
                est,
            });
        }
        match meta_source(row) {
            Some((p, _)) if p == "openrouter" || key_of(p).is_some() => out.push(Planned {
                name: name("CAT-8", p, row.model),
                kind: Kind::Cat8(row),
                est: 0.0,
            }),
            _ => skipped.push(format!("CAT-8 {}: no in-scope metadata source", row.model)),
        }
    }
    for m in MOUNTS {
        if key_of(m.key).is_none() {
            skipped.push(format!("CAT-13 {}: no key", m.name));
            continue;
        }
        for byo in [false, true] {
            out.push(Planned {
                name: name("CAT-13", m.name, if byo { "byo" } else { "managed" }),
                kind: Kind::Cat13(*m, byo),
                est: 0.0001,
            });
        }
    }
    skipped.push(
        "CAT-13 openai-codex (managed, byo): no key (ChatGPT subscription OAuth only)".into(),
    );
    (out, skipped)
}

fn run(name: &str, kind: &Kind) -> Result<(), Failed> {
    match *kind {
        Kind::Cat1(a) => cat1(name, a),
        Kind::Cat2(r) => cat2(name, r),
        Kind::Cat3(a, c) => cat3(name, a, c),
        Kind::Cat4(a) => cat4(name, a),
        Kind::Cat5(a) => cat5(name, a),
        Kind::Cat6(a) => cat6(name, a),
        Kind::Cat6Reasoning(a) => cat6_reasoning(name, a),
        Kind::Bil13(a) => bil13(name, a),
        Kind::Cat7(a) => cat7(name, a),
        Kind::Cat8(r) => cat8(name, r),
        Kind::Cat13(m, byo) => cat13(name, m, byo),
    }
}

// ---------------------------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------------------------

fn summary() {
    let path = sweep_file();
    let Ok(f) = std::fs::File::open(&path) else {
        println!("no sweep records at {}", path.display());
        return;
    };
    let recs: Vec<Value> = BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    let billed: Vec<&Value> = recs.iter().filter(|r| r["usd"].is_number()).collect();
    let total: f64 = billed
        .iter()
        .map(|r| r["usd"].as_f64().unwrap_or(0.0))
        .sum();
    let mut by_claim: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    let mut by_provider: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    for r in &billed {
        let usd = r["usd"].as_f64().unwrap_or(0.0);
        let claim = r["trial"]
            .as_str()
            .unwrap_or("")
            .split("::")
            .next()
            .unwrap_or("")
            .to_owned();
        let e = by_claim.entry(claim).or_default();
        e.0 += 1;
        e.1 += usd;
        let e = by_provider
            .entry(r["provider"].as_str().unwrap_or("").to_owned())
            .or_default();
        e.0 += 1;
        e.1 += usd;
    }
    println!(
        "sweep {}: {} billed calls, ${total:.4} at card prices",
        path.display(),
        billed.len()
    );
    println!("by claim group:");
    for (k, (n, usd)) in &by_claim {
        println!("  {k:<16} {n:>5} calls  ${usd:.4}");
    }
    println!("by provider:");
    for (k, (n, usd)) in &by_provider {
        println!("  {k:<16} {n:>5} calls  ${usd:.4}");
    }
    let mut top = billed.clone();
    top.sort_by(|a, b| b["usd"].as_f64().partial_cmp(&a["usd"].as_f64()).unwrap());
    println!("most expensive calls:");
    for r in top.iter().take(12) {
        println!(
            "  ${:.4}  {} [{}] in {} out {}",
            r["usd"].as_f64().unwrap_or(0.0),
            r["trial"].as_str().unwrap_or(""),
            r["probe"].as_str().unwrap_or(""),
            r["input"],
            r["output"]
        );
    }
    let provider_total: f64 = billed
        .iter()
        .filter_map(|r| r["provider_usd"].as_f64())
        .sum();
    println!("provider-reported cost on OpenRouter/xAI calls: ${provider_total:.4}");
}

fn main() {
    if std::env::var("VERIFY_CATALOG_SUMMARY").as_deref() == Ok("1") {
        summary();
        return;
    }
    let live = std::env::var("VERIFY_LIVE").as_deref() == Ok("1");
    if std::env::var("VERIFY_CATALOG_PLAN").as_deref() == Ok("1") {
        let (planned, skipped) = plan();
        let total: f64 = planned.iter().map(|p| p.est).sum();
        let mut by_claim: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
        for p in &planned {
            let e = by_claim
                .entry(p.name.split("::").next().unwrap_or(""))
                .or_default();
            e.0 += 1;
            e.1 += p.est;
        }
        for (k, (n, usd)) in &by_claim {
            println!("{k:<16} {n:>5} trials  est ${usd:.4}");
        }
        let mut top: Vec<&Planned> = planned.iter().collect();
        top.sort_by(|a, b| b.est.partial_cmp(&a.est).unwrap());
        for p in top.iter().take(15) {
            println!("  est ${:.4}  {}", p.est, p.name);
        }
        println!("{} trials, estimated ${total:.3}", planned.len());
        println!("{} left out:", skipped.len());
        for s in &skipped {
            println!("  {s}");
        }
        return;
    }
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if live && gateway_bin().exists() {
        for p in plan().0 {
            let Planned { name, kind, .. } = p;
            let n = name.clone();
            trials.push(Trial::test(name, move || run(&n, &kind)));
        }
    }
    let conclusion = libtest_mimic::run(&args, trials);
    cleanup();
    conclusion.exit();
}
