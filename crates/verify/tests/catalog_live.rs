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
//! | `CAT-16`           | candidate   | the vendor still offers it: listed by its own models API (or  |
//! |                    |             | live endpoints on OpenRouter), no retirement due, every       |
//! |                    |             | vendor deprecation notice recorded in `catalog_truth.toml`    |
//! | `CAT-16`           | vendor      | `new-models`: every Anthropic / OpenAI / xAI model in a       |
//! |                    |             | carried family, whatever its modality, is a row (or its alias |
//! |                    |             | or dated snapshot) or in `[[not_carried]]` / `[[retired]]`    |
//!
//! Scope (CAT-1): OpenAI, Anthropic, OpenRouter, xAI, Bedrock and Together candidates. Groq,
//! DeepSeek and Fireworks are out of scope by owner decision; `openai-codex` has no key. The
//! Mistral rows were removed from the catalog until there is a Mistral key (D156).
//! A trial that can't run (no key, over the per-call cost cap, a window the provider exceeds so an
//! over-limit request would be billed) is not listed — `VERIFY_CATALOG_PLAN=1` prints each skip and
//! its reason, and the estimated cost of what is listed.
//!
//! CAT-16 makes listing calls only (no completions), so it costs nothing and needs no gateway.
//! `VERIFY_CATALOG_GAPS=1` prints, without failing, recent Together and OpenRouter models in the
//! vendor namespaces the catalog carries that are not in it.
//!
//! Cost: every billed call appends a record to `target/catalog-live/<sweep>.jsonl`
//! (`VERIFY_CATALOG_SWEEP`, default `sweep`), priced from the ledger's tokens at the card;
//! `VERIFY_CATALOG_SUMMARY=1` totals it. Listed only with `VERIFY_LIVE=1`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/live.rs"]
mod common;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use base64::Engine as _;
use common::free_port;
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
            .timeout(REQUEST_TIMEOUT)
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
            "openrouter-embeddings" => http().get("https://openrouter.ai/api/v1/embeddings/models"),
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
            // Every model xAI serves, image and video generation too (`language-models` lists
            // text output only): what CAT-16 new-models holds to the catalog.
            "xai-all" => http()
                .get("https://api.x.ai/v1/models")
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

/// The tokens-per-minute limits OpenAI sets on the pool key's project for `model`, read with the
/// admin key from the organization's own rate-limit listing (read-only:
/// `GET /v1/organization/projects/{id}/rate_limits`, the project being the one whose key hint
/// matches `OPENAI_API_KEY`). Cached for six hours like the model listings. Returns the model's
/// own limit and its `-long-context` one, each `None` when not listed; `None` without
/// `OPENAI_ADMIN_KEY`.
fn openai_tpm(model: &str) -> Option<Tpm> {
    static LIMITS: OnceLock<Option<Value>> = OnceLock::new();
    let limits = LIMITS.get_or_init(|| {
        let path = sweep_dir().join("listing-openai-rate-limits.json");
        if fresh(&path, Duration::from_secs(6 * 3600))
            && let Some(v) = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        {
            return Some(v);
        }
        let (admin, pool) = (keys().get("OPENAI_ADMIN_KEY")?, key_of("openai")?);
        let get = |url: &str| -> Option<Value> {
            http()
                .get(url)
                .bearer_auth(admin)
                .send()
                .ok()?
                .error_for_status()
                .ok()?
                .json()
                .ok()
        };
        // Every page of a listing, following `last_id`.
        let all = |base: &str| -> Option<Vec<Value>> {
            let mut out = Vec::new();
            let mut after: Option<String> = None;
            loop {
                let url = match &after {
                    Some(a) => format!("{base}&after={a}"),
                    None => base.to_owned(),
                };
                let page = get(&url)?;
                out.extend(page["data"].as_array().cloned().unwrap_or_default());
                match (page["has_more"].as_bool(), page["last_id"].as_str()) {
                    (Some(true), Some(last)) => after = Some(last.to_owned()),
                    _ => return Some(out),
                }
            }
        };
        let projects = all("https://api.openai.com/v1/organization/projects?limit=100")?;
        let project = projects.iter().find_map(|p| {
            let id = p["id"].as_str()?;
            let keys = all(&format!(
                "https://api.openai.com/v1/organization/projects/{id}/api_keys?limit=100"
            ))?;
            keys.iter()
                .any(|k| hint_matches(k["redacted_value"].as_str().unwrap_or(""), pool))
                .then(|| id.to_owned())
        })?;
        let mut tpm = Map::new();
        for r in all(&format!(
            "https://api.openai.com/v1/organization/projects/{project}/rate_limits?limit=100"
        ))? {
            if let (Some(m), Some(t)) = (r["model"].as_str(), r["max_tokens_per_1_minute"].as_u64())
            {
                tpm.insert(m.to_owned(), json!(t));
            }
        }
        let v = Value::Object(tpm);
        let _ = std::fs::write(&path, v.to_string());
        Some(v)
    });
    let l = limits.as_ref()?;
    Some(Tpm {
        base: l[model].as_u64(),
        long: l[format!("{model}-long-context")].as_u64(),
    })
}

/// A model's tokens-per-minute limits on the pool key's project: its own, and the
/// `-long-context` one OpenAI lists beside it for some models.
#[derive(Clone, Copy, Debug)]
struct Tpm {
    base: Option<u64>,
    long: Option<u64>,
}

impl std::fmt::Display for Tpm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = |v: Option<u64>| v.map_or("none".to_owned(), |v| v.to_string());
        write!(f, "TPM {}, long-context {}", n(self.base), n(self.long))
    }
}

/// Whether `key` is the key a redacted hint (`sk-proj-****wxyz`, `sk-...wxyz`) describes: the
/// visible prefix and suffix both match.
fn hint_matches(hint: &str, key: &str) -> bool {
    let (pre, suf) = match hint.split_once("...") {
        Some(parts) => parts,
        None => match (hint.find('*'), hint.rfind('*')) {
            (Some(a), Some(b)) => (&hint[..a], &hint[b + 1..]),
            _ => return false,
        },
    };
    !pre.is_empty() && !suf.is_empty() && key.starts_with(pre) && key.ends_with(suf)
}

/// Whether OpenAI's rate limiter can admit a prompt of `tokens` on `model` for the pool key.
/// Measured 2026-10-01, both before any context check: a prompt over the TPM is refused at once,
/// 429 "Request too large for gpt-4.1 ... TPM: Limit 1000000, Requested 1101959. The input or
/// output tokens must be reduced" (the 1,101,954-word over-limit prompt; a context error never
/// comes); and the limiter counts a request against the minute before comparing, so a lone
/// 136,404-token gpt-5-pro request with nothing else in the minute answered, after 38-56s, 429
/// "Rate limit reached ... Limit 200000, Used 136626, Requested 136626" (Retry-After 22s), every
/// time. So a request is admitted only if twice its size fits the TPM.
///
/// Which TPM: the model's own, even for a long prompt where a larger `-long-context` limit is
/// listed. Measured 2026-10-01 on the Responses API: a 970,100-token prompt on gpt-5.5-pro (TPM
/// 500,000, long-context 2,000,000) answered 429 "You've exceeded the rate limit, please slow
/// down and try again after 1.5e-05 seconds" in 3s, every time, which is what gpt-5.4-pro
/// (200,000 both) answers the same prompt, while gpt-5.5 (2,000,000 both) answered the context
/// error. And gpt-4.1-mini (TPM 4,000,000, long-context 2,000,000) admits a 1,101,954-token
/// prompt, over half its long-context limit. A listed `-long-context` limit can still refuse a
/// prompt over it outright ("Request too large for gpt-4.1 (for limit gpt-4.1-long-context) ...
/// Limit 1000000"), so the prompt must fit that one too. `Err` names the limit that refuses;
/// `Ok` when no limit is known: the trial runs, and a 429 is judged as the rate limit it is.
fn openai_admits(tpm: Option<Tpm>, tokens: u64) -> Result<(), String> {
    let Some(tpm) = tpm else { return Ok(()) };
    match (tpm.base.or(tpm.long), tpm.long) {
        (Some(base), _) if 2 * tokens > base => Err(format!(
            "twice the prompt must fit the model's own {base} TPM ({tpm})"
        )),
        (_, Some(long)) if tokens > long => Err(format!(
            "the prompt is over the {long}-token long-context TPM ({tpm})"
        )),
        _ => Ok(()),
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
    cfg.push_str(common::DEV_ID_SIGNING_TOML);
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

/// Stops the gateway and nats-server. A failed run keeps the gateway's directory (its log holds
/// every warning and billing row), so a failure can be read after the fact, not only re-run.
fn cleanup(failed: bool) {
    for mut c in CHILDREN.lock().unwrap().drain(..) {
        let _ = c.kill();
        let _ = c.wait();
    }
    let dir = sweep_dir().join(format!("gw-{}", std::process::id()));
    if failed && dir.exists() {
        eprintln!("gateway log kept: {}", dir.join("gateway.log").display());
    } else {
        let _ = std::fs::remove_dir_all(dir);
    }
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
    headers: reqwest::header::HeaderMap,
    json: Value,
    raw: String,
}

impl Reply {
    fn header(&self, name: &str) -> Option<String> {
        self.headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    }

    fn excerpt(&self) -> String {
        let s = self.raw.trim();
        let end = s.floor_char_boundary(s.len().min(400));
        s[..end].to_owned()
    }

    /// The status the answer stands for. OpenRouter answers an error during a non-streaming
    /// generation with `200 OK`: a body holding only an `error` object, whose `code` is the HTTP
    /// status it means, or a choice with `finish_reason: "error"` beside partial content
    /// (https://openrouter.ai/docs/api-reference/errors). Either is the provider failing, not an
    /// answer, so it is retried and judged as that status (502 when it names none).
    fn meant_status(&self) -> u16 {
        if self.status != 200 {
            return self.status;
        }
        let j = &self.json;
        let error_only =
            j["error"].is_object() && j.get("choices").is_none() && j.get("output").is_none();
        if error_only {
            return j["error"]["code"]
                .as_u64()
                .and_then(|c| u16::try_from(c).ok())
                .filter(|c| (400..600).contains(c))
                .unwrap_or(502);
        }
        if j["choices"][0]["finish_reason"] == "error" {
            return 502;
        }
        200
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
    let headers = r.headers().clone();
    let raw = r.text().map_err(|e| format!("body: {e}"))?;
    Ok(Reply {
        status,
        provider,
        upstream,
        request_id,
        headers,
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
    // An answer the stock SDKs retry is retried as they do (common::sdk_retry): a stale pooled
    // upstream connection, or a provider's momentary overload, is not a catalog fact. If the last
    // answer is still the provider's own, on a candidate forced alone, the trial is INCONCLUSIVE.
    // Retries stop where one more would not finish inside the test's time budget, and every
    // request's timeout ends there too, so a cell reports instead of being killed by nextest.
    let mut retry = 0;
    loop {
        let mut req = http()
            .post(format!("{}{path}", gw.base))
            .bearer_auth(DEV_TOKEN)
            .header("x-beyond-model", arm.row.model)
            .header(walk_header, arm.provider())
            .header("x-beyond-cache", "off")
            .header("content-type", "application/json")
            .body(bytes.clone());
        if let Some(left) = common::time_left() {
            req = req.timeout(left.min(REQUEST_TIMEOUT));
        }
        let at = Instant::now();
        let r = send(req).map_err(|e| match common::time_left() {
            Some(left) if left.is_zero() => format!(
                "no answer inside the test's time budget ({:?}): {e}",
                common::test_budget().unwrap_or_default()
            ),
            _ => e,
        })?;
        let status = r.meant_status();
        if !common::sdk_retryable(status, |h| r.header(h)) {
            return Ok(r);
        }
        if let Some(wait) = common::sdk_retry(status, retry, at.elapsed(), |h| r.header(h)) {
            retry += 1;
            println!(
                "retry {retry} in {wait:?} after HTTP {} ({:?}): {}",
                r.status,
                at.elapsed(),
                r.excerpt()
            );
            std::thread::sleep(wait);
            continue;
        }
        // The provider's own answer (it named itself), on a request no other candidate could take.
        if walk == Walk::Only
            && let Some(p) = &r.provider
        {
            common::provider_unavailable(format!(
                "{p} answered {status} on all {} attempts at {}: {}",
                retry + 1,
                arm.cand.upstream_model,
                r.excerpt()
            ));
        }
        // The gateway's own 5xx because this candidate ended the connection without answering:
        // the gateway logged the upstream error, and it is the peer's (see `peer_ended`). Forced
        // alone, or after the provider had the whole request (the walk never resends one), no
        // other candidate could have answered.
        if r.provider.is_none()
            && r.status >= 500
            && let Some((p, err)) = r.request_id.as_deref().and_then(upstream_error)
            && p == arm.provider()
            && peer_ended(&err)
            && (walk == Walk::Only || r.raw.contains("after receiving the request"))
        {
            common::provider_unavailable(format!(
                "{p} ended the connection without answering on all {} attempts at {} \
                 (the gateway logged: {err}); the gateway answered {}: {}",
                retry + 1,
                arm.cand.upstream_model,
                r.status,
                r.excerpt()
            ));
        }
        return Ok(r);
    }
}

/// The longest one request may take: under nextest's 180s terminate-after, so a slow reasoning
/// call fails by name.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(170);

/// The upstream error the gateway logged for a request it could not answer with the provider's
/// own response (`upstream request errored`): the provider it was talking to and the error.
fn upstream_error(request_id: &str) -> Option<(String, String)> {
    let gw = gateway().ok()?;
    let f = std::fs::File::open(&gw.log).ok()?;
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| l.contains("upstream request errored") && l.contains(request_id))
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .map(|v| v.get("fields").cloned().unwrap_or(v))
        .filter(|f| f["request_id"] == request_id)
        .last()
        .map(|f| {
            (
                f["provider"].as_str().unwrap_or_default().to_owned(),
                f["error"].as_str().unwrap_or_default().to_owned(),
            )
        })
}

/// Whether an upstream error the gateway logged (pingora's `Upstream <type> ... cause: ...`) is
/// the provider ending the exchange without an answer: it refused or dropped the connection, or
/// sent a reset or GOAWAY of its own (h2's "error received"), which RFC 9113 lets it send for its
/// own reasons. Not when that frame names an error in what the gateway sent (a protocol, flow
/// control, frame size, closed-stream or header compression error), and not the gateway's own
/// timeouts or failures (`WriteTimedout`: D118, `ReadTimedout`, `InternalError`).
fn peer_ended(err: &str) -> bool {
    const GATEWAY_FAULT: &[&str] = &[
        "unspecific protocol error detected",
        "flow-control protocol violated",
        "received frame when stream half-closed",
        "frame with invalid size",
        "unable to maintain the header compression context",
    ];
    let Some(rest) = err.strip_prefix("Upstream ") else {
        return false;
    };
    if GATEWAY_FAULT.iter().any(|r| err.contains(r)) {
        return false;
    }
    match rest.split_whitespace().next().unwrap_or_default() {
        "ConnectionClosed"
        | "ConnectRefused"
        | "ConnectTimedout"
        | "ConnectNoRoute"
        | "ConnectError"
        | "TLSHandshakeFailure"
        | "TLSHandshakeTimedout" => true,
        "H2Error" | "ReadError" => {
            let lower = err.to_ascii_lowercase();
            lower.contains("error received:") || lower.contains("reset by peer")
        }
        _ => false,
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
    if r.meant_status() != 200 {
        return Err(format!(
            "{probe}: HTTP 200 carrying an error ({}): {}",
            r.meant_status(),
            r.excerpt()
        ));
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
/// Enough arithmetic that a reasoning model reasons at low effort, not so much that it spends 4k.
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
    let id = r.request_id.as_deref().ok_or_else(|| {
        format!(
            "no x-beyond-request-id on HTTP {} (not our gateway's proxy listener?): {}",
            r.status,
            r.excerpt()
        )
    })?;
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
        } else if r.status == 429 {
            // A rate limit is not the context answer, whatever its words ("exceeded the rate
            // limit" matches `CONTEXT_WORDS`). The provider's own 429 means this trial cannot
            // run on the pool key now: INCONCLUSIVE, never a pass.
            if r.provider.as_deref() == Some(arm.provider()) {
                common::provider_unavailable(format!(
                    "{} rate-limited the over-limit prompt ({words} words) instead of answering \
                     its context check: {}",
                    arm.provider(),
                    r.excerpt()
                ));
            }
            problems.push(format!(
                "over-limit: HTTP 429 is a rate limit, not a context error: {}",
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
                // A row whose truth entry follows the model's maker (`follows = "maker"`, D116)
                // keeps the maker's list price: Together charging less is a note. Together
                // charging more is billing below cost, which the owner decides, so it fails.
                let maker = truth(arm.row.model)
                    .and_then(|t| t.get("follows"))
                    .and_then(|v| v.as_str())
                    == Some("maker");
                for (field, card, k) in
                    [("input", p.input, "input"), ("output", p.output, "output")]
                {
                    match m["pricing"][k].as_f64() {
                        Some(v) if price_eq(card, v) => {}
                        Some(v) if maker && card.parse().is_ok_and(|c: f64| v < c) => note(
                            trial,
                            &format!(
                                "{field}: Together lists {v}, below the maker's {card} on the card"
                            ),
                        ),
                        Some(v) if maker => problems.push(format!(
                            "{field}: Together lists {v}, above the maker's {card} on the card: \
                             we bill below cost (owner pricing decision)"
                        )),
                        Some(v) => {
                            problems.push(format!("{field}: card {card}, Together lists {v}"))
                        }
                        None => {}
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

/// The OpenAI-model mounts use gpt-4o-mini: the cheapest OpenAI row not scheduled to retire, and
/// not a reasoning model, so the body's `max_tokens` passes through as sent (gpt-4.1-nano, their
/// model until then, left the catalog: D243).
const MOUNTS: &[Mount] = &[
    Mount {
        name: "openai",
        path: "/openai/v1/chat/completions",
        model: "gpt-4o-mini",
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
        model: "openai/gpt-4o-mini",
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
// CAT-16: catalog currency (listing calls only, no completions)
// ---------------------------------------------------------------------------------------------

/// A vendor page (deprecation notices, Together's serverless table), fetched once and cached on
/// disk for six hours, like the model listings.
fn page(name: &str, url: &str) -> Option<&'static str> {
    static CACHE: OnceLock<Mutex<BTreeMap<String, Option<&'static str>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut cache = cache.lock().unwrap();
    if let Some(v) = cache.get(name) {
        return *v;
    }
    let path = sweep_dir().join(format!("page-{name}.md"));
    let text = fresh(&path, Duration::from_secs(6 * 3600))
        .then(|| std::fs::read_to_string(&path).ok())
        .flatten()
        .or_else(|| {
            let t = http()
                .get(url)
                .send()
                .ok()?
                .error_for_status()
                .ok()?
                .text()
                .ok()?;
            let _ = std::fs::write(&path, &t);
            Some(t)
        });
    let leaked: Option<&'static str> = text.map(|t| &*Box::leak(t.into_boxed_str()));
    cache.insert(name.to_owned(), leaked);
    leaked
}

/// Where each vendor publishes its retirements, as markdown tables of (date, model).
fn deprecation_page(provider: &str) -> Option<(&'static str, &'static str)> {
    Some(match provider {
        "anthropic" => (
            "anthropic-deprecations",
            "https://platform.claude.com/docs/en/about-claude/model-deprecations.md",
        ),
        "openai" => (
            "openai-deprecations",
            "https://developers.openai.com/api/docs/deprecations.md",
        ),
        "together" => (
            "together-deprecations",
            "https://docs.together.ai/docs/deprecations.md",
        ),
        _ => return None,
    })
}

/// `October 23, 2026`, `Oct 1, 2026`, `2026-09-24` (also with U+2011 hyphens) → `YYYY-MM-DD`.
fn doc_date(cell: &str) -> Option<String> {
    let s = cell.trim().replace('\u{2011}', "-");
    let b = s.as_bytes();
    if b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && s.replace('-', "").bytes().all(|c| c.is_ascii_digit())
    {
        return Some(s);
    }
    const MONTHS: [&str; 12] = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    let mut words = s.split([' ', ',']).filter(|w| !w.is_empty());
    let (m, d, y) = (words.next()?, words.next()?, words.next()?);
    if words.next().is_some() {
        return None;
    }
    let m = MONTHS
        .iter()
        .position(|p| m.len() >= 3 && m.to_ascii_lowercase().starts_with(p))?
        + 1;
    let (d, y): (u32, u32) = (d.parse().ok()?, y.parse().ok()?);
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

/// Every `(model id, retirement date)` a vendor's deprecation page lists: each markdown table with
/// a date column and a model column (not the replacement), outside fine-tuning sections. A model
/// cell may hold several backticked ids (OpenAI: `` `gpt-4-0613` \| `gpt-4` ``); a cell that is
/// not a plain date (`Not sooner than ...`, `To be announced`) lists nothing.
fn deprecations(provider: &str) -> Option<Vec<(String, String)>> {
    let (name, url) = deprecation_page(provider)?;
    let text = page(name, url)?;
    let mut out = Vec::new();
    let mut cols: Option<(usize, usize)> = None;
    let mut skip_section = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with('#') {
            skip_section = line.to_ascii_lowercase().contains("fine-tun");
            cols = None;
            continue;
        }
        if !line.starts_with('|') {
            cols = None;
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split(" | ").map(str::trim).collect();
        let Some((date_col, model_col)) = cols else {
            let lower: Vec<String> = cells.iter().map(|c| c.to_ascii_lowercase()).collect();
            let date_col = lower.iter().position(|c| c.contains("date"));
            let model_col = lower.iter().position(|c| {
                (c.contains("model") || c == "system")
                    && !["replacement", "price", "type"]
                        .iter()
                        .any(|w| c.contains(w))
            });
            cols = date_col.zip(model_col);
            continue;
        };
        if skip_section || cells.len() <= date_col.max(model_col) {
            continue;
        }
        let Some(date) = doc_date(cells[date_col]) else {
            continue;
        };
        let cell = cells[model_col];
        let ids: Vec<&str> = if cell.contains('`') {
            cell.split('`').skip(1).step_by(2).collect()
        } else {
            vec![cell]
        };
        for id in ids
            .into_iter()
            .map(str::trim)
            .filter(|i| !i.is_empty() && !i.contains(' '))
        {
            out.push((id.to_owned(), date.clone()));
        }
    }
    Some(out)
}

/// An id without its dated snapshot suffix (`-YYYY-MM-DD`, `-YYYYMMDD`, `-MMDD`).
fn undated(id: &str) -> &str {
    let b = id.as_bytes();
    let n = b.len();
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
    if n > 11
        && b[n - 11] == b'-'
        && b[n - 6] == b'-'
        && b[n - 3] == b'-'
        && digits(n - 10..n - 6)
        && digits(n - 5..n - 3)
        && digits(n - 2..n)
    {
        &id[..n - 11]
    } else if n > 9 && b[n - 9] == b'-' && digits(n - 8..n) {
        &id[..n - 9]
    } else if n > 5 && b[n - 5] == b'-' && digits(n - 4..n) {
        &id[..n - 5]
    } else {
        id
    }
}

/// Today, `YYYY-MM-DD` (UTC).
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let z = i64::try_from(secs / 86_400).unwrap_or(0) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The `verify/catalog_truth.toml` entries of one array (`retired`, `not_carried`).
fn truth_entries(list: &str) -> &'static [toml::Value] {
    truth_file()
        .get(list)
        .and_then(toml::Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn field<'a>(e: &'a toml::Value, k: &str) -> Option<&'a str> {
    e.get(k).and_then(toml::Value::as_str)
}

/// The `[[retired]]` entry recording a vendor retirement of `id` at `provider`, or of `row`.
fn retirement(provider: &str, id: &str, row: Option<&str>) -> Option<&'static toml::Value> {
    truth_entries("retired").iter().find(|e| {
        (field(e, "provider") == Some(provider) && field(e, "id") == Some(id))
            || (row.is_some() && field(e, "model") == row)
    })
}

/// The date an entry records: `retired` (past) or `retires` (scheduled).
fn retirement_date(e: &toml::Value) -> Option<&str> {
    field(e, "retired").or_else(|| field(e, "retires"))
}

/// OpenRouter's endpoints for a model. Uncached: it is the live state.
fn openrouter_endpoints(id: &str) -> Result<Vec<Value>, String> {
    let url = format!("https://openrouter.ai/api/v1/models/{id}/endpoints");
    let mut last = String::new();
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(2 * attempt));
        }
        match http().get(&url).send() {
            Ok(r) if r.status().is_success() => {
                let v: Value = r.json().map_err(|e| e.to_string())?;
                return Ok(v["data"]["endpoints"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default());
            }
            Ok(r) if r.status().as_u16() == 404 => return Err(format!("{url}: 404")),
            Ok(r) => last = format!("{url}: HTTP {}", r.status()),
            Err(e) => last = format!("{url}: {e}"),
        }
    }
    Err(last)
}

/// Whether an OpenRouter endpoint can be counted on to serve: it answered in the last 30 minutes
/// (`uptime_last_30m` above 0), or it had no traffic to measure (`null`) and OpenRouter has not
/// flagged it (`status` 0). An endpoint at 0% uptime answers 410 Gone (D182). One with no recent
/// traffic and a negative `status` is not evidence of anything: Fireworks' `z-ai/glm-5.2`, whose
/// serverless ended on 2026-09-25 (D181), lists `status` -5 and `uptime_last_30m` null, and so
/// did SambaNova's `google/gemma-4-31b-it` (2026-10-01); counted live, either would vouch for a
/// card capability no host serves. A flagged endpoint that is still answering (`status` -2 or
/// -5 at 40-95% uptime, seen the same day) counts by its uptime.
fn endpoint_live(e: &Value) -> bool {
    match e["uptime_last_30m"].as_f64() {
        Some(up) => up > 0.0,
        None => e["status"].as_i64().unwrap_or(0) >= 0,
    }
}

/// Bedrock's control plane, read with the same bearer key the runtime takes.
fn bedrock_get(path: &str) -> Result<Value, String> {
    let key = key_of("bedrock").ok_or("no Bedrock key")?;
    let url = format!("https://bedrock.us-east-1.amazonaws.com{path}");
    let r = http()
        .get(&url)
        .bearer_auth(key)
        .send()
        .map_err(|e| format!("{url}: {e}"))?;
    let status = r.status();
    let body: Value = r.json().unwrap_or(Value::Null);
    if status.is_success() {
        Ok(body)
    } else {
        Err(format!("{url}: HTTP {status} {body}"))
    }
}

/// CAT-16 for one candidate: the vendor still offers it, today.
///
/// - Anthropic, OpenAI, xAI: listed by the vendor's own `/v1/models` (Anthropic by snapshot, xAI by
///   alias). Together: listed as a chat model and in the serverless models table. Bedrock: the
///   inference profile is `ACTIVE` and its foundation model is not `LEGACY` (a `LEGACY` model needs
///   a `[[retired]]` entry with AWS's end-of-life date). OpenRouter: listed, and at least one
///   endpoint answered in the last 30 minutes (an endpoint at 0% uptime answers 410 Gone); a card
///   capability (tools, structured outputs where the candidate serves them) needs a live endpoint
///   that lists it.
/// - Every retirement the vendor's deprecation page lists for the id, or a dated snapshot of it,
///   is recorded in `verify/catalog_truth.toml` `[[retired]]`, and none recorded for the row is
///   due: a retirement date of today or earlier fails.
fn cat16(_trial: &str, row: &'static ModelRoute, cand: Candidate) -> Result<(), Failed> {
    let provider = by_id(cand.provider).name;
    let id = cand.upstream_model;
    let today = today();
    let mut problems = Vec::new();
    match provider {
        "anthropic" | "openai" | "xai" => {
            if listing(provider).is_none() {
                return Err(format!("{provider}'s /v1/models could not be read").into());
            }
            match listed(provider, id) {
                Some(m) => println!("{provider} lists {id} as {}", m["id"]),
                None => problems.push(format!("{provider}'s /v1/models no longer lists {id}")),
            }
        }
        "together" => {
            match listed("together", id) {
                Some(m) if m["type"] == "chat" => println!("together lists {id} ({})", m["type"]),
                Some(m) => problems.push(format!("together lists {id} as {}, not chat", m["type"])),
                None => problems.push(format!("together's /v1/models no longer lists {id}")),
            }
            let table = page(
                "together-serverless",
                "https://docs.together.ai/docs/serverless/models.md",
            )
            .ok_or("Together's serverless models page could not be read")?;
            if !table.contains(&format!("| {id} |")) {
                problems.push(format!(
                    "{id} is not in Together's serverless models table (docs.together.ai/docs/serverless/models)"
                ));
            }
        }
        "bedrock" => {
            let p = bedrock_get(&format!("/inference-profiles/{id}"))?;
            println!("bedrock profile {id}: {}", p["status"]);
            if p["status"] != "ACTIVE" {
                problems.push(format!("inference profile {id} is {}", p["status"]));
            }
            let models = bedrock_get("/foundation-models?byProvider=anthropic")?;
            let arns: Vec<&str> = p["models"]
                .as_array()
                .map(|a| a.iter().filter_map(|m| m["modelArn"].as_str()).collect())
                .unwrap_or_default();
            let fm = models["modelSummaries"].as_array().and_then(|a| {
                a.iter().find(|m| {
                    m["modelArn"]
                        .as_str()
                        .is_some_and(|arn| arns.contains(&arn))
                })
            });
            match fm {
                None => problems.push(format!("no us-east-1 foundation model behind {id}")),
                Some(m) => {
                    let life = &m["modelLifecycle"];
                    println!("bedrock {}: {life}", m["modelId"]);
                    match life["status"].as_str() {
                        Some("ACTIVE") => {}
                        Some("LEGACY") => {
                            let eol = life["endOfLifeTime"].as_str().unwrap_or("?");
                            if retirement(provider, id, None).is_none() {
                                problems.push(format!(
                                    "{} is LEGACY on Bedrock (end of life {eol}): record it in [[retired]]",
                                    m["modelId"]
                                ));
                            }
                        }
                        other => problems.push(format!("{} is {other:?} on Bedrock", m["modelId"])),
                    }
                }
            }
        }
        "openrouter" => {
            let listed_here = listed("openrouter", id).is_some()
                || listing("openrouter-embeddings").is_some_and(|l| {
                    l["data"]
                        .as_array()
                        .is_some_and(|a| a.iter().any(|m| m["id"] == id))
                });
            if !listed_here {
                problems.push(format!("OpenRouter no longer lists {id}"));
            }
            if let Some(exp) = listed("openrouter", id).and_then(|m| m["expiration_date"].as_str())
            {
                problems.push(format!("OpenRouter will remove {id} on {exp}"));
            }
            let eps = openrouter_endpoints(id)?;
            let (live, dead): (Vec<&Value>, Vec<&Value>) =
                eps.iter().partition(|e| endpoint_live(e));
            println!(
                "openrouter {id}: {} live endpoints {:?}; down {:?}",
                live.len(),
                live.iter()
                    .map(|e| e["provider_name"].as_str().unwrap_or("?"))
                    .collect::<Vec<_>>(),
                dead.iter()
                    .map(|e| e["provider_name"].as_str().unwrap_or("?"))
                    .collect::<Vec<_>>(),
            );
            if live.is_empty() {
                problems.push(format!(
                    "{id} has no live OpenRouter endpoint ({} listed, all at 0% uptime)",
                    eps.len()
                ));
            }
            let lists = |param: &str| {
                live.iter().any(|e| {
                    e["supported_parameters"]
                        .as_array()
                        .is_some_and(|a| a.iter().any(|p| p == param))
                })
            };
            if row.card.features & TOOLS != 0 && !lists("tools") {
                problems.push(format!(
                    "the card lists tools, but no live OpenRouter endpoint for {id} takes them"
                ));
            }
            if row.card.features & STRUCTURED_OUTPUTS != 0
                && serves_structured_outputs(&cand)
                && !lists("structured_outputs")
            {
                problems.push(format!(
                    "the card lists structured outputs, but no live OpenRouter endpoint for {id} \
                     takes them (a json_schema request has nowhere to go)"
                ));
            }
        }
        _ => return Err(format!("{provider}: no listing to check").into()),
    }
    // The deprecation notices of the vendor, and of the model's maker when OpenRouter serves a
    // first-party model (`openai/o3-pro` is OpenAI's `o3-pro`; `anthropic/claude-opus-4.1` is
    // Anthropic's `claude-opus-4-1`): a maker's retirement retires the model under its name, even
    // where another host still runs it.
    let mut sources: Vec<(&str, Vec<String>)> = vec![(provider, vec![id.to_owned()])];
    if provider == "openrouter"
        && let Some((maker, model)) = id.split_once('/')
        && deprecation_page(maker).is_some()
    {
        sources.push((
            maker,
            vec![
                model.to_owned(),
                model.replace('.', "-"),
                row.model.to_owned(),
            ],
        ));
    }
    for (vendor, names) in sources {
        if deprecation_page(vendor).is_none() {
            continue;
        }
        let notices =
            deprecations(vendor).ok_or(format!("{vendor}'s deprecation page could not be read"))?;
        for (dep, date) in notices
            .iter()
            .filter(|(d, _)| names.iter().any(|n| d == n || undated(d) == n))
        {
            let recorded =
                retirement(vendor, dep, None).or_else(|| retirement(provider, id, Some(row.model)));
            match recorded {
                None => problems.push(format!(
                    "{vendor}'s deprecation page retires {dep} on {date}; no [[retired]] entry \
                     in verify/catalog_truth.toml"
                )),
                Some(e) => println!(
                    "{vendor} retires {dep} on {date}: recorded ({})",
                    retirement_date(e).unwrap_or("?")
                ),
            }
        }
    }
    // Nothing recorded for this row or candidate is due.
    for e in truth_entries("retired") {
        let touches = field(e, "model") == Some(row.model)
            || (field(e, "provider") == Some(provider) && field(e, "id") == Some(id));
        if let (true, Some(date)) = (touches, retirement_date(e))
            && date <= today.as_str()
        {
            problems.push(format!(
                "{provider}/{id} on {} was due to retire on {date}",
                row.model
            ));
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// Every model a first-party vendor lists in a family the catalog carries, as `(id, aliases)`:
/// Claude; GPT (`gpt-`, and ChatGPT's `chatgpt-` and `chat-latest`) and the o-series; Grok. No id
/// in a family is left out by its name or modality: [`cat16_new_models`] holds each to a row, a
/// vendor-listed alias of a row, a dated snapshot of one ([`undated`]), or a recorded decision.
/// Models outside the gateway's endpoints (audio, realtime, transcription, image and video
/// generation, the Live API) are recorded one by one in `[[not_carried]]` with the reason. xAI is
/// read from `/v1/models`, which lists its image and video models beside the language models.
fn first_party_models(vendor: &str) -> Option<Vec<(String, Vec<String>)>> {
    let l = listing(if vendor == "xai" { "xai-all" } else { vendor })?;
    let items = l
        .get("data")
        .or_else(|| l.get("models"))
        .and_then(Value::as_array)?;
    let family = |id: &str| match vendor {
        "anthropic" => id.starts_with("claude-"),
        "openai" => {
            id.starts_with("gpt-")
                || id.starts_with("chatgpt-")
                || id == "chat-latest"
                || (id.starts_with('o') && id.as_bytes().get(1).is_some_and(u8::is_ascii_digit))
        }
        "xai" => id.starts_with("grok-"),
        _ => false,
    };
    Some(
        items
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?;
                let aliases: Vec<String> = m["aliases"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_owned))
                            .collect()
                    })
                    .unwrap_or_default();
                // Anthropic lists only snapshots: the alias is the undated id. OpenAI lists dated
                // snapshots beside their alias, each checked through its undated form (so a
                // snapshot whose alias is gone is still caught); xAI's ids may carry a date and
                // name the alias in `aliases`.
                family(id).then(|| (id.to_owned(), aliases))
            })
            .collect(),
    )
}

/// Whether an id (or its undated form) is a catalog row or a candidate at `provider`.
fn carried(provider: &str, id: &str) -> bool {
    let base = undated(id);
    MODEL_ROUTES.iter().any(|r| {
        r.model == id
            || r.model == base
            || r.candidates.iter().chain(r.responses).any(|c| {
                by_id(c.provider).name == provider
                    && (c.upstream_model == id || c.upstream_model == base)
            })
    })
}

/// Whether `[[not_carried]]`, `[[retired]]` or `[[not_serverless]]` records the id at `provider`
/// (a deliberate gap).
fn recorded_gap(provider: &str, id: &str) -> bool {
    let base = undated(id);
    ["not_carried", "retired", "not_serverless"]
        .iter()
        .any(|list| {
            truth_entries(list).iter().any(|e| {
                (field(e, "provider") == Some(provider)
                    && field(e, "id").is_some_and(|x| x == id || x == base))
                    || field(e, "model").is_some_and(|x| x == id || x == base)
            })
        })
}

/// CAT-16 for a first-party vendor: every model it lists in a family the catalog carries is a
/// row, or a recorded decision (`[[not_carried]]`, `[[retired]]`), so a new release is a red cell.
fn cat16_new_models(_trial: &str, vendor: &str) -> Result<(), Failed> {
    let models =
        first_party_models(vendor).ok_or(format!("{vendor}'s /v1/models could not be read"))?;
    let mut gaps = Vec::new();
    for (id, aliases) in &models {
        let names: Vec<&str> = std::iter::once(id.as_str())
            .chain(aliases.iter().map(String::as_str))
            .collect();
        if names
            .iter()
            .any(|n| carried(vendor, n) || recorded_gap(vendor, n))
        {
            continue;
        }
        gaps.push(id.clone());
    }
    println!(
        "{vendor}: {} models in carried families, {} gaps",
        models.len(),
        gaps.len()
    );
    if gaps.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{vendor} lists {} model(s) in a family the catalog carries that are neither a row nor \
             recorded in verify/catalog_truth.toml [[not_carried]]: {}",
            gaps.len(),
            gaps.join(", ")
        )
        .into())
    }
}

/// `VERIFY_CATALOG_GAPS=1`: a report, never a failure, of recent models on Together and
/// OpenRouter in the vendor namespaces the catalog already carries (`qwen/`, `z-ai/`, ...) that
/// are neither candidates nor recorded as not carried. These hosts list hundreds of models, most
/// of them niche, so a gap here is a prompt to look, not a defect.
fn gaps_report() {
    const RECENT: i64 = 120 * 86_400;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0));
    for host in ["openrouter", "together"] {
        let namespaces: std::collections::BTreeSet<&str> = MODEL_ROUTES
            .iter()
            .flat_map(|r| r.candidates.iter())
            .filter(|c| by_id(c.provider).name == host)
            .filter_map(|c| c.upstream_model.split_once('/').map(|(ns, _)| ns))
            .collect();
        let Some(items) = listing(host).and_then(|l| {
            l.get("data")
                .and_then(Value::as_array)
                .or_else(|| l.as_array())
        }) else {
            println!("{host}: listing unavailable");
            continue;
        };
        let mut rows: Vec<(i64, &str)> = items
            .iter()
            .filter_map(|m| Some((m["created"].as_i64()?, m["id"].as_str()?)))
            .filter(|(created, id)| {
                now - created < RECENT
                    && !id.contains(':')
                    && id
                        .split_once('/')
                        .is_some_and(|(ns, _)| namespaces.contains(ns))
                    && !carried(host, id)
                    && !recorded_gap(host, id)
            })
            .filter(|(_, id)| {
                host != "together"
                    || items.iter().any(|m| {
                        m["id"] == *id
                            && m["type"] == "chat"
                            && m["pricing"]["input"].as_f64().unwrap_or(0.0) > 0.0
                    })
            })
            .collect();
        rows.sort_by(|a, b| b.cmp(a));
        println!(
            "{host}: {} models from the last 120 days in carried namespaces ({}) are not in the \
             catalog:",
            rows.len(),
            namespaces.iter().copied().collect::<Vec<_>>().join(", ")
        );
        for (created, id) in rows {
            println!("  {:>3} days ago  {id}", (now - created) / 86_400);
        }
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
    Cat16(&'static ModelRoute, Candidate),
    Cat16NewModels(&'static str),
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
            // A prompt OpenAI's rate limiter can never admit on our project gets a 429 every time,
            // not the context answer the trial is about.
            let tpm_skip = |words: u64, what: &str, skipped: &mut Vec<String>| {
                if arm.provider() != "openai" {
                    return false;
                }
                let model = arm.cand.upstream_model;
                match openai_admits(openai_tpm(model), words) {
                    Ok(()) => false,
                    Err(why) => {
                        skipped.push(format!(
                            "{} {what}: a {words}-token prompt is over what OpenAI's rate \
                             limiter admits for {model} on the pool key's project: {why}",
                            name("CAT-3", route, m),
                        ));
                        true
                    }
                }
            };
            let over = match theirs {
                _ if tpm_skip(window + window / 20 + 2000, "over-limit", &mut skipped) => None,
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
                    near_done = true;
                } else if !tpm_skip(window * 9 / 10 - 500, "near-limit", &mut skipped) {
                    // A rate-limited candidate leaves the near-limit call to the row's next one.
                    near = true;
                    near_done = true;
                }
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
        // CAT-16: listing calls only. One cell per distinct in-scope (provider, id) of the row (a
        // GPT row's Responses arm is the same OpenAI id as its first candidate).
        let mut seen = Vec::new();
        for c in row.candidates.iter().chain(row.responses) {
            if !in_scope(c) || seen.contains(&(c.provider, c.upstream_model)) {
                continue;
            }
            let route = if seen.iter().any(|(p, _)| *p == c.provider) {
                "openai-responses"
            } else {
                by_id(c.provider).name
            };
            seen.push((c.provider, c.upstream_model));
            out.push(Planned {
                name: name("CAT-16", route, row.model),
                kind: Kind::Cat16(row, *c),
                est: 0.0,
            });
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
    for vendor in ["anthropic", "openai", "xai"] {
        if key_of(vendor).is_some() {
            out.push(Planned {
                name: name("CAT-16", vendor, "new-models"),
                kind: Kind::Cat16NewModels(vendor),
                est: 0.0,
            });
        } else {
            skipped.push(format!("CAT-16 {vendor} new-models: no key"));
        }
    }
    (out, skipped)
}

// ---------------------------------------------------------------------------------------------
// The sweep's own checks (hermetic; named after their functions, with no `::`, so `verify` reads
// each as a tagged hermetic test, not a live cell)
// ---------------------------------------------------------------------------------------------

/// The time budget a live cell's retries must finish inside is the one nextest enforces, read
/// from this workspace's `.config/nextest.toml`: 60s × 3 under `verify` and `verify-isolated`,
/// the reconciliation and long-session overrides by binary, `ci`'s own, and none under `default`
/// (which never terminates a test). A cell then reports INCONCLUSIVE instead of a nextest
/// TIMEOUT (CAT-3 on gpt-5-pro, 2026-10-01: two 55s rate-limited attempts and 22s waits).
/// claim: CAT-3
fn retries_end_inside_the_nextest_budget() -> Result<(), Failed> {
    let text = std::fs::read_to_string(repo_root().join(".config/nextest.toml"))
        .map_err(|e| e.to_string())?;
    let config: toml::Value = toml::from_str(&text).map_err(|e| e.to_string())?;
    let min = |m: u64| Some(Duration::from_secs(m * 60));
    let cases: &[(&str, Option<&str>, Option<Duration>)] = &[
        ("verify", Some("beyond-ai-verify::catalog_live"), min(3)),
        ("verify", Some("beyond-ai-verify::reconcile_live"), min(18)),
        ("verify", Some("beyond-ai-verify::long_live"), min(35)),
        ("verify-isolated", Some("beyond-ai-verify::live"), min(3)),
        (
            "verify-isolated",
            Some("beyond-ai-verify::long_live"),
            min(35),
        ),
        ("ci", Some("beyond-ai-verify::catalog_live"), min(3)),
        ("default", Some("beyond-ai-verify::catalog_live"), None),
        ("verify", None, min(3)),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|&(profile, binary, want)| {
            let got = common::budget_in(&config, profile, binary);
            (got != want).then(|| format!("{profile} {binary:?}: {got:?}, want {want:?}"))
        })
        .collect();
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(wrong.join("; ").into())
    }
}

/// The CAT-3 planner holds an OpenAI prompt to the limits measured on 2026-10-01 (see
/// `openai_admits`): twice the prompt within the model's own TPM even where a larger
/// `-long-context` limit is listed (gpt-5.5-pro's 970,100-token over-limit prompt, rate-limited
/// three times in the clean run), the prompt itself within a listed long-context limit, and no
/// limit known means the trial runs.
/// claim: CAT-3
fn cat3_plans_openai_prompts_to_the_measured_limit() -> Result<(), Failed> {
    let tpm = |base, long| Some(Tpm { base, long });
    let cases: &[(&str, Option<Tpm>, u64, bool)] = &[
        (
            "gpt-5.5-pro over-limit",
            tpm(Some(500_000), Some(2_000_000)),
            970_100,
            false,
        ),
        (
            "gpt-5.5 over-limit",
            tpm(Some(2_000_000), Some(2_000_000)),
            970_100,
            true,
        ),
        (
            "gpt-4.1-mini over-limit",
            tpm(Some(4_000_000), Some(2_000_000)),
            1_101_954,
            true,
        ),
        (
            "gpt-4.1 over-limit",
            tpm(Some(800_000), Some(1_000_000)),
            1_101_954,
            false,
        ),
        (
            "gpt-5-pro near-limit",
            tpm(Some(200_000), None),
            136_404,
            false,
        ),
        (
            "long-context only",
            tpm(None, Some(2_000_000)),
            970_100,
            true,
        ),
        ("no limit known", None, 970_100, true),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|&(what, tpm, tokens, want)| {
            let got = openai_admits(tpm, tokens);
            (got.is_ok() != want).then(|| format!("{what}: {got:?}, want admitted={want}"))
        })
        .collect();
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(wrong.join("; ").into())
    }
}

/// What makes a catalog cell INCONCLUSIVE rather than failed, on the two shapes of 2026-10-01:
/// OpenRouter's error-in-200 (a body of only `error`, whose `code` is the status it means, or a
/// choice with `finish_reason: "error"`) is read as that status, while a real answer, an empty
/// answer and a Responses body (`"error": null`) stay 200; and a gateway 502 is the provider's
/// only when the error the gateway logged is the peer ending the exchange, never a frame error
/// against what the gateway sent, nor the gateway's own timeouts.
/// claim: CAT-3, CAT-6, CAT-7
fn provider_failures_are_attributed() -> Result<(), Failed> {
    let reply = |status: u16, body: &str| Reply {
        status,
        provider: None,
        upstream: None,
        request_id: None,
        headers: reqwest::header::HeaderMap::new(),
        json: serde_json::from_str(body).unwrap_or(Value::Null),
        raw: body.to_owned(),
    };
    let mut wrong = Vec::new();
    for (status, body, want) in [
        (
            200,
            r#"{"error":{"code":502,"message":"Provider returned error"}}"#,
            502,
        ),
        (
            200,
            r#"{"error":{"code":429,"message":"rate limited"}}"#,
            429,
        ),
        (200, r#"{"error":{"message":"no code"}}"#, 502),
        (
            200,
            r#"{"id":"g","choices":[{"finish_reason":"error","message":{"content":""}}],"error":{"code":502}}"#,
            502,
        ),
        (
            200,
            r#"{"id":"g","choices":[{"finish_reason":"stop","message":{"content":""}}]}"#,
            200,
        ),
        (200, r#"{"id":"r","output":[],"error":null}"#, 200),
        (400, r#"{"error":{"code":400,"message":"context"}}"#, 400),
    ] {
        let got = reply(status, body).meant_status();
        if got != want {
            wrong.push(format!("{status} {body}: means {got}, want {want}"));
        }
    }
    for (err, want) in [
        (
            "Upstream ConnectionClosed context: Peer: openrouter.ai:443",
            true,
        ),
        ("Upstream TLSHandshakeFailure context: unexpected eof", true),
        (
            "Upstream H2Error context: x cause: connection error received: not a result of an error",
            true,
        ),
        (
            "Upstream H2Error cause: stream error received: unexpected internal error encountered",
            true,
        ),
        (
            "Upstream ReadError cause: Connection reset by peer (os error 104)",
            true,
        ),
        (
            "Upstream H2Error cause: stream error received: flow-control protocol violated",
            false,
        ),
        (
            "Upstream H2Error cause: connection error detected: frame with invalid size",
            false,
        ),
        (
            "Upstream WriteTimedout context: while writing h2 request body, timeout: 60s",
            false,
        ),
        ("Upstream ReadTimedout", false),
        ("Internal InternalError", false),
        (
            "Downstream ReadError context: Peer: api.openai.com:443",
            false,
        ),
    ] {
        if peer_ended(err) != want {
            wrong.push(format!("peer_ended({err:?}) != {want}"));
        }
    }
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(wrong.join("; ").into())
    }
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
        Kind::Cat16(r, c) => cat16(name, r, c),
        Kind::Cat16NewModels(v) => cat16_new_models(name, v),
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
    common::started();
    if std::env::var("VERIFY_CATALOG_SUMMARY").as_deref() == Ok("1") {
        summary();
        return;
    }
    if std::env::var("VERIFY_CATALOG_GAPS").as_deref() == Ok("1") {
        gaps_report();
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
    if live {
        let gateway = gateway_bin().exists();
        for p in plan().0 {
            let Planned { name, kind, .. } = p;
            // The currency cells read vendor listings only; every other cell needs the gateway.
            if !gateway && !matches!(kind, Kind::Cat16(..) | Kind::Cat16NewModels(_)) {
                continue;
            }
            let n = name.clone();
            trials.push(Trial::test(name, move || common::judge(|| run(&n, &kind))));
        }
    }
    // Live traffic: no reconciliation window may be open while it runs (see common::live_traffic).
    let _traffic = (!trials.is_empty() && !args.list).then(common::live_traffic);
    trials.push(Trial::test(
        "retries_end_inside_the_nextest_budget",
        retries_end_inside_the_nextest_budget,
    ));
    trials.push(Trial::test(
        "cat3_plans_openai_prompts_to_the_measured_limit",
        cat3_plans_openai_prompts_to_the_measured_limit,
    ));
    trials.push(Trial::test(
        "provider_failures_are_attributed",
        provider_failures_are_attributed,
    ));
    let conclusion = libtest_mimic::run(&args, trials);
    cleanup(conclusion.has_failed());
    conclusion.exit();
}
