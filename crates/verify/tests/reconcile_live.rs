//! Billing reconciliation (claim BIL-5): the gateway's ledger against each provider's own usage
//! report.
//!
//! A trial boots a gateway (and nats-server) holding one provider's pool key, drives a fixed batch
//! of managed requests through it — non-stream and stream, a translated pairing, a ~3k-token system
//! prompt reused across turns so cache writes and reads occur, and (OpenAI) the Responses API — and
//! sums the batch's `ai.usage` rows. It then asks the provider's organization usage API, with the
//! admin key, for the same minutes, filtered to the pool key's id and the batch's model, and polls
//! until the provider's totals stop changing. The trial passes iff both sides agree exactly on
//! uncached input, cache reads, cache writes and output tokens (and, where the provider reports it,
//! the request count).
//!
//! Usage reports lag the traffic by minutes, so a trial takes up to ~15 minutes. The admin keys
//! touch only read-only endpoints (key listings and usage reports).
//!
//! Trials are listed only with `VERIFY_LIVE=1` *and* both of the provider's keys set (admin and
//! pool): a missing key means the trial is not listed, never that it fails.
//!
//! `long_live.rs` includes this file as a module to reconcile its long sessions the same way.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/live.rs"]
pub(crate) mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use common::free_port;
use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::{Value, json};

/// The dev signing key (seed `[7; 32]`, kid 1) and the tenant-1 token minted from it; the same
/// constants `mise run ai:mint-dev-key` prints (and `live.rs` uses).
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// How long to wait for the provider's report to settle on the batch, and how often to ask.
const SETTLE_BUDGET: Duration = Duration::from_secs(15 * 60);
const POLL_EVERY: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Provider {
    OpenAi,
    Anthropic,
}

/// One provider's reconciliation: which keys, which catalog row, which batch.
#[derive(Clone, Copy)]
struct Recon {
    provider: Provider,
    /// The `provider` every billing row must name, and the `[pool_keys]` entry.
    name: &'static str,
    pool_var: &'static str,
    admin_var: &'static str,
    /// The catalog row reconciled; [`Window`] keeps everyone else's traffic out of its key, model
    /// and minutes. Its first candidate is this provider.
    model: &'static str,
    /// The batch: `(endpoint, stream)` in order.
    batch: &'static [(Endpoint, bool)],
}

#[derive(Clone, Copy, Debug)]
enum Endpoint {
    Chat,
    Messages,
    Responses,
}

const OPENAI: Recon = Recon {
    provider: Provider::OpenAi,
    name: "openai",
    pool_var: "OPENAI_API_KEY",
    admin_var: "OPENAI_ADMIN_KEY",
    // Not a reasoning model, so the batch's tokens are the same on every run, and its Chat
    // primary has a Responses arm. gpt-4.1-nano, the batch's row until then, retires 2026-10-23;
    // gpt-4.1-mini is the same family (same caching), and no other live suite pins it.
    model: "gpt-4.1-mini",
    batch: &[
        // Native Chat, three turns on one system prompt (OpenAI caches a ≥1024-token prefix).
        (Endpoint::Chat, false),
        (Endpoint::Chat, true),
        (Endpoint::Chat, false),
        (Endpoint::Responses, false),
        (Endpoint::Responses, true),
        // Translated: Anthropic Messages on a GPT row.
        (Endpoint::Messages, false),
        (Endpoint::Messages, true),
    ],
};

/// The models BIL-5's batches reconcile on. `long_live.rs` checks its own reconciled sessions
/// against these: reconciled trials run at once, so no two may share a key and model.
#[allow(dead_code)] // read by long_live.rs, which includes this file
pub(crate) const MODELS: [&str; 2] = [OPENAI.model, ANTHROPIC.model];

const ANTHROPIC: Recon = Recon {
    provider: Provider::Anthropic,
    name: "anthropic",
    pool_var: "ANTHROPIC_API_KEY",
    admin_var: "ANTHROPIC_ADMIN_KEY",
    // Its minimum cacheable prompt is below the batch's system prompt (measured 2026-10-01: a
    // 1,400-token prompt was written), so writes and reads occur. claude-sonnet-4-5, the batch's
    // row until then, retires 2026-11-30. pi's long session reconciles `claude-sonnet-5` in the
    // same phase; [`model_matches`] keeps the two reports apart.
    model: "claude-sonnet-5-5",
    batch: &[
        // Native Messages with a cache_control breakpoint: a write, then reads.
        (Endpoint::Messages, false),
        (Endpoint::Messages, true),
        (Endpoint::Messages, false),
        // Translated: Chat Completions on a Claude row (the gateway adds the breakpoints).
        (Endpoint::Chat, false),
        (Endpoint::Chat, true),
        (Endpoint::Chat, false),
    ],
};

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

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

fn main() {
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") && gateway_bin().exists() {
        let keys = env_keys();
        for recon in [OPENAI, ANTHROPIC] {
            let (Some(pool), Some(admin)) = (keys.get(recon.pool_var), keys.get(recon.admin_var))
            else {
                continue;
            };
            let (pool, admin) = (pool.clone(), admin.clone());
            trials.push(Trial::test(
                format!("BIL-5::raw::{}::reconcile", recon.name),
                move || common::retrying(|| reconcile(recon, &pool, &admin)),
            ));
        }
    }
    libtest_mimic::run(&args, trials).exit();
}

/// Token totals in one normalized shape: `fresh` is uncached input on both wires.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Totals {
    pub(crate) fresh: u64,
    pub(crate) cache_read: u64,
    pub(crate) cache_write: u64,
    pub(crate) output: u64,
    /// Request count, where the side reports one (Anthropic's usage report doesn't).
    pub(crate) requests: Option<u64>,
}

impl Totals {
    /// Equal on every token class, and on requests where both sides count them.
    pub(crate) fn agrees(&self, other: &Totals) -> bool {
        self.fresh == other.fresh
            && self.cache_read == other.cache_read
            && self.cache_write == other.cache_write
            && self.output == other.output
            && match (self.requests, other.requests) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            }
    }
}

impl std::fmt::Display for Totals {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "fresh_input={} cache_read={} cache_write={} output={}",
            self.fresh, self.cache_read, self.cache_write, self.output
        )?;
        if let Some(n) = self.requests {
            write!(f, " requests={n}")?;
        }
        Ok(())
    }
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// The minutes a reconciled batch or session is compared over, with no other live traffic on the
/// host inside them (see [`common::isolated`]).
///
/// [`Window::open`] waits for every other live process to finish, then for the next whole minute,
/// so no earlier request shares the report's first bucket. [`Window::close`] holds the isolation
/// until the window's end (a minute past the one the last request landed in), so no later
/// request shares its last.
pub(crate) struct Window {
    _isolation: common::Isolation,
    pub(crate) start: u64,
}

impl Window {
    pub(crate) fn open() -> Window {
        let isolation = common::isolated();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let start = (now / 60.0).floor() as u64 * 60 + 60;
        std::thread::sleep(Duration::from_secs_f64(start as f64 - now));
        Window {
            _isolation: isolation,
            start,
        }
    }

    /// `(start, end)` in unix seconds, once the window is over.
    pub(crate) fn close(self) -> (u64, u64) {
        let end = now_secs().div_ceil(60) * 60 + 60;
        while now_secs() < end {
            std::thread::sleep(Duration::from_secs(end - now_secs()));
        }
        (self.start, end)
    }
}

fn reconcile(recon: Recon, pool: &str, admin: &str) -> Result<(), Failed> {
    // Find the pool key's id first: no point spending on a batch we can't reconcile.
    let key_id = pool_key_id(recon.provider, pool, admin)?;

    let dir = repo_root().join(format!(
        "target/verify-reconcile-{}-{}",
        recon.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let gw = Gateway::boot(&dir, recon.name, pool)?;

    // The window is whole minutes (the report's bucket width) around the batch.
    let window = Window::open();
    let nonce = format!("{}-{}", std::process::id(), now_secs());
    let system = system_prompt(&nonce);
    let mut ids = Vec::new();
    for (turn, &(endpoint, stream)) in recon.batch.iter().enumerate() {
        let question = QUESTIONS[turn % QUESTIONS.len()];
        ids.push(gw.send(&dir, recon.model, endpoint, stream, &system, question, turn)?);
    }

    // Every request has exactly one row; rows can land a beat after the response.
    let deadline = Instant::now() + Duration::from_secs(10);
    let rows = loop {
        let rows = usage_rows(&gw.log);
        let all = ids
            .iter()
            .all(|id| rows.iter().any(|r| r["request_id"] == id.as_str()));
        if all || Instant::now() >= deadline {
            break rows;
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    drop(gw);
    let (start, end) = window.close();
    let ledger = ledger_totals(&rows, &ids, recon)?;
    eprintln!(
        "BIL-5 {}: ledger for {} requests on {} (key {key_id}, window {start}..{end}): {ledger}",
        recon.name,
        ids.len(),
        recon.model
    );

    // Poll until the provider's totals match and hold for a second poll, or the budget runs out.
    let deadline = Instant::now() + SETTLE_BUDGET;
    let mut last: Option<Totals> = None;
    loop {
        let reported = match recon.provider {
            Provider::OpenAi => openai_usage(admin, &key_id, recon.model, start, end),
            Provider::Anthropic => anthropic_usage(admin, &key_id, recon.model, start, end),
        };
        match reported {
            Ok(t) => {
                eprintln!("BIL-5 {}: provider reports {t}", recon.name);
                if t.agrees(&ledger) && last == Some(t) {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Ok(());
                }
                last = Some(t);
            }
            Err(e) => eprintln!(
                "BIL-5 {}: usage report failed: {}",
                recon.name,
                e.message().unwrap_or("")
            ),
        }
        if Instant::now() + POLL_EVERY > deadline {
            break;
        }
        std::thread::sleep(POLL_EVERY);
    }
    Err(format!(
        "BIL-5 {}: the provider's usage report never settled on the ledger within {}s\n  \
         ledger:   {ledger}\n  provider: {}\n  (key {key_id}, model {}, window {start}..{end}; \
         rows kept in {})",
        recon.name,
        SETTLE_BUDGET.as_secs(),
        last.map_or("no report".to_owned(), |t| t.to_string()),
        recon.model,
        dir.display(),
    )
    .into())
}

/// Sum the batch's rows, normalized: an `anthropic`-wire row's input excludes the cache, an
/// OpenAI-wire row's includes reads and writes.
fn ledger_totals(rows: &[Value], ids: &[String], recon: Recon) -> Result<Totals, Failed> {
    let mut t = Totals {
        requests: Some(0),
        ..Totals::default()
    };
    let mut problems = Vec::new();
    for id in ids {
        let matching: Vec<&Value> = rows
            .iter()
            .filter(|r| r["request_id"] == id.as_str())
            .collect();
        let [row] = matching.as_slice() else {
            problems.push(format!("{id}: {} ai.usage rows, want 1", matching.len()));
            continue;
        };
        if row["provider"] != recon.name {
            problems.push(format!(
                "{id}: served by {}, want {}",
                row["provider"], recon.name
            ));
        }
        if row["requested_model"] != recon.model {
            problems.push(format!("{id}: row is for {}", row["requested_model"]));
        }
        if row["usage_estimated"] == true {
            problems.push(format!("{id}: row is an estimate ({row})"));
        }
        let n = |k: &str| row[k].as_u64().unwrap_or(0);
        let (input, read, write) = (
            n("input_tokens"),
            n("cache_read_tokens"),
            n("cache_write_tokens"),
        );
        let fresh = match row["usage_wire"].as_str() {
            Some("anthropic") => input,
            Some(_) => input.checked_sub(read + write).unwrap_or_else(|| {
                problems.push(format!("{id}: input below its cache ({row})"));
                0
            }),
            None => {
                problems.push(format!("{id}: row has no usage_wire ({row})"));
                0
            }
        };
        t.fresh += fresh;
        t.cache_read += read;
        t.cache_write += write;
        t.output += n("output_tokens");
        t.requests = t.requests.map(|r| r + 1);
    }
    if problems.is_empty() {
        Ok(t)
    } else {
        Err(format!("ledger rows are unusable:\n  {}", problems.join("\n  ")).into())
    }
}

/// A ~3k-token system prompt, the run's nonce first so the batch writes its own cache entry rather
/// than reading one an earlier run left.
fn system_prompt(nonce: &str) -> String {
    let mut s = format!(
        "Reconciliation run {nonce}. You are a terse assistant. Answer in one short sentence.\n\n"
    );
    for i in 0..200 {
        s.push_str(&format!(
            "Rule {i}: a billing ledger records every token exactly once, and rule {i} is no \
             exception to that.\n"
        ));
    }
    s
}

const QUESTIONS: &[&str] = &[
    "What is a ledger? One sentence.",
    "Name one prime number above ten.",
    "What colour is the sky on a clear day?",
    "Give one word that rhymes with token.",
];

/// A gateway and its nats-server, killed on drop.
struct Gateway {
    _nats: Guard,
    _gw: Guard,
    port: u16,
    log: PathBuf,
}

/// Kills its process on drop, so a failing trial never leaks a gateway or nats-server.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Gateway {
    fn boot(dir: &Path, provider: &str, pool: &str) -> Result<Gateway, Failed> {
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
        let cfg = format!(
            "listen = \"127.0.0.1:{port}\"\nmetrics_listen = \"127.0.0.1:{metrics_port}\"\n\
             nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\n\
             upstream_tls = true\n\n[pool_keys]\n{provider} = [{pool:?}]\n\n\
             [signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"
        );
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
        })
    }

    /// One managed request; returns its `x-beyond-request-id`.
    #[allow(clippy::too_many_arguments)]
    fn send(
        &self,
        dir: &Path,
        model: &str,
        endpoint: Endpoint,
        stream: bool,
        system: &str,
        question: &str,
        turn: usize,
    ) -> Result<String, Failed> {
        let (path, body, auth): (&str, Value, Vec<String>) = match endpoint {
            Endpoint::Chat => (
                "/v1/chat/completions",
                json!({
                    "model": model, "max_tokens": 64, "stream": stream,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                }),
                vec![format!("authorization: Bearer {DEV_TOKEN}")],
            ),
            Endpoint::Responses => (
                "/v1/responses",
                json!({
                    "model": model, "max_output_tokens": 64, "stream": stream,
                    "instructions": system, "input": question,
                }),
                vec![format!("authorization: Bearer {DEV_TOKEN}")],
            ),
            Endpoint::Messages => (
                "/v1/messages",
                json!({
                    "model": model, "max_tokens": 64, "stream": stream,
                    "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": question}],
                }),
                vec![
                    format!("x-api-key: {DEV_TOKEN}"),
                    "anthropic-version: 2023-06-01".to_owned(),
                ],
            ),
        };
        let headers = dir.join(format!("headers-{turn}"));
        let mut cmd = Command::new("curl");
        cmd.args([
            "-sS",
            "-N",
            "--max-time",
            "180",
            "-o",
            "-",
            "-w",
            "\n%{http_code}",
        ])
        .arg("-D")
        .arg(&headers)
        .args([
            "-H",
            "content-type: application/json",
            "--data-binary",
            "@-",
        ]);
        for h in &auth {
            cmd.args(["-H", h]);
        }
        cmd.arg(format!("http://127.0.0.1:{}{path}", self.port));
        let (status, out) = run_curl(cmd, &body.to_string())?;
        let what = format!("{endpoint:?} stream={stream} turn {turn}");
        if status != 200 {
            // The gateway holds one provider: its own retryable answer here makes the batch
            // run again under the SDK policy (`common::retrying`).
            std::thread::sleep(Duration::from_millis(200));
            common::note_if_ended_unavailable(&usage_rows(&self.log));
            return Err(format!("{what}: HTTP {status}: {out}").into());
        }
        let hdrs = std::fs::read_to_string(&headers).map_err(|e| e.to_string())?;
        hdrs.lines()
            .filter_map(|l| l.split_once(':'))
            .filter(|(k, _)| k.trim().eq_ignore_ascii_case("x-beyond-request-id"))
            .map(|(_, v)| v.trim().to_owned())
            .next_back()
            .ok_or_else(|| format!("{what}: no x-beyond-request-id header").into())
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
    let start = s.len().saturating_sub(2000);
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

/// Run curl with `stdin` on its standard input; the last line of its output is the HTTP status
/// (`-w "\n%{http_code}"`).
fn run_curl(mut cmd: Command, stdin: &str) -> Result<(u16, String), Failed> {
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("curl: {e}"))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .map_err(|e| format!("curl stdin: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let (body, code) = stdout.rsplit_once('\n').unwrap_or(("", &stdout));
    let status = code.trim().parse().unwrap_or(0);
    if status == 0 {
        return Err(format!("curl failed: {}", String::from_utf8_lossy(&out.stderr)).into());
    }
    Ok((status, body.to_owned()))
}

/// GET a provider admin endpoint. The admin key goes to curl on stdin (a `-K -` config), never on
/// its command line.
fn admin_get(provider: Provider, admin: &str, url: &str) -> Result<Value, Failed> {
    let config = match provider {
        Provider::OpenAi => format!("header = \"authorization: Bearer {admin}\"\n"),
        Provider::Anthropic => {
            format!("header = \"x-api-key: {admin}\"\nheader = \"anthropic-version: 2023-06-01\"\n")
        }
    };
    let mut cmd = Command::new("curl");
    cmd.args([
        "-sS",
        "--max-time",
        "60",
        "-K",
        "-",
        "-w",
        "\n%{http_code}",
        "-g",
    ])
    .arg(url);
    let (status, body) = run_curl(cmd, &config)?;
    if status != 200 {
        return Err(format!("GET {url}: HTTP {status}: {body}").into());
    }
    serde_json::from_str(&body).map_err(|e| format!("GET {url}: {e}: {body}").into())
}

/// Whether `key` is the key a redacted hint (`sk-ant-api03-ABC...wxyz`, `sk-proj-****wxyz`)
/// describes: the visible prefix and suffix both match.
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

/// The pool key's id in the provider's usage report, found by listing the org's keys (read-only)
/// and matching the redacted hint.
pub(crate) fn pool_key_id(provider: Provider, pool: &str, admin: &str) -> Result<String, Failed> {
    let mut found = BTreeSet::new();
    match provider {
        Provider::Anthropic => {
            let mut after: Option<String> = None;
            loop {
                let mut url =
                    "https://api.anthropic.com/v1/organizations/api_keys?limit=100".to_owned();
                if let Some(a) = &after {
                    url.push_str(&format!("&after_id={a}"));
                }
                let page = admin_get(provider, admin, &url)?;
                for k in page["data"].as_array().into_iter().flatten() {
                    if hint_matches(k["partial_key_hint"].as_str().unwrap_or(""), pool)
                        && let Some(id) = k["id"].as_str()
                    {
                        found.insert(id.to_owned());
                    }
                }
                match (page["has_more"].as_bool(), page["last_id"].as_str()) {
                    (Some(true), Some(last)) => after = Some(last.to_owned()),
                    _ => break,
                }
            }
        }
        Provider::OpenAi => {
            let projects = admin_get(
                provider,
                admin,
                "https://api.openai.com/v1/organization/projects?limit=100",
            )?;
            for p in projects["data"].as_array().into_iter().flatten() {
                let Some(pid) = p["id"].as_str() else {
                    continue;
                };
                let mut after: Option<String> = None;
                loop {
                    let mut url = format!(
                        "https://api.openai.com/v1/organization/projects/{pid}/api_keys?limit=100"
                    );
                    if let Some(a) = &after {
                        url.push_str(&format!("&after={a}"));
                    }
                    let page = admin_get(provider, admin, &url)?;
                    for k in page["data"].as_array().into_iter().flatten() {
                        if hint_matches(k["redacted_value"].as_str().unwrap_or(""), pool)
                            && let Some(id) = k["id"].as_str()
                        {
                            found.insert(id.to_owned());
                        }
                    }
                    match (page["has_more"].as_bool(), page["last_id"].as_str()) {
                        (Some(true), Some(last)) => after = Some(last.to_owned()),
                        _ => break,
                    }
                }
            }
        }
    }
    match found.len() {
        1 => Ok(found.pop_first().unwrap()),
        0 => Err("no API key in the organization matches the pool key's hint".into()),
        _ => Err(format!("the pool key's hint matches several keys: {found:?}").into()),
    }
}

/// The report's model may be dated (`gpt-4.1-mini-2025-04-14`, `claude-sonnet-4-5-20250929`):
/// the catalog row's upstream id followed by a date, and nothing else. Any other suffix is another
/// model: `claude-sonnet-5` must not take `claude-sonnet-5-5`'s tokens (both reconcile in the
/// same isolated phase), nor `gpt-4.1` take `gpt-4.1-mini`'s.
fn model_matches(reported: &str, model: &str) -> bool {
    let date = |d: &str| {
        let b = d.as_bytes();
        let digits = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
        match b.len() {
            8 => digits(0..8),
            10 => digits(0..4) && b[4] == b'-' && digits(5..7) && b[7] == b'-' && digits(8..10),
            _ => false,
        }
    };
    reported == model
        || reported
            .strip_prefix(model)
            .and_then(|rest| rest.strip_prefix('-'))
            .is_some_and(date)
}

/// OpenAI's completions usage (Chat Completions and Responses both report here). `input_tokens`
/// includes cached and cache-written tokens; `input_uncached_tokens` is what's left.
pub(crate) fn openai_usage(
    admin: &str,
    key_id: &str,
    model: &str,
    start: u64,
    end: u64,
) -> Result<Totals, Failed> {
    let mut t = Totals {
        requests: Some(0),
        ..Totals::default()
    };
    let mut page: Option<String> = None;
    loop {
        let mut url = format!(
            "https://api.openai.com/v1/organization/usage/completions?start_time={start}\
             &end_time={end}&bucket_width=1m&limit=1440&group_by=model&group_by=api_key_id\
             &api_key_ids={key_id}"
        );
        if let Some(p) = &page {
            url.push_str(&format!("&page={p}"));
        }
        let v = admin_get(Provider::OpenAi, admin, &url)?;
        for bucket in v["data"].as_array().into_iter().flatten() {
            for r in bucket["results"].as_array().into_iter().flatten() {
                if r["api_key_id"] != key_id
                    || !model_matches(r["model"].as_str().unwrap_or(""), model)
                {
                    continue;
                }
                let n = |k: &str| r[k].as_u64().unwrap_or(0);
                let write = n("input_cache_write_tokens");
                let uncached = match r["input_uncached_tokens"].as_u64() {
                    Some(u) => u,
                    None => n("input_tokens").saturating_sub(n("input_cached_tokens") + write),
                };
                t.fresh += uncached;
                t.cache_read += n("input_cached_tokens");
                t.cache_write += write;
                t.output += n("output_tokens");
                t.requests = t.requests.map(|x| x + n("num_model_requests"));
            }
        }
        match (v["has_more"].as_bool(), v["next_page"].as_str()) {
            (Some(true), Some(next)) => page = Some(next.to_owned()),
            _ => break,
        }
    }
    Ok(t)
}

/// Anthropic's messages usage report: `uncached_input_tokens`, `cache_read_input_tokens`, and
/// cache writes split by TTL under `cache_creation` (summed: the ledger's `cache_write_tokens`
/// counts both, with the 1-hour ones a subset in `cache_write_1h_tokens`). No request count.
pub(crate) fn anthropic_usage(
    admin: &str,
    key_id: &str,
    model: &str,
    start: u64,
    end: u64,
) -> Result<Totals, Failed> {
    let mut t = Totals::default();
    let mut page: Option<String> = None;
    loop {
        let mut url = format!(
            "https://api.anthropic.com/v1/organizations/usage_report/messages?starting_at={}\
             &ending_at={}&bucket_width=1m&limit=1440&group_by[]=model&group_by[]=api_key_id\
             &api_key_ids[]={key_id}",
            rfc3339(start),
            rfc3339(end)
        );
        if let Some(p) = &page {
            url.push_str(&format!("&page={p}"));
        }
        let v = admin_get(Provider::Anthropic, admin, &url)?;
        for bucket in v["data"].as_array().into_iter().flatten() {
            for r in bucket["results"].as_array().into_iter().flatten() {
                if r["api_key_id"] != key_id
                    || !model_matches(r["model"].as_str().unwrap_or(""), model)
                {
                    continue;
                }
                let n = |v: &Value| v.as_u64().unwrap_or(0);
                t.fresh += n(&r["uncached_input_tokens"]);
                t.cache_read += n(&r["cache_read_input_tokens"]);
                t.cache_write += n(&r["cache_creation"]["ephemeral_5m_input_tokens"])
                    + n(&r["cache_creation"]["ephemeral_1h_input_tokens"]);
                t.output += n(&r["output_tokens"]);
            }
        }
        match (v["has_more"].as_bool(), v["next_page"].as_str()) {
            (Some(true), Some(next)) => page = Some(next.to_owned()),
            _ => break,
        }
    }
    Ok(t)
}

/// Unix seconds as `YYYY-MM-DDTHH:MM:SSZ` (civil-from-days, proleptic Gregorian).
fn rfc3339(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}
