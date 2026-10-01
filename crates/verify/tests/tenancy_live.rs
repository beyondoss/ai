//! Tenancy cells (TEN-1, TEN-2): real SDK sessions through a gateway built from this commit while
//! the control plane changes under them, and two tenants side by side.
//!
//! Each trial is named `CLAIMS::client::route::scenario`, boots its own nats-server and gateway
//! with the real pool keys from `.env`, mints `bai_v2` keys for two tenants (the dev signing key,
//! seed `[7; 32]`, kid 1), and runs one scenario of `verify/clients/py/tenancy.py`. The scenario
//! drives the clients and the timing; this side owns NATS and the gateway's log. It asks over its
//! stdout and is answered on its stdin:
//!
//! - `KV put <key> <value>` / `KV del <key>`: write the `ai-gateway` bucket (`blackhole.*`,
//!   `allowance.*`), answered `OK` once JetStream acknowledged it. A bare NATS client over
//!   `std::net`: a publish to `$KV.ai-gateway.<key>`, a delete is the same with a `KV-Operation: DEL`
//!   header.
//! - `ROWS <n>`: wait until the log holds n `ai.usage` rows, answered `OK <count>`.
//! - `REFUSED <status>`: how many requests the gateway refused with that status, `OK <count>`.
//!
//! The scenario's verdict is the first witness. The second is the ledger: every call it made
//! has exactly one row carrying the tenant and key it used, the tokens it was shown, and the cache
//! hit and provider it observed; a refused call bills nothing; and no row is unaccounted for. A
//! Claude Code session's calls aren't visible to the scenario, so its rows are checked in
//! aggregate.
//!
//! Listed only with `VERIFY_LIVE=1` and the route's pool keys set.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::{Signer as _, SigningKey};
use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::{Value, json};

/// The dev signing key's public half (seed `[7; 32]`, kid 1), as `mise run ai:mint-dev-key` prints.
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const TENANT_A: u64 = 7001;
const TENANT_B: u64 = 7002;

#[derive(Clone, Copy)]
struct Route {
    name: &'static str,
    model: &'static str,
    pools: &'static [(&'static str, &'static str)],
}

const CLAUDE: Route = Route {
    name: "claude",
    model: "claude-haiku-4-5",
    pools: &[("anthropic", "ANTHROPIC_API_KEY")],
};
const GPT4O_MINI: Route = Route {
    name: "gpt4o-mini",
    model: "gpt-4o-mini",
    pools: &[("openai", "OPENAI_API_KEY")],
};
/// The Claude row with every candidate keyed, so a session pin has a real choice to make. Bedrock
/// is keyed too: the ranker's every-8th probe promotes the first *unmeasured* candidate, keyed or
/// not, so with Bedrock unkeyed it probes Bedrock, skips it, and OpenRouter is never measured or
/// pinned (D119).
const POOLED: Route = Route {
    name: "pooled3",
    model: "claude-haiku-4-5",
    pools: &[
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("bedrock", "AWS_BEARER_TOKEN_BEDROCK"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
};

/// `(claims, client, route, scenario, extra gateway config)`.
type Cell = (
    &'static str,
    &'static str,
    Route,
    &'static str,
    &'static str,
);

const CACHE: &str = "cache_ttl_secs = 300\n";
const SLOTS: &str = "tenant_max_in_flight = 2\n";
const RATE: &str = "rate_limit_rps = 3\n";

#[rustfmt::skip]
const CELLS: &[Cell] = &[
    // TEN-1: the control plane cuts a tenant off while a stream is in flight.
    ("TEN-1+A2+SEC-17", "openai-py",    GPT4O_MINI, "revoke_tenant",       ""),
    ("TEN-1+A2+SEC-17", "anthropic-py", CLAUDE,     "revoke_tenant",       ""),
    ("TEN-1+A2+SEC-17", "openai-py",    GPT4O_MINI, "revoke_key",          ""),
    ("TEN-1+A2+SEC-17", "anthropic-py", CLAUDE,     "revoke_key",          ""),
    ("TEN-1+A2",        "openai-py",    GPT4O_MINI, "exhaust_allowance",   ""),
    ("TEN-1+A2",        "anthropic-py", CLAUDE,     "exhaust_allowance",   ""),
    ("TEN-1+SEC-17",    "claude-code",  CLAUDE,     "claude_code_revoked", ""),
    // TEN-2: two tenants, identical traffic.
    ("TEN-2+SEC-14+K2", "openai-py",    GPT4O_MINI, "cache_isolation",     CACHE),
    ("TEN-2+SEC-14",    "anthropic-py", CLAUDE,     "cache_isolation",     CACHE),
    ("TEN-2+SEC-15",    "anthropic-py", POOLED,     "pin_isolation",       ""),
    ("TEN-2+SEC-15",    "openai-py",    POOLED,     "pin_isolation",       ""),
    ("TEN-2+SEC-15",    "openai-py",    GPT4O_MINI, "tenant_limit",        SLOTS),
    ("TEN-2+SEC-15",    "anthropic-py", CLAUDE,     "tenant_limit",        SLOTS),
    ("TEN-2+SEC-15",    "openai-py",    GPT4O_MINI, "rate_limit",          RATE),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn python() -> PathBuf {
    repo_root().join("verify/clients/py/.venv/bin/python")
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

/// `.env` at the repo root, then the process environment (which wins).
fn env_keys() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Ok(s) = std::fs::read_to_string(repo_root().join(".env")) {
        for line in s.lines().map(str::trim).filter(|l| !l.starts_with('#')) {
            if let Some((k, v)) = line.split_once('=') {
                out.insert(k.trim().to_owned(), v.trim().trim_matches('"').to_owned());
            }
        }
    }
    out.extend(std::env::vars());
    // No Bedrock key in the environment: mint a short-term one from the AWS credential chain.
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

fn installed(client: &str) -> bool {
    python().exists()
        && (client != "claude-code"
            || repo_root()
                .join("verify/clients/node/node_modules/.bin/claude")
                .exists())
}

fn main() {
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") {
        let keys = env_keys();
        for &(claims, client, route, scenario, extra) in CELLS {
            let have = route
                .pools
                .iter()
                .all(|(_, var)| keys.get(*var).is_some_and(|v| !v.is_empty()));
            if !have || !installed(client) || !gateway_bin().exists() {
                continue;
            }
            let name = format!("{claims}::{client}::{}::{scenario}", route.name);
            let keys = keys.clone();
            trials.push(Trial::test(name, move || {
                run_cell(client, route, scenario, extra, &keys)
            }));
        }
    }
    libtest_mimic::run(&args, trials).exit();
}

// --- keys ----------------------------------------------------------------------------------------

/// A `bai_v2` token: `bai_v2.{kid}.b64url(tenant || vpc || key_id, LE u64s).b64url(sig)`, the
/// layout `crates/gateway/src/key.rs` documents for the control plane.
fn mint_v2(tenant: u64, vpc: u64, key_id: u64) -> String {
    let sk = SigningKey::from_bytes(&[7; 32]);
    let mut payload = Vec::with_capacity(24);
    for v in [tenant, vpc, key_id] {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let signed = format!("bai_v2.1.{}", URL_SAFE_NO_PAD.encode(&payload));
    let sig = sk.sign(signed.as_bytes());
    format!("{signed}.{}", URL_SAFE_NO_PAD.encode(sig.to_bytes()))
}

/// Tenant A's `a1`, `a2` and tenant B's `b1` (distinct key ids, since the deny-set's key grain is
/// one global id space); and `paNN` / `pbNN`, the same vpc and key id under each tenant, so a pin
/// that ignored the tenant would show.
fn session_keys() -> Value {
    let mut m = serde_json::Map::new();
    let mut add = |name: String, tenant: u64, key_id: u64| {
        m.insert(
            name,
            json!({"token": mint_v2(tenant, 1, key_id), "tenant": tenant, "key_id": key_id}),
        );
    };
    add("a1".into(), TENANT_A, 101);
    add("a2".into(), TENANT_A, 102);
    add("b1".into(), TENANT_B, 201);
    for i in 1..=20u64 {
        add(format!("pa{i:02}"), TENANT_A, 300 + i);
        add(format!("pb{i:02}"), TENANT_B, 300 + i);
    }
    Value::Object(m)
}

// --- NATS ----------------------------------------------------------------------------------------

/// Put (`Some(value)`) or delete (`None`) one key of the `ai-gateway` bucket, and wait for
/// JetStream's acknowledgement. The gateway created the bucket before it reported ready.
fn kv_write(port: u16, key: &str, value: Option<&str>) -> Result<(), String> {
    let mut s =
        TcpStream::connect(("127.0.0.1", port)).map_err(|e| format!("nats connect: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    let mut r = BufReader::new(s.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    r.read_line(&mut line)
        .map_err(|e| format!("nats INFO: {e}"))?;
    if !line.starts_with("INFO") {
        return Err(format!("nats said {line:?}, want INFO"));
    }
    let subject = format!("$KV.ai-gateway.{key}");
    let mut msg = String::from(
        "CONNECT {\"verbose\":false,\"pedantic\":false,\"headers\":true,\"no_responders\":true,\"protocol\":1}\r\n\
         SUB _INBOX.verify 1\r\n",
    );
    match value {
        Some(v) => msg.push_str(&format!(
            "PUB {subject} _INBOX.verify {}\r\n{v}\r\n",
            v.len()
        )),
        None => {
            let hdr = "NATS/1.0\r\nKV-Operation: DEL\r\n\r\n";
            msg.push_str(&format!(
                "HPUB {subject} _INBOX.verify {0} {0}\r\n{hdr}\r\n",
                hdr.len()
            ));
        }
    }
    msg.push_str("PING\r\n");
    s.write_all(msg.as_bytes()).map_err(|e| e.to_string())?;
    loop {
        line.clear();
        if r.read_line(&mut line)
            .map_err(|e| format!("nats read: {e}"))?
            == 0
        {
            return Err("nats closed the connection before the ack".into());
        }
        let head = line.trim_end();
        if head == "PING" {
            s.write_all(b"PONG\r\n").map_err(|e| e.to_string())?;
        } else if head.starts_with("-ERR") {
            return Err(format!("nats: {head}"));
        } else if head.starts_with("MSG ") || head.starts_with("HMSG ") {
            let n: usize = head
                .rsplit(' ')
                .next()
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| format!("bad nats frame {head:?}"))?;
            let mut body = vec![0; n + 2];
            r.read_exact(&mut body).map_err(|e| e.to_string())?;
            let body = String::from_utf8_lossy(&body[..n]).into_owned();
            // A JetStream ack is `{"stream":"KV_ai-gateway","seq":N}`; anything else (an error
            // object, a 503 no-responders status) means the write did not land.
            return if body.contains("\"seq\"") && !body.contains("\"error\"") {
                Ok(())
            } else {
                Err(format!("nats did not ack {key}: {body}"))
            };
        }
    }
}

// --- process plumbing ----------------------------------------------------------------------------

/// A free port below the kernel's ephemeral range, as `live.rs` picks them: Pingora binds with
/// `SO_REUSEPORT`, so a `bind(0)` port another session's gateway also picked is shared silently.
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

/// Kills its process on drop, so a failing cell never leaks a gateway, nats-server or client.
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
        if let Ok(mut s) = TcpStream::connect(("127.0.0.1", metrics_port)) {
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

fn log_lines(log: &Path) -> Vec<Value> {
    let Ok(f) = std::fs::File::open(log) else {
        return Vec::new();
    };
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .collect()
}

fn usage_rows(log: &Path) -> Vec<Value> {
    log_lines(log)
        .into_iter()
        .filter(|v| v["target"] == "ai.usage")
        .map(|v| v.get("fields").cloned().unwrap_or(v))
        .collect()
}

fn refused(log: &Path, status: u64) -> usize {
    log_lines(log)
        .iter()
        .filter(|v| v["fields"]["message"] == "request rejected" && v["fields"]["status"] == status)
        .count()
}

// --- a cell --------------------------------------------------------------------------------------

fn run_cell(
    client: &str,
    route: Route,
    scenario: &str,
    extra: &str,
    keys: &BTreeMap<String, String>,
) -> Result<(), Failed> {
    let dir = repo_root().join(format!(
        "target/verify-tenancy/{}-{}",
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
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\nupstream_tls = true\n{extra}\n[pool_keys]\n"
    );
    for (provider, var) in route.pools {
        cfg.push_str(&format!("{provider} = [{:?}]\n", keys[*var]));
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

    let ids = session_keys();
    let mut child = Guard(
        Command::new(python())
            .arg(repo_root().join("verify/clients/py/tenancy.py"))
            .arg(scenario)
            .env("VERIFY_BASE", format!("http://127.0.0.1:{port}"))
            .env("VERIFY_MODEL", route.model)
            .env("VERIFY_CLIENT", client)
            .env("VERIFY_KEYS", ids.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("tenancy.py: {e}"))?,
    );
    let mut to_child = child.0.stdin.take().unwrap();
    let from_child = BufReader::new(child.0.stdout.take().unwrap());
    let mut verdict = None;
    let mut transcript = Vec::new();
    for line in from_child.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let words: Vec<&str> = line.splitn(4, ' ').collect();
        let answer = match words.as_slice() {
            ["KV", "put", key, value] => {
                Some(kv_write(nats_port, key, Some(value)).map(|()| "OK".to_owned()))
            }
            ["KV", "del", key] => Some(kv_write(nats_port, key, None).map(|()| "OK".to_owned())),
            ["ROWS", n] => {
                let n: usize = n.parse().unwrap_or(1);
                let deadline = Instant::now() + Duration::from_secs(120);
                let mut k = usage_rows(&log_path).len();
                while k < n && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(50));
                    k = usage_rows(&log_path).len();
                }
                Some(Ok(format!("OK {k}")))
            }
            ["REFUSED", status] => Some(Ok(format!(
                "OK {}",
                refused(&log_path, status.parse().unwrap_or(0))
            ))),
            _ => None,
        };
        if let Some(answer) = answer {
            transcript.push(line.clone());
            let answer = answer.unwrap_or_else(|e| format!("ERR {e}"));
            writeln!(to_child, "{answer}").map_err(|e| e.to_string())?;
            to_child.flush().map_err(|e| e.to_string())?;
        } else if let Some(j) = line.strip_prefix("VERIFY ") {
            verdict = serde_json::from_str::<Value>(j).ok();
        } else {
            eprintln!("{line}");
        }
    }
    let _ = child.0.wait();
    let verdict = verdict.ok_or("tenancy.py printed no VERIFY line")?;
    // The evidence, pass or fail (shown with --nocapture / nextest --no-capture).
    eprintln!("{scenario} detail: {}", verdict["detail"]);

    if verdict["ok"] != true {
        return Err(format!(
            "client verdict failed: {}\ncontrol: {transcript:?}\n--- gateway log ---\n{}",
            verdict["detail"],
            tail(&log_path)
        )
        .into());
    }

    // Rows can land a beat after the response.
    std::thread::sleep(Duration::from_millis(500));
    let rows = usage_rows(&log_path);
    let problems = if client == "claude-code" {
        session_problems(&rows, &ids["a1"])
    } else {
        ledger_problems(&verdict["calls"], &rows, &ids)
    };
    export_rows(&rows);
    if problems.is_empty() {
        let _ = std::fs::remove_dir_all(&dir);
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

fn n(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}

/// Input as the client counts it: the Anthropic wire reports cache reads and writes apart.
fn row_input(row: &Value) -> u64 {
    n(&row["input_tokens"])
        + if row["usage_wire"] == "anthropic" {
            n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
        } else {
            0
        }
}

/// Every call against the ledger, then every row against the calls.
fn ledger_problems(calls: &Value, rows: &[Value], ids: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let calls = calls.as_array().cloned().unwrap_or_default();
    for call in &calls {
        let key = &ids[call["key"].as_str().unwrap_or("")];
        let expect = &call["expect"];
        let Some(id) = call["request_id"].as_str() else {
            if expect["refused"] != true {
                problems.push(format!(
                    "a served call carried no x-beyond-request-id: {call}"
                ));
            }
            continue;
        };
        let matching: Vec<&Value> = rows.iter().filter(|r| r["request_id"] == id).collect();
        if expect["refused"] == true {
            // A refusal never reached a provider: no row, or one that bills nothing.
            for row in &matching {
                let billed = row_input(row) + n(&row["output_tokens"]);
                if billed != 0 || row["outcome"] == "ok" {
                    problems.push(format!(
                        "{id}: a refused call billed {billed} tokens ({row})"
                    ));
                }
            }
            continue;
        }
        let [row] = matching.as_slice() else {
            problems.push(format!(
                "{id}: {} ai.usage rows, want exactly 1",
                matching.len()
            ));
            continue;
        };
        if row["tenant_id"] != key["tenant"] || row["key_id"] != key["key_id"] {
            problems.push(format!(
                "{id}: billed to tenant {} key {}, the call used tenant {} key {} ({row})",
                row["tenant_id"], row["key_id"], key["tenant"], key["key_id"]
            ));
        }
        if row["usage_estimated"] == true {
            problems.push(format!(
                "{id}: row is an estimate on a completed call ({row})"
            ));
        }
        if let Some(want) = expect["cache_hit"].as_bool()
            && row["cache_hit"].as_bool() != Some(want)
        {
            problems.push(format!(
                "{id}: row cache_hit {}, the client saw {want} ({row})",
                row["cache_hit"]
            ));
        }
        if let Some(p) = expect["provider"].as_str()
            && row["provider"] != p
        {
            problems.push(format!(
                "{id}: served by {}, the client saw {p} ({row})",
                row["provider"]
            ));
        }
        let u = &call["usage"];
        if u.is_null() {
            problems.push(format!(
                "{id}: the client was shown no usage on a served call"
            ));
            continue;
        }
        if row_input(row) != n(&u["input_total"]) || n(&row["output_tokens"]) != n(&u["output"]) {
            problems.push(format!(
                "{id}: client saw {} in / {} out, row bills {} / {} ({row})",
                u["input_total"],
                u["output"],
                row_input(row),
                row["output_tokens"]
            ));
        }
    }
    let known: Vec<&str> = calls
        .iter()
        .filter_map(|c| c["request_id"].as_str())
        .collect();
    for row in rows {
        if !row["request_id"]
            .as_str()
            .is_some_and(|id| known.contains(&id))
        {
            problems.push(format!("a row no client call accounts for ({row})"));
        }
    }
    problems
}

/// A harness session: every row is the revoked key's, none an estimate, output was billed.
fn session_problems(rows: &[Value], key: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    if rows.is_empty() {
        problems.push("the session billed nothing".to_owned());
    }
    for row in rows {
        if row["tenant_id"] != key["tenant"] || row["key_id"] != key["key_id"] {
            problems.push(format!("row billed to another tenant or key ({row})"));
        }
        if row["usage_estimated"] == true {
            problems.push(format!("row is an estimate ({row})"));
        }
    }
    if rows.iter().map(|r| n(&r["output_tokens"])).sum::<u64>() == 0 {
        problems.push("no output tokens billed across the session".to_owned());
    }
    problems
}

/// With `VERIFY_ROWS_OUT=<file>`, append the cell's billing rows there as JSON lines.
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
