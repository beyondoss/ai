//! A hand-rolled MCP server implementing the **server** side of the MCP Events draft extension — a
//! test fixture, not a product. It exists so `crates/agent/tests/mcp_events_*.rs` can drive the real
//! `tools::mcp_events` client inside a real `serve` against a real process speaking the real wire:
//! `events/list`, `events/poll`, `events/stream` (push, with heartbeats), and webhook
//! `events/subscribe`/`events/unsubscribe` with Standard Webhooks signing, the verification
//! challenge, TTL grants, dual-signing after a secret rotation, and `terminated` envelopes.
//!
//! Dependency-free beyond what the crate already has (no `rmcp` server machinery).
//!
//! Transports: streamable HTTP by default (`/mcp` on a loopback port — POST JSON-RPC; `events/stream`
//! answers with an SSE response that stays open), or `--stdio` (newline-delimited JSON-RPC; push
//! notifications interleave on stdout). Either way a **control** API on the same/its own loopback
//! port lets a test emit events and read back what the server observed:
//!
//! - `POST /control/emit {name, data, project?, event_id?, tamper?, sign_with_generation?}` — append
//!   to the log, push to matching streams, POST to matching live webhooks; answers with each
//!   delivery's HTTP status. `tamper: "signature"|"stale"` corrupts the signature / backdates the
//!   timestamp; `sign_with_generation: n` signs with the n-th secret that subscription ever sent.
//! - `POST /control/redeliver {event_id}` — send an already-emitted occurrence again (same id).
//! - `POST /control/terminate {}` — end every subscription (`notifications/events/terminated`, or a
//!   signed `terminated` envelope).
//! - `GET /control/state` — methods seen, webhook subscriptions (with refresh counts and every
//!   secret seen), unsubscribes, cancellations, verifications, deliveries, open streams.
//!
//! The control address (`http://127.0.0.1:<port>`) is written to `$MCP_FIXTURE_CONTROL_FILE` and,
//! in HTTP mode, printed as the first stdout line together with the MCP URL.
//!
//! Env: `MCP_FIXTURE_ALLOW_HTTP_CALLBACK=1` (accept `http://` callback URLs — the draft requires
//! https), `MCP_FIXTURE_MAX_TTL_MS` (cap on granted webhook TTLs), `MCP_FIXTURE_HEARTBEAT_MS`,
//! `MCP_FIXTURE_NEXT_POLL_MS`, `MCP_FIXTURE_NO_EVENTS=1` (behave as a server without the extension).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

type HmacSha256 = Hmac<Sha256>;
type Shared = Arc<Mutex<State>>;
type Stdout = Arc<tokio::sync::Mutex<tokio::io::Stdout>>;

#[derive(Clone)]
struct Occ {
    seq: u64,
    event_id: String,
    name: String,
    timestamp: String,
    data: Value,
    project: Option<String>,
}

struct StreamSub {
    name: String,
    project: Option<String>,
    request_id: Value,
    tx: mpsc::UnboundedSender<Value>,
}

struct Hook {
    id: String,
    principal: String,
    url: String,
    name: String,
    project: Option<String>,
    secrets: Vec<Vec<u8>>,
    secret_strings: Vec<String>,
    expires_at_ms: Option<i64>,
    refreshes: u32,
    ttl_granted_ms: Option<i64>,
}

#[derive(Default)]
struct State {
    log: Vec<Occ>,
    methods: Vec<String>,
    /// The subset of `methods` whose client advertised MCP Apps (`io.modelcontextprotocol/ui`) —
    /// how a test tells an apps-flavored connection's requests from a plain one's.
    ui_methods: Vec<String>,
    streams: HashMap<u64, StreamSub>,
    next_stream: u64,
    hooks: HashMap<String, Hook>,
    verified: HashSet<(String, String)>,
    unsubscribes: Vec<Value>,
    cancelled: Vec<Value>,
    verifications: Vec<Value>,
    deliveries: Vec<Value>,
    /// Every `events/*` request seen over HTTP, with its params.
    requests: Vec<Value>,
    /// Legacy mode: requests refused for lacking the session id.
    sessionless_rejections: u64,
    /// `/control/truncate_next`: the next poll answers `truncated: true`.
    truncate_next: bool,
    /// `/control/jwks {status}` (initially `MCP_FIXTURE_JWKS_STATUS`, else 200 when a key is
    /// configured and 404 when not): what the JWKS endpoint answers.
    jwks_status: Option<u16>,
    /// `/control/events_down {down}` (initially `MCP_FIXTURE_EVENTS_DOWN=1`): every `events/*`
    /// request fails with an internal error — a server that is briefly broken.
    events_down: bool,
    /// `/control/jwks {stall: true}`: the JWKS endpoint sends its headers and then stalls.
    jwks_stall: bool,
    /// How many JWKS requests are stalled right now (they hold their connection open).
    jwks_stalled: u64,
    /// `MCP_FIXTURE_NESTED_DURING=poll|stream`: the client's answers to the nested
    /// `elicitation/create` raised during that `events/*` request.
    nested_answers: Vec<Value>,
    /// `MCP_FIXTURE_NESTED_ON_SIGNAL=1`: the nested request is held until `POST
    /// /control/raise_nested` sets this — so a test raises it once its client is attached, rather
    /// than guessing a delay.
    nested_released: bool,
    /// `MCP_FIXTURE_FORBID_FIRST=<n>`: the first `n` `events/poll`/`events/subscribe` requests are
    /// refused with `-32012` (forbidden) — credentials that are refreshed and then work.
    forbid_left: u64,
    /// Answers the client POSTed to requests this server raised over HTTP (`id` and no `method`).
    client_answers: Vec<Value>,
    /// Answer POSTs being handled right now, and the most at once (`MCP_FIXTURE_ANSWER_DELAY_MS`
    /// holds each a while, so concurrent ones overlap).
    answers_in_flight: u64,
    answers_max_in_flight: u64,
    /// Whether the HTTP nested request (`MCP_FIXTURE_NESTED_DURING`) has been raised.
    http_nested_done: bool,
}

/// The nested `elicitation/create` this server raises toward the client during an `events/*`
/// request (`MCP_FIXTURE_NESTED_DURING`).
fn nested_request(during: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": format!("nested-{during}"),
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": format!("Approve during events/{during}?"),
            "requestedSchema": { "type": "object", "properties": { "ok": { "type": "boolean" } } },
        },
    })
}

/// Over HTTP: wait for the client's answer to the nested request, and record it (or a timeout).
async fn await_http_answer(state: &Shared, during: &str) {
    let id = json!(format!("nested-{during}"));
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let answer = loop {
        let found = state
            .lock()
            .unwrap()
            .client_answers
            .iter()
            .find(|a| a["id"] == id)
            .cloned();
        if let Some(a) = found {
            break a;
        }
        if std::time::Instant::now() > deadline {
            break json!({ "timeout": true });
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    state
        .lock()
        .unwrap()
        .nested_answers
        .push(json!({ "during": during, "answer": answer }));
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn iso(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60,
        ms.rem_euclid(1000)
    )
}

fn event_types() -> Value {
    json!([
        {
            "name": "ticket.updated",
            "description": "A ticket in a project changed.",
            "delivery": ["webhook", "push", "poll"],
            "inputSchema": {
                "type": "object",
                "properties": { "project": { "type": "string" } }
            },
            "payloadSchema": {
                "type": "object",
                "properties": { "ticket_id": { "type": "string" }, "summary": { "type": "string" } }
            }
        },
        {
            "name": "build.finished",
            "description": "A CI build finished.",
            "delivery": ["poll"],
            "inputSchema": { "type": "object", "properties": {} },
            "payloadSchema": { "type": "object" }
        }
    ])
}

fn known_event(name: &str) -> bool {
    event_types()
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["name"] == name)
}

fn offers(name: &str, mode: &str) -> bool {
    event_types()
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == name)
        .and_then(|e| e["delivery"].as_array())
        .is_some_and(|d| d.iter().any(|m| m == mode))
}

fn occ_json(o: &Occ, cursor: bool) -> Value {
    let mut v = json!({
        "eventId": o.event_id,
        "name": o.name,
        "timestamp": o.timestamp,
        "data": o.data,
    });
    if cursor {
        v["cursor"] = json!(o.seq.to_string());
    }
    v
}

fn matches(o: &Occ, name: &str, project: &Option<String>) -> bool {
    o.name == name && (project.is_none() || project == &o.project)
}

/// Whether `seq` is replayed for a client at `cursor`. With `MCP_FIXTURE_REPLAY_INCLUSIVE=1` the
/// event *at* the cursor is sent again too — legal at-least-once behaviour that a client's
/// `eventId` dedup has to absorb.
fn after(seq: u64, cursor: u64) -> bool {
    if env_flag("MCP_FIXTURE_REPLAY_INCLUSIVE") {
        seq >= cursor
    } else {
        seq > cursor
    }
}

fn project_of(params: &Value) -> Option<String> {
    params
        .pointer("/arguments/project")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn canonical_args(params: &Value) -> String {
    let mut pairs: Vec<(String, String)> = params
        .get("arguments")
        .and_then(Value::as_object)
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.to_string())).collect())
        .unwrap_or_default();
    pairs.sort();
    format!("{pairs:?}")
}

type RpcErr = (i64, String, Option<Value>);

/// Every `2026-07-28` server (the official Python SDK included) attaches `resultType` and
/// `_meta.serverInfo` to every result. Mirrored here because that is exactly what broke rmcp's
/// decoding of custom results (see `mcp_events::rescue_line`) — a fixture without it hid the bug.
fn like_a_2026_server(mut result: Value) -> Value {
    // A legacy (pre-2026) server predates per-result `serverInfo`.
    if env_flag("MCP_FIXTURE_LEGACY_SESSION") {
        return result;
    }
    if let Value::Object(m) = &mut result {
        m.entry("resultType").or_insert(json!("complete"));
        m.entry("_meta").or_insert(json!({
            "io.modelcontextprotocol/serverInfo": { "name": "mcp-fixture-events-server", "version": "0.0.0" }
        }));
    }
    result
}

fn err(code: i64, msg: &str) -> RpcErr {
    (code, msg.to_owned(), None)
}

fn sign(secret: &[u8], id: &str, ts: &str, body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).unwrap();
    mac.update(format!("{id}.{ts}.").as_bytes());
    mac.update(body);
    base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
}

/// The server's Ed25519 webhook-signing key, when `MCP_FIXTURE_ED25519_SEED` (64 hex chars) is
/// set: published at `/.well-known/mcp-webhook-jwks.json`, and used to add a Standard Webhooks
/// `v1a,` signature to every delivery.
fn server_key() -> Option<ed25519_dalek::SigningKey> {
    let seed = hex::decode(std::env::var("MCP_FIXTURE_ED25519_SEED").ok()?).ok()?;
    Some(ed25519_dalek::SigningKey::from_bytes(
        &<[u8; 32]>::try_from(seed.as_slice()).ok()?,
    ))
}

fn jwks() -> Value {
    match server_key() {
        Some(k) => json!({ "keys": [{
            "kty": "OKP", "crv": "Ed25519", "kid": "fixture-1", "use": "sig",
            "x": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(k.verifying_key().as_bytes()),
        }] }),
        None => Value::Null,
    }
}

fn decode_secret(s: &str) -> Option<Vec<u8>> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(s.strip_prefix("whsec_")?)
        .ok()?;
    (24..=64).contains(&raw.len()).then_some(raw)
}

/// A minimal HTTP/1.1 POST. Returns `(status, body)`; status 0 when the connection failed.
async fn http_post(url: &str, headers: &[(String, String)], body: &[u8]) -> (u16, Vec<u8>) {
    let Some(rest) = url.strip_prefix("http://") else {
        return (0, Vec::new());
    };
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let Ok(Ok(mut stream)) =
        tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(hostport)).await
    else {
        return (0, Vec::new());
    };
    let mut req = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    if stream.write_all(req.as_bytes()).await.is_err() || stream.write_all(body).await.is_err() {
        return (0, Vec::new());
    }
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut buf)).await;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.as_bytes().to_vec())
        .unwrap_or_default();
    (status, body)
}

/// A minimal HTTP/1.1 GET. Returns `(status, body)`; status 0 when the connection failed.
async fn http_get(url: &str) -> (u16, Vec<u8>) {
    let Some(rest) = url.strip_prefix("http://") else {
        return (0, Vec::new());
    };
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let Ok(Ok(mut stream)) =
        tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(hostport)).await
    else {
        return (0, Vec::new());
    };
    let req = format!("GET {path} HTTP/1.1\r\nHost: {hostport}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).await.is_err() {
        return (0, Vec::new());
    }
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut buf)).await;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.as_bytes().to_vec())
        .unwrap_or_default();
    (status, body)
}

/// POST a signed body to a webhook subscriber. `generations` picks which of the subscription's
/// secrets sign it (Standard Webhooks multi-signature).
async fn post_signed(
    url: &str,
    sub_id: &str,
    msg_id: &str,
    secrets: &[Vec<u8>],
    body: &Value,
    tamper: Option<&str>,
) -> (u16, Vec<u8>) {
    let body = serde_json::to_vec(body).unwrap();
    let ts_secs = now_ms() / 1000
        + match tamper {
            Some("stale") => -600,
            Some("future") => 600,
            _ => 0,
        };
    let ts = ts_secs.to_string();
    let sigs: Vec<String> = secrets
        .iter()
        .map(|s| {
            let mut sig = sign(s, msg_id, &ts, &body);
            if tamper == Some("signature") {
                sig = sign(
                    b"definitely-not-the-subscription-secret",
                    msg_id,
                    &ts,
                    &body,
                );
            }
            format!("v1,{sig}")
        })
        .collect();
    let mut sigs = sigs;
    if let Some(key) = server_key()
        && tamper != Some("no_v1a")
    {
        use ed25519_dalek::Signer;
        let mut msg = format!("{msg_id}.{ts}.").into_bytes();
        msg.extend_from_slice(&body);
        if tamper == Some("v1a") {
            msg.extend_from_slice(b"tampered");
        }
        sigs.push(format!(
            "v1a,{}",
            base64::engine::general_purpose::STANDARD.encode(key.sign(&msg).to_bytes())
        ));
    }
    let headers = vec![
        ("webhook-id".to_owned(), msg_id.to_owned()),
        ("webhook-timestamp".to_owned(), ts),
        ("webhook-signature".to_owned(), sigs.join(" ")),
        ("X-MCP-Subscription-Id".to_owned(), sub_id.to_owned()),
    ];
    http_post(url, &headers, &body).await
}

/// The secrets a delivery is signed with: the current one, plus the previous one (dual-signing
/// for a grace window after a rotation, as the draft recommends) — unless a test picks.
fn signing_secrets(hook: &Hook, generation: Option<usize>) -> Vec<Vec<u8>> {
    match generation {
        Some(g) => hook.secrets.get(g).cloned().into_iter().collect(),
        None => hook.secrets.iter().rev().take(2).cloned().collect(),
    }
}

async fn rpc(
    state: &Shared,
    method: &str,
    params: Value,
    principal: Option<String>,
) -> Result<Value, RpcErr> {
    let (events_down, forbidden) = {
        let mut st = state.lock().unwrap();
        st.methods.push(method.to_owned());
        let forbidden = matches!(method, "events/poll" | "events/subscribe") && st.forbid_left > 0;
        if forbidden {
            st.forbid_left -= 1;
        }
        if params
            .pointer("/_meta/io.modelcontextprotocol~1clientCapabilities/extensions/io.modelcontextprotocol~1ui")
            .is_some()
        {
            st.ui_methods.push(method.to_owned());
        }
        (st.events_down, forbidden)
    };
    let no_events = env_flag("MCP_FIXTURE_NO_EVENTS");
    let capabilities = if no_events {
        json!({ "tools": {} })
    } else {
        json!({ "tools": {}, "events": { "listChanged": true } })
    };
    match method {
        "server/discover" => Ok(json!({
            "resultType": "complete",
            "supportedVersions": ["2026-07-28", "2025-11-25"],
            "capabilities": capabilities,
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": { "io.modelcontextprotocol/serverInfo": { "name": "mcp-fixture-events-server", "version": "0.0.0" } }
        })),
        "initialize" => Ok(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": capabilities,
            "serverInfo": { "name": "mcp-fixture-events-server", "version": "0.0.0" }
        })),
        "tools/list" => Ok(json!({ "tools": [{
            "name": "echo",
            "description": "Echoes back its `text` argument.",
            "inputSchema": { "type": "object", "properties": { "text": { "type": "string" } } }
        }, {
            "name": "blob",
            "description": "Returns a text result of `bytes` bytes.",
            "inputSchema": { "type": "object", "properties": { "bytes": { "type": "integer" } } }
        }] })),
        "tools/call" if params["name"] == "blob" => {
            let n = params
                .pointer("/arguments/bytes")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            Ok(json!({ "content": [{ "type": "text", "text": "x".repeat(n) }], "isError": false }))
        }
        "tools/call" => {
            let text = params
                .pointer("/arguments/text")
                .and_then(Value::as_str)
                .unwrap_or("");
            Ok(json!({ "content": [{ "type": "text", "text": text }], "isError": false }))
        }
        m if m.starts_with("events/") && no_events => {
            Err(err(-32601, &format!("Method not found: {m}")))
        }
        m if m.starts_with("events/") && events_down => {
            Err(err(-32603, &format!("temporarily unavailable: {m}")))
        }
        m if forbidden => Err(err(-32012, &format!("forbidden: {m}"))),
        "events/list" => {
            // `MCP_FIXTURE_LIST_DELAY_MS`: a slow server, for concurrency tests.
            let delay = env_u64("MCP_FIXTURE_LIST_DELAY_MS", 0);
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            // `MCP_FIXTURE_LIST_PAD_BYTES`: a discovery answer far larger than any client should
            // read whole.
            let pad = env_u64("MCP_FIXTURE_LIST_PAD_BYTES", 0) as usize;
            if pad > 0 {
                return Ok(json!({ "events": event_types(), "pad": "x".repeat(pad) }));
            }
            Ok(json!({ "events": event_types() }))
        }
        "events/poll" => {
            let name = params["name"].as_str().unwrap_or("").to_owned();
            if !known_event(&name) {
                return Err((-32011, "NotFound".into(), Some(json!({"kind": "event"}))));
            }
            if !offers(&name, "poll") {
                return Err((
                    -32014,
                    "Unsupported".into(),
                    Some(json!({"feature": "deliveryMode", "value": "poll"})),
                ));
            }
            let project = project_of(&params);
            let max = params["maxEvents"].as_u64().unwrap_or(100) as usize;
            let mut st = state.lock().unwrap();
            let head = st.log.last().map(|o| o.seq).unwrap_or(0);
            let next_poll = env_u64("MCP_FIXTURE_NEXT_POLL_MS", 100);
            // `MCP_FIXTURE_HASMORE_EMPTY=1`: a broken server that always claims more and sends none.
            if env_flag("MCP_FIXTURE_HASMORE_EMPTY") {
                return Ok(
                    json!({ "events": [], "cursor": head.to_string(), "truncated": false, "hasMore": true, "nextPollMs": next_poll }),
                );
            }
            // `MCP_FIXTURE_HASMORE_DUPS=1`: a broken server that always claims more and sends the
            // same (already delivered) event again — a page of nothing but duplicates.
            if env_flag("MCP_FIXTURE_HASMORE_DUPS") {
                return Ok(json!({
                    "events": [{ "eventId": "dup-forever", "name": "ticket.updated", "timestamp": "2026-01-01T00:00:00Z", "data": {}, "cursor": "1" }],
                    "cursor": head.to_string(), "truncated": false, "hasMore": true, "nextPollMs": next_poll,
                }));
            }
            if std::mem::take(&mut st.truncate_next) {
                return Ok(
                    json!({ "events": [], "cursor": head.to_string(), "truncated": true, "hasMore": false, "nextPollMs": next_poll }),
                );
            }
            let Some(cursor) = params["cursor"]
                .as_str()
                .and_then(|c| c.parse::<u64>().ok())
            else {
                return Ok(
                    json!({ "events": [], "cursor": head.to_string(), "truncated": false, "hasMore": false, "nextPollMs": next_poll }),
                );
            };
            let pending: Vec<&Occ> = st
                .log
                .iter()
                .filter(|o| after(o.seq, cursor) && matches(o, &name, &project))
                .collect();
            let batch: Vec<&Occ> = pending.iter().take(max).copied().collect();
            let new_cursor = if pending.len() > max {
                batch.last().map(|o| o.seq).unwrap_or(cursor)
            } else {
                head
            };
            Ok(json!({
                "events": batch.iter().map(|o| occ_json(o, false)).collect::<Vec<_>>(),
                "cursor": new_cursor.to_string(),
                "truncated": false,
                "hasMore": pending.len() > max,
                "nextPollMs": next_poll,
            }))
        }
        "events/subscribe" => subscribe(state, params, principal).await,
        "events/unsubscribe" => {
            let Some(principal) = principal else {
                return Err((
                    -32012,
                    "Forbidden".into(),
                    Some(json!({"reason": "webhook requires an authenticated principal"})),
                ));
            };
            let url = params
                .pointer("/delivery/url")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let key = format!(
                "{principal}|{url}|{}|{}",
                params["name"].as_str().unwrap_or(""),
                canonical_args(&params)
            );
            let mut st = state.lock().unwrap();
            st.unsubscribes.push(params.clone());
            if st.hooks.remove(&key).is_none() {
                return Err((
                    -32011,
                    "NotFound".into(),
                    Some(json!({"kind": "subscription"})),
                ));
            }
            Ok(json!({}))
        }
        other => Err(err(-32601, &format!("Method not found: {other}"))),
    }
}

async fn subscribe(
    state: &Shared,
    params: Value,
    principal: Option<String>,
) -> Result<Value, RpcErr> {
    let Some(principal) = principal else {
        return Err((
            -32012,
            "Forbidden".into(),
            Some(json!({"reason": "webhook requires an authenticated principal"})),
        ));
    };
    let name = params["name"].as_str().unwrap_or("").to_owned();
    if !known_event(&name) {
        return Err((-32011, "NotFound".into(), Some(json!({"kind": "event"}))));
    }
    if !offers(&name, "webhook") {
        return Err((
            -32014,
            "Unsupported".into(),
            Some(json!({"feature": "deliveryMode", "value": "webhook"})),
        ));
    }
    let url = params
        .pointer("/delivery/url")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let https = url.starts_with("https://");
    if !(https || (url.starts_with("http://") && env_flag("MCP_FIXTURE_ALLOW_HTTP_CALLBACK"))) {
        return Err(err(-32602, "delivery.url must be https"));
    }
    let secret_str = params
        .pointer("/delivery/secret")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let Some(secret) = decode_secret(&secret_str) else {
        return Err(err(
            -32602,
            "delivery.secret must be whsec_ + base64 of 24-64 bytes",
        ));
    };
    // Endpoint verification, once per (principal, url).
    let needs_verify = !state
        .lock()
        .unwrap()
        .verified
        .contains(&(principal.clone(), url.clone()));
    // `MCP_FIXTURE_VERIFY_VIA_WELL_KNOWN=1`: verify by the receiver-published document instead of
    // a challenge POST — the callback's origin serves `/.well-known/mcp-webhook-receiver.json`
    // listing path prefixes that accept deliveries.
    if needs_verify && env_flag("MCP_FIXTURE_VERIFY_VIA_WELL_KNOWN") {
        let (origin, path) = match url
            .strip_prefix("http://")
            .and_then(|r| r.find('/').map(|i| (r, i)))
        {
            Some((r, i)) => (format!("http://{}", &r[..i]), r[i..].to_owned()),
            None => (url.clone(), "/".to_owned()),
        };
        let (status, body) =
            http_get(&format!("{origin}/.well-known/mcp-webhook-receiver.json")).await;
        let covered = status == 200
            && serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|v| v["receivers"].as_array().cloned())
                .is_some_and(|r| {
                    r.iter()
                        .filter_map(Value::as_str)
                        .any(|p| path.starts_with(p))
                });
        state
            .lock()
            .unwrap()
            .verifications
            .push(json!({"url": url, "status": status, "ok": covered, "via": "well-known"}));
        if !covered {
            return Err((
                -32015,
                "CallbackEndpointError".into(),
                Some(json!({"reason": "challenge_failed"})),
            ));
        }
        state
            .lock()
            .unwrap()
            .verified
            .insert((principal.clone(), url.clone()));
    }
    let needs_verify = !state
        .lock()
        .unwrap()
        .verified
        .contains(&(principal.clone(), url.clone()));
    if needs_verify {
        let nonce = hex::encode(Sha256::digest(format!("{}{}", now_ms(), url).as_bytes()));
        let msg_id = format!("msg_verification_{}", &nonce[..12]);
        let (status, body) = post_signed(
            &url,
            "",
            &msg_id,
            std::slice::from_ref(&secret),
            &json!({"type": "verification", "challenge": nonce}),
            None,
        )
        .await;
        let echoed = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| v["challenge"].as_str().map(str::to_owned));
        let ok = (200..300).contains(&status) && echoed.as_deref() == Some(nonce.as_str());
        state
            .lock()
            .unwrap()
            .verifications
            .push(json!({"url": url, "status": status, "ok": ok}));
        if !ok {
            let reason = if status == 0 {
                "connection_refused"
            } else {
                "challenge_failed"
            };
            return Err((
                -32015,
                "CallbackEndpointError".into(),
                Some(json!({"reason": reason})),
            ));
        }
        state
            .lock()
            .unwrap()
            .verified
            .insert((principal.clone(), url.clone()));
    }
    let key = format!("{principal}|{url}|{name}|{}", canonical_args(&params));
    let key_for_replay = key.clone();
    let id = format!("sub_{}", &hex::encode(Sha256::digest(key.as_bytes()))[..16]);
    let max_ttl = env_u64("MCP_FIXTURE_MAX_TTL_MS", 3_600_000) as i64;
    let granted = match params.get("ttlMs") {
        Some(Value::Number(n)) => Some(n.as_i64().unwrap_or(max_ttl).min(max_ttl)),
        _ => Some(max_ttl),
    };
    let expires = granted.map(|t| now_ms() + t);
    let mut st = state.lock().unwrap();
    let head = st.log.last().map(|o| o.seq).unwrap_or(0);
    let live = st
        .hooks
        .get(&key)
        .is_some_and(|h| h.expires_at_ms.is_none_or(|e| e > now_ms()));
    let status = st
        .hooks
        .get(&key)
        .map(|_| json!({"active": true, "lastDeliveryAt": null, "lastError": null}));
    // A fresh subscription carrying a cursor (a client that restarted) is replayed from it: the
    // response's watermark stays at the cursor, and the backlog follows asynchronously.
    let replay_from = (!live)
        .then(|| {
            params["cursor"]
                .as_str()
                .and_then(|c| c.parse::<u64>().ok())
        })
        .flatten();
    match st.hooks.get_mut(&key) {
        Some(hook) if live => {
            hook.refreshes += 1;
            if hook.secret_strings.last() != Some(&secret_str) {
                hook.secrets.push(secret);
                hook.secret_strings.push(secret_str);
            }
            hook.expires_at_ms = expires;
            hook.ttl_granted_ms = granted;
        }
        _ => {
            st.hooks.insert(
                key,
                Hook {
                    id: id.clone(),
                    principal,
                    url,
                    name,
                    project: project_of(&params),
                    secrets: vec![secret],
                    secret_strings: vec![secret_str],
                    expires_at_ms: expires,
                    refreshes: 0,
                    ttl_granted_ms: granted,
                },
            );
        }
    }
    let mut result = json!({
        "id": id,
        "refreshBefore": expires.map(iso),
        "cursor": replay_from.unwrap_or(head).to_string(),
        "truncated": false,
    });
    if let Some(cursor) = replay_from {
        let state = state.clone();
        let key = key_for_replay.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            let backlog: Vec<(Occ, String, String, Vec<Vec<u8>>)> = {
                let st = state.lock().unwrap();
                let Some(h) = st.hooks.get(&key) else { return };
                st.log
                    .iter()
                    .filter(|o| after(o.seq, cursor) && matches(o, &h.name, &h.project))
                    .map(|o| {
                        (
                            o.clone(),
                            h.url.clone(),
                            h.id.clone(),
                            signing_secrets(h, None),
                        )
                    })
                    .collect()
            };
            for (o, url, id, secrets) in backlog {
                let (status, _) =
                    post_signed(&url, &id, &o.event_id, &secrets, &occ_json(&o, true), None).await;
                state.lock().unwrap().deliveries.push(
                    json!({"event_id": o.event_id, "url": url, "status": status, "replay": true}),
                );
            }
        });
    }
    if let Some(s) = status {
        result["deliveryStatus"] = s;
    }
    Ok(result)
}

/// Open a push stream: confirm, replay from the cursor, then live events and heartbeats.
fn open_stream(
    state: &Shared,
    params: &Value,
    request_id: Value,
) -> Result<(u64, mpsc::UnboundedReceiver<Value>), RpcErr> {
    if env_flag("MCP_FIXTURE_NO_EVENTS") {
        return Err(err(-32601, "Method not found: events/stream"));
    }
    state.lock().unwrap().methods.push("events/stream".into());
    let name = params["name"].as_str().unwrap_or("").to_owned();
    if !known_event(&name) {
        return Err((-32011, "NotFound".into(), Some(json!({"kind": "event"}))));
    }
    if !offers(&name, "push") {
        return Err((
            -32014,
            "Unsupported".into(),
            Some(json!({"feature": "deliveryMode", "value": "push"})),
        ));
    }
    let project = project_of(params);
    let (tx, rx) = mpsc::unbounded_channel();
    let mut st = state.lock().unwrap();
    let head = st.log.last().map(|o| o.seq).unwrap_or(0);
    let meta = json!({ "io.modelcontextprotocol/subscriptionId": request_id });
    let _ = tx.send(json!({"jsonrpc": "2.0", "method": "notifications/events/active", "params": {"cursor": head.to_string(), "truncated": false, "_meta": meta}}));
    if let Some(cursor) = params["cursor"]
        .as_str()
        .and_then(|c| c.parse::<u64>().ok())
    {
        for o in st
            .log
            .iter()
            .filter(|o| o.seq > cursor && matches(o, &name, &project))
        {
            let mut p = occ_json(o, true);
            p["_meta"] = meta.clone();
            let _ = tx.send(
                json!({"jsonrpc": "2.0", "method": "notifications/events/event", "params": p}),
            );
        }
    }
    st.next_stream += 1;
    let key = st.next_stream;
    st.streams.insert(
        key,
        StreamSub {
            name,
            project,
            request_id: request_id.clone(),
            tx: tx.clone(),
        },
    );
    drop(st);
    // Heartbeats carrying the head cursor.
    let hb = Duration::from_millis(env_u64("MCP_FIXTURE_HEARTBEAT_MS", 30_000));
    let state2 = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(hb).await;
            let head = {
                let st = state2.lock().unwrap();
                if !st.streams.contains_key(&key) {
                    return;
                }
                st.log.last().map(|o| o.seq).unwrap_or(0)
            };
            if tx.send(json!({"jsonrpc": "2.0", "method": "notifications/events/heartbeat", "params": {"cursor": head.to_string(), "_meta": {"io.modelcontextprotocol/subscriptionId": request_id}}})).is_err() {
                return;
            }
        }
    });
    Ok((key, rx))
}

fn cancel_stream(state: &Shared, request_id: &Value) {
    let mut st = state.lock().unwrap();
    st.cancelled.push(request_id.clone());
    st.streams.retain(|_, s| &s.request_id != request_id);
}

async fn control(state: &Shared, method: &str, path: &str, body: &[u8]) -> (u16, Value) {
    let req: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    match (method, path) {
        ("GET", "/control/state") => {
            let st = state.lock().unwrap();
            let hooks: Vec<Value> = st.hooks.values().map(|h| json!({
                "id": h.id, "principal": h.principal, "url": h.url, "name": h.name, "project": h.project,
                "refreshes": h.refreshes, "secrets": h.secret_strings, "ttl_granted_ms": h.ttl_granted_ms,
                "expired": h.expires_at_ms.is_some_and(|e| e <= now_ms()),
            })).collect();
            (
                200,
                json!({
                    "methods": st.methods, "ui_methods": st.ui_methods, "hooks": hooks, "unsubscribes": st.unsubscribes,
                    "cancelled": st.cancelled, "verifications": st.verifications,
                    "deliveries": st.deliveries, "streams": st.streams.len(), "log": st.log.len(),
                "requests": st.requests, "sessionless_rejections": st.sessionless_rejections,
                "jwks_stalled": st.jwks_stalled, "nested_answers": st.nested_answers,
                "client_answers": st.client_answers.len(), "answers_max_in_flight": st.answers_max_in_flight,
                }),
            )
        }
        ("POST", "/control/emit") => {
            let occ = {
                let mut st = state.lock().unwrap();
                let seq = st.log.len() as u64 + 1;
                let occ = Occ {
                    seq,
                    event_id: req["event_id"]
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("evt_{seq}")),
                    name: req["name"].as_str().unwrap_or("ticket.updated").to_owned(),
                    timestamp: iso(now_ms()),
                    data: req.get("data").cloned().unwrap_or(json!({})),
                    project: req["project"].as_str().map(str::to_owned),
                };
                st.log.push(occ.clone());
                occ
            };
            let deliveries = fan_out(
                state,
                &occ,
                req["tamper"].as_str(),
                req["sign_with_generation"].as_u64().map(|g| g as usize),
            )
            .await;
            (
                200,
                json!({ "event_id": occ.event_id, "deliveries": deliveries }),
            )
        }
        ("POST", "/control/redeliver") => {
            let id = req["event_id"].as_str().unwrap_or("");
            let occ = state
                .lock()
                .unwrap()
                .log
                .iter()
                .find(|o| o.event_id == id)
                .cloned();
            match occ {
                Some(o) => (
                    200,
                    json!({ "deliveries": fan_out(state, &o, None, None).await }),
                ),
                None => (404, json!({"error": "no such event"})),
            }
        }
        ("POST", "/control/raise_nested") => {
            state.lock().unwrap().nested_released = true;
            (200, json!({ "ok": true }))
        }
        ("POST", "/control/jwks") => {
            let mut st = state.lock().unwrap();
            st.jwks_status = req["status"].as_u64().map(|s| s as u16);
            st.jwks_stall = req["stall"].as_bool().unwrap_or(false);
            (200, json!({}))
        }
        ("POST", "/control/events_down") => {
            state.lock().unwrap().events_down = req["down"].as_bool().unwrap_or(true);
            (200, json!({}))
        }
        ("POST", "/control/truncate_next") => {
            state.lock().unwrap().truncate_next = true;
            (200, json!({}))
        }
        ("POST", "/control/emit_burst") => {
            // Append `count` events and push them to every stream in one go — faster than any
            // client can consume them.
            let count = req["count"].as_u64().unwrap_or(10);
            let st = &mut *state.lock().unwrap();
            let mut ids = Vec::new();
            for _ in 0..count {
                let seq = st.log.len() as u64 + 1;
                let occ = Occ {
                    seq,
                    event_id: format!("burst_{seq}"),
                    name: req["name"].as_str().unwrap_or("ticket.updated").to_owned(),
                    timestamp: iso(now_ms()),
                    data: json!({ "n": seq }),
                    project: req["project"].as_str().map(str::to_owned),
                };
                for s in st
                    .streams
                    .values()
                    .filter(|s| matches(&occ, &s.name, &s.project))
                {
                    let mut p = occ_json(&occ, true);
                    p["_meta"] = json!({ "io.modelcontextprotocol/subscriptionId": s.request_id });
                    let _ = s.tx.send(json!({"jsonrpc": "2.0", "method": "notifications/events/event", "params": p}));
                }
                ids.push(occ.event_id.clone());
                st.log.push(occ);
            }
            (200, json!({ "event_ids": ids }))
        }
        ("POST", "/control/schema_change") => {
            // The event type changed in place: end every push stream with the draft's
            // `-32014 Unsupported {reason: schema_changed}`.
            let streams: Vec<StreamSub> = state
                .lock()
                .unwrap()
                .streams
                .drain()
                .map(|(_, s)| s)
                .collect();
            let error = json!({"code": -32014, "message": "Unsupported", "data": {"feature": "payloadSchema", "reason": "schema_changed"}});
            let n = streams.len();
            for s in streams {
                let _ = s.tx.send(json!({"jsonrpc": "2.0", "method": "notifications/events/terminated", "params": {"error": error, "_meta": {"io.modelcontextprotocol/subscriptionId": s.request_id}}}));
            }
            (200, json!({ "streams": n }))
        }
        ("POST", "/control/terminate") => {
            let (streams, hooks) = {
                let mut st = state.lock().unwrap();
                let streams: Vec<StreamSub> = st.streams.drain().map(|(_, s)| s).collect();
                let hooks: Vec<Hook> = st.hooks.drain().map(|(_, h)| h).collect();
                (streams, hooks)
            };
            let error = json!({"code": -32012, "message": "Forbidden", "data": {"reason": "Access revoked"}});
            for s in streams {
                let _ = s.tx.send(json!({"jsonrpc": "2.0", "method": "notifications/events/terminated", "params": {"error": error, "_meta": {"io.modelcontextprotocol/subscriptionId": s.request_id}}}));
            }
            let mut statuses = Vec::new();
            for h in hooks {
                let (status, _) = post_signed(
                    &h.url,
                    &h.id,
                    "msg_terminated_1",
                    &signing_secrets(&h, None),
                    &json!({"type": "terminated", "error": error}),
                    None,
                )
                .await;
                statuses.push(status);
            }
            (200, json!({ "webhook_statuses": statuses }))
        }
        _ => (404, json!({"error": "unknown control route"})),
    }
}

async fn fan_out(
    state: &Shared,
    occ: &Occ,
    tamper: Option<&str>,
    generation: Option<usize>,
) -> Vec<Value> {
    let targets: Vec<(String, String, Vec<Vec<u8>>, bool)> = {
        let st = state.lock().unwrap();
        for s in st
            .streams
            .values()
            .filter(|s| matches(occ, &s.name, &s.project))
        {
            let mut p = occ_json(occ, true);
            p["_meta"] = json!({ "io.modelcontextprotocol/subscriptionId": s.request_id });
            let _ = s.tx.send(
                json!({"jsonrpc": "2.0", "method": "notifications/events/event", "params": p}),
            );
        }
        st.hooks
            .values()
            .filter(|h| matches(occ, &h.name, &h.project))
            .map(|h| {
                (
                    h.url.clone(),
                    h.id.clone(),
                    signing_secrets(h, generation),
                    h.expires_at_ms.is_some_and(|e| e <= now_ms()),
                )
            })
            .collect()
    };
    let mut out = Vec::new();
    for (url, id, secrets, expired) in targets {
        if expired {
            out.push(json!({"url": url, "status": null, "expired": true}));
            continue;
        }
        let (status, _) = post_signed(
            &url,
            &id,
            &occ.event_id,
            &secrets,
            &occ_json(occ, true),
            tamper,
        )
        .await;
        let record = json!({"event_id": occ.event_id, "url": url, "status": status, "tamper": tamper, "generation": generation});
        state.lock().unwrap().deliveries.push(record.clone());
        out.push(record);
    }
    out
}

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

async fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let head_end = loop {
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.lines();
    let mut first = lines.next()?.split_whitespace();
    let method = first.next()?.to_owned();
    let path = first.next()?.split('?').next()?.to_owned();
    let headers: HashMap<String, String> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    Some(Request {
        method,
        path,
        headers,
        body,
    })
}

async fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    respond_with(stream, status, content_type, "", body).await;
}

async fn respond_with(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    extra_headers: &str,
    body: &[u8],
) {
    let head = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.flush().await;
}

async fn handle_http(state: Shared, mut stream: TcpStream) {
    let Some(req) = read_request(&mut stream).await else {
        return;
    };
    if req.path.starts_with("/control/") {
        let (status, body) = control(&state, &req.method, &req.path, &req.body).await;
        respond(
            &mut stream,
            status,
            "application/json",
            body.to_string().as_bytes(),
        )
        .await;
        return;
    }
    if req.path == "/.well-known/mcp-webhook-jwks.json" {
        if state.lock().unwrap().jwks_stall {
            state.lock().unwrap().jwks_stalled += 1;
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{\"keys\": [")
                .await;
            let _ = stream.flush().await;
            // Never finishes the body; the client must give up on its own. Ends when the client
            // closes (a read sees EOF), so an aborted fetch is observable.
            let mut buf = [0u8; 64];
            loop {
                match tokio::time::timeout(Duration::from_secs(600), stream.read(&mut buf)).await {
                    Ok(Ok(0)) | Ok(Err(_)) | Err(_) => break,
                    Ok(Ok(_)) => {}
                }
            }
            state.lock().unwrap().jwks_stalled -= 1;
            return;
        }
        let doc = jwks();
        let forced = state.lock().unwrap().jwks_status.or_else(|| {
            std::env::var("MCP_FIXTURE_JWKS_STATUS")
                .ok()
                .and_then(|v| v.parse().ok())
        });
        let status = forced.unwrap_or(if doc.is_null() { 404 } else { 200 });
        if status != 200 || doc.is_null() {
            return respond(&mut stream, status, "text/plain", b"no keys").await;
        }
        return respond(
            &mut stream,
            200,
            "application/json",
            doc.to_string().as_bytes(),
        )
        .await;
    }
    if req.path != "/mcp" {
        respond(&mut stream, 404, "text/plain", b"not found").await;
        return;
    }
    match req.method.as_str() {
        "POST" => {}
        "DELETE" => return respond(&mut stream, 200, "text/plain", b"").await,
        _ => return respond(&mut stream, 405, "text/plain", b"").await,
    }
    let msg: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
    let method = msg["method"].as_str().unwrap_or("").to_owned();
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    // `MCP_FIXTURE_LEGACY_SESSION=1`: a pre-2026 streamable-HTTP server — no `server/discover`, an
    // `initialize` that mints an `Mcp-Session-Id`, and every later request refused without it. A
    // stateless direct request cannot reach it; only the connection's own session can.
    let legacy = env_flag("MCP_FIXTURE_LEGACY_SESSION");
    if legacy {
        if method == "initialize" {
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            let body = json!({"jsonrpc": "2.0", "id": id, "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": { "tools": {}, "events": { "listChanged": true } },
                "serverInfo": { "name": "mcp-fixture-events-server", "version": "0.0.0" }
            }});
            return respond_with(
                &mut stream,
                200,
                "application/json",
                "Mcp-Session-Id: fixture-session\r\n",
                body.to_string().as_bytes(),
            )
            .await;
        }
        if method == "server/discover" {
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            let body = json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}});
            return respond(
                &mut stream,
                200,
                "application/json",
                body.to_string().as_bytes(),
            )
            .await;
        }
        if req.headers.get("mcp-session-id").map(String::as_str) != Some("fixture-session") {
            state.lock().unwrap().sessionless_rejections += 1;
            return respond(&mut stream, 400, "text/plain", b"missing Mcp-Session-Id").await;
        }
    }
    if method.starts_with("events/") {
        state
            .lock()
            .unwrap()
            .requests
            .push(json!({ "method": method, "params": params }));
    }
    let Some(id) = msg.get("id").cloned() else {
        if method == "notifications/cancelled" {
            cancel_stream(&state, &params["requestId"]);
        }
        return respond(&mut stream, 202, "text/plain", b"").await;
    };
    // An answer to a request this server raised: kept for whoever is waiting on it. Like the
    // official Python SDK, a POST that does not accept both JSON and SSE is refused `406` (and so
    // never reaches the waiter).
    if msg.get("method").is_none() {
        let accept = req.headers.get("accept").map(String::as_str).unwrap_or("");
        if !(accept.contains("application/json") && accept.contains("text/event-stream")) {
            return respond(&mut stream, 406, "text/plain", b"Not Acceptable").await;
        }
        {
            let mut st = state.lock().unwrap();
            st.client_answers.push(msg.clone());
            st.answers_in_flight += 1;
            st.answers_max_in_flight = st.answers_max_in_flight.max(st.answers_in_flight);
        }
        let delay = env_u64("MCP_FIXTURE_ANSWER_DELAY_MS", 0);
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        state.lock().unwrap().answers_in_flight -= 1;
        return respond(&mut stream, 202, "text/plain", b"").await;
    }
    // Stateless streamable HTTP (2026-07-28) requires `Mcp-Method` to match the body, as the
    // official Python SDK enforces — so a client that forgets it fails here, not in production.
    if !legacy && req.headers.get("mcp-method").map(String::as_str) != Some(method.as_str()) {
        let body = json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32020, "message": "mcp-method header does not match the request body's method"}});
        return respond(
            &mut stream,
            400,
            "application/json",
            body.to_string().as_bytes(),
        )
        .await;
    }
    let principal = req
        .headers
        .get("authorization")
        .and_then(|a| a.strip_prefix("Bearer "))
        .map(str::to_owned);
    let nested_during = std::env::var("MCP_FIXTURE_NESTED_DURING").unwrap_or_default();
    let raise_nested = |during: &str| {
        nested_during == during
            && !std::mem::replace(&mut state.lock().unwrap().http_nested_done, true)
    };
    // `MCP_FIXTURE_NESTED_DURING=poll` over HTTP: the first `events/poll` is answered as an SSE
    // stream that raises the nested request first, and carries the result only after the client
    // has answered it (or the wait has timed out).
    // `MCP_FIXTURE_POLL_SSE_PAD_BYTES`: `events/poll` answered as an SSE stream whose one event is
    // padded far past any sane message size.
    let poll_pad = env_u64("MCP_FIXTURE_POLL_SSE_PAD_BYTES", 0) as usize;
    if method == "events/poll" && poll_pad > 0 {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(head.as_bytes()).await;
        let body = json!({"jsonrpc": "2.0", "id": id, "result": { "events": [], "cursor": "0", "pad": "x".repeat(poll_pad) }});
        let _ = stream
            .write_all(format!("data: {body}\n\n").as_bytes())
            .await;
        return;
    }
    if method == "events/poll" && raise_nested("poll") {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(head.as_bytes()).await;
        let _ = stream
            .write_all(format!("data: {}\n\n", nested_request("poll")).as_bytes())
            .await;
        let _ = stream.flush().await;
        await_http_answer(&state, "poll").await;
        let body = match rpc(&state, &method, params, principal).await {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": like_a_2026_server(result)}),
            Err((code, message, data)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}})
            }
        };
        let _ = stream
            .write_all(format!("data: {body}\n\n").as_bytes())
            .await;
        return;
    }
    if method == "events/stream" {
        let nested = raise_nested("stream");
        match open_stream(&state, &params, id.clone()) {
            Ok((key, mut rx)) => {
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                if stream.write_all(head.as_bytes()).await.is_err() {
                    state.lock().unwrap().streams.remove(&key);
                    return;
                }
                if nested {
                    let _ = stream
                        .write_all(format!("data: {}\n\n", nested_request("stream")).as_bytes())
                        .await;
                    let _ = stream.flush().await;
                    let state = state.clone();
                    tokio::spawn(async move { await_http_answer(&state, "stream").await });
                }
                // `MCP_FIXTURE_NESTED_FLOOD=<n>`: a hostile server raising `n` requests on the
                // stream at once.
                for i in 0..env_u64("MCP_FIXTURE_NESTED_FLOOD", 0) {
                    let mut request = nested_request("flood");
                    request["id"] = json!(format!("flood-{i}"));
                    let _ = stream
                        .write_all(format!("data: {request}\n\n").as_bytes())
                        .await;
                }
                let _ = stream.flush().await;
                let terminated_end = loop {
                    let Some(frame) = rx.recv().await else {
                        break false;
                    };
                    let terminal = frame["method"] == "notifications/events/terminated";
                    let line = format!("data: {}\n\n", frame_text(&frame));
                    if stream.write_all(line.as_bytes()).await.is_err()
                        || stream.flush().await.is_err()
                    {
                        break false;
                    }
                    if terminal {
                        break true;
                    }
                    if !state.lock().unwrap().streams.contains_key(&key) {
                        break false;
                    }
                };
                if terminated_end {
                    let _ = stream
                        .write_all(
                            format!(
                                "data: {}\n\n",
                                json!({"jsonrpc": "2.0", "id": id, "result": {"_meta": {}}})
                            )
                            .as_bytes(),
                        )
                        .await;
                }
                state.lock().unwrap().streams.remove(&key);
            }
            Err((code, message, data)) => {
                let body = json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}});
                respond(
                    &mut stream,
                    200,
                    "application/json",
                    body.to_string().as_bytes(),
                )
                .await;
            }
        }
        return;
    }
    let body = match rpc(&state, &method, params, principal).await {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": like_a_2026_server(result)}),
        Err((code, message, data)) => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}})
        }
    };
    respond(
        &mut stream,
        200,
        "application/json",
        body.to_string().as_bytes(),
    )
    .await;
}

/// A frame's wire text. `MCP_FIXTURE_KEY_ORDER=params_first` writes an event notification the
/// other way round from the usual key order — `params` before `method`, and the payload (`data`)
/// first inside `params` — as JSON allows: a client reading a bounded head of an over-cap one then
/// sees neither its method, its routing nor its cursor.
fn frame_text(v: &Value) -> String {
    let params_first = std::env::var("MCP_FIXTURE_KEY_ORDER").is_ok_and(|o| o == "params_first");
    let (Some(method), Some(Value::Object(params))) = (v["method"].as_str(), v.get("params"))
    else {
        return v.to_string();
    };
    if !params_first || method != "notifications/events/event" {
        return v.to_string();
    }
    let mut members: Vec<String> = Vec::new();
    if let Some(data) = params.get("data") {
        members.push(format!("\"data\":{data}"));
    }
    for (k, val) in params.iter().filter(|(k, _)| *k != "data") {
        members.push(format!("{}:{val}", Value::String(k.clone())));
    }
    format!(
        "{{\"params\":{{{}}},\"jsonrpc\":\"2.0\",\"method\":{}}}",
        members.join(","),
        Value::String(method.to_owned())
    )
}

async fn write_line(out: &Stdout, v: &Value) -> bool {
    let mut line = frame_text(v).into_bytes();
    line.push(b'\n');
    let mut out = out.lock().await;
    out.write_all(&line).await.is_ok() && out.flush().await.is_ok()
}

/// Raise a nested `elicitation/create` toward the client and record its answer (or an error) in
/// `nested_answers`. Answers come back through `waiters`, routed by `run_stdio`.
async fn nested_elicitation(state: &Shared, out: &Stdout, waiters: &Waiters, during: &str) {
    // `MCP_FIXTURE_NESTED_DELAY_MS`: hold the request (and so the `events/*` call it rides) a
    // while first, so a client can attach before it is raised.
    let delay = env_u64("MCP_FIXTURE_NESTED_DELAY_MS", 0);
    if delay > 0 {
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
    if env_flag("MCP_FIXTURE_NESTED_ON_SIGNAL") {
        while !state.lock().unwrap().nested_released {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    let id = format!("nested-{during}");
    let (tx, rx) = tokio::sync::oneshot::channel();
    waiters.lock().unwrap().insert(json!(id).to_string(), tx);
    let request = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": format!("Approve during events/{during}?"),
            "requestedSchema": {
                "type": "object",
                "properties": { "ok": { "type": "boolean" } },
            },
        },
    });
    if !write_line(out, &request).await {
        return;
    }
    let answer = match tokio::time::timeout(Duration::from_secs(30), rx).await {
        Ok(Ok(msg)) => msg,
        _ => json!({ "timeout": true }),
    };
    state
        .lock()
        .unwrap()
        .nested_answers
        .push(json!({ "during": during, "answer": answer }));
}

type Waiters = Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<Value>>>>;

async fn run_stdio(state: Shared) {
    let out: Stdout = Arc::new(tokio::sync::Mutex::new(tokio::io::stdout()));
    let waiters: Waiters = Arc::default();
    let nested_during = std::env::var("MCP_FIXTURE_NESTED_DURING").unwrap_or_default();
    let nested_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // `MCP_FIXTURE_GARBAGE_STDOUT=1`: a server that prints a line that is not UTF-8 before it
    // speaks MCP (a stray banner from a native library, say). A client must skip it, not die.
    if env_flag("MCP_FIXTURE_GARBAGE_STDOUT") {
        let mut o = out.lock().await;
        let _ = o.write_all(b"\xff\xfe\xfd not utf-8 \xc3\x28\n").await;
        let _ = o.flush().await;
    }
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let method = msg["method"].as_str().unwrap_or("").to_owned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let Some(id) = msg.get("id").cloned() else {
            if method == "notifications/cancelled" {
                cancel_stream(&state, &params["requestId"]);
            }
            continue;
        };
        if msg.get("method").is_none() {
            // An answer to a request this server raised, if it is one we are waiting for.
            if let Some(tx) = waiters.lock().unwrap().remove(&id.to_string()) {
                let _ = tx.send(msg.clone());
            }
            continue;
        }
        // `MCP_FIXTURE_NESTED_DURING=poll`: the first `events/poll` is answered only after the
        // client has answered a nested elicitation raised while it is in flight.
        if method == "events/poll"
            && nested_during == "poll"
            && !nested_done.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            let (state, out, waiters) = (state.clone(), out.clone(), waiters.clone());
            tokio::spawn(async move {
                nested_elicitation(&state, &out, &waiters, "poll").await;
                let reply = match rpc(&state, "events/poll", params, None).await {
                    Ok(result) => {
                        json!({"jsonrpc": "2.0", "id": id, "result": like_a_2026_server(result)})
                    }
                    Err((code, message, data)) => {
                        json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}})
                    }
                };
                write_line(&out, &reply).await;
            });
            continue;
        }
        if method == "events/stream" {
            if nested_during == "stream"
                && !nested_done.swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                let (state, out, waiters) = (state.clone(), out.clone(), waiters.clone());
                tokio::spawn(async move {
                    // Raised while the stream request is open (after its first notifications).
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    nested_elicitation(&state, &out, &waiters, "stream").await;
                });
            }
            match open_stream(&state, &params, id.clone()) {
                Ok((key, mut rx)) => {
                    let out = out.clone();
                    let state = state.clone();
                    tokio::spawn(async move {
                        while let Some(frame) = rx.recv().await {
                            let terminal = frame["method"] == "notifications/events/terminated";
                            if !write_line(&out, &frame).await {
                                return;
                            }
                            if terminal {
                                break;
                            }
                            if !state.lock().unwrap().streams.contains_key(&key) {
                                return;
                            }
                        }
                        state.lock().unwrap().streams.remove(&key);
                    });
                }
                Err((code, message, data)) => {
                    write_line(&out, &json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}})).await;
                }
            }
            continue;
        }
        let reply = match rpc(&state, &method, params, None).await {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": like_a_2026_server(result)}),
            Err((code, message, data)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}})
            }
        };
        if !write_line(&out, &reply).await {
            break;
        }
    }
}

#[tokio::main]
async fn main() {
    let stdio = std::env::args().any(|a| a == "--stdio");
    let state: Shared = Arc::new(Mutex::new(State {
        events_down: env_flag("MCP_FIXTURE_EVENTS_DOWN"),
        forbid_left: env_u64("MCP_FIXTURE_FORBID_FIRST", 0),
        ..State::default()
    }));
    // port-0: held for the server's life; the address is announced through the control file.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let control = format!("http://{addr}");
    if let Ok(path) = std::env::var("MCP_FIXTURE_CONTROL_FILE") {
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, &control).unwrap();
        std::fs::rename(&tmp, &path).unwrap();
    }
    let accept_state = state.clone();
    let accept = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            tokio::spawn(handle_http(accept_state.clone(), stream));
        }
    });
    if stdio {
        // `MCP_FIXTURE_ORPHAN_PIDFILE=<path>`: double-fork a grandchild away from ourselves (as a
        // browser-driving server does) and record its pid; only a process-group sweep reaches it.
        if let Ok(pidfile) = std::env::var("MCP_FIXTURE_ORPHAN_PIDFILE") {
            let _ = tokio::process::Command::new("sh")
                .arg("-c")
                // Written to a temporary name and renamed, so a test never reads a half-written pid.
                .arg(format!(
                    "sleep 600 & echo $! > {pidfile}.tmp && mv -f {pidfile}.tmp {pidfile}"
                ))
                .status()
                .await;
        }
        run_stdio(state).await;
        // `MCP_FIXTURE_EXIT_MARKER=<path>`: stdin closed — the MCP shutdown signal. Take a moment
        // to "clean up" (as a server closing a browser would), then record that we got to.
        if let Ok(path) = std::env::var("MCP_FIXTURE_EXIT_MARKER") {
            tokio::time::sleep(Duration::from_millis(500)).await;
            write_atomically(&path, "clean exit");
        }
    } else {
        println!(
            "{}",
            json!({ "mcp": format!("{control}/mcp"), "control": control })
        );
        let _ = accept.await;
    }
}

/// Write a file a test reads, all at once: to a temporary sibling, then `rename` it into place. A
/// reader polling for the file (or its content) can otherwise see it created but still empty,
/// between `write`'s create and its write.
fn write_atomically(path: &str, contents: &str) {
    let tmp = format!("{path}.tmp-{}", std::process::id());
    if std::fs::write(&tmp, contents).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}
