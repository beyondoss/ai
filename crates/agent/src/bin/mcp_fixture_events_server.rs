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
    streams: HashMap<u64, StreamSub>,
    next_stream: u64,
    hooks: HashMap<String, Hook>,
    verified: HashSet<(String, String)>,
    unsubscribes: Vec<Value>,
    cancelled: Vec<Value>,
    verifications: Vec<Value>,
    deliveries: Vec<Value>,
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
    let ts_secs = now_ms() / 1000 - if tamper == Some("stale") { 600 } else { 0 };
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
    state.lock().unwrap().methods.push(method.to_owned());
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
        }] })),
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
        "events/list" => Ok(json!({ "events": event_types() })),
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
            let st = state.lock().unwrap();
            let head = st.log.last().map(|o| o.seq).unwrap_or(0);
            let next_poll = env_u64("MCP_FIXTURE_NEXT_POLL_MS", 100);
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
                .filter(|o| o.seq > cursor && matches(o, &name, &project))
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
        "cursor": head.to_string(),
        "truncated": false,
    });
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
                    "methods": st.methods, "hooks": hooks, "unsubscribes": st.unsubscribes,
                    "cancelled": st.cancelled, "verifications": st.verifications,
                    "deliveries": st.deliveries, "streams": st.streams.len(), "log": st.log.len(),
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
    let head = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
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
    let Some(id) = msg.get("id").cloned() else {
        if method == "notifications/cancelled" {
            cancel_stream(&state, &params["requestId"]);
        }
        return respond(&mut stream, 202, "text/plain", b"").await;
    };
    // Stateless streamable HTTP (2026-07-28) requires `Mcp-Method` to match the body, as the
    // official Python SDK enforces — so a client that forgets it fails here, not in production.
    if req.headers.get("mcp-method").map(String::as_str) != Some(method.as_str()) {
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
    if method == "events/stream" {
        match open_stream(&state, &params, id.clone()) {
            Ok((key, mut rx)) => {
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                if stream.write_all(head.as_bytes()).await.is_err() {
                    state.lock().unwrap().streams.remove(&key);
                    return;
                }
                let terminated_end = loop {
                    let Some(frame) = rx.recv().await else {
                        break false;
                    };
                    let terminal = frame["method"] == "notifications/events/terminated";
                    let line = format!("data: {frame}\n\n");
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

async fn write_line(out: &Stdout, v: &Value) -> bool {
    let mut line = serde_json::to_vec(v).unwrap();
    line.push(b'\n');
    let mut out = out.lock().await;
    out.write_all(&line).await.is_ok() && out.flush().await.is_ok()
}

async fn run_stdio(state: Shared) {
    let out: Stdout = Arc::new(tokio::sync::Mutex::new(tokio::io::stdout()));
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
            continue; // a response to something we never asked
        }
        if method == "events/stream" {
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
    let state: Shared = Arc::new(Mutex::new(State::default()));
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
        run_stdio(state).await;
    } else {
        println!(
            "{}",
            json!({ "mcp": format!("{control}/mcp"), "control": control })
        );
        let _ = accept.await;
    }
}
