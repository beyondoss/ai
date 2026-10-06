//! Webhook delivery: the receiver behind `serve`'s `/_beyond/mcp-events/<token>` route, Standard
//! Webhooks signatures (`v1,` HMAC and `v1a,` Ed25519 server identity), and the per-subscription
//! subscribe/refresh/rotate/unsubscribe loop.
//!
//! **Key policy (`v1a`), fail closed.** The pinned draft lets a server add an Ed25519 `v1a,`
//! signature, with its key discovered from an origin the client already authenticates — this client
//! uses the standalone JWKS at `<MCP server origin>/.well-known/mcp-webhook-jwks.json`.
//!
//! - Every Ed25519 key ever published at an origin is remembered — in the session's state file and
//!   process-wide — and the set only grows. While it is non-empty, every delivery to a subscription
//!   on that origin (the verification challenge included) must carry a `v1a,` signature one of those
//!   keys verifies.
//! - A later `404`/`410`, an empty document, or a failed fetch never removes a key: there is no
//!   signal in the draft that could prove a server meant to stop signing, so none is trusted.
//! - On a subscription's *first* subscribe, a failed fetch (timeout, connection error, `5xx`,
//!   oversized or unparseable body) with no keys known for the origin is retried; if it still fails
//!   the subscription is refused rather than started without knowing whether to enforce identity.
//!   Only a clear `404`/`410` means "this server does not sign", and then `v1a,` is ignored.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use base64::Engine as _;
use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::Sha256;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::settings::McpEventDelivery;

use super::wire::{KeyFetch, RpcError, parse_rfc3339_ms};
use super::{
    Delivery, Hub, RPC_TIMEOUT, SubSpec, SubState, WEBHOOK_PATH_PREFIX, backoff, now_unix,
    now_unix_ms, ready_ok, webhook_ttl,
};

/// Standard Webhooks' replay window, in seconds, either side of now.
const TIMESTAMP_TOLERANCE_SECS: u64 = 300;
/// Most bytes one delivery body may be. The draft: servers SHOULD keep bodies at or under
/// 256 KiB, and receivers MAY refuse larger ones with `413`.
pub const MAX_WEBHOOK_BODY: usize = 1024 * 1024;
/// How long a delivery waits for its event to be made durable before answering `503`.
const ACK_TIMEOUT: Duration = Duration::from_secs(10);

type HmacSha256 = Hmac<Sha256>;

/// What a verified webhook POST carried, for the subscription's task, with where to send the
/// HTTP status once the task has handled (and, for an event, durably stored) it.
enum WebhookMsg {
    Event(Value, oneshot::Sender<u16>),
    Gap(Value, oneshot::Sender<u16>),
    Terminated(Value, oneshot::Sender<u16>),
}

struct Secrets {
    current: Vec<u8>,
    /// The secret before the last rotation, still accepted so an in-flight retry signed with it
    /// verifies. Exactly one generation back.
    previous: Option<Vec<u8>>,
}

/// One webhook subscription's receiving end, looked up by the token in its callback path.
struct Route {
    secrets: Mutex<Secrets>,
    /// The server-derived subscription id, once `events/subscribe` has answered. A delivery naming a
    /// different one is refused; before it is known (the verification challenge arrives *during*
    /// the subscribe request) it is not checked.
    subscription_id: Mutex<Option<String>>,
    /// The server-identity keys enforced for this subscription (see the module's key policy).
    server_keys: Mutex<Vec<ed25519_dalek::VerifyingKey>>,
    tx: mpsc::Sender<WebhookMsg>,
}

static ROUTES: LazyLock<Mutex<HashMap<String, Arc<Route>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Every key ever seen per origin, process-wide (the set only grows).
static ORIGIN_KEYS: LazyLock<Mutex<HashMap<String, Vec<String>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn routes() -> std::sync::MutexGuard<'static, HashMap<String, Arc<Route>>> {
    ROUTES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether `token` names a live subscription — the listener's cheap check before it reads a body.
pub fn route_exists(token: &str) -> bool {
    routes().contains_key(token)
}

/// CSPRNG bytes. A failure is an error, never a fallback to something predictable.
fn random_bytes<const N: usize>() -> Result<[u8; N], String> {
    let mut b = [0u8; N];
    getrandom::fill(&mut b)
        .map_err(|e| format!("no secure randomness for a webhook secret: {e}"))?;
    Ok(b)
}

/// A Standard Webhooks symmetric secret: `whsec_` + base64 of 32 CSPRNG bytes (the draft allows
/// 24–64). Returns the wire string and the raw key.
fn new_secret() -> Result<(String, Vec<u8>), String> {
    let raw = random_bytes::<32>()?;
    Ok((
        format!(
            "whsec_{}",
            base64::engine::general_purpose::STANDARD.encode(raw)
        ),
        raw.to_vec(),
    ))
}

/// The HTTP answer to one webhook POST.
pub struct WebhookReply {
    pub status: u16,
    pub reason: &'static str,
    pub body: Vec<u8>,
}

impl WebhookReply {
    fn json(status: u16, reason: &'static str, body: Value) -> Self {
        Self {
            status,
            reason,
            body: body.to_string().into_bytes(),
        }
    }

    pub(crate) fn error(status: u16, reason: &'static str, message: &str) -> Self {
        Self::json(status, reason, json!({ "error": message }))
    }
}

pub(super) fn verify_signature(
    secret: &[u8],
    msg_id: &str,
    ts: &str,
    body: &[u8],
    header: &str,
) -> bool {
    for part in header.split_whitespace() {
        let Some(b64) = part.strip_prefix("v1,") else {
            continue;
        };
        let Ok(sig) = base64::engine::general_purpose::STANDARD.decode(b64) else {
            continue;
        };
        let Ok(mut mac) = HmacSha256::new_from_slice(secret) else {
            return false;
        };
        mac.update(msg_id.as_bytes());
        mac.update(b".");
        mac.update(ts.as_bytes());
        mac.update(b".");
        mac.update(body);
        // Constant-time.
        if mac.verify_slice(&sig).is_ok() {
            return true;
        }
    }
    false
}

/// Standard Webhooks `v1a,`: an Ed25519 signature over the same `id.timestamp.body` the HMAC
/// covers, from the server's own key. Accepts if any listed `v1a,` signature verifies under any
/// known key (rotation publishes the new key beside the old).
pub(super) fn verify_server_signature(
    keys: &[ed25519_dalek::VerifyingKey],
    msg_id: &str,
    ts: &str,
    body: &[u8],
    header: &str,
) -> bool {
    let mut msg = Vec::with_capacity(msg_id.len() + ts.len() + body.len() + 2);
    msg.extend_from_slice(msg_id.as_bytes());
    msg.push(b'.');
    msg.extend_from_slice(ts.as_bytes());
    msg.push(b'.');
    msg.extend_from_slice(body);
    header
        .split_whitespace()
        .filter_map(|part| part.strip_prefix("v1a,"))
        .filter_map(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok())
        .filter_map(|raw| ed25519_dalek::Signature::from_slice(&raw).ok())
        .any(|sig| keys.iter().any(|k| k.verify_strict(&msg, &sig).is_ok()))
}

/// Parse a JWKS document's Ed25519 keys (`kty: OKP`, `crv: Ed25519`, base64url `x`).
pub(super) fn ed25519_keys_from_jwks(doc: &Value) -> Vec<ed25519_dalek::VerifyingKey> {
    doc.get("keys")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|k| k.get("kty") == Some(&json!("OKP")) && k.get("crv") == Some(&json!("Ed25519")))
        .filter_map(|k| k.get("x").and_then(Value::as_str))
        .filter_map(|x| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(x.trim_end_matches('='))
                .ok()
        })
        .filter_map(|raw| <[u8; 32]>::try_from(raw.as_slice()).ok())
        .filter_map(|raw| ed25519_dalek::VerifyingKey::from_bytes(&raw).ok())
        .collect()
}

fn key_to_b64(k: &ed25519_dalek::VerifyingKey) -> String {
    base64::engine::general_purpose::STANDARD.encode(k.as_bytes())
}

fn key_from_b64(s: &str) -> Option<ed25519_dalek::VerifyingKey> {
    let raw = base64::engine::general_purpose::STANDARD.decode(s).ok()?;
    ed25519_dalek::VerifyingKey::from_bytes(&<[u8; 32]>::try_from(raw.as_slice()).ok()?).ok()
}

/// Whether a `webhook-timestamp` is within the replay window either side of `now` — computed
/// without overflow for any `i64` an attacker can send.
pub(super) fn timestamp_fresh(ts_secs: i64, now: i64) -> bool {
    ts_secs.abs_diff(now) <= TIMESTAMP_TOLERANCE_SECS
}

/// Handle one POST to [`WEBHOOK_PATH_PREFIX`]`<token>`. Called by `serve`'s listener with the raw
/// body bytes — the signature is over them exactly as received, never over re-serialized JSON.
///
/// - unknown token → **410 Gone** (the draft's "do not retry this delivery"): the subscription is
///   over. Tokens are registered *before* `events/subscribe` is sent, so there is no window in
///   which a legitimate first delivery finds nothing.
/// - missing Standard Webhooks headers → 400; a timestamp more than five minutes off, in either
///   direction → 400; no valid `v1,` signature under the current or previous secret → 401; a
///   missing or invalid `v1a,` where the server's identity is enforced → 401; a mismatched
///   `X-MCP-Subscription-Id` → 401.
/// - `verification` → 200 echoing `{"challenge"}`.
/// - an event → **200 only once it is durably stored** with the session's state (and so will reach
///   the model, across a restart if need be — the draft's "SHOULD NOT 2xx before durably
///   persisted"); a duplicate → 200; **503** (retryable) when the session's pending queue is full
///   or the store does not answer in time.
pub async fn receive_webhook(
    token: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> WebhookReply {
    let Some(route) = routes().get(token).cloned() else {
        return WebhookReply::error(410, "Gone", "no such subscription");
    };
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    let (Some(msg_id), Some(ts), Some(sig)) = (
        header("webhook-id"),
        header("webhook-timestamp"),
        header("webhook-signature"),
    ) else {
        return WebhookReply::error(400, "Bad Request", "missing Standard Webhooks headers");
    };
    let Ok(ts_secs) = ts.trim().parse::<i64>() else {
        return WebhookReply::error(400, "Bad Request", "invalid webhook-timestamp");
    };
    if !timestamp_fresh(ts_secs, now_unix()) {
        return WebhookReply::error(400, "Bad Request", "webhook-timestamp outside tolerance");
    }
    let verified = {
        let secrets = route
            .secrets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        verify_signature(&secrets.current, msg_id, ts, body, sig)
            || secrets
                .previous
                .as_deref()
                .is_some_and(|p| verify_signature(p, msg_id, ts, body, sig))
    };
    if !verified {
        return WebhookReply::error(401, "Unauthorized", "invalid signature");
    }
    let keys = route
        .server_keys
        .lock()
        .map(|k| k.clone())
        .unwrap_or_default();
    if !keys.is_empty() && !verify_server_signature(&keys, msg_id, ts, body, sig) {
        return WebhookReply::error(
            401,
            "Unauthorized",
            "missing or invalid v1a server signature",
        );
    }
    if let Some(expected) = route.subscription_id.lock().ok().and_then(|id| id.clone())
        && header("x-mcp-subscription-id").is_some_and(|got| got != expected)
    {
        return WebhookReply::error(401, "Unauthorized", "subscription id mismatch");
    }
    let Ok(payload) = serde_json::from_slice::<Value>(body) else {
        return WebhookReply::error(400, "Bad Request", "body is not JSON");
    };
    let (ack_tx, ack_rx) = oneshot::channel();
    let msg = match payload.get("type").and_then(Value::as_str) {
        Some("verification") => {
            let challenge = payload.get("challenge").cloned().unwrap_or(Value::Null);
            return WebhookReply::json(200, "OK", json!({ "challenge": challenge }));
        }
        Some("gap") => WebhookMsg::Gap(payload, ack_tx),
        Some("terminated") => WebhookMsg::Terminated(payload, ack_tx),
        Some(other) => {
            tracing::debug!(
                r#type = other,
                "unknown webhook control envelope; acknowledged"
            );
            return WebhookReply::json(200, "OK", json!({}));
        }
        None => WebhookMsg::Event(payload, ack_tx),
    };
    if route.tx.try_send(msg).is_err() {
        return WebhookReply::error(503, "Service Unavailable", "subscription busy");
    }
    match tokio::time::timeout(ACK_TIMEOUT, ack_rx).await {
        Ok(Ok(200)) => WebhookReply::json(200, "OK", json!({})),
        Ok(Ok(410)) => WebhookReply::error(410, "Gone", "no such subscription"),
        _ => WebhookReply::error(503, "Service Unavailable", "not stored; retry later"),
    }
}

/// Removes a route from the table however its task ends.
struct RouteGuard(String);

impl Drop for RouteGuard {
    fn drop(&mut self) {
        routes().remove(&self.0);
    }
}

/// When to refresh a grant: three quarters of the way to `refreshBefore`, never sooner than
/// 500 ms. `null` (no expiry) still refreshes hourly — the draft recommends it, since the refresh is
/// where the cursor advances and `deliveryStatus` is reported. A `refreshBefore` that cannot be
/// parsed refreshes after [`UNPARSEABLE_REFRESH`] (and the caller warns), never later.
pub(super) fn refresh_delay(refresh_before: Option<&str>) -> Result<Duration, Duration> {
    let Some(raw) = refresh_before else {
        return Ok(Duration::from_secs(3600));
    };
    let Some(at_ms) = parse_rfc3339_ms(raw) else {
        return Err(UNPARSEABLE_REFRESH);
    };
    let remaining = at_ms.saturating_sub(now_unix_ms()).max(0) as u64;
    Ok(Duration::from_millis(remaining.saturating_mul(3) / 4).max(Duration::from_millis(500)))
}

/// The conservative refresh interval used when a server's `refreshBefore` cannot be parsed.
pub(super) const UNPARSEABLE_REFRESH: Duration = Duration::from_secs(30);

/// Apply the key policy for one subscribe/refresh. `Err` refuses a first subscribe whose identity
/// cannot be established.
async fn update_server_keys(
    hub: &Hub,
    conn: &super::wire::Conn,
    route: &Route,
    first: bool,
) -> Result<(), String> {
    let Some(origin) = conn.origin() else {
        return Ok(()); // stdio: no origin, no identity to enforce
    };
    let known = |hub: &Hub| -> Vec<String> {
        let mut all = hub.store.keys(&origin);
        if let Ok(global) = ORIGIN_KEYS.lock()
            && let Some(g) = global.get(&origin)
        {
            for k in g {
                if !all.contains(k) {
                    all.push(k.clone());
                }
            }
        }
        all
    };
    let mut attempts = 0;
    loop {
        attempts += 1;
        match conn.server_signing_keys().await {
            KeyFetch::Keys(keys) if !keys.is_empty() => {
                let encoded: Vec<String> = keys.iter().map(key_to_b64).collect();
                hub.store.add_keys(&origin, &encoded);
                if let Ok(mut global) = ORIGIN_KEYS.lock() {
                    let set = global.entry(origin.clone()).or_default();
                    for k in encoded {
                        if !set.contains(&k) {
                            set.push(k);
                        }
                    }
                }
                break;
            }
            KeyFetch::Keys(_) | KeyFetch::NotPublished => break,
            KeyFetch::Failed(e) => {
                if !known(hub).is_empty() || !first {
                    tracing::debug!(error = %e, "JWKS fetch failed; keeping the keys already known");
                    break;
                }
                if attempts >= 3 {
                    return Err(format!(
                        "could not fetch the server's webhook-signing keys ({e}); refusing to \
                         subscribe without knowing whether to enforce its identity"
                    ));
                }
                tokio::time::sleep(Duration::from_millis(500 * attempts)).await;
            }
        }
    }
    let keys: Vec<ed25519_dalek::VerifyingKey> =
        known(hub).iter().filter_map(|k| key_from_b64(k)).collect();
    if let Ok(mut slot) = route.server_keys.lock() {
        *slot = keys;
    }
    Ok(())
}

pub(super) async fn run_webhook(
    hub: Arc<Hub>,
    spec: Arc<SubSpec>,
    state: Arc<SubState>,
    cancel: CancellationToken,
    ready: oneshot::Sender<Result<(), String>>,
) {
    let mut ready = Some(ready);
    let Some(base) = hub.callback_url.clone() else {
        if let Some(tx) = ready.take() {
            let _ = tx.send(Err("webhook delivery is not configured".into()));
        }
        return;
    };
    let fresh = random_bytes::<16>().and_then(|t| Ok((hex::encode(t), new_secret()?)));
    let (token, (mut secret_wire, secret_raw)) = match fresh {
        Ok(v) => v,
        Err(e) => {
            if let Some(tx) = ready.take() {
                let _ = tx.send(Err(e));
            }
            return;
        }
    };
    let url = format!("{base}{WEBHOOK_PATH_PREFIX}{token}");
    let (tx, mut rx) = mpsc::channel::<WebhookMsg>(256);
    let route = Arc::new(Route {
        secrets: Mutex::new(Secrets {
            current: secret_raw,
            previous: None,
        }),
        subscription_id: Mutex::new(None),
        server_keys: Mutex::new(Vec::new()),
        tx,
    });
    routes().insert(token.clone(), route.clone());
    let _guard = RouteGuard(token);

    let ttl_ms = webhook_ttl().as_millis() as u64;
    let subscribe_params = |secret: &str, cursor: Option<String>| {
        let mut p = spec.params(cursor);
        p["delivery"] = json!({ "mode": "webhook", "url": url, "secret": secret });
        p["ttlMs"] = json!(ttl_ms);
        p
    };

    let mut next_refresh = Duration::ZERO;
    let mut failures = 0u32;
    // A rotation the server has not yet confirmed. Re-sent unchanged on retry, so a failed refresh
    // never leaves the server signing with a secret this receiver has already forgotten.
    let mut pending: Option<(String, Vec<u8>)> = None;
    let mut first = true;
    loop {
        tokio::select! {
            () = cancel.cancelled() => {
                if !first {
                    webhook_unsubscribe(&hub, &spec, &url).await;
                }
                return;
            }
            msg = rx.recv() => {
                match msg {
                    Some(WebhookMsg::Event(event, ack)) => {
                        let status = match hub.deliver(&spec, McpEventDelivery::Webhook, &state, event) {
                            Delivery::Accepted | Delivery::Duplicate => {
                                hub.store.commit().await;
                                200
                            }
                            Delivery::Full => 503,
                        };
                        let _ = ack.send(status);
                    }
                    Some(WebhookMsg::Gap(env, ack)) => {
                        hub.gap(&spec, &state, &env);
                        hub.store.commit().await;
                        let _ = ack.send(200);
                    }
                    Some(WebhookMsg::Terminated(env, ack)) => {
                        // The subscription no longer exists server-side; nothing to unsubscribe.
                        let _ = ack.send(200);
                        let error = env.get("error").cloned().unwrap_or(Value::Null);
                        hub.ended(&spec, &state, RpcError::from_json(&error));
                        return;
                    }
                    None => return,
                }
                continue;
            }
            () = tokio::time::sleep(next_refresh) => {}
        }

        // Subscribe (first pass) or refresh, rotating the secret on every refresh.
        let secret_for_call = if first {
            secret_wire.clone()
        } else {
            let (wire, raw) = match pending.clone() {
                Some(p) => p,
                None => match new_secret() {
                    Ok(p) => {
                        pending = Some(p.clone());
                        p
                    }
                    Err(e) => {
                        hub.ended(&spec, &state, RpcError::local(e));
                        webhook_unsubscribe(&hub, &spec, &url).await;
                        return;
                    }
                },
            };
            let mut s = route
                .secrets
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if s.current != raw {
                let old = std::mem::replace(&mut s.current, raw);
                s.previous = Some(old);
            }
            wire
        };
        let result = match hub.conn(&spec.server).await {
            Ok(conn) => {
                // Before every subscribe/refresh, so the verification challenge is already
                // checked and a key rotation is picked up (see the module's key policy).
                if let Err(e) = update_server_keys(&hub, &conn, &route, first).await {
                    if let Some(tx) = ready.take() {
                        let _ = tx.send(Err(e));
                    }
                    return;
                }
                let call_fut = conn.call(
                    "events/subscribe",
                    subscribe_params(&secret_for_call, state.cursor()),
                    RPC_TIMEOUT,
                );
                tokio::select! {
                    r = call_fut => r,
                    () = cancel.cancelled() => {
                        webhook_unsubscribe(&hub, &spec, &url).await;
                        return;
                    }
                }
            }
            Err(e) => Err(RpcError::local(e)),
        };
        match result {
            Ok(grant) => {
                failures = 0;
                if !first {
                    pending = None;
                    secret_wire = secret_for_call;
                }
                let refresh_before = grant
                    .get("refreshBefore")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let Some(id) = grant.get("id").and_then(Value::as_str)
                    && let Ok(mut slot) = route.subscription_id.lock()
                {
                    *slot = Some(id.to_owned());
                }
                state.set_cursor_from(&grant);
                state.with(|s| {
                    s.state = "active";
                    s.last_error = None;
                    s.refresh_before = refresh_before.clone();
                    s.subscription_id = grant.get("id").and_then(Value::as_str).map(str::to_owned);
                    if !first {
                        s.refreshes += 1;
                    }
                    s.delivery_status = grant.get("deliveryStatus").cloned();
                });
                if grant.get("truncated").and_then(Value::as_bool) == Some(true) {
                    hub.gap(&spec, &state, &grant);
                }
                if let Some(ds) = grant.get("deliveryStatus")
                    && (ds.get("active").and_then(Value::as_bool) == Some(false)
                        || ds.get("lastError").is_some_and(|e| !e.is_null()))
                {
                    hub.status_event(&spec, "delivery_status", json!({ "delivery_status": ds }));
                }
                first = false;
                ready_ok(&mut ready);
                next_refresh = match refresh_delay(refresh_before.as_deref()) {
                    Ok(d) => d,
                    Err(d) => {
                        let raw = refresh_before.unwrap_or_default();
                        tracing::warn!(
                            refresh_before = %raw,
                            "unparseable refreshBefore from the server; refreshing in {}s",
                            d.as_secs()
                        );
                        hub.status_event(
                            &spec,
                            "error",
                            json!({ "error": format!("unparseable refreshBefore {raw:?}; refreshing in {}s", d.as_secs()) }),
                        );
                        d
                    }
                };
            }
            Err(e) => {
                if let Some(tx) = ready.take() {
                    let _ = tx.send(Err(e.to_string()));
                    return;
                }
                if e.is_terminal() && e.code != Some(-32015) {
                    hub.ended(&spec, &state, e);
                    return;
                }
                failures += 1;
                state.with(|s| {
                    s.state = "retrying";
                    s.last_error = Some(e.to_string());
                });
                hub.status_event(&spec, "error", json!({ "error": e.to_json() }));
                next_refresh = backoff(failures, Duration::from_secs(60));
            }
        }
    }
}

async fn webhook_unsubscribe(hub: &Hub, spec: &SubSpec, url: &str) {
    let Ok(conn) = hub.conn(&spec.server).await else {
        return;
    };
    let params = json!({
        "name": spec.sub.name,
        "arguments": spec.arguments(),
        "delivery": { "mode": "webhook", "url": url },
    });
    if let Err(e) = conn
        .call("events/unsubscribe", params, Duration::from_secs(3))
        .await
        && e.code != Some(-32011)
    {
        tracing::debug!(error = %e, "events/unsubscribe failed; the subscription will lapse at its TTL");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn sign(secret: &[u8], id: &str, ts: &str, body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret).unwrap();
        mac.update(format!("{id}.{ts}.").as_bytes());
        mac.update(body);
        format!(
            "v1,{}",
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
        )
    }

    #[test]
    fn signature_accepts_any_listed_v1_and_rejects_tampering() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let body = br#"{"eventId":"e1"}"#;
        let good = sign(secret, "e1", "100", body);
        assert!(verify_signature(secret, "e1", "100", body, &good));
        let multi = format!("v1,AAAA {good}");
        assert!(verify_signature(secret, "e1", "100", body, &multi));
        assert!(!verify_signature(secret, "e1", "101", body, &good));
        assert!(!verify_signature(secret, "e1", "100", b"{}", &good));
        assert!(!verify_signature(
            b"other-secret-other-secret-other!",
            "e1",
            "100",
            body,
            &good
        ));
    }

    #[test]
    fn timestamp_freshness_is_symmetric_and_cannot_overflow() {
        let now = 1_800_000_000;
        assert!(timestamp_fresh(now, now));
        assert!(timestamp_fresh(now - 300, now));
        assert!(timestamp_fresh(now + 300, now));
        assert!(!timestamp_fresh(now - 301, now), "stale");
        assert!(!timestamp_fresh(now + 301, now), "future-dated");
        assert!(!timestamp_fresh(i64::MIN, now));
        assert!(!timestamp_fresh(i64::MAX, now));
        assert!(!timestamp_fresh(i64::MIN, i64::MAX));
    }

    #[tokio::test]
    async fn the_receiver_enforces_token_freshness_signature_and_echoes_challenges() {
        let (tx, mut rx) = mpsc::channel(4);
        let secret = b"0123456789abcdef0123456789abcdef".to_vec();
        let previous = b"fedcba9876543210fedcba9876543210".to_vec();
        routes().insert(
            "tok-unit".into(),
            Arc::new(Route {
                secrets: Mutex::new(Secrets {
                    current: secret.clone(),
                    previous: Some(previous.clone()),
                }),
                subscription_id: Mutex::new(Some("sub_1".into())),
                server_keys: Mutex::new(Vec::new()),
                tx,
            }),
        );
        // Ack every queued event as stored.
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let WebhookMsg::Event(_, ack) = msg {
                    let _ = ack.send(200);
                }
            }
        });
        let now = now_unix().to_string();
        let headers = |id: &str, ts: &str, sig: String, sub: &str| {
            vec![
                ("webhook-id".to_string(), id.to_string()),
                ("webhook-timestamp".to_string(), ts.to_string()),
                ("webhook-signature".to_string(), sig),
                ("X-MCP-Subscription-Id".to_string(), sub.to_string()),
            ]
        };
        let challenge = br#"{"type":"verification","challenge":"nonce-1"}"#;
        let r = receive_webhook(
            "tok-unit",
            &headers(
                "msg_v",
                &now,
                sign(&secret, "msg_v", &now, challenge),
                "sub_1",
            ),
            challenge,
        )
        .await;
        assert_eq!(r.status, 200);
        assert_eq!(
            serde_json::from_slice::<Value>(&r.body).unwrap(),
            json!({"challenge": "nonce-1"})
        );

        let event = br#"{"eventId":"e1","name":"x","timestamp":"t","data":{}}"#;
        let r = receive_webhook(
            "tok-unit",
            &headers("e1", &now, sign(&previous, "e1", &now, event), "sub_1"),
            event,
        )
        .await;
        assert_eq!(
            r.status, 200,
            "the previous secret still verifies (rotation grace)"
        );

        for ts in [
            (now_unix() - 301).to_string(),
            (now_unix() + 301).to_string(),
            i64::MIN.to_string(),
        ] {
            let r = receive_webhook(
                "tok-unit",
                &headers("e1", &ts, sign(&secret, "e1", &ts, event), "sub_1"),
                event,
            )
            .await;
            assert_eq!(r.status, 400, "timestamp {ts}");
        }
        let r = receive_webhook(
            "tok-unit",
            &headers(
                "e1",
                &now,
                sign(b"wrong-secret-wrong-secret-wrong!", "e1", &now, event),
                "sub_1",
            ),
            event,
        )
        .await;
        assert_eq!(r.status, 401);
        let r = receive_webhook(
            "tok-unit",
            &headers("e1", &now, sign(&secret, "e1", &now, event), "sub_other"),
            event,
        )
        .await;
        assert_eq!(r.status, 401);
        assert_eq!(receive_webhook("tok-missing", &[], event).await.status, 410);
        routes().remove("tok-unit");
    }

    #[test]
    fn v1a_server_signatures_verify_against_known_keys_only() {
        use ed25519_dalek::{Signer, SigningKey};
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let other = SigningKey::from_bytes(&[9u8; 32]);
        let jwks = json!({"keys": [{
            "kty": "OKP", "crv": "Ed25519", "kid": "k1",
            "x": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes()),
        }]});
        let keys = ed25519_keys_from_jwks(&jwks);
        assert_eq!(keys.len(), 1);
        assert_eq!(key_from_b64(&key_to_b64(&keys[0])), Some(keys[0]));
        let body = br#"{"eventId":"e1"}"#;
        let sig = |k: &SigningKey| {
            format!(
                "v1a,{}",
                base64::engine::general_purpose::STANDARD
                    .encode(k.sign(b"e1.100.{\"eventId\":\"e1\"}").to_bytes())
            )
        };
        assert!(verify_server_signature(
            &keys,
            "e1",
            "100",
            body,
            &format!("v1,AAAA {}", sig(&key))
        ));
        assert!(!verify_server_signature(
            &keys,
            "e1",
            "100",
            body,
            &sig(&other)
        ));
        assert!(!verify_server_signature(
            &keys,
            "e1",
            "101",
            body,
            &sig(&key)
        ));
        assert!(!verify_server_signature(
            &keys, "e1", "100", body, "v1,AAAA"
        ));
    }

    #[test]
    fn an_unparseable_refresh_before_refreshes_soon_not_in_an_hour() {
        assert_eq!(
            refresh_delay(Some("tomorrow-ish")),
            Err(UNPARSEABLE_REFRESH)
        );
        assert_eq!(refresh_delay(None), Ok(Duration::from_secs(3600)));
        // Minute precision (valid ISO 8601) parses.
        let soon = refresh_delay(Some("2000-01-01T00:00Z")).unwrap();
        assert_eq!(soon, Duration::from_millis(500), "already past: the floor");
    }
}
