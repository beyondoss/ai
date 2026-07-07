//! Live WebSocket transport for OpenAI-Codex-Responses (the ChatGPT-subscription/OAuth backend —
//! `RouteOverride::Prefixed`-routed requests, `client.rs`'s `req.is_codex`). Mirrors pi's real shipped
//! `packages/ai/src/api/openai-codex-responses.ts`: that backend accepts the same request/response
//! shape over a persistent WebSocket as it does over plain HTTP/SSE, and — critically — lets a
//! *second* turn on the same connection send only the *delta* of new conversation items (via
//! `previous_response_id` + a trimmed `input`) instead of resending the whole transcript every turn.
//! `dialect::openai_responses::build_body`/`Decoder` already do all the wire-shape work for the plain
//! HTTP/SSE path; this module sits *above* them — it builds on the exact same full request body and
//! decoder, and only adds the WebSocket connection lifecycle and the delta-diffing logic on top.
//!
//! # Transport selection
//!
//! There is no `"websocket" | "websocket-cached" | "auto" | "sse"` enum here, unlike pi's own
//! `options?.transport`: this crate always behaves like pi's `"auto"` default (attempt the WebSocket,
//! transparently fall back to the existing HTTP/SSE path on any failure/unavailability) — the task
//! that added this module found no concrete need for the other three modes (`"websocket"`/
//! `"websocket-cached"` never silently fall back at all; a bare `"sse"` is just this feature switched
//! off, already covered by [`GatewayClient::with_codex_websocket`](crate::client::GatewayClient::with_codex_websocket)).
//! If a real caller need for pinning one mode explicitly ever appears, add it then — see
//! `with_codex_websocket`'s own doc comment for the CLI/settings surface this still needs, which is
//! deliberately left to a later round (out of scope for `crates/agent`, which this round doesn't
//! touch).
//!
//! # Connection cache
//!
//! [`CodexWebSocketCache`] is one per [`GatewayClient`](crate::client::GatewayClient) (its whole
//! lifetime, not one per turn), keyed by [`ModelRequest::cache_key`](crate::transport::ModelRequest::cache_key)
//! — the same stable per-conversation identifier this dialect already sends as OpenAI's
//! `prompt_cache_key` (pi's Codex client literally reuses its own `sessionId` for both purposes, the
//! same way). A request with no `cache_key` still gets the WebSocket attempt, just with no connection
//! reuse — matches pi's own `sessionId`-optional `acquireWebSocket`.
//!
//! Modeled as "present in the map == idle, absent == in-flight-or-never-connected" rather than pi's
//! explicit `busy: boolean` flag: a connection is *removed* from the map for the duration of one turn
//! ([`CodexWebSocketCache::checkout`]) and only reinserted on a clean completion
//! ([`CodexWebSocketCache::checkin`]). A second concurrent turn on the same `cache_key` (two sessions
//! sharing one key, or two in-flight RPCs against one session) finds nothing to reuse and opens its
//! own fresh, uncached connection instead — the identical outcome pi's own busy-connection branch
//! reaches, just without a separate flag to fall out of sync with reality. This also sidesteps ever
//! holding a lock across an `.await`: [`checkout`](CodexWebSocketCache::checkout)/
//! [`checkin`](CodexWebSocketCache::checkin) are both plain, synchronous, sub-millisecond critical
//! sections over a [`std::sync::Mutex`].
//!
//! Unlike pi, there is no proactive idle-timeout/max-age background timer
//! (`SESSION_WEBSOCKET_CACHE_TTL_MS`/`SESSION_WEBSOCKET_MAX_AGE_MS`, both enforced via `setTimeout` in
//! the real client) — a spawned-per-connection sweep task is a meaningfully bigger commitment (this
//! crate has never spawned a background task of its own before) for a purely eager-resource-reclaim
//! optimization, not a correctness requirement. Instead, [`checkout`](CodexWebSocketCache::checkout)
//! lazily evicts an entry past [`MAX_CONNECTION_AGE`] the next time it's looked up — a connection can
//! sit open somewhat longer than pi's proactive close in a quiet session, which only matters for
//! server-side resource pressure, not for this client's own correctness. A cached connection the
//! *server* silently closed between turns (idle timeout, restart) is handled reactively instead of
//! pi's `readyState`-checked-before-reuse: see [`Attempt`]'s "stale reused connection" retry, one
//! connect-with-a-fresh-socket retry before the ordinary SSE-fallback path, so an otherwise-healthy
//! session doesn't stick to SSE for its whole remaining lifetime just because one connection went
//! idle. If a real need for the proactive sweep appears (a long-lived `serve` process visibly
//! accumulating idle sockets), add it as its own follow-up.
//!
//! # Delta diffing
//!
//! [`Continuation`] plus [`build_wire_body`] is this crate's take on pi's
//! `CachedWebSocketContinuationState`/`buildCachedWebSocketRequestBody`/`getCachedWebSocketInputDelta`.
//! One structural simplification: pi tracks `lastRequestBody`/`lastResponseItems` as two separate
//! fields and concatenates them at compare time; here they're stored pre-concatenated as
//! `baseline_input`, since nothing ever needs them apart. The harder question — where
//! `lastResponseItems` (the assistant's own last turn, converted back to input-item shape) comes from
//! — is answered differently than pi, and deliberately so: pi reconstructs it via
//! `convertResponsesMessages` (its `ContentBlock`-equivalent round-trip). This module instead harvests
//! each `response.output_item.done` event's own raw `item` object directly off the wire while decoding
//! (see [`Receiver`]) — this crate's `client.rs`/dialect layer builds the request body from
//! `ModelRequest.messages`/`ContentBlock`s, not from a live `AssistantMessage` object the transport
//! layer doesn't have (unlike pi's `stream()`, which holds the very `AssistantMessage` it's building).
//! The harvested item is used **only** as an internal comparison baseline for the *next* turn's prefix
//! check — it is never itself sent over the wire (a full resend always goes back through
//! `dialect::openai_responses::build_body`/`push_assistant_content`, and a delta send only ever slices
//! bytes out of the input array that same function built this turn). So if the harvested wire shape
//! ever drifts from what `push_assistant_content` reconstructs from the same turn's persisted
//! `ContentBlock`s (an extra field a future API revision adds, say), the *only* consequence is a
//! prefix-equality miss — the next turn silently falls back to a full resend instead of a delta, never
//! a correctness bug (no duplicated or dropped conversation content either way).
//!
//! `max_output_tokens` joins `input`/`previous_response_id` as a field [`body_fingerprint`] excludes
//! from the "is this still fundamentally the same request" comparison — a beyond-only field (pi's own
//! `RequestBody` interface has no such field at all; the real Codex backend manages its own output
//! ceiling) that `dialect::openai_responses::build_body` recomputes from the live prompt's *estimated
//! size* every turn (`clamp_max_tokens_to_context`), so it legitimately changes turn over turn as a
//! conversation grows — comparing it verbatim the way pi's own `requestBodyWithoutInput` compares
//! everything else would defeat the delta path on almost every second turn.
//!
//! # Failure classification
//!
//! Mirrors pi's `isCodexNonTransportError`/`websocketStarted` split, but only where it actually has to:
//! **before** the first event of a turn has streamed, a failure is classified (see [`Attempt`]) as a
//! connection-limit signal (retried once with a fresh connect — pi's own
//! `isWebSocketConnectionLimitReachedError` retry), an ordinary transport failure (silently falls back
//! to the existing HTTP/SSE path for this turn, and marks this `cache_key` sticky-SSE for every
//! subsequent turn — pi's `websocketSseFallbackSessions`), or a protocol/API failure (an in-band
//! `{"type":"error"}`/`"response.failed"` event, or malformed JSON — propagates directly, no fallback,
//! matching pi's `CodexApiError`/`CodexProtocolError`). **After** the first event has streamed, this
//! module needs no such classification at all: a Rust [`Stream`](futures::Stream) that has already
//! yielded some `Ok` items can simply yield an `Err` next and stop — exactly what the existing HTTP/SSE
//! decoder already does for its own mid-stream failures, and `Agent::is_retryable_mid_stream` already
//! knows how to react to (this module tags a genuine connectivity drop with the same
//! [`MID_STREAM_NETWORK_ERROR`] prefix the SSE path uses, so that existing classifier treats a
//! WebSocket-sourced network drop identically to an HTTP one).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};

use crate::dialect::StreamDecoder;
use crate::dialect::openai_responses::Decoder as CodexDecoder;
use crate::error::{Error, MID_STREAM_NETWORK_ERROR};
use crate::message::StreamEvent;
use crate::transport::EventStream;

/// Cap on the TCP+TLS+WS-upgrade handshake — mirrors `client.rs`'s own `CONNECT_TIMEOUT` for the
/// HTTP/SSE path (10s), not pi's slightly larger 15s default; there's no reason for this crate's two
/// transports to the same backend to disagree on how long a dead endpoint gets before it's treated as
/// unavailable.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Idle timeout *between* frames on an open WebSocket — mirrors `client.rs`'s own SSE `READ_TIMEOUT`
/// (600s): the gateway/backend can legitimately go quiet for a long extended-thinking gap, so this
/// isn't a ceiling on total turn duration, only on "has this socket gone silently dead".
const IDLE_TIMEOUT: Duration = Duration::from_secs(600);
/// A cached connection older than this is treated as expired at its next checkout and closed rather
/// than reused — mirrors pi's `SESSION_WEBSOCKET_MAX_AGE_MS` (55 minutes), a defensive ceiling
/// independent of activity (some backends cap a single connection's total lifetime regardless of how
/// recently it was used).
const MAX_CONNECTION_AGE: Duration = Duration::from_secs(55 * 60);
/// The OpenAI-Beta opt-in the real Codex backend's WebSocket upgrade requires — pi's
/// `OPENAI_BETA_RESPONSES_WEBSOCKETS`.
const OPENAI_BETA_RESPONSES_WEBSOCKETS: &str = "responses_websockets=2026-02-06";
/// Substring pi's own `WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE` checks for in a rejected handshake's
/// error body, alongside a bare `429` status — see [`is_connection_limit_signal`].
const CONNECTION_LIMIT_REACHED_CODE: &str = "websocket_connection_limit_reached";

/// The concrete WebSocket stream type `tokio_tungstenite::connect_async` hands back. A type alias
/// purely so the rest of this module doesn't repeat the full generic — not a public seam.
type WsSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

// ============================================================================
// Delta diffing (pure, no I/O — see the module doc comment)
// ============================================================================

/// Body fields excluded from [`body_fingerprint`]'s "is this still the same kind of request"
/// comparison — see the module doc comment for why `max_output_tokens` joins `input`/
/// `previous_response_id` here.
const DIFF_EXCLUDED_FIELDS: [&str; 3] = ["input", "previous_response_id", "max_output_tokens"];

/// `body` with [`DIFF_EXCLUDED_FIELDS`] removed — pi's `requestBodyWithoutInput`, plus the
/// `max_output_tokens` exclusion this crate's own body shape needs (see the module doc comment).
fn body_fingerprint(body: &Value) -> Value {
    let mut body = body.clone();
    if let Some(obj) = body.as_object_mut() {
        for key in DIFF_EXCLUDED_FIELDS {
            obj.remove(key);
        }
    }
    body
}

/// `body["input"]` as a plain vec, or empty if absent/not an array (never true for a body
/// `dialect::openai_responses::build_body` produced, but this stays total rather than assuming that).
fn input_items(body: &Value) -> Vec<Value> {
    body.get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Whether `item` (one `response.output_item.done` event's own `item` object) is a shape
/// `dialect::openai_responses::push_assistant_content` would ever reconstruct from a replayed
/// assistant turn — a `message`, `reasoning`, or `function_call` item. Anything else (a
/// `function_call_output` only ever comes from a *user*-role tool-result message, never an assistant
/// turn's own output; a genuinely new item type this dialect doesn't otherwise recognize either) is
/// not worth keeping as replay-baseline context.
fn is_replayable_output_item(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("message") | Some("reasoning") | Some("function_call")
    )
}

/// A response's own id, read off any event that embeds a `response` object (`response.created`,
/// `response.completed`/`.incomplete`/`.done` all carry one) — captured so a *subsequent* turn can
/// reference it as `previous_response_id`.
fn extract_response_id(frame: &Value) -> Option<String> {
    frame
        .get("response")?
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Per-connection continuation state, kept alongside a cached socket — pi's
/// `CachedWebSocketContinuationState`, restructured (see the module doc comment) so `baseline_input`
/// is already the concatenation [`build_wire_body`]'s prefix check needs, rather than two fields
/// combined at compare time.
#[derive(Clone, Debug, PartialEq)]
struct Continuation {
    /// [`body_fingerprint`] of the full body actually sent (BEFORE any delta-trimming) the turn this
    /// continuation was recorded for.
    fingerprint: Value,
    /// Everything the backend already has, in this connection's view: the full `input` array that
    /// turn's request carried, plus that turn's own response items (see [`is_replayable_output_item`]),
    /// in wire order.
    baseline_input: Vec<Value>,
    /// The response id to send as `previous_response_id` when this baseline's prefix still matches.
    last_response_id: String,
}

/// Compute the wire body to actually send: a full resend of `full_body` unless `continuation` is
/// present, its fingerprint still matches, and `full_body`'s own `input` starts with exactly
/// `continuation.baseline_input` — in which case only the new tail is sent, alongside
/// `previous_response_id`. Mirrors pi's `buildCachedWebSocketRequestBody`/
/// `getCachedWebSocketInputDelta`.
fn build_wire_body(full_body: &Value, continuation: Option<&Continuation>) -> Value {
    let Some(continuation) = continuation else {
        return full_body.clone();
    };
    if continuation.last_response_id.is_empty() {
        return full_body.clone();
    }
    if body_fingerprint(full_body) != continuation.fingerprint {
        return full_body.clone();
    }
    let current_input = input_items(full_body);
    if current_input.len() < continuation.baseline_input.len() {
        return full_body.clone();
    }
    if current_input[..continuation.baseline_input.len()] != continuation.baseline_input[..] {
        return full_body.clone();
    }
    let delta = &current_input[continuation.baseline_input.len()..];
    let mut trimmed = full_body.clone();
    if let Some(obj) = trimmed.as_object_mut() {
        obj.insert("input".to_string(), Value::Array(delta.to_vec()));
        obj.insert(
            "previous_response_id".to_string(),
            Value::String(continuation.last_response_id.clone()),
        );
    }
    trimmed
}

/// Wrap `body` (the wire body [`build_wire_body`] computed — full or delta) in the `response.create`
/// envelope the Codex WebSocket protocol expects as the single outbound text frame per turn — pi's
/// `socket.send(JSON.stringify({ type: "response.create", ...requestBody }))`.
fn wire_frame(body: &Value) -> String {
    let mut obj = match body {
        Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    obj.insert(
        "type".to_string(),
        Value::String("response.create".to_string()),
    );
    Value::Object(obj).to_string()
}

// ============================================================================
// Connection cache
// ============================================================================

struct CachedConnection {
    socket: WsSocket,
    created_at: Instant,
    continuation: Option<Continuation>,
}

/// One per [`GatewayClient`](crate::client::GatewayClient) — see the module doc comment for the
/// "present == idle" cache-key design.
pub(crate) struct CodexWebSocketCache {
    idle: Mutex<HashMap<String, CachedConnection>>,
    sse_fallback: Mutex<std::collections::HashSet<String>>,
}

impl CodexWebSocketCache {
    pub(crate) fn new() -> Self {
        Self {
            idle: Mutex::new(HashMap::new()),
            sse_fallback: Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Take ownership of `key`'s cached connection, if any is idle and not past
    /// [`MAX_CONNECTION_AGE`]. A brief, synchronous, lock-held critical section only — never awaits.
    fn checkout(&self, key: &str) -> Option<CachedConnection> {
        let mut guard = self
            .idle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = guard.remove(key)?;
        if entry.created_at.elapsed() >= MAX_CONNECTION_AGE {
            // Dropped here: past its age ceiling, close it rather than hand it back for reuse.
            None
        } else {
            Some(entry)
        }
    }

    /// Return a still-healthy connection to the idle pool after a clean turn completion. Overwrites
    /// (and thus closes) any entry already present for `key` — the only way that can happen is two
    /// concurrent turns on the same `cache_key` both completing and re-caching; whichever checks in
    /// last wins, which is a benign, documented simplification (see the module doc comment) rather
    /// than a correctness issue (each turn already sent/received its own complete, correct exchange).
    fn checkin(&self, key: String, connection: CachedConnection) {
        let mut guard = self
            .idle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.insert(key, connection);
    }

    /// Mark `key` as having had a genuine WebSocket transport failure — every subsequent turn for this
    /// `cache_key` skips the WebSocket attempt entirely and goes straight to HTTP/SSE, for the rest of
    /// this cache's lifetime (mirrors pi's `websocketSseFallbackSessions`, which is likewise never
    /// cleared except by an explicit debug/reset call this crate has no equivalent of).
    fn mark_fallback(&self, key: &str) {
        self.sse_fallback
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key.to_string());
    }

    fn is_fallback_active(&self, key: &str) -> bool {
        self.sse_fallback
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(key)
    }
}

// ============================================================================
// URL / header construction
// ============================================================================

/// Rewrite an `http(s)://` URL to the equivalent `ws(s)://` one — pi's `resolveCodexWebSocketUrl`.
/// `None` for anything else (never reached in practice: the only caller only ever passes the URL
/// `client.rs` already built for a `RouteOverride::Prefixed` Codex route, always `http(s)://`).
pub(crate) fn to_ws_url(url: &str) -> Option<String> {
    if let Some(rest) = url.strip_prefix("https://") {
        return Some(format!("wss://{rest}"));
    }
    url.strip_prefix("http://")
        .map(|rest| format!("ws://{rest}"))
}

/// A per-connection request id: the caller's own `cache_key` when there is one (matching pi's
/// `sessionId`, reused for both `prompt_cache_key` and this), else a freshly generated one — pi's
/// `options?.sessionId || createCodexRequestId()`.
pub(crate) fn generate_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    format!("codex_ws_{now}_{n}")
}

/// Build the WebSocket upgrade request's headers — pi's `buildWebSocketHeaders`: the same
/// credential/static headers the HTTP/SSE path already sends (`direct_headers` — Codex's
/// `chatgpt-account-id` among them), an `Authorization: Bearer` for the api key, and this transport's
/// own `OpenAI-Beta`/`x-client-request-id`/`session-id` in place of the HTTP path's `accept`/
/// `content-type`/experimental `OpenAI-Beta` value.
pub(crate) fn build_headers(
    api_key: &str,
    direct_headers: &[(&'static str, String)],
    request_id: &str,
) -> Vec<(String, String)> {
    let mut headers = Vec::with_capacity(direct_headers.len() + 4);
    headers.push(("Authorization".to_string(), format!("Bearer {api_key}")));
    for (name, value) in direct_headers {
        headers.push(((*name).to_string(), value.clone()));
    }
    headers.push((
        "OpenAI-Beta".to_string(),
        OPENAI_BETA_RESPONSES_WEBSOCKETS.to_string(),
    ));
    headers.push(("x-client-request-id".to_string(), request_id.to_string()));
    headers.push(("session-id".to_string(), request_id.to_string()));
    headers
}

// ============================================================================
// SSE-fallback body compression
// ============================================================================

/// Zstd level applied to a Codex SSE-fallback request body — matches pi's own
/// `REQUEST_COMPRESSION_ZSTD_LEVEL` (`openai-codex-responses.ts`), the same level the official Codex
/// client compresses against.
const SSE_FALLBACK_ZSTD_LEVEL: i32 = 3;

/// The header this backend's SSE-fallback endpoint expects on a zstd-compressed request body — mirrors
/// pi's own `sseHeaders.set("content-encoding", "zstd")`. Never sent on the WebSocket frame above: pi's
/// own comment there is explicit that transport always sends the uncompressed JSON frame, matching the
/// official Codex client, and this crate's `wire_frame`/`attempt_once` do the same.
///
/// Read from `client.rs::send_with_retry`, gated on `is_codex_sse_fallback`.
pub(crate) const SSE_FALLBACK_CONTENT_ENCODING: (&str, &str) = ("content-encoding", "zstd");

/// Compress a Codex request body's JSON bytes with zstd, for the one HTTP/SSE path this crate falls
/// back to when the live WebSocket transport ([`try_stream`] above) can't be used for a turn — a
/// bandwidth optimization only (the real backend's SSE endpoint also accepts the plain, uncompressed
/// JSON this crate sent before this function existed), not a correctness requirement. Returns `None` on
/// a compression failure; the caller should fall back to sending the plain JSON body with no
/// `Content-Encoding` header rather than failing the whole request over this optimization — mirrors
/// pi's own `compressRequestBodyZstd`, which returns `null` on the same "compression unavailable"
/// circumstance (there, a browser/Vite build with no native zstd binding; here, defensive parity with
/// that same fail-open contract even though `zstd::stream::encode_all` has no realistic failure mode
/// for an in-memory `&[u8]` source).
///
/// Deliberately narrow: this is the Codex backend's own SSE-fallback endpoint accepting
/// `Content-Encoding: zstd` request bodies (`packages/ai/src/api/openai-codex-responses.ts`), not a
/// general-purpose compression capability this crate's other request paths should reach for.
///
/// Wired into the live request path at `client.rs::send_with_retry`, gated on `is_codex_sse_fallback`
/// (the same `req.is_codex && dialect == Dialect::OpenAiResponses` check
/// `client.rs::GatewayClient::stream` already uses to decide whether to attempt the WebSocket
/// transport, precomputed once and reused rather than re-derived).
pub(crate) fn compress_sse_fallback_body(json: &[u8]) -> Option<Vec<u8>> {
    zstd::stream::encode_all(json, SSE_FALLBACK_ZSTD_LEVEL).ok()
}

// ============================================================================
// Connecting
// ============================================================================

enum ConnectFailure {
    /// The handshake was rejected with a signal matching the Codex backend's own connection-limit
    /// error (see [`is_connection_limit_signal`]) — retried once before falling back.
    ConnectionLimit(String),
    Other(String),
}

/// Whether a rejected WebSocket upgrade looks like the Codex backend's own connection-limit signal —
/// pi's `isWebSocketConnectionLimitReachedError`, which inspects a structured `error.code` field pi's
/// richer client-side error type carries; this inspects the raw rejected-handshake status/body
/// directly, the information `tungstenite::Error::Http` actually exposes.
fn is_connection_limit_signal(status: u16, body: &str) -> bool {
    status == 429 || body.contains(CONNECTION_LIMIT_REACHED_CODE)
}

fn classify_connect_error(error: tokio_tungstenite::tungstenite::Error) -> ConnectFailure {
    use tokio_tungstenite::tungstenite::Error as WsError;
    if let WsError::Http(response) = &error {
        let status = response.status();
        let body = response
            .body()
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        let message = format!("codex websocket handshake rejected: {status} {body}");
        return if is_connection_limit_signal(status.as_u16(), &body) {
            ConnectFailure::ConnectionLimit(message)
        } else {
            ConnectFailure::Other(message)
        };
    }
    ConnectFailure::Other(format!("codex websocket connect failed: {error}"))
}

/// Open one fresh WebSocket connection to `url` with `headers` attached to the upgrade request.
async fn connect(
    url: &str,
    headers: &[(String, String)],
) -> std::result::Result<WsSocket, ConnectFailure> {
    let mut request = url
        .into_client_request()
        .map_err(|e| ConnectFailure::Other(format!("invalid codex websocket url: {e}")))?;
    for (name, value) in headers {
        let name = match HeaderName::from_bytes(name.as_bytes()) {
            Ok(name) => name,
            Err(e) => {
                return Err(ConnectFailure::Other(format!(
                    "invalid header name {name:?}: {e}"
                )));
            }
        };
        let value = match HeaderValue::from_str(value) {
            Ok(value) => value,
            Err(e) => {
                return Err(ConnectFailure::Other(format!(
                    "invalid header value for {name:?}: {e}"
                )));
            }
        };
        request.headers_mut().insert(name, value);
    }
    match tokio::time::timeout(CONNECT_TIMEOUT, tokio_tungstenite::connect_async(request)).await {
        Err(_elapsed) => Err(ConnectFailure::Other(format!(
            "codex websocket connect timed out after {CONNECT_TIMEOUT:?}"
        ))),
        Ok(Err(e)) => Err(classify_connect_error(e)),
        Ok(Ok((socket, _response))) => Ok(socket),
    }
}

// ============================================================================
// Receiving / decoding
// ============================================================================

/// The outcome of reading and decoding frames until *something* worth reporting happens: one or more
/// [`StreamEvent`]s, a clean end of turn, or a definitive failure.
enum FrameOutcome {
    Events(Vec<StreamEvent>),
    /// The decoder's own terminal event was seen and flushed cleanly — nothing more to read.
    Done,
    /// `protocol: true` — an in-band API/protocol failure (never SSE-fallback-eligible if this is the
    /// very first outcome of a turn); `protocol: false` — a connectivity failure.
    Failure {
        error: Error,
        protocol: bool,
    },
}

/// Owns one live connection's decode state for one turn. Not `Clone`/`Copy` — consumed by the
/// generator `try_stream` builds once a turn's first event has been confirmed.
struct Receiver {
    socket: WsSocket,
    decoder: CodexDecoder,
    response_id: Option<String>,
    harvested: Vec<Value>,
    finished: bool,
}

impl Receiver {
    fn new(socket: WsSocket) -> Self {
        Self {
            socket,
            decoder: CodexDecoder::default(),
            response_id: None,
            harvested: Vec::new(),
            finished: false,
        }
    }

    /// Read (and, if necessary, wait for) frames until this call has something to report. Safe to
    /// call again after [`FrameOutcome::Done`] — it just returns `Done` again without touching the
    /// socket, so a turn is considered complete the instant its terminal event is decoded, without
    /// waiting for the peer to also close the TCP connection (which — for a connection this turn
    /// intends to cache for reuse — it deliberately never does).
    async fn next_events(&mut self) -> FrameOutcome {
        if self.finished {
            return FrameOutcome::Done;
        }
        loop {
            let message = match tokio::time::timeout(IDLE_TIMEOUT, self.socket.next()).await {
                Err(_elapsed) => {
                    return FrameOutcome::Failure {
                        error: Error::Transport(format!(
                            "{MID_STREAM_NETWORK_ERROR}: codex websocket idle timeout after {IDLE_TIMEOUT:?}"
                        )),
                        protocol: false,
                    };
                }
                Ok(None) => return self.closed_outcome(),
                Ok(Some(Err(e))) => {
                    return FrameOutcome::Failure {
                        error: Error::Transport(format!("{MID_STREAM_NETWORK_ERROR}: {e}")),
                        protocol: false,
                    };
                }
                Ok(Some(Ok(message))) => message,
            };

            let text = match message {
                Message::Text(t) => t.as_str().to_string(),
                Message::Binary(b) => match String::from_utf8(b.to_vec()) {
                    Ok(s) => s,
                    // Not a JSON event frame — ignore rather than fail the turn over it, mirroring
                    // pi's own best-effort `decodeWebSocketData`.
                    Err(_) => continue,
                },
                Message::Close(_) => return self.closed_outcome(),
                Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
            };

            let value: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    return FrameOutcome::Failure {
                        error: Error::Transport(format!("malformed codex websocket json: {e}")),
                        protocol: true,
                    };
                }
            };

            if let Some(msg) = crate::dialect::sse_error(&value) {
                return FrameOutcome::Failure {
                    error: Error::Transport(format!("provider stream error: {msg}")),
                    protocol: true,
                };
            }

            if let Some(id) = extract_response_id(&value) {
                self.response_id = Some(id);
            }
            if value.get("type").and_then(Value::as_str) == Some("response.output_item.done") {
                if let Some(item) = value.get("item").filter(|i| is_replayable_output_item(i)) {
                    self.harvested.push(item.clone());
                }
            }

            let mut events = self.decoder.push(&value);
            if self.decoder.is_terminal() {
                self.finished = true;
                return match self.decoder.finish() {
                    Ok(mut trailing) => {
                        events.append(&mut trailing);
                        FrameOutcome::Events(events)
                    }
                    Err(e) => FrameOutcome::Failure {
                        error: e,
                        protocol: true,
                    },
                };
            }
            if !events.is_empty() {
                return FrameOutcome::Events(events);
            }
            // Nothing decoded from this frame — keep reading. In practice this never loops more than
            // once: the decoder's very first `push` always yields at least `StreamEvent::MessageStart`
            // regardless of the frame's own type.
        }
    }

    /// The socket ended (a `Close` frame or the stream itself returning `None`) without
    /// `next_events` already having reached its terminal event.
    fn closed_outcome(&self) -> FrameOutcome {
        if self.finished {
            FrameOutcome::Done
        } else {
            FrameOutcome::Failure {
                error: Error::Transport(format!(
                    "{MID_STREAM_NETWORK_ERROR}: codex websocket closed before a terminal response event"
                )),
                protocol: false,
            }
        }
    }
}

// ============================================================================
// Orchestration
// ============================================================================

/// The result of one `try_stream` call — the decision `client.rs::GatewayClient::stream` acts on.
pub(crate) enum Attempt {
    /// The WebSocket turn started (its first event has already been decoded); the returned stream
    /// continues it to completion, updating the connection cache on a clean finish.
    Started(EventStream),
    /// Nothing streamed yet; the caller should fall through to the existing HTTP/SSE path for this
    /// turn unchanged.
    Fallback,
    /// A genuine, non-fallback-eligible failure (a protocol/API error) — propagate directly, matching
    /// what the HTTP/SSE path would have surfaced for the same in-band failure.
    Failed(Error),
}

enum AttemptOnceOutcome {
    Started {
        first_events: Vec<StreamEvent>,
        // Boxed: `Receiver` embeds the full `WsSocket` (a TLS-capable stream, ~1.5KB), which would
        // otherwise force every other (tiny) variant of this enum to pay for that size too.
        receiver: Box<Receiver>,
    },
    RetryConnectionLimit,
    RetryStaleReuse,
    Fallback,
    Failed(Error),
}

/// One connect-(or-reuse)-and-send-and-await-first-event pass. `prefer_cache: false` forces a fresh
/// connection even when a cached one might be available — used for both retry kinds `try_stream`
/// drives this with.
#[allow(clippy::too_many_arguments)]
async fn attempt_once(
    cache: &CodexWebSocketCache,
    cache_key: Option<&str>,
    url: &str,
    headers: &[(String, String)],
    full_body: &Value,
    prefer_cache: bool,
    retried_connection_limit: &mut bool,
) -> AttemptOnceOutcome {
    let checked_out = if prefer_cache {
        cache_key.and_then(|key| cache.checkout(key))
    } else {
        None
    };
    let (mut socket, continuation, reused) = match checked_out {
        Some(entry) => (entry.socket, entry.continuation, true),
        None => match connect(url, headers).await {
            Ok(socket) => (socket, None, false),
            Err(ConnectFailure::ConnectionLimit(msg)) if !*retried_connection_limit => {
                *retried_connection_limit = true;
                tracing::debug!(
                    error = %msg,
                    "codex websocket connection-limit signal on connect; retrying once"
                );
                return AttemptOnceOutcome::RetryConnectionLimit;
            }
            Err(ConnectFailure::ConnectionLimit(msg)) | Err(ConnectFailure::Other(msg)) => {
                tracing::debug!(error = %msg, "codex websocket unavailable; using SSE for this turn");
                return AttemptOnceOutcome::Fallback;
            }
        },
    };

    let wire_body = build_wire_body(full_body, continuation.as_ref());
    if let Err(e) = socket.send(Message::text(wire_frame(&wire_body))).await {
        tracing::debug!(error = %e, "codex websocket send failed; using SSE for this turn");
        return AttemptOnceOutcome::Fallback;
    }

    let mut receiver = Receiver::new(socket);
    match receiver.next_events().await {
        FrameOutcome::Events(events) => AttemptOnceOutcome::Started {
            first_events: events,
            receiver: Box::new(receiver),
        },
        FrameOutcome::Failure {
            error,
            protocol: true,
        } => AttemptOnceOutcome::Failed(error),
        FrameOutcome::Failure {
            error,
            protocol: false,
        } => {
            tracing::debug!(error = %error, reused, "codex websocket failed before streaming any events");
            if reused {
                AttemptOnceOutcome::RetryStaleReuse
            } else {
                AttemptOnceOutcome::Fallback
            }
        }
        // Can't actually happen (reaching a clean `Done` requires having already decoded at least one
        // frame, which — via the decoder's own forced `MessageStart` on its first `push` — always
        // yields at least one event first) but handled explicitly rather than assumed: falling back is
        // always a safe, correct response to "nothing usable came out of this attempt".
        FrameOutcome::Done => AttemptOnceOutcome::Fallback,
    }
}

/// Attempt the Codex WebSocket transport for one turn. See the module doc comment for the overall
/// design; `client.rs::GatewayClient::stream` calls this once, before it would otherwise build the
/// existing HTTP/SSE request, and acts on the returned [`Attempt`].
pub(crate) async fn try_stream(
    cache: Arc<CodexWebSocketCache>,
    ws_url: String,
    headers: Vec<(String, String)>,
    cache_key: Option<String>,
    full_body: Value,
) -> Attempt {
    if let Some(key) = &cache_key {
        if cache.is_fallback_active(key) {
            return Attempt::Fallback;
        }
    }

    let fingerprint = body_fingerprint(&full_body);
    let full_input = input_items(&full_body);

    let mut prefer_cache = true;
    let mut retried_connection_limit = false;

    // At most: one ordinary attempt, one connection-limit retry, one stale-reused-connection retry.
    // Bounded higher than the realistic max (2) purely as a defensive cap against a future change to
    // this loop accidentally looping forever.
    for _ in 0..4 {
        match attempt_once(
            &cache,
            cache_key.as_deref(),
            &ws_url,
            &headers,
            &full_body,
            prefer_cache,
            &mut retried_connection_limit,
        )
        .await
        {
            AttemptOnceOutcome::Started {
                first_events,
                receiver,
            } => {
                return Attempt::Started(build_event_stream(
                    cache,
                    cache_key,
                    fingerprint,
                    full_input,
                    first_events,
                    receiver,
                ));
            }
            AttemptOnceOutcome::RetryConnectionLimit => {
                prefer_cache = false;
                continue;
            }
            AttemptOnceOutcome::RetryStaleReuse => {
                prefer_cache = false;
                continue;
            }
            AttemptOnceOutcome::Fallback => {
                if let Some(key) = &cache_key {
                    cache.mark_fallback(key);
                }
                return Attempt::Fallback;
            }
            AttemptOnceOutcome::Failed(e) => return Attempt::Failed(e),
        }
    }

    if let Some(key) = &cache_key {
        cache.mark_fallback(key);
    }
    Attempt::Fallback
}

/// Build the `'static` event stream continuing a turn whose first event already decoded cleanly —
/// yields `first_events`, then keeps decoding frames via `receiver` until a clean finish (checking the
/// connection back in for reuse, with a fresh [`Continuation`]) or a failure (yielded as the stream's
/// last item, matching the existing HTTP/SSE path's own mid-stream-failure shape).
fn build_event_stream(
    cache: Arc<CodexWebSocketCache>,
    cache_key: Option<String>,
    fingerprint: Value,
    full_input: Vec<Value>,
    first_events: Vec<StreamEvent>,
    mut receiver: Box<Receiver>,
) -> EventStream {
    let stream = async_stream::stream! {
        for ev in first_events {
            yield Ok(ev);
        }
        loop {
            match receiver.next_events().await {
                FrameOutcome::Events(events) => {
                    for ev in events {
                        yield Ok(ev);
                    }
                }
                FrameOutcome::Done => {
                    if let (Some(key), Some(id)) = (cache_key.clone(), receiver.response_id.take()) {
                        let mut baseline = full_input.clone();
                        baseline.append(&mut receiver.harvested);
                        let Receiver { socket, .. } = *receiver;
                        cache.checkin(
                            key,
                            CachedConnection {
                                socket,
                                created_at: Instant::now(),
                                continuation: Some(Continuation {
                                    fingerprint: fingerprint.clone(),
                                    baseline_input: baseline,
                                    last_response_id: id,
                                }),
                            },
                        );
                    }
                    return;
                }
                FrameOutcome::Failure { error, .. } => {
                    yield Err(error);
                    return;
                }
            }
        }
    };
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn continuation(
        fingerprint: Value,
        baseline_input: Vec<Value>,
        last_response_id: &str,
    ) -> Continuation {
        Continuation {
            fingerprint,
            baseline_input,
            last_response_id: last_response_id.to_string(),
        }
    }

    #[test]
    fn first_turn_with_no_continuation_sends_the_full_body_untrimmed() {
        let body = json!({
            "model": "gpt-5-codex",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        });
        let wire = build_wire_body(&body, None);
        assert_eq!(wire, body, "no continuation yet — nothing to trim");
        assert!(wire.get("previous_response_id").is_none());
    }

    #[test]
    fn second_turn_sends_only_the_new_input_items_with_previous_response_id() {
        let first_user = json!({"type": "message", "role": "user", "content": "hi"});
        let assistant_reply = json!({"type": "message", "role": "assistant", "id": "msg_1", "status": "completed", "content": [{"type": "output_text", "text": "hello"}]});
        let second_user = json!({"type": "message", "role": "user", "content": "how are you"});

        let first_body = json!({
            "model": "gpt-5-codex",
            "input": [first_user.clone()],
        });
        let cont = continuation(
            body_fingerprint(&first_body),
            vec![first_user.clone(), assistant_reply.clone()],
            "resp_1",
        );

        let second_body = json!({
            "model": "gpt-5-codex",
            "input": [first_user, assistant_reply, second_user.clone()],
        });

        let wire = build_wire_body(&second_body, Some(&cont));
        assert_eq!(
            wire["previous_response_id"], "resp_1",
            "a matching prefix must chain off the cached response id"
        );
        assert_eq!(
            wire["input"],
            json!([second_user]),
            "the delta must be exactly the new tail, not the whole transcript: {wire:#?}"
        );
    }

    #[test]
    fn a_shape_change_outside_input_forces_a_full_resend() {
        // Changing anything but `input`/`previous_response_id`/`max_output_tokens` (a different
        // system prompt, a different tool set, …) must not silently chain off a stale
        // `previous_response_id` — pi's own `requestBodiesMatchExceptInput` guard.
        let first_body =
            json!({"model": "gpt-5-codex", "instructions": "be terse", "input": [{"a": 1}]});
        let cont = continuation(
            body_fingerprint(&first_body),
            vec![json!({"a": 1})],
            "resp_1",
        );

        let second_body = json!({"model": "gpt-5-codex", "instructions": "be verbose", "input": [{"a": 1}, {"b": 2}]});
        let wire = build_wire_body(&second_body, Some(&cont));
        assert_eq!(
            wire, second_body,
            "a fingerprint mismatch must force a full, untrimmed resend"
        );
        assert!(wire.get("previous_response_id").is_none());
    }

    #[test]
    fn max_output_tokens_drifting_alone_does_not_defeat_the_delta() {
        // beyond-only field (see the module doc comment) — must NOT be part of the "did the request
        // shape change" comparison, unlike every other field.
        let first_body =
            json!({"model": "gpt-5-codex", "max_output_tokens": 4000, "input": [{"a": 1}]});
        let cont = continuation(
            body_fingerprint(&first_body),
            vec![json!({"a": 1})],
            "resp_1",
        );

        let second_body = json!({"model": "gpt-5-codex", "max_output_tokens": 3500, "input": [{"a": 1}, {"b": 2}]});
        let wire = build_wire_body(&second_body, Some(&cont));
        assert_eq!(wire["previous_response_id"], "resp_1");
        assert_eq!(wire["input"], json!([{"b": 2}]));
        // The trimmed body still carries this turn's own (possibly-drifted) max_output_tokens, not a
        // stale cached one.
        assert_eq!(wire["max_output_tokens"], 3500);
    }

    #[test]
    fn a_shrunk_input_forces_a_full_resend() {
        // A shorter `input` than the cached baseline (e.g. after a compaction rewrote history) can't
        // possibly still start with that baseline — pi's own `currentInput.length < baseline.length`
        // guard.
        let first_body = json!({"model": "gpt-5-codex", "input": [{"a": 1}, {"b": 2}]});
        let cont = continuation(
            body_fingerprint(&first_body),
            vec![json!({"a": 1}), json!({"b": 2})],
            "resp_1",
        );
        let second_body = json!({"model": "gpt-5-codex", "input": [{"a": 1}]});
        let wire = build_wire_body(&second_body, Some(&cont));
        assert_eq!(wire, second_body);
        assert!(wire.get("previous_response_id").is_none());
    }

    #[test]
    fn a_diverged_prefix_forces_a_full_resend() {
        // Same length-and-later prefix as the baseline, but the baseline's own items don't match
        // byte-for-byte (a hand-edited session, a branch switch) — must not chain off a
        // `previous_response_id` whose actual server-side state no longer matches what's about to be
        // sent.
        let first_body = json!({"model": "gpt-5-codex", "input": [{"a": 1}]});
        let cont = continuation(
            body_fingerprint(&first_body),
            vec![json!({"a": 999})],
            "resp_1",
        );
        let second_body = json!({"model": "gpt-5-codex", "input": [{"a": 1}, {"b": 2}]});
        let wire = build_wire_body(&second_body, Some(&cont));
        assert_eq!(wire, second_body);
        assert!(wire.get("previous_response_id").is_none());
    }

    #[test]
    fn an_empty_cached_response_id_forces_a_full_resend() {
        let first_body = json!({"model": "gpt-5-codex", "input": [{"a": 1}]});
        let cont = continuation(body_fingerprint(&first_body), vec![json!({"a": 1})], "");
        let second_body = json!({"model": "gpt-5-codex", "input": [{"a": 1}, {"b": 2}]});
        let wire = build_wire_body(&second_body, Some(&cont));
        assert_eq!(wire, second_body);
    }

    #[test]
    fn is_replayable_output_item_matches_message_reasoning_and_function_call_only() {
        for kind in ["message", "reasoning", "function_call"] {
            assert!(is_replayable_output_item(&json!({"type": kind})), "{kind}");
        }
        for kind in ["function_call_output", "some_future_item_type"] {
            assert!(!is_replayable_output_item(&json!({"type": kind})), "{kind}");
        }
        assert!(!is_replayable_output_item(&json!({})));
    }

    #[test]
    fn extract_response_id_reads_the_nested_response_object() {
        assert_eq!(
            extract_response_id(
                &json!({"type": "response.created", "response": {"id": "resp_42"}})
            ),
            Some("resp_42".to_string())
        );
        assert_eq!(
            extract_response_id(&json!({"type": "response.output_text.delta"})),
            None
        );
        assert_eq!(extract_response_id(&json!({"response": {}})), None);
    }

    #[test]
    fn to_ws_url_rewrites_http_and_https_schemes() {
        assert_eq!(
            to_ws_url("https://chatgpt.com/backend-api/codex/responses"),
            Some("wss://chatgpt.com/backend-api/codex/responses".to_string())
        );
        assert_eq!(
            to_ws_url("http://127.0.0.1:8080/openai-codex/backend-api/codex/responses"),
            Some("ws://127.0.0.1:8080/openai-codex/backend-api/codex/responses".to_string())
        );
        assert_eq!(to_ws_url("ftp://example.com"), None);
    }

    #[test]
    fn is_connection_limit_signal_recognizes_the_backend_code_and_a_bare_429() {
        assert!(is_connection_limit_signal(429, ""));
        assert!(is_connection_limit_signal(
            409,
            r#"{"error":{"code":"websocket_connection_limit_reached"}}"#
        ));
        assert!(!is_connection_limit_signal(500, "internal error"));
    }

    #[test]
    fn build_headers_carries_direct_headers_and_the_websocket_specific_ones() {
        let direct = vec![("chatgpt-account-id", "acct_123".to_string())];
        let headers = build_headers("sk-test", &direct, "req-1");
        let get = |name: &str| {
            headers
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get("Authorization"), Some("Bearer sk-test"));
        assert_eq!(get("chatgpt-account-id"), Some("acct_123"));
        assert_eq!(get("OpenAI-Beta"), Some(OPENAI_BETA_RESPONSES_WEBSOCKETS));
        assert_eq!(get("x-client-request-id"), Some("req-1"));
        assert_eq!(get("session-id"), Some("req-1"));
    }

    #[test]
    fn compress_sse_fallback_body_round_trips_through_zstd() {
        let json = br#"{"model":"gpt-5-codex","input":[{"role":"user","content":"hi"}]}"#;
        let compressed =
            compress_sse_fallback_body(json).expect("zstd compression always succeeds here");
        assert_ne!(compressed, json);
        let decompressed =
            zstd::stream::decode_all(&compressed[..]).expect("a valid zstd frame decodes cleanly");
        assert_eq!(decompressed, json);
    }

    #[test]
    fn compress_sse_fallback_body_shrinks_a_realistic_repetitive_payload() {
        // A real Codex request body (JSON, repeated keys/structure across many input items) compresses
        // well — this is the whole point of the optimization. A single short, low-entropy fixture like
        // the round-trip test's above isn't representative of real bandwidth savings, so this uses a
        // larger, more realistic shape instead.
        let item = r#"{"role":"user","content":[{"type":"input_text","text":"repeat this text over and over"}]},"#;
        let json = format!(
            r#"{{"model":"gpt-5-codex","input":[{}]}}"#,
            item.repeat(200)
        );
        let compressed = compress_sse_fallback_body(json.as_bytes())
            .expect("zstd compression always succeeds here");
        assert!(
            compressed.len() < json.len() / 2,
            "expected the repetitive fixture to compress to under half its size, got {} of {} bytes",
            compressed.len(),
            json.len()
        );
    }

    #[test]
    fn wire_frame_wraps_the_body_in_a_response_create_envelope() {
        let body = json!({"model": "gpt-5-codex", "input": []});
        let frame = wire_frame(&body);
        let parsed: Value = serde_json::from_str(&frame).unwrap();
        assert_eq!(parsed["type"], "response.create");
        assert_eq!(parsed["model"], "gpt-5-codex");
    }

    #[test]
    fn cache_fallback_marker_is_per_key_and_starts_unset() {
        // The cache's checkin/checkout round-trip (with a *real* socket) is covered by the
        // mock-server integration tests in `tests/codex_websocket_socket.rs`; this is pure
        // bookkeeping-only coverage for the sticky-fallback marker, which doesn't need a socket at all.
        let cache = CodexWebSocketCache::new();
        assert!(cache.checkout("s1").is_none());
        assert!(!cache.is_fallback_active("s1"));
        cache.mark_fallback("s1");
        assert!(cache.is_fallback_active("s1"));
        assert!(
            !cache.is_fallback_active("s2"),
            "the marker must not leak across keys"
        );
    }
}
