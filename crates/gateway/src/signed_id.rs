//! Tenant-bound ids for the state a provider keeps under Beyond's pool key (D230, D231).
//!
//! Every managed tenant reaches OpenAI through the same pool key, so every response OpenAI stores
//! (`store` defaults to true) lives in one organization: its response id (`resp_…`), its
//! conversation id (`conv_…`) and its output item ids (`msg_…`, `rs_…`, `fc_…`, …). OpenAI resolves
//! any of them for anyone holding the key: `previous_response_id` and `conversation` continue the
//! conversation, an `item_reference` expands the item, and so does a *full* input item that carries
//! a stored item's `id` (OpenAI answers from the stored content, not the content sent: D231). A
//! tenant holding another tenant's id (a log, a ticket, a client bug) could read and continue that
//! tenant's conversation. OpenAI isolates organizations; this module isolates tenants.
//!
//! The gateway stays stateless. On a managed Responses relay to a store (a GPT row's Responses
//! arm, a managed `/{provider}/…/responses`), every provider id in a 2xx response is rewritten to
//! a **signed id** bound to the calling tenant ([`Relay`]), and every id the client sends back
//! (`previous_response_id`, `conversation`, each `input` item's `id`) must verify for that same
//! tenant and is stripped back to the provider's id before the body leaves
//! ([`Signer::unsign_request`]). An unsigned, foreign or altered id is a 400 before any upstream is
//! contacted. BYO keys are the caller's own organization and are never touched.
//!
//! **Format.** A signed id keeps the provider id's prefix (everything up to and including its
//! first `_`), so SDK schemas and prefix checks still pass. When the rest is lowercase hex (every
//! OpenAI id), it is packed:
//!
//! ```text
//! signed = prefix "x" kid base64url(hex_decode(rest) || tag)      e.g. resp_x1<55 chars>
//! ```
//!
//! and any other id is kept verbatim with the tag appended:
//!
//! ```text
//! signed = provider_id "_v" kid base64url(tag)                    (25 chars longer)
//! ```
//!
//! where `tag = HMAC-SHA256(secret[kid], "beyond-ai/signed-id/v1\0" || kid || tenant_id_le64 ||
//! provider_id)[..16]` and `kid` is one character `[0-9A-Za-z]`. Only base64url and the provider's
//! own characters appear, so the id is URL-safe whenever the provider's is. OpenAI caps every id
//! it accepts at 64 characters (`previous_response_id`, `conversation`, `input[].id`; measured
//! 2026-10-01: "Expected a string with maximum length 64"), and its ids are a prefix plus 50 hex
//! digits, so a packed id stays under that cap (`resp_` 55 → 62, `msg_` 54 → 61, `rs_` 53 → 60):
//! a client that sizes ids by OpenAI's own limit keeps working. The SDKs constrain nothing
//! (openai-python, openai-node and the AI SDK type ids as plain strings).
//!
//! **Margin.** The tag is 128 bits: half of SHA-256's output, the truncation RFC 2104 §5
//! recommends (no less than half the hash and no less than 80 bits). A forgery is an online guess
//! through the gateway, which rate-limits each credential, at 2^-128 per attempt; nothing about
//! the tag shortens that. The tenant id is inside the MAC, so tenant B cannot present tenant A's
//! signed id, nor strip it to the raw id: a raw id never verifies.
//!
//! **Rotation.** `id_signing_keys` maps kid → secret; `id_signing_kid` names the one that signs
//! new ids. Verification picks the key by the kid inside the id, so ids issued under a previous
//! key keep verifying while it stays configured. Add the new key everywhere first, then switch
//! `id_signing_kid`, and keep the old key at least as long as OpenAI keeps the responses (30 days).
//!
//! **Translated rows** (Claude, grok, …) have no store behind them: a translated response's ids are
//! the gateway's or another vendor's, nothing upstream can resolve them, and grok goes to xAI as
//! `store: false` (D145). They are not signed and not checked; D175's item_reference rule applies
//! there unchanged.

use std::borrow::Cow;
use std::collections::HashMap;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::hmac;

use crate::peek::{self, Member};
use crate::route::{self, ModelRoute};
use crate::secret::Secret;

/// Bytes of HMAC-SHA256 kept in a signed id (128 bits; see the module docs for the margin).
pub const TAG_LEN: usize = 16;
/// Shortest accepted secret, decoded: one SHA-256 block of key material.
pub const MIN_SECRET_LEN: usize = 32;
/// A non-streaming 2xx body held whole for signing, at most (the translated-body bound).
pub const MAX_JSON_BODY: usize = crate::translate::MAX_TRANSLATE_BUFFER;
/// Domain separation: no other HMAC the gateway computes can collide with a signed id's.
const CONTEXT: &[u8] = b"beyond-ai/signed-id/v1\0";
/// Marks a packed id, right after the provider's prefix. Never a hex digit.
const PACKED: u8 = b'x';
/// Marks a verbatim id's tail.
const VERBATIM: &[u8] = b"_v";
/// `"_v"`, the kid, and the 22-char base64url tag.
const VERBATIM_TAIL: usize = VERBATIM.len() + 1 + 22;
/// Largest provider-id body (decoded hex bytes) that is packed; anything longer goes verbatim.
const MAX_PACKED: usize = 32;
/// The gate on every response line: a key ending `id"` (`id`, `item_id`, `previous_response_id`).
/// Built once; a one-shot `memmem::find` rebuilds its searcher per call.
static ID_KEY: std::sync::LazyLock<memchr::memmem::Finder<'static>> =
    std::sync::LazyLock::new(|| memchr::memmem::Finder::new(b"id\""));

/// What a refused request is told when the refusal happens after the request headers left (a
/// `/{provider}` body that streamed in). A catalog walk is refused before connecting, naming the
/// field ([`Refusal`]).
pub const REFUSED: &str = "an id in this request does not belong to this tenant: send back only ids from responses this gateway returned to the same tenant";

/// The keys that sign and verify tenant-bound ids.
pub struct Signer {
    current: u8,
    keys: Vec<(u8, hmac::Key)>,
}

impl std::fmt::Debug for Signer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kids: String = self.keys.iter().map(|(k, _)| char::from(*k)).collect();
        f.debug_struct("Signer")
            .field("current", &char::from(self.current))
            .field("kids", &kids)
            .finish()
    }
}

/// An id the caller sent that does not verify for it. `field` names where it was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub field: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} does not belong to this tenant: send back only ids from responses this gateway returned to the same tenant (an id from another tenant, from the provider directly, or altered, is refused)",
            self.field
        )
    }
}

impl Signer {
    /// From raw secrets by kid. `current` signs new ids.
    pub fn new(keys: &[(u8, &[u8])], current: u8) -> Result<Self, String> {
        let mut out = Vec::with_capacity(keys.len());
        for &(kid, secret) in keys {
            if !kid.is_ascii_alphanumeric() {
                return Err(format!(
                    "id signing key id {:?} must be one character [0-9A-Za-z]",
                    char::from(kid)
                ));
            }
            if secret.len() < MIN_SECRET_LEN {
                return Err(format!(
                    "id signing key {} is {} bytes; at least {MIN_SECRET_LEN} random bytes are required (openssl rand -base64 32)",
                    char::from(kid),
                    secret.len()
                ));
            }
            if out.iter().any(|(k, _)| *k == kid) {
                return Err(format!("id signing key {} is set twice", char::from(kid)));
            }
            out.push((kid, hmac::Key::new(hmac::HMAC_SHA256, secret)));
        }
        if !out.iter().any(|(k, _)| *k == current) {
            return Err(format!(
                "id_signing_kid {:?} names no configured id signing key",
                char::from(current)
            ));
        }
        Ok(Self { current, keys: out })
    }

    /// From config: `id_signing_keys` (kid → base64 secret) and `id_signing_kid`. `Ok(None)` when no
    /// key is configured, which fails managed Responses relays to a store closed (503).
    pub fn from_config(
        keys: &HashMap<String, Secret>,
        current: &str,
    ) -> Result<Option<Self>, String> {
        if keys.is_empty() {
            return if current.is_empty() {
                Ok(None)
            } else {
                Err(format!(
                    "id_signing_kid = {current:?} but no id_signing_keys are configured"
                ))
            };
        }
        let mut decoded: Vec<(u8, zeroize::Zeroizing<Vec<u8>>)> = Vec::with_capacity(keys.len());
        for (kid, secret) in keys {
            let &[k] = kid.as_bytes() else {
                return Err(format!(
                    "id signing key id {kid:?} must be one character [0-9A-Za-z]"
                ));
            };
            let bytes = decode_secret(secret.expose()).ok_or_else(|| {
                format!(
                    "id signing key {kid} is not base64 (generate one with openssl rand -base64 32)"
                )
            })?;
            decoded.push((k, zeroize::Zeroizing::new(bytes)));
        }
        let current = match current.as_bytes() {
            [k] => *k,
            [] if decoded.len() == 1 => decoded[0].0,
            [] => {
                return Err(
                    "more than one id signing key is configured: set id_signing_kid to the one that signs new ids"
                        .to_owned(),
                );
            }
            _ => {
                return Err(format!(
                    "id_signing_kid {current:?} must be one character [0-9A-Za-z]"
                ));
            }
        };
        let refs: Vec<(u8, &[u8])> = decoded.iter().map(|(k, s)| (*k, s.as_slice())).collect();
        Self::new(&refs, current).map(Some)
    }

    fn key(&self, kid: u8) -> Option<&hmac::Key> {
        self.keys
            .iter()
            .find(|(k, _)| *k == kid)
            .map(|(_, key)| key)
    }

    fn tag(key: &hmac::Key, kid: u8, tenant: u64, raw: &[u8]) -> [u8; TAG_LEN] {
        let mut ctx = hmac::Context::with_key(key);
        ctx.update(CONTEXT);
        ctx.update(&[kid]);
        ctx.update(&tenant.to_le_bytes());
        ctx.update(raw);
        let full = ctx.sign();
        let mut tag = [0u8; TAG_LEN];
        tag.copy_from_slice(&full.as_ref()[..TAG_LEN]);
        tag
    }

    /// Append the signed form of provider id `raw`, bound to `tenant`, to `out`.
    pub fn sign_into(&self, tenant: u64, raw: &str, out: &mut Vec<u8>) {
        let kid = self.current;
        let Some(key) = self.key(kid) else { return };
        let tag = Self::tag(key, kid, tenant, raw.as_bytes());
        let mut data = [0u8; MAX_PACKED + TAG_LEN];
        let mut b64 = [0u8; (MAX_PACKED + TAG_LEN).div_ceil(3) * 4];
        if let Some((prefix, hex)) = packable(raw) {
            let n = hex.len() / 2;
            for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
                data[i] = (nibble(pair[0]) << 4) | nibble(pair[1]);
            }
            data[n..n + TAG_LEN].copy_from_slice(&tag);
            let len = URL_SAFE_NO_PAD
                .encode_slice(&data[..n + TAG_LEN], &mut b64)
                .unwrap_or(0);
            out.extend_from_slice(prefix.as_bytes());
            out.push(PACKED);
            out.push(kid);
            out.extend_from_slice(&b64[..len]);
        } else {
            let len = URL_SAFE_NO_PAD.encode_slice(tag, &mut b64).unwrap_or(0);
            out.extend_from_slice(raw.as_bytes());
            out.extend_from_slice(VERBATIM);
            out.push(kid);
            out.extend_from_slice(&b64[..len]);
        }
    }

    /// The signed form of provider id `raw` for `tenant`.
    pub fn sign(&self, tenant: u64, raw: &str) -> String {
        let mut out = Vec::with_capacity(raw.len() + VERBATIM_TAIL);
        self.sign_into(tenant, raw, &mut out);
        // Only ASCII and `raw`'s own (UTF-8) bytes were written.
        String::from_utf8(out).unwrap_or_default()
    }

    /// The provider id inside `token`, when `token` is a signed id this gateway issued to `tenant`
    /// under a key it still holds. `None` for anything else: a raw provider id, another tenant's
    /// signed id, an altered one, or one signed under a key since removed.
    pub fn verify(&self, tenant: u64, token: &str) -> Option<String> {
        self.verify_packed(tenant, token)
            .or_else(|| self.verify_verbatim(tenant, token))
    }

    fn verify_packed(&self, tenant: u64, token: &str) -> Option<String> {
        let b = token.as_bytes();
        let us = memchr::memchr(b'_', b)?;
        let rest = &b[us + 1..];
        if rest.first() != Some(&PACKED) || rest.len() < 3 {
            return None;
        }
        let kid = rest[1];
        let key = self.key(kid)?;
        let mut data = [0u8; MAX_PACKED + TAG_LEN + 3];
        let n = URL_SAFE_NO_PAD.decode_slice(&rest[2..], &mut data).ok()?;
        if n <= TAG_LEN || n > MAX_PACKED + TAG_LEN {
            return None;
        }
        let body = n - TAG_LEN;
        let mut raw = String::with_capacity(us + 1 + body * 2);
        raw.push_str(&token[..=us]);
        for &byte in &data[..body] {
            raw.push(char::from(HEX[usize::from(byte >> 4)]));
            raw.push(char::from(HEX[usize::from(byte & 0xf)]));
        }
        let want = Self::tag(key, kid, tenant, raw.as_bytes());
        ct_eq(&want, &data[body..n]).then_some(raw)
    }

    fn verify_verbatim(&self, tenant: u64, token: &str) -> Option<String> {
        let b = token.as_bytes();
        let split = b.len().checked_sub(VERBATIM_TAIL).filter(|&s| s > 0)?;
        let tail = &b[split..];
        if !tail.starts_with(VERBATIM) || !token.is_char_boundary(split) {
            return None;
        }
        let kid = tail[2];
        let key = self.key(kid)?;
        let mut tag = [0u8; TAG_LEN + 2];
        let n = URL_SAFE_NO_PAD.decode_slice(&tail[3..], &mut tag).ok()?;
        if n != TAG_LEN {
            return None;
        }
        let raw = &token[..split];
        let want = Self::tag(key, kid, tenant, raw.as_bytes());
        ct_eq(&want, &tag[..TAG_LEN]).then(|| raw.to_owned())
    }

    /// Check every id in a managed Responses request body and strip each back to its provider id:
    /// root `previous_response_id`, root `conversation` (a string, or an object's `id`), and the
    /// `id` of every `input` item (an `item_reference`, or a full item: OpenAI answers a full item
    /// carrying a stored item's id from the store, D231). Every occurrence of each key is checked,
    /// whatever its spelling (escapes included), since the provider's parser decodes them all.
    ///
    /// `catalog`: the body is a catalog walk's, whose relay cuts the reasoning items the gateway
    /// minted from Claude's thinking (`translate::strip_gateway_reasoning`, D50): they are cut here
    /// first, with that same function, so their `rs_gw…` ids are neither checked nor sent. A
    /// `/{provider}` relay cuts nothing, so there every id is checked.
    ///
    /// `Ok(None)`: nothing to rewrite. `Ok(Some(body))`: the body to send. `Err`: an id that does
    /// not verify for `tenant`, or a body the walk cannot read (the provider would read it some
    /// other way, and an unread id must not pass).
    pub fn unsign_request(
        &self,
        tenant: u64,
        body: &[u8],
        catalog: bool,
    ) -> Result<Option<Vec<u8>>, Refusal> {
        if catalog && memchr::memmem::find(body, b"rs_gw").is_some() {
            let stripped = crate::translate::strip_gateway_reasoning(body.to_vec());
            if stripped.len() != body.len() {
                return Ok(Some(
                    self.unsign_request(tenant, &stripped, false)?
                        .unwrap_or(stripped),
                ));
            }
        }
        // An id key is spelled `…id"` unless escaped (`\u0069d`), and `conversation"` likewise.
        if memchr::memmem::find(body, b"id\"").is_none()
            && memchr::memmem::find(body, b"conversation\"").is_none()
            && memchr::memmem::find(body, b"\\u").is_none()
        {
            return Ok(None);
        }
        let unreadable = || Refusal {
            field: "the request body".to_owned(),
        };
        let open = skip_ws(body, 0);
        if body.get(open) != Some(&b'{') {
            return Err(unreadable());
        }
        let mut edits = Edits::default();
        // The first id that failed, by where it was.
        let mut fail: Option<String> = None;
        let ok = each_member(body, open, |m| {
            let v = m.value;
            if m.key_is(body, "previous_response_id") {
                if body[v.0] == b'"' && !self.unsign_at(tenant, body, v, &mut edits) {
                    fail.get_or_insert_with(|| "previous_response_id".to_owned());
                }
            } else if m.key_is(body, "conversation") {
                match body[v.0] {
                    b'"' => {
                        if !self.unsign_at(tenant, body, v, &mut edits) {
                            fail.get_or_insert_with(|| "conversation".to_owned());
                        }
                    }
                    b'{' => {
                        let ok = each_member(body, v.0, |c| {
                            if c.key_is(body, "id")
                                && body[c.value.0] == b'"'
                                && !self.unsign_at(tenant, body, c.value, &mut edits)
                            {
                                fail.get_or_insert_with(|| "conversation.id".to_owned());
                            }
                        });
                        if !ok {
                            fail.get_or_insert_with(|| "conversation".to_owned());
                        }
                    }
                    _ => {}
                }
            } else if m.key_is(body, "input") && body[v.0] == b'[' {
                let mut n = 0usize;
                let ok = each_element(body, v.0, |(s, _)| {
                    let i = n;
                    n += 1;
                    if body[s] != b'{' {
                        return;
                    }
                    let ok = each_member(body, s, |c| {
                        if c.key_is(body, "id")
                            && body[c.value.0] == b'"'
                            && !self.unsign_at(tenant, body, c.value, &mut edits)
                        {
                            fail.get_or_insert_with(|| format!("input[{i}].id"));
                        }
                    });
                    if !ok {
                        fail.get_or_insert_with(|| format!("input[{i}]"));
                    }
                });
                if !ok {
                    fail.get_or_insert_with(|| "input".to_owned());
                }
            }
        });
        if let Some(field) = fail {
            return Err(Refusal { field });
        }
        if !ok {
            return Err(unreadable());
        }
        Ok(edits.apply(body))
    }

    /// Verify the JSON string at `span` for `tenant` and queue its provider id in its place.
    fn unsign_at(&self, tenant: u64, body: &[u8], span: (usize, usize), edits: &mut Edits) -> bool {
        match json_str(body, span).and_then(|s| self.verify(tenant, &s)) {
            Some(raw) => {
                edits.push_str(span, &raw);
                true
            }
            None => false,
        }
    }
}

/// `(prefix through the first '_', rest)` when the rest is non-empty, even-length lowercase hex of
/// at most [`MAX_PACKED`] bytes: the shape every OpenAI id has, packed in half the characters.
fn packable(raw: &str) -> Option<(&str, &str)> {
    let us = raw.find('_')?;
    let (prefix, hex) = raw.split_at(us + 1);
    (!hex.is_empty()
        && hex.len() % 2 == 0
        && hex.len() <= MAX_PACKED * 2
        && hex.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')))
    .then_some((prefix, hex))
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        _ => c - b'a' + 10,
    }
}

/// Equal in time independent of where the bytes differ.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// A secret as base64, standard or URL-safe, padded or not.
fn decode_secret(s: &str) -> Option<Vec<u8>> {
    let t: String = s
        .trim()
        .trim_end_matches('=')
        .chars()
        .map(|c| match c {
            '+' => '-',
            '/' => '_',
            c => c,
        })
        .collect();
    URL_SAFE_NO_PAD.decode(t.as_bytes()).ok()
}

/// Whether a managed catalog walk relays the Responses wire to a provider's store: inbound
/// `/v1/responses` (or its `compact` / `input_tokens` sub-resources) on a row with a Responses arm,
/// which every such request walks (GPT rows). A row without one translates, or relays to xAI as
/// `store: false`, and nothing upstream holds its ids.
pub fn catalog_applies(path: &str, row: &ModelRoute) -> bool {
    !row.responses.is_empty()
        && (route::is_responses_path(path)
            || matches!(
                route::SubResource::of_path(path),
                Some(route::SubResource::InputTokens | route::SubResource::Compact)
            ))
}

/// Whether a managed `/{provider}/…` forwarded path (query allowed) is a Responses endpoint: a
/// byte relay on the pool key to whatever store that provider keeps.
pub fn provider_route_applies(path_and_query: &str) -> bool {
    let path = path_and_query
        .split_once('?')
        .map_or(path_and_query, |(p, _)| p);
    let path = path.strip_suffix('/').unwrap_or(path);
    path.ends_with("/responses")
        || path.ends_with("/responses/compact")
        || path.ends_with("/responses/input_tokens")
}

// --- response side ---------------------------------------------------------------------------

/// Signs the provider ids in one managed Responses relay's 2xx response as it streams to the
/// client: in a JSON body the response object's `id`, `previous_response_id`, `conversation` and
/// each `output` item's `id`; in an SSE stream the same under each event's `response`, plus
/// `item.id` and `item_id`. Every other byte is relayed as sent.
///
/// An SSE line is relayed once its newline arrives (a split line waits for the rest); a JSON body
/// is held whole, at most [`MAX_JSON_BODY`]. A data line without `id"` is copied, unparsed; one
/// with it gets one structural pass over its root members, and a memo of the last few ids means a
/// delta event pays no HMAC for an item already signed.
pub struct Relay {
    tenant: u64,
    mode: Mode,
    pending: Vec<u8>,
    edits: Edits,
    memo: Memo,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Json,
    Sse,
}

/// The response outgrew [`MAX_JSON_BODY`] (a JSON body, or one SSE line).
#[derive(Debug)]
pub struct Overflow;

impl Relay {
    pub fn new(tenant: u64) -> Self {
        Self {
            tenant,
            mode: Mode::Off,
            pending: Vec::new(),
            edits: Edits::default(),
            memo: Memo::default(),
        }
    }

    /// A response head arrived (once per attempt). Only a 2xx is signed: an error carries no
    /// stored state. `true` when the body will be rewritten, so its `Content-Length` must go.
    pub fn begin(&mut self, status: u16, streaming: bool) -> bool {
        self.pending.clear();
        self.mode = match status {
            200..=299 if streaming => Mode::Sse,
            200..=299 => Mode::Json,
            _ => Mode::Off,
        };
        self.mode != Mode::Off
    }

    /// Feed one upstream chunk. `Ok(None)`: relay it as is. `Ok(Some(out))`: relay `out` instead
    /// (possibly empty while a line or body is incomplete).
    pub fn feed(
        &mut self,
        signer: &Signer,
        chunk: &[u8],
        end_of_stream: bool,
    ) -> Result<Option<Vec<u8>>, Overflow> {
        match self.mode {
            Mode::Off => Ok(None),
            Mode::Json => {
                if self.pending.len().saturating_add(chunk.len()) > MAX_JSON_BODY {
                    return Err(Overflow);
                }
                self.pending.extend_from_slice(chunk);
                if !end_of_stream {
                    return Ok(Some(Vec::new()));
                }
                let body = std::mem::take(&mut self.pending);
                Ok(Some(self.sign_json(signer, body)))
            }
            Mode::Sse => {
                let mut out = Vec::with_capacity(chunk.len() + self.pending.len() + 64);
                if self.pending.is_empty() {
                    let cut = complete_lines(chunk, end_of_stream);
                    self.sign_lines(signer, &chunk[..cut], &mut out);
                    self.pending.extend_from_slice(&chunk[cut..]);
                } else {
                    let mut buf = std::mem::take(&mut self.pending);
                    buf.extend_from_slice(chunk);
                    let cut = complete_lines(&buf, end_of_stream);
                    self.sign_lines(signer, &buf[..cut], &mut out);
                    buf.drain(..cut);
                    self.pending = buf;
                }
                if self.pending.len() > MAX_JSON_BODY {
                    return Err(Overflow);
                }
                Ok(Some(out))
            }
        }
    }

    fn sign_json(&mut self, signer: &Signer, body: Vec<u8>) -> Vec<u8> {
        if ID_KEY.find(&body).is_none() {
            return body;
        }
        let open = skip_ws(&body, 0);
        if body.get(open) != Some(&b'{') {
            return body;
        }
        self.edits.clear();
        let (tenant, edits, memo) = (self.tenant, &mut self.edits, &mut self.memo);
        sign_response_object(signer, tenant, &body, open, edits, memo);
        edits.apply(&body).unwrap_or(body)
    }

    /// Every line of `lines` (each ending in `\n`, or the stream's last), signed where it is a
    /// `data:` line that names an id.
    fn sign_lines(&mut self, signer: &Signer, lines: &[u8], out: &mut Vec<u8>) {
        let mut at = 0;
        while at < lines.len() {
            let end = memchr::memchr(b'\n', &lines[at..]).map_or(lines.len(), |k| at + k + 1);
            let line = &lines[at..end];
            at = end;
            let Some(data) = line.strip_prefix(b"data:") else {
                out.extend_from_slice(line);
                continue;
            };
            let start = line.len() - data.len() + usize::from(data.first() == Some(&b' '));
            let json = &line[start..];
            if ID_KEY.find(json).is_none() {
                out.extend_from_slice(line);
                continue;
            }
            let open = skip_ws(json, 0);
            if json.get(open) != Some(&b'{') {
                out.extend_from_slice(line);
                continue;
            }
            self.edits.clear();
            let (tenant, edits, memo) = (self.tenant, &mut self.edits, &mut self.memo);
            each_member(json, open, |m| {
                let v = m.value;
                if m.key_is(json, "response") && json[v.0] == b'{' {
                    sign_response_object(signer, tenant, json, v.0, edits, memo);
                } else if m.key_is(json, "item") && json[v.0] == b'{' {
                    each_member(json, v.0, |c| {
                        if c.key_is(json, "id") {
                            sign_str(signer, tenant, json, c.value, edits, memo);
                        }
                    });
                } else if m.key_is(json, "item_id") {
                    sign_str(signer, tenant, json, v, edits, memo);
                }
            });
            out.extend_from_slice(&line[..start]);
            edits.write(json, out);
        }
    }
}

/// The ids of one Responses object (a body, or an event's `response`): its own, the response it
/// continued, its conversation, and each output item's.
fn sign_response_object(
    signer: &Signer,
    tenant: u64,
    b: &[u8],
    open: usize,
    edits: &mut Edits,
    memo: &mut Memo,
) {
    each_member(b, open, |m| {
        let v = m.value;
        if m.key_is(b, "id") || m.key_is(b, "previous_response_id") {
            sign_str(signer, tenant, b, v, edits, memo);
        } else if m.key_is(b, "conversation") {
            match b[v.0] {
                b'"' => sign_str(signer, tenant, b, v, edits, memo),
                b'{' => {
                    each_member(b, v.0, |c| {
                        if c.key_is(b, "id") {
                            sign_str(signer, tenant, b, c.value, edits, memo);
                        }
                    });
                }
                _ => {}
            }
        } else if m.key_is(b, "output") && b[v.0] == b'[' {
            each_element(b, v.0, |(s, _)| {
                if b[s] == b'{' {
                    each_member(b, s, |c| {
                        if c.key_is(b, "id") {
                            sign_str(signer, tenant, b, c.value, edits, memo);
                        }
                    });
                }
            });
        }
    });
}

/// Sign the string at `span` (a non-empty JSON string; anything else is left as sent).
fn sign_str(
    signer: &Signer,
    tenant: u64,
    b: &[u8],
    span: (usize, usize),
    edits: &mut Edits,
    memo: &mut Memo,
) {
    let Some(raw) = json_str(b, span).filter(|s| !s.is_empty()) else {
        return;
    };
    let start = edits.arena.len();
    edits.arena.push(b'"');
    if let Some(token) = memo.get(&raw) {
        edits.arena.extend_from_slice(token);
    } else {
        let at = edits.arena.len();
        signer.sign_into(tenant, &raw, &mut edits.arena);
        let token = edits.arena[at..].to_vec();
        memo.put(&raw, token);
    }
    if raw.bytes().any(|c| c == b'"' || c == b'\\' || c < 0x20) {
        // A provider id that needs escaping is never seen in practice; keep the JSON valid anyway.
        let token = String::from_utf8_lossy(&edits.arena[start + 1..]).into_owned();
        edits.arena.truncate(start);
        if let Ok(s) = serde_json::to_string(&token) {
            edits.arena.extend_from_slice(s.as_bytes());
        }
    } else {
        edits.arena.push(b'"');
    }
    edits.spans.push((span.0, span.1, start, edits.arena.len()));
}

/// The last few `(provider id, signed id)` pairs: an item's id repeats on every one of its delta
/// events, so the HMAC runs once per item.
#[derive(Default)]
struct Memo {
    entries: Vec<(Box<str>, Box<[u8]>)>,
    next: usize,
}

impl Memo {
    const CAP: usize = 8;

    fn get(&self, raw: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(r, _)| &**r == raw)
            .map(|(_, t)| &**t)
    }

    fn put(&mut self, raw: &str, token: Vec<u8>) {
        let entry = (Box::from(raw), token.into_boxed_slice());
        if self.entries.len() < Self::CAP {
            self.entries.push(entry);
        } else {
            self.entries[self.next] = entry;
            self.next = (self.next + 1) % Self::CAP;
        }
    }
}

/// Span replacements against one buffer: `(start, end)` of the source replaced by
/// `arena[a..b]`, in source order.
#[derive(Default)]
struct Edits {
    spans: Vec<(usize, usize, usize, usize)>,
    arena: Vec<u8>,
}

impl Edits {
    fn clear(&mut self) {
        self.spans.clear();
        self.arena.clear();
    }

    /// Replace `span` with `s` as a JSON string.
    fn push_str(&mut self, span: (usize, usize), s: &str) {
        let start = self.arena.len();
        match serde_json::to_string(s) {
            Ok(q) => self.arena.extend_from_slice(q.as_bytes()),
            Err(_) => return,
        }
        self.spans.push((span.0, span.1, start, self.arena.len()));
    }

    /// `src` with every edit applied, appended to `out`.
    fn write(&self, src: &[u8], out: &mut Vec<u8>) {
        let mut last = 0;
        for &(s, e, a, b) in &self.spans {
            if s < last {
                continue;
            }
            out.extend_from_slice(&src[last..s]);
            out.extend_from_slice(&self.arena[a..b]);
            last = e;
        }
        out.extend_from_slice(&src[last..]);
    }

    /// `src` with every edit applied, or `None` when there are none.
    fn apply(&self, src: &[u8]) -> Option<Vec<u8>> {
        if self.spans.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(src.len() + self.arena.len());
        self.write(src, &mut out);
        Some(out)
    }
}

/// How much of `buf` is whole lines: through its last `\n`, or all of it at the end of the stream.
fn complete_lines(buf: &[u8], end_of_stream: bool) -> usize {
    if end_of_stream {
        return buf.len();
    }
    memchr::memrchr(b'\n', buf).map_or(0, |i| i + 1)
}

// --- a non-allocating walk over JSON members, on `peek`'s span rules -------------------------------

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Call `f` with each member of the object whose `{` is at `open`, in order. `false` when the
/// object is malformed (members already visited were still passed to `f`).
fn each_member(b: &[u8], open: usize, mut f: impl FnMut(Member)) -> bool {
    let mut i = skip_ws(b, open + 1);
    if b.get(i) == Some(&b'}') {
        return true;
    }
    loop {
        if b.get(i) != Some(&b'"') {
            return false;
        }
        let Some(key_end) = peek::value_end(b, i) else {
            return false;
        };
        let key = (i + 1, key_end - 1);
        i = skip_ws(b, key_end);
        if b.get(i) != Some(&b':') {
            return false;
        }
        let start = skip_ws(b, i + 1);
        let Some(end) = peek::value_end(b, start) else {
            return false;
        };
        f(Member {
            key,
            value: (start, end),
        });
        i = skip_ws(b, end);
        match b.get(i) {
            Some(b',') => i = skip_ws(b, i + 1),
            Some(b'}') => return true,
            _ => return false,
        }
    }
}

/// Call `f` with each element span of the array whose `[` is at `open`. `false` when malformed.
fn each_element(b: &[u8], open: usize, mut f: impl FnMut((usize, usize))) -> bool {
    let mut i = skip_ws(b, open + 1);
    if b.get(i) == Some(&b']') {
        return true;
    }
    loop {
        let Some(end) = peek::value_end(b, i) else {
            return false;
        };
        f((i, end));
        i = skip_ws(b, end);
        match b.get(i) {
            Some(b',') => i = skip_ws(b, i + 1),
            Some(b']') => return true,
            _ => return false,
        }
    }
}

/// The decoded string at `span`, when it is a JSON string.
fn json_str(b: &[u8], span: (usize, usize)) -> Option<Cow<'_, str>> {
    let s = b.get(span.0..span.1)?;
    if s.len() < 2 || s[0] != b'"' || s[s.len() - 1] != b'"' {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    if memchr::memchr(b'\\', inner).is_none() {
        return std::str::from_utf8(inner).ok().map(Cow::Borrowed);
    }
    serde_json::from_slice::<String>(s).ok().map(Cow::Owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u64 = 42;
    const B: u64 = 43;
    const RESP: &str = "resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26";
    const MSG: &str = "msg_0750331520328311006abeead0270c87d0b8128a81f8474fea";

    fn signer() -> Signer {
        Signer::new(&[(b'1', &[7u8; 32])], b'1').unwrap()
    }

    #[test]
    fn a_signed_id_keeps_its_prefix_stays_url_safe_and_under_openais_cap() {
        let s = signer();
        for raw in [
            RESP,
            MSG,
            "rs_0750331520328311006abeead0270c87d0b8128a81f8474f",
            "conv_68c1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f7",
        ] {
            let t = s.sign(A, raw);
            let prefix = &raw[..=raw.find('_').unwrap()];
            assert!(t.starts_with(prefix), "{t}");
            assert!(t.len() <= 64, "{t} is {} chars", t.len());
            assert!(
                t.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
                "{t}"
            );
            assert_eq!(s.verify(A, &t).as_deref(), Some(raw));
        }
        // Not hex: kept verbatim with the tag appended.
        let t = s.sign(A, "msg_gw18f0c2a9b3d0001");
        assert!(t.starts_with("msg_gw18f0c2a9b3d0001_v1"), "{t}");
        assert_eq!(s.verify(A, &t).as_deref(), Some("msg_gw18f0c2a9b3d0001"));
    }

    #[test]
    fn another_tenants_a_raw_or_an_altered_id_never_verifies() {
        let s = signer();
        let t = s.sign(A, RESP);
        assert_eq!(s.verify(B, &t), None);
        assert_eq!(s.verify(A, RESP), None);
        let mut bytes = t.clone().into_bytes();
        for i in 0..bytes.len() {
            let orig = bytes[i];
            bytes[i] = if orig == b'A' { b'B' } else { b'A' };
            let altered = String::from_utf8(bytes.clone()).unwrap();
            if altered != t {
                assert_eq!(s.verify(A, &altered), None, "flipped byte {i}: {altered}");
            }
            bytes[i] = orig;
        }
        assert_eq!(s.verify(A, &t[..t.len() - 1]), None);
    }

    #[test]
    fn a_previous_key_still_verifies_and_a_removed_one_does_not() {
        let old = Signer::new(&[(b'1', &[7u8; 32])], b'1').unwrap();
        let t = old.sign(A, RESP);
        let rotated = Signer::new(&[(b'1', &[7u8; 32]), (b'2', &[9u8; 32])], b'2').unwrap();
        assert_eq!(rotated.verify(A, &t).as_deref(), Some(RESP));
        assert!(rotated.sign(A, RESP).starts_with("resp_x2"));
        let retired = Signer::new(&[(b'2', &[9u8; 32])], b'2').unwrap();
        assert_eq!(retired.verify(A, &t), None);
    }

    #[test]
    fn config_rejects_bad_kids_short_secrets_and_an_ambiguous_current_key() {
        let k = |pairs: &[(&str, &str)]| -> HashMap<String, Secret> {
            pairs
                .iter()
                .map(|(a, b)| ((*a).to_owned(), Secret::new(*b)))
                .collect()
        };
        let good = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        assert!(Signer::from_config(&k(&[]), "").unwrap().is_none());
        assert!(Signer::from_config(&k(&[]), "1").is_err());
        assert!(
            Signer::from_config(&k(&[("1", good)]), "")
                .unwrap()
                .is_some()
        );
        assert!(Signer::from_config(&k(&[("12", good)]), "").is_err());
        assert!(Signer::from_config(&k(&[("1", "c2hvcnQ=")]), "").is_err());
        assert!(Signer::from_config(&k(&[("1", "!!!")]), "").is_err());
        assert!(Signer::from_config(&k(&[("1", good), ("2", good)]), "").is_err());
        assert!(
            Signer::from_config(&k(&[("1", good), ("2", good)]), "2")
                .unwrap()
                .is_some()
        );
        assert!(Signer::from_config(&k(&[("1", good)]), "3").is_err());
    }

    #[test]
    fn a_request_is_stripped_to_provider_ids_and_a_foreign_one_is_named() {
        let s = signer();
        let (r, m) = (s.sign(A, RESP), s.sign(A, MSG));
        let body = format!(
            r#"{{"model":"gpt-4o","previous_response_id":"{r}","input":[{{"role":"user","content":"hi"}},{{"type":"item_reference","id":"{m}"}}],"conversation":{{"id":"{r}"}}}}"#
        );
        let out = String::from_utf8(
            s.unsign_request(A, body.as_bytes(), false)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            out,
            format!(
                r#"{{"model":"gpt-4o","previous_response_id":"{RESP}","input":[{{"role":"user","content":"hi"}},{{"type":"item_reference","id":"{MSG}"}}],"conversation":{{"id":"{RESP}"}}}}"#
            )
        );
        let err = s.unsign_request(B, body.as_bytes(), false).unwrap_err();
        assert_eq!(err.field, "previous_response_id");
        let raw = format!(
            r#"{{"input":[{{"role":"user","content":"x"}},{{"type":"message","role":"assistant","id":"{MSG}","content":[]}}]}}"#
        );
        assert_eq!(
            s.unsign_request(A, raw.as_bytes(), false)
                .unwrap_err()
                .field,
            "input[1].id"
        );
        // An escaped key is the same key to the provider.
        let esc = format!(r#"{{"previous\u005fresponse_id":"{RESP}"}}"#);
        assert!(s.unsign_request(A, esc.as_bytes(), false).is_err());
        // The last of two keys is the one a provider keeps; both are checked.
        let dup = format!(r#"{{"previous_response_id":"{r}","previous_response_id":"{RESP}"}}"#);
        assert!(s.unsign_request(A, dup.as_bytes(), false).is_err());
        // A body with no id is untouched and unparsed.
        assert_eq!(
            s.unsign_request(A, br#"{"model":"m","input":"hi"}"#, false)
                .unwrap(),
            None
        );
        // `null` is no id.
        assert_eq!(
            s.unsign_request(A, br#"{"previous_response_id":null}"#, false)
                .unwrap(),
            None
        );
    }

    #[test]
    fn a_stream_signs_every_id_consistently_across_events_and_split_chunks() {
        let s = signer();
        let stream = format!(
            "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{\"id\":\"{RESP}\",\"output\":[]}}}}\n\n\
             event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"item\":{{\"id\":\"{MSG}\",\"type\":\"message\"}}}}\n\n\
             event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"item_id\":\"{MSG}\",\"delta\":\"id\\\"\"}}\n\n\
             event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"{RESP}\",\"output\":[{{\"id\":\"{MSG}\",\"type\":\"message\"}}]}}}}\n\n"
        );
        let want = stream
            .replace(RESP, &s.sign(A, RESP))
            .replace(MSG, &s.sign(A, MSG));
        for split in [1, 7, 64, stream.len()] {
            let mut relay = Relay::new(A);
            assert!(relay.begin(200, true));
            let mut out = Vec::new();
            let chunks: Vec<&[u8]> = stream.as_bytes().chunks(split).collect();
            for (i, c) in chunks.iter().enumerate() {
                let eos = i + 1 == chunks.len();
                out.extend(relay.feed(&s, c, eos).unwrap().unwrap());
            }
            assert_eq!(String::from_utf8(out).unwrap(), want, "split {split}");
        }
    }

    #[test]
    fn a_json_body_is_signed_and_an_error_is_relayed_untouched() {
        let s = signer();
        let body = format!(
            r#"{{"id":"{RESP}","object":"response","previous_response_id":null,"output":[{{"type":"message","id":"{MSG}","content":[{{"type":"output_text","text":"id\""}}]}}],"usage":{{"input_tokens":1}}}}"#
        );
        let mut relay = Relay::new(A);
        assert!(relay.begin(200, false));
        let (a, b) = body.as_bytes().split_at(10);
        assert_eq!(relay.feed(&s, a, false).unwrap().unwrap(), b"");
        let out = relay.feed(&s, b, true).unwrap().unwrap();
        let want = body
            .replace(RESP, &s.sign(A, RESP))
            .replace(MSG, &s.sign(A, MSG));
        assert_eq!(String::from_utf8(out).unwrap(), want);
        assert!(!relay.begin(400, false));
        assert!(relay.feed(&s, body.as_bytes(), true).unwrap().is_none());
    }
}
