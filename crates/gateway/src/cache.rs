//! In-process exact-match response cache.
//!
//! Identical managed catalog-walk requests (`/auto`, managed `/v1`) replay a stored 2xx and skip
//! the provider. A miss is still an unbuffered relay: the fill is a **tap** (copy, never withhold),
//! the same contract as payload capture. BYO and `/{provider}` passthrough are not cached — those
//! paths do not have the client body in hand before `upstream_peer`.
//!
//! The key is a hash of the **pre-rewrite** body + method + inbound path + `tenant_id` + the
//! resolved catalog row + the `anthropic-version` / `anthropic-beta` values + the effective
//! catalog-walk candidate order (provider ids). Not the pool key, not the serving candidate, not
//! the raw virtual key: a 429 that walks to a second key and then 200s is still one client request.
//! Split/order permute the walk, so each arm is its own cache entry rather than pinning A/B traffic
//! to whichever provider filled first.
//!
//! **Per-pod, not fleet-wide.** This table lives in the process. A second replica has its own
//! empty table; a hit on pod A is a miss on pod B. There is no Redis (or other shared store) on
//! the miss path — a miss is always the unbuffered upstream relay. `ai_cache_scope{kind="process"}`
//! is the metric that says so; do not dashboards this as a shared cache.

use crate::usage::Usage;
use bytes::Bytes;
use pingora::http::RequestHeader;
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

/// Process-secret ahash state for [`key`].
///
/// `std`'s `DefaultHasher::new()` is SipHash-1-3 keyed with zeros. The fingerprint covers the
/// whole pre-rewrite body (up to the 64 KiB peek), twice, on the lookup that sits in front of
/// `upstream_peer`. ahash is already in the build for the rate limiter; a per-process secret
/// keeps a crafted same-tenant collision from being precomputed offline. The table still hashes
/// `CacheKey` with `HashMap`'s own `RandomState`.
static KEY_HASHER: LazyLock<ahash::RandomState> = LazyLock::new(ahash::RandomState::new);

/// 128-bit fingerprint of a [`KeyParts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; 16]);

/// A complete 2xx we can replay without talking to a provider.
#[derive(Clone, Debug)]
pub struct CachedResponse {
    pub status: u16,
    pub content_type: Box<str>,
    pub body: Bytes,
    pub usage: Usage,
    pub billed_model: Box<str>,
    pub requested_model: Box<str>,
    pub routed_model: Option<&'static str>,
    pub provider: Box<str>,
    pub streaming: bool,
}

/// Per-request cache state, boxed on the catalog-walk path only.
pub enum Pending {
    /// Miss: tap the response; insert in `logging` only if the relay completes as a 2xx.
    Fill {
        key: CacheKey,
        tap: ResponseTap,
        content_type: Option<Box<str>>,
    },
    /// Hit: the cached response has already been written; `logging` emits stored tokens.
    Hit(CachedResponse),
}

/// Head-bounded copy of a response, taken as a tap — bytes are never withheld from the client.
pub struct ResponseTap {
    buf: Vec<u8>,
    truncated: bool,
    max: usize,
}

impl ResponseTap {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            buf: Vec::new(),
            truncated: false,
            max: max_bytes,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) {
        let room = self.max.saturating_sub(self.buf.len());
        if room == 0 {
            self.truncated |= !chunk.is_empty();
            return;
        }
        let take = room.min(chunk.len());
        self.buf.extend_from_slice(&chunk[..take]);
        self.truncated |= take < chunk.len();
    }

    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// The retained bytes, if the response fit under the cap. Truncation is a fill-skip, not a
    /// partial store: serving a cut body as a complete cached 2xx would be a silent wrong answer.
    pub fn complete_body(&self) -> Option<&[u8]> {
        (!self.truncated).then_some(self.buf.as_slice())
    }
}

/// Request headers whose value changes the upstream's answer for the same body, so they are part
/// of the key. Any other header the gateway forwards (`content-type`, `accept`, `user-agent`) does
/// not change what a provider generates.
pub const VARY_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];

/// Everything a cached answer depends on.
pub struct KeyParts<'a> {
    pub tenant_id: u64,
    pub method: &'a str,
    pub inbound_path: &'a str,
    /// The catalog row the request resolved to. With `x-beyond-model` the header names it, not the
    /// body, and the body's `model` is overwritten per candidate: two rows that share a provider
    /// set must not share an entry for the same body.
    pub model: &'a str,
    /// The request headers; only [`VARY_HEADERS`] are read.
    pub headers: &'a http::HeaderMap,
    pub body: &'a [u8],
    /// The effective catalog-walk order (`ProviderId::index` bytes), so split/order cache per arm
    /// instead of pinning both to the first fill.
    pub providers: &'a [u8],
}

/// Fingerprint used as the map key. Every variable-length field is length-prefixed, which stops
/// `path||body`-style concatenation collisions.
pub fn key(p: &KeyParts<'_>) -> CacheKey {
    fn field(h: &mut impl Hasher, bytes: &[u8]) {
        h.write_u64(bytes.len() as u64);
        h.write(bytes);
    }
    let mix = |seed: u64| {
        let mut h = KEY_HASHER.build_hasher();
        h.write_u64(seed);
        h.write_u64(p.tenant_id);
        field(&mut h, p.method.as_bytes());
        field(&mut h, p.inbound_path.as_bytes());
        field(&mut h, p.model.as_bytes());
        for name in VARY_HEADERS {
            let values = p.headers.get_all(name);
            h.write_u64(values.iter().count() as u64);
            for v in values {
                field(&mut h, v.as_bytes());
            }
        }
        field(&mut h, p.body);
        field(&mut h, p.providers);
        h.finish()
    };
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&mix(0).to_le_bytes());
    out[8..].copy_from_slice(&mix(1).to_le_bytes());
    CacheKey(out)
}

/// `Cache-Control: no-store` (any directive list that includes that token) skips lookup and fill.
pub fn cache_control_no_store(req: &RequestHeader) -> bool {
    req.headers
        .get("cache-control")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(',')
                .any(|d| d.trim().eq_ignore_ascii_case("no-store"))
        })
}

/// Skip lookup and store: `x-beyond-cache: off` or `Cache-Control: no-store`.
pub fn request_bypasses(req: &RequestHeader) -> bool {
    cache_control_no_store(req)
        || req
            .headers
            .get(crate::control::CACHE_HEADER)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("off"))
}

struct Slot {
    expires_at: Instant,
    value: CachedResponse,
}

struct Inner {
    map: HashMap<CacheKey, Slot>,
    /// Insertion order for bounded eviction. Oldest first.
    order: VecDeque<CacheKey>,
}

impl Inner {
    /// Drop `key` from the map and, only when it was present, from the insertion order.
    ///
    /// A fill of a key the table has never seen — the common miss — used to walk the whole
    /// order deque to confirm the absence the map already reports. The scan stays for a
    /// refresh, which has to move that one entry to the back.
    fn remove_key(&mut self, key: &CacheKey) {
        if self.map.remove(key).is_none() {
            return;
        }
        if let Some(i) = self.order.iter().position(|k| k == key) {
            self.order.remove(i);
        }
    }

    /// Drop the expired prefix.
    ///
    /// One TTL means insertion order is expiry order, including after a refresh (that key is
    /// removed and pushed back). The first entry that is still live proves every later entry
    /// is too, so a fill does not walk the live suffix under the mutex.
    fn evict_expired(&mut self, now: Instant) {
        while let Some(k) = self.order.front().copied() {
            match self.map.get(&k) {
                Some(s) if s.expires_at > now => break,
                _ => {
                    self.order.pop_front();
                    self.map.remove(&k);
                }
            }
        }
    }
}

/// TTL + max-entries + max-bytes/entry store. Lookups never error: a poisoned lock is recovered,
/// a full store evicts, an oversized body is dropped. The request path must not fail because the
/// cache could not help it.
pub struct ResponseCache {
    ttl: Duration,
    max_entries: usize,
    max_bytes: usize,
    inner: Mutex<Inner>,
}

impl ResponseCache {
    pub fn new(ttl: Duration, max_entries: usize, max_bytes: usize) -> Self {
        Self {
            ttl,
            max_entries: max_entries.max(1),
            max_bytes,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn get(&self, key: &CacheKey) -> Option<CachedResponse> {
        let mut g = self.lock();
        let now = Instant::now();
        match g.map.get(key) {
            Some(s) if s.expires_at > now => Some(s.value.clone()),
            Some(_) => {
                g.remove_key(key);
                None
            }
            None => None,
        }
    }

    pub fn insert(&self, key: CacheKey, value: CachedResponse) {
        if value.body.len() > self.max_bytes {
            return;
        }
        let mut g = self.lock();
        let now = Instant::now();
        g.evict_expired(now);
        g.remove_key(&key);
        while g.map.len() >= self.max_entries {
            if let Some(oldest) = g.order.pop_front() {
                g.map.remove(&oldest);
            } else {
                break;
            }
        }
        g.order.push_back(key);
        g.map.insert(
            key,
            Slot {
                expires_at: now + self.ttl,
                value,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pingora::http::RequestHeader;

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        }
    }

    fn entry(body: &[u8]) -> CachedResponse {
        CachedResponse {
            status: 200,
            content_type: "application/json".into(),
            body: Bytes::copy_from_slice(body),
            usage: usage(11, 7),
            billed_model: "gpt-4o".into(),
            requested_model: "gpt-4o".into(),
            routed_model: Some("gpt-4o"),
            provider: "openai".into(),
            streaming: false,
        }
    }

    /// A key for a POST on the `gpt-4o` row with no vary headers.
    fn key_of(tenant_id: u64, inbound_path: &str, body: &[u8], providers: &[u8]) -> CacheKey {
        key(&KeyParts {
            tenant_id,
            method: "POST",
            inbound_path,
            model: "gpt-4o",
            headers: &http::HeaderMap::new(),
            body,
            providers,
        })
    }

    fn req(headers: &[(&str, &str)]) -> RequestHeader {
        let mut r = RequestHeader::build("POST", b"/v1/chat/completions", None).unwrap();
        for (k, v) in headers {
            r.insert_header(k.to_string(), *v).unwrap();
        }
        r
    }

    #[test]
    fn key_is_tenant_path_body_and_candidate_order() {
        let body = br#"{"model":"gpt-4o","messages":[]}"#;
        let a = key_of(1, "/v1/chat/completions", body, &[0, 2]);
        assert_eq!(a, key_of(1, "/v1/chat/completions", body, &[0, 2]));
        assert_ne!(
            a,
            key_of(2, "/v1/chat/completions", body, &[0, 2]),
            "tenant isolation"
        );
        assert_ne!(
            a,
            key_of(1, "/auto/chat/completions", body, &[0, 2]),
            "inbound path is part of the key"
        );
        assert_ne!(
            a,
            key_of(
                1,
                "/v1/chat/completions",
                br#"{"model":"gpt-4o-mini"}"#,
                &[0, 2]
            ),
            "body is part of the key"
        );
        assert_ne!(
            a,
            key_of(1, "/v1/chat/completions", body, &[2, 0]),
            "candidate order is part of the key — split/order cache per arm"
        );
        assert_ne!(
            a,
            key_of(1, "/v1/chat/completions", body, &[0]),
            "a shorter walk is a different arm"
        );
        // Concatenation must not collide: path "ab" + body "c" vs path "a" + body "bc".
        assert_ne!(
            key_of(1, "ab", b"c", &[]),
            key_of(1, "a", b"bc", &[]),
            "length prefixes stop concatenation collisions"
        );
    }

    /// claim: SEC-14
    /// defect: D31
    #[test]
    fn key_includes_the_row_the_method_and_the_vary_headers() {
        let body = br#"{"model":"gpt-4o","messages":[]}"#;
        let none = http::HeaderMap::new();
        let parts = |method, model, headers| KeyParts {
            tenant_id: 1,
            method,
            inbound_path: "/v1/chat/completions",
            model,
            headers,
            body,
            providers: &[0],
        };
        let a = key(&parts("POST", "gpt-4o", &none));
        assert_eq!(a, key_of(1, "/v1/chat/completions", body, &[0]));
        assert_ne!(a, key(&parts("POST", "gpt-4o-mini", &none)), "row");
        assert_ne!(a, key(&parts("GET", "gpt-4o", &none)), "method");
        let mut beta = http::HeaderMap::new();
        beta.insert(
            "anthropic-beta",
            "prompt-caching-2024-07-31".parse().unwrap(),
        );
        let b = key(&parts("POST", "gpt-4o", &beta));
        assert_ne!(a, b, "anthropic-beta");
        let mut version = http::HeaderMap::new();
        version.insert(
            "anthropic-version",
            "prompt-caching-2024-07-31".parse().unwrap(),
        );
        assert_ne!(
            b,
            key(&parts("POST", "gpt-4o", &version)),
            "the same value under the other header"
        );
        let mut unrelated = http::HeaderMap::new();
        unrelated.insert("user-agent", "sdk/2".parse().unwrap());
        assert_eq!(
            a,
            key(&parts("POST", "gpt-4o", &unrelated)),
            "not a vary header"
        );
    }

    #[test]
    fn insert_then_get_replays_the_stored_bytes() {
        let c = ResponseCache::new(Duration::from_secs(60), 8, 1024);
        let k = key_of(1, "/v1/chat/completions", b"{}", &[]);
        c.insert(k, entry(b"{\"ok\":true}"));
        let hit = c.get(&k).expect("hit");
        assert_eq!(hit.status, 200);
        assert_eq!(hit.body.as_ref(), br#"{"ok":true}"#);
        assert_eq!(hit.usage, usage(11, 7));
        assert!(
            c.get(&key_of(2, "/v1/chat/completions", b"{}", &[]))
                .is_none()
        );
    }

    #[test]
    fn expired_entry_is_a_miss() {
        let c = ResponseCache::new(Duration::from_millis(1), 8, 1024);
        let k = key_of(1, "/v1", b"x", &[]);
        c.insert(k, entry(b"old"));
        std::thread::sleep(Duration::from_millis(5));
        assert!(c.get(&k).is_none(), "TTL expiry must miss, not serve stale");
    }

    #[test]
    fn max_entries_evicts_the_oldest() {
        let c = ResponseCache::new(Duration::from_secs(60), 2, 1024);
        let k1 = key_of(1, "/v1", b"a", &[]);
        let k2 = key_of(1, "/v1", b"b", &[]);
        let k3 = key_of(1, "/v1", b"c", &[]);
        c.insert(k1, entry(b"1"));
        c.insert(k2, entry(b"2"));
        c.insert(k3, entry(b"3"));
        assert!(c.get(&k1).is_none(), "oldest of 3 into a 2-slot store");
        assert_eq!(c.get(&k2).unwrap().body.as_ref(), b"2");
        assert_eq!(c.get(&k3).unwrap().body.as_ref(), b"3");
    }

    #[test]
    fn refreshing_a_key_moves_it_behind_older_entries() {
        // A re-fill must not stay at the front of the order, or the next insert evicts the
        // entry that was just written and keeps the one that is about to go stale.
        let c = ResponseCache::new(Duration::from_secs(60), 2, 1024);
        let k1 = key_of(1, "/v1", b"a", &[]);
        let k2 = key_of(1, "/v1", b"b", &[]);
        let k3 = key_of(1, "/v1", b"c", &[]);
        c.insert(k1, entry(b"1"));
        c.insert(k2, entry(b"2"));
        c.insert(k1, entry(b"1b"));
        c.insert(k3, entry(b"3"));
        assert_eq!(c.get(&k1).unwrap().body.as_ref(), b"1b");
        assert!(c.get(&k2).is_none(), "the unrefreshed key is the oldest");
        assert_eq!(c.get(&k3).unwrap().body.as_ref(), b"3");
    }

    #[test]
    fn insert_drops_an_expired_prefix_without_evicting_a_newer_live_entry() {
        let c = ResponseCache::new(Duration::from_millis(40), 2, 1024);
        let k1 = key_of(1, "/v1", b"a", &[]);
        let k2 = key_of(1, "/v1", b"b", &[]);
        c.insert(k1, entry(b"old"));
        std::thread::sleep(Duration::from_millis(50));
        c.insert(k2, entry(b"new"));
        let k3 = key_of(1, "/v1", b"c", &[]);
        c.insert(k3, entry(b"newer"));
        assert!(c.get(&k1).is_none());
        assert_eq!(c.get(&k2).unwrap().body.as_ref(), b"new");
        assert_eq!(c.get(&k3).unwrap().body.as_ref(), b"newer");
    }

    #[test]
    fn oversized_body_is_not_stored() {
        let c = ResponseCache::new(Duration::from_secs(60), 8, 4);
        let k = key_of(1, "/v1", b"x", &[]);
        c.insert(k, entry(b"12345"));
        assert!(c.get(&k).is_none());
        c.insert(k, entry(b"1234"));
        assert!(c.get(&k).is_some(), "exactly at the cap is kept");
    }

    #[test]
    fn tap_never_grows_past_the_cap_and_flags_truncation() {
        let mut t = ResponseTap::new(4);
        t.push(b"ab");
        t.push(b"cdef");
        assert_eq!(t.complete_body(), None);
        assert!(t.truncated());
        assert_eq!(&t.buf, b"abcd");

        let mut exact = ResponseTap::new(4);
        exact.push(b"abcd");
        assert_eq!(exact.complete_body(), Some(b"abcd".as_slice()));
        exact.push(b"");
        assert!(!exact.truncated());
    }

    #[test]
    fn cache_control_no_store_is_a_directive_token() {
        assert!(!cache_control_no_store(&req(&[])));
        assert!(cache_control_no_store(&req(&[(
            "cache-control",
            "no-store"
        )])));
        assert!(cache_control_no_store(&req(&[(
            "cache-control",
            "private, no-store"
        )])));
        assert!(cache_control_no_store(&req(&[(
            "cache-control",
            "NO-STORE"
        )])));
        assert!(!cache_control_no_store(&req(&[(
            "cache-control",
            "max-age=0"
        )])));
        // `no-store` as a substring of another token must not match.
        assert!(!cache_control_no_store(&req(&[(
            "cache-control",
            "no-stored"
        )])));
    }

    #[test]
    fn request_bypasses_on_off_header_or_no_store() {
        assert!(!request_bypasses(&req(&[])));
        assert!(request_bypasses(&req(&[("x-beyond-cache", "off")])));
        assert!(request_bypasses(&req(&[("x-beyond-cache", "OFF")])));
        assert!(!request_bypasses(&req(&[("x-beyond-cache", "on")])));
        assert!(request_bypasses(&req(&[("x-beyond-cache", " off ")])));
        assert!(request_bypasses(&req(&[("cache-control", "no-store")])));
    }
}
