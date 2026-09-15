//! In-process exact-match response cache.
//!
//! Identical managed catalog-walk requests (`/auto`, managed `/v1`) replay a stored 2xx and skip
//! the provider. A miss is still an unbuffered relay: the fill is a **tap** (copy, never withhold),
//! the same contract as payload capture. BYO and `/{provider}` passthrough are not cached — those
//! paths do not have the client body in hand before `upstream_peer`.
//!
//! The key is a hash of the **pre-rewrite** body + inbound path + `tenant_id` + the effective
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
use std::hash::Hasher;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// 128-bit fingerprint of `(tenant_id, inbound path, pre-rewrite body, candidate provider ids)`.
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

/// Fingerprint used as the map key. Length prefixes stop `path||body` concatenation collisions.
/// `providers` is the effective catalog-walk order (`ProviderId::index` bytes) so split/order
/// cache per arm instead of pinning both to the first fill.
pub fn key(tenant_id: u64, inbound_path: &str, body: &[u8], providers: &[u8]) -> CacheKey {
    fn sip(seed: u8, tenant_id: u64, path: &str, body: &[u8], providers: &[u8]) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write(&[seed]);
        h.write(&tenant_id.to_le_bytes());
        h.write(&(path.len() as u64).to_le_bytes());
        h.write(path.as_bytes());
        h.write(&(body.len() as u64).to_le_bytes());
        h.write(body);
        h.write(&(providers.len() as u64).to_le_bytes());
        h.write(providers);
        h.finish()
    }
    let a = sip(0, tenant_id, inbound_path, body, providers).to_le_bytes();
    let b = sip(1, tenant_id, inbound_path, body, providers).to_le_bytes();
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&a);
    out[8..].copy_from_slice(&b);
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
    fn remove_key(&mut self, key: &CacheKey) {
        self.map.remove(key);
        if let Some(i) = self.order.iter().position(|k| k == key) {
            self.order.remove(i);
        }
    }

    fn evict_expired(&mut self, now: Instant) {
        let mut i = 0;
        while i < self.order.len() {
            let k = self.order[i];
            match self.map.get(&k) {
                Some(s) if s.expires_at > now => i += 1,
                Some(_) | None => {
                    self.map.remove(&k);
                    self.order.remove(i);
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
        let a = key(1, "/v1/chat/completions", body, &[0, 2]);
        assert_eq!(a, key(1, "/v1/chat/completions", body, &[0, 2]));
        assert_ne!(
            a,
            key(2, "/v1/chat/completions", body, &[0, 2]),
            "tenant isolation"
        );
        assert_ne!(
            a,
            key(1, "/auto/chat/completions", body, &[0, 2]),
            "inbound path is part of the key"
        );
        assert_ne!(
            a,
            key(
                1,
                "/v1/chat/completions",
                br#"{"model":"gpt-4o-mini"}"#,
                &[0, 2]
            ),
            "body is part of the key"
        );
        assert_ne!(
            a,
            key(1, "/v1/chat/completions", body, &[2, 0]),
            "candidate order is part of the key — split/order cache per arm"
        );
        assert_ne!(
            a,
            key(1, "/v1/chat/completions", body, &[0]),
            "a shorter walk is a different arm"
        );
        // Concatenation must not collide: path "ab" + body "c" vs path "a" + body "bc".
        assert_ne!(
            key(1, "ab", b"c", &[]),
            key(1, "a", b"bc", &[]),
            "length prefixes stop concatenation collisions"
        );
    }

    #[test]
    fn insert_then_get_replays_the_stored_bytes() {
        let c = ResponseCache::new(Duration::from_secs(60), 8, 1024);
        let k = key(1, "/v1/chat/completions", b"{}", &[]);
        c.insert(k, entry(b"{\"ok\":true}"));
        let hit = c.get(&k).expect("hit");
        assert_eq!(hit.status, 200);
        assert_eq!(hit.body.as_ref(), br#"{"ok":true}"#);
        assert_eq!(hit.usage, usage(11, 7));
        assert!(c.get(&key(2, "/v1/chat/completions", b"{}", &[])).is_none());
    }

    #[test]
    fn expired_entry_is_a_miss() {
        let c = ResponseCache::new(Duration::from_millis(1), 8, 1024);
        let k = key(1, "/v1", b"x", &[]);
        c.insert(k, entry(b"old"));
        std::thread::sleep(Duration::from_millis(5));
        assert!(c.get(&k).is_none(), "TTL expiry must miss, not serve stale");
    }

    #[test]
    fn max_entries_evicts_the_oldest() {
        let c = ResponseCache::new(Duration::from_secs(60), 2, 1024);
        let k1 = key(1, "/v1", b"a", &[]);
        let k2 = key(1, "/v1", b"b", &[]);
        let k3 = key(1, "/v1", b"c", &[]);
        c.insert(k1, entry(b"1"));
        c.insert(k2, entry(b"2"));
        c.insert(k3, entry(b"3"));
        assert!(c.get(&k1).is_none(), "oldest of 3 into a 2-slot store");
        assert_eq!(c.get(&k2).unwrap().body.as_ref(), b"2");
        assert_eq!(c.get(&k3).unwrap().body.as_ref(), b"3");
    }

    #[test]
    fn oversized_body_is_not_stored() {
        let c = ResponseCache::new(Duration::from_secs(60), 8, 4);
        let k = key(1, "/v1", b"x", &[]);
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
