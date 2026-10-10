//! Drain-on-cancel: when a client hangs up on a managed stream from a provider that keeps
//! generating (and billing) after a disconnect, keep reading the upstream to its natural end,
//! discarding the bytes, so the `ai.usage` row carries the vendor's own final usage instead of an
//! estimate of what was relayed.
//!
//! # The mechanism: pingora's own "keep filling the cache" path
//!
//! Pingora 0.9's proxy loop (`proxy_h1.rs` / `proxy_h2.rs` / `proxy_custom.rs`,
//! `proxy_handle_downstream`) ends the request, and drops the upstream connection, on a downstream
//! read error, with one exception: while the response could still be written to a cache, it marks
//! the downstream errored and keeps reading the upstream to its end ("Downstream Error ignored
//! during caching"). The condition, evaluated per loop turn, is
//! `(!serve_from_cache.is_on() && session.cache.support_streaming_partial_write() == Some(true))
//! || serve_from_cache.is_miss()`.
//!
//! [`arm`] makes its first half true without caching anything. It puts the session's pingora
//! cache (not the gateway's own response cache, `crate::cache`, which this never touches) in its
//! `Bypass` phase on a storage that claims streaming partial writes:
//!
//! - `support_streaming_partial_write()` reads the storage of the enabled context, which `Bypass`
//!   keeps (only `disable` drops it), so it is `Some(true)`;
//! - `serve_from_cache` stays off: it is only turned on at a cache lookup or a cacheable response
//!   head, and an armed request has had both (`proxy_cache` runs before the upstream; the head was
//!   just seen);
//! - `cache_http_task` writes nothing in `Bypass`: a body is written only when `enabled()`, which
//!   `Bypass` is not. No lookup, no miss handler, no admission: [`NoStore`] is never called.
//!
//! It is armed from `response_filter`, after `cache_http_task` saw the head (in `Bypass` that call
//! would ask `response_cache_filter`, whose default disables the cache again). The client hanging
//! up is then reported through `ProxyHttp::suppress_proxy_warn_log` with
//! `ProxyWarnLogContext::DownstreamCache`, the one hook pingora calls on that path, and
//! `response_body_filter` stops relaying: each later chunk still feeds the usage taps and is then
//! replaced with nothing, which pingora writes as nothing (`response_duplex` skips an empty body).
//! The end of the stream may write a last chunk terminator to the dead socket and fail there,
//! after the upstream has ended; that is still a drain that settled.
//!
//! Alternatives weighed: a cache *miss* with a real streaming storage also continues upstream, but
//! then the client is served from that storage, so the body would be held in memory to relay it
//! (and `set_miss_handler` panics on a partial-write storage without a reader); a vendored pingora
//! patch would be three lines but a fork to carry through every upgrade. This uses only public
//! pingora API, and `tests/billing_drain.rs` fails if an upgrade changes the path.
//!
//! Limits: only a hang-up pingora sees as a downstream *read* error takes this path. One seen
//! first as a write error (a race the read side almost always wins: the loop polls the client
//! every turn) still ends the request with an estimate, as does a `FullBody` attempt, whose client
//! is its parent's pipe.

use pingora_cache::storage::{HitHandler, MissHandler};
use pingora_cache::trace::SpanHandle;
use pingora_cache::{
    CacheKey, CacheMeta, CachePhase, HttpCache, PurgeOutcome, PurgeTarget, PurgeType, Storage,
};
use pingora_core::{Error, ErrorType, Result};
use std::any::Any;

/// Where a managed stream is in drain-on-cancel. One byte on `RequestCtx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum Drain {
    /// Not eligible: the client leaving ends the request, as it always did.
    #[default]
    Off,
    /// Eligible and [`arm`]ed; the client is still reading.
    Armed,
    /// The client left; the upstream is being read on and its bytes discarded.
    Draining,
    /// The upstream stream ended while draining: the usage tail holds its final usage.
    Settled,
}

/// A storage that stores nothing. [`arm`] keeps the cache in `Bypass`, where pingora calls none of
/// these; each one refuses rather than pretending to succeed, in case a future pingora does.
pub(crate) struct NoStore;

static NO_STORE: NoStore = NoStore;

fn refused() -> Box<Error> {
    Error::explain(
        ErrorType::InternalError,
        "drain: the no-op cache storage stores nothing",
    )
}

#[async_trait::async_trait]
impl Storage for NoStore {
    async fn lookup(
        &'static self,
        _key: &CacheKey,
        _trace: &SpanHandle,
    ) -> Result<Option<(CacheMeta, HitHandler)>> {
        Ok(None)
    }

    async fn get_miss_handler(
        &'static self,
        _key: &CacheKey,
        _meta: &CacheMeta,
        _trace: &SpanHandle,
    ) -> Result<MissHandler> {
        Err(refused())
    }

    async fn purge(
        &'static self,
        _target: PurgeTarget<'_>,
        _purge_type: PurgeType,
        _trace: &SpanHandle,
    ) -> Result<PurgeOutcome> {
        Ok(PurgeOutcome::NotFound)
    }

    async fn update_meta(
        &'static self,
        _key: &CacheKey,
        _meta: &CacheMeta,
        _trace: &SpanHandle,
    ) -> Result<bool> {
        Err(refused())
    }

    /// The one answer that matters: pingora keeps reading the upstream after a downstream error
    /// only for a storage that says this.
    fn support_streaming_partial_write(&self) -> bool {
        true
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync + 'static) {
        self
    }
}

/// Put the session's pingora cache in `Bypass` on [`NoStore`], so a downstream read error no longer
/// ends the request (see the module docs). `true` when the session is armed. Idempotent: an armed
/// session stays armed. A cache in any other phase is in use by something else and is left alone
/// (`false`); the gateway never enables pingora's cache otherwise, so that is a pingora change.
pub(crate) fn arm(cache: &mut HttpCache) -> bool {
    match cache.phase() {
        CachePhase::Bypass => true,
        CachePhase::Disabled(_) => {
            cache.enable(&NO_STORE, None, None, None, None);
            cache.set_cache_key(CacheKey::new(Vec::new(), String::new()));
            cache.bypass();
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What pingora's proxy loop reads: partial writes claimed (so a downstream error is ignored),
    /// and the cache neither enabled (so `cache_http_task` writes no body) nor serving.
    #[test]
    fn armed_cache_claims_partial_writes_and_caches_nothing() {
        let mut cache = HttpCache::new();
        assert_eq!(cache.support_streaming_partial_write(), None);
        assert!(arm(&mut cache));
        assert_eq!(cache.phase(), CachePhase::Bypass);
        assert_eq!(cache.support_streaming_partial_write(), Some(true));
        assert!(!cache.enabled());
        assert!(cache.bypassing());
        // Idempotent: arming twice neither panics (pingora panics on a second `enable`) nor
        // changes the phase.
        assert!(arm(&mut cache));
        assert_eq!(cache.phase(), CachePhase::Bypass);
    }

    /// A cache disabled after being armed (pingora's `response_cache_filter` on a later attempt's
    /// head) is armed again.
    #[test]
    fn a_disabled_cache_is_armed_again() {
        let mut cache = HttpCache::new();
        assert!(arm(&mut cache));
        cache.disable(pingora_cache::NoCacheReason::Custom("default"));
        assert_eq!(cache.support_streaming_partial_write(), None);
        assert!(arm(&mut cache));
        assert_eq!(cache.support_streaming_partial_write(), Some(true));
    }

    /// A cache in use for something else is never touched.
    #[test]
    fn a_cache_in_use_is_left_alone() {
        let mut cache = HttpCache::new();
        cache.enable(&NO_STORE, None, None, None, None);
        assert!(!arm(&mut cache));
        assert_eq!(cache.phase(), CachePhase::Uninit);
    }
}
