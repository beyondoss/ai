//! The response relay: the pool-key scrub and drain-on-cancel.

use super::*;

/// One response chunk of an armed stream ([`Drain::Armed`] or later): `true` when it is drained
/// (its bytes already fed the usage taps, and are not sent), `false` while the client still reads.
/// The first chunk after the client left starts the drain; the stream's end settles it.
pub(super) fn drain_chunk(rc: &mut RequestCtx, held: &mut Held, end_of_stream: bool) -> bool {
    match held.drain {
        Drain::Off => return false,
        Drain::Armed => {
            if !held.client_left.load(Ordering::Relaxed) {
                return false;
            }
            held.open_drain();
            // The client got part of this answer: the rest is never stored as a cache fill.
            if let Some(a) = rc.auto.as_mut()
                && matches!(a.cache, Some(cache::Pending::Fill { .. }))
            {
                a.cache = None;
            }
        }
        Drain::Draining | Drain::Settled => {}
    }
    if end_of_stream {
        held.drain = Drain::Settled;
    }
    true
}

/// How a drain ended, at `logging`: `None` when none started (the client never left an armed
/// stream). A drain whose upstream ended is settled even if the request then errored, which can
/// only be the dead client's last write (the end of a chunked body); the deadline or an upstream
/// failure first leaves the estimate.
pub(super) fn drain_end(
    drain: Drain,
    client_left: bool,
    e: Option<&pingora_core::Error>,
) -> Option<DrainEnd> {
    if !client_left {
        return None;
    }
    Some(match (drain, e) {
        (Drain::Settled, _) => DrainEnd::Settled,
        (_, Some(e)) if is_deadline(e) => DrainEnd::Deadline,
        (_, Some(e)) if e.esource() != &pingora_core::ErrorSource::Downstream => DrainEnd::Error,
        _ => DrainEnd::Settled,
    })
}

/// Whether a response head starts a stream to drain if the client hangs up ([`crate::drain`]): a
/// managed 2xx stream from a provider in [`MAY_CONTINUE`]. BYO writes no billing row, and a
/// provider that stops generating when the client leaves bills only what it relayed.
pub(super) fn drains_on_cancel(rc: &RequestCtx, status: u16) -> bool {
    rc.managed()
        && rc.streaming
        && (200..300).contains(&status)
        && rc
            .leg
            .provider
            .id
            .is_some_and(|id| MAY_CONTINUE.contains(&id))
}

/// Scrubs a managed error body (status >= 400) as it streams past: the pool key the attempt sent
/// (D66), and a provider-account remedy (D174). The one rewrite of a provider's error body, made
/// before translation, capture and the cache see it.
///
/// **Pool key.** A provider, or a proxy in between, that echoes the credential it received in an
/// error message would otherwise hand Beyond's key to the client. Each occurrence is overwritten in
/// place with [`REDACTED`] padded to the key's length, so the body keeps its length. The last
/// `key.len() - 1` bytes of each chunk are held back until the next one, so a key split across
/// chunks is caught too; memory is bounded by the key, not the body. Each JSON-escaped spelling of
/// the key (`\/`, `+`: [`route::key_finders`]) is overwritten the same way, since a JSON client
/// decodes it to the key (D203).
///
/// **Account remedy.** A JSON error (`whole`) is held until its end, at most
/// [`remedy::MAX_BODY`], and handed to [`remedy::neutralize`] after the key is masked: "add your
/// own key", a billing page and the like become a neutral message (`response_filter` dropped the
/// `Content-Length`, since that changes the length). Past the cap it streams as above. An
/// out-of-credit answer sets `unfunded`, which `logging` reads to cool the key (D180).
#[derive(Default)]
pub(super) struct Redact {
    pub(super) carry: Vec<u8>,
    pub(super) whole: bool,
    /// The response's status: whether a remedy reads as a rate limit (see [`remedy::neutralize`]).
    pub(super) status: u16,
    pub(super) unfunded: bool,
}

/// What a scrubbed pool key reads as. Same length as the key (padded with `*`), truncated when the
/// key is shorter.
pub(super) const REDACTED: &[u8] = b"[redacted]";

impl Redact {
    /// Scrub `chunk` (with the bytes held back from the last one) and return what may be relayed
    /// now: everything at `end_of_stream`, nothing while a `whole` body is held, otherwise all but
    /// a key-sized tail. `keys` are the pool key's boot-built searchers
    /// ([`route::PoolAuth::finders`]: the key as sent, then its JSON-escaped spellings), empty when
    /// the attempt sent none.
    pub(super) fn feed(
        &mut self,
        keys: &[memchr::memmem::Finder<'_>],
        chunk: Option<Bytes>,
        end_of_stream: bool,
    ) -> Option<Bytes> {
        let keys = match keys.first() {
            Some(k) if !k.needle().is_empty() => keys,
            _ => &[],
        };
        if keys.is_empty() && !self.whole && self.carry.is_empty() {
            return chunk;
        }
        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(chunk.as_deref().unwrap_or(&[]));
        if self.whole && !end_of_stream {
            if buf.len() <= remedy::MAX_BODY {
                self.carry = buf;
                // Empty, not `None`: `None` would end the body.
                return Some(Bytes::new());
            }
            self.whole = false;
        }
        for key in keys {
            mask_all(&mut buf, key);
        }
        if self.whole {
            if let Some(n) = remedy::neutralize(&buf, self.status) {
                buf = n.body;
                self.unfunded = n.unfunded;
                // The rewrite decoded every string and wrote it back unescaped: an echo in a
                // spelling no searcher knew (`A` for `A`) is now the key as sent (D203).
                if let Some(key) = keys.first() {
                    mask_all(&mut buf, key);
                }
            } else if let Some(key) = keys.first()
                && memchr::memmem::find(&buf, b"\\u").is_some()
                && let Some(mut plain) = serde_json::from_slice::<serde_json::Value>(&buf)
                    .ok()
                    .and_then(|v| serde_json::to_vec(&v).ok())
                && mask_all(&mut plain, key)
            {
                // Any other `\uXXXX` spelling of the key: a JSON client decodes it to the key, so
                // the body is relayed decoded and masked (D203). Only a body with such an escape
                // and the key behind it is re-serialized.
                buf = plain;
            }
        } else if !end_of_stream {
            let longest = keys.iter().map(|k| k.needle().len()).max().unwrap_or(0);
            let keep = longest.saturating_sub(1).min(buf.len());
            self.carry = buf.split_off(buf.len() - keep);
        }
        Some(Bytes::from(buf))
    }
}

/// Overwrite every occurrence of `key`'s needle in `buf` with [`REDACTED`], padded to its length.
/// Returns whether there was one.
pub(super) fn mask_all(buf: &mut [u8], key: &memchr::memmem::Finder<'_>) -> bool {
    let len = key.needle().len();
    let mut at = 0;
    let mut found = false;
    while let Some(i) = key.find(&buf[at..]) {
        let hit = &mut buf[at + i..at + i + len];
        for (j, b) in hit.iter_mut().enumerate() {
            *b = REDACTED.get(j).copied().unwrap_or(b'*');
        }
        at += i + len;
        found = true;
    }
    found
}

impl AiProxy {
    /// Abort a translated response whose buffer crossed [`translate::MAX_TRANSLATE_BUFFER`].
    ///
    /// Headers are already downstream, so there is no clean error to send — the same trade as the
    /// request-body cap in `request_body_filter`: an abuse guard, not a client path.
    pub(super) fn translate_overflow(
        &self,
        request_id: &str,
        buffer: &'static str,
    ) -> Box<pingora_core::Error> {
        self.state
            .metrics
            .rejection(Rejection::ResponseTooLarge)
            .inc();
        warn!(
            request_id,
            buffer,
            limit = translate::MAX_TRANSLATE_BUFFER,
            "translated response exceeds the buffer limit; aborting",
        );
        pingora_core::Error::new_str("translated response exceeds buffer limit")
    }

    /// Hold, in the process body budget, the heap translating `body` will take
    /// ([`translate::translation_heap`]) for as long as the hold lives (D216). A body past the
    /// replay buffer only, like every other budget charge: smaller ones are bounded by concurrency.
    /// Never exempt on a `FullBody` re-run, whose parent reserved two copies of the body, not the
    /// `Value`s built from it. A translation that needs more than the whole budget is a 413 naming
    /// translation; one that does not fit now is the budget's retryable 503. With the budget off
    /// there is nothing to hold, and nothing is refused. On a response the status line is already
    /// downstream, so either refusal aborts it.
    pub(super) fn hold_translation_heap(
        &self,
        body: &[u8],
    ) -> Result<Option<crate::concurrency::BudgetHold<'_>>> {
        let Some(budget) = self.state.body_budget.as_ref() else {
            return Ok(None);
        };
        if !past_replay_buffer(body.len()) {
            return Ok(None);
        }
        let heap = translate::translation_heap(body);
        if heap > budget.limit() {
            return Err(self.refusal_error(Refusal::TranslateTooLarge).into_down());
        }
        match budget.hold(heap) {
            Some(hold) => Ok(Some(hold)),
            None => Err(self.refusal_error(Refusal::BodyMemory).into_down()),
        }
    }
}
