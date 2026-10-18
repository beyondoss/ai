//! Whether a failed attempt is retried, and how: one decision ([`decide`]) for every hook that
//! can ask pingora for another attempt or hand one back to a `FullBody` parent.
//!
//! The hooks report facts ([`Failed`]) and act on the [`Decision`]: they move the walk, the pool
//! key or the address, count it and log it. None of them decides inline. The one rule every
//! decision keeps: **a body the provider may already have is never sent again, unless the
//! provider proved it unprocessed** (an HTTP/2 refusal, or a pooled HTTP/1.1 connection it reset
//! with the request unread). A provider holding the whole body may be generating, and billing,
//! already (D09, D51). An error status at the head is the provider's answer, so the request is
//! over there and may go to the next key or candidate; whether pingora can resend it is then only
//! whether it still holds the body ([`Body::replayable`]).

/// What the next attempt is, as the hook that asked pingora for it declared: every decision that
/// resends declares one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Retry {
    /// The same leg again: a managed 429 or 401 key walk, a stale pooled connection, a refused
    /// stream's one resend, or the candidate's next address. Its breaker permit carries over (a
    /// 429 is a healthy throttle, and a dead connection is not the provider failing), and the
    /// catalog walk is not entered: that would send this provider's remaining keys to another
    /// vendor.
    SameLeg,
    /// A catalog walk's next candidate. The outgoing leg failed this attempt: a permit it still
    /// owes is that failure (a 5xx, a connect or upstream failure), resolved as the walk moves on.
    /// One already resolved (a refusing 401–403, the success it is) owes nothing more.
    NextCandidate,
}

/// A retry a `FullBody` subrequest hands back to its parent, which holds the body pingora cannot
/// replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelayRetry {
    /// A candidate failure (a 5xx, a pool-key 401/402/403, or a connection that failed before the
    /// upstream had the whole body) from this catalog index with another candidate left: skip it.
    Candidate(u8),
    /// A 429 (or a pool key's 401) from this catalog index with another pool key left: resume
    /// there.
    Key { candidate: u8, key: u8 },
    /// A reused connection failed before the upstream had the whole body (a pooled connection
    /// the provider had already closed), or a stream it refused: try this candidate once more on a
    /// fresh connection.
    Reset(u8),
}

/// One failed attempt, as its hook saw it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Failed {
    pub failure: Failure,
    /// The catalog walk, on a model-routed request.
    pub walk: Option<Walk>,
    pub body: Body,
}

/// What failed.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Failure {
    /// The connect failed (`fail_to_connect`): no byte of the body moved. `retries_left`: a
    /// provider-routed request may try its peer again (`MAX_CONNECT_RETRIES`).
    Connect { retries_left: bool },
    /// The provider answered at its head with this status (`upstream_response_filter`).
    Status {
        status: u16,
        managed: bool,
        /// The pool key after this one, when the provider has one.
        next_key: Option<u8>,
    },
    /// The connection failed once it was up (`error_while_proxy`).
    Lost(Lost),
}

/// A connection that failed once it was up, and what that says about the body.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Lost {
    /// The client is gone: its error, or a stalled upload retagged as its (D260).
    pub downstream: bool,
    /// The response already started reaching the client.
    pub responded: bool,
    /// The provider may have the whole body: the connection was up, the body read to its end,
    /// and nothing failed writing it (`body_delivered`).
    pub delivered: bool,
    /// The provider refused the stream unprocessed (D72): `RST_STREAM(REFUSED_STREAM)`, or a
    /// GOAWAY above its last stream id.
    pub refused: bool,
    /// The refusal was a GOAWAY.
    pub goaway: bool,
    /// A pooled HTTP/1.1 connection was reset with the request unread in it (D80).
    pub reset_unread: bool,
    /// The connection came from the pool.
    pub reused: bool,
    /// This leg already spent its one refused-stream resend.
    pub refused_resent: bool,
    /// An HTTP/2 body write met a stream already closed (D248): a reused connection's failure
    /// before delivery.
    pub h2_body_unsent: bool,
    /// Pingora's own verdict on the error, its `RetryType` decided for `reused`.
    pub pingora_retries: bool,
}

impl Lost {
    /// Whether this failure may be resent at all: the client is still there, nothing reached it,
    /// and the provider cannot have the whole body, or proved it took none of it.
    pub fn may_resend(&self) -> bool {
        let unproven = self.delivered && !self.refused && !self.reset_unread;
        !self.downstream && !self.responded && !unproven
    }

    /// A GOAWAY draining a connection this request found already open (D248): the provider
    /// retiring a connection that carried other streams, not refusing work, so its resend does
    /// not spend the leg's one.
    fn drain(&self) -> bool {
        self.refused && self.reused && self.goaway
    }

    /// Whether a refused stream is resent on its leg: once (D72), and on every drain.
    pub fn resends_refusal(&self) -> bool {
        self.refused && (self.drain() || !self.refused_resent)
    }

    /// Whether resending this failure on its leg spends the leg's one refused-stream resend: a
    /// refusal does, unless it was a drain.
    pub fn spends_resend(&self) -> bool {
        self.refused && !self.drain()
    }
}

/// Where a catalog walk can go from the candidate that failed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Walk {
    /// The failing walk slot's catalog index (absent past the walk's end).
    pub orig: Option<u8>,
    /// Another usable candidate follows it.
    pub next: bool,
    /// It resolved to more than one address, and an untried one remains.
    pub next_addr: bool,
}

/// Where the body can be resent from.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Body {
    /// Fully read and within pingora's 64 KiB retry buffer, so pingora can resend it whole
    /// (`body_replayable`).
    pub replayable: bool,
    /// Past pingora's retry buffer (`retry_buffer_truncated`), read so far.
    pub truncated: bool,
    /// A `FullBody` attempt: its parent holds the body and re-runs it.
    pub parent: Option<Parent>,
}

/// What a `FullBody` parent already spent on the failing candidate.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Parent {
    /// The candidate already had its one fresh-connection retry.
    pub reset_spent: bool,
}

/// What becomes of a failed attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    /// The answer or the error stands: no retry.
    Stand(Stand),
    /// Pingora resends the body it holds, on this leg or the walk's next candidate.
    Resend(Retry),
    /// The `FullBody` parent drops this attempt and re-runs the body it holds.
    Rerun(RelayRetry),
}

/// Why a status answer stands when a walk was due: the body could not be resent. Counted and
/// logged by the hook.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Stand {
    /// A key walk was due (a managed 429 or 401 with another key left).
    pub key_walk_unreplayable: bool,
    /// A failover was due (`ai_failover_unreplayable_total`).
    pub failover_unreplayable: bool,
}

impl Decision {
    const STAND: Self = Self::Stand(Stand {
        key_walk_unreplayable: false,
        failover_unreplayable: false,
    });
}

/// Whether, and how, a failed attempt is retried.
pub(crate) fn decide(f: &Failed) -> Decision {
    match f.failure {
        Failure::Connect { retries_left } => connect(f.walk, retries_left),
        Failure::Status {
            status,
            managed,
            next_key,
        } => answered(status, managed, next_key, f.walk, f.body),
        Failure::Lost(lost) => connection(&lost, f.walk, f.body),
    }
}

/// A connect failure moved no body byte, so it is always safe to resend. A walk tries the
/// candidate's next address, then the next candidate, never the same peer again: a different
/// provider covers the blip a same-peer retry exists for and a provider that is down, and keeps
/// one breaker outcome per candidate.
fn connect(walk: Option<Walk>, retries_left: bool) -> Decision {
    match walk {
        Some(w) if w.next_addr => Decision::Resend(Retry::SameLeg),
        Some(w) if w.next => Decision::Resend(Retry::NextCandidate),
        Some(_) => Decision::STAND,
        None if retries_left => Decision::Resend(Retry::SameLeg),
        None => Decision::STAND,
    }
}

/// An error status at the head. A managed 429 or 401 walks this provider's keys (a 401 is the
/// provider refusing the credential before it processed anything); a walk's 5xx or candidate
/// refusal (401–403) moves to the next candidate. A 429 is never a vendor failover: re-asking
/// another vendor turns a self-healing throttle into spend somewhere else.
fn answered(
    status: u16,
    managed: bool,
    next_key: Option<u8>,
    walk: Option<Walk>,
    body: Body,
) -> Decision {
    let mut stand = Stand::default();
    let rerun_at = walk.and_then(|w| w.orig).filter(|_| body.parent.is_some());
    if managed && (status == 429 || crate::proxy::is_pool_key_failure(status)) {
        if let Some(key) = next_key {
            if body.replayable {
                return Decision::Resend(Retry::SameLeg);
            }
            if let Some(candidate) = rerun_at {
                return Decision::Rerun(RelayRetry::Key { candidate, key });
            }
            stand.key_walk_unreplayable = true;
        }
        // The last key, or an unreplayable body: the 429 is relayed. A 401 goes on to the walk.
        if status == 429 {
            return Decision::Stand(stand);
        }
    }
    let Some(w) = walk else {
        return Decision::Stand(stand);
    };
    let refusal = managed && crate::proxy::is_candidate_refusal(status);
    if (status < 500 && !refusal) || !w.next {
        return Decision::Stand(stand);
    }
    // Only a body pingora holds whole is resent: one still streaming in fails over or not by how
    // fast the provider rejected it, and a rule that picks the vendor billed must not be timing.
    // The cost: a 5xx that lands while the client is still uploading is relayed even when it
    // would have replayed fine, which `ai_failover_unreplayable_total` counts.
    if !body.replayable {
        if let Some(orig) = rerun_at {
            return Decision::Rerun(RelayRetry::Candidate(orig));
        }
        stand.failover_unreplayable = true;
        return Decision::Stand(stand);
    }
    Decision::Resend(Retry::NextCandidate)
}

/// A connection that failed once it was up. Never resent once the provider may have the whole
/// body ([`Lost::may_resend`]), to the same candidate or the next, pingora's reused-connection
/// retry included.
///
/// A refused stream gets one resend on its candidate (D72). Refused again there, the provider is
/// failing to take work (its stream limit, a drain that never ends): a provider failure like any
/// other, so the walk fails over and the breaker hears of it (D91), rather than a tight loop of
/// resends up to pingora's retry limit. A drain GOAWAY does not spend that resend (D248): one
/// GOAWAY refuses every multiplexed stream above its last id at once (D160) and retires the
/// connection, so the resend lands on another, and pingora's retry limit bounds them all.
///
/// A body pingora cannot resend goes back to its `FullBody` parent, if it has one: a reused or
/// refused connection gets one more try on the same candidate, otherwise the walk moves on. A
/// small body: a reused connection gets one more try on a fresh one, otherwise the walk moves on,
/// and a provider-routed request follows pingora's own verdict.
fn connection(lost: &Lost, walk: Option<Walk>, body: Body) -> Decision {
    if !lost.may_resend() {
        return Decision::STAND;
    }
    // A slot past the walk's end has no catalog index: it is retried as a provider route is.
    let walk = walk.and_then(|w| Some((w.orig?, w.next)));
    if let Some(parent) = body.parent {
        return match walk {
            Some((orig, _)) if (lost.reused || lost.refused) && !parent.reset_spent => {
                Decision::Rerun(RelayRetry::Reset(orig))
            }
            Some((orig, true)) => Decision::Rerun(RelayRetry::Candidate(orig)),
            _ => Decision::STAND,
        };
    }
    if body.truncated {
        return Decision::STAND;
    }
    match walk {
        Some(_) if lost.resends_refusal() => Decision::Resend(Retry::SameLeg),
        Some(_) if lost.reused && !lost.refused => Decision::Resend(Retry::SameLeg),
        Some((_, true)) => Decision::Resend(Retry::NextCandidate),
        Some(_) => Decision::STAND,
        None if lost.resends_refusal() => Decision::Resend(Retry::SameLeg),
        None if lost.refused => Decision::STAND,
        None if lost.h2_body_unsent && lost.reused => Decision::Resend(Retry::SameLeg),
        None if lost.h2_body_unsent => Decision::STAND,
        None if lost.pingora_retries => Decision::Resend(Retry::SameLeg),
        None => Decision::STAND,
    }
}

#[cfg(test)]
#[path = "retry_tests.rs"]
mod tests;
