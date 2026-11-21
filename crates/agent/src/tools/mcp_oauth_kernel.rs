//! The decisions of mid-session MCP OAuth, as one pure state machine: when a request's token is
//! replaced, when the authorization server (AS) is asked, and what is remembered about a failure.
//!
//! [`step`] takes an [`Event`] and returns the [`Action`] to perform. It does no I/O, is not async,
//! reads no clock (time arrives in the events, as a [`Duration`] on the caller's monotonic scale)
//! and is deterministic, so a model checker can call exactly the function production calls
//! (`tests/mcp_oauth_model.rs`). Production ([`super::mcp_oauth::ServerAuth`]) takes its lock, feeds
//! the kernel an event, performs the action (the AS call, the persist inside it, the reply) and feeds
//! the AS's answer back as the next event, still under the same lock: the lock held across the AS
//! call is what makes a refresh single flight, and the kernel's state is only ever touched under it.
//!
//! The state ([`State`]) is the held token, the failure a 401's refresh left ([`Failed`]), when the
//! token expires, the scopes it was requested with and the step-ups that failed ([`Refused`]).
//!
//! | event | decision |
//! |---|---|
//! | [`Event::Need`] (a request wants its token) | client credentials: refresh when there is none or it is about to expire, unless a failure stands; otherwise reply the held token |
//! | [`Event::Rejected`] (401 for the token sent) | the token was already replaced: reply the current one, no AS call; a failure stands: reply it; else refresh |
//! | [`Event::Insufficient`] (403 `insufficient_scope`) | no client credentials: the login cannot widen; the token already covers it: reply the current one; else [`step_up`] (nothing named, already requested, refused earlier, or request the union) |
//! | [`Event::Refreshed`] (the AS answered) | `Ok`: install the token, clear the failure; definitive: remember for good; transient: remember for a doubling backoff. A failed step-up never touches the held token |
//! | [`Event::Stored`] (the `mcp-login` store was read) | a stored token other than the held one replaces it and clears the failure |
//!
//! A failure stands while it is definitive, or until its backoff has passed. Mutations (cargo
//! features `mut_*`, used only by the model's red runs) each remove one guard.

use std::time::Duration;

use crate::tools::mcp_client_credentials::{StepUp, step_up, union_scopes};

/// What the kernel needs that is configuration, not state.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// A client-credentials server: its token is acquired, scoped and stepped up.
    pub cc: bool,
    /// The first wait after a transient failure; each further one doubles it up to `max_backoff`.
    pub first_backoff: Duration,
    pub max_backoff: Duration,
    /// A client-credentials token is replaced this long before it expires.
    pub renew_skew: Duration,
}

/// One server's token state. Touched only under the lock.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    /// What every request to the server sends; `None` when there is no usable token.
    pub token: Option<String>,
    /// The last refresh after `token` was rejected failed.
    pub failed: Option<Failed>,
    /// Client credentials only: when `token` expires (`None`: the AS did not say).
    pub expires_at: Option<Duration>,
    /// Client credentials only: the scopes `token` was requested with.
    pub scopes: Vec<String>,
    /// Client credentials only: scope sets a step-up failed for.
    pub refused: Vec<Refused>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Failed {
    /// The AS's reason; the shell words the error from it.
    pub reason: String,
    /// `None`: for good (definitive). Otherwise when a refresh may be tried again.
    pub until: Option<Duration>,
    /// The backoff that produced `until`; the next transient failure doubles it.
    pub backoff: Duration,
}

/// A step-up to `scopes` failed: for good when the AS refused it, or until `until` (doubling) when
/// it was only unreachable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Refused {
    pub scopes: Vec<String>,
    pub until: Option<Duration>,
    pub backoff: Duration,
}

/// Why a token request to the AS failed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Failure {
    /// Only a new login helps (the AS rejected the grant, or there is none). Otherwise transient.
    pub definitive: bool,
    pub reason: String,
}

/// What an AS call was for; the answer is read differently for each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Purpose {
    /// [`Event::Need`]: the reply is the held token, whatever the answer.
    Need,
    /// [`Event::Rejected`]: the reply is the new token or the failure.
    Rejection,
    /// [`Event::Insufficient`]: the same, and a failure leaves the held token alone.
    StepUp,
}

#[derive(Debug)]
pub enum Event<'a> {
    /// A request wants its token.
    Need { now: Duration },
    /// A request sent `sent` and was answered 401.
    Rejected {
        now: Duration,
        sent: Option<&'a str>,
    },
    /// A request sent `sent` and was answered 403 `insufficient_scope` naming `challenged`.
    Insufficient {
        now: Duration,
        sent: Option<&'a str>,
        challenged: &'a [String],
    },
    /// The AS answered the call [`Action::Refresh`] asked for: `started` is the time the call
    /// began, `done` when it ended (a token's lifetime runs from `done`).
    Refreshed {
        started: Duration,
        done: Duration,
        purpose: Purpose,
        scopes: Vec<String>,
        result: Result<(String, Option<Duration>), Failure>,
    },
    /// The store was read: what it holds (`None`: nothing usable).
    Stored(Option<String>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    /// Call the AS for a token with `scopes` (persisting what it returns), then feed
    /// [`Event::Refreshed`] back.
    Refresh {
        purpose: Purpose,
        scopes: Vec<String>,
    },
    /// Answer the caller.
    Reply(Reply),
    /// Nothing to do.
    Nothing,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Reply {
    /// The token requests send now (an [`Event::Need`]).
    Token(Option<String>),
    /// The token to retry the request with.
    Granted(String),
    /// The error that ends the request.
    Denied(Denial),
}

/// Which error, with what the shell needs to word it.
#[derive(Debug, PartialEq, Eq)]
pub enum Denial {
    /// A refresh failed or a failure stands; `backoff` is the wait when it is transient.
    Failed {
        reason: String,
        definitive: bool,
        backoff: Duration,
    },
    /// An `mcp-login` server cannot widen its scopes mid-run.
    LoginCannotWiden,
    /// The challenge named no scope.
    NoScope,
    /// Everything the challenge names was already requested.
    AlreadyRequested,
    /// The step-up failed earlier and is not asked again (`transient`: yet).
    RefusedEarlier { transient: bool },
    /// The step-up just failed.
    StepUpFailed { reason: String },
}

impl Failed {
    /// Whether the failure still blocks a refresh at `now`.
    pub fn stands(&self, now: Duration) -> bool {
        #[cfg(feature = "mut_no_401_backoff")]
        {
            let _ = now;
            return self.until.is_none();
        }
        #[cfg(feature = "mut_backoff_never_ends")]
        {
            let _ = now;
            return true;
        }
        #[cfg(not(any(feature = "mut_no_401_backoff", feature = "mut_backoff_never_ends")))]
        self.until.is_none_or(|until| now < until)
    }
}

impl Refused {
    pub fn stands(&self, now: Duration) -> bool {
        self.until.is_none_or(|until| now < until)
    }
}

impl State {
    fn install(
        &mut self,
        token: String,
        lifetime: Option<Duration>,
        scopes: Vec<String>,
        done: Duration,
    ) {
        self.token = Some(token);
        self.expires_at = lifetime.map(|l| done + l);
        self.scopes = scopes;
    }

    fn blocked(&self, now: Duration) -> Option<Denial> {
        self.failed
            .as_ref()
            .filter(|f| f.stands(now))
            .map(|f| Denial::Failed {
                reason: f.reason.clone(),
                definitive: f.until.is_none(),
                backoff: f.backoff,
            })
    }
}

/// The one decision function. `state` is changed in place (a model clones it first).
pub fn step(state: &mut State, cfg: &Config, event: Event<'_>) -> Action {
    match event {
        Event::Need { now } => {
            if cfg.cc {
                let stale = state.token.is_none()
                    || state
                        .expires_at
                        .is_some_and(|at| now + cfg.renew_skew >= at);
                if stale && state.blocked(now).is_none() {
                    return Action::Refresh {
                        purpose: Purpose::Need,
                        scopes: state.scopes.clone(),
                    };
                }
            }
            Action::Reply(Reply::Token(state.token.clone()))
        }
        Event::Rejected { now, sent } => {
            #[cfg(not(feature = "mut_stale_refresh"))]
            if let Some(current) = state.token.as_deref()
                && Some(current) != sent
            {
                return Action::Reply(Reply::Granted(current.to_owned()));
            }
            if let Some(denial) = state.blocked(now) {
                return Action::Reply(Reply::Denied(denial));
            }
            Action::Refresh {
                purpose: Purpose::Rejection,
                scopes: state.scopes.clone(),
            }
        }
        Event::Insufficient {
            now,
            sent,
            challenged,
        } => {
            if !cfg.cc {
                return Action::Reply(Reply::Denied(Denial::LoginCannotWiden));
            }
            // Another call already stepped up past the token `sent`: take its token.
            if let Some(current) = state.token.as_deref()
                && Some(current) != sent
                && union_scopes(&state.scopes, challenged).len() == state.scopes.len()
            {
                return Action::Reply(Reply::Granted(current.to_owned()));
            }
            #[cfg(not(feature = "mut_no_refused_memory"))]
            let standing: Vec<Vec<String>> = state
                .refused
                .iter()
                .filter(|r| r.stands(now))
                .map(|r| r.scopes.clone())
                .collect();
            #[cfg(feature = "mut_no_refused_memory")]
            let standing: Vec<Vec<String>> = Vec::new();
            match step_up(&state.scopes, challenged, &standing) {
                StepUp::NoScope => Action::Reply(Reply::Denied(Denial::NoScope)),
                StepUp::AlreadyRequested => Action::Reply(Reply::Denied(Denial::AlreadyRequested)),
                StepUp::Refused => {
                    let transient = state
                        .refused
                        .iter()
                        .any(|r| r.until.is_some() && r.stands(now));
                    Action::Reply(Reply::Denied(Denial::RefusedEarlier { transient }))
                }
                StepUp::Request(wanted) => Action::Refresh {
                    purpose: Purpose::StepUp,
                    scopes: wanted,
                },
            }
        }
        Event::Refreshed {
            started,
            done,
            purpose,
            scopes,
            result,
        } => match (purpose, result) {
            (Purpose::StepUp, Ok((token, lifetime))) => {
                state.refused.retain(|r| r.scopes != scopes);
                state.install(token.clone(), lifetime, scopes, done);
                Action::Reply(Reply::Granted(token))
            }
            (Purpose::StepUp, Err(e)) => {
                // Remembered either way, as a 401's failure is: for good when the AS refused, a
                // doubling backoff when it was only unreachable. The held token is not touched.
                #[cfg(feature = "mut_no_stepup_backoff")]
                if !e.definitive {
                    return Action::Reply(Reply::Denied(Denial::StepUpFailed { reason: e.reason }));
                }
                let previous = state.refused.iter().position(|r| r.scopes == scopes);
                let backoff = match previous.map(|i| &state.refused[i]) {
                    Some(r) if r.until.is_some() => (r.backoff * 2).min(cfg.max_backoff),
                    _ => cfg.first_backoff,
                };
                let entry = Refused {
                    scopes,
                    until: if e.definitive {
                        None
                    } else {
                        Some(started + backoff)
                    },
                    backoff,
                };
                match previous {
                    Some(i) => state.refused[i] = entry,
                    None => state.refused.push(entry),
                }
                Action::Reply(Reply::Denied(Denial::StepUpFailed { reason: e.reason }))
            }
            (purpose, Ok((token, lifetime))) => {
                state.install(token.clone(), lifetime, scopes, done);
                state.failed = None;
                Action::Reply(match purpose {
                    Purpose::Need => Reply::Token(state.token.clone()),
                    _ => Reply::Granted(token),
                })
            }
            (purpose, Err(e)) => {
                let (until, backoff) = if e.definitive {
                    // A definitive failure stands for good, until a changed store token.
                    #[cfg(feature = "mut_no_definitive_latch")]
                    let until = Some(started);
                    #[cfg(not(feature = "mut_no_definitive_latch"))]
                    let until = None;
                    (until, Duration::ZERO)
                } else {
                    let backoff = match &state.failed {
                        Some(f) if f.until.is_some() => (f.backoff * 2).min(cfg.max_backoff),
                        _ => cfg.first_backoff,
                    };
                    (Some(started + backoff), backoff)
                };
                state.failed = Some(Failed {
                    reason: e.reason.clone(),
                    until,
                    backoff,
                });
                Action::Reply(match purpose {
                    Purpose::Need => Reply::Token(state.token.clone()),
                    _ => Reply::Denied(Denial::Failed {
                        reason: e.reason,
                        definitive: e.definitive,
                        backoff,
                    }),
                })
            }
        },
        Event::Stored(stored) => {
            if cfg.cc {
                return Action::Nothing; // client credentials: nothing stored to reload
            }
            // When the store yields nothing the token held is kept. A changed token clears a
            // remembered failure; an unchanged one does not.
            if let Some(stored) = stored
                && state.token.as_deref() != Some(stored.as_str())
            {
                #[cfg(not(feature = "mut_reload_keeps_failed"))]
                {
                    state.failed = None;
                }
                state.token = Some(stored);
            }
            Action::Nothing
        }
    }
}
