//! Stateright model of mid-session MCP OAuth: `ServerAuth`'s token lifecycle in
//! `src/tools/mcp_oauth.rs`.
//!
//! It checks every interleaving of two callers (two requests each) against an authorization
//! server (AS) that answers Ok, transient or definitive, a server that answers 401 or 403
//! `insufficient_scope` (naming one of two scopes) to the token it was sent, a clock that
//! advances across the backoff boundaries, and a second process that logs in again (the store
//! changes), for a client-credentials server (`cc`) and an `mcp-login` one (`login`).
//!
//! ## Faithfulness
//!
//! Every decision is made by the REAL kernel, `tools::mcp_oauth_kernel::step`, the function
//! production calls (`ServerAuth::drive`). The model owns only the environment, which contains no
//! copy of the kernel's branching: the lock (production's `tokio::sync::Mutex`, held across the AS
//! call, which is what makes a refresh single flight: one caller steps the kernel and has its AS
//! call outstanding at a time, and a caller's `sent` token can be stale by the time it gets the
//! lock), the server's answers, the AS, the store (a successful `mcp-login` refresh persists its
//! token, as `ServerAuth::refresh` does) and the clock. Transitions match the shell one for one:
//! lock, `Event`, `Action`, AS call (time `started` before, `done` after), `Event::Refreshed`,
//! reply; a reload is `lock, read the store, Event::Stored`.
//!
//! Each property is judged against the environment's own facts (`sh_*`, shadows of what the AS
//! answered and when, and the tokens the callers hold), never by asking the kernel what it
//! should have done.
//!
//! ## Properties (`always`)
//!
//! - S1 single flight: at most one AS call outstanding.
//! - S2 no stale refresh: a 401 for a token already replaced replies the current token, no AS call.
//! - S3 backoff honoured: while a transient failure stands (for a 401's refresh, and separately
//!   for a step-up), no AS call for that purpose.
//! - S4 a definitive failure is final until the store changes: no AS call for a 401's refresh
//!   until a reload reads a different stored token.
//! - S5 a step-up failure never disturbs the held token.
//! - S6 a step-up the AS refused definitively is never asked for again; nor one that succeeded.
//! - L1/L1b/L2 liveness, as a bounded measure under `always`: from every state where the cause
//!   has cleared (L1: a 401's transient failure's backoff has elapsed; L1b: the same for a
//!   step-up; L2: the store holds a token the process has not loaded), the next request that needs
//!   a token gets a fresh one within 2 kernel steps if the AS answers Ok (the measure: the
//!   request, then the AS's answer). Fairness assumed: the AS eventually answers Ok and time
//!   advances (the model's `Tick` and AS answers are always enabled, and the bound on time is the
//!   bound of the check). A TTL is never the only exit: L2's cause is the store, not a timer.
//!   The mutations `mut_backoff_never_ends` and `mut_reload_keeps_failed` remove those exits.
//!
//! Not modelled: a token's lifetime and the renew-before-expiry (`expires_at`) path, and the
//! rmcp refresh's own internals (its answer is the AS's Ok / transient / definitive).
//!
//! ## Mutations
//!
//! `cargo test -p beyond-agent --features <mut> --test it mcp_oauth_model` MUST fail: see
//! `crates/agent/ARCHITECTURE.md` for the feature → property table.

use std::time::Duration;

use beyond_agent::tools::mcp_oauth_kernel as kernel;
use kernel::{Action, Event, Failure, Purpose, Reply};
use stateright::{Checker, Model, Property};

const CALLERS: usize = 2;
const OPS: u8 = 2;
/// Time ends here (ticks of one second); the clock advances one tick at a time.
const TICKS: u8 = 6;
const FIRST: u64 = 2;
const MAX: u64 = 4;

const S1: u32 = 1;
const S2: u32 = 2;
const S3_401: u32 = 4;
const S3_UP: u32 = 8;
const S4: u32 = 16;
const S5: u32 = 32;
const S6: u32 = 64;

fn secs(t: u8) -> Duration {
    Duration::from_secs(u64::from(t))
}

fn name(n: u8) -> String {
    format!("t{n}")
}

fn scopes(l: &[&str]) -> Vec<String> {
    l.iter().map(|s| (*s).to_owned()).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Req {
    Need,
    Rejected(Option<String>),
    Insufficient(Option<String>, Vec<String>),
    Dial,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Ph {
    Idle,
    /// Wants the lock for this request.
    Want(Req),
    /// Holds the lock with an AS call outstanding.
    Call {
        purpose: Purpose,
        scopes: Vec<String>,
        started: u8,
    },
    /// Sent a request with this token; the server has not answered.
    Sent(Option<String>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Caller {
    ph: Ph,
    ops_left: u8,
}

/// A transient failure the environment saw the AS give: when it may be over at the earliest
/// (`lb`: the first backoff) and at the latest (`ub`: the longest), and for which held token.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Transient {
    lb: u8,
    ub: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct St {
    k: kernel::State,
    now: u8,
    /// Who holds the lock.
    lock: Option<usize>,
    callers: Vec<Caller>,
    /// What the `mcp-login` store holds.
    store: Option<String>,
    minted: u8,
    logins: u8,
    bad: u32,
    // What the environment saw the AS answer (the shadows the properties are judged against).
    /// A transient failure of a 401's refresh (or a need's), and the held token then.
    sh401: Option<(Transient, Option<String>)>,
    /// A definitive failure of a 401's refresh, and the held token then.
    sh_dead: Option<Option<String>>,
    /// A transient step-up failure.
    sh_up: Option<Transient>,
    /// Scope sets the AS refused definitively / granted to a step-up.
    up_refused: Vec<Vec<String>>,
    up_granted: Vec<Vec<String>>,
    /// Witnesses.
    saw: u32,
}

const W_SKIPPED: u32 = 1;
const W_STEPPED: u32 = 2;
const W_RECOVERED: u32 = 4;
const W_DEAD: u32 = 8;
const W_RELOADED: u32 = 16;
const W_STEP_REFUSED: u32 = 32;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Act {
    Begin(usize, Req),
    Lock(usize),
    /// The server answers the request the caller sent.
    Accepts(usize),
    Unauthorized(usize),
    Insufficient(usize, Vec<String>),
    As(usize, AsAnswer),
    Tick,
    Login,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum AsAnswer {
    Ok,
    Transient,
    Definitive,
}

struct Oauth {
    cc: bool,
}

impl Oauth {
    fn cfg(&self) -> kernel::Config {
        kernel::Config {
            cc: self.cc,
            first_backoff: Duration::from_secs(FIRST),
            max_backoff: Duration::from_secs(MAX),
            renew_skew: Duration::ZERO,
        }
    }

    /// The caller takes the lock and feeds the kernel `req`.
    fn lock_step(&self, s: &mut St, c: usize, req: &Req) {
        s.lock = Some(c);
        let now = secs(s.now);
        let held = s.k.token.clone();
        let cfg = self.cfg();
        let action = match req {
            Req::Need => kernel::step(&mut s.k, &cfg, Event::Need { now }),
            Req::Rejected(sent) => {
                let a = kernel::step(
                    &mut s.k,
                    &cfg,
                    Event::Rejected {
                        now,
                        sent: sent.as_deref(),
                    },
                );
                // S2, judged on the token the callers hold: replaced means current, no call.
                if let Some(cur) = &held
                    && Some(cur) != sent.as_ref()
                    && a != Action::Reply(Reply::Granted(cur.clone()))
                {
                    s.bad |= S2;
                }
                a
            }
            Req::Insufficient(sent, ch) => kernel::step(
                &mut s.k,
                &cfg,
                Event::Insufficient {
                    now,
                    sent: sent.as_deref(),
                    challenged: ch,
                },
            ),
            Req::Dial => {
                let stored = s.store.clone();
                if let Some(st) = &stored {
                    // A reload that reads a token other than the failed one ends a failure.
                    if s.sh401
                        .as_ref()
                        .is_some_and(|(_, t)| t.as_ref() != Some(st))
                    {
                        s.sh401 = None;
                    }
                    if s.sh_dead.as_ref().is_some_and(|t| t.as_ref() != Some(st)) {
                        s.sh_dead = None;
                    }
                    if held.as_ref() != Some(st) {
                        s.saw |= W_RELOADED;
                    }
                }
                kernel::step(&mut s.k, &cfg, Event::Stored(stored))
            }
        };
        self.perform(s, c, action, req);
    }

    fn perform(&self, s: &mut St, c: usize, action: Action, req: &Req) {
        match action {
            Action::Refresh { purpose, scopes } => {
                if s.callers.iter().any(|o| matches!(o.ph, Ph::Call { .. })) {
                    s.bad |= S1;
                }
                match purpose {
                    Purpose::Need | Purpose::Rejection => {
                        if s.sh401.as_ref().is_some_and(|(t, _)| s.now < t.lb) {
                            s.bad |= S3_401;
                        }
                        if s.sh_dead.is_some() {
                            s.bad |= S4;
                        }
                    }
                    Purpose::StepUp => {
                        if s.sh_up.as_ref().is_some_and(|t| s.now < t.lb) {
                            s.bad |= S3_UP;
                        }
                        if s.up_refused.contains(&scopes) || s.up_granted.contains(&scopes) {
                            s.bad |= S6;
                        }
                    }
                }
                s.callers[c].ph = Ph::Call {
                    purpose,
                    scopes,
                    started: s.now,
                };
            }
            Action::Reply(reply) => {
                if s.lock == Some(c) {
                    s.lock = None;
                }
                // A 401 granted a token without an AS call (the lock step replied straight away).
                if matches!(req, Req::Rejected(_)) && matches!(reply, Reply::Granted(_)) {
                    s.saw |= W_SKIPPED;
                }
                s.callers[c].ph = match (req, reply) {
                    (Req::Need, Reply::Token(t)) => Ph::Sent(t),
                    _ => Ph::Idle,
                };
            }
            Action::Nothing => {
                if s.lock == Some(c) {
                    s.lock = None;
                }
                s.callers[c].ph = Ph::Idle;
            }
        }
    }

    fn as_answer(&self, s: &mut St, c: usize, answer: AsAnswer) {
        let Ph::Call {
            purpose,
            scopes: asked,
            started,
        } = s.callers[c].ph.clone()
        else {
            unreachable!()
        };
        let held = s.k.token.clone();
        let result = match answer {
            AsAnswer::Ok => {
                let t = name(s.minted);
                s.minted += 1;
                if !self.cc {
                    s.store = Some(t.clone()); // the refresh persists what it obtained
                }
                Ok((t, None))
            }
            AsAnswer::Transient => Err(Failure {
                definitive: false,
                reason: "down".into(),
            }),
            AsAnswer::Definitive => Err(Failure {
                definitive: true,
                reason: "invalid_grant".into(),
            }),
        };
        let transient = Transient {
            lb: started + FIRST as u8,
            ub: started + MAX as u8,
        };
        match (purpose, answer) {
            (Purpose::StepUp, AsAnswer::Ok) => {
                s.sh_up = None;
                s.up_granted.push(asked.clone());
                s.saw |= W_STEPPED;
            }
            (Purpose::StepUp, AsAnswer::Transient) => s.sh_up = Some(transient),
            (Purpose::StepUp, AsAnswer::Definitive) => {
                s.sh_up = None;
                s.up_refused.push(asked.clone());
                s.saw |= W_STEP_REFUSED;
            }
            (_, AsAnswer::Ok) => {
                if s.sh401.take().is_some() {
                    s.saw |= W_RECOVERED;
                }
            }
            (_, AsAnswer::Transient) => s.sh401 = Some((transient, held.clone())),
            (_, AsAnswer::Definitive) => {
                s.sh401 = None;
                s.sh_dead = Some(held.clone());
                s.saw |= W_DEAD;
            }
        }
        let req = match purpose {
            Purpose::Need => Req::Need,
            Purpose::Rejection => Req::Rejected(None),
            Purpose::StepUp => Req::Insufficient(None, vec![]),
        };
        let cfg = self.cfg();
        let action = kernel::step(
            &mut s.k,
            &cfg,
            Event::Refreshed {
                started: secs(started),
                done: secs(s.now),
                purpose,
                scopes: asked,
                result,
            },
        );
        if purpose == Purpose::StepUp && answer != AsAnswer::Ok && s.k.token != held {
            s.bad |= S5;
        }
        self.perform(s, c, action, &req);
    }

    /// A progress check: the kernel, given `events` in turn with the AS answering Ok, grants a
    /// fresh token within `events.len() + 1` steps. `None`: it does.
    fn progress(&self, s: &St, request: Event<'_>, purpose: Purpose) -> bool {
        let mut k = s.k.clone();
        let cfg = self.cfg();
        let now = secs(s.now);
        let Action::Refresh {
            purpose: asked,
            scopes: sc,
        } = kernel::step(&mut k, &cfg, request)
        else {
            return false;
        };
        if asked != purpose {
            return false;
        }
        matches!(
            kernel::step(
                &mut k,
                &cfg,
                Event::Refreshed {
                    started: now,
                    done: now,
                    purpose,
                    scopes: sc,
                    result: Ok(("fresh".into(), None)),
                },
            ),
            Action::Reply(Reply::Granted(_))
        )
    }
}

impl Model for Oauth {
    type State = St;
    type Action = Act;

    fn init_states(&self) -> Vec<St> {
        let (token, store, sc) = if self.cc {
            (None, None, scopes(&["a"]))
        } else {
            (Some(name(0)), Some(name(0)), vec![])
        };
        vec![St {
            k: kernel::State {
                token,
                scopes: sc,
                ..kernel::State::default()
            },
            now: 0,
            lock: None,
            callers: vec![
                Caller {
                    ph: Ph::Idle,
                    ops_left: OPS,
                };
                CALLERS
            ],
            store,
            minted: 1,
            logins: 0,
            bad: 0,
            sh401: None,
            sh_dead: None,
            sh_up: None,
            up_refused: vec![],
            up_granted: vec![],
            saw: 0,
        }]
    }

    fn actions(&self, s: &St, out: &mut Vec<Act>) {
        if s.now < TICKS {
            out.push(Act::Tick);
        }
        if !self.cc && s.logins < 1 {
            out.push(Act::Login);
        }
        for (c, caller) in s.callers.iter().enumerate() {
            match &caller.ph {
                Ph::Idle if caller.ops_left > 0 => {
                    out.push(Act::Begin(c, Req::Need));
                    if !self.cc {
                        out.push(Act::Begin(c, Req::Dial));
                    }
                }
                Ph::Idle => {}
                Ph::Want(_) => {
                    if s.lock.is_none() || cfg!(feature = "mut_unlocked") {
                        out.push(Act::Lock(c));
                    }
                }
                Ph::Call { .. } => {
                    for a in [AsAnswer::Ok, AsAnswer::Transient, AsAnswer::Definitive] {
                        out.push(Act::As(c, a));
                    }
                }
                Ph::Sent(_) => {
                    out.push(Act::Accepts(c));
                    out.push(Act::Unauthorized(c));
                    if self.cc {
                        for ch in [vec![], scopes(&["a"]), scopes(&["b"])] {
                            out.push(Act::Insufficient(c, ch));
                        }
                    }
                }
            }
        }
    }

    fn next_state(&self, s: &St, a: Act) -> Option<St> {
        let mut n = s.clone();
        match a {
            Act::Tick => n.now += 1,
            Act::Login => {
                n.logins += 1;
                n.store = Some(name(n.minted));
                n.minted += 1;
            }
            Act::Begin(c, req) => {
                n.callers[c].ops_left -= 1;
                n.callers[c].ph = Ph::Want(req);
            }
            Act::Lock(c) => {
                let Ph::Want(req) = n.callers[c].ph.clone() else {
                    return None;
                };
                self.lock_step(&mut n, c, &req);
            }
            Act::Accepts(c) => n.callers[c].ph = Ph::Idle,
            Act::Unauthorized(c) => {
                let Ph::Sent(t) = n.callers[c].ph.clone() else {
                    return None;
                };
                n.callers[c].ph = Ph::Want(Req::Rejected(t));
            }
            Act::Insufficient(c, ch) => {
                let Ph::Sent(t) = n.callers[c].ph.clone() else {
                    return None;
                };
                n.callers[c].ph = Ph::Want(Req::Insufficient(t, ch));
            }
            Act::As(c, answer) => self.as_answer(&mut n, c, answer),
        }
        Some(n)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        let mut p = vec![
            Property::<Self>::always("S1: at most one AS call outstanding", |_, s| {
                s.bad & S1 == 0
            }),
            Property::<Self>::always(
                "S2: a 401 for a replaced token returns the current one with no AS call",
                |_, s| s.bad & S2 == 0,
            ),
            Property::<Self>::always(
                "S3: no AS call for a 401's refresh while its transient failure's backoff stands",
                |_, s| s.bad & S3_401 == 0,
            ),
            Property::<Self>::always(
                "S3: no AS call for a step-up while its transient failure's backoff stands",
                |_, s| s.bad & S3_UP == 0,
            ),
            Property::<Self>::always(
                "S4: no AS call after a definitive failure until a reload reads a new stored token",
                |_, s| s.bad & S4 == 0,
            ),
            Property::<Self>::always(
                "S5: a step-up failure never disturbs the held token",
                |_, s| s.bad & S5 == 0,
            ),
            Property::<Self>::always(
                "S6: a step-up refused (or granted) is never requested again",
                |_, s| s.bad & S6 == 0,
            ),
            Property::<Self>::always(
                "L1: after a 401 refresh's backoff elapsed, the next request is granted within 2 steps",
                |m, s| {
                    let Some((t, _)) = &s.sh401 else { return true };
                    if s.now < t.ub || s.lock.is_some() || s.sh_dead.is_some() {
                        return true;
                    }
                    let tok = s.k.token.clone();
                    m.progress(
                        s,
                        Event::Rejected {
                            now: secs(s.now),
                            sent: tok.as_deref(),
                        },
                        Purpose::Rejection,
                    )
                },
            ),
            Property::<Self>::always(
                "L1: after a step-up's backoff elapsed, the next 403 is granted within 2 steps",
                |m, s| {
                    let Some(t) = &s.sh_up else { return true };
                    if s.now < t.ub || s.lock.is_some() || !m.cc {
                        return true;
                    }
                    let tok = s.k.token.clone();
                    let ch = scopes(&["b"]);
                    m.progress(
                        s,
                        Event::Insufficient {
                            now: secs(s.now),
                            sent: tok.as_deref(),
                            challenged: &ch,
                        },
                        Purpose::StepUp,
                    )
                },
            ),
            Property::<Self>::always(
                "L2: once the stored token changed, the next request is granted within 3 steps",
                |m, s| {
                    let Some(stored) = &s.store else { return true };
                    if m.cc || s.lock.is_some() || s.k.token.as_ref() == Some(stored) {
                        return true;
                    }
                    // The reload (1), the request (2), the AS's answer (3).
                    let mut n = s.clone();
                    let cfg = m.cfg();
                    kernel::step(&mut n.k, &cfg, Event::Stored(Some(stored.clone())));
                    m.progress(
                        &n,
                        Event::Rejected {
                            now: secs(s.now),
                            sent: Some(stored),
                        },
                        Purpose::Rejection,
                    )
                },
            ),
            Property::<Self>::sometimes("W: a stale 401 takes the current token", |_, s| {
                s.saw & W_SKIPPED != 0
            }),
            Property::<Self>::sometimes("W: a definitive failure happens", |_, s| {
                s.saw & W_DEAD != 0
            }),
            Property::<Self>::sometimes("W: a 401's transient failure recovers", |_, s| {
                s.saw & W_RECOVERED != 0
            }),
        ];
        if self.cc {
            p.push(Property::<Self>::sometimes(
                "W: a step-up succeeds",
                |_, s| s.saw & W_STEPPED != 0,
            ));
            p.push(Property::<Self>::sometimes(
                "W: a step-up is refused definitively",
                |_, s| s.saw & W_STEP_REFUSED != 0,
            ));
        } else {
            p.push(Property::<Self>::sometimes(
                "W: a reload reads a changed token",
                |_, s| s.saw & W_RELOADED != 0,
            ));
        }
        p
    }

    fn within_boundary(&self, s: &St) -> bool {
        s.now <= TICKS
    }
}

fn check(cc: bool, label: &str) {
    let checker = Oauth { cc }.checker().spawn_bfs().join();
    println!(
        "{label}: {} states, {} unique",
        checker.state_count(),
        checker.unique_state_count()
    );
    checker.assert_properties();
}

/// Every property holds over every interleaving for a client-credentials server (needs, 401s,
/// step-ups) ...
#[test]
fn faithful_client_credentials() {
    check(true, "client credentials");
}

/// ... and for an `mcp-login` server (401s, reloads from a store another process changes).
#[test]
fn faithful_login() {
    check(false, "login");
}

/// With a `mut_*` feature on, the property that MUST be red for it, and the config it is checked in
/// (`cc`). The faithful tests above fail too under any mutation; this one names the property.
#[cfg(any(
    feature = "mut_no_stepup_backoff",
    feature = "mut_no_401_backoff",
    feature = "mut_stale_refresh",
    feature = "mut_no_definitive_latch",
    feature = "mut_no_refused_memory",
    feature = "mut_backoff_never_ends",
    feature = "mut_reload_keeps_failed",
    feature = "mut_unlocked"
))]
#[test]
fn the_mutation_turns_its_property_red() {
    let (cc, property) = if cfg!(feature = "mut_no_stepup_backoff") {
        (true, "S3: no AS call for a step-up")
    } else if cfg!(feature = "mut_no_401_backoff") {
        (false, "S3: no AS call for a 401's refresh")
    } else if cfg!(feature = "mut_stale_refresh") {
        (false, "S2: ")
    } else if cfg!(feature = "mut_no_definitive_latch") {
        (false, "S4: ")
    } else if cfg!(feature = "mut_no_refused_memory") {
        (true, "S6: ")
    } else if cfg!(feature = "mut_backoff_never_ends") {
        (false, "L1: after a 401 refresh's backoff elapsed")
    } else if cfg!(feature = "mut_reload_keeps_failed") {
        (false, "L2: ")
    } else {
        (false, "S1: ")
    };
    let checker = Oauth { cc }.checker().spawn_bfs().join();
    let red: Vec<&str> = Oauth { cc }
        .properties()
        .iter()
        .filter(|p| p.name.starts_with(property) && checker.discovery(p.name).is_some())
        .map(|p| p.name)
        .collect();
    println!("red: {red:?}");
    assert!(!red.is_empty(), "no property `{property}` went red");
}
