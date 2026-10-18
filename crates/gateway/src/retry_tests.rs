//! The retry decision, held to the inline rules the hooks ran before it (transcribed below as they
//! were) over every combination of facts the hooks can report, and pinned case by case.

use super::*;

const STAND: Decision = Decision::STAND;
const SAME: Decision = Decision::Resend(Retry::SameLeg);
const NEXT: Decision = Decision::Resend(Retry::NextCandidate);

/// `upstream_response_filter`'s rules as it ran them inline.
fn status_before(
    status: u16,
    managed: bool,
    next_key: Option<u8>,
    walk: Option<Walk>,
    body: Body,
) -> Decision {
    let key_auth = managed && status == 401;
    let mut stand = Stand::default();
    if managed && (status == 429 || key_auth) {
        if let Some(next) = next_key {
            if body.replayable {
                return SAME;
            }
            if body.parent.is_some()
                && let Some(orig) = walk.and_then(|w| w.orig)
            {
                return Decision::Rerun(RelayRetry::Key {
                    candidate: orig,
                    key: next,
                });
            }
            stand.key_walk_unreplayable = true;
        }
        if status == 429 {
            return Decision::Stand(stand);
        }
    }
    let Some(w) = walk else {
        return Decision::Stand(stand);
    };
    let key_failure = managed && (401..=403).contains(&status);
    if status < 500 && !key_failure {
        return Decision::Stand(stand);
    }
    if !w.next {
        return Decision::Stand(stand);
    }
    if !body.replayable
        && body.parent.is_some()
        && let Some(orig) = w.orig
    {
        return Decision::Rerun(RelayRetry::Candidate(orig));
    }
    if !body.replayable {
        stand.failover_unreplayable = true;
        return Decision::Stand(stand);
    }
    NEXT
}

/// `fail_to_connect`'s rules as it ran them inline.
fn connect_before(walk: Option<Walk>, retries_left: bool) -> Decision {
    match walk {
        Some(w) if w.next_addr => SAME,
        Some(w) if w.next => NEXT,
        Some(_) => STAND,
        None if retries_left => SAME,
        None => STAND,
    }
}

/// `error_while_proxy`'s rules as it ran them inline, with the leg's `refused_resent` after.
fn lost_before(l: &Lost, walk: Option<Walk>, body: Body) -> (Decision, bool) {
    let mut refused_resent = l.refused_resent;
    let delivered = !l.refused && !l.reset_unread && l.delivered;
    if l.downstream || l.responded || delivered {
        return (STAND, refused_resent);
    }
    let drained = l.refused && l.reused && l.goaway;
    let resend_refused = l.refused && (drained || !l.refused_resent);
    let walk = walk.and_then(|w| Some((w.orig?, w.next)));
    if let Some(parent) = body.parent {
        if let Some((orig, next)) = walk {
            let retry = if (l.reused || l.refused) && !parent.reset_spent {
                Some(RelayRetry::Reset(orig))
            } else if next {
                Some(RelayRetry::Candidate(orig))
            } else {
                None
            };
            if let Some(retry) = retry {
                return (Decision::Rerun(retry), refused_resent);
            }
        }
        return (STAND, refused_resent);
    }
    if body.truncated {
        return (STAND, refused_resent);
    }
    let decision = match walk {
        Some(_) if resend_refused || (l.reused && !l.refused) => {
            refused_resent |= l.refused && !drained;
            SAME
        }
        Some((_, true)) => NEXT,
        Some(_) => STAND,
        None if resend_refused => {
            refused_resent |= !drained;
            SAME
        }
        None if l.refused => STAND,
        None if l.h2_body_unsent => {
            if l.reused {
                SAME
            } else {
                STAND
            }
        }
        None if l.pingora_retries => SAME,
        None => STAND,
    };
    (decision, refused_resent)
}

fn walks() -> Vec<Option<Walk>> {
    let mut out = vec![None];
    for orig in [None, Some(0), Some(3)] {
        for next in [false, true] {
            for next_addr in [false, true] {
                out.push(Some(Walk {
                    orig,
                    next,
                    next_addr,
                }));
            }
        }
    }
    out
}

fn bodies() -> Vec<Body> {
    let mut out = Vec::new();
    for replayable in [false, true] {
        for truncated in [false, true] {
            for parent in [
                None,
                Some(Parent { reset_spent: false }),
                Some(Parent { reset_spent: true }),
            ] {
                out.push(Body {
                    replayable,
                    truncated,
                    parent,
                });
            }
        }
    }
    out
}

/// Every status answer, on every route and body, decides as `upstream_response_filter` did.
#[test]
fn every_status_answer_retries_as_before() {
    for status in [200, 304, 400, 401, 402, 403, 404, 408, 429, 500, 502, 503] {
        for managed in [false, true] {
            for next_key in [None, Some(1)] {
                for walk in walks() {
                    for body in bodies() {
                        let failed = Failed {
                            failure: Failure::Status {
                                status,
                                managed,
                                next_key,
                            },
                            walk,
                            body,
                        };
                        assert_eq!(
                            decide(&failed),
                            status_before(status, managed, next_key, walk, body),
                            "{failed:?}"
                        );
                    }
                }
            }
        }
    }
}

/// Below 400 nothing walks, so the hook reads no further fact for a success.
#[test]
fn a_success_never_retries() {
    for status in [200, 204, 304, 399] {
        for walk in walks() {
            for body in bodies() {
                let failed = Failed {
                    failure: Failure::Status {
                        status,
                        managed: true,
                        next_key: Some(1),
                    },
                    walk,
                    body,
                };
                assert_eq!(decide(&failed), STAND, "{failed:?}");
            }
        }
    }
}

/// Every connect failure decides as `fail_to_connect` did, whatever the body.
#[test]
fn every_connect_failure_retries_as_before() {
    for retries_left in [false, true] {
        for walk in walks() {
            for body in bodies() {
                let failed = Failed {
                    failure: Failure::Connect { retries_left },
                    walk,
                    body,
                };
                assert_eq!(
                    decide(&failed),
                    connect_before(walk, retries_left),
                    "{failed:?}"
                );
            }
        }
    }
}

/// Every lost connection decides as `error_while_proxy` did, and books the refused-stream resend
/// the same way: all 2^10 combinations of its facts, on every route and body.
#[test]
fn every_lost_connection_retries_as_before() {
    let mut checked = 0u32;
    for bits in 0u16..1024 {
        let bit = |i: u16| bits & (1 << i) != 0;
        let lost = Lost {
            downstream: bit(0),
            responded: bit(1),
            delivered: bit(2),
            refused: bit(3),
            goaway: bit(4),
            reset_unread: bit(5),
            reused: bit(6),
            refused_resent: bit(7),
            h2_body_unsent: bit(8),
            pingora_retries: bit(9),
        };
        for walk in walks() {
            for body in bodies() {
                let failed = Failed {
                    failure: Failure::Lost(lost),
                    walk,
                    body,
                };
                let decision = decide(&failed);
                let booked = lost.refused_resent || (decision == SAME && lost.spends_resend());
                assert_eq!(
                    (decision, booked),
                    lost_before(&lost, walk, body),
                    "{failed:?}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 1024 * 13 * 12);
}

/// The rule: a body the provider may have is never resent, unless it proved it unprocessed.
#[test]
fn a_delivered_body_is_resent_only_when_the_provider_proved_it_unprocessed() {
    let walk = Some(Walk {
        orig: Some(0),
        next: true,
        next_addr: true,
    });
    let delivered = Lost {
        delivered: true,
        reused: true,
        pingora_retries: true,
        ..Lost::default()
    };
    for body in bodies() {
        for walk in [None, walk] {
            let decide = |lost: Lost| {
                decide(&Failed {
                    failure: Failure::Lost(lost),
                    walk,
                    body,
                })
            };
            assert_eq!(decide(delivered), STAND, "{body:?} {walk:?}");
            // Proved unprocessed: an HTTP/2 refusal, or a pooled connection reset unread.
            for proof in [
                Lost {
                    refused: true,
                    ..delivered
                },
                Lost {
                    reset_unread: true,
                    ..delivered
                },
            ] {
                assert!(proof.may_resend(), "{proof:?}");
            }
            assert!(!delivered.may_resend());
        }
    }
}

/// Each hook's outcomes, by name.
#[test]
fn each_retry_decision_by_name() {
    let walk = |orig, next| {
        Some(Walk {
            orig: Some(orig),
            next,
            next_addr: false,
        })
    };
    let small = Body {
        replayable: true,
        ..Body::default()
    };
    let streaming = Body::default();
    let large = Body {
        truncated: true,
        ..Body::default()
    };
    let parent = Body {
        truncated: true,
        parent: Some(Parent::default()),
        ..Body::default()
    };
    let status = |status, next_key, walk, body| Failed {
        failure: Failure::Status {
            status,
            managed: true,
            next_key,
        },
        walk,
        body,
    };
    let lost = |lost, walk, body| Failed {
        failure: Failure::Lost(lost),
        walk,
        body,
    };
    let connect = |retries_left, walk| Failed {
        failure: Failure::Connect { retries_left },
        walk,
        body: Body::default(),
    };
    let reused = Lost {
        reused: true,
        ..Lost::default()
    };
    let refused = Lost {
        refused: true,
        delivered: true,
        ..Lost::default()
    };
    let unreplayable_key = Decision::Stand(Stand {
        key_walk_unreplayable: true,
        failover_unreplayable: false,
    });
    let unreplayable_failover = Decision::Stand(Stand {
        key_walk_unreplayable: false,
        failover_unreplayable: true,
    });
    let cases: [(&str, Failed, Decision); 24] = [
        ("429, next key", status(429, Some(1), None, small), SAME),
        (
            "429, last key",
            status(429, None, walk(0, true), small),
            STAND,
        ),
        (
            "429, next key, body streaming",
            status(429, Some(1), None, streaming),
            unreplayable_key,
        ),
        (
            "429, next key, FullBody",
            status(429, Some(2), walk(1, true), parent),
            Decision::Rerun(RelayRetry::Key {
                candidate: 1,
                key: 2,
            }),
        ),
        (
            "401, next key",
            status(401, Some(1), walk(0, true), small),
            SAME,
        ),
        (
            "401, last key, walk",
            status(401, None, walk(0, true), small),
            NEXT,
        ),
        (
            "401, last key, provider route",
            status(401, None, None, small),
            STAND,
        ),
        (
            "402, walk",
            status(402, Some(1), walk(0, true), small),
            NEXT,
        ),
        ("404, walk", status(404, None, walk(0, true), small), STAND),
        ("5xx, walk", status(503, None, walk(0, true), small), NEXT),
        (
            "5xx, last candidate",
            status(503, None, walk(0, false), small),
            STAND,
        ),
        ("5xx, provider route", status(503, None, None, small), STAND),
        (
            "5xx, body streaming",
            status(503, None, walk(0, true), streaming),
            unreplayable_failover,
        ),
        (
            "5xx, FullBody",
            status(503, None, walk(2, true), parent),
            Decision::Rerun(RelayRetry::Candidate(2)),
        ),
        ("connect, provider route", connect(true, None), SAME),
        ("connect, out of retries", connect(false, None), STAND),
        ("connect, walk", connect(false, walk(0, true)), NEXT),
        (
            "connect, next address",
            connect(
                false,
                Some(Walk {
                    orig: Some(0),
                    next: true,
                    next_addr: true,
                }),
            ),
            SAME,
        ),
        ("reused, walk", lost(reused, walk(0, true), small), SAME),
        (
            "fresh, walk",
            lost(Lost::default(), walk(0, true), small),
            NEXT,
        ),
        ("refused once", lost(refused, None, small), SAME),
        (
            "refused again",
            lost(
                Lost {
                    refused_resent: true,
                    ..refused
                },
                walk(0, true),
                small,
            ),
            NEXT,
        ),
        (
            "reused, FullBody",
            lost(reused, walk(1, false), parent),
            Decision::Rerun(RelayRetry::Reset(1)),
        ),
        (
            "reused, large body",
            lost(reused, walk(0, true), large),
            STAND,
        ),
    ];
    for (name, failed, want) in cases {
        assert_eq!(decide(&failed), want, "{name}");
    }
}

/// `error_while_proxy` skips (`proxy::read_lost`) the facts an earlier one makes unreadable (the
/// delivery check once the client is gone, a response went out, or the body is proved
/// unprocessed; a GOAWAY without a refusal; an unsent HTTP/2 body on a refusal). Skipping them
/// never changes the decision.
#[test]
fn the_facts_a_lost_connection_skips_never_change_its_decision() {
    for bits in 0u16..1024 {
        let bit = |i: u16| bits & (1 << i) != 0;
        let lost = Lost {
            downstream: bit(0),
            responded: bit(1),
            delivered: bit(2),
            refused: bit(3),
            goaway: bit(4),
            reset_unread: bit(5),
            reused: bit(6),
            refused_resent: bit(7),
            h2_body_unsent: bit(8),
            pingora_retries: bit(9),
        };
        let skipped = crate::proxy::read_lost(
            lost,
            || lost.delivered,
            || lost.goaway,
            || lost.h2_body_unsent,
        );
        assert_eq!(lost.spends_resend(), skipped.spends_resend(), "{lost:?}");
        for walk in walks() {
            for body in bodies() {
                let decide = |lost| {
                    decide(&Failed {
                        failure: Failure::Lost(lost),
                        walk,
                        body,
                    })
                };
                assert_eq!(decide(lost), decide(skipped), "{lost:?} {walk:?} {body:?}");
            }
        }
    }
}
