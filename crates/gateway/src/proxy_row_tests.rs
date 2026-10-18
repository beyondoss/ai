//! The pure pieces of the `ai.usage` row's new fields.

use super::*;

#[test]
fn price_variant_names_bedrock_profiles_and_openrouter_regions() {
    assert_eq!(
        price_variant(
            "bedrock",
            "global.anthropic.claude-opus-4-8",
            "bedrock-runtime.us-east-1.amazonaws.com"
        ),
        Some("global")
    );
    for m in [
        "us.anthropic.claude-opus-4-8",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-opus-4-8",
    ] {
        assert_eq!(price_variant("bedrock", m, "h"), Some("regional"), "{m}");
    }
    assert_eq!(
        price_variant("openrouter", "x", "us.openrouter.ai"),
        Some("regional")
    );
    assert_eq!(price_variant("openrouter", "x", "openrouter.ai"), None);
    assert_eq!(price_variant("openrouter", "x", "evilopenrouter.ai"), None);
    assert_eq!(
        price_variant("anthropic", "global.x", "api.anthropic.com"),
        None
    );
}

#[test]
fn only_a_streamed_cancel_on_a_continuing_provider_may_continue() {
    for p in MAY_CONTINUE_PROVIDERS {
        let may = |outcome, streaming| upstream_may_continue(Some(p), outcome, streaming, false);
        assert!(may("client_cancelled", true), "{p}");
        assert!(may("cut_short", true), "{p}");
        assert!(!may("client_cancelled", false), "{p}");
        assert!(!may("ok", true), "{p}");
        assert!(!may("upstream_error", true), "{p}");
        // A drained stream was read to its end: nothing past the row's counts.
        assert!(
            !upstream_may_continue(Some(p), "client_cancelled", true, true),
            "{p}"
        );
    }
    assert!(!upstream_may_continue(
        Some("anthropic"),
        "client_cancelled",
        true,
        false
    ));
    assert!(!upstream_may_continue(
        None,
        "client_cancelled",
        true,
        false
    ));
}

#[test]
fn estimated_parts_and_exclusions() {
    let reported = usage::Usage {
        reasoning_tokens: Some(3),
        ..usage::Usage::default()
    };
    let unreported = usage::Usage::default();
    let p = |input, output| EstimatedParts { input, output };
    assert_eq!(p(false, false).parts(), None);
    assert_eq!(p(true, false).parts(), Some("input"));
    assert_eq!(p(false, true).parts(), Some("output"));
    assert_eq!(p(true, true).parts(), Some("input,output"));
    assert_eq!(p(false, false).excludes(&unreported), None);
    assert_eq!(p(true, false).excludes(&unreported), Some("cache"));
    assert_eq!(p(false, true).excludes(&unreported), Some("reasoning"));
    assert_eq!(p(false, true).excludes(&reported), None);
    assert_eq!(p(true, true).excludes(&unreported), Some("cache,reasoning"));
    assert_eq!(p(true, true).excludes(&reported), Some("cache"));
}

#[test]
fn only_the_gateways_own_downstream_answer_is_a_refusal() {
    let down = |t| pingora_core::Error::new(t).into_down();
    let up = |t| pingora_core::Error::new(t).into_up();
    assert!(gateway_refusal(&down(pingora_core::ErrorType::CustomCode(
        "x", 400
    ))));
    assert!(gateway_refusal(&down(pingora_core::ErrorType::HTTPStatus(
        413
    ))));
    assert!(!gateway_refusal(&down(pingora_core::ErrorType::ReadError)));
    assert!(!gateway_refusal(&up(pingora_core::ErrorType::HTTPStatus(
        400
    ))));
}

/// The fast-mode beta is added once: a header already carrying it (alone or among others, with
/// or without spaces) is left as it is, and any other value gains it.
#[test]
fn merge_anthropic_beta_adds_a_beta_once() {
    let beta = translate::FAST_MODE_BETA;
    let header = |v: Option<&str>| {
        let mut req = pingora::http::RequestHeader::build("POST", b"/v1/messages", None).unwrap();
        if let Some(v) = v {
            req.insert_header("anthropic-beta", v.to_owned()).unwrap();
        }
        merge_anthropic_beta(&mut req, beta).unwrap();
        req.headers
            .get("anthropic-beta")
            .map(|v| v.to_str().unwrap().to_owned())
    };
    assert_eq!(header(None).as_deref(), Some(beta));
    assert_eq!(header(Some("")).as_deref(), Some(beta));
    assert_eq!(header(Some(beta)).as_deref(), Some(beta));
    let both = format!("prompt-caching-2024-07-31, {beta}");
    assert_eq!(header(Some(&both)), Some(both.clone()));
    assert_eq!(
        header(Some("prompt-caching-2024-07-31")),
        Some(format!("prompt-caching-2024-07-31,{beta}"))
    );
}

/// The row fields an ending decides, built as `logging` builds them: `outcome`,
/// `usage_estimated`, `cache_hit`, `usage_settled`, `usage_estimated_parts` and
/// `upstream_may_continue`. `None` writes no row.
type Fields = (
    &'static str,
    bool,
    bool,
    Option<&'static str>,
    Option<&'static str>,
    bool,
);

/// The row's `provider`, as `logging` names it: the cache entry's, or the leg's once a provider
/// was handed the request, and absent before any call.
fn named<'a>(f: &EndFacts, provider: &'a str) -> Option<&'a str> {
    (f.cache_hit || f.called).then_some(provider)
}

fn fields(f: &EndFacts, provider: &str, stream: bool) -> Option<Fields> {
    Ending::of(f).row().map(|end| {
        (
            end.outcome,
            end.count.estimated(),
            end.cache_hit,
            end.usage_settled(),
            end.count.parts().parts(),
            upstream_may_continue(named(f, provider), end.outcome, stream, end.drained),
        )
    })
}

/// The row gate, `outcome` and flag logic `logging` ran before [`Ending`], transcribed as it was:
/// the reference the new path is held to.
fn fields_before(f: &EndFacts, provider: &str, stream: bool) -> Option<Fields> {
    let refused = !f.cache_hit && f.status.is_none() && f.fault == Some(Fault::Refusal);
    if f.relay_abandoned || f.free || refused {
        return None;
    }
    let downstream = matches!(f.fault, Some(Fault::Client | Fault::Refusal));
    let outcome = if f.cache_hit {
        "ok"
    } else if f.status.is_some_and(|s| s >= 400) {
        "upstream_error"
    } else if f.client_left || downstream {
        "client_cancelled"
    } else if !f.called {
        "no_candidate"
    } else {
        match (f.fault, f.status) {
            (Some(_), None) => "upstream_error",
            (Some(_), Some(_)) => "cut_short",
            (None, _) => "ok",
        }
    };
    let usage_estimated = f.count.estimated();
    Some((
        outcome,
        usage_estimated,
        f.cache_hit,
        (f.drained && !usage_estimated).then_some("drained"),
        f.count.parts().parts(),
        upstream_may_continue(named(f, provider), outcome, stream, f.drained),
    ))
}

/// Facts `logging` can see. An estimate needs a 2xx head, or a connected attempt with no head
/// (`gave_up_waiting` on a delivered body), and never comes from the cache. A drain arms only on
/// a managed 2xx stream's head, and only the client leaving starts it. A response head comes from
/// a provider the request was handed, unless it is the cache's.
fn reachable(f: &EndFacts) -> bool {
    if f.status.is_some() && !f.cache_hit && !f.called {
        return false;
    }
    let ok_2xx = f.status.is_some_and(|s| (200..300).contains(&s));
    let estimate_ok =
        !f.count.estimated() || (!f.cache_hit && (ok_2xx || (f.status.is_none() && f.called)));
    let drain_ok = !f.drained || (f.client_left && ok_2xx && !f.cache_hit);
    estimate_ok && drain_ok
}

/// Every reachable combination of the facts bills exactly as it did before [`Ending`]: the same
/// row or none, with the same `outcome`, estimate flags, `usage_settled` and
/// `upstream_may_continue`.
#[test]
fn every_ending_bills_as_before() {
    let parts = |input, output| EstimatedParts { input, output };
    let counts = [
        Count::Reported,
        Count::Estimated(parts(false, false)),
        Count::Estimated(parts(true, false)),
        Count::Estimated(parts(false, true)),
        Count::Estimated(parts(true, true)),
    ];
    let faults = [
        None,
        Some(Fault::Client),
        Some(Fault::Refusal),
        Some(Fault::Upstream),
    ];
    let mut checked = 0;
    for bits in 0u8..64 {
        let bit = |i: u8| bits & (1 << i) != 0;
        for status in [
            None,
            Some(101),
            Some(200),
            Some(206),
            Some(304),
            Some(404),
            Some(429),
            Some(503),
        ] {
            for fault in faults {
                for count in counts {
                    let f = EndFacts {
                        relay_abandoned: bit(0),
                        free: bit(1),
                        cache_hit: bit(2),
                        client_left: bit(3),
                        called: bit(4),
                        drained: bit(5),
                        status,
                        fault,
                        count,
                    };
                    if !reachable(&f) {
                        continue;
                    }
                    for provider in ["openrouter", "anthropic"] {
                        for stream in [false, true] {
                            assert_eq!(
                                fields(&f, provider, stream),
                                fields_before(&f, provider, stream),
                                "{f:?} {provider} stream={stream}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 5376, "the reachable space shrank");
}

/// Each ending's row, pinned: what it writes, or that it writes none.
#[test]
fn each_ending_has_its_row() {
    let est = Count::Estimated(EstimatedParts {
        input: true,
        output: true,
    });
    let called = EndFacts {
        called: true,
        ..EndFacts::default()
    };
    let at = |status| EndFacts {
        status: Some(status),
        ..called
    };
    let cases: [(EndFacts, Ending, Option<Fields>); 14] = [
        (
            EndFacts {
                relay_abandoned: true,
                ..at(503)
            },
            Ending::RelayAbandoned,
            None,
        ),
        (
            EndFacts {
                free: true,
                ..at(200)
            },
            Ending::Free,
            None,
        ),
        (
            EndFacts {
                cache_hit: true,
                status: Some(200),
                ..EndFacts::default()
            },
            Ending::CacheHit(Count::Reported),
            Some(("ok", false, true, None, None, false)),
        ),
        (
            EndFacts {
                fault: Some(Fault::Refusal),
                ..called
            },
            Ending::Refused,
            None,
        ),
        (
            at(429),
            Ending::UpstreamStatus(Count::Reported),
            Some(("upstream_error", false, false, None, None, false)),
        ),
        (
            EndFacts {
                fault: Some(Fault::Client),
                ..at(503)
            },
            Ending::UpstreamStatus(Count::Reported),
            Some(("upstream_error", false, false, None, None, false)),
        ),
        (
            EndFacts {
                fault: Some(Fault::Client),
                count: est,
                ..at(200)
            },
            Ending::ClientCancelled {
                count: est,
                drained: false,
            },
            Some((
                "client_cancelled",
                true,
                false,
                None,
                Some("input,output"),
                true,
            )),
        ),
        (
            EndFacts {
                client_left: true,
                drained: true,
                ..at(200)
            },
            Ending::ClientCancelled {
                count: Count::Reported,
                drained: true,
            },
            Some((
                "client_cancelled",
                false,
                false,
                Some("drained"),
                None,
                false,
            )),
        ),
        (
            EndFacts {
                client_left: true,
                drained: true,
                count: est,
                ..at(200)
            },
            Ending::ClientCancelled {
                count: est,
                drained: true,
            },
            Some((
                "client_cancelled",
                true,
                false,
                None,
                Some("input,output"),
                false,
            )),
        ),
        (
            EndFacts::default(),
            Ending::NoCandidate(Count::Reported),
            Some(("no_candidate", false, false, None, None, false)),
        ),
        (
            EndFacts {
                fault: Some(Fault::Upstream),
                count: est,
                ..called
            },
            Ending::UpstreamFailed(est),
            Some((
                "upstream_error",
                true,
                false,
                None,
                Some("input,output"),
                false,
            )),
        ),
        (
            EndFacts {
                fault: Some(Fault::Upstream),
                ..at(200)
            },
            Ending::CutShort(Count::Reported),
            Some(("cut_short", false, false, None, None, true)),
        ),
        (
            at(200),
            Ending::Completed(Count::Reported),
            Some(("ok", false, false, None, None, false)),
        ),
        (
            EndFacts {
                count: est,
                ..at(200)
            },
            Ending::Completed(est),
            Some(("ok", true, false, None, Some("input,output"), false)),
        ),
    ];
    for (facts, ending, row) in cases {
        assert_eq!(Ending::of(&facts), ending, "{facts:?}");
        assert_eq!(fields(&facts, "openrouter", true), row, "{ending:?}");
        assert_eq!(fields_before(&facts, "openrouter", true), row, "{ending:?}");
    }
}

/// The error a request ended with blames the client, the gateway's own answer, or the upstream.
#[test]
fn a_fault_is_read_from_the_error_source() {
    use pingora_core::{Error, ErrorType as T};
    assert_eq!(Fault::of(&Error::new_down(T::ReadError)), Fault::Client);
    assert_eq!(
        Fault::of(&Error::new_down(T::CustomCode("x", 400))),
        Fault::Refusal
    );
    assert_eq!(
        Fault::of(&Error::new_up(T::ConnectRefused)),
        Fault::Upstream
    );
    assert_eq!(Fault::of(&Error::new(T::ReadTimedout)), Fault::Upstream);
}
