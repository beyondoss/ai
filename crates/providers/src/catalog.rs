//! The model catalog — canonical model name → the ordered providers that can serve it.
//!
//! This is the inverse of [`crate::for_model_id`], and the general form of it. `for_model_id` answers
//! "which provider serves this id *shape*, natively, when nobody said otherwise" and deliberately
//! returns nothing for an aggregator, because `moonshotai/kimi-k2.6` is equally a Fireworks,
//! Together and OpenRouter id and guessing between them is the mis-route this crate exists to
//! prevent. The catalog is where that guess becomes a *decision*: a named model, and the ordered
//! list of upstreams we are willing to serve it from.
//!
//! It carries routing facts only — which provider, and the id that provider spells it with. Model
//! *capability* facts (context window, thinking shape) stay in `agent_core::models`; a test keeps
//! this file from growing them.
//!
//! # Why every candidate in a row must share a wire format
//!
//! The gateway rewrites the request body's `model` field per attempt, so candidates may spell the
//! model differently (`claude-opus-4-8` at Anthropic, `anthropic/claude-opus-4-8` at OpenRouter).
//! What it does **not** do is translate between API shapes. A row mixing wire formats would take an
//! Anthropic Messages body and send it to a Chat Completions endpoint — a 400 at best, and at worst
//! a response the usage extractor parses with the wrong dialect and bills as zero tokens. So
//! `candidates_agree_on_wire` is load-bearing, not tidiness.
//!
//! The practical consequence, worth knowing before adding rows: **Anthropic is currently the only
//! Anthropic-wire provider the gateway can route to**, so an Anthropic-wire row has exactly one
//! candidate and no failover. Claude *can* be given a multi-candidate row, but only on the OpenAI
//! wire (via OpenRouter and friends), i.e. for a client speaking Chat Completions rather than
//! Messages. Adding a second Anthropic-wire provider is what unlocks native Claude failover.
//!
//! # Maintenance
//!
//! These rows are product data and they go stale — providers rename ids, deprecate models, and
//! change what they host. The seed below is deliberately small and limited to ids verified against
//! `crates/gateway/tests/smoke.rs` (which exercises each provider's real API) or already relied on
//! elsewhere in the workspace. Add a row when you have checked the id at that provider, not before:
//! a wrong entry does not fail loudly, it routes to a 404 that looks like the client's fault.

use crate::ProviderId;

/// One upstream that can serve a catalog model, and the id to ask it for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub provider: ProviderId,
    /// The model id **as this provider spells it**. The gateway splices this into the request body's
    /// `model` field before forwarding, so it may differ from the row's canonical name and from
    /// every other candidate's.
    pub upstream_model: &'static str,
}

/// A canonical model name and the ordered list of upstreams that serve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRoute {
    /// The value a client puts in the routing header. Lowercase and restricted to
    /// `[a-z0-9._/-]`, which is what lets the gateway log it verbatim without sanitizing.
    pub model: &'static str,
    /// Preference order: `[0]` is primary, the rest are failover candidates. Non-empty, at most
    /// [`MAX_CANDIDATES`], no provider repeated.
    pub candidates: &'static [Candidate],
}

/// Upper bound on candidates per row, so the gateway can track which are usable in a single `u8`
/// bitmask with no per-request allocation.
pub const MAX_CANDIDATES: usize = 8;

/// Every routable model, **sorted by `model`** — [`for_model`] binary-searches it.
pub const MODEL_ROUTES: &[ModelRoute] = &[
    // Anthropic-wire. One candidate each: Anthropic is the only Anthropic-wire provider the gateway
    // routes to, so there is nowhere to fail over *to* without changing the client's API shape.
    ModelRoute {
        model: "claude-haiku-4-5",
        candidates: &[Candidate {
            provider: ProviderId::Anthropic,
            upstream_model: "claude-haiku-4-5",
        }],
    },
    ModelRoute {
        model: "claude-opus-4-8",
        candidates: &[Candidate {
            provider: ProviderId::Anthropic,
            upstream_model: "claude-opus-4-8",
        }],
    },
    // OpenAI-wire, two candidates — the shape failover is actually for. Both ids are exercised
    // against the real providers by `smoke.rs`, and the two mounts differ (`/v1` vs `/api/v1`),
    // which is exactly what the model-routed path's mount-prefix handling exists for.
    ModelRoute {
        model: "gpt-4o-mini",
        candidates: &[
            Candidate {
                provider: ProviderId::OpenAi,
                upstream_model: "gpt-4o-mini",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                upstream_model: "openai/gpt-4o-mini",
            },
        ],
    },
];

/// The catalog row for a model name, or `None` if we do not serve it.
///
/// [`MODEL_ROUTES`] is sorted (asserted), so the common path is a binary search over `&'static str`
/// — no allocation, and the result is `&'static` so the caller stores a thin pointer. The
/// case-insensitive linear fallback covers a client that upcased the header; it never runs for a
/// well-formed request.
pub fn for_model(name: &str) -> Option<&'static ModelRoute> {
    match MODEL_ROUTES.binary_search_by(|r| r.model.cmp(name)) {
        Ok(i) => MODEL_ROUTES.get(i),
        Err(_) => MODEL_ROUTES
            .iter()
            .find(|r| r.model.eq_ignore_ascii_case(name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Provider-table lookups the invariants check the catalog against; test-only, since the catalog
    // itself stores ids and never resolves them.
    use crate::{WireFormat, by_id, gateway_providers};

    /// The binary search in `for_model` is only correct on a sorted table, and duplicate names would
    /// make which row wins depend on where the search landed.
    #[test]
    fn model_routes_are_sorted_and_unique() {
        for pair in MODEL_ROUTES.windows(2) {
            assert!(
                pair[0].model < pair[1].model,
                "MODEL_ROUTES must be sorted by `model` and free of duplicates, but {:?} \
                 is not strictly before {:?}",
                pair[0].model,
                pair[1].model,
            );
        }
    }

    /// The gateway logs the matched route's name into the `ai.usage` billing row without running it
    /// through `sanitize_model`. That is only sound because the name is *ours*, from this table, and
    /// restricted to a charset that cannot break out of a JSON string or inject a log line.
    #[test]
    fn model_names_are_lowercase_and_log_safe() {
        for route in MODEL_ROUTES {
            assert!(!route.model.is_empty(), "a route name must not be empty");
            for b in route.model.bytes() {
                assert!(
                    b.is_ascii_lowercase()
                        || b.is_ascii_digit()
                        || matches!(b, b'.' | b'_' | b'/' | b'-'),
                    "route name {:?} contains {:?}, outside the log-safe [a-z0-9._/-] set",
                    route.model,
                    b as char,
                );
            }
        }
    }

    #[test]
    fn every_route_has_between_one_and_max_candidates() {
        for route in MODEL_ROUTES {
            let n = route.candidates.len();
            assert!(
                (1..=MAX_CANDIDATES).contains(&n),
                "route {:?} has {n} candidates; must be 1..={MAX_CANDIDATES} \
                 (the gateway tracks usable candidates in a u8 bitmask)",
                route.model,
            );
        }
    }

    /// A candidate the gateway has no registry entry for is dead on arrival — it would be filtered
    /// out at request time and silently reduce the row's failover depth.
    #[test]
    fn candidates_are_gateway_routable() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                assert!(
                    gateway_providers().any(|p| p.id == c.provider),
                    "route {:?} names {:?}, which is not a gateway-routable provider",
                    route.model,
                    c.provider,
                );
            }
        }
    }

    /// A provider listed twice would burn two connect attempts on the same dead upstream.
    #[test]
    fn candidates_within_a_row_are_distinct() {
        for route in MODEL_ROUTES {
            for (i, c) in route.candidates.iter().enumerate() {
                assert!(
                    !route.candidates[..i]
                        .iter()
                        .any(|p| p.provider == c.provider),
                    "route {:?} lists {:?} more than once",
                    route.model,
                    c.provider,
                );
            }
        }
    }

    /// Load-bearing. The gateway rewrites the body's `model` id per attempt but does **not**
    /// translate between API shapes, so a mixed-wire row would forward an Anthropic Messages body to
    /// a Chat Completions endpoint — and, worse, parse the reply with the wrong dialect's usage
    /// extractor, which yields a zero-token billing row rather than an error.
    #[test]
    fn candidates_agree_on_wire() {
        for route in MODEL_ROUTES {
            assert!(
                shared_wire(route.candidates).is_some(),
                "route {:?} mixes wire formats across its candidates: {:?}",
                route.model,
                route
                    .candidates
                    .iter()
                    .map(|c| (c.provider, by_id(c.provider).wire))
                    .collect::<Vec<_>>(),
            );
        }
    }

    /// The one wire format every candidate shares, or `None` if they disagree.
    fn shared_wire(candidates: &[Candidate]) -> Option<WireFormat> {
        let first = by_id(candidates.first()?.provider).wire;
        candidates
            .iter()
            .all(|c| by_id(c.provider).wire == first)
            .then_some(first)
    }

    /// `candidates_agree_on_wire` is only meaningful if it can fail. The seed table happens to
    /// contain no mixed row, so prove the check itself rejects one — Anthropic is Anthropic-wire and
    /// OpenRouter is OpenAI-wire, which is the exact pairing someone reaches for when adding Claude
    /// failover without realizing it changes the client's API shape.
    #[test]
    fn the_wire_check_rejects_a_mixed_row() {
        let mixed = [
            Candidate {
                provider: ProviderId::Anthropic,
                upstream_model: "claude-opus-4-8",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                upstream_model: "anthropic/claude-opus-4-8",
            },
        ];
        assert_eq!(
            shared_wire(&mixed),
            None,
            "an Anthropic + OpenRouter row must be rejected as mixed-wire",
        );
        assert_eq!(
            shared_wire(&mixed[..1]),
            Some(WireFormat::Anthropic),
            "a single-candidate row trivially shares its own wire",
        );
    }

    /// If an id is one a provider serves *natively* by shape, the candidate holding it had better be
    /// that provider. Catches `claude-opus-4-8` listed under Groq. Vendor-slug ids
    /// (`openai/gpt-4o-mini`) resolve to no native provider and are correctly skipped.
    #[test]
    fn native_ids_are_not_misrouted() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                if let Some(native) = crate::for_model_id(c.upstream_model) {
                    assert_eq!(
                        native.id, c.provider,
                        "route {:?} asks {:?} for {:?}, but that id is natively {:?}'s",
                        route.model, c.provider, c.upstream_model, native.id,
                    );
                }
            }
        }
    }

    /// The catalog is routing data. Capability facts belong in `agent_core::models`, and the way
    /// this file would grow them is by someone adding a field — so pin the field set.
    #[test]
    fn a_candidate_carries_routing_facts_only() {
        let c = Candidate {
            provider: ProviderId::OpenAi,
            upstream_model: "gpt-4o-mini",
        };
        // Destructured exhaustively: adding a field fails to compile here, which is the prompt to
        // ask whether it is really routing knowledge.
        let Candidate {
            provider: _,
            upstream_model: _,
        } = c;
    }

    #[test]
    fn for_model_finds_every_row() {
        for route in MODEL_ROUTES {
            assert_eq!(
                for_model(route.model),
                Some(route),
                "{:?} must resolve to its own row",
                route.model,
            );
        }
    }

    #[test]
    fn for_model_is_case_insensitive() {
        assert_eq!(
            for_model("GPT-4O-MINI").map(|r| r.model),
            Some("gpt-4o-mini"),
        );
    }

    #[test]
    fn for_model_is_none_for_unknown_ids() {
        for unknown in ["", "gpt-4o-min", "gpt-4o-mini-x", "claude", "nonesuch"] {
            assert_eq!(
                for_model(unknown),
                None,
                "{unknown:?} must not resolve — a near-miss is not a match",
            );
        }
    }

    /// The failover row is the one every gateway test leans on; assert its exact shape so a careless
    /// edit to the seed data breaks here rather than in an opaque proxy test.
    #[test]
    fn the_openai_wire_row_offers_a_real_second_candidate() {
        let want = [
            Candidate {
                provider: ProviderId::OpenAi,
                upstream_model: "gpt-4o-mini",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                upstream_model: "openai/gpt-4o-mini",
            },
        ];
        assert_eq!(
            for_model("gpt-4o-mini").map(|r| r.candidates),
            Some(&want[..]),
            "the seed failover row must keep its exact shape and order",
        );
        // Same wire (so no translation is implied) but genuinely different mounts — this is the row
        // that exercises the model-routed path's mount-prefix rewrite.
        assert_eq!(by_id(ProviderId::OpenAi).wire, WireFormat::OpenAi);
        assert_eq!(by_id(ProviderId::OpenRouter).wire, WireFormat::OpenAi);
        assert_eq!(by_id(ProviderId::OpenAi).base_path(), "/v1");
        assert_eq!(by_id(ProviderId::OpenRouter).base_path(), "/api/v1");
    }
}
