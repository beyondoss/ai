//! The model catalog — canonical model name → the ordered upstreams that serve it.
//!
//! This is the inverse of [`crate::for_model_id`], and the general form of it. `for_model_id` answers
//! "which provider serves this id *shape*, natively, when nobody said otherwise" and deliberately
//! returns nothing for an aggregator, because `moonshotai/kimi-k2.6` is equally a Fireworks,
//! Together and OpenRouter id and guessing between them is the mis-route this crate exists to
//! prevent. The catalog is where that guess becomes a *decision*: a named model, the wire its
//! clients speak, and the ordered upstreams we are willing to serve it from.
//!
//! It carries routing facts only — provider, the id that provider spells it with, and the path to
//! send it to. Model *capability* facts (context window, thinking shape) stay in
//! `agent_core::models`; a test keeps this file from growing them.
//!
//! # Wire format belongs to the row, not the provider
//!
//! [`ProviderSpec::wire`](crate::ProviderSpec::wire) is a single value per provider, and that is an
//! approximation. OpenRouter is the clearest case: it serves the **OpenAI** wire at
//! `/api/v1/chat/completions` *and* the **Anthropic** wire at `/api/v1/messages`, and both are real
//! — the Anthropic one returns `message_start`/`message_delta` SSE with `input_tokens`,
//! `cache_read_input_tokens` and `output_tokens_details.thinking_tokens`, which is exactly what the
//! gateway's Anthropic usage extractor reads. (Fireworks is the same story from the other side: see
//! `agent_core::dialect::is_fireworks_anthropic_wire_model`.)
//!
//! So a row declares its own [`ModelRoute::wire`] and each candidate carries the [`Candidate::path`]
//! that serves it. Deriving the wire from the provider would have been wrong in a specifically nasty
//! way: an Anthropic-wire response parsed by the OpenAI extractor trips the dialect-mismatch guard
//! and emits a **zero-token billing row**, not an error.
//!
//! Every candidate in a row must still agree on the wire, because the gateway rewrites model ids but
//! does **not** translate between API shapes.
//!
//! # Maintenance
//!
//! These rows are product data and they go stale — providers rename ids, deprecate models, and
//! change what they host. Every id and path below was verified against the live API before being
//! added, and `catalog_rows_are_servable` (in `crates/gateway/tests/smoke.rs`) re-verifies the whole
//! table against real providers whenever the keys are present. Add a row the same way: check it,
//! then add it. A wrong entry does not fail loudly — it routes to a 404 that looks like the client's
//! fault.

use crate::{ProviderId, WireFormat};

/// One upstream that can serve a catalog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub provider: ProviderId,
    /// The model id **as this provider spells it**. The gateway splices this into the request body's
    /// `model` field before forwarding, so it may differ from the row's canonical name and from
    /// every other candidate's — `claude-opus-4-8` at Anthropic is `anthropic/claude-opus-4.8` at
    /// OpenRouter, dots and all.
    pub upstream_model: &'static str,
    /// The full upstream path for this candidate on this row's wire.
    ///
    /// Absolute and complete, not a suffix to be composed: providers do not agree on where an
    /// endpoint lives, and the disagreement is not a simple prefix. Anthropic serves Messages at
    /// `/v1/messages` from a base URL carrying no path; OpenRouter serves the same wire at
    /// `/api/v1/messages`. There is no client-supplied suffix that is correct for both, so the
    /// catalog states each one outright.
    pub path: &'static str,
}

/// A canonical model name, the wire its clients speak, and the ordered upstreams that serve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRoute {
    /// The value a client puts in the routing header. Lowercase and restricted to `[a-z0-9._/-]`,
    /// which is what lets the gateway log it verbatim without sanitizing.
    pub model: &'static str,
    /// The API shape a client of this row sends, and the shape its responses come back in. Drives
    /// the gateway's usage extractor. Declared here rather than read off the provider — see the
    /// module docs.
    pub wire: WireFormat,
    /// Preference order: `[0]` is primary, the rest are failover candidates. Non-empty, at most
    /// [`MAX_CANDIDATES`], no provider repeated.
    pub candidates: &'static [Candidate],
}

/// Upper bound on candidates per row, so the gateway can track which are usable in a single `u8`
/// bitmask with no per-request allocation.
pub const MAX_CANDIDATES: usize = 8;

/// Every routable model, **sorted by `model`** — [`for_model`] binary-searches it.
///
/// Each id/path pair below returned `200` from the real provider when it was added.
pub const MODEL_ROUTES: &[ModelRoute] = &[
    // Claude on the Anthropic wire, with a real second source: OpenRouter's `/api/v1/messages` is a
    // genuine Messages endpoint, reached with a different key over a different network path.
    //
    // Know what this does and does not buy. OpenRouter chooses its own backend per request —
    // observed serving these ids from both Anthropic directly and Amazon Bedrock — so the second
    // candidate is *not* a guaranteed independent supply of the model. It reliably covers the
    // failures that are ours: our egress blocked, our Anthropic key throttled or suspended,
    // api.anthropic.com unreachable from us. It does not guarantee cover for Anthropic's own
    // serving being down, because OpenRouter may be forwarding there too.
    ModelRoute {
        model: "claude-haiku-4-5",
        wire: WireFormat::Anthropic,
        candidates: &[
            Candidate {
                provider: ProviderId::Anthropic,
                upstream_model: "claude-haiku-4-5",
                path: "/v1/messages",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                upstream_model: "anthropic/claude-haiku-4.5",
                path: "/api/v1/messages",
            },
        ],
    },
    ModelRoute {
        model: "claude-opus-4-8",
        wire: WireFormat::Anthropic,
        candidates: &[
            Candidate {
                provider: ProviderId::Anthropic,
                upstream_model: "claude-opus-4-8",
                path: "/v1/messages",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                upstream_model: "anthropic/claude-opus-4.8",
                path: "/api/v1/messages",
            },
        ],
    },
    // The same shape on the OpenAI wire, where the two mounts differ as well (`/v1` vs `/api/v1`).
    ModelRoute {
        model: "gpt-4o-mini",
        wire: WireFormat::OpenAi,
        candidates: &[
            Candidate {
                provider: ProviderId::OpenAi,
                upstream_model: "gpt-4o-mini",
                path: "/v1/chat/completions",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                upstream_model: "openai/gpt-4o-mini",
                path: "/api/v1/chat/completions",
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
    use crate::{by_id, gateway_providers};

    /// The wire a path serves, inferred from its endpoint — `…/messages` is Anthropic, everything
    /// else is OpenAI-shaped. An independent read of the same fact the row declares, which is what
    /// lets the two be cross-checked.
    fn wire_of_path(path: &str) -> WireFormat {
        if path.ends_with("/messages") {
            WireFormat::Anthropic
        } else {
            WireFormat::OpenAi
        }
    }

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
    /// translate between API shapes, so a row whose candidates disagree on the wire would forward an
    /// Anthropic Messages body to a Chat Completions endpoint — and, worse, parse the reply with the
    /// wrong dialect's usage extractor, which yields a zero-token billing row rather than an error.
    ///
    /// Checked against each candidate's **path**, not its provider: OpenRouter serves both wires, so
    /// the provider's own `wire` field cannot answer this.
    #[test]
    fn candidate_paths_match_the_rows_declared_wire() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                assert_eq!(
                    wire_of_path(c.path),
                    route.wire,
                    "route {:?} declares {:?} but candidate {:?} points at {:?}",
                    route.model,
                    route.wire,
                    c.provider,
                    c.path,
                );
            }
        }
    }

    /// Every candidate in a row must serve the **same endpoint**, not merely the same wire.
    ///
    /// `wire_of_path` only separates Messages from everything else, so `/v1/chat/completions` and
    /// `/v1/responses` both read as OpenAI and would pass the wire check while behaving differently:
    /// the gateway's `stream_options.include_usage` injection is a Chat Completions construct, and
    /// `is_streamable_path` — which decides whether to inject at all — is computed once from the
    /// *first* candidate's path. A row mixing the two would inject into a Responses request, or skip
    /// injection on a Chat Completions one and silently lose the usage chunk it meters from.
    #[test]
    fn candidates_within_a_row_share_one_endpoint() {
        for route in MODEL_ROUTES {
            let endpoint = |p: &str| p.rsplit_once("/v1").map_or(p, |(_, tail)| tail).to_string();
            let Some(first) = route.candidates.first() else {
                continue;
            };
            let want = endpoint(first.path);
            for c in route.candidates {
                assert_eq!(
                    endpoint(c.path),
                    want,
                    "route {:?}: {:?} serves {:?} but {:?} serves {:?} — same wire, different \
                     endpoint, which the injection path cannot straddle",
                    route.model,
                    first.provider,
                    first.path,
                    c.provider,
                    c.path,
                );
            }
        }
    }

    /// ...and prove that check can fail, since the seed table contains no violation to catch.
    #[test]
    fn the_endpoint_check_rejects_chat_completions_mixed_with_responses() {
        let endpoint = |p: &str| p.rsplit_once("/v1").map_or(p, |(_, tail)| tail).to_string();
        assert_eq!(
            endpoint("/v1/chat/completions"),
            endpoint("/api/v1/chat/completions")
        );
        assert_ne!(
            endpoint("/v1/chat/completions"),
            endpoint("/v1/responses"),
            "the two OpenAI-wire endpoints must be distinguishable",
        );
    }

    #[test]
    fn candidate_paths_are_absolute() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                assert!(
                    c.path.starts_with('/') && !c.path.contains("://"),
                    "route {:?} candidate {:?} path {:?} must be an absolute path, not a URL",
                    route.model,
                    c.provider,
                    c.path,
                );
            }
        }
    }

    /// A candidate's path must sit under its provider's own base URL, where the provider publishes
    /// one. Catches a path copied from the wrong row — the failure mode a 404 at request time.
    #[test]
    fn candidate_paths_sit_under_their_providers_base() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                let spec = by_id(c.provider);
                let Some(base) = spec.base_url else { continue };
                let mount = spec.base_path();
                assert!(
                    c.path.starts_with(mount),
                    "route {:?}: {} serves from {base} (mount {mount:?}), but the candidate path is \
                     {:?}",
                    route.model,
                    spec.name,
                    c.path,
                );
            }
        }
    }

    /// If an id is one a provider serves *natively* by shape, the candidate holding it had better be
    /// that provider. Catches `claude-opus-4-8` listed under Groq. Vendor-slug ids
    /// (`anthropic/claude-opus-4.8`) resolve to no native provider and are correctly skipped.
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
            path: "/v1/chat/completions",
        };
        // Destructured exhaustively: adding a field fails to compile here, which is the prompt to
        // ask whether it is really routing knowledge.
        let Candidate {
            provider: _,
            upstream_model: _,
            path: _,
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

    /// `wire_of_path` is what `candidate_paths_match_the_rows_declared_wire` leans on, so prove it
    /// discriminates rather than always answering the same thing.
    #[test]
    fn wire_of_path_discriminates_messages_from_chat_completions() {
        assert_eq!(wire_of_path("/v1/messages"), WireFormat::Anthropic);
        assert_eq!(wire_of_path("/api/v1/messages"), WireFormat::Anthropic);
        assert_eq!(wire_of_path("/v1/chat/completions"), WireFormat::OpenAi);
        assert_eq!(wire_of_path("/api/v1/chat/completions"), WireFormat::OpenAi);
        assert_eq!(wire_of_path("/v1/responses"), WireFormat::OpenAi);
    }

    /// Claude has a genuine second source, and it is *not* the provider's declared wire that makes
    /// it work. Pinned explicitly because this row is the whole point of the wire rework: OpenRouter
    /// is an OpenAI-wire provider by `ProviderSpec`, yet serves this row's Anthropic-wire traffic.
    #[test]
    fn claude_fails_over_to_openrouter_on_the_anthropic_wire() {
        let want = [
            Candidate {
                provider: ProviderId::Anthropic,
                upstream_model: "claude-opus-4-8",
                path: "/v1/messages",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                // OpenRouter spells Claude with a vendor prefix and dots, not dashes.
                upstream_model: "anthropic/claude-opus-4.8",
                path: "/api/v1/messages",
            },
        ];
        assert_eq!(
            for_model("claude-opus-4-8").map(|r| (r.wire, r.candidates)),
            Some((WireFormat::Anthropic, &want[..])),
            "Claude must have a real Anthropic-wire fallback",
        );
        // The point: the fallback provider's own wire disagrees with the row's, and that is fine
        // because the row and the path are what decide.
        assert_eq!(by_id(ProviderId::OpenRouter).wire, WireFormat::OpenAi);
    }
}
