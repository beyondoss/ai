//! Catalog-walk order for a managed default walk: the computed session pin.
//!
//! A managed `/auto` or `/v1` request walks one catalog row — no provider the row does not already
//! name. What this module decides is the **order** of that walk when the caller did not fix it
//! (`x-beyond-order` / `x-beyond-split`). It is the only ordering function: there is no latency
//! ranking (removed after D253 left it ordering only walks with no caller identity, which a managed
//! request never is).
//!
//! A session pin keeps one caller on one provider. Provider prompt caches are per provider, so a
//! walk that moves an agent loop from Anthropic to Bedrock and back re-reads the whole prefix at
//! full input price plus a cache write (1.25× vs 0.1× on Claude) on every move.
//!
//! The pin key is the verified identity (`tenant_id`, `vpc_id`, `key_id`) plus the catalog row:
//! one virtual key is one app, and an app's sessions share system prompt and tool prefixes, so
//! keeping the whole key on one provider is what the provider cache wants.
//!
//! **The pin is computed every turn, never remembered.** A caller's walk is [`preferred`]: the
//! row's leading run of equally-preferred first-party hosts (the vendor, and clouds reselling it
//! under its own ids: Anthropic + Bedrock) ordered by rendezvous hash of the pin key and provider,
//! then the rest in catalog order. Candidates that cannot be used on this pod (no pool key, every
//! key cooling, cannot serve the body) move to the back. No latency, no probe, no failure history:
//! nothing a pod observes moves the pin, so every replica computes the same walk and there is no
//! per-pod table to disagree (D253: a remembered per-pod pin let pod A keep a key on its failover
//! after one transient 5xx while pod B served the primary, and alternating turns re-bought the
//! prefix every time). An aggregator is never hashed in; it serves only when no first-party host
//! before it can.
//!
//! A failure fails over for that request only: the walk moves on in-gateway, and the next turn
//! starts from the preferred host again. An open breaker is skipped by `upstream_peer`, on that
//! pod, while it is open, and the next candidate in the computed walk is the same on every pod.
//! That is the one place replicas can disagree: during a real outage one pod's breaker can be open
//! while another's is closed, and a session alternating between them then hops between the
//! preferred host (where it still answers) and its failover until the breakers agree. Rendezvous
//! hashing moves only the keys hashed to the sick host.

use crate::control::Walk;
use providers::{MAX_CANDIDATES, ModelRoute, catalog::MODEL_ROUTES};

/// The pin identity for a verified managed caller. A virtual key is deterministic per
/// `(tenant, app)`, so `(tenant_id, vpc_id, key_id)` is one app's key. Not keyed: tenant ids are
/// verified before this runs, and the hash only orders providers the caller's own row names.
pub fn affinity(tenant_id: u64, vpc_id: u64, key_id: Option<u64>) -> u64 {
    let mut h = mix(tenant_id ^ 0x243F_6A88_85A3_08D3);
    h = mix(h ^ vpc_id);
    mix(h ^ key_id.map_or(0, |k| k ^ 0x1319_8A2E_0370_7344))
}

/// Order `walk` for the caller whose [`affinity`] is `affinity`: its computed session pin
/// ([`preferred`]). `dispatchable` has bit `i` set when `route.candidates[i]` can be used on this
/// pod; the others go to the back. `None` (leave the walk alone) when `route` is not a catalog row
/// or `walk` is empty.
pub fn order(walk: Walk, route: &ModelRoute, affinity: u64, dispatchable: u8) -> Option<Walk> {
    if walk.len == 0 {
        return None;
    }
    // Name search, not the row's address. `MODEL_ROUTES` is a `const` slice, so each use site can
    // hold its own copy — pointer arithmetic against `as_ptr()` does not land on the `&'static`
    // row `for_model` returned.
    let row_i = MODEL_ROUTES
        .binary_search_by(|r| r.model.cmp(route.model))
        .ok()?;
    let usable = |orig: u8| dispatchable & (1 << orig) != 0;
    Some(preferred(walk, route, pin_key(affinity, row_i), usable))
}

/// `splitmix64`'s finalizer: every input bit reaches every output bit.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn pin_key(affinity: u64, row_i: usize) -> u64 {
    mix(affinity ^ (row_i as u64).wrapping_mul(0xA076_1D64_78BD_642F))
}

/// A first-party host for its row: the vendor's own API, or a cloud reselling the model under the
/// vendor's ids (Bedrock). An aggregator claims no id prefix (`ProviderSpec::model_id_match`
/// explains why), and it is the costlier failover a pin must not be hashed onto while a
/// first-party host can serve.
fn first_party(route: &ModelRoute, orig: u8) -> bool {
    route
        .candidates
        .get(usize::from(orig))
        .is_some_and(|c| !providers::by_id(c.provider).model_id_match.is_empty())
}

/// FNV-1a of the provider's name: its identity in the rendezvous hash. The name, not the
/// `ProviderId` discriminant, so two builds that order the enum differently (a rolling deploy)
/// still agree.
fn provider_hash(name: &str) -> u64 {
    name.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The computed session pin: the walk's leading run of [`first_party`] candidates ordered by
/// rendezvous hash of (`key`, provider), then the rest in the walk's (catalog) order; candidates
/// `usable` rejects go behind the others, keeping that order. A pure function of the key, the row
/// and the usable set, so every replica computes the same walk. Hashing the provider, not the
/// slot, keeps a key's choice when an `only` filter or an unusable host removes another candidate: only
/// the keys hashed to the removed one move.
fn preferred(walk: Walk, route: &ModelRoute, key: u64, usable: impl Fn(u8) -> bool) -> Walk {
    let n = usize::from(walk.len);
    let lead = walk.indices[..n]
        .iter()
        .take_while(|&&orig| first_party(route, orig))
        .count();
    // (sick, past the lead, inverted hash, slot, orig): usable first, the lead highest hash first.
    let mut keyed = [(false, false, 0u64, 0u8, 0u8); MAX_CANDIDATES];
    for (slot, (k, &orig)) in keyed.iter_mut().zip(&walk.indices[..n]).enumerate() {
        let rest = slot >= lead;
        let hash = match route.candidates.get(usize::from(orig)) {
            Some(c) if !rest => !mix(key ^ provider_hash(providers::by_id(c.provider).name)),
            _ => 0,
        };
        *k = (!usable(orig), rest, hash, slot as u8, orig);
    }
    keyed[..n].sort_unstable();
    let mut out = Walk {
        indices: [0u8; MAX_CANDIDATES],
        len: walk.len,
    };
    for (dst, k) in out.indices.iter_mut().zip(&keyed[..n]) {
        *dst = k.4;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Walk;
    use providers::by_id;

    /// Every candidate keyed.
    const ALL: u8 = u8::MAX;
    const APP: u64 = 0xA11CE;

    fn opus() -> &'static ModelRoute {
        providers::for_model("claude-opus-4-8").expect("catalog row")
    }

    fn mini() -> &'static ModelRoute {
        providers::for_model("gpt-4o-mini").expect("catalog row")
    }

    fn aff() -> u64 {
        affinity(42, 7, Some(APP))
    }

    fn names(walk: Walk, row: &ModelRoute) -> Vec<&'static str> {
        (0..walk.len)
            .map(|i| {
                let orig = walk.indices[i as usize];
                by_id(row.candidates[orig as usize].provider).name
            })
            .collect()
    }

    fn ordered(row: &ModelRoute, a: u64, dispatchable: u8) -> Vec<&'static str> {
        let walk = order(Walk::identity(row.candidates.len()), row, a, dispatchable);
        names(walk.expect("a catalog row"), row)
    }

    /// The walk any pod computes for `a` on `row`, every candidate usable.
    fn computed(row: &ModelRoute, a: u64) -> Vec<&'static str> {
        ordered(row, a, ALL)
    }

    fn index_of(row: &ModelRoute, name: &str) -> u8 {
        row.candidates
            .iter()
            .position(|c| by_id(c.provider).name == name)
            .expect("candidate") as u8
    }

    #[test]
    fn a_route_outside_the_catalog_slice_is_left_alone() {
        let route = ModelRoute {
            model: "not-a-catalog-row",
            wire: providers::WireFormat::OpenAi,
            candidates: &[],
            responses: &[],
            // Not a catalog row. The price and card are unused.
            price: providers::ListPrice {
                input: "0",
                output: "0",
                cache_read: "0",
                cache_write: "0",
            },
            card: providers::catalog::MODEL_ROUTES[0].card,
        };
        assert!(order(Walk::identity(3), &route, aff(), ALL).is_none());
    }

    /// D253: the walk is a pure function of the caller, the row and the usable set, so every pod
    /// computes the same one. Across callers the pooled first-party hosts share the load, and the
    /// aggregator never leads while one of them is usable.
    /// claim: R4
    /// defect: D253
    #[test]
    fn every_pod_computes_the_same_pin() {
        let row = opus();
        let mut leads = [0usize; 2];
        for app in 0..64 {
            let k = affinity(42, 7, Some(app));
            let walk = computed(row, k);
            assert_eq!(walk, computed(row, k), "app {app}");
            assert_eq!(walk[2], "openrouter", "the aggregator stays last");
            leads[usize::from(walk[0] == "bedrock")] += 1;
        }
        assert!(
            leads[0] > 8 && leads[1] > 8,
            "rendezvous spreads the lead: {leads:?}"
        );
    }

    /// A row with a single first-party host keeps catalog order: there is nothing to hash, and a
    /// failure (which this module never sees) moves no later turn.
    /// claim: R4
    /// defect: D253
    #[test]
    fn a_single_first_party_host_keeps_catalog_order() {
        assert_eq!(computed(mini(), aff()), ["openai", "openrouter"]);
    }

    /// A candidate with no pool key here (or cooling, or unable to serve the body) goes behind
    /// every one that can, keeping the others' order: the dispatchable mask D119 introduced.
    /// claim: R4
    /// defect: D119
    #[test]
    fn a_pin_skips_a_candidate_that_cannot_be_used_here() {
        let row = opus();
        let want = computed(row, aff());
        let lead = index_of(row, want[0]);
        let walk = ordered(row, aff(), ALL & !(1 << lead));
        assert_eq!(walk, [want[1], want[2], want[0]]);
    }

    /// The pin is a function of the caller's own verified identity: another key, tenant or model
    /// hashes on its own and cannot be steered by this one.
    /// claim: SEC-15
    #[test]
    fn pins_are_per_caller_and_per_model() {
        let row = opus();
        let k = |t, app| pin_key(affinity(t, 7, Some(app)), 0);
        assert_ne!(k(42, APP), k(42, APP + 1), "another key");
        assert_ne!(k(42, APP), k(43, APP), "another tenant");
        assert_ne!(pin_key(aff(), 0), pin_key(aff(), 1), "another model");
        // Over many tenants sharing one key id, the lead follows each tenant's own hash.
        let leads: std::collections::BTreeSet<_> = (0..32)
            .map(|t| computed(row, affinity(t, 7, Some(APP)))[0])
            .collect();
        assert_eq!(leads.len(), 2, "{leads:?}");
    }

    /// `x-beyond-only` filters first; the pin is computed over what is left, and the removed
    /// candidate moves no other (rendezvous).
    #[test]
    fn a_filtered_walk_keeps_the_pin_among_what_is_left() {
        let row = opus();
        let want = computed(row, aff());
        let walk = Walk {
            indices: [0, 1, 0, 0, 0, 0, 0, 0],
            len: 2,
        };
        let out = order(walk, row, aff(), ALL).expect("a catalog row");
        assert_eq!(names(out, row), want[..2]);
        let walk = Walk {
            indices: [2, 0, 0, 0, 0, 0, 0, 0],
            len: 1,
        };
        let out = order(walk, row, aff(), ALL).expect("a catalog row");
        assert_eq!(names(out, row), ["openrouter"]);
    }

    /// Only the leading run of first-party hosts is hashed: a row whose primary is an aggregator
    /// keeps the catalog's order.
    #[test]
    fn an_aggregator_primary_keeps_catalog_order() {
        const CANDIDATES: &[providers::Candidate] = &[
            providers::Candidate {
                provider: providers::ProviderId::OpenRouter,
                upstream_model: "x",
                path: "/api/v1/chat/completions",
            },
            providers::Candidate {
                provider: providers::ProviderId::OpenAi,
                upstream_model: "x",
                path: "/v1/chat/completions",
            },
            providers::Candidate {
                provider: providers::ProviderId::Anthropic,
                upstream_model: "x",
                path: "/v1/messages",
            },
        ];
        let route = ModelRoute {
            candidates: CANDIDATES,
            ..*opus()
        };
        for app in 0..16 {
            let k = pin_key(affinity(42, 7, Some(app)), 0);
            let out = preferred(Walk::identity(3), &route, k, |_| true);
            assert_eq!(out.indices[..3], [0, 1, 2]);
        }
    }
}
