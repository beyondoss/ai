//! Property tests for `route`'s path tables: random paths never panic, and paths built from the
//! documented grammar classify exactly as the table says.
//!
//! The documented table (`implied_endpoint`, `SubResource::of_path`, `catalog_wire_action`): an
//! optional `/auto` prefix, then `/v1` (optional only under `/auto`), then one of the endpoints or
//! sub-resources, then any number of trailing slashes. Anything else names no endpoint.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::route::{
    self, Endpoint, SubResource, WireAction, catalog_wire_action, implied_endpoint,
};
use props_support::*;
use proptest::prelude::*;

const ENDPOINTS: &[(&str, Endpoint)] = &[
    ("/chat/completions", Endpoint::ChatCompletions),
    ("/messages", Endpoint::Messages),
    ("/responses", Endpoint::Responses),
    ("/embeddings", Endpoint::Embeddings),
];

const SUBS: &[(&str, SubResource)] = &[
    ("/messages/count_tokens", SubResource::CountTokens),
    ("/responses/input_tokens", SubResource::InputTokens),
    ("/responses/compact", SubResource::Compact),
];

const ROWS: [Endpoint; 4] = [
    Endpoint::ChatCompletions,
    Endpoint::Messages,
    Endpoint::Responses,
    Endpoint::Embeddings,
];

/// Path segments real and almost-real paths are made of.
fn segment() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => prop::sample::select(vec![
            "", "v1", "auto", "chat", "completions", "messages", "responses", "embeddings",
            "count_tokens", "input_tokens", "compact", "V1", "v1beta", "models", "openai",
            "..", ".", "%2F", "chat%2Fcompletions", "messages ", "responses?x=1",
        ]).prop_map(str::to_owned),
        1 => text(3),
    ]
}

#[test]
fn prop_any_path_classifies_without_panicking() {
    check(
        "prop_any_path_classifies_without_panicking",
        (prop::collection::vec(segment(), 0..6), any::<bool>()),
        |(segs, trailing)| {
            let mut path = String::new();
            for s in &segs {
                path.push('/');
                path.push_str(s);
            }
            if trailing {
                path.push('/');
            }
            let e = implied_endpoint(&path);
            let sub = SubResource::of_path(&path);
            let _ = SubResource::of_forward_path(&path);
            let _ = route::is_default_prefix(&path);
            let _ = route::is_responses_path(&path);
            let _ = Endpoint::of_upstream_path(&path);
            // A path names at most one of an endpoint and a sub-resource.
            prop_assert!(e.is_none() || sub.is_none(), "{path:?}: {e:?} and {sub:?}");
            for row in ROWS {
                let action = catalog_wire_action(&path, row);
                match e {
                    Some(c) if c == row => prop_assert_eq!(action, WireAction::Relay),
                    Some(c) if c != Endpoint::Embeddings && row != Endpoint::Embeddings => {
                        prop_assert_eq!(action, WireAction::Translate { client: c })
                    }
                    Some(_) => prop_assert_eq!(action, WireAction::Reject),
                    None => prop_assert!(
                        action == WireAction::Relay || action == WireAction::Reject,
                        "{path:?}: {action:?}"
                    ),
                }
            }
            Ok(())
        },
    );
}

/// Paths from the documented grammar classify as the table says; one mutated segment and they
/// name nothing.
#[test]
fn prop_documented_paths_classify_as_the_table_says() {
    check(
        "prop_documented_paths_classify_as_the_table_says",
        (
            any::<bool>(),
            any::<bool>(),
            0usize..(ENDPOINTS.len() + SUBS.len()),
            0usize..4,
        ),
        |(auto, v1, which, slashes)| {
            let v1 = v1 || !auto;
            let mut path = String::new();
            if auto {
                path.push_str("/auto");
            }
            if v1 {
                path.push_str("/v1");
            }
            let tail = "/".repeat(slashes);
            if which < ENDPOINTS.len() {
                let (p, e) = ENDPOINTS[which];
                let path = format!("{path}{p}{tail}");
                prop_assert_eq!(implied_endpoint(&path), Some(e), "{}", path);
                prop_assert_eq!(SubResource::of_path(&path), None);
                prop_assert_eq!(route::is_responses_path(&path), e == Endpoint::Responses);
                prop_assert_eq!(catalog_wire_action(&path, e), WireAction::Relay);
                // Upper case is not the same path.
                prop_assert_eq!(implied_endpoint(&path.to_uppercase()), None, "{}", path);
            } else {
                let (p, s) = SUBS[which - ENDPOINTS.len()];
                let path = format!("{path}{p}{tail}");
                prop_assert_eq!(SubResource::of_path(&path), Some(s), "{}", path);
                prop_assert_eq!(implied_endpoint(&path), None);
                // A forwarded path keeps the provider's own spelling: one trailing slash is the
                // same resource, `//` is a path the provider 404s (so there is nothing to bill).
                if slashes <= 1 {
                    prop_assert_eq!(
                        SubResource::of_forward_path(&format!("{path}?beta=true")),
                        Some(s)
                    );
                }
                for row in ROWS {
                    prop_assert_eq!(
                        catalog_wire_action(&path, row),
                        WireAction::Reject,
                        "{}",
                        path
                    );
                }
            }
            Ok(())
        },
    );
}
