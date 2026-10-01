//! Property tests for `signed_id`: arbitrary tenant and provider-id pairs round-trip, an id never
//! verifies for any other tenant, no single-byte change to one verifies, and a request body or
//! response stream of arbitrary ids rewrites exactly the ids it names.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::signed_id::{Relay, Signer};
use props_support::*;
use proptest::prelude::*;
use serde_json::{Value, json};

fn signer() -> Signer {
    Signer::new(&[(b'1', &[7u8; 32]), (b'z', &[3u8; 32])], b'z').unwrap()
}

/// Provider ids as OpenAI issues them (a prefix and lowercase hex, packed), and anything else a
/// provider could send (kept verbatim): other alphabets, odd lengths, no prefix, unicode.
fn provider_id() -> impl Strategy<Value = String> {
    let prefix = prop::sample::select(vec![
        "resp_", "msg_", "rs_", "fc_", "conv_", "ws_", "ctc_", "cmp_",
    ]);
    prop_oneof![
        4 => (prefix.clone(), prop::collection::vec(any::<u8>(), 1..33)).prop_map(|(p, b)| {
            let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
            format!("{p}{hex}")
        }),
        1 => (prefix, "[0-9A-Za-z_-]{1,60}").prop_map(|(p, s)| format!("{p}{s}")),
        1 => "[0-9A-Za-z_-]{1,40}",
        1 => any::<String>().prop_filter("non-empty", |s| !s.is_empty()),
    ]
}

/// claim: SEC-25
#[test]
fn prop_ids_round_trip_for_their_tenant_and_never_verify_for_another() {
    let s = signer();
    check(
        "prop_ids_round_trip_for_their_tenant_and_never_verify_for_another",
        (any::<u64>(), any::<u64>(), provider_id()),
        |(a, b, raw)| {
            let t = s.sign(a, &raw);
            prop_assert_eq!(s.verify(a, &t), Some(raw.clone()));
            if a != b {
                prop_assert_eq!(s.verify(b, &t), None);
            }
            // A provider id is never mistaken for a signed one.
            prop_assert_eq!(s.verify(a, &raw), None);
            // Packed ids stay within OpenAI's 64-character cap for its own id shape.
            if raw.len() <= 55
                && raw.split_once('_').is_some_and(|(_, h)| {
                    h.len() % 2 == 0
                        && h.bytes()
                            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                })
            {
                prop_assert!(t.len() <= 64, "{} is {} chars", t, t.len());
            }
            prop_assert!(t.starts_with(raw.split_inclusive('_').next().unwrap_or("")));
            Ok(())
        },
    );
}

/// claim: SEC-25
#[test]
fn prop_no_single_byte_change_to_a_signed_id_verifies() {
    let s = signer();
    check(
        "prop_no_single_byte_change_to_a_signed_id_verifies",
        (
            any::<u64>(),
            provider_id(),
            any::<prop::sample::Index>(),
            any::<u8>(),
        ),
        |(a, raw, at, byte)| {
            let t = s.sign(a, &raw);
            let mut b = t.clone().into_bytes();
            let i = at.index(b.len());
            prop_assume!(b[i] != byte);
            b[i] = byte;
            if let Ok(altered) = String::from_utf8(b) {
                prop_assert_eq!(s.verify(a, &altered), None, "{} verified", altered);
            }
            Ok(())
        },
    );
}

/// A request naming ids in every position strips each of the caller's own to the provider id and
/// changes nothing else; any one foreign id among them refuses the whole request.
/// claim: SEC-25
#[test]
fn prop_a_request_strips_exactly_its_own_ids() {
    let s = signer();
    check(
        "prop_a_request_strips_exactly_its_own_ids",
        (
            any::<u64>(),
            prop::collection::vec(provider_id(), 1..6),
            prop::option::of(any::<prop::sample::Index>()),
        ),
        |(a, raws, foreign)| {
            let body = |ids: &[String]| {
                json!({
                    "model": "gpt-4o",
                    "previous_response_id": ids[0],
                    "input": ids.iter().map(|id| json!({"type": "item_reference", "id": id})).collect::<Vec<_>>(),
                    "conversation": {"id": ids[ids.len() - 1]},
                })
            };
            let mut sent: Vec<String> = raws.iter().map(|r| s.sign(a, r)).collect();
            if let Some(i) = foreign {
                let i = i.index(sent.len());
                sent[i] = s.sign(a.wrapping_add(1), &raws[i]);
                let out = s.unsign_request(a, &serde_json::to_vec(&body(&sent)).unwrap(), false);
                prop_assert!(out.is_err());
            } else {
                let out = s
                    .unsign_request(a, &serde_json::to_vec(&body(&sent)).unwrap(), false)
                    .unwrap()
                    .unwrap();
                let got: Value = serde_json::from_slice(&out).unwrap();
                prop_assert_eq!(got, body(&raws));
            }
            Ok(())
        },
    );
}

/// A stream of events carrying arbitrary ids, split at arbitrary points, comes out with every id
/// signed the same way and every other byte unchanged.
/// claim: SEC-25
#[test]
fn prop_a_stream_signs_every_id_whatever_the_chunking() {
    let s = signer();
    check(
        "prop_a_stream_signs_every_id_whatever_the_chunking",
        (
            any::<u64>(),
            provider_id(),
            provider_id(),
            prop::collection::vec(atom(), 1..4),
            prop::collection::vec(1usize..64, 1..20),
        ),
        |(a, resp, item, deltas, cuts)| {
            let ev = |v: Value| format!("data: {v}\n\n");
            let stream = |r: &str, m: &str| {
                let mut out =
                    ev(json!({"type": "response.created", "response": {"id": r, "output": []}}));
                out.push_str(&ev(json!({"type": "response.output_item.added", "item": {"id": m, "type": "message"}})));
                for d in &deltas {
                    out.push_str(&ev(
                        json!({"type": "response.output_text.delta", "item_id": m, "delta": d}),
                    ));
                }
                out.push_str(&ev(json!({"type": "response.completed", "response": {"id": r, "output": [{"id": m}]}})));
                out
            };
            let input = stream(&resp, &item);
            let want = stream(&s.sign(a, &resp), &s.sign(a, &item));
            let mut relay = Relay::new(a);
            prop_assert!(relay.begin(200, true));
            let bytes = input.as_bytes();
            let (mut at, mut out) = (0usize, Vec::new());
            for c in cuts.iter().cycle() {
                let end = (at + c).min(bytes.len());
                let eos = end == bytes.len();
                out.extend(relay.feed(&s, &bytes[at..end], eos).unwrap().unwrap());
                at = end;
                if eos {
                    break;
                }
            }
            prop_assert_eq!(String::from_utf8(out).unwrap(), want);
            Ok(())
        },
    );
}
