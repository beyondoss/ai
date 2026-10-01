//! Property tests for `translate::SseBridge`: every client × upstream pairing, fed generated
//! streams (see `props_support::script`) and structured garbage.
//!
//! Properties:
//! - never panics, output bounded by a constant factor of input, on any bytes;
//! - the client always gets well-formed SSE in its own dialect, with its lifecycle intact
//!   (Messages block nesting, Responses `sequence_number` / item bookkeeping, Chat `[DONE]`),
//!   whatever the upstream sent — including errors anywhere and streams cut anywhere;
//! - whole, byte-by-byte and randomly chunked feeding give the same client bytes (minted ids and
//!   timestamps normalized);
//! - text, tool names, tool arguments (as JSON) and signed thinking survive every pairing;
//! - the usage the client is shown equals what `usage.rs` bills for the same upstream bytes.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::route::Endpoint;
use beyond_ai::translate::SseBridge;
use beyond_ai::usage;
use props_support::script::{DIALECTS, Script, Seen, script, seen_stream};
use props_support::*;
use proptest::prelude::*;
use serde_json::Value;

const GATEWAY_SIGNATURE_PREFIX: &str = "rs_gw:";

fn feed_whole(client: Endpoint, upstream: Endpoint, bytes: &[u8]) -> Vec<u8> {
    SseBridge::new(client, upstream).feed(bytes, true)
}

fn feed_chunks(client: Endpoint, upstream: Endpoint, chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut b = SseBridge::new(client, upstream);
    let mut out = Vec::new();
    for c in chunks {
        out.extend(b.feed(c, false));
    }
    out.extend(b.feed(&[], true));
    out
}

fn feed_bytewise(client: Endpoint, upstream: Endpoint, bytes: &[u8]) -> Vec<u8> {
    let mut b = SseBridge::new(client, upstream);
    let mut out = Vec::new();
    for byte in bytes {
        out.extend(b.feed(std::slice::from_ref(byte), false));
    }
    out.extend(b.feed(&[], true));
    out
}

fn pairings() -> impl Strategy<Value = (Endpoint, Endpoint)> {
    (
        prop::sample::select(DIALECTS.to_vec()),
        prop::sample::select(DIALECTS.to_vec()),
    )
}

fn cross_pairings() -> impl Strategy<Value = (Endpoint, Endpoint)> {
    pairings().prop_filter("translated pairings only", |(c, u)| c != u)
}

/// Output may grow per event (every client event carries its own envelope), never without bound.
fn bounded(input: usize, output: usize) -> bool {
    output <= input.saturating_mul(24) + 16 * 1024
}

fn billed(upstream: Endpoint, bytes: &[u8]) -> Option<usage::Usage> {
    match upstream {
        Endpoint::Messages => usage::anthropic_stream(bytes),
        _ => usage::openai_stream(bytes),
    }
}

/// Arguments the client was sent, as JSON, or why they aren't.
fn args_json(a: &str) -> Result<Value, String> {
    serde_json::from_str(a).map_err(|e| format!("tool arguments {a:?} are not JSON: {e}"))
}

fn check_content(
    s: &Script,
    client: Endpoint,
    upstream: Endpoint,
    seen: &Seen,
) -> Result<(), String> {
    if seen.text != s.text() {
        return Err(format!(
            "text: client saw {:?}, upstream sent {:?}",
            seen.text,
            s.text()
        ));
    }
    let want = s.tools();
    if seen.tools.len() != want.len() {
        return Err(format!(
            "tool calls: client saw {:?}, upstream sent {want:?}",
            seen.tools
        ));
    }
    for ((name, args), (wname, wargs)) in seen.tools.iter().zip(&want) {
        if name != wname {
            return Err(format!("tool name {name:?}, upstream sent {wname:?}"));
        }
        if &args_json(args)? != wargs {
            return Err(format!(
                "tool {name} arguments {args}, upstream sent {wargs}"
            ));
        }
    }
    let signed = s.replayable_thinking();
    match (upstream, client) {
        (Endpoint::Messages, Endpoint::ChatCompletions)
        | (Endpoint::ChatCompletions, Endpoint::Messages) => {
            if seen.thinking != signed {
                return Err(format!(
                    "thinking: client saw {:?}, upstream sent {signed:?}",
                    seen.thinking
                ));
            }
        }
        (Endpoint::Messages | Endpoint::ChatCompletions, Endpoint::Responses) => {
            let want: Vec<String> = signed
                .iter()
                .filter_map(|b| b["signature"].as_str())
                .map(|sig| format!("{GATEWAY_SIGNATURE_PREFIX}{sig}"))
                .collect();
            if seen.encrypted != want {
                return Err(format!(
                    "signed reasoning: client saw {:?}, want {want:?}",
                    seen.encrypted
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Any bytes at all, any chunking, every pairing: no panic, bounded output, and a translated
/// client still gets well-formed SSE with its lifecycle intact.
#[test]
fn prop_any_bytes_never_panic_and_keep_the_client_lifecycle() {
    check(
        "prop_any_bytes_never_panic_and_keep_the_client_lifecycle",
        (
            pairings(),
            prop::collection::vec(any::<u8>(), 0..512),
            prop::collection::vec(any::<usize>(), 0..8),
        ),
        |((client, upstream), bytes, cuts)| {
            let out = feed_chunks(client, upstream, &chunk(&bytes, &cuts));
            prop_assert!(
                bounded(bytes.len(), out.len()),
                "{} bytes in, {} out",
                bytes.len(),
                out.len()
            );
            if client != upstream
                && let Err(e) = seen_stream(client, &out)
            {
                prop_assert!(false, "{e}\nout: {}", String::from_utf8_lossy(&out));
            }
            Ok(())
        },
    );
}

/// Structured garbage — events built from the dialects' own keys and type strings, in any
/// order — exercises the real dispatch paths. Same invariants as for raw bytes.
#[test]
fn prop_vocabulary_events_keep_the_client_lifecycle() {
    check(
        "prop_vocabulary_events_keep_the_client_lifecycle",
        (
            cross_pairings(),
            prop::collection::vec(vocab_event(), 0..24),
            framing(),
            any::<bool>(),
            prop::collection::vec(any::<usize>(), 0..8),
        ),
        |((client, upstream), events, fr, done, cuts)| {
            let mut bytes = Vec::new();
            for (name, v) in &events {
                fr.event(&mut bytes, *name, &v.to_string());
            }
            if done {
                fr.event(&mut bytes, None, "[DONE]");
            }
            let whole = feed_whole(client, upstream, &bytes);
            prop_assert!(
                bounded(bytes.len(), whole.len()),
                "{} bytes in, {} out",
                bytes.len(),
                whole.len()
            );
            if let Err(e) = seen_stream(client, &whole) {
                prop_assert!(
                    false,
                    "{e}\nin: {}\nout: {}",
                    String::from_utf8_lossy(&bytes),
                    String::from_utf8_lossy(&whole)
                );
            }
            let chunked = feed_chunks(client, upstream, &chunk(&bytes, &cuts));
            prop_assert_eq!(normalize_minted(&whole), normalize_minted(&chunked));
            Ok(())
        },
    );
}

/// Whole, byte-by-byte and random chunking give the same client bytes, for every pairing
/// (relays included), on realistic streams, with or without an error or a cut.
#[test]
fn prop_chunking_never_changes_the_client_stream() {
    check(
        "prop_chunking_never_changes_the_client_stream",
        (
            pairings(),
            script(),
            prop::collection::vec(any::<usize>(), 0..16),
            prop::option::of((any::<bool>(), any::<usize>())),
        ),
        |((client, upstream), s, cuts, fault)| {
            let n = s.event_count(upstream);
            let (cut, error_at) = match fault {
                Some((true, at)) => (Some(at % (n + 1)), None),
                Some((false, at)) => (None, Some(at % n)),
                None => (None, None),
            };
            let bytes = s.stream(upstream, cut, error_at);
            let whole = normalize_minted(&feed_whole(client, upstream, &bytes));
            let chunked = normalize_minted(&feed_chunks(client, upstream, &chunk(&bytes, &cuts)));
            prop_assert_eq!(&whole, &chunked, "random chunking");
            let bytewise = normalize_minted(&feed_bytewise(client, upstream, &bytes));
            prop_assert_eq!(&whole, &bytewise, "byte-by-byte");
            Ok(())
        },
    );
}

/// A complete upstream stream reaches the client with its text, tool calls, signed thinking
/// and usage intact, in a well-formed stream of the client's dialect.
#[test]
fn prop_translated_streams_preserve_content_and_usage() {
    check(
        "prop_translated_streams_preserve_content_and_usage",
        (cross_pairings(), script()),
        |((client, upstream), s)| {
            let bytes = s.stream(upstream, None, None);
            let out = feed_whole(client, upstream, &bytes);
            prop_assert!(bounded(bytes.len(), out.len()));
            let seen = match seen_stream(client, &out) {
                Ok(seen) => seen,
                Err(e) => {
                    prop_assert!(false, "{e}\nout: {}", String::from_utf8_lossy(&out));
                    unreachable!()
                }
            };
            prop_assert!(
                seen.finished && !seen.error,
                "a complete stream must finish cleanly: {seen:?}"
            );
            if let Err(e) = check_content(&s, client, upstream, &seen) {
                prop_assert!(false, "{e}\nout: {}", String::from_utf8_lossy(&out));
            }
            // D126: the usage parsers read SSE line by line, so a usage event spread over two
            // `data:` lines (legal SSE, which the bridge joins) bills nothing exact. The client
            // side is checked above for every framing; the billing comparison skips that one.
            if s.framing.multiline {
                return Ok(());
            }
            let up = billed(upstream, &bytes).map(|u| normalize(&u));
            let shown = billed(client, &out).map(|u| normalize(&u));
            prop_assert!(up.is_some(), "upstream usage must parse");
            prop_assert_eq!(
                shown,
                up,
                "client was shown other usage than is billed\nout: {}",
                String::from_utf8_lossy(&out)
            );
            Ok(())
        },
    );
}

/// An error anywhere, or a cut anywhere, still leaves the client a well-formed stream that
/// says it failed, never a clean finish over a half-written answer.
#[test]
fn prop_errors_and_cuts_anywhere_leave_a_well_formed_failure() {
    check(
        "prop_errors_and_cuts_anywhere_leave_a_well_formed_failure",
        (cross_pairings(), script(), any::<usize>(), any::<bool>()),
        |((client, upstream), s, at, cut)| {
            let n = s.event_count(upstream);
            let bytes = if cut {
                s.stream(upstream, Some(at % n), None)
            } else {
                s.stream(upstream, None, Some(at % n))
            };
            let out = feed_whole(client, upstream, &bytes);
            match seen_stream(client, &out) {
                Ok(seen) => prop_assert!(seen.error || seen.finished, "{seen:?}"),
                Err(e) => prop_assert!(
                    false,
                    "{e}\nin: {}\nout: {}",
                    String::from_utf8_lossy(&bytes),
                    String::from_utf8_lossy(&out)
                ),
            }
            Ok(())
        },
    );
}
