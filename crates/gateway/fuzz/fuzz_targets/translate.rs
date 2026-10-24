//! Translation on arbitrary bytes: a provider's response body (`response_json_status`), a provider's
//! SSE stream fed through `SseBridge` in fuzzer-chosen chunks, and a request body. Every client x
//! upstream dialect pairing. No panic; a request that is not a JSON object passes through
//! unchanged (`props_translate::prop_any_bytes_translate_without_panicking`). Chunked-versus-whole
//! equivalence is `props_sse`'s job (ids and timestamps are minted per bridge); here every
//! chunking only has to survive.
//!
//! Input: byte 0 picks the (client, upstream) pairing, byte 1 an HTTP status low byte, byte 2 the
//! cut count `n` (0..16), the next `n` bytes are chunk lengths, the rest is the body.
#![no_main]

use beyond_ai::__test::route::Endpoint;
use beyond_ai::__test::translate::{self, SseBridge};
use libfuzzer_sys::fuzz_target;

const DIALECTS: [Endpoint; 3] = [
    Endpoint::ChatCompletions,
    Endpoint::Messages,
    Endpoint::Responses,
];

fuzz_target!(|data: &[u8]| {
    let [pair, status, n, rest @ ..] = data else {
        return;
    };
    let client = DIALECTS[*pair as usize % 3];
    let upstream = DIALECTS[(*pair as usize / 3) % 3];
    let status = 200 + (*status as u16 % 400);
    let n = (*n as usize % 16).min(rest.len());
    let (cuts, body) = rest.split_at(n);

    let _ = translate::response_json_status(upstream, client, status, body);

    let out = translate::request(client, upstream, body, "claude-opus-4-8");
    if !matches!(
        serde_json::from_slice::<serde_json::Value>(body),
        Ok(serde_json::Value::Object(_))
    ) {
        assert_eq!(out, body, "a non-object request body must pass through");
    }

    let mut bridge = SseBridge::new(client, upstream);
    let mut at = 0;
    for c in cuts {
        let end = (at + *c as usize).min(body.len());
        let _ = bridge.feed(&body[at..end], false);
        at = end;
    }
    let _ = bridge.feed(&body[at..], false);
    let _ = bridge.feed(&[], true);
});
