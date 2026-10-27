//! `signed_id` on arbitrary bytes: `Signer::verify` on an arbitrary token, `unsign_request` on an
//! arbitrary request body, and `Relay::feed` on an arbitrary response (JSON or SSE) in
//! fuzzer-chosen chunks. No panic; and a token that verifies for a tenant must re-sign to itself
//! for that tenant (`props_signed_id`'s round-trip, run in reverse).
//!
//! Input: bytes 0..8 tenant, byte 8 flags (bit 0 streaming, bit 1 catalog, bits 2.. status
//! selector), byte 9 the cut count `n` (0..16), the next `n` bytes chunk lengths, the rest is the
//! payload.
#![no_main]

use beyond_gateway::__test::signed_id::{Relay, Signer};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

fn signer() -> &'static Signer {
    static S: OnceLock<Signer> = OnceLock::new();
    S.get_or_init(|| {
        Signer::new(&[(b'1', &[7u8; 32]), (b'z', &[3u8; 32])], b'z')
            .unwrap_or_else(|_| unreachable!("fixed valid keys"))
    })
}

fuzz_target!(|data: &[u8]| {
    let Some((tenant, rest)) = data.split_first_chunk::<8>() else {
        return;
    };
    let [flags, n, rest @ ..] = rest else {
        return;
    };
    let tenant = u64::from_le_bytes(*tenant);
    let s = signer();
    let n = (*n as usize % 16).min(rest.len());
    let (cuts, payload) = rest.split_at(n);

    if let Ok(token) = std::str::from_utf8(payload) {
        if let Some(raw) = s.verify(tenant, token) {
            // A verified token is a signed id this gateway issued: its own tenant's, so signing
            // the provider id again must verify back to the same provider id.
            assert_eq!(s.verify(tenant, &s.sign(tenant, &raw)), Some(raw));
        }
    }

    let _ = s.unsign_request(tenant, payload, flags & 2 != 0);

    let streaming = flags & 1 != 0;
    let status = [200, 204, 400, 500][(*flags as usize >> 2) % 4];
    let mut relay = Relay::new(tenant);
    if !relay.begin(status, streaming) {
        return;
    }
    let mut at = 0;
    for c in cuts {
        let end = (at + *c as usize).min(payload.len());
        if relay.feed(s, &payload[at..end], false).is_err() {
            return;
        }
        at = end;
    }
    let _ = relay.feed(s, &payload[at..], true);
});
