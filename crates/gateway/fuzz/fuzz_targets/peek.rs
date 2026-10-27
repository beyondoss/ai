//! `peek`'s scanners on arbitrary bytes, whole and cut at fuzzer-chosen points: no panic, and every
//! span and injection point lies inside the body (`props_peek::prop_scanners_never_panic_on_any_bytes`).
//!
//! Input: the first byte picks a cut count `n` (0..8), the next `n` bytes are cut lengths, the rest
//! is the body.
#![no_main]

use beyond_gateway::__test::peek::{self, ModelScanner};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&n, rest)) = data.split_first() else {
        return;
    };
    let n = (n as usize % 8).min(rest.len());
    let (cuts, body) = rest.split_at(n);

    let scan = peek::scan_buffered(body);
    for (s, e) in scan
        .model_span
        .into_iter()
        .chain(scan.limit_spans.into_iter().flatten())
    {
        assert!(s <= e && e <= body.len());
    }
    for at in scan.inject_at.into_iter().chain(scan.stream_options_at) {
        assert!(at <= body.len());
    }
    let _ = peek::structure(body);
    let _ = peek::root_members(body);
    let _ = peek::value_end(body, 0);

    for mut s in [ModelScanner::new(), ModelScanner::for_response()] {
        let mut at = 0;
        for c in cuts {
            let end = (at + *c as usize).min(body.len());
            s.feed(&body[at..end]);
            at = end;
        }
        s.feed(&body[at..]);
        let _ = s.take_model();
    }
});
