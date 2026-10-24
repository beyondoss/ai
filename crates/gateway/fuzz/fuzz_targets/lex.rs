//! `lex::unescape` / `unescapes_to` on arbitrary string contents: no panic, and the decoder agrees
//! with serde_json on what a JSON string decodes to (`props_lex::prop_unescape_matches_serde`, on
//! raw bytes).
#![no_main]

use beyond_ai::__test::lex::{unescape, unescapes_to};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|raw: &[u8]| {
    let got = unescape(raw);
    let mut quoted = Vec::with_capacity(raw.len() + 2);
    quoted.push(b'"');
    quoted.extend_from_slice(raw);
    quoted.push(b'"');
    let want = serde_json::from_slice::<String>(&quoted).ok();
    assert_eq!(got, want, "unescape disagrees with serde_json");
    if let Some(w) = &want {
        assert!(unescapes_to(raw, w));
    }
});
