//! ShellFs's hand-rolled base64 (file content moves through a command's stdout, a lossy `String`,
//! as base64). Round trip: `decode(encode(x)) == x`, and the encoding is the standard, padded one
//! (checked against the `base64` crate). Raw: no panic on any text; whatever decodes re-encodes to
//! something that decodes to the same bytes.
#![no_main]

use base64::Engine as _;
use beyond_agent::tools::fs::shell::__fuzz::{decode_base64, encode_base64};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let enc = encode_base64(data);
    assert_eq!(enc, base64::engine::general_purpose::STANDARD.encode(data));
    assert_eq!(decode_base64(&enc).as_deref(), Ok(data));
    // `base64 -w0` vs a wrapping `base64`: line breaks inside are tolerated.
    let wrapped: String = enc
        .as_bytes()
        .chunks(76)
        .flat_map(|c| c.iter().copied().chain(std::iter::once(b'\n')))
        .map(char::from)
        .collect();
    assert_eq!(decode_base64(&wrapped).as_deref(), Ok(data));

    let text = String::from_utf8_lossy(data);
    if let Ok(bytes) = decode_base64(&text) {
        assert_eq!(
            decode_base64(&encode_base64(&bytes)).as_deref(),
            Ok(&bytes[..])
        );
        // Decoded bytes cannot outnumber what 6-bit symbols can carry.
        assert!(bytes.len() <= text.len() / 4 * 3 + 2);
    }
});
