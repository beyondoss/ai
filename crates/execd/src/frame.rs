//! v1.2 frames: one JSON object per line, output as standard base64 with padding.

use base64::Engine as _;
use bytes::Bytes;
use serde_json::Value;

/// The most raw output one frame carries: the spec's ceiling (`stream::MAX_FRAME_PAYLOAD`, 64 KiB).
pub(crate) const FRAME_MAX: usize = 64 * 1024;

/// An output frame, `{"<key>":"<base64>"}\n`. `data` must be non-empty and at most [`FRAME_MAX`].
pub(crate) fn output(key: &str, data: &[u8]) -> Bytes {
    let mut s = String::with_capacity(key.len() + data.len().div_ceil(3) * 4 + 8);
    s.push_str("{\"");
    s.push_str(key);
    s.push_str("\":\"");
    base64::engine::general_purpose::STANDARD.encode_string(data, &mut s);
    s.push_str("\"}\n");
    Bytes::from(s)
}

/// Any other frame, from a JSON value.
pub(crate) fn json(frame: &Value) -> Bytes {
    let mut v = frame.to_string().into_bytes();
    v.push(b'\n');
    Bytes::from(v)
}

/// The heartbeat: `{}`.
pub(crate) fn heartbeat() -> Bytes {
    Bytes::from_static(b"{}\n")
}

/// Fuzz seam: every frame of `data` is one line of JSON whose payload decodes back to the bytes.
#[allow(clippy::expect_used)]
pub(crate) fn fuzz(data: &[u8]) {
    for chunk in data.chunks(FRAME_MAX) {
        let f = output("stdout", chunk);
        let (line, nl) = f.split_at(f.len() - 1);
        assert_eq!(nl, b"\n");
        assert!(!line.contains(&b'\n'));
        let v: Value = serde_json::from_slice(line).expect("a frame is JSON");
        let back = base64::engine::general_purpose::STANDARD
            .decode(v["stdout"].as_str().expect("a string payload"))
            .expect("the payload is base64");
        assert_eq!(back, chunk);
    }
}

/// Standard base64 with padding.
pub(crate) fn b64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}
