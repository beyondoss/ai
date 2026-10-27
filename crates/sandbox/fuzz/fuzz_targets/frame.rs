//! Exec protocol v1.2 stream frames (`stream::decode_line`).
//!
//! * The golden vectors (`tests/fixtures/exec/frames.json`) still decode as specified.
//! * Round trip: a frame built from the spec's table decodes to exactly its parts, output first, then
//!   `shell_exit`, then the one final frame; unknown fields change nothing.
//! * Raw: never panics; an accepted line is blank or a JSON *object* (not an array a struct happens
//!   to read positionally), carries at most one final frame and only as the last, and its output is
//!   standard padded base64 (re-encoding the payload gives the field back). An empty or oversized
//!   output frame is a rule for endpoints (`MUST NOT`); this client tolerates both on purpose, so
//!   neither is asserted.
#![no_main]

use std::sync::Once;

use base64::Engine as _;
use beyond_sandbox::exec_endpoint::stream::{Frame, decode_line};
use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use serde_json::{Value, json};

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn vectors() {
    let doc: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/exec/frames.json"))
            .expect("fixture is JSON");
    let b64 = |v: &Value| {
        B64.decode(v.as_str().unwrap_or(""))
            .expect("fixture base64")
    };
    for case in doc["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap_or("?");
        let (mut stdout, mut stderr, mut beats, mut end) = (Vec::new(), Vec::new(), 0u64, None);
        let mut rejected = false;
        'lines: for line in case["lines"].as_array().expect("lines") {
            match decode_line(line.as_str().expect("line").as_bytes()) {
                Err(_) => {
                    rejected = true;
                    break 'lines;
                }
                Ok(frames) => {
                    for f in frames {
                        assert!(end.is_none(), "{name}: a frame followed the final frame");
                        match f {
                            Frame::Stdout(b) => stdout.extend(b),
                            Frame::Stderr(b) => stderr.extend(b),
                            Frame::Heartbeat => beats += 1,
                            Frame::ShellExit(_) => {}
                            other => end = Some(other),
                        }
                    }
                }
            }
        }
        let want = &case["expect"];
        if want["invalid"].as_bool() == Some(true) {
            assert!(rejected, "{name}: must be rejected");
            continue;
        }
        assert!(!rejected, "{name}: rejected");
        assert_eq!(stdout, b64(&want["stdout_base64"]), "{name}");
        assert_eq!(stderr, b64(&want["stderr_base64"]), "{name}");
        assert_eq!(beats, want["heartbeats"].as_u64().unwrap_or(0), "{name}");
        assert_eq!(end.is_some(), !want["end"].is_null(), "{name}: final frame");
    }
}

fn structured(u: &mut Unstructured<'_>) -> libfuzzer_sys::arbitrary::Result<()> {
    let mut obj = serde_json::Map::new();
    let mut want = Vec::new();
    let payload = |u: &mut Unstructured<'_>| -> libfuzzer_sys::arbitrary::Result<Vec<u8>> {
        let n = u.int_in_range(1..=300usize)?;
        Ok(u.bytes(n.min(u.len()))?.to_vec())
    };
    if u.ratio(1, 2)? {
        let p = payload(u)?;
        obj.insert("stdout".into(), B64.encode(&p).into());
        want.push(Frame::Stdout(p));
    }
    if u.ratio(1, 2)? {
        let p = payload(u)?;
        obj.insert("stderr".into(), B64.encode(&p).into());
        want.push(Frame::Stderr(p));
    }
    if u.ratio(1, 3)? {
        let c = u.arbitrary::<i32>()?;
        obj.insert("shell_exit".into(), c.into());
        want.push(Frame::ShellExit(c));
    }
    match u.int_in_range(0..=4u8)? {
        0 => {
            let c = u.arbitrary::<i32>()?;
            obj.insert("exit_code".into(), c.into());
            want.push(Frame::Exit(c));
        }
        1 => {
            let s = u.arbitrary::<i32>()?;
            obj.insert("signal".into(), s.into());
            want.push(Frame::Signal(s));
        }
        2 => {
            obj.insert("timed_out".into(), true.into());
            want.push(Frame::TimedOut);
        }
        3 => {
            obj.insert("error".into(), "lost the sandbox".into());
            want.push(Frame::Error("lost the sandbox".into()));
        }
        _ => {}
    }
    if want.is_empty() {
        want.push(Frame::Heartbeat);
    }
    if u.ratio(1, 3)? {
        obj.insert("seq".into(), json!(u.arbitrary::<u32>()?));
    }
    let mut line = serde_json::to_vec(&Value::Object(obj)).expect("serializes");
    if u.ratio(1, 4)? {
        line.push(b'\r');
    }
    assert_eq!(
        decode_line(&line).as_deref(),
        Ok(&want[..]),
        "{}",
        String::from_utf8_lossy(&line)
    );
    Ok(())
}

fn raw(line: &[u8]) {
    let Ok(frames) = decode_line(line) else {
        return;
    };
    assert!(!frames.is_empty());
    let finals = frames.iter().filter(|f| f.is_final()).count();
    assert!(finals <= 1, "{finals} final frames from one line");
    if finals == 1 {
        assert!(
            frames.last().is_some_and(Frame::is_final),
            "a frame follows the final one"
        );
    }
    let body = line.strip_suffix(b"\r").unwrap_or(line);
    if body.iter().all(u8::is_ascii_whitespace) {
        assert_eq!(frames, vec![Frame::Heartbeat]);
        return;
    }
    // An object, not an array a struct happens to read positionally. (Not re-parsed as a `Value`:
    // the decoder skips the contents of unknown fields without validating them, which is fine.)
    assert_eq!(
        body.iter().find(|b| !b.is_ascii_whitespace()),
        Some(&b'{'),
        "accepted a frame that is not a JSON object"
    );
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return;
    };
    for f in &frames {
        let (field, bytes) = match f {
            Frame::Stdout(b) => ("stdout", b),
            Frame::Stderr(b) => ("stderr", b),
            _ => continue,
        };
        assert_eq!(
            v[field].as_str(),
            Some(B64.encode(bytes).as_str()),
            "non-canonical {field}"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    static ONCE: Once = Once::new();
    ONCE.call_once(vectors);
    raw(data);
    let _ = structured(&mut Unstructured::new(data));
});
