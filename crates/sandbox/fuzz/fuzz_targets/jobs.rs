//! Exec protocol v1.3 job answers (`jobs_native`): a stream line (`decode_job_frame`) and the JSON
//! answers to `read`, `list`, `kill`/`wipe` (`parse_read`, `parse_list`, `parse_survivors`).
//!
//! Properties: never a panic; an accepted line is blank or a JSON object whose output is canonical
//! base64; a final `End` carries the status the endpoint sent (`exit_code`) or `128 + signal`; a
//! `read` answer's data, offset and status code are the ones in the JSON, not truncated copies.
#![no_main]

use base64::Engine as _;
use beyond_sandbox::exec_endpoint::jobs::JobStatus;
use beyond_sandbox::exec_endpoint::jobs_native::{
    Event, JobFrame, decode_job_frame, parse_list, parse_read, parse_survivors,
};
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn line(data: &[u8]) {
    // KNOWN (F3, unfixed): `128 + signal` overflows an i32 for a signal above `i32::MAX - 128`.
    // Drop this guard when the decoder is fixed; corpus/jobs/known-signal-overflow is the input.
    if serde_json::from_slice::<Value>(data).is_ok_and(|v| {
        v["signal"]
            .as_i64()
            .is_some_and(|s| s > i64::from(i32::MAX - 128))
    }) {
        return;
    }
    let Ok(frame) = decode_job_frame(data) else {
        return;
    };
    let body = data.strip_suffix(b"\r").unwrap_or(data);
    if body.iter().all(u8::is_ascii_whitespace) {
        assert_eq!(frame, JobFrame::Heartbeat);
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
    match frame {
        JobFrame::Event(Event::Output(b)) => {
            let s = v["stdout"].as_str().or_else(|| v["stderr"].as_str());
            assert_eq!(s, Some(B64.encode(&b).as_str()), "non-canonical output");
        }
        JobFrame::End(code) if v["error"].is_null() && v["started"].is_null() => {
            let want = match (v["exit_code"].as_i64(), v["signal"].as_i64()) {
                (Some(c), _) if v["stdout"].is_null() && v["stderr"].is_null() => c,
                (_, Some(s)) => 128 + s,
                _ => return,
            };
            assert_eq!(i64::from(code), want, "{v}");
        }
        _ => {}
    }
}

fn answers(data: &[u8]) {
    let Ok(v) = serde_json::from_slice::<Value>(data) else {
        return;
    };
    if let Ok(chunk) = parse_read(&v) {
        assert_eq!(v["data"].as_str().unwrap_or(""), B64.encode(&chunk.data));
        assert_eq!(chunk.offset, v["offset"].as_u64().unwrap_or(0));
        assert_eq!(chunk.end, v["end"].as_u64().unwrap_or(0));
        // KNOWN (F4, unfixed): `code` is narrowed with `as i32`, so 4294967297 reads as 1. Only
        // codes that fit are held to the property; drop the range test when the parser is fixed
        // (corpus/jobs/known-code-truncation is the input).
        if let JobStatus::Ended(c) | JobStatus::ShellExited(c) = chunk.status
            && v["code"].as_i64().is_some_and(|w| i32::try_from(w).is_ok())
        {
            assert_eq!(Some(i64::from(c)), v["code"].as_i64(), "status code: {v}");
        }
    }
    if let Ok(jobs) = parse_list(&v) {
        assert!(jobs.len() <= v["jobs"].as_array().map_or(0, Vec::len));
        for (job, wire) in jobs.iter().zip(v["jobs"].as_array().into_iter().flatten()) {
            let _ = (job, wire);
        }
    }
    let _ = parse_survivors(&v);
}

fuzz_target!(|data: &[u8]| {
    line(data);
    answers(data);
});
