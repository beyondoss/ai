//! Exec protocol v1.3 job answers (`jobs_native`): a stream line (`decode_job_frame`) and the JSON
//! answers to `read`, `list`, `kill`/`wipe` (`parse_read`, `parse_list`, `parse_survivors`).
//!
//! Properties: never a panic; an accepted line is blank or a JSON object whose output is canonical
//! base64; a final `End` carries the status the endpoint sent (`exit_code`) or `128 + signal`; a
//! value out of range (a status or code that does not fit an `i32`) is an error, never a panic or
//! a truncated copy; a `read` answer's data and offset are the ones in the JSON.
#![no_main]

use base64::Engine as _;
use beyond_agent::exec_endpoint::jobs::JobStatus;
use beyond_agent::exec_endpoint::jobs_native::{
    Event, JobFrame, decode_job_frame, parse_list, parse_read, parse_survivors,
};
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn line(data: &[u8]) {
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
        // A status that does not fit an i32, or a `128 + signal` that does not, is an error (the
        // decoder returned `Err` and never reaches here). The expected value is computed checked,
        // so an out-of-range input that decodes to an `End` fails the assertion.
        JobFrame::End(code) if v["error"].is_null() && v["started"].is_null() => {
            let want = match (v["exit_code"].as_i64(), v["signal"].as_i64()) {
                (Some(c), _) if v["stdout"].is_null() && v["stderr"].is_null() => {
                    i32::try_from(c).ok()
                }
                (_, Some(s)) => i32::try_from(s).ok().and_then(|s| 128i32.checked_add(s)),
                _ => return,
            };
            assert_eq!(Some(code), want, "a value out of range was accepted: {v}");
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
        if let JobStatus::Ended(c) | JobStatus::ShellExited(c) = chunk.status {
            assert_eq!(Some(i64::from(c)), v["code"].as_i64(), "status code: {v}");
        }
    }
    // A present code that does not fit an i32 is an error, whatever the state says.
    let code_out_of_range = match v.get("code") {
        None | Some(Value::Null) => false,
        Some(c) => c.as_i64().and_then(|w| i32::try_from(w).ok()).is_none(),
    };
    if code_out_of_range {
        assert!(
            parse_read(&v).is_err(),
            "an out-of-range code was accepted: {v}"
        );
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
