//! Property tests: numbers in tool arguments keep their value through translation.
//!
//! A tool call's arguments are JSON the model wrote, and a client replays them verbatim in its
//! history. Wherever translation re-parses them (a Chat `arguments` string onto Messages, a
//! Messages `tool_use.input` onto Chat or Responses, non-stream), a number must come out with the
//! value it went in with — the spelling may change (`1E5` → `100000.0`), the value may not.
//!
//! Numbers are compared exactly, as decimal (sign, significant digits, exponent), never as `f64`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::route::Endpoint;
use beyond_ai::translate;
use props_support::*;
use proptest::prelude::*;
use serde_json::{Value, json};

/// A JSON number literal as an exact decimal: (negative, significant digits, exponent of the
/// last digit). `None` for zero's sign-insensitive form: `(false, "", 0)`.
fn canon(lit: &str) -> (bool, String, i64) {
    let (neg, rest) = match lit.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, lit),
    };
    let (mant, exp) = match rest.find(['e', 'E']) {
        Some(i) => (&rest[..i], rest[i + 1..].parse::<i64>().unwrap()),
        None => (rest, 0),
    };
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut digits = format!("{int}{frac}");
    let mut exp = exp - frac.len() as i64;
    let lead = digits.len() - digits.trim_start_matches('0').len();
    digits.drain(..lead);
    while digits.ends_with('0') {
        digits.pop();
        exp += 1;
    }
    if digits.is_empty() {
        return (false, String::new(), 0);
    }
    (neg, digits, exp)
}

/// The literal after `"n":` in a JSON text.
fn number_after_n(text: &str) -> Option<String> {
    let at = text.find("\"n\":")? + 4;
    let rest = text[at..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E')))
        .unwrap_or(rest.len());
    Some(rest[..end].to_owned())
}

/// Any JSON number literal: integers in and past `i64`/`u64`, decimals of any length (money,
/// coordinates, scores, and the 15-20 digit floats a model writes), exponent forms, and the
/// subnormal and huge magnitudes at `f64`'s edges. A `Value` keeps every integer within `u64` as
/// written; anything else was the D127 shape (`-1.32417719719006e-11` came out as
/// `-1.3241771971900598e-11`, a 30-digit integer as `1.2345678901234568e29`).
fn number() -> impl Strategy<Value = String> {
    let digits = |n: std::ops::Range<usize>| prop::collection::vec(0u8..10, n);
    prop_oneof![
        any::<i64>().prop_map(|n| n.to_string()),
        any::<u64>().prop_map(|n| n.to_string()),
        // Decimals of up to 7 significant digits, as money, coordinates and scores are written.
        (1u32..10_000_000, 0usize..8, any::<bool>()).prop_map(|(m, places, neg)| {
            let s = format!("{m:0>8}");
            let (int, frac) = s.split_at(8 - places);
            let int = int.trim_start_matches('0');
            let int = if int.is_empty() { "0" } else { int };
            let sign = if neg { "-" } else { "" };
            if frac.is_empty() {
                format!("{sign}{int}")
            } else {
                format!("{sign}{int}.{frac}")
            }
        }),
        (1u32..1000, -20i32..20).prop_map(|(m, e)| format!("{m}e{e}")),
        // Integers past `u64` (ids, account numbers).
        (1u8..10, digits(19..40), any::<bool>()).prop_map(|(lead, rest, neg)| {
            let rest: String = rest.iter().map(|d| char::from(b'0' + d)).collect();
            format!("{}{lead}{rest}", if neg { "-" } else { "" })
        }),
        // Long decimals in plain and exponent form, out to `f64`'s subnormal and huge edges.
        (1u8..10, digits(8..25), -330i32..308, any::<bool>()).prop_map(|(lead, rest, e, neg)| {
            let rest: String = rest.iter().map(|d| char::from(b'0' + d)).collect();
            format!("{}{lead}.{rest}e{e}", if neg { "-" } else { "" })
        }),
        (digits(1..10), digits(8..25)).prop_map(|(int, frac)| {
            let int: String = int.iter().map(|d| char::from(b'0' + d)).collect();
            let frac: String = frac.iter().map(|d| char::from(b'0' + d)).collect();
            let int = int.trim_start_matches('0');
            format!("{}.{frac}", if int.is_empty() { "0" } else { int })
        }),
    ]
}

/// The four places translation re-parses a tool call's arguments. Returns the literal that came
/// out, or why none did.
fn through(path: u8, lit: &str) -> Result<String, String> {
    let args = format!("{{\"n\":{lit}}}");
    let out = match path % 4 {
        // A Chat history call onto Messages.
        0 => {
            let body = format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}},{{"role":"assistant","content":null,"tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"f","arguments":{}}}}}]}},{{"role":"tool","tool_call_id":"call_1","content":"ok"}}]}}"#,
                json!(args)
            );
            let out = translate::request(
                Endpoint::ChatCompletions,
                Endpoint::Messages,
                body.as_bytes(),
                "claude-sonnet-4-5",
            );
            String::from_utf8(out).unwrap()
        }
        // A Messages tool_use onto a Chat client (non-stream).
        1 | 2 => {
            let body = format!(
                r#"{{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{{"type":"tool_use","id":"toolu_1","name":"f","input":{args}}}],"stop_reason":"tool_use","usage":{{"input_tokens":1,"output_tokens":1}}}}"#
            );
            let client = if path % 4 == 1 {
                Endpoint::ChatCompletions
            } else {
                Endpoint::Responses
            };
            let out = translate::response_json(Endpoint::Messages, client, body.as_bytes());
            let v: Value = serde_json::from_slice(&out).map_err(|e| e.to_string())?;
            let a = v
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .or_else(|| v.pointer("/output/0/arguments"))
                .and_then(Value::as_str)
                .ok_or_else(|| format!("no arguments in {v}"))?;
            a.to_owned()
        }
        // A Chat tool call onto a Messages client (non-stream).
        _ => {
            let body = format!(
                r#"{{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"f","arguments":{}}}}}]}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}}}"#,
                json!(args)
            );
            let out = translate::response_json(
                Endpoint::ChatCompletions,
                Endpoint::Messages,
                body.as_bytes(),
            );
            String::from_utf8(out).unwrap()
        }
    };
    number_after_n(&out).ok_or_else(|| format!("no \"n\" in {out}"))
}

#[test]
fn prop_tool_argument_numbers_keep_their_value() {
    check(
        "prop_tool_argument_numbers_keep_their_value",
        (any::<u8>(), number()),
        |(path, lit)| {
            let got = through(path, &lit).map_err(TestCaseError::fail)?;
            prop_assert_eq!(
                canon(&got),
                canon(&lit),
                "path {}: {} became {}",
                path % 4,
                lit,
                got
            );
            Ok(())
        },
    );
}
