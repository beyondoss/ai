//! Canonical JSON: keys sorted, two-space indent, one trailing newline. A snapshot written this
//! way is byte-identical for the same content however the vendor ordered its keys.

use serde_json::Value;
use std::fmt::Write as _;

pub fn canonical(v: &Value) -> String {
    let mut out = String::new();
    write(&mut out, v, 0);
    out.push('\n');
    out
}

fn write(out: &mut String, v: &Value, depth: usize) {
    let pad = |out: &mut String, d: usize| {
        for _ in 0..d {
            out.push_str("  ");
        }
    };
    match v {
        Value::Object(m) if !m.is_empty() => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push_str("{\n");
            for (i, k) in keys.iter().enumerate() {
                pad(out, depth + 1);
                let _ = write!(out, "{}: ", Value::String((*k).clone()));
                if let Some(x) = m.get(*k) {
                    write(out, x, depth + 1);
                }
                if i + 1 < keys.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            pad(out, depth);
            out.push('}');
        }
        Value::Array(a) if !a.is_empty() => {
            out.push_str("[\n");
            for (i, x) in a.iter().enumerate() {
                pad(out, depth + 1);
                write(out, x, depth + 1);
                if i + 1 < a.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            pad(out, depth);
            out.push(']');
        }
        other => {
            let _ = write!(out, "{other}");
        }
    }
}

/// `v[key]` as a string, or an error naming what was expected.
pub fn str_at<'a>(v: &'a Value, key: &str, what: &str) -> crate::Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{what}: no string `{key}`"))
}
