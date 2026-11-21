//!
//! A tool call's arguments are JSON the model wrote, and the client replays them in its history or
//! hands them to its tool. Translation moves them between a JSON string (Chat Completions and
//! Responses `arguments`) and a JSON value (Messages `tool_use.input`), which parsing into a `Value`
//! would do lossily: an integer past `u64` becomes an `f64`, and a decimal is rounded to the nearest
//! `f64`. serde_json's `arbitrary_precision` would keep every number's text, but it is a
//! crate-wide feature that breaks `untagged` enums holding numbers, which async-nats' JetStream
//! responses are. So only arguments holding a number a `Value` keeps as an `f64` (a decimal, an
//! exponent, a huge integer) keep their text, as a one-entry object (`RAW_JSON_KEY`) that [`Exact`]
//! writes back verbatim; every other argument, and every other number, is an ordinary `Value`.

use super::*;

/// The key of a one-entry object standing for JSON text kept verbatim. A client object carrying it
/// is written as its value's JSON text only if that is one valid JSON value, which the client could
/// have sent as is.
pub(super) const RAW_JSON_KEY: &str = "\u{0}beyond:raw-json";

/// Whether a `Value` holds a number parsed as an `f64`, whose text it may not reproduce.
pub(super) fn has_float(v: &Value) -> bool {
    match v {
        Value::Number(n) => n.is_f64(),
        Value::Array(a) => a.iter().any(has_float),
        Value::Object(m) => m.values().any(has_float),
        _ => false,
    }
}

/// `v`, parsed from `text`, or `text` kept verbatim when `v` would not reproduce its numbers.
pub(super) fn exact_value(text: &str, v: Value) -> Value {
    if has_float(&v) {
        let mut m = Map::new();
        m.insert(RAW_JSON_KEY.to_owned(), Value::String(text.to_owned()));
        Value::Object(m)
    } else {
        v
    }
}

/// The text a verbatim-kept value stands for.
pub(super) fn raw_json(v: &Value) -> Option<&str> {
    match v {
        Value::Object(m) if m.len() == 1 => m.get(RAW_JSON_KEY)?.as_str(),
        _ => None,
    }
}

/// Tool arguments as the JSON text a Chat Completions or Responses `arguments` string holds.
pub(super) fn arguments_text(v: &Value) -> String {
    raw_json(v).map_or_else(|| value_string(v), |raw| raw.trim().to_owned())
}

/// Serializes a `Value` as serde_json does, except that a verbatim-kept value is written as its
/// text.
pub(super) struct Exact<'a>(pub(super) &'a Value);

impl serde::Serialize for Exact<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self.0 {
            Value::Object(m) => {
                if let Some(raw) = raw_json(self.0)
                    && let Ok(raw) = serde_json::from_str::<&serde_json::value::RawValue>(raw)
                {
                    return raw.serialize(s);
                }
                let mut map = s.serialize_map(Some(m.len()))?;
                for (k, v) in m {
                    map.serialize_entry(k, &Exact(v))?;
                }
                map.end()
            }
            Value::Array(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for v in a {
                    seq.serialize_element(&Exact(v))?;
                }
                seq.end()
            }
            v => v.serialize(s),
        }
    }
}

/// A request or response body as a `Value`. A Messages body is parsed once with [`KeepInputs`],
/// which keeps each `input`'s text where a `Value` would not reproduce its numbers; any
/// other body is parsed once, plainly.
pub(super) fn parse_body(wire: Endpoint, body: &[u8]) -> Option<Value> {
    if wire != Endpoint::Messages {
        return serde_json::from_slice::<Value>(body).ok();
    }
    let mut de = serde_json::Deserializer::from_slice(body);
    let v = serde::de::DeserializeSeed::deserialize(KeepInputs, &mut de).ok()?;
    de.end().ok()?;
    Some(v)
}

/// Builds a `Value` as serde_json does, except that every `input` member keeps its text when it
/// holds a number a `Value` would not reproduce (see [`exact_value`]).
struct KeepInputs;

impl<'de> serde::de::DeserializeSeed<'de> for KeepInputs {
    type Value = Value;

    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        d.deserialize_any(self)
    }
}

/// Only the methods serde_json's `deserialize_any` calls: a string comes through `visit_str`
/// (borrowed or copied, never `visit_string`'s owned form), and JSON has no `Option` to visit.
impl<'de> serde::de::Visitor<'de> for KeepInputs {
    type Value = Value;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E>(self, b: bool) -> Result<Value, E> {
        Ok(Value::Bool(b))
    }

    fn visit_i64<E>(self, n: i64) -> Result<Value, E> {
        Ok(Value::from(n))
    }

    fn visit_u64<E>(self, n: u64) -> Result<Value, E> {
        Ok(Value::from(n))
    }

    fn visit_f64<E>(self, n: f64) -> Result<Value, E> {
        Ok(serde_json::Number::from_f64(n).map_or(Value::Null, Value::Number))
    }

    fn visit_str<E>(self, s: &str) -> Result<Value, E> {
        Ok(Value::String(s.to_owned()))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(v) = seq.next_element_seed(KeepInputs)? {
            out.push(v);
        }
        Ok(Value::Array(out))
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut out = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let v = if key == "input" {
                let raw: &'de serde_json::value::RawValue = map.next_value()?;
                let v =
                    serde_json::from_str::<Value>(raw.get()).map_err(serde::de::Error::custom)?;
                exact_value(raw.get(), v)
            } else {
                map.next_value_seed(KeepInputs)?
            };
            out.insert(key, v);
        }
        Ok(Value::Object(out))
    }
}

pub(super) fn parse_arguments(v: Option<&Value>) -> Value {
    match v {
        Some(Value::String(s)) => match serde_json::from_str(s) {
            Ok(v) => exact_value(s, v),
            Err(_) => json!({}),
        },
        Some(o) if o.is_object() => o.clone(),
        _ => json!({}),
    }
}

pub(super) fn max_tokens_of(v: &Value) -> Option<u64> {
    v.get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| v.get("max_completion_tokens").and_then(Value::as_u64))
}

pub(super) fn message_text(m: &Value) -> Option<String> {
    match m.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let t: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
                .collect();
            Some(t)
        }
        _ => None,
    }
}

/// Copy `key` when set. An explicit `null` is "not set" (OpenAI SDKs send unset options that way),
/// and forwarded it would be a 400 by name on an upstream without the field.
pub(super) fn copy_if(out: &mut Map<String, Value>, v: &Value, key: &str) {
    if let Some(x) = v.get(key).filter(|x| !x.is_null()) {
        out.insert(key.to_owned(), x.clone());
    }
}

/// Unix seconds, for a `created` / `created_at` the upstream did not supply.
pub(super) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A process-unique id (`{prefix}_gw…`) for an object the upstream left unnamed. Stable for the
/// life of one response because the caller keeps it; unique across responses so a client that
/// keys on it (a tool call id, a Responses item id) never sees two alike.
pub(super) fn fresh_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX));
    format!("{prefix}_gw{t:x}{n:04x}")
}

pub(super) fn id_or_fresh(v: Option<&Value>, prefix: &str) -> String {
    v.and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map_or_else(|| fresh_id(prefix), str::to_owned)
}

pub(super) fn non_empty_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
}

pub(super) fn value_string(v: &Value) -> String {
    serde_json::to_string(&Exact(v)).unwrap_or_else(|_| "{}".into())
}
