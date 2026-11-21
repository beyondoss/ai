//! The Streamable HTTP routing headers SEP-3415 puts on `events/*` requests: `Mcp-Name` (the event
//! name) and one `Mcp-Param-{Name}` per `inputSchema` property annotated `x-mcp-header` that has a
//! value in `params.arguments`, encoded as the transport defines (SEP-2243: plain header-safe ASCII,
//! else the `=?base64?…?=` sentinel).
//!
//! Also the transport's constraints on those annotations: an event type that violates them is
//! excluded from `events/list` ([`annotations_valid`]), so one malformed descriptor cannot take the
//! rest down.
//!
//! rmcp implements the same rules for `tools/call` but keeps them crate-private, so the agent's tool
//! path has nothing to reuse; this is the one encoder for events. Only the direct-HTTP connection
//! uses it (stdio ignores `x-mcp-header`, which the transport allows).

use base64::Engine as _;
use base64::prelude::BASE64_STANDARD;
use http::{HeaderName, HeaderValue};
use serde_json::Value;

const PARAM_PREFIX: &str = "Mcp-Param-";
const B64_PREFIX: &str = "=?base64?";
const B64_SUFFIX: &str = "?=";

/// The `events/*` methods that carry `params.name` and so the routing headers. `events/list` has no
/// name and gets none.
pub(super) fn is_named_method(method: &str) -> bool {
    matches!(
        method,
        "events/poll" | "events/stream" | "events/subscribe" | "events/unsubscribe"
    )
}

/// Whether `value` cannot travel as a bare header value: leading or trailing space or tab, a
/// control or non-ASCII character, or text that already looks like the base64 sentinel.
fn requires_base64(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let b = value.as_bytes();
    if matches!(b.first(), Some(b' ' | b'\t')) || matches!(b.last(), Some(b' ' | b'\t')) {
        return true;
    }
    if value
        .chars()
        .any(|c| (c as u32) < 0x20 || (c as u32) > 0x7E)
    {
        return true;
    }
    value.starts_with(B64_PREFIX) && value.ends_with(B64_SUFFIX)
}

fn encode_value(value: &str) -> String {
    if requires_base64(value) {
        format!("{B64_PREFIX}{}{B64_SUFFIX}", BASE64_STANDARD.encode(value))
    } else {
        value.to_owned()
    }
}

/// A JSON primitive's header text: strings as they are, integers in decimal, booleans lowercase.
/// `null` and anything structured have none (the header is omitted).
fn primitive_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) if n.is_i64() || n.is_u64() => Some(n.to_string()),
        _ => None,
    }
}

fn is_tchar(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            '!' | '#'
                | '$'
                | '%'
                | '&'
                | '\''
                | '*'
                | '+'
                | '-'
                | '.'
                | '^'
                | '_'
                | '`'
                | '|'
                | '~'
        )
}

/// One `x-mcp-header` annotation reached from the schema root through `properties` keys alone.
struct Annotation<'a> {
    path: Vec<&'a str>,
    schema: &'a Value,
    header: &'a Value,
}

fn collect<'a>(
    properties: Option<&'a Value>,
    path: &mut Vec<&'a str>,
    out: &mut Vec<Annotation<'a>>,
) {
    let Some(Value::Object(properties)) = properties else {
        return;
    };
    for (name, schema) in properties {
        path.push(name);
        if let Some(header) = schema.get("x-mcp-header") {
            out.push(Annotation {
                path: path.clone(),
                schema,
                header,
            });
        }
        collect(schema.get("properties"), path, out);
        path.pop();
    }
}

fn annotations(input_schema: &Value) -> Vec<Annotation<'_>> {
    let mut out = Vec::new();
    collect(input_schema.get("properties"), &mut Vec::new(), &mut out);
    out
}

/// Whether `schema` holds an `x-mcp-header` anywhere but on the `properties` chain from the root
/// (`on_chain`: `schema` is the root or a property reached through `properties` keys alone).
fn has_off_chain(schema: &Value, on_chain: bool) -> bool {
    let Value::Object(map) = schema else {
        return false;
    };
    map.iter().any(|(key, value)| match key.as_str() {
        "x-mcp-header" => !on_chain,
        "properties" => value
            .as_object()
            .is_some_and(|m| m.values().any(|s| has_off_chain(s, on_chain))),
        "$defs" | "definitions" | "dependentSchemas" | "patternProperties" => value
            .as_object()
            .is_some_and(|m| m.values().any(|s| has_off_chain(s, false))),
        // Instance data, not subschemas.
        "default" | "const" | "enum" | "examples" => false,
        _ => match value {
            Value::Object(_) => has_off_chain(value, false),
            Value::Array(items) => items.iter().any(|s| has_off_chain(s, false)),
            _ => false,
        },
    })
}

/// The transport's constraints on `x-mcp-header` values: non-empty RFC 9110 tokens, unique ignoring
/// case, on `string`/`integer`/`boolean` properties only, each reached through `properties` keys
/// alone, none on the schema root. `Err` names the first violation.
pub(super) fn annotations_valid(input_schema: &Value) -> Result<(), String> {
    if input_schema.get("x-mcp-header").is_some() {
        return Err("x-mcp-header must annotate a property, not the schema root".into());
    }
    if has_off_chain(input_schema, true) {
        return Err("x-mcp-header must sit on a `properties`-only path from the root".into());
    }
    let mut seen = std::collections::HashSet::new();
    for a in annotations(input_schema) {
        let prop = a.path.join(".");
        let Some(header) = a.header.as_str() else {
            return Err(format!("property `{prop}`: x-mcp-header must be a string"));
        };
        if header.is_empty() || !header.chars().all(is_tchar) {
            return Err(format!(
                "property `{prop}`: x-mcp-header `{header}` is not a valid HTTP token"
            ));
        }
        if !seen.insert(header.to_ascii_lowercase()) {
            return Err(format!(
                "property `{prop}`: duplicate x-mcp-header `{header}` (case-insensitive)"
            ));
        }
        match a.schema.get("type").and_then(Value::as_str) {
            Some("string" | "integer" | "boolean") => {}
            other => {
                return Err(format!(
                    "property `{prop}`: x-mcp-header needs a string, integer or boolean property, got {other:?}"
                ));
            }
        }
    }
    Ok(())
}

/// The routing headers for one `events/*` request: `Mcp-Name` from `params.name`, and an
/// `Mcp-Param-{Name}` for each annotated property of the event's `input_schema` that has a value in
/// `params.arguments`. With no schema (the event is not in the cache) only `Mcp-Name` goes out, and
/// a server that wanted more answers `HeaderMismatch`, which the caller handles by re-listing.
pub(super) fn routing_headers(
    params: &Value,
    input_schema: Option<&Value>,
) -> Vec<(HeaderName, HeaderValue)> {
    let mut out = Vec::new();
    let mut push = |name: &str, value: &str| {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            out.push((n, v));
        }
    };
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return out;
    };
    push("Mcp-Name", &encode_value(name));
    if let (Some(schema), Some(arguments)) = (input_schema, params.get("arguments")) {
        for a in annotations(schema) {
            let Some(header) = a.header.as_str().filter(|h| !h.is_empty()) else {
                continue;
            };
            let Some(value) = a
                .path
                .iter()
                .try_fold(arguments, |v, key| v.get(key))
                .and_then(primitive_text)
            else {
                continue;
            };
            push(&format!("{PARAM_PREFIX}{header}"), &encode_value(&value));
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use serde_json::json;

    fn get(h: &[(HeaderName, HeaderValue)], name: &str) -> Option<String> {
        h.iter()
            .find(|(k, _)| k.as_str().eq_ignore_ascii_case(name))
            .map(|(_, v)| v.to_str().unwrap().to_owned())
    }

    #[test]
    fn the_name_and_each_annotated_argument_become_headers() {
        let schema = json!({"type": "object", "properties": {
            "region": {"type": "string", "x-mcp-header": "Region"},
            "limit": {"type": "integer", "x-mcp-header": "Limit"},
            "urgent": {"type": "boolean", "x-mcp-header": "Urgent"},
            "plain": {"type": "string"},
            "absent": {"type": "string", "x-mcp-header": "Absent"},
        }});
        let params = json!({"name": "incident.created", "arguments":
            {"region": "us-west1", "limit": 7, "urgent": true, "plain": "x", "absent": null}});
        let h = routing_headers(&params, Some(&schema));
        assert_eq!(get(&h, "Mcp-Name").as_deref(), Some("incident.created"));
        assert_eq!(get(&h, "Mcp-Param-Region").as_deref(), Some("us-west1"));
        assert_eq!(get(&h, "Mcp-Param-Limit").as_deref(), Some("7"));
        assert_eq!(get(&h, "Mcp-Param-Urgent").as_deref(), Some("true"));
        assert_eq!(get(&h, "Mcp-Param-Absent"), None, "null: no header");
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn nested_properties_are_read_at_their_exact_path() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "object",
            "properties": {"b": {"type": "string", "x-mcp-header": "Deep"}}}}});
        let h = routing_headers(
            &json!({"name": "e", "arguments": {"a": {"b": "v"}}}),
            Some(&schema),
        );
        assert_eq!(get(&h, "Mcp-Param-Deep").as_deref(), Some("v"));
    }

    #[test]
    fn values_that_are_not_header_safe_use_the_base64_sentinel() {
        for (raw, enc) in [
            ("Hello, 世界", "=?base64?SGVsbG8sIOS4lueVjA==?="),
            (" padded ", "=?base64?IHBhZGRlZCA=?="),
            ("line1\nline2", "=?base64?bGluZTEKbGluZTI=?="),
            ("=?base64?literal?=", "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="),
        ] {
            assert_eq!(encode_value(raw), enc, "{raw:?}");
        }
        assert_eq!(encode_value("us-west1"), "us-west1");
        let h = routing_headers(&json!({"name": "événement"}), None);
        assert!(get(&h, "Mcp-Name").unwrap().starts_with("=?base64?"));
    }

    #[test]
    fn annotations_that_break_the_transports_rules_are_rejected() {
        let ok = json!({"properties": {"r": {"type": "string", "x-mcp-header": "Region"}}});
        assert!(annotations_valid(&ok).is_ok());
        for bad in [
            json!({"properties": {"r": {"type": "string", "x-mcp-header": ""}}}),
            json!({"properties": {"r": {"type": "string", "x-mcp-header": "Bad Name"}}}),
            json!({"properties": {"r": {"type": "number", "x-mcp-header": "N"}}}),
            json!({"properties": {"r": {"type": "array", "x-mcp-header": "A"}}}),
            json!({"properties": {
                "a": {"type": "string", "x-mcp-header": "Same"},
                "b": {"type": "string", "x-mcp-header": "SAME"}}}),
            json!({"x-mcp-header": "Root", "properties": {}}),
            json!({"properties": {"l": {"type": "array", "items":
                {"type": "string", "x-mcp-header": "InItems"}}}}),
            json!({"anyOf": [{"properties": {"r": {"type": "string", "x-mcp-header": "R"}}}]}),
        ] {
            assert!(annotations_valid(&bad).is_err(), "{bad}");
        }
    }
}
