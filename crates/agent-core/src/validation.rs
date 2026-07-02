//! Tool-call argument coercion against the tool's own declared JSON Schema.
//!
//! Mirrors pi's `validateToolArguments` (`packages/ai/src/utils/validation.ts`, AJV-backed): some
//! providers (or an OpenAI-compatible proxy in between) stringify primitives a model emitted as a
//! genuinely-typed value — `{"count": "42"}` instead of `{"count": 42}` — which would otherwise fail a
//! tool's own `as_i64()`/`as_bool()` extraction with a confusing "missing field" error rather than
//! running normally. This is a best-effort normalization pass, not a strict validator: on any coercion
//! failure the original, uncoerced value is left untouched (see [`coerce_tool_arguments`]'s doc
//! comment) so a genuinely malformed call still surfaces through each tool's own existing, clearer
//! validation error rather than a new, separate failure path.
//!
//! Coercion rules match AJV's documented `coerceTypes` table exactly (not just "parse the string"):
//! `"42.1"` does NOT coerce to `integer` (fractional), and `"1"`/`"0"` do NOT coerce to `boolean` (only
//! the literal strings `"true"`/`"false"` do) — see this module's tests, ported from pi's own
//! `validation.test.ts` fixture table.

use serde_json::Value;

/// Coerce `input` to match `schema`'s declared type(s), recursing into `object` schemas' `properties`.
/// Returns the coerced value on success, or an error string (unused by callers today beyond falling
/// back to the original value — see the module doc comment) describing which type(s) it couldn't
/// satisfy.
pub fn coerce_tool_arguments(schema: &Value, input: Value) -> Result<Value, String> {
    coerce_value(schema, input)
}

fn declared_types(schema: &Value) -> Option<Vec<&str>> {
    match schema.get("type") {
        Some(Value::String(s)) => Some(vec![s.as_str()]),
        Some(Value::Array(arr)) => {
            let types: Vec<&str> = arr.iter().filter_map(Value::as_str).collect();
            (!types.is_empty()).then_some(types)
        }
        _ => None,
    }
}

fn matches_type_raw(t: &str, v: &Value) -> bool {
    match t {
        "number" => v.is_number(),
        "integer" => v.as_f64().is_some_and(|n| n.fract() == 0.0),
        "boolean" => v.is_boolean(),
        "string" => v.is_string(),
        "null" => v.is_null(),
        "object" => v.is_object(),
        "array" => v.is_array(),
        _ => true, // an unrecognized/unconstrained type name: don't block on it
    }
}

/// Coerce `input` to exactly type `t`, or `None` if AJV's rules don't allow that coercion.
fn try_coerce(t: &str, input: &Value) -> Option<Value> {
    match t {
        "number" | "integer" => {
            let n = match input {
                Value::String(s) => s.trim().parse::<f64>().ok()?,
                Value::Bool(true) => 1.0,
                Value::Bool(false) => 0.0,
                Value::Null => 0.0,
                _ => return None,
            };
            if t == "integer" && n.fract() != 0.0 {
                return None;
            }
            // A whole-number result becomes an integer-tagged `Number` (`i64`), not
            // `Number::from_f64` — the latter always produces a float-tagged JSON number even for a
            // value like `21.0`, which `serde_json::Number::as_i64()` then can't read back (returns
            // `None`), defeating the whole point of coercing `"21"` into something a tool's own
            // `input.get("count").and_then(Value::as_i64)` can actually extract.
            if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
                Some(Value::Number((n as i64).into()))
            } else {
                serde_json::Number::from_f64(n).map(Value::Number)
            }
        }
        "boolean" => match input {
            Value::String(s) if s == "true" => Some(Value::Bool(true)),
            Value::String(s) if s == "false" => Some(Value::Bool(false)),
            Value::Number(n) if *n == serde_json::Number::from(1) => Some(Value::Bool(true)),
            Value::Number(n) if *n == serde_json::Number::from(0) => Some(Value::Bool(false)),
            _ => None,
        },
        "string" => match input {
            Value::Number(n) => Some(Value::String(n.to_string())),
            Value::Bool(b) => Some(Value::String(b.to_string())),
            Value::Null => Some(Value::String(String::new())),
            _ => None,
        },
        "null" => match input {
            Value::String(s) if s.is_empty() => Some(Value::Null),
            Value::Number(n) if *n == serde_json::Number::from(0) => Some(Value::Null),
            Value::Bool(false) => Some(Value::Null),
            _ => None,
        },
        _ => None,
    }
}

/// Recurse into an already-typed-correctly `object` value's `properties`, coercing each present
/// property against its own sub-schema. Missing/extra properties are left alone — presence/`required`
/// enforcement stays each tool's own job (its existing `input.get("x").ok_or_else(...)` pattern), not
/// this pass's.
fn coerce_object_properties(schema: &Value, input: Value) -> Result<Value, String> {
    let Value::Object(mut obj) = input else {
        return Ok(input);
    };
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return Ok(Value::Object(obj));
    };
    for (key, sub_schema) in props {
        if let Some(v) = obj.remove(key) {
            // A property that fails to coerce fails the *whole* call — matching AJV/pi's own
            // all-or-nothing `validateToolArguments` (it throws on the first sub-schema mismatch, it
            // doesn't silently null out just the offending field and let the rest through).
            obj.insert(key.clone(), coerce_value(sub_schema, v)?);
        }
    }
    Ok(Value::Object(obj))
}

fn coerce_value(schema: &Value, input: Value) -> Result<Value, String> {
    let Some(types) = declared_types(schema) else {
        // No `type` constraint at all — an object schema with only `properties` (no explicit
        // `"type":"object"`) still implies object shape, matching pi's own lenient JSON-Schema reading.
        if schema.get("properties").is_some() && input.is_object() {
            return coerce_object_properties(schema, input);
        }
        return Ok(input);
    };

    for t in &types {
        if matches_type_raw(t, &input) {
            return if *t == "object" {
                coerce_object_properties(schema, input)
            } else {
                Ok(input)
            };
        }
    }

    for t in &types {
        if let Some(coerced) = try_coerce(t, &input) {
            return Ok(coerced);
        }
    }

    Err(format!(
        "Validation failed: {input} does not match schema type(s) {types:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Wraps `schema` as `{"type":"object","properties":{"value":schema},"required":["value"]}` and
    /// coerces `{"value": input}` against it — mirrors pi's own `createToolCallWithPlainSchema` test
    /// helper exactly, including exercising the object-properties recursion, not just a bare scalar.
    fn coerce_wrapped(schema: Value, input: Value) -> Result<Value, String> {
        let wrapper = json!({
            "type": "object",
            "properties": { "value": schema },
            "required": ["value"],
        });
        coerce_tool_arguments(&wrapper, json!({ "value": input }))
    }

    #[test]
    fn coerces_serialized_plain_json_schemas_with_ajv_compatible_primitive_rules() {
        // Ported verbatim from pi's `validation.test.ts` "coerces serialized plain JSON schemas..."
        // passing-cases table.
        let cases: Vec<(Value, Value, Value)> = vec![
            // Whole-number results are integer-tagged (`json!(42)`, not `json!(42.0)`) so a caller's
            // `Value::as_i64()` can actually read them back — see `try_coerce`'s doc comment.
            (json!({"type": "number"}), json!("42"), json!(42)),
            (json!({"type": "number"}), json!(true), json!(1)),
            (json!({"type": "number"}), json!(null), json!(0)),
            (json!({"type": "integer"}), json!("42"), json!(42)),
            (json!({"type": "boolean"}), json!("true"), json!(true)),
            (json!({"type": "boolean"}), json!("false"), json!(false)),
            (json!({"type": "boolean"}), json!(1), json!(true)),
            (json!({"type": "boolean"}), json!(0), json!(false)),
            (json!({"type": "string"}), json!(null), json!("")),
            (json!({"type": "string"}), json!(true), json!("true")),
            (json!({"type": "null"}), json!(""), json!(null)),
            (json!({"type": "null"}), json!(0), json!(null)),
            (json!({"type": "null"}), json!(false), json!(null)),
            (
                json!({"type": ["number", "string"]}),
                json!("1"),
                json!("1"),
            ),
            (json!({"type": ["boolean", "number"]}), json!("1"), json!(1)),
        ];
        for (schema, input, expected) in cases {
            let got = coerce_wrapped(schema.clone(), input.clone()).unwrap_or_else(|e| {
                panic!("schema={schema} input={input}: expected Ok, got Err({e})")
            });
            assert_eq!(
                got,
                json!({ "value": expected }),
                "schema={schema} input={input}"
            );
        }
    }

    #[test]
    fn rejects_invalid_coercions_for_serialized_plain_json_schemas() {
        // Ported verbatim from pi's `validation.test.ts` "rejects invalid coercions..." failing-cases
        // table — the whole point of matching AJV's exact rules rather than a loose "just try to
        // parse it" coercion.
        let cases: Vec<(Value, Value)> = vec![
            (json!({"type": "boolean"}), json!("1")),
            (json!({"type": "boolean"}), json!("0")),
            (json!({"type": "null"}), json!("null")),
            (json!({"type": "integer"}), json!("42.1")),
        ];
        for (schema, input) in cases {
            assert!(
                coerce_wrapped(schema.clone(), input.clone()).is_err(),
                "schema={schema} input={input}: expected coercion to fail"
            );
        }
    }

    #[test]
    fn a_value_already_matching_the_declared_type_passes_through_unchanged() {
        assert_eq!(
            coerce_tool_arguments(&json!({"type": "number"}), json!(42.5)).unwrap(),
            json!(42.5)
        );
        assert_eq!(
            coerce_tool_arguments(&json!({"type": "string"}), json!("hi")).unwrap(),
            json!("hi")
        );
    }

    #[test]
    fn a_schema_with_no_type_constraint_passes_through_unchanged() {
        assert_eq!(
            coerce_tool_arguments(&json!({}), json!({"anything": "goes"})).unwrap(),
            json!({"anything": "goes"})
        );
    }

    #[test]
    fn recurses_into_nested_object_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "label": {"type": "string"},
            },
        });
        let input = json!({"count": "7", "label": 42});
        assert_eq!(
            coerce_tool_arguments(&schema, input).unwrap(),
            json!({"count": 7, "label": "42"})
        );
    }
}
