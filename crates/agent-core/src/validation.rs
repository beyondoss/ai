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
//!
//! Recursion follows pi's `coerceWithJsonSchema` into every shape a tool schema can nest a sub-schema
//! in, not just an object's declared `properties`: array `items` (both the tuple form, one schema per
//! index, and the single-schema-for-every-element form), `additionalProperties` when it's itself a
//! schema (not a bare `true`/`false`), and `allOf`/`anyOf`/`oneOf` composition. `allOf`/`anyOf`/`oneOf`
//! are best-effort the way pi's own composition handling is (never fails the surrounding value over a
//! member mismatch — see [`apply_composition`]); everywhere else in this module, a sub-schema that
//! can't be coerced fails the whole call, matching AJV's own all-or-nothing `coerceTypes`.
//!
//! That recursion runs against a schema this process does not necessarily author — an MCP tool's schema
//! is the remote server's verbatim advertisement — so the whole pass runs on a fixed work budget and
//! degrades to a no-op rather than spinning when it runs out. See [`COERCION_NODE_BUDGET`].

use serde_json::Value;

/// How many schema nodes one `coerce_tool_arguments` call may visit before it stops coercing and hands
/// the rest of the value back as-is.
///
/// This pass runs on *every* tool dispatch against the tool's own `input_schema()`, and for an MCP tool
/// that schema is whatever the remote server advertised — verbatim, and therefore attacker-controlled if
/// that server is hostile or compromised. Composition is what makes an un-budgeted pass dangerous:
/// `allOf`/`anyOf`/`oneOf` visit *every* member with a full deep clone of the value, and each member
/// re-enters the recursion, so the pass's cost is (schema nodes visited) × (size of the value being
/// cloned at each one) — a hostile schema of a few hundred KB (thousands of composition members, nothing
/// exotic) can therefore buy itself thousands of deep clones of the whole argument value and pin the
/// dispatching task. Recursion *depth* is already safe — serde_json refuses to parse past 128 levels of
/// nesting — so this is a work blowup, not a stack overflow, and a cap on nodes visited is the smallest
/// thing that bounds it. It bounds every other shape of pathological schema at the same time.
///
/// 10k is orders of magnitude above any real tool schema (the largest here visit a few dozen nodes), so
/// a legitimate call can never reach it, while a hostile one is capped at ~10k clones of a model-emitted
/// argument value — milliseconds, not minutes.
const COERCION_NODE_BUDGET: u32 = 10_000;

/// Remaining nodes this coercion pass may visit. Exhaustion is not an error: coercion is a best-effort
/// convenience (see the module doc comment), so running out just means the value stops being normalized
/// and is handed back untouched — exactly what a value with no coercible schema at all would get. The
/// tool's own validation still runs afterwards and still produces its own clear error if the arguments
/// really are wrong.
struct Budget(u32);

impl Budget {
    /// Charge one visited node. `false` once the budget is gone — the caller must then stop recursing
    /// and return its value as-is.
    fn spend(&mut self) -> bool {
        self.0 = self.0.saturating_sub(1);
        self.0 > 0
    }
}

/// Coerce `input` to match `schema`'s declared type(s), recursing into `object` schemas' `properties`.
/// Returns the coerced value on success, or an error string (unused by callers today beyond falling
/// back to the original value — see the module doc comment) describing which type(s) it couldn't
/// satisfy.
///
/// Bounded work: at most [`COERCION_NODE_BUDGET`] schema nodes are visited, after which the remaining
/// value is returned un-coerced rather than coerced.
pub fn coerce_tool_arguments(schema: &Value, input: Value) -> Result<Value, String> {
    coerce_value(schema, input, &mut Budget(COERCION_NODE_BUDGET))
}

/// Coerce with an explicit budget, reporting what was left of it — the only way to observe from a test
/// that the pass really did stop early rather than merely finishing fast.
#[cfg(test)]
fn coerce_with_budget(schema: &Value, input: Value, budget: u32) -> (Result<Value, String>, u32) {
    let mut budget = Budget(budget);
    let result = coerce_value(schema, input, &mut budget);
    (result, budget.0)
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

/// Recurse into an already-typed-correctly `object` value's `properties` (and, when it's itself a
/// schema rather than a bare `true`/`false`, `additionalProperties` for every key `properties` didn't
/// already claim), coercing each present property against its own sub-schema. Missing properties, and
/// extra ones with no `additionalProperties` schema to coerce against, are left alone —
/// presence/`required` enforcement stays each tool's own job (its existing
/// `input.get("x").ok_or_else(...)` pattern), not this pass's.
fn coerce_object_properties(
    schema: &Value,
    input: Value,
    budget: &mut Budget,
) -> Result<Value, String> {
    let Value::Object(mut obj) = input else {
        return Ok(input);
    };
    let props = schema.get("properties").and_then(Value::as_object);
    if let Some(props) = props {
        for (key, sub_schema) in props {
            // `get_mut` + `Value::take` coerces each present property in place: one map lookup, no
            // key-`String` clone, and no remove/re-insert churn — this runs on every tool dispatch.
            // `take()` leaves a transient `Null` in the slot that the assignment immediately
            // overwrites; on the `?` error path the whole object is discarded anyway, matching the
            // original all-or-nothing behavior (a property that fails to coerce fails the *whole*
            // call — as AJV/pi's `validateToolArguments` throws on the first sub-schema mismatch).
            if let Some(slot) = obj.get_mut(key) {
                *slot = coerce_value(sub_schema, slot.take(), budget)?;
            }
        }
    }
    if let Some(additional_schema) = schema.get("additionalProperties").filter(|v| v.is_object()) {
        // Same in-place take semantics for every key `properties` didn't already claim — no throwaway
        // `Vec<String>` of cloned keys, no remove/insert. Iteration order (sorted, as serde_json's
        // `Map` is a `BTreeMap` here) is unchanged from the original remove-then-insert.
        for (key, slot) in obj.iter_mut() {
            if props.is_some_and(|p| p.contains_key(key.as_str())) {
                continue;
            }
            *slot = coerce_value(additional_schema, slot.take(), budget)?;
        }
    }
    Ok(Value::Object(obj))
}

/// Recurse into an already-array-typed value's `items`: either a single schema applied to every
/// element (`"items": {...}`), or JSON Schema's positional "tuple validation" form (`"items": [...]`,
/// one schema per index — an index past the tuple list's own length is left alone, same
/// missing/extra-is-not-this-pass's-job rationale as `coerce_object_properties`).
fn coerce_array_items(schema: &Value, input: Value, budget: &mut Budget) -> Result<Value, String> {
    let Value::Array(mut items) = input else {
        return Ok(input);
    };
    match schema.get("items") {
        Some(Value::Array(item_schemas)) => {
            for (item, item_schema) in items.iter_mut().zip(item_schemas) {
                *item = coerce_value(item_schema, item.take(), budget)?;
            }
        }
        Some(item_schema) if item_schema.is_object() => {
            for item in items.iter_mut() {
                *item = coerce_value(item_schema, item.take(), budget)?;
            }
        }
        _ => {}
    }
    Ok(Value::Array(items))
}

/// Apply `allOf`/`anyOf`/`oneOf` composition, if present, before the schema's own `type`-driven
/// coercion runs. Matches pi's `coerceWithJsonSchema`/`coerceWithUnionSchema`: composition is always
/// best-effort and never fails the surrounding value over a member mismatch — an `allOf` member that
/// can't be coerced is skipped rather than aborting its siblings, and `anyOf`/`oneOf` take the first
/// member that *does* coerce, falling back to the value untouched if none do. Real, non-composition
/// type mismatches are still caught downstream by `coerce_value`'s own `type` handling once composition
/// hands its result off.
///
/// This is the expensive corner of the pass — every `allOf` member is visited with a full deep clone of
/// the value, and each one re-enters the recursion — so it is also where `budget` earns its keep: see
/// [`COERCION_NODE_BUDGET`]. Once the budget is out the members are left unapplied and the value passes
/// through, which is indistinguishable from a composition whose members simply didn't coerce anything.
fn apply_composition(schema: &Value, mut value: Value, budget: &mut Budget) -> Value {
    if let Some(members) = schema.get("allOf").and_then(Value::as_array) {
        for member in members {
            if let Ok(coerced) = coerce_value(member, value.clone(), budget) {
                value = coerced;
            }
        }
    }
    if let Some(members) = schema.get("anyOf").and_then(Value::as_array) {
        value = coerce_union(members, value, budget);
    }
    if let Some(members) = schema.get("oneOf").and_then(Value::as_array) {
        value = coerce_union(members, value, budget);
    }
    value
}

/// Try each union member in turn, keeping the first one that coerces cleanly; if none do, the value is
/// handed back untouched rather than treated as a failure — same rationale as `apply_composition`'s own
/// doc comment.
fn coerce_union(members: &[Value], value: Value, budget: &mut Budget) -> Value {
    for member in members {
        if let Ok(coerced) = coerce_value(member, value.clone(), budget) {
            return coerced;
        }
    }
    value
}

/// A JSON number that arrives float-tagged (e.g. parsed from the wire as `5000.0`, distinct in
/// `serde_json` from an int-tagged `5000`) but holds a whole-number value is re-tagged here as a
/// proper int-tagged `Value::Number` — the same normalization [`try_coerce`] already applies to a
/// parsed string (see its doc comment). Without this, `matches_type_raw`'s `"integer"` check happily
/// accepts a float-tagged whole number (`v.as_f64().is_some_and(|n| n.fract() == 0.0)`), but this
/// function's caller used to return it completely unchanged — still float-tagged — so a tool's own
/// `.as_u64()`/`.as_i64()` extraction returns `None` regardless of the value's fractional part,
/// either hard-erroring (`bash`'s `timeout_ms`) or silently falling back to a default with no signal
/// to the model (`read`'s `offset`, `grep`/`find`/`ls`'s `limit`). A non-whole-number `Value::Number`
/// (already a bare "number" match, e.g. `42.5`) is returned untouched either way.
fn retag_whole_number(value: Value) -> Value {
    let Value::Number(n) = &value else {
        return value;
    };
    // Already int-tagged — nothing to do (also covers the "number" schema branch matching an
    // already-correct integer).
    if n.is_i64() || n.is_u64() {
        return value;
    }
    match n.as_f64() {
        Some(f) if f.fract() == 0.0 && f.abs() < i64::MAX as f64 => {
            Value::Number((f as i64).into())
        }
        _ => value,
    }
}

fn coerce_value(schema: &Value, input: Value, budget: &mut Budget) -> Result<Value, String> {
    // Every re-entry into the recursion — a property, an array element, an `allOf`/`anyOf` member —
    // passes through here, so charging the budget at this single point bounds the whole pass. Bailing
    // returns the value un-coerced rather than an error: a hostile schema must degrade this pass into a
    // no-op, never into a failed dispatch.
    if !budget.spend() {
        return Ok(input);
    }
    let value = apply_composition(schema, input, budget);

    let Some(types) = declared_types(schema) else {
        // No `type` constraint at all — an object schema with only `properties`/`additionalProperties`
        // (no explicit `"type":"object"`), or an array schema with only `items`, still implies that
        // shape, matching pi's own lenient JSON-Schema reading.
        if value.is_object()
            && (schema.get("properties").is_some()
                || schema
                    .get("additionalProperties")
                    .is_some_and(Value::is_object))
        {
            return coerce_object_properties(schema, value, budget);
        }
        if value.is_array() && schema.get("items").is_some() {
            return coerce_array_items(schema, value, budget);
        }
        return Ok(value);
    };

    for t in &types {
        if matches_type_raw(t, &value) {
            return match *t {
                "object" => coerce_object_properties(schema, value, budget),
                "array" => coerce_array_items(schema, value, budget),
                "integer" | "number" => Ok(retag_whole_number(value)),
                _ => Ok(value),
            };
        }
    }

    for t in &types {
        if let Some(coerced) = try_coerce(t, &value) {
            return Ok(coerced);
        }
    }

    Err(format!(
        "Validation failed: {value} does not match schema type(s) {types:?}"
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

    #[test]
    fn recurses_into_array_items_sharing_a_single_schema() {
        let schema = json!({"type": "array", "items": {"type": "integer"}});
        assert_eq!(
            coerce_tool_arguments(&schema, json!(["1", "2", "3"])).unwrap(),
            json!([1, 2, 3])
        );
    }

    #[test]
    fn recurses_into_tuple_form_array_items() {
        let schema = json!({
            "type": "array",
            "items": [{"type": "integer"}, {"type": "string"}],
        });
        assert_eq!(
            coerce_tool_arguments(&schema, json!(["1", 2])).unwrap(),
            json!([1, "2"])
        );
    }

    #[test]
    fn recurses_into_array_of_objects_coercing_each_elements_nested_properties() {
        // pi-parity: mirrors `edit`'s real `edits` schema (`crates/agent/src/tools/edit.rs`) — a model
        // emitting a number for `old_string` inside the array used to sail through uncoerced because
        // `coerce_value` never recursed past a bare `properties` object into an array's `items`.
        let schema = json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["old_string", "new_string"],
            },
        });
        let input = json!([{"old_string": 42, "new_string": "bar"}]);
        assert_eq!(
            coerce_tool_arguments(&schema, input).unwrap(),
            json!([{"old_string": "42", "new_string": "bar"}])
        );
    }

    #[test]
    fn recurses_through_all_of_members_in_order() {
        let schema = json!({"allOf": [{"type": "integer"}]});
        assert_eq!(
            coerce_tool_arguments(&schema, json!("5")).unwrap(),
            json!(5)
        );
    }

    #[test]
    fn any_of_picks_the_first_member_schema_that_coerces_cleanly() {
        let schema = json!({"anyOf": [{"type": "boolean"}, {"type": "integer"}]});
        assert_eq!(
            coerce_tool_arguments(&schema, json!("42")).unwrap(),
            json!(42)
        );
    }

    #[test]
    fn any_of_leaves_the_value_untouched_when_no_member_coerces() {
        // Matches pi's `coerceWithUnionSchema`: a union with no coercible member is left as-is rather
        // than treated as a hard failure — that decision belongs to the tool's own downstream
        // validation, not this best-effort pass.
        let schema = json!({"anyOf": [{"type": "boolean"}, {"type": "null"}]});
        assert_eq!(
            coerce_tool_arguments(&schema, json!("hello")).unwrap(),
            json!("hello")
        );
    }

    #[test]
    fn recurses_into_schema_typed_additional_properties() {
        let schema = json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        });
        let input = json!({"id": 7, "extra": "42"});
        assert_eq!(
            coerce_tool_arguments(&schema, input).unwrap(),
            json!({"id": "7", "extra": 42})
        );
    }

    #[test]
    fn a_float_tagged_whole_number_coerces_to_an_int_tagged_value_for_an_integer_schema() {
        // pi-parity bug: a JSON number that arrives float-tagged with a whole-number value (distinct
        // in `serde_json` from an int-tagged number of the same value) used to pass
        // `matches_type_raw`'s `"integer"` check but then sail through `coerce_value` completely
        // unchanged — still float-tagged — so a tool's own `.as_u64()`/`.as_i64()` extraction
        // returned `None` regardless of the value's fractional part. `serde_json::from_str` is used
        // here (rather than a bare Rust `5000.0` literal) to guarantee a genuinely float-tagged
        // `Value::Number` — `serde_json::Number`'s internal representation is float-tagged whenever a
        // number arrives with a decimal point on the wire, not by the value's own fractional-ness.
        let float_tagged: Value = serde_json::from_str("5000.0").unwrap();
        assert!(
            float_tagged.as_i64().is_none() && float_tagged.as_u64().is_none(),
            "test fixture must be genuinely float-tagged to exercise the bug: {float_tagged:?}"
        );
        let got = coerce_tool_arguments(&json!({"type": "integer"}), float_tagged).unwrap();
        assert_eq!(got.as_u64(), Some(5000), "got {got:?}");

        // Same bug, same fix, for a bare "number" schema (not just "integer").
        let float_tagged: Value = serde_json::from_str("5000.0").unwrap();
        let got = coerce_tool_arguments(&json!({"type": "number"}), float_tagged).unwrap();
        assert_eq!(got.as_u64(), Some(5000), "got {got:?}");

        // A non-whole-number float must still pass through untouched (not force-truncated).
        let non_whole: Value = serde_json::from_str("42.5").unwrap();
        assert_eq!(
            coerce_tool_arguments(&json!({"type": "number"}), non_whole.clone()).unwrap(),
            non_whole
        );
    }

    #[test]
    fn a_pathological_nested_all_of_schema_stops_at_the_budget_instead_of_blowing_up() {
        // The shape that turns this pass into a denial of service: every level is a two-member `allOf`
        // whose members nest the level below, so the number of nodes the coercion visits doubles with
        // each level — and each of those visits deep-clones the value being coerced.
        //
        // 15 levels (~65k nodes) rather than the ~25 a real attacker would send: a JSON Schema is a
        // *tree*, with no `$ref` sharing here, so the schema document itself doubles right along with
        // the visit count and a depth-25 fixture is 30M+ nodes — it OOMs the test process while it's
        // still being *built*, before any coercion runs. Depth is capped here only to keep the fixture
        // buildable; what's being pinned down is that the coercion stops at its budget no matter how far
        // the schema goes, which the exhausted-budget assertion below shows directly (a wall-clock bound
        // alone would prove nothing at a depth this small).
        let mut schema = json!({"type": "string"});
        let mut input = json!("innermost");
        for _ in 0..15 {
            let member = json!({"type": "object", "properties": {"p": schema}});
            schema = json!({"allOf": [member.clone(), member]});
            input = json!({"p": input});
        }

        let started = std::time::Instant::now();
        let (got, remaining) = coerce_with_budget(&schema, input.clone(), COERCION_NODE_BUDGET);
        let elapsed = started.elapsed();

        assert_eq!(
            remaining, 0,
            "the fixture must actually exhaust the budget, or it isn't testing the bail-out at all"
        );
        // Bailing out is a no-op, not an error: the value comes back intact (every coercion this schema
        // asks for is an identity one) for the tool's own validation to judge.
        assert_eq!(
            got,
            Ok(input),
            "an exhausted budget must hand the value back un-coerced, never fail the dispatch"
        );
        // A loose sanity bound, not a benchmark — it must not flake on a loaded CI box.
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "coercion of a hostile schema must be bounded work, took {elapsed:?}"
        );
    }

    #[test]
    fn an_exhausted_budget_degrades_to_a_no_op_rather_than_an_error() {
        // The bail-out has to be indistinguishable from "this schema had nothing to coerce": a hostile
        // schema must be able to turn this pass off, never to turn a legitimate tool call into a failed
        // dispatch. A budget of 1 is spent by the first node, so nothing is coerced.
        let (got, remaining) = coerce_with_budget(&json!({"type": "integer"}), json!("42"), 1);
        assert_eq!(got, Ok(json!("42")));
        assert_eq!(remaining, 0);

        // Same schema, same input, with budget to spare: coercion happens as normal.
        let (got, _) = coerce_with_budget(&json!({"type": "integer"}), json!("42"), 2);
        assert_eq!(got, Ok(json!(42)));
    }

    #[test]
    fn boolean_additional_properties_is_not_treated_as_a_schema() {
        let schema = json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "additionalProperties": true,
        });
        let input = json!({"id": 7, "extra": "unchanged"});
        assert_eq!(
            coerce_tool_arguments(&schema, input).unwrap(),
            json!({"id": "7", "extra": "unchanged"})
        );
    }
}
