//! `structured_output` — the agent as a callable function.
//!
//! Registered only when a caller supplies a JSON Schema (`run --output-schema`, `serve`'s
//! `prompt {output_schema}`, or a `subagent` task's own `output_schema`). The tool advertises that
//! schema *as its own input schema*, so the model fills it in exactly the way it fills in any other
//! tool call, and the payload it produces is validated against the schema before anything downstream
//! sees it. On success the run ends ([`ToolOutput::terminate`]) and the value lands in an
//! [`OutputSlot`] the host reads once the run drains.
//!
//! **The slot is host-owned, not a field on the tool.** `serve` rebuilds its whole registry on every
//! `set_model`/`set_thinking`, so a value captured into the tool itself would vanish on a mid-run model
//! switch.
//!
//! **A schema can be model-authored** — a parent agent writes its child's `output_schema` — so nothing
//! here may panic on a hostile or malformed one. The schema is compiled once at construction and a
//! compile failure is returned to the caller (surfacing to a parent as an error `tool_result` it can
//! retry with a corrected schema), never at first call, so the model is never shown a tool whose schema
//! can't be satisfied. `jsonschema` is built with `default-features = false`, which disables
//! `$ref` resolution over HTTP and the filesystem: a schema must not be able to make the agent fetch a
//! URL or read `/etc/passwd`.
//!
//! There is no `tool_choice` forcing. It is per-request, so pinning it would stop the model doing any
//! real work before it answers, and the OpenAI Chat Completions dialect ignores it entirely. The
//! contract is carried by the system prompt plus `terminate`; a run that ends without a payload is
//! reported as a failure by its host rather than papered over.

use std::sync::{Arc, Mutex};

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::Value;

pub const NAME: &str = "structured_output";

/// How many validation failures to quote back at the model. Enough to fix a call in one retry; not so
/// many that a wrong-shaped payload floods the context.
const MAX_REPORTED_ERRORS: usize = 5;

const DEFAULT_DESCRIPTION: &str = "Return the final result of the task, conforming to this schema. \
     Call this exactly once, when the work is complete. The run ends after this call.";

/// Where the validated payload lands. Cloned into the tool at registration and kept by the host, which
/// reads it once `run_events` has drained — the return channel for a run that has no typed result of
/// its own (`Agent::run`/`run_events` both return `Result<()>`).
#[derive(Clone, Default)]
pub struct OutputSlot(Arc<Mutex<Option<Value>>>);

impl OutputSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// The captured value, leaving the slot empty. What a host calls after a run.
    pub fn take(&self) -> Option<Value> {
        self.lock().take()
    }

    /// The captured value, leaving it in place.
    pub fn get(&self) -> Option<Value> {
        self.lock().clone()
    }

    /// Drop any captured value — a host calls this before each new run so a prior prompt's answer can't
    /// be mistaken for this one's.
    pub fn clear(&self) {
        *self.lock() = None;
    }

    fn set(&self, value: Value) {
        *self.lock() = Some(value);
    }

    /// A poisoned lock means some other thread panicked while holding it; the `Option<Value>` inside is
    /// still structurally sound, and workspace lints forbid `unwrap` here anyway.
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Value>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct StructuredOutput {
    /// The caller's schema, returned verbatim as this tool's `input_schema`.
    schema: Value,
    /// Compiled once at construction — never per call, and never after the model has already been shown
    /// the tool.
    validator: jsonschema::Validator,
    description: String,
    slot: OutputSlot,
}

impl StructuredOutput {
    /// Compile `schema` and bind the tool to `slot`.
    ///
    /// `Err` carries a message meant to be read by whoever wrote the schema — an operator on the CLI, or
    /// a parent model that authored a child's `output_schema` and needs to correct it.
    pub fn new(
        schema: Value,
        description: Option<String>,
        slot: OutputSlot,
    ) -> Result<Self, String> {
        // A tool's arguments are always a JSON object on the wire, so a schema that isn't an object
        // schema can never be satisfied. `jsonschema` itself accepts a bare `true`/`false` (valid JSON
        // Schema, matching everything or nothing), which would compile happily and then advertise a
        // meaningless tool — so this is checked here, not left to the compiler below.
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(format!(
                "an output schema must be a JSON Schema object with `\"type\": \"object\"` at its root \
                 (a tool's arguments are always an object); got: {}",
                preview(&schema)
            ));
        }
        let validator = jsonschema::validator_for(&schema)
            .map_err(|e| format!("invalid output schema: {e}"))?;
        Ok(Self {
            schema,
            validator,
            description: description
                .filter(|d| !d.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_DESCRIPTION.to_string()),
            slot,
        })
    }

    /// Every way `value` fails `schema`, capped and rendered for the model to act on. Empty when valid.
    fn validation_errors(&self, value: &Value) -> Vec<String> {
        self.validator
            .iter_errors(value)
            .take(MAX_REPORTED_ERRORS)
            .map(|e| {
                let path = e.instance_path().to_string();
                if path.is_empty() {
                    e.to_string()
                } else {
                    format!("{path}: {e}")
                }
            })
            .collect()
    }
}

/// Parse a `--output-schema` argument: either an inline JSON object, or a path to a file containing
/// one. Distinguished by shape, not by a flag — a schema is always a JSON object, so a value whose
/// first non-whitespace byte is `{` can only be inline, and anything else can only be a path.
///
/// `Err` is meant to be printed to an operator, who is the only one who can pass this.
pub fn load_schema(arg: &str) -> Result<Value, String> {
    let trimmed = arg.trim();
    if trimmed.starts_with('{') {
        return serde_json::from_str(trimmed)
            .map_err(|e| format!("--output-schema is not valid JSON: {e}"));
    }
    let body = std::fs::read_to_string(trimmed)
        .map_err(|e| format!("--output-schema: cannot read {trimmed}: {e}"))?;
    serde_json::from_str(&body)
        .map_err(|e| format!("--output-schema: {trimmed} is not valid JSON: {e}"))
}

/// A short, safe rendering of an arbitrary (possibly model-authored, possibly huge) value for an error
/// message.
fn preview(value: &Value) -> String {
    let s = value.to_string();
    if s.chars().count() <= 120 {
        return s;
    }
    let kept: String = s.chars().take(120).collect();
    format!("{kept}…")
}

#[async_trait]
impl Tool for StructuredOutput {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn run(&self, mut input: Value) -> Result<ToolOutput, ToolError> {
        // The dispatch loop stamps `_model_supports_vision` onto *every* tool call's input. It is
        // schema-undocumented and every other tool simply ignores it — but this tool validates its input
        // against a caller-supplied schema, and `"additionalProperties": false` (the right thing for a
        // schema describing an exact payload) would otherwise reject every single call. Stripped before
        // validation *and* before capture, so the value a host reads is exactly what the model wrote.
        if let Some(obj) = input.as_object_mut() {
            obj.remove(agent_core::tool::MODEL_SUPPORTS_VISION_KEY);
        }

        let errors = self.validation_errors(&input);
        if !errors.is_empty() {
            // An error `tool_result`, not a terminated run: the loop hands the reason back and the model
            // retries with a corrected payload. `terminate` is never set on the error path.
            return Err(ToolError::InvalidInput(format!(
                "the result does not match the required schema:\n{}",
                errors.join("\n")
            )));
        }
        // Last write wins. A model that calls this, does more work, then calls it again with a revised
        // answer should not be pinned to the stale one.
        self.slot.set(input.clone());
        let rendered = serde_json::to_string_pretty(&input).unwrap_or_else(|_| input.to_string());
        Ok(ToolOutput::text(rendered).with_terminate(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "files_changed": { "type": "array", "items": { "type": "string" } },
                "summary": { "type": "string" }
            },
            "required": ["files_changed", "summary"],
            "additionalProperties": false
        })
    }

    fn tool() -> (StructuredOutput, OutputSlot) {
        let slot = OutputSlot::new();
        let tool = StructuredOutput::new(schema(), None, slot.clone()).expect("valid schema");
        (tool, slot)
    }

    #[test]
    fn the_advertised_input_schema_is_the_callers_schema_verbatim() {
        let (tool, _) = tool();
        assert_eq!(tool.input_schema(), schema());
    }

    #[test]
    fn a_default_description_is_used_when_none_is_given() {
        let (tool, _) = tool();
        assert_eq!(tool.description(), DEFAULT_DESCRIPTION);
        let blank = StructuredOutput::new(schema(), Some("   ".into()), OutputSlot::new()).unwrap();
        assert_eq!(blank.description(), DEFAULT_DESCRIPTION);
        let custom =
            StructuredOutput::new(schema(), Some("Return the diff.".into()), OutputSlot::new())
                .unwrap();
        assert_eq!(custom.description(), "Return the diff.");
    }

    #[tokio::test]
    async fn a_valid_payload_is_captured_and_terminates_the_run() {
        let (tool, slot) = tool();
        let value = json!({ "files_changed": ["a.rs"], "summary": "did the thing" });
        let out = tool.run(value.clone()).await.unwrap();
        assert!(out.terminate, "a completed result must end the run");
        assert_eq!(slot.get(), Some(value));
        // The model sees the payload it just produced, pretty-printed.
        assert!(out.text.contains("\"files_changed\""));
    }

    #[tokio::test]
    async fn a_schema_violation_is_invalid_input_and_never_terminates() {
        let (tool, slot) = tool();
        let err = tool
            .run(json!({ "files_changed": "not an array", "summary": "x" }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(
                    msg.contains("does not match the required schema"),
                    "got: {msg}"
                );
                assert!(
                    msg.contains("files_changed"),
                    "the model must be told where: {msg}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        assert_eq!(slot.get(), None, "an invalid payload must not be captured");
    }

    #[tokio::test]
    async fn a_missing_required_field_is_reported_with_its_name() {
        let (tool, _) = tool();
        let err = tool.run(json!({ "summary": "x" })).await.unwrap_err();
        let ToolError::InvalidInput(msg) = err else {
            panic!("expected InvalidInput");
        };
        assert!(msg.contains("files_changed"), "got: {msg}");
    }

    #[tokio::test]
    async fn reported_errors_are_capped() {
        let many = json!({
            "type": "object",
            "properties": { "a": {"type":"integer"}, "b": {"type":"integer"}, "c": {"type":"integer"},
                            "d": {"type":"integer"}, "e": {"type":"integer"}, "f": {"type":"integer"},
                            "g": {"type":"integer"} },
            "required": ["a","b","c","d","e","f","g"]
        });
        let tool = StructuredOutput::new(many, None, OutputSlot::new()).unwrap();
        let ToolError::InvalidInput(msg) = tool.run(json!({})).await.unwrap_err() else {
            panic!("expected InvalidInput");
        };
        assert_eq!(
            msg.lines().count(),
            1 + MAX_REPORTED_ERRORS,
            "header plus at most {MAX_REPORTED_ERRORS} errors: {msg}"
        );
    }

    #[tokio::test]
    async fn the_last_valid_call_wins() {
        // A model that revises its answer after doing more work must not be pinned to the stale one.
        let (tool, slot) = tool();
        tool.run(json!({ "files_changed": [], "summary": "first" }))
            .await
            .unwrap();
        tool.run(json!({ "files_changed": ["b.rs"], "summary": "second" }))
            .await
            .unwrap();
        assert_eq!(slot.get().unwrap()["summary"], "second");
    }

    #[test]
    fn take_empties_the_slot_and_clear_discards_a_stale_answer() {
        let slot = OutputSlot::new();
        slot.set(json!({ "a": 1 }));
        assert_eq!(slot.take(), Some(json!({ "a": 1 })));
        assert_eq!(slot.take(), None);

        slot.set(json!({ "b": 2 }));
        slot.clear();
        assert_eq!(slot.get(), None);
    }

    // ---- schema rejection: these arrive from a *model* (a parent writing a child's schema) ---------

    /// The rejection message for a schema that must not compile. (`StructuredOutput` holds a
    /// `jsonschema::Validator`, which isn't `Debug`, so `expect_err` isn't available here.)
    fn reject(schema: Value) -> String {
        match StructuredOutput::new(schema, None, OutputSlot::new()) {
            Err(msg) => msg,
            Ok(_) => panic!("schema must be rejected"),
        }
    }

    #[test]
    fn a_non_object_schema_is_rejected_rather_than_advertised() {
        // A tool's arguments are always an object on the wire.
        assert!(reject(json!({ "type": "array" })).contains("\"type\": \"object\""));
        assert!(reject(json!("garbage")).contains("\"type\": \"object\""));
        assert!(reject(json!(true)).contains("\"type\": \"object\""));
        assert!(reject(json!({})).contains("\"type\": \"object\""));
    }

    #[test]
    fn a_structurally_invalid_schema_is_rejected_without_panicking() {
        let msg = reject(json!({ "type": "object", "properties": { "a": { "type": 123 } } }));
        assert!(msg.contains("invalid output schema"), "got: {msg}");
    }

    #[test]
    fn a_schema_with_a_remote_ref_is_rejected() {
        // `default-features = false` disables `$ref` resolution, so a model-authored schema cannot make
        // the agent fetch a URL or read a local file. It fails at compile time, not silently at runtime.
        let http = reject(
            json!({ "type": "object", "properties": { "a": { "$ref": "https://example.com/s.json" } } }),
        );
        assert!(http.contains("invalid output schema"), "got: {http}");
        let file = reject(
            json!({ "type": "object", "properties": { "a": { "$ref": "file:///etc/passwd" } } }),
        );
        assert!(file.contains("invalid output schema"), "got: {file}");
    }

    #[test]
    fn a_rejection_message_does_not_echo_a_huge_schema_back() {
        let big = json!({ "type": "array", "items": { "enum": (0..500).collect::<Vec<_>>() } });
        let msg = reject(big);
        assert!(
            msg.chars().count() < 400,
            "message was {} chars",
            msg.chars().count()
        );
        assert!(msg.contains('…'));
    }

    // ---- the contract against the real agent loop -------------------------------------------------
    //
    // `terminate` had no production consumer before this tool, so these pin the behavior the whole
    // feature rests on: a valid payload ends the run, an invalid one does not, and a mixed batch keeps
    // going. They drive the real `Agent` over a scripted transport rather than asserting on the fold in
    // isolation.

    use agent_core::mock::{MockTransport, turn};
    use agent_core::{Agent, AgentEvent, Session, ToolRegistry};
    use std::sync::Arc;

    /// A no-op tool for exercising a batch where not every call wants to terminate.
    struct Noop;
    #[async_trait]
    impl Tool for Noop {
        fn name(&self) -> &str {
            "noop"
        }
        fn description(&self) -> &str {
            "does nothing"
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn run(&self, _: Value) -> Result<ToolOutput, ToolError> {
            Ok("ok".into())
        }
    }

    /// An `Agent` over `turns`, with the real `structured_output` tool registered.
    fn agent_with(
        turns: Vec<Vec<agent_core::StreamEvent>>,
    ) -> (Agent, Arc<MockTransport>, OutputSlot) {
        let slot = OutputSlot::new();
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(
            StructuredOutput::new(schema(), None, slot.clone()).expect("valid schema"),
        ));
        tools.register(Arc::new(Noop));
        let mock = Arc::new(MockTransport::new(turns));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        (agent, mock, slot)
    }

    const VALID: &str = r#"{"files_changed":["a.rs"],"summary":"done"}"#;

    #[tokio::test]
    async fn a_valid_call_ends_the_run_after_one_model_turn() {
        // Only one turn is scripted: if the loop asked for a second, the mock would be exhausted.
        let (agent, mock, slot) = agent_with(vec![turn::tool_call("tu_1", NAME, VALID)]);
        let mut session = Session::new();
        session.user("do it");

        let mut ended = 0;
        agent
            .run_events(&mut session, |e| {
                if matches!(e, AgentEvent::AgentEnd { .. }) {
                    ended += 1;
                }
            })
            .await
            .unwrap();

        assert_eq!(
            mock.calls(),
            1,
            "the run must not ask the model for a wrap-up turn"
        );
        assert_eq!(ended, 1, "AgentEnd must fire");
        assert_eq!(slot.take().unwrap()["summary"], "done");
    }

    #[tokio::test]
    async fn an_invalid_payload_does_not_terminate_and_the_model_retries() {
        // The error path can never set `terminate` — the loop feeds the reason back as an error
        // `tool_result` and the run continues, which is the entire retry mechanism.
        let (agent, mock, slot) = agent_with(vec![
            turn::tool_call("tu_1", NAME, r#"{"summary":"missing the array"}"#),
            turn::tool_call("tu_2", NAME, VALID),
        ]);
        let mut session = Session::new();
        session.user("do it");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(
            mock.calls(),
            2,
            "the run must continue so the model can correct itself"
        );
        assert_eq!(slot.take().unwrap()["summary"], "done");

        // The model was told exactly what was wrong.
        let complaint = session.messages.iter().any(|m| {
            m.content.iter().any(|b| {
                matches!(b, agent_core::ContentBlock::ToolResult { content, is_error, .. }
                    if *is_error && content.contains("files_changed"))
            })
        });
        assert!(
            complaint,
            "the failing call must produce an error tool_result naming the field"
        );
    }

    #[tokio::test]
    async fn a_mixed_batch_stages_the_value_and_lets_the_run_continue() {
        // `terminate` is ANDed across the batch, so a sibling call the model dispatched in the same turn
        // is never cut off. The payload is captured but the run goes on — and a later, solo call may
        // revise it.
        let (agent, mock, slot) = agent_with(vec![
            [
                turn::tool_call("tu_1", NAME, VALID),
                turn::tool_call("tu_2", "noop", "{}"),
            ]
            .concat(),
            turn::tool_call(
                "tu_3",
                NAME,
                r#"{"files_changed":["b.rs"],"summary":"revised"}"#,
            ),
        ]);
        let mut session = Session::new();
        session.user("do it");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(mock.calls(), 2, "a mixed batch must not end the run");
        assert_eq!(
            slot.take().unwrap()["summary"],
            "revised",
            "the host reads the slot after the run drains, so it sees the final value"
        );
    }

    #[tokio::test]
    async fn the_loops_injected_vision_flag_never_reaches_the_schema_or_the_captured_value() {
        // Regression: the dispatch loop stamps `_model_supports_vision` onto every tool call's coerced
        // input. `schema()` sets `"additionalProperties": false` — the right thing for a schema that
        // describes an exact payload — so before this was stripped, *every* call failed validation and
        // no run with a strict schema could ever return anything.
        let (agent, mock, slot) = agent_with(vec![turn::tool_call("tu_1", NAME, VALID)]);
        let mut session = Session::new();
        session.user("do it");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(mock.calls(), 1, "the call must validate on the first try");
        let captured = slot.take().expect("a payload");
        assert!(
            captured
                .get(agent_core::tool::MODEL_SUPPORTS_VISION_KEY)
                .is_none(),
            "the host must read back exactly what the model wrote: {captured}"
        );
    }

    #[tokio::test]
    async fn a_run_that_never_calls_the_tool_leaves_the_slot_empty() {
        // What every host reports as a failure: the contract the run was started with was not met.
        let (agent, _mock, slot) = agent_with(vec![turn::text("here is a prose answer")]);
        let mut session = Session::new();
        session.user("do it");
        agent.run(&mut session, |_| {}).await.unwrap();
        assert_eq!(slot.take(), None);
    }
}
