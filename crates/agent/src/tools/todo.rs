//! `todo` — the model's own task list for multi-step work.
//!
//! **Stateless by construction.** The model sends the *complete* list on every call (full replace), so
//! there is nothing for the tool to remember: the last `todo` `tool_use` block in the transcript *is*
//! the current list. Holding it in a field would also be actively wrong — `default_registry_with_config`
//! builds fresh `Arc<dyn Tool>` values on every `set_model`/`set_thinking` rebuild (see
//! `serve::build_agent`), so a tool-owned list would silently reset mid-session on a model switch.
//!
//! Durability is handled one layer down, not here. A compaction physically drops the `tool_use` block
//! that carried the list, so `Agent::compact` lifts it out of the doomed prefix into
//! `Session::compaction.todos` (folded forward every round, persisted with the session) and re-renders
//! it into each new summary as a `<todo_list>` block. That is the same deterministic-carry channel the
//! read/modified file list already rides, and it exists for the same reason: a summarizing model cannot
//! be relied on to preserve the plan in its own prose. See `agent_core::compaction`'s module doc.
//!
//! The live UI feed is [`ToolProgress::emit`]'s `details`, which reaches `serve` clients verbatim as
//! `AgentEvent::ToolProgress`. That stream is ephemeral, so a client attaching mid-run asks `serve` for
//! `get_todos` rather than replaying events.

use agent_core::tool::{Tool, ToolProgress};
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

pub const NAME: &str = "todo";

/// The `status` values the schema admits. The glyph each renders as lives with the shared renderer,
/// [`agent_core::compaction::render_todo_list`].
const STATUSES: [&str; 3] = ["pending", "in_progress", "completed"];

#[derive(Default)]
pub struct Todo;

impl Todo {
    pub fn new() -> Self {
        Self
    }
}

/// Validate the `todos` argument.
///
/// Hard errors are the ones that corrupt the tool's *purpose*: a second `in_progress` destroys the
/// "what am I doing right now" signal the list exists to carry, and a blank `content`/`activeForm`
/// renders as an empty checklist row. Everything a reasonable plan might legitimately contain — an
/// empty list ("plan cleared"), two similarly-worded steps — is accepted, because rejecting it only
/// teaches the model to retry over harmless phrasing.
fn validate(todos: &[Value]) -> Result<(), ToolError> {
    let mut in_progress = Vec::new();

    for (i, raw) in todos.iter().enumerate() {
        let field = |key: &str| -> Result<&str, ToolError> {
            let v = raw
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| ToolError::InvalidInput(format!("todos[{i}]: missing `{key}`")))?;
            if v.trim().is_empty() {
                return Err(ToolError::InvalidInput(format!(
                    "todos[{i}]: `{key}` must not be empty"
                )));
            }
            Ok(v)
        };

        let content = field("content")?;
        // Validated but unused in the rendering — it exists so a UI can label the running step in the
        // present tense without inferring it from the imperative `content`.
        let _active_form = field("activeForm")?;
        let status = field("status")?;

        if !STATUSES.contains(&status) {
            return Err(ToolError::InvalidInput(format!(
                "todos[{i}]: `status` must be one of {} (got `{status}`)",
                STATUSES.join(", ")
            )));
        }
        if status == "in_progress" {
            in_progress.push(content);
        }
    }

    if in_progress.len() > 1 {
        return Err(ToolError::InvalidInput(format!(
            "exactly one todo may be `in_progress` at a time, found {}: {}. \
             Mark all but the one you are actively working on as `pending` or `completed`.",
            in_progress.len(),
            in_progress.join(", ")
        )));
    }
    Ok(())
}

/// The text handed back to the model. A compact checklist rather than a JSON echo: the model re-reads
/// this on its next turn, so it should re-anchor it on the plan for a few dozen tokens, not make it
/// re-parse its own arguments.
///
/// The checklist body itself comes from [`agent_core::compaction::render_todo_list`] — the *same*
/// renderer a compaction summary's `<todo_list>` block uses, so the plan the model reads after a cut is
/// byte-identical to the one it read before.
fn render(todos: &[Value]) -> String {
    if todos.is_empty() {
        return "Todo list cleared.".to_string();
    }
    format!(
        "Todos updated ({} item{}):\n{}",
        todos.len(),
        if todos.len() == 1 { "" } else { "s" },
        agent_core::compaction::render_todo_list(todos)
    )
}

impl Todo {
    fn dispatch(&self, input: Value, progress: &ToolProgress) -> Result<ToolOutput, ToolError> {
        let todos = input
            .get("todos")
            .and_then(Value::as_array)
            .ok_or_else(|| ToolError::InvalidInput("missing `todos` array".into()))?;
        validate(todos)?;
        let text = render(todos);
        // The structured feed a client renders as a live checklist. `details` is arbitrary JSON passed
        // through verbatim to observers, so the array goes out exactly as the model wrote it.
        progress.emit(text.clone(), Some(json!({ "todos": todos })));
        Ok(text.into())
    }
}

#[async_trait]
impl Tool for Todo {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        "Track progress on multi-step work. Send the COMPLETE list every call — it fully replaces the \
         previous one. Keep exactly one item `in_progress`."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete task list. Fully replaces the previous list — always send every item, not just the ones that changed. Send an empty array to clear the plan.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "Imperative form, e.g. \"Add the retry loop\"." },
                            "activeForm": { "type": "string", "description": "Present continuous form, shown while the item is in progress, e.g. \"Adding the retry loop\"." },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "At most one item may be `in_progress`."
                            }
                        },
                        "required": ["content", "activeForm", "status"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        // The loop always calls `run_streaming`; this exists for a host invoking the tool directly. The
        // throwaway progress sink discards the update.
        let (tx, _rx) = futures::channel::mpsc::unbounded();
        let progress = ToolProgress::new(
            tx,
            String::new(),
            NAME.to_string(),
            agent_core::CancellationToken::new(),
        );
        self.dispatch(input, &progress)
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        self.dispatch(input, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(content: &str, status: &str) -> Value {
        json!({ "content": content, "activeForm": format!("{content}ing"), "status": status })
    }

    async fn run(input: Value) -> Result<ToolOutput, ToolError> {
        Todo::new().run(input).await
    }

    #[tokio::test]
    async fn renders_a_compact_checklist() {
        let out = run(json!({ "todos": [
            item("Parse the config", "completed"),
            item("Wire the retry loop", "in_progress"),
            item("Add tests", "pending"),
        ]}))
        .await
        .unwrap()
        .text;
        assert_eq!(
            out,
            "Todos updated (3 items):\n\
             [x] Parse the config\n\
             [>] Wire the retry loop\n\
             [ ] Add tests"
        );
    }

    #[tokio::test]
    async fn one_item_is_singular() {
        let out = run(json!({ "todos": [item("Ship it", "pending")] }))
            .await
            .unwrap()
            .text;
        assert!(out.starts_with("Todos updated (1 item):"), "got: {out}");
    }

    #[tokio::test]
    async fn an_empty_list_clears_the_plan() {
        // The legitimate "I'm done tracking" signal, not an error.
        let out = run(json!({ "todos": [] })).await.unwrap().text;
        assert_eq!(out, "Todo list cleared.");
    }

    #[tokio::test]
    async fn two_in_progress_items_are_rejected() {
        // The one invariant that carries the tool's whole purpose: which step is running *now*.
        let err = run(json!({ "todos": [
            item("First", "in_progress"),
            item("Second", "in_progress"),
        ]}))
        .await
        .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("found 2"), "got: {msg}");
                assert!(msg.contains("First, Second"), "got: {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn duplicate_content_is_accepted() {
        // Two genuinely similar steps happen; rejecting them only teaches the model to retry over
        // phrasing collisions that cost nothing.
        let out =
            run(json!({ "todos": [item("Add tests", "pending"), item("Add tests", "pending")] }))
                .await
                .unwrap()
                .text;
        assert!(out.starts_with("Todos updated (2 items):"), "got: {out}");
    }

    #[tokio::test]
    async fn an_empty_content_is_rejected() {
        let err = run(json!({ "todos": [
            { "content": "  ", "activeForm": "Doing", "status": "pending" }
        ]}))
        .await
        .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => assert!(msg.contains("`content` must not be empty")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_missing_active_form_is_rejected() {
        let err = run(json!({ "todos": [{ "content": "Do it", "status": "pending" }] }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("missing `activeForm`"), "got: {msg}")
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_unknown_status_is_rejected() {
        let err = run(json!({ "todos": [item("Do it", "blocked")] }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => assert!(msg.contains("got `blocked`"), "got: {msg}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_missing_todos_array_is_rejected() {
        let err = run(json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn the_progress_details_carry_the_list_verbatim() {
        // What a `serve` client renders as a live checklist. Arrives as `AgentEvent::ToolProgress`.
        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        let progress = ToolProgress::new(
            tx,
            "call-1".to_string(),
            NAME.to_string(),
            agent_core::CancellationToken::new(),
        );
        let todos = json!([item("Wire it up", "in_progress")]);
        Todo::new()
            .run_streaming(json!({ "todos": todos }), &progress)
            .await
            .unwrap();

        use futures::StreamExt;
        let update = rx.next().await.expect("one progress update");
        match update {
            agent_core::tool::ToolUpdate::Progress {
                snapshot, details, ..
            } => {
                assert!(snapshot.contains("[>] Wire it up"));
                assert_eq!(details.unwrap()["todos"], todos);
            }
            // `ToolUpdate` is not `Debug`, so there is nothing to print here.
            _ => panic!("expected ToolUpdate::Progress"),
        }
    }
}
