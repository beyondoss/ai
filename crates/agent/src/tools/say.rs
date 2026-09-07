//! `say` — tell the person something while the work is still going.
//!
//! # The gap this fills
//!
//! A run has two moments where it can speak: a status line the platform writes
//! on a timer, which knows only which tool is running, and the final summary.
//! Everything the agent learns in between — that the bug is somewhere nobody
//! expected, that it is about to rewrite history, that the next step will take
//! six minutes — is held until the end or lost.
//!
//! The agent is the only participant that knows when something is worth saying
//! now rather than later. This is how it says it.
//!
//! # Why free-form rather than a fixed set of signals
//!
//! The obvious safer design is an enum — `blocked`, `heads_up`, `long_running`
//! — on the theory that a model judges "is this worth interrupting people for"
//! badly. But a closed set cannot carry the message with the most value in it:
//! "the bug is not where you thought, it is in the billing service" is none of
//! those, and forcing it into a category or into silence loses exactly the thing
//! somebody wanted to hear.
//!
//! What the enum was really protecting against is volume, and volume is bounded
//! by a budget rather than by a vocabulary.
//!
//! # Not [`super::ask_user`]
//!
//! `ask_user` ends the turn because it needs an answer before the work can
//! continue. This does not: it is something worth knowing, not something the
//! run depends on, so the agent says it and keeps going.

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

pub const NAME: &str = "say";

/// A message longer than this is a summary, not an aside.
const MAX_MESSAGE: usize = 1500;

#[derive(Default)]
pub struct Say;

impl Say {
    pub fn new() -> Self {
        Self
    }

    fn message(input: &Value) -> Result<String, ToolError> {
        let raw = input
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();

        if raw.is_empty() {
            return Err(ToolError::InvalidInput(
                "say: `message` is required and cannot be empty".into(),
            ));
        }
        if raw.chars().count() > MAX_MESSAGE {
            return Err(ToolError::InvalidInput(format!(
                "say: `message` is {} characters; keep it under {MAX_MESSAGE}. \
                 This is an aside, not the summary — save the detail for the end.",
                raw.chars().count()
            )));
        }
        Ok(raw.to_string())
    }

    /// The result the model reads back.
    ///
    /// It says the message was delivered and that repeating it is not useful,
    /// because the failure mode of a tool that reports success is a model that
    /// says the same thing again in its summary and makes the reader read it
    /// twice.
    fn delivered(message: &str) -> ToolOutput {
        ToolOutput::text(format!(
            "Sent to the person who asked:\n\n{message}\n\n\
             They have seen this now. Do not repeat it in your final summary — \
             say what happened after it instead. Keep working."
        ))
    }
}

#[async_trait]
impl Tool for Say {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        "Tell the person who asked something while you are still working. Use it when \
         you learn something that changes what they expect, when you are about to do \
         something you cannot undo, when you change approach, or when you are about to \
         be quiet for a long time. Do not narrate routine progress — they can already \
         see that you are working. This does not pause you: say it and carry on."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "What you want them to know, in a sentence or two. Write it as a message to a colleague — they may be reading it on a phone, and they have not seen your reasoning."
                }
            },
            "required": ["message"],
            "additionalProperties": false
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        // With no progress channel there is nowhere to send it. Reporting that
        // plainly beats pretending it was delivered, which would have the model
        // leave it out of the summary as well — and then nobody ever hears it.
        let message = Self::message(&input)?;
        Err(ToolError::Execution(format!(
            "say: no channel to deliver a message on, so this was not sent. \
             Include it in your summary instead: {message}"
        )))
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &agent_core::tool::ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        let message = Self::message(&input)?;

        // The message rides on the tool's own progress event, which the run
        // watcher is already reading for the status line and the plan. No new
        // transport, no second connection: one stream carries the whole run.
        progress.emit(message.clone(), Some(json!({ "message": message })));

        Ok(Self::delivered(&message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::tool::{ToolProgress, ToolUpdate};
    use futures::StreamExt;
    use futures::channel::mpsc;
    use tokio_util::sync::CancellationToken;

    fn progress(tx: mpsc::UnboundedSender<ToolUpdate>) -> ToolProgress {
        ToolProgress::new(tx, "1".into(), NAME.into(), CancellationToken::new())
    }

    fn say(input: Value) -> Result<ToolOutput, ToolError> {
        futures::executor::block_on(async {
            let (tx, _rx) = mpsc::unbounded();
            Say::new().run_streaming(input, &progress(tx)).await
        })
    }

    #[test]
    fn the_message_rides_on_the_progress_event() {
        futures::executor::block_on(async {
            let (tx, mut rx) = mpsc::unbounded();
            let p = progress(tx);

            Say::new()
                .run_streaming(json!({"message": "The bug is in the billing service."}), &p)
                .await
                .unwrap();
            drop(p);

            // Structured, not just rendered into the snapshot text: the watcher
            // reads `details.message`, and a change that only updated the
            // snapshot would deliver an empty message to somebody's channel.
            let update = rx.next().await.expect("a progress update should have been emitted");
            match update {
                ToolUpdate::Progress { name, snapshot, details, .. } => {
                    assert_eq!(name, NAME);
                    assert_eq!(snapshot, "The bug is in the billing service.");
                    let details = details.expect("the message must ride as structured detail");
                    assert_eq!(details["message"], "The bug is in the billing service.");
                }
                _ => panic!("expected a progress update"),
            }
        });
    }

    #[test]
    fn saying_does_not_end_the_turn() {
        // The whole difference from ask_user: this is worth knowing, not
        // something the run is waiting on.
        let out = say(json!({"message": "About to force-push the branch."})).unwrap();
        assert!(!out.terminate);
    }

    #[test]
    fn it_tells_the_model_not_to_say_it_twice() {
        // A model that treats delivery as a draft repeats it in the summary, and
        // the reader reads the same sentence in two places.
        let text = format!("{:?}", say(json!({"message": "Switching to a different approach."})).unwrap());
        assert!(text.contains("Do not repeat it"));
    }

    #[test]
    fn an_empty_message_is_refused() {
        assert!(say(json!({"message": "   "})).is_err());
        assert!(say(json!({})).is_err());
    }

    #[test]
    fn a_summary_sized_message_is_refused() {
        let long = "x".repeat(MAX_MESSAGE + 1);
        assert!(say(json!({ "message": long })).is_err());
    }

    #[test]
    fn without_a_channel_it_says_so_rather_than_pretending() {
        // Silently succeeding would be worse than failing: the model would also
        // leave it out of the summary, and nobody would ever hear it.
        let err = futures::executor::block_on(Say::new().run(json!({"message": "hello"})));
        let text = format!("{err:?}");
        assert!(text.contains("not sent"));
        assert!(text.contains("hello"), "the message should survive into the error");
    }
}
