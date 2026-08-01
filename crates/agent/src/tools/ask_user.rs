//! `ask_user` — stop and ask, instead of guessing.
//!
//! An agent that does not know which of two things you meant has, until now,
//! had exactly one move: pick one and find out at the end whether it was right.
//! That is the most expensive way to be wrong — minutes of work, a pull request
//! nobody wanted, and a person who has to explain what they meant *after*
//! reading a diff instead of before.
//!
//! # Asking is ending the turn, not blocking in a tool
//!
//! The answer cannot come back through this tool's return value. It arrives
//! minutes or hours later, from a person, wherever the work was requested. So
//! the tool records the question and *terminates the turn*
//! ([`ToolOutput::with_terminate`]).
//!
//! A host learns about it the same way it learns about every other tool call —
//! the `ToolStart` event, carrying this tool's name and its arguments. There is
//! no second channel and nothing to subscribe to: a client already following
//! the session sees the question the moment it is asked.
//!
//! Blocking here instead would hold a model turn open across a coffee break,
//! burn the context window on nothing, and put the whole run at the mercy of
//! every timeout between here and the person.
//!
//! # Why nothing else is needed to resume
//!
//! The session outlives the turn. A session that has ended a turn is idle, not
//! gone, and the next `prompt` continues it — the same path a person's
//! "actually, do the other one" already takes. So the answer needs no special
//! channel, no pause state and no timeout to manage: it is just the next thing
//! said in a conversation that was already open.
//!
//! # The options are a courtesy, not a constraint
//!
//! When the question is a choice between known alternatives, saying so lets a
//! client render them as buttons and a person answer with one tap. Nothing
//! enforces that the answer is one of them — a person who replies "neither,
//! do X" is giving a better answer than the tool knew to offer.

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

pub const NAME: &str = "ask_user";

/// Cap on a question. Long enough for a real one with context, short enough
/// that it renders in a chat message rather than becoming a wall of text.
const MAX_QUESTION: usize = 2000;

/// Cap on how many alternatives are worth offering. Past a handful this is a
/// question to answer in prose, and chat surfaces stop rendering buttons.
const MAX_OPTIONS: usize = 5;

/// Cap on one option's label.
const MAX_OPTION: usize = 120;

#[derive(Default)]
pub struct AskUser;

impl AskUser {
    pub fn new() -> Self {
        Self
    }

    fn dispatch(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let question = input
            .get("question")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|q| !q.is_empty())
            .ok_or_else(|| {
                ToolError::InvalidInput(
                    "missing `question` — say what you need to know, in one sentence".into(),
                )
            })?;

        if question.chars().count() > MAX_QUESTION {
            return Err(ToolError::InvalidInput(format!(
                "`question` is too long ({} characters, limit {MAX_QUESTION}). \
                 Ask the one thing you are blocked on; the rest is context you already have.",
                question.chars().count()
            )));
        }

        let options = parse_options(input.get("options"))?;

        // Terminating is the point. The model must not answer its own question
        // or carry on guessing, and the text says so plainly rather than
        // relying on the flag alone — a model that reads this and keeps going
        // would produce exactly the run this tool exists to prevent.
        Ok(ToolOutput::from(render(question, &options)).with_terminate(true))
    }
}

/// Validates the suggested answers.
fn parse_options(raw: Option<&Value>) -> Result<Vec<String>, ToolError> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    if raw.is_null() {
        return Ok(Vec::new());
    }
    let items = raw
        .as_array()
        .ok_or_else(|| ToolError::InvalidInput("`options` must be an array of strings".into()))?;

    if items.len() > MAX_OPTIONS {
        return Err(ToolError::InvalidInput(format!(
            "too many `options` ({}, limit {MAX_OPTIONS}). \
             More than a handful is a question to ask in prose.",
            items.len()
        )));
    }

    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let s = item
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                ToolError::InvalidInput(format!("options[{i}]: must be a non-empty string"))
            })?;
        if s.chars().count() > MAX_OPTION {
            return Err(ToolError::InvalidInput(format!(
                "options[{i}]: too long ({} characters, limit {MAX_OPTION}). \
                 An option is a label, not an explanation.",
                s.chars().count()
            )));
        }
        out.push(s.to_string());
    }
    Ok(out)
}

/// The text the model sees as this tool's result.
///
/// It will be the last thing in the transcript before the turn ends, and the
/// first thing the model re-reads when the answer arrives — so it states what
/// happened rather than echoing the arguments back.
fn render(question: &str, options: &[String]) -> String {
    let mut s = format!("Asked: {question}");
    if !options.is_empty() {
        s.push_str("\nSuggested answers: ");
        s.push_str(&options.join(" / "));
    }
    s.push_str(
        "\n\nYour turn ends here. Do not answer this yourself and do not continue guessing — \
         the person's reply will arrive as the next message in this session.",
    );
    s
}

#[async_trait]
impl Tool for AskUser {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        "Ask the person who requested this work a question, and end your turn to wait for their \
         answer. Use it when you are genuinely blocked on a decision only they can make — which of \
         two interpretations they meant, or permission for something destructive or irreversible. \
         Do not use it for anything you can determine by reading the code, and do not use it to \
         confirm work you have already been asked to do. Their reply arrives as the next message."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The one thing you are blocked on, in a sentence or two. Include the context needed to answer it without reading the code — the person may be answering from their phone."
                },
                "options": {
                    "type": "array",
                    "description": "Optional. The alternatives, when the question is a choice between known ones — a client can then render them as buttons. Short labels, not explanations. The person may answer with something else entirely.",
                    "items": { "type": "string" }
                }
            },
            "required": ["question"]
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        self.dispatch(input)
    }

    async fn run_streaming(
        &self,
        input: Value,
        _progress: &agent_core::tool::ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        self.dispatch(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(input: Value) -> Result<ToolOutput, ToolError> {
        AskUser::new().dispatch(input)
    }

    #[test]
    fn asking_ends_the_turn() {
        // Terminating is the whole mechanism: the answer arrives later, as a new
        // message, and a turn left open would be waiting on a person.
        let out = ask(json!({
            "question": "Should the retry back off exponentially or linearly?",
            "options": ["Exponential", "Linear"],
        }))
        .unwrap();
        assert!(out.terminate);
    }

    #[test]
    fn tells_the_model_not_to_answer_itself() {
        // The failure mode this guards: a model that treats the tool result as
        // permission to carry on picks an interpretation anyway, which is the
        // exact run `ask_user` exists to prevent.
        let text = format!("{:?}", ask(json!({"question": "Which service?"})).unwrap());
        assert!(text.contains("Do not answer this yourself"));
        assert!(text.contains("next message"));
    }

    #[test]
    fn the_question_and_options_are_echoed_for_the_model_to_re_read() {
        // This result is the last thing in the transcript before the turn ends,
        // and the first thing re-read when the answer arrives.
        let text = format!(
            "{:?}",
            ask(json!({"question": "Which service?", "options": ["api", "worker"]})).unwrap()
        );
        assert!(text.contains("Which service?"));
        assert!(text.contains("api"));
        assert!(text.contains("worker"));
    }

    #[test]
    fn a_question_is_required() {
        for input in [json!({}), json!({"question": ""}), json!({"question": "   "})] {
            assert!(ask(input).is_err());
        }
    }

    #[test]
    fn options_are_optional_and_may_be_null() {
        for input in [
            json!({"question": "Which one?"}),
            json!({"question": "Which one?", "options": null}),
            json!({"question": "Which one?", "options": []}),
        ] {
            assert!(ask(input).is_ok());
        }
    }

    #[test]
    fn malformed_options_are_refused_rather_than_silently_dropped() {
        // Dropping them would end the turn on a question whose choices the model
        // believed it had offered.
        for input in [
            json!({"question": "Which?", "options": "Exponential"}),
            json!({"question": "Which?", "options": [""]}),
            json!({"question": "Which?", "options": [1, 2]}),
            json!({"question": "Which?", "options": ["a", "b", "c", "d", "e", "f"]}),
        ] {
            assert!(ask(input.clone()).is_err(), "should refuse: {input}");
        }
    }

    #[test]
    fn a_question_that_is_really_an_essay_is_refused() {
        assert!(ask(json!({"question": "x".repeat(MAX_QUESTION + 1)})).is_err());
    }
}
