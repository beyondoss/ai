//! The tool seam: capabilities the model can invoke.
//!
//! A [`Tool`] is a value; the agent is configured by registering tools in a [`ToolRegistry`]. This
//! is the harness's primary extensibility point (Pi's Extensions/Skills/Tools): the core four
//! (Read/Write/Edit/Bash) and the Beyond primitives (fork/sync/logs) are all just registered tools,
//! and tests register a mock tool to exercise the loop without a real capability.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::channel::mpsc::UnboundedSender;
use serde_json::Value;

use crate::error::ToolError;
use crate::message::{ImageSource, ToolDef};

/// One streamed progress update from a running tool: a **snapshot** of its output so far (not a delta
/// — clients render the latest, matching pi's `tool_execution_update`/`partialResult` model) plus
/// optional tool-specific `details` (e.g. bash's truncation info + full-output-file path).
pub struct ToolUpdate {
    pub id: String,
    pub name: String,
    /// The full accumulated output so far (a snapshot, not an incremental chunk).
    pub snapshot: String,
    /// Tool-specific structured detail (bash: `{ truncation, full_output_path }`). `None` when there's
    /// nothing extra to report (e.g. the initial empty update).
    pub details: Option<Value>,
}

/// A sink a tool uses to emit progress *while it is still running* — a long shell command's streaming
/// output, a large download, a multi-step tool. Each update is a **snapshot** (the full output so far)
/// that surfaces to the run's observers as an `AgentEvent::ToolProgress` (pi's `tool_execution_update`).
/// The loop hands one to every call; a tool that doesn't stream simply never calls
/// [`emit`](ToolProgress::emit). Cloneable and `Send`, so a tool can hand it to a background read loop.
#[derive(Clone)]
pub struct ToolProgress {
    tx: UnboundedSender<ToolUpdate>,
    id: String,
    name: String,
}

impl ToolProgress {
    /// Build a progress handle for one tool call. The loop constructs these; tools receive `&ToolProgress`.
    pub(crate) fn new(tx: UnboundedSender<ToolUpdate>, id: String, name: String) -> Self {
        Self { tx, id, name }
    }

    /// Emit a progress snapshot (the full output so far) plus optional `details`. Best-effort: if the
    /// run has already finished (the receiver is gone), the update is dropped rather than erroring.
    pub fn emit(&self, snapshot: impl Into<String>, details: Option<Value>) {
        let _ = self.tx.unbounded_send(ToolUpdate {
            id: self.id.clone(),
            name: self.name.clone(),
            snapshot: snapshot.into(),
            details,
        });
    }
}

/// A tool's successful output: text for the model, plus any images it produced (a screenshot, a
/// rendered chart, `read` on an image file) and an optional hint to end the run.
///
/// The overwhelmingly common case is plain text, so [`From<String>`]/[`From<&str>`] make returning it
/// a one-liner (`Ok("done".into())`). Image-producing tools attach [`ImageSource`]s the multimodal
/// model can actually see — without this, a tool could only ever hand back UTF-8 text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolOutput {
    /// Text shown to the model (may be empty if the result is purely visual).
    pub text: String,
    /// Images attached to the result. Empty for the typical text-only tool.
    pub images: Vec<ImageSource>,
    /// When set, asks the loop to end the run after this batch — provided *every* call in the batch
    /// agrees (an `attempt_completion`/`exit`-style tool). Defaults to `false`.
    pub terminate: bool,
}

impl ToolOutput {
    /// A text-only result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    /// A result carrying a single image plus optional accompanying text.
    pub fn image(text: impl Into<String>, source: ImageSource) -> Self {
        Self {
            text: text.into(),
            images: vec![source],
            terminate: false,
        }
    }

    /// Builder-style: mark this result as requesting the run end (see [`ToolOutput::terminate`]).
    pub fn with_terminate(mut self, terminate: bool) -> Self {
        self.terminate = terminate;
        self
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

/// A capability the model can call. Implementors are cheap, `Send + Sync` values stored behind an
/// `Arc` in the registry.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Stable identifier the model uses to call this tool.
    fn name(&self) -> &str;

    /// One-line description shown to the model — what the tool does and when to use it.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input arguments object.
    fn input_schema(&self) -> Value;

    /// Run the tool against a (schema-conformant) arguments object, returning a [`ToolOutput`] (text,
    /// and optionally images, for the model). Plain text converts in with `.into()`. Return
    /// [`ToolError`] on failure; the loop surfaces it as an error `tool_result` rather than aborting
    /// the run.
    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError>;

    /// Run with a [`ToolProgress`] sink for incremental output, for tools whose work is worth
    /// streaming (long shell commands, large transfers). Defaults to [`run`](Tool::run) — most tools
    /// produce their result in one shot — so overriding is opt-in and existing tools are unaffected.
    /// Whatever a tool emits via `progress` reaches the run's observers as `AgentEvent::ToolProgress`
    /// *before* the final `ToolEnd`/`tool_result`.
    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        let _ = progress;
        self.run(input).await
    }

    /// The filesystem path this call would write to, if any. The loop runs a turn's tool calls
    /// concurrently (see `Agent::run_events`), but two calls that write the *same* path — e.g. the
    /// model batches two `edit`s against one file — must not race: each tool independently
    /// reads-modifies-writes, so unordered execution can silently drop one write or interleave both
    /// into a corrupt file. Returning the target path here makes the loop run same-path calls
    /// sequentially, in call order, instead. Read-only tools (and tools whose input has no path) keep
    /// the default `None` and stay fully concurrent.
    fn write_target(&self, _input: &Value) -> Option<String> {
        None
    }

    /// The advertised definition sent to the model. Derived from the accessors; override only if a
    /// tool needs a non-default shape.
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

/// A name-keyed set of tools available to one agent.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, keyed by its `name()`. A later registration with the same name replaces the
    /// earlier one (last-wins), which is how an extension overrides a built-in.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> &mut Self {
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    /// Look up a tool by the name the model called.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// The definitions to advertise to the model, **sorted by name**. The order must be stable across
    /// calls (and process restarts): the Anthropic dialect marks a prompt-cache breakpoint on the tool
    /// block, and a cache hit needs a byte-identical prefix — `HashMap` iteration order would cold-miss
    /// the cache after every `serve` reattach.
    pub fn definitions(&self) -> Vec<ToolDef> {
        let mut defs: Vec<ToolDef> = self.tools.values().map(|t| t.definition()).collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A trivial tool that echoes its `text` argument back — the standard stand-in for exercising
    /// the registry and (later) the loop without a real capability.
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echo the `text` argument back."
        }
        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            })
        }
        async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
            input
                .get("text")
                .and_then(Value::as_str)
                .map(|s| s.into())
                .ok_or_else(|| ToolError::InvalidInput("missing `text`".into()))
        }
    }

    #[test]
    fn register_and_get() {
        let mut reg = ToolRegistry::new();
        assert!(reg.is_empty());
        reg.register(Arc::new(EchoTool));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("echo").is_some());
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn definition_derives_from_accessors() {
        let def = EchoTool.definition();
        assert_eq!(def.name, "echo");
        assert_eq!(def.input_schema["required"][0], "text");
    }

    #[test]
    fn last_registration_wins() {
        struct Other;
        #[async_trait]
        impl Tool for Other {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "different impl, same name"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(&self, _: Value) -> Result<ToolOutput, ToolError> {
                Ok("other".into())
            }
        }
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool)).register(Arc::new(Other));
        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.get("echo").unwrap().description(),
            "different impl, same name"
        );
    }

    #[test]
    fn definitions_are_sorted_by_name() {
        struct Named(&'static str);
        #[async_trait]
        impl Tool for Named {
            fn name(&self) -> &str {
                self.0
            }
            fn description(&self) -> &str {
                "stand-in for sort-order testing"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(&self, _: Value) -> Result<ToolOutput, ToolError> {
                Ok(self.0.into())
            }
        }
        let mut reg = ToolRegistry::new();
        // Registered out of name order — `definitions()` must still come back sorted: the Anthropic
        // dialect anchors a prompt-cache breakpoint on the *last* tool definition, and a cache hit
        // needs a byte-identical prefix across turns and process restarts, which `HashMap` iteration
        // order alone doesn't guarantee.
        for name in ["write", "bash", "edit", "read"] {
            reg.register(Arc::new(Named(name)));
        }
        let names: Vec<String> = reg.definitions().into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["bash", "edit", "read", "write"]);
    }

    #[tokio::test]
    async fn echo_runs() {
        let out = EchoTool.run(json!({ "text": "hi" })).await.unwrap();
        assert_eq!(out.text, "hi");
        let err = EchoTool.run(json!({})).await;
        assert!(matches!(err, Err(ToolError::InvalidInput(_))));
    }
}
