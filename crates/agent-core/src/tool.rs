//! The tool seam: capabilities the model can invoke.
//!
//! A [`Tool`] is a value; the agent is configured by registering tools in a [`ToolRegistry`]. This
//! is the harness's primary extensibility point (Pi's Extensions/Skills/Tools): the core four
//! (Read/Write/Edit/Bash) and the Beyond primitives (fork/sync/logs) are all just registered tools,
//! and tests register a mock tool to exercise the loop without a real capability.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::ToolError;
use crate::message::ToolDef;

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

    /// Run the tool against a (schema-conformant) arguments object, returning text for the model.
    /// Return [`ToolError`] on failure; the loop surfaces it as an error `tool_result` rather than
    /// aborting the run.
    async fn run(&self, input: Value) -> Result<String, ToolError>;

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
        async fn run(&self, input: Value) -> Result<String, ToolError> {
            input
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_string)
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
            async fn run(&self, _: Value) -> Result<String, ToolError> {
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
            async fn run(&self, _: Value) -> Result<String, ToolError> {
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
        assert_eq!(out, "hi");
        let err = EchoTool.run(json!({})).await;
        assert!(matches!(err, Err(ToolError::InvalidInput(_))));
    }
}
