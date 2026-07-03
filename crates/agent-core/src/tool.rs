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
use tokio_util::sync::CancellationToken;

use crate::error::ToolError;
use crate::message::{ImageSource, ToolDef};

/// One update from a running (or just-finished) tool call, streamed to the loop's per-turn progress
/// channel so it can forward each one to `sink` the moment it arrives — rather than batching
/// completions until every concurrently-dispatched call in the turn has finished.
pub enum ToolUpdate {
    /// A **snapshot** of the tool's output so far (not a delta — clients render the latest, matching
    /// pi's `tool_execution_update`/`partialResult` model) plus optional tool-specific `details`.
    Progress {
        id: String,
        name: String,
        /// The full accumulated output so far (a snapshot, not an incremental chunk).
        snapshot: String,
        /// Tool-specific structured detail (bash: `{ truncation, full_output_path }`). `None` when
        /// there's nothing extra to report (e.g. the initial empty update).
        details: Option<Value>,
    },
    /// A tool call's final result, sent the instant its own future resolves.
    End {
        id: String,
        name: String,
        result: String,
        is_error: bool,
    },
}

/// A sink a tool uses to emit progress *while it is still running* — a long shell command's streaming
/// output, a large download, a multi-step tool. Each update is a **snapshot** (the full output so far)
/// that surfaces to the run's observers as an `AgentEvent::ToolProgress` (pi's `tool_execution_update`).
/// The loop hands one to every call; a tool that doesn't stream simply never calls
/// [`emit`](ToolProgress::emit). Cloneable and `Send`, so a tool can hand it to a background read loop.
///
/// Also carries the run's cancellation token (see [`is_cancelled`](ToolProgress::is_cancelled)) — a
/// **deliberate** choice over adding a signal parameter to [`Tool::run`]/[`Tool::run_streaming`]. The
/// loop already cancels a tool by racing its future against the token and dropping it on a trip (a
/// `bash` subprocess dies via `kill_on_drop`), so the trait seams stay minimal; this only gives a
/// long-running *streaming* tool — the one kind of tool that does real work between yield points and so
/// can't rely on the drop alone happening promptly — a way to check in on its own schedule and wind
/// down early (flush what it has, stop polling a subprocess for more output) rather than being cut off
/// mid-operation with no chance to finalize partial state.
#[derive(Clone)]
pub struct ToolProgress {
    tx: UnboundedSender<ToolUpdate>,
    id: String,
    name: String,
    cancel: CancellationToken,
}

impl ToolProgress {
    /// Build a progress handle for one tool call. The loop constructs these for a model-invoked call;
    /// tools receive `&ToolProgress`. `pub` (not `pub(crate)`) so a host can invoke a registered tool
    /// directly outside the model loop — e.g. `serve`'s `bash` RPC command — while still reusing the
    /// same streaming/cancellation plumbing a model-driven call gets, rather than that host needing its
    /// own parallel progress-reporting mechanism.
    pub fn new(
        tx: UnboundedSender<ToolUpdate>,
        id: String,
        name: String,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            tx,
            id,
            name,
            cancel,
        }
    }

    /// Emit a progress snapshot (the full output so far) plus optional `details`. Best-effort: if the
    /// run has already finished (the receiver is gone), the update is dropped rather than erroring.
    pub fn emit(&self, snapshot: impl Into<String>, details: Option<Value>) {
        let _ = self.tx.unbounded_send(ToolUpdate::Progress {
            id: self.id.clone(),
            name: self.name.clone(),
            snapshot: snapshot.into(),
            details,
        });
    }

    /// Whether the run has been cancelled. A streaming tool doing real work between yield points (a
    /// polling loop reading subprocess output, say) can check this to wind down cooperatively — flush
    /// what it has and return — instead of relying solely on the loop dropping its future, which cuts
    /// it off with no chance to finalize partial state.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// A future that resolves the moment this call's run is cancelled — for a tool to race against its
    /// own work (e.g. in a `tokio::select!` alongside a subprocess read loop) so it notices cancellation
    /// promptly and can flush partial state itself, rather than only ever being cut off by an external
    /// `Drop` of its whole future with no chance to finalize.
    pub fn cancelled(&self) -> impl std::future::Future<Output = ()> + '_ {
        self.cancel.cancelled()
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

    /// Whether this call's mutation scope can't be named by [`write_target`](Tool::write_target) at
    /// all — an opaque shell command is the paradigm case: it can write anywhere reachable from its
    /// `cwd`, so there's no single path to group it against. Without this, such a call reports `None`
    /// like any other read-only tool and gets a unique `solo:` group, running fully concurrently with
    /// a same-turn `edit`/`write` it could easily race on disk (e.g. the model batching an `edit` on
    /// `foo.py` with a `bash: black foo.py` in one turn). When any call in a turn returns `true` here,
    /// the loop drops that whole turn's dispatch to one call at a time (see `Agent::run_events`)
    /// instead of its usual per-group concurrency — conservative, but correctness beats parallelism
    /// for a call whose blast radius can't be scoped. Defaults to `false`: most tools either don't
    /// mutate, or name their exact target via `write_target`.
    fn conservative_exclusive(&self) -> bool {
        false
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

    /// Keep only tools whose name satisfies `predicate`, dropping the rest — the one primitive a
    /// caller needs to apply an allow-list (`retain(|n| allowed.contains(n))`), a deny-list
    /// (`retain(|n| !excluded.contains(n))`), or "no tools at all" (`retain(|_| false)`), without the
    /// registry needing separate concepts for each.
    pub fn retain(&mut self, mut predicate: impl FnMut(&str) -> bool) -> &mut Self {
        self.tools.retain(|name, _| predicate(name));
        self
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

    #[test]
    fn retain_applies_an_allow_list() {
        let mut reg = ToolRegistry::new();
        for name in ["read", "write", "bash"] {
            reg.register(Arc::new(Named(name)));
        }
        let allow = ["read".to_string()];
        reg.retain(|n| allow.iter().any(|a| a == n));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("read").is_some());
    }

    #[test]
    fn retain_applies_a_deny_list() {
        let mut reg = ToolRegistry::new();
        for name in ["read", "write", "bash"] {
            reg.register(Arc::new(Named(name)));
        }
        let deny = ["bash".to_string()];
        reg.retain(|n| !deny.iter().any(|d| d == n));
        assert_eq!(reg.len(), 2);
        assert!(reg.get("bash").is_none());
    }

    #[test]
    fn retain_false_empties_the_registry() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        reg.retain(|_| false);
        assert!(reg.is_empty());
    }

    /// A trivial named tool, reused across the sort-order and `retain` tests.
    struct Named(&'static str);
    #[async_trait]
    impl Tool for Named {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "stand-in for name-keyed testing"
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn run(&self, _: Value) -> Result<ToolOutput, ToolError> {
            Ok(self.0.into())
        }
    }

    #[test]
    fn tool_progress_reflects_cancellation() {
        let (tx, _rx) = futures::channel::mpsc::unbounded();
        let cancel = CancellationToken::new();
        let progress = ToolProgress::new(tx, "id".into(), "name".into(), cancel.clone());
        assert!(!progress.is_cancelled());
        cancel.cancel();
        // A clone made *before* the trip still observes it — the token is shared, not snapshotted.
        assert!(progress.is_cancelled());
    }
}
