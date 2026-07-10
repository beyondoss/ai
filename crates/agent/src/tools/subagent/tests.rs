//! Unit tests for the `subagent` tool, driven by `MockTransport` so no network or real model is
//! involved. The factory hands each child a transport keyed by its model id and remembers it, so a test
//! can both script a child's reply and inspect exactly what request the child received (its system
//! prompt, and the user task after `{previous}` substitution).

use std::collections::HashMap;

use agent_core::{MockTransport, ModelTransport};

use super::*;
use crate::agents::{AgentDef, Isolation};

/// Hands out a fresh `MockTransport` per (model, call) and keeps a handle to each, so a test can assert
/// on what a child was actually sent. Keyed by model id; when several children share a model, the latest
/// wins the map slot — tests that need to inspect requests give each child a distinct model.
#[derive(Clone, Default)]
struct Factory {
    /// Model id -> the reply text that model's child should produce.
    replies: Arc<std::sync::Mutex<HashMap<String, String>>>,
    /// Model id -> the transport handed to that model's child, for request inspection.
    handed: Arc<std::sync::Mutex<HashMap<String, Arc<MockTransport>>>>,
    /// Every model id the factory was asked to build, in call order.
    calls: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Factory {
    fn reply(self, model: &str, text: &str) -> Self {
        self.replies
            .lock()
            .unwrap()
            .insert(model.to_string(), text.to_string());
        self
    }

    fn into_transport_factory(self) -> TransportFactory {
        Arc::new(move |model: &str| {
            self.calls.lock().unwrap().push(model.to_string());
            let text = self
                .replies
                .lock()
                .unwrap()
                .get(model)
                .cloned()
                .unwrap_or_else(|| format!("default reply from {model}"));
            let transport = Arc::new(MockTransport::new(vec![agent_core::mock::turn::text(
                &text,
            )]));
            self.handed
                .lock()
                .unwrap()
                .insert(model.to_string(), transport.clone());
            Ok(transport as Arc<dyn ModelTransport>)
        })
    }

    /// The recorded request a given model's child received (its most recent, if several).
    fn request_to(&self, model: &str) -> agent_core::ModelRequest {
        self.handed.lock().unwrap()[model].requests().pop().unwrap()
    }

    fn call_models(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

fn user_text(req: &agent_core::ModelRequest) -> String {
    req.messages
        .iter()
        .filter(|m| m.role == agent_core::Role::User)
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            agent_core::ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn def(
    name: &str,
    model: Option<&str>,
    tools: Option<Vec<&str>>,
    isolation: Isolation,
) -> AgentDef {
    AgentDef {
        name: name.to_string(),
        description: format!("test agent {name}"),
        tools: tools.map(|t| t.into_iter().map(str::to_string).collect()),
        model: model.map(str::to_string),
        isolation,
        system: format!("You are {name}."),
        path: std::path::PathBuf::from(format!("/tmp/{name}.md")),
        scope: "user",
    }
}

/// A ctx over the given defs and factory, with everything else defaulted to something inert. `cwd` is a
/// tempdir so a stray relative path can't touch the real tree.
fn ctx(defs: Vec<AgentDef>, factory: Factory) -> (Arc<SubagentCtx>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let ctx = Arc::new(SubagentCtx {
        factory: factory.into_transport_factory(),
        agents: Arc::new(defs),
        skills: Arc::new(Vec::new()),
        write_locks: Arc::new(agent_core::WriteLockRegistry::new()),
        mcp_tools: Vec::new(),
        tool_cfg: ChildToolConfig {
            image_auto_resize: true,
            ..Default::default()
        },
        cwd: dir.path().to_path_buf(),
        project_trusted: true,
        prompt_guidelines: Vec::new(),
        parent_model: "parent-model".to_string(),
        parent_cache_key: "parent-key".to_string(),
        parent_tools: vec!["read".to_string(), "grep".to_string()],
        deny_tool: Vec::new(),
        deny_bash_pattern: Vec::new(),
        deny_path: Vec::new(),
        child_max_steps: DEFAULT_CHILD_MAX_STEPS,
        max_depth: DEFAULT_MAX_DEPTH,
        approval: None,
    });
    (ctx, dir)
}

fn subagent(defs: Vec<AgentDef>, factory: Factory) -> (Subagent, tempfile::TempDir) {
    let (ctx, dir) = ctx(defs, factory);
    (Subagent::new(ctx), dir)
}

async fn run(tool: &Subagent, input: Value) -> Result<ToolOutput, ToolError> {
    tool.run(input).await
}

// ---- mode parsing ----

#[tokio::test]
async fn zero_modes_is_rejected() {
    let (tool, _d) = subagent(
        vec![def("scout", None, None, Isolation::None)],
        Factory::default(),
    );
    let err = run(&tool, json!({})).await.unwrap_err();
    assert!(matches!(err, ToolError::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("exactly one"), "{err}");
}

#[tokio::test]
async fn two_modes_is_rejected() {
    let (tool, _d) = subagent(
        vec![def("scout", None, None, Isolation::None)],
        Factory::default(),
    );
    let err = run(
        &tool,
        json!({ "agent": "scout", "task": "x", "tasks": [{ "agent": "scout", "task": "y" }] }),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("exactly one"), "{err}");
}

#[tokio::test]
async fn a_task_missing_its_agent_or_task_is_rejected() {
    let (tool, _d) = subagent(
        vec![def("scout", None, None, Isolation::None)],
        Factory::default(),
    );
    let err = run(&tool, json!({ "tasks": [{ "agent": "scout" }] }))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("needs a `task`"), "{err}");
}

// ---- single ----

#[tokio::test]
async fn single_returns_the_childs_final_message() {
    let factory = Factory::default().reply("parent-model", "the child's answer");
    let (tool, _d) = subagent(vec![def("scout", None, None, Isolation::None)], factory);
    let out = run(&tool, json!({ "agent": "scout", "task": "do it" }))
        .await
        .unwrap();
    assert_eq!(out.text, "the child's answer");
}

#[tokio::test]
async fn single_routes_to_the_defs_model_when_it_names_one() {
    let factory = Factory::default().reply("haiku-x", "from haiku");
    let f2 = factory.clone();
    let (tool, _d) = subagent(
        vec![def("scout", Some("haiku-x"), None, Isolation::None)],
        factory,
    );
    let out = run(&tool, json!({ "agent": "scout", "task": "do it" }))
        .await
        .unwrap();
    assert_eq!(out.text, "from haiku");
    assert_eq!(
        f2.call_models(),
        vec!["haiku-x"],
        "factory must be called with the def's model"
    );
}

#[tokio::test]
async fn single_inherits_the_parent_model_when_the_def_names_none() {
    let factory = Factory::default();
    let f2 = factory.clone();
    let (tool, _d) = subagent(vec![def("scout", None, None, Isolation::None)], factory);
    run(&tool, json!({ "agent": "scout", "task": "do it" }))
        .await
        .unwrap();
    assert_eq!(f2.call_models(), vec!["parent-model"]);
}

#[tokio::test]
async fn an_unknown_agent_is_an_execution_error_naming_the_available_ones() {
    let (tool, _d) = subagent(
        vec![def("scout", None, None, Isolation::None)],
        Factory::default(),
    );
    let err = run(&tool, json!({ "agent": "nope", "task": "x" }))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unknown agent"), "{err}");
    assert!(
        err.to_string().contains("scout"),
        "must list available agents: {err}"
    );
}

#[tokio::test]
async fn the_childs_system_prompt_is_its_definition_body() {
    let factory = Factory::default().reply("parent-model", "ok");
    let f2 = factory.clone();
    let (tool, _d) = subagent(vec![def("scout", None, None, Isolation::None)], factory);
    run(&tool, json!({ "agent": "scout", "task": "do it" }))
        .await
        .unwrap();
    let req = f2.request_to("parent-model");
    let system = req.system.unwrap_or_default();
    assert!(
        system.contains("You are scout."),
        "def body must be in the child prompt: {system}"
    );
}

#[tokio::test]
async fn the_childs_registry_is_restricted_to_its_tools() {
    // A def with `tools: read` must advertise only `read` to its model — the built-in base prompt lists
    // the registered tools, so the system prompt is a proxy for the registry.
    let factory = Factory::default().reply("parent-model", "ok");
    let f2 = factory.clone();
    let (tool, _d) = subagent(
        vec![def("reader", None, Some(vec!["read"]), Isolation::None)],
        factory,
    );
    run(&tool, json!({ "agent": "reader", "task": "look" }))
        .await
        .unwrap();
    let system = f2.request_to("parent-model").system.unwrap_or_default();
    assert!(system.contains("read"), "{system}");
    assert!(
        !system.contains("bash"),
        "a read-only agent must not be told it has bash: {system}"
    );
}

// ---- chain ----

#[tokio::test]
async fn chain_substitutes_previous_with_the_prior_childs_final_text() {
    let factory = Factory::default()
        .reply("model-a", "FIRST-OUTPUT")
        .reply("model-b", "SECOND-OUTPUT");
    let f2 = factory.clone();
    let (tool, _d) = subagent(
        vec![
            def("a", Some("model-a"), None, Isolation::None),
            def("b", Some("model-b"), None, Isolation::None),
        ],
        factory,
    );
    let out = run(
        &tool,
        json!({ "chain": [
            { "agent": "a", "task": "produce it" },
            { "agent": "b", "task": "given {previous}, continue" }
        ] }),
    )
    .await
    .unwrap();
    // Chain returns only the LAST step's output.
    assert_eq!(out.text, "SECOND-OUTPUT");
    // And step b's request must show the substitution happened.
    let b_task = user_text(&f2.request_to("model-b"));
    assert!(b_task.contains("given FIRST-OUTPUT, continue"), "{b_task}");
    assert!(
        !b_task.contains("{previous}"),
        "the placeholder must be gone: {b_task}"
    );
}

#[tokio::test]
async fn chain_short_circuits_on_the_first_failure_and_reports_partials() {
    let (tool, _d) = subagent(
        vec![def("a", Some("model-a"), None, Isolation::None)],
        Factory::default().reply("model-a", "A-DONE"),
    );
    // Second step names an unknown agent, so it fails; the error must carry step 1's result.
    let err = run(
        &tool,
        json!({ "chain": [
            { "agent": "a", "task": "ok" },
            { "agent": "ghost", "task": "boom" }
        ] }),
    )
    .await
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("chain stopped at step 2"), "{msg}");
    assert!(
        msg.contains("A-DONE"),
        "partial results must be included: {msg}"
    );
}

// ---- parallel ----

#[tokio::test]
async fn parallel_preserves_task_order_in_the_output() {
    let factory = Factory::default()
        .reply("m0", "ZERO")
        .reply("m1", "ONE")
        .reply("m2", "TWO");
    let (tool, _d) = subagent(
        vec![
            def("a0", Some("m0"), None, Isolation::None),
            def("a1", Some("m1"), None, Isolation::None),
            def("a2", Some("m2"), None, Isolation::None),
        ],
        factory,
    );
    let out = run(
        &tool,
        json!({ "tasks": [
            { "agent": "a0", "task": "t" },
            { "agent": "a1", "task": "t" },
            { "agent": "a2", "task": "t" }
        ] }),
    )
    .await
    .unwrap();
    let (p0, p1, p2) = (
        out.text.find("ZERO").unwrap(),
        out.text.find("ONE").unwrap(),
        out.text.find("TWO").unwrap(),
    );
    assert!(
        p0 < p1 && p1 < p2,
        "results must appear in task order:\n{}",
        out.text
    );
    assert!(out.text.contains("3/3 succeeded"), "{}", out.text);
}

#[tokio::test]
async fn parallel_does_not_early_abort_when_one_task_fails() {
    let factory = Factory::default()
        .reply("m0", "OK-ZERO")
        .reply("m2", "OK-TWO");
    let (tool, _d) = subagent(
        vec![
            def("a0", Some("m0"), None, Isolation::None),
            def("a2", Some("m2"), None, Isolation::None),
        ],
        factory,
    );
    let out = run(
        &tool,
        json!({ "tasks": [
            { "agent": "a0", "task": "t" },
            { "agent": "ghost", "task": "t" },
            { "agent": "a2", "task": "t" }
        ] }),
    )
    .await
    .unwrap();
    assert!(out.text.contains("OK-ZERO"), "{}", out.text);
    assert!(
        out.text.contains("OK-TWO"),
        "the task after the failure must still run: {}",
        out.text
    );
    assert!(out.text.contains("2/3 succeeded"), "{}", out.text);
    assert!(out.text.contains("failed"), "{}", out.text);
}

#[tokio::test]
async fn parallel_rejects_more_than_the_max_tasks() {
    let defs: Vec<AgentDef> = (0..(MAX_TASKS + 1))
        .map(|i| def(&format!("a{i}"), None, None, Isolation::None))
        .collect();
    let tasks: Vec<Value> = (0..(MAX_TASKS + 1))
        .map(|i| json!({ "agent": format!("a{i}"), "task": "t" }))
        .collect();
    let (tool, _d) = subagent(defs, Factory::default());
    let err = run(&tool, json!({ "tasks": tasks })).await.unwrap_err();
    assert!(err.to_string().contains("too many parallel tasks"), "{err}");
}

#[tokio::test]
async fn parallel_rejects_a_write_capable_agent_without_worktree_isolation() {
    // `bash` makes the agent write-capable; without `isolation: worktree` two such children would race.
    let (tool, _d) = subagent(
        vec![def(
            "builder",
            None,
            Some(vec!["read", "bash"]),
            Isolation::None,
        )],
        Factory::default(),
    );
    let err = run(
        &tool,
        json!({ "tasks": [{ "agent": "builder", "task": "build" }] }),
    )
    .await
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("isolation: worktree"), "{msg}");
    assert!(msg.contains("bash"), "must name the offending tool: {msg}");
}

#[tokio::test]
async fn parallel_allows_a_read_only_agent_without_isolation() {
    let (tool, _d) = subagent(
        vec![def(
            "reader",
            None,
            Some(vec!["read", "grep"]),
            Isolation::None,
        )],
        Factory::default().reply("parent-model", "read stuff"),
    );
    let out = run(
        &tool,
        json!({ "tasks": [{ "agent": "reader", "task": "look" }] }),
    )
    .await
    .unwrap();
    assert!(out.text.contains("read stuff"), "{}", out.text);
}

#[tokio::test]
async fn parallel_caps_a_huge_child_output_but_keeps_it_whole_in_details() {
    // The model-visible text is capped; the full output survives elsewhere. We can only see the text
    // here (details ride on ToolProgress, not ToolOutput), so assert the cap + marker.
    let big = "x".repeat(PER_TASK_CAP + 5_000);
    let (tool, _d) = subagent(
        vec![def("verbose", None, None, Isolation::None)],
        Factory::default().reply("parent-model", &big),
    );
    let out = run(
        &tool,
        json!({ "tasks": [{ "agent": "verbose", "task": "ramble" }] }),
    )
    .await
    .unwrap();
    assert!(
        out.text.contains("Output truncated"),
        "cap marker must be present"
    );
    assert!(
        out.text.len() < big.len(),
        "model-visible text must be shorter than the raw output"
    );
}

// ---- recursion ----

#[tokio::test]
async fn a_child_at_the_depth_cap_does_not_get_the_subagent_tool() {
    // A def that explicitly lists `subagent` still cannot exceed `max_depth`.
    let factory = Factory::default().reply("parent-model", "ok");
    let f2 = factory.clone();
    let (tool, _d) = subagent(
        vec![def(
            "recur",
            None,
            Some(vec!["read", "subagent"]),
            Isolation::None,
        )],
        factory,
    );
    run(&tool, json!({ "agent": "recur", "task": "x" }))
        .await
        .unwrap();
    // With DEFAULT_MAX_DEPTH = 1, the depth-1 child must NOT be told it has `subagent`.
    let system = f2.request_to("parent-model").system.unwrap_or_default();
    assert!(
        !system.contains("subagent"),
        "a depth-cap child must not advertise subagent: {system}"
    );
}

#[tokio::test]
async fn each_child_gets_a_distinct_cache_key() {
    // Sharing the parent's cache key would poison the prompt cache and contend the Codex WS pool. We
    // can't read the key off the request, but we can assert the seq counter advances by running two
    // children and checking they produced independent transports (a proxy that construction ran twice).
    let factory = Factory::default().reply("m0", "A").reply("m1", "B");
    let f2 = factory.clone();
    let (tool, _d) = subagent(
        vec![
            def("a0", Some("m0"), None, Isolation::None),
            def("a1", Some("m1"), None, Isolation::None),
        ],
        factory,
    );
    run(
        &tool,
        json!({ "chain": [{ "agent": "a0", "task": "t" }, { "agent": "a1", "task": "t" }] }),
    )
    .await
    .unwrap();
    assert_eq!(f2.call_models(), vec!["m0", "m1"]);
}

// ---- interactive approval inheritance ----

/// A gate that records what it was asked and answers with a fixed decision.
struct RecordingGate {
    answer: crate::approval::ApprovalDecision,
    seen: Arc<std::sync::Mutex<Vec<crate::approval::ApprovalRequest>>>,
}

#[async_trait]
impl crate::approval::ApprovalGate for RecordingGate {
    async fn request(
        &self,
        req: crate::approval::ApprovalRequest,
        _cancel: &agent_core::CancellationToken,
    ) -> Result<crate::approval::ApprovalDecision, crate::approval::ApprovalError> {
        self.seen.lock().unwrap().push(req);
        Ok(self.answer)
    }
}

/// A ctx whose children answer with one `bash` tool call, then a final message — enough to reach the
/// approval gate, which `Factory`'s text-only replies never do.
fn ctx_with_gate(
    defs: Vec<AgentDef>,
    command: &str,
    allow: bool,
) -> (
    Arc<SubagentCtx>,
    Arc<std::sync::Mutex<Vec<crate::approval::ApprovalRequest>>>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let gate = Arc::new(RecordingGate {
        answer: crate::approval::ApprovalDecision {
            allow,
            scope: crate::approval::ApprovalScope::Once,
        },
        seen: seen.clone(),
    });
    let args = json!({ "command": command }).to_string();
    let factory: TransportFactory = Arc::new(move |_model: &str| {
        Ok(Arc::new(agent_core::MockTransport::new(vec![
            agent_core::mock::turn::tool_call("t1", "bash", &args),
            agent_core::mock::turn::text("finished"),
        ])) as Arc<dyn ModelTransport>)
    });
    let ctx = Arc::new(SubagentCtx {
        factory,
        agents: Arc::new(defs),
        skills: Arc::new(Vec::new()),
        write_locks: Arc::new(agent_core::WriteLockRegistry::new()),
        mcp_tools: Vec::new(),
        tool_cfg: ChildToolConfig {
            image_auto_resize: true,
            ..Default::default()
        },
        cwd: dir.path().to_path_buf(),
        project_trusted: true,
        prompt_guidelines: Vec::new(),
        parent_model: "parent-model".to_string(),
        parent_cache_key: "parent-key".to_string(),
        parent_tools: vec!["bash".to_string()],
        deny_tool: Vec::new(),
        deny_bash_pattern: Vec::new(),
        deny_path: Vec::new(),
        child_max_steps: DEFAULT_CHILD_MAX_STEPS,
        max_depth: DEFAULT_MAX_DEPTH,
        approval: Some(crate::approval::ApprovalRuntime::new(
            gate,
            crate::approval::GatedSet::All,
        )),
    });
    (ctx, seen, dir)
}

#[tokio::test]
async fn a_child_tool_call_goes_through_the_parents_approval_gate() {
    // A child that could run `bash` without the human's approval *is* the bypass. `Agent::new` installs
    // `NoHooks`, so this only holds because `try_run_child` re-installs the composed hook.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("child-ran.txt");
    let command = format!("touch {}", marker.display());

    let (ctx, seen, _d) = ctx_with_gate(
        vec![def("worker", None, Some(vec!["bash"]), Isolation::None)],
        &command,
        false, // deny
    );
    let tool = Subagent::new(ctx);
    run(&tool, json!({ "agent": "worker", "task": "do it" }))
        .await
        .unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        1,
        "the child's `bash` call must have been gated"
    );
    assert_eq!(seen[0].tool, "bash");
    assert_eq!(seen[0].scope_key, format!("cmd:{command}"));
    assert!(!marker.exists(), "a denied child call must not have run");
}

#[tokio::test]
async fn a_childs_approval_request_names_which_child_is_asking() {
    // "Something wants to run `rm -rf`" is not an answerable question, and a parallel fan-out can have
    // several children blocked at once.
    let (ctx, seen, _d) = ctx_with_gate(
        vec![def("reviewer", None, Some(vec!["bash"]), Isolation::None)],
        "echo hi",
        true,
    );
    let tool = Subagent::new(ctx);
    run(&tool, json!({ "agent": "reviewer", "task": "look" }))
        .await
        .unwrap();

    let seen = seen.lock().unwrap();
    match &seen[0].origin {
        crate::approval::ApprovalOrigin::Subagent { agent, spawn_id } => {
            assert_eq!(agent, "reviewer");
            assert!(
                spawn_id.parse::<u64>().is_ok(),
                "spawn_id must be the child's own sequence number: {spawn_id}"
            );
        }
        other => panic!("a child must not report itself as the main agent: {other:?}"),
    }
}

#[tokio::test]
async fn an_allowed_child_call_actually_runs() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("child-ran.txt");
    let (ctx, seen, _d) = ctx_with_gate(
        vec![def("worker", None, Some(vec!["bash"]), Isolation::None)],
        &format!("touch {}", marker.display()),
        true,
    );
    let tool = Subagent::new(ctx);
    run(&tool, json!({ "agent": "worker", "task": "do it" }))
        .await
        .unwrap();
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(marker.exists(), "an approved child call must have run");
}
