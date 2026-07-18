//! `subagent` — delegate a task to a nested [`Agent`] with its own context window.
//!
//! The reference implementation (pi) ships this as an out-of-tree extension that spawns a `pi`
//! subprocess per child. That is a consequence of its extension boundary — extension code cannot reach
//! the agent loop, so the CLI was its only interface — not a design preference. We own the loop, so a
//! child is an in-process `Agent`: no MCP server reconnect per child, no lost HTTP connection pool, no
//! process-group killing, and child progress streams into the parent's own observers for free.
//!
//! # Three things this file must get right
//!
//! **The parent's policy must be inherited.** `Agent::new` installs `NoHooks`, so a child built without
//! [`ToolPolicy`] would run `bash` commands the parent's `--deny-bash-pattern` forbids. `subagent` would
//! then *be* the sandbox escape. Every child gets the parent's deny-lists, and every worktree child's
//! patch is re-checked against `--deny-path` before merge-back (`git apply` is not a tool call, so it
//! never reaches `before_tool_call`).
//!
//! **Parallelism has to live inside one tool call.** [`Tool::conservative_exclusive`] is `true` here —
//! a child's blast radius is as opaque as `bash`'s — and the loop drops a turn's concurrency to 1 as
//! soon as any call in it is exclusive. So N separate `subagent` calls in one turn would run strictly
//! serially. Fan-out is therefore an `tasks[]` array this tool spreads itself. pi's schema shape, arrived
//! at here for a reason pi never had.
//!
//! **Parallel writers need worktrees.** `bash` reports no `write_target`, so a shared
//! [`WriteLockRegistry`] provably cannot serialize two children's shell commands. A parallel task whose
//! agent can write must therefore declare `isolation: worktree`; the call is rejected otherwise.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_core::{
    Agent, AgentEvent, ModelTransport, Session, Tool, ToolError, ToolOutput, ToolProgress,
    ToolRegistry, WriteLockRegistry,
};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agents::{AgentDef, Isolation, find_by_name, last_assistant_text};
use crate::policy::ToolPolicy;
use crate::worktree::{self, ApplyOutcome, Worktree};

/// The tool's registered name. Referenced by the depth guard and by `resources.rs`, which only renders
/// `<available_agents>` when this tool is actually registered.
pub const NAME: &str = "subagent";

/// Hard cap on tasks in one `parallel` call — pi's own. Rejected outright rather than truncated: a
/// silently dropped task reads to the model as a completed one.
const MAX_TASKS: usize = 8;

/// How many children run at once inside a `parallel` call. pi's own value.
const MAX_CONCURRENCY: usize = 4;

/// Model-visible output budget per task in `parallel` mode. The full text always reaches the final
/// `ToolProgress` details, so nothing is lost — only what the *model* is asked to read is bounded.
const PER_TASK_CAP: usize = 50 * 1024;

/// A child at depth `d` gets its own `subagent` tool only when `d < max_depth`. With the default `1`,
/// children exist and grandchildren cannot. In-process there is no OS process limit to catch a fork
/// bomb, and pi has no depth guard at all.
pub const DEFAULT_MAX_DEPTH: u32 = 1;

/// Tools whose presence makes a child able to mutate the filesystem. `bash` counts even though it
/// declares no `write_target` — precisely *because* it declares none, which is what makes it unlockable.
const WRITE_CAPABLE_TOOLS: [&str; 3] = ["write", "edit", "bash"];

/// Disambiguates the per-child prompt-cache keys of a long-lived `serve` process.
static CHILD_SEQ: AtomicU64 = AtomicU64::new(0);

/// Builds the transport for a given model id.
///
/// A closure, not a config struct, for a reason: credentials and routing are **model-keyed**
/// (`gateway_credential::resolve_gateway_credential`, whose doc notes routing is never cached across a
/// model switch), so a child declaring its own `model:` cannot reuse the parent's `Arc<dyn
/// ModelTransport>`. And the two call sites build clients differently (`serve` injects a shared
/// `reqwest::Client`; `run` sets its own idle timeout), so a struct would force one shared build path
/// that immediately re-diverges with branches. Each site closes over its own builder instead.
pub type TransportFactory =
    Arc<dyn Fn(&str) -> Result<Arc<dyn ModelTransport>, String> + Send + Sync>;

/// The `bash`/`read` knobs a child's registry is rebuilt with, owned (not borrowed) because the context
/// outlives every call site's locals.
#[derive(Clone, Default)]
pub struct ChildToolConfig {
    pub bash_timeout_ms: Option<u64>,
    pub bash_shell_path: Option<String>,
    pub bash_command_prefix: Option<String>,
    pub image_auto_resize: bool,
    /// The parent's `web` egress policy, inherited so a child can't reach internal URLs the parent was
    /// forbidden — the same bypass class the deny-list/approval inheritance already guards.
    pub web_allow_private: bool,
    pub web_allow_hosts: Vec<String>,
    pub web_timeout_ms: Option<u64>,
}

/// Everything a [`Subagent`] needs to build and run a child. Shared by `Arc` across every depth level,
/// so a nested `Subagent` costs one pointer clone plus a `u32`.
pub struct SubagentCtx {
    pub factory: TransportFactory,
    pub agents: Arc<Vec<AgentDef>>,
    pub skills: Arc<Vec<crate::skills::Skill>>,
    /// The **parent's** registry. Passed to every child via `Agent::with_write_locks`, because
    /// `Agent::new`'s default is a fresh *private* one — leave it and N children editing one file race.
    pub write_locks: Arc<WriteLockRegistry>,
    /// Already-connected MCP tools. `Arc` clones share the live stdio child / HTTP connection, which is
    /// the entire reason an in-process child beats a subprocess.
    pub mcp_tools: Vec<Arc<dyn Tool>>,
    /// The **parent's** memory mounts, shared by `Arc` clone so a child reads and writes the *same*
    /// stores — the durable `/memories` project store *and* the `/session` working memory for this task.
    /// A subagent grounds its work in what the project already learned and in the parent's live session
    /// scratch, and its findings are visible to the parent and siblings. Shared as the `Arc`s (not
    /// re-derived from the child's cwd) so even a `worktree`-isolated child, whose cwd differs, still
    /// shares the parent's stores rather than private ones — including the `/session` dir cell that
    /// re-points on a serve session switch. Concurrent writes across the tree are serialized by each
    /// backend's own cross-process lock. Empty when the parent ran with `--no-memory`/`--no-tools`.
    pub memory_mounts: Vec<crate::memory::Mount>,
    pub tool_cfg: ChildToolConfig,
    pub cwd: PathBuf,
    pub project_trusted: bool,
    pub prompt_guidelines: Vec<String>,
    /// The model a child inherits when its definition names none.
    pub parent_model: String,
    /// The parent's own cache key, prefixed onto each child's so a `serve` session's children are
    /// traceable to it.
    pub parent_cache_key: String,
    /// The parent's effective tool names *after* `--tools`/`--exclude-tools` filtering, minus `subagent`
    /// itself. A definition with no `tools:` inherits exactly this: a parent launched with a restricted
    /// set cannot spawn a fully-armed child.
    pub parent_tools: Vec<String>,
    pub deny_tool: Vec<String>,
    pub deny_bash_pattern: Vec<String>,
    pub deny_path: Vec<String>,
    /// Opt-in cap on each child's model turns. `None` (the default) runs children unbounded, same as
    /// the parent — see `agent_core::Agent::with_max_steps` for why unbounded is the default.
    pub child_max_steps: Option<u32>,
    pub max_depth: u32,
    /// The parent's interactive approval gate, if it has one. Shared, not rebuilt: a child queues behind
    /// the same one-question-at-a-time lock its siblings do (eight parallel children must not raise eight
    /// simultaneous prompts), and reads the same [`SessionMemory`](crate::approval::SessionMemory) — so a
    /// decision the user already made is not asked again per child.
    ///
    /// That sharing **widens privilege across the subagent tree**: an "always allow" granted to one child
    /// applies to its siblings and to the parent for the rest of the session. Intended, and load-bearing
    /// for the "don't re-ask" property, but a real property nonetheless.
    pub approval: Option<crate::approval::ApprovalRuntime>,
}

/// The tool. `depth` is per-instance because [`Tool::run`] receives no ambient context — there is
/// nowhere else for a recursion guard to live.
pub struct Subagent {
    ctx: Arc<SubagentCtx>,
    depth: u32,
    description: String,
}

impl Subagent {
    /// The root instance, registered into the parent's registry.
    pub fn new(ctx: Arc<SubagentCtx>) -> Self {
        Self::at_depth(ctx, 0)
    }

    fn at_depth(ctx: Arc<SubagentCtx>, depth: u32) -> Self {
        Self {
            ctx,
            depth,
            description: describe(),
        }
    }
}

fn describe() -> String {
    format!(
        "Delegate work to a specialized agent running in its own context window; only its final message \
         comes back. Agent names and what each is for are listed in <available_agents>. Exactly one mode \
         per call: `single` ({{agent, task}}), `parallel` ({{tasks: [...]}} — up to {MAX_TASKS}, run \
         {MAX_CONCURRENCY} at a time, results in order), or `chain` ({{chain: [...]}} — sequential, where \
         each task's `{{previous}}` is replaced by the prior agent's final message). Use `chain` when a \
         later step needs an earlier one's conclusion but you do not want its intermediate work in your \
         own context. In `parallel`, an agent that can write files must declare `isolation: worktree`."
    )
}

/// One requested delegation.
#[derive(Clone, Debug)]
struct TaskSpec {
    agent: String,
    task: String,
    /// A JSON Schema the child must fill in and return via `structured_output` instead of ending in
    /// prose — so the parent gets `{"files_changed": [...]}` back rather than a paragraph it has to
    /// parse. Authored by the *parent model*, so it is never trusted: it is only checked for object-ness
    /// here, and a schema that fails to compile fails that one task rather than panicking or taking the
    /// whole fan-out down (see `try_run_child`).
    output_schema: Option<Value>,
}

/// Which of the three mutually exclusive shapes the model asked for.
enum Mode {
    Single(TaskSpec),
    Parallel(Vec<TaskSpec>),
    Chain(Vec<TaskSpec>),
}

/// What a finished child produced.
struct ChildOutcome {
    agent: String,
    task: String,
    /// The child's final assistant text, or the failure message when `ok` is false.
    text: String,
    ok: bool,
    /// Set when a worktree child's changes did not land cleanly: conflict markers, a policy refusal, or
    /// a preserved checkout the developer needs to look at.
    note: Option<String>,
}

impl ChildOutcome {
    fn failed(spec: &TaskSpec, message: impl Into<String>) -> Self {
        Self {
            agent: spec.agent.clone(),
            task: spec.task.clone(),
            text: message.into(),
            ok: false,
            note: None,
        }
    }
}

/// The one [`agent_core::AgentHooks`] a child installs: the parent's static deny-lists composed with its
/// interactive approval gate. `Agent::new` installs `NoHooks`, so without this a child ignores both.
struct ChildHooks {
    policy: ToolPolicy,
    approval: Option<crate::approval::ApprovalRuntime>,
    origin: crate::approval::ApprovalOrigin,
}

#[async_trait]
impl agent_core::AgentHooks for ChildHooks {
    async fn before_tool_call(
        &self,
        name: &str,
        input: &Value,
        session: &agent_core::Session,
        cancel: &agent_core::CancellationToken,
    ) -> Option<String> {
        crate::approval::gated_before_tool_call(
            &self.policy,
            self.approval.as_ref(),
            &self.origin,
            name,
            input,
            session,
            cancel,
        )
        .await
    }
}

/// Live per-task state, folded into a single coalesced progress snapshot.
#[derive(Clone)]
struct TaskState {
    agent: String,
    status: &'static str,
    step: u32,
    last_tool: Option<String>,
}

/// Shared progress board for one `subagent` call.
struct Board {
    states: std::sync::Mutex<Vec<TaskState>>,
    progress: ToolProgress,
    /// Throttles emission. Child token deltas are never forwarded at all (see `sink`), but tool-call
    /// boundaries from eight concurrent children still add up, and `serve`'s outbound channel is
    /// unbounded — a slow WebSocket consumer would otherwise grow it without limit.
    last_emit: std::sync::Mutex<std::time::Instant>,
}

/// Minimum wall-clock gap between progress snapshots. Terminal transitions bypass it.
const EMIT_THROTTLE: std::time::Duration = std::time::Duration::from_millis(100);

impl Board {
    fn new(progress: &ToolProgress, specs: &[TaskSpec]) -> Self {
        Self {
            states: std::sync::Mutex::new(
                specs
                    .iter()
                    .map(|s| TaskState {
                        agent: s.agent.clone(),
                        status: "queued",
                        step: 0,
                        last_tool: None,
                    })
                    .collect(),
            ),
            progress: progress.clone(),
            last_emit: std::sync::Mutex::new(
                std::time::Instant::now() - EMIT_THROTTLE - std::time::Duration::from_millis(1),
            ),
        }
    }

    fn update(&self, index: usize, f: impl FnOnce(&mut TaskState), force: bool) {
        {
            let Ok(mut states) = self.states.lock() else {
                return; // a poisoned board is not worth failing a subagent run over
            };
            if let Some(state) = states.get_mut(index) {
                f(state);
            }
        }
        if !force {
            let Ok(mut last) = self.last_emit.lock() else {
                return;
            };
            if last.elapsed() < EMIT_THROTTLE {
                return;
            }
            *last = std::time::Instant::now();
        }
        self.emit();
    }

    fn emit(&self) {
        let Ok(states) = self.states.lock() else {
            return;
        };
        let snapshot = states
            .iter()
            .map(|s| {
                let tool = s.last_tool.as_deref().unwrap_or("-");
                format!("[{}] {} step {} ({tool})", s.agent, s.status, s.step)
            })
            .collect::<Vec<_>>()
            .join("\n");
        let details = json!({
            "tasks": states.iter().map(|s| json!({
                "agent": s.agent,
                "status": s.status,
                "step": s.step,
                "last_tool": s.last_tool,
            })).collect::<Vec<_>>()
        });
        self.progress.emit(snapshot, Some(details));
    }
}

#[async_trait]
impl Tool for Subagent {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        const OUTPUT_SCHEMA_DESC: &str = "Optional JSON Schema (an object schema). When given, the \
             agent must return a payload conforming to it instead of a prose message, and its result \
             is that payload as compact JSON. Use this when you need to consume the answer \
             programmatically rather than read it.";
        let task_item = json!({
            "type": "object",
            "properties": {
                "agent": { "type": "string", "description": "Name from <available_agents>." },
                "task": { "type": "string", "description": "What this agent should do." },
                "output_schema": { "type": "object", "description": OUTPUT_SCHEMA_DESC }
            },
            "required": ["agent", "task"]
        });
        json!({
            "type": "object",
            "properties": {
                "agent": { "type": "string", "description": "Single mode: the agent to delegate to." },
                "task": { "type": "string", "description": "Single mode: what it should do." },
                "output_schema": { "type": "object", "description": OUTPUT_SCHEMA_DESC },
                "tasks": {
                    "type": "array",
                    "description": format!("Parallel mode: up to {MAX_TASKS} independent tasks. Results are returned in order."),
                    "items": task_item,
                },
                "chain": {
                    "type": "array",
                    "description": "Chain mode: sequential tasks; `{previous}` in a task is replaced by the prior agent's final message.",
                    "items": task_item,
                }
            }
        })
    }

    /// A child's blast radius is opaque — it may run `bash` against anything — exactly like the `bash`
    /// tool's own rationale. Marking the call exclusive stops the loop from running it concurrently with
    /// a sibling `edit` in the same turn. It is also *why* `tasks[]` exists: the loop caps an exclusive
    /// turn's concurrency at 1, so parallelism cannot come from the model emitting several calls.
    fn conservative_exclusive(&self) -> bool {
        true
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        // The loop always calls `run_streaming`; this exists for a host invoking the tool directly. The
        // throwaway progress sink discards updates and is never cancelled.
        let (tx, _rx) = futures::channel::mpsc::unbounded();
        let progress = ToolProgress::new(
            tx,
            String::new(),
            NAME.to_string(),
            agent_core::CancellationToken::new(),
        );
        self.dispatch(input, &progress).await
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        self.dispatch(input, progress).await
    }
}

impl Subagent {
    async fn dispatch(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        match self.parse_mode(&input)? {
            Mode::Single(spec) => self.run_single(spec, progress).await,
            Mode::Chain(specs) => self.run_chain(specs, progress).await,
            Mode::Parallel(specs) => self.run_parallel(specs, progress).await,
        }
    }

    /// Exactly one of the three shapes must be present. Zero is the model not having decided; two is it
    /// having decided twice, and silently preferring one would run work the model did not intend.
    fn parse_mode(&self, input: &Value) -> Result<Mode, ToolError> {
        let single = match (
            input.get("agent").and_then(Value::as_str),
            input.get("task").and_then(Value::as_str),
        ) {
            (Some(a), Some(t)) if !a.is_empty() && !t.is_empty() => Some(TaskSpec {
                agent: a.to_string(),
                task: t.to_string(),
                output_schema: parse_output_schema(input)?,
            }),
            _ => None,
        };
        let tasks = parse_task_array(input.get("tasks"))?;
        let chain = parse_task_array(input.get("chain"))?;

        let count = usize::from(single.is_some())
            + usize::from(tasks.is_some())
            + usize::from(chain.is_some());
        if count != 1 {
            return Err(ToolError::InvalidInput(format!(
                "provide exactly one of `agent`+`task`, `tasks`, or `chain` ({count} given). \
                 Available agents: {}",
                self.available_names()
            )));
        }
        if let Some(spec) = single {
            return Ok(Mode::Single(spec));
        }
        if let Some(specs) = chain {
            return Ok(Mode::Chain(specs));
        }
        let specs = tasks.unwrap_or_default();
        if specs.len() > MAX_TASKS {
            return Err(ToolError::InvalidInput(format!(
                "too many parallel tasks ({}); max is {MAX_TASKS}",
                specs.len()
            )));
        }
        // Checked before anything runs: a partial fan-out that then rejects task 5 has already spawned
        // four children and possibly written files.
        for spec in &specs {
            self.check_parallel_write_safety(spec)?;
        }
        Ok(Mode::Parallel(specs))
    }

    fn available_names(&self) -> String {
        if self.ctx.agents.is_empty() {
            return "(none defined)".to_string();
        }
        self.ctx
            .agents
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The tools a definition actually gets: its own `tools:` list, else the parent's effective set.
    fn effective_tools(&self, def: &AgentDef) -> Vec<String> {
        def.tools
            .clone()
            .unwrap_or_else(|| self.ctx.parent_tools.clone())
    }

    /// In `parallel`, a write-capable agent must be worktree-isolated. `bash` has no `write_target`, so
    /// the shared `WriteLockRegistry` cannot serialize two children's shell commands — no lock exists
    /// that would make this safe, only separate trees.
    fn check_parallel_write_safety(&self, spec: &TaskSpec) -> Result<(), ToolError> {
        // An unknown agent isn't a *batch* error — it becomes a failed section (`run_child` reports it),
        // so parallel keeps its "never early-abort" contract. Only a *known*, write-capable, unisolated
        // agent must fail the whole call, because that's a race we refuse to start.
        let Some(def) = find_by_name(&self.ctx.agents, &spec.agent) else {
            return Ok(());
        };
        if def.isolation == Isolation::Worktree {
            return Ok(());
        }
        let tools = self.effective_tools(def);
        let writers: Vec<&str> = WRITE_CAPABLE_TOOLS
            .iter()
            .copied()
            .filter(|w| tools.iter().any(|t| t == w))
            .collect();
        if writers.is_empty() {
            return Ok(());
        }
        Err(ToolError::InvalidInput(format!(
            "agent {:?} can write ({}) but does not declare `isolation: worktree`, so it cannot run in \
             `parallel` — two such agents would race on the same files. Use `chain`/`single` (which run \
             one at a time), give it `isolation: worktree`, or restrict its `tools:`.",
            spec.agent,
            writers.join(", ")
        )))
    }

    async fn run_single(
        &self,
        spec: TaskSpec,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        let board = Board::new(progress, std::slice::from_ref(&spec));
        let outcome = self.run_child(&spec, 0, &board).await;
        board.emit();
        if !outcome.ok {
            return Err(ToolError::Execution(outcome.text));
        }
        Ok(ToolOutput::text(with_note(outcome.text, outcome.note)))
    }

    /// Sequential. Each step's `{previous}` becomes the prior child's final assistant text — its
    /// conclusion, never its tool history, which is the point: the parent's context stays clean and so
    /// does the next child's.
    async fn run_chain(
        &self,
        specs: Vec<TaskSpec>,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        let board = Board::new(progress, &specs);
        let mut previous = String::new();
        let mut done: Vec<ChildOutcome> = Vec::new();

        for (index, spec) in specs.iter().enumerate() {
            let resolved = TaskSpec {
                agent: spec.agent.clone(),
                task: spec.task.replace("{previous}", &previous),
                output_schema: spec.output_schema.clone(),
            };
            let outcome = self.run_child(&resolved, index, &board).await;
            if !outcome.ok {
                board.emit();
                let partials = render_parallel(&done);
                return Err(ToolError::Execution(format!(
                    "chain stopped at step {} ({}): {}\n\n--- completed steps ---\n{partials}",
                    index + 1,
                    outcome.agent,
                    outcome.text
                )));
            }
            previous = outcome.text.clone();
            done.push(outcome);
        }
        board.emit();

        let last = done.pop().ok_or_else(|| {
            ToolError::InvalidInput("`chain` must contain at least one task".to_string())
        })?;
        Ok(ToolOutput::text(with_note(last.text, last.note)))
    }

    /// Bounded fan-out. `buffered` (not `buffer_unordered`) runs up to [`MAX_CONCURRENCY`] at a time
    /// while yielding results in the order the tasks were given — a task's position in the output is the
    /// only handle the model has on it. No early abort: a failed task becomes a failed *section*, and
    /// its siblings still finish.
    async fn run_parallel(
        &self,
        specs: Vec<TaskSpec>,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        use futures::stream::StreamExt;

        let board = Board::new(progress, &specs);
        // Build the child futures eagerly rather than via a `.map()` closure inside `stream::iter`: the
        // closure form makes the borrow checker try to prove the closure general over every lifetime of
        // the borrowed `spec`, which it can't (HRTB inference fails), whereas a plain loop pushes
        // already-formed futures with concrete lifetimes.
        let mut futs = Vec::with_capacity(specs.len());
        for (index, spec) in specs.iter().enumerate() {
            futs.push(self.run_child(spec, index, &board));
        }
        let outcomes: Vec<ChildOutcome> = futures::stream::iter(futs)
            .buffered(MAX_CONCURRENCY)
            .collect()
            .await;

        // The final snapshot carries every task's *full* output, uncapped — the model-visible text below
        // is truncated, but a client rendering `ToolProgress` details never loses anything.
        board.emit();
        progress.emit(
            format!("{} tasks complete", outcomes.len()),
            Some(json!({
                "tasks": outcomes.iter().map(|o| json!({
                    "agent": o.agent,
                    "task": o.task,
                    "ok": o.ok,
                    "output": o.text,
                    "note": o.note,
                })).collect::<Vec<_>>()
            })),
        );
        Ok(ToolOutput::text(render_parallel(&outcomes)))
    }

    /// Build and run one child to completion. Never returns `Err`: a failure is a [`ChildOutcome`] so
    /// `parallel` can report it alongside its siblings' successes.
    async fn run_child(&self, spec: &TaskSpec, index: usize, board: &Board) -> ChildOutcome {
        board.update(index, |s| s.status = "running", true);
        match self.try_run_child(spec, index, board).await {
            Ok(outcome) => {
                let status = if outcome.ok { "done" } else { "failed" };
                board.update(index, |s| s.status = status, true);
                outcome
            }
            Err(message) => {
                board.update(index, |s| s.status = "failed", true);
                ChildOutcome::failed(spec, message)
            }
        }
    }

    async fn try_run_child(
        &self,
        spec: &TaskSpec,
        index: usize,
        board: &Board,
    ) -> Result<ChildOutcome, String> {
        let def = find_by_name(&self.ctx.agents, &spec.agent).ok_or_else(|| {
            format!(
                "unknown agent {:?}. Available agents: {}",
                spec.agent,
                self.available_names()
            )
        })?;

        // A worktree, when asked for. Held for the whole call: dropping it removes the checkout, which is
        // exactly what should happen if the parent cancels us mid-run.
        let (root, worktree) = match def.isolation {
            Isolation::None => (self.ctx.cwd.clone(), None),
            Isolation::Worktree => {
                let repo_root = worktree::preflight(&self.ctx.cwd).await?;
                let wt = Worktree::create(&repo_root, &format!("{}-{index}", def.name)).await?;
                (wt.path().to_path_buf(), Some((wt, repo_root)))
            }
        };

        let mut registry = self.build_child_registry(def, &root);
        // The parent asked for typed data back. Registered *after* `build_child_registry`'s own
        // `--tools` filter, like every other explicitly-requested tool, and *before* the system prompt is
        // built so `has_structured_output` picks it up and the child is actually told the contract.
        //
        // `output_schema` was written by the parent *model*. A schema that won't compile fails this one
        // task — the parent gets a message it can act on and its siblings keep running — rather than
        // panicking or aborting the whole fan-out.
        let output_slot = crate::tools::structured_output::OutputSlot::new();
        if let Some(schema) = &spec.output_schema {
            match crate::tools::structured_output::StructuredOutput::new(
                schema.clone(),
                None,
                output_slot.clone(),
            ) {
                Ok(tool) => {
                    registry.register(Arc::new(tool));
                }
                Err(e) => {
                    return Ok(ChildOutcome::failed(
                        spec,
                        format!("invalid output_schema: {e}"),
                    ));
                }
            }
        }
        // The current shared indices, injected into the child's prompt so it starts aware of project
        // memory *and* the parent's session scratch (the same session-start recall the parent gets), read
        // from the shared mounts.
        let memory_sections = crate::memory::mount_sections(&self.ctx.memory_mounts).await;
        let system = self.build_child_system_prompt(def, &registry, &root, &memory_sections);
        let model = def
            .model
            .clone()
            .unwrap_or_else(|| self.ctx.parent_model.clone());
        let transport = (self.ctx.factory)(&model)?;

        // One id per spawned child: it disambiguates the prompt-cache key *and* names this child in an
        // approval request, so a UI can say "reviewer#7 wants to run bash" rather than "something does".
        let spawn_id = CHILD_SEQ.fetch_add(1, Ordering::Relaxed);

        let mut agent = Agent::new(transport, model)
            .with_tools(registry)
            .with_system(system)
            .with_max_steps(self.ctx.child_max_steps)
            // The parent's registry. Without this, `Agent::new` hands the child a private one and two
            // children editing the same path do not serialize.
            .with_write_locks(self.ctx.write_locks.clone())
            // A distinct key per child. Sharing the parent's would both poison the Anthropic prompt
            // cache and contend for one pooled Codex WebSocket connection, which is keyed on it.
            .with_cache_key(format!("{}:sub:{spawn_id}", self.ctx.parent_cache_key));

        // MANDATORY. `Agent::new` installs `NoHooks`; without this the child ignores every
        // `--deny-tool`/`--deny-bash-pattern`/`--deny-path` the parent was launched with, and `subagent`
        // becomes a policy bypass. Rooted at the child's own root so a relative path is checked as the
        // path that will actually be written.
        let policy = ToolPolicy::from_lists(
            &self.ctx.deny_tool,
            &self.ctx.deny_bash_pattern,
            &self.ctx.deny_path,
        )
        .with_root(&root);
        // The interactive gate is inherited for exactly the same reason the deny-lists are: a child that
        // could run `bash` without the human's approval *is* the bypass. `Agent::with_hooks` holds one
        // object, so the two compose here rather than stacking.
        if !policy.is_empty() || self.ctx.approval.is_some() {
            agent = agent.with_hooks(Arc::new(ChildHooks {
                policy: policy.clone(),
                approval: self.ctx.approval.clone(),
                origin: crate::approval::ApprovalOrigin::Subagent {
                    agent: def.name.clone(),
                    spawn_id: spawn_id.to_string(),
                },
            }));
        }
        // Deliberately no `.with_checkpoint_hook`: `Agent::new`'s `NoCheckpoint` default is correct. A
        // child inheriting the parent's would persist its turns into the parent's session store.

        let mut session = Session::new();
        session.user(spec.task.clone());

        let run = agent.run_events(&mut session, |event| {
            match event {
                AgentEvent::TurnStart { step } => board.update(index, |s| s.step = step, false),
                AgentEvent::ToolStart { name, .. } => {
                    board.update(index, |s| s.last_tool = Some(name), false);
                }
                AgentEvent::Error { message } => {
                    tracing::warn!(agent = %spec.agent, "subagent error event: {message}");
                }
                // Everything else, and `Stream` above all, is deliberately dropped. Forwarding a child's
                // token deltas would fan eight streams into `serve`'s unbounded outbound channel.
                _ => {}
            }
        });

        let result = run.await;

        // Read before `merge_back` below, so a worktree conflict note rides `outcome.note` rather than
        // being appended to what is now a machine-parseable JSON payload.
        let mut outcome = match result {
            Ok(()) => match &spec.output_schema {
                // The parent asked for a typed answer; the child's prose is not it. A child that ended
                // without calling `structured_output` did not do what it was told, and reporting that as
                // success would hand the parent a paragraph where it expected an object.
                Some(_) => match output_slot.take() {
                    Some(value) => ChildOutcome {
                        agent: spec.agent.clone(),
                        task: spec.task.clone(),
                        text: value.to_string(),
                        ok: true,
                        note: None,
                    },
                    None => ChildOutcome::failed(
                        spec,
                        format!(
                            "the agent finished without calling `{}`, so it produced no structured output",
                            crate::tools::structured_output::NAME
                        ),
                    ),
                },
                None => ChildOutcome {
                    agent: spec.agent.clone(),
                    task: spec.task.clone(),
                    text: nonempty(last_assistant_text(&session)),
                    ok: true,
                    note: None,
                },
            },
            Err(e) => ChildOutcome::failed(spec, format!("agent run failed: {e}")),
        };

        if let Some((wt, repo_root)) = worktree {
            self.merge_back(wt, &repo_root, &policy, &mut outcome).await;
        }
        Ok(outcome)
    }

    /// Land a worktree child's changes in the parent tree, or explain why they did not.
    ///
    /// A clean apply removes the checkout. A conflict, a policy refusal, or a git failure preserves it and
    /// names its path — the child's work is the only copy, and destroying it to keep the temp directory
    /// tidy would be the wrong trade.
    async fn merge_back(
        &self,
        wt: Worktree,
        repo_root: &Path,
        policy: &ToolPolicy,
        outcome: &mut ChildOutcome,
    ) {
        if !outcome.ok {
            // A failed child's half-finished edits are not worth merging, but they are worth keeping.
            let path = wt.preserve();
            outcome.note = Some(format!("worktree preserved at {}", path.display()));
            return;
        }
        let delta = match wt.child_delta().await {
            Ok(delta) => delta,
            Err(e) => {
                let path = wt.preserve();
                outcome.note = Some(format!(
                    "could not read changes ({e}); worktree preserved at {}",
                    path.display()
                ));
                return;
            }
        };
        if delta.is_empty() {
            // "Auto-cleaned if unchanged" — the same contract Claude Code's worktree isolation offers.
            let _ = wt.remove();
            return;
        }
        // Serialize the actual apply. `child_delta` above is per-worktree and safe to run concurrently,
        // but `git apply` mutates the shared repo's working tree and index — two children applying at
        // once would collide on `index.lock`, failing a merge that would otherwise have succeeded (or
        // conflicted cleanly). The shared `WriteLockRegistry` (the same one every child's file writes
        // key on) serializes them on a sentinel key that no real path can collide with.
        let _merge_guard = self.ctx.write_locks.lock("\0subagent-merge-back").await;
        match worktree::apply_patch(repo_root, &delta, policy.denied_paths()).await {
            Ok(ApplyOutcome::Clean) => {
                let _ = wt.remove();
            }
            Ok(ApplyOutcome::Conflicted { files }) => {
                let path = wt.preserve();
                outcome.note = Some(format!(
                    "changes merged WITH CONFLICTS in: {}. Conflict markers are in the working tree; \
                     resolve them. The agent's own version is at {}",
                    files.join(", "),
                    path.display()
                ));
            }
            Err(e) => {
                let path = wt.preserve();
                outcome.ok = false;
                outcome.text = format!("{}\n\n{e}", outcome.text);
                outcome.note = Some(format!(
                    "changes NOT merged; worktree preserved at {}",
                    path.display()
                ));
            }
        }
    }

    /// The child's tool set: rooted at `root`, restricted to its effective tools, and carrying a
    /// depth-incremented `subagent` only when the definition asked for one *and* the cap allows it.
    fn build_child_registry(&self, def: &AgentDef, root: &Path) -> ToolRegistry {
        let mut registry = super::default_registry_with_config(&super::ToolConfig {
            bash_timeout_ms: self.ctx.tool_cfg.bash_timeout_ms,
            bash_shell_path: self.ctx.tool_cfg.bash_shell_path.as_deref(),
            bash_command_prefix: self.ctx.tool_cfg.bash_command_prefix.as_deref(),
            image_auto_resize: self.ctx.tool_cfg.image_auto_resize,
            root: root.to_path_buf(),
            mcp_tools: &self.ctx.mcp_tools,
            web_allow_private: self.ctx.tool_cfg.web_allow_private,
            web_allow_hosts: &self.ctx.tool_cfg.web_allow_hosts,
            web_timeout_ms: self.ctx.tool_cfg.web_timeout_ms,
        });
        let allow = self.effective_tools(def);
        super::apply_filter(&mut registry, Some(&allow), None, false);

        // `parent_tools` never contains `subagent`, so a child only gets one by explicitly asking. The
        // depth cap is the backstop for exactly that case.
        let child_depth = self.depth + 1;
        let wants_subagent = allow.iter().any(|t| t == NAME);
        if wants_subagent && child_depth < self.ctx.max_depth {
            registry.register(Arc::new(Subagent::at_depth(self.ctx.clone(), child_depth)));
        }
        // Shared memory: registered *after* the child's `--tools` filter (like the parent's own memory,
        // and like `subagent`/`structured_output`), so a scoped child still has its project memory. It
        // holds the parent's mount `Arc`s, so the whole subagent tree reads and writes the same stores —
        // durable `/memories` and the task's `/session` — even a `worktree`-isolated child whose cwd would
        // otherwise key private ones.
        if !self.ctx.memory_mounts.is_empty() {
            registry.register(Arc::new(crate::tools::memory::Memory::mounted(
                self.ctx.memory_mounts.clone(),
            )));
        }
        registry
    }

    /// The definition's body is *appended* to the built-in base (pi passes it via
    /// `--append-system-prompt`), not substituted for it — the base is recomputed from the child's own
    /// restricted registry, so a child given `tools: read,grep` is told it has `read, grep` and nothing
    /// more. `cwd` is the child's root, so `dynamic_footer` reports the worktree as its working directory.
    fn build_child_system_prompt(
        &self,
        def: &AgentDef,
        registry: &ToolRegistry,
        root: &Path,
        memory_sections: &[(crate::memory::MountKind, String)],
    ) -> String {
        let default_base =
            crate::resources::default_system_prompt(registry, &self.ctx.prompt_guidelines);
        let body = def.system.trim();
        crate::resources::build_system_prompt(&crate::resources::PromptOptions {
            base: None,
            default_base: &default_base,
            append: (!body.is_empty()).then_some(body),
            cwd: root,
            include_context_files: true,
            skills: &self.ctx.skills,
            has_read: registry.get("read").is_some(),
            has_todo: registry.get(crate::tools::todo::NAME).is_some(),
            has_structured_output: registry
                .get(crate::tools::structured_output::NAME)
                .is_some(),
            // A child carries the shared `memory` tool whenever the parent has a backend (see
            // `build_child_registry`), so the section — guidance plus the current shared index — appears
            // for it exactly as for the parent.
            has_memory: registry.get(crate::tools::memory::NAME).is_some(),
            memory_sections,
            // A child never has `subagent` unless its definition asked for it and the depth cap allowed
            // it; there is no point listing agents it cannot delegate to.
            agents: if registry.get(NAME).is_some() {
                &self.ctx.agents
            } else {
                &[]
            },
            project_trusted: self.ctx.project_trusted,
        })
    }
}

/// Parse a `tasks`/`chain` array, rejecting a present-but-malformed one rather than silently treating it
/// as absent (which would surface as the far more confusing "provide exactly one mode").
fn parse_task_array(value: Option<&Value>) -> Result<Option<Vec<TaskSpec>>, ToolError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let items = value
        .as_array()
        .ok_or_else(|| ToolError::InvalidInput("`tasks`/`chain` must be an array".to_string()))?;
    if items.is_empty() {
        return Ok(None);
    }
    let mut specs = Vec::with_capacity(items.len());
    for item in items {
        let agent = item
            .get("agent")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ToolError::InvalidInput("each task needs an `agent`".to_string()))?;
        let task = item
            .get("task")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ToolError::InvalidInput("each task needs a `task`".to_string()))?;
        specs.push(TaskSpec {
            agent: agent.to_string(),
            task: task.to_string(),
            output_schema: parse_output_schema(item)?,
        });
    }
    Ok(Some(specs))
}

/// The optional `output_schema` on a task, checked only for the one thing that can be checked without
/// compiling it: a tool's arguments are always a JSON object, so anything else can never be satisfied.
/// Absent or `null` means "answer in prose", the default.
fn parse_output_schema(item: &Value) -> Result<Option<Value>, ToolError> {
    match item.get("output_schema") {
        None | Some(Value::Null) => Ok(None),
        Some(v) if v.is_object() => Ok(Some(v.clone())),
        Some(_) => Err(ToolError::InvalidInput(
            "`output_schema` must be a JSON Schema object".to_string(),
        )),
    }
}

/// A child that ended without producing assistant text (it exhausted `max_steps` mid-tool-call, say) must
/// not hand the parent an empty string — that reads as success with nothing to say.
fn nonempty(text: String) -> String {
    if text.trim().is_empty() {
        "(the agent produced no final message)".to_string()
    } else {
        text
    }
}

fn with_note(text: String, note: Option<String>) -> String {
    match note {
        Some(note) => format!("{text}\n\n[{note}]"),
        None => text,
    }
}

/// Truncate on a char boundary, appending a marker that names how much was cut and where to find it.
fn cap(text: &str) -> String {
    if text.len() <= PER_TASK_CAP {
        return text.to_string();
    }
    let mut end = PER_TASK_CAP;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let omitted = text.len() - end;
    format!(
        "{}\n\n[Output truncated: {omitted} bytes omitted. Full output is in the tool's progress details.]",
        &text[..end]
    )
}

/// The model-visible digest for `parallel` (and for a `chain`'s completed prefix).
fn render_parallel(outcomes: &[ChildOutcome]) -> String {
    if outcomes.is_empty() {
        return "(no tasks ran)".to_string();
    }
    let succeeded = outcomes.iter().filter(|o| o.ok).count();
    let mut out = format!("{succeeded}/{} succeeded\n", outcomes.len());
    for outcome in outcomes {
        let status = if outcome.ok { "completed" } else { "failed" };
        out.push_str("\n---\n\n");
        out.push_str(&format!("### [{}] {status}\n\n", outcome.agent));
        out.push_str(&cap(&outcome.text));
        if let Some(note) = &outcome.note {
            out.push_str(&format!("\n\n[{note}]"));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests;
