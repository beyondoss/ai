//! Beyond agent harness — CLI.
//!
//! `run` drives a one-shot coding task to completion through the gateway. `serve` exposes the
//! headless control protocol (newline-delimited JSON over stdio). `tools` lists the advertised tool
//! set. Model traffic always flows through the gateway (`AI_GATEWAY_URL`) authenticated with a
//! `bai_v1` key (`AI_AGENT_KEY`).

// Unit tests assert preconditions with `.unwrap()`; allow that under `test` (matches the gateway and
// agent-core crate roots). Production paths stay panic-free per the workspace lints.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

// mimalloc, matching `edge`/`logfwd`/`orchestrator`/`tunnel` (the fleet default); it also fixes
// musl's slow multithreaded malloc, which matters for the static musl build of this CLI.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use agent_core::{Agent, GatewayClient, Session, StreamEvent};
use beyond_ai_agent::{serve, tools};
use clap::{Parser, Subcommand};

/// Default model when neither `--model` nor `AI_AGENT_MODEL` is set.
const DEFAULT_MODEL: &str = "claude-opus-4-8";
/// Default gateway base URL.
const DEFAULT_GATEWAY: &str = "http://ai.internal";

/// The agent's base identity/instructions. The tool list is generated from `registry` — the tools this
/// process actually registered, after any `--tools`/`--exclude-tools`/`--no-tools` filtering — rather
/// than hand-listed as a static string or assumed to be the full default set. A prior hardcoded version
/// silently omitted the Beyond platform tools (fork/sync/logs) entirely, and a version that always
/// listed `default_registry()` regardless of filtering would claim tools a restricted agent doesn't
/// actually have, inviting the model to call one that gets rejected.
fn default_system_prompt(registry: &agent_core::ToolRegistry) -> String {
    let names: Vec<String> = registry.definitions().into_iter().map(|d| d.name).collect();
    format!(
        "You are the Beyond coding agent. You operate inside a real working directory with tools: {}. \
         Use them to accomplish the user's task directly — inspect before you change, make minimal \
         edits, and verify your work. Be concise.",
        names.join(", ")
    )
}

/// Parse `--reasoning-effort`'s value into the wire-neutral [`agent_core::ReasoningEffort`] enum.
fn parse_reasoning_effort(s: &str) -> Result<agent_core::ReasoningEffort, String> {
    use agent_core::ReasoningEffort::*;
    match s {
        "minimal" => Ok(Minimal),
        "low" => Ok(Low),
        "medium" => Ok(Medium),
        "high" => Ok(High),
        "xhigh" => Ok(XHigh),
        other => Err(format!(
            "invalid reasoning effort {other:?}; expected one of minimal/low/medium/high/xhigh"
        )),
    }
}

#[derive(Parser)]
#[command(name = "beyond-ai-agent", version, about = "Beyond agent harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a one-shot agent task to completion, streaming output to stdout.
    Run {
        /// The task prompt for the agent.
        task: String,
        /// Model id (default `claude-opus-4-8`, or `AI_AGENT_MODEL`).
        #[arg(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL (default `http://ai.internal`, or `AI_GATEWAY_URL`).
        #[arg(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// Virtual key (`bai_v1…`) or BYO provider key. Required; or set `AI_AGENT_KEY`.
        #[arg(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Max loop iterations before bailing.
        #[arg(long, default_value_t = agent_core::agent::DEFAULT_MAX_STEPS)]
        max_steps: u32,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one.
        #[arg(long, default_value_t = false)]
        trust_project: bool,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Combine with `--exclude-tools` to carve one back out of the allow-list.
        #[arg(long, value_delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files.
        #[arg(long, value_delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation run. Wins over `--tools`/`--exclude-tools`.
        #[arg(long, default_value_t = false)]
        no_tools: bool,
    },
    /// Run the headless agent server: a newline-delimited JSON control protocol over stdio.
    Serve {
        /// Model id (default `claude-opus-4-8`, or `AI_AGENT_MODEL`).
        #[arg(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL (default `http://ai.internal`, or `AI_GATEWAY_URL`).
        #[arg(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// Virtual key (`bai_v1…`) or BYO provider key. Required; or set `AI_AGENT_KEY`.
        #[arg(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Persist one session to this JSONL file so a later `serve` reattaches with the transcript.
        #[arg(long, env = "AI_AGENT_SESSION_FILE")]
        session_file: Option<String>,
        /// Persist many sessions under this directory (enables list/switch/fork/name commands).
        #[arg(long, env = "AI_AGENT_SESSION_DIR")]
        session_dir: Option<String>,
        /// Skip persistence entirely, even without `--session-file`/`--session-dir`. Without this,
        /// `serve` defaults to `~/.claude/sessions/<encoded-cwd>/` rather than silently running
        /// in-memory-only — pass this for the rare case that's genuinely what you want (e.g. a
        /// short-lived test harness).
        #[arg(long, default_value_t = false)]
        no_session_persistence: bool,
        /// Max loop iterations per prompt before bailing.
        #[arg(long, default_value_t = agent_core::agent::DEFAULT_MAX_STEPS)]
        max_steps: u32,
        /// Replace the built-in base system prompt entirely.
        #[arg(long, env = "AI_AGENT_SYSTEM_PROMPT")]
        system_prompt: Option<String>,
        /// Append extra instructions after the base system prompt.
        #[arg(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Option<String>,
        /// Do not discover/inject AGENTS.md / CLAUDE.md project-instruction files.
        #[arg(long, default_value_t = false)]
        no_context_files: bool,
        /// Model context window (tokens); the loop summarizes older turns to stay below it. Defaults
        /// to the model's own capability-table window (see `agent_core::models::capabilities`) — only
        /// pass this to pin a fixed budget that survives a `set_model` switch to a different model.
        #[arg(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Use the 1-hour prompt-cache TTL (vs 5 minutes); helps when turns are spaced out.
        #[arg(long, default_value_t = false)]
        cache_long: bool,
        /// Enable extended thinking with this token budget (must be below the per-turn max tokens).
        #[arg(long)]
        thinking: Option<u32>,
        /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
        /// reasoning models via `reasoning_effort`; Anthropic adaptive-thinking models via
        /// `output_config.effort`). One of minimal/low/medium/high/xhigh. Ignored by models that take
        /// neither shape.
        #[arg(long, value_parser = parse_reasoning_effort)]
        reasoning_effort: Option<agent_core::ReasoningEffort>,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one.
        #[arg(long, default_value_t = false)]
        trust_project: bool,
        /// Compaction headroom (tokens) reserved below the context window before it fires. Defaults to
        /// `CompactionConfig::default()`'s 24,000.
        #[arg(long, env = "AI_AGENT_COMPACTION_RESERVE_TOKENS")]
        compaction_reserve_tokens: Option<u32>,
        /// Roughly how many tokens of recent conversation compaction keeps verbatim. Defaults to
        /// `CompactionConfig::default()`'s 40,000.
        #[arg(long, env = "AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS")]
        compaction_keep_recent_tokens: Option<u32>,
        /// How many times to retry a gateway request that fails before the first response byte
        /// arrives. Defaults to 3.
        #[arg(long, env = "AI_AGENT_RETRY_MAX_RETRIES")]
        retry_max_retries: Option<u32>,
        /// Base of the exponential backoff between those retries, in milliseconds. Defaults to 250.
        #[arg(long, env = "AI_AGENT_RETRY_BASE_DELAY_MS")]
        retry_base_delay_ms: Option<u64>,
        /// Default `bash` command timeout (ms) when the model omits `timeout_ms`. Defaults to 1,800,000
        /// (30 minutes) — see `tools::bash`'s doc comment for why this deliberately deviates from the
        /// reference agent's no-default.
        #[arg(long, env = "AI_AGENT_BASH_TIMEOUT_MS")]
        bash_timeout_ms: Option<u64>,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Fixed for the process, like `--system-prompt`; survives `set_model`/`set_thinking` rebuilds.
        #[arg(long, env = "AI_AGENT_TOOLS", value_delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files.
        #[arg(long, env = "AI_AGENT_EXCLUDE_TOOLS", value_delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation session. Wins over `--tools`/`--exclude-tools`.
        #[arg(long, default_value_t = false)]
        no_tools: bool,
    },
    /// List the tools the agent advertises to the model.
    Tools,
    /// Record `path` (default: the current directory) in the persisted project-trust allowlist
    /// (`~/.claude/trusted-projects.json`), so its `.claude/SYSTEM.md` is honored on future runs
    /// without needing `--trust-project` every time. Idempotent — trusting an already-trusted path is
    /// a no-op.
    Trust {
        /// The project directory to trust. Defaults to the current directory.
        path: Option<String>,
    },
    /// Record `path` (default: the current directory) as explicitly *untrusted*, overriding any
    /// trust it would otherwise inherit from a trusted ancestor directory. Idempotent.
    Untrust {
        /// The project directory to untrust. Defaults to the current directory.
        path: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    match Cli::parse().command {
        Command::Run {
            task,
            model,
            gateway_url,
            key,
            max_steps,
            trust_project,
            tools,
            exclude_tools,
            no_tools,
        } => {
            run_task(
                task,
                model,
                gateway_url,
                key,
                max_steps,
                trust_project,
                tools,
                exclude_tools,
                no_tools,
            )
            .await?;
        }
        Command::Serve {
            model,
            gateway_url,
            key,
            session_file,
            session_dir,
            no_session_persistence,
            max_steps,
            system_prompt,
            append_system_prompt,
            no_context_files,
            context_window,
            cache_long,
            thinking,
            reasoning_effort,
            trust_project,
            compaction_reserve_tokens,
            compaction_keep_recent_tokens,
            retry_max_retries,
            retry_base_delay_ms,
            bash_timeout_ms,
            tools,
            exclude_tools,
            no_tools,
        } => {
            let key = key
                .ok_or("no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key)")?;
            let system = system_prompt.unwrap_or_else(|| {
                let mut reg = tools::default_registry_with(bash_timeout_ms);
                tools::apply_filter(
                    &mut reg,
                    tools.as_deref(),
                    exclude_tools.as_deref(),
                    no_tools,
                );
                default_system_prompt(&reg)
            });
            serve::serve(serve::ServeConfig {
                gateway: gateway_url.unwrap_or_else(|| DEFAULT_GATEWAY.to_string()),
                key,
                model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
                max_steps,
                system,
                append_system: append_system_prompt,
                context_files: !no_context_files,
                session_file,
                session_dir,
                no_session_persistence,
                context_window,
                cache_long,
                thinking,
                reasoning_effort,
                trust_project,
                compaction_reserve_tokens,
                compaction_keep_recent_tokens,
                retry_max_retries,
                retry_base_delay_ms: retry_base_delay_ms.map(std::time::Duration::from_millis),
                bash_timeout_ms,
                tools,
                exclude_tools,
                no_tools,
            })
            .await?;
            // `serve` reads stdin via `tokio::io::stdin()`, which parks a dedicated blocking OS
            // thread doing a blocking read for the life of the process. If stdin is never closed
            // (a client that doesn't hang up, or — the case this matters for — a SIGTERM/SIGINT
            // whose handler cancels the run and returns without stdin ever reaching EOF), that
            // thread is still parked here even though all async work is done. Falling through to
            // `#[tokio::main]`'s implicit runtime shutdown would then hang indefinitely: dropping
            // a `Runtime` waits for every outstanding blocking task, and a parked stdin read never
            // completes on its own. Exit explicitly instead — `serve` has already drained,
            // persisted, and flushed everything before returning, so there's nothing left to lose.
            std::process::exit(0);
        }
        Command::Tools => {
            let reg = tools::default_registry();
            println!("{} tools:\n", reg.len());
            println!("{}", serde_json::to_string_pretty(&reg.definitions())?);
        }
        Command::Trust { path } => {
            let dir = match path {
                Some(p) => PathBuf::from(p),
                None => std::env::current_dir()?,
            };
            let mut store = beyond_ai_agent::trust_store::TrustStore::open_default();
            store.trust(&dir)?;
            println!("trusted: {}", dir.display());
        }
        Command::Untrust { path } => {
            let dir = match path {
                Some(p) => PathBuf::from(p),
                None => std::env::current_dir()?,
            };
            let mut store = beyond_ai_agent::trust_store::TrustStore::open_default();
            store.distrust(&dir)?;
            println!("untrusted: {}", dir.display());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_task(
    task: String,
    model: Option<String>,
    gateway_url: Option<String>,
    key: Option<String>,
    max_steps: u32,
    trust_project: bool,
    tools_allow: Option<Vec<String>>,
    tools_exclude: Option<Vec<String>>,
    no_tools: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let gateway = gateway_url.unwrap_or_else(|| DEFAULT_GATEWAY.to_string());
    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let key =
        key.ok_or("no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key)")?;

    let cwd = std::env::current_dir().unwrap_or_default();
    let project_trusted =
        trust_project || beyond_ai_agent::trust_store::TrustStore::open_default().is_trusted(&cwd);
    let mut registry = tools::default_registry();
    tools::apply_filter(
        &mut registry,
        tools_allow.as_deref(),
        tools_exclude.as_deref(),
        no_tools,
    );
    let base = default_system_prompt(&registry);
    let system = beyond_ai_agent::resources::build_system_prompt(
        &beyond_ai_agent::resources::PromptOptions {
            base: &base,
            append: None,
            cwd: &cwd,
            include_context_files: true,
            include_skills: project_trusted,
            project_trusted,
        },
    );

    let client = GatewayClient::new(gateway, key)?;
    let agent = Agent::new(Arc::new(client), model)
        .with_tools(registry)
        .with_system(system)
        .with_max_steps(max_steps);

    let mut session = Session::new();
    session.user(task);

    // Render assistant text live; surface tool activity on its own line.
    agent
        .run(&mut session, |ev| match ev {
            StreamEvent::TextDelta { text } => {
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
            StreamEvent::ToolUseStart { name, .. } => {
                println!("\n[tool: {name}]");
            }
            _ => {}
        })
        .await?;

    println!();
    eprintln!(
        "[done in {} step(s); {} in / {} out tokens]",
        session.steps, session.input_tokens, session.output_tokens
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_system_prompt_lists_every_registered_tool() {
        // The whole point of generating this dynamically: it can't silently omit a tool the way the
        // prior hardcoded string did (it never mentioned the Beyond platform tools at all).
        let registry = tools::default_registry();
        let prompt = default_system_prompt(&registry);
        for def in tools::default_registry().definitions() {
            assert!(
                prompt.contains(&def.name),
                "system prompt is missing registered tool {:?}: {prompt}",
                def.name
            );
        }
    }

    #[test]
    fn default_system_prompt_reflects_a_restricted_registry() {
        // A tool-restricted agent's own system prompt must not claim tools it doesn't actually have —
        // otherwise the model is invited to call one that's guaranteed to be rejected.
        let mut registry = tools::default_registry();
        tools::apply_filter(&mut registry, None, Some(&["bash".to_string()]), false);
        let prompt = default_system_prompt(&registry);
        assert!(!prompt.contains("bash"));
        assert!(prompt.contains("read"));
    }
}
