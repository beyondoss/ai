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
use std::sync::Arc;

use agent_core::{Agent, GatewayClient, Session, StreamEvent};
use beyond_ai_agent::{serve, tools};
use clap::{Parser, Subcommand};

/// Default model when neither `--model` nor `AI_AGENT_MODEL` is set.
const DEFAULT_MODEL: &str = "claude-opus-4-8";
/// Default gateway base URL.
const DEFAULT_GATEWAY: &str = "http://ai.internal";
/// Default context window (tokens) the loop compacts against. Sized for a 200k-context Claude model.
const DEFAULT_CONTEXT_WINDOW: u32 = 200_000;

const SYSTEM_PROMPT: &str = "You are the Beyond coding agent. You operate inside a real working \
directory with tools: read, write, edit, bash, ls, grep, find. Use them to accomplish the user's \
task directly — inspect before you change, make minimal edits, and verify your work. Be concise.";

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
        #[arg(long, default_value_t = 24)]
        max_steps: u32,
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
        /// Max loop iterations per prompt before bailing.
        #[arg(long, default_value_t = 24)]
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
        /// Model context window (tokens); the loop summarizes older turns to stay below it.
        #[arg(long, env = "AI_AGENT_CONTEXT_WINDOW", default_value_t = DEFAULT_CONTEXT_WINDOW)]
        context_window: u32,
        /// Use the 1-hour prompt-cache TTL (vs 5 minutes); helps when turns are spaced out.
        #[arg(long, default_value_t = false)]
        cache_long: bool,
        /// Enable extended thinking with this token budget (must be below the per-turn max tokens).
        #[arg(long)]
        thinking: Option<u32>,
    },
    /// List the tools the agent advertises to the model.
    Tools,
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
        } => {
            run_task(task, model, gateway_url, key, max_steps).await?;
        }
        Command::Serve {
            model,
            gateway_url,
            key,
            session_file,
            session_dir,
            max_steps,
            system_prompt,
            append_system_prompt,
            no_context_files,
            context_window,
            cache_long,
            thinking,
        } => {
            let key = key
                .ok_or("no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key)")?;
            serve::serve(serve::ServeConfig {
                gateway: gateway_url.unwrap_or_else(|| DEFAULT_GATEWAY.to_string()),
                key,
                model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
                max_steps,
                system: system_prompt.unwrap_or_else(|| SYSTEM_PROMPT.to_string()),
                append_system: append_system_prompt,
                context_files: !no_context_files,
                session_file,
                session_dir,
                context_window,
                cache_long,
                thinking,
            })
            .await?;
        }
        Command::Tools => {
            let reg = tools::default_registry();
            println!("{} tools:\n", reg.len());
            println!("{}", serde_json::to_string_pretty(&reg.definitions())?);
        }
    }
    Ok(())
}

async fn run_task(
    task: String,
    model: Option<String>,
    gateway_url: Option<String>,
    key: Option<String>,
    max_steps: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let gateway = gateway_url.unwrap_or_else(|| DEFAULT_GATEWAY.to_string());
    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let key =
        key.ok_or("no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key)")?;

    let cwd = std::env::current_dir().unwrap_or_default();
    let system = beyond_ai_agent::resources::build_system_prompt(
        &beyond_ai_agent::resources::PromptOptions {
            base: SYSTEM_PROMPT,
            append: None,
            cwd: &cwd,
            include_context_files: true,
            include_skills: true,
        },
    );

    let client = GatewayClient::new(gateway, key)?;
    let agent = Agent::new(Arc::new(client), model)
        .with_tools(tools::default_registry())
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
