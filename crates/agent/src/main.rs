//! Beyond agent harness — CLI.
//!
//! Scaffold. The command surface is the one the harness targets: a one-shot `run` and a headless
//! `serve` (with an attachable control API, for remote control over SSH). Neither is wired to the
//! agent loop yet — that arrives with the loop (M4), the coding tools (M6), and the control API
//! (M7). `tools` is a working, no-network demo that the core links and the async `Tool` seam runs.

use std::sync::Arc;

use agent_core::error::ToolError;
use agent_core::{Tool, ToolRegistry};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "beyond-ai-agent", version, about = "Beyond agent harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a one-shot agent task to completion. (Agent loop lands in a later milestone.)
    Run {
        /// The task prompt for the agent.
        task: String,
    },
    /// Run the headless agent server exposing an attachable control API. (Later milestone.)
    Serve,
    /// List the tools the agent advertises to the model, and run one (no-network scaffold demo).
    Tools,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    match Cli::parse().command {
        Command::Tools => demo_tools().await?,
        Command::Run { task } => {
            println!("scaffold: `run` is not wired yet (agent loop = M4, CLI = M6).");
            println!("would drive the agent loop to completion for task: {task:?}");
        }
        Command::Serve => {
            println!("scaffold: `serve` is not wired yet (headless control API = M7).");
        }
    }
    Ok(())
}

/// Build the agent's tool registry. As milestones land, the core Read/Write/Edit/Bash and the Beyond
/// fork/sync/logs tools register here; today it carries a single placeholder.
fn registry() -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool));
    reg
}

async fn demo_tools() -> Result<(), Box<dyn std::error::Error>> {
    let reg = registry();
    println!("{} tool(s) registered:\n", reg.len());
    println!("{}", serde_json::to_string_pretty(&reg.definitions())?);
    // Prove the async tool seam runs end to end — no network, no model.
    if let Some(echo) = reg.get("echo") {
        let out = echo.run(json!({ "text": "harness online" })).await?;
        println!("\necho.run -> {out:?}");
    }
    Ok(())
}

/// Placeholder built-in so the scaffold demonstrates the registry + async tool path. Replaced by the
/// real coding tools (Read/Write/Edit/Bash) in M6.
struct EchoTool;

#[async_trait::async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echo the `text` argument back (scaffold placeholder)."
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
