//! `beyond-ai-demo`: the pieces of `deploy/demo` that are code rather than AWS calls.
//!
//! - `exec-stub`: the sandbox, a service-mode agent's exec endpoint (v1.1 to v1.4).
//! - `keys`: generate the demo's keys (once; `up.sh` stores them in SSM).
//! - `mint-grant`: mint one `bsg_v1` session grant from those keys.
//! - `chat`: one conversation step over the agent's WebSocket.

use beyond_ai_demo::{chat, keys, stub};

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "beyond-ai-demo")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the sandbox exec endpoint.
    ExecStub {
        #[arg(long, default_value = "0.0.0.0:8081")]
        listen: String,
        /// The tenant's workspace: where commands start.
        #[arg(long, default_value = "/workspace")]
        root: PathBuf,
        #[arg(long, default_value = "/home/sandbox")]
        home: PathBuf,
    },
    /// Print the demo's keys as JSON.
    #[cfg(feature = "gateway-key")]
    Keys,
    /// Print one session grant (secrets from DEMO_GRANT_SEED, DEMO_FLEET_SECRET, DEMO_GATEWAY_KEY, DEMO_DEK).
    MintGrant {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        session: String,
        #[arg(long, default_value = "demo")]
        shard: String,
        #[arg(long)]
        exec_url: String,
        #[arg(long, default_value = "/workspace")]
        workspace: String,
        #[arg(long, default_value_t = 3600)]
        ttl_secs: u64,
        /// Flip a signature bit: a grant the agent must refuse.
        #[arg(long)]
        tamper: bool,
    },
    /// One conversation step; the grant is in DEMO_GRANT. Prints a JSON summary.
    Chat {
        #[arg(long)]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long)]
        session: String,
        #[arg(long)]
        prompt: Option<String>,
        /// A raw protocol command (JSON) instead of a prompt.
        #[arg(long)]
        command: Option<String>,
        /// After the turn, wait this long for a background job's end to wake the session.
        #[arg(long, default_value_t = 0)]
        wait_wake_secs: u64,
        /// Append every frame to this file, one JSON line each.
        #[arg(long)]
        events: Option<PathBuf>,
        #[arg(long, default_value_t = 180)]
        timeout_secs: u64,
    },
}

#[tokio::main]
async fn main() {
    if let Err(e) = run(Cli::parse()).await {
        eprintln!("beyond-ai-demo: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Cmd::ExecStub { listen, root, home } => stub::run(stub::Args { listen, root, home }).await,
        #[cfg(feature = "gateway-key")]
        Cmd::Keys => {
            println!("{}", keys::generate()?);
            Ok(())
        }
        Cmd::MintGrant {
            tenant,
            session,
            shard,
            exec_url,
            workspace,
            ttl_secs,
            tamper,
        } => {
            println!(
                "{}",
                keys::mint_grant(
                    &keys::GrantKeys::from_env()?,
                    &keys::MintArgs {
                        tenant,
                        session,
                        shard,
                        exec_url,
                        workspace,
                        ttl_secs,
                        tamper,
                    }
                )?
            );
            Ok(())
        }
        Cmd::Chat {
            host,
            port,
            session,
            prompt,
            command,
            wait_wake_secs,
            events,
            timeout_secs,
        } => {
            let summary = chat::run(chat::Args {
                host,
                port,
                session,
                prompt,
                command,
                wait_wake_secs,
                events,
                timeout_secs,
            })
            .await?;
            println!("{summary}");
            Ok(())
        }
    }
}
