//! `beyond-ai-demo`: the pieces of `deploy/demo` that are code rather than AWS calls.
//!
//! - `keys`: generate the demo's keys (once; `up.sh` stores them in SSM).
//! - `mint-grant`: mint one `bsg_v1` session grant from those keys.
//! - `chat`: one conversation step over the agent's WebSocket.

use beyond_ai_demo::{chat, keys, load, load_host};

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
        /// Carry `Authorization: Bearer $DEMO_EXEC_BEARER` in the grant's exec headers.
        #[arg(long)]
        exec_bearer: bool,
        /// A header the exec endpoint needs, `name=Header-Name`; the value is read from the
        /// environment variable `DEMO_EXEC_HEADER_<NAME>` (upper case, `-` as `_`), never argv.
        #[arg(long = "exec-header")]
        exec_header: Vec<String>,
        /// An MCP connector: `name=url`. Repeatable.
        #[arg(long = "mcp")]
        mcp: Vec<String>,
        /// Mark the named connector `sandbox`: no standing stream, no pooled idle connection.
        #[arg(long = "mcp-sandbox")]
        mcp_sandbox: Vec<String>,
        /// Mark the named connector managed: it holds a `bmc_v1` token (secret `DEMO_MANAGED_SECRET`).
        #[arg(long = "mcp-managed")]
        mcp_managed: Vec<String>,
        /// A header the named connector sends, `name=Header-Name`; the value is read from the
        /// environment variable `DEMO_MCP_HEADER_<NAME>` (upper case, `-` as `_`), never argv.
        #[arg(long = "mcp-header")]
        mcp_header: Vec<String>,
        /// The grant's `limits.models` ceiling (comma-separated ids or `*` patterns).
        #[arg(long, value_delimiter = ',')]
        limits_models: Option<Vec<String>>,
        /// The grant's `limits.tools` ceiling.
        #[arg(long, value_delimiter = ',')]
        limits_tools: Option<Vec<String>>,
        /// The grant's `limits.max_steps` ceiling.
        #[arg(long)]
        limits_max_steps: Option<u32>,
        /// The grant's `limits.max_mcp_json_servers` ceiling: servers a session may add from `.mcp.json`.
        #[arg(long)]
        limits_max_mcp_json_servers: Option<u32>,
        /// The grant's `bundle` claim, `url=sha256hex`: the agent folder as a plain tar the replica
        /// fetches and verifies. With it, `--exec-url` may be empty (a session with no sandbox).
        #[arg(long)]
        bundle: Option<String>,
    },
    /// Unseal the grant in DEMO_GRANT and print what it holds, without the secrets: connector names,
    /// header names, and whether the string in DEMO_NEEDLE is anywhere inside.
    #[cfg(feature = "agent-lib")]
    InspectGrant,
    /// Run the stand-in managed-MCP endpoint and its provider (secrets from DEMO_MANAGED_SECRET and
    /// DEMO_PROVIDER_SECRET).
    #[cfg(feature = "agent-lib")]
    ManagedEndpoint {
        #[arg(long, default_value = "0.0.0.0:8082")]
        listen: String,
        #[arg(long, default_value = "127.0.0.1:8083")]
        provider_listen: String,
    },
    /// Run the broker that acts as a signed-in user, and the notes service it fronts (secrets from
    /// DEMO_MANAGED_SECRET, DEMO_BROKER_MASTER, DEMO_BROKER_ADMIN and DEMO_OIDC_CLIENT_SECRET).
    #[cfg(feature = "agent-lib")]
    OidcEndpoint {
        #[arg(long, default_value = "0.0.0.0:8082")]
        managed_listen: String,
        #[arg(long, default_value = "0.0.0.0:8085")]
        control_listen: String,
        #[arg(long, default_value = "127.0.0.1:8086")]
        resource_listen: String,
        /// `host:port` the identity provider is reached at.
        #[arg(long, default_value = "127.0.0.1:8180")]
        idp_internal: String,
        /// The provider's issuer URL, as its tokens name it.
        #[arg(long)]
        issuer: String,
        #[arg(long, default_value = "notes-broker")]
        client_id: String,
        /// The callback URL registered at the provider.
        #[arg(long)]
        redirect_uri: String,
        #[arg(long, default_value = "/data")]
        data_dir: PathBuf,
    },
    /// Print one `bmc_v1` token (secret from DEMO_MANAGED_SECRET): what a session's grant holds, for
    /// a check that plays a session presenting it.
    #[cfg(feature = "agent-lib")]
    ManagedToken {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        session: String,
        #[arg(long)]
        connector: String,
        #[arg(long, default_value_t = 600)]
        ttl_secs: u64,
    },
    /// Print a job-end callback token and its scope as JSON (secret from DEMO_FLEET_SECRET).
    #[cfg(feature = "agent-lib")]
    NotifyToken {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        session: String,
        /// A lowercase-hex job id.
        #[arg(long)]
        job: String,
        /// Corrupt the tag: the right kid, a wrong MAC.
        #[arg(long)]
        forge: bool,
    },
    /// The load proof's stand-in: a scripted model and a bundle host (token in DEMO_LOAD_TOKEN).
    LoadHost {
        #[arg(long, default_value = "0.0.0.0:8086")]
        listen: String,
        #[arg(long, default_value = "0.0.0.0:8083")]
        model_listen: String,
        /// Hold every bundle fetch this long, so concurrent fetches overlap and their peak shows.
        #[arg(long, default_value_t = 250)]
        fetch_hold_ms: u64,
        #[arg(long, default_value_t = 1000)]
        mcp_hold_ms: u64,
    },
    /// The load proof's driver: waves of sessions against one replica. Writes the client's record.
    Load {
        #[arg(long)]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// The bundle host as the replica reaches it, `ip:port`.
        #[arg(long)]
        loadhost: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 500)]
        sessions: usize,
        #[arg(long, default_value_t = 200)]
        burst: usize,
        #[arg(long, default_value_t = 50)]
        tenants: usize,
        #[arg(long, default_value_t = 60)]
        ramp_secs: u64,
        #[arg(long, default_value_t = 90)]
        hold_secs: u64,
        #[arg(long, default_value_t = 180)]
        idle_secs: u64,
        #[arg(long, default_value_t = 2)]
        waves: usize,
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
        /// Run `true` in the sandbox after connecting (a reopened session reconciles its jobs then).
        #[arg(long)]
        touch_sandbox: bool,
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
            exec_bearer,
            exec_header,
            mcp,
            mcp_sandbox,
            mcp_managed,
            mcp_header,
            limits_models,
            limits_tools,
            limits_max_steps,
            limits_max_mcp_json_servers,
            bundle,
        } => {
            let connectors = connectors(&mcp, &mcp_sandbox, &mcp_managed, &mcp_header)?;
            let exec_headers = exec_header
                .iter()
                .map(|h| {
                    let (name, header_name) = h
                        .split_once('=')
                        .ok_or("--exec-header takes name=Header-Name")?;
                    let var = format!(
                        "DEMO_EXEC_HEADER_{}",
                        name.to_ascii_uppercase().replace('-', "_")
                    );
                    let value = std::env::var(&var).map_err(|_| format!("{var} is not set"))?;
                    Ok::<_, String>((header_name.to_string(), value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let limits = (limits_models.is_some()
                || limits_tools.is_some()
                || limits_max_steps.is_some()
                || limits_max_mcp_json_servers.is_some())
            .then_some(keys::Limits {
                models: limits_models,
                tools: limits_tools,
                max_steps: limits_max_steps,
                max_mcp_json_servers: limits_max_mcp_json_servers,
            });
            let bundle = bundle
                .map(|b| {
                    let (url, sha256) = b.rsplit_once('=').ok_or("--bundle takes url=sha256hex")?;
                    Ok::<_, String>(keys::Bundle {
                        url: url.to_string(),
                        sha256: sha256.to_string(),
                    })
                })
                .transpose()?;
            let exec_bearer = if exec_bearer {
                Some(std::env::var("DEMO_EXEC_BEARER").map_err(|_| "DEMO_EXEC_BEARER is not set")?)
            } else {
                None
            };
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
                        exec_bearer,
                        exec_headers,
                        connectors,
                        limits,
                        bundle,
                    }
                )?
            );
            Ok(())
        }
        #[cfg(feature = "agent-lib")]
        Cmd::InspectGrant => {
            let token = std::env::var("DEMO_GRANT").map_err(|_| "DEMO_GRANT is not set")?;
            let needle = std::env::var("DEMO_NEEDLE").unwrap_or_default();
            println!(
                "{}",
                keys::inspect_grant(&keys::GrantKeys::from_env()?, &token, &needle)?
            );
            Ok(())
        }
        #[cfg(feature = "agent-lib")]
        Cmd::ManagedEndpoint {
            listen,
            provider_listen,
        } => {
            beyond_ai_demo::managed::run(beyond_ai_demo::managed::Args {
                listen,
                provider_listen,
            })
            .await
            .map_err(|e| e.to_string())?;
            Ok(())
        }
        #[cfg(feature = "agent-lib")]
        Cmd::ManagedToken {
            tenant,
            session,
            connector,
            ttl_secs,
        } => {
            println!(
                "{}",
                keys::mint_managed_token(&tenant, &session, &connector, ttl_secs)?
            );
            Ok(())
        }
        #[cfg(feature = "agent-lib")]
        Cmd::OidcEndpoint {
            managed_listen,
            control_listen,
            resource_listen,
            idp_internal,
            issuer,
            client_id,
            redirect_uri,
            data_dir,
        } => {
            beyond_ai_demo::oidc::run(beyond_ai_demo::oidc::Args {
                managed_listen,
                control_listen,
                resource_listen,
                idp_internal,
                issuer,
                client_id,
                redirect_uri,
                data_dir,
            })
            .await
            .map_err(|e| e.to_string())?;
            Ok(())
        }
        #[cfg(feature = "agent-lib")]
        Cmd::NotifyToken {
            tenant,
            session,
            job,
            forge,
        } => {
            let (token, scope) = keys::notify_token(&tenant, &session, &job, forge)?;
            println!("{}", serde_json::json!({"token": token, "scope": scope}));
            Ok(())
        }
        Cmd::LoadHost {
            listen,
            model_listen,
            fetch_hold_ms,
            mcp_hold_ms,
        } => {
            let token =
                std::env::var("DEMO_LOAD_TOKEN").map_err(|_| "DEMO_LOAD_TOKEN is not set")?;
            tokio::task::spawn_blocking(move || {
                load_host::run(load_host::Args {
                    listen,
                    model_listen,
                    token,
                    fetch_hold_ms,
                    mcp_hold_ms,
                })
                .map_err(|e| e.to_string())
            })
            .await??;
            Ok(())
        }
        Cmd::Load {
            host,
            port,
            loadhost,
            out,
            sessions,
            burst,
            tenants,
            ramp_secs,
            hold_secs,
            idle_secs,
            waves,
        } => {
            let token =
                std::env::var("DEMO_LOAD_TOKEN").map_err(|_| "DEMO_LOAD_TOKEN is not set")?;
            let phases = load::run(load::Args {
                host,
                port,
                loadhost,
                token,
                out,
                sessions,
                burst,
                tenants,
                ramp_secs,
                hold_secs,
                idle_secs,
                waves,
            })
            .await
            .map_err(|e| e.to_string())?;
            println!("{phases}");
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
            touch_sandbox,
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
                touch_sandbox,
            })
            .await?;
            println!("{summary}");
            Ok(())
        }
    }
}

/// The `--mcp*` flags as connectors. A flag naming a connector `--mcp` never declared is an error.
fn connectors(
    mcp: &[String],
    sandbox: &[String],
    managed: &[String],
    header: &[String],
) -> Result<Vec<keys::Connector>, Box<dyn std::error::Error>> {
    let mut out: Vec<keys::Connector> = Vec::new();
    for m in mcp {
        let (name, url) = m.split_once('=').ok_or("--mcp takes name=url")?;
        out.push(keys::Connector {
            name: name.into(),
            url: url.into(),
            ..Default::default()
        });
    }
    let find = |out: &[keys::Connector], name: &str| -> Result<usize, String> {
        out.iter()
            .position(|c| c.name == name)
            .ok_or_else(|| format!("no --mcp connector named {name}"))
    };
    for n in sandbox {
        let i = find(&out, n)?;
        out[i].sandbox = true;
    }
    for n in managed {
        let i = find(&out, n)?;
        out[i].managed = true;
    }
    for h in header {
        let (name, header_name) = h
            .split_once('=')
            .ok_or("--mcp-header takes name=Header-Name")?;
        let i = find(&out, name)?;
        let var = format!(
            "DEMO_MCP_HEADER_{}",
            name.to_ascii_uppercase().replace('-', "_")
        );
        let value = std::env::var(&var).map_err(|_| format!("{var} is not set"))?;
        out[i].headers.push((header_name.into(), value));
    }
    Ok(out)
}
