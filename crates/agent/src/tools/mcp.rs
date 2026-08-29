//! MCP (Model Context Protocol) client support — this crate's extension mechanism.
//!
//! pi (the TypeScript reference this codebase tracks) has its own in-process extension system
//! (`registerTool`/`registerCommand` modules loaded directly into the host). Beyond deliberately does
//! *not* copy that shape: MCP is a standardized, language-agnostic protocol (JSON-RPC over stdio or
//! HTTP) with an existing server ecosystem (filesystem, GitHub, Slack, Postgres, ...), and — unlike
//! loading `.so`/`.dylib` plugins via `libloading` — needs zero `unsafe` code, which this workspace
//! forbids (`unsafe_code = "forbid"`).
//!
//! [`connect_all`] connects to every [`McpServerConfig`](crate::settings::McpServerConfig) configured
//! in `settings.json` (global or a trusted project's own), lists each server's tools via `tools/list`,
//! and wraps each into an [`McpTool`] — an ordinary [`agent_core::Tool`] registered into the same
//! [`agent_core::ToolRegistry`] the built-in tools use, so the model sees no difference between a
//! built-in `read`/`bash` and an MCP-discovered tool. Every tool is namespaced
//! `mcp__<server-name>__<tool-name>` (the same convention Claude Code itself uses), so it can never
//! collide with a built-in tool, and a collision with another server's tool is scoped to that one
//! server's own name.
//!
//! Connecting happens exactly once, at startup (`main.rs`'s `run` path, and once before `serve`'s main
//! loop) — not per-session, and not re-done on a `serve` registry rebuild (`set_model`/`set_thinking`
//! reuse the already-connected tools; see `serve.rs::ServeConfig::mcp_tools`). A server that fails to
//! connect is skipped with a warning rather than failing the whole agent's startup (fail-soft — matches
//! this crate's general "skip and warn, don't silently lose data, but don't hard-fail a normal
//! invocation either" convention, e.g. `settings::read_store_file`).
//!
//! Out of scope for v1 (an explicit, narrower cut, not an oversight): MCP *resources* and *prompts* —
//! only the `tools` primitive is wired up, since that's the actual need this feature exists for. Also
//! out of scope: per-session dynamic add/remove of a single server (today's whole-registry-rebuild
//! pattern, matched here, is the existing precedent for every other tool-set change).

use std::collections::HashMap;
use std::sync::Arc;

use agent_core::{ImageSource, Tool, ToolError, ToolOutput};
use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use rmcp::model::{CallToolRequestParams, ContentBlock};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde_json::Value;

use crate::settings::{McpServerConfig, McpTransport};

/// A connected MCP server's live client handle. `()` is `rmcp`'s no-op [`rmcp::ClientHandler`] — this
/// process never needs to answer a server-initiated request (sampling, roots, elicitation), so there's
/// nothing to customize. Shared (via `Arc`) by every [`McpTool`] the server produced: all of a server's
/// tools reuse the one underlying stdio child process / HTTP connection rather than each dialing in
/// independently.
type McpClient = RunningService<RoleClient, ()>;

/// The prefix every MCP-discovered tool's registered name carries — `mcp__<server>__<tool>` — so it
/// can never collide with a built-in tool. Matches the convention Claude Code itself uses for the
/// identical problem.
fn registered_name(server: &str, remote_tool: &str) -> String {
    format!("mcp__{server}__{remote_tool}")
}

/// One MCP-discovered tool. Holds no local behavior at all — `run` forwards straight to the remote
/// server's `tools/call`; every byte of actual behavior lives on the other end of `client`.
struct McpTool {
    /// Registered/advertised name: `mcp__<server>__<tool>` (see [`registered_name`]).
    name: String,
    description: String,
    input_schema: Value,
    /// The bare tool name as the *server* knows it (unprefixed) — what actually goes out over
    /// `tools/call`.
    remote_name: String,
    /// For error messages only, so a failure names which server misbehaved.
    server_name: String,
    client: Arc<McpClient>,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.input_schema.clone()
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        // MCP's `tools/call` arguments are a JSON object (`CallToolRequestParams::arguments:
        // Option<JsonObject>`); the model always sends an object for a tool's arguments (matching every
        // built-in tool's own schema), and `Value::Null` (a tool with an empty input schema, called with
        // no arguments at all) maps to "no arguments" rather than an error.
        let arguments = match input {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return Err(ToolError::InvalidInput(format!(
                    "expected a JSON object of arguments for `{}`, got: {other}",
                    self.name
                )));
            }
        };
        let mut params = CallToolRequestParams::new(self.remote_name.clone());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }

        let result = self.client.call_tool(params).await.map_err(|e| {
            ToolError::Execution(format!(
                "mcp server `{}` tool `{}` call failed: {e}",
                self.server_name, self.remote_name
            ))
        })?;

        let mut text = String::new();
        let mut images = Vec::new();
        for block in result.content {
            match block {
                ContentBlock::Text(t) => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&t.text);
                }
                ContentBlock::Image(img) => {
                    images.push(ImageSource::base64(img.mime_type, img.data));
                }
                // Audio/embedded-resource/resource-link content has no representation in
                // `ToolOutput` today (text + images only, matching every built-in tool). Summarized as
                // text rather than silently dropped, so the model at least knows something came back
                // that it can't fully see — resources/prompts are an explicit v2 scope, not this.
                other => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&format!("[unsupported MCP content block: {other:?}]"));
                }
            }
        }

        if result.is_error == Some(true) {
            return Err(ToolError::Execution(if text.is_empty() {
                format!(
                    "mcp server `{}` tool `{}` reported an error with no message",
                    self.server_name, self.remote_name
                )
            } else {
                text
            }));
        }
        Ok(ToolOutput {
            text,
            images,
            terminate: false,
        })
    }
}

/// Connect to every configured MCP server, returning every tool discovered (already wrapped and ready
/// to [`register`](agent_core::ToolRegistry::register)) plus one warning string per server that failed
/// to connect. Fail-soft: a misconfigured or dead server never blocks another configured server, or the
/// agent's own startup — see the module doc comment for why.
///
/// Every server connects *concurrently* (`futures::future::join_all`), not one after another: each
/// connect is independent I/O (a process spawn + handshake, or a network round trip) with zero data
/// dependency on any other server, so connecting sequentially would needlessly add every server's own
/// latency to `run`/`serve` startup instead of paying only the slowest one — a real, user-visible cost
/// for an operator with several servers configured, not a micro-optimization.
pub async fn connect_all(configs: &[McpServerConfig]) -> (Vec<Arc<dyn Tool>>, Vec<String>) {
    let results = futures::future::join_all(configs.iter().map(connect_one)).await;
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    let mut warnings = Vec::new();
    for (config, result) in configs.iter().zip(results) {
        match result {
            Ok(server_tools) => tools.extend(server_tools),
            Err(e) => {
                tracing::warn!(
                    server = %config.name,
                    error = %e,
                    "failed to connect to MCP server; its tools will not be available"
                );
                warnings.push(format!("mcp server `{}`: {e}", config.name));
            }
        }
    }
    (tools, warnings)
}

async fn connect_one(config: &McpServerConfig) -> Result<Vec<Arc<dyn Tool>>, String> {
    let client = match &config.transport {
        McpTransport::Stdio { command, args, .. } => connect_stdio(config, command, args).await?,
        McpTransport::Http { url, .. } => connect_http(config, url).await?,
    };
    tools_from_client(config, client).await
}

/// Spawns `command` as a plain child process — deliberately *not* its own process-group leader the way
/// `tools::exec::RealRunner` spawns `bash` (`cmd.process_group(0)`, so a timeout/cancellation can kill
/// a whole backgrounded subtree): an MCP server is a single long-lived process, not a shell that can
/// fork off detached descendants, so there's no subtree to worry about.
///
/// Nor does this crate track the spawned pid for an explicit kill-before-exit barrier the way `bash`'s
/// `GroupKillGuard`/`wait_for_pending_group_kills` do for a run that's about to call `std::process::exit`
/// (which skips destructors entirely, so a merely-Rust-`Drop`-triggered cleanup — like `rmcp`'s own
/// `TokioChildProcess`, which kills its child on drop — might never get to run). That asymmetry is
/// deliberate, not an oversight: the MCP stdio transport's own contract is that a server watches its
/// stdin for EOF as its shutdown signal, and the OS closes every fd of a terminated process
/// unconditionally — including this one's end of the child's stdin pipe — regardless of *how* this
/// process exits (a clean return, `std::process::exit`, or a fatal signal). A spec-compliant server (this
/// crate's own `mcp_fixture_stdio_server` test fixture included) always sees that EOF and exits on its
/// own; only a hung or non-compliant server would be left running, a materially different (and much
/// lower-likelihood) risk than an arbitrary `bash`-run shell command backgrounding an uncooperative
/// descendant.
///
/// `stderr` is left at `TokioChildProcess`'s own default (`Stdio::inherit()`), not captured — a
/// deliberate choice, not an oversight: a server that fails to start or crashes typically explains why
/// on its own stderr, and inheriting it means that reaches the operator's own console (this process's
/// stderr) immediately, the same way a connect failure's `tracing::warn!` does.
async fn connect_stdio(
    config: &McpServerConfig,
    command: &str,
    args: &[String],
) -> Result<McpClient, String> {
    let env = config.resolved_env();
    let child = TokioChildProcess::new(tokio::process::Command::new(command).configure(|cmd| {
        cmd.args(args);
        for (k, v) in &env {
            cmd.env(k, v);
        }
    }))
    .map_err(|e| format!("failed to spawn `{command}`: {e}"))?;

    ().serve(child)
        .await
        .map_err(|e| format!("MCP handshake over stdio failed: {e}"))
}

async fn connect_http(config: &McpServerConfig, url: &str) -> Result<McpClient, String> {
    let mut custom_headers: HashMap<HeaderName, HeaderValue> = HashMap::new();
    for (k, v) in config.resolved_headers() {
        let name = match HeaderName::from_bytes(k.as_bytes()) {
            Ok(name) => name,
            Err(e) => {
                tracing::warn!(header = %k, error = %e, "skipping an MCP server header with an invalid name");
                continue;
            }
        };
        let value = match HeaderValue::from_str(&v) {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!(header = %k, error = %e, "skipping an MCP server header with an invalid value");
                continue;
            }
        };
        custom_headers.insert(name, value);
    }

    // A previously `agent mcp-login`'d server gets its (auto-refreshed, if needed) bearer token
    // attached; a server nobody has logged into (the common case — most MCP servers need no auth at
    // all, or use `headers` above for a static credential) connects exactly as before.
    let bearer_token = oauth_bearer_token(&config.name, url).await;
    let mut transport_config = StreamableHttpClientTransportConfig::with_uri(url.to_string())
        .custom_headers(custom_headers);
    if let Some(token) = &bearer_token {
        transport_config = transport_config.auth_header(token.clone());
    }
    agent_core::ensure_provider();
    let transport =
        StreamableHttpClientTransport::with_client(reqwest::Client::new(), transport_config);

    ().serve(transport).await.map_err(|e| {
        let hint = if bearer_token.is_none() {
            format!(
                " (if this server requires login, run `agent mcp-login {}` first)",
                config.name
            )
        } else {
            String::new()
        };
        format!("MCP handshake over streamable-HTTP to {url} failed: {e}{hint}")
    })
}

/// A currently-valid bearer access token for `server_name`'s MCP OAuth login, if one exists —
/// `agent mcp-login {server_name}` establishes it (see that command's own doc comment); this only
/// ever *reads and refreshes* an already-established one via
/// [`AuthorizationManager::initialize_from_store`]/[`AuthorizationManager::get_access_token`], the
/// same [`ScopedMcpCredentialStore`](crate::mcp_auth_store::ScopedMcpCredentialStore) `mcp-login`
/// wrote to — a refresh (silent, automatic, and re-persisted by `rmcp` itself) is indistinguishable
/// here from a token that never needed refreshing at all.
///
/// Returns `None` — not an error — for every case short of "found a token I could actually attach":
/// no login has ever happened for this server (the overwhelmingly common case, so this stays a single
/// cheap file check via [`McpAuthStore::has_credential`] rather than always paying a metadata-discovery
/// round trip), or a stored credential exists but can no longer be used (refresh token revoked/expired
/// with no way to recover). Either way, [`connect_http`] proceeds with an unauthenticated connect
/// attempt and lets *that* failure (or success, for a server that turns out not to need auth after
/// all) be the actual signal — this function never itself decides a missing/dead credential is fatal.
async fn oauth_bearer_token(server_name: &str, url: &str) -> Option<String> {
    let store = crate::mcp_auth_store::McpAuthStore::open_default();
    if !store.has_credential(server_name) {
        return None;
    }
    let mut manager = match rmcp::transport::auth::AuthorizationManager::new(url).await {
        Ok(manager) => manager,
        Err(e) => {
            tracing::warn!(server = %server_name, error = %e, "failed to set up MCP OAuth for a stored login");
            return None;
        }
    };
    manager.set_credential_store(store.scoped(server_name));
    match manager.initialize_from_store().await {
        Ok(true) => {}
        // `has_credential` above already confirmed a credential exists, so this shouldn't happen in
        // practice — treated the same as "no credential" rather than as an error either way.
        Ok(false) => return None,
        Err(e) => {
            tracing::warn!(server = %server_name, error = %e, "failed to restore a stored MCP OAuth login");
            return None;
        }
    }
    match manager.get_access_token().await {
        Ok(token) => Some(token),
        Err(e) => {
            tracing::warn!(
                server = %server_name,
                error = %e,
                "stored MCP OAuth login could not be refreshed; run `agent mcp-login {server_name}` again"
            );
            None
        }
    }
}

async fn tools_from_client(
    config: &McpServerConfig,
    client: McpClient,
) -> Result<Vec<Arc<dyn Tool>>, String> {
    let remote_tools = client
        .list_all_tools()
        .await
        .map_err(|e| format!("`tools/list` failed: {e}"))?;
    let client = Arc::new(client);
    Ok(remote_tools
        .into_iter()
        .map(|remote_tool| {
            let description = remote_tool
                .description
                .as_deref()
                .filter(|d| !d.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "MCP tool `{}` from server `{}` (no description provided)",
                        remote_tool.name, config.name
                    )
                });
            Arc::new(McpTool {
                name: registered_name(&config.name, &remote_tool.name),
                description,
                input_schema: remote_tool.schema_as_json_value(),
                remote_name: remote_tool.name.into_owned(),
                server_name: config.name.clone(),
                client: client.clone(),
            }) as Arc<dyn Tool>
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_name_uses_the_double_underscore_prefix_convention() {
        assert_eq!(
            registered_name("filesystem", "read_file"),
            "mcp__filesystem__read_file"
        );
    }

    #[test]
    fn registered_name_cannot_collide_with_a_bare_builtin_tool_name() {
        // Every built-in tool name (`read`, `write`, `edit`, `bash`, `ls`, `grep`, `find`, `fork`,
        // `sync`, `logs`) is a bare identifier with no `__` in it — the `mcp__` prefix guarantees no
        // MCP-discovered tool can ever land on one of those keys in the registry, regardless of what a
        // server names its own tool.
        for builtin in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert_ne!(registered_name("server", builtin), builtin);
        }
    }
}
