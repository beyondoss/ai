//! `login`, `logout`, `auth-status` (model providers) and `mcp-login`, `mcp-logout` (MCP servers).

use beyond_agent::session_store::canonical_cwd;

use crate::{CliLoginCallbacks, unknown_provider_error};

/// `login <provider>`
pub(crate) async fn login(provider: String) -> Result<(), Box<dyn std::error::Error>> {
    let provider_id = beyond_agent::oauth::OAuthProviderId::parse(&provider)
        .ok_or_else(|| unknown_provider_error(&provider))?;
    let cancel = agent_core::CancellationToken::new();
    let credential = beyond_agent::oauth::login(provider_id, &CliLoginCallbacks, &cancel).await?;
    let mut store = beyond_agent::auth_store::AuthStore::open_default();
    store.set(provider_id.store_key(), credential)?;
    println!("logged in: {provider_id}");
    Ok(())
}

/// `logout <provider>`
pub(crate) fn logout(provider: String) -> Result<(), Box<dyn std::error::Error>> {
    let provider_id = beyond_agent::oauth::OAuthProviderId::parse(&provider)
        .ok_or_else(|| unknown_provider_error(&provider))?;
    let mut store = beyond_agent::auth_store::AuthStore::open_default();
    if store.remove(provider_id.store_key())? {
        println!("logged out: {provider_id}");
    } else {
        println!("not logged in: {provider_id}");
    }
    Ok(())
}

/// `auth-status [provider]`
pub(crate) fn auth_status(provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let store = beyond_agent::auth_store::AuthStore::open_default();
    let providers = match &provider {
        Some(p) => vec![
            beyond_agent::oauth::OAuthProviderId::parse(p)
                .ok_or_else(|| unknown_provider_error(p))?,
        ],
        None => beyond_agent::oauth::OAuthProviderId::all().to_vec(),
    };
    for id in providers {
        let status = match store.get(id.store_key()) {
            None => "logged_out",
            Some(stored) if stored.last_refresh_error.is_some() => "needs_reauth",
            Some(_) => "logged_in",
        };
        println!("{id}: {status}");
    }
    Ok(())
}

/// `mcp-login <name>`: OAuth against an MCP server from the effective settings.
pub(crate) async fn mcp_login(name: String) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = canonical_cwd(&std::env::current_dir().unwrap_or_default());
    let settings = beyond_agent::settings::effective_settings_for_cwd(&cwd);
    let servers = settings.mcp_servers.unwrap_or_default();
    let server = servers.iter().find(|s| s.name == name).ok_or_else(|| {
        format!(
            "no MCP server named `{name}` is configured (checked global and, if this \
                 directory is trusted, project settings.json)"
        )
    })?;
    let url = match &server.transport {
        beyond_agent::settings::McpTransport::Http { url, .. } => url.clone(),
        beyond_agent::settings::McpTransport::Stdio { .. } => {
            return Err(format!(
                "`{name}` uses the stdio transport, which has no OAuth login of its own — \
                     configure a credential via its `env` instead"
            )
            .into());
        }
    };

    let mut manager = rmcp::transport::auth::AuthorizationManager::new(&url)
        .await
        .map_err(|e| format!("failed to start MCP OAuth for `{name}`: {e}"))?;
    let auth_store = beyond_agent::mcp_auth_store::McpAuthStore::open_default();
    manager.set_credential_store(auth_store.scoped(&name, &url));
    // The spec's two PRM discovery paths: the `resource_metadata` of the server's own 401
    // challenge, then the well-known URI. A server that answers 401 is asked first; rmcp tries
    // the well-known URIs itself when there is no challenge or it names nothing usable.
    let challenge = probe_challenge(&url).await;
    let resolution = manager
        .resolve_metadata_from_challenge(challenge.as_deref())
        .await
        .map_err(|e| format!("`{name}` does not support MCP OAuth (or discovery failed): {e}"))?;
    check_discovered(&name, &url, &resolution)?;
    manager.set_metadata(resolution.metadata);

    // Dynamic client registration means we choose our own redirect URI (unlike `agent
    // login`'s providers, each registered in advance against a fixed port). The callback
    // listener is bound first, on a port the kernel picks, and the URI names the port it got:
    // picking a free port, releasing it and binding it later let another process take it in
    // between, and the login then failed before printing its URL.
    let listener =
            // port-0: the listener is held for the whole login; the redirect URI names its port.
            beyond_agent::oauth::callback_server::CallbackServer::bind("127.0.0.1", 0)
                .map_err(|e| {
                    format!("failed to bind the local OAuth callback listener: {e}")
                })?;
    let port = listener.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    // rmcp 3.x folds scopes + client name into `AuthorizationRequest`; empty scopes
    // auto-select via the manager's scope policy inside `AuthorizationSession::new`.
    let request = rmcp::transport::auth::AuthorizationRequest::new(&redirect_uri)
        .with_client_name("beyond-agent");
    let session = rmcp::transport::auth::AuthorizationSession::new(manager, request)
        .await
        .map_err(|(_manager, e)| {
            format!("failed to start MCP OAuth authorization for `{name}`: {e}")
        })?;
    let auth_url = session.get_authorization_url().to_string();
    let csrf_token = url::Url::parse(&auth_url)
        .ok()
        .and_then(|u| {
            u.query_pairs()
                .find(|(k, _)| k == "state")
                .map(|(_, v)| v.into_owned())
        })
        .ok_or("the generated authorization URL is missing its `state` parameter")?;

    eprintln!("Open this URL in a browser to continue:\n\n  {auth_url}\n");
    let cancel = agent_core::CancellationToken::new();
    let code = listener
        .wait_for_callback("/callback", csrf_token.clone(), cancel)
        .await
        .ok_or("login cancelled (no callback received)")?;
    session
        .handle_callback(&code, &csrf_token)
        .await
        .map_err(|e| format!("failed to complete MCP OAuth login for `{name}`: {e}"))?;
    println!("logged in: {name}");
    Ok(())
}

/// `mcp-logout <name>`
pub(crate) fn mcp_logout(name: String) -> Result<(), Box<dyn std::error::Error>> {
    let store = beyond_agent::mcp_auth_store::McpAuthStore::open_default();
    let had_credential = store.has_credential(&name);
    store.clear(&name)?;
    if had_credential {
        println!("logged out: {name}");
    } else {
        println!("not logged in: {name}");
    }
    Ok(())
}

/// The `WWW-Authenticate` challenge of the server's answer to an unauthenticated request, when
/// that answer is a 401 (`None`: anything else, or the probe failed — discovery then goes by the
/// well-known URIs alone). The request is the one a client opens with, `initialize`.
async fn probe_challenge(url: &str) -> Option<String> {
    agent_core::ensure_provider();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .ok()?;
    let response = client
        .post(url)
        .header("accept", "application/json, text/event-stream")
        .json(&serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": { "name": "beyond-agent", "version": env!("CARGO_PKG_VERSION") }
            }
        }))
        .send()
        .await
        .ok()?;
    if response.status() != reqwest::StatusCode::UNAUTHORIZED {
        return None;
    }
    response
        .headers()
        .get_all(reqwest::header::WWW_AUTHENTICATE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .find(|v| v.trim_start().to_ascii_lowercase().starts_with("bearer"))
        .map(str::to_owned)
}

/// Refuse to log in on what the server did not publish: endpoints rmcp guessed (`/authorize`,
/// `/token`, `/register` on the base URL, with no issuer to check an `iss` against), and an
/// authorization server that does not advertise PKCE `S256` — the spec's "Authorization Code
/// Protection": a client MUST refuse to proceed without it. rmcp only warns on the latter.
fn check_discovered(
    name: &str,
    url: &str,
    resolution: &rmcp::transport::auth::AuthorizationMetadataResolution,
) -> Result<(), String> {
    if !resolution.source.is_discovered() {
        return Err(format!(
            "`{name}` published no OAuth metadata: neither its 401 challenge's \
             `resource_metadata` nor {url}'s well-known protected-resource or \
             authorization-server metadata could be fetched; not guessing endpoints"
        ));
    }
    let pkce = resolution
        .metadata
        .code_challenge_methods_supported
        .as_deref();
    if !pkce.is_some_and(|m| m.iter().any(|m| m == "S256")) {
        return Err(format!(
            "`{name}`'s authorization server does not advertise PKCE (`S256` in \
             `code_challenge_methods_supported`); refusing to log in without it"
        ));
    }
    Ok(())
}
