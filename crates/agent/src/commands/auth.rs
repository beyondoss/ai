//! `login`, `logout`, `auth-status` (model providers) and `mcp-login`, `mcp-logout` (MCP servers).

use beyond_ai_agent::session_store::canonical_cwd;

use crate::{CliLoginCallbacks, unknown_provider_error};

/// `login <provider>`
pub(crate) async fn login(provider: String) -> Result<(), Box<dyn std::error::Error>> {
    let provider_id = beyond_ai_agent::oauth::OAuthProviderId::parse(&provider)
        .ok_or_else(|| unknown_provider_error(&provider))?;
    let cancel = agent_core::CancellationToken::new();
    let credential =
        beyond_ai_agent::oauth::login(provider_id, &CliLoginCallbacks, &cancel).await?;
    let mut store = beyond_ai_agent::auth_store::AuthStore::open_default();
    store.set(provider_id.store_key(), credential)?;
    println!("logged in: {provider_id}");
    Ok(())
}

/// `logout <provider>`
pub(crate) fn logout(provider: String) -> Result<(), Box<dyn std::error::Error>> {
    let provider_id = beyond_ai_agent::oauth::OAuthProviderId::parse(&provider)
        .ok_or_else(|| unknown_provider_error(&provider))?;
    let mut store = beyond_ai_agent::auth_store::AuthStore::open_default();
    if store.remove(provider_id.store_key())? {
        println!("logged out: {provider_id}");
    } else {
        println!("not logged in: {provider_id}");
    }
    Ok(())
}

/// `auth-status [provider]`
pub(crate) fn auth_status(provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let store = beyond_ai_agent::auth_store::AuthStore::open_default();
    let providers = match &provider {
        Some(p) => vec![
            beyond_ai_agent::oauth::OAuthProviderId::parse(p)
                .ok_or_else(|| unknown_provider_error(p))?,
        ],
        None => beyond_ai_agent::oauth::OAuthProviderId::all().to_vec(),
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
    let settings = beyond_ai_agent::settings::effective_settings_for_cwd(&cwd);
    let servers = settings.mcp_servers.unwrap_or_default();
    let server = servers.iter().find(|s| s.name == name).ok_or_else(|| {
        format!(
            "no MCP server named `{name}` is configured (checked global and, if this \
                 directory is trusted, project settings.json)"
        )
    })?;
    let url = match &server.transport {
        beyond_ai_agent::settings::McpTransport::Http { url, .. } => url.clone(),
        beyond_ai_agent::settings::McpTransport::Stdio { .. } => {
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
    let auth_store = beyond_ai_agent::mcp_auth_store::McpAuthStore::open_default();
    manager.set_credential_store(auth_store.scoped(&name));
    // rmcp 3.x: discovery is `resolve_metadata` (no longer `discover_metadata`).
    let resolution = manager
        .resolve_metadata()
        .await
        .map_err(|e| format!("`{name}` does not support MCP OAuth (or discovery failed): {e}"))?;
    manager.set_metadata(resolution.metadata);

    // Dynamic client registration means we choose our own redirect URI (unlike `agent
    // login`'s providers, each registered in advance against a fixed port). The callback
    // listener is bound first, on a port the kernel picks, and the URI names the port it got:
    // picking a free port, releasing it and binding it later let another process take it in
    // between, and the login then failed before printing its URL.
    let listener =
            // port-0: the listener is held for the whole login; the redirect URI names its port.
            beyond_ai_agent::oauth::callback_server::CallbackServer::bind("127.0.0.1", 0)
                .map_err(|e| {
                    format!("failed to bind the local OAuth callback listener: {e}")
                })?;
    let port = listener.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    // rmcp 3.x folds scopes + client name into `AuthorizationRequest`; empty scopes
    // auto-select via the manager's scope policy inside `AuthorizationSession::new`.
    let request = rmcp::transport::auth::AuthorizationRequest::new(&redirect_uri)
        .with_client_name("beyond-ai-agent");
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
    let store = beyond_ai_agent::mcp_auth_store::McpAuthStore::open_default();
    let had_credential = store.has_credential(&name);
    store.clear(&name)?;
    if had_credential {
        println!("logged out: {name}");
    } else {
        println!("not logged in: {name}");
    }
    Ok(())
}
