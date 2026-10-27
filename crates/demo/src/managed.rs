//! The demo's stand-in for Beyond's managed-MCP endpoint, and the "third party" behind it.
//!
//! A managed connector (`McpConnector::managed`) points the agent at an endpoint that holds the
//! real third-party credential, so the agent only ever presents a `bmc_v1` token. This is a small,
//! honest version of that endpoint, in one process with two listeners:
//!
//! - **managed** (public): `POST /managed/<connector>/mcp`. Verifies the `bmc_v1` token with the
//!   agent crate's own [`ManagedKey`] (the same code the golden vectors pin), checks the token names
//!   this connector, then forwards the JSON-RPC body to the provider with the real secret added.
//! - **provider** (loopback only): `POST /mcp`, a tiny MCP server (`account_status`) that answers
//!   only a request carrying `Authorization: Bearer <the real secret>`.
//!
//! The real secret is in this process's environment and nowhere else: not in a grant, not in an
//! image, not in a response. Every request is one timeline line on stdout (never a token or the
//! secret).
//!
//! The HTTP is hand-rolled (`Connection: close`, `Content-Length` bodies): the demo needs a handful
//! of JSON-RPC POSTs answered, not a web framework.

use std::sync::Arc;

use beyond_agent::managed_mcp_token::{ManagedKey, Reject};
use serde_json::{Value, json};
use tokio::net::{TcpListener, TcpStream};

use crate::http::{
    Req, Res, constant_time_eq, error_body, json_headers, now_ms, post, read_request, respond,
};

pub struct Args {
    /// Where the managed endpoint listens (public).
    pub listen: String,
    /// Where the provider listens (loopback; the managed endpoint forwards to it).
    pub provider_listen: String,
}

/// Both secrets come from the environment, never argv: `DEMO_MANAGED_SECRET` (base64 of the 32
/// bytes the minter and this endpoint share) and `DEMO_PROVIDER_SECRET` (the real third-party key).
pub async fn run(args: Args) -> Res<()> {
    let managed_secret =
        std::env::var("DEMO_MANAGED_SECRET").map_err(|_| "DEMO_MANAGED_SECRET is not set")?;
    let provider_secret =
        std::env::var("DEMO_PROVIDER_SECRET").map_err(|_| "DEMO_PROVIDER_SECRET is not set")?;
    let standin = Standin::start(
        &args.listen,
        &args.provider_listen,
        &managed_secret,
        provider_secret,
    )
    .await?;
    line(
        "listening",
        json!({"managed": standin.managed_addr, "provider": standin.provider_addr}),
    );
    std::future::pending::<()>().await;
    Ok(())
}

/// A running stand-in: the addresses it really bound.
pub struct Standin {
    pub managed_addr: String,
    pub provider_addr: String,
}

struct Shared {
    key: ManagedKey,
    provider_secret: String,
    provider_addr: String,
    /// What a provider answer carries that the model could not have guessed: new at every start.
    vault_ref: String,
}

impl Standin {
    pub async fn start(
        managed: &str,
        provider: &str,
        managed_secret_b64: &str,
        provider_secret: String,
    ) -> Res<Self> {
        let managed_l = TcpListener::bind(managed).await?;
        let provider_l = TcpListener::bind(provider).await?;
        Self::serve(managed_l, provider_l, managed_secret_b64, provider_secret)
    }

    /// Serve on listeners the caller already holds.
    pub fn serve(
        managed_l: TcpListener,
        provider_l: TcpListener,
        managed_secret_b64: &str,
        provider_secret: String,
    ) -> Res<Self> {
        use base64::Engine as _;
        let secret: [u8; 32] = base64::engine::general_purpose::STANDARD
            .decode(managed_secret_b64.trim())?
            .try_into()
            .map_err(|_| "DEMO_MANAGED_SECRET is not base64 of 32 bytes")?;
        let mut nonce = [0u8; 6];
        getrandom::fill(&mut nonce)?;
        let shared = Arc::new(Shared {
            key: ManagedKey::from_secret_bytes(&secret),
            provider_secret,
            provider_addr: provider_l.local_addr()?.to_string(),
            vault_ref: format!(
                "VAULT-{}",
                nonce.iter().map(|b| format!("{b:02x}")).collect::<String>()
            ),
        });
        let out = Self {
            managed_addr: managed_l.local_addr()?.to_string(),
            provider_addr: shared.provider_addr.clone(),
        };
        tokio::spawn(accept(managed_l, shared.clone(), Side::Managed));
        tokio::spawn(accept(provider_l, shared, Side::Provider));
        Ok(out)
    }
}

#[derive(Clone, Copy)]
enum Side {
    Managed,
    Provider,
}

async fn accept(l: TcpListener, shared: Arc<Shared>, side: Side) {
    loop {
        let Ok((s, _)) = l.accept().await else {
            continue;
        };
        let shared = shared.clone();
        tokio::spawn(async move {
            let _ = serve(s, shared, side).await;
        });
    }
}

fn line(event: &str, detail: Value) {
    println!(
        "{}",
        json!({"ts_ms": now_ms(), "src": "standin", "event": event, "detail": detail})
    );
}

async fn serve(mut s: TcpStream, shared: Arc<Shared>, side: Side) -> Res<()> {
    let Some(req) = read_request(&mut s).await? else {
        return Ok(());
    };
    let (status, headers, body) = match side {
        Side::Managed => managed(&shared, &req).await,
        Side::Provider => provider(&shared, &req),
    };
    respond(&mut s, status, &headers, &body).await
}

/// The managed endpoint: verify, then forward with the real secret.
async fn managed(sh: &Shared, req: &Req) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let Some(connector) = req
        .path
        .strip_prefix("/managed/")
        .and_then(|p| p.strip_suffix("/mcp"))
        .filter(|c| !c.is_empty() && !c.contains('/'))
    else {
        return (404, json_headers(), error_body("not found"));
    };
    let refuse = |why: &str| {
        line(
            "managed_refused",
            json!({"connector": connector, "reason": why}),
        );
        (401, json_headers(), error_body("unauthorized"))
    };
    // A stateless endpoint: there is no standalone stream to open and no session to end.
    if req.method == "GET" {
        return (405, json_headers(), error_body("no standalone stream"));
    }
    if req.method == "DELETE" {
        return (200, json_headers(), Vec::new());
    }
    if req.method != "POST" {
        return (405, json_headers(), error_body("method not allowed"));
    }
    let Some(token) = req
        .header("authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
    else {
        return refuse("no bearer token");
    };
    let claims = match sh.key.verify(token, now_ms() / 1000) {
        Ok(c) => c,
        Err(e) => {
            return refuse(match e {
                Reject::Malformed => "malformed token",
                Reject::UnknownKid => "unknown kid",
                Reject::BadMac => "bad mac",
                Reject::Expired => "expired",
            });
        }
    };
    if claims.connector != connector {
        return refuse("token names another connector");
    }
    let mut fwd = vec![(
        "authorization".to_string(),
        format!("Bearer {}", sh.provider_secret),
    )];
    for name in [
        "content-type",
        "accept",
        "mcp-session-id",
        "mcp-protocol-version",
    ] {
        if let Some(v) = req.header(name) {
            fwd.push((name.to_string(), v.to_string()));
        }
    }
    let method = serde_json::from_slice::<Value>(&req.body)
        .ok()
        .and_then(|v| v["method"].as_str().map(str::to_owned))
        .unwrap_or_default();
    match post(&sh.provider_addr, "/mcp", &fwd, &req.body).await {
        Ok((status, headers, body)) => {
            line(
                "managed_call",
                json!({"connector": connector, "tenant": claims.tenant, "session": claims.session_id,
                       "method": method, "upstream_status": status}),
            );
            let keep: Vec<_> = headers
                .into_iter()
                .filter(|(n, _)| n == "content-type" || n == "mcp-session-id")
                .collect();
            (status, keep, body)
        }
        Err(e) => {
            line(
                "managed_upstream_error",
                json!({"connector": connector, "error": e.to_string()}),
            );
            (502, json_headers(), error_body("upstream unavailable"))
        }
    }
}

/// The "third party": a one-tool MCP server that wants `Authorization: Bearer <the real secret>`.
fn provider(sh: &Shared, req: &Req) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let given = req
        .header("authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if req.path != "/mcp" || req.method != "POST" {
        return (404, json_headers(), error_body("not found"));
    }
    if !constant_time_eq(given.as_bytes(), sh.provider_secret.as_bytes()) {
        line("provider_call", json!({"auth": "refused"}));
        return (401, json_headers(), error_body("invalid api key"));
    }
    let Ok(rpc) = serde_json::from_slice::<Value>(&req.body) else {
        return (400, json_headers(), error_body("not json"));
    };
    let method = rpc["method"].as_str().unwrap_or("");
    line("provider_call", json!({"auth": "ok", "method": method}));
    let Some(id) = rpc.get("id").cloned() else {
        return (202, json_headers(), Vec::new());
    };
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": rpc["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "acme-vault", "version": "0.1.0"},
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": [{
            "name": "account_status",
            "description": "Look up a customer's account in the Acme vault: plan, seats and the vault reference.",
            "inputSchema": {"type": "object",
                "properties": {"customer": {"type": "string", "description": "the customer's name"}},
                "required": ["customer"]},
        }]})),
        "tools/call" if rpc["params"]["name"] == "account_status" => {
            let customer = rpc["params"]["arguments"]["customer"]
                .as_str()
                .unwrap_or("unknown");
            Ok(json!({"content": [{"type": "text", "text": format!(
                "customer {customer}: plan=pro, seats=12, vault_ref={}, upstream_auth=verified",
                sh.vault_ref
            )}], "isError": false}))
        }
        _ => Err(json!({"code": -32601, "message": format!("method not found: {method}")})),
    };
    let body = match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    };
    (200, json_headers(), body.to_string().into_bytes())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use beyond_agent::managed_mcp_token::Claims;

    const SECRET_B64: &str = "WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo="; // 0x5a * 32
    const REAL: &str = "psk_live_not_for_the_agent";

    async fn up() -> Standin {
        use beyond_test_support::ports;
        Standin::serve(
            ports::tokio_listener().await,
            ports::tokio_listener().await,
            SECRET_B64,
            REAL.into(),
        )
        .unwrap()
    }

    fn token(connector: &str, exp: u64) -> String {
        ManagedKey::from_secret_bytes(&[0x5a; 32]).mint(&Claims {
            tenant: "t1".into(),
            session_id: "demo.r1".into(),
            connector: connector.into(),
            exp,
        })
    }

    fn far() -> u64 {
        now_ms() / 1000 + 3600
    }

    fn bearer(t: &str) -> Vec<(String, String)> {
        vec![
            ("authorization".into(), format!("Bearer {t}")),
            ("content-type".into(), "application/json".into()),
        ]
    }

    fn rpc(method: &str, params: Value) -> Vec<u8> {
        json!({"jsonrpc":"2.0","id":1,"method":method,"params":params})
            .to_string()
            .into_bytes()
    }

    #[tokio::test]
    async fn a_token_for_the_connector_reaches_the_provider_and_the_secret_stays_there() {
        let s = up().await;
        let h = bearer(&token("vault", far()));
        let call = rpc(
            "tools/call",
            json!({"name": "account_status", "arguments": {"customer": "acme"}}),
        );
        let (status, headers, body) = post(&s.managed_addr, "/managed/vault/mcp", &h, &call)
            .await
            .unwrap();
        assert_eq!(status, 200);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("vault_ref=VAULT-"), "{text}");
        assert!(text.contains("upstream_auth=verified"), "{text}");
        let everything = format!("{headers:?}{text}");
        assert!(!everything.contains(REAL), "the secret reached the caller");
    }

    #[tokio::test]
    async fn the_provider_refuses_a_caller_without_the_real_secret() {
        let s = up().await;
        let call = rpc("tools/list", json!({}));
        let (status, ..) = post(&s.provider_addr, "/mcp", &bearer("nope"), &call)
            .await
            .unwrap();
        assert_eq!(status, 401);
        let (status, ..) = post(&s.provider_addr, "/mcp", &bearer(REAL), &call)
            .await
            .unwrap();
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn a_bad_token_is_refused_before_anything_is_forwarded() {
        let s = up().await;
        let call = rpc("tools/list", json!({}));
        let other_key = ManagedKey::from_secret_bytes(&[0x5b; 32]).mint(&Claims {
            tenant: "t1".into(),
            session_id: "demo.r1".into(),
            connector: "vault".into(),
            exp: far(),
        });
        for (what, t, path) in [
            (
                "another connector's token",
                token("other", far()),
                "/managed/vault/mcp",
            ),
            ("an expired token", token("vault", 1), "/managed/vault/mcp"),
            ("another key's token", other_key, "/managed/vault/mcp"),
            (
                "a token that is not one",
                "bmc_v1.x".into(),
                "/managed/vault/mcp",
            ),
        ] {
            let (status, ..) = post(&s.managed_addr, path, &bearer(&t), &call)
                .await
                .unwrap();
            assert_eq!(status, 401, "{what}");
        }
        let no_auth = vec![("content-type".to_string(), "application/json".to_string())];
        let (status, ..) = post(&s.managed_addr, "/managed/vault/mcp", &no_auth, &call)
            .await
            .unwrap();
        assert_eq!(status, 401, "no token");
    }

    #[tokio::test]
    async fn the_handshake_a_real_client_makes_is_answered() {
        let s = up().await;
        let h = bearer(&token("vault", far()));
        let (status, _, body) = post(
            &s.managed_addr,
            "/managed/vault/mcp",
            &h,
            &rpc("initialize", json!({"protocolVersion": "2025-06-18"})),
        )
        .await
        .unwrap();
        assert_eq!(status, 200);
        assert!(String::from_utf8(body).unwrap().contains("acme-vault"));
        let note = json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string();
        let (status, ..) = post(&s.managed_addr, "/managed/vault/mcp", &h, note.as_bytes())
            .await
            .unwrap();
        assert_eq!(status, 202);
        let (_, _, body) = post(
            &s.managed_addr,
            "/managed/vault/mcp",
            &h,
            &rpc("tools/list", json!({})),
        )
        .await
        .unwrap();
        assert!(String::from_utf8(body).unwrap().contains("account_status"));
    }
}
