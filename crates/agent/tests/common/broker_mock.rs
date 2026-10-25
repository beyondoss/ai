//! A reference connector broker for the broker kit's own tests (`tests/broker_conformance.rs`): what
//! `BEYOND.md` asks of Beyond's broker, small, plus one broken mode per check. The kit must pass the
//! conforming broker and fail each broken one by name. Tokens are real `bmc_v1` ones.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use beyond_ai_agent::conformance::broker::Hooks;
use beyond_ai_agent::managed_mcp_token::{Claims, ManagedKey};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const LIVE: [u8; 32] = [0x77; 32];

/// How a broker gets one requirement wrong (or [`Fault::None`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    /// Mints a token that differs from the fixture's.
    MintsWrongBytes,
    /// A request naming a connection gets that connection.
    HonoursNamedConnection,
    /// The binding is looked up by session id alone, ignoring the token's tenant.
    IgnoresTenant,
    /// An expired token is accepted.
    AcceptsExpired,
    /// A revoke takes effect one call late.
    RevokeLags,
    /// Deleting a session leaves its binding.
    DeleteKeepsBinding,
    /// One tool list cache for every tenant.
    SharedCache,
    /// The upstream token is echoed to the agent.
    LeaksUpstreamToken,
}

#[derive(Default)]
struct Inner {
    /// connection -> (tenant, user, revoked, calls left before a lagging revoke bites)
    conns: HashMap<String, (String, String, bool, u32)>,
    bindings: HashMap<(String, String), String>,
    cache: Vec<String>, // tenants whose tool list is cached ("*" for the shared one)
    n: u32,
}

pub struct BrokerMock {
    pub addr: std::net::SocketAddr,
    inner: Arc<Mutex<Inner>>,
    fault: Fault,
}

impl BrokerMock {
    pub async fn start(fault: Fault) -> Self {
        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let addr = listener.local_addr().unwrap();
        let inner = Arc::new(Mutex::new(Inner::default()));
        let shared = inner.clone();
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let inner = shared.clone();
                tokio::spawn(async move { serve(sock, inner, fault).await });
            }
        });
        Self { addr, inner, fault }
    }

    pub fn url(&self) -> String {
        format!("http://{}/managed/notes/mcp", self.addr)
    }
}

fn secret(connection: &str) -> String {
    format!("upstream-access-token-{connection}-0123456789abcdef")
}

impl Hooks for BrokerMock {
    async fn mint(
        &self,
        secret_hex: Option<&str>,
        tenant: &str,
        session: &str,
        connector: &str,
        exp: u64,
    ) -> Result<String, String> {
        let key = match secret_hex {
            Some(h) => {
                let bytes: Vec<u8> = (0..h.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                    .collect();
                ManagedKey::from_secret_bytes(&bytes.try_into().unwrap())
            }
            None => ManagedKey::from_secret_bytes(&LIVE),
        };
        let exp = if self.fault == Fault::MintsWrongBytes {
            exp + 1
        } else {
            exp
        };
        Ok(key.mint(&Claims {
            tenant: tenant.into(),
            session_id: session.into(),
            connector: connector.into(),
            exp,
        }))
    }

    async fn connect(&self, tenant: &str, user: &str) -> Result<String, String> {
        let mut i = self.inner.lock().unwrap();
        i.n += 1;
        let id = format!("conn_{}", i.n);
        i.conns
            .insert(id.clone(), (tenant.into(), user.into(), false, 0));
        Ok(id)
    }

    async fn bind(&self, tenant: &str, session: &str, connection: &str) -> Result<(), String> {
        self.inner
            .lock()
            .unwrap()
            .bindings
            .insert((tenant.into(), session.into()), connection.into());
        Ok(())
    }

    async fn revoke(&self, connection: &str) -> Result<(), String> {
        let lag = self.fault == Fault::RevokeLags;
        let mut i = self.inner.lock().unwrap();
        let c = i.conns.get_mut(connection).ok_or("no such connection")?;
        if lag {
            c.3 = 1; // one more call goes through
        }
        c.2 = true;
        Ok(())
    }

    async fn delete_session(&self, tenant: &str, session: &str) -> Result<(), String> {
        if self.fault != Fault::DeleteKeepsBinding {
            self.inner
                .lock()
                .unwrap()
                .bindings
                .remove(&(tenant.to_string(), session.to_string()));
        }
        Ok(())
    }

    async fn secrets(&self, connection: &str) -> Result<Vec<String>, String> {
        Ok(vec![secret(connection)])
    }
}

async fn read_request(
    sock: &mut tokio::net::TcpStream,
) -> Option<(String, HashMap<String, String>, Vec<u8>)> {
    let mut buf = Vec::new();
    let head_end = loop {
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.lines();
    let path = lines.next()?.split(' ').nth(1)?.to_string();
    let headers: HashMap<_, _> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Some((path, headers, body))
}

async fn reply(sock: &mut tokio::net::TcpStream, status: u16, body: Value) {
    let b = body.to_string();
    let head = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        b.len()
    );
    let _ = sock.write_all(head.as_bytes()).await;
    let _ = sock.write_all(b.as_bytes()).await;
}

fn tool_error(id: &Value, text: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": {"isError": true, "content": [{"type": "text", "text": text}]}})
}

async fn serve(mut sock: tokio::net::TcpStream, inner: Arc<Mutex<Inner>>, fault: Fault) {
    let Some((path, headers, body)) = read_request(&mut sock).await else {
        return;
    };
    let token = headers
        .get("authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let key = ManagedKey::from_secret_bytes(&LIVE);
    // `verify` at time 0 checks only the MAC, so an expired token is told apart from a forged one.
    let claims = match key.verify(
        token,
        if fault == Fault::AcceptsExpired {
            0
        } else {
            now
        },
    ) {
        Ok(c) => c,
        Err(_) => return reply(&mut sock, 401, json!({"error": "unauthorized"})).await,
    };
    let rpc: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let id = rpc["id"].clone();
    let method = rpc["method"].as_str().unwrap_or("");
    if method == "ping" {
        return reply(
            &mut sock,
            200,
            json!({"jsonrpc": "2.0", "id": id, "result": {}}),
        )
        .await;
    }
    let named = headers
        .get("x-beyond-connection")
        .cloned()
        .or_else(|| path.split_once("?connection=").map(|(_, c)| c.to_string()));
    let (status, answer) = 'locked: {
        let mut i = inner.lock().unwrap();
        let bound_tenant = if fault == Fault::IgnoresTenant {
            i.bindings
                .iter()
                .find(|((_, s), _)| *s == claims.session_id)
                .map(|((t, _), _)| t.clone())
                .unwrap_or(claims.tenant.clone())
        } else {
            claims.tenant.clone()
        };
        let bound = i
            .bindings
            .get(&(bound_tenant, claims.session_id.clone()))
            .cloned();
        if let (Some(b), Some(n)) = (&bound, &named)
            && b != n
            && fault != Fault::HonoursNamedConnection
        {
            break 'locked (403, json!({"error": "not bound to that connection"}));
        }
        let using = named
            .filter(|_| fault == Fault::HonoursNamedConnection)
            .or(bound.clone());
        let active =
            using
                .as_ref()
                .and_then(|c| i.conns.get_mut(c))
                .and_then(|(_, user, revoked, lag)| {
                    if *revoked && *lag > 0 {
                        *lag -= 1;
                        Some(user.clone())
                    } else if *revoked {
                        None
                    } else {
                        Some(user.clone())
                    }
                });
        let shared = if fault == Fault::SharedCache {
            "*".to_string()
        } else {
            claims.tenant.clone()
        };
        let answer = match method {
            "tools/list" => {
                if bound.is_some() && active.is_some() {
                    i.cache.push(shared);
                    json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [{"name": "whoami", "inputSchema": {"type": "object"}}]}})
                } else if i.cache.contains(&shared) {
                    json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [{"name": "whoami", "inputSchema": {"type": "object"}}]}})
                } else {
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32001, "message": "no connection"}})
                }
            }
            _ => match (using, active) {
                (Some(conn), Some(user)) => {
                    let leak = if fault == Fault::LeaksUpstreamToken {
                        format!(" (via {})", secret(&conn))
                    } else {
                        String::new()
                    };
                    json!({"jsonrpc": "2.0", "id": id, "result": {"content": [{"type": "text", "text": format!("notes for {user}{leak}")}]}})
                }
                (None, _) => {
                    tool_error(&id, "No user has connected their account to this session.")
                }
                _ => tool_error(&id, "The connection to this account is no longer active."),
            },
        };
        (200, answer)
    };
    reply(&mut sock, status, answer).await;
}
