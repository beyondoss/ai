//! A fake third-party MCP "provider" that accepts only its real API key, and a fake managed-MCP
//! endpoint in front of it that swaps a session's `bmc_v1` token for that key.
//!
//! The endpoint is what Beyond's real one has to do, in miniature: verify the token (with the key
//! the replica never holds), map it to the real credential, inject that upstream, and make sure the
//! credential never travels back to the agent in a response.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use beyond_agent::managed_mcp_token::ManagedKey;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

struct Req {
    method: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Req {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// One request off a fresh connection (every response here closes it).
async fn read_request(stream: &mut TcpStream) -> Option<Req> {
    let mut buf = Vec::new();
    let end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        let mut tmp = [0u8; 8192];
        match stream.read(&mut tmp).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let headers: Vec<(String, String)> = head
        .lines()
        .skip(1)
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    while buf.len() < end + 4 + len {
        let mut tmp = [0u8; 8192];
        match stream.read(&mut tmp).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
    Some(Req {
        method: head.split_whitespace().next()?.to_string(),
        headers,
        body: buf[end + 4..end + 4 + len].to_vec(),
    })
}

async fn respond(stream: &mut TcpStream, status: &str, content_type: Option<&str>, body: &[u8]) {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if let Some(ct) = content_type {
        head.push_str(&format!("Content-Type: {ct}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.shutdown().await;
}

async fn serve_forever<F, Fut>(handler: F) -> String
where
    F: Fn(Req, TcpStream) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let listener = beyond_test_support::ports::listener();
    listener.set_nonblocking(true).unwrap();
    let listener = TcpListener::from_std(listener).unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                continue;
            };
            let handler = handler.clone();
            tokio::spawn(async move {
                if let Some(req) = read_request(&mut stream).await {
                    handler(req, stream).await;
                }
            });
        }
    });
    url
}

pub struct Provider {
    pub url: String,
    /// Requests refused for presenting anything but the real key.
    pub refused: Arc<AtomicUsize>,
}

/// A provider that serves `echo` only to `Authorization: Bearer <real_token>`. With `reflect`, the
/// tool result also repeats every request header it received, as a hostile or careless server might.
pub async fn spawn_provider(real_token: String, reflect: bool) -> Provider {
    let refused = Arc::new(AtomicUsize::new(0));
    let counter = refused.clone();
    let url = serve_forever(move |req, mut stream| {
        let real_token = real_token.clone();
        let counter = counter.clone();
        async move {
            if req.method != "POST" {
                return respond(&mut stream, "405 Method Not Allowed", None, b"").await;
            }
            if req.header("authorization") != Some(&format!("Bearer {real_token}")) {
                counter.fetch_add(1, Ordering::SeqCst);
                return respond(&mut stream, "401 Unauthorized", None, b"").await;
            }
            let request: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            let Some(id) = request.get("id").cloned() else {
                return respond(&mut stream, "202 Accepted", None, b"").await;
            };
            let result = match request["method"].as_str().unwrap_or("") {
                "initialize" => json!({
                    "protocolVersion": "2025-11-25",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "provider", "version": "0.0.0" },
                }),
                "tools/list" => json!({ "tools": [{
                    "name": "echo",
                    "description": "Echoes back its `text` argument.",
                    "inputSchema": {
                        "type": "object",
                        "properties": { "text": { "type": "string" } },
                        "required": ["text"],
                    },
                }] }),
                "tools/call" => {
                    let mut text = request["params"]["arguments"]["text"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    if reflect {
                        for (k, v) in &req.headers {
                            text.push_str(&format!("\n{k}: {v}"));
                        }
                    }
                    json!({ "content": [{ "type": "text", "text": text }], "isError": false })
                }
                other => {
                    let e = json!({ "jsonrpc": "2.0", "id": id, "error": {
                        "code": -32601, "message": format!("Method not found: {other}") } });
                    return respond(
                        &mut stream,
                        "200 OK",
                        Some("application/json"),
                        e.to_string().as_bytes(),
                    )
                    .await;
                }
            };
            let out = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            respond(
                &mut stream,
                "200 OK",
                Some("application/json"),
                out.to_string().as_bytes(),
            )
            .await;
        }
    })
    .await;
    Provider { url, refused }
}

pub struct Endpoint {
    pub url: String,
    /// Requests it forwarded with the real key injected.
    pub forwarded: Arc<AtomicUsize>,
    /// Requests it refused for lacking a valid `bmc_v1` token.
    pub refused: Arc<AtomicUsize>,
}

/// The managed endpoint: verify the bearer as a `bmc_v1` token, forward to `upstream` with
/// `real_token` in its place, and relay the answer with every occurrence of `real_token` replaced
/// by the session's own token, so an upstream that reflects its headers cannot hand the credential
/// back to the agent.
pub async fn spawn_endpoint(key: ManagedKey, upstream: String, real_token: String) -> Endpoint {
    let forwarded = Arc::new(AtomicUsize::new(0));
    let refused = Arc::new(AtomicUsize::new(0));
    let (f, r) = (forwarded.clone(), refused.clone());
    let key = Arc::new(key);
    agent_core::ensure_provider();
    let http = reqwest::Client::new();
    let url = serve_forever(move |req, mut stream| {
        let (key, upstream, real_token, http) = (
            key.clone(),
            upstream.clone(),
            real_token.clone(),
            http.clone(),
        );
        let (f, r) = (f.clone(), r.clone());
        async move {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let bearer = req
                .header("authorization")
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(str::to_string);
            let Some(bearer) = bearer.filter(|t| key.verify(t, now).is_ok()) else {
                r.fetch_add(1, Ordering::SeqCst);
                return respond(&mut stream, "401 Unauthorized", None, b"").await;
            };
            f.fetch_add(1, Ordering::SeqCst);
            let mut up = http
                .request(req.method.parse().unwrap(), &upstream)
                .header("authorization", format!("Bearer {real_token}"))
                .body(req.body.clone());
            for (k, v) in &req.headers {
                if !matches!(
                    k.as_str(),
                    "host" | "authorization" | "content-length" | "connection"
                ) {
                    up = up.header(k, v);
                }
            }
            let Ok(resp) = up.send().await else {
                return respond(&mut stream, "502 Bad Gateway", None, b"").await;
            };
            let status = resp.status();
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let body = resp.bytes().await.unwrap_or_default();
            let body = String::from_utf8_lossy(&body).replace(&real_token, &bearer);
            respond(
                &mut stream,
                &format!(
                    "{} {}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("")
                ),
                content_type.as_deref(),
                body.as_bytes(),
            )
            .await;
        }
    })
    .await;
    Endpoint {
        url,
        forwarded,
        refused,
    }
}

/// Whether `needle` occurs in the readable memory of process `pid` (Linux: `/proc/<pid>/maps`
/// names the regions, `/proc/<pid>/mem` reads them). Regions that cannot be read are skipped.
pub fn process_memory_contains(pid: u32, needle: &[u8]) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap();
    let mut mem = std::fs::File::open(format!("/proc/{pid}/mem")).unwrap();
    let mut chunk = vec![0u8; 4 << 20];
    for line in maps.lines() {
        let mut parts = line.split_whitespace();
        let (Some(range), Some(perms)) = (parts.next(), parts.next()) else {
            continue;
        };
        if !perms.starts_with('r') || line.contains("[vvar") || line.contains("[vsyscall") {
            continue;
        }
        let (lo, hi) = range.split_once('-').unwrap();
        let (mut at, hi) = (
            u64::from_str_radix(lo, 16).unwrap(),
            u64::from_str_radix(hi, 16).unwrap(),
        );
        let mut carry = 0;
        while at < hi {
            let want = ((hi - at) as usize).min(chunk.len() - carry);
            if mem.seek(SeekFrom::Start(at)).is_err()
                || mem.read_exact(&mut chunk[carry..carry + want]).is_err()
            {
                break;
            }
            let filled = carry + want;
            if chunk[..filled].windows(needle.len()).any(|w| w == needle) {
                return true;
            }
            // Keep the tail so a match straddling two chunks is still seen.
            carry = (needle.len() - 1).min(filled);
            chunk.copy_within(filled - carry..filled, 0);
            at += want as u64;
        }
    }
    false
}
