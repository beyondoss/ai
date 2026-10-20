//! A streamable-HTTP MCP server that hands out a session id, and counts what a client holds open.
//!
//! [`super::mcp_fixture`] is stateless (no `Mcp-Session-Id`), which is exactly the shape that never
//! makes a client open the standalone `GET` stream. This one issues a session id on `initialize`,
//! like a server that keeps per-session state, and answers a `GET` with an event stream it holds
//! open. It counts the `GET`s it was sent and the TCP connections open right now, which is what a
//! sleeping sandbox's idle detector sees.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Clone)]
pub struct SessionMcpFixture {
    /// The MCP endpoint, as a connector URL.
    pub url: String,
    gets: Arc<AtomicUsize>,
    open: Arc<AtomicUsize>,
}

impl SessionMcpFixture {
    /// Standalone-stream `GET`s received so far.
    pub fn gets(&self) -> usize {
        self.gets.load(Ordering::SeqCst)
    }

    /// TCP connections open at this instant.
    pub fn open_connections(&self) -> usize {
        self.open.load(Ordering::SeqCst)
    }
}

pub async fn spawn_session_mcp_fixture() -> SessionMcpFixture {
    let listener = beyond_ai_test_support::ports::listener();
    listener.set_nonblocking(true).unwrap();
    let listener = TcpListener::from_std(listener).unwrap();
    let addr = listener.local_addr().unwrap();
    let fixture = SessionMcpFixture {
        url: format!("http://{addr}/mcp"),
        gets: Arc::default(),
        open: Arc::default(),
    };
    let state = fixture.clone();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            let state = state.clone();
            state.open.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                serve(stream, &state).await;
                state.open.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    fixture
}

/// One connection, keep-alive: requests until the client closes it.
async fn serve(mut stream: TcpStream, state: &SessionMcpFixture) {
    let mut buf = Vec::new();
    loop {
        let end = loop {
            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break p;
            }
            let mut tmp = [0u8; 8192];
            match stream.read(&mut tmp).await {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        };
        let head = String::from_utf8_lossy(&buf[..end]).into_owned();
        let len: usize = head
            .lines()
            .find_map(|l| {
                l.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse().unwrap_or(0))
            })
            .unwrap_or(0);
        while buf.len() < end + 4 + len {
            let mut tmp = [0u8; 8192];
            match stream.read(&mut tmp).await {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        }
        let body: Vec<u8> = buf[end + 4..end + 4 + len].to_vec();
        buf.drain(..end + 4 + len);
        let ok = match head.split_whitespace().next().unwrap_or("") {
            "GET" => {
                state.gets.fetch_add(1, Ordering::SeqCst);
                let h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n: open\n\n";
                if stream.write_all(h.as_bytes()).await.is_err() {
                    return;
                }
                // Held until the client hangs up.
                let mut tmp = [0u8; 64];
                while matches!(stream.read(&mut tmp).await, Ok(n) if n > 0) {}
                return;
            }
            "DELETE" => respond(&mut stream, "200 OK", &[], b"").await,
            _ => post(&mut stream, &body).await,
        };
        if !ok {
            return;
        }
    }
}

async fn post(stream: &mut TcpStream, body: &[u8]) -> bool {
    let request: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let Some(id) = request.get("id").cloned() else {
        return respond(stream, "202 Accepted", &[], b"").await;
    };
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let mut extra = Vec::new();
    let result = match method {
        "initialize" => {
            extra.push(("Mcp-Session-Id", "fixture-session"));
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "mcp-session-fixture", "version": "0.0.0" },
            })
        }
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
            let text = request
                .pointer("/params/arguments/text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            json!({ "content": [{ "type": "text", "text": text }], "isError": false })
        }
        _ => {
            let e = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("Method not found: {method}") },
            });
            return respond(stream, "200 OK", &[], &serde_json::to_vec(&e).unwrap()).await;
        }
    };
    let out = serde_json::to_vec(&json!({ "jsonrpc": "2.0", "id": id, "result": result })).unwrap();
    respond(stream, "200 OK", &extra, &out).await
}

async fn respond(
    stream: &mut TcpStream,
    status: &str,
    extra: &[(&str, &str)],
    body: &[u8],
) -> bool {
    let mut head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n", body.len());
    if !body.is_empty() {
        head.push_str("Content-Type: application/json\r\n");
    }
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).await.is_ok() && stream.write_all(body).await.is_ok()
}
