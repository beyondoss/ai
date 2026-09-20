//! A minimal streamable-HTTP MCP server, and a record of every request header it saw.
//!
//! A plain blocking `TcpListener` thread (matching [`super::spawn_model_server`]'s own idiom — this
//! transport has no subprocess to spawn, so a hand-rolled server has to live somewhere). It handles
//! exactly what one client handshake plus a tool call needs: `server/discover`, `initialize`, the
//! `notifications/initialized` notification (a bodyless `202 Accepted`, matching
//! `StreamableHttpPostResponse::Accepted`), `tools/list` and `tools/call`. It relies on `rmcp`'s
//! client defaulting to `allow_stateless: true`, so no `Mcp-Session-Id` handshake is needed.
//!
//! It records **every request header**, not just one: that is what lets a test prove a credential
//! from a session grant reached the wire exactly as sealed (`!echo pwned` stays eleven literal
//! bytes), and that a connector nobody granted was never dialed at all.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};

/// One request's headers: `(lowercased name, value)`, in arrival order.
type RequestHeaders = Vec<(String, String)>;

/// A running fixture: where to reach it, and what it has been sent.
#[derive(Clone)]
pub struct HttpMcpFixture {
    /// The MCP endpoint, as a connector URL.
    pub url: String,
    /// One entry per request it has answered.
    requests: Arc<Mutex<Vec<RequestHeaders>>>,
}

impl HttpMcpFixture {
    /// How many requests reached this server. `0` proves it was never dialed.
    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    /// Every value this server ever saw for `name` (case-insensitive), in arrival order.
    pub fn header_values(&self, name: &str) -> Vec<String> {
        let name = name.to_ascii_lowercase();
        self.requests
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .filter(|(k, _)| *k == name)
            .map(|(_, v)| v.clone())
            .collect()
    }

    /// Whether any request carried `name` at all.
    pub fn saw_header(&self, name: &str) -> bool {
        !self.header_values(name).is_empty()
    }
}

/// Start one fixture on a loopback port. The thread runs for the test process's life.
pub fn spawn_http_mcp_fixture() -> HttpMcpFixture {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorder = requests.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let mut header_end = None;
            while header_end.is_none() {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                header_end = buf.windows(4).position(|w| w == b"\r\n\r\n");
            }
            let Some(pos) = header_end else { continue };
            let headers_text = String::from_utf8_lossy(&buf[..pos]).into_owned();
            let content_length: usize = headers_text
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap_or(0))
                })
                .unwrap_or(0);
            let body_start = pos + 4;
            while buf.len() < body_start + content_length {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            // Every header, verbatim — the name lowercased so a lookup is case-insensitive, the
            // value untouched so a test can assert on the exact bytes that arrived.
            let headers: RequestHeaders = headers_text
                .lines()
                .skip(1)
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                .collect();
            recorder.lock().unwrap().push(headers);

            let body = &buf[body_start..buf.len().min(body_start + content_length)];
            let request: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
            let is_notification = request.get("id").is_none();
            if is_notification {
                let _ = stream.write_all(
                    b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            }
            let id = request.get("id").cloned().unwrap_or(Value::Null);
            let method = request.get("method").and_then(Value::as_str).unwrap_or("");
            let result = match method {
                "server/discover" => json!({
                    "resultType": "complete",
                    "supportedVersions": ["2026-07-28", "2025-11-25"],
                    "capabilities": { "tools": {} },
                    "ttlMs": 0,
                    "cacheScope": "private",
                    "_meta": {
                        "io.modelcontextprotocol/serverInfo": {
                            "name": "mcp-fixture-http-server",
                            "version": "0.0.0",
                        }
                    }
                }),
                "initialize" => json!({
                    "protocolVersion": "2025-11-25",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "mcp-fixture-http-server", "version": "0.0.0" },
                }),
                "tools/list" => json!({ "tools": [
                    {
                        "name": "echo",
                        "description": "Echoes back its `text` argument.",
                        "inputSchema": {
                            "type": "object",
                            "properties": { "text": { "type": "string" } },
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "fail",
                        "description": "Always fails, with an error message.",
                        "inputSchema": { "type": "object", "properties": {} },
                    },
                ] }),
                "tools/call" => {
                    let name = request
                        .pointer("/params/name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if name == "fail" {
                        json!({
                            "content": [{ "type": "text", "text": "intentional http failure" }],
                            "isError": true,
                        })
                    } else {
                        let text = request
                            .pointer("/params/arguments/text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        json!({ "content": [{ "type": "text", "text": text }], "isError": false })
                    }
                }
                _ => {
                    // Method not found — keep the Auto discover→initialize fallback fast if
                    // discover is ever removed from this fixture again.
                    let envelope = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("Method not found: {method}") },
                    });
                    write_json(&mut stream, &envelope);
                    continue;
                }
            };
            write_json(
                &mut stream,
                &json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            );
        }
    });
    HttpMcpFixture {
        url: format!("http://{addr}/mcp"),
        requests,
    }
}

fn write_json(stream: &mut std::net::TcpStream, value: &Value) {
    let encoded = serde_json::to_vec(value).unwrap();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        encoded.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(&encoded);
    let _ = stream.flush();
}
