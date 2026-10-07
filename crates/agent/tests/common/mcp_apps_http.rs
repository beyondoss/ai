//! A minimal streamable-HTTP MCP Apps server — what a service-mode session's grant connector dials.
//!
//! Like [`super::mcp_fixture`], a blocking `TcpListener` thread answering one JSON-RPC request per
//! connection. It speaks only the `2026-07-28` lifecycle (`server/discover`), so every request
//! carries the client's capabilities in `_meta` and the server can tell, per request, whether the
//! MCP Apps extension was advertised — the plain and the apps-flavored connections of one session
//! dial the same URL. With the extension it lists `show_weather` with a `ui://` view; without, the
//! same tool as plain text.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use serde_json::{Value, json};

const VIEW_URI: &str = "ui://apps-http/weather";

#[derive(Clone)]
pub struct HttpAppsFixture {
    pub url: String,
    ui_requests: Arc<AtomicUsize>,
    plain_requests: Arc<AtomicUsize>,
}

impl HttpAppsFixture {
    /// Requests whose client advertised `io.modelcontextprotocol/ui`.
    pub fn ui_requests(&self) -> usize {
        self.ui_requests.load(Ordering::SeqCst)
    }

    /// Requests whose client did not.
    pub fn plain_requests(&self) -> usize {
        self.plain_requests.load(Ordering::SeqCst)
    }
}

fn advertises_ui(request: &Value) -> bool {
    request
        .pointer("/params/_meta/io.modelcontextprotocol~1clientCapabilities/extensions/io.modelcontextprotocol~1ui/mimeTypes")
        .and_then(Value::as_array)
        .is_some_and(|t| t.iter().any(|m| m == "text/html;profile=mcp-app"))
}

pub fn spawn_http_apps_fixture() -> HttpAppsFixture {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let ui_requests = Arc::new(AtomicUsize::new(0));
    let plain_requests = Arc::new(AtomicUsize::new(0));
    let (ui, plain) = (ui_requests.clone(), plain_requests.clone());
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
            let headers = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
            let length: usize = headers
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap_or(0))
                })
                .unwrap_or(0);
            while buf.len() < pos + 4 + length {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = &buf[pos + 4..buf.len().min(pos + 4 + length)];
            let request: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
            let has_ui = advertises_ui(&request);
            if has_ui {
                ui.fetch_add(1, Ordering::SeqCst);
            } else {
                plain.fetch_add(1, Ordering::SeqCst);
            }
            let Some(id) = request.get("id").cloned() else {
                let _ = stream.write_all(
                    b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            };
            let method = request.get("method").and_then(Value::as_str).unwrap_or("");
            let result = match method {
                "server/discover" => json!({
                    "resultType": "complete",
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": { "tools": {}, "resources": {} },
                    "ttlMs": 0,
                    "cacheScope": "private",
                    "_meta": { "io.modelcontextprotocol/serverInfo":
                               { "name": "mcp-apps-http-fixture", "version": "0.0.0" } },
                }),
                "tools/list" => {
                    let mut tool = json!({
                        "name": "show_weather",
                        "description": "Show the weather.",
                        "inputSchema": { "type": "object", "properties": {} },
                    });
                    if has_ui {
                        tool["_meta"] = json!({ "ui": { "resourceUri": VIEW_URI } });
                    }
                    json!({ "tools": [tool] })
                }
                "resources/list" => json!({ "resources": [] }),
                "prompts/list" => json!({ "prompts": [] }),
                "resources/read" => json!({ "contents": [{
                    "uri": VIEW_URI,
                    "mimeType": "text/html;profile=mcp-app",
                    "text": "<!DOCTYPE html><html><body>service view</body></html>",
                }] }),
                "tools/call" => json!({
                    "content": [{ "type": "text", "text": "service-weather" }],
                    "structuredContent": { "where": "service" },
                }),
                _ => {
                    write_json(
                        &mut stream,
                        &json!({ "jsonrpc": "2.0", "id": id,
                                 "error": { "code": -32601, "message": "Method not found" } }),
                    );
                    continue;
                }
            };
            write_json(
                &mut stream,
                &json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            );
        }
    });
    HttpAppsFixture {
        url: format!("http://{addr}/mcp"),
        ui_requests,
        plain_requests,
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
