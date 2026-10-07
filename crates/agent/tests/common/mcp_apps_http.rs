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
const HUGE_URI: &str = "ui://apps-http/huge";
/// The unadvertised oversized view `huge_view` serves: far past any host's view cap, and past what
/// loopback socket buffers absorb, so a client that refuses it unread leaves most of it unsent.
pub const HUGE_VIEW_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct HttpAppsFixture {
    pub url: String,
    ui_requests: Arc<AtomicUsize>,
    plain_requests: Arc<AtomicUsize>,
    huge_sent: Arc<AtomicUsize>,
}

impl HttpAppsFixture {
    /// How many bytes of `huge_view`'s view the last read actually got onto the wire before the
    /// client stopped reading (`HUGE_VIEW_BYTES` plus framing when read whole).
    pub fn huge_bytes_sent(&self) -> usize {
        self.huge_sent.load(Ordering::SeqCst)
    }

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
    let huge_sent = Arc::new(AtomicUsize::new(0));
    let huge = huge_sent.clone();
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
                    let mut tools = vec![];
                    if has_ui {
                        tool["_meta"] = json!({ "ui": { "resourceUri": VIEW_URI } });
                        tools.push(json!({
                            "name": "huge_view",
                            "description": "A tool whose (unadvertised) view is 64 MiB.",
                            "inputSchema": { "type": "object", "properties": {} },
                            "_meta": { "ui": { "resourceUri": HUGE_URI } },
                        }));
                    }
                    tools.insert(0, tool);
                    json!({ "tools": tools })
                }
                "resources/list" => json!({ "resources": [] }),
                "prompts/list" => json!({ "prompts": [] }),
                "resources/read" if request.pointer("/params/uri") == Some(&json!(HUGE_URI)) => {
                    let text = "x".repeat(HUGE_VIEW_BYTES);
                    let body = json!({ "jsonrpc": "2.0", "id": id, "result": { "contents": [{
                        "uri": HUGE_URI, "mimeType": "text/html;profile=mcp-app", "text": text,
                    }] } });
                    let encoded = serde_json::to_vec(&body).unwrap();
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        encoded.len()
                    );
                    huge.store(0, Ordering::SeqCst);
                    if stream.write_all(header.as_bytes()).is_ok() {
                        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(3)));
                        for chunk in encoded.chunks(64 * 1024) {
                            match stream.write_all(chunk) {
                                Ok(()) => {
                                    huge.fetch_add(chunk.len(), Ordering::SeqCst);
                                }
                                Err(_) => break,
                            }
                        }
                    }
                    continue;
                }
                "resources/read" => json!({ "contents": [{
                    "uri": VIEW_URI,
                    "mimeType": "text/html;profile=mcp-app",
                    "text": "<!DOCTYPE html><html><body>service view</body></html>",
                }] }),
                "tools/call" if request.pointer("/params/name") == Some(&json!("huge_view")) => {
                    json!({
                        "content": [{ "type": "text", "text": "huge-view-text" }],
                    })
                }
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
        huge_sent,
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
