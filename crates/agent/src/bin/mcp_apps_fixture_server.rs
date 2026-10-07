//! A hand-rolled MCP Apps (SEP-1865) server over stdio — a test fixture for
//! `crates/agent/tests/mcp_apps.rs`, not a real server. Dependency-free beyond `tokio`/`serde_json`,
//! like `mcp_fixture_stdio_server`, so it exercises the real client against the real wire.
//!
//! It behaves the way the spec asks a server to: it checks whether the client advertised
//! `io.modelcontextprotocol/ui` (on `initialize`, or per request under the `2026-07-28` lifecycle)
//! and only then lists UI metadata and its app-only tool — otherwise it is a plain text server.
//!
//! Tools (UI variants only when the extension was advertised):
//! - `show_weather {city}` — `_meta.ui.resourceUri = ui://apps-fixture/weather`; text content plus a
//!   `structuredContent.secret` that must reach the view and never the model.
//! - `refresh_weather {city}` — `visibility: ["app"]`; listed only when negotiated.
//! - `model_only_tool` — `visibility: ["model"]`; an app may not call it.
//! - `slow_view {ms}` — a view-bearing tool that sleeps first (view-before-result, abort).
//! - `broken_view` — names a view resource that does not exist (text-only fallback).
//! - `plain_echo {text}` — no UI at all.
//! - `app_only_always` — `visibility: ["app"]`, listed **whether or not** the extension was
//!   advertised, as the official SDK's `registerAppTool` does: a host must hide it from the model
//!   in every session.
//! - `huge_view` — a view bigger than any host should accept (5 MiB), listed in `resources/list`
//!   with its `size`, so a host can refuse it unread.
//! - `huge_unlisted_view` — the same 5 MiB view with no advertised size: a host must read it to know,
//!   and must still not cache it.
//! - `bad_uri_view` — a `resourceUri` that is not `ui://` (a host must not treat it as a view).
//! - `ask_view` — asks the client a nested `elicitation/create` mid-call and answers with what came
//!   back (`asked:<name>` or `asked:<action>`): who answers shows which session the host routed it to.
//! - `change_view` — bumps the view's version (its HTML then names `v<n>`) and sends
//!   `notifications/resources/list_changed` before answering, so a host's view cache must refresh.
//!
//! Resources: `ui://apps-fixture/weather` (the view: a real, minimal bridge client in vanilla JS)
//! and `fixture://notes` (an ordinary resource).
//!
//! Env:
//! - `MCP_APPS_FIXTURE_LOG` — append one JSON line per request: `{"method", "ui"}`, where `ui` is
//!   whether that request's client had advertised the extension.
//! - `MCP_APPS_FIXTURE_LEGACY=1` — refuse `server/discover`, forcing the legacy `initialize` path.
//! - `MCP_APPS_FIXTURE_FAIL_UI=1` — every process that sees the extension advertised exits (an
//!   apps dial that always fails).
//! - `MCP_APPS_FIXTURE_FAIL_FIRST_UI=<path>` — the first process to see the extension advertised
//!   creates `<path>` and exits, so a host's first apps dial fails and its retry succeeds.
//! - `MCP_APPS_FIXTURE_UI_DELAY_MS` — answer `tools/list` that late when the extension was
//!   advertised (a slow apps dial).
//! - `MCP_APPS_FIXTURE_VIEW_DELAY_MS` — answer `resources/read` of the view that late.

use std::io::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

const EXTENSION_ID: &str = "io.modelcontextprotocol/ui";
const MIME: &str = "text/html;profile=mcp-app";
const VIEW_URI: &str = "ui://apps-fixture/weather";

/// The view: speaks the SEP-1865 bridge by hand (no SDK), so a real browser can prove it initializes
/// against the frames the agent produced. Each step writes its outcome into `#status`/`#log`.
const VIEW_HTML: &str = r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>apps-fixture weather</title></head>
<body>
<div id="status">loading</div>
<pre id="log"></pre>
<script>
let nextId = 1;
const pending = {};
const log = (s) => { document.getElementById("log").textContent += s + "\n"; };
const status = (s) => { document.getElementById("status").textContent = s; };
const send = (m) => window.parent.postMessage(m, "*");
function request(method, params) {
  const id = nextId++;
  send({ jsonrpc: "2.0", id, method, params });
  return new Promise((resolve, reject) => { pending[id] = { resolve, reject }; });
}
window.addEventListener("message", (event) => {
  const m = event.data;
  if (!m || m.jsonrpc !== "2.0") return;
  if (m.id !== undefined && pending[m.id] && ("result" in m || "error" in m)) {
    const p = pending[m.id];
    delete pending[m.id];
    if (m.error) p.reject(m.error); else p.resolve(m.result);
    return;
  }
  if (m.method === "ui/notifications/tool-input") {
    log("input:" + JSON.stringify(m.params.arguments));
  } else if (m.method === "ui/notifications/tool-result") {
    const sc = m.params.structuredContent || {};
    log("structured:" + sc.secret);
    request("tools/call", { name: "refresh_weather", arguments: { city: sc.city } })
      .then((r) => { log("refresh:" + r.content[0].text); status("done"); })
      .catch((e) => status("error:" + JSON.stringify(e)));
  }
});
request("ui/initialize", {
  appInfo: { name: "apps-fixture", version: "0.0.0" },
  appCapabilities: {},
  protocolVersion: "2026-01-26",
}).then((r) => {
  log("host:" + (r.hostInfo && r.hostInfo.name));
  status("initialized");
  send({ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {} });
}).catch((e) => status("init-error:" + JSON.stringify(e)));
</script>
</body></html>
"#;

/// Whether the client advertised the extension on `initialize` (legacy lifecycle).
static INIT_UI: AtomicBool = AtomicBool::new(false);

fn advertises_ui(capabilities: Option<&Value>) -> bool {
    capabilities
        .and_then(|c| c.pointer(&format!("/extensions/{}", EXTENSION_ID.replace('/', "~1"))))
        .and_then(|ui| ui.get("mimeTypes"))
        .and_then(Value::as_array)
        .is_some_and(|types| types.iter().any(|t| t.as_str() == Some(MIME)))
}

/// Per request under `2026-07-28` (capabilities ride `_meta`), else what `initialize` said.
fn request_has_ui(params: &Value) -> bool {
    match params.pointer("/_meta/io.modelcontextprotocol~1clientCapabilities") {
        Some(caps) => advertises_ui(Some(caps)),
        None => INIT_UI.load(Ordering::Relaxed),
    }
}

/// The view's version, bumped by `change_view`; the served HTML names it.
static VIEW_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn view_html() -> String {
    VIEW_HTML.replace(
        "apps-fixture weather",
        &format!(
            "apps-fixture weather v{}",
            VIEW_VERSION.load(Ordering::Relaxed)
        ),
    )
}

fn log_request(method: &str, ui: bool, uri: Option<&str>) {
    let Ok(path) = std::env::var("MCP_APPS_FIXTURE_LOG") else {
        return;
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{}", json!({ "method": method, "ui": ui, "uri": uri }));
    }
}

fn env_ms(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn capabilities() -> Value {
    json!({ "tools": {}, "resources": { "listChanged": true } })
}

fn ui_meta(resource_uri: Option<&str>, visibility: Option<&[&str]>) -> Value {
    let mut ui = serde_json::Map::new();
    if let Some(uri) = resource_uri {
        ui.insert("resourceUri".into(), json!(uri));
    }
    if let Some(v) = visibility {
        ui.insert("visibility".into(), json!(v));
    }
    json!({ "ui": ui })
}

fn city_schema() -> Value {
    json!({ "type": "object", "properties": { "city": { "type": "string" } } })
}

fn tools(ui: bool) -> Value {
    let mut tools = vec![
        json!({ "name": "show_weather", "description": "Show the weather for a city.",
                "inputSchema": city_schema() }),
        json!({ "name": "model_only_tool", "description": "Only the model may call this.",
                "inputSchema": { "type": "object", "properties": {} } }),
        json!({ "name": "slow_view", "description": "A slow tool with a view.",
                "inputSchema": { "type": "object", "properties": { "ms": { "type": "integer" } } } }),
        json!({ "name": "broken_view", "description": "A tool whose view is missing.",
                "inputSchema": { "type": "object", "properties": {} } }),
        json!({ "name": "plain_echo", "description": "Echo `text`.",
                "inputSchema": { "type": "object", "properties": { "text": { "type": "string" } } } }),
        json!({ "name": "ask_view", "description": "Ask the client a question mid-call.",
                "inputSchema": { "type": "object", "properties": {} } }),
        json!({ "name": "change_view", "description": "Publish a new version of the view.",
                "inputSchema": { "type": "object", "properties": {} } }),
        json!({ "name": "app_only_always", "description": "App-only, listed unconditionally.",
                "inputSchema": { "type": "object", "properties": {} },
                "_meta": ui_meta(None, Some(&["app"])) }),
    ];
    if ui {
        tools.push(
            json!({ "name": "huge_view", "description": "A tool with an oversized view.",
            "inputSchema": { "type": "object", "properties": {} },
            "_meta": ui_meta(Some("ui://apps-fixture/huge"), None) }),
        );
        tools.push(
            json!({ "name": "huge_unlisted_view", "description": "An unlisted oversized view.",
            "inputSchema": { "type": "object", "properties": {} },
            "_meta": ui_meta(Some("ui://apps-fixture/huge-unlisted"), None) }),
        );
        tools.push(
            json!({ "name": "bad_uri_view", "description": "A view URI that is not ui://.",
            "inputSchema": { "type": "object", "properties": {} },
            "_meta": ui_meta(Some("https://evil.example/view.html"), None) }),
        );
        tools[0]["_meta"] = ui_meta(Some(VIEW_URI), None);
        tools[1]["_meta"] = ui_meta(None, Some(&["model"]));
        tools[2]["_meta"] = ui_meta(Some(VIEW_URI), None);
        tools[3]["_meta"] = ui_meta(Some("ui://apps-fixture/missing"), None);
        tools.push(json!({
            "name": "refresh_weather",
            "description": "Refresh the weather view (app-only).",
            "inputSchema": city_schema(),
            "_meta": ui_meta(Some(VIEW_URI), Some(&["app"])),
        }));
    }
    json!({ "tools": tools })
}

/// Server→client requests in flight: id → where its response goes.
type Pending =
    Arc<std::sync::Mutex<std::collections::HashMap<String, tokio::sync::oneshot::Sender<Value>>>>;

fn pending() -> &'static Pending {
    static P: std::sync::OnceLock<Pending> = std::sync::OnceLock::new();
    P.get_or_init(Pending::default)
}

fn out() -> &'static std::sync::OnceLock<Arc<Mutex<tokio::io::Stdout>>> {
    static O: std::sync::OnceLock<Arc<Mutex<tokio::io::Stdout>>> = std::sync::OnceLock::new();
    &O
}

/// Send the client a nested `elicitation/create` and wait for its result.
async fn ask_client() -> Value {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = format!("apps-elicit-{}", NEXT.fetch_add(1, Ordering::Relaxed));
    let (tx, rx) = tokio::sync::oneshot::channel();
    pending()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(id.clone(), tx);
    let req = json!({
        "jsonrpc": "2.0", "id": id, "method": "elicitation/create",
        "params": { "mode": "form", "message": "The view asks: your name?",
            "requestedSchema": { "type": "object", "properties": { "name": { "type": "string" } } } },
    });
    if let Some(stdout) = out().get() {
        let mut o = stdout.lock().await;
        let _ = o.write_all(format!("{req}\n").as_bytes()).await;
        let _ = o.flush().await;
    }
    rx.await.unwrap_or(Value::Null)
}

async fn call_tool(params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));
    let city = args
        .get("city")
        .and_then(Value::as_str)
        .unwrap_or("nowhere");
    match name {
        "show_weather" => json!({
            "content": [{ "type": "text", "text": format!("weather-text:{city}") }],
            "structuredContent": { "city": city, "secret": format!("STRUCTURED-ONLY-{city}") },
            "_meta": { "fixture": "result-meta-only" },
        }),
        "refresh_weather" => json!({
            "content": [{ "type": "text", "text": format!("refreshed:{city}") }],
            "structuredContent": { "city": city, "refreshed": true },
        }),
        "model_only_tool" => json!({ "content": [{ "type": "text", "text": "model-only" }] }),
        "slow_view" => {
            let ms = args.get("ms").and_then(Value::as_u64).unwrap_or(0);
            tokio::time::sleep(Duration::from_millis(ms)).await;
            json!({ "content": [{ "type": "text", "text": "slow-done" }] })
        }
        "broken_view" => json!({ "content": [{ "type": "text", "text": "broken-view-text" }] }),
        "huge_view" => json!({ "content": [{ "type": "text", "text": "huge-view-text" }] }),
        "bad_uri_view" => json!({ "content": [{ "type": "text", "text": "bad-uri-text" }] }),
        "huge_unlisted_view" => {
            json!({ "content": [{ "type": "text", "text": "huge-unlisted-text" }] })
        }
        "app_only_always" => json!({ "content": [{ "type": "text", "text": "app-only" }] }),
        "ask_view" => {
            let reply = ask_client().await;
            let answer = reply
                .pointer("/result/content/name")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| {
                    reply
                        .pointer("/result/action")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .or_else(|| {
                    reply
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .map(|m| format!("error {m}"))
                })
                .unwrap_or_else(|| "nothing".into());
            json!({ "content": [{ "type": "text", "text": format!("asked:{answer}") }] })
        }
        "change_view" => {
            let v = VIEW_VERSION.fetch_add(1, Ordering::Relaxed) + 1;
            json!({ "content": [{ "type": "text", "text": format!("view-v{v}") }] })
        }
        "plain_echo" => json!({
            "content": [{ "type": "text",
                          "text": args.get("text").and_then(Value::as_str).unwrap_or("") }],
        }),
        other => json!({
            "content": [{ "type": "text", "text": format!("unknown tool {other}") }],
            "isError": true,
        }),
    }
}

/// `Ok(result)` or `Err((code, message))`.
async fn handle(method: &str, params: &Value) -> Result<Value, (i64, String)> {
    let ui = request_has_ui(params);
    match method {
        "server/discover" if std::env::var("MCP_APPS_FIXTURE_LEGACY").is_ok() => {
            Err((-32601, "Method not found: server/discover".into()))
        }
        "server/discover" => Ok(json!({
            "resultType": "complete",
            "supportedVersions": ["2026-07-28", "2025-11-25"],
            "capabilities": capabilities(),
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": { "io.modelcontextprotocol/serverInfo":
                       { "name": "mcp-apps-fixture", "version": "0.0.0" } },
        })),
        "initialize" => {
            INIT_UI.store(advertises_ui(params.get("capabilities")), Ordering::Relaxed);
            Ok(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": capabilities(),
                "serverInfo": { "name": "mcp-apps-fixture", "version": "0.0.0" },
            }))
        }
        "tools/list" => {
            if ui && let Some(ms) = env_ms("MCP_APPS_FIXTURE_UI_DELAY_MS") {
                tokio::time::sleep(Duration::from_millis(ms)).await;
            }
            Ok(tools(ui))
        }
        "tools/call" => Ok(call_tool(params).await),
        "resources/list" => Ok(json!({ "resources": [
            { "uri": VIEW_URI, "name": "weather_view", "mimeType": MIME },
            { "uri": "fixture://notes", "name": "notes", "mimeType": "text/plain" },
            { "uri": "ui://apps-fixture/huge", "name": "huge_view", "mimeType": MIME,
              "size": 5 * 1024 * 1024 + 64 },
        ] })),
        "resources/read" => match params.get("uri").and_then(Value::as_str) {
            Some(uri @ ("ui://apps-fixture/huge" | "ui://apps-fixture/huge-unlisted")) => {
                Ok(json!({ "contents": [{
                "uri": uri,
                "mimeType": MIME,
                "text": format!("<!DOCTYPE html><html><body>{}</body></html>", "x".repeat(5 * 1024 * 1024)),
            }] }))
            }
            Some(VIEW_URI) => {
                if let Some(ms) = env_ms("MCP_APPS_FIXTURE_VIEW_DELAY_MS") {
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                }
                Ok(json!({ "contents": [{
                "uri": VIEW_URI,
                "mimeType": MIME,
                "text": view_html(),
                "_meta": { "ui": {
                    "csp": { "connectDomains": ["https://api.example.test"] },
                    "prefersBorder": true,
                } },
            }] }))
            }
            Some("fixture://notes") => Ok(json!({ "contents": [{
                "uri": "fixture://notes", "mimeType": "text/plain", "text": "notes-body",
            }] })),
            other => Err((-32002, format!("Resource not found: {other:?}"))),
        },
        "ping" => Ok(json!({})),
        other => Err((-32601, format!("Method not found: {other}"))),
    }
}

#[tokio::main]
async fn main() {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    let _ = out().set(stdout.clone());
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(method) = request.get("method").and_then(Value::as_str) else {
            // A response to one of our own server→client requests.
            if let Some(id) = request.get("id").and_then(Value::as_str)
                && let Some(tx) = pending()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(id)
            {
                let _ = tx.send(request);
            }
            continue;
        };
        let params = request.get("params").cloned().unwrap_or(Value::Null);
        let uri = params.get("uri").and_then(Value::as_str);
        let ui = if method == "initialize" {
            advertises_ui(params.get("capabilities"))
        } else {
            request_has_ui(&params)
        };
        log_request(method, ui, uri);
        if ui && std::env::var("MCP_APPS_FIXTURE_FAIL_UI").is_ok() {
            std::process::exit(3);
        }
        if ui
            && let Ok(marker) = std::env::var("MCP_APPS_FIXTURE_FAIL_FIRST_UI")
            && !std::path::Path::new(&marker).exists()
        {
            write_atomically(&marker, "failed once");
            std::process::exit(3);
        }
        let changes_view = method == "tools/call"
            && params.get("name").and_then(Value::as_str) == Some("change_view");
        // Notifications get no answer.
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let method = method.to_owned();
        let stdout = stdout.clone();
        // Concurrent, so a slow tool never blocks a `ping` or a cancel behind it.
        tokio::spawn(async move {
            let envelope = match handle(&method, &params).await {
                Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
                Err((code, message)) => json!({ "jsonrpc": "2.0", "id": id,
                                                "error": { "code": code, "message": message } }),
            };
            let mut out = stdout.lock().await;
            if changes_view {
                let note =
                    json!({ "jsonrpc": "2.0", "method": "notifications/resources/list_changed" });
                let _ = out.write_all(format!("{note}\n").as_bytes()).await;
            }
            let _ = out.write_all(format!("{envelope}\n").as_bytes()).await;
            let _ = out.flush().await;
        });
    }
}

/// Write a file a test reads, all at once: to a temporary sibling, then `rename` it into place. A
/// reader polling for the file (or its content) can otherwise see it created but still empty,
/// between `write`'s create and its write.
fn write_atomically(path: &str, contents: &str) {
    let tmp = format!("{path}.tmp-{}", std::process::id());
    if std::fs::write(&tmp, contents).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}
