//! `web` — a "curl for agents": fetch, discover, extract, with deny-by-default egress safety.
//!
//! Modeled on [ax](https://github.com/yusukebe/ax) ("the AI-era curl"), reimplemented as a native
//! [`Tool`] so the agent can read the web without shelling out or relying on an MCP server. Ax is a
//! local CLI a human runs; this is a server fetching URLs the *model* chose, so it adds the one thing ax
//! doesn't need — an egress-safety layer (see [`ssrf`]) that is deny-by-default and, because the tool is
//! read-only and never approval-gated, is the *sole* boundary between the model and an internal fetch.
//!
//! One tool, dispatched by an explicit `mode` (clearer for the model than ax's flag-inference):
//! `fetch` (the report), `markdown` (readable docs), and the extraction modes `outline`/`locate`/
//! `extract`/`table`. The tool owns its own [`reqwest::Client`] — the gateway's shared client is
//! h2-pinned with a 10-minute timeout, the wrong shape for arbitrary web fetches.

mod extract;
mod markdown;
pub mod ssrf;
mod whereexpr;

use std::time::{Duration, Instant};

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use ssrf::{Blocked, EgressPolicy, SsrfResolver};

pub const NAME: &str = "web";

/// ax's own limits — a fetch that hasn't finished by then is almost never worth continuing, and a body
/// past 20 MB is not something to buffer into a model's context.
const MAX_BODY_BYTES: usize = 20 * 1024 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Redirect hops we follow before giving up. reqwest's default is 10 with no per-hop revalidation; we
/// follow manually so [`ssrf::validate_url`] runs on every hop.
const MAX_REDIRECTS: usize = 5;

const USER_AGENT: &str = concat!(
    "beyond-ai-agent/",
    env!("CARGO_PKG_VERSION"),
    " (+web tool)"
);

pub struct Web {
    policy: EgressPolicy,
    timeout: Duration,
    /// Built lazily on first fetch, not in `new`. Constructing a reqwest client initializes the rustls
    /// crypto provider (~30 ms the first time), and `serve` rebuilds the whole registry on every
    /// `set_model`/`set_thinking` — so an eager build would pay that cost on every rebuild and slow
    /// startup, for a tool a given session may never call. `OnceLock` keeps `new` trivial and builds the
    /// client exactly once, on demand.
    client: std::sync::OnceLock<reqwest::Client>,
}

impl Web {
    /// Configure the tool. `allow_private`/`allow_hosts` come from `--web-allow-private`/
    /// `--web-allow-host`; `timeout_ms` from `--web-timeout-ms` (else 30 s). The client is built lazily.
    pub fn new(allow_private: bool, allow_hosts: &[String], timeout_ms: Option<u64>) -> Self {
        Self {
            policy: EgressPolicy::new(allow_private, allow_hosts),
            timeout: timeout_ms
                .map(Duration::from_millis)
                .unwrap_or(DEFAULT_TIMEOUT),
            client: std::sync::OnceLock::new(),
        }
    }

    /// The owned client, built on first use. `redirect::Policy::none()` — we follow redirects ourselves
    /// so the SSRF URL check runs on each hop — plus the [`SsrfResolver`], so every connection (initial
    /// and any hop) validates the *resolved* IP.
    fn client(&self) -> &reqwest::Client {
        self.client.get_or_init(|| {
            reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .connect_timeout(Duration::from_secs(10))
                .timeout(self.timeout)
                .redirect(reqwest::redirect::Policy::none())
                .dns_resolver(SsrfResolver::new(self.policy.clone()))
                .build()
                // The builder only fails if the TLS backend can't initialize — fatal and identical for
                // every call, so a fallback default client that will error consistently on use beats
                // panicking.
                .unwrap_or_else(|_| reqwest::Client::new())
        })
    }
}

/// One request/response round trip's outcome, before mode-specific rendering.
struct Fetched {
    final_url: reqwest::Url,
    status: reqwest::StatusCode,
    headers: reqwest::header::HeaderMap,
    body: Vec<u8>,
    elapsed: Duration,
    truncated: bool,
}

impl Fetched {
    fn text(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.body)
    }
}

impl Web {
    /// Perform the request, following redirects manually with a per-hop SSRF re-check.
    async fn fetch(&self, req: &Request) -> Result<Fetched, ToolError> {
        let mut url: reqwest::Url = req
            .url
            .parse()
            .map_err(|e| ToolError::InvalidInput(format!("invalid url `{}`: {e}", req.url)))?;
        ssrf::validate_url(&url, &self.policy).map_err(blocked_to_err)?;

        let start = Instant::now();
        let mut redirects = 0;
        loop {
            let mut rb = self.client().request(req.method.clone(), url.clone());
            for (k, v) in &req.headers {
                rb = rb.header(k, v);
            }
            if let Some(body) = &req.body {
                rb = rb.body(body.clone());
            }
            let resp = rb.send().await.map_err(fetch_err)?;

            if resp.status().is_redirection() {
                let Some(location) = resp.headers().get(reqwest::header::LOCATION) else {
                    // A 3xx with no Location — treat it as the terminal response rather than loop.
                    return finish(resp, url, start).await;
                };
                redirects += 1;
                if redirects > MAX_REDIRECTS {
                    return Err(ToolError::Execution(format!(
                        "too many redirects (>{MAX_REDIRECTS}) starting from `{}`",
                        req.url
                    )));
                }
                let location = location.to_str().map_err(|_| {
                    ToolError::Execution("redirect Location is not valid text".into())
                })?;
                // Resolve relative to the current URL, then re-validate scheme + host before the next hop.
                url = url.join(location).map_err(|e| {
                    ToolError::Execution(format!("bad redirect target `{location}`: {e}"))
                })?;
                ssrf::validate_url(&url, &self.policy).map_err(blocked_to_err)?;
                continue;
            }
            return finish(resp, url, start).await;
        }
    }
}

/// Read a terminal (non-redirect) response body, capped at [`MAX_BODY_BYTES`].
async fn finish(
    resp: reqwest::Response,
    final_url: reqwest::Url,
    start: Instant,
) -> Result<Fetched, ToolError> {
    let status = resp.status();
    let headers = resp.headers().clone();
    let mut body = Vec::new();
    let mut truncated = false;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(fetch_err)?;
        if body.len() + chunk.len() > MAX_BODY_BYTES {
            let take = MAX_BODY_BYTES - body.len();
            body.extend_from_slice(&chunk[..take]);
            truncated = true;
            break;
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Fetched {
        final_url,
        status,
        headers,
        body,
        elapsed: start.elapsed(),
        truncated,
    })
}

fn blocked_to_err(b: Blocked) -> ToolError {
    // The model chose the URL, so a refusal is invalid *input*, not an execution failure.
    ToolError::InvalidInput(b.to_string())
}

/// The sentinel [`Blocked::Address`]'s message carries, used to recognize our own SSRF refusal when it
/// surfaces buried in reqwest's error source chain (a resolver rejection reaches us as a generic
/// "could not connect", with the real reason as a `source()`).
const SSRF_SENTINEL: &str = "blocked to prevent access to internal services";

fn fetch_err(e: reqwest::Error) -> ToolError {
    // Walk the source chain: if this failure is our resolver refusing a blocked address, report *that*
    // (as InvalidInput — the model chose the URL) rather than an opaque connect error.
    let mut src: Option<&(dyn std::error::Error + 'static)> = Some(&e);
    while let Some(s) = src {
        let msg = s.to_string();
        if msg.contains(SSRF_SENTINEL) {
            return ToolError::InvalidInput(msg);
        }
        src = s.source();
    }
    if e.is_timeout() {
        ToolError::Execution(format!("request timed out: {e}"))
    } else if e.is_connect() {
        ToolError::Execution(format!("could not connect: {e}"))
    } else {
        ToolError::Execution(format!("request failed: {e}"))
    }
}

/// The parsed, validated tool input.
struct Request {
    url: String,
    method: reqwest::Method,
    headers: Vec<(String, String)>,
    body: Option<String>,
    mode: Mode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fetch,
    Markdown,
    Outline,
    Locate,
    Extract,
    Table,
}

impl Mode {
    fn parse(s: &str) -> Result<Self, ToolError> {
        Ok(match s {
            "fetch" => Self::Fetch,
            "markdown" => Self::Markdown,
            "outline" => Self::Outline,
            "locate" => Self::Locate,
            "extract" => Self::Extract,
            "table" => Self::Table,
            other => {
                return Err(ToolError::InvalidInput(format!(
                    "unknown mode `{other}` (expected fetch, markdown, outline, locate, extract, or table)"
                )));
            }
        })
    }
}

fn parse_request(input: &Value) -> Result<Request, ToolError> {
    let url = input
        .get("url")
        .and_then(Value::as_str)
        .filter(|u| !u.is_empty())
        .ok_or_else(|| ToolError::InvalidInput("missing `url`".into()))?
        .to_string();

    let mode = match input.get("mode").and_then(Value::as_str) {
        Some(m) => Mode::parse(m)?,
        None => Mode::Fetch,
    };

    let method = match input.get("method").and_then(Value::as_str) {
        None => reqwest::Method::GET,
        Some(m) => m
            .to_ascii_uppercase()
            .parse()
            .map_err(|_| ToolError::InvalidInput(format!("invalid HTTP method `{m}`")))?,
    };

    let headers = match input.get("headers") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Object(map)) => map
            .iter()
            .map(|(k, v)| {
                let val = v
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string());
                (k.clone(), val)
            })
            .collect(),
        Some(_) => {
            return Err(ToolError::InvalidInput(
                "`headers` must be an object of name→value".into(),
            ));
        }
    };

    let body = input
        .get("body")
        .and_then(Value::as_str)
        .map(str::to_string);

    Ok(Request {
        url,
        method,
        headers,
        body,
        mode,
    })
}

/// Render the `fetch` report — ax's `{status, ok, ms, headers, body}` shape, as readable text.
fn render_fetch(f: &Fetched) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    // `write!` into the buffer directly — no throwaway `format!` allocation per line/header. Writing
    // to a `String` is infallible, so the `Result` is deliberately ignored.
    let _ = writeln!(
        out,
        "{} {} — {}ms",
        f.status.as_u16(),
        f.status.canonical_reason().unwrap_or(""),
        f.elapsed.as_millis()
    );
    let _ = writeln!(out, "url: {}", f.final_url);
    out.push_str("headers:\n");
    for (k, v) in f.headers.iter() {
        let _ = writeln!(out, "  {}: {}", k, v.to_str().unwrap_or("<binary>"));
    }
    out.push('\n');
    let body = f.text();
    // Pretty-print a JSON body so the model reads structure, not a minified blob.
    let is_json = f
        .headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|c| c.to_str().ok())
        .is_some_and(|c| c.contains("json"));
    if is_json
        && let Ok(v) = serde_json::from_str::<Value>(&body)
        && let Ok(pretty) = serde_json::to_string_pretty(&v)
    {
        out.push_str(&pretty);
        return finalize(out, f.truncated);
    }
    out.push_str(&body);
    finalize(out, f.truncated)
}

fn finalize(mut out: String, truncated: bool) -> String {
    if truncated {
        out.push_str(&format!(
            "\n\n[body truncated at {}MB]",
            MAX_BODY_BYTES / (1024 * 1024)
        ));
    }
    super::output::cap_listing_bytes(&mut out, "narrow the request or use a different mode");
    out
}

#[async_trait]
impl Tool for Web {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        "Fetch a URL and return agent-shaped output. Modes: `fetch` (status + headers + body), \
         `markdown` (readable page text), `outline`/`locate`/`extract`/`table` (structured extraction \
         without dumping raw HTML). Only http/https; internal/private addresses are refused."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The http(s) URL to fetch." },
                "mode": {
                    "type": "string",
                    "enum": ["fetch", "markdown", "outline", "locate", "extract", "table"],
                    "description": "fetch: raw status/headers/body. markdown: readable page text. outline: the page's repeating structure. locate: which CSS selector holds `text`. extract: keyed rows from `fields`+`selector` (optionally filtered by `where`). table: HTML tables as rows. Default: fetch."
                },
                "method": { "type": "string", "description": "HTTP method (default GET)." },
                "headers": { "type": "object", "description": "Request headers, name→value." },
                "body": { "type": "string", "description": "Request body (for POST/PUT/PATCH)." },
                "text": { "type": "string", "description": "locate mode: the text to find a selector for." },
                "selector": { "type": "string", "description": "extract/table mode: the CSS selector for each row/container." },
                "fields": { "description": "extract mode: either \"title=a, href=a@href\" (a@attr reads an attribute, else text) or an object {name: selector}." },
                "where": { "type": "string", "description": "extract mode: filter rows, e.g. \"price > 100 && name ~ /^foo/i\"." },
                "max_items": { "type": "integer", "description": "Cap extracted rows (default 50)." }
            },
            "required": ["url"]
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let req = parse_request(&input)?;
        let fetched = self.fetch(&req).await?;

        let text = match req.mode {
            // `fetch` always reports the status itself.
            Mode::Fetch => render_fetch(&fetched),
            // The parsing modes work on the body regardless of status (an error page is still HTML), but
            // a model asking to `extract` from a 500 would otherwise get empty rows with no clue why —
            // so prepend the status when it isn't a success.
            other => {
                let mut out = String::new();
                if !fetched.status.is_success() {
                    out.push_str(&format!(
                        "[HTTP {} {}]\n\n",
                        fetched.status.as_u16(),
                        fetched.status.canonical_reason().unwrap_or("")
                    ));
                }
                let rendered = match other {
                    Mode::Markdown => finalize(
                        markdown::to_markdown(&fetched.text()).map_err(ToolError::Execution)?,
                        fetched.truncated,
                    ),
                    _ => extract::run(other, &input, &fetched.text())?,
                };
                out.push_str(&rendered);
                out
            }
        };
        Ok(text.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A one-shot loopback HTTP/1.1 server that replies with `response` to the first connection, and
    /// records the raw request line + headers it received. Returns `(port, join_handle_for_request)`.
    fn one_shot(response: &'static str) -> (u16, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).into_owned();
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
            req
        });
        (port, handle)
    }

    const HTML_200: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 46\r\nConnection: close\r\n\r\n<html><body><h1>Hi</h1><p>ok</p></body></html>";

    /// The tool with loopback allow-listed, so it can reach a test fixture.
    fn tool_allowing_loopback() -> Web {
        Web::new(false, &["127.0.0.1".to_string()], Some(5000))
    }

    #[tokio::test]
    async fn fetch_returns_a_report_through_the_real_client() {
        let (port, handle) = one_shot(HTML_200);
        let out = tool_allowing_loopback()
            .run(json!({ "url": format!("http://127.0.0.1:{port}/"), "mode": "fetch" }))
            .await
            .unwrap();
        let req = handle.join().unwrap();
        assert!(req.starts_with("GET / HTTP/1.1"), "outbound request: {req}");
        assert!(out.text.contains("200 OK"), "{}", out.text);
        assert!(out.text.contains("<h1>Hi</h1>"), "{}", out.text);
    }

    #[tokio::test]
    async fn markdown_mode_converts_the_body() {
        let (port, _h) = one_shot(HTML_200);
        let out = tool_allowing_loopback()
            .run(json!({ "url": format!("http://127.0.0.1:{port}/"), "mode": "markdown" }))
            .await
            .unwrap();
        assert!(out.text.contains("# Hi"), "{}", out.text);
        assert!(
            !out.text.contains("<h1>"),
            "raw HTML must be gone: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn method_and_headers_and_body_reach_the_server() {
        let (port, handle) = one_shot(HTML_200);
        tool_allowing_loopback()
            .run(json!({
                "url": format!("http://127.0.0.1:{port}/submit"),
                "method": "POST",
                "headers": { "X-Test": "marker-42" },
                "body": "the-body"
            }))
            .await
            .unwrap();
        let req = handle.join().unwrap();
        assert!(req.starts_with("POST /submit"), "{req}");
        assert!(
            req.contains("x-test: marker-42") || req.contains("X-Test: marker-42"),
            "{req}"
        );
        assert!(req.ends_with("the-body"), "{req}");
    }

    #[tokio::test]
    async fn loopback_is_refused_by_default_through_the_resolver() {
        // No allow-host, no allow-private: the SSRF layer must block it. A literal-IP URL is caught by
        // `validate_url` before DNS; use a hostname so it goes through the resolver seam instead.
        let (port, _h) = one_shot(HTML_200);
        let web = Web::new(false, &[], Some(5000));
        let err = web
            .run(json!({ "url": format!("http://localhost:{port}/") }))
            .await
            .unwrap_err();
        let ToolError::InvalidInput(msg) = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(
            msg.contains("blocked") || msg.contains("internal"),
            "must explain the refusal: {msg}"
        );
    }

    #[tokio::test]
    async fn a_literal_internal_ip_is_refused_before_any_connection() {
        let web = Web::new(false, &[], Some(5000));
        let err = web
            .run(json!({ "url": "http://169.254.169.254/latest/meta-data/" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "{err:?}");
    }

    #[tokio::test]
    async fn a_parsing_mode_prepends_the_status_on_a_non_2xx() {
        let not_found = "HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\nContent-Length: 32\r\nConnection: close\r\n\r\n<html><body>gone</body></html>\n\n";
        let (port, _h) = one_shot(not_found);
        let out = tool_allowing_loopback()
            .run(json!({ "url": format!("http://127.0.0.1:{port}/"), "mode": "markdown" }))
            .await
            .unwrap();
        assert!(out.text.contains("[HTTP 404 Not Found]"), "{}", out.text);
    }

    #[tokio::test]
    async fn a_non_http_scheme_is_refused() {
        let web = Web::new(false, &[], Some(5000));
        assert!(matches!(
            web.run(json!({ "url": "file:///etc/passwd" })).await,
            Err(ToolError::InvalidInput(_))
        ));
    }

    #[tokio::test]
    async fn a_redirect_to_a_blocked_host_is_refused_mid_chain() {
        // The fixture 302s to a link-local metadata address; the per-hop re-validation must catch it.
        let redirect = "HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let (port, _h) = one_shot(redirect);
        let err = tool_allowing_loopback()
            .run(json!({ "url": format!("http://127.0.0.1:{port}/") }))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(_)),
            "the redirect hop must be blocked: {err:?}"
        );
    }
}
