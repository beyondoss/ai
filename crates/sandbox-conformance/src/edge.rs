//! The edge: a proxy the harness puts between a driver and a backend, to show how a client behaves
//! when the path to the sandbox misbehaves.
//!
//! A real deployment has an edge in front of the sandbox: it issues short-lived bearers, it can be slow
//! to answer while a sandbox wakes, it can fail. The backends themselves are well behaved, so a scenario
//! with `connect.edge` gets its own `Edge` in front of the backend's endpoint:
//!
//! - `tokens`: the edge issues the bearer. The driver is given `CONFORMANCE_TOKEN_CMD` (a shell command
//!   that prints the current token) instead of a fixed bearer, and the edge answers `401` to any other
//!   token. `true` mints well, `"fail"` is a command that exits 1, `"empty"` one that prints nothing.
//!   Without `tokens` the driver sends no bearer and the edge accepts anything.
//! - what the edge does to a request that passed: wait `delay_ms` before it goes upstream (a sandbox
//!   waking up), or one `fault`, applied to every request until changed.
//!
//! A step with an `edge` key (instead of `op`) drives the edge between driver operations and answers with
//! the counters, so a scenario can say how many requests a client made. The edge always talks to the
//! backend with the backend's own bearer, one connection per request, `Connection: close`.
//!
//! Bounds: a request's head is at most [`MAX_HEAD`], its body [`MAX_BODY`]; the edge holds nothing else
//! (the response is streamed). It stops, with every connection it serves, when dropped.
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::{JoinHandle, JoinSet};

const MAX_HEAD: usize = 64 * 1024;
/// The largest request body the edge buffers (a bulk upload chunk is far below it).
const MAX_BODY: usize = 64 * 1024 * 1024;

pub struct Edge {
    /// The URL a driver is given in place of the backend's.
    pub url: String,
    /// `CONFORMANCE_TOKEN_CMD`, when the edge issues tokens.
    pub token_cmd: Option<String>,
    state: Arc<Mutex<State>>,
    dir: tempfile::TempDir,
    task: JoinHandle<()>,
}

impl Drop for Edge {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Default)]
struct State {
    upstream: Upstream,
    /// The token the edge accepts; `None` accepts any request.
    current: Option<String>,
    issued: u32,
    delay: Duration,
    fault: Option<Fault>,
    count: Counters,
}

#[derive(Default, Clone)]
struct Upstream {
    authority: String,
    bearer: Option<String>,
}

#[derive(Default, Clone, Copy)]
struct Counters {
    requests: u64,
    rejected: u64,
    faulted: u64,
    forwarded: u64,
}

#[derive(Clone)]
enum Fault {
    /// Answer with this status and body.
    Status(u16, String),
    /// Answer `200` with a body that is not the protocol.
    Garbage { ndjson: bool, body: String },
    /// Close the connection without an answer.
    Drop,
    /// Take the request and never answer.
    Hang,
    /// Pass the answer's head and this many bytes of its body, then close.
    Cut(usize),
}

impl Edge {
    /// Start an edge in front of `upstream` (an `http://` URL). `config` is the scenario's `connect.edge`.
    pub async fn start(
        upstream: &str,
        upstream_bearer: Option<String>,
        config: &Value,
    ) -> Result<Edge, String> {
        let rest = upstream
            .strip_prefix("http://")
            .ok_or("the edge fronts plain http endpoints only")?;
        let (authority, path) = rest.split_once('/').map_or((rest, ""), |(a, p)| (a, p));
        let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        // port-0: the edge holds its listener for the run and reads the port it got from it.
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| e.to_string())?;
        let addr = listener.local_addr().map_err(|e| e.to_string())?;

        let mut state = State {
            upstream: Upstream {
                authority: authority.to_string(),
                bearer: upstream_bearer,
            },
            ..State::default()
        };
        let mut token_cmd = None;
        if let Some(mode) = config.get("tokens").filter(|t| !t.is_null()) {
            state.issued = 1;
            let token = token_text(1);
            std::fs::write(dir.path().join("token"), format!("{token}\n"))
                .map_err(|e| e.to_string())?;
            state.current = Some(token);
            let d = dir.path().display();
            let body = match mode.as_str() {
                Some("fail") => "exit 1",
                Some("empty") => "true",
                _ => "cat",
            };
            let body = if body == "cat" {
                format!("cat '{d}/token'")
            } else {
                body.to_string()
            };
            // One line per mint, so a scenario can count them.
            token_cmd = Some(format!("echo >> '{d}/mints'; {body}"));
        }
        let state = Arc::new(Mutex::new(state));
        let task = tokio::spawn(serve(listener, state.clone()));
        Ok(Edge {
            url: format!("http://{addr}/{path}"),
            token_cmd,
            state,
            dir,
            task,
        })
    }

    fn mints(&self) -> u64 {
        std::fs::read_to_string(self.dir.path().join("mints")).map_or(0, |s| s.lines().count())
            as u64
    }

    fn mints_path(&self) -> PathBuf {
        self.dir.path().join("mints")
    }

    /// One `edge` step: `expire`, `set`, `reset`. The answer is the counters, from before a `reset`.
    pub fn control(&self, ctl: &Value) -> Value {
        let mut st = self.state.lock().expect("edge state");
        if let Some(e) = ctl.get("expire").filter(|e| !e.is_null()) {
            // The token every driver holds stops working. By default the token command starts printing a
            // new one; `"mint": false` leaves it printing the dead one.
            st.issued += 1;
            let token = token_text(st.issued);
            if st.current.is_some() {
                st.current = Some(token.clone());
                if e.get("mint") != Some(&json!(false)) {
                    let _ = std::fs::write(self.dir.path().join("token"), format!("{token}\n"));
                }
            }
        }
        if let Some(set) = ctl.get("set") {
            if let Some(d) = set.get("delay_ms") {
                st.delay = Duration::from_millis(d.as_u64().unwrap_or(0));
            }
            if let Some(f) = set.get("fault") {
                st.fault = parse_fault(f);
            }
        }
        let c = st.count;
        let answer = json!({
            "requests": c.requests,
            "rejected": c.rejected,
            "faulted": c.faulted,
            "forwarded": c.forwarded,
            "mints": self.mints(),
        });
        if ctl.get("reset") == Some(&json!(true)) {
            st.count = Counters::default();
            let _ = std::fs::remove_file(self.mints_path());
        }
        answer
    }
}

fn token_text(n: u32) -> String {
    format!("edge-token-{n}")
}

fn parse_fault(f: &Value) -> Option<Fault> {
    Some(match f.get("kind")?.as_str()? {
        "status" => Fault::Status(
            f["status"].as_u64().unwrap_or(500) as u16,
            f["body"].as_str().unwrap_or("").to_string(),
        ),
        "garbage" => Fault::Garbage {
            ndjson: f["ndjson"].as_bool().unwrap_or(false),
            body: f["body"].as_str().unwrap_or("this is not json").to_string(),
        },
        "drop" => Fault::Drop,
        "hang" => Fault::Hang,
        "cut" => Fault::Cut(f["body_bytes"].as_u64().unwrap_or(0) as usize),
        _ => return None,
    })
}

async fn serve(listener: TcpListener, state: Arc<Mutex<State>>) {
    let mut conns = JoinSet::new();
    loop {
        let Ok((sock, _)) = listener.accept().await else {
            return;
        };
        conns.spawn(handle(sock, state.clone()));
        // Reap finished connections so the set is bounded by the live ones.
        while conns.try_join_next().is_some() {}
    }
}

struct Request {
    head: String,
    body: Vec<u8>,
    /// What bytes followed the request on the socket (a pipelined request; clients here never send one).
    rest: Vec<u8>,
}

async fn read_request(sock: &mut TcpStream, mut buf: Vec<u8>) -> Option<Request> {
    let end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() > MAX_HEAD {
            return None;
        }
        let mut chunk = [0u8; 8192];
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let len: usize = header(&head, "content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if len > MAX_BODY || header(&head, "transfer-encoding").is_some() {
        return None;
    }
    let mut body = buf.split_off(end);
    while body.len() < len {
        let mut chunk = vec![0u8; (len - body.len()).min(1 << 20)];
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    let rest = body.split_off(len);
    Some(Request { head, body, rest })
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn header(head: &str, name: &str) -> Option<String> {
    head.lines().skip(1).find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim()
            .eq_ignore_ascii_case(name)
            .then(|| v.trim().to_string())
    })
}

async fn respond(sock: &mut TcpStream, status: u16, ctype: &str, body: &str) {
    let reason = match status {
        200 => "OK",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Status",
    };
    let msg = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    let _ = sock.write_all(msg.as_bytes()).await;
}

async fn handle(mut sock: TcpStream, state: Arc<Mutex<State>>) {
    let mut carry = Vec::new();
    while let Some(req) = read_request(&mut sock, std::mem::take(&mut carry)).await {
        carry = req.rest.clone();
        let (up, delay, fault, current) = {
            let mut st = state.lock().expect("edge state");
            st.count.requests += 1;
            (
                st.upstream.clone(),
                st.delay,
                st.fault.clone(),
                st.current.clone(),
            )
        };
        if let Some(want) = current {
            let ok =
                header(&req.head, "authorization").is_some_and(|a| a == format!("Bearer {want}"));
            if !ok {
                state.lock().expect("edge state").count.rejected += 1;
                respond(
                    &mut sock,
                    401,
                    "application/json",
                    r#"{"error":"token expired"}"#,
                )
                .await;
                continue;
            }
        }
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut cut = None;
        match fault {
            Some(Fault::Status(status, body)) => {
                state.lock().expect("edge state").count.faulted += 1;
                respond(&mut sock, status, "application/json", &body).await;
                continue;
            }
            Some(Fault::Garbage { ndjson, body }) => {
                state.lock().expect("edge state").count.faulted += 1;
                let ctype = if ndjson {
                    "application/x-ndjson"
                } else {
                    "application/json"
                };
                respond(&mut sock, 200, ctype, &format!("{body}\n")).await;
                continue;
            }
            Some(Fault::Drop) => {
                state.lock().expect("edge state").count.faulted += 1;
                return;
            }
            Some(Fault::Hang) => {
                state.lock().expect("edge state").count.faulted += 1;
                std::future::pending::<()>().await;
            }
            Some(Fault::Cut(n)) => {
                state.lock().expect("edge state").count.faulted += 1;
                cut = Some(n);
            }
            None => state.lock().expect("edge state").count.forwarded += 1,
        }
        forward(&mut sock, &up, &req, cut).await;
        // The upstream connection was `close`: so is this one.
        return;
    }
}

/// Send `req` upstream with the backend's bearer and stream the answer back until the upstream closes
/// or the client goes away. `cut`: forward only that many bytes of the body, then stop.
async fn forward(client: &mut TcpStream, up: &Upstream, req: &Request, cut: Option<usize>) {
    let Ok(mut upstream) = TcpStream::connect(&up.authority).await else {
        respond(client, 502, "text/plain", "edge: upstream unreachable").await;
        return;
    };
    let mut lines = req.head.lines();
    let request_line = lines.next().unwrap_or("");
    let mut out = format!(
        "{request_line}\r\nhost: {}\r\nconnection: close\r\n",
        up.authority
    );
    for l in lines {
        let name = l
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if matches!(name.as_str(), "" | "host" | "connection" | "authorization") {
            continue;
        }
        out.push_str(l);
        out.push_str("\r\n");
    }
    if let Some(b) = &up.bearer {
        out.push_str(&format!("authorization: Bearer {b}\r\n"));
    }
    out.push_str("\r\n");
    if upstream.write_all(out.as_bytes()).await.is_err()
        || upstream.write_all(&req.body).await.is_err()
    {
        return;
    }
    let mut head_seen = false;
    let mut window: Vec<u8> = Vec::new();
    let mut body_sent = 0usize;
    let mut chunk = vec![0u8; 16 * 1024];
    let mut gone = [0u8; 1];
    loop {
        tokio::select! {
            n = upstream.read(&mut chunk) => {
                let Ok(n) = n else { return };
                if n == 0 { break }
                let mut data = &chunk[..n];
                if let Some(limit) = cut {
                    // Count body bytes once the head is complete.
                    if !head_seen {
                        window.extend_from_slice(data);
                        let Some(end) = find(&window, b"\r\n\r\n") else {
                            if client.write_all(data).await.is_err() { return }
                            continue;
                        };
                        head_seen = true;
                        let already = window.len() - (end + 4);
                        // `data` is the tail of `window`; send what the head completes with.
                        let head_part = n - already;
                        if client.write_all(&data[..head_part]).await.is_err() { return }
                        data = &data[head_part..];
                    }
                    let room = limit.saturating_sub(body_sent);
                    let take = data.len().min(room);
                    if client.write_all(&data[..take]).await.is_err() { return }
                    body_sent += take;
                    if body_sent >= limit {
                        let _ = client.flush().await;
                        return;
                    }
                } else if client.write_all(data).await.is_err() {
                    return;
                }
            }
            // The client left (a cancelled stream): drop the upstream, which kills the command.
            n = client.read(&mut gone) => {
                if matches!(n, Ok(0) | Err(_)) { return }
            }
        }
    }
    let _ = client.shutdown().await;
}
