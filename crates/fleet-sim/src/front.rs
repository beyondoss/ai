//! The edge's front for exec notify job-end callbacks, as a double (FLEET.md, "Job-end callbacks",
//! E1–E7).
//!
//! A sandbox reports a kept job's end by POSTing `job_ended` to the session's own address, with a
//! `bjn_v1` token only replicas can check. The edge's whole part in that is routing: it classifies
//! the request by its `Authorization` header, routes it by its session id exactly as it routes the
//! session's WebSocket ([`crate::edge`]: the hashed target first, retried on `503`, then the ring
//! walked from the hash), and passes the replica's answer back unchanged. It attaches, mints and
//! checks nothing, never logs the token, answers within 25 s, and forwards at most 20 a second for
//! any one session id.
//!
//! Like `edge.rs`, this implements a contract the repo documents rather than one it imports, so it
//! is also the reference for what a real edge must do with a callback. It is plain HTTP/1.1 over
//! TCP on purpose: a front with no framework of its own has nothing to agree with the replica about,
//! and every byte it forwards is the byte it read.

use std::collections::{HashMap, VecDeque};
use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::edge::{Ring, Target, ring_order};

/// E4: the edge answers a callback within this long, `503` when no replica has answered by then.
pub const CALLBACK_BUDGET: Duration = Duration::from_secs(25);

/// E6: callbacks forwarded per second for any one session id; the excess is answered `429`.
pub const PER_SESSION_PER_SECOND: usize = 20;

/// The most of one request the front buffers. Far above the replica's own 4096-byte body limit, so
/// whatever the replica would refuse as too large still reaches it to be refused.
const MAX_REQUEST: usize = 64 * 1024;

/// How long the front waits for a request's head and body to arrive.
const READ_WAIT: Duration = Duration::from_secs(10);

/// Requirement 2: how many times the hashed target is tried while it answers `503` before the
/// front walks the ring, and the pause between attempts.
const SAME_TARGET_TRIES: usize = 3;
const PAUSE: Duration = Duration::from_millis(250);

/// The most attempts one log entry lists; the rest are only counted.
const LOGGED_ATTEMPTS: usize = 32;

/// What a request is, by its `Authorization` headers alone: exec notify's N21, which E1 applies at
/// the edge.
#[derive(Debug, PartialEq, Eq)]
pub enum Class {
    /// Exactly one `Authorization` header, `Bearer` (any case), one or more spaces, `bjn_v1.`.
    Callback,
    /// Some `Authorization` value is `bearer` (any case), any run of spaces and tabs, `bjn_v1.`,
    /// and the request is not a callback: refused `400`, never forwarded.
    LooksLike,
    /// Not a callback.
    Other,
}

/// A header's value with leading and trailing spaces and tabs removed.
fn ows(v: &str) -> &str {
    v.trim_matches([' ', '\t'])
}

pub fn classify(headers: &[(String, String)]) -> Class {
    let auth: Vec<&str> = headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("authorization"))
        .map(|(_, v)| ows(v))
        .collect();
    if let [only] = auth.as_slice()
        && is_bearer_token(only)
    {
        return Class::Callback;
    }
    if auth.iter().any(|v| looks_like(v)) {
        Class::LooksLike
    } else {
        Class::Other
    }
}

fn is_bearer_token(v: &str) -> bool {
    let (Some(scheme), Some(rest)) = (v.get(..6), v.get(6..)) else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer")
        && rest.starts_with(' ')
        && rest.trim_start_matches(' ').starts_with("bjn_v1.")
}

fn looks_like(v: &str) -> bool {
    let (Some(scheme), Some(rest)) = (v.get(..6), v.get(6..)) else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer")
        && rest.trim_start_matches([' ', '\t']).starts_with("bjn_v1.")
}

/// Every value of the query parameter `key`, keys and values decoded as
/// `application/x-www-form-urlencoded`, exactly as the replica decodes them (E1).
pub fn query_values(query: &str, key: &str) -> Vec<String> {
    query
        .split('&')
        .filter(|p| !p.is_empty())
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            (form_decode(k) == key).then(|| form_decode(v))
        })
        .collect()
}

/// `application/x-www-form-urlencoded` decoding: `+` is a space, `%XX` a byte, anything malformed
/// kept as it is, and invalid UTF-8 replaced (as the `url` crate the replica uses does).
fn form_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' => match (
                bytes.get(i + 1).and_then(|b| hex(*b)),
                bytes.get(i + 2).and_then(|b| hex(*b)),
            ) {
                (Some(h), Some(l)) => {
                    out.push(h << 4 | l);
                    i += 3;
                    continue;
                }
                _ => out.push(b'%'),
            },
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// One request as the front read it: every byte, and the parts it routes by.
pub struct Request {
    /// Head and body, exactly as they arrived — what E2 forwards.
    pub raw: Vec<u8>,
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ReadError {
    /// The peer went away, or sent nothing usable in time.
    Gone,
    /// Larger than [`MAX_REQUEST`].
    TooLarge,
    /// Not an HTTP/1.1 request head.
    Malformed,
}

/// Read one request: its head, then its body by `Transfer-Encoding: chunked` when it is chunked, else
/// by a single `Content-Length`, else none. Nothing is changed; the bytes read are the bytes kept.
pub async fn read_request<S: AsyncRead + Unpin>(sock: &mut S) -> Result<Request, ReadError> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > MAX_REQUEST {
            return Err(ReadError::TooLarge);
        }
        match sock.read(&mut chunk).await {
            Ok(0) | Err(_) => return Err(ReadError::Gone),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = std::str::from_utf8(&buf[..head_end])
        .map_err(|_| ReadError::Malformed)?
        .to_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = match (first.next(), first.next(), first.next()) {
        (Some(m), Some(t), Some(v)) if v.starts_with("HTTP/1.") && !m.is_empty() => (m, t),
        _ => return Err(ReadError::Malformed),
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut headers = Vec::new();
    for line in lines {
        let (n, v) = line.split_once(':').ok_or(ReadError::Malformed)?;
        headers.push((n.to_owned(), v.to_owned()));
    }
    let values = |name: &str| -> Vec<String> {
        headers
            .iter()
            .filter(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| ows(v).to_owned())
            .collect()
    };
    let body_start = head_end + 4;
    // Framed by chunks when the last coding named is `chunked`, as HTTP/1.1 frames it; any other
    // `Transfer-Encoding` leaves the body to a `Content-Length`, and the replica to refuse it.
    let chunked = values("transfer-encoding")
        .last()
        .and_then(|v| v.rsplit(',').next())
        .is_some_and(|c| c.trim().eq_ignore_ascii_case("chunked"));
    let lengths = values("content-length");
    let want = if chunked {
        None
    } else if let [one] = lengths.as_slice() {
        Some(one.parse::<usize>().map_err(|_| ReadError::Malformed)?)
    } else {
        // None, or more than one: the head alone is the request, and the replica refuses it.
        Some(0)
    };
    if want.is_some_and(|n| n > MAX_REQUEST) {
        return Err(ReadError::TooLarge);
    }
    loop {
        let body = &buf[body_start..];
        let done = match want {
            Some(n) => body.len() >= n,
            None => chunked_end(body).is_some(),
        };
        if done {
            let end = body_start + want.unwrap_or_else(|| chunked_end(body).unwrap_or(0));
            buf.truncate(end);
            break;
        }
        if buf.len() > MAX_REQUEST {
            return Err(ReadError::TooLarge);
        }
        match sock.read(&mut chunk).await {
            Ok(0) | Err(_) => return Err(ReadError::Gone),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let (method, path, query) = (method.to_owned(), path.to_owned(), query.to_owned());
    Ok(Request {
        raw: buf,
        method,
        path,
        query,
        headers,
    })
}

/// Where a chunked body ends (the length of the body including its terminating chunk and
/// trailers), or `None` while it is incomplete or cannot be read as chunks.
fn chunked_end(body: &[u8]) -> Option<usize> {
    let mut at = 0;
    loop {
        let line_end = at + body[at..].windows(2).position(|w| w == b"\r\n")?;
        let size_text = std::str::from_utf8(&body[at..line_end]).ok()?;
        let size_text = size_text.split(';').next()?.trim();
        let size = usize::from_str_radix(size_text, 16).ok()?;
        at = line_end + 2;
        if size == 0 {
            // Trailers, then an empty line.
            loop {
                let end = at + body[at..].windows(2).position(|w| w == b"\r\n")?;
                let empty = end == at;
                at = end + 2;
                if empty {
                    return Some(at);
                }
            }
        }
        at = at.checked_add(size)?.checked_add(2)?;
        if at > body.len() {
            return None;
        }
    }
}

/// The front: a listener, the ring it shares with [`crate::edge::Edge`], and what it has logged.
pub struct Front {
    /// `http://127.0.0.1:<port>` — what replicas are given as `--notify-origin`.
    pub origin: String,
    state: Arc<State>,
    task: tokio::task::AbortHandle,
}

struct State {
    ring: Ring,
    budget: Duration,
    log: Mutex<Vec<Value>>,
    log_file: Option<Mutex<std::fs::File>>,
    /// When each session id's recent callbacks were forwarded, for E6.
    recent: Mutex<HashMap<String, VecDeque<Instant>>>,
}

impl Drop for Front {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Front {
    /// Listen on loopback and route callbacks over `ring`, logging each to `log_path` as well as
    /// in memory.
    pub async fn start(ring: Ring, log_path: Option<&Path>) -> Result<Self, String> {
        Self::start_with_budget(ring, log_path, CALLBACK_BUDGET).await
    }

    pub async fn start_with_budget(
        ring: Ring,
        log_path: Option<&Path>,
        budget: Duration,
    ) -> Result<Self, String> {
        let listener = beyond_test_support::ports::tokio_listener().await;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("front address: {e}"))?;
        let log_file = match log_path {
            Some(p) => Some(Mutex::new(
                std::fs::File::create(p).map_err(|e| format!("front log {p:?}: {e}"))?,
            )),
            None => None,
        };
        let state = Arc::new(State {
            ring,
            budget,
            log: Mutex::new(Vec::new()),
            log_file,
            recent: Mutex::new(HashMap::new()),
        });
        let serving = Arc::clone(&state);
        let task = tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                tokio::spawn(handle(Arc::clone(&serving), sock));
            }
        });
        Ok(Self {
            origin: format!("http://{addr}"),
            state,
            task: task.abort_handle(),
        })
    }

    /// `host:port`, for a client that dials the front directly.
    pub fn authority(&self) -> &str {
        self.origin.trim_start_matches("http://")
    }

    /// Everything the front has logged, one entry per request it answered.
    pub fn log(&self) -> Vec<Value> {
        self.state.log.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// The entries for callbacks naming `session_id`.
    pub fn log_for(&self, session_id: &str) -> Vec<Value> {
        self.log()
            .into_iter()
            .filter(|e| e["session_id"] == session_id)
            .collect()
    }
}

/// A fixed answer the front writes itself. Its bodies say nothing beyond the status.
fn own_answer(status: u16) -> Vec<u8> {
    let (reason, body) = match status {
        400 => ("Bad Request", r#"{"error":"bad request"}"#),
        403 => ("Forbidden", r#"{"error":"forbidden"}"#),
        404 => ("Not Found", r#"{"error":"not found"}"#),
        413 => ("Payload Too Large", r#"{"error":"payload too large"}"#),
        429 => ("Too Many Requests", r#"{"error":"too many requests"}"#),
        _ => ("Service Unavailable", r#"{"error":"unavailable"}"#),
    };
    let retry = if status == 503 || status == 429 {
        "Retry-After: 1\r\n"
    } else {
        ""
    };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n{retry}\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// The status of a raw HTTP response.
pub fn status_of(raw: &[u8]) -> u16 {
    let line = raw.split(|b| *b == b'\r').next().unwrap_or(&[]);
    std::str::from_utf8(line)
        .ok()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}

async fn handle(state: Arc<State>, mut sock: TcpStream) {
    let began = Instant::now();
    let req = match tokio::time::timeout(READ_WAIT, read_request(&mut sock)).await {
        Ok(Ok(r)) => r,
        Ok(Err(ReadError::TooLarge)) => {
            let _ = sock.write_all(&own_answer(413)).await;
            return;
        }
        Ok(Err(ReadError::Malformed)) => {
            let _ = sock.write_all(&own_answer(400)).await;
            return;
        }
        Ok(Err(ReadError::Gone)) | Err(_) => return,
    };
    let mut entry = json!({
        "method": req.method,
        "path": req.path,
    });
    let answer = match classify(&req.headers) {
        Class::LooksLike => {
            entry["class"] = json!("looks-like");
            own_answer(400)
        }
        // This double fronts only the callback path; a session's own requests go through
        // `edge::place`. Refused here rather than forwarded anywhere, so E1's "never forwarded as
        // a user request" is checkable by what the replicas received.
        Class::Other => {
            entry["class"] = json!("other");
            own_answer(404)
        }
        Class::Callback => {
            entry["class"] = json!("callback");
            route_callback(&state, &req, &mut entry, began).await
        }
    };
    entry["answered"] = json!(status_of(&answer));
    entry["ms"] = json!(began.elapsed().as_millis() as u64);
    let _ = sock.write_all(&answer).await;
    let _ = sock.shutdown().await;
    record(&state, entry);
}

fn record(state: &State, entry: Value) {
    if let Some(f) = &state.log_file
        && let Ok(mut f) = f.lock()
    {
        let _ = writeln!(f, "{entry}");
        let _ = f.flush();
    }
    if let Ok(mut log) = state.log.lock() {
        log.push(entry);
    }
}

/// E1's checks, E6's limit, then E3's routing within E4's budget. The answer to write back.
async fn route_callback(
    state: &State,
    req: &Request,
    entry: &mut Value,
    began: Instant,
) -> Vec<u8> {
    if req.method != "POST" || req.path != "/_beyond/agent" {
        return own_answer(403);
    }
    let sessions = query_values(&req.query, "session_id");
    let [session_id] = sessions.as_slice() else {
        entry["session_ids"] = json!(sessions.len());
        return own_answer(400);
    };
    entry["session_id"] = json!(session_id);
    if !admit(state, session_id) {
        return own_answer(429);
    }

    let deadline = tokio::time::Instant::from_std(began + state.budget);
    let mut attempts: Vec<Value> = Vec::new();
    let mut tried = 0usize;
    let mut last_503: Option<Vec<u8>> = None;
    'walk: loop {
        let ring: Vec<Target> = state.ring.lock().map(|r| r.clone()).unwrap_or_default();
        let order = ring_order(&ring, session_id);
        let Some(hashed) = order.first() else {
            break 'walk;
        };
        // Requirement 2: the hashed target first, retried while it answers 503; then the walk from
        // it, owner first. Repeated, re-reading the ring each time, until the budget is spent.
        let plan =
            std::iter::repeat_n(*hashed, SAME_TARGET_TRIES).chain(order.iter().skip(1).copied());
        for target in plan {
            if tokio::time::Instant::now() >= deadline {
                break 'walk;
            }
            tried += 1;
            let outcome = tokio::time::timeout_at(deadline, forward(target, &req.raw)).await;
            let note = match outcome {
                // E4: the budget ran out with the replica still silent. Dropping the attempt closes
                // its connection, so the replica stops waiting for it.
                Err(_) => {
                    if attempts.len() < LOGGED_ATTEMPTS {
                        attempts.push(json!([target.name, "cancelled"]));
                    }
                    break 'walk;
                }
                Ok(Err(_)) => json!([target.name, "unreachable"]),
                Ok(Ok(raw)) => {
                    let status = status_of(&raw);
                    if status != 503 {
                        if attempts.len() < LOGGED_ATTEMPTS {
                            attempts.push(json!([target.name, status]));
                        }
                        entry["by"] = json!(target.name);
                        entry["attempts"] = Value::Array(attempts);
                        entry["tried"] = json!(tried);
                        return raw;
                    }
                    last_503 = Some(raw);
                    json!([target.name, 503])
                }
            };
            if attempts.len() < LOGGED_ATTEMPTS {
                attempts.push(note);
            }
            tokio::select! {
                () = tokio::time::sleep(PAUSE) => {}
                () = tokio::time::sleep_until(deadline) => break 'walk,
            }
        }
    }
    entry["attempts"] = Value::Array(attempts);
    entry["tried"] = json!(tried);
    // E5: when every replica said 503, the sandbox hears the replica's 503 as it was sent.
    last_503.unwrap_or_else(|| own_answer(503))
}

/// E6: whether one more callback for `session_id` may be forwarded this second, recording it if so.
fn admit(state: &State, session_id: &str) -> bool {
    let Ok(mut recent) = state.recent.lock() else {
        return false;
    };
    let now = Instant::now();
    let q = recent.entry(session_id.to_owned()).or_default();
    while q
        .front()
        .is_some_and(|t| now.duration_since(*t) >= Duration::from_secs(1))
    {
        q.pop_front();
    }
    if q.len() >= PER_SESSION_PER_SECOND {
        return false;
    }
    q.push_back(now);
    true
}

/// One attempt: the request's bytes to `target`, unchanged, and the whole answer back.
async fn forward(target: &Target, raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut up = TcpStream::connect((target.addr.host.as_str(), target.addr.port))
        .await
        .map_err(|e| format!("connect {}: {e}", target.addr))?;
    up.write_all(raw)
        .await
        .map_err(|e| format!("write {}: {e}", target.addr))?;
    let mut out = Vec::new();
    // Replicas answer callbacks with `Connection: close`, so the answer is everything until EOF. A
    // reset after a complete answer is still that answer.
    if let Err(e) = up.read_to_end(&mut out).await
        && out.is_empty()
    {
        return Err(format!("read {}: {e}", target.addr));
    }
    if out.is_empty() {
        return Err(format!("{} closed without answering", target.addr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth(values: &[&str]) -> Vec<(String, String)> {
        values
            .iter()
            .map(|v| ("Authorization".to_owned(), (*v).to_owned()))
            .collect()
    }

    /// E1: the two rules of N21, byte for byte as the replica applies them.
    #[test]
    fn classification_follows_the_two_rules() {
        for v in [
            "Bearer bjn_v1.x",
            "bearer bjn_v1.x",
            "BEARER    bjn_v1.x",
            "  Bearer bjn_v1.x \t",
        ] {
            assert_eq!(classify(&auth(&[v])), Class::Callback, "{v:?}");
        }
        for v in [
            &["Bearer\tbjn_v1.x"][..],
            &["Bearerbjn_v1.x"],
            &["Bearer bjn_v1.x", "Bearer bjn_v1.x"],
            &["Basic x", "bearer \t bjn_v1.y"],
        ] {
            assert_eq!(classify(&auth(v)), Class::LooksLike, "{v:?}");
        }
        for v in [
            &["Basic x"][..],
            &[],
            &["Bearer bai_v1.x"],
            &["Bearer  xbjn_v1."],
        ] {
            assert_eq!(classify(&auth(v)), Class::Other, "{v:?}");
        }
    }

    /// E1: `session_id` is counted after decoding keys, so an encoded key is the same key.
    #[test]
    fn session_ids_are_counted_as_the_replica_decodes_them() {
        assert_eq!(
            query_values("session_id=s1.a&tenant=t1", "session_id"),
            ["s1.a"]
        );
        assert_eq!(
            query_values("session%5Fid=s1.a&session_id=s1.b", "session_id").len(),
            2
        );
        assert_eq!(query_values("session_id=s1%2Ea", "session_id"), ["s1.a"]);
        assert_eq!(query_values("session_id=a+b", "session_id"), ["a b"]);
        assert_eq!(query_values("session_id=%zz", "session_id"), ["%zz"]);
        assert!(query_values("tenant=t1", "session_id").is_empty());
    }

    /// E2: the bytes kept are the bytes read, chunked bodies included.
    #[tokio::test]
    async fn a_request_is_kept_byte_for_byte() {
        let chunked: &[u8] = b"POST /_beyond/agent?session_id=s1.a HTTP/1.1\r\nHost: x\r\n\
            Transfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n2;ext=1\r\nef\r\n0\r\nX-T: 1\r\n\r\n";
        let mut extra = chunked.to_vec();
        extra.extend_from_slice(b"GARBAGE");
        let got = read_request(&mut &extra[..]).await.ok().map(|r| r.raw);
        assert_eq!(got.as_deref(), Some(chunked));

        let sized: &[u8] =
            b"POST /p HTTP/1.1\r\nContent-Length: 3\r\nAuthorization: Bearer bjn_v1.t\r\n\r\nabc";
        let got = read_request(&mut &sized[..]).await.ok().map(|r| r.raw);
        assert_eq!(got.as_deref(), Some(sized));

        let short: &[u8] = b"POST /p HTTP/1.1\r\nContent-Length: 9\r\n\r\nabc";
        assert_eq!(
            read_request(&mut &short[..]).await.err(),
            Some(ReadError::Gone)
        );
    }

    #[test]
    fn statuses_are_read_from_the_status_line() {
        assert_eq!(status_of(b"HTTP/1.1 503 Service Unavailable\r\n\r\n"), 503);
        assert_eq!(status_of(b"HTTP/1.1 200 OK\r\n\r\n{}"), 200);
        assert_eq!(status_of(b""), 0);
    }
}
