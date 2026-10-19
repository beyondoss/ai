//! The `job_ended` callback receiver: exec v1.4's "The receiver" ([`crate::exec_endpoint`]).
//!
//! A sandbox whose job was kept with `notify` POSTs `{"type":"job_ended","scope","id"}` to the
//! session's own address with `Authorization: Bearer bjn_v1.…`. Such a request is told apart from
//! every other one by its head alone ([`classify`]), before anything else about it is checked or
//! parsed (`Content-Length` included), and is then handled here, never by the front door's grant
//! check: the token is its only credential.
//!
//! [`serve`] checks it in the spec's order and answers at the first failure: path and method, the
//! query's `session_id` and `tenant`, the body (exactly one `Content-Length`, no
//! `Transfer-Encoding`, at most [`MAX_BODY`] bytes, exactly `type`, `scope` and `id`), service mode,
//! the token ([`crate::notify_token`]), and the scope. Only then does it look for the session, and
//! only among sessions a task in this process already holds ([`NotifyHub::find`]): a callback never
//! opens one. A session not held here is answered `503`, which the sender retries (the edge may
//! reach the owner next time). Every refusal has a fixed body for its status ([`refusal_body`]).
//!
//! A callback never pins a session, so it is never an attached client and never touches the idle
//! clock; its answer comes through its own reply path ([`SessionNotify::callback`]), never out of
//! the session's output.

use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::time::Instant;

use crate::exec_endpoint::jobs::{JobId, JobScope};
use crate::notify_token::Verdict;
use crate::tools::job_notify::NotifyHub;

/// The path a callback must be POSTed to: the session's own address.
pub const PATH: &str = "/_beyond/agent";

/// The largest callback body accepted.
pub const MAX_BODY: usize = 4096;

/// The longest `scope` a callback may name.
const MAX_SCOPE: usize = 64;

/// How long a callback's body may take to arrive, all of it.
const BODY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// What a request is, by its `Authorization` headers.
#[derive(Debug, PartialEq, Eq)]
pub enum Class<'a> {
    /// A callback, with its token.
    Callback(&'a str),
    /// It looks like a callback and is not one: refused `400` before anything else.
    LooksLike,
    /// Anything else: the front door's.
    Other,
}

/// A header's value with leading and trailing spaces and tabs removed.
fn ows(v: &str) -> &str {
    v.trim_matches([' ', '\t'])
}

/// Classify a request from its headers alone. A *callback* has exactly one `Authorization` header,
/// whose value is `Bearer` (any ASCII case), one or more spaces, then `bjn_v1.`; its token is
/// everything after the spaces. A request *looks like* one when any `Authorization` value is
/// `bearer` (any case), any run of spaces and tabs, then `bjn_v1.`.
pub fn classify(headers: &[(String, String)]) -> Class<'_> {
    let mut auth = headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("authorization"))
        .map(|(_, v)| ows(v));
    let first = auth.next();
    let second = auth.next();
    if let (Some(v), None) = (first, second)
        && let Some(token) = bearer_token(v)
    {
        return Class::Callback(token);
    }
    let looks = headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("authorization"))
        .any(|(_, v)| looks_like(ows(v)));
    if looks {
        Class::LooksLike
    } else {
        Class::Other
    }
}

/// `Bearer`, one or more spaces, `bjn_v1.…`: the token.
fn bearer_token(v: &str) -> Option<&str> {
    let scheme = v.get(..6)?;
    let rest = &v[6..];
    if !scheme.eq_ignore_ascii_case("bearer") || !rest.starts_with(' ') {
        return None;
    }
    let token = rest.trim_start_matches(' ');
    token.starts_with("bjn_v1.").then_some(token)
}

fn looks_like(v: &str) -> bool {
    v.get(..6).is_some_and(|s| s.eq_ignore_ascii_case("bearer"))
        && v[6..]
            .trim_start_matches([' ', '\t'])
            .starts_with("bjn_v1.")
}

/// Whether a request whose head could not be parsed may have been a callback: its raw bytes name
/// the token's prefix. Its refusal then gets the fixed body too.
pub fn may_be_callback(raw: &[u8]) -> bool {
    raw.windows(7).any(|w| w == b"bjn_v1.")
}

/// The fixed body of a callback's refusal with `status`, and the status line's reason. Says
/// nothing beyond the status.
pub fn refusal_body(status: u16) -> (&'static str, &'static str) {
    match status {
        400 => ("Bad Request", r#"{"error":"bad request"}"#),
        401 => ("Unauthorized", r#"{"error":"unauthorized"}"#),
        403 => ("Forbidden", r#"{"error":"forbidden"}"#),
        404 => ("Not Found", r#"{"error":"not found"}"#),
        410 => ("Gone", r#"{"error":"gone"}"#),
        411 => ("Length Required", r#"{"error":"length required"}"#),
        413 => ("Payload Too Large", r#"{"error":"payload too large"}"#),
        _ => ("Service Unavailable", r#"{"error":"unavailable"}"#),
    }
}

/// The body of an accepted callback's answer: a `job_ended` response.
pub const ANSWER_BODY: &str = r#"{"type":"response","command":"job_ended","success":true}"#;

/// What [`serve`] needs to know of the replica.
pub struct Receiver<'a> {
    /// `serve --service`: callbacks are refused (`403`) anywhere else.
    pub service_mode: bool,
    /// The replica's notify key and open sessions; `None` outside service mode.
    pub hub: Option<&'a NotifyHub>,
}

/// A callback's request head, as the listener parsed it.
pub struct Head<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub query: Option<&'a str>,
    pub headers: &'a [(String, String)],
}

/// Serve one callback: check it, look for its session, and answer. `leftover` is what was read
/// past the head; `arrived` is when the head was read.
pub async fn serve<S>(
    stream: &mut S,
    head: &Head<'_>,
    token: &str,
    leftover: &[u8],
    arrived: Instant,
    rx: &Receiver<'_>,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let status = match check(stream, head, token, leftover, arrived, rx).await {
        Ok(()) => return,
        Err(Some(status)) => status,
        // The connection is gone, or the answer was already written.
        Err(None) => return,
    };
    refuse(stream, status).await;
    if let Some(m) = rx.hub.and_then(NotifyHub::metrics) {
        m.job_callback(crate::metrics::JobCallback::Refused);
    }
}

/// Write a callback's refusal with its fixed body.
pub async fn refuse<S: AsyncWrite + Unpin>(stream: &mut S, status: u16) {
    let (reason, body) = refusal_body(status);
    let _ = write(stream, status, reason, body).await;
}

async fn write<S: AsyncWrite + Unpin>(
    stream: &mut S,
    status: u16,
    reason: &str,
    body: &str,
) -> std::io::Result<()> {
    let mut out = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if status == 503 {
        out.push_str("Retry-After: 1\r\n");
    }
    out.push_str("\r\n");
    out.push_str(body);
    stream.write_all(out.as_bytes()).await?;
    stream.flush().await
}

/// The query's `session_id` and `tenant`, each exactly once and well formed.
fn url_session(query: Option<&str>) -> Option<(String, String)> {
    let (mut sessions, mut tenants) = (Vec::new(), Vec::new());
    for (k, v) in url::form_urlencoded::parse(query.unwrap_or("").as_bytes()) {
        match k.as_ref() {
            "session_id" => sessions.push(v.into_owned()),
            "tenant" => tenants.push(v.into_owned()),
            _ => {}
        }
    }
    match (
        <[String; 1]>::try_from(sessions),
        <[String; 1]>::try_from(tenants),
    ) {
        (Ok([session]), Ok([tenant]))
            if crate::session_store::is_valid_session_id(&session)
                && crate::service::is_safe_segment(&tenant) =>
        {
            Some((tenant, session))
        }
        _ => None,
    }
}

/// The values of every header named `name`.
fn values<'a>(headers: &'a [(String, String)], name: &str) -> Vec<&'a str> {
    headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| ows(v))
        .collect()
}

/// A v1.3 job id: lowercase hex, 1 to 32 characters.
fn is_job_id(id: &str) -> bool {
    (1..=32).contains(&id.len()) && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The checks, in order. `Err(Some(status))` is a refusal to write (and count); `Err(None)` means nothing
/// more is to be written; `Ok` that the answer was written.
async fn check<S>(
    stream: &mut S,
    head: &Head<'_>,
    token: &str,
    leftover: &[u8],
    arrived: Instant,
    rx: &Receiver<'_>,
) -> Result<(), Option<u16>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // (1) Path and method.
    if head.path != PATH {
        return Err(Some(404));
    }
    if head.method != "POST" {
        return Err(Some(403));
    }
    // (2) The query.
    let Some((tenant, session_id)) = url_session(head.query) else {
        return Err(Some(400));
    };
    // (3) The body: exactly one Content-Length, no Transfer-Encoding.
    let lengths = values(head.headers, "content-length");
    let chunked = !values(head.headers, "transfer-encoding").is_empty();
    let len = match (lengths.as_slice(), chunked) {
        ([], false) => return Err(Some(411)),
        ([one], false) => {
            if one.is_empty() || !one.bytes().all(|b| b.is_ascii_digit()) {
                return Err(Some(400));
            }
            // Too many digits to be a size is too large.
            one.parse::<usize>().unwrap_or(usize::MAX)
        }
        _ => return Err(Some(400)),
    };
    if len > MAX_BODY {
        return Err(Some(413));
    }
    let body = match tokio::time::timeout(BODY_TIMEOUT, read_body(stream, leftover, len)).await {
        Ok(Some(b)) => b,
        _ => return Err(Some(400)),
    };
    let Ok(Value::Object(body)) = serde_json::from_slice::<Value>(&body) else {
        return Err(Some(400));
    };
    if body.get("type").and_then(Value::as_str) != Some("job_ended") {
        return Err(Some(403));
    }
    let (Some(scope), Some(id)) = (
        body.get("scope").and_then(Value::as_str),
        body.get("id").and_then(Value::as_str),
    ) else {
        return Err(Some(400));
    };
    if body.len() != 3 || !is_job_id(id) || scope.len() > MAX_SCOPE {
        return Err(Some(400));
    }
    // (4) Service mode.
    let (true, Some(hub)) = (rx.service_mode, rx.hub) else {
        return Err(Some(403));
    };
    // (5) The token, from the URL's tenant and session and the body's scope and id.
    match hub.key().verify(token, &tenant, &session_id, scope, id) {
        Verdict::Valid => {}
        v => return Err(v.refusal_status()),
    }
    // (6) The session's own scope.
    if scope != JobScope::for_session(Some(&tenant), &session_id).name() {
        return Err(Some(403));
    }
    // Only now: is the session held here? A callback never opens one.
    let Some(session) = hub.find(&tenant, &session_id) else {
        if let Some(m) = hub.metrics() {
            m.job_callback(crate::metrics::JobCallback::NotOpen);
        }
        refuse(stream, 503).await;
        return Err(None);
    };
    let Some(id) = JobId::parse(id) else {
        return Err(Some(400));
    };
    // Answer through this request's own path; a connection that closes first stops the wait.
    let answer = tokio::select! {
        a = session.callback(&id, arrived) => a,
        () = closed(stream) => return Err(None),
    };
    drop(session);
    // The session's driver counted its answer.
    if answer == 200 {
        let _ = write(stream, 200, "OK", ANSWER_BODY).await;
    } else {
        refuse(stream, answer).await;
    }
    Ok(())
}

/// Exactly `len` body bytes: what was read past the head, then the rest. `None` if the connection
/// ends first.
async fn read_body<S: AsyncRead + Unpin>(
    stream: &mut S,
    leftover: &[u8],
    len: usize,
) -> Option<Vec<u8>> {
    let mut body = leftover[..leftover.len().min(len)].to_vec();
    let mut tmp = [0u8; 1024];
    while body.len() < len {
        let want = (len - body.len()).min(tmp.len());
        match stream.read(&mut tmp[..want]).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
        }
    }
    Some(body)
}

/// Resolves when the peer closes the connection (or it fails). Bytes it sends meanwhile are
/// discarded.
async fn closed<S: AsyncRead + Unpin>(stream: &mut S) {
    let mut tmp = [0u8; 256];
    loop {
        match stream.read(&mut tmp).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn auth(values: &[&str]) -> Vec<(String, String)> {
        values
            .iter()
            .map(|v| ("Authorization".to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn the_token_header_is_read_exactly() {
        let t = "bjn_v1.4eba5aa3.x";
        for (values, want) in [
            (vec!["Bearer bjn_v1.4eba5aa3.x"], Class::Callback(t)),
            (vec!["bearer bjn_v1.4eba5aa3.x"], Class::Callback(t)),
            (vec!["BEARER   bjn_v1.4eba5aa3.x"], Class::Callback(t)),
            (vec!["  Bearer bjn_v1.4eba5aa3.x \t"], Class::Callback(t)),
            (vec!["Bearer\tbjn_v1.4eba5aa3.x"], Class::LooksLike),
            (vec!["Bearerbjn_v1.4eba5aa3.x"], Class::LooksLike),
            (vec!["Bearer \tbjn_v1.4eba5aa3.x"], Class::LooksLike),
            (
                vec!["Bearer bjn_v1.4eba5aa3.x", "Bearer bjn_v1.4eba5aa3.x"],
                Class::LooksLike,
            ),
            (vec!["Bearer bjn_v1.a", "Basic xyz"], Class::LooksLike),
            (vec!["Bearer bsg_v1.1.x.y"], Class::Other),
            (vec!["Basic bjn_v1.x"], Class::Other),
            (vec![], Class::Other),
            (vec!["Bearer bjn_v1"], Class::Other),
        ] {
            assert_eq!(classify(&auth(&values)), want, "{values:?}");
        }
        // Other headers are not looked at.
        let h = vec![("X-Authorization".into(), "Bearer bjn_v1.x".into())];
        assert_eq!(classify(&h), Class::Other);
    }

    #[test]
    fn refusals_are_fixed_per_status() {
        let mut seen = std::collections::HashSet::new();
        for s in [400, 401, 403, 404, 410, 411, 413, 503] {
            let (reason, body) = refusal_body(s);
            assert!(seen.insert(body), "{s}");
            assert!(!reason.is_empty());
            serde_json::from_str::<Value>(body).unwrap();
        }
    }
}
