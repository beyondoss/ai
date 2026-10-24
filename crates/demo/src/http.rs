//! Just enough HTTP/1.1 for the demo's stand-ins: `Connection: close`, `Content-Length` bodies, and
//! a client that also reads `chunked` answers (an identity provider may send them). The demo needs a
//! handful of JSON-RPC and OAuth POSTs answered, not a web framework.

use std::time::{SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpStream;

pub type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type Headers = Vec<(String, String)>;

const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 4 * 1024 * 1024;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

pub struct Req {
    pub method: String,
    /// The request target as sent: path and query.
    pub path: String,
    pub headers: Headers,
    pub body: Vec<u8>,
}

impl Req {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The path without its query.
    pub fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or("")
    }

    /// A query parameter, percent-decoded.
    pub fn query(&self, name: &str) -> Option<String> {
        let q = self.path.split_once('?')?.1;
        form(q).into_iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }
}

/// `a=1&b=two%20words` as decoded pairs.
pub fn form(s: &str) -> Vec<(String, String)> {
    s.split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (pct_decode(k), pct_decode(v))
        })
        .collect()
}

pub fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 1..i + 3]).ok();
                match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(v) => {
                        out.push(v);
                        i += 2;
                    }
                    None => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Percent-encode everything but the unreserved characters.
pub fn pct_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &c in s.as_bytes() {
        if c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'~') {
            out.push(c as char);
        } else {
            out.push_str(&format!("%{c:02X}"));
        }
    }
    out
}

/// `application/x-www-form-urlencoded` body.
pub fn form_body(pairs: &[(&str, &str)]) -> Vec<u8> {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", pct_encode(k), pct_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
        .into_bytes()
}

/// Split `buf` at the end of the head; `None` until the blank line has arrived.
fn head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

fn parse_headers(head: &str) -> Headers {
    head.lines()
        .skip(1)
        .filter_map(|l| l.split_once(':'))
        .map(|(n, v)| (n.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect()
}

pub async fn read_request(s: &mut TcpStream) -> Res<Option<Req>> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let end = loop {
        if let Some(e) = head_end(&buf) {
            break e;
        }
        if buf.len() > MAX_HEAD {
            return Err("request head too large".into());
        }
        let n = s.read(&mut chunk).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut first = head.lines().next().unwrap_or("").split_whitespace();
    let (method, path) = (
        first.next().unwrap_or("").to_string(),
        first.next().unwrap_or("").to_string(),
    );
    let headers = parse_headers(&head);
    let len: usize = headers
        .iter()
        .find(|(n, _)| n == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    if len > MAX_BODY {
        return Err("request body too large".into());
    }
    let mut body = buf[end..].to_vec();
    while body.len() < len {
        let n = s.read(&mut chunk).await?;
        if n == 0 {
            return Err("request body cut short".into());
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(len);
    Ok(Some(Req {
        method,
        path,
        headers,
        body,
    }))
}

pub async fn respond(
    s: &mut TcpStream,
    status: u16,
    headers: &[(String, String)],
    body: &[u8],
) -> Res<()> {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        302 => "Found",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        _ => "Error",
    };
    let mut out = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-length: {}\r\nconnection: close\r\n",
        body.len()
    );
    for (n, v) in headers {
        out.push_str(&format!("{n}: {v}\r\n"));
    }
    out.push_str("\r\n");
    s.write_all(out.as_bytes()).await?;
    s.write_all(body).await?;
    s.shutdown().await?;
    Ok(())
}

pub fn json_headers() -> Headers {
    vec![("content-type".into(), "application/json".into())]
}

pub fn error_body(msg: &str) -> Vec<u8> {
    serde_json::json!({"error": msg}).to_string().into_bytes()
}

pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// A chunked body, decoded.
fn dechunk(mut raw: &[u8]) -> Res<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let eol = raw
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or("bad chunk")?;
        let size = std::str::from_utf8(&raw[..eol])?
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let size = usize::from_str_radix(size, 16)?;
        raw = &raw[eol + 2..];
        if size == 0 {
            return Ok(out);
        }
        if raw.len() < size + 2 {
            return Err("chunk cut short".into());
        }
        out.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
}

/// One request over a fresh connection; the status, lowercased headers and body of the answer.
pub async fn request(
    method: &str,
    addr: &str,
    path: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> Res<(u16, Headers, Vec<u8>)> {
    let mut s = TcpStream::connect(addr).await?;
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nhost: {addr}\r\ncontent-length: {}\r\nconnection: close\r\n",
        body.len()
    );
    for (n, v) in headers {
        head.push_str(&format!("{n}: {v}\r\n"));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes()).await?;
    s.write_all(body).await?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).await?;
    let end = head_end(&raw).ok_or("no response head")?;
    let head = String::from_utf8_lossy(&raw[..end]).into_owned();
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("no status")?;
    let headers = parse_headers(&head);
    let chunked = headers
        .iter()
        .any(|(n, v)| n == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked"));
    let body = if chunked {
        dechunk(&raw[end..])?
    } else {
        raw[end..].to_vec()
    };
    Ok((status, headers, body))
}

/// One POST; see [`request`].
pub async fn post(
    addr: &str,
    path: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> Res<(u16, Headers, Vec<u8>)> {
    request("POST", addr, path, headers, body).await
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn percent_coding_round_trips() {
        let s = "a b&c=d/é~";
        assert_eq!(pct_decode(&pct_encode(s)), s);
        assert_eq!(pct_decode("x%3Ay+z%"), "x:y z%");
        assert_eq!(
            form("code=1.2&scope=openid%20notes%3Aread"),
            vec![
                ("code".to_string(), "1.2".to_string()),
                ("scope".to_string(), "openid notes:read".to_string())
            ]
        );
    }

    #[test]
    fn a_chunked_answer_is_decoded() {
        assert_eq!(
            dechunk(b"5\r\nhello\r\n6;x=1\r\n world\r\n0\r\n\r\n").unwrap(),
            b"hello world"
        );
        assert!(dechunk(b"5\r\nhe").is_err());
    }
}
