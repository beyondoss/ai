//! A bare HTTP/1.1 client over a TCP socket: one request, `Connection: close`, the whole answer.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::net::SocketAddr;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// What came back.
pub struct Answer {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Answer {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }

    /// The frames of an ndjson body.
    pub fn frames(&self) -> Vec<Value> {
        String::from_utf8_lossy(&self.body)
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
}

/// The request head and body as bytes.
pub fn request(method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
    let mut out = format!("{method} {path} HTTP/1.1\r\nHost: execd\r\nConnection: close\r\n");
    for (n, v) in headers {
        out.push_str(&format!("{n}: {v}\r\n"));
    }
    if !body.is_empty() || method == "POST" {
        out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    out.push_str("\r\n");
    let mut bytes = out.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

/// Send one request and read the answer to the end.
pub async fn send(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> Answer {
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(&request(method, path, headers, body))
        .await
        .unwrap();
    let mut raw = Vec::new();
    let _ = sock.read_to_end(&mut raw).await;
    parse(&raw)
}

/// POST a JSON body to `/exec`.
pub async fn post(addr: SocketAddr, headers: &[(&str, &str)], body: &Value) -> Answer {
    send(addr, "POST", "/exec", headers, body.to_string().as_bytes()).await
}

pub fn parse(raw: &[u8]) -> Answer {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a response head");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|s| s.parse().ok())
        .expect("a status");
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(n, v)| (n.trim().to_string(), v.trim().to_string()))
        .collect();
    let rest = &raw[split + 4..];
    let chunked = headers
        .iter()
        .any(|(n, v)| n.eq_ignore_ascii_case("transfer-encoding") && v.contains("chunked"));
    let body = if chunked {
        dechunk(rest)
    } else {
        rest.to_vec()
    };
    Answer {
        status,
        headers,
        body,
    }
}

fn dechunk(mut rest: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(eol) = rest.windows(2).position(|w| w == b"\r\n") else {
            return out;
        };
        let size = usize::from_str_radix(
            std::str::from_utf8(&rest[..eol])
                .unwrap()
                .split(';')
                .next()
                .unwrap()
                .trim(),
            16,
        )
        .unwrap();
        rest = &rest[eol + 2..];
        if size == 0 || rest.len() < size {
            return out;
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[(size + 2).min(rest.len())..];
    }
}
