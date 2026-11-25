//! A scripted exec endpoint for the job handle: each connection is answered by a closure that sees the
//! request's `job` object, so a test writes the frames an endpoint sends (the A1 frames:
//! `started{mark}`, `attached{mark,from}`) and decides when the front is down.
//!
//! It is a socket and a script, nothing more: the decisions under test are the client kernel's.

use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// What a connection gets.
pub enum Reply {
    /// A 200 ndjson stream of these lines, then the connection closes. With no final frame the
    /// client sees a cut.
    Stream(Vec<String>),
    /// The connection is dropped before any answer: a front that is down.
    Down,
    /// A refusal with this status and body.
    Status(u16, String),
}

/// `(connection index, the request's job object) -> the answer`.
pub type Script = Arc<dyn Fn(usize, &Value) -> Reply + Send + Sync>;

pub struct JobFront {
    pub url: String,
    /// Every request's `job` object, in arrival order.
    pub seen: Arc<Mutex<Vec<Value>>>,
}

impl JobFront {
    pub async fn start(script: Script) -> Self {
        let listener = crate::ports::tokio_listener().await;
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
        let log = seen.clone();
        tokio::spawn(async move {
            let mut n = 0usize;
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let (script, log, index) = (script.clone(), log.clone(), n);
                n += 1;
                tokio::spawn(async move {
                    let Some(job) = read_job(&mut sock).await else {
                        return;
                    };
                    log.lock().unwrap().push(job.clone());
                    let head = |status: u16, ctype: &str| {
                        format!(
                            "HTTP/1.1 {status} X\r\nContent-Type: {ctype}\r\nConnection: close\r\n\r\n"
                        )
                    };
                    match script(index, &job) {
                        Reply::Down => {}
                        Reply::Status(code, body) => {
                            let _ = sock.write_all(head(code, "text/plain").as_bytes()).await;
                            let _ = sock.write_all(body.as_bytes()).await;
                        }
                        Reply::Stream(lines) => {
                            let _ = sock
                                .write_all(head(200, "application/x-ndjson").as_bytes())
                                .await;
                            for l in lines {
                                let _ = sock.write_all(format!("{l}\n").as_bytes()).await;
                            }
                        }
                    }
                    let _ = sock.shutdown().await;
                });
            }
        });
        Self { url, seen }
    }
}

/// The request's `job` object, once its head and body are read.
async fn read_job(sock: &mut tokio::net::TcpStream) -> Option<Value> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let (head_end, len) = loop {
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..i]).to_ascii_lowercase();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            break (i + 4, len);
        }
    };
    while buf.len() < head_end + len {
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body: Value = serde_json::from_slice(&buf[head_end..head_end + len]).ok()?;
    body.get("job").cloned()
}
