//! A fleet for exec v1.4's notify suites: a `serve --service` replica, the reference v1.4
//! endpoint (`exec_mock`, which makes the callbacks), and a stand-in for the fleet's edge that
//! forwards them to the replica (`--notify-origin` names it).

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use super::exec_mock::MockOptions;
use super::service::{Options, Service};
use super::{spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_send};

pub const TENANT: &str = "00tenant1";
pub const SESSION: &str = "s1.alpha";

/// The edge's stand-in: forwards each connection to the replica it currently points at, and can
/// drop every callback instead (a broken path).
pub struct Edge {
    pub origin: String,
    pub target: Arc<AtomicU16>,
    drop_all: Arc<AtomicBool>,
    forwarded: Arc<AtomicUsize>,
}

impl Edge {
    pub async fn bind() -> (Self, tokio::net::TcpListener) {
        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let origin = format!("http://{}", listener.local_addr().unwrap());
        (
            Self {
                origin,
                target: Arc::new(AtomicU16::new(0)),
                drop_all: Arc::new(AtomicBool::new(false)),
                forwarded: Arc::new(AtomicUsize::new(0)),
            },
            listener,
        )
    }

    pub fn serve(&self, listener: tokio::net::TcpListener) {
        let (target, drop_all, forwarded) = (
            self.target.clone(),
            self.drop_all.clone(),
            self.forwarded.clone(),
        );
        tokio::spawn(async move {
            while let Ok((mut inbound, _)) = listener.accept().await {
                if drop_all.load(Ordering::SeqCst) {
                    drop(inbound);
                    continue;
                }
                forwarded.fetch_add(1, Ordering::SeqCst);
                let port = target.load(Ordering::SeqCst);
                tokio::spawn(async move {
                    if let Ok(mut out) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
                        let _ = tokio::io::copy_bidirectional(&mut inbound, &mut out).await;
                    }
                });
            }
        });
    }
}

/// A replica, its endpoint, and the edge before it.
pub struct Fleet {
    pub svc: Service,
    pub edge: Edge,
    pub requests: Arc<std::sync::Mutex<Vec<String>>>,
}

pub fn bash_bg(command: &str) -> String {
    turn_tool_use(
        "t1",
        "bash",
        &json!({"command": command, "run_in_background": true}).to_string(),
    )
}

/// A fleet whose model starts `command` in the background on the first prompt, then answers every
/// later request with text.
pub async fn fleet(mock: MockOptions, command: &str, origin: Option<&str>) -> Fleet {
    let mut turns = vec![bash_bg(command), turn_text("started")];
    turns.extend((0..8).map(|i| turn_text(&format!("noted {i}"))));
    fleet_with(mock, turns, origin).await
}

/// A fleet whose model answers with `turns`, in order.
pub async fn fleet_with(mock: MockOptions, turns: Vec<String>, origin: Option<&str>) -> Fleet {
    let (base, requests) = spawn_model_server(turns);
    let (edge, listener) = Edge::bind().await;
    let origin = origin.map_or_else(|| edge.origin.clone(), str::to_string);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--notify-origin".into(), origin],
            exec: Some(mock),
            ..Default::default()
        },
    )
    .await;
    edge.target.store(svc.port, Ordering::SeqCst);
    edge.serve(listener);
    Fleet {
        svc,
        edge,
        requests,
    }
}

impl Fleet {
    /// Open the session and run the prompt that starts the job; the client stays attached.
    pub async fn start_job(&self) -> super::TestWs {
        let token = self.svc.token(TENANT, SESSION);
        let mut ws =
            ws_connect_with_headers(self.svc.port, Some(SESSION), &self.svc.header(&token))
                .await
                .expect("a valid grant opens its session");
        ws_send(
            &mut ws,
            json!({"type":"prompt","id":"p","message":"start it"}),
        )
        .await;
        let frames = super::ws_read_until_response(&mut ws, "prompt").await;
        assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
        ws
    }

    /// The job's held stream, by a needle of its command.
    pub async fn stream(&self, needle: &str) -> u32 {
        self.svc
            .exec
            .wait_for_stream(needle, Duration::from_secs(20))
            .await
            .expect("the job's stream")
            .pgid
    }

    /// Every `keep` the endpoint received.
    pub fn keeps(&self) -> Vec<Value> {
        self.svc
            .exec
            .bodies()
            .into_iter()
            .filter(|b| b["job"]["op"] == "keep")
            .map(|b| b["job"].clone())
            .collect()
    }

    pub fn ops(&self, op: &str) -> usize {
        self.svc
            .exec
            .bodies()
            .iter()
            .filter(|b| b["job"]["op"] == op)
            .count()
    }

    /// Wait (bounded) for a model request that mentions `needle`.
    pub async fn model_saw(&self, needle: &str, within: Duration) -> bool {
        let deadline = std::time::Instant::now() + within;
        while std::time::Instant::now() < deadline {
            if self
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.contains(needle))
            {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }
}

/// Whether the endpoint's stream for `pgid` is still open after `within`.
pub async fn still_held(f: &Fleet, pgid: u32, within: Duration) -> bool {
    f.svc.exec.wait_for_end(pgid, within).await.is_none()
}

/// Kills a job's group a failed assertion would leave running.
pub struct Reap(pub u32);

impl Drop for Reap {
    fn drop(&mut self) {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &format!("-{}", self.0)])
            .status();
    }
}

pub fn seal_key(svc: &Service) -> beyond_ai_agent::notify_token::NotifyKey {
    use base64::Engine as _;
    let b64 = std::fs::read_to_string(svc.minter.seal_key()).unwrap();
    let bytes: [u8; 32] = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .unwrap()
        .try_into()
        .unwrap();
    beyond_ai_agent::notify_token::NotifyKey::from_secret_bytes(&bytes)
}

/// POST a callback to `origin` + `target`; the status.
pub async fn post(origin: &str, target: &str, token: &str, body: &str) -> u16 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let addr = origin.trim_start_matches("http://");
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST {target} HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(30), s.read_to_end(&mut out)).await;
    String::from_utf8_lossy(&out)
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}
