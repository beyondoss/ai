//! A stand-in exec endpoint: the other side of the v1.1 exec protocol, running each command as a
//! real process on this host.
//!
//! Shared rather than per-test because two suites need the same thing for different reasons.
//! `exec_endpoint_writes` needs a **v1** endpoint (`honors_stdin: false`) that ignores `stdin_base64`
//! like any unknown field, so the 128 KiB argv cap bites for real. `serve_service_mode` needs a
//! **sandbox**: a directory that stands in for the tenant's box, with its own `HOME`, so a test can
//! tell "the tool ran in the sandbox" from "the tool ran on the replica" by which marker file it
//! finds — and can assert, from [`ExecMock::requests`], exactly which headers the grant sent.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use serde_json::{Value, json};

/// One recorded request: its headers (lowercased names) and its JSON body.
#[derive(Clone, Debug)]
pub struct ExecRequest {
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

impl ExecRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn command(&self) -> &str {
        self.body["command"].as_str().unwrap_or("")
    }

    pub fn args(&self) -> Vec<String> {
        self.body["args"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// `Clone` shares the endpoint rather than starting a second one — the request log is behind the
/// same `Arc`. That is what a driver hosting one exec double for a whole run wants: every scenario
/// holds a handle to the same server, at the one address its replicas were started with.
#[derive(Clone)]
pub struct ExecMock {
    pub url: String,
    requests: Arc<Mutex<Vec<ExecRequest>>>,
}

impl ExecMock {
    /// A v1.1 endpoint rooted at `root`, with no `HOME` of its own.
    pub async fn start(root: &Path, honors_stdin: bool) -> Self {
        Self::start_with_home(root, None, honors_stdin).await
    }

    /// A sandbox: commands run with `cwd` defaulting to `root` and `HOME` set to `home`, so `$HOME`
    /// (what `serve --service`'s startup probe asks for) is the sandbox's, never the replica's.
    pub async fn start_with_home(root: &Path, home: Option<&Path>, honors_stdin: bool) -> Self {
        Self::start_on("127.0.0.1:0", root, home, honors_stdin).await
    }

    /// As [`Self::start_with_home`], bound where the caller says.
    ///
    /// A test wants an ephemeral loopback port. A **fleet the simulator attached to** needs the
    /// opposite: replicas carry the exec URL in their grant and are started before the driver, so
    /// the address has to be predictable and reachable from another host — which `127.0.0.1:0` is
    /// neither of.
    pub async fn start_on(
        bind: &str,
        root: &Path,
        home: Option<&Path>,
        honors_stdin: bool,
    ) -> Self {
        let listener = tokio::net::TcpListener::bind(bind).await.unwrap();
        let url = format!("http://{}/exec", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let root = root.to_path_buf();
        let home = home.map(Path::to_path_buf);
        let seen = requests.clone();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                tokio::spawn(serve_one(
                    sock,
                    root.clone(),
                    home.clone(),
                    seen.clone(),
                    honors_stdin,
                ));
            }
        });
        Self { url, requests }
    }

    pub fn requests(&self) -> Vec<ExecRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// Every request body, for suites that only care about the protocol.
    pub fn bodies(&self) -> Vec<Value> {
        self.requests().into_iter().map(|r| r.body).collect()
    }
}

async fn serve_one(
    mut sock: tokio::net::TcpStream,
    root: PathBuf,
    home: Option<PathBuf>,
    seen: Arc<Mutex<Vec<ExecRequest>>>,
    honors_stdin: bool,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (mut buf, mut chunk) = (Vec::new(), [0u8; 64 * 1024]);
    let (mut need, mut head_end) = (0usize, None);
    let mut headers = Vec::new();
    loop {
        let Ok(n) = sock.read(&mut chunk).await else {
            return;
        };
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if head_end.is_none()
            && let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n")
        {
            head_end = Some(p + 4);
            let head = String::from_utf8_lossy(&buf[..p]).into_owned();
            for line in head.lines().skip(1) {
                if let Some((name, value)) = line.split_once(':') {
                    headers.push((name.trim().to_lowercase(), value.trim().to_string()));
                }
            }
            need = headers
                .iter()
                .find(|(n, _)| n == "content-length")
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0);
        }
        if head_end.is_some_and(|h| buf.len() >= h + need) {
            break;
        }
    }
    let req: Value =
        serde_json::from_slice(&buf[head_end.unwrap_or(buf.len())..]).unwrap_or(json!({}));
    seen.lock().unwrap().push(ExecRequest {
        headers,
        body: req.clone(),
    });

    let args: Vec<String> = req["args"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let stdin = req["stdin_base64"]
        .as_str()
        .filter(|_| honors_stdin)
        .map(|b| base64::engine::general_purpose::STANDARD.decode(b).unwrap());

    let mut cmd = tokio::process::Command::new(req["command"].as_str().unwrap_or("true"));
    cmd.args(&args)
        .current_dir(req["cwd"].as_str().map_or(root, Into::into))
        .stdin(if stdin.is_some() {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    if let Some(home) = &home {
        cmd.env("HOME", home);
    }
    let payload = match cmd.spawn() {
        Ok(mut child) => {
            let pipe = child.stdin.take();
            let feed = async move {
                if let (Some(mut pipe), Some(bytes)) = (pipe, stdin) {
                    let _ = pipe.write_all(&bytes).await;
                }
            };
            let (out, ()) = tokio::join!(child.wait_with_output(), feed);
            let o = out.unwrap();
            json!({
                "exit_code": o.status.code().unwrap_or(-1),
                "stdout": String::from_utf8_lossy(&o.stdout),
                "stderr": String::from_utf8_lossy(&o.stderr),
            })
        }
        // `Argument list too long` lands here — the defect v1.1 exists to fix.
        Err(e) => json!({ "exit_code": 127, "stdout": "", "stderr": e.to_string() }),
    };
    let body = payload.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = sock.write_all(resp.as_bytes()).await;
    let _ = sock.shutdown().await;
}
