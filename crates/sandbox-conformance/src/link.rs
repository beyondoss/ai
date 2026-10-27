//! A driver as a child process: JSON lines on its stdin and stdout (`DRIVER.md`, section 1).
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt as _;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::run::{Connector, Link};

/// How long one request may take before the driver is declared hung. Not a protocol limit: the
/// slowest check in the shared scenarios takes seconds, and this only stops a wedged driver from
/// wedging the run.
const CALL_TIMEOUT: Duration = Duration::from_secs(300);

/// How much of a dead driver's stderr is shown.
const STDERR_TAIL: usize = 2000;

struct Io {
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    _child: Child,
}

pub struct ProcessLink {
    io: Mutex<Option<Io>>,
    stderr: std::fs::File,
}

impl ProcessLink {
    /// Start `cmd` (a shell command line) with `env` added to this process's environment.
    pub fn spawn(cmd: &str, env: &BTreeMap<String, Option<String>>) -> std::io::Result<Self> {
        let stderr = tempfile::tempfile()?;
        let mut c = Command::new("sh");
        c.arg("-c")
            .arg(format!("exec {cmd}"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr.try_clone()?))
            .kill_on_drop(true);
        for (k, v) in env {
            match v {
                Some(v) => c.env(k, v),
                None => c.env_remove(k),
            };
        }
        let mut child = c.spawn()?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = BufReader::new(child.stdout.take().expect("piped")).lines();
        Ok(Self {
            io: Mutex::new(Some(Io {
                stdin,
                stdout,
                _child: child,
            })),
            stderr,
        })
    }

    fn stderr_tail(&self) -> String {
        let len = self.stderr.metadata().map(|m| m.len()).unwrap_or(0) as usize;
        let from = len.saturating_sub(STDERR_TAIL);
        let mut buf = vec![0u8; len - from];
        let _ = self.stderr.read_at(&mut buf, from as u64);
        String::from_utf8_lossy(&buf).trim().to_string()
    }

    /// An answer no scenario expects, so a dead or confused driver fails the check it was asked.
    fn broken(&self, id: &Value, kind: &str, why: &str) -> Value {
        let tail = self.stderr_tail();
        let detail = if tail.is_empty() {
            why.to_string()
        } else {
            format!("{why}; driver stderr: {tail}")
        };
        json!({"id": id, "error": {"kind": kind, "detail": detail}})
    }
}

#[async_trait]
impl Link for ProcessLink {
    async fn call(&self, req: &Value) -> Value {
        let id = req["id"].clone();
        let mut g = self.io.lock().await;
        let Some(io) = g.as_mut() else {
            return self.broken(&id, "driver_died", "the driver process is gone");
        };
        let exchange = async {
            io.stdin.write_all(format!("{req}\n").as_bytes()).await?;
            io.stdin.flush().await?;
            io.stdout.next_line().await
        };
        let line = match tokio::time::timeout(CALL_TIMEOUT, exchange).await {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                *g = None;
                return self.broken(&id, "driver_died", "the driver closed its stdout");
            }
            Ok(Err(e)) => {
                *g = None;
                return self.broken(&id, "driver_died", &format!("pipe: {e}"));
            }
            Err(_) => {
                *g = None;
                return self.broken(&id, "driver_hung", "no answer within the call timeout");
            }
        };
        match serde_json::from_str::<Value>(&line) {
            Ok(v) if v.is_object() => v,
            _ => self.broken(
                &id,
                "driver_protocol",
                &format!(
                    "not a JSON object line: {}",
                    line.chars().take(200).collect::<String>()
                ),
            ),
        }
    }
}

/// Starts `cmd` once per connection, with the backend's coordinates in the environment.
pub struct ProcessConnector {
    pub cmd: String,
    pub url: String,
    pub bearer: Option<String>,
    pub jobs_root: String,
    pub caps: Option<Value>,
}

#[async_trait]
impl Connector for ProcessConnector {
    async fn connect(&self, bearer: Option<Option<String>>, url: Option<String>) -> Box<dyn Link> {
        let mut env = BTreeMap::new();
        env.insert(
            "CONFORMANCE_URL".to_string(),
            Some(url.unwrap_or(self.url.clone())),
        );
        env.insert(
            "CONFORMANCE_BEARER".to_string(),
            bearer.unwrap_or_else(|| self.bearer.clone()),
        );
        env.insert(
            "CONFORMANCE_JOBS_ROOT".to_string(),
            Some(self.jobs_root.clone()),
        );
        env.insert(
            "CONFORMANCE_CAPS".to_string(),
            Some(self.caps.as_ref().map(Value::to_string).unwrap_or_default()),
        );
        match ProcessLink::spawn(&self.cmd, &env) {
            Ok(l) => Box::new(l),
            Err(e) => Box::new(Dead(format!("cannot start `{}`: {e}", self.cmd))),
        }
    }
}

/// A driver that never started: every request says why.
struct Dead(String);

#[async_trait]
impl Link for Dead {
    async fn call(&self, req: &Value) -> Value {
        json!({"id": req["id"], "error": {"kind": "driver_died", "detail": self.0}})
    }
}
