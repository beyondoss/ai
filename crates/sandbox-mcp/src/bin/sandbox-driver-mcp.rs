//! The MCP server's conformance driver (`sdk/conformance/DRIVER.md`).
//!
//! It reads the contract's operations on stdin and implements each one by calling `sandbox-mcp`'s
//! tools over a real MCP session: it spawns the `sandbox-mcp` binary (next to this one, or
//! `$SANDBOX_MCP_BIN`) on stdio with the `CONFORMANCE_*` environment translated to `SANDBOX_*`, and
//! speaks to it with rmcp's client. It has no sandbox code of its own, so what it proves is the
//! server: the tools' schemas, their errors, and what a model would get back.
//!
//! Four ops are composed of tool calls because the contract's operation is a loop or a filter over
//! them: `job_attach` polls `job_output` to the job's end, `job_remove` and `job_wipe` list jobs
//! and `job_kill` the right ones with `remove`, and `exec_stream` is what an agent does to watch a
//! command: `job_start`, `job_output` from an offset until the job ends, `job_kill` at the
//! deadline, then remove the job. A job's log is one stream, so the command's stderr is sent to a
//! file of its own in the sandbox (`exec 2>file`) and read back with `read_file`. What the
//! contract's `exec_stream` has that a tool result cannot carry (frames, heartbeats) is not
//! reachable here; the scenarios that check only for it are left unanswered (see `hello`).
//!
//! `extract` has no tool of its own, and the local half of a download is the `download` tool. It
//! is run through a private `execd` (next to this binary, or `$EXECD_BIN`) whose `tar` is a
//! script that prints the archive the driver wrote with `write_file`: `download` runs its real
//! extractor, under the real `SANDBOX_LOCAL_ROOT` confinement, on bytes the harness chose.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

use base64::Engine as _;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ReadResourceRequestParams, ResourceContents,
};
use rmcp::service::{Peer, RunningService};
use rmcp::transport::TokioChildProcess;
use rmcp::{RoleClient, ServiceExt as _};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt as _, BufReader};

// The local trees of the bulk-copy ops, shared with the Rust driver.
#[path = "../../../sandbox-conformance/src/trees.rs"]
mod trees;

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;
/// The longest nap between two reads of a job that is quiet.
const FOLLOW_POLL_MAX: Duration = Duration::from_millis(250);
const DRIVER: &str = "sandbox-mcp";

/// One op's outcome in the contract's terms.
type Out = Result<Value, Value>;

const OPS: &[&str] = &[
    "hello",
    "capabilities",
    "exec",
    "exec_stream",
    "extract",
    "job_start",
    "job_list",
    "job_read",
    "job_attach",
    "job_kill",
    "job_remove",
    "job_wipe",
    "stat",
    "read",
    "write",
    "write_if_unchanged",
    "mkdir",
    "list",
    "search",
    "upload",
    "download",
];

fn err(kind: &str, detail: impl Into<String>) -> Value {
    json!({ "kind": kind, "detail": detail.into() })
}

fn b64(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

fn str_arg<'a>(req: &'a Value, key: &str) -> Result<&'a str, Value> {
    req[key]
        .as_str()
        .ok_or_else(|| err("invalid_request", format!("`{key}` must be a string")))
}

fn opt_str<'a>(req: &'a Value, key: &str) -> Result<Option<&'a str>, Value> {
    match &req[key] {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s)),
        _ => Err(err("invalid_request", format!("`{key}` must be a string"))),
    }
}

fn u64_arg(req: &Value, key: &str) -> Result<Option<u64>, Value> {
    match &req[key] {
        Value::Null => Ok(None),
        v => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| err("invalid_request", format!("`{key}` must be a count"))),
    }
}

fn flag(req: &Value, key: &str) -> Result<bool, Value> {
    match &req[key] {
        Value::Null => Ok(false),
        Value::Bool(b) => Ok(*b),
        _ => Err(err(
            "invalid_request",
            format!("`{key}` must be true or false"),
        )),
    }
}

fn strings(req: &Value, key: &str) -> Result<Vec<String>, Value> {
    match &req[key] {
        Value::Null => Ok(Vec::new()),
        Value::Array(a) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| err("invalid_request", format!("`{key}` must hold strings")))
            })
            .collect(),
        _ => Err(err("invalid_request", format!("`{key}` must be a list"))),
    }
}

fn counts(r: &Value) -> Value {
    json!({"files": r["files"], "dirs": r["dirs"], "links": r["links"], "bytes": r["bytes"]})
}

fn bytes_arg(req: &Value, key: &str) -> Result<Vec<u8>, Value> {
    B64.decode(str_arg(req, key)?)
        .map_err(|e| err("invalid_request", format!("`{key}` is not base64: {e}")))
}

/// The contract's `command` and `args` as the one shell command line the tools take: every word
/// single-quoted, so the shell hands the program exactly the arguments the contract sent, and
/// `exec`, so the program replaces the shell: a signal reaches the program itself (a job killed
/// with SIGKILL ends 137, not as the shell's 143 and `Terminated`).
fn command_line(req: &Value) -> Result<String, Value> {
    let quote = |w: &str| format!("'{}'", w.replace('\'', "'\\''"));
    let mut line = format!("exec {}", quote(str_arg(req, "command")?));
    for a in strings(req, "args")? {
        line.push(' ');
        line.push_str(&quote(&a));
    }
    Ok(line)
}

/// A tool's arguments: the keys given, with `null`s left out.
fn args(pairs: Vec<(&str, Value)>) -> Map<String, Value> {
    pairs
        .into_iter()
        .filter(|(_, v)| !v.is_null())
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

/// `s` as one shell word.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The sandbox `extract` downloads from: an `execd` of its own on loopback, with a `tar` ahead of
/// the real one on its `PATH` that prints one file, the archive. It is the far end of a download
/// that sends the bytes the harness chose; it is not the endpoint under test.
struct PrivateSandbox {
    mcp: Box<Mcp>,
    /// Where `write_file` puts the archive, and where the `tar` script reads it.
    archive: PathBuf,
    _execd: tokio::process::Child,
    _dir: tempfile::TempDir,
}

impl PrivateSandbox {
    async fn start() -> Result<Self, Value> {
        let io = |what: &str, e: &dyn std::fmt::Display| err("io", format!("{what}: {e}"));
        let execd = std::env::var_os("EXECD_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let mut p = std::env::current_exe().expect("current exe");
                p.set_file_name("execd");
                p
            });
        if !execd.is_file() {
            return Err(err(
                "io",
                format!(
                    "extract needs an execd to serve the archive: {} is not there (cargo build -p beyond-execd, or set EXECD_BIN)",
                    execd.display()
                ),
            ));
        }
        let dir = tempfile::tempdir().map_err(|e| io("tempdir", &e))?;
        let base = dir.path().canonicalize().map_err(|e| io("tempdir", &e))?;
        let bin = base.join("bin");
        let archive = base.join("archive.tar");
        std::fs::create_dir(&bin).map_err(|e| io("mkdir", &e))?;
        let tar = bin.join("tar");
        std::fs::write(
            &tar,
            format!(
                "#!/bin/sh\nexec cat -- {}\n",
                shell_quote(&archive.to_string_lossy())
            ),
        )
        .map_err(|e| io("write tar", &e))?;
        std::fs::set_permissions(&tar, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .map_err(|e| io("chmod tar", &e))?;
        let path = format!(
            "{}:{}",
            bin.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let mut child = tokio::process::Command::new(&execd)
            .env_clear()
            .env("PATH", path)
            .env("EXECD_LISTEN", "127.0.0.1:0")
            .env("EXECD_ROOT", &base)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| io("start execd", &e))?;
        let mut lines = BufReader::new(child.stderr.take().expect("piped")).lines();
        let addr = tokio::time::timeout(Duration::from_secs(20), async {
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(rest) = line.split(" listen=").nth(1) {
                    return rest.split_whitespace().next().map(str::to_string);
                }
            }
            None
        })
        .await
        .ok()
        .flatten()
        .ok_or_else(|| err("io", "execd did not report a listening address"))?;
        // Keep reading so a full pipe never stops execd.
        tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
        let mcp = Box::new(Mcp::start_at(&format!("http://{addr}/exec"), None).await?);
        Ok(Self {
            mcp,
            archive,
            _execd: child,
            _dir: dir,
        })
    }
}

/// What following a job to its end found.
struct Followed {
    output: Vec<u8>,
    /// Bytes the sandbox no longer had (a jump in the offset).
    gap: u64,
    code: Option<i64>,
    /// The deadline passed and the job was killed.
    timed_out: bool,
}

struct Mcp {
    // Held so the server (a child process) lives as long as the driver.
    _service: RunningService<RoleClient, ()>,
    peer: Peer<RoleClient>,
    /// Counts `exec_stream` jobs, for their ids.
    jobs: std::sync::atomic::AtomicU64,
    /// The sandbox `extract` reads archives from, started by the first `extract`.
    private: tokio::sync::OnceCell<PrivateSandbox>,
    /// The server's `SANDBOX_LOCAL_ROOT`: the trees of upload and download live below it.
    root: tempfile::TempDir,
}

impl Mcp {
    /// The server for the endpoint under test, from the `CONFORMANCE_*` environment.
    async fn start() -> Result<Self, Value> {
        let url = std::env::var("CONFORMANCE_URL").unwrap_or_default();
        let bearer = std::env::var("CONFORMANCE_BEARER")
            .ok()
            .filter(|b| !b.is_empty());
        Self::start_at(&url, bearer).await
    }

    async fn start_at(url: &str, bearer: Option<String>) -> Result<Self, Value> {
        let bin = std::env::var("SANDBOX_MCP_BIN").ok().unwrap_or_else(|| {
            let mut p = std::env::current_exe().expect("current exe");
            p.set_file_name("sandbox-mcp");
            p.to_string_lossy().into_owned()
        });
        let mut cmd = tokio::process::Command::new(&bin);
        for k in [
            "SANDBOX_TOKEN",
            "SANDBOX_TOKEN_CMD",
            "SANDBOX_CAPS",
            "SANDBOX_JOBS_SCOPE",
            "SANDBOX_LOCAL_ROOT",
        ] {
            cmd.env_remove(k);
        }
        let root = tempfile::tempdir().map_err(|e| err("io", e.to_string()))?;
        cmd.env("SANDBOX_LOCAL_ROOT", root.path());
        cmd.env("SANDBOX_URL", url);
        if let Some(b) = bearer {
            cmd.env("SANDBOX_TOKEN", b);
        }
        if let Some(c) = std::env::var("CONFORMANCE_CAPS")
            .ok()
            .filter(|c| !c.is_empty())
        {
            cmd.env("SANDBOX_CAPS", c);
        }
        let transport = TokioChildProcess::new(cmd)
            .map_err(|e| err("transport", format!("cannot start {bin}: {e}")))?;
        let service = ()
            .serve(transport)
            .await
            .map_err(|e| err("transport", format!("MCP handshake with {bin} failed: {e}")))?;
        let peer = service.peer().clone();
        Ok(Self {
            _service: service,
            peer,
            jobs: 0.into(),
            private: tokio::sync::OnceCell::new(),
            root,
        })
    }

    /// Call a tool; a failed call is the server's `{"error": {kind, detail}}`.
    async fn tool(&self, name: &str, arguments: Map<String, Value>) -> Out {
        let params = CallToolRequestParams::new(name.to_string()).with_arguments(arguments);
        let r: CallToolResult = self
            .peer
            .call_tool(params)
            .await
            .map_err(|e| err("protocol", format!("{name}: {e}")))?;
        let v = r
            .structured_content
            .ok_or_else(|| err("protocol", format!("{name}: no structured content")))?;
        if r.is_error == Some(true) {
            Err(v["error"].clone())
        } else {
            Ok(v)
        }
    }

    async fn capabilities(&self) -> Out {
        let r = self
            .peer
            .read_resource(ReadResourceRequestParams::new(
                "sandbox://capabilities".to_string(),
            ))
            .await
            .map_err(|e| err("protocol", format!("capabilities: {e}")))?;
        let Some(ResourceContents::TextResourceContents { text, .. }) = r.contents.first() else {
            return Err(err("protocol", "capabilities: no text content"));
        };
        let v: Value = serde_json::from_str(text).map_err(|e| err("protocol", e.to_string()))?;
        if v["error"].is_object() {
            Err(v["error"].clone())
        } else {
            Ok(v)
        }
    }

    async fn handle(&self, op: &str, req: &Value) -> Out {
        match op {
            "capabilities" => self.capabilities().await,
            "exec" => {
                let r = self
                    .tool(
                        "exec",
                        args(vec![
                            ("command", json!(command_line(req)?)),
                            ("cwd", json!(opt_str(req, "cwd")?)),
                            ("timeout_ms", json!(u64_arg(req, "timeout_ms")?)),
                            ("stdin_base64", json!(opt_str(req, "stdin_b64")?)),
                        ]),
                    )
                    .await?;
                let bytes = |k: &str| -> Result<Vec<u8>, Value> {
                    let text = r[k].as_str().unwrap_or("");
                    if r[format!("{k}_encoding")] == json!("base64") {
                        B64.decode(text)
                            .map_err(|e| err("protocol", format!("exec {k}: {e}")))
                    } else {
                        Ok(text.as_bytes().to_vec())
                    }
                };
                Ok(json!({
                    "exit_code": r["exit_code"].as_i64().unwrap_or(-1),
                    "stdout_b64": b64(&bytes("stdout")?),
                    "stderr_b64": b64(&bytes("stderr")?),
                }))
            }
            "job_start" => {
                let r = self
                    .tool(
                        "job_start",
                        args(vec![
                            ("scope", json!(str_arg(req, "scope")?)),
                            ("id", json!(str_arg(req, "job")?)),
                            ("command", json!(command_line(req)?)),
                            ("cwd", json!(opt_str(req, "cwd")?)),
                            ("label", json!(opt_str(req, "label")?)),
                        ]),
                    )
                    .await?;
                Ok(json!({ "pgid": r["pgid"] }))
            }
            "job_list" => {
                let r = self
                    .tool(
                        "job_list",
                        args(vec![("scope", json!(str_arg(req, "scope")?))]),
                    )
                    .await?;
                Ok(json!({ "jobs": r["jobs"] }))
            }
            "job_read" => {
                let r = self
                    .tool(
                        "job_output",
                        args(vec![
                            ("scope", json!(str_arg(req, "scope")?)),
                            ("id", json!(str_arg(req, "job")?)),
                            ("offset", json!(u64_arg(req, "offset")?)),
                            ("max", json!(u64_arg(req, "max")?)),
                            ("encoding", json!("base64")),
                        ]),
                    )
                    .await?;
                Ok(json!({
                    "state": r["state"], "code": r["code"], "offset": r["offset"], "end": r["end"],
                    "data_b64": r["content"],
                }))
            }
            "job_attach" => self.job_attach(req).await,
            "job_kill" => {
                let r = self
                    .tool(
                        "job_kill",
                        args(vec![
                            ("scope", json!(str_arg(req, "scope")?)),
                            ("ids", json!(strings(req, "jobs")?)),
                            ("remove", json!(flag(req, "remove")?)),
                        ]),
                    )
                    .await?;
                Ok(json!({ "survivors": r["survivors"] }))
            }
            "job_remove" => {
                let scope = str_arg(req, "scope")?;
                let wanted = strings(req, "jobs")?;
                let gone: Vec<String> = self
                    .listed(scope)
                    .await?
                    .into_iter()
                    .filter(|j| matches!(j["state"].as_str(), Some("ended" | "lost" | "gone")))
                    .filter_map(|j| j["id"].as_str().map(str::to_string))
                    .filter(|id| wanted.contains(id))
                    .collect();
                if !gone.is_empty() {
                    self.kill_all(scope, gone).await?;
                }
                Ok(json!({}))
            }
            "job_wipe" => {
                let scope = str_arg(req, "scope")?;
                let all: Vec<String> = self
                    .listed(scope)
                    .await?
                    .iter()
                    .filter_map(|j| j["id"].as_str().map(str::to_string))
                    .collect();
                if all.is_empty() {
                    return Ok(json!({ "survivors": [] }));
                }
                let survivors = self.kill_all(scope, all).await?;
                let details: Vec<Value> = survivors.iter().map(|s| s["detail"].clone()).collect();
                Ok(json!({ "survivors": details }))
            }
            "stat" => {
                let r = self
                    .tool("stat", args(vec![("path", json!(str_arg(req, "path")?))]))
                    .await?;
                if r["exists"] == json!(false) {
                    return Ok(json!({ "exists": false }));
                }
                Ok(json!({
                    "exists": true, "kind": r["kind"], "size": r["size"],
                    "mtime": r["mtime"], "writable": r["writable"],
                }))
            }
            "read" => self.read(req).await,
            "write" => {
                let path = str_arg(req, "path")?;
                bytes_arg(req, "data_b64")?;
                self.tool(
                    "write_file",
                    args(vec![
                        ("path", json!(path)),
                        ("content_base64", json!(str_arg(req, "data_b64")?)),
                        // The contract's write does not make parents; the tool's default does.
                        ("create_parents", json!(false)),
                    ]),
                )
                .await?;
                Ok(json!({}))
            }
            "upload" => {
                let tree = trees::Tree::build_in(Some(self.root.path()), req)?;
                let rel = tree.dir();
                let rel = rel
                    .strip_prefix(self.root.path())
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                let only = match &req["sources"] {
                    Value::Null => Value::Null,
                    _ => json!(strings(req, "sources")?),
                };
                let r = self
                    .tool(
                        "upload",
                        args(vec![
                            ("local", json!(rel)),
                            ("remote_dir", json!(str_arg(req, "path")?)),
                            ("only", only),
                        ]),
                    )
                    .await?;
                Ok(counts(&r))
            }
            "download" => self.download(&req["prefill"], str_arg(req, "path")?).await,
            "extract" => self.extract(req).await,
            "exec_stream" => self.exec_stream(req).await,
            "write_if_unchanged" => {
                let path = str_arg(req, "path")?;
                bytes_arg(req, "data_b64")?;
                // `expected_mtime` is required; null is "the file must not exist".
                let guard = match req.get("expected_mtime") {
                    None => {
                        return Err(err(
                            "invalid_request",
                            "`expected_mtime` is required (null means the file must not exist)",
                        ));
                    }
                    Some(Value::Null) => ("create_only", json!(true)),
                    Some(v) => match v.as_f64() {
                        Some(s) if s.is_finite() && s >= 0.0 => ("if_unchanged_since", json!(s)),
                        _ => {
                            return Err(err(
                                "invalid_request",
                                "`expected_mtime` must be a non-negative number or null",
                            ));
                        }
                    },
                };
                self.tool(
                    "write_file",
                    args(vec![
                        ("path", json!(path)),
                        ("content_base64", json!(str_arg(req, "data_b64")?)),
                        ("create_parents", json!(false)),
                        guard,
                    ]),
                )
                .await?;
                Ok(json!({}))
            }
            "mkdir" => {
                self.tool("mkdir", args(vec![("path", json!(str_arg(req, "path")?))]))
                    .await?;
                Ok(json!({}))
            }
            "list" => {
                let r = self
                    .tool(
                        "list_dir",
                        args(vec![
                            ("path", json!(str_arg(req, "path")?)),
                            ("include_hidden", json!(flag(req, "include_hidden")?)),
                            ("limit", json!(u64_arg(req, "limit")?.unwrap_or(10_000))),
                        ]),
                    )
                    .await?;
                Ok(json!({ "entries": r["entries"] }))
            }
            "search" => {
                let glob = opt_str(req, "glob")?;
                let r = self
                    .tool(
                        "search",
                        args(vec![
                            ("pattern", json!(str_arg(req, "pattern")?)),
                            ("path", json!(str_arg(req, "root")?)),
                            ("literal", json!(flag(req, "literal")?)),
                            ("ignore_case", json!(flag(req, "ignore_case")?)),
                            ("glob", json!(glob)),
                            (
                                "glob_exclude",
                                json!(glob.map(|_| flag(req, "glob_exclude")).transpose()?),
                            ),
                            ("limit", json!(u64_arg(req, "limit")?)),
                            ("before", json!(u64_arg(req, "before")?)),
                            ("after", json!(u64_arg(req, "after")?)),
                        ]),
                    )
                    .await?;
                Ok(json!({ "hits": r["hits"], "truncated": r["truncated"], "note": r["note"] }))
            }
            _ => Err(err("unsupported", op)),
        }
    }

    /// `download` into a fresh `local` below this server's root, after building `prefill` in it.
    async fn download(&self, prefill: &Value, remote_path: &str) -> Out {
        let layout = trees::Layout::new_in(Some(self.root.path()), prefill)?;
        let rel = layout
            .local
            .strip_prefix(self.root.path())
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let r = self
            .tool(
                "download",
                args(vec![
                    ("remote_path", json!(remote_path)),
                    ("local_dir", json!(rel)),
                ]),
            )
            .await
            .map(|r| counts(&r));
        layout.finish(r)
    }

    /// `exec_stream` as an agent would watch a command: a job, read from an offset until it ends,
    /// killed at the deadline, removed at the end. The command's stderr goes to its own file in the
    /// sandbox because a job's log is one stream.
    async fn exec_stream(&self, req: &Value) -> Out {
        let scope = format!("drv-stream-{}", std::process::id());
        let n = self.jobs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = format!("s{n}");
        let errfile = format!("/tmp/.{scope}-{id}.err");
        let line = format!("exec 2>{} && {}", shell_quote(&errfile), command_line(req)?);
        self.tool(
            "job_start",
            args(vec![
                ("scope", json!(scope)),
                ("id", json!(id)),
                ("command", json!(line)),
                ("cwd", json!(opt_str(req, "cwd")?)),
            ]),
        )
        .await?;
        let deadline = u64_arg(req, "timeout_ms")?.map(Duration::from_millis);
        let followed = self.follow(&scope, &id, 0, deadline).await;
        // Whatever happened, the job and its stderr file are removed.
        let stderr = self.read(&json!({ "path": errfile })).await;
        let _ = self.kill_all(&scope, vec![id]).await;
        let _ = self
            .tool(
                "exec",
                args(vec![(
                    "command",
                    json!(format!("rm -f -- {}", shell_quote(&errfile))),
                )]),
            )
            .await;
        let f = followed?;
        let stderr = stderr.map(|v| v["data_b64"].clone()).unwrap_or(json!(""));
        let mut out = json!({ "stdout_b64": b64(&f.output), "stderr_b64": stderr });
        match f.code {
            _ if f.timed_out => out["timed_out"] = json!(true),
            // A job's status is 128 + N for a command killed by signal N, and an exit of 129 to
            // 159 looks the same: the signal reading is the shell's own.
            Some(c @ 129..=159) => out["signal"] = json!(c - 128),
            c => out["exit_code"] = json!(c.unwrap_or(-1)),
        }
        Ok(out)
    }

    /// `extract`: write the archive into a private sandbox whose `tar` prints it, and `download`
    /// from there.
    async fn extract(&self, req: &Value) -> Out {
        let archive = bytes_arg(req, "tar_b64")?;
        let p = self.private.get_or_try_init(PrivateSandbox::start).await?;
        p.mcp
            .tool(
                "write_file",
                args(vec![
                    ("path", json!(p.archive.to_string_lossy())),
                    ("content_base64", json!(b64(&archive))),
                ]),
            )
            .await?;
        // The download script insists the path exists; the `tar` there ignores it.
        p.mcp
            .download(&req["prefill"], &p.archive.to_string_lossy())
            .await
    }

    /// `read` with the contract's `max` (default: to the end of the file), in as many `read_file`
    /// calls as it takes, each asking for bytes.
    async fn read(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        let mut offset = u64_arg(req, "offset")?.unwrap_or(0);
        let mut remaining = u64_arg(req, "max")?;
        let mut data = Vec::new();
        loop {
            let r = self
                .tool(
                    "read_file",
                    args(vec![
                        ("path", json!(path)),
                        ("offset", json!(offset)),
                        ("max", json!(remaining)),
                        ("encoding", json!("base64")),
                    ]),
                )
                .await?;
            let chunk = B64
                .decode(r["content"].as_str().unwrap_or(""))
                .map_err(|e| err("protocol", format!("read_file: {e}")))?;
            offset += chunk.len() as u64;
            data.extend_from_slice(&chunk);
            if let Some(rem) = remaining.as_mut() {
                *rem = rem.saturating_sub(chunk.len() as u64);
            }
            if r["eof"] == json!(true) || chunk.is_empty() || remaining == Some(0) {
                break;
            }
        }
        Ok(json!({ "data_b64": b64(&data) }))
    }

    /// `job_attach`: follow the job to its end.
    async fn job_attach(&self, req: &Value) -> Out {
        let f = self
            .follow(
                str_arg(req, "scope")?,
                str_arg(req, "job")?,
                u64_arg(req, "offset")?.unwrap_or(0),
                None,
            )
            .await?;
        Ok(json!({ "exit_code": f.code, "output_b64": b64(&f.output), "gap": f.gap }))
    }

    /// Follow a job by reading its output, as the client library's polling follow does: a jump in
    /// the offset is a gap, `gone` is an error, and the job has ended when everything it wrote has
    /// been read. At `deadline` (from now) the job is killed and followed to its end.
    async fn follow(
        &self,
        scope: &str,
        id: &str,
        mut offset: u64,
        deadline: Option<Duration>,
    ) -> Result<Followed, Value> {
        let started = std::time::Instant::now();
        let (mut out, mut gap, mut timed_out) = (Vec::new(), 0u64, false);
        let mut wait = Duration::from_millis(25);
        loop {
            let r = self
                .tool(
                    "job_output",
                    args(vec![
                        ("scope", json!(scope)),
                        ("id", json!(id)),
                        ("offset", json!(offset)),
                        ("encoding", json!("base64")),
                    ]),
                )
                .await?;
            if r["state"] == json!("gone") {
                return Err(err("io", "the job is no longer in the sandbox"));
            }
            let at = r["offset"].as_u64().unwrap_or(offset);
            if at > offset {
                gap += at - offset;
                offset = at;
            }
            let chunk = B64
                .decode(r["content"].as_str().unwrap_or(""))
                .map_err(|e| err("protocol", format!("job_output: {e}")))?;
            let end = r["end"].as_u64().unwrap_or(0);
            if !chunk.is_empty() {
                offset += chunk.len() as u64;
                out.extend_from_slice(&chunk);
                wait = Duration::from_millis(25);
                if offset < end {
                    continue;
                }
            }
            match r["state"].as_str() {
                Some("ended") if offset >= end || chunk.is_empty() => {
                    return Ok(Followed {
                        output: out,
                        gap,
                        code: r["code"].as_i64(),
                        timed_out,
                    });
                }
                Some("ended") => continue,
                Some("lost") => {
                    return Err(err("io", "the job's supervisor in the sandbox is gone"));
                }
                _ => {}
            }
            let mut nap = wait;
            if let Some(d) = deadline.filter(|_| !timed_out) {
                match d.checked_sub(started.elapsed()) {
                    Some(left) => nap = nap.min(left),
                    None => {
                        timed_out = true;
                        self.tool(
                            "job_kill",
                            args(vec![("scope", json!(scope)), ("ids", json!([id]))]),
                        )
                        .await?;
                        continue;
                    }
                }
            }
            tokio::time::sleep(nap).await;
            wait = (wait * 2).min(FOLLOW_POLL_MAX);
        }
    }

    async fn listed(&self, scope: &str) -> Result<Vec<Value>, Value> {
        let r = self
            .tool("job_list", args(vec![("scope", json!(scope))]))
            .await?;
        Ok(r["jobs"].as_array().cloned().unwrap_or_default())
    }

    async fn kill_all(&self, scope: &str, ids: Vec<String>) -> Result<Vec<Value>, Value> {
        let r = self
            .tool(
                "job_kill",
                args(vec![
                    ("scope", json!(scope)),
                    ("ids", json!(ids)),
                    ("remove", json!(true)),
                ]),
            )
            .await?;
        Ok(r["survivors"].as_array().cloned().unwrap_or_default())
    }
}

#[tokio::main]
async fn main() {
    // Connect on the first request that reaches the sandbox; `hello` never does.
    let mut mcp: Option<Result<Mcp, Value>> = None;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut out = std::io::stdout().lock();
    while let Some(line) = lines.next_line().await.expect("stdin") {
        if line.trim().is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Value>(&line) {
            Err(e) => json!({"id": null, "error": err("invalid_request", e.to_string())}),
            Ok(req) => {
                let id = req["id"].clone();
                let op = req["op"].as_str().unwrap_or("").to_string();
                let result = if op == "hello" {
                    Ok(json!({ "driver": DRIVER, "protocol": 1, "ops": OPS }))
                } else if !OPS.contains(&op.as_str()) {
                    Err(err("unsupported", "no such op"))
                } else {
                    if mcp.is_none() {
                        mcp = Some(Mcp::start().await);
                    }
                    match mcp.as_ref().unwrap() {
                        Ok(m) => m.handle(&op, &req).await,
                        Err(e) => Err(e.clone()),
                    }
                };
                match result {
                    Ok(v) => json!({ "id": id, "ok": v }),
                    Err(e) => json!({ "id": id, "error": e }),
                }
            }
        };
        writeln!(out, "{resp}").unwrap();
        out.flush().unwrap();
    }
}
