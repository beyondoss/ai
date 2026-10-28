//! The MCP server's conformance driver (`sdk/conformance/DRIVER.md`).
//!
//! It reads the contract's operations on stdin and implements each one by calling `sandbox-mcp`'s
//! tools over a real MCP session: it spawns the `sandbox-mcp` binary (next to this one, or
//! `$SANDBOX_MCP_BIN`) on stdio with the `CONFORMANCE_*` environment translated to `SANDBOX_*`, and
//! speaks to it with rmcp's client. It has no sandbox code of its own, so what it proves is the
//! server: the tools' schemas, their errors, and what a model would get back.
//!
//! The one op with no tool is answered `unsupported`, with the reason, and is left out of `hello`'s
//! list so the harness counts the checks that need it as unsupported, never as passes:
//! `exec_stream` (a tool result is not streamed).
//!
//! Two ops are composed of tool calls because the contract's operation is a loop or a filter over
//! them: `job_attach` polls `job_output` to the job's end, `job_remove` and `job_wipe` list jobs
//! and `job_kill` the right ones with `remove`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Write as _;
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
const DRIVER: &str = "sandbox-mcp";

/// One op's outcome in the contract's terms.
type Out = Result<Value, Value>;

const OPS: &[&str] = &[
    "hello",
    "capabilities",
    "exec",
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

fn unsupported_reason(op: &str) -> Option<&'static str> {
    match op {
        "exec_stream" => Some(
            "sandbox-mcp has no streaming tool: an MCP tool result arrives whole, so there is no live \
             output to compare (use exec, or job_start and job_output)",
        ),
        "extract" => Some(
            "sandbox-mcp has no tool that extracts an archive it is handed: the local half of a \
             download is inside its download tool",
        ),
        _ => None,
    }
}

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

struct Mcp {
    // Held so the server (a child process) lives as long as the driver.
    _service: RunningService<RoleClient, ()>,
    peer: Peer<RoleClient>,
    /// The server's `SANDBOX_LOCAL_ROOT`: the trees of upload and download live below it.
    root: tempfile::TempDir,
}

impl Mcp {
    async fn start() -> Result<Self, Value> {
        let url = std::env::var("CONFORMANCE_URL").unwrap_or_default();
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
        cmd.env("SANDBOX_URL", &url);
        if let Some(b) = std::env::var("CONFORMANCE_BEARER")
            .ok()
            .filter(|b| !b.is_empty())
        {
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
            "download" => {
                let layout = trees::Layout::new_in(Some(self.root.path()), &req["prefill"])?;
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
                            ("remote_path", json!(str_arg(req, "path")?)),
                            ("local_dir", json!(rel)),
                        ]),
                    )
                    .await
                    .map(|r| counts(&r));
                layout.finish(r)
            }
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
            _ => Err(err("unsupported", unsupported_reason(op).unwrap_or(op))),
        }
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

    /// `job_attach`: follow the job to its end by reading its output, as the client library's polling
    /// follow does: a jump in the offset is a gap, `gone` is an error, and the job has ended when
    /// everything it wrote has been read.
    async fn job_attach(&self, req: &Value) -> Out {
        let scope = str_arg(req, "scope")?;
        let id = str_arg(req, "job")?;
        let mut offset = u64_arg(req, "offset")?.unwrap_or(0);
        let (mut out, mut gap) = (Vec::new(), 0u64);
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
                    return Ok(json!({
                        "exit_code": r["code"], "output_b64": b64(&out), "gap": gap,
                    }));
                }
                Some("ended") => continue,
                Some("lost") => {
                    return Err(err("io", "the job's supervisor in the sandbox is gone"));
                }
                _ => {}
            }
            tokio::time::sleep(wait).await;
            wait = (wait * 2).min(Duration::from_secs(2));
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
                    Err(err(
                        "unsupported",
                        unsupported_reason(&op).unwrap_or("no such op"),
                    ))
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
