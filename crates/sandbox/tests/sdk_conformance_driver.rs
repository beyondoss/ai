//! The reference driver for `sdk/conformance`: the agent's own `HttpExecRunner` + `ShellFs`, speaking
//! the driver contract of `sdk/conformance/DRIVER.md`.
//!
//! Every op below is the thinnest possible mapping from the contract's request to one call of the
//! existing code, and from its result or error back to the contract's JSON. The error vocabulary
//! lives in [`classify`], which is the one place the spec's "Errors" table is implemented.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use beyond_agent::exec_endpoint::HttpExecRunner;
use beyond_agent::exec_endpoint::jobs::{
    JobEvent, JobId, JobScope, JobSpec, JobStatus, SandboxJobs,
};
use beyond_agent::tools::exec::CommandRunner;
use beyond_agent::tools::fs::shell::{Capabilities, SearchEngine, ShellFs};
use beyond_agent::tools::fs::{FileKind, FsBackend, FsError, SearchQuery};
use serde_json::{Value, json};

pub const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

/// What a scenario's `connect` and the backend profile hand the driver, the way the contract's
/// environment variables do.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub url: String,
    pub bearer: Option<String>,
    /// `CONFORMANCE_CAPS`: overrides of what the probes found.
    pub caps: CapsOverride,
    /// Where a baseline job lives, standing in for `/tmp/beyond-jobs`.
    pub jobs_root: PathBuf,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CapsOverride {
    pub stdin: Option<bool>,
    pub find_printf: Option<bool>,
    pub search_engine: Option<&'static str>,
}

pub struct Driver {
    runner: Arc<HttpExecRunner>,
    jobs: Arc<dyn SandboxJobs>,
    fs: ShellFs,
    jobs_root: PathBuf,
    /// The error every op returns when the endpoint could not be reached or refused the first
    /// request: probes that fail at that level are not capabilities, they are an outage.
    connect_error: Option<Value>,
    caps: Capabilities,
    streams: bool,
    native_jobs: bool,
}

/// One op's outcome in the contract's terms.
type Out = Result<Value, Value>;

fn err(kind: &str, detail: impl Into<String>) -> Value {
    json!({ "kind": kind, "detail": detail.into() })
}

fn io_err(e: &std::io::Error) -> Value {
    classify(&e.to_string(), Some(e.kind()))
}

fn fs_err(e: &FsError) -> Value {
    match e {
        FsError::InvalidQuery(m) => err("invalid_query", m.clone()),
        FsError::Backend(m) => classify(m, None),
    }
}

/// The contract's error vocabulary, from the text the existing code produces. See FILES-OVER-EXEC.md
/// "Errors" for the table this implements.
fn classify(msg: &str, kind: Option<std::io::ErrorKind>) -> Value {
    if kind == Some(std::io::ErrorKind::TimedOut) || msg.contains("timed out") {
        return err("timeout", msg);
    }
    if kind == Some(std::io::ErrorKind::Unsupported) {
        return err("not_offered", msg);
    }
    if msg.contains("returned 401") || msg.contains("returned 403") {
        return err("auth", msg);
    }
    if msg.contains("exceeded the") && msg.contains("-byte cap") {
        return err("too_large", msg);
    }
    if msg.contains("returned 422") && msg.contains("cap") {
        return err("too_large", msg);
    }
    if msg.contains("unreachable") {
        return err("transport", msg);
    }
    if msg.contains("unparseable JSON") {
        return err("protocol", msg);
    }
    if msg.contains("exec endpoint") && msg.contains(" returned ") {
        return err("endpoint", msg);
    }
    if msg.contains("exec endpoint") || msg.contains("job stream") {
        return err("transport", msg);
    }
    // A command that ran and failed: classify by what it said.
    let m = msg.to_ascii_lowercase();
    let kind = if m.contains("no such file or directory")
        || m.contains("directory nonexistent")
        || m.contains("nonexistent directory")
    {
        "not_found"
    } else if m.contains("is a directory") {
        "is_directory"
    } else if m.contains("not a directory") {
        "not_a_directory"
    } else if m.contains("permission denied") {
        "permission_denied"
    } else {
        "io"
    };
    err(kind, msg)
}

fn b64(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

fn str_arg<'a>(req: &'a Value, key: &str) -> Result<&'a str, Value> {
    req[key]
        .as_str()
        .ok_or_else(|| err("invalid_request", format!("`{key}` must be a string")))
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

fn bytes_arg(req: &Value, key: &str) -> Result<Vec<u8>, Value> {
    B64.decode(str_arg(req, key)?)
        .map_err(|e| err("invalid_request", format!("`{key}` is not base64: {e}")))
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

fn opt_str<'a>(req: &'a Value, key: &str) -> Result<Option<&'a str>, Value> {
    match &req[key] {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s)),
        _ => Err(err("invalid_request", format!("`{key}` must be a string"))),
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

fn kind_name(k: FileKind) -> &'static str {
    match k {
        FileKind::File => "file",
        FileKind::Dir => "dir",
        FileKind::Other => "other",
    }
}

fn epoch_secs(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn state_of(s: JobStatus) -> (&'static str, Option<i32>) {
    match s {
        JobStatus::Running => ("running", None),
        JobStatus::ShellExited(c) => ("exited", Some(c)),
        JobStatus::Ended(c) => ("ended", Some(c)),
        JobStatus::Lost => ("lost", None),
        JobStatus::Gone => ("gone", None),
    }
}

impl Driver {
    pub async fn connect(cfg: &Config) -> Self {
        let mut runner = HttpExecRunner::new(&cfg.url).expect("exec url");
        if let Some(token) = &cfg.bearer {
            runner = runner.with_header("Authorization", format!("Bearer {token}"));
        }
        let runner = Arc::new(runner);
        // ShellFs itself decides: a probe whose request failed (unreachable, `401`) fails connect, and
        // that error is what every op reports. The exec/stream/jobs paths then fail the same way
        // on their own first request.
        let (mut caps, connect_error) = match Capabilities::probe(runner.as_ref()).await {
            Ok(caps) => (caps, None),
            Err(e) => (Capabilities::default(), Some(io_err(&e))),
        };
        if let Some(s) = cfg.caps.stdin {
            caps.stdin = s;
        }
        if let Some(f) = cfg.caps.find_printf {
            caps.find_printf = f;
        }
        match cfg.caps.search_engine {
            Some("rg") => {
                caps.rg = true;
                caps.grep_null = true;
            }
            Some("grep") => {
                caps.rg = false;
                caps.grep_null = true;
            }
            Some("busybox-grep") => {
                caps.rg = false;
                caps.grep_null = false;
            }
            _ => {}
        }
        let fs = ShellFs::with_capabilities(runner.clone(), caps);
        let streams = connect_error.is_none() && runner.streams().await;
        let native_jobs = connect_error.is_none() && runner.offers_jobs().await;
        let jobs = runner.clone().sandbox_jobs().expect("http jobs");
        Self {
            runner,
            jobs,
            fs,
            jobs_root: cfg.jobs_root.clone(),
            connect_error,
            caps,
            streams,
            native_jobs,
        }
    }

    fn scope(&self, req: &Value) -> Result<JobScope, Value> {
        let name = str_arg(req, "scope")?;
        Ok(JobScope::under(self.jobs_root.to_string_lossy(), name))
    }

    fn job_id(req: &Value, key: &str) -> Result<JobId, Value> {
        let s = str_arg(req, key)?;
        JobId::parse(s).ok_or_else(|| err("invalid_request", format!("bad job id {s:?}")))
    }

    pub fn ops() -> Vec<&'static str> {
        vec![
            "hello",
            "capabilities",
            "exec",
            "exec_stream",
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
        ]
    }

    /// One request line in, one response line out.
    pub async fn handle(&self, req: &Value) -> Value {
        let id = req["id"].clone();
        let op = req["op"].as_str().unwrap_or("");
        let out = if op == "hello" || op == "capabilities" {
            self.meta(op).await
        } else if !Self::ops().contains(&op) {
            Err(err("unsupported", format!("no such op {op:?}")))
        } else if let Some(e) = &self.connect_error {
            Err(e.clone())
        } else {
            self.dispatch(op, req).await
        };
        match out {
            Ok(v) => json!({ "id": id, "ok": v }),
            Err(e) => json!({ "id": id, "error": e }),
        }
    }

    async fn meta(&self, op: &str) -> Out {
        if op == "hello" {
            return Ok(json!({
                "driver": "agent-reference",
                "protocol": 1,
                "ops": Self::ops(),
            }));
        }
        if let Some(e) = &self.connect_error {
            return Err(e.clone());
        }
        let engine = match self.caps.search_engine() {
            SearchEngine::Ripgrep => "rg",
            SearchEngine::PosixGrep => "grep",
            SearchEngine::BusyboxGrep => "busybox-grep",
        };
        let jobs = if self.native_jobs {
            "native"
        } else if self.jobs.available().await {
            "baseline"
        } else {
            "none"
        };
        Ok(json!({
            "stdin": self.caps.stdin,
            "stream": self.streams,
            "jobs": jobs,
            "search_engine": engine,
            "find_printf": self.caps.find_printf,
            "read_window": !self.caps.no_read_window,
        }))
    }

    async fn dispatch(&self, op: &str, req: &Value) -> Out {
        match op {
            "exec" => self.exec(req).await,
            "exec_stream" => self.exec_stream(req).await,
            "job_start" => self.job_start(req).await,
            "job_list" => self.job_list(req).await,
            "job_read" => self.job_read(req).await,
            "job_attach" => self.job_attach(req).await,
            "job_kill" => self.job_kill(req).await,
            "job_remove" => self.job_remove(req).await,
            "job_wipe" => self.job_wipe(req).await,
            "stat" => self.stat(req).await,
            "read" => self.read(req).await,
            "write" => self.write(req).await,
            "write_if_unchanged" => self.write_if_unchanged(req).await,
            "mkdir" => self.mkdir(req).await,
            "list" => self.list(req).await,
            "search" => self.search(req).await,
            _ => Err(err("unsupported", op)),
        }
    }

    // ---- layer 1: raw exec -------------------------------------------------------------------

    async fn exec(&self, req: &Value) -> Out {
        let command = str_arg(req, "command")?;
        let args = strings(req, "args")?;
        let cwd = opt_str(req, "cwd")?;
        let timeout = Duration::from_millis(u64_arg(req, "timeout_ms")?.unwrap_or(120_000));
        let r = if !req["stdin_b64"].is_null() {
            let stdin = bytes_arg(req, "stdin_b64")?;
            self.runner
                .run_with_stdin(command, &args, cwd, timeout, &stdin)
                .await
        } else {
            self.runner.run(command, &args, cwd, timeout).await
        }
        .map_err(|e| io_err(&e))?;
        Ok(json!({
            "exit_code": r.code.unwrap_or(-1),
            "stdout_b64": b64(r.stdout.as_bytes()),
            "stderr_b64": b64(r.stderr.as_bytes()),
        }))
    }

    async fn exec_stream(&self, req: &Value) -> Out {
        let command = str_arg(req, "command")?;
        let args = strings(req, "args")?;
        let cwd = opt_str(req, "cwd")?;
        let out = Mutex::new(Vec::new());
        let errs = Mutex::new(Vec::new());
        // `run_attached` keeps the two streams apart but has no deadline of its own; a request that
        // carries `timeout_ms` goes through `run_streaming`, which has one and merges the streams.
        let result = if let Some(ms) = u64_arg(req, "timeout_ms")? {
            if !self.streams {
                return Err(err(
                    "not_offered",
                    "this exec endpoint does not stream (protocol v1.1)",
                ));
            }
            self.runner
                .run_streaming(command, &args, cwd, Duration::from_millis(ms), &|b| {
                    out.lock().unwrap().extend_from_slice(b)
                })
                .await
        } else {
            self.runner
                .run_attached(
                    command,
                    &args,
                    cwd,
                    &|b| out.lock().unwrap().extend_from_slice(b),
                    &|b| errs.lock().unwrap().extend_from_slice(b),
                )
                .await
        }
        .map_err(|e| io_err(&e))?;
        let mut v = json!({
            "stdout_b64": b64(&out.lock().unwrap()),
            "stderr_b64": b64(&errs.lock().unwrap()),
        });
        if result.timed_out {
            v["timed_out"] = json!(true);
        } else if let Some(s) = result.signal {
            v["signal"] = json!(s);
        } else {
            v["exit_code"] = json!(result.code.unwrap_or(-1));
        }
        Ok(v)
    }

    // ---- layer 1: jobs -----------------------------------------------------------------------

    async fn job_start(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let spec = JobSpec {
            program: str_arg(req, "command")?.to_string(),
            args: strings(req, "args")?,
            cwd: opt_str(req, "cwd")?.map(str::to_string),
            label: opt_str(req, "label")?.unwrap_or("").to_string(),
        };
        let pgid = self
            .jobs
            .start_kept(&scope, &id, &spec)
            .await
            .map_err(|e| io_err(&e))?;
        Ok(json!({ "pgid": pgid }))
    }

    async fn job_list(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let jobs = self.jobs.list(&scope).await.map_err(|e| io_err(&e))?;
        let jobs: Vec<Value> = jobs
            .iter()
            .map(|j| {
                let (state, code) = state_of(j.status);
                json!({
                    "id": j.id.as_str(), "state": state, "code": code, "kept": j.kept,
                    "num": j.num, "kind": j.kind, "seen": j.seen,
                    "started": j.started, "ended": j.ended, "label": j.label,
                })
            })
            .collect();
        Ok(json!({ "jobs": jobs }))
    }

    async fn job_read(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let offset = u64_arg(req, "offset")?.unwrap_or(0);
        let max = u64_arg(req, "max")?.unwrap_or(128 * 1024) as usize;
        let c = self
            .jobs
            .read(&scope, &id, offset, max)
            .await
            .map_err(|e| io_err(&e))?;
        let (state, code) = state_of(c.status);
        Ok(json!({
            "state": state, "code": code, "offset": c.offset, "end": c.end,
            "data_b64": b64(&c.data),
        }))
    }

    async fn job_attach(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let offset = u64_arg(req, "offset")?.unwrap_or(0);
        let out = Mutex::new(Vec::new());
        let gaps = Mutex::new(0u64);
        let code = self
            .jobs
            .follow(&scope, &id, offset, &|ev: JobEvent<'_>| match ev {
                JobEvent::Output(b) => out.lock().unwrap().extend_from_slice(b),
                JobEvent::Gap(n) => *gaps.lock().unwrap() += n,
                JobEvent::Started(_) | JobEvent::ShellExit(_) => {}
            })
            .await
            .map_err(|e| io_err(&e))?;
        Ok(json!({
            "exit_code": code,
            "output_b64": b64(&out.lock().unwrap()),
            "gap": *gaps.lock().unwrap(),
        }))
    }

    async fn job_kill(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let ids: Vec<JobId> = strings(req, "jobs")?
            .iter()
            .map(|s| JobId::parse(s).ok_or_else(|| err("invalid_request", "bad job id")))
            .collect::<Result<_, _>>()?;
        let remove = flag(req, "remove")?;
        let survivors = self
            .jobs
            .kill(&scope, &ids, remove)
            .await
            .map_err(|e| io_err(&e))?;
        let survivors: Vec<Value> = survivors
            .iter()
            .map(|(id, d)| json!({ "id": id.as_str(), "detail": d }))
            .collect();
        Ok(json!({ "survivors": survivors }))
    }

    async fn job_remove(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let ids: Vec<JobId> = strings(req, "jobs")?
            .iter()
            .filter_map(|s| JobId::parse(s))
            .collect();
        self.jobs
            .remove(&scope, &ids)
            .await
            .map_err(|e| io_err(&e))?;
        Ok(json!({}))
    }

    async fn job_wipe(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let survivors = self.jobs.wipe(&scope).await.map_err(|e| io_err(&e))?;
        Ok(json!({ "survivors": survivors }))
    }

    // ---- layer 2: files ----------------------------------------------------------------------

    async fn stat(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        match self
            .fs
            .stat(Path::new(path))
            .await
            .map_err(|e| fs_err(&e))?
        {
            None => Ok(json!({ "exists": false })),
            Some(m) => Ok(json!({
                "exists": true,
                "kind": kind_name(m.kind),
                "size": m.len,
                "mtime": m.mtime.map(epoch_secs),
                "writable": m.writable,
            })),
        }
    }

    async fn read(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        let offset = u64_arg(req, "offset")?.unwrap_or(0);
        let max = u64_arg(req, "max")?.map_or(usize::MAX, |m| m as usize);
        let bytes = self
            .fs
            .read_bytes(Path::new(path), offset, max)
            .await
            .map_err(|e| fs_err(&e))?;
        Ok(json!({ "data_b64": b64(&bytes) }))
    }

    async fn write(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        let data = bytes_arg(req, "data_b64")?;
        self.fs
            .write_bytes(Path::new(path), data)
            .await
            .map_err(|e| fs_err(&e))?;
        Ok(json!({}))
    }

    async fn write_if_unchanged(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        let data = bytes_arg(req, "data_b64")?;
        if req.get("expected_mtime").is_none() {
            return Err(err(
                "invalid_request",
                "`expected_mtime` is required (null means the file must not exist)",
            ));
        }
        let expected = u64_arg(req, "expected_mtime")?.map(|s| UNIX_EPOCH + Duration::from_secs(s));
        let wrote = self
            .fs
            .write_if_unchanged(Path::new(path), data, expected)
            .await
            .map_err(|e| fs_err(&e))?;
        if wrote {
            Ok(json!({}))
        } else {
            Err(err("conflict", "the file changed since it was read"))
        }
    }

    async fn mkdir(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        self.fs
            .create_dir_all(Path::new(path))
            .await
            .map_err(|e| fs_err(&e))?;
        Ok(json!({}))
    }

    async fn list(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        let include_hidden = flag(req, "include_hidden")?;
        let limit = u64_arg(req, "limit")?.unwrap_or(10_000) as usize;
        let entries = self
            .fs
            .list_dir(Path::new(path), limit, include_hidden)
            .await
            .map_err(|e| fs_err(&e))?;
        let entries: Vec<Value> = entries
            .iter()
            .map(|e| json!({ "name": e.name, "kind": kind_name(e.kind), "size": e.len }))
            .collect();
        Ok(json!({ "entries": entries }))
    }

    async fn search(&self, req: &Value) -> Out {
        let glob = match opt_str(req, "glob")? {
            Some(g) => Some((g.to_string(), flag(req, "glob_exclude")?)),
            None => None,
        };
        let q = SearchQuery {
            pattern: str_arg(req, "pattern")?.to_string(),
            literal: flag(req, "literal")?,
            ignore_case: flag(req, "ignore_case")?,
            glob,
            root: PathBuf::from(str_arg(req, "root")?),
            limit: u64_arg(req, "limit")?.unwrap_or(100) as usize,
            before: u64_arg(req, "before")?.unwrap_or(0) as usize,
            after: u64_arg(req, "after")?.unwrap_or(0) as usize,
        };
        let o = self.fs.search(&q).await.map_err(|e| fs_err(&e))?;
        let hits: Vec<Value> = o
            .hits
            .iter()
            .map(|h| {
                json!({
                    "path": h.path.to_string_lossy(), "line": h.line,
                    "text": h.text, "is_match": h.is_match,
                })
            })
            .collect();
        Ok(json!({ "hits": hits, "truncated": o.truncated, "note": o.first_error }))
    }
}
