//! The reference driver for `sdk/conformance`: a `beyond_sandbox::Sandbox` over the HTTP exec client,
//! speaking the driver contract of `sdk/conformance/DRIVER.md`. Exec, files, jobs and bulk copy go
//! through the `Sandbox`; only what it has no method for (the two output streams of a stream kept
//! apart, a bulk copy with a chosen chunk size) uses its raw exec.
//!
//! Every op below is the thinnest possible mapping from the contract's request to one call of the
//! existing code, and from its result or error back to the contract's JSON. The error vocabulary
//! lives in [`classify`], which is the one place the spec's "Errors" table is implemented.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use beyond_sandbox::Sandbox;
use beyond_sandbox::exec::CommandRunner;
use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_sandbox::exec_endpoint::jobs::{
    JobEnd, JobError, JobEvent, JobId, JobScope, JobSpec, JobStatus, PtyInfo, PtySize, Stream,
};
use beyond_sandbox::exec_endpoint::jobs_native::NativeJobs;
use beyond_sandbox::fs::shell::{Capabilities, SearchEngine};
use beyond_sandbox::fs::{FileKind, FsBackend, FsError, SearchQuery};
use sandbox_conformance::trees::{Layout, Tree};
use serde_json::{Value, json};

pub const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

/// What a scenario's `connect` and the backend profile hand the driver, the way the contract's
/// environment variables do.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub url: String,
    pub bearer: Option<String>,
    /// `CONFORMANCE_TOKEN_CMD`: a shell command whose output is the bearer, run again after a `401`.
    /// It takes the place of `bearer`.
    pub token_cmd: Option<String>,
    /// `CONFORMANCE_CAPS`: overrides of what the probes found.
    pub caps: CapsOverride,
    /// Where a job lives when the jobs run over plain commands, standing in for `/tmp/beyond-jobs`.
    pub jobs_root: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CapsOverride {
    pub find_printf: Option<bool>,
    pub search_engine: Option<&'static str>,
}

pub struct Driver {
    sb: Sandbox,
    /// The HTTP runner under `sb`, when it is one: the jobs ops that take a grace or a heartbeat go
    /// to its [`NativeJobs`] directly (the `Jobs` trait carries neither).
    http: Option<Arc<HttpExecRunner>>,
    jobs_root: Option<PathBuf>,
    /// Set only when `CONFORMANCE_CAPS` overrides what the probe found, which needs the probe up
    /// front: the error every op returns when that probe's request failed (an outage is not a
    /// capability). Without an override the `Sandbox` probes lazily and each op reports its own.
    connect_error: Option<Value>,
}

/// One op's outcome in the contract's terms.
type Out = Result<Value, Value>;

fn err(kind: &str, detail: impl Into<String>) -> Value {
    json!({ "kind": kind, "detail": detail.into() })
}

fn io_err(e: &std::io::Error) -> Value {
    classify(&e.to_string(), Some(e.kind()))
}

/// A job op's error: a job condition carries its `code` beside its `kind` (DRIVER.md, `job_errors`);
/// a heartbeat out of range is `invalid_request`; anything else is as [`io_err`].
fn job_err(e: &std::io::Error) -> Value {
    if let Some(j) = JobError::of(e) {
        let mut v = json!({ "kind": j.kind(), "detail": j.to_string() });
        // A resize conflict has no code of its own (DRIVER.md): only the conditions of `job_errors`.
        if !matches!(j, JobError::Conflict { .. }) {
            v["code"] = json!(j.code());
        }
        // A refused stdin write says how much of the stream was applied (`job_pty`).
        if let Some(n) = written_in(&j.to_string()) {
            v["written"] = json!(n);
        }
        return v;
    }
    if e.kind() == std::io::ErrorKind::InvalidInput && e.to_string().contains("heartbeat_seconds") {
        return err("invalid_request", e.to_string());
    }
    io_err(e)
}

/// The `written` an endpoint's 409 body carries, when the error text still has it.
fn written_in(text: &str) -> Option<u64> {
    let rest = &text[text.find("\"written\"")? + "\"written\"".len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// `pty`: `{cols, rows}` or nothing; a size out of range is `invalid_request`.
fn pty_arg(req: &Value) -> Result<Option<PtySize>, Value> {
    let Some(p) = req.get("pty").filter(|p| !p.is_null()) else {
        return Ok(None);
    };
    let dim = |k: &str| {
        p.get(k)
            .and_then(Value::as_u64)
            .ok_or_else(|| err("invalid_request", format!("`pty.{k}` must be a count")))
    };
    PtySize::new(dim("cols")?, dim("rows")?)
        .map(Some)
        .ok_or_else(|| {
            err(
                "invalid_request",
                "pty: cols and rows are integers from 1 to 1024",
            )
        })
}

fn pty_echo(mut v: Value, pty: Option<PtySize>) -> Value {
    if let Some(p) = pty {
        v["pty"] = json!({ "cols": p.cols, "rows": p.rows });
    }
    v
}

fn pty_json(p: PtyInfo) -> Value {
    json!({ "cols": p.size.cols, "rows": p.size.rows, "open": p.open })
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
    if msg.contains("returned 401")
        || msg.contains("returned 403")
        || msg.contains("token source failed")
    {
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
    if msg.contains("unparseable JSON")
        || msg.contains("not a stream")
        || msg.contains("unparseable frame")
        || msg.contains("unparseable job frame")
        || msg.contains("is not standard base64")
        || msg.contains("without a newline")
        || msg.contains("unreadable job")
    {
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

fn bulk_err(e: &beyond_sandbox::BulkError) -> Value {
    use beyond_sandbox::BulkError;
    match e {
        BulkError::InvalidRequest(m) => err("invalid_request", m.clone()),
        BulkError::Archive(m) => err("protocol", m.clone()),
        BulkError::Command(m) => classify(m, None),
        BulkError::Io(e) => io_err(e),
    }
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

/// `env`: an object of strings, or nothing.
fn env_arg(req: &Value) -> Result<Vec<(String, String)>, Value> {
    match &req["env"] {
        Value::Null => Ok(Vec::new()),
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| {
                v.as_str()
                    .map(|v| (k.clone(), v.to_string()))
                    .ok_or_else(|| err("invalid_request", "`env` must hold strings"))
            })
            .collect(),
        _ => Err(err("invalid_request", "`env` must be an object")),
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

fn epoch_secs(t: SystemTime) -> f64 {
    beyond_sandbox::fs::epoch_f64(t)
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
    /// The reference driver over the HTTP exec client: what the shared matrix runs.
    pub async fn connect(cfg: &Config) -> Self {
        let mut runner = HttpExecRunner::new(&cfg.url).expect("exec url");
        if let Some(cmd) = &cfg.token_cmd {
            runner = runner.bearer(beyond_sandbox::TokenSource::command(["sh", "-c", cmd]));
        } else if let Some(token) = &cfg.bearer {
            runner = runner.bearer(beyond_sandbox::TokenSource::fixed(token.clone()));
        }
        let runner = Arc::new(runner);
        let mut d = Self::build(runner.clone(), cfg).await;
        d.http = Some(runner);
        d
    }

    /// The same driver over any other [`CommandRunner`]: the layers above the transport, unchanged.
    pub async fn over(runner: Arc<dyn CommandRunner>, cfg: &Config) -> Self {
        Self::build(runner, cfg).await
    }

    async fn build(runner: Arc<dyn CommandRunner>, cfg: &Config) -> Self {
        let overridden = cfg.caps.find_printf.is_some() || cfg.caps.search_engine.is_some();
        let (sb, connect_error) = if overridden {
            match Capabilities::probe(&*runner).await {
                Ok(mut caps) => {
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
                    (Sandbox::with_capabilities(runner, caps), None)
                }
                Err(e) => (
                    Sandbox::with_capabilities(runner, Capabilities::default()),
                    Some(io_err(&e)),
                ),
            }
        } else {
            (Sandbox::over(runner).await.expect("a lazy connect"), None)
        };
        Self {
            sb,
            http: None,
            jobs_root: cfg.jobs_root.clone(),
            connect_error,
        }
    }

    fn jobs(&self) -> Result<&beyond_sandbox::Jobs, Value> {
        self.sb
            .jobs()
            .map_err(|_| err("unsupported", "this exec does not own jobs"))
    }

    fn scope(&self, req: &Value) -> Result<JobScope, Value> {
        let name = str_arg(req, "scope")?;
        Ok(match &self.jobs_root {
            Some(root) => JobScope::under(root.to_string_lossy(), name),
            None => JobScope::under(beyond_sandbox::exec_endpoint::jobs::JOBS_ROOT, name),
        })
    }

    fn job_id(req: &Value, key: &str) -> Result<JobId, Value> {
        let s = str_arg(req, key)?;
        JobId::parse(s).ok_or_else(|| err("invalid_request", format!("bad job id {s:?}")))
    }

    /// The ops this driver offers: the job ops only over an exec that owns jobs.
    pub fn ops(&self) -> Vec<&'static str> {
        Self::ops_with(self.sb.jobs().is_ok())
    }

    /// [`Self::ops`] for an exec that owns jobs, or not.
    pub fn ops_with(jobs: bool) -> Vec<&'static str> {
        let mut ops = vec!["hello", "capabilities", "exec", "exec_stream"];
        if jobs {
            ops.extend([
                "job_start",
                "job_start_held",
                "job_keep",
                "job_seen",
                "job_list",
                "job_read",
                "job_attach",
                "job_kill",
                "job_remove",
                "job_wipe",
                "job_write",
                "job_signal",
                "job_resize",
            ]);
        }
        ops.extend([
            "stat",
            "read",
            "write",
            "write_if_unchanged",
            "mkdir",
            "list",
            "search",
            "upload",
            "download",
            "extract",
        ]);
        ops
    }

    /// One request line in, one response line out.
    pub async fn handle(&self, req: &Value) -> Value {
        let id = req["id"].clone();
        let op = req["op"].as_str().unwrap_or("");
        let out = if op == "hello" || op == "capabilities" {
            self.meta(op).await
        } else if !self.ops().contains(&op) {
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
                "ops": self.ops(),
                "features": [
                    "frames", "unbounded_stream", "exit_signal", "exec_env", "job_streams",
                    "job_stdin", "job_signal_op", "job_errors", "job_options", "job_marks", "job_pty"
                ],
            }));
        }
        if let Some(e) = &self.connect_error {
            return Err(e.clone());
        }
        let caps = self
            .sb
            .files()
            .capabilities()
            .await
            .map_err(|e| io_err(&e))?;
        let engine = match caps.search_engine() {
            SearchEngine::Ripgrep => "rg",
            SearchEngine::PosixGrep => "grep",
            SearchEngine::BusyboxGrep => "busybox-grep",
        };
        Ok(json!({
            "search_engine": engine,
            "find_printf": caps.find_printf,
            "read_window": !caps.no_read_window,
        }))
    }

    async fn dispatch(&self, op: &str, req: &Value) -> Out {
        match op {
            "exec" => self.exec(req).await,
            "exec_stream" => self.exec_stream(req).await,
            "job_start" => self.job_start(req).await,
            "job_start_held" => self.job_start_held(req).await,
            "job_keep" => self.job_keep(req).await,
            "job_seen" => self.job_seen(req).await,
            "job_list" => self.job_list(req).await,
            "job_read" => self.job_read(req).await,
            "job_attach" => self.job_attach(req).await,
            "job_kill" => self.job_kill(req).await,
            "job_remove" => self.job_remove(req).await,
            "job_wipe" => self.job_wipe(req).await,
            "job_write" => self.job_write(req).await,
            "job_signal" => self.job_signal(req).await,
            "job_resize" => self.job_resize(req).await,
            "stat" => self.stat(req).await,
            "read" => self.read(req).await,
            "write" => self.write(req).await,
            "write_if_unchanged" => self.write_if_unchanged(req).await,
            "mkdir" => self.mkdir(req).await,
            "upload" => self.upload(req).await,
            "download" => self.download(req).await,
            "extract" => self.extract(req).await,
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
        let stdin = if req["stdin_b64"].is_null() {
            None
        } else {
            Some(bytes_arg(req, "stdin_b64")?)
        };
        let mut cmd = self.sb.exec(command, args).timeout(timeout);
        if let Some(cwd) = cwd {
            cmd = cmd.cwd(cwd);
        }
        if let Some(stdin) = stdin {
            cmd = cmd.stdin(stdin);
        }
        for (k, v) in env_arg(req)? {
            cmd = cmd.env(k, v);
        }
        let r = cmd.run().await.map_err(|e| io_err(&e))?;
        Ok(json!({
            "exit_code": r.code,
            "stdout_b64": b64(&r.stdout),
            "stderr_b64": b64(&r.stderr),
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
            let mut cmd = self
                .sb
                .exec(command, args)
                .timeout(Duration::from_millis(ms));
            if let Some(cwd) = cwd {
                cmd = cmd.cwd(cwd);
            }
            cmd.stream(&|b| out.lock().unwrap().extend_from_slice(b))
                .await
        } else {
            // The `Sandbox` has no call that keeps stdout and stderr apart: its raw exec does.
            self.sb
                .raw()
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
            env: env_arg(req)?,
            stdin_open: flag(req, "stdin_open")?,
            pty: pty_arg(req)?,
        };
        let started = self
            .jobs()?
            .start_kept(&scope, &id, &spec)
            .await
            .map_err(|e| job_err(&e))?;
        // The client refused (`not_offered`) an endpoint that did not echo `pty`, so a start that
        // came back echoed it.
        Ok(pty_echo(
            json!({ "pgid": started.pgid, "mark": started.mark }),
            spec.pty,
        ))
    }

    /// Start held, learn the pgid from the first frame, optionally `keep`, then drop the stream: a
    /// job nobody kept dies with it (wire `hold: true`).
    async fn job_start_held(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let spec = JobSpec {
            program: str_arg(req, "command")?.to_string(),
            args: strings(req, "args")?,
            cwd: opt_str(req, "cwd")?.map(str::to_string),
            label: opt_str(req, "label")?.unwrap_or("").to_string(),
            env: env_arg(req)?,
            stdin_open: flag(req, "stdin_open")?,
            pty: pty_arg(req)?,
        };
        let keep = flag(req, "keep")?;
        let beat = self.beat(req)?;
        let http = self.http()?;
        let native = NativeJobs::over(http);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = |ev: JobEvent<'_>| {
            if let JobEvent::Started(pgid, mark) = ev {
                let _ = tx.send((pgid, mark.to_string()));
            }
        };
        let held = native.run_held_beat(&scope, &id, &spec, beat, &sink);
        tokio::pin!(held);
        let (pgid, mark) = tokio::select! {
            r = &mut held => {
                return Err(match r {
                    Err(e) => job_err(&e),
                    Ok(_) => err("protocol", "the held start ended before `started`"),
                });
            }
            p = rx.recv() => p.ok_or_else(|| err("protocol", "the held start ended before `started`"))?,
        };
        if keep {
            self.jobs()?
                .keep(&scope, &id, None, "", 0)
                .await
                .map_err(|e| job_err(&e))?;
        }
        // Dropping the future closes the stream.
        Ok(pty_echo(json!({ "pgid": pgid, "mark": mark }), spec.pty))
    }

    async fn job_keep(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let num = u64_arg(req, "num")?;
        let kind = opt_str(req, "kind")?.unwrap_or("");
        let seen = u64_arg(req, "seen")?.unwrap_or(0);
        self.jobs()?
            .keep(&scope, &id, num, kind, seen)
            .await
            .map_err(|e| job_err(&e))?;
        Ok(json!({}))
    }

    async fn job_seen(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let offset = u64_arg(req, "offset")?.unwrap_or(0);
        self.jobs()?
            .seen(&scope, &id, offset)
            .await
            .map_err(|e| job_err(&e))?;
        Ok(json!({}))
    }

    fn http(&self) -> Result<&HttpExecRunner, Value> {
        self.http
            .as_deref()
            .ok_or_else(|| err("unsupported", "this exec is not the HTTP client"))
    }

    /// `heartbeat_seconds`: a count; the client refuses what the endpoint would silently replace.
    fn beat(&self, req: &Value) -> Result<Option<u32>, Value> {
        Ok(u64_arg(req, "heartbeat_seconds")?.map(|n| u32::try_from(n).unwrap_or(u32::MAX)))
    }

    async fn job_list(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let jobs = self.jobs()?.list(&scope).await.map_err(|e| io_err(&e))?;
        let jobs: Vec<Value> = jobs
            .iter()
            .map(|j| {
                let (state, code) = state_of(j.status);
                let mut entry = json!({
                    "id": j.id.as_str(), "mark": j.mark, "command": j.command, "args": j.args,
                    "output_start": j.output_start, "output_end": j.output_end, "state": state, "code": code, "kept": j.kept,
                    "num": j.num, "kind": j.kind, "seen": j.seen,
                    "started": j.started, "ended": j.ended, "label": j.label,
                });
                if let Some(s) = j.signal {
                    entry["signal"] = json!(s);
                }
                if let Some(p) = j.pty {
                    entry["pty"] = pty_json(p);
                }
                entry
            })
            .collect();
        Ok(json!({ "jobs": jobs }))
    }

    async fn job_read(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let offset = u64_arg(req, "offset")?.unwrap_or(0);
        let max = u64_arg(req, "max")?.unwrap_or(128 * 1024) as usize;
        let c = if flag(req, "gone_as_error")? {
            NativeJobs::over(self.http()?)
                .read_or_gone(&scope, &id, offset, max)
                .await
        } else {
            self.jobs()?.read(&scope, &id, offset, max).await
        }
        .map_err(|e| job_err(&e))?;
        let (state, code) = state_of(c.status);
        let mut out = json!({
            "state": state, "code": code, "offset": c.offset, "end": c.end,
            "data_b64": b64(&c.data),
        });
        if !c.stderr.is_empty() {
            out["stderr_ranges"] = c.stderr.iter().map(|&(a, n)| json!([a, n])).collect();
        }
        if let Some(s) = c.signal {
            out["signal"] = json!(s);
        }
        if let Some(p) = c.pty {
            out["pty"] = pty_json(p);
        }
        Ok(out)
    }

    async fn job_attach(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let offset = u64_arg(req, "offset")?.unwrap_or(0);
        let want_mark = opt_str(req, "mark")?;
        let output = match req.get("output") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(err("invalid_request", "`output` is a boolean")),
        };
        let out = Mutex::new(Vec::new());
        let errs = Mutex::new(Vec::new());
        let gaps = Mutex::new(0u64);
        let attached = Mutex::new(None::<(String, u64)>);
        let sink = |ev: JobEvent<'_>| match ev {
            JobEvent::Attached { mark, from } => {
                *attached.lock().unwrap() = Some((mark.to_string(), from));
            }
            JobEvent::Output(b, s) => {
                out.lock().unwrap().extend_from_slice(b);
                if s == Stream::Stderr {
                    errs.lock().unwrap().extend_from_slice(b);
                }
            }
            JobEvent::Gap(n) => *gaps.lock().unwrap() += n,
            JobEvent::Started(..) | JobEvent::ShellExit(_) => {}
        };
        let end = NativeJobs::over(self.http()?)
            .follow_opts(
                &scope,
                &id,
                offset,
                want_mark,
                output,
                self.beat(req)?,
                &sink,
            )
            .await
            .map_err(|e| job_err(&e))?;
        let (mark, from) = attached
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| err("protocol", "the attach was not answered"))?;
        let mut v = json!({
            "attached": { "mark": mark, "from": from },
            "output_b64": b64(&out.lock().unwrap()),
            "gap": *gaps.lock().unwrap(),
        });
        if !errs.lock().unwrap().is_empty() {
            v["stderr_b64"] = json!(b64(&errs.lock().unwrap()));
        }
        // As `exec_stream` ends: `signal` in place of `exit_code` for a death by signal.
        match end {
            JobEnd::Exit(c) => v["exit_code"] = json!(c),
            JobEnd::Signal(s) => v["signal"] = json!(s),
        }
        Ok(v)
    }

    async fn job_kill(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let ids: Vec<JobId> = strings(req, "jobs")?
            .iter()
            .map(|s| JobId::parse(s).ok_or_else(|| err("invalid_request", "bad job id")))
            .collect::<Result<_, _>>()?;
        let remove = flag(req, "remove")?;
        let survivors = match u64_arg(req, "grace_ms")? {
            Some(grace) => {
                NativeJobs::over(self.http()?)
                    .kill_grace(&scope, &ids, remove, grace)
                    .await
            }
            None => self.jobs()?.kill(&scope, &ids, remove).await,
        }
        .map_err(|e| io_err(&e))?;
        let survivors: Vec<Value> = survivors
            .iter()
            .map(|(id, d)| json!({ "id": id.as_str(), "detail": d }))
            .collect();
        Ok(json!({ "survivors": survivors }))
    }

    async fn job_write(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let at =
            u64_arg(req, "at")?.ok_or_else(|| err("invalid_request", "`at` must be a count"))?;
        let data = bytes_arg(req, "data_b64")?;
        let eof = flag(req, "eof")?;
        if data.len() > beyond_sandbox::exec_endpoint::jobs::WRITE_MAX {
            return Err(err("invalid_request", "a write is at most 128 KiB"));
        }
        match self.jobs()?.write_stdin(&scope, &id, at, &data, eof).await {
            Ok(w) => {
                let mut v = json!({ "written": w.written });
                if w.closed {
                    v["closed"] = json!(true);
                }
                Ok(v)
            }
            Err(e) => {
                use std::io::ErrorKind::{BrokenPipe, InvalidInput, NotFound, TimedOut};
                Err(match e.kind() {
                    _ if JobError::of(&e).is_some() => job_err(&e),
                    NotFound => err("not_found", e.to_string()),
                    // 413 and a terminal's refusal of `eof` (400).
                    InvalidInput => err("invalid_request", e.to_string()),
                    BrokenPipe | TimedOut => err("conflict", e.to_string()),
                    _ => io_err(&e),
                })
            }
        }
    }

    async fn job_resize(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let id = Self::job_id(req, "job")?;
        let dim = |k: &str| {
            u64_arg(req, k)?.ok_or_else(|| err("invalid_request", format!("`{k}` must be a count")))
        };
        // Out of range is `invalid_request` and nothing is sent.
        let size = PtySize::new(dim("cols")?, dim("rows")?).ok_or_else(|| {
            err(
                "invalid_request",
                "pty: cols and rows are integers from 1 to 1024",
            )
        })?;
        match self.jobs()?.resize(&scope, &id, size).await {
            Ok(p) => Ok(json!({ "cols": p.cols, "rows": p.rows })),
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => {
                Err(err("invalid_request", e.to_string()))
            }
            Err(e) => Err(job_err(&e)),
        }
    }

    async fn job_signal(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let ids: Vec<JobId> = strings(req, "jobs")?
            .iter()
            .map(|s| JobId::parse(s).ok_or_else(|| err("invalid_request", "bad job id")))
            .collect::<Result<_, _>>()?;
        let signal = str_arg(req, "signal")?;
        if !beyond_sandbox::exec_endpoint::jobs::JOB_SIGNALS.contains(&signal) {
            return Err(err(
                "invalid_request",
                format!("unsupported signal {signal:?}"),
            ));
        }
        let sent = self
            .jobs()?
            .signal(&scope, &ids, signal)
            .await
            .map_err(|e| io_err(&e))?;
        let sent: Vec<&str> = sent.iter().map(JobId::as_str).collect();
        Ok(json!({ "sent": sent }))
    }

    async fn job_remove(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let ids: Vec<JobId> = strings(req, "jobs")?
            .iter()
            .filter_map(|s| JobId::parse(s))
            .collect();
        let marks: Vec<(JobId, String)> = match req.get("marks") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Object(m)) => m
                .iter()
                .map(|(k, v)| {
                    let id = JobId::parse(k).ok_or_else(|| err("invalid_request", "bad job id"))?;
                    let mark = v
                        .as_str()
                        .ok_or_else(|| err("invalid_request", "a mark is a string"))?;
                    Ok((id, mark.to_string()))
                })
                .collect::<Result<_, Value>>()?,
            Some(_) => return Err(err("invalid_request", "`marks` is an object")),
        };
        let left = self
            .jobs()?
            .remove(&scope, &ids, &marks)
            .await
            .map_err(|e| io_err(&e))?;
        let names = |v: &[JobId]| v.iter().map(|i| i.as_str().to_string()).collect::<Vec<_>>();
        Ok(json!({ "running": names(&left.running), "mismatch": names(&left.mismatch) }))
    }

    async fn job_wipe(&self, req: &Value) -> Out {
        let scope = self.scope(req)?;
        let survivors = match u64_arg(req, "grace_ms")? {
            Some(grace) => {
                NativeJobs::over(self.http()?)
                    .wipe_grace(&scope, grace)
                    .await
            }
            None => self.jobs()?.wipe(&scope).await,
        }
        .map_err(|e| io_err(&e))?;
        Ok(json!({ "survivors": survivors }))
    }

    // ---- layer 2: files ----------------------------------------------------------------------

    async fn stat(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        match self
            .sb
            .files()
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
            .sb
            .files()
            .read_bytes(Path::new(path), offset, max)
            .await
            .map_err(|e| fs_err(&e))?;
        Ok(json!({ "data_b64": b64(&bytes) }))
    }

    async fn write(&self, req: &Value) -> Out {
        let path = str_arg(req, "path")?;
        let data = bytes_arg(req, "data_b64")?;
        self.sb
            .files()
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
        let expected = match &req["expected_mtime"] {
            Value::Null => None,
            v => v
                .as_f64()
                .filter(|s| s.is_finite() && *s >= 0.0)
                .map(|s| UNIX_EPOCH + Duration::from_secs_f64(s)),
        };
        if expected.is_none() && !req["expected_mtime"].is_null() {
            return Err(err(
                "invalid_request",
                "`expected_mtime` must be a non-negative number or null",
            ));
        }
        let wrote = self
            .sb
            .files()
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
        self.sb
            .files()
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
            .sb
            .files()
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
        let o = self.sb.files().search(&q).await.map_err(|e| fs_err(&e))?;
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

// ---- bulk copy (FILES-OVER-EXEC.md 5.8) ---------------------------------------------------------

fn transfer_json(t: beyond_sandbox::Transfer) -> Value {
    json!({"files": t.files, "dirs": t.dirs, "links": t.links, "bytes": t.bytes})
}

impl Driver {
    async fn upload(&self, req: &Value) -> Out {
        let dest = str_arg(req, "path")?;
        let chunk = u64_arg(req, "chunk_bytes")?.map(|c| c as usize);
        let tree = Tree::build(req)?;
        let sources = match &req["sources"] {
            Value::Null => beyond_sandbox::Sources::Dir(tree.dir()),
            _ => beyond_sandbox::Sources::Paths(
                strings(req, "sources")?
                    .iter()
                    .map(|n| tree.dir().join(n))
                    .collect(),
            ),
        };
        let t = match chunk {
            Some(chunk) => {
                beyond_sandbox::bulk::upload_chunked(&**self.sb.raw(), sources, dest, chunk).await
            }
            None => self.sb.upload(sources, dest).await,
        }
        .map_err(|e| bulk_err(&e))?;
        Ok(transfer_json(t))
    }

    async fn download(&self, req: &Value) -> Out {
        let remote = str_arg(req, "path")?;
        let layout = Layout::new(&req["prefill"])?;
        let r = self.sb.download(remote, &layout.local).await;
        layout.finish(r.map(transfer_json).map_err(|e| bulk_err(&e)))
    }

    async fn extract(&self, req: &Value) -> Out {
        let tar = bytes_arg(req, "tar_b64")?;
        let layout = Layout::new(&req["prefill"])?;
        let r = beyond_sandbox::bulk::extract(&tar, &layout.local);
        layout.finish(r.map(transfer_json).map_err(|e| bulk_err(&e)))
    }
}
