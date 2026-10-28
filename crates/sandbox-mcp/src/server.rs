//! The MCP server: one sandbox connection, twelve tools (fourteen with a local root) and one resource.
//!
//! Every tool is a thin mapping from its arguments to one call of [`beyond_sandbox`] (raw exec,
//! the file helpers, jobs) and from its outcome to a structured result. The connection is made on
//! the first call that needs it, so a server whose sandbox is down (or whose token is wrong) still
//! starts, lists its tools, and answers each call with the error kind (`transport`, `auth`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use beyond_sandbox::exec_endpoint::jobs::{JobId, JobScope, JobSpec, JobStatus, READ_MAX};
use beyond_sandbox::fs::shell::{Capabilities, SearchEngine, ShellFs};
use beyond_sandbox::fs::{FileKind, FsBackend, FsError, SearchQuery};
use beyond_sandbox::{HttpExec, Sandbox, TokenSource};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ListResourcesResult, ReadResourceRequestParams,
    ReadResourceResponse, ReadResourceResult, ResourceContents, ServerCapabilities, ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler, schemars, tool, tool_handler, tool_router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::config::{CapsOverride, Config, Token};
use crate::error::{ToolError, reply};

/// The resource that reports what the connected sandbox can do.
pub const CAPABILITIES_URI: &str = "sandbox://capabilities";

/// How much of a file `read_file` returns unless asked for more: the same 128 KiB a job read
/// returns, which is also the head the exec client keeps whole.
pub const DEFAULT_READ_MAX: usize = READ_MAX;
/// How many entries `list_dir` returns unless asked for more.
pub const DEFAULT_LIST_LIMIT: usize = 1000;
/// How many hits `search` returns unless asked for more (the spec's default).
pub const DEFAULT_SEARCH_LIMIT: usize = 100;

type Out = Result<Value, ToolError>;

/// A connected sandbox, with what the capabilities resource needs from it.
struct Target {
    sandbox: Sandbox,
    /// File helpers whose probe results were overridden (`SANDBOX_CAPS`); `None` otherwise.
    files: Option<ShellFs>,
}

impl Target {
    fn files(&self) -> &ShellFs {
        self.files.as_ref().unwrap_or_else(|| self.sandbox.files())
    }
}

struct Inner {
    config: Config,
    target: Mutex<Option<Arc<Target>>>,
}

/// The server. Cloning shares the one sandbox connection.
#[derive(Clone)]
pub struct SandboxMcp {
    inner: Arc<Inner>,
    tool_router: ToolRouter<Self>,
}

// ---- arguments ------------------------------------------------------------------------------

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UploadArgs {
    /// A directory or file of this machine, relative to the server's local root. A directory
    /// uploads its contents; a file uploads that file.
    local: String,
    /// The sandbox directory to put it in (absolute; created if missing).
    remote_dir: String,
    /// When `local` is a directory: upload only these entries of it, each by its own name.
    only: Option<Vec<String>>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DownloadArgs {
    /// A directory (its contents) or file of the sandbox (absolute).
    remote_path: String,
    /// The directory of this machine to extract into, relative to the server's local root;
    /// created if missing.
    local_dir: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecArgs {
    /// The whole shell command line, run with `sh -c`: pipes, `&&`, redirects, quoting, `cd` and
    /// environment variables all work, e.g. `cd app && python -m pytest -q 2>&1 | tail -20`. This is
    /// the one string you would type at a prompt; there is no separate argument list.
    command: String,
    /// Working directory. Default: the sandbox's own.
    cwd: Option<String>,
    /// Kill the command after this many milliseconds. Default 120000.
    timeout_ms: Option<u64>,
    /// Text for the command's standard input.
    stdin: Option<String>,
    /// Bytes for the command's standard input, base64. Use instead of `stdin` for binary data.
    stdin_base64: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFileArgs {
    /// Absolute path, or relative to the sandbox's working directory.
    path: String,
    /// Byte offset to start at. Default 0.
    offset: Option<u64>,
    /// The most bytes to return. Default 131072. When the file is longer, `eof` is false: call
    /// again with `offset` set to the `next_offset` you were given.
    max: Option<u64>,
    /// `utf8` (default) returns text when the bytes are valid UTF-8 and base64 otherwise; `base64`
    /// always returns base64. The result's `encoding` says which you got.
    encoding: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteFileArgs {
    /// Absolute path, or relative to the sandbox's working directory.
    path: String,
    /// The new contents as text. Give this or `content_base64`.
    content: Option<String>,
    /// The new contents as base64, for binary files.
    content_base64: Option<String>,
    /// Write only if the file's mtime is still this: the `mtime` that `stat` returned, fraction
    /// included. If the file changed since (or was removed), nothing is written and the call fails
    /// with kind `conflict`. Leave out to write unconditionally.
    if_unchanged_since: Option<f64>,
    /// Write only if no file exists at `path` yet (kind `conflict` if one does). Default false.
    /// Not combinable with `if_unchanged_since`.
    create_only: Option<bool>,
    /// Create missing parent directories (`mkdir -p`). Default true. With false, a missing parent
    /// fails with kind `not_found` and a detail naming the directory.
    create_parents: Option<bool>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditFileArgs {
    /// Path of an existing text file.
    path: String,
    /// The exact text to replace. Must occur once, unless `replace_all` is true.
    old_str: String,
    /// The text to put in its place.
    new_str: String,
    /// Replace every occurrence instead of requiring exactly one. Default false.
    replace_all: Option<bool>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListDirArgs {
    /// The directory.
    path: String,
    /// Include entries whose names start with a dot. Default false.
    include_hidden: Option<bool>,
    /// The most entries to return. Default 1000; `truncated` says when there were more.
    limit: Option<u64>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PathArgs {
    /// Absolute path, or relative to the sandbox's working directory.
    path: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchArgs {
    /// What to find: a regular expression, or plain text with `literal`.
    pattern: String,
    /// The directory to search under. Default: the sandbox's working directory.
    path: Option<String>,
    /// Lines of context before and after each match. Default 0.
    context: Option<u64>,
    /// Lines of context before each match; overrides `context`.
    before: Option<u64>,
    /// Lines of context after each match; overrides `context`.
    after: Option<u64>,
    /// The most hits to return (context lines count). Default 100; `truncated` says when there were more.
    limit: Option<u64>,
    /// Treat `pattern` as plain text. Default false.
    literal: Option<bool>,
    /// Ignore case. Default false.
    ignore_case: Option<bool>,
    /// Only search files whose names match this glob, e.g. `*.rs`.
    glob: Option<String>,
    /// Search every file except those matching `glob`. Default false.
    glob_exclude: Option<bool>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobStartArgs {
    /// The whole shell command line, run with `sh -c`: pipes, `&&`, redirects, quoting and `cd` all
    /// work, e.g. `npm run dev` or `cd app && make 2>&1`. There is no separate argument list.
    command: String,
    /// Working directory. Default: the sandbox's own.
    cwd: Option<String>,
    /// A short note shown in `job_list`.
    label: Option<String>,
    /// The job's id: 1 to 32 characters of [0-9a-z]. Default: a random one, returned.
    id: Option<String>,
    /// The group the job belongs to. Default: this server's.
    scope: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobListArgs {
    /// The group to list. Default: this server's.
    scope: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobOutputArgs {
    /// The job's id, from `job_start` or `job_list`.
    id: String,
    /// Byte offset in the job's combined output to start at. Default 0.
    offset: Option<u64>,
    /// The most bytes to return. Default 131072. Call again with `offset` set to `next_offset`.
    max: Option<u64>,
    /// `utf8` (default) returns text when the bytes are valid UTF-8 and base64 otherwise; `base64`
    /// always returns base64. The result's `encoding` says which you got.
    encoding: Option<String>,
    /// The group the job belongs to. Default: this server's.
    scope: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobKillArgs {
    /// The jobs to stop.
    ids: Vec<String>,
    /// Also delete the stopped jobs and their output. Default false.
    remove: Option<bool>,
    /// The group the jobs belong to. Default: this server's.
    scope: Option<String>,
}

// ---- helpers --------------------------------------------------------------------------------

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn kind_name(k: FileKind) -> &'static str {
    match k {
        FileKind::File => "file",
        FileKind::Dir => "dir",
        FileKind::Other => "other",
    }
}

/// Epoch seconds with the nanoseconds `stat` reports, as the double every client compares.
fn epoch_secs(t: std::time::SystemTime) -> f64 {
    beyond_sandbox::fs::epoch_f64(t)
}

/// A count from a tool argument, as `usize`.
fn count(n: u64) -> usize {
    usize::try_from(n).unwrap_or(usize::MAX)
}

fn decode(what: &str, b64: &str) -> Result<Vec<u8>, ToolError> {
    B64.decode(b64)
        .map_err(|e| ToolError::invalid(format!("`{what}` is not base64: {e}")))
}

/// Bytes as the tool result carries them: `(encoding, content, bytes used)`. In `auto` mode text is
/// text, and a read that ended in the middle of a multi-byte character gives that character up (the
/// caller continues at `offset + bytes`); anything else is base64.
fn payload(
    bytes: &[u8],
    encoding: Option<&str>,
) -> Result<(&'static str, String, usize), ToolError> {
    match encoding {
        None | Some("utf8") => match std::str::from_utf8(bytes) {
            Ok(s) => Ok(("utf8", s.to_string(), bytes.len())),
            Err(e) if e.error_len().is_none() && e.valid_up_to() > 0 => {
                let n = e.valid_up_to();
                Ok(("utf8", String::from_utf8_lossy(&bytes[..n]).into_owned(), n))
            }
            Err(_) => Ok(("base64", B64.encode(bytes), bytes.len())),
        },
        Some("base64") => Ok(("base64", B64.encode(bytes), bytes.len())),
        Some(other) => Err(ToolError::invalid(format!(
            "`encoding` must be \"utf8\" or \"base64\", not {other:?}"
        ))),
    }
}

/// Command output as the result carries it: text when the bytes are UTF-8, base64 (marked) when not.
fn text_or_base64(bytes: Vec<u8>) -> (&'static str, String) {
    match String::from_utf8(bytes) {
        Ok(s) => ("utf8", s),
        Err(e) => ("base64", B64.encode(e.into_bytes())),
    }
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

/// `signal` in a job answer: present only for a job a signal killed (its `code` is then 128 + signal).
fn put_signal(obj: &mut Value, signal: Option<i32>) {
    if let Some(s) = signal {
        obj["signal"] = json!(s);
    }
}

/// Whether a failed write says its parent directory does not exist (FILES-OVER-EXEC.md section 5.3).
fn is_missing_parent(detail: &str) -> bool {
    detail.contains(": parent directory ") && detail.contains(" does not exist")
}

fn transfer(t: beyond_sandbox::Transfer) -> Value {
    json!({ "files": t.files, "dirs": t.dirs, "links": t.links, "bytes": t.bytes })
}

fn job_id(s: &str) -> Result<JobId, ToolError> {
    JobId::parse(s).ok_or_else(|| {
        ToolError::invalid(format!("bad job id {s:?}: 1 to 32 characters of [0-9a-z]"))
    })
}

// ---- the server -----------------------------------------------------------------------------

impl SandboxMcp {
    pub fn new(config: Config) -> Self {
        let local_root_unset = config.local_root.is_none();
        Self {
            inner: Arc::new(Inner {
                config,
                target: Mutex::new(None),
            }),
            tool_router: {
                let mut r = Self::tool_router();
                if local_root_unset {
                    r.remove_route("upload");
                    r.remove_route("download");
                }
                r
            },
        }
    }

    /// The sandbox, connected on first use. A failed connection is not remembered: the next call
    /// tries again (the endpoint may come up, the token command may start working).
    async fn target(&self) -> Result<Arc<Target>, ToolError> {
        let mut slot = self.inner.target.lock().await;
        if let Some(t) = &*slot {
            return Ok(t.clone());
        }
        let c = &self.inner.config;
        let mut http = HttpExec::new(&c.url).map_err(ToolError::invalid)?;
        match &c.token {
            Token::None => {}
            Token::Fixed(t) => http = http.bearer(TokenSource::fixed(t.clone())),
            Token::Command(cmd) => {
                http = http.bearer(TokenSource::command(["sh", "-c", cmd.as_str()]));
            }
        }
        let http = Arc::new(http);
        let sandbox = Sandbox::over(http.clone()).await?;
        let files = override_caps(&sandbox, &c.caps).await?;
        let target = Arc::new(Target { sandbox, files });
        *slot = Some(target.clone());
        Ok(target)
    }

    fn scope(&self, name: Option<&str>) -> Result<JobScope, ToolError> {
        let c = &self.inner.config;
        let name = name.unwrap_or(&c.jobs_scope);
        let ok = !name.is_empty()
            && name.len() <= 64
            && name
                .bytes()
                .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase() || b == b'-');
        if !ok {
            return Err(ToolError::invalid(format!(
                "bad scope {name:?}: 1 to 64 characters of [0-9a-z-]"
            )));
        }
        Ok(JobScope::under(
            beyond_sandbox::exec_endpoint::jobs::JOBS_ROOT,
            name,
        ))
    }

    /// What the connected sandbox has: the capabilities resource. The first call looks at the
    /// sandbox's tools, in one command.
    pub async fn capabilities(&self) -> Out {
        let t = self.target().await?;
        let caps = t.files().capabilities().await?;
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
}

/// The file helpers with their probe results replaced, or `None` when nothing is overridden.
async fn override_caps(sandbox: &Sandbox, o: &CapsOverride) -> Result<Option<ShellFs>, ToolError> {
    if *o == CapsOverride::default() {
        return Ok(None);
    }
    let mut caps: Capabilities = sandbox.files().capabilities().await?;
    if let Some(f) = o.find_printf {
        caps.find_printf = f;
    }
    match o.search_engine.as_deref() {
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
    Ok(Some(ShellFs::with_capabilities(
        sandbox.raw().clone(),
        caps,
    )))
}

type ToolReply = Result<rmcp::model::CallToolResult, ErrorData>;

#[tool_router]
impl SandboxMcp {
    #[tool(
        name = "exec",
        description = "Run a shell command line in the sandbox (sh -c) and wait for it. `command` is the one string you would type at a prompt, e.g. `cd app && python -m pytest -q 2>&1 | tail -20`; pipes, &&, redirects and quotes work, and there is no separate args parameter. Returns exit_code, stdout and stderr, each with its encoding (utf8, or base64 when the bytes are not UTF-8); a non-zero exit is a normal result, not an error, and 127 means the command was not found. The whole output comes back, so for long-running or noisy commands use job_start and page the output with job_output."
    )]
    async fn exec(&self, Parameters(a): Parameters<ExecArgs>) -> ToolReply {
        Ok(reply(self.do_exec(a).await))
    }

    #[tool(
        name = "read_file",
        description = "Read bytes of a file. Returns content, encoding (utf8, or base64 when the bytes are not UTF-8), bytes, eof and next_offset. At most max bytes (default 131072) come back: if eof is false, call again with offset = next_offset. Fails with not_found, is_directory or permission_denied."
    )]
    async fn read_file(&self, Parameters(a): Parameters<ReadFileArgs>) -> ToolReply {
        Ok(reply(self.do_read(a).await))
    }

    #[tool(
        name = "write_file",
        description = "Create a file or replace it entirely, atomically; an existing file keeps its permissions. Missing parent directories are created (create_parents: false to fail instead). Give content (text) or content_base64 (binary). To change part of a file use edit_file. To avoid overwriting a concurrent change, pass if_unchanged_since (the mtime from stat): a changed file fails with kind conflict and nothing is written. create_only refuses to replace a file that exists."
    )]
    async fn write_file(&self, Parameters(a): Parameters<WriteFileArgs>) -> ToolReply {
        Ok(reply(self.do_write(a).await))
    }

    #[tool(
        name = "edit_file",
        description = "Replace text in an existing file. old_str must match exactly once (or set replace_all). Never overwrites a concurrent change: if the file changed after it was read, fails with kind conflict and writes nothing; read the file again and retry."
    )]
    async fn edit_file(&self, Parameters(a): Parameters<EditFileArgs>) -> ToolReply {
        Ok(reply(self.do_edit(a).await))
    }

    #[tool(
        name = "list_dir",
        description = "List a directory: entries with name, kind (file, dir or other) and size in bytes, sorted by name. Hidden entries are left out unless include_hidden. At most limit entries (default 1000); truncated says when there were more."
    )]
    async fn list_dir(&self, Parameters(a): Parameters<ListDirArgs>) -> ToolReply {
        Ok(reply(self.do_list(a).await))
    }

    #[tool(
        name = "stat",
        description = "Describe a path. Returns exists; when it does, kind (file, dir or other), size in bytes, mtime (epoch seconds, with the fraction the filesystem records) and whether the sandbox user can write it. A missing path is exists: false, not an error."
    )]
    async fn stat(&self, Parameters(a): Parameters<PathArgs>) -> ToolReply {
        Ok(reply(self.do_stat(a).await))
    }

    #[tool(
        name = "search",
        description = "Search file contents under a directory for a regular expression (ripgrep when the sandbox has it, grep otherwise; see the sandbox://capabilities resource). Returns hits with path, line, text and is_match (false for context lines), and truncated. At most limit hits (default 100). Fails with kind invalid_query for a bad pattern or glob."
    )]
    async fn search(&self, Parameters(a): Parameters<SearchArgs>) -> ToolReply {
        Ok(reply(self.do_search(a).await))
    }

    #[tool(
        name = "mkdir",
        description = "Create a directory and any missing parents. Succeeds if it already exists."
    )]
    async fn mkdir(&self, Parameters(a): Parameters<PathArgs>) -> ToolReply {
        Ok(reply(self.do_mkdir(a).await))
    }

    #[tool(
        name = "upload",
        description = "Copy a directory or file of the server's machine (a path under its local root) into a sandbox directory, as a tar stream: modes, symlinks and empty directories survive, large trees go in chunks. Returns files, dirs, links and bytes. Merges into a destination that exists; a destination that is a file fails with not_a_directory."
    )]
    async fn upload(&self, Parameters(a): Parameters<UploadArgs>) -> ToolReply {
        Ok(reply(self.do_upload(a).await))
    }

    #[tool(
        name = "download",
        description = "Copy a sandbox directory (its contents) or file into a directory of the server's machine (a path under its local root), as a tar stream. Links are copied as links and never followed, and nothing is written outside the target. Returns files, dirs, links and bytes."
    )]
    async fn download(&self, Parameters(a): Parameters<DownloadArgs>) -> ToolReply {
        Ok(reply(self.do_download(a).await))
    }

    #[tool(
        name = "job_start",
        description = "Start a shell command line (sh -c, like exec) that keeps running after this call returns; the sandbox keeps it and its output. Returns the job id. Read output with job_output, check on it with job_list, stop it with job_kill. Use this for servers, builds and anything slower than a few seconds."
    )]
    async fn job_start(&self, Parameters(a): Parameters<JobStartArgs>) -> ToolReply {
        Ok(reply(self.do_job_start(a).await))
    }

    #[tool(
        name = "job_list",
        description = "List jobs: id, state (running, exited, ended, lost or gone), code (the exit status once ended), signal (only when a signal killed the job's shell: code is then 128 + signal, and a job that merely exited 137 has no signal), label, started and ended (epoch seconds). exited means the shell is done but processes it started still run."
    )]
    async fn job_list(&self, Parameters(a): Parameters<JobListArgs>) -> ToolReply {
        Ok(reply(self.do_job_list(a).await))
    }

    #[tool(
        name = "job_output",
        description = "Read a job's output (stdout and stderr together) from a byte offset. Returns state, code, signal (only when a signal killed the job's shell: code is then 128 + signal; a job that exited on its own has none), content, encoding (utf8, or base64 when the bytes are not UTF-8), offset, next_offset and end (the output's length so far). At most max bytes (default 131072) per call: to follow a job, call again with offset = next_offset until state is ended."
    )]
    async fn job_output(&self, Parameters(a): Parameters<JobOutputArgs>) -> ToolReply {
        Ok(reply(self.do_job_output(a).await))
    }

    #[tool(
        name = "job_kill",
        description = "Stop jobs: SIGTERM, a short grace, then SIGKILL. Returns survivors, the processes still alive afterwards (normally none). Stopping an ended job is not an error. With remove, the jobs and their output are deleted."
    )]
    async fn job_kill(&self, Parameters(a): Parameters<JobKillArgs>) -> ToolReply {
        Ok(reply(self.do_job_kill(a).await))
    }
}

impl SandboxMcp {
    async fn do_exec(&self, a: ExecArgs) -> Out {
        let stdin = match (a.stdin, a.stdin_base64) {
            (Some(_), Some(_)) => {
                return Err(ToolError::invalid("give stdin or stdin_base64, not both"));
            }
            (Some(t), None) => Some(t.into_bytes()),
            (None, Some(b)) => Some(decode("stdin_base64", &b)?),
            (None, None) => None,
        };
        let t = self.target().await?;
        let mut cmd = t.sandbox.exec("sh", ["-c".to_string(), a.command]);
        if let Some(cwd) = a.cwd {
            cmd = cmd.cwd(cwd);
        }
        if let Some(ms) = a.timeout_ms {
            cmd = cmd.timeout(Duration::from_millis(ms));
        }
        if let Some(stdin) = stdin {
            cmd = cmd.stdin(stdin);
        }
        let r = cmd.run().await?;
        let (stdout_enc, stdout) = text_or_base64(r.stdout);
        let (stderr_enc, stderr) = text_or_base64(r.stderr);
        Ok(json!({
            "exit_code": r.code,
            "stdout": stdout,
            "stdout_encoding": stdout_enc,
            "stderr": stderr,
            "stderr_encoding": stderr_enc,
        }))
    }

    async fn do_read(&self, a: ReadFileArgs) -> Out {
        let offset = a.offset.unwrap_or(0);
        let max = a.max.map_or(DEFAULT_READ_MAX, count);
        let t = self.target().await?;
        let bytes = t
            .files()
            .read_bytes(Path::new(&a.path), offset, max)
            .await?;
        let (encoding, content, used) = payload(&bytes, a.encoding.as_deref())?;
        Ok(json!({
            "path": a.path,
            "encoding": encoding,
            "content": content,
            "offset": offset,
            "bytes": used,
            "next_offset": offset + used as u64,
            "eof": bytes.len() < max,
        }))
    }

    async fn do_write(&self, a: WriteFileArgs) -> Out {
        let bytes = match (a.content, a.content_base64) {
            (Some(_), Some(_)) => {
                return Err(ToolError::invalid(
                    "give content or content_base64, not both",
                ));
            }
            (Some(t), None) => t.into_bytes(),
            (None, Some(b)) => decode("content_base64", &b)?,
            (None, None) => return Err(ToolError::invalid("give content or content_base64")),
        };
        let n = bytes.len();
        let create_only = a.create_only.unwrap_or(false);
        let expected = match (a.if_unchanged_since, create_only) {
            (Some(_), true) => {
                return Err(ToolError::invalid(
                    "give if_unchanged_since or create_only, not both",
                ));
            }
            (Some(secs), false) => Some(Some(
                std::time::Duration::try_from_secs_f64(secs)
                    .ok()
                    .and_then(|d| std::time::UNIX_EPOCH.checked_add(d))
                    .ok_or_else(|| {
                        ToolError::invalid("if_unchanged_since must be a mtime from stat")
                    })?,
            )),
            (None, true) => Some(None),
            (None, false) => None,
        };
        let t = self.target().await?;
        let files = t.files();
        let path = Path::new(&a.path);
        // Agents write `a/b/c.py` without making `a/b` first, so a missing parent is made, not
        // reported: try the write, and only when it says the parent is missing, `mkdir -p` and retry
        // once (a write into an existing directory costs no extra command).
        let create_parents = a.create_parents.unwrap_or(true);
        let mut made_parents = false;
        let mut bytes = bytes;
        loop {
            // The first attempt keeps a copy for the retry; the retry spends the original.
            let body = if made_parents {
                std::mem::take(&mut bytes)
            } else {
                bytes.clone()
            };
            let done = match expected {
                None => files.write_bytes(path, body).await.map(|()| true),
                Some(mtime) => files.write_if_unchanged(path, body, mtime).await,
            };
            match done {
                Ok(true) => break,
                Ok(false) => {
                    return Err(ToolError::new(
                        "conflict",
                        format!("{} changed since it was read; nothing was written", a.path),
                    ));
                }
                Err(FsError::Backend(m))
                    if create_parents && !made_parents && is_missing_parent(&m) =>
                {
                    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                        files.create_dir_all(parent).await?;
                    }
                    made_parents = true;
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(json!({ "path": a.path, "bytes": n }))
    }

    async fn do_edit(&self, a: EditFileArgs) -> Out {
        if a.old_str.is_empty() {
            return Err(ToolError::invalid("old_str is empty"));
        }
        let path = Path::new(&a.path);
        let t = self.target().await?;
        let files = t.files();
        let meta = files
            .stat(path)
            .await?
            .ok_or_else(|| ToolError::new("not_found", format!("{}: no such file", a.path)))?;
        match meta.kind {
            FileKind::File => {}
            FileKind::Dir => {
                return Err(ToolError::new(
                    "is_directory",
                    format!("{}: is a directory", a.path),
                ));
            }
            FileKind::Other => {
                return Err(ToolError::new(
                    "io",
                    format!("{}: not a regular file", a.path),
                ));
            }
        }
        let len = usize::try_from(meta.len).unwrap_or(usize::MAX);
        let bytes = files.read_bytes(path, 0, len).await?;
        let text = String::from_utf8(bytes)
            .map_err(|_| ToolError::invalid(format!("{} is not UTF-8 text", a.path)))?;
        let found = text.matches(a.old_str.as_str()).count();
        let all = a.replace_all.unwrap_or(false);
        if found == 0 {
            return Err(ToolError::invalid("old_str does not occur in the file"));
        }
        if found > 1 && !all {
            return Err(ToolError::invalid(format!(
                "old_str occurs {found} times; add context to make it unique, or set replace_all"
            )));
        }
        let new = text.replace(a.old_str.as_str(), &a.new_str);
        let n = new.len();
        if !files
            .write_if_unchanged(path, new.into_bytes(), meta.mtime)
            .await?
        {
            return Err(ToolError::new(
                "conflict",
                format!("{} changed since it was read; nothing was written", a.path),
            ));
        }
        Ok(json!({ "path": a.path, "replacements": found, "bytes": n }))
    }

    async fn do_list(&self, a: ListDirArgs) -> Out {
        let limit = a.limit.map_or(DEFAULT_LIST_LIMIT, count);
        let t = self.target().await?;
        let mut entries = t
            .files()
            .list_dir(
                Path::new(&a.path),
                limit.saturating_add(1),
                a.include_hidden.unwrap_or(false),
            )
            .await?;
        let truncated = entries.len() > limit;
        entries.truncate(limit);
        entries.sort_by(|x, y| x.name.cmp(&y.name));
        let entries: Vec<Value> = entries
            .iter()
            .map(|e| json!({ "name": e.name, "kind": kind_name(e.kind), "size": e.len }))
            .collect();
        Ok(json!({ "path": a.path, "entries": entries, "truncated": truncated }))
    }

    async fn do_stat(&self, a: PathArgs) -> Out {
        let t = self.target().await?;
        Ok(match t.files().stat(Path::new(&a.path)).await? {
            None => json!({ "path": a.path, "exists": false }),
            Some(m) => json!({
                "path": a.path,
                "exists": true,
                "kind": kind_name(m.kind),
                "size": m.len,
                "mtime": m.mtime.map(epoch_secs),
                "writable": m.writable,
            }),
        })
    }

    async fn do_search(&self, a: SearchArgs) -> Out {
        let context = a.context.unwrap_or(0);
        let q = SearchQuery {
            pattern: a.pattern,
            literal: a.literal.unwrap_or(false),
            ignore_case: a.ignore_case.unwrap_or(false),
            glob: a.glob.map(|g| (g, a.glob_exclude.unwrap_or(false))),
            root: PathBuf::from(a.path.as_deref().unwrap_or(".")),
            limit: a.limit.map_or(DEFAULT_SEARCH_LIMIT, count),
            before: count(a.before.unwrap_or(context)),
            after: count(a.after.unwrap_or(context)),
        };
        let t = self.target().await?;
        let o = t.files().search(&q).await?;
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

    async fn do_mkdir(&self, a: PathArgs) -> Out {
        let t = self.target().await?;
        t.files().create_dir_all(Path::new(&a.path)).await?;
        Ok(json!({ "path": a.path }))
    }

    /// A path the model named, under the local root: relative, no `..`, and no existing component
    /// that is a link leaving the root.
    fn local_path(&self, p: &str) -> Result<PathBuf, ToolError> {
        let root = self.inner.config.local_root.as_ref().ok_or_else(|| {
            ToolError::invalid("the server has no SANDBOX_LOCAL_ROOT: upload and download are off")
        })?;
        let rel = Path::new(p);
        let mut out = root.clone();
        for c in rel.components() {
            match c {
                std::path::Component::Normal(n) => out.push(n),
                std::path::Component::CurDir => {}
                _ => {
                    return Err(ToolError::invalid(format!(
                        "{p:?} must be a relative path below the local root, without `..`"
                    )));
                }
            }
        }
        // The deepest part that exists, resolved, must still be inside the root.
        let real_root = root.canonicalize()?;
        let mut probe = out.clone();
        loop {
            match probe.canonicalize() {
                Ok(real) => {
                    if !real.starts_with(&real_root) {
                        return Err(ToolError::invalid(format!(
                            "{p:?} leaves the local root through a link"
                        )));
                    }
                    break;
                }
                Err(_) if !probe.pop() => break,
                Err(_) => {}
            }
        }
        Ok(out)
    }

    async fn do_upload(&self, a: UploadArgs) -> Out {
        let local = self.local_path(&a.local)?;
        let sources = if local.is_dir() {
            match &a.only {
                None => beyond_sandbox::Sources::Dir(local),
                Some(names) => {
                    let mut paths = Vec::new();
                    for n in names {
                        if n.contains('/') || n == ".." || n == "." || n.is_empty() {
                            return Err(ToolError::invalid(format!(
                                "`only` names entries of the directory, not paths: {n:?}"
                            )));
                        }
                        paths.push(local.join(n));
                    }
                    beyond_sandbox::Sources::Paths(paths)
                }
            }
        } else {
            beyond_sandbox::Sources::Paths(vec![local])
        };
        let t = self.target().await?;
        let r = t.sandbox.upload(sources, &a.remote_dir).await?;
        Ok(transfer(r))
    }

    async fn do_download(&self, a: DownloadArgs) -> Out {
        let local = self.local_path(&a.local_dir)?;
        let t = self.target().await?;
        let r = t.sandbox.download(&a.remote_path, &local).await?;
        Ok(transfer(r))
    }

    async fn do_job_start(&self, a: JobStartArgs) -> Out {
        let scope = self.scope(a.scope.as_deref())?;
        let id = match a.id.as_deref() {
            Some(s) => job_id(s)?,
            None => JobId::fresh(),
        };
        let spec = JobSpec {
            program: "sh".to_string(),
            args: vec!["-c".to_string(), a.command],
            cwd: a.cwd,
            label: a.label.unwrap_or_default(),
        };
        let t = self.target().await?;
        let pgid = t.sandbox.jobs()?.start_kept(&scope, &id, &spec).await?;
        Ok(json!({ "id": id.as_str(), "scope": scope.name(), "pgid": pgid }))
    }

    async fn do_job_list(&self, a: JobListArgs) -> Out {
        let scope = self.scope(a.scope.as_deref())?;
        let t = self.target().await?;
        let jobs = t.sandbox.jobs()?.list(&scope).await?;
        let jobs: Vec<Value> = jobs
            .iter()
            .map(|j| {
                let (state, code) = state_of(j.status);
                let mut entry = json!({
                    "id": j.id.as_str(), "state": state, "code": code, "kept": j.kept,
                    "num": j.num, "kind": j.kind, "seen": j.seen,
                    "started": j.started, "ended": j.ended, "label": j.label,
                });
                put_signal(&mut entry, j.signal);
                entry
            })
            .collect();
        Ok(json!({ "scope": scope.name(), "jobs": jobs }))
    }

    async fn do_job_output(&self, a: JobOutputArgs) -> Out {
        let scope = self.scope(a.scope.as_deref())?;
        let id = job_id(&a.id)?;
        let max = a.max.map_or(DEFAULT_READ_MAX, count);
        let t = self.target().await?;
        let c = t
            .sandbox
            .jobs()?
            .read(&scope, &id, a.offset.unwrap_or(0), max)
            .await?;
        let (state, code) = state_of(c.status);
        let (encoding, content, used) = payload(&c.data, a.encoding.as_deref())?;
        let mut out = json!({
            "id": id.as_str(),
            "state": state,
            "code": code,
            "encoding": encoding,
            "content": content,
            "offset": c.offset,
            "bytes": used,
            "next_offset": c.offset + used as u64,
            "end": c.end,
        });
        put_signal(&mut out, c.signal);
        Ok(out)
    }

    async fn do_job_kill(&self, a: JobKillArgs) -> Out {
        let scope = self.scope(a.scope.as_deref())?;
        let ids = a
            .ids
            .iter()
            .map(|s| job_id(s))
            .collect::<Result<Vec<_>, _>>()?;
        let t = self.target().await?;
        let survivors = t
            .sandbox
            .jobs()?
            .kill(&scope, &ids, a.remove.unwrap_or(false))
            .await?;
        let survivors: Vec<Value> = survivors
            .iter()
            .map(|(id, d)| json!({ "id": id.as_str(), "detail": d }))
            .collect();
        Ok(json!({ "survivors": survivors }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SandboxMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(rmcp::model::Implementation::new(
            "sandbox-mcp",
            env!("CARGO_PKG_VERSION"),
        ))
        .with_instructions(
            "Tools for one sandbox: exec, files (read_file, write_file, edit_file, list_dir, stat, \
             search, mkdir), bulk copy between this machine and the sandbox (upload, download; only \
             when the server has a local root) and background jobs (job_start, job_list, job_output, job_kill). A failed \
             call returns isError with structuredContent.error.kind (not_found, permission_denied, \
             conflict, auth, transport, ...). The sandbox://capabilities resource says what the \
             sandbox supports.",
        )
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        match self.tool_router.call(tcc).await {
            // Arguments that do not fit the schema are the agent's mistake to see and fix. rmcp
            // answers them with a bare error result (no structured content) or, for some shapes,
            // a protocol error; both become the same `invalid_request` result as every other failure.
            Ok(CallToolResponse::Complete(r))
                if r.is_error == Some(true) && r.structured_content.is_none() =>
            {
                let text = r
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join(" ");
                Ok(ToolError::invalid(text).into_result().into())
            }
            Err(e) if e.code == rmcp::model::ErrorCode::INVALID_PARAMS => {
                Ok(ToolError::invalid(e.message.to_string())
                    .into_result()
                    .into())
            }
            other => other,
        }
    }

    async fn list_resources(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        let resource = rmcp::model::Resource::new(CAPABILITIES_URI, "capabilities")
            .with_description(
                "What the connected sandbox has: search_engine, find_printf, read_window; or an error when it cannot be reached.",
            )
            .with_mime_type("application/json");
        Ok(ListResourcesResult {
            resources: vec![resource],
            ..Default::default()
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        if request.uri != CAPABILITIES_URI {
            return Err(ErrorData::resource_not_found(
                format!("no resource {}", request.uri),
                None,
            ));
        }
        let body = match self.capabilities().await {
            Ok(v) => v,
            Err(e) => json!({ "error": { "kind": e.kind, "detail": e.detail } }),
        };
        Ok(ReadResourceResult::new(vec![
            ResourceContents::text(body.to_string(), CAPABILITIES_URI)
                .with_mime_type("application/json"),
        ])
        .into())
    }
}
