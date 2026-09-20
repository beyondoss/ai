//! Executing commands against a remote endpoint — any endpoint.
//!
//! The agent is handed a **URL** and it POSTs commands to it. That is the entire contract. It does not
//! know, and must never know, what is on the other side: Daytona, E2B, Modal, a container behind an
//! internal service, someone's laptop over a tunnel. Nothing here is specific to any vendor.
//!
//! ## Not the agent's business
//!
//! **Lifecycle.** Creating, warming, snapshotting, pausing or destroying a sandbox is the caller's job.
//! The agent receives an endpoint that already works and uses it until told otherwise. An agent that
//! provisions infrastructure has to model quotas, billing, placement and cleanup — none of which has
//! anything to do with running a coding task, and all of which differ per vendor.
//!
//! **Co-location.** The endpoint is a URL, so the agent runs wherever it likes and the sandbox runs
//! wherever it likes. Any design that shells out to a locally-installed control-plane binary quietly
//! requires the agent to sit on the compute node beside the workloads it is meant to be isolated from,
//! which is backwards.
//!
//! ## The protocol (v1.1)
//!
//! One `POST` to the configured URL:
//!
//! ```jsonc
//! // request
//! { "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
//! // request carrying stdin (v1.1, optional) — standard base64, fed to the command, then closed
//! { "command": "cat", "args": [], "timeout_ms": 10000, "stdin_base64": "c3RkaW4tcHJvYmU=" }
//! // response — 200
//! { "exit_code": 0, "stdout": "…", "stderr": "…" }
//! ```
//!
//! Deliberately the smallest thing that can carry a command and its result. Putting an adapter in
//! front of a vendor's SDK is a few dozen lines on their side, and that seam is where vendor
//! specifics belong — not in here.
//!
//! **v1.1 is backward compatible.** `stdin_base64` is the only addition, and only `ShellFs`'s file
//! writes send it — after a probe (`cat` must echo `stdin-probe`) has shown the endpoint honors it. A
//! v1 endpoint ignores the unknown field and answers the probe with empty stdout, so writes fall back
//! to argv-sized chunks. The field is never trusted blindly: writing through an endpoint that dropped
//! it would silently produce *empty files*.
//!
//! **The response is capped** ([`DEFAULT_MAX_RESPONSE_BYTES`], `--exec-max-response-bytes`). The body
//! is read incrementally and reading stops at the cap; an oversized response is an **error**, never a
//! partial result. The body is one JSON object, so a prefix of it cannot be parsed — there is no
//! honest partial `ExecResult` to return (not even the exit code, which may come after the output),
//! and presenting whatever fit as the command's output would be a confident wrong answer. The error
//! says the command *did run*, so the caller narrows its output instead of blindly repeating a
//! command that may not be idempotent.
//!
//! For targets that have no HTTP surface at all (`ssh`, `docker exec`, `kubectl exec`, any CLI), see
//! [`TemplateRunner`], which is the same idea expressed as an argv template.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::tools::exec::{
    ChunkSink, CommandRunner, ExecResult, STREAM_HEAD, STREAM_TAIL, drain_capped, feed_stdin,
};

/// What the agent sends.
#[derive(Debug, Serialize)]
struct ExecRequest<'a> {
    command: &'a str,
    args: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<&'a str>,
    timeout_ms: u64,
    /// v1.1: bytes for the command's stdin, standard base64. Absent means no stdin — v1 behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    stdin_base64: Option<String>,
}

/// What the agent expects back. Extra fields are ignored, so an endpoint may return more.
#[derive(Debug, Deserialize)]
struct ExecResponse {
    exit_code: i32,
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
}

/// The default response-body cap for [`HttpExecRunner`]: 16 MiB.
///
/// Far above anything a tool consumes — `bash` keeps ~30 KB of output and the local runner keeps
/// 256 KiB per stream — so a legitimate response never meets it. What it bounds is the response that
/// isn't legitimate: a `cat` of a multi-gigabyte file must not be buffered whole into a process that
/// may be serving other sessions.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// A [`CommandRunner`] that POSTs each command to a URL.
///
/// Holds a `reqwest::Client` so connections are pooled across calls, but no session and no state — a
/// call is a request. There is nothing to reconnect, nothing to keep alive, and no lifecycle to get
/// wrong.
pub struct HttpExecRunner {
    url: String,
    client: reqwest::Client,
    headers: Vec<(String, String)>,
    max_response_bytes: usize,
}

impl HttpExecRunner {
    /// Point the runner at `url`.
    pub fn new(url: impl Into<String>) -> Result<Self, String> {
        let url = url.into();
        // Fail at construction rather than on the first tool call, so a typo is an immediate startup
        // error instead of a confusing mid-turn failure the model tries to work around.
        let parsed =
            reqwest::Url::parse(&url).map_err(|e| format!("invalid exec URL {url:?}: {e}"))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(format!(
                "exec URL must be http or https, got {:?}",
                parsed.scheme()
            ));
        }
        Ok(Self {
            url,
            client: {
                agent_core::ensure_provider();
                reqwest::Client::new()
            },
            headers: Vec::new(),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        })
    }

    /// Cap how much of a response body is read before the call fails. See the module doc for why an
    /// oversized response is an error rather than a truncated result.
    pub fn with_max_response_bytes(mut self, max: usize) -> Self {
        self.max_response_bytes = max;
        self
    }

    /// Add a header sent with every request — an `Authorization`, an API key, a tenant id. Every real
    /// endpoint needs some form of auth, and which form is the endpoint's business, not the agent's.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Parse a `Name: value` pair, the shape a CLI flag or env var carries.
    pub fn parse_header(raw: &str) -> Result<(String, String), String> {
        let (name, value) = raw
            .split_once(':')
            .ok_or_else(|| format!("header must be `Name: value`, got {raw:?}"))?;
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("header name is empty in {raw:?}"));
        }
        Ok((name.to_string(), value.trim().to_string()))
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// One request, one response: the whole protocol.
    async fn post(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecResponse> {
        let body = ExecRequest {
            command: program,
            args,
            cwd,
            timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
            stdin_base64: stdin.map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
        };
        let mut req = self.client.post(&self.url).json(&body);
        for (name, value) in &self.headers {
            req = req.header(name.as_str(), value.as_str());
        }
        // A little past the command's own budget, so the endpoint gets the chance to enforce and
        // report the timeout itself — its message is more useful than a bare client-side abort.
        let resp = req
            .timeout(timeout.saturating_add(Duration::from_secs(30)))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("exec endpoint {} timed out: {e}", self.url),
                    )
                } else {
                    std::io::Error::other(format!("exec endpoint {} unreachable: {e}", self.url))
                }
            })?;

        let status = resp.status();
        let (body, over_cap) = self.read_capped(resp).await?;
        let excerpt = || {
            String::from_utf8_lossy(&body)
                .trim()
                .chars()
                .take(400)
                .collect::<String>()
        };
        if !status.is_success() {
            // The endpoint refused the *request* — a bad URL, auth, a dead sandbox. Distinct from the
            // command running and exiting non-zero, which is an ordinary result.
            return Err(std::io::Error::other(format!(
                "exec endpoint {} returned {}: {}",
                self.url,
                status.as_u16(),
                excerpt()
            )));
        }
        if over_cap {
            return Err(std::io::Error::other(format!(
                "exec endpoint {}: the response exceeded the {}-byte cap and was discarded — the \
                 command ran, but its output is too large to return; narrow it (e.g. pipe through \
                 `head`) rather than re-running it unchanged",
                self.url, self.max_response_bytes
            )));
        }
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::other(format!(
                "exec endpoint {} returned unparseable JSON ({e}): {}",
                self.url,
                excerpt()
            ))
        })
    }

    /// Read the body incrementally, stopping at the cap instead of buffering whatever the endpoint
    /// sends. `true` means the cap was hit and the bytes returned are only a prefix.
    async fn read_capped(&self, mut resp: reqwest::Response) -> std::io::Result<(Vec<u8>, bool)> {
        let cap = self.max_response_bytes;
        // A declared length over the cap fails before a byte of it is read.
        let declared = resp.content_length().unwrap_or(0);
        if declared > cap as u64 {
            return Ok((Vec::new(), true));
        }
        let mut body = Vec::with_capacity(declared as usize);
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| std::io::Error::other(format!("exec endpoint {}: {e}", self.url)))?
        {
            if chunk.len() > cap - body.len() {
                // Dropping `resp` abandons the rest of the body along with its connection.
                return Ok((body, true));
            }
            body.extend_from_slice(&chunk);
        }
        Ok((body, false))
    }
}

#[async_trait]
impl CommandRunner for HttpExecRunner {
    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        // Sent as `stdin_base64`. A v1 endpoint ignores it and runs the command with no input, so an
        // `Ok` here is not proof of delivery — see the module doc on the probe.
        let parsed = self.post(program, args, cwd, timeout, Some(stdin)).await?;
        Ok(ExecResult {
            code: Some(parsed.exit_code),
            stdout: parsed.stdout,
            stderr: parsed.stderr,
            ..Default::default()
        })
    }

    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let parsed = self.post(program, args, cwd, timeout, None).await?;
        Ok(ExecResult {
            code: Some(parsed.exit_code),
            signal: None,
            stdout: parsed.stdout,
            stderr: parsed.stderr,
            timed_out: false,
            truncated: false,
        })
    }

    // No `run_streaming` override, deliberately. One request, one response: there is nothing to
    // stream, so the default — which never calls the sink — is the honest answer, and `bash` then
    // builds its output from the final `stdout` *and* `stderr`. An override that fed the sink stdout
    // alone told `bash` the output had streamed, and every remote command that printed to stdout lost
    // its stderr.
}

/// A [`CommandRunner`] that runs each command through a caller-supplied **argv template**, for targets
/// with no HTTP surface: `ssh`, `docker exec`, `kubectl exec`, `podman`, a vendor's own CLI.
///
/// The template is a list of arguments; the literal `{}` element is replaced by the command and its
/// arguments. For example `["ssh", "build-host", "--", "{}"]` or
/// `["docker", "exec", "my-container", "{}"]`.
///
/// **The command is never rendered into a string.** `{}` expands to *multiple argv entries*, one per
/// token, so a path containing spaces, quotes or `;` stays a single argument on the far side and
/// cannot be reparsed as syntax. A template that pasted the command into a shell string would make
/// every model-supplied path an injection vector.
pub struct TemplateRunner {
    template: Vec<String>,
}

impl TemplateRunner {
    /// Build from a template that must contain exactly one `{}` placeholder.
    pub fn new(template: Vec<String>) -> Result<Self, String> {
        let holes = template.iter().filter(|t| *t == "{}").count();
        if holes != 1 {
            return Err(format!(
                "exec template needs exactly one `{{}}` placeholder for the command, found {holes} \
                 in {template:?}"
            ));
        }
        if template.first().is_none_or(|p| p == "{}") {
            return Err("exec template must start with a program to run".to_string());
        }
        Ok(Self { template })
    }

    /// Parse a whitespace-separated template, the shape a CLI flag carries.
    pub fn parse(raw: &str) -> Result<Self, String> {
        Self::new(raw.split_whitespace().map(str::to_string).collect())
    }

    fn expand(&self, program: &str, args: &[String]) -> Vec<String> {
        let mut out = Vec::with_capacity(self.template.len() + args.len());
        for part in &self.template {
            if part == "{}" {
                out.push(program.to_string());
                out.extend(args.iter().cloned());
            } else {
                out.push(part.clone());
            }
        }
        out
    }
}

#[async_trait]
impl CommandRunner for TemplateRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, timeout, None).await
    }

    /// Piped into the *transport* process. Whether it reaches the far side is the transport's
    /// business: `ssh` forwards stdin, but `docker exec`/`kubectl exec` do so only with `-i`. Without
    /// it the far command reads nothing, which `ShellFs`'s probe detects — writes then use argv-sized
    /// chunks instead.
    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, timeout, Some(stdin)).await
    }
}

impl TemplateRunner {
    async fn exec(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecResult> {
        // `cwd` is expressed by wrapping the command, not by setting the *local* process's directory —
        // the directory belongs to the far side. Positional parameters to a fixed script, never
        // substituted text.
        let (program, args): (String, Vec<String>) = match cwd {
            Some(dir) => {
                let mut v = vec![
                    "-c".to_string(),
                    r#"cd "$1" || exit 1; shift; exec "$@""#.to_string(),
                    "sh".to_string(),
                    dir.to_string(),
                    program.to_string(),
                ];
                v.extend(args.iter().cloned());
                ("sh".to_string(), v)
            }
            None => (program.to_string(), args.to_vec()),
        };
        let argv = self.expand(&program, &args);

        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn()?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // Fed concurrently with the drains, never before them — see `feed_stdin`.
        let feed = feed_stdin(child.stdin.take(), stdin);
        // `wait_with_output` used to collect both pipes here, which buffers *everything* the far side
        // prints into this process — a `cat` of a large file on the target, or a build's log, sized
        // the replica's memory rather than the command. Worse, it then reported `truncated: false`
        // unconditionally, so a caller had no way to know output had been lost when it was.
        //
        // `drain_capped` is the same streaming head+tail accumulator the local runner uses: memory
        // is bounded at `STREAM_HEAD + STREAM_TAIL` per stream regardless of how much arrives, and
        // it says honestly whether a middle was dropped. Not `HttpExecRunner::read_capped` — that
        // *discards* an over-cap body and errors, which is right for a response envelope that must
        // parse as a whole, and wrong here: a long-running command's output is still the answer.
        //
        // Both pipes are drained *concurrently* with the wait. A child that fills one pipe's OS
        // buffer while we read only the other deadlocks, and an unread pipe stalls its exit.
        let (exited_tx, exited_rx) = tokio::sync::watch::channel(false);
        let wait = async {
            let status = child.wait().await;
            let _ = exited_tx.send(true);
            status
        };
        let collect = async {
            tokio::join!(
                wait,
                drain_capped(stdout, STREAM_HEAD, STREAM_TAIL, None, exited_rx.clone()),
                drain_capped(stderr, STREAM_HEAD, STREAM_TAIL, None, exited_rx.clone()),
                feed,
            )
        };
        match tokio::time::timeout(timeout, collect).await {
            Ok((status, (stdout, out_truncated), (stderr, err_truncated), fed)) => {
                let status = status?;
                fed?;
                Ok(ExecResult {
                    code: status.code(),
                    // The transport's own exit signal, not the far command's — `ssh` reports a
                    // remote signal as an exit code, and a local `docker exec` killed by one says
                    // nothing about what ran inside. Left `None` rather than guessing.
                    signal: None,
                    stdout: String::from_utf8_lossy(&stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&stderr).into_owned(),
                    timed_out: false,
                    truncated: out_truncated || err_truncated,
                })
            }
            Err(_) => Ok(ExecResult {
                code: None,
                signal: None,
                stdout: String::new(),
                stderr: format!("exec template timed out after {timeout:?}"),
                timed_out: true,
                truncated: false,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn a_url_must_be_http_or_https() {
        assert!(HttpExecRunner::new("https://exec.example/run").is_ok());
        assert!(HttpExecRunner::new("http://127.0.0.1:9000/exec").is_ok());
        assert!(HttpExecRunner::new("file:///etc/passwd").is_err());
        assert!(HttpExecRunner::new("not a url").is_err());
    }

    #[test]
    fn headers_parse_as_name_colon_value() {
        let (n, v) = HttpExecRunner::parse_header("Authorization: Bearer abc").unwrap();
        assert_eq!(n, "Authorization");
        assert_eq!(v, "Bearer abc");
        assert!(HttpExecRunner::parse_header("no-colon").is_err());
        assert!(HttpExecRunner::parse_header(": empty-name").is_err());
    }

    #[test]
    fn a_template_needs_exactly_one_placeholder() {
        assert!(TemplateRunner::parse("ssh host -- {}").is_ok());
        assert!(TemplateRunner::parse("ssh host").is_err());
        assert!(TemplateRunner::parse("ssh {} {}").is_err());
        assert!(TemplateRunner::parse("{} foo").is_err());
    }

    #[test]
    fn the_placeholder_expands_to_argv_entries_not_a_string() {
        // The injection invariant. A path containing `;` and quotes must stay ONE argument on the far
        // side; if this ever renders to a string, the far side reparses it as syntax.
        let r = TemplateRunner::parse("docker exec ctr {}").unwrap();
        let argv = r.expand(
            "grep",
            &["-e".into(), "'; rm -rf / #".into(), "/a b.txt".into()],
        );
        assert_eq!(
            argv,
            vec![
                "docker",
                "exec",
                "ctr",
                "grep",
                "-e",
                "'; rm -rf / #",
                "/a b.txt"
            ]
        );
    }

    #[tokio::test]
    async fn a_template_runner_actually_runs_the_command() {
        // `env` is a harmless stand-in for a real transport: it runs whatever follows, which is
        // exactly the shape `ssh host --` / `docker exec ctr` have.
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run(
                "printf",
                &["%s".into(), "hello".into()],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(out.code, Some(0));
        assert_eq!(out.stdout, "hello");
    }

    #[tokio::test]
    async fn a_template_runner_honors_cwd_without_splicing_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run("ls", &[], dir.path().to_str(), Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(out.code, Some(0));
        assert!(out.stdout.contains("marker.txt"), "{out:?}");
    }

    /// The transport's output is bounded by the capture window and truncation is reported honestly.
    /// `wait_with_output` buffered the whole thing and hardcoded `truncated: false`, so a command
    /// that printed a gigabyte sized this process — and said nothing about what it had lost.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_template_runner_caps_its_output_and_says_when_it_truncated() {
        let r = TemplateRunner::parse("sh -c {}").unwrap();
        let bytes = 2 * (STREAM_HEAD + STREAM_TAIL);
        let out = r
            .run(
                &format!("head -c {bytes} /dev/zero | tr '\\0' 'a'"),
                &[],
                None,
                Duration::from_secs(30),
            )
            .await
            .unwrap();

        assert_eq!(out.code, Some(0), "{out:?}");
        assert!(out.truncated, "the middle was dropped, so say so: {out:?}");
        assert!(
            out.stdout.len() <= STREAM_HEAD + STREAM_TAIL,
            "kept {} bytes of a {bytes}-byte stream",
            out.stdout.len()
        );
    }

    /// …and output that fits is returned whole, still flagged untruncated.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_template_runner_leaves_output_under_the_cap_alone() {
        let r = TemplateRunner::parse("sh -c {}").unwrap();
        let out = r
            .run(
                "printf 'hello\\n'; printf 'oops\\n' >&2",
                &[],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(out.stdout, "hello\n");
        assert_eq!(out.stderr, "oops\n");
        assert!(!out.truncated, "{out:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Per-session targets, for a multi-tenant server.
// ---------------------------------------------------------------------------------------------

/// One configured target: the runner, and the filesystem backend built over it.
///
/// They are constructed together and handed out together because they must always name the *same*
/// machine — a `bash` on one and an `edit` on another is not a partial configuration, it is a bug.
#[derive(Clone)]
pub struct ExecTarget {
    runner: std::sync::Arc<dyn CommandRunner>,
    backend: std::sync::Arc<dyn crate::tools::fs::FsBackend>,
    caps: crate::tools::fs::shell::Capabilities,
}

impl ExecTarget {
    /// Build from an HTTP endpoint plus `Name: value` headers, reading at most `max_response_bytes` of
    /// any one response ([`DEFAULT_MAX_RESPONSE_BYTES`] unless configured).
    pub async fn http(
        url: &str,
        headers: &[String],
        max_response_bytes: usize,
    ) -> Result<Self, String> {
        let mut runner = HttpExecRunner::new(url)?.with_max_response_bytes(max_response_bytes);
        for raw in headers {
            let (name, value) = HttpExecRunner::parse_header(raw)?;
            runner = runner.with_header(name, value);
        }
        Ok(Self::over(std::sync::Arc::new(runner)).await)
    }

    /// Build from an argv template (`ssh host -- {}`, `docker exec ctr {}`, …).
    pub async fn template(template: &str) -> Result<Self, String> {
        Ok(Self::over(std::sync::Arc::new(TemplateRunner::parse(template)?)).await)
    }

    /// Wrap an already-built runner, probing it for the capability rung.
    pub async fn over(runner: std::sync::Arc<dyn CommandRunner>) -> Self {
        Self::over_with_home(runner, None).await
    }

    /// [`Self::over`], told the target's `$HOME` up front so a model-supplied `~/notes.md` expands
    /// against the sandbox's own home instead of being left alone. Service mode learns the home from
    /// its startup probe (`crate::service::ServiceSession::connect_exec`), which is the only caller
    /// that can know it before the first tool call.
    pub async fn over_with_home(
        runner: std::sync::Arc<dyn CommandRunner>,
        home: Option<String>,
    ) -> Self {
        let backend = crate::tools::fs::shell::ShellFs::connect(runner.clone()).await;
        let backend = match home {
            Some(home) => backend.with_home(home),
            None => backend,
        };
        let caps = backend.capabilities();
        Self {
            runner,
            backend: std::sync::Arc::new(backend),
            caps,
        }
    }

    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        self.runner.clone()
    }

    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        self.backend.clone()
    }

    /// Which search rung this target resolved to — reported to the client so a degraded box (no
    /// `rg`, or busybox) is visible at attach rather than inferred from bad results later.
    pub fn search_engine(&self) -> crate::tools::fs::shell::SearchEngine {
        self.caps.search_engine()
    }
}

/// A **per-session** exec target that can be re-pointed at runtime.
///
/// This exists because of a specific hazard. `serve`'s tool registry is rebuilt only when the model
/// or thinking level changes — a `switch_session` between two sessions on the same model does *not*
/// rebuild it. So a target bound at registry-construction time would leave the new session talking to
/// the previous session's machine. In a multi-tenant server that is not a stale-config bug, it is one
/// tenant's tools operating inside another tenant's sandbox.
///
/// The cell closes that by construction: the tools hold *this*, and read it on every call, so a
/// re-point takes effect immediately and cannot be missed by a skipped rebuild. It is the same shape
/// `memory::file::SessionDir` already uses for the per-session `/session` mount, for the same reason.
///
/// An empty cell means the host, so one registry serves both local and remote sessions — unless the
/// cell is [strict](Self::strict), in which case an empty cell is an error and there is no host to
/// fall back to.
#[derive(Clone, Default)]
pub struct ExecCell {
    target: std::sync::Arc<std::sync::RwLock<Option<ExecTarget>>>,
    /// Fail closed: an empty cell errors instead of running on this host. Service mode
    /// (`serve --service`) sets it, because there the host is the *replica* — a tenant's `bash`
    /// landing on it is not a stale-config bug but a breach. Fixed at construction.
    strict: bool,
}

/// What a strict cell answers when nothing is configured. Static text: it reaches the model as a
/// tool error, so it says what to do, not what went wrong internally.
const NO_TARGET: &str = "no sandbox is attached to this session";

impl ExecCell {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cell with **no host fallback**: until a target is set, every command and every filesystem
    /// call fails, rather than running on this host. Service mode (`serve --service`) uses this,
    /// because there "this host" is the *replica* — a tenant's `bash` landing on it is not a
    /// stale-config bug but a breach.
    pub fn strict() -> Self {
        Self {
            target: std::sync::Arc::default(),
            strict: true,
        }
    }

    /// Re-point at `target`, or back to the host with `None`. Takes effect on the next tool call.
    pub fn set(&self, target: Option<ExecTarget>) {
        // Recover a poisoned lock rather than panicking: the cell holds a single value with no
        // invariant a panicked writer could leave half-updated, and the workspace forbids `unwrap`.
        *self.target.write().unwrap_or_else(|e| e.into_inner()) = target;
    }

    pub fn get(&self) -> Option<ExecTarget> {
        self.target
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The target to dispatch to, or — in strict mode with an empty cell — the refusal.
    fn resolve(&self) -> Result<Option<ExecTarget>, &'static str> {
        match (self.get(), self.strict) {
            (Some(t), _) => Ok(Some(t)),
            (None, false) => Ok(None),
            (None, true) => Err(NO_TARGET),
        }
    }

    /// A [`CommandRunner`] that always dispatches to whatever this cell currently holds.
    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        std::sync::Arc::new(CellRunner(self.clone()))
    }

    /// An [`FsBackend`](crate::tools::fs::FsBackend) that always dispatches to whatever this cell
    /// currently holds.
    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        std::sync::Arc::new(CellFs(self.clone()))
    }
}

/// Dispatches each command to the cell's current target, or this host when it is empty.
struct CellRunner(ExecCell);

#[async_trait]
impl CommandRunner for CellRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        match self.0.resolve().map_err(std::io::Error::other)? {
            Some(t) => t.runner().run(program, args, cwd, timeout).await,
            None => {
                crate::tools::exec::RealRunner
                    .run(program, args, cwd, timeout)
                    .await
            }
        }
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.0.resolve().map_err(std::io::Error::other)? {
            Some(t) => {
                t.runner()
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
        }
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        match self.0.resolve().map_err(std::io::Error::other)? {
            Some(t) => {
                t.runner()
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
        }
    }
}

/// The filesystem half of the same indirection.
struct CellFs(ExecCell);

impl CellFs {
    fn inner(
        &self,
    ) -> Result<std::sync::Arc<dyn crate::tools::fs::FsBackend>, crate::tools::fs::FsError> {
        match self.0.resolve() {
            Ok(Some(t)) => Ok(t.backend()),
            Ok(None) => Ok(std::sync::Arc::new(crate::tools::fs::local::LocalFs::new())),
            Err(e) => Err(crate::tools::fs::FsError::Backend(e.to_owned())),
        }
    }
}

#[async_trait]
impl crate::tools::fs::FsBackend for CellFs {
    /// Not fallible, so a strict empty cell answers with the world it *will* have: remote, with no
    /// home known yet. Never `Local` — a tenant's path must not resolve against the replica even in
    /// the window before its sandbox is attached.
    fn world(&self) -> crate::tools::fs::PathWorld {
        match self.inner() {
            Ok(b) => b.world(),
            Err(_) => crate::tools::fs::PathWorld::Remote { home: None },
        }
    }
    async fn search(
        &self,
        q: &crate::tools::fs::SearchQuery,
    ) -> Result<crate::tools::fs::SearchOutcome, crate::tools::fs::FsError> {
        self.inner()?.search(q).await
    }
    async fn stat(
        &self,
        path: &std::path::Path,
    ) -> Result<Option<crate::tools::fs::Meta>, crate::tools::fs::FsError> {
        self.inner()?.stat(path).await
    }
    async fn read_bytes(
        &self,
        path: &std::path::Path,
        offset: u64,
        max: usize,
    ) -> Result<Vec<u8>, crate::tools::fs::FsError> {
        self.inner()?.read_bytes(path, offset, max).await
    }
    async fn write_bytes(
        &self,
        path: &std::path::Path,
        bytes: &[u8],
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner()?.write_bytes(path, bytes).await
    }
    async fn write_if_unchanged(
        &self,
        path: &std::path::Path,
        bytes: &[u8],
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, crate::tools::fs::FsError> {
        self.inner()?
            .write_if_unchanged(path, bytes, expected)
            .await
    }
    async fn create_dir_all(
        &self,
        path: &std::path::Path,
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner()?.create_dir_all(path).await
    }
    async fn list_dir(
        &self,
        path: &std::path::Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<crate::tools::fs::DirEntry>, crate::tools::fs::FsError> {
        self.inner()?.list_dir(path, cap, include_hidden).await
    }
    async fn glob(
        &self,
        q: &crate::tools::fs::GlobQuery,
    ) -> Result<crate::tools::fs::GlobOutcome, crate::tools::fs::FsError> {
        self.inner()?.glob(q).await
    }
}
