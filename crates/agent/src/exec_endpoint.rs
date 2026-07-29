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
//! ## The protocol
//!
//! One `POST` to the configured URL:
//!
//! ```jsonc
//! // request
//! { "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
//! // response — 200
//! { "exit_code": 0, "stdout": "…", "stderr": "…" }
//! ```
//!
//! Deliberately the smallest thing that can carry a command and its result. Putting an adapter in
//! front of a vendor's SDK is a few dozen lines on their side, and that seam is where vendor
//! specifics belong — not in here.
//!
//! For targets that have no HTTP surface at all (`ssh`, `docker exec`, `kubectl exec`, any CLI), see
//! [`TemplateRunner`], which is the same idea expressed as an argv template.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::tools::exec::{ChunkSink, CommandRunner, ExecResult};

/// What the agent sends.
#[derive(Debug, Serialize)]
struct ExecRequest<'a> {
    command: &'a str,
    args: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<&'a str>,
    timeout_ms: u64,
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

/// A [`CommandRunner`] that POSTs each command to a URL.
///
/// Holds a `reqwest::Client` so connections are pooled across calls, but no session and no state — a
/// call is a request. There is nothing to reconnect, nothing to keep alive, and no lifecycle to get
/// wrong.
pub struct HttpExecRunner {
    url: String,
    client: reqwest::Client,
    headers: Vec<(String, String)>,
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
            client: reqwest::Client::new(),
            headers: Vec::new(),
        })
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
}

#[async_trait]
impl CommandRunner for HttpExecRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let body = ExecRequest {
            command: program,
            args,
            cwd,
            timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
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
        let text = resp
            .text()
            .await
            .map_err(|e| std::io::Error::other(format!("exec endpoint {}: {e}", self.url)))?;
        if !status.is_success() {
            // The endpoint refused the *request* — a bad URL, auth, a dead sandbox. Distinct from the
            // command running and exiting non-zero, which is an ordinary result.
            return Err(std::io::Error::other(format!(
                "exec endpoint {} returned {}: {}",
                self.url,
                status.as_u16(),
                text.trim().chars().take(400).collect::<String>()
            )));
        }
        let parsed: ExecResponse = serde_json::from_str(&text).map_err(|e| {
            std::io::Error::other(format!(
                "exec endpoint {} returned unparseable JSON ({e}): {}",
                self.url,
                text.trim().chars().take(400).collect::<String>()
            ))
        })?;
        Ok(ExecResult {
            code: Some(parsed.exit_code),
            signal: None,
            stdout: parsed.stdout,
            stderr: parsed.stderr,
            timed_out: false,
            truncated: false,
        })
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        // One request, one response — there is nothing to stream incrementally, so the whole result
        // reaches the sink at once. Overridden rather than inherited so this is stated rather than
        // implied by a default that looks like it streams.
        let result = self.run(program, args, cwd, timeout).await?;
        if !result.stdout.is_empty() {
            on_chunk(result.stdout.as_bytes());
        }
        Ok(result)
    }
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
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let out = match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(r) => r?,
            Err(_) => {
                return Ok(ExecResult {
                    code: None,
                    signal: None,
                    stdout: String::new(),
                    stderr: format!("exec template timed out after {timeout:?}"),
                    timed_out: true,
                    truncated: false,
                });
            }
        };
        Ok(ExecResult {
            code: out.status.code(),
            signal: None,
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            timed_out: false,
            truncated: false,
        })
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
}
