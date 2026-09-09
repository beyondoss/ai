//! Reporting the outcome of a run to an endpoint of the operator's choosing.
//!
//! An agent that finishes a long task with nobody attached has no way to say so.
//! Whatever dispatched the work is left holding a connection open and watching
//! for the run to end — and that watcher can fail on its own, through a job
//! timeout, a dropped connection or a restart, while the run is perfectly
//! healthy. The outcome is then simply lost.
//!
//! So the agent states its own outcome. One HTTP POST, to a URL it is handed in
//! its environment, when a run reaches a terminal state.
//!
//! # Configuration
//!
//! | variable | meaning |
//! |---|---|
//! | `AGENT_RUN_EVENTS_URL` | Where to POST. Unset disables reporting entirely. |
//! | `AGENT_RUN_EVENTS_TOKEN` | Optional. Sent as `Authorization: Bearer <token>`. |
//! | `AGENT_RUN_ID` | Optional. Echoed back as `run_id` so a receiver can correlate. |
//!
//! `AGENT_RUN_EVENTS_URL` accepts:
//!
//! - `http://…` and `https://…` — an ordinary endpoint.
//! - `unix:/path/to.sock` (or `unix:///path/to.sock`) — a Unix socket, for a
//!   supervisor sharing the machine with the agent. There is no host to name and
//!   no TLS to terminate on a socket only the local process can open, which is
//!   why it is worth supporting separately rather than forcing a loopback port.
//!
//! # The body
//!
//! A single flat JSON object, stable and small:
//!
//! ```json
//! { "run_id": "…", "status": "succeeded", "summary": "…", "error": null }
//! ```
//!
//! `status` is `succeeded` or `failed`; `error` is present only on `failed`.
//! `summary` is the agent's own closing words — the last thing it said, with any
//! reasoning trace excluded. It is there so a receiver can show a person what the
//! agent did without opening a session to ask: a run report that says only "it
//! finished" makes whoever dispatched the work go and fetch the answer
//! separately, which is the round trip this whole mechanism removes.
//! Fields are added, never repurposed — a receiver that ignores what it does not
//! recognise keeps working across versions.
//!
//! # It never affects the run
//!
//! Unset configuration means no I/O at all. A refused, slow or broken endpoint is
//! logged and dropped: the work is finished either way, and failing a run because
//! a listener was down would turn a reporting problem into a work problem.

use std::time::Duration;

use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tracing::debug;

/// Where to report. Unset disables reporting.
const ENV_URL: &str = "AGENT_RUN_EVENTS_URL";

/// Optional bearer token for an HTTP endpoint. Ignored for `unix:` targets,
/// which are already protected by filesystem permissions.
const ENV_TOKEN: &str = "AGENT_RUN_EVENTS_TOKEN";

/// Optional identifier echoed back, so a receiver can correlate the report with
/// whatever it dispatched. The agent neither generates nor interprets it.
const ENV_RUN_ID: &str = "AGENT_RUN_ID";

/// A bound on pathology, not on the normal case: a local socket answers in well
/// under a millisecond. A receiver that stops reading must never hold up an agent
/// that has already finished its work.
const TIMEOUT: Duration = Duration::from_secs(5);

/// How a run ended.
///
/// Two states, because those are the two an agent can report without guessing.
/// A run that is still going is the dispatcher's to know — it started it — and
/// anything finer (cancelled, awaiting input) needs a signal this crate does not
/// yet produce. Adding one later is additive: receivers ignore what they do not
/// recognise.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// The run ran to completion.
    Succeeded,
    /// The run did not complete. `error` carries the account of why.
    Failed,
}

#[derive(Debug, Serialize)]
struct Report<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

/// Cap on the summary.
///
/// Bounded by what a person will read, not by what the model produced: this
/// becomes the body of a reply somewhere, and an unbounded field on a report is
/// also an unbounded write into whoever receives it.
const SUMMARY_LIMIT: usize = 4000;

/// The agent's closing words, for [`report_terminal`]'s `summary`.
///
/// The last assistant turn's visible text, and only that:
///
/// - Reasoning is excluded. `Thinking` and `RedactedThinking` are their own
///   content blocks, so taking `Text` alone drops them by construction. The
///   trace is not the answer, and posting it into somebody's chat thread would
///   be verbose, confusing, and not what the agent chose to say.
/// - Truncated on a character boundary, so a multi-byte glyph is never split.
///
/// `None` when the turn produced no prose — a run that ends in a tool call and
/// nothing else has no closing words, and an empty string would be a worse
/// answer than an absent field.
pub fn summarize(messages: &[agent_core::Message]) -> Option<String> {
    let last = messages
        .iter()
        .rev()
        .find(|m| m.role == agent_core::Role::Assistant)?;

    let mut out = String::new();
    for block in &last.content {
        if let agent_core::ContentBlock::Text { text, .. } = block {
            out.push_str(text);
        }
    }

    let trimmed = out.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() <= SUMMARY_LIMIT {
        return Some(trimmed.to_string());
    }
    let cut = (0..=SUMMARY_LIMIT)
        .rev()
        .find(|&i| trimmed.is_char_boundary(i))
        .unwrap_or(0);
    Some(trimmed[..cut].to_string())
}

/// Where a report goes, once the environment has been read.
enum Target {
    /// A Unix socket path.
    Unix(std::path::PathBuf),
    /// An absolute `http`/`https` URL.
    Http(String),
}

/// Parse `AGENT_RUN_EVENTS_URL`.
///
/// `unix:` is matched before anything else because it is not a hierarchical URL:
/// `unix:/run/x.sock` has no authority component, and treating it as one would
/// read `run` as a hostname. Both the two- and three-slash spellings are accepted
/// since both are in common use for socket URLs.
fn parse_target(raw: &str) -> Option<Target> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some(rest) = raw.strip_prefix("unix:") {
        let path = rest.trim_start_matches("//");
        if path.is_empty() {
            return None;
        }
        return Some(Target::Unix(std::path::PathBuf::from(path)));
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        return Some(Target::Http(raw.to_string()));
    }
    None
}

/// Report a terminal outcome, if an endpoint is configured.
///
/// Callers on a hot path should spawn this rather than await it: the run is over
/// and its own response has already gone out, so the report must not add latency.
///
/// Never fails. Every error is logged at debug and dropped.
pub async fn report_terminal(status: RunStatus, summary: Option<&str>, error: Option<&str>) {
    let Some(raw) = std::env::var(ENV_URL).ok() else {
        return;
    };
    let Some(target) = parse_target(&raw) else {
        debug!(
            value = %raw,
            "{ENV_URL} is not a supported target (expected http://, https:// or unix:/path)"
        );
        return;
    };

    let report = Report {
        run_id: std::env::var(ENV_RUN_ID).ok().filter(|v| !v.is_empty()),
        status,
        summary,
        error,
    };
    let body = match serde_json::to_vec(&report) {
        Ok(b) => b,
        Err(e) => {
            debug!(error = %e, "run report: could not encode");
            return;
        }
    };
    let token = std::env::var(ENV_TOKEN).ok().filter(|v| !v.is_empty());

    let sent = tokio::time::timeout(TIMEOUT, send(&target, &body, token.as_deref())).await;
    match sent {
        Ok(Ok(())) => debug!(?status, "run report delivered"),
        Ok(Err(e)) => debug!(error = %e, "run report not delivered"),
        Err(_) => debug!(secs = TIMEOUT.as_secs(), "run report timed out"),
    }
}

async fn send(target: &Target, body: &[u8], token: Option<&str>) -> std::io::Result<()> {
    match target {
        Target::Unix(path) => post_unix(path, body).await,
        Target::Http(url) => post_http(url, body, token).await,
    }
}

/// POST over a Unix socket, written by hand.
///
/// An HTTP client is built around hosts, pools, redirects and TLS, none of which
/// exist here: it is one fixed request to a local socket with a single caller.
/// `Connection: close` lets the receiver drop the socket as soon as it has the
/// body, and reading the status line back is the point of using HTTP at all —
/// a 2xx means the report was accepted, anything else is visible in the log
/// instead of vanishing the way a fire-and-forget line would.
async fn post_unix(path: &std::path::Path, body: &[u8]) -> std::io::Result<()> {
    let mut stream = UnixStream::connect(path).await?;

    let head = format!(
        "POST /run HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;

    // Enough for the status line; the rest of the response is of no interest.
    let mut buf = [0u8; 64];
    let n = stream.read(&mut buf).await?;
    let head = String::from_utf8_lossy(&buf[..n]);
    if is_2xx(&head) {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "endpoint rejected the run report: {}",
            head.lines().next().unwrap_or("<no status line>")
        )))
    }
}

/// True for a response head whose status line carries a 2xx code.
fn is_2xx(head: &str) -> bool {
    head.split_once(' ')
        .and_then(|(v, rest)| v.starts_with("HTTP/").then_some(rest))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|code| code.parse::<u16>().ok())
        .is_some_and(|code| (200..300).contains(&code))
}

async fn post_http(url: &str, body: &[u8], token: Option<&str>) -> std::io::Result<()> {
    let client = reqwest::Client::new();
    let mut req = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_vec());
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let status = resp.status();
    if status.is_success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "endpoint rejected the run report: {status}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::UnixListener;

    #[test]
    fn unix_targets_are_not_parsed_as_hierarchical_urls() {
        // The bug this guards: `unix:/run/x.sock` has no authority, so a general
        // URL parse reads `run` as a host and loses the leading slash.
        assert!(matches!(
            parse_target("unix:/run/agent/events.sock"),
            Some(Target::Unix(p)) if p == std::path::Path::new("/run/agent/events.sock")
        ));
        assert!(matches!(
            parse_target("unix:///run/agent/events.sock"),
            Some(Target::Unix(p)) if p == std::path::Path::new("/run/agent/events.sock")
        ));
    }

    #[test]
    fn http_targets_pass_through_and_junk_is_rejected() {
        assert!(matches!(
            parse_target("https://example.test/hook"),
            Some(Target::Http(_))
        ));
        assert!(matches!(
            parse_target("http://example.test/hook"),
            Some(Target::Http(_))
        ));
        // Neither scheme, so there is nothing sensible to do with it — better to
        // log once at startup of the run than to invent a default.
        assert!(parse_target("example.test/hook").is_none());
        assert!(parse_target("ftp://example.test").is_none());
        assert!(parse_target("   ").is_none());
        assert!(parse_target("unix:").is_none());
    }

    #[test]
    fn only_2xx_counts_as_accepted() {
        assert!(is_2xx("HTTP/1.1 204 No Content\r\n"));
        assert!(is_2xx("HTTP/1.0 200 OK\r\n"));
        assert!(!is_2xx("HTTP/1.1 400 Bad Request\r\n"));
        assert!(!is_2xx("HTTP/1.1 500 Internal Server Error\r\n"));
        assert!(!is_2xx("garbage"));
    }

    #[test]
    fn error_is_omitted_on_success_and_present_on_failure() {
        let ok = serde_json::to_value(Report {
            run_id: None,
            status: RunStatus::Succeeded,
            summary: Some("did the thing"),
            error: None,
        })
        .unwrap();
        assert_eq!(ok["status"], "succeeded");
        assert!(ok.get("error").is_none(), "no error key on success: {ok}");
        assert!(ok.get("run_id").is_none(), "no run_id when unset: {ok}");

        let bad = serde_json::to_value(Report {
            run_id: Some("r1".into()),
            status: RunStatus::Failed,
            summary: None,
            error: Some("boom"),
        })
        .unwrap();
        assert_eq!(bad["status"], "failed");
        assert_eq!(bad["error"], "boom");
        assert_eq!(bad["run_id"], "r1");
    }

    fn assistant(blocks: Vec<agent_core::ContentBlock>) -> agent_core::Message {
        agent_core::Message::assistant(blocks)
    }

    fn text(t: &str) -> agent_core::ContentBlock {
        agent_core::ContentBlock::text(t)
    }

    #[test]
    fn summary_is_the_last_assistant_turn_only() {
        let msgs = vec![
            assistant(vec![text("an earlier answer")]),
            agent_core::Message::user("and then?"),
            assistant(vec![text("the final answer")]),
        ];
        assert_eq!(summarize(&msgs).as_deref(), Some("the final answer"));
    }

    #[test]
    fn summary_excludes_the_reasoning_trace() {
        // The trace is not the answer. Posting it into somebody's chat thread
        // would be verbose, confusing, and not what the agent chose to say — and
        // it is excluded by construction, because thinking is its own block type.
        let msgs = vec![assistant(vec![
            agent_core::ContentBlock::Thinking {
                text: "let me consider the options".into(),
                signature: String::new(),
            },
            text("Done — renamed the field."),
        ])];
        let got = summarize(&msgs).expect("prose is present");
        assert_eq!(got, "Done — renamed the field.");
        assert!(
            !got.contains("consider"),
            "reasoning leaked into the summary: {got}"
        );
    }

    #[test]
    fn a_turn_with_no_prose_has_no_summary() {
        // A run that ends in a tool call and nothing else has no closing words;
        // an empty string would be a worse answer than an absent field.
        assert!(summarize(&[assistant(vec![])]).is_none());
        assert!(summarize(&[assistant(vec![text("   ")])]).is_none());
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn summary_truncation_never_splits_a_character() {
        // A naive byte slice at the limit panics or produces invalid UTF-8 the
        // moment the boundary lands mid-glyph.
        let long = "é".repeat(SUMMARY_LIMIT);
        let got = summarize(&[assistant(vec![text(&long)])]).expect("some prose");
        assert!(
            got.len() <= SUMMARY_LIMIT,
            "must respect the cap: {}",
            got.len()
        );
        assert!(got.chars().all(|c| c == 'é'), "must not split a character");
    }

    /// The whole wire contract against a real socket: a POST a receiver can
    /// parse, with a length it can trust.
    #[tokio::test]
    async fn posts_a_report_a_receiver_can_read() {
        let dir = std::env::temp_dir().join(format!("run-events-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("ok.sock");
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();

        let server = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut got = Vec::new();
            let mut chunk = [0u8; 512];
            loop {
                match s.read(&mut chunk).await.unwrap() {
                    0 => break,
                    n => {
                        got.extend_from_slice(&chunk[..n]);
                        if got.windows(4).any(|w| w == b"\r\n\r\n") && got.ends_with(b"}") {
                            break;
                        }
                    }
                }
            }
            s.write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            String::from_utf8(got).unwrap()
        });

        let body = serde_json::to_vec(&Report {
            run_id: Some("r1".into()),
            status: RunStatus::Failed,
            summary: None,
            error: Some("boom"),
        })
        .unwrap();
        post_unix(&sock, &body).await.unwrap();

        let raw = server.await.unwrap();
        let (head, json) = raw.split_once("\r\n\r\n").expect("head and body");
        assert!(head.starts_with("POST "), "must be a POST: {head}");
        assert!(
            head.contains(&format!("Content-Length: {}", body.len())),
            "must declare its length so the receiver knows when the body ends: {head}"
        );
        let v: serde_json::Value = serde_json::from_str(json).expect("body is JSON");
        assert_eq!(v["status"], "failed");
        assert_eq!(v["error"], "boom");

        let _ = std::fs::remove_file(&sock);
    }

    /// A non-2xx is an error the caller can log, not a silent success — a
    /// producer that ignored it would keep sending the same broken thing.
    #[tokio::test]
    async fn a_rejected_report_is_an_error() {
        let dir = std::env::temp_dir().join(format!("run-events-rej-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("bad.sock");
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();

        tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut chunk = [0u8; 512];
            let _ = s.read(&mut chunk).await;
            let _ = s
                .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
                .await;
        });

        let err = post_unix(&sock, b"{}")
            .await
            .expect_err("400 must be an error");
        assert!(
            err.to_string().contains("400"),
            "the reason should survive into the log: {err}"
        );
        let _ = std::fs::remove_file(&sock);
    }
}
