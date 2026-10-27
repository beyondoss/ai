//! Exec protocol **v1.3**: sandbox jobs as an endpoint capability, and **v1.4**'s `notify` on
//! `keep`. The normative spec is the parent module's doc ("Sandbox jobs (v1.3)", "Sandbox job notify
//! (v1.4)"); this is the client half — the probe that adopts them, and [`NativeJobs`], the
//! [`SandboxJobs`] that speaks them — plus [`HttpJobs`], which picks it or the baseline
//! ([`ShellJobs`]) per endpoint.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};

use super::HttpExecRunner;
use super::jobs::{
    EventSink, JobEvent, JobId, JobInfo, JobNotify, JobScope, JobSpec, JobStatus, ReadChunk,
    SandboxJobs, ShellJobs,
};
use super::stream::{FRAME_SILENCE_LIMIT, FrameReader};

/// What a v1.3 endpoint answers the probe with: `{"jobs": "v1.3"}`.
pub const CAPABILITY: &str = "v1.3";

/// What an endpoint that offers v1.4 adds to its probe answer: `"notify": "v1.4"`.
pub const NOTIFY_CAPABILITY: &str = "v1.4";

/// How long a job operation other than a stream may take.
const OP_TIMEOUT: Duration = Duration::from_secs(30);

/// The `SIGTERM` grace a kill asks for — the local runner's.
const KILL_GRACE_MS: u64 = 300;

/// The command every job operation but a start carries: harmless if an endpoint that does not know
/// `job` runs it.
const INERT: &str = "true";

/// What an endpoint's answer to the job probe offers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobOffer {
    /// Sandbox jobs natively (v1.3): `jobs` is exactly `"v1.3"`.
    pub native_jobs: bool,
    /// `notify` on a native `keep` (v1.4): `jobs` is exactly `"v1.3"` and `notify` exactly `"v1.4"`.
    pub notify: bool,
    /// `notify` for the baseline (v1.4): no `jobs`, and `notify` exactly `"v1.4"`. The baseline
    /// further needs `curl` and `stdin_base64` in the sandbox.
    pub baseline_notify: bool,
}

impl JobOffer {
    /// Read an answer to the probe. Only the spec's exact answers offer anything: a later version,
    /// a boolean, a `jobs` of another version, or a v1.1 envelope (the endpoint ran the command)
    /// offers nothing it does not name exactly.
    pub fn from_answer(answer: &Value) -> Self {
        let jobs = answer.get("jobs");
        let notify = answer.get("notify").and_then(Value::as_str) == Some(NOTIFY_CAPABILITY);
        let native_jobs = jobs.and_then(Value::as_str) == Some(CAPABILITY);
        Self {
            native_jobs,
            notify: native_jobs && notify,
            baseline_notify: jobs.is_none() && notify,
        }
    }
}

/// What the endpoint offers for jobs. The probe is a valid v1.1 request (`printf exec-v1.2`)
/// carrying `"job": {"op": "probe"}`: an endpoint without the capability ignores the unknown field
/// and runs the command, one with it answers `{"jobs": "v1.3"}` (plus `"notify": "v1.4"` when it
/// offers notify), and one that offers notify only for the baseline answers `{"notify": "v1.4"}`.
pub(crate) async fn probe(runner: &HttpExecRunner) -> std::io::Result<JobOffer> {
    let job = json!({ "op": "probe" });
    let resp = runner
        .send_job(
            "sh",
            &["-c".to_string(), "printf exec-v1.2".to_string()],
            None,
            Duration::from_secs(10),
            None,
            false,
            Some(&job),
        )
        .await?;
    let (body, _) = runner.read_capped(resp).await?;
    Ok(serde_json::from_slice::<Value>(&body)
        .map(|v| JobOffer::from_answer(&v))
        .unwrap_or_default())
}

/// The endpoint's jobs: native when it offers v1.3, the baseline otherwise — decided once, on the
/// first operation, and cached by the runner (a probe that could not reach it is retried).
pub struct HttpJobs {
    runner: Arc<HttpExecRunner>,
    shell: ShellJobs,
}

impl HttpJobs {
    pub(crate) fn new(runner: Arc<HttpExecRunner>, shell: ShellJobs) -> Self {
        Self { runner, shell }
    }

    fn native(&self) -> NativeJobs<'_> {
        NativeJobs {
            runner: &self.runner,
        }
    }

    /// Whether this endpoint's jobs are native (v1.3).
    pub async fn native_capability(&self) -> bool {
        self.runner.offers_jobs().await
    }
}

macro_rules! dispatch {
    ($self:ident, $method:ident ( $($arg:expr),* )) => {
        if $self.runner.offers_jobs().await {
            $self.native().$method($($arg),*).await
        } else {
            $self.shell.$method($($arg),*).await
        }
    };
}

#[async_trait]
impl SandboxJobs for HttpJobs {
    async fn available(&self) -> bool {
        if self.runner.offers_jobs().await {
            return true;
        }
        self.shell.available().await
    }

    async fn holds(&self) -> bool {
        // Native holds need a stream like any other (a v1.3 endpoint is a v1.2 one).
        self.runner.streams().await
    }

    async fn run_held(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
        events: EventSink<'_>,
    ) -> std::io::Result<i32> {
        dispatch!(self, run_held(scope, id, spec, events))
    }

    async fn start_kept(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
    ) -> std::io::Result<u32> {
        dispatch!(self, start_kept(scope, id, spec))
    }

    async fn follow(
        &self,
        scope: &JobScope,
        id: &JobId,
        from: u64,
        events: EventSink<'_>,
    ) -> std::io::Result<i32> {
        dispatch!(self, follow(scope, id, from, events))
    }

    async fn read(
        &self,
        scope: &JobScope,
        id: &JobId,
        offset: u64,
        max: usize,
    ) -> std::io::Result<ReadChunk> {
        dispatch!(self, read(scope, id, offset, max))
    }

    async fn keep(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
        kind: &str,
        seen: u64,
    ) -> std::io::Result<()> {
        dispatch!(self, keep(scope, id, num, kind, seen))
    }

    async fn offers_notify(&self) -> bool {
        dispatch!(self, offers_notify())
    }

    async fn keep_notify(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
        kind: &str,
        seen: u64,
        notify: &JobNotify,
    ) -> std::io::Result<()> {
        dispatch!(self, keep_notify(scope, id, num, kind, seen, notify))
    }

    async fn seen(&self, scope: &JobScope, id: &JobId, offset: u64) -> std::io::Result<()> {
        dispatch!(self, seen(scope, id, offset))
    }

    async fn kill(
        &self,
        scope: &JobScope,
        ids: &[JobId],
        remove: bool,
    ) -> std::io::Result<Vec<(JobId, String)>> {
        dispatch!(self, kill(scope, ids, remove))
    }

    async fn list(&self, scope: &JobScope) -> std::io::Result<Vec<JobInfo>> {
        dispatch!(self, list(scope))
    }

    async fn remove(&self, scope: &JobScope, ids: &[JobId]) -> std::io::Result<()> {
        dispatch!(self, remove(scope, ids))
    }

    async fn wipe(&self, scope: &JobScope) -> std::io::Result<Vec<String>> {
        dispatch!(self, wipe(scope))
    }
}

/// The v1.3 client: every [`SandboxJobs`] operation is one request carrying `"job"`.
pub struct NativeJobs<'a> {
    runner: &'a HttpExecRunner,
}

impl<'a> NativeJobs<'a> {
    /// Over `runner`, which must have answered the probe. For the conformance verifier.
    pub fn over(runner: &'a HttpExecRunner) -> Self {
        Self { runner }
    }

    fn op(op: &str, scope: &JobScope, extra: Value) -> Value {
        let mut job = json!({ "op": op, "scope": scope.name() });
        if let (Value::Object(job), Value::Object(extra)) = (&mut job, extra) {
            job.extend(extra);
        }
        job
    }

    /// One non-streamed operation; its JSON answer.
    #[doc(hidden)]
    pub async fn call(&self, job: Value) -> std::io::Result<Value> {
        let resp = self
            .runner
            .send_job(INERT, &[], None, OP_TIMEOUT, None, false, Some(&job))
            .await?;
        let (body, over) = self.runner.read_capped(resp).await?;
        if over {
            return Err(std::io::Error::other(
                "a job answer exceeded the response cap",
            ));
        }
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::other(format!(
                "unparseable job answer ({e}): {}",
                String::from_utf8_lossy(&body[..body.len().min(200)])
            ))
        })
    }

    /// One streamed operation: frames to `events`, until the final one.
    async fn stream(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        job: Value,
        events: EventSink<'_>,
    ) -> std::io::Result<i32> {
        let resp = self
            .runner
            .send_job(
                program,
                args,
                cwd,
                super::STREAM_DEADLINE,
                None,
                true,
                Some(&job),
            )
            .await?;
        let mut reader = FrameReader::new(resp);
        let broken = |e: String| std::io::Error::other(format!("job stream: {e}"));
        loop {
            let Some(line) = reader
                .next_line(FRAME_SILENCE_LIMIT)
                .await
                .map_err(broken)?
            else {
                return Err(broken(
                    "the stream ended before the job's final frame".into(),
                ));
            };
            match decode_job_frame(&line).map_err(broken)? {
                JobFrame::Event(Event::Started(pgid)) => events(JobEvent::Started(pgid)),
                JobFrame::Event(Event::Output(b)) => events(JobEvent::Output(&b)),
                JobFrame::Event(Event::Gap(n)) => events(JobEvent::Gap(n)),
                JobFrame::Event(Event::ShellExit(c)) => events(JobEvent::ShellExit(c)),
                JobFrame::Heartbeat => {}
                JobFrame::End(code) => {
                    reader.finish().await;
                    return Ok(code);
                }
                JobFrame::Error(e) => return Err(broken(e)),
            }
        }
    }
}

#[async_trait]
impl SandboxJobs for NativeJobs<'_> {
    async fn available(&self) -> bool {
        true
    }

    async fn holds(&self) -> bool {
        self.runner.streams().await
    }

    async fn run_held(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
        events: EventSink<'_>,
    ) -> std::io::Result<i32> {
        let job = Self::op(
            "start",
            scope,
            json!({ "id": id.as_str(), "hold": true, "label": spec.label }),
        );
        self.stream(&spec.program, &spec.args, spec.cwd.as_deref(), job, events)
            .await
    }

    async fn start_kept(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
    ) -> std::io::Result<u32> {
        let job = Self::op(
            "start",
            scope,
            json!({ "id": id.as_str(), "hold": false, "label": spec.label }),
        );
        let resp = self
            .runner
            .send_job(
                &spec.program,
                &spec.args,
                spec.cwd.as_deref(),
                OP_TIMEOUT,
                None,
                false,
                Some(&job),
            )
            .await?;
        let (body, _) = self.runner.read_capped(resp).await?;
        let started = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| v.get("started")?.as_u64())
            .ok_or_else(|| {
                std::io::Error::other(format!(
                    "the endpoint did not report the job's start: {}",
                    String::from_utf8_lossy(&body[..body.len().min(200)])
                ))
            })?;
        u32::try_from(started).map_err(|_| {
            std::io::Error::other(format!(
                "the endpoint reported an out-of-range job start: {started}"
            ))
        })
    }

    async fn follow(
        &self,
        scope: &JobScope,
        id: &JobId,
        from: u64,
        events: EventSink<'_>,
    ) -> std::io::Result<i32> {
        let job = Self::op(
            "attach",
            scope,
            json!({ "id": id.as_str(), "offset": from }),
        );
        self.stream(INERT, &[], None, job, events).await
    }

    async fn read(
        &self,
        scope: &JobScope,
        id: &JobId,
        offset: u64,
        max: usize,
    ) -> std::io::Result<ReadChunk> {
        let v = self
            .call(Self::op(
                "read",
                scope,
                json!({ "id": id.as_str(), "offset": offset, "max": max }),
            ))
            .await?;
        parse_read(&v)
    }

    async fn keep(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
        kind: &str,
        seen: u64,
    ) -> std::io::Result<()> {
        self.call(Self::op(
            "keep",
            scope,
            json!({ "id": id.as_str(), "num": num, "kind": kind, "seen": seen }),
        ))
        .await
        .map(drop)
    }

    async fn offers_notify(&self) -> bool {
        self.runner.job_offer().await.notify
    }

    async fn keep_notify(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
        kind: &str,
        seen: u64,
        notify: &JobNotify,
    ) -> std::io::Result<()> {
        self.call(Self::op(
            "keep",
            scope,
            json!({
                "id": id.as_str(), "num": num, "kind": kind, "seen": seen,
                "notify": { "url": notify.url, "token": notify.token },
            }),
        ))
        .await
        .map(drop)
    }

    async fn seen(&self, scope: &JobScope, id: &JobId, offset: u64) -> std::io::Result<()> {
        self.call(Self::op(
            "seen",
            scope,
            json!({ "id": id.as_str(), "offset": offset }),
        ))
        .await
        .map(drop)
    }

    async fn kill(
        &self,
        scope: &JobScope,
        ids: &[JobId],
        remove: bool,
    ) -> std::io::Result<Vec<(JobId, String)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<&str> = ids.iter().map(JobId::as_str).collect();
        let v = self
            .call(Self::op(
                "kill",
                scope,
                json!({ "ids": ids, "remove": remove, "grace_ms": KILL_GRACE_MS }),
            ))
            .await?;
        Ok(parse_survivors(&v)
            .into_iter()
            .filter_map(|(id, d)| Some((JobId::parse(&id?)?, d)))
            .collect())
    }

    async fn list(&self, scope: &JobScope) -> std::io::Result<Vec<JobInfo>> {
        let v = self.call(Self::op("list", scope, json!({}))).await?;
        parse_list(&v)
    }

    async fn remove(&self, scope: &JobScope, ids: &[JobId]) -> std::io::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let ids: Vec<&str> = ids.iter().map(JobId::as_str).collect();
        self.call(Self::op("remove", scope, json!({ "ids": ids })))
            .await
            .map(drop)
    }

    async fn wipe(&self, scope: &JobScope) -> std::io::Result<Vec<String>> {
        let v = self
            .call(Self::op(
                "wipe",
                scope,
                json!({ "grace_ms": KILL_GRACE_MS }),
            ))
            .await?;
        Ok(parse_survivors(&v).into_iter().map(|(_, d)| d).collect())
    }
}

/// What a job stream's line says.
#[derive(Debug, PartialEq, Eq)]
pub enum JobFrame {
    Event(Event),
    Heartbeat,
    /// Final: the job ended with this status (128 + N for signal N).
    End(i32),
    /// Final: the endpoint failed.
    Error(String),
}

/// A job stream's non-final news, owned.
#[derive(Debug, PartialEq, Eq)]
pub enum Event {
    Started(u32),
    Output(Vec<u8>),
    Gap(u64),
    ShellExit(i32),
}

#[derive(Deserialize)]
struct WireJobFrame {
    started: Option<u64>,
    stdout: Option<String>,
    stderr: Option<String>,
    gap: Option<u64>,
    shell_exit: Option<i64>,
    exit_code: Option<i64>,
    signal: Option<i64>,
    #[serde(default)]
    timed_out: bool,
    error: Option<String>,
}

/// Decode one line of a v1.3 job stream. The frames are v1.2's (`stdout`, `stderr`, `{}`,
/// `shell_exit`, `exit_code`, `signal`, `error`) plus `started` and `gap`; both output streams are
/// the job's combined output. Unknown fields are ignored, and a line with none known is a heartbeat.
pub fn decode_job_frame(line: &[u8]) -> Result<JobFrame, String> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(JobFrame::Heartbeat);
    }
    let w: WireJobFrame = serde_json::from_slice(line).map_err(|e| {
        format!(
            "unparseable job frame ({e}): {}",
            String::from_utf8_lossy(&line[..line.len().min(200)])
        )
    })?;
    let b64 = |s: &str| {
        base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| format!("output is not standard base64 ({e})"))
    };
    // Numbers are read wide and narrowed here, so an out-of-range value is an error: never a panic
    // (the signal sum) and never a truncated status.
    Ok(if let Some(e) = w.error {
        JobFrame::Error(e)
    } else if let Some(p) = w.started {
        let p = u32::try_from(p).map_err(|_| wire_out_of_range("started", p))?;
        JobFrame::Event(Event::Started(p))
    } else if let Some(s) = w.stdout.as_deref().or(w.stderr.as_deref()) {
        JobFrame::Event(Event::Output(b64(s)?))
    } else if let Some(n) = w.gap {
        JobFrame::Event(Event::Gap(n))
    } else if let Some(c) = w.shell_exit {
        JobFrame::Event(Event::ShellExit(wire_i32("shell_exit", c)?))
    } else if let Some(c) = w.exit_code {
        JobFrame::End(wire_i32("exit_code", c)?)
    } else if let Some(s) = w.signal {
        let s = wire_i32("signal", s)?;
        JobFrame::End(
            128i32
                .checked_add(s)
                .ok_or_else(|| wire_out_of_range("signal", s))?,
        )
    } else if w.timed_out {
        JobFrame::Error("the endpoint timed the job out".into())
    } else {
        JobFrame::Heartbeat
    })
}

/// A wire integer that must fit an `i32`; anything wider is a protocol error.
fn wire_i32(field: &str, v: i64) -> Result<i32, String> {
    i32::try_from(v).map_err(|_| wire_out_of_range(field, v))
}

fn wire_out_of_range(field: &str, v: impl std::fmt::Display) -> String {
    format!("job frame field `{field}` is out of range: {v}")
}

fn status(state: &str, code: Option<i32>) -> Option<JobStatus> {
    Some(match state {
        "running" => JobStatus::Running,
        "exited" => JobStatus::ShellExited(code?),
        "ended" => JobStatus::Ended(code?),
        "lost" => JobStatus::Lost,
        "gone" => JobStatus::Gone,
        _ => return None,
    })
}

/// The `code` of a job answer: absent or null is `None`; an integer that does not fit an `i32`
/// (or is not an integer) is a protocol error.
fn code_field(v: &Value) -> std::io::Result<Option<i32>> {
    match v.get("code") {
        None | Some(Value::Null) => Ok(None),
        Some(c) => c
            .as_i64()
            .and_then(|c| i32::try_from(c).ok())
            .map(Some)
            .ok_or_else(|| {
                std::io::Error::other(format!("unreadable job answer: `code` is not an i32: {c}"))
            }),
    }
}

fn bad(what: &str, v: &Value) -> std::io::Error {
    std::io::Error::other(format!("unreadable job {what}: {v}"))
}

/// A `read` answer: `{"state", "code", "offset", "end", "data"}`.
pub fn parse_read(v: &Value) -> std::io::Result<ReadChunk> {
    let code = code_field(v)?;
    let st = v
        .get("state")
        .and_then(Value::as_str)
        .and_then(|s| status(s, code))
        .ok_or_else(|| bad("read", v))?;
    let num = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    let data = base64::engine::general_purpose::STANDARD
        .decode(v.get("data").and_then(Value::as_str).unwrap_or(""))
        .map_err(|_| bad("read", v))?;
    Ok(ReadChunk {
        status: st,
        offset: num("offset"),
        end: num("end"),
        data,
    })
}

/// A `list` answer: `{"jobs": [{"id", "state", "code", "kept", "num", "kind", "seen", "started",
/// "ended", "label"}]}`. A job whose id is not one is skipped, never followed.
pub fn parse_list(v: &Value) -> std::io::Result<Vec<JobInfo>> {
    let jobs = v
        .get("jobs")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("list", v))?;
    let mut out = Vec::with_capacity(jobs.len());
    for j in jobs {
        // A code that does not fit is a protocol error for the whole answer, even on a job skipped below.
        let code = code_field(j)?;
        let (Some(id), Some(state)) = (
            j.get("id").and_then(Value::as_str).and_then(JobId::parse),
            j.get("state").and_then(Value::as_str),
        ) else {
            continue;
        };
        let Some(status) = status(state, code) else {
            continue;
        };
        let num = |k: &str| j.get(k).and_then(Value::as_u64);
        out.push(JobInfo {
            id,
            status,
            kept: j.get("kept").and_then(Value::as_bool).unwrap_or(false),
            num: num("num"),
            kind: j.get("kind").and_then(Value::as_str).map(str::to_string),
            seen: num("seen").unwrap_or(0),
            started: num("started").unwrap_or(0),
            ended: num("ended").unwrap_or(0),
            label: j
                .get("label")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    Ok(out)
}

/// A `kill`/`wipe` answer: `{"survivors": [{"id"?, "pids": [..]}]}`, described.
pub fn parse_survivors(v: &Value) -> Vec<(Option<String>, String)> {
    v.get("survivors")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|s| {
                    let pids: Vec<String> = s
                        .get("pids")
                        .and_then(Value::as_array)
                        .map(|p| {
                            p.iter()
                                .filter_map(Value::as_u64)
                                .map(|p| p.to_string())
                                .collect()
                        })
                        .unwrap_or_default();
                    (
                        s.get("id").and_then(Value::as_str).map(str::to_string),
                        super::jobs::describe_survivors(&pids.join(",")),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every probe answer in `v1.4-probe.json` means to this client what the spec says it means.
    #[test]
    fn probe_answers_offer_what_the_vectors_say() {
        let file: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/exec/v1.4-probe.json"))
                .unwrap();
        let answers = file["answers"].as_array().unwrap();
        assert!(!answers.is_empty());
        for case in answers {
            let want = &case["expect"];
            let got = JobOffer::from_answer(&case["answer"]);
            assert_eq!(
                got,
                JobOffer {
                    native_jobs: want["native_jobs"].as_bool().unwrap(),
                    notify: want["notify"].as_bool().unwrap(),
                    baseline_notify: want["baseline_notify"].as_bool().unwrap(),
                },
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn a_signal_past_the_i32_sum_is_an_error_not_a_panic() {
        let err = decode_job_frame(br#"{"signal":2147483647}"#).unwrap_err();
        assert!(
            err.contains("`signal`") && err.contains("2147483647"),
            "{err}"
        );
        assert!(decode_job_frame(br#"{"signal":4294967297}"#).is_err());
        assert_eq!(decode_job_frame(br#"{"signal":9}"#), Ok(JobFrame::End(137)));
    }

    #[test]
    fn a_wide_exit_code_is_an_error_not_a_truncation() {
        let err = decode_job_frame(br#"{"exit_code":4294967297}"#).unwrap_err();
        assert!(
            err.contains("`exit_code`") && err.contains("4294967297"),
            "{err}"
        );
        assert!(decode_job_frame(br#"{"shell_exit":4294967297}"#).is_err());
        assert_eq!(
            decode_job_frame(br#"{"exit_code":3}"#),
            Ok(JobFrame::End(3))
        );
        assert!(decode_job_frame(br#"{"started":4294967297}"#).is_err());
    }

    #[test]
    fn a_wide_read_code_is_an_error_not_ended_one() {
        let wide =
            json!({"state": "ended", "code": 4294967297_u64, "offset": 0, "end": 0, "data": ""});
        assert!(parse_read(&wide).is_err());
        let running =
            json!({"state": "running", "code": 4294967297_u64, "offset": 0, "end": 0, "data": ""});
        assert!(parse_read(&running).is_err());
        let ok = json!({"state": "ended", "code": 3, "offset": 0, "end": 0, "data": ""});
        assert_eq!(parse_read(&ok).unwrap().status, JobStatus::Ended(3));
    }

    #[test]
    fn a_wide_code_in_a_list_is_an_error() {
        let wide = json!({"jobs": [{"id": "j1", "state": "ended", "code": 4294967297_u64}]});
        assert!(parse_list(&wide).is_err());
    }
}
