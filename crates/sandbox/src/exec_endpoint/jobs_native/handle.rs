//! The Rust SDK's job handle: the shell around the client kernel (`job_client::step`).
//!
//! It decides nothing. It sends the request the kernel names, reads the answer or the next line,
//! turns it into a kernel [`Ev`], and does what the [`Action`]s say: open a stream, hand output to the
//! sink, sleep, finish or fail. Whether to reopen, from which offset, how long to wait, what a status
//! means and what a gap is are all the kernel's (proven in Verus; see `crates/job-client`). Dropping the
//! future is the caller's cancel: it drops the connection (a held job not kept by then is killed by the
//! sandbox) and nothing else.

use std::time::{Duration, Instant};

use job_client::step::{
    self, Action, Authority, Event as Ev, Fault, Frame as KFrame, Mark, State, silence_limit_ms,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::jobs::{
    EventSink, JobEnd, JobError, JobEvent, JobId, JobScope, JobSpec, SandboxJobs, Stream,
};
use super::super::stream::FrameReader;
use super::super::{EndpointStatus, HttpExecRunner, STREAM_DEADLINE, endpoint_status};
use super::{Event, INERT, JobFrame, NativeJobs, decode_job_frame, predates};

/// What a handle drives: the endpoint, the job, and (for a start) what to run.
pub(super) struct Handle<'a> {
    pub runner: &'a HttpExecRunner,
    pub scope: &'a JobScope,
    pub id: &'a JobId,
    /// The command a `Start` action sends; `None` for a handle that only attaches.
    pub spec: Option<&'a JobSpec>,
    pub beat: Option<u32>,
    /// Whether an attach carries the output (`false`: status only).
    pub output: bool,
}

/// The shell's clock: milliseconds since the handle began. A test makes it run `divisor` times
/// faster (`HttpExecRunner::with_time_divisor`); this is the one place that is read.
struct Clock {
    began: Instant,
    divisor: u64,
}

impl Clock {
    fn now(&self) -> u64 {
        let ms = u64::try_from(self.began.elapsed().as_millis()).unwrap_or(u64::MAX);
        ms.saturating_mul(self.divisor)
    }

    /// The real time to wait for the clock to reach `until`.
    fn wait(&self, until: u64) -> Duration {
        Duration::from_millis(until.saturating_sub(self.now()) / self.divisor.max(1) + 1)
    }
}

/// A refusal the endpoint gave, kept to say what ended the handle.
struct Refusal {
    kind: std::io::ErrorKind,
    status: u16,
    body: String,
    text: String,
}

impl Refusal {
    fn error(&self) -> std::io::Error {
        std::io::Error::new(
            self.kind,
            EndpointStatus::new(self.status, self.body.clone(), self.text.clone()),
        )
    }
}

/// What the stream's last decoded line carried that the kernel's [`KFrame`] does not.
#[derive(Default)]
struct Payload {
    bytes: Vec<u8>,
    stream: Option<Stream>,
    pgid: u32,
    /// The start's key as the `started` frame echoed it, and the key the start sent.
    key: Option<String>,
    sent: String,
    /// The text of an error frame.
    error: Option<String>,
}

fn draw() -> u32 {
    let mut b = [0u8; 4];
    let _ = getrandom::fill(&mut b);
    u32::from_le_bytes(b)
}

fn mark_text(m: &Mark) -> &str {
    std::str::from_utf8(m.as_bytes()).unwrap_or("")
}

/// Drive `state` to the job's end: every decision is `step`'s.
pub(super) async fn drive(
    h: &Handle<'_>,
    mut s: State,
    events: EventSink<'_>,
) -> std::io::Result<JobEnd> {
    let clock = Clock {
        began: Instant::now(),
        divisor: h.runner.time_divisor,
    };
    let key = match h.spec {
        Some(spec) => super::start_key(spec)?,
        None => String::new(),
    };
    let silence = |ms: u64| Duration::from_millis(ms);
    let mut reader: Option<FrameReader> = None;
    let mut refusal: Option<Refusal> = None;
    let mut payload = Payload {
        sent: key.clone(),
        ..Payload::default()
    };
    let mut end: Option<JobEnd> = None;
    let mut ev = Ev::Go;
    let mut limit = silence(silence_limit_ms(h.beat));
    loop {
        let (next_state, actions) = step::step(s, ev, clock.now(), draw());
        s = next_state;
        // The event the last I/O action produced, if it did.
        let mut next: Option<Ev> = None;
        for action in actions {
            match action {
                Action::Start { silence_ms, .. } => {
                    limit = silence(silence_ms);
                    next = Some(open(h, Opening::Start(&key), &mut reader, &mut refusal).await?);
                }
                Action::OpenAttach {
                    offset,
                    mark,
                    silence_ms,
                } => {
                    limit = silence(silence_ms);
                    let o = Opening::Attach { offset, mark };
                    next = Some(open(h, o, &mut reader, &mut refusal).await?);
                }
                // The kernel took the first frame: the sink hears of the incarnation.
                Action::Started { mark, .. } => {
                    events(JobEvent::Started(payload.pgid, mark_text(&mark)));
                }
                Action::Attached { mark, from } => {
                    events(JobEvent::Attached {
                        mark: mark_text(&mark),
                        from,
                    });
                }
                // Not an end: the shell left processes running, and the sink hears of it.
                Action::ShellExit { code } => events(JobEvent::ShellExit(code)),
                // A kept start whose echo was refused: stop it, so nothing can name it and nothing
                // is left running that expects its stdin or terminal. Best effort, as the refusal is.
                Action::Kill => {
                    let _ = NativeJobs::over(h.runner)
                        .kill(h.scope, std::slice::from_ref(h.id), true)
                        .await;
                }
                Action::CloseStream => {
                    // After the final frame the body is let end, so the endpoint sees a stream that
                    // finished (and the connection goes back to the pool); any other close is a drop.
                    if let Some(r) = reader.take()
                        && matches!(ev, Ev::Frame(KFrame::Exit { .. }))
                    {
                        r.finish().await;
                    }
                }
                Action::Deliver { len, dropped, .. } => {
                    if dropped > 0 {
                        events(JobEvent::Gap(dropped));
                    }
                    debug_assert_eq!(len as usize, payload.bytes.len());
                    let stream = payload.stream.unwrap_or(Stream::Stdout);
                    events(JobEvent::Output(&payload.bytes, stream));
                }
                Action::DeliverExit {
                    code,
                    signal,
                    dropped,
                    ..
                } => {
                    if dropped > 0 {
                        events(JobEvent::Gap(dropped));
                    }
                    end = Some(if signal != 0 {
                        JobEnd::Signal(signal)
                    } else {
                        JobEnd::Exit(code)
                    });
                }
                // Not an end: the front has been out of reach for a while. Said once a second try is
                // spent, then on every further miss; the handle keeps going until the job ends.
                Action::Reconnecting { attempts, since_ms } => {
                    tracing::warn!(
                        job = %h.id, attempts, since_ms,
                        "the exec endpoint is out of reach; reconnecting to the job"
                    );
                }
                Action::Sleep { until_ms } => {
                    tokio::time::sleep(clock.wait(until_ms)).await;
                    next = Some(Ev::Tick);
                }
                // A handle that never asked for a remove is never told to.
                Action::Remove { .. } => {
                    return Err(std::io::Error::other("job stream: unexpected remove"));
                }
                Action::Fail { fault, status, .. } => {
                    return Err(fault_error(fault, status, refusal.take(), &payload));
                }
                Action::Finish => {
                    return end.ok_or_else(|| std::io::Error::other("job stream: no end"));
                }
                Action::Detached => {
                    return Err(std::io::Error::other("job stream: closed"));
                }
            }
        }
        ev = match next {
            Some(e) => e,
            None => match s.phase {
                step::Phase::Confirming | step::Phase::Streaming => {
                    read(&mut reader, limit, &mut payload).await
                }
                // The exit was handed over and nobody holds it back.
                step::Phase::ExitHeld => Ev::ExitTaken,
                _ => return Err(std::io::Error::other("job stream: the handle stalled")),
            },
        };
    }
}

enum Opening<'k> {
    Start(&'k str),
    Attach { offset: u64, mark: Mark },
}

/// Send the request the kernel named. A 2xx opens the reader; a refusal is an `Answered` status; a
/// connection that cannot be had is `Broke`. Only a request that can never be sent is an error.
async fn open(
    h: &Handle<'_>,
    what: Opening<'_>,
    reader: &mut Option<FrameReader>,
    refusal: &mut Option<Refusal>,
) -> std::io::Result<Ev> {
    let sent = match what {
        Opening::Start(key) => {
            let Some(spec) = h.spec else {
                return Err(std::io::Error::other("job handle: no command to start"));
            };
            let job = NativeJobs::op(
                "start",
                h.scope,
                json!({
                    "id": h.id.as_str(), "hold": true, "label": spec.label,
                    "stdin_open": spec.stdin_open, "idempotency_key": key,
                }),
            );
            let job = match spec.pty {
                Some(p) => {
                    let mut job = job;
                    job["pty"] = json!({ "cols": p.cols, "rows": p.rows });
                    job
                }
                None => job,
            };
            h.runner
                .send_env_beat(
                    &spec.program,
                    &spec.args,
                    spec.cwd.as_deref(),
                    STREAM_DEADLINE,
                    None,
                    true,
                    Some(&job),
                    &spec.env,
                    h.beat,
                )
                .await
        }
        Opening::Attach { offset, mark } => {
            let mut extra = json!({ "id": h.id.as_str(), "offset": offset, "output": h.output });
            if mark.len > 0 {
                extra["mark"] = Value::String(mark_text(&mark).to_string());
            }
            let job = NativeJobs::op("attach", h.scope, extra);
            h.runner
                .send_env_beat(
                    INERT,
                    &[],
                    None,
                    STREAM_DEADLINE,
                    None,
                    true,
                    Some(&job),
                    &[],
                    h.beat,
                )
                .await
        }
    };
    match sent {
        Ok(resp) => {
            *reader = Some(FrameReader::new(resp));
            Ok(Ev::Answered {
                status: 200,
                retry_after_s: 0,
                authority: Authority::NotAuthoritative,
            })
        }
        Err(e) => {
            if let Some(st) = endpoint_status(&e) {
                *refusal = Some(Refusal {
                    kind: e.kind(),
                    status: st.status,
                    body: st.body.clone(),
                    text: st.to_string(),
                });
                return Ok(Ev::Answered {
                    status: st.status,
                    retry_after_s: st.retry_after_s,
                    authority: authority_of(st.status, &st.body, h.runner.sandbox_id()),
                });
            }
            // A request that cannot be made (a heartbeat out of range, an endpoint that does not
            // apply the command's env) will not be made by waiting.
            if matches!(
                e.kind(),
                std::io::ErrorKind::Unsupported | std::io::ErrorKind::InvalidInput
            ) {
                return Err(e);
            }
            // Nothing reached a server: refused, or no route. The kernel ends a start nothing answered
            // (`T`) and keeps every other handle reconnecting.
            if matches!(
                e.kind(),
                std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::AddrNotAvailable
                    | std::io::ErrorKind::HostUnreachable
                    | std::io::ErrorKind::NetworkUnreachable
            ) {
                return Ok(Ev::Unreached);
            }
            Ok(Ev::Broke)
        }
    }
}

/// What the kernel makes of a refusal (`job_client::step::authority`, proven in Verus: the one place that
/// decides whether an answer is the backend's own word that the sandbox is gone or the session has ended).
/// This only parses the body into the fields it takes: a body that is not JSON, or a field of another type,
/// is a field that is not there. The sandbox id is the endpoint's configured one (`HttpExecRunner::sandbox_id`),
/// never read out of the URL here.
///
/// The answers it recognises, as captured: E2B for a sandbox it no longer has is `502`,
/// `content-type: application/json`, `{"sandboxId":"<id>","message":"The sandbox was not found","code":502}`
/// (Beyond's execfront tests the same answer, `api/app/execfront/e2b.go` `sandboxNotFound`); Beyond for a
/// session that has ended is `401`, `{"code":"session_ended","message":"session ended"}`
/// (`api/app/execfront/front.go` `statusFor`, `serve.go` `retryLater`).
fn authority_of(status: u16, body: &str, ours: Option<&str>) -> Authority {
    #[derive(Deserialize, Default)]
    struct Said {
        code: Option<serde_json::Value>,
        message: Option<String>,
        #[serde(rename = "sandboxId")]
        sandbox_id: Option<String>,
    }
    let said: Said = serde_json::from_str(body).unwrap_or_default();
    let (code_num, code_text) = match &said.code {
        Some(serde_json::Value::Number(n)) => (n.as_u64(), ""),
        Some(serde_json::Value::String(t)) => (None, t.as_str()),
        _ => (None, ""),
    };
    step::authority(
        status,
        said.message.as_deref().unwrap_or("").as_bytes(),
        code_num,
        code_text.as_bytes(),
        said.sandbox_id.as_deref().unwrap_or("").as_bytes(),
        ours.unwrap_or("").as_bytes(),
    )
}

/// The next kernel event from the open stream: a frame, or `Broke` when it ended or went quiet.
async fn read(reader: &mut Option<FrameReader>, limit: Duration, p: &mut Payload) -> Ev {
    let Some(r) = reader.as_mut() else {
        return Ev::Broke;
    };
    let line = match r.next_line(limit).await {
        Ok(Some(l)) => l,
        Ok(None) | Err(_) => return Ev::Broke,
    };
    Ev::Frame(match decode_job_frame(&line) {
        Err(e) => {
            p.error = Some(e);
            KFrame::Bad
        }
        Ok(JobFrame::Heartbeat) => KFrame::Heartbeat,
        Ok(JobFrame::Error(e)) => {
            p.error = Some(e);
            KFrame::Bad
        }
        Ok(JobFrame::End(JobEnd::Exit(code))) => KFrame::Exit { code, signal: 0 },
        Ok(JobFrame::End(JobEnd::Signal(signal))) => KFrame::Exit { code: 0, signal },
        Ok(JobFrame::Event(Event::Started(pgid, mark, stdin, echoed, key))) => {
            p.pgid = pgid;
            p.key = key;
            KFrame::Started {
                mark: as_mark(mark.as_deref(), p),
                stdin,
                pty: echoed.is_some(),
                key: p.key.as_deref() == Some(p.sent.as_str()),
            }
        }
        Ok(JobFrame::Event(Event::Attached(mark, from))) => KFrame::Attached {
            mark: as_mark(Some(&mark), p),
            from,
        },
        Ok(JobFrame::Event(Event::Output(bytes, stream))) => {
            let len = bytes.len() as u64;
            p.bytes = bytes;
            p.stream = Some(stream);
            KFrame::Output { len }
        }
        Ok(JobFrame::Event(Event::Gap(n))) => KFrame::Gap { n },
        Ok(JobFrame::Event(Event::ShellExit(code))) => KFrame::ShellExit { code },
    })
}

/// A mark as the wire spelled it: no mark (an endpoint that predates marks) is the empty one, which
/// the kernel answers `NotOffered`; a malformed one is a bad frame.
fn as_mark(text: Option<&str>, p: &mut Payload) -> Mark {
    match text {
        None => Mark::none(),
        Some(t) => Mark::from_bytes(t.as_bytes()).unwrap_or_else(|| {
            p.error = Some(format!("the job's mark is not a mark: {t:?}"));
            Mark::none()
        }),
    }
}

/// A typed failure as the error a jobs call returns.
fn fault_error(fault: Fault, status: u16, refusal: Option<Refusal>, p: &Payload) -> std::io::Error {
    let body = refusal.as_ref().map(|r| r.body.clone()).unwrap_or_default();
    match fault {
        Fault::JobGone => JobError::gone(if body.is_empty() {
            "the job does not exist, or its id now names another job".to_string()
        } else {
            body
        })
        .into_io(),
        Fault::JobExists => JobError::Exists { text: body }.into_io(),
        Fault::DiskFull => JobError::DiskFull { text: body }.into_io(),
        Fault::NotOffered => predates("job marks"),
        // The backend said the session has ended: the job's sandbox went with it.
        Fault::SessionEnded => std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "job stream: the session has ended{}",
                if body.is_empty() {
                    String::new()
                } else {
                    format!(" ({body})")
                }
            ),
        ),
        // The backend said the sandbox is gone: the same error the unreachable sandbox has always ended in
        // ("failed after ..."), now only on that answer.
        Fault::SandboxGone => std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "job stream: the sandbox no longer exists{}",
                if body.is_empty() {
                    String::new()
                } else {
                    format!(" ({body})")
                }
            ),
        ),
        // A start nothing answered: the endpoint could not be reached at all.
        Fault::Transport => std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "job stream: the endpoint could not be reached",
        ),
        // No caller sends it: a Rust caller's deadline is a dropped future.
        Fault::Deadline => {
            std::io::Error::new(std::io::ErrorKind::TimedOut, "job stream: deadline")
        }
        Fault::Auth | Fault::NoPty | Fault::Protocol => match refusal {
            Some(r) if r.status == status => r.error(),
            _ => std::io::Error::other(format!(
                "job stream: {}",
                p.error
                    .clone()
                    .unwrap_or_else(|| format!("unexpected answer (status {status})"))
            )),
        },
    }
}
