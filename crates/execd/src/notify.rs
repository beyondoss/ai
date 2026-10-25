//! v1.4: job-end notify. When to call back is the kernel's to say ([`exec_notify::step`], the one
//! pure function every endpoint shares); this side makes the requests and keeps the timers.
//!
//! **Bounds.** At most one attempt per job in flight (N16) and one retry timer per job: two tasks
//! per job with a `notify`, each owned by the job and aborted when the job is replaced, re-kept or
//! removed. Each attempt is abandoned after 30 s ([`ATTEMPT_TIMEOUT`], N16) and a count has at most
//! 12 attempts (N18), so a notification is over within 723 s however the receiver behaves. The
//! callback client keeps no idle connection, so the token never sits on a pooled socket. N
//! concurrent cold callers (N jobs kept with `notify` at once) means N attempts at once, one
//! request each, bounded by the jobs the agent keeps.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::Ctx;
use crate::jobs::{Job, JobCell, Key};

/// What an endpoint that offers v1.4 adds to its probe answer.
pub const NOTIFY_VERSION: &str = "v1.4";

/// How long one attempt may take before it is abandoned as a connection error (N16).
pub const ATTEMPT_TIMEOUT: Duration = Duration::from_millis(exec_notify::ATTEMPT_TIMEOUT_MS);

/// `keep`'s `notify`, as recorded. `Debug` never shows the token (N20).
#[derive(Clone, PartialEq, Eq)]
pub struct NotifyTarget {
    pub url: String,
    pub token: String,
}

impl std::fmt::Debug for NotifyTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NotifyTarget")
            .field("url", &self.url)
            .field("token", &"<redacted>")
            .finish()
    }
}

/// What an attempt carried: the proof at `keep`, or a state change owed since.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackKind {
    /// At `keep`: the proof of reach.
    Proof,
    /// A state change: the shell exited leaving its group alive, or the job's end is observable.
    Change,
}

/// One callback attempt, as the endpoint made it.
#[derive(Clone, Debug)]
pub struct CallbackAttempt {
    pub scope: String,
    pub id: String,
    pub url: String,
    pub kind: CallbackKind,
    /// 1 for a count's first attempt.
    pub n: u32,
    pub started: Instant,
    pub ended: Instant,
    /// The answer's status; `None` for a connection error, a TLS failure or no answer in 30 s.
    pub status: Option<u16>,
    /// Why there was no status, when there was none: the client's error and its causes.
    pub error: Option<String>,
}

/// Check `keep`'s `notify`: its members' types here, the grammar by the shared kernel
/// ([`exec_notify::check_notify`]). The rule it breaks, if any, never quoting the token.
pub fn parse_notify(v: &Value) -> Result<NotifyTarget, &'static str> {
    let obj = v.as_object().ok_or("notify type")?;
    let url = match obj.get("url") {
        None => return Err("url missing"),
        Some(u) => u.as_str().ok_or("url type")?,
    };
    let token = match obj.get("token") {
        None => return Err("token missing"),
        Some(t) => t.as_str().ok_or("token type")?,
    };
    exec_notify::check_notify(url, token)?;
    Ok(NotifyTarget {
        url: url.to_string(),
        token: token.to_string(),
    })
}

/// A task the job owns: dropped (by the kernel's `Abandon`, a new timer, or the job's removal), it
/// is aborted.
pub(crate) struct Owned(tokio::task::AbortHandle);

impl Drop for Owned {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The job's notification state, driven by the kernel: this side only carries out its actions, an
/// attempt task or a timer, and feeds back what happened.
#[derive(Default)]
pub(crate) struct Notifier {
    kernel: exec_notify::Job,
    /// What the latest owed callback is, for the attempt log.
    kind: Option<CallbackKind>,
    attempt: Option<Owned>,
    timer: Option<Owned>,
    /// Bumped per attempt and timer begun, so one abandoned while it waited for the lock can tell.
    seq: u64,
}

/// Feed `event` to job `j`'s kernel (the caller holds the job's lock) and carry out what it says.
pub(crate) fn feed(
    ctx: &Arc<Ctx>,
    cell: &JobCell,
    key: &Key,
    j: &mut Job,
    event: exec_notify::Event,
) {
    match event {
        exec_notify::Event::Keep { notify: true } => j.notifier.kind = Some(CallbackKind::Proof),
        exec_notify::Event::Change if j.notifier.kernel.notify => {
            j.notifier.kind = Some(CallbackKind::Change);
        }
        _ => {}
    }
    let now = u64::try_from(ctx.epoch.elapsed().as_millis()).unwrap_or(u64::MAX);
    let (next, actions) = exec_notify::step(j.notifier.kernel, event, now);
    j.notifier.kernel = next;
    for action in actions {
        match action {
            exec_notify::Action::Abandon => j.notifier.attempt = None,
            exec_notify::Action::Begin { attempt: n } => {
                j.notifier.seq += 1;
                let kind = j.notifier.kind.unwrap_or(CallbackKind::Proof);
                let task = tokio::spawn(run_attempt(
                    ctx.clone(),
                    cell.clone(),
                    key.clone(),
                    j.notify.clone(),
                    kind,
                    n,
                    j.notifier.seq,
                ));
                j.notifier.attempt = Some(Owned(task.abort_handle()));
            }
            exec_notify::Action::WakeAt(at) => {
                j.notifier.seq += 1;
                let (ctx2, cell2, key2, seq) =
                    (ctx.clone(), cell.clone(), key.clone(), j.notifier.seq);
                let task = tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(at.saturating_sub(now))).await;
                    let Ok(mut j) = cell2.lock() else { return };
                    if !j.removed && j.notifier.seq == seq {
                        feed(&ctx2, &cell2, &key2, &mut j, exec_notify::Event::Wake);
                    }
                });
                j.notifier.timer = Some(Owned(task.abort_handle()));
            }
        }
    }
}

/// A state change of `j`: the shell exited leaving its group alive, or its end became observable.
pub(crate) fn change(ctx: &Arc<Ctx>, cell: &JobCell, key: &Key, j: &mut Job) {
    feed(ctx, cell, key, j, exec_notify::Event::Change);
}

/// One attempt, then its outcome fed back.
async fn run_attempt(
    ctx: Arc<Ctx>,
    cell: JobCell,
    key: Key,
    target: Option<NotifyTarget>,
    kind: CallbackKind,
    n: u32,
    seq: u64,
) {
    let outcome = match target {
        Some(target) => {
            let started = Instant::now();
            let (status, error) = match attempt(&ctx, &target, &key).await {
                Ok(s) => (Some(s), None),
                Err(e) => (None, Some(e)),
            };
            if let Some(o) = &ctx.opts.observer {
                o.callback(&CallbackAttempt {
                    scope: key.0.clone(),
                    id: key.1.clone(),
                    url: target.url.clone(),
                    kind,
                    n,
                    started,
                    ended: Instant::now(),
                    status,
                    error,
                });
            }
            tracing::info!(kind = "callback", job = %key.1, attempt = n, status, "request");
            status.map_or(exec_notify::Outcome::Failed, exec_notify::Outcome::Status)
        }
        None => exec_notify::Outcome::Failed,
    };
    let mut draw = [0u8; 4];
    let _ = getrandom::fill(&mut draw);
    let Ok(mut j) = cell.lock() else { return };
    if !j.removed && j.notifier.seq == seq {
        j.notifier.attempt = None;
        feed(
            &ctx,
            &cell,
            &key,
            &mut j,
            exec_notify::Event::AttemptEnded {
                outcome,
                draw: u32::from_le_bytes(draw),
            },
        );
    }
}

/// One attempt: one `POST`, no redirects, the server verified, 30 s at most. Its status, or why not.
async fn attempt(ctx: &Ctx, target: &NotifyTarget, key: &Key) -> Result<u16, String> {
    let client = ctx
        .client
        .get_or_init(|| async {
            let _ = rustls::crypto::ring::default_provider().install_default();
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(ATTEMPT_TIMEOUT)
                .http1_only()
                // No idle connection outlives its attempt (the token never sits on a pooled one).
                .pool_max_idle_per_host(0)
                .build()
                .map_err(|e| e.to_string())
        })
        .await;
    let client = client
        .as_ref()
        .map_err(|e| format!("callback client: {e}"))?;
    client
        .post(&target.url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", target.token))
        .body(exec_notify::body(&key.0, &key.1))
        .send()
        .await
        .map(|r| r.status().as_u16())
        .map_err(|e| {
            let mut msg = e.to_string();
            let mut cause = std::error::Error::source(&e);
            while let Some(c) = cause {
                msg.push_str(": ");
                msg.push_str(&c.to_string());
                cause = c.source();
            }
            msg
        })
}
