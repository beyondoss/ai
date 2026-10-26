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
use crate::proc::locked;

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

impl Notifier {
    /// A callback is owed and not settled: in flight, waiting on its backoff, or about to begin.
    /// Settled, abandoned at the attempt limit, or removed, it is not.
    pub fn owes(&self) -> bool {
        let k = &self.kernel;
        !k.removed && k.notify && k.owed > k.settled
    }
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
    // The host's activity view includes the callbacks owed.
    ctx.activity_changed();
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
                    let mut j = locked(&cell2);
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
            crate::log_info!(
                "request kind=callback job={:?} attempt={n} status={status:?} error={:?} ms={}",
                key.1,
                error.as_deref(),
                started.elapsed().as_millis()
            );
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
            status.map_or(exec_notify::Outcome::Failed, exec_notify::Outcome::Status)
        }
        None => exec_notify::Outcome::Failed,
    };
    let mut draw = [0u8; 4];
    let _ = getrandom::fill(&mut draw);
    let mut j = locked(&cell);
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

/// PEM certificates in `text`: (well formed, malformed). A certificate is well formed when its
/// body is base64 that decodes to a DER sequence.
fn count_pem(text: &str) -> (usize, usize) {
    use base64::Engine as _;
    let (mut good, mut bad) = (0, 0);
    let mut rest = text;
    while let Some(start) = rest.find("-----BEGIN CERTIFICATE-----") {
        let body = &rest[start + 27..];
        let Some(end) = body.find("-----END CERTIFICATE-----") else {
            bad += 1;
            break;
        };
        let b64: String = body[..end].split_whitespace().collect();
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(der) if der.first() == Some(&0x30) => good += 1,
            _ => bad += 1,
        }
        rest = &body[end + 25..];
    }
    (good, bad)
}

/// What is wrong with the trust store that https callbacks verify against, if anything. Where
/// `SSL_CERT_FILE` or `SSL_CERT_DIR` is set it is the only source, as it is for the verifier;
/// otherwise the system's bundle and directory. `None` when at least one certificate loads, none
/// is malformed and nothing named is unreadable. Read once, when the callback client is first
/// built, so a sandbox that never sends a callback never pays for it.
fn trust_store_problem() -> Option<String> {
    use std::path::PathBuf;
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let (file, dir, names) = match (env("SSL_CERT_FILE"), env("SSL_CERT_DIR")) {
        (None, None) => {
            let p = openssl_probe::probe();
            (
                p.cert_file,
                p.cert_dir,
                ["the system bundle", "the system directory"],
            )
        }
        (f, d) => (f, d, ["SSL_CERT_FILE", "SSL_CERT_DIR"]),
    };
    let (mut good, mut bad) = (0, 0);
    let mut notes = Vec::new();
    let mut sources = Vec::new();
    let mut tally = |text: &str| {
        let (g, b) = count_pem(text);
        good += g;
        bad += b;
    };
    if let Some(f) = &file {
        sources.push(format!("{}={}", names[0], f.display()));
        match std::fs::read_to_string(f) {
            Ok(t) => tally(&t),
            Err(e) => notes.push(format!("{}: {e}", f.display())),
        }
    }
    if let Some(d) = &dir {
        sources.push(format!("{}={}", names[1], d.display()));
        match std::fs::read_dir(d) {
            Ok(entries) => {
                for e in entries.flatten().take(5000) {
                    if let Ok(t) = std::fs::read_to_string(e.path()) {
                        tally(&t);
                    }
                }
            }
            Err(e) => notes.push(format!("{}: {e}", d.display())),
        }
    }
    if good > 0 && bad == 0 && notes.is_empty() {
        return None;
    }
    let mut why = Vec::new();
    if good == 0 {
        why.push("no certificates load".to_string());
    }
    if bad > 0 {
        why.push(format!("{bad} certificate(s) do not parse"));
    }
    why.extend(notes);
    let from = if sources.is_empty() {
        "no trust store found".to_string()
    } else {
        sources.join(", ")
    };
    Some(format!("{} (from {from})", why.join("; ")))
}

/// One attempt: one `POST`, no redirects, the server verified, 30 s at most. Its status, or why not.
async fn attempt(ctx: &Ctx, target: &NotifyTarget, key: &Key) -> Result<u16, String> {
    let client = ctx
        .client
        .get_or_init(|| async {
            let _ = rustls::crypto::ring::default_provider().install_default();
            if let Some(problem) = trust_store_problem() {
                crate::log_warn!(
                    "https callbacks may fail to verify their server: the trust store has a \
                     problem ({problem}); set SSL_CERT_FILE or SSL_CERT_DIR to a valid bundle"
                );
            }
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
            // The url carries the receiver's signature: it must not reach the log or an observer.
            let e = e.without_url();
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod trust_tests {
    use super::count_pem;

    const GOOD: &str = "-----BEGIN CERTIFICATE-----\nMAMCAQA=\n-----END CERTIFICATE-----\n";

    #[test]
    fn certificates_are_counted_and_malformed_ones_are_told_apart() {
        assert_eq!(count_pem(""), (0, 0));
        assert_eq!(count_pem("not a pem"), (0, 0));
        assert_eq!(count_pem(GOOD), (1, 0));
        assert_eq!(count_pem(&GOOD.repeat(3)), (3, 0));
        let bad_b64 = "-----BEGIN CERTIFICATE-----\n!!!!\n-----END CERTIFICATE-----\n";
        assert_eq!(count_pem(&format!("{GOOD}{bad_b64}")), (1, 1));
        // Valid base64 that is not a DER sequence, and a block that never ends.
        let not_der = "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n";
        assert_eq!(count_pem(not_der), (0, 1));
        assert_eq!(count_pem("-----BEGIN CERTIFICATE-----\nMAMC"), (0, 1));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod owed_tests {
    use super::Notifier;
    use exec_notify::{Event, Outcome, step};

    fn owes(kernel: exec_notify::Job) -> bool {
        Notifier {
            kernel,
            ..Notifier::default()
        }
        .owes()
    }

    fn kept() -> exec_notify::Job {
        step(exec_notify::Job::default(), Event::Keep { notify: true }, 0).0
    }

    /// Owed from the keep, through every failed attempt and its backoff, until the twelfth failure
    /// abandons the count; a settling answer and removal also end it.
    #[test]
    fn a_callback_is_owed_until_settled_abandoned_or_removed() {
        let mut k = kept();
        let mut now = 0;
        for attempt in 1..=exec_notify::MAX_ATTEMPTS {
            assert!(owes(k), "in flight, attempt {attempt}");
            let failed = Event::AttemptEnded {
                outcome: Outcome::Status(500),
                draw: 0,
            };
            let (next, actions) = step(k, failed, now);
            k = next;
            if let Some(exec_notify::Action::WakeAt(at)) = actions.first() {
                assert!(owes(k), "backing off after attempt {attempt}");
                now = *at;
                k = step(k, Event::Wake, now).0;
            }
        }
        assert!(!owes(k), "abandoned at the attempt limit");
        assert!(!owes(step(kept(), Event::Removed, 0).0));
        let ok = Event::AttemptEnded {
            outcome: Outcome::Status(200),
            draw: 0,
        };
        assert!(!owes(step(kept(), ok, 0).0));
        assert!(!owes(exec_notify::Job::default()));
    }
}
