//! A stdio MCP server that runs in the session's **sandbox**, spoken to from the replica.
//!
//! A hosted session's `.mcp.json` `command` entry cannot run on the replica (it is shared, and the
//! tenant's files are not there), so it runs where the tenant's files are: as a [sandbox
//! job](crate::exec_endpoint::jobs) in the session's own sandbox. The exec protocol has no live
//! stdin, so the job's stdin is a fifo the job script opens read-write (so it never sees EOF while
//! the job lives), and the transport is built from the two things the protocol does have:
//!
//! - **server → host**: [`SandboxJobs::follow`] streams the job's output, which is the server's
//!   stdout (its stderr goes to `<fifo>.log`, never into the JSON-RPC stream). The bytes go through
//!   [`mcp_stdio::pump`], the same line reader and size caps every stdio server uses.
//! - **host → server**: one plain exec, `cat >> <fifo>` with the bytes as `stdin_base64`.
//!
//! **Writes are serialized.** Two `cat >> fifo` running at once interleave once a message passes
//! `PIPE_BUF` (4096 bytes) and corrupt the JSON-RPC stream, so the transport has exactly one writer
//! task: everything rmcp writes is queued to it, and it runs one exec at a time (bytes queued
//! while one is in flight go in the next, in order). A message is whole in the fifo or not there.
//!
//! **One server per connector per session, started with the connection.** The job id is a digest
//! of the connector's name, and a connect first removes any job under that id (a previous replica's
//! server speaks to a client that no longer exists, and a fresh `initialize` to it is not part of
//! the protocol), then starts the server. Ending is the transport's: when it is dropped, the writer
//! task kills the job. A reap, a redeploy or a session's end reaches the same code, and the scope
//! ([`JobScope::for_session_mcp`]) is wiped when the session is deleted.
//!
//! The scope is the MCP one, not the `bash` table's: that table reads its scope back at every
//! reopen and stops what nobody kept.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sha2::Digest as _;
use tokio::io::AsyncWriteExt as _;
use tokio::sync::mpsc;

use super::exec::CommandRunner;
use crate::exec_endpoint::jobs::{JobEnd, JobEvent, JobId, JobScope, JobSpec, SandboxJobs};
use crate::grant::SandboxCommand;

/// How long one write may take. A write blocks while the fifo is full (the server not reading), so
/// this is also how long a stuck server is waited on before the connection is given up.
const WRITE_LIMIT: Duration = Duration::from_secs(60);

/// Re-attempts at following a job whose stream broke while the job lived.
const FOLLOW_RETRIES: u32 = 5;

/// Starts the server: the fifo is made and held open read-write as fd 3 (the server's stdin), the
/// server's stderr goes to `<fifo>.log`, and `env` (which runs `command` with the entry's variables
/// set) is exec'd in place so the job's process is the server.
///
/// The fifo is made under another name and renamed into place only once this shell holds it open,
/// so the path never exists without a reader behind it: a writer that finds the path cannot block
/// on a fifo nobody will ever read.
const START: &str = r#"f=$1; shift
d=${f%/*}
mkdir -p "$d" && chmod 700 "$d" || exit 1
t=$f.new
rm -f "$t" "$f.log"
mkfifo "$t" || exit 1
exec 3<>"$t"
mv "$t" "$f" || exit 1
exec env "$@" <&3 2>"$f.log""#;

/// Removes what a server left: the fifo, its half-made name and its stderr log.
const CLEAN: &str = r#"rm -f "$1" "$1.new" "$1.log""#;

/// Appends stdin to the fifo once it exists (the job may still be creating it on the first write).
/// Never creates it: `>>` on a missing path would make a plain file the server then never reads.
const APPEND: &str = r#"i=0
while [ ! -p "$1" ] && [ "$i" -lt 100 ]; do sleep 0.05; i=$((i + 1)); done
[ -p "$1" ] || exit 3
cat >> "$1""#;

/// How long a connect waits for a sandbox that is still starting (the front answers 503 once its
/// wake budget is spent while the VM is still being made) before it gives up and says so.
pub(crate) const STARTING_WAIT: Duration = Duration::from_secs(60);

/// The longest pause between two attempts.
const STARTING_BACKOFF_CAP: Duration = Duration::from_secs(5);

/// What to do with a failed start.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StartStep {
    /// Wait this long, then try again.
    Retry(Duration),
    /// Report the failure.
    GiveUp,
}

/// The decision, with no I/O: `starting` is whether the failure is the sandbox still starting
/// ([`crate::exec_endpoint::SANDBOX_STARTING`]), `attempt` counts failures so far (0 for the first),
/// `waited` is the time already spent waiting. Backoff doubles from 1 s to a cap, and the last
/// pause is cut so the total never passes [`STARTING_WAIT`].
pub(crate) fn start_step(starting: bool, attempt: u32, waited: Duration) -> StartStep {
    if !starting || waited >= STARTING_WAIT {
        return StartStep::GiveUp;
    }
    let pause = Duration::from_secs(1)
        .saturating_mul(1u32 << attempt.min(8))
        .min(STARTING_BACKOFF_CAP);
    StartStep::Retry(pause.min(STARTING_WAIT - waited))
}

/// Run `attempt` until it succeeds or [`start_step`] gives up, pausing with `pause`. The error is
/// the last attempt's, with the wait it cost named when it was a starting sandbox.
pub(crate) async fn start_waiting<T, F, Fut, P, PFut>(
    mut attempt: F,
    pause: P,
) -> std::io::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::io::Result<T>>,
    P: Fn(Duration) -> PFut,
    PFut: std::future::Future<Output = ()>,
{
    let mut waited = Duration::ZERO;
    let mut n = 0;
    loop {
        let e = match attempt().await {
            Ok(v) => return Ok(v),
            Err(e) => e,
        };
        // `contains`, not `starts_with`: the connect error reaches here wrapped by the session
        // (service.rs: "this session's sandbox is unreachable: the sandbox is starting: ..."), and a
        // prefix match never fired in production (I62 on AWS beyond, 11-22).
        let starting = e
            .to_string()
            .contains(crate::exec_endpoint::SANDBOX_STARTING);
        match start_step(starting, n, waited) {
            StartStep::Retry(d) => {
                pause(d).await;
                waited += d;
                n += 1;
            }
            StartStep::GiveUp if starting => {
                return Err(std::io::Error::other(format!(
                    "{e} (still starting after {}s)",
                    waited.as_secs()
                )));
            }
            StartStep::GiveUp => return Err(e),
        }
    }
}

/// The id of a connector's job: stable, so a connect can remove what an earlier one left.
fn job_id(name: &str) -> std::io::Result<JobId> {
    let digest = sha2::Sha256::digest(name.as_bytes());
    JobId::parse(&hex::encode(&digest[..8]))
        .ok_or_else(|| std::io::Error::other("a digest is not a job id"))
}

fn fifo_path(scope: &JobScope, id: &JobId) -> String {
    format!("/tmp/beyond-mcp-{}/{id}.in", scope.name())
}

/// What a connect hands rmcp: its read half, and the write half whose drop ends the server.
pub(crate) type Halves = (tokio::io::DuplexStream, SandboxWriter);

/// Start `server` (the connector `name`'s entry) as a job in `runner`'s sandbox, working in `cwd`,
/// and return the transport halves.
pub(crate) async fn connect(
    runner: Arc<dyn CommandRunner>,
    scope: JobScope,
    name: &str,
    server: &SandboxCommand,
    cwd: &str,
) -> std::io::Result<Halves> {
    runner.ready().await?;
    let store = runner.clone().sandbox_jobs().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this sandbox keeps no jobs, so it cannot run a stdio MCP server",
        )
    })?;
    let id = job_id(name)?;
    // Idempotent: nothing under this id is a no-op.
    store.kill(&scope, std::slice::from_ref(&id), true).await?;
    let fifo = fifo_path(&scope, &id);
    // A fifo left by an earlier connection (a killed job leaves its files) would be found by the
    // first write before this job has replaced it, and the write would wait on it for good.
    clean(&*runner, &fifo).await;
    let mut args = vec![
        "-c".to_owned(),
        START.to_owned(),
        "beyond-mcp".to_owned(),
        fifo.clone(),
    ];
    args.extend(server.env.iter().map(|(k, v)| format!("{k}={v}")));
    args.push(server.command.clone());
    args.extend(server.args.iter().cloned());
    let spec = JobSpec {
        program: "sh".to_owned(),
        args,
        cwd: Some(cwd.to_owned()),
        label: format!("mcp {name}"),
        env: Vec::new(),
        stdin_open: false,
        idempotency_key: None,
    };
    store.start_kept(&scope, &id, &spec).await?;

    // server -> host: the job's output, followed from byte 0 and carried over a stream that breaks
    // by following again from where it got to.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Vec<u8>>();
    let (mut feed, from_job) = tokio::io::duplex(64 * 1024);
    let ended = Arc::new(tokio::sync::Notify::new());
    {
        let (store, scope, id, ended) = (store.clone(), scope.clone(), id.clone(), ended.clone());
        tokio::spawn(async move {
            follow_to_end(&*store, &scope, &id, &out_tx).await;
            // The job is over (or unreachable for good): stop the writer, and with `out_tx`
            // dropped the read half ends, which closes the client's connection.
            ended.notify_one();
        });
    }
    tokio::spawn(async move {
        while let Some(bytes) = out_rx.recv().await {
            if feed.write_all(&bytes).await.is_err() {
                break;
            }
        }
    });
    let (rmcp_side, caps) = super::mcp_stdio::pump(from_job);

    // host -> server: one writer task, one exec at a time.
    let (tx, rx) = mpsc::unbounded_channel::<Vec<u8>>();
    tokio::spawn(write_loop(runner, store, scope, id, fifo, rx, ended));
    Ok((rmcp_side, SandboxWriter { tx, caps }))
}

/// Follow the job's output into `out` until it ends. A stream that breaks while the job lives is
/// followed again from the offset reached; a job that ended is the end.
async fn follow_to_end(
    store: &dyn SandboxJobs,
    scope: &JobScope,
    id: &JobId,
    out: &mpsc::UnboundedSender<Vec<u8>>,
) {
    let offset = AtomicU64::new(0);
    let sink = |event: JobEvent<'_>| match event {
        JobEvent::Output(bytes, _) => {
            offset.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            let _ = out.send(bytes.to_vec());
        }
        JobEvent::Gap(n) => {
            offset.fetch_add(n, Ordering::Relaxed);
        }
        JobEvent::Started(_) | JobEvent::ShellExit(_) => {}
    };
    let mut failures = 0;
    loop {
        match store
            .follow(scope, id, offset.load(Ordering::Relaxed), &sink)
            .await
        {
            Ok(JobEnd::Exit(_) | JobEnd::Signal(_)) => return,
            Err(e) => {
                failures += 1;
                if failures > FOLLOW_RETRIES {
                    tracing::warn!(error = %e, "an in-sandbox MCP server's output stream was lost");
                    return;
                }
                tokio::time::sleep(Duration::from_millis(100 << failures)).await;
            }
        }
    }
}

/// The single writer. Ends when the transport is dropped, a write fails, or the job ended; then
/// kills the job, which is what stops the server.
async fn write_loop(
    runner: Arc<dyn CommandRunner>,
    store: Arc<dyn SandboxJobs>,
    scope: JobScope,
    id: JobId,
    fifo: String,
    mut rx: mpsc::UnboundedReceiver<Vec<u8>>,
    ended: Arc<tokio::sync::Notify>,
) {
    let args = vec![
        "-c".to_owned(),
        APPEND.to_owned(),
        "beyond-mcp".to_owned(),
        fifo.clone(),
    ];
    loop {
        let mut batch = tokio::select! {
            next = rx.recv() => match next {
                Some(b) => b,
                None => break,
            },
            () = ended.notified() => break,
        };
        // What queued while the last exec was in flight goes in this one, in order.
        while let Ok(more) = rx.try_recv() {
            batch.extend_from_slice(&more);
        }
        let wrote = tokio::select! {
            r = runner.run_with_stdin("sh", &args, None, WRITE_LIMIT, &batch) => r,
            () = ended.notified() => break,
        };
        match wrote {
            Ok(r) if r.code == Some(0) => {}
            Ok(r) => {
                tracing::warn!(code = ?r.code, "a write to an in-sandbox MCP server failed");
                break;
            }
            Err(e) => {
                tracing::warn!(error = %e, "a write to an in-sandbox MCP server failed");
                break;
            }
        }
    }
    if let Err(e) = store.kill(&scope, std::slice::from_ref(&id), true).await {
        tracing::warn!(error = %e, "could not stop an in-sandbox MCP server");
    }
    clean(&*runner, &fifo).await;
}

/// Best effort: files in a sandbox's `/tmp` that a later connect removes anyway.
async fn clean(runner: &dyn CommandRunner, fifo: &str) {
    let args = ["-c", CLEAN, "beyond-mcp", fifo].map(str::to_owned);
    let _ = runner.run("sh", &args, None, Duration::from_secs(10)).await;
}

/// The write half rmcp holds. Queues what it is given for the single writer; dropping it ends the
/// writer, which stops the server.
pub(crate) struct SandboxWriter {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    caps: Arc<super::mcp_stdio::MessageCaps>,
}

impl tokio::io::AsyncWrite for SandboxWriter {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        self.caps.wrote(buf);
        std::task::Poll::Ready(match self.tx.send(buf.to_vec()) {
            Ok(()) => Ok(buf.len()),
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "the in-sandbox MCP server has ended",
            )),
        })
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn starting() -> std::io::Error {
        std::io::Error::other(format!(
            "{}: exec endpoint x returned 503",
            crate::exec_endpoint::SANDBOX_STARTING
        ))
    }

    /// The error as production delivers it: wrapped by the session's unreachable prefix.
    fn starting_wrapped() -> std::io::Error {
        std::io::Error::other(format!("{}: {}", crate::service::UNREACHABLE, starting()))
    }

    #[tokio::test]
    async fn a_starting_error_wrapped_by_the_session_is_still_retried() {
        let mut calls = 0u32;
        let r: std::io::Result<()> = start_waiting(
            || {
                calls += 1;
                let n = calls;
                async move {
                    if n < 3 {
                        Err(starting_wrapped())
                    } else {
                        Ok(())
                    }
                }
            },
            |_| async {},
        )
        .await;
        assert!(r.is_ok(), "a wrapped starting error must be retried: {r:?}");
        assert_eq!(calls, 3);
    }

    #[test]
    fn a_starting_sandbox_is_retried_with_capped_backoff_then_given_up_on() {
        let mut waited = Duration::ZERO;
        let mut pauses = vec![];
        for n in 0.. {
            match start_step(true, n, waited) {
                StartStep::Retry(d) => {
                    assert!(d <= STARTING_BACKOFF_CAP);
                    pauses.push(d.as_secs());
                    waited += d;
                }
                StartStep::GiveUp => break,
            }
        }
        assert_eq!(&pauses[..4], &[1, 2, 4, 5]);
        assert_eq!(waited, STARTING_WAIT);
    }

    #[test]
    fn any_other_failure_is_not_retried() {
        assert_eq!(start_step(false, 0, Duration::ZERO), StartStep::GiveUp);
    }

    #[tokio::test]
    async fn a_connect_that_finds_the_sandbox_starting_succeeds_once_it_is_up() {
        let calls = std::cell::Cell::new(0);
        let r = start_waiting(
            || {
                calls.set(calls.get() + 1);
                let up = calls.get() > 3;
                async move { if up { Ok(7) } else { Err(starting()) } }
            },
            |_| async {},
        )
        .await;
        assert_eq!(r.unwrap(), 7);
        assert_eq!(calls.get(), 4);
    }

    #[tokio::test]
    async fn a_sandbox_that_never_starts_fails_with_the_wait_named_and_a_real_refusal_at_once() {
        let e = start_waiting(|| async { Err::<(), _>(starting()) }, |_| async {})
            .await
            .unwrap_err();
        assert!(e.to_string().contains("still starting after 60s"), "{e}");
        let calls = std::cell::Cell::new(0);
        let e = start_waiting(
            || {
                calls.set(calls.get() + 1);
                async { Err::<(), _>(std::io::Error::other("the sandbox did not answer: x")) }
            },
            |_| async {},
        )
        .await
        .unwrap_err();
        assert_eq!(calls.get(), 1);
        assert!(e.to_string().contains("did not answer"));
    }

    fn front_503(code: &str) -> std::io::Error {
        let body = format!(r#"{{"code":"{code}","message":"words"}}"#);
        std::io::Error::other(crate::exec_endpoint::EndpointStatus::new(
            503,
            body.clone(),
            format!("exec endpoint x returned 503: {body}"),
        ))
    }

    #[tokio::test]
    async fn only_the_starting_code_is_waited_on_an_at_limit_503_fails_at_once() {
        // Through the probe's own classification: what the front's body says, not its sentence.
        let calls = std::cell::Cell::new(0);
        let e = start_waiting(
            || {
                calls.set(calls.get() + 1);
                let err = crate::exec_endpoint::probe_error(&front_503("sandbox_limit_reached"));
                async move { Err::<(), _>(std::io::Error::other(err)) }
            },
            |_| async {},
        )
        .await
        .unwrap_err();
        assert_eq!(calls.get(), 1);
        assert!(e.to_string().contains("sandbox_limit_reached"), "{e}");

        let calls = std::cell::Cell::new(0);
        let r = start_waiting(
            || {
                calls.set(calls.get() + 1);
                let up = calls.get() > 2;
                let err = crate::exec_endpoint::probe_error(&front_503("sandbox_starting"));
                async move {
                    if up {
                        Ok(())
                    } else {
                        Err(std::io::Error::other(err))
                    }
                }
            },
            |_| async {},
        )
        .await;
        assert!(r.is_ok());
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn a_bare_503_without_the_starting_code_is_not_starting() {
        let bare = std::io::Error::other(crate::exec_endpoint::EndpointStatus::new(
            503,
            "the sandbox is starting",
            "x",
        ));
        assert!(crate::exec_endpoint::probe_error(&bare).starts_with("the sandbox did not answer"));
    }

    #[test]
    fn a_connectors_job_id_is_stable_and_a_valid_id() {
        let id = |n| job_id(n).unwrap();
        assert_eq!(id("notes"), id("notes"));
        assert_ne!(id("notes"), id("files"));
        assert_eq!(id("notes").as_str().len(), 16);
    }
}
