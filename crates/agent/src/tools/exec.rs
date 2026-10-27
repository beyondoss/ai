//! The host command runner.
//!
//! `bash` runs real commands through [`RealRunner`]. The seam it implements ([`CommandRunner`], the
//! result and job types, the output capture helpers) and the process-group kill machinery live in
//! the `beyond-sandbox` crate and are re-exported here, so the agent's tools and tests keep one
//! path for them.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::watch;

pub use beyond_sandbox::exec::*;
pub use beyond_sandbox::procgroup::*;

/// The signal that killed `status`, if one did. `ExitStatus::code()` returns `None` in exactly this
/// case on Unix, which is why a `None` code must never be read as "fine" — see [`ExecResult::signal`].
#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(status)
}

/// Non-Unix has no signal concept — a `None` exit code there really is just "unknown".
#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

/// Human name for the signals a command realistically dies from, so the reported message says
/// something a model (or a human reading the transcript) can act on. Anything outside this short list
/// is reported by number alone rather than pulling in a libc-name lookup for it.
pub fn signal_name(signal: i32) -> Option<&'static str> {
    Some(match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        3 => "SIGQUIT",
        6 => "SIGABRT",
        8 => "SIGFPE",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        13 => "SIGPIPE",
        15 => "SIGTERM",
        24 => "SIGXCPU",
        25 => "SIGXFSZ",
        _ => return None,
    })
}

/// Spawns the command for real, capturing stdout/stderr and enforcing a wall-clock timeout
/// (`kill_on_drop` reaps the child if it overruns).
pub struct RealRunner;

impl RealRunner {
    /// Shared spawn/capture body. `on_chunk`, when present, is called with each stdout/stderr chunk as
    /// it streams (live progress); `run`, `run_streaming` and `run_with_stdin` all funnel through here.
    /// `stdin` is `None` for everything a model runs — see the `.stdin(..)` comment below.
    /// `bash`'s jobs go through [`super::bash_proc::run_tracked`] instead
    /// ([`CommandRunner::run_streaming_tracked`]). `on_stderr`, when present, takes stderr's chunks
    /// instead of `on_chunk` (`run_attached`, whose two streams must stay apart).
    #[allow(clippy::too_many_arguments)]
    async fn exec(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: Option<ChunkSink<'_>>,
        on_stderr: Option<ChunkSink<'_>>,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecResult> {
        // No `PATH` manipulation here — `Command` inherits the parent process's environment
        // (including `PATH`) as-is. Pi-parity note: this is a deliberate, documented DIVERGENCE, not
        // an oversight. Pi's `getShellEnv()` (`packages/coding-agent/src/utils/shell.ts`) prepends a
        // managed-binaries directory (pinned `rg`/`fd` it downloads itself) onto `PATH` for every
        // spawn, so its `grep`/`find` tools work even on a host with neither installed. Beyond's own
        // `grep`/`find` tools are native (ripgrep/globset linked in-process, see `grep.rs`/`find.rs`) —
        // unaffected either way — so this only matters for an arbitrary model-invoked `bash: rg ...` /
        // `fd ...` on a host that happens to lack them. Not worth a from-scratch bundled-binaries
        // mechanism just to cover that narrow case.
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args)
            // Closed, not inherited: a model-run command has no business reading the agent process's
            // real stdin (a shared terminal in `run`, or `serve`'s NDJSON control pipe). Without this,
            // a command that tries to read stdin (bare `cat`, `read`, a prompt left off `-y`) blocks
            // forever waiting for input that will never come, instead of seeing immediate EOF — matches
            // pi's own `spawn(..., { stdio: ["ignore", "pipe", "pipe"] })`. Piped only for
            // `run_with_stdin`, which carries bytes the *agent* supplied (file content), never the
            // agent's own stdin.
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Run the command as its own process-group leader so a timeout can kill the *whole* tree —
        // a `sh -c "foo &"` that backgrounds children would otherwise leak them as orphans, since
        // `kill_on_drop` only reaps the direct child.
        #[cfg(unix)]
        cmd.process_group(0);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let mut child = cmd.spawn()?;
        // Arms a group-kill for the lifetime of this call: `kill_on_drop` above only reaps the
        // *direct* child, so a future dropped mid-`await` (cancellation, not just a timeout) would
        // otherwise leak a backgrounded grandchild as an orphan. `disarm`ed below once the child has
        // actually exited on its own; still armed on both the timeout branch and an external drop, so
        // either one reaches the whole process group.
        let mut guard = GroupKillGuard {
            group: child.id().map(GroupId::leader),
        };
        // Streaming (`bash`'s live path) hands every chunk straight to `on_chunk`'s sink, then ignores
        // the `stdout`/`stderr`/`truncated` fields entirely (see the `Ok` arm's comment and `bash.rs`).
        // So when streaming we skip the ≤256 KiB head/tail `Capture` *and* the final `from_utf8_lossy`
        // allocations below — they'd only be produced to be discarded. The non-streaming `run` path
        // (`ShellFs`'s commands, worktree `git`) keeps full capture, since it *does* read these
        // fields.
        let streaming = on_chunk.is_some() || on_stderr.is_some();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // Fed concurrently with the drains, not before them — see `feed_stdin`.
        let feed = feed_stdin(child.stdin.take(), stdin);
        // Drain both pipes *concurrently* with the wait: a child that fills one pipe's OS buffer
        // while we read only the other would deadlock, and an unread pipe stalls the child's exit.
        // `exited` fans the wait's completion out to both drains so each can switch from "block on
        // reads" to "release after a short idle grace" the moment the direct child exits (see
        // `POST_EXIT_IDLE_GRACE`), independent of whether the *other* stream's holder has let go yet.
        let (exited_tx, exited_rx) = watch::channel(false);
        let wait = async {
            let status = child.wait().await;
            let _ = exited_tx.send(true);
            status
        };
        let collect = async {
            let (status, out, err, fed) = tokio::join!(
                wait,
                drain_capped(
                    stdout,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_chunk,
                    exited_rx.clone()
                ),
                drain_capped(
                    stderr,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_stderr.or(on_chunk),
                    exited_rx.clone()
                ),
                feed,
            );
            (status, out, err, fed)
        };
        match tokio::time::timeout(timeout, collect).await {
            // Lossy on purpose, not by oversight: `bash` (this crate's primary consumer) never
            // reads these two fields for real output — its `run_streaming` sink appends every chunk
            // to its `OutputAccumulator` as raw bytes as they arrive (see `bash.rs`'s `sink`
            // closure), and only falls back to `stdout`/`stderr` here when nothing streamed (a test
            // double, a remote runner). The live path this struct actually feeds for `bash` never
            // goes through `from_utf8_lossy`. The non-streaming callers that *do* read these fields
            // (`ShellFs`, worktree `git`) run commands whose output is text, or base64-encode bytes
            // that may not be (a worktree patch; see `worktree.rs`).
            Ok((status, (stdout, out_trunc), (stderr, err_trunc), fed)) => {
                guard.disarm();
                let status = status?;
                fed?;
                Ok(ExecResult {
                    code: status.code(),
                    signal: exit_signal(&status),
                    // Empty on the streaming path: `drain_capped` returned empty capture buffers there
                    // (nothing to decode), and `bash` never reads these anyway.
                    stdout: if streaming {
                        String::new()
                    } else {
                        String::from_utf8_lossy(&stdout).into_owned()
                    },
                    stderr: if streaming {
                        String::new()
                    } else {
                        String::from_utf8_lossy(&stderr).into_owned()
                    },
                    timed_out: false,
                    truncated: out_trunc || err_trunc,
                })
            }
            Err(_) => {
                // Timed out: unlike a future dropped mid-`await` (`GroupKillGuard`'s own `Drop`, which
                // can't `.await`), we're still in an async context here — so actually wait for the
                // process-group kill to finish via `spawn_blocking` before returning, rather than
                // firing it and hoping it lands before a caller re-checks side effects. Disarms the
                // guard (via `take`) first so its own `Drop` doesn't *also* fire a redundant kill.
                #[cfg(unix)]
                if let Some(group) = guard.group.take() {
                    let _ = tokio::task::spawn_blocking(move || kill_process_group(group)).await;
                }
                drop(guard);
                Ok(ExecResult {
                    code: None,
                    stdout: String::new(),
                    stderr: "command exceeded its timeout".into(),
                    timed_out: true,
                    ..Default::default()
                })
            }
        }
    }
}

#[async_trait]
impl CommandRunner for RealRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, timeout, None, None, None)
            .await
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, timeout, Some(on_chunk), None, None)
            .await
    }

    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        super::bash_proc::run_tracked(program, args, cwd, hooks).await
    }

    async fn tracks_jobs(&self) -> bool {
        true
    }

    /// For tests of the sandbox-job scripts, which run them on this host. No deadline beyond
    /// `STREAM_DEADLINE`'s; dropping the call kills the child's process group.
    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: ChunkSink<'_>,
        on_stderr: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            crate::exec_endpoint::STREAM_DEADLINE,
            Some(on_stdout),
            Some(on_stderr),
            None,
        )
        .await
    }

    async fn attaches(&self) -> bool {
        true
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, timeout, None, None, Some(stdin))
            .await
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawned_commands_see_stdin_already_closed_not_the_real_terminal() {
        // Pi-parity audit H1: a model-run command must never inherit the agent process's real stdin —
        // `serve`'s NDJSON control pipe or an interactive `run`'s terminal. `cat` with no args reads
        // stdin until EOF; if stdin were inherited (and this test's own harness stdin stays open) it
        // would hang until the timeout fired. With stdin closed, `cat` sees immediate EOF and exits
        // clean well within the generous timeout below.
        let res = RealRunner
            .run("cat", &[], None, Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(res.code, Some(0), "got: {res:?}");
        assert!(!res.timed_out, "cat blocked on stdin instead of seeing EOF");
        assert_eq!(res.stdout, "");
    }

    #[tokio::test]
    async fn run_with_stdin_feeds_the_bytes_then_closes_the_pipe() {
        // `cat` exits only on EOF, so a result at all proves the pipe was closed after the write.
        let res = RealRunner
            .run_with_stdin("cat", &[], None, Duration::from_secs(10), b"stdin-probe")
            .await
            .unwrap();
        assert_eq!(res.code, Some(0), "got: {res:?}");
        assert_eq!(res.stdout, "stdin-probe");
    }

    #[tokio::test]
    async fn run_with_stdin_feeds_while_draining_so_big_input_and_output_cannot_deadlock() {
        // 4 MiB through `cat` is far past both pipe buffers: written before draining (or drained
        // before writing), the child blocks on a full stdout while we block on its full stdin.
        let input = vec![b'x'; 4 * 1024 * 1024];
        let res = RealRunner
            .run_with_stdin("cat", &[], None, Duration::from_secs(30), &input)
            .await
            .unwrap();
        assert!(!res.timed_out, "deadlocked until the timeout");
        assert_eq!(res.code, Some(0));
        assert!(res.truncated, "4 MiB of output must still be capped");

        let res = RealRunner
            .run_with_stdin("wc", &["-c".into()], None, Duration::from_secs(30), &input)
            .await
            .unwrap();
        assert_eq!(
            res.stdout.trim(),
            input.len().to_string(),
            "every byte arrived"
        );
    }

    // Both tests below race a `kill -KILL -<pgid>` against a backgrounded grandchild's own delayed
    // write. See `kill_process_group`'s own doc comment for what these margins are guarding against —
    // a real GitHub Actions run proved this isn't merely a delivery-latency question (the grandchild
    // ran its *entire* multi-second sleep to completion, unaffected, while the group leader died
    // promptly) — `sweep_kill_remaining_group_members` is the actual fix; these wider margins
    // (vs. the original 1s/1.2s, sized for a fast uncontended dev box) are cheap insurance on top.
    const GRANDCHILD_DELAY: Duration = Duration::from_secs(3);
    const SAFETY_WAIT: Duration = Duration::from_millis(3500);

    #[tokio::test]
    async fn timeout_kills_backgrounded_grandchildren() {
        // A backgrounded grandchild would, after GRANDCHILD_DELAY, write a marker file. The command
        // times out at 300ms; the process-group kill must reach the grandchild so the marker is never
        // written.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("leaked");
        let script = format!(
            "( sleep {}; echo leaked > {} ) & sleep 30",
            GRANDCHILD_DELAY.as_secs(),
            marker.display()
        );
        let res = RealRunner
            .run(
                "sh",
                &["-c".into(), script],
                None,
                Duration::from_millis(300),
            )
            .await
            .unwrap();
        assert!(res.timed_out);

        // Wait past when the grandchild would have written the marker; it must not exist.
        tokio::time::sleep(SAFETY_WAIT).await;
        assert!(
            !marker.exists(),
            "backgrounded grandchild survived the timeout — process group not killed"
        );
    }

    #[tokio::test]
    async fn cancelling_the_future_kills_backgrounded_grandchildren() {
        // Same fixture as `timeout_kills_backgrounded_grandchildren`, but this time the *caller* drops
        // the `run` future outright (as cancellation does) rather than the command's own timeout
        // firing — `kill_on_drop` alone only reaps the direct child, so without `GroupKillGuard` the
        // backgrounded grandchild would survive this path even though the timeout path already
        // handles it.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("leaked");
        let script = format!(
            "( sleep {}; echo leaked > {} ) & sleep 30",
            GRANDCHILD_DELAY.as_secs(),
            marker.display()
        );

        let args = vec!["-c".to_string(), script];
        tokio::select! {
            _ = RealRunner.run(
                "sh",
                &args,
                None,
                Duration::from_secs(30), // long enough that the timeout branch never fires
            ) => panic!("the run future should have been dropped first"),
            _ = tokio::time::sleep(Duration::from_millis(200)) => {
                // The `run` future is dropped here as `select!`'s other arm is discarded.
            }
        }

        // Wait past when the grandchild would have written the marker; it must not exist.
        tokio::time::sleep(SAFETY_WAIT).await;
        assert!(
            !marker.exists(),
            "backgrounded grandchild survived dropping the future — process group not killed"
        );
    }

    #[tokio::test]
    async fn resolves_promptly_when_a_detached_child_holds_the_pipe_open_but_stays_quiet() {
        // pi issue #5303: a shell exits immediately but a backgrounded sleeper inherits the stdout
        // pipe and holds it open without writing. We must release via the post-exit idle grace rather
        // than block until the pipe's true EOF (which wouldn't arrive for 30s here) — and, before the
        // fix, not until the overall command timeout either.
        let start = std::time::Instant::now();
        let res = RealRunner
            .run(
                "sh",
                &["-c".into(), "printf 'DONE\\n'; ( sleep 30 ) &".into()],
                None,
                Duration::from_secs(25), // far longer than the idle grace should ever need
            )
            .await
            .unwrap();
        assert!(!res.timed_out);
        assert_eq!(res.stdout, "DONE\n");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "took {:?}; must release via the idle grace, not wait on the held-open pipe",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn captures_output_emitted_after_exit_while_a_detached_child_holds_stdout_open() {
        // Companion to the test above: the idle grace must not cut off output that's still actively
        // arriving from a detached descendant — only a genuinely quiet handle releases early. Ticks
        // land every 50ms, each one re-arming the 100ms grace, well past a single grace window.
        let res = RealRunner
            .run(
                "sh",
                &[
                    "-c".into(),
                    "printf 'HEAD\\n'; ( for i in 1 2 3 4 5 6; do sleep 0.05; printf \"TICK$i\\n\"; done ) &"
                        .into(),
                ],
                None,
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert!(!res.timed_out);
        assert!(res.stdout.contains("HEAD"), "got: {:?}", res.stdout);
        assert!(res.stdout.contains("TICK6"), "got: {:?}", res.stdout);
    }

    #[tokio::test]
    async fn run_streaming_emits_chunks_reconstructing_stdout() {
        // The streaming runner must hand each output chunk to `on_chunk` as it arrives, and the
        // concatenated chunks must reconstruct the full stdout (it streams before the cap, not after).
        // The sink *is* the output on this path — the `ExecResult::stdout`/`stderr` fields are left
        // empty on purpose (the head/tail capture is skipped when streaming, since the only consumer,
        // `bash`, reads the streamed chunks, never these fields — see the `exec` `Ok` arm).
        let collected = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let c = collected.clone();
        let sink = move |bytes: &[u8]| c.lock().unwrap().extend_from_slice(bytes);
        let res = RealRunner
            .run_streaming(
                "sh",
                &["-c".into(), "printf 'hello\\nworld\\n'".into()],
                None,
                Duration::from_secs(5),
                &sink,
            )
            .await
            .unwrap();
        assert_eq!(
            res.stdout, "",
            "streaming skips the discarded capture; the sink is the source of truth"
        );
        let streamed = String::from_utf8(collected.lock().unwrap().clone()).unwrap();
        assert_eq!(
            streamed, "hello\nworld\n",
            "streamed chunks must reconstruct stdout"
        );
    }

    #[tokio::test]
    async fn kill_process_group_completes_promptly_when_the_group_kill_finds_nothing_to_kill() {
        // Track L10: exercises the two-attempt sequence end to end. Once the process has already
        // exited and been reaped, the group kill (`kill -KILL -pgid`) finds nothing (ESRCH) and the
        // direct-pid fallback that follows must also complete cleanly — no hang, no panic — rather
        // than the two-step sequence somehow blocking on a process that's already gone.
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("true")
            .spawn()
            .unwrap();
        let pid = child.id().unwrap();
        child.wait().await.unwrap(); // let it actually exit and get reaped

        tokio::time::timeout(
            Duration::from_secs(5),
            tokio::task::spawn_blocking(move || kill_process_group(GroupId::leader(pid))),
        )
        .await
        .expect("kill_process_group must not hang when there's nothing left to kill")
        .expect("kill_process_group must not panic");
    }

    #[tokio::test]
    async fn bounded_capture_survives_a_firehose() {
        // `yes` emits unboundedly fast; with the old `wait_with_output()` this buffered the whole
        // stream and OOMed. Capture must hold ~HEAD+TAIL, return promptly on timeout, and the kept
        // bytes (after lossy decode) must start at the true beginning of the stream.
        let res = RealRunner
            .run(
                "sh",
                &["-c".into(), "yes AAAAAAAA".into()],
                None,
                Duration::from_millis(400),
            )
            .await
            .unwrap();
        // Timed out (yes never exits), so stdout was discarded by the timeout branch — prove instead
        // with a bounded *finite* firehose that exits on its own.
        assert!(res.timed_out);

        let res = RealRunner
            .run(
                "sh",
                // ~5 MiB of output that exits cleanly — far above the 256 KiB capture window.
                &[
                    "-c".into(),
                    "yes ABCDEFGHIJKLMNOPQRST | head -n 250000".into(),
                ],
                None,
                Duration::from_secs(30),
            )
            .await
            .unwrap();
        assert!(!res.timed_out);
        assert!(
            res.truncated,
            "5 MiB through a 256 KiB window must drop the middle"
        );
        assert!(
            res.stdout.len() <= STREAM_HEAD + STREAM_TAIL + 64,
            "captured {} bytes; expected bounded to ~{}",
            res.stdout.len(),
            STREAM_HEAD + STREAM_TAIL
        );
        assert!(
            res.stdout.starts_with("ABCDEFGHIJKLMNOPQRST"),
            "head preserved"
        );
        assert!(
            res.stdout.trim_end().ends_with("ABCDEFGHIJKLMNOPQRST"),
            "tail preserved"
        );
    }
}
