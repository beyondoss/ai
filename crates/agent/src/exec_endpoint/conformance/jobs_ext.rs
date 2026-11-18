//! The checks for what a job and a command can be given and asked: stderr told from stdout on one
//! clock, a command's own environment, positional writes to a job's stdin, and the `signal` op. The
//! parent module's doc ("Output streams", "Per-command environment", "Job stdin", "Job signal") is
//! the spec; each check measures it from the outside.

use std::time::{Duration, Instant};

use serde_json::json;

use super::{Check, EXEC_ENV, JOBS_SIGNAL_OP, JOBS_STDIN, JOBS_STREAMS, Verifier, fail, pass};
use crate::exec_endpoint::jobs::{
    JobEvent, JobId, JobScope, JobSpec, JobStatus, READ_MAX, SandboxJobs as _, Stream,
};
use crate::exec_endpoint::jobs_native::NativeJobs;

/// Every byte of a job's output so far, and which of them are stderr: read from offset 0 until the
/// window ends. Bounded: this verifier's jobs print a few bytes.
async fn read_all(
    jobs: &NativeJobs<'_>,
    scope: &JobScope,
    id: &JobId,
) -> std::io::Result<(Vec<u8>, Vec<(u64, u64)>, JobStatus)> {
    let (mut out, mut ranges) = (Vec::new(), Vec::new());
    let mut at = 0;
    loop {
        let c = jobs.read(scope, id, at, READ_MAX).await?;
        out.extend_from_slice(&c.data);
        ranges.extend(c.stderr);
        at = c.offset + c.data.len() as u64;
        if c.data.is_empty() || at >= c.end {
            return Ok((out, ranges, c.status));
        }
    }
}

/// Wait (bounded) until the job's output satisfies `want`; the output seen last.
async fn output_until(
    jobs: &NativeJobs<'_>,
    scope: &JobScope,
    id: &JobId,
    want: impl Fn(&[u8]) -> bool,
    within: Duration,
) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + within;
    loop {
        let (out, _, _) = read_all(jobs, scope, id)
            .await
            .map_err(|e| format!("read: {e}"))?;
        if want(&out) {
            return Ok(out);
        }
        if Instant::now() > deadline {
            return Err(format!(
                "no such output within {} s; the job printed {:?}",
                within.as_secs(),
                String::from_utf8_lossy(&out)
            ));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Wait (bounded) until the job has ended; its status.
async fn ended(
    jobs: &NativeJobs<'_>,
    scope: &JobScope,
    id: &JobId,
    within: Duration,
) -> Result<(JobStatus, Option<i32>), String> {
    let deadline = Instant::now() + within;
    loop {
        let c = jobs
            .read(scope, id, 0, 1)
            .await
            .map_err(|e| format!("read: {e}"))?;
        if matches!(c.status, JobStatus::Ended(_)) {
            return Ok((c.status, c.signal));
        }
        if Instant::now() > deadline {
            return Err(format!(
                "the job had not ended within {} s: {:?}",
                within.as_secs(),
                c.status
            ));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

const PAUSE: &str = "sleep 0.2 2>/dev/null || sleep 1";

impl Verifier {
    /// `printf AAA`, then `BBB` on stderr, then `CCC`, with a pause between so the order is not a
    /// race: frames name their pipe, a `read` names the stderr range, both on one clock.
    pub(super) async fn jobs_streams(&self) -> Check {
        let jobs = NativeJobs::over(&self.runner);
        let scope = self.scope("streams");
        let id = JobId::fresh();
        let script = format!("printf AAA; {PAUSE}; printf BBB >&2; {PAUSE}; printf CCC");
        let heard = std::sync::Mutex::new(Vec::<(u8, bool)>::new());
        let events = |e: JobEvent<'_>| {
            if let JobEvent::Output(b, s) = e {
                let mut h = heard.lock().unwrap_or_else(|e| e.into_inner());
                h.extend(b.iter().map(|&x| (x, s == Stream::Stderr)));
            }
        };
        let run = jobs
            .run_held(&scope, &id, &Self::spec(&script), &events)
            .await;
        let read = read_all(&jobs, &scope, &id).await;
        let _ = jobs.wipe(&scope).await;
        let check = (|| {
            run.map_err(|e| format!("held start: {e}"))?;
            let heard = heard.into_inner().unwrap_or_else(|e| e.into_inner());
            let want: Vec<(u8, bool)> = b"AAABBBCCC".iter().map(|&b| (b, b == b'B')).collect();
            if heard != want {
                return Err(format!(
                    "frames carried {:?}; want AAA on stdout, BBB on stderr, CCC on stdout, in \
                     that order (each frame keyed by the pipe its bytes came from)",
                    heard
                        .iter()
                        .map(|(b, e)| format!("{}{}", *b as char, if *e { "e" } else { "o" }))
                        .collect::<String>()
                ));
            }
            let (data, ranges, _) = read.map_err(|e| format!("read: {e}"))?;
            if data != b"AAABBBCCC" {
                return Err(format!(
                    "`read` returned {:?}, want the combined output AAABBBCCC",
                    String::from_utf8_lossy(&data)
                ));
            }
            if ranges != [(3, 3)] {
                return Err(format!(
                    "`read` named stderr ranges {ranges:?}; want [[3, 3]] (the BBB, on the \
                     output's own offsets)"
                ));
            }
            Ok(())
        })();
        match check {
            Ok(()) => pass(
                JOBS_STREAMS,
                "frames keyed by pipe, `read`'s `stderr_ranges` [[3,3]], combined bytes identical",
            ),
            Err(e) => fail(JOBS_STREAMS, e),
        }
    }

    /// A command's `env` reaches a one-shot and a job, the request wins over `HOME`, and every
    /// refused name runs nothing.
    pub(super) async fn exec_env(&self) -> Check {
        let r: Result<String, String> = async {
            let value = "a b=c $x\n'\"";
            let env = |k: &str, v: &str| vec![(k.to_string(), v.to_string())];
            // One-shot.
            let args = Self::args("printf %s \"$VFY_ENV\"");
            let resp = self
                .runner
                .send_env(
                    "sh",
                    &args,
                    None,
                    Duration::from_secs(10),
                    None,
                    false,
                    None,
                    &env("VFY_ENV", value),
                )
                .await
                .map_err(|e| format!("one-shot with env: {e}"))?;
            let got = self
                .runner
                .envelope(resp)
                .await
                .map_err(|e| e.to_string())?
                .into_result(self.runner.url(), 1 << 20)
                .map_err(|e| e.to_string())?;
            if got.stdout != value.as_bytes() {
                return Err(format!(
                    "a one-shot ran with VFY_ENV={:?}, want {value:?}",
                    String::from_utf8_lossy(&got.stdout)
                ));
            }
            // The request wins over the endpoint's own `HOME`.
            let args = Self::args("printf %s \"$HOME\"");
            let resp = self
                .runner
                .send_env(
                    "sh",
                    &args,
                    None,
                    Duration::from_secs(10),
                    None,
                    false,
                    None,
                    &env("HOME", "/vfy-home"),
                )
                .await
                .map_err(|e| format!("one-shot with HOME: {e}"))?;
            let home = self
                .runner
                .envelope(resp)
                .await
                .map_err(|e| e.to_string())?
                .into_result(self.runner.url(), 1 << 20)
                .map_err(|e| e.to_string())?;
            if home.stdout != b"/vfy-home" {
                return Err(format!(
                    "the request's HOME did not win: the command saw {:?}",
                    String::from_utf8_lossy(&home.stdout)
                ));
            }
            // A job.
            let jobs = NativeJobs::over(&self.runner);
            let scope = self.scope("env");
            let id = JobId::fresh();
            let mut spec = Self::spec("printf %s \"$VFY_ENV\"");
            spec.env = env("VFY_ENV", value);
            jobs.start_kept(&scope, &id, &spec)
                .await
                .map_err(|e| format!("job start with env: {e}"))?;
            let seen = output_until(&jobs, &scope, &id, |o| !o.is_empty(), Duration::from_secs(10))
                .await;
            let _ = jobs.wipe(&scope).await;
            if seen? != value.as_bytes() {
                return Err("a job did not see its env".into());
            }
            // Refusals run nothing: each is a `400`, and the marker the command would touch is absent.
            let marker = format!("/tmp/vfy-env-{}", self.nonce);
            let touch = Self::args(&format!("touch {marker}"));
            for (name, why) in [
                ("EXECD_X", "a name starting EXECD_"),
                ("EXECD_JOB_MARK", "EXECD_JOB_MARK"),
                ("1BAD", "a name starting with a digit"),
                ("A-B", "a name with a hyphen"),
                ("", "an empty name"),
            ] {
                match self
                    .runner
                    .send_env(
                        "sh",
                        &touch,
                        None,
                        Duration::from_secs(10),
                        None,
                        false,
                        None,
                        &env(name, "1"),
                    )
                    .await
                {
                    Err(e) if crate::exec_endpoint::endpoint_status(&e).is_some_and(|s| s.status == 400) => {}
                    Err(e) => return Err(format!("{why}: want a 400, got: {e}")),
                    Ok(_) => return Err(format!("{why} was accepted: want a 400")),
                }
            }
            let ran = self.sh(&format!("test -e {marker}; r=$?; rm -f {marker}; exit $r")).await?;
            if ran.code == Some(0) {
                return Err("a request refused for its env still ran the command".into());
            }
            Ok("applied to a one-shot and a job, the request's HOME wins, EXECD_* and bad names refused with nothing run".into())
        }
        .await;
        match r {
            Ok(d) => pass(EXEC_ENV, d),
            Err(e) => fail(EXEC_ENV, e),
        }
    }

    /// `cat` with its stdin open: writes land in position order however they arrive, a repeat applies
    /// nothing twice, a gap waits for its bytes, EOF ends the job; a job not started with
    /// `stdin_open` is not hung and has no stdin to write.
    pub(super) async fn jobs_stdin(&self) -> Check {
        let r: Result<String, String> = async {
            let jobs = NativeJobs::over(&self.runner);
            let scope = self.scope("stdin");
            let id = JobId::fresh();
            let mut cat = JobSpec {
                program: "cat".into(),
                args: Vec::new(),
                ..Self::spec("")
            };
            cat.stdin_open = true;
            jobs.start_kept(&scope, &id, &cat)
                .await
                .map_err(|e| format!("start with stdin_open: {e}"))?;
            let w = |at: u64, data: &'static [u8], eof: bool| {
                let (jobs, scope, id) = (&jobs, &scope, &id);
                async move {
                    jobs.write_stdin(scope, id, at, data, eof)
                        .await
                        .map_err(|e| format!("write at {at}: {e}"))
                }
            };
            let want = |n: u64, what: &str, got: crate::exec_endpoint::jobs::StdinWritten| {
                if got.written == n {
                    Ok(())
                } else {
                    Err(format!("{what}: `written` is {}, want {n}", got.written))
                }
            };
            want(6, "the first write", w(0, b"hello ", false).await?)?;
            want(12, "the second write", w(6, b"world\n", false).await?)?;
            want(
                12,
                "a retry of the first write",
                w(0, b"hello ", false).await?,
            )?;
            want(
                16,
                "an overlapping write",
                w(6, b"world\nmore", false).await?,
            )?;
            // A write ahead of the stream waits for the bytes before it, and both land in order.
            let (ahead, filler) = tokio::join!(w(20, b"Z", false), async {
                tokio::time::sleep(Duration::from_millis(300)).await;
                w(16, b"abcd", false).await
            });
            want(21, "a write ahead of the stream", ahead?)?;
            want(20, "the write that filled the gap", filler?)?;
            let expect = b"hello world\nmoreabcdZ";
            let got = output_until(
                &jobs,
                &scope,
                &id,
                |o| o.len() >= expect.len(),
                Duration::from_secs(10),
            )
            .await?;
            if got != expect {
                return Err(format!(
                    "the job read {:?}, want {:?}: writes must apply once, in position order",
                    String::from_utf8_lossy(&got),
                    String::from_utf8_lossy(expect)
                ));
            }
            let end = w(21, b"", true).await?;
            if !end.closed || end.written != 21 {
                return Err(format!("EOF answered {end:?}, want written 21, closed"));
            }
            let again = w(21, b"", true).await?;
            if !again.closed {
                return Err("closing a closed stdin must be idempotent".into());
            }
            ended(&jobs, &scope, &id, Duration::from_secs(10)).await?;
            match jobs.write_stdin(&scope, &id, 21, b"late", false).await {
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
                other => return Err(format!("a write after EOF answered {other:?}, want a 409")),
            }
            // Not opened: the job is not hung on a stdin nobody will write, and there is none to write.
            let plain = JobId::fresh();
            jobs.start_kept(
                &scope,
                &plain,
                &JobSpec {
                    program: "cat".into(),
                    args: Vec::new(),
                    ..Self::spec("")
                },
            )
            .await
            .map_err(|e| format!("start without stdin_open: {e}"))?;
            let (status, _) = ended(&jobs, &scope, &plain, Duration::from_secs(10))
                .await
                .map_err(|e| format!("a `cat` started without stdin_open hung: {e}"))?;
            if status != JobStatus::Ended(0) {
                return Err(format!("`cat` without stdin_open ended {status:?}, want 0"));
            }
            let sleeper = JobId::fresh();
            jobs.start_kept(&scope, &sleeper, &Self::spec("sleep 30"))
                .await
                .map_err(|e| format!("start: {e}"))?;
            match jobs.write_stdin(&scope, &sleeper, 0, b"x", false).await {
                Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => {}
                other => {
                    return Err(format!(
                        "a write to a job with no stdin answered {other:?}, want a 409"
                    ));
                }
            }
            match jobs
                .write_stdin(&scope, &JobId::fresh(), 0, b"x", false)
                .await
            {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                other => return Err(format!("a write to no job answered {other:?}, want a 404")),
            }
            let _ = jobs.wipe(&scope).await;
            Ok("in order, once, a gap waits, EOF ends the job, none without stdin_open".into())
        }
        .await;
        let _ = NativeJobs::over(&self.runner)
            .wipe(&self.scope("stdin"))
            .await;
        match r {
            Ok(d) => pass(JOBS_STDIN, d),
            Err(e) => fail(JOBS_STDIN, e),
        }
    }

    /// The `signal` op (with `SIGUSR1`/`SIGUSR2`, which no launcher pre-ignores: a shell's `cmd &`
    /// starts a child with `SIGINT` and `SIGQUIT` ignored, and an ignored signal cannot be trapped,
    /// so a sandbox-exec started that way could never show a `SIGINT` handler run): a trap in the job hears exactly what was sent, nothing outside the scope or
    /// the list is touched, an ended job is not signalled, a bad name is a 400.
    pub(super) async fn jobs_signal_op(&self) -> Check {
        let r: Result<String, String> = async {
            let jobs = NativeJobs::over(&self.runner);
            let scope = self.scope("sigop");
            let other = self.scope("sigop-other");
            let id = JobId::fresh();
            let script = format!(
                "trap 'printf 2' USR2; trap 'printf 1' USR1; printf ready; \
                 while :; do (trap '' USR2 USR1; {PAUSE}); done"
            );
            jobs.start_kept(&scope, &id, &Self::spec(&script))
                .await
                .map_err(|e| format!("start: {e}"))?;
            let sec = Duration::from_secs(10);
            output_until(&jobs, &scope, &id, |o| o == b"ready", sec).await?;
            let sent = jobs
                .signal(&scope, std::slice::from_ref(&id), "SIGUSR2")
                .await
                .map_err(|e| format!("signal SIGUSR2: {e}"))?;
            if sent != [id.clone()] {
                return Err(format!("SIGUSR2 to a running job answered sent {sent:?}"));
            }
            output_until(&jobs, &scope, &id, |o| o == b"ready2", sec).await?;
            jobs.signal(&scope, std::slice::from_ref(&id), "SIGUSR1")
                .await
                .map_err(|e| format!("signal SIGUSR1: {e}"))?;
            output_until(&jobs, &scope, &id, |o| o == b"ready21", sec).await?;
            // Another scope's signal reaches nothing, and an unknown id is absent, not an error.
            let foreign = jobs
                .signal(&other, &[id.clone(), JobId::fresh()], "SIGUSR1")
                .await
                .map_err(|e| format!("signal from another scope: {e}"))?;
            if !foreign.is_empty() {
                return Err(format!("another scope signalled the job: sent {foreign:?}"));
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
            let (out, _, _) = read_all(&jobs, &scope, &id).await.map_err(|e| e.to_string())?;
            if out != b"ready21" {
                return Err(format!(
                    "the job heard {:?} after a foreign-scope signal, want ready21",
                    String::from_utf8_lossy(&out)
                ));
            }
            // A name off the list is refused by the endpoint, and nothing is signalled.
            let bad = jobs
                .call(json!({
                    "op": "signal", "scope": scope.name(),
                    "ids": [id.as_str()], "signal": "SIGSTOP"
                }))
                .await;
            match bad {
                Err(e) if crate::exec_endpoint::endpoint_status(&e).is_some_and(|s| s.status == 400) => {}
                other => return Err(format!("SIGSTOP answered {other:?}, want a 400")),
            }
            // SIGKILL ends it as any death by signal 9; then a signal finds nothing to signal.
            jobs.signal(&scope, std::slice::from_ref(&id), "SIGKILL")
                .await
                .map_err(|e| format!("signal SIGKILL: {e}"))?;
            let (status, signal) = ended(&jobs, &scope, &id, sec).await?;
            if (status, signal) != (JobStatus::Ended(137), Some(9)) {
                return Err(format!(
                    "a SIGKILLed job reads as {status:?}, signal {signal:?}; want code 137, signal 9"
                ));
            }
            let late = jobs
                .signal(&scope, std::slice::from_ref(&id), "SIGTERM")
                .await
                .map_err(|e| format!("signal of an ended job: {e}"))?;
            if !late.is_empty() {
                return Err(format!("an ended job was signalled: sent {late:?}"));
            }
            Ok("exact signals heard, scope kept, ended and unknown ids absent, SIGSTOP refused".into())
        }
        .await;
        let _ = NativeJobs::over(&self.runner)
            .wipe(&self.scope("sigop"))
            .await;
        match r {
            Ok(d) => pass(JOBS_SIGNAL_OP, d),
            Err(e) => fail(JOBS_SIGNAL_OP, e),
        }
    }
}
