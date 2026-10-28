//! A seeded fault-injecting transport between `ShellFs` and the real HTTP runner.
//!
//! Per request, with probability `pct`, one of: fail **before** sending (the command never ran);
//! drop the response **after** the command ran (it did, and the client never learns the outcome);
//! time out before or after; or just delay. The draw is a splitmix64 stream seeded by a value Hegel
//! draws, so a shrunk failure replays the same faults for the same request sequence.
//!
//! Faults are off while `ShellFs` probes its target (`enabled` starts false) and while the oracle
//! runs; they apply to the file operation under test only.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use beyond_agent::tools::exec::{CommandRunner, ExecResult};

#[derive(Default, Debug, Clone)]
pub struct Report {
    /// A fault turned a request into an error.
    pub errored: bool,
    /// The cleanup command of a failed write (`rm -f "$tmp"`) was itself failed by a fault, so a temp
    /// the write left behind has no remaining owner. The spec's cleanup is one best-effort command.
    pub cleanup_faulted: bool,
    pub requests: usize,
}

struct State {
    rng: u64,
    pct: u64,
    enabled: bool,
    report: Report,
    total_errors: u64,
}

pub struct FaultRunner {
    inner: Arc<dyn CommandRunner>,
    state: Mutex<State>,
}

#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    FailBefore,
    DropAfter,
    TimeoutBefore,
    TimeoutAfter,
    Delay(u64),
}

fn is_cleanup(args: &[String]) -> bool {
    args.get(1)
        .is_some_and(|s| s.ends_with("rm -f \"$tmp\"") && !s.contains("mv -f"))
}

impl FaultRunner {
    pub fn new(inner: Arc<dyn CommandRunner>, seed: u64, pct: u64) -> Arc<Self> {
        Arc::new(Self {
            inner,
            state: Mutex::new(State {
                rng: seed,
                pct,
                enabled: false,
                report: Report::default(),
                total_errors: 0,
            }),
        })
    }

    pub fn enable(&self, on: bool) {
        self.state.lock().unwrap().enabled = on;
    }

    /// What happened since the last call.
    pub fn take(&self) -> Report {
        std::mem::take(&mut self.state.lock().unwrap().report)
    }

    pub fn total_errors(&self) -> u64 {
        self.state.lock().unwrap().total_errors
    }

    fn roll(&self, args: &[String]) -> Fault {
        let mut s = self.state.lock().unwrap();
        if !s.enabled {
            return Fault::None;
        }
        s.report.requests += 1;
        let pct = s.pct;
        let mut next = || {
            s.rng = s.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s.rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        if next() % 100 >= pct {
            return Fault::None;
        }
        let f = match next() % 6 {
            0 | 1 => Fault::FailBefore,
            2 | 3 => Fault::DropAfter,
            4 => {
                if next() % 2 == 0 {
                    Fault::TimeoutBefore
                } else {
                    Fault::TimeoutAfter
                }
            }
            _ => Fault::Delay(next() % 20),
        };
        let s = &mut *s;
        if !matches!(f, Fault::Delay(_)) {
            s.report.errored = true;
            s.total_errors += 1;
            if is_cleanup(args) {
                s.report.cleanup_faulted = true;
            }
        }
        f
    }

    async fn go<F>(&self, args: &[String], f: F) -> std::io::Result<ExecResult>
    where
        F: std::future::Future<Output = std::io::Result<ExecResult>>,
    {
        let injected = |what: &str, timeout: bool| {
            if timeout {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("exec endpoint timed out: injected {what}"),
                )
            } else {
                std::io::Error::other(format!("exec endpoint unreachable: injected {what}"))
            }
        };
        match self.roll(args) {
            Fault::None => f.await,
            Fault::Delay(ms) => {
                tokio::time::sleep(Duration::from_millis(ms)).await;
                f.await
            }
            Fault::FailBefore => Err(injected("failure before send", false)),
            Fault::TimeoutBefore => Err(injected("timeout before send", true)),
            Fault::DropAfter => {
                let _ = f.await;
                Err(injected("dropped response", false))
            }
            Fault::TimeoutAfter => {
                let _ = f.await;
                Err(injected("timeout after the command ran", true))
            }
        }
    }
}

#[async_trait]
impl CommandRunner for FaultRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.go(args, self.inner.run(program, args, cwd, timeout))
            .await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        self.go(
            args,
            self.inner
                .run_with_stdin(program, args, cwd, timeout, stdin),
        )
        .await
    }

    async fn run_streaming(
        &self,
        _: &str,
        _: &[String],
        _: Option<&str>,
        _: std::time::Duration,
        _: beyond_agent::tools::exec::ChunkSink<'_>,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        unimplemented!("a test double that only runs to completion")
    }

    async fn run_streaming_tracked(
        &self,
        _: &str,
        _: &[String],
        _: Option<&str>,
        _: &beyond_agent::tools::exec::JobHooks<'_>,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        unimplemented!("a test double that only runs to completion")
    }

    async fn run_attached(
        &self,
        _: &str,
        _: &[String],
        _: Option<&str>,
        _: beyond_agent::tools::exec::ChunkSink<'_>,
        _: beyond_agent::tools::exec::ChunkSink<'_>,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        unimplemented!("a test double that only runs to completion")
    }
}
