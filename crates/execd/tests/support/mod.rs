//! Running the real `execd` binary.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub mod http;

use beyond_ai_test_support::ports;

/// The binary under test.
pub const BIN: &str = env!("CARGO_BIN_EXE_execd");

/// A running `execd` process, killed when dropped.
pub struct Execd {
    pub child: Child,
    pub addr: std::net::SocketAddrV4,
    pub dir: tempfile::TempDir,
}

impl Execd {
    /// Start `execd` on a loopback port it picks, rooted in a fresh directory. `env` is added to its
    /// environment, `args` to its command line.
    pub fn start(args: &[&str], env: &[(&str, &str)]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut cmd = Command::new(BIN);
        cmd.args(["--listen", ports::CHILD_LISTEN, "--root"])
            .arg(dir.path())
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(dir.path().join("execd.log")).unwrap())
            .env_remove("EXECD_BEARER")
            .env_remove("EXECD_BEARER_FILE");
        for (k, v) in env {
            cmd.env(k, v);
        }
        let child = cmd.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        let addr = loop {
            if let Some(a) = ports::listening_on(child.id()).first() {
                break *a;
            }
            assert!(Instant::now() < deadline, "execd never listened");
            std::thread::sleep(Duration::from_millis(20));
        };
        Self { child, addr, dir }
    }

    pub fn url(&self) -> String {
        format!("http://{}/exec", self.addr)
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Execd {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
