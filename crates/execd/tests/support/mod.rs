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
    /// Started through `sudo`: the real execd's pid (the child is `sudo`).
    root_pid: Option<u32>,
}

impl Execd {
    /// Start `execd` on a loopback port it picks, rooted in a fresh directory. `env` is added to its
    /// environment, `args` to its command line.
    pub fn start(args: &[&str], env: &[(&str, &str)]) -> Self {
        Self::start_on(ports::CHILD_LISTEN, args, env)
    }

    /// [`Self::start`], listening on `listen` (a port-0 address, read back from the socket).
    pub fn start_on(listen: &str, args: &[&str], env: &[(&str, &str)]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut cmd = Command::new(BIN);
        cmd.args(["--listen", listen, "--root"])
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
        Self {
            child,
            addr,
            dir,
            root_pid: None,
        }
    }

    /// Run `execd` with these arguments (no `--listen` or `--root` added) until it exits on its own,
    /// for a start that must be refused: its exit status and everything it logged. Killed after
    /// 20 s, which a test reads as "it did not refuse".
    pub fn run_to_exit(
        args: &[&str],
        env: &[(&str, &str)],
        as_root: bool,
    ) -> (std::process::ExitStatus, String) {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("execd.log");
        let mut cmd = if as_root {
            let mut c = Command::new("sudo");
            c.args(["-n", "env"]);
            for (k, v) in env {
                c.arg(format!("{k}={v}"));
            }
            c.arg(BIN);
            c
        } else {
            Command::new(BIN)
        };
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .env_remove("EXECD_BEARER")
            .env_remove("EXECD_BEARER_FILE");
        if !as_root {
            for (k, v) in env {
                cmd.env(k, v);
            }
        }
        let mut child = cmd.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        let status = loop {
            if let Some(s) = child.try_wait().unwrap() {
                break s;
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                break child.wait().unwrap();
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        (status, std::fs::read_to_string(&log).unwrap_or_default())
    }

    /// Whether this host lets the tests run something as root: they are root, or `sudo -n` works.
    pub fn can_be_root() -> bool {
        nix_is_root()
            || Command::new("sudo")
                .args(["-n", "true"])
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
    }

    /// [`Self::start`], with execd running as root (through `sudo -n`, unless the test already is
    /// root), listening on a loopback port it picks, rooted in a fresh directory that anyone may
    /// write (the `--run-as` user is not the test's). `None`, after saying so, when this host gives
    /// the tests no way to be root.
    pub fn start_as_root(args: &[&str], env: &[(&str, &str)]) -> Option<Self> {
        Self::start_as_root_in(None, args, env)
    }

    /// [`Self::start_as_root`], with `--root` set to `root` instead of the fresh directory.
    pub fn start_as_root_in(
        root: Option<&std::path::Path>,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Option<Self> {
        if !Self::can_be_root() {
            eprintln!("SKIPPED: this test needs root (or passwordless sudo), and has neither");
            return None;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(
            dir.path(),
            std::os::unix::fs::PermissionsExt::from_mode(0o777),
        )
        .unwrap();
        let pidfile = dir.path().join("execd.pid");
        let mut cmd = Command::new("sudo");
        cmd.args(["-n", "sh", "-c", "echo $$ > \"$0\"; exec \"$@\""])
            .arg(&pidfile)
            .arg("env");
        for (k, v) in env {
            cmd.arg(format!("{k}={v}"));
        }
        cmd.arg(BIN)
            .args(["--listen", ports::CHILD_LISTEN, "--root"])
            .arg(root.unwrap_or(dir.path()))
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(dir.path().join("execd.log")).unwrap());
        let mut child = cmd.spawn().unwrap();
        let log = dir.path().join("execd.log");
        let deadline = Instant::now() + Duration::from_secs(20);
        let addr = loop {
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            if let Some(a) = text
                .split("listen=")
                .nth(1)
                .and_then(|r| r.split_whitespace().next())
                .and_then(|a| a.parse().ok())
            {
                break a;
            }
            assert!(
                child.try_wait().unwrap().is_none(),
                "execd exited before listening:\n{text}"
            );
            assert!(Instant::now() < deadline, "execd never listened:\n{text}");
            std::thread::sleep(Duration::from_millis(20));
        };
        let root_pid = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        Some(Self {
            child,
            addr,
            dir,
            root_pid: Some(root_pid),
        })
    }

    pub fn url(&self) -> String {
        format!("http://{}/exec", self.addr)
    }

    pub fn pid(&self) -> u32 {
        self.root_pid.unwrap_or_else(|| self.child.id())
    }

    /// Everything execd has logged so far.
    pub fn log(&self) -> String {
        std::fs::read_to_string(self.dir.path().join("execd.log")).unwrap_or_default()
    }
}

fn nix_is_root() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .is_ok_and(|s| s.lines().any(|l| l.starts_with("Uid:\t0\t")))
}

impl Drop for Execd {
    fn drop(&mut self) {
        if let Some(pid) = self.root_pid {
            let _ = Command::new("sudo")
                .args(["-n", "kill", "-TERM", &pid.to_string()])
                .stderr(Stdio::null())
                .status();
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            let _ = Command::new("sudo")
                .args(["-n", "kill", "-KILL", &pid.to_string()])
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
