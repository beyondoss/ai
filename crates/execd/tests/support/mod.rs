//! Running the real `execd` binary.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub mod baseline;
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
    /// Kills execd and everything under it if the test process dies without running `Drop`.
    _watchdog: Watchdog,
}

/// What survives a test process that dies by `SIGKILL` (or any way that skips destructors): nothing.
/// A detached shell watches the owning test process; when it is gone it kills the target and every
/// descendant (leaves first, so nothing is orphaned before it is listed), as root through `sudo`
/// when the target is root's. `PR_SET_PDEATHSIG` would not do: it cannot be set without `unsafe`,
/// it does not reach what execd's own children left behind, and `sudo` clears it. Dropping the
/// watchdog (the normal end of a test) kills the shell.
pub struct Watchdog(Child);

const WATCH: &str = r#"
owner=$0; target=$1; as_root=$2
while kill -0 "$owner" 2>/dev/null; do sleep 0.2; done
S=; if [ "$as_root" = 1 ]; then S="sudo -n"; target=$(cat "$target" 2>/dev/null); fi
[ -n "$target" ] || exit 0
kt() { for c in $(pgrep -P "$1"); do kt "$c"; done; $S kill -9 "$1" 2>/dev/null; }
kt "$target"
"#;

impl Watchdog {
    /// Watch the current process on behalf of `target`: a pid, or with `as_root` the path of a file
    /// holding one (the pid is only known once the server has written it).
    pub fn new(target: &str, as_root: bool) -> Self {
        let child = Command::new("sh")
            .args(["-c", WATCH, &std::process::id().to_string(), target])
            .arg(if as_root { "1" } else { "0" })
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        Self(child)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Execd {
    /// Start `execd` on a loopback port it picks, rooted in a fresh directory. `env` is added to its
    /// environment, and `settings` (`EXECD_*` variables) too: execd takes no arguments.
    pub fn start(settings: &[(&str, &str)], env: &[(&str, &str)]) -> Self {
        Self::start_on(ports::CHILD_LISTEN, settings, env)
    }

    /// [`Self::start`], listening on `listen` (a port-0 address, read back from the socket).
    pub fn start_on(listen: &str, settings: &[(&str, &str)], env: &[(&str, &str)]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut cmd = Command::new(BIN);
        cmd.env("EXECD_LISTEN", listen)
            .env("EXECD_ROOT", dir.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(dir.path().join("execd.log")).unwrap())
            .env_remove("EXECD_BEARER")
            .env_remove("EXECD_BEARER_FILE");
        for (k, v) in settings.iter().chain(env) {
            cmd.env(k, v);
        }
        let child = cmd.spawn().unwrap();
        // Owned from the moment it exists, so a panic while waiting for it still stops it.
        let mut me = Self {
            _watchdog: Watchdog::new(&child.id().to_string(), false),
            child,
            addr: std::net::SocketAddrV4::new(std::net::Ipv4Addr::UNSPECIFIED, 1),
            dir,
            root_pid: None,
        };
        let deadline = Instant::now() + Duration::from_secs(20);
        me.addr = loop {
            if let Some(a) = ports::listening_on(me.child.id()).first() {
                break *a;
            }
            assert!(Instant::now() < deadline, "execd never listened");
            std::thread::sleep(Duration::from_millis(20));
        };
        me
    }

    /// Run `execd` with these settings (no `EXECD_LISTEN` or `EXECD_ROOT` added) until it exits on its own,
    /// for a start that must be refused: its exit status and everything it logged. Killed after
    /// 20 s, which a test reads as "it did not refuse".
    pub fn run_to_exit(
        settings: &[(&str, &str)],
        env: &[(&str, &str)],
        as_root: bool,
    ) -> (std::process::ExitStatus, String) {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("execd.log");
        let mut cmd = if as_root {
            let mut c = Command::new("sudo");
            c.arg("-n");
            preserve(&mut c, &[settings, env].concat());
            c.arg(BIN);
            c
        } else {
            Command::new(BIN)
        };
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .env_remove("EXECD_BEARER")
            .env_remove("EXECD_BEARER_FILE");
        for (k, v) in settings.iter().chain(env) {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().unwrap();
        let _watchdog = Watchdog::new(&child.id().to_string(), false);
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
    /// write (the `EXECD_RUN_AS` user is not the test's). `None`, after saying so, when this host gives
    /// the tests no way to be root.
    pub fn start_as_root(settings: &[(&str, &str)], env: &[(&str, &str)]) -> Option<Self> {
        Self::start_as_root_in(None, settings, env)
    }

    /// [`Self::start_as_root`], with `EXECD_ROOT` set to `root` instead of the fresh directory.
    pub fn start_as_root_in(
        root: Option<&std::path::Path>,
        settings: &[(&str, &str)],
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
        let root_dir = root.unwrap_or(dir.path());
        let settings = [
            &[
                ("EXECD_LISTEN", ports::CHILD_LISTEN),
                ("EXECD_ROOT", root_dir.to_str().unwrap()),
            ][..],
            settings,
        ]
        .concat();
        let mut cmd = Command::new("sudo");
        cmd.arg("-n");
        preserve(&mut cmd, &[&settings[..], env].concat());
        cmd.args(["sh", "-c", "echo $$ > \"$0\"; exec \"$@\""])
            .arg(&pidfile)
            .arg(BIN)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(dir.path().join("execd.log")).unwrap());
        for (k, v) in settings.iter().chain(env) {
            cmd.env(k, v);
        }
        let child = cmd.spawn().unwrap();
        let log = dir.path().join("execd.log");
        let mut me = Self {
            _watchdog: Watchdog::new(pidfile.to_str().unwrap(), true),
            child,
            addr: std::net::SocketAddrV4::new(std::net::Ipv4Addr::UNSPECIFIED, 1),
            dir,
            root_pid: None,
        };
        let deadline = Instant::now() + Duration::from_secs(20);
        me.addr = loop {
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
                me.child.try_wait().unwrap().is_none(),
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
        me.root_pid = Some(root_pid);
        Some(me)
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

/// Pass `env` through `sudo` in its environment, not on a command line: `sudo` puts the command it
/// runs in `SUDO_COMMAND`, which every process below it inherits, so a token given as an argument
/// would sit in the environment of every command the server runs.
fn preserve(sudo: &mut Command, env: &[(&str, &str)]) {
    if !env.is_empty() {
        let names: Vec<&str> = env.iter().map(|(k, _)| *k).collect();
        sudo.arg(format!("--preserve-env={}", names.join(",")));
    }
}

fn nix_is_root() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .is_ok_and(|s| s.lines().any(|l| l.starts_with("Uid:\t0\t")))
}

impl Drop for Execd {
    fn drop(&mut self) {
        let root_pid = self.root_pid.or_else(|| {
            std::fs::read_to_string(self.dir.path().join("execd.pid"))
                .ok()
                .and_then(|t| t.trim().parse().ok())
        });
        let signal = |sig: &str| {
            if let Some(pid) = root_pid {
                let _ = Command::new("sudo")
                    .args(["-n", "kill", sig, &pid.to_string()])
                    .stderr(Stdio::null())
                    .status();
            }
        };
        if root_pid.is_some() {
            signal("-TERM");
        } else if let Ok(pid) = i32::try_from(self.child.id()) {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid),
                nix::sys::signal::Signal::SIGTERM,
            );
        }
        // execd's own SIGTERM sweep kills its commands and jobs; give it the time to.
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        signal("-KILL");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
