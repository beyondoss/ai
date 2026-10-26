//! `--run-as`: every command runs as another uid and gid than execd's own.
//!
//! Without it a command is a child of execd with execd's uid, so it can read execd's memory and
//! `/proc/<execd>/environ`, and any file execd owns (a 0600 bearer file included). With it execd
//! keeps its uid (root, or a user holding `CAP_SETUID`, `CAP_SETGID` and `CAP_KILL`) and a command
//! runs as the target, a different uid, so the kernel refuses all of that.
//!
//! **How.** The workspace forbids `unsafe`, which rules out `pre_exec`, and `std`'s supplementary
//! group setter is unstable. So the switch is done by a short-lived helper that is execd's own
//! binary: a command is spawned as `/proc/self/exe <flag> run <uid> <gid> <groups> <cwd> <command>
//! <args>...`, which calls `setgroups`, `setresgid` and `setresuid`, checks it cannot get back,
//! changes to the request's directory *as the target* (so the directory is checked against the
//! target's permissions, not execd's) and `exec`s the command. The helper is the command's own
//! process: the pid, the process group and the pipes are the ones execd spawned. If anything in the
//! switch fails the helper exits 126 without running the command. [`helper_main`] must be the first
//! thing the binary does; an embedding host that sets [`crate::Options::run_as`] must do the same.
//!
//! **Bounds.** Nothing is held: the helper is the command's process, one per command, and the
//! startup check spawns one short-lived helper.

use std::ffi::OsString;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};

use nix::unistd::{Gid, Group, Uid, User};

/// The argument that makes `execd` act as the switch helper instead of a server. It is the first
/// argument, ahead of any argument parsing.
pub const HELPER_FLAG: &str = "--execd-run-as-helper";

/// The identity commands run as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunAs {
    pub uid: u32,
    pub gid: u32,
    /// The supplementary groups (the primary gid included).
    pub groups: Vec<u32>,
}

/// What `--run-as` named, with the account's home directory when it has a passwd entry.
pub struct Resolved {
    pub run_as: RunAs,
    pub home: Option<PathBuf>,
}

impl RunAs {
    /// `user`, `uid`, `user:group`, `uid:gid`. A user with a passwd entry gets its primary group
    /// (unless one is named) and its supplementary groups from the group database; a bare numeric
    /// uid with no entry needs its gid named, and runs with that one group.
    pub fn resolve(spec: &str) -> Result<Resolved, String> {
        let (user, group) = match spec.split_once(':') {
            Some((u, g)) => (u, Some(g)),
            None => (spec, None),
        };
        if user.is_empty() || group.is_some_and(str::is_empty) {
            return Err(format!("--run-as {spec:?}: empty user or group"));
        }
        let numeric = user.parse::<u32>().ok();
        let entry = match numeric {
            Some(n) => User::from_uid(Uid::from_raw(n)),
            None => User::from_name(user),
        }
        .map_err(|e| format!("--run-as: looking up user {user:?}: {e}"))?;
        let uid = match (&entry, numeric) {
            (Some(u), _) => u.uid.as_raw(),
            (None, Some(n)) => n,
            (None, None) => return Err(format!("--run-as: no such user {user:?}")),
        };
        let gid = match group {
            Some(g) => match g.parse::<u32>() {
                Ok(n) => n,
                Err(_) => Group::from_name(g)
                    .map_err(|e| format!("--run-as: looking up group {g:?}: {e}"))?
                    .ok_or_else(|| format!("--run-as: no such group {g:?}"))?
                    .gid
                    .as_raw(),
            },
            None => match &entry {
                Some(u) => u.gid.as_raw(),
                None => {
                    return Err(format!(
                        "--run-as {spec:?}: uid {uid} has no passwd entry, so name its gid (uid:gid)"
                    ));
                }
            },
        };
        let mut groups = vec![gid];
        if let Some(u) = &entry
            && let Ok(name) = std::ffi::CString::new(u.name.as_str())
            && let Ok(list) = nix::unistd::getgrouplist(&name, Gid::from_raw(gid))
        {
            for g in list {
                if !groups.contains(&g.as_raw()) {
                    groups.push(g.as_raw());
                }
            }
        }
        Ok(Resolved {
            home: entry.map(|u| u.dir),
            run_as: RunAs { uid, gid, groups },
        })
    }

    /// Refuse a target that is execd's own identity, then prove execd can switch to the target and
    /// still signal what it started: a helper switches, and execd sends it signal 0 (the kernel's
    /// permission check for `SIGKILL`). The `Ok` carries a warning when `root` (execd's `--root`,
    /// where a command with no `cwd` runs) is not writable by the target.
    pub fn check(&self, root: &Path) -> Result<Option<String>, String> {
        let real = Uid::current().as_raw();
        let eff = Uid::effective().as_raw();
        if self.uid == real || self.uid == eff {
            return Err(format!(
                "--run-as uid {} is execd's own uid, which would protect nothing. Run execd as \
                 root (or with CAP_SETUID, CAP_SETGID and CAP_KILL) and name another user",
                self.uid
            ));
        }
        let mut child = std::process::Command::new("/proc/self/exe")
            .arg(HELPER_FLAG)
            .arg("check")
            .args(self.identity_args())
            .arg(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("--run-as: cannot start the switch helper: {e}"))?;
        // The helper answers one line once it has switched, then waits for stdin to close.
        let mut line = String::new();
        if let Some(out) = child.stdout.as_mut() {
            use std::io::BufRead as _;
            let _ = std::io::BufReader::new(out).read_line(&mut line);
        }
        let signalable = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(i32::try_from(child.id()).unwrap_or(0)),
            None,
        );
        drop(child.stdin.take());
        let out = child
            .wait_with_output()
            .map_err(|e| format!("--run-as: switch helper: {e}"))?;
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        if !matches!(line.trim(), "ok" | "root-not-writable") {
            return Err(format!(
                "--run-as: execd cannot switch to uid {} gid {}: {err}. Run execd as root (or \
                 with CAP_SETUID, CAP_SETGID and CAP_KILL)",
                self.uid, self.gid
            ));
        }
        if let Err(e) = signalable {
            return Err(format!(
                "--run-as: execd cannot signal processes of uid {} ({e}), so it could not kill a \
                 timed-out command or job. Run execd as root (or with CAP_KILL)",
                self.uid
            ));
        }
        Ok((line.trim() == "root-not-writable").then(|| {
            format!(
                "--root {} is not writable by uid {} gid {}: commands that write there will fail \
                 (chown it, or name another --root)",
                root.display(),
                self.uid,
                self.gid
            )
        }))
    }

    fn identity_args(&self) -> [OsString; 3] {
        let groups: Vec<String> = self.groups.iter().map(u32::to_string).collect();
        [
            self.uid.to_string().into(),
            self.gid.to_string().into(),
            groups.join(",").into(),
        ]
    }

    /// The command that runs `command` as this identity, from `cwd`: execd's own binary as the
    /// helper.
    pub(crate) fn wrap(
        &self,
        command: &str,
        args: &[String],
        cwd: &Path,
    ) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new("/proc/self/exe");
        cmd.arg(HELPER_FLAG)
            .arg("run")
            .args(self.identity_args())
            .arg(cwd)
            .arg(command)
            .args(args);
        cmd
    }
}

/// If this process was started as the switch helper, do its job and say how to exit; `None` for a
/// normal start. A successful `run` never returns: the process becomes the command.
pub fn helper_main() -> Option<ExitCode> {
    let mut args = std::env::args_os().skip(1);
    if args.next()? != HELPER_FLAG {
        return None;
    }
    Some(helper(&args.collect::<Vec<_>>()))
}

fn helper(args: &[OsString]) -> ExitCode {
    let fail = |why: String| {
        eprintln!("execd: {why}");
        ExitCode::from(126)
    };
    let [mode, uid, gid, groups, rest @ ..] = args else {
        return fail("bad helper arguments".into());
    };
    let parse = |s: &OsString| s.to_str().and_then(|s| s.parse::<u32>().ok());
    let (Some(uid), Some(gid)) = (parse(uid), parse(gid)) else {
        return fail("bad uid or gid".into());
    };
    let groups: Vec<Gid> = groups
        .to_str()
        .unwrap_or("")
        .split(',')
        .filter_map(|g| g.parse::<u32>().ok())
        .map(Gid::from_raw)
        .collect();
    if let Err(e) = switch(uid, gid, &groups) {
        return fail(format!("cannot switch to uid {uid} gid {gid}: {e}"));
    }
    match (mode.to_str(), rest) {
        (Some("check"), [root]) => {
            let writable = nix::unistd::access(
                Path::new(root),
                nix::unistd::AccessFlags::W_OK | nix::unistd::AccessFlags::X_OK,
            )
            .is_ok();
            println!("{}", if writable { "ok" } else { "root-not-writable" });
            // Hold still until execd has signalled us and closes our stdin.
            let _ = std::io::Read::read_to_end(&mut std::io::stdin(), &mut Vec::new());
            ExitCode::SUCCESS
        }
        (Some("run"), [cwd, command, argv @ ..]) => {
            if let Err(e) = nix::unistd::chdir(Path::new(cwd)) {
                eprint!("{cwd:?}: {}", std::io::Error::from(e));
                return ExitCode::from(127);
            }
            let err = std::process::Command::new(command).args(argv).exec();
            eprint!("{err}");
            ExitCode::from(127)
        }
        _ => fail("bad helper mode".into()),
    }
}

/// Become `uid`/`gid` with `groups`, all three ids of each, and prove it cannot be undone.
fn switch(uid: u32, gid: u32, groups: &[Gid]) -> Result<(), String> {
    let (u, g) = (Uid::from_raw(uid), Gid::from_raw(gid));
    nix::unistd::setgroups(groups).map_err(|e| format!("setgroups: {e}"))?;
    nix::unistd::setresgid(g, g, g).map_err(|e| format!("setresgid: {e}"))?;
    nix::unistd::setresuid(u, u, u).map_err(|e| format!("setresuid: {e}"))?;
    let ids = nix::unistd::getresuid().map_err(|e| format!("getresuid: {e}"))?;
    if ids.real != u || ids.effective != u || ids.saved != u {
        return Err("the uid did not change".into());
    }
    let root = Uid::from_raw(0);
    if uid != 0 && nix::unistd::setresuid(root, root, root).is_ok() {
        return Err("the switch can be undone".into());
    }
    Ok(())
}
