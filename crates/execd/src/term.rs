//! Terminals: a job whose stdin, stdout and stderr are one pseudo-terminal, kept by a **keeper**.
//!
//! A terminal needs its master held open, or the kernel hangs the session up. If execd held it, every
//! restart of execd would hang up every terminal, which a kept job survives (S3). So the master is
//! held by one small process per terminal, execd's own binary re-exec'd in keeper mode
//! (`--execd-run-as-helper keep ...`). The keeper:
//!
//! - opens the pty (at the size asked), spawns the session helper (`... session ...`: `setsid`, the
//!   slave as the controlling terminal, then `exec` of the command, which is a session leader) and
//!   waits for it to be ready;
//! - copies the master into the job's `out` spool file (its own stdout), so the same ring, offsets
//!   and restart resume as every job's apply;
//! - serves execd on `term.sock` in the job's spool directory (`0700`; a peer must be execd's uid,
//!   `SO_PEERCRED`): positional writes, sizes, the shell's exit;
//! - reaps the shell, drains the master, writes `status` and tells execd, then ends when the master
//!   reads `EIO` (no slave is open anywhere).
//!
//! Every decision is [`execd_jobs::keeper::step`]'s: the loop here performs what it returns. It is
//! single-threaded, blocks in `poll` with no timeout (an idle terminal costs no wakeups), and holds a
//! 64 KiB read buffer and one write in flight.
//!
//! The frames on the socket: `tag u8, len u32 le, payload`. From the keeper: `Ready` (version,
//! pgid, written, cols, rows), `Written` (the count that landed), `Exit` (code, signal). To it:
//! `Write` (position, bytes), `Resize` (cols, rows).

use std::ffi::OsString;
use std::io::{self, Read as _, Write as _};
use std::os::fd::{AsFd as _, AsRawFd as _, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};

use execd_jobs::keeper::{K, KAct, KEv, Shell, exit_status, start, step};
use execd_jobs::{
    Expected, KEEPER_VERSION, Keeper, Peer, PeerCred, Reply, Size, admit_peer, keeper_of,
};

/// The helper modes this module serves (the `HELPER_FLAG`'s first argument).
pub(crate) const KEEP: &str = "keep";
pub(crate) const SESSION: &str = "session";
/// The environment of the command, which the keeper hands to the session (hex JSON, as `EXECD_CMD_ENV`).
pub(crate) const TERM_ENV: &str = "EXECD_TERM_ENV";
pub(crate) const SOCK: &str = "term.sock";
pub(crate) const STATUS: &str = "status";
/// The largest frame the keeper takes: a write of at most 128 KiB plus its position.
const FRAME_MAX: usize = 128 * 1024 + 64;

const READY: u8 = 1;
const WRITTEN: u8 = 2;
const EXIT: u8 = 3;
const WRITE: u8 = 16;
const RESIZE: u8 = 17;

/// A frame, encoded.
pub(crate) fn frame(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(5 + payload.len());
    v.push(tag);
    v.extend_from_slice(
        &u32::try_from(payload.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    v.extend_from_slice(payload);
    v
}

pub(crate) fn write_frame(at: u64, data: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(8 + data.len());
    p.extend_from_slice(&at.to_le_bytes());
    p.extend_from_slice(data);
    frame(WRITE, &p)
}

pub(crate) fn resize_frame(size: Size) -> Vec<u8> {
    let mut p = Vec::with_capacity(4);
    p.extend_from_slice(&size.cols.to_le_bytes());
    p.extend_from_slice(&size.rows.to_le_bytes());
    frame(RESIZE, &p)
}

/// A frame buffer: bytes in, whole frames out.
#[derive(Default)]
pub(crate) struct Frames {
    buf: Vec<u8>,
}

impl Frames {
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// The next whole frame, `Ok(None)` if more bytes are needed, `Err` for a frame too large.
    pub fn next(&mut self) -> io::Result<Option<(u8, Vec<u8>)>> {
        if self.buf.len() < 5 {
            return Ok(None);
        }
        let len = u32::from_le_bytes([self.buf[1], self.buf[2], self.buf[3], self.buf[4]]) as usize;
        if len > FRAME_MAX {
            return Err(io::Error::other("a frame is too large"));
        }
        if self.buf.len() < 5 + len {
            return Ok(None);
        }
        let tag = self.buf[0];
        let payload = self.buf[5..5 + len].to_vec();
        self.buf.drain(..5 + len);
        Ok(Some((tag, payload)))
    }
}

/// What the keeper says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Said {
    Ready {
        version: u8,
        pgid: i32,
        written: u64,
        size: Size,
    },
    Written(u64),
    Exit {
        code: i32,
        signal: Option<i32>,
    },
}

pub(crate) fn parse_said(tag: u8, p: &[u8]) -> Option<Said> {
    let u64_at = |i: usize| Some(u64::from_le_bytes(p.get(i..i + 8)?.try_into().ok()?));
    let i32_at = |i: usize| Some(i32::from_le_bytes(p.get(i..i + 4)?.try_into().ok()?));
    let u16_at = |i: usize| Some(u16::from_le_bytes(p.get(i..i + 2)?.try_into().ok()?));
    match tag {
        READY if p.len() == 17 => Some(Said::Ready {
            version: p[0],
            pgid: i32_at(1)?,
            written: u64_at(5)?,
            size: Size {
                cols: u16_at(13)?,
                rows: u16_at(15)?,
            },
        }),
        WRITTEN if p.len() == 8 => Some(Said::Written(u64_at(0)?)),
        EXIT if p.len() == 8 => Some(Said::Exit {
            code: i32_at(0)?,
            signal: Some(i32_at(4)?).filter(|&s| s > 0),
        }),
        _ => None,
    }
}

fn ready_frame(pgid: i32, k: &K) -> Vec<u8> {
    let mut p = vec![KEEPER_VERSION];
    p.extend_from_slice(&pgid.to_le_bytes());
    p.extend_from_slice(&k.written.to_le_bytes());
    p.extend_from_slice(&k.size.cols.to_le_bytes());
    p.extend_from_slice(&k.size.rows.to_le_bytes());
    frame(READY, &p)
}

fn exit_frame(code: i32, signal: Option<i32>) -> Vec<u8> {
    let mut p = Vec::with_capacity(8);
    p.extend_from_slice(&code.to_le_bytes());
    p.extend_from_slice(&signal.unwrap_or(0).to_le_bytes());
    frame(EXIT, &p)
}

/// The shell's status a keeper wrote: `(code, signal)`.
pub(crate) fn read_status(dir: &Path) -> Option<(i32, Option<i32>)> {
    let text = std::fs::read_to_string(dir.join(STATUS)).ok()?;
    let mut it = text.split_whitespace();
    let code = it.next()?.parse().ok()?;
    let sig: i32 = it.next()?.parse().ok()?;
    Some((code, Some(sig).filter(|&s| s > 0)))
}

fn write_status(code: i32, signal: Option<i32>) {
    let body = format!("{code} {}\n", signal.unwrap_or(0));
    if std::fs::write("status.tmp", body).is_ok() {
        let _ = std::fs::rename("status.tmp", STATUS);
    }
}

// ---------------------------------------------------------------------------------------------
// The keeper process.
// ---------------------------------------------------------------------------------------------

/// `keep <spool dir> <cols> <rows> <chown uid|-> <chown gid|-> <cwd|-> <program> <args...>`. Never
/// returns on success until the keeper's end.
pub(crate) fn keeper_main(args: &[OsString]) -> ExitCode {
    match keeper(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("execd keeper: {e}");
            ExitCode::from(
                u8::try_from(if no_pty(&e) {
                    execd_jobs::NO_PTY_EXIT
                } else {
                    125
                })
                .unwrap_or(u8::MAX),
            )
        }
    }
}

/// What a keeper that found no pseudo-terminal to open says (its exit status is 124, which the start
/// answers 503).
const NO_PTY: &str = "no pseudo-terminal is free";

fn no_pty(e: &io::Error) -> bool {
    e.to_string().starts_with(NO_PTY)
}

fn arg(args: &[OsString], i: usize) -> io::Result<&str> {
    args.get(i)
        .and_then(|a| a.to_str())
        .ok_or_else(|| io::Error::other("bad keeper arguments"))
}

fn keeper(args: &[OsString]) -> io::Result<()> {
    use rustix::event::{PollFd, PollFlags, poll};
    use rustix::io::Errno;
    let dir = arg(args, 0)?;
    let cols: u16 = arg(args, 1)?.parse().map_err(io::Error::other)?;
    let rows: u16 = arg(args, 2)?.parse().map_err(io::Error::other)?;
    std::env::set_current_dir(dir)?;
    // The socket and the status are for execd's uid alone.
    rustix::process::umask(rustix::fs::Mode::from_raw_mode(0o077));
    let master = rustix::pty::openpt(
        rustix::pty::OpenptFlags::RDWR
            | rustix::pty::OpenptFlags::NOCTTY
            | rustix::pty::OpenptFlags::CLOEXEC,
    )
    .map_err(|e| io::Error::other(format!("{NO_PTY}: {e}")))?;
    rustix::pty::grantpt(&master)?;
    rustix::pty::unlockpt(&master)?;
    let slave = rustix::pty::ptsname(&master, Vec::new())?;
    set_size(&master, Size { cols, rows })?;
    let mut tio = rustix::termios::tcgetattr(&master)?;
    tio.input_modes |= rustix::termios::InputModes::IUTF8;
    rustix::termios::tcsetattr(&master, rustix::termios::OptionalActions::Now, &tio)?;
    rustix::fs::fcntl_setfl(&master, rustix::fs::OFlags::NONBLOCK)?;

    // Bind before anything else is slow: a connect from execd waits in the backlog until `Ready`.
    let _ = std::fs::remove_file(SOCK);
    let listener = UnixListener::bind(SOCK)?;
    listener.set_nonblocking(true)?;

    let mut cmd = Command::new("/proc/self/exe");
    cmd.arg(crate::runas::HELPER_FLAG)
        .arg(SESSION)
        .arg(slave.to_str().map_err(io::Error::other)?)
        .arg(arg(args, 3)?)
        .arg(arg(args, 4)?)
        .arg(arg(args, 5)?)
        .args(&args[6..])
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(blob) = std::env::var_os(TERM_ENV) {
        cmd.env(TERM_ENV, blob);
    }
    let mut child = cmd.spawn()?;
    let pgid = i32::try_from(child.id()).map_err(io::Error::other)?;
    let mut ready_pipe: Option<OwnedFd> = child.stdout.take().map(OwnedFd::from);
    let pidfd = rustix::process::pidfd_open(
        rustix::process::Pid::from_raw(pgid).ok_or_else(|| io::Error::other("pid"))?,
        rustix::process::PidfdFlags::empty(),
    )?;

    // The process that started this keeper: if it ends before it ever connected, nobody will ask.
    let mut parent = rustix::process::Pid::from_raw(
        i32::try_from(std::os::unix::process::parent_id()).unwrap_or(0),
    )
    .and_then(|p| rustix::process::pidfd_open(p, rustix::process::PidfdFlags::empty()).ok());
    let mut k = start(Size { cols, rows });
    let mut conn: Option<UnixStream> = None;
    let mut frames = Frames::default();
    let mut ready = false;
    let mut want_ready = false;
    let mut pending: Option<(Vec<u8>, usize)> = None;
    let mut buf = vec![0u8; 64 * 1024];
    let me = rustix::process::getuid();
    let out = rustix::stdio::stdout();

    // What one event does: the kernel decides, this performs.
    macro_rules! event {
        ($ev:expr, $data:expr) => {{
            let ev: KEv = $ev;
            let act: KAct = step(&mut k, ev);
            if let Some(sz) = act.set_size {
                let _ = set_size(&master, sz);
            }
            if let Some((skip, take)) = act.apply {
                let data: &[u8] = $data;
                let (s, t) = (skip as usize, take as usize);
                if let Some(slice) = data.get(s..s + t) {
                    pending = Some((slice.to_vec(), 0));
                }
            }
            if act.kill_shell {
                eprintln!("execd keeper: the output cannot be written; ending the terminal");
                let _ = rustix::process::kill_process_group(
                    rustix::process::Pid::from_raw(pgid).unwrap_or(rustix::process::Pid::INIT),
                    rustix::process::Signal::KILL,
                );
            }
            if act.write_status
                && let Some((code, signal)) = exit_status(&k)
            {
                write_status(code, signal);
            }
            if act.send_ready {
                want_ready = true;
            }
            if want_ready && ready {
                want_ready = false;
                send(&mut conn, &ready_frame(pgid, &k));
            }
            if act.send_exit
                && let Some((code, signal)) = exit_status(&k)
            {
                send(&mut conn, &exit_frame(code, signal));
            }
            act.exit
        }};
    }
    fn send(conn: &mut Option<UnixStream>, bytes: &[u8]) {
        let Some(c) = conn.as_mut() else { return };
        // A frame is a few bytes: a socket whose buffer is full means execd is not reading, so drop it.
        let ok = {
            let _ = c.set_nonblocking(false);
            let _ = c.set_write_timeout(Some(std::time::Duration::from_secs(1)));
            c.write_all(bytes).is_ok()
        };
        let _ = c.set_nonblocking(true);
        if !ok {
            *conn = None;
        }
    }

    loop {
        // What to wait for. A fd with nothing to say costs nothing: there is no timeout. The master
        // is read once the session holds the slave (before that it may report a hangup), and not once
        // it read EIO.
        let mut fds = vec![PollFd::from_borrowed_fd(listener.as_fd(), PollFlags::IN)];
        let master_at = (ready && !k.eio).then(|| {
            fds.push(PollFd::new(
                &master,
                PollFlags::IN
                    | if pending.is_some() {
                        PollFlags::OUT
                    } else {
                        PollFlags::empty()
                    },
            ));
            fds.len() - 1
        });
        let pid_at = matches!(k.shell, Shell::Running).then(|| {
            fds.push(PollFd::new(&pidfd, PollFlags::IN));
            fds.len() - 1
        });
        let parent_at = parent.as_ref().map(|p| {
            fds.push(PollFd::new(p, PollFlags::IN));
            fds.len() - 1
        });
        let pipe_at = ready_pipe.as_ref().map(|p| {
            fds.push(PollFd::new(p, PollFlags::IN));
            fds.len() - 1
        });
        let conn_at = conn.as_ref().map(|c| {
            fds.push(PollFd::new(c, PollFlags::IN));
            fds.len() - 1
        });
        match poll(&mut fds, None) {
            Ok(_) | Err(Errno::INTR) => {}
            Err(e) => return Err(e.into()),
        }
        let rev = |at: Option<usize>| at.map_or(PollFlags::empty(), |i| fds[i].revents());
        let (r_listen, r_master, r_pid, r_pipe, r_conn, r_parent) = (
            fds[0].revents(),
            rev(master_at),
            rev(pid_at),
            rev(pipe_at),
            rev(conn_at),
            rev(parent_at),
        );
        drop(fds);

        if !r_parent.is_empty() {
            parent = None;
            if event!(KEv::ParentGone, &[]) {
                return finish();
            }
        }
        // The session is ready: it is a session leader with the slave as its terminal.
        if !r_pipe.is_empty() {
            let mut b = [0u8; 1];
            // Nothing is ever written to it: EOF is the signal (the helper's `dup2` closed its end).
            let n = ready_pipe
                .as_ref()
                .map_or(0, |p| rustix::io::read(p, &mut b).unwrap_or(0));
            if n == 0 {
                ready_pipe = None;
                ready = true;
                if want_ready {
                    want_ready = false;
                    send(&mut conn, &ready_frame(pgid, &k));
                }
            }
        }
        // The shell was reaped.
        if !r_pid.is_empty()
            && let Shell::Running = k.shell
            && let Ok(Some(status)) = child.try_wait()
        {
            use std::os::unix::process::ExitStatusExt as _;
            let (code, signal) = match status.code() {
                Some(c) => (c, None),
                None => {
                    let sig = status.signal().unwrap_or(0);
                    (128 + sig, Some(sig).filter(|&n| n > 0))
                }
            };
            if event!(KEv::ChildExit { code, signal }, &[]) {
                break;
            }
        }
        // The terminal printed something: into the spool, in order.
        if ready && (!r_master.is_empty() || matches!(k.shell, Shell::Reaped { .. }) && !k.drained)
        {
            if r_master.contains(PollFlags::OUT)
                && let Some((data, at)) = pending.as_mut()
            {
                match rustix::io::write(&master, &data[*at..]) {
                    Ok(n) if n > 0 => {
                        *at += n;
                        let done = *at >= data.len();
                        if done {
                            pending = None;
                        }
                        if event!(KEv::Landed { n: n as u64 }, &[]) {
                            break;
                        }
                        send(&mut conn, &frame(WRITTEN, &k.written.to_le_bytes()));
                    }
                    Ok(_) | Err(Errno::AGAIN | Errno::INTR) => {}
                    Err(_) => pending = None,
                }
            }
            // Drain what is there; EAGAIN and EIO are events.
            loop {
                match rustix::io::read(&master, &mut buf) {
                    Ok(0) => {
                        if event!(KEv::MasterEio, &[]) {
                            return finish();
                        }
                        break;
                    }
                    Ok(n) => {
                        let mut off = 0;
                        while off < n && !k.failed {
                            match rustix::io::write(out, &buf[off..n]) {
                                Ok(w) => off += w,
                                Err(Errno::INTR) => {}
                                Err(_) => {
                                    // Fail closed (M6): the kernel ends the terminal; what is left to
                                    // read is dropped, not kept past a gap.
                                    if event!(KEv::SpoolFailed, &[]) {
                                        return finish();
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    Err(Errno::AGAIN) => {
                        if event!(KEv::MasterEagain, &[]) {
                            return finish();
                        }
                        break;
                    }
                    Err(Errno::INTR) => {}
                    Err(_) => {
                        if event!(KEv::MasterEio, &[]) {
                            return finish();
                        }
                        break;
                    }
                }
            }
        }
        // A new execd (the first, or a restarted one) connects.
        if !r_listen.is_empty() {
            while let Ok((s, _)) = listener.accept() {
                // Execd's uid, any of its processes: the keeper cannot know which one restarted.
                if admit_peer(
                    peer_cred(&s),
                    Expected {
                        uid: me.as_raw(),
                        pid: None,
                    },
                ) == Peer::Refuse
                {
                    continue;
                }
                if conn.take().is_some() && event!(KEv::ConnLost, &[]) {
                    return finish();
                }
                let _ = s.set_nonblocking(true);
                conn = Some(s);
                frames = Frames::default();
                if event!(KEv::Conn, &[]) {
                    return finish();
                }
            }
        }
        // execd says something.
        if !r_conn.is_empty() {
            let mut lost = false;
            if let Some(c) = conn.as_mut() {
                loop {
                    match c.read(&mut buf) {
                        Ok(0) => {
                            lost = true;
                            break;
                        }
                        Ok(n) => frames.push(&buf[..n]),
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                        Err(_) => {
                            lost = true;
                            break;
                        }
                    }
                }
            }
            loop {
                match frames.next() {
                    Ok(Some((WRITE, p))) if p.len() >= 8 => {
                        let at = u64::from_le_bytes(p[..8].try_into().unwrap_or([0; 8]));
                        let data = &p[8..];
                        if event!(
                            KEv::Write {
                                at,
                                len: data.len() as u64
                            },
                            data
                        ) {
                            return finish();
                        }
                        send(&mut conn, &frame(WRITTEN, &k.written.to_le_bytes()));
                    }
                    Ok(Some((RESIZE, p))) if p.len() == 4 => {
                        let size = Size {
                            cols: u16::from_le_bytes([p[0], p[1]]),
                            rows: u16::from_le_bytes([p[2], p[3]]),
                        };
                        if event!(KEv::Resize { size }, &[]) {
                            return finish();
                        }
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        lost = true;
                        break;
                    }
                }
            }
            if lost {
                conn = None;
                if event!(KEv::ConnLost, &[]) {
                    return finish();
                }
            }
        }
    }
    finish()
}

/// The keeper ends: its socket goes, the status stays.
fn finish() -> io::Result<()> {
    let _ = std::fs::remove_file(SOCK);
    Ok(())
}

fn set_size(master: &OwnedFd, size: Size) -> io::Result<()> {
    rustix::termios::tcsetwinsize(
        master,
        rustix::termios::Winsize {
            ws_row: size.rows,
            ws_col: size.cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The session helper: the command's own process, a session leader with the slave as its terminal.
// ---------------------------------------------------------------------------------------------

/// `session <slave> <chown uid|-> <chown gid|-> <cwd|-> <program> <args...>`: `setsid`, the slave as
/// the controlling terminal (before any user switch), the slave on 0, 1 and 2 (fd 1 last: the
/// keeper reads EOF on its pipe there, which says the group exists), then the command. Never returns
/// on success.
pub(crate) fn session_main(args: &[OsString]) -> ExitCode {
    let fail = |why: String| {
        eprintln!("execd: {why}");
        ExitCode::from(126)
    };
    let (Ok(slave), Ok(uid), Ok(gid), Ok(cwd)) =
        (arg(args, 0), arg(args, 1), arg(args, 2), arg(args, 3))
    else {
        return fail("bad session arguments".into());
    };
    let (Some(program), argv) = (args.get(4), args.get(5..).unwrap_or(&[])) else {
        return fail("bad session arguments".into());
    };
    let Some(env) =
        std::env::var_os(TERM_ENV).and_then(|b| b.to_str().and_then(crate::runas::decode_env))
    else {
        return fail("bad environment".into());
    };
    if rustix::process::setsid().is_err() {
        return fail("setsid failed".into());
    }
    let tty = match rustix::fs::open(
        slave,
        rustix::fs::OFlags::RDWR | rustix::fs::OFlags::NOCTTY,
        rustix::fs::Mode::empty(),
    ) {
        Ok(t) => t,
        Err(e) => return fail(format!("cannot open the terminal: {e}")),
    };
    if let Err(e) = rustix::process::ioctl_tiocsctty(&tty) {
        return fail(format!("cannot take the terminal: {e}"));
    }
    // What login does: the user owns the terminal they run on.
    if let (Ok(u), Ok(g)) = (uid.parse::<u32>(), gid.parse::<u32>()) {
        let _ = rustix::fs::fchown(
            &tty,
            Some(rustix::fs::Uid::from_raw(u)),
            Some(rustix::fs::Gid::from_raw(g)),
        );
        let _ = rustix::fs::fchmod(&tty, rustix::fs::Mode::from_raw_mode(0o620));
    }
    // Errors from here on are printed on the terminal; stdout goes last (the keeper reads EOF on its pipe there).
    if rustix::stdio::dup2_stderr(&tty).is_err() || rustix::stdio::dup2_stdin(&tty).is_err() {
        return fail("cannot use the terminal".into());
    }
    if rustix::stdio::dup2_stdout(&tty).is_err() {
        return fail("cannot use the terminal".into());
    }
    drop(tty);
    if cwd != "-"
        && let Err(e) = std::env::set_current_dir(cwd)
    {
        eprintln!("execd: cwd {cwd}: {}", crate::exec::strip_os_code(&e));
        return ExitCode::from(127);
    }
    use std::os::unix::process::CommandExt as _;
    let err = Command::new(program)
        .args(argv)
        .env_clear()
        .envs(env)
        .exec();
    if err.kind() == io::ErrorKind::NotFound {
        eprint!("{}", crate::exec::not_found(&program.to_string_lossy()));
        ExitCode::from(127)
    } else {
        eprintln!(
            "execd: {}: {}",
            program.to_string_lossy(),
            crate::exec::strip_os_code(&err)
        );
        ExitCode::from(126)
    }
}

/// The helper's dispatch for the two modes of this module; `None` for any other.
pub(crate) fn helper(mode: &OsString, rest: &[OsString]) -> Option<ExitCode> {
    match mode.to_str() {
        Some(KEEP) => Some(keeper_main(rest)),
        Some(SESSION) => Some(session_main(rest)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// execd's side: launching a keeper, and talking to it.
// ---------------------------------------------------------------------------------------------

/// The command a keeper is to run as the terminal's session: the program, arguments, directory and
/// the whole environment, as the job's `command_for` built it.
pub(crate) fn keeper_command(
    exe: &Path,
    spool: &Path,
    size: Size,
    cmd: &Command,
    chown: Option<(u32, u32)>,
    env: &[(String, String)],
) -> Command {
    let mut k = Command::new(exe);
    k.arg(crate::runas::HELPER_FLAG)
        .arg(KEEP)
        .arg(spool)
        .arg(size.cols.to_string())
        .arg(size.rows.to_string());
    match chown {
        Some((u, g)) => k.arg(u.to_string()).arg(g.to_string()),
        None => k.arg("-").arg("-"),
    };
    match cmd.get_current_dir() {
        Some(d) => k.arg(d),
        None => k.arg("-"),
    };
    k.arg(cmd.get_program()).args(cmd.get_args());
    k.env_clear().env(TERM_ENV, crate::runas::encode_env(env));
    k
}

/// The environment `cmd` will have: execd's own with `cmd`'s changes applied.
pub(crate) fn full_env(cmd: &Command) -> Vec<(String, String)> {
    let mut env: std::collections::BTreeMap<OsString, OsString> = std::env::vars_os().collect();
    for (k, v) in cmd.get_envs() {
        match v {
            Some(v) => {
                env.insert(k.to_os_string(), v.to_os_string());
            }
            None => {
                env.remove(k);
            }
        }
    }
    env.into_iter()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
        .collect()
}

/// The device number of the terminal the session `pgid` leads, from its standard input (the slave).
/// and the `st_dev` of the devpts it lives on (a nested devpts reuses the same device numbers).
/// `(0, 0)` if it could not be read.
pub(crate) fn pts_ids(pgid: i32) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt as _;
    std::fs::metadata(format!("/proc/{pgid}/fd/0")).map_or((0, 0), |m| (m.rdev(), m.dev()))
}

/// A connection to a terminal's keeper, as execd holds it.
pub(crate) struct Link {
    pub keeper_pid: i32,
    /// The slave's device number (`st_rdev` of its `/dev/pts/N`), or 0 if it could not be read.
    pub pts_rdev: u64,
    /// The `st_dev` of the devpts the slave lives on.
    pub pts_fs: u64,
    writer: tokio::sync::Mutex<Option<tokio::net::unix::OwnedWriteHalf>>,
}

impl Link {
    pub fn new(
        keeper_pid: i32,
        w: tokio::net::unix::OwnedWriteHalf,
        (pts_rdev, pts_fs): (u64, u64),
    ) -> Self {
        Self {
            keeper_pid,
            pts_rdev,
            pts_fs,
            writer: tokio::sync::Mutex::new(Some(w)),
        }
    }

    /// Send a frame. `false`: the keeper is gone.
    pub async fn send(&self, bytes: &[u8]) -> bool {
        use tokio::io::AsyncWriteExt as _;
        let mut w = self.writer.lock().await;
        let Some(half) = w.as_mut() else { return false };
        if half.write_all(bytes).await.is_ok() {
            true
        } else {
            *w = None;
            false
        }
    }

    /// Close the connection: the terminal's job is over (or execd lets go of it).
    pub async fn close(&self) {
        self.writer.lock().await.take();
    }
}

/// A socket peer's credentials, as `SO_PEERCRED` read them (`None`: they could not be read).
fn peer_cred(s: &UnixStream) -> Option<PeerCred> {
    let c = rustix::net::sockopt::socket_peercred(s).ok()?;
    Some(PeerCred {
        uid: c.uid.as_raw(),
        pid: c.pid.as_raw_nonzero().get(),
    })
}

/// What connecting to a keeper found: whom [`execd_jobs::admit_peer`] said the peer was, and, if it
/// was admitted, its first frame (`Ready`).
pub(crate) struct Met {
    pub peer: Peer,
    pub ready: Option<(UnixStream, Said, Frames)>,
}

impl Met {
    /// What answered, as [`keeper_of`] reads it.
    pub fn reply(&self) -> Reply {
        Reply {
            peer: self.peer,
            ready: match &self.ready {
                Some((_, Said::Ready { version, size, .. }, _)) => Some((*version, *size)),
                _ => None,
            },
        }
    }

    /// The connection, if [`keeper_of`] takes the answer as the keeper's.
    pub fn answered(self) -> Option<(UnixStream, Said, Frames)> {
        match keeper_of(true, Some(self.reply()), None) {
            Keeper::Answered { .. } => self.ready,
            _ => None,
        }
    }
}

/// Connect to the keeper of the terminal whose spool is `dir`, and read its `Ready` if its peer is the
/// keeper this execd started or adopted ([`execd_jobs::admit_peer`]: its uid is ours and its pid is
/// `keeper_pid`; nothing is read from any other). `Err`: nothing answered, or what did spoke no `Ready`.
pub(crate) fn connect_sync(dir: &Path, keeper_pid: i32) -> io::Result<Met> {
    let d = std::fs::File::open(dir)?;
    let path = format!("/proc/self/fd/{}/{SOCK}", d.as_raw_fd());
    let s = UnixStream::connect(path)?;
    drop(d);
    connect_checked(
        s,
        Expected {
            uid: rustix::process::getuid().as_raw(),
            pid: Some(keeper_pid),
        },
    )
}

/// [`connect_sync`] over a connected socket and the peer it expects.
pub(crate) fn connect_checked(s: UnixStream, expected: Expected) -> io::Result<Met> {
    let peer = admit_peer(peer_cred(&s), expected);
    if peer == Peer::Refuse {
        return Ok(Met { peer, ready: None });
    }
    s.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    let mut frames = Frames::default();
    let mut s2 = s.try_clone()?;
    let mut b = [0u8; 64];
    loop {
        if let Some((tag, p)) = frames.next()? {
            return match parse_said(tag, &p) {
                Some(r @ Said::Ready { .. }) => {
                    s.set_read_timeout(None)?;
                    Ok(Met {
                        peer,
                        ready: Some((s, r, frames)),
                    })
                }
                _ => Err(io::Error::other("the keeper's first frame is not a Ready")),
            };
        }
        let n = s2.read(&mut b)?;
        if n == 0 {
            return Err(io::Error::other("the keeper closed before Ready"));
        }
        frames.push(&b[..n]);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn pair_with(frame: &[u8]) -> UnixStream {
        let (a, mut b) = UnixStream::pair().unwrap();
        b.write_all(frame).unwrap();
        // `b` stays open until the test ends.
        std::mem::forget(b);
        a
    }

    fn ready(version: u8) -> Vec<u8> {
        let mut f = ready_frame(7, &start(Size { cols: 80, rows: 24 }));
        f[5] = version;
        f
    }

    /// P1: a peer whose uid or pid is not the expected one is refused and nothing it says is read; the
    /// expected one is admitted and its `Ready` taken as the keeper's answer. A socket pair's peer is
    /// this very process, so the credentials are ours: a different uid or pid is an impostor.
    #[test]
    fn a_peer_of_another_uid_or_pid_is_refused_and_the_expected_one_is_admitted() {
        let (uid, pid) = (
            rustix::process::getuid().as_raw(),
            i32::try_from(std::process::id()).unwrap(),
        );
        let ok = Expected {
            uid,
            pid: Some(pid),
        };
        let met = connect_checked(pair_with(&ready(KEEPER_VERSION)), ok).unwrap();
        assert_eq!(met.peer, Peer::Admit);
        assert!(met.answered().is_some(), "the expected peer was not taken");
        // Any pid of the uid (the keeper's side).
        let any = Expected { uid, pid: None };
        let met = connect_checked(pair_with(&ready(KEEPER_VERSION)), any).unwrap();
        assert_eq!(met.peer, Peer::Admit);
        for wrong in [
            Expected {
                uid: uid.wrapping_add(1),
                pid: Some(pid),
            },
            Expected {
                uid: uid.wrapping_add(1),
                pid: None,
            },
            Expected {
                uid,
                pid: Some(pid + 1),
            },
        ] {
            let met = connect_checked(pair_with(&ready(KEEPER_VERSION)), wrong).unwrap();
            assert_eq!(met.peer, Peer::Refuse, "{wrong:?}");
            assert!(met.ready.is_none(), "a refused peer was read: {wrong:?}");
            assert!(met.answered().is_none());
        }
    }

    /// T6a: a peer that is admitted but speaks a version this execd does not know is not the keeper's
    /// answer: the terminal is hung up, not adopted.
    #[test]
    fn an_unknown_version_is_not_an_answer() {
        let uid = rustix::process::getuid().as_raw();
        let pid = i32::try_from(std::process::id()).unwrap();
        let met = connect_checked(
            pair_with(&ready(KEEPER_VERSION + 1)),
            Expected {
                uid,
                pid: Some(pid),
            },
        )
        .unwrap();
        assert_eq!(met.peer, Peer::Admit);
        assert!(met.answered().is_none());
    }

    #[test]
    fn frames_round_trip_across_any_split() {
        let f = write_frame(7, b"hello");
        for cut in 0..f.len() {
            let mut fr = Frames::default();
            fr.push(&f[..cut]);
            assert!(fr.next().unwrap().is_none());
            fr.push(&f[cut..]);
            let (tag, p) = fr.next().unwrap().unwrap();
            assert_eq!(tag, WRITE);
            assert_eq!(u64::from_le_bytes(p[..8].try_into().unwrap()), 7);
            assert_eq!(&p[8..], b"hello");
        }
    }

    #[test]
    fn a_ready_frame_reads_back() {
        let k = start(Size { cols: 80, rows: 24 });
        let f = ready_frame(1234, &k);
        let mut fr = Frames::default();
        fr.push(&f);
        let (tag, p) = fr.next().unwrap().unwrap();
        assert_eq!(
            parse_said(tag, &p),
            Some(Said::Ready {
                version: KEEPER_VERSION,
                pgid: 1234,
                written: 0,
                size: Size { cols: 80, rows: 24 }
            })
        );
    }

    #[test]
    fn a_frame_too_large_is_refused() {
        let mut fr = Frames::default();
        let mut f = vec![WRITE];
        f.extend_from_slice(&(FRAME_MAX as u32 + 1).to_le_bytes());
        fr.push(&f);
        assert!(fr.next().is_err());
    }

    #[test]
    fn exit_carries_a_signal_or_none() {
        let f = exit_frame(130, Some(2));
        let (tag, p) = Frames { buf: f }.next().unwrap().unwrap();
        assert_eq!(
            parse_said(tag, &p),
            Some(Said::Exit {
                code: 130,
                signal: Some(2)
            })
        );
        let f = exit_frame(0, None);
        let (tag, p) = Frames { buf: f }.next().unwrap().unwrap();
        assert_eq!(
            parse_said(tag, &p),
            Some(Said::Exit {
                code: 0,
                signal: None
            })
        );
    }
}
