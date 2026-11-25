//! The keeper's decisions: one small process per terminal holds the pty's master, copies what the
//! terminal prints into the job's spool, applies positional writes and sizes, reaps the shell and
//! says so. `step` is a pure function from `(state, event)` to `(state, what to do)`; the keeper
//! (`execd::keeper`, a loop over `poll`) performs the actions and the model
//! (`crates/execd/tests/job_model.rs`) calls this very function.
//!
//! What the properties say is in `README.md` (K1, K2, T5).

use vstd::prelude::*;

use crate::{Plan, Size, advance, stdin_plan};

verus! {

/// The shell: running, or reaped with its status.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shell {
    Running,
    Reaped { code: i32, signal: Option<i32> },
}

/// The keeper's state.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct K {
    pub shell: Shell,
    /// The master read `EIO`: no slave is open anywhere.
    pub eio: bool,
    /// The shell was reaped and the master was then read to `EAGAIN` (or `EIO`): everything the
    /// shell printed before it exited is in the output.
    pub drained: bool,
    /// The shell's status is in the spool's `status` file.
    pub status_written: bool,
    /// The `Exit` frame went to the connection that is up.
    pub exit_sent: bool,
    /// execd is connected.
    pub conn: bool,
    /// execd has connected at least once: a shell that ends before it did is still served to it.
    pub seen: bool,
    /// The process that started the keeper is gone and never connected: nobody will ask.
    pub orphan: bool,
    /// How many bytes of the terminal's input stream landed. It lives here, not in execd, so a write
    /// retried across a restart of execd is still applied once.
    pub written: u64,
    pub size: Size,
    /// The spool could not be written: the output is no longer kept, so the terminal ends (Q6).
    pub failed: bool,
}

/// What the keeper learns.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KEv {
    /// The shell was reaped.
    ChildExit { code: i32, signal: Option<i32> },
    /// A read of the master returned `EAGAIN`: nothing more now.
    MasterEagain,
    /// A read of the master returned `EIO`: every slave is closed.
    MasterEio,
    /// execd asks for `len` bytes of input at stream position `at`.
    Write { at: u64, len: u64 },
    /// `n` bytes of an applied write reached the master.
    Landed { n: u64 },
    /// execd asks for a size.
    Resize { size: Size },
    /// execd connected (the first time, or after a restart).
    Conn,
    /// execd's connection closed.
    ConnLost,
    /// The process that started the keeper ended.
    ParentGone,
    /// A write of the terminal's output to the spool failed (no space, no device).
    SpoolFailed,
}

/// What the keeper does, all of it in this order: apply, size, status, exit frame, ready, end.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KAct {
    /// Skip `skip` bytes of the write (already applied) and write the next `take`.
    pub apply: Option<(u64, u64)>,
    /// `TIOCSWINSZ` to this size.
    pub set_size: Option<Size>,
    /// Write the shell's status to the spool, aside and renamed.
    pub write_status: bool,
    /// Tell execd the shell exited.
    pub send_exit: bool,
    /// Tell execd the keeper's state (`Ready`).
    pub send_ready: bool,
    /// `SIGKILL` the shell's session: its output has nowhere to go.
    pub kill_shell: bool,
    /// The keeper ends.
    pub exit: bool,
}

/// The code a terminal ends with when its output could not be kept: `EX_IOERR`, never a shell's own
/// status, so the reader sees why.
pub const SPOOL_FAILED_CODE: i32 = 74;

/// Q6: the status the keeper writes and tells (`(code, signal)`), once the shell is reaped: the shell's
/// own, or [`SPOOL_FAILED_CODE`] with no signal if the spool failed, whatever the shell did.
pub fn exit_status(k: &K) -> (r: Option<(i32, Option<i32>)>)
    ensures
        k.shell is Running ==> r is None,
        k.shell is Reaped && k.failed ==> r == Some((SPOOL_FAILED_CODE, None::<i32>)),
        k.shell is Reaped && !k.failed ==> r == Some((k.shell->Reaped_code, k.shell->Reaped_signal)),
{
    match k.shell {
        Shell::Running => None,
        Shell::Reaped { code, signal } => {
            if k.failed && !cfg!(feature = "mut_spool_failure_keeps_code") {
                Some((SPOOL_FAILED_CODE, None))
            } else {
                Some((code, signal))
            }
        },
    }
}

/// The state's own invariant, which every step keeps.
pub open spec fn inv(k: K) -> bool {
    &&& (k.drained ==> k.shell is Reaped)
    &&& (k.status_written ==> k.drained)
    &&& (k.exit_sent ==> k.status_written)
    &&& (k.eio && k.shell is Reaped ==> k.drained)
}

/// A new keeper at `size`: nothing happened yet.
pub fn start(size: Size) -> (k: K)
    ensures
        inv(k),
        k.written == 0,
        k.size == size,
        !k.conn,
{
    K { shell: Shell::Running, eio: false, drained: false, status_written: false, exit_sent: false, conn: false, seen: false, orphan: false, written: 0, size, failed: false }
}

/// K1, K2, T5: what one event does to a keeper.
///
/// - K1: the shell's exit is told (and its status written) only after everything it printed is in the
///   output, and the status is written before the exit is told.
/// - K2: once the shell is reaped and no slave is open, the keeper ends in the same step, after it
///   wrote the status and told a connected execd (F6: the kernel reports `EIO` once the last slave
///   closes; F7: the keeper is scheduled).
/// - T5: `written` only grows, by exactly the bytes that landed; a connection made, lost or made
///   again never changes it, and an apply starts exactly at it, so no input is lost or repeated.
pub fn step(k: &mut K, ev: KEv) -> (act: KAct)
    requires
        inv(*old(k)),
    ensures
        inv(*final(k)),
        // The shell is reaped once.
        old(k).shell is Reaped ==> final(k).shell == old(k).shell,
        // T5.
        final(k).written >= old(k).written,
        !(ev is Landed) ==> final(k).written == old(k).written,
        ev is Landed && old(k).written as int + ev->Landed_n as int <= u64::MAX as int
            ==> final(k).written as int == old(k).written as int + ev->Landed_n as int,
        (ev is Conn || ev is ConnLost) ==> final(k).written == old(k).written,
        act.apply is Some ==> (ev is Write && !old(k).eio
            && ev->Write_at as int + act.apply.unwrap().0 as int == old(k).written as int
            && act.apply.unwrap().0 as int + act.apply.unwrap().1 as int == ev->Write_len as int
            && act.apply.unwrap().1 > 0),
        // A write that is already applied is never applied again.
        ev is Write && ev->Write_at as int + ev->Write_len as int <= old(k).written as int ==> act.apply is None,
        // A write on a master that read EIO is never applied.
        old(k).eio ==> act.apply is None,
        // The size.
        ev is Resize && crate::in_bounds(ev->Resize_size) ==> (final(k).size == ev->Resize_size
            && act.set_size == Some(ev->Resize_size)),
        !(ev is Resize) ==> final(k).size == old(k).size,
        act.set_size is Some ==> (ev is Resize && crate::in_bounds(ev->Resize_size)),
        // Q6: a spool that failed is final, and the shell it fed is killed, once, in that very step.
        ev is SpoolFailed ==> final(k).failed,
        old(k).failed ==> final(k).failed,
        !(ev is SpoolFailed) ==> final(k).failed == old(k).failed,
        act.kill_shell ==> (ev is SpoolFailed && !old(k).failed && old(k).shell is Running),
        ev is SpoolFailed && !old(k).failed && old(k).shell is Running ==> act.kill_shell,
        // K1.
        act.write_status ==> (final(k).shell is Reaped && final(k).drained && final(k).status_written
            && !old(k).status_written
            && (old(k).drained || ev is MasterEagain || ev is MasterEio || (ev is ChildExit && old(k).eio))),
        act.send_exit ==> (final(k).shell is Reaped && final(k).drained && final(k).status_written
            && final(k).conn),
        // Reconnect: told again, with the same count.
        ev is Conn ==> (act.send_ready && final(k).conn),
        !(ev is Conn) ==> !act.send_ready,
        ev is ConnLost ==> !final(k).conn,
        ev is Conn ==> final(k).seen,
        old(k).seen ==> final(k).seen,
        // K2: a keeper that execd asked (or whose starter is gone) ends in the step its shell is reaped
        // and its master reads EIO; and not before execd has had the chance to ask (a shell that ends
        // at once is still served).
        final(k).shell is Reaped && final(k).eio && (final(k).seen || final(k).orphan) ==> act.exit,
        act.exit ==> (final(k).eio && final(k).status_written && (final(k).exit_sent || !final(k).conn)
            && (final(k).seen || final(k).orphan)),
{
    let mut apply: Option<(u64, u64)> = None;
    let mut set_size: Option<Size> = None;
    let mut send_ready = false;
    let mut kill_shell = false;
    match ev {
        KEv::ChildExit { code, signal } => {
            if let Shell::Running = k.shell {
                k.shell = Shell::Reaped { code, signal };
                k.drained = k.eio;
            }
        },
        KEv::MasterEagain => {
            if let Shell::Reaped { .. } = k.shell {
                k.drained = true;
            }
        },
        KEv::MasterEio => {
            k.eio = true;
            if let Shell::Reaped { .. } = k.shell {
                k.drained = true;
            }
        },
        KEv::Write { at, len } => {
            // The same decision the job's stdin makes (`stdin_plan`): the master is the open pipe, and
            // it has ended once it read EIO. A wait is not the keeper's: execd waits and sends it in order.
            let p = stdin_plan(k.written, at, len, false, !k.eio, k.eio);
            if let Plan::Apply { skip, take, .. } = p {
                apply = Some((skip, take));
            }
        },
        KEv::Landed { n } => {
            k.written = advance(k.written, n);
        },
        KEv::Resize { size } => {
            if size.cols >= 1 && size.cols <= 1024 && size.rows >= 1 && size.rows <= 1024 {
                k.size = size;
                set_size = Some(size);
            }
        },
        KEv::ParentGone => {
            if !k.seen {
                k.orphan = true;
            }
        },
        KEv::Conn => {
            k.conn = true;
            k.seen = true;
            k.exit_sent = false;
            send_ready = true;
            if cfg!(feature = "mut_keeper_reconnect_resets_written") {
                k.written = 0;
            }
        },
        KEv::ConnLost => {
            k.conn = false;
        },
        KEv::SpoolFailed => {
            if !cfg!(feature = "mut_spool_failure_silent") {
                kill_shell = !k.failed && match k.shell {
                    Shell::Running => true,
                    Shell::Reaped { .. } => false,
                };
                k.failed = true;
            }
        },
    }
    let drained = if cfg!(feature = "mut_exit_before_drain") {
        match k.shell {
            Shell::Reaped { .. } => true,
            Shell::Running => false,
        }
    } else {
        k.drained
    };
    let write_status = drained && !k.status_written;
    if write_status {
        k.status_written = true;
        if cfg!(feature = "mut_exit_before_drain") {
            k.drained = true;
        }
    }
    let send_exit = k.status_written && k.conn && !k.exit_sent;
    if send_exit {
        k.exit_sent = true;
    }
    let exit = !cfg!(feature = "mut_keeper_waits_forever")
        && k.eio
        && k.status_written
        && (k.exit_sent || !k.conn)
        && (k.seen || k.orphan || cfg!(feature = "mut_keeper_exits_unasked"));
    KAct { apply, set_size, write_status, send_exit, send_ready, kill_shell, exit }
}

} // verus!
