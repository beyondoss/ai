"""A held job, followed by the client kernel (``crates/job-client``, proven in Verus).

This file is the shell: it makes the request the kernel names, reads the answer, turns it into an event,
and does what the kernel's actions say. It holds no status table, no offset sum, no retry counter and no
time constant; every one of those is the kernel's. The only decision a line here makes is which fact an
exception or a frame is (a cut stream is ``broke``; a line past the cap is ``bad``).
``tests/test_no_decision.py`` fails if this file grows one.

It is a sans-IO generator, like every op: it yields the effects of ``_drive.py`` (open, read a line, sleep,
emit) and the sync and async drivers do them.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import _kernel as K
from . import _wire
from ._drive import Call, Emit, NextLine, Op, Open, Release, Sleep
from .errors import (
    AuthError,
    ConnectError,
    DiskFullError,
    EndpointError,
    InvalidRequestError,
    JobExistsError,
    JobGoneError,
    NotOfferedError,
    ProtocolError,
    SandboxError,
    TimeoutError,
    TransportError,
)
from .models import (
    JobAttached,
    JobEnd,
    JobGap,
    JobOutput,
    JobShellExit,
    JobStarted,
    Reconnecting,
)


@dataclass(frozen=True, slots=True)
class Start:
    """A job to start: the command, held by the stream (``hold``) or kept from the start."""

    command: str
    args: Sequence[str] = ()
    cwd: str | None = None
    label: str = ""
    env: dict[str, str] | None = None
    stdin_open: bool = False
    hold: bool = False


@dataclass(frozen=True, slots=True)
class Attach:
    """A job to follow from ``offset``, naming the incarnation ``mark`` (empty: whichever holds the id)."""

    offset: int = 0
    mark: str = ""
    output: bool = True


def fault_error(fault: str, status: int, detail: str, job: str) -> SandboxError:
    """The error class of each of the kernel's faults. One to one."""
    said = detail or f"status {status}"
    if fault == "job_gone":
        return JobGoneError(f"job {job}: {said}", id=job, status=status or None)
    if fault == "job_exists":
        return JobExistsError(f"job {job} exists: {said}", id=job, status=status or None)
    if fault == "disk_full":
        return DiskFullError(said, status=status or None)
    if fault == "no_pty":
        return EndpointError(said, status=status or None)
    if fault == "deadline":
        return TimeoutError(f"job {job}: the deadline passed")
    if fault == "not_offered":
        return NotOfferedError(
            "the endpoint did not name the job's mark: it predates marks, so a cursor cannot be "
            "trusted to name one job"
        )
    if fault == "transport":
        return ConnectError(f"job {job}: {said}")
    if fault == "auth":
        return AuthError(said, status=status or None)
    return EndpointError(said, status=status) if status else ProtocolError(said)


def held(
    scope: str,
    job: str,
    what: Start | Attach,
    heartbeat_seconds: int | None = None,
    on_reconnecting: Callable[[Reconnecting], None] | None = None,
    timeout: float | None = None,
) -> Op[None]:
    """Frames in order, ``started`` first for a start and the last a :class:`~beyond_sandbox.JobEnd`.
    Closing the generator detaches. Every decision is the kernel's; every request, read and wait is a
    driver's."""
    k = K.load()
    policy = K.Policy(
        watch=isinstance(what, Attach) and not what.output,
        stdin_open=isinstance(what, Start) and what.stdin_open,
        kept=isinstance(what, Start) and not what.hold,
        beat_s=heartbeat_seconds or 0,
    )
    try:
        state = (
            k.state_for_start(K.key(), policy)
            if isinstance(what, Start)
            else k.state_for_attach(what.mark, what.offset, policy)
        )
    except ProtocolError as e:
        # The kernel refuses a mark that is not 1 to 32 hex digits: nothing was sent.
        raise InvalidRequestError("a job mark is 1 to 32 hex digits") from e

    limit = K.Limit(timeout)
    line: _wire.JobLine | None = None
    refused = ""
    event: bytes | None = K.Event.GO
    while True:
        read = event is None
        if limit.passed():
            # The caller's deadline is the shell's clock and the kernel's event: it ends the handle.
            event, read = K.Event.DEADLINE, False
        elif event is None:
            event, line = yield from _read()
        state, actions = k.step(state, event, K.now_ms(), K.draw())
        event = None
        for a in actions:
            if isinstance(a, K.Start) and isinstance(what, Start):
                event, refused = yield from _send(
                    _start_open(scope, job, what, heartbeat_seconds, a.silence_ms)
                )
            elif isinstance(a, K.OpenAttach):
                event, refused = yield from _send(
                    _attach_open(scope, job, what, a, heartbeat_seconds)
                )
            elif isinstance(a, K.Reconnecting):
                if on_reconnecting is not None:
                    on_reconnecting(Reconnecting(a.attempts, K.elapsed_ms(a.since_ms)))
            elif isinstance(a, K.Started):
                yield Emit(JobStarted(line.pgid if line else 0, a.stdin, a.mark))
            elif isinstance(a, K.Attached):
                yield Emit(JobAttached(a.mark, a.from_offset))
            elif isinstance(a, K.ShellExit):
                yield Emit(JobShellExit(a.code))
            elif isinstance(a, K.Kill):
                with contextlib.suppress(SandboxError):
                    yield Call(
                        _wire.job_op(
                            "kill", scope, ids=[job], remove=True, grace_ms=_wire.KILL_GRACE_MS
                        )
                    )
            elif isinstance(a, K.CloseStream):
                yield Release()
            elif isinstance(a, K.Deliver):
                if a.dropped:
                    yield Emit(JobGap(a.dropped))
                if line is not None and line.data is not None:
                    yield Emit(JobOutput(line.data, line.stderr))
            elif isinstance(a, K.DeliverExit):
                if a.dropped:
                    yield Emit(JobGap(a.dropped))
                yield Emit(JobEnd.of(a.code, a.signal, a.killed))
                event = K.Event.EXIT_TAKEN
            elif isinstance(a, K.Sleep):
                yield Sleep(a.until_ms, limit.ends_ms)
                event = K.Event.TICK
            elif isinstance(a, K.Fail):
                note = line.note if line is not None and read else ""
                raise fault_error(a.fault, a.status, note or refused, job)
            elif isinstance(a, (K.Finish, K.Detached)):
                return
            elif isinstance(a, K.Remove):
                # The handle asks for a remove only when told to remove on exit; this shell never is.
                raise ProtocolError("the kernel asked for a remove that was not requested")


def _start_open(
    scope: str, job: str, s: Start, heartbeat_seconds: int | None, silence_ms: int
) -> Open:
    return Open(
        _wire.job_op("start", scope, id=job, hold=s.hold, label=s.label, stdin_open=s.stdin_open),
        silence_ms,
        s.command,
        s.args,
        s.cwd,
        s.env,
        heartbeat_seconds,
    )


def _attach_open(
    scope: str, job: str, what: Start | Attach, a: K.OpenAttach, heartbeat_seconds: int | None
) -> Open:
    fields: dict[str, Any] = {}
    if a.mark:
        fields["mark"] = a.mark
    if isinstance(what, Attach) and not what.output:
        fields["output"] = False
    return Open(
        _wire.job_op("attach", scope, id=job, offset=a.offset, **fields),
        a.silence_ms,
        heartbeat_seconds=heartbeat_seconds,
    )


def _send(open_: Open) -> Op[tuple[bytes, str]]:
    """Send a request and report its answer as an event, with what a refusal said."""
    try:
        opened = yield open_
    except ConnectError as e:
        return K.Event.UNREACHED, e.detail
    except (TransportError, TimeoutError):
        return K.Event.BROKE, ""
    return K.Event.answered(opened.status, opened.retry_after_s), opened.detail


def _read() -> Op[tuple[bytes, _wire.JobLine | None]]:
    """The next line of the open stream, as an event, and the line."""
    try:
        raw = yield NextLine()
    except (TransportError, TimeoutError):
        return K.Event.BROKE, None
    except ProtocolError as e:
        bad = _wire.JobLine(K.Frame.BAD, note=e.detail)
        return K.Event.frame(bad.frame), bad
    if raw is None:
        return K.Event.BROKE, None
    line = _wire.decode_job_line(raw)
    return K.Event.frame(line.frame), line
