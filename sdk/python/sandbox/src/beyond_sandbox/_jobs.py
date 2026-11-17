"""Sandbox jobs: the operations, as sans-IO generators (see :mod:`beyond_sandbox._drive`).

A job is kept by the sandbox
from the start (``hold: false``), so closing a connection never kills
it. Every operation asks the transport for a ``Call`` (one JSON answer) or a ``Follow`` (a stream).
"""

from __future__ import annotations

import contextlib
import re
from typing import Sequence

from . import _wire
from ._drive import Call, Follow, Op
from .errors import (
    ConflictError,
    InvalidRequestError,
    IoError,
    NotFoundError,
    NotOfferedError,
    SandboxError,
)
from .models import JobEnd, JobInfo, JobRead, StdinWritten, Survivor

#: The most one ``read`` returns by default: its base64 fits the smallest answer a runner keeps whole.
READ_MAX = 128 * 1024
#: The SIGTERM grace a kill gives a job before SIGKILL.
KILL_GRACE_MS = 300
#: The most decoded bytes one ``write`` carries.
WRITE_MAX = 128 * 1024
#: The signals ``signal`` sends: names only.
SIGNALS = ("SIGHUP", "SIGINT", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGTERM", "SIGKILL")

_JOB_ID = _wire.JOB_ID
_SCOPE = re.compile(r"[0-9a-z-]{1,64}")


def check_id(job: str) -> str:
    if not _JOB_ID.fullmatch(job):
        raise InvalidRequestError(f"bad job id {job!r}: 1 to 32 characters of [0-9a-z]")
    return job


def check_scope(scope: str) -> str:
    if not _SCOPE.fullmatch(scope):
        raise InvalidRequestError(f"bad job scope {scope!r}: 1 to 64 characters of [0-9a-z-]")
    return scope


def start(
    scope: str,
    job: str,
    command: str,
    args: Sequence[str] = (),
    cwd: str | None = None,
    label: str = "",
    env: dict[str, str] | None = None,
    stdin_open: bool = False,
) -> Op[int]:
    check_scope(scope)
    check_id(job)
    v = yield Call(
        _wire.job_op("start", scope, id=job, hold=False, label=label, stdin_open=stdin_open),
        command,
        args,
        cwd,
        env,
    )
    pgid = v.get("started") if isinstance(v, dict) else None
    if not isinstance(pgid, int) or isinstance(pgid, bool):
        raise IoError(f"the endpoint did not report the job's start: {str(v)[:200]}")
    if stdin_open and not (isinstance(v, dict) and v.get("stdin") is True):
        # Started without its stdin: stop it, so nothing is left running that reads EOF.
        with contextlib.suppress(SandboxError):
            yield Call(_wire.job_op("kill", scope, ids=[job], remove=True, grace_ms=KILL_GRACE_MS))
        raise NotOfferedError("the endpoint predates job stdin: it ran the job without it")
    return pgid


def read(scope: str, job: str, offset: int = 0, max_bytes: int = READ_MAX) -> Op[JobRead]:
    check_scope(scope)
    check_id(job)
    v = yield Call(_wire.job_op("read", scope, id=job, offset=offset, max=max_bytes))
    return _wire.parse_read(v)


def follow(scope: str, job: str, offset: int = 0) -> Op[JobEnd]:
    """Emits the job's events; returns how it ended."""
    check_scope(scope)
    check_id(job)
    end: JobEnd = yield Follow(_wire.job_op("attach", scope, id=job, offset=offset))
    return end


def list_jobs(scope: str) -> Op[list[JobInfo]]:
    check_scope(scope)
    return _wire.parse_list((yield Call(_wire.job_op("list", scope))))


def kill(scope: str, ids: Sequence[str], remove: bool = False) -> Op[list[Survivor]]:
    check_scope(scope)
    for i in ids:
        check_id(i)
    if not ids:
        return []
    v = yield Call(
        _wire.job_op("kill", scope, ids=list(ids), remove=remove, grace_ms=KILL_GRACE_MS)
    )
    return [s for s in _wire.parse_survivors(v) if s.id is not None and _JOB_ID.fullmatch(s.id)]


def remove(scope: str, ids: Sequence[str]) -> Op[None]:
    check_scope(scope)
    for i in ids:
        check_id(i)
    if ids:
        yield Call(_wire.job_op("remove", scope, ids=list(ids)))


def wipe(scope: str) -> Op[list[Survivor]]:
    check_scope(scope)
    v = yield Call(_wire.job_op("wipe", scope, grace_ms=KILL_GRACE_MS))
    return _wire.parse_survivors(v)


def write_stdin(scope: str, job: str, at: int, data: bytes, eof: bool = False) -> Op[StdinWritten]:
    """Write ``data`` at position ``at`` of the job's stdin (it must have been started with
    ``stdin_open``). Positional: a repeat applies nothing twice, a write ahead of the stream waits for
    the bytes before it. ``NotFoundError``: no such job; ``ConflictError``: no stdin, closed, or a gap
    that never filled; ``InvalidRequestError``: over :data:`WRITE_MAX`."""
    check_scope(scope)
    check_id(job)
    if len(data) > WRITE_MAX:
        raise InvalidRequestError(f"a write is at most {WRITE_MAX} bytes, got {len(data)}")
    import base64

    try:
        v = yield Call(
            _wire.job_op(
                "write", scope, id=job, at=at, eof=eof, data=base64.b64encode(data).decode("ascii")
            )
        )
    except SandboxError as e:
        if e.status == 404:
            raise NotFoundError(f"job stdin: {e.detail}") from None
        if e.status in (409, 429):
            raise ConflictError(f"job stdin: {e.detail}") from None
        if e.status == 413:
            raise InvalidRequestError(f"job stdin: {e.detail}") from None
        raise
    return _wire.parse_written(v)


def signal(scope: str, ids: Sequence[str], signal_name: str) -> Op[list[str]]:
    """Send ``signal_name`` (one of :data:`SIGNALS`) to each running job's group, with no grace and
    no wait. The ids that were running; an ended or unknown job is absent."""
    check_scope(scope)
    for i in ids:
        check_id(i)
    if signal_name not in SIGNALS:
        raise InvalidRequestError(f"unsupported signal {signal_name!r}; one of {SIGNALS}")
    if not ids:
        return []
    v = yield Call(_wire.job_op("signal", scope, ids=list(ids), signal=signal_name))
    sent = v.get("sent") if isinstance(v, dict) else None
    if not isinstance(sent, list):
        raise IoError(f"unreadable job signal: {v!r}"[:300])
    return [i for i in sent if _wire._is_job_id(i)]
