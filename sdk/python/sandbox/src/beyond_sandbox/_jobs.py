"""Sandbox jobs: the operations, as sans-IO generators (see :mod:`beyond_sandbox._drive`).

A job is kept by the sandbox from the start (``hold: false``), so closing a connection never kills
it. Every operation asks the transport for a ``Call`` (one JSON answer) or a ``Follow`` (a stream).
"""

from __future__ import annotations

import re
from typing import Sequence

from . import _wire
from ._drive import Call, Follow, Op
from .errors import InvalidRequest, IoError
from .models import JobEnd, JobInfo, JobRead, Survivor

#: The most one ``read`` returns by default: its base64 fits the smallest answer a runner keeps whole.
READ_MAX = 128 * 1024
#: The SIGTERM grace a kill gives a job before SIGKILL.
KILL_GRACE_MS = 300

_JOB_ID = _wire.JOB_ID
_SCOPE = re.compile(r"[0-9a-z-]{1,64}")


def check_id(job: str) -> str:
    if not _JOB_ID.fullmatch(job):
        raise InvalidRequest(f"bad job id {job!r}: 1 to 32 characters of [0-9a-z]")
    return job


def check_scope(scope: str) -> str:
    if not _SCOPE.fullmatch(scope):
        raise InvalidRequest(f"bad job scope {scope!r}: 1 to 64 characters of [0-9a-z-]")
    return scope


def start(
    scope: str,
    job: str,
    command: str,
    args: Sequence[str] = (),
    cwd: str | None = None,
    label: str = "",
) -> Op[int]:
    check_scope(scope)
    check_id(job)
    v = yield Call(
        _wire.job_op("start", scope, id=job, hold=False, label=label), command, args, cwd
    )
    pgid = v.get("started") if isinstance(v, dict) else None
    if not isinstance(pgid, int) or isinstance(pgid, bool):
        raise IoError(f"the endpoint did not report the job's start: {str(v)[:200]}")
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


def wipe(scope: str) -> Op[list[str]]:
    check_scope(scope)
    v = yield Call(_wire.job_op("wipe", scope, grace_ms=KILL_GRACE_MS))
    return [s.detail for s in _wire.parse_survivors(v)]
