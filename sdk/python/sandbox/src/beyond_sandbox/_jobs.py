"""Sandbox jobs: the operations, as sans-IO generators (see :mod:`beyond_sandbox._drive`).

A job is kept by the sandbox
from the start (``hold: false``), so closing a connection never kills
it. Every operation asks the transport for a ``Call`` (one JSON answer) or a ``Follow`` (a stream).
"""

from __future__ import annotations

import contextlib
import re
import uuid
from typing import Any, Sequence

from . import _wire
from ._drive import Call, Follow, Op
from .errors import (
    DiskFullError,
    InvalidRequestError,
    IoError,
    JobExistsError,
    JobGoneError,
    KeyConflictError,
    NotOfferedError,
    SandboxError,
    StdinBusyError,
    StdinClosedError,
    StdinGapError,
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


_KEY = re.compile(r"[0-9A-Za-z._:-]{1,128}")


def start_key(key: str | None) -> str:
    """The start's idempotency key: the caller's, checked, else a fresh one. Taken once per call, before the first
    send, so a start sent again for a lost answer carries the key of the first send."""
    if key is None:
        return uuid.uuid4().hex
    if not _KEY.fullmatch(key):
        raise InvalidRequestError(
            f"bad idempotency key {key!r}: 1 to 128 characters of [0-9A-Za-z._:-]"
        )
    return key


def check_scope(scope: str) -> str:
    if not _SCOPE.fullmatch(scope):
        raise InvalidRequestError(f"bad job scope {scope!r}: 1 to 64 characters of [0-9a-z-]")
    return scope


def _written_in(detail: str) -> int | None:
    m = re.search(r'"written"\s*:\s*(\d+)', detail)
    return int(m.group(1)) if m else None


def _start_failure(e: SandboxError, job: str) -> SandboxError:
    """A refused ``start``: 409 is an id that exists, 507 a full disk (execd's ``spool_failure``)."""
    if e.status == 409 and "key_conflict" in e.detail:
        return KeyConflictError(f"job {job}: {e.detail}", id=job, status=409)
    if e.status == 409:
        return JobExistsError(f"job {job} exists: {e.detail}", id=job, status=409)
    if e.status == 507:
        return DiskFullError(e.detail, status=507)
    return e


def _gone_failure(e: SandboxError, job: str) -> SandboxError:
    """404 from an op on a job: it does not exist, or was removed."""
    if e.status == 404:
        return JobGoneError(f"job {job}: {e.detail}", id=job, status=404)
    return e


def _stdin_failure(e: SandboxError, job: str) -> SandboxError:
    """A refused stdin ``write``, told apart as the reference client tells them
    (``crates/sandbox/src/exec_endpoint/jobs_native.rs``, ``write_stdin``): 404 no such job; 409
    ``has no stdin`` (never opened); 409 ``gap`` (a write ahead of the stream); any other 409
    (closed); 429 too many waiting writes; 413 too large."""
    d = e.detail
    detail = f"job stdin: {d}"
    if e.status == 404:
        return JobGoneError(detail, id=job, status=404)
    if e.status == 409 and "has no stdin" in d:
        return StdinClosedError(detail, reason="never_opened", status=409)
    if e.status == 409 and '"gap"' in d:
        return StdinGapError(detail, written=_written_in(d), status=409)
    if e.status == 409:
        return StdinClosedError(detail, reason="closed", written=_written_in(d), status=409)
    if e.status == 429:
        return StdinBusyError(detail, status=429)
    if e.status == 413:
        return InvalidRequestError(detail, status=413)
    return e


def start(
    scope: str,
    job: str,
    command: str,
    args: Sequence[str] = (),
    cwd: str | None = None,
    label: str = "",
    env: dict[str, str] | None = None,
    stdin_open: bool = False,
    heartbeat_seconds: int | None = None,
    idempotency_key: str | None = None,
) -> Op[int]:
    check_scope(scope)
    check_id(job)
    key = start_key(idempotency_key)
    try:
        v = yield Call(
            _wire.job_op(
                "start",
                scope,
                id=job,
                hold=False,
                label=label,
                stdin_open=stdin_open,
                idempotency_key=key,
            ),
            command,
            args,
            cwd,
            env,
            heartbeat_seconds,
        )
    except SandboxError as e:
        raise _start_failure(e, job) from None
    pgid = v.get("started") if isinstance(v, dict) else None
    if not isinstance(pgid, int) or isinstance(pgid, bool):
        raise IoError(f"the endpoint did not report the job's start: {str(v)[:200]}")
    if stdin_open and not (isinstance(v, dict) and v.get("stdin") is True):
        # Started without its stdin: stop it, so nothing is left running that reads EOF.
        with contextlib.suppress(SandboxError):
            yield Call(_wire.job_op("kill", scope, ids=[job], remove=True, grace_ms=KILL_GRACE_MS))
        raise NotOfferedError("the endpoint predates job stdin: it ran the job without it")
    return pgid


def start_held(
    scope: str,
    job: str,
    command: str,
    args: Sequence[str] = (),
    cwd: str | None = None,
    label: str = "",
    env: dict[str, str] | None = None,
    stdin_open: bool = False,
    heartbeat_seconds: int | None = None,
    idempotency_key: str | None = None,
) -> Op[JobEnd]:
    """Start a job held by this stream (wire ``hold: true``). Emits the job's events, the first a
    :class:`~beyond_sandbox.JobStarted`; returns how it ended. If the stream stops before the end and
    the job was not kept (:func:`keep`), the sandbox kills its group."""
    check_scope(scope)
    check_id(job)
    key = start_key(idempotency_key)
    try:
        end: JobEnd = yield Follow(
            _wire.job_op(
                "start",
                scope,
                id=job,
                hold=True,
                label=label,
                stdin_open=stdin_open,
                idempotency_key=key,
            ),
            command,
            args,
            cwd,
            env,
            heartbeat_seconds,
        )
    except SandboxError as e:
        raise _start_failure(e, job) from None
    return end


def read(scope: str, job: str, offset: int = 0, max_bytes: int = READ_MAX) -> Op[JobRead]:
    """Up to ``max_bytes`` from ``offset``. A job that does not exist or was removed (wire state
    ``gone``) is :class:`~beyond_sandbox.JobGoneError`."""
    check_scope(scope)
    check_id(job)
    v = yield Call(_wire.job_op("read", scope, id=job, offset=offset, max=max_bytes))
    r = _wire.parse_read(v)
    if r.state == "gone":
        raise JobGoneError(f"job {job} is gone: it does not exist or was removed", id=job)
    return r


def follow(
    scope: str, job: str, offset: int = 0, heartbeat_seconds: int | None = None
) -> Op[JobEnd]:
    """Emits the job's events; returns how it ended. A job that does not exist is
    :class:`~beyond_sandbox.JobGoneError`."""
    check_scope(scope)
    check_id(job)
    try:
        end: JobEnd = yield Follow(
            _wire.job_op("attach", scope, id=job, offset=offset),
            heartbeat_seconds=heartbeat_seconds,
        )
    except SandboxError as e:
        raise _gone_failure(e, job) from None
    return end


def keep(
    scope: str,
    job: str,
    num: int | None = None,
    kind: str | None = None,
    seen: int | None = None,
) -> Op[None]:
    """Hand a job to the sandbox for good (wire ``keep``): after it a closed held stream kills
    nothing. Records ``num``, ``kind`` and ``seen`` for ``list``. (``notify`` is not offered: it carries
    a bearer token, which this client does not hold.)"""
    check_scope(scope)
    check_id(job)
    fields: dict[str, Any] = {}
    for name, value in (("num", num), ("seen", seen)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidRequestError(f"`{name}` must be a non-negative integer")
            fields[name] = value
    if kind is not None:
        fields["kind"] = kind
    try:
        yield Call(_wire.job_op("keep", scope, id=job, **fields))
    except SandboxError as e:
        raise _gone_failure(e, job) from None


def seen(scope: str, job: str, offset: int) -> Op[None]:
    """Record the offset of the output a person has been shown (wire ``seen``)."""
    check_scope(scope)
    check_id(job)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidRequestError("`offset` must be a non-negative integer")
    try:
        yield Call(_wire.job_op("seen", scope, id=job, offset=offset))
    except SandboxError as e:
        raise _gone_failure(e, job) from None


def list_jobs(scope: str) -> Op[list[JobInfo]]:
    check_scope(scope)
    return _wire.parse_list((yield Call(_wire.job_op("list", scope))))


def _grace(grace_ms: int) -> int:
    if isinstance(grace_ms, bool) or not isinstance(grace_ms, int) or grace_ms < 0:
        raise InvalidRequestError("`grace_ms` must be a non-negative integer")
    return grace_ms


def kill(
    scope: str, ids: Sequence[str], remove: bool = False, grace_ms: int = KILL_GRACE_MS
) -> Op[list[Survivor]]:
    check_scope(scope)
    _grace(grace_ms)
    for i in ids:
        check_id(i)
    if not ids:
        return []
    v = yield Call(_wire.job_op("kill", scope, ids=list(ids), remove=remove, grace_ms=grace_ms))
    return [s for s in _wire.parse_survivors(v) if s.id is not None and _JOB_ID.fullmatch(s.id)]


def remove(scope: str, ids: Sequence[str]) -> Op[None]:
    check_scope(scope)
    for i in ids:
        check_id(i)
    if ids:
        yield Call(_wire.job_op("remove", scope, ids=list(ids)))


def wipe(scope: str, grace_ms: int = KILL_GRACE_MS) -> Op[list[Survivor]]:
    check_scope(scope)
    v = yield Call(_wire.job_op("wipe", scope, grace_ms=_grace(grace_ms)))
    return _wire.parse_survivors(v)


def write_stdin(scope: str, job: str, at: int, data: bytes, eof: bool = False) -> Op[StdinWritten]:
    """Write ``data`` at position ``at`` of the job's stdin (it must have been started with
    ``stdin_open``). Positional: a repeat applies nothing twice, a write ahead of the stream waits for
    the bytes before it. All ``ConflictError``, told apart by class: ``StdinClosedError`` (closed, or
    never opened: retrying cannot succeed), ``StdinGapError`` (the bytes before it never came) and
    ``StdinBusyError`` (16 writes already wait: the same write, later, is safe). No such job is
    ``JobGoneError`` (a ``NotFoundError``); over :data:`WRITE_MAX` is ``InvalidRequestError``."""
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
        raise _stdin_failure(e, job) from None
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
