"""Sandbox jobs (exec v1.3): the native calls, and the baseline over plain commands.

Jobs are layer 1 in the spec; here they are the one helper that has two implementations behind
one set of operations. A job is kept by the sandbox from the start (``hold: false``), so closing a
connection never kills it. The operations are sans-IO generators (see :mod:`beyond_sandbox._drive`):
the native ones ask the transport for ``Call``/``Follow``, the baseline ones for ``Run``.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any, Sequence

from . import _scripts, _wire
from ._drive import Call, Emit, Follow, Offer, Op, Run, Sleep
from .errors import InvalidRequest, IoError, NotFound, NotOffered, SandboxError
from .models import (
    ExecResult,
    JobGap,
    JobInfo,
    JobOutput,
    JobRead,
    JobShellExit,
    Survivor,
)

#: Where the baseline keeps jobs in the sandbox.
JOBS_ROOT = "/tmp/beyond-jobs"
#: How much of a job's output the baseline keeps, at least.
OUTPUT_CAP = 8 * 1024 * 1024
#: The most one ``read`` returns by default: its base64 fits the smallest answer a runner keeps whole.
READ_MAX = 128 * 1024
#: The SIGTERM grace a kill gives a job before SIGKILL.
KILL_GRACE_MS = 300
#: One job operation: a round trip and a few forks.
OP_TIMEOUT_MS = 30_000
#: A polling follower's first and longest wait between reads, and the failures it tolerates.
POLL_MIN_S = 0.25
POLL_MAX_S = 2.0
POLL_FAILURES = 5

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


def one_line(label: str) -> str:
    return "".join(" " if (ord(c) < 32 or 0x7F <= ord(c) < 0xA0) else c for c in label)


def kind(root: str) -> Op[str]:
    """``native`` if the transport offers v1.3, else ``baseline`` if the sandbox's tools allow it."""
    offer = yield Offer()
    if offer.native:
        return "native"
    return "baseline" if (yield from baseline_available(root)) else "none"


def baseline_available(root: str) -> Op[bool]:
    argv = ["-c", _scripts.JOB, "sh", "probe", root, "probe"]
    r: ExecResult = yield Run("sh", argv, stdin=b"stdin-probe\n", timeout_ms=OP_TIMEOUT_MS)
    lines = r.stdout.decode("utf-8", "replace").split("\n")
    return r.exit_code == 0 and lines[0].strip() == "ok"


# ---- baseline ------------------------------------------------------------------------------------


def _sh(root: str, scope: str, op: str, *args: str) -> Run:
    return Run("sh", ["-c", _scripts.JOB, "sh", op, root, scope, *args], timeout_ms=OP_TIMEOUT_MS)


def _op(root: str, scope: str, op: str, *args: str) -> Op[str]:
    r: ExecResult = yield _sh(root, scope, op, *args)
    if r.exit_code != 0:
        why = r.stderr.decode("utf-8", "replace").strip()
        raise IoError(f"sandbox job `{op}` failed (exit {r.exit_code}): {why}")
    return r.stdout.decode("utf-8", "replace")


def _b_start(
    root: str, scope: str, job: str, command: str, args: Sequence[str], cwd: str | None, label: str
) -> Op[int]:
    out = yield from _op(
        root, scope, "start", job, "0", str(OUTPUT_CAP), cwd or "", one_line(label),
        _scripts.SUPERVISOR, command, *args,
    )  # fmt: skip
    try:
        return int(out.strip())
    except ValueError:
        raise IoError(f"the sandbox did not report the job's group: {out!r}") from None


_STATES = {"running": False, "lost": False, "gone": False, "exited": True, "ended": True}


def _status(state: str, code: str) -> tuple[str, int | None] | None:
    if state not in _STATES:
        return None
    if not _STATES[state]:
        return state, None
    try:
        return state, int(code.strip())
    except ValueError:
        return None


def parse_read(out: str) -> JobRead:
    """``<state> <code> <offset> <end>``, then the base64 of the bytes."""

    def bad() -> IoError:
        return IoError(f"unreadable job read: {out[:200]!r}")

    head, _, rest = out.partition("\n")
    head = head.strip()
    if head == "gone":
        return JobRead("gone", None, 0, 0, b"")
    f = head.split(" ")
    if len(f) != 4:
        raise bad()
    st = _status(f[0], f[1])
    if st is None or not (f[2].isascii() and f[2].isdigit() and f[3].isascii() and f[3].isdigit()):
        raise bad()
    try:
        data = base64.b64decode(rest.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise bad() from None
    return JobRead(st[0], st[1], int(f[2]), int(f[3]), data)


def parse_list_line(line: str) -> JobInfo | None:
    """``<id> <state> <code> <num|-> <kind|-> <seen> <started> <ended> <label...>``."""
    f = line.split(" ", 8)
    if len(f) < 8 or not _JOB_ID.fullmatch(f[0]):
        return None
    st = _status(f[1], f[2])
    if st is None:
        return None
    num, kind_ = f[3], f[4]

    def n(s: str) -> int:
        return int(s) if s.isascii() and s.isdigit() else 0

    return JobInfo(
        id=f[0],
        state=st[0],
        code=st[1],
        kept=kind_ != "-" or num != "-",
        num=int(num) if num.isascii() and num.isdigit() else None,
        kind=None if kind_ == "-" else kind_,
        seen=n(f[5]),
        started=n(f[6]),
        ended=n(f[7]),
        label=f[8] if len(f) > 8 else "",
    )


def _b_follow(root: str, scope: str, job: str, offset: int) -> Op[int]:
    """Poll ``read`` with a growing wait until the job has ended and its output is drained."""
    wait, failures, shell_told = POLL_MIN_S, 0, False
    while True:
        try:
            out = yield from _op(root, scope, "read", job, str(offset), str(READ_MAX))
            chunk = parse_read(out)
            failures = 0
        except SandboxError:
            failures += 1
            if failures >= POLL_FAILURES:
                raise
            yield Sleep(wait)
            wait = min(wait * 2, POLL_MAX_S)
            continue
        if chunk.state == "gone":
            raise NotFound("the job is no longer in the sandbox")
        if chunk.offset > offset:
            yield Emit(JobGap(chunk.offset - offset))
            offset = chunk.offset
        if chunk.data:
            yield Emit(JobOutput(chunk.data))
            offset += len(chunk.data)
            wait = POLL_MIN_S
            if offset < chunk.end:
                continue
        if chunk.state == "exited" and not shell_told:
            shell_told = True
            assert chunk.code is not None
            yield Emit(JobShellExit(chunk.code))
        elif chunk.state == "ended" and (offset >= chunk.end or not chunk.data):
            assert chunk.code is not None
            return chunk.code
        elif chunk.state == "ended":
            continue
        elif chunk.state == "lost":
            raise IoError(
                "the job's supervisor in the sandbox is gone, and how it ended was not recorded"
            )
        yield Sleep(wait)
        wait = min(wait * 2, POLL_MAX_S)


def _b_kill(root: str, scope: str, ids: Sequence[str], remove: bool) -> Op[list[Survivor]]:
    if not ids:
        return []
    out = yield from _op(root, scope, "kill", str(KILL_GRACE_MS), "1" if remove else "0", *ids)
    survivors: list[Survivor] = []
    for line in out.split("\n"):
        f = line.split(" ")
        if len(f) >= 3 and f[0] == "survived" and _JOB_ID.fullmatch(f[1]):
            survivors.append(Survivor(f[1], _wire.describe_survivors(f[2])))
    return survivors


def _b_wipe(root: str, scope: str) -> Op[list[str]]:
    out = yield from _op(root, scope, "wipe", str(KILL_GRACE_MS))
    return [
        _wire.describe_survivors(line.removeprefix("survived - "))
        for line in out.split("\n")
        if line.startswith("survived - ")
    ]


# ---- native ----------------------------------------------------------------------------------------


def _n_start(
    scope: str, job: str, command: str, args: Sequence[str], cwd: str | None, label: str
) -> Op[int]:
    v = yield Call(
        _wire.job_op("start", scope, id=job, hold=False, label=label), command, args, cwd
    )
    pgid = v.get("started") if isinstance(v, dict) else None
    if not isinstance(pgid, int) or isinstance(pgid, bool):
        raise IoError(f"the endpoint did not report the job's start: {str(v)[:200]}")
    return pgid


# ---- the operations, whichever the sandbox has -----------------------------------------------------


@dataclass(slots=True)
class JobsState:
    """What a :class:`~beyond_sandbox.Jobs` learned: ``native``, ``baseline`` or ``none``."""

    kind: str | None = None


def resolve(state: JobsState, root: str) -> Op[str]:
    if state.kind is None:
        state.kind = yield from kind(root)
    return state.kind


def _available(state: JobsState, root: str) -> Op[str]:
    k = yield from resolve(state, root)
    if k == "none":
        raise NotOffered(
            "this sandbox cannot keep jobs: no native jobs, and its tools lack what the baseline needs"
        )
    return k


def start(
    state: JobsState,
    root: str,
    scope: str,
    job: str,
    command: str,
    args: Sequence[str] = (),
    cwd: str | None = None,
    label: str = "",
) -> Op[int]:
    check_scope(scope)
    check_id(job)
    k = yield from _available(state, root)
    if k == "native":
        return (yield from _n_start(scope, job, command, args, cwd, label))
    return (yield from _b_start(root, scope, job, command, args, cwd, label))


def read(
    state: JobsState, root: str, scope: str, job: str, offset: int = 0, max_bytes: int = READ_MAX
) -> Op[JobRead]:
    check_scope(scope)
    check_id(job)
    k = yield from _available(state, root)
    if k == "native":
        v = yield Call(_wire.job_op("read", scope, id=job, offset=offset, max=max_bytes))
        return _wire.parse_read(v)
    return parse_read((yield from _op(root, scope, "read", job, str(offset), str(max_bytes))))


def follow(state: JobsState, root: str, scope: str, job: str, offset: int = 0) -> Op[int]:
    """Emits the job's events; returns its status."""
    check_scope(scope)
    check_id(job)
    k = yield from _available(state, root)
    if k == "native":
        code = yield Follow(_wire.job_op("attach", scope, id=job, offset=offset))
        return int(code)
    return (yield from _b_follow(root, scope, job, offset))


def list_jobs(state: JobsState, root: str, scope: str) -> Op[list[JobInfo]]:
    check_scope(scope)
    k = yield from _available(state, root)
    if k == "native":
        return _wire.parse_list((yield Call(_wire.job_op("list", scope))))
    out = yield from _op(root, scope, "list")
    return [j for j in map(parse_list_line, out.split("\n")) if j is not None]


def kill(
    state: JobsState, root: str, scope: str, ids: Sequence[str], remove: bool = False
) -> Op[list[Survivor]]:
    check_scope(scope)
    for i in ids:
        check_id(i)
    k = yield from _available(state, root)
    if k == "native":
        if not ids:
            return []
        v = yield Call(
            _wire.job_op("kill", scope, ids=list(ids), remove=remove, grace_ms=KILL_GRACE_MS)
        )
        return [s for s in _wire.parse_survivors(v) if s.id is not None and _JOB_ID.fullmatch(s.id)]
    return (yield from _b_kill(root, scope, ids, remove))


def remove(state: JobsState, root: str, scope: str, ids: Sequence[str]) -> Op[None]:
    check_scope(scope)
    for i in ids:
        check_id(i)
    k = yield from _available(state, root)
    if not ids:
        return
    if k == "native":
        yield Call(_wire.job_op("remove", scope, ids=list(ids)))
    else:
        yield from _op(root, scope, "rm", *ids)


def wipe(state: JobsState, root: str, scope: str) -> Op[list[str]]:
    check_scope(scope)
    k = yield from _available(state, root)
    if k == "native":
        v = yield Call(_wire.job_op("wipe", scope, grace_ms=KILL_GRACE_MS))
        return [s.detail for s in _wire.parse_survivors(v)]
    return (yield from _b_wipe(root, scope))


__all__: Sequence[Any] = ["JOBS_ROOT", "JobsState"]
