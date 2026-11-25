"""Layer 2: sandbox jobs over any transport.

A job is a command the *sandbox* keeps: it runs until it exits or is killed, whichever connection
started it, and its output can be read from any offset.
"""

from __future__ import annotations

import builtins
from typing import AsyncGenerator, AsyncIterator, Generator, Iterator, Sequence

from . import _jobs
from ._drive import _DONE, adrive, adrive_emit, aiter_events, drive, drive_iter
from .atransport import AsyncJobTransport
from .errors import NotOfferedError
from .models import (
    Attached,
    JobEnd,
    JobFrame,
    JobGap,
    JobInfo,
    JobOutput,
    JobRead,
    JobShellExit,
    JobStarted,
    StdinWritten,
    Survivor,
)
from .transport import JobTransport

READ_MAX = _jobs.READ_MAX


def _no_stdin(ev: JobFrame, stdin_open: bool) -> bool:
    """A held start asked for its stdin and the endpoint's ``started`` frame does not echo it."""
    return stdin_open and isinstance(ev, JobStarted) and not ev.stdin


_NO_STDIN = "the endpoint predates job stdin: it ran the job without it"


def _no_key(ev: JobFrame, key: str) -> bool:
    """A held start's ``started`` frame does not echo the start's idempotency key (the endpoint ignores keys)."""
    return isinstance(ev, JobStarted) and ev.key != key


def _collect(a: Attached, ev: JobFrame) -> None:
    if isinstance(ev, JobOutput):
        a.output += ev.data
        if ev.stderr:
            a.stderr += ev.data
    elif isinstance(ev, JobGap):
        a.gap += ev.bytes
    elif isinstance(ev, JobShellExit):
        a.shell_exit = ev.code


class Jobs:
    """Jobs, grouped in *scopes*: a scope is an opaque name (1 to 64 of ``[0-9a-z-]``) for one
    session's jobs. A job id is 1 to 32 characters of ``[0-9a-z]``, chosen by the caller."""

    def __init__(self, transport: JobTransport) -> None:
        self._t = transport

    def start(
        self,
        scope: str,
        job: str,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        label: str = "",
        env: dict[str, str] | None = None,
        stdin_open: bool = False,
        heartbeat_seconds: int | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        """Start a job the sandbox keeps from the start; returns its process group. ``env`` is the
        command's own environment; ``stdin_open`` keeps its stdin open for :meth:`write_stdin`
        (an endpoint that predates it is a ``NotOfferedError`` and the job is stopped). An id that
        exists is :class:`~beyond_sandbox.JobExistsError`; a full disk is
        :class:`~beyond_sandbox.DiskFullError`. The start carries an idempotency key
        (``idempotency_key``, 1 to 128 characters of ``[0-9A-Za-z._:-]``; omitted, one is generated for this
        call): sent again, it answers the same job and runs nothing, and the key of another command is
        :class:`~beyond_sandbox.JobExistsError` too (``409``). Pass your own to make a re-run of a whole
        program return its job."""
        op = _jobs.start(
            scope,
            job,
            command,
            args,
            cwd,
            label,
            env,
            stdin_open,
            heartbeat_seconds,
            idempotency_key,
        )
        return drive(op, self._t)

    def start_held(
        self,
        scope: str,
        job: str,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        label: str = "",
        env: dict[str, str] | None = None,
        stdin_open: bool = False,
        heartbeat_seconds: int | None = None,
        idempotency_key: str | None = None,
    ) -> Generator[JobFrame, None, None]:
        """Start a job held by the returned stream (wire ``hold: true``): events, the first a
        :class:`JobStarted`, ending with a :class:`JobEnd`. If you stop before the end and did not
        :meth:`keep` the job, the sandbox kills its group."""
        key = _jobs.start_key(idempotency_key)
        op = _jobs.start_held(
            scope,
            job,
            command,
            args,
            cwd,
            label,
            env,
            stdin_open,
            heartbeat_seconds,
            key,
        )
        items = drive_iter(op, self._t)
        try:
            for item in items:
                if isinstance(item, tuple) and item and item[0] is _DONE:
                    yield item[1]
                else:
                    if _no_stdin(item, stdin_open):
                        raise NotOfferedError(_NO_STDIN)
                    if _no_key(item, key):
                        raise NotOfferedError(_jobs.NO_START_KEY)
                    yield item
        finally:
            items.close()

    def keep(
        self,
        scope: str,
        job: str,
        *,
        num: int | None = None,
        kind: str | None = None,
        seen: int | None = None,
    ) -> None:
        """Hand a job to the sandbox for good (wire ``keep``): a closed :meth:`start_held` stream then
        kills nothing. Records ``num``, ``kind`` and ``seen`` for :meth:`list`."""
        drive(_jobs.keep(scope, job, num, kind, seen), self._t)

    def seen(self, scope: str, job: str, offset: int) -> None:
        """Record the offset of the output a person has been shown (wire ``seen``)."""
        drive(_jobs.seen(scope, job, offset), self._t)

    def write_stdin(
        self, scope: str, job: str, at: int, data: bytes, eof: bool = False
    ) -> StdinWritten:
        """Write ``data`` at position ``at`` of the job's stdin; ``eof`` closes it after."""
        return drive(_jobs.write_stdin(scope, job, at, data, eof), self._t)

    def signal(self, scope: str, jobs: Sequence[str], signal: str) -> builtins.list[str]:
        """Send a named signal to the running jobs' groups; the ids that were running."""
        return drive(_jobs.signal(scope, jobs, signal), self._t)

    def list(self, scope: str) -> builtins.list[JobInfo]:
        return drive(_jobs.list_jobs(scope), self._t)

    def read(self, scope: str, job: str, offset: int = 0, max: int = READ_MAX) -> JobRead:  # noqa: A002
        """Up to ``max`` bytes of output from ``offset``. A job that does not exist or was removed is
        :class:`~beyond_sandbox.JobGoneError`."""
        return drive(_jobs.read(scope, job, offset, max), self._t)

    def follow(
        self, scope: str, job: str, offset: int = 0, *, heartbeat_seconds: int | None = None
    ) -> Iterator[JobFrame]:
        """Events (:class:`JobOutput`, :class:`JobGap`, :class:`JobShellExit`, …), ending with a
        :class:`JobEnd` once the job's group is empty. Stopping early leaves the job running. A job
        that does not exist is :class:`~beyond_sandbox.JobGoneError`. ``heartbeat_seconds`` (1 to 240,
        the ``Keepalive-Ping-Interval`` header) sets the stream's beat; it is presumed dead only after
        two beats and ten seconds of silence, never under 30 s."""
        op = _jobs.follow(scope, job, offset, heartbeat_seconds)
        for item in drive_iter(op, self._t):
            if isinstance(item, tuple) and item and item[0] is _DONE:
                yield item[1]
            else:
                yield item

    def collect(
        self, scope: str, job: str, offset: int = 0, *, heartbeat_seconds: int | None = None
    ) -> Attached:
        """Follow the job to its end and collect everything."""
        result = Attached(-1)
        for ev in self.follow(scope, job, offset, heartbeat_seconds=heartbeat_seconds):
            if isinstance(ev, JobEnd):
                result.exit_code, result.signal = ev.code, ev.signal
            else:
                _collect(result, ev)
        return result

    def kill(
        self,
        scope: str,
        jobs: Sequence[str],
        *,
        remove: bool = False,
        grace_ms: int = _jobs.KILL_GRACE_MS,
    ) -> builtins.list[Survivor]:
        """SIGTERM, ``grace_ms`` (default 300), SIGKILL to each job's group; the groups that survived."""
        return drive(_jobs.kill(scope, jobs, remove, grace_ms), self._t)

    def remove(self, scope: str, jobs: Sequence[str]) -> None:
        """Delete ended jobs; never a running one."""
        drive(_jobs.remove(scope, jobs), self._t)

    def wipe(self, scope: str, *, grace_ms: int = _jobs.KILL_GRACE_MS) -> builtins.list[Survivor]:
        """Kill and delete every job of the scope; the groups that survived."""
        return drive(_jobs.wipe(scope, grace_ms), self._t)


class AsyncJobs:
    """:class:`Jobs` for an :class:`~beyond_sandbox.AsyncJobTransport`."""

    def __init__(self, transport: AsyncJobTransport) -> None:
        self._t = transport

    async def start(
        self,
        scope: str,
        job: str,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        label: str = "",
        env: dict[str, str] | None = None,
        stdin_open: bool = False,
        heartbeat_seconds: int | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        op = _jobs.start(
            scope,
            job,
            command,
            args,
            cwd,
            label,
            env,
            stdin_open,
            heartbeat_seconds,
            idempotency_key,
        )
        return await adrive(op, self._t)

    async def start_held(
        self,
        scope: str,
        job: str,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        label: str = "",
        env: dict[str, str] | None = None,
        stdin_open: bool = False,
        heartbeat_seconds: int | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[JobFrame, None]:
        """As :meth:`Jobs.start_held`."""
        key = _jobs.start_key(idempotency_key)
        op = _jobs.start_held(
            scope,
            job,
            command,
            args,
            cwd,
            label,
            env,
            stdin_open,
            heartbeat_seconds,
            key,
        )
        events = aiter_events(op, self._t)
        try:
            async for item in events:
                if isinstance(item, tuple) and item and item[0] is _DONE:
                    yield item[1]
                else:
                    if _no_stdin(item, stdin_open):
                        raise NotOfferedError(_NO_STDIN)
                    if _no_key(item, key):
                        raise NotOfferedError(_jobs.NO_START_KEY)
                    yield item
        finally:
            await events.aclose()

    async def keep(
        self,
        scope: str,
        job: str,
        *,
        num: int | None = None,
        kind: str | None = None,
        seen: int | None = None,
    ) -> None:
        await adrive(_jobs.keep(scope, job, num, kind, seen), self._t)

    async def seen(self, scope: str, job: str, offset: int) -> None:
        await adrive(_jobs.seen(scope, job, offset), self._t)

    async def write_stdin(
        self, scope: str, job: str, at: int, data: bytes, eof: bool = False
    ) -> StdinWritten:
        return await adrive(_jobs.write_stdin(scope, job, at, data, eof), self._t)

    async def signal(self, scope: str, jobs: Sequence[str], signal: str) -> builtins.list[str]:
        return await adrive(_jobs.signal(scope, jobs, signal), self._t)

    async def list(self, scope: str) -> builtins.list[JobInfo]:
        return await adrive(_jobs.list_jobs(scope), self._t)

    async def read(self, scope: str, job: str, offset: int = 0, max: int = READ_MAX) -> JobRead:  # noqa: A002
        return await adrive(_jobs.read(scope, job, offset, max), self._t)

    async def follow(
        self, scope: str, job: str, offset: int = 0, *, heartbeat_seconds: int | None = None
    ) -> AsyncIterator[JobFrame]:
        """As :meth:`Jobs.follow`."""
        op = _jobs.follow(scope, job, offset, heartbeat_seconds)
        events = aiter_events(op, self._t)
        try:
            async for item in events:
                if isinstance(item, tuple) and item and item[0] is _DONE:
                    yield item[1]
                else:
                    yield item
        finally:
            await events.aclose()

    async def collect(
        self, scope: str, job: str, offset: int = 0, *, heartbeat_seconds: int | None = None
    ) -> Attached:
        result = Attached(-1)
        op = _jobs.follow(scope, job, offset, heartbeat_seconds)

        async def collect(ev: JobFrame) -> None:
            _collect(result, ev)

        end = await adrive_emit(op, self._t, collect)
        result.exit_code, result.signal = end.code, end.signal
        return result

    async def kill(
        self,
        scope: str,
        jobs: Sequence[str],
        *,
        remove: bool = False,
        grace_ms: int = _jobs.KILL_GRACE_MS,
    ) -> builtins.list[Survivor]:
        return await adrive(_jobs.kill(scope, jobs, remove, grace_ms), self._t)

    async def remove(self, scope: str, jobs: Sequence[str]) -> None:
        await adrive(_jobs.remove(scope, jobs), self._t)

    async def wipe(
        self, scope: str, *, grace_ms: int = _jobs.KILL_GRACE_MS
    ) -> builtins.list[Survivor]:
        return await adrive(_jobs.wipe(scope, grace_ms), self._t)
