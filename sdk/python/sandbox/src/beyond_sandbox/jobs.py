"""Layer 2: sandbox jobs over any transport.

A job is a command the *sandbox* keeps: it runs until it exits or is killed, whichever connection
started it, and its output can be read from any offset.
"""

from __future__ import annotations

import builtins
from typing import AsyncGenerator, AsyncIterator, Callable, Generator, Iterator, Sequence

from . import _jobs
from ._drive import adrive, adrive_emit, aiter_items, drive, iter_items
from .atransport import AsyncJobTransport
from .models import (
    Attached,
    JobAttached,
    JobEnd,
    JobFrame,
    JobGap,
    JobInfo,
    JobOutput,
    JobRead,
    JobRemoved,
    JobShellExit,
    JobStart,
    Reconnecting,
    StdinWritten,
    Survivor,
)
from .transport import JobTransport

READ_MAX = _jobs.READ_MAX


def _collect(a: Attached, ev: JobFrame) -> None:
    if isinstance(ev, JobOutput):
        a.output += ev.data
        if ev.stderr:
            a.stderr += ev.data
    elif isinstance(ev, JobGap):
        a.gap += ev.bytes
    elif isinstance(ev, JobShellExit):
        a.shell_exit = ev.code
    elif isinstance(ev, JobAttached):
        a.attached = ev


class Jobs:
    """Jobs, grouped in *scopes*: a scope is an opaque name (1 to 64 of ``[0-9a-z-]``) for one
    session's jobs. A job id is 1 to 64 characters of ``[0-9a-z_-]``, chosen by the caller."""

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
    ) -> JobStart:
        """Start a job the sandbox keeps from the start; returns its process group and mark. ``env`` is the
        command's own environment; ``stdin_open`` keeps its stdin open for :meth:`write_stdin`
        (an endpoint that predates it is a ``NotOfferedError`` and the job is stopped). An id that
        exists is :class:`~beyond_sandbox.JobExistsError`; a full disk is
        :class:`~beyond_sandbox.DiskFullError`."""
        op = _jobs.start(scope, job, command, args, cwd, label, env, stdin_open, heartbeat_seconds)
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
        on_reconnecting: Callable[[Reconnecting], None] | None = None,
    ) -> Generator[JobFrame, None, None]:
        """Start a job held by the returned stream (wire ``hold: true``): events, the first a
        :class:`JobStarted`, ending with a :class:`JobEnd`. If you stop before the end and did not
        :meth:`keep` the job, the sandbox kills its group. A cut or silent stream is reopened at the
        offset reached, by the client kernel; a handle that cannot reopen says so through
        ``on_reconnecting`` and keeps trying."""
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
            on_reconnecting,
        )
        yield from iter_items(op, self._t)

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
        self,
        scope: str,
        job: str,
        offset: int = 0,
        *,
        heartbeat_seconds: int | None = None,
        mark: str | None = None,
        output: bool = True,
        on_reconnecting: Callable[[Reconnecting], None] | None = None,
    ) -> Iterator[JobFrame]:
        """Events, the first a :class:`JobAttached` (the job's mark and where the stream starts), then
        (:class:`JobOutput`, :class:`JobGap`, :class:`JobShellExit`, …), ending with a
        :class:`JobEnd` once the job's group is empty. Stopping early leaves the job running. A job
        that does not exist is :class:`~beyond_sandbox.JobGoneError`. A cut or silent stream is
        reopened at the offset reached, by the client kernel (``crates/job-client``); a handle that
        cannot reopen says so through ``on_reconnecting`` and keeps trying. ``heartbeat_seconds`` (1 to
        240, the ``Keepalive-Ping-Interval`` header) sets the stream's beat; the stream is presumed
        dead after two beats and ten seconds of silence, never under 30 s. ``mark`` names the
        incarnation of the id to follow: another is a ``JobGoneError`` and nothing is streamed.
        ``output=False`` sends no output and no gap, only the end. An endpoint that does not name the
        mark is :class:`~beyond_sandbox.NotOfferedError`, before any byte."""
        op = _jobs.follow(scope, job, offset, heartbeat_seconds, mark, output, on_reconnecting)
        yield from iter_items(op, self._t)

    def collect(
        self,
        scope: str,
        job: str,
        offset: int = 0,
        *,
        heartbeat_seconds: int | None = None,
        mark: str | None = None,
        output: bool = True,
        on_reconnecting: Callable[[Reconnecting], None] | None = None,
    ) -> Attached:
        """Follow the job to its end and collect everything."""
        result = Attached(-1)
        for ev in self.follow(
            scope,
            job,
            offset,
            heartbeat_seconds=heartbeat_seconds,
            mark=mark,
            output=output,
            on_reconnecting=on_reconnecting,
        ):
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

    def remove(
        self, scope: str, jobs: Sequence[str], *, marks: dict[str, str] | None = None
    ) -> JobRemoved:
        """Delete ended jobs; never a running one. What was skipped is answered: ``running`` jobs, and
        ``mismatch`` jobs whose mark is not the one ``marks`` names for their id."""
        return drive(_jobs.remove(scope, jobs, marks), self._t)

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
    ) -> JobStart:
        op = _jobs.start(scope, job, command, args, cwd, label, env, stdin_open, heartbeat_seconds)
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
        on_reconnecting: Callable[[Reconnecting], None] | None = None,
    ) -> AsyncGenerator[JobFrame, None]:
        """As :meth:`Jobs.start_held`."""
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
            on_reconnecting,
        )
        async for item in aiter_items(op, self._t):
            yield item

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
        self,
        scope: str,
        job: str,
        offset: int = 0,
        *,
        heartbeat_seconds: int | None = None,
        mark: str | None = None,
        output: bool = True,
        on_reconnecting: Callable[[Reconnecting], None] | None = None,
    ) -> AsyncIterator[JobFrame]:
        """As :meth:`Jobs.follow`."""
        op = _jobs.follow(scope, job, offset, heartbeat_seconds, mark, output, on_reconnecting)
        async for item in aiter_items(op, self._t):
            yield item

    async def collect(
        self,
        scope: str,
        job: str,
        offset: int = 0,
        *,
        heartbeat_seconds: int | None = None,
        mark: str | None = None,
        output: bool = True,
        on_reconnecting: Callable[[Reconnecting], None] | None = None,
    ) -> Attached:
        result = Attached(-1)
        op = _jobs.follow(scope, job, offset, heartbeat_seconds, mark, output, on_reconnecting)

        async def collect(ev: JobFrame) -> None:
            if isinstance(ev, JobEnd):
                result.exit_code, result.signal = ev.code, ev.signal
            else:
                _collect(result, ev)

        await adrive_emit(op, self._t, collect)
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

    async def remove(
        self, scope: str, jobs: Sequence[str], *, marks: dict[str, str] | None = None
    ) -> JobRemoved:
        return await adrive(_jobs.remove(scope, jobs, marks), self._t)

    async def wipe(
        self, scope: str, *, grace_ms: int = _jobs.KILL_GRACE_MS
    ) -> builtins.list[Survivor]:
        return await adrive(_jobs.wipe(scope, grace_ms), self._t)
