"""Layer 2: sandbox jobs over any transport.

A job is a command the *sandbox* keeps: it runs until it exits or is killed, whichever connection
started it, and its output can be read from any offset.
"""

from __future__ import annotations

import builtins
from typing import Any, AsyncIterator, Iterator, Sequence

from . import _jobs
from ._drive import _DONE, adrive, adrive_emit, aiter_events, drive, drive_iter
from .atransport import AsyncJobTransport
from .models import Attached, JobEnd, JobGap, JobInfo, JobOutput, JobRead, JobShellExit, Survivor
from .transport import JobTransport

READ_MAX = _jobs.READ_MAX


def _collect(a: Attached, ev: Any) -> None:
    if isinstance(ev, JobOutput):
        a.output += ev.data
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
    ) -> int:
        """Start a job the sandbox keeps from the start; returns its process group."""
        return drive(_jobs.start(scope, job, command, args, cwd, label), self._t)

    def list(self, scope: str) -> builtins.list[JobInfo]:
        return drive(_jobs.list_jobs(scope), self._t)

    def read(self, scope: str, job: str, offset: int = 0, max: int = READ_MAX) -> JobRead:  # noqa: A002
        """Up to ``max`` bytes of output from ``offset``. An unknown job is ``gone``, not an error."""
        return drive(_jobs.read(scope, job, offset, max), self._t)

    def follow(self, scope: str, job: str, offset: int = 0) -> Iterator[Any]:
        """Events (:class:`JobOutput`, :class:`JobGap`, :class:`JobShellExit`), ending with a
        :class:`JobEnd` once the job's group is empty. Stopping early leaves the job running."""
        op = _jobs.follow(scope, job, offset)
        for item in drive_iter(op, self._t):
            if isinstance(item, tuple) and item and item[0] is _DONE:
                yield JobEnd(item[1])
            else:
                yield item

    def attach(self, scope: str, job: str, offset: int = 0) -> Attached:
        """Follow the job to its end and collect everything."""
        result = Attached(-1)
        for ev in self.follow(scope, job, offset):
            if isinstance(ev, JobEnd):
                result.exit_code = ev.code
            else:
                _collect(result, ev)
        return result

    def kill(
        self, scope: str, jobs: Sequence[str], *, remove: bool = False
    ) -> builtins.list[Survivor]:
        """SIGTERM, a grace, SIGKILL to each job's group; the groups that survived."""
        return drive(_jobs.kill(scope, jobs, remove), self._t)

    def remove(self, scope: str, jobs: Sequence[str]) -> None:
        """Delete ended jobs; never a running one."""
        drive(_jobs.remove(scope, jobs), self._t)

    def wipe(self, scope: str) -> builtins.list[str]:
        """Kill and delete every job of the scope."""
        return drive(_jobs.wipe(scope), self._t)


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
    ) -> int:
        op = _jobs.start(scope, job, command, args, cwd, label)
        return await adrive(op, self._t)

    async def list(self, scope: str) -> builtins.list[JobInfo]:
        return await adrive(_jobs.list_jobs(scope), self._t)

    async def read(self, scope: str, job: str, offset: int = 0, max: int = READ_MAX) -> JobRead:  # noqa: A002
        return await adrive(_jobs.read(scope, job, offset, max), self._t)

    async def follow(self, scope: str, job: str, offset: int = 0) -> AsyncIterator[Any]:
        """As :meth:`Jobs.follow`."""
        op = _jobs.follow(scope, job, offset)
        events = aiter_events(op, self._t)
        try:
            async for item in events:
                if isinstance(item, tuple) and item and item[0] is _DONE:
                    yield JobEnd(item[1])
                else:
                    yield item
        finally:
            await events.aclose()

    async def attach(self, scope: str, job: str, offset: int = 0) -> Attached:
        result = Attached(-1)
        op = _jobs.follow(scope, job, offset)

        async def collect(ev: Any) -> None:
            _collect(result, ev)

        result.exit_code = await adrive_emit(op, self._t, collect)
        return result

    async def kill(
        self, scope: str, jobs: Sequence[str], *, remove: bool = False
    ) -> builtins.list[Survivor]:
        return await adrive(_jobs.kill(scope, jobs, remove), self._t)

    async def remove(self, scope: str, jobs: Sequence[str]) -> None:
        await adrive(_jobs.remove(scope, jobs), self._t)

    async def wipe(self, scope: str) -> builtins.list[str]:
        return await adrive(_jobs.wipe(scope), self._t)
