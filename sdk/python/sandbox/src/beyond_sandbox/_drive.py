"""Sans-IO plumbing: helpers are generators that *ask* for commands; a driver performs them.

A helper (``_fs``, ``_jobs``) is written once as a generator that yields :class:`Run` and receives
the :class:`~beyond_sandbox.ExecResult` back. The sync driver answers with ``transport.run``, the
async one awaits it. So the control flow of every helper exists exactly once, and only the I/O
differs. Jobs add :class:`Call` (one job operation, one JSON answer) and the handle's four effects:
:class:`Open` (send a request, learn its status), :class:`NextLine` (the next line of the open stream),
:class:`Sleep` and :class:`Emit` (an event for the caller of ``follow``). What to send, read and wait for
is the kernel's call (``_held.py``); a driver only does it.

A transport error is thrown into the generator at the ``yield``, so cleanup (``except``/``finally``)
works as it would in straight-line code.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Sequence,
    TypeVar,
    Union,
)

from . import _kernel
from .atransport import AsyncJobOpened
from .errors import SandboxError
from .models import ExecResult, Exit, Signal, Stderr, Stdout, TimedOut
from .transport import JobOpened

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Run:
    command: str
    args: Sequence[str] = ()
    stdin: bytes | None = None
    timeout_ms: int = 120_000


@dataclass(frozen=True, slots=True)
class Emit:
    """Something for the caller of a streaming helper (an event of a followed job)."""

    item: Any


@dataclass(frozen=True, slots=True)
class Call:
    """A job operation; answered with its parsed JSON."""

    job: dict[str, Any]
    command: str = "true"
    args: Sequence[str] = ()
    cwd: str | None = None
    env: dict[str, str] | None = None
    #: The request's own heartbeat, whole seconds 1 to 240 (``Keepalive-Ping-Interval``).
    heartbeat_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class Open:
    """One request of a job handle: a held ``start`` (which runs ``command``) or an ``attach``. Answered
    with what the endpoint said (a :class:`~beyond_sandbox.transport.JobOpened`): its status, never an
    error for a refusal. A driver keeps the one connection it opens; a later :class:`Open` replaces it."""

    job: dict[str, Any]
    silence_ms: int
    command: str = "true"
    args: Sequence[str] = ()
    cwd: str | None = None
    env: dict[str, str] | None = None
    heartbeat_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class NextLine:
    """The next line of the open stream, answered as bytes; ``None`` when the stream ended."""


@dataclass(frozen=True, slots=True)
class Release:
    """Close the open stream, if there is one. Closing a held start's stream detaches nothing it kept."""


@dataclass(frozen=True, slots=True)
class Sleep:
    """Wait until the kernel's time ``until_ms``."""

    until_ms: int
    #: The caller's deadline: a wait never runs past it.
    deadline_ms: int | None = None


@dataclass(slots=True)
class Pipe:
    """A streamed command whose stdout goes to ``on_stdout`` as it arrives (never held whole).

    Answered with an :class:`~beyond_sandbox.ExecResult` whose ``stdout`` is empty and whose
    ``stderr`` is the first ``stderr_limit`` bytes of what the command wrote there. An exception
    from ``on_stdout`` closes the stream (which kills the command) and propagates.
    """

    command: str
    args: Sequence[str]
    on_stdout: Callable[[bytes], None]
    timeout_ms: int
    stderr_limit: int


class _PipeAcc:
    """The frames of one :class:`Pipe`, folded: stdout out, stderr kept, the final status."""

    def __init__(self, effect: Pipe) -> None:
        self.effect = effect
        self.stderr = bytearray()
        self.code: int | None = None

    def take(self, frame: Any) -> None:
        if isinstance(frame, Stdout):
            self.effect.on_stdout(frame.data)
        elif isinstance(frame, Stderr):
            room = self.effect.stderr_limit - len(self.stderr)
            if room > 0:
                self.stderr += frame.data[:room]
        elif isinstance(frame, Exit):
            self.code = frame.code
        elif isinstance(frame, Signal):
            self.code = 128 + frame.signal
        elif isinstance(frame, TimedOut):
            self.code = 124
            self.stderr += b"\ncommand exceeded its timeout"

    def result(self) -> ExecResult:
        return ExecResult(self.code if self.code is not None else -1, b"", bytes(self.stderr))


Effect = Union[Run, Emit, Call, Open, NextLine, Release, Sleep, Pipe]
Op = Generator[Effect, Any, T]

#: Marks the end of a driven op's items; carries its result.
_DONE = object()
#: What an :class:`Emit` effect is answered with.
_NOTHING = ExecResult(0, b"", b"")


def _call_extras(effect: Call) -> dict[str, Any]:
    """What a ``Call`` passes beyond ``job, command, args, cwd``: only what is set, so a transport
    written before ``env`` or a heartbeat existed keeps working."""
    extras: dict[str, Any] = {}
    if effect.env:
        extras["env"] = effect.env
    if effect.heartbeat_seconds is not None:
        extras["heartbeat_seconds"] = effect.heartbeat_seconds
    return extras


def _open_extras(effect: Open) -> dict[str, Any]:
    """What an ``Open`` passes beyond its ``job`` and silence limit: only what is set."""
    extras: dict[str, Any] = {}
    if effect.command != "true":
        extras["command"] = effect.command
    if effect.args:
        extras["args"] = effect.args
    if effect.cwd is not None:
        extras["cwd"] = effect.cwd
    if effect.env:
        extras["env"] = effect.env
    if effect.heartbeat_seconds is not None:
        extras["heartbeat_seconds"] = effect.heartbeat_seconds
    return extras


def drive_iter(op: Op[T], transport: Any) -> Generator[Any, None, None]:
    """Run ``op`` on a sync transport, yielding each emitted item; the last is ``(_DONE, result)``."""
    conn: JobOpened | None = None
    lines: Iterator[bytes] = iter(())
    try:
        effect = next(op)
        while True:
            try:
                if isinstance(effect, Run):
                    answer: Any = transport.run(
                        effect.command,
                        effect.args,
                        timeout_ms=effect.timeout_ms,
                        stdin=effect.stdin,
                    )
                elif isinstance(effect, Emit):
                    yield effect.item
                    answer = _NOTHING
                elif isinstance(effect, Call):
                    answer = transport.job_call(
                        effect.job, effect.command, effect.args, effect.cwd, **_call_extras(effect)
                    )
                elif isinstance(effect, Pipe):
                    frames = transport.stream(
                        effect.command, effect.args, timeout_ms=effect.timeout_ms
                    )
                    acc = _PipeAcc(effect)
                    try:
                        for frame in frames:
                            acc.take(frame)
                    finally:
                        frames.close()
                    answer = acc.result()
                elif isinstance(effect, Open):
                    if conn is not None:
                        conn.close()
                        conn = None
                    conn = transport.job_open(effect.job, effect.silence_ms, **_open_extras(effect))
                    lines = iter(conn.lines)
                    answer = conn
                elif isinstance(effect, NextLine):
                    answer = next(lines, None)
                elif isinstance(effect, Release):
                    if conn is not None:
                        conn.close()
                        conn = None
                    answer = _NOTHING
                elif isinstance(effect, Sleep):
                    time.sleep(_kernel.wait_s(effect.until_ms, effect.deadline_ms))
                    answer = _NOTHING
                else:
                    raise AssertionError(f"unknown effect {effect!r}")
            except SandboxError as e:
                effect = op.throw(e)
                continue
            effect = op.send(answer)
    except StopIteration as stop:
        yield _DONE, stop.value
    finally:
        op.close()
        if conn is not None:
            conn.close()


def iter_items(op: Op[Any], transport: Any) -> Generator[Any, None, None]:
    """Run ``op`` on a sync transport, yielding each emitted item; its result is dropped."""
    items = drive_iter(op, transport)
    try:
        for item in items:
            if not (isinstance(item, tuple) and item and item[0] is _DONE):
                yield item
    finally:
        items.close()


def drive(op: Op[T], transport: Any) -> T:
    for item in drive_iter(op, transport):
        if isinstance(item, tuple) and item and item[0] is _DONE:
            return item[1]  # type: ignore[no-any-return]
    raise AssertionError("unreachable: drive_iter always ends with its result")


async def adrive_emit(
    op: Op[T], transport: Any, emit: Callable[[Any], Awaitable[None]] | None = None
) -> T:
    """Run ``op`` on an async transport; each emitted item is awaited into ``emit``."""
    conn: AsyncJobOpened | None = None
    lines: AsyncIterator[bytes] | None = None
    try:
        effect = next(op)
        while True:
            try:
                if isinstance(effect, Run):
                    answer: Any = await transport.run(
                        effect.command,
                        effect.args,
                        timeout_ms=effect.timeout_ms,
                        stdin=effect.stdin,
                    )
                elif isinstance(effect, Emit):
                    if emit is not None:
                        await emit(effect.item)
                    answer = _NOTHING
                elif isinstance(effect, Call):
                    answer = await transport.job_call(
                        effect.job, effect.command, effect.args, effect.cwd, **_call_extras(effect)
                    )
                elif isinstance(effect, Pipe):
                    aframes = await transport.stream(
                        effect.command, effect.args, timeout_ms=effect.timeout_ms
                    )
                    acc = _PipeAcc(effect)
                    try:
                        async for frame in aframes:
                            acc.take(frame)
                    finally:
                        await aframes.aclose()
                    answer = acc.result()
                elif isinstance(effect, Open):
                    if conn is not None:
                        await conn.close()
                        conn = None
                    conn = await transport.job_open(
                        effect.job, effect.silence_ms, **_open_extras(effect)
                    )
                    lines = conn.lines.__aiter__()
                    answer = conn
                elif isinstance(effect, NextLine):
                    answer = None if lines is None else await anext_or_none(lines)
                elif isinstance(effect, Release):
                    if conn is not None:
                        await conn.close()
                        conn = None
                    answer = _NOTHING
                elif isinstance(effect, Sleep):
                    await asyncio.sleep(_kernel.wait_s(effect.until_ms, effect.deadline_ms))
                    answer = _NOTHING
                else:
                    raise AssertionError(f"unknown effect {effect!r}")
            except SandboxError as e:
                effect = op.throw(e)
                continue
            effect = op.send(answer)
    except StopIteration as stop:
        return stop.value  # type: ignore[no-any-return]
    finally:
        op.close()
        if conn is not None:
            await conn.close()


async def anext_or_none(lines: AsyncIterator[bytes]) -> bytes | None:
    """The next item of an async iterator, or ``None`` at its end."""
    try:
        return await lines.__anext__()
    except StopAsyncIteration:
        return None


async def adrive(op: Op[T], transport: Any) -> T:
    return await adrive_emit(op, transport)


async def aiter_items(op: Op[Any], transport: Any) -> AsyncGenerator[Any, None]:
    """Run ``op`` on an async transport, yielding each emitted item; its result is dropped."""
    events = aiter_events(op, transport)
    try:
        async for item in events:
            if not (isinstance(item, tuple) and item and item[0] is _DONE):
                yield item
    finally:
        await events.aclose()


#: How many emitted items may wait for a slow consumer of :func:`aiter_events`.
EVENT_QUEUE = 16


async def aiter_events(op: Op[T], transport: Any) -> AsyncGenerator[Any, None]:
    """Run ``op`` and yield its emitted items as they happen; the last is ``(_DONE, result)``."""
    queue: asyncio.Queue[Any] = asyncio.Queue(EVENT_QUEUE)

    async def produce() -> T:
        try:
            return await adrive_emit(op, transport, queue.put)
        finally:
            await queue.put(_DONE)

    task = asyncio.ensure_future(produce())
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                yield _DONE, await task
                return
            yield item
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
