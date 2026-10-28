"""Sans-IO plumbing: helpers are generators that *ask* for commands; a driver performs them.

A helper (``_fs``, ``_jobs``) is written once as a generator that yields :class:`Run` and receives
the :class:`~beyond_sandbox.ExecResult` back. The sync driver answers with ``transport.run``, the
async one awaits it. So the control flow of every helper exists exactly once, and only the I/O
differs. Jobs add :class:`Call` and :class:`Follow` (job operations) and :class:`Emit` (an event for
the caller of ``follow``).

A transport error is thrown into the generator at the ``yield``, so cleanup (``except``/``finally``)
works as it would in straight-line code.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Sequence,
    TypeVar,
    Union,
)

from .errors import SandboxError
from .models import ExecResult, Exit, JobEnd, Signal, Stderr, Stdout, TimedOut

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


@dataclass(frozen=True, slots=True)
class Follow:
    """A native job stream (``attach``): its events are emitted; answered with the final status."""

    job: dict[str, Any]


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


Effect = Union[Run, Emit, Call, Follow, Pipe]
Op = Generator[Effect, Any, T]

#: Marks the end of a driven op's items; carries its result.
_DONE = object()
#: What an :class:`Emit` effect is answered with.
_NOTHING = ExecResult(0, b"", b"")


def drive_iter(op: Op[T], transport: Any) -> Iterator[Any]:
    """Run ``op`` on a sync transport, yielding each emitted item; the last is ``(_DONE, result)``."""
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
                    answer = transport.job_call(effect.job, effect.command, effect.args, effect.cwd)
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
                elif isinstance(effect, Follow):
                    stream = transport.job_stream(effect.job)
                    try:
                        answer = None
                        for frame in stream:
                            if isinstance(frame, JobEnd):
                                answer = frame
                                break
                            yield frame
                    finally:
                        stream.close()
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


def drive(op: Op[T], transport: Any) -> T:
    for item in drive_iter(op, transport):
        if isinstance(item, tuple) and item and item[0] is _DONE:
            return item[1]  # type: ignore[no-any-return]
    raise AssertionError("unreachable: drive_iter always ends with its result")


async def adrive_emit(
    op: Op[T], transport: Any, emit: Callable[[Any], Awaitable[None]] | None = None
) -> T:
    """Run ``op`` on an async transport; each emitted item is awaited into ``emit``."""
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
                        effect.job, effect.command, effect.args, effect.cwd
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
                elif isinstance(effect, Follow):
                    stream = await transport.job_stream(effect.job)
                    try:
                        answer = None
                        async for frame in stream:
                            if isinstance(frame, JobEnd):
                                answer = frame
                                break
                            if emit is not None:
                                await emit(frame)
                    finally:
                        await stream.aclose()
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


async def adrive(op: Op[T], transport: Any) -> T:
    return await adrive_emit(op, transport)


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
