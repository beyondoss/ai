"""Sans-IO plumbing: helpers are generators that *ask* for commands; a driver performs them.

A helper (``_fs``, ``_jobs``) is written once as a generator that yields :class:`Run` and receives
the :class:`~beyond_sandbox.ExecResult` back. The sync driver answers with ``transport.run``, the
async one awaits it. So the control flow of every fallback rung exists exactly once, and only the I/O
differs. Jobs add :class:`Call`, :class:`Follow` and :class:`Offer` (native v1.3 operations),
:class:`Sleep` (the baseline's poll) and :class:`Emit` (an event for the caller of ``follow``).

A transport error is thrown into the generator at the ``yield``, so cleanup (``except``/``finally``)
works as it would in straight-line code.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
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

from ._wire import JobOffer
from .errors import NotOffered, SandboxError
from .models import ExecResult, JobEnd

if TYPE_CHECKING:
    from .atransport import AsyncTransport
    from .transport import Transport

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Run:
    command: str
    args: Sequence[str] = ()
    stdin: bytes | None = None
    timeout_ms: int = 120_000


@dataclass(frozen=True, slots=True)
class Sleep:
    seconds: float


@dataclass(frozen=True, slots=True)
class Emit:
    """Something for the caller of a streaming helper (an event of a followed job)."""

    item: Any


@dataclass(frozen=True, slots=True)
class Call:
    """A native job operation (v1.3); answered with its parsed JSON."""

    job: dict[str, Any]
    command: str = "true"
    args: Sequence[str] = ()
    cwd: str | None = None


@dataclass(frozen=True, slots=True)
class Follow:
    """A native job stream (``attach``): its events are emitted; answered with the final status."""

    job: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Offer:
    """Ask the transport what it offers for jobs; answered with a :class:`~._wire.JobOffer`."""


Effect = Union[Run, Sleep, Emit, Call, Follow, Offer]
Op = Generator[Effect, Any, T]

#: Marks the end of a driven op's items; carries its result.
_DONE = object()
#: What a :class:`Sleep` or :class:`Emit` effect is answered with.
_NOTHING = ExecResult(0, b"", b"")


def _no_native(what: str) -> NotOffered:
    return NotOffered(f"this transport has no native {what}")


def drive_iter(op: Op[T], transport: Transport) -> Iterator[Any]:
    """Run ``op`` on a sync transport, yielding each emitted item; the last is ``(_DONE, result)``."""
    try:
        effect = next(op)
        while True:
            try:
                if isinstance(effect, Run):
                    answer: Any = getattr(transport, "run_text", transport.run)(
                        effect.command,
                        effect.args,
                        timeout_ms=effect.timeout_ms,
                        stdin=effect.stdin,
                    )
                elif isinstance(effect, Sleep):
                    time.sleep(effect.seconds)
                    answer = _NOTHING
                elif isinstance(effect, Emit):
                    yield effect.item
                    answer = _NOTHING
                elif isinstance(effect, Call):
                    call = getattr(transport, "job_call", None)
                    if call is None:
                        raise _no_native("jobs")
                    answer = call(effect.job, effect.command, effect.args, effect.cwd)
                elif isinstance(effect, Follow):
                    open_stream = getattr(transport, "job_stream", None)
                    if open_stream is None:
                        raise _no_native("jobs")
                    stream = open_stream(effect.job)
                    try:
                        answer = None
                        for frame in stream:
                            if isinstance(frame, JobEnd):
                                answer = frame.code
                                break
                            yield frame
                    finally:
                        stream.close()
                else:
                    offer = getattr(transport, "job_offer", None)
                    answer = offer() if offer is not None else JobOffer()
            except SandboxError as e:
                effect = op.throw(e)
                continue
            effect = op.send(answer)
    except StopIteration as stop:
        yield _DONE, stop.value
    finally:
        op.close()


def drive(op: Op[T], transport: Transport) -> T:
    for item in drive_iter(op, transport):
        if isinstance(item, tuple) and item and item[0] is _DONE:
            return item[1]  # type: ignore[no-any-return]
    raise AssertionError("unreachable: drive_iter always ends with its result")


async def adrive_emit(
    op: Op[T], transport: AsyncTransport, emit: Callable[[Any], Awaitable[None]] | None = None
) -> T:
    """Run ``op`` on an async transport; each emitted item is awaited into ``emit``."""
    try:
        effect = next(op)
        while True:
            try:
                if isinstance(effect, Run):
                    answer: Any = await getattr(transport, "run_text", transport.run)(
                        effect.command,
                        effect.args,
                        timeout_ms=effect.timeout_ms,
                        stdin=effect.stdin,
                    )
                elif isinstance(effect, Sleep):
                    await asyncio.sleep(effect.seconds)
                    answer = _NOTHING
                elif isinstance(effect, Emit):
                    if emit is not None:
                        await emit(effect.item)
                    answer = _NOTHING
                elif isinstance(effect, Call):
                    call = getattr(transport, "job_call", None)
                    if call is None:
                        raise _no_native("jobs")
                    answer = await call(effect.job, effect.command, effect.args, effect.cwd)
                elif isinstance(effect, Follow):
                    open_stream = getattr(transport, "job_stream", None)
                    if open_stream is None:
                        raise _no_native("jobs")
                    stream = await open_stream(effect.job)
                    try:
                        answer = None
                        async for frame in stream:
                            if isinstance(frame, JobEnd):
                                answer = frame.code
                                break
                            if emit is not None:
                                await emit(frame)
                    finally:
                        await stream.aclose()
                else:
                    offer = getattr(transport, "job_offer", None)
                    answer = await offer() if offer is not None else JobOffer()
            except SandboxError as e:
                effect = op.throw(e)
                continue
            effect = op.send(answer)
    except StopIteration as stop:
        return stop.value  # type: ignore[no-any-return]
    finally:
        op.close()


async def adrive(op: Op[T], transport: AsyncTransport) -> T:
    return await adrive_emit(op, transport)


#: How many emitted items may wait for a slow consumer of :func:`aiter_events`.
EVENT_QUEUE = 16


async def aiter_events(op: Op[T], transport: AsyncTransport) -> AsyncGenerator[Any, None]:
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
