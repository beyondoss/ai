"""Layer 3: a :class:`Sandbox` that gathers exec, files and jobs, and always exposes raw exec."""

from __future__ import annotations

import ssl
from typing import Sequence

from .atransport import AsyncFrameStream, AsyncHttpTransport, AsyncToken, AsyncTransport
from .files import AsyncFiles, CapsOverride, Files
from .jobs import AsyncJobs, Jobs
from .models import ExecResult, Frame, SandboxCapabilities
from .transport import DEFAULT_TIMEOUT_MS, FrameStream, HttpTransport, Token, Transport


class Frames:
    """The frames of a running command: iterate them; ``close()`` (or leaving ``with``) kills it."""

    def __init__(self, inner: FrameStream) -> None:
        self._inner = inner

    def __iter__(self) -> Frames:
        return self

    def __next__(self) -> Frame:
        return next(self._inner)

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> Frames:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncFrames:
    """:class:`Frames` for asyncio: ``async for``, ``async with``, ``aclose()``."""

    def __init__(self, inner: AsyncFrameStream) -> None:
        self._inner = inner

    def __aiter__(self) -> AsyncFrames:
        return self

    async def __anext__(self) -> Frame:
        return await self._inner.__anext__()

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> AsyncFrames:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class Sandbox:
    """One sandbox behind a :class:`Transport`.

    ``sandbox.exec`` and ``sandbox.stream`` are the raw calls; ``sandbox.files`` and
    ``sandbox.jobs`` are built on them. ``sandbox.transport`` is yours to use directly.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        jobs_root: str | None = None,
        override: CapsOverride | None = None,
    ) -> None:
        self.transport = transport
        self.files = Files(transport, override=override)
        self.jobs = Jobs(transport, **({"root": jobs_root} if jobs_root else {}))

    @classmethod
    def connect(
        cls,
        url: str,
        token: Token = None,
        *,
        headers: dict[str, str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        jobs_root: str | None = None,
        override: CapsOverride | None = None,
    ) -> Sandbox:
        """A sandbox behind an exec endpoint. Nothing is sent until the first call."""
        http = HttpTransport(url, token, headers=headers, ssl_context=ssl_context)
        return cls(http, jobs_root=jobs_root, override=override)

    def exec(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        stdin: bytes | None = None,
    ) -> ExecResult:
        """Run one command. A non-zero exit is a result; a transport failure raises."""
        return self.transport.run(command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin)

    def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> Frames:
        """Run a command and iterate its frames as they arrive (v1.2). ``close()`` kills it."""
        return Frames(self.transport.stream(command, args, cwd=cwd, timeout_ms=timeout_ms))

    def capabilities(self) -> SandboxCapabilities:
        caps = self.files.capabilities()
        probe = getattr(self.transport, "streams", None)
        return SandboxCapabilities(
            caps.stdin, bool(probe()) if probe else False, self.jobs.kind(),
            caps.search_engine, caps.find_printf, caps.read_window,
        )  # fmt: skip

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close:
            close()

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncSandbox:
    """:class:`Sandbox` for asyncio."""

    def __init__(
        self,
        transport: AsyncTransport,
        *,
        jobs_root: str | None = None,
        override: CapsOverride | None = None,
    ) -> None:
        self.transport = transport
        self.files = AsyncFiles(transport, override=override)
        self.jobs = AsyncJobs(transport, **({"root": jobs_root} if jobs_root else {}))

    @classmethod
    def connect(
        cls,
        url: str,
        token: AsyncToken = None,
        *,
        headers: dict[str, str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        jobs_root: str | None = None,
        override: CapsOverride | None = None,
    ) -> AsyncSandbox:
        http = AsyncHttpTransport(url, token, headers=headers, ssl_context=ssl_context)
        return cls(http, jobs_root=jobs_root, override=override)

    async def exec(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        stdin: bytes | None = None,
    ) -> ExecResult:
        return await self.transport.run(command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin)

    async def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> AsyncFrames:
        return AsyncFrames(
            await self.transport.stream(command, args, cwd=cwd, timeout_ms=timeout_ms)
        )

    async def capabilities(self) -> SandboxCapabilities:
        caps = await self.files.capabilities()
        probe = getattr(self.transport, "streams", None)
        return SandboxCapabilities(
            caps.stdin, bool(await probe()) if probe else False, await self.jobs.kind(),
            caps.search_engine, caps.find_printf, caps.read_window,
        )  # fmt: skip

    async def aclose(self) -> None:
        close = getattr(self.transport, "aclose", None)
        if close:
            await close()

    async def __aenter__(self) -> AsyncSandbox:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
