"""Layer 3: a :class:`Sandbox` that gathers exec, files and jobs, and always exposes raw exec."""

from __future__ import annotations

import os
import ssl
from typing import Sequence

from . import _bulk
from ._drive import adrive, drive
from .atransport import AsyncFrameStream, AsyncHttpTransport, AsyncJobTransport, AsyncToken
from .files import AsyncFiles, CapsOverride, Files
from .jobs import AsyncJobs, Jobs
from .models import Capabilities, ExecResult, Frame, Transfer
from .transport import DEFAULT_TIMEOUT_MS, FrameStream, HttpTransport, JobTransport, Token


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
        transport: JobTransport,
        *,
        override: CapsOverride | None = None,
    ) -> None:
        self.transport = transport
        self.files = Files(transport, override=override)
        self.jobs = Jobs(transport)

    @classmethod
    def connect(
        cls,
        url: str,
        token: Token = None,
        *,
        headers: dict[str, str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        override: CapsOverride | None = None,
    ) -> Sandbox:
        """A sandbox behind an exec endpoint. Nothing is sent until the first call.

        A bad URL or token fails on :meth:`ready` or the first call that needs the endpoint
        (same promise as the TypeScript and Rust clients).
        """
        http = HttpTransport(url, token, headers=headers, ssl_context=ssl_context)
        return cls(http, override=override)

    def ready(self) -> Capabilities:
        """Probe the endpoint once (capabilities).

        Use when you want a bad URL or token to fail before the first real call; otherwise the
        first helper that needs a probe does the same work.
        """
        return self.capabilities()

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
        """Run a command and iterate its frames as they arrive. ``close()`` kills it."""
        return Frames(self.transport.stream(command, args, cwd=cwd, timeout_ms=timeout_ms))

    def upload(
        self,
        sources: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        remote_dir: str,
        *,
        chunk_bytes: int | None = None,
    ) -> Transfer:
        """Copy local files into the sandbox directory ``remote_dir`` as a tar stream.

        ``sources`` is a local directory (its contents land directly in ``remote_dir``) or a list
        of paths (each lands under its base name). Modes, symlinks and mtimes are kept; a link is
        never followed. One request when the archive fits ``chunk_bytes`` (default 16 MiB), else
        a staged upload in chunks.
        """
        return drive(_bulk.upload(sources, remote_dir, chunk_bytes), self.transport)

    def download(self, remote_path: str, local_dir: str | os.PathLike[str]) -> Transfer:
        """Copy a sandbox directory's contents (or one file) into ``local_dir`` (created if
        missing), streaming. Nothing is ever written through a link."""
        return drive(_bulk.download(remote_path, local_dir), self.transport)

    def capabilities(self) -> Capabilities:
        """What the sandbox's environment offers (probed once, cached)."""
        return self.files.capabilities()

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
        transport: AsyncJobTransport,
        *,
        override: CapsOverride | None = None,
    ) -> None:
        self.transport = transport
        self.files = AsyncFiles(transport, override=override)
        self.jobs = AsyncJobs(transport)

    @classmethod
    def connect(
        cls,
        url: str,
        token: AsyncToken = None,
        *,
        headers: dict[str, str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        override: CapsOverride | None = None,
    ) -> AsyncSandbox:
        """A sandbox behind an exec endpoint. Nothing is sent until the first call.

        A bad URL or token fails on :meth:`ready` or the first call that needs the endpoint.
        """
        http = AsyncHttpTransport(url, token, headers=headers, ssl_context=ssl_context)
        return cls(http, override=override)

    async def ready(self) -> Capabilities:
        """Probe the endpoint once (capabilities). See :meth:`Sandbox.ready`."""
        return await self.capabilities()

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

    async def upload(
        self,
        sources: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        remote_dir: str,
        *,
        chunk_bytes: int | None = None,
    ) -> Transfer:
        """Copy local files into the sandbox directory ``remote_dir`` as a tar stream.

        ``sources`` is a local directory (its contents land directly in ``remote_dir``) or a list
        of paths (each lands under its base name). Modes, symlinks and mtimes are kept; a link is
        never followed. One request when the archive fits ``chunk_bytes`` (default 16 MiB), else
        a staged upload in chunks.
        """
        return await adrive(_bulk.upload(sources, remote_dir, chunk_bytes), self.transport)

    async def download(self, remote_path: str, local_dir: str | os.PathLike[str]) -> Transfer:
        """Copy a sandbox directory's contents (or one file) into ``local_dir`` (created if
        missing), streaming. Nothing is ever written through a link."""
        return await adrive(_bulk.download(remote_path, local_dir), self.transport)

    async def capabilities(self) -> Capabilities:
        return await self.files.capabilities()

    async def aclose(self) -> None:
        close = getattr(self.transport, "aclose", None)
        if close:
            await close()

    async def __aenter__(self) -> AsyncSandbox:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
