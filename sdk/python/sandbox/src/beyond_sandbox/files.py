"""Layer 2: files, directories and search over any transport (FILES-OVER-EXEC.md section 5)."""

from __future__ import annotations

import asyncio
import builtins
from typing import TypedDict

from . import _fs
from ._drive import Op, adrive, drive
from .atransport import AsyncTransport
from .errors import InvalidRequestError
from .models import Capabilities, Entry, SearchResult, Stat
from .transport import Transport


class CapsOverride(TypedDict, total=False):
    """Probe results to override after probing, so one sandbox can walk every fallback rung."""

    find_printf: bool
    search_engine: str  # "rg" | "grep" | "busybox-grep"


def apply_override(caps: Capabilities, override: CapsOverride | None) -> Capabilities:
    if not override:
        return caps
    engine = override.get("search_engine")
    if engine is not None:
        if engine not in ("rg", "grep", "busybox-grep"):
            raise InvalidRequestError(f"unknown search engine {engine!r}")
        caps = _fs.with_engine(caps, engine)
    return _fs.capabilities(
        caps.rg, caps.grep_null, override.get("find_printf", caps.find_printf), caps.read_window
    )


def _count(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value < 0):
        raise InvalidRequestError(f"`{name}` must be a non-negative integer")


def _bytes(data: bytes | str) -> bytes:
    return data.encode("utf-8") if isinstance(data, str) else data


def _query(
    pattern: str,
    root: str,
    literal: bool,
    ignore_case: bool,
    glob: str | None,
    glob_exclude: bool,
    limit: int,
    before: int,
    after: int,
) -> _fs.Query:
    for name, value in (("limit", limit), ("before", before), ("after", after)):
        _count(name, value)
    return _fs.Query(pattern, root, literal, ignore_case, glob, glob_exclude, limit, before, after)


class Files:
    """Files over a :class:`~beyond_sandbox.Transport`: ``run`` is all it needs.

    Probes the sandbox environment once, in one round trip on first need, and picks the best rung of
    each fallback ladder (writes need no probe). A probe that fails at the transport level raises
    from the operation that needed it and is not cached.
    """

    def __init__(self, transport: Transport, *, override: CapsOverride | None = None) -> None:
        self._t = transport
        self._override = override
        self._state = _fs.FsState()

    def capabilities(self) -> Capabilities:
        if self._state.caps is None:
            self._state.caps = apply_override(drive(_fs.probe(), self._t), self._override)
        return self._state.caps

    def stat(self, path: str) -> Stat | None:
        """``None`` if the path does not exist (including a missing parent or a broken link)."""
        return drive(_fs.stat(path), self._t)

    def read(self, path: str, offset: int = 0, max: int | None = None) -> bytes:  # noqa: A002
        _count("offset", offset)
        _count("max", max)
        return drive(_fs.read(path, offset, max, self.capabilities()), self._t)

    def read_text(self, path: str, offset: int = 0, max: int | None = None) -> str:  # noqa: A002
        """:meth:`read` decoded as UTF-8 (invalid sequences replaced, same as ``stdout_text``)."""
        return self.read(path, offset, max).decode("utf-8", "replace")

    def write(self, path: str, data: bytes | str) -> None:
        """Replace the file's content atomically, keeping its mode and writing through symlinks.

        ``data`` may be ``str`` (encoded as UTF-8) or ``bytes``.
        """
        drive(_fs.write(path, _bytes(data)), self._t)

    def write_if_unchanged(
        self, path: str, data: bytes | str, expected_mtime: float | None
    ) -> None:
        """Write unless the file's mtime (epoch seconds, with the fraction) differs; ``None`` means "must not exist".

        Raises :class:`~beyond_sandbox.ConflictError` on a mismatch. Two round trips: not a lock.
        ``data`` may be ``str`` (UTF-8) or ``bytes``.
        """
        drive(_fs.write_if_unchanged(path, _bytes(data), expected_mtime), self._t)

    def mkdir(self, path: str) -> None:
        """``mkdir -p``: idempotent."""
        drive(_fs.mkdir(path), self._t)

    def list(
        self, path: str, *, include_hidden: bool = False, limit: int = _fs.DEFAULT_LIST_LIMIT
    ) -> builtins.list[Entry]:
        _count("limit", limit)
        return drive(_fs.list_dir(path, limit, include_hidden, self.capabilities()), self._t)

    def search(
        self,
        pattern: str,
        root: str,
        *,
        literal: bool = False,
        ignore_case: bool = False,
        glob: str | None = None,
        glob_exclude: bool = False,
        limit: int = _fs.DEFAULT_SEARCH_LIMIT,
        before: int = 0,
        after: int = 0,
    ) -> SearchResult:
        q = _query(pattern, root, literal, ignore_case, glob, glob_exclude, limit, before, after)
        return drive(_fs.search(q, self._state, self.capabilities()), self._t)


class AsyncFiles:
    """:class:`Files` for an :class:`~beyond_sandbox.AsyncTransport`."""

    def __init__(self, transport: AsyncTransport, *, override: CapsOverride | None = None) -> None:
        self._t = transport
        self._override = override
        self._state = _fs.FsState()
        self._lock = asyncio.Lock()

    async def capabilities(self) -> Capabilities:
        async with self._lock:
            if self._state.caps is None:
                probed: Op[Capabilities] = _fs.probe()
                self._state.caps = apply_override(await adrive(probed, self._t), self._override)
            return self._state.caps

    async def stat(self, path: str) -> Stat | None:
        return await adrive(_fs.stat(path), self._t)

    async def read(self, path: str, offset: int = 0, max: int | None = None) -> bytes:  # noqa: A002
        _count("offset", offset)
        _count("max", max)
        return await adrive(_fs.read(path, offset, max, await self.capabilities()), self._t)

    async def read_text(self, path: str, offset: int = 0, max: int | None = None) -> str:  # noqa: A002
        """:meth:`read` decoded as UTF-8 (invalid sequences replaced, same as ``stdout_text``)."""
        return (await self.read(path, offset, max)).decode("utf-8", "replace")

    async def write(self, path: str, data: bytes | str) -> None:
        await adrive(_fs.write(path, _bytes(data)), self._t)

    async def write_if_unchanged(
        self, path: str, data: bytes | str, expected_mtime: float | None
    ) -> None:
        await adrive(_fs.write_if_unchanged(path, _bytes(data), expected_mtime), self._t)

    async def mkdir(self, path: str) -> None:
        await adrive(_fs.mkdir(path), self._t)

    async def list(
        self, path: str, *, include_hidden: bool = False, limit: int = _fs.DEFAULT_LIST_LIMIT
    ) -> builtins.list[Entry]:
        _count("limit", limit)
        op = _fs.list_dir(path, limit, include_hidden, await self.capabilities())
        return await adrive(op, self._t)

    async def search(
        self,
        pattern: str,
        root: str,
        *,
        literal: bool = False,
        ignore_case: bool = False,
        glob: str | None = None,
        glob_exclude: bool = False,
        limit: int = _fs.DEFAULT_SEARCH_LIMIT,
        before: int = 0,
        after: int = 0,
    ) -> SearchResult:
        q = _query(pattern, root, literal, ignore_case, glob, glob_exclude, limit, before, after)
        return await adrive(_fs.search(q, self._state, await self.capabilities()), self._t)
