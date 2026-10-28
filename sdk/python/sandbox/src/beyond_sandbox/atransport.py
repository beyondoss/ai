"""Layer 1 for asyncio: the same calls as :mod:`beyond_sandbox.transport`, awaited.

``AsyncHttpTransport`` is a small HTTP/1.1 client over ``asyncio.open_connection``; it shares all
framing and protocol code with the sync transport and differs only in how bytes move.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import time
from collections import deque
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from . import _http, _wire
from .errors import ProtocolError, SandboxError, Timeout, TooLarge, TransportError
from .models import ExecResult, Frame, JobFrame
from .transport import DEFAULT_TIMEOUT_MS

AsyncToken = Union[str, Callable[[], Union[str, Awaitable[str]]], None]


class AsyncFrameStream(Protocol):
    """An async iterator of stream frames. ``aclose()`` ends it early; closing kills the command."""

    def __aiter__(self) -> AsyncIterator[Frame]: ...
    async def __anext__(self) -> Frame: ...
    async def aclose(self) -> None: ...


class AsyncJobFrameStream(Protocol):
    def __aiter__(self) -> AsyncIterator[JobFrame]: ...
    async def __anext__(self) -> JobFrame: ...
    async def aclose(self) -> None: ...


class AsyncTransport(Protocol):
    """The whole of layer 1 a helper needs, awaited."""

    async def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        stdin: bytes | None = None,
    ) -> ExecResult: ...

    async def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> AsyncFrameStream:
        """As :meth:`Transport.stream`."""
        ...


@runtime_checkable
class AsyncJobTransport(AsyncTransport, Protocol):
    """A transport that also keeps jobs in the sandbox."""

    async def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> Any: ...

    async def job_stream(self, job: dict[str, Any]) -> AsyncJobFrameStream: ...


class _Response:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        parsed: _http.ResponseReader,
        deadline: float,
        url: str,
    ) -> None:
        self._r, self._w, self._p, self._deadline, self._url = reader, writer, parsed, deadline, url

    @property
    def status(self) -> int:
        return self._p.status

    @property
    def headers(self) -> dict[str, str]:
        return self._p.headers

    async def fill(self, silence: float | None) -> None:
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise Timeout(f"exec endpoint {self._url} timed out")
        wait = left if silence is None else min(left, silence)
        try:
            self._p.feed(await asyncio.wait_for(self._r.read(65536), wait))
        except asyncio.TimeoutError:
            if time.monotonic() >= self._deadline:
                raise Timeout(f"exec endpoint {self._url} timed out") from None
            raise TransportError(
                f"no frame for {silence:.0f} s: the endpoint stopped sending heartbeats, so the "
                "connection is presumed dead"
            ) from None
        except OSError as e:
            raise TransportError(f"exec endpoint {self._url}: the connection broke: {e}") from None
        except _http.Truncated as e:
            raise TransportError(f"exec endpoint {self._url}: {e}") from None
        except _http.Malformed as e:
            raise ProtocolError(f"exec endpoint {self._url}: {e}") from None

    async def read_capped(self, cap: int) -> bytes:
        declared = self._p.content_length
        if declared is not None and declared > cap:
            await self.close()
            raise _wire.too_large(self._url, cap)
        body = bytearray()
        while True:
            body += self._p.take()
            if len(body) > cap:
                await self.close()
                raise _wire.too_large(self._url, cap)
            if self._p.done:
                return bytes(body)
            await self.fill(None)

    async def next_chunk(self, silence: float) -> bytes:
        while True:
            data = self._p.take()
            if data:
                return data
            if self._p.done:
                return b""
            await self.fill(silence)

    async def close(self) -> None:
        self._w.close()
        with contextlib.suppress(Exception):
            await self._w.wait_closed()


class _Frames:
    def __init__(self, resp: _Response, decoder: _wire.StreamDecoder[Any]) -> None:
        self._resp, self._decoder = resp, decoder
        self._pending: deque[Any] = deque()
        self._closed = False

    def __aiter__(self) -> _Frames:
        return self

    async def __anext__(self) -> Any:
        while not self._pending:
            if self._closed or self._decoder.finished:
                await self.aclose()
                raise StopAsyncIteration
            try:
                data = await self._resp.next_chunk(_wire.SILENCE_S)
                if data:
                    self._pending.extend(self._decoder.feed(data))
                else:
                    self._decoder.eof()
            except SandboxError:
                await self.aclose()
                raise
        return self._pending.popleft()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._resp.close()

    async def __aenter__(self) -> _Frames:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class AsyncHttpTransport:
    """The exec protocol over HTTP, on asyncio alone. See :class:`~beyond_sandbox.HttpTransport`.

    ``token`` may be a string, or a function (plain or ``async``) returning one.
    """

    def __init__(
        self,
        url: str,
        token: AsyncToken = None,
        *,
        headers: dict[str, str] | None = None,
        max_response_bytes: int = _wire.DEFAULT_MAX_RESPONSE,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.url = url
        self._target = _http.parse_url(url)
        self._token = token
        self._token_value: str | None = token if isinstance(token, str) else None
        self._headers = list((headers or {}).items())
        self._cap = max_response_bytes
        self._ssl = ssl_context
        self._lock = asyncio.Lock()

    async def _bearer(self, refresh: bool) -> list[tuple[str, str]]:
        async with self._lock:
            if callable(self._token) and (refresh or self._token_value is None):
                try:
                    value = self._token()
                    if isinstance(value, Awaitable):
                        value = await value
                    self._token_value = _wire.checked_token(value)
                except SandboxError:
                    raise
                except Exception as e:
                    raise _wire.token_failed(e) from e
            value_ = self._token_value
        return [("Authorization", f"Bearer {value_}")] if value_ else []

    async def _open(self, body: bytes, timeout_ms: int, *, refresh: bool = False) -> _Response:
        deadline = time.monotonic() + timeout_ms / 1000 + _wire.GRACE_S
        request = _http.build_request(
            self._target, self._headers + await self._bearer(refresh), body
        )
        t = self._target
        try:
            ctx = (self._ssl or ssl.create_default_context()) if t.tls else None
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    t.host, t.port, ssl=ctx, server_hostname=t.host if t.tls else None
                ),
                max(deadline - time.monotonic(), 0.001),
            )
        except asyncio.TimeoutError:
            raise Timeout(f"exec endpoint {self.url} timed out") from None
        except OSError as e:
            raise TransportError(f"exec endpoint {self.url} unreachable: {e}") from None
        parsed = _http.ResponseReader()
        resp = _Response(reader, writer, parsed, deadline, self.url)
        try:
            writer.write(request)
            await asyncio.wait_for(writer.drain(), max(deadline - time.monotonic(), 0.001))
            while not parsed.head_done:
                await resp.fill(None)
        except asyncio.TimeoutError:
            await resp.close()
            raise Timeout(f"exec endpoint {self.url} timed out") from None
        except OSError as e:
            await resp.close()
            raise TransportError(f"exec endpoint {self.url} unreachable: {e}") from None
        except BaseException:
            await resp.close()
            raise
        if 200 <= parsed.status < 300:
            return resp
        try:
            err_body = await resp.read_capped(self._cap)
        except SandboxError:
            err_body = b""
        await resp.close()
        if parsed.status == 401 and callable(self._token) and not refresh:
            return await self._open(body, timeout_ms, refresh=True)
        raise _wire.status_error(self.url, parsed.status, err_body)

    async def _post(self, body: bytes, timeout_ms: int, cap: int | None = None) -> bytes:
        resp = await self._open(body, timeout_ms)
        try:
            return await resp.read_capped(self._cap if cap is None else cap)
        finally:
            await resp.close()

    async def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        stdin: bytes | None = None,
    ) -> ExecResult:
        """As :meth:`HttpTransport.run`: the command's bytes."""
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin)
        raw = await self._post(body, timeout_ms, _wire.body_cap(self._cap))
        return _wire.parse_envelope(self.url, raw, self._cap)

    async def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> AsyncFrameStream:
        ms = _wire.STREAM_DEADLINE_MS if timeout_ms is None else timeout_ms
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=ms, stream=True)
        resp = await self._open(body, ms)
        if not _http.is_ndjson(resp.headers):
            await resp.close()
            raise _wire.not_a_stream(self.url)
        return _Frames(resp, _wire.frame_decoder())

    async def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> Any:
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=30_000, job=job)
        try:
            return _wire.parse_json(self.url, await self._post(body, 30_000))
        except TooLarge:
            raise TooLarge("a job answer exceeded the response cap") from None

    async def job_stream(self, job: dict[str, Any]) -> AsyncJobFrameStream:
        body = _wire.request_body(
            "true", [], cwd=None, timeout_ms=_wire.STREAM_DEADLINE_MS, stream=True, job=job
        )
        resp = await self._open(body, _wire.STREAM_DEADLINE_MS)
        return _Frames(resp, _wire.job_decoder())

    async def aclose(self) -> None:
        """Nothing is held between requests; present so a transport can be used as a context."""

    async def __aenter__(self) -> AsyncHttpTransport:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
