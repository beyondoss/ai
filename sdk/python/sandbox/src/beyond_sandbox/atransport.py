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
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from . import _http, _wire
from .errors import (
    NotOfferedError,
    ProtocolError,
    SandboxError,
    TimeoutError,
    TooLargeError,
    TransportError,
)
from .models import ExecResult, Frame
from .transport import DEFAULT_TIMEOUT_MS


async def _nothing() -> None:
    return None


AsyncToken = Union[str, Callable[[], Union[str, Awaitable[str]]], None]


class AsyncFrameStream(Protocol):
    """An async iterator of stream frames. ``aclose()`` ends it early; closing kills the command."""

    def __aiter__(self) -> AsyncIterator[Frame]: ...
    async def __anext__(self) -> Frame: ...
    async def aclose(self) -> None: ...


@dataclass(slots=True)
class AsyncJobOpened:
    """:class:`~beyond_sandbox.transport.JobOpened`, awaited: the answer to one request of a job handle,
    whatever its status. ``lines`` raises as the sync one does; ``close`` is awaited."""

    status: int
    retry_after_s: int = 0
    detail: str = ""
    lines: AsyncIterable[bytes] = field(default_factory=lambda: _no_lines())
    close: Callable[[], Awaitable[None]] = lambda: _nothing()


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
        env: dict[str, str] | None = None,
    ) -> Any: ...

    async def job_open(self, job: dict[str, Any], silence_ms: int) -> AsyncJobOpened: ...

    # As :meth:`JobTransport.job_open`. A caller that chose a heartbeat also passes
    # ``heartbeat_seconds``; a start passes ``command``, ``args``, ``cwd`` and ``env`` too. A transport
    # is only ever passed what is set, so one without a parameter keeps working.


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
            raise TimeoutError(f"exec endpoint {self._url} timed out")
        wait = left if silence is None else min(left, silence)
        try:
            self._p.feed(await asyncio.wait_for(self._r.read(65536), wait))
        except asyncio.TimeoutError:
            if time.monotonic() >= self._deadline:
                raise TimeoutError(f"exec endpoint {self._url} timed out") from None
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
            self._p.take_into(body)
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
    def __init__(
        self,
        resp: _Response,
        decoder: _wire.StreamDecoder[Any],
        silence: float = _wire.SILENCE_S,
    ) -> None:
        self._resp, self._decoder, self._silence = resp, decoder, silence
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
                data = await self._resp.next_chunk(self._silence)
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


class _Lines:
    """The lines of a job stream, each without its newline; the end of the body is the end of the iteration."""

    def __init__(self, resp: _Response, silence: float) -> None:
        self._resp, self._silence = resp, silence
        self._split = _wire.LineSplitter()
        self._pending: deque[bytes] = deque()

    def __aiter__(self) -> _Lines:
        return self

    async def __anext__(self) -> bytes:
        while not self._pending:
            try:
                data = await self._resp.next_chunk(self._silence)
            except SandboxError:
                await self._resp.close()
                raise
            if not data:
                await self._resp.close()
                raise StopAsyncIteration
            self._pending.extend(self._split.feed(data))
        return self._pending.popleft()


@dataclass(frozen=True, slots=True)
class _Refusal:
    status: int
    retry_after_s: int
    body: bytes


async def _no_lines() -> AsyncIterator[bytes]:
    return
    yield b""


async def _one_line(line: bytes) -> AsyncIterator[bytes]:
    yield line


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

    def _tls_context(self) -> ssl.SSLContext:
        """The SSL context for HTTPS: the one passed in, or a cached ``create_default_context``."""
        ctx = self._ssl
        if ctx is None:
            ctx = ssl.create_default_context()
            self._ssl = ctx
        return ctx

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

    async def _open(
        self,
        body: bytes,
        timeout_ms: int,
        *,
        refresh: bool = False,
        heartbeat_seconds: int | None = None,
    ) -> _Response:
        got = await self._open_raw(
            body, timeout_ms, refresh=refresh, heartbeat_seconds=heartbeat_seconds
        )
        if isinstance(got, _Refusal):
            raise _wire.status_error(self.url, got.status, got.body)
        return got

    async def _open_raw(
        self,
        body: bytes,
        timeout_ms: int,
        *,
        refresh: bool = False,
        heartbeat_seconds: int | None = None,
    ) -> _Response | _Refusal:
        deadline = time.monotonic() + timeout_ms / 1000 + _wire.GRACE_S
        beat = _wire.check_heartbeat(heartbeat_seconds)
        extra = [] if beat is None else [(_wire.HEARTBEAT_HEADER, str(beat))]
        request = _http.build_request(
            self._target, self._headers + await self._bearer(refresh) + extra, body
        )
        t = self._target
        try:
            ctx = self._tls_context() if t.tls else None
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    t.host, t.port, ssl=ctx, server_hostname=t.host if t.tls else None
                ),
                max(deadline - time.monotonic(), 0.001),
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"exec endpoint {self.url} timed out") from None
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
            raise TimeoutError(f"exec endpoint {self.url} timed out") from None
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
            return await self._open_raw(
                body, timeout_ms, refresh=True, heartbeat_seconds=heartbeat_seconds
            )
        return _Refusal(parsed.status, _wire.retry_after_s(parsed.headers), err_body)

    async def _post(
        self,
        body: bytes,
        timeout_ms: int,
        cap: int | None = None,
        heartbeat_seconds: int | None = None,
    ) -> bytes:
        resp = await self._open(body, timeout_ms, heartbeat_seconds=heartbeat_seconds)
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
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """As :meth:`HttpTransport.run`: the command's bytes."""
        if env:
            await self._prove_env()
        body = _wire.request_body(
            command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin, env=env
        )
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
        env: dict[str, str] | None = None,
        heartbeat_seconds: int | None = None,
    ) -> Any:
        if env:
            await self._prove_env()
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=30_000, job=job, env=env)
        try:
            return _wire.parse_json(
                self.url, await self._post(body, 30_000, heartbeat_seconds=heartbeat_seconds)
            )
        except TooLargeError:
            raise TooLargeError("a job answer exceeded the response cap") from None

    async def job_open(
        self,
        job: dict[str, Any],
        silence_ms: int,
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        heartbeat_seconds: int | None = None,
    ) -> AsyncJobOpened:
        if env:
            await self._prove_env()
        kept_start = job.get("op") == "start" and not job.get("hold")
        ms = 30_000 if kept_start else _wire.STREAM_DEADLINE_MS
        body = _wire.request_body(
            command, args, cwd=cwd, timeout_ms=ms, stream=not kept_start, job=job, env=env
        )
        got = await self._open_raw(body, ms, heartbeat_seconds=heartbeat_seconds)
        if isinstance(got, _Refusal):
            return AsyncJobOpened(
                got.status, got.retry_after_s, _wire.excerpt(got.body), _no_lines()
            )
        if kept_start:
            # A kept job's start is one JSON answer, the same record a held start's first frame is.
            try:
                answer = await got.read_capped(self._cap)
            finally:
                await got.close()
            return AsyncJobOpened(got.status, lines=_one_line(answer))
        if not _http.is_ndjson(got.headers):
            await got.close()
            raise _wire.not_a_stream(self.url)
        return AsyncJobOpened(got.status, lines=_Lines(got, silence_ms / 1000), close=got.close)

    _env_proved = False

    async def _prove_env(self) -> None:
        """An endpoint that predates ``env`` ignores it silently: prove once that this one applies
        it (a command that prints a variable only ``env`` could have set), and fail closed."""
        if self._env_proved:
            return
        body = _wire.request_body(
            "sh",
            ["-c", 'printf %s "$BEYOND_ENV_PROBE"'],
            cwd=None,
            timeout_ms=10_000,
            env={"BEYOND_ENV_PROBE": "1"},
        )
        r = _wire.parse_envelope(
            self.url, await self._post(body, 10_000, _wire.body_cap(self._cap)), self._cap
        )
        if r.exit_code != 0 or r.stdout != b"1":
            raise NotOfferedError(
                f"exec endpoint {self.url} predates per-command env: it ran the command without "
                "the environment it was given"
            )
        self._env_proved = True

    async def aclose(self) -> None:
        """Nothing is held between requests; present so a transport can be used as a context."""

    async def __aenter__(self) -> AsyncHttpTransport:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
