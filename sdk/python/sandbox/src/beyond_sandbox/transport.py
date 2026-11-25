"""Layer 1: raw exec. The :class:`Transport` protocol and its HTTP implementation.

A transport is anything with ``run`` and ``stream``: an HTTP endpoint, an SSH session, a vendor SDK,
a test double. Everything above (files, jobs, the :class:`~beyond_sandbox.Sandbox`) is built on
these calls alone. Jobs need the ``job_*`` methods of :class:`JobTransport` on top.
"""

from __future__ import annotations

import builtins
import contextlib
import socket
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
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

DEFAULT_TIMEOUT_MS = 120_000

Token = Union[str, Callable[[], str], None]


class FrameStream(Protocol):
    """An iterator of stream frames. ``close()`` ends it early; closing kills the command."""

    def __iter__(self) -> Iterator[Frame]: ...
    def __next__(self) -> Frame: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class JobOpened:
    """The answer to one request of a job handle, whatever its status.

    A refusal is an answer, never an exception: ``status`` is what the endpoint said, and the job handle
    (the client kernel) decides what it means. ``lines`` are the stream's lines, each without its newline,
    when the status is a success: iterating them raises :class:`~beyond_sandbox.TransportError` or
    :class:`~beyond_sandbox.TimeoutError` when the stream is cut or silent for the limit asked for, and
    :class:`~beyond_sandbox.ProtocolError` for a line it cannot hold. A transport that has no answer
    (the endpoint was not reached) raises those from ``job_open`` itself.
    """

    status: int
    #: The server's ``Retry-After``, whole seconds; 0 when it sent none.
    retry_after_s: int = 0
    #: What a refusal said, for the error that reports it.
    detail: str = ""
    lines: Iterable[bytes] = ()
    #: Release the connection.
    close: Callable[[], None] = lambda: None


class Transport(Protocol):
    """The whole of layer 1 a helper needs."""

    def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        stdin: bytes | None = None,
    ) -> ExecResult:
        """Run one command to its end. A non-zero exit is a result; a transport failure raises."""
        ...

    def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> FrameStream:
        """Run a command and yield its frames as they arrive. Closing the iterator kills it."""
        ...


@runtime_checkable
class JobTransport(Transport, Protocol):
    """A transport that also keeps jobs in the sandbox."""

    def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> Any:
        """One job operation; its JSON answer. A caller that chose a heartbeat also passes
        ``heartbeat_seconds`` (the ``Keepalive-Ping-Interval`` header); a transport that has no use for it may omit the parameter, and is
        only ever passed it when it is set."""
        ...

    def job_open(self, job: dict[str, Any], silence_ms: int) -> JobOpened:
        """One request of a job handle (an ``attach``, or a ``start``): what the endpoint answered.

        A held start (``hold: true``) and an attach answer with a stream of lines; a start that is not held
        answers with its one JSON record as the only line. ``silence_ms`` is the longest the stream may go
        without a line before it is presumed dead. When set, the caller also passes ``command``, ``args``,
        ``cwd`` and ``env`` (a start runs a command) and ``heartbeat_seconds``; a transport is only ever
        passed what is set. It never retries and never raises for a refusal."""
        ...


# ---- the HTTP implementation -----------------------------------------------------------------------


class _Response:
    """One open HTTP response: its head is read; the body is read on demand under a deadline."""

    def __init__(
        self, sock: socket.socket, reader: _http.ResponseReader, deadline: float, url: str
    ) -> None:
        self._sock, self._reader, self._deadline, self._url = sock, reader, deadline, url

    @property
    def status(self) -> int:
        return self._reader.status

    @property
    def headers(self) -> dict[str, str]:
        return self._reader.headers

    def _fill(self, silence: float | None) -> None:
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"exec endpoint {self._url} timed out")
        wait = left if silence is None else min(left, silence)
        try:
            self._sock.settimeout(_wire.socket_s(wait))
            data = self._sock.recv(65536)
            self._reader.feed(data)
        except builtins.TimeoutError:
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

    def read_capped(self, cap: int) -> bytes:
        declared = self._reader.content_length
        if declared is not None and declared > cap:
            self.close()
            raise _wire.too_large(self._url, cap)
        body = bytearray()
        while True:
            self._reader.take_into(body)
            if len(body) > cap:
                self.close()
                raise _wire.too_large(self._url, cap)
            if self._reader.done:
                return bytes(body)
            self._fill(None)

    def next_chunk(self, silence: float) -> bytes:
        """The next piece of body, or ``b""`` when the body has ended."""
        while True:
            data = self._reader.take()
            if data:
                return data
            if self._reader.done:
                return b""
            self._fill(silence)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


class _Frames:
    """Items of an NDJSON stream (command frames, or a job stream's). ``close()`` drops the connection."""

    def __init__(
        self,
        resp: _Response,
        decoder: _wire.StreamDecoder[Any],
        silence: float = _wire.SILENCE_S,
    ) -> None:
        self._resp, self._decoder, self._silence = resp, decoder, silence
        self._pending: deque[Any] = deque()
        self._closed = False

    def __iter__(self) -> _Frames:
        return self

    def __next__(self) -> Any:
        while not self._pending:
            if self._closed or self._decoder.finished:
                self.close()
                raise StopIteration
            try:
                data = self._resp.next_chunk(self._silence)
                if data:
                    self._pending.extend(self._decoder.feed(data))
                else:
                    self._decoder.eof()
            except SandboxError:
                self.close()
                raise
        return self._pending.popleft()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._resp.close()

    def __enter__(self) -> _Frames:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class _Lines:
    """The lines of a job stream, each without its newline. A body that ends is the end of the iteration:
    whether that is a clean end or a cut is the kernel's to say."""

    def __init__(self, resp: _Response, silence: float) -> None:
        self._resp, self._silence = resp, silence
        self._split = _wire.LineSplitter()
        self._pending: deque[bytes] = deque()

    def __iter__(self) -> _Lines:
        return self

    def __next__(self) -> bytes:
        while not self._pending:
            try:
                data = self._resp.next_chunk(self._silence)
            except SandboxError:
                self._resp.close()
                raise
            if not data:
                self._resp.close()
                raise StopIteration
            self._pending.extend(self._split.feed(data))
        return self._pending.popleft()


@dataclass(frozen=True, slots=True)
class _Refusal:
    """A non-2xx answer, read to its end."""

    status: int
    retry_after_s: int
    body: bytes


class HttpTransport:
    """The exec protocol over HTTP, on the standard library alone.

    ``token`` is a bearer token, or a callable returning one: it is called on first use, and once
    more if the endpoint answers 401, after which that one request is sent again. That is the only
    retry; every other failure is reported as it happened. One connection per request.
    """

    def __init__(
        self,
        url: str,
        token: Token = None,
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
        self._lock = threading.Lock()

    # ---- the wire ---------------------------------------------------------------------------

    def _tls_context(self) -> ssl.SSLContext:
        """The SSL context for HTTPS: the one passed in, or a cached ``create_default_context``."""
        ctx = self._ssl
        if ctx is None:
            ctx = ssl.create_default_context()
            self._ssl = ctx
        return ctx

    def _bearer(self, refresh: bool) -> list[tuple[str, str]]:
        with self._lock:
            if callable(self._token) and (refresh or self._token_value is None):
                try:
                    self._token_value = _wire.checked_token(self._token())
                except SandboxError:
                    raise
                except Exception as e:
                    raise _wire.token_failed(e) from e
            value = self._token_value
        return [("Authorization", f"Bearer {value}")] if value else []

    def _connect(self, deadline: float) -> socket.socket:
        t = self._target
        try:
            sock = socket.create_connection(
                (t.host, t.port), timeout=_wire.socket_s(deadline - time.monotonic())
            )
            if t.tls:
                sock = self._tls_context().wrap_socket(sock, server_hostname=t.host)
            return sock
        except builtins.TimeoutError:
            raise TimeoutError(f"exec endpoint {self.url} timed out") from None
        except OSError as e:
            raise TransportError(f"exec endpoint {self.url} unreachable: {e}") from None

    def _open(
        self,
        body: bytes,
        timeout_ms: int,
        *,
        refresh: bool = False,
        heartbeat_seconds: int | None = None,
    ) -> _Response:
        """POST ``body``; return the response once its head is read. A non-2xx answer raises."""
        got = self._open_raw(body, timeout_ms, refresh=refresh, heartbeat_seconds=heartbeat_seconds)
        if isinstance(got, _Refusal):
            raise _wire.status_error(self.url, got.status, got.body)
        return got

    def _open_raw(
        self,
        body: bytes,
        timeout_ms: int,
        *,
        refresh: bool = False,
        heartbeat_seconds: int | None = None,
    ) -> _Response | _Refusal:
        """POST ``body``; the response once its head is read, or the refusal a non-2xx answer is."""
        deadline = time.monotonic() + timeout_ms / 1000 + _wire.GRACE_S
        beat = _wire.check_heartbeat(heartbeat_seconds)
        extra = [] if beat is None else [(_wire.HEARTBEAT_HEADER, str(beat))]
        request = _http.build_request(
            self._target, self._headers + self._bearer(refresh) + extra, body
        )
        sock = self._connect(deadline)
        reader = _http.ResponseReader()
        resp = _Response(sock, reader, deadline, self.url)
        try:
            sock.settimeout(_wire.socket_s(deadline - time.monotonic()))
            sock.sendall(request)
            while not reader.head_done:
                resp._fill(None)
        except builtins.TimeoutError:
            resp.close()
            raise TimeoutError(f"exec endpoint {self.url} timed out") from None
        except OSError as e:
            resp.close()
            raise TransportError(f"exec endpoint {self.url} unreachable: {e}") from None
        except SandboxError:
            resp.close()
            raise
        if 200 <= reader.status < 300:
            return resp
        try:
            err_body = resp.read_capped(self._cap)
        except SandboxError:
            err_body = b""
        resp.close()
        if reader.status == 401 and callable(self._token) and not refresh:
            return self._open_raw(
                body, timeout_ms, refresh=True, heartbeat_seconds=heartbeat_seconds
            )
        return _Refusal(reader.status, _wire.retry_after_s(reader.headers), err_body)

    def _post(
        self,
        body: bytes,
        timeout_ms: int,
        cap: int | None = None,
        heartbeat_seconds: int | None = None,
    ) -> bytes:
        resp = self._open(body, timeout_ms, heartbeat_seconds=heartbeat_seconds)
        try:
            return resp.read_capped(self._cap if cap is None else cap)
        finally:
            resp.close()

    # ---- run ---------------------------------------------------------------------------------

    def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run one command; its output is the **bytes** the command wrote."""
        if env:
            self._prove_env()
        body = _wire.request_body(
            command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin, env=env
        )
        raw = self._post(body, timeout_ms, _wire.body_cap(self._cap))
        return _wire.parse_envelope(self.url, raw, self._cap)

    def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> FrameStream:
        ms = _wire.STREAM_DEADLINE_MS if timeout_ms is None else timeout_ms
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=ms, stream=True)
        resp = self._open(body, ms)
        if not _http.is_ndjson(resp.headers):
            resp.close()
            raise _wire.not_a_stream(self.url)
        return _Frames(resp, _wire.frame_decoder())

    # ---- jobs --------------------------------------------------------------------------------

    def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        heartbeat_seconds: int | None = None,
    ) -> Any:
        if env:
            self._prove_env()
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=30_000, job=job, env=env)
        try:
            return _wire.parse_json(
                self.url, self._post(body, 30_000, heartbeat_seconds=heartbeat_seconds)
            )
        except TooLargeError:
            raise TooLargeError("a job answer exceeded the response cap") from None

    def job_open(
        self,
        job: dict[str, Any],
        silence_ms: int,
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        heartbeat_seconds: int | None = None,
    ) -> JobOpened:
        if env:
            self._prove_env()
        kept_start = job.get("op") == "start" and not job.get("hold")
        ms = 30_000 if kept_start else _wire.STREAM_DEADLINE_MS
        body = _wire.request_body(
            command, args, cwd=cwd, timeout_ms=ms, stream=not kept_start, job=job, env=env
        )
        got = self._open_raw(body, ms, heartbeat_seconds=heartbeat_seconds)
        if isinstance(got, _Refusal):
            return JobOpened(got.status, got.retry_after_s, _wire.excerpt(got.body))
        if kept_start:
            # A kept job's start is one JSON answer, the same record a held start's first frame is.
            try:
                answer = got.read_capped(self._cap)
            finally:
                got.close()
            return JobOpened(got.status, lines=(answer,))
        if not _http.is_ndjson(got.headers):
            got.close()
            raise _wire.not_a_stream(self.url)
        return JobOpened(got.status, lines=_Lines(got, silence_ms / 1000), close=got.close)

    _env_proved = False

    def _prove_env(self) -> None:
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
            self.url, self._post(body, 10_000, _wire.body_cap(self._cap)), self._cap
        )
        if r.exit_code != 0 or r.stdout != b"1":
            raise NotOfferedError(
                f"exec endpoint {self.url} predates per-command env: it ran the command without "
                "the environment it was given"
            )
        self._env_proved = True

    def close(self) -> None:
        """Nothing is held between requests; present so a transport can be used as a context."""

    def __enter__(self) -> HttpTransport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
