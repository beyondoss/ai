"""Layer 1: raw exec. The :class:`Transport` protocol and its HTTP implementation.

A transport is anything with ``run`` and ``stream``: an HTTP endpoint, an SSH session, a vendor SDK,
a test double. Everything above (files, jobs, the :class:`~beyond_sandbox.Sandbox`) is built on
these calls alone. A transport that also has the ``job_*`` methods of :class:`JobTransport` offers
v1.3 jobs natively; without them jobs run on plain commands.
"""

from __future__ import annotations

import contextlib
import socket
import ssl
import threading
import time
from collections import deque
from typing import Any, Callable, Iterator, Protocol, Sequence, Union, runtime_checkable

from . import _http, _wire
from .errors import (
    EndpointError,
    NotOffered,
    ProtocolError,
    SandboxError,
    Timeout,
    TooLarge,
    TransportError,
)
from .models import ExecResult, Exit, Frame, JobFrame, Stderr, Stdout

DEFAULT_TIMEOUT_MS = 120_000

Token = Union[str, Callable[[], str], None]


class FrameStream(Protocol):
    """An iterator of v1.2 frames. ``close()`` ends it early; closing kills the command."""

    def __iter__(self) -> Iterator[Frame]: ...
    def __next__(self) -> Frame: ...
    def close(self) -> None: ...


class JobFrameStream(Protocol):
    def __iter__(self) -> Iterator[JobFrame]: ...
    def __next__(self) -> JobFrame: ...
    def close(self) -> None: ...


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
        """Run a command and yield its frames as they arrive (v1.2).

        Raises :class:`~beyond_sandbox.NotOffered` if the endpoint cannot stream, rather than
        running the command one-shot. Closing the iterator kills the command.
        """
        ...


@runtime_checkable
class JobTransport(Transport, Protocol):
    """Optional: native sandbox jobs (v1.3)."""

    def job_offer(self) -> _wire.JobOffer: ...

    def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> Any:
        """One job operation; its JSON answer."""
        ...

    def job_stream(self, job: dict[str, Any]) -> JobFrameStream:
        """A job operation answered by a stream (``attach``)."""
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
            raise Timeout(f"exec endpoint {self._url} timed out")
        wait = left if silence is None else min(left, silence)
        try:
            self._sock.settimeout(wait)
            data = self._sock.recv(65536)
            self._reader.feed(data)
        except TimeoutError:
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

    def read_capped(self, cap: int) -> bytes:
        declared = self._reader.content_length
        if declared is not None and declared > cap:
            self.close()
            raise _wire.too_large(self._url, cap)
        body = bytearray()
        while True:
            body += self._reader.take()
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
    """Items of an NDJSON stream (v1.2 frames, or a job stream's). ``close()`` drops the connection."""

    def __init__(
        self, resp: _Response, decoder: _wire.StreamDecoder[Any], first: list[Any]
    ) -> None:
        self._resp, self._decoder = resp, decoder
        self._pending: deque[Any] = deque(first)
        self._closed = False

    def __iter__(self) -> _Frames:
        return self

    def __next__(self) -> Any:
        while not self._pending:
            if self._closed or self._decoder.finished:
                self.close()
                raise StopIteration
            try:
                data = self._resp.next_chunk(_wire.SILENCE_S)
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


class HttpTransport:
    """The exec protocol over HTTP (v1.1 to v1.3), on the standard library alone.

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
        self._streams: bool | None = None
        self._output: bool | None = None
        self._offer: _wire.JobOffer | None = None

    # ---- the wire ---------------------------------------------------------------------------

    def _bearer(self, refresh: bool) -> list[tuple[str, str]]:
        with self._lock:
            if callable(self._token) and (refresh or self._token_value is None):
                self._token_value = self._token()
            value = self._token_value
        return [("Authorization", f"Bearer {value}")] if value else []

    def _connect(self, deadline: float) -> socket.socket:
        t = self._target
        try:
            sock = socket.create_connection(
                (t.host, t.port), timeout=max(deadline - time.monotonic(), 0.001)
            )
            if t.tls:
                ctx = self._ssl or ssl.create_default_context()
                sock = ctx.wrap_socket(sock, server_hostname=t.host)
            return sock
        except TimeoutError:
            raise Timeout(f"exec endpoint {self.url} timed out") from None
        except OSError as e:
            raise TransportError(f"exec endpoint {self.url} unreachable: {e}") from None

    def _open(self, body: bytes, timeout_ms: int, *, refresh: bool = False) -> _Response:
        """POST ``body``; return the response once its head is read. A non-2xx answer raises."""
        deadline = time.monotonic() + timeout_ms / 1000 + _wire.GRACE_S
        request = _http.build_request(self._target, self._headers + self._bearer(refresh), body)
        sock = self._connect(deadline)
        reader = _http.ResponseReader()
        resp = _Response(sock, reader, deadline, self.url)
        try:
            sock.settimeout(max(deadline - time.monotonic(), 0.001))
            sock.sendall(request)
            while not reader.head_done:
                resp._fill(None)
        except TimeoutError:
            resp.close()
            raise Timeout(f"exec endpoint {self.url} timed out") from None
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
            return self._open(body, timeout_ms, refresh=True)
        raise _wire.status_error(self.url, reader.status, err_body)

    def _post(self, body: bytes, timeout_ms: int, cap: int | None = None) -> bytes:
        resp = self._open(body, timeout_ms)
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
    ) -> ExecResult:
        """Run one command; its output is the **bytes** the command wrote (``exact``): over
        ``output_base64`` when the endpoint honours it, else over a v1.2 stream's base64 frames. A
        plain v1.1 endpoint (or a call with stdin on one without ``output_base64``) can only answer
        text, which is returned as UTF-8 with ``exact=False``. A probe that failed raises."""
        if self.outputs_base64():
            body = _wire.request_body(
                command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin, output_base64=True
            )
            raw = self._post(body, timeout_ms, _wire.body_cap(self._cap))
            return _wire.decode_output(self.url, raw, self._cap)
        # A stream has no stdin in the spec: a call with stdin is a v1.1 call, whose text it is.
        if stdin is None and self.streams():
            return self._run_over_stream(command, args, cwd, timeout_ms)
        return self.run_text(command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin)

    def run_text(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        stdin: bytes | None = None,
    ) -> ExecResult:
        """:meth:`run` the plain v1.1 way: the output is the endpoint's text, ``exact=False``. For
        the helpers, which read text (or base64 they asked the shell for) and need neither the
        probe nor the stream; a transport without this method is called through ``run``."""
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=timeout_ms, stdin=stdin)
        return _wire.parse_envelope(self.url, self._post(body, timeout_ms))

    def outputs_base64(self) -> bool:
        """Whether the endpoint honours ``output_base64`` (exec v1.1), probing once:
        ``printf '\\377\\000\\n'`` must come back as ``stdout_base64`` decoding to ``ff 00 0a``. A
        ``400`` is a strict front refusing the field: no. A request that failed raises; it is never
        read as "no"."""
        if self._output is None:
            body = _wire.request_body(
                "sh",
                ["-c", _wire.OUTPUT_PROBE_SCRIPT],
                cwd=None,
                timeout_ms=10_000,
                output_base64=True,
            )
            try:
                raw = self._post(body, 10_000, _wire.body_cap(len(_wire.OUTPUT_PROBE_BYTES)))
            except EndpointError as e:
                if e.status != 400:
                    raise
                self._output = False
                return False
            obj = _wire.parse_json(self.url, raw)
            if not isinstance(obj, dict) or not isinstance(obj.get("exit_code"), int):
                raise ProtocolError(
                    f"exec endpoint {self.url} returned unparseable JSON (no exit_code)"
                )
            self._output = _wire.probe_honoured(obj)
        return self._output

    def _run_over_stream(
        self, command: str, args: Sequence[str], cwd: str | None, timeout_ms: int
    ) -> ExecResult:
        """A one-shot call over a v1.2 stream: the frames are base64 already."""
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=timeout_ms, stream=True)
        resp = self._open(body, timeout_ms)
        try:
            if not resp.headers.get("content-type", "").lower().startswith("application/x-ndjson"):
                # One object after all (a fleet mid-rollout): text.
                return _wire.parse_envelope(self.url, resp.read_capped(self._cap))
            decoder, fold = _wire.frame_decoder(), _wire.OneShot()
            while True:
                data = resp.next_chunk(_wire.SILENCE_S)
                if not data:
                    decoder.eof()
                for frame in decoder.feed(data):
                    done = fold.feed(frame)
                    if done is not None:
                        return done
                    if fold.size > self._cap:
                        raise _wire.too_large(self.url, self._cap)
        finally:
            resp.close()

    # ---- stream (v1.2) -----------------------------------------------------------------------

    def streams(self) -> bool:
        """Whether the endpoint has proved it streams, probing once. An outage is not an answer."""
        if self._streams is None:
            self._streams = self._probe_streams()
        return self._streams

    def _probe_streams(self) -> bool:
        body = _wire.request_body(
            "sh", ["-c", _wire.STREAM_PROBE_SCRIPT], cwd=None, timeout_ms=10_000, stream=True
        )
        try:
            resp = self._open(body, 10_000)
        except EndpointError as e:
            if e.status in (400, 422):  # a strict front refusing the field has answered
                return False
            raise
        try:
            if not resp.headers.get("content-type", "").lower().startswith("application/x-ndjson"):
                return False
            decoder, seen = _wire.frame_decoder(), _wire.ProbeObserver()
            while not decoder.finished:
                data = resp.next_chunk(_wire.SILENCE_S)
                if not data:
                    return False
                for frame in decoder.feed(data):
                    seen.see(frame)
            return seen.streams()
        except ProtocolError:
            return False
        finally:
            resp.close()

    def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> FrameStream:
        if not self.streams():
            raise NotOffered("this exec endpoint does not stream (protocol v1.1)")
        ms = _wire.STREAM_DEADLINE_MS if timeout_ms is None else timeout_ms
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=ms, stream=True)
        resp = self._open(body, ms)
        if resp.headers.get("content-type", "").lower().startswith("application/x-ndjson"):
            return _Frames(resp, _wire.frame_decoder(), [])
        # One object after all (a fleet mid-rollout): everything arrives at once, at the end.
        try:
            result = _wire.parse_envelope(self.url, resp.read_capped(self._cap))
        finally:
            resp.close()
        frames: list[Frame] = []
        if result.stdout:
            frames.append(Stdout(result.stdout))
        if result.stderr:
            frames.append(Stderr(result.stderr))
        frames.append(Exit(result.exit_code))
        return iter_frames(frames)

    # ---- jobs (v1.3) -------------------------------------------------------------------------

    def job_offer(self) -> _wire.JobOffer:
        """What the endpoint offers for jobs, probing once."""
        if self._offer is None:
            body = _wire.request_body(
                "sh", ["-c", "printf exec-v1.2"], cwd=None, timeout_ms=10_000, job={"op": "probe"}
            )
            try:
                answer = _wire.parse_json(self.url, self._post(body, 10_000))
                self._offer = _wire.parse_offer(answer)
            except EndpointError as e:
                if e.status != 400:
                    raise
                self._offer = _wire.JobOffer()
            except ProtocolError:
                self._offer = _wire.JobOffer()
        return self._offer

    def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> Any:
        body = _wire.request_body(command, args, cwd=cwd, timeout_ms=30_000, job=job)
        try:
            return _wire.parse_json(self.url, self._post(body, 30_000))
        except TooLarge:
            raise TooLarge("a job answer exceeded the response cap") from None

    def job_stream(self, job: dict[str, Any]) -> JobFrameStream:
        body = _wire.request_body(
            "true", [], cwd=None, timeout_ms=_wire.STREAM_DEADLINE_MS, stream=True, job=job
        )
        return _Frames(self._open(body, _wire.STREAM_DEADLINE_MS), _wire.job_decoder(), [])

    def close(self) -> None:
        """Nothing is held between requests; present so a transport can be used as a context."""

    def __enter__(self) -> HttpTransport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_frames(frames: list[Frame]) -> FrameStream:
    """A :class:`FrameStream` over frames already in hand."""
    return _Prebuilt(frames)


class _Prebuilt:
    def __init__(self, frames: list[Frame]) -> None:
        self._it = iter(frames)

    def __iter__(self) -> _Prebuilt:
        return self

    def __next__(self) -> Frame:
        return next(self._it)

    def close(self) -> None:
        self._it = iter(())
