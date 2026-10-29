"""A minimal HTTP/1.1 client core with no I/O: it builds a request and parses a response.

The transports feed it bytes from a socket (sync) or an asyncio stream (async), so the framing is
written once. One request per connection (``Connection: close``): the exec protocol is one POST per
call, and a connection that is never reused can never be stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from .errors import InvalidRequestError

#: The longest response head accepted. Bounds what a broken endpoint can make us buffer.
MAX_HEAD = 64 * 1024
#: The longest chunk-size or trailer line accepted.
MAX_LINE = 4096


class Malformed(Exception):
    """The peer did not speak HTTP/1.1."""


class Truncated(Exception):
    """The connection ended before the message did."""


@dataclass(frozen=True, slots=True)
class Target:
    scheme: str
    host: str
    port: int
    path: str
    host_header: str

    @property
    def tls(self) -> bool:
        return self.scheme == "https"


def parse_url(url: str) -> Target:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as e:
        raise InvalidRequestError(f"invalid exec URL {url!r}: {e}") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InvalidRequestError(f"exec URL must be http or https with a host, got {url!r}")
    host = parts.hostname
    default = 443 if parts.scheme == "https" else 80
    shown = f"[{host}]" if ":" in host else host
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return Target(
        parts.scheme,
        host,
        port or default,
        path,
        shown if port in (None, default) else f"{shown}:{port}",
    )


def build_request(target: Target, headers: list[tuple[str, str]], body: bytes) -> bytes:
    lines = [
        f"POST {target.path} HTTP/1.1",
        f"Host: {target.host_header}",
        "Connection: close",
        "Accept: */*",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
    ]
    for name, value in headers:
        if any(c in name + value for c in "\r\n\0"):
            raise InvalidRequestError(f"header {name!r} contains a control character")
        lines.append(f"{name}: {value}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


class ResponseReader:
    """Feed it bytes with :meth:`feed` (``b""`` is end of stream); take the body with :meth:`take`."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._out = bytearray()
        self.status = 0
        self.headers: dict[str, str] = {}
        self.head_done = False
        self.done = False
        self._mode = "head"  # head | length | chunked | close | none
        self._remaining = 0
        self._chunk = "size"  # size | data | crlf | trailer

    # ---- input --------------------------------------------------------------------------------

    def feed(self, data: bytes) -> None:
        if self.done:
            return
        if not data:
            self._eof()
            return
        self._buf += data
        self._advance()

    def _eof(self) -> None:
        if self._mode == "close":
            self.done = True
        else:
            raise Truncated("the connection closed before the response was complete")

    def _advance(self) -> None:
        while not self.done:
            if self._mode == "head":
                if not self._head():
                    return
            elif self._mode == "length":
                n = min(self._remaining, len(self._buf))
                self._out += self._buf[:n]
                del self._buf[:n]
                self._remaining -= n
                if self._remaining == 0:
                    self.done = True
                return
            elif self._mode == "close":
                self._out += self._buf
                self._buf.clear()
                return
            elif self._mode == "chunked":
                if not self._chunked():
                    return
            else:  # none
                self.done = True

    def _head(self) -> bool:
        end = self._buf.find(b"\r\n\r\n")
        if end < 0:
            if len(self._buf) > MAX_HEAD:
                raise Malformed("the response head is too long")
            return False
        raw = bytes(self._buf[:end]).decode("latin-1")
        del self._buf[: end + 4]
        lines = raw.split("\r\n")
        bits = lines[0].split(" ", 2)
        if len(bits) < 2 or not bits[0].startswith("HTTP/1.") or not bits[1].isdigit():
            raise Malformed(f"bad status line {lines[0][:80]!r}")
        status = int(bits[1])
        if 100 <= status < 200 and status != 101:
            return True  # an interim response; the real head follows
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, sep, value = line.partition(":")
            if not sep or not name or name != name.strip():
                raise Malformed(f"bad header line {line[:80]!r}")
            headers[name.lower()] = value.strip()
        self.status, self.headers, self.head_done = status, headers, True
        if status in (204, 304):
            self._mode = "none"
        elif "chunked" in headers.get("transfer-encoding", "").lower():
            self._mode = "chunked"
        elif "content-length" in headers:
            value = headers["content-length"]
            if not value.isdigit():
                raise Malformed(f"bad Content-Length {value!r}")
            self._mode, self._remaining = "length", int(value)
            if self._remaining == 0:
                self.done = True
        else:
            self._mode = "close"
        return True

    def _chunked(self) -> bool:
        buf = self._buf
        if self._chunk == "data":
            n = min(self._remaining, len(buf))
            self._out += buf[:n]
            del buf[:n]
            self._remaining -= n
            if self._remaining:
                return False
            self._chunk = "crlf"
        if self._chunk == "crlf":
            if len(buf) < 2:
                return False
            if buf[:2] != b"\r\n":
                raise Malformed("a chunk is not followed by CRLF")
            del buf[:2]
            self._chunk = "size"
        end = buf.find(b"\r\n")
        if end < 0:
            if len(buf) > MAX_LINE:
                raise Malformed("a chunk header line is too long")
            return False
        line = bytes(buf[:end])
        del buf[: end + 2]
        if self._chunk == "trailer":
            if not line:
                self.done = True
            return True
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            raise Malformed(f"bad chunk size {line[:40]!r}") from None
        if size == 0:
            self._chunk = "trailer"
        else:
            self._chunk, self._remaining = "data", size
        return True

    # ---- output -------------------------------------------------------------------------------

    def take(self) -> bytes:
        data = bytes(self._out)
        self._out.clear()
        return data

    @property
    def content_length(self) -> int | None:
        value = self.headers.get("content-length", "")
        return int(value) if value.isdigit() else None

    @property
    def is_ndjson(self) -> bool:
        return is_ndjson(self.headers)


def is_ndjson(headers: dict[str, str]) -> bool:
    """Whether a response says it is a stream (``Content-Type: application/x-ndjson``)."""
    media = headers.get("content-type", "").split(";")[0].strip().lower()
    return media == "application/x-ndjson"
