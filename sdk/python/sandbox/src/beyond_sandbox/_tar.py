"""The tar subset of FILES-OVER-EXEC.md section 5.8: header writing for upload, and a push parser
for download.

Written by hand rather than with ``tarfile``: the parser is *pushed* chunks as they arrive (one
stream frame at a time), so the async client streams without a thread between the network and the
extractor, and the same bytes mean the same thing in every client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .errors import ProtocolError

BLOCK = 512
#: The longest meta member (long name, long link, pax records) the reader holds.
META_MAX = 1 << 20


@dataclass(slots=True)
class Entry:
    """A member of an archive. ``name`` is the name as the archive gives it (prefix, GNU ``L`` and
    pax already applied)."""

    name: bytes
    kind: bytes  # one byte: b"0" file, b"5" dir, b"2" symlink, b"1" hard link, ...
    mode: int = 0
    size: int = 0
    mtime: int = 0
    link: bytes = b""


class Sink(Protocol):
    def begin(self, entry: Entry) -> None: ...
    def data(self, chunk: bytes | memoryview) -> None: ...
    def end(self) -> None: ...


# ---- writing ---------------------------------------------------------------------------------------


def _octal(value: int, width: int) -> bytes:
    w = width - 1
    return f"{value:0{w}o}".encode()[-w:] + b"\0"


def _raw_header(name: bytes, kind: bytes, mode: int, size: int, mtime: int, link: bytes) -> bytes:
    h = bytearray(BLOCK)
    h[0 : min(len(name), 100)] = name[:100]
    h[100:108] = _octal(mode & 0o7777, 8)
    h[108:116] = _octal(0, 8)
    h[116:124] = _octal(0, 8)
    if size >= 8**11:
        # Base-256: the high bit of the first byte, then the number big-endian.
        h[124:136] = b"\x80" + size.to_bytes(11, "big")
    else:
        h[124:136] = _octal(size, 12)
    h[136:148] = _octal(mtime, 12)
    h[156:157] = kind
    h[157 : 157 + min(len(link), 100)] = link[:100]
    h[257:263] = b"ustar\0"
    h[263:265] = b"00"
    h[148:156] = b" " * 8
    h[148:154] = f"{sum(h):06o}".encode()
    h[154] = 0
    h[155] = 0x20
    return bytes(h)


def pad(size: int) -> bytes:
    """The zero bytes that round ``size`` up to a block."""
    return bytes(-size % BLOCK)


def _extension(kind: bytes, text: bytes) -> bytes:
    body = text + b"\0"
    return _raw_header(b"././@LongLink", kind, 0, len(body), 0, b"") + body + pad(len(body))


def header(e: Entry) -> bytes:
    """The header of one member, preceded by its GNU ``L``/``K`` members when its name or link
    target is over 100 bytes. ``e.name`` is relative with no trailing slash; a directory gets one."""
    name = e.name + b"/" if e.kind == b"5" else e.name
    out = b""
    if len(name) > 100:
        out += _extension(b"L", name)
    if len(e.link) > 100:
        out += _extension(b"K", e.link)
    return out + _raw_header(name, e.kind, e.mode & 0o777, e.size, e.mtime, e.link)


END = bytes(2 * BLOCK)

# ---- reading ---------------------------------------------------------------------------------------


def _str_field(f: bytes) -> bytes:
    i = f.find(b"\0")
    return f if i < 0 else f[:i]


def _number(f: bytes, what: str) -> int:
    if f[0] & 0x80:
        if f[0] == 0xFF:
            raise ProtocolError(f"negative {what} in a tar header")
        return int.from_bytes(bytes([f[0] & 0x7F]) + f[1:], "big")
    v = 0
    seen = False
    for b in f:
        if 0x30 <= b <= 0x37:
            seen = True
            v = v * 8 + (b - 0x30)
        elif b in (0x20, 0) and not seen:
            continue
        elif b in (0x20, 0):
            break
        else:
            raise ProtocolError(f"bad {what} in a tar header")
    return v


def _checksum_ok(h: bytes) -> bool:
    try:
        want = _number(h[148:156], "checksum")
    except ProtocolError:
        return False
    blank = h[:148] + b" " * 8 + h[156:]
    unsigned = sum(blank)
    signed = sum(b - 256 if b > 127 else b for b in blank)
    return want in (unsigned, signed)


class Parser:
    """An incremental reader: :meth:`feed` it chunks as they arrive, then :meth:`finish`."""

    def __init__(self, sink: Sink) -> None:
        self._sink = sink
        self._buf = bytearray()
        # states: "header", "data", "meta", "pad", "done"
        self._state = "header"
        self._left = 0
        self._pad = 0
        self._meta_kind = b""
        self._meta = bytearray()
        self._zero = 0
        self._long_name: bytes | None = None
        self._long_link: bytes | None = None
        self._pax_path: bytes | None = None
        self._pax_link: bytes | None = None
        self._pax_size: int | None = None

    def feed(self, chunk: bytes | memoryview) -> None:
        view = memoryview(chunk)
        while True:
            st = self._state
            if st == "done":
                return
            if st == "header":
                take = min(BLOCK - len(self._buf), len(view))
                self._buf += view[:take]
                view = view[take:]
                if len(self._buf) < BLOCK:
                    return
                block = bytes(self._buf)
                self._buf.clear()
                self._header(block)
            elif st == "data":
                if self._left > 0:
                    take = min(self._left, len(view))
                    if take == 0:
                        return
                    self._sink.data(view[:take])
                    view = view[take:]
                    self._left -= take
                    if self._left > 0:
                        return
                self._sink.end()
                self._state = "pad"
            elif st == "meta":
                take = min(self._left, len(view))
                self._meta += view[:take]
                view = view[take:]
                self._left -= take
                if self._left > 0:
                    return
                self._apply_meta(self._meta_kind, bytes(self._meta))
                self._meta = bytearray()
                self._state = "pad"
            else:  # pad
                take = min(self._pad, len(view))
                view = view[take:]
                self._pad -= take
                if self._pad > 0:
                    return
                self._state = "header"

    def finish(self) -> None:
        """The input is over: it must have ended at the end-of-archive block."""
        if self._state == "done" or (self._state == "header" and not self._buf and self._zero > 0):
            return
        raise ProtocolError("the archive ended before its end-of-archive block")

    def _header(self, h: bytes) -> None:
        if not any(h):
            self._zero += 1
            if self._zero >= 2:
                self._state = "done"
            return
        self._zero = 0
        if not _checksum_ok(h):
            raise ProtocolError("bad tar header checksum")
        kind = h[156:157]
        if kind == b"\0":
            kind = b"0"
        size = _number(h[124:136], "size")
        if kind in (b"L", b"K", b"x", b"g"):
            if size > META_MAX:
                raise ProtocolError(f"a {size}-byte tar extension header")
            self._state = "meta"
            self._meta_kind = kind
            self._left = size
            self._pad = -size % BLOCK
            return
        name = _str_field(h[0:100])
        # POSIX ustar only: a GNU header ("ustar  \0") keeps other fields where `prefix` is.
        if h[257:263] == b"ustar\0":
            prefix = _str_field(h[345:500])
            if prefix:
                name = prefix + b"/" + name
        link = _str_field(h[157:257])
        if self._long_name is not None:
            name, self._long_name = self._long_name, None
        if self._long_link is not None:
            link, self._long_link = self._long_link, None
        if self._pax_path is not None:
            name = self._pax_path
        if self._pax_link is not None:
            link = self._pax_link
        if self._pax_size is not None:
            size = self._pax_size
        self._pax_path = self._pax_link = self._pax_size = None
        entry = Entry(
            name, kind, _number(h[100:108], "mode"), size, _number_or_zero(h[136:148]), link
        )
        # Only regular files carry content; a link or directory with a size has none.
        content = size if kind in (b"0", b"7") else 0
        self._sink.begin(entry)
        self._state = "data"
        self._left = content
        self._pad = -content % BLOCK

    def _apply_meta(self, kind: bytes, buf: bytes) -> None:
        text = _str_field(buf)
        if kind == b"L":
            self._long_name = text
        elif kind == b"K":
            self._long_link = text
        elif kind == b"x":
            rest = buf
            while rest:
                sp = rest.find(b" ")
                if sp < 0:
                    raise ProtocolError("bad pax record")
                try:
                    n = int(rest[:sp])
                except ValueError:
                    raise ProtocolError("bad pax record length") from None
                if n <= sp + 1 or n > len(rest):
                    raise ProtocolError("bad pax record length")
                rec = rest[sp + 1 : n].removesuffix(b"\n")
                key, eq, val = rec.partition(b"=")
                if eq:
                    if key == b"path":
                        self._pax_path = val
                    elif key == b"linkpath":
                        self._pax_link = val
                    elif key == b"size":
                        try:
                            self._pax_size = int(val)
                        except ValueError:
                            self._pax_size = None
                rest = rest[n:]
        # `g`: global records are not used.


def _number_or_zero(f: bytes) -> int:
    try:
        return _number(f, "mtime")
    except ProtocolError:
        return 0
