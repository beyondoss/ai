"""The tar subset of FILES-OVER-EXEC.md section 5.8 on the standard library's ``tarfile``.

Headers are written with ``TarInfo.tobuf`` (GNU format: long names and links use ``L``/``K``, large
sizes base-256), so the upload can be produced lazily, one piece at a time. A download is read with
``tarfile`` and each member is handed to the extractor of ``_bulk``, which applies the rules of 5.8;
``tarfile``'s own extraction filters are not used because ``data`` refuses a symlink whose target
leaves the destination, which 5.8 allows (a link is never followed), and the rules here are the same
in every client.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from typing import IO, Protocol

from .errors import ProtocolError

BLOCK = 512
_READ = 64 * 1024
_ENCODING = "utf-8"
_ERRORS = "surrogateescape"


@dataclass(slots=True)
class Entry:
    """A member of an archive. ``name`` is the name as the archive gives it, as bytes."""

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


def pad(size: int) -> bytes:
    """The zero bytes that round ``size`` up to a block."""
    return bytes(-size % BLOCK)


#: The end of the archive: two zero blocks.
END = bytes(2 * BLOCK)


def header(e: Entry) -> bytes:
    """The header of one member, preceded by its GNU ``L``/``K`` members when its name or link
    target is long. ``e.name`` is relative with no trailing slash; a directory gets one."""
    ti = tarfile.TarInfo(e.name.decode(_ENCODING, _ERRORS))
    ti.type = e.kind
    ti.mode = e.mode & 0o777
    ti.size = e.size if e.kind == b"0" else 0
    ti.mtime = e.mtime
    ti.linkname = e.link.decode(_ENCODING, _ERRORS)
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti.tobuf(tarfile.GNU_FORMAT, _ENCODING, _ERRORS)


def _kind(ti: tarfile.TarInfo) -> bytes:
    return b"0" if ti.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.CONTTYPE) else ti.type


def read_archive(f: IO[bytes], sink: Sink) -> None:
    """Read the archive in ``f`` (seekable) and hand each member to ``sink``.

    A malformed or truncated archive, or one that lacks its end-of-archive block, is a
    :class:`~beyond_sandbox.ProtocolError`.
    """
    f.seek(0, 2)
    total = f.tell()
    f.seek(0)
    try:
        # An empty file is not an archive: `tarfile` calls it "empty file".
        with tarfile.open(fileobj=f, mode="r:", encoding=_ENCODING, errors=_ERRORS) as tf:
            for ti in tf:
                sink.begin(
                    Entry(
                        ti.name.encode(_ENCODING, _ERRORS),
                        _kind(ti),
                        ti.mode,
                        ti.size,
                        max(int(ti.mtime), 0),
                        ti.linkname.encode(_ENCODING, _ERRORS),
                    )
                )
                if _kind(ti) == b"0":
                    src = tf.extractfile(ti)
                    if src is not None:
                        while True:
                            chunk = src.read(_READ)
                            if not chunk:
                                break
                            sink.data(chunk)
                sink.end()
            ended = total - tf.offset >= BLOCK
    except tarfile.TarError as e:
        raise ProtocolError(f"malformed tar archive: {e}") from None
    if not ended:
        raise ProtocolError("the archive ended before its end-of-archive block")
