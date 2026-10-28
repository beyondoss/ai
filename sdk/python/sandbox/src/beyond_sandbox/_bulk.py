"""Bulk copy: ``upload`` and ``download`` of whole trees as a tar stream over raw exec
(FILES-OVER-EXEC.md section 5.8), as sans-IO generators (see :mod:`beyond_sandbox._drive`).

The archive is written and read here, so the client's machine needs no ``tar``; the sandbox needs
``sh`` and a ``tar``, GNU or busybox. The local side of a download is untrusted input and is
extracted by rules that never write through a link.
"""

from __future__ import annotations

import contextlib
import io
import os
import stat as stat_mod
import tempfile
from collections.abc import Iterable, Iterator
from typing import Any

from . import _scripts, _tar
from ._drive import Op, Pipe, Run
from ._fs import fail, new_suffix, sh
from .errors import (
    InvalidRequest,
    IoError,
    ProtocolError,
    SandboxError,
    classify_stderr,
)
from .models import ExecResult, Transfer

#: Raw tar bytes per upload request: 16 MiB is 21.3 MB of base64, under the 64 MiB an ``execd``
#: accepts and equal to the one-shot response cap. A call holds about two chunks.
BULK_CHUNK_BYTES = 16 * 1024 * 1024
#: The deadline of the extraction command and of the download stream: a hung ``tar`` ends.
BULK_TIMEOUT_MS = 600_000
#: The stderr a download keeps for its error detail.
BULK_STDERR_BYTES = 64 * 1024
_STAGE_TIMEOUT_MS = 120_000
_READ_BLOCK = 64 * 1024


def local_error(path: object, e: OSError) -> SandboxError:
    """A local filesystem failure, classified from its text as section 3.3 does."""
    return classify_stderr(f"{path}: {e.strerror or e} (os error {e.errno})")


def _remote_path(what: str, p: str, *, not_root: bool) -> None:
    if not p.startswith("/"):
        raise InvalidRequest(f"{what} must be absolute: {p!r}")
    if not_root and not p.rstrip("/"):
        raise InvalidRequest(f"{what} must not be /")
    if "\0" in p:
        raise InvalidRequest(f"{what} has a NUL")


# ---- upload: the archive, lazily ----------------------------------------------------------------


class _Counts:
    def __init__(self) -> None:
        self.files = self.dirs = self.links = self.bytes = 0

    def transfer(self) -> Transfer:
        return Transfer(self.files, self.dirs, self.links, self.bytes)


def _pieces(name: bytes, path: bytes, counts: _Counts) -> Iterator[bytes]:
    """The archive bytes of one source path, in pre-order."""
    shown = os.fsdecode(path)
    try:
        st = os.lstat(path)
    except OSError as err:
        raise local_error(shown, err) from None
    try:
        name.decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidRequest(f"{shown}: the name is not valid UTF-8") from None
    mode = st.st_mode
    mtime = max(st.st_mtime_ns // 1_000_000_000, 0)
    e = _tar.Entry(name, b"0", stat_mod.S_IMODE(mode) & 0o777, 0, mtime)
    if stat_mod.S_ISDIR(mode):
        e.kind = b"5"
        yield _tar.header(e)
        counts.dirs += 1
        try:
            names = sorted(os.listdir(path))
        except OSError as err:
            raise local_error(shown, err) from None
        for n in names:
            child = name + b"/" + n if name else n
            yield from _pieces(child, os.path.join(path, n), counts)
    elif stat_mod.S_ISLNK(mode):
        e.kind, e.mode = b"2", 0o777
        try:
            e.link = os.readlink(path)
        except OSError as err:
            raise local_error(shown, err) from None
        yield _tar.header(e)
        counts.links += 1
    elif stat_mod.S_ISREG(mode):
        e.size = st.st_size
        yield _tar.header(e)
        try:
            f = open(path, "rb")  # noqa: SIM115 - closed in the finally below
        except OSError as err:
            raise local_error(shown, err) from None
        try:
            left = e.size
            while left > 0:
                try:
                    data = f.read(min(left, _READ_BLOCK))
                except OSError as err:
                    raise local_error(shown, err) from None
                if not data:
                    raise IoError(f"{shown}: changed while being archived")
                left -= len(data)
                counts.bytes += len(data)
                yield data
        finally:
            f.close()
        yield _tar.pad(e.size)
        counts.files += 1
    else:
        what = (
            "a FIFO"
            if stat_mod.S_ISFIFO(mode)
            else "a socket"
            if stat_mod.S_ISSOCK(mode)
            else "a device"
        )
        raise InvalidRequest(f"{shown}: {what} cannot be uploaded")


def _all_pieces(sources: Any, counts: _Counts) -> Iterator[bytes]:
    if isinstance(sources, (str, os.PathLike)):
        d = os.fsencode(sources)
        try:
            st = os.stat(d)
            if not stat_mod.S_ISDIR(st.st_mode):
                raise IoError(f"{sources}: Not a directory (os error 20)") from None
            names = sorted(os.listdir(d))
        except OSError as e:
            raise local_error(sources, e) from None
        for n in names:
            yield from _pieces(n, os.path.join(d, n), counts)
    else:
        for p in sources:
            b = os.fsencode(p)
            base = os.path.basename(b.rstrip(b"/")) if b.rstrip(b"/") else b""
            if base in (b"", b".", b".."):
                raise InvalidRequest(f"{os.fsdecode(b)}: no base name")
            yield from _pieces(base, b, counts)
    yield _tar.END


def chunks(pieces: Iterable[bytes], chunk: int) -> Iterator[tuple[bytes, bool]]:
    """The archive as ``(bytes, last)`` chunks of at most ``chunk`` bytes, produced as the pieces
    are. The last chunk may be empty (an archive that is an exact multiple of the chunk size)."""
    buf = bytearray()
    for p in pieces:
        buf += p
        while len(buf) >= chunk:
            yield bytes(buf[:chunk]), False
            del buf[:chunk]
    # `pieces` ends with the end-of-archive blocks, so what is left is the tail.
    while len(buf) > chunk:
        yield bytes(buf[:chunk]), False
        del buf[:chunk]
    yield bytes(buf), True


def upload(sources: Any, remote_dir: str, chunk_bytes: int | None = None) -> Op[Transfer]:
    """Upload ``sources`` into the sandbox directory ``remote_dir`` (created if missing)."""
    _remote_path("remote_dir", remote_dir, not_root=True)
    dest = remote_dir.rstrip("/")
    chunk = max(chunk_bytes or BULK_CHUNK_BYTES, _tar.BLOCK)
    counts = _Counts()
    it = chunks(_all_pieces(sources, counts), chunk)
    first, last = next(it)
    if last:
        r: ExecResult = yield Run(
            "sh", sh(_scripts.UPLOAD, dest, ""), stdin=first, timeout_ms=BULK_TIMEOUT_MS
        )
        if r.exit_code != 0:
            raise fail(f"upload {dest}", r)
        return counts.transfer()

    stage = f"{dest}.upload.{new_suffix()}"
    try:
        offset = 0
        data, last = first, False
        while True:
            if data:
                r = yield Run(
                    "sh",
                    sh(_scripts.STAGE, stage, str(offset), str(len(data))),
                    stdin=data,
                    timeout_ms=_STAGE_TIMEOUT_MS,
                )
                if r.exit_code != 0:
                    raise fail(f"upload {dest}", r)
                offset += len(data)
            if last:
                break
            data, last = next(it)
        r = yield Run("sh", sh(_scripts.UPLOAD, dest, stage), timeout_ms=BULK_TIMEOUT_MS)
        if r.exit_code != 0:
            raise fail(f"upload {dest}", r)
    except SandboxError:
        # The compensating action: the scripts remove the stage on the failures they see.
        with contextlib.suppress(SandboxError):
            yield Run("rm", ["-f", "--", stage], timeout_ms=_STAGE_TIMEOUT_MS)
        raise
    return counts.transfer()


# ---- download: the extractor --------------------------------------------------------------------


def _quote(name: bytes) -> str:
    return name.decode("utf-8", "replace").encode("unicode_escape").decode("ascii")


def _unsafe(name: bytes, why: str) -> ProtocolError:
    return ProtocolError(f"unsafe archive member {_quote(name)}: {why}")


def _components(name: bytes) -> list[str]:
    """A name or link target read the way rule 1 reads it: the components below the root."""
    if b"\0" in name:
        raise ValueError("it has a NUL")
    if name[:1] == b"/":
        raise ValueError("it is absolute")
    try:
        text = name.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("it is not valid UTF-8") from None
    out: list[str] = []
    for c in text.split("/"):
        if c in ("", "."):
            continue
        if c == "..":
            raise ValueError('it has a ".." component')
        out.append(c)
    return out


class Extractor:
    """Extracts an archive under ``root`` by the rules of section 5.8; a :class:`~._tar.Sink`."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = os.fspath(root)
        self._fd: int | None = None
        self._cur: tuple[str, int, int] | None = None
        #: (depth, path, mode, mtime) of directories to finish at the end.
        self._dirs: list[tuple[int, str, int, int]] = []
        self._files: set[str] = set()
        self.counts = _Counts()

    # -- helpers
    def _parents(self, name: bytes, comps: list[str]) -> str:
        p = self.root
        for c in comps[:-1]:
            p = os.path.join(p, c)
            try:
                st = os.lstat(p)
            except FileNotFoundError:
                os.mkdir(p, 0o755)
                os.chmod(p, 0o755)
                continue
            if not stat_mod.S_ISDIR(st.st_mode):
                raise _unsafe(name, "a parent is not a directory (a link or a file)")
        return os.path.join(p, comps[-1])

    @staticmethod
    def _clear(path: str) -> bool:
        """True if a real directory is there; a non-directory in the way is unlinked."""
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return False
        if stat_mod.S_ISDIR(st.st_mode):
            return True
        os.unlink(path)
        return False

    # -- the sink
    def begin(self, e: _tar.Entry) -> None:
        try:
            self._begin(e)
        except OSError as err:
            raise local_error(_quote(e.name), err) from None

    def _begin(self, e: _tar.Entry) -> None:
        try:
            comps = _components(e.name)
        except ValueError as why:
            raise _unsafe(e.name, str(why)) from None
        if not comps:
            # The root sets nothing on the directory the caller chose.
            if e.kind == b"5":
                return
            raise _unsafe(e.name, "it names the root")
        kind = e.kind
        if kind == b"5":
            path = self._parents(e.name, comps)
            if not self._clear(path):
                os.mkdir(path, 0o700)
            self._dirs.append((len(comps), path, e.mode, e.mtime))
            self.counts.dirs += 1
        elif kind in (b"0", b"7"):
            path = self._parents(e.name, comps)
            if self._clear(path):
                raise classify_stderr(f"{path}: Is a directory (os error 21)")
            # O_EXCL never follows a link at the final component.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            self._files.add("/".join(comps))
            self.counts.files += 1
            self._fd = fd
            self._cur = (path, e.mode, e.mtime)
        elif kind == b"2":
            path = self._parents(e.name, comps)
            if self._clear(path):
                raise classify_stderr(f"{path}: Is a directory (os error 21)")
            os.symlink(os.fsdecode(e.link), path)
            self.counts.links += 1
        elif kind == b"1":
            try:
                target = _components(e.link)
            except ValueError as why:
                raise _unsafe(e.name, f"its link target is unsafe: {why}") from None
            if not target or "/".join(target) not in self._files:
                raise _unsafe(e.name, "a hard link must name a file this archive already wrote")
            path = self._parents(e.name, comps)
            if self._clear(path):
                raise classify_stderr(f"{path}: Is a directory (os error 21)")
            src = os.path.join(self.root, *target)
            try:
                ok = stat_mod.S_ISREG(os.lstat(src).st_mode)
            except OSError:
                ok = False
            if not ok:
                raise _unsafe(e.name, "its link target is not a file")
            os.link(src, path)
            self._files.add("/".join(comps))
            self.counts.links += 1
        else:
            name = e.name.decode("utf-8", "replace")
            raise IoError(f"{name}: unsupported member type {chr(kind[0])!r}")

    def data(self, chunk: bytes | memoryview) -> None:
        if self._fd is None:
            return
        try:
            view = memoryview(chunk)
            while view:
                n = os.write(self._fd, view)
                view = view[n:]
        except OSError as err:
            raise local_error(self._cur[0] if self._cur else "", err) from None
        self.counts.bytes += len(chunk)

    def end(self) -> None:
        if self._fd is None or self._cur is None:
            return
        fd, (path, mode, mtime) = self._fd, self._cur
        self._fd = self._cur = None
        try:
            try:
                os.fchmod(fd, mode & 0o777)
                os.utime(fd, (mtime, mtime))
            finally:
                os.close(fd)
        except OSError as err:
            raise local_error(path, err) from None

    def abort(self) -> None:
        """Close a file left open by an error."""
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

    def finish(self) -> Transfer:
        """The directories' modes and mtimes, deepest first."""
        for _, path, mode, mtime in sorted(self._dirs, key=lambda d: -d[0]):
            try:
                os.chmod(path, mode & 0o777)
                os.utime(path, (mtime, mtime))
            except OSError as err:
                raise local_error(path, err) from None
        self._dirs = []
        return self.counts.transfer()


def _make_local(local_dir: str | os.PathLike[str]) -> None:
    try:
        os.makedirs(local_dir, exist_ok=True)
    except OSError as e:
        raise local_error(local_dir, e) from None


def extract(archive: bytes, local_dir: str | os.PathLike[str]) -> Transfer:
    """Extract an archive already in memory under ``local_dir``: the local half of a download."""
    _make_local(local_dir)
    ex = Extractor(local_dir)
    try:
        _tar.read_archive(io.BytesIO(archive), ex)
        return ex.finish()
    finally:
        ex.abort()


def download(remote_path: str, local_dir: str | os.PathLike[str]) -> Op[Transfer]:
    """Download ``remote_path`` (a directory's contents, or one file) into ``local_dir``.

    The stream is spooled to an unnamed temporary file as it arrives (disk, not memory: it is the
    archive's size) and read back by ``tarfile``, which wants a file to pull from.
    """
    _remote_path("remote_path", remote_path, not_root=False)
    _make_local(local_dir)
    ex = Extractor(local_dir)
    try:
        with tempfile.TemporaryFile() as spool:

            def spool_write(b: bytes) -> None:
                spool.write(b)

            r: ExecResult = yield Pipe(
                "sh",
                sh(_scripts.DOWNLOAD, remote_path),
                spool_write,
                BULK_TIMEOUT_MS,
                BULK_STDERR_BYTES,
            )
            if r.exit_code != 0:
                raise fail(f"download {remote_path}", r)
            _tar.read_archive(spool, ex)
        return ex.finish()
    finally:
        ex.abort()
