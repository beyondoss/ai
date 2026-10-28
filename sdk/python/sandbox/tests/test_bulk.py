"""Bulk copy (FILES-OVER-EXEC.md 5.8): the tar subset, the safe extractor, and upload/download
over a transport double that runs the commands on this host (so the sandbox's own ``tar`` reads
and writes what the client does)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import stat
import subprocess
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from beyond_sandbox import (
    AsyncSandbox,
    ExecResult,
    Exit,
    InvalidRequest,
    IoError,
    IsDirectory,
    NotADirectory,
    NotFound,
    ProtocolError,
    Sandbox,
    Stderr,
    Stdout,
    Transfer,
    _bulk,
    _tar,
)

# ---- a transport that runs on this host ----------------------------------------------------------


class HostTransport:
    """``run`` and ``stream`` in local subprocesses. ``frame`` is the size of a stdout frame."""

    def __init__(self, frame: int = 65536) -> None:
        self.frame = frame
        self.calls: list[tuple[list[str], int | None]] = []

    def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = 120_000,
        stdin: bytes | None = None,
    ) -> ExecResult:
        self.calls.append(([command, *args], None if stdin is None else len(stdin)))
        p = subprocess.run([command, *args], input=stdin or b"", capture_output=True)
        return ExecResult(p.returncode, p.stdout, p.stderr)

    def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        self.calls.append(([command, *args], None))
        p = subprocess.run([command, *args], capture_output=True)
        return _Frames(p.stdout, p.stderr, p.returncode, self.frame)


class _Frames:
    def __init__(self, out: bytes, err: bytes, code: int, size: int) -> None:
        def gen() -> Iterator[Any]:
            if err:
                yield Stderr(err)
            for i in range(0, len(out), size):
                yield Stdout(out[i : i + size])
            yield Exit(code)

        self._it = gen()
        self.closed = False

    def __iter__(self) -> _Frames:
        return self

    def __next__(self) -> Any:
        return next(self._it)

    def __aiter__(self) -> _Frames:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class AsyncHostTransport:
    def __init__(self, inner: HostTransport) -> None:
        self.inner = inner

    async def run(self, command: str, args: Sequence[str] = (), **kw: Any) -> ExecResult:
        return self.inner.run(command, args, **kw)

    async def stream(self, command: str, args: Sequence[str] = (), **kw: Any) -> Any:
        return self.inner.stream(command, args, **kw)


def tree(root: Path) -> dict[str, tuple[str, ...]]:
    """path -> (kind, mode, content hash or link target), the way a copy must preserve it."""
    out: dict[str, tuple[str, ...]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            rel = str(p.relative_to(root))
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode):
                out[rel] = ("link", os.readlink(p))
            elif stat.S_ISDIR(st.st_mode):
                out[rel] = ("dir", f"{stat.S_IMODE(st.st_mode):04o}")
            else:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                out[rel] = ("file", f"{stat.S_IMODE(st.st_mode):04o}", h, str(int(st.st_mtime)))
    return out


def make_tree(root: Path) -> None:
    (root / "empty dir").mkdir(parents=True)
    (root / "a b.txt").write_bytes(b"spaces")
    (root / "new\nline").write_bytes(b"newline")
    (root / "-rf").write_bytes(b"dash")
    (root / ("n" * 200)).write_bytes(b"long")
    deep = root / ("p" * 60) / ("q" * 60) / ("r" * 60)
    deep.mkdir(parents=True)
    (deep / "file").write_bytes(b"deep")
    (root / "run.sh").write_bytes(b"#!/bin/sh\n")
    (root / "run.sh").chmod(0o755)
    os.utime(root / "run.sh", (1500000000, 1500000000))
    (root / "ro.txt").write_bytes(b"ro")
    (root / "ro.txt").chmod(0o444)
    (root / "link").symlink_to("run.sh")
    (root / "dangling").symlink_to("no/such")
    (root / "far").symlink_to("t" * 130)
    (root / "big.bin").write_bytes(os.urandom(70_000))
    (root / "private").mkdir()
    (root / "private" / "s").write_bytes(b"s")
    (root / "private" / "s").chmod(0o600)
    (root / "private").chmod(0o700)


# ---- the archive -----------------------------------------------------------------------------------


class Collect:
    def __init__(self) -> None:
        self.seen: list[tuple[_tar.Entry, bytearray]] = []

    def begin(self, entry: _tar.Entry) -> None:
        self.seen.append((entry, bytearray()))

    def data(self, chunk: bytes | memoryview) -> None:
        self.seen[-1][1].extend(chunk)

    def end(self) -> None:
        pass


def archive(members: list[tuple[_tar.Entry, bytes]]) -> bytes:
    out = b""
    for e, data in members:
        out += _tar.header(e) + data + _tar.pad(len(data))
    return out + _tar.END


@pytest.mark.parametrize("step", [1, 7, 511, 512, 513, 100_000])
def test_what_is_written_is_read_back_at_every_chunk_size(step: int) -> None:
    long, tgt = "d" * 150, "t" * 130
    a = archive(
        [
            (_tar.Entry(b"a.txt", b"0", 0o640, 5, 1_700_000_000), b"hello"),
            (_tar.Entry(b"sub", b"5", 0o755), b""),
            (_tar.Entry(f"sub/{long}".encode(), b"0", 0o644, 3), b"xyz"),
            (_tar.Entry(b"l", b"2", 0o777, 0, 0, tgt.encode()), b""),
            (_tar.Entry(b"e", b"0", 0o644), b""),
        ]
    )
    c = Collect()
    p = _tar.Parser(c)
    for i in range(0, len(a), step):
        p.feed(a[i : i + step])
    p.finish()
    assert [e.name for e, _ in c.seen] == [b"a.txt", b"sub/", f"sub/{long}".encode(), b"l", b"e"]
    assert bytes(c.seen[0][1]) == b"hello" and bytes(c.seen[2][1]) == b"xyz"
    assert c.seen[3][0].link == tgt.encode()
    assert (c.seen[0][0].mode, c.seen[0][0].mtime) == (0o640, 1_700_000_000)


def test_base_256_sizes_round_trip() -> None:
    h = _tar.header(_tar.Entry(b"big", b"0", 0o644, 9 << 30))
    assert _tar._number(h[124:136], "size") == 9 << 30


def test_a_bad_checksum_and_a_missing_end_are_refused() -> None:
    a = archive([(_tar.Entry(b"a", b"0", 0o644, 1), b"x")])
    p = _tar.Parser(Collect())
    p.feed(a[:-1024])
    with pytest.raises(ProtocolError):
        p.finish()
    with pytest.raises(ProtocolError):
        _tar.Parser(Collect()).feed(bytes([a[0] ^ 1]) + a[1:])


def py_tar(fmt: int, members: list[tuple[str, int, bytes, str]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as t:
        for name, kind, data, link in members:
            ti = tarfile.TarInfo(name)
            ti.mode, ti.mtime = (0o755 if kind == 5 else 0o640), 1500000000
            if kind == 5:
                ti.type = tarfile.DIRTYPE
                t.addfile(ti)
            elif kind == 2:
                ti.type, ti.linkname = tarfile.SYMTYPE, link
                t.addfile(ti)
            else:
                ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.parametrize("fmt", [tarfile.GNU_FORMAT, tarfile.PAX_FORMAT])
def test_gnu_and_pax_archives_extract(tmp_path: Path, fmt: int) -> None:
    long = "L" * 130 + "/" + "n" * 120
    raw = py_tar(
        fmt,
        [
            ("plain.txt", 0, b"plain", ""),
            (long, 0, b"long", ""),
            ("dir/", 5, b"", ""),
            ("dir/ln", 2, b"", "t" * 150),
            ("üñï.txt", 0, b"uni", ""),
        ],
    )
    t = _bulk.extract(raw, tmp_path)
    assert (tmp_path / long).read_bytes() == b"long"
    assert os.readlink(tmp_path / "dir/ln") == "t" * 150
    assert (tmp_path / "üñï.txt").read_bytes() == b"uni"
    assert t.files == 3 and t.links == 1


def test_ustar_prefix_extracts(tmp_path: Path) -> None:
    name = "p" * 90 + "/" + "q" * 60 + "/file.txt"
    _bulk.extract(py_tar(tarfile.USTAR_FORMAT, [(name, 0, b"deep", "")]), tmp_path)
    assert (tmp_path / name).read_bytes() == b"deep"


# ---- the extractor's rules -------------------------------------------------------------------------


def raw_tar(members: list[tuple[str, bytes, bytes, str, int]], end: bool = True) -> bytes:
    out = b""
    for name, kind, data, link, mode in members:
        out += _tar.header(
            _tar.Entry(name.encode(), kind, mode, len(data), 1500000000, link.encode())
        )
        out += data + _tar.pad(len(data))
    return out + (_tar.END if end else b"")


def f(name: str, data: bytes = b"x", mode: int = 0o644) -> tuple[str, bytes, bytes, str, int]:
    return (name, b"0", data, "", mode)


@pytest.mark.parametrize(
    "members,why",
    [
        ([f("../escape.txt")], r"\.\."),
        ([f("a/b/../../../escape.txt")], r"\.\."),
        ([("../escape-dir", b"5", b"", "", 0o755)], r"\.\."),
        ([f("a/..")], r"\.\."),
        ([f("/tmp/abs.txt")], "absolute"),
        ([("up", b"2", b"", "..", 0o777), f("up/escape.txt")], "not a directory"),
        ([("up", b"2", b"", "..", 0o777), ("up/sub", b"5", b"", "", 0o755)], "not a directory"),
        ([("h", b"1", b"", "../../../etc/passwd", 0o644)], "link target is unsafe"),
        ([("h", b"1", b"", "/etc/passwd", 0o644)], "absolute"),
        ([("h", b"1", b"", "never", 0o644)], "already wrote"),
        ([("s", b"2", b"", "/etc/passwd", 0o777), ("h", b"1", b"", "s", 0o644)], "already wrote"),
    ],
)
def test_hostile_members_are_refused(
    tmp_path: Path, members: list[tuple[str, bytes, bytes, str, int]], why: str
) -> None:
    local = tmp_path / "local"
    with pytest.raises(ProtocolError, match=f"unsafe archive member.*{why}"):
        _bulk.extract(raw_tar(members), local)
    assert [p.name for p in tmp_path.iterdir()] == ["local"], "nothing outside the target"


def test_nothing_is_written_through_a_link_already_there(tmp_path: Path) -> None:
    local, outside = tmp_path / "local", tmp_path / "outside"
    local.mkdir()
    outside.mkdir()
    (local / "link").symlink_to(outside)
    with pytest.raises(ProtocolError, match="not a directory"):
        _bulk.extract(raw_tar([f("link/f.txt")]), local)
    assert list(outside.iterdir()) == []
    # A link in the way of a directory member is replaced by the directory.
    _bulk.extract(raw_tar([("link", b"5", b"", "", 0o755), f("link/f.txt")]), local)
    assert (local / "link").is_dir() and not (local / "link").is_symlink()
    assert list(outside.iterdir()) == []
    # ... and one in the way of a file is replaced by the file.
    (local / "l2").symlink_to(outside / "target")
    _bulk.extract(raw_tar([f("l2", b"now a file")]), local)
    assert (local / "l2").read_bytes() == b"now a file" and list(outside.iterdir()) == []


def test_links_hard_links_and_modes(tmp_path: Path) -> None:
    raw = raw_tar(
        [
            ("abs", b"2", b"", "/etc/passwd", 0o777),
            f("a", b"data"),
            ("./b", b"1", b"", "./a", 0o644),
            ("./", b"5", b"", "", 0o700),
            ("x/./y//z.txt", b"0", b"z", "", 0o4755),
            ("d", b"5", b"", "", 0o555),
            f("d/f", b"f", 0o444),
        ]
    )
    tmp_path = tmp_path / "t"
    tmp_path.mkdir(mode=0o755)
    t = _bulk.extract(raw, tmp_path)
    assert os.readlink(tmp_path / "abs") == "/etc/passwd"
    assert (tmp_path / "a").stat().st_ino == (tmp_path / "b").stat().st_ino
    assert stat.S_IMODE((tmp_path / "x/y/z.txt").stat().st_mode) == 0o755
    assert stat.S_IMODE((tmp_path / "d").stat().st_mode) == 0o555
    assert stat.S_IMODE((tmp_path / "d/f").stat().st_mode) == 0o444
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o755, "the root member sets nothing"
    assert (t.files, t.links) == (3, 2)
    os.chmod(tmp_path / "d", 0o755)


def test_what_it_cannot_make(tmp_path: Path) -> None:
    for kind in (b"3", b"6"):
        with pytest.raises(IoError, match="unsupported member type"):
            _bulk.extract(raw_tar([("dev", kind, b"", "", 0o644)]), tmp_path)
    with pytest.raises(ProtocolError):
        _bulk.extract(raw_tar([f("a")], end=False), tmp_path)
    with pytest.raises(ProtocolError):
        _bulk.extract(b"this is not a tar archive at all" * 40, tmp_path)
    (tmp_path / "d").mkdir()
    with pytest.raises(IsDirectory):
        _bulk.extract(raw_tar([f("d")]), tmp_path)


# ---- upload and download -----------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [None, 4096, 50_000, 512])
def test_a_tree_round_trips_through_the_sandbox(tmp_path: Path, chunk: int | None) -> None:
    src, remote, back = tmp_path / "src", tmp_path / "remote", tmp_path / "back"
    make_tree(src)
    t = HostTransport()
    sb = Sandbox(t)  # type: ignore[arg-type]
    up = sb.upload(src, str(remote), chunk_bytes=chunk)
    down = sb.download(str(remote), back)
    assert isinstance(up, Transfer) and up == down
    assert tree(src) == tree(remote)
    # Directory mtimes depend on the sandbox's tar; everything else is exact.
    assert tree(back) == tree(src)
    if chunk:
        assert any("upload." in " ".join(c[0]) for c in t.calls), "staged"
        assert not [p for p in tmp_path.iterdir() if "upload" in p.name], "no stage left"
    else:
        assert [c[1] is not None for c in t.calls][:1] == [True]


def test_one_request_when_it_fits_and_staged_when_it_does_not(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a").write_bytes(b"a" * 5000)
    t = HostTransport()
    Sandbox(t).upload(tmp_path / "src", str(tmp_path / "d1"))  # type: ignore[arg-type]
    assert len(t.calls) == 1 and t.calls[0][0][:2] == ["sh", "-c"]
    assert t.calls[0][0][4:] == [str(tmp_path / "d1"), ""] and t.calls[0][1] is not None
    t = HostTransport()
    Sandbox(t).upload(tmp_path / "src", str(tmp_path / "d2"), chunk_bytes=2048)  # type: ignore[arg-type]
    sizes = [c[1] for c in t.calls]
    # ceil(size / chunk) appends of at most one chunk each, then the extraction with no stdin.
    assert sizes[-1] is None and all(s is not None and s <= 2048 for s in sizes[:-1])
    assert len(sizes) == -(-(512 + 5120 + 1024) // 2048) + 1
    assert t.calls[-1][0][4] == str(tmp_path / "d2")
    assert t.calls[-1][0][5].startswith(str(tmp_path / "d2") + ".upload.")


def test_chunks_are_produced_lazily(tmp_path: Path) -> None:
    (tmp_path / "big").write_bytes(b"z" * 1_000_000)
    counts = _bulk._Counts()
    it = _bulk.chunks(_bulk._all_pieces(str(tmp_path), counts), 4096)
    data, last = next(it)
    assert len(data) == 4096 and not last
    assert counts.bytes < 100_000, "the file is read as the chunks are asked for"


def test_sources_list_uploads_by_base_name(tmp_path: Path) -> None:
    s = tmp_path / "s"
    (s / "d").mkdir(parents=True)
    (s / "a.txt").write_bytes(b"a")
    (s / "skip").write_bytes(b"no")
    (s / "d" / "inner").write_bytes(b"in")
    sb = Sandbox(HostTransport())  # type: ignore[arg-type]
    sb.upload([s / "a.txt", str(s / "d")], str(tmp_path / "dst"))
    assert sorted(tree(tmp_path / "dst")) == ["a.txt", "d", "d/inner"]


def test_failures(tmp_path: Path) -> None:
    sb = Sandbox(HostTransport())  # type: ignore[arg-type]
    (tmp_path / "f").write_bytes(b"file")
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "x").write_bytes(b"x")
    with pytest.raises(NotADirectory):
        sb.upload(tmp_path / "s", str(tmp_path / "f"))
    assert (tmp_path / "f").read_bytes() == b"file"
    for bad in ("relative/dir", "/"):
        with pytest.raises(InvalidRequest):
            sb.upload(tmp_path / "s", bad)
    os.mkfifo(tmp_path / "s" / "pipe")
    with pytest.raises(InvalidRequest, match="pipe"):
        sb.upload(tmp_path / "s", str(tmp_path / "d"))
    assert not (tmp_path / "d").exists()
    with pytest.raises(NotFound):
        sb.download(str(tmp_path / "nope"), tmp_path / "back")
    with pytest.raises(InvalidRequest):
        sb.download("nope", tmp_path / "back")
    with pytest.raises(NotFound):
        sb.upload(tmp_path / "missing", str(tmp_path / "d"))


def test_download_of_one_file_and_of_one_byte_frames(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_bytes(b"hello")
    (tmp_path / "-rf").write_bytes(b"x")
    sb = Sandbox(HostTransport(frame=1))  # type: ignore[arg-type]
    sb.download(str(tmp_path / "note.txt"), tmp_path / "o1")
    sb.download(str(tmp_path / "-rf"), tmp_path / "o2")
    assert (tmp_path / "o1" / "note.txt").read_bytes() == b"hello"
    assert (tmp_path / "o2" / "-rf").read_bytes() == b"x"


def test_an_unsafe_member_closes_the_stream(tmp_path: Path) -> None:
    class Hostile(HostTransport):
        def stream(self, command: str, args: Sequence[str] = (), **kw: Any) -> Any:
            self.frames = _Frames(raw_tar([f("../escape.txt")]), b"", 0, 100)
            return self.frames

    t = Hostile()
    with pytest.raises(ProtocolError, match="unsafe archive member"):
        Sandbox(t).download("/x", tmp_path / "local")  # type: ignore[arg-type]
    assert t.frames.closed and not (tmp_path / "escape.txt").exists()


def test_a_failed_download_reports_the_command(tmp_path: Path) -> None:
    class Failing(HostTransport):
        def stream(self, command: str, args: Sequence[str] = (), **kw: Any) -> Any:
            return _Frames(b"", b"tar: x: Cannot open: Permission denied\n", 2, 10)

    with pytest.raises(
        Exception, match="download /x: tar: x: Cannot open: Permission denied"
    ) as ei:
        Sandbox(Failing()).download("/x", tmp_path / "l")  # type: ignore[arg-type]
    assert ei.value.kind == "permission_denied"  # type: ignore[attr-defined]


def test_async_matches_sync(tmp_path: Path) -> None:
    src = tmp_path / "src"
    make_tree(src)

    async def go() -> tuple[Transfer, Transfer]:
        sb = AsyncSandbox(AsyncHostTransport(HostTransport(frame=1000)))  # type: ignore[arg-type]
        up = await sb.upload(src, str(tmp_path / "remote"), chunk_bytes=20_000)
        return up, await sb.download(str(tmp_path / "remote"), tmp_path / "back")

    up, down = asyncio.run(go())
    assert up == down and tree(src) == tree(tmp_path / "back")
