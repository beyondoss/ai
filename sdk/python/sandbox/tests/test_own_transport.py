"""Layers 2 and 3 over a transport that is not HTTP: the helpers need only ``run``.

``LocalTransport`` is the "bring your own transport" example of the README: commands run in a local
subprocess. Files and search work on it unchanged.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from beyond_sandbox import (
    AsyncFiles,
    AsyncFrameStream,
    Conflict,
    ExecResult,
    Files,
    FrameStream,
    NotFound,
    Sandbox,
)
from beyond_sandbox.atransport import AsyncJobFrameStream
from beyond_sandbox.transport import JobFrameStream


class LocalTransport:
    """Layer 1 in a dozen lines. ``calls`` records (command, stdin size) per request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = 120_000,
        stdin: bytes | None = None,
    ) -> ExecResult:
        self.calls.append((command, None if stdin is None else len(stdin)))
        try:
            p = subprocess.run(
                [command, *args], cwd=cwd, input=stdin or b"", capture_output=True,
                timeout=timeout_ms / 1000,
            )  # fmt: skip
        except FileNotFoundError as e:
            return ExecResult(127, b"", str(e).encode())
        return ExecResult(
            p.returncode if p.returncode >= 0 else 128 - p.returncode, p.stdout, p.stderr
        )

    def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> FrameStream:
        raise NotImplementedError("this transport only runs commands")

    def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> Any:
        raise NotImplementedError("this transport only runs commands")

    def job_stream(self, job: dict[str, Any]) -> JobFrameStream:
        raise NotImplementedError("this transport only runs commands")


class AsyncLocalTransport:
    async def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = 120_000,
        stdin: bytes | None = None,
    ) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            command, *args, cwd=cwd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        out, err = await proc.communicate(stdin or b"")
        code = proc.returncode or 0
        return ExecResult(code if code >= 0 else 128 - code, out, err)

    async def stream(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> AsyncFrameStream:
        raise NotImplementedError("this transport only runs commands")

    async def job_call(
        self,
        job: dict[str, Any],
        command: str = "true",
        args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> Any:
        raise NotImplementedError("this transport only runs commands")

    async def job_stream(self, job: dict[str, Any]) -> AsyncJobFrameStream:
        raise NotImplementedError("this transport only runs commands")


def test_files_over_a_subprocess(tmp_path: Path) -> None:
    f = Files(LocalTransport())
    assert f.capabilities().read_window
    p = str(tmp_path / "a.txt")
    f.write(p, b"hello\nworld\n")
    assert f.read(p) == b"hello\nworld\n"
    assert f.read(p, 6, 3) == b"wor"
    st = f.stat(p)
    assert st is not None and (st.kind, st.size, st.writable) == ("file", 12, True)
    assert f.stat(str(tmp_path / "missing")) is None
    f.mkdir(str(tmp_path / "d" / "e"))
    assert sorted(e.name for e in f.list(str(tmp_path))) == ["a.txt", "d"]
    assert [(h.line, h.text) for h in f.search("wor", str(tmp_path)).hits] == [(2, "world")]
    with pytest.raises(NotFound):
        f.read(str(tmp_path / "missing"))
    with pytest.raises(Conflict):
        f.write_if_unchanged(p, b"x", (st.mtime if st else 0) + 100)
    f.write_if_unchanged(p, b"new", st.mtime if st else None)
    assert f.read(p) == b"new"
    # a write leaves no temp behind
    assert sorted(x.name for x in tmp_path.iterdir()) == ["a.txt", "d"]


def test_a_missing_parent_is_said_in_plain_words(tmp_path: Path) -> None:
    f = Files(LocalTransport())
    missing = str(tmp_path / "no" / "such" / "dir")
    with pytest.raises(NotFound) as e:
        f.write(f"{missing}/f", b"x")
    assert e.value.detail == (
        f"write {missing}/f: parent directory {missing} does not exist: "
        "No such file or directory (os error 2)"
    )
    assert ".tmp." not in e.value.detail
    assert not (tmp_path / "no").exists()


def test_a_large_write_is_one_request_with_stdin(tmp_path: Path) -> None:
    t = LocalTransport()
    f = Files(t)
    data = bytes(range(256)) * 600  # 150 KiB
    p = str(tmp_path / "big")
    f.write(p, data)
    assert f.read(p) == data
    assert [e.name for e in f.list(str(tmp_path))] == ["big"]
    writes = [c for c in t.calls if c[1] is not None]
    assert len(writes) == 1 and writes[0][1] == len(data)


def test_the_first_file_op_is_one_probe_and_one_command(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"hi")
    t = LocalTransport()
    assert Files(t).read(str(p)) == b"hi"
    assert len(t.calls) == 2  # the environment probe, then the read
    t = LocalTransport()
    Files(t).write(str(tmp_path / "g"), b"x")
    assert len(t.calls) == 1  # a write needs no probe


def test_the_sandbox_surface_keeps_raw_exec_in_reach(tmp_path: Path) -> None:
    sb = Sandbox(LocalTransport())
    assert sb.exec("echo", ["x"]).stdout == b"x\n"
    assert sb.transport.run("true").exit_code == 0
    sb.files.write(str(tmp_path / "f"), b"1")
    assert sb.capabilities().read_window


def test_a_user_helper_on_raw_exec_alone(tmp_path: Path) -> None:
    """The helper of the README: files changed since a marker, with no help from the library."""
    (tmp_path / "old").write_text("o")
    (tmp_path / "marker").write_text("m")
    (tmp_path / "new").write_text("n")
    import os

    os.utime(tmp_path / "old", (1, 1))
    os.utime(tmp_path / "marker", (1000, 1000))
    os.utime(tmp_path / "new", (2000, 2000))

    def changed_since(sb: Sandbox, root: str, marker: str) -> list[str]:
        r = sb.exec("find", [root, "-type", "f", "-newer", marker])
        return sorted(r.stdout_text.split())

    assert changed_since(Sandbox(LocalTransport()), str(tmp_path), str(tmp_path / "marker")) == [
        str(tmp_path / "new")
    ]


def test_async_files_over_an_async_subprocess(tmp_path: Path) -> None:
    async def main() -> None:
        t = AsyncLocalTransport()
        f = AsyncFiles(t)
        p = str(tmp_path / "a")
        await f.write(p, b"abc")
        assert await f.read(p) == b"abc"
        assert [h.text for h in (await f.search("b", str(tmp_path))).hits] == ["abc"]

    asyncio.run(main())
