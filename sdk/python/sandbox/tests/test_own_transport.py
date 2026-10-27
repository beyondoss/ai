"""Layers 2 and 3 over a transport that is not HTTP: the helpers need only ``run``.

``LocalTransport`` is the "bring your own transport" example of the README: commands run in a local
subprocess. Files, search and the baseline jobs all work on it unchanged.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from beyond_sandbox import (
    AsyncFiles,
    AsyncFrameStream,
    AsyncJobs,
    Conflict,
    ExecResult,
    Files,
    FrameStream,
    Jobs,
    NotFound,
    NotOffered,
    Sandbox,
)


class LocalTransport:
    """Layer 1 in a dozen lines."""

    def run(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | None = None,
        timeout_ms: int = 120_000,
        stdin: bytes | None = None,
    ) -> ExecResult:
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
        raise NotOffered("this transport does not stream")


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
        raise NotOffered("this transport does not stream")


def test_files_over_a_subprocess(tmp_path: Path) -> None:
    f = Files(LocalTransport())
    caps = f.capabilities()
    assert caps.stdin and caps.read_window
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


def test_a_large_write_without_stdin_goes_in_48k_chunks(tmp_path: Path) -> None:
    f = Files(LocalTransport(), override={"stdin": False})
    data = bytes(range(256)) * 600  # 150 KiB: four chunks
    p = str(tmp_path / "big")
    f.write(p, data)
    assert f.read(p) == data
    assert [e.name for e in f.list(str(tmp_path))] == ["big"]


def test_baseline_jobs_over_a_subprocess(tmp_path: Path) -> None:
    jobs = Jobs(LocalTransport(), root=str(tmp_path / "jobs"))
    assert jobs.kind() == "baseline"
    pgid = jobs.start("scope1", "j1", "sh", ["-c", "echo hi; echo err >&2; exit 3"], label="demo")
    assert pgid > 0
    a = jobs.attach("scope1", "j1")
    assert (a.exit_code, a.output, a.gap) == (3, b"hi\nerr\n", 0)
    [info] = jobs.list("scope1")
    assert (info.id, info.state, info.code, info.label) == ("j1", "ended", 3, "demo")
    assert jobs.read("scope1", "j1").data == b"hi\nerr\n"
    assert jobs.read("scope1", "nope").state == "gone"
    jobs.remove("scope1", ["j1"])
    assert jobs.list("scope1") == []
    jobs.start("scope1", "j2", "sleep", ["30"])
    assert jobs.kill("scope1", ["j2"]) == []
    assert jobs.wipe("scope1") == []


def test_the_sandbox_surface_keeps_raw_exec_in_reach(tmp_path: Path) -> None:
    sb = Sandbox(LocalTransport())
    assert sb.exec("echo", ["x"]).stdout == b"x\n"
    assert sb.transport.run("true").exit_code == 0
    sb.files.write(str(tmp_path / "f"), b"1")
    assert sb.capabilities().stream is False
    with pytest.raises(NotOffered):
        sb.stream("true")


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


def test_async_files_and_jobs_over_an_async_subprocess(tmp_path: Path) -> None:
    async def main() -> None:
        t = AsyncLocalTransport()
        f = AsyncFiles(t)
        p = str(tmp_path / "a")
        await f.write(p, b"abc")
        assert await f.read(p) == b"abc"
        assert [h.text for h in (await f.search("b", str(tmp_path))).hits] == ["abc"]
        jobs = AsyncJobs(t, root=str(tmp_path / "jobs"))
        assert await jobs.kind() == "baseline"
        await jobs.start("s", "j", "sh", ["-c", "echo out; exit 2"])
        events = [e async for e in jobs.follow("s", "j")]
        assert [type(e).__name__ for e in events] == ["JobOutput", "JobEnd"]
        a = await jobs.attach("s", "j")
        assert (a.exit_code, a.output) == (2, b"out\n")
        assert await jobs.wipe("s") == []

    asyncio.run(main())
