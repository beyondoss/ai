"""The package against a real loopback execd: exec, stream, files and jobs, sync and async.

Needs the endpoint binary (``cargo build -p beyond-execd``, or ``EXECD_BIN``); skipped without it.
The shared conformance matrix is the full proof; this is what the package proves by itself.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from beyond_sandbox import AsyncSandbox, Exit, Sandbox, Signal, Stderr, Stdout


def test_exec_and_stream(execd_url: str) -> None:
    sb = Sandbox.connect(execd_url)
    r = sb.exec("sh", ["-c", "echo out; echo err >&2; exit 7"])
    assert (r.exit_code, r.stdout, r.stderr) == (7, b"out\n", b"err\n")
    raw = sb.exec("cat", stdin=b"\x00\xff bytes")  # bytes, exactly: invalid UTF-8 and NUL survive
    assert raw.stdout == b"\x00\xff bytes"

    t0 = time.monotonic()
    seen: list[tuple[float, object]] = []
    with sb.stream("sh", ["-c", "echo a; sleep 0.4; echo b >&2"]) as frames:
        for f in frames:
            seen.append((time.monotonic() - t0, f))
    assert [f for _, f in seen] == [Stdout(b"a\n"), Stderr(b"b\n"), Exit(0)]
    assert seen[1][0] - seen[0][0] > 0.2, "the first frame arrived before the command finished"


def test_stream_close_kills_the_command(execd_url: str, tmp_path: Path) -> None:
    sb = Sandbox.connect(execd_url)
    marker = tmp_path / "ticks"
    with sb.stream("sh", ["-c", f"while :; do echo x >> {marker}; sleep 0.05; done"]) as _frames:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
    time.sleep(1.0)  # the endpoint kills the group when the connection closes
    size = marker.stat().st_size
    time.sleep(0.5)
    assert marker.stat().st_size == size, "the command kept running after the stream was closed"


def test_a_signal_is_a_final_frame(execd_url: str) -> None:
    frames = list(Sandbox.connect(execd_url).stream("sh", ["-c", "kill -9 $$"]))
    assert frames[-1] == Signal(9)


def test_files(execd_url: str, tmp_path: Path) -> None:
    sb = Sandbox.connect(execd_url)
    assert sb.capabilities().read_window
    p = str(tmp_path / "f.txt")
    sb.files.write(p, bytes(range(256)))
    assert sb.files.read(p) == bytes(range(256))
    assert [e.name for e in sb.files.list(str(tmp_path))] == ["f.txt"]


def test_jobs(execd_url: str) -> None:
    sb = Sandbox.connect(execd_url)
    scope = f"pytest{time.monotonic_ns()}"
    try:
        sb.jobs.start(scope, "a1", "sh", ["-c", "echo one; sleep 0.2; echo two; exit 5"], label="x")
        a = sb.jobs.attach(scope, "a1")
        assert (a.exit_code, a.output) == (5, b"one\ntwo\n")
        [j] = sb.jobs.list(scope)
        assert (j.id, j.state, j.code, j.label) == ("a1", "ended", 5, "x")
        sb.jobs.start(scope, "a2", "sleep", ["30"])
        assert sb.jobs.kill(scope, ["a2"]) == []
        assert sb.jobs.read(scope, "a2").state == "ended"
    finally:
        sb.jobs.wipe(scope)
    assert sb.jobs.list(scope) == []


def test_async_sandbox(execd_url: str, tmp_path: Path) -> None:
    async def main() -> None:
        async with AsyncSandbox.connect(execd_url) as sb:
            assert (await sb.exec("echo", ["hi"])).stdout == b"hi\n"
            frames = [f async for f in await sb.stream("sh", ["-c", "echo a; echo b >&2"])]
            assert {Stdout(b"a\n"), Stderr(b"b\n"), Exit(0)} == set(frames)
            p = str(tmp_path / "x")
            await sb.files.write(p, b"async")
            assert await sb.files.read(p) == b"async"
            scope = f"pyasync{time.monotonic_ns()}"
            await sb.jobs.start(scope, "b1", "sh", ["-c", "echo hey; exit 1"])
            a = await sb.jobs.attach(scope, "b1")
            assert (a.exit_code, a.output) == (1, b"hey\n")
            await sb.jobs.wipe(scope)

    asyncio.run(main())
