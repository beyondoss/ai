"""Shared fixtures: a loopback execd (the real endpoint) and a scripted fake endpoint."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]


def execd_binary() -> Path | None:
    """``$EXECD_BIN``, else a built ``target/debug|release/execd`` of this repo, else ``execd``."""
    if env := os.environ.get("EXECD_BIN"):
        return Path(env)
    for profile in ("debug", "release"):
        candidate = REPO / "target" / profile / "execd"
        if candidate.is_file():
            return candidate
    found = shutil.which("execd")
    return Path(found) if found else None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session")
def execd_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A loopback execd with no bearer (`cargo build -p beyond-execd`, or set EXECD_BIN)."""
    binary = execd_binary()
    if binary is None:
        pytest.skip("no execd binary: cargo build -p beyond-execd, or set EXECD_BIN")
    port = free_port()
    root = tmp_path_factory.mktemp("execd-root")
    env = {**os.environ, "EXECD_LISTEN": f"127.0.0.1:{port}", "EXECD_ROOT": str(root)}
    env["EXECD_LOG"] = "off"
    proc = subprocess.Popen(
        [str(binary)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("execd did not start listening")
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
