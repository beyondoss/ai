"""The conformance driver's envelope, without an endpoint: hello, unknown ops, bad requests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from beyond_sandbox import driver

SRC = str(Path(__file__).resolve().parents[1] / "src")


def ask(d: driver.Driver, **req: Any) -> dict[str, Any]:
    answer: dict[str, Any] = d.handle(json.dumps(req))
    return answer


@pytest.fixture
def d(monkeypatch: pytest.MonkeyPatch) -> driver.Driver:
    monkeypatch.setenv("CONFORMANCE_URL", "http://127.0.0.1:1/")  # nothing listens: never reached
    return driver.Driver(use_async=False)


def test_hello_does_not_reach_the_endpoint(d: driver.Driver) -> None:
    out = ask(d, id=1, op="hello")
    assert out["id"] == 1 and out["ok"]["driver"] == "python" and out["ok"]["protocol"] == 1
    assert "exec" in out["ok"]["ops"] and "search" in out["ok"]["ops"]


def test_an_unknown_op_is_unsupported_never_a_silent_pass(d: driver.Driver) -> None:
    for op in ("upload_tar", "remove", "frobnicate", ""):
        assert ask(d, id="x", op=op)["error"]["kind"] == "unsupported", op


@pytest.mark.parametrize(
    "req",
    [
        {"op": "stat"},
        {"op": "stat", "path": 3},
        {"op": "read", "path": "/x", "offset": -1},
        {"op": "read", "path": "/x", "max": 1.5},
        {"op": "read", "path": "/x", "max": True},
        {"op": "write", "path": "/x"},
        {"op": "write", "path": "/x", "data_b64": "!!!"},
        {"op": "write_if_unchanged", "path": "/x", "data_b64": ""},
        {"op": "write_if_unchanged", "path": "/x", "data_b64": "", "expected_mtime": "1"},
        {"op": "exec", "command": "true", "args": "x"},
        {"op": "exec", "command": "true", "args": [1]},
        {"op": "list", "path": "/x", "include_hidden": "yes"},
        {"op": "search", "pattern": "x"},
        {"op": "job_start", "scope": "s", "job": "UPPER", "command": "true"},
        {"op": "job_start", "scope": "s", "job": "../x", "command": "true"},
        {"op": "job_read", "scope": "s", "job": ""},
        {"op": "job_kill", "scope": "s", "jobs": ["Bad"]},
        {"op": "job_remove", "scope": "s", "jobs": ["Bad"]},
    ],
)
def test_a_bad_request_is_invalid_request_and_sends_nothing(
    d: driver.Driver, req: dict[str, Any]
) -> None:
    # the endpoint is unreachable, so anything sent would be a `transport` error instead
    assert ask(d, id=0, **req)["error"]["kind"] == "invalid_request"


def test_a_line_that_is_not_json_is_answered(d: driver.Driver) -> None:
    assert d.handle("not json")["error"]["kind"] == "invalid_request"


def test_unreachable_is_a_transport_error(d: driver.Driver) -> None:
    assert ask(d, id=0, op="exec", command="true")["error"]["kind"] == "transport"
    assert ask(d, id=0, op="capabilities")["error"]["kind"] == "transport"


def run_driver(*args: str, stdin: str) -> list[dict[str, Any]]:
    p = subprocess.run(
        [sys.executable, "-m", "beyond_sandbox.driver", *args], input=stdin, capture_output=True,
        text=True, timeout=60, env={"PYTHONPATH": SRC, "CONFORMANCE_URL": "http://127.0.0.1:1/"},
    )  # fmt: skip
    assert p.returncode == 0, p.stderr
    return [json.loads(line) for line in p.stdout.splitlines()]


@pytest.mark.parametrize(("flag", "name"), [("--sync", "python"), ("--async", "python-async")])
def test_the_process_speaks_json_lines_and_exits_when_stdin_closes(flag: str, name: str) -> None:
    out = run_driver(flag, stdin='{"id":1,"op":"hello"}\n\n{"id":2,"op":"nope"}\n')
    assert [o["id"] for o in out] == [1, 2]
    assert out[0]["ok"]["driver"] == name and out[1]["error"]["kind"] == "unsupported"
