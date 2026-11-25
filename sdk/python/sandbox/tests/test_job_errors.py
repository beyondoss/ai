"""Typed job errors and the wire options (hold, keep, seen, grace_ms, heartbeat), against a scripted endpoint.

Each typed error is a subclass of the kind it was before, so an existing ``except`` keeps working.
The 429 case (16 writes already waiting) cannot be made by the one-request-at-a-time conformance
driver, so it is here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_transport import Endpoint, reply

from beyond_sandbox import (
    AsyncHttpTransport,
    ConflictError,
    DiskFullError,
    HttpTransport,
    InvalidRequestError,
    IoError,
    JobExistsError,
    JobGoneError,
    Jobs,
    KeyConflictError,
    NotFoundError,
    StdinBusyError,
    StdinClosedError,
    StdinGapError,
    _wire,
)
from beyond_sandbox.jobs import AsyncJobs

NDJSON = "application/x-ndjson"


def jobs_over(handler: Any) -> tuple[Jobs, Endpoint]:
    e = Endpoint()
    e.handler = handler
    return Jobs(HttpTransport(e.url)), e


def test_start_409_key_conflict_is_its_own_conflict() -> None:
    sb, _ = jobs_over(
        lambda h, b: reply(
            h,
            409,
            {"error": "the key was used for a different command or id", "code": "key_conflict"},
        )
    )
    with pytest.raises(KeyConflictError) as e:
        sb.start("s", "aa", "true")
    assert isinstance(e.value, ConflictError) and not isinstance(e.value, JobExistsError)
    assert e.value.id == "aa"


def test_start_409_and_507_are_typed() -> None:
    sb, _ = jobs_over(lambda h, b: reply(h, 409, {"error": "the job exists"}))
    with pytest.raises(JobExistsError) as exists:
        sb.start("s", "aa", "true")
    assert isinstance(exists.value, ConflictError) and exists.value.kind == "conflict"
    assert exists.value.id == "aa"
    sb, _ = jobs_over(lambda h, b: reply(h, 507, {"error": "the sandbox's disk is full"}))
    with pytest.raises(DiskFullError) as full:
        sb.start("s", "aa", "true")
    assert isinstance(full.value, IoError) and full.value.kind == "io"


def test_attach_404_and_read_gone_are_the_same_error() -> None:
    sb, _ = jobs_over(lambda h, b: reply(h, 404, {"error": "no such job"}))
    with pytest.raises(JobGoneError) as g:
        list(sb.follow("s", "aa"))
    assert isinstance(g.value, NotFoundError)

    gone = {"state": "gone", "code": None, "offset": 0, "end": 0, "data": ""}
    sb, _ = jobs_over(lambda h, b: reply(h, 200, gone))
    with pytest.raises(JobGoneError):
        sb.read("s", "aa")


def test_stdin_conditions_are_told_apart_as_the_reference_client_does() -> None:
    def write(status: int, body: dict[str, Any]) -> Exception:
        sb, _ = jobs_over(lambda h, b: reply(h, status, body))
        with pytest.raises(ConflictError if status != 404 else NotFoundError) as e:
            sb.write_stdin("s", "aa", 0, b"x")
        return e.value

    never = write(409, {"error": "the job has no stdin"})
    assert isinstance(never, StdinClosedError) and never.reason == "never_opened"
    closed = write(409, {"error": "stdin is closed", "written": 7})
    assert isinstance(closed, StdinClosedError) and closed.reason == "closed"
    assert closed.written == 7
    gap = write(409, {"error": "gap", "written": 2})
    assert isinstance(gap, StdinGapError) and gap.written == 2
    busy = write(429, {"error": "too many writes wait"})
    assert isinstance(busy, StdinBusyError) and not isinstance(busy, StdinClosedError)
    assert isinstance(write(404, {"error": "no such job"}), JobGoneError)


def test_keep_seen_and_grace_ms_go_out_under_their_wire_names() -> None:
    sb, e = jobs_over(lambda h, b: reply(h, 200, {"survivors": []}))
    sb.keep("s", "aa", num=3, kind="background", seen=120)
    sb.seen("s", "aa", 9)
    sb.kill("s", ["aa"], remove=True, grace_ms=2500)
    sb.wipe("s", grace_ms=7)
    sb.kill("s", ["aa"])
    assert [r["job"] for r in e.requests] == [
        {"op": "keep", "scope": "s", "id": "aa", "num": 3, "kind": "background", "seen": 120},
        {"op": "seen", "scope": "s", "id": "aa", "offset": 9},
        {"op": "kill", "scope": "s", "ids": ["aa"], "remove": True, "grace_ms": 2500},
        {"op": "wipe", "scope": "s", "grace_ms": 7},
        {"op": "kill", "scope": "s", "ids": ["aa"], "remove": False, "grace_ms": 300},
    ]
    with pytest.raises(InvalidRequestError):
        sb.kill("s", ["aa"], grace_ms=-1)


def held_stream(h: Any, b: Any) -> None:
    body = b'{"started":42}\n{"exit_code":0}\n'
    reply(h, 200, body, NDJSON)


def test_a_held_start_sends_hold_true_stream_and_the_heartbeat() -> None:
    sb, e = jobs_over(held_stream)
    frames = list(sb.start_held("s", "aa", "true", heartbeat_seconds=60))
    assert [type(f).__name__ for f in frames] == ["JobStarted", "JobEnd"]
    assert e.requests[0]["stream"] is True
    assert e.requests[0]["job"]["hold"] is True
    assert e.beats == ["60"]


def test_the_heartbeat_is_1_to_240_whole_seconds_and_checked_before_anything_is_sent() -> None:
    e = Endpoint()
    e.handler = lambda h, b: reply(h, 200, {"jobs": []})
    t = HttpTransport(e.url)
    t.job_call({"op": "list", "scope": "s"}, heartbeat_seconds=240)
    t.job_call({"op": "list", "scope": "s"})
    assert e.beats == ["240", None]
    for bad in (0, 241, 1.5, -1, True):
        with pytest.raises(InvalidRequestError):
            t.job_call({"op": "list", "scope": "s"}, heartbeat_seconds=bad)  # type: ignore[arg-type]
    assert len(e.requests) == 2


def test_the_silence_limit_follows_the_heartbeat_the_request_chose() -> None:
    assert _wire.silence_s() == 30.0
    assert _wire.silence_s(1) == 30.0
    assert _wire.silence_s(5) == 30.0
    # A 60 s beat would be killed by a 30 s limit; two beats and slack hold.
    assert _wire.silence_s(60) == 130.0
    assert _wire.silence_s(240) == 490.0


def test_async_jobs_carry_the_same_options() -> None:
    e = Endpoint()
    e.handler = lambda h, b: reply(h, 200, {"survivors": []})

    async def go() -> None:
        sb = AsyncJobs(AsyncHttpTransport(e.url))
        await sb.kill("s", ["aa"], grace_ms=11)
        await sb.keep("s", "aa", num=1)
        await sb.seen("s", "aa", 4)

    asyncio.run(go())
    assert [r["job"]["op"] for r in e.requests] == ["kill", "keep", "seen"]
    assert e.requests[0]["job"]["grace_ms"] == 11
