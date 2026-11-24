"""The job mark on the wire, against a scripted endpoint: which incarnation of an id a client talks to.

``mark`` on start, list and the streams' first frame; an attach and a remove that name a mark;
a status-only attach; what ``remove`` skipped; ids with ``-`` and ``_``. An endpoint that predates
the mark (no echo) is ``NotOfferedError``, never a silent success.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_transport import Endpoint, reply

from beyond_sandbox import (
    AsyncHttpTransport,
    HttpTransport,
    InvalidRequestError,
    JobAttached,
    JobEnd,
    JobGoneError,
    JobRemoved,
    Jobs,
    JobStart,
    JobStarted,
    NotOfferedError,
)
from beyond_sandbox.jobs import AsyncJobs

NDJSON = "application/x-ndjson"


def jobs_over(handler: Any) -> tuple[Jobs, Endpoint]:
    e = Endpoint()
    e.handler = handler
    return Jobs(HttpTransport(e.url)), e


def test_start_answers_the_pgid_and_the_mark() -> None:
    sb, _ = jobs_over(lambda h, b: reply(h, 200, {"started": 7, "mark": "ab12"}))
    assert sb.start("s", "aa", "true") == JobStart(7, "ab12")


def test_a_start_without_a_mark_is_not_offered() -> None:
    sb, _ = jobs_over(lambda h, b: reply(h, 200, {"started": 7}))
    with pytest.raises(NotOfferedError):
        sb.start("s", "aa", "true")


def test_a_held_start_names_its_mark_in_started_and_without_it_is_not_offered() -> None:
    sb, _ = jobs_over(
        lambda h, b: reply(h, 200, b'{"started":4,"mark":"m1"}\n{"exit_code":0}\n', NDJSON)
    )
    assert list(sb.start_held("s", "aa", "true")) == [JobStarted(4, False, "m1"), JobEnd(0)]
    sb, _ = jobs_over(lambda h, b: reply(h, 200, b'{"started":4}\n{"exit_code":0}\n', NDJSON))
    with pytest.raises(NotOfferedError):
        list(sb.start_held("s", "aa", "true"))


def test_list_reads_the_mark_the_command_and_the_output_window() -> None:
    entry = {
        "id": "a-b_c", "mark": "m9", "command": "sh", "args": ["-c", "x"], "outputStart": 3,
        "outputEnd": 8, "state": "running", "code": None, "kept": False, "num": None,
        "kind": None, "seen": 0, "started": 1, "ended": 0, "label": "",
    }  # fmt: skip
    sb, _ = jobs_over(lambda h, b: reply(h, 200, {"jobs": [entry]}))
    [j] = sb.list("s")
    assert (j.id, j.mark, j.command, j.args, j.output_start, j.output_end) == (
        "a-b_c", "m9", "sh", ("-c", "x"), 3, 8,
    )  # fmt: skip


def test_ids_are_1_to_64_of_lowercase_digits_dash_and_underscore() -> None:
    sb, e = jobs_over(lambda h, b: reply(h, 200, {"started": 1, "mark": "m"}))
    for ok in ("a", "a-b_c", "z" * 64):
        sb.start("s", ok, "true")
    for bad in ("", "A", "a.b", "a/b", "z" * 65):
        with pytest.raises(InvalidRequestError):
            sb.start("s", bad, "true")
    assert len(e.requests) == 3


def attach_stream(h: Any, b: Any) -> None:
    body = b'{"attached":{"mark":"m1","from":3}}\n{"stdout":"aGk="}\n{"exit_code":0}\n'
    reply(h, 200, body, NDJSON)


def test_attach_yields_attached_first_and_sends_the_mark_and_output_options() -> None:
    sb, e = jobs_over(attach_stream)
    frames = list(sb.follow("s", "aa", 3, mark="m1", output=False))
    assert frames[0] == JobAttached("m1", 3)
    job = e.requests[0]["job"]
    assert (job["mark"], job["output"], job["offset"]) == ("m1", False, 3)
    # Defaults send neither field.
    list(sb.follow("s", "aa"))
    assert "mark" not in e.requests[1]["job"] and "output" not in e.requests[1]["job"]
    a = sb.collect("s", "aa")
    assert (a.attached, a.output, a.exit_code) == (JobAttached("m1", 3), b"hi", 0)


def test_an_attach_that_does_not_open_with_attached_is_not_offered() -> None:
    sb, _ = jobs_over(lambda h, b: reply(h, 200, b'{"stdout":"aGk="}\n{"exit_code":0}\n', NDJSON))
    with pytest.raises(NotOfferedError):
        list(sb.follow("s", "aa"))


def test_an_attach_naming_a_replaced_incarnation_is_job_gone() -> None:
    sb, _ = jobs_over(lambda h, b: reply(h, 404, {"error": "no such job"}))
    with pytest.raises(JobGoneError):
        list(sb.follow("s", "aa", mark="old"))


def test_remove_sends_marks_and_reports_what_it_skipped() -> None:
    sb, e = jobs_over(lambda h, b: reply(h, 200, {"running": ["r"], "mismatch": ["m"]}))
    got = sb.remove("s", ["r", "m", "x"], marks={"m": "old"})
    assert got == JobRemoved(("r",), ("m",))
    job = e.requests[0]["job"]
    assert (job["ids"], job["marks"]) == (["r", "m", "x"], {"m": "old"})
    sb.remove("s", ["r"])
    assert "marks" not in e.requests[1]["job"]
    with pytest.raises(InvalidRequestError):
        sb.remove("s", ["r"], marks={"r": ""})


def test_a_remove_that_reports_nothing_is_not_offered() -> None:
    sb, _ = jobs_over(lambda h, b: reply(h, 200, {}))
    with pytest.raises(NotOfferedError):
        sb.remove("s", ["r"])


def test_the_async_client_does_the_same() -> None:
    async def main() -> None:
        e = Endpoint()

        def handler(h: Any, b: Any) -> None:
            op = b["job"]["op"]
            if op == "attach":
                attach_stream(h, b)
            elif op == "remove":
                reply(h, 200, {"running": [], "mismatch": ["m"]})
            else:
                reply(h, 200, {"started": 7, "mark": "ab12"})

        e.handler = handler
        t = AsyncHttpTransport(e.url)
        try:
            jobs = AsyncJobs(t)
            assert await jobs.start("s", "aa", "true") == JobStart(7, "ab12")
            a = await jobs.collect("s", "aa", mark="m1", output=False)
            assert a.attached == JobAttached("m1", 3)
            assert await jobs.remove("s", ["m"], marks={"m": "x"}) == JobRemoved((), ("m",))
            assert e.requests[1]["job"]["output"] is False
        finally:
            await t.aclose()

    asyncio.run(main())
