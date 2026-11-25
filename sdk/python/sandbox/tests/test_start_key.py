"""The start's idempotency key on the wire: one per ``start()`` call, the caller's when given, checked."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from test_transport import Endpoint, reply

from beyond_sandbox import (
    AsyncHttpTransport,
    HttpTransport,
    InvalidRequestError,
    Jobs,
    NotOfferedError,
)
from beyond_sandbox.jobs import AsyncJobs


def jobs_over() -> tuple[Jobs, Endpoint]:
    e = Endpoint()
    e.handler = lambda h, b: reply(h, 200, {"started": 7, "key": b["job"]["idempotency_key"]})
    return Jobs(HttpTransport(e.url)), e


def ignoring_keys() -> tuple[Jobs, Endpoint]:
    """An endpoint that predates start keys: it starts the job and says nothing of the key."""
    e = Endpoint()
    e.handler = lambda h, b: reply(
        h, 200, {"started": 7} if b["job"]["op"] == "start" else {"survivors": []}
    )
    return Jobs(HttpTransport(e.url)), e


def keys(e: Endpoint) -> list[Any]:
    return [r["job"].get("idempotency_key") for r in e.requests]


def test_each_start_call_sends_a_fresh_valid_key() -> None:
    sb, e = jobs_over()
    sb.start("s", "aa", "true")
    sb.start("s", "ab", "true")
    first, second = keys(e)
    assert first != second
    for k in (first, second):
        assert isinstance(k, str) and re.fullmatch(r"[0-9A-Za-z._:-]{1,128}", k)


def test_the_callers_key_is_sent_as_given() -> None:
    sb, e = jobs_over()
    sb.start("s", "aa", "true", idempotency_key="deploy-2026-11-24")
    sb.start("s", "ab", "true", idempotency_key="deploy-2026-11-24")
    assert keys(e) == ["deploy-2026-11-24", "deploy-2026-11-24"]


@pytest.mark.parametrize("bad", ["", "a b", "k/1", "k" * 129])
def test_a_bad_key_is_refused_before_anything_is_sent(bad: str) -> None:
    sb, e = jobs_over()
    with pytest.raises(InvalidRequestError):
        sb.start("s", "aa", "true", idempotency_key=bad)
    assert e.requests == []


def test_an_endpoint_that_ignores_the_key_is_not_offered_and_what_it_started_is_stopped() -> None:
    sb, e = ignoring_keys()
    with pytest.raises(NotOfferedError):
        sb.start("s", "aa", "true")
    assert [r["job"]["op"] for r in e.requests] == ["start", "kill"]


def test_a_held_start_whose_first_frame_does_not_echo_the_key_is_not_offered() -> None:
    e = Endpoint()
    e.handler = lambda h, b: reply(
        h, 200, b'{"started":42}\n{"exit_code":0}\n', "application/x-ndjson"
    )
    sb = Jobs(HttpTransport(e.url))
    with pytest.raises(NotOfferedError):
        list(sb.start_held("s", "aa", "true"))


def test_async_jobs_fail_closed_on_a_missing_echo_too() -> None:
    e = Endpoint()
    e.handler = lambda h, b: reply(
        h, 200, {"started": 7} if b["job"]["op"] == "start" else {"survivors": []}
    )
    held = Endpoint()
    held.handler = lambda h, b: reply(
        h, 200, b'{"started":42}\n{"exit_code":0}\n', "application/x-ndjson"
    )

    async def go() -> None:
        with pytest.raises(NotOfferedError):
            await AsyncJobs(AsyncHttpTransport(e.url)).start("s", "aa", "true")
        with pytest.raises(NotOfferedError):
            async for _ in AsyncJobs(AsyncHttpTransport(held.url)).start_held("s", "aa", "true"):
                pass

    asyncio.run(go())
    assert [r["job"]["op"] for r in e.requests] == ["start", "kill"]
