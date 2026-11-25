"""The start's idempotency key on the wire: one per ``start()`` call, the caller's when given, checked."""

from __future__ import annotations

import re
from typing import Any

import pytest
from test_transport import Endpoint, reply

from beyond_sandbox import (
    HttpTransport,
    InvalidRequestError,
    Jobs,
    KeyConflictError,
    NotOfferedError,
)


def jobs_over(echo: bool = True) -> tuple[Jobs, Endpoint]:
    """An endpoint that keeps start keys (it echoes the one it was sent), or one that ignores them."""
    e = Endpoint()
    e.handler = lambda h, b: reply(
        h,
        200,
        {"started": 7, "mark": "ab"}
        | ({"key": b["job"]["idempotency_key"]} if echo and b["job"]["op"] == "start" else {})
        if b["job"]["op"] == "start"
        else {"survivors": []},
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


def test_an_endpoint_that_ignores_keys_is_not_offered_and_the_job_is_stopped() -> None:
    sb, e = jobs_over(echo=False)
    with pytest.raises(NotOfferedError):
        sb.start("s", "aa", "true")
    assert [r["job"]["op"] for r in e.requests] == ["start", "kill"]


def test_a_held_start_of_an_endpoint_that_ignores_keys_is_not_offered() -> None:
    e = Endpoint()
    e.handler = lambda h, b: reply(
        h, 200, b'{"started":42,"mark":"ab"}\n{"exit_code":0}\n', "application/x-ndjson"
    )
    with pytest.raises(NotOfferedError):
        list(Jobs(HttpTransport(e.url)).start_held("s", "aa", "true"))


def test_a_409_key_conflict_is_its_own_error() -> None:
    e = Endpoint()
    e.handler = lambda h, b: reply(
        h, 409, {"error": "the key was used for a different command or id", "code": "key_conflict"}
    )
    with pytest.raises(KeyConflictError):
        Jobs(HttpTransport(e.url)).start("s", "aa", "true")
