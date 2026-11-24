"""The start's idempotency key on the wire: one per ``start()`` call, the caller's when given, checked."""

from __future__ import annotations

import re
from typing import Any

import pytest
from test_transport import Endpoint, reply

from beyond_sandbox import HttpTransport, InvalidRequestError, Jobs


def jobs_over() -> tuple[Jobs, Endpoint]:
    e = Endpoint()
    e.handler = lambda h, b: reply(h, 200, {"started": 7})
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
