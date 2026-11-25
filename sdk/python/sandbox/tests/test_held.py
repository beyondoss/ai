"""The job handle, run by the client kernel, against a scripted endpoint that cuts, refuses and misnames.

What a handle does across a failure (reattach, wait, give up, say which error) is the kernel's: these tests
observe it from outside. Port of the TypeScript SDK's ``held.test.ts``. ``test_held_broken_kernel.py`` runs
one of them against a library with a constant changed, and it goes red.
"""

from __future__ import annotations

import asyncio
import types
from collections.abc import Iterator
from typing import Any

import pytest
from held_server import MARK, Script, Server, serve

from beyond_sandbox import (
    AsyncHttpTransport,
    AuthError,
    ConnectError,
    DiskFullError,
    EndpointError,
    HttpTransport,
    JobEnd,
    JobExistsError,
    JobFrame,
    JobGoneError,
    Jobs,
    JobStarted,
    NotOfferedError,
    Reconnecting,
    TimeoutError,
    TransportError,
    _drive,
)
from beyond_sandbox import _kernel as K
from beyond_sandbox.jobs import AsyncJobs
from beyond_sandbox.models import JobGap, JobOutput


class Rig:
    def __init__(self, script: Script) -> None:
        self._gen = serve(script)
        self.server: Server = next(self._gen)
        self.jobs = Jobs(HttpTransport(self.server.url))

    def close(self) -> None:
        next(self._gen, None)


@pytest.fixture
def rig() -> Iterator[Any]:
    made: list[Rig] = []

    def make(script: Script) -> Rig:
        r = Rig(script)
        made.append(r)
        return r

    yield make
    for r in made:
        r.close()


def text(frames: list[JobFrame]) -> str:
    return b"".join(f.data for f in frames if isinstance(f, JobOutput)).decode()


def offsets(s: Server) -> list[int]:
    return [a["offset"] for a in s.attaches]


# ---- a followed job survives cuts ---------------------------------------------------------------


def test_a_cut_stream_is_reattached_at_the_offset_reached_with_the_mark(rig: Any) -> None:
    r = rig(Script("0123456789", cut_after=[3, 4]))
    frames = list(r.jobs.start_held("s", "aa", "true"))
    assert [type(f).__name__ for f in frames] == [
        "JobStarted", "JobOutput", "JobOutput", "JobOutput", "JobEnd",
    ]  # fmt: skip
    assert text(frames) == "0123456789"
    assert offsets(r.server) == [3, 7]
    assert all(a["mark"] == MARK for a in r.server.attaches)


def test_a_follow_from_a_later_offset_starts_there_and_holds_the_mark_it_learns(rig: Any) -> None:
    r = rig(Script("0123456789", cut_after=[2]))
    frames = list(r.jobs.follow("s", "aa", 4))
    assert text(frames) == "456789"
    assert "mark" not in r.server.attaches[0]
    assert (r.server.attaches[1]["offset"], r.server.attaches[1]["mark"]) == (6, MARK)


def test_a_gap_the_server_states_on_a_reattach_is_told_once(rig: Any) -> None:
    r = rig(Script("0123456789", base=5, cut_after=[0]))
    frames = list(r.jobs.follow("s", "aa", 0))
    assert [f for f in frames if isinstance(f, JobGap)] == [JobGap(5)]
    assert text(frames) == "56789"


def test_the_started_echo_the_shell_exit_and_the_first_attached_only_are_told(rig: Any) -> None:
    r = rig(Script("ab", shell_exit=3, cut_after=[1]))
    frames = list(r.jobs.follow("s", "aa", 0))
    # One `attached` for the follow, though the cut made a second echo; the shell's exit is passed on.
    assert [type(f).__name__ for f in frames] == [
        "JobAttached", "JobOutput", "JobOutput", "JobShellExit", "JobEnd",
    ]  # fmt: skip
    started = list(r.jobs.start_held("s", "aa", "true"))
    assert started[0] == JobStarted(42, False, MARK)


def test_a_busy_answer_is_waited_out_and_tried_again(rig: Any) -> None:
    r = rig(Script("abc", refuse=[(429, {"Retry-After": "0"})]))
    assert text(list(r.jobs.follow("s", "aa", 0))) == "abc"
    assert len(r.server.attaches) == 2


def test_a_cut_after_the_exit_frame_is_not_reattached(rig: Any) -> None:
    r = rig(Script("abc"))
    frames = list(r.jobs.follow("s", "aa", 0))
    assert isinstance(frames[-1], JobEnd) and len(r.server.attaches) == 1


def test_the_start_is_a_kept_job_answered_in_one_record(rig: Any) -> None:
    r = rig(Script("abc"))
    started = r.jobs.start("s", "aa", "true")
    assert (started.pgid, started.mark) == (42, MARK)
    assert r.server.starts[0]["hold"] is False


# ---- a refusal is one typed error, from the kernel's table -----------------------------------------


def follow_error(rig: Any, script: Script) -> BaseException | None:
    try:
        list(rig(script).jobs.follow("s", "aa", 0))
    except BaseException as e:
        return e
    return None


def test_404_is_job_gone(rig: Any) -> None:
    assert isinstance(follow_error(rig, Script("x", refuse=[(404, {})])), JobGoneError)


def test_401_and_403_are_auth_and_400_and_500_are_the_endpoint_with_the_status(rig: Any) -> None:
    assert isinstance(follow_error(rig, Script("x", refuse=[(401, {})])), AuthError)
    assert isinstance(follow_error(rig, Script("x", refuse=[(403, {})])), AuthError)
    e400 = follow_error(rig, Script("x", refuse=[(400, {})]))
    assert isinstance(e400, EndpointError) and e400.status == 400
    e500 = follow_error(rig, Script("x", refuse=[(500, {})]))
    assert isinstance(e500, EndpointError) and e500.status == 500


def test_a_start_409_is_job_exists_and_507_is_disk_full(rig: Any) -> None:
    for status, cls in ((409, JobExistsError), (507, DiskFullError)):
        r = rig(Script("x", start_status=status))
        with pytest.raises(cls):
            list(r.jobs.start_held("s", "aa", "true"))
        with pytest.raises(cls):
            r.jobs.start("s", "aa", "true")


def test_an_endpoint_that_does_not_name_the_mark_is_not_offered_before_any_byte(rig: Any) -> None:
    r = rig(Script("abc", echo=None))
    got: list[JobFrame] = []
    with pytest.raises(NotOfferedError):
        for f in r.jobs.follow("s", "aa", 0):
            got.append(f)
    assert got == []


def test_a_start_without_a_mark_is_not_offered_and_the_job_it_started_is_stopped(rig: Any) -> None:
    r = rig(Script("abc", echo=None))
    with pytest.raises(NotOfferedError):
        r.jobs.start("s", "aa", "true")
    # The kernel names the kill: the job that was started and cannot be told from another is stopped.
    assert [j["op"] for j in r.server.starts] == ["start"]
    assert [(k["ids"], k["remove"]) for k in r.server.kills] == [(["aa"], True)]


def test_a_reattach_answered_by_another_incarnation_is_job_gone_before_any_of_its_bytes(
    rig: Any,
) -> None:
    r = rig(Script("0123456789", cut_after=[2], reattach_echo="ffff"))
    got: list[JobFrame] = []
    with pytest.raises(JobGoneError):
        for f in r.jobs.start_held("s", "aa", "true"):
            got.append(f)
    # Only what the first incarnation sent before the cut: nothing of the other's 2..9.
    assert text(got) == "01"


def test_a_bad_mark_is_refused_before_anything_is_sent(rig: Any) -> None:
    from beyond_sandbox import InvalidRequestError

    r = rig(Script("abc"))
    with pytest.raises(InvalidRequestError):
        list(r.jobs.follow("s", "aa", 0, mark="not hex"))
    assert r.server.attaches == []


def test_the_heartbeat_the_request_chose_is_sent_and_stretches_the_silence_limit(rig: Any) -> None:
    r = rig(Script("abc"))
    list(r.jobs.follow("s", "aa", 0, heartbeat_seconds=60))
    assert r.server.beats == ["60"]


# ---- a handle that cannot reach the endpoint keeps trying and says so -----------------------------


def test_it_reports_reconnecting_from_the_12th_failed_open_on_and_finishes_when_the_endpoint_returns(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1_000_000]
    monkeypatch.setattr(K, "now_ms", lambda: clock[0])

    def sleep(seconds: float) -> None:
        clock[0] += round(seconds * 1000)

    monkeypatch.setattr(_drive, "time", types.SimpleNamespace(sleep=sleep))
    r = rig(Script("0123456789", cut_after=[4], refuse=[(503, {}) for _ in range(20)]))
    seen: list[Reconnecting] = []
    frames = list(r.jobs.start_held("s", "aa", "true", on_reconnecting=seen.append))
    # Never an end: all the bytes, once each, and the exit.
    assert text(frames) == "0123456789"
    assert frames[-1] == JobEnd(0)
    # 20 refused opens after the cut; the status starts at the 12th failure in a row and counts up.
    assert len(r.server.attaches) == 21
    assert [s.attempts for s in seen] == list(range(12, 22))
    assert all(s.since_ms > 0 for s in seen)


# ---- the async handle is the same handle ---------------------------------------------------------


def test_async_a_cut_stream_is_reattached_and_a_refusal_is_typed(rig: Any) -> None:
    r = rig(Script("0123456789", cut_after=[3], refuse=[(429, {"Retry-After": "0"})]))

    async def main() -> None:
        jobs = AsyncJobs(AsyncHttpTransport(r.server.url))
        frames = [f async for f in jobs.start_held("s", "aa", "true")]
        assert text(frames) == "0123456789" and isinstance(frames[-1], JobEnd)
        assert offsets(r.server) == [3, 3]  # cut at 3; the reattach is told to wait, then goes
        got = await jobs.collect("s", "aa", 0)
        assert got.output == b"0123456789"
        missing = AsyncJobs(AsyncHttpTransport(rig(Script("x", refuse=[(404, {})])).server.url))
        with pytest.raises(JobGoneError):
            await missing.collect("s", "aa")

    asyncio.run(main())


# ---- the shell does what the kernel says, nothing it does not ------------------------------------


class Scripted:
    """A kernel that says what a test tells it to. The handle must obey it exactly: a shell that decided
    anything itself would do something else."""

    def __init__(self, answers: dict[int, list[K.Action]]) -> None:
        self.answers = answers
        self.events: list[bytes] = []

    def state_for_attach(self, mark: str, offset: int, p: K.Policy) -> bytes:
        return b"s"

    def step(
        self, state: bytes, event: bytes, now_ms: int, draw: int
    ) -> tuple[bytes, list[K.Action]]:
        self.events.append(event)
        return state, self.answers.get(event[0], [])


def test_the_shell_obeys_the_kernel_even_against_what_the_endpoint_said(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = rig(Script("0123456789"))  # a healthy 200 stream
    open_ = K.OpenAttach(0, "", 30_000)
    stub = Scripted({0: [open_], 1: [K.Fail("job_gone", 0, 0)]})  # go -> open; answered -> fail
    monkeypatch.setattr(K, "_loaded", stub)
    with pytest.raises(JobGoneError):
        list(r.jobs.follow("s", "aa"))
    assert [e[0] for e in stub.events] == [0, 1]


def test_the_shell_neither_retries_nor_counts_when_the_kernel_finishes_on_a_busy_answer(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = rig(Script("abc", refuse=[(429, {"Retry-After": "0"})]))
    stub = Scripted({0: [K.OpenAttach(0, "", 30_000)], 1: [K.Finish()]})
    monkeypatch.setattr(K, "_loaded", stub)
    assert list(r.jobs.follow("s", "aa")) == []
    assert len(r.server.attaches) == 1


def test_the_shell_delivers_exactly_the_gap_and_the_frame_the_kernel_names(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = rig(Script("abc"))
    stub = Scripted(
        {
            0: [K.OpenAttach(0, "", 30_000)],
            1: [],  # answered: read on
            2: [K.Deliver(at=3, len=3, dropped=7)],  # a frame (the echo, the output...): deliver it
        }
    )
    stub.answers[2] = [K.Deliver(at=3, len=3, dropped=7), K.Finish()]
    monkeypatch.setattr(K, "_loaded", stub)
    # The first line is the `attached` echo, which has no data: the gap is told, no output follows.
    assert list(r.jobs.follow("s", "aa")) == [JobGap(7)]


# ---- a start nothing answers, and the caller's deadline ------------------------------------------


def closed_url() -> str:
    """An endpoint nothing listens on: the connection is refused."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}/exec"


def test_a_start_nothing_answers_ends_with_a_connect_error_at_once() -> None:
    import time

    t0 = time.monotonic()
    with pytest.raises(ConnectError) as e:
        Jobs(HttpTransport(closed_url())).start("s", "aa", "true")
    assert isinstance(e.value, TransportError)  # `except TransportError` keeps working
    assert time.monotonic() - t0 < 5
    with pytest.raises(ConnectError):
        list(Jobs(HttpTransport(closed_url())).start_held("s", "aa", "true"))


def test_async_a_start_nothing_answers_ends_with_a_connect_error() -> None:
    async def main() -> None:
        with pytest.raises(ConnectError):
            await AsyncJobs(AsyncHttpTransport(closed_url())).start("s", "aa", "true")

    asyncio.run(main())


def test_once_anything_answered_a_failure_to_connect_never_ends_the_handle(rig: Any) -> None:
    import threading
    import time

    r = rig(Script("abc", start_status=429))
    threading.Timer(0.15, r.server.close).start()  # the endpoint goes away after its first answer
    t0 = time.monotonic()
    # It is not `ConnectError`: the first start was answered (429), so the handle keeps trying until the
    # caller's own deadline ends it.
    with pytest.raises(TimeoutError):
        r.jobs.start("s", "aa", "true", timeout=1.5)
    assert 1.4 < time.monotonic() - t0 < 6
    assert len(r.server.starts) >= 1


def test_the_callers_deadline_ends_a_follow_that_cannot_connect() -> None:
    import time

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        list(Jobs(HttpTransport(closed_url())).follow("s", "aa", timeout=0.6))
    assert 0.5 < time.monotonic() - t0 < 5


def test_async_the_callers_deadline_ends_a_follow_that_cannot_connect() -> None:
    async def main() -> None:
        jobs = AsyncJobs(AsyncHttpTransport(closed_url()))
        with pytest.raises(TimeoutError):
            await jobs.collect("s", "aa", timeout=0.6)
        with pytest.raises(TimeoutError):
            [f async for f in jobs.follow("s", "aa", timeout=0.6)]

    asyncio.run(main())


def test_a_deadline_that_is_not_a_number_of_seconds_is_refused(rig: Any) -> None:
    from beyond_sandbox import InvalidRequestError

    r = rig(Script("abc"))
    for bad in (-1, "5", True):
        with pytest.raises(InvalidRequestError):
            list(r.jobs.follow("s", "aa", timeout=bad))
    assert list(r.jobs.follow("s", "aa", timeout=30))[-1] == JobEnd(0)
