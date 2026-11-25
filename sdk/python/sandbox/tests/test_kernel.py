"""The native kernel from Python: the library is the one the sources build, and its ABI works through ctypes."""

from __future__ import annotations

import hashlib
import resource
import struct
from pathlib import Path

import pytest

from beyond_sandbox import _kernel as K
from beyond_sandbox.errors import ProtocolError

REPO = Path(__file__).resolve().parents[4]
CRATES = REPO / "crates"


def sources_hash() -> str:
    """``crates/job-client-ffi/source-hash.sh`` in Python: this crate and the kernel, by content."""
    names = ["job-client-ffi/toolchain", "job-client-ffi/Cargo.toml", "job-client/Cargo.toml"]
    for crate in ("job-client-ffi", "job-client"):  # each crate's sources in byte order
        names += sorted(
            f"{crate}/src/{p.name}" for p in (CRATES / crate / "src").iterdir() if p.suffix == ".rs"
        )
    h = hashlib.sha256()
    for n in names:
        h.update((CRATES / n).read_bytes())
    return h.hexdigest()


def test_the_library_is_built_from_the_sources_beside_it() -> None:
    """A source changed and ``mise run build:py-kernel`` did not run: the library is not what is proven."""
    assert (CRATES / "job-client").is_dir(), "a checkout of the repository"
    pin = K.library_path().parent / "kernel.src.sha256"
    assert pin.is_file(), f"{pin} is missing: build the kernel with `mise run build:py-kernel`"
    assert pin.read_text().strip() == sources_hash()


def test_the_typescript_module_and_this_library_are_built_from_the_same_sources() -> None:
    ts = REPO / "sdk/ts/sandbox/src/kernel/kernel.src.sha256"
    assert ts.read_text().strip() == sources_hash()


def test_the_abi_is_the_one_this_codec_speaks() -> None:
    assert K.load()._lib.jc_abi_version() == K.ABI_VERSION


def test_it_classifies_one_table_for_every_status() -> None:
    k = K.load()
    assert k.classify("attach", 200) == K.Class("ok")
    assert k.classify("attach", 429) == K.Class("retry")
    assert k.classify("attach", 404) == K.Class("terminal", "job_gone")
    assert k.classify("start", 409) == K.Class("terminal", "job_exists")
    assert k.classify("start", 507) == K.Class("terminal", "disk_full")
    assert k.classify("attach", 401) == K.Class("terminal", "auth")
    assert k.classify("remove", 404) == K.Class("ok")


def test_the_silence_limit_follows_the_heartbeat() -> None:
    k = K.load()
    assert [k.silence_ms(b) for b in (0, 1, 10, 11, 60, 240)] == [
        30_000, 30_000, 30_000, 32_000, 130_000, 490_000,
    ]  # fmt: skip


def test_a_state_resumed_from_a_cursor_reopens_at_its_offset_and_mark() -> None:
    k = K.load()
    s = k.state_for_attach("00ff1a", 4096, K.Policy())
    cursor = k.cursor_of(s, "sess-1", "a1b2c3")
    assert cursor.startswith("j1.sess-1.a1b2c3.00ff1a.0000000000001000.")
    _, actions = k.step(k.state_for_cursor(cursor, K.Policy()), K.Event.GO, 0, 0)
    assert actions == [K.OpenAttach(4096, "00ff1a", 30_000)]
    with pytest.raises(ProtocolError):
        k.state_for_cursor(cursor[:-1], K.Policy())
    with pytest.raises(ProtocolError):
        k.state_for_attach("not hex", 0, K.Policy())


def test_an_input_longer_than_the_buffer_is_refused_and_the_library_is_unharmed() -> None:
    k = K.load()
    for n in (1024 - 5, 1024 - 4, 5000, 70_000):
        with pytest.raises(ProtocolError):
            k.state_for_cursor("j1." + "a" * n, K.Policy())
    s = k.state_for_attach("00ff1a", 7, K.Policy())
    assert k.step(s, K.Event.GO, 0, 0)[1] == [K.OpenAttach(7, "00ff1a", 30_000)]


def test_a_mark_longer_than_32_is_refused_by_the_codec() -> None:
    with pytest.raises(ProtocolError):
        K.Frame.started("a" * 33, False)


def test_every_action_the_kernel_makes_decodes() -> None:
    """Drive a start through its life and look at every action tag the kernel can say."""
    k = K.load()
    s = k.state_for_start(7, K.Policy(stdin_open=True))
    seen: set[type] = set()
    now = 1_000
    for ev in (
        K.Event.GO,
        K.Event.answered(200, 0),
        K.Event.frame(K.Frame.started("00ff1a", True)),
        K.Event.frame(K.Frame.output(3)),
        K.Event.frame(K.Frame.gap(2)),
        K.Event.frame(K.Frame.shell_exit(1)),
        K.Event.BROKE,
        K.Event.TICK,
        K.Event.answered(503, 1),
        K.Event.TICK,
        K.Event.answered(200, 0),
        K.Event.frame(K.Frame.attached("00ff1a", 3)),
        K.Event.frame(K.Frame.exit(0, 0)),
        K.Event.EXIT_TAKEN,
    ):
        s, acts = k.step(s, ev, now, 0)
        seen |= {type(a) for a in acts}
        now += 100_000  # past every backoff
    assert {
        K.Start,
        K.OpenAttach,
        K.Started,
        K.Deliver,
        K.ShellExit,
        K.Sleep,
        K.DeliverExit,
    } <= seen
    assert K.Finish in seen


def test_nothing_leaks_across_20000_calls() -> None:
    k = K.load()
    s = k.state_for_attach("00ff1a", 0, K.Policy())
    ev = K.Event.frame(K.Frame.HEARTBEAT)
    for i in range(1000):  # warm the allocator
        k.step(s, ev, i, i)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for i in range(20_000):
        k.step(s, ev, i, i)
        k.classify("attach", 429)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert after - before < 4096, f"maxrss grew {after - before} KiB over 40 000 calls"


def test_a_step_costs_microseconds_not_milliseconds() -> None:
    import time

    k = K.load()
    s = k.state_for_attach("00ff1a", 0, K.Policy())
    ev = K.Event.frame(K.Frame.output(1000))
    t0 = time.perf_counter()
    for i in range(5000):
        k.step(s, ev, i, i)
    per_step_us = (time.perf_counter() - t0) / 5000 * 1e6
    assert per_step_us < 200, f"{per_step_us:.1f} us per step"


def test_the_frame_encodings_are_the_codec_layout() -> None:
    assert K.Frame.output(5) == b"\2" + struct.pack("<Q", 5)
    assert K.Frame.exit(-1, 9) == b"\5" + struct.pack("<ii", -1, 9)
    assert K.Frame.started("ab", True) == b"\1\2ab" + b"\0" * 30 + b"\1\0\0"
    assert K.Frame.started("ab", False, False, True) == b"\1\2ab" + b"\0" * 30 + b"\0\0\1"
    assert K.Frame.started("ab", False, True) == b"\1\2ab" + b"\0" * 30 + b"\0\1\0"
