"""The job client's kernel (``crates/job-client``, proven in Verus), loaded as a native library.

The library is ``crates/job-client-ffi`` built as a ``cdylib``: the same function the Rust SDK links and
the TypeScript SDK runs as WASM, behind one integer ABI (``crates/job-client-ffi/src/codec.rs``). This
file is its codec: bytes in, bytes out. It decides nothing. ``_held.py`` does the I/O the kernel's
actions name.

A wheel carries the library in ``_native/``; a source checkout builds it with ``mise run build:py-kernel``.
A library that is missing, or whose ABI is not the one this codec speaks, is an ``ImportError`` at first
use: there is no fallback to a Python implementation of the decisions.
"""

from __future__ import annotations

import ctypes
import random
import secrets
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .errors import ProtocolError

#: The ABI ``codec.rs`` lays out. A change to a layout there bumps ``ABI_VERSION`` in ``abi.rs``.
ABI_VERSION = 5
#: Room for any one call's input or answer: a state (90 B) and an event (45 B at most), or a cursor.
_BUF = 1024
_STATE = 90

#: The one vocabulary of typed failures: the kernel's ``Fault``, in its declared order.
FAULTS = (
    "job_gone",
    "job_exists",
    "disk_full",
    "no_pty",
    "deadline",
    "not_offered",
    "protocol",
    "auth",
    "transport",
)

_LIB_NAMES = {
    "linux": "libjob_client_ffi.so",
    "darwin": "libjob_client_ffi.dylib",
    "win32": "job_client_ffi.dll",
}


# ---- what the kernel is told ------------------------------------------------------------------------


def _mark(hex_: str) -> bytes:
    raw = hex_.encode("ascii", "replace")
    if len(raw) > 32:
        raise ProtocolError("a job mark is at most 32 characters")
    return bytes([len(raw)]) + raw.ljust(32, b"\0")


class Frame:
    """What a stream line said, as the kernel's bytes. The payload stays in the shell."""

    @staticmethod
    def attached(mark: str, from_: int) -> bytes:
        return b"\0" + _mark(mark) + struct.pack("<Q", from_)

    @staticmethod
    def started(mark: str, stdin: bool, pty: bool = False) -> bytes:
        return b"\1" + _mark(mark) + bytes([stdin, pty])

    @staticmethod
    def output(length: int) -> bytes:
        return b"\2" + struct.pack("<Q", length)

    @staticmethod
    def gap(n: int) -> bytes:
        return b"\3" + struct.pack("<Q", n)

    HEARTBEAT = b"\4"

    @staticmethod
    def exit(code: int, signal: int) -> bytes:
        return b"\5" + struct.pack("<ii", code, signal)

    BAD = b"\6"

    @staticmethod
    def shell_exit(code: int) -> bytes:
        return b"\7" + struct.pack("<i", code)


class Event:
    """What the shell observed, as the kernel's bytes."""

    GO = b"\0"

    @staticmethod
    def answered(status: int, retry_after_s: int) -> bytes:
        return b"\1" + struct.pack("<HI", status, retry_after_s)

    @staticmethod
    def frame(frame: bytes) -> bytes:
        return b"\2" + frame

    BROKE = b"\3"
    TICK = b"\4"
    EXIT_TAKEN = b"\5"
    CLOSE = b"\6"
    DEADLINE = b"\7"
    #: The request never reached a server: refused, unreachable, or the name did not resolve.
    UNREACHED = b"\x09"


@dataclass(frozen=True, slots=True)
class Policy:
    watch: bool = False
    remove_on_exit: bool = False
    #: A start that needs the job's stdin opened: an echo that says it was not is refused.
    stdin_open: bool = False
    #: A start on a terminal: an echo that says none was opened is refused.
    pty_open: bool = False
    #: A start of a kept job (``hold: false``): a refused echo stops the job.
    kept: bool = False
    #: The heartbeat the request chose, whole seconds; 0 for the default.
    beat_s: int = 0


# ---- what the kernel says ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Start:
    key: int
    silence_ms: int


@dataclass(frozen=True, slots=True)
class OpenAttach:
    offset: int
    mark: str
    silence_ms: int


@dataclass(frozen=True, slots=True)
class CloseStream:
    pass


@dataclass(frozen=True, slots=True)
class Kill:
    """A kept start whose echo was refused: nothing else can name the job, so stop it."""


@dataclass(frozen=True, slots=True)
class Started:
    mark: str
    stdin: bool
    pty: bool


@dataclass(frozen=True, slots=True)
class Attached:
    mark: str
    from_offset: int


@dataclass(frozen=True, slots=True)
class ShellExit:
    code: int


@dataclass(frozen=True, slots=True)
class Reconnecting:
    attempts: int
    since_ms: int


@dataclass(frozen=True, slots=True)
class Deliver:
    at: int
    len: int
    dropped: int


@dataclass(frozen=True, slots=True)
class DeliverExit:
    code: int
    signal: int
    dropped: int
    #: A signal killed the shell: the end is ``signal``, not an exit with ``code``.
    killed: bool


@dataclass(frozen=True, slots=True)
class Sleep:
    until_ms: int


@dataclass(frozen=True, slots=True)
class Remove:
    mark: str


@dataclass(frozen=True, slots=True)
class Fail:
    fault: str
    status: int
    offset: int


@dataclass(frozen=True, slots=True)
class Finish:
    pass


@dataclass(frozen=True, slots=True)
class Detached:
    pass


Action = Union[
    Start,
    OpenAttach,
    CloseStream,
    Kill,
    Started,
    Attached,
    ShellExit,
    Reconnecting,
    Deliver,
    DeliverExit,
    Sleep,
    Remove,
    Fail,
    Finish,
    Detached,
]


@dataclass(frozen=True, slots=True)
class Class:
    """What an HTTP answer to one op means: ``"ok"``, ``"retry"`` or ``"terminal"`` (with its fault)."""

    kind: str
    fault: str = ""


# ---- the library ------------------------------------------------------------------------------------


class _Rd:
    __slots__ = ("a", "o")

    def __init__(self, a: bytes) -> None:
        self.a, self.o = a, 0

    def u8(self) -> int:
        self.o += 1
        return self.a[self.o - 1]

    def take(self, fmt: str) -> int:
        (v,) = struct.unpack_from(fmt, self.a, self.o)
        self.o += struct.calcsize(fmt)
        return int(v)

    def mark(self) -> str:
        n = self.u8()
        raw = self.a[self.o : self.o + 32]
        self.o += 32
        return raw[:n].decode("ascii", "replace")


def _action(r: _Rd) -> Action:
    tag = r.u8()
    if tag == 0:
        return Start(r.take("<Q"), r.take("<Q"))
    if tag == 1:
        return OpenAttach(r.take("<Q"), r.mark(), r.take("<Q"))
    if tag == 2:
        return CloseStream()
    if tag == 3:
        return Deliver(r.take("<Q"), r.take("<Q"), r.take("<Q"))
    if tag == 4:
        return DeliverExit(r.take("<i"), r.take("<i"), r.take("<Q"), bool(r.u8()))
    if tag == 5:
        return Sleep(r.take("<Q"))
    if tag == 6:
        return Remove(r.mark())
    if tag == 7:
        i = r.u8()
        if i >= len(FAULTS):
            raise ProtocolError("kernel: unknown fault")
        return Fail(FAULTS[i], r.take("<H"), r.take("<Q"))
    if tag == 8:
        return Finish()
    if tag == 9:
        return Detached()
    if tag == 10:
        return Reconnecting(r.take("<I"), r.take("<Q"))
    if tag == 11:
        return Started(r.mark(), bool(r.u8()), bool(r.u8()))
    if tag == 12:
        return Attached(r.mark(), r.take("<Q"))
    if tag == 13:
        return ShellExit(r.take("<i"))
    if tag == 14:
        return Kill()
    raise ProtocolError(f"kernel: unknown action {tag}")


class Kernel:
    """One loaded library. Calls keep nothing between them: a state is a value the caller holds."""

    def __init__(self, path: str) -> None:
        lib = ctypes.CDLL(path)
        lib.jc_abi_version.restype = ctypes.c_uint32
        lib.jc_abi_version.argtypes = []
        if lib.jc_abi_version() != ABI_VERSION:
            raise ImportError(
                f"the job kernel at {path} speaks ABI {lib.jc_abi_version()}, not {ABI_VERSION}"
            )
        for name in (
            "jc_state_for_start",
            "jc_state_for_attach",
            "jc_state_for_cursor",
            "jc_step",
            "jc_classify",
            "jc_silence_ms",
            "jc_cursor_of",
        ):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_int32
            fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        self._lib = lib

    def _call(self, name: str, data: bytes) -> bytes:
        if len(data) > _BUF:
            # The answer buffer is this size; an input this long is no record the kernel makes.
            raise ProtocolError(
                f"kernel: an input of {len(data)} bytes is past the {_BUF} it takes"
            )
        out = ctypes.create_string_buffer(_BUF)
        n = getattr(self._lib, name)(data, len(data), out, _BUF)
        if n < 0:
            raise ProtocolError(f"kernel: refused its input ({n})")
        return out.raw[:n]

    def state_for_start(self, key: int, p: Policy) -> bytes:
        return self._call(
            "jc_state_for_start",
            struct.pack(
                "<QBBIBBB",
                key,
                p.watch,
                p.remove_on_exit,
                p.beat_s,
                p.stdin_open,
                p.pty_open,
                p.kept,
            ),
        )

    def state_for_attach(self, mark: str, offset: int, p: Policy) -> bytes:
        return self._call(
            "jc_state_for_attach",
            _mark(mark) + struct.pack("<QBBI", offset, p.watch, p.remove_on_exit, p.beat_s),
        )

    def state_for_cursor(self, cursor: str, p: Policy) -> bytes:
        head = struct.pack("<BBI", p.watch, p.remove_on_exit, p.beat_s)
        return self._call("jc_state_for_cursor", head + cursor.encode("utf-8"))

    def step(
        self, state: bytes, event: bytes, now_ms: int, draw: int
    ) -> tuple[bytes, list[Action]]:
        r = _Rd(self._call("jc_step", state + event + struct.pack("<QI", now_ms, draw)))
        next_state = r.a[:_STATE]
        r.o = _STATE
        return next_state, [_action(r) for _ in range(r.u8())]

    def classify(self, op: str, status: int) -> Class:
        r = _Rd(
            self._call(
                "jc_classify",
                struct.pack("<BH", ("start", "attach", "remove", "other").index(op), status),
            )
        )
        tag = r.u8()
        if tag == 0:
            return Class("ok")
        if tag == 1:
            return Class("retry")
        return Class("terminal", FAULTS[r.u8()])

    def silence_ms(self, beat_s: int) -> int:
        return _Rd(self._call("jc_silence_ms", struct.pack("<I", beat_s))).take("<Q")

    def cursor_of(self, state: bytes, scope: str, job: str) -> str:
        data = state + bytes([len(scope)]) + (scope + job).encode("ascii")
        return self._call("jc_cursor_of", data).decode("ascii")


_loaded: Kernel | None = None


def library_path() -> Path:
    """Where this package keeps the native library."""
    return (
        Path(__file__).resolve().parent
        / "_native"
        / _LIB_NAMES.get(sys.platform, "libjob_client_ffi.so")
    )


def load(path: str | None = None) -> Kernel:
    """The kernel, loaded once per process from ``path`` (the package's own library when omitted)."""
    global _loaded
    if path is not None:
        return Kernel(path)
    if _loaded is None:
        lib = library_path()
        if not lib.is_file():
            raise ImportError(
                f"the job kernel is missing ({lib}): install a wheel for this platform, or build it "
                "from a source checkout with `mise run build:py-kernel`"
            )
        _loaded = Kernel(str(lib))
    return _loaded


# ---- the shell's clock and entropy (inputs, never decisions) ----------------------------------------


def now_ms() -> int:
    """The shell's clock, in the milliseconds the kernel's times are on."""
    return time.monotonic_ns() // 1_000_000


def wait_s(until_ms: int, deadline_ms: int | None = None) -> float:
    """Seconds from now to a time the kernel named, or to the caller's deadline if that comes first."""
    end = until_ms if deadline_ms is None else min(until_ms, deadline_ms)
    return max(end - now_ms(), 0) / 1000


class Limit:
    """The caller's timeout, on the shell's clock. The kernel's `Deadline` event is what ends a handle;
    this only says when the shell sends it."""

    __slots__ = ("ends_ms",)

    def __init__(self, timeout_s: float | None) -> None:
        self.ends_ms = None if timeout_s is None else now_ms() + round(timeout_s * 1000)

    def passed(self) -> bool:
        return self.ends_ms is not None and now_ms() >= self.ends_ms


def elapsed_ms(since_ms: int) -> int:
    """Milliseconds since a time the kernel named."""
    return max(now_ms() - since_ms, 0)


def draw() -> int:
    """A uniform 32-bit value: the jitter of a backoff."""
    return random.getrandbits(32)


def key() -> int:
    """A fresh idempotency key's handle."""
    return secrets.randbits(64)
