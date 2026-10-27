"""Plain data returned by the transport and the helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

Kind = Literal["file", "dir", "other"]


@dataclass(frozen=True, slots=True)
class ExecResult:
    """The result of a one-shot exec: the command's output as **bytes**. A non-zero ``exit_code`` is a
    normal result (``128 + N`` for a signal, ``124`` for a deadline, ``127`` for a command that could
    not be started)."""

    exit_code: int
    stdout: bytes
    stderr: bytes
    #: Whether ``stdout`` and ``stderr`` are exactly the bytes the command wrote. ``False`` only when
    #: the transport could get nothing but text from the target (a plain v1.1 endpoint, which already
    #: replaced invalid UTF-8 with U+FFFD before this client saw it): the bytes are then the UTF-8
    #: of that text. A transport that has the bytes (ssh, a vendor SDK) leaves the default.
    exact: bool = True

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", "replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", "replace")


# ---- v1.2 stream frames -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stdout:
    data: bytes


@dataclass(frozen=True, slots=True)
class Stderr:
    data: bytes


@dataclass(frozen=True, slots=True)
class ShellExit:
    """The command exited but its process group runs on. Not final."""

    code: int


@dataclass(frozen=True, slots=True)
class Exit:
    """Final: the command exited with ``code``."""

    code: int


@dataclass(frozen=True, slots=True)
class Signal:
    """Final: the command was killed by signal ``signal``."""

    signal: int


@dataclass(frozen=True, slots=True)
class TimedOut:
    """Final: the endpoint killed the command at its ``timeout_ms``."""


Frame = Union[Stdout, Stderr, ShellExit, Exit, Signal, TimedOut]
FINAL = (Exit, Signal, TimedOut)


# ---- files -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stat:
    kind: Kind
    size: int
    mtime: float  # epoch seconds, with the fraction stat reports (nanoseconds on GNU and busybox)
    writable: bool


@dataclass(frozen=True, slots=True)
class Entry:
    name: str
    kind: Kind
    size: int


@dataclass(frozen=True, slots=True)
class Hit:
    path: str
    line: int
    text: str
    is_match: bool


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[Hit]
    truncated: bool
    note: str | None


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What the probes found. ``search_engine`` is ``rg``, ``grep`` or ``busybox-grep``."""

    stdin: bool
    find_printf: bool
    search_engine: str
    read_window: bool
    rg: bool = False
    grep_null: bool = False


@dataclass(frozen=True, slots=True)
class SandboxCapabilities:
    """What a sandbox offers: the file probes plus ``stream`` and ``jobs`` (native/baseline/none), and
    ``output_base64`` (a one-shot call returns the command's bytes)."""

    stdin: bool
    stream: bool
    jobs: str
    search_engine: str
    find_printf: bool
    read_window: bool
    output_base64: bool = False


# ---- jobs --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobInfo:
    id: str
    state: str  # running | exited | ended | lost | gone
    code: int | None
    kept: bool
    num: int | None
    kind: str | None
    seen: int
    started: int
    ended: int
    label: str


@dataclass(frozen=True, slots=True)
class JobRead:
    state: str
    code: int | None
    offset: int  # where ``data`` starts (later than asked when old output was dropped)
    end: int  # the end of the job's output so far
    data: bytes


@dataclass(frozen=True, slots=True)
class Survivor:
    id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class JobOutput:
    """Output of a job, in order."""

    data: bytes


@dataclass(frozen=True, slots=True)
class JobGap:
    """``bytes`` of output are gone (past the cap) before they could be read."""

    bytes: int


@dataclass(frozen=True, slots=True)
class JobShellExit:
    code: int


@dataclass(frozen=True, slots=True)
class JobStarted:
    pgid: int


@dataclass(frozen=True, slots=True)
class JobEnd:
    """Final frame of a job stream: the job's status (the exit code, or 128 + signal)."""

    code: int


JobEvent = Union[JobOutput, JobGap, JobShellExit, JobStarted]
JobFrame = Union[JobOutput, JobGap, JobShellExit, JobStarted, JobEnd]


@dataclass(slots=True)
class Attached:
    """What following a job to its end collected."""

    exit_code: int
    output: bytes = b""
    gap: int = 0
    shell_exit: int | None = field(default=None)
