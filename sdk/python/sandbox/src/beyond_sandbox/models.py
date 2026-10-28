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

    # The one place the client reads output as text: helpers parse ``stdout_text`` / ``stderr_text``.
    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", "replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", "replace")


# ---- stream frames -----------------------------------------------------------------------


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
    """What the environment probe found. ``search_engine`` is ``rg``, ``grep`` or ``busybox-grep``."""

    find_printf: bool
    search_engine: str
    read_window: bool
    rg: bool = False
    grep_null: bool = False


# ---- jobs --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobInfo:
    id: str
    state: str  # running | exited | ended | lost | gone
    code: int | None  # 128 + N when signal N killed the shell
    kept: bool
    num: int | None
    kind: str | None
    seen: int
    started: int
    ended: int
    label: str
    signal: int | None = None  # the signal that killed the shell; None for a shell that exited


@dataclass(frozen=True, slots=True)
class JobRead:
    state: str
    code: int | None
    offset: int  # where ``data`` starts (later than asked when old output was dropped)
    end: int  # the end of the job's output so far
    data: bytes
    signal: int | None = None  # the signal that killed the shell; None for a shell that exited


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
    """Final frame of a job stream: the job's status (``code`` is the exit code, or 128 + signal
    when a signal killed the shell, in which case ``signal`` is that signal)."""

    code: int
    signal: int | None = None


JobEvent = Union[JobOutput, JobGap, JobShellExit, JobStarted]
JobFrame = Union[JobOutput, JobGap, JobShellExit, JobStarted, JobEnd]


@dataclass(slots=True)
class Attached:
    """What following a job to its end collected."""

    exit_code: int  # the shell's status: 128 + ``signal`` when a signal killed it
    output: bytes = b""
    gap: int = 0
    shell_exit: int | None = field(default=None)
    signal: int | None = None  # the signal that killed the shell; None for a shell that exited


@dataclass(frozen=True, slots=True)
class Transfer:
    """What a bulk copy moved: members that are regular files, directories (not the root), links,
    and the bytes of file content."""

    files: int = 0
    dirs: int = 0
    links: int = 0
    bytes: int = 0
