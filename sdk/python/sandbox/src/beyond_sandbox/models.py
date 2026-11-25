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
    mark: str  # names this incarnation of the id
    command: str
    args: tuple[str, ...]
    output_start: int  # the offset of the oldest byte of output the sandbox holds
    output_end: int  # the offset of the next byte of output
    signal: int | None = None  # the signal that killed the shell; None for a shell that exited


@dataclass(frozen=True, slots=True)
class JobRead:
    state: str
    code: int | None
    offset: int  # where ``data`` starts (later than asked when old output was dropped)
    end: int  # the end of the job's output so far
    data: bytes
    signal: int | None = None  # the signal that killed the shell; None for a shell that exited
    #: Absolute ``(from, len)`` offsets of the bytes of ``data`` that came from stderr (the rest is
    #: stdout); empty when the window holds none.
    stderr_ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class Survivor:
    id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class JobOutput:
    """Output of a job, in order. ``stderr`` is the pipe it came from."""

    data: bytes
    stderr: bool = False


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
    stdin: bool = False  # the job was started with its stdin open (the echo of ``stdin_open``)
    mark: str = ""  # names this incarnation of the id; empty only from an endpoint that predates it


@dataclass(frozen=True, slots=True)
class JobStart:
    """What ``start`` answers: the job's process group and the mark of this incarnation of its id."""

    pgid: int
    mark: str


@dataclass(frozen=True, slots=True)
class JobAttached:
    """First frame of an attach: the incarnation it is following and the offset it starts at."""

    mark: str
    from_offset: int


@dataclass(frozen=True, slots=True)
class JobRemoved:
    """What ``remove`` skipped: ``running`` jobs (their group is not empty) and ``mismatch`` jobs (their
    mark is not the one named). An id with no job is in neither."""

    running: tuple[str, ...]
    mismatch: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StdinWritten:
    """What a ``write`` left: bytes of the stdin stream applied in all, and whether it is closed."""

    written: int
    closed: bool = False


@dataclass(frozen=True, slots=True)
class JobEnd:
    """Final frame of a job stream: the job's status (``code`` is the exit code, or 128 + signal
    when a signal killed the shell, in which case ``signal`` is that signal)."""

    code: int
    signal: int | None = None

    @classmethod
    def of(cls, code: int, signal: int, killed: bool) -> JobEnd:
        """The end the kernel delivered: an exit with ``code``, or (``killed``) ``signal`` killed the shell."""
        return cls(128 + signal, signal) if killed else cls(code)


@dataclass(frozen=True, slots=True)
class Reconnecting:
    """The handle cannot reach the endpoint and keeps trying: ``attempts`` failed opens in a row, the
    first ``since_ms`` milliseconds ago. A status, never an end."""

    attempts: int
    since_ms: int


JobEvent = Union[JobOutput, JobGap, JobShellExit, JobStarted, JobAttached]
JobFrame = Union[JobOutput, JobGap, JobShellExit, JobStarted, JobAttached, JobEnd]


@dataclass(slots=True)
class Attached:
    """What following a job to its end collected."""

    exit_code: int  # the shell's status: 128 + ``signal`` when a signal killed it
    output: bytes = b""
    stderr: bytes = b""  # the part of ``output`` that came from stderr frames
    gap: int = 0
    shell_exit: int | None = field(default=None)
    signal: int | None = None  # the signal that killed the shell; None for a shell that exited
    attached: JobAttached | None = None  # the stream's first frame


@dataclass(frozen=True, slots=True)
class Transfer:
    """What a bulk copy moved: members that are regular files, directories (not the root), links,
    and the bytes of file content."""

    files: int = 0
    dirs: int = 0
    links: int = 0
    bytes: int = 0
