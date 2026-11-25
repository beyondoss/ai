"""The exec protocol on the wire, as pure functions: requests, answers, frames, job messages."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeGuard, TypeVar

from . import _kernel as K
from .errors import (
    AuthError,
    EndpointError,
    InvalidRequestError,
    NotOfferedError,
    ProtocolError,
    SandboxError,
    TooLargeError,
    TransportError,
)
from .models import (
    ExecResult,
    Exit,
    Frame,
    JobInfo,
    JobRead,
    JobRemoved,
    ShellExit,
    Signal,
    Stderr,
    StdinWritten,
    Stdout,
    Survivor,
    TimedOut,
)

#: The SIGTERM grace a kill gives a job before SIGKILL.
KILL_GRACE_MS = 300
#: The response body cap (FILES-OVER-EXEC.md section 2).
DEFAULT_MAX_RESPONSE = 16 * 1024 * 1024
#: Grace added to ``timeout_ms`` for the whole exchange, so the endpoint's own timeout report wins.
GRACE_S = 30.0
#: An exec stream with no frame (not even a heartbeat) for this long is a dead connection. (A job
#: stream's limit is the kernel's, derived from the heartbeat the request chose.)
SILENCE_S = 30.0
#: The request header that names a heartbeat interval, whole seconds, and its range.
HEARTBEAT_HEADER = "Keepalive-Ping-Interval"
HEARTBEAT_MIN_S = 1
HEARTBEAT_MAX_S = 240
#: The ``timeout_ms`` of a stream with no deadline of its own: the most every endpoint can hold.
STREAM_DEADLINE_MS = 2**31 - 1
#: The longest NDJSON line accepted: a full 64 KiB frame is about 87 KiB of base64.
MAX_FRAME_LINE = 1024 * 1024
#: Room for the JSON around two base64 strings.
ENVELOPE_SLACK = 4096


def check_heartbeat(heartbeat_seconds: int | None) -> int | None:
    """The heartbeat a request may ask for: whole seconds, 1 to 240, or nothing. Anything else would be
    silently replaced by the endpoint's own, so it is refused before anything is sent."""
    if heartbeat_seconds is None:
        return None
    if (
        isinstance(heartbeat_seconds, bool)
        or not isinstance(heartbeat_seconds, int)
        or not HEARTBEAT_MIN_S <= heartbeat_seconds <= HEARTBEAT_MAX_S
    ):
        raise InvalidRequestError(
            f"heartbeat_seconds must be a whole number from {HEARTBEAT_MIN_S} to "
            f"{HEARTBEAT_MAX_S}, got {heartbeat_seconds!r}"
        )
    return heartbeat_seconds


def request_body(
    command: str,
    args: Sequence[str],
    *,
    cwd: str | None,
    timeout_ms: int,
    stdin: bytes | None = None,
    stream: bool = False,
    job: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> bytes:
    body: dict[str, Any] = {"command": command, "args": list(args)}
    if cwd is not None:
        body["cwd"] = cwd
    body["timeout_ms"] = timeout_ms
    if stdin is not None:
        body["stdin"] = base64.b64encode(stdin).decode("ascii")
    if stream:
        body["stream"] = True
    if job is not None:
        body["job"] = job
    if env:
        body["env"] = dict(env)
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def excerpt(body: bytes) -> str:
    return body.decode("utf-8", "replace").strip()[:400]


def status_error(url: str, status: int, body: bytes) -> SandboxError:
    """The error of a non-2xx answer (FILES-OVER-EXEC.md 3.2)."""
    text = excerpt(body)
    detail = f"exec endpoint {url} returned {status}: {text}"
    if status in (401, 403):
        return AuthError(detail, status=status)
    if status == 422 and "cap" in text.lower():
        return TooLargeError(detail, status=status)
    return EndpointError(detail, status=status)


def retry_after_s(headers: dict[str, str]) -> int:
    """The ``Retry-After`` of an answer, whole seconds; 0 when it sent none (or a date, which no endpoint here sends)."""
    value = headers.get("retry-after", "").strip()
    return int(value) if value.isdigit() else 0


def too_large(url: str, cap: int) -> TooLargeError:
    return TooLargeError(
        f"exec endpoint {url}: the response exceeded the {cap}-byte cap and was discarded; the "
        "command ran, but its output is too large to return; narrow it (e.g. pipe through "
        "`head`) rather than re-running it unchanged"
    )


def parse_json(url: str, body: bytes) -> Any:
    try:
        return json.loads(body)
    except ValueError as e:
        raise ProtocolError(
            f"exec endpoint {url} returned unparseable JSON ({e}): {excerpt(body)}"
        ) from None


def body_cap(cap: int) -> int:
    """The most body a client reads for a one-shot answer under response cap ``cap``: the base64 of
    ``cap`` decoded bytes plus the envelope. The cap itself is on the decoded bytes."""
    return -(-cap // 3) * 4 + ENVELOPE_SLACK


def parse_envelope(url: str, body: bytes, cap: int) -> ExecResult:
    """A one-shot answer: ``{"exit_code", "stdout", "stderr"}``, both streams standard base64. The
    cap is on the decoded bytes of both together."""
    obj = parse_json(url, body)
    code = obj.get("exit_code") if isinstance(obj, dict) else None
    if not isinstance(code, int) or isinstance(code, bool):
        raise ProtocolError(
            f"exec endpoint {url} returned unparseable JSON (no exit_code): {excerpt(body)}"
        )
    out = _b64("stdout", obj.get("stdout"))
    err = _b64("stderr", obj.get("stderr"))
    if len(out) + len(err) > cap:
        raise too_large(url, cap)
    return ExecResult(code, out, err)


# ---- stream frames ---------------------------------------------------------------------------------


def _b64(field: str, value: Any) -> bytes:
    if not isinstance(value, str):
        raise ProtocolError(f"`{field}` is not a string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ProtocolError(f"`{field}` is not standard base64 ({e})") from None


def decode_line(line: bytes) -> list[Frame]:
    """One NDJSON line. A frame naming more than one field is split, output first.

    A line with no known field is a heartbeat and decodes to no frame.
    """
    line = line.removesuffix(b"\r")
    if not line.strip():
        return []
    try:
        obj = json.loads(line)
    except ValueError as e:
        raise ProtocolError(f"unparseable frame ({e}): {excerpt(line)[:200]}") from None
    if not isinstance(obj, dict):
        raise ProtocolError(f"unparseable frame: {excerpt(line)[:200]}")
    frames: list[Frame] = []
    if "stdout" in obj:
        frames.append(Stdout(_b64("stdout", obj["stdout"])))
    if "stderr" in obj:
        frames.append(Stderr(_b64("stderr", obj["stderr"])))
    if _num(obj.get("shell_exit")):
        frames.append(ShellExit(obj["shell_exit"]))
    if "error" in obj:
        raise TransportError(f"the endpoint failed mid-command: {obj['error']}")
    if isinstance(obj.get("exit_code"), int):
        frames.append(Exit(obj["exit_code"]))
    elif isinstance(obj.get("signal"), int):
        frames.append(Signal(obj["signal"]))
    elif obj.get("timed_out") is True:
        frames.append(TimedOut())
    return frames


class LineSplitter:
    """Cuts a byte stream into lines, keeping at most one partial line (bounded)."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._scanned = 0

    def feed(self, data: bytes) -> list[bytes]:
        self._buf += data
        lines: list[bytes] = []
        while True:
            nl = self._buf.find(b"\n", self._scanned)
            if nl < 0:
                break
            lines.append(bytes(self._buf[:nl]))
            del self._buf[: nl + 1]
            self._scanned = 0
        self._scanned = len(self._buf)
        if self._scanned > MAX_FRAME_LINE:
            raise ProtocolError(f"a frame exceeded {MAX_FRAME_LINE} bytes without a newline")
        return lines

    @property
    def pending(self) -> bool:
        return bool(self._buf)


# ---- jobs -----------------------------------------------------------------------------------


def job_op(op: str, scope: str, **extra: Any) -> dict[str, Any]:
    return {"op": op, "scope": scope, **extra}


@dataclass(frozen=True, slots=True)
class JobLine:
    """What one line of a job stream said: the frame the kernel is told of, and what the shell keeps of it."""

    #: The kernel's bytes for the frame (``_kernel.Frame``).
    frame: bytes
    #: An output frame's bytes: they stay here, and the kernel is told only their length.
    data: bytes | None = None
    stderr: bool = False
    #: A ``started`` frame's process group.
    pgid: int = 0
    #: Why the frame is ``bad``.
    note: str = ""


def _bad(note: str) -> JobLine:
    return JobLine(K.Frame.BAD, note=note)


def _count(v: Any) -> TypeGuard[int]:
    """A non-negative integer of the wire: a byte count or an offset."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _num(v: Any) -> TypeGuard[float]:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int32(v: Any) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool) and -(2**31) <= v < 2**31


def decode_job_line(line: bytes) -> JobLine:
    """One line of a job stream, read for the facts the kernel decides on. Never raises.

    Which known key a line has is a codec question (the precedence is the wire's: an error, then the
    echoes, output, gap, the shell's exit, the end); what each fact means is the kernel's.
    """
    line = line.removesuffix(b"\r")
    if not line.strip():
        return JobLine(K.Frame.HEARTBEAT)
    try:
        obj = json.loads(line)
    except ValueError as e:
        return _bad(f"unparseable job frame ({e}): {excerpt(line)[:200]}")
    if not isinstance(obj, dict):
        return _bad(f"unparseable job frame: {excerpt(line)[:200]}")
    try:
        return _line_of(obj)
    except ProtocolError as e:
        return _bad(e.detail)


def _line_of(obj: dict[str, Any]) -> JobLine:
    if "error" in obj:
        return _bad(str(obj["error"])[:400])
    att = obj.get("attached")
    if isinstance(att, dict):
        mark, from_ = att.get("mark"), att.get("from")
        if isinstance(mark, str) and _count(from_):
            return JobLine(K.Frame.attached(mark, from_))
        return _bad("an `attached` frame without its mark and offset")
    started = obj.get("started")
    if _num(started):
        mark = obj.get("mark")
        return JobLine(
            K.Frame.started(mark if isinstance(mark, str) else "", obj.get("stdin") is True),
            pgid=int(started),
        )
    for key in ("stdout", "stderr"):
        if isinstance(obj.get(key), str):
            data = _b64(key, obj[key])
            return JobLine(K.Frame.output(len(data)), data=data, stderr=key == "stderr")
    if _num(obj.get("gap")):
        return (
            JobLine(K.Frame.gap(obj["gap"]))
            if _count(obj["gap"])
            else _bad("a gap that is not a count")
        )
    if _num(obj.get("shell_exit")):
        code = obj["shell_exit"]
        return (
            JobLine(K.Frame.shell_exit(code))
            if _int32(code)
            else _bad("a shell_exit that is not an integer")
        )
    if _num(obj.get("exit_code")):
        code = obj["exit_code"]
        return (
            JobLine(K.Frame.exit(code, 0))
            if _int32(code)
            else _bad("an exit_code that is not an integer")
        )
    if _num(obj.get("signal")):
        sig = obj["signal"]
        return (
            JobLine(K.Frame.exit(0, sig))
            if _int32(sig)
            else _bad("a signal that is not an integer")
        )
    if obj.get("timed_out") is True:
        return _bad("the endpoint timed the job out")
    return JobLine(K.Frame.HEARTBEAT)


JOB_ID = re.compile(r"[0-9a-z_-]{1,64}")


def _is_job_id(v: Any) -> bool:
    """An id read back from a sandbox is never a path: 1 to 64 characters of ``[0-9a-z_-]``."""
    return isinstance(v, str) and JOB_ID.fullmatch(v) is not None


def _int(d: dict[str, Any], key: str) -> int | None:
    v = d.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _state(state: Any, code: Any) -> tuple[str, int | None] | None:
    if state in ("running", "lost", "gone"):
        return state, None
    if state in ("exited", "ended") and isinstance(code, int):
        return state, code
    return None


def _signal(v: dict[str, Any], state: str) -> int | None:
    """A job answer's ``signal``, kept only where the state has a code to go with it."""
    return _int(v, "signal") if state in ("exited", "ended") else None


def parse_read(v: Any) -> JobRead:
    st = _state(v.get("state"), v.get("code")) if isinstance(v, dict) else None
    if st is None:
        raise ProtocolError(f"unreadable job read: {v!r}"[:300])
    return JobRead(
        st[0],
        st[1],
        _int(v, "offset") or 0,
        _int(v, "end") or 0,
        _b64("data", v.get("data", "")),
        _signal(v, st[0]),
        _ranges(v),
    )


def _ranges(v: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    r = v.get("stderr_ranges")
    if r is None:
        return ()
    ok = isinstance(r, list) and all(
        isinstance(p, list)
        and len(p) == 2
        and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in p)
        for p in r
    )
    if not ok:
        raise ProtocolError(f"unreadable job read: {v!r}"[:300])
    return tuple((p[0], p[1]) for p in r)


def parse_written(v: Any) -> StdinWritten:
    n = _int(v, "written") if isinstance(v, dict) else None
    if n is None:
        raise ProtocolError(f"unreadable job write: {v!r}"[:300])
    return StdinWritten(n, v.get("closed") is True)


def parse_list(v: Any) -> list[JobInfo]:
    jobs = v.get("jobs") if isinstance(v, dict) else None
    if not isinstance(jobs, list):
        raise ProtocolError(f"unreadable job list: {v!r}"[:300])
    out: list[JobInfo] = []
    for j in jobs:
        if not isinstance(j, dict) or not _is_job_id(j.get("id")):
            continue
        st = _state(j.get("state"), j.get("code"))
        if st is None:
            continue
        out.append(
            JobInfo(
                id=j["id"],
                state=st[0],
                code=st[1],
                kept=j.get("kept") is True,
                num=_int(j, "num"),
                kind=j["kind"] if isinstance(j.get("kind"), str) else None,
                seen=_int(j, "seen") or 0,
                started=_int(j, "started") or 0,
                ended=_int(j, "ended") or 0,
                label=j["label"] if isinstance(j.get("label"), str) else "",
                mark=j["mark"] if isinstance(j.get("mark"), str) else "",
                command=j["command"] if isinstance(j.get("command"), str) else "",
                args=tuple(a for a in (j.get("args") or ()) if isinstance(a, str)),
                output_start=_int(j, "outputStart") or 0,
                output_end=_int(j, "outputEnd") or 0,
                signal=_signal(j, st[0]),
            )
        )
    return out


def describe_survivors(pids: str) -> str:
    parts = [p for p in pids.split(",") if p]
    if not parts:
        return "a process in the sandbox"
    if len(parts) == 1:
        return f"process {parts[0]} in the sandbox"
    return f"processes {', '.join(parts)} in the sandbox"


def parse_survivors(v: Any) -> list[Survivor]:
    items = v.get("survivors") if isinstance(v, dict) else None
    out: list[Survivor] = []
    for s in items if isinstance(items, list) else []:
        if not isinstance(s, dict):
            continue
        pids = ",".join(str(p) for p in s.get("pids", []) if isinstance(p, int))
        out.append(
            Survivor(s["id"] if isinstance(s.get("id"), str) else None, describe_survivors(pids))
        )
    return out


T = TypeVar("T")


class StreamDecoder(Generic[T]):
    """Bytes of an NDJSON body in, items out, up to and including the final one."""

    def __init__(self, decode: Callable[[bytes], list[T]], is_final: Callable[[T], bool]) -> None:
        self._decode = decode
        self._is_final = is_final
        self._lines = LineSplitter()
        self.finished = False

    def feed(self, data: bytes) -> list[T]:
        out: list[T] = []
        if self.finished:
            return out
        for line in self._lines.feed(data):
            for item in self._decode(line):
                out.append(item)
                if self._is_final(item):
                    self.finished = True
                    return out
        return out

    def eof(self) -> None:
        """The body ended: without a final frame that is a broken stream, never a success."""
        if not self.finished:
            raise TransportError(
                "the stream ended before the final frame (the command may have been killed)"
            )


def frame_decoder() -> StreamDecoder[Frame]:
    return StreamDecoder(decode_line, lambda f: isinstance(f, (Exit, Signal, TimedOut)))


def not_a_stream(url: str) -> ProtocolError:
    """A stream request answered 200 without ``application/x-ndjson``: whatever its body, it is not
    read as a one-shot answer (an object that looks like a result would be a clean empty exit 0)."""
    return ProtocolError(
        f"exec endpoint {url} answered a stream request with something that is not a stream"
    )


def token_failed(e: BaseException) -> SandboxError:
    """A token source that raised: the call cannot be authorised, so it is ``auth`` (detail: the
    source's own message), never the source's exception."""
    return AuthError(f"token source failed: {e}")


def checked_token(value: object) -> str:
    """What a token source returned, or ``auth`` when it is not a token (empty, not text)."""
    if not isinstance(value, str) or not value.strip():
        raise AuthError("token source failed: it gave no token")
    return value


def parse_removed(v: Any) -> JobRemoved:
    """A ``remove`` answer: the ids it skipped. An endpoint that predates the mark answers neither."""
    running, mismatch = (v.get(k) if isinstance(v, dict) else None for k in ("running", "mismatch"))
    if not (
        isinstance(running, list)
        and isinstance(mismatch, list)
        and all(isinstance(i, str) for i in (*running, *mismatch))
    ):
        raise NotOfferedError(
            "the endpoint predates the job mark: its remove did not report what it skipped"
        )
    return JobRemoved(tuple(running), tuple(mismatch))
