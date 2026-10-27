"""The exec protocol on the wire, as pure functions: requests, answers, frames, job messages."""

from __future__ import annotations

import base64
import binascii
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

from .errors import AuthError, EndpointError, ProtocolError, SandboxError, TooLarge, TransportError
from .models import (
    ExecResult,
    Exit,
    Frame,
    JobEnd,
    JobFrame,
    JobGap,
    JobInfo,
    JobOutput,
    JobRead,
    JobShellExit,
    JobStarted,
    ShellExit,
    Signal,
    Stderr,
    Stdout,
    Survivor,
    TimedOut,
)

#: The response body cap (FILES-OVER-EXEC.md section 2).
DEFAULT_MAX_RESPONSE = 16 * 1024 * 1024
#: Grace added to ``timeout_ms`` for the whole exchange, so the endpoint's own timeout report wins.
GRACE_S = 30.0
#: A stream with no frame (not even a heartbeat) for this long is a dead connection.
SILENCE_S = 30.0
#: The ``timeout_ms`` of a stream with no deadline of its own: the most every endpoint can hold.
STREAM_DEADLINE_MS = 2**31 - 1
#: The longest NDJSON line accepted: a full 64 KiB frame is about 87 KiB of base64.
MAX_FRAME_LINE = 1024 * 1024
#: The probe's script (exec v1.2).
STREAM_PROBE_SCRIPT = "printf a; sleep 0.3 2>/dev/null || sleep 1; printf b"
#: The least gap between the first output frame and the final one that counts as incremental.
PROBE_MIN_GAP_S = 0.1


def request_body(
    command: str,
    args: Sequence[str],
    *,
    cwd: str | None,
    timeout_ms: int,
    stdin: bytes | None = None,
    stream: bool = False,
    job: dict[str, Any] | None = None,
) -> bytes:
    body: dict[str, Any] = {"command": command, "args": list(args)}
    if cwd is not None:
        body["cwd"] = cwd
    body["timeout_ms"] = timeout_ms
    if stdin is not None:
        body["stdin_base64"] = base64.b64encode(stdin).decode("ascii")
    if stream:
        body["stream"] = True
    if job is not None:
        body["job"] = job
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
        return TooLarge(detail, status=status)
    return EndpointError(detail, status=status)


def too_large(url: str, cap: int) -> TooLarge:
    return TooLarge(
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


def parse_envelope(url: str, body: bytes) -> ExecResult:
    """A v1.1 answer: ``{"exit_code", "stdout", "stderr"}``."""
    obj = parse_json(url, body)
    code = obj.get("exit_code") if isinstance(obj, dict) else None
    if not isinstance(code, int) or isinstance(code, bool):
        raise ProtocolError(
            f"exec endpoint {url} returned unparseable JSON (no exit_code): {excerpt(body)}"
        )
    out, err = obj.get("stdout", ""), obj.get("stderr", "")
    if not isinstance(out, str) or not isinstance(err, str):
        raise ProtocolError(f"exec endpoint {url} returned unparseable JSON: {excerpt(body)}")
    return ExecResult(code, out.encode("utf-8", "replace"), err.encode("utf-8", "replace"))


# ---- v1.2 frames ---------------------------------------------------------------------------------


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
    if isinstance(obj.get("shell_exit"), int):
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


# ---- v1.3 jobs -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobOffer:
    native: bool = False
    notify: bool = False


def parse_offer(answer: Any) -> JobOffer:
    """Only the spec's exact answers offer anything (FILES-OVER-EXEC.md / exec v1.3, N2)."""
    if not isinstance(answer, dict):
        return JobOffer()
    native = answer.get("jobs") == "v1.3"
    return JobOffer(native, native and answer.get("notify") == "v1.4")


def job_op(op: str, scope: str, **extra: Any) -> dict[str, Any]:
    return {"op": op, "scope": scope, **extra}


def decode_job_line(line: bytes) -> list[JobFrame]:
    """One line of a job stream: an event, the final :class:`JobEnd`, or nothing (a heartbeat)."""
    line = line.removesuffix(b"\r")
    if not line.strip():
        return []
    try:
        obj = json.loads(line)
    except ValueError as e:
        raise ProtocolError(f"unparseable job frame ({e}): {excerpt(line)[:200]}") from None
    if not isinstance(obj, dict):
        raise ProtocolError(f"unparseable job frame: {excerpt(line)[:200]}")
    if "error" in obj:
        raise TransportError(f"job stream: the endpoint failed mid-job: {obj['error']}")
    if isinstance(obj.get("started"), int):
        return [JobStarted(obj["started"])]
    for key in ("stdout", "stderr"):
        if key in obj:
            return [JobOutput(_b64(key, obj[key]))]
    if isinstance(obj.get("gap"), int):
        return [JobGap(obj["gap"])]
    if isinstance(obj.get("shell_exit"), int):
        return [JobShellExit(obj["shell_exit"])]
    if isinstance(obj.get("exit_code"), int):
        return [JobEnd(obj["exit_code"])]
    if isinstance(obj.get("signal"), int):
        return [JobEnd(128 + obj["signal"])]
    if obj.get("timed_out") is True:
        raise TransportError("job stream: the endpoint timed the job out")
    return []


JOB_ID = re.compile(r"[0-9a-z]{1,32}")


def _is_job_id(v: Any) -> bool:
    """An id read back from a sandbox is never a path: 1 to 32 characters of ``[0-9a-z]``."""
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


def parse_read(v: Any) -> JobRead:
    st = _state(v.get("state"), v.get("code")) if isinstance(v, dict) else None
    if st is None:
        raise ProtocolError(f"unreadable job read: {v!r}"[:300])
    return JobRead(
        st[0], st[1], _int(v, "offset") or 0, _int(v, "end") or 0, _b64("data", v.get("data", ""))
    )


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


def job_decoder() -> StreamDecoder[JobFrame]:
    return StreamDecoder(decode_job_line, lambda f: isinstance(f, JobEnd))


class ProbeObserver:
    """Judges the v1.2 probe: incremental delivery of ``a`` then ``b``, then exit 0."""

    def __init__(self) -> None:
        self.stdout = bytearray()
        self.output_frames = 0
        self.first: float | None = None
        self.gap = 0.0
        self.end: Frame | None = None

    def see(self, frame: Frame) -> None:
        if isinstance(frame, (Stdout, Stderr)):
            self.first = self.first if self.first is not None else time.monotonic()
            self.output_frames += 1
            if isinstance(frame, Stdout):
                self.stdout += frame.data
        elif isinstance(frame, (Exit, Signal, TimedOut)):
            self.end = frame
            self.gap = time.monotonic() - self.first if self.first is not None else 0.0

    def streams(self) -> bool:
        return (
            self.output_frames >= 2
            and bytes(self.stdout) == b"ab"
            and self.end == Exit(0)
            and self.gap >= PROBE_MIN_GAP_S
        )
