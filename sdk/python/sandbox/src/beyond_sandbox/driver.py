"""The conformance driver (sdk/conformance/DRIVER.md): JSON lines on stdin, answers on stdout.

    python -m beyond_sandbox.driver            # the sync client, driver name "python"
    python -m beyond_sandbox.driver --async    # the asyncio client, driver name "python-async"

(``BEYOND_SANDBOX_DRIVER=async`` also selects the async client.) Every op is the thinnest mapping
from a request to one public call of :class:`~beyond_sandbox.Sandbox` or its async twin; the error
vocabulary is the ``kind`` of the exception. An op this driver does not implement is ``unsupported``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import os
import sys
import traceback
from typing import Any, Callable, cast

from . import AsyncSandbox, Exit, Sandbox, SandboxError, Signal, Stderr, Stdout, TimedOut
from .errors import InvalidRequest
from .files import CapsOverride
from .jobs import JOBS_ROOT
from .models import Stat

Result = dict[str, Any]


class _Unsupported(SandboxError):
    """An op this driver does not implement."""

    kind = "unsupported"


OPS = [
    "hello", "capabilities", "exec", "exec_stream",
    "job_start", "job_list", "job_read", "job_attach", "job_kill", "job_remove", "job_wipe",
    "stat", "read", "write", "write_if_unchanged", "mkdir", "list", "search",
]  # fmt: skip


# ---- request fields --------------------------------------------------------------------------------


def _str(req: dict[str, Any], key: str) -> str:
    v = req.get(key)
    if not isinstance(v, str):
        raise InvalidRequest(f"`{key}` must be a string")
    return v


def _opt_str(req: dict[str, Any], key: str) -> str | None:
    v = req.get(key)
    if v is None or isinstance(v, str):
        return v
    raise InvalidRequest(f"`{key}` must be a string")


def _count(req: dict[str, Any], key: str) -> int | None:
    v = req.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise InvalidRequest(f"`{key}` must be a count")
    return v


def _flag(req: dict[str, Any], key: str) -> bool:
    v = req.get(key)
    if v is None:
        return False
    if not isinstance(v, bool):
        raise InvalidRequest(f"`{key}` must be true or false")
    return v


def _strings(req: dict[str, Any], key: str) -> list[str]:
    v = req.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise InvalidRequest(f"`{key}` must be a list")
    if not all(isinstance(x, str) for x in v):
        raise InvalidRequest(f"`{key}` must hold strings")
    return v


def _bytes(req: dict[str, Any], key: str) -> bytes:
    try:
        return base64.b64decode(_str(req, key), validate=True)
    except (binascii.Error, ValueError) as e:
        raise InvalidRequest(f"`{key}` is not base64: {e}") from None


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


# ---- ops: each is (call on the sandbox, shape of the answer) ---------------------------------------

Plan = tuple[Callable[[Any], Any], Callable[[Any], Result]]


def _stat_result(st: Stat | None) -> Result:
    if st is None:
        return {"exists": False}
    return {
        "exists": True,
        "kind": st.kind,
        "size": st.size,
        "mtime": st.mtime,
        "writable": st.writable,
    }


def _stream_result(items: list[Any], stdout: bytearray, stderr: bytearray) -> Result:
    out: Result = {"stdout_b64": _b64(bytes(stdout)), "stderr_b64": _b64(bytes(stderr))}
    for frame in items:
        if isinstance(frame, Exit):
            out["exit_code"] = frame.code
        elif isinstance(frame, Signal):
            out["signal"] = frame.signal
        elif isinstance(frame, TimedOut):
            out["timed_out"] = True
    return out


def plan(op: str, req: dict[str, Any]) -> Plan:
    if op == "exec":
        command, args = _str(req, "command"), _strings(req, "args")
        cwd, timeout = _opt_str(req, "cwd"), _count(req, "timeout_ms")
        stdin = _bytes(req, "stdin_b64") if req.get("stdin_b64") is not None else None
        kw: dict[str, Any] = {"cwd": cwd, "stdin": stdin}
        if timeout is not None:
            kw["timeout_ms"] = timeout
        return (
            lambda sb: sb.exec(command, args, **kw),
            lambda r: {
                "exit_code": r.exit_code,
                "stdout_b64": _b64(r.stdout),
                "stderr_b64": _b64(r.stderr),
                "exact": r.exact,
            },
        )
    if op == "exec_stream":
        command, args = _str(req, "command"), _strings(req, "args")
        cwd, timeout = _opt_str(req, "cwd"), _count(req, "timeout_ms")
        return (lambda sb: _collect_stream(sb, command, args, cwd, timeout), lambda r: r)
    if op in (
        "job_start",
        "job_list",
        "job_read",
        "job_attach",
        "job_kill",
        "job_remove",
        "job_wipe",
    ):
        return _job_plan(op, req)
    if op == "stat":
        path = _str(req, "path")
        return (lambda sb: sb.files.stat(path), _stat_result)
    if op == "read":
        path, offset, limit = _str(req, "path"), _count(req, "offset"), _count(req, "max")
        return (
            lambda sb: sb.files.read(path, offset or 0, limit),
            lambda b: {"data_b64": _b64(b)},
        )
    if op == "write":
        path, data = _str(req, "path"), _bytes(req, "data_b64")
        return (lambda sb: sb.files.write(path, data), lambda _: {})
    if op == "write_if_unchanged":
        path, data = _str(req, "path"), _bytes(req, "data_b64")
        if "expected_mtime" not in req:
            raise InvalidRequest(
                "`expected_mtime` is required (null means the file must not exist)"
            )
        expected = _count(req, "expected_mtime")
        return (lambda sb: sb.files.write_if_unchanged(path, data, expected), lambda _: {})
    if op == "mkdir":
        path = _str(req, "path")
        return (lambda sb: sb.files.mkdir(path), lambda _: {})
    if op == "list":
        path, hidden, limit = _str(req, "path"), _flag(req, "include_hidden"), _count(req, "limit")
        kws: dict[str, Any] = {"include_hidden": hidden}
        if limit is not None:
            kws["limit"] = limit
        return (
            lambda sb: sb.files.list(path, **kws),
            lambda es: {"entries": [{"name": e.name, "kind": e.kind, "size": e.size} for e in es]},
        )
    if op == "search":
        pattern, root = _str(req, "pattern"), _str(req, "root")
        glob = _opt_str(req, "glob")
        kwq: dict[str, Any] = {
            "literal": _flag(req, "literal"),
            "ignore_case": _flag(req, "ignore_case"),
            "glob": glob,
            "glob_exclude": _flag(req, "glob_exclude"),
            "before": _count(req, "before") or 0,
            "after": _count(req, "after") or 0,
        }
        limit = _count(req, "limit")
        if limit is not None:
            kwq["limit"] = limit
        return (
            lambda sb: sb.files.search(pattern, root, **kwq),
            lambda r: {
                "hits": [
                    {"path": h.path, "line": h.line, "text": h.text, "is_match": h.is_match}
                    for h in r.hits
                ],
                "truncated": r.truncated,
                "note": r.note,
            },
        )
    raise _Unsupported(f"no such op {op!r}")  # unreachable: callers check OPS


def _job_plan(op: str, req: dict[str, Any]) -> Plan:
    scope = _str(req, "scope")
    if op == "job_wipe":
        return (lambda sb: sb.jobs.wipe(scope), lambda s: {"survivors": s})
    if op == "job_list":
        return (
            lambda sb: sb.jobs.list(scope),
            lambda js: {
                "jobs": [
                    {
                        "id": j.id, "state": j.state, "code": j.code, "kept": j.kept, "num": j.num,
                        "kind": j.kind, "seen": j.seen, "started": j.started, "ended": j.ended,
                        "label": j.label,
                    }
                    for j in js
                ]
            },
        )  # fmt: skip
    if op in ("job_kill", "job_remove"):
        ids = _strings(req, "jobs")
        if op == "job_remove":
            return (lambda sb: sb.jobs.remove(scope, ids), lambda _: {})
        remove = _flag(req, "remove")
        return (
            lambda sb: sb.jobs.kill(scope, ids, remove=remove),
            lambda ss: {"survivors": [{"id": s.id, "detail": s.detail} for s in ss]},
        )
    job = _str(req, "job")
    if op == "job_start":
        command, args = _str(req, "command"), _strings(req, "args")
        cwd, label = _opt_str(req, "cwd"), _opt_str(req, "label") or ""
        return (
            lambda sb: sb.jobs.start(scope, job, command, args, cwd=cwd, label=label),
            lambda pgid: {"pgid": pgid},
        )
    offset = _count(req, "offset") or 0
    if op == "job_read":
        limit = _count(req, "max")
        kw: dict[str, Any] = {} if limit is None else {"max": limit}
        return (
            lambda sb: sb.jobs.read(scope, job, offset, **kw),
            lambda r: {
                "state": r.state, "code": r.code, "offset": r.offset, "end": r.end,
                "data_b64": _b64(r.data),
            },
        )  # fmt: skip
    return (  # job_attach
        lambda sb: sb.jobs.attach(scope, job, offset),
        lambda a: {"exit_code": a.exit_code, "output_b64": _b64(a.output), "gap": a.gap},
    )


def _collect_stream(
    sb: Any, command: str, args: list[str], cwd: str | None, timeout: int | None
) -> Any:
    """Sync or async, by what the sandbox is: the one place the driver's two clients differ."""
    if isinstance(sb, AsyncSandbox):
        return _collect_async(sb, command, args, cwd, timeout)
    out, err, final = bytearray(), bytearray(), list[Any]()
    with sb.stream(command, args, cwd=cwd, timeout_ms=timeout) as frames:
        for f in frames:
            _take(f, out, err, final)
    return _stream_result(final, out, err)


async def _collect_async(
    sb: AsyncSandbox, command: str, args: list[str], cwd: str | None, timeout: int | None
) -> Result:
    out, err, final = bytearray(), bytearray(), list[Any]()
    frames = await sb.stream(command, args, cwd=cwd, timeout_ms=timeout)
    try:
        async for f in frames:
            _take(f, out, err, final)
    finally:
        await frames.aclose()
    return _stream_result(final, out, err)


def _take(frame: Any, out: bytearray, err: bytearray, final: list[Any]) -> None:
    if isinstance(frame, Stdout):
        out += frame.data
    elif isinstance(frame, Stderr):
        err += frame.data
    elif isinstance(frame, (Exit, Signal, TimedOut)):
        final.append(frame)


# ---- the loop --------------------------------------------------------------------------------------


def _override() -> CapsOverride | None:
    raw = os.environ.get("CONFORMANCE_CAPS", "").strip()
    if not raw:
        return None
    caps = json.loads(raw)
    out: dict[str, Any] = {}
    for key in ("stdin", "find_printf", "search_engine"):
        if key in caps and caps[key] is not None:
            out[key] = caps[key]
    return cast(CapsOverride, out)


def _fail(id_: Any, e: BaseException) -> Result:
    if isinstance(e, SandboxError):
        return {"id": id_, "error": {"kind": e.kind, "detail": e.detail}}
    traceback.print_exception(e, file=sys.stderr)
    return {"id": id_, "error": {"kind": "io", "detail": f"driver bug: {type(e).__name__}: {e}"}}


class Driver:
    def __init__(self, use_async: bool) -> None:
        self.use_async = use_async
        self.name = "python-async" if use_async else "python"
        url = os.environ.get("CONFORMANCE_URL", "")
        token = os.environ.get("CONFORMANCE_BEARER") or None
        root = os.environ.get("CONFORMANCE_JOBS_ROOT") or JOBS_ROOT
        override = _override()
        if use_async:
            self.sb: Any = AsyncSandbox.connect(url, token, jobs_root=root, override=override)
        else:
            self.sb = Sandbox.connect(url, token, jobs_root=root, override=override)

    def parse(self, line: str) -> tuple[Any, str, dict[str, Any]]:
        try:
            req = json.loads(line)
        except ValueError:
            raise InvalidRequest("the request is not JSON") from None
        if not isinstance(req, dict):
            raise InvalidRequest("the request is not an object")
        op = req.get("op")
        return req.get("id"), op if isinstance(op, str) else "", req

    def meta(self, op: str) -> Any:
        if op == "hello":
            return {"driver": self.name, "protocol": 1, "ops": OPS}
        return self.sb.capabilities()

    @staticmethod
    def caps_result(c: Any) -> Result:
        return {
            "stdin": c.stdin, "stream": c.stream, "jobs": c.jobs, "search_engine": c.search_engine,
            "find_printf": c.find_printf, "read_window": c.read_window,
            "output_base64": c.output_base64,
            # Raw exec returns the command's bytes: over `output_base64`, else over a v1.2 stream.
            "exact_exec": c.output_base64 or c.stream,
        }  # fmt: skip

    def handle(self, line: str) -> Result:
        id_: Any = None
        try:
            id_, op, req = self.parse(line)
            if op not in OPS:
                raise _Unsupported(f"no such op {op!r}")
            if op == "hello":
                return {"id": id_, "ok": self.meta(op)}
            if op == "capabilities":
                return {"id": id_, "ok": self.caps_result(self.meta(op))}
            call, shape = plan(op, req)
            return {"id": id_, "ok": shape(call(self.sb))}
        except Exception as e:
            return _fail(id_, e)

    async def ahandle(self, line: str) -> Result:
        id_: Any = None
        try:
            id_, op, req = self.parse(line)
            if op not in OPS:
                raise _Unsupported(f"no such op {op!r}")
            if op == "hello":
                return {"id": id_, "ok": self.meta(op)}
            if op == "capabilities":
                return {"id": id_, "ok": self.caps_result(await self.meta(op))}
            call, shape = plan(op, req)
            result = call(self.sb)
            if inspect.isawaitable(result):
                result = await result
            return {"id": id_, "ok": shape(result)}
        except Exception as e:
            return _fail(id_, e)


def _emit(answer: Result) -> None:
    sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    use_async = "--async" in args or os.environ.get("BEYOND_SANDBOX_DRIVER") == "async"
    driver = Driver(use_async)
    if use_async:
        asyncio.run(_serve_async(driver))
    else:
        for line in sys.stdin:
            if line.strip():
                _emit(driver.handle(line))
    return 0


async def _serve_async(driver: Driver) -> None:
    # One request at a time and nothing else to run, so a blocking read of stdin is fine.
    while line := sys.stdin.readline():
        if line.strip():
            _emit(await driver.ahandle(line))
    await driver.sb.aclose()


if __name__ == "__main__":
    raise SystemExit(main())
