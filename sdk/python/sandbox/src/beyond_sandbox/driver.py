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
import contextlib
import hashlib
import inspect
import json
import os
import shutil
import stat as stat_mod
import sys
import tempfile
import traceback
from typing import Any, Callable, cast

from . import AsyncSandbox, Exit, Sandbox, SandboxError, Signal, Stderr, Stdout, TimedOut, _bulk
from .errors import InvalidRequest, IoError
from .files import CapsOverride
from .models import Stat

Result = dict[str, Any]


class _Unsupported(SandboxError):
    """An op this driver does not implement."""

    kind = "unsupported"


# What a scenario's `requires.features` may name (sdk/conformance/DRIVER.md): this client reads the
# raw stream, so it has all of them.
FEATURES = ["frames", "unbounded_stream", "exit_signal"]

OPS = [
    "hello", "capabilities", "exec", "exec_stream",
    "job_start", "job_list", "job_read", "job_attach", "job_kill", "job_remove", "job_wipe",
    "stat", "read", "write", "write_if_unchanged", "mkdir", "list", "search",
    "upload", "download", "extract",
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


def _number(req: dict[str, Any], key: str) -> float | None:
    v = req.get(key)
    if v is None:
        return None
    if (
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or v < 0
        or v != v
        or v == float("inf")
    ):
        raise InvalidRequest(f"`{key}` must be a non-negative number")
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
        expected = _number(req, "expected_mtime")
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


# ---- bulk copy: trees (DRIVER.md, "Trees") ---------------------------------------------------------

_BULK_OPS = ("upload", "download", "extract")


def _os_error(e: OSError, path: object = "") -> SandboxError:
    return _bulk.local_error(path, e)


def _tree(req: dict[str, Any], key: str) -> list[dict[str, Any]]:
    v = req.get(key)
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(e, dict) for e in v):
        raise InvalidRequest(f"`{key}` must be a list of entries")
    return v


def _materialize(root: str, entries: list[dict[str, Any]], outside: str) -> None:
    """Build the local tree an entry list describes. Directory modes are applied last."""
    dirs: list[tuple[str, int]] = []
    times: list[tuple[str, int]] = []
    for e in entries:
        rel = _str(e, "path")
        path = os.path.join(root, rel)
        # The directories this creates are 0755 whatever the umask.
        missing: list[str] = []
        p = os.path.dirname(path)
        while p != root and not os.path.exists(p):
            missing.append(p)
            p = os.path.dirname(p)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dirs.extend((m, 0o755) for m in missing)
        raw_mode = e.get("mode")
        try:
            mode = int(raw_mode, 8) if isinstance(raw_mode, str) else 0o644
        except ValueError:
            raise InvalidRequest(f"bad mode {raw_mode!r}") from None
        kind = e.get("type", "file")
        if kind == "dir":
            os.makedirs(path, exist_ok=True)
            dirs = [d for d in dirs if d[0] != path]
            dirs.append((path, mode if isinstance(raw_mode, str) else 0o755))
        elif kind == "symlink":
            os.symlink(_str(e, "target").replace("${OUTSIDE}", outside), path)
        elif kind == "fifo":
            os.mkfifo(path)
        else:
            with open(path, "wb") as f:
                if isinstance(e.get("pattern_b64"), str):
                    pat = _bytes(e, "pattern_b64")
                    left = _count(e, "size") or 0
                    if not pat:
                        raise InvalidRequest("`pattern_b64` is empty")
                    step = (1 << 20) - (1 << 20) % len(pat) if len(pat) <= 1 << 20 else len(pat)
                    block = pat * (step // len(pat))
                    while left > 0:
                        n = min(left, len(block))
                        f.write(block[:n])
                        left -= n
                elif isinstance(e.get("data_b64"), str):
                    f.write(_bytes(e, "data_b64"))
            os.chmod(path, mode)
            if (t := _count(e, "mtime")) is not None:
                times.append((path, t))
    for path, t in times:
        os.utime(path, (t, t))
    for path, mode in sorted(dirs, key=lambda d: -d[0].count("/")):
        os.chmod(path, mode)


def _describe(root: str) -> list[dict[str, Any]]:
    """What is under ``root``, sorted by path."""
    out: list[dict[str, Any]] = []

    def walk(d: str) -> None:
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            rel = os.path.relpath(p, root)
            st = os.lstat(p)
            m = st.st_mode
            if stat_mod.S_ISLNK(m):
                out.append({"path": rel, "type": "symlink", "target": os.readlink(p)})
            elif stat_mod.S_ISDIR(m):
                out.append(
                    {
                        "path": rel,
                        "type": "dir",
                        "mode": f"{m & 0o7777:04o}",
                        "mtime": int(st.st_mtime),
                    }
                )
                # A mode-000 directory is listed, not entered.
                if m & 0o500 == 0o500:
                    walk(p)
            else:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    while block := f.read(1 << 20):
                        h.update(block)
                out.append(
                    {
                        "path": rel, "type": "file", "mode": f"{m & 0o7777:04o}",
                        "size": st.st_size, "sha256": h.hexdigest(), "mtime": int(st.st_mtime),
                    }
                )  # fmt: skip

    walk(root)
    return out


def _transfer(t: Any, entries: list[dict[str, Any]] | None) -> Result:
    out: Result = {"files": t.files, "dirs": t.dirs, "links": t.links, "bytes": t.bytes}
    if entries is not None:
        out["entries"] = entries
    return out


class _Scratch:
    """The sentinel layout of a download-side op: the target is ``<tmp>/local``; ``<tmp>/outside``
    is where a ``${OUTSIDE}`` link points. Anything else that appears, or anything in
    ``outside``, is a write that escaped its target, and fails the op."""

    def __init__(self, req: dict[str, Any]) -> None:
        self.tmp = tempfile.mkdtemp(prefix="bulk-")
        self.local = os.path.join(self.tmp, "local")
        self.outside = os.path.join(self.tmp, "outside")
        os.mkdir(self.local)
        os.mkdir(self.outside)
        try:
            _materialize(self.local, _tree(req, "prefill"), self.outside)
        except OSError as e:
            self.close()
            raise _os_error(e) from None

    def check(self) -> None:
        extra = [n for n in os.listdir(self.tmp) if n not in ("local", "outside")]
        if extra:
            raise IoError(f"the extraction wrote outside its directory: {extra[0]}")
        if os.listdir(self.outside):
            raise IoError("the extraction wrote through a link into `outside`")

    def describe(self) -> list[dict[str, Any]]:
        try:
            return _describe(self.local)
        except OSError as e:
            raise _os_error(e) from None

    def close(self) -> None:
        # Modes may have made directories unreadable; the scratch is ours to open up.
        for dirpath, dirnames, _ in os.walk(self.tmp):
            for d in dirnames:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(dirpath, d), 0o700)
        shutil.rmtree(self.tmp, ignore_errors=True)


def _upload_args(
    req: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[str] | None, int | None]:
    dest = _str(req, "path")
    if not isinstance(req.get("entries"), list):
        raise InvalidRequest("`entries` must be a list")
    sources = _strings(req, "sources") if req.get("sources") is not None else None
    return dest, _tree(req, "entries"), sources, _count(req, "chunk_bytes")


def _bulk_sync(op: str, req: dict[str, Any], sb: Any) -> Result:
    if op == "upload":
        dest, entries, names, chunk = _upload_args(req)
        tmp = tempfile.mkdtemp(prefix="bulk-up-")
        try:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            try:
                _materialize(src, entries, tmp)
            except OSError as e:
                raise _os_error(e) from None
            sources: Any = [os.path.join(src, n) for n in names] if names is not None else src
            return _transfer(sb.upload(sources, dest, chunk_bytes=chunk), None)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    scratch = _Scratch(req)
    try:
        err: BaseException | None = None
        result = None
        try:
            if op == "download":
                result = sb.download(_str(req, "path"), scratch.local)
            else:
                result = _bulk.extract(_bytes(req, "tar_b64"), scratch.local)
        except SandboxError as e:
            err = e
        scratch.check()
        if err is not None:
            raise err
        return _transfer(result, scratch.describe())
    finally:
        scratch.close()


async def _bulk_async(op: str, req: dict[str, Any], sb: Any) -> Result:
    if op == "upload":
        dest, entries, names, chunk = _upload_args(req)
        tmp = tempfile.mkdtemp(prefix="bulk-up-")
        try:
            src = os.path.join(tmp, "src")
            os.mkdir(src)
            try:
                _materialize(src, entries, tmp)
            except OSError as e:
                raise _os_error(e) from None
            sources: Any = [os.path.join(src, n) for n in names] if names is not None else src
            return _transfer(await sb.upload(sources, dest, chunk_bytes=chunk), None)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    scratch = _Scratch(req)
    try:
        err: BaseException | None = None
        result = None
        try:
            if op == "download":
                result = await sb.download(_str(req, "path"), scratch.local)
            else:
                result = _bulk.extract(_bytes(req, "tar_b64"), scratch.local)
        except SandboxError as e:
            err = e
        scratch.check()
        if err is not None:
            raise err
        return _transfer(result, scratch.describe())
    finally:
        scratch.close()


# ---- the loop --------------------------------------------------------------------------------------


def _override() -> CapsOverride | None:
    raw = os.environ.get("CONFORMANCE_CAPS", "").strip()
    if not raw:
        return None
    caps = json.loads(raw)
    out: dict[str, Any] = {}
    for key in ("find_printf", "search_engine"):
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
        override = _override()
        if use_async:
            self.sb: Any = AsyncSandbox.connect(url, token, override=override)
        else:
            self.sb = Sandbox.connect(url, token, override=override)

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
            return {"driver": self.name, "protocol": 1, "ops": OPS, "features": FEATURES}
        return self.sb.capabilities()

    @staticmethod
    def caps_result(c: Any) -> Result:
        return {
            "search_engine": c.search_engine,
            "find_printf": c.find_printf,
            "read_window": c.read_window,
        }

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
            if op in _BULK_OPS:
                return {"id": id_, "ok": _bulk_sync(op, req, self.sb)}
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
            if op in _BULK_OPS:
                return {"id": id_, "ok": await _bulk_async(op, req, self.sb)}
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
