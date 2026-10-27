"""File, directory and search helpers over raw exec (FILES-OVER-EXEC.md sections 4 and 5).

Pure parsing and command building, plus the operations themselves as sans-IO generators
(see :mod:`beyond_sandbox._drive`): :func:`probe`, :func:`stat`, :func:`read`, :func:`write`,
:func:`write_if_unchanged`, :func:`mkdir`, :func:`list_dir` and :func:`search`.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import re
import secrets
from dataclasses import dataclass, field
from typing import Sequence

from . import _scripts
from ._drive import Op, Run
from .errors import (
    Conflict,
    InvalidQuery,
    IoError,
    NotADirectory,
    NotFound,
    SandboxError,
    Timeout,
    classify_stderr,
)
from .models import Capabilities, Entry, ExecResult, Hit, Kind, SearchResult, Stat

PROBE_TIMEOUT_MS = 10_000
#: The whole environment probe: one ``sh -c`` round trip, one line per capability the sandbox has.
PROBE_SCRIPT = r"""rg --version >/dev/null 2>&1 && echo rg
grep -rnZ -e __cap_probe__ -- /dev/null >/dev/null 2>&1; [ $? -le 1 ] && echo grep_null
[ "$(find /dev/null -maxdepth 0 -printf %y 2>/dev/null)" = c ] && echo find_printf
[ "$(printf abc | dd iflag=skip_bytes,count_bytes skip=1 count=1 status=none 2>/dev/null | base64 -w0 2>/dev/null)" = Yg== ] && echo read_window
true"""
#: Raw bytes per read command: 96 KiB, which is 128 KiB of base64.
READ_CHUNK = 96 * 1024
#: Context lines asked of the engine are clamped to this.
MAX_CONTEXT = 100
#: The longest line of a hit, in characters.
MAX_LINE = 500
LINE_TRUNCATED_SUFFIX = "… [truncated]"
GIT_DIR = ".git"
DEFAULT_LIST_LIMIT = 10_000
DEFAULT_SEARCH_LIMIT = 100


@dataclass(slots=True)
class FsState:
    """What a :class:`~beyond_sandbox.Files` learns about its sandbox: the probe and git roots."""

    caps: Capabilities | None = None
    git_roots: dict[str, bool] = field(default_factory=dict)


def sh(script: str, *args: str) -> list[str]:
    """``sh -c SCRIPT sh ARG...``: a fixed script whose variable parts arrive as ``$1``, ``$2``..."""
    return ["-c", script, "sh", *args]


_OS_ERROR_NAMES = (
    "no such file or directory",
    "directory nonexistent",
    "nonexistent directory",
    "is a directory",
    "not a directory",
    "permission denied",
)


def first_line(stderr: str, fallback: str = "command failed") -> str:
    """The detail of a failed command (3.3): the first line of trimmed stderr; when that line names
    no OS error, the lines through the first one that does (a path containing a newline splits the
    message)."""
    lines = stderr.strip().split("\n")
    if lines == [""]:
        return fallback
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(n in low for n in _OS_ERROR_NAMES):
            return "\n".join(lines[: i + 1]).rstrip()
    return lines[0].rstrip()


def fail(prefix: str, r: ExecResult) -> SandboxError:
    """A command that ran and failed (3.3): ``prefix: first line``, classified by it."""
    line = first_line(r.stderr_text)
    err = classify_stderr(line)
    return type(err)(f"{prefix}: {line}")


# ---- probe -----------------------------------------------------------------------------------------


def probe() -> Op[Capabilities]:
    """FILES-OVER-EXEC.md 4: every environment check in one round trip. A request that fails
    raises; only a script that ran and answered "absent" means absent."""
    r: ExecResult = yield Run("sh", ["-c", PROBE_SCRIPT], timeout_ms=PROBE_TIMEOUT_MS)
    found = {line.strip() for line in r.stdout_text.splitlines()}
    return capabilities(
        "rg" in found, "grep_null" in found, "find_printf" in found, "read_window" in found
    )


def capabilities(rg: bool, grep_null: bool, find_printf: bool, read_window: bool) -> Capabilities:
    engine = "rg" if rg else "grep" if grep_null else "busybox-grep"
    return Capabilities(find_printf, engine, read_window, rg, grep_null)


def with_engine(caps: Capabilities, engine: str) -> Capabilities:
    """``caps`` forced onto one rung of the search ladder (for testing every rung)."""
    rg, grep_null = {"rg": (True, True), "grep": (False, True), "busybox-grep": (False, False)}[
        engine
    ]
    return capabilities(rg, grep_null, caps.find_printf, caps.read_window)


# ---- stat ------------------------------------------------------------------------------------------

_KINDS: dict[str, Kind] = {"f": "file", "d": "dir"}


def parse_stat(stdout: str) -> Stat | None:
    """``W<TAB>T<TAB>SIZE<TAB>MTIME``; anything else is "does not exist"."""
    line = stdout.split("\n", 1)[0]
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    writable = parts[0] == "w"
    kind: Kind = _KINDS.get(parts[1], "other")
    size = _uint(parts[2].strip())
    if size is None:
        return None
    mtime = _uint(parts[3].split(".")[0].strip()) if len(parts) > 3 else None
    return Stat(kind, size, mtime or 0, writable)


def _uint(s: str) -> int | None:
    return int(s) if s.isascii() and s.isdigit() else None


def stat(path: str) -> Op[Stat | None]:
    r = yield Run("sh", sh(_scripts.STAT, path))
    out = r.stdout_text
    if r.exit_code != 0 or not out.strip():
        return None
    return parse_stat(out)


# ---- read ------------------------------------------------------------------------------------------


def decode_base64(s: bytes) -> bytes:
    """Standard base64, whitespace and padding ignored."""
    cleaned = re.sub(rb"[\s=]", b"", s)
    if len(cleaned) % 4 == 1:
        raise ValueError("truncated base64")
    try:
        return base64.b64decode(cleaned + b"=" * (-len(cleaned) % 4), validate=True)
    except binascii.Error as e:
        raise ValueError(str(e)) from None


def read(path: str, offset: int, max_bytes: int | None, caps: Capabilities) -> Op[bytes]:
    if not caps.read_window:
        raise IoError(
            f"read {path}: this target's dd or base64 does not support "
            "iflag=skip_bytes,count_bytes / -w0, so files cannot be read"
        )
    limit = max_bytes if max_bytes is not None else 1 << 62
    out = bytearray()
    while True:
        want = min(limit - len(out), READ_CHUNK)
        r = yield Run("sh", sh(_scripts.READ_WINDOW, path, str(offset + len(out)), str(want)))
        if r.exit_code != 0:
            raise fail(f"read {path}", r)
        try:
            chunk = decode_base64(r.stdout.strip())
        except ValueError as e:
            raise IoError(f"read {path}: {e}") from None
        short = len(chunk) < want
        if not out and (short or want == limit):
            return chunk
        out += chunk
        if short or len(out) >= limit:
            return bytes(out)


# ---- write -----------------------------------------------------------------------------------------


def new_suffix() -> str:
    """Unpredictable (64 bits), lowercase hex: a preplanted temp name cannot redirect a write."""
    return secrets.token_hex(8)


def write(path: str, data: bytes, suffix: str | None = None) -> Op[None]:
    suffix = suffix or new_suffix()
    try:
        r = yield Run("sh", sh(_scripts.WRITE_STDIN, path, suffix, str(len(data))), stdin=data)
    except SandboxError:
        yield from _cleanup(path, suffix)
        raise
    if r.exit_code != 0:
        yield from _cleanup(path, suffix)
        raise fail(f"write {path}", r)


def _cleanup(path: str, suffix: str) -> Op[None]:
    """The compensating action: remove the temp, best effort."""
    with contextlib.suppress(SandboxError):
        yield Run("sh", sh(_scripts.REMOVE_TEMP, path, suffix))


def write_if_unchanged(path: str, data: bytes, expected_mtime: int | None) -> Op[None]:
    """Compare whole epoch seconds, then write. Two round trips; the guard is not a lock."""
    current = yield from stat(path)
    if (current.mtime if current else None) != expected_mtime:
        raise Conflict("the file changed since it was read")
    yield from write(path, data)


# ---- mkdir -----------------------------------------------------------------------------------------


def mkdir(path: str) -> Op[None]:
    r = yield Run("mkdir", ["-p", "--", path])
    if r.exit_code != 0:
        raise fail(f"create {path}", r)


# ---- list ------------------------------------------------------------------------------------------


def parse_dir_entries(stdout: str, limit: int, include_hidden: bool) -> list[Entry]:
    """NUL-terminated ``TYPE<TAB>SIZE<TAB>NAME`` records. Hidden names are dropped before counting."""
    out: list[Entry] = []
    for record in stdout.split("\0"):
        if len(out) >= limit:
            break
        if not record:
            continue
        parts = record.split("\t", 2)
        if len(parts) < 3:
            continue
        kind, size, name = parts
        if not include_hidden and name.startswith("."):
            continue
        if kind in ("N", "?"):
            continue  # a broken link, or unstattable
        size_n = _uint(size.strip())
        out.append(Entry(name, _KINDS.get(kind, "other"), size_n or 0))
    return out


def _find_children(path: str, *tail: str) -> Run:
    return Run("find", ["-H", path, "-mindepth", "1", "-maxdepth", "1", *tail])


def list_dir(path: str, limit: int, include_hidden: bool, caps: Capabilities) -> Op[list[Entry]]:
    if caps.find_printf:
        r = yield _find_children(path, "-printf", "%Y\\t%s\\t%f\\0")
        if r.exit_code != 0 and not r.stdout:
            raise fail(f"ls {path}", r)
        return parse_dir_entries(r.stdout_text, limit, include_hidden)
    # busybox: no -printf, so the type comes from a second pass restricted to directories
    listing = yield _find_children(path, "-print0")
    if listing.exit_code != 0 and not listing.stdout:
        raise fail(f"ls {path}", listing)
    dirs = yield _find_children(path, "-type", "d", "-print0")
    dir_set = {r for r in dirs.stdout_text.split("\0") if r}
    entries: list[Entry] = []
    for raw in listing.stdout_text.split("\0"):
        if not raw or len(entries) >= limit:
            continue
        name = raw.rstrip("/").rsplit("/", 1)[-1] if raw.rstrip("/") else ""
        if not name or name in (".", "..") or (not include_hidden and name.startswith(".")):
            continue
        entries.append(Entry(name, "dir" if raw in dir_set else "file", 0))
    return entries


# ---- search ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Query:
    pattern: str
    root: str
    literal: bool = False
    ignore_case: bool = False
    glob: str | None = None
    glob_exclude: bool = False
    limit: int = DEFAULT_SEARCH_LIMIT
    before: int = 0
    after: int = 0


def _context(q: Query) -> list[str]:
    args: list[str] = []
    before, after = min(q.before, MAX_CONTEXT), min(q.after, MAX_CONTEXT)
    if before > 0:
        args += ["-B", str(before)]
    if after > 0:
        args += ["-A", str(after)]
    return args


def rg_args(q: Query, in_git_repo: bool) -> list[str]:
    args = [
        "--null",
        "--with-filename",
        "--line-number",
        "--no-heading",
        "--color=never",
        "--hidden",
        "--crlf",
    ]
    if not in_git_repo:
        args.append("--no-require-git")
    if q.ignore_case:
        args.append("--ignore-case")
    if q.literal:
        args.append("--fixed-strings")
    if q.glob is not None:
        args += ["--glob", f"!{q.glob}" if q.glob_exclude else q.glob]
    # after the caller's glob on purpose: ripgrep gives the last matching glob precedence
    args += ["--glob", f"!{GIT_DIR}"]
    args += _context(q)
    args += ["-e", q.pattern, "--", q.root]
    return args


def posix_grep_args(q: Query, gnu: bool) -> list[str]:
    args = ["-r", "-n", "-H"]
    if gnu:
        args += ["-Z", "-I", "--color=never"]
    args.append("-F" if q.literal else "-E")
    if q.ignore_case:
        args.append("-i")
    if gnu and q.glob is not None:
        args.append(f"--exclude={q.glob}" if q.glob_exclude else f"--include={q.glob}")
    if gnu and q.root.rstrip("/").rsplit("/", 1)[-1] != GIT_DIR:
        args.append(f"--exclude-dir={GIT_DIR}")
    args += _context(q)
    args += ["-e", q.pattern if q.literal else crlf_tolerant_ere(q.pattern), "--", q.root]
    return args


def crlf_tolerant_ere(pattern: str) -> str:
    """Each ``$`` that anchors the end of a branch becomes ``\\r?$`` (a literal CR byte)."""
    if "$" not in pattern:
        return pattern
    out: list[str] = []
    n = len(pattern)
    i = 0
    while i < n:
        c = pattern[i]
        if c == "\\":
            out.append(c)
            if i + 1 < n:
                out.append(pattern[i + 1])
                i += 1
        elif c == "[":
            start = i
            i += 1
            if i < n and pattern[i] == "^":
                i += 1
            if i < n and pattern[i] == "]":
                i += 1
            while i < n and pattern[i] != "]":
                if pattern[i] == "[" and i + 1 < n and pattern[i + 1] in ":=.":
                    d = pattern[i + 1]
                    i += 2
                    while i + 1 < n and not (pattern[i] == d and pattern[i + 1] == "]"):
                        i += 1
                    i += 1
                i += 1
            end = min(i, n - 1)
            out.append(pattern[start : end + 1])
        elif c == "$" and (i + 1 >= n or pattern[i + 1] in "|)"):
            out.append("\r?$")
        else:
            out.append(c)
        i += 1
    return "".join(out)


_OS_ERRORS = (
    "No such file or directory",
    "Permission denied",
    "Is a directory",
    "Not a directory",
    "Too many levels of symbolic links",
    "Input/output error",
)


@dataclass(frozen=True, slots=True)
class SearchFailure:
    """What an exit above 1 means. ``kind`` is ``bad_regex``, ``missing_root`` or ``skipped``."""

    kind: str
    note: str = ""


def classify_search_failure(stderr: str, root: str) -> SearchFailure:
    """The three causes share exit 2; only stderr tells them apart.

    A path may contain a newline, so stderr is read as records that end at an OS error, not as
    lines; the root's own error is found by its exact string (newlines included).
    """

    def ends_in_os_error(line: str) -> bool:
        line = line.rstrip()
        if line.endswith(")") and " (os error " in line:
            line = line[: line.rindex(" (os error ")]
        return any(line.endswith(e) for e in _OS_ERRORS)

    records: list[str] = []
    start = 0
    end = 0
    for line in _split_lf(stderr):
        end += len(line)
        if ends_in_os_error(line):
            records.append(stderr[start:end].strip())
            start = end
    if not records or stderr[start:].strip():
        return SearchFailure("bad_regex")
    marker = f": {root}: "
    io_prefix = f"IO error for operation on {root}: "
    for rec in records:
        at = rec.find(marker)
        while at >= 0:
            rest = rec[at + len(marker) :].removeprefix(io_prefix)
            if rest.startswith(("No such file or directory", "Not a directory")):
                return SearchFailure("missing_root", rec)
            at = rec.find(marker, at + 1)
    return SearchFailure(
        "skipped", f"{len(records)} path(s) could not be searched; first: {records[0]}"
    )


def _split_lf(s: str) -> list[str]:
    """Lines split on LF alone, each keeping its LF."""
    parts = s.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _eol(s: str) -> str:
    """Remove one trailing LF, then one trailing CR, then every remaining CR."""
    s = s.removesuffix("\n").removesuffix("\r")
    return s.replace("\r", "") if "\r" in s else s


def parse_records(stdout: str) -> list[Hit]:
    """``PATH NUL LINE SEP TEXT LF`` records of ``rg --null`` and ``grep -Z``.

    Scans for the NUL, not for newlines: a path may contain a newline. The group separator is
    matched first, against exactly ``--`` LF (or ``--`` CR LF); a file named ``--`` has a NUL
    after the dashes and is not mistaken for one.
    """
    hits: list[Hit] = []
    i, n = 0, len(stdout)
    while i < n:
        if stdout.startswith("--\n", i):
            i += 3
            continue
        if stdout.startswith("--\r\n", i):
            i += 4
            continue
        if n - i <= 3 and stdout[i:] in ("--", "--\r"):
            break
        nul = stdout.find("\0", i)
        if nul < 0:
            break
        nl = stdout.find("\n", nul)
        if nl < 0:
            nl = n
        hit = _parse_body(stdout[i:nul], stdout[nul + 1 : nl])
        if hit is not None:
            hits.append(hit)
        i = nl + 1
    return hits


_BODY = re.compile(r"([0-9]+)([:-])", re.S)


def _parse_body(path: str, rest: str) -> Hit | None:
    m = _BODY.match(rest)
    if m is None:
        return None
    return Hit(path, int(m.group(1)), _eol(rest[m.end() :]), m.group(2) == ":")


def split_colon_record(line: str) -> tuple[str, int, bool, str] | None:
    """busybox ``PATH:LINE:TEXT`` / ``PATH-LINE-TEXT``: split at the first ``SEP DIGITS SEP``."""
    for i, c in enumerate(line):
        if c not in ":-":
            continue
        j = i + 1
        while j < len(line) and "0" <= line[j] <= "9":
            j += 1
        if j == i + 1 or j >= len(line) or line[j] != c:
            continue
        return line[:i], int(line[i + 1 : j]), c == ":", line[j + 1 :]
    return None


def _lines(stdout: str) -> list[str]:
    """Rust ``str::lines``: split at LF, a CR before it goes too, no empty last line."""
    parts = stdout.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return [p.removesuffix("\r") for p in parts]


def parse_colon_records(stdout: str, q: Query) -> list[Hit]:
    matcher = compile_glob(q.glob, q.ignore_case) if q.glob is not None else None
    hits: list[Hit] = []
    for line in _lines(stdout):
        if line == "--":
            continue
        rec = split_colon_record(line)
        if rec is None:
            continue
        path, line_no, is_match, body = rec
        if matcher is not None and (matcher.fullmatch(path) is not None) == q.glob_exclude:
            continue
        hits.append(Hit(path, line_no, _eol(body), is_match))
    return hits


# ---- glob (the semantics of Rust's globset, with no literal separator) ---------------------------


def compile_glob(pattern: str | None, case_insensitive: bool) -> re.Pattern[str] | None:
    """A glob as a regex: ``*`` crosses ``/``, ``?``, ``[a-z]``/``[!a]``, ``{a,b}``, ``**/`` and
    ``/**``. A pattern that does not parse filters nothing (None)."""
    if pattern is None:
        return None
    try:
        rx = _glob_regex(pattern)
        return re.compile(rx, re.S | (re.I if case_insensitive else 0))
    except (ValueError, re.error):
        return None


def _glob_regex(p: str) -> str:
    out: list[str] = []
    i, n = 0, len(p)
    in_alt = False
    while i < n:
        c = p[i]
        if c == "\\":
            if i + 1 >= n:
                raise ValueError("dangling escape")
            out.append(re.escape(p[i + 1]))
            i += 2
        elif c == "?":
            out.append(".")
            i += 1
        elif c == "*":
            if i + 1 < n and p[i + 1] == "*":
                before_ok = i == 0 or p[i - 1] == "/"
                after = p[i + 2] if i + 2 < n else ""
                if before_ok and i + 2 >= n and i > 0:
                    out.append(".*")  # `/**`: the slash is already emitted, matches the rest
                    i += 2
                    continue
                if before_ok and after == "/":
                    out.append("(?:/?|.*/)" if i == 0 else "(?:|.*/)")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                while i < n and p[i] == "*":
                    i += 1
            else:
                out.append(".*")
                i += 1
        elif c == "[":
            j = i + 1
            neg = j < n and p[j] in "!^"
            if neg:
                j += 1
            start = j
            if j < n and p[j] == "]":
                j += 1
            while j < n and p[j] != "]":
                j += 1
            if j >= n:
                raise ValueError("unclosed class")
            body = p[start:j]
            cls = "".join("\\" + ch if ch in "\\[]^" else ch for ch in body)
            if not cls:
                raise ValueError("empty class")
            out.append(f"[{'^' if neg else ''}{cls}]")
            i = j + 1
        elif c == "{":
            if in_alt:
                raise ValueError("nested alternates")
            in_alt = True
            out.append("(?:")
            i += 1
        elif c == "}" and in_alt:
            in_alt = False
            out.append(")")
            i += 1
        elif c == "," and in_alt:
            out.append("|")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    if in_alt:
        raise ValueError("unclosed alternate")
    return "".join(out)


# ---- post-processing ------------------------------------------------------------------------------


def _components(path: str) -> tuple[bool, list[str]]:
    return path.startswith("/"), [c for c in path.split("/") if c not in ("", ".")]


def is_under_git_dir(root: str, path: str) -> bool:
    """Whether ``path`` is or lies inside a ``.git`` entry *below* ``root``."""
    r_abs, r = _components(root)
    p_abs, p = _components(path)
    if r_abs != p_abs or p[: len(r)] != r:
        return False
    return GIT_DIR in p[len(r) :]


def _sort_key(h: Hit) -> tuple[tuple[str, ...], int]:
    absolute, comps = _components(h.path)
    return (("",) if absolute else ()) + tuple(comps), h.line


def clip(line: str) -> str:
    if len(line) <= MAX_LINE:
        return line
    return line[:MAX_LINE] + LINE_TRUNCATED_SUFFIX


def clip_match(line: str, q: Query) -> str:
    """A long match line is cut to the window around its first match."""
    if len(line) <= MAX_LINE:
        return line
    try:
        rx = re.compile(
            re.escape(q.pattern) if q.literal else q.pattern, re.I if q.ignore_case else 0
        )
    except re.error:
        return clip(line)
    m = rx.search(line)
    if m is None:
        return clip(line)
    cs, ce = m.start(), m.end()
    if ce <= MAX_LINE:
        return clip(line)
    total = len(line)
    pad = max(MAX_LINE - (ce - cs), 0) // 2
    start = max(cs - pad, 0)
    end = start + MAX_LINE
    if end > total:
        end = total
        start = total - MAX_LINE
    tail = " …" if end < total else ""
    return f"… {line[start:end]}{tail} [truncated; columns {start + 1}-{end} of {total}]"


def finalize(hits: list[Hit], limit: int) -> tuple[list[Hit], bool]:
    """Sort by (path components, line), count only matches toward ``limit``, drop dangling context."""
    hits = sorted(hits, key=_sort_key)
    total = sum(1 for h in hits if h.is_match)
    truncated = total > limit
    if truncated:
        seen = 0
        cut = len(hits)
        for i, h in enumerate(hits):
            if h.is_match:
                seen += 1
                if seen > limit:
                    cut = i
                    break
        hits = hits[:cut]
        while hits and not hits[-1].is_match:
            hits.pop()
    return hits, truncated


def root_in_git_repo(state: FsState, root: str) -> Op[bool]:
    """Whether ``root`` or an ancestor holds a ``.git``, asked in the sandbox and cached."""
    if root in state.git_roots:
        return state.git_roots[root]
    try:
        r = yield Run("sh", sh(_scripts.GIT_REPO, root))
    except SandboxError:
        return False  # a failed probe counts as "not inside", and is not remembered
    state.git_roots[root] = r.exit_code == 0
    return state.git_roots[root]


def search(q: Query, state: FsState, caps: Capabilities) -> Op[SearchResult]:
    engine = caps.search_engine
    if engine == "rg":
        args, program = rg_args(q, (yield from root_in_git_repo(state, q.root))), "rg"
    else:
        args, program = posix_grep_args(q, engine == "grep"), "grep"
    r = yield Run(program, args)
    stderr = r.stderr_text
    note: str | None = None
    if r.exit_code in (0, 1):
        pass
    elif r.exit_code == 124:
        raise Timeout(f"{program} timed out")
    else:
        failure = classify_search_failure(stderr, q.root)
        if failure.kind == "bad_regex":
            detail = stderr.strip()
            raise InvalidQuery(f"bad regex: {detail or program + ' rejected the pattern'}")
        if failure.kind == "missing_root":
            raise (NotADirectory if "Not a directory" in failure.note else NotFound)(failure.note)
        note = failure.note
    stdout = r.stdout_text
    hits = parse_colon_records(stdout, q) if engine == "busybox-grep" else parse_records(stdout)
    hits = [h for h in hits if not is_under_git_dir(q.root, h.path)]
    hits = [
        Hit(h.path, h.line, clip_match(h.text, q) if h.is_match else clip(h.text), h.is_match)
        for h in hits
    ]
    hits, truncated = finalize(hits, q.limit)
    if note is None:
        detail = stderr.strip()
        note = detail.split("\n", 1)[0] if detail else None
    return SearchResult(hits, truncated, note)


__all__: Sequence[str] = [
    "FsState",
    "Query",
    "list_dir",
    "mkdir",
    "probe",
    "read",
    "search",
    "stat",
    "write",
    "write_if_unchanged",
]
