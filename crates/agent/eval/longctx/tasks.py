"""Task construction: fetch pinned repos, pick functions to stub out, build the stubbed checkouts.

A task is one real repository with ~N functions across its package cut back to their signature and
docstring (`raise NotImplementedError`), Commit0-style. The agent must reimplement all of them so the
repository's own test suite passes again; the grade is that suite, with the test files restored from
the pristine copy first. Reimplementing a function well means reading the code around it, so the
work, and the context, spreads across the whole package.

`mine` is a one-time search whose output (`specs/<repo>.json`) is committed, so `build` is
deterministic: same tag + same spec = same task, on any machine.

Every test run (baseline, build, grading, the agent's own) happens inside `sandbox.py`'s view, with
the tree at `sandbox.WS/<repo>`: the same absolute path for every run and arm, and no pristine tree
in sight. `build` refuses a task tree holding compiled or packaged artifacts (`leaks`) or a removed
body that survives verbatim elsewhere in the tree (`copies`).
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import io
import json
import os
import posixpath
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sandbox
from repos import BY_NAME, REPOS, VENV_DEPS, Repo

HERE = Path(__file__).resolve().parent
SPECS = HERE / "specs"
REPO_ROOT = HERE.parents[3]
# Bump when task construction or the test environment changes: cached baselines and task trees
# built under another version are rebuilt.
BUILD = 2


def work_dir() -> Path:
    return Path(os.environ.get("LONGCTX_WORK", REPO_ROOT / "target" / "longctx")).resolve()


# ---------------------------------------------------------------- fetch + venv


def fetch(repo: Repo) -> Path:
    """The pristine tree at `repo.tag` (idempotent: a complete extract is reused)."""
    dest = work_dir() / "pristine" / repo.name
    if (dest / ".fetched").exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    url = f"https://codeload.github.com/{repo.slug}/tar.gz/refs/tags/{repo.tag}"
    data = urllib.request.urlopen(url, timeout=120).read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(tmp, filter="data")
    (top,) = list(tmp.iterdir())
    top.rename(dest)
    tmp.rmdir()
    (dest / ".fetched").write_text(f"{repo.slug}@{repo.tag}\n")
    return dest


# Never copied into a task tree. Compiled bytecode of the original code is the answer key (agents
# disassembled `__pycache__/*.pyc` built from the pristine tree), and a VCS dir, packaging metadata
# or a vendored install can hold the original source.
_ARTIFACT_DIRS = {"__pycache__", ".git", ".hg", ".svn", ".tox", ".nox", ".venv", "venv", "site-packages",
                  ".pytest_cache", ".mypy_cache", ".ruff_cache", ".eggs"}
_ARTIFACT_SUFFIXES = (".pyc", ".pyo", ".so", ".pyd", ".egg-info", ".dist-info", ".egg", ".whl",
                      ".orig", ".rej", ".patch", ".diff")
_ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def _artifact(name: str) -> bool:
    return name in _ARTIFACT_DIRS or name.endswith(_ARTIFACT_SUFFIXES) or name == ".fetched"


def clean_copy(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=True, ignore=lambda _d, names: [n for n in names if _artifact(n)])


def leaks(repo: Repo, tree: Path) -> list[str]:
    """Everything in a task tree that could hand the agent the original code: compiled or packaged
    artifacts anywhere, archives outside the test dirs, and archives in them that carry a library
    module's name. Empty means clean."""
    stubbed = {Path(s["file"]).name for s in json.loads((SPECS / f"{repo.name}.json").read_text())["stubs"]}
    tests = tuple(t.rstrip("/") + "/" for t in repo.tests)
    out = []
    for dirpath, dirs, files in os.walk(tree):
        for n in [*dirs, *files]:
            rel = (Path(dirpath) / n).relative_to(tree).as_posix()
            if _artifact(n):
                out.append(rel)
            elif n.endswith(_ARCHIVES):
                if not rel.startswith(tests):
                    out.append(rel)
                    continue
                try:
                    with zipfile.ZipFile(tree / rel) if zipfile.is_zipfile(tree / rel) else tarfile.open(tree / rel) as a:
                        names = a.namelist() if isinstance(a, zipfile.ZipFile) else a.getnames()
                except (zipfile.BadZipFile, tarfile.TarError):
                    out.append(rel)
                    continue
                out += [f"{rel}!{m}" for m in names if _artifact(Path(m).name) or Path(m).name in stubbed]
    return sorted(out)


def _norm(lines: list[str]) -> list[str]:
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def copies(pristine: Path, tree: Path, stubs: list[dict], own: int = 0) -> list[str]:
    """Stubs whose removed body (4+ code lines, whitespace and comments aside) appears verbatim in
    `tree` more than `own` times (0 for a stubbed tree, 1 for the pristine one): a vendored duplicate
    is a free answer, not a reimplementation. Each as `file:qualname`."""
    corpus = []
    for p in sorted(tree.rglob("*")):
        if p.is_file() and not p.is_symlink():
            try:
                corpus.append("\n".join(_norm(p.read_text().split("\n"))))
            except (UnicodeDecodeError, OSError):
                pass
    out = []
    for s in stubs:
        body = _norm((pristine / s["file"]).read_text().split("\n")[s["line"] - 1:s["end_line"]])
        if len(body) >= 4 and sum(text.count("\n".join(body)) for text in corpus) > own:
            out.append(f"{s['file']}:{s['qualname']}")
    return out


def venv() -> Path:
    """One shared venv: pytest plus every repo's runtime deps, never the packages under test."""
    v = work_dir() / "venv"
    marker = v / ".deps"
    want = sorted(set(VENV_DEPS) | {d for r in REPOS for d in r.deps})
    if marker.exists() and marker.read_text().split() == want:
        return v
    if not (v / "bin" / "python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(v)], check=True)
    subprocess.run([str(v / "bin" / "pip"), "install", "-q", *want], check=True)
    marker.write_text("\n".join(want))
    return v


def site_dir() -> Path:
    """Metadata-only dist-infos for repos whose tests ask for their own installed version."""
    d = work_dir() / "site"
    for r in REPOS:
        if r.dist:
            m = d / f"{r.dist}-{r.version}.dist-info" / "METADATA"
            if not m.exists():
                m.parent.mkdir(parents=True, exist_ok=True)
                m.write_text(f"Metadata-Version: 2.1\nName: {r.dist}\nVersion: {r.version}\n")
    d.mkdir(parents=True, exist_ok=True)
    return d


def view(rw: list[tuple[Path, str]], binary: Path | None = None) -> list:
    """The sandbox for one run: `rw` binds (host dir → path inside) on top of the shared read-only
    toolchain, with the work dir, the repo and every home directory hidden."""
    v = venv()
    cfg = dict(l.split(" = ", 1) for l in (v / "pyvenv.cfg").read_text().splitlines() if " = " in l)
    base = Path(cfg["home"]).resolve().parent  # the venv's base interpreter prefix
    ro = [(str(v), str(v)), (str(base), str(base)), (str(site_dir()), f"{sandbox.ROOT}/site")]
    if binary:
        ro.append((str(binary), str(binary)))
    if rg := shutil.which("rg"):
        ro.append((os.path.realpath(rg), f"{sandbox.ROOT}/bin/rg"))
    # The base interpreter's own site-packages is not on the venv's path but is readable, and may hold
    # one of the packages under test (a mise CPython ships pyparsing).
    after = [str(p) for p in sorted(base.glob("lib/python3*/site-packages")) if p.is_dir()]
    return sandbox.view(hide=[str(Path.home()), str(work_dir()), str(REPO_ROOT)], ro=ro, after=after,
                        rw=[(str(s), d) for s, d in rw])


def test_env(repos: list[Repo]) -> dict[str, str]:
    """The environment of every test run, the agent's included: paths are the sandbox's."""
    v = venv()
    env = {k: val for k, val in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV", "PYTEST"))}
    env["PATH"] = f"{v / 'bin'}:{sandbox.ROOT}/bin:/usr/local/bin:/usr/bin:/bin"
    env["VIRTUAL_ENV"] = str(v)
    env["PYTHONPATH"] = ":".join([*(posixpath.normpath(f"{sandbox.WS}/{r.name}/{r.pythonpath}") for r in repos),
                                  f"{sandbox.ROOT}/site"])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if desel := [d for r in repos for d in r.deselect]:
        env["PYTEST_ADDOPTS"] = " ".join(shlex.quote(f"--deselect={d}") for d in desel)
    return env


def test_command(repo: Repo) -> list[str]:
    return ["python", "-m", "pytest", *repo.tests, *repo.pytest_args]


# A parameter holding today's date and a time of day is `now()` (marshmallow parametrizes on
# `dt.datetime.now().strftime(...)`): its id differs on every run, so a baseline id would never match
# again. Such ids collapse to one, which passes only if every instance passes. Static dates and times
# (`01:23:45`, `2013-11-10`) are left alone.
_PARAM = re.compile(r"\[([^\[\]]*)\]$")
_TIME = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
_DATES = ("%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%d %b %Y", "%b %d %Y")


def test_id(classname: str | None, name: str | None) -> str:
    tid = f"{classname}::{name}"
    m = _PARAM.search(tid)
    if m and _TIME.search(m.group(1)):
        now = datetime.datetime.now()
        days = (now, now - datetime.timedelta(days=1))  # a run that spans midnight
        if any(d.strftime(f) in m.group(1) for d in days for f in _DATES):
            return tid[:m.start()] + "[<now>]"
    return tid


def run_tests(repo: Repo, tree: Path, timeout: float = 600) -> tuple[set[str], set[str], float]:
    """(passed, failed) test ids and wall seconds, run in the sandbox with `tree` at
    `sandbox.WS/<repo>`. A timeout counts every test as failed (`<timeout>`)."""
    scratch = work_dir() / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=scratch))
    ws = f"{sandbox.WS}/{repo.name}"
    env = test_env([repo])
    env["HOME"] = "/tmp"  # the sandbox's own, private and writable
    cmd = [*test_command(repo), "-q", "-p", "no:cacheprovider", f"--junitxml={sandbox.ROOT}/out/junit.xml"]
    argv = sandbox.argv(view([(tree, ws), (out, f"{sandbox.ROOT}/out")]), ws, cmd)
    t0 = time.monotonic()
    with open(out / "stderr", "w") as err:
        p = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=err, start_new_session=True)
        try:
            code = p.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)  # the namespace's init dies with it, and so does the suite
            p.wait()
            shutil.rmtree(out)
            return set(), {"<timeout>"}, time.monotonic() - t0
    dt = time.monotonic() - t0
    if code == sandbox.SETUP_FAILED:
        msg = (out / "stderr").read_text()[-2000:]
        shutil.rmtree(out)
        raise RuntimeError(f"{repo.name}: sandbox setup failed: {msg}")
    passed: set[str] = set()
    failed: set[str] = set()
    try:
        root = ET.parse(out / "junit.xml").getroot()
    except (ET.ParseError, FileNotFoundError):
        shutil.rmtree(out)
        return set(), {"<no-report>"}, dt
    shutil.rmtree(out)
    for tc in root.iter("testcase"):
        tid = test_id(tc.get("classname"), tc.get("name"))
        kinds = {child.tag for child in tc}
        if kinds & {"failure", "error"}:
            failed.add(tid)
        elif "skipped" not in kinds:
            passed.add(tid)
    return passed - failed, failed, dt


# ---------------------------------------------------------------- stubs


def candidates(path: Path, rel: str, min_lines: int, max_lines: int) -> list[dict]:
    """Every function (or method) in one file whose body can be cut back to its docstring."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.split("\n")
    out: list[dict] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{child.name}"
                body = child.body
                has_doc = (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                           and isinstance(body[0].value.value, str))
                stmts = body[1:] if has_doc else body
                decos = {getattr(d, "id", getattr(d, "attr", "")) for d in child.decorator_list}
                if stmts and not decos & {"overload", "abstractmethod"}:
                    start, end = stmts[0].lineno, child.end_lineno
                    first = lines[start - 1]
                    n = end - start + 1
                    if (start > child.lineno and min_lines <= n <= max_lines
                            and not first.lstrip().startswith(("@", "#"))):
                        indent = first[: len(first) - len(first.lstrip())]
                        removed = "\n".join(lines[start - 1:end])
                        out.append(dict(file=rel, qualname=qual, line=start, end_line=end, lines=n,
                                        doc=has_doc, indent=indent,
                                        sha1=hashlib.sha1(removed.encode()).hexdigest()))
                walk(child, f"{prefix}{child.name}.<locals>.")

    walk(tree, "")
    return out


def apply(tree: Path, stubs: list[dict]) -> None:
    """Cut each stub's body to `raise NotImplementedError`, bottom-up so line numbers stay valid."""
    by_file: dict[str, list[dict]] = {}
    for s in stubs:
        by_file.setdefault(s["file"], []).append(s)
    for rel, ss in by_file.items():
        p = tree / rel
        lines = p.read_text().split("\n")
        for s in sorted(ss, key=lambda s: s["line"], reverse=True):
            removed = "\n".join(lines[s["line"] - 1:s["end_line"]])
            if hashlib.sha1(removed.encode()).hexdigest() != s["sha1"]:
                raise ValueError(f"{rel}:{s['line']}: body of {s['qualname']} does not match the spec")
            lines[s["line"] - 1:s["end_line"]] = [f'{s["indent"]}raise NotImplementedError("{s["qualname"]}")']
        p.write_text("\n".join(lines))


def source_files(repo: Repo, tree: Path) -> list[Path]:
    root = tree / repo.src
    if root.is_file():
        return [root]
    files = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(tree).as_posix()
        if any(rel.startswith(t.rstrip("/") + "/") for t in repo.tests) or "/tests/" in rel or "/_vendor/" in rel:
            continue
        files.append(p)
    return files


def baseline(repo: Repo) -> dict:
    """Pristine pass set, cached next to the pristine tree. Run on a clean copy, so nothing a test
    run writes (bytecode) ever lands in the pristine tree a task is built from."""
    tree = fetch(repo)
    f = work_dir() / "pristine" / f"{repo.name}.baseline.json"
    if f.exists():
        b = json.loads(f.read_text())
        if b.get("build") == BUILD:
            return b
    d = Path(tempfile.mkdtemp(dir=work_dir()))
    try:
        clean_copy(tree, d / repo.name)
        passed, failed, dt = run_tests(repo, d / repo.name)
    finally:
        shutil.rmtree(d)
    b = dict(build=BUILD, passed=sorted(passed), failed=sorted(failed), seconds=round(dt, 2))
    f.write_text(json.dumps(b))
    return b


def mine(repo: Repo, target_lines: int, min_lines: int, max_lines: int, sample: int, seed: int, jobs: int) -> dict:
    """Pick functions to stub: each must break some (not most) tests on its own; together they
    total about `target_lines` of removed body, spread across the package's files."""
    import threading

    tree = fetch(repo)
    base = baseline(repo)
    p0 = set(base["passed"])
    if not p0:
        raise RuntimeError(f"{repo.name}: pristine suite has no passing tests")
    cands = [c for p in source_files(repo, tree)
             for c in candidates(p, p.relative_to(tree).as_posix(), min_lines, max_lines)]
    rng = random.Random(seed)
    rng.shuffle(cands)
    pool = cands[:sample]
    timeout = max(30.0, base["seconds"] * 3 + 10)
    scratch = work_dir() / "mine" / repo.name
    if scratch.exists():
        shutil.rmtree(scratch)
    free = []
    for i in range(jobs):
        d = scratch / f"w{i}"
        clean_copy(tree, d)
        free.append(d)
    lock = threading.Lock()

    def probe(s: dict) -> dict | None:
        with lock:
            d = free.pop()
        try:
            orig = (d / s["file"]).read_bytes()
            apply(d, [s])
            passed, failed, _ = run_tests(repo, d, timeout=timeout)
            (d / s["file"]).write_bytes(orig)
            broke = p0 - passed
            if any(f.startswith("<") for f in failed) or not (1 <= len(broke) <= 0.4 * len(p0)):
                return None
            return {**s, "breaks": len(broke)}
        finally:
            with lock:
                free.append(d)

    with ThreadPoolExecutor(jobs) as ex:
        good = [r for r in ex.map(probe, pool) if r]
    shutil.rmtree(scratch)
    # A body that also lives verbatim elsewhere in the tree (a vendored duplicate) is no task.
    dup = set(copies(tree, tree, good, own=1))
    good = [s for s in good if f"{s['file']}:{s['qualname']}" not in dup]
    # Round-robin over files so the work spreads across the package; never two overlapping
    # ranges (a method and a function nested in it).
    by_file: dict[str, list[dict]] = {}
    for s in good:
        by_file.setdefault(s["file"], []).append(s)
    chosen: list[dict] = []
    total = 0
    while total < target_lines and any(by_file.values()):
        for f in sorted(by_file):
            if not by_file[f] or total >= target_lines:
                continue
            s = by_file[f].pop()
            if any(c["file"] == s["file"] and not (s["end_line"] < c["line"] or s["line"] > c["end_line"])
                   for c in chosen):
                continue
            chosen.append(s)
            total += s["lines"]
    d = Path(tempfile.mkdtemp(dir=work_dir()))
    clean_copy(tree, d / "t")
    apply(d / "t", chosen)
    passed, failed, _ = run_tests(repo, d / "t", timeout=timeout)
    shutil.rmtree(d)
    chosen.sort(key=lambda s: (s["file"], s["line"]))
    return dict(repo=repo.name, slug=repo.slug, tag=repo.tag, seed=seed, candidates=len(cands),
                probed=len(pool), good=len(good), lines=total, stubs=chosen,
                baseline_passed=len(p0), broken=len(p0 - passed))


def build(repo: Repo) -> Path:
    """The buggy tree for `repo` from its committed spec (idempotent). Fails if the tree holds
    anything that could hand the agent the original code (`leaks`, `copies`), cached or not."""
    spec = json.loads((SPECS / f"{repo.name}.json").read_text())
    if spec["tag"] != repo.tag:
        raise ValueError(f"{repo.name}: spec is for {spec['tag']}, repo pins {repo.tag}")
    pristine = fetch(repo)
    base = baseline(repo)
    dest = work_dir() / "tasks" / repo.name
    meta = dest.parent / f"{repo.name}.json"
    fresh = not (meta.exists() and dest.exists() and json.loads(meta.read_text()).get("build") == BUILD)
    if fresh:
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        clean_copy(pristine, dest)
        apply(dest, spec["stubs"])
    if bad := leaks(repo, dest):
        raise RuntimeError(f"{repo.name}: task tree leaks the original code: {bad[:10]}")
    if not fresh:
        return dest
    if dup := copies(pristine, dest, spec["stubs"]):
        raise RuntimeError(f"{repo.name}: removed bodies survive verbatim in the task tree: {dup}")
    passed, failed, dt = run_tests(repo, dest)
    p0 = set(base["passed"])
    broken = sorted(p0 - passed)
    if not broken:
        raise RuntimeError(f"{repo.name}: the stubs break nothing")
    meta.write_text(json.dumps(dict(build=BUILD, spec=spec, baseline_passed=sorted(p0), broken=broken,
                                    seconds=round(dt, 2))))
    return dest


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="tasks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("repos", nargs="*")
    m.add_argument("--lines", type=int, default=2500, help="total function-body lines to remove")
    m.add_argument("--min-lines", type=int, default=6)
    m.add_argument("--max-lines", type=int, default=120)
    m.add_argument("--sample", type=int, default=400)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--jobs", type=int, default=8)
    b = sub.add_parser("build")
    b.add_argument("repos", nargs="*")
    bl = sub.add_parser("baseline")
    bl.add_argument("repos", nargs="*")
    a = ap.parse_args(argv)
    names = a.repos or [r.name for r in REPOS]
    if a.cmd == "baseline":
        for n in names:
            b0 = baseline(BY_NAME[n])
            print(f"{n}: {len(b0['passed'])} pass, {len(b0['failed'])} fail, {b0['seconds']}s")
    elif a.cmd == "mine":
        SPECS.mkdir(exist_ok=True)
        for n in names:
            spec = mine(BY_NAME[n], a.lines, a.min_lines, a.max_lines, a.sample, a.seed, a.jobs)
            (SPECS / f"{n}.json").write_text(json.dumps(spec, indent=2) + "\n")
            print(f"{n}: {len(spec['stubs'])} stubs, {spec['lines']} lines ({spec['good']}/{spec['probed']} probes "
                  f"usable); break {spec['broken']}/{spec['baseline_passed']} tests", flush=True)
    elif a.cmd == "build":
        for n in names:
            print(build(BY_NAME[n]))


if __name__ == "__main__":
    main(sys.argv[1:])
