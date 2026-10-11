"""Task construction: fetch pinned repos, pick functions to stub out, build the stubbed checkouts.

A task is one real repository with ~N functions across its package cut back to their signature and
docstring (`raise NotImplementedError`), Commit0-style. The agent must reimplement all of them so the
repository's own test suite passes again; the grade is that suite, with the test files restored from
the pristine copy first. Reimplementing a function well means reading the code around it, so the
work, and the context, spreads across the whole package.

`mine` is a one-time search whose output (`specs/<repo>.json`) is committed, so `build` is
deterministic: same tag + same spec = same task, on any machine.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repos import BY_NAME, REPOS, VENV_DEPS, Repo

HERE = Path(__file__).resolve().parent
SPECS = HERE / "specs"


def work_dir() -> Path:
    root = HERE.parents[3]
    return Path(os.environ.get("LONGCTX_WORK", root / "target" / "longctx")).resolve()


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


def test_env(repo: Repo, tree: Path) -> dict[str, str]:
    v = venv()
    env = {k: val for k, val in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    env["PATH"] = f"{v / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}"
    env["VIRTUAL_ENV"] = str(v)
    env["PYTHONPATH"] = str(tree / repo.pythonpath)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def test_command(repo: Repo) -> list[str]:
    return ["python", "-m", "pytest", *repo.tests, *repo.pytest_args]


def run_tests(repo: Repo, tree: Path, timeout: float = 600) -> tuple[set[str], set[str], float]:
    """(passed, failed) test ids and wall seconds. A timeout counts every test as failed."""
    import time

    xml = Path(tempfile.mkstemp(suffix=".xml")[1])
    cmd = [*test_command(repo), "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"]
    t0 = time.monotonic()
    try:
        subprocess.run(cmd, cwd=tree, env=test_env(repo, tree), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        xml.unlink(missing_ok=True)
        return set(), {"<timeout>"}, time.monotonic() - t0
    dt = time.monotonic() - t0
    passed: set[str] = set()
    failed: set[str] = set()
    try:
        root = ET.parse(xml).getroot()
    except (ET.ParseError, FileNotFoundError):
        xml.unlink(missing_ok=True)
        return set(), {"<no-report>"}, dt
    xml.unlink(missing_ok=True)
    for tc in root.iter("testcase"):
        tid = f"{tc.get('classname')}::{tc.get('name')}"
        kinds = {child.tag for child in tc}
        if kinds & {"failure", "error"}:
            failed.add(tid)
        elif "skipped" in kinds:
            continue
        else:
            passed.add(tid)
    return passed, failed, dt


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
    """Pristine pass set, cached next to the pristine tree."""
    tree = fetch(repo)
    f = work_dir() / "pristine" / f"{repo.name}.baseline.json"
    if f.exists():
        return json.loads(f.read_text())
    passed, failed, dt = run_tests(repo, tree)
    b = dict(passed=sorted(passed), failed=sorted(failed), seconds=round(dt, 2))
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
        shutil.copytree(tree, d, symlinks=True)
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
    shutil.copytree(tree, d / "t", symlinks=True)
    apply(d / "t", chosen)
    passed, failed, _ = run_tests(repo, d / "t", timeout=timeout)
    shutil.rmtree(d)
    chosen.sort(key=lambda s: (s["file"], s["line"]))
    return dict(repo=repo.name, slug=repo.slug, tag=repo.tag, seed=seed, candidates=len(cands),
                probed=len(pool), good=len(good), lines=total, stubs=chosen,
                baseline_passed=len(p0), broken=len(p0 - passed))


def build(repo: Repo) -> Path:
    """The buggy tree for `repo` from its committed spec (idempotent)."""
    spec = json.loads((SPECS / f"{repo.name}.json").read_text())
    if spec["tag"] != repo.tag:
        raise ValueError(f"{repo.name}: spec is for {spec['tag']}, repo pins {repo.tag}")
    pristine = fetch(repo)
    base = baseline(repo)
    dest = work_dir() / "tasks" / repo.name
    meta = dest.parent / f"{repo.name}.json"
    if meta.exists() and dest.exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pristine, dest, symlinks=True, ignore=shutil.ignore_patterns(".fetched", ".git"))
    apply(dest, spec["stubs"])
    passed, failed, dt = run_tests(repo, dest)
    p0 = set(base["passed"])
    broken = sorted(p0 - passed)
    if not broken:
        raise RuntimeError(f"{repo.name}: the stubs break nothing")
    meta.write_text(json.dumps(dict(spec=spec, baseline_passed=sorted(p0), broken=broken, seconds=round(dt, 2))))
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
