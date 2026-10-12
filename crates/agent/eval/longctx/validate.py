"""Check that the harness measures what it claims before an A/B trusts it. Local only: no model calls.

    validate.py [--repos a,b,…] [--jobs N]

Per repo:
  build      the task tree builds from its spec with no compiled or packaged artifact and no removed
             body surviving verbatim elsewhere in it (`tasks.leaks`, `tasks.copies`)
  pristine   the pristine tree passes 100% in the grading environment: no failure, no error
  stable     every baseline test id comes back on a later run (no clock or random in an id)
  solvable   the stubbed tree with the pristine library source put back, graded by `run.grade` (the
             sweep's own code path), passes with every broken test fixed and none regressed
  paths      ... in every run-dir layout below, with the same result: grading is path-independent
  fails      the stubbed tree fails (the grader can say no; from the build's own run of it)
Then, once:
  sandbox    from inside the agent's view, the pristine trees, the built tasks, other runs, the work
             dir and the repo are unreachable, no process outside the namespace is visible, and no
             copy of a package under test exists anywhere on the reachable filesystem outside the
             workspace
  harness    the run-record plumbing: soft errors, end reasons, assistant text, the mid-edit checkpoint
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run
import sandbox
import tasks
from repos import BY_NAME, REPOS

# Run dirs as the sweep and the A/B gate name them: every arm, a model name, an A/B label with a flag
# suffix. Several hold an `f`, which broke boltons' path-sensitive test before the sandbox.
LAYOUTS = ("runs/utils/claude-opus-5-5/default/a1", "runs/utils/gpt-6.1-sol/100k/a2",
           "ab/f0100041f6+no-tool-masking/utils/claude-sonnet-5-5/model/a1", "s500/x/claude-haiku-5-5/200k/a3")


def solved(repo, tree: Path) -> None:
    """Put the pristine library source back into a stubbed copy: the perfect answer."""
    src = tasks.work_dir() / "pristine" / repo.name / repo.src
    dst = tree / repo.src
    if src.is_dir():
        shutil.rmtree(dst)
        tasks.clean_copy(src, dst)
    else:
        shutil.copy2(src, dst)


def check_repo(repo, scratch: Path) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    try:
        task = tasks.build(repo)
    except Exception as e:  # noqa: BLE001
        return [("build", False, str(e)[:400])]
    meta = json.loads((tasks.work_dir() / "tasks" / f"{repo.name}.json").read_text())
    out.append(("build", True, f"{len(meta['spec']['stubs'])} stubs break {len(meta['broken'])} tests"))
    base = tasks.baseline(repo)
    out.append(("pristine", not base["failed"],
                f"{len(base['passed'])} passed, {len(base['failed'])} failed {base['failed'][:3]}"))
    results = {}
    for lay in LAYOUTS:
        root = scratch / repo.name / lay / "work"
        shutil.copytree(task, root / repo.name, symlinks=True)
        solved(repo, root / repo.name)
        g = run.grade([repo], root)
        shutil.rmtree(scratch / repo.name / lay.split("/")[0])
        results[lay] = {k: g["per_repo"][repo.name][k] for k in ("passed", "fixed", "broken", "regressed")}
    first = results[LAYOUTS[0]]
    # Stability: the perfect answer's ids are a later run of the same suite; a missing baseline id is
    # one that changes between runs.
    out.append(("stable", first["passed"], "every baseline id recurs" if first["passed"] else
                f"{first['broken'] - first['fixed']} baseline ids did not recur"))
    ok = first["passed"] and first["fixed"] == first["broken"] and first["regressed"] == 0
    out.append(("solvable", ok, f"fixed {first['fixed']}/{first['broken']}, regressed {first['regressed']}"))
    same = all(r == first for r in results.values())
    out.append(("paths", same, f"{len(LAYOUTS)} layouts agree" if same else json.dumps(results)))
    shutil.rmtree(scratch / repo.name)
    # The build graded the stubbed tree on the same `run_tests`; `broken` is what it failed.
    n = len(meta["broken"])
    out.append(("fails", n > 0, f"stubbed tree fails {n}/{len(meta['baseline_passed'])} tests"))
    return out


PROBE = r'''
import json, os, sys
forbidden, ws, names = json.loads(sys.argv[1])
reach = [p for p in forbidden if os.path.exists(p) and (not os.path.isdir(p) or os.listdir(p))]


def found(top, dirs, files):
    out = []
    for d in dirs:
        if d in names and os.path.exists(os.path.join(top, d, "__init__.py")):
            out.append(os.path.join(top, d))
        stem = d.split("-")[0].lower().replace("_", "-")
        if d.endswith((".dist-info", ".egg-info")) and stem in names:
            if not set(os.listdir(os.path.join(top, d))) <= {"METADATA"}:  # metadata-only is ours
                out.append(os.path.join(top, d))
    return out + [os.path.join(top, f) for f in files if f.endswith(".py") and f[:-3] in names]


copies, walked = [], 0
for top, dirs, files in os.walk("/", onerror=lambda e: None):
    if top in ("/proc", "/sys", "/dev") or top == ws or top.startswith(ws + "/"):
        dirs[:] = []
        continue
    walked += 1
    copies += found(top, dirs, files)
# The control: the same detector does find the workspace's own copies.
control = [p for top, dirs, files in os.walk(ws) for p in found(top, dirs, files)]
procs = [p for p in os.listdir("/proc") if p.isdigit()]
escape = [p for p in procs for f in forbidden if os.path.exists(f"/proc/{p}/root{f}")]
print(json.dumps(dict(reach=reach, copies=copies, walked=walked, control=control, procs=len(procs), escape=escape)))
'''


def check_sandbox(scratch: Path) -> list[tuple[str, bool, str]]:
    """Probe the agent's own view: same binds, same env, as `run.run_one` builds them."""
    d = scratch / "runs" / "utils" / "claude-opus-5-5" / "default" / "a1"
    repos = [BY_NAME[n] for n in ("boltons", "toolz", "marshmallow")]
    for r in repos:
        shutil.copytree(tasks.work_dir() / "tasks" / r.name, d / "work" / r.name, symlinks=True)
    for sub in ("home", "tmp", "sessions"):
        (d / sub).mkdir(parents=True)
    (d / "work" / ".probe.py").write_text(PROBE)
    inside = [(d / "work", sandbox.WS), (d / "home", f"{sandbox.ROOT}/home"), (d / "tmp", f"{sandbox.ROOT}/tmp"),
              (d / "sessions", f"{sandbox.ROOT}/sessions")]
    binary = tasks.REPO_ROOT / "target" / "release" / "beyond-ai-agent"
    view = tasks.view(inside, binary=binary if binary.exists() else None)
    w = tasks.work_dir()
    # Host paths that exist and must not be reachable from inside (a missing one would pass vacuously).
    forbidden = sorted(p for p in {str(w / "pristine"), str(w / "tasks"), str(w / "runs"), str(w / "ab"), str(scratch),
                                   str(tasks.REPO_ROOT / "crates"), str(Path.home() / ".cache"),
                                   str(Path.home() / ".local" / "lib")} if os.path.exists(p))
    names = sorted({Path(r.src).name.removesuffix(".py") for r in REPOS} | {r.name for r in REPOS})
    env = {**tasks.test_env(repos), "HOME": f"{sandbox.ROOT}/home"}
    t0 = time.monotonic()
    p = subprocess.run(sandbox.argv(view, sandbox.WS, ["python", ".probe.py", json.dumps([forbidden, sandbox.WS, names])]),
                       env=env, capture_output=True, text=True, timeout=900)
    shutil.rmtree(scratch / "runs")
    if p.returncode != 0:
        return [("sandbox", False, f"probe exited {p.returncode}: {p.stderr[-600:]}")]
    r = json.loads(p.stdout)
    return [
        ("sandbox: hidden", not r["reach"], f"{len(forbidden)} host paths unreachable" if not r["reach"] else f"reachable: {r['reach']}"),
        ("sandbox: no copies", not r["copies"] and len(r["control"]) >= len(repos),
         f"no package under test in {r['walked']} reachable dirs outside the workspace; the same detector finds "
         f"the workspace's {len(r['control'])} ({time.monotonic() - t0:.0f}s)" if not r["copies"]
         else f"copies: {r['copies'][:10]}"),
        ("sandbox: /proc", not r["escape"], f"{r['procs']} processes visible, none reaching the host view"
         if not r["escape"] else f"escape via /proc: {r['escape'][:5]}"),
    ]


def check_harness(scratch: Path) -> list[tuple[str, bool, str]]:
    """The run-record plumbing, on synthetic events and a scratch tree."""
    out = []
    live = run.Live("claude-haiku-5-5")
    cases = [("python x.py; echo done", "Traceback (most recent call last):\n  File ...\nValueError: x\ndone", True),
             ("pytest | tail", "E       AssertionError: assert 1 == 2\n1 failed", True),
             ("python -c 'x'", '  File "<string>", line 1\n    if\nSyntaxError: invalid syntax', True),
             ("grep -rn AssertionError", "tests/a.py:12:    raise AssertionError('no')", False),
             ("cat x.py", "def f():\n    raise IndentationError  # never", False)]
    for i, (cmd, result, want) in enumerate(cases):
        live.feed(dict(kind="tool_start", id=str(i), name="bash", input=dict(command=cmd)))
        ev = dict(kind="tool_end", id=str(i), name="bash", result=result, is_error=False)
        live.feed(ev)
        if bool(ev.get("soft_error")) != want:
            out.append(("harness: soft errors", False, f"{cmd!r} → {ev.get('soft_error')}, want {want}"))
    live.feed(dict(kind="tool_end", id="e", name="bash", result="boom", is_error=True))
    if not out:
        ok = live.soft_errors == 3 and live.tool_errors == 1
        out.append(("harness: soft errors", ok, f"{live.soft_errors} soft, {live.tool_errors} hard"))
    for t in ("message_start", "text_delta", "text_delta"):
        live.tick(dict(kind="stream", type=t, index=0, text="All suites " if t == "text_delta" and not live.blocks else "pass."))
    text = live.turn_text()
    out.append(("harness: assistant text", text == "All suites pass.", repr(text)))
    reasons = []
    for errors, ended, killed in (([], True, None), (["reached max steps (250) without completion"], False, None),
                                  ([], False, "wall"), ([], False, "cap"), (["api: 500"], False, None), ([], False, None)):
        lv = run.Live("claude-haiku-5-5")
        lv.errors, lv.ended = errors, ended
        reasons.append(lv.end_reason(killed))
    want = ["completed", "max_steps", "wall", "cap", "error", "crash"]
    out.append(("harness: end reasons", reasons == want, ", ".join(reasons)))
    repo = BY_NAME["toolz"]
    tree = scratch / "ckpt" / "work"
    shutil.copytree(tasks.work_dir() / "tasks" / repo.name, tree / repo.name, symlinks=True)
    ck = run.Checkpoint([repo], tree, scratch / "ckpt" / "snap")
    f = tree / repo.name / "toolz" / "dicttoolz.py"
    good = f.read_text() + "\nGOOD = 1\n"
    f.write_text(good)
    ck.poll(3)
    time.sleep(0.01)
    f.write_text(good + "def broken(:\n")
    ck.poll(4)
    bad = run.unparseable(repo, tree / repo.name)
    alt = scratch / "ckpt" / "alt"
    shutil.copytree(tree, alt, symlinks=True)
    ck.restore_into(alt)
    restored = (alt / repo.name / "toolz" / "dicttoolz.py").read_text()
    shutil.rmtree(scratch / "ckpt")
    out.append(("harness: checkpoint", ck.step == 3 and restored == good and len(bad) == 1,
                f"mid-edit tree flagged ({bad[:1]}), last parseable state is step {ck.step}"))
    return out


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="validate")
    ap.add_argument("--repos", default="", help="comma-separated (default: all)")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    repos = [BY_NAME[n] for n in a.repos.split(",")] if a.repos else list(REPOS)
    t0 = time.monotonic()
    (tasks.work_dir() / "tmp").mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="validate-", dir=tasks.work_dir() / "tmp"))
    rows: list[tuple[str, str, bool, str]] = []
    try:
        tasks.venv()
        with ThreadPoolExecutor(a.jobs) as ex:
            for r, res in zip(repos, ex.map(lambda r: check_repo(r, scratch), repos)):
                rows += [(r.name, *c) for c in res]
                for c in res:
                    print(f"{'PASS' if c[1] else 'FAIL'}  {r.name:15} {c[0]:9} {c[2]}", flush=True)
        for c in [*check_sandbox(scratch), *check_harness(scratch)]:
            rows.append(("-", *c))
            print(f"{'PASS' if c[1] else 'FAIL'}  {'-':15} {c[0]:9} {c[2]}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    bad = [r for r in rows if not r[2]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} checks pass in {time.monotonic() - t0:.0f}s"
          f"{'' if not bad else f'; {len(bad)} FAIL'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
