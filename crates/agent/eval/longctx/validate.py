"""Check that the harness measures what it claims before an A/B trusts it. Local only: no model calls.

    validate.py [--repos a,b,…] [--jobs N] [--bg-only] [--real-run <run dir>]

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
Background tasks (`bg_tasks.py`), all eight set up at once, as eight parallel runs would:
  bg: <task> the untouched task fails its check, three times; the reference solution passes, three
             times; for a task with answer files, the answers alone and the edits alone each fail;
             `bg-cleanup` fails with a process left running or no leftover report
  bg: ports  the eight tasks' ports are distinct, below the ephemeral range, held while the task is
             open (the port squatter's port refuses a bind) and free once it is closed
  bg: procs  no process the graders started outlives them
  bg: build  `build.py` run for real takes 90–150 s and writes exactly the bundle `check` expects
  bg: leftovers  the sandbox reports what a command left running (`&`, `setsid`, `nohup`), and
             nothing for a command that cleaned up
  bg: metrics  `bg_metrics` on a synthetic event log with every counted pattern; with --real-run, on
             a real run's log (its bash count matches the log's)
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bg_metrics
import bg_tasks
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


def _bg_task(name: str, scratch: Path, ports: dict) -> tuple[str, bool, str]:
    """One background task end to end: untouched fails ×3, the reference passes ×3, partial ones fail.
    Records in `ports[name]` the ports it held, the ports its config names, and whether each was
    bindable while the task was open and once it was closed."""
    work = scratch / "bg" / name / "work"
    work.mkdir(parents=True)
    t = bg_tasks.BG_TASKS[name](work)
    held = list(t.lease.ports) if t.lease else []
    cfg = json.loads((work / "config.json").read_text()) if (work / "config.json").exists() else {}
    ports[name] = dict(held=held, config=[v for k, v in cfg.items() if k.endswith("port")],
                       open=[bg_tasks._bindable(p) for p in held])
    try:
        notes, ok = [], True
        before = [t.check([])[0] for _ in range(3)]
        ok &= not any(before)
        notes.append(f"untouched fails {before.count(False)}/3")
        if t.answers:  # half a solution must fail too
            orig = scratch / "bg" / name / "orig"
            shutil.copytree(work, orig, symlinks=True)
            for part, what in ((dataclasses.replace(t, fix=[]), "answers alone"), (dataclasses.replace(t, answers={}), "edits alone")):
                bg_tasks.apply_fix(work, part)
                got = t.check([])[0]
                shutil.rmtree(work)
                shutil.copytree(orig, work, symlinks=True)
                ok &= not got
                notes.append(f"{what} {'PASS' if got else 'fail'}")
        bg_tasks.apply_fix(work, t)
        after = [t.check([]) for _ in range(3)]
        ok &= all(a[0] for a in after)
        notes.append(f"reference passes {sum(a[0] for a in after)}/3")
        if name == "bg-cleanup":
            left, none = t.check([{"pid": 7, "cmd": "python kv.py", "job_tag": False}])[0], t.check(None)[0]
            ok &= not left and not none
            notes.append(f"with a process left {'PASS' if left else 'fail'}, unsandboxed {'PASS' if none else 'fail'}")
        if not all(a[0] for a in after):
            notes.append(after[0][1][:200])
        return f"bg: {name}", ok, ", ".join(notes)
    except Exception as e:  # noqa: BLE001
        return f"bg: {name}", False, f"{type(e).__name__}: {e}"[:300]
    finally:
        t.close()
        ports[name]["closed"] = [bg_tasks._bindable(p) for p in held]


def _graders_left(scratch: Path) -> list[str]:
    """Processes whose working directory is under `scratch` or the graders' scratch copies."""
    roots = (str(scratch), str(tasks.work_dir() / "tmp" / "bgcheck-"))
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if cwd.startswith(roots):
            out.append(f"{pid} {cwd}")
    return out


def _bg_build(scratch: Path) -> tuple[str, bool, str]:
    work = scratch / "bg-build" / "work"
    work.mkdir(parents=True)
    t = bg_tasks.BG_TASKS["bg-long-build"](work)
    try:
        t0 = time.monotonic()
        p = subprocess.run([bg_tasks.python(), "build.py"], cwd=work, capture_output=True, text=True, timeout=300)
        dt = time.monotonic() - t0
        same = (work / "dist" / "bundle.json").exists() and (work / "dist" / "bundle.json").read_text() == t.answers["dist/bundle.json"]
        ok = p.returncode == 0 and same and 90 <= dt <= 150
        return "bg: build", ok, f"{dt:.0f} s, {len(p.stdout.splitlines())} progress lines, bundle {'matches' if same else 'DIFFERS'}"
    finally:
        t.close()


def _bg_leftovers(scratch: Path) -> tuple[str, bool, str]:
    d = scratch / "bg-sandbox"
    for sub in ("work", "home", "tmp"):
        (d / sub).mkdir(parents=True)
    view = tasks.view([(d / "work", sandbox.WS), (d / "home", f"{sandbox.ROOT}/home"), (d / "tmp", f"{sandbox.ROOT}/tmp")])
    got = []
    for cmd in ("sleep 60 & setsid sleep 61 & nohup sleep 62 > /dev/null 2>&1 & echo started", "sleep 60 & kill $!; echo clean"):
        out = d / "left.json"
        subprocess.run(sandbox.argv(view, sandbox.WS, ["bash", "-c", cmd], leftovers=str(out)), capture_output=True, timeout=60,
                       env={**tasks.test_env([]), "HOME": f"{sandbox.ROOT}/home"})
        got.append(sorted(p["cmd"] for p in json.loads(out.read_text())))
    ok = got == [["sleep 60", "sleep 61", "sleep 62"], []]
    return "bg: leftovers", ok, f"left behind: {got[0]}; after cleaning up: {got[1]}"


def _ev(i: int, name: str, inp: dict, result: str) -> list[dict]:
    return [dict(kind="tool_start", id=str(i), name=name, input=inp),
            dict(kind="tool_end", id=str(i), name=name, result=result, is_error=False)]


SYNTHETIC = [
    *_ev(1, "bash", {"command": "python app/server.py > /tmp/s.log 2>&1", "run_in_background": True},
         "(no output yet)\n\nRunning in the background as job 1 (it is stopped when the run ends)."),
    *_ev(2, "bash", {"command": "sleep 2; curl -s localhost:9/items", "timeout_ms": 5000}, "{}"),
    *_ev(3, "bash", {"command": "python app/server.py &"},
         "The command exited but left a process (`python app/server.py`) running; it is tracked as job 2"),
    *_ev(4, "bash", {"command": "tail -20 logs/server.log"}, "INFO ..."),
    *_ev(5, "bash", {"command": "tail -20  logs/server.log"}, "INFO ..."),
    dict(kind="background", reason="finished", text="[job 1 exited with code 1 after 9 s: `python app/server.py`]\nOSError"),
    *_ev(6, "bash", {"command": "python build.py", "timeout_ms": 1000},
         "[1/40] compiled\n\nReturned after your timeout_ms of 1000 ms. The command is still running as job 3 (1 s so far)."),
    *_ev(7, "bash", {"command": "python -m pytest -x", "timeout_ms": 600000}, "Command timed out after 600 seconds"),
    *_ev(8, "bash_job", {"job": 2, "kill": True}, "Job 2 killed after 3 s."),
    *_ev(9, "bash", {"command": "python app/server.py > /tmp/s2.log 2>&1", "run_in_background": True},
         "Running in the background as job 4 (it is stopped when the run ends)."),
    *_ev(10, "bash", {"command": "pkill -f server.py"}, ""),
    *_ev(11, "bash", {"command": "nohup python app/server.py > /tmp/s3.log 2>&1 &"}, ""),
    *_ev(12, "bash", {"command": "while ! curl -s localhost:9/health; do sleep 1; done"},
         "No output for 60 s. The command is still running as job 5 (60 s so far)."),
    *_ev(13, "bash_job", {"job": 3}, "[40/40] compiled\nBUILD OK\n\nCommand exited with code 0"),
    dict(kind="background", reason="still_running", text="[background jobs] You are ending your turn with jobs still running"),
    dict(kind="bash_jobs_stopped", jobs=[{"job": 4}, {"job": 5}]),
]
SYNTHETIC_WANT = dict(bash_calls=11, bash_job_calls=2, job_waits=1, job_kills=1, run_in_background=2, shell_background=2,
                      timeout_ms_set=3, timeout_ms_short=2, timeout_returns=1, timeout_kills=1, idle_yields=1, notices=1,
                      reminders=1, sleep_polls=3, launches=5, duplicate_launches=1, jobs_stopped_at_end=2)


def _bg_metrics(real: str) -> list[tuple[str, bool, str]]:
    got = bg_metrics.metrics(SYNTHETIC)
    diff = {k: (got[k], v) for k, v in SYNTHETIC_WANT.items() if got[k] != v}
    out = [("bg: metrics", not diff, f"synthetic log: all {len(SYNTHETIC_WANT)} counts as expected" if not diff
            else f"(got, want): {diff}")]
    if real:
        evs = bg_metrics.load(Path(real) / "events.jsonl")
        m = bg_metrics.metrics(evs)
        n = sum(1 for e in evs if e.get("kind") == "tool_start" and e.get("name") == "bash")
        out.append(("bg: metrics (real)", bool(evs) and m["bash_calls"] == n,
                    f"{real}: {n} bash calls, {m['timeout_ms_set']} with timeout_ms, {m['timeout_kills']} killed by it, "
                    f"{m['sleep_polls']} sleep-polls"))
    return out


def check_bg(scratch: Path, real: str) -> list[tuple[str, bool, str]]:
    """Every background task at once (eight parallel runs' worth), plus the build, the sandbox's
    leftover report and the metrics."""
    ports: dict = {}
    names = list(bg_tasks.BG_TASKS)
    with ThreadPoolExecutor(len(names) + 2) as ex:
        build = ex.submit(_bg_build, scratch)
        rows = list(ex.map(lambda n: _bg_task(n, scratch, ports), names))
        rows.append(build.result())
    held = [p for n in names for p in ports[n]["held"]]
    cfg = [p for n in names for p in ports[n]["config"]]
    lo, hi = bg_tasks._port_range()
    squatted = ports["bg-port-conflict"]["open"]
    freed = all(all(ports[n].get("closed", [False])) for n in names)
    ok = (len(held) == len(set(held)) and sorted(held) == sorted(cfg) and all(lo <= p < hi for p in held)
          and squatted == [False] and freed)
    rows.append(("bg: ports", ok, f"{len(held)} ports, {len(set(held))} distinct, each its task's config, in [{lo}, {hi}); "
                 f"the squatted port {'refused' if squatted == [False] else 'ACCEPTED'} a bind while open; "
                 f"{'all free' if freed else 'NOT all free'} after close"))
    stray = _graders_left(scratch)
    rows.append(("bg: procs", not stray, "no grader process outlived its check" if not stray else f"left: {stray[:5]}"))
    rows.append(_bg_leftovers(scratch))
    return rows + _bg_metrics(real)


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="validate")
    ap.add_argument("--repos", default="", help="comma-separated (default: all)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--bg-only", action="store_true", help="only the background-task checks")
    ap.add_argument("--real-run", default="", help="a run dir whose events.jsonl the metrics check also reads")
    a = ap.parse_args(argv)
    repos = [] if a.bg_only else [BY_NAME[n] for n in a.repos.split(",")] if a.repos else list(REPOS)
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
        rest = [] if a.bg_only else [*check_sandbox(scratch), *check_harness(scratch)]
        for c in [*rest, *check_bg(scratch, a.real_run)]:
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
