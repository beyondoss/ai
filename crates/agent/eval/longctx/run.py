"""Run the long-context sweep: task × model × compaction trigger × attempt, headless.

Each run copies a buggy task tree, runs `beyond-ai-agent run --json` in it with compaction set to
fire at the arm's trigger, prices every request from the provider-reported usage
on the event stream (including compaction's own summarization calls, carried on `compacted`), then
restores the pristine tests and grades with the repository's own suite.

Results land in `$LONGCTX_WORK/runs/<task>/<model>/<arm>/a<n>/result.json`. A run with a result is
never repeated, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mcp_tasks
import prefix_tasks
import tasks
from repos import BY_NAME, TASKS

_LOCKS: list = []  # run locks held for the process lifetime (see run_one)

RESERVE = 16_384  # CompactionConfig::default().reserve_tokens; the arms leave it alone.

# name → compaction trigger in tokens. `default` is the model's own window − reserve (the trigger
# before per-model caps); `model` is the binary's own per-model `compact_at` (`list-models` shows it).
# A numeric arm sets both `--context-window` (trigger + reserve, which older binaries understand) and
# `AI_AGENT_COMPACT_AT` (the trigger, which newer ones apply as-is); `default` sets
# `AI_AGENT_COMPACT_AT=0` so a newer binary drops its per-model cap. Any `<n>k` arm name works.
ARMS: dict[str, int | None] = {"100k": 100_000, "200k": 200_000, "default": None, "model": None}


def arm_trigger(arm: str) -> int | None:
    if arm in ARMS:
        return ARMS[arm]
    if arm.endswith("k") and arm[:-1].isdigit():
        return int(arm[:-1]) * 1000
    raise ValueError(f"unknown arm {arm!r}")

# USD per MTok: (input, output, cache read, cache write 5m, cache write 1h); `long` is the
# whole-request card above `long_over` prompt tokens. From crates/providers/src/rates/generated.rs.
PRICES: dict[str, dict] = {
    "claude-opus-5-5": dict(base=(4, 20, 0.2, 5, 8)),
    "claude-sonnet-5-5": dict(base=(2, 10, 0.1, 2.5, 4)),
    "gpt-6.1-sol": dict(base=(2, 10, 0.1, 0, 0), long=(4, 15, 0.2, 0, 0), long_over=272_000),
    "claude-haiku-5-5": dict(base=(0.1, 0.5, 0.01, 0.125, 0.2), long=(0.5, 2.5, 0.05, 0.625, 1), long_over=100_000),
    "gpt-6-luna": dict(base=(0.1, 0.5, 0.01, 0, 0), long=(0.2, 0.75, 0.02, 0, 0), long_over=272_000),
}

# Rough $/run for `--estimate`, by arm; replace with measured means once a pilot exists.
PRIOR_USD = {"claude-opus-5-5": 4.0, "claude-sonnet-5-5": 2.0, "gpt-6.1-sol": 2.0, "claude-haiku-5-5": 0.1,
             "gpt-6-luna": 0.1}


class Ledger:
    """Spend across invocations, with a hard cap. A run reserves its own kill-switch cap (its worst
    case) before it starts and only starts if spent + reserved + cap fits under the hard cap."""

    def __init__(self, path: Path | None, hard_cap: float):
        self.path, self.cap, self.reserved = path, hard_cap, 0.0
        self.lock = threading.Lock()

    def spent(self) -> float:
        if self.path and self.path.exists():
            return sum(json.loads(l)["usd"] for l in self.path.read_text().splitlines() if l.strip())
        return 0.0

    def reserve(self, worst: float) -> bool:
        with self.lock:
            if self.spent() + self.reserved + worst > self.cap:
                return False
            self.reserved += worst
            return True

    def settle(self, worst: float, usd: float, what: str) -> float:
        with self.lock:
            self.reserved -= worst
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"run": what, "usd": round(usd, 5)}) + "\n")
            return self.spent()

_SUSPECT = re.compile(r"\bpip3? (install|download)\b|site-packages|dist-packages|\bcurl\b|\bwget\b|git clone|codeload|pypi\.org")
_DROP = {"text_delta", "thinking_delta", "input_json_delta", "signature_delta"}
_FIRST = {"text_delta", "thinking_delta", "tool_use_start", "input_json_delta", "text_final"}


def price(model: str, u: dict) -> tuple[float, int]:
    """(USD, prompt tokens) for one request's usage."""
    card = PRICES[model]
    prompt = u.get("input_tokens", 0) + u.get("cache_read_tokens", 0) + u.get("cache_write_tokens", 0)
    p = card["base"]
    if "long" in card and prompt > card["long_over"]:
        p = card["long"]
    inp, out, rd, w5, w1 = p
    w1h = u.get("cache_write_1h_tokens", 0)
    usd = (u.get("input_tokens", 0) * inp + u.get("output_tokens", 0) * out
           + u.get("cache_read_tokens", 0) * rd + (u.get("cache_write_tokens", 0) - w1h) * w5 + w1h * w1) / 1e6
    return usd, prompt


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    dotenv = Path(os.environ.get("LONGCTX_ENV", HERE_ROOT / ".env"))
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


HERE_ROOT = Path(__file__).resolve().parents[4]


def prompt_text(repos: list) -> str:
    where = ", ".join(f"`{r.name}/` ({r.slug} at {r.tag})" for r in repos)
    cmds = "\n".join(f"  - `{r.name}/`: `{' '.join(tasks.test_command(r))}`, run from `{r.name}/`" for r in repos)
    srcs = ", ".join(f"`{r.name}/{r.src}`" for r in repos)
    return f"""This workspace holds {"these repositories" if len(repos) > 1 else "this repository"}: {where}. A number of functions and methods across the library source ({srcs}) have had their bodies removed. Each one now consists of its signature, its docstring if it had one, and `raise NotImplementedError("<qualified name>")`. The test suites fail as a result.

Reimplement every removed function body so that every full test suite passes.

Rules:
- Change the library source only. Do not modify, add, or delete test files or test configuration: they are restored to their original versions before grading.
- Keep each function's signature. Implement it to fit the code around it.
- Work only from this workspace. No network access, no pip, no other installed copies of the packages.
- Test commands (the Python environment and PYTHONPATH are already set up):
{cmds}

When every suite passes, reply with a short summary of what you implemented."""


class Live:
    """Folds the event stream into run metrics as it arrives."""

    def __init__(self, model: str):
        self.model = model
        self.usd = 0.0
        self.compaction_usd = 0.0
        self.turns = 0
        self.requests: list[dict] = []
        self.compactions: list[dict] = []
        self.maskings: list[dict] = []
        self.compaction_failed = 0
        self.errors: list[str] = []
        self.suspect: list[str] = []
        self.tools = 0
        self.ended = False
        self.tok = dict(input=0, output=0, cache_read=0, cache_write=0)
        self.turn_t0: float | None = None
        self.ttft: list[float] = []

    def tick(self, ev: dict) -> None:
        """Time to first token: turn start to the first streamed content of that turn."""
        if ev.get("kind") == "turn_start":
            self.turn_t0 = time.monotonic()
        elif self.turn_t0 is not None and ev.get("kind") == "stream" and ev.get("type") in _FIRST:
            self.ttft.append(round(time.monotonic() - self.turn_t0, 3))
            self.turn_t0 = None

    def feed(self, ev: dict) -> None:
        k = ev.get("kind")
        if k == "stream" and ev.get("type") == "usage":
            usd, prompt = price(self.model, ev)
            self.usd += usd
            self.requests.append(dict(turn=self.turns + 1, prompt=prompt, usd=round(usd, 5),
                                      i=ev.get("input_tokens", 0), cr=ev.get("cache_read_tokens", 0),
                                      cw=ev.get("cache_write_tokens", 0), o=ev.get("output_tokens", 0)))
            self._add(ev)
        elif k == "turn_end":
            self.turns += 1
        elif k == "compacted":
            u = ev.get("usage") or {}
            usd, prompt = price(self.model, u)
            self.usd += usd
            self.compaction_usd += usd
            self._add(u)
            self.compactions.append(dict(after_turn=self.turns, reason=ev.get("reason"),
                                         tokens_before=ev.get("tokens_before"), tokens_after=ev.get("tokens_after"),
                                         summary_chars=len(ev.get("summary", "")), usd=round(usd, 5),
                                         i=u.get("input_tokens", 0), cr=u.get("cache_read_tokens", 0),
                                         cw=u.get("cache_write_tokens", 0), o=u.get("output_tokens", 0)))
        elif k == "tool_output_masked":
            self.maskings.append(dict(after_turn=self.turns, masked=ev.get("masked"),
                                      tokens_before=ev.get("tokens_before"), tokens_freed=ev.get("tokens_freed")))
        elif k == "compaction_failed":
            self.compaction_failed += 1
        elif k == "error":
            self.errors.append(str(ev.get("message"))[:500])
        elif k == "tool_start":
            self.tools += 1
            cmd = json.dumps(ev.get("input", {}))
            if ev.get("name") == "bash" and _SUSPECT.search(cmd):
                self.suspect.append(cmd[:300])
        elif k == "agent_end":
            self.ended = True

    def _add(self, u: dict) -> None:
        self.tok["input"] += u.get("input_tokens", 0)
        self.tok["output"] += u.get("output_tokens", 0)
        self.tok["cache_read"] += u.get("cache_read_tokens", 0)
        self.tok["cache_write"] += u.get("cache_write_tokens", 0)


def events_usd(model: str, events: Path) -> float:
    """What a run's event log says it spent, whether or not it finished."""
    live = Live(model)
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                live.feed(json.loads(line))
            except json.JSONDecodeError:
                pass
    return live.usd


def grade_repo(repo, tree: Path, meta: dict) -> dict:
    pristine = tasks.work_dir() / "pristine" / repo.name
    for rel in (*repo.tests, *repo.restore):
        dst, src = tree / rel, pristine / rel
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        elif src.exists():
            shutil.copy2(src, dst)
    passed, failed, dt = tasks.run_tests(repo, tree)
    p0 = set(meta["baseline_passed"])
    broken = set(meta["broken"])
    fixed = broken & passed
    regressed = (p0 - broken) - passed
    return dict(passed=(p0 <= passed), fixed=len(fixed), broken=len(broken), regressed=len(regressed))


def grade(repos: list, root: Path) -> dict:
    per = {r.name: grade_repo(r, root / r.name, json.loads((tasks.work_dir() / "tasks" / f"{r.name}.json").read_text()))
           for r in repos}
    fixed = sum(g["fixed"] for g in per.values())
    broken = sum(g["broken"] for g in per.values())
    return dict(passed=all(g["passed"] for g in per.values()), fixed=fixed, broken=broken,
                regressed=sum(g["regressed"] for g in per.values()), fixed_frac=round(fixed / broken, 4),
                per_repo=per)


def run_one(binary: Path, task: str, model: str, arm: str, attempt: int, cap_usd: float,
            max_steps: int, wall_s: int, extra: list[str], runs_dir: Path, label: str) -> dict:
    d = runs_dir / task / model / arm / f"a{attempt}"
    res_path = d / "result.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    # One process per run: a second sweep over the same item would delete this run's tree from
    # under its agent. The lock lives beside the run dir, which is about to be reset.
    d.parent.mkdir(parents=True, exist_ok=True)
    lock = open(d.parent / f"a{attempt}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"{d}: already running in another process") from None
    _LOCKS.append(lock)  # held until this process exits
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    tree = d / "work"
    home = d / "home"
    home.mkdir()
    env = load_env()
    server = None
    checker = None
    memory = ["--no-memory"]
    if task in prefix_tasks.PREFIX_TASKS:
        tree.mkdir()
        prompt, checker = prefix_tasks.PREFIX_TASKS[task](home, tree)
        memory = []  # the memory index is part of what these tasks measure
        repos = []
    elif task in mcp_tasks.MCP_TASKS:
        tree.mkdir()
        prompt, want = mcp_tasks.MCP_TASKS[task](tree)
        server, base = serve_dir(tree / "site")
        prompt = prompt.replace("{base}", base)
        (home / ".claude").mkdir()
        (home / ".claude" / "settings.json").write_text(
            json.dumps(mcp_tasks.settings(tree, tasks.work_dir() / "mcp-cache")))
        env["GH_TOKEN"] = mcp_tasks.gh_token()
        repos = []
    else:
        repos = [BY_NAME[n] for n in TASKS[task]]
        for r in repos:
            shutil.copytree(tasks.work_dir() / "tasks" / r.name, tree / r.name, symlinks=True)
        env.update(tasks.test_env(repos[0], tree / repos[0].name))
        env["PYTHONPATH"] = ":".join(str(tree / r.name / r.pythonpath) for r in repos)
        prompt = prompt_text(repos)
    env["HOME"] = str(home)  # no user CLAUDE.md, skills, settings or memory leak into the run
    # The run's own temp dir: a full shared /tmp must not fail the agent's test runs (ENOSPC).
    (d / "tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(d / "tmp")
    env["AI_DIRECT"] = "1"
    for k in [k for k in env if k.startswith("AI_AGENT_") or k == "AI_GATEWAY_URL"]:
        del env[k]
    trigger = arm_trigger(arm)
    window = (trigger + RESERVE) if trigger else None
    cmd = [str(binary), "run", "--json", "--model", model, "--max-steps", str(max_steps), *memory,
           "--session-dir", str(d / "sessions"), *extra]
    if window:
        cmd += ["--context-window", str(window)]
        env["AI_AGENT_COMPACT_AT"] = str(trigger)
    elif arm == "default":
        env["AI_AGENT_COMPACT_AT"] = "0"
    cmd.append(prompt)
    live = Live(model)
    t0 = time.monotonic()
    killed = None
    with open(d / "events.jsonl", "w") as evf, open(d / "stderr.txt", "w") as errf:
        p = subprocess.Popen(cmd, cwd=tree, env=env, stdout=subprocess.PIPE, stderr=errf, text=True,
                             start_new_session=True)

        def kill(why: str) -> None:
            nonlocal killed
            if killed is None:
                killed = why
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        timer = threading.Timer(wall_s, kill, args=("wall",))
        timer.start()
        for line in p.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            live.tick(ev)
            if ev.get("type") in _DROP or ev.get("kind") == "tool_progress":
                continue
            evf.write(line)
            live.feed(ev)
            if live.usd > cap_usd:
                kill("cap")
        p.wait()
        timer.cancel()
        try:  # reap any orphaned grandchildren (test runs, MCP servers the agent left behind)
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    wall = time.monotonic() - t0
    if server:
        server.shutdown()
    if not live.requests and killed is None:
        # Never reached the model (credentials, bad flag): not a data point. Leave no result so a
        # rerun retries it.
        raise RuntimeError(f"{d}: agent exited {p.returncode} before any request; see stderr.txt")
    if repos:
        g = grade(repos, tree)
    elif checker:
        ok = bool(checker())
        g = dict(passed=ok, fixed=int(ok), broken=1, regressed=0, fixed_frac=float(ok))
    else:
        ok = mcp_tasks.check(task, tree, want)
        g = dict(passed=ok, fixed=int(ok), broken=1, regressed=0, fixed_frac=float(ok))
    first_c = live.compactions[0]["after_turn"] if live.compactions else None
    prompt_total = sum(r["prompt"] for r in live.requests)
    res = dict(
        task=task, model=model, arm=arm, attempt=attempt, label=label, window=window,
        trigger=(window - RESERVE) if window else None,
        exit=p.returncode, killed=killed, ended=live.ended, wall_s=round(wall, 1),
        usd=round(live.usd, 4), compaction_usd=round(live.compaction_usd, 4),
        turns=live.turns, tools=live.tools, requests=len(live.requests),
        compactions=len(live.compactions), compaction_failed=live.compaction_failed,
        maskings=len(live.maskings), masking_log=live.maskings, agent_args=extra,
        peak_prompt=max((r["prompt"] for r in live.requests), default=0),
        first_prompt=live.requests[0]["prompt"] if live.requests else 0,
        cache_read_share=round(sum(r["cr"] for r in live.requests) / prompt_total, 4) if prompt_total else 0,
        ttft_first=live.ttft[0] if live.ttft else None,
        ttft_median=sorted(live.ttft)[len(live.ttft) // 2] if live.ttft else None,
        tokens=live.tok, errors=live.errors[:5], suspect=live.suspect[:10],
        turns_after_first_compaction=(live.turns - first_c) if first_c is not None else None,
        compaction_log=live.compactions, request_log=live.requests, **g,
    )
    shutil.rmtree(tree)  # graded; the event log and session hold the trajectory
    res_path.write_text(json.dumps(res))
    return res


def serve_dir(root: Path):
    """A localhost static server for a browser task's pages; returns (server, base URL)."""
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    root.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def plan(args) -> list[tuple]:
    names = args.tasks.split(",") if args.tasks else list(TASKS)
    return [(t, m, a, n) for n in range(1, args.attempts + 1) for t in names
            for m in args.models.split(",") for a in args.arms.split(",")]


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="run")
    ap.add_argument("--bin", default=str(HERE_ROOT / "target/release/beyond-ai-agent"))
    ap.add_argument("--models", default="claude-opus-5-5,claude-sonnet-5-5,gpt-6.1-sol")
    ap.add_argument("--arms", default="100k,200k,default",
                    help=f"comma-separated; known: {','.join(ARMS)}, or any <n>k")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cap-usd", type=float, default=10.0, help="kill a run once it has spent this much")
    ap.add_argument("--budget-usd", type=float, default=150.0, help="stop starting runs past this total")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--wall-min", type=int, default=75)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--ledger", default="", help="append-only spend ledger shared across invocations")
    ap.add_argument("--hard-cap-usd", type=float, default=float("inf"),
                    help="with --ledger: never start a run whose worst case (--cap-usd) could pass this total")
    ap.add_argument("--runs-dir", default="", help="where results go (default $LONGCTX_WORK/runs)")
    ap.add_argument("--label", default="", help="recorded on each result (e.g. the binary's commit)")
    ap.add_argument("--agent-arg", dest="extra", action="append", default=[],
                    help="extra flag passed through to `beyond-ai-agent run` (repeatable)")
    a = ap.parse_args(argv)
    todo = plan(a)
    if a.estimate:
        est = sum(PRIOR_USD[m] for _, m, _, _ in todo)
        print(f"{len(todo)} runs, prior estimate ${est:.0f}")
        return
    binary = Path(a.bin).resolve()
    # Grading's junit files go under the work dir too, not the shared /tmp.
    tmp = tasks.work_dir() / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    import tempfile as _tf
    _tf.tempdir = str(tmp)
    for n in {n for t, *_ in todo if t in TASKS for n in TASKS[t]}:
        tasks.build(BY_NAME[n])
    runs_dir = Path(a.runs_dir) if a.runs_dir else tasks.work_dir() / "runs"
    if any(t in mcp_tasks.MCP_TASKS for t, *_ in todo):
        mcp_tasks.prewarm(tasks.work_dir() / "mcp-cache")
    spent = 0.0
    lock = threading.Lock()
    ledger = Ledger(Path(a.ledger) if a.ledger else None, a.hard_cap_usd)

    def go(item):
        nonlocal spent
        with lock:
            if spent > a.budget_usd:
                return None
        d = runs_dir / item[0] / item[1] / item[2] / f"a{item[3]}"
        done = (d / "result.json").exists()
        if not done and not ledger.reserve(a.cap_usd):
            print(f"SKIP {item}: worst case ${a.cap_usd:.2f} would not fit under the ${ledger.cap:.2f} cap "
                  f"(spent ${ledger.spent():.2f})", flush=True)
            return None
        try:
            r = run_one(binary, *item, a.cap_usd, a.max_steps, a.wall_min * 60, a.extra, runs_dir, a.label)
        except Exception as e:  # one broken run must not stop the sweep; it has no result, so reruns retry it
            print(f"ERROR {item}: {e}", flush=True)
            r = None
        finally:
            if not done:
                ledger.settle(a.cap_usd, events_usd(item[1], d / "events.jsonl"), f"{a.label}:{'/'.join(map(str, item))}")
        if r is None:
            return None
        with lock:
            spent += r["usd"]
            tot = spent
        print(f"{r['task']:15} {r['model']:18} {r['arm']:8} a{r['attempt']} pass={int(r['passed'])} "
              f"fixed={r['fixed']}/{r['broken']} ${r['usd']:.2f} turns={r['turns']} comp={r['compactions']} "
              f"mask={r.get('maskings', 0)} "
              f"peak={r['peak_prompt'] // 1000}K killed={r['killed']}  [run total ${tot:.2f}, ledger ${ledger.spent():.2f}]",
              flush=True)
        return r

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, todo))


if __name__ == "__main__":
    main(sys.argv[1:])
