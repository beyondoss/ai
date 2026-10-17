"""Run the long-context sweep: task × model × compaction trigger × attempt, headless.

Each run copies a buggy task tree, runs `beyond-ai-agent run --json` in it with compaction set to
fire at the arm's trigger, prices every request from the provider-reported usage
on the event stream (including compaction's own summarization calls, carried on `compacted`), then
restores the pristine tests and grades with the repository's own suite.

A code task's agent runs in `sandbox.py`'s view: its workspace is `sandbox.WS` whatever the run dir is
called, and no pristine tree, other run or installed copy of a package under test is reachable.
Grading runs every suite in the same view.

Every event in `events.jsonl` carries `t`, seconds since the agent started; each turn's assistant
text is logged as an `assistant_text` event (the stream's deltas are not kept), and a bash
`tool_end` whose output shows a Python error behind a zero exit carries `soft_error`. The result
records why the run ended (`end_reason`), whether the graded tree was cut off mid-edit or its suite
hung (`tree_state`), and, for a tree that no longer parses, the grade of the last state in which
every library file parsed (`last_parseable`). A code task also logs each stubbed function's body as
it changes (`units.jsonl`, see `Units`), which is what redone work is scored from.

`--script` adds scripted user turns: each is injected as a steer once the run has finished its `at`
turns (a `script` event marks the send, `steered` the injection), so a task can add or change a
constraint mid-run and an eval can see whether it survives compaction. The agent then runs as
`serve` over stdio rather than `run --json`; compare scripted runs with scripted runs.

A background task (`bg_tasks.py`: a server to start and curl, an API plus its worker, a watcher, a
slow build) runs in the same sandbox and is graded by its own `check`, which starts the agent's code
afresh. Every sandboxed run records `processes_left` (what the agent left running when it exited,
from the sandbox's report) and `bg` (`bg_metrics.py`: `run_in_background` uses, `timeout_ms` returns
and kills, finish notices, sleep-polls, duplicate launches).

Results land in `$LONGCTX_WORK/runs/<task>/<model>/<arm>/a<n>/result.json`. A run with a result is
never repeated, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import fcntl
import functools
import hashlib
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

import bg_metrics
import bg_tasks
import mcp_tasks
import prefix_tasks
import sandbox
import tasks
from repos import BY_NAME, TASKS

_LOCKS: list = []  # run locks held for the process lifetime (see run_one)

RESERVE = 16_384  # CompactionConfig::default().reserve_tokens; the arms leave it alone.

# name → compaction trigger in tokens. `default` is the model's own window − reserve (the trigger
# before per-model caps); `model` is the binary's own per-model `compact_at` (`list-models` shows it).
# A numeric arm sets `AI_AGENT_COMPACT_AT` (the trigger) and, only for a binary without
# `--compact-at`, `--context-window` (trigger + reserve); `default` sets
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
# A Python error in a bash result the exit code hid (`python x.py; echo done`, `pytest | tail`): a
# traceback, or an error line where Python or pytest puts one (column 0, `E   …`, `FAILED … - …`).
# Anchored so a file listing or a grep hit that merely mentions an error class doesn't count.
_SOFT = re.compile(r"Traceback \(most recent call last\)"
                   r"|^(?:E +|FAILED \S+ - |Sorry: )?(?:\w+\.)*(?:AssertionError|SyntaxError|IndentationError|TabError)\b",
                   re.M)
_CUT_OFF = {"max_steps", "wall", "cap"}


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

    def __init__(self, model: str, flag_net: bool = True):
        self.model = model
        self.flag_net = flag_net  # a background task's curl to its own localhost server is not suspect
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
        self.tool_errors = 0
        self.soft_errors = 0
        self.ended = False
        self.steered: list[int] = []  # the turn after which each scripted steer was injected
        self.tok = dict(input=0, output=0, cache_read=0, cache_write=0)
        self.turn_t0: float | None = None
        self.ttft: list[float] = []
        self.blocks: dict[int, str] = {}  # this message's text blocks, by index
        self.final_text = ""

    def tick(self, ev: dict) -> None:
        """Every event, before the deltas are dropped: time to first token (turn start to the first
        streamed content of that turn) and the turn's assistant text."""
        k, t = ev.get("kind"), ev.get("type")
        if k == "turn_start":
            self.turn_t0 = time.monotonic()
        elif self.turn_t0 is not None and k == "stream" and t in _FIRST:
            self.ttft.append(round(time.monotonic() - self.turn_t0, 3))
            self.turn_t0 = None
        if k == "stream":
            if t == "message_start":
                self.blocks = {}
            elif t == "text_delta":
                self.blocks[ev.get("index", 0)] = self.blocks.get(ev.get("index", 0), "") + ev.get("text", "")
            elif t == "text_final":
                self.blocks[ev.get("index", 0)] = ev.get("text", "")

    def turn_text(self) -> str:
        """The finished message's text (call at `turn_end`); empty for a tool-only turn."""
        text = "".join(self.blocks[i] for i in sorted(self.blocks)).strip()
        self.blocks = {}
        if text:
            self.final_text = text
        return text

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
            if ev.get("name") == "bash" and self.flag_net and _SUSPECT.search(cmd):
                self.suspect.append(cmd[:300])
        elif k == "tool_end":
            if ev.get("is_error"):
                self.tool_errors += 1
            elif ev.get("name") == "bash" and _SOFT.search(str(ev.get("result", ""))):
                self.soft_errors += 1
                ev["soft_error"] = True  # logged with the event
        elif k == "steered":
            self.steered.append(self.turns)
        elif k == "agent_start":
            self.ended = False  # a scripted turn sent as a new prompt starts another run
        elif k == "agent_end":
            self.ended = True

    def end_reason(self, killed: str | None) -> str:
        """Why the run stopped: completed (the agent ended its turn), max_steps, wall, cap, error
        (any other agent error) or crash (exited with neither)."""
        if killed:
            return killed
        if self.ended:
            return "completed"
        if any(e.startswith("reached max steps") for e in self.errors):
            return "max_steps"
        return "error" if self.errors else "crash"

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


def unparseable(repo, tree: Path) -> list[str]:
    """Library files under `tree` that don't parse, as `file:line: message`."""
    out = []
    for p in tasks.source_files(repo, tree):
        try:
            ast.parse(p.read_bytes(), str(p))
        except (SyntaxError, ValueError) as e:
            out.append(f"{p.relative_to(tree).as_posix()}:{getattr(e, 'lineno', '?')}: {type(e).__name__}")
    return out


def grade_repo(repo, tree: Path, meta: dict) -> dict:
    pristine = tasks.work_dir() / "pristine" / repo.name
    for rel in (*repo.tests, *repo.restore):
        dst, src = tree / rel, pristine / rel
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        if src.is_dir():
            tasks.clean_copy(src, dst)
        elif src.exists():
            shutil.copy2(src, dst)
    passed, failed, dt = tasks.run_tests(repo, tree)
    p0 = set(meta["baseline_passed"])
    broken = set(meta["broken"])
    fixed = broken & passed
    regressed = (p0 - broken) - passed
    return dict(passed=(p0 <= passed), fixed=len(fixed), broken=len(broken), regressed=len(regressed),
                hung="<timeout>" in failed, no_report="<no-report>" in failed, unparseable=unparseable(repo, tree))


def grade(repos: list, root: Path) -> dict:
    """The repos' suites on `root`, plus `tree_state`: ok, mid_edit (a library file doesn't parse),
    suite_hang (a suite timed out) or no_report (a suite died before reporting)."""
    per = {r.name: grade_repo(r, root / r.name, json.loads((tasks.work_dir() / "tasks" / f"{r.name}.json").read_text()))
           for r in repos}
    fixed = sum(g["fixed"] for g in per.values())
    broken = sum(g["broken"] for g in per.values())
    bad = [f"{n}/{u}" for n, g in per.items() for u in g["unparseable"]]
    state = ("mid_edit" if bad else "suite_hang" if any(g["hung"] for g in per.values())
             else "no_report" if any(g["no_report"] for g in per.values()) else "ok")
    return dict(passed=all(g["passed"] for g in per.values()), fixed=fixed, broken=broken,
                regressed=sum(g["regressed"] for g in per.values()), fixed_frac=round(fixed / broken, 4),
                tree_state=state, unparseable=bad[:10], per_repo=per)


_COMMENT = re.compile(r"^\s*(#.*?)\s*$")


class Units:
    """Each stubbed function's body over the run, in `units.jsonl`: one line per change, with the
    turn it was seen at (`step`), `state` (`stub` while it still raises the stub's NotImplementedError,
    `impl` once it doesn't, `missing` if no def of that name is left), `digest` (the body's AST, so
    comments and formatting don't count as a change) and its first comment lines (`comments`). The
    initial stubbed state is logged at step 0. What redone work and per-function instructions are
    scored from."""

    def __init__(self, repos: list, tree: Path, out: Path):
        self.want: dict[Path, set[str]] = {}
        for r in repos:
            spec = json.loads((tasks.work_dir() / "tasks" / f"{r.name}.json").read_text())["spec"]
            for s in spec["stubs"]:
                self.want.setdefault(tree / r.name / s["file"], set()).add(s["qualname"])
        self.tree, self.out = tree, out
        self.last: dict[tuple[str, str], tuple] = {}

    def update(self, path: Path, src: bytes, mod: ast.Module, step: int, t: float | None) -> None:
        names = self.want.get(path)
        if not names:
            return
        lines = src.decode("utf-8", "replace").split("\n")
        found: dict[str, tuple] = {}

        def walk(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    walk(child, f"{prefix}{child.name}.")
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{prefix}{child.name}"
                    if qual in names and qual not in found:
                        body = child.body
                        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                                and isinstance(body[0].value.value, str)):
                            body = body[1:]
                        stub = (len(body) == 1 and isinstance(body[0], ast.Raise)
                                and "NotImplementedError" in ast.dump(body[0]) and qual in ast.dump(body[0]))
                        span = lines[child.lineno - 1:child.end_lineno]  # a leading comment precedes body[0]
                        comments = [m.group(1) for l in span if (m := _COMMENT.match(l))][:3]
                        digest = hashlib.sha1("".join(ast.dump(s) for s in body).encode()).hexdigest()[:12]
                        found[qual] = ("stub" if stub else "impl", digest, comments)
                    walk(child, f"{prefix}{child.name}.<locals>.")

        walk(mod, "")
        rel = path.relative_to(self.tree).as_posix()
        with open(self.out, "a") as f:
            for q in sorted(names):
                now = found.get(q, ("missing", None, []))
                if self.last.get((rel, q)) != now:
                    self.last[(rel, q)] = now
                    f.write(json.dumps(dict(step=step, t=t, file=rel, qualname=q, state=now[0], digest=now[1],
                                            comments=now[2])) + "\n")


class Checkpoint:
    """The library source as of the last moment every file in it parsed, kept outside the agent's
    view. Polled after each tool call: only changed files are parsed and copied. A run cut off
    mid-edit is graded on its final tree, and on this too (`last_parseable`). Each changed file that
    parses is also handed to `units`."""

    def __init__(self, repos: list, tree: Path, dest: Path, units: Units | None = None):
        self.roots = [tree / r.name / r.src for r in repos]
        self.tree, self.dest = tree, dest
        self.units = units
        self.mtimes: dict[Path, int] = {}
        self.bad: set[Path] = set()
        self.step = 0  # the turn the snapshot dates from
        self.poll(0)

    def _files(self) -> dict[Path, tuple[int, int]]:
        out = {}
        for root in self.roots:
            for p in ([root] if root.is_file() else root.rglob("*.py")):
                try:
                    st = p.stat()
                    out[p] = (st.st_mtime_ns, st.st_size)
                except OSError:
                    pass
        return out

    def poll(self, step: int, t: float | None = None) -> None:
        now = self._files()
        changed = [p for p, m in now.items() if self.mtimes.get(p) != m]
        removed = [p for p in self.mtimes if p not in now]
        data = {}
        for p in changed:
            try:
                data[p] = p.read_bytes()
                mod = ast.parse(data[p])
                self.bad.discard(p)
            except (SyntaxError, ValueError, OSError):
                self.bad.add(p)
                continue
            if self.units:
                self.units.update(p, data[p], mod, step, t)
        self.bad.difference_update(removed)
        # A file still being written (a background process, a parallel call) may parse half-done:
        # take the snapshot only if nothing moved while it was read.
        if self._files() != now:
            return
        self.mtimes = now
        if (changed or removed) and not self.bad:
            for p in changed:
                out = self.dest / p.relative_to(self.tree)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data[p])
            for p in removed:
                (self.dest / p.relative_to(self.tree)).unlink(missing_ok=True)
            self.step = step

    def restore_into(self, tree: Path) -> None:
        """Make the library `.py` files of `tree` (a copy of the run's tree) the snapshot's."""
        for root in self.roots:
            dst, src = tree / root.relative_to(self.tree), self.dest / root.relative_to(self.tree)
            for p in ([dst] if dst.is_file() else dst.rglob("*.py")):
                p.unlink()
            for p in ([src] if src.is_file() else src.rglob("*.py")):
                out = dst / p.relative_to(src) if src.is_dir() else dst
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)


@functools.cache
def supports_compact_at(binary: Path) -> bool:
    out = subprocess.run([str(binary), "run", "--help"], capture_output=True, text=True, timeout=60)
    return "--compact-at" in out.stdout + out.stderr


def run_one(*args, **kwargs) -> dict:
    """One run. `script` (see `--script`) adds user turns after the prompt, which needs a live
    control channel: the agent then runs as `serve` over stdio, with the same flags, instead of
    `run --json`. Whatever a task holds for the run (a background task's ports and port squatter)
    is released however the run ends."""
    with contextlib.ExitStack() as held:
        return _run_one(held, *args, **kwargs)


def _run_one(held: contextlib.ExitStack, binary: Path, task: str, model: str, arm: str, attempt: int, cap_usd: float,
             max_steps: int, wall_s: int, extra: list[str], runs_dir: Path, label: str,
             script: list[dict] | None = None) -> dict:
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
    bg = None
    memory = ["--no-memory"]
    if task in bg_tasks.BG_TASKS:
        tree.mkdir()
        bg = bg_tasks.BG_TASKS[task](tree)
        held.callback(bg.close)
        prompt = bg.prompt
        env.update(tasks.test_env([]))  # the code tasks' environment: the venv first on PATH
        repos = []
    elif task in prefix_tasks.PREFIX_TASKS:
        tree.mkdir()
        prompt, checker = prefix_tasks.PREFIX_TASKS[task](home, tree)
        memory = []  # the memory index is part of what these tasks measure
        repos = []
    elif task in mcp_tasks.MCP_TASKS:
        tree.mkdir()
        prompt, want = mcp_tasks.MCP_TASKS[task](tree)
        server, base = serve_dir(tree / "site")
        prompt = prompt.replace("{base}", base)
        (home / ".beyond").mkdir()
        (home / ".beyond" / "settings.json").write_text(
            json.dumps(mcp_tasks.settings(tree, tasks.work_dir() / "mcp-cache")))
        env["GH_TOKEN"] = mcp_tasks.gh_token()
        repos = []
    else:
        repos = [BY_NAME[n] for n in TASKS[task]]
        for r in repos:
            shutil.copytree(tasks.work_dir() / "tasks" / r.name, tree / r.name, symlinks=True)
        env.update(tasks.test_env(repos))
        prompt = prompt_text(repos)
    # The run's own temp dir: a full shared /tmp must not fail the agent's test runs (ENOSPC).
    (d / "tmp").mkdir(exist_ok=True)
    (d / "sessions").mkdir(exist_ok=True)
    # A code or background task runs in the sandbox, which sees these four dirs at fixed paths and
    # nothing else of the work dir, and reports what the agent left running when it exited. HOME is
    # the run's own either way: no user CLAUDE.md, skills, settings or memory.
    sandboxed = bool(repos) or bg is not None
    inside = {tree: sandbox.WS, home: f"{sandbox.ROOT}/home", d / "tmp": f"{sandbox.ROOT}/tmp",
              d / "sessions": f"{sandbox.ROOT}/sessions"} if sandboxed else {p: str(p) for p in (tree, home, d / "tmp", d / "sessions")}
    env["HOME"] = inside[home]
    env["TMPDIR"] = inside[d / "tmp"]
    env["AI_DIRECT"] = "1"
    for k in [k for k in env if k.startswith("AI_AGENT_") or k == "AI_GATEWAY_URL"]:
        del env[k]
    trigger = arm_trigger(arm)
    window = (trigger + RESERVE) if trigger else None
    cmd = [str(binary), *(["serve"] if script else ["run", "--json"]), "--model", model, "--max-steps",
           str(max_steps), *memory, "--session-dir", inside[d / "sessions"], *extra]
    if window:
        # A binary that knows `--compact-at` gets only that: an artificial `--context-window` would
        # also shrink what the cached summary may send (live prompt + summary budget must fit the
        # window), so a turn that overshoots the trigger falls back to a full re-render.
        if not supports_compact_at(binary):
            cmd += ["--context-window", str(window)]
        env["AI_AGENT_COMPACT_AT"] = str(trigger)
    elif arm == "default":
        env["AI_AGENT_COMPACT_AT"] = "0"
    # The prompt goes in on stdin, not argv. It names the test command ("python -m pytest tests"),
    # so a model that cleans up with `pkill -f pytest` would match the agent's own argv and kill it.
    # A live probe saw exactly that. `run` reads piped stdin as the first message, trimmed, so the
    # model sees the same text either way.
    cwd = tree
    if sandboxed:
        cmd = sandbox.argv(tasks.view(list(inside.items()), binary=binary), sandbox.WS, cmd,
                           leftovers=str(d / "leftovers.json"))
        cwd = d
    checkpoint = Checkpoint(repos, tree, d / "checkpoint", Units(repos, tree, d / "units.jsonl")) if repos else None
    live = Live(model, flag_net=bg is None)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    t0 = time.monotonic()
    killed = None
    with open(d / "events.jsonl", "w") as evf, open(d / "stderr.txt", "w") as errf:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=errf, text=True, start_new_session=True)
        pending = list(script or [])
        script_log: list[dict] = []

        def send(frame: dict) -> None:
            try:
                p.stdin.write(json.dumps(frame) + "\n")
                p.stdin.flush()
            except BrokenPipeError:
                pass

        def deliver(mode: str, t: float) -> None:
            """The next scripted turn: a `steer` mid-run, or a new `prompt` once the run has ended."""
            item = pending.pop(0)
            i = len(script_log)
            send(dict(id=f"{mode}-{i + 1}", type=mode, message=item["message"]))
            script_log.append(dict(i=i, at=item["at"], mode=mode, turn=live.turns, t=t))
            evf.write(json.dumps(dict(kind="script", i=i, at=item["at"], mode=mode, turn=live.turns, t=t)) + "\n")

        if script:
            send(dict(id="prompt-0", type="prompt", message=prompt))
        else:
            p.stdin.write(prompt)  # a few KB: fits the pipe buffer, so this never waits on the agent
            p.stdin.close()

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
            if script:  # `serve` frames: the agent's events, and each command's response
                if ev.get("type") == "response" and str(ev.get("id", "")).startswith("prompt-"):
                    if not ev.get("success", True):
                        live.errors.append(str(ev.get("error"))[:500])
                    # A run that ended on its own gets the turns it never reached as new prompts; one
                    # that hit an error or the step cap is over.
                    if pending and live.ended and killed is None:
                        deliver("prompt", round(time.monotonic() - t0, 3))
                    else:
                        p.stdin.close()
                    continue
                if ev.get("type") != "event" or not isinstance(ev.get("event"), dict):
                    continue
                ev = ev["event"]
            live.tick(ev)
            if ev.get("type") in _DROP or ev.get("kind") == "tool_progress":
                continue
            live.feed(ev)
            t = round(time.monotonic() - t0, 3)
            evf.write(json.dumps({**ev, "t": t}) + "\n")
            k = ev.get("kind")
            if k == "turn_end" and (text := live.turn_text()):
                evf.write(json.dumps(dict(kind="assistant_text", step=ev.get("step"), text=text, t=t)) + "\n")
            elif k == "tool_end" and checkpoint:
                checkpoint.poll(live.turns, t)
            if k == "turn_end" and pending and live.turns >= pending[0]["at"]:
                deliver("steer", t)
            if live.usd > cap_usd:
                kill("cap")
        p.wait()
        timer.cancel()
        try:  # reap any orphaned grandchildren (test runs, MCP servers the agent left behind)
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if live.blocks:  # cut off mid-message: keep what it had said
            live.turn_text()
    wall = time.monotonic() - t0
    if server:
        server.shutdown()
    # What the agent left running when it exited (the sandbox's report; None if unsandboxed or lost).
    try:
        leftovers = json.loads((d / "leftovers.json").read_text()) if sandboxed else None
    except (OSError, json.JSONDecodeError):
        leftovers = None
    if not live.requests and killed is None:
        # Never reached the model (credentials, bad flag): not a data point. Leave no result so a
        # rerun retries it.
        raise RuntimeError(f"{d}: agent exited {p.returncode} before any request; see stderr.txt")
    end_reason = live.end_reason(killed)
    last_parseable = None
    if repos:
        checkpoint.poll(live.turns, round(wall, 3))  # a kill can land mid-edit, after the last tool_end
        g = grade(repos, tree)
        if g["tree_state"] == "mid_edit":
            # Grade the last state in which every library file parsed, on a copy of the final tree.
            alt = d / "work-last-parseable"
            shutil.copytree(tree, alt, symlinks=True)
            checkpoint.restore_into(alt)
            a = grade(repos, alt)
            shutil.rmtree(alt)
            last_parseable = dict(step=checkpoint.step, passed=a["passed"], fixed=a["fixed"], fixed_frac=a["fixed_frac"],
                                  regressed=a["regressed"], tree_state=a["tree_state"])
        shutil.rmtree(d / "checkpoint", ignore_errors=True)
    elif bg:
        ok, why = bg.check(leftovers)
        g = dict(passed=ok, fixed=int(ok), broken=1, regressed=0, fixed_frac=float(ok), check=why)
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
        exit=p.returncode, killed=killed, ended=live.ended, wall_s=round(wall, 1), started_at=started,
        end_reason=end_reason, cut_off=end_reason in _CUT_OFF, last_parseable=last_parseable,
        sandboxed=sandboxed, tool_errors=live.tool_errors, soft_errors=live.soft_errors,
        final_text=live.final_text[:4000],
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
        script_log=script_log, steered=live.steered,
        compaction_log=live.compactions, request_log=live.requests,
        processes_left=len(leftovers) if leftovers is not None else None, leftovers=(leftovers or [])[:10],
        bg=bg_metrics.metrics(bg_metrics.load(d / "events.jsonl"), bg.launches if bg else ()), **g,
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
    ap.add_argument("--script", default="",
                    help="JSON file of scripted user turns, [{\"at\": <turn>, \"message\": ...}]: each is "
                         "injected as a steer once the run has finished that many turns, or sent as a new "
                         "prompt if the run ended first. Runs the agent as `serve` over stdio.")
    a = ap.parse_args(argv)
    script = json.loads(Path(a.script).read_text()) if a.script else None
    if script and (not isinstance(script, list)
                   or any(set(i) != {"at", "message"} or not isinstance(i["at"], int) for i in script)):
        raise SystemExit(f"{a.script}: want [{{\"at\": <turn>, \"message\": <text>}}, ...]")
    script = sorted(script, key=lambda i: i["at"]) if script else None
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
            r = run_one(binary, *item, a.cap_usd, a.max_steps, a.wall_min * 60, a.extra, runs_dir, a.label, script)
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
              f"mask={r.get('maskings', 0)} left={r.get('processes_left')} "
              f"peak={r['peak_prompt'] // 1000}K killed={r['killed']}  [run total ${tot:.2f}, ledger ${ledger.spent():.2f}]",
              flush=True)
        return r

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, todo))


if __name__ == "__main__":
    main(sys.argv[1:])
