"""Before/after A/B of two agent commits on the same tasks: a no-regression gate.

    ab.py run --base <sha> --cand <sha> [--models …] [--tasks …] [--attempts N]
    ab.py report --base <sha> --cand <sha>

Builds each commit from `git archive` in its own target dir (the binary is cached under
`$LONGCTX_WORK/bin`, the target dir deleted), then runs every task × model × attempt on both at
the same compaction arm (`--arm`, default `default`: window − reserve, no per-model cap; `model` uses
each binary's own per-model trigger). `--base-arg`/`--cand-arg` pass extra agent flags to one side, so
one commit can be A/B'd against itself with a mechanism toggled (the labels then carry the flags). MCP tasks get a third arm, the candidate with `--code-mode` (built with
`--features code-mode`); the plain candidate arm is tool search, the default. Results go to
`$LONGCTX_WORK/ab/<label>/…`; the report pairs each candidate arm against the base on task × attempt.

The background tasks (`bg_tasks.py`, `--bg-tasks`) get a section of their own: per arm, what each run
did with long-running processes (`bg_metrics.py`) and what it left running, and the pre-registered
ship rule for background jobs (`SHIP_RULE`), which also checks the code tasks against an earlier
baseline run (`--code-baseline`, a runs root such as `$LONGCTX_WORK/ab/<label>`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import bg_tasks
import mcp_tasks
import prefix_tasks
import run
import tasks
from repos import TASKS
from report import boot_ci, endings, errs, mcnemar, wilson

ROOT = run.HERE_ROOT


def resolve(sha: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{sha}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def build(sha: str, code_mode: bool = False) -> Path:
    """The release `beyond-ai-agent` at `sha` (idempotent: a cached binary is reused)."""
    full = resolve(sha)
    label = full[:10] + ("-cm" if code_mode else "")
    out = tasks.work_dir() / "bin" / f"beyond-ai-agent-{label}"
    if out.exists():
        return out
    src = tasks.work_dir() / "src" / full[:10]
    if not (src / "Cargo.toml").exists():
        src.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", full], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(src)], input=archive, check=True)
    target = tasks.work_dir() / f"target-{label}"
    cmd = ["cargo", "build", "--release", "-p", "beyond-ai-agent"] + (["--features", "code-mode"] if code_mode else [])
    subprocess.run(cmd, cwd=src, env={**os.environ, "CARGO_TARGET_DIR": str(target)}, check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / "release" / "beyond-ai-agent", out)
    shutil.rmtree(target)
    return out


def tag(flags: list[str]) -> str:
    """A label suffix naming extra agent flags: `--no-tool-masking` → `+no-tool-masking`."""
    return "".join("+" + f.lstrip("-").replace("=", "-").replace(" ", "-") for f in flags)


def arms(a) -> list[tuple[str, str, bool, list[str], bool]]:
    """(label, sha, code-mode build, extra flags, MCP tasks only)."""
    b, c = resolve(a.base)[:10], resolve(a.cand)[:10]
    bx, cx = list(a.base_arg), list(a.cand_arg)
    out = [(b + tag(bx), a.base, False, bx, False), (c + tag(cx), a.cand, False, cx, False)]
    if out[0][0] == out[1][0]:
        raise SystemExit("base and candidate are the same commit with the same flags")
    if not cx:  # the Code Mode arm only makes sense against the plain candidate
        out.append((f"{c}-cm", a.cand, True, ["--code-mode"], True))
    return out


def cmd_run(a) -> None:
    code = a.tasks.split(",") if a.tasks else list(TASKS)
    mcp = a.mcp_tasks.split(",") if a.mcp_tasks else list(mcp_tasks.MCP_TASKS)
    ctx = a.ctx_tasks.split(",") if a.ctx_tasks else list(prefix_tasks.PREFIX_TASKS)
    bg = [] if a.bg_tasks == "none" else a.bg_tasks.split(",") if a.bg_tasks else list(bg_tasks.BG_TASKS)
    for label, sha, cm, extra, mcp_only in arms(a):
        binary = build(sha, cm)
        names = mcp if mcp_only else code + ctx + mcp + bg
        if not names:
            continue
        argv = ["--bin", str(binary), "--models", a.models, "--arms", a.arm, "--tasks", ",".join(names),
                "--attempts", str(a.attempts), "--jobs", str(a.jobs), "--cap-usd", str(a.cap_usd),
                "--budget-usd", str(a.budget_usd), "--runs-dir", str(tasks.work_dir() / "ab" / label),
                "--hard-cap-usd", str(a.hard_cap_usd), *(["--ledger", a.ledger] if a.ledger else []),
                "--label", label]
        argv += [f"--agent-arg={e}" for e in extra]  # `=`: a value starting with `--` is otherwise a flag
        run.main(argv)
    cmd_report(a)


def load(label: str, arm: str) -> dict[tuple, dict]:
    root = tasks.work_dir() / "ab" / label
    out = {}
    for p in root.glob(f"*/*/{arm}/*/result.json"):
        r = json.loads(p.read_text())
        out[(r["task"], r["model"], r["attempt"])] = r
    return out


def verdict_pass(b: int, c: int, ci: tuple[float, float, float]) -> str:
    if (c > b and mcnemar(b, c) < 0.05) or ci[2] < 0:
        return "regression"
    if (b > c and mcnemar(b, c) < 0.05) or ci[1] > 0:
        return "improvement"
    return "neutral"


def verdict_cost(ci: tuple[float, float, float]) -> str:
    """For a lower-is-better quantity: Δ = cand − base."""
    if ci[1] > 0:
        return "regression"
    if ci[2] < 0:
        return "improvement"
    return "neutral"


def med(xs: list) -> float:
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else float("nan")


def cmd_report(a) -> None:
    labels = arms(a)
    base_label = labels[0][0]
    base = load(base_label, a.arm)
    family = lambda t: ("mcp" if t in mcp_tasks.MCP_TASKS else "prefix" if t in prefix_tasks.PREFIX_TASKS
                        else "bg" if t in bg_tasks.BG_TASKS else "code")
    print(f"Base {base_label}, candidate {labels[1][0]} (`-cm`: candidate with --code-mode, MCP tasks only).\n")
    print("| arm | model | family | n | pass (95% CI) | $/task | wall s/task | TTFT first / median s | turns | first-request tok | cache-read share | compactions | maskings | end reasons | tool errors hard / soft |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    data = {lab: load(lab, a.arm) for lab, *_ in labels}
    for lab, *_ in labels:
        cell = defaultdict(list)
        for (t, m, n), r in data[lab].items():
            cell[(m, family(t))].append(r)
        for (m, f), rs in sorted(cell.items()):
            k, n = sum(r["passed"] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            avg = lambda key: sum(r.get(key) or 0 for r in rs) / n
            print(f"| {lab} | {m} | {f} | {n} | {k}/{n} ({lo:.0%}–{hi:.0%}) | ${avg('usd'):.2f} | {avg('wall_s'):.0f} "
                  f"| {med([r.get('ttft_first') for r in rs]):.2f} / {med([r.get('ttft_median') for r in rs]):.2f} "
                  f"| {avg('turns'):.0f} | {avg('first_prompt'):.0f} | {avg('cache_read_share'):.0%} | {avg('compactions'):.2f} "
                  f"| {avg('maskings'):.2f} | {endings(rs)} | {errs(rs)} |")
    print("\nPaired against the base (task × attempt). Verdicts: pass = McNemar p<0.05 or the Δ-fixed CI excludes 0; "
          "cost and time = the bootstrap CI of the difference excludes 0. A pair where either side was cut off "
          "(max_steps, wall, cap) is still counted; `cut-off pairs` says how many.\n")
    print("| arm | model | family | pairs | pass cand-only / base-only (p) | Δ fixed (95% CI) | pass verdict | Δ $ (95% CI) | cost verdict | Δ wall s (95% CI) | time verdict | Δ first-request tok | cut-off pairs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, *_ in labels[1:]:
        groups = defaultdict(list)
        for key, r in data[lab].items():
            if key in base:
                groups[(key[1], family(key[0]))].append((key, r, base[key]))
        for (m, f), ps in sorted(groups.items()):
            b = sum(1 for _, x, y in ps if x["passed"] and not y["passed"])
            c = sum(1 for _, x, y in ps if y["passed"] and not x["passed"])
            dq = boot_ci([(k[0], x["fixed_frac"] - y["fixed_frac"]) for k, x, y in ps])
            du = boot_ci([(k[0], x["usd"] - y["usd"]) for k, x, y in ps])
            dw = boot_ci([(k[0], x["wall_s"] - y["wall_s"]) for k, x, y in ps])
            dfp = sum(x.get("first_prompt", 0) - y.get("first_prompt", 0) for _, x, y in ps) / len(ps)
            print(f"| {lab} | {m} | {f} | {len(ps)} | {b} / {c} ({mcnemar(b, c):.2f}) | {dq[0]:+.1%} ({dq[1]:+.1%}, {dq[2]:+.1%}) "
                  f"| {verdict_pass(b, c, dq)} | {du[0]:+.2f} ({du[1]:+.2f}, {du[2]:+.2f}) | {verdict_cost(du)} "
                  f"| {dw[0]:+.0f} ({dw[1]:+.0f}, {dw[2]:+.0f}) | {verdict_cost(dw)} | {dfp:+.0f} "
                  f"| {sum(1 for _, x, y in ps if x.get('cut_off', bool(x.get('killed'))) or y.get('cut_off', bool(y.get('killed'))))} |")
    bg_report(labels[0][0], labels[1][0], data, family, a.code_baseline)


# Fixed before any background-task run (2026-10-13) and not to be edited after one: what "ship"
# means for background jobs (P0 + P2) against the build before them.
SHIP_RULE = (
    "1. the background tasks' pass rate does not drop (candidate ≥ base);",
    "2. the median wall time of a background-task run falls (candidate < base);",
    "3. the candidate leaves no process running at exit in any run (`processes_left` = 0; a run the harness cuts "
    "off at its wall or cost cap is killed together with its sandbox, so it has no report and is not counted);",
    "4. the four code tasks (marshmallow, sqlparse, tabulate, toolz) lose no passes against the baseline run "
    "(`--code-baseline`): Σ over tasks of the candidate's pass rate × the baseline's run count ≥ the baseline's passes.",
)
CODE_GATE = ("marshmallow", "sqlparse", "tabulate", "toolz")

BG_COLS = (("bash calls", "bash_calls"), ("run_in_background", "run_in_background"), ("shell `&`", "shell_background"),
           ("timeout_ms set (≤10 s)", None), ("timeout returns", "timeout_returns"), ("timeout kills", "timeout_kills"),
           ("idle yields", "idle_yields"), ("job wait s", "job_wait_s"), ("notices", "notices"), ("reminders", "reminders"), ("sleep-polls", "sleep_polls"),
           ("launches", "launches"), ("duplicate launches", "duplicate_launches"), ("stopped at end", "jobs_stopped_at_end"))


def bg_report(base_label: str, cand_label: str, data: dict, family, code_baseline: str) -> None:
    """The background-task section: what each arm did with long-running processes, per task, and the ship rule."""
    bg = {lab: [r for (t, _, _), r in rows.items() if family(t) == "bg"] for lab, rows in data.items()}
    if not any(bg.values()):
        return
    print("\n## Background tasks\n")
    print("Means per run (`bg_metrics.py` defines each); `left` counts processes still running when the agent exited "
          "(total, and runs with any).\n")
    print("| arm | n | pass | median wall s | turns | $/run | " + " | ".join(c for c, _ in BG_COLS) + " | left |")
    print("|---|---|---|---|---|---|" + "---|" * len(BG_COLS) + "---|")
    for lab, rs in bg.items():
        if not rs:
            continue
        n = len(rs)
        mean = lambda k: sum((r.get("bg") or {}).get(k, 0) for r in rs) / n
        cols = [f"{mean('timeout_ms_set'):.1f} ({mean('timeout_ms_short'):.1f})" if k is None else f"{mean(k):.1f}"
                for _, k in BG_COLS]
        left = [r.get("processes_left") for r in rs]
        print(f"| {lab} | {n} | {sum(r['passed'] for r in rs)}/{n} | {med([r['wall_s'] for r in rs]):.0f} "
              f"| {sum(r['turns'] for r in rs) / n:.0f} | ${sum(r['usd'] for r in rs) / n:.3f} | " + " | ".join(cols)
              + f" | {sum(x or 0 for x in left)} ({sum(1 for x in left if x)} runs) |")
    print("\n| task | " + " | ".join(f"{lab}: pass · median wall s · left" for lab, rs in bg.items() if rs) + " |")
    print("|---|" + "---|" * sum(1 for rs in bg.values() if rs))
    for t in bg_tasks.BG_TASKS:
        cells = []
        for lab, rs in bg.items():
            if not rs:
                continue
            tr = [r for r in rs if r["task"] == t]
            cells.append(f"{sum(r['passed'] for r in tr)}/{len(tr)} · {med([r['wall_s'] for r in tr]):.0f} · "
                         f"{sum(r.get('processes_left') or 0 for r in tr)}" if tr else "–")
        print(f"| {t} | " + " | ".join(cells) + " |")
    print("\n### Ship rule (pre-registered)\n")
    print("Ship background jobs if all of:\n")
    for line in SHIP_RULE:
        print(f"{line}")
    b, c = bg.get(base_label, []), bg.get(cand_label, [])
    checks = []
    if b and c:
        pb, pc = sum(r["passed"] for r in b) / len(b), sum(r["passed"] for r in c) / len(c)
        checks.append((pc >= pb, f"pass rate {pc:.0%} ({len(c)} runs) vs base {pb:.0%} ({len(b)} runs)"))
        wb, wc = med([r["wall_s"] for r in b]), med([r["wall_s"] for r in c])
        checks.append((wc < wb, f"median wall {wc:.0f} s vs base {wb:.0f} s"))
    else:
        checks += [(None, "no background runs for both arms"), (None, "no background runs for both arms")]
    cand_all = list(data.get(cand_label, {}).values())
    left = [r.get("processes_left") for r in cand_all if r.get("sandboxed")]
    reported = [x for x in left if x is not None]
    checks.append((not any(reported) if reported else None,
                   f"{sum(1 for x in reported if x)} of {len(reported)} candidate runs left processes"
                   + (f"; {left.count(None)} have no report (cut off by the harness, or run before reports existed)"
                      if None in left else "")))
    checks.append(code_gate(cand_all, family, code_baseline))
    print()
    for i, (ok, why) in enumerate(checks, 1):
        print(f"- rule {i}: {'PASS' if ok else 'FAIL' if ok is False else 'NO DATA'} — {why}")
    verdict = "SHIP" if all(ok for ok, _ in checks) else "DON'T SHIP" if any(ok is False for ok, _ in checks) else "INCOMPLETE"
    print(f"\n**Verdict: {verdict}**")


def code_gate(cand: list[dict], family, root: str) -> tuple[bool | None, str]:
    """Rule 4: the candidate's code-task passes, scaled to the baseline's run counts, against the baseline's."""
    if not root:
        return None, "no --code-baseline given"
    base = [json.loads(p.read_text()) for p in Path(root).glob("*/*/*/*/result.json")]
    base = [r for r in base if r["task"] in CODE_GATE]
    mine = [r for r in cand if r["task"] in CODE_GATE]
    if not base or not mine:
        return None, f"{len(base)} baseline and {len(mine)} candidate runs of {', '.join(CODE_GATE)}"
    want = got = 0.0
    parts = []
    for t in CODE_GATE:
        bt, ct = [r for r in base if r["task"] == t], [r for r in mine if r["task"] == t]
        if not bt or not ct:
            return None, f"{t}: {len(bt)} baseline, {len(ct)} candidate runs"
        kb, kc = sum(r["passed"] for r in bt), sum(r["passed"] for r in ct)
        want += kb
        got += kc / len(ct) * len(bt)
        parts.append(f"{t} {kc}/{len(ct)} vs {kb}/{len(bt)}")
    return got >= want, f"{got:.1f} scaled passes vs baseline {want:.0f} ({'; '.join(parts)})"


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="ab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--base", required=True)
        p.add_argument("--cand", required=True)
        p.add_argument("--models", default="claude-opus-5-5,claude-sonnet-5-5,gpt-6.1-sol")
        p.add_argument("--tasks", default="", help="code tasks (default: all)")
        p.add_argument("--mcp-tasks", default="", help="MCP tasks (default: all)")
        p.add_argument("--ctx-tasks", default="", help="prompt-prefix tasks (default: all)")
        p.add_argument("--bg-tasks", default="", help="background-process tasks (default: all; `none` for none)")
        p.add_argument("--code-baseline", default="",
                       help="runs root of an earlier code-task run that ship rule 4 compares the candidate against")
        p.add_argument("--attempts", type=int, default=1)
        p.add_argument("--jobs", type=int, default=6)
        p.add_argument("--cap-usd", type=float, default=12.0)
        p.add_argument("--budget-usd", type=float, default=150.0)
        p.add_argument("--ledger", default="", help="spend ledger shared across invocations (see run.py)")
        p.add_argument("--hard-cap-usd", type=float, default=float("inf"), help="with --ledger: total spend cap")
        p.add_argument("--arm", default="default", help="compaction arm for both sides (see run.ARMS)")
        p.add_argument("--base-arg", action="append", default=[], help="extra agent flag, base side only (write --base-arg=--flag)")
        p.add_argument("--cand-arg", action="append", default=[], help="extra agent flag, candidate side only (write --cand-arg=--flag)")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
