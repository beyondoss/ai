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
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import mcp_tasks
import prefix_tasks
import run
import tasks
from repos import TASKS
from report import boot_ci, mcnemar, wilson

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
    for label, sha, cm, extra, mcp_only in arms(a):
        binary = build(sha, cm)
        names = mcp if mcp_only else code + ctx + mcp
        if not names:
            continue
        argv = ["--bin", str(binary), "--models", a.models, "--arms", a.arm, "--tasks", ",".join(names),
                "--attempts", str(a.attempts), "--jobs", str(a.jobs), "--cap-usd", str(a.cap_usd),
                "--budget-usd", str(a.budget_usd), "--runs-dir", str(tasks.work_dir() / "ab" / label),
                "--label", label]
        for e in extra:
            argv += ["--agent-arg", e]
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
    family = lambda t: "mcp" if t in mcp_tasks.MCP_TASKS else "prefix" if t in prefix_tasks.PREFIX_TASKS else "code"
    print(f"Base {base_label}, candidate {labels[1][0]} (`-cm`: candidate with --code-mode, MCP tasks only).\n")
    print("| arm | model | family | n | pass (95% CI) | $/task | wall s/task | TTFT first / median s | turns | first-request tok | cache-read share | compactions |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
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
                  f"| {avg('turns'):.0f} | {avg('first_prompt'):.0f} | {avg('cache_read_share'):.0%} | {avg('compactions'):.2f} |")
    print("\nPaired against the base (task × attempt). Verdicts: pass = McNemar p<0.05 or the Δ-fixed CI excludes 0; "
          "cost and time = the bootstrap CI of the difference excludes 0.\n")
    print("| arm | model | family | pairs | pass cand-only / base-only (p) | Δ fixed (95% CI) | pass verdict | Δ $ (95% CI) | cost verdict | Δ wall s (95% CI) | time verdict | Δ first-request tok |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
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
                  f"| {dw[0]:+.0f} ({dw[1]:+.0f}, {dw[2]:+.0f}) | {verdict_cost(dw)} | {dfp:+.0f} |")


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
        p.add_argument("--attempts", type=int, default=1)
        p.add_argument("--jobs", type=int, default=6)
        p.add_argument("--cap-usd", type=float, default=12.0)
        p.add_argument("--budget-usd", type=float, default=150.0)
        p.add_argument("--arm", default="default", help="compaction arm for both sides (see run.ARMS)")
        p.add_argument("--base-arg", action="append", default=[], help="extra agent flag, base side only")
        p.add_argument("--cand-arg", action="append", default=[], help="extra agent flag, candidate side only")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
