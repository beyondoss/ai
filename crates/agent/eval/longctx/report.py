"""Tabulate the sweep: per model × arm, and paired against each model's `default` arm.

Pass rates carry a Wilson 95% interval. The paired comparison matches runs on (task, attempt):
an exact McNemar test on pass/fail and a task-bootstrap 95% interval on the mean difference in
fraction-of-bugs-fixed and in dollars.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import tasks


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def boot_ci(pairs: list[tuple[str, float]], reps: int = 4000, seed: int = 7) -> tuple[float, float, float]:
    """Mean of per-pair differences and a 95% interval resampling whole tasks."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for t, d in pairs:
        by_task[t].append(d)
    keys = list(by_task)
    if not keys:
        return (math.nan, math.nan, math.nan)
    mean = sum(d for _, d in pairs) / len(pairs)
    rng = random.Random(seed)
    ms = []
    for _ in range(reps):
        xs = [d for k in (rng.choice(keys) for _ in keys) for d in by_task[k]]
        ms.append(sum(xs) / len(xs))
    ms.sort()
    return (mean, ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1])


def load(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.glob("runs/*/*/*/*/result.json"))]


ENDS = ("completed", "max_steps", "wall", "cap", "error", "crash")


def endings(rows: list[dict]) -> str:
    """How runs ended, for one cell: `completed 5 · max_steps 2 …`, and how many cut-off runs left a
    tree that doesn't parse (mid-edit) or a suite that hung."""
    c = defaultdict(int)
    for r in rows:
        c[r.get("end_reason") or ("completed" if r.get("ended") else r.get("killed") or "?")] += 1
    s = " · ".join(f"{k} {c[k]}" for k in (*ENDS, "?") if c[k])
    mid = sum(1 for r in rows if r.get("tree_state") == "mid_edit")
    hang = sum(1 for r in rows if r.get("tree_state") == "suite_hang")
    return s + (f" (mid-edit {mid})" if mid else "") + (f" (suite hang {hang})" if hang else "")


def errs(rows: list[dict]) -> str:
    """Mean tool errors per run: `hard / soft` (soft: a Python error behind a zero exit)."""
    n = len(rows) or 1
    return f"{sum(r.get('tool_errors', 0) for r in rows) / n:.1f} / {sum(r.get('soft_errors', 0) for r in rows) / n:.1f}"


def main(argv: list[str]) -> None:
    rows = load(tasks.work_dir())
    if not rows:
        print("no results")
        return
    cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cell[(r["model"], r["arm"])].append(r)
    arm_order = {"100k": 0, "200k": 1, "model": 2, "default": 3}
    print("| model | trigger | n | pass (95% CI) | bugs fixed | $/run | turns | compactions | peak ctx | fails w/ compaction | killed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, a) in sorted(cell, key=lambda k: (k[0], arm_order.get(k[1], 9))):
        rs = cell[(m, a)]
        n = len(rs)
        k = sum(r["passed"] for r in rs)
        lo, hi = wilson(k, n)
        mean = lambda f: sum(f(r) for r in rs) / n
        fails_c = sum(1 for r in rs if not r["passed"] and r["compactions"])
        fails = n - k
        trig = rs[0].get("trigger")
        trig = f"{trig // 1000}K" if trig and trig < 900_000 else "model window − 16K"
        print(f"| {m} | {a} ({trig}) | {n} | {k}/{n} = {k / n:.0%} ({lo:.0%}–{hi:.0%}) "
              f"| {mean(lambda r: r['fixed_frac']):.1%} | ${mean(lambda r: r['usd']):.2f} | {mean(lambda r: r['turns']):.0f} "
              f"| {mean(lambda r: r['compactions']):.2f} | {mean(lambda r: r['peak_prompt']) / 1000:.0f}K "
              f"(max {max(r['peak_prompt'] for r in rs) // 1000}K) | {fails_c}/{fails} | {sum(bool(r['killed']) for r in rs)} |")
    print()
    print("Paired vs each model's default arm (matched on task × attempt):")
    print()
    print("| model | arm | pairs | pass: arm-only / default-only (McNemar p) | Δ bugs fixed (95% CI) | Δ $/run (95% CI) | Δ $ % |")
    print("|---|---|---|---|---|---|---|")
    for m in sorted({m for m, _ in cell}):
        base = {(r["task"], r["attempt"]): r for r in cell.get((m, "default"), [])}
        for a in sorted({x for mm, x in cell if mm == m and x != "default"}, key=lambda x: arm_order.get(x, 9)):
            arm = {(r["task"], r["attempt"]): r for r in cell.get((m, a), [])}
            keys = sorted(set(base) & set(arm))
            if not keys:
                continue
            b = sum(1 for k in keys if arm[k]["passed"] and not base[k]["passed"])
            c = sum(1 for k in keys if base[k]["passed"] and not arm[k]["passed"])
            dq = boot_ci([(k[0], arm[k]["fixed_frac"] - base[k]["fixed_frac"]) for k in keys])
            du = boot_ci([(k[0], arm[k]["usd"] - base[k]["usd"]) for k in keys])
            base_usd = sum(base[k]["usd"] for k in keys) / len(keys)
            print(f"| {m} | {a} | {len(keys)} | {b} / {c} (p={mcnemar(b, c):.2f}) | {dq[0]:+.1%} ({dq[1]:+.1%}, {dq[2]:+.1%}) "
                  f"| {du[0]:+.2f} ({du[1]:+.2f}, {du[2]:+.2f}) | {du[0] / base_usd:+.0%} |")
    print()
    print("How runs ended. A cut-off run (max_steps, wall, cap) is graded on the tree it left; `mid-edit` is one")
    print("whose library no longer parses. `last parseable` grades those on the last state in which it did.")
    print()
    print("| model | trigger | n | end reasons | pass excl. cut-off | mid-edit: bugs fixed final → last parseable | tool errors hard / soft per run |")
    print("|---|---|---|---|---|---|---|")
    for (m, a) in sorted(cell, key=lambda k: (k[0], arm_order.get(k[1], 9))):
        rs = cell[(m, a)]
        kept = [r for r in rs if not r.get("cut_off", bool(r.get("killed")))]
        k = sum(r["passed"] for r in kept)
        mids = [r for r in rs if r.get("last_parseable")]
        mid = (f"{sum(r['fixed_frac'] for r in mids) / len(mids):.1%} → "
               f"{sum(r['last_parseable']['fixed_frac'] for r in mids) / len(mids):.1%} ({len(mids)})") if mids else "–"
        print(f"| {m} | {a} | {len(rs)} | {endings(rs)} | {k}/{len(kept)} | {mid} | {errs(rs)} |")
    print()
    tot = sum(r["usd"] for r in rows)
    print(f"{len(rows)} runs, total ${tot:.2f}; suspect commands in "
          f"{sum(1 for r in rows if r['suspect'])} runs")


if __name__ == "__main__":
    main(sys.argv[1:])
