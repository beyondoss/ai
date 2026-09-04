#!/usr/bin/env python3
"""Print pass/fail + tokens + one list-price `$` for a Harbor jobs dir.

Ignores each adapter's own `cost_usd` (Pi = billed, beyond = list) and reprices
both from the same card (`token_price.py`). Harbor `n_input_tokens` includes cache.

    python3 crates/agent/eval/tabulate_harbor_job.py /tmp/beyond-ab-glm53 /tmp/pi-ab-glm53
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_price import list_usd, token_rates


def _trial_results(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("result.json")):
        data = json.loads(path.read_text())
        if "task_name" not in data or "agent_result" not in data:
            continue
        agent = data["agent_result"] or {}
        cfg = (data.get("config") or {}).get("agent") or {}
        model = cfg.get("model_name") or (data.get("agent_info") or {}).get("model_info", {}).get(
            "name"
        )
        inp = int(agent.get("n_input_tokens") or 0)
        cache = int(agent.get("n_cache_tokens") or 0)
        out = int(agent.get("n_output_tokens") or 0)
        fresh, usd = list_usd(inp, cache, out, model)
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rows.append(
            {
                "task": data["task_name"].rsplit("/", 1)[-1],
                "pass": reward == 1.0 or reward == 1,
                "fresh": fresh,
                "cache": cache,
                "out": out,
                "usd": usd,
                "model": model,
                "billed": agent.get("cost_usd"),
            }
        )
    return rows


def _print(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{label}: no trials")
        return
    model = rows[0]["model"]
    rates = token_rates(model)
    print(f"## {label}  ({model}; list ${rates[0]:.2f} / ${rates[1]:.2f} cache / ${rates[2]:.2f})")
    print("| Task | Pass | Fresh in | Cache | Out | List $ |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    pf = pc = po = pu = 0
    n_pass = 0
    for r in rows:
        mark = "pass" if r["pass"] else "fail"
        if r["pass"]:
            n_pass += 1
        pf += r["fresh"]
        pc += r["cache"]
        po += r["out"]
        pu += r["usd"]
        print(
            f"| {r['task']} | {mark} | {r['fresh']:,} | {r['cache']:,} | {r['out']:,} | ${r['usd']:.4f} |"
        )
    print(
        f"| **total** | **{n_pass}/{len(rows)}** | **{pf:,}** | **{pc:,}** | **{po:,}** | **${pu:.4f}** |"
    )
    print()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    for raw in argv[1:]:
        root = Path(raw)
        if not root.exists():
            print(f"missing: {root}", file=sys.stderr)
            return 1
        _print(str(root), _trial_results(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
