"""How well compaction's sizes match the prompts providers actually billed, over finished eval runs.

    token_sizing.py <eval root>…            every check below
    token_sizing.py --budgets <eval root>…  also search the keep_recent budget per family

Reads each run's `events.jsonl` (the provider-reported prompt size of every request: input +
cache read + cache write) and, where the session file holds the whole history, `sessions/*.jsonl`.
Mirrors `agent_core::token_size` (the calibrated estimate and anchored sizes) and
`compaction::find_cut`, so these are the numbers the doc comments there cite:

- whole: the calibrated estimate alone against the real prompt, at every request with 20K+ of
  content beyond the first request's (system prompt and tools cancel), and chars/4 for comparison.
- tail: the cut simulated at every request; the real tail is P_k - P_j (j is an assistant boundary,
  so it is exact). The old cut (chars/4, nominal 20K), and the new one at the family budgets, both
  anchored (turns measured by earlier requests exact, the trailing step estimated) and estimate-only.
- budgets: the keep_recent per family whose anchored pooled median real tail matches the old one.
- mask window: masking's kept window (a quarter of each model's trigger) as the chars/4 walk kept it,
  in real tokens: the ratio the masking defaults scale by.
- trigger: how often a crossing fires one request late or early, chars/4 vs calibrated, weighted by
  step size over every clean request pair.
- tokens_after: a compaction's post-compaction estimate against the next request's real prompt, and
  the chars per token of the summaries.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

FAMILY = {
    "claude-haiku-5-5": ("haiku", "claude"),
    "claude-sonnet-5-5": ("sonnet", "claude"),
    "claude-opus-5-5": ("opus", "claude"),
    "gpt-6-luna": ("luna", "gpt"),
    "gpt-6.1-sol": ("sol", "gpt"),
}
# tool chars/token, framing tokens per tool result, prose chars/token, % of reasoning not carried
CONST = {"claude": (2.5, 80, 2.3, 0), "gpt": (3.5, 0, 4.0, 70)}
KEEP = {"claude": 40_500, "gpt": 22_000}
MARKER = len("[Earlier conversation compacted to save context]\n\nCompacted from 88888 tokens\n\n")
MODELS = ["haiku", "sonnet", "opus", "luna", "sol"]


def block_chars(b: dict) -> int:
    t = b["type"]
    if t == "text":
        return len(b["text"])
    if t == "thinking":
        return len(b.get("thinking") or b.get("text") or "")
    if t == "redacted_thinking":
        return len(b.get("data", ""))
    if t == "tool_use":
        return len(b["name"]) + len(json.dumps(b["input"], separators=(",", ":"), ensure_ascii=False))
    if t == "tool_result":
        c = b["content"]
        return len(c) if isinstance(c, str) else len(json.dumps(c))
    if t == "image":
        return 4800
    return 0


def legacy(m: dict) -> int:
    return -(-sum(block_chars(b) for b in m["content"]) // 4)


def estimate(m: dict, family: str) -> int:
    tool, framing, text, dropped = CONST[family]
    u = m.get("usage") or {}
    if m["role"] == "assistant" and u.get("output_tokens", 0) > 0:
        return max(0, u["output_tokens"] - u.get("reasoning_tokens", 0) * dropped // 100)
    tool_chars = text_chars = results = images = 0
    for b in m["content"]:
        if b["type"] == "tool_result":
            results += 1
            images += len(b.get("images") or [])
            tool_chars += block_chars(b)
        elif b["type"] == "image":
            images += 1
        else:
            text_chars += block_chars(b)
    return math.ceil(tool_chars / tool) + math.ceil(text_chars / text) + results * framing + 1200 * images


def load(roots: list[str]) -> list[dict]:
    """Every run: its requests (prompt size, output, reasoning, what happened after) and, when the
    session file holds the whole history, its messages aligned with the requests."""
    runs, seen = [], set()
    for root in roots:
        for ev in sorted(Path(root).glob("**/events.jsonl")):
            if "/venv/" in str(ev):
                continue
            d = ev.parent
            res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
            model, sid, reqs, tools = res.get("model"), None, [], {}
            for line in ev.open():
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                k = o.get("kind")
                if k == "session":
                    sid, model = o.get("id"), model or o.get("model")
                elif k == "stream" and o.get("type") == "usage":
                    p = o.get("input_tokens", 0) + o.get("cache_read_tokens", 0) + o.get("cache_write_tokens", 0)
                    reqs.append(dict(P=p, o=o.get("output_tokens", 0), r=o.get("reasoning_tokens", 0), after=[]))
                elif k == "tool_start":
                    tools[o.get("id")] = o.get("name")
                elif reqs and k == "tool_end":
                    res_s = o.get("result") or ""
                    reqs[-1]["after"].append(dict(kind=k, chars=len(res_s if isinstance(res_s, str) else json.dumps(res_s))))
                elif reqs and k not in ("turn_start", "turn_end", "tool_start", "stream"):
                    x = {kk: vv for kk, vv in o.items() if kk != "summary"}
                    if k == "compacted":
                        x["summary_chars"] = len(o.get("summary") or "")
                    reqs[-1]["after"].append(x)
            if not reqs or model not in FAMILY:
                continue
            key = (sid, len(reqs), reqs[0]["P"], reqs[-1]["P"])
            if key in seen:
                continue
            seen.add(key)
            fam, family = FAMILY[model]
            run = dict(d=d, fam=fam, family=family, q=reqs, msgs=None)
            rewritten = any(a["kind"] in ("compacted", "tool_output_masked", "retry", "agent_start") for r in reqs for a in r["after"])
            sessions = list((d / "sessions").glob("*.jsonl"))
            if not rewritten and len(sessions) == 1:
                msgs = [m for m in map(json.loads, sessions[0].open()) if m.get("type") == "message"]
                a = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
                if len(a) == len(reqs):
                    run.update(msgs=msgs, A=a)
            first = next((json.loads(line) for line in sessions[0].open() if '"type":"message"' in line), None) if len(sessions) == 1 else None
            run["first"] = first if first and first.get("role") == "user" else None
            runs.append(run)
    return runs


def anchored(run: dict, k: int) -> list[int]:
    """Sizes as the agent sees them just before request k: turns measured by requests < k exact."""
    msgs, a, q = run["msgs"], run["A"], run["q"]
    sizes = [estimate(m, run["family"]) for m in msgs[: a[k]]]
    for j in range(1, k):
        lo, hi = a[j - 1], a[j]
        real = q[j]["P"] - q[j - 1]["P"]
        if hi - lo == 2 and real >= 0:
            sizes[lo] = min(sizes[lo], real)
            sizes[lo + 1] = real - sizes[lo]
    return sizes


def find_cut(roles: list[str], sizes: list[int], keep: int) -> int | None:
    n = len(sizes)
    if n < 4:
        return None
    suffix, fits, last = 0, None, None
    for i in range(n - 1, 0, -1):
        suffix += sizes[i]
        if roles[i] != "assistant":
            continue
        last = last if last is not None else i
        if suffix > keep:
            break
        fits = i
    return fits if fits is not None else last


def old_find_cut(roles: list[str], est: list[int], keep: int) -> int | None:
    n, kept, idx = len(est), 0, len(est) - 1
    if n < 4:
        return None
    while idx > 0:
        kept += est[idx]
        if kept >= keep:
            break
        idx -= 1
    fw = idx
    while fw < n and roles[fw] != "assistant":
        fw += 1
    if fw >= n:
        fw = idx
        while fw > 0 and roles[fw] != "assistant":
            fw -= 1
    return fw if fw and roles[fw] == "assistant" else None


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def whole(clean: list[dict]) -> None:
    print("== whole: real growth over the first request / estimate, requests with 20K+ of content ==")
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in clean:
        for lab, fn in (("chars/4", legacy), ("calibrated", lambda m: estimate(m, r["family"]))):
            cum = np.cumsum([0] + [fn(m) for m in r["msgs"]])
            for k, ai in enumerate(r["A"]):
                e = cum[ai] - cum[r["A"][0]]
                if e >= 20_000:
                    out[r["fam"]][lab].append((r["q"][k]["P"] - r["q"][0]["P"]) / e)
    for f in MODELS:
        for lab, a in out[f].items():
            print(f"  {f:7s} {lab:10s} n={len(a):6d} real/est median {np.median(a):.3f}  p10 {pct(a,10):.3f}  p90 {pct(a,90):.3f}")


def simulate(clean: list[dict], mode: str, keep: dict | None) -> dict:
    out = collections.defaultdict(list)
    for r in clean:
        a, q, msgs = r["A"], r["q"], r["msgs"]
        pos = {ai: k for k, ai in enumerate(a)}
        roles = [m["role"] for m in msgs]
        leg = [legacy(m) for m in msgs]
        est = [estimate(m, r["family"]) for m in msgs]
        for k, n in enumerate(a):
            if n < 4:
                continue
            if mode == "old":
                if sum(leg[1:n]) < 20_000:
                    continue
                j, tail_est = old_find_cut(roles[:n], leg[:n], 20_000), None
            else:
                sizes = anchored(r, k) if mode == "anchored" else est[:n]
                if sum(sizes[1:]) <= keep[r["family"]]:
                    continue
                j = find_cut(roles[:n], sizes, keep[r["family"]])
                tail_est = sum(sizes[j:]) if j is not None else None
            if j is None or j not in pos:
                continue
            out[r["fam"]].append((q[k]["P"] - q[pos[j]]["P"], tail_est))
    return out


def report_tail(title: str, out: dict) -> None:
    print(title)
    for f in MODELS:
        real = [x[0] for x in out[f]]
        if not real:
            continue
        line = f"  {f:7s} n={len(real):6d} real tail median {np.median(real):7.0f}  p10 {pct(real,10):7.0f}  p90 {pct(real,90):7.0f}"
        rat = [x[0] / x[1] for x in out[f] if x[1]]
        if rat:
            line += f"  real/est median {np.median(rat):.3f}  p10 {pct(rat,10):.3f}  p90 {pct(rat,90):.3f}"
        print(line)
    for g in ("claude", "gpt"):
        real = [x[0] for f in out if FAMILY_OF[f] == g for x in out[f]]
        if real:
            print(f"  pooled {g:6s} median real tail {np.median(real):7.0f}")


FAMILY_OF = {fam: family for fam, family in FAMILY.values()}


TRIGGER = {"haiku": 90_000, "sonnet": 200_000, "opus": 200_000, "luna": 240_000, "sol": 240_000}


def mask_window(clean: list[dict]) -> None:
    print("== mask window: real tokens the chars/4 walk kept at a quarter of the trigger ==")
    out = collections.defaultdict(list)
    for r in clean:
        a, q, msgs = r["A"], r["q"], r["msgs"]
        pos = {ai: k for k, ai in enumerate(a)}
        roles = [m["role"] for m in msgs]
        leg = [legacy(m) for m in msgs]
        budget = TRIGGER[r["fam"]] // 4
        for k, n in enumerate(a):
            held, start = 0, None
            for i in range(n - 1, -1, -1):
                held += leg[i]
                if held >= budget:
                    start = next((x for x in range(i, n) if roles[x] == "assistant"), None)
                    break
            if start is not None and start in pos:
                out[r["fam"]].append(q[k]["P"] - q[pos[start]]["P"])
    for f in MODELS:
        if out[f]:
            print(f"  {f:7s} n={len(out[f]):6d} budget {TRIGGER[f] // 4:6d}  real window median {np.median(out[f]):7.0f}"
                  f"  ratio {np.median(out[f]) / (TRIGGER[f] // 4):.2f}")


def trigger(runs: list[dict]) -> None:
    print("== trigger: chance a crossing fires one request late / early (step-size weighted) ==")
    rows = collections.defaultdict(list)
    for r in runs:
        tool, framing, _, dropped = CONST[r["family"]]
        q = r["q"]
        for k in range(len(q) - 1):
            after = q[k]["after"]
            if any(x["kind"] not in ("tool_end", "assistant_text", "tool_progress") for x in after):
                continue
            te = [x for x in after if x["kind"] == "tool_end"]
            real = q[k + 1]["P"] - q[k]["P"]
            if not te or real <= 0:
                continue
            chars = sum(x["chars"] for x in te)
            old = q[k]["o"] + -(-chars // 4)
            new = max(0, q[k]["o"] - q[k]["r"] * dropped // 100) + math.ceil(chars / tool) + framing * len(te)
            rows[r["fam"]].append((real, old, new))
    for f in MODELS:
        for lab, i in (("chars/4", 1), ("calibrated", 2)):
            span = sum(max(x[0], x[i]) for x in rows[f])
            late = sum(max(0, x[0] - x[i]) for x in rows[f])
            early = sum(max(0, x[i] - x[0]) for x in rows[f])
            if span:
                print(f"  {f:7s} {lab:10s} pairs={len(rows[f]):6d} late {late/span:6.1%}  early {early/span:6.1%}")


def tokens_after(runs: list[dict]) -> None:
    print("== tokens_after: next request's real prompt / estimate; summary chars per token ==")
    out = collections.defaultdict(list)
    for r in runs:
        if not r["first"]:
            continue
        family, q = r["family"], r["q"]
        tool, framing, text, dropped = CONST[family]
        c = q[0]["P"] - estimate(r["first"], family)
        for k in range(len(q) - 1):
            comp = [a for a in q[k]["after"] if a["kind"] == "compacted"]
            if len(comp) != 1 or any(x["kind"] in ("tool_output_masked", "steered") for x in q[k]["after"]):
                continue
            a = comp[0]
            mb, fk = a.get("messages_before"), a.get("first_kept")
            if mb is None or fk is None or (mb - fk) % 2:
                continue
            m = (mb - fk) // 2
            if m < 1 or k - m + 1 < 0:
                continue
            if any(x["kind"] in ("compacted", "tool_output_masked") for j in range(k - m + 1, k) for x in q[j]["after"]):
                continue
            te = [x for x in q[k]["after"] if x["kind"] == "tool_end"]
            trail = max(0, q[k]["o"] - q[k]["r"] * dropped // 100) + math.ceil(sum(x["chars"] for x in te) / tool) + framing * len(te)
            kept = q[k]["P"] - q[k - m + 1]["P"]
            chars = a.get("summary_chars", 0) + MARKER
            est_after = c + kept + trail + math.ceil(chars / text)
            real = q[k + 1]["P"]
            out[r["fam"]].append((real / est_after, real / max(1, a.get("tokens_after") or 1), chars, real - c - kept - trail))
    for f in MODELS:
        x = out[f]
        if not x:
            continue
        new = [v[0] for v in x]
        old = [v[1] for v in x]
        rate = [v[2] / v[3] for v in x if v[3] > 300]
        print(f"  {f:7s} n={len(x):4d} real/est median {np.median(new):.3f}  p10 {pct(new,10):.3f}  p90 {pct(new,90):.3f}"
              f"  (shipped chars/4 estimate: {np.median(old):.2f})  summary chars/token median {np.median(rate):.2f}")


def main(argv: list[str]) -> None:
    budgets = "--budgets" in argv
    roots = [a for a in argv if a != "--budgets"]
    if not roots:
        sys.exit(__doc__)
    runs = load(roots)
    clean = [r for r in runs if r["msgs"] is not None]
    print(f"{len(runs)} runs, {len(clean)} with the whole history in the session file\n")
    whole(clean)
    old = simulate(clean, "old", None)
    report_tail("\n== tail: the shipped cut (chars/4, nominal 20K) ==", old)
    report_tail(f"\n== tail: new cut, anchored, budgets {KEEP} ==", simulate(clean, "anchored", KEEP))
    report_tail(f"\n== tail: new cut, estimate only, budgets {KEEP} ==", simulate(clean, "estimate", KEEP))
    if budgets:
        print("\n== budgets: anchored pooled median real tail per budget (target: the old cut's) ==")
        for g, lo, hi in (("claude", 36_000, 45_000), ("gpt", 19_000, 25_000)):
            target = np.median([x[0] for f in old if FAMILY_OF[f] == g for x in old[f]])
            for b in range(lo, hi + 1, 500):
                out = simulate(clean, "anchored", {"claude": b, "gpt": b})
                med = np.median([x[0] for f in out if FAMILY_OF[f] == g for x in out[f]])
                print(f"  {g:6s} B={b:6d} median {med:7.0f}  target {target:7.0f}")
    print()
    mask_window(clean)
    print()
    trigger(runs)
    print()
    tokens_after(runs)


if __name__ == "__main__":
    main(sys.argv[1:])
