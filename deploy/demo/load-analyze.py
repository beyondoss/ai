#!/usr/bin/env python3
"""Turn the load proof's two records into one run.json.

    load-analyze.py --driver driver.json --samples samples.jsonl --meta meta.json > run.json

driver.json is what `beyond-ai-demo load` wrote (what a client saw, and the bundle host's counters);
samples.jsonl is what the sampler in the replica's task logged (what the kernel and the replica say
about the process). This joins them on time, measures the per-session cost, compares each idle period
with the one before it, and grades the bounds the proof is about. Standard library only.
"""
import argparse
import json
import statistics
import sys

KB = 1024.0
CACHE_DIR_OVERHEAD_KB = 16  # the cache's own directory and its lock file, which du counts and the cache does not


def load_samples(path):
    seen, out = set(), []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except ValueError:
            continue
        if s["t"] in seen:
            continue
        seen.add(s["t"])
        out.append(s)
    out.sort(key=lambda s: s["t"])
    return out


def within(samples, lo, hi):
    return [s for s in samples if lo <= s["t"] <= hi]


def med(xs):
    return statistics.median(xs) if xs else 0.0


def slope(xs, ys):
    """Least-squares slope of y on x; 0 with fewer than two distinct x."""
    n = len(xs)
    if n < 2 or max(xs) == min(xs):
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def settled(samples, phase, quiet_secs=90, margin_secs=8):
    """The replica at the end of an idle phase, over its last `quiet_secs`. Memory is the *floor* (the
    lowest value) because the allocator hands pages back only when it next runs, so the median of a quiet
    stretch says when it last ran; `drift` is how far the end of the stretch sits above that floor."""
    win = within(samples, phase["end_ms"] - (quiet_secs + margin_secs) * 1000, phase["end_ms"] - margin_secs * 1000)
    out = {k: med([s.get(k, 0) for s in win]) for k in ["Threads", "fds", "cache_kb", "cache_dirs", "agent_sessions_live", "AnonHugePages"]}
    for k in ["RssAnon", "VmRSS"]:
        out[k] = min([s.get(k, 0) for s in win] or [0])
    out["RssAnon_end"] = win[-1].get("RssAnon", 0) if win else 0
    out["drift"] = out["RssAnon_end"] - out["RssAnon"]
    return out


def with_cpu(samples):
    """Each sample's CPU use since the one before, as a share of one vCPU (the task has half of one)."""
    prev = None
    for s in samples:
        if prev and s["t"] > prev["t"] and "cpu_ticks" in s and "cpu_ticks" in prev:
            s["cpu_pct"] = max(0.0, (s["cpu_ticks"] - prev["cpu_ticks"]) / 100.0 / ((s["t"] - prev["t"]) / 1000.0) * 100.0)
        else:
            s["cpu_pct"] = 0.0
        prev = s
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    driver = json.load(open(a.driver))
    meta = json.load(open(a.meta))
    samples = with_cpu(load_samples(a.samples))
    phases = {p["name"]: p for p in driver["phases"]}
    budget_kb = meta["cache_budget_mb"] * 1024
    first_t = driver["phases"][0]["start_ms"]

    # Before anything ran.
    before = settled(samples, {"end_ms": first_t}, 30)

    waves = []
    n_waves = driver["config"].get("waves", sum(1 for n in phases if n.startswith("wave")))
    for w in range(1, n_waves + 1):
        p, idle = phases[f"wave{w}"], phases[f"idle{w}"]
        win = within(samples, p["start_ms"], p["end_ms"])
        peak = max(win, key=lambda s: s.get("RssAnon", 0)) if win else {}
        xs = [s.get("agent_sessions_live", 0) for s in win]
        ys = [s.get("RssAnon", 0) for s in win]
        base = waves[-1]["after_idle"] if waves else before
        live_peak = max(xs) if xs else 0
        waves.append({
            "wave": w,
            "open_sessions_client_peak": p["peak_open_sessions"],
            "sessions_live_peak": live_peak,
            "rss_anon_peak_mb": peak.get("RssAnon", 0) / KB,
            "rss_peak_mb": max([s.get("VmRSS", 0) for s in win] or [0]) / KB,
            "threads_peak": max([s.get("Threads", 0) for s in win] or [0]),
            "fds_peak": max([s.get("fds", 0) for s in win] or [0]),
            "per_session_kb_slope": slope(xs, ys),
            "per_session_kb_from_baseline": (peak.get("RssAnon", 0) - base["RssAnon"]) / live_peak if live_peak else 0,
            "cpu_mean_pct": med([s.get("cpu_pct", 0) for s in win]) if win else 0,
            "cpu_peak_pct": max([s.get("cpu_pct", 0) for s in win] or [0]),
            "open_p50_ms": p["open_p50_ms"], "open_p99_ms": p["open_p99_ms"],
            "open_p50_ms_net_of_rtt": max(p["open_p50_ms"] - driver["rtt_ms"], 0),
            "open_p99_ms_net_of_rtt": max(p["open_p99_ms"] - driver["rtt_ms"], 0),
            "opens": p["opens"], "open_failures": p["open_failures"], "turns": p["turns"],
            "turns_that_read_the_skill": p["turns_that_read_the_skill"],
            "after_idle": settled(samples, idle),
        })

    burst = phases.get("burst")
    burst_win = within(samples, burst["start_ms"], burst["end_ms"]) if burst else []
    mcp = {}
    for label in ("mcp-unset", "mcp-limit5"):
        p = phases.get(label)
        if not p:
            continue
        win = within(samples, p["start_ms"], p["end_ms"])
        h = driver["host"].get(label, {})
        mcp[label] = {
            "sessions": p["opens"],
            "dials_at_the_host": h.get("mcp_connections"),
            "distinct_servers": h.get("mcp_distinct_servers"),
            "peak_open_at_the_host": h.get("mcp_peak_open"),
            "peak_established_from_the_replica": max([s.get("to_loadhost", 0) for s in win] or [0]),
            "open_p50_ms": p["open_p50_ms"],
        }

    cache_peak_kb = max([s.get("cache_kb", 0) for s in samples] or [0])
    idle_cache = [settled(samples, phases[n])["cache_kb"] for n in phases if n.startswith("idle")]
    dirs = [s.get("cache_dirs", 0) for s in samples]
    dir_drops = sum(max(dirs[i - 1] - dirs[i], 0) for i in range(1, len(dirs)))
    refetches = sum(h.get("refetches", 0) for h in driver["host"].values())
    fetch_peak = max([h.get("fetch_peak", 0) for h in driver["host"].values()] or [0])
    burst_fetch_peak = driver["host"].get("burst", {}).get("fetch_peak", 0)

    # --- the grades --------------------------------------------------------------------------
    checks = []

    def check(name, title, ok, detail, headline=None):
        checks.append({"name": name, "title": title, "status": "pass" if ok else "fail", "pass": bool(ok),
                       "detail": detail, "headline": headline})

    sess = driver["config"]["sessions"]
    check("sessions-reached", f"{sess} sessions open at once, in every wave",
          all(w["sessions_live_peak"] >= sess * 0.95 and w["open_failures"] == 0 for w in waves),
          "; ".join(f"wave {w['wave']}: {w['sessions_live_peak']} live at the replica, {w['open_failures']} failed opens" for w in waves),
          {"label": "peak live", "value": max([w["sessions_live_peak"] for w in waves] or [0]), "unit": "sessions"})
    allturns = all(w["turns"] > 0 and w["turns_that_read_the_skill"] == w["turns"] for w in waves)
    check("skill-read", "every turn read its bundle's skill", allturns,
          "; ".join(f"wave {w['wave']}: {w['turns_that_read_the_skill']} of {w['turns']}" for w in waves))
    if len(waves) >= 2:
        w1, w2 = waves[0], waves[-1]
        tol = max(0.05 * w1["after_idle"]["RssAnon"], 4 * KB)
        grew = w2["after_idle"]["RssAnon"] - w1["after_idle"]["RssAnon"]
        check("memory-returns", "the memory floor after the second wave is the first wave's",
              grew <= tol,
              f"lowest anonymous memory in the last 90 s of idle 1: {w1['after_idle']['RssAnon'] / KB:.1f} MB; of idle {len(waves)}: "
              f"{w2['after_idle']['RssAnon'] / KB:.1f} MB ({grew / KB:+.1f} MB, tolerance {tol / KB:.1f} MB); "
              f"before the first wave {before['RssAnon'] / KB:.1f} MB",
              {"label": "growth across waves", "value": round(grew / KB, 1), "unit": "MB"})
        pk = w2["rss_anon_peak_mb"] - w1["rss_anon_peak_mb"]
        check("peak-flat", "the second wave's peak is the first wave's", pk <= max(0.05 * w1["rss_anon_peak_mb"], 4),
              f"peak anonymous memory {w1['rss_anon_peak_mb']:.1f} MB, then {w2['rss_anon_peak_mb']:.1f} MB ({pk:+.1f} MB)")
        dfd = w2["after_idle"]["fds"] - w1["after_idle"]["fds"]
        dth = w2["after_idle"]["Threads"] - w1["after_idle"]["Threads"]
        check("fds-threads-return", "fds and threads after the second idle are the first idle's",
              dfd <= 10 and dth <= 2,
              f"fds {w1['after_idle']['fds']:.0f} then {w2['after_idle']['fds']:.0f}; threads "
              f"{w1['after_idle']['Threads']:.0f} then {w2['after_idle']['Threads']:.0f}")
    drifts = [w["after_idle"]["drift"] for w in waves]
    check("quiet-replica-is-quiet", "with no sessions and no work, anonymous memory stays put",
          all(d <= max(0.05 * w["after_idle"]["RssAnon"], 4 * KB) for d, w in zip(drifts, waves)),
          "; ".join(f"idle {w['wave']}: ends {d / KB:+.1f} MB above its floor of {w['after_idle']['RssAnon'] / KB:.1f} MB "
                    f"({w['after_idle']['AnonHugePages'] / KB:.0f} MB of it in transparent huge pages)" for d, w in zip(drifts, waves)),
          {"label": "drift while idle", "value": round(max(drifts or [0]) / KB, 1), "unit": "MB"})
    check("fetch-cap", "at most 4 bundle fetches at once (and the cap was reached)",
          fetch_peak <= 4 and burst_fetch_peak >= 4,
          f"peak concurrent fetches at the bundle host: {fetch_peak} overall, {burst_fetch_peak} in the cold burst "
          f"({driver['config']['bundles_in_burst']} never-seen bundles, {driver['config']['burst_sessions']} sessions at once)",
          {"label": "peak fetches", "value": fetch_peak, "unit": "of 4"})
    check("cache-bounded", "the bundle cache stays within its budget once sessions release their bundles",
          max(idle_cache or [0]) <= budget_kb + CACHE_DIR_OVERHEAD_KB,
          f"budget {budget_kb / KB:.0f} MB (the cache counts the blocks of its bundles; du also counts its own directory and lock file); on disk at the end of each idle: "
          + ", ".join(f"{c / KB:.1f} MB" for c in idle_cache) + f"; peak with sessions open {cache_peak_kb / KB:.1f} MB "
          "(bundles a session holds are never evicted, so the budget can be exceeded by exactly those)",
          {"label": "peak on disk", "value": round(cache_peak_kb / KB, 1), "unit": f"MB of {budget_kb / KB:.0f}"})
    check("evictions", "eviction happened", refetches > 0 or dir_drops > 0 or any(s.get("cache_trash", 0) for s in samples),
          f"{refetches} bundles fetched again after an eviction (bundle host), {dir_drops} cache directories removed (sampler)")
    u, l5 = mcp.get("mcp-unset"), mcp.get("mcp-limit5")
    if u and l5:
        per_u = u["distinct_servers"]
        check("connector-cap-unset", "a 200-server .mcp.json with no grant limit is capped by the agent",
              0 < per_u < 200,
              f"{u['sessions']} sessions on a 200-server file dialed {per_u} distinct servers "
              f"({u['dials_at_the_host']} connections in all, {u['peak_open_at_the_host']} open at once)",
              {"label": "servers dialed", "value": per_u, "unit": "of 200"})
        check("connector-cap-limit", "the same file with the grant limit at 5 dials 5",
              l5["distinct_servers"] == 5,
              f"{l5['sessions']} sessions dialed {l5['distinct_servers']} distinct servers "
              f"({l5['dials_at_the_host']} connections, {l5['peak_open_at_the_host']} open at once)",
              {"label": "servers dialed", "value": l5["distinct_servers"], "unit": "of 200"})
    check("teardown-zero", "every resource type counted 0 after the run", False, "pending: filled in by down.sh")
    checks[-1]["status"] = "pending"

    # Headline numbers.
    per = [w["per_session_kb_slope"] for w in waves if w["per_session_kb_slope"] > 0]
    out = {
        "kind": "load",
        "commit": meta["commit"], "region": meta["region"], "name": meta["name"], "started_at": meta["started_at"],
        "task": meta["task"], "cache_budget_mb": meta["cache_budget_mb"],
        "config": driver["config"], "rtt_ms": driver["rtt_ms"],
        "phases": driver["phases"], "host": driver["host"],
        "samples": samples, "client_open_sessions": driver["client_open_sessions"],
        "analysis": {
            "before_first_wave": before, "waves": waves,
            "per_session_kb": med(per), "mcp": mcp,
            "cache": {"budget_kb": budget_kb, "peak_kb": cache_peak_kb, "idle_kb": idle_cache,
                      "refetches": refetches, "directories_removed": dir_drops},
            "fetch_peak": fetch_peak, "burst_fetch_peak": burst_fetch_peak,
            "burst": {"open_p50_ms": burst["open_p50_ms"], "open_p99_ms": burst["open_p99_ms"],
                      "peak_rss_anon_mb": max([s.get("RssAnon", 0) for s in burst_win] or [0]) / KB} if burst else None,
        },
        "checks": checks, "teardown": None,
    }
    json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
