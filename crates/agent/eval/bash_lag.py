#!/usr/bin/env python3
"""Summarize `bash_lag` events: how long `bash` kept the model waiting after a failure was visible.

Reads every `beyond.stderr` under the given Harbor jobs dirs (or the given files). The Harbor
adapter enables the events by default; elsewhere run the agent with `RUST_LOG=error,bash_lag=info`.

    python3 crates/agent/eval/bash_lag.py /tmp/beyond-frontier-beyond

"Lead" is `elapsed_ms - first_failure_ms` on a failed call: wall time after the first
failure-looking line during which the model could not act on it.
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")
_FIELD = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
_FAILED = {"exit", "signal", "timeout"}


def _events(path: Path) -> list[dict[str, str]]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = _ANSI.sub("", line)
        if "bash call finished" not in line:
            continue
        out.append({k: v.strip('"') for k, v in _FIELD.findall(line)})
    return out


def _files(args: list[str]) -> list[Path]:
    files: list[Path] = []
    for arg in args:
        p = Path(arg)
        files.extend(sorted(p.rglob("beyond.stderr")) if p.is_dir() else [p])
    return files


def _pct(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    events = [e for f in _files(argv[1:]) for e in _events(f)]
    streamed = [e for e in events if e.get("streamed") == "true"]
    if not streamed:
        print(f"{len(events)} bash_lag events, none streamed; nothing to measure")
        return 0

    failed = [e for e in streamed if e.get("outcome") in _FAILED]
    hits = [e for e in failed if "first_failure_ms" in e]
    false_pos = [e for e in streamed if e.get("outcome") == "success" and "first_failure_ms" in e]
    leads = [int(e["elapsed_ms"]) - int(e["first_failure_ms"]) for e in hits]
    total_ms = sum(int(e["elapsed_ms"]) for e in streamed)

    print(f"bash calls: {len(events)} ({len(streamed)} streamed, {len(failed)} failed)")
    print(f"failed calls with a failure line: {len(hits)}/{len(failed)}")
    print(f"successful calls with a failure line (false positives): {len(false_pos)}")
    print(f"total bash wall time: {total_ms / 1000:.1f}s")
    if leads:
        print(
            f"lead after first failure: total {sum(leads) / 1000:.1f}s "
            f"({100 * sum(leads) / max(total_ms, 1):.1f}% of bash time), "
            f"median {statistics.median(leads) / 1000:.1f}s, p90 {_pct(leads, 0.9) / 1000:.1f}s, "
            f"max {max(leads) / 1000:.1f}s"
        )
        print("patterns:", dict(Counter(e.get("pattern", "?") for e in hits).most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
