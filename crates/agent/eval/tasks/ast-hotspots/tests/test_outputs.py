"""Hidden verifier: AST-precise dangerous calls, not grep hits."""
from __future__ import annotations

import json
from pathlib import Path

GOLD = [
    {
        "file": "jobs.py",
        "function": "run_job",
        "line": 7,
        "callee": "subprocess.Popen",
    },
    {
        "file": "jobs.py",
        "function": "wipe",
        "line": 11,
        "callee": "os.system",
    },
    {
        "file": "legacy.py",
        "function": "load_plugin",
        "line": 2,
        "callee": "exec",
    },
    {
        "file": "routes.py",
        "function": "handle",
        "line": 6,
        "callee": "eval",
    },
]


def test_report_matches_gold():
    path = Path("/app/report.json")
    assert path.is_file(), "missing /app/report.json"
    data = json.loads(path.read_text())
    assert isinstance(data, list), "report.json must be a JSON array"
    normalized = []
    for row in data:
        normalized.append(
            {
                "file": str(row["file"]).removeprefix("svc/").removeprefix("/app/svc/"),
                "function": row["function"],
                "line": int(row["line"]),
                "callee": row["callee"],
            }
        )
    normalized.sort(key=lambda r: (r["file"], r["line"]))
    assert normalized == GOLD, f"got {normalized!r}"


if __name__ == "__main__":
    test_report_matches_gold()
    print("ok")
