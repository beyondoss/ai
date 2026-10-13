"""What a run did with long-running processes, read from its `events.jsonl` (any task, either build).

    bg_metrics.py <run dir>…        print each run's metrics as JSON

Every count comes from the event stream, so a run from before background jobs existed (no
`run_in_background`, `timeout_ms` that kills, no notices) is measured the same way as one after:

- `bash_calls`, `bash_job_calls` (`job_waits`, `job_kills`): tool calls.
- `job_wait_s`: seconds spent inside `bash_job` calls that were not kills (waits and peeks): the
  model blocked on a job.
- `run_in_background`: bash calls with `run_in_background: true`.
- `shell_background`: bash calls that background with the shell instead: a trailing `&`, `nohup`,
  `setsid`, `disown`, `tmux`/`screen -d`.
- `timeout_ms_set` / `timeout_ms_short`: calls that set `timeout_ms` / set it to 10 s or less.
- `timeout_returns`: calls that came back at their `timeout_ms` with the command still running.
- `timeout_kills`: calls whose command was killed by a timeout ("Command timed out after").
- `idle_yields`: calls that returned early because the command went quiet or kept printing.
- `notices`: finish notices the agent pushed without being asked (one per `[job N …]` header in a
  `background` event); `reminders`: end-of-run "jobs still running" reminders.
- `sleep_polls`: calls that wait by sleeping (`sleep N`, a `while … sleep` loop) or look again at a
  log they already looked at (the same `tail`/`cat`/`grep` of a `*.log` or `logs/` file).
- `launches`: backgrounded starts (by any means above, or a call that came back as a job) of a
  task's long-lived scripts (`launches=` names them; otherwise any `*.py`).
- `duplicate_launches`: launches of the same command (script plus arguments, so a restart on another
  port is not one) while an earlier launch of it is still taken to be running: not since stopped by a kill command (`kill`, `pkill`, `killall`, `fuser -k`), a
  `bash_job` kill, a finish notice for its job, or failing at once ("Address already in use", a
  traceback, a background start that reported its exit). A kill command clears every launch, so
  this undercounts rather than overcounts.
- `jobs_stopped_at_end`: jobs the agent itself stopped as the run ended (`bash_jobs_stopped`).

`processes_left` is not here: it comes from the sandbox (`sandbox.survivors`), not the events.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SLEEP = re.compile(r"(?:^|[;&|(\s])sleep\s+\d")
_LOG_READ = re.compile(r"^\s*(?:tail|cat|head|grep|less|wc)\b.*(?:\.log\b|\blogs?/)")
_SHELL_BG = re.compile(r"(?<![&>|])&\s*(?:$|;|\n|\)|disown\b|wait\b)|\bnohup\b|\bsetsid\b|\bdisown\b|\btmux\b|\bscreen\s+-d")
_KILL = re.compile(r"\b(?:kill|pkill|killall)\b|\bfuser\s+(?:-\w+\s+)*-k")
_SCRIPT = re.compile(r"\bpython[\d.]*\s+(?:-[A-Za-z]+\s+)*([\w./-]+\.py)((?:[ \t]+[^\s<>&|;()]+(?=\s|$|[;&|)]))*)")
_JOB = re.compile(r"\bas job (\d+)\b")
_NOTICE = re.compile(r"^\[job (\d+) ", re.M)
_YIELD = re.compile(r"^No output for \d+ s\.|\bStill printing after \d+ s\b", re.M)
_DIED = re.compile(r"Address already in use|Traceback \(most recent call last\)")

FIELDS = ("bash_calls", "bash_job_calls", "job_waits", "job_wait_s", "job_kills", "run_in_background",
          "shell_background", "timeout_ms_set", "timeout_ms_short", "timeout_returns", "timeout_kills", "idle_yields", "notices",
          "reminders", "sleep_polls", "launches", "duplicate_launches", "jobs_stopped_at_end")


def targets(command: str, launches: tuple[str, ...] = ()) -> set[str]:
    """The long-lived processes `command` starts, each as its script's file name plus its arguments
    (up to a redirect or a shell operator): `server.py --port 9001` and `server.py` are two servers."""
    found = {(Path(m.group(1)).name, " ".join(m.group(2).split())) for m in _SCRIPT.finditer(command)}
    return {" ".join(filter(None, f)) for f in found if not launches or f[0] in launches}


def metrics(events: list[dict], launches: tuple[str, ...] = ()) -> dict:
    out = dict.fromkeys(FIELDS, 0)
    starts: dict[str, dict] = {}
    running: dict[str, int | None] = {}  # script → the job it runs as (None: a shell `&`, untracked)
    looked: set[str] = set()

    def ended(job: int) -> None:
        for t in [t for t, j in running.items() if j == job]:
            del running[t]

    for ev in events:
        kind = ev.get("kind")
        if kind == "tool_start":
            starts[ev.get("id", "")] = ev
        elif kind == "background":
            if ev.get("reason") == "finished":
                ids = [int(m) for m in _NOTICE.findall(ev.get("text", ""))]
                out["notices"] += len(ids) or 1
                for job in ids:
                    ended(job)
            else:
                out["reminders"] += 1
        elif kind == "bash_jobs_stopped":
            out["jobs_stopped_at_end"] += len(ev.get("jobs") or [])
        elif kind == "tool_end":
            start = starts.pop(ev.get("id", ""), None) or {}
            inp = start.get("input") or {}
            res = str(ev.get("result", ""))
            if ev.get("name") == "bash_job":
                out["bash_job_calls"] += 1
                job = inp.get("job")
                if inp.get("kill"):
                    out["job_kills"] += 1
                else:
                    out["job_waits"] += 1
                    if isinstance(start.get("t"), (int, float)) and isinstance(ev.get("t"), (int, float)):
                        out["job_wait_s"] = round(out["job_wait_s"] + ev["t"] - start["t"], 3)
                if isinstance(job, int) and (inp.get("kill") or "exited with code" in res or "was killed" in res):
                    ended(job)
                continue
            if ev.get("name") != "bash":
                continue
            cmd = str(inp.get("command", ""))
            out["bash_calls"] += 1
            flag = inp.get("run_in_background") is True
            shell = bool(_SHELL_BG.search(cmd))
            out["run_in_background"] += flag
            out["shell_background"] += shell
            tmo = inp.get("timeout_ms")
            if isinstance(tmo, (int, float)) and not isinstance(tmo, bool):
                out["timeout_ms_set"] += 1
                out["timeout_ms_short"] += tmo <= 10_000
            out["timeout_returns"] += "Returned after your timeout_ms" in res
            out["timeout_kills"] += "Command timed out after" in res
            out["idle_yields"] += bool(_YIELD.search(res))
            norm = " ".join(cmd.split())
            if _SLEEP.search(cmd) or (_LOG_READ.search(cmd) and norm in looked):
                out["sleep_polls"] += 1
            if _LOG_READ.search(cmd):
                looked.add(norm)
            if _KILL.search(cmd):
                running.clear()
            m = _JOB.search(res)
            job = int(m.group(1)) if m else None
            started = targets(cmd, launches)
            if started and (flag or shell or job is not None):
                out["launches"] += 1
                out["duplicate_launches"] += sum(1 for t in started if t in running)
                died = bool(_DIED.search(res)) or (flag and job is None)
                for t in started:
                    if died:
                        running.pop(t, None)
                    else:
                        running[t] = job
    return out


def load(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        p = Path(d)
        print(json.dumps({"run": d, **metrics(load(p / "events.jsonl" if p.is_dir() else p))}))
