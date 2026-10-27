"""A scripted stand-in for the model, to run the background tasks end to end without a paid call.

    mock_model.py --bin <agent> --runs-dir <dir> [--tasks bg-…,…] [--jobs N]

Serves the OpenAI Responses API (`POST …/responses`, streamed) on localhost and runs `run.py` with
`gpt-6-luna` pointed at it (`OPENAI_BASE_URL`), with every provider key in the environment replaced
by a dummy and `.env` not read, so nothing can reach a real provider. Each request is answered from
the task's reference trajectory (`TRAJECTORIES`): the task is recognized by its prompt, the step by
how many tool calls the conversation already holds, and a step may read earlier tool results (a job
number, a port). A trajectory does what a competent model would: start the processes, reproduce,
apply the task's reference fix (`bg_tasks.Task.fix`), restart, confirm. It starts processes with
`run_in_background` when the agent's `bash` offers it and with `nohup … &` when it does not, so the
same script drives the build before background jobs and the build after; past its last step it
ends the turn. What this exercises is the harness, not a model: task setup, the sandbox and its
leftover report, the event log and `bg_metrics`, the graders, and the agent's job handling.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import bg_tasks

PORT_VAR = """PORT=$(python -c 'import json; print(json.load(open("config.json"))["port"])'); """


class Ctx:
    """What a step may look at: whether `bash` has `run_in_background`, and earlier tool results."""

    def __init__(self, flag: bool, outputs: list[str]):
        self.flag, self.outputs = flag, outputs

    def job(self, i: int) -> int | None:
        m = re.search(r"\bas job (\d+)\b", self.outputs[i]) if i < len(self.outputs) else None
        return int(m.group(1)) if m else None

    def number(self, i: int) -> int:
        return int(re.search(r"\d+", self.outputs[i]).group())


Step = Callable[[Ctx], tuple[str, dict]]


def sh(command: str) -> Step:
    return lambda c: ("bash", {"command": command})


def start(command: str, log: str) -> Step:
    """Start a long-lived process: the flag where `bash` has it, else the shell's `nohup … &`."""
    return lambda c: (("bash", {"command": f"{command} > {log} 2>&1", "run_in_background": True}) if c.flag
                      else ("bash", {"command": f"nohup {command} > {log} 2>&1 &"}))


def stop(i: int, script: str) -> Step:
    """Stop what step `i` started: its job where it is one, else by name (only inside the sandbox)."""
    def step(c: Ctx) -> tuple[str, dict]:
        job = c.job(i)
        if job is not None:
            return "bash_job", {"job": job, "kill": True}
        # `[x]yz` so the pattern does not match the shell running pkill, whose own argv holds it.
        return "bash", {"command": f"pkill -f '[{script[0]}]{script[1:]}'; sleep 0.5"}
    return step


def patch(task: str) -> Step:
    """Apply the task's reference edits (`FIXES`, filled by `_load_tasks`)."""
    def step(c: Ctx) -> tuple[str, dict]:
        data = base64.b64encode(json.dumps(FIXES[task]).encode()).decode()
        return sh("python - <<'EOF'\nimport base64, json, pathlib\n"
                  f"for f, old, new in json.loads(base64.b64decode('{data}')):\n"
                  "    p = pathlib.Path(f)\n    s = p.read_text()\n    assert s.count(old) == 1, f\n"
                  "    p.write_text(s.replace(old, new))\nprint('patched')\nEOF")(c)
    return step


def _build_wait(c: Ctx) -> tuple[str, dict]:
    if c.flag and c.job(0) is not None:
        return "bash_job", {"job": c.job(0)}
    return "bash", {"command": "while ! grep -q 'BUILD OK' /tmp/build.log; do sleep 5; tail -1 /tmp/build.log; done"}


def _alt(c: Ctx, then: str) -> tuple[str, dict]:
    return start(f"python metrics_service.py --port {c.number(2)}", "/tmp/metrics2.log")(c) if then == "start" else \
        sh(f"sleep 1; curl -s localhost:{c.number(2)}/health; curl -s localhost:{c.number(2)}/metrics")(c)


TRAJECTORIES: dict[str, list[Step]] = {
    "bg-api-bugs": [
        start("python app/server.py", "/tmp/server.log"),
        sh(PORT_VAR + "sleep 1; curl -s \"localhost:$PORT/items?tag=fruit\" | head -c 300; echo; curl -s localhost:$PORT/items | head -c 300"),
        patch("bg-api-bugs"),
        stop(0, "app/server.py"),
        start("python app/server.py", "/tmp/server2.log"),
        sh(PORT_VAR + "sleep 1; curl -s localhost:$PORT/items | python -m json.tool | grep -c '\"id\"'"),
    ],
    "bg-queue-worker": [
        start("python queue_api.py", "/tmp/api.log"),
        start("python worker.py", "/tmp/worker.log"),
        sh(PORT_VAR + "sleep 1; curl -s -X POST -d '{\"text\": \"hello big world\"}' localhost:$PORT/jobs; sleep 2; "
           "curl -s localhost:$PORT/jobs/1"),
        patch("bg-queue-worker"),
        stop(1, "worker.py"),
        stop(0, "queue_api.py"),
        start("python queue_api.py", "/tmp/api2.log"),
        start("python worker.py", "/tmp/worker2.log"),
        sh(PORT_VAR + "sleep 1; curl -s -X POST -d '{\"text\": \"one two three\"}' localhost:$PORT/jobs; sleep 2; "
           "curl -s localhost:$PORT/jobs/2"),
    ],
    "bg-watch-tests": [
        start("python watch.py", "/tmp/watch.log"),
        sh("sleep 2; cat /tmp/watch.log"),
        patch("bg-watch-tests"),
        sh("sleep 3; tail -5 /tmp/watch.log"),
    ],
    "bg-long-build": [
        lambda c: ("bash", {"command": "python build.py", "run_in_background": True}) if c.flag
        else ("bash", {"command": "nohup python build.py > /tmp/build.log 2>&1 &"}),
        patch("bg-long-build"),
        sh("python -m unittest tests.test_duration"),
        _build_wait,
        sh("python - <<'EOF'\nimport json\nc = json.load(open('dist/bundle.json'))['checksum']\n"
           "open('RELEASE.md', 'w').write(f'# Release 2.3.0\\n\\nchecksum: {c}\\n')\nprint(c)\nEOF"),
    ],
    "bg-dev-server": [
        start("python serve.py", "/tmp/serve.log"),
        sh(PORT_VAR + "sleep 6; curl -s localhost:$PORT/api/stats"),
        patch("bg-dev-server"),
        stop(0, "serve.py"),
        start("python serve.py", "/tmp/serve2.log"),
        sh(PORT_VAR + "sleep 6; curl -s localhost:$PORT/api/stats"),
    ],
    "bg-port-conflict": [
        start("python metrics_service.py", "/tmp/metrics.log"),
        sh("sleep 1; cat /tmp/metrics.log"),
        sh("python -c 'import socket; s = socket.socket(); s.bind((\"127.0.0.1\", 0)); print(s.getsockname()[1])'"),
        lambda c: _alt(c, "start"),
        patch("bg-port-conflict"),
        stop(3, "metrics_service.py"),
        lambda c: _alt(c, "start"),
        lambda c: _alt(c, "check"),
    ],
    "bg-log-errors": [
        start("python pricing_server.py", "/tmp/pricing.log"),
        sh("sleep 1; python load.py"),
        sh("grep -A12 ERROR logs/server.log | head -60"),
        patch("bg-log-errors"),
        stop(0, "pricing_server.py"),
        start("python pricing_server.py", "/tmp/pricing2.log"),
        sh("sleep 1; python load.py"),
    ],
    "bg-cleanup": [
        start("python kv.py", "/tmp/kv.log"),
        sh("sleep 1; python seed.py"),
        start("python api.py", "/tmp/api.log"),
        sh("sleep 1; python smoke.py"),
        patch("bg-cleanup"),
        stop(2, "api.py"),
        start("python api.py", "/tmp/api2.log"),
        sh("sleep 1; python smoke.py | tee smoke.txt"),
        stop(0, "kv.py"),
        stop(6, "api.py"),
    ],
}

FIXES: dict[str, list] = {}
PROMPTS: dict[str, str] = {}


def _load_tasks() -> None:
    """Each task's reference edits and prompt, from a throwaway copy of it."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, make in bg_tasks.BG_TASKS.items():
            work = Path(tmp) / name
            work.mkdir()
            t = make(work)
            t.close()
            FIXES[name] = [list(f) for f in t.fix]
            PROMPTS[name] = t.prompt.split("\n")[0][:40]


def reply(body: dict) -> list[dict]:
    """The SSE events answering one request."""
    text = json.dumps(body.get("input", []))
    task = next((n for n, p in PROMPTS.items() if p in text), None)
    tools = body.get("tools") or []
    bash = next((t for t in tools if t.get("name") == "bash"), {})
    flag = "run_in_background" in json.dumps(bash.get("parameters", {}))
    items = body.get("input", [])
    calls = [i for i in items if isinstance(i, dict) and i.get("type") == "function_call"]
    outputs = {i.get("call_id"): i.get("output") for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"}
    usage = {"input_tokens": len(text) // 4, "output_tokens": 40, "input_tokens_details": {"cached_tokens": 0}}
    script = TRAJECTORIES.get(task, []) if tools else []
    if len(calls) < len(script):
        hist = [o if isinstance(o, str) else json.dumps(o) for o in (outputs.get(c.get("call_id"), "") for c in calls)]
        name, args = script[len(calls)](Ctx(flag, hist))
        n = len(calls) + 1
        item = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": name, "arguments": json.dumps(args)}
        return [{"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": ""}},
                {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": item["arguments"]},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": {"status": "completed", "usage": usage}}]
    msg = "Done: fixed, restarted and confirmed." if task else "Mock title"
    return [{"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": msg},
            {"type": "response.output_item.done", "output_index": 0,
             "item": {"type": "message", "id": "msg_1", "phase": "final_answer", "content": [{"type": "output_text", "text": msg}]}},
            {"type": "response.completed", "response": {"status": "completed", "usage": usage}}]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        sse = "".join(f"data: {json.dumps(e)}\n\n" for e in reply(body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)

    def log_message(self, *a) -> None:
        pass


def serve() -> str:
    """Start the mock on a leased port; returns its base URL."""
    _load_tasks()
    lease = bg_tasks.Lease(1)
    srv = ThreadingHTTPServer(("127.0.0.1", lease.ports[0]), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{lease.ports[0]}/v1"


def offline_env(base: str) -> None:
    """No real credential left in this process's environment, `.env` not read, OpenAI at `base`."""
    for k in [k for k in os.environ if any(s in k for s in ("KEY", "TOKEN", "SECRET")) or k in ("AI_BASE_URL", "AI_PROVIDER")]:
        del os.environ[k]
    os.environ.update(OPENAI_API_KEY="mock-key-not-a-credential", OPENAI_BASE_URL=base, LONGCTX_ENV=os.devnull)


def main(argv: list[str]) -> None:
    import argparse

    import run

    ap = argparse.ArgumentParser(prog="mock_model")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--tasks", default=",".join(bg_tasks.BG_TASKS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--label", default="mock")
    a = ap.parse_args(argv)
    base = serve()
    offline_env(base)
    assert os.environ["OPENAI_BASE_URL"].startswith("http://127.0.0.1:"), "the mock must be local"
    run.main(["--bin", a.bin, "--models", "gpt-6-luna", "--arms", "model", "--tasks", a.tasks, "--attempts",
              str(a.attempts), "--jobs", str(a.jobs), "--cap-usd", "1", "--max-steps", "40", "--wall-min", "10",
              "--runs-dir", a.runs_dir, "--label", a.label])


if __name__ == "__main__":
    main(sys.argv[1:])
