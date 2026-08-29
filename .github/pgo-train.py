#!/usr/bin/env python3
"""PGO training workload for `beyond-ai-agent`.

Drives a PGO-*instrumented* build through the paths that decide the shipped binary's resident text,
so the optimizer knows which code actually runs and can lay it out contiguously. See
`.github/workflows/release-agent.yml` for why that matters (it is a memory win, not just a speed one:
the agent's idle RSS is dominated by resident *code* pages, and those are locality-bound rather than
size-bound — `opt-level="z"` measurably makes it *worse*).

Talks to a mock Anthropic-SSE gateway on loopback rather than a real provider: training must be
hermetic, free, and identical on every runner. The point is exercising *our* decode/dispatch/session
code, and a mock drives all of it.

**Every process must exit cleanly.** The LLVM profiling runtime writes its `.profraw` at exit, so a
`SIGKILL`ed process contributes nothing. `serve` is stopped with `SIGTERM` (its graceful-shutdown
path) and `run` is left to finish on its own.

Fails loudly if no profile was produced. A silent training failure is the dangerous outcome: the
build would still succeed and quietly ship a binary optimized against an *empty* profile, which is
worse than not using PGO at all.
"""

import http.server
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

# Absolute, because every child below runs with `cwd` set to a scratch dir: a caller-relative path
# (which is what the workflow passes) would not resolve from there.
BIN = os.path.abspath(sys.argv[1])
PROFRAW_DIR = os.path.abspath(sys.argv[2])


def sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def turn_text(text):
    """A plain assistant turn — exercises the streaming text-delta decode path."""
    return sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 1200, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *[{"type": "content_block_delta", "index": 0,
           "delta": {"type": "text_delta", "text": text}} for _ in range(40)],
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 180}},
        {"type": "message_stop"},
    ])


def turn_tool(call_id, name, args):
    """A tool-use turn — exercises argument coercion, dispatch, and result framing."""
    return sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 900, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ])


# One scripted turn per request, cycling: each `run` below therefore covers a tool call *and* the
# text turn that consumes its result, which is the real two-turn shape of an agent step.
SCRIPT = [
    lambda w: turn_tool("t1", "bash", {"command": "echo hello from training"}),
    lambda w: turn_text("Training the profile. "),
    lambda w: turn_tool("t2", "write", {"path": f"{w}/note.md", "content": "# training\n" * 40}),
    lambda w: turn_text("Wrote the file. "),
    lambda w: turn_tool("t3", "read", {"path": f"{w}/note.md"}),
    lambda w: turn_text("Read it back. "),
    lambda w: turn_tool("t4", "grep", {"pattern": "training", "path": w}),
    lambda w: turn_text("Searched. "),
    lambda w: turn_tool("t5", "ls", {"path": w}),
    lambda w: turn_text("Listed. "),
]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    n = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = SCRIPT[Handler.n % len(SCRIPT)](self.server.workdir).encode()
        Handler.n += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    workdir = tempfile.mkdtemp(prefix="pgo-train-")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    srv = Server(("127.0.0.1", port), Handler)
    srv.workdir = workdir
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    env = dict(
        os.environ,
        AI_AGENT_KEY="sk-pgo-training",
        AI_GATEWAY_URL=f"http://127.0.0.1:{port}",
        # A HOME that does not exist: training must not read (or write) the runner's real `~/.claude`.
        HOME=os.path.join(workdir, "home"),
    )

    # `run`: the one-shot path. Each invocation covers process start, config/skill discovery, prompt
    # assembly, a streamed turn, a tool call, and a clean exit that flushes the profile.
    for i in range(5):
        subprocess.run([BIN, "run", f"training task {i}"], cwd=workdir, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)

    # `serve`: the shape that actually ships in a guest (see beyond's beyond-agent.service). SIGTERM
    # rather than SIGKILL so its graceful-shutdown path runs and the profile is written.
    sock_path = os.path.join(workdir, "agent.sock")
    proc = subprocess.Popen(
        [BIN, "serve", "--listen-uds", sock_path, "--session-dir", os.path.join(workdir, "sessions")],
        cwd=workdir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Non-turn entry points, cheap and each its own clean exit.
    for args in (["--version"], ["tools"], ["models"]):
        subprocess.run([BIN, *args], cwd=workdir, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)

    raw = [f for f in os.listdir(PROFRAW_DIR) if f.endswith(".profraw")] \
        if os.path.isdir(PROFRAW_DIR) else []
    if not raw:
        sys.exit(
            f"PGO training produced no .profraw in {PROFRAW_DIR} (the mock served {Handler.n} model "
            f"turns, so a non-zero count here means the binary ran but was not built with "
            f"-Cprofile-generate). Refusing to continue: the build would otherwise succeed and ship "
            f"a binary optimized against an empty profile."
        )
    print(f"PGO training wrote {len(raw)} .profraw file(s) from {Handler.n} model turns")


if __name__ == "__main__":
    main()
