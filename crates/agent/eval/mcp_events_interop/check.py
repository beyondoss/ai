"""MCP Events interop check — NOT part of CI. Drives the real `beyond-ai-agent serve --listen`
against an independent server implementation: the `mcp-webhook-events` PyPI package (0.2.0) on
the official Python MCP SDK (`mcp` 2.3.0), whose deliveries are signed by the official
`standardwebhooks` library.

    python3 -m venv venv && venv/bin/pip install mcp-webhook-events==0.2.0 uvicorn websockets
    cargo build -p beyond-ai-agent
    AGENT_BIN=target/debug/beyond-ai-agent venv/bin/python crates/agent/eval/mcp_events_interop/check.py [http|stdio]

Asserts: discovery, subscribe (incl. the package's verification challenge), a delivery reaching
the session as an `mcp_event` frame, a refresh that rotates the secret (and the package
dual-signing old+new), a re-published eventId not surfacing twice, and unsubscribe on shutdown."""
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = sys.executable
AGENT = os.path.abspath(os.environ.get("AGENT_BIN", "target/debug/beyond-ai-agent"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_port(port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.5).close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"port {port} never opened")


def http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"null")


results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


async def frames_until(ws, pred, timeout):
    deadline = time.time() + timeout
    while True:
        left = deadline - time.time()
        if left <= 0:
            return None
        try:
            raw = await asyncio.wait_for(ws.recv(), left)
        except asyncio.TimeoutError:
            return None
        f = json.loads(raw)
        if pred(f):
            return f


async def main():
    tmp = tempfile.mkdtemp()
    py_port, agent_port = free_port(), free_port()
    stdio = len(sys.argv) > 1 and sys.argv[1] == "stdio"
    print(f"--- MCP transport to the independent server: {'stdio' if stdio else 'streamable HTTP'}")
    server_cmd = [VENV_PY, os.path.join(HERE, "watch_server.py"), str(py_port), os.path.join(tmp, "events.sqlite3")]
    server = subprocess.Popen(["sleep", "3600"]) if stdio else subprocess.Popen(server_cmd)
    agent = None
    try:
        if not stdio:
            wait_port(py_port)
        home = os.path.join(tmp, "home")
        os.makedirs(os.path.join(home, ".claude"))
        with open(os.path.join(home, ".claude", "settings.json"), "w") as f:
            transport = ({"transport": "stdio", "command": server_cmd[0], "args": server_cmd[1:] + ["stdio"]}
                         if stdio else {"transport": "http", "url": f"http://127.0.0.1:{py_port}/mcp"})
            json.dump({"mcp_servers": [{
                "name": "pywatch", **transport,
                "events": [{"name": "ticket.updated", "arguments": {"ticket_id": "T-1"},
                            "delivery": "webhook", "action": "notify"}],
            }]}, f)
        env = dict(os.environ, HOME=home, BEYOND_AI_AGENT_MCP_IDLE_SECS="0")
        for k in ["AI_DIRECT", "AI_PROVIDER", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "AI_AGENT_LIFECYCLE_URL"]:
            env.pop(k, None)
        agent = subprocess.Popen([AGENT, "serve", "--listen", f"127.0.0.1:{agent_port}",
                                  "--mcp-events-callback-url", f"http://127.0.0.1:{agent_port}",
                                  "--gateway-url", "http://127.0.0.1:9", "--key", "bai_v1.test",
                                  "--model", "claude-test", "--session-dir", os.path.join(tmp, "sessions"),
                                  # The configured subscription is owned by this daemon session.
                                  "--mcp-events-session", "indep"],
                                 env=env, stdout=subprocess.DEVNULL, stderr=open(os.path.join(tmp, "agent.err"), "w"))
        wait_port(agent_port)
        if stdio:
            wait_port(py_port)
        async with websockets.connect(f"ws://127.0.0.1:{agent_port}/_beyond/agent?session_id=indep") as ws:
            sub = None
            for i in range(100):
                await ws.send(json.dumps({"type": "mcp_events_list", "id": f"l{i}"}))
                r = await frames_until(ws, lambda f: f.get("id") == f"l{i}", 10)
                subs = (r or {}).get("data", {}).get("subscriptions", [])
                if r and i == 0:
                    av = r["data"]["available"][0]
                    check("discovery: events/list from the independent server", av.get("supported") and av["events"][0]["name"] == "ticket.updated", json.dumps(av)[:200])
                if subs and subs[0]["state"] == "active":
                    sub = subs[0]
                    break
                await asyncio.sleep(0.2)
            check("subscribe: verification challenge answered, subscription active", sub is not None, json.dumps(sub))
            st = http("GET", f"http://127.0.0.1:{py_port}/control/state")
            check("server records exactly one live subscription", len(st["subscriptions"]) == 1, json.dumps(st)[:300])
            if sub:
                check("subscription id matches the server-derived id", sub["subscription_id"] == st["subscriptions"][0]["id"])

            r = http("POST", f"http://127.0.0.1:{py_port}/control/emit", {"event_id": "evt-indep-1", "data": {"ticket_id": "T-1", "summary": "from python"}})
            check("delivery acknowledged 2xx by our receiver", r["stats"]["delivered"] == 1, json.dumps(r))
            f = await frames_until(ws, lambda f: f.get("type") == "mcp_event", 10)
            check("delivery surfaces as an mcp_event frame", f is not None and f["event"]["data"]["summary"] == "from python", json.dumps(f)[:300])

            # Non-matching arguments are filtered server-side.
            r = http("POST", f"http://127.0.0.1:{py_port}/control/emit", {"data": {"ticket_id": "T-2", "summary": "other"}})
            check("server-side argument matching (no delivery for T-2)", r["queued"] == 0, json.dumps(r))

            # Wait for at least one refresh (grant is 3 s) → secret rotated, server dual-signs.
            refreshed = None
            for i in range(60):
                await ws.send(json.dumps({"type": "mcp_events_list", "id": f"r{i}"}))
                r = await frames_until(ws, lambda f: f.get("id") == f"r{i}", 10)
                s = r["data"]["subscriptions"][0]
                if s["refreshes"] >= 1:
                    refreshed = s
                    break
                await asyncio.sleep(0.25)
            check("TTL refresh re-subscribed (secret rotated on refresh)", refreshed is not None, json.dumps(refreshed)[:200])
            r = http("POST", f"http://127.0.0.1:{py_port}/control/emit", {"event_id": "evt-indep-2", "data": {"ticket_id": "T-1", "summary": "after rotation"}})
            check("post-rotation delivery (dual-signed old+new) accepted", r["stats"]["delivered"] == 1, json.dumps(r))
            f = await frames_until(ws, lambda f: f.get("type") == "mcp_event", 10)
            check("post-rotation event surfaces", f is not None and f["event"]["eventId"] == "evt-indep-2")

            # Same eventId published again: the server sends it (outbox dedups only per row), we drop it.
            r = http("POST", f"http://127.0.0.1:{py_port}/control/emit", {"event_id": "evt-indep-2", "data": {"ticket_id": "T-1", "summary": "after rotation"}})
            f = await frames_until(ws, lambda f: f.get("type") == "mcp_event", 1.5)
            check("re-published eventId not surfaced twice (the package's outbox also dedups it server-side)", f is None, json.dumps(r))

        agent.send_signal(signal.SIGTERM)
        agent.wait(timeout=20)
        # Read the package's own store: over stdio the server exits with the agent.
        import sqlite3
        rows = sqlite3.connect(os.path.join(tmp, "events.sqlite3")).execute("SELECT id FROM subscriptions").fetchall()
        check("unsubscribe on shutdown removed the server's subscription (its sqlite store is empty)", rows == [], repr(rows))
    finally:
        if agent and agent.poll() is None:
            agent.kill()
        server.terminate()
        try:
            print(open(os.path.join(tmp, "agent.err")).read()[-2000:], file=sys.stderr)
        except OSError:
            pass
    failed = [r for r in results if not r[1]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


asyncio.run(main())
