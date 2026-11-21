"""End-to-end MCP Events conformance check. Drives the real `agent serve` against server.py (a strict,
independent SEP-3415 server written from the spec text) over Streamable HTTP and stdio, for each delivery
mode, and judges the agent by what it sends the server (requests.log, violations.log) and what it does
with events (WebSocket frames, and the requests it makes to a mock model gateway).

    python3 -m venv venv && venv/bin/pip install -r crates/agent/eval/mcp_events_conformance/requirements.txt
    cargo build -p beyond-agent
    AGENT_BIN=target/debug/agent venv/bin/python crates/agent/eval/mcp_events_conformance/check.py [http|stdio|all] [--only SUBSTRING]

Every assertion prints PASS or FAIL with a reason. Exit status is nonzero on any FAIL. Nothing is skipped:
a dependency that is missing is a FAIL with the install line, and an assertion that cannot run because an
earlier step failed is a FAIL that says which step.

The agent needs a model to react to an event. It is pointed at a mock gateway (this file) that speaks the
Anthropic Messages streaming API and records each request body, so "the event reached a model prompt" is
asserted on the request the model received, not on an agent-side frame."""
import asyncio
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL = f"{sys.executable} -m pip install -r {os.path.join(HERE, 'requirements.txt')}"
try:
    import websockets
    import httpx  # noqa: F401  (server.py)
    import jsonschema  # noqa: F401
    import uvicorn  # noqa: F401
    import starlette  # noqa: F401
    import cryptography  # noqa: F401
    import standardwebhooks  # noqa: F401
except ImportError as e:
    print(f"FAIL dependencies: {e}. Install with: {INSTALL}")
    sys.exit(1)

AGENT = os.path.abspath(os.environ.get("AGENT_BIN", "target/debug/agent"))
SERVER = os.path.join(HERE, "server.py")
EXT = "io.modelcontextprotocol/events"
SUMMARY = "conformance-summary-7f3a"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def ctl(port, body, timeout=10):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/control", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_port(port, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.5).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


class MockGateway:
    """Anthropic Messages API, streamed. Records every request body."""

    def __init__(self):
        self.bodies = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                outer.bodies.append(self.rfile.read(n).decode("utf-8", "replace"))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()

                def ev(name, d):
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(d)}\n\n".encode())
                ev("message_start", {"type": "message_start", "message": {
                    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
                    "stop_reason": None, "usage": {"input_tokens": 1, "output_tokens": 1}}})
                ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "noted"}})
                ev("content_block_stop", {"type": "content_block_stop", "index": 0})
                ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}})
                ev("message_stop", {"type": "message_stop"})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


class Flow:
    """One server + one agent + one WebSocket, and the assertions made against them."""

    def __init__(self, name, transport, mode, *, server_flags=(), subs=None, extra_env=None):
        self.name, self.transport, self.mode = name, transport, mode
        self.out = []
        self.fails = 0
        self.tmp = tempfile.mkdtemp(prefix="evconf-")
        self.sp, self.ap = free_port(), free_port()
        self.state = os.path.join(self.tmp, "state")
        self.server_flags = ["--allow-http-callback", *server_flags]
        self.frames = []
        self.gw = MockGateway()
        self.server = self.agent = self.reader = self.ws = None
        self.subs = subs if subs is not None else self.default_subs()
        self.extra_env = extra_env or {}

    # ---- plumbing
    def default_subs(self):
        m = self.mode
        subs = [{"name": "ticket.updated", "arguments": {"ticket_id": "T-1"}, "delivery": m, "action": "follow_up"}]
        # Annotated properties (x-mcp-header): plain string, non-ASCII string, integer. No poll for this type.
        subs.append({"name": "region.alert", "arguments": {"region": "us-west1", "label": "héllo wörld", "level": 3},
                     "delivery": "webhook" if m == "webhook" else "push", "action": "notify"})
        return subs

    def say(self, ok, label, detail=""):
        if not ok:
            self.fails += 1
        self.out.append(("PASS " if ok else "FAIL ") + f"[{self.name}] {label}" + (f": {detail}" if detail and not ok else ""))

    def server_cmd(self, extra=()):
        base = [sys.executable, SERVER, "--state-dir", self.state, *self.server_flags, *extra]
        return base + (["--http", str(self.sp)] if self.transport == "http" else ["--stdio", "--control-port", str(self.sp)])

    def start_server(self, extra=()):
        self.server = subprocess.Popen(self.server_cmd(extra), stdout=subprocess.DEVNULL,
                                       stderr=open(os.path.join(self.tmp, "server.err"), "a"))
        return wait_port(self.sp)

    async def start(self):
        if self.transport == "http" and not self.start_server():
            self.say(False, "server starts", open(os.path.join(self.tmp, "server.err")).read()[-500:])
            return False
        home = os.path.join(self.tmp, "home")
        os.makedirs(os.path.join(home, ".beyond"))
        if self.transport == "http":
            transport = {"transport": "http", "url": f"http://127.0.0.1:{self.sp}/mcp",
                         "headers": {"Authorization": "Bearer conformance-token-alice"}}
        else:
            cmd = self.server_cmd()
            transport = {"transport": "stdio", "command": cmd[0], "args": cmd[1:]}
        json.dump({"mcp_servers": [{"name": "ev", **transport, "events": self.subs}]},
                  open(os.path.join(home, ".beyond", "settings.json"), "w"))
        env = dict(os.environ, HOME=home, BEYOND_AI_AGENT_MCP_IDLE_SECS="0", **self.extra_env)
        for k in ["AI_POOL_KEY_ANTHROPIC", "DATABASE_URL", "AI_DIRECT", "AI_PROVIDER", "OPENROUTER_API_KEY",
                  "ANTHROPIC_API_KEY", "AI_AGENT_LIFECYCLE_URL"]:
            env.pop(k, None)
        self.agent = subprocess.Popen(
            [AGENT, "serve", "--listen", f"127.0.0.1:{self.ap}", "--mcp-events-callback-url", f"http://127.0.0.1:{self.ap}",
             "--gateway-url", f"http://127.0.0.1:{self.gw.port}", "--key", "bai_v1.test", "--model", "claude-test",
             "--session-dir", os.path.join(self.tmp, "sessions"), "--mcp-events-session", "conf"],
            env=env, stdout=subprocess.DEVNULL, stderr=open(os.path.join(self.tmp, "agent.err"), "w"))
        if not wait_port(self.ap):
            self.say(False, "agent starts", open(os.path.join(self.tmp, "agent.err")).read()[-500:])
            return False
        self.ws = await websockets.connect(f"ws://127.0.0.1:{self.ap}/_beyond/agent?session_id=conf")

        async def read():
            try:
                async for raw in self.ws:
                    try:
                        self.frames.append((time.time(), json.loads(raw)))
                    except ValueError:
                        pass
            except Exception:
                pass
        self.reader = asyncio.create_task(read())
        return True

    async def stop(self):
        if self.reader:
            self.reader.cancel()
        if self.ws:
            try:
                await asyncio.wait_for(self.ws.close(), 3)
            except Exception:
                pass
        for p in (self.agent, self.server):
            if p and p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in (self.agent, self.server):
            if p:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.gw.close()

    # ---- observations
    def st(self, since=0):
        try:
            return ctl(self.sp, {"op": "state", "since": since})
        except Exception:
            return {"requests": [], "subs": [], "streams": 0, "listeners": 0, "stats": {}, "now": time.time()}

    def reqs(self, since=0, prefix="events/"):
        return [r for r in self.st(since)["requests"] if (r["method"] or "").startswith(prefix)]

    def events(self, event_id=None):
        return [f for _, f in self.frames if f.get("type") == "mcp_event"
                and (event_id is None or f["event"].get("eventId") == event_id)]

    def statuses(self, since=0):
        return [f for t, f in self.frames if f.get("type") == "mcp_event_status" and t > since]

    def violations(self):
        p = os.path.join(self.state, "violations.log")
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

    async def until(self, pred, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            await asyncio.sleep(0.25)
        return pred()

    def last_status(self):
        s = self.statuses()
        return json.dumps(s[-1])[:300] if s else "no mcp_event_status frame"

    def established(self):
        st = self.st()
        if self.mode == "poll":
            return st["stats"].get("polls_ok", 0) >= 1
        if self.mode == "push":
            return st["streams"] >= 1
        return len(st["subs"]) >= 1

    def emit(self, type_, data, event_id=None):
        return ctl(self.sp, {"op": "emit", "type": type_, "data": data, **({"event_id": event_id} if event_id else {})})

    # ---- the assertions, one function per spec requirement area
    async def run_main(self):
        m = self.mode
        # -- establish
        ok = await self.until(self.established, 25)
        self.say(ok, f"{m}: subscription established on the server", f"never established; {self.last_status()}")
        if self.transport == "http":
            self.check_headers()
        # -- (a) delivery to the session and into a model prompt
        t0 = time.time()
        self.emit("ticket.updated", {"ticket_id": "T-2", "summary": "other-ticket"}, "e-other")
        self.emit("ticket.updated", {"ticket_id": "T-1", "summary": SUMMARY}, "e1")
        self.emit("region.alert", {"region": "us-west1", "summary": "alert-1"}, "a1")
        got = await self.until(lambda: self.events("e1"), 20)
        self.say(got, f"(a) {m}: emitted event reaches the session as an mcp_event frame", self.last_status())
        got_prompt = await self.until(lambda: any("e1" in b and SUMMARY in b for b in self.gw.bodies), 20)
        self.say(got_prompt, f"(a) {m}: the event is injected into a model prompt",
                 f"{len(self.gw.bodies)} model request(s), none carrying the event")
        await asyncio.sleep(1.5)
        self.say(not self.events("e-other"), f"(a) {m}: a non-matching subscription argument is not delivered")
        got_region = await self.until(lambda: self.events("a1"), 20)
        self.say(got_region, f"(a) {m}: region.alert (annotated arguments, no replay) is delivered", self.last_status())
        if self.transport == "http":
            self.check_headers()
        # -- poll rate: the server's nextPollMs is 1000
        if m == "poll":
            a = len(self.reqs(prefix="events/poll"))
            t1 = time.time()
            await asyncio.sleep(5)
            n = len([r for r in self.reqs(prefix="events/poll") if r["ts"] > t1])
            self.say(n <= 7, "poll: honours nextPollMs (no tight loop)", f"{n} polls in 5s against nextPollMs=1000")
        # -- (c) list_changed
        before = time.time()
        ctl(self.sp, {"op": "add_type"})
        relist = await self.until(lambda: self.reqs(before, "events/list"), 8)
        st = self.st()
        self.say(relist, "(c) after `add event type` the client re-lists",
                 f"no events/list in 8s; subscriptions/listen opened {len([r for r in st['requests'] if r['method'] == 'subscriptions/listen'])} times; "
                 f"listeners={st['listeners']}")
        listens = [r for r in self.st()["requests"] if r["method"] == "subscriptions/listen"]
        self.say(bool(listens), "(c) the client opted into eventsListChanged on subscriptions/listen",
                 "no subscriptions/listen request was ever sent")
        # -- (d) schema_changed
        before = time.time()
        ctl(self.sp, {"op": "schema_change", "name": "ticket.updated"})
        sub_method = {"poll": "events/poll", "push": "events/stream", "webhook": "events/subscribe"}[m]
        # A re-list that happens only because of list_changed is still a re-list; what the SEP asks is that
        # the client resubscribes against the new descriptor and delivery resumes.
        relist = await self.until(lambda: self.reqs(before, "events/list"), 10)
        self.say(relist, f"(d) {m}: schema_changed leads to re-list", "no events/list after the change")
        if m == "poll":
            # poll holds no subscription: the stale-epoch cursor answers Unsupported and the client starts again.
            resub = await self.until(lambda: any(r["name"] == "ticket.updated" and r.get("cursor") is None and not r.get("err")
                                                 for r in self.reqs(before, sub_method)), 20)
        else:
            resub = await self.until(lambda: any(r["name"] == "ticket.updated" and not r.get("err")
                                                 for r in self.reqs(before, sub_method)), 25)
        self.say(resub, f"(d) {m}: schema_changed leads to a successful resubscribe ({sub_method})", f"none in 25s; {self.last_status()}")
        await asyncio.sleep(1.0)
        self.emit("ticket.updated", {"ticket_id": "T-1", "summary": SUMMARY + "-after"}, "e2")
        got2 = await self.until(lambda: self.events("e2"), 20)
        self.say(got2, f"(d) {m}: delivery resumes after schema_changed", self.last_status())
        if self.transport == "http" and m != "poll":
            # Churn: terminate both subscriptions repeatedly so resubscribes race the re-lists they trigger.
            # A request sent while the descriptor registry is being refreshed must still carry Mcp-Param-*.
            t_churn = time.time()
            for _ in range(5):
                for n in ("ticket.updated", "region.alert"):
                    ctl(self.sp, {"op": "schema_change", "name": n})
                await asyncio.sleep(0.4)
            await asyncio.sleep(6)
            rs = [r for r in self.reqs(t_churn, "events/") if r["method"] in ("events/stream", "events/subscribe") and r["name"] == "region.alert"]
            lost = [r for r in rs if r["hdr"].get("mcp-param-region") != "us-west1" or r["hdr"].get("mcp-name") != "region.alert"]
            self.say(bool(rs) and not lost, f"(b) {m}: Mcp-Param-* present on every resubscribe while descriptors churn",
                     f"{len(lost)} of {len(rs)} region.alert requests lacked Mcp-Name/Mcp-Param-Region" if rs else "no resubscribe seen")
        # -- (e) revoke: refusal surfaced, no tight retry loop
        t_rev = time.time()
        ctl(self.sp, {"op": "revoke"})
        await asyncio.sleep(20)
        # events/list continuation pages (cursor set) are one logical list: count the first page only.
        reqs = [r for r in self.reqs(t_rev) if not (r["method"] == "events/list" and r.get("cursor"))]
        sts = self.statuses(t_rev)
        surfaced = any(("32024" in json.dumps(s) or "orbidden" in json.dumps(s) or "revoked" in json.dumps(s).lower()
                        or s.get("kind") in ("terminated", "forbidden", "revoked")) for s in sts)
        self.say(surfaced, f"(e) {m}: Forbidden (-32024) is surfaced as a refusal/termination",
                 f"{len(sts)} status frame(s) after revoke, none naming it; last: {self.last_status()}")
        # Bounded retry means backoff: per request kind, at most 6 attempts in the first 10 s and at most
        # 3 in the next 10 s (events/list: per subscription). A tight loop (fixed or no delay) fails both; exponential backoff passes.
        first, second = {}, {}
        for r in reqs:
            k = f"{r['method']}:{r['name']}"
            d = first if r["ts"] - t_rev < 10 else second
            d[k] = d.get(k, 0) + 1
        # events/list is made once per subscription attempt, and a flow has two subscriptions.
        subs = len(self.subs)
        lim = lambda k, n: n * (subs if k.startswith("events/list") else 1)
        ok = all(v <= lim(k, 6) for k, v in first.items()) and all(v <= lim(k, 3) for k, v in second.items())
        self.say(ok, f"(e) {m}: no tight retry loop after revoke (backoff)", f"attempts 0-10s {first}; 10-20s {second}")
        if self.transport == "http" and m != "poll":
            # Where the race was seen: resubscribes overlapping the re-lists a refusal triggers.
            rs = [r for r in reqs if r["method"] in ("events/stream", "events/subscribe") and r["name"] == "region.alert"]
            lost = [r for r in rs if r["hdr"].get("mcp-param-region") != "us-west1"]
            self.say(bool(rs) and not lost, f"(b) {m}: Mcp-Param-* present on every retry while refusals trigger re-lists",
                     f"{len(lost)} of {len(rs)} retries lacked Mcp-Param-Region" if rs else "no retry seen")
        ctl(self.sp, {"op": "restore"})

    def check_headers(self):
        """(b) every events/* request over HTTP carries correct routing headers. Judged on all attempts,
        including the ones the server rejected."""
        rs = [r for r in self.reqs() if r["method"] in ("events/poll", "events/stream", "events/subscribe", "events/unsubscribe")]
        bad = [f"{r['method']} {r['name']}: Mcp-Name={r['hdr'].get('mcp-name')!r}" for r in rs
               if r["hdr"].get("mcp-name") != r["name"]]
        lbl = f"(b) {self.mode}: events/* carry Mcp-Name equal to params.name"
        self.out = [o for o in self.out if lbl not in o]
        self.say(bool(rs) and not bad, lbl, ("no events/* request reached the server" if not rs else "; ".join(bad[:2])))
        reg = [r for r in rs if r["name"] == "region.alert"]
        want = {"mcp-param-region": "us-west1",
                "mcp-param-label": "=?base64?" + base64.b64encode("héllo wörld".encode()).decode() + "?=",
                "mcp-param-level": "3"}
        lbl2 = f"(b) {self.mode}: x-mcp-header arguments mirrored as Mcp-Param-* (plain, Base64 sentinel, integer)"
        self.out = [o for o in self.out if lbl2 not in o]
        miss = []
        for r in reg:
            for k, v in want.items():
                if r["hdr"].get(k) != v:
                    miss.append(f"{r['method']}: {k}={r['hdr'].get(k)!r} want {v!r}")
        self.say(bool(reg) and not miss, lbl2, ("no region.alert request reached the server" if not reg else "; ".join(miss[:3])))

    def finish_common(self):
        v = self.violations()
        self.say(not v, "violations.log is empty (every client-side MUST the server checks)",
                 "; ".join(sorted({f"{x['rule']}: {x['detail']}"[:110] for x in v})[:5]) + (f" (+{len(v)} total)" if v else ""))


# ------------------------------------------------------------------------------------ scenarios

async def scenario_main(transport, mode):
    f = Flow(f"{transport}/{mode}", transport, mode)
    try:
        if await f.start():
            await f.run_main()
            f.finish_common()
    finally:
        await f.stop()
    return f


async def scenario_nocap(transport):
    f = Flow(f"{transport}/no-capability", transport, "push", server_flags=["--no-events-capability"])
    try:
        if await f.start():
            await asyncio.sleep(9)
            st = f.st()
            seen = [r["method"] for r in st["requests"]]
            ev = [m for m in seen if m.startswith("events/")]
            f.say("server/discover" in seen, "(f) the agent reached the server (server/discover seen)", f"saw {seen}")
            f.say(not ev, "(f) a server without the events capability gets zero events/* requests", f"saw {ev[:6]}")
            f.finish_common()
    finally:
        await f.stop()
    return f


async def scenario_core(transport):
    """subscriptions/listen for core list changes (2026-07-28 Subscriptions). No events configured: the
    client still owns one listen stream per connection. The new tool/resource is observed in the agent
    session's own tool list (`get_mcp`), not only on the wire."""
    f = Flow(f"{transport}/core-list-changed", transport, "push", subs=[])
    try:
        if await f.start():
            listen = lambda: [r for r in f.st()["requests"] if r["method"] == "subscriptions/listen"]
            ok = await f.until(lambda: listen() and f.st()["listeners"] >= 1, 20)
            f.say(ok, "(iii) the client opens subscriptions/listen", "no subscriptions/listen request (the client never listens)")
            asked = {k for r in listen() for k, v in (r.get("notifications") or {}).items() if v}
            f.say(bool(asked) and asked <= {"toolsListChanged", "resourcesListChanged", "eventsListChanged"},
                  "(iii, beyond SEP: SHOULD-level) the filter asks only for list-changed types the server declared", f"asked {sorted(asked)}")

            async def mcp():
                await f.ws.send(json.dumps({"type": "get_mcp", "id": f"g{time.time()}"}))
                await asyncio.sleep(1.5)
                gm = [fr for _, fr in f.frames if fr.get("command") == "get_mcp" and fr.get("success")]
                return gm[-1]["data"] if gm else {}
            for kind, op, list_m, nfilter in (("tools", "add_tool", "tools/list", "toolsListChanged"),
                                              ("resources", "add_resource", "resources/list", "resourcesListChanged")):
                before = time.time()
                name = ctl(f.sp, {"op": op})["name"]
                relist = await f.until(lambda: f.reqs(before, list_m), 8)
                if kind == "tools":  # strict: tools are model-visible state, refreshed on the notification
                    f.say(relist, f"(i) tools: after a runtime change the client re-lists ({list_m})",
                          f"no {list_m} in 8s; requested {nfilter}={nfilter in asked}")
                else:  # the spec does not mandate an immediate resources/list: informational only
                    f.out.append(f"INFO [{f.name}] (ii) resources: immediate resources/list after the notification: {'yes' if relist else 'no (lazy refresh allowed)'}")
                seen = False
                for _ in range(6):
                    d = await mcp()
                    seen = any(name in t for t in d.get("tools", [])) or any(name in json.dumps(r) for r in d.get("resources", []))
                    if seen:
                        break
                how = "get_mcp"
                if not seen and kind == "resources":
                    # Next use: the next model turn. The mock gateway records the tool catalog it was sent.
                    n0 = len(f.gw.bodies)
                    await f.ws.send(json.dumps({"type": "prompt", "id": f"next{time.time()}", "message": "ping"}))
                    seen = await f.until(lambda: any(name in b for b in f.gw.bodies[n0:]), 15)
                    how = "the next model turn's tool catalog"
                f.say(seen, f"({'i' if kind == 'tools' else 'ii'}) {kind}: the new {kind[:-1]} `{name}` is in the agent session by next use ({how})",
                      f"absent from get_mcp and from the next model turn; get_mcp: {json.dumps(d)[:200]}")
            f.finish_common()
    finally:
        await f.stop()
    return f


async def scenario_v1a():
    f = Flow("http/webhook-v1a", "http", "webhook", subs=[
        {"name": "ticket.updated", "arguments": {"ticket_id": "T-1"}, "delivery": "webhook", "action": "notify"}])
    try:
        if await f.start():
            ok = await f.until(f.established, 25)
            f.say(ok, "webhook subscription established", f.last_status())
            f.emit("ticket.updated", {"ticket_id": "T-1", "summary": "signed"}, "g1")
            f.say(await f.until(lambda: f.events("g1"), 15),
                  "(g, beyond SEP: our fail-closed policy) a delivery carrying v1 and a valid v1a (Server Card key) is accepted and delivered", f.last_status())
            st = f.st()
            f.say(st["stats"].get("card_gets", 0) >= 1,
                  "(g, beyond SEP: our fail-closed policy) the agent fetched the Server Card from the MCP origin to learn the signing key",
                  "no GET /mcp/server-card (nor any card fetch) was seen")
            r = ctl(f.sp, {"op": "bad_v1a", "data": {"ticket_id": "T-1", "summary": "forged"}})
            f.say(bool(r["statuses"]) and all(s == 401 for s in r["statuses"]),
                  "(g, beyond SEP: our fail-closed policy) a delivery with a correct v1 HMAC but a v1a not from the Server Card key is rejected (401)",
                  f"statuses {r['statuses']}")
            await asyncio.sleep(1.5)
            f.say(not f.events("evt-bad-v1a"), "(g, beyond SEP: our fail-closed policy) the forged delivery does not reach the session")
            f.finish_common()
    finally:
        await f.stop()
    return f


async def scenario_resume(mode):
    f = Flow(f"http/{mode}-restart", "http", mode, subs=[
        {"name": "ticket.updated", "arguments": {"ticket_id": "T-1"}, "delivery": mode, "action": "notify"}])
    try:
        if await f.start():
            ok = await f.until(f.established, 25)
            f.say(ok, f"{mode}: subscription established", f.last_status())
            f.emit("ticket.updated", {"ticket_id": "T-1", "summary": "r1"}, "r1")
            f.say(await f.until(lambda: f.events("r1"), 15), f"(h) {mode}: first event delivered before the restart", f.last_status())
            f.server.kill()
            f.server.wait()
            t_down = time.time()
            await asyncio.sleep(1.5)
            f.start_server(["--emit-on-start", json.dumps([{"type": "ticket.updated",
                            "data": {"ticket_id": "T-1", "summary": "while-down"}, "event_id": "r2"}])])
            got = await f.until(lambda: f.events("r2"), 40)
            f.say(got, f"(h) {mode}: event emitted while the server was down is replayed from the client's cursor",
                  f"not delivered in 40s; {f.last_status()}")
            await asyncio.sleep(2)
            f.say(len(f.events("r1")) == 1 and len(f.events("r2")) <= 1,
                  f"(h) {mode}: no event is delivered twice across the restart",
                  f"r1 x{len(f.events('r1'))}, r2 x{len(f.events('r2'))}")
            sub_m = {"poll": "events/poll", "push": "events/stream", "webhook": "events/subscribe"}[mode]
            resumed = [r for r in f.reqs(t_down, sub_m) if r.get("cursor")]
            f.say(bool(resumed), f"(h) {mode}: the client resumed with its persisted cursor after the restart",
                  f"post-restart {sub_m} requests: {[r.get('cursor') for r in f.reqs(t_down, sub_m)][:4]}")
            f.finish_common()
    finally:
        await f.stop()
    return f


async def main():
    args = sys.argv[1:]
    only = args[args.index("--only") + 1] if "--only" in args else None
    which = [a for a in args if a in ("http", "stdio", "all")] or ["all"]
    if not os.path.exists(AGENT):
        print(f"FAIL agent binary: {AGENT} not found; build it with `cargo build -p beyond-agent` or set AGENT_BIN")
        return 1
    transports = ["http", "stdio"] if "all" in which else which
    jobs = []
    for t in transports:
        for m in ("poll", "push", "webhook"):
            jobs.append((f"{t}/{m}", lambda t=t, m=m: scenario_main(t, m)))
        jobs.append((f"{t}/no-capability", lambda t=t: scenario_nocap(t)))
        jobs.append((f"{t}/core-list-changed", lambda t=t: scenario_core(t)))
    if "http" in transports:
        jobs.append(("http/webhook-v1a", scenario_v1a))
        for m in ("poll", "push", "webhook"):
            jobs.append((f"http/{m}-restart", lambda m=m: scenario_resume(m)))
    if only:
        jobs = [j for j in jobs if only in j[0]]
    sem = asyncio.Semaphore(int(os.environ.get("CONFORMANCE_PARALLEL", "4")))

    async def run(j):
        async with sem:
            try:
                f = await asyncio.wait_for(j[1](), 300)
            except Exception as e:  # a crashed scenario is a FAIL, never a skip
                print(f"FAIL [{j[0]}] scenario crashed: {type(e).__name__}: {e}", flush=True)
                return j[0], 1, []
            print(f"--- {j[0]}", flush=True)
            print("\n".join(f.out), flush=True)
            return j[0], f.fails, f.out
    res = await asyncio.gather(*(run(j) for j in jobs))
    total = sum(len(o) for _, _, o in res)
    fails = sum(n for _, n, _ in res)
    print(f"\n{total - fails}/{total} assertions passed, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
