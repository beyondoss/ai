"""Is the conformance server itself right? Drives server.py with a raw client, no agent.

A strict server that is wrong in a lenient direction would let a broken agent pass, so each server
behaviour that check.py relies on is asserted here against the spec text (section named per test).
Runs in seconds. Exit status is nonzero on any failure.

    venv/bin/python crates/agent/eval/mcp_events_conformance/selftest.py"""
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from standardwebhooks import Webhook

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "server.py")
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}
SECRET = "whsec_" + base64.b64encode(os.urandom(32)).decode()
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""), flush=True)


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Srv:
    def __init__(self, *flags):
        self.port, self.dir = port(), tempfile.mkdtemp(prefix="evself-")
        self.p = subprocess.Popen([sys.executable, SERVER, "--http", str(self.port), "--state-dir", self.dir, "--heartbeat-ms", "300",
                                   "--ttl-ms", "5000", *flags], stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.3).close()
                break
            except OSError:
                time.sleep(0.1)
        self.c = httpx.Client(timeout=10)
        self.n = 0

    def rpc(self, method, params=None, headers=None, auth=False, raw_headers=None):
        self.n += 1
        params = dict(params or {})
        params["_meta"] = META
        h = {"content-type": "application/json", "accept": "application/json, text/event-stream",
             "mcp-protocol-version": "2026-07-28", "mcp-method": method}
        if params.get("name") and method.startswith("events/"):
            h["mcp-name"] = params["name"]
        if auth:
            h["authorization"] = "Bearer conformance-token-alice"
        h.update(headers or {})
        for k in (raw_headers or []):
            h.pop(k, None)
        return self.c.post(f"http://127.0.0.1:{self.port}/mcp", headers=h,
                           json={"jsonrpc": "2.0", "id": self.n, "method": method, "params": params})

    def ctl(self, **b):
        return self.c.post(f"http://127.0.0.1:{self.port}/control", json=b).json()

    def violations(self):
        p = os.path.join(self.dir, "violations.log")
        return [json.loads(l)["rule"] for l in open(p)] if os.path.exists(p) else []

    def stop(self):
        self.p.terminate()
        self.p.wait(10)


def err(r):
    return (r.json().get("error") or {}).get("code")


class Recv(BaseHTTPRequestHandler):
    log = []
    mode = "echo"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        Recv.log.append(({k.lower(): v for k, v in self.headers.items()}, body))
        out = b"{}"
        if b'"verification"' in body and Recv.mode == "echo":
            out = json.dumps({"challenge": json.loads(body)["challenge"]}).encode()
        elif b'"verification"' in body:
            out = json.dumps({"challenge": "wrong"}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def sse(c, port, method, params, auth=False):
    """-> iterator of decoded SSE JSON frames."""
    h = {"content-type": "application/json", "accept": "application/json, text/event-stream", "mcp-protocol-version": "2026-07-28",
         "mcp-method": method}
    if params.get("name"):
        h["mcp-name"] = params["name"]
    params = dict(params, _meta=META)
    with c.stream("POST", f"http://127.0.0.1:{port}/mcp", headers=h, json={"jsonrpc": "2.0", "id": 77, "method": method, "params": params}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


def main():
    s = Srv()
    try:
        # 2026-07-28 discover + SEP-3415 "Capability Negotiation"
        d = s.rpc("server/discover").json()["result"]
        check("capability: extensions[events].listChanged, resultType complete",
              d["capabilities"]["extensions"]["io.modelcontextprotocol/events"] == {"listChanged": True} and d["resultType"] == "complete")
        # "Request: events/list": pagination reaches every type; x-mcp-header present
        names, cur = [], None
        while True:
            r = s.rpc("events/list", {"cursor": cur} if cur else {}).json()["result"]
            names += [e["name"] for e in r["events"]]
            cur = r.get("nextCursor")
            if not cur:
                break
        check("events/list: paginated, all types reachable", {"ticket.updated", "region.alert", "temp.removed", "decoy.pushonly"} <= set(names), names)
        # 2026-07-28 "Server Validation"
        for label, kw, rule in [
            ("missing Mcp-Name", dict(raw_headers=["mcp-name"]), "http-mcp-name-present"),
            ("mismatched Mcp-Name", dict(headers={"mcp-name": "other"}), "http-mcp-name-matches"),
            ("missing Mcp-Method", dict(raw_headers=["mcp-method"]), "http-mcp-method-present"),
            ("mismatched Mcp-Method", dict(headers={"mcp-method": "events/stream"}), "http-mcp-method-matches"),
        ]:
            r = s.rpc("events/poll", {"name": "ticket.updated", "cursor": None}, **kw)
            check(f"HTTP: {label} -> 400 + -32020 (and logged)", r.status_code == 400 and err(r) == -32020 and rule in s.violations(), r.text)
        args = {"region": "us-west1", "label": "héllo", "level": 3}
        ok_h = {"mcp-param-region": "us-west1", "mcp-param-label": "=?base64?" + base64.b64encode("héllo".encode()).decode() + "?=",
                "mcp-param-level": "3"}
        r = s.rpc("events/subscribe", {"name": "region.alert", "arguments": args, "delivery": {"mode": "webhook", "url": "http://x", "secret": SECRET}},
                  headers=ok_h, auth=True)
        check("HTTP: correct Mcp-Param-* (plain, Base64 sentinel, integer) pass header validation", err(r) != -32020, r.text)
        for hn in ("mcp-param-region", "mcp-param-label", "mcp-param-level"):
            r = s.rpc("events/stream", {"name": "region.alert", "arguments": args}, headers={k: v for k, v in ok_h.items() if k != hn},
                      raw_headers=[])
            check(f"HTTP: missing {hn} -> 400 + -32020", r.status_code == 400 and err(r) == -32020, r.text)
        r = s.rpc("events/stream", {"name": "region.alert", "arguments": args}, headers=dict(ok_h, **{"mcp-param-label": "héllo".encode("utf-8")}))
        check("HTTP: unencoded non-ASCII Mcp-Param value is rejected", r.status_code == 400 and err(r) == -32020, r.text)
        r = s.rpc("events/list", headers={"mcp-name": "x"})
        check("HTTP: Mcp-Name on events/list is logged as a violation", "events-list-has-no-mcp-name" in s.violations())
        r = s.c.post(f"http://127.0.0.1:{s.port}/mcp", headers={"origin": "http://evil.example", "content-type": "application/json"}, json={})
        check("HTTP: invalid Origin -> 403", r.status_code == 403)
        check("HTTP: GET on the MCP endpoint -> 405", s.c.get(f"http://127.0.0.1:{s.port}/mcp").status_code == 405)
        # errors: SEP-3415 "Error Handling"
        r = s.rpc("events/poll", {"name": "nope", "cursor": None})
        check("NotFound -32023 kind event", err(r) == -32023 and r.json()["error"]["data"] == {"kind": "event"}, r.text)
        r = s.rpc("events/poll", {"name": "region.alert", "cursor": None})
        check("Unsupported -32026 {feature: deliveryMode, value: poll}",
              err(r) == -32026 and r.json()["error"]["data"] == {"feature": "deliveryMode", "value": "poll"}, r.text)
        r = s.rpc("events/poll", {"name": "ticket.updated", "arguments": {"ticket_id": 5}, "cursor": None})
        check("InvalidParams -32602 for arguments that do not match inputSchema", err(r) == -32602, r.text)
        # poll: Poll-Based Delivery
        p = {"name": "ticket.updated", "arguments": {"ticket_id": "T-1"}}
        first = s.rpc("events/poll", dict(p, cursor=None)).json()["result"]
        check("poll: null cursor -> no events, a fresh cursor, resultType complete",
              first["events"] == [] and first["cursor"] and first["resultType"] == "complete" and first["truncated"] is False)
        for i in range(3):
            s.ctl(op="emit", type="ticket.updated", data={"ticket_id": "T-1", "summary": f"s{i}"}, event_id=f"p{i}")
        s.ctl(op="emit", type="ticket.updated", data={"ticket_id": "T-2", "summary": "other"}, event_id="other")
        r = s.rpc("events/poll", dict(p, cursor=first["cursor"], maxEvents=2)).json()["result"]
        check("poll: maxEvents caps the batch and sets hasMore, argument filter applies",
              [e["eventId"] for e in r["events"]] == ["p0", "p1"] and r["hasMore"] is True and all(e.get("cursor") for e in r["events"]))
        r2 = s.rpc("events/poll", dict(p, cursor=r["cursor"])).json()["result"]
        check("poll: resuming from the returned cursor drains the rest, once", [e["eventId"] for e in r2["events"]] == ["p2"] and r2["hasMore"] is False)
        # schema epoch cursors: "Event type removal and breaking changes"
        s.ctl(op="schema_change", name="ticket.updated")
        r = s.rpc("events/poll", dict(p, cursor=r2["cursor"]))
        check("poll: a cursor from before an in-place schema change -> Unsupported schema_changed",
              err(r) == -32026 and r.json()["error"]["data"]["reason"] == "schema_changed", r.text)
        r = s.rpc("events/poll", dict(p, cursor=None))
        check("poll: a null cursor starts afresh against the new schema", "result" in r.json())
        # push: Push-Based Delivery
        frames = []

        def go():
            try:
                for f in sse(s.c, s.port, "events/stream", {"name": "ticket.updated", "arguments": {"ticket_id": "T-1"}, "cursor": None}):
                    frames.append(f)
            except httpx.HTTPError:
                pass
        t = threading.Thread(target=go, daemon=True)
        t.start()
        time.sleep(1.0)
        s.ctl(op="emit", type="ticket.updated", data={"ticket_id": "T-1", "summary": "pushed"}, event_id="push1")
        time.sleep(0.8)
        m = [f.get("method") for f in frames]
        sub = "io.modelcontextprotocol/subscriptionId"
        check("push: notifications/events/active first, heartbeats while idle, then the event",
              m[:1] == ["notifications/events/active"] and "notifications/events/heartbeat" in m and "notifications/events/event" in m, m)
        check("push: every notification carries the request id as subscriptionId",
              all(f["params"]["_meta"][sub] == 77 for f in frames if "method" in f), frames[:2])
        check("push: granted heartbeatMs on active", frames[0]["params"]["heartbeatMs"] == 300)
        s.ctl(op="revoke")
        time.sleep(0.8)
        term = [f for f in frames if f.get("method") == "notifications/events/terminated"]
        check("push: revoke -> notifications/events/terminated Forbidden -32024, then the empty final result",
              term and term[0]["params"]["error"]["code"] == -32024 and "result" in frames[-1] and frames[-1]["result"]["resultType"] == "complete", frames[-2:])
        r = s.rpc("events/poll", dict(p, cursor=None))
        check("poll after revoke -> Forbidden -32024", err(r) == -32024)
        s.ctl(op="restore")
        # listen: subscriptions/listen + eventsListChanged
        got = {}

        def listen(opt, key):
            try:
                for f in sse(s.c, s.port, "subscriptions/listen", {"notifications": {"eventsListChanged": opt}}):
                    got.setdefault(key, []).append(f)
            except httpx.HTTPError:
                pass
        for opt, key in ((True, "in"), (False, "out")):
            threading.Thread(target=listen, args=(opt, key), daemon=True).start()
        time.sleep(0.6)
        s.ctl(op="add_type")
        time.sleep(0.6)
        check("listen: acknowledged first, echoing eventsListChanged only if requested",
              got["in"][0]["method"] == "notifications/subscriptions/acknowledged" and got["in"][0]["params"]["notifications"] == {"eventsListChanged": True}
              and got["out"][0]["params"]["notifications"] == {})
        got2 = []

        def listen2():
            try:
                for f in sse(s.c, s.port, "subscriptions/listen", {"notifications": {"toolsListChanged": True, "resourcesListChanged": True, "promptsListChanged": True}}):
                    got2.append(f)
            except httpx.HTTPError:
                pass
        threading.Thread(target=listen2, daemon=True).start()
        time.sleep(0.6)
        s.ctl(op="add_tool")
        s.ctl(op="add_resource")
        time.sleep(0.6)
        check("listen: only requested and advertised types are acknowledged and sent (tools and resources yes; prompts not advertised)",
              got2[0]["params"]["notifications"] == {"toolsListChanged": True, "resourcesListChanged": True}
              and [f["method"] for f in got2[1:]] == ["notifications/tools/list_changed", "notifications/resources/list_changed"], got2)
        check("listen: requesting an undeclared type is logged", "client-listen-filter-requests-undeclared-type" in s.violations())
        check("listen: list_changed is sent only to the opted-in listener",
              [f["method"] for f in got["in"]][1:] == ["notifications/events/list_changed"] and len(got["out"]) == 1, got)
    finally:
        s.stop()

    # webhook: needs a receiver, auth, and the TLS rule both ways
    s = Srv("--allow-http-callback")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Recv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/hook"
    sub = {"name": "ticket.updated", "arguments": {"ticket_id": "T-1"}, "delivery": {"mode": "webhook", "url": url, "secret": SECRET}}
    try:
        r = s.rpc("events/subscribe", sub)
        check("webhook: unauthenticated subscribe -> Forbidden -32024", err(r) == -32024, r.text)
        r = s.rpc("events/subscribe", dict(sub, delivery=dict(sub["delivery"], secret="whsec_c2hvcnQ=")), auth=True)
        check("webhook: a secret that is not whsec_<24..64 bytes> -> -32602 (and logged)", err(r) == -32602 and "client-secret-is-whsec-24-to-64-bytes" in s.violations(), r.text)
        Recv.mode = "wrong"
        r = s.rpc("events/subscribe", sub, auth=True)
        check("webhook: endpoint that does not echo the challenge -> CallbackEndpointError -32027 challenge_failed",
              err(r) == -32027 and r.json()["error"]["data"]["reason"] == "challenge_failed", r.text)
        check("webhook: nothing is delivered before verification", s.ctl(op="state")["subs"] == [])
        Recv.mode = "echo"
        Recv.log.clear()
        r = s.rpc("events/subscribe", dict(sub, ttlMs=60000), auth=True).json()["result"]
        check("webhook: subscribe result has id, refreshBefore (clamped to the server grant, never null), cursor, truncated",
              r["id"].startswith("sub_") and r["refreshBefore"] and r["truncated"] is False and "cursor" in r, r)
        v = Recv.log[0]
        check("webhook: verification envelope signed (Standard Webhooks v1) with X-MCP-Subscription-Id and a msg_verification_ id",
              json.loads(v[1])["type"] == "verification" and v[0]["webhook-id"].startswith("msg_verification_") and v[0]["x-mcp-subscription-id"] == r["id"]
              and Webhook(SECRET).verify(v[1].decode(), v[0]) is not None)
        s.ctl(op="emit", type="ticket.updated", data={"ticket_id": "T-1", "summary": "hooked"}, event_id="w1")
        time.sleep(1.0)
        d = [x for x in Recv.log if b'"w1"' in x[1]]
        check("webhook: delivery arrives, signature verifies with the standardwebhooks library", len(d) == 1 and
              Webhook(SECRET).verify(d[0][1].decode(), d[0][0]) is not None)
        hd = d[0][0]
        card = httpx.get(f"http://127.0.0.1:{s.port}/mcp/server-card").json()
        jwk = card["remotes"][0]["extensionsInfo"]["io.modelcontextprotocol/events"]["webhooks"]["signingKeys"]["keys"][0]
        pub = Ed25519PublicKey.from_public_bytes(base64.urlsafe_b64decode(jwk["x"] + "=="))
        v1a = [x for x in hd["webhook-signature"].split(" ") if x.startswith("v1a,")]
        try:
            pub.verify(base64.b64decode(v1a[0][4:]), f"{hd['webhook-id']}.{hd['webhook-timestamp']}.{d[0][1].decode()}".encode())
            ok = True
        except Exception:
            ok = False
        check("webhook: v1a (Ed25519) signature verifies against the key on the Server Card (kty OKP, crv Ed25519, kid)",
              ok and jwk["kty"] == "OKP" and jwk["crv"] == "Ed25519" and jwk["kid"])
        check("webhook: payload cursor is a watermark (non-null for a replaying type)", json.loads(d[0][1])["cursor"])
        r3 = s.rpc("events/subscribe", dict(sub, delivery=dict(sub["delivery"], secret="whsec_" + base64.b64encode(os.urandom(32)).decode())), auth=True).json()["result"]
        check("webhook: refresh keeps the id and reports deliveryStatus", r3["id"] == r["id"] and r3["deliveryStatus"]["active"] is True)
        Recv.log.clear()
        s.ctl(op="emit", type="ticket.updated", data={"ticket_id": "T-1", "summary": "rotated"}, event_id="w2")
        time.sleep(1.0)
        sigs = [x for x in Recv.log if b'"w2"' in x[1]][0][0]
        n = len([x for x in sigs["webhook-signature"].split(" ") if x.startswith("v1,")])
        check("webhook: secret rotation is dual-signed (v1 old + v1 new)", n == 2, n)
        s.ctl(op="schema_change", name="ticket.updated")
        time.sleep(1.0)
        term = [x for x in Recv.log if b'"terminated"' in x[1]]
        check("webhook: schema change -> signed terminated envelope with Unsupported schema_changed",
              term and json.loads(term[0][1])["error"]["data"]["reason"] == "schema_changed")
        r = s.rpc("events/unsubscribe", {"name": "ticket.updated", "arguments": {"ticket_id": "T-1"}, "delivery": {"url": url}}, auth=True)
        check("unsubscribe of a subscription that no longer exists -> NotFound -32023 kind subscription",
              err(r) == -32023 and r.json()["error"]["data"] == {"kind": "subscription"}, r.text)
    finally:
        s.stop()
        srv.shutdown()

    s = Srv()
    try:
        r = s.rpc("events/subscribe", {"name": "ticket.updated", "arguments": {"ticket_id": "T-1"},
                                       "delivery": {"mode": "webhook", "url": "http://127.0.0.1:1/hook", "secret": SECRET}}, auth=True)
        check("webhook: non-https callback -> InvalidParams -32602 without --allow-http-callback", err(r) == -32602, r.text)
    finally:
        s.stop()

    s = Srv("--no-events-capability")
    try:
        d = s.rpc("server/discover").json()["result"]
        r = s.rpc("events/list")
        check("no-capability: discover omits the extension; any events/* -> 404 -32601 and a violation",
              "extensions" not in d["capabilities"] and r.status_code == 404 and err(r) == -32601
              and "client-must-not-send-events-without-capability" in s.violations())
    finally:
        s.stop()
    fails = results.count(False)
    print(f"\n{len(results) - fails}/{len(results)} server self-checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
