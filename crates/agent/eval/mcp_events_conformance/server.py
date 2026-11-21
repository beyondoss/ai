"""A strict MCP Events (SEP-3415) server, written only from the spec text.

Pinned sources (modelcontextprotocol/modelcontextprotocol @ bbc31f8ded5762eae4df115b93942f5e71c8b3a7):
  seps/3415-events-extension.md                       -> "SEP-3415 <section>"
  docs/specification/2026-07-28/**                    -> "2026-07-28 <page>"
  modelcontextprotocol/ext-server-card (Remote.extensionsInfo) -> "Server Card"
  Standard Webhooks                                    -> "Standard Webhooks"

Strict means two things. It performs every server-side MUST. And it checks every client-side MUST it
can observe, answers the violation with the spec's error, and appends a JSON line to violations.log.
The client is under test, so a violation is a finding, not noise. requests.log records every request
for the check to count.

Transports: Streamable HTTP (a small Starlette app, `--http PORT`) and stdio (`--stdio`). The control
endpoint (POST /control) is always HTTP: the `--control-port` in stdio mode, the same port in HTTP mode.
The `mcp` SDK (2.3.0, pinned in requirements.txt) implements none of events/*, so the JSON-RPC
envelope, SSE framing and every events method is hand-rolled here; the SDK is used by nothing else.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import httpx
import jsonschema
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from standardwebhooks import Webhook
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

EXT = "io.modelcontextprotocol/events"
VERSION = "2026-07-28"
SUB_ID = "io.modelcontextprotocol/subscriptionId"

# SEP-3415 "Error Handling" + 2026-07-28 transports/versioning.
INVALID_PARAMS, METHOD_NOT_FOUND = -32602, -32601
HEADER_MISMATCH, UNSUPPORTED_VERSION = -32020, -32022
NOT_FOUND, FORBIDDEN, EXHAUSTED, UNSUPPORTED, CALLBACK = -32023, -32024, -32025, -32026, -32027

EVENT_METHODS = {"events/list", "events/poll", "events/stream", "events/subscribe", "events/unsubscribe"}
NAMED = {"events/poll", "events/stream", "events/subscribe", "events/unsubscribe"}
TOKENS = {"conformance-token-alice": "alice"}  # the one authenticated principal


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


class RpcError(Exception):
    def __init__(self, code, message, data=None, http=200):
        self.code, self.message, self.data, self.http = code, message, data, http


def err_body(rid, code, message, data=None):
    e = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return {"jsonrpc": "2.0", "id": rid, "error": e}


# ---------------------------------------------------------------------------------- event types

def base_types():
    """The server's initial catalogue. Order matters: it is paginated two per page, so the types the
    check subscribes to sit on page 2 (SEP-3415 "Request: events/list": `nextCursor`)."""
    return [
        {"name": "temp.removed", "description": "Removed by the control endpoint.", "delivery": ["poll", "push", "webhook"],
         "inputSchema": {"type": "object", "properties": {}}, "payloadSchema": {"type": "object"}, "_replay": True},
        {"name": "decoy.pushonly", "description": "Push only.", "delivery": ["push"],
         "inputSchema": {"type": "object", "properties": {}}, "payloadSchema": {"type": "object"}, "_replay": False},
        # poll + push + webhook, replays from a cursor.
        {"name": "ticket.updated", "description": "A ticket changed.", "delivery": ["poll", "push", "webhook"],
         "inputSchema": {"type": "object", "properties": {"ticket_id": {"type": "string"}}},
         "payloadSchema": {"type": "object", "properties": {"ticket_id": {"type": "string"}, "summary": {"type": "string"}}},
         "_replay": True},
        # No replay (cursor is always null/absent) and annotated properties (x-mcp-header): a plain string,
        # a non-ASCII string (Base64 sentinel) and an integer.
        {"name": "region.alert", "description": "An alert in a region.", "delivery": ["push", "webhook"],
         "inputSchema": {"type": "object", "properties": {
             "region": {"type": "string", "x-mcp-header": "Region"},
             "label": {"type": "string", "x-mcp-header": "Label"},
             "level": {"type": "integer", "x-mcp-header": "Level"}}},
         "payloadSchema": {"type": "object", "properties": {"region": {"type": "string"}, "summary": {"type": "string"}}},
         "_replay": False},
    ]


def late_type():
    return {"name": "late.added", "description": "Added at runtime.", "delivery": ["poll", "push", "webhook"],
            "inputSchema": {"type": "object", "properties": {}}, "payloadSchema": {"type": "object"}, "_replay": True}


def public(t):
    return {k: v for k, v in t.items() if not k.startswith("_")}


def annotated(schema):
    """x-mcp-header properties statically reachable via `properties` only (2026-07-28 Streamable HTTP,
    "Schema Extension"). Returns {path-tuple: header-name}."""
    out = {}

    def walk(s, path):
        for k, v in (s.get("properties") or {}).items():
            if isinstance(v, dict):
                if "x-mcp-header" in v:
                    out[path + (k,)] = v["x-mcp-header"]
                walk(v, path + (k,))
    walk(schema, ())
    return out


def at_path(obj, path):
    for p in path:
        if not isinstance(obj, dict) or p not in obj:
            return None
        obj = obj[p]
    return obj


def decode_header(v: str):
    """2026-07-28 Streamable HTTP "Value Encoding": the Base64 sentinel."""
    if v.startswith("=?base64?") and v.endswith("?="):
        raw = v[len("=?base64?"):-len("?=")]
        return base64.b64decode(raw, validate=True).decode("utf-8")
    return v


# ---------------------------------------------------------------------------------- the server

class Server:
    def __init__(self, a):
        self.a = a
        self.state_dir = a.state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.types = {t["name"]: t for t in base_types()}
        self.epoch = {}            # per event type: bumps when the descriptor changes incompatibly
        self.revoked = False
        self.log = []              # [(seq, occurrence, type)] retained events
        self.seq = 0
        self.floor = 0             # events with seq <= floor are no longer replayable
        self.bus = []              # asyncio.Queue per live stream / listener
        self.subs = {}             # webhook subs: key -> dict
        self.verified = set()      # (principal, url)
        self.listeners = 0
        self.streams = 0
        self.tools = [{"name": "echo", "description": "Echo.",
                       "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]
        self.resources = [{"uri": "file:///conformance/initial.txt", "name": "initial"}]
        self.stats = {"deliveries_ok": 0, "deliveries_failed": 0, "violations": 0}
        self.req_log = open(os.path.join(self.state_dir, "requests.log"), "a", buffering=1)
        self.vio_log = open(os.path.join(self.state_dir, "violations.log"), "a", buffering=1)
        self.http = httpx.AsyncClient(follow_redirects=False, timeout=5.0)  # SEP-3415 "SSRF prevention": no redirects
        self.http_origin = None
        self._load()
        key_path = os.path.join(self.state_dir, "ed25519.pem")
        if os.path.exists(key_path):
            self.key = serialization.load_pem_private_key(open(key_path, "rb").read(), None)
        else:
            self.key = Ed25519PrivateKey.generate()
            open(key_path, "wb").write(self.key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        self.rogue = Ed25519PrivateKey.generate()
        if a.emit_on_start:
            for item in json.loads(a.emit_on_start):
                self.emit_sync(item["type"], item["data"], item.get("event_id"))

    # -- persistence (so a restarted server can replay: SEP-3415 "Cursor Lifecycle")
    def _load(self):
        meta = os.path.join(self.state_dir, "meta.json")
        if os.path.exists(meta):
            m = json.load(open(meta))
            self.epoch, self.revoked = m["epoch"], m["revoked"]
            self.floor = m.get("floor", 0)
            for n in m.get("removed", []):
                self.types.pop(n, None)
            for n in m.get("added", []):
                self.types[n] = late_type()
            for n, p in m.get("payload", {}).items():
                if n in self.types:
                    self.types[n]["payloadSchema"] = p
        ev = os.path.join(self.state_dir, "events.jsonl")
        if os.path.exists(ev):
            for line in open(ev):
                r = json.loads(line)
                self.seq = max(self.seq, r["seq"])
                self.log.append((r["seq"], r["occ"], r["type"]))

    def _save_meta(self):
        removed = [n for n in ("temp.removed",) if n not in self.types]
        added = [n for n in ("late.added",) if n in self.types]
        payload = {n: t["payloadSchema"] for n, t in self.types.items() if self.epoch.get(n)}
        json.dump({"epoch": self.epoch, "revoked": self.revoked, "floor": self.floor, "removed": removed,
                   "added": added, "payload": payload}, open(os.path.join(self.state_dir, "meta.json"), "w"))

    # -- logging
    def violate(self, rule, detail, **kw):
        self.stats["violations"] += 1
        self.vio_log.write(json.dumps({"ts": time.time(), "rule": rule, "detail": detail, **kw}) + "\n")

    # -- cursors: c<epoch>_<seq>; null/absent for event types that do not replay
    def mk_cursor(self, name, seq):
        t = self.types.get(name)
        if not t or not t["_replay"]:
            return None
        return f"c{self.epoch.get(name, 0)}_{seq}"

    def parse_cursor(self, name, cursor, check_epoch=False):
        """-> seq, or raises. A cursor from an older schema epoch is `Unsupported(schema_changed)`
        (SEP-3415 "Event type removal and breaking changes": a server MAY mint epoch cursors)."""
        m = re.fullmatch(r"c(\d+)_(\d+)", cursor)
        if not m:
            raise RpcError(INVALID_PARAMS, "InvalidParams", {"reason": "cursor not issued by this server"})
        if check_epoch and int(m.group(1)) != self.epoch.get(name, 0):
            raise RpcError(UNSUPPORTED, "Unsupported", {"feature": "payloadSchema", "reason": "schema_changed"})
        return int(m.group(2))

    def head(self):
        return self.seq

    # -- principal / access
    def principal(self, headers):
        if headers is None:
            return "stdio-local"  # a stdio server's trust boundary is the process
        auth = headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return TOKENS.get(auth[7:])
        return None

    def check_access(self, p):
        if self.revoked:
            raise RpcError(FORBIDDEN, "Forbidden", {"reason": "Access revoked"})

    # ------------------------------------------------------------------------------ validation
    def type_or_404(self, name):
        t = self.types.get(name)
        if not t:
            raise RpcError(NOT_FOUND, "NotFound", {"kind": "event"})
        return t

    def validate_args(self, t, args):
        args = args if args is not None else {}
        try:
            jsonschema.validate(args, t["inputSchema"])
        except jsonschema.ValidationError as e:
            self.violate("arguments-valid-for-inputSchema", f"{t['name']}: {e.message}")
            raise RpcError(INVALID_PARAMS, "InvalidParams", {"reason": e.message})
        return args

    def validate_headers(self, method, params, headers):
        """2026-07-28 Streamable HTTP "Server Validation" + SEP-3415 "Streamable HTTP: Routing Headers"."""
        def bad(msg, rule):
            self.violate(rule, msg, method=method)
            raise RpcError(HEADER_MISMATCH, f"Header mismatch: {msg}", http=400)
        if headers.get("mcp-method") is None:
            bad("Mcp-Method header missing", "http-mcp-method-present")
        if headers["mcp-method"] != method:
            bad(f"Mcp-Method {headers['mcp-method']!r} != body method {method!r}", "http-mcp-method-matches")
        pv = headers.get("mcp-protocol-version")
        body_pv = (params.get("_meta") or {}).get("io.modelcontextprotocol/protocolVersion")
        if pv is None:
            bad("MCP-Protocol-Version header missing", "http-protocol-version-header")
        if body_pv is not None and pv != body_pv:
            bad(f"MCP-Protocol-Version {pv!r} != _meta protocolVersion {body_pv!r}", "http-protocol-version-matches")
        acc = headers.get("accept", "")
        if "application/json" not in acc or "text/event-stream" not in acc:
            self.violate("http-accept-both", f"Accept: {acc!r}", method=method)
        if method == "events/list" and "mcp-name" in headers:
            self.violate("events-list-has-no-mcp-name", f"Mcp-Name sent on events/list: {headers['mcp-name']!r}")
        if method in NAMED:
            if "mcp-name" not in headers:
                bad("Mcp-Name header missing", "http-mcp-name-present")
            try:
                got = decode_header(headers["mcp-name"])
            except Exception:
                bad("Mcp-Name is not valid Base64 sentinel", "http-mcp-name-encoding")
            if got != params.get("name"):
                bad(f"Mcp-Name {got!r} != params.name {params.get('name')!r}", "http-mcp-name-matches")
            t = self.types.get(params.get("name"))
            if t:
                args = params.get("arguments") or {}
                for path, hname in annotated(t["inputSchema"]).items():
                    val = at_path(args, path)
                    hv = headers.get("mcp-param-" + hname.lower())
                    if val is None:
                        continue
                    if hv is None:
                        bad(f"Mcp-Param-{hname} missing for annotated property {'.'.join(path)}",
                            "http-mcp-param-present")
                    try:
                        dec = decode_header(hv)
                    except Exception:
                        bad(f"Mcp-Param-{hname} is not valid Base64 sentinel", "http-mcp-param-encoding")
                    if isinstance(val, bool):
                        want = "true" if val else "false"
                        ok = dec == want
                    elif isinstance(val, (int, float)):
                        try:
                            ok = float(dec) == float(val)
                        except ValueError:
                            ok = False
                    else:
                        ok = dec == val
                    if not ok:
                        bad(f"Mcp-Param-{hname} {dec!r} != body {val!r}", "http-mcp-param-matches")
                    # a value that is not header-safe must be base64-sentinel encoded
                    if isinstance(val, str) and (not val.isascii() or val != val.strip() or not val.isprintable()
                                                 or (val.startswith("=?base64?") and val.endswith("?="))):
                        if hv == val:
                            bad(f"Mcp-Param-{hname} must use the Base64 sentinel for {val!r}", "http-mcp-param-base64-required")

    # ------------------------------------------------------------------------------ dispatch
    async def handle(self, msg, headers):
        rec = {}
        try:
            return await self._handle(msg, headers, rec)
        except RpcError as e:
            rec["err"] = e.code
            raise
        finally:
            rec["ts"] = time.time()
            self.req_log.write(json.dumps(rec) + "\n")

    async def _handle(self, msg, headers, rec):
        """-> ("result", dict) | ("stream", async-generator). Raises RpcError."""
        method, params, rid = msg.get("method"), msg.get("params") or {}, msg.get("id")
        meta = params.get("_meta") or {}
        pv = meta.get("io.modelcontextprotocol/protocolVersion")
        rec.update(method=method, id=rid, name=params.get("name"))
        if pv is not None and pv != VERSION:
            raise RpcError(UNSUPPORTED_VERSION, "UnsupportedProtocolVersion",
                           {"supported": [VERSION], "requested": pv}, http=400)
        rec.update(transport="http" if headers is not None else "stdio", method=method, id=rid,
                   name=params.get("name"), arguments=params.get("arguments"), cursor=params.get("cursor"),
                   notifications=params.get("notifications"),
                   hdr={k: v for k, v in (headers or {}).items() if k.startswith("mcp-") or k == "origin"},
                   has_auth=bool(headers and headers.get("authorization")),
                   caps=(meta.get("io.modelcontextprotocol/clientCapabilities") or {}))
        if headers is not None:
            self.validate_headers(method, params, headers)
        if method == "ping":
            return "result", {"resultType": "complete"}
        if method == "server/discover":
            return "result", self.discover()
        if method == "tools/list":
            return "result", {"resultType": "complete", "tools": list(self.tools)}
        if method == "tools/call":
            return "result", {"resultType": "complete", "content": [{"type": "text", "text": str(params.get("arguments", {}).get("text", ""))}]}
        if method == "resources/list":
            return "result", {"resultType": "complete", "resources": list(self.resources)}
        if method == "resources/read":
            return "result", {"resultType": "complete", "contents": [{"uri": params.get("uri"), "text": "x"}]}
        if method == "prompts/list":
            return "result", {"resultType": "complete", "prompts": []}
        if method == "subscriptions/listen":
            return "stream", self.listen(rid, params)
        if method in EVENT_METHODS or method.startswith("events/"):
            if self.a.no_events_capability:
                # SEP-3415 "Capability Negotiation / Fallback"
                self.violate("client-must-not-send-events-without-capability", method)
                raise RpcError(METHOD_NOT_FOUND, "Method not found", http=404)
            if method not in EVENT_METHODS:
                raise RpcError(METHOD_NOT_FOUND, "Method not found", http=404)
            p = self.principal(headers)
            if method == "events/list":
                return "result", self.events_list(params)
            if method == "events/poll":
                res = self.poll(params, p)
                self.stats["polls_ok"] = self.stats.get("polls_ok", 0) + 1
                return "result", res
            if method == "events/stream":
                return "stream", self.stream(rid, params, p)
            if method == "events/subscribe":
                return "result", await self.subscribe(params, p)
            if method == "events/unsubscribe":
                return "result", await self.unsubscribe(params, p)
        raise RpcError(METHOD_NOT_FOUND, "Method not found", http=404)

    def discover(self):
        caps = {"tools": {"listChanged": True}, "resources": {"listChanged": True}}  # 2026-07-28 Subscriptions: advertised listChanged
        if not self.a.no_events_capability:
            caps["extensions"] = {EXT: {"listChanged": True}}  # SEP-3415 "Capability Negotiation"
        return {"resultType": "complete", "supportedVersions": [VERSION], "capabilities": caps,
                "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "events-conformance", "version": "1"}},
                "ttlMs": 0, "cacheScope": "private"}

    def events_list(self, params):
        names = [public(t) for t in self.types.values()]
        start = 0
        c = params.get("cursor")
        if c is not None:
            m = re.fullmatch(r"p(\d+)", str(c))
            if not m:
                raise RpcError(INVALID_PARAMS, "InvalidParams", {"reason": "bad pagination cursor"})
            start = int(m.group(1))
        size = self.a.page_size
        out = {"resultType": "complete", "events": names[start:start + size]}
        if start + size < len(names):
            out["nextCursor"] = f"p{start + size}"
        return out

    # ------------------------------------------------------------------------------ poll
    def start_position(self, t, params, mode):
        """-> (cursor_seq, truncated). SEP-3415 "Cursor Lifecycle"."""
        name, cur = t["name"], params.get("cursor")
        if cur is None or not t["_replay"]:
            if cur is not None:
                self.violate("client-must-not-replay-from-null-cursor", f"{mode} {name} sent cursor {cur!r} for a no-replay type")
            return self.head(), False
        seq = self.parse_cursor(name, cur, check_epoch=(mode == "poll"))
        trunc = seq < self.floor
        max_age = params.get("maxAgeMs")
        if max_age is not None:
            cutoff = time.time() - max_age / 1000.0
            floor_seq = max([s for s, o, n in self.log if n == name and _ts(o) < cutoff] or [0])
            if floor_seq > seq:
                seq, trunc = floor_seq, True
        if trunc:
            return max(seq, self.floor), True
        return seq, False

    def matches(self, t, args, occ):
        data = occ["data"]
        for k, v in (args or {}).items():
            if k in data and data[k] != v:
                return False
        return True

    def poll(self, params, p):
        t = self.type_or_404(params.get("name"))
        self.check_access(p)
        if "poll" not in t["delivery"]:
            self.violate("client-must-use-offered-delivery-mode", f"poll on {t['name']}")
            raise RpcError(UNSUPPORTED, "Unsupported", {"feature": "deliveryMode", "value": "poll"})
        args = self.validate_args(t, params.get("arguments"))
        seq, trunc = self.start_position(t, params, "poll")
        if params.get("cursor") is None:
            return {"resultType": "complete", "events": [], "cursor": self.mk_cursor(t["name"], self.head()),
                    "truncated": False, "hasMore": False, "nextPollMs": self.a.poll_ms}
        cap = params.get("maxEvents") or 50
        evs = [(s, o) for s, o, n in self.log if n == t["name"] and s > seq and self.matches(t, args, o)]
        batch, more = evs[:cap], len(evs) > cap
        last = batch[-1][0] if batch else (self.head() if not more else seq)
        out = []
        for s, o in batch:
            out.append(dict(o, cursor=self.mk_cursor(t["name"], s)))
        return {"resultType": "complete", "events": out, "cursor": self.mk_cursor(t["name"], last),
                "truncated": trunc, "hasMore": more, "nextPollMs": self.a.poll_ms}

    # ------------------------------------------------------------------------------ push
    def stream(self, rid, params, p):
        t = self.type_or_404(params.get("name"))
        self.check_access(p)
        if "push" not in t["delivery"]:
            self.violate("client-must-use-offered-delivery-mode", f"push on {t['name']}")
            raise RpcError(UNSUPPORTED, "Unsupported", {"feature": "deliveryMode", "value": "push"})
        args = self.validate_args(t, params.get("arguments"))
        seq, trunc = self.start_position(t, params, "push")
        hb = params.get("heartbeatMs")
        hb = min(max(int(hb), 250), 30000) if isinstance(hb, int) else self.a.heartbeat_ms
        return self._stream(rid, t["name"], args, seq, trunc, hb, params.get("cursor") is None)

    async def _stream(self, rid, name, args, seq, trunc, hb, fresh):
        meta = {SUB_ID: rid}
        q = asyncio.Queue()
        self.bus.append(q)
        self.streams += 1
        try:
            t = self.types[name]
            start = self.head() if fresh else seq
            yield {"jsonrpc": "2.0", "method": "notifications/events/active",
                   "params": {"cursor": self.mk_cursor(name, start), "truncated": trunc, "heartbeatMs": hb, "_meta": meta}}
            last = start
            # replay what happened after the cursor, then go live
            for s, o, n in list(self.log):
                if n == name and s > last and self.matches(t, args, o):
                    yield self._ev(name, s, o, meta)
                    last = s
            last = max(last, self.head() if fresh else last)
            while True:
                try:
                    sig = await asyncio.wait_for(q.get(), hb / 1000.0)
                except asyncio.TimeoutError:
                    yield {"jsonrpc": "2.0", "method": "notifications/events/heartbeat",  # SEP-3415 "Lifecycle: Heartbeat"
                           "params": {"cursor": self.mk_cursor(name, self.head()), "_meta": meta}}
                    continue
                kind = sig[0]
                if kind == "event" and sig[3] == name and sig[1] > last and self.matches(self.types[name], args, sig[2]):
                    last = sig[1]
                    yield self._ev(name, sig[1], sig[2], meta)
                elif kind == "terminate" and sig[1] in (None, name):
                    yield {"jsonrpc": "2.0", "method": "notifications/events/terminated",
                           "params": {"error": sig[2], "_meta": meta}}
                    yield {"jsonrpc": "2.0", "id": rid, "result": {"resultType": "complete", "_meta": meta}}
                    return
                elif kind == "error":
                    yield {"jsonrpc": "2.0", "method": "notifications/events/error",
                           "params": {"error": {"code": -32603, "message": "UpstreamError", "data": {"reason": "injected"}},
                                      "_meta": meta}}
        finally:
            self.streams -= 1
            self.bus.remove(q)

    def _ev(self, name, s, o, meta):
        params = dict(o)
        c = self.mk_cursor(name, s)
        if self.types[name]["_replay"]:
            params["cursor"] = c
        elif self.a.cursor_null_in_events:
            params["cursor"] = None
        # else: absent. SEP-3415 "Absent means null".
        params["_meta"] = meta
        return {"jsonrpc": "2.0", "method": "notifications/events/event", "params": params}

    # ------------------------------------------------------------------------------ listen
    def listen(self, rid, params):
        want = params.get("notifications") or {}
        return self._listen(rid, want)

    async def _listen(self, rid, want):
        meta = {SUB_ID: rid}
        advertised = {"toolsListChanged": True, "resourcesListChanged": True}
        if not self.a.no_events_capability:
            advertised["eventsListChanged"] = True  # SEP-3415 "Dynamic event types": echoed if honored
        honored = {}
        for k, v in want.items():
            if v is True and k in advertised:
                honored[k] = True
            elif v is True and k.endswith("ListChanged"):
                # A client shouldn't request what the server didn't declare (2026-07-28 Subscriptions:
                # "The client SHOULD check the acknowledged filter ... unsupported types are omitted").
                self.violate("client-listen-filter-requests-undeclared-type", f"{k} requested, server does not advertise it (SHOULD-level)")
        q = asyncio.Queue()
        self.bus.append(q)
        self.listeners += 1
        method = {"list_changed": ("eventsListChanged", "notifications/events/list_changed"),
                  "tools_changed": ("toolsListChanged", "notifications/tools/list_changed"),
                  "resources_changed": ("resourcesListChanged", "notifications/resources/list_changed")}
        try:
            yield {"jsonrpc": "2.0", "method": "notifications/subscriptions/acknowledged",
                   "params": {"_meta": meta, "notifications": honored}}
            while True:
                sig = await q.get()
                m = method.get(sig[0])
                if m and honored.get(m[0]):  # "MUST NOT send notification types the client has not explicitly requested"
                    yield {"jsonrpc": "2.0", "method": m[1], "params": {"_meta": meta}}
        finally:
            self.listeners -= 1
            self.bus.remove(q)

    def publish(self, *sig):
        for q in list(self.bus):
            q.put_nowait(sig)

    # ------------------------------------------------------------------------------ webhook
    def check_url(self, url):
        u = urllib.parse.urlparse(url or "")
        if u.scheme not in ("http", "https") or not u.hostname:
            raise RpcError(INVALID_PARAMS, "InvalidParams", {"reason": "delivery.url malformed"})
        if u.scheme != "https" and not self.a.allow_http_callback:
            # SEP-3415 "TLS requirement"
            raise RpcError(INVALID_PARAMS, "InvalidParams", {"reason": "delivery.url must be https"})
        if not self.a.allow_http_callback:
            for ai in socket.getaddrinfo(u.hostname, u.port or 443, type=socket.SOCK_STREAM):
                if not ipaddress.ip_address(ai[4][0]).is_global:  # SEP-3415 "SSRF prevention"
                    raise RpcError(INVALID_PARAMS, "InvalidParams", {"reason": "callback is not globally routable"})

    async def subscribe(self, params, p):
        t = self.type_or_404(params.get("name"))
        if p is None:  # SEP-3415 "Authentication required"
            raise RpcError(FORBIDDEN, "Forbidden", {"reason": "authentication required for webhook"})
        self.check_access(p)
        d = params.get("delivery") or {}
        if "webhook" not in t["delivery"]:
            self.violate("client-must-use-offered-delivery-mode", f"webhook on {t['name']}")
            raise RpcError(UNSUPPORTED, "Unsupported", {"feature": "deliveryMode", "value": "webhook"})
        if d.get("mode") != "webhook":
            raise RpcError(UNSUPPORTED, "Unsupported", {"feature": "deliveryMode", "value": d.get("mode")})
        args = self.validate_args(t, params.get("arguments"))
        url, secret = d.get("url"), d.get("secret")
        try:
            self.check_url(url)
        except RpcError as e:
            self.violate("client-callback-url-https", f"{url!r}: {e.data}")
            raise
        m = re.fullmatch(r"whsec_([A-Za-z0-9+/]+={0,2})", secret or "")
        try:
            raw = base64.b64decode(m.group(1), validate=True) if m else b""
        except Exception:
            raw = b""
        if not m or not 24 <= len(raw) <= 64:
            self.violate("client-secret-is-whsec-24-to-64-bytes", f"secret has {len(raw)} bytes after decode / bad shape")
            raise RpcError(INVALID_PARAMS, "InvalidParams", {"reason": "delivery.secret must be whsec_<base64 of 24..64 bytes>"})
        key = (p, url, t["name"], canon(args))
        sid = "sub_" + hashlib.sha256(canon(list(key)).encode()).hexdigest()[:16]
        t_now = time.time()
        sub = self.subs.get(key)
        if sub and sub["expires"] is not None and t_now > sub["expires"] + 0.75:
            # SEP-3415 "Subscribing": the client MUST refresh before refreshBefore.
            self.violate("client-must-refresh-before-refreshBefore",
                         f"{t['name']} refreshed {t_now - sub['expires']:.2f}s after refreshBefore")
        ttl = params.get("ttlMs", "absent")
        grant_ms = self.a.ttl_ms if ttl in ("absent", None) else min(max(int(ttl), 2000), self.a.ttl_ms)
        # "A server MUST NOT return null unless the client suggested ttlMs: null": we always grant finite.
        cur = params.get("cursor")
        if cur is not None and not t["_replay"]:
            self.violate("client-must-not-replay-from-null-cursor", f"webhook {t['name']} sent cursor {cur!r}")
        if sub and sub["expires"] is not None and t_now > sub["expires"]:
            sub = None  # lapsed: a refresh is a fresh subscribe
            self.subs.pop(key, None)
        if sub is None:
            seq, trunc = self.start_position(t, params, "webhook")
            if (p, url) not in self.verified:
                await self.verify(url, secret, p, sid)
            sub = {"key": key, "id": sid, "name": t["name"], "args": args, "url": url, "secrets": [(secret, None)],
                   "principal": p, "last": seq, "watermark": seq, "q": asyncio.Queue(), "lastDeliveryAt": None,
                   "lastError": None, "active": True, "expires": None}
            self.subs[key] = sub
            sub["task"] = asyncio.create_task(self.pump(sub))
            # replay after the cursor
            for s, o, n in list(self.log):
                if n == t["name"] and s > seq and self.matches(t, args, o):
                    sub["q"].put_nowait((s, o))
                    sub["last"] = s
            truncated = trunc
            refresh = False
        else:
            truncated, refresh = False, True
            if sub["secrets"][-1][0] != secret:  # rotation: dual-sign for a grace window
                sub["secrets"] = [(sub["secrets"][-1][0], t_now + 30), (secret, None)]
            if not sub["active"]:
                sub["active"] = True
        sub["expires"] = t_now + grant_ms / 1000.0
        out = {"resultType": "complete", "id": sid,
               "refreshBefore": datetime.fromtimestamp(sub["expires"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "cursor": self.mk_cursor(t["name"], sub["watermark"]), "truncated": truncated}
        if refresh:
            out["deliveryStatus"] = {"active": sub["active"], "lastDeliveryAt": sub["lastDeliveryAt"], "lastError": sub["lastError"]}
        return out

    async def unsubscribe(self, params, p):
        if p is None:
            raise RpcError(FORBIDDEN, "Forbidden", {"reason": "authentication required for webhook"})
        args = canon((params.get("arguments") or {}))
        url = (params.get("delivery") or {}).get("url")
        key = (p, url, params.get("name"), args)
        sub = self.subs.pop(key, None)
        if not sub:
            raise RpcError(NOT_FOUND, "NotFound", {"kind": "subscription"})
        sub["task"].cancel()
        return {"resultType": "complete"}

    def sign(self, sub_or_secrets, msg_id, ts, body: str, v1a=True, rogue=False):
        sigs = []
        for secret, until in sub_or_secrets:
            if until is not None and time.time() > until:
                continue
            sigs.append(Webhook(secret).sign(msg_id, datetime.fromtimestamp(ts, timezone.utc), body))
        if v1a and self.http_origin:  # no origin on stdio => no Server Card => no v1a
            k = self.rogue if rogue else self.key
            sigs.append("v1a," + base64.b64encode(k.sign(f"{msg_id}.{ts}.{body}".encode())).decode())
        return " ".join(sigs)

    async def post(self, url, secrets_, body_obj, msg_id, sid, rogue=False):
        body = json.dumps(body_obj, separators=(",", ":"))
        ts = int(time.time())  # regenerated per attempt
        headers = {"content-type": "application/json", "webhook-id": msg_id, "webhook-timestamp": str(ts),
                   "webhook-signature": self.sign(secrets_, msg_id, ts, body, rogue=rogue), "x-mcp-subscription-id": sid}
        return await self.http.post(url, content=body.encode(), headers=headers)

    async def verify(self, url, secret, p, sid):
        """SEP-3415 "Endpoint verification": the challenge handshake, before any delivery."""
        nonce = secrets.token_urlsafe(24)
        mid = "msg_verification_" + secrets.token_hex(8)
        try:
            r = await self.post(url, [(secret, None)], {"type": "verification", "challenge": nonce}, mid, sid)
        except httpx.ConnectError:
            raise RpcError(CALLBACK, "CallbackEndpointError", {"reason": "connection_refused"})
        except httpx.TimeoutException:
            raise RpcError(CALLBACK, "CallbackEndpointError", {"reason": "timeout"})
        if not 200 <= r.status_code < 300:
            raise RpcError(CALLBACK, "CallbackEndpointError",
                           {"reason": "http_4xx" if r.status_code < 500 else "http_5xx"})
        try:
            echoed = r.json().get("challenge", "")
        except Exception:
            echoed = ""
        if not hmac.compare_digest(str(echoed), nonce):
            raise RpcError(CALLBACK, "CallbackEndpointError", {"reason": "challenge_failed"})
        self.verified.add((p, url))

    async def pump(self, sub):
        """One serial delivery worker per subscription; retries with backoff (SEP-3415 "Delivery model")."""
        while True:
            s, o = await sub["q"].get()
            body = dict(o)
            c = self.mk_cursor(sub["name"], s)
            if self.types.get(sub["name"], {}).get("_replay"):
                body["cursor"] = c  # all earlier events are acked or abandoned: serial worker
            elif self.a.cursor_null_in_events:
                body["cursor"] = None
            mid = "msg_" + secrets.token_hex(12)
            for attempt in range(4):
                try:
                    r = await self.post(sub["url"], sub["secrets"], body, mid, sub["id"])
                    code = r.status_code
                except httpx.HTTPError:
                    code = 0
                if 200 <= code < 300:
                    self.stats["deliveries_ok"] += 1
                    sub["lastDeliveryAt"] = now_iso()
                    sub["lastError"] = None
                    break
                sub["lastError"] = "connection_refused" if code == 0 else ("http_4xx" if code < 500 else "http_5xx")
                if code in (410, 413):
                    break
                await asyncio.sleep(0.3 * (attempt + 1))
            else:
                self.stats["deliveries_failed"] += 1
            sub["watermark"] = s

    async def terminate_subs(self, name, error):
        for key, sub in list(self.subs.items()):
            if name is None or sub["name"] == name:
                self.subs.pop(key)
                sub["task"].cancel()
                with contextlib.suppress(Exception):
                    await self.post(sub["url"], sub["secrets"], {"type": "terminated", "error": error},
                                    "msg_terminated_" + secrets.token_hex(8), sub["id"])

    # ------------------------------------------------------------------------------ control
    def emit_sync(self, name, data, event_id=None):
        self.seq += 1
        occ = {"eventId": event_id or f"evt-{self.seq}", "name": name, "timestamp": now_iso(), "data": data}
        self.log.append((self.seq, occ, name))
        with open(os.path.join(self.state_dir, "events.jsonl"), "a") as f:
            f.write(json.dumps({"seq": self.seq, "occ": occ, "type": name}) + "\n")
        return self.seq, occ

    async def control(self, req):
        op = req.get("op")
        if op == "emit":
            name = req.get("type", "ticket.updated")
            if name not in self.types:
                return {"error": "unknown type"}
            s, occ = self.emit_sync(name, req.get("data", {}), req.get("event_id"))
            self.publish("event", s, occ, name)
            for sub in self.subs.values():
                if sub["name"] == name and self.matches(self.types[name], sub["args"], occ) and s > sub["last"]:
                    sub["last"] = s
                    sub["q"].put_nowait((s, occ))
            return {"seq": s, "eventId": occ["eventId"]}
        if op == "add_tool":
            n = len(self.tools)
            self.tools.append({"name": f"dyn_tool_{n}", "description": "Added at runtime.", "inputSchema": {"type": "object", "properties": {}}})
            self.publish("tools_changed")
            return {"name": f"dyn_tool_{n}"}
        if op == "add_resource":
            n = len(self.resources)
            self.resources.append({"uri": f"file:///conformance/dyn_{n}.txt", "name": f"dyn_resource_{n}"})
            self.publish("resources_changed")
            return {"name": f"dyn_resource_{n}"}
        if op == "add_type":
            self.types["late.added"] = late_type()
            self._save_meta()
            self.publish("list_changed")
            return {"ok": True}
        if op == "remove_type":
            name = req["name"]
            self.types.pop(name, None)
            self._save_meta()
            err = {"code": NOT_FOUND, "message": "NotFound", "data": {"kind": "event"}}
            self.publish("terminate", name, err)
            await self.terminate_subs(name, err)
            self.publish("list_changed")
            return {"ok": True}
        if op == "schema_change":
            name = req["name"]
            self.epoch[name] = self.epoch.get(name, 0) + 1
            props = self.types[name]["payloadSchema"].setdefault("properties", {})
            props[f"added_{self.epoch[name]}"] = {"type": "string"}
            self._save_meta()
            err = {"code": UNSUPPORTED, "message": "Unsupported",
                   "data": {"feature": "payloadSchema", "reason": "schema_changed"}}
            self.publish("terminate", name, err)
            await self.terminate_subs(name, err)
            self.publish("list_changed")
            return {"ok": True, "epoch": self.epoch[name]}
        if op == "revoke":
            self.revoked = True
            self._save_meta()
            err = {"code": FORBIDDEN, "message": "Forbidden", "data": {"reason": "Access revoked"}}
            self.publish("terminate", None, err)
            await self.terminate_subs(None, err)
            return {"ok": True}
        if op == "restore":
            self.revoked = False
            self._save_meta()
            return {"ok": True}
        if op == "bad_v1a":
            # a delivery whose HMAC is right but whose v1a is signed by a key that is not on the Server Card
            out = []
            for sub in list(self.subs.values()):
                body = {"eventId": "evt-bad-v1a", "name": sub["name"], "timestamp": now_iso(),
                        "data": req.get("data", {"ticket_id": "T-1", "summary": "forged"})}
                r = await self.post(sub["url"], sub["secrets"], body, "msg_" + secrets.token_hex(8), sub["id"], rogue=True)
                out.append(r.status_code)
            return {"statuses": out}
        if op == "state":
            since = req.get("since", 0)
            reqs = [json.loads(l) for l in open(os.path.join(self.state_dir, "requests.log"))]
            return {"subs": [{"id": s["id"], "name": s["name"], "url": s["url"], "expires": s["expires"],
                              "watermark": s["watermark"]} for s in self.subs.values()],
                    "streams": self.streams, "listeners": self.listeners, "head": self.head(), "stats": self.stats,
                    "requests": [r for r in reqs if r["ts"] > since], "now": time.time()}
        return {"error": f"unknown op {op!r}"}

    # ------------------------------------------------------------------------------ cards
    def card(self):
        x = self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        jwk = {"kty": "OKP", "crv": "Ed25519", "kid": "conformance-1",
               "x": base64.urlsafe_b64encode(x).rstrip(b"=").decode()}
        return {"$schema": "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json",
                "name": "io.example/events-conformance", "version": "1", "description": "SEP-3415 conformance server",
                "remotes": [{"type": "streamable-http", "url": f"{self.http_origin}/mcp",
                             "supportedProtocolVersions": [VERSION],
                             "extensionsInfo": {EXT: {"webhooks": {"signingKeys": {"keys": [jwk]}}}}}]}


def _ts(occ):
    return datetime.strptime(occ["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------------- HTTP app

def sse(obj):
    return ("event: message\ndata: " + json.dumps(obj) + "\n\n").encode()


def make_app(srv: Server):
    async def mcp(request: Request):
        origin = request.headers.get("origin")
        if origin and urllib.parse.urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
            return JSONResponse(err_body(None, -32000, "Forbidden origin"), status_code=403)  # 2026-07-28 "Security & Endpoint"
        try:
            msg = json.loads(await request.body())
        except Exception:
            return JSONResponse(err_body(None, -32700, "Parse error"), status_code=400)
        if "method" not in msg:
            srv.violate("client-must-not-send-responses", "JSON-RPC response sent to server")
            return JSONResponse(err_body(None, -32600, "Invalid Request"), status_code=400)
        if "id" not in msg:
            return Response(status_code=202)  # notifications: 2026-07-28 "Sending Messages"
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            kind, res = await srv.handle(msg, headers)
        except RpcError as e:
            return JSONResponse(err_body(msg["id"], e.code, e.message, e.data), status_code=e.http)
        if kind == "result":
            return JSONResponse({"jsonrpc": "2.0", "id": msg["id"], "result": res})

        async def body():
            try:
                async for frame in res:
                    yield sse(frame)
            finally:
                await res.aclose()
        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    async def control(request: Request):
        return JSONResponse(await srv.control(await request.json()))

    async def card(request: Request):
        srv.stats["card_gets"] = srv.stats.get("card_gets", 0) + 1
        return JSONResponse(srv.card())

    async def method_not_allowed(request: Request):
        return Response(status_code=405)  # 2026-07-28 "Earlier Streamable HTTP Revisions"

    async def catalog(request: Request):
        return JSONResponse({"entries": [{"identifier": "urn:air:example.com:mcp:events", "type": "application/mcp-server-card+json",
                                          "url": f"{srv.http_origin}/mcp/server-card"}]})

    return Starlette(routes=[
        Route("/mcp", mcp, methods=["POST"]),
        Route("/mcp", method_not_allowed, methods=["GET", "DELETE"]),
        Route("/mcp/server-card", card, methods=["GET"]),
        Route("/.well-known/ai-catalog.json", catalog, methods=["GET"]),
        Route("/control", control, methods=["POST"]),
    ])


# ---------------------------------------------------------------------------------- stdio

async def run_stdio(srv: Server):
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=1 << 26)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    lock = asyncio.Lock()
    tasks = {}

    async def write(obj):
        async with lock:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()

    async def run_stream(rid, gen):
        try:
            async for frame in gen:
                await write(frame)
        except asyncio.CancelledError:
            pass
        finally:
            await gen.aclose()
            tasks.pop(rid, None)

    async def one(msg):
        rid = msg["id"]
        try:
            kind, res = await srv.handle(msg, None)
        except RpcError as e:
            return await write(err_body(rid, e.code, e.message, e.data))
        if kind == "result":
            return await write({"jsonrpc": "2.0", "id": rid, "result": res})
        tasks[rid] = asyncio.create_task(run_stream(rid, res))

    while True:
        line = await reader.readline()
        if not line:
            break  # 2026-07-28 stdio "Shutdown": exit when stdin closes
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("method") == "notifications/cancelled":
            t = tasks.get((msg.get("params") or {}).get("requestId"))
            if t:
                t.cancel()  # SEP-3415 "Lifecycle: Cancellation": stop delivering, release resources
            continue
        if "method" in msg and "id" in msg:
            asyncio.create_task(one(msg))
        elif "method" not in msg:
            srv.violate("client-must-not-send-responses", "JSON-RPC response written to stdin")


async def amain(a):
    srv = Server(a)
    port = a.http if a.http else a.control_port
    srv.http_origin = f"http://127.0.0.1:{a.http}" if a.http else None
    cfg = uvicorn.Config(make_app(srv), host="127.0.0.1", port=port, log_level=os.environ.get("SRV_LOG","warning"), access_log=bool(os.environ.get("SRV_LOG")))
    web = uvicorn.Server(cfg)
    task = asyncio.create_task(web.serve())
    if a.stdio:
        await run_stdio(srv)
        web.should_exit = True
        await task
    else:
        await task


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--http", type=int, metavar="PORT", help="Streamable HTTP on 127.0.0.1:PORT (also serves /control)")
    m.add_argument("--stdio", action="store_true", help="stdio; needs --control-port")
    ap.add_argument("--control-port", type=int, help="control endpoint port in stdio mode")
    ap.add_argument("--state-dir", required=True, help="events log, meta, signing key, requests.log, violations.log")
    ap.add_argument("--no-events-capability", action="store_true",
                    help="do not advertise the events extension; any events/* request is a violation")
    ap.add_argument("--allow-http-callback", action="store_true",
                    help="LOCAL TESTING ONLY: accept http:// and loopback callback URLs. Without it a non-https "
                         "delivery.url is -32602 (SEP-3415 'TLS requirement') and non-global IPs are refused.")
    ap.add_argument("--page-size", type=int, default=2, help="events/list page size (exercises nextCursor)")
    ap.add_argument("--poll-ms", type=int, default=1000, help="nextPollMs")
    ap.add_argument("--heartbeat-ms", type=int, default=2000, help="granted heartbeatMs when the client suggests none")
    ap.add_argument("--ttl-ms", type=int, default=8000, help="maximum webhook TTL grant")
    ap.add_argument("--cursor-null-in-events", action="store_true",
                    help="write cursor:null on no-replay events instead of omitting the field")
    ap.add_argument("--emit-on-start", help="JSON list of {type,data,event_id} appended to the log before serving")
    a = ap.parse_args()
    if a.stdio and not a.control_port:
        ap.error("--stdio needs --control-port")
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
