"""Long live sessions (LNG-1, LNG-2) and large tool sets (TOOL-1), through the gateway.

Usage:
  harness_long.py long <harness>[:<mode>]    a 30-50 turn coding session with forced compaction
                                             (claude-code:uncompacted: Claude Code's own threshold)
  harness_long.py tools <scenario>           an SDK request with a large tool set (TOOL-1)
  harness_long.py mcp <harness>              a coding agent offered ~150 MCP tools (TOOL-1)
  harness_long.py cell <harness>[:<mode>]+<scenario>
                                             a short recorded session for a live.rs cell (see
                                             "recorded live.rs cells" below)

Env: VERIFY_BASE (gateway origin), VERIFY_KEY (bai_ key), VERIFY_MODEL.

Every client here talks to a recording proxy (`Recorder`) in front of the gateway, never to the
gateway itself, so the Rust trial sees every HTTP call a harness made — its path, status,
`x-beyond-request-id` and what kind of request it was — and can hold each one to exactly one
ledger row (a harness's calls are otherwise invisible; `harness.py` checks them only in aggregate).

`long` builds a fixture repo (`ledgerlib`, eight ordered steps in TASKS.md, each with its own test
file) that takes a coding agent 30-50 model turns, and turns each harness's own compaction knob
down so that auto-compaction happens at least once inside the session:

| harness     | knob                                                                       |
| ----------- | -------------------------------------------------------------------------- |
| claude-code | CLAUDE_CODE_AUTO_COMPACT_WINDOW (its auto-compact window, in tokens)       |
| codex       | model_auto_compact_token_limit in config.toml                              |
| pi          | compaction.reserveTokens in settings.json (compacts when context > window - reserve); pi checks only between prompts, so it gets the task one step per prompt |
| opencode    | none used (LNG-2 only; its sessions are long enough without compaction)    |

Compaction is witnessed twice: by the harness's own events (Claude Code's `compact_boundary`,
pi's `compaction_end` with a result, Codex's compaction items) and by the request itself (the
harness's summarization prompt in the body, or Codex's POST /v1/responses/compact).

Prints `VERIFY {json}`: ok (the repo's full test run passes and no test file changed), calls (every
proxied call), detail.
"""
import hashlib
import http.client
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import harness as H

ROOT = H.ROOT
BIN = H.BIN
BASE = H.BASE
KEY = H.KEY
MODEL = H.MODEL
KEEP = bool(os.environ.get("VERIFY_LONG_KEEP"))
# A long session takes 10-25 minutes; harness.run reads this at call time.
H.TIMEOUT = 2400

# Request bodies that are a harness asking its model to summarize the session (compaction).
COMPACTION_MARKERS = (
    "detailed summary of the conversation so far",    # Claude Code
    "CONTEXT CHECKPOINT COMPACTION",                  # Codex (local compaction)
    "Create a structured context checkpoint summary",  # pi
    "NEW conversation messages to incorporate into the existing summary",  # pi (update)
)
# Paths the gateway bills (one ai.usage row per call) and the free ones (no row).
BILLED = ("/v1/messages", "/v1/chat/completions", "/v1/responses", "/v1/responses/compact")
FREE = ("/v1/messages/count_tokens", "/v1/responses/input_tokens", "/v1/models")


# --- the recording proxy -------------------------------------------------------------------------

class Recorder:
    """A streaming HTTP/1.1 reverse proxy to the gateway that records every call."""

    def __init__(self, upstream):
        u = urllib.parse.urlparse(upstream)
        # An https upstream is a provider called directly (a baseline, never billed by us).
        self.tls = u.scheme == "https"
        self.host, self.port = u.hostname, u.port or (443 if self.tls else 80)
        self.calls = []
        self.progress = {}
        self.lock = threading.Lock()
        rec = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _body(self):
                if self.headers.get("transfer-encoding", "").lower() == "chunked":
                    out = bytearray()
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                                pass
                            return bytes(out)
                        out += self.rfile.read(size)
                        self.rfile.readline()
                n = int(self.headers.get("content-length") or 0)
                return self.rfile.read(n) if n else b""

            def _proxy(self):
                body = self._body()
                t0 = time.time()
                call = {"method": self.command, "path": self.path.split("?")[0], "t0": t0,
                        "req_bytes": len(body), **summarize(self.path, body)}
                if self.headers.get("upgrade"):
                    call.update(status=501, why="websocket upgrade not proxied")
                    rec.add(call)
                    self.send_response(501)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                hdrs = {k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "content-length", "transfer-encoding", "connection",
                                             "keep-alive", "accept-encoding")}
                hdrs["content-length"] = str(len(body))
                conn = (http.client.HTTPSConnection if rec.tls else http.client.HTTPConnection)(
                    rec.host, rec.port, timeout=900)
                try:
                    conn.request(self.command, self.path, body=body, headers=hdrs)
                    resp = conn.getresponse()
                except Exception as e:  # noqa: BLE001
                    call.update(status=502, why=f"proxy upstream: {e}")
                    rec.add(call)
                    self.send_response(502)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                call.update(status=resp.status, request_id=resp.getheader("x-beyond-request-id"))
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "connection", "content-length", "keep-alive"):
                        self.send_header(k, v)
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                head, size, tail_buf = b"", 0, b""
                scan = UsageScan(resp.getheader("content-type") or "")
                deltas = 0
                # Bytes relayed so far on each response still in flight (a cell that cuts a
                # client off mid-stream waits on this).
                rec.progress[id(call)] = (call, 0)
                try:
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        size += len(chunk)
                        rec.progress[id(call)] = (call, size)
                        scan.feed(chunk)
                        # When the stream's first byte and first content delta reached the client,
                        # and how many delta events it carried (S1).
                        now = time.time()
                        call.setdefault("t_first_byte", now)
                        n = (chunk.count(b"content_block_delta") + chunk.count(b'"delta":{"content"')
                             + chunk.count(b"output_text.delta"))
                        if n:
                            call.setdefault("t_first_delta", now)
                            deltas += n
                        if len(head) < 4096 and resp.status >= 400:
                            head += chunk[:4096]
                        tail_buf = (tail_buf + chunk)[-3000:]
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    call["client_gone"] = True
                finally:
                    conn.close()
                    rec.progress.pop(id(call), None)
                    call.update(t1=time.time(), resp_bytes=size, deltas=deltas)
                    if resp.status < 300 and (u := scan.usage()) is not None:
                        call["seen_usage"] = u
                    if head:
                        call["error_body"] = head.decode("utf-8", "replace")[:1500]
                    call["resp_tail"] = tail_buf.decode("utf-8", "replace")
                    rec.add(call)

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _proxy

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add(self, call):
        with self.lock:
            self.calls.append(call)

    def stop(self):
        """The calls in the order they were made. Each response's tail is kept only on the last
        three (where a harness that died mid-stream shows why), or all with VERIFY_LONG_KEEP."""
        self.server.shutdown()
        with self.lock:
            calls = sorted(self.calls, key=lambda c: c["t0"])
        if not KEEP:
            for c in calls[:-3]:
                c.pop("resp_tail", None)
        return calls


class UsageScan:
    """The usage a response showed its client, read from the bytes the recorder relays: the final
    usage of an SSE stream (Messages `message_start` merged with `message_delta`, a Chat chunk's
    `usage`, a Responses terminal event) or of a JSON body. Normalized as a probe's are (live.rs
    `call_problems`): input_total includes cached tokens, cache_read is the cache-read count."""

    def __init__(self, content_type):
        self.sse = "text/event-stream" in content_type
        self.buf = b""
        self.anthropic = {}
        self.openai = None
        self.kind = None

    def feed(self, chunk):
        self.buf += chunk
        if not self.sse:
            return
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            if line.startswith(b"data:"):
                self.event(line[5:].strip())

    def event(self, data):
        if not data or data == b"[DONE]":
            return
        try:
            v = json.loads(data)
        except ValueError:
            return
        if not isinstance(v, dict):
            return
        t = v.get("type")
        if t == "message_start":
            self.kind = "messages"
            self.anthropic.update({k: x for k, x in ((v.get("message") or {}).get("usage") or {}).items()
                                   if x is not None})
        elif t == "message_delta":
            self.kind = "messages"
            self.anthropic.update({k: x for k, x in (v.get("usage") or {}).items() if x is not None})
        elif isinstance(t, str) and t.startswith("response.") and isinstance(v.get("response"), dict):
            if v["response"].get("usage"):
                self.kind, self.openai = "responses", v["response"]["usage"]
        elif t == "message":
            self.kind, self.anthropic = "messages", dict(v.get("usage") or {})
        elif v.get("object") == "response" and v.get("usage"):
            self.kind, self.openai = "responses", v["usage"]
        elif v.get("usage") and "choices" in v:
            self.kind, self.openai = "chat", v["usage"]

    def usage(self):
        if not self.sse:
            self.event(self.buf.strip())

        def n(d, k):
            return (d or {}).get(k) or 0

        if self.kind == "messages":
            u = self.anthropic
            cr, cw = n(u, "cache_read_input_tokens"), n(u, "cache_creation_input_tokens")
            return {"input_total": n(u, "input_tokens") + cr + cw, "output": n(u, "output_tokens"), "cache_read": cr}
        if self.kind == "chat":
            u = self.openai
            return {"input_total": n(u, "prompt_tokens"), "output": n(u, "completion_tokens"),
                    "cache_read": n(u.get("prompt_tokens_details"), "cached_tokens")}
        if self.kind == "responses":
            u = self.openai
            return {"input_total": n(u, "input_tokens"), "output": n(u, "output_tokens"),
                    "cache_read": n(u.get("input_tokens_details"), "cached_tokens")}
        return None


def summarize(path, body):
    """What kind of request this is, from its body: model, tool count, compaction."""
    out = {}
    try:
        v = json.loads(body) if body else {}
    except ValueError:
        return {"body": "unparsed"}
    if not isinstance(v, dict):
        return out
    out["model"] = v.get("model")
    out["stream"] = v.get("stream")
    tools = v.get("tools") or []
    out["tools"] = len(tools)
    out["namespace_members"] = sum(len(t.get("tools") or []) for t in tools
                                   if isinstance(t, dict) and t.get("type") == "namespace")
    msgs = v.get("messages") if isinstance(v.get("messages"), list) else v.get("input")
    out["items"] = len(msgs) if isinstance(msgs, list) else 1
    # Prompt caching the client asked for itself, and thinking: whether the request enables it, and
    # how many signed thinking blocks / encrypted reasoning items its history replays (T2, K1).
    out["cache_control"] = body.count(b'"cache_control"')
    th = v.get("thinking")
    out["thinking"] = bool((isinstance(th, dict) and th.get("type") in ("enabled", "adaptive"))
                           or v.get("reasoning") or v.get("reasoning_effort"))
    replayed = 0
    for m in msgs if isinstance(msgs, list) else []:
        if not isinstance(m, dict):
            continue
        if m.get("type") == "reasoning" and m.get("encrypted_content"):
            replayed += 1
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            replayed += sum(1 for b in m["content"] if isinstance(b, dict) and (
                (b.get("type") == "thinking" and b.get("signature")) or b.get("type") == "redacted_thinking"))
    out["replayed_thinking"] = replayed
    # The history a turn replays (S2): every role it names, its assistant messages, and those that
    # carry neither text nor a tool call (an accumulator that lost the message). Images sent (T3).
    roles, turns, empty, images = set(), 0, 0, 0
    for m in msgs if isinstance(msgs, list) else []:
        if not isinstance(m, dict):
            continue
        if isinstance(m.get("role"), str):
            roles.add(m["role"])
        content = m.get("content")
        if m.get("role") == "assistant":
            turns += 1
            if not content and not m.get("tool_calls"):
                empty += 1
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and part.get("type") in ("image", "image_url", "input_image"):
                images += 1
    out.update(roles=sorted(roles), assistant_turns=turns, assistant_empty=empty, images=images)
    # Tool round trips the history replays (T1): results whose id names a tool call the same history
    # carries. Chat: an assistant's `tool_calls` and a `tool` message's `tool_call_id`. Messages:
    # `tool_use` and `tool_result` blocks. Responses: `*_call` items and `*_call_output` items by
    # `call_id` (function, custom and shell tools alike).
    call_ids, result_ids = set(), []
    for m in msgs if isinstance(msgs, list) else []:
        if not isinstance(m, dict):
            continue
        for tc in m.get("tool_calls") or [] if m.get("role") == "assistant" else []:
            if isinstance(tc, dict):
                call_ids.add(tc.get("id"))
        if m.get("role") == "tool":
            result_ids.append(m.get("tool_call_id"))
        t = m.get("type")
        if isinstance(t, str) and t.endswith("_call_output"):
            result_ids.append(m.get("call_id"))
        elif isinstance(t, str) and t.endswith("_call") and m.get("call_id"):
            call_ids.add(m.get("call_id"))
        for b in m.get("content") if isinstance(m.get("content"), list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                call_ids.add(b.get("id"))
            elif isinstance(b, dict) and b.get("type") == "tool_result":
                result_ids.append(b.get("tool_use_id"))
    call_ids.discard(None)
    out["tool_round_trips"] = sum(1 for r in result_ids if r in call_ids)
    # Only the last message asks for the summary: a summary kept in the history afterwards may
    # quote the prompt (gpt-5-mini's do), and that turn isn't a compaction.
    # (Claude Code can follow it with a role "system" note, so: the last user message.)
    last = msgs
    if isinstance(msgs, list):
        users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        last = users[-1] if users else (msgs[-1] if msgs else "")
    text = json.dumps(last) if not isinstance(last, str) else last
    out["compaction"] = path.split("?")[0].endswith("/responses/compact") or any(
        m in text for m in COMPACTION_MARKERS)
    if KEEP and not out["compaction"]:
        whole = body.decode("utf-8", "replace")
        out["marker_elsewhere"] = any(m in whole for m in COMPACTION_MARKERS)
        if out["marker_elsewhere"]:
            DUMPS.append(whole)
    return out


DUMPS = []  # with VERIFY_LONG_KEEP: bodies quoting a compaction prompt outside the last message


# --- the fixture repo ----------------------------------------------------------------------------

CATEGORIES = ["groceries", "rent", "salary", "utilities", "dining", "transport", "books",
              "insurance", "refund", "gifts", "travel", "health"]
INCOME = {"salary", "refund", "gifts"}


def gen_rows(seed=20260930, n=420):
    """The fixture's transactions, and the CSV text with its quirks."""
    rnd = random.Random(seed)
    rows, lines = [], ["date,description,amount,category"]
    day = date(2025, 1, 3)
    for i in range(n):
        day += timedelta(days=rnd.choice([0, 0, 1, 1, 2]))
        cat = rnd.choice(CATEGORIES)
        cents = rnd.randint(150, 250000) if cat in INCOME else -rnd.randint(150, 90000)
        desc = rnd.choice(["Payment", "Card", "Transfer", "Direct debit", "Standing order"])
        desc += f" {i:04d}"
        if i % 7 == 0:
            desc += ", ref " + str(rnd.randint(1000, 9999))
        rows.append((day, desc, cents, cat))
        ds = day.isoformat() if i % 3 else day.strftime("%d/%m/%Y")
        mag = abs(cents)
        amt = f"{mag // 100:,}.{mag % 100:02d}"
        if cents < 0:
            amt = f"({amt})" if i % 5 == 0 else f"-{amt}"
        q = lambda s: f'"{s}"' if "," in s else s  # noqa: E731
        lines.append(f"{ds},{q(desc)},{q(amt)},{cat}")
        if i % 50 == 25:
            lines.append(f"# checkpoint {i}: reconciled with bank statement")
        if i % 80 == 40:
            lines.append("")
    return rows, "\n".join(lines) + "\n"


def fmt(c):
    s = f"{abs(c) // 100:,}.{abs(c) % 100:02d}"
    return "-" + s if c < 0 else s


def fixture():
    """`{relative path: text}` for the repo, and the expected numbers baked into its tests."""
    rows, csv_text = gen_rows()
    total = sum(r[2] for r in rows)
    cats = sorted({r[3] for r in rows})
    months = {}
    for d, _, c, _ in rows:
        m = months.setdefault(d.strftime("%Y-%m"), [0, 0])
        if c >= 0:
            m[0] += c
        else:
            m[1] += -c
    summary = [(m, i, e, i - e) for m, (i, e) in sorted(months.items())]
    spend = {}
    for _, _, c, cat in rows:
        if c < 0:
            spend[cat] = spend.get(cat, 0) - c
    top3 = sorted(spend.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    cli_out = "".join(f"{m} income={fmt(i)} expense={fmt(e)} net={fmt(n)}\n" for m, i, e, n in summary)
    cli_out += f"TOTAL net={fmt(total)}\n"

    files = {}
    files["README.md"] = "# ledgerlib\n\nA tiny double-entry ledger. See TASKS.md for the work to do.\n"
    files["ledgerlib/__init__.py"] = '"""ledgerlib: a tiny double-entry ledger."""\n'
    files["ledgerlib/util.py"] = '''"""Small shared helpers."""
from datetime import date


def parse_date(s):
    """Parse 'YYYY-MM-DD' or 'DD/MM/YYYY' into a datetime.date. Raise ValueError otherwise."""
    s = s.strip()
    if "-" in s:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    if "/" in s:
        d, m, y = s.split("/")
        return date(int(y), int(d), int(m))
    raise ValueError(f"unrecognized date: {s!r}")
'''
    stub = '"""{doc}"""\n\n# TODO: implement (see TASKS.md, step {n}).\n'
    files["ledgerlib/money.py"] = stub.format(doc="Amounts as integer cents.", n=2)
    files["ledgerlib/accounts.py"] = stub.format(doc="Accounts.", n=3)
    files["ledgerlib/ledger.py"] = stub.format(doc="The double-entry ledger.", n=4)
    files["ledgerlib/importer.py"] = stub.format(doc="CSV import.", n=5)
    files["ledgerlib/report.py"] = stub.format(doc="Reports over imported rows.", n=6)
    files["ledgerlib/cli.py"] = stub.format(doc="Command-line interface.", n=8)
    files["data/transactions.csv"] = csv_text
    files["run_tests.py"] = '''"""Run the tests: `python3 run_tests.py` (all) or `python3 run_tests.py step3` (one step)."""
import sys
import unittest

pattern = f"test_{sys.argv[1]}_*.py" if len(sys.argv) > 1 else "test_*.py"
suite = unittest.defaultTestLoader.discover("tests", pattern=pattern, top_level_dir=".")
if suite.countTestCases() == 0:
    print(f"no tests match {pattern}")
    sys.exit(2)
result = unittest.TextTestRunner(verbosity=1).run(suite)
if result.wasSuccessful():
    print("ALL TESTS PASS")
    sys.exit(0)
sys.exit(1)
'''
    files["tests/__init__.py"] = ""
    files["tests/test_step1_util.py"] = '''import unittest
from datetime import date

from ledgerlib.util import parse_date


class TestParseDate(unittest.TestCase):
    def test_iso(self):
        self.assertEqual(parse_date("2024-03-05"), date(2024, 3, 5))

    def test_day_first(self):
        self.assertEqual(parse_date("05/03/2024"), date(2024, 3, 5))
        self.assertEqual(parse_date(" 31/12/2025 "), date(2025, 12, 31))

    def test_garbage(self):
        for bad in ("yesterday", "2024.03.05", ""):
            with self.assertRaises(ValueError):
                parse_date(bad)
'''
    files["tests/test_step2_money.py"] = '''import unittest

from ledgerlib.money import format_cents, parse_amount


class TestMoney(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_amount("12.34"), 1234)
        self.assertEqual(parse_amount("-5"), -500)
        self.assertEqual(parse_amount("1,234.5"), 123450)
        self.assertEqual(parse_amount("(7.25)"), -725)
        self.assertEqual(parse_amount(" 3.10 "), 310)
        self.assertEqual(parse_amount("-1,000,000.01"), -100000001)

    def test_parse_rejects(self):
        for bad in ("abc", "1.234", "", "12..3", "(5"):
            with self.assertRaises(ValueError):
                parse_amount(bad)

    def test_format(self):
        self.assertEqual(format_cents(123450), "1,234.50")
        self.assertEqual(format_cents(-725), "-7.25")
        self.assertEqual(format_cents(0), "0.00")
        self.assertEqual(format_cents(5), "0.05")
        self.assertEqual(format_cents(-100000001), "-1,000,000.01")
'''
    files["tests/test_step3_accounts.py"] = '''import unittest

from ledgerlib.accounts import KINDS, Account, InsufficientFunds


class TestAccounts(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(KINDS, ("asset", "liability", "income", "expense", "equity"))
        with self.assertRaises(ValueError):
            Account("x", "stuff")

    def test_balance(self):
        a = Account("bank", "asset")
        self.assertEqual((a.name, a.kind, a.balance), ("bank", "asset", 0))
        a.apply(500)
        a.apply(-200)
        self.assertEqual(a.balance, 300)

    def test_normal_sign(self):
        self.assertEqual(Account("a", "asset").normal_sign, 1)
        self.assertEqual(Account("e", "expense").normal_sign, 1)
        for k in ("liability", "income", "equity"):
            self.assertEqual(Account("x", k).normal_sign, -1)

    def test_withdraw(self):
        a = Account("bank", "asset")
        a.apply(1000)
        a.withdraw(400)
        self.assertEqual(a.balance, 600)
        with self.assertRaises(InsufficientFunds):
            a.withdraw(601)
        self.assertEqual(a.balance, 600)
'''
    files["tests/test_step4_ledger.py"] = '''import unittest
from datetime import date

from ledgerlib.ledger import Ledger, Txn


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.l = Ledger()
        self.l.open("bank", "asset")
        self.l.open("food", "expense")

    def test_open_twice(self):
        with self.assertRaises(ValueError):
            self.l.open("bank", "asset")

    def test_post(self):
        t = self.l.post(date(2025, 1, 1), "lunch", [("bank", -1200), ("food", 1200)])
        self.assertIsInstance(t, Txn)
        self.assertEqual((t.date, t.description, t.entries),
                         (date(2025, 1, 1), "lunch", [("bank", -1200), ("food", 1200)]))
        self.assertEqual(self.l.balance("bank"), -1200)
        self.assertEqual(self.l.balance("food"), 1200)
        self.assertEqual(self.l.transactions, [t])

    def test_unbalanced(self):
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            self.l.post(date(2025, 1, 1), "x", [("bank", -1), ("food", 2)])
        self.assertEqual(self.l.transactions, [])

    def test_unknown_account(self):
        with self.assertRaises(KeyError):
            self.l.post(date(2025, 1, 1), "x", [("bank", -1), ("nope", 1)])
        self.assertEqual(self.l.balance("bank"), 0)

    def test_trial_balance(self):
        self.l.open("pay", "income")
        self.l.post(date(2025, 1, 1), "pay", [("bank", 5000), ("pay", -5000)])
        self.l.post(date(2025, 1, 2), "eat", [("bank", -700), ("food", 700)])
        tb = self.l.trial_balance()
        self.assertEqual(tb, {"bank": 4300, "food": 700, "pay": -5000})
        self.assertEqual(sum(tb.values()), 0)
'''
    files["tests/test_step5_importer.py"] = f'''import unittest
from datetime import date

from ledgerlib.importer import Row, import_into, load_csv
from ledgerlib.ledger import Ledger


class TestImporter(unittest.TestCase):
    def setUp(self):
        self.rows = load_csv("data/transactions.csv")

    def test_rows(self):
        self.assertEqual(len(self.rows), {len(rows)})
        self.assertIsInstance(self.rows[0], Row)
        first = self.rows[0]
        self.assertEqual(first.date, date({rows[0][0].year}, {rows[0][0].month}, {rows[0][0].day}))
        self.assertEqual(first.description, {rows[0][1]!r})
        self.assertEqual(first.amount, {rows[0][2]})
        self.assertEqual(first.category, {rows[0][3]!r})
        self.assertEqual(self.rows[7].description, {rows[7][1]!r})

    def test_totals(self):
        self.assertEqual(sum(r.amount for r in self.rows), {total})
        self.assertEqual(sorted({{r.category for r in self.rows}}), {cats!r})

    def test_import(self):
        led = Ledger()
        import_into(led, self.rows)
        self.assertEqual(led.balance("bank"), {total})
        self.assertEqual(len(led.transactions), {len(rows)})
        self.assertEqual(sum(led.trial_balance().values()), 0)
        self.assertEqual(led.accounts["salary"].kind, "income")
        self.assertEqual(led.accounts["rent"].kind, "expense")
'''
    files["tests/test_step6_report.py"] = f'''import unittest

from ledgerlib.importer import load_csv
from ledgerlib.report import monthly_summary, top_categories


class TestReport(unittest.TestCase):
    def setUp(self):
        self.rows = load_csv("data/transactions.csv")

    def test_monthly(self):
        got = monthly_summary(self.rows)
        self.assertEqual(got, {summary!r})

    def test_top(self):
        self.assertEqual(top_categories(self.rows, 3), {top3!r})
        self.assertEqual(top_categories(self.rows, 0), [])
'''
    files["tests/test_step7_refactor.py"] = '''import inspect
import unittest
import warnings
from datetime import date

import ledgerlib.importer as importer
from ledgerlib.ledger import Ledger


class TestRefactor(unittest.TestCase):
    def setUp(self):
        self.l = Ledger()
        self.l.open("bank", "asset")
        self.l.open("food", "expense")

    def test_record(self):
        t = self.l.record(date(2025, 1, 1), "x", [("bank", -5), ("food", 5)])
        self.assertEqual(self.l.transactions, [t])

    def test_post_is_deprecated_alias(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.l.post(date(2025, 1, 1), "x", [("bank", -5), ("food", 5)])
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertEqual(self.l.balance("food"), 5)

    def test_importer_uses_record(self):
        src = inspect.getsource(importer)
        self.assertIn(".record(", src)
        self.assertNotIn(".post(", src)
'''
    files["tests/test_step8_cli.py"] = f'''import subprocess
import sys
import unittest

EXPECTED = {cli_out!r}


class TestCli(unittest.TestCase):
    def test_summary(self):
        p = subprocess.run([sys.executable, "-m", "ledgerlib.cli", "summary", "data/transactions.csv"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout, EXPECTED)

    def test_usage(self):
        p = subprocess.run([sys.executable, "-m", "ledgerlib.cli"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("usage", p.stderr.lower())
'''
    files["TASKS.md"] = TASKS
    return files


TASKS = """# Tasks

Eight steps, in order. Each step has its own tests in `tests/test_stepN_*.py`; run one step's tests
with `python3 run_tests.py stepN` and all of them with `python3 run_tests.py`. Never edit anything
under `tests/`, nor `run_tests.py`.

## Step 1: fix parse_date

`ledgerlib/util.py` has `parse_date`, which accepts `YYYY-MM-DD` and day-first `DD/MM/YYYY` and must
raise `ValueError` for anything else. It has a bug. Fix it.

## Step 2: money

In `ledgerlib/money.py`, write `parse_amount(s) -> int` (cents) and `format_cents(c) -> str`.
`parse_amount` strips whitespace, accepts an optional leading `-`, accounting negatives in
parentheses `(7.25)`, thousands separators, and at most two decimals; anything else is a
`ValueError`. `format_cents` renders thousands separators and exactly two decimals.

## Step 3: accounts

In `ledgerlib/accounts.py`: `KINDS` (the tuple of the five account kinds), `InsufficientFunds`
(an exception), and `Account(name, kind)` with `balance` (int cents, starts at 0), `apply(delta)`,
`normal_sign` (+1 for asset and expense, -1 otherwise) and `withdraw(cents)` (raises
`InsufficientFunds`, changing nothing, when the balance is too low).

## Step 4: ledger

In `ledgerlib/ledger.py`: `Txn` (a namedtuple of `date, description, entries`) and `Ledger` with
`accounts` (name -> Account), `transactions` (a list), `open(name, kind)` (ValueError if it
exists), `post(date, description, entries)` where entries are `(account name, cents)` pairs that
must sum to zero (else `ValueError("unbalanced ...")`; an unknown account is a `KeyError`; nothing
changes on error) and which returns the Txn, `balance(name)` and `trial_balance()` (name ->
balance).

## Step 5: CSV import

In `ledgerlib/importer.py`: `Row` (a namedtuple of `date, description, amount, category`),
`load_csv(path)` and `import_into(ledger, rows, bank="bank")`. Read `data/transactions.csv` first:
it has a header, comment lines starting with `#`, blank lines, quoted fields that contain commas,
and both date formats. Use `parse_date` and `parse_amount`. `import_into` opens the bank account
(asset) and each category account (income if the row's amount is positive, else expense) as
needed, and posts `[(bank, amount), (category, -amount)]` for each row.

## Step 6: reports

In `ledgerlib/report.py`: `monthly_summary(rows)` -> a list of `(YYYY-MM, income, expense, net)`
tuples sorted by month (income: the sum of positive amounts; expense: the sum of the magnitudes of
negative amounts; net = income - expense), and `top_categories(rows, n)` -> the `n` categories with
the most spending as `(category, total spent)` pairs, largest first, ties by name.

## Step 7: refactor

Rename `Ledger.post` to `Ledger.record`. Keep `post` as a deprecated alias that emits a
`DeprecationWarning` and delegates to `record`. Update the importer to call `record`.

## Step 8: CLI

`python3 -m ledgerlib.cli summary <csv>` prints one line per month,
`YYYY-MM income=<x> expense=<y> net=<z>` (amounts formatted with `format_cents`), then
`TOTAL net=<z>`. With no or wrong arguments it prints a usage line to stderr and exits 2.
"""

STEPS = 8
INTRO = ("This repository is a small Python ledger library with eight ordered steps of work in TASKS.md. "
         "Never modify anything under tests/ or run_tests.py.")
ONE_STEP = ("Do step {n} of TASKS.md now. Read tests/test_step{n}_*.py first, implement the step, then run "
            "`python3 run_tests.py step{n}` and keep fixing until it passes. Do only step {n}.")
WHOLE = (INTRO + " Do the steps strictly in order, one at a time: for each step, read its test file, "
         "implement it, run `python3 run_tests.py stepN` and fix until it passes before you start the next "
         "step. Keep going on your own, without stopping to report or ask, until all eight steps pass; then run "
         "`python3 run_tests.py` and confirm it prints ALL TESTS PASS.")


def write_repo(work, files):
    for rel, text in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return {rel: hashlib.sha256(t.encode()).hexdigest() for rel, t in files.items()
            if rel.startswith("tests/") or rel == "run_tests.py"}


def check_repo(work, hashes):
    test = subprocess.run([sys.executable, "run_tests.py"], cwd=work, capture_output=True, text=True,
                          timeout=120)
    changed = [rel for rel, h in hashes.items()
               if not (work / rel).exists() or hashlib.sha256((work / rel).read_bytes()).hexdigest() != h]
    steps = {}
    for n in range(1, STEPS + 1):
        r = subprocess.run([sys.executable, "run_tests.py", f"step{n}"], cwd=work, capture_output=True,
                           text=True, timeout=120)
        steps[n] = r.returncode == 0
    ok = test.returncode == 0 and "ALL TESTS PASS" in test.stdout and not changed
    return ok, {"test_rc": test.returncode, "tests_changed": changed, "steps_passing": steps,
                "test_tail": (test.stdout + test.stderr)[-600:]}


# --- harness environments ------------------------------------------------------------------------

def clean_env(home):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI", "CLAUDE", "CODEX"))}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local/share"),
               XDG_CACHE_HOME=str(home / ".cache"), XDG_STATE_HOME=str(home / ".local/state"))
    return env


def claude_env(env, base):
    env.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_API_KEY=KEY, DISABLE_TELEMETRY="1",
               DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL, ANTHROPIC_SMALL_FAST_MODEL=MODEL)


def codex_home(home, base, extra=""):
    ch = home / ".codex"
    ch.mkdir(exist_ok=True)
    (ch / "config.toml").write_text(
        f'model = "{MODEL}"\nmodel_provider = "beyond"\n{extra}\n[model_providers.beyond]\nname = "Beyond"\n'
        f'base_url = "{base}/v1"\nenv_key = "BEYOND_API_KEY"\nwire_api = "responses"\n')
    return ch


def pi_setup(home, base, mode, models, settings=None):
    api = {"chat": "openai-completions", "messages": "anthropic-messages", "responses": "openai-responses"}[mode]
    agent_dir = home / ".pi-agent"
    agent_dir.mkdir(exist_ok=True)
    url = base if mode == "messages" else f"{base}/v1"
    (agent_dir / "models.json").write_text(json.dumps(
        {"providers": {"beyond": {"baseUrl": url, "api": api, "apiKey": KEY, "models": H.pi_models(models)}}}))
    if settings:
        (agent_dir / "settings.json").write_text(json.dumps(settings))
    return agent_dir


def opencode_setup(work, base, models):
    cfg = {"$schema": "https://opencode.ai/config.json",
           "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
           "provider": {"beyond": {"npm": "@ai-sdk/openai-compatible", "name": "Beyond",
                                   "options": {"baseURL": f"{base}/v1", "apiKey": KEY},
                                   "models": H.opencode_models(models)}}}
    (work / "opencode.json").write_text(json.dumps(cfg))


# Compaction thresholds, in tokens, each derived from the prompt sizes the recorder sees (fresh +
# cache tokens per call, printed by long_live.rs as `prompts`): above the context a compaction
# leaves (the harness's fixed prompt, its system prompt and tool definitions, plus the summary),
# so it doesn't compact every turn, and below the prompt a session typically ends on, so it
# compacts at least once before the task is done. Measured on the pinned versions.
# Claude Code's window can't go below 100k (it clamps); its threshold is min(window - 13k,
# window * CLAUDE_AUTOCOMPACT_PCT_OVERRIDE%) of an 80k effective window (its debug log:
# `effectiveWindow=80000`), so the percentage is the knob that bites. Its count is the last
# prompt plus that answer's output (compacting at 28.3k and 29.7k after prompts of 26.8k + 0.5k
# out and 21.7k + 6.6k out). On a Claude row, 50% = 40k: the fixed prompt is ~29k, the session
# peaks ~50k. On a translated GPT row (gpt-5.4-mini, 2026-10-01): the fixed prompt is 12.8k and
# the summary 7.7-9.1k (so ~21k after a compaction), and uncompacted sessions end at 25.8k
# (28 calls) and ~27k (29 calls), growing ~0.5k a call. 35% = 28k sat above that end: two of
# three sessions never compacted. 30% = 24k is crossed ~5 calls before the end. The band between
# ~21k and ~26k is narrow, so this route is marginal both ways: at 30% one session compacted 3
# times in 39 calls and passed, another 8 times in ~100 calls and was still on the last step after
# 16 minutes (stopped by hand; each summary loses some of the task).
# Codex compacts above model_auto_compact_token_limit: its fixed prompt is 10.0k, it resumes at
# ~11k after a compaction, and it passes 20k around call 21 of ~38 (2026-10-01: once, at 20.6k).
# pi compacts above window - reserveTokens = 16k: its fixed prompt is 2.0k, it resumes at 1.6-9k,
# and a session grows past 16k twice in ~70 calls (2026-10-01, on claude-haiku-4-5 over Chat).
CLAUDE_COMPACT_WINDOW = 100000
CLAUDE_COMPACT_PCT = int(os.environ.get("VERIFY_LONG_CC_PCT", "50" if MODEL.startswith("claude") else "30"))
CODEX_COMPACT_LIMIT = int(os.environ.get("VERIFY_LONG_CODEX_LIMIT", "20000"))
PI_COMPACT_AT = int(os.environ.get("VERIFY_LONG_PI_AT", "16000"))


def long_session(spec):
    harness, _, mode = spec.partition(":")
    home = pathlib.Path(tempfile.mkdtemp(prefix=f"verify-long-{harness}-"))
    work = home / "repo"
    work.mkdir()
    hashes = write_repo(work, fixture())
    rec = Recorder(BASE)
    env = clean_env(home)
    models = H.catalog()
    card = {m["id"]: m for m in models}[MODEL]
    detail = {"harness": harness, "mode": mode or None, "pricing": H.prices(card),
              "context_window": card["context_window"]}
    events = {"compactions": 0, "types": {}}
    prompts = [INTRO + " " + ONE_STEP.format(n=1)] + [ONE_STEP.format(n=n) for n in range(2, STEPS + 1)]
    prompts.append("Run `python3 run_tests.py` and confirm every step passes; fix anything that fails.")

    if harness == "claude-code":
        claude_env(env, rec.base)
        if mode != "uncompacted":
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(CLAUDE_COMPACT_WINDOW)
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(CLAUDE_COMPACT_PCT)
            detail["knob"] = (f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={CLAUDE_COMPACT_WINDOW} "
                              f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={CLAUDE_COMPACT_PCT}")
        cmds = [[str(BIN / "claude"), "-p", WHOLE, "--model", MODEL, "--dangerously-skip-permissions",
                 "--output-format", "stream-json", "--verbose", "--max-turns", "120"]]
        if KEEP:
            # Its debug log has a line per turn with the count the threshold is held to
            # ("autocompact: tokens=N level=... effectiveWindow=W").
            cmds[0] += ["--debug-file", str(home / "claude-debug.txt")]
    elif harness == "codex":
        ch = codex_home(home, rec.base, f"model_auto_compact_token_limit = {CODEX_COMPACT_LIMIT}\n")
        env.update(CODEX_HOME=str(ch), BEYOND_API_KEY=KEY)
        detail["knob"] = f"model_auto_compact_token_limit={CODEX_COMPACT_LIMIT}"
        # One step per prompt, the session resumed each time (`exec resume --last`): given the
        # whole task at once Codex batches steps and finishes in ~13 turns.
        base_cmd = [str(BIN / "codex"), "exec", "--json", "--skip-git-repo-check",
                    "--dangerously-bypass-approvals-and-sandbox", "-m", MODEL]
        cmds = [base_cmd + [prompts[0]]] + [base_cmd + ["resume", "--last", p] for p in prompts[1:]]
    elif harness == "pi":
        reserve = max(card["context_window"] - PI_COMPACT_AT, 0)
        # keepRecentTokens below the threshold, so a compaction actually cuts something.
        settings = {"compaction": {"enabled": True, "reserveTokens": reserve,
                                   "keepRecentTokens": PI_COMPACT_AT // 4}}
        agent_dir = pi_setup(home, rec.base, mode, models, settings)
        env.update(PI_CODING_AGENT_DIR=str(agent_dir))
        detail["knob"] = f"compaction.reserveTokens={reserve} (compacts above {PI_COMPACT_AT})"
        # pi asks a reasoning row for thinking.type "enabled", which Claude 5.x refuses ("use
        # adaptive"), and a native Messages relay passes it as-is: thinking off there.
        thinking = ["--thinking", "off"] if re.match(r"claude-(sonnet|opus|fable)-5", MODEL) else []
        cmds = [[str(BIN / "pi"), "-p", "--mode", "json", "--provider", "beyond", "--model", MODEL,
                 *thinking, "--no-session", *prompts]]
    elif harness == "opencode":
        opencode_setup(work, rec.base, models)
        cmds = [[str(BIN / "opencode"), "run", "--print-logs", "--format", "json", "--title", "ledger",
                 "-m", f"beyond/{MODEL}", WHOLE]]
    else:
        return False, rec.stop(), {"why": f"unknown harness {harness}"}

    t0 = time.time()
    code, out, err = 0, "", ""
    outs, errs = [], []
    for cmd in cmds:
        code, o, e = H.run(cmd, work, env)  # -9 at H.TIMEOUT or the cell's deadline
        outs.append(o)
        errs.append(e)
        if code != 0:
            break
    out, err = "\n".join(outs), "\n".join(errs)
    detail["seconds"] = round(time.time() - t0)

    for e in H.json_lines(out):
        t = e.get("type")
        sub = e.get("subtype")
        key = f"{t}:{sub}" if sub else str(t)
        events["types"][key] = events["types"].get(key, 0) + 1
        if harness == "claude-code" and t == "system" and sub == "compact_boundary":
            events["compactions"] += 1
        elif harness == "pi" and t == "compaction_end" and e.get("result"):
            events["compactions"] += 1
    if harness == "codex":
        # `codex exec --json` shows no compaction event; its session rollout records each one.
        for f in (home / ".codex" / "sessions").rglob("*.jsonl"):
            for line in f.read_text(errors="replace").splitlines():
                if '"type":"compacted"' in line:
                    events["compactions"] += 1
    if harness == "claude-code":
        result = next((e for e in H.json_lines(out) if e.get("type") == "result"), {})
        detail["result"] = {k: result.get(k) for k in ("subtype", "is_error", "num_turns", "total_cost_usd")}
    detail["events"] = events

    ok, repo = check_repo(work, hashes)
    calls = rec.stop()
    if KEEP:
        (home / "calls.json").write_text(json.dumps(calls, indent=1, default=str))
        for i, b in enumerate(DUMPS[:3]):
            (home / f"marker-body-{i}.json").write_text(b)
    detail.update(repo)
    detail.update(exit=code, stdout_tail=out[-1500:], stderr_tail=err[-1500:])
    if KEEP:
        detail["kept"] = str(home)
        (home / "harness.stdout").write_text(out)
        (home / "harness.stderr").write_text(err)
    else:
        shutil.rmtree(home, ignore_errors=True)
    return ok, calls, detail


# --- TOOL-1: large tool sets ---------------------------------------------------------------------

def fn_tool(i, schema=None, wire="chat"):
    """Tool `tool_{i:03d}` taking `{x: int}` (or `schema`), in a wire's shape."""
    params = schema or {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    name, desc = f"tool_{i:03d}", f"Dummy tool number {i}. Returns a code for x."
    if wire == "chat":
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}
    if wire == "responses":
        return {"type": "function", "name": name, "description": desc, "parameters": params}
    return {"name": name, "description": desc, "input_schema": params}


def deep_schema(depth):
    """An object nested `depth` levels deep, a required integer `x` at the top."""
    node = {"type": "object", "properties": {"leaf": {"type": "string"}}}
    for d in range(depth):
        node = {"type": "object", "properties": {f"level{depth - d}": node, "note": {"type": "string"}}}
    node["properties"]["x"] = {"type": "integer"}
    node["required"] = ["x"]
    return node


def enum_schema(n):
    return {"type": "object", "properties": {
        "x": {"type": "integer"},
        "color": {"type": "string", "enum": [f"shade_{i:04d}_{'abcdefgh'[i % 8] * 6}" for i in range(n)]}},
        "required": ["x"]}


# (wire, n tools, schema kind) per scenario; the provider's documented limit decides the expectation.
OPENAI_TOOL_LIMIT = 128


def tools_probe(scenario):
    """One SDK request offering many tools, forced to call the last one.

    Scenario: `<wire>_<n>[_deep<d>|_enum<k>]` (wire: chat | responses | messages | namespace).
    Passes iff the call succeeded and called the last tool (none silently dropped), or the gateway
    or provider refused it with a 4xx that names the limit. The Rust trial decides which outcome
    the route must give.
    """
    import anthropic
    import openai

    rec = Recorder(BASE)
    parts = scenario.split("_")
    wire, n = parts[0], int(parts[1])
    schema = None
    for p in parts[2:]:
        if p.startswith("deep"):
            schema = deep_schema(int(p[4:]))
        elif p.startswith("enum"):
            schema = enum_schema(int(p[4:]))
    last = f"tool_{n - 1:03d}"
    prompt = f"Call the tool {last} with x=7. Do nothing else."
    detail = {"scenario": scenario, "tools": n, "last": last}
    outcome = {}
    try:
        if wire == "chat":
            c = openai.OpenAI(base_url=f"{rec.base}/v1", api_key=KEY, max_retries=0, timeout=180)
            r = c.chat.completions.create(
                model=MODEL, max_completion_tokens=2000, messages=[{"role": "user", "content": prompt}],
                tools=[fn_tool(i, schema, "chat") for i in range(n)],
                tool_choice={"type": "function", "function": {"name": last}})
            calls = r.choices[0].message.tool_calls or []
            outcome = {"status": 200, "called": [t.function.name for t in calls]}
        elif wire in ("responses", "namespace"):
            c = openai.OpenAI(base_url=f"{rec.base}/v1", api_key=KEY, max_retries=0, timeout=180)
            if wire == "responses":
                tools = [fn_tool(i, schema, "responses") for i in range(n)]
                choice = {"type": "function", "name": last}
            else:
                # Codex's shape: MCP tools grouped in one namespace tool (`mcp__dummy__`).
                tools = [{"type": "namespace", "name": "mcp__dummy__",
                          "description": "Tools from the dummy MCP server.",
                          "tools": [fn_tool(i, schema, "responses") for i in range(n)]}]
                choice = "required"
                prompt = f"Call the tool mcp__dummy__{last} (the {last} tool of the dummy server) with x=7."
            r = c.responses.create(model=MODEL, max_output_tokens=2000, input=prompt, tools=tools,
                                   tool_choice=choice, store=False)
            called = []
            for item in r.output:
                if item.type == "function_call":
                    ns = getattr(item, "namespace", None)
                    called.append(f"{ns}{item.name}" if ns else item.name)
            outcome = {"status": 200, "called": called}
        elif wire == "messages":
            c = anthropic.Anthropic(base_url=rec.base, api_key=KEY, max_retries=0, timeout=180)
            r = c.messages.create(model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": prompt}],
                                  tools=[fn_tool(i, schema, "messages") for i in range(n)],
                                  tool_choice={"type": "tool", "name": last})
            outcome = {"status": 200, "called": [b.name for b in r.content if b.type == "tool_use"]}
        else:
            return False, rec.stop(), {"why": f"unknown wire {wire}"}
    except (openai.APIStatusError, anthropic.APIStatusError) as e:
        body = e.response.text if e.response is not None else str(e)
        outcome = {"status": e.status_code, "error": body[:1500]}
    except Exception as e:  # noqa: BLE001
        outcome = {"status": None, "error": f"{type(e).__name__}: {e}"[:1500]}
    calls = rec.stop()
    detail.update(outcome)
    want = last if wire != "namespace" else f"mcp__dummy__{last}"
    detail["called_last"] = want in (outcome.get("called") or [])
    return outcome.get("status") is not None, calls, detail


# --- TOOL-1: a harness offered ~150 MCP tools ----------------------------------------------------

MCP_TOOLS = int(os.environ.get("VERIFY_MCP_TOOLS", "150"))
MCP_SERVER = r'''
import json, sys
N = int(sys.argv[1])
def send(m):
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    mid, meth = m.get("id"), m.get("method")
    if meth == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": m["params"].get("protocolVersion", "2025-06-18"),
              "capabilities": {"tools": {}}, "serverInfo": {"name": "dummy", "version": "1.0"}}})
    elif meth == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": f"probe_{i:03d}", "description": f"Dummy probe {i}: returns its secret code for x.",
             "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}
            for i in range(N)]}})
    elif meth == "tools/call":
        name = m["params"]["name"]; x = m["params"].get("arguments", {}).get("x")
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text",
              "text": f"secret code for {name} x={x}: ZQ-{name[-3:]}-{x}-OK"}]}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
'''


def mcp_session(harness):
    home = pathlib.Path(tempfile.mkdtemp(prefix=f"verify-mcp-{harness}-"))
    work = home / "repo"
    work.mkdir()
    server = home / "dummy_mcp.py"
    server.write_text(MCP_SERVER)
    rec = Recorder(BASE)
    env = clean_env(home)
    last = f"probe_{MCP_TOOLS - 1:03d}"
    secret = f"ZQ-{last[-3:]}-7-OK"
    task = (f"Use the MCP tool {last} from the dummy server with x=7, then reply with exactly the secret "
            "code it returns and nothing else.")
    detail = {"harness": harness, "mcp_tools": MCP_TOOLS, "secret": secret}
    if harness == "claude-code":
        claude_env(env, rec.base)
        cfg = home / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"dummy": {"command": sys.executable,
                                                            "args": [str(server), str(MCP_TOOLS)]}}}))
        cmd = [str(BIN / "claude"), "-p", task, "--model", MODEL, "--dangerously-skip-permissions",
               "--mcp-config", str(cfg), "--strict-mcp-config", "--output-format", "json", "--max-turns", "8"]
    elif harness == "codex":
        extra = (f'\n[mcp_servers.dummy]\ncommand = "{sys.executable}"\n'
                 f'args = ["{server}", "{MCP_TOOLS}"]\n')
        # web_search off: on a Claude row Codex's hosted web_search is refused outright (D78),
        # which would hide what this case is about.
        ch = codex_home(home, rec.base, 'web_search = "disabled"\n')
        # mcp_servers must follow the top-level keys, before or after the provider table is fine.
        with open(ch / "config.toml", "a") as f:
            f.write(extra)
        env.update(CODEX_HOME=str(ch), BEYOND_API_KEY=KEY)
        cmd = [str(BIN / "codex"), "exec", "--json", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox", "-m", MODEL, task]
    else:
        return False, rec.stop(), {"why": f"unknown harness {harness}"}
    H.TIMEOUT = 600
    code, out, err = H.run(cmd, work, env)  # -9 at H.TIMEOUT or the cell's deadline
    calls = rec.stop()
    detail.update(exit=code, found_secret=secret in out, stdout_tail=out[-1500:], stderr_tail=err[-800:])
    shutil.rmtree(home, ignore_errors=True)
    return secret in out, calls, detail


# --- recorded live.rs cells ----------------------------------------------------------------------
#
# `cell <harness>[:<mode>]+<scenario>`: one short session behind the recorder, for a `live.rs` cell
# that holds every call the harness made to the ledger (`recorded_problems` there). A call may carry
# `expect` (the same keys as a probe's: rows=0, estimated, output_min, row_min, provider), and the
# verdict `same_provider` when every row must name one provider. The scenarios:
#
# | scenario | what the harness does                                                      | claims |
# | -------- | -------------------------------------------------------------------------- | ------ |
# | task     | harness.py's fixture task (fix calc.py)                                     | R1, E3, T1 |
# | big      | the task, and its requests are over 64 KiB (Claude Code's always are)       | R5     |
# | pin      | the task on a two-provider row: one provider every turn, cache read from turn 2 | R4 |
# | cache    | the task, no `cache_control` in any request, cache read from turn 2         | K1     |
# | thinking | the task with thinking on; a later turn replays signed thinking / encrypted reasoning | T2, E2 |
# | vision   | pi reads a PNG fixture (`@codeword.png`, sent base64) and copies its code word | T3   |
# | stream   | Claude Code streams a long answer, directly and through the gateway: TTFT and spread compared | S1 |
# | context  | Claude Code's `/context`, which counts tokens (free: no row)                | E5     |
# | abort    | Claude Code interrupted (SIGINT) mid-stream: that call is billed an estimate | B2    |
# | byo      | Claude Code with the managed key, a forged one (401, nothing billed), and the provider's own key through /anthropic (served, no row) | A1 |
#
# Claims that hold the client's usage to the ledger (E1, E2, B3, B1) get each served call's usage as
# the client saw it on the wire (`UsageScan`) as `usage`, which `recorded_problems` compares with
# the row: input, output and cache reads. With T1, a served turn must replay one of the agent's tool
# calls with its result (`tool_round_trips`), and no turn feeding a result back may be refused. With S2, every turn after the first must replay the history
# the harness accumulated from its streamed answers (only known roles, no assistant message that
# lost both its text and its tool calls), and be accepted.

BILLED_SUFFIXES = ("/v1/messages", "/v1/chat/completions", "/v1/responses", "/v1/responses/compact",
                   "/v1/embeddings")


def billed(call):
    return call["method"] == "POST" and call["path"].endswith(BILLED_SUFFIXES)


def served(calls):
    """The billed calls that succeeded, in order."""
    return [c for c in calls if billed(c) and 200 <= (c.get("status") or 0) < 300]


def run_task(rec, target):
    """harness.py's fixture task, its traffic through the recorder."""
    H.BASE = rec.base
    return H.main(target)


CLAIMS = set(os.environ.get("VERIFY_CLAIMS", "").split("+"))
ROLES = {"system", "developer", "user", "assistant", "tool"}


def cell(spec):
    ok, calls, detail, extra = cell_session(spec)
    if CLAIMS & {"E1", "E2", "B3", "B1"}:
        for c in served(calls):
            if c.get("seen_usage") is None:
                ok, detail["why"] = False, f"no usage found in the response to {c.get('request_id')}"
            else:
                c["usage"] = c["seen_usage"]
    if "T1" in CLAIMS:
        # The agent's tool calls went out through the gateway and came back: a later turn replays
        # a call with its result (ids paired), the gateway served that turn, no turn carrying a
        # result was refused, and the task's outcome (the fixture test passes) needed the results.
        done = served(calls)
        trips = [c.get("tool_round_trips") or 0 for c in done]
        refused = [c.get("request_id") for c in calls
                   if billed(c) and (c.get("status") or 0) >= 400 and c.get("tool_round_trips")]
        detail["t1"] = {"round_trips_per_turn": trips, "refused": refused}
        if not any(trips) or refused:
            ok, detail["why"] = False, ("T1: no served turn replayed a tool call with its result"
                                        if not any(trips) else "T1: a turn feeding a tool result back was refused")
    if "S2" in CLAIMS:
        done = served(calls)
        later = done[1:]
        bad = [(c.get("request_id"), c.get("roles"), c.get("assistant_empty")) for c in later
               if not c.get("assistant_turns") or c.get("assistant_empty") or not set(c.get("roles") or ()) <= ROLES]
        detail["s2"] = {"turns": len(done), "streamed": [c.get("stream") for c in done], "bad": bad}
        if not later or bad or not all(c.get("stream") for c in done):
            ok, detail["why"] = False, ("S2: every turn must stream, and every turn after the first must replay "
                                        "the accumulated assistant messages cleanly")
        if any(billed(c) and (c.get("status") or 0) >= 400 for c in calls):
            ok, detail["why"] = False, "S2: a turn was refused"
    return ok, calls, detail, extra


def cell_session(spec):
    target, _, scenario = spec.partition("+")
    harness = target.partition(":")[0]
    H.TIMEOUT = 420
    rec = Recorder(BASE)
    extra = {}
    if scenario == "thinking":
        if harness == "claude-code":
            os.environ["MAX_THINKING_TOKENS"] = "2048"
        elif harness == "pi":
            H.EXTRA_ARGS = ["--thinking", "low"]
        elif harness == "codex":
            # At Codex's default effort gpt-5.3-codex often answers this task without reasoning.
            H.EXTRA_ARGS = ["-c", 'model_reasoning_effort="high"']
    if scenario in ("task", "big", "pin", "cache", "thinking"):
        ok, detail = run_task(rec, target)
        calls = rec.stop()
        done = served(calls)
        detail = {k: detail.get(k) for k in ("harness", "mode", "exit", "test_rc", "test_unchanged", "stdout_tail",
                                              "stderr_tail", "exception", "timed_out")}
        detail["served"] = len(done)
        if not done:
            ok, detail["why"] = False, "no billed call succeeded"
        if scenario == "big":
            biggest = max((c["req_bytes"] for c in done), default=0)
            detail["largest_request"] = biggest
            if biggest <= 65536:
                ok, detail["why"] = False, f"no request over 64 KiB (largest {biggest} bytes)"
        elif scenario == "pin":
            extra["same_provider"] = True
            for c in done[1:]:
                c["expect"] = {"row_min": {"cache_read_tokens": 1}}
        elif scenario == "cache":
            marked = [c["request_id"] for c in calls if c.get("cache_control")]
            detail["cache_control_sent"] = marked
            if marked:
                ok, detail["why"] = False, "the harness sent cache_control itself"
            for c in done[1:]:
                c["expect"] = {"row_min": {"cache_read_tokens": 1}}
        elif scenario == "thinking":
            detail["thinking"] = [(c.get("thinking"), c.get("replayed_thinking")) for c in done]
            replayed = [c for c in done if c.get("thinking") and c.get("replayed_thinking")]
            if not replayed:
                ok, detail["why"] = False, "no successful turn replayed signed thinking or encrypted reasoning"
            if any(billed(c) and (c.get("status") or 0) >= 400 for c in calls):
                ok, detail["why"] = False, "a turn was refused"
        return ok, calls, detail, extra
    if scenario == "vision":
        if harness != "pi":
            return False, rec.stop(), {"why": "scenario vision is pi only"}, extra
        home = pathlib.Path(tempfile.mkdtemp(prefix="verify-cell-vision-"))
        try:
            ok, detail = pi_vision(rec, home, target.partition(":")[2])
        finally:
            calls = rec.stop()
            shutil.rmtree(home, ignore_errors=True)
        sent = [c.get("images") for c in served(calls)]
        detail["images_sent"] = sent
        if not any(sent):
            ok, detail["why"] = False, "no served request carried the image"
        return ok, calls, detail, extra
    if harness != "claude-code":
        return False, rec.stop(), {"why": f"scenario {scenario} is Claude Code only"}, extra
    fn = {"context": cc_context, "abort": cc_abort, "byo": cc_byo, "stream": cc_stream}.get(scenario)
    if fn is None:
        return False, rec.stop(), {"why": f"unknown scenario {scenario}"}, extra
    home = pathlib.Path(tempfile.mkdtemp(prefix=f"verify-cell-{scenario}-"))
    work = home / "repo"
    work.mkdir()
    try:
        ok, detail = fn(rec, home, work)
    finally:
        calls = rec.stop()
        shutil.rmtree(home, ignore_errors=True)
    return ok, calls, detail, extra


def pi_vision(rec, home, mode):
    """T3: pi attaches a PNG (`@file`; pi sends images base64, and has no PDF or image-URL input)
    and the model copies the code word it shows."""
    work = home / "repo"
    work.mkdir()
    shutil.copy(ROOT / "fixtures" / "codeword.png", work / "codeword.png")
    env = clean_env(home)
    agent_dir = pi_setup(home, rec.base, mode, H.catalog())
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    cmd = [str(BIN / "pi"), "-p", "--mode", "json", "--provider", "beyond", "--model", MODEL, "--no-session",
           "--no-tools", "@codeword.png",
           "What words and digits does this image show? Reply with exactly them and nothing else."]
    code, out, err = H.run(cmd, work, env)
    msgs = [e["message"] for e in H.json_lines(out)
            if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant"]
    text = " ".join(b.get("text", "") for m in msgs for b in m.get("content") or [] if isinstance(b, dict))
    ok = code == 0 and "4821" in text and "KESTREL" in text.upper()
    return ok, {"exit": code, "text": text[:200], "stderr_tail": err[-400:]}


# S1's budget: the gateway's time to first token may exceed the direct call's by at most the larger
# of the direct TTFT itself and a second (provider TTFT varies run to run by about that much), and a
# stream that took the provider a while must arrive spread out through the gateway too, at the wire
# and in Claude Code's own delta events: per delta, at least a fifth of the direct spread
# (parity_live.rs `incremental`'s ratio). Per delta, because the two answers differ in length (how
# long the model thinks varies run to run); a buffered stream arrives all at once, whatever its length.
S1_SLACK_S = 1.0
S1_MIN_SPREAD = 0.2


def cc_stream(rec, home, work):
    """S1: Claude Code streams the same long answer twice, first straight to Anthropic (the
    provider key, through a second recorder that relays to api.anthropic.com), then through the
    gateway. Both recorders time the answer as it reaches Claude Code (first content delta, last
    byte), and Claude Code's own `--include-partial-messages` events are timed as it emits them."""
    prompt = ("Write the whole numbers from one to one hundred and twenty in words, one per line, "
              "and nothing else. Do not use any tools.")

    def once(recorder, key):
        env = clean_env(home)
        claude_env(env, recorder.base)
        env["ANTHROPIC_API_KEY"] = key
        cmd = [str(BIN / "claude"), "-p", prompt, "--model", MODEL, "--dangerously-skip-permissions",
               "--output-format", "stream-json", "--verbose", "--include-partial-messages"]
        p = subprocess.Popen(cmd, cwd=work, env={**env, "PWD": str(work)}, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        timer = threading.Timer(150, p.kill)
        timer.start()
        seen, result = [], {}
        try:
            for line in p.stdout:
                t = time.time()
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                ev = e.get("event") or {}
                # Text and thinking alike: on Haiku 4.5 Claude Code often thinks first, and then most
                # of the answer's time is its thinking.
                delta = (ev.get("delta") or {}).get("type")
                if (e.get("type") == "stream_event" and ev.get("type") == "content_block_delta"
                        and delta in ("text_delta", "thinking_delta")):
                    seen.append(t)
                elif e.get("type") == "result":
                    result = e
            p.wait()
        finally:
            timer.cancel()
        # The relay records a call once its last chunk is written; let every one finish.
        end = time.time() + 10
        while recorder.progress and time.time() < end:
            time.sleep(0.05)
        time.sleep(0.2)
        main = max((c for c in recorder.calls if billed(c) and c.get("t_first_delta")),
                   key=lambda c: c.get("deltas", 0), default=None)
        out = {"exit": p.returncode, "result": (result.get("result") or "")[:60],
               "client_deltas": len(seen), "client_spread_s": round(seen[-1] - seen[0], 3) if seen else 0}
        if main:
            out.update(request_id=main.get("request_id"), wire_deltas=main.get("deltas"),
                       ttft_s=round(main["t_first_delta"] - main["t0"], 3),
                       wire_spread_s=round(main["t1"] - main["t_first_delta"], 3))
        return out

    direct_rec = Recorder("https://api.anthropic.com")
    try:
        direct = once(direct_rec, os.environ["VERIFY_BYO_KEY"])
    finally:
        direct_rec.stop()
    gw = once(rec, KEY)
    detail = {"direct": direct, "gateway": gw, "slack_s": S1_SLACK_S, "min_spread": S1_MIN_SPREAD}
    why = []
    for name, r in (("direct", direct), ("gateway", gw)):
        if r["exit"] != 0 or "ttft_s" not in r:
            why.append(f"{name}: Claude Code exited {r['exit']} or its answer never streamed")
    if not why:
        if direct["wire_spread_s"] < 0.4 or direct["wire_deltas"] < 4 or direct["client_deltas"] < 4:
            why.append("the direct answer was too short to measure a spread")
        budget = direct["ttft_s"] + max(direct["ttft_s"], S1_SLACK_S)
        if gw["ttft_s"] > budget:
            why.append(f"gateway TTFT {gw['ttft_s']}s over budget {budget:.3f}s")
        for k, n in (("wire_spread_s", "wire_deltas"), ("client_spread_s", "client_deltas")):
            if gw[k] / max(gw[n], 1) < direct[k] / max(direct[n], 1) * S1_MIN_SPREAD:
                why.append(f"stream buffered: {k} gateway {gw[k]}s over {gw[n]} deltas, "
                           f"direct {direct[k]}s over {direct[n]}")
    if why:
        detail["why"] = "; ".join(why)
    return not why, detail


def cc_cmd(prompt, *more):
    return [str(BIN / "claude"), "-p", prompt, "--model", MODEL, "--dangerously-skip-permissions",
            "--output-format", "json", *more]


def cc_result(out):
    return next((e for e in H.json_lines(out) if e.get("type") == "result"), {})


def cc_context(rec, home, work):
    """E5: `/context` counts the session's tokens through POST /v1/messages/count_tokens."""
    env = clean_env(home)
    claude_env(env, rec.base)
    code, out, _ = H.run(cc_cmd("/context"), work, env)
    counts = [c for c in rec.calls if c["path"] == "/v1/messages/count_tokens"]
    shown = cc_result(out).get("result") or ""
    ok = (code == 0 and counts and all(c.get("status") == 200 for c in counts)
          and re.search(r"\d+(\.\d+)?k", shown) is not None)
    return ok, {"exit": code, "count_tokens_calls": len(counts), "shown": shown[:400]}


def cc_abort(rec, home, work):
    """B2: Claude Code interrupted mid-answer, the way a user's Ctrl-C does it."""
    env = clean_env(home)
    claude_env(env, rec.base)
    cmd = cc_cmd("Write a 1500-word story about a lighthouse keeper. Do not use any tools.")
    p = subprocess.Popen(cmd, cwd=work, env={**env, "PWD": str(work)}, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    cut = None
    while time.time() < deadline and p.poll() is None:
        inflight = [(c, n) for c, n in list(rec.progress.values()) if billed(c) and n >= 6000]
        if inflight:
            cut = inflight[0][0]
            p.send_signal(2)
            break
        time.sleep(0.05)
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    # The recorder notices the client gone on its next write; let every relay finish.
    end = time.time() + 30
    while rec.progress and time.time() < end:
        time.sleep(0.1)
    if cut is None:
        return False, {"why": "the answer never streamed 6 KB before the session ended", "exit": p.returncode}
    cut["expect"] = {"estimated": True, "output_min": 1}
    return cut.get("client_gone") is True, {"exit": p.returncode, "cut_request": cut.get("request_id"),
                                            "relayed_bytes": cut.get("resp_bytes")}


def cc_byo(rec, home, work):
    """A1: the managed key is served and billed; a forged bai key is a 401 that bills nothing; the
    provider's own key through /anthropic is served and writes no row."""
    detail = {}
    ask = "Reply with the single word: pong"
    env = clean_env(home)
    claude_env(env, rec.base)
    code, out, _ = H.run(cc_cmd(ask), work, env)
    managed = cc_result(out)
    detail["managed"] = {"exit": code, "result": (managed.get("result") or "")[:80]}
    ok = code == 0 and "pong" in (managed.get("result") or "").lower()

    n = len(rec.calls)
    forged = KEY[:-6] + ("AAAAAA" if not KEY.endswith("AAAAAA") else "BBBBBB")
    # Claude Code retries a 401 ten times with backoff (minutes); one retry still shows it.
    env.update(ANTHROPIC_API_KEY=forged, CLAUDE_CODE_MAX_RETRIES="1")
    code, out, err = H.run(cc_cmd(ask), work, env)
    del env["CLAUDE_CODE_MAX_RETRIES"]
    refused = [c for c in rec.calls[n:] if billed(c)]
    detail["forged"] = {"exit": code, "statuses": [c.get("status") for c in refused],
                        "shown": (cc_result(out).get("result") or err)[-200:]}
    ok = ok and code != 0 and refused and all(c.get("status") == 401 for c in refused)

    n = len(rec.calls)
    env.update(ANTHROPIC_BASE_URL=f"{rec.base}/anthropic", ANTHROPIC_API_KEY=os.environ["VERIFY_BYO_KEY"])
    code, out, _ = H.run(cc_cmd(ask), work, env)
    byo = [c for c in rec.calls[n:] if billed(c)]
    for c in rec.calls[n:]:
        c["expect"] = {"rows": 0}
    detail["byo"] = {"exit": code, "paths": sorted({c["path"] for c in byo}),
                     "result": (cc_result(out).get("result") or "")[:80]}
    ok = (ok and code == 0 and byo and all(c["path"].startswith("/anthropic/") and c.get("status") == 200 for c in byo)
          and "pong" in (cc_result(out).get("result") or "").lower())
    return bool(ok), detail


if __name__ == "__main__":
    kind, arg = sys.argv[1], sys.argv[2]
    calls = []
    if kind == "cell":
        # A live.rs cell: the verdict says its calls were recorded, so they are held one by one.
        extra = {}
        try:
            ok, calls, detail, extra = cell(arg)
        except Exception as e:  # noqa: BLE001
            import traceback
            ok, detail = False, {"exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        if H.TIMED_OUT is not None:
            detail["timed_out"] = H.TIMED_OUT
        for c in calls:
            c.pop("resp_tail", None)
        if os.environ.get("VERIFY_CELL_TRACE"):
            # What a passing cell saw (the Rust cell prints the detail only on a failure).
            print("CELL " + json.dumps({"ok": bool(ok), "detail": detail, "calls": calls}, default=str), file=sys.stderr)
        print("VERIFY " + json.dumps({"ok": bool(ok), "recorded": True, "calls": calls, "detail": detail, **extra},
                                     default=str))
        sys.exit(0)
    try:
        fn = {"long": long_session, "tools": tools_probe, "mcp": mcp_session}[kind]
        ok, calls, detail = fn(arg)
    except Exception as e:  # noqa: BLE001
        import traceback
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    if H.TIMED_OUT is not None:
        detail["timed_out"] = H.TIMED_OUT
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": calls, "detail": detail}, default=str))
