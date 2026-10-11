#!/usr/bin/env python3
"""Measure the three ways MCP tools reach the model: eager, tool search (default), Code Mode.

The numbers in crates/agent/ARCHITECTURE.md ("Deferred MCP tools") come from this script. Every
subcommand drives the real release binaries; keys come from the environment or the repo's `.env`.

    cargo build --release -p beyond-ai-agent
    CARGO_TARGET_DIR=target/code-mode cargo build --release -p beyond-ai-agent --features code-mode

    python3 crates/agent/eval/mcp_loading.py first-request   # first-request tokens, real usage
    python3 crates/agent/eval/mcp_loading.py session         # cache over a multi-turn session
    python3 crates/agent/eval/mcp_loading.py tasks [REPEATS] # live tasks on real MCP servers
    python3 crates/agent/eval/mcp_loading.py rss [RUNS]      # RSS, tool search vs Code Mode

`first-request` and `session` replay real servers' captured `tools/list` (`mcp_catalogs/`, refreshed
with `capture OUT.json CMD…`); `tasks` runs the real servers (`npx` playwright, github, filesystem,
plus `npx playwright install chromium` once). Spend: about $0.70 for all four.
"""
import http.server
import json
import os
import select
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))


def find_root(d):
    """The workspace root: the nearest ancestor holding `Cargo.lock`."""
    while not os.path.exists(os.path.join(d, "Cargo.lock")):
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("run from inside the repository")
        d = parent
    return d


ROOT = find_root(HERE)
CATALOGS = os.path.join(HERE, "mcp_catalogs")
WORK = os.path.join(ROOT, "target", "mcp-loading-eval")
BINS = {
    "default": os.environ.get("AGENT_BIN", os.path.join(ROOT, "target/release/beyond-ai-agent")),
    "code": os.environ.get("AGENT_BIN_CODE_MODE", os.path.join(ROOT, "target/code-mode/release/beyond-ai-agent")),
}
MODES = {
    "eager": ("default", ["--mcp-tool-loading", "eager"]),
    "search": ("default", ["--mcp-tool-loading", "search"]),
    "code": ("code", ["--code-mode"]),
}
MODELS = ["claude-sonnet-5-5", "gpt-6.1-sol"]
# $/MTok: input, output, cache read, cache write (5 min) — crates/providers' rate table.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5), "gpt-6.1-sol": (2.0, 10.0, 0.10, 0.0)}
SETS = {
    "1 server (github, 26 tools)": ["github"],
    "3 servers (playwright, github, filesystem; 65 tools)": ["playwright", "github", "filesystem"],
    "7 servers (+memory, everything, sequential-thinking, notion; 112 tools)": [
        "playwright", "github", "filesystem", "memory", "everything", "thinking", "notion"],
}


def load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"'))


def post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        return json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode()}")


def replay_servers(names):
    return [{"name": n, "transport": "stdio", "command": sys.executable,
             "args": ["-I", os.path.abspath(__file__), "replay", os.path.join(CATALOGS, n + ".json")], "env": {}}
            for n in names]


def tempdir(prefix):
    os.makedirs(WORK, exist_ok=True)
    return tempfile.mkdtemp(dir=WORK, prefix=prefix)


def home_with(servers):
    home = tempdir("home-")
    os.makedirs(os.path.join(home, ".claude"))
    json.dump({"mcp_servers": servers}, open(os.path.join(home, ".claude", "settings.json"), "w"))
    return home


# --- a scripted local model ----------------------------------------------------------------------

def sse(events):
    return "".join("data: %s\n\n" % json.dumps(e) for e in events)


def anthropic_text(text):
    return sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"}])


def anthropic_tool(tid, name, args):
    return sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 1, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"}])


RESPONSES_TEXT = sse([
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1"}},
    {"type": "response.output_text.delta", "output_index": 0, "delta": "hi"},
    {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1", "phase": "final_answer", "content": [{"type": "output_text", "text": "hi"}]}},
    {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}}}])


class Mock:
    """Records each request body; answers from `script` (or a text turn in the request's wire)."""

    def __init__(self):
        self.bodies, self.script, self.on_first = [], None, None
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                raw = self.rfile.read(n) if n else self._chunked()
                body = json.loads(raw)
                if not mock.bodies and mock.on_first:
                    mock.on_first()
                mock.bodies.append(body)
                if mock.script:
                    out = mock.script[min(len(mock.bodies), len(mock.script)) - 1]
                else:
                    out = RESPONSES_TEXT if "input" in body else anthropic_text("hi")
                out = out.encode()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def _chunked(self):
                raw = b""
                while True:
                    size = int(self.rfile.readline().strip() or b"0", 16)
                    if size == 0:
                        self.rfile.readline()
                        return raw
                    raw += self.rfile.read(size)
                    self.rfile.readline()

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % srv.server_address[1]

    def reset(self, script=None, on_first=None):
        self.bodies, self.script, self.on_first = [], script, on_first


def against_mock(mock, mode, model, servers, extra=(), popen_kw=None):
    binkind, flags = MODES[mode]
    home = home_with(servers)
    args = [BINS[binkind], "run", "go", "--gateway-url", mock.base, "--key", "bai_v1.test", "--model", model,
            "--no-session-persistence", "--no-context-files", "--max-steps", "5"] + flags + list(extra)
    return subprocess.Popen(args, env={"HOME": home, "PATH": os.environ["PATH"]}, cwd=home,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, **(popen_kw or {}))


# --- a real provider, through the real binary ----------------------------------------------------

def run_agent(model, mode, messages, servers, workdir=None, timeout=900):
    binkind, flags = MODES[mode]
    env = dict(os.environ, HOME=home_with(servers))
    for k in ("AI_GATEWAY_URL", "AI_AGENT_KEY"):
        env.pop(k, None)
    t0 = time.time()
    # `--no-context-files`: the work directories sit under this repo's `target/`, whose AGENTS.md
    # and CLAUDE.md would otherwise join every prompt.
    p = subprocess.run([BINS[binkind], "run", "--model", model, "--json", "--no-session-persistence",
                        "--no-context-files", "--max-steps", "30"] + flags + list(messages),
                       env=env, cwd=workdir or tempdir("work-"), capture_output=True, text=True, timeout=timeout)
    m = {"exit": p.returncode, "wall_s": round(time.time() - t0, 1), "requests": [], "tools": [], "text": "",
         "stderr": p.stderr[-1500:]}
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "usage":
            m["requests"].append({k: ev.get(k, 0) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")})
        elif ev.get("kind") == "tool_start":
            m["tools"].append(ev.get("name"))
        elif ev.get("type") == "text_delta":
            m["text"] += ev.get("text", ev.get("delta", ""))
    t = {k: sum(r[k] for r in m["requests"]) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
    pin, pout, pread, pwrite = PRICES[model]
    # The agent reports uncached input apart from cache reads and writes on both wires.
    m["cost"] = (t["input_tokens"] * pin + t["cache_read_tokens"] * pread + t["cache_write_tokens"] * pwrite
                 + t["output_tokens"] * pout) / 1e6
    m["prompt"] = t["input_tokens"] + t["cache_read_tokens"] + t["cache_write_tokens"]
    m["read_share"] = t["cache_read_tokens"] / m["prompt"] if m["prompt"] else 0
    m["totals"] = t
    return m


# --- subcommands ---------------------------------------------------------------------------------

def first_request():
    """The agent's real first request, replayed to the real API for its billed input tokens."""
    load_env()
    mock = Mock()

    def billed(body, model):
        body = dict(body, stream=False)
        if model.startswith("claude"):
            r = post("https://api.anthropic.com/v1/messages", dict(body, max_tokens=16), {
                "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                "content-type": "application/json"})
            u = r["usage"]
            return u["input_tokens"] + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        body.pop("include", None)
        r = post("https://api.openai.com/v1/responses", dict(body, max_output_tokens=16), {
            "authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "content-type": "application/json"})
        return r["usage"]["input_tokens"]

    def first(mode, model, names):
        mock.reset()
        p = against_mock(mock, mode, model, replay_servers(names))
        _, err = p.communicate(timeout=300)
        if p.returncode or not mock.bodies:
            raise SystemExit(err[-1500:])
        return mock.bodies[0]

    base = {m: billed(first("eager", m, []), m) for m in MODELS}
    print("| servers | mode | Claude tokens (MCP share) | GPT tokens (MCP share) |")
    print("|---|---|---|---|")
    for set_name, names in SETS.items():
        for mode in MODES:
            cells = []
            for model in MODELS:
                n = billed(first(mode, model, names), model)
                cells.append(f"{n:,} ({n - base[model]:,})")
            print(f"| {set_name} | {mode} | {cells[0]} | {cells[1]} |", flush=True)
    print(f"| none | — | {base[MODELS[0]]:,} | {base[MODELS[1]]:,} |")


def session():
    """Four user turns over three replayed catalogs; the 2nd and 3rd each need a different server."""
    load_env()
    messages = [
        "Run `ls -la` with bash and tell me how many entries there are. Do not use any MCP tool.",
        "Now use the GitHub MCP tools to fetch issue #1 of octocat/Hello-World and tell me what the tool returned.",
        "Now use the Playwright MCP browser tools to navigate to https://example.com and report what the tool returned.",
        "Summarize in one sentence what you did.",
    ]
    servers = replay_servers(["playwright", "github", "filesystem"])
    print("| model | mode | requests | uncached in | cache read | cache write | read share | cost $ |")
    print("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for mode in MODES:
            m = run_agent(model, mode, messages, servers)
            t = m["totals"]
            print(f"| {model} | {mode} | {len(m['requests'])} | {t['input_tokens']:,} | {t['cache_read_tokens']:,} | "
                  f"{t['cache_write_tokens']:,} | {m['read_share']:.3f} | {m['cost']:.4f} |", flush=True)
            for i, r in enumerate(m["requests"]):
                print(f"<!-- req {i}: {r} -->")


def tasks(repeats):
    """Two tasks on the real servers, each needing a tool tool search has not loaded up front."""
    load_env()
    work = tempdir("live-")
    servers = [
        {"name": "playwright", "transport": "stdio", "command": "npx",
         "args": ["-y", "@playwright/mcp@latest", "--headless", "--isolated", "--browser", "chromium"], "env": {}},
        {"name": "github", "transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
         "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")}},
        {"name": "filesystem", "transport": "stdio", "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", work], "env": {}},
    ]
    cases = {
        "playwright: example.com's JS-rendered link": (
            "Open https://example.com in the browser (part of the page is rendered by JavaScript) and tell me "
            "the exact text of the page's only link. Reply with just the link text.", "learn more"),
        "github: issue #1 title in octocat/Hello-World": (
            "Using the GitHub tools, find the title of issue #1 in the repository octocat/Hello-World. "
            "Reply with just the title.", "edited readme via github"),
    }
    print("| task | model | mode | success | requests | wall s | prompt tokens | read share | cost $ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for task, (prompt, want) in cases.items():
        for model in MODELS:
            for mode in MODES:
                ms = [run_agent(model, mode, [prompt], servers, workdir=work) for _ in range(repeats)]
                ok = sum(m["exit"] == 0 and want in m["text"].lower() for m in ms)
                mean = lambda f: statistics.mean(f(m) for m in ms)
                print(f"| {task} | {model} | {mode} | {ok}/{repeats} | {mean(lambda m: len(m['requests'])):.1f} | "
                      f"{mean(lambda m: m['wall_s']):.1f} | {mean(lambda m: m['prompt']):,.0f} | "
                      f"{mean(lambda m: m['read_share']):.2f} | {mean(lambda m: m['cost']):.4f} |", flush=True)


def rss(runs):
    """Resident memory with one MCP call through each path, against a scripted local model."""
    mock = Mock()
    get_issue = ("mcp__github__get_issue", {"owner": "o", "repo": "r", "issue_number": 1})
    scripts = {
        "eager": [anthropic_tool("t1", *get_issue), anthropic_text("done")],
        "search": [anthropic_tool("t1", "tool_search", {"query": "github get issue"}),
                   anthropic_tool("t2", *get_issue), anthropic_text("done")],
        "code": [anthropic_tool("t1", "execute", {"code": "return await tools.github.get_issue({owner:'o',repo:'r',issue_number:1});"}),
                 anthropic_text("done")],
    }
    servers = replay_servers(["playwright", "github", "filesystem"])

    def kb(pid, field):
        try:
            return next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith(field + ":"))
        except (FileNotFoundError, StopIteration):
            return 0

    samples = {m: [] for m in MODES}
    for _ in range(runs):
        for mode in MODES:  # interleaved, so host drift hits every mode alike
            idle = {}
            holder = {}
            mock.reset(scripts[mode], on_first=lambda: (time.sleep(0.3), idle.update(
                anon=kb(holder["pid"], "RssAnon"), file=kb(holder["pid"], "RssFile"))))
            # PR_SET_THP_DISABLE: 2 MiB huge pages otherwise quantize anon RSS into noise.
            p = against_mock(mock, mode, "claude-sonnet-5-5", servers,
                             popen_kw={"preexec_fn": lambda: __import__("ctypes").CDLL(None).prctl(41, 1, 0, 0, 0)})
            holder["pid"] = p.pid
            peak_anon = peak_file = 0
            while p.poll() is None:
                peak_anon = max(peak_anon, kb(p.pid, "RssAnon"))
                peak_file = max(peak_file, kb(p.pid, "RssFile"))
                time.sleep(0.005)
            if p.returncode:
                raise SystemExit(p.stderr.read()[-1500:])
            samples[mode].append((idle["anon"], idle["file"], peak_anon, peak_file))
    mib = lambda xs: statistics.median(xs) / 1024
    print("| mode | idle anon MiB | idle file MiB | peak anon MiB | peak file MiB | peak total MiB |")
    print("|---|---|---|---|---|---|")
    for mode, s in samples.items():
        cols = [mib([x[i] for x in s]) for i in range(4)]
        print(f"| {mode} | " + " | ".join(f"{c:.2f}" for c in cols) + f" | {mib([x[2] + x[3] for x in s]):.2f} |")
    for name, path in [("default", BINS["default"]), ("code-mode", BINS["code"])]:
        text = [l for l in subprocess.run(["size", "-A", path], capture_output=True, text=True).stdout.splitlines()
                if l.startswith(".text")]
        print(f"{name}: {os.path.getsize(path):,} bytes; {text[0].split()[1] if text else '?'} bytes of .text")


def replay(catalog):
    """A stdio MCP server answering `tools/list` from a captured catalog, and every call with "ok"."""
    tools = json.load(open(catalog))
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:
            continue
        result = {
            "initialize": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                           "serverInfo": {"name": "replay", "version": "0"}},
            "tools/list": {"tools": tools},
            "tools/call": {"content": [{"type": "text", "text": "ok"}]},
        }.get(method)
        reply = {"result": result} if result is not None else {"error": {"code": -32601, "message": method}}
        print(json.dumps(dict(reply, jsonrpc="2.0", id=mid)), flush=True)


def capture(out, cmd):
    """Save a real server's `tools/list` (e.g. `capture mcp_catalogs/github.json npx -y @modelcontextprotocol/server-github`)."""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def call(mid, method, params):
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params}) + "\n")
        p.stdin.flush()
        end = time.time() + 180
        while time.time() < end:
            if select.select([p.stdout], [], [], 1)[0]:
                line = p.stdout.readline()
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("id") == mid:
                    return msg["result"]
        raise SystemExit("timeout")

    call(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "capture", "version": "0"}})
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    p.stdin.flush()
    tools = call(2, "tools/list", {})["tools"]
    json.dump(tools, open(out, "w"), indent=2)
    p.kill()
    print(out, len(tools), "tools")


if __name__ == "__main__":
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "first-request":
        first_request()
    elif cmd == "session":
        session()
    elif cmd == "tasks":
        tasks(int(rest[0]) if rest else 2)
    elif cmd == "rss":
        rss(int(rest[0]) if rest else 15)
    elif cmd == "replay":
        replay(rest[0])
    elif cmd == "capture":
        capture(rest[0], rest[1:])
    else:
        raise SystemExit(__doc__)
