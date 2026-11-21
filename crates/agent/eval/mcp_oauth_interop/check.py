#!/usr/bin/env python3
"""MCP OAuth interop check for the agent's MCP client, against the OFFICIAL Python MCP SDK's
`examples/servers/simple-auth` (a separate authorization server and a protected MCP resource
server). NOT part of the default CI gate; see README.md.

    AGENT_BINS=main=/path/to/agent-main,pr=/path/to/agent-pr  python check.py

Each binary gets the same assertions (see README for the list). `AGENT_LEGACY_BIN` (default: the
`main` binary) writes the pre-binding credential used by the legacy-entry assertion on `pr`.
Exit status is nonzero when any observation differs from what that binary is expected to do."""
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.dirname(sys.executable)
LOGDIR = os.path.abspath(os.environ.get("CHECK_LOG_DIR") or tempfile.mkdtemp(prefix="mcp-oauth-interop-"))
os.makedirs(LOGDIR, exist_ok=True)
TOOL = "mcp__simple__get_time"
procs = []
rows = []  # (assertion, label, observed, expected, detail)


def free_port():
    """A port free on both loopback families (the servers bind `localhost`, which is both)."""
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        try:
            s6 = socket.socket(socket.AF_INET6)
            s6.bind(("::1", p))
            s6.close()
        except OSError:
            continue
        return p


def wait_port(port, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        for host in ("127.0.0.1", "::1"):
            try:
                socket.create_connection((host, port), 0.5).close()
                return
            except OSError:
                pass
        time.sleep(0.1)
    raise RuntimeError(f"port {port} never opened")


def spawn(argv, logname, env=None):
    log = open(os.path.join(LOGDIR, logname), "ab")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    procs.append(p)
    return p


def logsize(name):
    return os.path.getsize(os.path.join(LOGDIR, name))


def logslice(name, start):
    with open(os.path.join(LOGDIR, name), "rb") as f:
        f.seek(start)
        return f.read().decode(errors="replace")


def requests_in(text):
    """(method, path, status) from uvicorn access lines."""
    return re.findall(r'"(GET|POST|DELETE) (\S+) HTTP/1\.1" (\d+)', text)


class Stack:
    """One official authorization server plus one official protected MCP resource server."""

    def __init__(self, name, variant="", ttl=None):
        self.name = name
        self.as_port, self.rs_port = free_port(), free_port()
        env = dict(os.environ, AS_VARIANT=variant)
        if ttl:
            env["AS_ACCESS_TTL"] = str(ttl)
        self.as_log, self.rs_log = f"{name}-as.log", f"{name}-rs.log"
        spawn([sys.executable, os.path.join(HERE, "as_variant.py"), "--port", str(self.as_port)], self.as_log, env)
        wait_port(self.as_port)
        spawn([os.path.join(BIN_DIR, "mcp-simple-auth-rs"), "--port", str(self.rs_port),
               "--auth-server", f"http://localhost:{self.as_port}", "--transport", "streamable-http", "--oauth-strict"],
              self.rs_log)
        wait_port(self.rs_port)
        self.url = f"http://localhost:{self.rs_port}/mcp"
        self.as_url = f"http://localhost:{self.as_port}"


class Agent:
    def __init__(self, label, binary):
        self.label, self.binary = label, binary
        self.model_log = os.path.join(LOGDIR, f"{label}-model-results.jsonl")
        self.model_port = free_port()
        spawn([sys.executable, os.path.join(HERE, "fake_model.py"), str(self.model_port), TOOL, self.model_log],
              f"{label}-fake-model.log")
        wait_port(self.model_port)

    def home(self, url, binary=None):
        h = tempfile.mkdtemp(prefix="agent-home-")
        self.write_settings(h, url)
        return h

    @staticmethod
    def write_settings(h, url):
        os.makedirs(os.path.join(h, ".beyond"), exist_ok=True)
        with open(os.path.join(h, ".beyond", "settings.json"), "w") as f:
            json.dump({"mcp_servers": [{"name": "simple", "transport": "http", "url": url, "headers": {}}]}, f)

    def env(self, home):
        env = dict(os.environ, HOME=home)
        for k in ["AI_POOL_KEY_ANTHROPIC", "DATABASE_URL", "AI_DIRECT", "AI_PROVIDER", "OPENROUTER_API_KEY",
                  "ANTHROPIC_API_KEY", "AI_AGENT_CONFIG_DIR", "AI_AGENT_LIFECYCLE_URL", "AI_BASE_URL", "AI_API_KEY"]:
            env.pop(k, None)
        return env

    def login(self, home, as_url, binary=None, who=None):
        """`agent mcp-login simple`, with the browser step done headlessly: the page the agent prints is
        the official AS's login form (plain HTML, demo credentials printed on it), so it is submitted
        over HTTP. Returns (ok, stderr_text)."""
        binary = binary or self.binary
        errf = tempfile.NamedTemporaryFile("w+", prefix="login-err-", delete=False)
        p = subprocess.Popen([binary, "mcp-login", "simple"], env=self.env(home), cwd=home,
                             stdout=subprocess.PIPE, stderr=errf, text=True, start_new_session=True)
        procs.append(p)
        auth_url = None
        end = time.time() + 20
        while time.time() < end and auth_url is None and p.poll() is None:
            m = re.search(r"(https?://\S+)", open(errf.name).read())
            if m:
                auth_url = m.group(1)
            else:
                time.sleep(0.1)
        if auth_url is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
            return False, scrub(open(errf.name).read())
        err = ""
        c = httpx.Client(follow_redirects=False, timeout=15)
        url, state = auth_url, None
        for _ in range(6):
            r = c.get(url)
            if r.status_code in (301, 302, 303, 307):
                url = httpx.URL(url).join(r.headers["location"]).__str__()
                continue
            break
        state = parse_qs(urlparse(url).query).get("state", [None])[0]
        r = c.post(f"{as_url}/login/callback", data={"username": "demo_user", "password": "demo_password", "state": state})
        if r.status_code in (301, 302, 303, 307):
            c.get(r.headers["location"])  # the agent's own loopback callback listener
        try:
            out, _ = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
        e = open(errf.name).read()
        os.unlink(errf.name)
        return p.returncode == 0, scrub(e + (out or ""))

    def run(self, home, who=None, rust_log=None):
        """`agent run`: one scripted model turn calls the tool, the next ends. Returns (tool_result|None, stderr)."""
        before = os.path.getsize(self.model_log) if os.path.exists(self.model_log) else 0
        cwd = tempfile.mkdtemp(prefix="agent-cwd-")
        p = subprocess.run([who or self.binary, "run", "call the tool", "--gateway-url", f"http://127.0.0.1:{self.model_port}",
                            "--key", "bai_v1.test", "--model", "claude-test", "--max-steps", "6", "--no-session-persistence"],
                           env=dict(self.env(home), **({"RUST_LOG": rust_log} if rust_log else {})), cwd=cwd, capture_output=True, text=True, timeout=120)
        res = None
        if os.path.exists(self.model_log):
            with open(self.model_log) as f:
                f.seek(before)
                lines = [json.loads(l) for l in f if l.strip()]
            res = lines[-1] if lines else None
        shutil.rmtree(cwd, ignore_errors=True)
        self.nrun = getattr(self, "nrun", 0) + 1
        out = scrub(p.stderr + p.stdout)
        with open(os.path.join(LOGDIR, f"{self.label}-run-{self.nrun:02d}.log"), "w") as f:
            f.write(f"exit={p.returncode}\ntool_result={json.dumps(res)}\n--- agent stderr+stdout (scrubbed)\n{ANSI.sub('', out)}")
        return res, out


SECRET = re.compile(r'("?(?:access_token|refresh_token|client_secret|code)"?\s*[:=]\s*"?)[A-Za-z0-9_\-\.]{8,}', re.I)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
AUTH = re.compile(r'(Bearer\s+)[A-Za-z0-9_\-\.]{8,}')


def scrub(t):
    return AUTH.sub(r"\1<scrubbed>", SECRET.sub(r"\1<scrubbed>", t or ""))


def stored_fields(home):
    try:
        d = json.load(open(os.path.join(home, ".beyond", "mcp_auth.json")))
    except (OSError, ValueError):
        return None
    return sorted(next(iter(d.values())).keys()) if d else []


def record(assertion, label, observed, expected, detail=""):
    rows.append((assertion, label, observed, expected, detail))
    flag = "ok" if observed == expected else "UNEXPECTED"
    print(f"[{label}] {assertion}: {observed} (expected {expected}) {flag}  {detail}", flush=True)


def tool_ok(res):
    return bool(res) and not res["is_error"] and "current_time" in res["text"]


def main():
    bins = dict(kv.split("=", 1) for kv in os.environ["AGENT_BINS"].split(","))
    legacy_bin = os.environ.get("AGENT_LEGACY_BIN") or bins.get("main")
    for label, b in bins.items():
        assert os.access(b, os.X_OK), f"{label}: {b} is not executable"
    pr_like = lambda label: label != "main"  # any non-baseline binary is expected to carry #683's behaviour
    normal = Stack("normal")
    nopkce = Stack("nopkce", "no-pkce")
    refresh = Stack("refresh", "refresh", ttl=3)
    # A2 (independent of the agent): how the official server advertises protected-resource metadata.
    r = httpx.post(normal.url, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe", "version": "0"}}},
        headers={"accept": "application/json, text/event-stream"})
    chal = r.headers.get("www-authenticate", "")
    m = re.search(r'resource_metadata="([^"]+)"', chal)
    wk_path = httpx.get(f"http://localhost:{normal.rs_port}/.well-known/oauth-protected-resource/mcp").status_code
    wk_root = httpx.get(f"http://localhost:{normal.rs_port}/.well-known/oauth-protected-resource").status_code
    print(f"[server] unauthenticated POST {r.status_code}; challenge carries resource_metadata={bool(m)} ({m.group(1) if m else ''}); "
          f"PRM well-known path-suffixed={wk_path} root={wk_root}", flush=True)
    as_md = httpx.get(f"{normal.as_url}/.well-known/oauth-authorization-server").json()
    print(f"[server] AS advertises code_challenge_methods_supported={as_md.get('code_challenge_methods_supported')}", flush=True)

    for label, binary in bins.items():
        agent = Agent(label, binary)
        pr = pr_like(label)

        # A1 + A2: login (headless) then a run calls a tool with the token.
        home = agent.home(normal.url)
        s = logsize(normal.rs_log)
        ok, err = agent.login(home, normal.as_url)
        seen = requests_in(logslice(normal.rs_log, s))
        probed = any(m_ == "POST" and p_.startswith("/mcp") and st == "401" for m_, p_, st in seen)
        prm = [p_ for m_, p_, st in seen if m_ == "GET" and "oauth-protected-resource" in p_]
        record("A1a mcp-login succeeds against simple-auth", label, ok, True, err.strip().splitlines()[-1] if err.strip() else "")
        res, rerr = agent.run(home)
        record("A1b run calls the protected tool with the token", label, tool_ok(res), True, (res or {}).get("text", rerr)[:120].replace("\n", " "))
        record("A2a login meets the server's 401 challenge", label, probed or any(m_ == "GET" and p_.startswith("/mcp") and st == "401" for m_, p_, st in seen), True,
               f"RS saw {seen[:6]}")
        record("A2b PRM fetched from the challenge's resource_metadata URL", label,
               bool(m) and urlparse(m.group(1)).path in prm, True, f"PRM GETs: {prm}")
        fields = stored_fields(home)
        record("A1c stored credential records its server URL", label, "resource" in (fields or []), pr, f"fields={fields}")

        # A3: the same name now points at B.
        b_log = os.path.join(LOGDIR, f"lookalike-B-{label}.jsonl")
        b_port = free_port()
        spawn([sys.executable, os.path.join(HERE, "lookalike.py"), str(b_port), b_log], f"lookalike-B-{label}.log")
        wait_port(b_port)
        b_url = f"http://localhost:{b_port}/mcp"
        Agent.write_settings(home, b_url)
        res, rerr = agent.run(home)
        blog = [json.loads(l) for l in open(b_log)] if os.path.exists(b_log) else []
        withauth = [e for e in blog if e["authorization_present"]]
        sent = bool(withauth)
        first = f'{withauth[0]["method"]} {withauth[0]["path"]} scheme={withauth[0]["scheme"]}' if withauth else "none"
        print(f"[{label}] A3 B received {len(blog)} requests, {len(withauth)} with an Authorization header; first such: {first}", flush=True)
        text = (res or {}).get("text", "") + " " + rerr
        record("A3a token sent to another URL under the same name", label, sent, not pr, f"B requests={blog[:4]}")
        if pr:
            record("A3b refusal names mcp-login", label, "mcp-login" in text, True, text.strip()[:200].replace("\n", " "))
            _, werr = agent.run(home, rust_log="warn")
            record("A3d with RUST_LOG=warn the specific refusal is shown (issued for the other URL)", label,
                   "was issued for" in werr and "not sent" in werr, True, "default logging shows only the generic `Auth required ... run mcp-login` hint")
            Agent.write_settings(home, normal.url)
            res, rerr = agent.run(home)
            record("A3c back at the original URL the login still works", label, tool_ok(res), True)

        # A4: legacy entry (written by main's binary) used by this binary.
        if pr and legacy_bin and legacy_bin != binary:
            h = agent.home(normal.url)
            ok, err = agent.login(h, normal.as_url, binary=legacy_bin)
            lf = stored_fields(h)
            res0, _ = agent.run(h, who=legacy_bin)
            s = logsize(normal.rs_log)
            res, rerr = agent.run(h)
            ok200 = [x for x in requests_in(logslice(normal.rs_log, s)) if x[2] == "200"]
            text = (res or {}).get("text", "") + " " + rerr
            record("A4a legacy entry (no resource) works on main's binary", label, tool_ok(res0), True, f"fields={lf}")
            record("A4b legacy entry refused, with a re-login message", label,
                   not tool_ok(res) and "mcp-login" in text and not ok200, True, text.strip()[:200].replace("\n", " "))
            _, werr = agent.run(h, rust_log="warn")
            record("A4d with RUST_LOG=warn the specific refusal is shown (not bound to a server URL)", label,
                   "not bound to a server URL" in werr and "not sent" in werr, True)
            ok, err = agent.login(h, normal.as_url)
            res, rerr = agent.run(h)
            record("A4c re-login then works", label, ok and tool_ok(res), True)

        # A5: AS omits PKCE metadata.
        h = agent.home(nopkce.url)
        ok, err = agent.login(h, nopkce.as_url)
        record("A5 login against an AS that omits S256 PKCE metadata proceeds", label, ok, not pr,
               err.strip().splitlines()[-1][:160] if err.strip() else "")
        if pr:
            record("A5b refusal names PKCE", label, "PKCE" in err, True)

        # A6: refresh after the 3 s access token expired.
        h = agent.home(refresh.url)
        ok, err = agent.login(h, refresh.as_url)
        s = logsize(refresh.as_log)
        res, rerr = agent.run(h)  # right after login: fresh token
        time.sleep(4.5)
        res2, rerr2 = agent.run(h)  # expired: must refresh
        grants = [x for x in requests_in(logslice(refresh.as_log, s)) if x[1].startswith("/token")]
        record("A6a login to a refresh-issuing AS", label, ok, True)
        record("A6b call after access-token expiry succeeds via refresh", label, tool_ok(res2), True,
               (res2 or {}).get("text", rerr2)[:100].replace("\n", " "))
        record("A6c AS saw a token-endpoint call for the refresh", label, len(grants) >= 1, True, f"/token calls after login: {len(grants)}")

    bad = [r_ for r_ in rows if r_[2] != r_[3]]
    print("\n| assertion | " + " | ".join(bins) + " |\n|---|" + "---|" * len(bins))
    for a in dict.fromkeys(r_[0] for r_ in rows):
        cells = []
        for label in bins:
            r_ = next((x for x in rows if x[0] == a and x[1] == label), None)
            cells.append("n/a" if r_ is None else f"{r_[2]}" + ("" if r_[2] == r_[3] else " (UNEXPECTED)"))
        print(f"| {a} | " + " | ".join(cells) + " |")
    print(f"\nlogs: {LOGDIR}\n{len(rows) - len(bad)}/{len(rows)} observations as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except OSError:
                    pass
        time.sleep(0.5)
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
    sys.exit(code)
