#!/usr/bin/env python3
"""The demo's exec front: a session's sandbox that does not exist until the session needs it.

A grant's `exec_url` has to be a stable address, but an E2B microVM costs something from the moment it
exists. This front is the stable address: `http://<front>:8084/exec/<session>?bundle=<tar url>`. It
answers nothing itself. On the FIRST request for a session it creates that session's microVM (the same
`vm_for` the rest of the demo uses: a private E2B sandbox running the exec stub, requiring a bearer),
seeds the agent folder into it from the bundle, and from then on forwards every request to the VM,
streaming the answer back as it arrives. A session that never calls a sandbox tool never costs a VM.

A sandbox MCP server is fronted the same way: `http://<front>:8084/mcp/<session>?bundle=<tar url>&browser=<url
of the rustwright-mcp binary>&site=<dir>` is the connector's stable address. The first request for a
session creates its VM like an exec request does and, because it is an MCP request, also installs the
browser stack and starts the bridge (`vm_browser_up`, `vm_bridge_start`) before forwarding to it. The
agent advertises the server's tools from a list saved in its bundle, so nothing dials this address (and
no VM exists) until the model first calls one of them.

One JSON line per event goes to stdout (CloudWatch): `first_request`, `vm_create_start`, `vm_ready`,
`vm_seeded`, `browser_ready`, and `exec_request` / `mcp_request` for every request forwarded. The E2B key comes from the environment
and is only ever handed to the helper functions in e2b.sh.
"""
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PORT = int(os.environ.get("FRONT_PORT", "8084"))
# Where things are in the image; a test run elsewhere points these at its own copies.
VMDIR = os.environ.get("FRONT_VMDIR", "/tmp/vm")
E2B_SH = os.environ.get("FRONT_E2B_SH", "/opt/demo/e2b.sh")
STUB = os.environ.get("FRONT_STUB", "/usr/local/bin/beyond-ai-demo")
CA_FILE = os.environ.get("FRONT_CA_FILE", "/tmp/ca.pem")
# A VM dies on its own after E2B_TIMEOUT (e2b.sh); an entry older than that points at nothing.
VM_TTL = int(os.environ.get("E2B_TIMEOUT", "1800"))
# An idle or stalled client holds a thread until it speaks; this bounds it. The agent's exec calls are
# short requests (the long wait is the upstream stream, which this does not time), so two minutes is
# generous.
CLIENT_TIMEOUT = float(os.environ.get("FRONT_CLIENT_TIMEOUT", "120"))
# Requests in flight at once, VM creation and streaming included. A demo runs a handful of sessions;
# 64 leaves room for retries and refuses a flood instead of spawning a thread and a VM for each.
MAX_INFLIGHT = int(os.environ.get("FRONT_MAX_INFLIGHT", "64"))
inflight = threading.BoundedSemaphore(MAX_INFLIGHT)
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8931"))
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "host", "content-length"}

sessions = {}  # session -> {"lock": Lock, "vm": dict | None, "born": float}
sessions_lock = threading.Lock()


def now_ms():
    return int(time.time() * 1000)


def log(event, **detail):
    print(json.dumps({"ts_ms": now_ms(), "src": "front", "event": event, "detail": detail}), flush=True)


# Runs in bash so the creation is the demo's own tested code. The folder is fetched by the VM itself
# (a presigned URL), so the bytes never pass through here.
SCRIPT = r'''
. "$E2B_SH"
BRIDGE_PORT=${BRIDGE_PORT:-8931}
vm_for "$1" || exit 1
id=$(vm_field "$1" id); tok=$(vm_field "$1" envd_token)
echo READY
[ -z "$3" ] || vm_browser_warm "$id" "$tok"
# The browser stack and the bridge come up while the bundle is being fetched: they do not need it.
if [ -n "$3" ]; then
  { vm_browser_up "$id" "$tok" "$3" && vm_bridge_start "$id" "$tok"; } & bpid=$!
fi
if [ -n "$2" ]; then
  vm_run "$id" "$tok" "mkdir -p $VM_HOME/workspace && curl -fsS -m 60 -o $VM_HOME/bundle.tar '$2' && tar -xf $VM_HOME/bundle.tar -C $VM_HOME/workspace && rm -f $VM_HOME/bundle.tar && echo seeded" | grep -q seeded || exit 3
fi
echo SEEDED
if [ -n "$3" ]; then
  wait "$bpid" || exit 4
  # The page it serves is in the bundle, so it starts once the bundle is there.
  [ -z "$4" ] || vm_site_start "$id" "$tok" "$VM_HOME/workspace/$4"
  echo BROWSER
fi
'''


def make_vm(session, bundle, browser="", site=""):
    env = dict(os.environ, VMDIR=VMDIR, STUB=STUB, CA_FILE=CA_FILE, E2B_SH=E2B_SH,
               VM_PRIVATE="1", TAG_KEY="beyond-ai-demo", SANDBOX_PORT="8081")
    env.setdefault("NAME", os.environ.get("DEMO_NAME", "demo"))
    if browser and os.environ.get("E2B_BROWSER_TEMPLATE"):
        env["E2B_TEMPLATE"] = os.environ["E2B_BROWSER_TEMPLATE"]  # the browser stack is already in it
    log("vm_create_start", session=session)
    p = subprocess.Popen(["bash", "-c", SCRIPT, "front", session, bundle or "", browser, site], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.strip() == "READY":
            log("vm_ready", session=session)
        elif line.strip() == "SEEDED":
            log("vm_seeded", session=session, bundle=bool(bundle))
        elif line.strip() == "BROWSER":
            log("browser_ready", session=session)
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError(f"creating the sandbox failed ({p.returncode}): {err[-300:]}")
    with open(f"{VMDIR}/{session}.json") as f:
        return json.load(f)


def drop(session, s, kill=True):
    """Forget a session's VM (and kill it, best effort) so the next request makes a fresh one."""
    with sessions_lock:
        if sessions.get(session) is s:
            del sessions[session]
    vm, s["vm"] = s["vm"], None
    try:
        os.remove(f"{VMDIR}/{session}.json")
    except FileNotFoundError:
        pass
    if kill and vm:
        env = dict(os.environ, E2B_SH=E2B_SH, VMDIR=VMDIR, VM_PRIVATE="1", SANDBOX_PORT="8081", TAG_KEY="", NAME="")
        subprocess.run(["bash", "-c", '. "$E2B_SH"; e2b_kill "$1"', "front", vm["id"]], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
    log("vm_dropped", session=session)


def vm_for(session, bundle, browser="", site=""):
    with sessions_lock:
        old = [(k, v) for k, v in sessions.items() if v["vm"] and time.time() - v["born"] > VM_TTL]
    for k, v in old:  # past its timeout: E2B has already killed it
        drop(k, v, kill=False)
    with sessions_lock:
        s = sessions.setdefault(session, {"lock": threading.Lock(), "vm": None, "born": 0.0})
    with s["lock"]:
        if s["vm"] is None:
            log("first_request", session=session)
            s["vm"] = make_vm(session, bundle, browser, site)
            s["born"] = time.time()
        return s["vm"], s


class Front(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = CLIENT_TIMEOUT

    def log_message(self, *a):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {"ok": True}) if self.path == "/healthz" else self.reply(404, {"error": "not found"})

    def do_DELETE(self):  # ends an MCP session; same path as a POST
        self.do_POST()

    def do_POST(self):
        if not inflight.acquire(blocking=False):
            self.close_connection = True  # the body is unread
            return self.reply(503, {"error": "too many requests in flight"})
        try:
            self.forward()
        finally:
            inflight.release()

    def forward(self):
        u = urlsplit(self.path)
        m = re.fullmatch(r"/(exec|mcp)/([A-Za-z0-9._-]+)", u.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not m:
            return self.reply(404, {"error": "POST /exec/<session> or /mcp/<session>"})
        kind, session = m.groups()
        q = parse_qs(u.query)
        bundle = (q.get("bundle") or [""])[0]
        browser, site = "", ""
        if kind == "mcp":
            browser = (q.get("browser") or [""])[0]
            site = (q.get("site") or [""])[0]
            if (browser and not browser.startswith("https://")) or not re.fullmatch(r"[A-Za-z0-9._/-]*", site) or ".." in site:
                return self.reply(400, {"error": "browser must be an https URL; site a relative directory"})
        try:
            vm, entry = vm_for(session, bundle, browser, site)
        except Exception as e:  # the next request tries again
            with sessions_lock:
                sessions.pop(session, None)
            log("vm_failed", session=session, error=str(e))
            return self.reply(502, {"error": str(e)})
        log("mcp_request" if kind == "mcp" else "exec_request", session=session)
        # An MCP request goes to the bridge's port on the same VM, which takes no bearer; an exec
        # request goes to the exec stub, which does.
        up = urlsplit(f"https://{BRIDGE_PORT}-{vm['id']}.e2b.app/mcp" if kind == "mcp" else vm["url"])
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP | {"authorization"}}
        if kind == "exec":
            headers["Authorization"] = "Bearer " + vm["bearer"]
        if vm.get("traffic_token"):
            headers["e2b-traffic-access-token"] = vm["traffic_token"]
        conn = http.client.HTTPSConnection(up.hostname, timeout=3700)
        try:
            conn.connect()
        except OSError as e:  # the VM is gone (past its timeout, or killed): the next request makes a new one
            log("proxy_failed", session=session, error=str(e))
            drop(session, entry)
            return self.reply(502, {"error": str(e)})
        try:
            conn.request(self.command, up.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log("proxy_failed", session=session, error=str(e))
            try:
                self.reply(502, {"error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()


def main():
    os.makedirs(VMDIR, exist_ok=True)
    with open(CA_FILE, "w") as f:
        f.write(os.environ.get("CA_PEM", ""))
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Front)
    srv.daemon_threads = True
    log("listening", port=PORT)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
