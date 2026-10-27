#!/usr/bin/env python3
"""Runs the exec front conformance kit (crates/agent, `exec_front_conformance`) against front.py.

    python3 deploy/demo/front_conformance.py        # needs cargo; no E2B, no network beyond localhost

front.py is the demo's stand-in for Beyond's edge plus sandbox-exec. Here its substrate is local: a
"VM" is a Python process serving a small exec endpoint on a loopback port, made by a fake e2b.sh
(`vm_for`, `e2b_kill`, `vm_run`) in place of the real one, and killed with SIGKILL. The kit's operator
hook (BEYOND.md "Conformance"; the hook table is in `exec_endpoint::conformance::front`) is this file's
`hook` subcommand, which sees the substrate through its state directory:

    token <tenant> <session>   a header; front.py has no tokens, so it ignores it
    state <tenant>             {"vms": live VM processes of the tenant's sessions,
                                "routes": front.py's record files (VMDIR/<session>.json) for them}
    fail-next-create ...       the next seeding of the session's VM fails, after the VM exists
    kill-vm / end ...          SIGKILL the session's VM process

The run passes when the kit's failures are exactly KNOWN_GAPS: what front.py, a demo, does not do. A
gap that closes fails the run too, so the list cannot go stale.
"""
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# What front.py does not do, by the kit's check names (the checks are in the kit, not here).
KNOWN_GAPS = {
    # The demo front has no tenants: its only bound is the in-flight request cap.
    "the tenant's N+1th VM is 503 with Retry-After",
    # It learns that a VM is gone only when a request finds it, then recovers on that request.
    "a VM that is gone leaves no route entry",
    # It has no agent token: it forwards to the VM with the VM's own bearer and checks nothing itself.
    "a missing token is refused",
    "a garbage token is refused",
    "a token for sandbox A is refused at sandbox B",
    "a token below the sandbox's epoch is refused",
}

FAKE_E2B = r'''
vm_for() { # session
  local s=$1 f="$VMDIR/$1.json" pid bearer i
  [ -f "$f" ] && return 0
  bearer=$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')
  python3 -I "$HARNESS" substrate "$STATE" "$s" "$bearer" >/dev/null 2>&1 </dev/null &
  pid=$!
  for i in $(seq 1 100); do [ -f "$STATE/vms/$pid.json" ] && break; sleep 0.1; done
  [ -f "$STATE/vms/$pid.json" ] || { kill -KILL "$pid" 2>/dev/null; return 1; }
  jq -nc --arg id "vm$pid" --arg b "$bearer" --argjson port "$(jq .port "$STATE/vms/$pid.json")" \
    '{id:$id, envd_token:"t", bearer:$b, traffic_token:"", url:("http://127.0.0.1:"+($port|tostring)+"/exec")}' > "$f"
}
vm_field() { jq -r ".$2" "$VMDIR/$1.json"; }
e2b_kill() { rm -f "$STATE/vms/${1#vm}.json"; kill -KILL "${1#vm}" 2>/dev/null; return 0; }
# Seeding: fails once if the hook asked for it, after the VM exists (the case a failed create must clean up).
vm_run() {
  local s; s=$(jq -r .session "$STATE/vms/${1#vm}.json")
  [ -f "$STATE/fail/$s" ] && { rm -f "$STATE/fail/$s"; return 1; }
  echo seeded
}
vm_browser_start() { :; }
vm_site_start() { :; }
'''


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


# --- the substrate: one "VM" ---------------------------------------------------------------------


def substrate(state, session, bearer):
    class Exec(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def answer(self, status, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.headers.get("Authorization") != "Bearer " + bearer:
                return self.answer(401, {"error": "bearer required"})
            try:
                req = json.loads(body)
                assert isinstance(req.get("args"), list)
            except (ValueError, AssertionError):
                return self.answer(400, {"error": "bad request"})
            job = req.get("job")
            if not isinstance(req.get("stream", False), bool) or (job is not None and not isinstance(job, dict)):
                return self.answer(400, {"error": "unsupported extension"})
            if job is not None and (job.get("op") not in ("start", "keep") or not isinstance(job.get("notify", {}), dict)):
                return self.answer(422, {"error": "unsupported extension"})
            r = subprocess.run([req["command"]] + req["args"], capture_output=True)
            if req.get("stream"):
                frames = [{"stdout": __import__("base64").b64encode(r.stdout).decode()}, {"exit_code": r.returncode}]
                return self.answer(200, "".join(json.dumps(f) + "\n" for f in frames).encode(), "application/x-ndjson")
            b64 = __import__("base64").b64encode
            self.answer(200, {"exit_code": r.returncode, "stdout": b64(r.stdout).decode(), "stderr": b64(r.stderr).decode()})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Exec)
    srv.daemon_threads = True
    tmp = f"{state}/vms/{os.getpid()}.part"
    with open(tmp, "w") as f:
        json.dump({"pid": os.getpid(), "session": session, "port": srv.server_port}, f)
    os.rename(tmp, f"{state}/vms/{os.getpid()}.json")
    srv.serve_forever()


# --- the operator hook ---------------------------------------------------------------------------


def hook(state, op, args):
    vms = []
    for name in os.listdir(f"{state}/vms"):
        if name.endswith(".json"):
            with open(f"{state}/vms/{name}") as f:
                vms.append(json.load(f))
    if op == "token":
        tenant, session = args
        with open(f"{state}/tenants/{tenant}", "a") as f:
            f.write(session + "\n")
        print(f"Authorization: Bearer demo.{tenant}.{session}")
    elif op == "state":
        try:
            with open(f"{state}/tenants/{args[0]}") as f:
                sessions = set(f.read().split())
        except FileNotFoundError:
            sessions = set()
        vmdir = f"{state}/vmdir"
        print(json.dumps({
            "vms": sum(1 for v in vms if v["session"] in sessions and alive(v["pid"])),
            "routes": sum(1 for s in sessions if os.path.exists(f"{vmdir}/{s}.json")),
        }))
    elif op == "fail-next-create":
        open(f"{state}/fail/{args[1]}", "w").close()
    elif op in ("kill-vm", "end"):
        for v in vms:
            if v["session"] == args[1]:
                os.kill(v["pid"], signal.SIGKILL)
                for _ in range(40):
                    if not alive(v["pid"]):
                        break
                    time.sleep(0.05)
    else:
        sys.exit(f"unknown hook op {op!r}")


# --- the run -------------------------------------------------------------------------------------


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run():
    front_py = os.environ.get("FRONT_PY", os.path.join(HERE, "front.py"))
    with tempfile.TemporaryDirectory(prefix="front-conformance-") as tmp:
        state = os.path.join(tmp, "state")
        for d in ("vms", "vmdir", "tenants", "fail"):
            os.makedirs(os.path.join(state, d))
        e2b = os.path.join(tmp, "e2b.sh")
        with open(e2b, "w") as f:
            f.write(FAKE_E2B)
        port = free_port()
        env = dict(os.environ, FRONT_PORT=str(port), FRONT_VMDIR=f"{state}/vmdir", FRONT_E2B_SH=e2b,
                   FRONT_CA_FILE=f"{tmp}/ca.pem", STATE=state, HARNESS=os.path.abspath(__file__),
                   FRONT_CONFORMANCE_STATE=state)
        # Its own process group: the front and every VM it made die with it, and nothing else does.
        front = subprocess.Popen([sys.executable, front_py], env=env, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).read()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                sys.exit("front.py did not start")
            env.update(
                AGENT_CONFORMANCE_FRONT_URL=f"http://127.0.0.1:{port}/exec/{{session}}?bundle=http://127.0.0.1:1/none.tar",
                AGENT_CONFORMANCE_FRONT_HOOK=f"{shlex.quote(sys.executable)} -I {shlex.quote(os.path.abspath(__file__))} hook",
                AGENT_CONFORMANCE_FRONT_TENANT_CAP="3",
            )
            out = subprocess.run(
                ["cargo", "test", "-p", "beyond-agent", "--test", "exec_front_conformance",
                 "the_configured_front_conforms", "--", "--nocapture"],
                cwd=ROOT, env=env, capture_output=True, text=True).stdout
        finally:
            os.killpg(front.pid, signal.SIGKILL)
            front.wait()
    print(out)
    failed = set(re.findall(r"^\s+FAIL\s+(.*?): ", out, re.M))
    if not re.search(r"^\s+(PASS|FAIL)\s", out, re.M):
        sys.exit("the kit did not run")
    new, closed = failed - KNOWN_GAPS, KNOWN_GAPS - failed
    for name in sorted(new):
        print(f"NEW FAILURE: {name}")
    for name in sorted(closed):
        print(f"GAP CLOSED (drop it from KNOWN_GAPS): {name}")
    if new or closed:
        sys.exit(1)
    print(f"front.py: {len(KNOWN_GAPS)} known gaps, no other failure")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "substrate":
        substrate(*sys.argv[2:5])
    elif cmd == "hook":
        hook(os.environ["FRONT_CONFORMANCE_STATE"], sys.argv[2], sys.argv[3:])
    else:
        run()
