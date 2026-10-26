#!/usr/bin/env python3
"""PGO training workload for `execd`: python3 .github/pgo-train-execd.py <instrumented execd> <profdir>

Drives a PGO-instrumented build through what a sandbox's execd does: v1.1 commands (with and
without stdin), a v1.2 stream, a kept v1.3 job (start, read, list, kill), a command that leaves a
process behind, and a client that disconnects. The point is the layout of the code that runs, which
decides resident text (see `.github/workflows/release-agent.yml`).

execd is stopped with SIGTERM, its graceful shutdown, because the profiling runtime writes the
profile at exit. Fails if no profile was written.
"""
import base64
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(port, body, read_all=True, wait=None):
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps(body).encode()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(b) + b)
    data = b""
    s.settimeout(10)
    try:
        while True:
            d = s.recv(65536)
            if not d:
                break
            data += d
            if not read_all or (wait and wait in data):
                break
            if b"\r\n\r\n" in data and b"content-length" in data.lower():
                head, _, rest = data.partition(b"\r\n\r\n")
                n = int([l for l in head.lower().split(b"\r\n") if l.startswith(b"content-length")][0].split(b":")[1])
                if len(rest) >= n:
                    break
    except socket.timeout:
        pass
    s.close()
    return data


def req(cmd, args, cwd, **extra):
    r = {"command": cmd, "args": args, "cwd": cwd, "timeout_ms": 30000}
    r.update(extra)
    return r


def main():
    exe, profdir = sys.argv[1], sys.argv[2]
    os.makedirs(profdir, exist_ok=True)
    port = free_port()
    with tempfile.TemporaryDirectory() as root:
        open(os.path.join(root, "f"), "wb").write(b"x" * 4096)
        # Configured by environment, which every execd build reads (flags are going away).
        env = dict(os.environ, LLVM_PROFILE_FILE=os.path.join(profdir, "execd-%p.profraw"),
                   EXECD_LISTEN=f"127.0.0.1:{port}", EXECD_ROOT=root, EXECD_HOME=root,
                   EXECD_KEPT_JOBS_SURVIVE_IDLE="true")
        p = subprocess.Popen([exe], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        try:
            for _ in range(200):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.02)
            call(port, req("true", [], root))  # warm
            for _ in range(200):
                call(port, req("cat", ["f"], root))
                call(port, req("cat", [], root, stdin_base64=base64.b64encode(b"hello").decode()))
            call(port, req("sh", ["-c", "echo err >&2; exit 3"], root))
            call(port, req("no-such-command", [], root))
            # v1.2 streams, finishing and abandoned.
            for _ in range(20):
                call(port, req("head", ["-c", "1048576", "/dev/zero"], root, stream=True))
            call(port, req("sh", ["-c", "printf a; sleep 30"], root, stream=True), wait=b'"a"')
            call(port, req("sh", ["-c", "printf a; sleep 30"], root, stream=True), wait=b"YQ==")
            # A command that leaves a process behind, then a disconnect.
            call(port, req("sh", ["-c", "sleep 1 & echo up"], root, stream=True))
            # Kept jobs.
            for i in range(4):
                jid = f"{i:02x}"
                call(port, req("sh", ["-c", "echo job; sleep 30"], root,
                               job={"op": "start", "scope": "t", "id": jid, "hold": False, "label": "x"}))
                time.sleep(0.1)
                call(port, req("true", [], root, job={"op": "read", "scope": "t", "id": jid, "offset": 0, "max": 100}))
                call(port, req("true", [], root, job={"op": "list", "scope": "t"}))
                call(port, req("true", [], root, job={"op": "kill", "scope": "t", "id": jid, "remove": True}))
            call(port, {"command": "true", "args": [], "timeout_ms": 1000})  # probe-ish
            s = socket.create_connection(("127.0.0.1", port))
            s.sendall(b"GET /livez HTTP/1.1\r\nHost: t\r\n\r\n")
            s.recv(1000)
            s.close()
        finally:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
    if not glob.glob(os.path.join(profdir, "*.profraw")):
        sys.exit("no profile was written")


if __name__ == "__main__":
    main()
