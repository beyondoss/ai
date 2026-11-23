#!/usr/bin/env python3
"""First-byte latency of a streamed job: the time from sending a held `start` to the first frame that
carries the job's output, and of a streamed (non-job) command as the control; and the time from a
running job's write to its arrival at an attached client (`date +%s%N` prints the clock at the write,
the client reads the same clock on arrival: one host, one clock).

    benches/jobs_first_byte.py --bin /path/to/execd [--n 40] [--env K=V ...]

A job's output reaches a client through the job's spool file (execd tails it); a streamed command's
through a pipe. Prints JSON: p50 and p95 in ms over `n` runs of each. Standard library only.
"""
import argparse
import json
import os
import socket
import subprocess
import tempfile
import time


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(port, body):
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps(body).encode()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
    data = b""
    while b"\r\n\r\n" not in data:
        data += s.recv(1 << 16)
    head, _, rest = data.partition(b"\r\n\r\n")
    n = int([l for l in head.lower().split(b"\r\n") if l.startswith(b"content-length")][0].split(b":")[1])
    while len(rest) < n:
        rest += s.recv(1 << 16)
    s.close()
    return json.loads(rest)


def first_byte(port, body, needle=b"stdout"):
    """Seconds from sending `body` until the answer holds `needle`."""
    s = socket.create_connection(("127.0.0.1", port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    b = json.dumps(body).encode()
    t = time.perf_counter()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
    seen = b""
    while needle not in seen:
        d = s.recv(1 << 16)
        if not d:
            raise SystemExit(f"no {needle!r} in {seen!r}")
        seen += d
    dt = time.perf_counter() - t
    s.close()
    return dt


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    port = free_port()
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    env["EXECD_LISTEN"] = f"127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as root:
        env["EXECD_ROOT"] = env["EXECD_HOME"] = root
        p = subprocess.Popen([a.bin], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        try:
            for _ in range(500):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.02)
            # Output at once, then the command stays alive past the first byte.
            script = "echo x; sleep 0.3"

            def req(i, job):
                r = {"command": "sh", "args": ["-c", script], "cwd": root, "timeout_ms": 60000, "stream": True}
                if job:
                    r["job"] = {"op": "start", "scope": "b", "id": f"j{i}", "hold": True, "label": ""}
                return r

            out = {"bin": a.bin, "n": a.n}
            for name, job in (("streamed_command", False), ("held_job_start", True)):
                for i in range(5):  # warm up
                    first_byte(port, req(f"w{name}{i}", job))
                xs = [first_byte(port, req(f"{name}{i}", job)) for i in range(a.n)]
                out[name] = {"p50_ms": round(pct(xs, 0.5), 3), "p95_ms": round(pct(xs, 0.95), 3), "n": a.n}
            # A job that is running and quiet writes once; a client attached to it times the arrival.
            xs = []
            for i in range(a.n + 5):
                jid = f"a{i}"
                start = {"command": "sh", "args": ["-c", "sleep 0.3; date +%s%N; sleep 5"], "cwd": root,
                         "timeout_ms": 60000,
                         "job": {"op": "start", "scope": "b", "id": jid, "hold": False, "label": ""}}
                call(port, start)
                att = {"command": "true", "stream": True,
                       "job": {"op": "attach", "scope": "b", "id": jid, "offset": 0}}
                s = socket.create_connection(("127.0.0.1", port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                b = json.dumps(att).encode()
                s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
                seen = b""
                while b"stdout" not in seen:
                    seen += s.recv(1 << 16)
                got = time.time_ns()
                s.close()
                # The frame is base64 of "<digits>\n".
                import base64, re
                m = re.search(rb'"stdout":"([A-Za-z0-9+/=]+)"', seen)
                wrote = int(base64.b64decode(m.group(1)).strip())
                if i >= 5:
                    xs.append((got - wrote) / 1e9)
                call(port, {"command": "true", "job": {"op": "kill", "scope": "b", "ids": [jid], "grace_ms": 0}})
                call(port, {"command": "true", "job": {"op": "remove", "scope": "b", "ids": [jid]}})
            out["attach_write_to_client"] = {"p50_ms": round(pct(xs, 0.5), 3), "p95_ms": round(pct(xs, 0.95), 3), "n": a.n}
            print(json.dumps(out, indent=1))
        finally:
            p.kill()
            p.wait()


if __name__ == "__main__":
    main()
