#!/usr/bin/env python3
"""Job output I/O: read-back speed from an offset, and what a job writing flat out does to the loop.

    benches/jobs_io.py --bin /path/to/execd [--env K=V ...] [--cpus 0,1]

Prints JSON: (1) one job that has written 64 MiB (8 MiB retained): the speed of `read` with max 8 MiB
from the oldest offset held and of an `attach` stream from there, in MB/s of decoded output;
(2) the latency of `true` (v1.1, sequential) while 16 jobs each write 1 GiB as fast as they can,
against the same request on an idle server. Standard library only.
"""
import argparse
import base64
import json
import os
import socket
import statistics
import subprocess
import tempfile
import threading
import time


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(port, body, until_len=True):
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps(body).encode()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
    data = b""
    while b"\r\n\r\n" not in data:
        data += s.recv(1 << 20)
    head, _, rest = data.partition(b"\r\n\r\n")
    n = int([l for l in head.lower().split(b"\r\n") if l.startswith(b"content-length")][0].split(b":")[1])
    chunks = [rest]
    got = len(rest)
    while got < n:
        d = s.recv(1 << 20)
        if not d:
            break
        chunks.append(d)
        got += len(d)
    s.close()
    return json.loads(b"".join(chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--cpus", default="")
    a = ap.parse_args()
    port = free_port()
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    cmd = (["taskset", "-c", a.cpus] if a.cpus else []) + [a.bin, "--listen", f"127.0.0.1:{port}"]
    with tempfile.TemporaryDirectory() as root:
        cmd += ["--root", root, "--home", root]
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        try:
            for _ in range(500):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.02)
            out = {"bin": a.bin, "env": a.env, "cpus": a.cpus}

            def req(cmd, args, **extra):
                r = {"command": cmd, "args": args, "cwd": root, "timeout_ms": 600000}
                r.update(extra)
                return r

            def job(jid, size, op="start"):
                r = req("head", ["-c", str(size), "/dev/zero"])
                r["job"] = {"op": op, "scope": "j", "id": jid, "hold": False, "label": ""}
                return r

            def jobop(jid, op, **kw):
                r = req("true", [])
                r["job"] = dict({"op": op, "scope": "j", "id": jid}, **kw)
                return r

            # (1) read-back from an offset.
            call(port, job("r0", 64 << 20))
            while call(port, jobop("r0", "read", offset=0, max=1))["state"] != "ended":
                time.sleep(0.1)
            first = call(port, jobop("r0", "read", offset=0, max=1))["offset"]
            rates = []
            for _ in range(5):
                t = time.time()
                v = call(port, jobop("r0", "read", offset=first, max=8 << 20))
                n = len(base64.b64decode(v["data"]))
                rates.append(n / 1e6 / (time.time() - t))
            out["read_8MiB_from_oldest_MB_s_median"] = statistics.median(rates)
            out["read_bytes"] = n
            # attach: a stream from the oldest offset to the end.
            arates = []
            for _ in range(3):
                s = socket.create_connection(("127.0.0.1", port))
                b = json.dumps(dict(jobop("r0", "attach", offset=first), stream=True)).encode()
                s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
                t = time.time()
                total = 0
                tail = b""
                while True:
                    d = s.recv(1 << 20)
                    if not d:
                        break
                    total += len(d)
                    tail = (tail + d)[-200:]
                    if b"exit_code" in tail:
                        break
                s.close()
                arates.append(total * 0.75 / 1e6 / (time.time() - t))
            out["attach_from_oldest_MB_s_median(approx, wire bytes*3/4)"] = statistics.median(arates)

            # (2) the loop under 16 writers.
            def lat(n):
                xs = []
                for _ in range(n):
                    t = time.time()
                    call(port, req("true", []))
                    xs.append((time.time() - t) * 1e3)
                xs.sort()
                return {"p50_ms": xs[len(xs) // 2], "p99_ms": xs[int(len(xs) * 0.99)], "max_ms": xs[-1], "n": n}

            out["true_idle"] = lat(300)
            for i in range(16):
                call(port, job(f"w{i}", 1 << 30))
            time.sleep(0.5)
            out["true_while_16_jobs_write_flat_out"] = lat(300)
            print(json.dumps(out, indent=1))
        finally:
            p.kill()
            p.wait()
            subprocess.run(["pkill", "-9", "-x", "head"])


if __name__ == "__main__":
    main()
