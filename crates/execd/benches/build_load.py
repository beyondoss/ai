#!/usr/bin/env python3
"""A build-like load against one server, and what it costs the server.

    benches/build_load.py --bin /path/to/execd [--n 5000] [--strace]

One stream holds a leftover (`sh -c 'sleep 3600 &'`, a kept dev server); another runs a shell that
starts N short processes whose parents exit at once (`(/bin/true &)`), so each one is an orphan that
execd adopts and sees exit, as a compiler's helpers do. Prints JSON: the server's own CPU
(user + system) for the whole load, and with `--strace` the number of syscalls the server made.
Standard library only.
"""
import argparse
import json
import os
import re
import signal
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


def stream(port, root, script):
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps({"command": "sh", "args": ["-c", script], "cwd": root, "timeout_ms": 600000,
                    "stream": True}).encode()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--strace", action="store_true")
    a = ap.parse_args()
    port = free_port()
    root = tempfile.mkdtemp()
    p = subprocess.Popen([a.bin, "--listen", f"127.0.0.1:{port}", "--root", root, "--home", root, "--log-level", "off"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(500):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.02)
        hold = stream(port, root, "sleep 3600 &")
        time.sleep(1)

        def cpu():
            f = open(f"/proc/{p.pid}/stat").read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")

        st = None
        if a.strace:
            out = f"/tmp/build_load_strace_{p.pid}.txt"
            st = subprocess.Popen(["strace", "-c", "-p", str(p.pid), "-o", out], stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        c0 = cpu()
        t0 = time.time()
        load = stream(port, root, f"i=0; while [ $i -lt {a.n} ]; do (/bin/true &); i=$((i+1)); done; sleep 2")
        buf = b""
        load.settimeout(300)
        while b"exit_code" not in buf:
            d = load.recv(65536)
            if not d:
                break
            buf += d
        wall = time.time() - t0
        c1 = cpu()
        res = {"bin": a.bin, "n": a.n, "wall_s": round(wall, 2), "server_cpu_s": round(c1 - c0, 3)}
        if st:
            st.send_signal(signal.SIGINT)
            st.wait()
            txt = open(out).read()
            m = re.search(r"\n\s*100\.00\s+[\d.]+\s+\S*\s+(\d+)\s+\d*\s*total", txt)
            res["server_syscalls"] = int(m.group(1)) if m else None
        print(json.dumps(res))
    finally:
        p.kill()
        p.wait()


if __name__ == "__main__":
    main()
