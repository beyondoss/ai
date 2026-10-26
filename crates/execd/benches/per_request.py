#!/usr/bin/env python3
"""What one request costs execd: its own CPU per v1.1 `true`, and the latency, with and without --run-as.

    benches/per_request.py --bin /path/to/execd [--n 2000] [--run-as nobody]

Runs N sequential `true` requests on a kept-alive connection (after 100 warm-ups) against a server
started with the given flags, and prints JSON: the server's own user+system CPU per request in
microseconds (its commands' CPU is not included), and p50/p99 round trip in microseconds.
`--run-as` starts the server through `sudo -n` and needs passwordless sudo. Standard library only.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--run-as", default="")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--log-level", default="off")
    a = ap.parse_args()
    port = free_port()
    root = tempfile.mkdtemp()
    os.chmod(root, 0o777)
    cmd = []
    if a.run_as:
        cmd += ["sudo", "-n"]
    cmd += [a.bin, "--listen", f"127.0.0.1:{port}", "--root", root, "--home", root, "--log-level", a.log_level]
    if a.run_as:
        cmd += ["--run-as", a.run_as]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(500):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.02)
        body = {"command": "true", "args": [], "cwd": root, "timeout_ms": 60000}
        if a.stream:
            body["stream"] = True
        b = json.dumps(body).encode()
        req = b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b
        s = socket.create_connection(("127.0.0.1", port))

        def once():
            s.sendall(req)
            data = b""
            while True:
                data += s.recv(65536)
                if a.stream:
                    if data.endswith(b"0\r\n\r\n"):
                        return
                elif b"exit_code" in data:
                    return

        for _ in range(100):
            once()
        # The server's pid: under sudo, the child of the sudo process; find by listening port.
        pid = None
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    if os.readlink(f"/proc/{d}/exe").endswith(os.path.basename(a.bin)) and \
                            any(f":{port:04X}" in l for l in open(f"/proc/{d}/net/tcp") if " 0A " in l):
                        pid = int(d)
                        break
                except OSError:
                    pass

        def cpu():
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")

        c0 = cpu() if pid else 0
        lat = []
        for _ in range(a.n):
            t = time.perf_counter()
            once()
            lat.append((time.perf_counter() - t) * 1e6)
        c1 = cpu() if pid else 0
        lat.sort()
        print(json.dumps({
            "bin": a.bin, "run_as": a.run_as, "stream": a.stream, "n": a.n,
            "server_cpu_us_per_request": round((c1 - c0) * 1e6 / a.n, 1),
            "p50_us": round(lat[len(lat) // 2]), "p99_us": round(lat[int(len(lat) * 0.99)]),
        }))
    finally:
        subprocess.run(["sudo", "-n", "kill", str(p.pid)] if a.run_as else ["kill", str(p.pid)])
        p.wait()


if __name__ == "__main__":
    main()
