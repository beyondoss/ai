#!/usr/bin/env python3
"""Wakeups of an idle endpoint: context switches over a window, three ways.

    benches/wakeups.py --bin /path/to/execd [--env K=V ...] [--secs 60]

Starts three servers at once and leaves each in a state: (1) idle, (2) one open stream running
`sleep 3600`, (3) one open stream whose shell exited leaving `sleep 3600` behind (a kept dev
server). Over the window it reads the voluntary and involuntary context switches of every thread
(`/proc/<pid>/task/*/status`); each is a wakeup of the process. Prints JSON: switches per second.
Standard library only.
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


def switches(pid):
    total = 0
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            for line in open(f"/proc/{pid}/task/{t}/status"):
                if "ctxt_switches" in line:
                    total += int(line.split()[1])
        except OSError:
            pass
    return total


def threads(pid):
    return len(os.listdir(f"/proc/{pid}/task"))


def start(a, root):
    port = free_port()
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    p = subprocess.Popen([a.bin, "--listen", f"127.0.0.1:{port}", "--root", root, "--home", root],
                         env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for _ in range(500):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return p, port
        except OSError:
            time.sleep(0.02)
    raise SystemExit("did not start")


def stream(port, root, cmd, args):
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps({"command": cmd, "args": args, "cwd": root, "timeout_ms": 3600000,
                    "stream": True}).encode()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: w\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
    s.recv(4096)
    return s


def kill_tree(pid):
    """Kill a server and the process groups of its children, and nothing else."""
    kids = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = open(f"/proc/{d}/stat").read()
                if int(st.rsplit(")", 1)[1].split()[1]) == pid:
                    kids.append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    os.kill(pid, 9)
    for k in kids:
        try:
            os.killpg(k, 9)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--secs", type=int, default=60)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as root:
        servers = {}
        keep = []
        for name in ("idle", "one_stream", "one_leftover"):
            servers[name] = start(a, root)
        keep.append(stream(servers["one_stream"][1], root, "sleep", ["3600"]))
        keep.append(stream(servers["one_leftover"][1], root, "sh", ["-c", "sleep 3600 &"]))
        time.sleep(3)
        before = {n: switches(p.pid) for n, (p, _) in servers.items()}
        t0 = time.time()
        time.sleep(a.secs)
        dt = time.time() - t0
        out = {"bin": a.bin, "env": a.env, "secs": round(dt, 1), "threads":
               {n: threads(p.pid) for n, (p, _) in servers.items()}}
        out["switches_per_second"] = {n: round((switches(p.pid) - before[n]) / dt, 2)
                                      for n, (p, _) in servers.items()}
        print(json.dumps(out, indent=1))
        for p, _ in servers.values():
            kill_tree(p.pid)
            p.wait()


if __name__ == "__main__":
    main()
