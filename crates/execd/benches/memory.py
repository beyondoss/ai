#!/usr/bin/env python3
"""Memory of one exec endpoint binary: idle, 100 idle streams, 20 kept jobs, a burst of slow readers.

    benches/memory.py --label musl --bin /path/to/execd [--env K=V ...] [--cpus 0,1] [--n 3]
        [--scenarios idle,streams,jobs,burst] [--stub] [--style execd|addr|busybox]
(--stub: `beyond-ai-demo exec-stub`; --style addr: the Rust floor; --style busybox: busybox httpd)

Prints one JSON object: per scenario, the median of `--n` fresh servers of smaps_rollup fields in
KiB (Rss, Pss, Pss_Anon, Pss_File, Private_Dirty, Anonymous, AnonHugePages), VmRSS and the thread
count, and the THP setting. Standard library only. The server runs with its stdio discarded;
`--cpus` pins it with taskset(1). Measure the artifact that ships: a static musl release build.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROLLUP = ["Rss", "Pss", "Pss_Anon", "Pss_File", "Private_Dirty", "Anonymous", "AnonHugePages"]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, a, root):
        self.port = free_port()
        cmd = []
        if a.cpus:
            cmd += ["taskset", "-c", a.cpus]
        cmd += [a.bin]
        if a.style == "addr":  # the Rust floor: `floor ADDR`
            cmd += [f"127.0.0.1:{self.port}"]
        elif a.style == "busybox":  # the C reference: `busybox httpd -f -p PORT -h DIR`
            cmd += ["httpd", "-f", "-p", f"127.0.0.1:{self.port}", "-h", root]
        else:
            if a.stub:
                cmd += ["exec-stub"]
            cmd += ["--listen", f"127.0.0.1:{self.port}", "--root", root, "--home", root]
        env = dict(os.environ)
        for kv in a.env:
            k, v = kv.split("=", 1)
            env[k] = v
        def no_thp():
            # PR_SET_THP_DISABLE: no transparent huge pages for this process, so a result does not
            # depend on the host's THP setting.
            import ctypes
            ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)

        self.p = subprocess.Popen(cmd, env=env, preexec_fn=no_thp if a.no_thp else None,
                                  stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pid = self.p.pid
        t = time.time()
        while True:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                break
            except OSError:
                if time.time() - t > 20:
                    raise
                time.sleep(0.02)

    def measure(self):
        out = {}
        with open(f"/proc/{self.pid}/smaps_rollup") as f:
            for line in f:
                k = line.split(":")[0]
                if k in ROLLUP:
                    out[k] = int(line.split()[1])
        with open(f"/proc/{self.pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["VmRSS"] = int(line.split()[1])
                if line.startswith("Threads:"):
                    out["Threads"] = int(line.split()[1])
        return out

    def close(self):
        # Whatever it still runs dies with it.
        kids = []
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    st = open(f"/proc/{d}/stat").read()
                    if int(st.rsplit(")", 1)[1].split()[1]) == self.pid:
                        kids.append(int(d))
                except (OSError, ValueError, IndexError):
                    pass
        self.p.kill()
        self.p.wait()
        for k in kids:
            try:
                os.killpg(k, 9)
            except OSError:
                pass


def post(port, body, stream=False):
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps(body).encode()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: m\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(b) + b)
    return s


def read_head(s):
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(65536)
        if not d:
            break
        buf += d
    return buf


def request(cmd, args, root, **extra):
    r = {"command": cmd, "args": args, "cwd": root, "timeout_ms": 600000}
    r.update(extra)
    return r


def scen_idle(a, srv, root):
    time.sleep(1.0)
    return {"idle": srv.measure()}


def scen_streams(a, srv, root):
    time.sleep(1.0)
    base = srv.measure()
    socks = []
    for _ in range(100):
        s = post(srv.port, request("sleep", ["30"], root, stream=True))
        read_head(s)
        socks.append(s)
    time.sleep(1.0)
    m = srv.measure()
    for s in socks:
        s.close()
    return {"streams100": m,
            "per_stream_pss_kib": (m["Pss"] - base["Pss"]) / 100.0,
            "per_stream_rss_kib": (m["Rss"] - base["Rss"]) / 100.0}


def scen_jobs(a, srv, root):
    time.sleep(1.0)
    base = srv.measure()
    n = 20
    for i in range(n):
        body = request("sh", ["-c", "head -c 9437184 /dev/zero"], root)
        body["job"] = {"op": "start", "scope": "m", "id": f"{i:02x}", "hold": False, "label": ""}
        s = post(srv.port, body)
        read_head(s)
        s.close()
    # `head` writes 9 MiB in milliseconds; give every job time to end with its output held.
    time.sleep(3.0)
    time.sleep(1.0)
    m = srv.measure()
    return {"jobs20_9MiB": m, "per_job_pss_kib": (m["Pss"] - base["Pss"]) / n,
            "per_job_rss_kib": (m["Rss"] - base["Rss"]) / n}


def scen_burst(a, srv, root):
    time.sleep(1.0)
    base = srv.measure()
    stop = threading.Event()
    socks = []

    def slow(s):
        try:
            while not stop.is_set():
                if not s.recv(65536):
                    return
                time.sleep(0.05)
        except OSError:
            pass

    ts = []
    for _ in range(16):
        s = post(srv.port, request("head", ["-c", "268435456", "/dev/zero"], root, stream=True))
        socks.append(s)
        t = threading.Thread(target=slow, args=(s,), daemon=True)
        t.start()
        ts.append(t)
    peak = 0
    t0 = time.time()
    while time.time() - t0 < 8:
        peak = max(peak, srv.measure()["Rss"])
        time.sleep(0.1)
    hwm = 0
    with open(f"/proc/{srv.pid}/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                hwm = int(line.split()[1])
    stop.set()
    for s in socks:
        s.close()
    time.sleep(1.0)
    after_drop = srv.measure()
    time.sleep(float(os.environ.get("MEM_BURST_IDLE_SECS", "30")))
    after_idle = srv.measure()
    return {"burst_peak_rss_kib": peak, "burst_hwm_kib": hwm, "burst_after_drop": after_drop,
            "burst_after_idle": after_idle, "burst_baseline": base}


SCEN = {"idle": scen_idle, "streams": scen_streams, "jobs": scen_jobs, "burst": scen_burst}


def median_of(runs):
    out = {}
    for k in runs[0]:
        vals = [r[k] for r in runs]
        if isinstance(vals[0], dict):
            out[k] = median_of(vals)
        else:
            out[k] = statistics.median(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--cpus", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scenarios", default="idle,streams,jobs")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--no-thp", action="store_true", help="PR_SET_THP_DISABLE on the server")
    ap.add_argument("--style", default="execd", choices=["execd", "addr", "busybox"])
    a = ap.parse_args()
    res = {}
    for name in a.scenarios.split(","):
        runs = []
        for _ in range(a.n):
            with tempfile.TemporaryDirectory() as root:
                srv = Server(a, root)
                try:
                    runs.append(SCEN[name](a, srv, root))
                finally:
                    srv.close()
        res.update(median_of(runs))
    try:
        thp = open("/sys/kernel/mm/transparent_hugepage/enabled").read().strip()
    except OSError:
        thp = "unknown"
    json.dump({"label": a.label, "bin": a.bin, "env": a.env, "cpus": a.cpus, "n": a.n,
               "thp": thp, "thp_disabled_for_server": a.no_thp, "unit": "KiB (medians)", "results": res}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
