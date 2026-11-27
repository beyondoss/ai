#!/usr/bin/env python3
"""A job's end: what one job-life cycle costs, against the sandbox's process table.

    benches/proc_scan.py --bin target/release/execd [--prof-bin DIR/execd] \\
        [--procs 330,730] [--cycles 15] [--busy] [--assert-opens 0] [--assert-allocs 64]

One cycle is what a client does per command: `start` a job, wait for it to end, `remove` it. On the
shipped code every cycle pays one full `/proc` scan (`proc::members`), which opens
`/proc/<pid>/stat` and, for every process the job's group does not reach, `/proc/<pid>/environ`:
the cycle's cost tracks how many processes the sandbox holds, not what the job did.

`--procs N[,M]` spawns that many `sleep`s (children of this script, so only the box's process table
grows) and measures at each. Reported per process count:

- `cycle_p50_ms`, `cycles_per_s`: fresh server, `--cycles` cycles, no tracing, table empty;
- `opens_per_cycle`: the same cycles with the server under `strace -f` (counts only: strace slows
  the process down, so its timings are not reported): `/proc` readdir, `<pid>/stat`,
  `<pid>/environ`, and every file open;
- `allocs_per_cycle`: the same cycles with the server under `valgrind --tool=dhat`, summing the
  blocks that valgrind attributes to frames inside `proc.rs` (needs `--prof-bin`, a build with
  `CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=1`).

`--busy` starts one long-lived job on the same server first, so execd has a child of its own while
the cycles run: that is the case a check on execd's own child list cannot answer, and it must fall
back to the scan. Compare `--busy` with the default to see which path a cycle took.

`--assert-proc-opens N` makes this a gate: exit 1 when a cycle opens more than N files under
`/proc` (the scan is what a cycle must not pay). `--assert-flat F` is the other half: exit 1 when any
per-cycle number grows by more than the factor F between the smallest and the largest process count
measured — a cost that tracks the box's process table rather than the job. Standard library only.
"""
import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time

BIG = 1 << 40
# Cycles before the counted ones: the first pays for the spool directory and cold page cache.
WARMUP = 3


# --------------------------------------------------------------------------------------------
# The client: keep-alive HTTP/1.1 POST /exec, as the agent speaks it.

class Client:
    def __init__(self, port, timeout=600):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = b""

    def close(self):
        self.s.close()

    def _fill(self):
        d = self.s.recv(1 << 20)
        if not d:
            raise EOFError("closed")
        self.buf += d

    def req(self, body):
        b = json.dumps(body).encode()
        self.s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
        while b"\r\n\r\n" not in self.buf:
            self._fill()
        head, _, self.buf = self.buf.partition(b"\r\n\r\n")
        n = 0
        for line in head.split(b"\r\n")[1:]:
            k, _, v = line.partition(b":")
            if k.lower() == b"content-length":
                n = int(v)
        while len(self.buf) < n:
            self._fill()
        body, self.buf = self.buf[:n], self.buf[n:]
        return json.loads(body)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    """execd on a free port, over a fresh root, optionally under strace or valgrind."""

    def __init__(self, binary, root, wrap=(), out=None):
        self.port = free_port()
        self.root = root
        self.out = out
        env = dict(os.environ)
        env.update(
            EXECD_LISTEN="127.0.0.1:%d" % self.port,
            EXECD_ROOT=root,
            EXECD_STATE_DIR=os.path.join(root, "state"),
            EXECD_JOBS_DIR=os.path.join(root, "jobs"),
            EXECD_LOG="warn",
        )
        os.makedirs(env["EXECD_STATE_DIR"], exist_ok=True)
        os.makedirs(env["EXECD_JOBS_DIR"], exist_ok=True)
        cmd = list(wrap) + [binary]
        self.proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def client(self, timeout=600):
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                return Client(self.port, timeout)
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError("execd exited before it listened") from None
                time.sleep(0.02)
        raise RuntimeError("execd never listened")

    def stop(self):
        # A wrapper (strace, valgrind) need not pass the signal on, and the measurement is done:
        # ask, wait a moment, then take the whole group down.
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


def jobop(c, root, jid, op, **kw):
    return c.req({"command": "true", "cwd": root,
                  "job": dict({"op": op, "scope": "p", "id": jid}, **kw)})


def run_cycles(c, root, n, prefix="cy"):
    """n cycles of start -> end -> remove; the wall time of each, in ms."""
    lat = []
    for i in range(n):
        jid = "%s%d" % (prefix, i)
        t = time.time()
        c.req({"command": "sh", "args": ["-c", "echo hi"], "cwd": root,
               "job": {"op": "start", "scope": "p", "id": jid}})
        while True:
            v = jobop(c, root, jid, "read", offset=BIG, max=1)
            if v["state"] in ("ended", "gone"):
                break
            time.sleep(0.002)
        jobop(c, root, jid, "remove", ids=[jid])
        lat.append((time.time() - t) * 1e3)
    return lat


def run_busy(c, root):
    """One long-lived job, so execd keeps a child of its own while the cycles run."""
    c.req({"command": "sleep", "args": ["600"], "cwd": root,
           "job": {"op": "start", "scope": "p", "id": "busy"}})


# --------------------------------------------------------------------------------------------
# The three measurements.

def measure_time(binary, procs, cycles, busy):
    with tempfile.TemporaryDirectory() as root:
        s = Server(binary, root)
        try:
            c = s.client()
            if busy:
                run_busy(c, root)
                time.sleep(0.1)
            run_cycles(c, root, WARMUP, "warm")
            lat = run_cycles(c, root, cycles)
            if busy:
                jobop(c, root, "busy", "wipe", grace_ms=200)
            c.close()
        finally:
            s.stop()
    p50 = statistics.median(lat)
    return {"cycle_p50_ms": round(p50, 3), "cycles_per_s": round(1000.0 / p50, 1) if p50 else None,
            "procs": procs}


STAT_RX = re.compile(r'openat?\(.*?"/proc/\d+/stat"')
ENV_RX = re.compile(r'openat?\(.*?"/proc/\d+/environ"')
DIR_RX = re.compile(r'openat?\(.*?"/proc"')
OPEN_RX = re.compile(r"\bopen(at)?\(")


def measure_opens(binary, procs, cycles, busy):
    trace = tempfile.mktemp(suffix=".strace")
    with tempfile.TemporaryDirectory() as root:
        s = Server(binary, root, wrap=["strace", "-f", "-qq",
                                      "-e", "trace=openat,open,getdents64", "-o", trace])
        try:
            c = s.client()
            if busy:
                run_busy(c, root)
                time.sleep(0.2)
            run_cycles(c, root, WARMUP + cycles)
            if busy:
                jobop(c, root, "busy", "wipe", grace_ms=200)
            c.close()
        finally:
            s.stop()
    stat = env = directory = opens = 0
    with open(trace, errors="replace") as f:
        for line in f:
            if STAT_RX.search(line):
                stat += 1
            if ENV_RX.search(line):
                env += 1
            if DIR_RX.search(line):
                directory += 1
            if OPEN_RX.search(line):
                opens += 1
    os.unlink(trace)
    return {
        "procs": procs,
        "opens": {
            "per_cycle": {
                "proc_readdir": round(directory / (WARMUP + cycles), 1),
                "proc_stat": round(stat / (WARMUP + cycles), 1),
                "proc_environ": round(env / (WARMUP + cycles), 1),
                "all_files": round(opens / (WARMUP + cycles), 1),
            }
        },
    }


def measure_allocs(prof_bin, procs, cycles, busy):
    out = tempfile.mktemp(suffix=".dhat.json")  # valgrind needs to create it: not from mkstemp
    vg = tempfile.mktemp(suffix=".vg.log")
    with tempfile.TemporaryDirectory() as root:
        s = Server(prof_bin, root, wrap=["valgrind", "--tool=dhat", "--quiet",
                                        "--dhat-out-file=" + out], out=vg)
        try:
            c = s.client()
            if busy:
                run_busy(c, root)
                time.sleep(0.5)
            run_cycles(c, root, WARMUP + cycles)
            if busy:
                jobop(c, root, "busy", "wipe", grace_ms=200)
            c.close()
        finally:
            s.stop()
    blocks = bytes_ = 0
    try:
        a = json.load(open(out))
    except (OSError, ValueError):
        a = None
    finally:
        for p in (out, vg):
            try:
                os.unlink(p)
            except OSError:
                pass
    if a is None:
        return {"procs": procs, "allocs": {"blocks_per_cycle": None,
                                           "note": "valgrind wrote no DHAT JSON"}}
    ftbl = a.get("ftbl") or []
    for p in a.get("pps") or []:
        frames = []
        for i in p.get("fs") or []:
            frames.append(ftbl[i - 1] if 0 < i <= len(ftbl) else "")
        site = " ".join(frames)
        if "proc.rs" not in site and "proc::" not in site:
            continue
        blocks += p.get("tbk", 0)
        bytes_ += p.get("tb", 0)
    return {"procs": procs,
            "allocs": {"blocks_per_cycle": round(blocks / (WARMUP + cycles), 1),
                       "bytes_per_cycle": round(bytes_ / (WARMUP + cycles), 1)}}


def spawn_sleeps(n):
    pids = []
    for _ in range(n):
        pids.append(subprocess.Popen(["sleep", "900"]).pid)
    time.sleep(0.3)
    return pids


def kill(pids):
    for p in pids:
        try:
            os.kill(p, 9)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/execd")
    ap.add_argument("--prof-bin", default="", help="a build with symbols, for the allocation count")
    ap.add_argument("--procs", default="0", help="comma separated process counts to add to the box")
    ap.add_argument("--cycles", type=int, default=15)
    ap.add_argument("--busy", action="store_true",
                    help="keep one long-lived job running, so a child of execd's own exists")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-opens", action="store_true")
    ap.add_argument("--skip-allocs", action="store_true")
    ap.add_argument("--assert-proc-opens", type=float, default=None)
    ap.add_argument("--assert-flat", type=float, default=None,
                    help="fail when a per-cycle number grows by more than this factor with the box")
    a = ap.parse_args()

    rows = []
    for n in [int(x) for x in a.procs.split(",") if x.strip()]:
        pids = spawn_sleeps(n)
        procs = sum(1 for p in os.listdir("/proc") if p.isdigit())
        row = {"procs": procs, "added": n, "busy": a.busy,
               "cycles": a.cycles}
        try:
            if not a.skip_time:
                row.update(measure_time(a.bin, procs, a.cycles, a.busy))
            if not a.skip_opens:
                row.update(measure_opens(a.bin, procs, a.cycles, a.busy))
            if a.prof_bin and not a.skip_allocs:
                row.update(measure_allocs(a.prof_bin, procs, a.cycles, a.busy))
        finally:
            kill(pids)
        rows.append(row)
        print(json.dumps(row), flush=True)

    bad = []
    for r in rows:
        per_cycle = (r.get("opens") or {}).get("per_cycle") or {}
        proc_opens = sum(per_cycle.get(k, 0) for k in ("proc_readdir", "proc_stat", "proc_environ"))
        if a.assert_proc_opens is not None and per_cycle and proc_opens > a.assert_proc_opens:
            bad.append("procs=%s: %s opens under /proc per cycle > %s"
                       % (r["procs"], proc_opens, a.assert_proc_opens))
    if a.assert_flat is not None and len(rows) > 1:
        metrics = [
            ("opens per cycle", [sum(((r.get("opens") or {}).get("per_cycle") or {})
                                     .get(k, 0) for k in ("all_files",)) for r in rows]),
            ("allocations per cycle in proc.rs",
             [((r.get("allocs") or {}).get("blocks_per_cycle")) for r in rows]),
            ("cycle p50 ms", [r.get("cycle_p50_ms") for r in rows]),
        ]
        for name, values in metrics:
            if any(v is None or v == 0 for v in values):
                continue
            ratio = max(values) / min(values)
            if ratio > a.assert_flat:
                bad.append("%s grew %.2fx across procs %s (limit %.2fx): %s"
                           % (name, ratio, [r["procs"] for r in rows], a.assert_flat, values))
    for b in bad:
        print("FAIL: " + b, file=sys.stderr)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
