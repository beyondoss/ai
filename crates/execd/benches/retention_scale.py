#!/usr/bin/env python3
"""Ended-job retention at scale: what `list` and the ledger cost at 256, 1024 and 4096 entries.

    benches/retention_scale.py --bin /path/to/execd [--sizes 256,1024,4096] [--cpus 0,1]

The only way to reach thousands of ended entries in a second is a ledger a restart adopts, which is
also the state a 24 h clock reaps: each row here is what execd itself writes (`ledger.rs`), with a
realistic id and argv and no spool (the tombstone a 15-minute output clock leaves). Per size:

- `start_ms`: process start to the first `list` answer, with that many rows to adopt;
- `list_bytes`, `list_ms`: the answer and the handler's time, p50/p95 over 30 requests;
- `livez_ms`: `GET /livez` from another connection while a `list` runs, on the default one-thread
  runtime: what a big answer costs everyone else;
- `ledger_bytes`: `execd-jobs.json`;
- `keep_ms` (a job-life change, which rewrites the ledger whole) against `seen_ms` (no ledger
  write), from the client's side: the per-event write;
- `reap_ms`, `reap_passes`: with `EXECD_JOB_ENTRY_MS=24h` and every row older than that, how long the
  first sweeps take to reap them all, and how many ledger writes (`wchar`, from /proc) that took.

Standard library only.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import tempfile
import threading
import time

ROW = {
    "scope": "s",
    "label": "migration",
    "command": "bash",
    "args": ["-lc", "cargo test --workspace --release -- --test-threads=4"],
    "phase": "ended",
    "code": 0,
    "signal": None,
    "told": True,
    "kept": {"num": None, "kind": "background"},
    "seen": 1,
    "term": None,
    "idem": None,
    "spool": None,
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(port, body, timeout=600):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    b = json.dumps(body).encode()
    s.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
    data = b""
    while b"\r\n\r\n" not in data:
        data += s.recv(1 << 20)
    head, _, rest = data.partition(b"\r\n\r\n")
    n = int(
        [l for l in head.lower().split(b"\r\n") if l.startswith(b"content-length")][0]
        .split(b":")[1]
    )
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


def livez(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    s.sendall(b"GET /livez HTTP/1.1\r\nHost: t\r\n\r\n")
    data = b""
    while True:
        d = s.recv(4096)
        if not d:
            break
        data += d
        if b"\r\n\r\n" in data:
            break
    s.close()
    return data


def ledger(size):
    """`size` ended rows, `at` a day and a bit old (the last `size // 2` younger than that)."""
    now = int(time.time())
    out = []
    for i in range(size):
        r = dict(ROW)
        r["id"] = "job_01J8Z9K3M4N5P6Q7R8S9T0V1W%05d" % i
        r["mark"] = "%016x" % (0x9F3A0000 + i)
        r["pgid"] = 400000 + i
        # Half the rows are past the 24 h clock (the ones the first sweeps reap); half are not.
        r["at"] = now - 2 * 86400 if i < size // 2 else now - 60
        r["started"] = r["at"] - 1
        out.append(r)
    return {"version": 1, "jobs": out}


def start(root, size, extra):
    state = os.path.join(root, "state")
    jobs = os.path.join(root, "jobs")
    for d in (state, jobs):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(state, "execd-jobs.json"), "w") as f:
        json.dump(ledger(size), f)
    port = free_port()
    env = dict(os.environ)
    env.update(
        {
            "EXECD_LISTEN": "127.0.0.1:%d" % port,
            "EXECD_ROOT": root,
            "EXECD_HOME": root,
            "EXECD_STATE_DIR": state,
            "EXECD_JOBS_DIR": jobs,
            "EXECD_LOG": "info",
            "EXECD_JOB_OUTPUT_MS": "0",
            "EXECD_JOBS_ENDED_BYTES": "0",
            "EXECD_JOBS_ENDED_MAX": "0",
            "EXECD_JOB_ENTRY_MS": "0",
        }
    )
    env.update(extra)
    log = open(os.path.join(root, "execd.log"), "wb")
    t = time.time()
    p = subprocess.Popen(
        BIN,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=log,
    )
    list_body = {"command": "true", "job": {"op": "list", "scope": "s"}}
    for _ in range(3000):
        try:
            call(port, list_body, timeout=120)
            break
        except (OSError, ValueError, KeyError):
            time.sleep(0.005)
    return p, port, state, (time.time() - t) * 1000


def wchar(pid):
    with open("/proc/%d/io" % pid) as f:
        for line in f:
            if line.startswith("wchar:"):
                return int(line.split()[1])
    return 0


def measure(root, size):
    p, port, state, start_ms = start(root, size, {})
    out = {"entries": size, "start_ms": round(start_ms, 1)}
    try:
        out["ledger_bytes"] = os.path.getsize(os.path.join(state, "execd-jobs.json"))
        listing = {"command": "true", "job": {"op": "list", "scope": "s"}}
        sizes, times = [], []
        for _ in range(30):
            t = time.time()
            v = call(port, listing)
            times.append((time.time() - t) * 1000)
            sizes.append(len(json.dumps({"jobs": v["jobs"]})))
        out["rows"] = len(v["jobs"])
        out["list_bytes"] = sizes[-1]
        out["list_ms_p50"] = round(statistics.median(times), 2)
        out["list_ms_p95"] = round(sorted(times)[int(len(times) * 0.95) - 1], 2)
        out["list_ms_max"] = round(max(times), 2)

        # Everyone else while a list runs: the handler builds the answer on the runtime thread.
        stalls = []
        stop = threading.Event()

        def probe():
            while not stop.is_set():
                t = time.time()
                try:
                    livez(port)
                except OSError:
                    pass
                stalls.append((time.time() - t) * 1000)
                time.sleep(0.002)

        th = threading.Thread(target=probe, daemon=True)
        th.start()
        time.sleep(0.2)
        base = sorted(stalls)
        out["livez_idle_ms_p50"] = round(statistics.median(base), 2)
        stalls.clear()
        for _ in range(10):
            call(port, listing)
        time.sleep(0.05)
        stop.set()
        th.join()
        out["livez_during_list_ms_max"] = round(max(stalls), 2)
        out["livez_during_list_ms_p95"] = round(sorted(stalls)[int(len(stalls) * 0.95) - 1], 2)

        # The per-event ledger write: `keep` rewrites the file whole, `seen` writes nothing.
        def op(i, name, **kw):
            body = {"command": "true", "job": dict({"op": name, "scope": "s"}, **kw)}
            body["job"]["id"] = "job_01J8Z9K3M4N5P6Q7R8S9T0V1W%05d" % i
            return body

        def timed(name, **kw):
            ts = []
            for i in range(20):
                t = time.time()
                r = call(port, op(i, name, **kw))
                assert "error" not in r, (name, r)
                ts.append((time.time() - t) * 1000)
            return round(statistics.median(ts), 2)

        out["seen_ms_p50"] = timed("seen", offset=0)
        before = wchar(p.pid)
        out["keep_ms_p50"] = timed("keep", num=1, kind="background", seen=1)
        out["keep_wchar_bytes"] = round((wchar(p.pid) - before) / 20)
        out["ledger_bytes_after_keeps"] = os.path.getsize(
            os.path.join(state, "execd-jobs.json")
        )
    finally:
        p.kill()
        p.wait()
    return out


def measure_reap(root, size):
    """A restart where every entry is past a 24 h clock: the first sweeps reap them all."""
    state = os.path.join(root, "state")
    jobs = os.path.join(root, "jobs")
    for d in (state, jobs):
        os.makedirs(d, exist_ok=True)
    rows = ledger(size)
    for r in rows["jobs"]:
        r["at"] = int(time.time()) - 3 * 86400
    with open(os.path.join(state, "execd-jobs.json"), "w") as f:
        json.dump(rows, f)
    port = free_port()
    env = dict(os.environ)
    env.update(
        {
            "EXECD_LISTEN": "127.0.0.1:%d" % port,
            "EXECD_ROOT": root,
            "EXECD_HOME": root,
            "EXECD_STATE_DIR": state,
            "EXECD_JOBS_DIR": jobs,
            "EXECD_LOG": "debug",
            "EXECD_JOB_OUTPUT_MS": "0",
            "EXECD_JOBS_ENDED_BYTES": "0",
            "EXECD_JOBS_ENDED_MAX": "0",
            "EXECD_JOB_ENTRY_MS": str(86400 * 1000),
        }
    )
    log = open(os.path.join(root, "execd.log"), "wb")
    t = time.time()
    p = subprocess.Popen(
        BIN, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log
    )
    listing = {"command": "true", "job": {"op": "list", "scope": "s"}}
    first = None
    left = None
    before = 0
    try:
        for _ in range(6000):
            try:
                v = call(port, listing, timeout=120)
            except (OSError, ValueError, KeyError):
                time.sleep(0.005)
                continue
            if first is None:
                first = (time.time() - t) * 1000
                before = wchar(p.pid)
            left = len(v["jobs"])
            if left == 0:
                break
            time.sleep(0.01)
        done = (time.time() - t) * 1000
        out = {
            "entries": size,
            "start_ms": round(first or -1, 1),
            "reap_ms": round(done, 1),
            "left": left,
            "wchar_bytes": wchar(p.pid) - before,
            "ledger_bytes": os.path.getsize(os.path.join(state, "execd-jobs.json")),
        }
        spools = sum(len(d) for _, d, _ in os.walk(jobs) if d)
        out["spool_dirs_left"] = spools
        # A reaped entry answers 404 on attach and `gone` on read, never 200 with nothing.
        out["read"] = call(
            port,
            {
                "command": "true",
                "job": {
                    "op": "read",
                    "scope": "s",
                    "id": "job_01J8Z9K3M4N5P6Q7R8S9T0V1W00000",
                    "offset": 0,
                    "max": 16,
                },
            },
        )["state"]
    finally:
        p.kill()
        p.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--sizes", default="256,1024,4096")
    ap.add_argument("--cpus", default="")
    ap.add_argument("--only", default="list,reap")
    a = ap.parse_args()
    global BIN
    BIN = (
        ["taskset", "-c", a.cpus] + [a.bin] if a.cpus else [a.bin]
    )
    out = {"bin": a.bin, "cpus": a.cpus}
    want = set(a.only.split(","))
    for size in [int(s) for s in a.sizes.split(",")]:
        root = tempfile.mkdtemp(dir="/tmp")
        if "list" in want:
            out["list_%d" % size] = measure(root, size)
        if "reap" in want:
            root = tempfile.mkdtemp(dir="/tmp")
            out["reap_%d" % size] = measure_reap(root, size)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
