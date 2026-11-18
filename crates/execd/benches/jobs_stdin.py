#!/usr/bin/env python3
"""Job stdin and job-protocol additions: the P1-P7 bars of the plan.

    benches/jobs_stdin.py --bin /path/to/new/execd [--base /path/to/baseline/execd] [-n 2000]

Prints JSON. Keep-alive HTTP/1.1 over loopback, standard library only. One server per measurement
(fresh root), the baseline and the new binary alternating per round so host drift lands on both.
  B1  `list` op latency, keep-alive            P1  `write` 1 KiB to a `cat >/dev/null` job
  B2  one-shot `cat` with base64 stdin         P2  `write` to a `cat` job until the bytes arrive on an attach
  B3  one-shot 16 MiB stdin MiB/s              P3  1 GiB as 128 KiB writes, 8 connections, into `cat >/dev/null`
  P4  threads/fds after 10,000 writes          P5  RSS: 100 idle stdin-open jobs; stalled job with 16 queued writers
  P7  one-shot `true`, job start+read latency (new vs baseline)
"""
import argparse, base64, http.client, json, os, socket, statistics, subprocess, tempfile, threading, time


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Server:
    def __init__(self, binary):
        self.dir = tempfile.mkdtemp(prefix="jstdin-")
        self.port = free_port()
        env = dict(os.environ, EXECD_LISTEN=f"127.0.0.1:{self.port}", EXECD_ROOT=self.dir, EXECD_JOBS_DIR=self.dir)
        env.pop("EXECD_BEARER", None); env.pop("EXECD_BEARER_FILE", None)
        self.p = subprocess.Popen([binary], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", self.port)).close(); break
            except OSError:
                time.sleep(0.02)

    def conn(self):
        return http.client.HTTPConnection("127.0.0.1", self.port)

    def call(self, c, body):
        b = json.dumps(body)
        c.request("POST", "/exec", b, {"Content-Type": "application/json"})
        r = c.getresponse(); d = r.read()
        return r.status, (json.loads(d) if d[:1] == b"{" else d)

    def job(self, c, op, **kw):
        return self.call(c, {"command": "true", "job": dict(op=op, scope="b", **kw)})

    def start(self, c, jid, cmd, **kw):
        return self.call(c, {"command": "sh", "args": ["-c", cmd], "job": dict(op="start", scope="b", id=jid, hold=False, **kw)})

    def rss_kb(self):
        for l in open(f"/proc/{self.p.pid}/status"):
            if l.startswith("VmRSS"): return int(l.split()[1])

    def counts(self):
        return len(os.listdir(f"/proc/{self.p.pid}/task")), len(os.listdir(f"/proc/{self.p.pid}/fd"))

    def stop(self):
        self.p.kill(); self.p.wait()


def pct(xs):
    xs = sorted(xs); n = len(xs)
    return {"n": n, "p50_ms": round(xs[n // 2] * 1e3, 4), "p95_ms": round(xs[int(n * .95)] * 1e3, 4)}


def timed(n, f):
    out = []
    for _ in range(n):
        t = time.perf_counter(); f(); out.append(time.perf_counter() - t)
    return out


def latency(binary, n):
    s = Server(binary); c = s.conn(); r = {}
    try:
        for _ in range(50): s.job(c, "list")
        r["B1_list"] = pct(timed(n, lambda: s.job(c, "list")))
        r["P7_oneshot_true"] = pct(timed(n, lambda: s.call(c, {"command": "true", "timeout_ms": 5000})))
        k = [0]
        def sr():
            k[0] += 1; i = f"j{k[0]}"
            s.start(c, i, "printf x"); s.job(c, "read", id=i, offset=0, max=10); s.job(c, "remove", ids=[i])
        r["P7_job_start_read_remove"] = pct(timed(min(n, 500), sr))
        d = base64.b64encode(b"x" * 1024).decode()
        r["B2_oneshot_cat_stdin"] = pct(timed(n, lambda: s.call(c, {"command": "cat", "stdin": d, "timeout_ms": 5000})))
    finally:
        s.stop()
    return r


def new_only(binary, n):
    s = Server(binary); c = s.conn(); r = {}
    try:
        data = base64.b64encode(b"x" * 1024).decode()
        s.start(c, "w1", "cat >/dev/null", stdin_open=True)
        pos = [0]
        def w():
            st, _ = s.job(c, "write", id="w1", at=pos[0], data=data); assert st == 200; pos[0] += 1024
        for _ in range(50): w()
        r["P1_write_1KiB"] = pct(timed(n, w))
        # P2: echo round trip through an attach stream
        s.start(c, "e1", "cat", stdin_open=True)
        a = socket.create_connection(("127.0.0.1", s.port))
        b = json.dumps({"command": "true", "stream": True, "job": {"op": "attach", "scope": "b", "id": "e1", "offset": 0}}).encode()
        a.sendall(b"POST /exec HTTP/1.1\r\nHost: t\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
        buf = b""
        while b"\r\n\r\n" not in buf: buf += a.recv(65536)
        buf = buf.split(b"\r\n\r\n", 1)[1]
        epos = [0]; lat = []
        for _ in range(n):
            t = time.perf_counter()
            s.job(c, "write", id="e1", at=epos[0], data=data); epos[0] += 1024
            need = epos[0]; got = [0]
            # count decoded bytes arriving on the stream
            while True:
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if b'"stdout"' in line:
                        got[0] += len(base64.b64decode(json.loads(line)["stdout"]))
                if got[0] >= 1024: break
                buf += a.recv(65536)
            lat.append(time.perf_counter() - t)
        # got counted per round; frames may coalesce, fine for latency
        r["P2_echo_round_trip"] = pct(lat); a.close()
        # P4: threads/fds before/after 10,000 writes
        before = s.counts()
        for _ in range(10000):
            s.job(c, "write", id="w1", at=pos[0], data=data); pos[0] += 1024
        r["P4_threads_fds_before_after_10k_writes"] = {"before": before, "after": s.counts()}
        # P3: 1 GiB as 128 KiB writes, 8 connections
        chunk = base64.b64encode(b"x" * (128 * 1024)).decode()
        s.start(c, "t1", "cat >/dev/null", stdin_open=True)
        total = 1 << 30; nchunks = total // (128 * 1024); lock = threading.Lock(); nxt = [0]
        def worker():
            cc = s.conn()
            while True:
                with lock:
                    i = nxt[0]; nxt[0] += 1
                if i >= nchunks: return
                st, _ = s.job(cc, "write", id="t1", at=i * 128 * 1024, data=chunk); assert st == 200, st
        t = time.perf_counter(); th = [threading.Thread(target=worker) for _ in range(8)]
        [x.start() for x in th]; [x.join() for x in th]
        r["P3_throughput_MiBps_8conn"] = round(1024 / (time.perf_counter() - t), 1)
    finally:
        s.stop()
    # P5 memory
    s = Server(binary); c = s.conn()
    try:
        base = s.rss_kb()
        for i in range(100): s.start(c, f"i{i}", "sleep 600", stdin_open=True)
        idle = s.rss_kb()
        r["P5_idle_100_stdin_jobs"] = {"rss_kb_before": base, "rss_kb_after": idle, "per_job_kb": round((idle - base) / 100, 2)}
        s.start(c, "z", "sleep 600", stdin_open=True)
        big = base64.b64encode(b"x" * (128 * 1024)).decode(); ths = []
        def w(i):
            cc = s.conn(); s.call(cc, {"command": "true", "timeout_ms": 60000, "job": dict(op="write", scope="b", id="z", at=i * 131072, data=big)})
        r0 = s.rss_kb()
        for i in range(16): t = threading.Thread(target=w, args=(i,), daemon=True); t.start(); ths.append(t)
        time.sleep(3); r1 = s.rss_kb(); time.sleep(20); r2 = s.rss_kb()
        r["P5_stalled_16_writers_rss_kb"] = {"before": r0, "after_3s": r1, "after_23s": r2, "bound_kb": 16 * 512}
    finally:
        s.stop()
    # B3 one-shot stdin throughput (16 MiB)
    s = Server(binary); c = s.conn()
    try:
        d = base64.b64encode(b"x" * (16 << 20)).decode()
        t = time.perf_counter(); st, _ = s.call(c, {"command": "sh", "args": ["-c", "cat >/dev/null"], "stdin": d, "timeout_ms": 60000})
        r["B3_oneshot_stdin_MiBps"] = round(16 / (time.perf_counter() - t), 1) if st == 200 else f"status {st}"
    finally:
        s.stop()
    return r


def stalled_sweep(binary):
    """P5 stalled: RSS growth against the number of queued 128 KiB writers on a job that never reads.
    A per-writer cost shows as a slope, a fixed cost as the intercept."""
    out = {}
    for w in (0, 1, 4, 8, 16):
        s = Server(binary); c = s.conn()
        try:
            s.start(c, "z", "sleep 600", stdin_open=True)
            big = base64.b64encode(b"x" * (128 * 1024)).decode()
            r0 = s.rss_kb()
            def wr(i):
                cc = s.conn(); s.call(cc, {"command": "true", "timeout_ms": 30000, "job": dict(op="write", scope="b", id="z", at=i * 131072, data=big)})
            ths = [threading.Thread(target=wr, args=(i,), daemon=True) for i in range(w)]
            [t.start() for t in ths]
            time.sleep(3); r1 = s.rss_kb(); time.sleep(10); r2 = s.rss_kb()
            out[w] = {"rss_kb_before": r0, "delta_3s_kb": r1 - r0, "delta_13s_kb": r2 - r0, "kb_per_writer": round((r1 - r0) / w, 1) if w else None}
        finally:
            s.stop()
    return out


if __name__ == "__main__":
    if "--p5" in os.sys.argv:
        print(json.dumps(stalled_sweep(os.sys.argv[os.sys.argv.index("--bin") + 1]), indent=1)); raise SystemExit
    ap = argparse.ArgumentParser(); ap.add_argument("--bin", required=True); ap.add_argument("--base"); ap.add_argument("-n", type=int, default=2000); ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(); out = {"load": os.getloadavg(), "rounds": []}
    for rd in range(a.rounds):
        row = {}
        order = [("new", a.bin)] + ([("base", a.base)] if a.base else [])
        if rd % 2: order.reverse()
        for name, b in order: row[name] = latency(b, a.n)
        out["rounds"].append(row)
    out["new_only"] = new_only(a.bin, a.n)
    out["load_end"] = os.getloadavg()
    print(json.dumps(out, indent=1))
