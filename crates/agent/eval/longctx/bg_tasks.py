"""Background-process tasks: small generated repos whose work needs a process kept running.

Each task is a few files of stdlib Python (an HTTP service, a worker, a test watcher, a slow build)
and a prompt asking for what a developer would do with them: start the server and reproduce with
curl, run an API and its worker together, keep a watcher going while fixing tests, run a two-minute
build while doing other edits, find an error that only shows in a server log under load, work
around a port someone else holds, and leave nothing running. The grade never reads the transcript:
`check` starts the agent's code afresh, on ports of its own, and asks it, or reads the files the
task asked for; `bg-cleanup` also requires that nothing the agent started was still running when it
exited (the sandbox's leftover report, `sandbox.survivors`).

Determinism: no clock, randomness or network in any answer (generated data is seeded), every wait
is bounded, and every port comes from a `Lease`: an flock-held slot in a host-wide directory, below
the kernel's ephemeral range, so parallel runs (and both arms of an A/B, in separate processes)
never share a port and no outgoing connection is handed one. A task's ports are written to its
`config.json`, which its services read; each also takes `--port`, which the grader uses.

`fix` and `answers` are the reference solution: exact edits, and the files the task asks the agent
to write from what a process printed. `validate.py --bg-only` applies them by hand; `mock_model.py`
drives the real agent through them with a scripted model.
"""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import tasks

# ---------------------------------------------------------------- ports


def _port_range() -> tuple[int, int]:
    """Ports to lease: under the kernel's ephemeral range, so no client socket is ever bound to one."""
    try:
        lo = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        lo = 32768
    return 20000, min(lo, 32768)


def _bindable(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class Lease:
    """`n` ports held for a run's lifetime: each an exclusive flock on `$LONGCTX_PORTS/<port>`
    (default `/tmp/longctx-ports-<uid>`, host-wide: ports are), released by `release` or when the
    process dies. A port is only taken if it is also bindable right now."""

    def __init__(self, n: int):
        d = Path(os.environ.get("LONGCTX_PORTS", f"/tmp/longctx-ports-{os.getuid()}"))
        d.mkdir(parents=True, exist_ok=True)
        lo, hi = _port_range()
        rng = random.SystemRandom()
        self.ports: list[int] = []
        self._fds: list[int] = []
        while len(self.ports) < n:
            port = rng.randrange(lo, hi)
            fd = os.open(d / str(port), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            if not _bindable(port):
                os.close(fd)
                continue
            self.ports.append(port)
            self._fds.append(fd)

    def release(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds = []


# ---------------------------------------------------------------- the grader's process handling


def python() -> str:
    """The interpreter the agent's `python` is (the shared venv's), for the grader too."""
    return str(tasks.venv() / "bin" / "python")


def _env(home: Path) -> dict[str, str]:
    """A grader process's environment: the agent's interpreter first on PATH, nothing inherited."""
    v = tasks.venv()
    return {"PATH": f"{v / 'bin'}:/usr/local/bin:/usr/bin:/bin", "VIRTUAL_ENV": str(v), "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(home), "LANG": "C.UTF-8", "TMPDIR": str(home)}


class Proc:
    """A process the grader starts in its own session, output to a log file; `stop` kills the group."""

    def __init__(self, argv: list[str], cwd: Path, log: Path):
        self.log = log
        with open(log, "wb") as out:
            self.p = subprocess.Popen(argv, cwd=cwd, env=_env(cwd.parent), stdin=subprocess.DEVNULL, stdout=out,
                                      stderr=subprocess.STDOUT, start_new_session=True)

    def output(self) -> str:
        return self.log.read_text(errors="replace")[-2000:]

    def stop(self) -> None:
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.p.wait()


def wait_port(port: int, proc: Proc, timeout: float = 20.0) -> bool:
    """True once something accepts on `port`; False if `proc` exits or `timeout` passes first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.p.poll() is not None:
            return False
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.05)
    return False


def request(port: int, method: str, path: str, body: object = None, timeout: float = 10.0) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def raw_get(port: int, path: str, timeout: float = 10.0) -> tuple[int, dict[str, str], bytes]:
    """A GET read to EOF over a bare socket: the whole body the server sent, whatever its
    Content-Length said (http.client would stop at Content-Length)."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".encode())
        chunks = []
        while chunk := s.recv(65536):
            chunks.append(chunk)
    raw = b"".join(chunks)
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:])}
    return int(lines[0].split()[1]), headers, body


class Scratch:
    """A grader's scratch copy of the agent's tree (so grading leaves the tree alone), its started
    processes, and their logs; `close` stops them all and removes the copy."""

    def __init__(self, tree: Path, skip: tuple[str, ...] = ()):
        base = tasks.work_dir() / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix="bgcheck-", dir=base))
        self.root = self.dir / "repo"
        shutil.copytree(tree, self.root, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.db", "*.db-*", *skip))
        self.procs: list[Proc] = []
        self.lease: Lease | None = None

    def ports(self, n: int) -> list[int]:
        self.lease = Lease(n)
        return self.lease.ports

    def start(self, name: str, *args: str) -> Proc:
        p = Proc([python(), *args], self.root, self.dir / f"{name}.log")
        self.procs.append(p)
        return p

    def run(self, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run([python(), *args], cwd=self.root, env=_env(self.dir), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)

    def close(self) -> None:
        for p in self.procs:
            p.stop()
        if self.lease:
            self.lease.release()
        shutil.rmtree(self.dir, ignore_errors=True)


def checked(fn: Callable[[Scratch], tuple[bool, str]], tree: Path, skip: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Run a grader body on a scratch copy; any exception (a hung request, bad JSON) is a fail."""
    s = Scratch(tree, skip)
    try:
        return fn(s)
    except Exception as e:  # noqa: BLE001 — the agent's code can fail any way at all
        return False, f"{type(e).__name__}: {e}"[:300]
    finally:
        s.close()


# ---------------------------------------------------------------- task plumbing


@dataclass
class Task:
    prompt: str
    check: Callable[[list[dict] | None], tuple[bool, str]]  # the run's leftovers → (passed, why)
    fix: list[tuple[str, str, str]]  # the reference edits: (file, old, new), each `old` exactly once
    answers: dict[str, str] = field(default_factory=dict)  # files the reference solution writes
    lease: Lease | None = None
    helpers: list[Callable[[], None]] = field(default_factory=list)  # harness-side teardown (a squatter)
    launches: tuple[str, ...] = ()  # the scripts a run starts as long-lived processes

    def close(self) -> None:
        for stop in self.helpers:
            stop()
        self.helpers = []
        if self.lease:
            self.lease.release()
            self.lease = None


def write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip("\n") if text.startswith("\n") else text)


def apply_fix(root: Path, task: Task) -> None:
    """The reference solution, by hand: every edit (each must match exactly once) and every answer."""
    for rel, old, new in task.fix:
        p = root / rel
        text = p.read_text()
        if text.count(old) != 1:
            raise AssertionError(f"{rel}: reference edit matches {text.count(old)} times: {old[:60]!r}")
        p.write_text(text.replace(old, new))
    for rel, text in task.answers.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def config(root: Path, **ports: int) -> None:
    (root / "config.json").write_text(json.dumps({"host": "127.0.0.1", **ports}, indent=2) + "\n")


def _squatter(port: int) -> Callable[[], None]:
    """Another program holding `port` (an unrelated dev server answering 404 to everything), run in
    the harness: outside the sandbox's PID namespace, so the agent can neither see nor kill it."""

    class NotOurs(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"404: no such app on this port\n"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), NotOurs)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def stop() -> None:
        srv.shutdown()
        srv.server_close()
    return stop


# ---------------------------------------------------------------- bg-api-bugs: server + curl

_ITEMS = [
    (1, "Apple", ["fruit"], 0.5), (2, "Banana", ["fruit"], 0.25), (3, "Carrot", ["vegetable"], 0.3),
    (4, "Café crème", ["drink", "hot"], 3.2), (5, "Jalapeño", ["vegetable", "spicy"], 0.4),
    (6, "Crème brûlée", ["dessert"], 5.5), (7, "Mango", ["fruit"], 1.1), (8, "Smörgåstårta", ["dessert", "savory"], 9.0),
    (9, "Lemonade", ["drink", "cold"], 2.0), (10, "Kiwi", ["fruit"], 0.6), (11, "Piña colada", ["drink", "cold"], 6.5),
    (12, "Leek", ["vegetable"], 0.9),
]

_INVENTORY = {
    "app/server.py": '''
        """Inventory API over the standard library's HTTP server.

            python app/server.py [--port N]

        Reads host and port from config.json at the repository root (--port overrides the port).

        Routes (all JSON):
            GET /health        {"ok": true}
            GET /items         {"items": [...]}: every item; ?tag=<tag> keeps the items with that tag,
                               ?limit=<n> the first n
            GET /items/<id>    one item, or 404
        """

        from __future__ import annotations

        import argparse
        import json
        import sys
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path
        from urllib.parse import parse_qs, urlsplit

        sys.path.insert(0, str(Path(__file__).resolve().parent))

        from query import apply_filters, parse_filters  # noqa: E402
        from store import Store  # noqa: E402

        ROOT = Path(__file__).resolve().parent.parent
        STORE = Store(ROOT / "data" / "items.json")


        class Handler(BaseHTTPRequestHandler):
            server_version = "inventory/1.4"

            def do_GET(self) -> None:
                url = urlsplit(self.path)
                parts = [p for p in url.path.split("/") if p]
                if parts == ["health"]:
                    return self.send_json(200, {"ok": True})
                if parts == ["items"]:
                    filters = parse_filters(parse_qs(url.query))
                    return self.send_json(200, {"items": apply_filters(STORE.all(), filters)})
                if len(parts) == 2 and parts[0] == "items" and parts[1].isdigit():
                    item = STORE.get(int(parts[1]))
                    if item is None:
                        return self.send_json(404, {"error": f"no item {parts[1]}"})
                    return self.send_json(200, item)
                self.send_json(404, {"error": "not found"})

            def send_json(self, status: int, payload: object) -> None:
                body = json.dumps(payload, ensure_ascii=False, indent=2)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def log_message(self, fmt: str, *args) -> None:
                sys.stderr.write("%s %s\\n" % (self.address_string(), fmt % args))


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["port"])
            args = ap.parse_args()
            srv = ThreadingHTTPServer((cfg["host"], args.port), Handler)
            print(f"inventory API listening on http://{cfg['host']}:{args.port}", flush=True)
            srv.serve_forever()


        if __name__ == "__main__":
            main()
        ''',
    "app/query.py": '''
        """Query-string handling for GET /items."""

        DEFAULTS = {"tag": None, "limit": None}


        def parse_filters(qs: dict[str, list[str]], filters: dict = DEFAULTS) -> dict:
            """The filters one request asked for: `tag` (str) and `limit` (int); absent ones are None."""
            if "tag" in qs:
                filters["tag"] = qs["tag"][0]
            if "limit" in qs:
                filters["limit"] = int(qs["limit"][0])
            return filters


        def apply_filters(items: list[dict], filters: dict) -> list[dict]:
            out = [i for i in items if filters["tag"] is None or filters["tag"] in i["tags"]]
            return out if filters["limit"] is None else out[: filters["limit"]]
        ''',
    "app/store.py": '''
        """The item store: data/items.json, loaded once at startup."""

        from __future__ import annotations

        import json
        from pathlib import Path


        class Store:
            def __init__(self, path: Path):
                self.items = json.loads(path.read_text(encoding="utf-8"))

            def all(self) -> list[dict]:
                return list(self.items)

            def get(self, item_id: int) -> dict | None:
                return next((i for i in self.items if i["id"] == item_id), None)
        ''',
}


def api_bugs(work: Path) -> Task:
    lease = Lease(1)
    write(work, _INVENTORY)
    every = [dict(id=i, name=n, tags=t, price=p) for i, n, t, p in _ITEMS]
    (work / "data").mkdir()
    (work / "data" / "items.json").write_text(json.dumps(every, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config(work, port=lease.ports[0])
    fruit = [i for i in every if "fruit" in i["tags"]]

    def grade(s: Scratch) -> tuple[bool, str]:
        (port,) = s.ports(1)
        srv = s.start("server", "app/server.py", "--port", str(port))
        if not wait_port(port, srv):
            return False, f"server did not start: {srv.output()[-300:]}"

        def fetch(path: str) -> dict:
            status, headers, body = raw_get(port, path)
            if status != 200:
                raise AssertionError(f"{path}: HTTP {status}")
            if int(headers.get("content-length", -1)) != len(body):
                raise AssertionError(f"{path}: Content-Length {headers.get('content-length')} but {len(body)} bytes sent")
            return json.loads(body.decode("utf-8"))

        for path, want in (("/items?tag=fruit", fruit), ("/items", every), ("/items?limit=2", every[:2]),
                           ("/items", every), ("/items?tag=drink&limit=1", [every[3]]), ("/items", every)):
            got = fetch(path)["items"]
            if got != want:
                return False, f"{path}: {len(got)} items, want {len(want)}"
        if fetch("/items/4") != every[3] or fetch("/items/8") != every[7]:
            return False, "/items/<id> body differs from data/items.json"
        return True, "filters are per request; every body matches its Content-Length"

    return Task(
        prompt="The inventory API in `app/` has two bugs that users have reported:\n\n"
               "1. After anyone requests a filtered listing (for example `/items?tag=fruit`), later plain `/items` "
               "requests come back filtered too.\n"
               "2. Some responses arrive cut off, so clients fail to parse the JSON.\n\n"
               "Start the server (`python app/server.py`; it reads its port from `config.json`) and reproduce both "
               "bugs with curl before you change anything. Then fix them in `app/`, restart the server, and confirm "
               "with curl that both are gone. Don't change the routes or the shape of the responses.",
        check=lambda leftovers: checked(grade, work),
        fix=[("app/query.py", "def parse_filters(qs: dict[str, list[str]], filters: dict = DEFAULTS) -> dict:",
              "def parse_filters(qs: dict[str, list[str]], filters: dict | None = None) -> dict:"),
             ("app/query.py", '    if "tag" in qs:\n', '    filters = dict(DEFAULTS if filters is None else filters)\n    if "tag" in qs:\n'),
             ("app/server.py", "body = json.dumps(payload, ensure_ascii=False, indent=2)\n",
              'body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")\n'),
             ("app/server.py", 'self.wfile.write(body.encode("utf-8"))', "self.wfile.write(body)")],
        lease=lease, launches=("server.py",))


# ---------------------------------------------------------------- bg-queue-worker: two cooperating processes

_QUEUE = {
    "README.md": '''
        # Job queue

        Two processes share `jobs.db` (SQLite):

        - `python queue_api.py` serves the HTTP API on the port in `config.json`.
        - `python worker.py` claims queued jobs one at a time and stores each result.

        A job moves `queued` → `running` → `done`. For a job with text T its result is
        `{"words": <number of words in T>, "longest": <the longest word, the first one on a tie>}`.
        ''',
    "db.py": '''
        """The job table the API and the worker share."""

        from __future__ import annotations

        import sqlite3
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            result TEXT
        )"""


        def connect(path: str | Path = ROOT / "jobs.db") -> sqlite3.Connection:
            """A connection in autocommit mode: every statement commits on its own."""
            conn = sqlite3.connect(path, timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(SCHEMA)
            return conn
        ''',
    "queue_api.py": '''
        """Job queue HTTP API.

            python queue_api.py [--port N] [--db PATH]

            POST /jobs        {"text": "..."} -> 201 {"id": <int>, "status": "queued"}
            GET  /jobs/<id>   -> {"id": <int>, "status": "queued" | "running" | "done",
                                  "result": <the worker's result once done, else null>}
        """

        from __future__ import annotations

        import argparse
        import json
        import sys
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        import db

        ROOT = Path(__file__).resolve().parent


        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/jobs":
                    return self.send_json(404, {"error": "not found"})
                try:
                    text = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))["text"]
                except (ValueError, KeyError, TypeError):
                    return self.send_json(400, {"error": "want {\\"text\\": ...}"})
                conn = db.connect(self.server.db_path)
                try:
                    job_id = conn.execute("INSERT INTO jobs (text) VALUES (?)", (str(text),)).lastrowid
                finally:
                    conn.close()
                self.send_json(201, {"id": job_id, "status": "queued"})

            def do_GET(self) -> None:
                parts = [p for p in self.path.split("/") if p]
                if len(parts) != 2 or parts[0] != "jobs" or not parts[1].isdigit():
                    return self.send_json(404, {"error": "not found"})
                conn = db.connect(self.server.db_path)
                try:
                    row = conn.execute("SELECT id, status, result FROM jobs WHERE id = ?", (int(parts[1]),)).fetchone()
                finally:
                    conn.close()
                if row is None:
                    return self.send_json(404, {"error": f"no job {parts[1]}"})
                result = json.loads(row["result"]) if row["status"] == "complete" else None
                self.send_json(200, {"id": row["id"], "status": row["status"], "result": result})

            def send_json(self, status: int, payload: object) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                sys.stderr.write("api: %s\\n" % (fmt % args))


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["port"])
            ap.add_argument("--db", default=str(ROOT / "jobs.db"))
            args = ap.parse_args()
            db.connect(args.db).close()
            srv = ThreadingHTTPServer((cfg["host"], args.port), Handler)
            srv.db_path = args.db
            print(f"queue API on http://{cfg['host']}:{args.port} (db {args.db})", flush=True)
            srv.serve_forever()


        if __name__ == "__main__":
            main()
        ''',
    "worker.py": '''
        """The worker: claims queued jobs one at a time and stores each result.

            python worker.py [--db PATH] [--poll SECONDS]

        For a job with text T the result is {"words": <number of words in T>,
        "longest": <its longest word, the first one on a tie>}.
        """

        from __future__ import annotations

        import argparse
        import json
        import time
        from pathlib import Path

        import db

        ROOT = Path(__file__).resolve().parent


        def claim(conn):
            """The oldest queued job, now marked running; None if there is none (or another worker won it)."""
            row = conn.execute("SELECT id, text FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            won = conn.execute("UPDATE jobs SET status = 'running' WHERE id = ? AND status = 'queued'", (row["id"],)).rowcount
            return row if won == 1 else None


        def finish(conn, job_id: int, result: dict) -> None:
            conn.execute("UPDATE jobs SET status = 'done', result = ? WHERE id = ? AND status = 'queued'",
                         (json.dumps(result), job_id))


        def process(text: str) -> dict:
            words = text.split()
            return {"words": len(words), "longest": max(words, key=len) if words else ""}


        def main() -> None:
            ap = argparse.ArgumentParser()
            ap.add_argument("--db", default=str(ROOT / "jobs.db"))
            ap.add_argument("--poll", type=float, default=0.5)
            args = ap.parse_args()
            conn = db.connect(args.db)
            print(f"worker: polling {args.db} every {args.poll} s", flush=True)
            while True:
                row = claim(conn)
                if row is None:
                    time.sleep(args.poll)
                    continue
                finish(conn, row["id"], process(row["text"]))
                print(f"worker: finished job {row['id']}", flush=True)


        if __name__ == "__main__":
            main()
        ''',
}

_JOBS = ["the quick brown fox jumps", "hello world", "a bb ccc dd e", "stdin stdout stderr"]


def _words(text: str) -> dict:
    w = text.split()
    return {"words": len(w), "longest": max(w, key=len)}


def queue_worker(work: Path) -> Task:
    lease = Lease(1)
    write(work, _QUEUE)
    config(work, port=lease.ports[0])

    def grade(s: Scratch) -> tuple[bool, str]:
        (port,) = s.ports(1)
        dbp = str(s.dir / "check.db")
        api = s.start("api", "queue_api.py", "--port", str(port), "--db", dbp)
        if not wait_port(port, api):
            return False, f"API did not start: {api.output()[-300:]}"
        s.start("worker", "worker.py", "--db", dbp, "--poll", "0.1")
        ids = {}
        for text in _JOBS:
            t0 = time.monotonic()
            status, body = request(port, "POST", "/jobs", {"text": text})
            if status != 201 or time.monotonic() - t0 > 3:
                return False, f"POST /jobs: HTTP {status} after {time.monotonic() - t0:.1f}s"
            ids[json.loads(body)["id"]] = text
        deadline = time.monotonic() + 15
        last: dict = {}
        while time.monotonic() < deadline:
            last = {i: json.loads(request(port, "GET", f"/jobs/{i}")[1]) for i in ids}
            if all(j["status"] == "done" for j in last.values()):
                break
            time.sleep(0.2)
        for i, text in ids.items():
            j = last.get(i, {})
            if j.get("status") != "done" or j.get("result") != _words(text):
                return False, f"job {i}: {j}"
        return True, f"{len(ids)} jobs reached done with their results"

    return Task(
        prompt="Jobs submitted to our queue never finish: `POST /jobs` returns an id, but `GET /jobs/<id>` keeps "
               "reporting `running`. The system is two processes sharing `jobs.db`: the API (`python queue_api.py`, "
               "port in `config.json`) and the worker (`python worker.py`); see `README.md`. Run both, reproduce the "
               "problem with curl, and fix it so that a submitted job reaches `done` and `GET /jobs/<id>` returns its "
               "result. Don't change the HTTP API.",
        check=lambda leftovers: checked(grade, work),
        fix=[("worker.py", "result = ? WHERE id = ? AND status = 'queued'\"", "result = ? WHERE id = ? AND status = 'running'\""),
             ("queue_api.py", 'row["status"] == "complete"', 'row["status"] == "done"')],
        lease=lease, launches=("queue_api.py", "worker.py"))


# ---------------------------------------------------------------- bg-watch-tests: a watcher kept running

_TEXTKIT = {
    "textkit/__init__.py": '"""Small text utilities."""\n',
    "textkit/slug.py": '''
        import re
        import unicodedata


        def slugify(text: str) -> str:
            """A lowercase ASCII slug: accents stripped, every run of other characters one hyphen, no hyphen at
            either end. slugify("  Hello, World!  ") == "hello-world"."""
            text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
            return re.sub(r"[^a-z0-9]", "-", text.lower())
        ''',
    "textkit/wrap.py": '''
        def wrap(text: str, width: int) -> list[str]:
            """Greedy word wrap: whole words joined by single spaces, each line at most `width` characters
            (a word longer than `width` gets a line of its own)."""
            lines: list[str] = []
            cur = ""
            for word in text.split():
                if cur and len(cur) + 1 + len(word) > width + 1:
                    lines.append(cur)
                    cur = word
                else:
                    cur = f"{cur} {word}" if cur else word
            if cur:
                lines.append(cur)
            return lines
        ''',
    "textkit/stats.py": '''
        from collections import Counter


        def top_words(text: str, n: int) -> list[tuple[str, int]]:
            """The n most frequent words (case-insensitive, surrounding punctuation stripped), most frequent
            first, ties broken alphabetically."""
            words = [w.strip(".,;:!?\\"'()").lower() for w in text.split()]
            counts = Counter(w for w in words if w)
            return counts.most_common(n)
        ''',
    "textkit/roman.py": '''
        NUMERALS = [(1000, "M"), (500, "D"), (100, "C"), (50, "L"), (10, "X"), (5, "V"), (1, "I")]
        VALUES = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}


        def to_roman(n: int) -> str:
            """The Roman numeral for 1 <= n <= 3999, with subtractive pairs (IV, IX, XL, XC, CD, CM)."""
            if not 1 <= n <= 3999:
                raise ValueError(n)
            out = []
            for value, symbol in NUMERALS:
                while n >= value:
                    out.append(symbol)
                    n -= value
            return "".join(out)


        def from_roman(s: str) -> int:
            """The inverse of to_roman."""
            total = 0
            for a, b in zip(s, s[1:] + " "):
                v = VALUES[a]
                total += -v if b != " " and VALUES[b] > v else v
            return total
        ''',
    "watch.py": '''
        """Test watcher: reruns the suite whenever a .py file under textkit/ or tests/ changes.

            python watch.py [--interval SECONDS]

        Prints one summary line per run, then the tests that failed.
        """

        import argparse
        import re
        import subprocess
        import sys
        import time
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent


        def snapshot() -> dict:
            return {p: p.stat().st_mtime_ns for d in ("textkit", "tests") for p in sorted((ROOT / d).rglob("*.py"))}


        def run_suite(reason: str) -> None:
            p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                               cwd=ROOT, capture_output=True, text=True)
            ran = re.search(r"^Ran (\\d+) tests?", p.stderr, re.M)
            bad = sum(int(n) for n in re.findall(r"(?:failures|errors)=(\\d+)", p.stderr))
            total = int(ran.group(1)) if ran else 0
            stamp = time.strftime("%H:%M:%S")
            print(f"[watch {stamp}] {total - bad} passed, {bad} failed ({reason})", flush=True)
            for line in p.stderr.splitlines():
                if line.startswith(("FAIL:", "ERROR:")):
                    print(f"    {line}", flush=True)


        def main() -> None:
            ap = argparse.ArgumentParser()
            ap.add_argument("--interval", type=float, default=0.5)
            args = ap.parse_args()
            seen = snapshot()
            run_suite("start")
            while True:
                time.sleep(args.interval)
                now = snapshot()
                if now != seen:
                    changed = sorted({p.relative_to(ROOT).as_posix() for p in set(now) ^ set(seen)}
                                     | {p.relative_to(ROOT).as_posix() for p in now if seen.get(p) not in (None, now[p])})
                    seen = now
                    run_suite(", ".join(changed[:3]) + " changed")


        if __name__ == "__main__":
            main()
        ''',
}

_TEXTKIT_TESTS = {
    "tests/__init__.py": "",
    "tests/test_slug.py": '''
        import unittest

        from textkit.slug import slugify


        class TestSlug(unittest.TestCase):
            def test_simple(self):
                self.assertEqual(slugify("hello"), "hello")

            def test_punctuation_and_spaces(self):
                self.assertEqual(slugify("  Hello, World!  "), "hello-world")

            def test_accents(self):
                self.assertEqual(slugify("Crème Brûlée -- Recipe"), "creme-brulee-recipe")
        ''',
    "tests/test_wrap.py": '''
        import unittest

        from textkit.wrap import wrap


        class TestWrap(unittest.TestCase):
            def test_fits(self):
                self.assertEqual(wrap("a b c", 10), ["a b c"])

            def test_width_is_a_maximum(self):
                text = "the quick brown fox jumps over the lazy dog"
                self.assertEqual(wrap(text, 9), ["the quick", "brown fox", "jumps", "over the", "lazy dog"])
                self.assertTrue(all(len(line) <= 9 for line in wrap(text, 9)))

            def test_long_word(self):
                self.assertEqual(wrap("a extraordinarily b", 5), ["a", "extraordinarily", "b"])
        ''',
    "tests/test_stats.py": '''
        import unittest

        from textkit.stats import top_words


        class TestStats(unittest.TestCase):
            def test_counts(self):
                self.assertEqual(top_words("b a b c b a", 2), [("b", 3), ("a", 2)])

            def test_case_and_punctuation(self):
                self.assertEqual(top_words("Dog, dog. DOG!", 1), [("dog", 3)])

            def test_ties_alphabetical(self):
                self.assertEqual(top_words("pear fig apple fig pear apple kiwi", 3),
                                 [("apple", 2), ("fig", 2), ("pear", 2)])
        ''',
    "tests/test_roman.py": '''
        import unittest

        from textkit.roman import from_roman, to_roman


        class TestRoman(unittest.TestCase):
            def test_additive(self):
                self.assertEqual(to_roman(3), "III")
                self.assertEqual(to_roman(2026), "MMXXVI")

            def test_subtractive(self):
                self.assertEqual([to_roman(n) for n in (4, 9, 40, 90, 400, 900)], ["IV", "IX", "XL", "XC", "CD", "CM"])

            def test_mixed(self):
                self.assertEqual(to_roman(1994), "MCMXCIV")

            def test_round_trip(self):
                for n in range(1, 4000):
                    self.assertEqual(from_roman(to_roman(n)), n)

            def test_range(self):
                with self.assertRaises(ValueError):
                    to_roman(0)
        ''',
}


def _suite(s: Scratch, tests: dict[str, str], module: str, want: int) -> tuple[bool, str]:
    """Restore `tests` from the task's own copy, run `module` (unittest), and require `want` tests
    run and passed."""
    shutil.rmtree(s.root / "tests", ignore_errors=True)
    write(s.root, tests)
    p = s.run("-m", "unittest", *module.split(), timeout=120)
    ran = [l for l in p.stderr.splitlines() if l.startswith("Ran ")]
    ok = p.returncode == 0 and bool(ran) and ran[-1].split()[1] == str(want)
    return ok, (ran[-1] if ran else "no unittest summary") + ("" if ok else f": {p.stderr[-300:]}")


def watch_tests(work: Path) -> Task:
    write(work, {**_TEXTKIT, **_TEXTKIT_TESTS})
    return Task(
        prompt="The test suite in `tests/` has failures. Start the test watcher with `python watch.py` and keep it "
               "running while you work: it reruns the suite whenever a file under `textkit/` or `tests/` changes "
               "and prints a summary. Fix the library code in `textkit/` until the watcher reports every test "
               "passing. Don't modify the tests.",
        check=lambda leftovers: checked(lambda s: _suite(s, _TEXTKIT_TESTS, "discover -s tests -t .", 14), work),
        fix=[("textkit/slug.py", 're.sub(r"[^a-z0-9]", "-", text.lower())', 're.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")'),
             ("textkit/wrap.py", "> width + 1:", "> width:"),
             ("textkit/stats.py", "return counts.most_common(n)", "return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]"),
             ("textkit/roman.py", 'NUMERALS = [(1000, "M"), (500, "D"), (100, "C"), (50, "L"), (10, "X"), (5, "V"), (1, "I")]',
              'NUMERALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),\n'
              '            (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]')],
        launches=("watch.py",))


# ---------------------------------------------------------------- bg-long-build: a slow build plus other work

BUILD_MODULES = 40
COMPILE_SECONDS = 2.5  # per module: the build takes BUILD_MODULES × this, about 100 s

_BUILD = {
    "build.py": '''
        """Release build: compiles every module under modules/ into dist/bundle.json.

            python build.py

        Each module takes a couple of seconds to compile, the whole build a little under two minutes.
        The bundle's checksum (printed last, and stored in the bundle) covers every compiled module, in order.
        """

        import hashlib
        import json
        import time
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        COMPILE_SECONDS = 2.5  # stands in for the real compiler's time per module


        def compile_module(source: str) -> str:
            digest = source.encode()
            for _ in range(20000):
                digest = hashlib.sha256(digest).digest()
            return digest.hex()


        def main() -> None:
            modules = sorted((ROOT / "modules").glob("*.mod"))
            out = []
            t0 = time.monotonic()
            for i, path in enumerate(modules, 1):
                start = time.monotonic()
                digest = compile_module(path.read_text())
                time.sleep(max(0.0, COMPILE_SECONDS - (time.monotonic() - start)))
                out.append({"module": path.stem, "digest": digest})
                print(f"[{i}/{len(modules)}] compiled {path.stem} ({time.monotonic() - t0:.0f}s)", flush=True)
            checksum = hashlib.sha256("".join(o["digest"] for o in out).encode()).hexdigest()
            (ROOT / "dist").mkdir(exist_ok=True)
            (ROOT / "dist" / "bundle.json").write_text(json.dumps({"modules": out, "checksum": checksum}, indent=2) + "\\n")
            print(f"BUILD OK: {len(out)} modules, checksum {checksum}", flush=True)


        if __name__ == "__main__":
            main()
        ''',
    "tools/__init__.py": "",
    "tools/duration.py": '''
        def parse_duration(text: str) -> int:
            """The number of seconds in a duration such as "90s", "15m", "2h", "1h30m" or "1d2h3m4s".

            Units are d, h, m and s, in that order, each at most once, at least one, each with a whole
            number before it. Surrounding whitespace is ignored. Anything else raises ValueError,
            including "", "h", "5x", "1m2h" and "1h1h".
            """
            raise NotImplementedError
        ''',
    "VERSION": "2.2.1\n",
    "RELEASE.md": "# Release 2.3.0\n\nchecksum: (pending)\n",
}

_BUILD_TESTS = {
    "tests/__init__.py": "",
    "tests/test_duration.py": '''
        import unittest

        from tools.duration import parse_duration


        class TestDuration(unittest.TestCase):
            def test_single_units(self):
                self.assertEqual([parse_duration(s) for s in ("90s", "15m", "2h", "1d")], [90, 900, 7200, 86400])

            def test_combined(self):
                self.assertEqual(parse_duration("1h30m"), 5400)
                self.assertEqual(parse_duration("1d2h3m4s"), 93784)

            def test_whitespace(self):
                self.assertEqual(parse_duration("  10m "), 600)

            def test_zero(self):
                self.assertEqual(parse_duration("0s"), 0)

            def test_rejects(self):
                for bad in ("", "h", "5x", "1m2h", "1h1h", "1.5h", "-3s", "10"):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        parse_duration(bad)
        ''',
}


def _modules() -> dict[str, str]:
    rng = random.Random(4242)
    words = ["alpha", "beta", "gamma", "delta", "kernel", "buffer", "socket", "parser", "lexer", "token", "frame",
             "cache", "queue", "stream", "index", "shard"]
    return {f"modules/m{i:02d}_{rng.choice(words)}.mod": "\n".join(" ".join(rng.choice(words) for _ in range(12))
                                                                  for _ in range(30)) + "\n"
            for i in range(1, BUILD_MODULES + 1)}


def bundle(modules: dict[str, str]) -> str:
    """dist/bundle.json as build.py writes it, computed without its per-module wait."""
    out = []
    for rel in sorted(modules):
        digest = modules[rel].encode()
        for _ in range(20000):
            digest = hashlib.sha256(digest).digest()
        out.append({"module": Path(rel).stem, "digest": digest.hex()})
    checksum = hashlib.sha256("".join(o["digest"] for o in out).encode()).hexdigest()
    return json.dumps({"modules": out, "checksum": checksum}, indent=2) + "\n"


def long_build(work: Path) -> Task:
    mods = _modules()
    write(work, {**_BUILD, **_BUILD_TESTS})
    write(work, mods)
    want = bundle(mods)
    checksum = json.loads(want)["checksum"]

    def grade(s: Scratch) -> tuple[bool, str]:
        b = work / "dist" / "bundle.json"
        if not b.exists() or json.loads(b.read_text()) != json.loads(want):
            return False, "dist/bundle.json missing or not the build's output"
        if f"checksum: {checksum}" not in (work / "RELEASE.md").read_text().splitlines():
            return False, "RELEASE.md has no `checksum: <the build's checksum>` line"
        if (work / "VERSION").read_text().strip() != "2.3.0":
            return False, f"VERSION is {(work / 'VERSION').read_text().strip()!r}"
        ok, why = _suite(s, _BUILD_TESTS, "tests.test_duration", 5)
        return ok, f"bundle, checksum and version right; {why}"

    return Task(
        prompt="Prepare release 2.3.0:\n\n"
               "1. Run the release build, `python build.py`. It takes about two minutes and writes `dist/bundle.json`. "
               "When it has finished, put its checksum in `RELEASE.md` as the line `checksum: <hex>` (replacing the "
               "placeholder).\n"
               "2. Implement `parse_duration` in `tools/duration.py` (see its docstring) so that "
               "`python -m unittest tests.test_duration` passes. Don't change the tests.\n"
               "3. Set `VERSION` to `2.3.0`.\n\n"
               "The build reads only `modules/`, so steps 2 and 3 don't affect it.",
        check=lambda leftovers: checked(grade, work, skip=("modules",)),
        fix=[("tools/duration.py", "    raise NotImplementedError\n",
              '    import re\n\n'
              '    m = re.fullmatch(r"(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", text.strip())\n'
              '    if not m or not text.strip():\n'
              '        raise ValueError(text)\n'
              '    d, h, mins, s = (int(g or 0) for g in m.groups())\n'
              '    return ((d * 24 + h) * 60 + mins) * 60 + s\n'),
             ("VERSION", "2.2.1", "2.3.0")],
        answers={"dist/bundle.json": want, "RELEASE.md": f"# Release 2.3.0\n\nchecksum: {checksum}\n"},
        launches=("build.py",))


# ---------------------------------------------------------------- bg-dev-server: "start it, check it"

_DEVSERVER = {
    "serve.py": '''
        """Orders dashboard dev server.

            python serve.py [--port N]

        Loads data/orders.csv (warming its cache takes a few seconds), then serves:

            GET /            the dashboard (HTML)
            GET /api/stats   {"orders": <number of orders>,
                              "revenue": <sum of quantity * unit_price over every order, rounded to 2 places>,
                              "average_order": <revenue / orders, rounded to 2 places>,
                              "top_customer": <the customer with the highest revenue>}

        Every request is logged to stdout.
        """

        from __future__ import annotations

        import argparse
        import csv
        import json
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent


        def load_orders(path: Path) -> list[dict]:
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            header, body = rows[0], rows[1:-1]
            orders = [dict(zip(header, r)) for r in body]
            for o in orders:
                o["quantity"] = int(o["quantity"])
                o["unit_price"] = float(o["unit_price"])
            return orders


        def stats(orders: list[dict]) -> dict:
            revenue = 0.0
            by_customer: dict[str, float] = {}
            for o in orders:
                amount = o["quantity"] * o["unit_price"]
                revenue += amount
                by_customer[o["customer"]] = by_customer.get(o["customer"], 0.0) + amount
            return {"orders": len(orders), "revenue": round(revenue, 2),
                    "average_order": round(revenue // len(orders), 2),
                    "top_customer": max(by_customer, key=by_customer.get)}


        PAGE = """<!doctype html><title>Orders</title><h1>Orders dashboard</h1>
        <p>Live numbers: <a href="/api/stats">/api/stats</a></p>"""


        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/api/stats":
                    body, kind = json.dumps(stats(self.server.orders)).encode(), "application/json"
                elif self.path == "/":
                    body, kind = PAGE.encode(), "text/html"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["port"])
            args = ap.parse_args()
            orders = load_orders(ROOT / "data" / "orders.csv")
            for step in range(1, 5):
                print(f"warming cache {step}/4 ...", flush=True)
                time.sleep(1)
            srv = ThreadingHTTPServer((cfg["host"], args.port), Handler)
            srv.orders = orders
            print(f"dashboard ready on http://{cfg['host']}:{args.port}/ ({len(orders)} orders)", flush=True)
            srv.serve_forever()


        if __name__ == "__main__":
            main()
        ''',
}


def _orders() -> list[dict]:
    rng = random.Random(77)
    customers = ["acme", "globex", "initech", "umbrella", "hooli", "stark"]
    rows = [dict(order_id=1000 + i, customer=rng.choice(customers[:5]), quantity=rng.randint(1, 9),
                 unit_price=f"{rng.randint(199, 4999) / 100:.2f}") for i in range(47)]
    rows.append(dict(order_id=1047, customer="stark", quantity=40, unit_price="49.99"))  # the last order decides the top
    return rows


def _stats(rows: list[dict]) -> dict:
    revenue = 0.0
    by: dict[str, float] = {}
    for r in rows:
        amount = int(r["quantity"]) * float(r["unit_price"])
        revenue += amount
        by[r["customer"]] = by.get(r["customer"], 0.0) + amount
    return {"orders": len(rows), "revenue": round(revenue, 2), "average_order": round(revenue / len(rows), 2),
            "top_customer": max(by, key=by.get)}


def dev_server(work: Path) -> Task:
    lease = Lease(1)
    write(work, _DEVSERVER)
    rows = _orders()
    (work / "data").mkdir()
    lines = ["order_id,customer,quantity,unit_price"] + [f"{r['order_id']},{r['customer']},{r['quantity']},{r['unit_price']}" for r in rows]
    (work / "data" / "orders.csv").write_text("\n".join(lines) + "\n")
    config(work, port=lease.ports[0])
    want = _stats(rows)

    def grade(s: Scratch) -> tuple[bool, str]:
        (port,) = s.ports(1)
        srv = s.start("server", "serve.py", "--port", str(port))
        if not wait_port(port, srv, timeout=30):
            return False, f"server did not start: {srv.output()[-300:]}"
        status, body = request(port, "GET", "/api/stats")
        got = json.loads(body) if status == 200 else {}
        ok = (status == 200 and got.get("orders") == want["orders"] and got.get("top_customer") == want["top_customer"]
              and abs(float(got.get("revenue", -1)) - want["revenue"]) < 0.005
              and abs(float(got.get("average_order", -1)) - want["average_order"]) < 0.0105)
        return ok, f"/api/stats {got} (want {want})"

    return Task(
        prompt="Start the dashboard dev server (`python serve.py`; it reads its port from `config.json`) and check "
               "`/api/stats` with curl. Its numbers don't match `data/orders.csv`. Work out the correct figures from "
               "the CSV, fix the server so `/api/stats` reports them as `serve.py`'s docstring describes, then "
               "restart it and confirm with curl.",
        check=lambda leftovers: checked(grade, work),
        fix=[("serve.py", "rows[1:-1]", "rows[1:]"), ("serve.py", "round(revenue // len(orders), 2)", "round(revenue / len(orders), 2)")],
        lease=lease, launches=("serve.py",))


# ---------------------------------------------------------------- bg-port-conflict: the configured port is taken

_METRICS = {
    "README.md": '''
        # Metrics service

        `python metrics_service.py` listens on the port in `config.json`. Override it with `--port N` or the
        `PORT` environment variable; `config.json` is shared with the deploy tooling, so leave it as it is.
        ''',
    "metrics_service.py": '''
        """Metrics service.

            python metrics_service.py [--port N]

        Listens on the port in config.json; --port or the PORT environment variable override it.

            GET /health    {"status": "ok"}
            GET /metrics   Prometheus text format, one line per path requested so far (sorted by path):
                           requests_total{path="<path>"} <count>
                           /metrics itself is not counted.
        """

        from __future__ import annotations

        import argparse
        import json
        import os
        import sys
        import threading
        import traceback
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path
        from urllib.parse import urlsplit

        ROOT = Path(__file__).resolve().parent
        COUNTS: dict[str, int] = {}
        LOCK = threading.Lock()


        def render(counts: dict[str, int]) -> str:
            lines = []
            for path, n in sorted(counts):
                lines.append(f'requests_total{{path="{path}"}} {n}')
            return "\\n".join(lines) + "\\n"


        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                try:
                    path = urlsplit(self.path).path
                    if path == "/metrics":
                        with LOCK:
                            text = render(dict(COUNTS))
                        return self.reply(200, "text/plain; version=0.0.4", text)
                    with LOCK:
                        COUNTS[path] = COUNTS.get(path, 0) + 1
                    if path == "/health":
                        return self.reply(200, "application/json", '{"status": "ok"}\\n')
                    self.reply(404, "application/json", '{"error": "not found"}\\n')
                except Exception:
                    traceback.print_exc()
                    self.reply(500, "application/json", '{"error": "internal error"}\\n')

            def reply(self, status: int, kind: str, text: str) -> None:
                body = text.encode()
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                sys.stderr.write("%s\\n" % (fmt % args))


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", cfg["port"])))
            args = ap.parse_args()
            srv = ThreadingHTTPServer((cfg["host"], args.port), Handler)
            print(f"metrics service on http://{cfg['host']}:{args.port}", flush=True)
            srv.serve_forever()


        if __name__ == "__main__":
            main()
        ''',
}


def port_conflict(work: Path) -> Task:
    lease = Lease(1)
    write(work, _METRICS)
    config(work, port=lease.ports[0])
    cfg = (work / "config.json").read_bytes()
    stop = _squatter(lease.ports[0])

    def grade(s: Scratch) -> tuple[bool, str]:
        if (work / "config.json").read_bytes() != cfg:
            return False, "config.json was changed"
        (port,) = s.ports(1)
        srv = s.start("service", "metrics_service.py", "--port", str(port))
        if not wait_port(port, srv):
            return False, f"service did not start: {srv.output()[-300:]}"
        for path in ("/health", "/health", "/nope"):
            request(port, "GET", path)
        status, body = request(port, "GET", "/metrics")
        want = 'requests_total{path="/health"} 2\nrequests_total{path="/nope"} 1\n'
        return status == 200 and body.decode() == want, f"/metrics HTTP {status}: {body.decode()[:200]!r}"

    return Task(
        prompt="Start the metrics service (`python metrics_service.py`) and use curl to check `/health` and "
               "`/metrics`. `/metrics` is broken: find out why and fix it, then show with curl that `/metrics` "
               "reports the requests you made. Don't edit `config.json`: other tools read it.",
        check=lambda leftovers: checked(grade, work),
        fix=[("metrics_service.py", "in sorted(counts):", "in sorted(counts.items()):")],
        lease=lease, helpers=[stop], launches=("metrics_service.py",))


# ---------------------------------------------------------------- bg-log-errors: an error only the server log shows

_PRICING = {
    "pricing.py": '''
        """Quote pricing.

        CATALOG holds unit prices in cents. A quote for `qty` units of `sku` is qty x unit price, then:

        - a bulk discount: 5% off for 100-199 units, 10% off for 200 units or more;
        - an optional coupon: COUPONS maps a code to a percentage off, applied after the bulk discount
          (a 100% coupon makes the order free);
        - the total is rounded half up to whole cents.

        quote() returns {"total_cents": <int>, "savings_pct": <percent of the undiscounted price
        (qty x unit price) saved, rounded to one decimal place>}. An unknown SKU or coupon raises
        KeyError, which the server answers with 400.
        """

        from __future__ import annotations

        CATALOG = {"A100": 1250, "B200": 899, "C300": 4999, "D400": 120, "E500": 15000}
        COUPONS = {"SAVE10": 10, "SAVE25": 25, "STAFF50": 50, "FREE100": 100}
        BULK = [0, 5, 10]  # percent off by hundreds of units: 0-99, 100-199, 200 and up


        def quote(sku: str, qty: int, coupon: str | None = None) -> dict:
            full = CATALOG[sku] * qty
            total = full * (100 - BULK[qty // 100]) / 100
            if coupon:
                total = total * (100 - COUPONS[coupon]) / 100
            cents = int(total + 0.5)
            return {"total_cents": cents, "savings_pct": round(100 * (full - cents) / cents, 1)}
        ''',
    "pricing_server.py": '''
        """Pricing HTTP server.

            python pricing_server.py [--port N] [--log PATH]

            POST /quote   {"sku": "A100", "qty": 3, "coupon": "SAVE10" (optional)} -> pricing.quote()'s result

        Logs every request to logs/server.log (and any error with its traceback); prints nothing after startup.
        """

        from __future__ import annotations

        import argparse
        import json
        import logging
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        import pricing

        ROOT = Path(__file__).resolve().parent
        log = logging.getLogger("pricing")


        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/quote":
                    return self.reply(404, {"error": "not found"})
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                    sku, qty, coupon = req["sku"], int(req["qty"]), req.get("coupon")
                except (ValueError, KeyError, TypeError):
                    return self.reply(400, {"error": "bad request"})
                try:
                    result = pricing.quote(sku, qty, coupon)
                except KeyError as e:
                    log.info("POST /quote sku=%s qty=%s coupon=%s -> 400 unknown %s", sku, qty, coupon, e)
                    return self.reply(400, {"error": f"unknown {e}"})
                except Exception:
                    log.exception("POST /quote sku=%s qty=%s coupon=%s -> 500", sku, qty, coupon)
                    return self.reply(500, {"error": "internal error"})
                log.info("POST /quote sku=%s qty=%s coupon=%s -> 200 %s", sku, qty, coupon, result["total_cents"])
                self.reply(200, result)

            def reply(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["port"])
            ap.add_argument("--log", default=str(ROOT / "logs" / "server.log"))
            args = ap.parse_args()
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(filename=args.log, level=logging.INFO,
                                format="%(asctime)s %(levelname)s %(name)s: %(message)s")
            srv = ThreadingHTTPServer((cfg["host"], args.port), Handler)
            print(f"pricing server on http://{cfg['host']}:{args.port}, logging to {args.log}", flush=True)
            log.info("started on port %s", args.port)
            srv.serve_forever()


        if __name__ == "__main__":
            main()
        ''',
    "load.py": '''
        """Replays a day of quote traffic against the pricing server and reports how it went.

            python load.py [--port N]
        """

        import argparse
        import http.client
        import json
        import random
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent


        def traffic(n: int = 300):
            rng = random.Random(2026)
            for _ in range(n):
                r = rng.random()
                qty = rng.randint(1, 99) if r < 0.9 else rng.randint(100, 299) if r < 0.96 else rng.randint(300, 400)
                c = rng.random()
                coupon = (None if c < 0.65 else "SAVE10" if c < 0.8 else "SAVE25" if c < 0.9 else "STAFF50" if c < 0.95
                          else "FREE100" if c < 0.98 else "BOGUS")
                yield {"sku": rng.choice(["A100", "B200", "C300", "D400", "E500"]), "qty": qty, "coupon": coupon}


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["port"])
            args = ap.parse_args()
            counts = {"ok": 0, "rejected": 0, "failed": 0}
            reqs = list(traffic())
            for i, body in enumerate(reqs, 1):
                conn = http.client.HTTPConnection(cfg["host"], args.port, timeout=10)
                conn.request("POST", "/quote", body=json.dumps(body), headers={"Content-Type": "application/json"})
                status = conn.getresponse().status
                conn.close()
                counts["ok" if status == 200 else "rejected" if status < 500 else "failed"] += 1
                if i % 100 == 0:
                    print(f"{i}/{len(reqs)} sent", flush=True)
            print(f"{len(reqs)} requests: {counts['ok']} ok, {counts['rejected']} rejected (4xx), "
                  f"{counts['failed']} failed (5xx)", flush=True)


        if __name__ == "__main__":
            main()
        ''',
}


def _quote(sku: str, qty: int, coupon: str | None) -> dict:
    catalog = {"A100": 1250, "B200": 899, "C300": 4999, "D400": 120, "E500": 15000}
    coupons = {"SAVE10": 10, "SAVE25": 25, "STAFF50": 50, "FREE100": 100}
    full = catalog[sku] * qty
    total = full * (100 - (0 if qty < 100 else 5 if qty < 200 else 10)) / 100
    if coupon:
        total = total * (100 - coupons[coupon]) / 100
    cents = int(total + 0.5)
    return {"total_cents": cents, "savings_pct": round(100 * (full - cents) / full, 1)}


def log_errors(work: Path) -> Task:
    lease = Lease(1)
    write(work, _PRICING)
    config(work, port=lease.ports[0])
    probes = [("A100", 350, None), ("B200", 10, "SAVE25"), ("C300", 120, "FREE100"), ("E500", 250, "STAFF50"),
              ("D400", 7, None), ("A100", 199, "SAVE10")]

    def grade(s: Scratch) -> tuple[bool, str]:
        (port,) = s.ports(1)
        logp = s.dir / "server.log"
        srv = s.start("server", "pricing_server.py", "--port", str(port), "--log", str(logp))
        if not wait_port(port, srv):
            return False, f"server did not start: {srv.output()[-300:]}"
        load = s.run("load.py", "--port", str(port), timeout=120)
        summary = load.stdout.strip().splitlines()[-1:] or ["(no output)"]
        if load.returncode != 0 or " 0 failed" not in summary[0]:
            return False, f"load: {summary[0]} {load.stderr[-200:]}"
        for sku, qty, coupon in probes:
            status, body = request(port, "POST", "/quote", {"sku": sku, "qty": qty, "coupon": coupon})
            if status != 200 or json.loads(body) != _quote(sku, qty, coupon):
                return False, f"quote({sku}, {qty}, {coupon}): HTTP {status} {body[:120]!r}, want {_quote(sku, qty, coupon)}"
        if " ERROR " in logp.read_text():
            return False, "the server logged an error"
        return True, f"{summary[0]}; {len(probes)} quotes match the docstring"

    return Task(
        prompt="Customers occasionally get HTTP 500 from the pricing server, and nobody can reproduce it by hand. "
               "Start the server (`python pricing_server.py`, port in `config.json`; it logs to `logs/server.log`), "
               "then run `python load.py`, which replays a day of traffic against it and reports how many requests "
               "failed. Find the causes in the server log, fix them following the docstring in `pricing.py`, and rerun "
               "the load until nothing fails.",
        check=lambda leftovers: checked(grade, work),
        fix=[("pricing.py", "BULK[qty // 100]", "BULK[min(qty // 100, 2)]"),
             ("pricing.py", "(full - cents) / cents, 1)", "(full - cents) / full, 1)")],
        lease=lease, launches=("pricing_server.py",))


# ---------------------------------------------------------------- bg-cleanup: two services, leave nothing running

_PROFILES = [("ada", 1990, "London"), ("linus", 1969, "Helsinki"), ("grace", 1985, "Arlington"),
             ("ken", 1993, "Berkeley"), ("margaret", 1986, "Boston")]

_STACK = {
    "kv.py": '''
        """A tiny in-memory key-value store over HTTP.

            python kv.py [--port N]       (default: kv_port in config.json)

            PUT /kv/<key>   store the request body          -> 204
            GET /kv/<key>   the stored body, or 404
        """

        import argparse
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        DATA: dict[str, bytes] = {}


        class Handler(BaseHTTPRequestHandler):
            def do_PUT(self) -> None:
                DATA[self.path.removeprefix("/kv/")] = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(204)
                self.end_headers()

            def do_GET(self) -> None:
                value = DATA.get(self.path.removeprefix("/kv/"))
                if value is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(value)))
                self.end_headers()
                self.wfile.write(value)

            def log_message(self, *args) -> None:
                pass


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["kv_port"])
            args = ap.parse_args()
            print(f"kv store on {cfg['host']}:{args.port}", flush=True)
            ThreadingHTTPServer((cfg["host"], args.port), Handler).serve_forever()


        if __name__ == "__main__":
            main()
        ''',
    "seed.py": '''
        """Loads the user records into the kv store: key user:<name>, value JSON {"name", "born", "city"}.

            python seed.py [--kv-port N]
        """

        import argparse
        import http.client
        import json
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        USERS = USERS_LITERAL


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--kv-port", type=int, default=cfg["kv_port"])
            args = ap.parse_args()
            for name, born, city in USERS:
                conn = http.client.HTTPConnection(cfg["host"], args.kv_port, timeout=10)
                conn.request("PUT", f"/kv/user:{name}", body=json.dumps({"name": name, "born": born, "city": city}))
                conn.getresponse().read()
                conn.close()
            print(f"seeded {len(USERS)} users", flush=True)


        if __name__ == "__main__":
            main()
        ''',
    "api.py": '''
        """Profile API, backed by the kv store.

            python api.py [--port N] [--kv-port N]     (defaults: api_port and kv_port in config.json)

            GET /profile/<name>  -> {"name": <name>, "age": <age in YEAR>, "city": <city>}, or 404 for an unknown user
        """

        import argparse
        import http.client
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        YEAR = 2026


        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if not self.path.startswith("/profile/"):
                    return self.reply(404, {"error": "not found"})
                name = self.path.removeprefix("/profile/")
                conn = http.client.HTTPConnection(self.server.cfg["host"], self.server.kv_port, timeout=10)
                conn.request("GET", f"/kv/users:{name}")
                r = conn.getresponse()
                raw = r.read()
                conn.close()
                if r.status != 200:
                    return self.reply(404, {"error": f"unknown user {name}"})
                user = json.loads(raw)
                self.reply(200, {"name": user["name"], "age": user["born"] - YEAR, "city": user["city"]})

            def reply(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["api_port"])
            ap.add_argument("--kv-port", type=int, default=cfg["kv_port"])
            args = ap.parse_args()
            srv = ThreadingHTTPServer((cfg["host"], args.port), Handler)
            srv.cfg, srv.kv_port = cfg, args.kv_port
            print(f"profile API on {cfg['host']}:{args.port} (kv on {args.kv_port})", flush=True)
            srv.serve_forever()


        if __name__ == "__main__":
            main()
        ''',
    "smoke.py": '''
        """Smoke test for the profile stack (kv store seeded, API up).

            python smoke.py [--port N]     (default: api_port in config.json)

        Prints SMOKE OK <digest of the profiles> or SMOKE FAIL <why>.
        """

        import argparse
        import hashlib
        import http.client
        import json
        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        NAMES = ["ada", "linus", "grace", "ken", "margaret"]


        def main() -> None:
            cfg = json.loads((ROOT / "config.json").read_text())
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=cfg["api_port"])
            args = ap.parse_args()
            profiles = []
            for name in NAMES:
                conn = http.client.HTTPConnection(cfg["host"], args.port, timeout=10)
                conn.request("GET", f"/profile/{name}")
                r = conn.getresponse()
                body = r.read()
                conn.close()
                if r.status != 200:
                    sys.exit(f"SMOKE FAIL /profile/{name}: HTTP {r.status} {body[:100]!r}")
                p = json.loads(body)
                if p.get("name") != name or not 0 < p.get("age", -1) < 120:
                    sys.exit(f"SMOKE FAIL /profile/{name}: {p}")
                profiles.append(p)
            digest = hashlib.sha256(json.dumps(profiles, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            print(f"SMOKE OK {digest[:16]}", flush=True)


        if __name__ == "__main__":
            main()
        ''',
}


def smoke_line() -> str:
    profiles = [{"name": n, "age": 2026 - b, "city": c} for n, b, c in _PROFILES]
    digest = hashlib.sha256(json.dumps(profiles, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"SMOKE OK {digest[:16]}"


def cleanup(work: Path) -> Task:
    lease = Lease(2)
    files = dict(_STACK)
    files["seed.py"] = files["seed.py"].replace("USERS_LITERAL", repr([list(u) for u in _PROFILES]))
    write(work, files)
    config(work, kv_port=lease.ports[0], api_port=lease.ports[1])
    want = smoke_line()

    def grade(s: Scratch) -> tuple[bool, str]:
        got = (work / "smoke.txt").read_text().strip() if (work / "smoke.txt").exists() else ""
        if got != want:
            return False, f"smoke.txt is {got[:80]!r}, want {want!r}"
        kv_port, api_port = s.ports(2)
        kv = s.start("kv", "kv.py", "--port", str(kv_port))
        if not wait_port(kv_port, kv):
            return False, f"kv did not start: {kv.output()[-200:]}"
        seed = s.run("seed.py", "--kv-port", str(kv_port), timeout=30)
        if seed.returncode != 0:
            return False, f"seed failed: {seed.stderr[-200:]}"
        api = s.start("api", "api.py", "--port", str(api_port), "--kv-port", str(kv_port))
        if not wait_port(api_port, api):
            return False, f"api did not start: {api.output()[-200:]}"
        smoke = s.run("smoke.py", "--port", str(api_port), timeout=30)
        line = (smoke.stdout + smoke.stderr).strip()
        return line == want, f"fresh smoke run: {line[:120]!r}"

    def check(leftovers: list[dict] | None) -> tuple[bool, str]:
        ok, why = checked(grade, work)
        if leftovers is None:
            return False, "no leftover report (the run was not sandboxed)"
        if leftovers:
            return False, f"{len(leftovers)} process(es) left running: {[p['cmd'][:60] for p in leftovers[:3]]}; {why}"
        return ok, f"nothing left running; {why}"

    return Task(
        prompt="Bring up the profile stack and get its smoke test passing. Start the key-value store "
               "(`python kv.py`), seed it (`python seed.py`), start the API (`python api.py`), then run "
               "`python smoke.py` (ports are in `config.json`). The smoke test fails: fix the bugs, get it to print "
               "`SMOKE OK ...`, and write that line to `smoke.txt`. When you are done, stop every process you "
               "started: leave nothing running.",
        check=check,
        fix=[("api.py", 'f"/kv/users:{name}"', 'f"/kv/user:{name}"'), ("api.py", 'user["born"] - YEAR', 'YEAR - user["born"]')],
        answers={"smoke.txt": want + "\n"},
        lease=lease, launches=("kv.py", "api.py"))


BG_TASKS: dict[str, Callable[[Path], Task]] = {
    "bg-api-bugs": api_bugs,
    "bg-queue-worker": queue_worker,
    "bg-watch-tests": watch_tests,
    "bg-long-build": long_build,
    "bg-dev-server": dev_server,
    "bg-port-conflict": port_conflict,
    "bg-log-errors": log_errors,
    "bg-cleanup": cleanup,
}
