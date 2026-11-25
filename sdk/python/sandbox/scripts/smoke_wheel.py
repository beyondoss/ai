"""A wheel's smoke test, run by CI on the platform runner that built it (cibuildwheel's test command).

It imports the *installed* package (never the checkout's), loads the native kernel the wheel carries, and
follows a job over loopback with a cut stream: so ctypes, the library, the transport and the driver all work
on this platform, and the handle reattaches by the kernel's decision. It needs nothing but the standard
library.
"""

from __future__ import annotations

import base64
import json
import platform
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import beyond_sandbox
from beyond_sandbox import HttpTransport, Jobs, _kernel

MARK = "00ff1a"
OUTPUT = b"0123456789"


def check(cond: bool, what: str) -> None:
    if not cond:
        print(f"FAIL: {what}", file=sys.stderr)
        raise SystemExit(1)


class Endpoint(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    attaches: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        job = body["job"]
        Endpoint.attaches.append(job)
        first = len(Endpoint.attaches) == 1
        at = job["offset"]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        lines = [{"attached": {"mark": MARK, "from": at}}]
        # The first stream sends four bytes and is cut; the reattach sends the rest.
        piece = OUTPUT[at : at + 4] if first else OUTPUT[at:]
        lines.append({"stdout": base64.b64encode(piece).decode()})
        if not first:
            lines.append({"exit_code": 0})
        for line in lines:
            data = json.dumps(line).encode() + b"\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()
        if first:
            self.connection.close()
            self.close_connection = True
        else:
            self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *args: object) -> None:
        pass


def main() -> None:
    where = Path(beyond_sandbox.__file__).resolve()
    check("site-packages" in where.parts or "dist-packages" in where.parts, f"imported {where}")
    lib = _kernel.library_path()
    check(lib.is_file(), f"the wheel carries {lib}")
    k = _kernel.load()
    check([k.silence_ms(b) for b in (0, 60)] == [30_000, 130_000], "the kernel's silence limit")
    check(k.classify("attach", 429).kind == "retry", "the kernel's table")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    got = Jobs(HttpTransport(url)).collect("smoke", "aa", mark=MARK)
    server.shutdown()
    check(got.output == OUTPUT and got.exit_code == 0, f"followed {got!r}")
    check([j["offset"] for j in Endpoint.attaches] == [0, 4], f"reattached at {Endpoint.attaches}")
    print(
        f"ok: {platform.system()} {platform.machine()}, Python {platform.python_version()}, {lib.name}"
    )


if __name__ == "__main__":
    main()
