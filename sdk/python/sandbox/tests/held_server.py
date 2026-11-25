"""A scripted job endpoint that echoes marks as ``crates/execd`` does: ``started{mark}`` and
``attached{mark, from}``. It cuts streams, refuses attaches and answers for another incarnation on demand,
so a handle's behaviour across failures is observable. Port of the TypeScript SDK's ``held-server.ts``."""

from __future__ import annotations

import base64
import contextlib
import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MARK = "00ff00ff00ff00ff1a"


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


@dataclass
class Script:
    #: The job's whole output; the exit follows it.
    output: str
    #: Statuses (and headers) to answer the next attaches with, before any stream.
    refuse: list[tuple[int, dict[str, str]]] = field(default_factory=list)
    #: The first streams send this many bytes of output and then drop the connection.
    cut_after: list[int] = field(default_factory=list)
    #: Echo this mark instead of the held one (``None``: none).
    echo: str | None = MARK
    #: Echo this mark on a reattach only: another incarnation answers it.
    reattach_echo: str | None = None
    #: Bytes of the front of the output the server no longer keeps.
    base: int = 0
    #: The shell's exit, sent before the final frame.
    shell_exit: int | None = None
    #: A start's answer, when it is a refusal.
    start_status: int | None = None
    #: The seconds after the mark echo a stream says nothing, before its output (a silent endpoint).
    hang_s: float = 0.0


class _Httpd(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that left, or a connection this script cut, is not a failure to print."""


class Server:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.attaches: list[dict[str, Any]] = []
        self.starts: list[dict[str, Any]] = []
        self.kills: list[dict[str, Any]] = []
        self.beats: list[str | None] = []
        self._cuts = list(script.cut_after)
        self._refuse = list(script.refuse)
        server = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.beats.append(self.headers.get("Keepalive-Ping-Interval"))
                server.handle(self, body)

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = _Httpd(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/exec"
        threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- the endpoint ---------------------------------------------------------------------------

    def handle(self, h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        job = body["job"]
        if job["op"] == "start":
            self.starts.append(job)
            if self.script.start_status is not None:
                return self._json(h, self.script.start_status, {"error": "refused"})
            if not job.get("hold"):
                echo = {
                    "started": 42,
                    **({} if self.script.echo is None else {"mark": self.script.echo}),
                    **({} if "idempotency_key" not in job else {"key": job["idempotency_key"]}),
                }
                return self._json(h, 200, echo)
            return self._stream(h, 0, True, job.get("idempotency_key"))
        if job["op"] == "kill":
            self.kills.append(job)
            return self._json(h, 200, {"survivors": []})
        self.attaches.append(job)
        if self._refuse:
            status, headers = self._refuse.pop(0)
            return self._json(h, status, {"error": "refused"}, headers)
        if job.get("mark") not in (None, MARK):
            return self._json(h, 404, {"error": "no such job"})
        self._stream(h, job["offset"], False)

    def _json(
        self,
        h: BaseHTTPRequestHandler,
        status: int,
        obj: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(obj).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(data)

    def _write(self, h: BaseHTTPRequestHandler, line: dict[str, Any] | None) -> None:
        data = b"\n" if line is None else json.dumps(line).encode() + b"\n"
        h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        h.wfile.flush()

    def _stream(
        self, h: BaseHTTPRequestHandler, offset: int, start: bool, key: str | None = None
    ) -> None:
        sc = self.script
        h.send_response(200)
        h.send_header("Content-Type", "application/x-ndjson")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        at = max(offset, sc.base)
        mark = sc.reattach_echo if not start and sc.reattach_echo is not None else sc.echo
        if start:
            self._write(
                h,
                {
                    "started": 42,
                    **({} if mark is None else {"mark": mark}),
                    **({} if key is None else {"key": key}),
                },
            )
        elif mark is not None:
            self._write(h, {"attached": {"mark": mark, "from": at}})
        if sc.hang_s:
            threading.Event().wait(sc.hang_s)
        cut = self._cuts.pop(0) if self._cuts else None
        body = sc.output[at:]
        piece = body if cut is None else body[:cut]
        if cut is not None and cut < len(body):
            self._write(h, {"stdout": b64(piece)} if piece else None)
            self._drop(h)
            return
        if piece:
            self._write(h, {"stdout": b64(piece)})
        if sc.shell_exit is not None:
            self._write(h, {"shell_exit": sc.shell_exit})
        self._write(h, {"exit_code": 0})
        h.wfile.write(b"0\r\n\r\n")

    def _drop(self, h: BaseHTTPRequestHandler) -> None:
        """Drop the connection mid-stream: no terminating chunk, the peer sees a cut."""
        h.close_connection = True
        with contextlib.suppress(OSError):
            h.connection.shutdown(socket.SHUT_RDWR)


def serve(script: Script) -> Iterator[Server]:
    s = Server(script)
    try:
        yield s
    finally:
        s.close()
