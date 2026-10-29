"""The HTTP transports against a scripted in-process endpoint: auth, caps, framing, closing."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import ssl
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from beyond_sandbox import (
    AsyncHttpTransport,
    AuthError,
    EndpointError,
    Exit,
    HttpTransport,
    ProtocolError,
    Stdout,
    TooLargeError,
    TransportError,
)
from beyond_sandbox.errors import TimeoutError

Handler = Callable[[BaseHTTPRequestHandler, dict[str, Any]], None]


class Endpoint:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.auth: list[str | None] = []
        self.handler: Handler = lambda h, body: reply(
            h, 200, {"exit_code": 0, "stdout": "", "stderr": ""}
        )
        self.disconnected = threading.Event()
        endpoint = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(body)
                endpoint.auth.append(self.headers.get("Authorization"))
                endpoint.handler(self, body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/exec"
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()


def reply(
    h: BaseHTTPRequestHandler, status: int, obj: Any, ctype: str = "application/json"
) -> None:
    data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    h.send_response(status)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def chunk(h: BaseHTTPRequestHandler, line: dict[str, Any]) -> None:
    data = json.dumps(line).encode() + b"\n"
    h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
    h.wfile.flush()


def ndjson_head(h: BaseHTTPRequestHandler) -> None:
    h.send_response(200)
    h.send_header("Content-Type", "application/x-ndjson")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


@pytest.fixture
def ep() -> Iterator[Endpoint]:
    e = Endpoint()
    yield e
    e.server.shutdown()


# ---- run -------------------------------------------------------------------------------------------


def test_run_sends_the_request_and_the_bearer(ep: Endpoint) -> None:
    ep.handler = lambda h, b: reply(
        h, 200, {"exit_code": 3, "stdout": b64(b"o"), "stderr": b64(b"e")}
    )
    t = HttpTransport(ep.url, "tok", headers={"X-Tenant": "acme"})
    r = t.run("sh", ["-c", "x"], cwd="/w", timeout_ms=5000, stdin=b"in")
    assert (r.exit_code, r.stdout, r.stderr) == (3, b"o", b"e")
    assert ep.requests[0] == {
        "command": "sh", "args": ["-c", "x"], "cwd": "/w", "timeout_ms": 5000,
        "stdin": b64(b"in"),
    }  # fmt: skip
    assert ep.auth == ["Bearer tok"]


def test_no_token_sends_no_authorization(ep: Endpoint) -> None:
    HttpTransport(ep.url).run("true")
    assert ep.auth == [None]


def test_a_401_refreshes_the_token_once_and_retries_that_request_once(ep: Endpoint) -> None:
    def handler(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        if h.headers.get("Authorization") == "Bearer fresh":
            reply(h, 200, {"exit_code": 0, "stdout": b64(b"ok"), "stderr": ""})
        else:
            reply(h, 401, {"error": "expired"})

    ep.handler = handler
    tokens = iter(["stale", "fresh", "never"])
    t = HttpTransport(ep.url, lambda: next(tokens))
    assert t.run("true").stdout == b"ok"
    assert ep.auth == ["Bearer stale", "Bearer fresh"]
    assert t.run("true").stdout == b"ok"  # the fresh token is kept, not fetched again
    assert ep.auth[2:] == ["Bearer fresh"]


def test_a_second_401_is_an_auth_error_and_not_retried_again(ep: Endpoint) -> None:
    ep.handler = lambda h, b: reply(h, 401, {"error": "no"})
    calls: list[int] = []

    def token() -> str:
        calls.append(1)
        return "t"

    t = HttpTransport(ep.url, token)
    with pytest.raises(AuthError):
        t.run("true")
    assert len(ep.requests) == 2 and len(calls) == 2


def test_a_token_source_that_fails_is_an_auth_error_and_sends_nothing(ep: Endpoint) -> None:
    def boom() -> str:
        raise RuntimeError("vault sealed")

    for source in (boom, lambda: "  "):
        with pytest.raises(AuthError):
            HttpTransport(ep.url, source).run("true")
    with pytest.raises(AuthError, match="token source failed: vault sealed"):
        HttpTransport(ep.url, boom).run("true")

    async def aboom() -> str:
        raise RuntimeError("vault sealed")

    async def go() -> None:
        with pytest.raises(AuthError, match="token source failed: vault sealed"):
            await AsyncHttpTransport(ep.url, aboom).run("true")

    asyncio.run(go())
    assert ep.requests == []


def test_a_fixed_token_is_not_retried_on_401_and_403_is_never_retried(ep: Endpoint) -> None:
    ep.handler = lambda h, b: reply(h, 401, {})
    with pytest.raises(AuthError):
        HttpTransport(ep.url, "fixed").run("true")
    assert len(ep.requests) == 1
    ep.requests.clear()
    ep.handler = lambda h, b: reply(h, 403, {})
    with pytest.raises(AuthError):
        HttpTransport(ep.url, lambda: "x").run("true")
    assert len(ep.requests) == 1


def test_status_errors(ep: Endpoint) -> None:
    ep.handler = lambda h, b: reply(h, 500, b"boom")
    with pytest.raises(EndpointError) as e:
        HttpTransport(ep.url).run("true")
    assert e.value.status == 500 and "boom" in e.value.detail
    ep.handler = lambda h, b: reply(h, 422, {"error": "output exceeded the cap"})
    with pytest.raises(TooLargeError):
        HttpTransport(ep.url).run("true")


def test_a_200_that_is_not_json_is_a_protocol_error(ep: Endpoint) -> None:
    ep.handler = lambda h, b: reply(h, 200, b"<html>")
    with pytest.raises(ProtocolError):
        HttpTransport(ep.url).run("true")


def test_the_response_cap_stops_reading_and_says_the_command_ran(ep: Endpoint) -> None:
    ep.handler = lambda h, b: reply(
        h, 200, {"exit_code": 0, "stdout": b64(b"x" * 20000), "stderr": ""}
    )
    with pytest.raises(TooLargeError) as e:
        HttpTransport(ep.url, max_response_bytes=1000).run("true")
    assert "the command ran" in e.value.detail


def test_the_cap_applies_to_a_chunked_body_with_no_length(ep: Endpoint) -> None:
    def handler(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        h.send_response(200)
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        data = b'{"exit_code":0,"stdout":"' + b"eA==" * 5000 + b'","stderr":""}'
        h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")

    ep.handler = handler
    with pytest.raises(TooLargeError):
        HttpTransport(ep.url, max_response_bytes=1000).run("true")


def test_unreachable_is_a_transport_error() -> None:
    with pytest.raises(TransportError):
        HttpTransport("http://127.0.0.1:1/exec").run("true")


def test_the_request_deadline_is_timeout_plus_grace(
    ep: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    from beyond_sandbox import _wire

    monkeypatch.setattr(_wire, "GRACE_S", 0.3)
    ep.handler = lambda h, b: time.sleep(2)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        HttpTransport(ep.url).run("true", timeout_ms=100)
    assert 0.3 <= time.monotonic() - t0 < 1.5


# ---- stream ----------------------------------------------------------------------------------------


def streaming_endpoint(ep: Endpoint, frames: list[dict[str, Any]], final: bool = True) -> None:
    def handler(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        ndjson_head(h)
        for f in frames:
            chunk(h, f)
        if final:
            h.wfile.write(b"0\r\n\r\n")
        else:
            try:
                h.rfile.read(1)  # hold the stream open until the client goes away
            finally:
                ep.disconnected.set()

    ep.handler = handler


def test_stream_yields_frames_and_stops_after_the_final_one(ep: Endpoint) -> None:
    streaming_endpoint(ep, [{"stdout": b64(b"hi")}, {}, {"exit_code": 4}])
    t = HttpTransport(ep.url)
    assert list(t.stream("echo")) == [Stdout(b"hi"), Exit(4)]
    assert [r.get("stream") for r in ep.requests] == [True]  # no probe first
    assert ep.requests[0]["timeout_ms"] == 2**31 - 1


def test_a_stream_that_ends_without_a_final_frame_is_a_transport_error(ep: Endpoint) -> None:
    streaming_endpoint(ep, [{"stdout": b64(b"hi")}], final=True)
    with pytest.raises(TransportError):
        list(HttpTransport(ep.url).stream("echo"))


def test_a_stream_request_answered_with_a_one_shot_body_is_a_protocol_error(ep: Endpoint) -> None:
    # The default handler answers 200 with a result object and no ndjson Content-Type.
    with pytest.raises(ProtocolError, match="not a stream"):
        HttpTransport(ep.url).stream("echo")

    async def go() -> None:
        with pytest.raises(ProtocolError, match="not a stream"):
            await AsyncHttpTransport(ep.url).stream("echo")

    asyncio.run(go())


def test_an_error_frame_is_a_transport_error(ep: Endpoint) -> None:
    streaming_endpoint(ep, [{"error": "sandbox lost"}])
    with pytest.raises(TransportError, match="sandbox lost"):
        list(HttpTransport(ep.url).stream("echo"))


def test_closing_the_stream_closes_the_connection(ep: Endpoint) -> None:
    streaming_endpoint(ep, [{"stdout": b64(b"x")}], final=False)
    s = HttpTransport(ep.url).stream("sleep")
    assert next(s) == Stdout(b"x")
    s.close()
    assert ep.disconnected.wait(5), "the endpoint never saw the connection close"


# ---- async -----------------------------------------------------------------------------------------


def test_async_run_refresh_and_errors(ep: Endpoint) -> None:
    def handler(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        if h.headers.get("Authorization") == "Bearer fresh":
            reply(h, 200, {"exit_code": 0, "stdout": b64(b"ok"), "stderr": ""})
        else:
            reply(h, 401, {})

    ep.handler = handler

    async def main() -> None:
        tokens = iter(["stale", "fresh"])

        async def token() -> str:
            return next(tokens)

        t = AsyncHttpTransport(ep.url, token)
        assert (await t.run("true")).stdout == b"ok"
        assert ep.auth == ["Bearer stale", "Bearer fresh"]
        ep.handler = lambda h, b: reply(
            h, 200, {"exit_code": 0, "stdout": b64(b"x" * 20000), "stderr": ""}
        )
        with pytest.raises(TooLargeError):
            await AsyncHttpTransport(ep.url, max_response_bytes=1000).run("true")
        with pytest.raises(TransportError):
            await AsyncHttpTransport("http://127.0.0.1:1/").run("true")

    asyncio.run(main())


def test_async_stream_and_close(ep: Endpoint) -> None:
    streaming_endpoint(ep, [{"stdout": b64(b"hi")}, {"exit_code": 0}])

    async def main() -> None:
        t = AsyncHttpTransport(ep.url)
        assert [f async for f in await t.stream("echo")] == [Stdout(b"hi"), Exit(0)]
        streaming_endpoint(ep, [{"stdout": b64(b"x")}], final=False)
        s = await t.stream("sleep")
        assert await s.__anext__() == Stdout(b"x")
        await s.aclose()

    asyncio.run(main())
    assert ep.disconnected.wait(5)


# ---- TLS -------------------------------------------------------------------------------------------


@pytest.fixture
def tls_endpoint(tmp_path: Path) -> Iterator[tuple[Endpoint, ssl.SSLContext]]:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is needed to make a test certificate")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
         "-out", str(cert), "-days", "1", "-subj", "/CN=localhost",
         "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"],
        check=True, capture_output=True,
    )  # fmt: skip
    e = Endpoint()
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert, key)
    e.server.socket = server_ctx.wrap_socket(e.server.socket, server_side=True)
    e.url = e.url.replace("http://", "https://")
    client_ctx = ssl.create_default_context(cafile=str(cert))
    yield e, client_ctx
    e.server.shutdown()


def test_https_sync_and_async(tls_endpoint: tuple[Endpoint, ssl.SSLContext]) -> None:
    ep, ctx = tls_endpoint
    ep.handler = lambda h, b: reply(
        h, 200, {"exit_code": 0, "stdout": b64(b"secure"), "stderr": ""}
    )
    assert HttpTransport(ep.url, ssl_context=ctx).run("true").stdout == b"secure"

    async def main() -> None:
        assert (await AsyncHttpTransport(ep.url, ssl_context=ctx).run("true")).stdout == b"secure"

    asyncio.run(main())
    # the default context does not trust a self-signed certificate
    with pytest.raises(TransportError):
        HttpTransport(ep.url).run("true")
