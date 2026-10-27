"""Raw exec returns bytes (exec v1.1 ``output_base64``): the shared vectors, and the transports."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from beyond_sandbox import (
    AsyncHttpTransport,
    EndpointError,
    ExecResult,
    HttpTransport,
    ProtocolError,
    SandboxError,
    TooLarge,
    _wire,
)
from beyond_sandbox._wire import OneShot, decode_output, probe_honoured
from beyond_sandbox.models import ShellExit

FIXTURE = (
    Path(__file__).resolve().parents[4] / "crates/sandbox/tests/fixtures/exec/v1.1-output.json"
)
# The vectors are the repository's: a missing file fails the run, it does not skip it.
VECTORS: dict[str, Any] = json.loads(FIXTURE.read_text())

ALL = bytes(range(256))


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def check(name: str, got: Callable[[], ExecResult], want: dict[str, Any]) -> None:
    if want.get("invalid"):
        with pytest.raises((SandboxError, AssertionError)):
            got()
        return
    r = got()
    assert r.exit_code == want["exit_code"], name
    assert b64(r.stdout) == want["stdout_base64"], name
    assert b64(r.stderr) == want["stderr_base64"], name
    assert r.exact == want["exact"], name


# ---- the vectors ----------------------------------------------------------------------------------


def test_the_probe_is_the_bytes_the_vectors_name() -> None:
    assert VECTORS["probe"]["script"] == _wire.OUTPUT_PROBE_SCRIPT
    assert b64(_wire.OUTPUT_PROBE_BYTES) == VECTORS["probe"]["bytes_base64"]


@pytest.mark.parametrize("case", VECTORS.get("probes", []), ids=lambda c: c["name"])
def test_probe_vectors(case: dict[str, Any]) -> None:
    if case.get("error"):
        with pytest.raises(ProtocolError):
            decode_output("u", json.dumps(case["answer"]).encode(), 1 << 20)
    else:
        assert probe_honoured(case["answer"]) is case["supported"]


@pytest.mark.parametrize("case", VECTORS.get("answers", []), ids=lambda c: c["name"])
def test_answer_vectors(case: dict[str, Any]) -> None:
    check(
        case["name"],
        lambda: decode_output("u", json.dumps(case["answer"]).encode(), 1 << 20),
        case["expect"],
    )


@pytest.mark.parametrize("case", VECTORS.get("streams", []), ids=lambda c: c["name"])
def test_stream_vectors(case: dict[str, Any]) -> None:
    def run() -> ExecResult:
        fold = OneShot()
        for line in case["lines"]:
            for frame in _wire.decode_line(line.encode()):
                done = fold.feed(frame)
                if done is not None:
                    return done
        raise AssertionError("the body ended without a final frame")

    check(case["name"], run, case["expect"])


def test_the_cap_is_on_the_decoded_bytes_of_both_streams_together() -> None:
    body = json.dumps({"exit_code": 0, "stdout_base64": "AAAA", "stderr_base64": "AAAA"}).encode()
    assert decode_output("u", body, 6).exact
    with pytest.raises(TooLarge, match="the command ran"):
        decode_output("u", body, 5)


def test_the_body_bound_holds_the_base64_of_a_caps_worth_of_output() -> None:
    assert _wire.body_cap(16 * 1024 * 1024) >= 22_369_620
    assert _wire.body_cap(3) == 4 + 4096


def test_a_shell_exit_ends_a_one_shot_call() -> None:
    assert OneShot().feed(ShellExit(2)) == ExecResult(2, b"", b"", exact=True)


# ---- the transports against a scripted endpoint ---------------------------------------------------


class Endpoint:
    def __init__(self, handler: Callable[[BaseHTTPRequestHandler, dict[str, Any]], None]) -> None:
        self.requests: list[dict[str, Any]] = []
        endpoint = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(body)
                handler(self, body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/exec"
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()


def reply(h: BaseHTTPRequestHandler, status: int, obj: Any) -> None:
    data = json.dumps(obj).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def script(body: dict[str, Any]) -> str:
    return str((body["args"] + ["", ""])[1])


def is_probe(body: dict[str, Any]) -> bool:
    return "377" in script(body)


def honours(payload: bytes) -> Callable[[BaseHTTPRequestHandler, dict[str, Any]], None]:
    def handle(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        if body.get("output_base64") is not True:
            return reply(h, 200, {"exit_code": 0, "stdout": "text", "stderr": ""})
        out = _wire.OUTPUT_PROBE_BYTES if is_probe(body) else payload
        reply(h, 200, {"exit_code": 0, "stdout_base64": b64(out), "stderr_base64": ""})

    return handle


@pytest.fixture
def serve() -> Iterator[Callable[[Callable[..., None]], Endpoint]]:
    started: list[Endpoint] = []

    def start(handler: Callable[..., None]) -> Endpoint:
        e = Endpoint(handler)
        started.append(e)
        return e

    yield start
    for e in started:
        e.server.shutdown()


def test_run_sends_output_base64_and_returns_every_byte_value_exactly(
    serve: Callable[..., Endpoint],
) -> None:
    e = serve(honours(ALL))
    t = HttpTransport(e.url)
    r = t.run("gen")
    assert (r.stdout, r.exact) == (ALL, True)
    t.run("gen")
    # The probe once per connection, then each call.
    assert [q.get("output_base64") for q in e.requests] == [True, True, True]
    assert e.requests[0]["args"] == ["-c", _wire.OUTPUT_PROBE_SCRIPT]


def test_run_text_skips_the_probe_and_answers_the_plain_way(serve: Callable[..., Endpoint]) -> None:
    e = serve(lambda h, b: reply(h, 200, {"exit_code": 0, "stdout": "plain", "stderr": ""}))
    r = HttpTransport(e.url).run_text("x")
    assert (r.stdout, r.exact) == (b"plain", False)
    assert len(e.requests) == 1 and "output_base64" not in e.requests[0]


def test_a_plain_v11_endpoint_answers_text_and_the_result_says_it_is_not_exact(
    serve: Callable[..., Endpoint],
) -> None:
    e = serve(lambda h, b: reply(h, 200, {"exit_code": 4, "stdout": "café �", "stderr": "e"}))
    r = HttpTransport(e.url).run("x")
    assert (r.exit_code, r.exact, r.stderr) == (4, False, b"e")
    assert r.stdout == "café �".encode()


def test_a_probe_whose_request_failed_is_an_error_never_not_supported(
    serve: Callable[..., Endpoint],
) -> None:
    e = serve(lambda h, b: reply(h, 500, {}))
    t = HttpTransport(e.url)
    with pytest.raises(EndpointError):
        t.run("x")
    assert len(e.requests) == 1  # the call did not run as text instead
    with pytest.raises(EndpointError):
        t.run("x")
    assert len(e.requests) == 2  # and the probe was asked again


def test_a_strict_front_answering_400_has_answered_not_supported(
    serve: Callable[..., Endpoint],
) -> None:
    def handle(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        if body.get("output_base64") or body.get("stream"):
            return reply(h, 400, {"error": "unknown field"})
        reply(h, 200, {"exit_code": 0, "stdout": "hi", "stderr": ""})

    r = HttpTransport(serve(handle).url).run("x")
    assert (r.stdout, r.exact) == (b"hi", False)


def ndjson(h: BaseHTTPRequestHandler, frames: list[dict[str, Any]], pause: float = 0.0) -> None:
    h.send_response(200)
    h.send_header("Content-Type", "application/x-ndjson")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()
    for i, f in enumerate(frames):
        data = json.dumps(f).encode() + b"\n"
        h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        h.wfile.flush()
        if pause and i == 0:
            time.sleep(pause)
    h.wfile.write(b"0\r\n\r\n")


def streams_only(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
    if body.get("output_base64") or body.get("job"):
        return reply(h, 200, {"exit_code": 0, "stdout": "", "stderr": ""})
    if script(body).startswith("printf a;"):
        return ndjson(h, [{"stdout": b64(b"a")}, {"stdout": b64(b"b")}, {"exit_code": 0}], 0.25)
    ndjson(h, [{"stdout": b64(ALL)}, {"stderr": b64(b"\xff\x00")}, {"signal": 9}])


def test_an_endpoint_that_only_streams_is_exact_over_the_stream(
    serve: Callable[..., Endpoint],
) -> None:
    r = HttpTransport(serve(streams_only).url).run("gen")
    assert (r.stdout, r.stderr, r.exit_code, r.exact) == (ALL, b"\xff\x00", 137, True)


def test_a_call_with_stdin_without_output_base64_is_text_not_exact(
    serve: Callable[..., Endpoint],
) -> None:
    e = serve(lambda h, b: reply(h, 200, {"exit_code": 0, "stdout": "echo", "stderr": ""}))
    assert not HttpTransport(e.url).run("cat", stdin=b"\xff").exact


def test_output_past_the_cap_is_too_large_and_says_the_command_ran(
    serve: Callable[..., Endpoint],
) -> None:
    def handle(h: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        if body.get("output_base64") is not True:
            return reply(h, 200, {"exit_code": 0, "stdout": "", "stderr": ""})
        probe = is_probe(body)
        reply(
            h,
            200,
            {
                "exit_code": 0,
                "stdout_base64": b64(_wire.OUTPUT_PROBE_BYTES if probe else bytes(600)),
                "stderr_base64": "" if probe else b64(bytes(600)),
            },
        )

    with pytest.raises(TooLarge, match="the command ran"):
        HttpTransport(serve(handle).url, max_response_bytes=1000).run("x")


def test_the_async_transport_returns_the_same_bytes(serve: Callable[..., Endpoint]) -> None:
    async def main() -> None:
        r = await AsyncHttpTransport(serve(honours(ALL)).url).run("gen")
        assert (r.stdout, r.exact) == (ALL, True)
        r = await AsyncHttpTransport(serve(streams_only).url).run("gen")
        assert (r.stdout, r.exit_code, r.exact) == (ALL, 137, True)
        t = await AsyncHttpTransport(
            serve(lambda h, b: reply(h, 200, {"exit_code": 0, "stdout": "t", "stderr": ""})).url
        ).run("x")
        assert (t.stdout, t.exact) == (b"t", False)

    asyncio.run(main())
