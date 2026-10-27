"""The protocol's pure parts: frames, job messages, the HTTP framing. Against the shared vectors."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from beyond_sandbox import _wire
from beyond_sandbox._http import Malformed, ResponseReader, Truncated, build_request, parse_url
from beyond_sandbox.errors import InvalidRequest, ProtocolError, TooLarge, TransportError
from beyond_sandbox.models import (
    Exit,
    JobEnd,
    JobGap,
    JobOutput,
    JobShellExit,
    JobStarted,
    ShellExit,
    Signal,
    Stderr,
    Stdout,
    TimedOut,
)

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / "crates/sandbox/tests/fixtures/exec"


def fixture(name: str) -> dict[str, Any]:
    path = FIXTURES / name
    if not path.is_file():
        # Inside a checkout a missing vector is a broken tree, not a skip.
        if (REPO / "Cargo.toml").is_file():
            pytest.fail(f"{path} is missing: expected the shared exec vectors in {FIXTURES}")
        pytest.skip(f"{path} is not available outside the repository")
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def b64(s: str) -> bytes:
    return base64.b64decode(s)


# ---- stream frames: the reference vectors ------------------------------------------------------------


def run_frames(lines: list[str]) -> tuple[bytes, bytes, list[int], Any, str | None]:
    out, err, shell, end = bytearray(), bytearray(), [], None
    dec = _wire.frame_decoder()
    try:
        for line in lines:
            for f in dec.feed(line.encode() + b"\n"):
                if isinstance(f, Stdout):
                    out += f.data
                elif isinstance(f, Stderr):
                    err += f.data
                elif isinstance(f, ShellExit):
                    shell.append(f.code)
                else:
                    end = f
    except (ProtocolError, TransportError) as e:
        return bytes(out), bytes(err), shell, end, str(e)
    return bytes(out), bytes(err), shell, end, None


def test_the_v12_frame_vectors() -> None:
    for case in fixture("frames.json")["cases"]:
        name, want = case["name"], case["expect"]
        out, err, shell, end, error = run_frames(case["lines"])
        if want.get("invalid"):
            assert error is not None, name
            continue
        wanted_end = want["end"]
        if wanted_end and "error" in wanted_end:
            assert error and wanted_end["error"] in error, name
            continue
        assert error is None, (name, error)
        assert out == b64(want["stdout_base64"]), name
        assert err == b64(want["stderr_base64"]), name
        assert shell == want.get("shell_exits", []), name
        if wanted_end is None:
            assert end is None, name
        elif "exit_code" in wanted_end:
            assert end == Exit(wanted_end["exit_code"]), name
        elif "signal" in wanted_end:
            assert end == Signal(wanted_end["signal"]), name
        else:
            assert end == TimedOut(), name


def test_a_body_that_ends_without_a_final_frame_is_a_broken_stream() -> None:
    dec = _wire.frame_decoder()
    dec.feed(b'{"stdout":"YQ=="}\n')
    with pytest.raises(TransportError):
        dec.eof()


def test_frames_split_across_reads_decode_the_same() -> None:
    body = b'{"stdout":"aGVsbG8K"}\n{}\n{"stderr":"b29wcwo="}\n{"exit_code":3}\n'
    for cut in range(len(body) + 1):
        dec = _wire.frame_decoder()
        got = dec.feed(body[:cut]) + dec.feed(body[cut:])
        assert got == [Stdout(b"hello\n"), Stderr(b"oops\n"), Exit(3)], cut


def test_a_frame_naming_more_than_one_field_is_split_output_first() -> None:
    frames = _wire.decode_line(b'{"stdout":"YQ==","exit_code":1}')
    assert frames == [Stdout(b"a"), Exit(1)]


def test_an_overlong_line_is_refused() -> None:
    with pytest.raises(ProtocolError):
        _wire.LineSplitter().feed(b"x" * (_wire.MAX_FRAME_LINE + 1))


# ---- jobs: the reference vectors --------------------------------------------------------------


def test_the_v13_job_stream_vectors() -> None:
    for case in fixture("jobs.json")["streams"]:
        name, want = case["name"], case["expect"]
        started, out, gap, shell, end, error = None, bytearray(), 0, [], None, None
        dec = _wire.job_decoder()
        try:
            for line in case["lines"]:
                for f in dec.feed(line.encode() + b"\n"):
                    if isinstance(f, JobStarted):
                        started = f.pgid
                    elif isinstance(f, JobOutput):
                        out += f.data
                    elif isinstance(f, JobGap):
                        gap += f.bytes
                    elif isinstance(f, JobShellExit):
                        shell.append(f.code)
                    elif isinstance(f, JobEnd):
                        end = f.code
            if not dec.finished:
                dec.eof()
        except (ProtocolError, TransportError) as e:
            error = str(e)
        if want.get("error"):
            assert error and want["error"] in error, name
            continue
        if want["end"] is None:
            assert error is not None, name
            continue
        assert error is None, (name, error)
        assert (started, bytes(out), gap, shell, end) == (
            want["started"],
            b64(want["output_base64"]),
            want["gap"],
            want["shell_exits"],
            want["end"],
        ), name


def test_the_v13_answer_vectors() -> None:
    for case in fixture("jobs.json")["answers"]:
        want, answer = case["expect"], case["answer"]
        if case["op"] == "read":
            r = _wire.parse_read(answer)
            assert (r.state, r.code, r.offset, r.end, r.data) == (
                want["state"], want["code"], want["offset"], want["end"], b64(want["data_base64"]),
            )  # fmt: skip
        elif case["op"] == "list":
            got = _wire.parse_list(answer)
            assert [j.id for j in got] == [j["id"] for j in want["jobs"]]
            for g, w in zip(got, want["jobs"], strict=True):
                assert (g.state, g.code, g.kept, g.num, g.kind, g.seen, g.ended, g.label) == (
                    w["state"], w["code"], w["kept"], w["num"], w["kind"], w["seen"],
                    w["ended"], w["label"],
                )  # fmt: skip
        elif case["op"] == "kill":
            got_s = _wire.parse_survivors(answer)
            assert [(s.id, s.detail) for s in got_s] == [
                (s["id"], s["described"]) for s in want["survivors"]
            ]


# ---- the one-shot envelope, errors ---------------------------------------------------------------------


def test_envelope_streams_are_base64_bytes() -> None:
    body = json.dumps({"exit_code": 2, "stdout": "/wA=", "stderr": ""}).encode()
    r = _wire.parse_envelope("u", body, 100)
    assert (r.exit_code, r.stdout, r.stderr) == (2, b"\xff\x00", b"")


def test_the_cap_is_on_the_decoded_bytes_of_both_streams() -> None:
    body = json.dumps({"exit_code": 0, "stdout": "YWJj", "stderr": "ZGU="}).encode()  # 3 + 2
    assert _wire.parse_envelope("u", body, 5).stdout == b"abc"
    with pytest.raises(TooLarge):
        _wire.parse_envelope("u", body, 4)


@pytest.mark.parametrize(
    "body",
    [
        b"nope",
        b"[]",
        b'{"stdout":"","stderr":""}',
        b'{"exit_code":"0","stdout":"","stderr":""}',
        b'{"exit_code":0,"stdout":"","stderr":"!!"}',
        b'{"exit_code":0,"stdout":""}',
        b'{"exit_code":0,"stdout":"hi there","stderr":""}',
    ],
)
def test_a_200_that_is_not_the_envelope_is_a_protocol_error(body: bytes) -> None:
    with pytest.raises(ProtocolError):
        _wire.parse_envelope("u", body, 100)


def test_status_codes_map_to_kinds() -> None:
    assert _wire.status_error("u", 401, b"").kind == "auth"
    assert _wire.status_error("u", 403, b"").kind == "auth"
    assert _wire.status_error("u", 422, b"output exceeded the cap").kind == "too_large"
    assert _wire.status_error("u", 422, b"other").kind == "endpoint"
    assert _wire.status_error("u", 500, b"x").kind == "endpoint"
    assert _wire.status_error("u", 500, b"x").status == 500


def test_the_request_body_omits_what_is_absent() -> None:
    body = json.loads(_wire.request_body("ls", ["-l"], cwd=None, timeout_ms=5))
    assert body == {"command": "ls", "args": ["-l"], "timeout_ms": 5}
    body = json.loads(
        _wire.request_body(
            "cat", [], cwd="/w", timeout_ms=1, stdin=b"hi", stream=True, job={"op": "x"}
        )
    )
    assert body["stdin"] == "aGk=" and body["stream"] is True and body["cwd"] == "/w"
    assert body["job"] == {"op": "x"}


# ---- HTTP framing ----------------------------------------------------------------------------------

CHUNKED = (
    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: application/x-ndjson\r\n\r\n"
    b"5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\nTrailer: x\r\n\r\n"
)
LENGTH = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\nhello world"
CLOSE = b"HTTP/1.1 200 OK\r\n\r\nhello world"
INTERIM = b"HTTP/1.1 100 Continue\r\n\r\n" + LENGTH


@pytest.mark.parametrize("raw", [CHUNKED, LENGTH, CLOSE, INTERIM])
def test_a_response_parses_at_every_split_point(raw: bytes) -> None:
    for cut in range(len(raw) + 1):
        r = ResponseReader()
        r.feed(raw[:cut]) if cut else None
        r.feed(raw[cut:]) if cut < len(raw) else None
        r.feed(b"")
        assert r.done and r.status == 200 and r.take() == b"hello world", cut


def test_one_byte_at_a_time() -> None:
    r = ResponseReader()
    for i in range(len(CHUNKED)):
        r.feed(CHUNKED[i : i + 1])
    assert r.done and r.take() == b"hello world" and r.is_ndjson


def test_a_truncated_body_is_an_error_but_close_delimited_is_not() -> None:
    for raw in (LENGTH[:-1], CHUNKED[:-6]):
        r = ResponseReader()
        r.feed(raw)
        with pytest.raises(Truncated):
            r.feed(b"")


@pytest.mark.parametrize(
    "raw",
    [
        b"SMTP 250 ok\r\n\r\n",
        b"HTTP/1.1 abc\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nbad header\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Length: x\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nabc",
    ],
)
def test_malformed_responses(raw: bytes) -> None:
    r = ResponseReader()
    with pytest.raises(Malformed):
        r.feed(raw)


def test_urls() -> None:
    t = parse_url("https://example.com:8443/exec?x=1")
    assert (t.host, t.port, t.path, t.tls, t.host_header) == (
        "example.com",
        8443,
        "/exec?x=1",
        True,
        "example.com:8443",
    )
    t = parse_url("http://[::1]:80")
    assert (t.host, t.port, t.path, t.host_header) == ("::1", 80, "/", "[::1]")
    for bad in ("ftp://x/", "http://", "x", "http://h:99999/"):
        with pytest.raises(InvalidRequest):
            parse_url(bad)


def test_a_header_with_a_newline_is_refused() -> None:
    with pytest.raises(InvalidRequest):
        build_request(parse_url("http://h/"), [("Authorization", "Bearer a\r\nX: y")], b"")


def test_the_one_shot_answer_vectors() -> None:
    for case in fixture("output.json")["answers"]:
        name, want = case["name"], case["expect"]
        body = json.dumps(case["answer"]).encode()
        if want.get("invalid"):
            with pytest.raises(ProtocolError):
                _wire.parse_envelope("u", body, 1 << 20)
            continue
        r = _wire.parse_envelope("u", body, 1 << 20)
        assert r.exit_code == want["exit_code"], name
        assert r.stdout == b64(want["stdout_base64"]), name
        assert r.stderr == b64(want["stderr_base64"]), name
