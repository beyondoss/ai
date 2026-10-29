#!/usr/bin/env python3
"""Hot-path microbenchmarks for the beyond-sandbox client.

Measures the exploit targets from the performance audit:
  1. TLS context creation vs reuse (HTTPS connect path)
  2. Response body accumulation copies (read_capped)
  3. decode_base64 scrub path

Writes JSON to --out (default stdout) and a human summary to stderr.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beyond_sandbox import (  # noqa: E402
    HttpTransport,
    _fs,
    _http,
)


def timed(n: int, fn: Callable[[], None], warmup: int = 2) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "n": float(n),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[max(0, int(0.95 * (n - 1)))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "total_ms": sum(samples),
    }


# ---- TLS endpoint (mirrors tests/test_transport.py) ---------------------------------------------


def make_tls_endpoint(tmpdir: Path) -> tuple[ThreadingHTTPServer, str, Any, Path]:
    import ssl

    cert, key = tmpdir / "cert.pem", tmpdir / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            n = int(self.headers["Content-Length"])
            self.rfile.read(n)
            body = b'{"exit_code":0,"stdout":"","stderr":""}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert, key)
    server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
    url = f"https://127.0.0.1:{server.server_address[1]}/exec"
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    client_ctx = ssl.create_default_context(cafile=str(cert))
    return server, url, client_ctx, cert


def bench_ssl_context(n: int) -> dict[str, Any]:
    import ssl

    def create() -> None:
        ssl.create_default_context()

    ctx = ssl.create_default_context()

    def reuse() -> None:
        _ = ctx

    return {
        "create_default_context": timed(n, create),
        "reuse_cached": timed(n, reuse, warmup=0),
    }


def bench_https_run(n: int) -> dict[str, Any]:
    """End-to-end HTTPS run: passed ssl_context vs None (default-context path).

    For the None path we monkeypatch ``ssl.create_default_context`` to still pay the
    full system-trust-store load, then attach our test CA — matching production cost
    of the uncached path while allowing the self-signed bench server.
    """
    import ssl

    with tempfile.TemporaryDirectory() as td:
        server, url, client_ctx, cert = make_tls_endpoint(Path(td))
        try:
            t_passed = HttpTransport(url, ssl_context=client_ctx)

            def with_passed() -> None:
                t_passed.run("true", timeout_ms=5_000)

            real_cdc = ssl.create_default_context
            ca = str(cert)

            def trusting_cdc(*args: Any, **kwargs: Any) -> ssl.SSLContext:
                # Pay the real create_default_context cost, then trust the bench CA.
                ctx = real_cdc()
                ctx.load_verify_locations(cafile=ca)
                return ctx

            ssl.create_default_context = trusting_cdc  # type: ignore[assignment]
            try:
                t_default = HttpTransport(url)  # ssl_context=None

                def with_default() -> None:
                    t_default.run("true", timeout_ms=5_000)

                return {
                    "https_run_passed_ssl_context": timed(n, with_passed),
                    "https_run_ssl_context_none": timed(n, with_default),
                }
            finally:
                ssl.create_default_context = real_cdc  # type: ignore[assignment]
        finally:
            server.shutdown()


def bench_body_accumulate(size: int, chunk: int, n: int) -> dict[str, Any]:
    """Simulate ResponseReader.take() + read_capped accumulation for ``size`` bytes."""
    payload = os.urandom(size)
    head = (
        f"HTTP/1.1 200 OK\r\nContent-Length: {size}\r\nContent-Type: application/json\r\n\r\n"
    ).encode()

    def via_take() -> None:
        # Pre-fix path: intermediate bytes() per socket chunk.
        reader = _http.ResponseReader()
        reader.feed(head)
        off = 0
        body = bytearray()
        while not reader.done:
            piece = payload[off : off + chunk]
            off += len(piece)
            if piece:
                reader.feed(piece)
            body += reader.take()
        _ = bytes(body)

    def via_take_into() -> None:
        reader = _http.ResponseReader()
        reader.feed(head)
        off = 0
        body = bytearray()
        while not reader.done:
            piece = payload[off : off + chunk]
            off += len(piece)
            if piece:
                reader.feed(piece)
            reader.take_into(body)
        _ = bytes(body)

    return {
        "size_bytes": float(size),
        "chunk_bytes": float(chunk),
        "accumulate_via_take": timed(n, via_take),
        "accumulate_via_take_into": timed(n, via_take_into),
    }


def bench_decode_base64(raw_size: int, n: int) -> dict[str, Any]:
    raw = os.urandom(raw_size)
    # Realistic dd|base64 -w0 output, plus occasional whitespace like a non -w0 engine
    encoded = base64.b64encode(raw)
    messy = b"\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76)) + b"\n"

    def current() -> None:
        _fs.decode_base64(messy)

    # Legacy path kept for A/B: regex scrub (what decode_base64 did before the exploit).
    def legacy_re_sub() -> None:
        import re

        cleaned = re.sub(rb"[\s=]", b"", messy)
        if len(cleaned) % 4 == 1:
            raise ValueError("truncated base64")
        base64.b64decode(cleaned + b"=" * (-len(cleaned) % 4), validate=True)

    return {
        "raw_bytes": float(raw_size),
        "wire_bytes": float(len(messy)),
        "decode_current": timed(n, current),
        "decode_legacy_re_sub": timed(n, legacy_re_sub),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="Write JSON results here")
    ap.add_argument("--label", default="run", help="Label stored in the JSON")
    ap.add_argument("--https-n", type=int, default=40)
    ap.add_argument("--ssl-n", type=int, default=50)
    ap.add_argument("--body-n", type=int, default=30)
    ap.add_argument("--b64-n", type=int, default=40)
    args = ap.parse_args()

    results: dict[str, Any] = {
        "label": args.label,
        "python": sys.version.split()[0],
        "ssl_context": bench_ssl_context(args.ssl_n),
        "https": bench_https_run(args.https_n),
        "body_4mib": bench_body_accumulate(4 * 1024 * 1024, 65536, args.body_n),
        "body_16mib": bench_body_accumulate(16 * 1024 * 1024, 65536, max(8, args.body_n // 3)),
        "b64_96kib": bench_decode_base64(96 * 1024, args.b64_n),
        "b64_1mib": bench_decode_base64(1024 * 1024, max(10, args.b64_n // 2)),
    }

    # Derived deltas (for the human summary)
    ssl_create = results["ssl_context"]["create_default_context"]["mean_ms"]
    https_none = results["https"]["https_run_ssl_context_none"]["mean_ms"]
    https_pass = results["https"]["https_run_passed_ssl_context"]["mean_ms"]
    body_old = results["body_16mib"]["accumulate_via_take"]["mean_ms"]
    body_new = results["body_16mib"]["accumulate_via_take_into"]["mean_ms"]
    b64_now = results["b64_1mib"]["decode_current"]["mean_ms"]
    b64_legacy = results["b64_1mib"]["decode_legacy_re_sub"]["mean_ms"]

    summary = {
        "ssl_create_default_ms": ssl_create,
        "https_ssl_context_none_ms": https_none,
        "https_passed_ssl_context_ms": https_pass,
        "https_none_minus_passed_ms": https_none - https_pass,
        "body_16mib_take_ms": body_old,
        "body_16mib_take_into_ms": body_new,
        "body_16mib_delta_pct": (body_old - body_new) / body_old * 100 if body_old else 0.0,
        "b64_1mib_current_ms": b64_now,
        "b64_1mib_legacy_re_sub_ms": b64_legacy,
        "b64_1mib_vs_legacy_pct": (b64_legacy - b64_now) / b64_legacy * 100 if b64_legacy else 0.0,
        "has_take_into": hasattr(_http.ResponseReader, "take_into"),
    }
    results["summary"] = summary

    text = json.dumps(results, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)

    print(
        f"[{args.label}] SSL create_default_context mean={ssl_create:.2f}ms\n"
        f"[{args.label}] HTTPS run ssl_context=None mean={https_none:.2f}ms "
        f"vs passed={https_pass:.2f}ms (delta {https_none - https_pass:+.2f}ms)\n"
        f"[{args.label}] body 16MiB take={body_old:.2f}ms take_into={body_new:.2f}ms "
        f"({summary['body_16mib_delta_pct']:+.1f}%) has_take_into={summary['has_take_into']}\n"
        f"[{args.label}] b64 1MiB current={b64_now:.2f}ms legacy_re_sub={b64_legacy:.2f}ms "
        f"(current faster by {summary['b64_1mib_vs_legacy_pct']:+.1f}%)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
