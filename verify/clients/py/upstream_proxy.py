"""A provider that dies mid-session: a tiny HTTP -> HTTPS reverse proxy in front of one real provider.

Usage: upstream_proxy.py <upstream host> [<serve n>]

The gateway under test (with `upstream_tls = false`) reaches the provider through this proxy, which
re-originates each request over TLS to the real host with the real `Host` header and streams the
response back as it arrives. After it has finished `serve n` responses it exits: the listener and
every pooled connection close together, so the gateway's next attempt at this provider is refused
and must fail over, without the gateway restarting. With no `serve n` it serves forever.

Prints `READY <port>` once listening on an OS-chosen loopback port and one line per request to stderr (`#k METHOD path -> status`), the
evidence of which turns reached this provider.
"""
import http.client
import http.server
import os
import ssl
import sys
import threading

HOST = sys.argv[1]
SERVE = int(sys.argv[2]) if len(sys.argv) > 2 else None
CTX = ssl.create_default_context()
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade", "host",
       "content-length"}
lock = threading.Lock()
served = 0


def read_chunked(f):
    out = bytearray()
    while True:
        size = int(f.readline().split(b";")[0].strip() or b"0", 16)
        if size == 0:
            while f.readline() not in (b"\r\n", b"\n", b""):
                pass
            return bytes(out)
        out += f.read(size)
        f.readline()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def relay(self):
        global served
        if "chunked" in self.headers.get("transfer-encoding", "").lower():
            body = read_chunked(self.rfile)
        else:
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Host"] = HOST
        up = http.client.HTTPSConnection(HOST, 443, context=CTX, timeout=600)
        for attempt in range(3):
            # A failed TCP/TLS handshake sent nothing upstream, so it is safe to retry: the proxy
            # stands in for the network path, and must not be the flaky part of a trial.
            try:
                up.connect()
                break
            except OSError as e:
                print(f"connect {HOST} failed ({e}); retry {attempt + 1}", file=sys.stderr, flush=True)
                up.close()
                if attempt == 2:
                    raise
        up.request(self.command, self.path, body=body or None, headers=headers)
        r = up.getresponse()
        with lock:
            served += 1
            n = served
        last = SERVE is not None and n >= SERVE
        self.send_response_only(r.status, r.reason)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        if last:
            # The gateway must not pool the connection this provider is about to drop.
            self.send_header("Connection", "close")
        self.end_headers()
        while chunk := r.read1(65536):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        up.close()
        print(f"#{n} {self.command} {self.path} -> {r.status}", file=sys.stderr, flush=True)
        if last:
            # The provider dies: no listener, no pooled connection survives.
            print(f"served {n}; dying", file=sys.stderr, flush=True)
            os._exit(0)

    do_POST = do_GET = do_PUT = do_DELETE = relay


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
server.daemon_threads = True
print(f"READY {server.server_address[1]}", flush=True)
server.serve_forever()
