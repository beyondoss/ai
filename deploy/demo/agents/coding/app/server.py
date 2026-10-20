"""A tiny web app: one page that greets someone. Standard library only; serves on port 8000."""

import os
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def greeting(name):
    return f"Hello, {name.title}!"


def build():
    try:
        with open(os.path.join(HERE, "BUILD")) as f:
            return f.read().strip()
    except OSError:
        return "dev"


def page(name="world"):
    return (
        "<!doctype html><html><head><title>Greeter</title></head><body>"
        f"<h1>{greeting(name)}</h1><p>build {build()}</p></body></html>"
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
