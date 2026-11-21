"""The "other server" B for the URL-binding assertion: a bare HTTP listener that answers 401 to
everything and appends, per request, whether it carried an Authorization header (presence and
scheme only, never the value) to a JSON-lines file. Usage: lookalike.py PORT LOG"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, LOG = int(sys.argv[1]), sys.argv[2]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _any(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        auth = self.headers.get("authorization")
        with open(LOG, "a") as f:
            f.write(json.dumps({"method": self.command, "path": self.path,
                                "authorization_present": auth is not None,
                                "scheme": auth.split(" ")[0] if auth else None}) + "\n")
        self.send_response(401)
        self.send_header("www-authenticate", "Bearer")
        self.send_header("content-length", "0")
        self.end_headers()

    do_GET = do_POST = do_DELETE = _any


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
