"""A scripted Anthropic-SSE model: asks for one tool call, then, once it sees the tool result,
says "done". Records each tool result (text + is_error) as a JSON line. Usage: fake_model.py PORT TOOL RESULTS_FILE"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, TOOL, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def turn_tool():
    return sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": TOOL, "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
        {"type": "message_stop"}])


def turn_text(t):
    return sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 6}},
        {"type": "message_stop"}])


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        results = []
        for m in body.get("messages", []):
            c = m.get("content")
            for b in c if isinstance(c, list) else []:
                if b.get("type") == "tool_result":
                    t = b.get("content")
                    if isinstance(t, list):
                        t = "".join(x.get("text", "") for x in t)
                    results.append({"text": t, "is_error": bool(b.get("is_error"))})
        if results:
            with open(OUT, "a") as f:
                f.write(json.dumps(results[-1]) + "\n")
            out = turn_text("done")
        else:
            out = turn_tool()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
