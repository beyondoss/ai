"""FLT-1 fault probe: one call through the gateway with a stock SDK at its DEFAULT retry policy.

Usage: fault_probe.py <openai|anthropic> <stream|nonstream>, env VERIFY_BASE, VERIFY_KEY,
VERIFY_MODEL. Prints one final line `VERIFY {json}`:
  ok        the call returned (non-stream) or the stream was read to its end without an error
  terminal  the stream carried its terminal event (finish_reason / message_stop); a non-stream
            success is terminal by definition
  usage     what the client was shown: {input_total, output, cache_read}, input_total including cache
  error     {kind, status, message, body, request_id, retry_after} for the error the SDK raised
  attempts  one per HTTP attempt the SDK made: {t (seconds since start), status, request_id,
            retry_after}; status is null for an attempt that got no response head
  elapsed   seconds for the whole call, retries included
The Rust trial (crates/verify/tests/fault_live.rs) holds this against the fault proxy's records
and the gateway's ai.usage rows.
"""
import json
import os
import sys
import time

BASE = os.environ["VERIFY_BASE"]
KEY = os.environ["VERIFY_KEY"]
MODEL = os.environ["VERIFY_MODEL"]
PROMPT = "Count from 1 to 20, separated by spaces. Output only the numbers."
MAX_TOKENS = 80

START = time.monotonic()
attempts = []


def on_request(_req):
    attempts.append({"t": round(time.monotonic() - START, 3), "status": None, "request_id": None,
                     "retry_after": None})


def on_response(resp):
    if attempts:
        a = attempts[-1]
        a["status"] = resp.status_code
        a["request_id"] = resp.headers.get("x-beyond-request-id")
        a["retry_after"] = resp.headers.get("retry-after")


def http(mod):
    # No timeout of our own: the SDK's per-request timeout and retry policy are what is under test.
    return mod.Client(event_hooks={"request": [on_request], "response": [on_response]},
                      timeout=None)


def describe(e):
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) or {}
    body = getattr(e, "body", None)
    return {"kind": type(e).__name__, "status": getattr(e, "status_code", None),
            "message": str(e)[:500], "body": body if isinstance(body, (dict, list)) else None,
            "request_id": headers.get("x-beyond-request-id") if headers else None,
            "retry_after": headers.get("retry-after") if headers else None}


def openai_call(stream):
    import httpx
    import openai
    c = openai.OpenAI(base_url=f"{BASE}/v1", api_key=KEY, http_client=http(httpx))
    msgs = [{"role": "user", "content": PROMPT}]
    if not stream:
        r = c.chat.completions.create(model=MODEL, max_tokens=MAX_TOKENS, messages=msgs)
        u = r.usage
        d = getattr(u, "prompt_tokens_details", None)
        return True, {"input_total": u.prompt_tokens, "output": u.completion_tokens,
                      "cache_read": (getattr(d, "cached_tokens", None) or 0) if d else 0}, \
            r.choices[0].message.content or ""
    s = c.chat.completions.create(model=MODEL, max_tokens=MAX_TOKENS, messages=msgs, stream=True,
                                  stream_options={"include_usage": True})
    terminal, usage, text = False, None, ""
    for chunk in s:
        if chunk.usage:
            d = getattr(chunk.usage, "prompt_tokens_details", None)
            usage = {"input_total": chunk.usage.prompt_tokens, "output": chunk.usage.completion_tokens,
                     "cache_read": (getattr(d, "cached_tokens", None) or 0) if d else 0}
        for ch in chunk.choices or []:
            text += ch.delta.content or ""
            if ch.finish_reason:
                terminal = True
    return terminal, usage, text


def anthropic_call(stream):
    import anthropic
    try:
        import httpx2 as mod
    except ImportError:
        import httpx as mod
    c = anthropic.Anthropic(base_url=BASE, api_key=KEY, http_client=http(mod))
    msgs = [{"role": "user", "content": PROMPT}]

    def usage_of(u, base=None):
        cr = getattr(u, "cache_read_input_tokens", None) or 0
        cw = getattr(u, "cache_creation_input_tokens", None) or 0
        return {"input_total": (u.input_tokens or 0) + cr + cw, "output": u.output_tokens or 0,
                "cache_read": cr}

    if not stream:
        r = c.messages.create(model=MODEL, max_tokens=MAX_TOKENS, messages=msgs)
        return True, usage_of(r.usage), "".join(b.text for b in r.content if b.type == "text")
    s = c.messages.create(model=MODEL, max_tokens=MAX_TOKENS, messages=msgs, stream=True)
    terminal, usage, text = False, None, ""
    for ev in s:
        if ev.type == "message_start":
            usage = usage_of(ev.message.usage)
        elif ev.type == "message_delta" and usage is not None:
            # Counts are cumulative; a translated stream learns its input only at the end, and
            # the SDK's own accumulator takes it from here when present.
            usage["output"] = ev.usage.output_tokens or usage["output"]
            if getattr(ev.usage, "input_tokens", None) is not None:
                late = usage_of(ev.usage)
                usage["input_total"] = max(usage["input_total"], late["input_total"])
                usage["cache_read"] = max(usage["cache_read"], late["cache_read"])
        elif ev.type == "content_block_delta" and getattr(ev.delta, "type", "") == "text_delta":
            text += ev.delta.text
        elif ev.type == "message_stop":
            terminal = True
    return terminal, usage, text


def main():
    sdk, mode = sys.argv[1], sys.argv[2]
    stream = mode == "stream"
    out = {"ok": False, "terminal": False, "usage": None, "error": None, "text": ""}
    try:
        terminal, usage, text = (openai_call if sdk == "openai" else anthropic_call)(stream)
        out.update(ok=True, terminal=terminal, usage=usage, text=text[:200])
    except Exception as e:  # noqa: BLE001 - every outcome is data for the trial
        out["error"] = describe(e)
    out["attempts"] = attempts
    out["elapsed"] = round(time.monotonic() - START, 3)
    print("VERIFY " + json.dumps(out), flush=True)


main()
