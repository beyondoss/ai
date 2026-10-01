"""Live probes: drive a stock Python SDK through the gateway the way an application would.

Usage: probe.py <probe> with env VERIFY_BASE (gateway origin), VERIFY_KEY (bai_ key), VERIFY_MODEL.

Prints one final line `VERIFY {json}`:
  ok        the client's own verdict (it parsed everything and the task's structure held)
  calls     one entry per HTTP call the SDK made: request_id, wire ("chat" | "messages" |
            "responses" | "embeddings"), and the usage the client was shown, normalized to
            {input_total, output, cache_read} where input_total includes cached tokens
  detail    free-form evidence for a failing cell
The Rust cell checks each call against the gateway's ai.usage rows (the second witness).
"""
import json
import os
import sys
import traceback

import httpx

BASE = os.environ["VERIFY_BASE"]
KEY = os.environ["VERIFY_KEY"]
MODEL = os.environ["VERIFY_MODEL"]

calls = []
_ids = []


def _hook(resp):
    _ids.append(resp.headers.get("x-beyond-request-id"))


def http(sdk="openai"):
    """An HTTP client that records every response's x-beyond-request-id. Each SDK insists on its
    own transport package (anthropic>=1.11 uses httpx2), so build the one it expects."""
    mod = httpx
    if sdk == "anthropic":
        try:
            import httpx2 as mod  # noqa: F811
        except ImportError:
            mod = httpx
    return mod.Client(event_hooks={"response": [_hook]}, timeout=120)


def take_id():
    return _ids.pop(0) if _ids else None


def record(wire, usage):
    calls.append({"request_id": take_id(), "wire": wire, "usage": usage})


def chat_usage(u):
    if u is None:
        return None
    d = getattr(u, "prompt_tokens_details", None)
    return {"input_total": u.prompt_tokens, "output": u.completion_tokens,
            "cache_read": (getattr(d, "cached_tokens", None) or 0) if d else 0}


def messages_usage(u):
    cr = u.cache_read_input_tokens or 0
    cw = u.cache_creation_input_tokens or 0
    return {"input_total": u.input_tokens + cr + cw, "output": u.output_tokens, "cache_read": cr}


def responses_usage(u):
    d = getattr(u, "input_tokens_details", None)
    return {"input_total": u.input_tokens, "output": u.output_tokens,
            "cache_read": (getattr(d, "cached_tokens", None) or 0) if d else 0}


def openai_client():
    import openai
    return openai.OpenAI(base_url=f"{BASE}/v1", api_key=KEY, http_client=http(), max_retries=0)


def anthropic_client():
    import anthropic
    return anthropic.Anthropic(base_url=BASE, api_key=KEY, http_client=http("anthropic"), max_retries=0)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def chat_basic():
    """E1: Chat Completions, non-stream then stream with the SDK's own accumulator."""
    c = openai_client()
    r = c.chat.completions.create(model=MODEL, max_tokens=1024,
                                  messages=[{"role": "user", "content": "Reply with the single word: pong"}])
    record("chat", chat_usage(r.usage))
    ok = bool(r.choices[0].message.content) and r.choices[0].message.role == "assistant"
    with c.chat.completions.stream(model=MODEL, max_tokens=1024,
                                   messages=[{"role": "user", "content": "Count from 1 to 5."}]) as s:
        final = s.get_final_completion()
    record("chat", chat_usage(final.usage))
    msg = final.choices[0].message
    ok = ok and bool(msg.content) and msg.role == "assistant"
    return ok, {"nonstream": r.choices[0].message.content, "stream_role": msg.role}


def messages_basic():
    """E2: Anthropic Messages, non-stream then stream via the SDK's final-message helper."""
    c = anthropic_client()
    r = c.messages.create(model=MODEL, max_tokens=1024,
                          messages=[{"role": "user", "content": "Reply with the single word: pong"}])
    record("messages", messages_usage(r.usage))
    ok = r.content and r.content[0].type == "text" and r.stop_reason in ("end_turn", "max_tokens")
    with c.messages.stream(model=MODEL, max_tokens=1024,
                           messages=[{"role": "user", "content": "Count from 1 to 5."}]) as s:
        final = s.get_final_message()
    record("messages", messages_usage(final.usage))
    ok = ok and final.content and final.content[0].type == "text"
    return bool(ok), {"stop": r.stop_reason}


def responses_basic():
    """E3 / TRN-1: the stock responses.create() (no `store`), non-stream then stream."""
    c = openai_client()
    r = c.responses.create(model=MODEL, input="Reply with the single word: pong", max_output_tokens=1024)
    record("responses", responses_usage(r.usage))
    ok = r.status in ("completed", "incomplete") and bool(r.output_text or r.status == "incomplete")
    with c.responses.stream(model=MODEL, input="Count from 1 to 5.", max_output_tokens=1024) as s:
        final = s.get_final_response()
    record("responses", responses_usage(final.usage))
    ok = ok and final.status in ("completed", "incomplete")
    return bool(ok), {"status": r.status}


def models_list():
    """E4: stock SDKs list the catalog and read the card fields."""
    import anthropic
    o = list(openai_client().models.list())
    _ids.clear()
    a = list(anthropic.Anthropic(base_url=BASE, api_key=KEY, http_client=http("anthropic"), max_retries=0).models.list(limit=1000))
    _ids.clear()
    m = next((x for x in o if x.id == MODEL), None)
    extra = (m.model_extra or {}) if m else {}
    ok = (len(o) >= 100 and len(a) == len(o) and m is not None
          and extra.get("context_window", 0) > 0 and "pricing" in extra and "capabilities" in extra)
    return ok, {"openai": len(o), "anthropic": len(a), "card": {k: extra.get(k) for k in ("context_window", "max_output_tokens", "pricing")}}


def tools_chat():
    """T1: forced tool call over Chat, result fed back, final answer uses it."""
    c = openai_client()
    msgs = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    r = c.chat.completions.create(model=MODEL, max_tokens=1024, messages=msgs, tools=[WEATHER],
                                  tool_choice={"type": "function", "function": {"name": "get_weather"}})
    record("chat", chat_usage(r.usage))
    tc = (r.choices[0].message.tool_calls or [None])[0]
    if tc is None:
        return False, {"why": "no tool call", "message": r.choices[0].message.model_dump()}
    args = json.loads(tc.function.arguments or "{}")
    msgs.append(r.choices[0].message.model_dump(exclude_none=True))
    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": "Sunny, 31C, code ZEBRA-7"})
    with c.chat.completions.stream(model=MODEL, max_tokens=1024, messages=msgs, tools=[WEATHER]) as s:
        final = s.get_final_completion()
    record("chat", chat_usage(final.usage))
    text = final.choices[0].message.content or ""
    return ("paris" in json.dumps(args).lower() and "31" in text), {"args": args, "final": text[:200]}


def tools_messages():
    """T1: forced tool call over Messages (streaming), result fed back."""
    c = anthropic_client()
    tool = {"name": "get_weather", "description": "Weather for a city",
            "input_schema": WEATHER["function"]["parameters"]}
    msgs = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    with c.messages.stream(model=MODEL, max_tokens=1024, messages=msgs, tools=[tool],
                           tool_choice={"type": "tool", "name": "get_weather"}) as s:
        r = s.get_final_message()
    record("messages", messages_usage(r.usage))
    tu = next((b for b in r.content if b.type == "tool_use"), None)
    if tu is None:
        return False, {"why": "no tool_use", "content": [b.model_dump() for b in r.content]}
    msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id,
                                              "content": "Sunny, 31C, code ZEBRA-7"}]})
    f = c.messages.create(model=MODEL, max_tokens=1024, messages=msgs, tools=[tool])
    record("messages", messages_usage(f.usage))
    text = "".join(b.text for b in f.content if b.type == "text")
    return ("paris" in json.dumps(tu.input).lower() and "31" in text), {"input": tu.input, "final": text[:200]}


def embeddings():
    """M1: embeddings, including a batch past 64 KiB (openai-python puts input before model)."""
    c = openai_client()
    r = c.embeddings.create(model=MODEL, input=["alpha", "beta", "gamma"])
    record("embeddings", {"input_total": r.usage.prompt_tokens, "output": 0, "cache_read": 0})
    chunk = "lorem ipsum dolor sit amet " * 16
    big = c.embeddings.create(model=MODEL, input=[f"{i} {chunk}" for i in range(200)])
    record("embeddings", {"input_total": big.usage.prompt_tokens, "output": 0, "cache_read": 0})
    ok = len(r.data) == 3 and len(big.data) == 200 and len(r.data[0].embedding) > 100
    return ok, {"dims": len(r.data[0].embedding)}


PROBES = {f.__name__: f for f in [chat_basic, messages_basic, responses_basic, models_list,
                                  tools_chat, tools_messages, embeddings]}

if __name__ == "__main__":
    name = sys.argv[1]
    try:
        ok, detail = PROBES[name]()
    except Exception as e:  # noqa: BLE001 - every failure shape is evidence
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": calls, "detail": detail}, default=str))
