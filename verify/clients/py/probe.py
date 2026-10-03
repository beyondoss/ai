"""Live probes: drive a stock Python SDK through the gateway the way an application would.

Usage: probe.py <probe> with env VERIFY_BASE (gateway origin), VERIFY_KEY (bai_ key), VERIFY_MODEL,
VERIFY_PROVIDER (the route's first pool), VERIFY_CLIENT (the cell's client name), and, for a BYO
probe only, VERIFY_BYO_KEY (that provider's real key).

Prints one final line `VERIFY {json}`:
  ok        the client's own verdict (it parsed everything and the task's structure held)
  calls     one entry per HTTP call the SDK made: request_id, wire ("chat" | "messages" |
            "responses" | "embeddings"), and the usage the client was shown, normalized to
            {input_total, output, cache_read} where input_total includes cached tokens, plus
            optional reasoning / service_tier breakouts; and `expect`, when the call's row is
            not the ordinary one (no row, a refusal, an estimate; see `record`)
  errors    every answer with status >= 400: request_id, status, the x-beyond-provider that sent
            it (none when the gateway itself refused), and its retry headers, so the cell can
            tell a provider that was unavailable from a gateway failure
  detail    free-form evidence for a failing cell
The Rust cell checks each call against the gateway's ai.usage rows (the second witness).
"""
import json
import os
import sys
import time
import traceback

import httpx

BASE = os.environ["VERIFY_BASE"]
KEY = os.environ["VERIFY_KEY"]
MODEL = os.environ["VERIFY_MODEL"]
# E4: how many rows the catalog holds (the Rust side counts MODEL_ROUTES), so a listing is held
# to exactly that, never to a floor that goes stale as rows come and go.
CATALOG_ROWS = int(os.environ.get("VERIFY_CATALOG_ROWS", "-1"))

calls = []
_ids = []
# Every error answer, with what the cell's retry policy reads (see live.rs `retryable_failure`).
errors = []


def _hook(resp):
    _ids.append(resp.headers.get("x-beyond-request-id"))
    if resp.status_code >= 400:
        h = resp.headers.get
        errors.append({"request_id": h("x-beyond-request-id"), "status": resp.status_code,
                       "provider": h("x-beyond-provider"), "retry_after": h("retry-after"),
                       "retry_after_ms": h("retry-after-ms"), "should_retry": h("x-should-retry")})


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


def record(wire, usage, **expect):
    """One client call. `expect` shapes what the ledger must hold for it (see live.rs
    `call_problems`): rows=0 (free / BYO), error=True (refused, bills nothing), estimated=True
    (cut short), exact=True (no usage shown, still billed exactly), provider, row_min."""
    call = {"request_id": take_id(), "wire": wire, "usage": usage}
    if expect:
        call["expect"] = expect
    calls.append(call)


def record_tagged(tag, wire, **expect):
    """A call that never saw a response head, so it has no request id: the ledger finds its row by
    the `x-beyond-metadata: {"verify": tag}` it sent."""
    calls.append({"request_id": None, "tag": tag, "wire": wire, "usage": None, "expect": expect})


def outside_reasoning(prompt, completion, total, reasoning):
    """Reasoning a provider reports beside the completion count (xAI: total = prompt + completion +
    reasoning) is output the client is billed for; OpenAI counts it inside completion_tokens."""
    return reasoning if reasoning and total == prompt + completion + reasoning else 0


def _with(out, reasoning=0, service_tier=None):
    """Optional breakouts the ledger also checks: reasoning (BIL-9) and the echoed tier (BIL-11)."""
    if reasoning:
        out["reasoning"] = reasoning
    if isinstance(service_tier, str):
        out["service_tier"] = service_tier
    return out


def chat_usage(u, service_tier=None):
    if u is None:
        return None
    d = getattr(u, "prompt_tokens_details", None)
    cd = getattr(u, "completion_tokens_details", None)
    reasoning = (getattr(cd, "reasoning_tokens", None) or 0) if cd else 0
    extra = outside_reasoning(u.prompt_tokens, u.completion_tokens, u.total_tokens, reasoning)
    return _with({"input_total": u.prompt_tokens, "output": u.completion_tokens + extra,
                  "cache_read": (getattr(d, "cached_tokens", None) or 0) if d else 0}, reasoning, service_tier)


def messages_usage(u):
    cr = u.cache_read_input_tokens or 0
    cw = u.cache_creation_input_tokens or 0
    return _with({"input_total": u.input_tokens + cr + cw, "output": u.output_tokens, "cache_read": cr},
                 service_tier=getattr(u, "service_tier", None))


def responses_usage(u, service_tier=None):
    d = getattr(u, "input_tokens_details", None)
    od = getattr(u, "output_tokens_details", None)
    return _with({"input_total": u.input_tokens, "output": u.output_tokens,
                  "cache_read": (getattr(d, "cached_tokens", None) or 0) if d else 0},
                 (getattr(od, "reasoning_tokens", None) or 0) if od else 0, service_tier)


def openai_client():
    import openai
    return openai.OpenAI(base_url=f"{BASE}/v1", api_key=KEY, http_client=http(), max_retries=0)


def anthropic_client():
    import anthropic
    return anthropic.Anthropic(base_url=BASE, api_key=KEY, http_client=http("anthropic"), max_retries=0)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def chat_basic():
    """E1: Chat Completions, non-stream then stream with the SDK's own accumulator. S2: the message
    the accumulator built (its role as accumulated, its text) goes back as the next turn's history,
    and that turn is accepted: a repeated `delta.role` would have made the role "assistantassistant"."""
    c = openai_client()
    r = c.chat.completions.create(model=MODEL, max_tokens=1024,
                                  messages=[{"role": "user", "content": "Reply with the single word: pong"}])
    record("chat", chat_usage(r.usage))
    ok = bool(r.choices[0].message.content) and r.choices[0].message.role == "assistant"
    history = [{"role": "user", "content": "Count from 1 to 5."}]
    with c.chat.completions.stream(model=MODEL, max_tokens=1024, messages=history) as s:
        final = s.get_final_completion()
    record("chat", chat_usage(final.usage))
    msg = final.choices[0].message
    ok = ok and bool(msg.content) and msg.role == "assistant"
    history += [{"role": msg.role, "content": msg.content},
                {"role": "user", "content": "Now reply with only the next number after the last one you wrote."}]
    nxt = c.chat.completions.create(model=MODEL, max_tokens=1024, messages=history)
    record("chat", chat_usage(nxt.usage))
    ok = ok and bool(nxt.choices[0].message.content)
    return ok, {"nonstream": r.choices[0].message.content, "stream_role": msg.role,
                "next_turn": nxt.choices[0].message.content}


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
    ok = (len(o) == CATALOG_ROWS and len(a) == len(o) and m is not None
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


def langchain_chat():
    """E1 / T1 via LangChain: ChatOpenAI with a bound tool, then the answer."""
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=MODEL, base_url=f"{BASE}/v1", api_key=KEY, http_client=http(), max_retries=0,
                     max_completion_tokens=1024)
    r = llm.invoke("Reply with the single word: pong")
    record("chat", _lc_usage(r))
    bound = llm.bind_tools([WEATHER], tool_choice="get_weather")
    msgs = [HumanMessage("What's the weather in Paris? Use the tool.")]
    ai = bound.invoke(msgs)
    record("chat", _lc_usage(ai))
    if not ai.tool_calls:
        return False, {"why": "no tool call"}
    msgs += [ai, ToolMessage("Sunny, 31C", tool_call_id=ai.tool_calls[0]["id"])]
    final = llm.bind_tools([WEATHER]).invoke(msgs)
    record("chat", _lc_usage(final))
    return bool(r.content) and "31" in str(final.content), {"final": str(final.content)[:120]}


def _lc_usage(m):
    u = m.usage_metadata or {}
    i, o = u.get("input_tokens", 0), u.get("output_tokens", 0)
    r = (u.get("output_token_details") or {}).get("reasoning", 0)
    return {"input_total": i, "output": o + outside_reasoning(i, o, u.get("total_tokens"), r),
            "cache_read": (u.get("input_token_details") or {}).get("cache_read", 0)}


def agents_sdk():
    """E3 / T1 via the OpenAI Agents SDK (Responses by default): a tool-using agent run."""
    import asyncio
    from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled
    from openai import AsyncOpenAI
    set_tracing_disabled(True)
    set_default_openai_client(AsyncOpenAI(base_url=f"{BASE}/v1", api_key=KEY, max_retries=0,
                                          http_client=httpx.AsyncClient(event_hooks={"response": [_ahook]}, timeout=120)))

    @function_tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"Sunny in {city}, 31C"

    agent = Agent(name="weather", instructions="Use the tool, then answer in one sentence.",
                  model=MODEL, tools=[get_weather])
    res = asyncio.run(Runner.run(agent, "What's the weather in Paris?"))
    for resp in res.raw_responses:
        u = resp.usage
        record("responses", {"input_total": u.input_tokens, "output": u.output_tokens,
                             "cache_read": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0})
    used = any(getattr(i, "type", "") == "tool_call_item" for i in res.new_items)
    return used and "31" in str(res.final_output), {"final": str(res.final_output)[:120]}


async def _ahook(resp):
    _hook(resp)


def agents_chat():
    """E1 via the OpenAI Agents SDK on Chat Completions (`OpenAIChatCompletionsModel`, the model
    apps use with any OpenAI-compatible endpoint): a tool-using run, then the same streamed. The
    SDK asks no stream usage of a non-OpenAI base URL (`include_usage` unset), so a streamed turn's
    usage is the chunk the gateway adds; each turn's usage is the ModelResponse the SDK built."""
    import asyncio
    from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
    from openai import AsyncOpenAI
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=f"{BASE}/v1", api_key=KEY, max_retries=0,
                         http_client=httpx.AsyncClient(event_hooks={"response": [_ahook]}, timeout=120))

    @function_tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"Sunny in {city}, 31C"

    agent = Agent(name="weather", instructions="Use the tool, then answer in one sentence.",
                  model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client), tools=[get_weather])

    def rec(res):
        for resp in res.raw_responses:
            u = resp.usage
            record("chat", {"input_total": u.input_tokens, "output": u.output_tokens,
                            "cache_read": getattr(u.input_tokens_details, "cached_tokens", 0) or 0})

    async def both():
        # One event loop for both runs: the client's pooled connections belong to it.
        res = await Runner.run(agent, "What's the weather in Paris?")
        sres = Runner.run_streamed(agent, "What's the weather in Rome?")
        events = [e.type async for e in sres.stream_events()]
        return res, sres, events

    res, sres, events = asyncio.run(both())
    rec(res)
    rec(sres)
    detail = {}
    ok = True
    for name, r in (("run", res), ("streamed", sres)):
        kinds = [getattr(i, "type", "") for i in r.new_items]
        turns = len(r.raw_responses)
        detail[name] = {"items": kinds, "turns": turns, "final": str(r.final_output)[:120]}
        ok = ok and "tool_call_item" in kinds and turns >= 2 and "31" in str(r.final_output)
    return ok and "raw_response_event" in events, detail


# --- Endpoint, translation and billing detail -------------------------------------------------
# Each probe below asserts its claim's oracle on structure (a fixture value, a typed error, a
# counter), never on wording. Fixtures: a PNG reading "KESTREL 4821", a PDF reading "MARIGOLD
# 7316" (verify/clients/fixtures/), and the public Google logo for the URL case (Anthropic
# can't fetch Wikimedia URLs).

PROVIDER = os.environ.get("VERIFY_PROVIDER", "")
# Probes shared by both SDKs (errors, big bodies, clamps) drive the one the cell names.
CLIENT = "anthropic" if os.environ.get("VERIFY_CLIENT", "").startswith("anthropic") else "openai"
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")
IMAGE_URL = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
# A per-run nonce at the head of every cached prefix: a cell's first call is always a cache write,
# never a read left over from the previous run.
NONCE = os.urandom(6).hex()


def fixture_b64(name):
    import base64
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


def long_prefix(lines=420):
    """~6k tokens of distinct facts: past Claude Haiku 4.5's 4096-token cache minimum."""
    colors = ["red", "teal", "amber", "violet", "olive", "coral", "slate"]
    animals = ["heron", "otter", "lynx", "marmot", "ibis", "gecko", "bison", "wren"]
    cities = ["Oslo", "Lima", "Hanoi", "Accra", "Perth", "Quito", "Riga", "Turin", "Kyoto"]
    body = "\n".join(f"Fact {i}: the {colors[i % 7]} {animals[i % 8]} numbered {i * 37 % 1000} lives in "
                     f"{cities[i % 9]}." for i in range(lines))
    return f"Session {NONCE}. You are a terse assistant. Reference facts follow.\n{body}"


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["city", "country", "population_millions"],
          "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                         "population_millions": {"type": "number"}}}


def validates(obj):
    """The response matches SCHEMA exactly: every key, no extras, the right JSON types."""
    if not isinstance(obj, dict) or set(obj) != set(SCHEMA["properties"]):
        return False
    return (isinstance(obj["city"], str) and isinstance(obj["country"], str)
            and isinstance(obj["population_millions"], (int, float)) and not isinstance(obj["population_millions"], bool))


def openai_at(prefix, key=KEY, **kw):
    import openai
    return openai.OpenAI(base_url=f"{BASE}{prefix}", api_key=key, http_client=http(), max_retries=0, **kw)


def anthropic_at(prefix, key=KEY):
    import anthropic
    return anthropic.Anthropic(base_url=f"{BASE}{prefix}", api_key=key, http_client=http("anthropic"), max_retries=0)


def text_of(content):
    return "".join(b.text for b in content if b.type == "text")


def responses_count_compact():
    """E5 / B4: Responses input_tokens.count is free (no row); responses.compact is billed."""
    c = openai_client()
    n = c.responses.input_tokens.count(model=MODEL, input="Count the tokens in this sentence, please.")
    record("responses", None, rows=0)
    convo = [{"role": "user", "content": "My favourite colour is teal and my cat is called Miso."},
             {"role": "assistant", "content": "Noted: teal, and a cat named Miso."},
             {"role": "user", "content": "Remember both."}]
    comp = c.responses.compact(model=MODEL, input=convo)
    record("responses", responses_usage(comp.usage), row_min={"output_tokens": 1})
    ok = n.input_tokens > 0 and len(comp.output) > 0 and comp.usage.output_tokens > 0
    return ok, {"count": n.input_tokens, "compact_items": len(comp.output), "usage": comp.usage.model_dump()}


def count_tokens():
    """E5 / B4: Anthropic count_tokens forwards and is free (no row)."""
    c = anthropic_client()
    n = c.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": "Count the tokens in this sentence."}])
    record("messages", None, rows=0)
    tool = {"name": "get_weather", "description": "Weather for a city", "input_schema": WEATHER["function"]["parameters"]}
    m = c.messages.count_tokens(model=MODEL, tools=[tool], messages=[{"role": "user", "content": "Weather in Paris?"}])
    record("messages", None, rows=0)
    return n.input_tokens > 0 and m.input_tokens > n.input_tokens, {"plain": n.input_tokens, "with_tool": m.input_tokens}


def thinking_replay():
    """T2: extended thinking plus a tool; turn 2 replays the signed thinking block and is accepted."""
    c = anthropic_client()
    tool = {"name": "get_weather", "description": "Weather for a city", "input_schema": WEATHER["function"]["parameters"]}
    msgs = [{"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."}]
    think = {"type": "enabled", "budget_tokens": 1024}
    r = c.messages.create(model=MODEL, max_tokens=4096, thinking=think, tools=[tool], messages=msgs)
    record("messages", messages_usage(r.usage))
    th = next((b for b in r.content if b.type == "thinking"), None)
    tu = next((b for b in r.content if b.type == "tool_use"), None)
    if th is None or not th.signature or tu is None:
        return False, {"why": "turn 1 lacks a signed thinking block or a tool_use", "types": [b.type for b in r.content]}
    msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": "Sunny, 31C"}]})
    with c.messages.stream(model=MODEL, max_tokens=4096, thinking=think, tools=[tool], messages=msgs) as s:
        f = s.get_final_message()
    record("messages", messages_usage(f.usage))
    return "31" in text_of(f.content), {"sig_len": len(th.signature), "final": text_of(f.content)[:160]}


RESP_WEATHER = {"type": "function", "name": "get_weather", "description": "Weather for a city",
                "parameters": WEATHER["function"]["parameters"]}


def reasoning_replay():
    """T2: Responses with store=false and encrypted reasoning; turn 2 replays the reasoning items."""
    c = openai_client()
    inp = [{"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."}]
    kw = dict(model=MODEL, store=False, include=["reasoning.encrypted_content"], reasoning={"effort": "low"},
              tools=[RESP_WEATHER], max_output_tokens=4096)
    r = c.responses.create(input=inp, **kw)
    record("responses", responses_usage(r.usage, r.service_tier))
    rs = [i for i in r.output if i.type == "reasoning"]
    fc = next((i for i in r.output if i.type == "function_call"), None)
    if not rs or not all(i.encrypted_content for i in rs) or fc is None:
        return False, {"why": "no encrypted reasoning or no function_call", "types": [i.type for i in r.output]}
    inp += [i.model_dump(exclude_none=True) for i in r.output]
    inp.append({"type": "function_call_output", "call_id": fc.call_id, "output": "Sunny, 31C"})
    f = c.responses.create(input=inp, **kw)
    record("responses", responses_usage(f.usage, f.service_tier))
    reasoning = r.usage.output_tokens_details.reasoning_tokens
    return "31" in f.output_text and reasoning > 0, {"reasoning_tokens": reasoning, "final": f.output_text[:160]}


def _agents_client():
    from agents import set_default_openai_client, set_tracing_disabled
    from openai import AsyncOpenAI
    set_tracing_disabled(True)
    set_default_openai_client(AsyncOpenAI(base_url=f"{BASE}/v1", api_key=KEY, max_retries=0,
                                          http_client=httpx.AsyncClient(event_hooks={"response": [_ahook]}, timeout=120)))


def _agents_record(res):
    for resp in res.raw_responses:
        u = resp.usage
        od = getattr(u, "output_tokens_details", None)
        record("responses", _with({"input_total": u.input_tokens, "output": u.output_tokens,
                                   "cache_read": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0},
                                  getattr(od, "reasoning_tokens", 0) or 0))


def agents_reasoning():
    """TRN-8 / T2: the Agents SDK with store=false and encrypted reasoning: the tool turn replays
    the reasoning items, without server-side item ids to lean on."""
    import asyncio
    from agents import Agent, ModelSettings, Runner, function_tool
    from openai.types.shared import Reasoning
    _agents_client()

    @function_tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"Sunny in {city}, 31C"

    settings = ModelSettings(store=False, reasoning=Reasoning(effort="low"), response_include=["reasoning.encrypted_content"])
    agent = Agent(name="weather", instructions="Always call get_weather first, then answer in one sentence.",
                  model=MODEL, tools=[get_weather], model_settings=settings)
    res = asyncio.run(Runner.run(agent, "What's the weather in Paris?"))
    _agents_record(res)
    kinds = [getattr(i, "type", "") for i in res.new_items]
    ok = "tool_call_item" in kinds and "reasoning_item" in kinds and len(res.raw_responses) >= 2
    return ok and "31" in str(res.final_output), {"items": kinds, "final": str(res.final_output)[:120]}


VISION_ASK = ("Three attachments follow: two images (one is a company logo) and a PDF. For each one, reply with "
              "the words, digits or brand name it shows, one line per attachment, copied exactly.")


def _vision_ok(text):
    t = text.upper()
    return "4821" in t and "7316" in t and "GOOGLE" in t


def vision_chat():
    """T3: base64 image, image URL and a PDF in one Chat message; the model reads each fixture."""
    c = openai_client()
    content = [{"type": "text", "text": VISION_ASK},
               {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fixture_b64('codeword.png')}"}},
               {"type": "image_url", "image_url": {"url": IMAGE_URL}},
               {"type": "file", "file": {"filename": "codeword.pdf",
                                         "file_data": f"data:application/pdf;base64,{fixture_b64('codeword.pdf')}"}}]
    r = c.chat.completions.create(model=MODEL, max_completion_tokens=2048, messages=[{"role": "user", "content": content}])
    record("chat", chat_usage(r.usage))
    text = r.choices[0].message.content or ""
    return _vision_ok(text), {"text": text[:200]}


def vision_messages():
    """T3: base64 image, URL image and a base64 PDF document over Messages."""
    c = anthropic_client()
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": fixture_b64("codeword.png")}},
               {"type": "image", "source": {"type": "url", "url": IMAGE_URL}},
               {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                               "data": fixture_b64("codeword.pdf")}},
               {"type": "text", "text": VISION_ASK}]
    r = c.messages.create(model=MODEL, max_tokens=2048, messages=[{"role": "user", "content": content}])
    record("messages", messages_usage(r.usage))
    text = text_of(r.content)
    return _vision_ok(text), {"text": text[:200]}


def structured_chat():
    """T4: json_schema response_format; the answer validates against the schema."""
    c = openai_client()
    r = c.chat.completions.create(model=MODEL, max_completion_tokens=2048,
                                  messages=[{"role": "user", "content": "Give the city, country and population (millions) of Paris."}],
                                  response_format={"type": "json_schema",
                                                   "json_schema": {"name": "city_facts", "strict": True, "schema": SCHEMA}})
    record("chat", chat_usage(r.usage))
    text = r.choices[0].message.content or ""
    try:
        obj = json.loads(text)
    except ValueError:
        return False, {"why": "not JSON", "text": text[:300]}
    return validates(obj), {"obj": obj}


def structured_messages():
    """T4: Messages output_config json_schema; the text block validates against the schema."""
    c = anthropic_client()
    r = c.messages.create(model=MODEL, max_tokens=2048,
                          messages=[{"role": "user", "content": "Give the city, country and population (millions) of Paris."}],
                          output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    record("messages", messages_usage(r.usage))
    text = text_of(r.content)
    try:
        obj = json.loads(text)
    except ValueError:
        return False, {"why": "not JSON", "text": text[:300]}
    return validates(obj), {"obj": obj}


def langchain_structured():
    """T4 via LangChain: with_structured_output(method="json_schema") parses a valid object."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=MODEL, base_url=f"{BASE}/v1", api_key=KEY, http_client=http(), max_retries=0,
                     max_completion_tokens=2048)
    out = llm.with_structured_output({"title": "city_facts", **SCHEMA}, method="json_schema", strict=True,
                                     include_raw=True).invoke("Give the city, country and population (millions) of Paris.")
    record("chat", _lc_usage(out["raw"]))
    return out["parsing_error"] is None and validates(out["parsed"]), {"parsed": out["parsed"], "err": str(out["parsing_error"])}


def reasoning_effort():
    """T5: reasoning_effort low and high both reach the model family without an upstream 400."""
    c = openai_client()
    seen = {}
    for effort in ("low", "high"):
        r = c.chat.completions.create(model=MODEL, reasoning_effort=effort, max_completion_tokens=4096,
                                      messages=[{"role": "user", "content": "What is 17 * 23? Reply with just the number."}])
        record("chat", chat_usage(r.usage))
        seen[effort] = r.choices[0].message.content or ""
    return all("391" in t for t in seen.values()), seen


def typed_error():
    """T6: an input only the provider can reject (a corrupt PNG; no gateway rule can see it)
    arrives as the SDK's own typed 400 in the SDK's own envelope, carrying the provider's message,
    and bills nothing."""
    import anthropic
    import openai
    corrupt = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfAAAA"
    wire = "messages" if CLIENT == "anthropic" else "chat"
    try:
        if CLIENT == "anthropic":
            anthropic_client().messages.create(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": corrupt}},
                {"type": "text", "text": "What is this?"}]}])
        else:
            openai_client().chat.completions.create(model=MODEL, max_completion_tokens=64, messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{corrupt}"}},
                {"type": "text", "text": "What is this?"}]}])
    except (openai.BadRequestError, anthropic.BadRequestError) as e:
        record(wire, None, error=True)
        body = e.body if isinstance(e.body, dict) else {}
        # The envelope each SDK documents: OpenAI's `error` object (the SDK unwraps it into
        # `body`) with a message; Anthropic's {"type": "error", "error": {"type", "message"}}.
        if CLIENT == "anthropic":
            err = body.get("error") if isinstance(body.get("error"), dict) else {}
            envelope = body.get("type") == "error" and err.get("type") == "invalid_request_error"
            msg = err.get("message") or ""
        else:
            envelope = isinstance(body.get("message"), str)
            msg = body.get("message") or ""
        ok = e.status_code == 400 and envelope and "image" in msg.lower()
        return ok, {"type": type(e).__name__, "body": body or e.body}
    record(wire, None, error=True)
    return False, {"why": "a corrupt image was accepted"}


def _raw_chat(c, headers, **kw):
    raw = c.chat.completions.with_raw_response.create(model=MODEL, max_completion_tokens=256, extra_headers=headers,
                                                      messages=[{"role": "user", "content": "Reply with the single word: pong"}], **kw)
    return raw.parse(), raw.headers.get("x-beyond-provider")


def steer_providers():
    """R3: x-beyond-only and x-beyond-order pick the serving provider: the response header and the
    billing row both name the requested one."""
    c = openai_client()
    plan = [({"x-beyond-only": "openrouter"}, "openrouter"), ({"x-beyond-only": "openrouter"}, "openrouter"),
            ({"x-beyond-order": "anthropic,openrouter"}, "anthropic"), ({"x-beyond-order": "openrouter,anthropic"}, "openrouter"),
            ({"x-beyond-only": "anthropic"}, "anthropic")]
    got = []
    for headers, want in plan:
        r, served = _raw_chat(c, headers)
        record("chat", chat_usage(r.usage), provider=want)
        got.append(served)
    return got == [w for _, w in plan], {"served": got}


def session_pin():
    """R4: with two pools on the Claude row, a multi-turn conversation stays on one provider and
    its prompt cache is read from turn 2 on."""
    c = anthropic_client()
    system = [{"type": "text", "text": long_prefix(), "cache_control": {"type": "ephemeral"}}]
    msgs, served, reads = [], [], []
    for q in ("Which city does fact 3 name?", "And fact 4?", "And fact 5?"):
        msgs.append({"role": "user", "content": q})
        raw = c.messages.with_raw_response.create(model=MODEL, max_tokens=128, system=system, messages=msgs)
        r = raw.parse()
        served.append(raw.headers.get("x-beyond-provider"))
        reads.append(r.usage.cache_read_input_tokens or 0)
        record("messages", messages_usage(r.usage), provider=served[0], **({"row_min": {"cache_read_tokens": 1}} if len(reads) > 1 else {}))
        msgs.append({"role": "assistant", "content": text_of(r.content) or "ok"})
    ok = len(set(served)) == 1 and served[0] is not None and reads[1] > 0 and reads[2] >= reads[1]
    return ok, {"served": served, "cache_read": reads}


def _filler(n_bytes):
    words = ["granite", "meadow", "lantern", "harbor", "cobalt", "thistle", "ember", "quarry", "saffron", "willow"]
    out, i, size = [], 0, 0
    while size < n_bytes:
        w = f"{words[i % 10]}{i % 97}"
        out.append(w)
        size += len(w) + 1
        i += 1
    return " ".join(out)


def big_body():
    """R5: a ~210 KB conversation (past the 64 KiB replay buffer) is served, with one row."""
    turns = []
    for k in range(6):
        turns.append({"role": "user", "content": f"Part {k} of a log to keep:\n{_filler(35_000)}"})
        turns.append({"role": "assistant", "content": f"Stored part {k}."})
    turns.append({"role": "user", "content": "Reply with the single word: done"})
    size = len(json.dumps(turns))
    if CLIENT == "anthropic":
        r = anthropic_client().messages.create(model=MODEL, max_tokens=64, messages=turns)
        record("messages", messages_usage(r.usage))
        text = text_of(r.content)
    else:
        r = openai_client().chat.completions.create(model=MODEL, max_completion_tokens=1024, messages=turns)
        record("chat", chat_usage(r.usage))
        text = r.choices[0].message.content or ""
    return size > 200_000 and bool(text), {"bytes": size, "text": text[:80]}


COUNT_ASK = [{"role": "user", "content": "Count from 1 to 400 in words (one, two, ...), comma separated. No other text."}]


def stream_abort():
    """B2 / BIL-3 / BIL-20: the same stream completed, then aborted after a few content chunks.
    The aborted call's row is an estimate with output > 0, and never more than the completed
    call's real usage."""
    c = openai_client()
    s = c.chat.completions.create(model=MODEL, max_completion_tokens=6000, stream=True,
                                  stream_options={"include_usage": True}, messages=COUNT_ASK)
    usage = None
    for chunk in s:
        usage = chunk.usage or usage
    full = chat_usage(usage)
    record("chat", full)
    s = c.chat.completions.create(model=MODEL, max_completion_tokens=6000, stream=True,
                                  stream_options={"include_usage": True}, messages=COUNT_ASK)
    got = 0
    for chunk in s:
        if any(ch.delta.content for ch in chunk.choices):
            got += 1
            if got >= 25:
                break
    s.response.close()
    record("chat", None, estimated=True, output_min=1, output_max=full["output"], input_max=full["input_total"])
    return full is not None and got >= 25, {"completed": full, "chunks_before_abort": got}


def cancel_before_head():
    """BIL-3: a non-stream call the client abandons before the response head is still billed, as
    an estimate (matched by its metadata tag; it never saw a request id)."""
    import openai
    tag = f"cancel-{NONCE}"
    c = openai_at("/v1", timeout=2.0)
    try:
        c.chat.completions.create(model=MODEL, max_completion_tokens=4000, extra_headers={"x-beyond-metadata": json.dumps({"verify": tag})},
                                  messages=[{"role": "user", "content": "Write a 1500-word story about a lighthouse keeper."}])
    except openai.APITimeoutError:
        _ids.clear()
        record_tagged(tag, "chat", estimated=True)
        return True, {"timed_out": True}
    record("chat", None)
    return False, {"why": "the call finished inside the 2s timeout; nothing was cancelled"}


def stream_no_usage():
    """BIL-2: a stream with stream_options.include_usage=false is still billed exactly."""
    c = openai_client()
    s = c.chat.completions.create(model=MODEL, max_completion_tokens=1024, stream=True,
                                  stream_options={"include_usage": False},
                                  messages=[{"role": "user", "content": "Count from 1 to 5."}])
    text, usage = "", None
    for chunk in s:
        usage = chunk.usage or usage
        text += "".join(ch.delta.content or "" for ch in chunk.choices)
    if usage is not None:
        record("chat", chat_usage(usage))
    else:
        record("chat", None, exact=True)
    return bool(text), {"text": text[:80], "client_saw_usage": usage is not None}


def prompt_cache():
    """B3 / BIL-8 / BIL-11: explicit cache_control: call 1 writes, call 2 (streamed) reads; both
    sides agree on the cache counts, and the echoed service tier is recorded."""
    c = anthropic_client()
    system = [{"type": "text", "text": long_prefix(), "cache_control": {"type": "ephemeral"}}]
    q = [{"role": "user", "content": "Which city does fact 2 name?"}]
    r = c.messages.create(model=MODEL, max_tokens=64, system=system, messages=q)
    record("messages", messages_usage(r.usage), row_min={"cache_write_tokens": 1})
    with c.messages.stream(model=MODEL, max_tokens=64, system=system, messages=q) as s:
        f = s.get_final_message()
    record("messages", messages_usage(f.usage), row_min={"cache_read_tokens": 1})
    ok = (r.usage.cache_creation_input_tokens or 0) > 0 and (f.usage.cache_read_input_tokens or 0) > 0
    return ok, {"write": r.usage.cache_creation_input_tokens, "read": f.usage.cache_read_input_tokens,
                "tier": getattr(r.usage, "service_tier", None)}


def cache_ttl_1h():
    """BIL-11: a 1-hour cache_control write is recorded as cache_write_1h_tokens."""
    c = anthropic_client()
    system = [{"type": "text", "text": long_prefix(), "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    r = c.messages.create(model=MODEL, max_tokens=64, system=system,
                          messages=[{"role": "user", "content": "Which city does fact 2 name?"}])
    record("messages", messages_usage(r.usage), row_min={"cache_write_1h_tokens": 1})
    cc = getattr(r.usage, "cache_creation", None)
    h1 = (getattr(cc, "ephemeral_1h_input_tokens", 0) or 0) if cc else 0
    return h1 > 0, {"ephemeral_1h": h1, "tier": getattr(r.usage, "service_tier", None)}


def auto_cache():
    """K1 / B3: a multi-turn OpenAI-SDK conversation with a long system prompt and no
    cache_control reads the cache after turn 1, and every row equals the cached tokens the client
    saw.

    Claude rows: the gateway places the breakpoints, and Anthropic documents a cache entry as
    available once the response that wrote it begins, so turns 2 and 3, sent after it, must read.

    OpenAI rows: caching is OpenAI's own and best effort. Its prompt-caching guide: "A request can
    reuse a cached prefix only if it reaches a machine holding a matching entry that has not
    expired", and on models before GPT-5.6 a stable `prompt_cache_key` (sent here; the gateway relays
    it) "help[s] route related requests to the same cache ... Keys influence routing; they do not
    pin requests to a machine or guarantee a cache hit." No turn is guaranteed a read (live
    2026-10-01 on gpt-5.1: [0, 0, 6144]), so the oracle is a read on any later turn: that proves the
    gateway kept the prefix byte-stable and metered the hit. No read at all is OpenAI's documented
    miss, not a gateway fault: the cell is INCONCLUSIVE (`best_effort`), never a pass."""
    c = openai_client()
    best_effort = PROVIDER == "openai"
    extra = {"prompt_cache_key": f"verify-{NONCE}"} if best_effort else {}
    msgs = [{"role": "system", "content": long_prefix()}]
    reads = []
    for q in ("Which city does fact 3 name?", "And fact 4?", "And fact 5?"):
        msgs.append({"role": "user", "content": q})
        r = c.chat.completions.create(model=MODEL, max_completion_tokens=1024, messages=msgs, extra_body=extra)
        u = chat_usage(r.usage, r.service_tier)
        reads.append(u["cache_read"])
        # The row must equal what the client saw either way; a read is required of the row only
        # where the provider guarantees one.
        record("chat", u, **({"row_min": {"cache_read_tokens": 1}} if len(reads) > 1 and not best_effort else {}))
        msgs.append({"role": "assistant", "content": r.choices[0].message.content or "ok"})
    detail = {"cache_read": reads}
    if not best_effort:
        return all(x > 0 for x in reads[1:]), detail
    if any(x > 0 for x in reads[1:]):
        return True, detail
    detail["best_effort"] = (f"OpenAI served no cache read on turns 2-{len(reads)} (cache_read {reads}, "
                             "prompt_cache_key sent); its prompt caching guarantees no hit")
    return False, detail


def langchain_cache():
    """K1 via LangChain's ChatOpenAI: turn 2+ reports cache reads on a Claude row."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=MODEL, base_url=f"{BASE}/v1", api_key=KEY, http_client=http(), max_retries=0,
                     max_completion_tokens=256)
    msgs, reads = [SystemMessage(long_prefix())], []
    for q in ("Which city does fact 3 name?", "And fact 4?", "And fact 5?"):
        msgs.append(HumanMessage(q))
        ai = llm.invoke(msgs)
        u = _lc_usage(ai)
        reads.append(u["cache_read"])
        record("chat", u)
        msgs.append(AIMessage(ai.content or "ok"))
    return all(x > 0 for x in reads[1:]), {"cache_read": reads}


def byo_key():
    """A1: the managed key works on /v1; a forged bai key is a 401; the provider's own key sent
    through /{provider}/ is served and writes no row."""
    import anthropic
    import openai
    byo, ask = os.environ["VERIFY_BYO_KEY"], [{"role": "user", "content": "Reply with the single word: pong"}]
    forged = KEY[:-6] + ("AAAAAA" if not KEY.endswith("AAAAAA") else "BBBBBB")
    if PROVIDER == "anthropic":
        r = anthropic_client().messages.create(model=MODEL, max_tokens=64, messages=ask)
        record("messages", messages_usage(r.usage))
        b = anthropic_at("/anthropic", byo).messages.create(model=MODEL, max_tokens=64, messages=ask)
        record("messages", messages_usage(b.usage), rows=0)
        bad, ok_bad = anthropic.AuthenticationError, bool(text_of(b.content))
        call = lambda: anthropic_at("", forged).messages.create(model=MODEL, max_tokens=64, messages=ask)  # noqa: E731
    else:
        r = openai_client().chat.completions.create(model=MODEL, max_completion_tokens=256, messages=ask)
        record("chat", chat_usage(r.usage))
        b = openai_at("/openai/v1", byo).chat.completions.create(model=MODEL, max_completion_tokens=256, messages=ask)
        record("chat", chat_usage(b.usage), rows=0)
        bad, ok_bad = openai.AuthenticationError, bool(b.choices[0].message.content)
        call = lambda: openai_at("/v1", forged).chat.completions.create(model=MODEL, max_completion_tokens=64, messages=ask)  # noqa: E731
    try:
        call()
        record("chat", None, rows=0)
        return False, {"why": "a forged bai key was accepted"}
    except bad as e:
        record("chat", None, rows=0)
        return ok_bad and e.status_code == 401, {"forged_status": e.status_code}


def provider_routed():
    """BIL-1: a managed key on the provider route (/{provider}/...) is billed from that path's wire,
    non-stream and streamed, on Chat, Responses and Messages."""
    if PROVIDER == "anthropic":
        c = anthropic_at("/anthropic")
        r = c.messages.create(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": "Reply: pong"}])
        record("messages", messages_usage(r.usage))
        with c.messages.stream(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": "Count to 3."}]) as s:
            f = s.get_final_message()
        record("messages", messages_usage(f.usage))
        return bool(text_of(r.content)) and bool(text_of(f.content)), {}
    c = openai_at(f"/{PROVIDER}/v1")
    r = c.chat.completions.create(model=MODEL, max_completion_tokens=1024, messages=[{"role": "user", "content": "Reply: pong"}])
    record("chat", chat_usage(r.usage, r.service_tier))
    with c.chat.completions.stream(model=MODEL, max_completion_tokens=1024, messages=[{"role": "user", "content": "Count to 3."}]) as s:
        final = s.get_final_completion()
    record("chat", chat_usage(final.usage))
    ok = bool(r.choices[0].message.content) and bool(final.choices[0].message.content)
    if PROVIDER == "openai":
        rr = c.responses.create(model=MODEL, input="Reply: pong", max_output_tokens=1024)
        record("responses", responses_usage(rr.usage, rr.service_tier))
        with c.responses.stream(model=MODEL, input="Count to 3.", max_output_tokens=1024) as s:
            fr = s.get_final_response()
        record("responses", responses_usage(fr.usage, fr.service_tier))
        ok = ok and bool(rr.output_text) and bool(fr.output_text)
    return ok, {}


def reasoning_metered():
    """BIL-9: reasoning is counted once: the client sees reasoning tokens, and the row's output
    and reasoning breakout equal what the client was shown (xAI reports it beside completion)."""
    c = openai_client()
    ask = [{"role": "user", "content": "What is the sum of the decimal digits of 2 to the power 20? Reply with the number."}]
    r = c.chat.completions.create(model=MODEL, max_completion_tokens=4096, messages=ask, reasoning_effort="medium")
    u = chat_usage(r.usage, r.service_tier)
    record("chat", u)
    rr = c.responses.create(model=MODEL, input=ask[0]["content"], max_output_tokens=4096, reasoning={"effort": "medium"})
    ru = responses_usage(rr.usage, rr.service_tier)
    record("responses", ru)
    ok = u.get("reasoning", 0) > 0 and ru.get("reasoning", 0) > 0 and "31" in (r.choices[0].message.content or "")
    return ok, {"chat": u, "responses": ru}


def web_search():
    """BIL-10: Anthropic's web_search server tool is metered: server_tool_calls on the row."""
    c = anthropic_client()
    r = c.messages.create(model=MODEL, max_tokens=1024, tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
                          tool_choice={"type": "any"},
                          messages=[{"role": "user", "content": "Use web search once: what is today's top story on bbc.com? One sentence."}])
    stu = getattr(r.usage, "server_tool_use", None)
    n = (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
    record("messages", messages_usage(r.usage), row_min={"server_tool_calls": max(n, 1)})
    return n >= 1 and any(b.type == "web_search_tool_result" for b in r.content), {"web_search_requests": n}


def mid_system():
    """TRN-2: a system message and a developer message mid-conversation both take effect."""
    c = openai_client()
    seen = {}
    for role in ("system", "developer"):
        msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello."},
                {"role": role, "content": "From now on, end every reply with the exact token ZEBRA-7."},
                {"role": "user", "content": "Name one fruit."}]
        r = c.chat.completions.create(model=MODEL, max_completion_tokens=1024, messages=msgs)
        record("chat", chat_usage(r.usage))
        seen[role] = r.choices[0].message.content or ""
    return all("ZEBRA-7" in t for t in seen.values()), seen


def developer_role():
    """TRN-16: the developer role reaches a non-OpenAI host mapped (Chat and Responses)."""
    c = openai_client()
    dev = {"role": "developer", "content": "End every reply with the exact token ZEBRA-7."}
    user = {"role": "user", "content": "Name one fruit."}
    r = c.chat.completions.create(model=MODEL, max_completion_tokens=1024, messages=[dev, user])
    record("chat", chat_usage(r.usage))
    rr = c.responses.create(model=MODEL, input=[dev, user], max_output_tokens=1024)
    record("responses", responses_usage(rr.usage))
    chat, resp = r.choices[0].message.content or "", rr.output_text or ""
    return "ZEBRA-7" in chat and "ZEBRA-7" in resp, {"chat": chat[:100], "responses": resp[:100]}


DOC_TOOL = {"name": "get_doc", "description": "Fetch the reference document", "input_schema": {"type": "object", "properties": {}}}


def cache_control_turns():
    """TRN-4: cache_control on a tool_result block and on an assistant turn is honored: the next
    call reads the cache."""
    c = anthropic_client()
    msgs = [{"role": "user", "content": "Fetch the reference document."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01VerifyDoc", "name": "get_doc", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01VerifyDoc", "content": long_prefix(),
                                          "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I have read it.", "cache_control": {"type": "ephemeral"}}]}]
    reads = []
    for q in ("Which city does fact 3 name?", "Which city does fact 4 name?"):
        r = c.messages.create(model=MODEL, max_tokens=64, tools=[DOC_TOOL], messages=msgs + [{"role": "user", "content": q}])
        reads.append(r.usage.cache_read_input_tokens or 0)
        record("messages", messages_usage(r.usage), **({"row_min": {"cache_read_tokens": 1}} if len(reads) > 1 else {}))
    return reads[1] > 0, {"cache_read": reads}


def cache_control_parts():
    """TRN-4 over Chat: part-level cache_control on a tool message and an assistant message."""
    c = openai_client()
    tool = {"type": "function", "function": {"name": "get_doc", "description": "Fetch the reference document",
                                             "parameters": {"type": "object", "properties": {}}}}
    msgs = [{"role": "user", "content": "Fetch the reference document."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_verify_doc", "type": "function",
                                                                 "function": {"name": "get_doc", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_verify_doc",
             "content": [{"type": "text", "text": long_prefix(), "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I have read it.", "cache_control": {"type": "ephemeral"}}]}]
    reads = []
    for q in ("Which city does fact 3 name?", "Which city does fact 4 name?"):
        r = c.chat.completions.create(model=MODEL, max_completion_tokens=256, tools=[tool], messages=msgs + [{"role": "user", "content": q}])
        u = chat_usage(r.usage)
        reads.append(u["cache_read"])
        record("chat", u, **({"row_min": {"cache_read_tokens": 1}} if len(reads) > 1 else {}))
    return reads[1] > 0, {"cache_read": reads}


def max_tokens_clamp():
    """TRN-5: a Claude-Code-sized max_tokens (64000) to a 16k-output model is clamped, not 400'd."""
    ask = [{"role": "user", "content": "Reply with the single word: pong"}]
    if CLIENT == "anthropic":
        r = anthropic_client().messages.create(model=MODEL, max_tokens=64000, messages=ask)
        record("messages", messages_usage(r.usage))
        return bool(text_of(r.content)), {"stop": r.stop_reason}
    r = openai_client().chat.completions.create(model=MODEL, max_tokens=64000, messages=ask)
    record("chat", chat_usage(r.usage))
    return bool(r.choices[0].message.content), {"finish": r.choices[0].finish_reason}


STRICT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["city", "unit"],
                 "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}}


def strict_tools():
    """TRN-11: a strict tool definition is accepted onto Messages, and the call obeys its schema."""
    ask = "What's the weather in Paris, in Celsius? Use the tool."
    if CLIENT == "anthropic":
        tool = {"name": "get_weather", "description": "Weather for a city", "strict": True, "input_schema": STRICT_SCHEMA}
        r = anthropic_client().messages.create(model=MODEL, max_tokens=512, tools=[tool], tool_choice={"type": "tool", "name": "get_weather"},
                                               messages=[{"role": "user", "content": ask}])
        record("messages", messages_usage(r.usage))
        tu = next((b for b in r.content if b.type == "tool_use"), None)
        args = tu.input if tu else None
    else:
        tool = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "strict": True,
                                                 "parameters": STRICT_SCHEMA}}
        r = openai_client().chat.completions.create(model=MODEL, max_completion_tokens=512, tools=[tool],
                                                    tool_choice={"type": "function", "function": {"name": "get_weather"}},
                                                    messages=[{"role": "user", "content": ask}])
        record("chat", chat_usage(r.usage))
        tc = (r.choices[0].message.tool_calls or [None])[0]
        args = json.loads(tc.function.arguments) if tc else None
    ok = isinstance(args, dict) and set(args) == {"city", "unit"} and args["unit"] in ("C", "F")
    return ok, {"args": args}


def explicit_nulls():
    """TRN-15: explicit JSON nulls for optional fields never cause a 400 (Chat and Responses)."""
    c = openai_client()
    r = c.chat.completions.create(model=MODEL, max_completion_tokens=1024, stop=None, temperature=None, top_p=None, seed=None,
                                  presence_penalty=None, frequency_penalty=None, tools=None, tool_choice=None, user=None,
                                  messages=[{"role": "user", "content": "Reply with the single word: pong"}])
    record("chat", chat_usage(r.usage))
    rr = c.responses.create(model=MODEL, input="Reply with the single word: pong", max_output_tokens=1024, instructions=None,
                            temperature=None, top_p=None, tools=None, tool_choice=None, previous_response_id=None)
    record("responses", responses_usage(rr.usage))
    return bool(r.choices[0].message.content) and bool(rr.output_text), {}


def context_overflow():
    """TRN-18: a prompt past the window is refused with what harnesses compact on: OpenAI's code
    context_length_exceeded, or Anthropic's "prompt is too long" phrasing, and bills nothing.
    700 KB of filler is ~167k tokens (0.24 per byte in o200k_base): past a 128k window."""
    import anthropic
    import openai
    big = _filler(700_000)
    try:
        if CLIENT == "anthropic":
            anthropic_client().messages.create(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": big}])
        else:
            openai_client().chat.completions.create(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": big}])
    except anthropic.BadRequestError as e:
        record("messages", None, error=True)
        msg = str(e.message)
        return "prompt is too long" in msg, {"message": msg[:300]}
    except openai.BadRequestError as e:
        record("chat", None, error=True)
        return e.code == "context_length_exceeded", {"code": e.code, "message": str(e.message)[:300]}
    record("chat", None, error=True)
    return False, {"why": "an over-window prompt was accepted"}


def agents_handoff():
    """W5 / TRN-24: an OpenAI Agents SDK triage agent hands off to a tool-using agent, streamed."""
    import asyncio
    from agents import Agent, ModelSettings, Runner, function_tool
    _agents_client()

    @function_tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"Sunny in {city}, 31C"

    weather = Agent(name="weather", handoff_description="Answers weather questions with the get_weather tool.",
                    instructions="Call get_weather, then answer in one sentence.", model=MODEL, tools=[get_weather],
                    # Forced for the first turn only: the SDK resets tool_choice after a tool call.
                    model_settings=ModelSettings(tool_choice="get_weather"))
    triage = Agent(name="triage", instructions="You never answer yourself: hand weather questions to the weather agent.",
                   model=MODEL, handoffs=[weather])

    async def run():
        res = Runner.run_streamed(triage, "What's the weather in Paris?")
        events = [e.type async for e in res.stream_events()]
        return res, events

    res, events = asyncio.run(run())
    _agents_record(res)
    kinds = [getattr(i, "type", "") for i in res.new_items]
    ok = ("handoff_output_item" in kinds and "tool_call_item" in kinds and res.last_agent.name == "weather"
          and "raw_response_event" in events)
    return ok and "31" in str(res.final_output), {"items": kinds, "final": str(res.final_output)[:120]}


def langchain_agent():
    """W6: a LangChain create_agent loop: the tool is used and the final answer carries its result."""
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"Sunny in {city}, 31C"

    llm = ChatOpenAI(model=MODEL, base_url=f"{BASE}/v1", api_key=KEY, http_client=http(), max_retries=0,
                     max_completion_tokens=1024)
    agent = create_agent(model=llm, tools=[get_weather], system_prompt="Use get_weather, then answer in one sentence.")
    out = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Paris?"}]})
    for m in out["messages"]:
        if isinstance(m, AIMessage):
            record("chat", _lc_usage(m))
    used = any(isinstance(m, ToolMessage) for m in out["messages"])
    final = str(out["messages"][-1].content)
    return used and "31" in final, {"final": final[:120], "turns": len(out["messages"])}


def langchain_embeddings():
    """M1 via LangChain's OpenAIEmbeddings, stock settings (it tokenizes and sends token arrays):
    three inputs, then a 200-input batch past 64 KiB. LangChain hands back vectors only, so the
    usage the client was shown is read off the response it received."""
    from langchain_openai import OpenAIEmbeddings
    shown = []

    def hook(resp):
        resp.read()
        shown.append(resp.json().get("usage", {}).get("prompt_tokens"))

    client = httpx.Client(event_hooks={"response": [_hook, hook]}, timeout=120)
    emb = OpenAIEmbeddings(model=MODEL, base_url=f"{BASE}/v1", api_key=KEY, http_client=client, max_retries=0)
    small = emb.embed_documents(["alpha", "beta", "gamma"])
    chunk = "lorem ipsum dolor sit amet " * 16
    big = emb.embed_documents([f"{i} {chunk}" for i in range(200)])
    for n in shown:
        record("embeddings", {"input_total": n, "output": 0, "cache_read": 0})
    ok = len(small) == 3 and len(big) == 200 and len(small[0]) > 100 and len(shown) == 2
    return ok, {"dims": len(small[0]), "calls": len(shown), "prompt_tokens": shown}


def agents_structured():
    """T4 via the OpenAI Agents SDK: an agent with a typed output (`output_type`), which the SDK
    sends as a strict json_schema text format over Responses and parses into the type."""
    import asyncio
    from agents import Agent, Runner
    from pydantic import BaseModel, ConfigDict
    _agents_client()

    class CityFacts(BaseModel):
        model_config = ConfigDict(extra="forbid")
        city: str
        country: str
        population_millions: float

    agent = Agent(name="facts", instructions="Answer with the requested facts.", model=MODEL, output_type=CityFacts)
    res = asyncio.run(Runner.run(agent, "Give the city, country and population (millions) of Paris."))
    _agents_record(res)
    out = res.final_output
    obj = out.model_dump() if isinstance(out, CityFacts) else out
    return isinstance(out, CityFacts) and validates(obj), {"obj": obj, "type": type(out).__name__}


def thinking_no_echo():
    """TRN-7: a thinking + tool loop from a client that never sends thinking back. Turn 1 asks for
    reasoning and gets a tool call; turn 2 echoes only the tool call (Chat: the assistant message
    without our `thinking`; Messages: the tool_use block without its thinking block) and must
    still be accepted and use the tool result."""
    ask = "What's the weather in Paris? Use the get_weather tool, then answer in one sentence."
    if CLIENT == "anthropic":
        c = anthropic_client()
        tool = {"name": "get_weather", "description": "Weather for a city", "input_schema": WEATHER["function"]["parameters"]}
        think = {"type": "enabled", "budget_tokens": 1024}
        msgs = [{"role": "user", "content": ask}]
        r = c.messages.create(model=MODEL, max_tokens=4096, thinking=think, tools=[tool], messages=msgs)
        record("messages", messages_usage(r.usage))
        tu = next((b for b in r.content if b.type == "tool_use"), None)
        if tu is None:
            return False, {"why": "no tool_use on turn 1", "types": [b.type for b in r.content]}
        msgs.append({"role": "assistant", "content": [tu.model_dump(exclude_none=True)]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": "Sunny, 31C"}]})
        f = c.messages.create(model=MODEL, max_tokens=4096, thinking=think, tools=[tool], messages=msgs)
        record("messages", messages_usage(f.usage))
        dropped = [b.type for b in r.content if b.type != "tool_use"]
        return "31" in text_of(f.content), {"dropped": dropped, "final": text_of(f.content)[:160]}
    c = openai_client()
    msgs = [{"role": "user", "content": ask}]
    r = c.chat.completions.create(model=MODEL, max_completion_tokens=4096, reasoning_effort="high", tools=[WEATHER], messages=msgs)
    record("chat", chat_usage(r.usage))
    m = r.choices[0].message
    tc = (m.tool_calls or [None])[0]
    if tc is None:
        return False, {"why": "no tool call on turn 1", "message": m.model_dump()}
    extra = sorted(k for k in (m.model_extra or {}) if (m.model_extra or {}).get(k))
    msgs.append({"role": "assistant", "content": m.content, "tool_calls": [tc.model_dump(exclude_none=True)]})
    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": "Sunny, 31C"})
    f = c.chat.completions.create(model=MODEL, max_completion_tokens=4096, reasoning_effort="high", tools=[WEATHER], messages=msgs)
    record("chat", chat_usage(f.usage))
    text = f.choices[0].message.content or ""
    return "31" in text, {"dropped": extra, "final": text[:160]}


# --- raw HTTP (client "raw") ---------------------------------------------------------------------
# No SDK: httpx on the wire, the way a customer's own HTTP code calls the gateway. Streams are read
# as SSE by hand; usage is what the response carried.

ANTHROPIC_VERSION = {"anthropic-version": "2023-06-01"}


def bearer(key=KEY):
    return {"authorization": f"Bearer {key}"}


def sse(resp):
    """The JSON payload of every `data:` line of a streamed response."""
    for line in resp.iter_lines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                yield json.loads(data)


def wire_chat_usage(u):
    if not u:
        return None
    cd = u.get("completion_tokens_details") or {}
    extra = outside_reasoning(u["prompt_tokens"], u["completion_tokens"], u.get("total_tokens"), cd.get("reasoning_tokens") or 0)
    return {"input_total": u["prompt_tokens"], "output": u["completion_tokens"] + extra,
            "cache_read": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0}


def wire_messages_usage(u):
    cr, cw = u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0
    return {"input_total": u["input_tokens"] + cr + cw, "output": u["output_tokens"], "cache_read": cr}


def raw_chat(c, messages, headers=None, stream=False, **kw):
    """One Chat Completions call. Returns (status, provider header, text, usage shown)."""
    body = {"model": MODEL, "messages": messages, "max_completion_tokens": kw.pop("max_tokens", 1024), **kw}
    if not stream:
        r = c.post(f"{BASE}/v1/chat/completions", json=body, headers={**bearer(), **(headers or {})})
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return r.status_code, r.headers.get("x-beyond-provider"), text, wire_chat_usage(j.get("usage")), j
    body.update(stream=True, stream_options={"include_usage": True})
    with c.stream("POST", f"{BASE}/v1/chat/completions", json=body, headers={**bearer(), **(headers or {})}) as r:
        text, usage = "", None
        for ev in sse(r):
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                text += (ch.get("delta") or {}).get("content") or ""
        return r.status_code, r.headers.get("x-beyond-provider"), text, wire_chat_usage(usage), None


def raw_messages(c, messages, headers=None, stream=False, **kw):
    """One Messages call with Anthropic's own headers. Returns (status, provider, text, usage)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": kw.pop("max_tokens", 1024), **kw}
    hdrs = {"x-api-key": KEY, **ANTHROPIC_VERSION, **(headers or {})}
    if not stream:
        r = c.post(f"{BASE}/v1/messages", json=body, headers=hdrs)
        j = r.json()
        text = "".join(b.get("text", "") for b in j.get("content") or [] if b.get("type") == "text")
        return r.status_code, r.headers.get("x-beyond-provider"), text, wire_messages_usage(j["usage"]) if "usage" in j else None
    body["stream"] = True
    with c.stream("POST", f"{BASE}/v1/messages", json=body, headers=hdrs) as r:
        text, usage = "", {}
        for ev in sse(r):
            if ev.get("type") == "message_start":
                usage.update(ev["message"].get("usage") or {})
            elif ev.get("type") == "message_delta":
                usage.update({k: v for k, v in (ev.get("usage") or {}).items() if v is not None})
            elif ev.get("type") == "content_block_delta":
                text += (ev.get("delta") or {}).get("text") or ""
        return r.status_code, r.headers.get("x-beyond-provider"), text, wire_messages_usage(usage) if usage else None


PONG = [{"role": "user", "content": "Reply with the single word: pong"}]


def raw_models():
    """E4 on the wire: GET /v1/models lists every row with its card, under a Bearer key or
    Anthropic's x-api-key; HEAD answers without a body; a forged key is refused. All free."""
    c = http()
    o = c.get(f"{BASE}/v1/models", headers=bearer())
    record("models", None, rows=0)
    a = c.get(f"{BASE}/v1/models", params={"limit": 1000}, headers={"x-api-key": KEY, **ANTHROPIC_VERSION})
    record("models", None, rows=0)
    h = c.head(f"{BASE}/v1/models", headers=bearer())
    record("models", None, rows=0)
    forged = KEY[:-6] + ("AAAAAA" if not KEY.endswith("AAAAAA") else "BBBBBB")
    f = c.get(f"{BASE}/v1/models", headers=bearer(forged))
    record("models", None, rows=0)
    body = o.json()
    rows = body.get("data") or []
    ids = [m.get("id") for m in rows]
    bad = []
    for m in rows:
        p = m.get("pricing") or {}
        if not (m.get("object") == "model" and m.get("display_name") and m.get("wire") in ("openai", "anthropic")
                and (m.get("context_window") or 0) > 0 and isinstance(m.get("capabilities"), list)
                and m.get("endpoints") and all(isinstance(p.get(k), (int, float, str)) for k in
                                                ("input", "output", "cache_read", "cache_write"))):
            bad.append(m.get("id"))
    ok = (o.status_code == 200 and body.get("object") == "list" and body.get("has_more") is False and len(rows) == CATALOG_ROWS
          and len(set(ids)) == len(ids) and MODEL in ids and not bad
          and a.status_code == 200 and [m["id"] for m in a.json()["data"]] == ids
          and h.status_code == 200 and not h.content and f.status_code == 401)
    return ok, {"rows": len(rows), "bad_cards": bad[:5], "anthropic_auth": a.status_code, "head": h.status_code,
                "forged": f.status_code}


def raw_count_compact():
    """E5 / B4 on the wire: Anthropic count_tokens (Claude row), OpenAI input_tokens and compact
    (GPT row) forward to their provider; the counts are free, compact is billed."""
    c = http()
    if PROVIDER == "anthropic":
        hdrs = {"x-api-key": KEY, **ANTHROPIC_VERSION}
        counts = []
        for body in ({"model": MODEL, "messages": [{"role": "user", "content": "Count the tokens in this sentence."}]},
                     {"model": MODEL, "messages": [{"role": "user", "content": "Weather in Paris?"}],
                      "tools": [{"name": "get_weather", "description": "Weather for a city",
                                 "input_schema": WEATHER["function"]["parameters"]}]}):
            r = c.post(f"{BASE}/v1/messages/count_tokens", json=body, headers=hdrs)
            record("messages", None, rows=0)
            counts.append((r.status_code, r.json().get("input_tokens", 0)))
        ok = all(s == 200 and n > 0 for s, n in counts) and counts[1][1] > counts[0][1]
        return ok, {"counts": counts}
    r = c.post(f"{BASE}/v1/responses/input_tokens", headers=bearer(),
               json={"model": MODEL, "input": "Count the tokens in this sentence, please."})
    record("responses", None, rows=0)
    n = r.json().get("input_tokens", 0) if r.status_code == 200 else 0
    convo = [{"role": "user", "content": "My favourite colour is teal and my cat is called Miso."},
             {"role": "assistant", "content": "Noted: teal, and a cat named Miso."},
             {"role": "user", "content": "Remember both."}]
    comp = c.post(f"{BASE}/v1/responses/compact", headers=bearer(), json={"model": MODEL, "input": convo})
    j = comp.json()
    u = j.get("usage") or {}
    record("responses", {"input_total": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
                         "cache_read": (u.get("input_tokens_details") or {}).get("cached_tokens") or 0},
           row_min={"output_tokens": 1})
    ok = r.status_code == 200 and n > 0 and comp.status_code == 200 and j.get("output") and u.get("output_tokens", 0) > 0
    return ok, {"input_tokens": n, "compact_status": comp.status_code, "usage": u}


def raw_failover():
    """R1 on the wire: with the row's first provider unreachable, Chat (non-stream) and Messages
    (streamed) are served by the next candidate, and the response says which."""
    c = http()
    s1, p1, t1, u1, _ = raw_chat(c, PONG)
    record("chat", u1)
    s2, p2, t2, u2 = raw_messages(c, [{"role": "user", "content": "Count from 1 to 5."}], stream=True)
    record("messages", u2)
    ok = s1 == 200 and s2 == 200 and bool(t1) and bool(t2) and p1 == p2 == "openrouter"
    return ok, {"status": [s1, s2], "served": [p1, p2]}


def raw_steer():
    """R3 on the wire: x-beyond-only, x-beyond-order and x-beyond-split pick the serving provider;
    the response header and the billing row agree. A 50/50 split over ten calls uses both."""
    c = http()
    plan = [({"x-beyond-only": "openrouter"}, "openrouter"), ({"x-beyond-only": "anthropic"}, "anthropic"),
            ({"x-beyond-order": "openrouter,anthropic"}, "openrouter"), ({"x-beyond-order": "anthropic,openrouter"}, "anthropic"),
            ({"x-beyond-split": "openrouter=100"}, "openrouter"), ({"x-beyond-split": "anthropic=100"}, "anthropic")]
    got = []
    for headers, want in plan:
        s, served, _, u, _ = raw_chat(c, PONG, headers=headers, max_tokens=64)
        record("chat", u, provider=want)
        got.append(served)
    split = []
    for _ in range(10):
        s, served, _, u, _ = raw_chat(c, PONG, headers={"x-beyond-split": "anthropic=50,openrouter=50"}, max_tokens=64)
        record("chat", u, provider=served or "?")
        split.append(served)
    ok = got == [w for _, w in plan] and set(split) == {"anthropic", "openrouter"}
    return ok, {"served": got, "split": split}


def raw_session_pin():
    """R4 on the wire: with two pools on the Claude row and no steering headers, a three-turn
    Messages conversation stays on one provider and reads its prompt cache from turn 2."""
    c = http()
    system = [{"type": "text", "text": long_prefix(), "cache_control": {"type": "ephemeral"}}]
    msgs, served, reads = [], [], []
    for q in ("Which city does fact 3 name?", "And fact 4?", "And fact 5?"):
        msgs.append({"role": "user", "content": q})
        s, p, text, u = raw_messages(c, msgs, system=system, max_tokens=128)
        served.append(p)
        reads.append((u or {}).get("cache_read", 0))
        record("messages", u, provider=served[0] or "?", **({"row_min": {"cache_read_tokens": 1}} if len(reads) > 1 else {}))
        msgs.append({"role": "assistant", "content": text or "ok"})
    ok = len(set(served)) == 1 and served[0] is not None and reads[1] > 0 and reads[2] >= reads[1]
    return ok, {"served": served, "cache_read": reads}


def raw_big_body():
    """R5 on the wire: a ~210 KB conversation (past the 64 KiB replay buffer) is served over Chat
    and Messages, one row each."""
    turns = []
    for k in range(6):
        turns.append({"role": "user", "content": f"Part {k} of a log to keep:\n{_filler(35_000)}"})
        turns.append({"role": "assistant", "content": f"Stored part {k}."})
    turns.append({"role": "user", "content": "Reply with the single word: done"})
    size = len(json.dumps(turns))
    c = http()
    s1, _, t1, u1, _ = raw_chat(c, turns, max_tokens=256)
    record("chat", u1)
    s2, _, t2, u2 = raw_messages(c, turns, max_tokens=64)
    record("messages", u2)
    return size > 200_000 and s1 == s2 == 200 and bool(t1) and bool(t2), {"bytes": size, "status": [s1, s2]}


def raw_stream_abort():
    """B2 on the wire: the same Chat stream completed, then dropped after 25 content chunks by
    closing the connection. The dropped call's row is an estimate with output > 0, bounded by the
    completed call's usage."""
    c = http()
    _, _, _, full, _ = raw_chat(c, COUNT_ASK, stream=True, max_tokens=6000)
    record("chat", full)
    body = {"model": MODEL, "messages": COUNT_ASK, "max_completion_tokens": 6000, "stream": True,
            "stream_options": {"include_usage": True}}
    got = 0
    with c.stream("POST", f"{BASE}/v1/chat/completions", json=body, headers=bearer()) as r:
        for ev in sse(r):
            if any((ch.get("delta") or {}).get("content") for ch in ev.get("choices") or []):
                got += 1
                if got >= 25:
                    break
    c.close()
    record("chat", None, estimated=True, output_min=1, output_max=full["output"], input_max=full["input_total"])
    return full is not None and got >= 25, {"completed": full, "chunks_before_abort": got}


def byo_raw():
    """A1 on the wire: the managed key is served and billed; a forged bai key is a 401 that bills
    nothing; the provider's own key through /{provider}/ is served and writes no row."""
    byo = os.environ["VERIFY_BYO_KEY"]
    forged = KEY[:-6] + ("AAAAAA" if not KEY.endswith("AAAAAA") else "BBBBBB")
    c = http()
    if PROVIDER == "anthropic":
        s, _, t, u = raw_messages(c, PONG, max_tokens=64)
        record("messages", u)
        b = c.post(f"{BASE}/anthropic/v1/messages", headers={"x-api-key": byo, **ANTHROPIC_VERSION},
                   json={"model": MODEL, "max_tokens": 64, "messages": PONG})
        bj = b.json()
        record("messages", wire_messages_usage(bj["usage"]) if "usage" in bj else None, rows=0)
        f = c.post(f"{BASE}/v1/messages", headers={"x-api-key": forged, **ANTHROPIC_VERSION},
                   json={"model": MODEL, "max_tokens": 64, "messages": PONG})
        record("messages", None, rows=0)
        byo_ok = b.status_code == 200 and any(x.get("type") == "text" for x in bj.get("content") or [])
    else:
        s, _, t, u, _ = raw_chat(c, PONG, max_tokens=256)
        record("chat", u)
        b = c.post(f"{BASE}/openai/v1/chat/completions", headers=bearer(byo),
                   json={"model": MODEL, "max_completion_tokens": 256, "messages": PONG})
        bj = b.json()
        record("chat", wire_chat_usage(bj.get("usage")), rows=0)
        f = c.post(f"{BASE}/v1/chat/completions", headers=bearer(forged),
                   json={"model": MODEL, "max_completion_tokens": 64, "messages": PONG})
        record("chat", None, rows=0)
        byo_ok = b.status_code == 200 and bool((bj.get("choices") or [{}])[0].get("message", {}).get("content"))
    ok = s == 200 and bool(t) and byo_ok and f.status_code == 401
    return ok, {"managed": s, "byo": b.status_code, "forged": f.status_code}


def byo_models_raw():
    """E4 + A1 on the wire (D254): a BYO key's GET /v1/models is its own provider's list, not the
    catalog. Listed through the gateway and fetched directly from the provider with the same key,
    the ids match; the listing writes no row."""
    byo = os.environ["VERIFY_BYO_KEY"]
    if PROVIDER == "anthropic":
        headers, direct_url = {"x-api-key": byo, **ANTHROPIC_VERSION}, "https://api.anthropic.com/v1/models"
    else:
        headers, direct_url = bearer(byo), "https://api.openai.com/v1/models"
    params = {"limit": 1000}
    g = http().get(f"{BASE}/v1/models", params=params, headers=headers)
    record("models", None, rows=0)
    with httpx.Client(timeout=60) as c:
        d = c.get(direct_url, params=params, headers=headers)
    gj, dj = (g.json() if g.status_code == 200 else {}), (d.json() if d.status_code == 200 else {})
    gids = sorted(m.get("id") for m in gj.get("data") or [])
    dids = sorted(m.get("id") for m in dj.get("data") or [])
    # The catalog's rows carry `pricing`; a provider's own list never does.
    catalog_shaped = any("pricing" in m for m in gj.get("data") or [])
    ok = g.status_code == 200 and d.status_code == 200 and bool(dids) and gids == dids and not catalog_shaped
    return ok, {"gateway": g.status_code, "direct": d.status_code, "gateway_ids": len(gids), "direct_ids": len(dids),
                "only_gateway": sorted(set(gids) - set(dids))[:5], "only_direct": sorted(set(dids) - set(gids))[:5],
                "catalog_shaped": catalog_shaped}


def raw_auto_cache():
    """K1 on the wire: a three-turn Chat conversation with a long system prompt and no
    cache_control reads the cache from turn 2 on; the row agrees."""
    c = http()
    msgs, reads = [{"role": "system", "content": long_prefix()}], []
    for q in ("Which city does fact 3 name?", "And fact 4?", "And fact 5?"):
        msgs.append({"role": "user", "content": q})
        s, _, text, u, _ = raw_chat(c, msgs)
        reads.append((u or {}).get("cache_read", 0))
        record("chat", u, **({"row_min": {"cache_read_tokens": 1}} if len(reads) > 1 else {}))
        msgs.append({"role": "assistant", "content": text or "ok"})
    return all(x > 0 for x in reads[1:]), {"cache_read": reads}


def raw_embeddings():
    """M1 on the wire: three inputs, then a 200-input batch past 64 KiB, input tokens billed."""
    c = http()
    chunk = "lorem ipsum dolor sit amet " * 16
    out = []
    for inputs in (["alpha", "beta", "gamma"], [f"{i} {chunk}" for i in range(200)]):
        body = json.dumps({"model": MODEL, "input": inputs})
        r = c.post(f"{BASE}/v1/embeddings", content=body, headers={**bearer(), "content-type": "application/json"})
        j = r.json()
        record("embeddings", {"input_total": (j.get("usage") or {}).get("prompt_tokens", 0), "output": 0, "cache_read": 0})
        out.append((r.status_code, len(j.get("data") or []), len(body)))
    ok = out[0][:2] == (200, 3) and out[1][:2] == (200, 200) and out[1][2] > 65536
    return ok, {"calls": out}


def leak_scan():
    """SEC-7 on the wire: the route's pool key (VERIFY_BYO_KEY here) appears in no header or body
    of anything the gateway answers: a success, a stream, a provider's refusal, the model list, a
    token count, a catalog miss and a forged key. Any 16-character run of the key counts."""
    pool = os.environ["VERIFY_BYO_KEY"]
    needles = {pool[i:i + 16] for i in range(0, max(len(pool) - 15, 1))}
    c = http()
    seen, leaks = [], []

    def scan(label, status, headers, body):
        blob = "\n".join(f"{k}: {v}" for k, v in headers.items()) + "\n" + body
        seen.append((label, status))
        if any(n in blob for n in needles):
            leaks.append(label)

    corrupt = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfAAAA"
    bad_image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{corrupt}"}},
                                              {"type": "text", "text": "What is this?"}]}]
    for label, body, expect in (("chat", {"messages": PONG}, {}), ("refusal", {"messages": bad_image}, {"error": True})):
        r = c.post(f"{BASE}/v1/chat/completions", headers=bearer(),
                   json={"model": MODEL, "max_completion_tokens": 256, **body})
        j = r.json() if r.status_code == 200 else {}
        record("chat", wire_chat_usage(j.get("usage")), **expect)
        scan(label, r.status_code, r.headers, r.text)
    body = {"model": MODEL, "messages": PONG, "max_completion_tokens": 256, "stream": True,
            "stream_options": {"include_usage": True}}
    with c.stream("POST", f"{BASE}/v1/chat/completions", json=body, headers=bearer()) as r:
        raw = "".join(r.iter_text())
        usage = None
        for line in raw.splitlines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                usage = json.loads(line[5:]).get("usage") or usage
        record("chat", wire_chat_usage(usage))
        scan("stream", r.status_code, r.headers, raw)
    r = c.get(f"{BASE}/v1/models", headers=bearer())
    record("models", None, rows=0)
    scan("models", r.status_code, r.headers, r.text)
    if PROVIDER == "anthropic":
        r = c.post(f"{BASE}/v1/messages/count_tokens", headers={"x-api-key": KEY, **ANTHROPIC_VERSION},
                   json={"model": MODEL, "messages": PONG})
    else:
        r = c.post(f"{BASE}/v1/responses/input_tokens", headers=bearer(), json={"model": MODEL, "input": "pong"})
    record("count", None, rows=0)
    scan("count", r.status_code, r.headers, r.text)
    r = c.post(f"{BASE}/v1/chat/completions", headers=bearer(), json={"model": "no-such-model", "messages": PONG})
    record("chat", None, rows=0)
    scan("catalog_miss", r.status_code, r.headers, r.text)
    forged = KEY[:-6] + ("AAAAAA" if not KEY.endswith("AAAAAA") else "BBBBBB")
    r = c.post(f"{BASE}/v1/chat/completions", headers=bearer(forged), json={"model": MODEL, "messages": PONG})
    record("chat", None, rows=0)
    scan("forged", r.status_code, r.headers, r.text)
    statuses = dict(seen)
    ok = (not leaks and statuses["chat"] == 200 and statuses["stream"] == 200 and statuses["refusal"] == 400
          and statuses["models"] == 200 and statuses["count"] == 200 and statuses["forged"] == 401)
    return ok, {"leaks": leaks, "statuses": statuses}


def h2_burst():
    """REL-22: 24 concurrent streams through the gateway to the provider, which negotiates h2. All
    are served, one row each, and while they are in flight the gateway holds at most half as many
    TLS connections to the provider as requests: they share connections (multiplexed, D160), not
    one each."""
    import subprocess
    import threading
    pid = os.environ["VERIFY_GATEWAY_PID"]
    n, results, lock = 24, [], threading.Lock()
    in_flight, peak_flight = [0], [0]

    def one(i):
        c = httpx.Client(event_hooks={"response": [_hook]}, timeout=120)
        body = {"model": MODEL, "messages": [{"role": "user", "content": f"Count from 1 to 30, one number per line. ({i})"}],
                "max_completion_tokens": 512, "stream": True, "stream_options": {"include_usage": True}}
        with lock:
            in_flight[0] += 1
            peak_flight[0] = max(peak_flight[0], in_flight[0])
        try:
            with c.stream("POST", f"{BASE}/v1/chat/completions", json=body, headers=bearer()) as r:
                usage, rid = None, r.headers.get("x-beyond-request-id")
                for ev in sse(r):
                    usage = ev.get("usage") or usage
                with lock:
                    results.append((r.status_code, rid, wire_chat_usage(usage)))
        finally:
            with lock:
                in_flight[0] -= 1

    def conns():
        out = subprocess.run(["ss", "-tnpH", "state", "established", "( dport = :443 )"], capture_output=True, text=True).stdout
        return sum(1 for line in out.splitlines() if f"pid={pid}," in line)

    # One call first, the steady state: a connection to the provider is already open. Then the
    # burst, 20 ms apart, as agent traffic arrives (not all in one instant, when each request
    # would race to open its own connection before the first is back in the pool).
    one(-1)
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
        time.sleep(0.02)
    peak_conns = 0
    while any(t.is_alive() for t in threads):
        if in_flight[0] >= n // 2:
            peak_conns = max(peak_conns, conns())
        time.sleep(0.05)
    for t in threads:
        t.join()
    _ids.clear()
    for status, rid, usage in results:
        calls.append({"request_id": rid, "wire": "chat", "usage": usage})
    served = sum(1 for s, _, u in results if s == 200 and u)
    ok = served == n + 1 and 0 < peak_conns <= peak_flight[0] // 2
    return ok, {"served": served, "peak_in_flight": peak_flight[0], "peak_upstream_connections": peak_conns}


# Long outputs: ~32k output tokens through a translated path, where only short generations had
# been exercised. The task is deterministic (the integers 1..LONG_N, one per line), so the client
# can check every line arrived. Measured on claude-haiku-4-5 (2026-10-02): 1..6000 is 17,004
# output tokens in 72s, ~2.85 tokens per 4-digit line, so 1..11000 is ~32k tokens and ~135s.
# LONG_MAX fits that with ~25% headroom, under Haiku 4.5's 64k output cap. Cost per call: ~60
# input tokens plus ~32k output at $5/MTok, about $0.16; the three cells about $0.50 per run.
LONG_N = 11000
LONG_MAX = 40000
LONG_ASK = [{"role": "user", "content": (
    f"Write every integer from 1 to {LONG_N} in order, one per line, digits only. Output nothing "
    "else: no introduction, no commentary, no ellipses, never skip or abbreviate. The last line "
    f"must be {LONG_N}.")}]


class _Tail(httpx.SyncByteStream):
    """A response body stream that keeps the last bytes the client read, so a probe can see the
    raw terminal event (`data: [DONE]`) that the SDK swallows."""

    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def __iter__(self):
        for chunk in self.inner:
            self.sink[0] = (self.sink[0] + chunk)[-256:]
            yield chunk

    def close(self):
        self.inner.close()


def long_client():
    """openai-py on a transport that records each body's tail, with a read timeout past a ~135s
    non-streamed generation (the SDK's own default is 600s; `http()`'s is 120s)."""
    import openai
    tail = [b""]

    class Transport(httpx.HTTPTransport):
        def handle_request(self, request):
            resp = super().handle_request(request)
            tail[0] = b""
            resp.stream = _Tail(resp.stream, tail)
            return resp

    hc = httpx.Client(transport=Transport(), event_hooks={"response": [_hook]}, timeout=600)
    return openai.OpenAI(base_url=f"{BASE}/v1", api_key=KEY, http_client=hc, max_retries=0), tail


def long_text_problems(text):
    """Every line 1..LONG_N, in order, nothing missing at the end (a truncation) or in between."""
    lines = (text or "").strip().splitlines()
    want = [str(i) for i in range(1, LONG_N + 1)]
    if lines == want:
        return None
    first = next((i for i, (a, b) in enumerate(zip(lines, want)) if a != b), min(len(lines), len(want)))
    return {"lines": len(lines), "want": LONG_N, "first_divergence": first,
            "around": lines[max(0, first - 2):first + 2], "last": lines[-3:]}


def long_output():
    """S1 / B1 / BIL-6 on a long generation: openai-py Chat, streamed, translated to Messages on
    the Claude row. Every line arrives, in many chunks (incremental, not buffered); the stream
    ends with finish_reason "stop" (Messages end_turn), a usage chunk and the raw `data: [DONE]`
    terminator, not a cut; and the row's output tokens equal the usage shown, exactly."""
    c, tail = long_client()
    t0 = time.monotonic()
    s = c.chat.completions.create(model=MODEL, max_tokens=LONG_MAX, stream=True, messages=LONG_ASK,
                                  stream_options={"include_usage": True})
    parts, usage, finish, chunks, ttft = [], None, None, 0, None
    for chunk in s:
        usage = chunk.usage or usage
        for ch in chunk.choices:
            if ch.delta.content:
                ttft = ttft if ttft is not None else time.monotonic() - t0
                chunks += 1
                parts.append(ch.delta.content)
            finish = ch.finish_reason or finish
    u = chat_usage(usage)
    record("chat", u)
    text_bad = long_text_problems("".join(parts))
    done = tail[0].rstrip().endswith(b"data: [DONE]")
    ok = text_bad is None and finish == "stop" and done and u is not None and chunks > 100
    return ok, {"finish_reason": finish, "usage": u, "content_chunks": chunks, "done_terminator": done,
                "ttft_s": ttft, "elapsed_s": round(time.monotonic() - t0, 1), "text": text_bad,
                "tail": tail[0][-120:].decode(errors="replace")}


def long_output_responses():
    """TRN-1 / S1 / B1 / BIL-6 on a long generation: openai-py Responses, streamed, translated to
    Messages on the Claude row. Every line arrives; the last event is response.completed with
    status "completed" and no incomplete_details (Messages end_turn), the text the deltas built
    equals the final response's; the row's output tokens equal the usage shown, exactly."""
    c, _ = long_client()
    t0 = time.monotonic()
    deltas, last, n = [], None, 0
    with c.responses.stream(model=MODEL, input=LONG_ASK, max_output_tokens=LONG_MAX) as s:
        for ev in s:
            last = ev.type
            if ev.type == "response.output_text.delta":
                n += 1
                deltas.append(ev.delta)
        final = s.get_final_response()
    u = responses_usage(final.usage) if final.usage else None
    record("responses", u)
    streamed = "".join(deltas)
    text_bad = long_text_problems(streamed)
    ok = (text_bad is None and last == "response.completed" and final.status == "completed"
          and final.incomplete_details is None and final.output_text == streamed and u is not None and n > 100)
    return ok, {"last_event": last, "status": final.status, "incomplete": final.incomplete_details,
                "usage": u, "delta_events": n, "final_matches_deltas": final.output_text == streamed,
                "elapsed_s": round(time.monotonic() - t0, 1), "text": text_bad}


def long_output_nonstream():
    """B1 / BIL-6 on a long generation, non-streamed: openai-py Chat on the Claude row, one ~135s
    response held open. Every line arrives, finish_reason is "stop" (Messages end_turn), and the
    row's output tokens equal the usage shown, exactly."""
    c, _ = long_client()
    t0 = time.monotonic()
    r = c.chat.completions.create(model=MODEL, max_tokens=LONG_MAX, messages=LONG_ASK)
    u = chat_usage(r.usage)
    record("chat", u)
    text_bad = long_text_problems(r.choices[0].message.content)
    finish = r.choices[0].finish_reason
    ok = text_bad is None and finish == "stop" and u is not None
    return ok, {"finish_reason": finish, "usage": u, "elapsed_s": round(time.monotonic() - t0, 1), "text": text_bad}


PROBES = {f.__name__: f for f in [
    chat_basic, messages_basic, responses_basic, models_list, tools_chat, tools_messages, embeddings,
    langchain_chat, agents_sdk, agents_chat, responses_count_compact, count_tokens, thinking_replay, reasoning_replay,
    agents_reasoning, vision_chat, vision_messages, structured_chat, structured_messages, langchain_structured,
    reasoning_effort, typed_error, steer_providers, session_pin, big_body, stream_abort, cancel_before_head,
    stream_no_usage, prompt_cache, cache_ttl_1h, auto_cache, langchain_cache, byo_key, provider_routed,
    reasoning_metered, web_search, mid_system, developer_role, cache_control_turns, cache_control_parts,
    max_tokens_clamp, strict_tools, explicit_nulls, context_overflow, agents_handoff, langchain_agent,
    langchain_embeddings, agents_structured, thinking_no_echo, raw_models, raw_count_compact, raw_failover, raw_steer,
    raw_session_pin, raw_big_body, raw_stream_abort, byo_raw, byo_models_raw, raw_auto_cache, raw_embeddings, leak_scan, h2_burst,
    long_output, long_output_responses, long_output_nonstream]}

if __name__ == "__main__":
    name = sys.argv[1]
    try:
        ok, detail = PROBES[name]()
    except Exception as e:  # noqa: BLE001 - every failure shape is evidence
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": calls, "errors": errors, "detail": detail}, default=str))
