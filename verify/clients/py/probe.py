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
    ok = (len(o) >= 90 and len(a) == len(o) and m is not None
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
    _ids.append(resp.headers.get("x-beyond-request-id"))


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
    out, i = [], 0
    while sum(len(w) + 1 for w in out) < n_bytes:
        out.append(f"{words[i % 10]}{i % 97}")
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
    cache_control: turn 2 and later report cached tokens, and the row agrees."""
    c = openai_client()
    extra = {"prompt_cache_key": f"verify-{NONCE}"} if PROVIDER == "openai" else {}
    msgs = [{"role": "system", "content": long_prefix()}]
    reads = []
    for q in ("Which city does fact 3 name?", "And fact 4?", "And fact 5?"):
        msgs.append({"role": "user", "content": q})
        r = c.chat.completions.create(model=MODEL, max_completion_tokens=1024, messages=msgs, extra_body=extra)
        u = chat_usage(r.usage, r.service_tier)
        reads.append(u["cache_read"])
        record("chat", u, **({"row_min": {"cache_read_tokens": 1}} if len(reads) > 1 else {}))
        msgs.append({"role": "assistant", "content": r.choices[0].message.content or "ok"})
    return all(x > 0 for x in reads[1:]), {"cache_read": reads}


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
    context_length_exceeded, or Anthropic's "prompt is too long" phrasing, and bills nothing."""
    import anthropic
    import openai
    big = _filler(60_000)
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


PROBES = {f.__name__: f for f in [
    chat_basic, messages_basic, responses_basic, models_list, tools_chat, tools_messages, embeddings,
    langchain_chat, agents_sdk, responses_count_compact, count_tokens, thinking_replay, reasoning_replay,
    agents_reasoning, vision_chat, vision_messages, structured_chat, structured_messages, langchain_structured,
    reasoning_effort, typed_error, steer_providers, session_pin, big_body, stream_abort, cancel_before_head,
    stream_no_usage, prompt_cache, cache_ttl_1h, auto_cache, langchain_cache, byo_key, provider_routed,
    reasoning_metered, web_search, mid_system, developer_role, cache_control_turns, cache_control_parts,
    max_tokens_clamp, strict_tools, explicit_nulls, context_overflow, agents_handoff, langchain_agent]}

if __name__ == "__main__":
    name = sys.argv[1]
    try:
        ok, detail = PROBES[name]()
    except Exception as e:  # noqa: BLE001 - every failure shape is evidence
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": calls, "detail": detail}, default=str))
