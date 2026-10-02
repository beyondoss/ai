"""Live session probes (SES-*): one stock-SDK conversation over many turns through the gateway.

Usage: session_probe.py <probe>, with probe.py's environment plus:
  VERIFY_PRIMARY        the provider that serves until it dies (SES-1), or that serves throughout
  VERIFY_PRIMARY_TURNS  requests the primary serves before it dies (its proxy exits after them)
  VERIFY_FALLBACK       the provider every later request must land on
  VERIFY_MODELS         SES-2: the models to alternate, comma-separated, each `model=provider`

Prints `VERIFY {json}` exactly like probe.py: every HTTP call is one entry in `calls`, and each one
names the provider that must serve it, so the Rust trial holds every turn to one ledger row on that
provider. Assertions are on structure (a tool call where one was asked for, the codename and the
tool's values recalled), never on wording.
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe as p  # noqa: E402

MODEL = p.MODEL
PRIMARY = os.environ.get("VERIFY_PRIMARY", "")
FALLBACK = os.environ.get("VERIFY_FALLBACK", "")
PRIMARY_TURNS = int(os.environ.get("VERIFY_PRIMARY_TURNS", "1000"))
# A per-run codename the model can only know from turn 1.
CODE = "ZEPHYR-" + p.NONCE[:4].upper()
TEMPS = {"Paris": "21", "Tokyo": "27", "Berlin": "14"}
TOOL_MSG = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TOOL_ANT = {"name": "get_weather", "description": "Current weather for a city",
            "input_schema": TOOL_MSG["function"]["parameters"]}
TOOL_RESP = {"type": "function", "name": "get_weather", "description": "Current weather for a city",
             "parameters": TOOL_MSG["function"]["parameters"]}
SYSTEM = ("You are a weather assistant. Whenever the user asks about the weather in a city, call get_weather; "
          "never answer weather from memory. Keep answers to one short sentence.")


def weather(city):
    city = city.strip().title()
    return f"{city}: sunny, {TEMPS.get(city, '18')}C"


def serving(i):
    """The provider that must serve request i (0-based): the primary until it dies, then the fallback."""
    return PRIMARY if i < PRIMARY_TURNS else FALLBACK


def ask(city, first=False):
    lead = f"Our project codename is {CODE}; remember it. " if first else ""
    return f"{lead}What's the weather in {city} right now? Use the get_weather tool."


FINAL = "Then, on a new line, write our project codename from the start of this conversation."


class Miss(Exception):
    """The model didn't do what the turn asked (no tool call, no answer): a failed verdict."""


# A user turn's tool loop: answer every tool call and ask again, until the model answers in text.
# Models sometimes call the tool twice in one turn; that is a valid agent loop, not a failure.
MAX_STEPS = 4


def messages_turn(c, log, msgs, model, provider_for, kw, tool=False, final=None, mark=False):
    """One user turn over Messages. `final` rides along with the first tool results; `mark` puts a
    cache_control breakpoint on the first tool_result (so a cache marker sits in replayed history)."""
    for step in range(MAX_STEPS):
        i = len(log)
        raw = c.messages.with_raw_response.create(model=model, messages=msgs, **kw)
        r = raw.parse()
        p.record("messages", p.messages_usage(r.usage), provider=provider_for(i))
        tus = [b for b in r.content if b.type == "tool_use"]
        log.append({"req": i + 1, "model": model, "served": raw.headers.get("x-beyond-provider"),
                    "types": [b.type for b in r.content],
                    "signed": any(b.type == "thinking" and b.signature for b in r.content),
                    "text": p.text_of(r.content)[:160]})
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
        if not tus:
            if tool and step == 0:
                raise Miss(f"request {i + 1} ({model}): no tool_use")
            return p.text_of(r.content)
        content = [{"type": "tool_result", "tool_use_id": tu.id, "content": weather(tu.input.get("city", ""))}
                   for tu in tus]
        if mark and step == 0:
            content[0]["cache_control"] = {"type": "ephemeral"}
        if final and step == 0:
            content.append({"type": "text", "text": final})
        msgs.append({"role": "user", "content": content})
    raise Miss(f"no answer after {MAX_STEPS} requests")


def chat_turn(c, log, msgs, model, provider_for, kw, tool=False, final=None):
    """One user turn over Chat Completions; `final` is a user message after the first tool results."""
    for step in range(MAX_STEPS):
        i = len(log)
        raw = c.chat.completions.with_raw_response.create(model=model, messages=msgs, **kw)
        r = raw.parse()
        p.record("chat", p.chat_usage(r.usage), provider=provider_for(i))
        m = r.choices[0].message
        log.append({"req": i + 1, "model": model, "served": raw.headers.get("x-beyond-provider"),
                    "finish": r.choices[0].finish_reason, "tool_calls": len(m.tool_calls or []),
                    "text": (m.content or "")[:160]})
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            if tool and step == 0:
                raise Miss(f"request {i + 1} ({model}): no tool call")
            return m.content or ""
        for tc in m.tool_calls:
            arg = json.loads(tc.function.arguments or "{}").get("city", "")
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": weather(arg)})
        if final and step == 0:
            msgs.append({"role": "user", "content": final})
    raise Miss(f"no answer after {MAX_STEPS} requests")


# --- SES-1: the primary dies between turns ---------------------------------------------------------

def _failover(turn_fn, msgs, kw, **extra):
    """Paris, Tokyo, Berlin, one tool-using user turn each, on one model. The primary serves
    requests 1..k and dies (k = 3: Paris's loop and Tokyo's tool call), so the rest replay the
    whole history onto the fallback; the last answer must also recall the codename from turn 1."""
    log = []
    try:
        for turn, city in enumerate(["Paris", "Tokyo", "Berlin"]):
            msgs.append({"role": "user", "content": ask(city, first=turn == 0)})
            text = turn_fn(log, msgs, MODEL, serving, kw, tool=True, final=FINAL if turn == 2 else None,
                           **({"mark": turn == 1} if extra.get("mark") else {}))
            if TEMPS[city] not in text:
                raise Miss(f"turn {turn + 1}: the answer lacks {city}'s tool value")
            if turn == 2 and CODE not in text:
                raise Miss("the last turn doesn't recall the codename")
    except Miss as e:
        return False, {"why": str(e), "log": log}
    served = [e["served"] for e in log]
    want = [serving(i) for i in range(len(log))]
    return served == want, {"log": log, "want": want}


def failover_messages():
    """SES-1 over Messages: thinking + tools + cache_control (the system prompt, and a tool_result
    that later turns replay); after the primary dies the fallback gets signed thinking blocks and
    tool_use / tool_result ids it never issued."""
    c = p.anthropic_client()
    system = [{"type": "text", "text": p.long_prefix() + "\n\n" + SYSTEM, "cache_control": {"type": "ephemeral"}}]
    kw = dict(max_tokens=3000, thinking={"type": "enabled", "budget_tokens": 1024}, tools=[TOOL_ANT], system=system)
    return _failover(lambda *a, **k: messages_turn(c, *a, **k), [], kw, mark=True)


def failover_chat():
    """SES-1 over Chat Completions on a Claude row: tools, reasoning_effort, part-level
    cache_control on the system prompt."""
    c = p.openai_client()
    system = {"role": "system", "content": [{"type": "text", "text": p.long_prefix() + "\n\n" + SYSTEM,
                                             "cache_control": {"type": "ephemeral"}}]}
    kw = dict(max_completion_tokens=3000, tools=[TOOL_MSG], reasoning_effort="low")
    return _failover(lambda *a, **k: chat_turn(c, *a, **k), [system], kw)


# --- SES-2: the user switches models between turns -------------------------------------------------

def plan():
    """`VERIFY_MODELS` = `claude-haiku-4-5=anthropic,gpt-5.1=openai`: user turn t uses entry t mod n."""
    return [tuple(x.split("=")) for x in os.environ["VERIFY_MODELS"].split(",")]


SWITCH_TURNS = [
    ("tool", "Paris", None),
    ("tool", "Tokyo", None),
    ("recall-cities", None, "Which cities have I asked about so far, and what temperature did the tool report for "
                            "each? Answer from our conversation, in one line."),
    ("recall-code", None, "What is our project codename from the start of this conversation? Reply with only the "
                          "codename."),
]


def _switch(turn_fn, msgs, kw_for):
    """Four user turns, the model alternating each turn (so the history crosses dialects every time):
    a tool loop on each of the first two, then two turns answerable only from the history."""
    models, log = plan(), []
    try:
        for t, (kind, city, q) in enumerate(SWITCH_TURNS):
            model, provider = models[t % len(models)]
            msgs.append({"role": "user", "content": ask(city, first=t == 0) if city else q})
            text = turn_fn(log, msgs, model, lambda _i, pr=provider: pr, kw_for(kind), tool=kind == "tool")
            ok = {"tool": lambda: TEMPS.get(city, "?") in text,
                  "recall-cities": lambda: all(x in text for x in ("Paris", "Tokyo", "21", "27")),
                  "recall-code": lambda: CODE in text}[kind]()
            if not ok:
                raise Miss(f"turn {t + 1} ({model}, {kind}) missed what it should know")
    except Miss as e:
        return False, {"why": str(e), "log": log}
    return True, {"log": log}


def switch_chat():
    """SES-2 over Chat: tools on the first two turns, reasoning_effort on the last two."""
    c = p.openai_client()
    kw = lambda kind: dict(max_completion_tokens=3000, tools=[TOOL_MSG],
                           **({} if kind == "tool" else {"reasoning_effort": "low"}))
    return _switch(lambda *a, **k: chat_turn(c, *a, **k), [{"role": "system", "content": SYSTEM}], kw)


def switch_messages():
    """SES-2 over Messages with extended thinking on every turn: each Claude turn replays its own
    signed thinking beside GPT turns that have none, and GPT turns get Claude's thinking blocks."""
    c = p.anthropic_client()
    kw = lambda _kind: dict(max_tokens=3000, thinking={"type": "enabled", "budget_tokens": 1024}, tools=[TOOL_ANT],
                            system=SYSTEM)
    return _switch(lambda *a, **k: messages_turn(c, *a, **k), [], kw)


# --- SES-3: stateful Responses -----------------------------------------------------------------

def responses_chain():
    """SES-3: five turns chained only by previous_response_id (store left at its default, true):
    no turn resends history. A tool call and its output cross a turn boundary; turn 5 recalls the
    code word from turn 1 and the tool's value from turn 3."""
    c = p.openai_client()
    log, prev = [], None

    def turn(inp, **kw):
        nonlocal prev
        r = c.responses.create(model=MODEL, input=inp, previous_response_id=prev, tools=[TOOL_RESP],
                               instructions=SYSTEM, max_output_tokens=3000, reasoning={"effort": "low"}, **kw)
        p.record("responses", p.responses_usage(r.usage, r.service_tier), provider=PRIMARY or None)
        log.append({"id": r.id[:16], "status": r.status, "types": [i.type for i in r.output], "text": r.output_text[:160]})
        prev = r.id
        return r

    turn(f"Our project codename is {CODE}; remember it. Reply with just OK.", tool_choice="none")
    r = turn(ask("Paris"))
    fc = next((i for i in r.output if i.type == "function_call"), None)
    if fc is None:
        return False, {"why": "turn 2: no function_call", "log": log}
    r = turn([{"type": "function_call_output", "call_id": fc.call_id, "output": weather("Paris")}])
    if TEMPS["Paris"] not in r.output_text:
        return False, {"why": "turn 3: the answer lacks the tool's value", "log": log}
    turn("What is 17 + 25? Reply with just the number.", tool_choice="none")
    # tool_choice none: a recall turn the model answers from the chain, not by calling the tool again.
    r = turn("What is our project codename, and what temperature did the tool report for Paris? One line.",
             tool_choice="none")
    ok = CODE in r.output_text and TEMPS["Paris"] in r.output_text and "42" in log[3]["text"]
    return ok, {"log": log}


def responses_failover():
    """SES-3 failover within the Responses arm. A GPT row's Responses arm has one candidate (OpenAI's
    store is not OpenRouter's), and every inbound Responses request on such a row walks only that
    arm (D74's decision: "a failover stays inside the arm"). So when OpenAI dies after turn 1 the
    documented behavior is a clear error, never a hollow answer from another vendor: for the chained
    turn (previous_response_id), and for a store:false one-shot on the same row too."""
    import openai
    c = p.openai_client()
    r = c.responses.create(model=MODEL, input=f"Our project codename is {CODE}. Reply with just OK.",
                           max_output_tokens=1024, reasoning={"effort": "low"})
    p.record("responses", p.responses_usage(r.usage, r.service_tier), provider=PRIMARY)
    detail, clear = {"turn1": r.status}, []
    for name, kw in (("chained", {"input": "What is our project codename?", "previous_response_id": r.id}),
                     ("one_shot", {"input": "Reply with the single word: pong", "store": False})):
        try:
            r2 = c.responses.create(model=MODEL, max_output_tokens=1024, **kw)
            p.record("responses", p.responses_usage(r2.usage, r2.service_tier), provider=PRIMARY)
            detail[name] = {"why": "served with the arm's only upstream dead",
                            "served": r2.model, "text": r2.output_text[:160]}
            clear.append(False)
        except openai.APIStatusError as e:
            detail[name] = {"status": e.status_code, "body": e.body}
            # Nothing generated: one row, no tokens, on the arm's provider (never the fallback).
            p.record("responses", None, error=True, provider=PRIMARY)
            body = e.body if isinstance(e.body, dict) else {}
            msg = (body.get("error") if isinstance(body.get("error"), dict) else body).get("message")
            clear.append(e.status_code in (502, 503, 504) and bool(msg))
    return all(clear), detail


def agents_chain():
    """SES-3 via the OpenAI Agents SDK: three Runner.run calls carried only by previous_response_id
    (each run sends just its new input); the last recalls the codename and both tool results."""
    import asyncio
    from agents import Agent, Runner, function_tool
    p._agents_client()

    @function_tool
    def get_weather(city: str) -> str:
        """Current weather for a city."""
        return weather(city)

    agent = Agent(name="weather", instructions=SYSTEM, model=MODEL, tools=[get_weather])

    async def go():
        runs, prev = [], None
        for q in (ask("Paris", first=True), ask("Tokyo"),
                  "What is our project codename, and what temperatures did the tool report? One line."):
            res = await Runner.run(agent, q, previous_response_id=prev)
            runs.append(res)
            prev = res.last_response_id
        return runs

    runs = asyncio.run(go())
    for res in runs:
        p._agents_record(res)
    finals = [str(r.final_output) for r in runs]
    tools = [sum(getattr(i, "type", "") == "tool_call_item" for i in r.new_items) for r in runs]
    ok = tools[0] >= 1 and tools[1] >= 1 and CODE in finals[2] and "21" in finals[2] and "27" in finals[2]
    return ok, {"finals": [f[:160] for f in finals], "tool_calls": tools}


PROBES = {f.__name__: f for f in [failover_messages, failover_chat, switch_chat, switch_messages, responses_chain,
                                  responses_failover, agents_chain]}

if __name__ == "__main__":
    name = sys.argv[1]
    try:
        ok, detail = PROBES[name]()
    except Exception as e:  # noqa: BLE001 - every failure shape is evidence
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": p.calls, "detail": detail}, default=str))
