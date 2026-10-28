"""An outside agent (OpenAI Agents SDK) given only sandbox-mcp's tools. Writes transcript.jsonl and summary.json."""
import asyncio, json, os, sys, time
from agents import Agent, Runner, RunHooks
from agents.mcp import MCPServerStdio

PROMPT = ("In the sandbox, create a small Python package `wordstats` with a function that counts word frequencies in a text file. "
          "Add a pytest test, run the tests until they pass, then start a background job that runs the tests in a loop 3 times, "
          "read its output, and report the final test result and the file tree.")
MODEL = os.environ.get("MODEL", "gpt-5-mini")
# USD per 1M tokens (input, cached input, output); gpt-5-mini list prices, override for other models.
P_IN, P_CACHED, P_OUT = (float(os.environ.get(k, d)) for k, d in
                         (("PRICE_IN", "0.25"), ("PRICE_CACHED", "0.025"), ("PRICE_OUT", "2.00")))
CAP = float(os.environ.get("COST_CAP_USD", "0.25"))
OUT = os.environ["OUT_DIR"]


def cost(u):
    cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
    return ((u.input_tokens - cached) * P_IN + cached * P_CACHED + u.output_tokens * P_OUT) / 1e6


class Hooks(RunHooks):
    def __init__(self):
        self.t0 = {}
        self.rows = []

    async def on_tool_start(self, ctx, agent, tool):
        self.t0[ctx.tool_call_id] = time.monotonic()

    async def on_tool_end(self, ctx, agent, tool, result):
        d = int((time.monotonic() - self.t0.pop(ctx.tool_call_id, time.monotonic())) * 1000)
        # An MCP tool result reaches the SDK as text: a failed call carries {"error":{kind,detail}}; a command that ran
        # and failed carries a nonzero exit_code. Keep the two apart.
        import re
        t = str(result)
        m = re.search(r'\\?"exit_code\\?":(-?\d+)', t)
        kind = "tool_error" if re.search(r'\\?"error\\?":\{', t) else ("nonzero_exit" if m and m.group(1) != "0" else "ok")
        self.rows.append({"tool": tool.name, "args": (ctx.tool_arguments or "")[:300], "result_kind": kind,
                          "result": str(result)[:300], "ms": d})
        with open(f"{OUT}/transcript.jsonl", "a") as f:
            f.write(json.dumps(self.rows[-1]) + "\n")

    async def on_llm_end(self, ctx, agent, response):
        c = cost(ctx.usage)
        if c > CAP:
            raise RuntimeError(f"cost cap hit: ${c:.4f} > ${CAP}")


async def main():
    env = {k: os.environ[k] for k in ("SANDBOX_URL", "SANDBOX_TOKEN", "PATH")}
    async with MCPServerStdio(params={"command": os.environ["SANDBOX_MCP_BIN"], "env": env},
                              client_session_timeout_seconds=120, cache_tools_list=True) as srv:
        agent = Agent(name="coder", model=MODEL, mcp_servers=[srv],
                      instructions="You are a software engineer working in a sandbox. Use your tools to do the task.")
        hooks = Hooks()
        err = None
        try:
            res = await Runner.run(agent, PROMPT, hooks=hooks, max_turns=int(os.environ.get("MAX_TURNS", "40")))
            final, u = res.final_output, res.context_wrapper.usage
        except Exception as e:  # cap or max_turns: still record what happened
            err, final, u = repr(e), None, None
            u = getattr(getattr(e, "run_data", None), "context_wrapper", None) and e.run_data.context_wrapper.usage
    counts = {}
    for r in hooks.rows:
        counts[r["tool"]] = counts.get(r["tool"], 0) + 1
    summary = {"model": MODEL, "tool_calls": len(hooks.rows), "by_tool": counts,
               "results": {k: sum(r["result_kind"] == k for r in hooks.rows) for k in ("ok", "tool_error", "nonzero_exit")},
               "tokens": {"input": u.input_tokens, "output": u.output_tokens,
                          "cached": getattr(u.input_tokens_details, "cached_tokens", 0), "requests": u.requests} if u else None,
               "cost_usd": round(cost(u), 5) if u else None, "run_error": err, "final_output": final}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "final_output"}, indent=1))
    print("--- agent's final report ---\n" + str(final))

asyncio.run(main())
