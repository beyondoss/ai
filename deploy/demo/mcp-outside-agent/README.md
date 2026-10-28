# An outside agent on sandbox-mcp

Shows that an agent framework we did not write (the OpenAI Agents SDK for Python) can do a real coding task in a
sandbox using only [`sandbox-mcp`](../../../crates/sandbox-mcp)'s tools: no local shell, no local files.

```sh
E2B_API_KEY=... OPENAI_API_KEY=... ./run.sh          # MODEL=gpt-5-mini COST_CAP_USD=0.25 VERSION=v0.1.16
```

`run.sh` downloads the released `execd` and `sandbox-mcp` (sha256-checked), starts one E2B VM (hard timeout 20 minutes) running
`execd` with a bearer, runs `agent.py` (the SDK's `MCPServerStdio` launching `sandbox-mcp` with `SANDBOX_URL` and
`SANDBOX_TOKEN`), then checks the result itself with a raw `POST /exec` and kills the VM, printing how many sandboxes of that
id are still running (0). Keys come only from the environment and are never written.

The agent gets one prompt: create a `wordstats` package with a word-frequency function, add a pytest test, run it until it
passes, start a background job that runs the tests 3 times, read its output, report the result and the file tree.

Outputs, in `$WORK/out`: `transcript.jsonl` (per tool call: name, arguments truncated to 300 chars, result kind `ok`,
`tool_error` or `nonzero_exit`, duration), `summary.json` (calls by tool, token totals, cost from list prices, settable with
`PRICE_IN`, `PRICE_CACHED`, `PRICE_OUT` per 1M tokens for other models), `verify.log`. `example-*` are the outputs of one run.
The run aborts if its estimated cost passes `COST_CAP_USD`.
