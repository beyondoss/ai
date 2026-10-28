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

## What the agent taught us

The first run (v0.1.16, 18 tool calls) lost 8 of them to friction that was ours: `write_file` into a directory that did not exist (`sh: 23: cannot create wordstats/.tmp.…: Directory nonexistent`), a whole command line in `exec`'s `command` (`pytest -q` answered `No such file or directory (os error 2)`, four times), and `job_output` refusing `encoding: "utf8"` that `exec` itself reports. All three are fixed in the tools (`write_file` makes parents, `exec` and `job_start` take one `sh -c` command line, one `utf8`/`base64` vocabulary) and in `execd` (a missing program is `execd: pytest -q: command not found`). Same model, same prompt, on local builds:

|                               | before (v0.1.16) | after                                                        |
| ----------------------------- | ---------------- | ------------------------------------------------------------ |
| tool calls                    | 18               | 14                                                           |
| tool errors                   | 2                | 0                                                            |
| commands that exited non-zero | 6                | 2 (pytest not installed in the VM, then a real import error) |
| cost                          | $0.0086          | $0.0072                                                      |

`SANDBOX_MCP_BIN` and `EXECD_BIN` point `run.sh` at local builds instead of the release.
