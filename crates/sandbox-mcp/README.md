# sandbox-mcp

Give any MCP agent a sandbox: run commands, read, write, edit and search files, and run background jobs, in a sandbox
that speaks the [exec protocol](../execd/README.md) (`execd`, or anything that implements one-shot or later).

```json
{
  "mcpServers": {
    "sandbox": {
      "command": "sandbox-mcp",
      "env": { "SANDBOX_URL": "http://127.0.0.1:8080", "SANDBOX_TOKEN": "..." }
    }
  }
}
```

Paste that into `.mcp.json` (Claude Code), `claude_desktop_config.json`, or any client's MCP config. Install the binary with
`curl -fsSL https://github.com/beyondoss/ai/releases/latest/download/install-sandbox-mcp.sh | sh`, or
`cargo install --path crates/sandbox-mcp`.

It passes the shared conformance matrix (`mise run test:clients`) with 0 failures on every backend, through a real MCP
session: 463 checks pass on a plain `execd`, and 13 are reported `unsupported` (not skipped): all of them need the raw
exec stream, which an MCP tool result cannot carry (see below).

## Tools

| tool         | what it does                                                                                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `exec`       | run a program and wait: `command`, `args`, `cwd`, `timeout_ms`, `stdin`. Always present: the escape hatch                                                                                                    |
| `read_file`  | bytes of a file from `offset`, at most `max` (default 128 KiB); text, or base64 for binary                                                                                                                   |
| `write_file` | create or replace a file atomically, from `content` or `content_base64`; keeps an existing file's permissions; `if_unchanged_since` (the `stat` mtime) or `create_only` refuse a lost update with `conflict` |
| `edit_file`  | replace `old_str` (must match once, or `replace_all`) with `new_str`; never overwrites a concurrent change                                                                                                   |
| `list_dir`   | entries with name, kind and size; `include_hidden`, `limit`                                                                                                                                                  |
| `stat`       | exists, kind, size, mtime, writable                                                                                                                                                                          |
| `search`     | regular-expression search under a directory, with `context`, `glob`, `literal`, `ignore_case`, `limit`                                                                                                       |
| `mkdir`      | create a directory and its parents                                                                                                                                                                           |
| `job_start`  | start a command that outlives the call; the sandbox keeps it and its output                                                                                                                                  |
| `job_list`   | jobs and their state: `running`, `exited`, `ended`, `lost`                                                                                                                                                   |
| `job_output` | a job's output from an offset; follow it by passing back `next_offset`                                                                                                                                       |
| `job_kill`   | SIGTERM, a grace, SIGKILL; `remove` also deletes the job                                                                                                                                                     |

A failed call is a tool result with `isError: true` and `structuredContent.error = {kind, detail}`, never a protocol error.
`kind` is one of `not_found`, `is_directory`, `not_a_directory`, `permission_denied`, `io`, `invalid_query`, `conflict`
(`edit_file`: the file changed after it was read; nothing was written), `too_large`, `auth`, `timeout`, `transport`,
`protocol`, `endpoint`, `not_offered`, `invalid_request`. An unreachable sandbox or a wrong token does not stop the server:
the tools are listed, and each call says `transport` or `auth`.

The resource `sandbox://capabilities` reports what the connected sandbox has: `search_engine` (`rg` honours `.gitignore`; `grep` does not), `find_printf`, `read_window`.

## Configuration

Environment variables only, as `execd` does; there are no flags.

| variable                       | meaning                                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `SANDBOX_URL`                  | the sandbox's exec endpoint (required)                                                                |
| `SANDBOX_TOKEN`                | the bearer token                                                                                      |
| `SANDBOX_TOKEN_CMD`            | a shell command whose stdout is a fresh token; run again once after a `401`. Not with `SANDBOX_TOKEN` |
| `SANDBOX_JOBS_SCOPE`           | the name jobs are kept under (default `mcp`); a later server with the same scope sees them            |
| `SANDBOX_MCP_LISTEN`           | `127.0.0.1:PORT`: serve streamable HTTP at `/mcp` instead of stdio                                    |
| `SANDBOX_MCP_INSECURE_NO_AUTH` | allow a non-loopback `SANDBOX_MCP_LISTEN`; the server then checks nobody (default 0)                  |
| `SANDBOX_CAPS`                 | JSON that overrides the probe (`find_printf`, `search_engine`), to test a search rung                 |

The server holds the sandbox's token, so over HTTP it listens on loopback only unless told otherwise, and rejects a `Host`
that is not a loopback name.

## What is not offered

`exec` returns the whole output at once: there is no streaming tool, because an MCP tool result is not streamed. Use
`job_start` and `job_output` for anything long or noisy. So the conformance matrix's `exec_stream` op (frames, signals,
heartbeats, a timeout that is a frame) is `unsupported`: the 13 checks that use it are the only ones.

To write without losing a concurrent change, pass `write_file` the `mtime` that `stat` returned, as `if_unchanged_since`
(fractional seconds, exactly as `stat` gave it): a file changed in the meantime, even within the same second, or removed,
fails with `conflict` and nothing is written. `create_only` is the same guard for "the file must not exist".

More: [ARCHITECTURE.md](ARCHITECTURE.md), the file helpers' spec
[FILES-OVER-EXEC.md](../../sdk/conformance/FILES-OVER-EXEC.md).
