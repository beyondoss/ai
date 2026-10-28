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
session, with no check reported `unsupported`: `exec_stream` runs as `job_start` plus `job_output`, and `extract` runs the
real `download` tool against a private `execd` that serves the archive.

## Tools

| tool         | what it does                                                                                                                                                                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `exec`       | run a shell command line (`sh -c`) and wait: `command`, `cwd`, `timeout_ms`, `stdin`. Always present: the escape hatch                                                                                                                                                                     |
| `read_file`  | bytes of a file from `offset`, at most `max` (default 128 KiB); `encoding` `utf8` (default; base64 when the bytes are not UTF-8) or `base64`                                                                                                                                               |
| `write_file` | create or replace a file atomically, from `content` or `content_base64`; creates missing parent directories (`create_parents: false` to fail instead); keeps an existing file's permissions; `if_unchanged_since` (the `stat` mtime) or `create_only` refuse a lost update with `conflict` |
| `edit_file`  | replace `old_str` (must match once, or `replace_all`) with `new_str`; never overwrites a concurrent change                                                                                                                                                                                 |
| `list_dir`   | entries with name, kind and size; `include_hidden`, `limit`                                                                                                                                                                                                                                |
| `stat`       | exists, kind, size, mtime, writable                                                                                                                                                                                                                                                        |
| `search`     | regular-expression search under a directory, with `context`, `glob`, `literal`, `ignore_case`, `limit`                                                                                                                                                                                     |
| `mkdir`      | create a directory and its parents                                                                                                                                                                                                                                                         |
| `upload`     | copy a directory or file of this machine into a sandbox directory as tar (needs `SANDBOX_LOCAL_ROOT`)                                                                                                                                                                                      |
| `download`   | copy a sandbox directory or file to a directory of this machine, never writing through a link (same)                                                                                                                                                                                       |
| `job_start`  | start a shell command line that outlives the call; the sandbox keeps it and its output                                                                                                                                                                                                     |
| `job_list`   | jobs and their state: `running`, `exited`, `ended`, `lost`                                                                                                                                                                                                                                 |
| `job_output` | a job's output from an offset; follow it by passing back `next_offset`; `encoding` as for `read_file`                                                                                                                                                                                      |
| `job_kill`   | SIGTERM, a grace, SIGKILL; `remove` also deletes the job                                                                                                                                                                                                                                   |

`exec` and `job_start` take the one string you would type at a prompt, run by `sh -c`: pipes, `&&`, redirects and quotes work, and
there is no `args`. The schema is closed (`additionalProperties: false`), so a call that still sends `args` is refused by name. A
command that is not found exits 127 with the shell's `sh: 1: pytest: not found`. One encoding vocabulary runs through every
tool: results say `utf8` or `base64` (`stdout_encoding`, `encoding`), and `encoding` arguments take exactly those two words.

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
| `SANDBOX_LOCAL_ROOT`           | an absolute directory of this machine; turns on `upload` and `download`, confined to it               |
| `SANDBOX_CAPS`                 | JSON that overrides the probe (`find_printf`, `search_engine`), to test a search rung                 |

The server holds the sandbox's token, so over HTTP it listens on loopback only unless told otherwise, and rejects a `Host`
that is not a loopback name.

## What is not offered

`exec` returns the whole output at once: there is no streaming tool, because an MCP tool result is not streamed. Use
`job_start` and `job_output` for anything long or noisy. A job keeps the last 8 MiB of its output, so a command that writes more than that
faster than it is read loses the head (the one `xfail` of this driver).

To write without losing a concurrent change, pass `write_file` the `mtime` that `stat` returned, as `if_unchanged_since`
(fractional seconds, exactly as `stat` gave it): a file changed in the meantime, even within the same second, or removed,
fails with `conflict` and nothing is written. `create_only` is the same guard for "the file must not exist".

More: [ARCHITECTURE.md](ARCHITECTURE.md), the file helpers' spec
[FILES-OVER-EXEC.md](../../sdk/conformance/FILES-OVER-EXEC.md).
