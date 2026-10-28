# beyond-sandbox-mcp architecture

An MCP server over one sandbox. It is a thin mapping from MCP tool calls to `beyond-sandbox` (raw exec, the file helpers,
jobs) and back; the behaviour behind every tool is specified in `sdk/conformance/FILES-OVER-EXEC.md`, not here.

## Modules

| file                            | what                                                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `src/lib.rs`                    | the library root: `Config`, `SandboxMcp`, and the modules below                                                            |
| `src/main.rs`                   | reads the environment, serves stdio or streamable HTTP, exits on SIGTERM / SIGINT / stdin closing                          |
| `src/config.rs`                 | `Config::from_env`: every variable parsed in one place, errors name the variable                                           |
| `src/server.rs`                 | `SandboxMcp`: the twelve tools, the `sandbox://capabilities` resource, the lazily made connection                          |
| `src/error.rs`                  | `ToolError` and `classify`: the spec's error kinds, as a tool result                                                       |
| `src/http.rs`                   | the streamable-HTTP listener (loopback unless told otherwise)                                                              |
| `src/bin/sandbox-driver-mcp.rs` | the conformance driver: DRIVER.md on stdin/stdout, every op a tool call over a real MCP session to a spawned `sandbox-mcp` |
| `tests/tools_snapshot.rs`       | pins the tools' names, descriptions and input schemas (`tests/tools.snapshot.json`)                                        |
| `tests/e2e.rs`                  | a generic MCP client driving the real binary: list tools, write, `cat`, read back; errors; jobs; HTTP; token command       |

## One connection, made on first use

`SandboxMcp` holds `Option<Target>` behind an async mutex. The first tool call (or capabilities read) builds the
`HttpExec`, probes the sandbox through `Sandbox::over`, and keeps the result; a failed attempt is not kept, so the next call
retries. The server therefore starts, answers `initialize` and `tools/list`, and reports an unreachable sandbox or a
wrong token as that call's `transport` or `auth` error, instead of exiting before the agent can read why.

`SANDBOX_TOKEN_CMD` becomes `TokenSource::command(["sh", "-c", cmd])`: the client library runs it before the first request
and once more after a `401`. That is the whole of the server's auth logic. Streamable HTTP shares the one `Target` across
MCP sessions; there is no per-session state, because the sandbox holds the state (files, jobs).

## Tools

`exec`, `read_file`, `write_file`, `edit_file`, `list_dir`, `stat`, `search`, `mkdir`, `job_start`, `job_list`,
`job_output`, `job_kill`. Each is one `do_*` method returning `Result<Value, ToolError>`; `reply` turns it into
`CallToolResult::structured` (`isError: false`) or `structured_error` with `{"error": {"kind", "detail"}}`.

- **Errors are results.** `kind` is the DRIVER.md vocabulary. Arguments that do not fit a tool's schema are caught in
  `call_tool` (rmcp rejects them as `invalid_params`) and re-issued as `invalid_request` results, so the agent sees and
  fixes them.
- **`exec` and `job_start` take a command line, not an argv.** Models call a bash tool with one string (`cd app && pytest -q`);
  given `command` and `args` they put the whole line in `command` and get exit 127. So `command` is the line, run as
  `sh -c <line>` (`do_exec`, `do_job_start`), and the args structs are `deny_unknown_fields`: a stray `args` is refused as
  `invalid_request: unknown field`args``, and the schema is a closed object. Raw argv exec stays one layer down, in the client
  library; the conformance driver quotes each argv word into a line.
- **One encoding vocabulary.** Every result that carries bytes says `encoding` (or `stdout_encoding` / `stderr_encoding`) as `utf8`
  or `base64`, and every `encoding` argument accepts exactly those. `read_file` and `job_output` return text when the bytes are
  UTF-8 and base64 otherwise (`encoding` says which; `encoding: "base64"` forces it). A window that ends inside a multi-byte character gives that character up and
  reports `bytes` / `next_offset` for what it returned, so a client that follows `next_offset` never sees a split
  character. `exec` output is text: the client library already decodes it.
- **`edit_file`** stats the file, reads it whole, requires exactly one match (or `replace_all`), and writes with
  `write_if_unchanged` on the mtime it stat'ed. A false return is `conflict`; nothing is written.
- **`write_file` makes missing parents.** The first attempt is the plain write; when it fails with the spec's
  `parent directory DIR does not exist` (FILES-OVER-EXEC.md section 5.3) the server runs `create_dir_all(parent)` and retries once, so
  a write into an existing directory costs no extra command. `create_parents: false` skips the retry (the conformance driver
  sets it, because the contract's `write` does not make parents) and the caller gets `not_found` with that same plain detail.
- **`write_file`** is unconditional unless a guard is given. `if_unchanged_since` (an `f64` of epoch seconds, the `stat`
  tool's `mtime`) and `create_only` (the file must not exist) both go to `write_if_unchanged`, which compares mtimes as
  the doubles `stat` reports, so a same-second change is still a change. A false return is `conflict`; the two guards
  together, or a negative or non-finite mtime, are `invalid_request`. Every mtime the server returns comes from `stat`
  (`epoch_f64`, nanosecond fraction kept); `read_file` returns none.
- **Jobs** are kept jobs (`start_kept`): the sandbox owns them, so they outlive the MCP server and are found again by the
  next one with the same `SANDBOX_JOBS_SCOPE`. `scope` and `id` are optional arguments only so a harness can pick them.

Defaults that bound a response: `read_file` / `job_output` 128 KiB (`READ_MAX`, the size the exec client already
reads in), `list_dir` 1000 entries, `search` 100 hits. Each is a default for the argument, not a ceiling.

## Capabilities

`sandbox://capabilities` (JSON): `search_engine`, `find_printf`,
`read_window`: what the one probe found, which tells an agent whether `search` honours `.gitignore` (`rg`). A resource rather than a tool because it describes the connection, not an action. When the sandbox
cannot be reached the content is `{"error": {...}}`.

`SANDBOX_CAPS` overrides the probe results the same way the conformance harness's `CONFORMANCE_CAPS` does, so one sandbox
can walk every rung of the file helpers' fallback ladder. `override_caps` builds a second `ShellFs` with the changed
`Capabilities` over the same exec.

## Transports

stdio by default. `SANDBOX_MCP_LISTEN=host:port` serves streamable HTTP at `/mcp` (rmcp's `StreamableHttpService` under
hyper). The choice is an environment variable, not a flag, as in execd: the server is started from an MCP client's `env`
block, and flags would be a second place to say the same thing. The server holds the sandbox's token, so a listener that is
not loopback is refused unless `SANDBOX_MCP_INSECURE_NO_AUTH=1` (an authenticating proxy is then the caller's job), and
rmcp's default `Host` check (loopback names only) stays on for a loopback listener, which stops DNS rebinding from a
browser.

## The conformance driver

`sandbox-driver-mcp` lives here, not in `crates/sandbox-conformance`: it needs rmcp's client and the `sandbox-mcp` binary
built beside it, and the harness crate should not depend on the thing it judges. It reports `driver: "sandbox-mcp"` in
`hello` and implements each op by calling a tool; `CONFORMANCE_*` is translated to `SANDBOX_*` for the child it spawns.

| op                                                                                      | how                                                                                 |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `exec`, `stat`, `mkdir`, `write`, `list`, `search`, `job_start`, `job_list`, `job_kill` | one tool call                                                                       |
| `read`, `job_read`                                                                      | `read_file` / `job_output` asking for base64; `read` loops until `max` bytes or EOF |
| `job_attach`                                                                            | `job_output` polled to the job's end (the client library's own follow)              |
| `job_remove`, `job_wipe`                                                                | `job_list`, then `job_kill` with `remove` on the ended jobs / on every job          |
| `capabilities`                                                                          | the `sandbox://capabilities` resource                                               |
| `exec_stream`                                                                           | `unsupported`: a tool result is not streamed                                        |
| `write_if_unchanged`                                                                    | `write_file` with `if_unchanged_since` (a number) or `create_only` (null)           |

An unsupported op is left out of `hello`'s list, so the harness counts the scenarios that need it as `unsupported`, never
as passes.

## Release

`.github/workflows/release-sandbox-mcp.yml` builds a static musl `sandbox-mcp` for linux/amd64 and linux/arm64 on a tag
and attaches `sandbox-mcp-<tag>-linux-<arch>.tar.gz`, its `.sha256`, and `deploy/sandbox-mcp/install-sandbox-mcp.sh`,
the same way `release-execd.yml` does for execd.
