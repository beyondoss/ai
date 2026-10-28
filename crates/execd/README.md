# beyond-execd

**Give the agent its full toolset in any sandbox.** `execd` is one static binary that speaks the whole of the agent's exec
protocol. Put it in the image and the agent works in your sandbox as it does on your laptop: same tools,
same schemas, same descriptions, so the prompt cache stays warm.

A hand-written endpoint usually stops at one POST and one answer, where `bash` cannot stream, cannot tell a silent command
from a hung one, and cannot kill. execd implements all of it, and a client sends what it needs without asking first:

| what      | what the agent gets                                                                                                                                                                                                                     |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| one-shot  | one request, one answer, with stdin in and the command's bytes out: `stdin`, `stdout` and `stderr` are standard base64, so nothing is lost to text                                                                                      |
| streaming | the connection is the command's lifetime. `bash` shows progress, a silent command is handed back as a background job instead of blocking the turn, and closing the stream kills the process group                                       |
| jobs      | sandbox-native jobs: jobs live in the endpoint, survive the agent replica moving, and their output is readable from an offset; a job a signal killed reports `signal` beside `code` 128 + S, so a kill by 9 differs from an exit of 137 |
| notify    | job-end notify: the sandbox calls back when a kept job ends                                                                                                                                                                             |

**Proven:** exec conformance 32/32 live on AWS Fargate and E2B; a 5 MB static binary; 2,372 KiB idle PSS and zero
idle wakeups ([measurements](ARCHITECTURE.md#performance)); commands isolated from execd's token and memory under `EXECD_RUN_AS`;
it fails closed without a token.

Lifecycle stays yours. execd never creates or destroys a sandbox, so it has nothing vendor-specific in it: E2B, Modal,
Daytona, Fargate, docker, Kubernetes or your own.

## Why you can trust it

- **The agent's own verifier passes it.** The conformance kit that grades every endpoint runs against the
  real binary as a separate process, behind a bearer, with the spec's own timings: framing, heartbeats, the 64 KiB frame cap,
  leftover processes, jobs, and the full notify retry rules
  ([`tests/conformance.rs`](tests/conformance.rs)).
- **The agent's integration suites run this code.** The test double the remote-exec, `serve` and notify suites use
  ([`exec_mock`](../test-support/src/exec_mock.rs)) runs execd for its conforming configuration, so those suites exercise what
  ships.
- **Disconnect kills the whole tree.** The verifier starts a grandchild that writes a marker file, drops the connection, and
  checks the marker stops.
- **Nothing grows.** Every buffer names its bound and what frees it
  ([the table](ARCHITECTURE.md#bounds)). A client that stops reading stalls the command; memory stays flat. A test
  ([`tests/endpoint.rs`](tests/endpoint.rs)) runs a command that writes 256 MiB against a client that never reads and asserts
  it stalls and stays stalled.
- **PID 1 ready.** It adopts and reaps orphans without ever stealing the exit status of a command it awaits, and a zombie never
  counts as a live process ([`tests/reaper.rs`](tests/reaper.rs)).
- **Safe to expose.** A bearer compared in constant time, read from a file or the environment and never from argv, and never
  passed to the commands it runs. With none set it refuses to start on any address but loopback. `EXECD_RUN_AS` runs every command as
  another user, so a prompt-injected command cannot read the bearer, execd's memory or its environment. Hundreds of hostile
  requests over HTTP/1.1 and h2c are tested to be refused and to run nothing
  ([`tests/auth_matrix.rs`](tests/auth_matrix.rs), [what is and is not guaranteed](ARCHITECTURE.md#security)).

## Run it

Every recipe, with the agent side (`--exec-url`, `--exec-header`), is in
[Run the tools somewhere else](../agent/README.md#run-the-tools-somewhere-else). In short:

```dockerfile
COPY --from=ghcr.io/beyondoss/execd:v0.1.11 /execd /usr/local/bin/execd
```

```sh
curl -fsSL https://github.com/beyondoss/ai/releases/latest/download/install-execd.sh | sh   # Alpine, Debian, busybox
echo -n "$TOKEN" > /run/execd-bearer && chmod 600 /run/execd-bearer
EXECD_LISTEN=0.0.0.0:8080 EXECD_BEARER_FILE=/run/execd-bearer EXECD_RUN_AS=sandbox execd   # as root
```

The tarballs, checksums and the other ways to install are in [`deploy/execd`](../../deploy/execd/README.md). To build from
source (the crate is `beyond-execd`, the binary `execd`): `cargo install --path crates/execd`.

**Security, in three lines.** Run execd as root with `EXECD_RUN_AS`. Give it a bearer file. Put TLS in front of it, or use the
provider's (E2B does).

Keep `EXECD_MAX_RESPONSE_BYTES` equal to the agent's `--exec-max-response-bytes`. Run execd as PID 1 in a container, or under
any init: it makes itself the subreaper either way.

## Configuration

execd takes no command-line arguments, only environment variables, read once at startup. A value that does not parse is an
error that names the variable and the value (exit 2), and so is any argument: execd prints this list and exits. A boolean is
`1`, `true`, `0` or `false`; an empty variable is unset.

| variable                   | default                                            | meaning                                                                                                  |
| -------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `EXECD_LISTEN`             | `127.0.0.1:8080`                                   | address to listen on                                                                                     |
| `EXECD_ROOT`               | the current directory                              | where a command runs when its request names no directory                                                 |
| `EXECD_HOME`               | execd's own `HOME`; the user's with `EXECD_RUN_AS` | `HOME` for every command                                                                                 |
| `EXECD_BEARER_FILE`        | unset                                              | a file holding the bearer token (mode 0600)                                                              |
| `EXECD_BEARER`             | unset                                              | the bearer token itself, a weaker way: execd re-execs to move it out of its environment                  |
| `EXECD_INSECURE_NO_AUTH`   | `false`                                            | serve with no bearer on a non-loopback address; without it execd refuses to start                        |
| `EXECD_RUN_AS`             | unset                                              | run every command as `user`, `uid`, `user:group` or `uid:gid` with the user's groups; execd must be root |
| `EXECD_MAX_RESPONSE_BYTES` | 16 MiB                                             | most output one one-shot answer carries; match the agent's `--exec-max-response-bytes`                   |
| `EXECD_JOBS_DIR`           | `$TMPDIR`, else `/tmp`                             | where a kept job's last 8 MiB of output is spooled; use a disk-backed directory                          |
| `EXECD_LOG`                | `info`                                             | `off`, `error`, `warn`, `info`, `debug` or `trace`                                                       |
| `EXECD_WORKER_THREADS`     | `1`                                                | `1` one thread; `0` a worker per CPU; `N` workers                                                        |
| `EXECD_IDLE_TIMEOUT_MS`    | `0` (off)                                          | close a connection with no request in flight for this long (a stream in flight is never cut)             |

No `EXECD_*` variable reaches the commands execd runs. The startup line logs execd's version.

`GET /livez` (or `/healthz`) answers 200 without authentication, and nothing else is open. `SIGTERM` and `SIGINT` kill every running process
group and exit.

Without `EXECD_RUN_AS` a command is the same user as execd and can read its memory and any file it owns, the bearer file included.
With it, and with the token in a root-owned 0600 file, it cannot. `EXECD_ROOT` must be writable by the `EXECD_RUN_AS` user (execd says at
startup when it is not). See [ARCHITECTURE.md](ARCHITECTURE.md#security) for what each control guarantees and what it does not.

## Embedding

```rust
let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
let server = execd::serve(listener, execd::Options::new("/work"))?;
// ...
server.shutdown().await;
```

An `Observer` set on `Options` is told of every request, stream and callback attempt; unset it costs one branch.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the request flow, the process model and what bounds every resource.
