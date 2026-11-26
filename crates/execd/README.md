# beyond-execd

**Give the agent its full toolset in any sandbox.** `execd` is one static binary that speaks the whole of the agent's exec
protocol. Put it in the image and the agent works in your sandbox as it does on your laptop: same tools,
same schemas, same descriptions, so the prompt cache stays warm.

A hand-written endpoint usually stops at one POST and one answer, where `bash` cannot stream, cannot tell a silent command
from a hung one, and cannot kill. execd implements all of it, and a client sends what it needs without asking first:

| what       | what the agent gets                                                                                                                                                                                                                     |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| one-shot   | one request, one answer, with stdin in and the command's bytes out: `stdin`, `stdout` and `stderr` are standard base64, so nothing is lost to text                                                                                      |
| streaming  | the connection is the command's lifetime. `bash` shows progress, a silent command is handed back as a background job instead of blocking the turn, and closing the stream kills the process group                                       |
| jobs       | sandbox-native jobs: jobs live in the endpoint, survive the agent replica moving, and their output is readable from an offset; a job a signal killed reports `signal` beside `code` 128 + S, so a kill by 9 differs from an exit of 137 |
| job extras | a job's stderr told from its stdout on one offset clock, positional writes to a kept job's stdin, a named `signal` op, and a command's own `env` (never `EXECD_*`)                                                                      |
| notify     | job-end notify: the sandbox calls back when a kept job ends                                                                                                                                                                             |

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

Prove it locally (stdout and stderr are standard base64; `aGkK` is `hi\n`):

```sh
EXECD_LISTEN=127.0.0.1:8080 EXECD_BEARER_FILE=/run/execd-bearer execd &
curl -fs http://127.0.0.1:8080/livez
curl -s -X POST -H "Authorization: Bearer $(cat /run/execd-bearer)" \
  -d '{"command":"sh","args":["-c","echo hi"]}' http://127.0.0.1:8080/
# {"exit_code":0,"stderr":"","stdout":"aGkK"}
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

| variable                   | default                                            | meaning                                                                                                                                                                                                                                                                          |
| -------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EXECD_LISTEN`             | `127.0.0.1:8080`                                   | address to listen on                                                                                                                                                                                                                                                             |
| `EXECD_ROOT`               | the current directory                              | where a command runs when its request names no directory                                                                                                                                                                                                                         |
| `EXECD_HOME`               | execd's own `HOME`; the user's with `EXECD_RUN_AS` | `HOME` for every command                                                                                                                                                                                                                                                         |
| `EXECD_BEARER_FILE`        | unset                                              | a file holding the bearer token (mode 0600)                                                                                                                                                                                                                                      |
| `EXECD_BEARER`             | unset                                              | the bearer token itself, a weaker way: execd re-execs to move it out of its environment, and refuses to start if it cannot                                                                                                                                                       |
| `EXECD_INSECURE_NO_AUTH`   | `false`                                            | serve with no bearer on a non-loopback address; without it execd refuses to start                                                                                                                                                                                                |
| `EXECD_RUN_AS`             | unset                                              | run every command as `user`, `uid`, `user:group` or `uid:gid` with the user's groups; execd must be root                                                                                                                                                                         |
| `EXECD_MAX_RESPONSE_BYTES` | 16 MiB                                             | most output one one-shot answer carries; match the agent's `--exec-max-response-bytes`                                                                                                                                                                                           |
| `EXECD_JOBS_DIR`           | `$TMPDIR`, else `/tmp`                             | where a kept job's last 8 MiB of output is spooled, with a ledger too (in `execd-spool-<tag>/` there, which a restart finds); use a disk-backed directory                                                                                                                        |
| `EXECD_STATE_DIR`          | none                                               | where the jobs ledger is kept (only the ledger: output files are under `EXECD_JOBS_DIR`), so a restart (`SIGHUP`) of execd leaves its jobs running and the next one finds them (a stop, `SIGTERM`, always ends them); without it a restart kills them too; use a tmpfs directory |
| `EXECD_LOG`                | `info`                                             | `off`, `error`, `warn`, `info`, `debug` or `trace`                                                                                                                                                                                                                               |
| `EXECD_WORKER_THREADS`     | `1`                                                | `1` one thread; `0` a worker per CPU; `N` workers                                                                                                                                                                                                                                |
| `EXECD_IDLE_TIMEOUT_MS`    | `0` (off)                                          | close a connection with no request in flight for this long (a stream in flight is never cut)                                                                                                                                                                                     |
| `EXECD_JOB_OUTPUT_MS`      | 900000 (15 min)                                    | drop an ended job's output this long after its end, keeping its status (a late reader gets a gap); 0 never                                                                                                                                                                       |
| `EXECD_JOBS_ENDED_BYTES`   | 256 MiB                                            | the most output kept over ended jobs; past it the oldest's output is dropped first; 0 unbounded                                                                                                                                                                                  |
| `EXECD_JOBS_ENDED_MAX`     | 256                                                | the most ended jobs kept; past it the oldest reapable entry is reaped (never a keyed job, one with a stream, or one owed a callback); 0 unbounded                                                                                                                                |
| `EXECD_JOB_ENTRY_MS`       | 0 (off)                                            | reap an ended job's entry, status included, this long after its end                                                                                                                                                                                                              |

## The heartbeat interval

A stream, a held job's stream and a long one-shot answer carry a heartbeat when they have been silent for a beat (5 s by default),
so no proxy on the path sees a silence. A request may choose its own beat with the `Keepalive-Ping-Interval` header, a whole number
of seconds (E2B's envd takes the same header):

| header value                                         | beat                                                                                |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `1` to `240`                                         | that many seconds                                                                   |
| absent, `0`, negative, not an integer, or over `240` | the default, 5 s. There is no "off": a silent command is never left without a byte. |

240 s is below every idle limit on Beyond's path (the gateway's 295 s, the edge's 300 s, the load balancer's 350 s), and the front
lengthens its own stall limit for the request to stay over the interval. A longer interval means a dead connection is noticed later: a
client that reads one frame per interval (and must keep its own silence limit above it: the agent's is 30 s) cannot tell a vanished sandbox from a quiet one sooner than one interval, so the "client sees a
failure within 10 s" bound holds only at the default beat.

## Errors

An error before a command starts is the status below with `{"error": "<reason>"}`. A job start that cannot make its output spool
because the sandbox's disk is full answers `507` (`the sandbox's disk is full: no room for the job's output`), as E2B's envd does for a
full disk; any other failure to make the spool is `500`.

No `EXECD_*` variable reaches the commands execd runs. The startup line logs execd's version.

`GET /livez` (or `/healthz`) answers 200 without authentication, and nothing else is open. `SIGTERM` and `SIGINT` stop execd: every running process
group, a job's included, is ended (`SIGTERM`, then `SIGKILL` after 300 ms) and execd exits. `SIGHUP` restarts it: with `EXECD_STATE_DIR` the jobs keep running and the next execd adopts them.

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

What the binary does beyond `serve()`, and what an embedder should call:

| call                                           | when                            | if omitted                                              |
| ---------------------------------------------- | ------------------------------- | ------------------------------------------------------- |
| `helper_main()` first, before parsing anything | required with `Options::run_as` | `run_as` commands cannot start                          |
| `log::set_level(Level)`                        | to see library logs             | the library logs nothing                                |
| `become_subreaper()`                           | PID 1 / orphan reaping          | orphans go to the host's init; waiters poll every 50 ms |
| `remove_stale_jobs_dirs(jobs_dir)`             | after listening                 | crashed predecessors' job output dirs stay on disk      |
| `serve` / `Handle::shutdown`                   | always                          | —                                                       |

An `Observer` set on `Options` is told of every request, stream and callback attempt; unset it costs one branch.
`examples/embed.rs` is the minimal correct `main()`. See [ARCHITECTURE.md](ARCHITECTURE.md#embedding) for auth,
idle timeout, activity, `job_op_hook`, and `route_hook` (claim a path before execd reads it; uncounted as activity).
