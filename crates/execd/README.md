# beyond-ai-execd

`execd` is the exec endpoint that runs inside a sandbox. The agent POSTs a command to its URL and execd runs it as a real
process: the whole of the agent's exec protocol, v1.1 through v1.4 (stdin, streaming, sandbox jobs, job-end notify), in one
binary.

```sh
execd --listen 0.0.0.0:8080
```

It passes the agent's conformance verifier, the same one that grades every other endpoint, as a real process in CI.

- **Streams that cannot grow.** Output flows through bounded channels, so a client that stops reading stalls the command
  instead of filling memory.
- **A command is its connection.** When the client leaves, the whole process group is killed at once, grandchildren included.
- **PID 1 ready.** It adopts and reaps orphans without ever stealing the exit status of a command it awaits, and a zombie never
  counts as a live process.
- **Safe to expose.** An optional bearer, compared in constant time, read from a file or the environment and never from argv.
  With none set it says so at startup.

## Options

| flag                       | environment                    | meaning                                                                                   |
| -------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------- |
| `--listen`                 | `EXECD_LISTEN`                 | address to listen on (default `127.0.0.1:8080`)                                           |
| `--root`                   | `EXECD_ROOT`                   | where a command runs when its request names no directory (default: the current directory) |
| `--home`                   | `EXECD_HOME`                   | `HOME` for every command                                                                  |
| `--bearer-file`            | `EXECD_BEARER_FILE`            | a file holding the bearer token; `EXECD_BEARER` holds the token itself                    |
| `--kept-jobs-survive-idle` | `EXECD_KEPT_JOBS_SURVIVE_IDLE` | advertise v1.4 job-end notify; say so only if this sandbox is not stopped when idle       |
| `--log-level`              | `EXECD_LOG`                    | `error`, `warn`, `info`, `debug` or `trace`                                               |

`GET /livez` answers 200 without authentication. `SIGTERM` and `SIGINT` kill every running process group and exit.

## Embedding

```rust
let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
let server = execd::serve(listener, execd::Options::new("/work"))?;
// ...
server.shutdown().await;
```

An `Observer` set on `Options` is told of every request, stream and callback attempt; unset it costs one branch.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the request flow, the process model and what bounds every resource.
