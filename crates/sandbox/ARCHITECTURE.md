# beyond-sandbox architecture

A client for the exec protocol, in three layers that each stand alone. It was extracted from `crates/agent`, which now
depends on it; it depends on neither `beyond-agent` nor `beyond-agent-core`.

## Modules

| file                               | what                                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `src/lib.rs`                       | the public names: `Exec`, `HttpExec`, `Files`, `Jobs`, `Sandbox`, `TokenSource`               |
| `src/exec.rs`                      | `CommandRunner` (`Exec`), `ExecResult`, `Kill`/`LocalKill`/`kill_all`, bounded output capture |
| `src/exec_endpoint.rs`             | the protocol spec (module doc), `HttpExecRunner`, `TemplateRunner`                            |
| `src/exec_endpoint/stream.rs`      | v1.2 frames, the frame reader, the streaming probe                                            |
| `src/exec_endpoint/jobs.rs`        | `SandboxJobs`, `ShellJobs` (the baseline over plain exec), scopes, ids                        |
| `src/exec_endpoint/jobs_native.rs` | v1.3 native jobs and `HttpJobs`                                                               |
| `src/exec_endpoint/far_kill.rs`    | killing a template transport's command on the far side                                        |
| `src/fs.rs`                        | `FsBackend`, its types, and the result policy every backend shares                            |
| `src/fs/shell.rs`                  | `ShellFs` (`Files`): file operations as commands, capability probe, parsers, `__fuzz`         |
| `src/procgroup.rs`                 | process-group identity and kill guards for local children                                     |
| `src/sandbox.rs`                   | `Jobs`, `Sandbox`, the `Command` builder                                                      |
| `src/token.rs`                     | `TokenSource` and the one-renewal bearer                                                      |
| `src/tls.rs`                       | installs the `ring` rustls provider before a client is built                                  |
| `src/testing.rs`                   | a plain local runner for unit tests                                                           |

## Layers

```
Sandbox            layer 3   connect, .exec() .files() .jobs() .raw()          src/sandbox.rs
Files  Jobs        layer 2   built only on layer 1                              src/fs/shell.rs, src/exec_endpoint/jobs*.rs
Exec               layer 1   run, run_streaming(_tracked), run_attached,        src/exec.rs
                             run_with_stdin, sandbox_jobs, ready
  HttpExec   TemplateRunner   your own                                          src/exec_endpoint.rs
```

**Layer 1 is one trait.** `Exec` is `exec::CommandRunner`; only `run` is required. Everything else defaults to the
safe answer: `run_streaming` runs and never calls the sink, `run_streaming_tracked` runs under a 30-minute deadline and
reports no `Kill`, `run_attached` and `run_with_stdin` answer `Unsupported`, `tracks_jobs` and `attaches` say `false`.
A runner that does more says so, and the callers ask before relying on it. Calling a sink is a promise that every byte
of both streams went through it.

**Layer 2 never reaches below the trait.** `ShellFs` (`Files`) builds every file operation from `Exec::run` and
`run_with_stdin`; `Jobs` is `Exec::sandbox_jobs()` when the runner has native jobs, else `ShellJobs` over plain `run` and
`run_attached`. Neither knows about HTTP. The proof is a test-written transport (`sh` through `std::process::Command`)
that runs the shared scenarios with the same counts as the HTTP client (`tests/sdk_conformance.rs`, profile `local-sh`
against `http-subset`).

**Layer 3 is a holder.** `Sandbox` owns an `Arc<dyn Exec>`, the `ShellFs` probed over it and the `Jobs`, and hands the
raw `Exec` back (`.raw()`), so nothing the helpers do is out of reach of a caller who needs something else.

## The HTTP client (`exec_endpoint.rs`)

`HttpExec` (`HttpExecRunner`) is a `reqwest::Client` and a URL: no session, no lifecycle. One request path,
`send_job`, carries every call, which is why auth lives in one place.

- **Capabilities are proved, not assumed.** Streaming (v1.2) is probed once on the first streaming call (a command that
  prints, sleeps, prints: two `output` frames at least 100 ms apart, `application/x-ndjson`); native jobs (v1.3) and
  notify (v1.4) by one `job` request. A probe that could not reach the endpoint is an outage and is retried; a probe
  the endpoint answered is cached. A `400`/`422` is the endpoint declining one request, not a reason to distrust it.
- **Auth** is `bearer(TokenSource)`. The source (`token.rs`) is a fixed token, a closure, or a command. The first request
  asks it; on a `401` the client asks again once and repeats that request once, and a source that returns the token it
  was already using, or a fixed source, ends it there. Concurrent requests compare the token they failed with against
  the current one, so a renewal is shared, not repeated. Error text never contains a token (`Debug` is redacted).
- **Bounded.** A response is read incrementally up to `with_max_response_bytes` (16 MiB default) and an over-cap body is
  an error; a stream frame is at most 64 KiB and a line at most 1 MiB; a silent stream is abandoned after 30 s
  (heartbeats are every 5 s).
- **Disconnect kills.** For a stream the connection is the command's lifetime: dropping the call, or the `Kill` a
  tracked run hands out, closes it, and the endpoint kills the process group.

`TemplateRunner` drives targets with no HTTP surface (`ssh host -- {}`, `docker exec ctr {}`). Killing the local transport
does not kill the far command (measured with a real `docker exec`), so every command runs under a wrapper that starts it
in a new session and prints its pid first; `far_kill.rs` kills the group through a second invocation of the transport.

## Files (`fs.rs`, `fs/shell.rs`)

`FsBackend` is the seam the agent's `read`/`write`/`edit`/`ls`/`grep`/`find` use; `ShellFs` is its implementation over
`Exec`. It probes the target once (`Capabilities`: stdin delivery, `rg`, GNU or busybox `grep`, `find -printf`, read
windows) and picks the best rung of the fallback ladder; the exact commands, parsing and error kinds are
`sdk/conformance/FILES-OVER-EXEC.md`, and `ShellFs` is its reference implementation. The impl-independent tail every
backend shares (`clip`, `finalize`, `finalize_glob`) is in `fs.rs`, so a result's shape holds by construction. The
agent's `LocalFs` implements the same trait on the host's disk.

Parsers of command output are private; `fs::shell::__fuzz` is the `#[doc(hidden)]` seam for `fuzz/`.

## Jobs (`exec_endpoint/jobs.rs`, `jobs_native.rs`)

`SandboxJobs` is the operation set (run held, start kept, follow, read from an offset, keep, kill, list, remove, wipe).
`NativeJobs` speaks v1.3 to an endpoint that offers it. `ShellJobs` is the baseline for one that does not: a supervisor
script run over plain exec, a job directory under `/tmp/beyond-jobs/<scope>/`, output read back with `dd`/`base64`. It
is client logic that any user of the crate needs on endpoints without v1.3, so it lives here, not in the agent.
`HttpJobs` picks native or shell per endpoint, once. The v1.4 notifier script comes from `beyond-exec-notify`.

## Process-group machinery (`procgroup.rs`)

A `TemplateRunner` and the agent's host runner both spawn a child as its own process-group leader and must kill the whole
group if the owning future is dropped. `GroupId` names a group in a way that survives pid recycling, `GroupKillGuard`
kills it on drop, and `spawn_tracked_cleanup` runs the blocking kill on its own thread that the exit path can wait for
(`wait_for_pending_group_kills`). `Kill` is either a hook (a stream close, a far-side kill, a sandbox job's kill) or a
`LocalKill`, which the embedding program implements for its own local jobs; `kill_all` spends one SIGTERM grace for any
number of them.

## What is not here

The host command runner (`RealRunner`), `LocalFs`, the model-facing tools, the lazily attached `ExecCell`, the v1.4
callback receiver and the server-conformance verifier are agent policy and stay in `crates/agent`.

## Specification and tests

The normative protocol spec is the module doc of `src/exec_endpoint.rs` (v1.1 to v1.4); the vectors are in
`tests/fixtures/exec/`. Tests are small sibling files: `exec_stream_*` (frames, kill on disconnect), `exec_notify_*`
(v1.4 against the mock), `exec_jobs_vectors`, `token_refresh`, `sandbox_surface`, and `sdk_conformance` (the shared
matrix, with the reference driver in `tests/sdk_conformance/driver.rs`). Run them with `cargo nextest run -p beyond-sandbox`.
