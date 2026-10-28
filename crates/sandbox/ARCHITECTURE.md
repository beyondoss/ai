# beyond-sandbox architecture

A client for the exec protocol, in three layers that each stand alone. It was extracted from `crates/agent`, which now
depends on it; it depends on neither `beyond-agent` nor `beyond-agent-core`.

## Modules

| file                               | what                                                                                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `src/lib.rs`                       | the public names: `Exec`, `HttpExec`, `Files`, `Jobs`, `Sandbox`, `TokenSource`                                              |
| `src/exec.rs`                      | `CommandRunner` (`Exec`), `ExecResult`, `ExecOutput` (code and bytes), `Kill`/`LocalKill`/`kill_all`, bounded output capture |
| `src/exec_endpoint.rs`             | the protocol spec (module doc), `HttpExecRunner`                                                                             |
| `src/exec_endpoint/stream.rs`      | stream frames and the frame reader                                                                                           |
| `src/exec_endpoint/jobs.rs`        | `SandboxJobs`, scopes, ids                                                                                                   |
| `src/exec_endpoint/jobs_native.rs` | the endpoint's jobs: `NativeJobs`, `HttpJobs`                                                                                |
| `src/fs.rs`                        | `FsBackend`, its types, and the result policy every backend shares                                                           |
| `src/fs/shell.rs`                  | `ShellFs` (`Files`): file operations as commands, the one lazy environment probe, parsers, `__fuzz`                          |
| `src/bulk.rs`                      | `upload`/`download`/`extract`: tar over exec, the three scripts, the staged upload, `Sources`, `Transfer`, `BulkError`       |
| `src/bulk/extract.rs`              | the policy layer over the `tar` crate for untrusted archives: never writes through a link, directory modes last              |
| `src/procgroup.rs`                 | process-group identity and kill guards for local children                                                                    |
| `src/sandbox.rs`                   | `Jobs`, `Sandbox`, the `Command` builder                                                                                     |
| `src/token.rs`                     | `TokenSource` and the one-renewal bearer                                                                                     |
| `src/tls.rs`                       | installs the `ring` rustls provider before a client is built                                                                 |
| `src/testing.rs`                   | a plain local runner for unit tests                                                                                          |

## Layers

```
Sandbox            layer 3   connect, .exec() .files() .jobs() .raw()          src/sandbox.rs
Files  Jobs        layer 2   built only on layer 1                              src/fs/shell.rs, src/exec_endpoint/jobs*.rs
Exec               layer 1   run, run_streaming(_tracked), run_attached,        src/exec.rs
                             run_with_stdin, sandbox_jobs, ready
  HttpExec   your own                                          src/exec_endpoint.rs
```

**Layer 1 is one trait.** `Exec` is `exec::CommandRunner`. `run`, `run_streaming`, `run_streaming_tracked` and
`run_attached` are required: every implementation streams and can be stopped. `run_with_stdin` answers `Unsupported`
unless overridden, `ready` is a no-op, and `sandbox_jobs` is `None` for a runner whose commands run on this host.
Calling a sink is a promise that every byte of both streams went through it.

**Layer 2 never reaches below the trait.** `ShellFs` (`Files`) builds every file operation from `Exec::run` and
`run_with_stdin`; `Jobs` is `Exec::sandbox_jobs()`: the target's own jobs (the HTTP client's), and `None` for a runner without any, where
`Sandbox::jobs()` answers `Unsupported`. Neither knows about HTTP. The proof is a test-written transport (`sh` through
`std::process::Command`) that runs the files, helpers and bulk scenarios with the same counts as the HTTP client
(`tests/sdk_conformance.rs`, profile `local-sh`).

**Layer 3 is a holder.** `Sandbox` owns an `Arc<dyn Exec>`, the `ShellFs` over it and the `Jobs`, and hands the
raw `Exec` back (`.raw()`), so nothing the helpers do is out of reach of a caller who needs something else.

## The HTTP client (`exec_endpoint.rs`)

`HttpExec` (`HttpExecRunner`) is a `reqwest::Client` and a URL: no session, no lifecycle. One request path,
`send_job`, carries every call, which is why auth lives in one place.

- **Nothing is negotiated or probed.** An endpoint speaks the whole protocol (one-shot with stdin, streaming, jobs,
  notify), so a request is just sent. A `400`/`422` is the endpoint declining one request, not a sign the sandbox was
  replaced.
- **Auth** is `bearer(TokenSource)`. The source (`token.rs`) is a fixed token, a closure, or a command. The first request
  asks it; on a `401` the client asks again once and repeats that request once, and a source that returns the token it
  was already using, or a fixed source, ends it there. Concurrent requests compare the token they failed with against
  the current one, so a renewal is shared, not repeated. Error text never contains a token (`Debug` is redacted).
- **Bounded.** A one-shot answer is read incrementally up to the base64 of `with_max_response_bytes` (16 MiB of decoded bytes by
  default) and an over-cap answer is an error; a stream frame is at most 64 KiB and a line at most 1 MiB; a silent stream is abandoned after 30 s
  (heartbeats are every 5 s).
- **Disconnect kills.** For a stream the connection is the command's lifetime: dropping the call, or the `Kill` a
  tracked run hands out, closes it, and the endpoint kills the process group.

## Files (`fs.rs`, `fs/shell.rs`)

`FsBackend` is the seam the agent's `read`/`write`/`edit`/`ls`/`grep`/`find` use; `ShellFs` is its implementation over
`Exec`. It looks at the target's tools once, in one `sh -c` the first operation that needs it sends (`Capabilities`: `rg`, GNU
or busybox `grep`, `find -printf`, read windows; a write never probes) and picks the best rung of the search ladder; the exact commands, parsing and error kinds are
`sdk/conformance/FILES-OVER-EXEC.md`, and `ShellFs` is its reference implementation. The impl-independent tail every
backend shares (`clip`, `finalize`, `finalize_glob`) is in `fs.rs`, so a result's shape holds by construction. The
agent's `LocalFs` implements the same trait on the host's disk.

Parsers of command output are private; `fs::shell::__fuzz` is the `#[doc(hidden)]` seam for `fuzz/`.

## Bulk copy (`bulk.rs`, `bulk/`)

`bulk::upload(exec, sources, remote_dir)` and `bulk::download(exec, remote_path, local_dir)` move whole trees as a tar
stream built only on `Exec` (spec: `sdk/conformance/FILES-OVER-EXEC.md` 5.8). The archive is written and read by the `tar` crate (no system `tar` on the client machine); the sandbox needs `sh` and a GNU or busybox
`tar`.

- **Upload** is a producer on a blocking thread (a walk that appends headers and file bytes to a buffer) feeding a channel
  of capacity one, and the async side sending what arrives: a tree that fits `BULK_CHUNK_BYTES` (16 MiB) is one command with
  the archive on stdin; a larger one is appended to a stage file beside the destination (each append checks the stage's
  size first, so a repeated or lost chunk fails) and extracted once. Memory is about two chunks. A failure removes the stage
  (`rm -f`) and, for a destination this call created, the script removes the destination.
- **Download** streams through `run_attached` (stdout frames go down a bounded channel to a blocking thread that
  reads them with `tar::Archive` and feeds `extract::Extractor`, with stderr kept to 64 KiB). An unsafe
  member closes the stream. The extractor lstat-walks every parent, creates files `O_EXCL`, replaces (never follows) a link
  in the way, applies directory modes and mtimes last, and refuses names with `..`, absolute names, hard links to anything
  the archive did not write, and unsupported member types.
- **Errors** are `BulkError`: `InvalidRequest` (nothing sent), `Archive` (`protocol`: unsafe or malformed), `Command`
  (a failed script, classified from its stderr like the file helpers), `Io` (transport or local).
- **Tests**: `tests/bulk.rs` (wire shape, the staged path, the cleanup, over a transport written in the test),
  `bulk/extract.rs` unit tests, the `extract` fuzz target (`fuzz/fuzz_targets/extract.rs`: nothing lands beside the target or through a link), and `sdk/conformance/scenarios/bulk.json` through every driver.

## Jobs (`exec_endpoint/jobs.rs`, `jobs_native.rs`)

`SandboxJobs` is the operation set (run held, start kept, follow, read from an offset, keep, kill, list, remove, wipe).
`NativeJobs` speaks the operations to an endpoint, and `HttpJobs` is the one an `HttpExec` hands out. The endpoint owns
the processes, the output and the notify sender (`execd` is the reference); the client holds none of it.

## Process-group machinery (`procgroup.rs`)

The agent's host runner spawns a child as its own process-group leader and must kill the whole
group if the owning future is dropped. `GroupId` names a group in a way that survives pid recycling, `GroupKillGuard`
kills it on drop, and `spawn_tracked_cleanup` runs the blocking kill on its own thread that the exit path can wait for
(`wait_for_pending_group_kills`). `Kill` is either a hook (a stream close, a sandbox job's kill) or a
`LocalKill`, which the embedding program implements for its own local jobs; `kill_all` spends one SIGTERM grace for any
number of them.

## Raw exec is bytes

`CommandRunner::run` returns an `ExecResult` whose `stdout` and `stderr` are `Vec<u8>`: exactly what the command wrote.
The one-shot answer carries them as standard base64, so there is no lossy path. Text is
`ExecResult::stdout_text`/`stderr_text` (UTF-8, invalid sequences replaced), which is what the helpers parse and the
agent's `bash` tool shows. `Sandbox::exec().run()` returns `ExecOutput`: the exit code (`128 + N` for a signal, `124`
for a deadline) and the bytes. The response cap is on the decoded bytes of both streams together.

## What is not here

The host command runner (`RealRunner`), `LocalFs`, the model-facing tools, the lazily attached `ExecCell`, the notify
callback receiver and the server-conformance verifier are agent policy and stay in `crates/agent`.

## Specification and tests

The normative protocol spec is the module doc of `src/exec_endpoint.rs` (the whole protocol); the vectors are in
`tests/fixtures/exec/`. Tests are small sibling files: `exec_stream_*` (frames, kill on disconnect), `exec_output` (raw exec is bytes: all 256 values on both streams, corrupt base64 is an error, the cap), `exec_notify_*`
(notify against the mock), `exec_jobs_vectors`, `token_refresh`, `sandbox_surface`, and `sdk_conformance` (the shared
matrix, with the reference driver in `tests/sdk_conformance/driver.rs`). Run them with `cargo nextest run -p beyond-sandbox`.
