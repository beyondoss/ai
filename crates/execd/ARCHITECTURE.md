# execd architecture

execd is the exec endpoint that runs inside a sandbox. The agent POSTs a command to a URL; execd runs it as a real process and
answers. It implements the whole of the agent's exec protocol: v1.1 (one request, one JSON answer, stdin), v1.2 streaming,
v1.3 sandbox jobs and v1.4 job-end notify. The normative spec is the module doc of `crates/agent/src/exec_endpoint.rs`; this
file says how execd meets it and what bounds every resource it holds.

It is one crate: a library (`execd`) that embeds in a test or a larger host, and a thin `execd` binary (`--listen`, `--root`,
`--home`, `--bearer-file`, `--kept-jobs-survive-idle`, `--log-level`).

## Modules

| file            | role                                                                                           |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `src/lib.rs`    | `Options`, the `Observer` trait, `serve` and `Handle`, and the derived limits                  |
| `src/main.rs`   | the CLI: flags, bearer sources, subreaper and reaper, signals                                  |
| `src/http.rs`   | hyper (HTTP/1.1 and h2c), routing, `/livez`, the bearer check, the request limit, the log line |
| `src/exec.rs`   | v1.1 envelope and v1.2 stream: pumps, the sink, heartbeat, leftovers, timeout, disconnect      |
| `src/jobs.rs`   | v1.3 jobs: the table, the output ring, the supervisor, job streams                             |
| `src/notify.rs` | v1.4: the sender that drives the `exec-notify` kernel                                          |
| `src/proc.rs`   | process groups by syscall, the spawn registry, the subreaper and orphan reaper                 |
| `src/frame.rs`  | v1.2 frame encoding                                                                            |

## Request flow

```text
accept ──► hyper (HTTP/1.1, or prior-knowledge HTTP/2 on the same port)
              │
   /livez ◄───┤ 200, no auth
              ▼
   POST ──► bearer check (constant time) ──► 401
              ▼
   read the body, at most `max_request_bytes` ──► 413
              ▼
   parse the JSON envelope ──► 400
              ▼
   observer.request(headers, body)            (only when an observer is set)
              ▼
   ┌──────────┴────────────┬─────────────────────────┐
   │ `job` object          │ `"stream": true`        │ otherwise
   ▼                       ▼                         ▼
 jobs::serve            exec::stream              exec::envelope
 (v1.3, and the         (v1.2: the connection     (v1.1: run to the end,
  v1.4 `keep`)           is the command's life)    one JSON object)
```

A request that is none of those fails before a process exists. A command that cannot be spawned is an ordinary result (exit
127), as a shell reports it.

### v1.1: `exec::envelope`

The command runs as the leader of a new process group. Its stdout and stderr are read into buffers capped at 16 MiB across both
(`--max-response-bytes`; see the bounds table). If output went past the cap it was drained and dropped, and the answer is a `422` (the agent treats 400 and 422 as a refusal of this request, not a dead sandbox) whose message says the command ran and its output exceeded the cap, so the model narrows it instead of rerunning. The answer is sent when the command exits and the pipes have been read, with a 100 ms grace for pipes
that a descendant holds open. The group is then killed (the spec's SHOULD for v1.1), and it is killed if the client leaves
first.

### v1.2: `exec::stream`

```text
child ──stdout──► pump ─┐
                        ├─► channel(1) ─► drive ─► Sink::put ─► channel(1) ─► hyper body ─► socket
child ──stderr──► pump ─┘                  │
                                           ├─ heartbeat when nothing was written for 5 s
                                           ├─ deadline timer (`timeout_ms`)
                                           └─ the receiver's death (the client left)
```

Both channels are bounded. A client that stops reading stalls `Sink::put`, which stalls the pumps, which fills the pipes, which
blocks the command on its own write. Nothing grows. There are four ways out of a stream, and `run_stream` ends in exactly one:

- **Finished.** The shell exits. If its group has no live member, the pipes are settled (read until they close, go quiet for
  100 ms, or one pipe-capacity of further bytes has gone by) and the final frame follows.
- **Leftovers.** If the group still has a live member, the pipes are settled, `{"shell_exit": N}` is sent, and the stream stays
  open (output, heartbeats, the deadline and disconnect kills all still apply) until the group is empty. Then the final frame
  carries the shell's own status.
- **Timed out.** At `timeout_ms` the group is killed and `{"timed_out": true}` is sent, ignoring the deadline for that frame.
- **Disconnect.** The client left: the response body is dropped, the sender's `closed()` resolves, and the group is killed at
  once. This does not wait for a heartbeat write to fail.

### v1.3 and v1.4: `jobs`, `notify`

A job is a command in its own process group whose output is kept by byte offset. `start` registers the job and spawns a
supervisor task; `hold: true` answers with a job stream, `hold: false` answers `{"started": pgid}`. Every operation is scoped:
it only ever sees the jobs of its own scope.

The supervisor waits for the shell, and when the shell exits leaving its group alive records `shell_exit`, waits for the group to
empty, and records the end. Each of those is a state change that v1.4 calls back about.

v1.4 is the `notify` module driving `crates/exec-notify`, the pure timing kernel every endpoint shares. The kernel says when an
attempt begins, when to wait, and when to stop; `notify` makes the request (one `POST`, no redirect, the server's certificate
verified, 30 s at most) and feeds back what happened. A job's latest `keep` governs, a `keep` abandons the attempt in flight,
and a removed job's timers are dropped with it. The token is held in the job and in the request header, and appears in no log,
no answer, no argument and no environment; `Debug` for it prints `<redacted>`.

## Process model

- **Groups.** Every command and job is the leader of a new process group. Liveness is `kill(-pgid, 0)`; a kill is `killpg`. Both
  are syscalls (the `nix` crate; the workspace forbids `unsafe`). `kill(2)` counts zombies, and the spec says a zombie is not a
  live member, so a group that `kill(-pgid, 0)` reports as present is then checked member by member and only non-zombie members
  count. As the subreaper, execd is the ancestor of every group member, so the members are found by walking its own process tree
  through the kernel's per-process `children` lists (`/proc/<pid>/task/*/children`), which costs the size of that tree and not
  the number of processes on the host; without a subreaper, or on a kernel without those files, it reads all of `/proc`, and it
  does so too when the tree shows an existing group with no live member, because "empty" ends a stream. A group that is gone is
  never signalled again, because its number may be reused.
- **Orphans.** The binary makes itself the child subreaper and runs `reap_orphans`: on every `SIGCHLD` it lists its own
  children (the same `children` files) and `waitpid`s the zombies among them that it did not spawn. There is no `waitpid(-1)` and
  no scan of `/proc`. Every spawn holds the `SPAWNING` lock shared from before the child exists until its pid is in the `OWNED`
  set, so spawns do not wait for each other; the reaper takes it exclusively, for one `waitpid` at a time, before it reads
  `OWNED`. A child is therefore either already registered when the reaper looks (and skipped, so a command execd awaits is never
  reaped out from under its owner) or not yet created; there is no moment at which a forked child is visible to the reaper and
  unregistered. The reaper holds nothing while it looks for zombies.
- **Shutdown.** `SIGTERM` or `SIGINT` stops the accept loop, drops every open connection (which drops every in-flight handler, and
  with it the guard that kills that handler's group), then kills every group still registered, commands and jobs alike, and
  exits.
- **Embedding.** The library never sets a subreaper or installs a signal handler. A host that wants the PID 1 duties calls
  `become_subreaper` and `reap_orphans`, as the binary does.

## Bounds

Every buffer, map, file, task and connection execd holds, what bounds it, what frees it, and what happens under N concurrent
cold callers. No limit here is invented: each is the spec's, the agent's own, or a number read from the kernel.

| what                          | bound                                                                                                                         | derivation                                                                                                                                                                                                                                                                                                                                                                                                          | freed by                                                          | N concurrent cold callers                                                                                                                                                                         |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Request body                  | 91,575,640 bytes (`MAX_REQUEST_BYTES`)                                                                                        | The agent's largest request is a file's content as `stdin_base64`: the biggest file it writes is an edit's, `MAX_EDIT_BYTES` = 64 MiB, which is `ceil(64 MiB / 3) * 4` = 89,478,488 bytes of base64. Plus 2 MiB, Linux's default `ARG_MAX`, for the command and arguments the kernel itself refuses past.                                                                                                           | The handler returning                                             | N times the cap, held only while parsed; the callers are the agent's own tool calls. Over the cap: `413`, read stops at the cap, and `Content-Length` over it is refused before any byte is read. |
| v1.1 captured output          | 16 MiB (`MAX_RESPONSE_BYTES`) across stdout and stderr                                                                        | The agent refuses a bigger answer (`DEFAULT_MAX_RESPONSE_BYTES`), and a JSON string is never shorter than the bytes it encodes, so nothing the agent would have accepted is lost. Past it, output is read and dropped so the command never blocks, and the answer is a `422` that says the command ran and its output was too large; a prefix is never answered as the result.                                      | The answer being built                                            | N times 16 MiB at the worst                                                                                                                                                                       |
| v1.1 stdin                    | the decoded request body                                                                                                      | already counted in the request body                                                                                                                                                                                                                                                                                                                                                                                 | the handler returning                                             | as above                                                                                                                                                                                          |
| v1.2 stream buffers           | two 64 KiB read buffers, one frame in each of two depth-1 channels, hyper's write buffer (400 KiB, its default)               | A frame carries at most 64 KiB (the spec). Depth 1 is the least that lets the next frame be built while the last is written; nothing observable depends on it.                                                                                                                                                                                                                                                      | the stream ending                                                 | N streams hold N times a few hundred KiB. A stalled client holds the same, and stalls its command.                                                                                                |
| Settle on exit                | 100 ms quiet, or one pipe capacity of further bytes (`F_GETPIPE_SZ`, 64 KiB by default)                                       | The spec's grace is 100 ms (the client's). A command can only have written, and not been read, as much as its pipes hold.                                                                                                                                                                                                                                                                                           | the settle returning                                              | one timer per exiting stream                                                                                                                                                                      |
| Leftover poll                 | every 50 ms while a shell has exited and its group lives                                                                      | It only sets how soon the final frame follows the last member; it caps nothing.                                                                                                                                                                                                                                                                                                                                     | the group emptying, the deadline, or the client leaving           | one walk of execd's process tree per polling stream per tick, only while leftovers live                                                                                                           |
| Heartbeat                     | every 5 s of silence                                                                                                          | The spec.                                                                                                                                                                                                                                                                                                                                                                                                           | the stream ending                                                 | one timer per stream                                                                                                                                                                              |
| Stream lifetime               | the command's `timeout_ms` (a job stream has none)                                                                            | The spec: the endpoint enforces it and kills the group. A request without `timeout_ms`, or one too large to represent, waits 30 years, tokio's own "never".                                                                                                                                                                                                                                                         | the deadline, the client leaving, or the command ending           | bounded by the clients                                                                                                                                                                            |
| Job output                    | the last 8 MiB per job (`JOB_OUTPUT_RETENTION`), in whole chunks of at most 64 KiB, so between 8 MiB and 8 MiB plus one chunk | The spec: the endpoint keeps at least the last 8 MiB.                                                                                                                                                                                                                                                                                                                                                               | the job's `remove`, `kill` with `remove`, or `wipe`               | N jobs hold N times 8 MiB                                                                                                                                                                         |
| Job table                     | one entry per job started and not removed                                                                                     | The agent keeps at most `MAX_YIELDED_JOBS` (4) + `MAX_INTENDED_JOBS` (8) running and `MAX_ENDED_JOBS` (8) ended per session and removes them with the session, so a session holds at most 20 entries. execd adds no second limit; an ended job whose client never removes it stays until execd exits.                                                                                                               | `remove`, `wipe`, `kill` with `remove`                            | N sessions hold N times 20 entries, 160 MiB at the floor                                                                                                                                          |
| Job `read`                    | at most the output held                                                                                                       | A read cannot ask for more than the ring has.                                                                                                                                                                                                                                                                                                                                                                       | the answer being sent                                             | N answers of at most 8 MiB plus a chunk, base64 encoded                                                                                                                                           |
| Job pumps                     | two tasks per job                                                                                                             | One per pipe. They end when their pipes close, when the job is removed, or when the supervisor lets go of a pipe a descendant holds open after the settle.                                                                                                                                                                                                                                                          | as stated                                                         | 2N tasks for N running jobs                                                                                                                                                                       |
| Job supervisor                | one task per job                                                                                                              | Ends when the shell has exited, its group is empty and the end is recorded.                                                                                                                                                                                                                                                                                                                                         | the end                                                           | N tasks for N running jobs                                                                                                                                                                        |
| Callback attempts             | one in flight per job, 30 s each, 12 per count                                                                                | N16, N18: a count is over within 723 s however the receiver behaves.                                                                                                                                                                                                                                                                                                                                                | the verdict, the 12th failure, a new `keep`, or the job's removal | N attempts at once, one request each, none pooled                                                                                                                                                 |
| Callback client               | one `reqwest` client, built on the first attempt, with no idle connection kept                                                | A deployment that never keeps a job with `notify` never builds it. A token never sits on a pooled socket.                                                                                                                                                                                                                                                                                                           | process exit                                                      | built once                                                                                                                                                                                        |
| Process groups registered     | one pid per running command or job                                                                                            | The OS bounds how many processes can run (`pids.max`).                                                                                                                                                                                                                                                                                                                                                              | the command or job ending                                         | N entries of 4 bytes                                                                                                                                                                              |
| Owned pids                    | one per child execd spawned and still awaits                                                                                  | as above                                                                                                                                                                                                                                                                                                                                                                                                            | the child's handle dropping                                       | N entries of 4 bytes                                                                                                                                                                              |
| Connections                   | one task and one socket each; not capped by execd                                                                             | The file-descriptor limit ends it. An accept error pauses the loop for 100 ms instead of spinning. A connection with no complete request head for 60 s is closed (twice the agent's 30 s idle pool timeout, so the client never loses a race to it). A peer whose data goes unacknowledged for two heartbeat intervals (10 s, the spec's bound on a vanished client) is dropped by the kernel (`TCP_USER_TIMEOUT`). | the client closing, the head timeout, or the user timeout         | bounded by the clients and the descriptor limit                                                                                                                                                   |
| HTTP/2 streams per connection | 200                                                                                                                           | hyper's default; not set by execd                                                                                                                                                                                                                                                                                                                                                                                   | the stream ending                                                 | as above                                                                                                                                                                                          |
| Logs                          | one line per request, written when it finishes                                                                                | A stream logs once, when it ends.                                                                                                                                                                                                                                                                                                                                                                                   | the log sink                                                      | one short line each                                                                                                                                                                               |
| Observer                      | none unless the host sets one                                                                                                 | An unset observer costs one branch per event and builds none of its arguments.                                                                                                                                                                                                                                                                                                                                      | n/a                                                               | n/a                                                                                                                                                                                               |

### What a stalled client costs

A client that opens a stream and stops reading holds the buffers above for as long as its command lives, and no more: the
command blocks on a full pipe and writes nothing further. The command's own `timeout_ms` ends it, as does the client closing, and
the kernel drops a peer that stopped acknowledging after 10 s of unsent heartbeat. `tests/endpoint.rs` runs a command that
writes 256 MiB against a client that never reads and asserts it stalls and stays stalled.

### How it is tested against a plain process

`tests/differential.rs` generates seeded shell programs (arbitrary bytes on both streams, sizes from 0 to several MiB, stdin,
exit codes, self-kills, background children) and runs each directly with `std::process::Command` and through a real server over
v1.1 and v1.2. Stdout, stderr, exit code and signal must agree per stream; the spec's allowances (lossy UTF-8 in v1.1, `128 + S`
for a signal in v1.1, a late background write possibly lost in v1.1) are encoded in the comparison. A failure prints its seed
(`EXECD_DIFF_SEED`, `EXECD_DIFF_CASES`) and a shrunk program.

## Deviations from the reference mock

The reference mock in `crates/test-support` (`exec_mock`) now runs this code for its conforming configuration. It was a test
double, and these are the places this crate differs from it on purpose:

- A real HTTP stack (hyper) replaces a hand-rolled request parser, so HTTP/1.1 keep-alive, chunked request bodies and HTTP/2
  work.
- Liveness and kills are syscalls; the mock spawned `kill`.
- Channels are bounded; the mock's were unbounded, so a slow client grew its memory without limit.
- A v1.1 command is killed after it has answered, as the spec says it SHOULD be. The mock left it running.
- A command killed by a signal answers `exit_code` 128 + the signal in v1.1, as `shell_exit` does; the mock answered `-1`.
- v1.1 output is capped at what the agent would accept (`--max-response-bytes`), and past the cap the answer is a `422` saying the command ran; the mock buffered all of it.
- Job output is a ring of whole chunks, not a `Vec` drained on every read of the pipe.
- A job's table entry is the claim on its id, taken before the process is spawned, so two starts of one id cannot both run.
- The orphan reaper, the subreaper, the signal-driven shutdown and the request limit exist only here.

## Performance

execd against the endpoint it replaces, measured by `benches/exec_compare.rs` over HTTP on loopback. **A** is
`beyond-ai-demo exec-stub` built from `origin/main` at 0b1ac87b, before execd (the reference mock); **B** is `execd` from this
branch. Both are release builds started with the same flags on fresh root and home directories. The run is 5 rounds; in each,
A and B run the whole battery back to back, in alternating order (AB, BA, AB, ...), so drift in the load of this shared host
lands on both. Spreads below are the lowest and highest per-round median (latencies) or the lowest and highest round
(throughput, CPU). Date 2026-10-25, AMD Ryzen 7 255 (16 threads), Linux 6.12, a host running other work (load average 7 to 9).
Per-round summaries (n, p50, p90, p99, max, mean) are in `benches/results/2026-10-25-sandbox.json` and `benches/results/2026-10-25-host.json`; `EXEC_BENCH_RAW=1` keeps every sample, outside the repository.

To rerun: build the baseline from a checkout of the commit before execd (`cargo build --release -p beyond-ai-demo`), then

```sh
EXEC_BENCH_A=/path/to/beyond-ai-demo mise run bench:execd:sandbox   # the numbers below, first table
EXEC_BENCH_A=/path/to/beyond-ai-demo mise run bench:execd           # on the host as it is, second table
```

`bench:execd:sandbox` runs the same harness in a fresh PID namespace, where `/proc` lists a few processes, as it does in the
sandbox execd is built for. The baseline answers every request with `Connection: close`, so it cannot hold a kept-alive
connection; its sequential round trips include a connect on each request, and the "new connection each" row gives execd the
same.

### Sandbox-like host (few processes in `/proc`)

| metric                                          | N (each side) | A: reference mock         | B: execd                  |
| ----------------------------------------------- | ------------- | ------------------------- | ------------------------- |
| v1.1 `true`, sequential p50 / p99 / max         | 2000          | 0.53 / 1.04 / 11.6 ms     | 0.55 / 1.05 / 1.2 ms      |
| v1.1 `cat` of 4 KiB, sequential p50 / p99 / max | 2000          | 0.63 / 1.23 / 10.5 ms     | 0.64 / 1.19 / 1.7 ms      |
| v1.1 `true`, new connection each p50 / p99      | 2000          | 0.57 / 1.07 ms            | 0.63 / 1.07 ms            |
| v1.1 `true`, 16 concurrent p50 / p99 / max      | 3200          | 0.89 / 5.3 / 36 ms        | 2.8 / 53 / 128 ms         |
| v1.1 `cat` of 4 KiB, 16 concurrent p50 / p99    | 3200          | 1.06 / 2.3 ms             | 2.7 / 56 ms               |
| time to first frame p50 / p99 / max             | 200           | 2.25 / 2.94 / 13.5 ms     | 0.49 / 1.54 / 1.9 ms      |
| kill on disconnect, whole group gone p50 / p99  | 50            | 1.20 / 2.13 ms            | 1.12 / 2.16 ms            |
| stream `/dev/zero`, 1 GiB, decoded MB/s         | 5             | 954 (740 to 1088)         | 1396 (426 to 1409)        |
| stream random file, 1 GiB, decoded MB/s         | 5             | 958 (809 to 1026)         | 1396 (491 to 1461)        |
| server CPU per GB, zeros                        | 5             | 2.08 s (1.89 to 2.50)     | 1.29 s (1.29 to 1.77)     |
| server CPU per GB, random                       | 5             | 1.88 s (1.74 to 2.23)     | 1.14 s (1.09 to 1.66)     |
| RSS idle after start                            | 2             | 5.9 MiB                   | 5.9 MiB                   |
| RSS with 100 idle streams (`sleep 30`)          | 2             | 15.3 MiB                  | 11.9 MiB                  |
| peak RSS, 16 slow readers of 256 MiB streams    | 1             | 9838 MiB (VmHWM 9920 MiB) | 33.5 MiB (VmHWM 33.6 MiB) |

Read this table as follows.

- The sequential v1.1 round trip is a tie: the median differs by 0.02 ms and the per-round medians overlap (A 0.44 to 0.75, B
  0.52 to 0.68). B's slowest request was 1.2 ms; A's was 11.6 ms.
- Streaming is faster and cheaper: B moves about 1.4 GB/s against about 1.0 GB/s and spends about 40% less server CPU per GB.
  The throughput minimums (426 and 491 MB/s) are single rounds that overlapped another session's load on the host; the other
  four rounds are 1246 to 1461 MB/s.
- A slow reader shows backpressure: 16 clients reading 256 MiB streams at about 1 MB/s each make A buffer them in memory (9.8
  GiB at the sample peak) and leave B flat at 33.5 MiB.
- Concurrent v1.1 was **slower** in B when this table was measured: 2.8 ms against 0.9 ms at the median, and a p99 of 53 ms
  against 5 ms (one of B's five rounds was hit by other work on the host: its median was 22.7 ms, the other four 2.5 to 3.2
  ms). The cause was `proc::spawn` holding a global lock across the spawn, so concurrent commands spawned one at a time. It is
  fixed; see below.
- Kill on disconnect is a tie at about 1.1 ms (the harness polls every 0.5 ms on a 1 ms timer, so differences under 1 ms are
  not resolved). The spec allows 10 s.

### What a busy `/proc` cost, and the fix

The first measurement (above, and the "before" columns below) found execd's round trip growing with the number of processes on
the host. The orphan reaper scanned all of `/proc` on every `SIGCHLD`, holding the `OWNED` lock for the whole scan; the next
spawn and the exiting command's own `Proc` drop both take that lock, so the scan sat on the critical path of every request, at
about 10 us for each process visible. With the reaper's task disabled, a sequential `true` on this host fell from 24.2 ms to
0.45 ms, and `strace` showed the response landing right after the scan's `getdents64` and per-pid `stat` reads. The same scan
ran at 20 Hz for every stream whose command had left a process behind. The fix is in "Process model" (the reaper and the
liveness check read execd's own children; spawns hold a lock shared and the reaper holds it exclusively only around one
`waitpid`).

Before and after, same harness, 5 rounds with A and B interleaved. After: 2026-10-25, `benches/results/2026-10-25-*-reaper.json`.
Before: `benches/results/2026-10-25-{sandbox,host}.json`. Each cell is B's p50 / p99 in ms against the reference mock A in the
same run; N is per side.

| `true` sequential                   | N    | A, after    | B, before   | B, after    |
| ----------------------------------- | ---- | ----------- | ----------- | ----------- |
| 2 processes in `/proc`              | 1000 | 0.76 / 1.48 | 0.55 / 1.07 | 0.44 / 0.90 |
| plus 100 idle `sleep` processes     | 1000 | 0.45 / 1.08 | 1.67 / 4.1  | 0.43 / 0.88 |
| plus 1000 idle `sleep` processes    | 1000 | 0.43 / 0.98 | 10.2 / 29.5 | 0.41 / 0.88 |
| this host as it is (2471 processes) | 2000 | 0.54 / 1.00 | 25.5 / 53.3 | 0.45 / 1.10 |

| host as it is (2471 processes) | N    | A, after   | B, before   | B, after    |
| ------------------------------ | ---- | ---------- | ----------- | ----------- |
| `true`, 16 concurrent          | 3200 | 0.88 / 3.4 | 55.3 / 106  | 0.87 / 1.7  |
| `cat` of 4 KiB, 16 concurrent  | 3200 | 1.10 / 3.6 | 57.0 / 108  | 1.07 / 2.2  |
| time to first frame            | 200  | 1.89 / 2.9 | 25.7 / 53.1 | 0.43 / 0.88 |

| `true`, 16 concurrent, 2 processes in `/proc` | N    | A, after   | B, before | B, after   |
| --------------------------------------------- | ---- | ---------- | --------- | ---------- |
| p50 / p99                                     | 3200 | 0.93 / 7.6 | 2.8 / 53  | 0.95 / 2.2 |

Reading these: after the fix B is within the noise of A at every process count (the per-round medians overlap, or B is lower),
and the concurrency-16 gap is closed on a small `/proc` too, which confirms the cause was the lock held across the spawn (and
the scan under it), not anything in the request path. Streaming, RSS and kill-on-disconnect did not change (host run after the
fix: 1418 MB/s zeros and 1441 MB/s random against A's 1042 and 1074; 1.30 and 1.11 server CPU-s per GB against A's 2.09 and
1.79; peak RSS with 16 slow readers 34 MiB against A's 10.0 GiB).

A stream whose command exited leaving a process running (a kept dev server) is polled every 50 ms for the group to empty. The
server's CPU over 30 s with one such stream, on this host (2471 processes), N=1 per side:

| reference mock A | execd before | execd after |
| ---------------- | ------------ | ----------- |
| 1.25 s           | 15.86 s      | 0.25 s      |

Before, that was more than half a core spent for as long as the dev server lived. After, it is a tenth of a percent of a core
(the walk of execd's few descendants every 50 ms, 0.25 s in 30 s) and does not depend on the host's process count; it is not
zero, and the reference mock spends five times as much. The `children` files need `CONFIG_PROC_CHILDREN` in the kernel (it
exists on this 6.12 kernel); where they are missing, `/proc` is read as before, correct but at the old cost. This was not
checked on the Fargate or E2B kernels.
