# execd architecture

execd is the exec endpoint that runs inside a sandbox. The agent POSTs a command to a URL; execd runs it as a real process and
answers. It implements the whole of the agent's exec protocol: v1.1 (one request, one JSON answer, stdin), v1.2 streaming,
v1.3 sandbox jobs and v1.4 job-end notify. The normative spec is the module doc of `crates/agent/src/exec_endpoint.rs`; this
file says how execd meets it and what bounds every resource it holds.

It is one crate: a library (`execd`) that embeds in a test or a larger host, and a thin `execd` binary (`--listen`, `--root`,
`--home`, `--bearer-file`, `--insecure-no-auth`, `--run-as`, `--kept-jobs-survive-idle`, `--log-level`).

## Modules

| file            | role                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `src/lib.rs`    | `Options`, the `Observer` trait, `serve` and `Handle`, and the derived limits                   |
| `src/main.rs`   | the CLI: flags, bearer sources and the environment scrub, fail-closed start, subreaper, signals |
| `src/runas.rs`  | `--run-as`: resolving the user, proving execd can switch, the switch helper                     |
| `src/http.rs`   | hyper (HTTP/1.1 and h2c), routing, `/livez`, the bearer check, the request limit, the log line  |
| `src/exec.rs`   | v1.1 envelope and v1.2 stream: pumps, the sink, heartbeat, leftovers, timeout, disconnect       |
| `src/jobs.rs`   | v1.3 jobs: the table, the output ring, the supervisor, job streams                              |
| `src/notify.rs` | v1.4: the sender that drives the `exec-notify` kernel                                           |
| `src/proc.rs`   | process groups by syscall, the spawn registry, the subreaper and orphan reaper                  |
| `src/frame.rs`  | v1.2 frame encoding                                                                             |

## Request flow

```text
accept ──► hyper (HTTP/1.1, or prior-knowledge HTTP/2 on the same port)
              │
   /livez ◄───┤ GET or HEAD only: 200, no auth
              ▼
   not POST ──► 405
              ▼
   POST ──► bearer check (one header, constant time) ──► 401
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

## Security

execd is remote code execution by design: whoever holds its bearer can run any command in the sandbox. Security here means
three things, in this order, and nothing else:

1. **No request without the bearer runs anything.**
2. **The bearer cannot leak**, including to the commands execd itself runs, because those may be prompt-injected or malicious
   repository code.
3. **One job scope can never see or affect another's jobs.**

Out of scope: denial of service (the bounds above are for memory, not for an adversary), except where a panic or a poisoned
lock would break 1 to 3 or the kill guarantees; the network (the bearer crosses it in the clear unless the listener is behind
TLS or a private network); and a holder of the bearer doing what the bearer allows.

### Controls

| control                | guarantees                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | does not guarantee                                                                                                                                                                                                                                                | tested by                                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| The bearer check       | Every request but `GET`/`HEAD /livez` is refused (`401`, or `405` for a method that cannot exec) before its body is read, unless it carries exactly one `Authorization` header whose value is `Bearer`, one space, and the token (compared in constant time). The path does not matter: `//livez`, `/livez/`, `/%6civez` are paths like any other, and a `POST` to `/livez` is an exec request that needs the bearer. Two `Authorization` headers are refused in either order. The scheme is case-sensitive (`bearer <token>` is refused, as is `Bearer  <token>`). `Proxy-Authorization`, cookies, query strings and forwarding headers are never read. | HTTP/1.1 strips optional whitespace around a header value before execd sees it, so `Bearer <token>` followed by a space or tab on the wire is the same credential and runs; over HTTP/2 nothing is stripped and it is refused. There is no rate limit and no TLS. | `tests/auth_matrix.rs`: about 180 requests over HTTP/1.1 and h2c, each answered and each proven to run nothing    |
| Fail-closed start      | With no bearer, execd refuses to start (non-zero exit, a message) unless every address in `--listen` is loopback, or `--insecure-no-auth` is passed. An empty `EXECD_BEARER` is no bearer; a named bearer file that is empty or unreadable is an error. With `--insecure-no-auth` or on loopback it starts and logs `NO BEARER CONFIGURED`. The library does the same: `serve` returns `PermissionDenied` for a non-loopback listener with no bearer unless `Options::insecure_no_auth` is set.                                                                                                                                                          | Loopback is not private on a multi-tenant host: any local user can connect.                                                                                                                                                                                       | `tests/security.rs`; the loopback decision is a unit test in `src/main.rs`                                        |
| The token stays secret | The bearer is never an argument. Every `EXECD_*` variable is removed from every command's environment (v1.1, v1.2, jobs). A bearer handed over in `EXECD_BEARER` is moved out of execd's own environment by re-executing it (see below). No log line, `Debug` output or error carries the bearer, a notify token or a callback URL.                                                                                                                                                                                                                                                                                                                      | A process that runs as execd's uid can still read execd's memory and a bearer file execd owns. The observer an embedding host sets sees every request's headers, `Authorization` included; the host is trusted.                                                   | `tests/security.rs`, `tests/conformance.rs` (the whole verifier at `trace` level, then a search of its log)       |
| `--run-as`             | Commands run as another uid, gid and groups, so they cannot read execd's memory, `/proc/<execd>/environ`, or a 0600 bearer file owned by execd's uid, and cannot signal or trace execd.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | A setuid program the user can run (`sudo`, `su`) can still become root, and then reads everything. A process that leaves its process group (`setsid`) outlives a kill of the group, as it does without `--run-as`.                                                | `tests/security.rs` (as root), `tests/conformance.rs` (the whole verifier under `--run-as`)                       |
| Job scopes             | A job is addressed by the exact pair `(scope, id)`. Every operation filters by its own scope before it looks at an id, so another scope's id reads as `gone`/`404` and an answer, a listing or an error never names, orders or times anything of another scope. `kill` and `wipe` only signal groups of their own scope's jobs that have not ended.                                                                                                                                                                                                                                                                                                      | A scope is a name, not a secret: a client with the bearer can ask for any scope it can name. The agent sends a digest of tenant and session, which is the capability; there is no operation that lists scopes.                                                    | `tests/jobtable.rs`: random operations across scopes with colliding ids, a model, every invariant after each step |
| No panic, no poison    | Every lock is taken with `locked`, which recovers a poisoned mutex (the data under each is a map or a plain record), so one panicking handler cannot stop a kill, a list or the shutdown sweep. After the sweep the group set is closed: a group registered by a spawn that raced the shutdown is killed at once.                                                                                                                                                                                                                                                                                                                                        | A panic still ends that one request.                                                                                                                                                                                                                              | `tests/jobtable.rs` (the log never says `panicked`, every later request works), `tests/parsers.rs`, `fuzz/`       |

### The bearer and the environment

`/proc/<pid>/environ` shows the environment a process was **started** with; removing a variable from a running process does not
change it. So a bearer delivered in `EXECD_BEARER` would sit in execd's `environ` for any command that runs as execd's uid to
read, however carefully execd keeps it from the commands' own environment. execd therefore starts by writing the token to a
memfd and re-executing itself without the variable (`EXECD_BEARER_FD` names the descriptor, which the new image reads and closes
before it parses a flag or spawns anything). The new image's `environ` never held the token. Delivery by environment is still
the weaker choice: the launcher's own environment, `docker inspect`, a process supervisor's unit file and the first instants
before the re-execution all hold it. **Prefer `--bearer-file`** (mode 0600; execd warns when others can read it), and **use
`--run-as`** when the commands are not trusted: without it a command is the same uid as execd and can read the token from execd's
memory (`/proc/<execd>/mem` is refused only where the kernel's Yama `ptrace_scope` is 1 or more) and from a bearer file execd
owns.

### `--run-as`

`--run-as <user|uid[:gid]>` (or `user:group`) makes every command run as that identity, with the user's groups from the group
database (`setgroups`, then `setresgid`, then `setresuid`, all three ids of each). execd keeps its own uid, which must be root or
hold `CAP_SETUID`, `CAP_SETGID` and `CAP_KILL` (it signals processes it does not own). It refuses to start unless it has proved
it can: a helper switches to the target and execd signals it, and the target must not be execd's uid. The workspace forbids
`unsafe`, which rules out `pre_exec`, and `std`'s supplementary-group setter is unstable, so the switch is made by `execd`
itself: a command is spawned as `/proc/self/exe --execd-run-as-helper run <uid> <gid> <groups> <cwd> <command> <args>...`, which
switches, proves the switch cannot be undone (`setresuid(0)` must fail), changes into the request's directory **as the target**
(so the target's permissions decide whether it may enter it) and `exec`s the command. The helper is the command's own process:
the pid, the process group and the pipes are the ones execd spawned, so every kill, timeout and disconnect path is unchanged. Any
failure in the switch exits 126 before the command is reached.

What the spec and the conformance verifier assume, and what holds:

- **uid.** Neither says anything about the command's uid; the verifier passes unchanged under `--run-as`
  (`the_binary_passes_every_check_under_run_as`).
- **`HOME`.** `--home` if given, otherwise the user's own home from the passwd database (execd's would be root's).
- **`cwd` and `--root`.** A command with no `cwd` runs in `--root`, which must be writable by the target. execd says at startup
  when it is not (`--root ... is not writable by uid ...`), and a command whose directory the target cannot enter exits 127 with
  `Permission denied`, as it would for any user.
- **Jobs.** The supervisor stays execd (root) and signals the user's groups; `kill`, `wipe`, a dropped hold and the shutdown
  sweep all work (`tests/security.rs` kills a user's job by `SIGTERM` to execd).
- **A missing program** is exit 127 with `No such file or directory`, as without `--run-as`. A job whose program is missing is
  a started job that ends 127, not a `500` at `start`.
- **argv.** For the instant between the spawn and the `exec`, a command's `/proc/<pid>/cmdline` is the helper's: the uid, gid,
  groups, directory, command and arguments. The bearer is never among them.

### What is deliberately not done

- No rate limiting, lockout or TLS: put execd behind the network boundary that has them.
- No `no_new_privs`: it would break `sudo` in a sandbox image that wants it.
- A kill is of a process group. A command that daemonizes out of its group (`setsid`) survives the kill of its job, a timeout and
  the shutdown sweep, with or without `--run-as`; a cgroup per job would close that, and execd does not own one.
- A capability held by execd as an _ambient_ capability would reach the commands (`--run-as` from a non-root execd with ambient
  `CAP_SETUID`): run execd as root, or with those capabilities permitted and effective only.

### How it is tested

| test                                            | proves                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/security.rs`                             | the bearer is in none of a command's `env`, its parent's environ, or any readable `/proc/*/environ` (by environment, by file, for streams and jobs); fail-closed start in all its cases; `--run-as` refusals; as root, a user who cannot read execd's memory, environ or a root-owned bearer file, gets the right uid, groups, `HOME` and directory, and is killed by the shutdown |
| `tests/auth_matrix.rs`                          | path, method, `Authorization`, body-framing and smuggling cases over HTTP/1.1 and h2c, each refused and each proven to have run nothing by a command that would create a marker file; a control per family proves the vehicle runs                                                                                                                                                 |
| `tests/jobtable.rs`                             | the job table against a model, over random operations (`EXECD_JOBS_SEED`, `EXECD_JOBS_CASES`, `EXECD_JOBS_SECS`, `EXECD_JOBS_OPS`; a failure prints its seed and a shrunk list)                                                                                                                                                                                                    |
| `tests/parsers.rs`, `fuzz/` (`request`, `ring`) | no body panics the parsers, every refused body is a `4xx`, and the output ring agrees with a plain buffer for every offset and size                                                                                                                                                                                                                                                |
| `tests/conformance.rs`                          | the full verifier, also at `trace` level (then its log is searched for the bearer) and under `--run-as`                                                                                                                                                                                                                                                                            |

The tests that need root run execd through `sudo -n` (or as root) and print `SKIPPED` where a host gives them neither. The long
runs (ten minutes of job-table seeds, two of parser seeds, four of each fuzz target) are in `.github/workflows/deep.yml`.

## Process model

- **Groups.** Every command and job is the leader of a new process group. Liveness is `kill(-pgid, 0)`; a kill is `killpg`. Both
  are syscalls (the `nix` crate; the workspace forbids `unsafe`). `kill(2)` counts zombies, and the spec says a zombie is not a
  live member, so a group that `kill(-pgid, 0)` reports as present is then read from `/proc` and only non-zombie members count.
  A group that is gone is never signalled again, because its number may be reused.
- **Orphans.** The binary makes itself the child subreaper and runs `reap_orphans`: on every `SIGCHLD` it scans `/proc` for
  zombie children of its own and `waitpid`s them. It reaps only pids it did not spawn. Every spawn registers its pid in a set
  under a lock the scan also takes, so a command being spawned is never seen unregistered, and a command execd awaits is never
  reaped out from under its owner. There is no `waitpid(-1)`.
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
| Leftover poll                 | every 50 ms while a shell has exited and its group lives                                                                      | It only sets how soon the final frame follows the last member; it caps nothing.                                                                                                                                                                                                                                                                                                                                     | the group emptying, the deadline, or the client leaving           | one `/proc` scan per polling stream per tick, only while leftovers live                                                                                                                           |
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
