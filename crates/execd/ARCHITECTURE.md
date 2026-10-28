# execd architecture

execd is the exec endpoint that runs inside a sandbox. The agent POSTs a command to a URL; execd runs it as a real process and
answers. It implements the whole of the agent's exec protocol: one-shot (one request, one JSON answer: stdin and the output are base64), streaming,
sandbox jobs and job-end notify. The normative spec is the module doc of `crates/sandbox/src/exec_endpoint.rs`; this
file says how execd meets it and what bounds every resource it holds.

It is one crate: a library (`execd`) that embeds in a test or a larger host, and a thin `execd` binary configured by environment
variables alone ([Configuration](#configuration)).

## Modules

| file            | role                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| `src/lib.rs`    | `Options`, the `Observer` trait, `serve` and `Handle`, and the derived limits                            |
| `src/main.rs`   | the binary: bearer sources and the environment scrub, fail-closed start, subreaper, signals              |
| `src/config.rs` | the binary's settings: every `EXECD_*` variable, parsed once at startup, and the usage text              |
| `src/log.rs`    | the logger: one line per event to stderr, a level and a format (text or JSON), no subscriber             |
| `src/runas.rs`  | `EXECD_RUN_AS`: resolving the user, proving execd can switch, the switch helper                          |
| `src/http.rs`   | hyper (HTTP/1.1 and h2c), routing, `/livez`, the bearer check, the request limit, the log line           |
| `src/exec.rs`   | one-shot envelope and streams: pumps, the sink, heartbeat, leftovers, timeout, disconnect                |
| `src/jobs.rs`   | jobs: the table, the output ring, the supervisor, job streams                                            |
| `src/notify.rs` | notify: the sender that drives the `exec-notify` kernel                                                  |
| `src/proc.rs`   | spawning, the reaper that collects every child, process groups by syscall, the subreaper                 |
| `src/spool.rs`  | the jobs directory: where a kept job's retained output lives as chunk files, and stale-directory cleanup |
| `src/frame.rs`  | streaming frame encoding                                                                                 |

## Configuration

execd has no command-line flags. It is configured by `EXECD_*` environment variables, each read once at startup in
`src/config.rs` and nowhere else: `EXECD_LISTEN`, `EXECD_ROOT`, `EXECD_HOME`, `EXECD_BEARER_FILE`, `EXECD_BEARER`,
`EXECD_INSECURE_NO_AUTH`, `EXECD_RUN_AS`, `EXECD_MAX_RESPONSE_BYTES`, `EXECD_JOBS_DIR`,
`EXECD_LOG`, `EXECD_WORKER_THREADS` and `EXECD_IDLE_TIMEOUT_MS` (the table is in the README). A value that does not parse is an error
naming the variable and the value, with exit status 2; a boolean is `1`, `true`, `0` or `false`; an empty variable is unset.
Any argument other than the internal `--execd-run-as-helper` re-exec marker (handled before anything else) prints the variable
list to stderr and exits 2. There is no `--help` or `--version`: the startup line carries the version.

Why not flags: a flag parser was 284 KiB of the resident memory of a process whose floor is under a megabyte, and a sandbox is
configured by its launcher's environment anyway (a Dockerfile `ENV`, a task definition, a supervisor unit). The settings are not
secrets, with one exception, so they stay in execd's own `/proc/<pid>/environ`; the bearer does not (see below). No `EXECD_*`
variable reaches a command's environment.

## Request flow

```text
accept ──► hyper (HTTP/1.1, or prior-knowledge HTTP/2 on the same port)
              │
   /livez ◄───┤ GET or HEAD only (also /healthz): 200, no auth
              ▼
   not POST ──► 405
              ▼
   POST ──► bearer check (one header, constant time), or `Options::authorizer` ──► 401
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
 (jobs, and the         (streaming: the connection     (one-shot: run to the end,
  notify `keep`)           is the command's life)    one JSON object)
```

A request that is none of those fails before a process exists. A command that cannot be spawned is an ordinary result (exit
127 when the program or the `cwd` is not there, 126 when the program is there and cannot be run, as a shell reports them) whose stderr names what failed, as the caller sent it (`execd: pytest -q: command not found`, `execd: cwd /x: No such
file or directory`): `exec::spawn_failure` for the direct path, the helper's own `exec` failure under `EXECD_RUN_AS`.

### one-shot: `exec::envelope`

The command runs as the leader of a new process group. `timeout_ms` is enforced here too (the spec's "Timeouts"): at the
deadline the whole group is killed and the answer is `200` with `exit_code` 124 (`timeout(1)`'s status), the output read so far, and
stderr ending in `command exceeded its timeout`. A request without `timeout_ms` has no deadline but the connection. Its stdout and stderr are read into buffers capped at 16 MiB across both
(`EXECD_MAX_RESPONSE_BYTES`; see the bounds table). The two pipes reserve from that one cap by an atomic compare-and-exchange, so a runtime with more than one worker cannot count the same remainder twice and then keep every later byte. If output went past the cap it was drained and dropped, and the answer is a `422` (the agent treats 400 and 422 as a refusal of this request, not a dead sandbox) whose message says the command ran and its output exceeded the cap, so the model narrows it instead of rerunning. The answer is sent when the command exits and the pipes have been read, with a 100 ms grace for pipes
that a descendant holds open. The group is then killed (the spec's SHOULD for one-shot), and it is killed if the client leaves
first.

**Output is bytes.** `stdout` and `stderr` are always standard base64 of the bytes the command wrote, both always present
(the empty string for none). The cap is on the raw bytes across both pipes, so it is on the decoded bytes, as the spec says.
`answer` is the one place that builds the object, and it never converts. The note a deadline appends to stderr is appended
to the bytes. `tests/output.rs` pins every byte value on both streams, the empty answer, the deadline, a command that
cannot start, stdin (and a stdin that is not base64: `400`) and the cap; the verifier's `every byte value round-trips`
check runs against the binary in `tests/conformance.rs`.

### streaming: `exec::stream`

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

### jobs and notify: `jobs`, `notify`

A job is a command in its own process group whose output is kept by byte offset. `start` registers the job and spawns a
supervisor task; `hold: true` answers with a job stream, `hold: false` answers `{"started": pgid}`. Every operation is scoped:
it only ever sees the jobs of its own scope.

The supervisor waits for the shell, and when the shell exits leaving its group alive records `shell_exit`, waits for the group to
empty, and records the end. Each of those is a state change that notify calls back about.

The supervisor keeps the shell's real wait status, not only a number: when a signal killed the shell the job records `signal: S`
beside `code` 128 + S, set in the same step as `shell_exit` or the end, so no reader can see a code without its signal. `read` and
`list` answer `signal` only for such a job (absent for an exit, even of 137), and the job stream's final frame is `{"signal": S}`
in place of `{"exit_code": 128 + S}`, as a one-shot stream's is. A job stopped by `kill` therefore reports the signal that
killed its shell (15, or 9 after the grace). Nothing is persisted: a job lives in the table and is gone with execd.

notify is the `notify` module driving `crates/exec-notify`, the pure timing kernel every endpoint shares. The kernel says when an
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
| Fail-closed start      | With no bearer, execd refuses to start (non-zero exit, a message) unless every address in `EXECD_LISTEN` is loopback, or `EXECD_INSECURE_NO_AUTH=1` is passed. An empty `EXECD_BEARER` is no bearer; a named bearer file that is empty or unreadable is an error. With `EXECD_INSECURE_NO_AUTH=1` or on loopback it starts and logs `NO BEARER CONFIGURED`. The library does the same: `serve` returns `PermissionDenied` for a non-loopback listener with no bearer unless `Options::insecure_no_auth` is set.                                                                                                                                          | Loopback is not private on a multi-tenant host: any local user can connect.                                                                                                                                                                                       | `tests/security.rs`; the loopback decision is a unit test in `src/main.rs`                                        |
| The token stays secret | The bearer is never an argument. Every `EXECD_*` variable is removed from every command's environment (one-shot, streaming, jobs). A bearer handed over in `EXECD_BEARER` is moved out of execd's own environment by re-executing it, and execd exits before it listens when that move fails (see below). No log line, `Debug` output or error carries the bearer, a notify token or a callback URL.                                                                                                                                                                                                                                                                                                             | A process that runs as execd's uid can still read execd's memory and a bearer file execd owns. The observer an embedding host sets sees every request's headers, `Authorization` included; the host is trusted.                                                   | `tests/security.rs`, `tests/conformance.rs` (the whole verifier at `trace` level, then a search of its log)       |
| `EXECD_RUN_AS`         | Commands run as another uid, gid and groups, so they cannot read execd's memory, `/proc/<execd>/environ`, or a 0600 bearer file owned by execd's uid, and cannot signal or trace execd.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | A setuid program the user can run (`sudo`, `su`) can still become root, and then reads everything. A process that leaves its process group (`setsid`) outlives a kill of the group, as it does without `EXECD_RUN_AS`.                                            | `tests/security.rs` (as root), `tests/conformance.rs` (the whole verifier under `EXECD_RUN_AS`)                   |
| Job scopes             | A job is addressed by the exact pair `(scope, id)`. Every operation filters by its own scope before it looks at an id, so another scope's id reads as `gone`/`404` and an answer, a listing or an error never names, orders or times anything of another scope. `kill` and `wipe` only signal groups of their own scope's jobs that have not ended.                                                                                                                                                                                                                                                                                                      | A scope is a name, not a secret: a client with the bearer can ask for any scope it can name. The agent sends a digest of tenant and session, which is the capability; there is no operation that lists scopes.                                                    | `tests/jobtable.rs`: random operations across scopes with colliding ids, a model, every invariant after each step |
| No panic, no poison    | Every lock is taken with `locked`, which recovers a poisoned mutex (the data under each is a map or a plain record), so one panicking handler cannot stop a kill, a list or the shutdown sweep. After the sweep the group set is closed: a group registered by a spawn that raced the shutdown is killed at once.                                                                                                                                                                                                                                                                                                                                        | A panic still ends that one request.                                                                                                                                                                                                                              | `tests/jobtable.rs` (the log never says `panicked`, every later request works), `tests/parsers.rs`, `fuzz/`       |

### The bearer and the environment

`/proc/<pid>/environ` shows the environment a process was **started** with; removing a variable from a running process does not
change it. So a bearer delivered in `EXECD_BEARER` would sit in execd's `environ` for any command that runs as execd's uid to
read, however carefully execd keeps it from the commands' own environment. execd therefore starts by writing the token to a
memfd and re-executing itself without the variable (`EXECD_BEARER_FD` names the descriptor, which the new image reads and closes
before it spawns anything). The new image's `environ` never held the token. If the memfd cannot be created or the re-exec fails, execd exits before it listens, because serving would leave the token in `/proc/<pid>/environ`. Delivery by environment is still
the weaker choice: the launcher's own environment, `docker inspect`, a process supervisor's unit file and the first instants
before the re-execution all hold it. **Prefer `EXECD_BEARER_FILE`** (mode 0600; execd warns when others can read it), and **use
`EXECD_RUN_AS`** when the commands are not trusted: without it a command is the same uid as execd and can read the token from execd's
memory (`/proc/<execd>/mem` is refused only where the kernel's Yama `ptrace_scope` is 1 or more) and from a bearer file execd
owns.

### `EXECD_RUN_AS`

`EXECD_RUN_AS=<user|uid[:gid]>` (or `user:group`) makes every command run as that identity, with the user's groups from the group
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

- **uid.** Neither says anything about the command's uid; the verifier passes unchanged under `EXECD_RUN_AS`
  (`the_binary_passes_every_check_under_run_as`).
- **`HOME`.** `EXECD_HOME` if given, otherwise the user's own home from the passwd database (execd's would be root's).
- **`cwd` and `EXECD_ROOT`.** A command with no `cwd` runs in `EXECD_ROOT`, which must be writable by the target. execd says at startup
  when it is not (`EXECD_ROOT ... is not writable by uid ...`), and a command whose directory the target cannot enter exits 127 with
  `Permission denied`, as it would for any user.
- **Jobs.** The supervisor stays execd (root) and signals the user's groups; `kill`, `wipe`, a dropped hold and the shutdown
  sweep all work (`tests/security.rs` kills a user's job by `SIGTERM` to execd).
- **A missing program** is exit 127 with `execd: PROGRAM: command not found`, and one that cannot be run (`Permission denied`) is 126, as without `EXECD_RUN_AS`. A job whose program is missing is
  a started job that ends 127, not a `500` at `start`.
- **argv.** For the instant between the spawn and the `exec`, a command's `/proc/<pid>/cmdline` is the helper's: the uid, gid,
  groups, directory, command and arguments. The bearer is never among them.

### What is deliberately not done

- No rate limiting, lockout or TLS: put execd behind the network boundary that has them.
- Only `EXECD_*` is scrubbed from a command's environment; the rest is inherited, as a sandbox expects. A launcher that puts the
  bearer on a command line leaks it that way: `sudo` records the command it runs in `SUDO_COMMAND`, which every command below
  it inherits. Hand the token over in a file, or in the environment, never as an argument.
- No `no_new_privs`: it would break `sudo` in a sandbox image that wants it.
- A kill is of a process group. A command that daemonizes out of its group (`setsid`) survives the kill of its job, a timeout and
  the shutdown sweep, with or without `EXECD_RUN_AS`; a cgroup per job would close that, and execd does not own one.
- A capability held by execd as an _ambient_ capability would reach the commands (`EXECD_RUN_AS` from a non-root execd with ambient
  `CAP_SETUID`): run execd as root, or with those capabilities permitted and effective only.

### How it is tested

| test                                            | proves                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/security.rs`                             | the bearer is in none of a command's `env`, its parent's environ, or any readable `/proc/*/environ` (by environment, by file, for streams and jobs); fail-closed start in all its cases, including a bearer that cannot be moved out of the environment; `EXECD_RUN_AS` refusals; as root, a user who cannot read execd's memory, environ or a root-owned bearer file, gets the right uid, groups, `HOME` and directory, and is killed by the shutdown |
| `tests/auth_matrix.rs`                          | path, method, `Authorization`, body-framing and smuggling cases over HTTP/1.1 and h2c, each refused and each proven to have run nothing by a command that would create a marker file; a control per family proves the vehicle runs                                                                                                                                                     |
| `tests/jobtable.rs`                             | the job table against a model, over random operations (`EXECD_JOBS_SEED`, `EXECD_JOBS_CASES`, `EXECD_JOBS_SECS`, `EXECD_JOBS_OPS`; a failure prints its seed and a shrunk list); at the end of every case, once every scope is wiped, execd is back at its starting state (see `tests/baseline.rs`)                                                                                    |
| `tests/baseline.rs`                             | after a mixed workload (one-shot, streams finished and dropped, jobs past the 8 MiB spool, kills, removes, a wipe), `/proc` shows the starting open fds and threads, no zombie child, and an empty jobs directory                                                                                                                                                                      |
| `tests/parsers.rs`, `fuzz/` (`request`, `ring`) | no body panics the parsers, every refused body is a `4xx`, and the output ring agrees with a plain buffer for every offset and size                                                                                                                                                                                                                                                    |
| `tests/conformance.rs`                          | the full verifier, also at `debug` level (then its log is searched for the bearer) and under `EXECD_RUN_AS`                                                                                                                                                                                                                                                                            |
| `tests/output.rs`, `tests/timeout.rs`           | one-shot output is bytes (every byte value on both streams, the empty answer, the deadline note, stdin, the cap on decoded bytes) and `timeout_ms` kills the group and answers 124                                                                                                                                                                                                     |

Every test that starts the binary owns it from the moment it exists, and a detached watchdog shell kills execd and everything under it if the test process dies without running destructors (`tests/guard.rs` SIGKILLs a test mid-run, as user and as root, and looks for survivors). The tests that need root run execd through `sudo -n` (or as root) and print `SKIPPED` where a host gives them neither. The long
runs (ten minutes of job-table seeds, two of parser seeds, four of each fuzz target) are in `.github/workflows/deep.yml`.

## Process model

- **Groups.** Every command and job is the leader of a new process group. Liveness is `kill(-pgid, 0)`; a kill is `killpg`. Both
  are syscalls (the `nix` crate; the workspace forbids `unsafe`). `kill(2)` counts zombies, and the spec says a zombie is not a
  live member. Where execd is the subreaper it reaps every child at once, so a zombie can outlast an instant only while its
  parent is alive; if that parent is in the group the group is alive, and the one case left is a zombie whose live parent is
  outside the group. So a group that `kill(-pgid, 0)` reports as present counts as alive when a `SIGCHLD`-driven wait asks, and
  the slow tick (5 s) reads `/proc` to find out whether any live member remains. That read is the bytes of `/proc/<pid>/stat`, split after the last `)`. The command name inside the parentheses is a raw `comm`, which may contain spaces, parentheses, and bytes that are not UTF-8; such a process still counts as a live member. Without the subreaper (an embedding host,
  whose orphans go to init) every check reads `/proc`. A group that is gone is never signalled again, because its number may be
  reused. No kernel feature is needed: the children files (`CONFIG_PROC_CHILDREN`) that an earlier version walked are absent
  from the Fargate and E2B kernels, and are not used.
- **Reaping.** execd reaps every child itself, like an init, and tokio's own child handling is never used (it would race).
  A command is spawned with std's `Command`, which never waits for a child, its pipes become async descriptors, and its exit
  status comes down a oneshot channel. The reaper task runs a pass at start and after every `SIGCHLD`: it calls `waitpid` until
  nothing is left, looks each pid up in the table of children execd spawned and hands the status to its waiter. Where execd is
  the subreaper (the binary) the pass is `waitpid(-1)`, so an adopted orphan is reaped the moment it exits and a pid not in the
  table is an orphan, reaped and forgotten; elsewhere (a host that embeds `execd::serve` and owns other children) the pass
  collects only the pids in the table, so it never takes another's exit status. A pass makes a few `waitpid` calls and reads
  nothing from `/proc`, whatever the host's process count and however many short processes a build starts. Every spawn holds
  the `SPAWNING` lock shared from before the child exists until its pid is in the table; a pass holds it exclusively. So a pass
  sees every spawn entirely before it (the child is in the table, and its status goes to its waiter) or entirely after it (the
  child does not exist yet); the pid lookup is inside the pass because a pid is reused as soon as it is collected. Spawns do
  not wait for each other; with one thread the lock is never contended, and it stays for a host that runs `execd::serve` on a
  multi-thread runtime.
  `std::sync::RwLock` has no writer preference, so in a host that embeds `execd::serve` in a multi-thread runtime, many
  concurrent spawns can delay a reaper pass; that is liveness only, and impossible in the shipped one-thread binary.
- **Runtime.** One thread: a tokio `current_thread` runtime. execd moves bytes between sockets and pipes and waits on timers and
  child exits; the little CPU work it has (copying, base64) is about 1 GB/s on one core, more than a sandbox's commands write.
  `EXECD_WORKER_THREADS` sets it: unset or `1` is one thread, `0` one worker per CPU, `N` N workers. The reaper's `waitpid` passes
  run inline, since they take microseconds; only a scan of `/proc` goes to the blocking pool. The spawn and reap locks are
  still needed (a host can embed `execd::serve` in a multi-thread runtime, as the test mock does) and cost two uncontended
  atomics per spawn.
- **Shutdown.** `SIGTERM` or `SIGINT` stops the accept loop, drops every open connection (which drops every in-flight handler, and
  with it the guard that kills that handler's group), then kills every group still registered, commands and jobs alike, and
  exits.
- **Embedding.** The library never sets a subreaper. `serve` starts its own reaper (it listens for `SIGCHLD`); a host that
  wants the PID 1 duties calls `become_subreaper` first, as the binary does.

## Bounds

Every buffer, map, file, task and connection execd holds, what bounds it, what frees it, and what happens under N concurrent
cold callers. No limit here is invented: each is the spec's, the agent's own, or a number read from the kernel.

| what                          | bound                                                                                                                                                                                 | derivation                                                                                                                                                                                                                                                                                                                                                                                                          | freed by                                                                          | N concurrent cold callers                                                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Request body                  | 91,575,640 bytes (`MAX_REQUEST_BYTES`)                                                                                                                                                | The agent's largest request is a file's content as `stdin_base64`: the biggest file it writes is an edit's, `MAX_EDIT_BYTES` = 64 MiB, which is `ceil(64 MiB / 3) * 4` = 89,478,488 bytes of base64. Plus 2 MiB, Linux's default `ARG_MAX`, for the command and arguments the kernel itself refuses past.                                                                                                           | The handler returning                                                             | N times the cap, held only while parsed; the callers are the agent's own tool calls. Over the cap: `413`, read stops at the cap, and `Content-Length` over it is refused before any byte is read. |
| one-shot captured output      | 16 MiB (`MAX_RESPONSE_BYTES`) across stdout and stderr                                                                                                                                | The agent refuses a bigger answer (`DEFAULT_MAX_RESPONSE_BYTES`), and a JSON string is never shorter than the bytes it encodes, so nothing the agent would have accepted is lost. Past it, output is read and dropped so the command never blocks, and the answer is a `422` that says the command ran and its output was too large; a prefix is never answered as the result. Stdout and stderr take from that one counter by compare-and-exchange, so concurrent readers cannot both keep the same remainder.                                      | The answer being built                                                            | N times 16 MiB at the worst                                                                                                                                                                       |
| one-shot stdin                | the decoded request body                                                                                                                                                              | already counted in the request body                                                                                                                                                                                                                                                                                                                                                                                 | the handler returning                                                             | as above                                                                                                                                                                                          |
| streams buffers               | two 64 KiB read buffers, one frame in each of two depth-1 channels, hyper's write buffer (400 KiB, its default)                                                                       | A frame carries at most 64 KiB (the spec). Depth 1 is the least that lets the next frame be built while the last is written; nothing observable depends on it.                                                                                                                                                                                                                                                      | the stream ending                                                                 | N streams hold N times a few hundred KiB. A stalled client holds the same, and stalls its command.                                                                                                |
| Settle on exit                | 100 ms quiet, or one pipe capacity of further bytes (`F_GETPIPE_SZ`, 64 KiB by default)                                                                                               | The spec's grace is 100 ms (the client's). A command can only have written, and not been read, as much as its pipes hold.                                                                                                                                                                                                                                                                                           | the settle returning                                                              | one timer per exiting stream                                                                                                                                                                      |
| Leftover wait                 | until a `SIGCHLD`, with a 5 s fallback, while a shell has exited and its group lives                                                                                                  | The reaper collects execd's children at once, so the last member of a group is nearly always collected and a waiter is woken for it; the slow tick (a heartbeat period) covers a zombie whose live parent is neither execd nor in the group, and only sets how late that is noticed. Without a subreaper (an embedding host) it polls every 50 ms.                                                                  | the group emptying, the deadline, or the client leaving                           | one `/proc` scan per 5 s tick per waiting stream, only while leftovers live                                                                                                                       |
| Heartbeat                     | every 5 s of silence                                                                                                                                                                  | The spec.                                                                                                                                                                                                                                                                                                                                                                                                           | the stream ending                                                                 | one timer per stream                                                                                                                                                                              |
| Stream lifetime               | the command's `timeout_ms` (a job stream has none)                                                                                                                                    | The spec: the endpoint enforces it and kills the group. A request without `timeout_ms`, or one too large to represent, waits 30 years, tokio's own "never".                                                                                                                                                                                                                                                         | the deadline, the client leaving, or the command ending                           | bounded by the clients                                                                                                                                                                            |
| Job output                    | the last 8 MiB per job (`JOB_OUTPUT_RETENTION`), in whole chunks of 64 KiB **on disk** (`EXECD_JOBS_DIR`), so between 8 MiB and 8 MiB plus one chunk; in memory, one unfinished chunk | The spec: the endpoint keeps at least the last 8 MiB.                                                                                                                                                                                                                                                                                                                                                               | a chunk's eviction; the job's `remove`, `kill` with `remove`, or `wipe`; shutdown | N jobs hold N times 8 MiB of files                                                                                                                                                                |
| Job table                     | one entry per job started and not removed                                                                                                                                             | The agent keeps at most `MAX_YIELDED_JOBS` (4) + `MAX_INTENDED_JOBS` (8) running and `MAX_ENDED_JOBS` (8) ended per session and removes them with the session, so a session holds at most 20 entries. execd adds no second limit; an ended job whose client never removes it stays until execd exits.                                                                                                               | `remove`, `wipe`, `kill` with `remove`                                            | N sessions hold N times 20 entries, 160 MiB of files at the floor                                                                                                                                 |
| Job `read`                    | at most the output held                                                                                                                                                               | A read cannot ask for more than the ring has.                                                                                                                                                                                                                                                                                                                                                                       | the answer being sent                                                             | N answers of at most 8 MiB plus a chunk, base64 encoded                                                                                                                                           |
| Job pumps                     | two tasks per job                                                                                                                                                                     | One per pipe. They end when their pipes close, when the job is removed, or when the supervisor lets go of a pipe a descendant holds open after the settle.                                                                                                                                                                                                                                                          | as stated                                                                         | 2N tasks for N running jobs                                                                                                                                                                       |
| Job supervisor                | one task per job                                                                                                                                                                      | Ends when the shell has exited, its group is empty and the end is recorded.                                                                                                                                                                                                                                                                                                                                         | the end                                                                           | N tasks for N running jobs                                                                                                                                                                        |
| Callback attempts             | one in flight per job, 30 s each, 12 per count                                                                                                                                        | N16, N18: a count is over within 723 s however the receiver behaves.                                                                                                                                                                                                                                                                                                                                                | the verdict, the 12th failure, a new `keep`, or the job's removal                 | N attempts at once, one request each, none pooled                                                                                                                                                 |
| Callback client               | one `reqwest` client, built on the first attempt, with no idle connection kept                                                                                                        | A deployment that never keeps a job with `notify` never builds it. A token never sits on a pooled socket.                                                                                                                                                                                                                                                                                                           | process exit                                                                      | built once                                                                                                                                                                                        |
| Process groups registered     | one pid per running command or job                                                                                                                                                    | The OS bounds how many processes can run (`pids.max`).                                                                                                                                                                                                                                                                                                                                                              | the command or job ending                                                         | N entries of 4 bytes                                                                                                                                                                              |
| Children table                | one entry per child execd spawned and has not yet seen exit                                                                                                                           | The OS bounds how many processes can run (`pids.max`).                                                                                                                                                                                                                                                                                                                                                              | the reaper collecting the child                                                   | N entries (a pid and a oneshot sender)                                                                                                                                                            |
| Connections                   | one task and one socket each; not capped by execd                                                                                                                                     | The file-descriptor limit ends it. An accept error pauses the loop for 100 ms instead of spinning. A connection with no complete request head for 60 s is closed (twice the agent's 30 s idle pool timeout, so the client never loses a race to it). A peer whose data goes unacknowledged for two heartbeat intervals (10 s, the spec's bound on a vanished client) is dropped by the kernel (`TCP_USER_TIMEOUT`). | the client closing, the head timeout, or the user timeout                         | bounded by the clients and the descriptor limit                                                                                                                                                   |
| HTTP/2 streams per connection | 200                                                                                                                                                                                   | hyper's default; not set by execd                                                                                                                                                                                                                                                                                                                                                                                   | the stream ending                                                                 | as above                                                                                                                                                                                          |
| Logs                          | one line per request, written when it finishes                                                                                                                                        | A stream logs once, when it ends.                                                                                                                                                                                                                                                                                                                                                                                   | the log sink                                                                      | one short line each                                                                                                                                                                               |
| Observer                      | none unless the host sets one                                                                                                                                                         | An unset observer costs one branch per event and builds none of its arguments.                                                                                                                                                                                                                                                                                                                                      | n/a                                                                               | n/a                                                                                                                                                                                               |

### What a stalled client costs

A client that opens a stream and stops reading holds the buffers above for as long as its command lives, and no more: the
command blocks on a full pipe and writes nothing further. The command's own `timeout_ms` ends it, as does the client closing, and
the kernel drops a peer that stopped acknowledging after 10 s of unsent heartbeat. `tests/endpoint.rs` runs a command that
writes 256 MiB against a client that never reads and asserts it stalls and stays stalled.

### How it is tested against a plain process

`tests/differential.rs` generates seeded shell programs (arbitrary bytes on both streams, sizes from 0 to several MiB, stdin,
exit codes, self-kills, background children) and runs each directly with `std::process::Command` and through a real server over
one-shot and streaming. Stdout, stderr, exit code and signal must agree per stream; the spec's allowances (`128 + S`
for a signal in one-shot, a late background write possibly lost in one-shot) are encoded in the comparison. A failure prints its seed
(`EXECD_DIFF_SEED`, `EXECD_DIFF_CASES`) and a shrunk program.

### What the machine failing costs

`tests/faults.rs` runs the real binary and applies each fault to it: a full tmpfs under the jobs directory (mounted with
`sudo -n`; `SKIPPED` without it), `RLIMIT_NOFILE` and `RLIMIT_NPROC` lowered on the live process with `prlimit`.

- **Disk full.** A kept job keeps running. Output that cannot be written reads back as a gap (`offset` moves forward, never back,
  and the bytes at it are the bytes written), a refused chunk leaves no file behind, and other jobs, one-shot and streaming are unaffected.
  Output written after space returns is retained again.
- **No descriptors, no processes.** A command that cannot be spawned is an ordinary result on one-shot (`exit_code` 126 with the
  reason) and streaming (a `stderr` frame, then 126); a job that cannot start is `500` and leaves no job behind. `/livez` and the jobs
  already running are unaffected, and everything works again, with no descriptor leaked, once the limit clears. With no descriptor
  left to accept with, a request waits in the kernel's queue rather than failing. The `/proc` scans that run off the runtime
  (`proc::off_runtime`) fall back to running in place when no thread can be created, since `tokio::task::spawn_blocking` panics there.

## Deviations from the reference mock

The reference mock in `crates/test-support` (`exec_mock`) now runs this code for its conforming configuration. It was a test
double, and these are the places this crate differs from it on purpose:

- A real HTTP stack (hyper) replaces a hand-rolled request parser, so HTTP/1.1 keep-alive, chunked request bodies and HTTP/2
  work.
- Liveness and kills are syscalls; the mock spawned `kill`.
- Channels are bounded; the mock's were unbounded, so a slow client grew its memory without limit.
- A one-shot command is killed after it has answered, as the spec says it SHOULD be. The mock left it running.
- A command killed by a signal answers `exit_code` 128 + the signal in one-shot, as `shell_exit` does; the mock answered `-1`.
- one-shot output is capped at what the agent would accept (`EXECD_MAX_RESPONSE_BYTES`), and past the cap the answer is a `422` saying the command ran; the mock buffered all of it.
- Job output is a set of chunk files, not a `Vec` drained on every read of the pipe (see "Job output on disk").
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

To rerun: build the baseline from a checkout of the commit before execd (`cargo build --release -p beyond-demo`), then

```sh
EXEC_BENCH_A=/path/to/beyond-ai-demo mise run bench:execd:sandbox   # the numbers below, first table
EXEC_BENCH_A=/path/to/beyond-ai-demo mise run bench:execd           # on the host as it is, second table
```

`bench:execd:sandbox` runs the same harness in a fresh PID namespace, where `/proc` lists a few processes, as it does in the
sandbox execd is built for. The baseline answers every request with `Connection: close`, so it cannot hold a kept-alive
connection; its sequential round trips include a connect on each request, and the "new connection each" row gives execd the
same.

### Sandbox-like host (few processes in `/proc`)

| metric                                              | N (each side) | A: reference mock         | B: execd                  |
| --------------------------------------------------- | ------------- | ------------------------- | ------------------------- |
| one-shot `true`, sequential p50 / p99 / max         | 2000          | 0.53 / 1.04 / 11.6 ms     | 0.55 / 1.05 / 1.2 ms      |
| one-shot `cat` of 4 KiB, sequential p50 / p99 / max | 2000          | 0.63 / 1.23 / 10.5 ms     | 0.64 / 1.19 / 1.7 ms      |
| one-shot `true`, new connection each p50 / p99      | 2000          | 0.57 / 1.07 ms            | 0.63 / 1.07 ms            |
| one-shot `true`, 16 concurrent p50 / p99 / max      | 3200          | 0.89 / 5.3 / 36 ms        | 2.8 / 53 / 128 ms         |
| one-shot `cat` of 4 KiB, 16 concurrent p50 / p99    | 3200          | 1.06 / 2.3 ms             | 2.7 / 56 ms               |
| time to first frame p50 / p99 / max                 | 200           | 2.25 / 2.94 / 13.5 ms     | 0.49 / 1.54 / 1.9 ms      |
| kill on disconnect, whole group gone p50 / p99      | 50            | 1.20 / 2.13 ms            | 1.12 / 2.16 ms            |
| stream `/dev/zero`, 1 GiB, decoded MB/s             | 5             | 954 (740 to 1088)         | 1396 (426 to 1409)        |
| stream random file, 1 GiB, decoded MB/s             | 5             | 958 (809 to 1026)         | 1396 (491 to 1461)        |
| server CPU per GB, zeros                            | 5             | 2.08 s (1.89 to 2.50)     | 1.29 s (1.29 to 1.77)     |
| server CPU per GB, random                           | 5             | 1.88 s (1.74 to 2.23)     | 1.14 s (1.09 to 1.66)     |
| RSS idle after start                                | 2             | 5.9 MiB                   | 5.9 MiB                   |
| RSS with 100 idle streams (`sleep 30`)              | 2             | 15.3 MiB                  | 11.9 MiB                  |
| peak RSS, 16 slow readers of 256 MiB streams        | 1             | 9838 MiB (VmHWM 9920 MiB) | 33.5 MiB (VmHWM 33.6 MiB) |

Read this table as follows.

- The sequential one-shot round trip is a tie: the median differs by 0.02 ms and the per-round medians overlap (A 0.44 to 0.75, B
  0.52 to 0.68). B's slowest request was 1.2 ms; A's was 11.6 ms.
- Streaming is faster and cheaper: B moves about 1.4 GB/s against about 1.0 GB/s and spends about 40% less server CPU per GB.
  The throughput minimums (426 and 491 MB/s) are single rounds that overlapped another session's load on the host; the other
  four rounds are 1246 to 1461 MB/s.
- A slow reader shows backpressure: 16 clients reading 256 MiB streams at about 1 MB/s each make A buffer them in memory (9.8
  GiB at the sample peak) and leave B flat at 33.5 MiB.
- Concurrent one-shot was **slower** in B when this table was measured: 2.8 ms against 0.9 ms at the median, and a p99 of 53 ms
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
zero, and the reference mock spends five times as much. That version walked the kernel's `children` files, which need `CONFIG_PROC_CHILDREN`; the Fargate (6.1.188) and E2B (6.1.158)
kernels do not have it, so there it fell back to reading `/proc` on every `SIGCHLD`. It is replaced; see "Reaping without
`/proc`".

### Memory

execd runs in every sandbox, so its resident memory is a per-VM cost. Everything here is the shipped artifact: a static musl,
non-PIE (`-C relocation-model=static`), release build, for both the reference mock (A) and execd (B), measured by
`benches/memory.py` (smaps_rollup PSS in KiB, medians of 3 fresh servers, `PR_SET_THP_DISABLE` on the server, so the host's
THP=always does not enter) and `benches/exec_compare.rs`. Raw summaries: `benches/results/2026-10-25-memory.json`. The 16-thread
host differs from a VM; "2 CPUs" rows pin the server with `taskset -c 0,1`.

**Idle, and what each cut is worth** (PSS KiB; each row is one change from the row above unless it says otherwise):

| build                                              | idle PSS | threads | change                                 |
| -------------------------------------------------- | -------- | ------- | -------------------------------------- |
| Rust floor: `current_thread` + hyper, `ok` reply   | 972      | 1       | `examples/floor.rs`, built as execd is |
| C reference: busybox `httpd` (dynamic glibc, host) | 492      | 1       | RSS is 2144; shared libc counts part   |
| reference mock (A)                                 | 3548     | 17      |                                        |
| execd before this change (multi-thread)            | 3104     | 17      |                                        |
| (a) `current_thread`                               | 2844     | 1       | -260                                   |
| (b) lazy notify client                             | 2844     | 1       | 0: already lazy (`OnceCell`)           |
| (c) no log subscriber at all                       | 2352     | 1       | -492, an upper bound; not shipped      |
| (d) (c) and no clap                                | 2068     | 1       | -284 more; not shipped                 |
| (f) PGO, on top of (a)                             | 2544     | 1       | -300 (not combined with (c) and (d))   |
| (e) mimalloc, arena alloc disabled, on top of (a)  | 3252     | 1       | +408; default arena multi-thread: 4044 |

(a) ships. (b) was already true: the callback client is built on the first notify attempt, so nothing of reqwest, rustls or ring
runs at startup. (c) and (d) are bounds, measured by removing the code, not by a replacement: an `EnvFilter` pulls in a regex
engine, and clap's parser accounts for the 284 KiB of (d). Both are now replaced: `src/log.rs` is a one-knob logger and
`src/config.rs` reads environment variables, with no flags at all (see "Footprint of the replacements" below). (f) ships in the
distribution build: `release-execd.yml` and `Dockerfile.execd` follow `release-agent.yml`, and `.github/pgo-train-execd.py` trains it
(re-measured on the shipped musl binary, n=3: 2728 to 2544 KiB, -185). It belongs there, not in a cargo profile. On 2 CPUs the
multi-thread build idles at 2960 and `current_thread` at 2852 (-108), because tokio then starts 2 workers, not 16.

**Footprint of the replacements** (no clap, no tracing-subscriber, 16 KiB frames, the stale-directory scan off the serving path;
static musl release builds with `-C relocation-model=static` of origin/main and of this change, both pinned to 2 CPUs,
`benches/memory.py --n 5 --no-thp`, and `exec_compare` over 3 rounds with `EXEC_BENCH_CPUS=0,1`):

| (PSS KiB unless noted)                          | origin/main | this change |
| ----------------------------------------------- | ----------- | ----------- |
| idle                                            | 3048        | 2372 (-22%) |
| 100 idle streams (`sleep 30`)                   | 8616        | 7228        |
| per idle stream                                 | 55.7        | 48.6        |
| binary size, bytes                              | 6,199,600   | 5,005,840   |
| single-stream throughput, 2 CPUs, median MB/s   | 326 to 302  | 775 to 712  |
| single-stream throughput, 2 CPUs, slowest round | 292         | 311         |
| server CPU per GB streamed, median s            | 2.4 to 2.6  | 1.2 to 1.4  |

The throughput rows are noisy: this host is shared and the rounds spread by 2.5x on both sides, so read them as "not slower,
and well above the 100 MB/s floor in every round", not as a speed-up. The idle and per-stream rows repeat to within 1%.

**Marginal cost** (medians of 3, in the JSON):

| what                                            | reference (A) | execd, musl malloc       | execd, mimalloc (no arena) |
| ----------------------------------------------- | ------------- | ------------------------ | -------------------------- |
| one idle stream (`sleep 30`), PSS KiB, over 100 | 105.7         | 57.0                     | 37.4                       |
| one kept job that wrote 9 MiB, PSS KiB, over 20 | 8304          | 130 (was 9501 in memory) | 8535 (in memory)           |

A stream's 57 KiB is far below the nominal sizes in the Bounds table (a 400 KiB hyper write buffer, 64 KiB read buffers): an
idle command has written nothing, so those pages are not resident. I did not itemize the 57 KiB by buffer.

**Job output on disk.** A kept job's retained output was anonymous memory: 20 jobs that each wrote more than 8 MiB held about 190
MB (PSS) in execd, 9.3 MiB each, which cannot be reclaimed without swap. It is now chunk files, so the same 20 jobs hold 5.5 MB
(130 KiB each, n=3, `benches/results/2026-10-25-memory.json`), of which 2.8 MB is idle execd; the 160 MiB of output is 20 x 8 MiB of
files, page cache that the kernel can reclaim. A job that has written less than a chunk (64 KiB) touches no disk at all.

| 20 kept jobs, 9 MiB each        | execd PSS | anonymous | job output on disk |
| ------------------------------- | --------- | --------- | ------------------ |
| in memory (before)              | 192.1 MB  | 189.1 MB  | none               |
| files, jobs dir on ext4         | 5.5 MB    | 2.6 MB    | 160 MiB            |
| files, jobs dir on tmpfs (/tmp) | 5.6 MB    | 2.8 MB    | 160 MiB, as RAM    |

On tmpfs the files are RAM, as before, only outside execd's own accounting: execd says so at startup and suggests a disk-backed
`EXECD_JOBS_DIR`.

Reading and writing, one 64 MiB job (8 MiB retained), 2 CPUs, 3 runs:

|                                                                     | in memory (before) | files on ext4                   |
| ------------------------------------------------------------------- | ------------------ | ------------------------------- |
| `read` of 8 MiB from the oldest offset, MB/s (3 runs)               | 167, 181, 187      | 169, 187, 166                   |
| `attach` from the oldest offset, MB/s decoded, approximate (3 runs) | 830, 1301, 830     | 566, 548, 580                   |
| `true` p50 while 16 jobs write 1 GiB flat out                       | 7.5, 9.9, 27 ms    | 0.48, 0.48, 0.51 ms             |
| the same, p99                                                       | 119 to 126 ms      | 224 to 234 ms (125 ms on tmpfs) |
| `true`, idle server, p50                                            | 0.44 to 0.46 ms    | 0.46 to 0.47 ms                 |

`read` is bound by JSON and base64, not by where the bytes are. `attach` loses about a third of its peak, still far above any
command's output. The chunk writes run inline on the loop, not on the blocking pool: one chunk is a create and a 64 KiB write at
45 us p50 (p99 120 us, worst 0.3 ms) and an unlink at 16 us (worst 1.5 ms) on this ext4, about what encoding the same chunk for the
wire costs, and a blocking-pool thread would cost a stack and a hand-off per chunk. With 16 jobs writing flat out the loop stays
responsive (p50 0.5 ms, where the in-memory ring starved it at 7 to 27 ms, because the writers are slowed instead of the loop);
the p99 doubles to 230 ms, against 125 ms on tmpfs, which is the disk's tail latency on a saturated 2-CPU host and not a per-chunk
cost.

**Where the files are and who can read them.** `EXECD_JOBS_DIR` (default `$TMPDIR`, else `/tmp`) holds one
directory of execd's own, `execd-jobs-<uid>-<pid>-<random>`, mode 0700; under it one directory per job, named by a counter and
never by the client's id, and in it a file per chunk, mode 0600, created with `O_EXCL` and `O_NOFOLLOW` and read with
`O_NOFOLLOW`, so a command cannot plant a link that redirects a write or a read. Files go when their chunk is evicted, a job's
directory when the job is removed, and the whole directory at shutdown. A disk that refuses a chunk turns what was held and that
chunk into a gap, which is what a slow reader already reads. Without `EXECD_RUN_AS` a command runs as execd's uid and can read the
directory, as it can read execd's memory; with `EXECD_RUN_AS` the commands run as another uid and the 0700 directory is closed to
them. At startup the binary removes directories named like its own, owned by its uid, that are real directories (not links) and
name a pid that is no longer running, and nothing else; a killed execd leaves its directory (up to 160 MiB for 20 jobs) until the
next start.

**Burst** (16 slow readers of 256 MiB streams, 8 s, then dropped; 2 CPUs; n=2): the reference mock peaks at 4.4 GiB and settles
at 3.7 MiB. execd with musl malloc peaks at 20 MiB, and after the clients leave sits at 3.0 MiB (2.8 MiB before), unchanged 30 s
later: the memory returns. With mimalloc it peaks at 31 MiB and stays at 7.3 MiB (4.3 MiB of anonymous memory it does not give
back).

**Runtime and allocator, on throughput** (static musl, 3 rounds interleaved, a fast client, 1 GiB streams and 16 concurrent 256
MiB streams; decoded MB/s, medians; `true` at concurrency 16 is p50 ms):

| server                                | single stream | 16 streams, aggregate | server CPU-s/GB | `true` x16 p50 | first frame p50 |
| ------------------------------------- | ------------- | --------------------- | --------------- | -------------- | --------------- |
| multi-thread, unpinned (16 CPUs)      | 936           | 3237                  | 1.87            | 2.2 ms         | 0.51 ms         |
| `current_thread`, unpinned            | 732           | 657                   | 1.36            | 4.5 ms         | 0.51 ms         |
| multi-thread, 2 CPUs                  | 790           | 1047                  | 1.54            | 4.7 ms         | 0.50 ms         |
| `current_thread`, 2 CPUs              | 709           | 628                   | 1.37            | 4.7 ms         | 0.53 ms         |
| `current_thread`, musl malloc, 2 CPUs | 706           | 650                   | 1.36            | 4.8 ms         | 0.52 ms         |
| `current_thread`, mimalloc, 2 CPUs    | 1140          | 1084                  | 0.83            | 4.5 ms         | 0.42 ms         |

Verdict. `current_thread` saves 260 KiB at idle on this 16-thread host and 108 KiB on a 2-CPU VM (the 17 thread stacks are
mostly untouched, so the saving is small), costs 10 to 22% of single-stream rate (709 to 732 MB/s, far above what the commands of a sandbox write) and, on a many-CPU host, the 16-stream aggregate (657 against 3237 MB/s, still far
above a sandbox's output) and the concurrent-`true` latency (4.5 ms against 2.2 ms for 16 at once, spawn-bound on one thread).
On 2 CPUs the aggregate is 628 against 1047 and the latency is a tie. It uses less server CPU per GB in both. That is a small
memory win for a rate loss nothing in a sandbox reaches, so it ships as the default; `EXECD_WORKER_THREADS=0` restores a worker
per CPU. mimalloc is a large throughput win on one thread (+60%, -40% CPU per GB), because musl's malloc is the cost of the
64 KiB frame buffers, but it costs 408 KiB at idle and keeps 4 MiB after a burst, so it is not enabled; the `mimalloc` cargo
feature builds it. Reusing the frame buffer would take the allocator out of the stream path without either cost; that is a
change to the stream code and is not made here. With (c) and (d) done, nothing else in the process is as large.

**Wakeups** (context switches per second over a window, all threads, static musl build; `benches/wakeups.py`,
`benches/results/2026-10-25-wakeups.json`):

| state                                       | before (30 s) | after (60 s) |
| ------------------------------------------- | ------------- | ------------ |
| idle                                        | 2.2           | 0.0          |
| one open stream (`sleep 3600`)              | 2.5           | 0.6          |
| one stream whose shell left `sleep 3600` up | 36.3          | 0.6          |

Idle execd does not wake. The reaper slept one second between passes as a backstop, which was not needed: a `SIGCHLD` that
arrives while a pass runs is kept by tokio's signal stream and wakes the next wait. A stream's 0.6 is the spec's 5 s heartbeat.
A held leftover used to be polled at 20 Hz; it now sleeps until a `SIGCHLD` (the 0.6 is the heartbeat again) and looks every
5 s in case its last member's live parent is neither execd nor in the group. An embedding host without the subreaper still
polls every 50 ms, because its orphans go to init and nothing signals their exit.

### Startup and `EXECD_RUN_AS` cost

Two numbers a launcher and a prompt-injected-command defence each need. Static musl release build (`-C relocation-model=static`,
`musl-gcc`) of this change, on 2026-10-26 on an AMD Ryzen 7 255 (16 threads), Linux 6.12, a host running other work (load
average 55 to 62). The run is a one-off script, not in the repository.

**Startup to the first accepted connection** (spawn the binary, then connect in a loop until one succeeds; n=10 medians, repeated
5 times for `EXECD_BEARER_FILE` and 4 times for `EXECD_BEARER`, the range is of those medians):

| bearer                                   | median startup | range of medians |
| ---------------------------------------- | -------------- | ---------------- |
| `EXECD_BEARER_FILE`                      | 0.5 ms         | 0.45 to 0.83 ms  |
| `EXECD_BEARER` (execd re-execs to scrub) | 0.8 ms         | 0.66 to 0.83 ms  |

The environment form pays a second `exec` of the same binary, about 0.3 ms, to leave the token out of
`/proc/<pid>/environ`. Both are far below a sandbox's own boot, so use whichever the launcher can do; the file is the better one.

**`EXECD_RUN_AS` per command** (one-shot `true` over one kept-alive connection to each of two execd processes, as root through
`sudo -n`, one without `EXECD_RUN_AS` and one with it, n=2000 each, interleaved request by request; two runs):

| run | without run-as, median | with run-as, median | added by run-as |
| --- | ---------------------- | ------------------- | --------------- |
| 1   | 776 us                 | 1026 us             | 250 us          |
| 2   | 761 us                 | 1013 us             | 252 us          |

The 0.25 ms is the second `exec` (the helper that switches identity, then `exec`s the command), paid once per command and not
per byte. It does not touch an open stream or a job already running.

### Reaping without `/proc`

The children-file walk of the previous section does not exist on the production kernels (Fargate 6.1.188 and E2B 6.1.158 both
lack `CONFIG_PROC_CHILDREN`), so there execd read all of `/proc` after every `SIGCHLD`, and a build that leaves many
short-lived orphans made it do so thousands of times. The reaper now makes `waitpid` calls only (see "Process model",
"Reaping"). Measured with the children files forced absent, which is what those kernels are, against the same tree before this
change, on a host with about 2,500 processes (static musl, one thread):

| load                                                                                        | before               | after                |
| ------------------------------------------------------------------------------------------- | -------------------- | -------------------- |
| server CPU: a stream starts 5,000 orphans (`(/bin/true &)`), with a held leftover (n=3)     | 1.74 to 1.80 s       | 0.07 to 0.08 s       |
| server CPU per one-shot `true` on a kept-alive connection (n=2000, +0 and +2000 idle procs) | 720 to 755 us        | 135 to 140 us        |
| one-shot `true` p50 / p99, +0 idle processes                                                | 513-614 / 914-918 us | 555-584 / 881-889 us |
| one-shot `true` p50 / p99, +2000 idle processes                                             | 487-551 / 934 us     | 452-617 / 914-923 us |
| context switches per second: idle / one stream / a held leftover (30 s)                     | 0 / 0.4 / 0          | 0 / 0.6 / 0.7        |

Server CPU drops 22 times under the build-like load and 5 times per request; the round trips are the same, because the scan ran
on the blocking pool beside the request, not in front of it. The wakeups were already zero, because they are driven by the signal
and a 5 s tick, not by a scan. `benches/build_load.py` and `benches/per_request.py` rerun these.

## Embedding

The `execd` crate is the library the binary wraps, so another program can link it and serve the exec protocol itself. What the
binary does beyond `serve()` is process setup, and each piece is public:

| the binary does                                       | an embedder calls                                         | if it does not                                                      |
| ----------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------- |
| `helper_main()` first thing, before parsing anything  | `execd::helper_main()`; required with `run_as`            | `run_as` commands cannot start (the re-exec has no switch helper)   |
| `log::set_level` from `EXECD_LOG`                     | `execd::log::set_level(Level)`                            | the library logs nothing                                            |
| `become_subreaper()`, a warning when it fails         | `execd::become_subreaper()`                               | orphans go to the host's init; a waiter polls its group every 50 ms |
| `remove_stale_jobs_dirs` on a thread, after listening | `execd::remove_stale_jobs_dirs(jobs_dir)`                 | job output directories of a crashed predecessor stay on disk        |
| moves `EXECD_BEARER` out of `/proc/<pid>/environ`     | nothing: the binary alone has a bearer in its environment | n/a                                                                 |
| `serve(listener, opts)`, `shutdown()` on a signal     | `execd::serve`, `Handle::shutdown`                        |                                                                     |

A host that calls none of the setup is still correct: it only loses what the right-hand column says. `EXECD_*` variables in the
host's environment are scrubbed from commands, but the rest of the host's environment is inherited, so keep secrets out of it.

```rust
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

fn main() -> ExitCode {
    // First, before anything else, if commands run as another user (`Options::run_as`): as the
    // switch helper this process becomes a command and never returns here.
    if let Some(code) = execd::helper_main() {
        return code;
    }
    // The library logs nothing until told a level.
    execd::log::set_level(execd::log::Level::Info);
    let key = std::env::var("EMBED_KEY").expect("EMBED_KEY");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result: std::io::Result<()> = runtime.block_on(async move {
        // Orphans reparent to this process and are collected at once. Without it execd still
        // works, polling a command's group every 50 ms until it is empty.
        if let Err(e) = execd::become_subreaper() {
            execd::log_warn!("cannot become a subreaper: {e}");
        }
        let mut opts = execd::Options::new("/tmp");
        opts.authorizer = Some(Arc::new(move |headers: &execd::HeaderMap| {
            headers
                .get("x-host-key")
                .is_some_and(|v| v.as_bytes() == key.as_bytes())
        }));
        opts.idle_timeout = Some(Duration::from_secs(30));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
        let server = execd::serve(listener, opts)?;
        tokio::signal::ctrl_c().await?;
        server.shutdown().await;
        Ok(())
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            execd::log_error!("embedded execd failed: {e}");
            ExitCode::FAILURE
        }
    }
}
```

(`examples/embed.rs` is this program; clippy builds it.)

**Authentication.** `Options::authorizer` is `Arc<dyn Fn(&HeaderMap) -> bool + Send + Sync>`. With it set, every request except
`/livez` and `/healthz` runs if and only if it returns `true`, and gets `401` (as for a bad bearer) otherwise. It is called once
per request, before the body is read, inline on the connection's task, so it must be quick. Setting it together with `bearer` makes
`serve()` return `InvalidInput`. It counts as authentication for the fail-closed rule, so a non-loopback listener needs no bearer
with it. The binary cannot supply a closure and is unchanged.

**Idle connections.** `Options::idle_timeout` (`EXECD_IDLE_TIMEOUT_MS` for the binary; unset or 0 is off) closes a connection with
no request in flight for that long: HTTP/1.1 is closed, h2 gets a GOAWAY. A request is in flight from the moment hyper hands it to
execd until its response body is dropped, so a stream, a held job start or an `attach` counts for as long as it lasts, even while it
sends only heartbeats. Each connection has one counter and, only when the timeout is set, one timer armed for the remaining idle
time; unset, no timer exists and the connection loop is exactly what it was (an idle execd holding one idle connection made 0 context switches in 20 s unset, and 0 with a 100 s timeout set). This is for a VM that sleeps when it has no network
traffic, behind a proxy that pools connections with TCP keepalives.

**Activity.** `Handle::activity()` answers what the server is doing now, and `Handle::activity_changed()` is a `watch::Receiver<u64>`
that moves whenever that may have changed (a request starts or ends, a job starts, is kept, ends or is removed):

- `in_flight`: requests from handler start to the end of their response body. `/livez` and `/healthz` never count. It is the same
  guard the idle timeout counts with, summed over connections, so the two cannot disagree.
- `running_kept`: every job that was kept and has not ended, with its process group.
- `jobs_alive`: every job in the table that has not ended, in any scope, held or kept.
- `callbacks_owed`: the jobs whose notify notification is owed and not settled, as the `exec-notify` kernel defines it (`notify` set,
  `owed > settled`, not removed): an attempt in flight or a retry waiting on its backoff, from the moment a callback becomes owed
  until it is settled, abandoned at the attempt limit, or the job is removed. A host that pauses idle machines must stay awake
  while it is not zero, or a paused machine never sends the callback. Every kernel step bumps `activity_changed`.

A host that pauses an idle machine subscribes and re-reads:

```rust
let mut changed = server.activity_changed();
loop {
    changed.changed().await.ok();
    let a = server.activity();
    if a.in_flight == 0 && a.jobs_alive == 0 {
        // nothing is running: safe to let the machine sleep
    }
}
```

**Unknown job ops.** `Options::job_op_hook` is `Arc<dyn Fn(&str, &serde_json::Value) -> Option<(u16, serde_json::Value)>>`. It
receives the op name and the whole `job` object for an op execd does not implement, after authentication and parsing, never for a known
op, and its `Some((status, body))` is sent as a JSON response; `None` is the usual `400`. It is synchronous and runs on the server's
thread, so it must be quick.
