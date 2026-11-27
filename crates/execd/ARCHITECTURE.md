# execd architecture

execd is the exec endpoint that runs inside a sandbox. The agent POSTs a command to a URL; execd runs it as a real process and
answers. It implements the whole of the agent's exec protocol: one-shot (one request, one JSON answer: stdin and the output are base64), streaming,
sandbox jobs and job-end notify. The normative spec is the module doc of `crates/sandbox/src/exec_endpoint.rs`; this
file says how execd meets it and what bounds every resource it holds.

It is one crate: a library (`execd`) that embeds in a test or a larger host, and a thin `execd` binary configured by environment
variables alone ([Configuration](#configuration)).

## Modules

| file                | role                                                                                                                                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lib.rs`        | `Options`, the `Observer` trait, `serve` and `Handle`, and the derived limits                                                                                                                 |
| `src/main.rs`       | the binary: bearer sources and the environment scrub, fail-closed start, subreaper, signals                                                                                                   |
| `src/config.rs`     | the binary's settings: every `EXECD_*` variable, parsed once at startup, and the usage text                                                                                                   |
| `src/log.rs`        | the logger: one text line per event to stderr, level via `EXECD_LOG` / `log::set_level`, no subscriber                                                                                        |
| `src/fault.rs`      | the test knob `EXECD_TEST_FAULT` (debug builds only): where a keyed start's claim write fails, where execd exits after the spawn, where the ledger's append is cut short or its fold is split |
| `src/runas.rs`      | `EXECD_RUN_AS`: resolving the user, proving execd can switch, the switch helper                                                                                                               |
| `src/admit.rs`      | the count of requests in flight and its admission against `max_in_flight` (one compare-and-swap; loom models)                                                                                 |
| `src/http.rs`       | hyper (HTTP/1.1 and h2c), routing, `/livez`, the bearer check, the request limit, the log line                                                                                                |
| `src/term.rs`       | a terminal job's keeper (the process that holds the pty master), its session helper, the keeper's socket frames, and execd's connection to it                                                 |
| `src/exec.rs`       | one-shot envelope and streams: pumps, the sink, heartbeat, leftovers, timeout, disconnect                                                                                                     |
| `src/jobs.rs`       | jobs: the table, the supervisor, job streams                                                                                                                                                  |
| `src/ring.rs`       | the job output ring's I/O: chunk files, the tail, the bits' storage; every decision is `crates/execd-ring`                                                                                    |
| `crates/execd-ring` | the ring's offset, retention and stderr-label decisions, pure and proven in Verus (R1 to R6, `execd::ring`)                                                                                   |
| `crates/execd-jobs` | the job table's decisions, pure and proven in Verus (`execd::job_kernel`); `tests/job_model.rs` model-checks them                                                                             |
| `crates/execd-wire` | the heartbeat wait (W1), the one-shot's first answer (W2) and which processes a job's signal reaches (W3), pure and proven in Verus (`execd::wire_kernel`); `exec.rs` and `proc.rs` call them |
| `src/notify.rs`     | notify: the sender that drives the `exec-notify` kernel                                                                                                                                       |
| `src/proc.rs`       | spawning, the reaper that collects every child, process groups by syscall, the subreaper                                                                                                      |
| `src/spool.rs`      | the jobs directory: where a kept job's retained output lives as chunk files, and stale-directory cleanup                                                                                      |
| `src/tail.rs`       | a job's output spool: the two `O_APPEND` files its stdout and stderr go to, execd's tail of them (woken by one `inotify`), and the spool's directory                                          |
| `src/ledger.rs`     | the jobs ledger: the one typed value kept across a restart of execd — the table as one file and a log of the changes since, appended per life change and folded when it outgrows the table    |
| `src/frame.rs`      | streaming frame encoding                                                                                                                                                                      |

## Configuration

execd has no command-line flags. It is configured by `EXECD_*` environment variables, each read once at startup in
`src/config.rs` and nowhere else: `EXECD_LISTEN`, `EXECD_ROOT`, `EXECD_HOME`, `EXECD_BEARER_FILE`, `EXECD_BEARER`,
`EXECD_INSECURE_NO_AUTH`, `EXECD_RUN_AS`, `EXECD_MAX_RESPONSE_BYTES`, `EXECD_JOBS_DIR`, `EXECD_STATE_DIR`,
`EXECD_LOG`, `EXECD_WORKER_THREADS`, `EXECD_IDLE_TIMEOUT_MS`, `EXECD_DISK_WRITERS`, and the four retention knobs `EXECD_JOB_OUTPUT_MS`,
`EXECD_JOBS_ENDED_BYTES`, `EXECD_JOBS_ENDED_MAX` and `EXECD_JOB_ENTRY_MS` (the table is in the README; see "Ended jobs: retention"). A value that does not parse is an error
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
   (route_hook set: authorize, then the hook may claim the request, uncounted; see Embedding)
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

**The beat is the request's.** `Keepalive-Ping-Interval` (whole seconds) names a request's own heartbeat; `http::beat_of` reads it and
`execd_wire::beat_ms` (W1b, proven in Verus) decides: 1 to 240 s is the client's, anything else (absent, 0, negative, not an integer,
over 240) is `Options::heartbeat`. It rides on `Request::beat` to the one-shot envelope, the stream and a job's stream. 240 s is below the
gateway's 295 s, the edge's 300 s and the load balancer's 350 s; the front lengthens its own stall limit for the request. A longer
interval means a vanished client is noticed later by the heartbeat write, so the 10 s bound below holds only at the default
(`TCP_USER_TIMEOUT` stays twice `Options::heartbeat`: it counts unacknowledged data, so a frame sent after a long interval is still
dropped within 10 s of going unacknowledged).

**A long command's answer starts early.** A command that answers within one heartbeat (`Options::heartbeat`, 5 s) gets
its answer as ever, status included. One still running then gets `200` headers at that first silence and one space per
heartbeat after it, then the answer: instd's gateway cuts a response with no headers at 295 s, which ended every longer
one-shot with a `502`. Leading whitespace is JSON's own (RFC 8259), so the body is still one document and every client's
decoder (serde_json, `JSON.parse`, `json.loads`, Go's `encoding/json`) reads it unchanged. After the headers a status can no longer change: an
output over the cap arrives as its `{"error"}` body under `200` (a client reports an answer with no `exit_code`). A client that
leaves ends the command. `tests/oneshot_headers.rs` pins all of it.

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

**The start key.** `start` may carry `idempotency_key` (1 to 128 characters of `[0-9A-Za-z._:-]`, else `400`; `execd_jobs::key_valid`, K12 and K13): a start retried
after its answer was lost never runs a second job. What `start` does is `Table::begin`'s decision (`start_decision`, K1 to K5, proven in
`crates/execd-jobs`): a key already used for this command, by a job still in the table, answers that job (`200`, its process group, `stdin`
as the first answer had it, and `key`, the key echoed: the answer, or a held start's first frame, says the endpoint kept the key, and a client that
sent one and got no echo fails closed with `not_offered`; a held start streams the job from its first byte) and runs nothing; the same key for another command is `409`
"the key was used for a different command or id" (`"code": "key_conflict"`; the id is part of what the client asked for); an id in the table with no key that names it (no key, or another) is `409` "the job exists".
"The same command" is a SHA-256 digest of command, args, cwd, env (as a set), `stdin_open` and `hold` (`jobs::spec_digest`); only the
digest is kept, never the env, which can hold secrets, and the label and the id are not in it.

_Where the record lives, and how long._ The key's record (`Rec`: scope, key, id, digest) is the table's, written under its lock with the entry
(K7), saved in the job's ledger entry (`Idem`, `execd-jobs.json`, a field older files do not carry) and put back by `recover` for
every job it brings back (K6, `keeps_record`), so a retry after a restart of execd replays. It leaves only with its job, in `Table::take`
(K8: exactly that job's records, no other's, never one a successor of the id could be mistaken for). The retention (see "Ended jobs:
retention") drops a keyed job's _output_ like anyone's, and never its _entry_: no ceiling, clock or budget reaps a keyed job, so a job
that ended, even one whose group the restart found gone, is still answered by its key until a client removes it
(`tests/retention.rs`, `a_keyed_job_replays_after_every_deadline`, red under `mut_retention_reaps_keyed`).

_The claim comes first (K9, K10)._ A keyed start's ledger entry is written as an intent (phase `spawning`, the key, digest and id) before
anything is spawned, then rewritten with the group. The write of the claim is durable: the file is written aside, synced, renamed, and
its directory synced (`Ledger::claim`; the best-effort `save` of every other change is not). How it came out goes to the kernel
(`launch(Option<Persist>)`, K10 and its liveness partner K16): a claim that is not durable is a start that does not run, `507` when the disk is full and
`500` otherwise, with nothing spawned and the entry taken out of the table (its key's record with it, K8), so the retry, once the disk takes the
claim, starts. A server with no state directory keeps nothing across a restart, so there is no claim to lose.

_The restart (K9, K11)._ A restart that finds an intent with no group (a crash between the claim and the group's save) asks
`recover(.., keyed = true)`: the job comes back ended with the lost status (`-1`, no group), in the table, its record kept, and `recover`
returns the action `Sweep::KillMarked`, whatever the OS saw: the shell kills every process carrying the intent's mark (`proc::job_members`
then `SIGKILL`, looking again until a pass finds none), because the spawn may have happened and nobody can find those processes by a group.
A process between its fork and its exec carries execd's environment and not the mark yet; the repeated look catches it once it execs.
The retry replays the lost job (`started` is `0`: there is no group), and no restart sequence runs the command twice. An unkeyed start with no group is still not saved (R4).

_The replay's wait (K14, K15)._ A replay that arrives while the original is still spawning polls for its group; what each poll answers
(`replay_poll`: answer, wait, or refuse with `500` "the job did not start") is the kernel's, over the group, whether the job ended, whether it was removed
and how many polls were made. A job that has a group or has ended is always answered; a refusal is only for a job that never started, and the wait is bounded.

_Test knob._ `EXECD_TEST_FAULT` (a debug build only) names a file whose lines `claim_save=full|eio` and `after_spawn=exit` make the claim's write fail or
execd exit between the spawn and the group's save, `sweep_after_save=exit` and `ledger_slow=<ms>` a retention sweep's crash window and a ledger write
that costs that long, and `ledger_append_crash=<n>`, `ledger_fold=1` and `ledger_fold_crash=exit` a ledger record cut short, a forced fold and the crash
between a fold's two files; removing the file clears the fault. It is read in `fault.rs` alone.

**Output streams.** (`src/ring.rs` does the I/O; which offsets are held, where a read starts, which chunks retention drops and which bytes are stderr are the Verus-proven kernel `crates/execd-ring`, R1 to R6 in its docs, and `ring.rs`'s tests hold the whole to a plain `Vec` under random pushes, reads and refused chunks.) The ring keeps which pipe each byte came from, one bit per byte: an in-memory bitmap for the unfinished tail,
allocated at the job's first stderr byte, and a `<chunk>.e` sidecar (8 KiB, same `0600`, `O_EXCL|O_NOFOLLOW`) beside each full
chunk that holds any stderr. A job that never writes stderr allocates and writes nothing extra. Eviction removes the sidecar with
its chunk; a sidecar that cannot be written turns its chunk into a gap, so bytes are never read back mislabelled. `read` answers
`stderr_ranges` (absolute `[from, len]`, ascending, disjoint, at most `RANGES_MAX` = 1024: the window is cut before the range that
would exceed it, so an alternating writer cannot grow an answer), and a job stream's frames are keyed `stdout` or `stderr` and
never span a change of stream. Arrival order between the two files is the order the job's tail read them: a buffer from each in turn (see "A job's output is a file").

**Stdin.** `start` with `stdin_open` spawns with a piped stdin (default `/dev/null`: a job that reads to EOF must not hang) and
the answer echoes `stdin: true`. The job holds a `JobStdin`: the pipe behind an async (FIFO) mutex, `written` (a `watch`), a
cancel token and a waiter count. `write` is positional (`at`): bytes below `written` are skipped, a write ahead waits on the
`watch` without holding the pipe (so the write that fills the gap can run), then applies under the mutex, advancing `written`
after every `write(2)`. So writes are ordered, idempotent and exact when cut off. Every decision of the write is the job kernel's (`execd_jobs::stdin_plan`: wait, refuse, already applied, or apply from `written`; `admit_waiter`; `advance`), proven in Verus: bytes below `written` are never applied again (W2a), `written` only grows (W2b) and an apply starts exactly at `written` (W2c); EOF closes only after every byte landed (`remaining`, `closes`), nothing lands on an ended job (W3) or after EOF (W4), a write ahead waits rather than applies (W5), the 17th waiter is refused (W1), and a refused or waiting write is always answered within `stdin_rank` steps (W1-live, `lemma_stdin_progress`). The shell only does the pipe's I/O, the `watch` and the mutex. There is no queue and no polling: a full pipe
parks the request on the reactor, `timeout_ms` bounds the wait (short write), a dropped connection cancels it. `WRITE_MAX`
128 KiB, `WAITERS_MAX` 16 per job (429). The token cancels and the pipe is dropped when the job ends or is removed, so no fd
outlives the job. The std `Mutex<Job>` is never held across an await.

**Terminals** (`src/term.rs`). `start` with `pty: {cols, rows}` makes the job a terminal: its stdin, stdout and stderr are one
pseudo-terminal (`TERM=xterm-256color` unless the request sets it), the command leads a session of its own with the terminal as
its controlling one, and `write`, `resize`, `read`, `attach`, `signal`, `kill` and a restart all work on it as on a job. The
master is held by a **keeper**, one small process per terminal (execd's own binary again: `--execd-run-as-helper keep ...`), so
a restart or crash of execd does not hang the terminal up, as it does not end a kept job. The keeper opens the pty, spawns the
**session helper** (`... session ...`: `setsid`, `TIOCSCTTY` on the slave before any user switch, the slave on 0, 1 and 2, then
`exec`; under `EXECD_RUN_AS` the command is the `run` helper, which switches after the terminal is the command's, and the slave is
chowned to the target), copies the master into the job's `out` spool file (its own stdout: the same spool, ring, offsets,
journal and resume as every job; `err` has no writer, so a terminal's output is stdout alone), and serves execd on
`<spool>/term.sock` (the spool directory is `0700`; a peer must be execd's uid, `SO_PEERCRED`; the first frame, `Ready`,
carries a format version, the session's group, the count of input that landed and the size). It blocks in `poll` with no
timeout, so an idle terminal costs no wakeups, and holds one 64 KiB buffer and at most one write in flight.

The keeper owns the **input count** (`written`): execd sends `Write{at, bytes}`, the keeper plans it with the kernel's
`stdin_plan` (bytes below `written` are skipped) and answers `Written{n}`, so a write retried across a restart of execd lands
once. On the shell's exit the keeper drains the master, writes `<spool>/status` (aside and renamed) and sends `Exit`, then ends
when the master reads `EIO` (no slave open anywhere). A restarted execd asks each saved terminal's keeper (`recover`: the
ledger entry has the keeper's pid and the size): an answer is adopted open at the keeper's size; none is hung up
(`pty.open: false`; input and resize 409), and a dead keeper's `status` is the shell's real code where a plain job's is lost.
Who may talk to whom is the kernel's `admit_peer` (P1), asked at both ends: the keeper admits a peer of execd's uid, execd a
peer of its uid whose pid is the keeper's (the ledger's `keeper_pid`); a failed credential read, another uid or another pid is
refused, and nothing is read from a refused peer. Which answer counts as the keeper's is `keeper_of` (T6a): an admitted peer's
`Ready` in a version execd knows (`KEEPER_VERSION`). Adoption and the start's own wait for `Ready` both call it. A keeper
whose spool cannot be written (no space, no device) fails closed (Q6): the kernel kills the shell's session once, and the
status the keeper writes and tells is `SPOOL_FAILED_CODE` (74, no signal), so the reader sees an end with that code instead
of a terminal that looks alive with its output gone. A start refused for want of a pty is `pty_cap` / `keeper_ended` (Q1: 503
when a keeper exits 124 or the test cap is passed, else 500). A resize is idempotent (T14): asked twice at one size it leaves
the job as asked once.
A hung-up terminal is still a job: it ends when its processes do. A held terminal nobody kept dies with its client, every
process of its session included (`Kill{marked}`).

**Behaviour change for every held job.** A held start nobody kept that loses its client dies with the group and every process that carries its mark (`Kill{marked}`), on a dropped stream and when a restarted execd finds one in the ledger. Before, a restart's recovery killed the group alone. One guarantee for every job; for a terminal it is what makes a shell's job-control groups die with it.

**For an embedding host.** `Handle::activity().terminals` lists each kept terminal that has not ended: `{scope, id, pgid, pts_rdev, pts_fs, attached, members}` (the slave's `st_rdev` and the `st_dev` of its devpts, whether a stream is open on it, and the job's processes: its group plus the marked ones that left it, the set `kill` signals). `Options::helper_exe` names the binary a keeper is started from when the host is not execd's own binary (a test binary); `EXECD_TERMINALS_MAX` caps open terminals (503 beyond it) and exists so a test can reach that refusal without filling the machine's ptys. A write refused because the terminal hung up is `409 {"error": "the terminal hung up", "written": N}`; the SDKs type it `StdinClosed` with reason `hung_up`.

Every decision is the kernel's: `size_ok`, `write_gate` (a terminal has no end of input; nothing goes to one that hung up),
`resizable`, `Ev::Resize` (applied at once if the job runs, at the spawn if it is still spawning), `Ev::Hangup`, `recover`'s
`Keeper`, `sinks`, and the keeper's own `execd_jobs::keeper::step` (what it copies, applies, writes and tells, and when it
ends). `tests/job_model.rs` (`faithful_terminal`: T1, T1-live, T6 to T10, T12) and `tests/keeper_model.rs` (K1, K2, T5) call
those functions; `tests/pty.rs` drives the real binary. Each `mut_*` of `crates/execd-jobs` listed below turns its property
red in Verus (`mise run verify:kernels -- --features <mut>`), and the ones the models name turn them red too.

**Signal.** `signal` sends one of seven named signals with `proc::signal_job` (the group, plus marked processes that left
it: each process of the job hears it once) off the runtime, never to a job that has ended (its group number may be reused), and
answers the ids it signalled. No grace, no table change; a signal that kills the shell is seen by the supervisor like any death.

**Per-command env.** `Request.env` is parsed once, before dispatch, by `http::parse_env` (names, 128 variables, 32 KiB, no NUL,
`EXECD_` prefix refused, a `400` that names the rule and never a value). `exec::command_for` applies it last (after the scrub
and `HOME`); the job mark is set after that. Under `EXECD_RUN_AS` it travels in one reserved variable `EXECD_CMD_ENV` (hex JSON)
that the helper reads and removes before it switches users and applies only to the final `Command`, so a request's `LD_PRELOAD`
takes effect at the exec, as the target, never in the root helper. An embedder's `Observer.request` receives the raw body, env
included.

notify is the `notify` module driving `crates/exec-notify`, the pure timing kernel every endpoint shares. The kernel says when an
attempt begins, when to wait, and when to stop; `notify` makes the request (one `POST`, no redirect, the server's certificate
verified, 30 s at most) and feeds back what happened. A job's latest `keep` governs, a `keep` abandons the attempt in flight,
and a removed job's timers are dropped with it. The token is held in the job and in the request header, and appears in no log,
no answer, no argument and no environment; `Debug` for it prints `<redacted>`.

The notifier lives in the job's cell and is tied to the cell, never to its `(scope, id)`: an attempt and a timer hold the cell, and
the key only names the callback's body, so a later job of the same id is never settled by an old job's answer. When the table lets a
job go, `Jobs::remove` leaves its notifier as the kernel's removed job (`Notifier::removed`, `exec_notify::Event::Removed`), which
owes nothing and answers every later event with no action. A `keep` looks its job up by id and takes the job's lock afterwards, so
it can meet a job that left in between; the removed notifier is what keeps it from calling back about a job that is gone. Before this,
removal reset the notifier to its default and such a `keep` began a callback for the removed job. The agent's fleet model
(`crates/agent/tests/notify_model.rs`, properties X1 to X4) runs this table's kernel next to the notifier to hold that seam.

### The job kernel: `job_kernel`

The job table's decisions are pure functions in `crates/execd-jobs` (re-exported as `execd::job_kernel`, so the model and production call the same code; written inside `verus!` and proven for every J and L below with no bound, see its README): no I/O, no async, no clock (a time arrives in the event),
no closure in a decision. `jobs.rs` takes a lock, calls the kernel, and performs the action it returns, in the same order as
before. The two locks stay as they were, and the kernel follows them:

- the table's (`Jobs::map`): `Table` (`admit`, `begin`, `insert`, `record`, `get`, `in_scope`, `take`). A key in the table is `Exists` (409), a lookup
  never leaves its scope, and `take` is the only way out. `begin` is what a start asks: `Start`, `Replay(id)` or `Refuse` (K1 to K5, the start key above), over the table and the key records it holds.
- one job's (`Job`'s mutex): `State` (`Phase`: `Spawning`, `Running`, `Exited`, `Ended`, one way only; `Claim`: `Held` or `Kept`,
  never back; `seen`; `removed`, final), changed only by `step(&mut State, Ev) -> Act`.
- output offsets: `follow` (a stream's next frame: gap, data,
  the shell's exit, the end, or wait; where a read starts is `execd_ring::window`, the ring kernel's). The ring's bytes stay in the shell.

| event                                | what the kernel decides                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `Spawned`, `ShellExit`, `GroupEmpty` | the phase advances once; an ended job never changes; a callback is owed for the end only if the job is not removed |
| `Keep`, `Seen`                       | `Held` becomes `Kept` for good; `seen` is what the model was shown                                                 |
| `Taken`                              | the table let the job go: removed, final                                                                           |
| `Push`                               | a pump's bytes are stored unless the job is removed                                                                |
| `Signal`, `Stop`                     | a group is signalled only while the job has a group and has not ended (its number may be reused)                   |
| `Remove`                             | out of the table only with its group observed empty, or, for `remove`, once ended                                  |
| `Disconnect`                         | a held job nobody kept, still running, dies with its client's connection; a kept one never does                    |

A stream is bound to the job it attached to (`bound`): after that job is removed it is gone to the stream, and a later job of the same id is never followed or killed by it (the model found the by-key lookup that did, J9).

`remove`, and `kill` with `remove` or `wipe`, check a job (`removable`, which may await its group) and take it later. The take is bound to the job checked (J10, `takes`, `Table::take` with the shell's `Arc::ptr_eq`): if the key was freed and started again in between, the successor stays, signalled by nobody and checked by nobody. So does the spawn failure's take, which names the cell it created.

**The mark on the wire.** A job's mark (`EXECD_JOB_MARK`, its spool directory's name) names one incarnation of an id, and the wire says it, so a client never mistakes a later job that reused the id for the one it knew. `start` answers `{"started": pgid, "mark": m}` and a held start's first frame is `{"started": pgid, "mark": m}`; `list` carries each job's `mark`, `command`, `args` (never the environment; the ledger keeps them too), `outputStart` and `outputEnd`. `attach` answers first with `{"attached": {"mark": m, "from": n}}` (`from` is the request's `offset`); it takes an optional `mark`, and a job whose mark differs is refused as `404 no such job` (`mark_ok`, M1, then `attach_plan`, M4; the refusal is not a state, so the retry with the right mark streams, M4-live). `attach` with `"output": false` is status-only: it starts at the output's current end whatever `offset` it names (`from`, M6) and sends no byte and no gap, only the shell's exit and the end (`stream_next`, M7, applied at every frame). `remove` takes an optional `marks` (id to mark) and answers `{"running": [ids], "mismatch": [ids]}`: a job whose mark is not the one named stays (M2) and is reported as a mismatch, a job whose group is not empty stays and is reported as running (M3), an id with no job is in neither, and the retry with the right mark takes it (M2-live). A start's id is 1 to 64 bytes of `[0-9a-z_-]` (`id_ok`, I1, I2; `400` otherwise). This is a clean break: execd used to accept any id. `list` reports each job's argv (`command`, `args`), which the 0600 ledger stores, within the job's scope; never its environment, and past `LIST_ENDED_MAX` ended rows it answers the count it left out as `"more"` ("The `list` bound", under retention). Every decision above is in `execd-jobs`, proven in Verus with a mutant per row (`mise run verify:execd-jobs-mark-mutants`); `job_model` runs them over the table (`M1`, `M3`, `M7` red under `mut_mark_ignored`, `mut_remove_ignores_mark`, `mut_attach_ignores_mark`, `mut_running_unreported`, `mut_status_only_streams`, `mut_status_only_sends`) and `tests/jobmark.rs` drives them against the binary.

A client frees a job with `remove`, `kill` with `remove` or `wipe`; a job no client frees is bounded by the retention below.

#### Ended jobs: retention (`jobs::sweep`)

An ended job a client never removes used to stay until execd exited: up to its 8 MiB of output (plus stderr labels and the unfinished
chunk in memory) and one table entry, forever. `Options::retention` bounds both, in two tiers, with **no protocol change**:

| knob                     | default             | tier  | what it does                                                                       |
| ------------------------ | ------------------- | ----- | ---------------------------------------------------------------------------------- |
| `EXECD_JOB_OUTPUT_MS`    | 900000 (15 min)     | 1, on | drop an ended job's output this long after its end; keep its entry                 |
| `EXECD_JOBS_ENDED_BYTES` | 268435456 (256 MiB) | 1, on | past this much output over ended jobs, drop the oldest's output first; never reaps |
| `EXECD_JOBS_ENDED_MAX`   | 4096                | 2, on | past this many ended entries, reap the oldest reapable one (a backstop)            |
| `EXECD_JOB_ENTRY_MS`     | 86400000 (24 h)     | 2, on | reap an ended job's entry this long after its end (a predictable lifetime)         |

`0` is off for each, and **all four at `0` is exactly the behaviour before**: the sweeper task is never started
(`Retention::enabled`, `start_retention`), so nothing is dropped or reaped and nothing wakes (`every_knob_off_is_todays_behaviour`).

**Why 4096 and 24 h, and what a full table costs.** The clock is the deadline and the ceiling is its backstop: 4096 entries is one
job every 21 seconds for a day, so the ceiling does not undercut the promise until a sandbox runs faster than that, while 256 (one
every 5.6 minutes) did undercut it constantly — a ceiling smaller than the clock's rate is a ceiling that fires first and reaps
entries the clock promised to keep. The costs, measured with `benches/retention_scale.py` (release binary, this 2-CPU host, a
synthetic ledger a restart adopts — the rows execd writes, realistic ids and argv, no spool):

| ended entries | `list` before -> now (bytes, p50 / p95)         | `/livez` in a `list`, before -> now | a life change's own write | `/livez` during that write, before -> now | start with that ledger, before -> now | RSS     |
| ------------- | ----------------------------------------------- | ----------------------------------- | ------------------------- | ----------------------------------------- | ------------------------------------- | ------- |
| 256           | 90.9 KB, 2.2 / 3.4 ms -> the same               | 4.0 -> 1.0 ms                       | 2.2 ms                    | 601 -> 0.4 ms                             | 0.45 s -> 0.02 s                      | 7.0 MB  |
| 1024          | 363.5 KB, 11.3 / 13.2 -> 181.8 KB, 5.2 / 6.2 ms | 11.5 -> 6.4 ms                      | 3.9 ms                    | 1206 -> 0.4 ms                            | 1.9 s -> 0.04 s                       | 9.5 MB  |
| 4096          | 1.45 MB, 36.9 / 45.7 -> 181.8 KB, 3.0 / 5.7 ms  | 46.6 -> 3.0 ms                      | 14.7 ms                   | 3917 -> 0.4 ms                            | 7.0 s -> 0.11 s                       | 14.6 MB |

Each of the three costs that scales with the table is answered here:

- **`list` is bounded** (`LIST_ENDED_MAX`, below), and a first sweep after a restart reaps a full table in 64 passes
  (`SWEEP_ACTIONS_MAX`), which leaves the ledger and the table empty and every reaped id answering `gone`/`404`.
- **The ledger write is off the runtime** (`Ctx::save_jobs`): one of the `disk_writers` turns, the write and the rename on a thread
  of its own (`proc::off_runtime`), one write per change of a job's life, and the change's own answer awaits it. What that keeps is
  the promise the inline write made — when a `start` answers, the file names the job; when a `remove` answers, it does not — and
  what it gives back is everybody else's latency: with every ledger write held 300 ms, an unrelated `GET /livez` waited 601 ms
  before and 0.4 ms after at 256 (1206 -> 0.4 at 1024, 3917 -> 0.4 at 4096). The request that asked still waits for its own
  write, as it must: 14.7 ms at a full table on tmpfs, 3.9 s on that disk.
  **Ordering against the sweep's write, or another life change's:** `Ledger::save` takes the file's writer lock and only then
  snapshots the table, so writes are serialized and the last file written is the last state snapshotted, whichever thread wrote
  it; a write asked for earlier can never land over a newer one. A keyed start's claim keeps its stronger promise the same way:
  written, synced, renamed and its directory synced off the runtime, its result awaited before the start spawns (`claim_jobs`,
  K10).
- **Recovery does not scan `/proc` for a row already saved as `ended`** (`jobs::may_be_alive`). A job ends only with its group and
  every marked process gone, so such a row's processes cannot exist, and `execd_jobs::recover` discards the answer for that phase
  anyway (an ended state comes back `Restore`, unchanged). Scanning anyway cost one pass over `/proc` per row: 63,000 opens and
  7.0 s of start at 4096 rows on an 86-process host, 10.3 s at 4000 rows on this one once the scan is put back, and it scales with
  the host's process count. With the skip, a 4096-row start is 0.11 s: the ledger's parse and the table's insert.

- **A life change writes one record, not the table** (`ledger.rs`). The ledger is two files: `execd-jobs.json`, the table
  alone, and `execd-jobs.log`, one framed record per change of a job's life since — the job's whole row, or the fact that it is
  gone. A change appends its record; the log is folded back into a new file when it outgrows the table (`LOG_MIN` = 64 KiB, or
  the size the table takes, whichever is more), so the pair never holds more than two tables. Measured on the release binary, one
  start→end→remove cycle (three life changes) at 256 / 1024 / 4096 ended entries:

  | ended entries | table   | bytes written per cycle, before → now | per cycle p50, before → now | cycles/s, before → now |
  | ------------- | ------- | ------------------------------------- | --------------------------- | ---------------------- |
  | 256           | 94 KB   | 284 443 → 1 850                       | 7.6 → 6.7 ms                | 132 → 149              |
  | 1024          | 377 KB  | 1 132 236 → 1 778                     | 13.3 → 12.6 ms              | 76 → 80                |
  | 4096          | 1.51 MB | 4 523 581 → 1 617                     | 60.8 → 35.9 ms              | 16 → 28                |

  The before column is the release binary at `87347af`, built in its own worktree and run by the same harness
  (`handoffs/CAPY-12/ledger_cycles.py --before-bin`); the byte counts move by a few hundred bytes with where the fold lands,
  the timings by about 15% run to run.

  On a disk that bills 300 ms a write _and_ a rename (the `handoffs/CAPY-8/slowdisk.c` shim), a cycle is 912 / 915 / 640 ms
  against 1813 / 1825 / 1885 ms: the old path paid three renames a cycle, the new one none. Every guarantee the whole-file write
  made is kept, and each is a test in `tests/ledger.rs` that goes red when it is taken out:
  - **A record is framed** by a length and a CRC-32, and a replay stops at the first record that does not hold together — a
    record cut short by a crash is dropped, not believed (`a_record_cut_short_is_dropped_and_the_log_folded`,
    `a_record_that_does_not_checksum_is_not_applied`). The load marks such a log for a fold, so the next write rewrites it rather
    than appending behind the fragment, where every later record would be unreadable.
  - **The fold is the old state or the new one, never a mix.** `execd-jobs.json` is written aside, synced, renamed, its
    directory synced; then `execd-jobs.log` is replaced by a new one whose header names that file by length and checksum. A crash
    between the two leaves the new table with a log that names the old one, and the log is dropped
    (`a_crash_between_the_fold_and_its_log_leaves_the_new_table`). The log in force therefore always names its table.
  - **The answer still implies the change is written.** The change's own record is written (and, for a keyed start's claim, the
    whole file is written, synced, renamed and its directory synced) before the answer goes out: a `keep` answers with the ledger
    already holding it, and a kill the instant the answer arrives recovers it
    (`a_life_change_is_on_disk_when_its_answer_arrives`; `tests/startkey.rs` for the claim, K10).
  - **Ordering against a sweep is the writer lock, and the log no longer needs it for ordering.** `Ledger::save` reads the table
    under the file's writer lock and appends what differs from what the log says, so each record is a row as of the moment of the
    write and the last record for a row is the newest state; two writers cannot lose each other's fact
    (`a_write_that_overlaps_a_sweep_loses_nothing`). The lock is still what a fold needs exclusively.
  - **The write that must leave the table whole still does.** A retention sweep's one pass is `save_whole` (a fold): the file
    is the table and the log is empty, so the ledger-then-unlink order of the retention and its crash test
    (`a_crash_after_the_ledger_write_leaves_no_spool_behind`) are unchanged. A restart's write is an ordinary change — one
    record for what recovery changed, or nothing — except into a log a crash left torn, which is folded instead of appended
    to, so the torn tail is repaired (`a_record_cut_short_is_dropped_and_the_log_folded`).
  - **No version to serve.** A sandbox gets the binary it was built for, so the ledger owes no older execd anything and
    there is no compatibility flag to keep. `execd-jobs.json` keeps the version-1 shape and the log names it by length and
    checksum because those are the cheap whole-table form and the fold's identity check, not because a rollback reads them; the
    version markers stay as a diagnostic. What is mandatory is same-build recovery, and it is what the four guarantees above
    are: a restarted execd reads exactly what it wrote, a torn tail is dropped rather than resurrecting or losing a job, and a
    crash mid-fold leaves one consistent pair.

**The `list` bound.** `list` answers with every job of its scope that has not ended, and with the newest `LIST_ENDED_MAX` (512)
ended ones; past that it adds `"more": n`, the number it did not list. The newest are the rows it keeps, because the count ceiling
reaps the oldest first and a client that reattaches wants what just happened, and the response carries the count instead of silently
dropping rows: a client that ignores an unknown field sees the defined answer for a table today's ceiling already bounds (256 ended
rows, byte for byte what execd answered before), and one that reads `more` knows to ask for less or wait. Rows that have not ended
are not bounded — the table cannot hold many, since each needs processes and a spool, and the byte budget governs a job's output,
not its existence — and the cap covers ended entries only, which are what a ceiling raises.

**Tier 1, the tombstone.** The entry keeps its mark, argv, `state: "ended"`, code, signal, end time and offset; the ring goes to
`base == end` (`Ring::evict_all`) and its spool is deleted. A late client gets the existing gap semantics: `read` answers the
status with `offset == end` and no data; `attach` from an older offset streams one `{"gap": n}` and the final frame; `list` keeps
the row with `outputStart == outputEnd`. That is the answer a job past the 8 MiB floor already gives, applied to the whole ring. The
ledger records it (`Entry::out`, serde-default), so a restart rebuilds an empty ring at the same offset (`Ring::evicted`).

**Tier 2, the reap.** Only a count ceiling or an operator's clock takes an entry, through **the gate `remove` takes**:
`kernel::step(Ev::Remove { empty: false, ended_counts: true })` must answer `Take` (L1/J7), `Table::take` with `Arc::ptr_eq` takes only the
cell that was gated (J10), and `Ev::Taken` sets `removed` — all in one critical section, map lock then cell lock (the order
`entries` takes). Only then does a client see `read` → `gone` and `attach` → `404 no such job`.

**The vetoes** (`jobs::candidate`), each checked under the job's lock in the same critical section as the change, each with a test
that is red when it is deleted (`tests/retention.rs`, the `mut_retention_*` / `mut_attach_*` features):

| veto                                | why                                                                                                    | in the gate before?                                         | test, and the feature that turns it red                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| not `Phase::Ended`                  | a `Running`/`Exited` job can still gain bytes; `removable`'s "group observed empty" step is not enough | partly (J7 refuses a live group, not an observed-empty one) | `only_an_ended_job_is_a_candidate` (unit, every phase), `a_running_job_is_never_touched`; `mut_retention_any_phase` |
| an open stream (`streams > 0`)      | the ring is the only copy of its last bytes; a take ends it early (J9)                                 | no                                                          | `an_attached_stream_holds_its_job_past_every_deadline`; `mut_retention_ignores_streams`                             |
| an owed callback (`Notifier::owes`) | a take abandons the count; the fleet never learns the end                                              | no                                                          | `an_owed_callback_holds_its_job` (a silent receiver and a 503); `mut_retention_ignores_owed`                        |
| a keyed job (never reaped)          | its key's record leaves with it (K8): a retry would run the command twice                              | no, the opposite                                            | `a_keyed_job_replays_after_every_deadline`; `mut_retention_reaps_keyed`                                             |

**The attach window.** `attach` used to decide (`attach_plan`, under the job's lock) and register its stream later
(`Watching::new` in the spawned stream task), so a sweep could see `streams == 0` between the two. The stream is now registered in
the critical section that computes `from` (`Watching::open`, handed to `stream_job`), and an attach that finds its cell already
`removed` answers `404`. Two debug-only fault seams (`fault::park`, `attach_after_decide`, `attach_before_decide`) park an attach
on each side of that lock so the race is forced, not hoped for: `an_attach_mid_decision_is_never_reaped` (red under
`mut_attach_registers_late`) and `an_attach_that_loses_to_a_take_is_404` (red under `mut_attach_ignores_removed`).

**When it runs.** One task, asleep until a job ends, a veto lifts (the last stream closes, a callback stops being owed), a restart
recovers, or the nearest deadline of an unvetoed job arrives (`next_wake`). A vetoed job never sets a timer, so nothing polls. A
pass acts on at most 64 jobs (`SWEEP_ACTIONS_MAX`) and wakes itself if it hit that; the shortest sleep is 50 ms. Deadlines resolve
in whole seconds (`Phase::Ended.at` is epoch seconds) and follow the wall clock: a clock step moves them.

**Atomic and idempotent.** The decision and the state change are one critical section per job; the filesystem and the ledger are one
pass **off the runtime thread** (`proc::off_runtime`), one ledger write per sweep, ledger first: the entry stops naming the spool and
names the offset, then the spool and chunk files are unlinked. A crash between the two leaves a spool no entry names, which the next
start deletes (`Spool::retain`; `a_crash_after_the_ledger_write_leaves_no_spool_behind`, through the `sweep_after_save` seam). A
second sweep finds nothing held and nothing due (`a_tombstone_keeps_the_status_and_answers_a_gap_across_a_restart` compares the ledger
row across one). Against a concurrent `remove`, `kill`, `wipe` or a failed start's take, both sides take the map lock and
`Table::take` with `Arc::ptr_eq`: the loser finds no cell (`a_sweep_concurrent_with_remove_and_start_leaves_one_state`).

`tests/job_model.rs` is a Stateright model that calls these functions, never a copy, and an environment (the shell dying with or
without a descendant, `kill`, pumps, eviction, a held connection dropping at any point). Five bounded configs, each under ten
seconds (CI kills a test at 180s): `REUSE` (one scope, one id, two jobs, one client, a one-byte ring: an id reused after a removal),
`SCOPES` (two scopes that both use one id), `IDS` (two ids in one scope), `STREAMS` (one job, two clients, a two-byte ring), `SLOW` (a `remove`/`wipe` that decides and takes as two steps, other clients' removals and restarts of the id between), `KEYED` (two keys, two specs, two ids, two jobs: a keyed start sent again at any later point, which is the retry of a lost answer, across a restart of execd, a removal and another key's record). Properties: J1 an id is the claim (a start never replaces a job in the table), J2 no
operation resolves another scope's job, J3 a removed job never answers again (and a stream never follows a later job of its id),
J4 an ended job never changes, J5 output offsets only move forward over what is held, J6 a kept job never dies with a held client,
J7 a job leaves only with its group empty, J8 no signal reaches an ended job, J9 a stream follows the job it attached to, never a later job of its id, J10 a take removes only the job that was checked, never its key's successor, J11 (OS boundary, below), K1 a keyed start never runs a second job for its key and spec, K2 a replay names the job started under its key and spec and a key used for another spec never starts, K3 (liveness, `k = 1`) a retry of a keyed start whose job is in the table is answered with it and a fresh key on a free id starts, K4 a start with no key never replays, L1 an ended job can always be removed, L2 an ended
job's stream sends its final frame. `production_calls_the_kernel` fails if `jobs.rs` stops calling the kernel or writes job state
outside it. `crates/execd-jobs/tests/production.rs` fails if `execd::job_kernel` stops being that crate, a decision leaves its `verus!` block, or a second copy of the state machine appears.

The kernel is a crate of its own because Verus verifies a crate whole, and execd's async stack is not Verus's to build. `mise run verify:kernels` proves it (CI's `verus` job); `mise run verify:kernels -- --features mut_x` must go red for each mutation below (the table's storage is a `Vec` inside `verus!`, so `mut_remove_keeps_entry` and `mut_scope_ignored` are proven too; the crate README names what stays outside).

```sh
cargo test -p beyond-execd --test job_model faithful          # both bounds, all properties
cargo test -p beyond-execd --features <mut> --test job_model the_mutation   # each MUST pass only if its property is red
```

| mutation                              | property it turns red                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------- |
| `mut_start_overwrites`                | J1                                                                                    |
| `mut_scope_ignored`                   | J2                                                                                    |
| `mut_remove_keeps_entry`              | J3                                                                                    |
| `mut_end_rewritten`                   | J4 (`ended_is_final`, over every phase and event)                                     |
| `mut_disconnect_kills_kept`           | J6                                                                                    |
| `mut_remove_live`                     | J7                                                                                    |
| `mut_signal_ended`                    | J8                                                                                    |
| `mut_ended_not_removable`             | L1                                                                                    |
| `mut_stream_follows_key`              | J9                                                                                    |
| `mut_take_by_key`                     | J10                                                                                   |
| `mut_follow_waits_after_end`          | L2                                                                                    |
| `mut_signal_after_reap`               | J11 (`os_model`, and Verus on `may_deliver`)                                          |
| `mut_held_forgets`                    | J11-live (`os_model`, and Verus on `Held::insert`)                                    |
| `mut_stdin_replays`                   | W2 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_written_shrinks`           | W2b (Verus on `advance`)                                                              |
| `mut_stdin_apply_off_start`           | W2c (Verus on `stdin_plan`)                                                           |
| `mut_stdin_wait_sticks`               | W5-live (progress lemma)                                                              |
| `mut_stdin_closes_early`              | W4 (Verus on `closes`)                                                                |
| `mut_stdin_applies_ahead`             | W5 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_writes_ended`              | W3 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_writes_after_eof`          | W4 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_waiters_unbounded`         | W1 (Verus on `admit_waiter`)                                                          |
| `mut_stdin_waiter_stuck`              | W1-live (Verus on `admit_waiter`)                                                     |
| `mut_stdin_waits_at_written`          | W5-live (Verus on `stdin_plan`)                                                       |
| `mut_resize_ended`                    | T1 (Verus on `step`; `job_model` `PTY`)                                               |
| `mut_resize_lost_while_spawning`      | T1-live (Verus on `step`, `lemma_resize_progress`; `job_model` `PTY`)                 |
| `mut_size_unbounded`                  | T3 (Verus on `size_ok`)                                                               |
| `mut_size_off_by_one`                 | T4 (Verus on `size_ok`, the completeness half)                                        |
| `mut_pty_err_sink`                    | T5 (Verus on `sinks`)                                                                 |
| `mut_recover_hangs_up_live_terminal`  | T6 (Verus on `recover`; `job_model` `PTY`)                                            |
| `mut_recover_opens_silent_terminal`   | T7 (Verus on `recover`; `job_model` `PTY`)                                            |
| `mut_recover_drops_status`            | T7b (Verus on `recover`)                                                              |
| `mut_end_keeps_terminal`              | T8 (Verus on `step`; `job_model` `PTY`)                                               |
| `mut_pty_kill_group_only`             | T9 (Verus on `step`; `job_model` `PTY`)                                               |
| `mut_pty_eof_closes`                  | T10 (Verus on `write_gate`; `job_model` `PTY`)                                        |
| `mut_gate_refuses_open`               | T11 (Verus on `write_gate`, the completeness half)                                    |
| `mut_write_after_hangup`              | T12 (Verus on `write_gate`; `job_model` `PTY`)                                        |
| `mut_exit_before_drain`               | K1 (Verus on `keeper::step`; `keeper_model`)                                          |
| `mut_keeper_waits_forever`            | K2 (Verus on `keeper::step`; `keeper_model`)                                          |
| `mut_keeper_reconnect_resets_written` | T5b (Verus on `keeper::step`; `keeper_model`)                                         |
| `mut_hangup_keeps_terminal`           | T7 (Verus on `step`; `job_model` `PTY`; conformance `jobs.pty.keeper-killed-is-gone`) |
| `mut_admit_any_peer`                  | P1 (Verus on `admit_peer`)                                                            |
| `mut_pty_cap_ignored`                 | Q1 (Verus on `pty_cap`)                                                               |
| `mut_keeper_no_pty_is_500`            | Q1 (Verus on `keeper_ended`)                                                          |
| `mut_keeper_answer_unchecked`         | T6a (Verus on `keeper_of`)                                                            |
| `mut_resize_not_idempotent`           | T14 (Verus on `step`, `resized`)                                                      |
| `mut_spool_failure_silent`            | Q6 (Verus on `keeper::step`; `keeper_model`; `tests/pty.rs`)                          |
| `mut_spool_failure_keeps_code`        | Q6 (Verus on `keeper::exit_status`; `keeper_model`; `tests/pty.rs`)                   |
| `mut_start_overwrites`                | J1                                                                                    |
| `mut_scope_ignored`                   | J2                                                                                    |
| `mut_remove_keeps_entry`              | J3                                                                                    |
| `mut_end_rewritten`                   | J4 (`ended_is_final`, over every phase and event)                                     |
| `mut_disconnect_kills_kept`           | J6                                                                                    |
| `mut_remove_live`                     | J7                                                                                    |
| `mut_signal_ended`                    | J8                                                                                    |
| `mut_ended_not_removable`             | L1                                                                                    |
| `mut_stream_follows_key`              | J9                                                                                    |
| `mut_take_by_key`                     | J10                                                                                   |
| `mut_follow_waits_after_end`          | L2                                                                                    |
| `mut_signal_after_reap`               | J11 (`os_model`, and Verus on `may_deliver`)                                          |
| `mut_held_forgets`                    | J11-live (`os_model`, and Verus on `Held::insert`)                                    |
| `mut_stdin_replays`                   | W2 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_written_shrinks`           | W2b (Verus on `advance`)                                                              |
| `mut_stdin_apply_off_start`           | W2c (Verus on `stdin_plan`)                                                           |
| `mut_stdin_wait_sticks`               | W5-live (progress lemma)                                                              |
| `mut_stdin_closes_early`              | W4 (Verus on `closes`)                                                                |
| `mut_stdin_applies_ahead`             | W5 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_writes_ended`              | W3 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_writes_after_eof`          | W4 (Verus on `stdin_plan`)                                                            |
| `mut_stdin_waiters_unbounded`         | W1 (Verus on `admit_waiter`)                                                          |
| `mut_stdin_waiter_stuck`              | W1-live (Verus on `admit_waiter`)                                                     |
| `mut_stdin_waits_at_written`          | W5-live (Verus on `stdin_plan`)                                                       |
| `mut_id_length_unbounded`             | I1 (Verus on `id_ok`)                                                                 |
| `mut_id_any_byte`                     | I2 (Verus on `id_ok`)                                                                 |
| `mut_id_rejects_dash`                 | I1-live (Verus on `id_ok`)                                                            |
| `mut_mark_ignored`                    | M1 (Verus on `mark_ok`, and `job_model`)                                              |
| `mut_mark_refuses`                    | M1-live (Verus on `mark_ok`)                                                          |
| `mut_remove_ignores_mark`             | M2 (Verus on `removal`, and `job_model`)                                              |
| `mut_remove_never_takes`              | M2-live (Verus on `removal`)                                                          |
| `mut_running_unreported`              | M3 (Verus on `removal`)                                                               |
| `mut_attach_ignores_mark`             | M4 (Verus on `attach_plan`, and `job_model`)                                          |
| `mut_attach_refuses`                  | M4-live (Verus on `attach_plan`)                                                      |
| `mut_attach_offset_shifted`           | M5 (Verus on `attach_plan`)                                                           |
| `mut_status_only_streams`             | M6 (Verus on `quiet_from`)                                                            |
| `mut_status_only_sends`               | M7 (Verus on `stream_next`)                                                           |
| `mut_key_replay_starts`               | K1 (`job_model`, Verus on `start_decision`)                                           |
| `mut_key_digest_ignored`              | K2 (`job_model`, Verus on `start_decision`)                                           |
| `mut_key_fresh_refused`               | K3 (`job_model`, Verus on `start_decision`)                                           |
| `mut_unkeyed_replays`                 | K4 (`job_model`, Verus on `Table::begin`)                                             |
| `mut_key_forgotten_on_restart`        | K6 (`job_model` K3, Verus on `keeps_record`)                                          |
| `mut_key_record_clears`               | K7 (`job_model` K3, Verus on `Table::record`)                                         |
| `mut_key_take_forgets_none`           | K8 (`job_model` K2, Verus on `Table::forget`)                                         |
| `mut_intent_respawns`                 | K9 (`job_model` K1, Verus on `recover`)                                               |
| `mut_key_id_ignored`                  | K1, K2 (`job_model` K2, Verus on `Table::lookup`)                                     |
| `mut_key_held_starts`                 | K5 (`job_model` J1, Verus on `start_decision`)                                        |
| `mut_spawn_without_claim`             | K10 (`job_model` K10, Verus on `launch`)                                              |
| `mut_orphan_survives`                 | K11 (`job_model` K11, Verus on `recover`)                                             |
| `mut_durable_claim_refused`           | K16 (`job_model` K16, Verus on `launch`)                                              |
| `mut_bad_key_accepted`                | K12 (Verus on `key_valid`)                                                            |
| `mut_valid_key_refused`               | K13 (Verus on `key_valid`)                                                            |
| `mut_replay_refuses_started`          | K14 (Verus on `replay_poll`)                                                          |
| `mut_replay_waits_forever`            | K15 (Verus on `replay_poll`)                                                          |

### The OS boundary (J11)

A group's number is free for any process the moment its last member is collected, and execd is the subreaper, so the reaper
can collect it between the kernel's decision to signal (`Ev::Signal`, `Ev::Stop`, `Ev::Disconnect`) and the `kill` by number
(the pump drain, the `kill` grace wait). Signalling it then would hit a group that is not the job's. So the numbers execd
holds are a kernel ledger (`job_kernel::Held`, keyed by the number and the spawn): a spawn inserts its group under
`SPAWNING` shared; each reaper pass, still under `SPAWNING` exclusive, frees every held number whose group no longer exists;
every send by number (`proc::signal_job`, `proc::kill_group`) checks `may_deliver` under `SPAWNING` shared, so the group
cannot be freed between the check and the `kill`. A refusal means the group is gone and nothing is left to signal
(J11-live: a group with a member is always held). `tests/os_model.rs` model-checks the real ledger against an OS actor
(about 500 states, under a second); its first version keyed the ledger by number only and the model found a stale decision
reaching a later job's group on the same number, hence the spawn in the key. Not covered: a host without the subreaper,
where init reaps orphans outside the exclusion; and the supervisor's end of a job whose number a foreign group now holds
(`job_gone` observes by number, so the end waits for that group; a delay, not a stray signal).

```sh
cargo test -p beyond-execd --test os_model                              # J11, J11-live
cargo test -p beyond-execd --features mut_signal_after_reap --test os_model the_mutation
```

### A restart of execd: the jobs ledger (R1-R5)

A job's processes outlive execd when it is killed or crashes; what died is the table that knew them. With
`EXECD_STATE_DIR` (`Options::state_dir`) set, execd keeps one typed value across its own restart, in two files
(`src/ledger.rs`): `execd-jobs.json`, the table's jobs that have a group and are not removed, each as the job kernel's
`State` plus its address, label and mark, and `execd-jobs.log`, one framed record per change of a job's life since that
file was written (spawn, the shell's exit, the end, `keep`, removal), never marker files. A change appends its record; the
log is folded into a new `execd-jobs.json` when it outgrows the table ("the ledger", under retention, has the cost and the
guarantees). A new execd reads both before it serves (`jobs::recover`) — the file, then the log's records over it, up to the
first that does not hold together — and asks the pure `job_kernel::recover(state, alive, now)` what each job becomes, where
`alive` is whether any process of the group or carrying the mark runs:

| saved                    | group | becomes                                                                        |
| ------------------------ | ----- | ------------------------------------------------------------------------------ |
| ended                    | any   | in the table exactly as it ended: code, signal and time (R1)                   |
| running or exited        | alive | adopted as saved, its group held for signals and watched until it empties (R2) |
| running                  | gone  | ended, code `-1` (`LOST_CODE`): the shell's status was seen by no one (R3)     |
| exited                   | gone  | ended with the shell's saved code and signal (R3)                              |
| no group yet, or removed |       | not in the table (R4)                                                          |

A file that does not parse is read as empty; a log of a file that is not the one there (a fold renamed its file and died
before its log) is dropped, which costs nothing because the file it names already holds those records.

R5: recovering what recovery returned changes nothing, so a restart during a restart is safe. Proven in Verus
(`execd-jobs`, `recover` and `lemma_recover_idempotent`), run by production, and driven through the real binary by
`tests/restart.rs`; four mutations (`mut_recover_drops_live`, `mut_recover_adopts_gone`, `mut_recover_reends_ended`,
`mut_recover_forgets_shell_status`) are red in both. A held start no one kept dies with its client, whose connection went
with the old process: recovery sends it the same `Disconnect` any client leaving does.

What does not survive: a `notify` target and a job's open stdin (a pipe: a job that reads stdin sees EOF). A job's output does
(below): its ledger entry names its spool directory.

### A job's output is a file; a shutdown leaves its groups (S1-S5)

**Output.** A job's stdout and stderr are two files, `out` and `err`, that execd opens with `O_APPEND` and the job inherits
(`src/tail.rs`); there is no pipe. A pipe's reader is execd, so a write after execd died was `SIGPIPE`: a job could not run
through a restart. A file has no reader to lose. `job_kernel::output_sink` decides it (S1: a job's is `Spool`, never a
pipe; S2: a command's is a pipe, which ends with its client), and `exec::command_for` and `start_job` call it.

The files are always under the jobs directory (`EXECD_JOBS_DIR`, a disk), never the state directory (S7: `job_kernel::spool_root`,
which `spool_base` calls: with a ledger the stable directory, else the server's own, and there is no variant for the state directory). With `EXECD_STATE_DIR` they are in `<jobs>/execd-spool-<tag>/<mark>/` (`<tag>` hashes the state directory's path, so two ledgers in one jobs directory never clean up each other; `0700`, files `0600`, `O_EXCL`,
`O_NOFOLLOW`), a directory no execd removes when it stops, which survives the process: the ledger names each job's spool by its
path, so a restart finds it there. That name does not match `execd-jobs-<uid>-<pid>-<random>`, so the stale-directory cleanup at
start never reads it. Without a state directory they are in this execd's own `execd-jobs-...` directory and die with it, as the
jobs' ledger would not have kept them anyway. The state directory holds the ledger and nothing else: in a sandbox it is a tmpfs,
where 8 MiB per job would be RAM and part of every memory snapshot. The spool's directory goes when the job is removed, and at a
restart for every spool no recovered job names (`Spool::retain` over `<jobs>/execd-spool-<tag>`).

**The stable directory is a trust boundary.** Its name is predictable and the jobs directory is shared (`/var/tmp` is world-writable), while execd (root) deletes inside it. `JobsDir::open` makes it non-recursively with mode `0700` and, every start, `lstat`s it: a real directory (never a link), owned by execd's uid, mode `0700`; anything else is refused and execd does not start (fail closed, logged). The directory is opened once (`O_DIRECTORY|O_NOFOLLOW`) and checked with `fstat` on the descriptor, and that descriptor is the only handle used afterwards: `Spool::create` makes a job's directory and files with `mkdirat` and `openat(O_NOFOLLOW|O_CREAT|O_EXCL)`, `Spool::at` and `Spool::retain` open and remove by name relative to it, and the opens that remain path-based (the tail, the journal, the ring) go through `/proc/self/fd/<n>` of the job's own held directory. A link swapped in at the stable directory's name after start is never followed (`a_link_swapped_in_after_start_is_not_followed`). The ledger still records the real name. A process that plants a link or a directory there gets a refusal, and nothing outside is touched.

**The tail.** One task per job reads each file from its own offset into the job's ring (offsets, stderr labels and retention
are still `crates/execd-ring`'s, R1-R6), a buffer from each file in turn, and does all of it (reads, chunk and journal writes, punching) on a thread off the runtime (`proc::off_runtime`), 16 rounds per batch, so a slow disk stalls that job's tail and no connection. Backpressure is the spool file itself: the job writes to disk, never to the tail, so a tail that falls behind lets the file grow (and the job is never slowed), one batch in flight per job. A file gives
no readiness, so one `inotify` instance per server (`tail::Hub`) wakes the job's tail on `IN_MODIFY`: a write reaches the ring
as soon as it reached it through a pipe, with no polling. The task always reads to the end before it waits, so a lost or doubled
wake costs nothing, and a queue overflow wakes every tail. Bytes read are returned to the filesystem in 64 KiB steps
(`FALLOC_FL_PUNCH_HOLE`), so a spool holds what execd has not read plus under 64 KiB per stream.
The settle of a pipe (read until it closes, go quiet 100 ms, or one pipe's worth of late bytes) is gone for jobs: when the shell
exits or the group empties, everything written is already in the files, so execd reads to their ends and is done; a descendant
that holds the output open past its job's end costs nothing. A restarted execd opens a recovered job's files and reads from the
exact byte it stopped at (X1-X3, below), so what the job wrote while execd was down, on each stream and labelled, is read back.

**Resuming at an offset (X1-X3).** Each ledger entry names its job's spool directory (`spool`), and in it, beside the two files,
are the ring's chunk files (`ring/`) and `journal`: the cursors of `out` and `err` at its start, then every piece execd consumed
and has not yet written to a chunk, in the order it ring-ed it, with its stream. It is appended before the bytes read are returned
to the filesystem and replaced (written aside, renamed) whenever the ring flushes a chunk, so it never holds more than the
unfinished chunk and a byte is always in a spool file or the journal. A restarted execd takes the chunks, then replays the
journal, then reads each file from `execd_jobs::resume`: each stream resumes at its cursor plus its journalled bytes (X1), and the
ring ends at the bytes consumed, `out + err` (X2), so a client's offset names the same byte before and after the restart, with
the same stderr labels, and `read` at that offset answers the next byte with no gap. A journalled byte a chunk already holds is
skipped (`replay_skip`, X3), so none is served twice; a crash between a chunk's write and the journal's replacement is that case.
With no journal, or one that disagrees with the chunks, the ring starts empty and the files are read from what they still hold: a
gap, logged. Proven in Verus (`resume`, `replay_skip`; `mut_resume_cursor_lags`, `mut_resume_end_drops_journal`,
`mut_resume_replays_chunked` are red); `tests/restart.rs` drives a crash mid-write and compares the answer before and after.

A job's ring chunks therefore live in its spool (`<jobs>/execd-spool-<tag>/<mark>/ring/` with a ledger), and the disk under it is a
dependency: a job needs room to start (`507` when the disk is full, `500` for any other reason its spool cannot be made), and a job writing to a full disk sees `ENOSPC`, as any
write to a file does.

**Shutdown.** Two signals stop execd, and they differ in what happens to jobs (service-init sends `SIGTERM` for a sandbox stop and
`SIGHUP` for a restart of execd, `workload_signal`). Both stop the accept loop and drop every open connection; the kernel then decides
each registered group (`job_kernel::shutdown_of(kind, recorded, stop)`, `recorded` is whether there is a ledger).

**A stop (`SIGTERM`, `SIGINT`) ends every group (S6)**, a job's included, with or without a ledger: jobs get `SIGTERM`, then `SIGKILL` after
a 300 ms grace (`JOB_STOP_GRACE`), commands are killed, and the ledger is emptied so the next execd adopts nothing.
`mut_term_keeps_jobs` (a stop that behaves as a restart) fails Verus and `tests/restart.rs::a_sigterm_to_execd_ends_every_job_even_with_a_ledger`.

**A restart (`SIGHUP`)** decides as follows (`Handle::restart`); `SIGHUP` for the workload's own commands is unchanged (they start with `SIG_DFL`):

| group                        | signalled | record in the ledger                           |
| ---------------------------- | --------- | ---------------------------------------------- |
| a command (one-shot, stream) | yes (S5)  | none                                           |
| a job, with a ledger         | no (S3)   | kept (S4): the next execd adopts it (R2)       |
| a job, without a ledger      | yes (S3b) | none: nothing could ever find or stop it again |

Only an explicit `kill` or `wipe` signals a job that a restart can find. A held `start` nobody kept dies with its client's
connection (J6); a shutdown is not the client leaving, so that kill waits for the next execd, which ends it as any recovered
disconnect (R2 above). Who restarts execd and keeps its process group out of the supervisor's kill is the supervisor's
(`KillMode=process`, or its own).

Proven in Verus (`execd-jobs`, `output_sink`, `spool_root` and `shutdown_of`): each of these `mut_*` features each turn one clause red
(`mut_job_output_pipe`, `mut_command_output_spool`, `mut_ledger_spool_per_server`, `mut_unledgered_spool_stable`, `mut_shutdown_signals_jobs`, `mut_shutdown_orphans_unrecorded`,
`mut_shutdown_forgets_jobs`, `mut_shutdown_spares_commands`), and `tests/restart.rs` drives the binary through S1, S7 (`a_ledgers_spools_are_in_the_jobs_directory_and_survive_a_restart`, `a_forked_disks_stale_spools_are_removed_before_serving`), S3 (a `SIGHUP`), S4 and S6 (a `SIGTERM`).

**Measured** (`benches/jobs_first_byte.py`, n=40 per run, three runs each, release builds, this 2-CPU ext4/tmpfs host), before
the pipe and after the file:

|                                                                             | before (pipe, pump)      | after (file, `inotify`, tail)                                 |
| --------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------- |
| a running job's write to its arrival at an attached client, p50 / p95       | 0.18-0.20 / 0.22 ms      | 0.21-0.22 / 0.31-0.33 ms                                      |
| first output frame of a held `start`, p50 / p95                             | 6.3-7.1 / 7.5-8.3 ms     | 5.8-6.0 / 6.9-7.2 ms (the same 6 ms of start work; unchanged) |
| a streamed command's first byte (not a job; unchanged path), p50 / p95      | 0.45-0.64 / 0.91-0.94 ms | 0.46-0.79 / 0.89-0.99 ms                                      |
| `true` while 16 jobs write 1 GiB flat out (`benches/jobs_io.py`), p50 / p99 | 1.0 / 94 ms              | 3.5 / 5.9 ms                                                  |
| `read` of 8 MiB from the oldest offset / `attach` from it, MB/s             | 167 / 1,100              | 166 / 1,080                                                   |

The hop through the file and the journal's append costs about 0.03 ms at the median (0.1 ms at p95) on the write-to-client path. Under 16 flat-out writers the median is higher
(a writer is no longer slowed by a full pipe, so the tails have more to read) and the tail is far lower: the pump held the
one thread for a pipe's worth of data per wake, and the tail reads 64 KiB per file per turn and yields.

```sh
cargo test -p beyond-execd --test restart
cargo test -p beyond-execd --features mut_job_output_pipe --test restart      # red
cargo test -p beyond-execd --features mut_ledger_spool_per_server --test restart # red
cargo test -p beyond-execd --features mut_shutdown_signals_jobs --test restart # red
mise run verify:kernels -- --features mut_shutdown_signals_jobs               # red
```

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
| The token stays secret | The bearer is never an argument. Every `EXECD_*` variable is removed from every command's environment (one-shot, streaming, jobs). A bearer handed over in `EXECD_BEARER` is moved out of execd's own environment by re-executing it, and execd exits before it listens when that move fails (see below). No log line, `Debug` output or error carries the bearer, a notify token or a callback URL.                                                                                                                                                                                                                                                     | A process that runs as execd's uid can still read execd's memory and a bearer file execd owns. The observer an embedding host sets sees every request's headers, `Authorization` included; the host is trusted.                                                   | `tests/security.rs`, `tests/conformance.rs` (the whole verifier at `trace` level, then a search of its log)       |
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
- A one-shot or a stream is killed by its process group. A command that leaves its group (`timeout`, `setsid`, a daemon) survives
  the kill of its timeout, a disconnect and the shutdown sweep, with or without `EXECD_RUN_AS`.
- A _job_ is its process group plus every process that carries `EXECD_JOB_MARK=<token>` (the job's own, set in its environment and
  inherited), so `kill`, `wipe`, `survivors` and the job's end cover what `timeout`, `setsid` and a double fork took out of the
  group. The mark is found by reading `/proc/<pid>/environ`: a process that clears its environment (`env -i`, a sanitizing wrapper), or
  one of another uid when execd is not root, is not found, and a job that holds only such a process reads as ended. A cgroup per job
  would close that; execd does not own one, and the second kill path it needs is not taken until a real escape of the scan shows up.
  `tests/job_escape.rs` pins it.
- A capability held by execd as an _ambient_ capability would reach the commands (`EXECD_RUN_AS` from a non-root execd with ambient
  `CAP_SETUID`): run execd as root, or with those capabilities permitted and effective only.

### How it is tested

| test                                            | proves                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/security.rs`                             | the bearer is in none of a command's `env`, its parent's environ, or any readable `/proc/*/environ` (by environment, by file, for streams and jobs); fail-closed start in all its cases, including a bearer that cannot be moved out of the environment; `EXECD_RUN_AS` refusals; as root, a user who cannot read execd's memory, environ or a root-owned bearer file, gets the right uid, groups, `HOME` and directory, and is killed by the shutdown                  |
| `tests/auth_matrix.rs`                          | path, method, `Authorization`, body-framing and smuggling cases over HTTP/1.1 and h2c, each refused and each proven to have run nothing by a command that would create a marker file; a control per family proves the vehicle runs                                                                                                                                                                                                                                      |
| `tests/jobtable.rs`                             | the job table against a model, over random operations (`EXECD_JOBS_SEED`, `EXECD_JOBS_CASES`, `EXECD_JOBS_SECS`, `EXECD_JOBS_OPS`; a failure prints its seed and a shrunk list); at the end of every case, once every scope is wiped, execd is back at its starting state (see `tests/baseline.rs`)                                                                                                                                                                     |
| `tests/jobmark.rs`                              | the mark on the wire: `mark` on start, list and the held start's first frame, `attached{mark,from}`, an attach naming another incarnation refused (404) and its retry, a status-only attach, `remove`'s `running` and `mismatch` and its retry, the id rule                                                                                                                                                                                                             |
| `tests/jobproto.rs`                             | env (applied, hostile shapes refused with nothing run, `EXECD_*` unreachable, `HOME` wins, scope isolation, under `EXECD_RUN_AS` a preload runs only as the target uid), stderr/stdout told apart on one clock with frames and the 1024-range clip, positional stdin writes (retry, overlap, out-of-order over several connections, EOF, after end, 413, bounded waiters, a stalled write does not freeze `list`), the `signal` op (once, scope, ended job, allow-list) |
| `tests/signals.rs`                              | started with `HUP INT QUIT USR1 USR2 TERM ALRM` ignored (as `cmd &` or `nohup` leaves them), every command still dies of, and every job's `trap` still runs on, each of them                                                                                                                                                                                                                                                                                            |
| `tests/baseline.rs`                             | after a mixed workload (one-shot, streams finished and dropped, jobs past the 8 MiB spool, kills, removes, a wipe), `/proc` shows the starting open fds and threads, no zombie child, and an empty jobs directory                                                                                                                                                                                                                                                       |
| `tests/parsers.rs`, `fuzz/` (`request`, `ring`) | no body panics the parsers, every refused body is a `4xx`, and the output ring agrees with a plain buffer for every offset and size                                                                                                                                                                                                                                                                                                                                     |
| `tests/conformance.rs`                          | the full verifier, also at `debug` level (then its log is searched for the bearer) and under `EXECD_RUN_AS`                                                                                                                                                                                                                                                                                                                                                             |
| `tests/oneshot_headers.rs`                      | a one-shot answer to a command that outlives a heartbeat has its headers before the command ends and a body that is one JSON document with leading whitespace; a quick one is unchanged; an error before the start keeps its status; a client that leaves stops the command                                                                                                                                                                                             |
| `tests/output.rs`, `tests/timeout.rs`           | one-shot output is bytes (every byte value on both streams, the empty answer, the deadline note, stdin, the cap on decoded bytes) and `timeout_ms` kills the group and answers 124                                                                                                                                                                                                                                                                                      |

Every test that starts the binary owns it from the moment it exists, and a detached watchdog shell kills execd and everything under it if the test process dies without running destructors (`tests/guard.rs` SIGKILLs a test mid-run, as user and as root, and looks for survivors). The tests that need root run execd through `sudo -n` (or as root) and print `SKIPPED` where a host gives them neither. The long
runs (ten minutes of job-table seeds, two of parser seeds, four of each fuzz target) are in `.github/workflows/deep.yml`.

## Process model

**Signal dispositions.** A launcher that starts execd with `cmd &` or `nohup` leaves `SIGINT`, `SIGQUIT` and `SIGHUP` as
`SIG_IGN`, and an ignored disposition survives `exec`: every command and job would inherit it, a shell could not
even `trap` it, and a customer's `Ctrl-C` to a job would do nothing (E2B starts sandbox-exec that way). A _handler_
is reset to the default by `exec`, so `serve` installs do-nothing handlers (tokio's signal driver, safe) for `HUP`, `INT`,
`QUIT`, `TERM`, `USR1`, `USR2` and `ALRM` before any spawn (`proc::release_signals`), and every command starts with
`SIG_DFL`. The cost: execd itself survives a stray `SIGHUP`/`SIGUSR1`/`SIGQUIT`; it still stops on `SIGTERM` and `SIGINT`,
and restarts on `SIGHUP`, where `main` waits for them (the do-nothing handler is for the commands' sake; `main` listens too). `tests/signals.rs` starts execd with all of them ignored and shows a one-shot dying of
each and a job's `trap` running on each.

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
- **Shutdown.** `SIGTERM` or `SIGINT` (a stop) stops the accept loop, drops every open connection (which drops every in-flight
  handler, and with it the guard that kills that handler's group), ends every job (`SIGTERM`, then `SIGKILL` after 300 ms) and
  kills every command, ledger or not, and exits. `SIGHUP` (a restart) decides every group by `job_kernel::shutdown_of`: commands
  are killed; jobs are left running and kept in the ledger when there is one, killed when there is not (see "A job's output is a
  file; a shutdown leaves its groups"); and exits.
- **Embedding.** The library never sets a subreaper. `serve` starts its own reaper (it listens for `SIGCHLD`); a host that
  wants the PID 1 duties calls `become_subreaper` first, as the binary does.

## Bounds

Every buffer, map, file, task and connection execd holds, what bounds it, what frees it, and what happens under N concurrent
cold callers. No limit here is invented: each is the spec's, the agent's own, or a number read from the kernel.

| what                          | bound                                                                                                                                                                                                                | derivation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | freed by                                                                          | N concurrent cold callers                                                                                                                                                                         |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Request body                  | 91,575,640 bytes (`MAX_REQUEST_BYTES`)                                                                                                                                                                               | The agent's largest request is a file's content as `stdin_base64`: the biggest file it writes is an edit's, `MAX_EDIT_BYTES` = 64 MiB, which is `ceil(64 MiB / 3) * 4` = 89,478,488 bytes of base64. Plus 2 MiB, Linux's default `ARG_MAX`, for the command and arguments the kernel itself refuses past.                                                                                                                                                                                                       | The handler returning                                                             | N times the cap, held only while parsed; the callers are the agent's own tool calls. Over the cap: `413`, read stops at the cap, and `Content-Length` over it is refused before any byte is read. |
| one-shot captured output      | 16 MiB (`MAX_RESPONSE_BYTES`) across stdout and stderr                                                                                                                                                               | The agent refuses a bigger answer (`DEFAULT_MAX_RESPONSE_BYTES`), and a JSON string is never shorter than the bytes it encodes, so nothing the agent would have accepted is lost. Past it, output is read and dropped so the command never blocks, and the answer is a `422` that says the command ran and its output was too large; a prefix is never answered as the result. Stdout and stderr take from that one counter by compare-and-exchange, so concurrent readers cannot both keep the same remainder. | The answer being built                                                            | N times 16 MiB at the worst                                                                                                                                                                       |
| one-shot stdin                | the decoded request body                                                                                                                                                                                             | already counted in the request body                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | the handler returning                                                             | as above                                                                                                                                                                                          |
| streams buffers               | two 64 KiB read buffers, one frame in each of two depth-1 channels, hyper's write buffer (400 KiB, its default)                                                                                                      | A frame carries at most 64 KiB (the spec). Depth 1 is the least that lets the next frame be built while the last is written; nothing observable depends on it.                                                                                                                                                                                                                                                                                                                                                  | the stream ending                                                                 | N streams hold N times a few hundred KiB. A stalled client holds the same, and stalls its command.                                                                                                |
| Settle on exit                | 100 ms quiet on the pipe (measured by the reader, not by whoever waits for it), or what the pipe held at the exit (`FIONREAD`, not its nominal `F_GETPIPE_SZ` size: a pipe can hold more) plus 64 KiB of late writes | The spec's grace is 100 ms (the client's). What a command wrote and had not had read is in its pipes when it exits, however much that is.                                                                                                                                                                                                                                                                                                                                                                       | the readers finishing                                                             | one timer per exiting stream                                                                                                                                                                      |
| Leftover wait                 | until a `SIGCHLD`, with a 5 s fallback, while a shell has exited and its group lives                                                                                                                                 | The reaper collects execd's children at once, so the last member of a group is nearly always collected and a waiter is woken for it; the slow tick (a heartbeat period) covers a zombie whose live parent is neither execd nor in the group, and only sets how late that is noticed. Without a subreaper (an embedding host) it polls every 50 ms.                                                                                                                                                              | the group emptying, the deadline, or the client leaving                           | one `/proc` scan per 5 s tick per waiting stream, only while leftovers live                                                                                                                       |
| Heartbeat                     | every 5 s of silence, or the request's own `Keepalive-Ping-Interval` (1 to 240 s)                                                                                                                                    | The spec.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | the stream ending                                                                 | one timer per stream                                                                                                                                                                              |
| Stream lifetime               | the command's `timeout_ms` (a job stream has none)                                                                                                                                                                   | The spec: the endpoint enforces it and kills the group. A request without `timeout_ms`, or one too large to represent, waits 30 years, tokio's own "never".                                                                                                                                                                                                                                                                                                                                                     | the deadline, the client leaving, or the command ending                           | bounded by the clients                                                                                                                                                                            |
| Job output                    | the last 8 MiB per job (`JOB_OUTPUT_RETENTION`), in whole chunks of 64 KiB **on disk** (`EXECD_JOBS_DIR`), so between 8 MiB and 8 MiB plus one chunk; in memory, one unfinished chunk                                | The spec: the endpoint keeps at least the last 8 MiB.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | a chunk's eviction; the job's `remove`, `kill` with `remove`, or `wipe`; shutdown | N jobs hold N times 8 MiB of files                                                                                                                                                                |
| Job table                     | one entry per job started and not removed                                                                                                                                                                            | The agent keeps at most `MAX_YIELDED_JOBS` (4) + `MAX_INTENDED_JOBS` (8) running and `MAX_ENDED_JOBS` (8) ended per session and removes them with the session, so a session holds at most 20 entries. execd adds no second limit; an ended job whose client never removes it stays until execd exits.                                                                                                                                                                                                           | `remove`, `wipe`, `kill` with `remove`                                            | N sessions hold N times 20 entries, 160 MiB of files at the floor                                                                                                                                 |
| Job `read`                    | at most the output held                                                                                                                                                                                              | A read cannot ask for more than the ring has.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | the answer being sent                                                             | N answers of at most 8 MiB plus a chunk, base64 encoded                                                                                                                                           |
| Job tail                      | one task per job                                                                                                                                                                                                     | It reads the job's two spool files into the ring and ends with the job: the group is empty and the files are read to their ends. Its spool holds the unread bytes plus under 64 KiB per stream.                                                                                                                                                                                                                                                                                                                 | as stated                                                                         | 2N tasks for N running jobs                                                                                                                                                                       |
| Job supervisor                | one task per job                                                                                                                                                                                                     | Ends when the shell has exited, its group is empty and the end is recorded.                                                                                                                                                                                                                                                                                                                                                                                                                                     | the end                                                                           | N tasks for N running jobs                                                                                                                                                                        |
| Callback attempts             | one in flight per job, 30 s each, 12 per count                                                                                                                                                                       | N16, N18: a count is over within 723 s however the receiver behaves.                                                                                                                                                                                                                                                                                                                                                                                                                                            | the verdict, the 12th failure, a new `keep`, or the job's removal                 | N attempts at once, one request each, none pooled                                                                                                                                                 |
| Callback client               | one `reqwest` client, built on the first attempt, with no idle connection kept                                                                                                                                       | A deployment that never keeps a job with `notify` never builds it. A token never sits on a pooled socket.                                                                                                                                                                                                                                                                                                                                                                                                       | process exit                                                                      | built once                                                                                                                                                                                        |
| Process groups registered     | one pid per running command or job                                                                                                                                                                                   | The OS bounds how many processes can run (`pids.max`).                                                                                                                                                                                                                                                                                                                                                                                                                                                          | the command or job ending                                                         | N entries of 4 bytes                                                                                                                                                                              |
| Children table                | one entry per child execd spawned and has not yet seen exit                                                                                                                                                          | The OS bounds how many processes can run (`pids.max`).                                                                                                                                                                                                                                                                                                                                                                                                                                                          | the reaper collecting the child                                                   | N entries (a pid and a oneshot sender)                                                                                                                                                            |
| Connections                   | one task and one socket each, and the requests in flight capped by the next row                                                                                                                                      | The soft `RLIMIT_NOFILE` is raised to the hard limit at start (logged; idempotent). When accept fails with `EMFILE`/`ENFILE` the loop releases a spare descriptor, answers up to 64 waiting connections `503` (`out of file descriptors`) and closes them, then pauses 100 ms; a connection with no complete request head for 60 s is closed (twice the agent's 30 s idle pool timeout); a peer whose data goes unacknowledged for two heartbeat intervals is dropped by the kernel (`TCP_USER_TIMEOUT`).       | the client closing, the head timeout, the user timeout, or a shed                 | bounded by the clients and the descriptor limit                                                                                                                                                   |
| Requests in flight            | `stream_limit` of the soft `RLIMIT_NOFILE`, or `EXECD_MAX_STREAMS`                                                                                                                                                   | An eighth of the limit (at least 64) stays for the listener, the inotify instance, the ledger, the spare and idle connections; the rest is shared at 4 descriptors a request (measured: 3 a stream or one-shot, 4 a job start), never below 1. One past it is answered `503` with `retry-after: 1` and holds nothing; `/livez` is not counted. Admission is one compare-and-swap (`src/admit.rs`, loom models)                                                                                                  | the request finishing, or its connection going                                    | refused, not queued: a client acts on a 503, an unanswered socket is the bug                                                                                                                      |
| HTTP/2 streams per connection | 200                                                                                                                                                                                                                  | hyper's default; not set by execd                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | the stream ending                                                                 | as above                                                                                                                                                                                          |
| Logs                          | one line per request, written when it finishes                                                                                                                                                                       | A stream logs once, when it ends.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | the log sink                                                                      | one short line each                                                                                                                                                                               |
| Observer                      | none unless the host sets one                                                                                                                                                                                        | An unset observer costs one branch per event and builds none of its arguments.                                                                                                                                                                                                                                                                                                                                                                                                                                  | n/a                                                                               | n/a                                                                                                                                                                                               |

### What a stalled client costs

A client that opens a stream and stops reading holds the buffers above for as long as its command lives, and no more: the
command blocks on a full pipe and writes nothing further. The command's own `timeout_ms` ends it, as does the client closing, and
the kernel drops a peer that stopped acknowledging after 10 s of unsent heartbeat. (At a client-chosen interval the stream is silent that long between heartbeats; the user timeout is unchanged.) `tests/endpoint.rs` runs a command that
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
  left to accept with, the accept loop frees its spare descriptor and answers each waiting connection `503` rather than leaving it in the kernel's queue with no answer (`a_connection_with_no_descriptor_left_is_refused_503_not_left_waiting`); the soft limit is raised to the hard limit at start (`the_soft_descriptor_limit_is_raised_to_the_hard_limit_at_start`). The `/proc` scans that run off the runtime
  (`proc::off_runtime`) fall back to running in place when no thread can be created, since `tokio::task::spawn_blocking` panics there.

### Descriptor exhaustion, the cap on requests in flight, and job output off the runtime

**The gate chain for a request**, in the order it meets each gate, what each enforced before this
change, and what it enforces now:

1. **Accept** (`src/http.rs:316` `accept_loop`). Before: connections were not capped, and an accept
   error logged and paused the loop 100 ms; at `EMFILE` the connections already in the kernel's queue
   got no answer at all. Now, on `EMFILE`/`ENFILE`, the loop takes a spare descriptor (opened from
   `/dev/null` at start), accepts up to `SHED_MAX` = 64 (`http.rs:306`) waiting connections, reads the
   request bytes that have arrived (waiting at most `SHED_WAIT` = 20 ms each), writes
   `503` `{"error":"out of file descriptors"}` with `retry-after: 1`, shuts the connection down, then
   reopens the spare and pauses 100 ms as before. What the client sees: a `503` with a reason, or, for
   an h2c client, a failed handshake. Worst case: a connection whose request arrives more than 20 ms
   after accept, or one beyond the 64th in a round, is closed without the body read; the next round,
   100 ms later, answers it (the kernel queue holds it). `EXECD_SHED_WHEN_OUT_OF_FDS=0` restores the
   old behaviour exactly, and `with_shedding_off_a_connection_past_the_descriptor_limit_waits_as_before`
   holds that (it goes red when the knob is ignored).
2. **Header read** (`http.rs:365` `HEAD_TIMEOUT` 60 s): unchanged; a connection with no complete head
   is closed.
3. **Request cap** (`http.rs:413`, `InFlight::enter_within` `http.rs:517`, the counter in
   `src/admit.rs`). Before: none. Now every request except the health paths and one a route hook
   claims takes a slot by one compare-and-swap; the request that finds the count at the limit is
   answered `503` `{"error":"too many requests in flight (limit N); retry shortly"}` with
   `retry-after: 1`, holds no descriptor and no slot, and is logged `request kind=overload`. `/livez`
   is never counted, so health answers at the limit. Sizing is `stream_limit` (`lib.rs:98`): from the
   soft `RLIMIT_NOFILE` after the raise, an eighth (at least 64) stays for the listener, the inotify
   instance, the ledger, the spare and idle connections, and the rest is divided by
   `FDS_PER_REQUEST` = 4 (measured: a stream or a one-shot 3, a job start 4), never below 1.
   `EXECD_MAX_STREAMS`: unset sizes it, `0` is no cap (what ran before; `max_streams_zero_is_no_cap`
   goes red when the `0` arm is removed), `N` is N. Worst case: at the limit a client is refused
   though descriptors remain, which is deliberate: a refusal is answerable in 1 s, an unanswered
   connection is not.
4. **Spawn** (unchanged): no descriptor for a command's pipes is `exit_code` 126 one-shot, `500` for a
   job start, never a hang (`tests/faults.rs`).

**The soft limit** is raised to the hard limit at start (`main.rs:86` `raise_open_file_limit`):
idempotent (nothing is set when the soft is already the hard, or unlimited), logged either way, never
fatal (a refusal logs and the cap is sized from what the process really has), and inherited by the
commands it starts. `EXECD_RAISE_NOFILE=0` leaves the inherited limit, as before.

**Atomicity.** The cap's admission is a compare-and-swap loop over the counter in `src/admit.rs`,
which is loom's atomic under `--cfg beyond_loom`: `mise run test:loom` explores every interleaving of
two and three racing callers and of a release racing an arrival, and the first four models go red
under a load-then-add version of `enter_within`.

**Job output is written off the runtime.** The reads, ring chunk writes, journal appends and hole
punching (`tail.rs` `Tail::drain_batch`, driven from `jobs.rs` `Tailing::flush`) run on a thread
(`proc::off_runtime`), in batches of up to `tail::ROUNDS` = 16 rounds per file, and each batch holds
one of `disk_writers` turns (`Ctx::disk_turn`) — the same turn the retention sweep's ledger write and
unlinks take (`jobs.rs` `sweep`), so there is one gate for all off-loop disk work and the threads are
bounded however many jobs write and sweeps run. Nothing is queued in memory: the spool file is the
queue, written by the job itself, so a slow disk makes the tail fall behind and the file grow, and
never blocks another connection. A tail that finds no turn waits for one (logged at debug).
`EXECD_DISK_WRITERS=0` is the behaviour before this change — the batches on the runtime, one round at
a time, yielding between rounds — and
`with_disk_writers_zero_a_stuck_output_write_stalls_every_caller_as_before` holds it.

A job's **end** is recorded after its last flush (`supervise_after` → `flush`), so a disk that stops
answering defers that job's `ended` state, and its retention, until the disk answers again. It defers
no other job's end and never stalls the runtime; before this change the same wait ran on the runtime
and stalled every caller instead.

**The retention sweep takes the same turn.** Its decision is made in memory first (the output is
evicted and the spool taken), then `Ctx::disk_turn` before the closure that writes the ledger and
unlinks: with every turn held by a stuck batch, the decision shows at once (a read is a gap) and its
durable side waits, bounded by the disk answering. It is one pass at a time by construction (one
retention task, sequential sweeps), and the ledger-before-unlinks order is unchanged. The ordering
across a crash is `tests/retention.rs` `a_crash_after_the_ledger_write_leaves_no_spool_behind`
(`sweep_after_save=exit`); the turn is `a_retention_sweep_waits_for_a_disk_writer_turn`, red when
`disk_turn` is deleted.

**Ordering and durability** are the ones the ring and the journal already had: one tail task per job,
one batch at a time, the journal appended before the batch punches the spool's bytes, and nothing
else writes the ring. Ordering across batches is covered by
`output_past_one_batch_keeps_its_order_and_every_byte` (4.9 MB through a stuck disk, every byte once,
contiguous offsets), which goes red when a batch advances its offset short of what it read; durability
across a crash is `tests/restart.rs` (`a_client_resumes_at_its_offset_across_a_crash_with_no_gap`),
which goes red when the journal append is removed.

**Tests** (`tests/faults.rs` unless named): the gate-seam pair
`a_stuck_output_write_does_not_stall_an_unrelated_caller` /
`with_disk_writers_zero_a_stuck_output_write_stalls_every_caller_as_before`;
`writer_threads_are_bounded_by_disk_writers_and_the_rest_wait`;
`output_past_one_batch_keeps_its_order_and_every_byte`;
`a_connection_with_no_descriptor_left_is_refused_503_not_left_waiting`;
`with_shedding_off_a_connection_past_the_descriptor_limit_waits_as_before`;
`a_request_past_the_cap_is_answered_503_with_a_reason_and_recovers`; `max_streams_zero_is_no_cap`;
`a_stingy_descriptor_limit_sizes_the_cap_and_every_stream_is_answered` (soft 64, hard 128, so the
raise and the sizing are seen on a limit this machine does not have);
`the_soft_descriptor_limit_is_raised_to_the_hard_limit_at_start`;
`with_raise_nofile_zero_the_soft_limit_is_left_as_inherited`; `stream_limit` (in `lib.rs`) and the
`admit` loom models. The stuck-disk seam is `EXECD_TEST_FAULT` `tail_gate=<dir>` (debug builds only,
like every fault point): a batch with output to write creates `<dir>/held-<n>`, waits while
`<dir>/closed` exists, records the most ever held at once in `<dir>/most`, and leaves. No test sleeps
to reach a state.

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

### Job stdin, streams, signal and env (`benches/jobs_stdin.py`)

Loopback, keep-alive HTTP/1.1, release builds, n = 2000 per row, 3 rounds with the baseline (`f8dd7249`) and this tree alternating
per round (so drift lands on both); date 2026-11-17, load average 17 to 19 from other work on the host (runnable threads 1 to 3
at start and end), exclusive `homelab-live`, `instd-suite` and all `rust-build` slots held. Raw JSON: `benches/results/2026-11-17-jobs-stdin.json`.

| bar        | measure                                                                                                                                                          | result (p50 / p95, per-round medians)                                                                                                                                                                                       | verdict                                                                                                                                                                                                                                                                               |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B1         | `list` op, keep-alive                                                                                                                                            | new 0.047-0.049 / 0.064-0.080 ms; baseline 0.047-0.054 / 0.055-0.077                                                                                                                                                        | baseline                                                                                                                                                                                                                                                                              |
| P1         | `write` 1 KiB to a `cat >/dev/null` job                                                                                                                          | 0.052 / 0.069 ms against B1 0.050-0.072 / 0.056-0.095 (the same run's `list`): about 1.0-1.05 x                                                                                                                             | met (<= 1.15 x, <= 1.5 x)                                                                                                                                                                                                                                                             |
| P2         | `write` 1 KiB, until the bytes arrive on an open `attach`                                                                                                        | 0.065 / 0.084 ms = 1.3 x / 1.5 x B1                                                                                                                                                                                         | met (<= 2 x, <= 3 x): no polling tick                                                                                                                                                                                                                                                 |
| P3         | 1 GiB as 128 KiB writes, 8 in flight on ONE h2c connection (`benches/jobs_stdin_h2.rs`), `cat >/dev/null`; B3 = one 16 MiB one-shot stdin on the same connection | 684-734 MiB/s against B3 181-211 MiB/s: 3.2-3.8 x B3                                                                                                                                                                        | met (>= 0.7 x). A one-shot is bounded by one request's JSON + base64 + one spawn; writes pipeline, so the constraint of P3 is the same JSON + base64 per request, not the pipe, which stays 64 KiB. Load 14-15, 1 to 4 runnable threads                                               |
| P4         | threads and fds, 10,000 writes                                                                                                                                   | 1 thread, 17 fds before and after                                                                                                                                                                                           | met                                                                                                                                                                                                                                                                                   |
| P5 idle    | 100 idle jobs: what `stdin_open` adds per job                                                                                                                    | +0.6 KiB per job (38.8 against 38.2 KiB without `stdin_open`). The 4 KiB bar is the cost stdin adds, by ruling; a job's own 38 KiB (two pump buffers, a table entry; 46.5 KiB on the baseline) is not stdin's               | met                                                                                                                                                                                                                                                                                   |
| P5 stalled | a job that never reads, N queued 128 KiB writers, RSS growth after 3 s and 13 s                                                                                  | N = 0: 0 KiB; 1: 164; 4: 888; 8: 1568; 16: 2980 KiB = 186 KiB per writer = 1.45 x the payload, flat from 3 s to 13 s (the 4-copy build before: 516 KiB per writer = 4.1 x). 16 writers cost 2.9 MiB against the 8.0 MiB bar | met with room (36% of the bar). A queued write holds its decoded bytes and a small header: the body is read into one buffer sized by `Content-Length`, parsed, dropped, and a write's base64 text is decoded once in `jobs::serve` and removed from the request before anything waits |
| P6         | idle wakeups, 100 stdin-open jobs, 20 s                                                                                                                          | 0.0 switches/s (baseline 0.0)                                                                                                                                                                                               | met                                                                                                                                                                                                                                                                                   |
| P7         | one-shot `true`; job start + read + remove; one-shot `cat` with stdin                                                                                            | 0.94-0.99 / 1.14-1.19 ms vs 0.95-0.97 / 1.17-1.18; 9.3-10.5 ms vs 9.5-10.0; 1.10-1.13 vs 1.08-1.14 ms                                                                                                                       | within noise                                                                                                                                                                                                                                                                          |

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

**Route hook.** `Options::route_hook` is `Arc<dyn Fn(HttpRequest) -> Route + Send + Sync>`, where `Route` is `Claimed(future of a
Response<RouteBody>)` or `Declined(HttpRequest)` (the request handed back untouched). It sees every request except `/livez` and `/healthz`
after the authorizer or bearer has accepted it, and before any of the body is read. A refused request gets `401` and the hook never sees
it (so with a hook set a refused non-`POST` is a `401` rather than the unchecked `405`; without one nothing changes). A declined request
goes on exactly as without a hook, the authorizer not run a second time. A claimed request is **not counted**: `service_fn` in
`http.rs` returns before `InFlight::enter`, so it adds to neither `Activity::in_flight` nor the connection's idle timer, and a stream
that handler holds open for hours keeps neither the server nor a sandbox awake. An embedder whose claimed work must keep the host up
says so itself. The future resolves to the response head, which is sent at once; the body then streams while the handler reads the
request body, so the exchange is full duplex over HTTP/2. The request body is not bounded by `max_request_bytes`; the handler bounds
its own. The hook is generic: execd knows nothing of what it serves. The exec protocol's "every `POST` is exec" holds for the
reference endpoint, which claims nothing; an embedder that claims a path owns it, and the agent never sends one.

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
