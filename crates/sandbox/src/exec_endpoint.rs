//! Executing commands against a remote endpoint — any endpoint.
//!
//! The agent is handed a **URL** and it POSTs commands to it. That is the entire contract. It does not
//! know, and must never know, what is on the other side: Daytona, E2B, Modal, a container behind an
//! internal service, someone's laptop over a tunnel. Nothing here is specific to any vendor.
//!
//! ## Not the agent's business
//!
//! **Lifecycle.** Creating, warming, snapshotting, pausing or destroying a sandbox is the caller's job.
//! The agent receives an endpoint that already works and uses it until told otherwise. An agent that
//! provisions infrastructure has to model quotas, billing, placement and cleanup — none of which has
//! anything to do with running a coding task, and all of which differ per vendor.
//!
//! **Co-location.** The endpoint is a URL, so the agent runs wherever it likes and the sandbox runs
//! wherever it likes. Any design that shells out to a locally-installed control-plane binary quietly
//! requires the agent to sit on the compute node beside the workloads it is meant to be isolated from,
//! which is backwards.
//!
//! ## The protocol
//!
//! One `POST` to the configured URL:
//!
//! ```jsonc
//! // request
//! { "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
//! // request carrying stdin — standard base64, fed to the command, then closed
//! { "command": "cat", "args": [], "timeout_ms": 10000, "stdin": "c3RkaW4=" }
//! // response — 200: stdout and stderr are standard base64, the empty string for no output
//! { "exit_code": 0, "stdout": "aGkK", "stderr": "" }
//! ```
//!
//! The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.
//!
//! **Bytes.** `stdout` and `stderr` are the bytes the command wrote, as **standard base64 with
//! padding** (RFC 4648 section 4), the empty string when there is none, both present on every `200`
//! answer. There is one shape and no flag: the endpoint MUST NOT apply a lossy conversion (UTF-8
//! replacement, newline translation, trimming, truncation) to what they carry, and the decoded bytes
//! are the bytes the command wrote. A client reports an error for an answer whose `stdout` or
//! `stderr` is missing or is not standard base64, never reads it as empty. `stdin` is standard
//! base64 too; an endpoint answers `400` to one that is not. Everything else in the answer is as
//! ever: `exit_code` is the command's, `128 + S` for a command killed by signal `S`, `124` for one
//! that hit its deadline (with a note, below), `127` for one that was not found and `126` for one that could not be run (below), with the
//! reason as its stderr (see below); none of these is an error. [`stream`] frames carry their output as base64
//! alike, so every path moves the same bytes. `tests/fixtures/exec/output.json` holds the answer
//! vectors every client decodes.
//!
//! **A command that cannot be started** has the shell's status and a reason on stderr that **names
//! what failed, as the caller sent it**. `exit_code` is **127** when `command` is not a program the
//! endpoint can find (`execd: pytest -q: command not found`; the usual cause is a whole command line
//! in `command`) or when `cwd` cannot be entered (`execd: cwd /no/dir: No such file or directory`,
//! `... Not a directory`), and **126** when the program is there and cannot be run (`execd: /:
//! Permission denied`, a script with a bad interpreter, `Argument list too long`). A bare OS error
//! (`No such file or directory (os error 2)`) names neither and is not conforming. For a missing
//! program the endpoint MUST put `command` and the words `command not found` in stderr; for the
//! 126 cases it MUST still name `command`. The streaming surface carries the same text in a
//! `stderr` frame before the final frame with the same status.
//!
//! **Timeouts.** `timeout_ms` is the endpoint's to enforce: at the deadline it MUST kill the
//! command's whole process group and SHOULD answer 200 with `exit_code` 124 and the note
//! `command exceeded its timeout` appended to the stderr bytes (a newline first if stderr did not
//! end in one). A request without `timeout_ms` has no deadline but the connection.
//!
//! Deliberately the smallest thing that can carry a command and its result. Putting an adapter in
//! front of a vendor's SDK is a few dozen lines on their side, and that seam is where vendor
//! specifics belong — not in here. An endpoint implements the whole of this document: the one-shot
//! call, streaming, jobs and job notify. A client sends what it needs and probes nothing.
//!
//! **The response is capped** ([`DEFAULT_MAX_RESPONSE_BYTES`], `--exec-max-response-bytes`). The body
//! is read incrementally and reading stops at the cap; an oversized response is an **error**, never a
//! partial result. The body is one JSON object, so a prefix of it cannot be parsed — there is no
//! honest partial `ExecResult` to return (not even the exit code, which may come after the output),
//! and presenting whatever fit as the command's output would be a confident wrong answer. The error
//! says the command *did run*, so the caller narrows its output instead of blindly repeating a
//! command that may not be idempotent. The cap is on the **decoded** bytes, `len(stdout) +
//! len(stderr)` together, whatever the base64 or the JSON around it weighs: an endpoint with a cap
//! MUST answer a command that wrote more as `422`, naming the cap and saying the command ran, never
//! with a prefix. A client reading the body incrementally MAY stop at `4 * ceil(cap / 3) + 4096`
//! body bytes (the base64 of a cap's worth of output and the envelope around it) and treats that as
//! over the cap; it checks the decoded sum once it has the answer. The cap applies to one-shot
//! answers only; a stream is read a frame at a time and never buffered whole.
//!
//! ## Streaming
//!
//! A tool that behaves differently depending on where it runs is a broken API. Over a one-shot
//! call, `bash` cannot see a remote command's output until it exits, so it cannot tell a silent
//! build from a hung one (the idle yield never fires: a hung command burns its whole timeout),
//! cannot show progress, and cannot kill a command it has given up on. A stream sends the output
//! as it is produced and makes **the connection the command's lifetime**, so the endpoint stays
//! stateless — no job ids, no poll or kill routes.
//!
//! **Request.** The request plus `"stream": true`:
//!
//! ```jsonc
//! { "command": "sh", "args": ["-c", "cargo build"], "cwd": "/work", "timeout_ms": 2147483647, "stream": true }
//! ```
//!
//! **Response.** `200` with `Content-Type: application/x-ndjson`, a body of frames, one JSON object
//! per line, each terminated by `\n` (chunked transfer on HTTP/1.1, DATA frames on HTTP/2; no
//! `Content-Length`). A frame carries exactly one of:
//!
//! | frame | meaning |
//! |---|---|
//! | `{"stdout": "<base64>"}` | bytes the command wrote to stdout |
//! | `{"stderr": "<base64>"}` | bytes the command wrote to stderr |
//! | `{}` | heartbeat: the connection is alive; never output |
//! | `{"shell_exit": N}` | the command exited with code N, but processes of its group run on — see "Leftover processes"; not final, never output |
//! | `{"exit_code": N}` | **final**: the command exited with code N |
//! | `{"signal": S}` | **final**: the command was killed by signal S (9 for SIGKILL). `128 + S` must fit an `i32`; a wider S is a broken frame |
//! | `{"timed_out": true}` | **final**: the endpoint killed the command at `timeout_ms` |
//! | `{"error": "<message>"}` | **final**: the endpoint failed after answering 200 (lost the sandbox) |
//!
//! - **A 200 that is not a stream.** A client MUST fail a stream request whose `200` does not carry
//!   `Content-Type: application/x-ndjson` as `protocol` (FILES-OVER-EXEC.md section 3.2), whatever the
//!   body holds: it never reads that body as a one-shot answer. A non-2xx answer is an error as for a
//!   one-shot call.
//! - **A line that is not a frame** (not a JSON object, an output field that is not standard base64,
//!   a line past the client's limit with no newline) is `protocol` for a client (FILES-OVER-EXEC.md
//!   section 3.2); a stream that stops (no final frame, `{"error"}`, silence) is `transport`.
//! - Output is **standard base64 with padding** (RFC 4648 §4), so arbitrary bytes — invalid UTF-8,
//!   NULs, terminal escapes — survive. An output frame MUST NOT be empty and MUST carry at most
//!   64 KiB of decoded bytes ([`stream::MAX_FRAME_PAYLOAD`]); split anything longer.
//! - **Exactly one final frame, last.** After it the endpoint MUST end the body. A body that ends
//!   without one is a broken stream, never a success. Failures *before* answering are non-2xx
//!   statuses exactly as for a one-shot call; a command that cannot be spawned is an ordinary result (`stderr`
//!   frame, then `{"exit_code": 127}` or `126`), as a shell reports it.
//! - **Ordering.** Frames of one stream are in the order the bytes were read; the relative order of
//!   stdout and stderr is best-effort, as between two local pipes. The final frame comes after all
//!   output read from both pipes. Once the command's process group is empty, the endpoint SHOULD
//!   stop waiting for pipes held open by a descendant that left the group after a short grace (the
//!   client uses 100 ms) rather than until they close.
//! - **Flushing.** Each frame MUST reach the connection without buffering: the endpoint MAY coalesce
//!   output read within 100 ms into one frame, and MUST NOT hold any longer. Anything between the
//!   endpoint and the client must not buffer either (send `Cache-Control: no-store`, and e.g.
//!   `X-Accel-Buffering: no` behind nginx) — the conformance verifier refuses an endpoint whose frames arrive
//!   all at once.
//! - **Heartbeat.** When no frame has been written for [`stream::HEARTBEAT_INTERVAL`] (5 s), the
//!   endpoint MUST write `{}`. It keeps a silent command's connection alive through proxies (whose
//!   idle timeouts start around 60 s), and it bounds how long a vanished client can go unnoticed. A
//!   client MUST NOT treat a heartbeat as output — `bash` measures a command's silence on output
//!   frames only — and this client treats 30 s with no frame at all as a dead connection.
//! - **Unknown fields are ignored**, and a frame with no known field is a heartbeat, so a later
//!   version can add fields (a sequence number, timings) without breaking this client.
//! - **Timeouts.** `timeout_ms` is the endpoint's to enforce: at the deadline it MUST kill the
//!   command's whole process group and send `{"timed_out": true}`. The client's own deadline is
//!   `timeout_ms` + 30 s for the whole exchange, so the endpoint's verdict normally wins. `bash`
//!   jobs have no deadline — they end with the turn, the run or the session, or a `bash_job` kill,
//!   all of which close the stream — so a streamed job sends [`STREAM_DEADLINE`] (`i32::MAX` ms,
//!   ~24.8 days, the most a Node endpoint's `setTimeout` can hold): the connection is the lifetime,
//!   and the deadline only bounds one that outlives every session.
//!
//! **Disconnect kills — normative.** Every streamed command MUST run in its own process group (or an
//! equivalent the endpoint can kill as a unit, such as a cgroup). **If the connection closes before
//! the final frame, the endpoint MUST kill that whole group** — SIGKILL, grandchildren included —
//! within two heartbeat intervals (10 s), and SHOULD do so at once by watching the connection for
//! EOF or `RST_STREAM` rather than waiting for a heartbeat write to fail. This is how the agent stops
//! a command: `bash_job` kill, a cancelled call, a dropped job and the agent's own exit all close the
//! stream (see [`Kill`]), and nothing else is ever sent.
//!
//! **Leftover processes — normative.** A command often exits leaving processes it started running
//! (`server &`, `nohup … &`, a daemon that stays in its group). Locally those become a job the agent
//! tracks and stops with the session; remotely they must too, or `nohup server &` holds its port
//! until the sandbox dies. So when the command's own process exits **while its process group still
//! has a live (non-zombie) member**, the endpoint:
//!
//! - MUST send `{"shell_exit": N}` — N the command's exit code, `128 + S` if signal S killed it —
//!   after the output read from both pipes up to that exit (the same short grace as before a final
//!   frame);
//! - MUST then keep the stream open — output, heartbeats, `timeout_ms` and disconnect-kills all
//!   still apply — until the group has no live member, and only then send the final frame, which
//!   carries the command's own result (`{"exit_code": N}`, or `{"signal": S}`).
//!
//! A group that is empty when the command exits gets the final frame directly; `shell_exit` is
//! never sent for it. Zombies do not count: a sandbox whose pid 1 is not an init (`sleep infinity`)
//! never reaps the orphans it inherits. A process that leaves the group (`setsid`) is not covered,
//! unless the endpoint tracks a cgroup (it MAY). The client adopts the leftovers as a job at
//! `shell_exit` (`on_orphans`), and closes the stream to stop them, like any job. A call that is not
//! a job (nothing tracks it) returns at `shell_exit` and closes the stream, so the endpoint kills
//! the leftovers.
//!
//! **A one-shot call's leftovers are untracked.** One request, one response: the endpoint answers
//! when the command exits, and nothing the agent can do afterwards reaches what it left running. An
//! endpoint SHOULD kill the command's process group once it has answered; one that does not leaves
//! `server &` running until the sandbox ends.
//!
//! **Incremental delivery.** A stream whose frames arrive all at once is no stream: `bash` measures a
//! command's silence on them. The conformance verifier runs `sh -c 'printf a; sleep 0.3; printf b'`
//! with `"stream": true` and requires `application/x-ndjson` with at least two output frames, `ab`,
//! `{"exit_code": 0}`, and the first output frame at least 100 ms before the final one; an endpoint
//! or proxy that buffers sends one frame, or every frame at once, and fails it.
//!
//! **Conformance.** [`conformance::verify`] checks an endpoint against all of the above — framing,
//! incremental delivery, exit/signal/timeout frames, heartbeats, the 64 KiB frame cap, kill on disconnect
//! measured against a grandchild's marker file, and leftover processes (`shell_exit`, the stream
//! held until they exit, killed on a disconnect after it) — and `tests/exec_conformance.rs` runs it against any
//! URL. `tests/fixtures/exec/frames.json` holds the decoding vectors.
//!
//! ## Sandbox jobs
//!
//! A stream is the right owner for a command someone is waiting on and the wrong one for a job: a
//! replica that is reaped, redeployed or failed over closes every stream it holds. So a `bash` job
//! the model meant to keep (`run_in_background`, an adopted `cmd &`) is owned by the **sandbox**: it
//! runs until it exits, is killed, or its session ends, and any replica serving the session can
//! list it, read its output from an offset, and kill it. The endpoint keeps them: a vendor's
//! background commands, the platform's own guest, `execd`. [`jobs_native`] is the client.
//!
//! **Requests.** Every job operation is a request (the same URL, headers and envelope) plus a
//! `"job"` object naming `op` and `scope`. A **scope** is an opaque name for one session's jobs
//! (the agent sends a digest of tenant and session); an operation MUST only ever see the jobs of
//! its own scope. A job **id** is chosen by the agent (lowercase hex, at most 32 characters) and is
//! unique within its scope. A **status** is a job's exit code, or 128 + N for signal N; a job a signal
//! killed also reports **`signal`**: N (see "States"). Except for
//! `start`, the request's `command` is `true` with no arguments: harmless if run. Unknown fields are
//! ignored. A refusal before answering (an unknown job for `attach`, a bad request) is a non-2xx
//! status, as for a one-shot call.
//!
//! | op | `job` fields | answer |
//! |---|---|---|
//! | `start` | `id`, `hold`, `label` — the command is the request's `command`/`args`/`cwd` | `hold: true` (with `"stream": true`): a job stream; `hold: false`: `{"started": pgid}` |
//! | `attach` | `id`, `offset` (with `"stream": true`) | a job stream from `offset` |
//! | `read` | `id`, `offset`, `max` | `{"state", "code", "signal"?, "offset", "end", "data"}` |
//! | `keep` | `id`, `num`, `kind`, `seen` | `{}` |
//! | `seen` | `id`, `offset` | `{}` |
//! | `kill` | `ids`, `remove`, `grace_ms` | `{"survivors": [{"id", "pids": [..]}]}` |
//! | `list` | | `{"jobs": [{"id", "state", "code", "signal"?, "kept", "num", "kind", "seen", "started", "ended", "label"}]}` |
//! | `remove` | `ids` | `{}` |
//! | `wipe` | `grace_ms` | `{"survivors": [{"pids": [..]}]}` |
//!
//! - **A job** runs in a process group (or cgroup) of its own, its stdin empty, its stdout and stderr
//!   one combined **output**, of which the endpoint MUST keep at least the last 8 MiB, addressed by
//!   byte offset from the job's first byte. Offsets never move: dropping old output leaves a gap.
//! - **States** (`read`, `list`): `running`; `exited` — the shell exited with status `code` but
//!   processes of its group run on; `ended` — the group is empty, `code` the shell's status and
//!   `ended` its epoch second; `lost` — the endpoint no longer knows how it ended; `gone` — no such
//!   job. `started` is the job's epoch second, `label` what `start` sent.
//! - **Signal.** `signal` is an integer member of a `read` answer and of a `list` entry, present
//!   only in `exited` and `ended`, and only when a signal killed the shell: `signal` is that signal
//!   (9 for SIGKILL) and `code` is `128 + signal`, the shell's convention. A shell that exited
//!   reports `code` and no `signal` (the member is absent, never `null` or `0`), whatever the code:
//!   an exit of 137 and a death by signal 9 differ only by `signal`. A job stopped by `kill` reports
//!   the signal that killed it (15, or 9 when it outlived the grace period). It is the same fact a
//!   stream's `{"signal": S}` final frame carries, and a job stream ends with that frame, in place of
//!   `{"exit_code": 128 + S}`, for such a job. `shell_exit` stays a code (`128 + S`). The callback
//!   body carries no status, so it has no `signal`.
//! - **Held.** A job started with `hold: true` lives as long as its stream unless it is **kept**: if
//!   the connection closes before the job's final frame and no `keep` arrived, the endpoint MUST kill
//!   its whole group, exactly as "Disconnect kills". `keep` hands it to the endpoint for good
//!   (recording `num`, `kind` and `seen` for `list`); after it a closed stream kills nothing. A job
//!   started with `hold: false` is kept from the start. Closing an `attach` never kills.
//! - **A job stream** is a stream (`application/x-ndjson`, heartbeats, frame size, flushing,
//!   unknown fields) with two more frames: `{"started": pgid}` first on a held start, and
//!   `{"gap": n}` when `n` bytes of output from the requested offset are no longer kept. Output
//!   frames (`stdout`; `stderr` is accepted too) carry the combined output in order. `{"shell_exit":
//!   N}` follows the output read up to the shell's exit when the shell exits leaving its group alive,
//!   once per stream. The final frame — after all output up to the job's end — is the job's status:
//!   `{"exit_code": N}` (or `{"signal": S}` for a shell a signal killed), or `{"error"}`. An `attach` of an ended job streams
//!   what is left from `offset` and the final frame at once.
//! - **`read`** returns up to `max` bytes from `offset` (or from the oldest byte kept, reported in
//!   `offset`, when that is later) as `data` in base64, and `end`, the offset of the next byte.
//! - **`kill`** sends `SIGTERM` to each named job's group, waits up to `grace_ms` for it to empty,
//!   sends `SIGKILL`, waits up to a second more, and reports the live members of any group still not
//!   empty. It MUST NOT signal a group whose job has ended (its number may be reused). With `remove`,
//!   it then deletes each job whose group is empty. `remove` deletes ended jobs and never a running
//!   one; `wipe` kills every job of the scope and deletes them all.
//!
//! Mapping a vendor's own primitives is the adapter's business; two that fit: **E2B** background
//! commands (`commands.run(cmd, background=True, timeout=0)` returns a handle whose `pid`
//! `commands.connect(pid)` reattaches to; `commands.list()`, `kill()`) — its streamed output reaches
//! only the process that started the command, so an adapter redirects it to a file to serve
//! `attach`/`read` from offsets; **Daytona** session commands (`execute_session_command` with
//! `run_async`, `get_session_command` for the exit code, `get_session_command_logs` to stream logs;
//! its docs name no kill for one command, so an adapter kills the process group itself).
//!
//! [`conformance::verify`] checks it: a held job's
//! live output and end, a dropped hold killing an unkept job (measured on the far side), a kept job
//! surviving its hold and reading back from an offset (read and attach), `kill` reaching a job that
//! ignores `SIGTERM`, a job's `signal` told from an exit of 128 + S, and `list`/`wipe` staying inside
//! their scope. `tests/fixtures/exec/jobs.json`
//! holds the decoding vectors.
//!
//! ## Sandbox job notify
//!
//! A replica holds a stream to every kept job it serves, and the endpoint writes a heartbeat
//! on it at least every 5 s for as long as the job runs. That traffic keeps the connection open,
//! and it keeps awake a sandbox that would otherwise sleep: an idle dev server holds its sandbox up
//! for as long as its session stays open (a client attached, or kept alive; a detached session in
//! service mode is reaped after 60 s, which drops its streams). Notify has the sandbox report when a
//! kept job's state changes, so the replica can let go of the stream and send nothing about the job
//! until then.
//!
//! **Notify changes how a replica learns of a job's end, and nothing else.** What a session does
//! with an end, or with leftover processes, is exactly what it does when a held stream or a follow
//! tells it: the same notice, through the same path, starting or joining a turn, whether or not a
//! client is attached. Notify is only ever available in service mode (N36), so everywhere else
//! nothing changes at all.
//!
//! **The one accepted difference.** After a job is unheld, an end that no callback delivers (the
//! sandbox died, stopped or was replaced, every attempt failed, or a superseded owner that has not
//! yet noticed took the callback) is noticed at the session's next real use of its sandbox (N29), a
//! `bash_job` call about the job, or the session's next open, where a heartbeat would have noticed
//! it within seconds. Then it takes the held stream's path, its error notice included. Until then
//! the session still counts the job as running: a `bash` start that the limit on running jobs
//! refuses names it among them. Nothing here contacts a sandbox in the background to notice
//! sooner: any periodic contact would keep the sandbox awake, which is the cost notify exists to
//! remove.
//!
//! Every requirement in this section is tagged in bold (N1, N2, …) and names the check that tests
//! it: a conformance check ([`conformance::verify`], against any endpoint), a test in this repo, or
//! vectors in `tests/fixtures/exec/notify-*.json`. `tests/fixtures/exec/notify-checks.json` maps every
//! tag to its check and `tests/exec_notify_vectors.rs` fails when a tag has none. The edge's half
//! is FLEET.md's "Job-end callbacks", tagged E1, E2, ….
//!
//! **A callback is a hint, not the truth.** It says only "look at job `id` of scope `scope`". The
//! receiver learns what happened the way a reattach does, with `list` and today's follow. So a lost,
//! late, duplicated, reordered or forged callback can delay a report, but it can never make one
//! wrong. Everything below follows from that.
//!
//! **The protocol defines messages, not a transport.** The messages are the `keep` extension, the
//! callback request and its answer, and the token. The callback is written here as an HTTPS request
//! the sandbox dials to the fleet's edge. A sandbox the fleet cannot dial, such as one in a
//! customer's own network that keeps a tunnel open to the fleet, carries the same request and the
//! same answer over that tunnel, which is not specified here.
//!
//! **Terms.** A *notification* is the delivery of callbacks for one job: one or more *attempts*,
//! each one HTTP request. A job's *state* is what `read` and `list` report (`running`, `exited`,
//! `ended`, `lost`, `gone`). *Observable* means that a `read` or `list` of the job that starts after
//! that moment reports it. A job is *proving*, *held* or *unheld* as the release table says
//! ("Release"). *Unheld* is notify's own state, a kept job whose stream the replica closed after its
//! proof; it is not `JobShared::released`, which today marks every kept job, held or not. A job is
//! *taken* while a tool call in flight (`bash_job`) has it out of the session's job table. *Today's
//! follow* is what a replica does when a kept job's held stream breaks (`tools::bash_sandbox`'s
//! `follow`: an `attach`, with its own failure policy and error notice). The *session id* is the one the session is routed by, which the URL names.
//!
//! ### The endpoint's guarantee
//!
//! Notify is part of the protocol, not an option an endpoint advertises, so the guarantee is a fact
//! of it: a kept job survives its sandbox going idle (it keeps running, or is paused and resumed
//! intact). Releasing a job removes the heartbeat that kept the sandbox busy, so a sandbox that
//! stops or kills idle processes (a vendor's auto-stop) would lose the job. An adapter for such a
//! sandbox either keeps it awake itself or does not implement this contract.
//!
//! **A replica never relies on reachability.** It sends `notify` with every `keep` it can
//! name an origin for, and releases the job's stream only when the first notice has actually
//! arrived (N35). A sandbox that cannot dial out, or an origin that is wrong, sends no first notice,
//! so the stream stays held and the job's end is observed over it, as before notify (N38).
//!
//! - **N1** A kept job MUST survive its sandbox going idle, with nothing sent to the endpoint for
//!   a window of the verifier's choosing (30 s by default). (check: conformance `a kept job
//!   survives idling`)
//!
//! ### `keep` with `notify`
//!
//! `keep` gains one optional field:
//!
//! ```jsonc
//! { "command": "true", "args": [], "timeout_ms": 30000,
//!   "job": { "op": "keep", "scope": "3f…", "id": "0123456789abcdef", "num": 3,
//!            "kind": "background", "seen": 120,
//!            "notify": { "url": "https://edge.example/_beyond/agent?session_id=a07-s3.0f1e&tenant=acme",
//!                        "token": "bjn_v1.5d1c0a39.Jb…" } } }
//! ```
//!
//! - **N4** An endpoint MUST refuse a `notify` unless all of these hold. `url` is an absolute URL
//!   of at most 2048 bytes with a non-empty host, no userinfo and no fragment, and its scheme is
//!   `https`, or `http` when its host is `localhost`, an address in `127.0.0.0/8`, or `[::1]`, for
//!   tests. `token` is 1 to 512 bytes of RFC 7235 `token68` (`A-Z a-z 0-9 - . _ ~ + /`, then any
//!   `=`). Unknown members of `notify` are ignored. The `url` is opaque past these checks. (check:
//!   conformance `keep refuses an invalid notify`; fixture `notify-keep.json`)
//! - **N5** A refused `notify` is a `400` answer, and the endpoint MUST then record nothing: the
//!   job is not kept, and no notify is stored. A `list` after it reports the job exactly as before
//!   the request. (check: conformance `keep refuses an invalid notify`; fixture `notify-keep.json`)
//! - **N6** An accepted `keep` MUST record the keep and its notify together, and the latest `keep`
//!   governs: one with `notify` replaces the job's notify, and one without it removes it. An
//!   accepted `keep` MUST abandon the job's attempt in flight, if any, and every attempt that
//!   begins after a `keep` is recorded MUST use that keep's notify. (check: conformance `notify
//!   follows the latest keep`)
//! - **N7** The token is opaque. An endpoint MUST send it byte for byte as it was given, and MUST
//!   NOT parse, check or rewrite it. (check: conformance `callback request shape`; fixture
//!   `notify-callback.json`)
//! - **N8** One sender. An endpoint MUST make a job's proof (N11) and its state-change callbacks
//!   (N12) from the same component, over the same network path, so that a proof that arrives shows
//!   the job's end can arrive by the same way. An endpoint that cannot notify about a particular job
//!   (it started the job before it could) MUST refuse a `notify` for that job (N5), so the job is
//!   held as today. (check: conformance `the proof and the end reach the same url`)
//! - **N10** `keep` does not wait. The endpoint MUST answer `keep` without waiting for any attempt,
//!   and every attempt MUST be made by a process that outlives the `keep` request: never one in the
//!   process group that is killed once a command has answered. (check: conformance `keep does not
//!   wait for its callback`)
//!
//! ### When the endpoint calls back
//!
//! A job has at most one notification at a time, and it carries every change it owes.
//!
//! - **N11** Proof of reach. On every accepted `keep` that carries `notify`, the endpoint MUST owe
//!   the job a callback. The first attempt begins after the keep is recorded (a `list` made then
//!   reports the job kept) and no later than 2 s after the `keep` is answered. (check: conformance
//!   `first callback at keep`)
//! - **N12** State changes. While a job's latest `keep` carries `notify`, the endpoint MUST owe the
//!   job a callback at each moment a held stream of the job would send `shell_exit` (the shell
//!   exited and a live member of its group was found), and when the job's end becomes observable
//!   (`ended`, once its group is empty, or `lost`). An attempt for an owed callback begins after the
//!   change and no later than 2 s after it, or, when an attempt for the job is in flight then, as soon
//!   as that attempt ends. An answer settles only what was owed when its attempt began; a change
//!   owed after that begins a new count of attempts. (check: conformance `callback after the job
//!   ends`; conformance `a change during an attempt is called back`)
//! - **N13** The endpoint MUST NOT begin an attempt for a job after it has been removed (`remove`,
//!   `kill` with `remove`, `wipe`), or for a job whose latest `keep` carried no `notify`, or for a
//!   job never kept. (check: conformance `no callback without notify`)
//!
//! ### The callback request
//!
//! - **N14** Each attempt MUST be one `POST` to `url` exactly as given, with `Content-Type:
//!   application/json`, `Authorization: Bearer <token>` (one space), a `Content-Length`, no
//!   `Transfer-Encoding`, no cookies, and the body
//!   `{"type":"job_ended","scope":"<scope>","id":"<id>"}`: exactly these three members, `scope` and
//!   `id` byte for byte as the `keep` named them, at most 512 bytes, whatever the job's state.
//!   (check: conformance `callback request shape`; fixture `notify-callback.json`)
//! - **N15** The endpoint MUST NOT follow a redirect: a redirect would carry the token to an origin
//!   `keep` did not name. Over `https` it MUST verify the server's certificate and name; a
//!   connection that fails verification sends no request and counts as a connection error. (check:
//!   conformance `callback does not follow redirects`; conformance `callback verifies TLS`)
//! - **N16** An attempt that has no complete answer 30 s after it began counts as a connection
//!   error, and the endpoint MUST abandon it. The endpoint MUST NOT have more than one attempt for a
//!   job in flight at a time. (check: fixture `notify-retry.json`; conformance `callback retry and
//!   stop`)
//!
//! ### The answer, and retries
//!
//! - **N17** The endpoint MUST act on the status as this table says. (check: fixture
//!   `notify-retry.json`; conformance `callback retry and stop`)
//!
//! | answer | meaning | the endpoint |
//! |---|---|---|
//! | `2xx` | the receiver has taken the hint | settles what the attempt carried: done |
//! | `401`, `403`, `404`, `410` | the token, the session or the job is refused, and always will be | settles it: stop |
//! | `407`, `408`, `425`, `429`, `5xx`, a connection error, a TLS failure, no answer in 30 s | not now | retries, per N18 |
//! | anything else (`1xx`, `3xx`, `400`, `411`, `413`, `421`, other `4xx`) | the request itself is wrong; sending it again cannot help | settles it: stop |
//!
//! `Retry-After` is ignored.
//!
//! - **N18** A count of attempts has at most 12. After a failed attempt *n* (n = 1 … 11), the
//!   endpoint MUST wait `d(n) · (1 − u/2)` before attempt *n* + 1, where `d(n) = min(2^(n−1), 60)`
//!   seconds, `u` is drawn uniformly from `[0, 1)` afresh for every wait, and the wait starts when
//!   attempt *n* ended. After the twelfth failed attempt what it carried is given up. (check:
//!   fixture `notify-retry.json`; conformance `callback retry and stop`)
//!
//! So the waits are `1, 2, 4, 8, 16, 32, 60, 60, 60, 60, 60` s at most and half that at least, and a
//! count is over within 723 s. A storage-lease takeover after a hard kill took at most 102.5 s in
//! the fleet's measurements, and the edge retries for 120 s (FLEET.md, Requirement 3); the tenth
//! attempt begins at 121.5 s at the earliest, so a count outlasts one. One that is given up, or
//! stopped, loses nothing: the end is noticed at the next real use or open (the accepted
//! difference).
//!
//! ### The token stays secret
//!
//! - **N19** The token MUST NOT appear in the arguments or the environment of any process in the
//!   sandbox, or in any answer or stream the endpoint sends (`list`, `read`, `attach`, a job's
//!   output, an error). A model that runs `ps` there would otherwise copy it into a transcript.
//!   (check: conformance `the token never reaches argv, environment or answers`)
//! - **N20** The token MUST NOT be written to a log. A verifier cannot see an external endpoint's
//!   logs, so this repo tests it for the endpoints it ships: `execd` never prints it, and
//!   neither does `exec_mock`. (check: `security::notify_tokens_and_callback_urls_never_reach_the_log`)
//!
//! These keep the token out of transcripts by accident. They do not hide it from a program that
//! looks: a sandbox that runs the model's commands as the same user as the endpoint lets them
//! read the token. What such a program gains is small by construction: it can start a live
//! reconcile of one job of one session, at most one a second (N30), or fake its own job's proof and
//! then break the path, which delays only that job's report to the next use. It cannot open a
//! session, run a command, read anything, or reach another session or tenant (N23, N24, N44).
//!
//! ### The receiver
//!
//! A callback is a `serve` command, `job_ended`, POSTed to the session's own address,
//! `/_beyond/agent?session_id=<id>&tenant=<tenant>`. The edge (FLEET.md, E1–E6) routes it exactly
//! as any request for that session id, retries and ring walk included, and attaches, mints and
//! verifies nothing: the replica verifies the token itself.
//!
//! **A callback never opens a session.** Today a session that is closed, reaped or moved learns of
//! a job's end only when it next opens and reads its sandbox's jobs back, and notify changes nothing
//! about that. So a callback matters only to a session open in some replica's process right now. A
//! replica that does not have the session open answers `503`, as it answers any request for a
//! session it does not hold, and the edge goes on to the replica that does (FLEET.md, Requirement
//! 2). When no replica has it open, the sandbox's retries run out and nothing has happened. No
//! requirement here relies on a session id being secret: a request without a valid token for the
//! job it names is refused before the replica looks for the session, so it reaches nothing and
//! learns nothing.
//!
//! - **N21** A request's value of a header is its field value with leading and trailing spaces and
//!   tabs removed. A request is a *callback* when it has exactly one `Authorization` header and
//!   that value is `Bearer` (compared without regard to ASCII case), one or more spaces, and then
//!   `bjn_v1.`; the token is everything after the spaces. A request *looks like* a callback when
//!   any of its `Authorization` values is `bearer` (any case), any run of spaces and tabs (none
//!   included), and then `bjn_v1.`, read on the value's bytes: a value that is not UTF-8 is never a
//!   callback, and is a look-alike when its bytes match. A replica MUST classify a request this way
//!   from its head's header lines alone, before any other check or parse of them, `Content-Length`
//!   included; MUST handle a callback only as this section says, never through the front door's
//!   grant check, pinning or reply-picking, and ignoring any grant it carries; and MUST answer
//!   `400` a request that looks like a callback but is not one. The edge classifies with these
//!   same two rules (E1). (check: `serve_job_ended::the_token_header_is_read_exactly`; fixture
//!   `notify-callback.json`)
//! - **N22** `job_ended` MUST be accepted only in a callback. Sent any other way (under a session
//!   grant, over a WebSocket, on stdio) it is refused: `403` over HTTP, a failed `response`
//!   otherwise. (check: `serve_job_ended::job_ended_is_only_a_callback`; fixture
//!   `notify-callback.json`)
//! - **N23** The receiver MUST check a callback in this order, and answer at the first failure. (1)
//!   Its path is `/_beyond/agent` (`404`, as for any other path today) and it is a `POST` (`403`).
//!   (2) Its query names `session_id` and `tenant` exactly once each, keys and values decoded as
//!   `application/x-www-form-urlencoded`, a well-formed session id and tenant (`400`); other
//!   parameters are ignored. (3) The body: exactly one `Content-Length` and no `Transfer-Encoding`
//!   (`411` with none, `400` with two, with a `Transfer-Encoding`, or with one that is not a number),
//!   at most 4096 bytes (`413`), all of it within 10 s (`400`), a JSON object (`400`) whose `type`
//!   is `job_ended` (`403`), with exactly the members `type`, `scope` and `id`, `id` a job id
//!   and `scope` a string of at most 64 bytes (`400`). (4) The replica is in service mode (`403`).
//!   (5) The token (N43–N47): `401` when it is `malformed` or `bad_mac`, `503` when its kid is
//!   unknown (N47). (6) `scope` is the session's own, [`jobs::JobScope::for_session`] of the URL's
//!   tenant and session (`401`: a replica mints only for that scope, N46, so a token for any other
//!   is as forged as a bad tag). Only a callback that passes all of them may look for the session,
//!   or cause anything to be sent to the sandbox. (check:
//!   `serve_job_ended::a_forged_callback_reaches_nothing`; fixture `notify-callback.json`)
//! - **N24** A callback MUST NOT open, create, start or load a session, and MUST NOT write to
//!   storage. When no session task in this process holds the URL's session for the URL's tenant, a
//!   callback that passes N23 MUST be answered `503`, and nothing else happens. (check:
//!   `serve_job_ended::a_callback_never_opens_a_session`; fixture `notify-callback.json`)
//! - **N25** Every refusal of a callback, those of N21 and N23 included, MUST have a fixed body for
//!   its status that says nothing about which check failed beyond the status. (check:
//!   `serve_job_ended::refusals_say_nothing`)
//! - **N26** A replica MUST serve `job_ended` in any state of the session (a run in flight, a
//!   compaction, a retry's backoff, a branch summary), and MUST answer it only with a status this
//!   section names: never with the `200` and failed `response` with which a busy session answers
//!   other commands. A callback *arrives* when the replica has read its request head. (check:
//!   `serve_job_ended::served_during_a_run`)
//! - **N27** A replica MUST answer each callback through a reply path that belongs to that request
//!   alone, never by picking an answer out of the session's output, and MUST NOT send it to the
//!   session's clients. A callback MUST NOT count as an attached client, or restart or extend the
//!   idle clock of a detached session. A callback whose connection closes before it is answered MUST
//!   stop waiting. A session MUST hold at most 16 callbacks waiting for an answer, and answer any
//!   more `503`. (check: `serve_job_ended::each_callback_gets_its_own_answer`;
//!   `serve_job_ended::the_callback_is_not_a_client`)
//!
//! **Reconciles.** A reconcile reads one session's jobs back from its sandbox. There are two, and
//! they differ in what they may change.
//!
//! The *open reconcile* is today's read-back when a transcript opens (`tools::bash_sandbox`'s
//! reattach): it runs when the session opens with its sandbox attached, or else at its first use of
//! it, lists the scope, adopts each kept job with a number, reads what ended, and kills what nobody
//! kept. A failed one leaves the session's jobs untracked, as today. Notify adds only N42 to it. It
//! never runs because of a callback, so a callback never wakes or changes a session that has not
//! used its sandbox since it opened.
//!
//! - **N28** The *live reconcile* runs only in an open session whose open reconcile has finished.
//!   In the sandbox it MUST `list` the scope and send nothing else. In the session it MUST act only
//!   on jobs that are unheld, by the release table's `unheld` rows, and MUST leave every other job
//!   exactly as it was. When it hands its findings over it MUST look each job up again by its
//!   sandbox job id, and act only if the job is still unheld and the session is still on the
//!   transcript it started on (`Persistence::transcript`, which `serve` compares to notice a
//!   transcript switch). A callback changes nothing by itself: only what a reconcile lists changes a
//!   job's state. (check: `job_notify::the_live_reconcile_changes_only_unheld_jobs`)
//! - **N29** A live reconcile MUST start only for these: a callback for an unheld job; a release
//!   (the release table); and a real use of the sandbox while the session has an unheld job. A
//!   *real use* is a command or filesystem call the session sends its sandbox through its exec
//!   target: a tool's command, a filesystem operation, discovery. A sandbox job operation (this
//!   section's `list`, a `bash_job` call, a kill) is not one; a `bash_job` call about an unheld job
//!   has its own row in the release table. A use never delays itself, and is never dropped, because
//!   the reconcile that last listed may have listed before the end the use would find: with none
//!   running, a use starts one at once when no reconcile ended in the last 30 s, and otherwise asks
//!   for one that begins 30 s after the last one ended; a use that arrives while one runs asks for
//!   the next, as a callback does (N30). A replica MUST
//!   NOT contact a sandbox about its unheld jobs on any timer of its own: every reconcile, however
//!   long it waits for its spacing, is one of these asked for. (check:
//!   `job_notify::reconciles_only_on_a_callback_or_a_use`)
//! - **N30** A session runs at most one reconcile at a time. A callback or a real use that arrives
//!   while one runs waits for the next, and every callback and use that arrives during the same
//!   reconcile shares that next one. The live reconcile a release starts MUST begin after that
//!   job's stream is closed (any frame already read taken in first), and never shares one that
//!   began earlier. A reconcile that a callback, a release or a use during a reconcile started MUST
//!   NOT begin less than 1 s after the session's previous reconcile ended, so a program in the
//!   sandbox that sends a real token in a loop costs at most one reconcile a second. (check:
//!   `serve_job_ended::one_reconcile_at_a_time`)
//! - **N31** A callback for an unheld job MUST be answered `200` (with a `job_ended` `response`)
//!   once a live reconcile that began after it arrived has listed the scope and handed its findings
//!   to the session; reading the output is left to today's follow and is not waited for. It MUST be
//!   answered `503` when that reconcile could not list the sandbox, or when no answer is ready 20 s
//!   after it arrived. (check: `serve_job_ended::answers_after_the_list`; fixture
//!   `notify-callback.json`)
//! - **N32** Any other callback for an open session MUST be answered without contacting the sandbox:
//!   `200` when the session's open reconcile has not succeeded yet, when the job is proving or held,
//!   or when its end is recorded; `503` when the job is taken, so the sandbox asks again once the
//!   call has recorded the end or put the job back; `410` when the session neither tracks the job nor
//!   has recorded its end. (check: `serve_job_ended::repeats_are_free`; fixture
//!   `notify-callback.json`)
//!
//! **What the session does with what it learns.**
//!
//! - **N33** A job's end, or its leftovers, MUST reach the session exactly as from a held stream
//!   today. An end is always read by today's follow, which an `unheld` row starts from the offset the
//!   job's stream reached, so the notice is the one today's follow produces; output the sandbox
//!   dropped past its cap is a gap there, as in today's reattach. An end a listing found is recorded
//!   at the listing's `ended` time (`max(ended, started)` for `lost`), as today's reattach records it
//!   (`JobShared::finish_at`), and with the listed status when the follow then fails to read the
//!   rest. A listed `exited` is applied as a held stream's `shell_exit` is (once per job), so whether
//!   it produces a notice is decided by today's per-job record of what was already reported: a
//!   background or adopted job's leftovers are never reported, except for a background job that a
//!   `bash_job` call put back while its shell still ran (today's `BashJobs::park`), whose leftovers
//!   are. A job whose shell exit and end both fall between two listings shows only its end, as it can
//!   to today's polling follow. (check:
//!   `notify_e2e_release::a_released_job_is_silent_until_its_end`)
//! - **N34** Notify MUST NOT add a window in which an end or leftovers are reported twice or not
//!   at all, beyond those today's code has. Within a session task each is reported at most once
//!   across every source (a held stream, a follow, a live reconcile, an open reconcile and a
//!   `bash_job` call), because every one of them reaches the session through the job's one shared
//!   state and today's per-job record of what was reported. Across a replica's death, the windows
//!   are exactly today's: the end of a job whose record the dead replica had not yet removed is
//!   reported again by the next owner's open reconcile, as it is today for a held stream (see
//!   ARCHITECTURE.md, "Known gap: a crash can report a sandbox job's end twice"). (check:
//!   `serve_job_ended::one_report_per_end`; fleet-sim `sandbox-job-exit-noticed-once`)
//!
//! A replica that dies while a job is `proving` or `unheld` leaves the job running, kept; its
//! callbacks reach no session, and the next owner adopts it at its open reconcile and keeps it
//! again (N42), or reads its end back, exactly as today.
//!
//! ### Release (normative)
//!
//! Per kept job, in the session task that owns the job:
//!
//! | state | the replica holds | event | action | next |
//! |---|---|---|---|---|
//! | (parking) | the start's stream | notify is available (N36) | open the proof window, then send `keep` with `notify` | proving |
//! | (parking) | the start's stream | notify is not available | send `keep` without `notify`, as today | held |
//! | proving | the stream | `keep` is answered `400` | send `keep` again without `notify` | held |
//! | proving | the stream | `keep` fails any other way | nothing more: as today's failed keep | held |
//! | proving | the stream | a verified callback for the job has arrived inside the window, and `keep` has succeeded (in either order) | close the stream; then start a live reconcile | unheld |
//! | proving | the stream | the window closes with no such callback | nothing | held |
//! | proving, held | the stream or a follow | its final frame, or its break | as today: report the end, or follow | as today |
//! | unheld | nothing | a live reconcile lists it `running` | nothing | unheld |
//! | unheld | nothing | a live reconcile lists it `exited` | apply it as a stream's `shell_exit` (N33) | unheld |
//! | unheld | nothing | a live reconcile lists it `ended` or `lost`, does not list it, or cannot list the sandbox | today's follow, from the offset its stream reached | held |
//! | unheld | nothing | a `bash_job` call about it | read in what the sandbox holds from that offset, as the open reconcile's catch-up does (`read_in`), then today's follow for the rest of the call; a call that returns with the job running closes the follow | unheld, or held once ended |
//!
//! A live reconcile that cannot list the sandbox follows every unheld job, and a `bash_job` call
//! follows its own: whichever way a dead sandbox is found, its jobs take today's error path.
//!
//! - **N35** The replica MUST release a job's stream only by the release row: `keep` with `notify`
//!   succeeded, and a verified callback for that job arrived inside its proof window. Nothing else
//!   releases a stream: not a `keep` alone, not a callback after the window.
//!   (check: `job_notify::release_waits_for_the_first_callback`)
//! - **N36** The replica MUST treat notify as available only when all of these hold: it is in
//!   service mode and the operator has configured the origin the fleet's edge serves sessions at.
//!   The `url` the replica sends is
//!   that origin's `/_beyond/agent?session_id=<session id>&tenant=<tenant>`, and the `token` is
//!   minted for this session and job (N46). (check: `job_notify::keep_without_notify_when_unavailable`)
//! - **N37** The proof window MUST open before `keep` is sent, so a callback that overtakes `keep`'s
//!   answer still counts, and MUST close 30 s after `keep` is answered. With instant failures, the
//!   first five attempts of the proof all begin within 17 s of that answer (2 s, N11, then 15 s,
//!   N18). (check: `job_notify::proof_window_opens_before_keep`)
//! - **N38** When the window closes with no proof, or `keep` with `notify` fails, the job MUST stay
//!   held for good, exactly as today. (check: `job_notify::no_callback_keeps_the_stream`;
//!   `job_notify::a_refused_notify_falls_back`)
//! - **N39** Until an unheld job's end is recorded, the replica MUST send nothing concerning it
//!   except: the `list` of a live reconcile (N29) or of an open one; today's follow and the read-in
//!   an `unheld` row starts; what a tool call or client command asks for (a `bash_job` call, a kill);
//!   the `seen` that records output the model was just shown; and the `kill` or `wipe` of a
//!   transcript switch or of `delete_session`. After its end, today's removal of its record. (check:
//!   `job_notify::silent_after_release`; `job_notify::a_call_lets_go_again`)
//! - **N40** A replica with notify available MUST close a connection to the endpoint that has
//!   carried nothing for 30 s, and MUST NOT send TCP keepalives on one that is idle: the exec
//!   runner's HTTP client keeps an idle pooled connection at most 30 s and has TCP keepalive off.
//!   (check: `job_notify::no_idle_connection_to_the_endpoint`)
//!
//! ### Crash and takeover
//!
//! The release state lives only in the session task. A replica that dies loses it, and so does a
//! session that is reaped: the jobs run on in the sandbox, kept, for the next open reconcile to read
//! back, as today.
//!
//! - **N41** A token MUST be a function of the key and of (tenant, session, scope, job id) alone,
//!   with no time, nonce or replica in it (N44), so every replica that holds the key mints the same
//!   token for the same job, and a session that moves has nothing to hand over. (check: fixture
//!   `notify-token.json`)
//! - **N42** A replica MUST keep a job again only when its open reconcile adopts it, kept and not
//!   ended, while notify is available: it sends `keep` with the `num`, `kind` and `seen` its listing
//!   reported and `notify`, and treats the job as `proving`, its follow standing for the stream in
//!   the release table. It MUST NOT keep in a live reconcile, so a reconcile never leads to another
//!   keep. (check: `job_notify::takeover_proves_again`; `job_notify::a_live_reconcile_never_keeps`)
//!
//! **Takeover.** A session that moves to another replica (a deploy, a kill, a failover, a ring
//! change) is opened there as any session is: its open reconcile adopts each kept job that still
//! runs and keeps it again (N42), so the job is held on its new owner until a callback proves the
//! path again. A callback that reaches a replica without the session is answered `503`, and the
//! edge's ring walk takes it on to the replica that has the session, wherever the ring now places
//! its hash; one that finds no replica with the session is retried until its count runs out. An end
//! that happened while no one held the session is read back by that open reconcile, exactly as
//! today.
//!
//! ### The notify token (`bjn_v1`)
//!
//! The fleet mints the token and the fleet verifies it. Only replicas hold the key, so the token is
//! opaque to the sandbox and to the edge.
//!
//! ```text
//! token     "bjn_v1." kid "." mac
//! kid       8 lowercase hex digits = hex(SHA-256("bjn_v1 kid" ‖ K)[0..4])
//! mac       base64url without padding of the 32-byte tag (43 characters), canonical
//! K         HKDF-SHA256(ikm = S, salt = empty, info = "bjn_v1 notify"), 32 bytes
//! S         the fleet seal secret: the 32 bytes the seal key decodes to (`--seal-key`,
//!           AI_AGENT_SEAL_KEY_B64), exactly as given, before X25519 clamps them
//! tag       HMAC-SHA256(K, m)
//! m         lp("bjn_v1") ‖ lp(kid) ‖ lp(purpose) ‖ lp(tenant) ‖ lp(session id) ‖ lp(scope)
//!           ‖ lp(job id)
//! lp(x)     the length of x in bytes as a 4-byte big-endian unsigned integer, then x's UTF-8 bytes
//! purpose   "job_ended"
//! ```
//!
//! - **N43** A token that is not exactly that grammar MUST be refused as malformed: another prefix
//!   or version, a field count other than three, a kid that is not eight lowercase hex digits, a
//!   `mac` that is not 43 characters of the base64url alphabet, or one whose final character
//!   carries non-zero padding bits (it would not re-encode to itself). (check: fixture
//!   `notify-token.json`)
//! - **N44** The verifier MUST compute `m` from the URL's tenant and session id, the body's `scope`
//!   and `id`, and the constant purpose, and MUST take none of them from the token. So a token
//!   verifies only for the tenant, session, scope, job and purpose it was minted for. (check:
//!   fixture `notify-token.json`)
//! - **N45** The tag MUST be compared in constant time (`hmac::Mac::verify_slice`), and in full: no
//!   truncated tags. (check: `notify_token::the_tag_is_compared_in_constant_time`; fixture
//!   `notify-token.json`)
//! - **N46** A replica MUST mint and verify under exactly one key, the one derived from its seal
//!   secret, and MUST mint only for the session's own scope ([`jobs::JobScope::for_session`] of its
//!   tenant and session) and for a job of that session. (check: fixture `notify-token.json`;
//!   `notify_token::mints_only_its_own_scope`)
//! - **N47** The verdicts are `valid`, `malformed`, `unknown_kid` and `bad_mac`. A replica MUST
//!   answer `malformed` and `bad_mac` with `401`, and `unknown_kid` with `503`. (check: fixture
//!   `notify-token.json`; fixture `notify-callback.json`)
//!
//! **Why this key.** The only secret that every replica of a fleet already holds and that no one
//! else holds is the fleet's X25519 seal secret. A replica holds only the public halves of the
//! grant signing keys, which cannot key a MAC, and a tenant's DEK is per tenant rather than one key
//! across replicas. HKDF under its own `info` label makes `K` independent of every other use of
//! `S`: the grant's key agreement uses `S` only as an X25519 scalar, and its own HKDF runs over the
//! shared secret with the `info` label `bsg_v1 seal`. `S` is never used as a MAC key directly. The
//! kid is a fingerprint of `K`, because the seal key has no kid of its own, and reveals nothing
//! about `K`.
//!
//! **Rotation.** The seal secret has no key set: a replica holds one, and rotating it is the grant
//! seal's own procedure. A token under the old secret then names a kid the replica does not hold,
//! so it is answered `503` (never `401`, which would stop the notification while a rollout is half
//! done), and once its retries run out its end is noticed at the next use or open: later, never
//! lost or wrong.
//!
//! **Why no expiry.** The jobs this exists for are the long ones: a dev server runs for days, and a
//! token that expired first would lose exactly those reports. The token's power is already bounded:
//! it can start a reconcile of one named job of one open session, at most one a second, and that is
//! all. Because it does not expire, whatever holds a copy can keep a session's sandbox from idling
//! for as long as the session is open, so no party but the sandbox and the replica may record it
//! (N20, and FLEET.md's rule for the edge).
//!
//! **What binds tenant and session.** The callback carries no grant: the URL names the tenant and
//! the session, and the token's tag covers both (N44), so a token minted for one session fails on
//! every other, in every tenant. A token that verifies reaches only a session task holding that
//! session for that tenant (N24).
//!
//! ### What each invariant rests on
//!
//! | invariant | requirements |
//! |---|---|
//! | I1: notify adds no window of duplicate or lost reports beyond today's: within a session task every job end is reported at most once; and at least once: while the session is open, by its stream, a callback, or at its next real use of the sandbox or `bash_job` call about the job; otherwise when it next opens, as today | N12, N18, N24, N28, N29, N30, N31, N32, N33, N34 |
//! | I2: an end or leftovers learned by notify behave exactly as when learned by the held stream | N28, N33, N34 |
//! | I3: a callback is a hint: a lost, duplicated or forged one never produces wrong state, never opens a session, and costs at most one reconcile a second | N14, N21, N23, N24, N28, N30, N32 |
//! | I4: a stream is let go only after a callback for that job arrived over the end's path, after a successful `keep` with `notify` | N8, N9, N11, N35, N37, N38 |
//! | I5: once a job is unheld, nothing crosses the replica-to-endpoint connection for it but what N39 lists, nothing is sent on a timer, and no idle connection stays open | N29, N39, N40 |
//! | I6: the token is narrow: a callback can only start a live reconcile of the one job it names, in a session open in that process | N21, N22, N23, N24, N43, N44, N45, N46, N47 |
//! | I7: never worse than today, with one exception: once a job is unheld, an end no callback delivers (the sandbox died or stopped, every attempt failed, or a superseded owner that has not yet noticed took it) is noticed at the next real use, `bash_job` call or open instead of within seconds, and then takes today's path | N1, N2, N3, N5, N17, N24, N29, N33, N34, N36, N38 |

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::exec::{ChunkSink, CommandRunner, ExecResult, JobHooks, Kill, SpawnSink};

pub mod jobs;
pub mod jobs_native;
pub mod stream;

use stream::{FRAME_SILENCE_LIMIT, Frame, FrameReader, NDJSON};

/// What the agent sends.
#[derive(Debug, Serialize)]
struct ExecRequest<'a> {
    command: &'a str,
    args: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<&'a str>,
    timeout_ms: u64,
    /// one-shot: bytes for the command's stdin, standard base64. Absent means no stdin.
    #[serde(skip_serializing_if = "Option::is_none")]
    stdin: Option<String>,
    /// streaming: answer with a stream of frames instead of one object. Absent means one-shot.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    /// jobs: a job operation (see "Sandbox jobs"). Absent means the command is run as ever.
    #[serde(skip_serializing_if = "Option::is_none")]
    job: Option<&'a serde_json::Value>,
}

/// What the agent expects back. Extra fields are ignored, so an endpoint may return more.
#[derive(Debug, Deserialize)]
#[doc(hidden)]
pub struct ExecResponse {
    pub exit_code: i32,
    /// Standard base64 of what the command wrote: the empty string for no output.
    pub stdout: String,
    pub stderr: String,
}

impl ExecResponse {
    /// The answer as a result, the base64 decoded. An answer whose `stdout` or `stderr` is not
    /// standard base64 is an error, never read as empty.
    pub fn into_result(self, url: &str, cap: usize) -> std::io::Result<ExecResult> {
        let decode = |field: &str, b: &str| {
            base64::engine::general_purpose::STANDARD
                .decode(b)
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "exec endpoint {url}: `{field}` is not standard base64 ({e})"
                    ))
                })
        };
        let (stdout, stderr) = (
            decode("stdout", &self.stdout)?,
            decode("stderr", &self.stderr)?,
        );
        if stdout.len().saturating_add(stderr.len()) > cap {
            return Err(too_large(url, cap));
        }
        Ok(ExecResult {
            code: Some(self.exit_code),
            stdout,
            stderr,
            ..Default::default()
        })
    }
}

/// Room for the JSON around two base64 strings (the keys, the exit code).
const ENVELOPE_SLACK: usize = 4096;

/// The most body a client reads for an answer under response cap `cap`: the base64 of `cap` decoded
/// bytes, plus the envelope. The cap itself is on the decoded bytes.
fn body_cap(cap: usize) -> usize {
    cap.div_ceil(3)
        .saturating_mul(4)
        .saturating_add(ENVELOPE_SLACK)
}

/// The error for an answer past the response cap: the command ran. `FileTooLarge`, so a probe reads
/// it as an answer (not a transport failure) and a caller handles one kind.
fn too_large(url: &str, cap: usize) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::FileTooLarge,
        format!(
            "exec endpoint {url}: the response exceeded the {cap}-byte cap and was discarded — the \
             command ran, but its output is too large to return; narrow it (e.g. pipe through \
             `head`) rather than re-running it unchanged"
        ),
    )
}

/// The `timeout_ms` a streamed `bash` job sends: `i32::MAX` ms (~24.8 days), the longest deadline
/// every endpoint can honor — Node's `setTimeout` fires at once for anything longer. A job's lifetime
/// is the connection's (see "Disconnect kills"), so this only has to outlast any session; it is not a
/// limit anyone should meet.
pub const STREAM_DEADLINE: Duration = Duration::from_millis(i32::MAX as u64);

/// What a remote command left running, as `on_orphans` describes it: the runner cannot list the far
/// side's processes, only that its process group still has some.
pub(crate) const REMOTE_LEFTOVERS: &str = "processes on the exec target";

/// Where a runner reports a shell that exited leaving processes running — [`JobHooks::on_orphans`].
type OrphanSink<'a> = &'a (dyn Fn(&ExecResult, String) + Send + Sync);

/// The default response-body cap for [`HttpExecRunner`]: 16 MiB.
///
/// Far above anything a tool consumes — `bash` keeps ~30 KB of output and the local runner keeps
/// 256 KiB per stream — so a legitimate response never meets it. What it bounds is the response that
/// isn't legitimate: a `cat` of a multi-gigabyte file must not be buffered whole into a process that
/// may be serving other sessions.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// A [`CommandRunner`] that POSTs each command to a URL.
///
/// Holds a `reqwest::Client` so connections are pooled across calls, but no session and no state — a
/// call is a request. There is nothing to reconnect, nothing to keep alive, and no lifecycle to get
/// wrong.
pub struct HttpExecRunner {
    url: String,
    client: reqwest::Client,
    headers: Vec<(String, String)>,
    /// Where the `Authorization: Bearer` token comes from, when the endpoint wants one.
    auth: Option<crate::token::Bearer>,
    max_response_bytes: usize,
    /// Raised when the endpoint could not be reached or refused a request: a sign the sandbox
    /// behind it may have been replaced, so whatever was learned about it (the probe, the
    /// capabilities above) should be learned again. The agent's lazy attach reads this flag.
    suspect: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl HttpExecRunner {
    /// Point the runner at `url`.
    pub fn new(url: impl Into<String>) -> Result<Self, String> {
        let url = url.into();
        // Fail at construction rather than on the first tool call, so a typo is an immediate startup
        // error instead of a confusing mid-turn failure the model tries to work around.
        let parsed =
            reqwest::Url::parse(&url).map_err(|e| format!("invalid exec URL {url:?}: {e}"))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(format!(
                "exec URL must be http or https, got {:?}",
                parsed.scheme()
            ));
        }
        Ok(Self {
            url,
            client: {
                crate::tls::ensure_provider();
                // An idle connection to the endpoint is closed after 30 s and never probed with TCP
                // keepalives: once a job is unheld (exec notify), nothing may keep the sandbox awake.
                reqwest::Client::builder()
                    .pool_idle_timeout(Duration::from_secs(30))
                    .tcp_keepalive(None)
                    .build()
                    .unwrap_or_else(|_| reqwest::Client::new())
            },
            headers: Vec::new(),
            auth: None,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            suspect: None,
        })
    }

    /// Raise `flag` whenever a request fails to reach the endpoint or is refused by it (a non-2xx
    /// answer), so a caller watching for a replaced sandbox can re-probe.
    pub fn with_suspect_flag(mut self, flag: Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.suspect = Some(flag);
        self
    }

    fn suspect(&self) {
        if let Some(flag) = &self.suspect {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Cap how much of a response body is read before the call fails. See the module doc for why an
    /// oversized response is an error rather than a truncated result.
    pub fn with_max_response_bytes(mut self, max: usize) -> Self {
        self.max_response_bytes = max;
        self
    }

    /// Add a header sent with every request — an `Authorization`, an API key, a tenant id. Every real
    /// endpoint needs some form of auth, and which form is the endpoint's business, not the agent's.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Send `Authorization: Bearer <token>` with every request, the token from `source`. On a `401`
    /// the source is asked once for a new token and that request is repeated once: the only retry.
    pub fn bearer(mut self, source: impl Into<crate::token::TokenSource>) -> Self {
        self.auth = Some(crate::token::Bearer::new(source.into()));
        self
    }

    /// Parse a `Name: value` pair, the shape a CLI flag or env var carries.
    pub fn parse_header(raw: &str) -> Result<(String, String), String> {
        let (name, value) = raw
            .split_once(':')
            .ok_or_else(|| format!("header must be `Name: value`, got {raw:?}"))?;
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("header name is empty in {raw:?}"));
        }
        Ok((name.to_string(), value.trim().to_string()))
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Send one request and return the response once its status is known to be a success. `stream`
    /// asks for streaming frames; whether they came is the response's `Content-Type` to say.
    #[doc(hidden)]
    pub async fn send(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
        stream: bool,
    ) -> std::io::Result<reqwest::Response> {
        self.send_job(program, args, cwd, timeout, stdin, stream, None)
            .await
    }

    /// [`Self::send`], carrying a job operation.
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub async fn send_job(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
        stream: bool,
        job: Option<&serde_json::Value>,
    ) -> std::io::Result<reqwest::Response> {
        let body = ExecRequest {
            command: program,
            args,
            cwd,
            timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
            stdin: stdin.map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
            stream,
            job,
        };
        self.send_request(timeout, &body).await
    }

    /// Send one request body and return the response once its status is known to be a success.
    async fn send_request(
        &self,
        timeout: Duration,
        body: &ExecRequest<'_>,
    ) -> std::io::Result<reqwest::Response> {
        let mut token = match &self.auth {
            Some(auth) => Some(auth.token().await?),
            None => None,
        };
        let mut renewed = false;
        let resp = loop {
            let mut req = self.client.post(&self.url).json(body);
            for (name, value) in &self.headers {
                req = req.header(name.as_str(), value.as_str());
            }
            if let Some(t) = &token {
                req = req.bearer_auth(t);
            }
            // A little past the command's own budget, so the endpoint gets the chance to enforce
            // and report the timeout itself — its message is more useful than a bare client-side
            // abort. It bounds the whole exchange, a stream's body included.
            let resp = req
                .timeout(timeout.saturating_add(Duration::from_secs(30)))
                .send()
                .await
                .map_err(|e| {
                    if !e.is_timeout() {
                        self.suspect();
                    }
                    if e.is_timeout() {
                        std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!("exec endpoint {} timed out: {e}", self.url),
                        )
                    } else {
                        std::io::Error::other(format!(
                            "exec endpoint {} unreachable: {e}",
                            self.url
                        ))
                    }
                })?;
            if resp.status() == reqwest::StatusCode::UNAUTHORIZED
                && !renewed
                && let (Some(auth), Some(stale)) = (&self.auth, &token)
            {
                renewed = true;
                if let Some(fresh) = auth.renew(stale).await? {
                    token = Some(fresh);
                    continue;
                }
            }
            break resp;
        };
        let status = resp.status();
        if !status.is_success() {
            // The endpoint refused the *request* — a bad URL, auth, a dead sandbox. Distinct from the
            // command running and exiting non-zero, which is an ordinary result. A `400`/`422` is
            // the exception: the endpoint answered, and it is this request it declines as written
            // (a strict front refusing the `stream`, `job` or `notify` extension it does not
            // know). That says nothing of the sandbox, and treating it as if it did would rebuild
            // the target — every probe again — on the very next call, forever.
            if !matches!(
                status,
                reqwest::StatusCode::BAD_REQUEST | reqwest::StatusCode::UNPROCESSABLE_ENTITY
            ) {
                self.suspect();
            }
            let (body, _) = self.read_capped(resp).await?;
            // A `400` refuses the request as written (notify's refused `notify` among others), which
            // a caller may answer by sending another; anything else is an error of the endpoint's.
            let kind = if status == reqwest::StatusCode::BAD_REQUEST {
                std::io::ErrorKind::InvalidInput
            } else {
                std::io::ErrorKind::Other
            };
            return Err(std::io::Error::new(
                kind,
                format!(
                    "exec endpoint {} returned {}: {}",
                    self.url,
                    status.as_u16(),
                    excerpt(&body)
                ),
            ));
        }
        Ok(resp)
    }

    /// One request, one response: the whole one-shot protocol.
    #[doc(hidden)]
    pub async fn post(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecResponse> {
        let resp = self.send(program, args, cwd, timeout, stdin, false).await?;
        self.envelope(resp).await
    }

    /// Read and parse a one-shot response body: one JSON object, capped.
    #[doc(hidden)]
    pub async fn envelope(&self, resp: reqwest::Response) -> std::io::Result<ExecResponse> {
        let (body, over_cap) = self
            .read_capped_to(resp, body_cap(self.max_response_bytes))
            .await?;
        if over_cap {
            return Err(too_large(&self.url, self.max_response_bytes));
        }
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::other(format!(
                "exec endpoint {} returned unparseable JSON ({e}): {}",
                self.url,
                excerpt(&body)
            ))
        })
    }

    /// A one-shot response, read and decoded.
    async fn answer(&self, resp: reqwest::Response) -> std::io::Result<ExecResult> {
        self.envelope(resp)
            .await?
            .into_result(&self.url, self.max_response_bytes)
    }

    /// Read the body incrementally, stopping at the cap instead of buffering whatever the endpoint
    /// sends. `true` means the cap was hit and the bytes returned are only a prefix.
    #[doc(hidden)]
    pub async fn read_capped(&self, resp: reqwest::Response) -> std::io::Result<(Vec<u8>, bool)> {
        self.read_capped_to(resp, self.max_response_bytes).await
    }

    /// [`Self::read_capped`] with the cap given.
    async fn read_capped_to(
        &self,
        mut resp: reqwest::Response,
        cap: usize,
    ) -> std::io::Result<(Vec<u8>, bool)> {
        // A declared length over the cap fails before a byte of it is read.
        let declared = resp.content_length().unwrap_or(0);
        if declared > cap as u64 {
            return Ok((Vec::new(), true));
        }
        let mut body = Vec::with_capacity(declared as usize);
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| std::io::Error::other(format!("exec endpoint {}: {e}", self.url)))?
        {
            if chunk.len() > cap - body.len() {
                // Dropping `resp` abandons the rest of the body along with its connection.
                return Ok((body, true));
            }
            body.extend_from_slice(&chunk);
        }
        Ok((body, false))
    }
}

#[async_trait]
impl CommandRunner for HttpExecRunner {
    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        // An endpoint that predates `stdin` ignores it and runs the command with no input, so an
        // `Ok` here is not proof of delivery — see the module doc on the probe.
        let resp = self
            .send(program, args, cwd, timeout, Some(stdin), false)
            .await?;
        self.answer(resp).await
    }

    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let resp = self.send(program, args, cwd, timeout, None, false).await?;
        self.answer(resp).await
    }

    /// Streams: the sink gets every byte of both streams, as the command produces them.
    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.run_stream(program, args, cwd, timeout, on_chunk, None, None)
            .await
    }

    /// As [`Self::run_streaming`], reporting a [`Kill`] once the stream is open — which is also the
    /// promise that the output is live. Killing it closes the stream; the endpoint MUST then kill the
    /// command's process group (see the module doc).
    ///
    /// The job has no deadline of its own, but the wire needs a `timeout_ms`: a stream sends
    /// [`STREAM_DEADLINE`], so the connection (closed by a kill, a dropped job or the agent's exit) is
    /// what ends it. See the module doc's "Timeouts".
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        let deadline = STREAM_DEADLINE;
        // The endpoint owns the command's process group: it says when the shell exits leaving some
        // of it running (`shell_exit`), and keeps the stream — and so the job — open until the group
        // is empty. See the module doc's "Leftover processes".
        self.run_stream(
            program,
            args,
            cwd,
            deadline,
            hooks.on_chunk,
            Some(hooks.on_spawn),
            Some(hooks.on_orphans),
        )
        .await
    }

    /// Stdout frames to `on_stdout`, stderr frames to `on_stderr`, until the final frame.
    /// Dropping the future drops the response, which closes the connection — and the endpoint kills
    /// the command's group ("Disconnect kills").
    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: ChunkSink<'_>,
        on_stderr: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        let resp = self
            .send(program, args, cwd, STREAM_DEADLINE, None, true)
            .await?;
        if !is_ndjson(&resp) {
            return Err(self.not_a_stream(resp).await);
        }
        let mut frames = FrameReader::new(resp);
        let broken = |e: String| std::io::Error::other(format!("exec endpoint {}: {e}", self.url));
        loop {
            match frames.next(FRAME_SILENCE_LIMIT).await.map_err(broken)? {
                Some(Frame::Stdout(b)) => on_stdout(&b),
                Some(Frame::Stderr(b)) => on_stderr(&b),
                Some(Frame::Heartbeat | Frame::ShellExit(_)) => {}
                Some(Frame::Exit(code)) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        code: Some(code),
                        ..Default::default()
                    });
                }
                Some(Frame::Signal(signal)) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        signal: Some(signal),
                        ..Default::default()
                    });
                }
                Some(Frame::TimedOut) => {
                    return Ok(ExecResult {
                        timed_out: true,
                        ..Default::default()
                    });
                }
                Some(Frame::Error(e)) => {
                    return Err(broken(format!("the endpoint failed mid-command: {e}")));
                }
                None => {
                    return Err(broken(
                        "the stream ended before the command's final frame".to_string(),
                    ));
                }
            }
        }
    }

    /// The endpoint's own jobs (jobs).
    fn sandbox_jobs(self: Arc<Self>) -> Option<Arc<dyn jobs::SandboxJobs>> {
        Some(Arc::new(jobs_native::HttpJobs::new(self)))
    }
}

/// The start of a body, for an error message.
fn excerpt(body: &[u8]) -> String {
    String::from_utf8_lossy(body)
        .trim()
        .chars()
        .take(400)
        .collect()
}

/// How long a [`Kill`] of a stream waits for the runner to confirm it has let go of the connection.
/// The runner's future runs on the async runtime, which a process on its way out may no longer poll;
/// past this the exit itself closes the socket, which the endpoint sees the same way.
const STOP_WAIT: Duration = Duration::from_secs(2);

/// The shared state between a stream's runner future and the [`Kill`] handed out for it.
#[derive(Default)]
struct StreamStop {
    /// Asked to stop: the runner drops the response, which closes the connection.
    requested: tokio::sync::Notify,
    /// Set once the runner future has ended (by any path, a drop included) and the response is gone.
    released: std::sync::Mutex<bool>,
    released_cv: std::sync::Condvar,
}

impl StreamStop {
    /// Blocking: ask the runner to drop the stream and wait (bounded) until it has.
    fn stop(&self) {
        self.requested.notify_one();
        let released = self.released.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self
            .released_cv
            .wait_timeout_while(released, STOP_WAIT, |r| !*r);
    }
}

/// Marks the stream released when the runner future ends — dropped by an abort, or returning. Must
/// be declared before the response in the future's scope, so it drops *after* the response does.
struct Released(Arc<StreamStop>);

impl Drop for Released {
    fn drop(&mut self) {
        *self.0.released.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.0.released_cv.notify_all();
    }
}

impl HttpExecRunner {
    /// The error for a stream request answered `200` without `application/x-ndjson` (`protocol`).
    /// The body is read as a one-shot answer first, so one that is over the cap or not an answer
    /// at all is reported as that; a clean result is the silent data loss this refuses.
    async fn not_a_stream(&self, resp: reqwest::Response) -> std::io::Error {
        if let Err(e) = self.answer(resp).await {
            return e;
        }
        std::io::Error::other(format!(
            "exec endpoint {} answered a stream request with something that is not a stream",
            self.url
        ))
    }

    /// The streaming exchange. An endpoint that answers a stream request with anything but
    /// `application/x-ndjson` is not speaking the protocol: [`Self::not_a_stream`], never a guess at
    /// what its body means.
    #[allow(clippy::too_many_arguments)]
    async fn run_stream(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
        on_spawn: Option<SpawnSink<'_>>,
        on_orphans: Option<OrphanSink<'_>>,
    ) -> std::io::Result<ExecResult> {
        let stop = Arc::new(StreamStop::default());
        // Declared before the response: dropped after it, so "released" is only ever set once the
        // connection has been let go.
        let _released = Released(stop.clone());
        let resp = self.send(program, args, cwd, timeout, None, true).await?;
        if !is_ndjson(&resp) {
            return Err(self.not_a_stream(resp).await);
        }
        if let Some(report) = on_spawn {
            let stop = stop.clone();
            report(Kill::Hook(Arc::new(move || {
                stop.stop();
                None
            })));
        }
        let mut frames = FrameReader::new(resp);
        let broken = |e: String| {
            std::io::Error::other(format!(
                "exec endpoint {}: {e} (the command may have been killed)",
                self.url
            ))
        };
        loop {
            let next = tokio::select! {
                biased;
                () = stop.requested.notified() => {
                    return Err(std::io::Error::other("stopped: the exec stream was closed"));
                }
                next = frames.next(FRAME_SILENCE_LIMIT) => next,
            };
            match next.map_err(broken)? {
                Some(Frame::Stdout(b) | Frame::Stderr(b)) => on_chunk(&b),
                Some(Frame::Heartbeat) => {}
                Some(Frame::ShellExit(code)) => {
                    let shell = ExecResult {
                        code: Some(code),
                        ..Default::default()
                    };
                    match on_orphans {
                        // A job: its leftovers are adopted, and the stream stays open until they
                        // have exited or a kill closes it.
                        Some(sink) => sink(&shell, REMOTE_LEFTOVERS.to_string()),
                        // A call that ends with its command: returning drops the stream, and the
                        // endpoint kills what is left.
                        None => return Ok(shell),
                    }
                }
                Some(Frame::Exit(code)) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        code: Some(code),
                        ..Default::default()
                    });
                }
                Some(Frame::Signal(signal)) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        signal: Some(signal),
                        ..Default::default()
                    });
                }
                Some(Frame::TimedOut) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        timed_out: true,
                        stderr: b"command exceeded its timeout".to_vec(),
                        ..Default::default()
                    });
                }
                Some(Frame::Error(e)) => {
                    return Err(broken(format!("the endpoint failed mid-command: {e}")));
                }
                None => {
                    return Err(broken(
                        "the stream ended before the command's final frame".to_string(),
                    ));
                }
            }
        }
    }
}

/// Whether a response is a stream rather than a one-shot object.
pub fn is_ndjson(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .is_some_and(|m| m.trim().eq_ignore_ascii_case(NDJSON))
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// Every answer in `output.json` decodes as the vectors say: the bytes, or an error.
    #[test]
    fn the_answer_vectors_decode_as_specified() {
        let doc: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/exec/output.json")).unwrap();
        let b64 = |v: &serde_json::Value| {
            base64::engine::general_purpose::STANDARD
                .decode(v.as_str().unwrap())
                .unwrap()
        };
        for case in doc["answers"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let got = serde_json::from_value::<ExecResponse>(case["answer"].clone())
                .map_err(|e| e.to_string())
                .and_then(|r| {
                    r.into_result("http://x", 1 << 20)
                        .map_err(|e| e.to_string())
                });
            let want = &case["expect"];
            if want["invalid"] == true {
                assert!(got.is_err(), "{name}: a conforming client reports an error");
                continue;
            }
            let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                got.code,
                Some(want["exit_code"].as_i64().unwrap() as i32),
                "{name}"
            );
            assert_eq!(got.stdout, b64(&want["stdout_base64"]), "{name}: stdout");
            assert_eq!(got.stderr, b64(&want["stderr_base64"]), "{name}: stderr");
        }
    }

    #[test]
    fn the_cap_is_on_the_decoded_bytes_of_both_streams_together() {
        let answer = |cap| {
            ExecResponse {
                exit_code: 0,
                stdout: "AAAA".into(),
                stderr: "AAAA".into(),
            }
            .into_result("http://x", cap)
        };
        assert!(answer(6).is_ok(), "3 + 3 decoded bytes fit a cap of 6");
        let over = answer(5).unwrap_err();
        assert_eq!(over.kind(), std::io::ErrorKind::FileTooLarge);
        assert!(over.to_string().contains("the command ran"), "{over}");
    }

    #[test]
    fn the_body_cap_holds_the_base64_of_a_cap_of_output() {
        // 16 MiB of output is 21_845_334 base64 characters (4 per 3 bytes, padded up).
        assert!(body_cap(16 * 1024 * 1024) >= 22_369_620);
        assert_eq!(body_cap(3), 4 + ENVELOPE_SLACK);
        assert_eq!(body_cap(4), 8 + ENVELOPE_SLACK);
    }

    #[test]
    fn a_url_must_be_http_or_https() {
        assert!(HttpExecRunner::new("https://exec.example/run").is_ok());
        assert!(HttpExecRunner::new("http://127.0.0.1:9000/exec").is_ok());
        assert!(HttpExecRunner::new("file:///etc/passwd").is_err());
        assert!(HttpExecRunner::new("not a url").is_err());
    }

    #[test]
    fn headers_parse_as_name_colon_value() {
        let (n, v) = HttpExecRunner::parse_header("Authorization: Bearer abc").unwrap();
        assert_eq!(n, "Authorization");
        assert_eq!(v, "Bearer abc");
        assert!(HttpExecRunner::parse_header("no-colon").is_err());
        assert!(HttpExecRunner::parse_header(": empty-name").is_err());
    }
}
