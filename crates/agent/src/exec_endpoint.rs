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
//! ## The protocol (v1.1)
//!
//! One `POST` to the configured URL:
//!
//! ```jsonc
//! // request
//! { "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
//! // request carrying stdin (v1.1, optional) — standard base64, fed to the command, then closed
//! { "command": "cat", "args": [], "timeout_ms": 10000, "stdin_base64": "c3RkaW4tcHJvYmU=" }
//! // response — 200
//! { "exit_code": 0, "stdout": "…", "stderr": "…" }
//! ```
//!
//! Deliberately the smallest thing that can carry a command and its result. Putting an adapter in
//! front of a vendor's SDK is a few dozen lines on their side, and that seam is where vendor
//! specifics belong — not in here.
//!
//! **v1.1 is backward compatible.** `stdin_base64` is the only addition, and only `ShellFs`'s file
//! writes send it — after a probe (`cat` must echo `stdin-probe`) has shown the endpoint honors it. A
//! v1 endpoint ignores the unknown field and answers the probe with empty stdout, so writes fall back
//! to argv-sized chunks. The field is never trusted blindly: writing through an endpoint that dropped
//! it would silently produce *empty files*.
//!
//! **The response is capped** ([`DEFAULT_MAX_RESPONSE_BYTES`], `--exec-max-response-bytes`). The body
//! is read incrementally and reading stops at the cap; an oversized response is an **error**, never a
//! partial result. The body is one JSON object, so a prefix of it cannot be parsed — there is no
//! honest partial `ExecResult` to return (not even the exit code, which may come after the output),
//! and presenting whatever fit as the command's output would be a confident wrong answer. The error
//! says the command *did run*, so the caller narrows its output instead of blindly repeating a
//! command that may not be idempotent. The cap applies to v1.1 objects only; a v1.2 stream is read
//! a frame at a time and never buffered whole.
//!
//! ## Streaming (v1.2)
//!
//! A tool that behaves differently depending on where it runs is a broken API. Over v1.1, `bash`
//! cannot see a remote command's output until it exits, so it cannot tell a silent build from a hung
//! one (the idle yield never fires: a hung command burns its whole timeout), cannot show progress,
//! and cannot kill a command it has given up on. v1.2 streams the output and makes **the connection
//! the command's lifetime**, so the endpoint stays stateless — no job ids, no poll or kill routes.
//!
//! The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.
//!
//! **Request.** The v1.1 request plus `"stream": true`:
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
//! | `{"signal": S}` | **final**: the command was killed by signal S (9 for SIGKILL) |
//! | `{"timed_out": true}` | **final**: the endpoint killed the command at `timeout_ms` |
//! | `{"error": "<message>"}` | **final**: the endpoint failed after answering 200 (lost the sandbox) |
//!
//! - Output is **standard base64 with padding** (RFC 4648 §4), so arbitrary bytes — invalid UTF-8,
//!   NULs, terminal escapes — survive. An output frame MUST NOT be empty and MUST carry at most
//!   64 KiB of decoded bytes ([`stream::MAX_FRAME_PAYLOAD`]); split anything longer.
//! - **Exactly one final frame, last.** After it the endpoint MUST end the body. A body that ends
//!   without one is a broken stream, never a success. Failures *before* answering are non-2xx
//!   statuses exactly as in v1.1; a command that cannot be spawned is an ordinary result (`stderr`
//!   frame, then `{"exit_code": 127}`), as a shell reports it.
//! - **Ordering.** Frames of one stream are in the order the bytes were read; the relative order of
//!   stdout and stderr is best-effort, as between two local pipes. The final frame comes after all
//!   output read from both pipes. Once the command's process group is empty, the endpoint SHOULD
//!   stop waiting for pipes held open by a descendant that left the group after a short grace (the
//!   client uses 100 ms) rather than until they close.
//! - **Flushing.** Each frame MUST reach the connection without buffering: the endpoint MAY coalesce
//!   output read within 100 ms into one frame, and MUST NOT hold any longer. Anything between the
//!   endpoint and the client must not buffer either (send `Cache-Control: no-store`, and e.g.
//!   `X-Accel-Buffering: no` behind nginx) — the probe below refuses an endpoint whose frames arrive
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
//!   and the deadline only bounds one that outlives every session. Over v1.1 nothing streams and
//!   nothing can be killed, so a `bash` command there sends [`UNTRACKED_DEADLINE`] (30 minutes): the
//!   only thing that ends a command that hangs.
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
//! the leftovers. Compatible both ways: an endpoint predating this rule ends the stream with the
//! shell (its leftovers run on, untracked), and a client predating it reads `shell_exit` as a
//! heartbeat (an unknown field) and keeps the job until the group ends.
//!
//! **Over v1.1 leftovers are untracked.** One request, one response: the endpoint answers when the
//! command exits, and nothing the agent can do afterwards reaches what it left running. A v1.1
//! endpoint SHOULD kill the command's process group once it has answered; one that does not leaves
//! `server &` running until the sandbox ends.
//!
//! **The probe.** Streaming is adopted only after the endpoint proves it, once per runner, on the
//! first streaming call ([`HttpExecRunner::streams`]): `sh -c 'printf a; sleep 0.3; printf b'` with
//! `"stream": true` must come back as `application/x-ndjson` with at least two output frames, `ab`,
//! `{"exit_code": 0}`, and the first output frame at least 100 ms before the final one. A v1.1
//! endpoint answers with one JSON object (it ignores the unknown field); a buffering endpoint or proxy
//! sends one frame, or every frame at once. Either way the runner stays on v1.1 for good, and `bash`
//! keeps v1.1's behavior exactly: no yield, no live progress, and its description says so. A probe
//! that could not reach the endpoint is not an answer and is retried on the next call. Only `bash`
//! streams: the filesystem backend's commands stay v1.1, whose single object they need anyway.
//!
//! **Conformance.** [`conformance::verify`] checks an endpoint against all of the above — framing,
//! the probe, exit/signal/timeout frames, heartbeats, the 64 KiB frame cap, kill on disconnect
//! measured against a grandchild's marker file, and leftover processes (`shell_exit`, the stream
//! held until they exit, killed on a disconnect after it) — and `tests/exec_conformance.rs` runs it against any
//! URL. `tests/fixtures/exec/v1.2-frames.json` holds the decoding vectors.
//!
//! ## Sandbox jobs (v1.3)
//!
//! A stream is the right owner for a command someone is waiting on and the wrong one for a job: a
//! replica that is reaped, redeployed or failed over closes every stream it holds. So a `bash` job
//! the model meant to keep (`run_in_background`, an adopted `cmd &`) is owned by the **sandbox**: it
//! runs until it exits, is killed, or its session ends, and any replica serving the session can
//! list it, read its output from an offset, and kill it. Every endpoint gets this without doing
//! anything: the agent builds it from plain exec commands ([`jobs::ShellJobs`], "the baseline" — a
//! directory per job under `/tmp/beyond-jobs/`, a `setsid` supervisor, a relay that writes the
//! output as offset-named chunks; see [`jobs`]). v1.3 is the same contract offered **natively**, for
//! an endpoint that would rather keep jobs its own way (a vendor's background commands, the
//! platform's own guest). [`jobs_native`] is the client.
//!
//! **Adoption.** Optional and probed, like stdin and streaming: the first job operation sends a
//! valid v1.1 request — `sh -c 'printf exec-v1.2'` — carrying `"job": {"op": "probe"}`. An endpoint
//! without the capability ignores the unknown field and runs the command; one with it MUST answer
//! `{"jobs": "v1.3"}`. A probe that could not reach the endpoint is retried later. A v1.3 endpoint
//! MUST also be a conforming v1.2 endpoint.
//!
//! **Requests.** Every job operation is a v1.1 request (the same URL, headers and envelope) plus a
//! `"job"` object naming `op` and `scope`. A **scope** is an opaque name for one session's jobs
//! (the agent sends a digest of tenant and session); an operation MUST only ever see the jobs of
//! its own scope. A job **id** is chosen by the agent (lowercase hex, at most 32 characters) and is
//! unique within its scope. A **status** is a job's exit code, or 128 + N for signal N. Except for
//! `start`, the request's `command` is `true` with no arguments: harmless if run. Unknown fields are
//! ignored. A refusal before answering (an unknown job for `attach`, a bad request) is a non-2xx
//! status, as in v1.1.
//!
//! | op | `job` fields | answer |
//! |---|---|---|
//! | `probe` | | `{"jobs": "v1.3"}` |
//! | `start` | `id`, `hold`, `label` — the command is the request's `command`/`args`/`cwd` | `hold: true` (with `"stream": true`): a job stream; `hold: false`: `{"started": pgid}` |
//! | `attach` | `id`, `offset` (with `"stream": true`) | a job stream from `offset` |
//! | `read` | `id`, `offset`, `max` | `{"state", "code", "offset", "end", "data"}` |
//! | `keep` | `id`, `num`, `kind`, `seen` | `{}` |
//! | `seen` | `id`, `offset` | `{}` |
//! | `kill` | `ids`, `remove`, `grace_ms` | `{"survivors": [{"id", "pids": [..]}]}` |
//! | `list` | | `{"jobs": [{"id", "state", "code", "kept", "num", "kind", "seen", "started", "ended", "label"}]}` |
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
//! - **Held.** A job started with `hold: true` lives as long as its stream unless it is **kept**: if
//!   the connection closes before the job's final frame and no `keep` arrived, the endpoint MUST kill
//!   its whole group, exactly as v1.2's "Disconnect kills". `keep` hands it to the endpoint for good
//!   (recording `num`, `kind` and `seen` for `list`); after it a closed stream kills nothing. A job
//!   started with `hold: false` is kept from the start. Closing an `attach` never kills.
//! - **A job stream** is a v1.2 stream (`application/x-ndjson`, heartbeats, frame size, flushing,
//!   unknown fields) with two more frames: `{"started": pgid}` first on a held start, and
//!   `{"gap": n}` when `n` bytes of output from the requested offset are no longer kept. Output
//!   frames (`stdout`; `stderr` is accepted too) carry the combined output in order. `{"shell_exit":
//!   N}` follows the output read up to the shell's exit when the shell exits leaving its group alive,
//!   once per stream. The final frame — after all output up to the job's end — is the job's status:
//!   `{"exit_code": N}` (or `{"signal": S}`), or `{"error"}`. An `attach` of an ended job streams
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
//! [`conformance::verify`] checks the capability when the probe says it is offered: a held job's
//! live output and end, a dropped hold killing an unkept job (measured on the far side), a kept job
//! surviving its hold and reading back from an offset (read and attach), `kill` reaching a job that
//! ignores `SIGTERM`, and `list`/`wipe` staying inside their scope. `tests/fixtures/exec/v1.3-jobs.json`
//! holds the decoding vectors.
//!
//! For targets that have no HTTP surface at all (`ssh`, `docker exec`, `kubectl exec`, any CLI), see
//! [`TemplateRunner`], which is the same idea expressed as an argv template. Its transport is a local
//! process, so its output streams natively; killing it is the hard part — see `far_kill`.
//!
//! ## Sandbox job notify (v1.4)
//!
//! A replica holds a v1.2 stream to every kept job it serves, and the endpoint writes a heartbeat on
//! it at least every 5 s for as long as the job runs. That traffic keeps the connection open, and it
//! keeps awake a sandbox that would otherwise sleep: an idle dev server holds its sandbox up for days.
//! v1.4 has the sandbox report when a kept job's state changes, so the replica can let go of the
//! stream and send nothing about the job until then.
//!
//! Every requirement in this section is tagged in bold (N1, N2, …) and names the check that
//! tests it: a conformance check ([`conformance::verify`], against any endpoint), a test in this repo,
//! or vectors in `tests/fixtures/exec/v1.4-*.json`. `tests/fixtures/exec/v1.4-checks.json` maps every
//! tag to its check and `tests/exec_notify_vectors.rs` fails when a tag has none. The edge's half is
//! FLEET.md's "Job-end callbacks", tagged E1, E2, …. A check named here and not yet written
//! is listed in that file with the task that adds it.
//!
//! **A callback is a hint, not the truth.** It says only "look at job `id` of scope `scope`". The
//! receiver learns what happened the way a reattach does, with `list` and `read`. So a lost, late,
//! duplicated, reordered or forged callback can delay a report, but it can never make one wrong.
//! Everything below follows from that.
//!
//! **The protocol defines messages, not a transport.** The messages are the `keep` extension, the
//! callback request and its answer, and the token. The callback is written here as an HTTPS request
//! the sandbox dials to the fleet's edge. A sandbox the fleet cannot dial, such as one in a customer's
//! own network (BYOC) that keeps a tunnel open to the fleet, carries the same request and the same
//! answer over that tunnel. Nothing in this section depends on which one is used, and the tunnel
//! itself is not specified here.
//!
//! **Terms.** A *notification* is the delivery of one callback for one job: one or more *attempts*,
//! each one HTTP request. A job's *state* is what `read` and `list` report (`running`, `exited`,
//! `ended`, `lost`, `gone`). A *state change* is `running` → `exited`, or any state → `ended` or
//! `lost`. *Observable* means that a `read` or `list` of the job that starts after that moment reports
//! it.
//!
//! ### Adoption
//!
//! - **N1** An endpoint that offers v1.4 MUST answer the v1.3 probe with
//!   `{"jobs": "v1.3", "notify": "v1.4"}`, and MUST be a conforming v1.3 endpoint. `jobs` stays
//!   `"v1.3"`, because a client that predates v1.4 compares it exactly and would otherwise lose
//!   native jobs. (check: conformance `notify offered (v1.4)`; fixture `v1.4-probe.json`)
//! - **N2** A client MUST treat notify as offered only when `jobs` is exactly `"v1.3"` and `notify` is
//!   exactly `"v1.4"`. Any other answer means no notify, and the client sends `keep` without it.
//!   (check: fixture `v1.4-probe.json`)
//! - **N3** The baseline ([`jobs::ShellJobs`]) offers notify only when its probe finds a notifier that
//!   reads the token from standard input or from a file, never from its arguments (`curl -K -`, or GNU
//!   `wget --config`), and the endpoint honours v1.1 `stdin_base64`. Busybox `wget` takes headers only
//!   as arguments, so a sandbox with nothing else gets no notify. The baseline's `keep` receives
//!   `notify` on standard input and stores it in a file of mode 0600 in the job's directory.
//!   (check: `sandbox_jobs_notify::baseline_offers_notify_only_with_a_safe_notifier`)
//!
//! An endpoint SHOULD NOT advertise notify when it knows it has no outbound network. No client relies
//! on the advertisement alone: nothing is released until a callback has actually arrived (N37).
//!
//! ### `keep` with `notify`
//!
//! `keep` gains one optional field:
//!
//! ```jsonc
//! { "command": "true", "args": [], "timeout_ms": 30000,
//!   "job": { "op": "keep", "scope": "3f…", "id": "0123456789abcdef", "num": 3,
//!            "kind": "background", "seen": 120,
//!            "notify": { "url": "https://edge.example/_beyond/agent?session_id=a07-s3.0f1e",
//!                        "token": "bjn_v1.5d1c0a39.Jb…" } } }
//! ```
//!
//! - **N4** An endpoint MUST refuse a `notify` unless all of these hold. `url` is an absolute URL of
//!   at most 2048 bytes with a non-empty host, no userinfo and no fragment, and its scheme is `https`,
//!   or `http` when its host is `localhost`, an address in `127.0.0.0/8`, or `[::1]`, for tests.
//!   `token` is 1 to 512 bytes of RFC 7235 `token68` (`A-Z a-z 0-9 - . _ ~ + /`, then any `=`).
//!   Unknown members of `notify` are ignored. (check: conformance
//!   `keep refuses an invalid notify`; fixture `v1.4-keep.json`)
//! - **N5** A refused `notify` is a `400` answer, and the endpoint MUST then record nothing: the job is
//!   not kept, and no notify is stored. A `list` after it reports the job exactly as before the
//!   request. (check: conformance `keep refuses an invalid notify`; fixture `v1.4-keep.json`)
//! - **N6** An accepted `keep` MUST record the keep and its notify together, and the latest `keep`
//!   governs: one with `notify` replaces the job's notify, and one without it removes it. (check:
//!   conformance `notify follows the latest keep`)
//! - **N7** The token is opaque. An endpoint MUST send it byte for byte as it was given, and MUST NOT
//!   parse, check or rewrite it. (check: conformance `callback request shape`; fixture
//!   `v1.4-callback.json`)
//!
//! ### When the endpoint calls back
//!
//! - **N8** Proof of reach. On every accepted `keep` that carries `notify`, the endpoint MUST start a
//!   notification for that job. Its first attempt begins after the keep is recorded (a `list` made
//!   then reports the job kept) and no later than 2 s after the `keep` is answered. The job may well
//!   be running; the receiver checks, finds that, and does nothing else. The callback's arrival is
//!   what tells the replica that the path from this sandbox to its session works (N37). (check:
//!   conformance `first callback at keep`)
//! - **N9** State changes. When a job whose latest `keep` carries `notify` makes a state change, the
//!   endpoint MUST start a notification whose first attempt begins after the new state is observable
//!   and no later than 2 s after it became observable. An end is reported only once the job's group is
//!   empty and `read` says `ended`; `exited` (the shell is gone, its group is not) is reported too,
//!   because the replica reports leftover processes the same way it does over a held stream. (check:
//!   conformance `callback after the job ends`)
//! - **N10** Coalescing. A notification that has not begun its first attempt MAY stand for a later
//!   state change as well. One that has begun MUST NOT: every state change gets an attempt that begins
//!   after it is observable. (check: conformance `callback after the job ends`)
//! - **N11** The endpoint MUST NOT begin an attempt for a job after it has been removed (`remove`,
//!   `kill` with `remove`, `wipe`), or after a `keep` without `notify` has removed its notify.
//!   (check: conformance `no callback after remove`)
//! - **N12** The endpoint MUST NOT send a callback for a job whose latest `keep` carried no `notify`,
//!   or for a job never kept. (check: conformance `no callback without notify`)
//!
//! ### The callback request
//!
//! - **N13** Each attempt is one `POST` to `url` exactly as given, with `Content-Type:
//!   application/json`, `Authorization: Bearer <token>`, no cookies, and the body
//!   `{"type":"job_ended","scope":"<scope>","id":"<id>"}`: exactly these three members, `scope` and
//!   `id` byte for byte as the `keep` named them, at most 512 bytes. The endpoint MUST NOT send
//!   anything else in the body, whatever the job's state. `type` is constant: it names the `serve`
//!   command the body is (see "The receiver"), so `id`, the job's id, is also that command's
//!   correlation id. (check: conformance `callback request shape`; fixture `v1.4-callback.json`)
//! - **N14** The endpoint MUST NOT follow a redirect. A `3xx` answer ends the notification, since a
//!   redirect would carry the token to an origin `keep` did not name. (check: conformance
//!   `callback does not follow redirects`; fixture `v1.4-retry.json`)
//! - **N15** Over `https`, the endpoint MUST verify the server's certificate and name. A connection
//!   that fails verification sends no request and counts as a connection error. (check: conformance
//!   `callback verifies TLS`)
//! - **N16** An attempt that has no complete answer 30 s after it began counts as a connection error,
//!   and the endpoint MUST abandon it. (check: fixture `v1.4-retry.json`; conformance
//!   `callback retry and stop`)
//! - **N17** The endpoint MUST NOT have more than one attempt for a job in flight at a time. (check: conformance
//!   `callback retry and stop`)
//!
//! ### The answer, and retries
//!
//! - **N18** The endpoint MUST act on the status as this table says. (check: fixture
//!   `v1.4-retry.json`; conformance `callback retry and stop`)
//!
//! | answer | meaning | the endpoint |
//! |---|---|---|
//! | `2xx` | the receiver has looked at the job | ends the notification: done |
//! | `401`, `403`, `404`, `410` | the token, the session or the job is not accepted, and never will be | ends the notification: stop |
//! | `429`, `5xx`, a connection error, a TLS failure, no answer in 30 s | not now | retries, per N19 |
//! | anything else (`1xx`, `3xx`, `400`, `413`, `421`, other `4xx`) | the request itself is wrong; sending it again cannot help | ends the notification: stop |
//!
//! `Retry-After` is ignored: the schedule below already outlasts a failover.
//!
//! - **N19** A notification makes at most 12 attempts. After a failed attempt *n* (n = 1 … 11), the
//!   endpoint MUST wait `d(n) · (1 − u/2)` before attempt *n* + 1, where `d(n) = min(2^(n−1), 60)`
//!   seconds, `u` is drawn uniformly from `[0, 1)` afresh for every wait, and the wait starts when
//!   attempt *n* ended (its answer, its connection error, or its 30 s limit). After the twelfth failed
//!   attempt the notification is over. (check: fixture `v1.4-retry.json`; conformance
//!   `callback retry and stop`)
//!
//! So the waits are `1, 2, 4, 8, 16, 32, 60, 60, 60, 60, 60` s at most and half that at least. The
//! twelfth attempt begins between 181.5 s and 363 s (plus the time the earlier attempts took) after
//! the first, and the whole notification is over within 723 s. A storage-lease takeover after a hard
//! kill took at most 120 s in the fleet's measurements (FLEET.md, Requirement 3), so a notification
//! whose attempts fail instantly is still trying after one: its tenth attempt begins at 121.5 s at the
//! earliest.
//!
//! ### The token stays secret
//!
//! - **N20** The token MUST NOT appear in the arguments or the environment of any process in the
//!   sandbox. A model that runs `ps` there would otherwise copy it into a transcript. (check:
//!   conformance `the token never reaches argv, environment or answers`)
//! - **N21** The token MUST NOT appear in any answer or stream the endpoint sends: `list`, `read`,
//!   `attach`, a job's output, an error. (check: conformance
//!   `the token never reaches argv, environment or answers`)
//! - **N22** The token MUST NOT be written to a log. A verifier cannot see an external endpoint's logs,
//!   so this repo tests it for the endpoints it ships: the baseline writes the token only to its 0600
//!   `notify` file, and `exec_mock` never prints it. (check:
//!   `sandbox_jobs_notify::the_token_is_never_logged`)
//!
//! What a leaked token costs is small by construction. It can only start a reconcile of the one job it
//! names, in the one session it names (N45).
//!
//! ### The receiver
//!
//! A callback is a `serve` command, `job_ended`, POSTed to the session's own address,
//! `/_beyond/agent?session_id=<id>`. The edge routes it like any request for that session, to the
//! replica that owns it or to one that opens it (FLEET.md, E1–E6). In service mode the edge
//! also attaches a session grant, as on every request: the grant says which tenant and session the
//! request is for and carries what the replica needs to open the session. The grant does not
//! authorise the callback; the token does.
//!
//! - **N23** A request whose `Authorization` is `Bearer bjn_v1.…` is a callback, and the token MUST
//!   authorise nothing but `job_ended` for the job it names. A callback that is not a `POST` whose
//!   body is a `job_ended` command, a WebSocket upgrade included, is answered `403`. (check:
//!   `serve_job_ended::a_token_authorises_nothing_else`)
//! - **N24** A `job_ended` without a notify token is answered `403`. A session grant alone, a user's
//!   included, never authorises it. A replica that holds no notify key (outside service mode) answers
//!   every `job_ended` `401`. (check: `serve_job_ended::a_grant_alone_cannot_end_a_job`)
//! - **N25** The receiver MUST check, in this order, and answer at the first failure: the grant (`401`,
//!   as for any request), the URL's session id against the grant (`400`/`403`, as for any request), a
//!   body over 4096 bytes (`413`), the body: a JSON object of exactly `type`, `scope` and `id`, with
//!   `id` a v1.3 job id and `scope` a string of at most 64 bytes (`400`), the token (N45; `401`),
//!   then `scope` against the session's own scope ([`jobs::JobScope::for_session`] of the grant's
//!   tenant and session; `403`). Only a request that passes all of them may open, create or start a
//!   session, or send anything to the sandbox. (check: `serve_job_ended::a_forged_callback_opens_nothing`;
//!   fixture `v1.4-callback.json`)
//! - **N26** A `job_ended` MUST NOT create a session. One for a session that does not exist is
//!   answered `404`. (check: `serve_job_ended::never_creates_a_session`)
//! - **N27** A `job_ended` request MUST NOT count as an attached client, for the idle reaper or for
//!   N33. (check: `serve_job_ended::the_callback_is_not_a_client`)
//! - **N28** Each refusal's body is fixed text for its status and MUST NOT say which check failed
//!   beyond the status itself. (check: `serve_job_ended::refusals_say_nothing`)
//!
//! A **reconcile** is the reattach read-back (`tools::bash_sandbox`) of one session's sandbox jobs:
//! `list` the scope, adopt what runs, read what ended up to its end, and record notices.
//!
//! - **N29** A verified callback MUST be answered `200` (with the command's `response` frame, like any
//!   other POSTed command) only once the job's state has been recorded
//!   in the session: either a reconcile that began after the request arrived has finished, or the
//!   session had already recorded the job's end, which no later reconcile can change. "Recorded"
//!   means a notice that is due has been persisted to the transcript (N33), or delivered. It is
//!   answered `410` when, after that reconcile, the session has no such job, and `503` when the
//!   reconcile could not finish (the sandbox could not be listed). The endpoint retries a `503`, so
//!   its schedule is what carries the report across a replica that fails mid-reconcile. (check:
//!   `serve_job_ended::answers_after_the_reconcile`; fixture `v1.4-callback.json`)
//! - **N30** A session runs at most one reconcile at a time. A callback that arrives while one runs
//!   waits for the next, and every callback that arrives during the same reconcile shares that next
//!   one: callbacks in a burst cost at most two reconciles. A reconcile that callbacks started MUST
//!   NOT begin less than 1 s after the session's previous reconcile ended, so a program in the
//!   sandbox that sends a real token in a loop costs at most one reconcile a second. (check:
//!   `serve_job_ended::one_reconcile_at_a_time`)
//! - **N31** A callback for a job whose end the session has already recorded MUST be answered `200`
//!   without contacting the sandbox. (check: `serve_job_ended::repeats_are_free`)
//! - **N32** A callback changes nothing by itself. Only what a reconcile reads changes a job's state
//!   or produces a notice. A callback for a job that is still running leaves it running, and produces
//!   no notice and no model turn. (check: `serve_job_ended::a_callback_is_only_a_hint`)
//!
//! **Waking.** This applies to every job notice in a `serve` session, however the replica learned of
//! it (a held stream, a reconcile, a reattach):
//!
//! - **N33** With at least one client attached when a job's notice becomes due, the session behaves as
//!   it does today: the notice starts a turn. With none attached, the notice MUST be persisted to the
//!   session's transcript and MUST NOT start a model turn. The next turn a client starts carries it,
//!   exactly once, and attaching alone starts no turn. (check:
//!   `serve_job_ended::no_turn_without_a_client`)
//! - **N34** A job's end, and its leftover processes, are each reported at most once per session,
//!   across every source: a held stream, a callback's reconcile, a reattach at open, and a `bash_job`
//!   wait. A notice persisted under N33 counts as reported. (check:
//!   `serve_job_ended::one_report_per_end`; fleet-sim `sandbox-job-exit-noticed-once`)
//! - **N35** A session that a `job_ended` request opens MUST reconcile at once, rather than defer its
//!   reattach to its first use of the sandbox: the sandbox just called, so it is awake. (check:
//!   `serve_job_ended::a_callback_open_reconciles_now`)
//! - **N36** A job's notice MUST be persisted to the transcript, or delivered, before the job's record
//!   is removed from the sandbox (`remove`). A replica that dies between the two leaves a record that
//!   the next reconcile matches against the transcript, and does not report again. (check:
//!   `serve_job_ended::the_notice_is_persisted_before_the_record_goes`)
//!
//! ### Release (normative)
//!
//! Per kept job, in the session task that owns the job:
//!
//! | state | what the replica holds | event | action | next |
//! |---|---|---|---|---|
//! | (parking) | the start's stream | notify is available (N38) | arm a 30 s proof window, then send `keep` with `notify` | proving |
//! | (parking) | the start's stream | notify is not available | send `keep` without `notify`, as today | held |
//! | proving | the stream | `keep` is answered `400` | send `keep` again without `notify` | held |
//! | proving | the stream | a verified callback for this job, inside the window | reconcile; if the job is `running` or `exited`, close the stream | released |
//! | proving | the stream | the same, and the reconcile finds it `ended` or `lost` | report the end | ended |
//! | proving | the stream | the window closes with no such callback | nothing | held |
//! | proving, held | the stream | the stream's final frame | report the end, as today | ended |
//! | proving, held | the stream | the stream breaks | follow the job, as today | unchanged |
//! | held | the stream, or a follow | a verified callback | reconcile; never release | held, or ended |
//! | released | nothing | a verified callback | reconcile: `running` changes nothing, `exited` reports the leftovers, `ended` or `lost` reports the end | released, or ended |
//! | released | nothing | a tool call or command about the job (`bash_job` peek, wait or kill) | do it; a wait that attaches lets go again when it returns | released, or ended |
//! | ended | nothing | a verified callback | answer `200`, contact nothing (N31) | ended |
//!
//! - **N37** The replica MUST release a job's stream only on the `proving` → `released` row: a verified
//!   callback for that job arrived inside its proof window, and a reconcile that began after it found
//!   the job `running` or `exited`. Nothing else releases a stream: not the probe's answer, not an
//!   accepted `keep`, not a callback after the window. (check:
//!   `job_notify::release_waits_for_the_first_callback`)
//! - **N38** Notify is available only when every one of these holds: the replica is in service mode
//!   and holds a notify key (N47), the operator has configured the origin the fleet's edge serves
//!   sessions at, and the endpoint offers notify (N2) or the baseline does (N3). The `url` the
//!   replica sends is that origin's `/_beyond/agent?session_id=<session id>`, and the `token` is
//!   minted for this session and job (N48). (check:
//!   `job_notify::keep_without_notify_when_unavailable`)
//! - **N39** The proof window opens before `keep` is sent, so a callback that overtakes `keep`'s answer
//!   still counts, and it closes 30 s later. With instant failures, the first five attempts of the
//!   proof notification all begin within 17 s of `keep`'s answer (2 s, N8, then 15 s, N19), which
//!   leaves time for the fifth one's answer.
//!   (check: `job_notify::proof_window_opens_before_keep`)
//! - **N40** When the window closes with no callback, or `keep` with `notify` is refused, the job MUST
//!   stay held for good, exactly as today. A later callback is reconciled, and never releases it.
//!   (check: `job_notify::no_callback_keeps_the_stream`; `job_notify::a_refused_notify_falls_back`)
//! - **N41** Once a job is released, the replica MUST send nothing concerning it (no request, stream,
//!   attach or poll) except for a reconcile that a callback or a session's open started, and the
//!   operations a tool call or client command asks for: a `bash_job` peek, wait or kill, the `seen`
//!   that records output the model was just shown, and the `wipe` that ends the session. A wait that
//!   attaches to the job MUST close the attach when it returns with the job still running. (check:
//!   `job_notify::silent_after_release`; `job_notify::a_wait_lets_go_again`)
//!
//! What release costs: once released, a job whose end callback is lost after every attempt (12
//! attempts over 3 to 12 minutes) is reported at its session's next reconcile, which is the next
//! callback, `bash_job` call or session open, rather than at once. N41 forbids the polling that
//! would bound that delay. A session MAY reconcile its released jobs when it uses its sandbox for
//! another reason, since the sandbox is awake then anyway.
//!
//! ### Crash, takeover and rotation
//!
//! The release state lives only in the session task. A replica that dies loses it, and so does a
//! session that is reaped: the jobs run on in the sandbox, kept.
//!
//! - **N42** A token is a function of the fleet key and of (tenant, session, scope, job id) alone
//!   (N45), so every replica mints the same token for the same job. A callback minted by one
//!   replica is verified by any other, and a session that moves has nothing to hand over. (check:
//!   fixture `v1.4-token.json`)
//! - **N43** A replica that reattaches a running kept job while notify is available MUST send it
//!   `keep` again, with the `num`, `kind` and `seen` its listing reported, `notify`, and a token under
//!   its current key, and treat it as `proving`, holding a follow until the proof arrives. It MUST NOT
//!   go on following a job it has released. (check: `job_notify::takeover_proves_again`)
//!
//! So a replica killed in `proving` leaves the job kept and running, and the proof callback, or the
//! next client request, opens the session elsewhere, which reattaches and proves again. One killed
//! in `released` leaves nothing to hold, and the job's end callback opens the session wherever the
//! edge places it (N35). A sandbox that restarts loses its pending notifications along with its
//! jobs, which a reconcile then reports `lost`.
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
//! S         the fleet secret: the 32 bytes the replica's seal key decodes to (`--seal-key`,
//!           AI_AGENT_SEAL_KEY_B64), exactly as given, before X25519 clamps them
//! tag       HMAC-SHA256(K, m)
//! m         lp("bjn_v1") ‖ lp(kid) ‖ lp(purpose) ‖ lp(tenant) ‖ lp(session id) ‖ lp(scope) ‖ lp(job id)
//! lp(x)     the length of x in bytes as a 4-byte big-endian unsigned integer, then x's UTF-8 bytes
//! purpose   "job_ended"
//! ```
//!
//! - **N44** A token that is not exactly that grammar MUST be refused as malformed: another prefix or
//!   version, a field count other than three, a kid that is not eight lowercase hex digits, a `mac`
//!   that is not 43 characters of the base64url alphabet, or one whose final character carries
//!   non-zero padding bits (it would not re-encode to itself). (check: fixture `v1.4-token.json`)
//! - **N45** The verifier computes `m` from the grant's tenant, the URL's session id (which N25
//!   has already checked against the grant), the body's `scope` and `id`, and the constant purpose.
//!   It takes none of them from the token. So a token verifies only for the tenant, session, scope,
//!   job and purpose it was minted for. (check: fixture `v1.4-token.json`)
//! - **N46** The tag MUST be compared in constant time (`hmac::Mac::verify_slice`), and in full: no
//!   truncated tags. (check: `notify_token::the_tag_is_compared_in_constant_time`; fixture
//!   `v1.4-token.json`)
//! - **N47** The verifier MUST select a key by kid, from the key derived from its current seal key and
//!   from each retired seal key it is given for this purpose. A kid it holds no key for is refused as
//!   an unknown kid. The minter MUST mint only under the current key. (check: fixture
//!   `v1.4-token.json`; `notify_token::mints_only_under_the_current_key`)
//! - **N48** The minter MUST mint only for the session's own scope ([`jobs::JobScope::for_session`]
//!   of its tenant and session) and for a job of that session. (check:
//!   `notify_token::mints_only_its_own_scope`)
//!
//! The verdicts are `valid`, `malformed`, `unknown_kid` and `bad_mac`. Every verdict but `valid` is
//! answered `401` with the same body (N28).
//!
//! **Why this key.** The only secret that every replica of a fleet already holds and that no one else
//! holds is the fleet's X25519 seal secret. A replica holds only the public halves of the grant
//! signing keys, which cannot key a MAC, and a tenant's DEK is per tenant rather than one key across
//! replicas. HKDF under its own `info` label makes `K` independent of every other use of `S`: the
//! grant's key agreement uses `S` only as an X25519 scalar, and its own HKDF runs over the shared
//! secret with the `info` label `bsg_v1 seal`. `S` is never used as a MAC key directly. The kid is a
//! fingerprint of `K` rather than an assigned number, because the seal key has no kid of its own. It
//! reveals nothing about `K`.
//!
//! **Why no expiry.** The jobs this exists for are the long ones: a dev server runs for days, and a
//! token that expired first would lose exactly those reports. Expiry would buy nothing, because the
//! token's power is already bounded: it can start a reconcile of one named job of one session, and
//! that is all. It reads nothing, runs nothing and returns nothing, and once the job's end is
//! recorded a repeat costs no sandbox request at all (N31).
//!
//! **What binds tenant and session.** The control plane signs both into the session grant, and the
//! replica checks the URL's session id against the grant before it checks the token (N25). The
//! token's tag covers both (N45), so a token minted for one session fails on every other, in
//! every tenant.
//!
//! **Rotation.** Rotating the seal key changes `K` and its kid. A replica given the old seal key as a
//! retired notify key goes on verifying tokens minted under it, and never opens a grant with it. A
//! reattach re-mints the tokens of the jobs still running under the current key (N43). Once the
//! retired key is dropped, a callback under it is answered `401` and the endpoint stops, and that
//! job's end is reported at the next reconcile.
//!
//! ### What each invariant rests on
//!
//! | invariant | requirements |
//! |---|---|
//! | I1: every job end is reported exactly once, at worst at the session's next open | N9, N10, N19, N29, N33, N34, N35, N36 |
//! | I2: no model turn without an attached client | N27, N32, N33 |
//! | I3: a callback is a hint | N13, N29, N30, N31, N32 |
//! | I4: a stream is released only after the first callback | N8, N37, N39, N40 |
//! | I5: nothing crosses the connection for a released job | N41 |
//! | I6: the token is narrow | N23, N24, N25, N44–N48 |
//! | I7: never worse than today | N2, N5, N38, N40, and "What release costs" |

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::tools::exec::{
    ChunkSink, CommandRunner, ExecResult, JobHooks, Kill, STREAM_HEAD, STREAM_TAIL, SpawnSink,
    UNTRACKED_DEADLINE, drain_capped, feed_stdin,
};

pub mod conformance;
mod far_kill;
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
    /// v1.1: bytes for the command's stdin, standard base64. Absent means no stdin — v1 behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    stdin_base64: Option<String>,
    /// v1.2: answer with a stream of frames instead of one object. Absent means v1.1.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    /// v1.3: a job operation (see "Sandbox jobs (v1.3)"). Absent means the command is run as ever.
    #[serde(skip_serializing_if = "Option::is_none")]
    job: Option<&'a serde_json::Value>,
}

/// What the agent expects back. Extra fields are ignored, so an endpoint may return more.
#[derive(Debug, Deserialize)]
struct ExecResponse {
    exit_code: i32,
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
}

/// The `timeout_ms` a streamed `bash` job sends: `i32::MAX` ms (~24.8 days), the longest deadline
/// every endpoint can honor — Node's `setTimeout` fires at once for anything longer. A job's lifetime
/// is the connection's (see "Disconnect kills"), so this only has to outlast any session; it is not a
/// limit anyone should meet.
pub const STREAM_DEADLINE: Duration = Duration::from_millis(i32::MAX as u64);

/// What a remote command left running, as `on_orphans` describes it: the runner cannot list the far
/// side's processes, only that its process group still has some.
pub(crate) const REMOTE_LEFTOVERS: &str = "processes on the exec target";

/// How long a template command's output may still be in flight once its shell-exit marker has
/// arrived — the local runner's 100 ms (`bash_proc::SETTLE`).
const SHELL_EXIT_SETTLE: Duration = Duration::from_millis(100);

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
    max_response_bytes: usize,
    /// Whether the endpoint has proved it streams (v1.2) — probed once, on the first streaming call,
    /// and cached. A probe that could not reach the endpoint is not cached: that is an outage, not
    /// an answer.
    streams: tokio::sync::OnceCell<bool>,
    /// Whether the sandbox can run jobs of its own ([`jobs::ShellJobs`]) — probed once, the same way.
    jobs_probe: Arc<tokio::sync::OnceCell<bool>>,
    /// Whether the endpoint offers jobs natively (v1.3, [`jobs_native`]) — probed once, the same way.
    native_jobs: tokio::sync::OnceCell<bool>,
    /// Raised when the endpoint could not be reached or refused a request: a sign the sandbox
    /// behind it may have been replaced, so whatever was learned about it (the probe, the
    /// capabilities above) should be learned again. Read by the [`ExecTarget`] that shares it.
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
                agent_core::ensure_provider();
                reqwest::Client::new()
            },
            headers: Vec::new(),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            streams: tokio::sync::OnceCell::new(),
            jobs_probe: Arc::default(),
            native_jobs: tokio::sync::OnceCell::new(),
            suspect: None,
        })
    }

    /// Raise `flag` whenever a request fails to reach the endpoint or is refused by it (a non-2xx
    /// answer) — see [`ExecTarget::watching`].
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
    /// asks for v1.2 frames; whether they came is the response's `Content-Type` to say.
    async fn send(
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

    /// [`Self::send`], carrying a v1.3 job operation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn send_job(
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
            stdin_base64: stdin.map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
            stream,
            job,
        };
        let mut req = self.client.post(&self.url).json(&body);
        for (name, value) in &self.headers {
            req = req.header(name.as_str(), value.as_str());
        }
        // A little past the command's own budget, so the endpoint gets the chance to enforce and
        // report the timeout itself — its message is more useful than a bare client-side abort. It
        // bounds the whole exchange, a stream's body included.
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
                    std::io::Error::other(format!("exec endpoint {} unreachable: {e}", self.url))
                }
            })?;
        let status = resp.status();
        if !status.is_success() {
            // The endpoint refused the *request* — a bad URL, auth, a dead sandbox. Distinct from the
            // command running and exiting non-zero, which is an ordinary result.
            self.suspect();
            let (body, _) = self.read_capped(resp).await?;
            return Err(std::io::Error::other(format!(
                "exec endpoint {} returned {}: {}",
                self.url,
                status.as_u16(),
                excerpt(&body)
            )));
        }
        Ok(resp)
    }

    /// One request, one response: the whole v1.1 protocol.
    async fn post(
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

    /// Read and parse a v1.1 response body: one JSON object, capped.
    async fn envelope(&self, resp: reqwest::Response) -> std::io::Result<ExecResponse> {
        let (body, over_cap) = self.read_capped(resp).await?;
        if over_cap {
            return Err(std::io::Error::other(format!(
                "exec endpoint {}: the response exceeded the {}-byte cap and was discarded — the \
                 command ran, but its output is too large to return; narrow it (e.g. pipe through \
                 `head`) rather than re-running it unchanged",
                self.url, self.max_response_bytes
            )));
        }
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::other(format!(
                "exec endpoint {} returned unparseable JSON ({e}): {}",
                self.url,
                excerpt(&body)
            ))
        })
    }

    /// Read the body incrementally, stopping at the cap instead of buffering whatever the endpoint
    /// sends. `true` means the cap was hit and the bytes returned are only a prefix.
    pub(crate) async fn read_capped(
        &self,
        mut resp: reqwest::Response,
    ) -> std::io::Result<(Vec<u8>, bool)> {
        let cap = self.max_response_bytes;
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
        // Sent as `stdin_base64`. A v1 endpoint ignores it and runs the command with no input, so an
        // `Ok` here is not proof of delivery — see the module doc on the probe.
        let parsed = self.post(program, args, cwd, timeout, Some(stdin)).await?;
        Ok(ExecResult {
            code: Some(parsed.exit_code),
            stdout: parsed.stdout,
            stderr: parsed.stderr,
            ..Default::default()
        })
    }

    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let parsed = self.post(program, args, cwd, timeout, None).await?;
        Ok(ExecResult {
            code: Some(parsed.exit_code),
            signal: None,
            stdout: parsed.stdout,
            stderr: parsed.stderr,
            timed_out: false,
            truncated: false,
        })
    }

    /// v1.2 when the endpoint has proved it streams; otherwise v1.1, which never calls the sink.
    ///
    /// The sink gets *every* byte of *both* streams or none: a v1.1 answer (or a v1.2 endpoint that
    /// answers this request with one object anyway) goes back as `stdout`/`stderr` for `bash` to
    /// assemble. An earlier override fed the sink stdout alone, told `bash` the output had streamed,
    /// and every remote command that printed to stdout lost its stderr.
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
    /// what ends it; a v1.1 endpoint gets [`UNTRACKED_DEADLINE`], the only thing that can end a
    /// command it cannot stream or stop. See the module doc's "Timeouts".
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        let deadline = if self.streams().await {
            STREAM_DEADLINE
        } else {
            UNTRACKED_DEADLINE
        };
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

    async fn tracks_jobs(&self) -> bool {
        self.streams().await
    }

    /// v1.2 only: stdout frames to `on_stdout`, stderr frames to `on_stderr`, until the final frame.
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
        if !self.streams().await {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "this exec endpoint does not stream (protocol v1.1)",
            ));
        }
        let resp = self
            .send(program, args, cwd, STREAM_DEADLINE, None, true)
            .await?;
        if !is_ndjson(&resp) {
            // One object after all (a fleet mid-rollout): everything arrives at once, at the end.
            let parsed = self.envelope(resp).await?;
            on_stdout(parsed.stdout.as_bytes());
            on_stderr(parsed.stderr.as_bytes());
            return Ok(ExecResult {
                code: Some(parsed.exit_code),
                ..Default::default()
            });
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

    async fn attaches(&self) -> bool {
        self.streams().await
    }

    /// Native (v1.3) when the endpoint offers it, the baseline over plain exec commands otherwise.
    fn sandbox_jobs(self: Arc<Self>) -> Option<Arc<dyn jobs::SandboxJobs>> {
        let shell = jobs::ShellJobs::with_probe(self.clone(), self.jobs_probe.clone());
        Some(Arc::new(jobs_native::HttpJobs::new(self, shell)))
    }
}

impl HttpExecRunner {
    /// Whether the endpoint offers sandbox jobs natively (v1.3), probing it the first time. See the
    /// module doc's "Sandbox jobs (v1.3)".
    pub async fn offers_jobs(&self) -> bool {
        self.native_jobs
            .get_or_try_init(|| async {
                let native = jobs_native::probe(self).await?;
                if native {
                    tracing::info!(url = %self.url, "exec endpoint offers sandbox jobs (v1.3)");
                }
                Ok::<bool, std::io::Error>(native)
            })
            .await
            .copied()
            .unwrap_or(false)
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
    /// Whether this endpoint streams (v1.2), probing it the first time. See the module doc's "The
    /// probe".
    pub async fn streams(&self) -> bool {
        self.streams
            .get_or_try_init(|| async {
                let resp = self
                    .send(
                        "sh",
                        &["-c".to_string(), stream::PROBE_SCRIPT.to_string()],
                        None,
                        Duration::from_secs(10),
                        None,
                        true,
                    )
                    .await?;
                if !is_ndjson(&resp) {
                    return Ok::<bool, std::io::Error>(false);
                }
                let seen = stream::observe_probe(resp)
                    .await
                    .map_err(std::io::Error::other)?;
                let streams = seen.streams();
                if !streams {
                    tracing::info!(url = %self.url, why = %seen.explain(), "exec endpoint does not stream; `bash` waits for each command to exit");
                }
                Ok(streams)
            })
            .await
            .copied()
            .unwrap_or(false)
    }

    /// The v1.2 exchange. Falls back to v1.1 when the endpoint has not proved it streams, or when it
    /// answers this request with one JSON object after all (a fleet mid-rollout).
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
        if !self.streams().await {
            return self.run(program, args, cwd, timeout).await;
        }
        let stop = Arc::new(StreamStop::default());
        // Declared before the response: dropped after it, so "released" is only ever set once the
        // connection has been let go.
        let _released = Released(stop.clone());
        let resp = self.send(program, args, cwd, timeout, None, true).await?;
        if !is_ndjson(&resp) {
            let parsed = self.envelope(resp).await?;
            return Ok(ExecResult {
                code: Some(parsed.exit_code),
                stdout: parsed.stdout,
                stderr: parsed.stderr,
                ..Default::default()
            });
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
                        stderr: "command exceeded its timeout".into(),
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

/// Whether a response is a v1.2 stream rather than a v1.1 object.
fn is_ndjson(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .is_some_and(|m| m.trim().eq_ignore_ascii_case(NDJSON))
        })
}

/// A [`CommandRunner`] that runs each command through a caller-supplied **argv template**, for targets
/// with no HTTP surface: `ssh`, `docker exec`, `kubectl exec`, `podman`, a vendor's own CLI.
///
/// The template is a list of arguments; the literal `{}` element is replaced by the command and its
/// arguments. For example `["ssh", "build-host", "--", "{}"]` or
/// `["docker", "exec", "my-container", "{}"]`.
///
/// **The command is never rendered into a string.** `{}` expands to *multiple argv entries*, one per
/// token, so a path containing spaces, quotes or `;` stays a single argument on the far side and
/// cannot be reparsed as syntax. A template that pasted the command into a shell string would make
/// every model-supplied path an injection vector.
pub struct TemplateRunner {
    template: Vec<String>,
    /// Whether the far side can run jobs of its own ([`jobs::ShellJobs`]) — probed once.
    jobs_probe: Arc<tokio::sync::OnceCell<bool>>,
}

impl TemplateRunner {
    /// Build from a template that must contain exactly one `{}` placeholder.
    pub fn new(template: Vec<String>) -> Result<Self, String> {
        let holes = template.iter().filter(|t| *t == "{}").count();
        if holes != 1 {
            return Err(format!(
                "exec template needs exactly one `{{}}` placeholder for the command, found {holes} \
                 in {template:?}"
            ));
        }
        if template.first().is_none_or(|p| p == "{}") {
            return Err("exec template must start with a program to run".to_string());
        }
        Ok(Self {
            template,
            jobs_probe: Arc::default(),
        })
    }

    /// Parse a whitespace-separated template, the shape a CLI flag carries.
    pub fn parse(raw: &str) -> Result<Self, String> {
        Self::new(raw.split_whitespace().map(str::to_string).collect())
    }

    fn expand(&self, program: &str, args: &[String]) -> Vec<String> {
        expand(&self.template, program, args)
    }
}

#[async_trait]
impl CommandRunner for TemplateRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            Some(timeout),
            None,
            None,
            None,
            None,
            None,
        )
        .await
    }

    /// The transport is a local process, so its output streams for free: every chunk the far side
    /// writes arrives on the transport's pipes as it happens.
    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            Some(timeout),
            Some(on_chunk),
            None,
            None,
            None,
            None,
        )
        .await
    }

    /// As [`Self::run_streaming`], reporting a [`Kill`] that stops the command **on the far side**,
    /// not just the local transport — see `far_kill`. No deadline: that kill is how it ends early.
    /// What the command leaves running in its process group is reported to `on_orphans` and waited
    /// for, as a local shell's leftovers are (see `far_kill::WRAPPER`).
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            None,
            Some(hooks.on_chunk),
            None,
            None,
            Some(hooks.on_spawn),
            Some(hooks.on_orphans),
        )
        .await
    }

    async fn tracks_jobs(&self) -> bool {
        true
    }

    /// The transport's two pipes, kept apart. No deadline; dropping the future kills the far side
    /// (`TemplateKillGuard`), so the held start dies with it — as a v1.2 stream's does.
    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: ChunkSink<'_>,
        on_stderr: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            None,
            Some(on_stdout),
            Some(on_stderr),
            None,
            None,
            None,
        )
        .await
    }

    async fn attaches(&self) -> bool {
        true
    }

    fn sandbox_jobs(self: Arc<Self>) -> Option<Arc<dyn jobs::SandboxJobs>> {
        let probe = self.jobs_probe.clone();
        Some(Arc::new(jobs::ShellJobs::with_probe(self, probe)))
    }

    /// Piped into the *transport* process. Whether it reaches the far side is the transport's
    /// business: `ssh` forwards stdin, but `docker exec`/`kubectl exec` do so only with `-i`. Without
    /// it the far command reads nothing, which `ShellFs`'s probe detects — writes then use argv-sized
    /// chunks instead.
    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            Some(timeout),
            None,
            None,
            Some(stdin),
            None,
            None,
        )
        .await
    }
}

/// How long the far-side kill may take before it is abandoned: one more round trip through the
/// transport, which for `docker exec` is a fraction of a second.
const FAR_KILL_LIMIT: Duration = Duration::from_secs(10);

/// Stops a template command wherever it is: on the far side through the transport (once its
/// marker has arrived), then the local transport's own process group. Blocking.
#[derive(Clone)]
struct TemplateKill {
    template: Arc<Vec<String>>,
    far: far_kill::FarIdCell,
    local_group: Option<crate::tools::exec::GroupId>,
}

impl TemplateKill {
    fn kill(&self) {
        let far = *self.far.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(id) = far {
            let (program, args) = far_kill::far_kill_argv(id);
            far_kill::run_bounded(&expand(&self.template, &program, &args), FAR_KILL_LIMIT);
        }
        #[cfg(unix)]
        if let Some(group) = self.local_group {
            crate::tools::exec::kill_process_group(group);
        }
    }
}

/// Runs a [`TemplateKill`] if the command's future is dropped before the command exits — a
/// cancelled call, an aborted job. Disarmed once the transport has exited on its own.
struct TemplateKillGuard(Option<TemplateKill>);

impl Drop for TemplateKillGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(kill) = self.0.take() {
            crate::tools::exec::spawn_tracked_cleanup(move || kill.kill());
        }
    }
}

/// `{}` replaced by the command and its arguments, as argv entries.
fn expand(template: &[String], program: &str, args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(template.len() + args.len());
    for part in template {
        if part == "{}" {
            out.push(program.to_string());
            out.extend(args.iter().cloned());
        } else {
            out.push(part.clone());
        }
    }
    out
}

impl TemplateRunner {
    #[allow(clippy::too_many_arguments)]
    async fn exec(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Option<Duration>,
        on_chunk: Option<ChunkSink<'_>>,
        on_stderr: Option<ChunkSink<'_>>,
        stdin: Option<&[u8]>,
        on_spawn: Option<SpawnSink<'_>>,
        on_orphans: Option<OrphanSink<'_>>,
    ) -> std::io::Result<ExecResult> {
        // `cwd` is expressed by wrapping the command, not by setting the *local* process's directory —
        // the directory belongs to the far side. Positional parameters to a fixed script, never
        // substituted text.
        //
        // Everything but a stdin-carrying call runs under `far_kill::WRAPPER`, which makes the
        // command killable on the far side (killing the transport does not reach it — see
        // `far_kill`). A stdin call cannot: the wrapper backgrounds the command, and a background
        // job's stdin is `/dev/null`.
        let killable = stdin.is_none();
        let (program, args): (String, Vec<String>) = if killable {
            // A tracked job lingers on the far side while its leftovers run; any other call ends
            // with its command, as before.
            let linger = if on_orphans.is_some() { "1" } else { "0" };
            let mut v = vec![
                "-c".to_string(),
                far_kill::WRAPPER.to_string(),
                "sh".to_string(),
                linger.to_string(),
                cwd.unwrap_or("").to_string(),
                program.to_string(),
            ];
            v.extend(args.iter().cloned());
            ("sh".to_string(), v)
        } else {
            match cwd {
                Some(dir) => {
                    let mut v = vec![
                        "-c".to_string(),
                        r#"cd "$1" || exit 1; shift; exec "$@""#.to_string(),
                        "sh".to_string(),
                        dir.to_string(),
                        program.to_string(),
                    ];
                    v.extend(args.iter().cloned());
                    ("sh".to_string(), v)
                }
                None => (program.to_string(), args.to_vec()),
            }
        };
        let argv = self.expand(&program, &args);

        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Its own group, so the local half of a kill reaches whatever helpers the transport forks.
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd.spawn()?;
        let far = far_kill::FarIdCell::default();
        let kill = TemplateKill {
            template: Arc::new(self.template.clone()),
            far: far.clone(),
            local_group: child.id().map(crate::tools::exec::GroupId::leader),
        };
        let mut guard = TemplateKillGuard(Some(kill.clone()));
        if let Some(report) = on_spawn {
            let kill = kill.clone();
            report(Kill::Hook(Arc::new(move || {
                kill.kill();
                None
            })));
        }
        let stdout = child.stdout.take();
        // The wrapper's shell-exit marker, heard on stderr: the command exited leaving processes of
        // its group running.
        let (shell_exit_tx, mut shell_exit_rx) = tokio::sync::watch::channel(None::<i32>);
        let stderr = child.stderr.take().map(|e| {
            let on_shell_exit = on_orphans.map(|_| -> far_kill::ShellExitSink<'_> {
                Box::new(move |code| {
                    let _ = shell_exit_tx.send(Some(code));
                })
            });
            far_kill::MarkerStrip::new(e, far.clone(), on_shell_exit)
        });
        // Fed concurrently with the drains, never before them — see `feed_stdin`.
        let feed = feed_stdin(child.stdin.take(), stdin);
        // `drain_capped` is the same streaming head+tail accumulator the local runner uses: memory
        // is bounded at `STREAM_HEAD + STREAM_TAIL` per stream regardless of how much arrives, and
        // it says honestly whether a middle was dropped (`wait_with_output`, used here once, buffered
        // everything and reported `truncated: false` regardless). When streaming, every chunk goes to
        // the sink and nothing is captured, exactly as `RealRunner` does. Not
        // `HttpExecRunner::read_capped` — that *discards* an over-cap body and errors, which is right
        // for a response envelope that must parse as a whole, and wrong here: a long-running
        // command's output is still the answer.
        //
        // Both pipes are drained *concurrently* with the wait. A child that fills one pipe's OS
        // buffer while we read only the other deadlocks, and an unread pipe stalls its exit.
        let (exited_tx, exited_rx) = tokio::sync::watch::channel(false);
        let wait = async {
            let status = child.wait().await;
            let _ = exited_tx.send(true);
            status
        };
        // Report the leftovers once the marker arrives, after the same 100 ms the local runner gives
        // output already in flight on the other pipe (`bash_proc::SETTLE`), so what the shell
        // printed lands in the call's result, not the job's.
        let orphans = async {
            let Some(sink) = on_orphans else {
                return;
            };
            let mut exited = exited_rx.clone();
            tokio::select! {
                // Each guard is dropped inside its own arm: neither may live across the sleep.
                code = async { shell_exit_rx.wait_for(Option::is_some).await.ok().and_then(|c| *c) } => {
                    tokio::time::sleep(SHELL_EXIT_SETTLE).await;
                    let shell = ExecResult {
                        code,
                        ..Default::default()
                    };
                    sink(&shell, REMOTE_LEFTOVERS.to_string());
                }
                () = async { let _ = exited.wait_for(|e| *e).await; } => {}
            }
        };
        let collect = async {
            tokio::join!(
                wait,
                drain_capped(
                    stdout,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_chunk,
                    exited_rx.clone()
                ),
                drain_capped(
                    stderr,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_stderr.or(on_chunk),
                    exited_rx.clone()
                ),
                feed,
                orphans,
            )
        };
        // `None`: a tracked job, whose kill is how it ends early.
        let collected = match timeout {
            Some(limit) => tokio::time::timeout(limit, collect).await.ok(),
            None => Some(collect.await),
        };
        match collected {
            Some((status, (stdout, out_truncated), (stderr, err_truncated), fed, ())) => {
                guard.0 = None;
                let status = status?;
                fed?;
                Ok(ExecResult {
                    code: status.code(),
                    // The transport's own exit signal, not the far command's — `ssh` reports a
                    // remote signal as an exit code, and a local `docker exec` killed by one says
                    // nothing about what ran inside. Left `None` rather than guessing.
                    signal: None,
                    stdout: String::from_utf8_lossy(&stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&stderr).into_owned(),
                    timed_out: false,
                    truncated: out_truncated || err_truncated,
                })
            }
            None => {
                // Reaping the transport is not enough: the far command outlives it. Kill both and
                // wait for it, so a caller that re-checks side effects sees the command gone.
                if let Some(kill) = guard.0.take() {
                    let _ = tokio::task::spawn_blocking(move || kill.kill()).await;
                }
                Ok(ExecResult {
                    code: None,
                    signal: None,
                    stdout: String::new(),
                    stderr: format!(
                        "exec template timed out after {:?}",
                        timeout.unwrap_or_default()
                    ),
                    timed_out: true,
                    truncated: false,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

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

    #[test]
    fn a_template_needs_exactly_one_placeholder() {
        assert!(TemplateRunner::parse("ssh host -- {}").is_ok());
        assert!(TemplateRunner::parse("ssh host").is_err());
        assert!(TemplateRunner::parse("ssh {} {}").is_err());
        assert!(TemplateRunner::parse("{} foo").is_err());
    }

    #[test]
    fn the_placeholder_expands_to_argv_entries_not_a_string() {
        // The injection invariant. A path containing `;` and quotes must stay ONE argument on the far
        // side; if this ever renders to a string, the far side reparses it as syntax.
        let r = TemplateRunner::parse("docker exec ctr {}").unwrap();
        let argv = r.expand(
            "grep",
            &["-e".into(), "'; rm -rf / #".into(), "/a b.txt".into()],
        );
        assert_eq!(
            argv,
            vec![
                "docker",
                "exec",
                "ctr",
                "grep",
                "-e",
                "'; rm -rf / #",
                "/a b.txt"
            ]
        );
    }

    #[tokio::test]
    async fn a_template_runner_actually_runs_the_command() {
        // `env` is a harmless stand-in for a real transport: it runs whatever follows, which is
        // exactly the shape `ssh host --` / `docker exec ctr` have.
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run(
                "printf",
                &["%s".into(), "hello".into()],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(out.code, Some(0));
        assert_eq!(out.stdout, "hello");
    }

    #[tokio::test]
    async fn a_template_runner_honors_cwd_without_splicing_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run("ls", &[], dir.path().to_str(), Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(out.code, Some(0));
        assert!(out.stdout.contains("marker.txt"), "{out:?}");
    }

    /// The transport's output is bounded by the capture window and truncation is reported honestly.
    /// `wait_with_output` buffered the whole thing and hardcoded `truncated: false`, so a command
    /// that printed a gigabyte sized this process — and said nothing about what it had lost.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_template_runner_caps_its_output_and_says_when_it_truncated() {
        let r = TemplateRunner::parse("env {}").unwrap();
        let bytes = 2 * (STREAM_HEAD + STREAM_TAIL);
        let out = r
            .run(
                "sh",
                &[
                    "-c".into(),
                    format!("head -c {bytes} /dev/zero | tr '\\0' 'a'"),
                ],
                None,
                Duration::from_secs(30),
            )
            .await
            .unwrap();

        assert_eq!(out.code, Some(0), "{out:?}");
        assert!(out.truncated, "the middle was dropped, so say so: {out:?}");
        assert!(
            out.stdout.len() <= STREAM_HEAD + STREAM_TAIL,
            "kept {} bytes of a {bytes}-byte stream",
            out.stdout.len()
        );
    }

    /// …and output that fits is returned whole, still flagged untruncated.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_template_runner_leaves_output_under_the_cap_alone() {
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run(
                "sh",
                &[
                    "-c".into(),
                    "printf 'hello\\n'; printf 'oops\\n' >&2".into(),
                ],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(out.stdout, "hello\n");
        assert_eq!(out.stderr, "oops\n");
        assert!(!out.truncated, "{out:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Per-session targets, for a multi-tenant server.
// ---------------------------------------------------------------------------------------------

/// One configured target: the runner, and the filesystem backend built over it.
///
/// They are constructed together and handed out together because they must always name the *same*
/// machine — a `bash` on one and an `edit` on another is not a partial configuration, it is a bug.
#[derive(Clone)]
pub struct ExecTarget {
    runner: std::sync::Arc<dyn CommandRunner>,
    backend: std::sync::Arc<dyn crate::tools::fs::FsBackend>,
    caps: crate::tools::fs::shell::Capabilities,
    /// Shared with the runner ([`HttpExecRunner::with_suspect_flag`]): raised by a request the
    /// endpoint did not answer or refused. A lazily attached [`ExecCell`] then probes again before
    /// its next call instead of trusting what this target learned.
    suspect: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl ExecTarget {
    /// Build from an HTTP endpoint plus `Name: value` headers, reading at most `max_response_bytes` of
    /// any one response ([`DEFAULT_MAX_RESPONSE_BYTES`] unless configured).
    pub async fn http(
        url: &str,
        headers: &[String],
        max_response_bytes: usize,
    ) -> Result<Self, String> {
        let mut runner = HttpExecRunner::new(url)?.with_max_response_bytes(max_response_bytes);
        for raw in headers {
            let (name, value) = HttpExecRunner::parse_header(raw)?;
            runner = runner.with_header(name, value);
        }
        Ok(Self::over(std::sync::Arc::new(runner)).await)
    }

    /// Build from an argv template (`ssh host -- {}`, `docker exec ctr {}`, …).
    pub async fn template(template: &str) -> Result<Self, String> {
        Ok(Self::over(std::sync::Arc::new(TemplateRunner::parse(template)?)).await)
    }

    /// Wrap an already-built runner, probing it for the capability rung.
    pub async fn over(runner: std::sync::Arc<dyn CommandRunner>) -> Self {
        Self::over_with_home(runner, None).await
    }

    /// [`Self::over`], told the target's `$HOME` up front so a model-supplied `~/notes.md` expands
    /// against the sandbox's own home instead of being left alone. Service mode learns the home from
    /// its probe (`crate::service::ServiceSession::connect_exec`), which runs before the first tool
    /// call is dispatched, and is the only caller that can know it then.
    pub async fn over_with_home(
        runner: std::sync::Arc<dyn CommandRunner>,
        home: Option<String>,
    ) -> Self {
        let backend = crate::tools::fs::shell::ShellFs::connect(runner.clone()).await;
        let backend = match home {
            Some(home) => backend.with_home(home),
            None => backend,
        };
        let caps = backend.capabilities();
        Self {
            runner,
            backend: std::sync::Arc::new(backend),
            caps,
            suspect: None,
        }
    }

    /// Watch `flag` — the one its runner raises on a failed request — so a cell holding this target
    /// knows to probe the endpoint again ([`ExecCell::set_lazy`]).
    pub fn watching(mut self, flag: Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.suspect = Some(flag);
        self
    }

    /// Whether a request through this target failed in a way that suggests the sandbox behind it
    /// was replaced (unreachable, or refused).
    fn suspect(&self) -> bool {
        self.suspect
            .as_ref()
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        self.runner.clone()
    }

    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        self.backend.clone()
    }

    /// Which search rung this target resolved to — reported to the client so a degraded box (no
    /// `rg`, or busybox) is visible at attach rather than inferred from bad results later.
    pub fn search_engine(&self) -> crate::tools::fs::shell::SearchEngine {
        self.caps.search_engine()
    }
}

/// A **per-session** exec target that can be re-pointed at runtime.
///
/// This exists because of a specific hazard. `serve`'s tool registry is rebuilt only when the model
/// or thinking level changes — a `switch_session` between two sessions on the same model does *not*
/// rebuild it. So a target bound at registry-construction time would leave the new session talking to
/// the previous session's machine. In a multi-tenant server that is not a stale-config bug, it is one
/// tenant's tools operating inside another tenant's sandbox.
///
/// The cell closes that by construction: the tools hold *this*, and read it on every call, so a
/// re-point takes effect immediately and cannot be missed by a skipped rebuild. It is the same shape
/// `memory::file::SessionDir` already uses for the per-session `/session` mount, for the same reason.
///
/// An empty cell means the host, so one registry serves both local and remote sessions — unless the
/// cell is [strict](Self::strict), in which case an empty cell is an error and there is no host to
/// fall back to.
#[derive(Clone, Default)]
pub struct ExecCell {
    route: CellRoute,
    /// The session's `bash` jobs. Here because they share the cell's lifetime and reach: one per
    /// session, handed to every registry rebuild (`build_tools` takes the cell), so a `set_model`
    /// keeps them running.
    ///
    /// Deliberately *not* in [`CellRoute`], the part [`Self::runner`] and [`Self::backend`] clone:
    /// every running job's task holds the runner, so a runner that held the table would make a
    /// cycle (task → runner → table → job → task), and a session that ended without stopping its
    /// jobs would leak them, unreachable, for the life of the process.
    bash_jobs: std::sync::Arc<crate::tools::bash_jobs::BashJobs>,
}

/// Where an [`ExecCell`] dispatches: the part its runner and filesystem backend share.
#[derive(Clone, Default)]
struct CellRoute {
    target: std::sync::Arc<std::sync::RwLock<Option<ExecTarget>>>,
    /// How to build the target on first use, when it was attached lazily ([`ExecCell::set_lazy`]).
    /// Kept after the first connect, so a target that turns [suspect](ExecTarget::suspect) is
    /// built again — probe included — rather than trusted.
    lazy: std::sync::Arc<std::sync::RwLock<Option<std::sync::Arc<Lazy>>>>,
    /// Fail closed: an empty cell errors instead of running on this host. Service mode
    /// (`serve --service`) sets it, because there the host is the *replica* — a tenant's `bash`
    /// landing on it is not a stale-config bug but a breach. Fixed at construction.
    strict: bool,
}

/// Builds a lazily attached target: the probe and everything it derives, run on first use. Its
/// error reaches the tool call that needed the target, word for word.
pub type Connect = std::sync::Arc<
    dyn Fn() -> futures::future::BoxFuture<'static, Result<ExecTarget, String>> + Send + Sync,
>;

/// A lazily attached target, not built yet (or to be built again).
struct Lazy {
    connect: Connect,
    /// The target's `$HOME` as last seen, so a path with a `~` resolves the same way before the
    /// probe as after it ([`CellFs::world`] is not async and cannot wait for one).
    home: Option<String>,
    /// Single flight: concurrent first calls wait on one connect rather than each probing.
    gate: tokio::sync::Mutex<()>,
}

/// What a cell dispatches to right now.
enum Route {
    Target(ExecTarget),
    /// Nothing configured, and the cell is not strict: this host.
    Host,
    /// A lazily attached target not built yet (or suspect): connect before dispatching.
    Pending(std::sync::Arc<Lazy>),
    /// Nothing configured, and the cell is strict.
    Refused,
}

impl CellRoute {
    fn target(&self) -> Option<ExecTarget> {
        self.target
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn lazy(&self) -> Option<std::sync::Arc<Lazy>> {
        self.lazy.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Where a call goes, without waiting for anything. A built target wins unless it is suspect
    /// and there is a way to build it again.
    fn route(&self) -> Route {
        match (self.target(), self.lazy()) {
            (Some(t), Some(lazy)) if t.suspect() => Route::Pending(lazy),
            (Some(t), _) => Route::Target(t),
            (None, Some(lazy)) => Route::Pending(lazy),
            (None, None) if self.strict => Route::Refused,
            (None, None) => Route::Host,
        }
    }

    /// The target to dispatch to — connecting a lazily attached one first, once for every caller
    /// waiting on it — or `None` for this host, or the refusal.
    async fn resolve(&self) -> Result<Option<ExecTarget>, String> {
        match self.route() {
            Route::Target(t) => Ok(Some(t)),
            Route::Host => Ok(None),
            Route::Refused => Err(NO_TARGET.to_string()),
            Route::Pending(lazy) => self.connect(&lazy).await.map(Some),
        }
    }

    async fn connect(&self, lazy: &std::sync::Arc<Lazy>) -> Result<ExecTarget, String> {
        let _flight = lazy.gate.lock().await;
        // Whoever held the gate before may have connected already.
        if let Route::Target(t) = self.route() {
            return Ok(t);
        }
        let target = (lazy.connect)().await?;
        // Only if this is still the cell's connector: a re-point while the probe ran wins.
        if self
            .lazy()
            .is_some_and(|c| std::sync::Arc::ptr_eq(&c, lazy))
        {
            *self.target.write().unwrap_or_else(|e| e.into_inner()) = Some(target.clone());
        }
        Ok(target)
    }
}

/// What a strict cell answers when nothing is configured. Static text: it reaches the model as a
/// tool error, so it says what to do, not what went wrong internally.
const NO_TARGET: &str = "no sandbox is attached to this session";

impl ExecCell {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cell with **no host fallback**: until a target is set, every command and every filesystem
    /// call fails, rather than running on this host. Service mode (`serve --service`) uses this,
    /// because there "this host" is the *replica* — a tenant's `bash` landing on it is not a
    /// stale-config bug but a breach.
    pub fn strict() -> Self {
        Self {
            route: CellRoute {
                strict: true,
                ..CellRoute::default()
            },
            ..Self::default()
        }
    }

    /// The session's `bash` job table — see [`crate::tools::bash_jobs`].
    pub fn bash_jobs(&self) -> &std::sync::Arc<crate::tools::bash_jobs::BashJobs> {
        &self.bash_jobs
    }

    /// Re-point at `target`, or back to the host with `None`. Takes effect on the next tool call.
    /// Replaces a lazily attached target too.
    pub fn set(&self, target: Option<ExecTarget>) {
        // Recover a poisoned lock rather than panicking: the cell holds a single value with no
        // invariant a panicked writer could leave half-updated, and the workspace forbids `unwrap`.
        *self.route.lazy.write().unwrap_or_else(|e| e.into_inner()) = None;
        *self.route.target.write().unwrap_or_else(|e| e.into_inner()) = target;
    }

    /// Attach a target **lazily**: nothing is sent until a call needs it, and then `connect` runs
    /// once — concurrent first calls share it — before that call is dispatched. A failure is that
    /// call's error and is not remembered, so the next call tries again. `connect` stays with the
    /// cell: a target whose requests start failing as though its sandbox was replaced (unreachable,
    /// or refused — see [`ExecTarget::watching`]) is built again on the call after.
    ///
    /// `home` is the target's `$HOME` as last seen, answered by the filesystem backend's
    /// [`world`](crate::tools::fs::FsBackend::world) until the probe has run.
    pub fn set_lazy(&self, connect: Connect, home: Option<String>) {
        let lazy = std::sync::Arc::new(Lazy {
            connect,
            home,
            gate: tokio::sync::Mutex::new(()),
        });
        *self.route.target.write().unwrap_or_else(|e| e.into_inner()) = None;
        *self.route.lazy.write().unwrap_or_else(|e| e.into_inner()) = Some(lazy);
    }

    /// Connect a lazily attached target now, as a call would. A no-op for any other cell.
    pub async fn connect(&self) -> Result<(), String> {
        self.route.resolve().await.map(|_| ())
    }

    /// Whether a lazily attached target is still to be built: nothing has needed it yet, or the
    /// one built turned suspect. A caller that must not wake the target checks this first.
    pub fn pending(&self) -> bool {
        matches!(self.route.route(), Route::Pending(_))
    }

    /// The target, if one is built and usable. Never connects.
    pub fn get(&self) -> Option<ExecTarget> {
        match self.route.route() {
            Route::Target(t) => Some(t),
            _ => None,
        }
    }

    /// A [`CommandRunner`] that always dispatches to whatever this cell currently holds.
    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        std::sync::Arc::new(CellRunner(self.route.clone()))
    }

    /// An [`FsBackend`](crate::tools::fs::FsBackend) that always dispatches to whatever this cell
    /// currently holds.
    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        std::sync::Arc::new(CellFs(self.route.clone()))
    }
}

/// Dispatches each command to the cell's current target, or this host when it is empty.
struct CellRunner(CellRoute);

impl CellRunner {
    async fn target(&self) -> std::io::Result<Option<ExecTarget>> {
        self.0.resolve().await.map_err(std::io::Error::other)
    }
}

#[async_trait]
impl CommandRunner for CellRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => t.runner().run(program, args, cwd, timeout).await,
            None => {
                crate::tools::exec::RealRunner
                    .run(program, args, cwd, timeout)
                    .await
            }
        }
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
        }
    }

    /// Forwarded, not left to the default: without it a `bash` job behind a cell — every `serve`
    /// session's — reported no [`Kill`], so the exit path could not reach it, a remote job never
    /// yielded, and on this host a leftover `cmd &` went untracked.
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &crate::tools::exec::JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_streaming_tracked(program, args, cwd, hooks)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_streaming_tracked(program, args, cwd, hooks)
                    .await
            }
        }
    }

    /// Forwarded for the same reason: the target behind the cell decides whether `bash` can run a
    /// command in the background. A strict cell with no target answers `false`; running anything
    /// then fails with the cell's own refusal.
    async fn tracks_jobs(&self) -> bool {
        match self.0.resolve().await {
            Ok(Some(t)) => t.runner().tracks_jobs().await,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: ChunkSink<'_>,
        on_stderr: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_attached(program, args, cwd, on_stdout, on_stderr)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_attached(program, args, cwd, on_stdout, on_stderr)
                    .await
            }
        }
    }

    async fn attaches(&self) -> bool {
        match self.0.resolve().await {
            Ok(Some(t)) => t.runner().attaches().await,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    /// Connects a lazily attached target, so the synchronous [`Self::sandbox_jobs`] after it sees
    /// one. Its error is the probe's, for the call that needed it.
    async fn ready(&self) -> std::io::Result<()> {
        self.target().await.map(|_| ())
    }

    /// The jobs of the sandbox the cell points at **now**. A job keeps the store it started with,
    /// so a later re-point never sends its kill to another machine. `None` while a lazily attached
    /// target is unbuilt: [`Self::ready`] first.
    fn sandbox_jobs(self: Arc<Self>) -> Option<Arc<dyn jobs::SandboxJobs>> {
        match self.0.route() {
            Route::Target(t) => t.runner().sandbox_jobs(),
            _ => None,
        }
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
        }
    }
}

/// The filesystem half of the same indirection.
struct CellFs(CellRoute);

impl CellFs {
    async fn inner(
        &self,
    ) -> Result<std::sync::Arc<dyn crate::tools::fs::FsBackend>, crate::tools::fs::FsError> {
        match self.0.resolve().await {
            Ok(Some(t)) => Ok(t.backend()),
            Ok(None) => Ok(std::sync::Arc::new(crate::tools::fs::local::LocalFs::new())),
            Err(e) => Err(crate::tools::fs::FsError::Backend(e)),
        }
    }
}

#[async_trait]
impl crate::tools::fs::FsBackend for CellFs {
    /// Not fallible, so a strict empty cell answers with the world it *will* have: remote, with no
    /// home known yet. Never `Local` — a tenant's path must not resolve against the replica even in
    /// the window before its sandbox is attached. A lazily attached target that has not been built
    /// answers with the home it was last seen with, so `~` means the same before the probe as after.
    fn world(&self) -> crate::tools::fs::PathWorld {
        match self.0.route() {
            Route::Target(t) => t.backend().world(),
            Route::Host => crate::tools::fs::PathWorld::Local,
            Route::Pending(lazy) => crate::tools::fs::PathWorld::Remote {
                home: lazy.home.clone(),
            },
            Route::Refused => crate::tools::fs::PathWorld::Remote { home: None },
        }
    }
    async fn search(
        &self,
        q: &crate::tools::fs::SearchQuery,
    ) -> Result<crate::tools::fs::SearchOutcome, crate::tools::fs::FsError> {
        self.inner().await?.search(q).await
    }
    async fn stat(
        &self,
        path: &std::path::Path,
    ) -> Result<Option<crate::tools::fs::Meta>, crate::tools::fs::FsError> {
        self.inner().await?.stat(path).await
    }
    async fn read_bytes(
        &self,
        path: &std::path::Path,
        offset: u64,
        max: usize,
    ) -> Result<Vec<u8>, crate::tools::fs::FsError> {
        self.inner().await?.read_bytes(path, offset, max).await
    }
    async fn write_bytes(
        &self,
        path: &std::path::Path,
        bytes: Vec<u8>,
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner().await?.write_bytes(path, bytes).await
    }
    async fn write_if_unchanged(
        &self,
        path: &std::path::Path,
        bytes: Vec<u8>,
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, crate::tools::fs::FsError> {
        self.inner()
            .await?
            .write_if_unchanged(path, bytes, expected)
            .await
    }
    async fn create_dir_all(
        &self,
        path: &std::path::Path,
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner().await?.create_dir_all(path).await
    }
    async fn list_dir(
        &self,
        path: &std::path::Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<crate::tools::fs::DirEntry>, crate::tools::fs::FsError> {
        self.inner()
            .await?
            .list_dir(path, cap, include_hidden)
            .await
    }
    async fn glob(
        &self,
        q: &crate::tools::fs::GlobQuery,
    ) -> Result<crate::tools::fs::GlobOutcome, crate::tools::fs::FsError> {
        self.inner().await?.glob(q).await
    }
}

#[cfg(test)]
mod lazy_cell_tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::tools::fs::PathWorld;

    /// What a counting connector shares with its test.
    #[derive(Default)]
    struct Probe {
        runs: AtomicUsize,
        fail: AtomicBool,
        /// Each built target's suspect flag, in build order.
        flags: Mutex<Vec<Arc<AtomicBool>>>,
    }

    /// A connector that counts its runs, takes a while (so first callers overlap), fails while
    /// `fail` is set, and builds a target over this host whose home is `/sandbox/home`.
    fn counting(probe: Arc<Probe>) -> Connect {
        Arc::new(move || {
            let probe = probe.clone();
            Box::pin(async move {
                probe.runs.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                if probe.fail.load(Ordering::SeqCst) {
                    return Err("the sandbox did not answer".to_string());
                }
                let flag = Arc::new(AtomicBool::new(false));
                probe.flags.lock().unwrap().push(flag.clone());
                let runner: Arc<dyn CommandRunner> =
                    Arc::new(TemplateRunner::parse("env {}").map_err(|e| e.to_string())?);
                Ok(
                    ExecTarget::over_with_home(runner, Some("/sandbox/home".into()))
                        .await
                        .watching(flag),
                )
            })
        })
    }

    async fn run_true(cell: &ExecCell) -> std::io::Result<Option<i32>> {
        cell.runner()
            .run("true", &[], None, Duration::from_secs(10))
            .await
            .map(|r| r.code)
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_lazy_cell_sends_nothing_until_used_and_concurrent_first_calls_share_one_connect() {
        let probe = Arc::new(Probe::default());
        let cell = ExecCell::strict();
        cell.set_lazy(counting(probe.clone()), Some("/seen/home".into()));
        assert!(cell.pending());
        // The home it was last seen with answers `world` before any probe.
        assert_eq!(
            cell.backend().world(),
            PathWorld::Remote {
                home: Some("/seen/home".into())
            }
        );
        assert_eq!(
            probe.runs.load(Ordering::SeqCst),
            0,
            "attaching sent nothing"
        );

        let calls = (0..8).map(|_| {
            let cell = cell.clone();
            tokio::spawn(async move { run_true(&cell).await })
        });
        for call in futures::future::join_all(calls).await {
            assert_eq!(call.unwrap().unwrap(), Some(0));
        }
        assert_eq!(
            probe.runs.load(Ordering::SeqCst),
            1,
            "one connect for eight first calls"
        );
        assert!(!cell.pending());
        assert_eq!(
            cell.backend().world(),
            PathWorld::Remote {
                home: Some("/sandbox/home".into())
            },
            "the probe's answer replaces the one last seen"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_failed_connect_is_that_calls_error_and_the_next_call_tries_again() {
        let probe = Arc::new(Probe::default());
        probe.fail.store(true, Ordering::SeqCst);
        let cell = ExecCell::strict();
        cell.set_lazy(counting(probe.clone()), None);

        let err = run_true(&cell).await.unwrap_err();
        assert!(
            err.to_string().contains("the sandbox did not answer"),
            "{err}"
        );
        assert!(
            cell.backend()
                .stat(std::path::Path::new("/"))
                .await
                .is_err(),
            "never the host: the filesystem half waits on the same connect"
        );
        assert!(cell.pending(), "a failure is not remembered");

        probe.fail.store(false, Ordering::SeqCst);
        assert_eq!(run_true(&cell).await.unwrap(), Some(0));
        assert_eq!(probe.runs.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_suspect_target_is_probed_again_and_an_explicit_target_ends_the_laziness() {
        let probe = Arc::new(Probe::default());
        let cell = ExecCell::strict();
        cell.set_lazy(counting(probe.clone()), None);
        cell.connect().await.unwrap();
        cell.connect().await.unwrap();
        assert_eq!(
            probe.runs.load(Ordering::SeqCst),
            1,
            "a good target is kept"
        );

        // A request was refused or never answered: the sandbox may have been replaced.
        probe.flags.lock().unwrap()[0].store(true, Ordering::SeqCst);
        assert!(cell.pending());
        assert_eq!(run_true(&cell).await.unwrap(), Some(0));
        assert_eq!(probe.runs.load(Ordering::SeqCst), 2, "probed again");
        assert!(!cell.pending());

        // An explicit target (here: none, on a strict cell) replaces the lazy one outright.
        cell.set(None);
        assert!(!cell.pending());
        assert!(run_true(&cell).await.is_err(), "strict and empty: refused");
        assert_eq!(probe.runs.load(Ordering::SeqCst), 2);
    }
}
