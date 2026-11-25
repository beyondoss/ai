# beyond-job-client

The job client's kernel, in two parts.

- The job cursor (`src/lib.rs`): one job incarnation (scope, id, mark) at one output offset, as a string a
  caller keeps and hands back to `resume`. A cursor names exactly one incarnation at exactly one offset, so a
  resume never reads another incarnation's bytes (a reused id has another mark) and never re-reads or skips a
  byte (the offset is 16 fixed hex digits, so a cursor cut short cannot decode to a smaller one).
- The step function (`src/step.rs`): everything a handle decides while it holds a job, as
  `step(state, event, now, draw) -> (state, actions)`. It covers start, attach, reconnect, exit and remove, and
  the `wait` watch mode. See [The step function](#the-step-function).

No I/O, no clock, no randomness, no dependency but `vstd`. Every decision is written inside `verus!`, so the
function production runs is the one [Verus](https://github.com/verus-lang/verus) checks, with no bound on
anything. A normal build erases the specs and proofs; `mise run verify:kernels`
(`.github/scripts/verify-kernels.sh`, the release pinned in `.github/scripts/verus.env`, run by CI's `verus`
job) checks them.

```rust
let c = Cursor::from_parts("session-1", "a1b2c3", "00ff00ff00ff00ff1a", 4096)?;
let s: String = job_client::encode(&c);   // j1.session-1.a1b2c3.00ff00ff00ff00ff1a.0000000000001000.0a3f
let back = job_client::decode(&s)?;       // Ok(c); Err(CursorError) for anything else
```

## Format

The leading `j1` is the cursor format's own version, not a wire protocol version: a change to the format
bumps it and old cursors are refused as `BadVersion`.

```text
j1.<scope>.<id>.<mark>.<offset>.<check>
```

| Field  | Rule                                                                                | Taken from                                                                                                                |
| ------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `j1`   | the version                                                                         | this crate                                                                                                                |
| scope  | 1 to 64 bytes of `[0-9a-z-]`                                                        | `JobScope::named` (`crates/sandbox/src/exec_endpoint/jobs.rs`), the TypeScript scope check (`sdk/ts/sandbox/src/jobs.ts`) |
| id     | 1 to 64 bytes of `[0-9a-z_-]`                                                       | `JobId::parse` (same file), the TypeScript id check (same)                                                                |
| mark   | 1 to 32 bytes of `[0-9a-f]`                                                         | execd's `new_job_mark` (`crates/execd/src/proc.rs`): 16 hex, then the sequence in hex                                     |
| offset | 16 bytes of `[0-9a-f]`: the `u64`                                                   | fixed width                                                                                                               |
| check  | 4 bytes of `[0-9a-f]`: the sum of every byte before the `.` that precedes the check | an additive check                                                                                                         |

`.` is outside every alphabet, so a string has one parse and a cursor has one string. The check is for a
cursor cut or garbled in a database column or a log line, not for security (the endpoint enforces scope and
auth; a mark is unguessable). It is additive, not a CRC: a one-byte change moves the sum by a non-zero amount
below 256, which Verus proves with no bound. It does not detect two bytes swapped, or two changes that cancel
in the sum; the alphabets and the structure may still refuse those, and nothing is claimed.

## Proven

In `src/lib.rs`. `accepts(s)` says `s` is the encoding of some cursor whose fields obey the rules. Each
property has one `mut_*` feature that removes the guard it rests on and turns the proof red
(`mise run verify:kernels -- --features <mut>`).

| Property                                                                                                                                                                                                       | Where                                                                                     | Mutant                                                                          |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| C1 `decode(encode(c)) == Ok(c)` for every cursor: nothing valid is refused. `decode` is `Ok` exactly for the strings `encode` writes, and there is one each. This is the liveness (completeness) partner of C3 | `decode_bytes`'s ensures, `encode_fields`, `lemma_enc_facts`                              | `mut_rejects_max_offset`                                                        |
| C2 the offset is exact: all 64 bits survive, `u64::MAX` included                                                                                                                                               | `lemma_offset_hex_round_trip`, `lemma_hex_roundtrip`                                      | `mut_offset_truncated`                                                          |
| C3a changing any one byte of an accepted string leaves a string that is refused                                                                                                                                | `lemma_single_byte_change_refused`, `lemma_sum_one_diff`, `decode_bytes`'s C3a ensures    | `mut_check_skips_a_byte`                                                        |
| C3b a strict prefix of an accepted string is refused (a cursor cut short)                                                                                                                                      | `lemma_prefix_refused`, `decode_bytes`'s C3b ensures                                      | `mut_prefix_accepted`                                                           |
| C3c an accepted string with bytes after it is refused                                                                                                                                                          | `lemma_extension_refused`, `decode_bytes`'s C3c ensures                                   | `mut_trailing_accepted`                                                         |
| C4 a decoded or constructed cursor has a scope of 1 to 64 `[0-9a-z-]`, an id of 1 to 64 `[0-9a-z_-]` and a mark of 1 to 32 `[0-9a-f]`: the type invariant, whoever made the value                              | `Cursor::inv`, `Cursor::scope`, `Cursor::id`, `Cursor::mark`, `decode_bytes`'s C4 ensures | `mut_scope_unchecked`, `mut_id_unchecked`, `mut_mark_unchecked` (one per field) |

The mutants are the guards, not the lemmas: a mutant that makes `decode` accept what it must not breaks
`decode_bytes`'s own ensures (`Ok` iff `accepts`), and with it every C3 clause, which rests on that. Run
`mise run verify:kernels -- --features <mut> -- --multiple-errors 50` to see each failing clause named.
`mut_check_skipped` (decode takes any check) is red at the `accepts` link and also covers C1 and C3a.

C1 is the liveness (completeness) partner of the refusals (C3): a valid cursor is always accepted, so a retry
with the cursor a handle held never fails on the cursor. It is stated in `decode_bytes`'s ensures and needs no fairness assumption.
The properties are registered in Beyond's `proof/properties.toml` (C1 as `kind = "liveness"`, partner C3, `proof` the
`decode_bytes` completeness ensures) when Beyond consumes this crate; this crate carries no registry. `decode` and `encode` are total and take no step.

No `assume`, `admit` or `external_body`.

`mise run verify:job-client-mutants` runs every mutant in turn and fails if any verifies green (or fails to
build instead of failing Verus), printing each mutant's red line.

## What stays outside `verus!`

- `encode` and `decode` over `String` and `&str`: one-line wrappers that turn the proven byte functions'
  ASCII output into a `String` (no failure path) and hand `str::as_bytes` in.
- `Cursor`'s `Clone` (a clone of valid fields is valid), `Debug`, `PartialEq`, `Eq`, and the `*_str` accessors
  that reshape a field.
- The type invariant holds where a cursor is made (`Cursor::new`, `decode_bytes`): Verus checks it at both.

## The step function

`job_client::step::step` is the one place an SDK's handle decides anything. The shell around it (TypeScript
through WASM, Python through a native library, the Rust SDK) does I/O only: it sends the request an
[`Action`] names, turns what it saw into an [`Event`], and calls `step` again. It keeps no reconnect counter,
no status table, no offset sum and no silence constant. Nothing in it is a pointer: the state is a fixed
record of integers (a mark is 32 bytes and a length), events and actions are plain enums, and a job's output
bytes never cross the boundary (the shell keeps the frame it decoded and hands the consumer the `len` the
kernel names).

```text
Go -> Starting|Opening -> (2xx) Confirming -> (echo of the mark) Streaming -> (exit frame) ExitHeld
        ^        |  refused / cut                |  cut                         -> (consumer took it) Removing -> Done
        +-- Backoff <------------------------- +
```

- **Events:** `Go`, `Answered{status, retry_after_s}`, `Frame(Attached | Started | Output | Gap | Heartbeat |
  Exit | Bad)`, `Broke` (reset, silence, EOF, truncated line), `Tick`, `ExitTaken`, `Close`, `Deadline`. A stale event (a
  frame after the cut, a tick when not waiting) changes nothing and asks for nothing.
- **Actions:** `Start{key, silence_ms}`, `OpenAttach{offset, mark, silence_ms}`, `CloseStream`,
  `Deliver{at, len, dropped}`, `DeliverExit{code, signal, dropped}`, `Sleep{until_ms}`, `Remove{mark}`,
  `Reconnecting{attempts, since_ms}` (status only), `Fail{fault, status, offset}`, `Finish`, `Detached`. Every handle ends with exactly one of the last three.
- **Time and randomness are inputs:** `now` (the shell's clock) and `draw` (a uniform `u32` for the jitter).
- **A mark** is `Mark::from_bytes` of 1 to 32 bytes of `[0-9a-f]`, or the empty mark ("whatever holds the id
  now", for a start or a `get`).
- **The idempotency key** is a handle: the shell holds the key itself; the kernel only guarantees every send of
  the start carries the one it was given.

Fail closed: until the server sends the `mark` on `attached` and `started`, the kernel answers a held stream
with `NotOffered`, never a guess. There is no version op: the echo is the feature detection.

### Properties

Each is an `ensures` of `step` (a `post_*` spec, all conjoined in `step_post`), proven with no bound, and has
`mut_*` mutants that turn the proof red. `mise run verify:job-client-mutants` runs every one.

| Property                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Where                                                    | Mutants                                                                                                                         |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **O** Offsets. Each delivery starts exactly where the last ended (`at - dropped` is the end of the previous), and the position never moves back; an addition that would overflow fails, it never wraps. Every reopen asks from the held offset for the held mark.                                                                                                                                                                                                                                                                              | `post_offsets`                                           | `mut_reattach_from_zero`, `mut_offset_wraps`                                                                                    |
| **G** Gaps fold into chunks. A gap frame is held and told with the next chunk (`dropped`); a gap with no chunk before the exit is delivered with the exit; none is told twice, none is dropped, and a delivery leaves nothing pending.                                                                                                                                                                                                                                                                                                         | `post_gaps`                                              | `mut_gap_not_counted`, `mut_trailing_gap_dropped`                                                                               |
| **A** Only these end a handle: the job's end (exit taken, remove answered), `JobGone` or a mark mismatch, a terminal class (auth, not-offered, protocol, 500, `NoPty` on a start, exists, disk full), the caller's `Close` or `Deadline`. A cut, silence, a tick or a retryable answer never do, however many failed opens came before: reconnecting is a status, and there is no exhaustion state. The deadline is an event the shell sends; the kernel has no clock of its own.                                                              | `post_only_these_end`, `post_end`, `post_classify`       | `mut_gives_up_on_budget`, `mut_start_lost_gives_up`                                                                             |
| **B** The failed-open count resets on a confirmed reattach only (the echo of the held mark), never on a bare connect. After `RECONNECTING_AFTER` (12) failed opens in a row every further one also emits `Reconnecting{attempts, since_ms}` (a status for the SDK to surface) and the handle keeps retrying, with waits capped at 30 s. A quiet healthy job that is cut again and again is never ended.                                                                                                                                        | `post_budget`                                            | `mut_budget_reset_on_connect`, `mut_budget_not_reset_on_echo`                                                                   |
| **M** Incarnation guard. Output is delivered only after the echo; an `attached` or `started` whose mark is not the held one fails `JobGone` before any byte; no echo, or an empty one, is `NotOffered`; the handle only enters `Streaming` through a matching echo of the right kind.                                                                                                                                                                                                                                                          | `post_incarnation`                                       | `mut_echo_mark_unchecked`, `mut_echo_skipped`                                                                                   |
| **X** Exit. The exit is delivered once, by the stream's final frame; nothing after it reopens the stream, delivers, or leaves the exit-then-remove phases.                                                                                                                                                                                                                                                                                                                                                                                     | `post_exit`                                              | `mut_reattach_after_exit`                                                                                                       |
| **R** Remove. A remove is sent only after the consumer took the exit, only if the policy asked, scoped to the held mark. Once taken, it is sent (liveness partner: a handle at `ExitHeld` that is taken moves on, and a `2xx` answer finishes it).                                                                                                                                                                                                                                                                                             | `post_remove`, `post_classify`                           | `mut_remove_on_exit_frame`, `mut_remove_unscoped`, `mut_exit_taken_stalls`                                                      |
| **C** Classification, one table: any 2xx goes on; `404` on an attach is `JobGone`; `409`/`507` on a start are `JobExists`/`DiskFull`; `503` on a start (no pseudo-terminal free) and `500` are terminal; `429`/`502`/`504` (and `503` off a start) are retried, and nothing else is; `401`/`403` are `Auth`; a remove of a job already gone is done. A terminal answer finishes with its fault; a retry waits. The completeness half is clause (d) and (i) of `classify`: the busy statuses are never terminal and no other status is retried. | `classify`'s ensures, `post_classify`                    | `mut_429_terminal`, `mut_503_start_retried`, `mut_auth_retried`, `mut_404_retried`, `mut_500_retried`, `mut_everything_retried` |
| **W** Waiting. Every wait is at most 30 s from now (`Retry-After` included), honours the server's `Retry-After` up to that cap, and is at least 125 ms.                                                                                                                                                                                                                                                                                                                                                                                        | `wait_ms`'s ensures, `post_backoff`                      | `mut_backoff_no_cap`, `mut_retry_after_ignored`, `mut_retry_after_uncapped`                                                     |
| **P** Progress (liveness). Once the server is reachable (fairness `fair`: a wait ends, a request is answered 2xx, the echo carries the held mark, the consumer takes the exit), every fair step lowers `rank` and never fails, so a held job is reattached or ended within `rank` (at most 5) fair steps, from any count of failed opens: there is no exhaustion state. The end of a wait always reopens.                                                                                                                                      | `lemma_progress`, `lemma_exit_progress`, `post_progress` | `mut_backoff_never_reopens` (and `mut_exit_taken_stalls`, `mut_echo_skipped`)                                                   |
| **S** Silence limit. Every request names the dead-stream limit `max(30 s, 2 beats + 10 s)`; the Rust SDK's `silence_limit` calls it.                                                                                                                                                                                                                                                                                                                                                                                                           | `silence_ms`'s ensures, `post_silence`                   | `mut_silence_fixed`, `mut_silence_one_beat`                                                                                     |
| **K** The start key. Every send of the start carries the key of the first send, and a lost start is sent again (not given up).                                                                                                                                                                                                                                                                                                                                                                                                                 | `post_key`                                               | `mut_start_fresh_key`, `mut_start_lost_gives_up`                                                                                |
| **V** Watch. A watching handle counts output and delivers none of it (its exit carries no gap).                                                                                                                                                                                                                                                                                                                                                                                                                                                | `post_watch`                                             | `mut_watch_delivers`                                                                                                            |
| **D** Detach and end. A close detaches and never removes; a handle that is done stays done; a handle is done exactly when it issued its one terminal action (always the last action).                                                                                                                                                                                                                                                                                                                                                          | `post_end`                                               | `mut_close_removes`                                                                                                             |

No `assume`, `admit` or `external_body`. `wf` is the invariant of every state `step` takes and returns
(`State::for_start`, `for_attach`, `for_cursor` make well-formed ones).

Outside `verus!`: the constructors and `Mark::as_bytes`, which reshape a record, and
`silence_limit_ms(Option<u32>)`, a one-line wrapper over `silence_ms`.

### The model

`tests/step_model.rs` is a Stateright model that drives `step` and carries out the actions it returns, as a
shell does. It owns only the environment: the server's job slot (output, the ring dropping its oldest byte,
the job ending, removal and reuse of the id by a new incarnation), cuts, restarts (503), a lost start answer, an
endpoint that predates marks, and the consumer. It checks M1 to M9 (M9: reconnecting never ends the handle, including a front down for longer than the notice threshold) (the properties above, judged against the
environment's facts) and witnesses that each path is reachable.
`mise run test:job-client-model-mutants` runs it under each mutant it can see and fails if one stays green. A
property no model state can reach (`mut_offset_wraps` needs an offset near `u64::MAX`; the classification
rows and the wait cap are single-call facts) is Verus's alone.

### Not here

- The frame codec (JSON and base64 to a `Frame`, the precedence of the keys on a line) stays in each shell,
  covered by the shared fixtures. The kernel takes the decoded `Frame`.
- The stdin positions and their retry are a later slice.
- The FFI crate (WASM for TypeScript, a native library for Python) is lane B3: this crate stays free of unsafe.
