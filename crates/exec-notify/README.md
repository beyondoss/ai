# beyond-exec-notify

The notification rules of exec protocol notify, as one pure function.

In notify a sandbox tells the agent that a kept job ended by calling back
(`POST` of `{"type":"job_ended","scope":…,"id":…}`) instead of holding a stream
open. This crate decides _when_ that callback is sent: when one is owed, when an
attempt starts, how long to wait after a failure, and when to stop. It makes no
request and reads no clock, so an endpoint on either side of the repo boundary
can link it, and a model checker can call it exactly as production does.

Use it if you write an exec endpoint (a vendor adapter, a sandbox service) that
supports `keep` with `notify`. The agent and the reference endpoint in
`crates/test-support` link it.

## Use

Keep one `Job` per job. Feed every `Event` to `step` with the current time in
milliseconds; keep the `Job` it returns and carry out the `Action`s:

| `Event`                          | Feed it when                                                  |
| -------------------------------- | ------------------------------------------------------------- |
| `Keep { notify }`                | A `keep` is accepted, with or without `notify`.               |
| `Change`                         | The job's state changes while its latest `keep` has `notify`. |
| `Removed`                        | The job is removed (`remove`, `kill` with `remove`, `wipe`).  |
| `AttemptEnded { outcome, draw }` | An attempt ended; `draw` is a fresh uniform `u32`.            |
| `Wake`                           | A time the kernel asked for (`WakeAt`) has come.              |

| `Action`            | Do                                                        |
| ------------------- | --------------------------------------------------------- |
| `Begin { attempt }` | Send the `POST`, with the job's `notify` as recorded now. |
| `Abandon`           | Drop the attempt in flight, connection included.          |
| `WakeAt(ms)`        | Feed `Event::Wake` at that time.                          |

```rust
use exec_notify::{Action, Event, Job, Outcome, step};

// `keep` with notify: the first attempt begins at once.
let (job, actions) = step(Job::default(), Event::Keep { notify: true }, 0);
assert_eq!(actions, [Action::Begin { attempt: 1 }]);

// The receiver answers 503: wait, then retry.
let ended = Event::AttemptEnded { outcome: Outcome::Status(503), draw: 0 };
let (job, actions) = step(job, ended, 100);
assert_eq!(actions, [Action::WakeAt(1100)]);

let (_job, actions) = step(job, Event::Wake, 1100);
assert_eq!(actions, [Action::Begin { attempt: 2 }]);
```

## Rules it encodes

- **One notification per job at a time.** It carries every change owed. An
  answer settles only what was owed when its attempt began.
- **Status decides.** `2xx` is done. `401`, `403`, `404`, `410` and any other
  status outside the retry list stop it. `407`, `408`, `425`, `429`, `5xx`, a
  connection or TLS failure, and no answer in 30 s (`ATTEMPT_TIMEOUT_MS`)
  retry. `Retry-After` is ignored.
- **Schedule.** At most 12 attempts (`MAX_ATTEMPTS`). After failed attempt _n_
  the wait is `min(2^(n-1), 60)` s times `(1 − u/2)`, `u` drawn fresh per wait
  (`wait_ms`), so a count lasts at most 723 s. The arithmetic is integer-only.
- **Nothing after removal,** or for a job whose latest `keep` carried no
  `notify`. A new `keep` abandons the attempt in flight.

Also exported: `check_notify(url, token)`, which applies the `keep` grammar
(absolute `https` URL, or `http` to loopback, at most 2048 bytes, no userinfo or
fragment; token of 1 to 512 `token68` bytes); `body(scope, id)`, the callback
body.

## Proven

`step` and its helpers (`verdict`, `wait_ms`) are written inside `verus!`, so the
function production runs is the one [Verus](https://github.com/verus-lang/verus)
checks, with no bound on how long a job lives. A normal build erases the specs
and proofs; `mise run verify:kernels` (`.github/scripts/verify-kernels.sh`, the
release pinned in `.github/scripts/verus.env`, run by CI's `verus` job) checks
them. What it proves, in `src/lib.rs`:

| Property                                                                                                                                                | Where                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| The invariant holds from `Job::default()` on, through every event                                                                                       | `wf`, `step`'s `wf(r.0)`                       |
| Never lost: a live job that owes a callback is in flight or waiting for a wake; settling takes an answer or a spent count                               | `wf`, `post_owed`, `post_answer`               |
| One notification per job: an attempt begins only when none is in flight; a delivered notice is not sent again unless new owed                           | `post_one`                                     |
| Retries are bounded: at most `MAX_ATTEMPTS` per count; each wait is `d(n)·(1 − u/2)`, between `d(n)/2` and `d(n)` ms                                    | `wf`, `post_answer`, `wait_ms`                 |
| Nothing after removal                                                                                                                                   | `post_removed`                                 |
| Liveness: under F1 (attempts end), F2 (time advances, wakes arrive), F3 (the job quiesces) a job that owes settles in at most `rank_bound()` fair steps | `lemma_progress`, `lemma_rank_bounded`, `rank` |
| The reporter's window: a count of 12 attempts, each at most 30 s, and the 11 waits between them (at most `d(n)`) lasts at most 723 s; a retry past the 12th is given up (settled, the end left for the next contact); a refusal is final | `count_ms`, `lemma_retry_window`, `lemma_a_count_is_bounded`, `lemma_a_stop_is_final`, `lemma_4xx_table` (red under `mut_wait_doubled`, `mut_retry_unbounded`, `mut_stop_is_retried`; test `a_count_of_failed_attempts_ends_723_seconds_after_it_began`) |

`step` requires a job that `step` or `Default` made, `owed < u64::MAX` and `now`
within a minute of `u64::MAX`; a millisecond clock and a count of events reach
neither. Each `mut_*` feature breaks one decision and turns the proof red
(`mise run verify:kernels -- --features mut_x`); the agent forwards them, and
`mise run test:notify-model-mutants` turns the notify model red under those it
can see.

## Reference

The normative spec is the "Sandbox job notify" section of the module doc
in [`crates/sandbox/src/exec_endpoint.rs`](../sandbox/src/exec_endpoint.rs). Its
requirements are tagged N1, N2, …; this crate implements N16 to N18 and the
callback grammar and body (N4, N14). `cargo nextest run -p beyond-exec-notify`
runs the tests, including an exhaustive check of every short event sequence.
