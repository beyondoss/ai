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

## Reference

The normative spec is the "Sandbox job notify" section of the module doc
in [`crates/sandbox/src/exec_endpoint.rs`](../sandbox/src/exec_endpoint.rs). Its
requirements are tagged N1, N2, …; this crate implements N16 to N18 and the
callback grammar and body (N4, N14). `cargo nextest run -p beyond-exec-notify`
runs the tests, including an exhaustive check of every short event sequence.
