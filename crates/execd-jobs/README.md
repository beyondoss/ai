# beyond-execd-jobs

The decisions of execd's job table, as pure functions: which `(scope, id)` a start may take, which
jobs an operation sees, what each event does to a job, when a signal is sent, when a job may leave
the table, when a held job dies with its client, and where an output read or a stream frame starts.
execd re-exports it as `execd::job_kernel`; `jobs.rs` is the shell (locks, processes, the ring's
bytes), and `crates/execd/tests/job_model.rs` model-checks the same functions. See
`crates/execd/ARCHITECTURE.md` ("The job kernel").

## Proven

The decisions are written inside `verus!`, so the function production runs is the one
[Verus](https://github.com/verus-lang/verus) checks, with no bound on how many jobs, events or bytes
there are. A normal build erases the specs and proofs; `mise run verify:kernels`
(`.github/scripts/verify-kernels.sh`, the release pinned in `.github/scripts/verus.env`, run by CI's
`verus` job) checks them. What it proves, in `src/lib.rs`:

| Property                                                                                                                                                                                                                                                                                                                    | Where                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| J1 an id is the claim: a key in the table is `Exists`, a free one `Fresh`                                                                                                                                                                                                                                                   | `admit`                                               |
| J2 an entry is seen only in its scope, and only if named                                                                                                                                                                                                                                                                    | `visible`                                             |
| J3 a removed job's pump is discarded, its end owes no callback, `Taken` is final                                                                                                                                                                                                                                            | `post_act`, `post_life`                               |
| J4 an ended job's code, signal, time and `shell_told` never change; phases only go forward; `removed` and `Kept` are final                                                                                                                                                                                                  | `post_life`                                           |
| J5 a gap is exactly `base - sent` (where a read starts is `execd_ring::window`, R1); a frame is `Data` only over held bytes                                                                                                                                                                                                 | `post_follow`                                         |
| J6 a kept job never dies with a held client; only a held, unkept, unended job does                                                                                                                                                                                                                                          | `post_act`                                            |
| J7 `Take` exactly when the group is observed empty or the ended job counts                                                                                                                                                                                                                                                  | `post_act`, `removable`                               |
| J8 a signal reaches a group only while the job has one and has not ended                                                                                                                                                                                                                                                    | `post_act`                                            |
| J9 a stream goes on only with the job it attached to                                                                                                                                                                                                                                                                        | `bound`                                               |
| J10 a take removes only the job that was checked, not its key's successor                                                                                                                                                                                                                                                   | `takes`                                               |
| J11 a signal by number reaches only a group its spawn still holds: never one the reaper freed (its number is any process's), never a later spawn's group on the same number; a held group is always deliverable (the completeness half, J11's liveness partner: a refusal clears only when there is nothing left to signal) | `Held::{has, insert, freed, numbers}`, `may_deliver`  |
| L1 an ended job can always be removed (`Take` whatever `empty` says); a job ends in at most two fair steps (F1: the supervisor delivers the spawn and the group's emptying)                                                                                                                                                 | `post_act`, `lemma_end_progress`, `end_rank`          |
| L2 an ended job's stream sends the shell's exit once, then the final frame, never `Wait`; each frame lowers `follow_rank` (F2: a `Data` frame takes at least a byte, which `execd_ring::window`'s R1 (proven in `crates/execd-ring`) promises; F3: the stream is scheduled)                                                 | `post_follow`, `lemma_follow_progress`, `follow_rank` |
| S1, S2 a job's output goes to a spool file it inherits, never a pipe (a write after execd restarted would be `SIGPIPE`); a command's goes to the pipe its client reads                                                                                                                                                      | `output_sink`                                         |
| S7 a job's spool is under the jobs directory, ledger or not, never the ledger's state directory (tmpfs in a sandbox: RAM)                                                                                                                                                                                                   | `spool_root`                                          |
| S6 a stop of execd (`SIGTERM`, `SIGINT`) signals every group, jobs included, and keeps no record                                                                                                                                                                                                                            | `shutdown_of`                                         |
| S3, S4, S5 a restart of execd (`SIGHUP`) never signals a job whose record a ledger keeps, and keeps that record (S3b: a job no ledger keeps is signalled, nothing could find it); it signals every command and keeps none                                                                                                   | `shutdown_of`                                         |
| X1, X2 a restarted execd resumes each stream at its cursor plus its journalled bytes, and its ring ends at the bytes consumed: a client's offset names the same byte before and after                                                                                                                                       | `resume`                                              |
| X3 a journalled byte already in a chunk file is not put in the ring again; every offset below the chunks' end is a chunk's, every offset at or above it the journal's, each once                                                                                                                                            | `replay_skip`                                         |

No `assume`, `admit` or `external_body`. Each `mut_*` feature breaks one decision and turns the
proof red (`mise run verify:kernels -- --features mut_x`; execd forwards the features to its model).

## What stays outside `verus!`

Not the storage any more. `Table` is a `Vec<(Key, C)>` inside `verus!`: vstd specifies `String`'s equality and
clone (`vstd-0.0.0-2026-11-01-0251/string.rs` 369-377, 411-421) and `Vec`'s push, remove and insert
(`std_specs/vec.rs` 149, 211, 230), so Verus proves what a `HashMap` could not (its key model is axiomatized
only for `bool` and the integers, `std_specs/hash.rs` 13-18, 113-215; an interning map would only move the
gap). Proven in `src/lib.rs`: the keys are distinct (`Table::wf`, kept by `new` and `insert`); `get` is `None`
exactly for a free key and otherwise returns a clone of the job under that exact `(scope, id)` (J2's
exact-key half, `mut_scope_ignored`); `in_scope` returns exactly the entries in the scope whose id is named
(`Pick::has`); `insert` puts the job under its key and changes no other key; `take` takes only a job whose
key was checked and `same`, and leaves its key resolving to nothing (J3, `mut_remove_keeps_entry`; J10).
Lookup is a linear scan (execd has no job cap; measured below).

What stays outside:

- `Table::wf` is a precondition of `insert` and `take` that no unverified caller discharges: it holds
  because `new` makes a table that satisfies it, every method keeps it, and the field is private. (A type
  invariant would be the checked form; Verus asks `no_unwind` of every function that then mutates the field,
  and vstd's `Vec::push` and `remove` unwind.)
- `same`, the pointer-identity test `take` is handed, is the shell's (its result is opaque to the proof; the
  decision to take on it is `takes`, proven).
- The table's `Clone`, `Debug` and `Eq` (the model's state), which only copy or print the entries.
- Accessors that only reshape a field (`State::kind`, `Named::as_str`) and the one-line `Table`
  methods that call the proven decisions.
- That the shell calls the kernel for every decision, in the right order under the right lock: the
  structure test (`tests/production.rs`) and the model.
- The fairness assumptions F1-F3 of L1 and L2: the environment delivers the events.
- J11's ordering. The kernel proves the ledger (`Held`) and the check (`may_deliver`); that the reaper frees a number in the
  same exclusion in which it collects the last member, and that a signal checks and sends inside the exclusion that keeps the
  reaper out (`proc::sweep`, `proc::signal_job`), is the shell's: model-checked in `crates/execd/tests/os_model.rs` (the
  real `Held` and `may_deliver` against an OS actor: exits, reaper passes, foreign reuse of the number, a later job on it, a
  decision delivered late) and pinned by its source-text check `production_signals_only_held_groups`. Not covered: a host
  without the subreaper (init reaps orphans outside the exclusion), and the supervisor's by-number observation of a group
  while a foreign group holds the number (it delays the end, never a signal).
