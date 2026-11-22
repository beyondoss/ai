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

| Property                                                                                                                                                                                            | Where                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| J1 an id is the claim: a key in the table is `Exists`, a free one `Fresh`                                                                                                                           | `admit`                                                         |
| J2 an entry is seen only in its scope, and only if named                                                                                                                                            | `visible`                                                       |
| J3 a removed job's pump is discarded, its end owes no callback, `Taken` is final                                                                                                                    | `post_act`, `post_life`                                         |
| J4 an ended job's code, signal, time and `shell_told` never change; phases only go forward; `removed` and `Kept` are final                                                                          | `post_life`                                                     |
| J5 a read starts at `max(offset, base)`, never before the oldest byte held, and answers at most `max` bytes within the ring; a gap is exactly `base - sent`; a frame is `Data` only over held bytes | `window`, `post_follow`                                         |
| J6 a kept job never dies with a held client; only a held, unkept, unended job does                                                                                                                  | `post_act`                                                      |
| J7 `Take` exactly when the group is observed empty or the ended job counts                                                                                                                          | `post_act`, `removable`                                         |
| J8 a signal reaches a group only while the job has one and has not ended                                                                                                                            | `post_act`                                                      |
| J9 a stream goes on only with the job it attached to                                                                                                                                                | `bound`                                                         |
| L1 an ended job can always be removed (`Take` whatever `empty` says); a job ends in at most two fair steps (F1: the supervisor delivers the spawn and the group's emptying)                         | `post_act`, `lemma_end_progress`, `end_rank`                    |
| L2 an ended job's stream sends the shell's exit once, then the final frame, never `Wait`; each frame lowers `follow_rank` (F2: a `Data` frame takes at least a byte; F3: the stream is scheduled)   | `post_follow`, `lemma_follow_progress`, `follow_rank`, `window` |

No `assume`, `admit` or `external_body`. Each `mut_*` feature breaks one decision and turns the
proof red (`mise run verify:kernels -- --features mut_x`; execd forwards the features to its model).

## What stays outside `verus!`

Storage and string comparison, because Verus has no model of a `HashMap` keyed by strings (its hash
axioms cover integers, and a key model for `String` would be an `assume`):

- `Table`'s `HashMap`: that `contains_key` finds the key, that `get` returns the entry under exactly
  that `(scope, id)`, and that `take` removes it. `admit` and `visible` take what the storage
  observed and are proven; the lookup itself is not. J3's "no operation resolves a removed job"
  (`mut_remove_keeps_entry`) and the exact-key half of J2 (`Table::get`) are **gaps** here, judged by
  the model only.
- `Pick::has` (is this id one of the named ones: a string comparison).
- Accessors that only reshape a field (`State::kind`, `Named::as_str`) and the one-line `Table`
  methods that call the proven decisions.
- That the shell calls the kernel for every decision, in the right order under the right lock: the
  structure test (`tests/production.rs`) and the model.
- The fairness assumptions F1-F3 of L1 and L2: the environment delivers the events.
