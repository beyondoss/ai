# beyond-bash-gate

The agent's `bash` decisions as pure functions, written inside `verus!` and checked by
`mise run verify:kernels`. The agent's `tools/bash_jobs.rs` and `tools/bash.rs` call them.

| Property                                                                                                                                                                 | Where                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| B1 a start is admitted below 20 running commands and refused at 20; the decision has no arm that stops a running one                                                     | `admit_start`, `lemma_a_start_stops_nothing`                                            |
| B2 a foreground call hands back to the model after 60 s of silence, at its ceiling or at the model's `timeout_ms`, and always by its ceiling; handing back kills nothing | `yield_reason`, `lemma_sixty_seconds_hands_back`, `lemma_a_call_returns_by_its_ceiling` |
| B3 a job's end is told once across a restart: a notice the persisted transcript holds is never due again, an ended job it does not hold always is                        | `end_due`                                                                               |
| B4 a kept job is followed while the turn lives: after a break the follow goes on, waiting 500 ms doubling to at most 5 s, and stops only on the job's end (`Lost`/`Gone`), a terminal error class, or the turn's own cancel; no count of breaks or timer ends it (liveness: once the store answers, the next follow reattaches; fairness axiom F, the front eventually answers) | `follow_after_break`, `backoff_ms`, `lemma_follow_reattaches_once_the_front_answers`, `lemma_backoff_is_monotone_and_capped` |

## Mutations

| Mutation                      | Test that goes red                                                                                                                                                                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mut_gate_ignores_limit`      | `the_limit_gates_a_start` (this crate); `bash_background_tests::the_limit_counts_every_running_job_and_refuses_before_starting`, `bash_jobs_tests::a_command_at_the_limit_is_refused_with_the_list` (`beyond-agent`)                    |
| `mut_quiet_never_hands_back`  | `sixty_quiet_seconds_hand_a_call_back` (this crate); `bash_idle_yield_tests::a_quiet_command_returns_early_and_keeps_running_until_killed` (`beyond-agent`; the command outlasts the 60 s, so it fails by the call not returning early) |
| `mut_told_ignores_transcript` | `a_persisted_end_is_not_told_again` (this crate); `bash_jobs_serve_sandbox::a_job_end_the_transcript_already_holds_is_not_told_again_after_a_restart` (`beyond-agent`)                                                                  |
| `mut_follow_gives_up`         | `a_broken_stream_is_followed_until_the_job_is_over` (this crate); `a_kept_command_is_followed_through_a_front_outage_longer_than_the_old_cap` (`beyond-agent`, `bash_sandbox_jobs`); Verus: `follow_after_break`'s stop-only-on-cause clause |
| `mut_follow_past_the_end`     | Verus: `follow_after_break`'s `Over` clause; the unit test above |
| `mut_follow_past_cancel`      | Verus: `follow_after_break`'s `Cancelled` clause; the unit test above |
| `mut_backoff_no_cap`          | Verus: `backoff_ms`'s cap clause |
| `mut_backoff_spins`           | Verus: `backoff_ms`'s base clause |
