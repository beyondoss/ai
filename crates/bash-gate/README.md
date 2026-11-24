# beyond-bash-gate

The agent's `bash` decisions as pure functions, written inside `verus!` and checked by
`mise run verify:kernels`. The agent's `tools/bash_jobs.rs` and `tools/bash.rs` call them.

| Property                                                                                                                                                                 | Where                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| B1 a start is admitted below 20 running commands and refused at 20; the decision has no arm that stops a running one                                                     | `admit_start`, `lemma_a_start_stops_nothing`                                            |
| B2 a foreground call hands back to the model after 60 s of silence, at its ceiling or at the model's `timeout_ms`, and always by its ceiling; handing back kills nothing | `yield_reason`, `lemma_sixty_seconds_hands_back`, `lemma_a_call_returns_by_its_ceiling` |
| B3 a job's end is told once across a restart: a notice the persisted transcript holds is never due again, an ended job it does not hold always is | `end_due` |

## Mutations

| Mutation                     | Test that goes red                                                                                                                                                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mut_gate_ignores_limit`     | `the_limit_gates_a_start` (this crate); `bash_background_tests::the_limit_counts_every_running_job_and_refuses_before_starting`, `bash_jobs_tests::a_command_at_the_limit_is_refused_with_the_list` (`beyond-agent`)                    |
| `mut_quiet_never_hands_back` | `sixty_quiet_seconds_hand_a_call_back` (this crate); `bash_idle_yield_tests::a_quiet_command_returns_early_and_keeps_running_until_killed` (`beyond-agent`; the command outlasts the 60 s, so it fails by the call not returning early) |
| `mut_told_ignores_transcript` | `a_persisted_end_is_not_told_again` (this crate); `bash_jobs_serve_sandbox::a_job_end_the_transcript_already_holds_is_not_told_again_after_a_restart` (`beyond-agent`) |
