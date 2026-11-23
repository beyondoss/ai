# beyond-execd-wire

execd's wire timing and kill-scope decisions as pure functions, written inside `verus!` so the function
execd runs is the one [Verus](https://github.com/verus-lang/verus) checks (`mise run verify:kernels`).
execd re-exports it as `execd::wire_kernel`; `exec.rs` and `proc.rs` call it and keep only the I/O.

| Property                                                                                                                                                                     | Where                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| W1 a stream is never silent past one beat, and a silence of any length (335 s included) has a heartbeat in every beat-long window; the 5 s beat is below the gateway's 295 s | `quiet_wait`, `lemma_never_silent_past_a_beat`, `lemma_silence_is_carried`, `lemma_beat_below_idle` |
| W2 a one-shot still running at the first beat is committed (`200` headers); one that has ended answers as ever                                                               | `oneshot_mode`, `lemma_committed_is_200`                                                            |
| W3 every process of a job, in its group or marked and gone (`setsid`, a double fork), hears the job's signal                                                                 | `is_member`, `group_reaches`, `lemma_every_member_is_signalled`                                     |

## Mutations

Each removes one guard; the proof (`mise run verify:kernels -- --features <mut>`) and the test go red.

| Mutation                    | Test that goes red                                                                                                                                         |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mut_beat_late`             | `a_stream_is_never_silent_past_a_beat`, `a_335_second_silence_has_no_gap_past_a_beat` (this crate)                                                         |
| `mut_beat_never`            | the same two, and `embed::a_five_second_stream_with_heartbeats_is_not_cut_over_http1` (`beyond-execd`)                                                     |
| `mut_oneshot_never_commits` | `a_one_shot_still_running_at_the_first_beat_is_committed` (this crate); `oneshot_headers::a_long_command_gets_its_headers_before_it_ends` (`beyond-execd`) |
| `mut_stray_not_signalled`   | `a_marked_process_outside_the_group_is_signalled` (this crate); `job_escape::a_wipe_kills_what_timeout_and_setsid_took_out_of_the_group` (`beyond-execd`)  |
