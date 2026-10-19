//! The decision logic of exec v1.4's job notify, as one pure state machine: what a session does
//! with each event about its kept jobs, its callbacks and its reconciles.
//!
//! [`Kernel::step`] takes an [`Event`] and returns the [`Action`]s to perform. It does no I/O, is
//! not async, reads no clock (time arrives in the events, as [`Millis`] on the caller's monotonic
//! scale) and is deterministic, so a model checker can call exactly the function production calls.
//! Production ([`super::SessionNotify`], `tools::bash_sandbox`, the `serve` receiver) only feeds it
//! events and performs its actions.
//!
//! What it owns is v1.4's own state, per sandbox job id ([`JobState`]): a job **proving** its path
//! (its proof window open), one whose stream is **closing** after its proof, and one **unheld** (no
//! stream: only a callback, a use or a `bash_job` call brings news of it), or unheld and **taken**
//! by a `bash_job` call. A job in none of these is as today, and what today's job table says of it
//! arrives with the event that needs it ([`Facts`]). It also owns the session's reconcile schedule
//! (one at a time, the next shared by everyone who asks while one runs, spaced
//! [`RECONCILE_SPACING`] after the last for a callback or a release, and [`USE_SPACING`] for a
//! use) and the callbacks waiting for an answer (at most [`MAX_WAITING`], each answered `503` at
//! [`ANSWER_LIMIT`] after it arrived).
//!
//! The release table ("Release (normative)" in [`crate::exec_endpoint`]), row by row:
//!
//! | state | event | actions | next |
//! |---|---|---|---|
//! | (parking) | [`Event::Kept`] with notify available | [`Action::SendKeep`] with notify | proving |
//! | (parking) | [`Event::Kept`] without | [`Action::SendKeep`] without (none for an adoption) | held |
//! | proving | [`Event::KeepAnswered`] `Refused` | [`Action::SendKeep`] without notify | held |
//! | proving | [`Event::KeepAnswered`] `Failed` | none | held |
//! | proving | proof inside the window and `keep` succeeded, either order | [`Action::CloseStream`] | closing |
//! | closing | [`Event::StreamClosed`] | [`Action::StartReconcile`] (a new one) | unheld |
//! | proving | the window closes ([`Event::Tick`]) | none | held |
//! | proving, closing | [`Event::StreamEnded`] | none (as today) | held |
//! | unheld | a reconcile lists it `running` | none | unheld |
//! | unheld | a reconcile lists it `exited` | [`Action::ApplyShellExit`], once | unheld |
//! | unheld | listed `ended`/`lost`, missing, or the list failed | [`Action::Follow`] | held |
//! | unheld | [`Event::Taken`] | [`Action::ReadInThenFollow`] | taken |
//! | taken | [`Event::PutBack`] running / ended | [`Action::CloseFollow`] / none | unheld / held |

use std::collections::BTreeMap;

/// A time on the caller's monotonic scale, in milliseconds.
pub type Millis = u64;

/// A callback waiting for its answer, as the caller numbers them.
pub type CallId = u64;

/// One live reconcile, as the kernel numbers them.
pub type ReconcileId = u64;

/// Most callbacks one session holds waiting for an answer; one more is answered `503`.
pub const MAX_WAITING: usize = 16;

/// How long after its arrival a callback still waiting is answered `503`.
pub const ANSWER_LIMIT: Millis = 20_000;

/// The least time from the end of a reconcile to the start of one a callback or a release asked for.
pub const RECONCILE_SPACING: Millis = 1_000;

/// A real use of the sandbox starts a reconcile only when none ended in this long.
pub const USE_SPACING: Millis = 30_000;

/// The proof window stays open this long after `keep` is answered.
pub const PROOF_WINDOW: Millis = 30_000;

/// How `keep` with notify was answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeepOutcome {
    Succeeded,
    /// `400`: the endpoint refused the notify; the job is kept again without it.
    Refused,
    /// Any other failure, as today's failed keep.
    Failed,
}

/// v1.4's own state of one kept job. A job not in the kernel is held, or ended, as today.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum JobState {
    /// Its proof window is open. `window_until` is `None` until `keep` is answered (the window
    /// opened before `keep` was sent); `proof` is set once a callback arrived inside the window.
    Proving {
        keep_succeeded: bool,
        window_until: Option<Millis>,
        proof: bool,
    },
    /// Proven: its stream (or the follow standing for it) is being closed.
    Closing,
    /// No stream. `offset` is where its stream got to; `shell_exit_seen` once its shell's exit has
    /// been applied (by its stream, or by a listing).
    Unheld { offset: u64, shell_exit_seen: bool },
    /// Unheld, and a `bash_job` call in flight has it.
    Taken { offset: u64, shell_exit_seen: bool },
}

/// What today's job table says of the job a callback names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TableJob {
    /// In the table and running: its stream or a follow has it.
    Running,
    /// Its end is recorded (in the table ended, or reported and awaiting removal).
    Ended,
    /// A `bash_job` call in flight has it out of the table.
    Taken,
    /// Neither tracked nor recorded.
    Absent,
}

/// What the session knows of a callback's job beyond the kernel's own state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Facts {
    /// The session's open reconcile has succeeded.
    pub open_reconcile_done: bool,
    pub job: TableJob,
}

/// What a listing said of one job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Listed {
    Running,
    /// The shell exited with `code`; processes of its group run on.
    Exited {
        code: i32,
    },
    /// Ended with `code`, at `at` (the listing's `ended`, epoch seconds).
    Ended {
        code: i32,
        at: u64,
    },
    /// The sandbox lost track of it; `at` is `max(ended, started)`.
    Lost {
        at: u64,
    },
}

/// An end a listing found, handed to today's follow to record if it cannot read the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ListedEnd {
    /// The listed status, `None` for `lost`.
    pub code: Option<i32>,
    /// Epoch seconds.
    pub at: u64,
}

/// Something that happened to the session.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// A verified callback for `job` arrived at `now` (its request head was read).
    Callback {
        call: CallId,
        job: String,
        facts: Facts,
        now: Millis,
    },
    /// A waiting callback's connection closed: stop waiting, answer nothing.
    CallClosed { call: CallId },
    /// Time passed.
    Tick { now: Millis },
    /// A job is being kept: parked (`adopted: false`) or adopted by the open reconcile.
    Kept {
        job: String,
        notify_available: bool,
        adopted: bool,
    },
    /// `keep` with notify was answered.
    KeepAnswered {
        job: String,
        outcome: KeepOutcome,
        now: Millis,
    },
    /// The stream a release closed is closed; `offset` is where it got to.
    StreamClosed {
        job: String,
        offset: u64,
        shell_exit_seen: bool,
        now: Millis,
    },
    /// A proving or closing job's stream (or follow) delivered its final frame or broke: as today.
    StreamEnded { job: String },
    /// A live reconcile listed the scope (`Some`), or could not (`None`).
    Listed {
        reconcile: ReconcileId,
        jobs: Option<BTreeMap<String, Listed>>,
        now: Millis,
    },
    /// A `bash_job` call took the job out of the table.
    Taken { job: String },
    /// The call put it back, still running at `offset`, or with its end recorded.
    PutBack {
        job: String,
        running: bool,
        offset: u64,
    },
    /// The job is gone from the session (killed, removed).
    Forgotten { job: String },
    /// A real use of the sandbox.
    Use { now: Millis },
    /// The session moved to another transcript: the old one's jobs are wiped with it.
    TranscriptSwitched,
}

/// Why a callback got the answer it got: what the replica's metrics count it as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Why {
    /// The job's proof, inside its window.
    Proof,
    /// A hint about a job held or proving (or a proof after its window): nothing to do.
    Hint,
    /// The session has not read its sandbox's jobs back yet.
    NotReadBack,
    /// A live reconcile that began after it arrived listed the scope.
    Listed,
    /// The job's end is already recorded: a repeat.
    Recorded,
    /// A `bash_job` call has the job.
    Taken,
    /// The session neither tracks the job nor has recorded its end.
    Unknown,
    /// No answer was ready in time.
    Late,
    /// Too many callbacks were waiting.
    Full,
    /// The reconcile could not list the sandbox.
    Unlisted,
}

/// Why a kept job stays held (by its stream, as today) rather than being unheld.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeldWhy {
    /// Notify is not available (no origin configured, or the endpoint does not offer it).
    Unavailable,
    /// The endpoint refused the `notify` (`400`).
    Refused,
    /// `keep` with notify failed some other way.
    KeepFailed,
    /// The proof window closed with no proof.
    NoProof,
    /// Its stream ended (or broke) while it was proving.
    Ended,
}

/// Something the caller must do.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Answer a callback with this status: `200` (a `job_ended` response), `410` or `503`.
    Answer { call: CallId, status: u16, why: Why },
    /// The job stays held, as today (for the replica's metrics; nothing to send).
    Held { job: String, why: HeldWhy },
    /// Send `keep` for the job, with notify or without.
    SendKeep { job: String, notify: bool },
    /// Close the job's stream (or the follow standing for it), then report [`Event::StreamClosed`].
    CloseStream { job: String },
    /// `list` the scope and report [`Event::Listed`]. Nothing else is sent.
    StartReconcile { reconcile: ReconcileId },
    /// Apply a listed `exited` as a held stream's `shell_exit`.
    ApplyShellExit { job: String, code: i32 },
    /// Today's follow, from `from`, recording `end` if it cannot read the rest.
    Follow {
        job: String,
        from: u64,
        end: Option<ListedEnd>,
    },
    /// A `bash_job` call about an unheld job: read in what the sandbox holds from `from`, then
    /// today's follow for the rest of the call.
    ReadInThenFollow { job: String, from: u64 },
    /// The call put the unheld job back still running: close the follow it had.
    CloseFollow { job: String },
}

/// The next reconcile, asked for and not begun: who waits on it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct Batch {
    waiters: Vec<CallId>,
}

/// The reconcile running now.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Running {
    id: ReconcileId,
    transcript: u64,
    waiters: Vec<CallId>,
}

/// One session's v1.4 state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Kernel {
    jobs: BTreeMap<String, JobState>,
    /// Callbacks waiting for an answer, with when each must be answered by.
    waiting: BTreeMap<CallId, Millis>,
    running: Option<Running>,
    next: Option<Batch>,
    last_end: Option<Millis>,
    next_reconcile: ReconcileId,
    /// Bumped by every transcript switch.
    transcript: u64,
}

/// [`Kernel::step`] as a function of the state: the next state and the actions.
pub fn step(state: &Kernel, event: Event) -> (Kernel, Vec<Action>) {
    let mut next = state.clone();
    let actions = next.step(event);
    (next, actions)
}

impl Kernel {
    pub fn new() -> Self {
        Self::default()
    }

    /// v1.4's state of `job`, if it has one.
    pub fn job(&self, job: &str) -> Option<&JobState> {
        self.jobs.get(job)
    }

    /// Every job v1.4 has a state for.
    pub fn jobs(&self) -> impl Iterator<Item = (&String, &JobState)> {
        self.jobs.iter()
    }

    /// Whether the session has an unheld job (taken or not).
    pub fn has_unheld(&self) -> bool {
        self.jobs
            .values()
            .any(|s| matches!(s, JobState::Unheld { .. } | JobState::Taken { .. }))
    }

    /// How many callbacks are waiting for an answer.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Whether a reconcile is running.
    pub fn reconciling(&self) -> bool {
        self.running.is_some()
    }

    /// When [`Event::Tick`] next has something to do: a waiting callback's limit, a proof
    /// window's close, or the earliest start of an asked-for reconcile.
    pub fn next_wake(&self) -> Option<Millis> {
        let waiters = self.waiting.values().copied();
        let windows = self.jobs.values().filter_map(|s| match s {
            JobState::Proving {
                window_until: Some(t),
                ..
            } => Some(*t),
            _ => None,
        });
        let start = match (&self.running, &self.next, self.last_end) {
            (None, Some(_), Some(end)) => Some(end + RECONCILE_SPACING),
            _ => None,
        };
        waiters.chain(windows).chain(start).min()
    }

    /// Apply one event.
    pub fn step(&mut self, event: Event) -> Vec<Action> {
        let mut out = Vec::new();
        match event {
            Event::Callback {
                call,
                job,
                facts,
                now,
            } => self.callback(call, job, facts, now, &mut out),
            Event::CallClosed { call } => {
                self.waiting.remove(&call);
            }
            Event::Tick { now } => self.tick(now, &mut out),
            Event::Kept {
                job,
                notify_available,
                adopted,
            } => {
                if notify_available {
                    // The window opens before `keep` is sent, so a proof that overtakes its
                    // answer still counts.
                    self.jobs.insert(
                        job.clone(),
                        JobState::Proving {
                            keep_succeeded: false,
                            window_until: None,
                            proof: false,
                        },
                    );
                    out.push(Action::SendKeep { job, notify: true });
                } else if !adopted {
                    self.jobs.remove(&job);
                    out.push(Action::SendKeep {
                        job: job.clone(),
                        notify: false,
                    });
                    out.push(Action::Held {
                        job,
                        why: HeldWhy::Unavailable,
                    });
                } else {
                    out.push(Action::Held {
                        job,
                        why: HeldWhy::Unavailable,
                    });
                }
            }
            Event::KeepAnswered { job, outcome, now } => {
                if let Some(JobState::Proving {
                    keep_succeeded,
                    window_until,
                    proof,
                }) = self.jobs.get_mut(&job)
                {
                    match outcome {
                        KeepOutcome::Succeeded => {
                            *keep_succeeded = true;
                            *window_until = Some(now + PROOF_WINDOW);
                            if *proof {
                                self.release(job, &mut out);
                            }
                        }
                        KeepOutcome::Refused => {
                            self.jobs.remove(&job);
                            out.push(Action::SendKeep {
                                job: job.clone(),
                                notify: false,
                            });
                            out.push(Action::Held {
                                job,
                                why: HeldWhy::Refused,
                            });
                        }
                        KeepOutcome::Failed => {
                            self.jobs.remove(&job);
                            out.push(Action::Held {
                                job,
                                why: HeldWhy::KeepFailed,
                            });
                        }
                    }
                }
            }
            Event::StreamClosed {
                job,
                offset,
                shell_exit_seen,
                now,
            } => {
                if matches!(self.jobs.get(&job), Some(JobState::Closing)) {
                    self.jobs.insert(
                        job,
                        JobState::Unheld {
                            offset,
                            shell_exit_seen,
                        },
                    );
                    // After the stream is closed, and never one that began earlier: the next
                    // batch has not begun.
                    self.next.get_or_insert_with(Batch::default);
                    self.maybe_start(now, &mut out);
                }
            }
            Event::StreamEnded { job } => {
                if matches!(
                    self.jobs.get(&job),
                    Some(JobState::Proving { .. } | JobState::Closing)
                ) {
                    self.jobs.remove(&job);
                    out.push(Action::Held {
                        job,
                        why: HeldWhy::Ended,
                    });
                }
            }
            Event::Listed {
                reconcile,
                jobs,
                now,
            } => self.listed(reconcile, jobs, now, &mut out),
            Event::Taken { job } => {
                if let Some(JobState::Unheld {
                    offset,
                    shell_exit_seen,
                }) = self.jobs.get(&job).cloned()
                {
                    self.jobs.insert(
                        job.clone(),
                        JobState::Taken {
                            offset,
                            shell_exit_seen,
                        },
                    );
                    out.push(Action::ReadInThenFollow { job, from: offset });
                }
            }
            Event::PutBack {
                job,
                running,
                offset,
            } => {
                if let Some(JobState::Taken {
                    shell_exit_seen, ..
                }) = self.jobs.get(&job).cloned()
                {
                    if running {
                        self.jobs.insert(
                            job.clone(),
                            JobState::Unheld {
                                offset,
                                shell_exit_seen,
                            },
                        );
                        out.push(Action::CloseFollow { job });
                    } else {
                        self.jobs.remove(&job);
                    }
                }
            }
            Event::Forgotten { job } => {
                self.jobs.remove(&job);
            }
            Event::Use { now } => {
                let quiet = self.last_end.is_none_or(|e| now >= e + USE_SPACING);
                if self.has_unheld() && self.running.is_none() && self.next.is_none() && quiet {
                    self.next = Some(Batch::default());
                    self.maybe_start(now, &mut out);
                }
            }
            Event::TranscriptSwitched => {
                self.jobs.clear();
                self.transcript += 1;
            }
        }
        out
    }

    fn callback(
        &mut self,
        call: CallId,
        job: String,
        facts: Facts,
        now: Millis,
        out: &mut Vec<Action>,
    ) {
        let answer = |status, why| Action::Answer { call, status, why };
        if self.waiting.len() >= MAX_WAITING {
            out.push(answer(503, Why::Full));
            return;
        }
        // Not read back yet: nothing here is known to be the session's, and nothing is contacted.
        if !facts.open_reconcile_done {
            out.push(answer(200, Why::NotReadBack));
            return;
        }
        // A call has it: asked again once the call records the end or puts it back.
        if facts.job == TableJob::Taken {
            out.push(answer(503, Why::Taken));
            return;
        }
        match self.jobs.get_mut(&job) {
            Some(JobState::Proving {
                keep_succeeded,
                window_until,
                proof,
            }) => {
                let inside = window_until.is_none_or(|t| now < t);
                if inside {
                    *proof = true;
                    if *keep_succeeded {
                        self.release(job, out);
                    }
                }
                out.push(answer(200, if inside { Why::Proof } else { Why::Hint }));
            }
            Some(JobState::Closing) => out.push(answer(200, Why::Hint)),
            Some(JobState::Taken { .. }) => out.push(answer(503, Why::Taken)),
            Some(JobState::Unheld { .. }) => {
                self.waiting.insert(call, now + ANSWER_LIMIT);
                self.next
                    .get_or_insert_with(Batch::default)
                    .waiters
                    .push(call);
                self.maybe_start(now, out);
            }
            None => out.push(match facts.job {
                TableJob::Running => answer(200, Why::Hint),
                TableJob::Ended => answer(200, Why::Recorded),
                TableJob::Taken => answer(503, Why::Taken),
                TableJob::Absent => answer(410, Why::Unknown),
            }),
        }
    }

    /// The release row: close the stream; the reconcile starts once it is closed.
    fn release(&mut self, job: String, out: &mut Vec<Action>) {
        self.jobs.insert(job.clone(), JobState::Closing);
        out.push(Action::CloseStream { job });
    }

    fn tick(&mut self, now: Millis, out: &mut Vec<Action>) {
        let late: Vec<CallId> = self
            .waiting
            .iter()
            .filter(|(_, by)| **by <= now)
            .map(|(c, _)| *c)
            .collect();
        for call in late {
            self.waiting.remove(&call);
            out.push(Action::Answer {
                call,
                status: 503,
                why: Why::Late,
            });
        }
        // A window that closed with no proof leaves the job held for good.
        let closed: Vec<String> = self
            .jobs
            .iter()
            .filter(
                |(_, s)| matches!(s, JobState::Proving { window_until: Some(t), .. } if *t <= now),
            )
            .map(|(j, _)| j.clone())
            .collect();
        for job in closed {
            self.jobs.remove(&job);
            out.push(Action::Held {
                job,
                why: HeldWhy::NoProof,
            });
        }
        self.maybe_start(now, out);
    }

    fn maybe_start(&mut self, now: Millis, out: &mut Vec<Action>) {
        if self.running.is_some() || self.next.is_none() {
            return;
        }
        if self.last_end.is_some_and(|e| now < e + RECONCILE_SPACING) {
            return;
        }
        let Some(batch) = self.next.take() else {
            return;
        };
        let id = self.next_reconcile;
        self.next_reconcile += 1;
        self.running = Some(Running {
            id,
            transcript: self.transcript,
            waiters: batch.waiters,
        });
        out.push(Action::StartReconcile { reconcile: id });
    }

    fn listed(
        &mut self,
        reconcile: ReconcileId,
        listing: Option<BTreeMap<String, Listed>>,
        now: Millis,
        out: &mut Vec<Action>,
    ) {
        let Some(running) = self.running.take_if(|r| r.id == reconcile) else {
            return;
        };
        self.last_end = Some(now);
        // Hand over: each unheld job looked up again, and only on the transcript it began on.
        if running.transcript == self.transcript {
            let unheld: Vec<(String, u64, bool)> = self
                .jobs
                .iter()
                .filter_map(|(j, s)| match s {
                    JobState::Unheld {
                        offset,
                        shell_exit_seen,
                    } => Some((j.clone(), *offset, *shell_exit_seen)),
                    _ => None,
                })
                .collect();
            for (job, offset, shell_exit_seen) in unheld {
                let seen = listing.as_ref().map(|l| l.get(&job).copied());
                let end = match seen {
                    Some(Some(Listed::Running)) => continue,
                    Some(Some(Listed::Exited { code })) => {
                        if !shell_exit_seen {
                            self.jobs.insert(
                                job.clone(),
                                JobState::Unheld {
                                    offset,
                                    shell_exit_seen: true,
                                },
                            );
                            out.push(Action::ApplyShellExit { job, code });
                        }
                        continue;
                    }
                    Some(Some(Listed::Ended { code, at })) => Some(ListedEnd {
                        code: Some(code),
                        at,
                    }),
                    Some(Some(Listed::Lost { at })) => Some(ListedEnd { code: None, at }),
                    // Missing from the listing, or no listing at all.
                    Some(None) | None => None,
                };
                self.jobs.remove(&job);
                out.push(Action::Follow {
                    job,
                    from: offset,
                    end,
                });
            }
        }
        let (status, why) = if listing.is_some() {
            (200, Why::Listed)
        } else {
            (503, Why::Unlisted)
        };
        for call in running.waiters {
            if self.waiting.remove(&call).is_some() {
                out.push(Action::Answer { call, status, why });
            }
        }
        self.maybe_start(now, out);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const J: &str = "0123456789abcdef";

    fn facts(job: TableJob) -> Facts {
        Facts {
            open_reconcile_done: true,
            job,
        }
    }

    fn unheld() -> Kernel {
        let mut k = Kernel::new();
        k.jobs.insert(
            J.into(),
            JobState::Unheld {
                offset: 7,
                shell_exit_seen: false,
            },
        );
        k
    }

    fn cb(call: CallId, now: Millis) -> Event {
        Event::Callback {
            call,
            job: J.into(),
            facts: facts(TableJob::Running),
            now,
        }
    }

    #[test]
    fn the_release_row_needs_both_keep_and_proof_in_either_order() {
        for proof_first in [false, true] {
            let mut k = Kernel::new();
            let a = k.step(Event::Kept {
                job: J.into(),
                notify_available: true,
                adopted: false,
            });
            assert_eq!(
                a,
                [Action::SendKeep {
                    job: J.into(),
                    notify: true
                }]
            );
            let keep = Event::KeepAnswered {
                job: J.into(),
                outcome: KeepOutcome::Succeeded,
                now: 10,
            };
            let (first, second) = if proof_first {
                (cb(1, 5), keep)
            } else {
                (keep, cb(1, 20))
            };
            let a1 = k.step(first);
            assert!(!a1.iter().any(|a| matches!(a, Action::CloseStream { .. })));
            let a2 = k.step(second);
            assert!(
                a1.iter()
                    .chain(&a2)
                    .any(|a| *a == Action::CloseStream { job: J.into() })
            );
            assert_eq!(k.job(J), Some(&JobState::Closing));
            let a = k.step(Event::StreamClosed {
                job: J.into(),
                offset: 3,
                shell_exit_seen: false,
                now: 30,
            });
            assert_eq!(a, [Action::StartReconcile { reconcile: 0 }]);
        }
    }

    #[test]
    fn a_proof_after_the_window_is_a_hint() {
        let mut k = Kernel::new();
        k.step(Event::Kept {
            job: J.into(),
            notify_available: true,
            adopted: false,
        });
        k.step(Event::KeepAnswered {
            job: J.into(),
            outcome: KeepOutcome::Succeeded,
            now: 0,
        });
        let a = k.step(cb(1, PROOF_WINDOW));
        assert_eq!(
            a,
            [Action::Answer {
                call: 1,
                status: 200,
                why: Why::Hint
            }]
        );
        let a = k.step(Event::Tick { now: PROOF_WINDOW });
        assert_eq!(
            a,
            [Action::Held {
                job: J.into(),
                why: HeldWhy::NoProof
            }]
        );
        assert_eq!(k.job(J), None, "held for good");
    }

    #[test]
    fn a_refused_notify_keeps_again_without_it() {
        let mut k = Kernel::new();
        k.step(Event::Kept {
            job: J.into(),
            notify_available: true,
            adopted: false,
        });
        let a = k.step(Event::KeepAnswered {
            job: J.into(),
            outcome: KeepOutcome::Refused,
            now: 0,
        });
        assert_eq!(
            a,
            [
                Action::SendKeep {
                    job: J.into(),
                    notify: false
                },
                Action::Held {
                    job: J.into(),
                    why: HeldWhy::Refused
                }
            ]
        );
        assert_eq!(k.job(J), None);
    }

    #[test]
    fn a_call_takes_an_unheld_job_and_lets_go_again() {
        let mut k = unheld();
        assert_eq!(
            k.step(Event::Taken { job: J.into() }),
            [Action::ReadInThenFollow {
                job: J.into(),
                from: 7
            }]
        );
        // While the call has it, its callback is asked again later.
        let a = k.step(cb(1, 0));
        assert_eq!(
            a,
            [Action::Answer {
                call: 1,
                status: 503,
                why: Why::Taken
            }]
        );
        assert_eq!(
            k.step(Event::PutBack {
                job: J.into(),
                running: true,
                offset: 40
            }),
            [Action::CloseFollow { job: J.into() }]
        );
        assert_eq!(
            k.job(J),
            Some(&JobState::Unheld {
                offset: 40,
                shell_exit_seen: false
            })
        );
        k.step(Event::Taken { job: J.into() });
        assert!(
            k.step(Event::PutBack {
                job: J.into(),
                running: false,
                offset: 50
            })
            .is_empty()
        );
        assert_eq!(k.job(J), None, "its end is recorded: held, as today");
    }

    #[test]
    fn callbacks_during_a_reconcile_share_the_next() {
        let mut k = unheld();
        assert_eq!(k.step(cb(1, 0)), [Action::StartReconcile { reconcile: 0 }]);
        assert!(k.step(cb(2, 10)).is_empty());
        assert!(k.step(cb(3, 20)).is_empty());
        let mut l = BTreeMap::new();
        l.insert(J.to_string(), Listed::Running);
        let a = k.step(Event::Listed {
            reconcile: 0,
            jobs: Some(l.clone()),
            now: 100,
        });
        assert_eq!(
            a,
            [Action::Answer {
                call: 1,
                status: 200,
                why: Why::Listed
            }]
        );
        // The next waits for the spacing.
        assert_eq!(k.next_wake(), Some(100 + RECONCILE_SPACING));
        assert!(k.step(Event::Tick { now: 500 }).is_empty());
        let a = k.step(Event::Tick {
            now: 100 + RECONCILE_SPACING,
        });
        assert_eq!(a, [Action::StartReconcile { reconcile: 1 }]);
        let a = k.step(Event::Listed {
            reconcile: 1,
            jobs: Some(l),
            now: 1200,
        });
        assert_eq!(
            a,
            [
                Action::Answer {
                    call: 2,
                    status: 200,
                    why: Why::Listed
                },
                Action::Answer {
                    call: 3,
                    status: 200,
                    why: Why::Listed
                }
            ]
        );
    }
}
