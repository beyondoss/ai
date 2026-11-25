//! Stateright model of the job handle: `job_client::step::step` against a model server.
//!
//! The handle's decisions are the REAL kernel's: every transition below calls `step` and carries out the
//! actions it returns, the way a shell does. The model owns only the environment: the server's one job
//! slot (output produced, the ring's oldest byte, the job's end, removal and id reuse by a new
//! incarnation), the connection (answered, cut, restarted), the frames the server sends in order, and the
//! consumer (takes the exit, closes). Time is normal (every step is called at `now = 0`, a tick at the
//! wake time it asked for), so the state space is finite.
//!
//! Each property is judged against the environment's own facts (which incarnation a connection reads,
//! where its frame sits in the output, what the server holds), never by asking the kernel what it should
//! have done.
//!
//! ## Properties
//!
//! - M1 no incarnation but the held one is ever streamed: a delivery comes from a connection to the
//!   incarnation the handle committed to (the cursor's mark, or the one its first echo named).
//! - M2 each byte is delivered at most once, in order, with no hole: a delivery starts exactly where the
//!   frame it carries sits in the server's output, and the bytes before it were delivered or reported as
//!   dropped.
//! - M3 every reopen asks from the end of what the consumer was given, for the held mark.
//! - M4 the exit is delivered once, only for the held incarnation, with every byte before it delivered
//!   or dropped (a trailing gap included).
//! - M5 a remove is sent only after the consumer took the exit, scoped to the held mark.
//! - M6 a start is sent again with the key of its first send; the server never runs two jobs for one
//!   handle.
//! - M7 an endpoint that does not echo marks is never streamed: the handle fails closed.
//! - M9 reconnecting never ends the handle: a cut, a tick or a busy answer is never followed by an end,
//!   however many failed opens came before.
//! - Witnesses (`sometimes`): the handle finishes after delivering the exit and removing the job; the front
//!   stays down past the notice threshold and the handle still streams again; the caller's deadline ends it;
//!   it ends `JobGone` when the id was reused; a gap is delivered in a chunk and with the exit; it reattaches
//!   after a cut and goes on.
//!
//! Mutations: `cargo test -p beyond-job-client --features <mut> --test step_model the_model` MUST fail for
//! each `mut_*` feature of `step.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use job_client::step::{
    self, Action, Event, Fault, Flow, Frame, Mark, Mode, Phase, State, step as kernel_step,
};
use stateright::{Checker, Model, Property};

/// The most output bytes the job writes.
const MAX_BYTES: u64 = 3;
/// The idempotency key of the handle.
const KEY: u64 = 7;

fn mark_of(inc: u8) -> Mark {
    Mark::from_bytes(&[b'a', b'0' + inc]).unwrap()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Req {
    None,
    Start { key: u64 },
    Attach { offset: u64, mark: Mark },
    Remove { mark: Mark },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Conn {
    /// The incarnation this connection reads.
    inc: u8,
    /// Where the next output frame sits in the incarnation's output.
    pos: u64,
    /// The first frame (the echo) was sent.
    echoed: bool,
    /// The exit frame was sent.
    exited: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Job {
    inc: u8,
    base: u64,
    end: u64,
    ended: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct St {
    c: State,
    req: Req,
    /// A request was sent and not answered: it is in flight.
    in_flight: bool,
    conn: Option<Conn>,
    /// The time the handle asked to be woken at.
    wake: Option<u64>,
    /// The server's one slot for the id.
    slot: Option<Job>,
    /// How many incarnations the id has had.
    incs: u8,
    /// The key the server created its job under.
    created_key: Option<u64>,
    // ghosts: the environment's facts about what the consumer was given.
    held: Option<u8>,
    cover: u64,
    exits: u8,
    exit_taken: bool,
    starts: u8,
    peak: u32,
    start_lost: bool,
    saw_gap_chunk: bool,
    saw_gap_exit: bool,
    reattached: bool,
    ended: Option<Ended>,
    bad: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ended {
    Finished,
    Fault(Fault),
    Detached,
}

const B_INC: u32 = 1 << 0;
const B_TILE: u32 = 1 << 1;
const B_OPEN: u32 = 1 << 2;
const B_EXIT: u32 = 1 << 3;
const B_REMOVE: u32 = 1 << 4;
const B_KEY: u32 = 1 << 5;
const B_ECHO: u32 = 1 << 6;
const B_FRESH: u32 = 1 << 7;
const B_GIVEUP: u32 = 1 << 8;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum A {
    /// The server answers the request with this status.
    Answer(u16),
    /// The server sends its next frame on the open connection.
    Frame,
    /// The stream breaks; or, with the request in flight, its answer is lost.
    Cut,
    /// The wake time comes.
    Tick,
    /// The consumer takes the exit.
    TakeExit,
    /// The consumer stops.
    Close,
    /// The caller's deadline passes.
    Deadline,
    /// The job writes a byte.
    Produce,
    /// The ring drops its oldest byte.
    Evict,
    /// The job exits.
    End,
    /// Someone removes the ended job.
    RemoveEnded,
    /// A new job starts under the same id.
    Reuse,
    /// The jitter draw.
    Draw(u32),
}

struct Cfg {
    flow: Flow,
    mode: Mode,
    remove_on_exit: bool,
    /// The endpoint predates marks.
    legacy: bool,
}

impl Cfg {
    fn init(&self) -> St {
        let (c, slot, held, incs) = match self.flow {
            Flow::Start => (
                State::for_start(KEY, self.mode, self.remove_on_exit, 0, false, false, false),
                None,
                None,
                0,
            ),
            _ => (
                State::for_attach(mark_of(0), 0, self.mode, self.remove_on_exit, 0),
                Some(Job {
                    inc: 0,
                    base: 0,
                    end: 0,
                    ended: false,
                }),
                Some(0),
                1,
            ),
        };
        St {
            c,
            req: Req::None,
            in_flight: false,
            conn: None,
            wake: None,
            slot,
            incs,
            created_key: None,
            held,
            cover: 0,
            exits: 0,
            exit_taken: false,
            starts: 0,
            peak: 0,
            start_lost: false,
            saw_gap_chunk: false,
            saw_gap_exit: false,
            reattached: false,
            ended: None,
            bad: 0,
        }
    }

    /// Feed the kernel an event and carry out what it says, as a shell does.
    fn feed(&self, s: &mut St, ev: Event, now: u64, draw: u32) {
        let before = s.c;
        let conn_before = s.conn.clone();
        let req_before = s.req.clone();
        let (c2, acts) = kernel_step(s.c, ev, now, draw);
        s.c = c2;
        if before.phase != Phase::Streaming && c2.phase == Phase::Streaming && s.held.is_none() {
            s.held = s.conn.as_ref().map(|k| k.inc);
        }
        if before.phase == Phase::Confirming && c2.phase == Phase::Streaming {
            s.reattached = s.reattached || before.flow == Flow::Attach;
        }
        if let (Event::Frame(Frame::Output { .. }), Mode::Watch) = (ev, self.mode) {
            // A watching handle delivers nothing; what it has counted is what the frames carried.
            if let Some(k) = &conn_before {
                s.cover = k.pos;
            }
        }
        for a in acts {
            match a {
                Action::Start { key, .. } => {
                    if key != KEY {
                        s.bad |= B_KEY;
                    }
                    if s.exits > 0 {
                        s.bad |= B_EXIT;
                    }
                    s.starts = s.starts.saturating_add(1);
                    s.req = Req::Start { key };
                    s.in_flight = true;
                }
                Action::OpenAttach { offset, mark, .. } => {
                    if offset != s.cover || s.held.map(mark_of) != Some(mark) && s.held.is_some() {
                        s.bad |= B_OPEN;
                    }
                    if s.exits > 0 {
                        // after the exit nothing reopens the stream
                        s.bad |= B_EXIT;
                    }
                    s.req = Req::Attach { offset, mark };
                    s.in_flight = true;
                }
                Action::CloseStream => {
                    s.conn = None;
                    s.in_flight = false;
                }
                Action::Deliver { at, len, dropped } => {
                    let k = conn_before
                        .clone()
                        .expect("a delivery without a connection");
                    if Some(k.inc) != s.held {
                        s.bad |= B_INC;
                    }
                    // The frame just sent sits at `k.pos - len` (it was advanced when it was sent).
                    if at != k.pos - len || at - dropped != s.cover {
                        s.bad |= B_TILE;
                    }
                    if dropped > 0 {
                        s.saw_gap_chunk = true;
                    }
                    s.cover = at + len;
                }
                Action::DeliverExit { dropped, .. } => {
                    let k = conn_before.clone().expect("an exit without a connection");
                    s.exits += 1;
                    if s.exits > 1 || Some(k.inc) != s.held {
                        s.bad |= B_EXIT;
                    }
                    let end = s.slot.as_ref().map(|j| j.end).unwrap_or(0);
                    if s.c.offset != end || (self.mode == Mode::Follow && s.cover + dropped != end)
                    {
                        s.bad |= B_EXIT;
                    }
                    if dropped > 0 {
                        s.saw_gap_exit = true;
                    }
                    s.cover = end;
                }
                Action::Sleep { until_ms } => s.wake = Some(until_ms),
                // Told to the consumer only: the model has no consumer-side state to change.
                Action::Reconnecting { .. }
                | Action::Started { .. }
                | Action::Attached { .. }
                | Action::ShellExit { .. } => {}
                // A kept start's refused echo: the model starts held jobs and kept ones alike and never refuses one.
                Action::Kill => {}
                Action::Remove { mark } => {
                    if !s.exit_taken || Some(mark) != s.held.map(mark_of) {
                        s.bad |= B_REMOVE;
                    }
                    s.req = Req::Remove { mark };
                    s.in_flight = true;
                }
                Action::Fail { fault, .. } => {
                    s.ended = Some(Ended::Fault(fault));
                }
                Action::Finish => s.ended = Some(Ended::Finished),
                Action::Detached => s.ended = Some(Ended::Detached),
            }
        }
        s.peak = s.peak.max(s.c.misses);
        // Reconnecting is a status: a cut, a tick or a busy answer never ends the handle, however
        // many came before.
        let busy = matches!(ev, Event::Broke | Event::Tick)
            || matches!(
                ev,
                Event::Answered {
                    status: 429 | 502 | 504,
                    ..
                }
            )
            || (matches!(ev, Event::Answered { status: 503, .. })
                && !matches!(req_before, Req::Start { .. }));
        if busy && s.ended.is_some() {
            s.bad |= B_GIVEUP;
        }
    }
}

impl Model for Cfg {
    type State = St;
    type Action = A;

    fn init_states(&self) -> Vec<St> {
        vec![self.init()]
    }

    fn actions(&self, s: &St, out: &mut Vec<A>) {
        if s.ended.is_some() {
            return;
        }
        // The consumer.
        if s.c.phase == Phase::ExitHeld {
            out.push(A::TakeExit);
        }
        out.push(A::Close);
        out.push(A::Deadline);
        // The shell's first request.
        if s.c.phase == Phase::Idle {
            out.push(A::Draw(0));
        }
        // The server's world.
        if let Some(j) = &s.slot {
            if !j.ended && j.end < MAX_BYTES {
                out.push(A::Produce);
            }
            if j.base < j.end {
                out.push(A::Evict);
            }
            if !j.ended {
                out.push(A::End);
            } else if self.flow != Flow::Start || s.held.is_some() {
                // The server keeps a keyed job until its start was confirmed (lane A2: the key survives).
                out.push(A::RemoveEnded);
            }
        } else if self.flow == Flow::Attach && s.incs < 2 {
            out.push(A::Reuse);
        }
        // A request in flight is answered, or its answer is lost.
        if s.in_flight && s.conn.is_none() {
            for st in [200u16, 503, 429, 404, 500] {
                out.push(A::Answer(st));
            }
            out.push(A::Cut);
        }
        // A break reported late, after the exit: the handle must let it pass.
        if s.conn.is_none() && !s.in_flight && matches!(s.c.phase, Phase::ExitHeld) {
            out.push(A::Cut);
        }
        // An open connection sends frames, or is cut.
        if let Some(k) = &s.conn {
            if !k.exited {
                out.push(A::Frame);
            }
            out.push(A::Cut);
        }
        if let Some(w) = s.wake
            && s.c.phase == Phase::Backoff
            && w == s.c.until_ms
        {
            out.push(A::Tick);
        }
    }

    fn next_state(&self, s: &St, a: A) -> Option<St> {
        let mut n = s.clone();
        match a {
            A::Draw(d) => {
                // Both ends of the jitter range are tried at every wait: see `Tick`.
                self.feed(&mut n, Event::Go, 0, d);
            }
            A::Close => {
                self.feed(&mut n, Event::Close, 0, 0);
                n.conn = None;
            }
            A::Deadline => {
                self.feed(&mut n, Event::Deadline, 0, 0);
                n.conn = None;
            }
            A::TakeExit => {
                n.exit_taken = true;
                self.feed(&mut n, Event::ExitTaken, 0, 0);
            }
            A::Produce => {
                n.slot.as_mut()?.end += 1;
            }
            A::Evict => {
                n.slot.as_mut()?.base += 1;
            }
            A::End => {
                n.slot.as_mut()?.ended = true;
            }
            A::RemoveEnded => {
                n.slot = None;
            }
            A::Reuse => {
                n.slot = Some(Job {
                    inc: n.incs,
                    base: 0,
                    end: 0,
                    ended: false,
                });
                n.incs += 1;
            }
            A::Tick => {
                let w = n.c.until_ms;
                n.wake = None;
                self.feed(&mut n, Event::Tick, w, 0);
            }
            A::Cut => {
                if n.in_flight && n.conn.is_none() {
                    // The answer is lost; a start has reached the server all the same.
                    if let Req::Start { key } = n.req {
                        self.server_creates(&mut n, key);
                        n.start_lost = true;
                    }
                    n.in_flight = false;
                } else {
                    n.conn = None;
                }
                self.feed(&mut n, Event::Broke, 0, u32::MAX);
            }
            A::Answer(status) => {
                n.in_flight = false;
                let req = n.req.clone();
                let mut status = status;
                match &req {
                    Req::Start { key } => {
                        if status == 200 {
                            match self.server_creates(&mut n, *key) {
                                true => {}
                                false => status = 409,
                            }
                        }
                    }
                    Req::Attach { .. } => {
                        if status == 200 && n.slot.is_none() {
                            status = 404;
                        }
                    }
                    Req::Remove { mark } => {
                        if status == 200 {
                            // A remove scoped to a mark removes only that incarnation, only if ended.
                            if let Some(j) = &n.slot
                                && mark_of(j.inc) == *mark
                                && j.ended
                            {
                                n.slot = None;
                            }
                        }
                    }
                    Req::None => return None,
                }
                if status == 200 && !matches!(req, Req::Remove { .. }) {
                    let j = n.slot.clone().expect("a job to read");
                    let pos = match &req {
                        Req::Attach { offset, .. } => (*offset).max(j.base),
                        _ => 0,
                    };
                    n.conn = Some(Conn {
                        inc: j.inc,
                        pos,
                        echoed: false,
                        exited: false,
                    });
                }
                self.feed(
                    &mut n,
                    Event::Answered {
                        status,
                        retry_after_s: if status == 503 { 1 } else { 0 },
                    },
                    0,
                    0,
                );
            }
            A::Frame => {
                let ev = self.next_frame(&mut n)?;
                self.feed(&mut n, ev, 0, 0);
                if n.c.phase != Phase::Streaming && n.c.phase != Phase::Confirming {
                    // the handle released the connection (CloseStream) or finished
                }
            }
        }
        // M7: an endpoint that does not echo marks is never streamed.
        if self.legacy && n.c.phase == Phase::Streaming {
            n.bad |= B_ECHO;
        }
        // Done: nothing is open.
        if n.ended.is_some() {
            n.conn = None;
            n.in_flight = false;
        }
        Some(n)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        macro_rules! always {
            ($name:expr, $bit:expr) => {
                Property::<Self>::always($name, |_, s: &St| s.bad & $bit == 0)
            };
        }
        let mut p = vec![
            always!("M1 no other incarnation is ever streamed", B_INC),
            always!("M2 each byte once, in order, no hole", B_TILE),
            always!(
                "M3 every reopen asks from the delivered end for the held mark",
                B_OPEN
            ),
            always!("M4 the exit is delivered once, complete", B_EXIT),
            always!(
                "M5 a remove follows the taken exit, scoped to the mark",
                B_REMOVE
            ),
            always!("M6 a start is sent again with its first key", B_KEY),
            always!("M7 no echo, no stream", B_ECHO),
            always!("M8 a job is never created twice for one handle", B_FRESH),
            always!("M9 reconnecting never ends the handle", B_GIVEUP),
        ];
        if self.legacy {
            p.push(Property::<Self>::sometimes(
                "W8 a legacy endpoint is refused with NotOffered",
                |_, s| s.ended == Some(Ended::Fault(Fault::NotOffered)),
            ));
            return p;
        }
        if self.remove_on_exit {
            p.push(Property::<Self>::sometimes(
                "W1 the handle finishes after the exit and the remove",
                |_, s| s.ended == Some(Ended::Finished) && s.exit_taken,
            ));
        } else {
            p.push(Property::<Self>::sometimes(
                "W2 the handle finishes without a remove",
                |_, s| s.ended == Some(Ended::Finished),
            ));
        }
        p.push(Property::<Self>::sometimes(
            "W3 the front is down past the notice and the handle streams again",
            |_, s| {
                s.peak > job_client::step::RECONNECTING_AFTER
                    && s.reattached
                    && s.c.phase == Phase::Streaming
            },
        ));
        p.push(Property::<Self>::sometimes(
            "W10 the caller's deadline ends it",
            |_, s| s.ended == Some(Ended::Fault(Fault::Deadline)),
        ));
        if self.flow == Flow::Attach {
            p.push(Property::<Self>::sometimes(
                "W4 the id was reused: JobGone",
                |_, s| s.ended == Some(Ended::Fault(Fault::JobGone)),
            ));
        }
        if self.mode == Mode::Follow {
            p.push(Property::<Self>::sometimes(
                "W5 a gap rides on a chunk",
                |_, s| s.saw_gap_chunk,
            ));
            p.push(Property::<Self>::sometimes(
                "W6 a trailing gap rides on the exit",
                |_, s| s.saw_gap_exit,
            ));
        }
        if self.flow == Flow::Start {
            p.push(Property::<Self>::sometimes(
                "W9 a lost start is sent again and the handle streams",
                |_, s| {
                    s.start_lost
                        && s.starts >= 2
                        && s.created_key.is_some()
                        && s.c.phase == Phase::Streaming
                },
            ));
        }
        p.push(Property::<Self>::sometimes(
            "W7 the handle reattaches after a cut and streams",
            |_, s| s.reattached && s.c.phase == Phase::Streaming,
        ));
        p
    }
}

impl Cfg {
    /// The server creates the job for a start, once per key. `false` if another key made it already.
    fn server_creates(&self, n: &mut St, key: u64) -> bool {
        match n.created_key {
            None => {
                n.created_key = Some(key);
                n.slot = Some(Job {
                    inc: n.incs,
                    base: 0,
                    end: 0,
                    ended: false,
                });
                n.incs += 1;
                true
            }
            Some(k) if k == key => true,
            Some(_) => {
                // A start under another key for an id that exists: the server runs a second job.
                n.bad |= B_FRESH;
                false
            }
        }
    }

    /// The frame the server sends next on the open connection, in the order the wire has.
    fn next_frame(&self, n: &mut St) -> Option<Event> {
        let j = n.slot.clone();
        let k = n.conn.as_mut()?;
        // The slot may have been removed or reused under the connection: the connection reads its own
        // incarnation, which the model keeps as the slot's while it lives.
        let j = j?;
        if j.inc != k.inc {
            // The incarnation this connection read is gone: it ends.
            return None;
        }
        if !k.echoed && !self.legacy {
            k.echoed = true;
            return Some(match n.req {
                Req::Start { .. } => Event::Frame(Frame::Started {
                    mark: mark_of(k.inc),
                    stdin: false,
                    pty: false,
                }),
                _ => Event::Frame(Frame::Attached {
                    mark: mark_of(k.inc),
                    from: k.pos,
                }),
            });
        }
        k.echoed = true;
        if k.pos < j.base {
            let g = j.base - k.pos;
            k.pos = j.base;
            return Some(Event::Frame(Frame::Gap { n: g }));
        }
        if k.pos < j.end {
            k.pos += 1;
            return Some(Event::Frame(Frame::Output { len: 1 }));
        }
        if j.ended {
            k.exited = true;
            return Some(Event::Frame(Frame::Exit { code: 0, signal: 0 }));
        }
        Some(Event::Frame(Frame::Heartbeat))
    }
}

fn check(cfg: Cfg) -> usize {
    let checker = cfg.checker().target_max_depth(44).spawn_bfs().join();
    checker.assert_properties();
    checker.unique_state_count()
}

#[test]
fn the_model_attach_follow_remove() {
    let n = check(Cfg {
        flow: Flow::Attach,
        mode: Mode::Follow,
        remove_on_exit: true,
        legacy: false,
    });
    eprintln!("attach/follow/remove: {n} states");
}

#[test]
fn the_model_attach_watch_keep() {
    let n = check(Cfg {
        flow: Flow::Attach,
        mode: Mode::Watch,
        remove_on_exit: false,
        legacy: false,
    });
    eprintln!("attach/watch/keep: {n} states");
}

#[test]
fn the_model_start_follow() {
    let n = check(Cfg {
        flow: Flow::Start,
        mode: Mode::Follow,
        remove_on_exit: true,
        legacy: false,
    });
    eprintln!("start/follow/remove: {n} states");
}

#[test]
fn the_model_legacy_endpoint() {
    let n = check(Cfg {
        flow: Flow::Attach,
        mode: Mode::Follow,
        remove_on_exit: false,
        legacy: true,
    });
    eprintln!("legacy: {n} states");
}

#[test]
fn the_kernel_is_what_the_model_calls() {
    // The model names `step::step` only through `kernel_step`: a quick check the symbol is the crate's.
    let _: fn(State, Event, u64, u32) -> (State, Vec<Action>) = step::step;
}
