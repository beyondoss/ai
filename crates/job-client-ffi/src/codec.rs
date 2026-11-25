//! Bytes in, bytes out: the encoding of the kernel's state, events and actions. Every function here is
//! one call into `job_client` between a decode and an encode. A decision does not belong here:
//! `tests/no_decision.rs` fails on a branch, a loop or a comparison in this file.
//!
//! Layout. All integers little-endian. A mark is 33 bytes: its length, then 32 bytes. A state is the
//! 87 bytes `[phase, flow, mode, remove_on_exit, stdin_open, announced, key u64, mark, offset u64, pending u64, misses u32,
//! since_ms u64, until_ms u64, beat_s u32]`. Tags are the order the variants are declared in `job_client::step`, except the actions `Reconnecting` (10), `Started` (11), `Attached` (12) and `ShellExit` (13).

use job_client::step::{
    Action, Class, Event, Flow, Frame, Mark, Mode, Op, Phase, State, classify as k_classify,
    silence_ms, step as k_step,
};

/// The input is cut short or has an unknown tag or an invalid mark.
pub const BAD_INPUT: i32 = -1;
/// The output buffer is smaller than the answer.
pub const SHORT_OUT: i32 = -2;
/// The cursor's fields are not a cursor.
pub const BAD_CURSOR: i32 = -3;

type R<T> = Result<T, i32>;

struct Rd<'a>(&'a [u8]);

impl Rd<'_> {
    fn take<const N: usize>(&mut self) -> R<[u8; N]> {
        let (head, tail) = self.0.split_first_chunk::<N>().ok_or(BAD_INPUT)?;
        self.0 = tail;
        Ok(*head)
    }
    fn u8(&mut self) -> R<u8> {
        Ok(self.take::<1>()?[0])
    }
    fn u16(&mut self) -> R<u16> {
        self.take().map(u16::from_le_bytes)
    }
    fn u32(&mut self) -> R<u32> {
        self.take().map(u32::from_le_bytes)
    }
    fn i32(&mut self) -> R<i32> {
        self.take().map(i32::from_le_bytes)
    }
    fn u64(&mut self) -> R<u64> {
        self.take().map(u64::from_le_bytes)
    }
    fn flag(&mut self) -> R<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(BAD_INPUT),
        }
    }
    fn rest(&mut self) -> &[u8] {
        std::mem::take(&mut self.0)
    }
    fn mark(&mut self) -> R<Mark> {
        let len = self.u8()?;
        let body = self.take::<32>()?;
        match len {
            0 => Ok(Mark::none()),
            n => Mark::from_bytes(body.get(..usize::from(n)).ok_or(BAD_INPUT)?).ok_or(BAD_INPUT),
        }
    }
    fn mode(&mut self) -> R<Mode> {
        match self.u8()? {
            0 => Ok(Mode::Follow),
            1 => Ok(Mode::Watch),
            _ => Err(BAD_INPUT),
        }
    }
    fn state(&mut self) -> R<State> {
        let phase = match self.u8()? {
            0 => Phase::Idle,
            1 => Phase::Starting,
            2 => Phase::Opening,
            3 => Phase::Confirming,
            4 => Phase::Streaming,
            5 => Phase::Backoff,
            6 => Phase::ExitHeld,
            7 => Phase::Removing,
            8 => Phase::Done,
            _ => return Err(BAD_INPUT),
        };
        let flow = match self.u8()? {
            0 => Flow::Start,
            1 => Flow::Attach,
            2 => Flow::Remove,
            _ => return Err(BAD_INPUT),
        };
        Ok(State {
            phase,
            flow,
            mode: self.mode()?,
            remove_on_exit: self.flag()?,
            stdin_open: self.flag()?,
            announced: self.flag()?,
            key: self.u64()?,
            mark: self.mark()?,
            offset: self.u64()?,
            pending: self.u64()?,
            misses: self.u32()?,
            since_ms: self.u64()?,
            until_ms: self.u64()?,
            beat_s: self.u32()?,
        })
    }
    fn frame(&mut self) -> R<Frame> {
        match self.u8()? {
            0 => Ok(Frame::Attached {
                mark: self.mark()?,
                from: self.u64()?,
            }),
            1 => Ok(Frame::Started {
                mark: self.mark()?,
                stdin: self.flag()?,
            }),
            2 => Ok(Frame::Output { len: self.u64()? }),
            3 => Ok(Frame::Gap { n: self.u64()? }),
            4 => Ok(Frame::Heartbeat),
            5 => Ok(Frame::Exit {
                code: self.i32()?,
                signal: self.i32()?,
            }),
            6 => Ok(Frame::Bad),
            7 => Ok(Frame::ShellExit { code: self.i32()? }),
            _ => Err(BAD_INPUT),
        }
    }
    fn event(&mut self) -> R<Event> {
        match self.u8()? {
            0 => Ok(Event::Go),
            1 => Ok(Event::Answered {
                status: self.u16()?,
                retry_after_s: self.u32()?,
            }),
            2 => Ok(Event::Frame(self.frame()?)),
            3 => Ok(Event::Broke),
            4 => Ok(Event::Tick),
            5 => Ok(Event::ExitTaken),
            6 => Ok(Event::Close),
            7 => Ok(Event::Deadline),
            _ => Err(BAD_INPUT),
        }
    }
}

fn put_mark(v: &mut Vec<u8>, m: &Mark) {
    v.push(m.len);
    v.extend_from_slice(&m.b);
}

fn put_state(v: &mut Vec<u8>, s: &State) {
    v.push(s.phase as u8);
    v.push(s.flow as u8);
    v.push(s.mode as u8);
    v.push(u8::from(s.remove_on_exit));
    v.push(u8::from(s.stdin_open));
    v.push(u8::from(s.announced));
    v.extend_from_slice(&s.key.to_le_bytes());
    put_mark(v, &s.mark);
    v.extend_from_slice(&s.offset.to_le_bytes());
    v.extend_from_slice(&s.pending.to_le_bytes());
    v.extend_from_slice(&s.misses.to_le_bytes());
    v.extend_from_slice(&s.since_ms.to_le_bytes());
    v.extend_from_slice(&s.until_ms.to_le_bytes());
    v.extend_from_slice(&s.beat_s.to_le_bytes());
}

fn put_action(v: &mut Vec<u8>, a: &Action) {
    match a {
        Action::Start { key, silence_ms } => {
            v.push(0);
            v.extend_from_slice(&key.to_le_bytes());
            v.extend_from_slice(&silence_ms.to_le_bytes());
        }
        Action::OpenAttach {
            offset,
            mark,
            silence_ms,
        } => {
            v.push(1);
            v.extend_from_slice(&offset.to_le_bytes());
            put_mark(v, mark);
            v.extend_from_slice(&silence_ms.to_le_bytes());
        }
        Action::CloseStream => v.push(2),
        Action::Deliver { at, len, dropped } => {
            v.push(3);
            v.extend_from_slice(&at.to_le_bytes());
            v.extend_from_slice(&len.to_le_bytes());
            v.extend_from_slice(&dropped.to_le_bytes());
        }
        Action::DeliverExit {
            code,
            signal,
            dropped,
        } => {
            v.push(4);
            v.extend_from_slice(&code.to_le_bytes());
            v.extend_from_slice(&signal.to_le_bytes());
            v.extend_from_slice(&dropped.to_le_bytes());
        }
        Action::Sleep { until_ms } => {
            v.push(5);
            v.extend_from_slice(&until_ms.to_le_bytes());
        }
        Action::Remove { mark } => {
            v.push(6);
            put_mark(v, mark);
        }
        Action::Fail {
            fault,
            status,
            offset,
        } => {
            v.push(7);
            v.push(*fault as u8);
            v.extend_from_slice(&status.to_le_bytes());
            v.extend_from_slice(&offset.to_le_bytes());
        }
        Action::Finish => v.push(8),
        Action::Detached => v.push(9),
        Action::Started { mark, stdin } => {
            v.push(11);
            put_mark(v, mark);
            v.push(u8::from(*stdin));
        }
        Action::Attached { mark, from } => {
            v.push(12);
            put_mark(v, mark);
            v.extend_from_slice(&from.to_le_bytes());
        }
        Action::ShellExit { code } => {
            v.push(13);
            v.extend_from_slice(&code.to_le_bytes());
        }
        Action::Reconnecting { attempts, since_ms } => {
            v.push(10);
            v.extend_from_slice(&attempts.to_le_bytes());
            v.extend_from_slice(&since_ms.to_le_bytes());
        }
    }
}

/// The answer, copied to the front of `out`; its length.
fn finish(out: &mut [u8], answer: &[u8]) -> R<usize> {
    out.get_mut(..answer.len())
        .ok_or(SHORT_OUT)?
        .copy_from_slice(answer);
    Ok(answer.len())
}

/// `State::for_start`: `[key u64, mode, remove_on_exit, beat_s u32, stdin_open]` -> a state.
pub fn state_for_start(input: &[u8], out: &mut [u8]) -> R<usize> {
    let mut r = Rd(input);
    let s = State::for_start(r.u64()?, r.mode()?, r.flag()?, r.u32()?, r.flag()?);
    let mut v = Vec::new();
    put_state(&mut v, &s);
    finish(out, &v)
}

/// `State::for_attach`: `[mark, offset u64, mode, remove_on_exit, beat_s u32]` -> a state.
pub fn state_for_attach(input: &[u8], out: &mut [u8]) -> R<usize> {
    let mut r = Rd(input);
    let s = State::for_attach(r.mark()?, r.u64()?, r.mode()?, r.flag()?, r.u32()?);
    let mut v = Vec::new();
    put_state(&mut v, &s);
    finish(out, &v)
}

/// `State::for_cursor`: `[mode, remove_on_exit, beat_s u32, cursor bytes...]` -> a state.
pub fn state_for_cursor(input: &[u8], out: &mut [u8]) -> R<usize> {
    let mut r = Rd(input);
    let (mode, remove, beat) = (r.mode()?, r.flag()?, r.u32()?);
    let c = job_client::decode_bytes(r.rest()).map_err(|_| BAD_CURSOR)?;
    let s = State::for_cursor(&c, mode, remove, beat);
    let mut v = Vec::new();
    put_state(&mut v, &s);
    finish(out, &v)
}

/// `step`: `[state, event, now u64, draw u32]` -> `[state, n u8, action * n]`.
pub fn step(input: &[u8], out: &mut [u8]) -> R<usize> {
    let mut r = Rd(input);
    let (s, ev, now, draw) = (r.state()?, r.event()?, r.u64()?, r.u32()?);
    let (t, actions) = k_step(s, ev, now, draw);
    let mut v = Vec::new();
    put_state(&mut v, &t);
    v.push(u8::try_from(actions.len()).map_err(|_| SHORT_OUT)?);
    actions.iter().for_each(|a| put_action(&mut v, a));
    finish(out, &v)
}

/// `classify`: `[op, status u16]` -> `[class]` (0 go on, 1 retry, 2 + fault: terminal).
pub fn classify(input: &[u8], out: &mut [u8]) -> R<usize> {
    let mut r = Rd(input);
    let op = match r.u8()? {
        0 => Op::Start,
        1 => Op::Attach,
        2 => Op::Remove,
        _ => return Err(BAD_INPUT),
    };
    let v = match k_classify(op, r.u16()?) {
        Class::Ok => vec![0],
        Class::Retry => vec![1],
        Class::Terminal(f) => vec![2, f as u8],
    };
    finish(out, &v)
}

/// `silence_ms`: `[beat_s u32]` -> `[u64]`.
pub fn silence(input: &[u8], out: &mut [u8]) -> R<usize> {
    let ms = silence_ms(Rd(input).u32()?);
    finish(out, &ms.to_le_bytes())
}

/// The cursor a held state resumes from: `[state, scope len u8, scope, id...]` -> the cursor string.
pub fn cursor_of(input: &[u8], out: &mut [u8]) -> R<usize> {
    let mut r = Rd(input);
    let s = r.state()?;
    let scope_len = usize::from(r.u8()?);
    let rest = r.rest();
    let (scope, id) = rest.split_at_checked(scope_len).ok_or(BAD_INPUT)?;
    let c =
        job_client::Cursor::new(scope, id, s.mark.as_bytes(), s.offset).map_err(|_| BAD_CURSOR)?;
    finish(out, &job_client::encode_bytes(&c))
}
