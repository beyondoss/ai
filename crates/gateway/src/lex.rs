//! The one JSON lexer: strings, escapes and nesting depth, for every walker that reads a body
//! without parsing it (`peek`'s scanners and span reads, `translate`'s argument-end check).
//!
//! A [`Lexer`] is a resumable state machine, eight bytes, fed a body whole or chunk by chunk with
//! the same result. It knows where strings are and how deep the structure is, and reports each
//! byte outside a string, each string's opening and closing quote, and (to a consumer that keeps
//! one) a string's raw content, to a [`Visit`]or. Consumers are generic, so each walker's loop is
//! monomorphized and inlined: no indirect call per byte.
//!
//! What a member *is* (which string is a key, which `,` re-arms one) is left to each walker. Their
//! readings of malformed JSON differ, and each is part of its caller's contract: the streaming
//! scanner takes any `,` as the start of a key, the buffered one only a root `,`. One lexer, so a
//! string or an escape is never read two ways; several grammars, so none of those answers moves.
//!
//! Inside a string the lexer jumps to the next `"` or `\` with `memchr2`, so a multi-megabyte
//! prompt or base64 image costs a SIMD scan, not a branch per byte. A backslash escapes exactly
//! one byte for lexing: that is all it takes to find a string's end, and it keeps the end where
//! every walker before this one put it on malformed input (`"\u"` ends at its second quote).
//! Decoding, `\uXXXX` and surrogate pairs included, is [`unescape`] and [`unescapes_to`], run only
//! on a string that held a backslash, so the common string is compared as sent.
//!
//! Every function here indexes client bytes, so a slip is a panic on a hostile body: indexing and
//! unchecked arithmetic are denied, and the offsets are `get`s and wrapping adds of positions
//! inside a slice, which cannot wrap.
#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use std::ops::ControlFlow;

/// What a [`Visit`] callback answers: keep lexing, or stop at this byte.
pub type Flow = ControlFlow<()>;

/// Where the lexer is between two bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum State {
    /// Between strings: every byte is structure, part of a scalar, or whitespace.
    #[default]
    Outside,
    /// Inside a string that has held no backslash so far.
    Plain,
    /// Inside a string that has held a backslash.
    Escaped,
    /// Inside a string, just past a backslash: the next byte is content, whatever it is.
    Escape,
}

/// A resumable JSON lexer. See the module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lexer {
    /// Open `{` and `[` not yet closed. A `}` or `]` with none open leaves it at zero, so a
    /// stray close cannot make the root look nested.
    depth: u32,
    state: State,
}

/// The events of one lex. Positions are offsets into the chunk being fed.
pub trait Visit {
    /// A byte outside every string, other than the `"` that opens one. `depth` is the depth
    /// before the byte takes effect: a root `{` sees 0, and the `}` that closes it sees 1.
    fn byte(&mut self, at: usize, b: u8, depth: u32) -> Flow;

    /// A string opens at `at`, its quote.
    #[inline(always)]
    fn open(&mut self, at: usize, depth: u32) {
        let _ = (at, depth);
    }

    /// The raw bytes of the open string, escapes as sent, in order, in pieces split anywhere
    /// (at a chunk boundary, at a backslash). A consumer that keeps no string content ignores
    /// them; one that does sees every byte between the quotes exactly once.
    #[inline(always)]
    fn content(&mut self, raw: &[u8]) {
        let _ = raw;
    }

    /// The open string closes at `at`, its closing quote. `escaped`: it held a backslash, so its
    /// text is [`unescape`]d rather than read as sent.
    #[inline(always)]
    fn close(&mut self, at: usize, depth: u32, escaped: bool) -> Flow {
        let _ = (at, depth, escaped);
        Flow::Continue(())
    }
}

/// What lexing a string leaves: lex on from an offset past it, or `feed`'s answer.
type Step = ControlFlow<Option<usize>, usize>;

impl Lexer {
    /// A lexer before the first byte of a value.
    pub const fn new() -> Self {
        Lexer {
            depth: 0,
            state: State::Outside,
        }
    }

    /// A lexer just past a string's opening quote, at the root.
    pub const fn in_string() -> Self {
        Lexer {
            depth: 0,
            state: State::Plain,
        }
    }

    /// Open `{` and `[` not yet closed. Only the unit tests read it; the lex keeps the count.
    #[cfg(test)]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Lex the next chunk. `Some(end)` when a callback stopped the lex, `end` just past the byte
    /// it stopped on (the lexer is then past that byte); `None` when the whole chunk was lexed.
    #[inline]
    pub fn feed<V: Visit>(&mut self, bytes: &[u8], v: &mut V) -> Option<usize> {
        // The depth lives in a local for the walk, so a bracket costs a register, not a store.
        let mut depth = self.depth;
        let mut i = 0;
        let stopped = 'lex: {
            if self.state != State::Outside {
                // The string the last chunk ended inside. After a backslash, the first byte here
                // is the one it escaped: content, even a quote.
                let escaped = self.state != State::Plain;
                let from = usize::from(self.state == State::Escape);
                if from > bytes.len() {
                    break 'lex None;
                }
                self.state = State::Outside;
                match self.string(bytes, 0, from, escaped, depth, v) {
                    ControlFlow::Continue(next) => i = next,
                    ControlFlow::Break(stopped) => break 'lex stopped,
                }
            }
            // Bytes between strings. A string is lexed in place where it opens, so the loop
            // stays tight on structure-dense JSON (many short strings).
            while let Some(&b) = bytes.get(i) {
                let at = i;
                // `i` indexes a byte of the slice, so one past it cannot wrap.
                i = i.wrapping_add(1);
                // Each arm calls the visitor itself, so once it is inlined the byte's class is
                // known there and the visitor's own test of the byte folds away.
                let flow = match b {
                    b'"' => {
                        v.open(at, depth);
                        match self.string(bytes, i, i, false, depth, v) {
                            ControlFlow::Continue(next) => i = next,
                            ControlFlow::Break(stopped) => break 'lex stopped,
                        }
                        continue;
                    }
                    b'{' | b'[' => {
                        let flow = v.byte(at, b, depth);
                        depth = depth.saturating_add(1);
                        flow
                    }
                    b'}' | b']' => {
                        let flow = v.byte(at, b, depth);
                        depth = depth.saturating_sub(1);
                        flow
                    }
                    _ => v.byte(at, b, depth),
                };
                if flow.is_break() {
                    break 'lex Some(i);
                }
            }
            None
        };
        self.depth = depth;
        stopped
    }

    /// The content of the open string, which starts at `start`, from `i` (past `start` only
    /// when it begins with an escaped byte) through its closing quote if it is in `bytes`.
    /// `escaped`: the string has held a backslash; `depth`: the depth it sits at. The state is
    /// `Outside` coming in, and is set only when the chunk ends inside the string, so a string
    /// that closes in the chunk it opened in (nearly all of them) costs no state write.
    #[inline(always)]
    fn string<V: Visit>(
        &mut self,
        bytes: &[u8],
        start: usize,
        mut i: usize,
        mut escaped: bool,
        depth: u32,
        v: &mut V,
    ) -> Step {
        loop {
            let rest = bytes.get(i..).unwrap_or_default();
            let Some(k) = memchr::memchr2(b'"', b'\\', rest) else {
                v.content(bytes.get(start..).unwrap_or_default());
                self.state = if escaped {
                    State::Escaped
                } else {
                    State::Plain
                };
                return ControlFlow::Break(None);
            };
            // `k` is an offset inside `rest`, so `at` is a byte of `bytes`.
            let at = i.wrapping_add(k);
            if bytes.get(at) == Some(&b'"') {
                v.content(bytes.get(start..at).unwrap_or_default());
                let next = at.wrapping_add(1);
                return match v.close(at, depth, escaped) {
                    ControlFlow::Continue(()) => ControlFlow::Continue(next),
                    ControlFlow::Break(()) => ControlFlow::Break(Some(next)),
                };
            }
            escaped = true;
            let next = at.wrapping_add(2);
            if next > bytes.len() {
                // The backslash is the chunk's last byte: what it escapes comes next feed.
                v.content(bytes.get(start..).unwrap_or_default());
                self.state = State::Escape;
                return ControlFlow::Break(None);
            }
            i = next;
        }
    }
}

/// The end (exclusive) of the string whose opening quote is at `open`: just past its closing
/// quote. `None` when it does not close inside `b`.
pub fn string_end(b: &[u8], open: usize) -> Option<usize> {
    struct Close;
    impl Visit for Close {
        #[inline(always)]
        fn byte(&mut self, _: usize, _: u8, _: u32) -> Flow {
            Flow::Continue(())
        }
        #[inline(always)]
        fn close(&mut self, _: usize, _: u32, _: bool) -> Flow {
            Flow::Break(())
        }
    }
    let from = open.checked_add(1)?;
    let end = Lexer::in_string().feed(b.get(from..)?, &mut Close)?;
    from.checked_add(end)
}

/// The text of a JSON string's content (between its quotes), decoded as a strict parser decodes
/// it (RFC 8259, as `serde_json` reads a `String`): escapes resolved, a `\u` surrogate pair to
/// the one character it spells. `None` for anything such a parser refuses: an unknown escape, a
/// short or non-hex `\u`, an unpaired surrogate, a raw control character or quote, or bytes that
/// are not UTF-8.
pub fn unescape(raw: &[u8]) -> Option<String> {
    let mut out = Vec::with_capacity(raw.len());
    decode(raw, |piece| {
        out.extend_from_slice(piece);
        Flow::Continue(())
    })?;
    String::from_utf8(out).ok()
}

/// Whether `raw` (a JSON string's content) [`unescape`]s to exactly `want`, decoded on the fly
/// with nothing allocated. A decoded byte takes at most six raw ones (`\u00XX`), so a longer
/// `raw` (it may be a whole prompt) is refused unread.
pub fn unescapes_to(raw: &[u8], want: &str) -> bool {
    if raw.len() > want.len().saturating_mul(6) {
        return false;
    }
    let mut rest = want.as_bytes();
    // Decoded pieces that each match `want` in turn are valid UTF-8 together (they are `want`),
    // so no separate UTF-8 check is needed here, unlike in `unescape`.
    let matched = decode(raw, |piece| match rest.strip_prefix(piece) {
        Some(r) => {
            rest = r;
            Flow::Continue(())
        }
        None => Flow::Break(()),
    });
    matched.is_some() && rest.is_empty()
}

/// Walk a string's content, handing `sink` each run of bytes as sent and each escape decoded to
/// UTF-8. `None` on a byte or escape a strict parser refuses, or when `sink` stops.
fn decode(raw: &[u8], mut sink: impl FnMut(&[u8]) -> Flow) -> Option<()> {
    let mut rest = raw;
    loop {
        let run_end = memchr::memchr(b'\\', rest).unwrap_or(rest.len());
        let (run, tail) = rest.split_at(run_end);
        // A raw quote would have ended the string; a raw control character is not allowed in one.
        if run.iter().any(|&c| c < 0x20 || c == b'"') {
            return None;
        }
        if !run.is_empty() && sink(run).is_break() {
            return None;
        }
        let Some((_, after)) = tail.split_first() else {
            return Some(());
        };
        let (&e, after) = after.split_first()?;
        let mut buf = [0u8; 4];
        let (piece, after): (&[u8], &[u8]) = match e {
            b'"' => (b"\"", after),
            b'\\' => (b"\\", after),
            b'/' => (b"/", after),
            b'b' => (b"\x08", after),
            b'f' => (b"\x0c", after),
            b'n' => (b"\n", after),
            b'r' => (b"\r", after),
            b't' => (b"\t", after),
            b'u' => {
                let (c, after) = unicode_escape(after)?;
                (c.encode_utf8(&mut buf).as_bytes(), after)
            }
            _ => return None,
        };
        if sink(piece).is_break() {
            return None;
        }
        rest = after;
    }
}

/// The character a `\u` escape spells, `rest` starting at its hex digits: a lone code unit, or
/// a high surrogate and the `\u` low surrogate that must follow it. What follows it, too.
fn unicode_escape(rest: &[u8]) -> Option<(char, &[u8])> {
    let (hi, rest) = hex4(rest)?;
    match hi {
        0xD800..=0xDBFF => {
            let rest = rest.strip_prefix(b"\\u")?;
            let (lo, rest) = hex4(rest)?;
            if !(0xDC00..=0xDFFF).contains(&lo) {
                return None;
            }
            // Both units are inside their ranges, so the subtractions cannot wrap and the sum
            // is at most 0x10FFFF.
            let c =
                0x1_0000u32.wrapping_add((hi.wrapping_sub(0xD800) << 10) | lo.wrapping_sub(0xDC00));
            Some((char::from_u32(c)?, rest))
        }
        0xDC00..=0xDFFF => None,
        c => Some((char::from_u32(c)?, rest)),
    }
}

/// Four hex digits, either case, as a code unit.
fn hex4(rest: &[u8]) -> Option<(u32, &[u8])> {
    let (digits, rest) = rest.split_first_chunk::<4>()?;
    let mut n = 0u32;
    for &d in digits {
        let v = char::from(d).to_digit(16)?;
        // Four digits of at most 15: at most 0xFFFF.
        n = (n << 4) | v;
    }
    Some((n, rest))
}

#[cfg(test)]
#[path = "lex_tests.rs"]
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
mod tests;
