//! Whether a relayed SSE stream has ended, for the client, on its protocol's terminal event.
//!
//! Each streaming protocol says where an answer ends: Chat Completions with `data: [DONE]`,
//! Messages with `message_stop`, Responses with `response.completed` (or `response.incomplete`, a
//! whole answer cut at its output limit). A client may close as soon as it has that event, and
//! the stock ones do: openai-python breaks out at `[DONE]` and closes the response, Codex closes
//! at `response.completed`. When that close reaches the gateway before the provider's own end of
//! stream, the request ends in a downstream error even though the client has the whole answer.
//! [`TerminalTracker`] is what lets `proxy` tell that close from a cancel: it reads the bytes the
//! client is sent and reports whether they end with a complete terminal event (the event and the
//! blank line that dispatches it).
//!
//! Only the last event matters, so the per-chunk cost is bounded by the chunk's last event, not
//! the chunk: a reverse scan from the end to the last event boundary, and a prefix match of at
//! most [`MAX_PATTERN`] bytes of that event. Nothing is buffered and nothing allocates; the state
//! is three bytes. An event split across chunks keeps its match state, and a `\n\n` boundary split
//! across chunks is seen through the carried count of trailing newlines.
//!
//! A terminal event is recognized by its first line: an `event:` name, or a `data:` payload that
//! starts with the sentinel or the event's `type`. That is how every provider and every
//! translation in this gateway writes them; an event that leads with an `id:` or a comment line
//! is not recognized, which only means the stream is judged as before (a close is a cancel).

/// First-line prefixes of a terminal event. Both `field: value` spellings, since SSE strips the
/// optional space after the colon.
const PATTERNS: [&[u8]; 14] = [
    b"data: [DONE]",
    b"data:[DONE]",
    b"event: message_stop",
    b"event:message_stop",
    b"data: {\"type\":\"message_stop\"",
    b"data:{\"type\":\"message_stop\"",
    b"event: response.completed",
    b"event:response.completed",
    b"event: response.incomplete",
    b"event:response.incomplete",
    b"data: {\"type\":\"response.completed\"",
    b"data:{\"type\":\"response.completed\"",
    b"data: {\"type\":\"response.incomplete\"",
    b"data:{\"type\":\"response.incomplete\"",
];

/// The longest pattern: no event needs more of its head read than this.
const MAX_PATTERN: usize = 35;

const ALL: u16 = (1 << PATTERNS.len()) - 1;

/// `alive` once the current event matched a pattern whole: it is a terminal event. `0` means it
/// matched none. A bit outside [`ALL`], so the verdict costs no field of its own.
const TERMINAL: u16 = 1 << 15;

/// `meta`'s low six bits: bytes of the current event's head matched so far (at most
/// [`MAX_PATTERN`]). The top two: `\n` bytes at the end of the stream so far (`\r` ignored),
/// saturating at 3. Two or more means the last event is complete (dispatched) and the next byte
/// starts a new one.
const MATCHED: u8 = 0x3F;
const NL_SHIFT: u32 = 6;

/// See the module docs. `Default` is a stream that has sent nothing.
///
/// Three bytes, alignment one: it lives on `proxy::RequestCtx`, which is touched once per response
/// chunk and has a size ceiling, and this fits in that struct's existing padding.
#[derive(Clone, Copy, Debug)]
pub struct TerminalTracker {
    /// Patterns the current event's head still matches (little-endian `u16`); [`TERMINAL`] or `0`
    /// once decided.
    alive: [u8; 2],
    /// See [`MATCHED`].
    meta: u8,
}

impl Default for TerminalTracker {
    fn default() -> Self {
        let mut t = Self {
            alive: ALL.to_le_bytes(),
            meta: 0,
        };
        // The stream's first byte starts an event.
        t.set_trailing_nl(2);
        t
    }
}

impl TerminalTracker {
    /// Feed the next bytes the client is sent.
    pub fn feed(&mut self, chunk: &[u8]) {
        let body_len = chunk
            .iter()
            .rposition(|&b| b != b'\n' && b != b'\r')
            .map_or(0, |i| i + 1);
        let trailing = count_nl(&chunk[body_len..]);
        if body_len == 0 {
            self.set_trailing_nl(self.trailing_nl().saturating_add(trailing));
            return;
        }
        let body = &chunk[..body_len];
        match last_event_start(body, self.trailing_nl()) {
            Some(start) => {
                self.alive = ALL.to_le_bytes();
                self.meta &= !MATCHED;
                self.match_head(&body[start..]);
            }
            // The chunk continues the event already in progress.
            None => self.match_head(body),
        }
        self.set_trailing_nl(trailing);
    }

    /// The bytes sent so far end with a whole terminal event.
    pub fn ended(&self) -> bool {
        u16::from_le_bytes(self.alive) == TERMINAL && self.trailing_nl() >= 2
    }

    fn trailing_nl(&self) -> u8 {
        self.meta >> NL_SHIFT
    }

    fn set_trailing_nl(&mut self, n: u8) {
        self.meta = (self.meta & MATCHED) | (n.min(3) << NL_SHIFT);
    }

    fn match_head(&mut self, bytes: &[u8]) {
        let mut alive = u16::from_le_bytes(self.alive);
        if alive == 0 || alive == TERMINAL {
            return;
        }
        let mut matched = usize::from(self.meta & MATCHED);
        for &b in bytes.iter().take(MAX_PATTERN - matched) {
            for (i, p) in PATTERNS.iter().enumerate() {
                if p.get(matched) != Some(&b) {
                    alive &= !(1 << i);
                }
            }
            matched += 1;
            if PATTERNS
                .iter()
                .enumerate()
                .any(|(i, p)| alive & (1 << i) != 0 && p.len() == matched)
            {
                alive = TERMINAL;
                break;
            }
            if alive == 0 {
                break;
            }
        }
        self.alive = alive.to_le_bytes();
        // `matched <= MAX_PATTERN < 64`: it fits the six bits.
        self.meta = (self.meta & !MATCHED) | (matched as u8 & MATCHED);
    }
}

fn count_nl(bytes: &[u8]) -> u8 {
    u8::try_from(memchr::memchr_iter(b'\n', bytes).count()).unwrap_or(u8::MAX)
}

/// Where the last event that starts in `body` begins: just after its last blank line, or at `0`
/// when the stream before `body` already ended on a boundary (`prev_nl` trailing newlines, two
/// for a blank line, one when `body` opens with the second). `None` when `body` only continues
/// an event. `body` does not end in a newline.
fn last_event_start(body: &[u8], prev_nl: u8) -> Option<usize> {
    for nl in memchr::memrchr_iter(b'\n', body) {
        let before = &body[..nl];
        let before = before.strip_suffix(b"\r").unwrap_or(before);
        if before.ends_with(b"\n") || (before.is_empty() && prev_nl >= 1) {
            return Some(nl + 1);
        }
    }
    (prev_nl >= 2).then_some(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ended(chunks: &[&[u8]]) -> bool {
        let mut t = TerminalTracker::default();
        for c in chunks {
            t.feed(c);
        }
        t.ended()
    }

    /// Every split of `stream` into two chunks gives the same verdict as the whole.
    fn every_split(stream: &[u8]) -> bool {
        let whole = ended(&[stream]);
        for i in 0..=stream.len() {
            let (a, b) = stream.split_at(i);
            assert_eq!(ended(&[a, b]), whole, "split at {i} of {stream:?}");
        }
        let bytes: Vec<&[u8]> = stream.chunks(1).collect();
        assert_eq!(ended(&bytes), whole, "byte by byte");
        whole
    }

    #[test]
    fn chat_ends_at_done() {
        let s = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{}}\n\ndata: [DONE]\n\n";
        assert!(every_split(s));
        assert!(every_split(b"data:[DONE]\r\n\r\n"));
        // Not dispatched yet: no blank line after it.
        assert!(!every_split(b"data: x\n\ndata: [DONE]\n"));
        assert!(!every_split(b"data: x\n\ndata: [DONE]"));
        // The usage chunk is not the end.
        assert!(!every_split(b"data: {\"choices\":[],\"usage\":{}}\n\n"));
    }

    #[test]
    fn messages_end_at_message_stop() {
        let s = b"event: message_delta\ndata: {\"type\":\"message_delta\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        assert!(every_split(s));
        assert!(every_split(b"data: {\"type\":\"message_stop\"}\n\n"));
        assert!(!every_split(
            b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n"
        ));
        assert!(!every_split(
            b"event: message_delta\ndata: {\"type\":\"message_delta\"}\n\n"
        ));
    }

    #[test]
    fn responses_end_at_completed_or_incomplete() {
        let big = format!(
            "event: response.output_text.delta\ndata: {{}}\n\nevent: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"instructions\":\"{}\"}}}}\n\n",
            "x".repeat(100_000)
        );
        let mut t = TerminalTracker::default();
        for c in big.as_bytes().chunks(4096) {
            t.feed(c);
        }
        assert!(t.ended(), "a terminal event far bigger than one chunk");
        assert!(every_split(
            b"data: {\"type\":\"response.incomplete\",\"response\":{}}\n\n"
        ));
        assert!(!every_split(
            b"event: response.failed\ndata: {\"type\":\"response.failed\"}\n\n"
        ));
    }

    #[test]
    fn an_event_after_the_terminal_one_is_the_end_now() {
        assert!(!every_split(b"data: [DONE]\n\ndata: {}\n\n"));
        assert!(!every_split(b"data: [DONE]\n\n: keepalive\n\n"));
        // Trailing blank lines after the terminal event change nothing.
        assert!(every_split(b"data: [DONE]\n\n\n\n"));
    }

    #[test]
    fn max_pattern_is_the_longest_pattern() {
        assert_eq!(PATTERNS.iter().map(|p| p.len()).max(), Some(MAX_PATTERN));
        assert_eq!(std::mem::size_of::<TerminalTracker>(), 3);
    }

    #[test]
    fn nothing_sent_is_not_an_end() {
        assert!(!ended(&[]));
        assert!(!ended(&[b"", b"\n\n"]));
    }
}
