//! Reading the command stream: one line at a time, bounded, with an oversized line reported rather than buffered.

use crate::serve::*;

/// Depth of a session's inbound command queue, shared by all three transports.
///
/// The counterpart to [`CONN_BACKLOG_MAX_BYTES`] on the way *in*. The consumer is a single
/// session loop that runs one command to completion — and some commands are genuinely slow (a
/// `list_sessions` walks the session directory) — so a client that pipelines faster than the loop drains
/// will always be able to get ahead of it. Left unbounded, that queue is a client-driven allocation with
/// no ceiling.
///
/// Bounded, but explicitly **not** shed-on-full the way the outbound side is (a connection that falls
/// too far behind is closed, to reconnect and be caught up; a `Pong` is simply dropped): a *dropped
/// command* is a correctness bug — the client is left waiting on a response that will never come, with no signal that
/// anything went wrong. So a full queue makes the producer wait instead. On the WebSocket path that
/// means the read loop stops pulling from the socket and TCP's own window carries the backpressure to
/// the client; on HTTP POST the handler awaits the same send; on stdio it simply pauses the stdin
/// reader. Deep enough that ordinary pipelining never
/// touches it, shallow enough that a flood can't grow memory without limit.
pub(crate) const IN_CHANNEL_BOUND: usize = 256;

/// The largest single command `serve` takes, on any transport: a stdin line here, a WebSocket message
/// in [`crate::serve_ws`]. Every transport feeds the same queue, and this is what bounds the cost of
/// any one queued command. Generous enough for a `prompt` carrying a large pasted body.
pub(crate) const MAX_COMMAND_BYTES: usize = 8 * 1024 * 1024;

/// One line read from stdin by [`next_stdin_line`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StdinLine {
    /// A complete line, without its `\n` (or `\r\n`).
    Text(String),
    /// A line whose bytes are not UTF-8 — answered with an error, never taken for the end of input.
    NotUtf8(std::str::Utf8Error),
    /// A line longer than the cap; it was read and discarded up to its end. Carries its length.
    TooLong(usize),
}

/// Read the next line from `reader`, holding at most `max` bytes of it: `Ok(None)` at end of input.
///
/// `AsyncBufReadExt::lines` did two wrong things here. A line that isn't UTF-8 comes back as an
/// `Err`, which the read loop could only take for the end of input, so one bad byte ended the
/// session with exit 0 and nothing said; and a line is buffered whole however long it is, so one
/// unterminated 300 MB line costs that much (and more) in memory. This one reads bytes, reports an
/// undecodable line as such, and stops buffering an oversized one at `max`, discarding the rest of
/// it as it streams past.
pub(crate) async fn next_stdin_line<R>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    max: usize,
) -> std::io::Result<Option<StdinLine>>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    buf.clear();
    // `Some(n)` once the line has passed `max`: `n` bytes seen so far, none of them kept.
    let mut discarded: Option<usize> = None;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            // End of input: a final line without a `\n` is still a line.
            return Ok(match discarded {
                Some(n) => Some(StdinLine::TooLong(n)),
                None if buf.is_empty() => None,
                None => Some(decode_stdin_line(buf)),
            });
        }
        let (chunk, newline) = match available.iter().position(|&b| b == b'\n') {
            Some(i) => (&available[..i], true),
            None => (available, false),
        };
        let taken = chunk.len() + usize::from(newline);
        match discarded.as_mut() {
            Some(n) => *n += chunk.len(),
            None if buf.len() + chunk.len() > max => {
                discarded = Some(buf.len() + chunk.len());
                *buf = Vec::new(); // give the memory back now, not at the next line
            }
            None => buf.extend_from_slice(chunk),
        }
        reader.consume(taken);
        if newline {
            return Ok(Some(match discarded {
                Some(n) => StdinLine::TooLong(n),
                None => decode_stdin_line(buf),
            }));
        }
    }
}

pub(super) fn decode_stdin_line(buf: &[u8]) -> StdinLine {
    let line = buf.strip_suffix(b"\r").unwrap_or(buf);
    match std::str::from_utf8(line) {
        Ok(s) => StdinLine::Text(s.to_owned()),
        Err(e) => StdinLine::NotUtf8(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every line `next_stdin_line` reads from `input`, through a 4-byte buffer so lines and the cap
    /// both straddle reads.
    async fn stdin_lines(input: &[u8], max: usize) -> Vec<StdinLine> {
        let mut reader = tokio::io::BufReader::with_capacity(4, input);
        let mut buf = Vec::new();
        let mut out = Vec::new();
        while let Some(line) = next_stdin_line(&mut reader, &mut buf, max).await.unwrap() {
            out.push(line);
        }
        out
    }

    /// An undecodable line and an oversized one are reported as such and reading goes on; only the
    /// end of input ends it.
    #[tokio::test]
    async fn a_bad_stdin_line_is_reported_and_reading_continues() {
        let text = |s: &str| StdinLine::Text(s.to_string());
        let lines = stdin_lines(b"one\r\nb\xffd\n0123456789abc\n\ntwo\nlast", 10).await;
        assert_eq!(lines[0], text("one"));
        assert!(matches!(lines[1], StdinLine::NotUtf8(_)), "{:?}", lines[1]);
        assert_eq!(lines[2], StdinLine::TooLong(13));
        assert_eq!(lines[3..], [text(""), text("two"), text("last")]);
        // A line exactly at the cap is still a line; an unterminated oversized tail is reported.
        assert_eq!(stdin_lines(b"0123456789\n", 10).await, [text("0123456789")]);
        assert_eq!(
            stdin_lines(b"ok\n0123456789abc", 10).await,
            [text("ok"), StdinLine::TooLong(13)]
        );
        assert!(stdin_lines(b"", 10).await.is_empty());
    }
}
