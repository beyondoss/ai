//! Hosted-tool calls in an OpenAI Responses answer, counted as the response streams past.
//!
//! OpenAI's Responses `usage` carries no tool counts (its API reference lists only token fields),
//! yet web search, file search, code interpreter containers and image generation are billed per
//! call or per session. The calls are output items (`{"type":"web_search_call",…}`), and a usage
//! tap that reads only the retained 64 KiB tail misses them: the items come first in the output,
//! and Codex-sized `response.completed` events (or non-stream bodies) are bigger than the tail.
//!
//! So a managed Responses response is fed here chunk by chunk. One SIMD `memmem` per chunk looks
//! for `_call"`, the end of an item type: an unescaped `"` can only close a JSON string, so a
//! generated text containing those letters is escaped and never matches. Each hit is then checked
//! structurally in a small window around it: the string must be the value of a `"type"` member,
//! and on a stream it must sit in a `response.output_item.done` event, so the same item's
//! `.added` event and its copy in `response.completed` are not counted again. A non-stream body
//! lists each item once. Hits near a chunk edge are finished from a carry of the last
//! [`WINDOW`] bytes, so memory is bounded whatever the response size.

use super::vendor::{IdStr, ServerTools, id_from};
use memchr::memmem::Finder;
use std::sync::LazyLock;

const NEEDLE: &[u8] = b"_call\"";
/// Bytes before a hit that may hold its event's type and the `"type"` key.
const LOOKBACK: usize = 256;
/// Bytes after a hit that may hold its `action` (web search) or `container_id` (code interpreter).
const LOOKAHEAD: usize = 256;
const WINDOW: usize = LOOKBACK + NEEDLE.len() + LOOKAHEAD;

static FINDER: LazyLock<Finder<'static>> = LazyLock::new(|| Finder::new(NEEDLE));

/// Counts hosted-tool items in one Responses response. See the module docs.
#[derive(Debug, Default)]
pub struct ToolTally {
    stream: bool,
    /// The last [`WINDOW`] bytes fed.
    carry: Vec<u8>,
    /// Hits in `carry` before this offset are already counted.
    scan_from: usize,
    pub tools: ServerTools,
    /// The first code-interpreter container seen.
    pub container_id: Option<IdStr>,
}

impl ToolTally {
    pub fn new(stream: bool) -> Self {
        Self {
            stream,
            ..Self::default()
        }
    }

    /// Feed the next chunk of the response.
    pub fn feed(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        let n = NEEDLE.len();
        let p = self.carry.len();
        let c = chunk.len();
        // Offsets are into the virtual `carry ++ chunk`.
        let mut first_deferred: Option<usize> = None;
        // Hits that start in the carry, or so early in the chunk that their lookback is in it:
        // read from the carry joined to the chunk's first `WINDOW` bytes.
        let mut chunk_from = 0;
        if p > 0 {
            let take = c.min(WINDOW);
            let mut joined = Vec::with_capacity(p + take);
            joined.extend_from_slice(&self.carry);
            joined.extend_from_slice(&chunk[..take]);
            chunk_from = LOOKBACK.min(c);
            for h in FINDER.find_iter(&joined) {
                if h < self.scan_from {
                    continue;
                }
                if h >= p + chunk_from {
                    break;
                }
                // `joined` is short of the lookahead only when it holds every byte so far.
                if h + n + LOOKAHEAD > joined.len() && take == c {
                    first_deferred.get_or_insert(h);
                    continue;
                }
                self.classify(&joined, h);
            }
        }
        for h in FINDER.find_iter(&chunk[chunk_from..]) {
            let h = h + chunk_from;
            if h + n + LOOKAHEAD > c {
                first_deferred.get_or_insert(p + h);
                continue;
            }
            let start = h.saturating_sub(LOOKBACK);
            self.classify(&chunk[start..h + n + LOOKAHEAD], h - start);
        }
        // Keep the last `WINDOW` bytes, and where in them counting resumes: the first deferred
        // hit, or a needle that may straddle this chunk's end.
        let total = p + c;
        let keep = WINDOW.min(total);
        let drop = total - keep;
        if c >= keep {
            self.carry.clear();
            self.carry.extend_from_slice(&chunk[c - keep..]);
        } else {
            self.carry.drain(..drop);
            self.carry.extend_from_slice(chunk);
        }
        let straddle = keep.saturating_sub(n - 1);
        self.scan_from = first_deferred.map_or(straddle, |d| (d - drop).min(straddle));
    }

    /// The response has ended: count the hits still waiting for a lookahead that never came.
    pub fn finish(&mut self) {
        let carry = std::mem::take(&mut self.carry);
        for h in FINDER.find_iter(&carry) {
            if h >= self.scan_from {
                self.classify(&carry, h);
            }
        }
        self.scan_from = 0;
    }

    /// Count the hit at `at` in `w` (its lookback and lookahead, as far as `w` reaches) if it is an
    /// item type this counts.
    fn classify(&mut self, w: &[u8], at: usize) {
        let Some(q) = memchr::memrchr(b'"', &w[..at]) else {
            return;
        };
        // The item type, without its `_call` suffix.
        let kind = &w[q + 1..at];
        let slot = match kind {
            b"web_search" => None,
            b"file_search" => Some(&mut self.tools.file_search),
            b"code_interpreter" => Some(&mut self.tools.code_execution),
            b"image_generation" => Some(&mut self.tools.image_generation),
            b"computer" => Some(&mut self.tools.computer_use),
            b"mcp" => Some(&mut self.tools.mcp),
            b"shell" | b"local_shell" => Some(&mut self.tools.shell),
            b"tool_search" => Some(&mut self.tools.tool_search),
            _ => return,
        };
        // The value of a `"type"` member: `"type"`, optional whitespace, `:`, optional whitespace.
        let before = w[..q].trim_ascii_end();
        let Some(before) = before.strip_suffix(b":") else {
            return;
        };
        if !before.trim_ascii_end().ends_with(b"\"type\"") {
            return;
        }
        let line_start = memchr::memrchr(b'\n', &w[..q]).map_or(0, |i| i + 1);
        if self.stream && memchr::memmem::find(&w[line_start..q], b"output_item.done\"").is_none() {
            return;
        }
        let mut after = &w[at + NEEDLE.len()..];
        // This item's members only: a stream's event ends at its line, a body's item before the
        // next item type.
        let end = if self.stream {
            memchr::memchr(b'\n', after)
        } else {
            FINDER.find(after)
        };
        after = &after[..end.unwrap_or(after.len())];
        match slot {
            Some(n) => *n = n.saturating_add(1),
            // Only a `search` action carries the per-call fee; `open_page` / `find_in_page` do
            // not. An action not in the window is counted as a search: never under-bill.
            None => {
                let page = memchr::memmem::find(after, b"\"action\"").is_some_and(|at| {
                    member_str(&after[at..], b"\"type\"")
                        .is_some_and(|t| t == b"open_page" || t == b"find_in_page")
                });
                let n = if page {
                    &mut self.tools.web_search_page
                } else {
                    &mut self.tools.web_search
                };
                *n = n.saturating_add(1);
            }
        }
        if kind == b"code_interpreter"
            && self.container_id.is_none()
            && let Some(id) = member_str(after, b"\"container_id\"")
        {
            self.container_id = std::str::from_utf8(id).ok().and_then(id_from);
        }
    }
}

/// The string value of the first `key` member in `w` (`key`, whitespace, `:`, whitespace, a
/// string with no escapes), or `None`.
fn member_str<'a>(w: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    let at = memchr::memmem::find(w, key)? + key.len();
    let rest = w[at..].trim_ascii_start().strip_prefix(b":")?;
    let rest = rest.trim_ascii_start().strip_prefix(b"\"")?;
    let end = memchr::memchr2(b'"', b'\\', rest)?;
    (rest[end] == b'"').then(|| &rest[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tally(stream: bool, body: &[u8], chunk: usize) -> ToolTally {
        let mut t = ToolTally::new(stream);
        for c in body.chunks(chunk.max(1)) {
            t.feed(c);
        }
        t.finish();
        t
    }

    const STREAM: &[u8] = b"event: response.created\n\
data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"output\":[]}}\n\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"in_progress\"}}\n\n\
data: {\"type\":\"response.web_search_call.completed\",\"item_id\":\"ws_1\"}\n\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"q\"}}}\n\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"ws_2\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"open_page\",\"url\":\"u\"}}}\n\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"id\":\"ci_1\",\"type\":\"code_interpreter_call\",\"status\":\"completed\",\"code\":\"print(1)\",\"container_id\":\"cntr_abc\"}}\n\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":3,\"item\":{\"id\":\"ig_1\",\"type\":\"image_generation_call\",\"status\":\"completed\"}}\n\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":4,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"name\":\"f\"}}\n\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":5,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"a \\\"web_search_call\\\" b\"}]}}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"id\":\"ws_1\",\"type\":\"web_search_call\"},{\"id\":\"ci_1\",\"type\":\"code_interpreter_call\"}],\"usage\":{}}}\n\n";

    /// Each finished item counts once (not its `.added` event, not its copy in
    /// `response.completed`), whatever the chunking, and a quoted kind in text never counts.
    #[test]
    fn a_stream_counts_each_finished_item_once_at_every_chunking() {
        for chunk in [1, 2, 3, 7, 26, 64, 255, 256, 257, 600, STREAM.len()] {
            let t = tally(true, STREAM, chunk);
            assert_eq!(t.tools.web_search, 1, "chunk {chunk}");
            assert_eq!(t.tools.web_search_page, 1, "chunk {chunk}");
            assert_eq!(t.tools.code_execution, 1, "chunk {chunk}");
            assert_eq!(t.tools.image_generation, 1, "chunk {chunk}");
            assert_eq!(t.tools.file_search, 0, "chunk {chunk}");
            assert_eq!(t.container_id.as_deref(), Some("cntr_abc"), "chunk {chunk}");
        }
    }

    /// A pretty-printed non-stream body lists each item once.
    #[test]
    fn a_body_counts_each_item_including_pretty_printed() {
        let body = br#"{
  "id": "resp_1",
  "output": [
    { "id": "ws_1", "type": "web_search_call", "status": "completed", "action": { "type": "search", "query": "x" } },
    { "id": "fs_1", "type": "file_search_call", "status": "completed" },
    { "id": "mcp_1", "type": "mcp_call", "name": "n" },
    { "id": "sh_1", "type": "shell_call" },
    { "id": "cu_1", "type": "computer_call" },
    { "id": "m", "type": "message", "content": [{ "type": "output_text", "text": "type: \"web_search_call\"" }] }
  ],
  "tools": [{ "type": "web_search" }],
  "include": ["web_search_call.action.sources"],
  "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
}"#;
        for chunk in [1, 5, 100, body.len()] {
            let t = tally(false, body, chunk);
            let want = ServerTools {
                web_search: 1,
                file_search: 1,
                mcp: 1,
                shell: 1,
                computer_use: 1,
                ..ServerTools::default()
            };
            assert_eq!(t.tools, want, "chunk {chunk}");
        }
    }

    /// Hits far apart across many large chunks (the steady state of a long stream).
    #[test]
    fn hits_spread_over_a_long_stream_are_all_counted() {
        let item = b"data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"ws_x\",\"type\":\"web_search_call\",\"status\":\"completed\"}}\n\n";
        let filler =
            b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"lorem ipsum dolor\"}\n\n";
        let mut s = Vec::new();
        for i in 0..50 {
            s.extend_from_slice(item);
            for _ in 0..(i % 7) * 10 {
                s.extend_from_slice(filler);
            }
        }
        for chunk in [1, 13, 300, 4096, 16384] {
            assert_eq!(tally(true, &s, chunk).tools.web_search, 50, "chunk {chunk}");
        }
    }
}
