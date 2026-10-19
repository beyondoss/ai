use std::io::{BufRead, Write};
use std::path::Path;

use agent_core::{ContentBlock, Message};
use serde::Serialize;

use super::*;

/// fsync a path's parent directory so a just-created or just-renamed entry in it is itself durable
/// (a file `sync_all` persists contents, not the dentry that names them). A best-effort no-op if the
/// path has no parent.
/// Parse one session-file line into an [`Entry`], recovering from one specific, narrow corruption
/// instead of dropping the whole line the way the caller's generic `Err` handling otherwise would
/// (pi-parity fix, pass 20): a hand-edited or externally corrupted `message` line with its `content`
/// field set to `null` (or removed outright). `agent_core::Message::content` is a plain
/// `Vec<ContentBlock>` with no `#[serde(default)]` — deliberately, since a *live*, in-process write
/// should never omit it — so neither shape deserializes as a valid `Message`, and ordinary recovery
/// would discard the entry's `role`/`id`/`parent_id`/tree position along with it, potentially orphaning
/// whatever later message chained its own `parent_id` off this one's `id`. pi's own recovery for the
/// same corruption keeps the entry and simply treats it as having empty content instead of dropping it.
///
/// This only ever retries that one specific shape — `"type":"message"` with `content` absent or
/// `null` — by re-parsing the line as a bare [`Value`](serde_json::Value), patching in an empty
/// `content` array, and re-attempting the real typed deserialize. A `content` field that's present but
/// some *other* invalid shape (a string, a number), a missing `role`, an unknown future `type`, or
/// genuinely unparseable JSON all fall straight through to the original error — this is a targeted
/// repair for one known corruption, not a generally permissive parse.
pub(super) fn parse_entry_lenient(path: &Path, line: &str) -> Result<Entry, serde_json::Error> {
    let original_err = match serde_json::from_str::<Entry>(line) {
        Ok(entry) => return Ok(entry),
        Err(e) => e,
    };
    (|| -> Option<Entry> {
        let mut value = serde_json::from_str::<serde_json::Value>(line).ok()?;
        let obj = value.as_object_mut()?;
        if obj.get("type").and_then(serde_json::Value::as_str) != Some("message") {
            return None;
        }
        match obj.get("content") {
            None | Some(serde_json::Value::Null) => {}
            Some(_) => return None,
        }
        obj.insert("content".to_string(), serde_json::Value::Array(Vec::new()));
        serde_json::from_value::<Entry>(value).ok()
    })()
    .inspect(|_| {
        tracing::warn!(
            path = %path.display(),
            "recovered session entry line with null/missing content by treating it as empty"
        );
    })
    .or_else(|| degrade_unknown_variants(line))
    .inspect(|_| {
        tracing::warn!(
            path = %path.display(),
            "read a session message holding content this build does not know as a degraded copy"
        );
    })
    .ok_or(original_err)
}

/// A message (or message edit) written by a newer build, holding a content block or stop reason this
/// build has no variant for, read as a degraded copy instead of being dropped: each unknown block
/// becomes a text block naming its type, an unknown stop reason becomes [`StopReason::Other`]. The
/// message keeps its id, parent and place in the tree, so nothing that chains off it is lost, and
/// the original line stays on disk untouched for the newer build to read in full (only a full
/// rewrite by this build — a `/new`, a fork — would write the degraded copy back).
///
/// `None` for anything else: an unknown entry `type`, a missing role, JSON that does not parse.
pub(super) fn degrade_unknown_variants(line: &str) -> Option<Entry> {
    use agent_core::StopReason;
    let mut value = serde_json::from_str::<serde_json::Value>(line).ok()?;
    let obj = value.as_object_mut()?;
    match obj.get("type").and_then(serde_json::Value::as_str) {
        Some("message" | "message_edit") => {}
        _ => return None,
    }
    let mut changed = false;
    if let Some(serde_json::Value::Array(blocks)) = obj.get_mut("content") {
        for block in blocks.iter_mut() {
            if serde_json::from_value::<ContentBlock>(block.clone()).is_ok() {
                continue;
            }
            let kind = block
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            *block = serde_json::to_value(ContentBlock::text(format!(
                "[content this version cannot read: {kind}]"
            )))
            .ok()?;
            changed = true;
        }
    }
    if let Some(reason) = obj.get_mut("stop_reason")
        && !reason.is_null()
        && serde_json::from_value::<StopReason>(reason.clone()).is_err()
    {
        *reason = serde_json::to_value(StopReason::Other).ok()?;
        changed = true;
    }
    if !changed {
        return None;
    }
    serde_json::from_value::<Entry>(value).ok()
}

/// Ceiling on one session-file line (one JSON entry) while loading. A legitimate entry — even one
/// carrying a large embedded payload (a big base64 image in a message) — stays orders of magnitude
/// under this; it exists to bound how much a corrupted or hand-edited file (a stray length delimiter,
/// concatenated lines) can make [`SessionStore::open`] allocate before the line is treated as
/// unreadable, the same recovery path already used for a torn write.
pub(super) const MAX_LINE_BYTES: usize = 64 * 1024 * 1024;

/// Ceiling on the opaque `data` payload of a single [`SessionStore::append_custom`] entry.
///
/// Comfortably above what the real producers write — a todo-list snapshot, a structured-output record,
/// a client's own bit of session metadata — and far below [`MAX_LINE_BYTES`], which is a *corruption*
/// backstop for the reader rather than a statement about what a well-behaved writer should send. The
/// gap between the two is deliberate: an entry this large is a caller bug or an abuse, not a big-but-
/// legitimate transcript line, and it should be refused at the point of writing rather than tolerated
/// on the way back in.
pub(super) const MAX_CUSTOM_ENTRY_BYTES: usize = 256 * 1024;

/// Read one `\n`-terminated line from `reader` into `buf` (cleared first, bytes only — the caller
/// validates UTF-8), via `fill_buf`/`consume` so a pathologically long or unterminated line is
/// processed in bounded chunks rather than by growing an unbounded `String` one byte at a time, the
/// way [`BufReader::lines`] would. `buf` never grows past [`MAX_LINE_BYTES`]; bytes beyond the cap are
/// still consumed from `reader` (so it lands correctly on the next line) but discarded, and the
/// returned flag reports whether that happened. `Ok(None)` at EOF with no more lines.
pub(super) fn read_capped_line(
    reader: &mut impl BufRead,
    buf: &mut Vec<u8>,
) -> std::io::Result<Option<bool>> {
    let mut unbounded = None;
    Ok(read_capped_line_limited(reader, buf, &mut unbounded)?.map(|line| line.oversized))
}

/// What one [`read_capped_line_limited`] call actually read.
pub(super) struct LineRead {
    /// The line was longer than [`MAX_LINE_BYTES`]; `buf` holds only its prefix.
    pub(super) oversized: bool,
    /// The line ended with a `\n` rather than running into the end of the range. A `false` here at
    /// the end of a segment's range is a torn write (see [`SegReader::next_line`]).
    pub(super) terminated: bool,
    /// Bytes taken off `reader`, newline included — how far the range cursor advanced.
    pub(super) consumed: u64,
}

/// [`read_capped_line`] bounded to `limit` more bytes (`None` = to EOF). Decrementing the caller's
/// budget here, rather than wrapping the reader in a fresh `Take` per line, is what lets one buffered
/// reader walk several ranges of one file without re-buffering at every boundary.
pub(super) fn read_capped_line_limited(
    reader: &mut impl BufRead,
    buf: &mut Vec<u8>,
    limit: &mut Option<u64>,
) -> std::io::Result<Option<LineRead>> {
    buf.clear();
    let mut total = 0usize;
    let mut consumed_total = 0u64;
    let mut saw_any_byte = false;
    let mut terminated = false;
    loop {
        if *limit == Some(0) {
            break;
        }
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        let available = match *limit {
            Some(rem) => {
                &available[..available
                    .len()
                    .min(usize::try_from(rem).unwrap_or(usize::MAX))]
            }
            None => available,
        };
        saw_any_byte = true;
        let (chunk, hit_newline) = match available.iter().position(|&b| b == b'\n') {
            Some(pos) => (&available[..pos], true),
            None => (available, false),
        };
        total += chunk.len();
        let room = MAX_LINE_BYTES.saturating_sub(buf.len());
        let take = chunk.len().min(room);
        buf.extend_from_slice(&chunk[..take]);
        let consumed = if hit_newline {
            chunk.len() + 1
        } else {
            chunk.len()
        };
        reader.consume(consumed);
        consumed_total += consumed as u64;
        if let Some(rem) = limit {
            *rem = rem.saturating_sub(consumed as u64);
        }
        if hit_newline {
            terminated = true;
            break;
        }
    }
    if !saw_any_byte {
        return Ok(None);
    }
    Ok(Some(LineRead {
        oversized: total > MAX_LINE_BYTES,
        terminated,
        consumed: consumed_total,
    }))
}

/// Serialize one entry as a single JSON line (no embedded newlines — `serde_json` escapes them).
///
/// Straight to the writer, in one pass. It used to go `Entry` → owned `Value` tree → `String` → `w`,
/// which is the same two-pass shape `serve::event_frame` was fixed out of and `benches/serve_events.rs`
/// was written to condemn — the fix just never reached this module. An entry carrying a 100 KB tool
/// result built a 100 KB `Value` tree solely to throw it away. See `benches/persistence.rs`.
pub(super) fn write_line(w: &mut impl Write, entry: &Entry) -> std::io::Result<()> {
    serde_json::to_writer(&mut *w, entry).map_err(std::io::Error::other)?;
    w.write_all(b"\n")
}

/// The write-side mirror of [`Entry::Message`] that **borrows** its `Message` rather than owning one.
///
/// Byte-identical on the wire (same internal tag, same field names, same `flatten`) — it exists purely
/// so [`SessionStore::append_new`] can persist a message without cloning it first. The in-memory `Node`
/// legitimately owns its copy; the serializer never needed one, so a turn whose tool result is a 100 KB
/// `read` output was copying those bytes once just to hand them to `serde`.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum EntryRef<'a> {
    Message {
        id: Option<&'a str>,
        parent_id: Option<&'a str>,
        timestamp: u64,
        #[serde(flatten)]
        message: &'a Message,
    },
}

/// [`write_line`]'s borrowing counterpart — same single-pass write, no owned `Entry` required.
pub(super) fn write_line_ref(w: &mut impl Write, entry: &EntryRef<'_>) -> std::io::Result<()> {
    serde_json::to_writer(&mut *w, entry).map_err(std::io::Error::other)?;
    w.write_all(b"\n")
}
