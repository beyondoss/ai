//! The hot job answers, `list` and `read`, written straight into one buffer.
//!
//! Both used to build a `serde_json::Value` and then serialize the tree again: `list` at 1024
//! entries (512 rows answered) cost ~32 000 allocations and 3.2 MB for a 181 KB answer, because
//! every row was a `BTreeMap` with fifteen owned `String` keys and the whole tree was then walked
//! to bytes; `read` of 64 KiB cost ~400 KB for 64 KiB delivered (the window, a base64 `String`, the
//! `Value`, and the re-escaped copy of it). Both now serialize the answer's shape into one `Vec<u8>`
//! that is allocated once [`ROW_BUDGET`] per answered row (or one base64's worth for a `read`) and
//! grows only if a row is longer than that. Same work, one buffer, no tree. Measured before/after in
//! `ARCHITECTURE.md`, "Performance".
//!
//! **The wire does not change.** A `serde_json::Value` is a `BTreeMap` (this workspace does not
//! enable `preserve_order`), so every one of those answers carried its keys sorted by byte value;
//! each struct here declares its fields in exactly that order, and every value goes through the same
//! serde_json serializer, so numbers, escapes and the `null`s a `None` becomes are the same code
//! that wrote them before. `jobs::answer_tests` compares these bytes against the builders this
//! change replaced, over a fixture of every shape either op answers (running, exited, ended,
//! signalled, tombstoned, a terminal open and hung up, a held start, a keyed job, empty argv, and
//! strings carrying every byte that needs escaping), and the container shape — `{"jobs":[…],
//! "more":n}` with the count last and only when rows were left out — with it.

use std::fmt;

use base64::Engine as _;
use bytes::Bytes;
use serde::{Serialize, Serializer};
use serde_json::Error;

use crate::jobs::{JobCell, Key, row_of};
use crate::proc::locked;

/// What one listed row is given up front: the measured answer is 181 776 B for 512 rows of the
/// seeded ledger (355 B each), so this leaves every shipped shape one allocation for the whole
/// answer. A row longer than this makes the `Vec` grow once, geometrically — never one allocation
/// per row.
const ROW_BUDGET: usize = 384;

/// A terminal's `pty`, on a `read` answer and a `list` row: its size and whether its keeper still
/// holds it. Absent for a job with no terminal.
#[derive(Serialize)]
pub(crate) struct Pty {
    pub cols: u16,
    pub open: bool,
    pub rows: u16,
}

/// One `list` row. The field order is the wire's order: these are the keys the old `Value` map held
/// sorted, and serde writes a struct's fields in declaration order.
#[derive(Serialize)]
pub(crate) struct Row<'a> {
    pub args: &'a [String],
    /// The shell's exit code, `null` until it has one.
    pub code: Option<i32>,
    pub command: &'a str,
    /// When the job ended, 0 until it has.
    pub ended: u64,
    pub id: &'a str,
    pub kept: bool,
    /// The kind of keeping (`background`, …); `null` for a held start.
    pub kind: Option<&'a str>,
    pub label: &'a str,
    pub mark: &'a str,
    /// The `keep`'s number; `null` for a held start.
    pub num: Option<u64>,
    #[serde(rename = "outputEnd")]
    pub output_end: u64,
    #[serde(rename = "outputStart")]
    pub output_start: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pty: Option<Pty>,
    pub seen: u64,
    /// The signal that killed the shell, once the job has a code; absent for an exit, even of 137.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal: Option<i32>,
    pub started: u64,
    pub state: &'static str,
}

/// A `read` answer: the window, the job's status, and which bytes of the window are stderr.
#[derive(Serialize)]
pub(crate) struct ReadBody<'a> {
    pub code: Option<i32>,
    pub data: B64<'a>,
    pub end: u64,
    pub offset: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pty: Option<Pty>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal: Option<i32>,
    pub state: &'static str,
    /// Absent when the window holds no stderr byte: `[start, length]` on the output's own offsets.
    #[serde(skip_serializing_if = "<[(u64, u64)]>::is_empty")]
    pub stderr_ranges: &'a [(u64, u64)],
}

/// The answer a `read` of a job the table does not hold gets: nothing to read, at offset 0. There is
/// no `code`: the job is not there, so it never had one. Written as bytes because there is nothing
/// in it to vary; `jobs::answer_tests` holds it to the `Value` this replaced.
pub(crate) const GONE: &[u8] = br#"{"data":"","end":0,"offset":0,"state":"gone"}"#;

/// A base64 payload written into the answer as it goes, with no intermediate `String`.
pub(crate) struct B64<'a>(pub(crate) &'a [u8]);

impl fmt::Display for B64<'_> {
    /// serde_json's `collect_str` writes a `Display`'s output between quotes as the string's
    /// contents (and escapes it), so the encoder's bytes land in the answer's own buffer. The chunk
    /// is a stack buffer whose length is exactly one base64 group's worst case, so `encode_slice`
    /// cannot be short of room.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const RAW: usize = 3 * 1024;
        let mut buf = [0u8; RAW / 3 * 4];
        for chunk in self.0.chunks(RAW) {
            let n = base64::engine::general_purpose::STANDARD
                .encode_slice(chunk, &mut buf)
                .map_err(|_| fmt::Error)?;
            f.write_str(std::str::from_utf8(&buf[..n]).map_err(|_| fmt::Error)?)?;
        }
        Ok(())
    }
}

impl Serialize for B64<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

/// A `list` answer: `{"jobs":[…],"more":n}` — the count last, and only when rows were left out.
/// Rows go in the order the table gave them, skipping the ones the bound dropped.
pub(crate) fn list(
    cells: &[(Key, JobCell)],
    dropped: &[bool],
    more: usize,
) -> Result<Bytes, Error> {
    let rows = cells.len() - dropped.iter().filter(|d| **d).count();
    let mut buf = Vec::with_capacity(rows * ROW_BUDGET + 32);
    buf.extend_from_slice(b"{\"jobs\":[");
    let mut first = true;
    for (i, ((_, id), cell)) in cells.iter().enumerate() {
        if dropped[i] {
            continue;
        }
        let j = locked(cell);
        if !first {
            buf.push(b',');
        }
        first = false;
        serde_json::to_writer(&mut buf, &row_of(id, &j))?;
    }
    buf.push(b']');
    if more > 0 {
        buf.extend_from_slice(b",\"more\":");
        serde_json::to_writer(&mut buf, &more)?;
    }
    buf.push(b'}');
    Ok(Bytes::from(buf))
}

/// Serialize one answer into a buffer allocated once for it.
pub(crate) fn body<T: Serialize>(body: &T, capacity: usize) -> Result<Bytes, Error> {
    let mut buf = Vec::with_capacity(capacity);
    serde_json::to_writer(&mut buf, body)?;
    Ok(Bytes::from(buf))
}

/// What a caller answers when an answer could not be written. Unreachable in practice — serde_json
/// raises an error on a `Vec` write only through I/O, which a `Vec` does not do, and every key is a
/// `&'static str` — but a half-written answer must never be sent as if it were whole.
pub(crate) fn failed() -> crate::http::Resp {
    crate::http::error_response(500, "the answer could not be written")
}
