//! JSONL session persistence: a per-session append log and a multi-session repository.
//!
//! Each session is one newline-delimited JSON file: a header line carrying [`SessionMeta`] (stable id,
//! cwd, model, timestamps, optional title and `parent` lineage), then one line per conversation
//! message. A turn **appends** its new messages — O(new), not O(transcript) — and a torn final line
//! (a crash mid-append) is dropped on load and cut off the file by the next writable open, so at most
//! the last entry is lost and the next append never lands glued onto the fragment; an append that
//! fails part-way is rolled back to the length it started from. More generally, any single line that
//! fails to deserialize — not only the last one — is skipped rather than truncating the scan, and an
//! entry whose parent was such a line re-chains to the entry before the gap (`rechain_orphans`), so
//! the damage costs only the unreadable entries, never the history before them. A message holding a
//! content block or stop reason from a newer build is read as a degraded copy rather than skipped
//! (`degrade_unknown_variants`).
//!
//! **Tree-shaped history.** Every message entry carries an `id` and a `parent_id`, so a session's
//! history is a tree, not just a line — the "active path" (what `Session.messages` holds) is the
//! chain from the root to the active tip. In the common case (no branching) that's just every message
//! in file order, identical to a flat file. A `Leaf` entry marks a navigation event — the active tip
//! moved to some other message's id — so a later `open()` can tell the active branch apart from "the
//! last line in the file" once one exists (see [`SessionStore::switch_active`]). Branching is
//! append-only: navigating never deletes anything.
//!
//! `agent_core::Message`/`Session` stay flat and untouched — the tree lives entirely at this storage
//! layer, in the `id`/`parent_id` this module stamps onto each [`Entry::Message`] line.
//!
//! **Compaction vs. branching.** [`SessionStore::rewrite`] (used by compaction, which replaces the
//! active path's messages with a shorter summarized prefix) is still destructive **to the active
//! path's own entries** — the prior flat-history simplification stands there. But it now preserves
//! every entry that belongs to some *other* branch (created by navigating away and appending
//! elsewhere): those are re-written verbatim alongside the new, freshly-compacted active path, rather
//! than being discarded along with it. Net effect: "everything not kept is deleted" becomes
//! "everything not on the compacted active path is deleted; other branches persist" — a real, but
//! bounded and honest, weakening of the original simplification, not a full reversal of it. One known
//! edge case: an *off-branch* entry whose `parent_id` points at an active-path message that compaction
//! just summarized away is not deleted (its content survives on disk), but its link to the root is
//! severed — orphaned, not lost. No feature reconnects it today; a future branch-listing feature
//! should treat an unreachable `parent_id` as a second kind of root.
//!
//! **Legacy migration.** A file written before this module tracked ids has `id`/`parent_id` absent on
//! every message line (`#[serde(default)]` reads them as `None`). `SessionStore::open` synthesizes ids
//! for those in memory — position-based, each chained to the message immediately before it — so the
//! reconstructed active path is identical to the old flat-file behavior. Synthesized ids are never
//! persisted back; they're a derived field, following the same pattern already used for `updated_at`/
//! `message_count`/`preview`. No `VERSION` bump: `id`/`parent_id` are additive `#[serde(default)]`
//! fields, so old and new binaries still round-trip through each other's files.
//!
//! A [`SessionRepo`] is a directory of those files: it lists sessions (newest first), creates and
//! opens them, deletes, and **forks** one (copy its prefix into a new session that links back via
//! `parent`) — the headless-relevant slice of pi's session tree. (A "fork" is a *new file*, a
//! session-level split; the tree above is *within* one file — the two are independent mechanisms.)

mod entry;
mod fsutil;
mod ids;
mod journal_key;
mod layout;
mod lines;
mod listing;
mod lock;
mod log;
mod log_read;
mod lookup;
mod meta;
mod repo;
mod seal;
mod segment;
mod store;
mod store_compact;
mod store_tree;
mod store_write;
mod trash;

pub use self::entry::*;
pub(crate) use self::fsutil::*;
pub use self::ids::*;
use self::journal_key::*;
pub use self::layout::*;
use self::lines::*;
pub(crate) use self::listing::*;
pub use self::lock::*;
use self::log::*;
use self::log_read::*;
pub use self::lookup::*;
pub use self::meta::*;
pub use self::repo::*;
pub use self::seal::*;
use self::segment::*;
pub use self::store::*;
pub use self::trash::*;

#[cfg(test)]
mod tests;
