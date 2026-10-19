use std::sync::Arc;

use super::*;

// ---- Layout and repo options --------------------------------------------------------------------

/// Which on-disk shape a [`SessionRepo`]'s sessions take.
#[derive(Clone, Default)]
pub enum Layout {
    /// One `<created_at>_<id>.jsonl` per session. The default, and what every local invocation uses.
    #[default]
    File,
    /// One `<id>/` directory of epoch segments per session, optionally sealed per tenant.
    Segmented { codec: Option<Arc<TenantCodec>> },
}

impl Layout {
    pub(super) fn codec(&self) -> Option<&Arc<TenantCodec>> {
        match self {
            Layout::File => None,
            Layout::Segmented { codec } => codec.as_ref(),
        }
    }

    pub(super) fn is_segmented(&self) -> bool {
        matches!(self, Layout::Segmented { .. })
    }
}

/// How a [`SessionRepo`] addresses and stores its sessions.
#[derive(Clone, Default)]
pub struct RepoOptions {
    pub layout: Layout,
    /// Prefixes every id this repo *mints* (a fork, a clone, an archive) with `<prefix>.`, so a
    /// derived session stays addressable on the shard its parent lives on. Ids the caller supplies
    /// are never rewritten.
    pub id_prefix: Option<String>,
}

/// Returned by every write once another owner has taken the session over — see [`Log`]'s module
/// comment. Not a transient failure: the store is poisoned and every later write fails too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Superseded;

impl std::fmt::Display for Superseded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("session was taken over by another writer")
    }
}

impl std::error::Error for Superseded {}

pub(super) fn superseded_error() -> std::io::Error {
    std::io::Error::other(Superseded)
}

/// A sealed line or file that would not open. Distinct from an ordinary read failure, because it
/// does not mean "the read stopped early" — it means these bytes are not what they claim to be, and
/// a reader that shrugged and carried on would present a *truncated* session as a complete one, which
/// the next write would then durably overwrite.
#[derive(Debug, Clone, Copy)]
pub(super) struct SealFailure(&'static str);

impl std::fmt::Display for SealFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for SealFailure {}

pub(super) fn seal_error(what: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, SealFailure(what))
}

pub(super) fn is_seal_failure(e: &std::io::Error) -> bool {
    e.get_ref().is_some_and(|inner| inner.is::<SealFailure>())
}

/// Whether an error is the takeover signal rather than an I/O failure — what a service-mode owner
/// checks to decide "end this session" instead of "retry".
pub fn is_superseded(e: &std::io::Error) -> bool {
    e.get_ref().is_some_and(|inner| inner.is::<Superseded>())
}
