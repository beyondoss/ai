use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A collision-resistant, roughly time-ordered session id: `<nanos>-<salt>-<seq>`.
///
/// The old `{nanos}{seq}` scheme was only process-local — two processes that started in the same
/// nanosecond produced the same id. We splice in a per-process **salt** seeded once from `RandomState`
/// (the OS-seeded hasher `HashMap` uses), so independent processes diverge even at the same instant;
/// within a process the monotonic `seq` breaks same-nanosecond ties. The nanosecond prefix keeps ids
/// sortable by creation. A UUID-shaped string isn't required — global uniqueness and ordering are.
pub(crate) fn new_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Seeded once per process. `RandomState` draws OS entropy at construction, so hashing a fixed value
    // with it yields a value that's stable for this process but random across processes.
    static SALT: OnceLock<u64> = OnceLock::new();
    let salt = *SALT.get_or_init(|| RandomState::new().hash_one(0xC0FFEEu64));

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{salt:016x}-{seq:x}")
}

/// Whether `id` is safe to embed directly in a filename component — alphanumeric, optionally with
/// `.`/`_`/`-` in the middle, starting and ending with a letter or digit. Matches pi's
/// `assertValidSessionId` (`^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$`); rejects anything that could
/// resolve to a path outside the sessions directory (a leading `/` or `..`, an embedded `/`, etc.) or
/// be empty. Lives here (not just in `main.rs`) so the WebSocket transport ([`crate::serve_ws`]) can
/// validate a client-supplied `?session_id=` before it ever becomes a filename component.
pub fn is_valid_session_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    let is_alnum = |b: u8| b.is_ascii_alphanumeric();
    match bytes {
        [] => false,
        [only] => is_alnum(*only),
        [first, .., last] => {
            is_alnum(*first)
                && is_alnum(*last)
                && bytes
                    .iter()
                    .all(|&b| is_alnum(b) || b == b'.' || b == b'_' || b == b'-')
        }
    }
}
