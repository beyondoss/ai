//! `IdMap`: a `u64 -> u8` map the watch loop mutates **in place** while the request path reads it
//! lock-free. It backs the two sets that can grow large (deny, allowance).
//!
//! Why not clone-and-swap (`ArcSwap::rcu` over a `HashMap`): that copies the whole set on every
//! batch, so one revocation in a 500M-entry set is an O(N) copy and holds 2x the memory while it
//! runs. Here an update is O(K) in the batch size and allocates nothing.
//!
//! Why not a concurrent-map crate: measured against `std::HashMap` on 10M ids, `dashmap` and `scc`
//! cost +30-100% per lookup (a lock or epoch pin per read serialises the otherwise overlapping cache
//! misses), and `flurry`/`papaya` boxed entries cost 40-180 B each. This table is the minimum that
//! meets both: open addressing over flat atomic arrays, so a lookup is plain loads, and a slot
//! is 10 bytes.
//!
//! Layout: three parallel arrays. `ctl[i]` is one byte: `EMPTY`, `TOMB` (removed), or `FULL` plus 7
//! bits of the id's hash; `keys[i]` is the id and `vals[i]` its value. Probing is linear from a
//! Fibonacci hash of the id, and the table stays at most three-quarters full. A miss (the allow path:
//! nearly every request) reads only `ctl`, one byte a slot, so the arrays a lookup touches stay in
//! cache the way a swiss table's control bytes do; `keys` is read only when a tag matches and `vals`
//! only on a hit. A slot is 10 bytes. The two ids that collide with the sentinels, `0` and
//! `u64::MAX`, live in `special`.
//!
//! Concurrency. Writers take `writer` (uncontended: the watch loop is the only one), so there is
//! exactly one mutator at a time and it needs no CAS. A lookup returns a state the key really had (a slot
//! changes owner only through its `ctl` byte, the commit point). A new id is
//! published key, value, then `ctl` with `Release`; a reader that finds the tag with `Acquire`
//! therefore reads its key and value. A removed slot may later hold another id, so a hit re-reads the
//! key after the value and treats a change as "removed" (hits are the rare path; misses pay nothing).
//!
//! Growth. The table is fixed-size: a lookup loads the slots directly, with no pointer to chase and
//! no `ArcSwap` guard (its fence per read costs more than the lookup, serialising the cache misses a
//! plain probe overlaps). When used slots (live + removed) would pass three-quarters of the table, `insert`
//! returns [`Full`] and the owner publishes a bigger copy of the whole set ([`IdMap::grown`]) through
//! the same `Watched` slot a rescan uses. That copy is the one O(N) step, taken only when a set has
//! grown by half since it was seeded; a seed sizes the table for its entries up front. Removed slots
//! are reused by later inserts, so delete/re-add churn does not trigger it.

use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

const EMPTY: u64 = 0;
const TOMB: u64 = u64::MAX;
const MIN_SLOTS: usize = 16;
/// `ctl` bytes: free, removed, or `FULL | tag`.
const CTL_EMPTY: u8 = 0;
const CTL_TOMB: u8 = 1;
const CTL_FULL: u8 = 0x80;
/// 2^64 / golden ratio: the multiplier of Fibonacci hashing. **Deliberately not** a keyed hash
/// (SipHash): HashDoS resistance stops an attacker *choosing* keys that collide, and the ids stored
/// here are written by the control plane, never by a caller. A lookup only ever asks about an id
/// that `keyring.verify` already authenticated (an Ed25519 check over a token the control plane
/// minted), and the one exception, a signing key id read off an unverified token, only *reads* a
/// set the caller cannot add to. A keyed hash would cost ~4x on a lookup that runs on every managed
/// request (SipHash 6.21 ns vs Fx 1.49 ns single-threaded, 13.44 vs 3.40 at 16 threads, 1M entries).
/// If a path ever inserts an id a caller supplied, fix that first and the hash second.
const FIB: u64 = 0x9E37_79B9_7F4A_7C15;

struct Table {
    ctl: Box<[AtomicU8]>,
    keys: Box<[AtomicU64]>,
    vals: Box<[AtomicU8]>,
    shift: u32,
}

impl Table {
    fn new(slots: usize) -> Self {
        let slots = slots.max(MIN_SLOTS).next_power_of_two();
        Self {
            ctl: (0..slots).map(|_| AtomicU8::new(CTL_EMPTY)).collect(),
            keys: (0..slots).map(|_| AtomicU64::new(EMPTY)).collect(),
            vals: (0..slots).map(|_| AtomicU8::new(0)).collect(),
            shift: 64 - slots.trailing_zeros(),
        }
    }

    fn slots(&self) -> usize {
        self.ctl.len()
    }

    /// Home slot and the 7-bit tag, from the high bits of one multiply (the tag is the 7 bits just
    /// below the index bits).
    fn home(&self, k: u64) -> (usize, u8) {
        let h = k.wrapping_mul(FIB);
        (
            (h >> self.shift) as usize,
            CTL_FULL | ((h >> (self.shift - 7)) & 0x7f) as u8,
        )
    }

    fn get(&self, k: u64) -> Option<u8> {
        let mask = self.slots() - 1;
        let (mut i, tag) = self.home(k);
        loop {
            let c = self.ctl[i].load(Ordering::Acquire);
            if c == tag && self.keys[i].load(Ordering::Relaxed) == k {
                let v = self.vals[i].load(Ordering::Acquire);
                // The slot may have been removed and reused between the loads.
                return (self.keys[i].load(Ordering::Relaxed) == k).then_some(v);
            }
            if c == CTL_EMPTY {
                return None;
            }
            i = (i + 1) & mask;
        }
    }
}

/// The table is full: grow the set (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Full;

pub struct IdMap {
    table: Table,
    /// Ids `0` and `u64::MAX`: `0` = absent, else `0x100 | value`.
    special: [AtomicU16; 2],
    live: AtomicUsize,
    /// Slots that are not `EMPTY` (live + removed), guarded by the writer lock.
    writer: Mutex<usize>,
}

impl Default for IdMap {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl std::fmt::Debug for IdMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdMap").field("len", &self.len()).finish()
    }
}

impl IdMap {
    /// Sized for `n` entries at most three-quarters full.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            table: Table::new(n.saturating_mul(4) / 3 + 2),
            special: [AtomicU16::new(0), AtomicU16::new(0)],
            live: AtomicUsize::new(0),
            writer: Mutex::new(0),
        }
    }

    /// A map holding exactly `pairs`, sized for them. For seeds, which know their entries.
    pub fn from_pairs(pairs: &[(u64, u8)]) -> Self {
        let map = Self::with_capacity(pairs.len());
        for &(k, v) in pairs {
            map.place(k, v);
        }
        map
    }

    /// A copy of the live entries with room for `extra` more and a half again of what it holds: the
    /// owner publishes it in place of `self` when an insert reports [`Full`]. Always has room for
    /// `extra` new ids, whatever `self`'s removed slots hold.
    pub fn grown(&self, extra: usize) -> Self {
        let live = self.len();
        let new = Self::with_capacity(live + live / 2 + extra);
        let t = &self.table;
        for i in 0..t.slots() {
            if t.ctl[i].load(Ordering::Relaxed) >= CTL_FULL {
                new.place(
                    t.keys[i].load(Ordering::Relaxed),
                    t.vals[i].load(Ordering::Relaxed),
                );
            }
        }
        for (s, k) in [(0, EMPTY), (1, TOMB)] {
            let v = self.special[s].load(Ordering::Relaxed);
            if v != 0 {
                new.place(k, v as u8);
            }
        }
        new
    }

    pub fn len(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held by the slot arrays: 10 per slot.
    #[cfg(test)]
    pub fn heap_bytes(&self) -> usize {
        self.table.slots() * 10
    }

    pub fn get(&self, k: u64) -> Option<u8> {
        if let Some(s) = Self::special_index(k) {
            let v = self.special[s].load(Ordering::Acquire);
            return (v != 0).then_some(v as u8);
        }
        self.table.get(k)
    }

    pub fn contains(&self, k: u64) -> bool {
        self.get(k).is_some()
    }

    /// Insert or overwrite. [`Full`] (and nothing changed) if `k` is new and the table has no room.
    pub fn insert(&self, k: u64, v: u8) -> Result<(), Full> {
        let mut used = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        self.insert_locked(&mut used, k, v)
    }

    /// Remove; absent is a no-op.
    pub fn remove(&self, k: u64) {
        let _used = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(s) = Self::special_index(k) {
            if self.special[s].swap(0, Ordering::Release) != 0 {
                self.live.fetch_sub(1, Ordering::Relaxed);
            }
            return;
        }
        let t = &self.table;
        let mask = t.slots() - 1;
        let (mut i, tag) = t.home(k);
        loop {
            // The writer lock makes this thread the only mutator: Relaxed reads of its own writes.
            let c = t.ctl[i].load(Ordering::Relaxed);
            if c == tag && t.keys[i].load(Ordering::Relaxed) == k {
                t.ctl[i].store(CTL_TOMB, Ordering::Release);
                self.live.fetch_sub(1, Ordering::Relaxed);
                return;
            }
            if c == CTL_EMPTY {
                return;
            }
            i = (i + 1) & mask;
        }
    }

    fn special_index(k: u64) -> Option<usize> {
        match k {
            EMPTY => Some(0),
            TOMB => Some(1),
            _ => None,
        }
    }

    /// Insert into a table `with_capacity` sized for it (a seed or a copy), which cannot be full.
    fn place(&self, k: u64, v: u8) {
        let mut used = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        // Sized for every entry it is given, so `Full` cannot happen (the tests would catch it).
        let placed = self.insert_locked(&mut used, k, v);
        debug_assert!(placed.is_ok(), "table undersized");
    }

    fn insert_locked(&self, used: &mut usize, k: u64, v: u8) -> Result<(), Full> {
        if let Some(s) = Self::special_index(k) {
            let was = self.special[s].swap(0x100 | u16::from(v), Ordering::Release);
            if was == 0 {
                self.live.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(());
        }
        let t = &self.table;
        let mask = t.slots() - 1;
        let (mut i, tag) = t.home(k);
        let mut reusable = None;
        loop {
            let c = t.ctl[i].load(Ordering::Relaxed);
            if c == tag && t.keys[i].load(Ordering::Relaxed) == k {
                t.vals[i].store(v, Ordering::Release);
                return Ok(());
            }
            if c == CTL_TOMB && reusable.is_none() {
                reusable = Some(i);
            }
            if c == CTL_EMPTY {
                break;
            }
            i = (i + 1) & mask;
        }
        let slot = match reusable {
            Some(j) => j,
            None if (*used + 1) * 4 > t.slots() * 3 => return Err(Full),
            None => {
                *used += 1;
                i
            }
        };
        t.keys[slot].store(k, Ordering::Relaxed);
        t.vals[slot].store(v, Ordering::Relaxed);
        t.ctl[slot].store(tag, Ordering::Release);
        self.live.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// What the owner does on `Full`: publish a bigger copy.
    fn put(m: &mut IdMap, k: u64, v: u8) {
        if m.insert(k, v).is_err() {
            *m = m.grown(1);
            assert!(m.insert(k, v).is_ok(), "a grown map has room");
        }
    }

    #[test]
    fn matches_a_std_map_through_growth_churn_and_sentinels() {
        let mut m = IdMap::default();
        let mut model = HashMap::new();
        let mut x = 1u64;
        let mut grows = 0;
        for step in 0..60_000u64 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // A small id space forces collisions, reuse of removed slots and growth.
            let k = match (x >> 33) % 2048 {
                0 => 0,
                1 => u64::MAX,
                n => n * 7919,
            };
            if (x >> 20) & 3 == 0 {
                m.remove(k);
                model.remove(&k);
            } else {
                let v = (step % 200) as u8 + 1;
                let slots = m.table.slots();
                put(&mut m, k, v);
                grows += usize::from(m.table.slots() != slots);
                model.insert(k, v);
            }
            assert_eq!(m.len(), model.len());
        }
        assert!(grows > 0, "the run must exercise growth");
        for k in (0..2048u64).map(|n| n * 7919).chain([0, u64::MAX]) {
            assert_eq!(m.get(k), model.get(&k).copied(), "key {k}");
        }
    }

    #[test]
    fn a_seeded_map_holds_what_it_was_sized_for_and_then_reports_full() {
        let pairs: Vec<(u64, u8)> = (1..=1000u64).map(|k| (k, 1)).collect();
        let m = IdMap::from_pairs(&pairs);
        assert_eq!(m.len(), 1000);
        let slots = m.table.slots();
        // Overwrites and removals never need room; only a new id past half full does.
        assert!(m.insert(1, 2).is_ok());
        m.remove(2);
        assert!(m.insert(2, 1).is_ok(), "reuses the removed slot");
        let mut next = 1001;
        while m.insert(next, 1).is_ok() {
            next += 1;
        }
        assert_eq!(m.table.slots(), slots, "a full table never grows by itself");
        assert_eq!(
            m.len(),
            slots * 3 / 4,
            "full at three-quarters of the slots"
        );
        assert_eq!(m.get(next), None, "the refused id was not stored");
        let g = m.grown(1);
        assert_eq!(g.len(), m.len());
        assert!(g.insert(next, 1).is_ok());
        assert_eq!(g.get(1), Some(2));
    }

    #[test]
    fn a_reader_never_sees_a_wrong_value_while_the_writer_churns() {
        // Key k always carries value (k % 250) + 1, so any other value is a torn read.
        let m = Arc::new(IdMap::with_capacity(400));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let (m, stop) = (Arc::clone(&m), Arc::clone(&stop));
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        for k in 1..400u64 {
                            if let Some(v) = m.get(k) {
                                assert_eq!(v, (k % 250) as u8 + 1, "torn read of {k}");
                            }
                        }
                    }
                })
            })
            .collect();
        for round in 0..300u64 {
            for k in 1..400u64 {
                if (k + round) % 3 == 0 {
                    m.remove(k);
                } else {
                    assert!(m.insert(k, (k % 250) as u8 + 1).is_ok());
                }
            }
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().expect("reader panicked");
        }
    }
}
