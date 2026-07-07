//! A process-scoped registry of per-path locks, so two concurrent tool dispatches that would mutate
//! the same file serialize against each other even across separate turns — and, when one
//! [`WriteLockRegistry`] is shared, across separate concurrent [`Agent`](crate::agent::Agent) runs
//! (e.g. two sessions in one `serve` process).
//!
//! The agent loop's own per-turn dispatch already groups same-target calls *within* one turn (see
//! `agent.rs`'s `group_runs`); this registry extends that guarantee across turn and session
//! boundaries, closing the gap where a fresh `HashMap` built each turn couldn't see a concurrently
//! in-flight turn on the same path.
//!
//! Deliberately minimal: an in-process map of async mutexes, not a full queue or scheduler. It only
//! serializes within one OS process — two separate `serve` processes against the same file would need
//! a filesystem advisory lock (e.g. `flock`), which is out of scope here.
//!
//! Entries are evicted once nothing references them any more: when a [`WriteLockGuard`] drops, it
//! checks whether the registry's map is the *only* remaining holder of that key's `Arc` (`strong_count
//! == 1`) — i.e. no other guard and no other in-flight `lock()` call still needs it — and if so removes
//! the map entry. The check-and-remove happens while holding the map's own lock, so a concurrent
//! `lock()` call for the same key can never observe (or create) a state where the entry is removed out
//! from under it: it either wins the race and clones the `Arc` before eviction runs (bumping the count
//! so eviction is skipped), or it runs after eviction and simply inserts a fresh mutex. This mirrors
//! pi's `file-mutation-queue.ts` `withFileMutationQueue`, which deletes its map entry in a `finally`
//! block only if the map still points at that same queue object — the identity check there and the
//! refcount check here both exist to avoid tearing down an entry a concurrent caller still needs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as SyncMutex};

// This registry has nothing to do with any one connection or executor (unlike `codex_websocket`,
// this crate's one exception to being otherwise executor-agnostic — see the crate doc comment), so
// the per-key lock stays `futures::lock::Mutex`, not `tokio::sync::Mutex`: a pure `Future`-based mutex
// with no ties to any particular executor, sharable across any caller regardless of what runtime (if
// any beyond tokio) drives it.
use futures::lock::{Mutex as AsyncMutex, OwnedMutexGuard};

type LockMap = Arc<SyncMutex<HashMap<String, Arc<AsyncMutex<()>>>>>;

/// A registry of per-key async mutexes. Cheap to construct and share via `Arc`. An entry exists only
/// while at least one caller holds or is waiting on the lock for its key; the last [`WriteLockGuard`]
/// to drop for a given key removes it (see the module doc comment), so the map stays bounded by the
/// number of *currently* contended paths rather than growing for the registry's whole lifetime.
#[derive(Default)]
pub struct WriteLockRegistry {
    locks: LockMap,
}

impl WriteLockRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire the lock for `key`, creating its entry if this is the first call for that key. The
    /// returned guard is `'static` (owns its `Arc`), so it can be held across `.await` points and moved
    /// into a spawned task; dropping it releases the lock and, if no one else is waiting on this key,
    /// evicts the key's entry from the registry.
    pub async fn lock(&self, key: &str) -> WriteLockGuard {
        // Scoped to a block (rather than a named `let` binding dropped explicitly) so the sync
        // `MutexGuard` — not `Send` — provably ends its lifetime before the `.await` below, keeping
        // this function's returned future `Send`.
        let mutex: Arc<AsyncMutex<()>> = {
            let mut locks = self
                .locks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Look up by borrowed `&str` first — the common case (a path already contended, or a
            // session repeatedly editing the same handful of files) hits here and skips
            // `key.to_string()`'s allocation entirely; only a genuinely new key pays for it, via
            // `entry`, below.
            match locks.get(key) {
                Some(m) => m.clone(),
                None => locks
                    .entry(key.to_string())
                    .or_insert_with(|| Arc::new(AsyncMutex::new(())))
                    .clone(),
            }
        };
        let guard = mutex.lock_owned().await;
        WriteLockGuard {
            guard: Some(guard),
            locks: self.locks.clone(),
            key: key.to_string(),
        }
    }

    #[cfg(test)]
    fn key_count(&self) -> usize {
        self.locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

/// RAII guard returned by [`WriteLockRegistry::lock`]. Releases the per-key mutex on drop and, if that
/// was the last outstanding reference to the key (no other guard or in-flight `lock()` call still holds
/// a clone of its `Arc`), removes the key's entry from the registry so it doesn't linger forever.
pub struct WriteLockGuard {
    guard: Option<OwnedMutexGuard<()>>,
    locks: LockMap,
    key: String,
}

impl Drop for WriteLockGuard {
    fn drop(&mut self) {
        // Drop the mutex guard first: this both unlocks the per-key mutex and releases this holder's
        // own strong reference to its `Arc`, so the strong-count check below reflects reality.
        self.guard.take();
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(arc) = locks.get(&self.key) {
            // Only the map itself still references this key's `Arc` — no other guard or queued
            // `lock()` caller holds a clone — so it's safe to evict. Checked while holding `locks`'
            // own lock, which every `lock()` call must also acquire to clone the `Arc`, so a
            // concurrent caller can't slip in between this check and the removal.
            if Arc::strong_count(arc) == 1 {
                locks.remove(&self.key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_key_serializes_concurrent_lockers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        let registry = Arc::new(WriteLockRegistry::new());
        let concurrent = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..5 {
            let registry = registry.clone();
            let concurrent = concurrent.clone();
            let max_concurrent = max_concurrent.clone();
            handles.push(tokio::spawn(async move {
                let _guard = registry.lock("shared.rs").await;
                let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                max_concurrent.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                concurrent.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            max_concurrent.load(Ordering::SeqCst),
            1,
            "lockers on the same key must never overlap"
        );
    }

    #[tokio::test]
    async fn aborting_a_lock_holder_mid_critical_section_releases_the_lock_only_at_that_point() {
        // pi's file-mutation-queue had a real bug here (`file-mutation-queue.test.ts`, "keeps
        // write/edit queue locked while an aborted write is still in flight"): releasing the lock the
        // instant a caller *requests* cancellation, even though the underlying write was still
        // physically in progress, let a second write interleave with the first's still-unflushed
        // bytes. Our guard's release is tied to `Drop`, not a separate "I'm done" signal, so a second
        // locker can only ever proceed once the first's guard has genuinely dropped — including when
        // that drop is triggered by cancellation partway through an `.await` inside the critical
        // section (simulating a future `spawn_blocking`-wrapped write). Pinning this so a refactor that
        // moved the guard's lifetime *outside* the actual work (e.g. dropping it before awaiting a
        // spawned blocking write's completion) would be caught here first.
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;

        let registry = Arc::new(WriteLockRegistry::new());
        let holder_registry = registry.clone();
        let holder_progressed_past_abort_point = Arc::new(AtomicBool::new(false));
        let flag = holder_progressed_past_abort_point.clone();
        let holder = tokio::spawn(async move {
            let _guard = holder_registry.lock("target.rs").await;
            // An internal await inside the locked section — the thing a `spawn_blocking`-wrapped write
            // would introduce. Long enough that the abort below always lands while still pending.
            tokio::time::sleep(Duration::from_millis(200)).await;
            flag.store(true, Ordering::SeqCst); // only reached if NOT aborted in time
        });
        tokio::time::sleep(Duration::from_millis(20)).await; // let the holder actually acquire first

        let waiter_registry = registry.clone();
        let waiter_acquired = Arc::new(AtomicBool::new(false));
        let waiter_flag = waiter_acquired.clone();
        let waiter = tokio::spawn(async move {
            let _guard = waiter_registry.lock("target.rs").await;
            waiter_flag.store(true, Ordering::SeqCst);
        });

        // The waiter must not have snuck in while the holder is still mid-critical-section.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !waiter_acquired.load(Ordering::SeqCst),
            "the waiter acquired the lock while the holder's critical section was still in flight"
        );

        holder.abort();
        let _ = holder.await; // a cancelled JoinHandle — expected
        assert!(
            !holder_progressed_past_abort_point.load(Ordering::SeqCst),
            "sanity check: the abort must have actually landed mid-sleep, not after it completed"
        );

        // Now that the holder's guard has dropped (via cancellation), the waiter proceeds promptly.
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiter must not deadlock once the holder's guard is dropped")
            .unwrap();
        assert!(waiter_acquired.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn different_keys_run_concurrently() {
        use std::time::Duration;

        let registry = Arc::new(WriteLockRegistry::new());
        let start = tokio::time::Instant::now();
        let a = {
            let registry = registry.clone();
            tokio::spawn(async move {
                let _guard = registry.lock("a.rs").await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            })
        };
        let b = {
            let registry = registry.clone();
            tokio::spawn(async move {
                let _guard = registry.lock("b.rs").await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            })
        };
        a.await.unwrap();
        b.await.unwrap();
        assert!(
            start.elapsed() < Duration::from_millis(180),
            "distinct keys must run concurrently, not serialize"
        );
    }

    #[tokio::test]
    async fn releasing_the_only_holder_evicts_the_key() {
        let registry = WriteLockRegistry::new();
        assert_eq!(registry.key_count(), 0);

        let guard = registry.lock("evict.rs").await;
        assert_eq!(registry.key_count(), 1);
        drop(guard);

        assert_eq!(
            registry.key_count(),
            0,
            "the map entry must be evicted once its only holder drops"
        );
    }

    #[tokio::test]
    async fn entry_survives_until_every_concurrent_holder_has_finished() {
        use std::time::Duration;

        let registry = Arc::new(WriteLockRegistry::new());

        let first_registry = registry.clone();
        let first = tokio::spawn(async move {
            let _guard = first_registry.lock("contended.rs").await;
            tokio::time::sleep(Duration::from_millis(80)).await;
        });
        // Let the first task actually acquire before the second queues up behind it.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(registry.key_count(), 1);

        let second_registry = registry.clone();
        let second = tokio::spawn(async move {
            let _guard = second_registry.lock("contended.rs").await;
            tokio::time::sleep(Duration::from_millis(80)).await;
        });
        // Give the second task time to queue up (clone the `Arc`) behind the first.
        tokio::time::sleep(Duration::from_millis(20)).await;

        first.await.unwrap();
        // The first holder finished, but the second is still queued/holding — must not be evicted yet.
        assert_eq!(
            registry.key_count(),
            1,
            "entry must not be evicted while a second caller is still using it"
        );

        second.await.unwrap();
        assert_eq!(
            registry.key_count(),
            0,
            "entry must be evicted once every concurrent holder has finished"
        );
    }
}
