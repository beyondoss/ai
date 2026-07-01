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

use std::collections::HashMap;
use std::sync::{Arc, Mutex as SyncMutex};

// This crate is runtime-agnostic (no executor dependency in production code — see the crate doc
// comment's "no tokio in the library" note), so the per-key lock is `futures::lock::Mutex`, not
// `tokio::sync::Mutex`: it's a pure `Future`-based mutex with no ties to any particular executor.
use futures::lock::{Mutex as AsyncMutex, OwnedMutexGuard};

/// A registry of per-key async mutexes. Cheap to construct and share via `Arc`; entries accumulate
/// for the registry's lifetime (one per distinct path ever locked) rather than being cleaned up, since
/// a long-running `serve` process touches a bounded set of paths in practice.
#[derive(Default)]
pub struct WriteLockRegistry {
    locks: SyncMutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

impl WriteLockRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire the lock for `key`, creating it if this is the first call for that key. The returned
    /// guard is `'static` (owns its `Arc`), so it can be held across `.await` points and moved into a
    /// spawned task.
    pub async fn lock(&self, key: &str) -> OwnedMutexGuard<()> {
        let mutex: Arc<AsyncMutex<()>> = self
            .locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone();
        mutex.lock_owned().await
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
}
