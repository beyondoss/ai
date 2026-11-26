//! The server-wide count of requests in flight, and admission against [`crate::Options::max_in_flight`].
//!
//! Under `--cfg beyond_loom` the atomic is loom's, so the models below explore every interleaving of
//! the real code (`mise run test:loom`).

#[cfg(beyond_loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(beyond_loom))]
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct InFlightCount(AtomicUsize);

impl InFlightCount {
    pub fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    pub fn get(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }

    pub fn enter(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }

    /// [`Self::enter`] unless `max` are in already. One compare-and-swap decides, so concurrent
    /// callers can never take the count past `max`.
    pub fn enter_within(&self, max: usize) -> bool {
        let mut n = self.0.load(Ordering::Acquire);
        loop {
            if n >= max {
                return false;
            }
            match self
                .0
                .compare_exchange_weak(n, n + 1, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(now) => n = now,
            }
        }
    }

    pub fn leave(&self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Permutation models (`cargo test -p beyond-execd --lib loom` under `--cfg beyond_loom`).
#[cfg(all(test, beyond_loom))]
mod loom_models {
    use super::InFlightCount;
    use loom::sync::Arc;
    use loom::thread;

    fn race(threads: usize, max: usize, pre: usize) -> (usize, usize) {
        let c = Arc::new(InFlightCount::new());
        for _ in 0..pre {
            c.enter();
        }
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                let c = c.clone();
                thread::spawn(move || {
                    let ok = c.enter_within(max);
                    // The bound holds at every point any admitted caller can see.
                    assert!(c.get() <= max.max(pre));
                    ok
                })
            })
            .collect();
        let admitted = hs
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&ok| ok)
            .count();
        (admitted, c.get())
    }

    /// Two callers racing for the last slot: exactly one gets it, however they interleave.
    #[test]
    fn two_callers_for_one_slot_admit_exactly_one() {
        loom::model(|| {
            let (admitted, count) = race(2, 1, 0);
            assert_eq!((admitted, count), (1, 1));
        });
    }

    /// Three callers for two slots: exactly two.
    #[test]
    fn three_callers_for_two_slots_admit_exactly_two() {
        loom::model(|| {
            let (admitted, count) = race(3, 2, 0);
            assert_eq!((admitted, count), (2, 2));
        });
    }

    /// At the limit already, every caller is refused and the count is untouched.
    #[test]
    fn at_the_limit_every_caller_is_refused() {
        loom::model(|| {
            let (admitted, count) = race(2, 1, 1);
            assert_eq!((admitted, count), (0, 1));
        });
    }

    /// A request leaving while another arrives at the limit: the arrival is admitted if and only if
    /// it saw the slot freed, and the count ends equal to what is in.
    #[test]
    fn a_leave_racing_an_arrival_never_loses_or_overshoots_a_slot() {
        loom::model(|| {
            let c = Arc::new(InFlightCount::new());
            c.enter();
            let l = {
                let c = c.clone();
                thread::spawn(move || c.leave())
            };
            let a = {
                let c = c.clone();
                thread::spawn(move || c.enter_within(1))
            };
            l.join().unwrap();
            let ok = a.join().unwrap();
            assert_eq!(c.get(), usize::from(ok));
            assert!(c.get() <= 1);
        });
    }
}
