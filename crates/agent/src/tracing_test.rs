//! A minimal `tracing::Subscriber` for asserting a module actually logged something, without pulling
//! in a dedicated test-logging crate for what's otherwise a one-off need. Every method beyond `event`
//! is an inert stub — nothing here uses spans, only bare `tracing::warn!`/`info!`-style events. Only
//! ever compiled in under `#[cfg(test)]` — see the `mod tracing_test` declaration in `lib.rs`.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, Once};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

/// Captures every event's `message` field, as recorded during a [`capture`] scope.
#[derive(Clone, Default)]
pub struct CaptureSubscriber(Arc<Mutex<Vec<String>>>);

impl CaptureSubscriber {
    pub fn messages(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

thread_local! {
    static ACTIVE_BUFFER: RefCell<Option<Arc<Mutex<Vec<String>>>>> = const { RefCell::new(None) };
}

static INSTALL: Once = Once::new();

/// Run `f`, returning a [`CaptureSubscriber`] the caller can inspect via
/// [`CaptureSubscriber::messages`] to see what `f` logged on the current thread.
///
/// Deliberately does *not* use `tracing::subscriber::with_default` per-call. `tracing`'s callsite
/// interest cache is global and shared across the whole test binary's threads, not scoped to
/// whichever subscriber is currently installed on one thread: the *first* time a given
/// `tracing::warn!`/`info!` call site is reached from any thread, its "is anyone interested" result
/// is cached process-wide. Under `cargo test`'s parallel execution, one thread's `with_default` scope
/// can end (and rebuild/invalidate the cache) concurrently with another thread's scope starting,
/// racing the same global cache — a small but real flakiness source, not hypothetical (a
/// warn-log-capturing test failing depending on unrelated tests' scheduling).
///
/// Instead we install exactly one real `Subscriber` globally, once, for the process's whole test
/// binary lifetime (interest is then decided permanently — this subscriber is always "interested",
/// so there's nothing left to race), and route each thread's captured messages through a
/// thread-local buffer so concurrent `capture()` calls on different threads never see each other's
/// events.
pub fn capture<F: FnOnce()>(f: F) -> CaptureSubscriber {
    INSTALL.call_once(|| {
        // Ignore the error: if some other global subscriber already won (e.g. a test binary that
        // also calls `tracing_subscriber::fmt().init()`), our thread-local routing simply won't
        // receive events, and callers will see an empty capture — there's nothing else to do about a
        // process that already claimed the one global default.
        let _ = tracing::subscriber::set_global_default(RoutingSubscriber);
    });

    let buffer = Arc::new(Mutex::new(Vec::new()));
    ACTIVE_BUFFER.with(|cell| *cell.borrow_mut() = Some(buffer.clone()));
    f();
    ACTIVE_BUFFER.with(|cell| *cell.borrow_mut() = None);
    CaptureSubscriber(buffer)
}

#[derive(Default)]
struct MessageVisitor(String);

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

/// The single process-wide `Subscriber`, installed once by [`capture`]. Forwards each event to
/// whichever thread-local buffer (if any) is active on the recording thread.
struct RoutingSubscriber;

impl Subscriber for RoutingSubscriber {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _span: &Id, _values: &Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn event(&self, event: &Event<'_>) {
        ACTIVE_BUFFER.with(|cell| {
            if let Some(buffer) = cell.borrow().as_ref() {
                let mut visitor = MessageVisitor::default();
                event.record(&mut visitor);
                buffer.lock().unwrap().push(visitor.0);
            }
        });
    }
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
}
