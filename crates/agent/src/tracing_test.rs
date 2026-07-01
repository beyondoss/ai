//! A minimal `tracing::Subscriber` for asserting a module actually logged something, without pulling
//! in a dedicated test-logging crate for what's otherwise a one-off need. Every method beyond `event`
//! is an inert stub — nothing here uses spans, only bare `tracing::warn!`/`info!`-style events. Only
//! ever compiled in under `#[cfg(test)]` — see the `mod tracing_test` declaration in `lib.rs`.

use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

/// Captures every event's `message` field. Cheaply `Clone` (an `Arc<Mutex<..>>` underneath), so one
/// handle is installed via [`tracing::subscriber::with_default`] and another kept by the test to read
/// back what was recorded once the scope ends.
#[derive(Clone, Default)]
pub struct CaptureSubscriber(Arc<Mutex<Vec<String>>>);

impl CaptureSubscriber {
    pub fn messages(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
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

impl Subscriber for CaptureSubscriber {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _span: &Id, _values: &Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        self.0.lock().unwrap().push(visitor.0);
    }
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
}
