//! The `ai.payload` writer: bounded, lossy, and off the serving threads.
//!
//! `ai.usage` is written straight to stdout on the worker thread that finished the request. That is
//! fine for a ~400-byte billing line, and it is deliberately **lossless** — a dropped billing row is
//! money we can't account for.
//!
//! A captured payload is a different animal: up to `capture_max_bytes` per direction, on a target
//! that only exists to explain incidents. Writing that synchronously would put the log pipeline on
//! the critical path — if logfwd stops draining the stdout pipe, the pipe buffer fills, `write(2)`
//! blocks, and a *log sink* is now applying backpressure to the proxy. Observability that can stall
//! the data plane is a worse bug than the missing observability it was added to fix.
//!
//! So payloads go through a bounded queue drained by one dedicated OS thread, and **overflow drops
//! the line** rather than waiting. Every drop is counted: `ai_capture_dropped_total` is what makes a
//! missing payload diagnosable ("capture was on and we lost it") instead of ambiguous ("was capture
//! even on?"), which is the question that gets asked during the incident this feature exists for.
//!
//! Deliberately `std::sync::mpsc` and a plain thread rather than tokio: this is constructed in
//! `main` before any runtime exists, and coupling log egress to the runtime that serves traffic is
//! the exact entanglement the queue is here to prevent. On shutdown the process exits without
//! draining — a handful of lost debug payloads is the correct thing to lose, and blocking teardown
//! on a log flush is not.

use prometheus::IntCounter;
use std::io::{self, Write};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use tracing_subscriber::fmt::MakeWriter;

/// Handle to the payload queue. Cloned by `tracing` once per emitted event; a clone is two `Arc`
/// bumps, no allocation.
#[derive(Clone)]
pub struct CaptureSink {
    tx: SyncSender<Vec<u8>>,
    dropped: IntCounter,
}

impl CaptureSink {
    /// Spawn the drain thread and return the handle to install as a `tracing` writer.
    ///
    /// `depth` is the queue bound in lines — how long a sink stall we absorb before dropping.
    pub fn spawn(depth: usize, dropped: IntCounter) -> io::Result<Self> {
        Self::spawn_to(depth, dropped, io::stdout())
    }

    /// [`Self::spawn`] with an explicit destination, so the drop-on-full behaviour is testable
    /// without capturing the process's real stdout.
    pub fn spawn_to<W: Write + Send + 'static>(
        depth: usize,
        dropped: IntCounter,
        mut out: W,
    ) -> io::Result<Self> {
        // `sync_channel(depth)` is the bound. `depth` of 0 would make every send rendezvous with the
        // drain thread — i.e. exactly the blocking behaviour this module exists to avoid — so floor
        // it at 1.
        let (tx, rx) = sync_channel::<Vec<u8>>(depth.max(1));
        std::thread::Builder::new()
            .name("ai-capture-sink".to_string())
            .spawn(move || {
                // Ends when every sender is dropped, which in practice means process shutdown.
                for line in rx {
                    // A failed write is not worth escalating: the payload channel is best-effort by
                    // construction, and there is nowhere useful to report a logging failure *to*
                    // except the log we just failed to write.
                    let _ = out.write_all(&line);
                }
                let _ = out.flush();
            })?;
        Ok(Self { tx, dropped })
    }
}

impl<'a> MakeWriter<'a> for CaptureSink {
    type Writer = QueueWriter;

    fn make_writer(&'a self) -> Self::Writer {
        QueueWriter {
            buf: Vec::new(),
            sink: self.clone(),
        }
    }
}

/// Accumulates one formatted event, then enqueues it whole on drop.
///
/// Whole-line rather than per-`write` enqueueing because `tracing`'s JSON formatter emits an event
/// across several `write` calls; forwarding each one separately would let two concurrent events
/// interleave into a corrupt line under a queue that is drained by a single thread.
pub struct QueueWriter {
    buf: Vec<u8>,
    sink: CaptureSink,
}

impl Write for QueueWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for QueueWriter {
    fn drop(&mut self) {
        let line = std::mem::take(&mut self.buf);
        if line.is_empty() {
            return;
        }
        // `try_send`, never `send`: this runs on a worker thread that has just finished serving a
        // request, and blocking it on a log queue is the failure mode this whole module prevents.
        match self.sink.tx.try_send(line) {
            Ok(()) => {}
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => self.sink.dropped.inc(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Opts;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// A destination that blocks forever on first write, simulating a wedged log pipeline.
    struct Wedged;
    impl Write for Wedged {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            std::thread::sleep(Duration::from_secs(3600));
            Ok(0)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);
    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn counter() -> IntCounter {
        IntCounter::with_opts(Opts::new("t", "t")).expect("counter")
    }

    fn write_line(sink: &CaptureSink, line: &[u8]) {
        let mut w = sink.make_writer();
        w.write_all(line).expect("buffered write never fails");
        // Enqueue happens on drop — that's the seam being exercised.
        drop(w);
    }

    #[test]
    fn lines_reach_the_destination_whole() {
        let dest = Shared::default();
        let sink = CaptureSink::spawn_to(16, counter(), dest.clone()).expect("spawn");
        // Split across two `write` calls, as the JSON formatter does; must arrive as one line.
        let mut w = sink.make_writer();
        w.write_all(br#"{"a":1"#).expect("write");
        w.write_all(b"}\n").expect("write");
        drop(w);

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if dest.0.lock().expect("lock").as_slice() == b"{\"a\":1}\n" {
                return;
            }
            assert!(Instant::now() < deadline, "line never arrived");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn a_wedged_sink_drops_instead_of_blocking() {
        // The property this module exists for: with the destination stuck, enqueueing must stay
        // fast and start counting drops rather than parking the calling thread.
        let dropped = counter();
        let sink = CaptureSink::spawn_to(2, dropped.clone(), Wedged).expect("spawn");

        let started = Instant::now();
        for _ in 0..64 {
            write_line(&sink, b"{}\n");
        }
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "enqueue blocked on a wedged sink ({elapsed:?}) — this is the backpressure path into \
             the data plane that the bounded queue exists to cut"
        );
        // Queue depth 2 (+1 in the drain thread's hands); the rest must be counted as dropped.
        assert!(
            dropped.get() >= 60,
            "expected most of 64 lines dropped, got {}",
            dropped.get()
        );
    }

    #[test]
    fn an_empty_event_enqueues_nothing() {
        let dropped = counter();
        let sink = CaptureSink::spawn_to(1, dropped.clone(), Wedged).expect("spawn");
        // A writer that never wrote must not consume a queue slot, or a stream of them would evict
        // real payloads.
        for _ in 0..32 {
            drop(sink.make_writer());
        }
        assert_eq!(dropped.get(), 0);
    }
}
