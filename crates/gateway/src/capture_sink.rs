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
//! the exact entanglement the queue is here to prevent. On a graceful shutdown the drain in `main`
//! writes out what is queued before it exits ([`CaptureDrain::finish`]), bounded so a wedged log
//! pipeline cannot hold teardown up; anything still queued then is lost with a warn line. A line
//! the destination refuses is counted as dropped, like one the full queue drops (D208).
//!
//! The queue is bounded twice: by line count (`depth`) and by the bytes those lines hold
//! (`max_bytes`, D262). A line count alone is not a memory bound here — one payload line holds up to
//! two `capture_max_bytes` bodies, more after JSON escaping — so 1 024 queued lines could pin
//! gigabytes while the log pipeline stalls, uncounted by any other budget. A line that would take the
//! queued bytes past `max_bytes` is dropped and counted like a full-queue drop.
//!
//! The same sink carries the diagnostic log (every target but `ai.usage` and `ai.payload`) on a
//! second instance with its own thread and counter (`ai_log_dropped_total`, D263): a stalled stdout
//! must not park the Tokio workers that emit a `warn!`. `ai.usage` stays synchronous and lossless.

use prometheus::IntCounter;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TrySendError, channel, sync_channel,
};
use std::time::{Duration, Instant};
use tracing_subscriber::fmt::MakeWriter;

/// Handle to the payload queue. Cloned by `tracing` once per emitted event; a clone is three `Arc`
/// bumps, no allocation.
#[derive(Clone)]
pub struct CaptureSink {
    tx: SyncSender<Vec<u8>>,
    dropped: IntCounter,
    queued: Arc<QueuedBytes>,
}

/// The bytes held by lines that are queued or being written, against the sink's byte budget (D262).
/// Charged when a line is enqueued, released once the drain thread has written it (or failed to).
struct QueuedBytes {
    held: AtomicUsize,
    max: usize,
}

impl QueuedBytes {
    /// Charge `n` bytes if they fit under the budget. A charge that does not fit is undone, so two
    /// racing lines near the bound may both be refused; neither is ever admitted over it.
    fn charge(&self, n: usize) -> bool {
        let before = self.held.fetch_add(n, Ordering::Relaxed);
        if before.saturating_add(n) > self.max {
            self.held.fetch_sub(n, Ordering::Relaxed);
            return false;
        }
        true
    }

    fn release(&self, n: usize) {
        self.held.fetch_sub(n, Ordering::Relaxed);
    }
}

/// The shutdown handle for a [`CaptureSink`]'s drain thread (D208).
///
/// The `tracing` subscriber keeps a sender for the life of the process, so dropping senders can
/// never close the queue. [`Self::finish`] instead enqueues an empty line, which no event produces
/// ([`QueueWriter`] never sends one): the drain thread writes everything queued before it, flushes,
/// and stops, and a line enqueued after that is counted as dropped.
pub struct CaptureDrain {
    tx: SyncSender<Vec<u8>>,
    done: Receiver<()>,
}

impl CaptureDrain {
    /// Write out every line queued so far and stop the drain thread, waiting at most `timeout`
    /// (a wedged destination must not hold the process up past its shutdown budget). Returns
    /// whether the thread finished in time.
    pub fn finish(self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            match self.tx.try_send(Vec::new()) {
                Ok(()) => break,
                Err(TrySendError::Disconnected(_)) => return true,
                Err(TrySendError::Full(_)) if Instant::now() >= deadline => return false,
                Err(TrySendError::Full(_)) => std::thread::sleep(Duration::from_millis(5)),
            }
        }
        let left = deadline.saturating_duration_since(Instant::now());
        !matches!(self.done.recv_timeout(left), Err(RecvTimeoutError::Timeout))
    }
}

impl CaptureSink {
    /// Spawn the drain thread and return the handle to install as a `tracing` writer, and the
    /// handle that drains it at shutdown.
    ///
    /// `depth` is the queue bound in lines — how long a sink stall we absorb before dropping — and
    /// `max_bytes` the bound on the bytes those lines hold (`0`: no byte bound). `name` names the
    /// drain thread.
    pub fn spawn(
        name: &str,
        depth: usize,
        max_bytes: usize,
        dropped: IntCounter,
    ) -> io::Result<(Self, CaptureDrain)> {
        Self::spawn_to(name, depth, max_bytes, dropped, io::stdout())
    }

    /// [`Self::spawn`] with an explicit destination, so the drop-on-full behaviour is testable
    /// without capturing the process's real stdout.
    pub fn spawn_to<W: Write + Send + 'static>(
        name: &str,
        depth: usize,
        max_bytes: usize,
        dropped: IntCounter,
        mut out: W,
    ) -> io::Result<(Self, CaptureDrain)> {
        // `sync_channel(depth)` is the bound. `depth` of 0 would make every send rendezvous with the
        // drain thread — i.e. exactly the blocking behaviour this module exists to avoid — so floor
        // it at 1.
        let (tx, rx) = sync_channel::<Vec<u8>>(depth.max(1));
        let (done_tx, done) = channel::<()>();
        let lost = dropped.clone();
        let queued = Arc::new(QueuedBytes {
            held: AtomicUsize::new(0),
            max: if max_bytes == 0 {
                usize::MAX
            } else {
                max_bytes
            },
        });
        let written = Arc::clone(&queued);
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                // Ends at `CaptureDrain::finish`'s empty line, or when every sender is dropped.
                for line in rx.iter().take_while(|l| !l.is_empty()) {
                    // A line the destination refused is as lost as one the full queue dropped:
                    // counted the same way, so `ai_capture_dropped_total` means every lost line
                    // (D208). Nothing else to report it to: the failed write was the log.
                    if out.write_all(&line).is_err() {
                        lost.inc();
                    }
                    // Held until written, not until dequeued: a line in the thread's hands while
                    // the destination is wedged still pins its bytes.
                    written.release(line.len());
                }
                // Close the queue before reporting done: a line sent after `finish` returns must
                // find it disconnected and be counted, not sit in a buffer no one will drain.
                drop(rx);
                let _ = out.flush();
                let _ = done_tx.send(());
            })?;
        Ok((
            Self {
                tx: tx.clone(),
                dropped,
                queued,
            },
            CaptureDrain { tx, done },
        ))
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
        // Over the byte budget: dropped and counted, like a full queue (D262).
        let len = line.len();
        if !self.sink.queued.charge(len) {
            self.sink.dropped.inc();
            return;
        }
        // `try_send`, never `send`: this runs on a worker thread that has just finished serving a
        // request, and blocking it on a log queue is the failure mode this whole module prevents.
        match self.sink.tx.try_send(line) {
            Ok(()) => {}
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.sink.queued.release(len);
                self.sink.dropped.inc();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Opts;
    use std::sync::{Arc, Mutex};

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
        let (sink, _drain) =
            CaptureSink::spawn_to("t", 16, 0, counter(), dest.clone()).expect("spawn");
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
        let (sink, _drain) =
            CaptureSink::spawn_to("t", 2, 0, dropped.clone(), Wedged).expect("spawn");

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

    /// A destination whose every write fails (a closed stdout pipe).
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
    }

    /// A destination that takes its time with each line, so lines are still queued at shutdown.
    #[derive(Clone, Default)]
    struct Slow(Arc<Mutex<Vec<u8>>>);
    impl Write for Slow {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            std::thread::sleep(Duration::from_millis(20));
            self.0.lock().expect("lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Every capture line that does not land is counted: one the destination refused is as lost
    /// as one the full queue dropped, and lines still queued at shutdown are written before the
    /// process exits (`finish`), not lost with it. After `finish`, a late line is counted.
    /// claim: O2
    /// defect: D208
    #[test]
    fn every_lost_capture_line_is_counted_and_shutdown_drains_the_queue() {
        let dropped = counter();
        let (sink, drain) =
            CaptureSink::spawn_to("t", 16, 0, dropped.clone(), Broken).expect("spawn");
        for _ in 0..5 {
            write_line(&sink, b"{}\n");
        }
        assert!(
            drain.finish(Duration::from_secs(5)),
            "the drain thread stopped"
        );
        assert_eq!(dropped.get(), 5, "each refused write is a dropped line");

        let dropped = counter();
        let dest = Slow::default();
        let (sink, drain) =
            CaptureSink::spawn_to("t", 16, 0, dropped.clone(), dest.clone()).expect("spawn");
        for i in 0..8 {
            write_line(&sink, format!("{i}\n").as_bytes());
        }
        assert!(
            drain.finish(Duration::from_secs(5)),
            "the drain thread stopped"
        );
        assert_eq!(
            dest.0.lock().expect("lock").as_slice(),
            b"0\n1\n2\n3\n4\n5\n6\n7\n",
            "every queued line was written before finish returned"
        );
        assert_eq!(dropped.get(), 0);
        write_line(&sink, b"late\n");
        assert_eq!(
            dropped.get(),
            1,
            "a line after shutdown is counted, not lost silently"
        );
    }

    #[test]
    fn an_empty_event_enqueues_nothing() {
        let dropped = counter();
        let (sink, _drain) =
            CaptureSink::spawn_to("t", 1, 0, dropped.clone(), Wedged).expect("spawn");
        // A writer that never wrote must not consume a queue slot, or a stream of them would evict
        // real payloads.
        for _ in 0..32 {
            drop(sink.make_writer());
        }
        assert_eq!(dropped.get(), 0);
    }

    /// The queue is bounded in bytes as well as lines: with the destination wedged, a line that
    /// would take the held bytes past the budget is dropped and counted, while one that still fits
    /// is admitted. The line in the drain thread's hands keeps its bytes charged until written.
    /// defect: D262
    #[test]
    fn a_line_over_the_byte_budget_is_dropped_and_counted() {
        let dropped = counter();
        let (sink, _drain) =
            CaptureSink::spawn_to("t", 1024, 100, dropped.clone(), Wedged).expect("spawn");
        write_line(&sink, &[b'a'; 60]);
        write_line(&sink, &[b'b'; 60]);
        assert_eq!(dropped.get(), 1, "60 + 60 bytes is over a 100-byte budget");
        write_line(&sink, &[b'c'; 40]);
        assert_eq!(
            dropped.get(),
            1,
            "60 + 40 bytes fits a 100-byte budget exactly"
        );
        write_line(&sink, b"d");
        assert_eq!(dropped.get(), 2, "the budget is full");
        assert_eq!(sink.queued.held.load(Ordering::Relaxed), 100);
    }

    /// A written line gives its bytes back: a stream of lines, each alone within the budget but
    /// together far over it, all land when the destination keeps up.
    /// defect: D262
    #[test]
    fn a_written_line_releases_its_bytes() {
        let dropped = counter();
        let dest = Shared::default();
        let (sink, _drain) =
            CaptureSink::spawn_to("t", 1024, 100, dropped.clone(), dest.clone()).expect("spawn");
        for i in 0..10u8 {
            write_line(&sink, &[b'0' + i; 60]);
            let deadline = Instant::now() + Duration::from_secs(5);
            while sink.queued.held.load(Ordering::Relaxed) != 0 {
                assert!(
                    Instant::now() < deadline,
                    "line {i} never released its bytes"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        assert_eq!(dropped.get(), 0);
        assert_eq!(dest.0.lock().expect("lock").len(), 600);
    }

    /// A line the full queue refuses gives its charge back, or full-queue drops would leak budget
    /// until every later line was refused.
    /// defect: D262
    #[test]
    fn a_full_queue_drop_releases_its_bytes() {
        let dropped = counter();
        let (sink, _drain) =
            CaptureSink::spawn_to("t", 1, 1000, dropped.clone(), Wedged).expect("spawn");
        for _ in 0..10 {
            write_line(&sink, &[b'x'; 10]);
        }
        // One line in the wedged thread's hands, at most one queued: the rest were refused.
        let held = sink.queued.held.load(Ordering::Relaxed);
        assert!(
            held <= 20,
            "refused lines kept their charge: {held} bytes held"
        );
        assert!(dropped.get() >= 8, "dropped {}", dropped.get());
    }
}
