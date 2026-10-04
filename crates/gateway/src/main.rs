//! Beyond AI gateway binary: clap `Run`/`Doctor`, Pingora server bootstrap, services.

// See `lib.rs`: deny the panic surface in production, allow it in `#[cfg(test)]` assertions.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

// jemalloc (not mimalloc) for the gateway: under a memory cgroup it returns reclaimed pages via
// MADV_DONTNEED so the cgroup uncharges them immediately, whereas mimalloc's MADV_FREE leaves freed
// pages charged — the same reason `compute/instd` runs jemalloc. The rest of the fleet uses mimalloc.
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use beyond_ai::admin::AdminApp;
use beyond_ai::capture_sink::{CaptureDrain, CaptureSink};
use beyond_ai::config::AiConfig;
use beyond_ai::doctor;
use beyond_ai::metrics::Metrics;
use beyond_ai::proxy::{AiProxy, LIVE_REQUESTS};
use beyond_ai::state::GatewayState;
use beyond_ai::store_watch::{Allowance, Capture, Deny, WatcherService};
use beyond_ai::usage::{USAGE_TARGET, usage_log_filter};
use clap::{Parser, Subcommand};
use pingora_core::apps::HttpServerOptions;
use pingora_core::apps::http_app::HttpServer;
use pingora_core::listeners::TcpSocketOptions;
use pingora_core::server::configuration::ServerConf;
use pingora_core::server::{
    RunArgs, Server, ShutdownSignal, ShutdownSignalWatch, UnixShutdownSignalWatch,
};
use pingora_core::services::background::background_service;
use pingora_core::services::listening::Service as ListeningService;
use pingora_proxy::ProxyServiceBuilder;
use std::io::Write as _;
use std::path::Path;
use std::process::exit;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::{FilterExt, filter_fn};
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser)]
#[command(
    name = "beyond-ai",
    about = "Beyond AI gateway — egress proxy to LLM providers"
)]
struct Cli {
    /// Path to config file (defaults to ./config.toml).
    #[arg(short, long, env = "AI_CONFIG_PATH", global = true)]
    config: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run prerequisite diagnostics and exit.
    Doctor,
    /// Start the gateway (default).
    Run,
}

fn load_config(path: Option<&Path>) -> AiConfig {
    match AiConfig::load_with_path(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load config: {e}");
            exit(1);
        }
    }
}

/// Pingora's own signal watch, plus a drain that ends the process as soon as nothing is in flight.
///
/// On SIGTERM pingora stops accepting and then sleeps the whole `grace_period_seconds` before it
/// tears the runtimes down, whether or not anything is still running: an idle gateway took the full
/// 600 s to stop, so every deploy on a platform with a shorter stop timeout (ECS Fargate: 120 s)
/// ended in SIGKILL. The drain watches [`LIVE_REQUESTS`] (every request context, dropped only after
/// its `logging`, billing row included) and exits the moment it reaches zero. Pingora's sleep stays
/// the upper bound: a request still running when the grace ends is cut as before.
///
/// Before it exits, the drain writes out the `ai.payload` and diagnostic lines still queued
/// ([`LogDrains`], D208, D263): `exit` does not wait for the sinks' threads, so a capture of the last
/// requests was lost silently. `ai.usage` rows need no such step: they are written synchronously.
struct DrainOnTerm {
    grace: Duration,
    logs: Arc<Mutex<Option<LogDrains>>>,
}

/// The shutdown handles of the two lossy log sinks: `ai.payload` and diagnostics.
struct LogDrains {
    capture: CaptureDrain,
    diagnostics: CaptureDrain,
}

impl LogDrains {
    /// Write out what both sinks hold, within `budget` in all. Payloads first, so the warn line
    /// saying they did not drain still has a diagnostic sink to land in.
    fn finish(self, budget: Duration) {
        let deadline = Instant::now() + budget;
        if !self.capture.finish(budget) {
            tracing::warn!(
                "capture sink did not drain before exit; queued ai.payload lines are lost"
            );
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if !self.diagnostics.finish(left) {
            let _ = writeln!(
                std::io::stderr(),
                "diagnostic log sink did not drain before exit; queued log lines are lost"
            );
        }
    }
}

/// Take the drains (once: whoever takes them first finishes them) and finish them.
fn finish_logs(logs: &Mutex<Option<LogDrains>>, budget: Duration) {
    if let Some(d) = logs.lock().ok().and_then(|mut d| d.take()) {
        d.finish(budget);
    }
}

#[async_trait::async_trait]
impl ShutdownSignalWatch for DrainOnTerm {
    async fn recv(&self) -> ShutdownSignal {
        let signal = UnixShutdownSignalWatch.recv().await;
        if matches!(signal, ShutdownSignal::GracefulTerminate) {
            let grace = self.grace;
            let logs = Arc::clone(&self.logs);
            // A plain thread: pingora's main thread is about to block in its grace sleep, and the
            // service runtimes are what is being drained.
            let spawned = std::thread::Builder::new()
                .name("ai-drain".into())
                .spawn(move || drain_then_exit(grace, &logs));
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "could not start the shutdown drain; waiting out the grace period");
            }
        }
        signal
    }
}

/// How long the drain lets pingora's shutdown broadcast land (listeners stop accepting, idle
/// keep-alive connections drop) before it trusts a zero [`LIVE_REQUESTS`].
const DRAIN_SETTLE: Duration = Duration::from_millis(200);

/// The longest the drain waits for queued `ai.payload` and diagnostic lines to be written before it
/// exits. A wedged log pipeline must not hold the process past its stop timeout; what is still
/// queued then is lost, and a warn line says so.
const LOG_FLUSH: Duration = Duration::from_secs(5);

/// The diagnostic sink's queue, in lines and in bytes (D263). A diagnostic line is a few hundred
/// bytes, so the line bound binds first in practice: ~8 k lines absorb a multi-second stdout stall
/// at a heavy log rate, and the byte bound caps the odd outsized error string.
const DIAG_QUEUE_DEPTH: usize = 8192;
const DIAG_QUEUE_BYTES: usize = 16 * 1024 * 1024;

fn drain_then_exit(grace: Duration, logs: &Mutex<Option<LogDrains>>) {
    let start = Instant::now();
    std::thread::sleep(DRAIN_SETTLE);
    while start.elapsed() < grace {
        let live = LIVE_REQUESTS.load(Ordering::Acquire);
        if live == 0 {
            tracing::info!(
                after_ms = start.elapsed().as_millis() as u64,
                "drained: no request in flight; exiting"
            );
            finish_logs(logs, grace.saturating_sub(start.elapsed()).min(LOG_FLUSH));
            exit(0);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The target carrying captured request/response payloads. Split onto its own writer — see below.
const PAYLOAD_TARGET: &str = "ai.payload";

/// Stdout for `ai.usage` rows that notices when a row fails to land.
///
/// The fmt layer discards a writer's error, so a closed or broken stdout pipe silently lost every
/// billing row. This counts each failed row on `ai_usage_write_errors_total` and says so on
/// stderr — the only record left of it. One row is one `write_all`, which stops at the first error,
/// so a failed row counts once.
struct UsageStdout(prometheus::IntCounter);

struct UsageLine<'a>(&'a prometheus::IntCounter, std::io::Stdout);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for UsageStdout {
    type Writer = UsageLine<'a>;
    fn make_writer(&'a self) -> Self::Writer {
        UsageLine(&self.0, std::io::stdout())
    }
}

impl std::io::Write for UsageLine<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.1.write(buf).inspect_err(|e| {
            // `write_all` retries an interrupted write itself; that is not a lost row.
            if e.kind() != std::io::ErrorKind::Interrupted {
                self.0.inc();
                let _ = writeln!(
                    std::io::stderr(),
                    "ai.usage billing row lost: stdout write failed: {e}"
                );
            }
        })
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.1.flush()
    }
}

fn init_tracing(metrics: &Metrics, queue_depth: usize, queue_bytes: usize) -> LogDrains {
    // JSON to stdout; the `ai.usage` target carries billing facts that logfwd/OTLP ships to
    // ClickHouse. `AI_LOG` sets the level filter for everything **except** those rows.
    let env_filter =
        || EnvFilter::try_from_env("AI_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    // Three layers, split by target, because the three kinds of line want different guarantees.
    //
    //  * `ai.usage` — billing rows, not diagnostics. Its own layer with **no** `AI_LOG` filter (only a
    //    max level hint of INFO, so pingora debug/trace records are never dispatched): an operator
    //    turning `AI_LOG` down to `warn` must not silently stop billing, which is what one
    //    global filter over every layer did. The **blocking** stdout writer: at a few hundred bytes
    //    each the synchronous write is free, and a row must never be dropped — a failed write is
    //    counted (`UsageStdout`).
    //  * Everything else — diagnostics, filtered by `AI_LOG`, on a **bounded, lossy** queue drained
    //    by its own thread (the `capture_sink` writer, second instance): a stalled stdout pipe
    //    drops diagnostics (counted on `ai_log_dropped_total`) instead of parking every Tokio
    //    worker that emits a `warn!` with its request — and its tenant slot — in hand (D263).
    //  * `ai.payload` gets a **bounded, lossy** queue drained by its own thread, under `AI_LOG` as
    //    before. A captured payload is orders of magnitude larger and exists only to explain
    //    incidents, so if the log pipeline stalls we drop payloads (counted on
    //    `ai_capture_dropped_total`) rather than let a stalled stdout pipe backpressure the proxy.
    //    See `capture_sink`.
    //
    // The target filters are exact complements: every event lands in at most one layer, so nothing
    // is duplicated.
    // A gateway that can't spawn a thread at boot won't serve traffic either — fail visibly rather
    // than run with payload capture silently disabled. Same eprintln+exit shape as the config and
    // metrics failures above, which is why this isn't a `tracing` error: nothing is initialized yet.
    let spawn = |name: &str, depth: usize, bytes: usize, dropped: &prometheus::IntCounter| {
        match CaptureSink::spawn(name, depth, bytes, dropped.clone()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("failed to start the {name} log sink: {e}");
                exit(1);
            }
        }
    };
    let (payload_sink, payload_drain) = spawn(
        "ai-capture-sink",
        queue_depth,
        queue_bytes,
        &metrics.capture_dropped_total,
    );
    let (diag_sink, diag_drain) = spawn(
        "ai-log-sink",
        DIAG_QUEUE_DEPTH,
        DIAG_QUEUE_BYTES,
        &metrics.log_dropped_total,
    );
    let usage_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(UsageStdout(metrics.usage_write_errors_total.clone()))
        .with_filter(usage_log_filter());
    let payload_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(payload_sink)
        .with_filter(filter_fn(|meta| meta.target() == PAYLOAD_TARGET).and(env_filter()));
    let main_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(diag_sink)
        .with_filter(
            filter_fn(|meta| meta.target() != PAYLOAD_TARGET && meta.target() != USAGE_TARGET)
                .and(env_filter()),
        );

    tracing_subscriber::registry()
        .with(usage_layer)
        .with(main_layer)
        .with(payload_layer)
        .init();
    LogDrains {
        capture: payload_drain,
        diagnostics: diag_drain,
    }
}

// Boot path: every `.expect()` here is a fatal start-up invariant (no runtime to build, no Pingora
// server) — a panic before we serve a single request is the correct, visible failure.
#[allow(clippy::expect_used)]
fn main() {
    // rustls 0.23 requires a process-wide crypto provider for the TLS connections to providers.
    // Idempotent: an `Err` means a provider is already installed (e.g. a second init in tests),
    // which is fine to ignore — the provider we want is in place either way.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cli = Cli::parse();

    // Doctor runs before any server setup (minimal current-thread runtime), exits 0/1.
    if matches!(cli.command, Some(Commands::Doctor)) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let config = load_config(cli.config.as_deref());
        let results = rt.block_on(doctor::run_checks(&config));
        doctor::print_results("Beyond AI Gateway Doctor", &results);
        exit(if results.iter().all(|r| r.passed) {
            0
        } else {
            1
        });
    }

    let config = load_config(cli.config.as_deref());
    let listen = config.listen.clone();
    let metrics_listen = config.metrics_listen.clone();
    let downstream_h2c = config.downstream_h2c;
    // `0` ⇒ one worker per core. Resolved here, before `config` moves into the gateway state.
    let worker_threads = match config.worker_threads {
        0 => std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
        n => n,
    };
    // Capture the shutdown knobs before `config` is moved into the gateway state below.
    let grace_period_secs = config.shutdown_grace_period_secs;
    let runtime_timeout_secs = config.shutdown_runtime_timeout_secs;
    let capture_queue_depth = config.capture_queue_depth;
    let capture_queue_bytes = config.capture_queue_bytes;
    let metrics = match Metrics::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to register metrics: {e}");
            exit(1);
        }
    };

    // After metrics (the payload layer owns a drop counter) and before `GatewayState::new`, which is
    // the first thing on this path that logs through `tracing` rather than `eprintln!`. Everything
    // above reports its own failures directly to stderr and exits, so nothing is lost by initializing
    // here rather than at the top of `main`.
    let logs = Arc::new(Mutex::new(Some(init_tracing(
        &metrics,
        capture_queue_depth,
        capture_queue_bytes,
    ))));
    let state = match GatewayState::new(config, metrics) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to build gateway state: {e}");
            // The warn lines that explain the failure are queued, not written: flush them first.
            finish_logs(&logs, LOG_FLUSH);
            exit(1);
        }
    };

    // Make the graceful-shutdown drain window explicit instead of inheriting Pingora's silent
    // defaults (300s grace / 5s runtime teardown). `grace_period_seconds` is how long in-flight
    // requests get to finish after SIGTERM before teardown; `graceful_shutdown_timeout_seconds` is
    // the final runtime-exit backstop. See the `AiConfig` field docs for the read_timeout /
    // orchestrator-stopTimeout tradeoffs.
    let conf = ServerConf {
        grace_period_seconds: Some(grace_period_secs),
        graceful_shutdown_timeout_seconds: Some(runtime_timeout_secs),
        ..ServerConf::default()
    };
    let mut server = Server::new_with_opt_and_conf(None, conf);
    server.bootstrap();

    // Client (app) traffic. Enable downstream HTTP/2 cleartext (h2c) when configured: Pingora peeks
    // the H2 connection preface and serves h2c, transparently falling back to HTTP/1.1 for h1
    // clients — so this is backward-compatible. Stays plaintext (no TLS); `add_tcp` is unchanged.
    let mut proxy_builder =
        ProxyServiceBuilder::new(&server.configuration, AiProxy::new(state.clone()));
    if downstream_h2c {
        // `HttpServerOptions` is `#[non_exhaustive]`, so build via `Default` and set the field.
        let mut opts = HttpServerOptions::default();
        opts.h2c = true;
        proxy_builder = proxy_builder.server_options(opts);
    }
    let mut proxy_svc = proxy_builder.build();
    // TCP keepalive on accepted client sockets (see `tcp_keepalive_idle_secs`): a client that
    // vanished during a silent model turn is dropped, and its slots freed, within the keepalive
    // bound rather than when the model next speaks. `TcpSocketOptions` is `#[non_exhaustive]`.
    let mut sock = TcpSocketOptions::default();
    sock.tcp_keepalive = state.config.downstream_tcp_keepalive();
    proxy_svc.add_tcp_with_settings(&listen, sock);
    // Size the proxy's worker pool. Pingora resolves a service's thread count as
    // `service.threads().unwrap_or(conf.threads)` (`server/mod.rs`), and `ServerConf::default()` is
    // `threads: 1` — so leaving this `None` runs every request filter, the Ed25519 verify, the body
    // scanners and the usage tap for the whole gateway on a **single** core regardless of box size.
    // Set on the service rather than on `conf`: `conf.threads` applies to every service, which would
    // also give the admin listener (one scrape every 15s) a full pool of its own.
    proxy_svc.threads = Some(worker_threads);
    server.add_service(proxy_svc);

    // slipstream watchers + NATS connectivity (connects on Pingora's runtime; see WatcherService).
    // One service per watched set, each with its own connection, cursor, and reconnect loop — so a
    // capture-set outage backs off on its own schedule and can't disturb deny or allowance.
    server.add_service(background_service(
        "ai-watch-deny",
        WatcherService::<Deny>::new(state.clone()),
    ));
    server.add_service(background_service(
        "ai-watch-allowance",
        WatcherService::<Allowance>::new(state.clone()),
    ));
    server.add_service(background_service(
        "ai-watch-capture",
        WatcherService::<Capture>::new(state.clone()),
    ));

    // Metrics listener now also serves /livez + /readyz for the ECS/k8s probes. Pingora's built-in
    // prometheus service only does /metrics, so we hand-route all three in one small ServeHttp.
    let mut admin = ListeningService::new(
        "ai-admin".to_string(),
        HttpServer::new_app(AdminApp {
            metrics: state.metrics.clone(),
            managed: !state.config.signing_keys.is_empty(),
        }),
    );
    admin.add_tcp(&metrics_listen);
    server.add_service(admin);

    tracing::info!(
        %listen,
        %metrics_listen,
        worker_threads,
        grace_period_secs,
        runtime_timeout_secs,
        downstream_h2c,
        "starting beyond-ai"
    );
    server.run(RunArgs {
        shutdown_signal: Box::new(DrainOnTerm {
            grace: Duration::from_secs(grace_period_secs),
            logs: Arc::clone(&logs),
        }),
    });
    // Pingora's own exit path (the grace period ran out, or a fast shutdown): the queued lines still
    // deserve their bounded flush.
    finish_logs(&logs, LOG_FLUSH);
    exit(0);
}
