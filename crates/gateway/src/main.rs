//! Beyond AI gateway binary: clap `Run`/`Doctor`, Pingora server bootstrap, services.

// See `lib.rs`: deny the panic surface in production, allow it in `#[cfg(test)]` assertions.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

// jemalloc (not mimalloc) for the gateway: under a memory cgroup it returns reclaimed pages via
// MADV_DONTNEED so the cgroup uncharges them immediately, whereas mimalloc's MADV_FREE leaves freed
// pages charged — the same reason `compute/instd` runs jemalloc. The rest of the fleet uses mimalloc.
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use beyond_ai::admin::AdminApp;
use beyond_ai::capture_sink::CaptureSink;
use beyond_ai::config::AiConfig;
use beyond_ai::doctor;
use beyond_ai::metrics::Metrics;
use beyond_ai::proxy::AiProxy;
use beyond_ai::state::GatewayState;
use beyond_ai::store_watch::{Capture, Deny, WatcherService};
use clap::{Parser, Subcommand};
use pingora_core::apps::HttpServerOptions;
use pingora_core::apps::http_app::HttpServer;
use pingora_core::server::Server;
use pingora_core::server::configuration::ServerConf;
use pingora_core::services::background::background_service;
use pingora_core::services::listening::Service as ListeningService;
use pingora_proxy::ProxyServiceBuilder;
use std::path::Path;
use std::process::exit;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::filter_fn;
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

/// The target carrying captured request/response payloads. Split onto its own writer — see below.
const PAYLOAD_TARGET: &str = "ai.payload";

fn init_tracing(metrics: &Metrics, queue_depth: usize) {
    // JSON to stdout; the `ai.usage` target carries billing facts that logfwd/OTLP ships to
    // ClickHouse. `AI_LOG` overrides the level filter.
    let filter = EnvFilter::try_from_env("AI_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    // Two layers, split by target, because the two kinds of line want opposite guarantees.
    //
    //  * Everything else — including `ai.usage` — keeps the **blocking** stdout writer. Billing rows
    //    must never be dropped, and at a few hundred bytes each the synchronous write is free.
    //  * `ai.payload` gets a **bounded, lossy** queue drained by its own thread. A captured payload
    //    is orders of magnitude larger and exists only to explain incidents, so if the log pipeline
    //    stalls we drop payloads (counted on `ai_capture_dropped_total`) rather than let a stalled
    //    stdout pipe backpressure the proxy. See `capture_sink`.
    //
    // The two filters are exact complements: every event lands in exactly one layer, so nothing is
    // duplicated and nothing is silently swallowed.
    // A gateway that can't spawn a thread at boot won't serve traffic either — fail visibly rather
    // than run with payload capture silently disabled. Same eprintln+exit shape as the config and
    // metrics failures above, which is why this isn't a `tracing` error: nothing is initialized yet.
    let payload_sink = match CaptureSink::spawn(queue_depth, metrics.capture_dropped_total.clone())
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to start the capture-payload log sink: {e}");
            exit(1);
        }
    };
    let payload_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(payload_sink)
        .with_filter(filter_fn(|meta| meta.target() == PAYLOAD_TARGET));
    let main_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_filter(filter_fn(|meta| meta.target() != PAYLOAD_TARGET));

    tracing_subscriber::registry()
        .with(main_layer)
        .with(payload_layer)
        .with(filter)
        .init();
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
    init_tracing(&metrics, capture_queue_depth);
    let state = match GatewayState::new(config, metrics) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to build gateway state: {e}");
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
    let mut proxy_builder = ProxyServiceBuilder::new(
        &server.configuration,
        AiProxy {
            state: state.clone(),
        },
    );
    if downstream_h2c {
        // `HttpServerOptions` is `#[non_exhaustive]`, so build via `Default` and set the field.
        let mut opts = HttpServerOptions::default();
        opts.h2c = true;
        proxy_builder = proxy_builder.server_options(opts);
    }
    let mut proxy_svc = proxy_builder.build();
    proxy_svc.add_tcp(&listen);
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
    // capture-set outage backs off on its own schedule and can't disturb deny enforcement.
    server.add_service(background_service(
        "ai-watch-deny",
        WatcherService::<Deny>::new(state.clone()),
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
    server.run_forever();
}
