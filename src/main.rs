//! Beyond AI gateway binary: clap `Run`/`Doctor`, Pingora server bootstrap, services.

// See `lib.rs`: deny the panic surface in production, allow it in `#[cfg(test)]` assertions.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use beyond_ai::admin::AdminApp;
use beyond_ai::config::AiConfig;
use beyond_ai::doctor;
use beyond_ai::metrics::Metrics;
use beyond_ai::proxy::AiProxy;
use beyond_ai::state::GatewayState;
use beyond_ai::store_watch::WatcherService;
use clap::{Parser, Subcommand};
use pingora_core::apps::http_app::HttpServer;
use pingora_core::server::Server;
use pingora_core::server::configuration::ServerConf;
use pingora_core::services::background::background_service;
use pingora_core::services::listening::Service as ListeningService;
use pingora_proxy::http_proxy_service;
use std::path::Path;
use std::process::exit;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
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

fn init_tracing() {
    // JSON to stdout; the `ai.usage` target carries billing facts that logfwd/OTLP ships to
    // ClickHouse. `AI_LOG` overrides the level filter.
    let filter = EnvFilter::try_from_env("AI_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().json())
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

    init_tracing();
    let config = load_config(cli.config.as_deref());
    let listen = config.listen.clone();
    let metrics_listen = config.metrics_listen.clone();
    // Capture the shutdown knobs before `config` is moved into the gateway state below.
    let grace_period_secs = config.shutdown_grace_period_secs;
    let runtime_timeout_secs = config.shutdown_runtime_timeout_secs;
    let metrics = match Metrics::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to register metrics: {e}");
            exit(1);
        }
    };
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

    // Client (app) traffic.
    let mut proxy_svc = http_proxy_service(
        &server.configuration,
        AiProxy {
            state: state.clone(),
        },
    );
    proxy_svc.add_tcp(&listen);
    server.add_service(proxy_svc);

    // slipstream watchers + NATS connectivity (connects on Pingora's runtime; see WatcherService).
    server.add_service(background_service(
        "ai-watchers",
        WatcherService {
            state: state.clone(),
        },
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
        grace_period_secs,
        runtime_timeout_secs,
        "starting beyond-ai"
    );
    server.run_forever();
}
