//! Beyond AI gateway binary: clap `Run`/`Doctor`, Pingora server bootstrap, services.

use beyond_ai::config::AiConfig;
use beyond_ai::doctor;
use beyond_ai::metrics::Metrics;
use beyond_ai::proxy::AiProxy;
use beyond_ai::state::GatewayState;
use beyond_ai::store_watch::WatcherService;
use clap::{Parser, Subcommand};
use pingora_core::server::Server;
use pingora_core::services::background::background_service;
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

fn main() {
    // rustls 0.23 requires a process-wide crypto provider for the TLS connections to providers.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

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

    let mut server = Server::new(None).expect("failed to init pingora server");
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

    // Prometheus /metrics (serves the default registry that `Metrics` registered on).
    let mut prom = pingora::services::listening::Service::prometheus_http_service();
    prom.add_tcp(&metrics_listen);
    server.add_service(prom);

    tracing::info!(%listen, %metrics_listen, "starting beyond-ai");
    server.run_forever();
}
