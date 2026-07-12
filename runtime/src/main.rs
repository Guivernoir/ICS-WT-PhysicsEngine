mod api;
mod commands;
mod config;
mod control;
mod modbus_server;
mod models;
mod runtime;
mod simulation_worker;
mod snapshot;

use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::Router;
use config::Settings;
use runtime::RuntimeHandle;
use tokio::net::TcpListener;
use tower_http::{
    cors::{Any, CorsLayer},
    services::{ServeDir, ServeFile},
};
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Clone)]
struct AppState {
    runtime: RuntimeHandle,
    modbus_enabled: bool,
    modbus_bind_addr: Option<SocketAddr>,
    simulation_worker: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let settings = Settings::from_env()?;
    let bind_addr = settings.bind_addr;
    let modbus_enabled = settings.modbus.enabled;
    let modbus_bind_addr = settings.modbus.bind_addr;
    let hmi_dir = settings.hmi_dir;
    let simulation_worker = format!(
        "{} -m {}",
        settings.simulation.python, settings.simulation.module
    );
    let runtime = RuntimeHandle::start(&settings.simulation, settings.tick_interval)?;
    let state = Arc::new(AppState {
        runtime: runtime.clone(),
        modbus_enabled,
        modbus_bind_addr: modbus_enabled.then_some(modbus_bind_addr),
        simulation_worker,
    });

    if modbus_enabled {
        let runtime_for_modbus = runtime.clone();
        tokio::spawn(async move {
            if let Err(error) = modbus_server::serve(modbus_bind_addr, runtime_for_modbus).await {
                error!("HydraSim Rust Modbus server stopped: {error}");
            }
        });
    }

    let app = Router::new()
        .nest("/api", api::router(Arc::clone(&state)))
        .fallback_service(hmi_service(hmi_dir.clone()))
        .layer(cors_layer());

    let listener = TcpListener::bind(bind_addr).await?;
    log_startup(bind_addr, modbus_enabled, modbus_bind_addr, &hmi_dir);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("hydrasim_runtime=info,tower_http=info"));
    fmt().with_env_filter(filter).init();
}

fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

fn hmi_service(hmi_dir: PathBuf) -> ServeDir<ServeFile> {
    ServeDir::new(&hmi_dir)
        .precompressed_br()
        .precompressed_gzip()
        .fallback(ServeFile::new(hmi_dir.join("index.html")))
}

fn log_startup(
    bind_addr: SocketAddr,
    modbus_enabled: bool,
    modbus_bind_addr: SocketAddr,
    hmi_dir: &Path,
) {
    if hmi_dir.join("index.html").is_file() {
        info!("HydraSim HMI served from {}", hmi_dir.display());
    } else {
        warn!(
            "HydraSim HMI build not found at {}; run `cd hmi && npm run build` or set HS_RUNTIME_HMI_DIR",
            hmi_dir.display()
        );
    }

    if modbus_enabled {
        info!(
            "HydraSim Rust runtime listening on http://{} with Modbus {}",
            bind_addr, modbus_bind_addr
        );
    } else {
        info!(
            "HydraSim Rust runtime listening on http://{} with Modbus disabled",
            bind_addr
        );
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
