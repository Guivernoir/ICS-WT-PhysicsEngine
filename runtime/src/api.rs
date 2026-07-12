use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, put},
};
use serde::Serialize;

use crate::{
    AppState,
    commands::CommandError,
    models::{BinaryCommandRequest, HealthResponse, NumericCommandRequest},
    runtime::RuntimeError,
};

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/snapshot", get(snapshot))
        .route("/commands/{id}", put(write_command))
        .route("/coils/{id}", put(write_coil))
        .with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        runtime: "rust",
        modbus_enabled: state.modbus_enabled,
        modbus_bind: state.modbus_bind_addr.map(|addr| addr.to_string()),
        simulation_worker: state.simulation_worker.clone(),
        simulation_only: true,
    })
}

async fn snapshot(State(state): State<Arc<AppState>>) -> Result<Response, ApiError> {
    let snapshot = state.runtime.snapshot().await?;
    Ok(Json(snapshot).into_response())
}

async fn write_command(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(payload): Json<NumericCommandRequest>,
) -> Result<Response, ApiError> {
    state.runtime.write_numeric(&id, payload.value).await?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn write_coil(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(payload): Json<BinaryCommandRequest>,
) -> Result<Response, ApiError> {
    state.runtime.write_binary(&id, payload.enabled).await?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

#[derive(Debug)]
enum ApiError {
    BadCommand(CommandError),
    Runtime(RuntimeError),
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

impl From<CommandError> for ApiError {
    fn from(error: CommandError) -> Self {
        Self::BadCommand(error)
    }
}

impl From<RuntimeError> for ApiError {
    fn from(error: RuntimeError) -> Self {
        match error {
            RuntimeError::Command(error) => Self::BadCommand(error),
            other => Self::Runtime(other),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error) = match self {
            Self::BadCommand(error) => (StatusCode::BAD_REQUEST, error.to_string()),
            Self::Runtime(error) => (StatusCode::BAD_GATEWAY, error.to_string()),
        };
        (status, Json(ErrorBody { error })).into_response()
    }
}
