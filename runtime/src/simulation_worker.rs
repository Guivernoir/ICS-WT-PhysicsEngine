use std::{env, ffi::OsString, path::Path, process::Stdio};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
};

use crate::{
    config::SimulationSettings,
    control::{ProcessCoils, ProcessCommand},
};

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ProcessSnapshot {
    pub elapsed_seconds: f64,
    pub ph_inlet: f64,
    pub ph_middle: f64,
    pub ph_outlet: f64,
    pub chlorine_inlet: f64,
    pub chlorine_outlet: f64,
    pub flow_rate: f64,
    pub temperature_inlet: f64,
    pub temperature_outlet: f64,
    pub system_status: u16,
    pub acid_flow: f64,
    pub chlorine_flow: f64,
    pub inlet_flow: f64,
    pub simulation_running: bool,
}

pub(crate) struct SimulationWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

#[derive(Debug, Error)]
pub(crate) enum WorkerError {
    #[error("failed to start Python simulation worker: {0}")]
    Start(std::io::Error),
    #[error("Python simulation worker did not expose {0}")]
    MissingPipe(&'static str),
    #[error("failed writing to Python simulation worker: {0}")]
    Write(std::io::Error),
    #[error("failed reading from Python simulation worker: {0}")]
    Read(std::io::Error),
    #[error("Python simulation worker exited without a response")]
    Closed,
    #[error("failed to encode worker request: {0}")]
    Encode(serde_json::Error),
    #[error("failed to decode worker response {line:?}: {source}")]
    Decode {
        line: String,
        source: serde_json::Error,
    },
    #[error("Python simulation worker returned an error: {0}")]
    Worker(String),
    #[error("Python simulation worker response did not include a snapshot")]
    MissingSnapshot,
}

#[derive(Serialize)]
struct TickRequest<'a> {
    #[serde(rename = "type")]
    request_type: &'static str,
    dt: f64,
    commands: &'a ProcessCommand,
    coils: &'a ProcessCoils,
}

#[derive(Deserialize)]
struct WorkerResponse<T> {
    ok: bool,
    snapshot: Option<T>,
    error: Option<String>,
}

impl SimulationWorker {
    pub(crate) fn start(settings: &SimulationSettings) -> Result<Self, WorkerError> {
        let mut command = Command::new(&settings.python);
        command
            .arg("-m")
            .arg(&settings.module)
            .env("PYTHONPATH", python_path(&settings.python_path))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = command.spawn().map_err(WorkerError::Start)?;
        let stdin = child
            .stdin
            .take()
            .ok_or(WorkerError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(WorkerError::MissingPipe("stdout"))?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    pub(crate) async fn tick(
        &mut self,
        dt: f64,
        commands: &ProcessCommand,
        coils: &ProcessCoils,
    ) -> Result<ProcessSnapshot, WorkerError> {
        self.request(&TickRequest {
            request_type: "tick",
            dt,
            commands,
            coils,
        })
        .await
    }

    async fn request<T: Serialize, R: DeserializeOwned>(
        &mut self,
        request: &T,
    ) -> Result<R, WorkerError> {
        let mut line = serde_json::to_string(request).map_err(WorkerError::Encode)?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(WorkerError::Write)?;
        self.stdin.flush().await.map_err(WorkerError::Write)?;

        let line = self
            .stdout
            .next_line()
            .await
            .map_err(WorkerError::Read)?
            .ok_or(WorkerError::Closed)?;
        let response: WorkerResponse<R> =
            serde_json::from_str(&line).map_err(|source| WorkerError::Decode {
                line: line.clone(),
                source,
            })?;
        if !response.ok {
            return Err(WorkerError::Worker(
                response
                    .error
                    .unwrap_or_else(|| "unknown worker error".to_string()),
            ));
        }
        response.snapshot.ok_or(WorkerError::MissingSnapshot)
    }
}

impl Drop for SimulationWorker {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn python_path(source_path: &Path) -> OsString {
    let mut paths = vec![source_path.to_path_buf()];
    if let Some(existing) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).unwrap_or_else(|_| source_path.as_os_str().to_os_string())
}
