use std::{sync::Arc, time::Duration};

use thiserror::Error;
use tokio::{
    sync::Mutex,
    time::{MissedTickBehavior, interval},
};
use tracing::warn;

use crate::{
    commands::CommandError,
    config::SimulationSettings,
    control::{ControlEvaluation, OperatorState},
    models::{BinaryValues, CommandValues, HmiSnapshot},
    simulation_worker::{ProcessSnapshot, SimulationWorker, WorkerError},
    snapshot::build_hmi_snapshot,
};

#[derive(Clone)]
pub(crate) struct RuntimeHandle {
    inner: Arc<Mutex<RuntimeState>>,
}

struct RuntimeState {
    operator: OperatorState,
    worker: SimulationWorker,
    last_process: Option<ProcessSnapshot>,
    last_control: ControlEvaluation,
    last_error: Option<String>,
}

#[derive(Debug, Error)]
pub(crate) enum RuntimeError {
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    Worker(#[from] WorkerError),
}

impl RuntimeHandle {
    pub(crate) fn start(
        simulation: &SimulationSettings,
        tick_interval: Duration,
    ) -> Result<Self, RuntimeError> {
        let operator = OperatorState::default();
        let worker = SimulationWorker::start(simulation)?;
        let last_control = operator.evaluate();
        let handle = Self {
            inner: Arc::new(Mutex::new(RuntimeState {
                operator,
                worker,
                last_process: None,
                last_control,
                last_error: None,
            })),
        };
        handle.spawn_tick_loop(tick_interval);
        Ok(handle)
    }

    pub(crate) async fn snapshot(&self) -> Result<HmiSnapshot, RuntimeError> {
        let mut state = self.inner.lock().await;
        if state.last_process.is_none() {
            state.tick_once(0.0).await?;
        }
        let process = state
            .last_process
            .as_ref()
            .ok_or(WorkerError::MissingSnapshot)?;
        Ok(build_hmi_snapshot(
            process,
            &state.operator,
            &state.last_control,
            state.last_error.as_deref(),
        ))
    }

    pub(crate) async fn write_numeric(&self, id: &str, value: f64) -> Result<(), RuntimeError> {
        let mut state = self.inner.lock().await;
        state.operator.write_numeric(id, value)?;
        state.tick_once(0.0).await?;
        Ok(())
    }

    pub(crate) async fn write_binary(&self, id: &str, enabled: bool) -> Result<(), RuntimeError> {
        let mut state = self.inner.lock().await;
        state.operator.write_binary(id, enabled)?;
        state.tick_once(0.0).await?;
        Ok(())
    }

    pub(crate) async fn modbus_image(&self) -> Result<ModbusImage, RuntimeError> {
        let mut state = self.inner.lock().await;
        if state.last_process.is_none() {
            state.tick_once(0.0).await?;
        }
        let process = state
            .last_process
            .as_ref()
            .ok_or(WorkerError::MissingSnapshot)?;
        Ok(ModbusImage::from_state(
            process,
            &state.operator.command_values(),
            &state.operator.binary_values(),
        ))
    }

    fn spawn_tick_loop(&self, tick_interval: Duration) {
        let handle = self.clone();
        tokio::spawn(async move {
            let mut ticker = interval(tick_interval);
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                if let Err(error) = handle.tick(tick_interval.as_secs_f64()).await {
                    warn!("runtime tick failed: {error}");
                }
            }
        });
    }

    async fn tick(&self, dt: f64) -> Result<(), RuntimeError> {
        let mut state = self.inner.lock().await;
        state.tick_once(dt).await
    }
}

impl RuntimeState {
    async fn tick_once(&mut self, dt: f64) -> Result<(), RuntimeError> {
        let control = self.operator.evaluate();
        match self
            .worker
            .tick(dt, &control.process_command, &control.process_coils)
            .await
        {
            Ok(process) => {
                self.last_process = Some(process);
                self.last_control = control;
                self.last_error = None;
                Ok(())
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                Err(error.into())
            }
        }
    }
}

pub(crate) struct ModbusImage {
    pub input_registers: Vec<u16>,
    pub holding_registers: Vec<u16>,
    pub coils: Vec<bool>,
}

impl ModbusImage {
    fn from_state(
        process: &ProcessSnapshot,
        command_values: &CommandValues,
        binary_values: &BinaryValues,
    ) -> Self {
        let mut input_registers = vec![0; 104];
        write_f32(&mut input_registers, 0, process.ph_inlet);
        write_f32(&mut input_registers, 2, process.ph_middle);
        write_f32(&mut input_registers, 4, process.ph_outlet);
        write_f32(&mut input_registers, 6, process.chlorine_inlet);
        write_f32(&mut input_registers, 8, process.chlorine_outlet);
        write_f32(&mut input_registers, 10, process.flow_rate);
        write_f32(&mut input_registers, 12, process.temperature_inlet);
        write_f32(&mut input_registers, 14, process.temperature_outlet);
        write_f32(&mut input_registers, 100, process.elapsed_seconds);
        input_registers[102] = process.system_status;

        let mut holding_registers = vec![0; 14];
        write_command(&mut holding_registers, 0, command_values, "acid-flow");
        write_command(&mut holding_registers, 2, command_values, "chlorine-flow");
        write_command(&mut holding_registers, 4, command_values, "inlet-flow");
        write_command(
            &mut holding_registers,
            10,
            command_values,
            "acid-concentration",
        );
        write_command(
            &mut holding_registers,
            12,
            command_values,
            "chlorine-concentration",
        );

        let coils = vec![
            *binary_values.get("acid-pump-enable").unwrap_or(&true),
            *binary_values.get("chlorine-pump-enable").unwrap_or(&true),
            *binary_values.get("simulation-running").unwrap_or(&true),
        ];
        Self {
            input_registers,
            holding_registers,
            coils,
        }
    }
}

fn write_command(words: &mut [u16], offset: usize, values: &CommandValues, key: &str) {
    write_f32(words, offset, *values.get(key).unwrap_or(&0.0));
}

fn write_f32(words: &mut [u16], offset: usize, value: f64) {
    let bits = (value as f32).to_bits();
    if let Some(word) = words.get_mut(offset) {
        *word = (bits >> 16) as u16;
    }
    if let Some(word) = words.get_mut(offset + 1) {
        *word = bits as u16;
    }
}
