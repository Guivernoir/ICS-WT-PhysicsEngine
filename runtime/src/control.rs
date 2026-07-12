use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    commands::{BinaryCommandKind, CommandError, NumericCommand},
    models::{BinaryValues, CommandValues},
};

#[derive(Clone, Debug)]
pub(crate) struct OperatorState {
    numeric: CommandValues,
    binary: BinaryValues,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProcessCommand {
    pub acid_flow: f64,
    pub chlorine_flow: f64,
    pub inlet_flow: f64,
    pub acid_concentration: f64,
    pub chlorine_concentration: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProcessCoils {
    pub acid_pump_enable: bool,
    pub chlorine_pump_enable: bool,
    pub simulation_running: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ControlEvaluation {
    pub process_command: ProcessCommand,
    pub process_coils: ProcessCoils,
    pub mode: &'static str,
    pub interlock_active: bool,
    pub status_message: String,
}

impl Default for OperatorState {
    fn default() -> Self {
        Self {
            numeric: BTreeMap::from([
                numeric_default(NumericCommand::AcidFlow),
                numeric_default(NumericCommand::ChlorineFlow),
                numeric_default(NumericCommand::InletFlow),
                numeric_default(NumericCommand::AcidConcentration),
                numeric_default(NumericCommand::ChlorineConcentration),
            ]),
            binary: BTreeMap::from([
                binary_default(BinaryCommandKind::AcidPumpEnable),
                binary_default(BinaryCommandKind::ChlorinePumpEnable),
                binary_default(BinaryCommandKind::SimulationRunning),
            ]),
        }
    }
}

impl OperatorState {
    pub(crate) fn command_values(&self) -> CommandValues {
        self.numeric.clone()
    }

    pub(crate) fn binary_values(&self) -> BinaryValues {
        self.binary.clone()
    }

    pub(crate) fn write_numeric(&mut self, id: &str, value: f64) -> Result<(), CommandError> {
        let command = NumericCommand::from_id(id)?;
        command.validate(value)?;
        self.numeric.insert(command.id().to_string(), value);
        Ok(())
    }

    pub(crate) fn write_binary(&mut self, id: &str, enabled: bool) -> Result<(), CommandError> {
        let command = BinaryCommandKind::from_id(id)?;
        self.binary.insert(command.id().to_string(), enabled);
        Ok(())
    }

    pub(crate) fn evaluate(&self) -> ControlEvaluation {
        let acid_enabled = self.binary_value(BinaryCommandKind::AcidPumpEnable);
        let chlorine_enabled = self.binary_value(BinaryCommandKind::ChlorinePumpEnable);
        let simulation_running = self.binary_value(BinaryCommandKind::SimulationRunning);
        let inlet_flow = self.numeric_value(NumericCommand::InletFlow);

        let mut acid_flow = self.numeric_value(NumericCommand::AcidFlow);
        let mut chlorine_flow = self.numeric_value(NumericCommand::ChlorineFlow);
        let mut interlock_active = false;
        let mut status_message = "PCS outputs accepted operator intent.".to_string();

        if !acid_enabled {
            acid_flow = 0.0;
            interlock_active = true;
            status_message = "Acid pump disabled by Rust PCS coil.".to_string();
        }
        if !chlorine_enabled {
            chlorine_flow = 0.0;
            interlock_active = true;
            status_message = "Chlorine pump disabled by Rust PCS coil.".to_string();
        }
        if inlet_flow < 0.1 && (acid_flow > 0.0 || chlorine_flow > 0.0) {
            acid_flow = 0.0;
            chlorine_flow = 0.0;
            interlock_active = true;
            status_message = "Dosing interlocked because inlet flow is closed.".to_string();
        }

        ControlEvaluation {
            process_command: ProcessCommand {
                acid_flow,
                chlorine_flow,
                inlet_flow,
                acid_concentration: self.numeric_value(NumericCommand::AcidConcentration),
                chlorine_concentration: self.numeric_value(NumericCommand::ChlorineConcentration),
            },
            process_coils: ProcessCoils {
                acid_pump_enable: acid_enabled,
                chlorine_pump_enable: chlorine_enabled,
                simulation_running,
            },
            mode: if simulation_running { "auto" } else { "hold" },
            interlock_active,
            status_message,
        }
    }

    fn numeric_value(&self, command: NumericCommand) -> f64 {
        self.numeric
            .get(command.id())
            .copied()
            .unwrap_or_else(|| command.default_value())
    }

    fn binary_value(&self, command: BinaryCommandKind) -> bool {
        self.binary.get(command.id()).copied().unwrap_or(true)
    }
}

fn numeric_default(command: NumericCommand) -> (String, f64) {
    (command.id().to_string(), command.default_value())
}

fn binary_default(command: BinaryCommandKind) -> (String, bool) {
    (command.id().to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pump_disable_interlocks_final_flow() {
        let mut state = OperatorState::default();
        assert!(state.write_numeric("chlorine-flow", 0.8).is_ok());
        assert!(state.write_binary("chlorine-pump-enable", false).is_ok());

        let output = state.evaluate();

        assert_eq!(output.process_command.chlorine_flow, 0.0);
        assert!(output.interlock_active);
    }
}
