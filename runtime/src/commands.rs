use thiserror::Error;

pub(crate) const HR_ACID_FLOW: u16 = 0;
pub(crate) const HR_CHLORINE_FLOW: u16 = 2;
pub(crate) const HR_INLET_FLOW: u16 = 4;
pub(crate) const HR_ACID_CONCENTRATION: u16 = 10;
pub(crate) const HR_CHLORINE_CONCENTRATION: u16 = 12;

const COIL_ACID_ENABLE: u16 = 0;
const COIL_CHLORINE_ENABLE: u16 = 1;
const COIL_SIMULATION_RUNNING: u16 = 2;

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("unknown HMI command {0:?}")]
    Unknown(String),
    #[error("{id} value {value} outside allowed range [{min}, {max}]")]
    OutOfRange {
        id: String,
        value: f64,
        min: f64,
        max: f64,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum NumericCommand {
    AcidFlow,
    ChlorineFlow,
    InletFlow,
    AcidConcentration,
    ChlorineConcentration,
}

impl NumericCommand {
    pub(crate) fn from_id(id: &str) -> Result<Self, CommandError> {
        match id {
            "acid-flow" => Ok(Self::AcidFlow),
            "chlorine-flow" => Ok(Self::ChlorineFlow),
            "inlet-flow" => Ok(Self::InletFlow),
            "acid-concentration" => Ok(Self::AcidConcentration),
            "chlorine-concentration" => Ok(Self::ChlorineConcentration),
            _ => Err(CommandError::Unknown(id.to_string())),
        }
    }

    pub(crate) const fn from_address(address: u16) -> Option<Self> {
        match address {
            HR_ACID_FLOW => Some(Self::AcidFlow),
            HR_CHLORINE_FLOW => Some(Self::ChlorineFlow),
            HR_INLET_FLOW => Some(Self::InletFlow),
            HR_ACID_CONCENTRATION => Some(Self::AcidConcentration),
            HR_CHLORINE_CONCENTRATION => Some(Self::ChlorineConcentration),
            _ => None,
        }
    }

    pub(crate) const fn id(self) -> &'static str {
        self.bounds().0
    }

    const fn bounds(self) -> (&'static str, f64, f64) {
        match self {
            Self::AcidFlow => ("acid-flow", 0.0, 2.0),
            Self::ChlorineFlow => ("chlorine-flow", 0.0, 1.0),
            Self::InletFlow => ("inlet-flow", 0.0, 20.0),
            Self::AcidConcentration => ("acid-concentration", 0.0, 2.0),
            Self::ChlorineConcentration => ("chlorine-concentration", 0.0, 200.0),
        }
    }

    pub(crate) fn validate(self, value: f64) -> Result<(), CommandError> {
        let (id, min, max) = self.bounds();
        if (min..=max).contains(&value) {
            Ok(())
        } else {
            Err(CommandError::OutOfRange {
                id: id.to_string(),
                value,
                min,
                max,
            })
        }
    }

    pub(crate) const fn default_value(self) -> f64 {
        match self {
            Self::AcidFlow => 0.0,
            Self::ChlorineFlow => 0.2,
            Self::InletFlow => 5.0,
            Self::AcidConcentration => 0.1,
            Self::ChlorineConcentration => 60.0,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum BinaryCommandKind {
    AcidPumpEnable,
    ChlorinePumpEnable,
    SimulationRunning,
}

impl BinaryCommandKind {
    pub(crate) fn from_id(id: &str) -> Result<Self, CommandError> {
        match id {
            "acid-pump-enable" => Ok(Self::AcidPumpEnable),
            "chlorine-pump-enable" => Ok(Self::ChlorinePumpEnable),
            "simulation-running" => Ok(Self::SimulationRunning),
            _ => Err(CommandError::Unknown(id.to_string())),
        }
    }

    pub(crate) const fn from_address(address: u16) -> Option<Self> {
        match address {
            COIL_ACID_ENABLE => Some(Self::AcidPumpEnable),
            COIL_CHLORINE_ENABLE => Some(Self::ChlorinePumpEnable),
            COIL_SIMULATION_RUNNING => Some(Self::SimulationRunning),
            _ => None,
        }
    }

    pub(crate) const fn id(self) -> &'static str {
        match self {
            Self::AcidPumpEnable => "acid-pump-enable",
            Self::ChlorinePumpEnable => "chlorine-pump-enable",
            Self::SimulationRunning => "simulation-running",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_range_numeric_commands() {
        let command = NumericCommand::from_id("chlorine-flow");
        assert!(command.is_ok_and(|item| item.validate(0.5).is_ok()));
        assert!(
            NumericCommand::from_id("chlorine-flow").is_ok_and(|item| item.validate(1.5).is_err())
        );
    }

    #[test]
    fn maps_binary_commands_to_hydrasim_coils() {
        assert_eq!(
            BinaryCommandKind::from_address(2).map(BinaryCommandKind::id),
            Some("simulation-running"),
        );
    }
}
