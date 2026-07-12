use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub type CommandValues = BTreeMap<String, f64>;
pub type BinaryValues = BTreeMap<String, bool>;

#[derive(Debug, Deserialize)]
pub struct NumericCommandRequest {
    pub value: f64,
}

#[derive(Debug, Deserialize)]
pub struct BinaryCommandRequest {
    pub enabled: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    pub status: &'static str,
    pub runtime: &'static str,
    pub modbus_enabled: bool,
    pub modbus_bind: Option<String>,
    pub simulation_worker: String,
    pub simulation_only: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HmiSnapshot {
    pub source: &'static str,
    pub connection: &'static str,
    pub elapsed_seconds: u64,
    pub mode: &'static str,
    pub scenario: ScenarioOption,
    pub signals: Vec<ProcessSignal>,
    pub areas: Vec<ProcessArea>,
    pub alarms: Vec<AlarmEvent>,
    pub commands: Vec<OperatorCommand>,
    pub command_values: CommandValues,
    pub binary_commands: Vec<BinaryCommand>,
    pub binary_values: BinaryValues,
    pub trends: Vec<TrendPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ScenarioOption {
    pub id: &'static str,
    pub name: &'static str,
    pub summary: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessSignal {
    pub tag: &'static str,
    pub label: &'static str,
    pub value: f64,
    pub unit: &'static str,
    pub decimals: u8,
    pub warning_low: Option<f64>,
    pub warning_high: Option<f64>,
    pub alarm_low: Option<f64>,
    pub alarm_high: Option<f64>,
    pub trend: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessArea {
    pub id: &'static str,
    pub name: &'static str,
    pub status: &'static str,
    pub flow_rate: f64,
    pub flow_unit: &'static str,
    pub tank_level_percent: f64,
    pub residual_mg_l: f64,
    pub turbidity_ntu: f64,
    pub controller_mode: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AlarmEvent {
    pub id: &'static str,
    pub severity: &'static str,
    pub area: &'static str,
    pub message: String,
    pub active: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OperatorCommand {
    pub id: &'static str,
    pub label: &'static str,
    pub target: &'static str,
    pub value_label: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub unit: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BinaryCommand {
    pub id: &'static str,
    pub label: &'static str,
    pub target: &'static str,
    pub value_label: &'static str,
}

#[derive(Debug, Serialize)]
pub struct TrendPoint {
    pub minute: i32,
    pub ph: f64,
    pub chlorine: f64,
    pub flow: f64,
    pub turbidity: f64,
}
