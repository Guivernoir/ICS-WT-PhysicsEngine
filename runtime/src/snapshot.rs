use crate::{
    control::{ControlEvaluation, OperatorState},
    models::{
        AlarmEvent, BinaryCommand, HmiSnapshot, OperatorCommand, ProcessArea, ProcessSignal,
        ScenarioOption, TrendPoint,
    },
    simulation_worker::ProcessSnapshot,
};

pub(crate) fn build_hmi_snapshot(
    raw: &ProcessSnapshot,
    operator: &OperatorState,
    control: &ControlEvaluation,
    last_error: Option<&str>,
) -> HmiSnapshot {
    let status = process_status(raw);
    HmiSnapshot {
        source: "runtime",
        connection: "connected",
        elapsed_seconds: raw.elapsed_seconds.max(0.0).round() as u64,
        mode: "Rust Runtime PCS",
        scenario: ScenarioOption {
            id: "live-runtime",
            name: "Live Runtime",
            summary: "Rust PCS state driving Python physical simulation.",
        },
        signals: signals(raw),
        areas: areas(raw, status, control),
        alarms: alarms(raw, status, control, last_error),
        commands: numeric_commands(),
        command_values: operator.command_values(),
        binary_commands: binary_commands(),
        binary_values: operator.binary_values(),
        trends: trends(raw),
        last_error: last_error.map(str::to_string),
    }
}

fn process_status(raw: &ProcessSnapshot) -> &'static str {
    if raw.system_status > 0 || raw.chlorine_outlet < 0.6 {
        "alarm"
    } else if raw.chlorine_outlet < 0.9 || !(6.8..=7.8).contains(&raw.ph_outlet) {
        "warning"
    } else {
        "normal"
    }
}

fn signals(raw: &ProcessSnapshot) -> Vec<ProcessSignal> {
    vec![
        signal(
            "FIT-INT-001",
            "Flow Rate",
            raw.flow_rate,
            "L/min",
            2,
            Some(1.0),
            Some(18.0),
            Some(0.1),
            Some(20.0),
        ),
        signal(
            "AIT-PH-IN",
            "Inlet pH",
            raw.ph_inlet,
            "pH",
            2,
            Some(6.8),
            Some(7.8),
            Some(6.5),
            Some(8.2),
        ),
        signal(
            "AIT-PH-MID",
            "Reactor pH",
            raw.ph_middle,
            "pH",
            2,
            Some(6.8),
            Some(7.8),
            Some(6.5),
            Some(8.2),
        ),
        signal(
            "AIT-PH-001",
            "Outlet pH",
            raw.ph_outlet,
            "pH",
            2,
            Some(6.8),
            Some(7.8),
            Some(6.5),
            Some(8.2),
        ),
        signal(
            "AIT-CL-001",
            "Outlet Chlorine",
            raw.chlorine_outlet,
            "mg/L",
            2,
            Some(0.9),
            Some(2.2),
            Some(0.6),
            Some(2.8),
        ),
        signal(
            "TIT-OUT-001",
            "Outlet Temperature",
            raw.temperature_outlet,
            "C",
            1,
            Some(5.0),
            Some(35.0),
            Some(1.0),
            Some(45.0),
        ),
        signal(
            "TIT-IN-001",
            "Inlet Temperature",
            raw.temperature_inlet,
            "C",
            1,
            Some(5.0),
            Some(35.0),
            Some(1.0),
            Some(45.0),
        ),
        signal(
            "HS-STAT-001",
            "System Status",
            f64::from(raw.system_status),
            "code",
            0,
            None,
            Some(0.0),
            None,
            Some(1.0),
        ),
        signal(
            "FIT-ACID-ACT",
            "Acid Actual Flow",
            raw.acid_flow,
            "L/min",
            2,
            Some(0.0),
            Some(1.5),
            Some(0.0),
            Some(2.0),
        ),
        signal(
            "FIT-CL-ACT",
            "Chlorine Actual Flow",
            raw.chlorine_flow,
            "L/min",
            2,
            Some(0.0),
            Some(0.8),
            Some(0.0),
            Some(1.0),
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn signal(
    tag: &'static str,
    label: &'static str,
    value: f64,
    unit: &'static str,
    decimals: u8,
    warning_low: Option<f64>,
    warning_high: Option<f64>,
    alarm_low: Option<f64>,
    alarm_high: Option<f64>,
) -> ProcessSignal {
    ProcessSignal {
        tag,
        label,
        value,
        unit,
        decimals,
        warning_low,
        warning_high,
        alarm_low,
        alarm_high,
        trend: "steady",
    }
}

fn areas(
    raw: &ProcessSnapshot,
    status: &'static str,
    control: &ControlEvaluation,
) -> Vec<ProcessArea> {
    let turbidity = turbidity_proxy(raw);
    vec![
        area(
            "intake",
            "Intake",
            "normal",
            raw.flow_rate,
            raw.chlorine_inlet,
            turbidity,
            "auto",
        ),
        area(
            "reactor",
            "Reactor",
            status,
            raw.flow_rate,
            raw.chlorine_outlet,
            turbidity,
            "auto",
        ),
        area(
            "disinfection",
            "Disinfection",
            status,
            raw.flow_rate,
            raw.chlorine_outlet,
            turbidity,
            "auto",
        ),
        area(
            "controls",
            "Controls",
            status,
            raw.inlet_flow,
            raw.chlorine_flow,
            turbidity,
            control.mode,
        ),
    ]
}

fn area(
    id: &'static str,
    name: &'static str,
    status: &'static str,
    flow: f64,
    residual: f64,
    turbidity_proxy: f64,
    controller_mode: &'static str,
) -> ProcessArea {
    ProcessArea {
        id,
        name,
        status,
        flow_rate: flow,
        flow_unit: "L/min",
        tank_level_percent: 0.0,
        residual_mg_l: residual,
        turbidity_ntu: turbidity_proxy,
        controller_mode,
    }
}

fn turbidity_proxy(raw: &ProcessSnapshot) -> f64 {
    let chlorine_penalty = (0.9 - raw.chlorine_outlet).max(0.0) * 0.08;
    let ph_penalty = (raw.ph_outlet - 7.3).abs() * 0.03;
    let flow_penalty = (raw.flow_rate - raw.inlet_flow).abs() * 0.01;
    (0.18 + chlorine_penalty + ph_penalty + flow_penalty).clamp(0.05, 1.2)
}

fn alarms(
    raw: &ProcessSnapshot,
    status: &'static str,
    control: &ControlEvaluation,
    last_error: Option<&str>,
) -> Vec<AlarmEvent> {
    let mut alarms = vec![
        AlarmEvent {
            id: "simulation-boundary",
            severity: "notice",
            area: "HMI",
            message: "Rust runtime owns PCS/network state; Python owns physical simulation."
                .to_string(),
            active: true,
        },
        AlarmEvent {
            id: "pcs-state",
            severity: if control.interlock_active {
                "warning"
            } else {
                "notice"
            },
            area: "PCS",
            message: control.status_message.clone(),
            active: control.interlock_active,
        },
        AlarmEvent {
            id: "process-status",
            severity: if status == "alarm" {
                "critical"
            } else {
                "warning"
            },
            area: "Process",
            message: process_message(raw, status),
            active: status != "normal" || !raw.simulation_running,
        },
    ];
    if let Some(error) = last_error {
        alarms.push(AlarmEvent {
            id: "runtime-worker-error",
            severity: "critical",
            area: "Runtime",
            message: format!("Python simulation worker error: {error}"),
            active: true,
        });
    }
    alarms
}

fn process_message(raw: &ProcessSnapshot, status: &str) -> String {
    if !raw.simulation_running {
        return "Simulation is paused through Rust PCS run-state coil.".to_string();
    }
    match status {
        "alarm" => "HydraSim process status is in alarm range.".to_string(),
        "warning" => "HydraSim process status is in warning range.".to_string(),
        _ => "No active process alarms from Python simulation snapshot.".to_string(),
    }
}

fn numeric_commands() -> Vec<OperatorCommand> {
    vec![
        command(
            "acid-flow",
            "Acid Flow",
            "HR 40001",
            "Setpoint",
            0.0,
            2.0,
            0.01,
            "L/min",
        ),
        command(
            "chlorine-flow",
            "Chlorine Flow",
            "HR 40003",
            "Setpoint",
            0.0,
            1.0,
            0.01,
            "L/min",
        ),
        command(
            "inlet-flow",
            "Inlet Flow",
            "HR 40005",
            "Setpoint",
            0.0,
            20.0,
            0.1,
            "L/min",
        ),
        command(
            "chlorine-concentration",
            "Chlorine Stock",
            "HR 40013",
            "Concentration",
            0.0,
            200.0,
            1.0,
            "mg/L",
        ),
        command(
            "acid-concentration",
            "Acid Stock",
            "HR 40011",
            "Concentration",
            0.0,
            2.0,
            0.01,
            "mol/L",
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn command(
    id: &'static str,
    label: &'static str,
    target: &'static str,
    value_label: &'static str,
    min: f64,
    max: f64,
    step: f64,
    unit: &'static str,
) -> OperatorCommand {
    OperatorCommand {
        id,
        label,
        target,
        value_label,
        min,
        max,
        step,
        unit,
    }
}

fn binary_commands() -> Vec<BinaryCommand> {
    vec![
        binary("acid-pump-enable", "Acid Pump", "Coil 00001", "Enable"),
        binary(
            "chlorine-pump-enable",
            "Chlorine Pump",
            "Coil 00002",
            "Enable",
        ),
        binary(
            "simulation-running",
            "Simulation Running",
            "Coil 00003",
            "Run state",
        ),
    ]
}

fn binary(
    id: &'static str,
    label: &'static str,
    target: &'static str,
    value_label: &'static str,
) -> BinaryCommand {
    BinaryCommand {
        id,
        label,
        target,
        value_label,
    }
}

fn trends(raw: &ProcessSnapshot) -> Vec<TrendPoint> {
    let turbidity = turbidity_proxy(raw);
    (-27..=0)
        .map(|minute| {
            let phase = (raw.elapsed_seconds + f64::from(minute * 60)) / 60.0;
            TrendPoint {
                minute,
                ph: raw.ph_outlet + phase.sin() * 0.01,
                chlorine: raw.chlorine_outlet + phase.cos() * 0.02,
                flow: raw.flow_rate + phase.sin() * 0.04,
                turbidity: turbidity + phase.cos() * 0.01,
            }
        })
        .collect()
}
