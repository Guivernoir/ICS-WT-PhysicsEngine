"""Runtime helper functions for HydraSim's process loop."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Dict, Optional, Tuple

from .actuators import create_realistic_actuator_suite
from .core import BoundaryConditions, ReactorState
from .sensors import (
    SensorFault,
    SensorReading,
    SensorStatus,
    create_realistic_sensor_suite,
)

try:
    from .modbus import ModbusSlave
except ModuleNotFoundError as exc:
    if exc.name != "pymodbus":
        raise
    ModbusSlave = Any  # type: ignore

logger = logging.getLogger(__name__)


# Input validation helpers
def validate_flow_rate(value: float, max_value: float = 20.0) -> float:
    """Validate and clamp flow rate within safe bounds."""
    if not isinstance(value, (int, float)):
        return 0.0
    if value != value:  # Check for NaN
        return 0.0
    return max(0.0, min(float(value), max_value))


def validate_concentration(value: float, max_value: float = 1.0) -> float:
    """Validate and clamp concentration within safe bounds."""
    if not isinstance(value, (int, float)):
        return 0.0
    if value != value:  # Check for NaN
        return 0.0
    return max(0.0, min(float(value), max_value))


def validate_ph(value: float) -> float:
    """Validate pH value within physical bounds."""
    if not isinstance(value, (int, float)):
        return 7.0
    if value != value:  # Check for NaN
        return 7.0
    return max(0.0, min(float(value), 14.0))


def initialize_sensors(config, sim_start_time: float, verbose: bool = False):
    """Initialize and calibrate sensors with error handling."""
    logger.info("Initializing sensor suite...")

    try:
        sensors = create_realistic_sensor_suite(config)
    except Exception as e:
        logger.error(f"Failed to create sensor suite: {type(e).__name__}")
        raise RuntimeError("Sensor initialization failed")

    # Calibrate all sensors at startup
    calibration_errors = 0
    for name, sensor in sensors.items():
        try:
            if "pH" in name:
                sensor.calibrate(7.0, sim_start_time, "system_init")
            elif "chlorine" in name:
                sensor.calibrate(config.initial_chlorine, sim_start_time, "system_init")
            elif "temp" in name:
                sensor.calibrate(config.temperature, sim_start_time, "system_init")
            elif "flow" in name:
                sensor.calibrate(config.flow_rate, sim_start_time, "system_init")

            if verbose:
                logger.info(f"  Calibrated {name}")

        except Exception:
            calibration_errors += 1
            logger.warning(f"  Could not calibrate {name}")

    if calibration_errors > len(sensors) // 2:
        raise RuntimeError("Too many sensor calibration failures")

    logger.info(f"Initialized {len(sensors)} sensors ({calibration_errors} errors)")
    return sensors


def read_all_sensors(
    sensors: Dict, state: ReactorState, sim_time: float, verbose: bool = False
) -> Dict[str, SensorReading]:
    """Read all sensors with graceful error handling."""
    readings = {}
    error_count = 0

    for name, sensor in sensors.items():
        try:
            reading = sensor.read(state, current_time=sim_time)
            readings[name] = reading

            # Log warnings/faults
            if reading.status != SensorStatus.NORMAL:
                if verbose or reading.status not in [
                    SensorStatus.WARMING_UP,
                    SensorStatus.CALIBRATING,
                ]:
                    logger.warning(f"{name}: {reading.status.value}")

            if reading.fault != SensorFault.NONE:
                logger.error(f"{name}: FAULT - {reading.fault.value}")
                error_count += 1

        except Exception:
            error_count += 1
            # Graceful degradation: use safe default values
            readings[name] = SensorReading(
                timestamp=sim_time,
                value=float("nan"),
                raw_value=float("nan"),
                noise=0.0,
                drift=0.0,
                status=SensorStatus.FAILED,
                uncertainty=float("inf"),
                fault=SensorFault.OPEN_CIRCUIT,
            )

    # Alert if too many sensors are failing
    if error_count > len(sensors) // 2:
        logger.error(f"{error_count}/{len(sensors)} sensors in fault state")

    return readings


def update_modbus_inputs(
    slave: Optional[ModbusSlave], readings: Dict[str, SensorReading], sim_time: float
) -> bool:
    """
    Update Modbus input registers with sensor values.

    Returns:
        True if update succeeded, False otherwise
    """
    if slave is None or not slave.is_running:
        return False

    # Helper to safely get value (return 0.0 if None or NaN)
    def safe_value(key: str) -> float:
        reading = readings.get(key)
        if reading is None:
            return 0.0
        val = reading.value
        if val != val or val == float("inf") or val == float("-inf"):
            return 0.0
        return val

    # Helper to check fault status
    def has_fault(key: str) -> bool:
        reading = readings.get(key)
        return reading is not None and reading.fault != SensorFault.NONE

    try:
        # Update analog inputs (input registers)
        slave.update_input_register("pH_inlet", safe_value("pH_inlet"))
        slave.update_input_register("pH_middle", safe_value("pH_middle"))
        slave.update_input_register("pH_outlet", safe_value("pH_outlet"))

        slave.update_input_register("chlorine_inlet", safe_value("chlorine_inlet"))
        slave.update_input_register("chlorine_outlet", safe_value("chlorine_outlet"))

        slave.update_input_register("flow_rate", safe_value("flow_main"))

        slave.update_input_register("temperature_inlet", safe_value("temp_inlet"))
        slave.update_input_register("temperature_outlet", safe_value("temp_outlet"))

        # Update system status inputs
        slave.update_input_register("simulation_time", sim_time)

        # Calculate aggregate system status (0=OK, 1=Fault)
        any_fault = any(r.fault != SensorFault.NONE for r in readings.values())
        slave.update_input_register("system_status", 1 if any_fault else 0)

        # Update discrete inputs (fault bits)
        slave.update_discrete_input("sensor_fault_pH_inlet", has_fault("pH_inlet"))
        slave.update_discrete_input("sensor_fault_pH_outlet", has_fault("pH_outlet"))

        chlorine_fault = has_fault("chlorine_inlet") or has_fault("chlorine_outlet")
        slave.update_discrete_input("sensor_fault_chlorine", chlorine_fault)

        return True

    except Exception as e:
        logger.error(f"Modbus update failed: {type(e).__name__}")
        return False


def read_modbus_commands(slave: Optional[ModbusSlave]) -> Tuple[float, float, float]:
    """
    Read actuator commands from Modbus with range checks.

    Returns:
        Tuple of (acid_flow_rate, chlorine_flow_rate, inlet_flow_rate)
    """
    if slave is None or not slave.is_running:
        return 0.0, 0.0, 0.0

    try:
        # Read commands with validation
        acid_rate = slave.read_holding_register("acid_flow_rate")
        chlorine_rate = slave.read_holding_register("chlorine_flow_rate")
        inlet_rate = slave.read_holding_register("inlet_flow_rate")

        # Clamp to configured ranges.
        acid_rate = validate_flow_rate(acid_rate, max_value=2.0)
        chlorine_rate = validate_flow_rate(chlorine_rate, max_value=1.0)
        inlet_rate = validate_flow_rate(inlet_rate, max_value=20.0)

        return acid_rate, chlorine_rate, inlet_rate

    except Exception as e:
        logger.error(f"Modbus read failed: {type(e).__name__}")
        return 0.0, 0.0, 0.0


def read_modbus_enable_bits(slave: Optional[ModbusSlave]) -> Tuple[bool, bool, bool]:
    """
    Read actuator and simulation enable bits from Modbus coils.

    Returns:
        (acid_enabled, chlorine_enabled, simulation_running)
    """
    if slave is None or not slave.is_running:
        return True, True, True

    try:
        acid_enabled = slave.read_coil("acid_pump_enable")
        chlorine_enabled = slave.read_coil("chlorine_pump_enable")
        simulation_running = slave.read_coil("simulation_running")
        return bool(acid_enabled), bool(chlorine_enabled), bool(simulation_running)
    except Exception:
        # Fail-open for operation continuity if comms are degraded
        return True, True, True


def read_modbus_dosing_concentrations(
    slave: Optional[ModbusSlave], boundary: BoundaryConditions
) -> Tuple[float, float]:
    """
    Read dosing stock concentrations from Modbus holding registers.

    Returns:
        (acid_concentration_mol_L, chlorine_concentration_mg_L)
    """
    if slave is None or not slave.is_running:
        return boundary.acid_concentration, boundary.chlorine_concentration

    try:
        acid_conc = slave.read_holding_register("acid_concentration")
        chlorine_conc = slave.read_holding_register("chlorine_concentration")

        acid_conc = validate_concentration(acid_conc, max_value=5.0)  # mol/L
        chlorine_conc = validate_concentration(chlorine_conc, max_value=200.0)  # mg/L

        return acid_conc, chlorine_conc
    except Exception:
        return boundary.acid_concentration, boundary.chlorine_concentration


def initialize_actuators(boundary: BoundaryConditions) -> Dict[str, Any]:
    """Initialize and prime realistic actuator models."""
    actuators: Dict[str, Any] = create_realistic_actuator_suite(
        {
            "max_acid_flow": 2.0,
            "max_chlorine_flow": 1.0,
            "max_inlet_flow": 20.0,
            "acid_pressure_drop": 2.0,
            "chlorine_discharge_pressure": 2.0,
            "inlet_pressure_drop": 1.5,
        }
    )

    # Prime actuators to initial operating point so simulation starts near steady state.
    actuators["acid_valve"].set_flow_rate(
        validate_flow_rate(boundary.acid_flow_rate, 2.0)
    )
    actuators["chlorine_pump"].set_flow_rate(
        validate_flow_rate(boundary.chlorine_flow_rate, 1.0)
    )
    actuators["inlet_valve"].set_flow_rate(
        validate_flow_rate(boundary.inlet_flow_rate, 20.0)
    )

    for _ in range(3):
        actuators["acid_valve"].step(10.0)
        actuators["chlorine_pump"].step(10.0)
        actuators["inlet_valve"].step(10.0)

    return actuators


def apply_actuator_commands(
    actuators: Dict[str, Any],
    commands: Tuple[float, float, float],
    enable_bits: Tuple[bool, bool, bool],
    current_inlet_flow: float,
) -> None:
    """Apply desired flow commands to actuator setpoints."""
    acid_rate, chlorine_rate, inlet_rate = commands
    acid_enabled, chlorine_enabled, _simulation_running = enable_bits

    target_acid = acid_rate if acid_enabled else 0.0
    target_chlorine = chlorine_rate if chlorine_enabled else 0.0
    # Respect explicit low-flow/closed commands for main inlet valve.
    # Keep current flow only for invalid negative values (defensive fallback).
    target_inlet = inlet_rate if inlet_rate >= 0.0 else current_inlet_flow

    actuators["acid_valve"].set_flow_rate(validate_flow_rate(target_acid, 2.0))
    actuators["chlorine_pump"].set_flow_rate(validate_flow_rate(target_chlorine, 1.0))
    actuators["inlet_valve"].set_flow_rate(validate_flow_rate(target_inlet, 20.0))


def step_actuators_into_boundary(
    actuators: Dict[str, Any],
    boundary: BoundaryConditions,
    dt: float,
) -> None:
    """Advance actuator dynamics and map actual outputs into reactor boundary flows."""
    acid_flow = actuators["acid_valve"].step(dt)
    chlorine_flow = actuators["chlorine_pump"].step(dt)
    inlet_flow = actuators["inlet_valve"].step(dt)

    boundary.acid_flow_rate = validate_flow_rate(acid_flow, max_value=2.0)
    boundary.chlorine_flow_rate = validate_flow_rate(chlorine_flow, max_value=1.0)
    boundary.inlet_flow_rate = validate_flow_rate(inlet_flow, max_value=20.0)


def initialize_modbus_defaults(
    slave: Optional[ModbusSlave], boundary: BoundaryConditions
) -> None:
    """Prime Modbus writable registers/coils with meaningful startup defaults."""
    if slave is None or not slave.is_running:
        return

    with suppress(Exception):
        slave.write_holding_register("acid_flow_rate", boundary.acid_flow_rate)
        slave.write_holding_register("chlorine_flow_rate", boundary.chlorine_flow_rate)
        slave.write_holding_register("inlet_flow_rate", boundary.inlet_flow_rate)
        slave.write_holding_register("acid_concentration", boundary.acid_concentration)
        slave.write_holding_register(
            "chlorine_concentration", boundary.chlorine_concentration
        )
        slave.write_coil("acid_pump_enable", True)
        slave.write_coil("chlorine_pump_enable", True)
        slave.write_coil("simulation_running", True)
