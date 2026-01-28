"""
Main Simulation Orchestrator (HARDENED)
========================================

Security-hardened entry point for water treatment reactor simulation.

Author: Guilherme F. G. Santos (hardened)
Date: January 2026
"""

import argparse
import time
import logging
import signal
import sys
from typing import Dict, Tuple, Optional
from contextlib import suppress

# Physics engine
from .core import IntegratedCSTR, ReactorConfiguration, BoundaryConditions, ReactorState

# Sensors
from .sensors import (
    create_realistic_sensor_suite,
    SensorReading,
    SensorStatus,
    SensorFault,
)

# Modbus (use hardened version)
from .modbus import ModbusSlave, ModbusRegisterMap, ModbusServerConfig

# Configure logging (no verbose stack traces in production)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global running flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for clean shutdown."""
    global running
    logger.info("Shutdown signal received. Stopping simulation...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Security: Input validation functions (zero-trust principle)
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
                logger.info(f"  ✓ Calibrated {name}")

        except Exception:
            calibration_errors += 1
            logger.warning(f"  ⚠ Could not calibrate {name}")

    if calibration_errors > len(sensors) // 2:
        raise RuntimeError("Too many sensor calibration failures")

    logger.info(f"✓ Initialized {len(sensors)} sensors ({calibration_errors} errors)")
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
                status=SensorStatus.FAULT,
                uncertainty=float("inf"),
                fault=SensorFault.SENSOR_ERROR,
            )

    # Alert if too many sensors are failing
    if error_count > len(sensors) // 2:
        logger.error(f"CRITICAL: {error_count}/{len(sensors)} sensors in fault state")

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
    Read actuator commands from Modbus with validation (zero-trust).

    Returns:
        Tuple of (acid_flow_rate, chlorine_flow_rate, inlet_flow_rate)
    """
    if slave is None or not slave.is_running:
        return 0.0, 0.0, 5.0  # Safe defaults

    try:
        # Read commands with validation
        acid_rate = slave.read_holding_register("acid_flow_rate")
        chlorine_rate = slave.read_holding_register("chlorine_flow_rate")
        inlet_rate = slave.read_holding_register("inlet_flow_rate")

        # Apply zero-trust validation
        acid_rate = validate_flow_rate(acid_rate, max_value=2.0)
        chlorine_rate = validate_flow_rate(chlorine_rate, max_value=1.0)
        inlet_rate = validate_flow_rate(inlet_rate, max_value=20.0)

        return acid_rate, chlorine_rate, inlet_rate

    except Exception as e:
        logger.error(f"Modbus read failed: {type(e).__name__}")
        return 0.0, 0.0, 5.0  # Safe defaults


def apply_boundary_conditions(
    boundary: BoundaryConditions, commands: Tuple[float, float, float]
) -> None:
    """
    Apply actuator commands to boundary conditions with validation.

    Zero-trust principle: Validate all inputs at the boundary.
    """
    acid_rate, chlorine_rate, inlet_rate = commands

    # Apply with safety limits (defense in depth)
    boundary.acid_flow_rate = validate_flow_rate(acid_rate, max_value=2.0)
    boundary.chlorine_flow_rate = validate_flow_rate(chlorine_rate, max_value=1.0)

    # Only update inlet flow if command is significant
    if inlet_rate > 0.1:
        boundary.inlet_flow_rate = validate_flow_rate(inlet_rate, max_value=20.0)


def main():
    parser = argparse.ArgumentParser(description="Water Treatment Reactor Simulation")
    parser.add_argument("--port", type=int, default=5020, help="Modbus TCP port")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Modbus bind address"
    )
    parser.add_argument(
        "--dt", type=float, default=1.0, help="Simulation timestep [seconds]"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float("inf"),
        help="Total simulation duration [seconds]",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose sensor warnings"
    )
    parser.add_argument(
        "--no-modbus",
        action="store_true",
        help="Run without Modbus server (testing mode)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("WATER TREATMENT REACTOR SIMULATION (HARDENED)")
    logger.info("=" * 70)

    # ========================================================================
    # PHASE 1: Initialize Physics Engine
    # ========================================================================
    logger.info("\n[PHASE 1] Initializing physics engine...")

    try:
        config = ReactorConfiguration(
            volume=1000.0,
            n_zones=5,
            flow_rate=5.0,
            initial_pH=7.2,
            initial_chlorine=2.0,
            temperature=20.0,
        )

        reactor = IntegratedCSTR(config)
        logger.info("✓ Physics engine initialized")

    except Exception as e:
        logger.error(f"Physics engine initialization failed: {type(e).__name__}")
        sys.exit(1)

    # ========================================================================
    # PHASE 2: Initialize Boundary Conditions
    # ========================================================================
    boundary = BoundaryConditions(
        inlet_flow_rate=5.0,
        inlet_pH=7.5,
        inlet_chlorine=0.0,
        inlet_temperature=20.0,
        acid_flow_rate=0.0,
        acid_concentration=0.1,
        chlorine_flow_rate=0.0,
    )

    # ========================================================================
    # PHASE 3: Initialize Sensors
    # ========================================================================
    sim_start_time = time.monotonic()

    try:
        sensors = initialize_sensors(config, sim_start_time, args.verbose)
    except Exception as e:
        logger.error(f"Sensor initialization failed: {type(e).__name__}")
        sys.exit(1)

    # ========================================================================
    # PHASE 4: Initialize Modbus Interface
    # ========================================================================
    slave = None

    if not args.no_modbus:
        logger.info("\n[PHASE 4] Initializing Modbus server...")

        reg_map = ModbusRegisterMap()
        modbus_config = ModbusServerConfig(
            host=args.host,
            port=args.port,
            unit_id=1,
            startup_timeout_sec=5.0,
            shutdown_timeout_sec=3.0,
        )

        try:
            slave = ModbusSlave(reg_map, modbus_config)
            slave.start(blocking=False)
            logger.info(f"✓ Modbus server started on {args.host}:{args.port}")

        except RuntimeError as e:
            logger.error(f"Modbus server startup failed: {e}")
            logger.warning("Continuing in no-Modbus mode")
            slave = None

        except Exception as e:
            logger.error(f"Modbus initialization error: {type(e).__name__}")
            logger.warning("Continuing in no-Modbus mode")
            slave = None
    else:
        logger.info("\n[PHASE 4] Skipping Modbus (--no-modbus)")

    # ========================================================================
    # PHASE 5: Main Simulation Loop
    # ========================================================================
    logger.info("\n[PHASE 5] Starting simulation loop...")
    logger.info("Press Ctrl+C to stop gracefully")

    sim_time = 0.0
    step_count = 0
    log_interval = 60
    warmup_steps = int(10.0 / args.dt)

    modbus_error_count = 0
    max_modbus_errors = 10

    try:
        while running and sim_time < args.duration:
            step_start = time.monotonic()

            # --- Step 1: Advance physics ---
            try:
                state = reactor.step(args.dt, boundary=boundary)
            except Exception as e:
                logger.error(f"Physics step failed: {type(e).__name__}")
                break

            # --- Step 2: Read sensors ---
            current_sim_time = sim_start_time + sim_time
            readings = read_all_sensors(sensors, state, current_sim_time, args.verbose)

            # --- Step 3: Update Modbus inputs ---
            if slave:
                if not update_modbus_inputs(slave, readings, sim_time):
                    modbus_error_count += 1
                    if modbus_error_count >= max_modbus_errors:
                        logger.error("Too many Modbus errors, disabling interface")
                        slave = None

            # --- Step 4: Read Modbus commands ---
            if slave:
                commands = read_modbus_commands(slave)
                apply_boundary_conditions(boundary, commands)

            # --- Periodic logging ---
            if step_count % log_interval == 0:
                sensors_ready = all(
                    r.status not in [SensorStatus.WARMING_UP, SensorStatus.CALIBRATING]
                    for r in readings.values()
                )

                if sensors_ready or step_count >= warmup_steps:
                    # Safe access to readings
                    pH_in = readings.get("pH_inlet")
                    pH_out = readings.get("pH_outlet")
                    cl_out = readings.get("chlorine_outlet")
                    flow = readings.get("flow_main")

                    logger.info(
                        f"t={sim_time:.0f}s | "
                        f"pH_in={pH_in.value if pH_in else 0:.2f} | "
                        f"pH_out={pH_out.value if pH_out else 0:.2f} | "
                        f"Cl_out={cl_out.value if cl_out else 0:.2f} | "
                        f"Flow={flow.value if flow else 0:.1f} | "
                        f"AcidCmd={boundary.acid_flow_rate:.2f}"
                    )
                else:
                    logger.info(f"t={sim_time:.0f}s | Sensors warming up...")

            step_count += 1
            sim_time += args.dt

            # --- Real-time pacing ---
            elapsed = time.monotonic() - step_start
            sleep_time = max(0.0, args.dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received")

    except Exception as e:
        logger.error(f"Simulation error: {type(e).__name__}")

    finally:
        # ====================================================================
        # CLEANUP: Ensure resources are properly released
        # ====================================================================
        logger.info("\nShutting down...")

        if slave:
            logger.info("Stopping Modbus server...")
            with suppress(Exception):
                slave.stop()

        logger.info("Simulation stopped cleanly")


if __name__ == "__main__":
    main()
