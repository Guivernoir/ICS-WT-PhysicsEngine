"""
Main simulation orchestrator.

Entry point for the HydraSim process simulator runtime.
"""

import argparse
import time
import logging
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

# Physics engine
from .core import BoundaryConditions, IntegratedCSTR, ReactorConfiguration

# Sensors
from .sensors import SensorStatus

# Maintenance
from .maintenance import MaintenanceManager
from .runtime_loop import (
    apply_actuator_commands as apply_actuator_commands,
    initialize_actuators as initialize_actuators,
    initialize_modbus_defaults as initialize_modbus_defaults,
    initialize_sensors as initialize_sensors,
    read_all_sensors as read_all_sensors,
    read_modbus_commands as read_modbus_commands,
    read_modbus_dosing_concentrations as read_modbus_dosing_concentrations,
    read_modbus_enable_bits as read_modbus_enable_bits,
    step_actuators_into_boundary as step_actuators_into_boundary,
    update_modbus_inputs as update_modbus_inputs,
    validate_concentration as validate_concentration,
    validate_flow_rate as validate_flow_rate,
    validate_ph as validate_ph,
)

# Modbus (optional dependency)
MODBUS_AVAILABLE = True
MODBUS_IMPORT_ERROR: Optional[str] = None
try:
    from .modbus import ModbusSlave, ModbusRegisterMap, ModbusServerConfig
except ModuleNotFoundError as exc:
    if exc.name != "pymodbus":
        raise
    MODBUS_AVAILABLE = False
    MODBUS_IMPORT_ERROR = str(exc)
    ModbusSlave = Any  # type: ignore
    ModbusRegisterMap = Any  # type: ignore
    ModbusServerConfig = Any  # type: ignore

# Configure logging
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


def main():
    parser = argparse.ArgumentParser(description="HydraSim Process Simulation")
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
    parser.add_argument(
        "--scenario",
        help="Optional built-in scenario ID or alias to replay against this simulator",
    )
    parser.add_argument(
        "--scenario-custom-json",
        type=Path,
        help="Optional custom scenario JSON file for replay",
    )
    parser.add_argument(
        "--scenario-time-scale",
        type=float,
        default=1.0,
        help="Scenario replay timing scale; 0 runs immediately",
    )
    parser.add_argument(
        "--scenario-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after Modbus startup before scenario replay",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("HYDRASIM REACTOR SIMULATION")
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
        logger.info("Physics engine initialized")

    except Exception as e:
        logger.error(f"Physics engine initialization failed: {type(e).__name__}")
        sys.exit(1)

    # ========================================================================
    # PHASE 2: Initialize Boundary Conditions
    # ========================================================================
    boundary = BoundaryConditions(
        inlet_flow_rate=5.0,
        inlet_pH=config.inlet_pH,
        inlet_chlorine=config.inlet_chlorine,
        inlet_chloramine=config.inlet_chloramine,
        inlet_ammonia=config.inlet_ammonia,
        inlet_chlorine_demand=config.inlet_chlorine_demand,
        inlet_temperature=config.inlet_temperature,
        acid_flow_rate=0.0,
        acid_concentration=0.1,
        chlorine_flow_rate=0.0,
    )

    # ========================================================================
    # PHASE 3: Initialize Actuators
    # ========================================================================
    logger.info("\n[PHASE 3] Initializing actuator suite...")
    try:
        actuators = initialize_actuators(boundary)
        logger.info("Actuator suite initialized")
    except Exception as e:
        logger.error(f"Actuator initialization failed: {type(e).__name__}")
        sys.exit(1)

    # ========================================================================
    # PHASE 4: Initialize Sensors
    # ========================================================================
    sim_start_time = time.monotonic()

    try:
        sensors = initialize_sensors(config, sim_start_time, args.verbose)
    except Exception as e:
        logger.error(f"Sensor initialization failed: {type(e).__name__}")
        sys.exit(1)

    # ========================================================================
    # PHASE 5: Initialize Modbus Interface
    # ========================================================================
    slave = None

    if not args.no_modbus and not MODBUS_AVAILABLE:
        logger.warning(
            f"\n[PHASE 5] Modbus unavailable ({MODBUS_IMPORT_ERROR}). "
            "Run with --no-modbus or install pymodbus."
        )
    elif not args.no_modbus:
        logger.info("\n[PHASE 5] Initializing Modbus server...")

        reg_map = ModbusRegisterMap()
        modbus_config = ModbusServerConfig(
            host=args.host,
            port=args.port,
            unit_id=1,
            startup_timeout_sec=5.0,
            shutdown_timeout_sec=3.0,
        )

        try:
            maintenance_manager = MaintenanceManager(sensors, actuators)
            slave = ModbusSlave(
                reg_map, modbus_config, maintenance_manager=maintenance_manager
            )
            slave.start(blocking=False)
            initialize_modbus_defaults(slave, boundary)
            logger.info(f"Modbus server started on {args.host}:{args.port}")
            if args.scenario or args.scenario_custom_json:
                from .scenarios.live import start_live_scenario_thread

                start_live_scenario_thread(
                    args.scenario or "custom",
                    args.host,
                    args.port,
                    unit_id=1,
                    time_scale=args.scenario_time_scale,
                    startup_delay_s=args.scenario_delay,
                    custom_json=args.scenario_custom_json,
                    logger=logger,
                )
                logger.info("Scenario replay scheduled")

        except RuntimeError as e:
            logger.error(f"Modbus server startup failed: {e}")
            logger.warning("Continuing in no-Modbus mode")
            slave = None

        except Exception as e:
            logger.error(f"Modbus initialization error: {type(e).__name__}")
            logger.warning("Continuing in no-Modbus mode")
            slave = None
    else:
        logger.info("\n[PHASE 5] Skipping Modbus (--no-modbus)")
        if args.scenario or args.scenario_custom_json:
            logger.warning("Scenario replay skipped because Modbus is disabled")

    # ========================================================================
    # PHASE 6: Main Simulation Loop
    # ========================================================================
    logger.info("\n[PHASE 6] Starting simulation loop...")
    logger.info("Press Ctrl+C to stop gracefully")

    sim_time = 0.0
    step_count = 0
    log_interval = 60
    warmup_steps = int(10.0 / args.dt)

    modbus_error_count = 0
    max_modbus_errors = 10
    state = reactor.state

    try:
        while running and sim_time < args.duration:
            step_start = time.monotonic()
            current_sim_time = sim_start_time + sim_time

            # --- Step 1: Read external commands ---
            commands = (
                boundary.acid_flow_rate,
                boundary.chlorine_flow_rate,
                boundary.inlet_flow_rate,
            )
            enable_bits = (True, True, True)

            if slave:
                commands = read_modbus_commands(slave)
                enable_bits = read_modbus_enable_bits(slave)
                acid_conc, chlorine_conc = read_modbus_dosing_concentrations(
                    slave, boundary
                )
                boundary.acid_concentration = acid_conc
                boundary.chlorine_concentration = chlorine_conc

            simulation_enabled = enable_bits[2]

            # --- Step 2: Apply actuator dynamics and run physics ---
            if simulation_enabled:
                apply_actuator_commands(
                    actuators, commands, enable_bits, boundary.inlet_flow_rate
                )
                step_actuators_into_boundary(actuators, boundary, args.dt)
                try:
                    state = reactor.step(args.dt, boundary=boundary)
                except Exception as e:
                    logger.error(f"Physics step failed: {type(e).__name__}")
                    break

            # --- Step 3: Read sensors ---
            readings = read_all_sensors(sensors, state, current_sim_time, args.verbose)

            # --- Step 4: Update Modbus inputs ---
            if slave:
                if not update_modbus_inputs(slave, readings, sim_time):
                    modbus_error_count += 1
                    if modbus_error_count >= max_modbus_errors:
                        logger.error("Too many Modbus errors, disabling interface")
                        slave = None

            # --- Step 5: Dispatch maintenance commands ---
            if slave:
                with suppress(Exception):
                    slave.poll_maintenance()

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
                        f"AcidSP={commands[0]:.2f} | "
                        f"AcidAct={boundary.acid_flow_rate:.2f}"
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
