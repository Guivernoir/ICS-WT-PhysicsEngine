"""
Actuators module for water treatment simulator.

Provides realistic simulation of industrial actuators:
- Control valves (pneumatic/electric)
- Dosing pumps (diaphragm/peristaltic)
- On/off valves
- VFD pumps

Features:
- Thread-safe operation
- Realistic dynamics (lag, stiction, hysteresis)
- Wear and degradation tracking
- Fault injection for testing
- Bounded memory (no unbounded growth)

Usage:
    from actuators import create_realistic_actuator_suite

    actuators = create_realistic_actuator_suite()

    # Set position
    actuators["acid_valve"].set_position(50.0)

    # Step dynamics
    actual_flow = actuators["acid_valve"].step(dt=1.0)

    # Check status
    if actuators["acid_valve"].has_fault():
        print(actuators["acid_valve"].fault_code())
"""

# Version
__version__ = "1.0.0"
__author__ = "Guilherme F. G. Santos"

from .base_actuator import (
    BaseActuator,
    ActuatorStatus,
    ActuatorFault,
    ActuatorDiagnostics,
)

from .valve_characteristics import (
    ValveType,
    ValveCharacteristics,
    generate_characteristic_curve_data,
)

from .control_valve import (
    ControlValve,
    ActuatorType,
    FailPosition,
)

from .dosing_pump import (
    DosingPump,
    PumpType,
)

from typing import Dict, Any, Optional

__all__ = [
    # Base classes
    "BaseActuator",
    "ActuatorStatus",
    "ActuatorFault",
    "ActuatorDiagnostics",
    # Valve characteristics
    "ValveType",
    "ValveCharacteristics",
    "generate_characteristic_curve_data",
    # Control valve
    "ControlValve",
    "ActuatorType",
    "FailPosition",
    # Dosing pump
    "DosingPump",
    "PumpType",
    # Convenience functions
    "create_realistic_actuator_suite",
    "create_simple_actuator_suite",
]


def create_realistic_actuator_suite(
    reactor_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, BaseActuator]:
    """
    Create complete set of actuators for water treatment simulation.

    Creates actuators with realistic parameters based on typical
    water treatment plant configurations:
    - Acid dosing valve (fail-closed pneumatic)
    - Chlorine dosing pump (diaphragm metering pump)
    - Inlet flow control valve (fail-last electric)

    Args:
        reactor_config: Optional configuration dictionary with:
            - max_acid_flow: Maximum acid flow rate (L/min)
            - max_chlorine_flow: Maximum chlorine flow rate (L/min)
            - max_inlet_flow: Maximum inlet flow rate (L/min)
            - acid_pressure_drop: Pressure drop across acid valve (bar)
            - ... etc

    Returns:
        Dictionary mapping actuator name to actuator object

    Example:
        >>> actuators = create_realistic_actuator_suite()
        >>> actuators["acid_valve"].set_position(30.0)
        >>> flow = actuators["acid_valve"].step(dt=1.0)
    """
    # Default configuration
    if reactor_config is None:
        reactor_config = {}

    max_chlorine_flow = reactor_config.get("max_chlorine_flow", 1.0)  # L/min

    acid_pressure_drop = reactor_config.get("acid_pressure_drop", 2.0)  # bar
    chlorine_discharge_pressure = reactor_config.get(
        "chlorine_discharge_pressure", 5.0
    )  # bar
    inlet_pressure_drop = reactor_config.get("inlet_pressure_drop", 1.5)  # bar

    actuators = {
        "acid_valve": ControlValve(
            name="acid_dosing_valve",
            valve_type=ValveType.LINEAR,
            cv_full_open=0.5,  # Small valve for dosing
            actuator_type=ActuatorType.PNEUMATIC,
            response_time=5.0,
            fail_position=FailPosition.FAIL_CLOSED,  # Safety: fail closed
            stiction=0.02,
            hysteresis=0.015,
            seat_leakage_percent=0.5,
            pressure_drop_bar=acid_pressure_drop,
            rangeability=30.0,
            leakage_class="Class IV",
        ),
        "chlorine_pump": DosingPump(
            name="chlorine_metering_pump",
            pump_type=PumpType.DIAPHRAGM,
            max_flow_rate=max_chlorine_flow,
            max_stroke_rate=300.0,
            stroke_volume_mL=max_chlorine_flow * 1000.0 / 300.0,
            discharge_pressure_bar=chlorine_discharge_pressure,
            max_pressure_bar=10.0,
            response_time=2.0,
            check_valve_efficiency=0.98,
            pulsation_damping=0.3,
        ),
        "inlet_valve": ControlValve(
            name="inlet_flow_control",
            valve_type=ValveType.EQUAL_PERCENTAGE,
            cv_full_open=10.0,  # Larger valve for main flow
            actuator_type=ActuatorType.ELECTRIC,
            response_time=8.0,
            fail_position=FailPosition.FAIL_LAST,  # Maintain position on failure
            stiction=0.015,
            hysteresis=0.01,
            seat_leakage_percent=0.1,
            pressure_drop_bar=inlet_pressure_drop,
            rangeability=50.0,
            leakage_class="Class III",
        ),
    }

    return actuators


def create_simple_actuator_suite() -> Dict[str, BaseActuator]:
    """
    Create simplified actuator suite with minimal parameters.

    Useful for:
    - Quick testing
    - Educational purposes
    - Baseline simulations

    Returns:
        Dictionary with basic actuators
    """
    actuators = {
        "acid_valve": ControlValve(
            name="acid_valve",
            valve_type=ValveType.LINEAR,
            cv_full_open=0.5,
            actuator_type=ActuatorType.PNEUMATIC,
            response_time=5.0,
            fail_position=FailPosition.FAIL_CLOSED,
        ),
        "chlorine_pump": DosingPump(
            name="chlorine_pump",
            pump_type=PumpType.DIAPHRAGM,
            max_flow_rate=1.0,
            max_stroke_rate=300.0,
        ),
        "inlet_valve": ControlValve(
            name="inlet_valve",
            valve_type=ValveType.LINEAR,
            cv_full_open=10.0,
            actuator_type=ActuatorType.ELECTRIC,
            response_time=8.0,
            fail_position=FailPosition.FAIL_LAST,
        ),
    }

    return actuators


def initialize_actuator_suite(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, BaseActuator]:
    """
    Initialize actuator suite from configuration.

    This is the main entry point for creating actuators from a configuration
    dictionary (typically loaded from YAML or JSON).

    Args:
        config: Configuration dictionary with actuator specifications

    Returns:
        Dictionary of initialized actuators

    Example:
        >>> config = {
        ...     "actuators": {
        ...         "acid_valve": {
        ...             "type": "control_valve",
        ...             "valve_type": "linear",
        ...             "cv_full_open": 0.5,
        ...             ...
        ...         }
        ...     }
        ... }
        >>> actuators = initialize_actuator_suite(config)
    """
    if config is None or "actuators" not in config:
        # Use default realistic suite
        return create_realistic_actuator_suite()

    actuators: Dict[str, BaseActuator] = {}

    for name, actuator_config in config["actuators"].items():
        actuator_type = actuator_config.get("type", "control_valve")

        if actuator_type == "control_valve":
            actuators[name] = _create_control_valve_from_config(name, actuator_config)
        elif actuator_type == "dosing_pump":
            actuators[name] = _create_dosing_pump_from_config(name, actuator_config)
        else:
            raise ValueError(f"Unknown actuator type: {actuator_type}")

    return actuators


def _create_control_valve_from_config(
    name: str, config: Dict[str, Any]
) -> ControlValve:
    """Create control valve from configuration dictionary."""
    valve_type_str = config.get("valve_type", "linear").upper()
    valve_type = ValveType[valve_type_str]

    actuator_type_str = config.get("actuator_type", "pneumatic").upper()
    actuator_type = ActuatorType[actuator_type_str]

    fail_position_str = config.get("fail_position", "fail_last").upper()
    fail_position = FailPosition[fail_position_str]

    return ControlValve(
        name=name,
        valve_type=valve_type,
        cv_full_open=config.get("cv_full_open", 1.0),
        actuator_type=actuator_type,
        response_time=config.get("response_time", 5.0),
        fail_position=fail_position,
        stiction=config.get("stiction", 0.02),
        hysteresis=config.get("hysteresis", 0.01),
        seat_leakage_percent=config.get("seat_leakage_percent", 0.5),
        pressure_drop_bar=config.get("pressure_drop_bar", 2.0),
        rangeability=config.get("rangeability", 50.0),
    )


def _create_dosing_pump_from_config(name: str, config: Dict[str, Any]) -> DosingPump:
    """Create dosing pump from configuration dictionary."""
    pump_type_str = config.get("pump_type", "diaphragm").upper()
    pump_type = PumpType[pump_type_str]

    return DosingPump(
        name=name,
        pump_type=pump_type,
        max_flow_rate=config.get("max_flow_rate", 1.0),
        max_stroke_rate=config.get("max_stroke_rate", 300.0),
        stroke_volume_mL=config.get("stroke_volume_mL"),
        discharge_pressure_bar=config.get("discharge_pressure_bar", 5.0),
        max_pressure_bar=config.get("max_pressure_bar", 10.0),
        response_time=config.get("response_time", 2.0),
    )
