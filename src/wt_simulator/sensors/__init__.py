"""
Sensors Package
===============

Sensor simulations for water treatment reactor.

Available Sensors:
- pHSensor: Glass electrode
- ChlorineSensor: Amperometric and DPD colorimetric
- FlowSensor: Turbine and magnetic
- TemperatureSensor: RTD and thermocouple

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Guilherme F. G. Santos"

from typing import Dict

# Import base classes
from .base_sensor import (
    BaseSensor,
    SensorReading,
    SensorStatus,
    SensorFault,
    CalibrationRecord,
    InstallationQuality,
    SampleLine,
)

# Import sensor implementations
from .ph_sensor import pHSensor
from .chlorine_sensor import ChlorineSensor, ChlorineSensorType, ChlorineMeasurementType
from .flow_sensor import FlowSensor, FlowSensorType
from .temperature_sensor import TemperatureSensor, TemperatureSensorType


def create_realistic_sensor_suite(reactor_config):
    """
    Create a complete set of sensors for the reactor configuration.

    Args:
        reactor_config: Reactor configuration object with flow_rate, etc.

    Returns:
        Dict[str, BaseSensor]: Named sensors
    """

    # Common installation quality
    good_installation = InstallationQuality(
        flow_velocity=0.5,
        air_bubble_frequency=0.0,
        grounding_quality=0.9,
        pipe_vibration_g=0.1,
        ambient_temperature=30.0,
    )

    # Sample lines
    inlet_sample_line = SampleLine(
        volume_mL=250, flow_rate_mL_min=500, ambient_temp=25.0
    )
    outlet_sample_line = SampleLine(
        volume_mL=250, flow_rate_mL_min=500, ambient_temp=25.0
    )

    sensors = {
        # pH sensors
        "pH_inlet": pHSensor(
            name="pH_inlet",
            zone_index=0,
            sample_line=inlet_sample_line,
            installation=good_installation,
        ),
        "pH_outlet": pHSensor(
            name="pH_outlet",
            zone_index=-1,
            sample_line=outlet_sample_line,
            installation=good_installation,
        ),
        # Chlorine sensors
        "chlorine_inlet": ChlorineSensor(
            name="chlorine_inlet",
            zone_index=0,
            sensor_type=ChlorineSensorType.AMPEROMETRIC,
            installation=good_installation,
        ),
        "chlorine_outlet": ChlorineSensor(
            name="chlorine_outlet",
            zone_index=-1,
            sensor_type=ChlorineSensorType.DPD_COLORIMETRIC,
            installation=good_installation,
        ),
        # Flow sensor
        "flow_main": FlowSensor(
            name="flow_main",
            sensor_type=FlowSensorType.MAGNETIC,
            full_scale=reactor_config.flow_rate * 2.0,
            installation=good_installation,
        ),
        # Temperature sensors
        "temp_inlet": TemperatureSensor(
            name="temp_inlet",
            zone_index=0,
            sensor_type=TemperatureSensorType.RTD_PT100,
            sample_line=inlet_sample_line,
            installation=good_installation,
        ),
        "temp_outlet": TemperatureSensor(
            name="temp_outlet",
            zone_index=-1,
            sensor_type=TemperatureSensorType.RTD_PT100,
            sample_line=outlet_sample_line,
            installation=good_installation,
        ),
    }

    return sensors


if __name__ == "__main__":
    print("Sensors Package v1.0.0")
    print("Available sensors: pHSensor, ChlorineSensor, FlowSensor, TemperatureSensor")
