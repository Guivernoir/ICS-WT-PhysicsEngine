"""
Flow Sensor Module (Enhanced Realism)
======================================

Flow measurement sensors for water treatment.

Two measurement principles:
1. Turbine Flowmeter - Mechanical rotor, bearing wear, vibration-sensitive
2. Magnetic Flowmeter - Electromagnetic, no moving parts, conductivity-dependent

Key Enhancements for Realism:
- Air bubble detection (major cause of errors)
- Vibration effects on turbine bearings
- Empty pipe detection (magnetic)
- Velocity profile distortion from installation

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from typing import Optional
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_sensor import BaseSensor, SensorReading, InstallationQuality, SampleLine


class FlowSensorType(Enum):
    """Flow sensor measurement principle."""

    TURBINE = "turbine"
    MAGNETIC = "magnetic"


class FlowSensor(BaseSensor):
    """
    Flow rate sensor with installation and operational effects.

    Typical Use:
    >>> sensor = FlowSensor(
    ...     name="flow_main",
    ...     sensor_type=FlowSensorType.MAGNETIC
    ... )
    >>> reading = sensor.read_flow(50.0)
    >>> print(f"Flow: {reading.value:.2f} L/min")
    """

    def __init__(
        self,
        name: str,
        sensor_type: FlowSensorType = FlowSensorType.MAGNETIC,
        full_scale: float = 100.0,
        precision: Optional[float] = None,
        response_time: float = 0.5,
        drift_rate: float = 0.0,
        max_history_length: int = 1000,
        sample_line: Optional[SampleLine] = None,
        installation: Optional[InstallationQuality] = None,
    ):
        """Initialize flow sensor."""
        if sensor_type == FlowSensorType.TURBINE:
            default_precision = 0.01 * full_scale
        else:
            default_precision = 0.005 * full_scale

        super().__init__(
            name=name,
            measurement_range=(0.0, full_scale),
            precision=precision or default_precision,
            response_time=response_time,
            drift_rate=drift_rate,
            max_history_length=max_history_length,
            sample_line=sample_line,
            installation=installation,
            warmup_time_s=10.0,
            hysteresis_magnitude=0.005 * full_scale,
            calibration_validity_hours=8760.0,  # 1 year
            max_rate_of_change=full_scale,  # Flow can change very quickly
        )

        self.sensor_type = sensor_type
        self.full_scale = full_scale

        if sensor_type == FlowSensorType.TURBINE:
            self.bearing_friction = 0.01
            self.bearing_wear_days = 0.0
        else:
            self.electrode_fouling = 0.0
            self.fluid_conductivity = 100.0  # µS/cm

        self.current_value = 0.0

    def _get_true_value(self, reactor_state) -> float:
        """Read flow rate from reactor state."""
        if hasattr(reactor_state, "flow_rate"):
            return reactor_state.flow_rate
        raise AttributeError("reactor_state missing flow_rate attribute")

    def read_flow(
        self, flow_rate: float, current_time: Optional[float] = None
    ) -> SensorReading:
        """
        Read flow from known value (convenience method).

        Args:
            flow_rate: True flow rate [L/min]
            current_time: Timestamp

        Returns:
            SensorReading
        """

        class MockState:
            pass

        state = MockState()
        state.flow_rate = flow_rate
        return self.read(state, current_time)

    def read(
        self, reactor_state, current_time: Optional[float] = None
    ) -> SensorReading:
        """Take flow reading with sensor-specific effects."""
        reading = super().read(reactor_state, current_time)

        if not np.isfinite(reading.value):
            return reading

        # Update aging
        if len(self.reading_history) >= 2:
            dt = reading.timestamp - self.reading_history[-2].timestamp
            if self.sensor_type == FlowSensorType.TURBINE:
                # Accelerated wear with vibration
                wear_factor = 1.0 + self.installation.pipe_vibration_g * 5.0
                self.bearing_wear_days += (dt / 86400.0) * wear_factor
            else:
                self.electrode_fouling += 0.001 * (dt / 86400.0)

        # Apply sensor-specific effects
        if self.sensor_type == FlowSensorType.TURBINE:
            final_value = self._apply_turbine_effects(reading.value)
        else:
            final_value = self._apply_magnetic_effects(reading.value)

        # Air bubbles cause dropouts (CRITICAL for realism)
        rng = self._get_rng()
        if self.installation.air_bubble_frequency > 0:
            if rng.random() < self.installation.air_bubble_frequency / 60.0:
                final_value = 0.0  # Air bubble = no flow reading

        # Zero cutoff
        min_flow = 0.01 * self.full_scale
        if final_value < min_flow:
            final_value = 0.0

        final_value = np.clip(final_value, 0.0, self.max_value)

        final_reading = SensorReading(
            timestamp=reading.timestamp,
            value=final_value,
            raw_value=reading.raw_value,
            noise=reading.noise,
            drift=reading.drift,
            status=reading.status,
            uncertainty=self.precision * 2.0,
            fault=reading.fault,
        )

        if self.reading_history:
            self.reading_history[-1] = final_reading

        self.current_value = final_value
        return final_reading

    def _apply_turbine_effects(self, value: float) -> float:
        """Apply turbine-specific effects."""
        rng = self._get_rng()

        # Bearing friction (increases with wear and vibration)
        friction_increase = 1.0 + 0.01 * (self.bearing_wear_days / 365.0)
        friction_threshold = self.bearing_friction * friction_increase
        friction_loss = friction_threshold * self.full_scale

        if value < friction_loss:
            effective_value = 0.0
        else:
            effective_value = value - friction_loss

        # Mechanical noise (vibration)
        vibration_noise = rng.normal(
            0.0, self.installation.pipe_vibration_g * 0.01 * self.full_scale
        )

        return effective_value + vibration_noise

    def _apply_magnetic_effects(self, value: float) -> float:
        """Apply magnetic flowmeter effects."""
        rng = self._get_rng()

        # Electrode fouling
        fouling_factor = max(0.9, 1.0 - 0.005 * self.electrode_fouling)

        # Conductivity effect
        if self.fluid_conductivity < 5.0:
            conductivity_factor = 0.0  # Cannot measure
        elif self.fluid_conductivity < 20.0:
            conductivity_factor = self.fluid_conductivity / 20.0
        else:
            conductivity_factor = 1.0

        # Electrical noise
        electrical_noise = rng.normal(0.0, 0.001 * self.full_scale)

        return value * fouling_factor * conductivity_factor + electrical_noise


def validate_flow_sensor():
    """Validate flow sensor."""
    sensor = FlowSensor(name="flow_test", sensor_type=FlowSensorType.MAGNETIC)

    reading = sensor.read_flow(50.0)
    if not (0.0 <= reading.value <= 100.0):
        raise AssertionError(f"Reading out of range: {reading.value}")

    # Test zero flow
    reading_zero = sensor.read_flow(0.0)
    if reading_zero.value != 0.0:
        raise AssertionError("Should read zero at zero flow")

    print("✓ Flow sensor validation passed")


if __name__ == "__main__":
    validate_flow_sensor()
