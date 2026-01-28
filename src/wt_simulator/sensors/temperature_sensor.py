"""
Temperature Sensor Module (Enhanced Realism)
=============================================

Temperature measurement with RTD and thermocouple physics.

Key Enhancements for Realism:
- Proper thermal time constants
- Self-heating effects (RTD)
- Cold junction compensation errors (thermocouple)
- Stem conduction errors

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


class TemperatureSensorType(Enum):
    """Temperature sensor type."""

    RTD_PT100 = "rtd_pt100"
    RTD_PT1000 = "rtd_pt1000"
    THERMOCOUPLE_K = "thermocouple_k"
    THERMOCOUPLE_J = "thermocouple_j"


class TemperatureSensor(BaseSensor):
    """
    Temperature sensor with realistic physics.

    Typical Use:
    >>> sensor = TemperatureSensor(
    ...     name="temp_inlet",
    ...     zone_index=0,
    ...     sensor_type=TemperatureSensorType.RTD_PT100
    ... )
    >>> reading = sensor.read(reactor_state)
    >>> print(f"T: {reading.value:.1f}°C")
    """

    def __init__(
        self,
        name: str,
        zone_index: int = 0,
        sensor_type: TemperatureSensorType = TemperatureSensorType.RTD_PT100,
        precision: Optional[float] = None,
        response_time: float = 15.0,
        drift_rate: float = 0.0,
        max_history_length: int = 1000,
        sample_line: Optional[SampleLine] = None,
        installation: Optional[InstallationQuality] = None,
    ):
        """Initialize temperature sensor."""
        if "rtd" in sensor_type.value:
            default_precision = 0.1  # °C
        else:
            default_precision = 0.5  # °C

        super().__init__(
            name=name,
            measurement_range=(-10.0, 110.0),
            precision=precision or default_precision,
            response_time=response_time,
            drift_rate=drift_rate,
            max_history_length=max_history_length,
            sample_line=sample_line,
            installation=installation,
            warmup_time_s=30.0,
            hysteresis_magnitude=0.05,  # °C
            calibration_validity_hours=8760.0,  # 1 year
            max_rate_of_change=10.0,  # °C/s maximum physical rate
        )

        self.zone_index = zone_index
        self.sensor_type = sensor_type

        if "rtd" in sensor_type.value:
            self.nominal_resistance = (
                100.0 if sensor_type == TemperatureSensorType.RTD_PT100 else 1000.0
            )
            self.alpha = 0.00385  # °C⁻¹
            self.lead_resistance = 0.5  # Ω
            self.excitation_current_mA = 1.0
            self.self_heating_C_per_mW = 0.001
        else:
            self.seebeck_coefficient = 40.0  # µV/°C
            self.cold_junction_temp = 25.0
            self.cold_junction_drift = 0.0

        self.current_value = 20.0

    def _get_true_value(self, reactor_state) -> float:
        """Read temperature from reactor state."""
        n_zones = len(reactor_state.temperature)
        if self.zone_index >= n_zones or self.zone_index < -n_zones:
            raise IndexError(f"zone_index {self.zone_index} out of bounds")
        return reactor_state.temperature[self.zone_index]

    def read(
        self, reactor_state, current_time: Optional[float] = None
    ) -> SensorReading:
        """Take temperature reading."""
        reading = super().read(reactor_state, current_time)

        if not np.isfinite(reading.value):
            return reading

        # Apply sensor-specific effects
        if "rtd" in self.sensor_type.value:
            final_value = self._apply_rtd_effects(reading.value)
        else:
            final_value = self._apply_thermocouple_effects(reading.value)

        # Stem conduction error
        ambient = self.installation.ambient_temperature
        stem_error = 0.01 * (reading.value - ambient)
        final_value += stem_error

        final_value = np.clip(final_value, self.min_value, self.max_value)

        final_reading = SensorReading(
            timestamp=reading.timestamp,
            value=final_value,
            raw_value=reading.raw_value,
            noise=reading.noise,
            drift=reading.drift + stem_error,
            status=reading.status,
            uncertainty=self.precision * 2.0,
            fault=reading.fault,
        )

        if self.reading_history:
            self.reading_history[-1] = final_reading

        self.current_value = final_value
        return final_reading

    def _apply_rtd_effects(self, value: float) -> float:
        """Apply RTD-specific effects."""
        rng = self._get_rng()

        # Calculate resistance
        R_true = self.nominal_resistance * (1.0 + self.alpha * value)

        # Lead wire resistance (2-wire RTD)
        R_measured = R_true + 2.0 * self.lead_resistance

        # Self-heating (corrected units)
        I_A = self.excitation_current_mA / 1000.0  # Convert mA to A
        power_W = (I_A**2) * R_measured
        power_mW = power_W * 1000.0
        self_heating_error = self.self_heating_C_per_mW * power_mW

        # Convert back to temperature
        T_measured = (R_measured / self.nominal_resistance - 1.0) / self.alpha

        # ADC noise
        adc_noise = rng.normal(0.0, 0.001)

        return T_measured + self_heating_error + adc_noise

    def _apply_thermocouple_effects(self, value: float) -> float:
        """Apply thermocouple-specific effects."""
        rng = self._get_rng()

        # Seebeck voltage
        V_seebeck = self.seebeck_coefficient * (value - self.cold_junction_temp)

        # Cold junction error
        self.cold_junction_drift += rng.normal(0.0, 0.01)

        # Thermal EMF noise
        emf_noise = rng.normal(0.0, 0.5)

        # Convert back to temperature
        V_total = V_seebeck + emf_noise
        T_measured = (
            (V_total / self.seebeck_coefficient)
            + self.cold_junction_temp
            + self.cold_junction_drift
        )

        return T_measured


def validate_temperature_sensor():
    """Validate temperature sensor."""

    class MockState:
        def __init__(self):
            self.temperature = np.array([20.0, 21.0, 22.0])

    sensor = TemperatureSensor(
        name="temp_test", zone_index=0, sensor_type=TemperatureSensorType.RTD_PT100
    )

    state = MockState()
    reading = sensor.read(state)

    if not (15.0 < reading.value < 25.0):
        raise AssertionError(f"Reading out of expected range: {reading.value}")

    print("✓ Temperature sensor validation passed")


if __name__ == "__main__":
    validate_temperature_sensor()
