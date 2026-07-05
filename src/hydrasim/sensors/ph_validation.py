"""Validation routine for the HydraSim pH sensor."""

from __future__ import annotations

import time as time_module

import numpy as np

from .base_sensor import InstallationQuality, SampleLine
from .ph_sensor import pHSensor


def validate_pH_sensor():
    """Validate pH sensor implementation."""

    # Need to create a mock reactor state for testing
    class MockReactorState:
        def __init__(self):
            self.pH = np.array([7.0, 7.1, 7.2, 7.3, 7.4])
            self.temperature = np.array([20.0, 20.0, 20.0, 20.0, 20.0])

    # Create sensor with sample line (realistic)
    sample_line = SampleLine(volume_mL=100, flow_rate_mL_min=500, ambient_temp=20.0)

    installation = InstallationQuality(
        flow_velocity=0.5, air_bubble_frequency=0.0, grounding_quality=1.0
    )

    sensor = pHSensor(
        name="pH_test", zone_index=0, sample_line=sample_line, installation=installation
    )

    # Take readings
    state = MockReactorState()
    current_time = time_module.monotonic()

    readings = []
    for i in range(10):
        reading = sensor.read(state, current_time + i)
        if np.isfinite(reading.value):
            readings.append(reading.value)

    # Validate
    if len(readings) > 0:
        mean_pH = np.mean(readings)
        std_pH = np.std(readings)

        if not (6.0 < mean_pH < 8.0):
            raise AssertionError(f"Mean pH should be near 7.0, got {mean_pH}")
        if std_pH >= 0.2:
            raise AssertionError(f"pH std should be small, got {std_pH}")

    if not sensor.reading_history:
        raise AssertionError("Should have reading history")

    # Test calibration
    sensor.calibrate_two_point(4.0, 7.0, 4.05, 7.02, current_time)
    if not (90 < sensor.slope_percentage < 110):
        raise AssertionError(
            f"Slope should be reasonable, got {sensor.slope_percentage}"
        )

    # Test slope health
    health = sensor.check_slope_health()
    if health["health"] not in ["excellent", "good", "fair", "poor"]:
        raise AssertionError(f"Unknown health status: {health['health']}")

    # Test cleaning
    sensor.clean_electrode("water_rinse")
    if sensor.membrane_fouling >= 0.5:
        raise AssertionError("Cleaning should reduce fouling")

    print("✓ pH sensor validation passed")


if __name__ == "__main__":

    validate_pH_sensor()
