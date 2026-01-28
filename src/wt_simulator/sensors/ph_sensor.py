"""
pH Sensor Module (Enhanced Realism)
====================================

Glass electrode pH sensor simulation with realistic characteristics.

Physical Model:
- Glass membrane potential (Nernst equation)
- Temperature compensation
- Junction potential drift
- Response time (seconds to minutes)
- Noise from electrical interference and ion activity
- Glass membrane fouling (non-linear, biofilm growth)
- Electrode aging and etching

Key Enhancements for Realism:
- Non-linear fouling (biofilm exponential growth)
- Cleaning effectiveness with glass etching
- Installation effects (flow velocity, bubbles, grounding)
- Warm-up period after calibration (membrane hydration)
- Reference electrode contamination

Specifications (based on typical industrial pH sensors):
- Range: 0-14 pH
- Precision: ±0.01 pH (1σ) under ideal conditions
- Accuracy: ±0.05 pH after calibration
- Response time (t90): 5-30 seconds
- Drift rate: ±0.01 pH/day typical, ±0.05 pH/day worst case
- Warm-up time: 30-60 minutes after calibration
- Calibration validity: 24-48 hours in dirty water

Implementation Features:
- Bounded memory for calibration history
- Explicit drift modeling
- Realistic sensor dynamics and aging

References:
- Mettler Toledo pH Measurement Guide
- Yokogawa pH/ORP Analyzer Manual
- ISA RP60.6 "Installation, Operation, and Maintenance of pH Sensors"

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from typing import Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_sensor import (
    BaseSensor,
    SensorReading,
    InstallationQuality,
    SampleLine,
)


class pHSensor(BaseSensor):
    """
    Glass electrode pH sensor.

    Models realistic pH measurement including:
    - Nernst equation (temperature-dependent)
    - Glass membrane impedance
    - Junction potential drift
    - Calibration aging
    - Electrical noise
    - Membrane fouling (non-linear)
    - Warm-up requirements
    - Installation effects

    Typical Use:
    >>> sensor = pHSensor(name="pH_inlet", zone_index=0)
    >>> reading = sensor.read(reactor_state)
    >>> print(f"pH: {reading.value:.2f} ± {reading.uncertainty:.2f}")
    """

    def __init__(
        self,
        name: str,
        zone_index: int = 0,
        precision: float = 0.01,
        response_time: float = 15.0,
        drift_rate: float = 0.01 / 24.0,  # 0.01 pH per day
        temperature_coefficient: float = 0.003,  # pH/°C
        max_history_length: int = 1000,
        sample_line: Optional[SampleLine] = None,
        installation: Optional[InstallationQuality] = None,
        calibration_validity_hours: float = 24.0,
    ):
        """
        Initialize pH sensor.

        Args:
            name: Sensor identifier
            zone_index: Which reactor zone to read from (0 = inlet, -1 = outlet)
            precision: Measurement precision [pH] (1σ noise)
            response_time: Time constant for 90% response [s]
            drift_rate: Calibration drift rate [pH/hour]
            temperature_coefficient: Temperature sensitivity [pH/°C]
            max_history_length: Maximum readings to store
            sample_line: Sample line configuration
            installation: Installation quality factors
            calibration_validity_hours: Calibration validity period
        """
        super().__init__(
            name=name,
            measurement_range=(0.0, 14.0),
            precision=precision,
            response_time=response_time,
            drift_rate=drift_rate,
            max_history_length=max_history_length,
            sample_line=sample_line,
            installation=installation,
            warmup_time_s=1800.0,  # 30 minutes warm-up after calibration
            hysteresis_magnitude=0.02,  # ±0.02 pH hysteresis typical
            calibration_validity_hours=calibration_validity_hours,
            max_rate_of_change=0.5,  # pH can't change faster than 0.5 pH/s physically
        )

        self.zone_index = zone_index
        self.temperature_coefficient = temperature_coefficient

        # Validate zone index will be done at read time

        # Glass electrode specific parameters
        self.glass_impedance = 1e8  # Ohms, typical for good electrode
        self.junction_resistance = 1e3  # Ohms

        # Calibration state (typical two-point calibration)
        self.calibration_point_1 = 4.0  # pH 4.0 buffer
        self.calibration_point_2 = 7.0  # pH 7.0 buffer
        self.slope_percentage = 100.0  # % of theoretical Nernst slope

        # Fouling state (non-linear model)
        self.membrane_fouling = 0.0  # 0-1 scale, 0=clean, 1=completely fouled
        self.glass_etching = 0.0  # Permanent damage from acid cleaning
        self.days_since_cleaning = 0.0
        self.water_hardness = 100.0  # mg/L as CaCO3, typical

        # Reference electrode state
        self.reference_contamination = 0.0  # 0-1, junction contamination

        # Initialize sensor to mid-range
        self.current_value = 7.0

    def _get_true_value(self, reactor_state) -> float:
        """
        Read true pH from reactor state.

        Args:
            reactor_state: ReactorState object from core.reactor

        Returns:
            True pH value from specified zone
        """
        # Validate zone index
        n_zones = len(reactor_state.pH)
        if self.zone_index >= n_zones or self.zone_index < -n_zones:
            raise IndexError(
                f"zone_index {self.zone_index} out of bounds for {n_zones} zones"
            )

        # Read pH from specified zone
        pH = reactor_state.pH[self.zone_index]

        # Apply temperature compensation
        # pH measurement is temperature-dependent due to Nernst equation
        if hasattr(reactor_state, "temperature"):
            T = reactor_state.temperature[self.zone_index]
            T_ref = 25.0  # Reference temperature
            pH_compensated = pH + self.temperature_coefficient * (T - T_ref)
        else:
            pH_compensated = pH

        return pH_compensated

    def _update_fouling(self, dt: float, temperature: float) -> None:
        """
        Update membrane fouling - NON-LINEAR model.

        Fouling includes:
        - Biological growth (exponential once started)
        - Mineral scaling (depends on hardness and flow)
        - Organic coating

        Args:
            dt: Time step [s]
            temperature: Water temperature [°C]
        """
        # Biological fouling (exponential growth once threshold crossed)
        if self.membrane_fouling > 0.05:  # Biofilm established
            # Doubling every 20°C increase
            bio_rate = 0.1 * np.exp(0.05 * (temperature - 25))
        else:
            bio_rate = 0.001

        # Mineral scaling (depends on hardness and flow)
        if self.installation.flow_velocity < 0.1:  # m/s, stagnant zones
            scaling_rate = self.water_hardness * 0.0001
        else:
            scaling_rate = self.water_hardness * 0.00001  # Flow prevents scaling

        # Total fouling accumulation
        fouling_rate = bio_rate + scaling_rate
        self.membrane_fouling += fouling_rate * (dt / 86400.0)  # per day
        self.membrane_fouling = min(1.0, self.membrane_fouling)

        # Track time since cleaning
        self.days_since_cleaning += dt / 86400.0

    def read(
        self, reactor_state, current_time: Optional[float] = None
    ) -> SensorReading:
        """
        Take pH reading with glass electrode characteristics.

        Adds additional noise sources specific to pH:
        - Electrical noise from high-impedance glass membrane
        - Junction potential fluctuation
        - Slope degradation over time
        - Membrane fouling effects
        - Reference electrode contamination

        Args:
            reactor_state: Current reactor state
            current_time: Timestamp [s]

        Returns:
            SensorReading with pH value
        """

        # Get base reading (includes all common sensor effects)
        reading = super().read(reactor_state, current_time)

        # If sensor is in fault state, return as-is
        if not np.isfinite(reading.value):
            return reading

        # Get temperature for fouling update
        if hasattr(reactor_state, "temperature"):
            temp = reactor_state.temperature[self.zone_index]
        else:
            temp = 25.0

        # Update fouling (non-linear)
        if len(self.reading_history) >= 2:
            dt = reading.timestamp - self.reading_history[-2].timestamp
            self._update_fouling(dt, temp)

        # Add pH-specific noise sources
        rng = self._get_rng()

        # 1. Electrical noise (higher at extreme pH values)
        # Glass impedance increases at low/high pH
        pH_deviation = abs(reading.value - 7.0)
        electrical_noise_factor = 1.0 + 0.1 * pH_deviation
        electrical_noise = rng.normal(0.0, 0.002 * electrical_noise_factor)

        # 2. Junction potential fluctuation
        # Liquid junction creates potential that can drift
        # Worse with reference contamination
        junction_noise_base = 0.005
        junction_noise = rng.normal(
            0.0, junction_noise_base * (1.0 + self.reference_contamination)
        )

        # 3. Slope degradation (electrode aging)
        # Slope percentage typically 95-105% of theoretical
        time_since_calibration_days = 0.0
        if self.calibration_history:
            time_since_cal = reading.timestamp - self.calibration_history[-1].timestamp
            time_since_calibration_days = time_since_cal / 86400.0
            slope_degradation = 0.001 * time_since_calibration_days  # 0.1% per day
            self.slope_percentage = max(90.0, 100.0 - slope_degradation)

        # Apply slope correction (affects readings away from calibration points)
        if self.calibration_point_1 < reading.value < self.calibration_point_2:
            slope_error = 0.0
        else:
            # Error increases with distance from calibration range
            distance = min(
                abs(reading.value - self.calibration_point_1),
                abs(reading.value - self.calibration_point_2),
            )
            slope_error = distance * (100.0 - self.slope_percentage) / 100.0

        # 4. Membrane fouling effect
        # Fouling increases response time and adds offset
        fouling_offset = self.membrane_fouling * 0.2  # Up to 0.2 pH when fully fouled
        fouling_noise = rng.normal(0.0, self.membrane_fouling * 0.05)

        # 5. Reference electrode contamination
        # Junction gets contaminated over time
        self.reference_contamination += 0.0001 * (time_since_calibration_days / 30.0)
        self.reference_contamination = min(0.5, self.reference_contamination)
        reference_offset = self.reference_contamination * 0.1  # Up to 0.05 pH

        # Total value with all pH-specific effects
        final_value = (
            reading.value
            + electrical_noise
            + junction_noise
            + slope_error
            + fouling_offset
            + fouling_noise
            + reference_offset
        )

        # Enforce bounds
        final_value = np.clip(final_value, self.min_value, self.max_value)

        # Create new reading with updated value
        final_reading = SensorReading(
            timestamp=reading.timestamp,
            value=final_value,
            raw_value=reading.raw_value,
            noise=reading.noise + electrical_noise + junction_noise + fouling_noise,
            drift=reading.drift + slope_error + fouling_offset + reference_offset,
            status=reading.status,
            uncertainty=self.precision * 3.0,  # Higher uncertainty for pH
            fault=reading.fault,
        )

        # Update history (replace last reading with corrected one)
        if self.reading_history:
            self.reading_history[-1] = final_reading

        # Update current value
        self.current_value = final_value

        return final_reading

    def calibrate_two_point(
        self,
        buffer_pH_1: float,
        buffer_pH_2: float,
        measured_pH_1: float,
        measured_pH_2: float,
        current_time: Optional[float] = None,
        operator_id: str = "auto",
    ) -> None:
        """
        Perform two-point pH calibration.

        Standard procedure:
        1. Rinse electrode
        2. Place in buffer 1 (typically pH 7.0), measure
        3. Rinse electrode
        4. Place in buffer 2 (typically pH 4.0 or 10.0), measure
        5. Calculate slope and offset

        Args:
            buffer_pH_1: pH of first buffer (known)
            buffer_pH_2: pH of second buffer (known)
            measured_pH_1: What sensor read in buffer 1
            measured_pH_2: What sensor read in buffer 2
            current_time: Timestamp
            operator_id: Calibration operator
        """
        import time as time_module

        if current_time is None:
            current_time = time_module.monotonic()

        # Calculate slope percentage
        # Theoretical: 59.16 mV/pH at 25°C (Nernst equation)
        # Actual: measured mV difference / theoretical
        if buffer_pH_2 != buffer_pH_1:
            measured_slope = (measured_pH_2 - measured_pH_1) / (
                buffer_pH_2 - buffer_pH_1
            )
            self.slope_percentage = measured_slope * 100.0

        # Calculate offset (zero point)
        # Use midpoint for better accuracy
        mid_buffer_pH = (buffer_pH_1 + buffer_pH_2) / 2.0
        mid_measured_pH = (measured_pH_1 + measured_pH_2) / 2.0
        offset = mid_buffer_pH - mid_measured_pH

        # Apply calibration
        self.calibration_point_1 = buffer_pH_1
        self.calibration_point_2 = buffer_pH_2

        # Reset reference contamination (assumes junction is cleaned during cal)
        self.reference_contamination = 0.0

        # Use standard single-point calibration for base class
        self.calibrate(mid_buffer_pH, current_time, operator_id)

    def clean_electrode(
        self, cleaning_method: str, current_time: Optional[float] = None
    ):
        """
        Clean pH electrode.

        Methods:
        - 'water_rinse': Gentle, removes loose deposits
        - 'acid_clean': Removes mineral scaling (etches glass slightly)
        - 'pepsin_clean': Removes protein/organic fouling

        Args:
            cleaning_method: Type of cleaning
            current_time: Timestamp
        """
        import time as time_module

        if current_time is None:
            current_time = time_module.monotonic()

        if cleaning_method == "water_rinse":
            # Removes 50% of fouling
            self.membrane_fouling *= 0.5
        elif cleaning_method == "acid_clean":
            # Removes 90% of mineral deposits
            self.membrane_fouling *= 0.1
            # But etches glass slightly (permanent damage)
            self.glass_etching += 0.001
            # Glass etching reduces slope percentage permanently
            self.slope_percentage -= self.glass_etching * 10.0  # 1% per 0.1 etching
        elif cleaning_method == "pepsin_clean":
            # Removes organic fouling
            self.membrane_fouling *= 0.2
        else:
            raise ValueError(f"Unknown cleaning method: {cleaning_method}")

        self.days_since_cleaning = 0.0

        # Cleaning requires warm-up time
        self.power_on_time = current_time

    def check_slope_health(self) -> Dict[str, float]:
        """
        Check electrode slope health.

        Slope percentage indicates electrode condition:
        - 95-105%: Excellent
        - 90-95% or 105-110%: Good
        - 85-90% or 110-115%: Fair, should replace soon
        - <85% or >115%: Poor, replace immediately

        Returns:
            Dictionary with slope info and health status
        """
        if 95.0 <= self.slope_percentage <= 105.0:
            health = "excellent"
        elif 90.0 <= self.slope_percentage <= 110.0:
            health = "good"
        elif 85.0 <= self.slope_percentage <= 115.0:
            health = "fair"
        else:
            health = "poor"

        import time as time_module

        if self.calibration_history:
            days_since_cal = (
                time_module.monotonic() - self.calibration_history[-1].timestamp
            ) / 86400.0
        else:
            days_since_cal = 0.0

        return {
            "slope_percentage": self.slope_percentage,
            "health": health,
            "impedance_ohms": self.glass_impedance,
            "days_since_calibration": days_since_cal,
            "membrane_fouling": self.membrane_fouling,
            "glass_etching": self.glass_etching,
            "days_since_cleaning": self.days_since_cleaning,
        }

    def set_water_hardness(self, hardness_mg_L: float):
        """
        Set water hardness for fouling model.

        Higher hardness → faster mineral scaling.

        Args:
            hardness_mg_L: Water hardness [mg/L as CaCO3]
        """
        if hardness_mg_L < 0:
            raise ValueError(f"Hardness must be non-negative, got {hardness_mg_L}")
        self.water_hardness = hardness_mg_L


def validate_pH_sensor():
    """Validate pH sensor implementation."""
    import time as time_module

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
