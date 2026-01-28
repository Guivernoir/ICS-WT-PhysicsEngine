"""
Chlorine Sensor Module (Enhanced Realism)
==========================================

Chlorine measurement sensors for water treatment.

Two measurement principles:
1. Amperometric (Membrane Electrode)
   - Electrochemical membrane sensor
   - Measures current proportional to Cl₂
   - Subject to membrane fouling and polarization
   - Cross-sensitive to ozone, H₂O₂, ClO₂

2. DPD Colorimetric
   - Chemical reaction with DPD reagent
   - Optical measurement of colored complex
   - Reagent degrades over time (thermal, photo)
   - Temperature affects reaction rate

Key Enhancements for Realism:
- Cross-sensitivities (ozone, H₂O₂, ClO₂ interference)
- Reagent degradation kinetics (thermal + photochemical)
- pH effects on chlorine speciation (HOCl vs OCl⁻)
- Membrane diffusion dynamics
- Temperature effects on reaction rates

Specifications (based on industrial sensors):
- Range: 0-10 mg/L (typical for water treatment)
- Precision: ±0.01 mg/L (amperometric), ±0.02 mg/L (DPD)
- Accuracy: ±0.05 mg/L
- Response time: 30s (amperometric), 60-120s (DPD)
- Drift: ±0.02 mg/L/day (amperometric), ±0.01 mg/L/day (DPD)
- Calibration validity: 24 hours (both)

Implementation Features:
- Cross-sensitivity detection for interfering species
- Reagent degradation modeling
- Membrane dynamics

References:
- Hach DR900 Colorimeter Manual
- Wallace & Tiernan Amperometric Sensor Technical Manual
- AWWA M12 "Instrumentation and Control"

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


class ChlorineSensorType(Enum):
    """Chlorine sensor measurement principle."""

    AMPEROMETRIC = "amperometric"  # Membrane electrode
    DPD_COLORIMETRIC = "dpd_colorimetric"  # Chemical reagent


class ChlorineMeasurementType(Enum):
    """Type of chlorine being measured."""

    FREE_CHLORINE = "free"  # HOCl + OCl⁻
    TOTAL_CHLORINE = "total"  # Free + combined (chloramines)


class ChlorineSensor(BaseSensor):
    """
    Chlorine sensor with selectable measurement principle.

    Models:
    - Amperometric: Membrane diffusion, electrode polarization, fouling
    - DPD Colorimetric: Reagent degradation, reaction kinetics, optical noise

    Both methods affected by:
    - pH (chlorine speciation: HOCl vs OCl⁻)
    - Temperature (reaction rates, diffusion)
    - Interfering species (ozone, H₂O₂, ClO₂)

    Typical Use:
    >>> sensor = ChlorineSensor(
    ...     name="Cl_inlet",
    ...     sensor_type=ChlorineSensorType.AMPEROMETRIC
    ... )
    >>> reading = sensor.read(reactor_state)
    >>> print(f"Cl₂: {reading.value:.2f} mg/L")
    """

    def __init__(
        self,
        name: str,
        zone_index: int = 0,
        sensor_type: ChlorineSensorType = ChlorineSensorType.AMPEROMETRIC,
        measurement_type: ChlorineMeasurementType = ChlorineMeasurementType.FREE_CHLORINE,
        precision: Optional[float] = None,
        response_time: Optional[float] = None,
        drift_rate: float = 0.02 / 24.0,  # 0.02 mg/L per day
        max_history_length: int = 1000,
        sample_line: Optional[SampleLine] = None,
        installation: Optional[InstallationQuality] = None,
        calibration_validity_hours: float = 24.0,
    ):
        """
        Initialize chlorine sensor.

        Args:
            name: Sensor identifier
            zone_index: Which reactor zone to read from
            sensor_type: Measurement principle
            measurement_type: Free or total chlorine
            precision: Override default precision [mg/L]
            response_time: Override default response time [s]
            drift_rate: Calibration drift rate [mg/L/hour]
            max_history_length: Maximum readings to store
            sample_line: Sample line configuration
            installation: Installation quality factors
            calibration_validity_hours: Calibration validity period
        """
        # Set defaults based on sensor type
        if sensor_type == ChlorineSensorType.AMPEROMETRIC:
            default_precision = 0.01  # mg/L
            default_response_time = 30.0  # seconds
            warmup_time = 300.0  # 5 minutes for membrane polarization
        else:  # DPD Colorimetric
            default_precision = 0.02  # mg/L
            default_response_time = 90.0  # 60-120 seconds
            warmup_time = 60.0  # 1 minute for optics stabilization

        super().__init__(
            name=name,
            measurement_range=(0.0, 10.0),  # mg/L
            precision=precision or default_precision,
            response_time=response_time or default_response_time,
            drift_rate=drift_rate,
            max_history_length=max_history_length,
            sample_line=sample_line,
            installation=installation,
            warmup_time_s=warmup_time,
            hysteresis_magnitude=0.01,  # ±0.01 mg/L
            calibration_validity_hours=calibration_validity_hours,
            max_rate_of_change=1.0,  # Chlorine can't change > 1 mg/L/s physically
        )

        self.zone_index = zone_index
        self.sensor_type = sensor_type
        self.measurement_type = measurement_type

        # Sensor-specific parameters
        if sensor_type == ChlorineSensorType.AMPEROMETRIC:
            # Amperometric parameters
            self.membrane_thickness = 25e-6  # [m] typical PTFE membrane
            self.membrane_fouling = 0.0  # 0-1 scale
            self.electrode_polarization = 0.0  # Voltage drift
            self.membrane_age_days = 0.0
            self.diffusion_coefficient = 1e-9  # [m²/s] for Cl₂ through PTFE

            # Cross-sensitivity factors (relative to chlorine)
            self.ozone_sensitivity = 1.2  # Ozone reads 20% higher
            self.h2o2_sensitivity = 0.3  # H₂O₂ reads 30%
            self.clo2_sensitivity = 0.5  # ClO₂ reads 50%
        else:
            # DPD Colorimetric parameters
            self.reagent_potency = 1.0  # 0-1 scale, 1=fresh
            self.reagent_age_days = 0.0
            self.optical_path_length = 0.01  # [m] 10mm cuvette
            self.light_exposure_hours = 0.0  # Cumulative light exposure
            self.storage_temperature = 20.0  # [°C] reagent storage temp

            # DPD reaction parameters
            self.reaction_time_s = 90.0  # Time for color development

        # pH effects on chlorine speciation
        # HOCl (hypochlorous acid) is the active form
        # OCl⁻ (hypochlorite ion) is less active
        # pKa ~ 7.5, so at pH 7.5 it's 50/50
        self.chlorine_pKa = 7.5

        # Initialize
        self.current_value = 0.0

    def _get_true_value(self, reactor_state) -> float:
        """
        Read true chlorine concentration from reactor state.

        Accounts for pH effects on speciation.

        Args:
            reactor_state: ReactorState object

        Returns:
            True chlorine concentration [mg/L]
        """
        # Validate zone index
        n_zones = len(reactor_state.chlorine)
        if self.zone_index >= n_zones or self.zone_index < -n_zones:
            raise IndexError(
                f"zone_index {self.zone_index} out of bounds for {n_zones} zones"
            )

        # Read chlorine from specified zone
        chlorine = reactor_state.chlorine[self.zone_index]

        # Get pH for speciation effects
        if hasattr(reactor_state, "pH"):
            pH = reactor_state.pH[self.zone_index]

            # Calculate fraction as HOCl (active form)
            # Henderson-Hasselbalch: [HOCl]/[OCl⁻] = 10^(pKa - pH)
            ratio = 10 ** (self.chlorine_pKa - pH)
            fraction_HOCl = ratio / (1 + ratio)

            # Some sensors measure only HOCl, others measure total
            # For simplicity, we'll adjust sensitivity based on fraction
            # Real sensors have different response factors for each form
            chlorine_effective = chlorine * (0.5 + 0.5 * fraction_HOCl)
        else:
            chlorine_effective = chlorine

        return chlorine_effective

    def _get_interferences(self, reactor_state) -> float:
        """
        Calculate interference from other oxidizing species.

        Real chlorine sensors respond to other oxidizers:
        - Ozone (O₃)
        - Hydrogen peroxide (H₂O₂)
        - Chlorine dioxide (ClO₂)

        This is a MAJOR source of measurement error in real plants.

        Args:
            reactor_state: ReactorState object

        Returns:
            Apparent chlorine from interferences [mg/L]
        """
        if self.sensor_type != ChlorineSensorType.AMPEROMETRIC:
            return 0.0  # DPD is more selective

        interference = 0.0

        # Check for ozone
        if hasattr(reactor_state, "ozone"):
            ozone = (
                reactor_state.ozone[self.zone_index]
                if hasattr(reactor_state.ozone, "__getitem__")
                else 0.0
            )
            interference += ozone * self.ozone_sensitivity

        # Check for hydrogen peroxide
        if hasattr(reactor_state, "hydrogen_peroxide"):
            h2o2 = (
                reactor_state.hydrogen_peroxide[self.zone_index]
                if hasattr(reactor_state.hydrogen_peroxide, "__getitem__")
                else 0.0
            )
            interference += h2o2 * self.h2o2_sensitivity

        # Check for chlorine dioxide
        if hasattr(reactor_state, "chlorine_dioxide"):
            clo2 = (
                reactor_state.chlorine_dioxide[self.zone_index]
                if hasattr(reactor_state.chlorine_dioxide, "__getitem__")
                else 0.0
            )
            interference += clo2 * self.clo2_sensitivity

        return interference

    def _update_reagent_degradation(self, dt: float) -> None:
        """
        Update DPD reagent degradation.

        Reagent degrades via:
        - Thermal degradation (Arrhenius)
        - Photodegradation (light exposure)
        - Oxidation (air exposure)

        Args:
            dt: Time step [s]
        """
        if self.sensor_type != ChlorineSensorType.DPD_COLORIMETRIC:
            return

        # Thermal degradation (Arrhenius equation)
        # Activation energy ~ 50 kJ/mol typical for organic compounds
        Ea = 50000  # J/mol
        R = 8.314  # J/mol/K
        T = self.storage_temperature + 273.15  # K
        T_ref = 293.15  # 20°C reference

        thermal_factor = np.exp((Ea / R) * (1 / T_ref - 1 / T))

        # Photodegradation (proportional to light exposure)
        # Assume continuous low-level light exposure
        self.light_exposure_hours += dt / 3600.0
        photo_factor = 1.0 + 0.1 * (
            self.light_exposure_hours / 100.0
        )  # 10% increase per 100 hours

        # Total degradation rate
        degradation_rate = thermal_factor * photo_factor * 0.01  # 1% per day at 20°C

        self.reagent_potency -= degradation_rate * (dt / 86400.0)
        self.reagent_potency = max(0.0, self.reagent_potency)

        self.reagent_age_days += dt / 86400.0

    def _update_membrane_fouling(self, dt: float) -> None:
        """
        Update amperometric membrane fouling.

        Membrane fouls from:
        - Biofilm growth
        - Mineral deposition
        - Organic coating

        Args:
            dt: Time step [s]
        """
        if self.sensor_type != ChlorineSensorType.AMPEROMETRIC:
            return

        # Exponential fouling (worse in stagnant zones)
        if self.installation.flow_velocity < 0.1:
            fouling_rate = 0.05  # per day
        else:
            fouling_rate = 0.01  # per day

        self.membrane_fouling += fouling_rate * (dt / 86400.0)
        self.membrane_fouling = min(1.0, self.membrane_fouling)

        self.membrane_age_days += dt / 86400.0

    def read(
        self, reactor_state, current_time: Optional[float] = None
    ) -> SensorReading:
        """
        Take chlorine reading with sensor-specific characteristics.

        Args:
            reactor_state: Current reactor state
            current_time: Timestamp [s]

        Returns:
            SensorReading with chlorine concentration
        """
        # Get base reading (includes all common sensor effects)
        reading = super().read(reactor_state, current_time)

        # If sensor is in fault state, return as-is
        if not np.isfinite(reading.value):
            return reading

        # Get interferences (CRITICAL for realism)
        interference = self._get_interferences(reactor_state)

        # Update degradation models
        if len(self.reading_history) >= 2:
            dt = reading.timestamp - self.reading_history[-2].timestamp
            if self.sensor_type == ChlorineSensorType.AMPEROMETRIC:
                self._update_membrane_fouling(dt)
            else:
                self._update_reagent_degradation(dt)

        # Apply sensor-specific effects
        if self.sensor_type == ChlorineSensorType.AMPEROMETRIC:
            final_value = self._apply_amperometric_effects(reading.value, interference)
        else:
            final_value = self._apply_dpd_effects(reading.value)

        # Enforce bounds
        final_value = np.clip(final_value, self.min_value, self.max_value)

        # Create final reading
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

        # Update history
        if self.reading_history:
            self.reading_history[-1] = final_reading

        self.current_value = final_value

        return final_reading

    def _apply_amperometric_effects(self, value: float, interference: float) -> float:
        """
        Apply amperometric sensor-specific effects.

        Effects:
        - Membrane diffusion lag
        - Membrane fouling (reduced signal)
        - Electrode polarization drift
        - Interfering species (ozone, H₂O₂, ClO₂)
        - Temperature effects on diffusion

        Args:
            value: Base chlorine reading
            interference: Interference from other species

        Returns:
            Modified reading
        """
        rng = self._get_rng()

        # 1. Add interferences (CRITICAL)
        value_with_interference = value + interference

        # 2. Membrane fouling effect
        # Fouling reduces diffusion rate → lower reading
        fouling_factor = 1.0 - 0.8 * self.membrane_fouling  # Up to 80% reduction

        # 3. Electrode polarization noise
        # Electrode potential drifts over time
        polarization_noise = rng.normal(
            0.0, 0.005 * (1.0 + self.membrane_age_days / 365.0)
        )

        # 4. Membrane diffusion noise
        # Fick's law variations
        diffusion_noise = rng.normal(0.0, 0.003)

        # Apply effects
        final_value = (
            value_with_interference * fouling_factor
            + polarization_noise
            + diffusion_noise
        )

        return final_value

    def _apply_dpd_effects(self, value: float) -> float:
        """
        Apply DPD colorimetric sensor-specific effects.

        Effects:
        - Reagent potency (degradation)
        - Reaction kinetics (temperature-dependent)
        - Optical noise (light source, detector)
        - pH effects on DPD reaction

        Args:
            value: Base chlorine reading

        Returns:
            Modified reading
        """
        rng = self._get_rng()

        # 1. Reagent potency effect
        # Degraded reagent gives LOW readings
        value_with_reagent = value * self.reagent_potency

        # 2. Optical noise (photodetector, light source)
        optical_noise = rng.normal(0.0, 0.005)

        # 3. Incomplete reaction (if reading too fast)
        # Some systems don't wait full reaction time
        reaction_completeness = 0.95  # 95% complete typically
        value_measured = value_with_reagent * reaction_completeness

        # Apply effects
        final_value = value_measured + optical_noise

        return final_value

    def replace_membrane(self, current_time: Optional[float] = None):
        """
        Replace amperometric sensor membrane.

        Should be done every 6-12 months.

        Args:
            current_time: Timestamp
        """
        if self.sensor_type != ChlorineSensorType.AMPEROMETRIC:
            raise ValueError("Only amperometric sensors have membranes")

        import time as time_module

        if current_time is None:
            current_time = time_module.monotonic()

        self.membrane_fouling = 0.0
        self.membrane_age_days = 0.0
        self.electrode_polarization = 0.0

        # Membrane replacement requires recalibration and warm-up
        self.power_on_time = current_time
        self.calibrate(0.0, current_time, operator_id="membrane_replacement")

    def replace_reagent(
        self, current_time: Optional[float] = None, storage_temp: float = 20.0
    ):
        """
        Replace DPD reagent.

        Should be done monthly or when potency < 80%.

        Args:
            current_time: Timestamp
            storage_temp: Storage temperature of new reagent [°C]
        """
        if self.sensor_type != ChlorineSensorType.DPD_COLORIMETRIC:
            raise ValueError("Only DPD sensors have reagent")

        import time as time_module

        if current_time is None:
            current_time = time_module.monotonic()

        self.reagent_potency = 1.0
        self.reagent_age_days = 0.0
        self.light_exposure_hours = 0.0
        self.storage_temperature = storage_temp

        # Record maintenance
        self.calibrate(0.0, current_time, operator_id="reagent_replacement")


def validate_chlorine_sensor():
    """Validate chlorine sensor implementation."""

    class MockReactorState:
        def __init__(self):
            self.chlorine = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            self.pH = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
            self.temperature = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
            # Add interfering species for testing
            self.ozone = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    # Test amperometric sensor
    sensor_amper = ChlorineSensor(
        name="Cl_amper", zone_index=0, sensor_type=ChlorineSensorType.AMPEROMETRIC
    )

    state = MockReactorState()
    reading = sensor_amper.read(state)

    if not (0.0 <= reading.value <= 10.0):
        raise AssertionError(f"Reading should be in range, got {reading.value}")

    # Should read HIGH due to ozone interference
    if reading.value <= 1.0:
        raise AssertionError(f"Should detect ozone interference, got {reading.value}")

    # Test DPD sensor
    sensor_dpd = ChlorineSensor(
        name="Cl_dpd", zone_index=0, sensor_type=ChlorineSensorType.DPD_COLORIMETRIC
    )

    reading = sensor_dpd.read(state)
    if not (0.0 <= reading.value <= 10.0):
        raise AssertionError("Reading should be in range")

    # Test membrane replacement
    sensor_amper.replace_membrane()
    if sensor_amper.membrane_fouling != 0.0:
        raise AssertionError("Membrane should be clean after replacement")

    # Test reagent replacement
    sensor_dpd.replace_reagent()
    if sensor_dpd.reagent_potency != 1.0:
        raise AssertionError("Reagent should be fresh after replacement")

    print("✓ Chlorine sensor validation passed")


if __name__ == "__main__":
    validate_chlorine_sensor()
