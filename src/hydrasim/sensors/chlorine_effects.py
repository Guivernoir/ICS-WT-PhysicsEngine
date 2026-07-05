"""Chlorine sensor effect and replacement helpers."""

from __future__ import annotations

from typing import Any, Deque, Optional, TYPE_CHECKING

from .base_sensor import CalibrationRecord, SensorFault, SensorStatus


class ChlorineEffectsMixin:
    sensor_type: Any
    membrane_fouling: float
    membrane_age_days: float
    electrode_polarization: float
    power_on_time: float
    calibration_offset: float
    cumulative_drift: float
    status: SensorStatus
    fault: SensorFault
    current_value: float
    calibration_history: Deque[CalibrationRecord]
    reagent_potency: float
    reagent_age_days: float
    light_exposure_hours: float
    storage_temperature: float
    _state_lock: Any

    if TYPE_CHECKING:

        def _get_rng(self) -> Any: ...

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

        # 1. Add interferences.
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

        After replacement the sensor enters warm-up and the calibration is
        marked immediately expired so the operator must perform a fresh
        calibration.  The calibration offset is explicitly cleared to zero
        so readings during the re-calibration window are not artificially
        biased by the pre-replacement process value.

        Args:
            current_time: Timestamp
        """
        if getattr(self.sensor_type, "value", self.sensor_type) != "amperometric":
            raise ValueError("Only amperometric sensors have membranes")

        import time as time_module

        if current_time is None:
            current_time = time_module.monotonic()

        with self._state_lock:
            # Reset hardware wear counters
            self.membrane_fouling = 0.0
            self.membrane_age_days = 0.0
            self.electrode_polarization = 0.0

            # Restart warm-up timer (new membrane must polarise)
            self.power_on_time = current_time

            # Zero the calibration offset so readings are not biased by the
            # pre-replacement process value.  The record below is marked with
            # validity_hours=0 so _check_calibration_valid() returns False
            # immediately, forcing a proper field re-calibration.
            self.calibration_offset = 0.0
            self.cumulative_drift = 0.0
            self.status = SensorStatus.CALIBRATION_EXPIRED
            self.fault = SensorFault.NONE

            record = CalibrationRecord(
                timestamp=current_time,
                reference_value=0.0,
                measured_value=self.current_value,
                offset=0.0,
                operator_id="membrane_replacement",
                validity_hours=0.0,  # Immediately expired — must recalibrate
                skip_warmup=False,
            )
            self.calibration_history.append(record)

    def replace_reagent(
        self, current_time: Optional[float] = None, storage_temp: float = 20.0
    ):
        """
        Replace DPD reagent.

        Should be done monthly or when potency < 80%.

        After replacement the calibration is marked immediately expired so
        the operator must perform a fresh calibration.  The calibration
        offset is cleared to zero to avoid biasing readings during the
        re-calibration window.

        Args:
            current_time: Timestamp
            storage_temp: Storage temperature of new reagent [°C]
        """
        if getattr(self.sensor_type, "value", self.sensor_type) != "dpd_colorimetric":
            raise ValueError("Only DPD sensors have reagent")

        import time as time_module

        if current_time is None:
            current_time = time_module.monotonic()

        with self._state_lock:
            # Reset reagent state
            self.reagent_potency = 1.0
            self.reagent_age_days = 0.0
            self.light_exposure_hours = 0.0
            self.storage_temperature = storage_temp

            # Zero the calibration offset; mark calibration expired so the
            # operator is forced to perform a fresh field calibration.
            self.calibration_offset = 0.0
            self.cumulative_drift = 0.0
            self.status = SensorStatus.CALIBRATION_EXPIRED
            self.fault = SensorFault.NONE

            record = CalibrationRecord(
                timestamp=current_time,
                reference_value=0.0,
                measured_value=self.current_value,
                offset=0.0,
                operator_id="reagent_replacement",
                validity_hours=0.0,  # Immediately expired — must recalibrate
                skip_warmup=True,  # DPD optics are already warm
            )
            self.calibration_history.append(record)
