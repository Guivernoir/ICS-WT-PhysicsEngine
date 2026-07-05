"""Shared sensor operation mixins."""

from __future__ import annotations

import time
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .sensor_models import (
    CalibrationRecord,
    InstallationQuality,
    SampleLine,
    SensorFault,
    SensorReading,
    SensorStatus,
)


class SensorDiagnosticsMixin:
    voltage_tolerance: tuple[float, float]
    supply_voltage: float
    min_value: float
    max_value: float
    max_rate_of_change: Optional[float]
    installation: InstallationQuality
    precision: float
    hysteresis_magnitude: float
    current_value: float
    _last_direction: int
    calibration_history: Deque[CalibrationRecord]
    power_on_time: float
    warmup_time_s: float

    if TYPE_CHECKING:

        def _get_rng(self) -> Any: ...

    def _check_for_faults(
        self, value: float, rate_of_change: float
    ) -> Optional[SensorFault]:
        """
        Check for sensor fault conditions.

        Real sensors have internal diagnostics that detect:
        - Open circuit (disconnected wire)
        - Short circuit
        - Out of range (physically impossible reading)
        - Rate of change fault (changed too fast)
        - Power supply issues

        Args:
            value: Current reading
            rate_of_change: Rate of change [units/s]

        Returns:
            SensorFault if detected, None otherwise
        """
        # Check power supply
        if not (
            self.voltage_tolerance[0] < self.supply_voltage < self.voltage_tolerance[1]
        ):
            if self.supply_voltage < self.voltage_tolerance[0]:
                return SensorFault.POWER_LOW
            else:
                return SensorFault.POWER_HIGH

        # Check for out of range (some margin for sensor overrange)
        range_span = self.max_value - self.min_value
        if (
            value < self.min_value - 0.1 * range_span
            or value > self.max_value + 0.1 * range_span
        ):
            return SensorFault.OUT_OF_RANGE

        # Check rate of change
        if (
            self.max_rate_of_change is not None
            and abs(rate_of_change) > self.max_rate_of_change
        ):
            return SensorFault.RATE_FAULT

        # Random faults (very rare - 0.01% per reading)
        rng = self._get_rng()
        if rng.random() < 0.0001:
            faults = (SensorFault.OPEN_CIRCUIT, SensorFault.SHORT_CIRCUIT)
            return faults[int(rng.integers(0, len(faults)))]

        return None

    def _check_warmup(self, current_time: float) -> bool:
        """
        Check if sensor has completed warm-up period.

        Real sensors need time to stabilize after power-on or calibration.

        Returns:
            True if warmed up, False if still warming
        """
        elapsed = current_time - self.power_on_time
        return elapsed >= self.warmup_time_s

    def _check_calibration_valid(self, current_time: float) -> bool:
        """
        Check if calibration is still valid.

        Real calibrations expire after specified time.

        Returns:
            True if valid, False if expired
        """
        if not self.calibration_history:
            return False

        last_cal = self.calibration_history[-1]
        return not last_cal.is_expired(current_time)

    def _apply_hysteresis(self, value: float) -> float:
        """
        Apply hysteresis effect.

        Real sensors read differently depending on direction of change.

        Args:
            value: Input value

        Returns:
            Value with hysteresis applied
        """
        if self.hysteresis_magnitude == 0.0:
            return value

        # Detect direction of change
        direction = np.sign(value - self.current_value)

        # Apply hysteresis if direction reversed
        if direction != self._last_direction and direction != 0:
            # Add offset based on direction
            value += direction * self.hysteresis_magnitude / 2.0
            self._last_direction = direction

        return value

    def _apply_installation_effects(
        self, value: float, rng: np.random.Generator
    ) -> float:
        """
        Apply installation quality effects.

        Real sensors are affected by:
        - Low flow (stagnant zones cause scatter)
        - Air bubbles (intermittent readings)
        - Poor grounding (electrical noise)
        - Vibration (mechanical noise)

        Args:
            value: Input value
            rng: Random number generator

        Returns:
            Value with installation effects
        """
        # Low flow causes increased scatter
        if self.installation.flow_velocity < 0.1:  # m/s
            stagnant_noise = rng.normal(0.0, self.precision * 2.0)
            value += stagnant_noise

        # Air bubbles cause intermittent faults (return NaN)
        if self.installation.air_bubble_frequency > 0:
            bubble_prob = self.installation.air_bubble_frequency / 60.0  # per second
            if rng.random() < bubble_prob:
                return np.nan  # Air bubble on sensor

        # Poor grounding increases electrical noise
        if self.installation.grounding_quality < 0.8:
            noise_multiplier = 2.0 - self.installation.grounding_quality
            electrical_noise = rng.normal(0.0, self.precision * noise_multiplier)
            value += electrical_noise

        # Vibration adds mechanical noise
        if self.installation.pipe_vibration_g > 0.2:  # g
            vibration_noise = rng.normal(
                0.0, self.installation.pipe_vibration_g * self.precision
            )
            value += vibration_noise

        return value


class SensorServiceMixin:
    name: str
    min_value: float
    max_value: float
    calibration_validity_hours: float
    current_value: float
    calibration_offset: float
    last_calibration_time: float
    cumulative_drift: float
    status: SensorStatus
    fault: SensorFault
    power_on_time: float
    reading_history: Deque[SensorReading]
    calibration_history: Deque[CalibrationRecord]
    sample_line: Optional[SampleLine]
    _state_lock: Any
    _last_direction: int

    def calibrate(
        self,
        reference_value: float,
        current_time: Optional[float] = None,
        operator_id: str = "auto",
        validity_hours: Optional[float] = None,
        skip_warmup: bool = False,
    ) -> CalibrationRecord:
        """
        Calibrate sensor against known reference.

        Calculates offset between reference and current reading,
        then resets calibration to this offset.

        Parameters
        ----------
        reference_value : float
            Known true value to calibrate against.
        current_time : float, optional
            Monotonic timestamp of calibration event.
        operator_id : str
            Who or what performed the calibration (e.g. "auto", "modbus_remote",
            "technician").
        validity_hours : float, optional
            Override default calibration validity window.
        skip_warmup : bool
            When True the warm-up timer is NOT reset after calibration.
            Use this for remote/automated re-calibrations where the sensor
            has already been running and the membrane/electrode is stable.
            When False (default) the sensor enters WARMING_UP until the
            full ``warmup_time_s`` has elapsed, matching behaviour after a
            physical replacement or initial power-on.

        Returns
        -------
        CalibrationRecord
            Full record of the calibration event including the skip_warmup flag.
        """
        with self._state_lock:
            if current_time is None:
                current_time = time.monotonic()

            # Measure current value
            measured_value = self.current_value

            # Calculate offset needed
            offset = reference_value - measured_value

            # Apply calibration
            self.calibration_offset = offset
            self.current_value = reference_value
            self.last_calibration_time = current_time
            self.cumulative_drift = 0.0  # Reset drift tracking
            self.status = SensorStatus.NORMAL
            self.fault = SensorFault.NONE

            # Warm-up control:
            #   skip_warmup=False → reset timer (standard physical replacement)
            #   skip_warmup=True  → sensor already warm; keep power_on_time unchanged
            if not skip_warmup:
                self.power_on_time = current_time

            # Record calibration event
            record = CalibrationRecord(
                timestamp=current_time,
                reference_value=reference_value,
                measured_value=measured_value,
                offset=offset,
                operator_id=operator_id,
                validity_hours=validity_hours or self.calibration_validity_hours,
                skip_warmup=skip_warmup,
            )

            self.calibration_history.append(record)

            return record

    def get_recent_readings(self, window_seconds: float) -> List[SensorReading]:
        """
        Get readings from recent time window.

        Args:
            window_seconds: Time window to retrieve [s]

        Returns:
            List of readings within window (newest first)
        """
        with self._state_lock:
            if not self.reading_history:
                return []

            current_time = self.reading_history[-1].timestamp
            cutoff_time = current_time - window_seconds

            return [
                r for r in reversed(self.reading_history) if r.timestamp >= cutoff_time
            ]

    def calculate_drift_rate(self, window_seconds: float = 3600.0) -> float:
        """
        Calculate current drift rate from recent readings.

        Uses linear regression on drift values over time window.

        Args:
            window_seconds: Time window for calculation [s]

        Returns:
            Drift rate [units/hour]
        """
        recent = self.get_recent_readings(window_seconds)

        if len(recent) < 2:
            return 0.0

        # Extract timestamps and drift values
        times = np.array([r.timestamp for r in recent])
        drifts = np.array([r.drift for r in recent])

        # Linear regression: drift = a * time + b
        if len(times) > 1:
            dt = times[-1] - times[0]
            if dt > 0:
                ddrift = drifts[-1] - drifts[0]
                drift_rate = (ddrift / dt) * 3600.0  # Convert to per hour
                return drift_rate

        return 0.0

    def get_statistics(self, window_seconds: float = 60.0) -> Dict[str, float]:
        """
        Calculate statistics over recent readings.

        Args:
            window_seconds: Time window [s]

        Returns:
            Dictionary with mean, std, min, max, drift_rate
        """
        recent = self.get_recent_readings(window_seconds)

        if not recent:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "drift_rate": 0.0,
                "fault_rate": 0.0,
            }

        # Filter out NaN values for statistics
        values = np.array([r.value for r in recent if np.isfinite(r.value)])

        if len(values) == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "count": len(recent),
                "drift_rate": 0.0,
                "fault_rate": 1.0,  # All readings were faults
            }

        fault_count = sum(1 for r in recent if not np.isfinite(r.value))

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(recent),
            "drift_rate": self.calculate_drift_rate(window_seconds),
            "fault_rate": fault_count / len(recent),
        }

    def reset(self) -> None:
        """
        Reset sensor to initial state.

        Clears history and resets calibration.
        Used for testing and commissioning.
        """
        with self._state_lock:
            self.current_value = (self.min_value + self.max_value) / 2.0
            self.calibration_offset = 0.0
            self.cumulative_drift = 0.0
            self.reading_history.clear()
            self.calibration_history.clear()
            self.status = SensorStatus.NORMAL
            self.fault = SensorFault.NONE
            self.last_calibration_time = time.monotonic()
            self.power_on_time = time.monotonic()
            self._last_direction = 0

            if self.sample_line:
                self.sample_line.delay_buffer.clear()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"value={self.current_value:.3f}, "
            f"status={self.status.value}, "
            f"drift={self.cumulative_drift:.4f})"
        )
