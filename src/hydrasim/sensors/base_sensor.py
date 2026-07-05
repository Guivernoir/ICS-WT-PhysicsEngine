"""
Base Sensor Module (Enhanced Realism)
======================================

Abstract base class for all sensor implementations.

Provides common functionality:
- Noise generation (Gaussian + drift)
- Response time dynamics (first-order lag)
- Calibration with drift and expiration
- Measurement history (bounded buffer)
- Timestamp generation
- Physical bounds enforcement
- Sample line transport delays
- Sensor fault detection
- Warm-up periods
- Hysteresis effects
- Remote maintenance support (skip_warmup for field re-calibration)

Security Features:
- Bounded memory (circular buffers with deque)
- Explicit validation (no asserts)
- Thread-safe with locks
- Monotonic time enforcement

Author: Guilherme F. G. Santos
Last updated: February 2026
License: MIT
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional, Tuple
import secrets
import threading
import time

from .sensor_models import (
    CalibrationRecord,
    InstallationQuality,
    SampleLine,
    SensorFault,
    SensorReading,
    SensorStatus,
)
from .sensor_operations import SensorDiagnosticsMixin, SensorServiceMixin

__all__ = [
    "BaseSensor",
    "CalibrationRecord",
    "InstallationQuality",
    "SampleLine",
    "SensorFault",
    "SensorReading",
    "SensorStatus",
]


class BaseSensor(SensorServiceMixin, SensorDiagnosticsMixin, ABC):
    """
    Abstract base class for all sensors.

    Provides common sensor characteristics:
    - Gaussian noise
    - Calibration drift over time
    - First-order response lag
    - Measurement history (bounded)
    - Physical range limits
    - Sample line transport delays
    - Fault detection
    - Warm-up periods
    - Hysteresis
    - Remote recalibration with optional warm-up bypass

    Security Properties:
    - Bounded memory: all buffers use deque with maxlen
    - Thread-safe: locks protect mutable state
    - Explicit validation: no assert statements
    - Monotonic time: enforced with time.monotonic()
    """

    def __init__(
        self,
        name: str,
        measurement_range: Tuple[float, float],
        precision: float,
        response_time: float = 15.0,
        drift_rate: float = 0.0,
        max_history_length: int = 1000,
        sample_line: Optional[SampleLine] = None,
        installation: Optional[InstallationQuality] = None,
        warmup_time_s: float = 1800.0,
        hysteresis_magnitude: float = 0.0,
        calibration_validity_hours: float = 24.0,
        max_rate_of_change: Optional[float] = None,
    ):
        """
        Initialize base sensor.

        Args:
            name: Sensor identifier (e.g., "pH_inlet")
            measurement_range: (min, max) physical range
            precision: Measurement precision (1 sigma noise)
            response_time: First-order time constant [s]
            drift_rate: Calibration drift rate [units/hour]
            max_history_length: Maximum readings to store
            sample_line: Sample line configuration (None = direct measurement)
            installation: Installation quality factors
            warmup_time_s: Time required after power-on [s]
            hysteresis_magnitude: Hysteresis band [units]
            calibration_validity_hours: How long calibration lasts
            max_rate_of_change: Max physically possible rate [units/s]
        """
        # Validate inputs
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("Sensor name must be non-empty string")
        if measurement_range[0] >= measurement_range[1]:
            raise ValueError(f"Invalid range: {measurement_range}")
        if precision <= 0:
            raise ValueError(f"Precision must be positive, got {precision}")
        if response_time <= 0:
            raise ValueError(f"Response time must be positive, got {response_time}")
        if max_history_length < 1:
            raise ValueError(
                f"History length must be positive, got {max_history_length}"
            )

        self.name = name
        self.min_value, self.max_value = measurement_range
        self.precision = precision
        self.response_time = response_time
        self.drift_rate = drift_rate
        self.max_history_length = max_history_length
        self.warmup_time_s = warmup_time_s
        self.hysteresis_magnitude = hysteresis_magnitude
        self.calibration_validity_hours = calibration_validity_hours
        self.max_rate_of_change = max_rate_of_change

        # Sample line model (optional)
        self.sample_line = sample_line

        # Installation quality
        self.installation = installation or InstallationQuality()
        self.installation.validate()

        # State
        self.current_value: float = (self.min_value + self.max_value) / 2.0
        self.status = SensorStatus.NORMAL
        self.fault = SensorFault.NONE

        # Power state
        self.power_on_time = time.monotonic()
        self.supply_voltage = 24.0  # [VDC] nominal
        self.voltage_tolerance = (20.0, 28.0)

        # Calibration
        self.calibration_offset: float = 0.0
        self.last_calibration_time: float = time.monotonic()
        self.calibration_history: Deque[CalibrationRecord] = deque(maxlen=100)

        # Measurement history (bounded with deque)
        self.reading_history: Deque[SensorReading] = deque(maxlen=max_history_length)

        # Drift tracking
        self.cumulative_drift: float = 0.0

        # Hysteresis tracking
        self._last_direction: int = 0  # -1, 0, +1

        # Random number generator (thread-safe, cryptographically seeded)
        self._rng_lock = threading.Lock()
        self._rng = np.random.default_rng(seed=secrets.randbits(128))

        # Thread safety for all mutable state
        self._state_lock = threading.RLock()

    def _get_rng(self) -> np.random.Generator:
        """Get thread-safe random number generator."""
        with self._rng_lock:
            return self._rng

    @abstractmethod
    def _get_true_value(self, reactor_state) -> float:
        """
        Read the true physical value from reactor state.

        Must be implemented by subclasses to extract the relevant
        parameter from the reactor.

        Args:
            reactor_state: Current reactor state object

        Returns:
            True physical value (before sensor effects)
        """
        pass

    def read(
        self, reactor_state, current_time: Optional[float] = None
    ) -> SensorReading:
        """
        Take a sensor reading from the reactor.

        Process:
        1. Validate time is monotonic
        2. Check warm-up status
        3. Check calibration validity
        4. Get true value from reactor
        5. Apply sample line transport delay
        6. Apply calibration drift
        7. Add measurement noise
        8. Apply response lag (first-order)
        9. Apply hysteresis
        10. Apply installation effects
        11. Check for faults
        12. Enforce physical bounds
        13. Create reading with metadata
        14. Add to history

        Args:
            reactor_state: Current reactor state
            current_time: Timestamp [s], defaults to time.monotonic()

        Returns:
            SensorReading with all metadata
        """
        with self._state_lock:
            if current_time is None:
                current_time = time.monotonic()

            # Validate monotonic time
            if (
                self.reading_history
                and current_time < self.reading_history[-1].timestamp
            ):
                raise ValueError(
                    f"Non-monotonic time: {current_time} < {self.reading_history[-1].timestamp}"
                )

            rng = self._get_rng()

            # Check for power fault BEFORE randomizing voltage
            initial_voltage = self.supply_voltage
            if not (
                self.voltage_tolerance[0] < initial_voltage < self.voltage_tolerance[1]
            ):
                # Power fault detected
                reading = SensorReading(
                    timestamp=current_time,
                    value=np.nan,
                    raw_value=np.nan,
                    noise=0.0,
                    drift=0.0,
                    status=SensorStatus.POWER_FAULT,
                    uncertainty=0.0,
                    fault=(
                        SensorFault.POWER_LOW
                        if initial_voltage < self.voltage_tolerance[0]
                        else SensorFault.POWER_HIGH
                    ),
                )
                self.reading_history.append(reading)
                return reading

            # Simulate normal power supply fluctuations
            self.supply_voltage = 24.0 + rng.normal(0.0, 1.0)

            # Check warm-up status
            if not self._check_warmup(current_time):
                reading = SensorReading(
                    timestamp=current_time,
                    value=np.nan,
                    raw_value=np.nan,
                    noise=0.0,
                    drift=0.0,
                    status=SensorStatus.WARMING_UP,
                    uncertainty=0.0,
                    fault=SensorFault.NONE,
                )
                self.reading_history.append(reading)
                return reading

            # Check calibration validity
            cal_expired = not self._check_calibration_valid(current_time)
            if cal_expired:
                self.status = SensorStatus.CALIBRATION_EXPIRED

            # Step 1: Get true physical value
            true_value = self._get_true_value(reactor_state)

            # Step 2: Apply sample line transport delay.
            if self.sample_line is not None:
                # Get temperature if available
                if hasattr(reactor_state, "temperature"):
                    zone_idx = getattr(self, "zone_index", 0)
                    temp = reactor_state.temperature[zone_idx]
                else:
                    temp = 25.0

                delayed_value, delayed_temp = self.sample_line.transport_sample(
                    true_value, temp, current_time
                )
                true_value = delayed_value

            # Step 3: Apply calibration drift
            time_since_calibration = current_time - self.last_calibration_time
            drift_hours = time_since_calibration / 3600.0
            aging_drift = self.drift_rate * drift_hours
            current_drift = aging_drift + self.calibration_offset
            self.cumulative_drift = aging_drift

            # Step 4: Add Gaussian measurement noise
            noise = rng.normal(0.0, self.precision)

            # Step 5: Apply first-order response lag using configured time constant.
            # Discrete form of x_dot = (u - x) / tau:
            # alpha = 1 - exp(-dt/tau)
            if len(self.reading_history) > 0:
                dt_filter = max(0.0, current_time - self.reading_history[-1].timestamp)
            else:
                # First sample after startup/calibration should settle quickly.
                dt_filter = self.response_time

            if dt_filter > 0:
                alpha = 1.0 - np.exp(-dt_filter / max(self.response_time, 1e-9))
            else:
                alpha = 0.0

            raw_with_noise = true_value + noise + current_drift
            self.current_value = (
                alpha * raw_with_noise + (1 - alpha) * self.current_value
            )

            # Step 6: Apply hysteresis
            self.current_value = self._apply_hysteresis(self.current_value)

            # Step 7: Apply installation effects
            self.current_value = self._apply_installation_effects(
                self.current_value, rng
            )

            # Step 8: Calculate rate of change
            if len(self.reading_history) > 0:
                last_reading = self.reading_history[-1]
                dt = current_time - last_reading.timestamp
                if dt > 0 and np.isfinite(last_reading.value):
                    rate_of_change = (self.current_value - last_reading.value) / dt
                else:
                    rate_of_change = 0.0
            else:
                rate_of_change = 0.0

            # Step 9: Check for faults
            fault = self._check_for_faults(self.current_value, rate_of_change)
            if fault is not None and fault != SensorFault.NONE:
                self.fault = fault
                if fault in [SensorFault.OPEN_CIRCUIT, SensorFault.SHORT_CIRCUIT]:
                    self.status = SensorStatus.FAILED
                    self.current_value = np.nan
                elif fault == SensorFault.OUT_OF_RANGE:
                    self.status = SensorStatus.OUT_OF_RANGE
                elif fault in [SensorFault.POWER_LOW, SensorFault.POWER_HIGH]:
                    self.status = SensorStatus.POWER_FAULT
                elif fault == SensorFault.RATE_FAULT:
                    self.status = SensorStatus.RATE_OF_CHANGE_FAULT
            else:
                self.fault = SensorFault.NONE

                # Check for saturation
                if not np.isnan(self.current_value):
                    bounded_value = np.clip(
                        self.current_value, self.min_value, self.max_value
                    )
                    if bounded_value != self.current_value:
                        self.status = SensorStatus.SATURATED
                    elif not cal_expired:  # Only set NORMAL if cal not expired
                        self.status = SensorStatus.NORMAL

                    self.current_value = bounded_value

                # Check for excessive drift (but don't override calibration expired)
                # Calibration offset is an intentional correction and should not
                # trigger a drift alarm. Only aging drift contributes to warnings.
                if abs(aging_drift) > 0.1 * (self.max_value - self.min_value):
                    if self.status != SensorStatus.CALIBRATION_EXPIRED:
                        self.status = SensorStatus.DRIFT_WARNING

            # Step 10: Create reading object
            reading = SensorReading(
                timestamp=current_time,
                value=self.current_value,
                raw_value=true_value,
                noise=noise,
                drift=current_drift,
                status=self.status,
                uncertainty=self.precision * 2.0,  # ±2σ ~95% confidence
                fault=self.fault,
            )

            # Step 11: Add to history (deque automatically maintains max length)
            self.reading_history.append(reading)

            return reading
