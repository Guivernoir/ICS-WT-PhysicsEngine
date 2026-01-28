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
- Sample line transport delays (CRITICAL for realism)
- Sensor fault detection
- Warm-up periods
- Hysteresis effects

Key Enhancements for Realism:
- Sample line delays (10-60s typical in real plants)
- Calibration expiration (sensors need re-cal after time limit)
- Fault states (open circuit, out of range, power issues)
- Warm-up requirements (sensors need stabilization time)
- Hysteresis (different readings on rising vs falling)
- Installation quality effects (flow velocity, bubbles)

Security Features:
- Bounded memory (circular buffers with deque)
- Explicit validation (no asserts)
- Thread-safe with locks
- Monotonic time enforcement

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Deque
from enum import Enum
from collections import deque
import time
import threading
import secrets


class SensorStatus(Enum):
    """Sensor operational status."""

    NORMAL = "normal"
    CALIBRATING = "calibrating"
    WARMING_UP = "warming_up"
    FAILED = "failed"
    SATURATED = "saturated"
    DRIFT_WARNING = "drift_warning"
    CALIBRATION_EXPIRED = "calibration_expired"
    OPEN_CIRCUIT = "open_circuit"
    SHORT_CIRCUIT = "short_circuit"
    OUT_OF_RANGE = "out_of_range"
    POWER_FAULT = "power_fault"
    RATE_OF_CHANGE_FAULT = "rate_of_change_fault"


class SensorFault(Enum):
    """Specific fault types that sensors can detect."""

    NONE = "none"
    OPEN_CIRCUIT = "open_circuit"  # Wire disconnected
    SHORT_CIRCUIT = "short_circuit"  # Wire shorted
    OUT_OF_RANGE = "out_of_range"  # Reading physically impossible
    RATE_FAULT = "rate_fault"  # Changed too fast (unphysical)
    POWER_LOW = "power_low"  # Supply voltage too low
    POWER_HIGH = "power_high"  # Supply voltage too high


@dataclass
class SensorReading:
    """
    Single sensor reading with metadata.

    Immutable data class representing one measurement.
    """

    timestamp: float  # [s] Unix timestamp
    value: float  # Measured value in sensor units (or np.nan if fault)
    raw_value: float  # Value before noise/drift/transport
    noise: float  # Noise component added
    drift: float  # Calibration drift component
    status: SensorStatus = SensorStatus.NORMAL
    uncertainty: float = 0.0  # Measurement uncertainty (±)
    fault: SensorFault = SensorFault.NONE

    def __post_init__(self):
        """Validate reading values - use explicit checks, not assert."""
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError(f"Timestamp must be numeric, got {type(self.timestamp)}")
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be positive, got {self.timestamp}")
        # Allow NaN for fault conditions
        if not (np.isfinite(self.value) or np.isnan(self.value)):
            raise ValueError(f"Sensor reading must be finite or NaN, got {self.value}")


@dataclass
class CalibrationRecord:
    """Record of sensor calibration event."""

    timestamp: float  # [s] When calibration occurred
    reference_value: float  # Known reference value
    measured_value: float  # What sensor read before calibration
    offset: float  # Calibration offset applied
    operator_id: str = "auto"
    notes: str = ""
    validity_hours: float = 24.0  # How long calibration is valid

    def is_expired(self, current_time: float) -> bool:
        """Check if this calibration has expired."""
        hours_elapsed = (current_time - self.timestamp) / 3600.0
        return hours_elapsed > self.validity_hours


@dataclass
class InstallationQuality:
    """
    Sensor installation quality factors.

    These affect measurement accuracy in real plants.
    """

    flow_velocity: float = 0.5  # [m/s] at sensor location (0.1-2.0 typical)
    air_bubble_frequency: float = 0.0  # [bubbles/min]
    grounding_quality: float = 1.0  # 0-1, poor to excellent
    pipe_vibration_g: float = 0.0  # [g RMS] mechanical vibration
    ambient_temperature: float = 25.0  # [°C] around sensor electronics

    def validate(self):
        """Validate installation parameters."""
        if not 0.0 <= self.flow_velocity <= 5.0:
            raise ValueError(f"Flow velocity {self.flow_velocity} m/s out of range")
        if not 0.0 <= self.grounding_quality <= 1.0:
            raise ValueError("Grounding quality must be 0-1")
        if self.pipe_vibration_g < 0:
            raise ValueError("Vibration must be non-negative")


@dataclass
class SampleLine:
    """
    Sample line characteristics - CRITICAL for realism.

    In real plants, sensors are often in sample lines with:
    - Transport delay (10-60s typical)
    - Temperature change
    - Dilution effects

    This is the #1 difference between simulation and reality.
    """

    volume_mL: float = 100.0  # Sample line internal volume
    flow_rate_mL_min: float = 500.0  # Sample flow rate
    ambient_temp: float = 20.0  # Temperature sample line is exposed to

    def __post_init__(self):
        """Calculate derived parameters."""
        self.volume_L = self.volume_mL / 1000.0
        self.flow_rate_L_s = self.flow_rate_mL_min / 1000.0 / 60.0
        self.transport_delay_s = (
            self.volume_L / self.flow_rate_L_s if self.flow_rate_L_s > 0 else 0.0
        )

        # Circular buffer for transport delay simulation
        max_samples = max(100, int(self.transport_delay_s) + 10)
        self.delay_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=max_samples)

    def transport_sample(
        self, value: float, temp: float, timestamp: float
    ) -> Tuple[float, float]:
        """
        Simulate sample transport through line.

        Returns delayed and temperature-adjusted sample.
        """
        # Add current sample to buffer
        self.delay_buffer.append((timestamp, value, temp))

        # Find sample that matches the transport delay
        target_time = timestamp - self.transport_delay_s

        # Get closest sample from buffer
        if len(self.delay_buffer) == 0:
            return value, temp  # No delay data yet

        # Linear search for closest sample (buffer is small, <100 samples)
        closest_sample = self.delay_buffer[0]
        min_time_diff = abs(closest_sample[0] - target_time)

        for sample in self.delay_buffer:
            time_diff = abs(sample[0] - target_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_sample = sample

        delayed_time, delayed_value, delayed_temp = closest_sample

        # Temperature change in sample line (heat transfer to ambient)
        # Exponential approach: T(t) = T_amb + (T0 - T_amb) * exp(-t/tau)
        # Simplified: 10% approach per second
        time_in_line = timestamp - delayed_time
        temp_fraction = np.exp(-0.1 * time_in_line)
        actual_temp = (
            self.ambient_temp + (delayed_temp - self.ambient_temp) * temp_fraction
        )

        return delayed_value, actual_temp


class BaseSensor(ABC):
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

        # Sample line (CRITICAL for realism)
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
            fault_type = rng.choice(
                [SensorFault.OPEN_CIRCUIT, SensorFault.SHORT_CIRCUIT]
            )
            return fault_type

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
        5. Apply sample line transport delay (CRITICAL)
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

            # Step 2: Apply sample line transport delay (CRITICAL for realism)
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
            current_drift = self.drift_rate * drift_hours + self.calibration_offset
            self.cumulative_drift = current_drift

            # Step 4: Add Gaussian measurement noise
            noise = rng.normal(0.0, self.precision)

            # Step 5: Apply first-order response lag
            alpha = 0.5  # More responsive than before (was 0.1)
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
                if abs(current_drift) > 0.1 * (self.max_value - self.min_value):
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

    def calibrate(
        self,
        reference_value: float,
        current_time: Optional[float] = None,
        operator_id: str = "auto",
        validity_hours: Optional[float] = None,
    ) -> CalibrationRecord:
        """
        Calibrate sensor against known reference.

        Calculates offset between reference and current reading,
        then resets calibration to this offset.

        Args:
            reference_value: Known true value
            current_time: Timestamp of calibration
            operator_id: Who performed calibration
            validity_hours: Override default calibration validity

        Returns:
            CalibrationRecord documenting the calibration
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
            self.last_calibration_time = current_time
            self.cumulative_drift = 0.0  # Reset drift tracking
            self.status = SensorStatus.NORMAL
            self.fault = SensorFault.NONE

            # Reset warm-up timer (calibration requires stabilization)
            self.power_on_time = current_time

            # Record calibration event
            record = CalibrationRecord(
                timestamp=current_time,
                reference_value=reference_value,
                measured_value=measured_value,
                offset=offset,
                operator_id=operator_id,
                validity_hours=validity_hours or self.calibration_validity_hours,
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
