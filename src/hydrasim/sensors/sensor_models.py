"""Shared sensor data models and value types."""

import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Tuple


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
    skip_warmup: bool = False  # Whether warmup was bypassed (remote maintenance)

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
    Sample line characteristics.

    In real plants, sensors are often in sample lines with:
    - Transport delay (10-60s typical)
    - Temperature change
    - Dilution effects

    This often has a large effect in plant deployments.
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
