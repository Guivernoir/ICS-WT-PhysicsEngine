"""
Base actuator abstract class.

Implements core features required by all actuators:
- Thread-safe state management
- Monotonic time validation
- Bounded memory buffers
- Wear/degradation tracking
- Fault injection for testing
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import threading
import time
from typing import Any, Deque, Dict, Optional
import secrets

import numpy as np


class ActuatorStatus(Enum):
    """Actuator operational status."""

    NORMAL = auto()
    DEGRADED = auto()
    FAILED = auto()


class ActuatorFault(Enum):
    """Fault conditions for actuators."""

    NONE = auto()
    STUCK = auto()
    SLOW = auto()
    LEAKING = auto()
    OVER_CYCLE = auto()
    CALIBRATION = auto()


@dataclass
class ActuatorDiagnostics:
    """Diagnostic information for actuator health monitoring."""

    cycles_count: int
    hours_runtime: float
    wear_factor: float
    health_status: ActuatorStatus
    fault_code: ActuatorFault
    last_maintenance: Optional[float] = None
    position_deviation_rms: float = 0.0
    average_response_time: float = 0.0


class BaseActuator(ABC):
    """
    Abstract base class for all actuators.

    Implements:
    - Thread-safe state management with RLock
    - Monotonic time validation
    - Bounded memory buffers (deque with maxlen)
    - Wear and degradation tracking
    - First-order lag dynamics
    - Stiction, hysteresis, and deadband models

    Design notes:
    - All inputs are validated
    - All state mutations are protected by locks
    - No unbounded memory growth
    - Opaque error messages (no internal paths)
    """

    def __init__(
        self,
        name: str,
        response_time: float = 5.0,
        deadband: float = 0.01,
        hysteresis: float = 0.005,
        stiction: float = 0.02,
        max_history: int = 1000,
    ):
        """
        Initialize base actuator.

        Args:
            name: Actuator identifier
            response_time: Time constant for first-order lag (seconds)
            deadband: Position change threshold (±% ignored)
            hysteresis: Different response up vs down (%)
            stiction: Initial resistance to movement (%)
            max_history: Maximum history buffer size (bounded memory)

        Raises:
            TypeError: If arguments are not of expected types
            ValueError: If arguments are out of valid ranges
        """
        # Validate constructor inputs.
        if not isinstance(name, str):
            raise TypeError("name must be str")
        if not isinstance(response_time, (int, float)) or response_time <= 0:
            raise ValueError("response_time must be positive number")
        if not isinstance(deadband, (int, float)) or not 0 <= deadband <= 1:
            raise ValueError("deadband must be in [0, 1]")
        if not isinstance(hysteresis, (int, float)) or not 0 <= hysteresis <= 1:
            raise ValueError("hysteresis must be in [0, 1]")
        if not isinstance(stiction, (int, float)) or not 0 <= stiction <= 1:
            raise ValueError("stiction must be in [0, 1]")
        if not isinstance(max_history, int) or max_history <= 0:
            raise ValueError("max_history must be positive integer")

        # Identity
        self._name = name

        # State variables (protected by lock)
        self._state_lock = threading.RLock()
        self._commanded_position = 0.0
        self._actual_position = 0.0
        self._actual_flow = 0.0
        self._last_movement_direction = 0  # -1: closing, 0: stationary, 1: opening

        # Dynamics parameters
        self._response_time = response_time
        self._deadband = deadband
        self._hysteresis = hysteresis
        self._stiction = stiction

        # Wear and degradation
        self._cycles_count = 0
        self._hours_runtime = 0.0
        self._wear_factor = 0.0  # 0=new, 1=worn out

        # Health status
        self._health_status = ActuatorStatus.NORMAL
        self._fault_code = ActuatorFault.NONE

        # Time tracking (monotonic)
        self._last_step_time = time.monotonic()
        self._initialized_time = self._last_step_time

        # Bounded memory history buffers
        self._position_history: Deque[float] = deque(maxlen=max_history)
        self._command_history: Deque[float] = deque(maxlen=max_history)
        self._timestamp_history: Deque[float] = deque(maxlen=max_history)

        # Thread-safe, cryptographically seeded RNG (same pattern as sensors)
        self._rng_lock = threading.Lock()
        self._rng = np.random.default_rng(seed=secrets.randbits(128))

    @property
    def name(self) -> str:
        """Get actuator name."""
        return self._name

    def _get_rng(self) -> np.random.Generator:
        """Get thread-safe random number generator."""
        with self._rng_lock:
            return self._rng

    def set_position(self, setpoint: float) -> None:
        """
        Set commanded position/setpoint.

        Args:
            setpoint: Desired position (units depend on actuator type)

        Raises:
            TypeError: If setpoint is not numeric
            ValueError: If setpoint is out of valid range
        """
        # Validate setpoint input.
        if not isinstance(setpoint, (int, float)):
            raise TypeError("setpoint must be numeric")

        # Validate range (override in subclasses if needed)
        self._validate_setpoint(setpoint)

        with self._state_lock:
            self._commanded_position = float(setpoint)
            self._command_history.append(setpoint)

    @abstractmethod
    def _validate_setpoint(self, setpoint: float) -> None:
        """
        Validate setpoint range (subclass-specific).

        Args:
            setpoint: Value to validate

        Raises:
            ValueError: If setpoint is invalid
        """
        pass

    def step(self, dt: float, current_time: Optional[float] = None) -> float:
        """
        Advance actuator dynamics by dt seconds.

        Implements:
        - First-order lag dynamics
        - Stiction and hysteresis
        - Wear accumulation
        - Monotonic time enforcement

        Args:
            dt: Time step in seconds
            current_time: Current monotonic time (auto-generated if None)

        Returns:
            Actual output value (flow rate or position)

        Raises:
            TypeError: If arguments are invalid types
            ValueError: If dt is non-positive or time is non-monotonic
        """
        # Validation
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError("dt must be positive number")

        if current_time is None:
            current_time = time.monotonic()

        if not isinstance(current_time, (int, float)):
            raise TypeError("current_time must be numeric")

        # Enforce monotonic time
        with self._state_lock:
            if current_time < self._last_step_time:
                raise ValueError("Non-monotonic time detected")

            # Update runtime hours
            self._hours_runtime += dt / 3600.0

            # Compute dynamics
            self._update_position_dynamics(dt)

            # Update wear
            self._update_wear()

            # Check for faults
            self._check_faults()

            # Record history
            self._position_history.append(self._actual_position)
            self._timestamp_history.append(current_time)

            self._last_step_time = current_time

            return self._actual_flow

    def _update_position_dynamics(self, dt: float) -> None:
        """
        Update actuator position using first-order lag with stiction.

        Implements:
        - Stiction (deadband to overcome static friction)
        - Hysteresis (different response opening vs closing)
        - First-order lag (exponential approach to setpoint)

        Args:
            dt: Time step in seconds
        """
        position_error = self._commanded_position - self._actual_position

        # Apply deadband (ignore small changes)
        if abs(position_error) < self._deadband:
            return

        # Determine movement direction
        movement_direction = 1 if position_error > 0 else -1

        # Apply stiction (need to overcome initial resistance)
        if self._last_movement_direction == 0:
            # Stationary - check if error exceeds stiction
            if abs(position_error) < self._stiction:
                return

        # Apply hysteresis (different response depending on direction)
        effective_response_time = self._response_time
        if (
            movement_direction != self._last_movement_direction
            and self._last_movement_direction != 0
        ):
            # Direction reversal - add hysteresis penalty
            effective_response_time *= 1.0 + self._hysteresis

        # Apply wear factor (increases response time)
        effective_response_time *= 1.0 + self._wear_factor

        # First-order lag: dx/dt = (setpoint - actual) / tau
        tau = effective_response_time
        delta_position = (position_error / tau) * dt

        # Update position
        old_position = self._actual_position
        self._actual_position += delta_position

        # Track movement direction and cycles
        if abs(self._actual_position - old_position) > 0.001:
            new_direction = 1 if (self._actual_position - old_position) > 0 else -1
            if (
                new_direction != self._last_movement_direction
                and self._last_movement_direction != 0
            ):
                self._cycles_count += 1
            self._last_movement_direction = new_direction
        else:
            self._last_movement_direction = 0

        # Update flow (override in subclasses)
        self._actual_flow = self._compute_flow()

    @abstractmethod
    def _compute_flow(self) -> float:
        """
        Compute actual flow based on position (subclass-specific).

        Returns:
            Flow rate in L/min
        """
        pass

    def _update_wear(self) -> None:
        """
        Update wear factor based on cycles and runtime.

        Wear model:
        - Increases with cycle count
        - Increases with runtime hours
        - Affects response time and creates potential faults
        """
        # Simple linear wear model (can be enhanced)
        cycle_wear = self._cycles_count / 1_000_000.0  # 1M cycles = full wear
        time_wear = self._hours_runtime / 87_600.0  # 10 years = full wear

        self._wear_factor = min(1.0, cycle_wear + time_wear)

        # Update health status based on wear
        if self._wear_factor > 0.8:
            self._health_status = ActuatorStatus.FAILED
        elif self._wear_factor > 0.5:
            self._health_status = ActuatorStatus.DEGRADED

    def _check_faults(self) -> None:
        """Check for fault conditions."""
        # High wear can cause faults
        if self._wear_factor > 0.9:
            self._fault_code = ActuatorFault.OVER_CYCLE
        elif self._wear_factor > 0.7:
            self._fault_code = ActuatorFault.SLOW

    def actual_position(self) -> float:
        """Get current actual position."""
        with self._state_lock:
            return self._actual_position

    def actual_flow(self) -> float:
        """Get actual flow output in L/min."""
        with self._state_lock:
            return self._actual_flow

    def commanded_position(self) -> float:
        """Get commanded setpoint."""
        with self._state_lock:
            return self._commanded_position

    def has_fault(self) -> bool:
        """Check if actuator has any active faults."""
        with self._state_lock:
            return self._fault_code != ActuatorFault.NONE

    def health_status(self) -> ActuatorStatus:
        """Get health status."""
        with self._state_lock:
            return self._health_status

    def fault_code(self) -> ActuatorFault:
        """Get current fault code."""
        with self._state_lock:
            return self._fault_code

    def calibrate_zero(self) -> None:
        """Perform zero/span calibration."""
        with self._state_lock:
            # Reset to known state
            self._actual_position = 0.0
            self._commanded_position = 0.0
            self._actual_flow = 0.0
            self._last_movement_direction = 0

    def reset_faults(self) -> None:
        """Reset fault conditions (requires explicit action)."""
        with self._state_lock:
            if self._wear_factor < 0.8:  # Only if not critically worn
                self._fault_code = ActuatorFault.NONE
                if self._wear_factor < 0.5:
                    self._health_status = ActuatorStatus.NORMAL
                else:
                    self._health_status = ActuatorStatus.DEGRADED

    def simulate_fault(self, fault: ActuatorFault) -> None:
        """
        Inject fault for testing.

        Args:
            fault: Fault type to simulate
        """
        with self._state_lock:
            self._fault_code = fault
            if fault != ActuatorFault.NONE:
                self._health_status = ActuatorStatus.FAILED

    def diagnostics(self) -> ActuatorDiagnostics:
        """
        Get comprehensive diagnostics.

        Returns:
            ActuatorDiagnostics object with health metrics
        """
        with self._state_lock:
            # Compute position deviation RMS
            if len(self._position_history) > 1:
                deviations = [
                    abs(cmd - pos)
                    for cmd, pos in zip(
                        list(self._command_history)[-len(self._position_history) :],
                        self._position_history,
                    )
                ]
                position_deviation_rms = (
                    sum(d**2 for d in deviations) / len(deviations)
                ) ** 0.5
            else:
                position_deviation_rms = 0.0

            return ActuatorDiagnostics(
                cycles_count=self._cycles_count,
                hours_runtime=self._hours_runtime,
                wear_factor=self._wear_factor,
                health_status=self._health_status,
                fault_code=self._fault_code,
                last_maintenance=None,  # Track separately if needed
                position_deviation_rms=position_deviation_rms,
                average_response_time=self._response_time * (1.0 + self._wear_factor),
            )

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get current state as dictionary (for serialization/logging).

        Returns:
            Dictionary with current state
        """
        with self._state_lock:
            return {
                "name": self._name,
                "commanded_position": self._commanded_position,
                "actual_position": self._actual_position,
                "actual_flow": self._actual_flow,
                "cycles_count": self._cycles_count,
                "hours_runtime": self._hours_runtime,
                "wear_factor": self._wear_factor,
                "health_status": self._health_status.name,
                "fault_code": self._fault_code.name,
            }
