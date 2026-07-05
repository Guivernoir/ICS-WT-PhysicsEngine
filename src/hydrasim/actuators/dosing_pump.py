"""
Dosing Pump module.

Implements metering pumps for chemical dosing:
- Diaphragm pumps (membrane displacement)
- Peristaltic pumps (tube compression)
- Pulsation modeling (inherent to positive displacement)
- Back pressure effects on flow rate
- Stroke rate and volume control
- Check valve failures
"""

from .base_actuator import BaseActuator, ActuatorFault, ActuatorStatus
from enum import Enum, auto
from typing import Optional
import math


class PumpType(Enum):
    """Type of dosing pump."""

    DIAPHRAGM = auto()
    PERISTALTIC = auto()
    PISTON = auto()


class DosingPump(BaseActuator):
    """
    Positive displacement metering pump for chemical dosing.

    Models:
    - Stroke-based flow (flow rate = stroke_rate * stroke_volume)
    - Pulsation (instantaneous flow varies with stroke cycle)
    - Back pressure effects (reduces effective flow)
    - Check valve failures (causes backflow)
    - Diaphragm wear (reduces stroke volume)
    - Tube wear for peristaltic pumps

    Control modes:
    - Stroke rate control (strokes per minute)
    - Flow rate control (L/min) - converted to stroke rate

    Thread-safe: All state access protected by lock from BaseActuator.
    """

    def __init__(
        self,
        name: str,
        pump_type: PumpType,
        max_flow_rate: float,
        max_stroke_rate: float = 300.0,
        stroke_volume_mL: Optional[float] = None,
        discharge_pressure_bar: float = 5.0,
        max_pressure_bar: float = 10.0,
        response_time: float = 2.0,
        check_valve_efficiency: float = 0.98,
        pulsation_damping: float = 0.3,
        max_history: int = 1000,
    ):
        """
        Initialize dosing pump.

        Args:
            name: Pump identifier
            pump_type: DIAPHRAGM, PERISTALTIC, or PISTON
            max_flow_rate: Maximum flow rate (L/min)
            max_stroke_rate: Maximum strokes per minute
            stroke_volume_mL: Volume per stroke (auto-calculated if None)
            discharge_pressure_bar: System back pressure (bar)
            max_pressure_bar: Maximum rated pressure (bar)
            response_time: Time to reach commanded stroke rate (seconds)
            check_valve_efficiency: Check valve sealing efficiency (0-1)
            pulsation_damping: Pulsation dampener effectiveness (0-1)
            max_history: Maximum history buffer size

        Raises:
            TypeError: If arguments are not expected types
            ValueError: If arguments are out of valid ranges
        """
        # Validate pump-specific parameters
        if not isinstance(pump_type, PumpType):
            raise TypeError("pump_type must be PumpType enum")
        if not isinstance(max_flow_rate, (int, float)) or max_flow_rate <= 0:
            raise ValueError("max_flow_rate must be positive")
        if not isinstance(max_stroke_rate, (int, float)) or max_stroke_rate <= 0:
            raise ValueError("max_stroke_rate must be positive")
        if (
            not isinstance(discharge_pressure_bar, (int, float))
            or discharge_pressure_bar < 0
        ):
            raise ValueError("discharge_pressure_bar must be non-negative")
        if not isinstance(max_pressure_bar, (int, float)) or max_pressure_bar <= 0:
            raise ValueError("max_pressure_bar must be positive")
        if (
            not isinstance(check_valve_efficiency, (int, float))
            or not 0 <= check_valve_efficiency <= 1
        ):
            raise ValueError("check_valve_efficiency must be in [0, 1]")
        if (
            not isinstance(pulsation_damping, (int, float))
            or not 0 <= pulsation_damping <= 1
        ):
            raise ValueError("pulsation_damping must be in [0, 1]")

        # Calculate stroke volume if not provided
        if stroke_volume_mL is None:
            stroke_volume_mL = (max_flow_rate * 1000.0) / max_stroke_rate
        else:
            if not isinstance(stroke_volume_mL, (int, float)) or stroke_volume_mL <= 0:
                raise ValueError("stroke_volume_mL must be positive")

        # Initialize base actuator
        super().__init__(
            name=name,
            response_time=response_time,
            deadband=0.01,  # 1% deadband for stroke rate
            hysteresis=0.005,
            stiction=0.01,  # Less stiction than valves
            max_history=max_history,
        )

        # Pump-specific parameters
        self._pump_type = pump_type
        self._max_flow_rate = max_flow_rate
        self._max_stroke_rate = max_stroke_rate
        self._stroke_volume_mL = stroke_volume_mL
        self._discharge_pressure_bar = discharge_pressure_bar
        self._max_pressure_bar = max_pressure_bar
        self._check_valve_efficiency = check_valve_efficiency
        self._pulsation_damping = pulsation_damping

        # Pump state
        self._stroke_rate = 0.0  # Current strokes per minute
        self._commanded_stroke_rate = 0.0
        self._stroke_phase = 0.0  # 0-1 for pulsation modeling

        # Wear-specific for pumps
        self._diaphragm_wear = 0.0  # Reduces effective stroke volume
        self._check_valve_wear = 0.0  # Reduces sealing efficiency
        self._tube_wear = 0.0  # For peristaltic pumps

        # Stroke counter (lifetime total — never reset)
        self._total_strokes = 0
        # Per-component wear counters (reset on hardware replacement)
        self._diaphragm_strokes = 0  # Strokes since last diaphragm replacement
        self._tube_strokes = 0  # Strokes since last tube replacement

    def _validate_setpoint(self, setpoint: float) -> None:
        """
        Validate setpoint range for pump.

        Accepts either:
        - Stroke rate (0 to max_stroke_rate)
        - Flow rate (0 to max_flow_rate)

        Args:
            setpoint: Stroke rate or flow rate

        Raises:
            ValueError: If setpoint is out of range
        """
        # Allow either stroke rate or flow rate as setpoint
        # Will be converted in set_flow_rate or set_stroke_rate
        if not 0.0 <= setpoint <= max(self._max_stroke_rate, self._max_flow_rate * 10):
            raise ValueError("setpoint must be in valid range")

    def set_flow_rate(self, flow_rate: float) -> None:
        """
        Set desired flow rate (converted to stroke rate).

        Args:
            flow_rate: Desired flow rate in L/min

        Raises:
            ValueError: If flow rate is out of range
        """
        if not isinstance(flow_rate, (int, float)):
            raise TypeError("flow_rate must be numeric")
        if not 0.0 <= flow_rate <= self._max_flow_rate:
            raise ValueError(f"flow_rate must be in [0, {self._max_flow_rate}]")

        # Read _diaphragm_wear under the lock — it is mutated by _update_wear()
        # which runs inside step() under _state_lock.
        with self._state_lock:
            effective_stroke_volume = self._stroke_volume_mL * (
                1.0 - self._diaphragm_wear
            )

        # Convert flow rate to stroke rate
        # Q (L/min) = stroke_rate (strokes/min) * stroke_volume (mL) / 1000
        if effective_stroke_volume > 0:
            stroke_rate = (flow_rate * 1000.0) / effective_stroke_volume
            stroke_rate = min(stroke_rate, self._max_stroke_rate)
        else:
            stroke_rate = 0.0

        self.set_position(stroke_rate)

    def set_stroke_rate(self, stroke_rate: float) -> None:
        """
        Set desired stroke rate directly.

        Args:
            stroke_rate: Strokes per minute

        Raises:
            ValueError: If stroke rate is out of range
        """
        if not isinstance(stroke_rate, (int, float)):
            raise TypeError("stroke_rate must be numeric")
        if not 0.0 <= stroke_rate <= self._max_stroke_rate:
            raise ValueError(f"stroke_rate must be in [0, {self._max_stroke_rate}]")

        self.set_position(stroke_rate)

    def _compute_flow(self) -> float:
        """
        Compute actual flow rate from stroke rate.

        Includes:
        - Effective stroke volume (reduced by wear)
        - Back pressure effects
        - Check valve leakage
        - Pulsation (instantaneous variation)

        Returns:
            Instantaneous flow rate in L/min
        """
        # Effective stroke volume reduced by wear
        effective_stroke_volume = self._stroke_volume_mL * (1.0 - self._diaphragm_wear)

        if self._pump_type == PumpType.PERISTALTIC:
            effective_stroke_volume *= 1.0 - self._tube_wear

        # Base flow rate
        base_flow = (self._stroke_rate * effective_stroke_volume) / 1000.0  # L/min

        # Apply back pressure effects
        pressure_factor = 1.0 - (self._discharge_pressure_bar / self._max_pressure_bar)
        pressure_factor = max(0.0, min(1.0, pressure_factor))

        # Apply check valve efficiency (reduced by wear)
        effective_check_valve_efficiency = self._check_valve_efficiency * (
            1.0 - self._check_valve_wear
        )

        # Calculate flow with pressure and leakage effects
        flow_rate = base_flow * pressure_factor * effective_check_valve_efficiency

        # Add pulsation (instantaneous variation)
        pulsation_amplitude = (
            1.0 - self._pulsation_damping
        ) * 0.3  # Up to 30% variation
        pulsation = 1.0 + pulsation_amplitude * math.sin(
            2.0 * math.pi * self._stroke_phase
        )
        flow_rate *= pulsation

        return max(0.0, flow_rate)

    def _update_position_dynamics(self, dt: float) -> None:
        """
        Update pump stroke rate dynamics.

        Args:
            dt: Time step in seconds
        """
        # "Position" for pumps is stroke rate
        # Use first-order lag to reach commanded stroke rate
        stroke_rate_error = self._commanded_position - self._stroke_rate

        # Apply deadband
        if abs(stroke_rate_error) < self._deadband * self._max_stroke_rate:
            return

        # First-order lag
        tau = self._response_time * (1.0 + self._wear_factor)
        delta_stroke_rate = (stroke_rate_error / tau) * dt

        self._stroke_rate += delta_stroke_rate
        self._stroke_rate = max(0.0, min(self._max_stroke_rate, self._stroke_rate))

        # Update stroke phase for pulsation
        if self._stroke_rate > 0:
            phase_increment = (self._stroke_rate / 60.0) * dt  # Convert to Hz
            self._stroke_phase = (self._stroke_phase + phase_increment) % 1.0

            # Count strokes
            if (
                self._stroke_phase < 0.5
                and (self._stroke_phase + phase_increment) >= 0.5
            ):
                self._total_strokes += 1
                self._diaphragm_strokes += 1
                self._tube_strokes += 1
                self._cycles_count = self._total_strokes

        # Update actual position (for base class compatibility)
        self._actual_position = self._stroke_rate

        # Compute flow
        self._actual_flow = self._compute_flow()

    def _update_wear(self) -> None:
        """
        Update pump-specific wear models.

        Wear affects:
        - Diaphragm/piston: Reduces stroke volume
        - Check valves: Reduces sealing efficiency (increases backflow)
        - Peristaltic tube: Reduces effective volume (pinch wear)
        """
        # Base wear from cycles and time
        super()._update_wear()

        # Diaphragm wear (based on strokes since last replacement)
        stroke_wear = self._diaphragm_strokes / 10_000_000.0  # 10M strokes = full wear
        self._diaphragm_wear = min(0.5, stroke_wear)  # Max 50% reduction

        # Check valve wear (mainly from strokes)
        self._check_valve_wear = min(0.3, self._total_strokes / 20_000_000.0)

        # Tube wear (peristaltic only, based on strokes since last replacement)
        if self._pump_type == PumpType.PERISTALTIC:
            self._tube_wear = min(
                0.8, self._tube_strokes / 5_000_000.0
            )  # Tubes wear faster

    def _check_faults(self) -> None:
        """
        Check for pump-specific fault conditions.

        Faults:
        - Check valve failure (excessive backflow)
        - Diaphragm rupture (sudden flow loss)
        - Tube rupture (peristaltic)
        - Over-pressure
        """
        super()._check_faults()

        # Check valve failure
        if self._check_valve_wear > 0.2:
            self._fault_code = ActuatorFault.LEAKING

        # Over-pressure condition
        if self._discharge_pressure_bar > self._max_pressure_bar * 0.9:
            self._fault_code = ActuatorFault.SLOW
            self._health_status = ActuatorStatus.DEGRADED

        # Catastrophic failure (very rare, high wear)
        if self._diaphragm_wear > 0.4 or self._tube_wear > 0.7:
            rng = self._get_rng()
            if rng.random() < 0.001:  # 0.1% chance per check
                self._fault_code = ActuatorFault.LEAKING
                self._health_status = ActuatorStatus.FAILED

    def set_discharge_pressure(self, pressure_bar: float) -> None:
        """
        Update system back pressure.

        Args:
            pressure_bar: Back pressure in bar

        Raises:
            ValueError: If pressure is invalid
        """
        if not isinstance(pressure_bar, (int, float)) or pressure_bar < 0:
            raise ValueError("pressure_bar must be non-negative")

        with self._state_lock:
            self._discharge_pressure_bar = pressure_bar

    def replace_diaphragm(self) -> None:
        """
        Perform diaphragm replacement (maintenance action).

        Resets diaphragm wear and the per-diaphragm stroke counter.
        The lifetime ``_total_strokes`` counter is preserved so service
        history is not lost.
        """
        with self._state_lock:
            self._diaphragm_wear = 0.0
            self._diaphragm_strokes = 0
            if self._fault_code == ActuatorFault.LEAKING:
                self._fault_code = ActuatorFault.NONE
                self._health_status = ActuatorStatus.NORMAL

    def replace_check_valves(self) -> None:
        """
        Replace check valves (maintenance action).

        Resets check valve wear and sealing efficiency.
        """
        with self._state_lock:
            self._check_valve_wear = 0.0
            if self._fault_code == ActuatorFault.LEAKING:
                self._fault_code = ActuatorFault.NONE

    def replace_tube(self) -> None:
        """
        Replace peristaltic tube (maintenance action).

        Only applicable for peristaltic pumps.  Resets tube wear and the
        per-tube stroke counter; the lifetime ``_total_strokes`` counter
        is preserved.
        """
        with self._state_lock:
            if self._pump_type != PumpType.PERISTALTIC:
                raise ValueError(
                    "Tube replacement only applicable for peristaltic pumps"
                )
            self._tube_wear = 0.0
            self._tube_strokes = 0

    def get_stroke_rate(self) -> float:
        """Get current actual stroke rate."""
        with self._state_lock:
            return self._stroke_rate

    def get_total_strokes(self) -> int:
        """Get total stroke count."""
        with self._state_lock:
            return self._total_strokes

    def get_instantaneous_flow(self) -> float:
        """
        Get instantaneous flow rate (includes pulsation).

        Returns:
            Current flow rate with pulsation effect
        """
        with self._state_lock:
            return self._actual_flow

    def get_average_flow(self) -> float:
        """
        Get average flow rate (pulsation removed).

        Returns:
            Average flow rate over stroke cycle
        """
        with self._state_lock:
            effective_stroke_volume = self._stroke_volume_mL * (
                1.0 - self._diaphragm_wear
            )
            if self._pump_type == PumpType.PERISTALTIC:
                effective_stroke_volume *= 1.0 - self._tube_wear

            base_flow = (self._stroke_rate * effective_stroke_volume) / 1000.0

            pressure_factor = 1.0 - (
                self._discharge_pressure_bar / self._max_pressure_bar
            )
            pressure_factor = max(0.0, min(1.0, pressure_factor))

            effective_check_valve_efficiency = self._check_valve_efficiency * (
                1.0 - self._check_valve_wear
            )

            return base_flow * pressure_factor * effective_check_valve_efficiency

    def get_pump_info(self) -> dict:
        """
        Get pump configuration and current state.

        Returns:
            Dictionary with pump parameters and state
        """
        with self._state_lock:
            base_info = self.get_state_dict()
            pump_info = {
                **base_info,
                "pump_type": self._pump_type.name,
                "max_flow_rate": self._max_flow_rate,
                "max_stroke_rate": self._max_stroke_rate,
                "stroke_volume_mL": self._stroke_volume_mL,
                "current_stroke_rate": self._stroke_rate,
                "total_strokes": self._total_strokes,
                "discharge_pressure_bar": self._discharge_pressure_bar,
                "diaphragm_wear": self._diaphragm_wear,
                "check_valve_wear": self._check_valve_wear,
                "instantaneous_flow": self.get_instantaneous_flow(),
                "average_flow": self.get_average_flow(),
            }

            if self._pump_type == PumpType.PERISTALTIC:
                pump_info["tube_wear"] = self._tube_wear

            return pump_info
