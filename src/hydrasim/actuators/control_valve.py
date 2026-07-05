"""
Control Valve module.

Implements pneumatic and electric control valves with:
- Valve characteristic curves (linear, equal %, quick-open)
- Actuator response time (3-20 seconds typical)
- Stiction (0.5-5% deadband)
- Hysteresis (1-3% band)
- Seat leakage when "closed"
- Positioner drift
- Fail-safe positions (fail-closed, fail-open, fail-last)
"""

from .base_actuator import BaseActuator, ActuatorFault
from .valve_characteristics import ValveType, ValveCharacteristics
from enum import Enum, auto
import math


class ActuatorType(Enum):
    """Actuator mechanism type."""

    PNEUMATIC = auto()
    ELECTRIC = auto()


class FailPosition(Enum):
    """Fail-safe position on loss of control signal or power."""

    FAIL_CLOSED = auto()
    FAIL_OPEN = auto()
    FAIL_LAST = auto()


class ControlValve(BaseActuator):
    """
    Pneumatic or electric control valve (0-100% position).

    Models:
    - Valve characteristic curves (linear, equal %, quick-open)
    - Actuator response time (first-order lag)
    - Stiction (static friction deadband)
    - Hysteresis (different response opening vs closing)
    - Seat leakage when "closed"
    - Positioner drift
    - Fail-safe behavior

    Pressure-flow relationship:
    Q = Cv * sqrt(ΔP / SG)

    Thread-safe: All state access protected by lock from BaseActuator.
    """

    def __init__(
        self,
        name: str,
        valve_type: ValveType,
        cv_full_open: float,
        actuator_type: ActuatorType,
        response_time: float = 5.0,
        fail_position: FailPosition = FailPosition.FAIL_LAST,
        stiction: float = 0.02,
        hysteresis: float = 0.01,
        seat_leakage_percent: float = 0.5,
        pressure_drop_bar: float = 2.0,
        specific_gravity: float = 1.0,
        rangeability: float = 50.0,
        leakage_class: str = "Class IV",
        positioner_drift_rate: float = 0.01,  # % per hour
        max_history: int = 1000,
    ):
        """
        Initialize control valve.

        Args:
            name: Valve identifier
            valve_type: Characteristic curve (LINEAR, EQUAL_PERCENTAGE, QUICK_OPENING)
            cv_full_open: Flow coefficient at 100% open
            actuator_type: PNEUMATIC or ELECTRIC
            response_time: Time constant (seconds to 63% response)
            fail_position: Behavior on signal loss (FAIL_CLOSED, FAIL_OPEN, FAIL_LAST)
            stiction: Static friction deadband (% of span)
            hysteresis: Hysteresis band (% of span)
            seat_leakage_percent: Leakage when closed (% of max Cv)
            pressure_drop_bar: Pressure drop across valve (bar)
            specific_gravity: Fluid specific gravity (water = 1.0)
            rangeability: Cv_max/Cv_min for equal percentage valve
            leakage_class: ANSI/FCI leakage class
            positioner_drift_rate: Position drift per hour (%)
            max_history: Maximum history buffer size

        Raises:
            TypeError: If arguments are not expected types
            ValueError: If arguments are out of valid ranges
        """
        # Validate valve-specific parameters
        if not isinstance(valve_type, ValveType):
            raise TypeError("valve_type must be ValveType enum")
        if not isinstance(actuator_type, ActuatorType):
            raise TypeError("actuator_type must be ActuatorType enum")
        if not isinstance(fail_position, FailPosition):
            raise TypeError("fail_position must be FailPosition enum")
        if not isinstance(cv_full_open, (int, float)) or cv_full_open <= 0:
            raise ValueError("cv_full_open must be positive")
        if not isinstance(pressure_drop_bar, (int, float)) or pressure_drop_bar < 0:
            raise ValueError("pressure_drop_bar must be non-negative")
        if not isinstance(specific_gravity, (int, float)) or specific_gravity <= 0:
            raise ValueError("specific_gravity must be positive")
        if not isinstance(rangeability, (int, float)) or rangeability < 1.0:
            raise ValueError("rangeability must be >= 1")
        if (
            not isinstance(seat_leakage_percent, (int, float))
            or not 0 <= seat_leakage_percent <= 100
        ):
            raise ValueError("seat_leakage_percent must be in [0, 100]")
        if (
            not isinstance(positioner_drift_rate, (int, float))
            or positioner_drift_rate < 0
        ):
            raise ValueError("positioner_drift_rate must be non-negative")

        # Initialize base actuator
        super().__init__(
            name=name,
            response_time=response_time,
            deadband=0.005,  # Small deadband for valves
            hysteresis=hysteresis,
            stiction=stiction,
            max_history=max_history,
        )

        # Valve-specific parameters
        self._valve_type = valve_type
        self._cv_full_open = cv_full_open
        self._actuator_type = actuator_type
        self._fail_position = fail_position
        self._seat_leakage_percent = seat_leakage_percent
        self._pressure_drop_bar = pressure_drop_bar
        self._specific_gravity = specific_gravity
        self._rangeability = rangeability
        self._leakage_class = leakage_class
        self._positioner_drift_rate = positioner_drift_rate

        # Positioner drift accumulator
        self._accumulated_drift = 0.0

        # Air supply variations (for pneumatic actuators)
        self._air_supply_pressure = 6.0  # bar (typical instrument air)
        self._air_supply_nominal = 6.0

    def _validate_setpoint(self, setpoint: float) -> None:
        """
        Validate setpoint range for control valve (0-100%).

        Args:
            setpoint: Position setpoint

        Raises:
            ValueError: If setpoint is out of range
        """
        if not 0.0 <= setpoint <= 100.0:
            raise ValueError("setpoint must be in [0, 100]%")

    def _compute_flow(self) -> float:
        """
        Compute actual flow rate based on valve position.

        Steps:
        1. Get Cv for current position using characteristic curve
        2. Add seat leakage if valve is nearly closed
        3. Apply positioner drift
        4. Calculate flow from Cv and pressure drop

        Returns:
            Flow rate in L/min
        """
        # Apply positioner drift to actual position
        effective_position = self._actual_position + self._accumulated_drift
        effective_position = max(0.0, min(100.0, effective_position))

        # Get Cv from characteristic curve
        cv = ValveCharacteristics.get_cv_for_position(
            self._valve_type, effective_position, self._cv_full_open, self._rangeability
        )

        # Add seat leakage when valve is nearly closed
        if effective_position < 1.0:
            leakage_cv = ValveCharacteristics.compute_seat_leakage(
                effective_position, self._cv_full_open, self._leakage_class
            )
            cv += leakage_cv

        # Adjust for air supply variations (pneumatic actuators only)
        if self._actuator_type == ActuatorType.PNEUMATIC:
            air_supply_factor = self._air_supply_pressure / self._air_supply_nominal
            cv *= air_supply_factor

        # Calculate flow rate from Cv and pressure drop
        flow_rate = ValveCharacteristics.compute_flow_rate(
            cv,
            self._pressure_drop_bar,
            self._specific_gravity,
            use_metric=True,  # L/min
        )

        return flow_rate

    def _update_position_dynamics(self, dt: float) -> None:
        """
        Update valve position with positioner drift.

        Extends base class to add positioner drift accumulation.

        Args:
            dt: Time step in seconds
        """
        # Call base class dynamics
        super()._update_position_dynamics(dt)

        # Accumulate positioner drift (increases with time and wear)
        drift_per_second = (self._positioner_drift_rate / 3600.0) * (
            1.0 + self._wear_factor
        )
        rng = self._get_rng()
        self._accumulated_drift += rng.uniform(-drift_per_second, drift_per_second) * dt

        # Limit drift accumulation
        self._accumulated_drift = max(-5.0, min(5.0, self._accumulated_drift))

    def _check_faults(self) -> None:
        """
        Check for valve-specific fault conditions.

        Extends base class to add valve-specific faults:
        - Air supply issues (pneumatic)
        - Position drift exceeding limits
        - Stuck valve detection
        """
        # Call base class fault checking
        super()._check_faults()

        # Check for excessive drift
        if abs(self._accumulated_drift) > 3.0:
            self._fault_code = ActuatorFault.CALIBRATION
            self._health_status = self._health_status  # Keep existing status

        # Check for stuck valve (commanded != actual for extended time)
        position_error = abs(self._commanded_position - self._actual_position)
        if position_error > 10.0 and self._hours_runtime > 1.0:
            self._fault_code = ActuatorFault.STUCK

        # Check air supply (pneumatic only)
        if self._actuator_type == ActuatorType.PNEUMATIC:
            if self._air_supply_pressure < 4.0:
                self._fault_code = ActuatorFault.SLOW

    def simulate_air_supply_variation(self, pressure_bar: float) -> None:
        """
        Simulate air supply pressure variations (pneumatic actuators only).

        Args:
            pressure_bar: Air supply pressure in bar (typical 4-7 bar)

        Raises:
            ValueError: If pressure is out of valid range
        """
        if not isinstance(pressure_bar, (int, float)):
            raise TypeError("pressure_bar must be numeric")
        if not 0 <= pressure_bar <= 10:
            raise ValueError("pressure_bar must be in [0, 10]")

        with self._state_lock:
            self._air_supply_pressure = pressure_bar

    def calibrate_zero(self) -> None:
        """
        Perform zero/span calibration.

        Extends base class to also reset drift accumulator.
        """
        super().calibrate_zero()
        with self._state_lock:
            self._accumulated_drift = 0.0

    def recalibrate_positioner(self) -> None:
        """
        Recalibrate the positioner to eliminate accumulated drift.

        Called by MaintenanceManager for the RECALIBRATE_POSITIONER action.
        Unlike a full calibrate_zero() this does not move the valve to 0% —
        it only resets the drift accumulator and clears the CALIBRATION fault,
        leaving the valve at its current commanded position.  This matches
        real field practice where positioner recalibration is performed
        in-service without interrupting flow.
        """
        with self._state_lock:
            self._accumulated_drift = 0.0
            if self._fault_code == ActuatorFault.CALIBRATION:
                self._fault_code = ActuatorFault.NONE

    def set_pressure_drop(self, pressure_drop_bar: float) -> None:
        """
        Update pressure drop across valve (affects flow calculation).

        Args:
            pressure_drop_bar: New pressure drop in bar

        Raises:
            ValueError: If pressure drop is invalid
        """
        if not isinstance(pressure_drop_bar, (int, float)) or pressure_drop_bar < 0:
            raise ValueError("pressure_drop_bar must be non-negative")

        with self._state_lock:
            self._pressure_drop_bar = pressure_drop_bar

    def max_flow_rate(self) -> float:
        """
        Calculate valve flow at 100% opening under current process conditions.

        Returns:
            Maximum achievable flow in L/min
        """
        with self._state_lock:
            return ValveCharacteristics.compute_flow_rate(
                self._cv_full_open,
                self._pressure_drop_bar,
                self._specific_gravity,
                use_metric=True,
            )

    def set_flow_rate(self, flow_rate: float) -> None:
        """
        Command valve to achieve a target flow rate.

        The method converts flow target -> Cv target -> valve position
        using the configured valve characteristic curve.

        Args:
            flow_rate: Desired flow rate [L/min]
        """
        if not isinstance(flow_rate, (int, float)):
            raise TypeError("flow_rate must be numeric")
        if flow_rate < 0:
            raise ValueError("flow_rate must be non-negative")

        with self._state_lock:
            if self._pressure_drop_bar <= 0 or self._specific_gravity <= 0:
                target_position = 0.0
            else:
                denominator = 1.67 * math.sqrt(
                    self._pressure_drop_bar / self._specific_gravity
                )
                cv_target = flow_rate / denominator if denominator > 0 else 0.0
                cv_target = max(0.0, min(cv_target, self._cv_full_open))

                if self._cv_full_open <= 0:
                    target_position = 0.0
                elif self._valve_type == ValveType.LINEAR:
                    target_position = 100.0 * (cv_target / self._cv_full_open)
                elif self._valve_type == ValveType.QUICK_OPENING:
                    target_position = 100.0 * (cv_target / self._cv_full_open) ** 2
                elif self._valve_type == ValveType.EQUAL_PERCENTAGE:
                    if cv_target <= 0.0:
                        target_position = 0.0
                    else:
                        if self._rangeability <= 1.0:
                            target_position = 100.0 * (cv_target / self._cv_full_open)
                        else:
                            min_ratio = 1.0 / self._rangeability
                            ratio = max(min_ratio, cv_target / self._cv_full_open)
                            target_position = 100.0 * (
                                1.0 + (math.log(ratio) / math.log(self._rangeability))
                            )
                else:
                    target_position = 0.0

            target_position = max(0.0, min(100.0, target_position))

        self.set_position(target_position)

    def get_cv(self) -> float:
        """
        Get current flow coefficient (Cv) at actual position.

        Returns:
            Current Cv value
        """
        with self._state_lock:
            effective_position = self._actual_position + self._accumulated_drift
            effective_position = max(0.0, min(100.0, effective_position))

            return ValveCharacteristics.get_cv_for_position(
                self._valve_type,
                effective_position,
                self._cv_full_open,
                self._rangeability,
            )

    def apply_fail_safe(self) -> None:
        """
        Move valve to fail-safe position (simulates signal loss).

        This method simulates loss of control signal or power.
        The valve moves to its configured fail-safe position.
        """
        with self._state_lock:
            if self._fail_position == FailPosition.FAIL_CLOSED:
                self._commanded_position = 0.0
                self._actual_position = 0.0
            elif self._fail_position == FailPosition.FAIL_OPEN:
                self._commanded_position = 100.0
                self._actual_position = 100.0
            # FAIL_LAST: keep current position

            self._actual_flow = self._compute_flow()

    def get_valve_info(self) -> dict:
        """
        Get valve configuration and current state.

        Returns:
            Dictionary with valve parameters and state
        """
        with self._state_lock:
            base_info = self.get_state_dict()
            valve_info = {
                **base_info,
                "valve_type": self._valve_type.name,
                "actuator_type": self._actuator_type.name,
                "fail_position": self._fail_position.name,
                "cv_full_open": self._cv_full_open,
                "current_cv": self.get_cv(),
                "pressure_drop_bar": self._pressure_drop_bar,
                "accumulated_drift": self._accumulated_drift,
            }

            if self._actuator_type == ActuatorType.PNEUMATIC:
                valve_info["air_supply_pressure"] = self._air_supply_pressure

            return valve_info
