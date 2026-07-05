"""
Valve characteristics module.

Implements flow coefficient (Cv) curves for different valve types:
- Linear: Cv proportional to position
- Equal Percentage: Logarithmic curve, common for process control
- Quick Opening: Rapid increase at low position, good for on/off service

Flow calculation:
Q = Cv * sqrt(ΔP / SG)

Where:
- Q: Flow rate (GPM or L/min)
- Cv: Flow coefficient
- ΔP: Pressure drop across valve
- SG: Specific gravity of fluid
"""

from enum import Enum, auto
import math


class ValveType(Enum):
    """Valve characteristic curve types."""

    LINEAR = auto()
    EQUAL_PERCENTAGE = auto()
    QUICK_OPENING = auto()


class ValveCharacteristics:
    """
    Calculate flow coefficients (Cv) based on valve position and type.

    Zero-trust principles:
    - All inputs validated
    - No side effects
    - Pure functions where possible
    """

    @staticmethod
    def linear(position_percent: float, cv_max: float) -> float:
        """
        Linear characteristic: Cv increases linearly with position.

        Cv = cv_max * (position / 100)

        Best for: Applications where flow should be proportional to valve position.

        Args:
            position_percent: Valve position (0-100%)
            cv_max: Maximum Cv at 100% open

        Returns:
            Flow coefficient Cv

        Raises:
            TypeError: If arguments are not numeric
            ValueError: If arguments are out of valid ranges
        """
        if not isinstance(position_percent, (int, float)):
            raise TypeError("position_percent must be numeric")
        if not isinstance(cv_max, (int, float)):
            raise TypeError("cv_max must be numeric")

        if not 0.0 <= position_percent <= 100.0:
            raise ValueError("position_percent must be in [0, 100]")
        if cv_max < 0:
            raise ValueError("cv_max must be non-negative")

        return cv_max * (position_percent / 100.0)

    @staticmethod
    def equal_percentage(
        position_percent: float, cv_max: float, rangeability: float = 50.0
    ) -> float:
        """
        Equal percentage characteristic: Each increment in position produces
        an equal percentage change in Cv.

        Cv = cv_max * R^((position/100) - 1)

        Where R = rangeability (typical 20-50)

        Best for: Process control where valve authority varies with position.
        Most common in liquid level and pressure control.

        Args:
            position_percent: Valve position (0-100%)
            cv_max: Maximum Cv at 100% open
            rangeability: Cv_max / Cv_min ratio (typically 20-50)

        Returns:
            Flow coefficient Cv

        Raises:
            TypeError: If arguments are not numeric
            ValueError: If arguments are out of valid ranges
        """
        if not isinstance(position_percent, (int, float)):
            raise TypeError("position_percent must be numeric")
        if not isinstance(cv_max, (int, float)):
            raise TypeError("cv_max must be numeric")
        if not isinstance(rangeability, (int, float)):
            raise TypeError("rangeability must be numeric")

        if not 0.0 <= position_percent <= 100.0:
            raise ValueError("position_percent must be in [0, 100]")
        if cv_max < 0:
            raise ValueError("cv_max must be non-negative")
        if rangeability < 1.0:
            raise ValueError("rangeability must be >= 1")

        # Equal percentage formula
        exponent = (position_percent / 100.0) - 1.0
        cv = cv_max * (rangeability**exponent)

        return cv

    @staticmethod
    def quick_opening(position_percent: float, cv_max: float) -> float:
        """
        Quick opening characteristic: Cv increases rapidly at low position.

        Cv = cv_max * sqrt(position / 100)

        Best for: On/off service, safety relief, where maximum flow is needed
        quickly. Not recommended for throttling control.

        Args:
            position_percent: Valve position (0-100%)
            cv_max: Maximum Cv at 100% open

        Returns:
            Flow coefficient Cv

        Raises:
            TypeError: If arguments are not numeric
            ValueError: If arguments are out of valid ranges
        """
        if not isinstance(position_percent, (int, float)):
            raise TypeError("position_percent must be numeric")
        if not isinstance(cv_max, (int, float)):
            raise TypeError("cv_max must be numeric")

        if not 0.0 <= position_percent <= 100.0:
            raise ValueError("position_percent must be in [0, 100]")
        if cv_max < 0:
            raise ValueError("cv_max must be non-negative")

        return cv_max * math.sqrt(position_percent / 100.0)

    @staticmethod
    def compute_flow_rate(
        cv: float,
        pressure_drop_bar: float,
        specific_gravity: float = 1.0,
        use_metric: bool = True,
    ) -> float:
        """
        Calculate flow rate from Cv and pressure drop.

        Metric (L/min):  Q = 1.67 * Cv * sqrt(ΔP / SG)
        Imperial (GPM):  Q = Cv * sqrt(ΔP_psi / SG)

        Args:
            cv: Flow coefficient
            pressure_drop_bar: Pressure drop across valve (bar for metric, psi for imperial)
            specific_gravity: Fluid specific gravity (water = 1.0)
            use_metric: If True, return L/min; if False, return GPM

        Returns:
            Flow rate (L/min or GPM depending on use_metric)

        Raises:
            TypeError: If arguments are not numeric
            ValueError: If arguments are out of valid ranges
        """
        if not isinstance(cv, (int, float)):
            raise TypeError("cv must be numeric")
        if not isinstance(pressure_drop_bar, (int, float)):
            raise TypeError("pressure_drop_bar must be numeric")
        if not isinstance(specific_gravity, (int, float)):
            raise TypeError("specific_gravity must be numeric")

        if cv < 0:
            raise ValueError("cv must be non-negative")
        if pressure_drop_bar < 0:
            raise ValueError("pressure_drop_bar must be non-negative")
        if specific_gravity <= 0:
            raise ValueError("specific_gravity must be positive")

        # Avoid division by zero
        if specific_gravity == 0:
            raise ValueError("specific_gravity cannot be zero")

        if use_metric:
            # Metric formula: Q (L/min) = 1.67 * Cv * sqrt(ΔP / SG)
            flow_rate = 1.67 * cv * math.sqrt(pressure_drop_bar / specific_gravity)
        else:
            # Imperial formula: Q (GPM) = Cv * sqrt(ΔP_psi / SG)
            flow_rate = cv * math.sqrt(pressure_drop_bar / specific_gravity)

        return flow_rate

    @staticmethod
    def get_cv_for_position(
        valve_type: ValveType,
        position_percent: float,
        cv_max: float,
        rangeability: float = 50.0,
    ) -> float:
        """
        Get Cv for a given valve type and position.

        Convenience method that dispatches to the appropriate characteristic function.

        Args:
            valve_type: Type of valve characteristic
            position_percent: Valve position (0-100%)
            cv_max: Maximum Cv at 100% open
            rangeability: Rangeability for equal percentage (ignored for others)

        Returns:
            Flow coefficient Cv

        Raises:
            TypeError: If arguments are invalid types
            ValueError: If arguments are out of valid ranges
        """
        if not isinstance(valve_type, ValveType):
            raise TypeError("valve_type must be ValveType enum")

        if valve_type == ValveType.LINEAR:
            return ValveCharacteristics.linear(position_percent, cv_max)
        elif valve_type == ValveType.EQUAL_PERCENTAGE:
            return ValveCharacteristics.equal_percentage(
                position_percent, cv_max, rangeability
            )
        elif valve_type == ValveType.QUICK_OPENING:
            return ValveCharacteristics.quick_opening(position_percent, cv_max)
        else:
            raise ValueError(f"Unknown valve type: {valve_type}")

    @staticmethod
    def compute_seat_leakage(
        position_percent: float, cv_max: float, leakage_class: str = "Class IV"
    ) -> float:
        """
        Compute seat leakage when valve is "closed".

        ANSI/FCI 70-2 Leakage Classes:
        - Class II: 0.5% of Cv max
        - Class III: 0.1% of Cv max
        - Class IV: 0.01% of Cv max
        - Class V: 0.0005% of Cv max
        - Class VI: Gas tight (bubble test)

        Args:
            position_percent: Valve position (0-100%)
            cv_max: Maximum Cv at 100% open
            leakage_class: ANSI/FCI leakage class

        Returns:
            Leakage Cv when position < 1%

        Raises:
            ValueError: If leakage_class is unknown
        """
        # Only apply leakage when valve is nearly closed
        if position_percent > 1.0:
            return 0.0

        leakage_percentages = {
            "Class II": 0.005,
            "Class III": 0.001,
            "Class IV": 0.0001,
            "Class V": 0.000005,
            "Class VI": 0.0,
        }

        if leakage_class not in leakage_percentages:
            raise ValueError(f"Unknown leakage_class: {leakage_class}")

        return cv_max * leakage_percentages[leakage_class]


def generate_characteristic_curve_data(
    valve_type: ValveType,
    cv_max: float,
    num_points: int = 101,
    rangeability: float = 50.0,
) -> list[tuple[float, float]]:
    """
    Generate (position, Cv) data points for plotting valve characteristics.

    Args:
        valve_type: Type of valve characteristic
        cv_max: Maximum Cv at 100% open
        num_points: Number of data points to generate
        rangeability: Rangeability for equal percentage

    Returns:
        List of (position_percent, cv) tuples
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    data = []
    for i in range(num_points):
        position = (i / (num_points - 1)) * 100.0
        cv = ValveCharacteristics.get_cv_for_position(
            valve_type, position, cv_max, rangeability
        )
        data.append((position, cv))

    return data
