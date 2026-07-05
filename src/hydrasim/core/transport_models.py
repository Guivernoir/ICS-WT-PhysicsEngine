"""Transport parameter models."""

import numpy as np
from dataclasses import dataclass


@dataclass
class GeometryParameters:
    """
    Physical geometry of the reactor tank.

    Attributes:
        volume: Total tank volume [L]
        height: Tank height [m]
        diameter: Tank diameter [m]
        n_zones: Number of vertical zones for discretization
    """

    volume: float  # [L]
    height: float  # [m]
    diameter: float  # [m]
    n_zones: int = 5

    def validate(self) -> None:
        """Validate geometric consistency."""
        # Check volume consistency with geometry
        calculated_volume = (
            np.pi * (self.diameter / 2) ** 2 * self.height * 1000
        )  # Convert m³ to L

        volume_error = abs(calculated_volume - self.volume) / self.volume
        if volume_error > 0.1:  # Allow 10% tolerance
            raise ValueError(
                f"Volume inconsistency: specified {self.volume}L, "
                f"calculated {calculated_volume:.1f}L from geometry"
            )

        if self.n_zones < 2:
            raise ValueError(f"Need at least 2 zones, got {self.n_zones}")

    @property
    def zone_height(self) -> float:
        """Height of each zone [m]."""
        return self.height / self.n_zones

    @property
    def zone_volume(self) -> float:
        """Volume of each zone [L]."""
        return self.volume / self.n_zones

    @property
    def cross_sectional_area(self) -> float:
        """Cross-sectional area [m²]."""
        return np.pi * (self.diameter / 2) ** 2


@dataclass
class FlowParameters:
    """
    Flow characteristics of the reactor.

    Attributes:
        flow_rate: Volumetric flow rate [L/min]
        turbulent_intensity: Turbulence intensity (0-1)
        recirculation_ratio: Internal recirculation / inlet flow
        impeller_speed: Impeller rotation speed [rpm]
        impeller_diameter: Impeller diameter [m]
        power_number: Impeller power number (dimensionless, ~5 for typical Rushton turbine)
    """

    flow_rate: float  # [L/min]
    turbulent_intensity: float = 0.15  # Typical for stirred tanks
    recirculation_ratio: float = 5.0  # High internal mixing
    impeller_speed: float = 60.0  # [rpm] Conservative for water treatment
    impeller_diameter: float = 0.3  # [m] ~1/3 of tank diameter
    power_number: float = 5.0  # Rushton turbine standard value

    def validate(self) -> None:
        """Validate flow parameters."""
        if self.flow_rate < 0:
            raise ValueError(f"Flow rate cannot be negative: {self.flow_rate}")
        if not 0 <= self.turbulent_intensity <= 1:
            raise ValueError(
                f"Turbulent intensity must be in [0,1]: {self.turbulent_intensity}"
            )
        if self.recirculation_ratio < 0:
            raise ValueError(
                f"Recirculation ratio cannot be negative: {self.recirculation_ratio}"
            )
        if self.impeller_speed < 0:
            raise ValueError(
                f"Impeller speed cannot be negative: {self.impeller_speed}"
            )
        if self.impeller_diameter <= 0:
            raise ValueError(
                f"Impeller diameter must be positive: {self.impeller_diameter}"
            )
