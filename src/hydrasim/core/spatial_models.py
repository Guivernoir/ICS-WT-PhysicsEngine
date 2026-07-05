"""Spatial model parameter types."""

from dataclasses import dataclass


@dataclass
class StratificationParameters:
    """
    Parameters controlling stratification behavior.

    Attributes:
        enable_thermal_stratification: Include temperature effects
        enable_density_stratification: Include dissolved species effects
        critical_richardson: Ri above which stratification is stable
        mixing_suppression_factor: Reduction in K_exchange when stratified
    """

    enable_thermal_stratification: bool = True
    enable_density_stratification: bool = True
    critical_richardson: float = 0.25  # Typical value
    mixing_suppression_factor: float = 0.5  # 50% reduction when stratified
