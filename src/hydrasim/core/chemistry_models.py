"""Aqueous chemistry data models."""

from dataclasses import dataclass
import warnings


@dataclass
class BufferSystem:
    """
    Parameters for a buffer system in water.

    Attributes:
        alkalinity: Total alkalinity [mg/L as CaCO₃]
        total_carbonate: Total carbonate species [mmol/L]
        temperature: Operating temperature [°C]
    """

    alkalinity: float  # [mg/L as CaCO₃]
    total_carbonate: float  # [mmol/L]
    temperature: float = 20.0  # [°C]

    def validate(self) -> None:
        """Validate buffer system parameters."""
        if self.alkalinity < 0:
            raise ValueError(f"Alkalinity cannot be negative: {self.alkalinity}")
        if self.total_carbonate < 0:
            raise ValueError(
                f"Total carbonate cannot be negative: {self.total_carbonate}"
            )
        if self.temperature < 0 or self.temperature > 40:
            warnings.warn(
                f"Temperature {self.temperature}°C outside typical range [0, 40]"
            )
