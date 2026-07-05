"""Bounded turbulence/mixing helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixingModelConfig:
    """A small-grid eddy-diffusivity approximation.

    This is a bounded mixing model for local simulation. It is not a calibrated
    turbulence closure for plant design.
    """

    turbulent_viscosity: float = 1.0e-5
    mixing_length: float = 0.05
    intensity: float = 0.05

    def validate(self) -> None:
        if self.turbulent_viscosity < 0.0:
            raise ValueError("turbulent_viscosity cannot be negative")
        if self.mixing_length < 0.0:
            raise ValueError("mixing_length cannot be negative")
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError("intensity must be between 0 and 1")


def effective_diffusivity(base_diffusivity: float, config: MixingModelConfig) -> float:
    config.validate()
    if base_diffusivity < 0.0:
        raise ValueError("base_diffusivity cannot be negative")
    mixing_term = config.turbulent_viscosity * (1.0 + config.intensity)
    return base_diffusivity + mixing_term
