"""CFD field containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import StructuredMesh


@dataclass
class FlowField:
    """Cell-centered velocity and pressure fields."""

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    pressure: np.ndarray

    @classmethod
    def zeros(cls, mesh: StructuredMesh) -> "FlowField":
        shape = mesh.shape
        return cls(
            u=np.zeros(shape, dtype=np.float64),
            v=np.zeros(shape, dtype=np.float64),
            w=np.zeros(shape, dtype=np.float64),
            pressure=np.zeros(shape, dtype=np.float64),
        )

    def validate(self, mesh: StructuredMesh) -> None:
        for name, values in (
            ("u", self.u),
            ("v", self.v),
            ("w", self.w),
            ("pressure", self.pressure),
        ):
            if values.shape != mesh.shape:
                raise ValueError(f"{name} shape {values.shape} != {mesh.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass
class ScalarField:
    """Cell-centered scalar concentration or process field."""

    name: str
    units: str
    values: np.ndarray
    nonnegative: bool = True

    @classmethod
    def uniform(
        cls, mesh: StructuredMesh, *, name: str, units: str, value: float
    ) -> "ScalarField":
        return cls(
            name=name,
            units=units,
            values=np.full(mesh.shape, value, dtype=np.float64),
        )

    def validate(self, mesh: StructuredMesh) -> None:
        if not self.name:
            raise ValueError("scalar name is required")
        if self.values.shape != mesh.shape:
            raise ValueError(f"{self.name} shape {self.values.shape} != {mesh.shape}")
        if not np.isfinite(self.values).all():
            raise ValueError(f"{self.name} contains non-finite values")
        if self.nonnegative and np.any(self.values < -1e-12):
            raise ValueError(f"{self.name} contains negative concentrations")
