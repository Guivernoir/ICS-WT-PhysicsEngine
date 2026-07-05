"""Finite-volume scalar transport for water-treatment species."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import FlowField, ScalarField
from .mesh import StructuredMesh
from .numerics import gradient, laplacian


@dataclass(frozen=True)
class ScalarTransportConfig:
    dt: float = 0.05
    diffusivity: float = 1.0e-5
    first_order_decay: float = 0.0
    max_cfl: float = 0.8

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.diffusivity < 0.0:
            raise ValueError("diffusivity cannot be negative")
        if self.first_order_decay < 0.0:
            raise ValueError("first_order_decay cannot be negative")


@dataclass(frozen=True)
class ScalarTransportResult:
    mass_before: float
    mass_after: float
    min_value: float
    max_value: float
    stable: bool


def step_scalar_transport(
    mesh: StructuredMesh,
    scalar: ScalarField,
    flow: FlowField,
    config: ScalarTransportConfig | None = None,
    *,
    source_cells: tuple[tuple[int, int, int, float], ...] = (),
) -> tuple[ScalarField, ScalarTransportResult]:
    """Advance one scalar with advection, diffusion, decay, and fixed sources."""

    config = config or ScalarTransportConfig()
    config.validate()
    scalar.validate(mesh)
    flow.validate(mesh)

    values = scalar.values
    mass_before = float(values.sum() * mesh.dx * mesh.dy * mesh.dz)
    dcdx, dcdy, dcdz = gradient(values, mesh.dx, mesh.dy, mesh.dz)
    advection = -((flow.u * dcdx) + (flow.v * dcdy) + (flow.w * dcdz))
    diffusion = config.diffusivity * laplacian(values, mesh.dx, mesh.dy, mesh.dz)
    reaction = -config.first_order_decay * values
    next_values = values + config.dt * (advection + diffusion + reaction)

    for i, j, k, strength in source_cells:
        mesh.cell_id(i, j, k)
        next_values[k, j, i] += config.dt * strength

    if scalar.nonnegative:
        next_values = np.maximum(next_values, 0.0)

    active = mesh.active_mask()
    next_values[~active] = 0.0
    mass_after = float(next_values.sum() * mesh.dx * mesh.dy * mesh.dz)
    max_velocity = float(np.max(np.abs(flow.u) + np.abs(flow.v) + np.abs(flow.w)))
    min_step = min(mesh.dx, mesh.dy, mesh.dz)
    cfl = max_velocity * config.dt / min_step if min_step > 0.0 else float("inf")
    stable = bool(np.isfinite(cfl) and cfl <= config.max_cfl)

    result = ScalarTransportResult(
        mass_before=mass_before,
        mass_after=mass_after,
        min_value=float(next_values.min()),
        max_value=float(next_values.max()),
        stable=stable,
    )
    return (
        ScalarField(
            name=scalar.name,
            units=scalar.units,
            values=next_values,
            nonnegative=scalar.nonnegative,
        ),
        result,
    )
