"""Bounded incompressible-flow stepping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import FlowField
from .mesh import StructuredMesh
from .numerics import divergence, gradient, laplacian


@dataclass(frozen=True)
class FlowSolverConfig:
    """Numerical controls for the small-grid flow solver."""

    dt: float = 0.05
    density: float = 997.0
    viscosity: float = 8.9e-4
    inlet_velocity: float = 0.04
    pressure_iterations: int = 20
    relaxation: float = 0.7
    max_cfl: float = 0.8

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        if self.pressure_iterations < 1:
            raise ValueError("pressure_iterations must be positive")
        if not (0.0 < self.relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1]")


@dataclass(frozen=True)
class FlowStepResult:
    """Flow-step diagnostics used by tests and lab bundles."""

    mass_residual: float
    max_cfl: float
    stable: bool
    active_cells: int


def initialize_flow(mesh: StructuredMesh) -> FlowField:
    return FlowField.zeros(mesh)


def _apply_boundaries(
    mesh: StructuredMesh, u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> None:
    active = mesh.active_mask()
    u[~active] = 0.0
    v[~active] = 0.0
    w[~active] = 0.0
    for boundary in mesh.boundaries:
        if boundary.face == "xmin" and boundary.kind == "inlet":
            u[:, :, 0] = boundary.value
        elif boundary.face == "xmax" and boundary.kind == "outlet" and mesh.nx > 1:
            u[:, :, -1] = u[:, :, -2]
        elif boundary.kind == "wall":
            if boundary.face == "xmin":
                u[:, :, 0] = 0.0
            elif boundary.face == "xmax":
                u[:, :, -1] = 0.0
            elif boundary.face == "ymin":
                v[:, 0, :] = 0.0
            elif boundary.face == "ymax":
                v[:, -1, :] = 0.0
            elif boundary.face == "zmin":
                w[0, :, :] = 0.0
            elif boundary.face == "zmax":
                w[-1, :, :] = 0.0
    u[~active] = 0.0
    v[~active] = 0.0
    w[~active] = 0.0


def solve_flow_step(
    mesh: StructuredMesh,
    field: FlowField,
    config: FlowSolverConfig | None = None,
) -> tuple[FlowField, FlowStepResult]:
    """Advance a small-grid incompressible flow approximation by one step."""

    config = config or FlowSolverConfig()
    config.validate()
    field.validate(mesh)

    nu = config.viscosity / config.density
    u = field.u + (nu * config.dt * laplacian(field.u, mesh.dx, mesh.dy, mesh.dz))
    v = field.v + (nu * config.dt * laplacian(field.v, mesh.dx, mesh.dy, mesh.dz))
    w = field.w + (nu * config.dt * laplacian(field.w, mesh.dx, mesh.dy, mesh.dz))
    pressure = np.array(field.pressure, copy=True)

    if not any(b.kind == "inlet" for b in mesh.boundaries):
        u[:, :, 0] = config.inlet_velocity
    _apply_boundaries(mesh, u, v, w)

    for _ in range(config.pressure_iterations):
        div = divergence(u, v, w, mesh.dx, mesh.dy, mesh.dz)
        pressure -= config.relaxation * config.density * config.dt * div
        dpx, dpy, dpz = gradient(pressure, mesh.dx, mesh.dy, mesh.dz)
        u -= (config.dt / config.density) * dpx
        v -= (config.dt / config.density) * dpy
        w -= (config.dt / config.density) * dpz
        _apply_boundaries(mesh, u, v, w)

    final_div = divergence(u, v, w, mesh.dx, mesh.dy, mesh.dz)
    active = mesh.active_mask()
    max_velocity = float(np.max(np.sqrt((u * u) + (v * v) + (w * w))))
    min_step = min(mesh.dx, mesh.dy, mesh.dz)
    max_cfl = max_velocity * config.dt / min_step
    mass_residual = float(np.mean(np.abs(final_div[active]))) if active.any() else 0.0
    stable = bool(np.isfinite(max_cfl) and max_cfl <= config.max_cfl)

    return (
        FlowField(u=u, v=v, w=w, pressure=pressure),
        FlowStepResult(
            mass_residual=mass_residual,
            max_cfl=float(max_cfl),
            stable=stable,
            active_cells=int(active.sum()),
        ),
    )
