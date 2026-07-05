"""Numerical verification cases for bounded CFD primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import ScalarField
from .mesh import BoundaryPatch, StructuredMesh, create_rectangular_mesh
from .scalar import ScalarTransportConfig, step_scalar_transport
from .solver import FlowSolverConfig, initialize_flow, solve_flow_step


@dataclass(frozen=True)
class NumericalVerificationResult:
    case_id: str
    category: str
    metric: str
    value: float
    tolerance: float
    passed: bool
    evidence_status: str
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.case_id:
            raise ValueError("verification case_id is required")
        if not self.category:
            raise ValueError(f"{self.case_id}: category is required")
        if not self.metric:
            raise ValueError(f"{self.case_id}: metric is required")
        if self.tolerance < 0.0:
            raise ValueError(f"{self.case_id}: tolerance cannot be negative")
        if not np.isfinite(self.value):
            raise ValueError(f"{self.case_id}: value must be finite")
        if self.evidence_status != "synthetic_numerical_verification":
            raise ValueError(f"{self.case_id}: unsupported evidence status")
        if not self.limitations:
            raise ValueError(f"{self.case_id}: limitations are required")
        joined = " ".join(self.limitations)
        if "not real-plant validation" not in joined:
            raise ValueError(f"{self.case_id}: missing validation caveat")


@dataclass(frozen=True)
class NumericalVerificationSuite:
    suite_id: str
    results: tuple[NumericalVerificationResult, ...]
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.suite_id:
            raise ValueError("verification suite_id is required")
        if not self.results:
            raise ValueError(f"{self.suite_id}: results are required")
        if not self.limitations:
            raise ValueError(f"{self.suite_id}: limitations are required")
        for result in self.results:
            result.validate()

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


def _verification_limitations(*extra: str) -> tuple[str, ...]:
    return (
        "synthetic numerical verification case",
        "not real-plant validation, certification, or commissioning evidence",
        *extra,
    )


def _reference_mesh(cells: tuple[int, int, int]) -> StructuredMesh:
    return create_rectangular_mesh(
        cells=cells,
        extents=(2.0, 1.0, 1.0),
        boundaries=(
            BoundaryPatch("verification-inlet", "xmin", "inlet", 0.02),
            BoundaryPatch("verification-outlet", "xmax", "outlet", 0.0),
            BoundaryPatch("verification-floor", "zmin", "wall", 0.0),
            BoundaryPatch("verification-surface", "zmax", "symmetry", 0.0),
        ),
    )


def verify_constant_scalar_conservation() -> NumericalVerificationResult:
    mesh = _reference_mesh((6, 3, 2))
    flow = initialize_flow(mesh)
    scalar = ScalarField.uniform(mesh, name="constant_scalar", units="a.u.", value=2.0)
    _, result = step_scalar_transport(
        mesh,
        scalar,
        flow,
        ScalarTransportConfig(dt=0.01, diffusivity=0.0),
    )
    drift = abs(result.mass_after - result.mass_before)
    verification = NumericalVerificationResult(
        case_id="nv-constant-scalar-conservation",
        category="manufactured_constant_solution",
        metric="absolute_mass_drift",
        value=drift,
        tolerance=1.0e-12,
        passed=drift <= 1.0e-12,
        evidence_status="synthetic_numerical_verification",
        limitations=_verification_limitations(
            "constant scalar with zero velocity and no source terms"
        ),
    )
    verification.validate()
    return verification


def verify_flow_mass_residual_bound() -> NumericalVerificationResult:
    mesh = _reference_mesh((8, 3, 2))
    _, result = solve_flow_step(
        mesh,
        initialize_flow(mesh),
        FlowSolverConfig(dt=0.005, pressure_iterations=4, max_cfl=0.8),
    )
    tolerance = 0.2
    verification = NumericalVerificationResult(
        case_id="nv-flow-mass-residual-bound",
        category="conservation_check",
        metric="mean_absolute_divergence",
        value=result.mass_residual,
        tolerance=tolerance,
        passed=result.stable and result.mass_residual <= tolerance,
        evidence_status="synthetic_numerical_verification",
        limitations=_verification_limitations(
            "small-grid residual check, not a validated hydraulic benchmark"
        ),
    )
    verification.validate()
    return verification


def verify_mesh_refinement_sensitivity() -> NumericalVerificationResult:
    coarse = _reference_mesh((4, 2, 2))
    fine = _reference_mesh((8, 4, 2))
    coarse_scalar = ScalarField.uniform(
        coarse, name="constant_scalar", units="a.u.", value=1.5
    )
    fine_scalar = ScalarField.uniform(
        fine, name="constant_scalar", units="a.u.", value=1.5
    )
    coarse_mass = float(coarse_scalar.values.sum() * coarse.dx * coarse.dy * coarse.dz)
    fine_mass = float(fine_scalar.values.sum() * fine.dx * fine.dy * fine.dz)
    difference = abs(coarse_mass - fine_mass)
    verification = NumericalVerificationResult(
        case_id="nv-mesh-refinement-constant-mass",
        category="mesh_refinement_sensitivity",
        metric="coarse_fine_mass_difference",
        value=difference,
        tolerance=1.0e-12,
        passed=difference <= 1.0e-12,
        evidence_status="synthetic_numerical_verification",
        limitations=_verification_limitations(
            "constant field integral check across two structured grids"
        ),
    )
    verification.validate()
    return verification


def verify_boundary_condition_response() -> NumericalVerificationResult:
    mesh = _reference_mesh((6, 3, 2))
    flow, _ = solve_flow_step(
        mesh,
        initialize_flow(mesh),
        FlowSolverConfig(dt=0.005, pressure_iterations=2, max_cfl=0.8),
    )
    inlet_mean = float(np.mean(flow.u[:, :, 0]))
    expected = 0.02
    error = abs(inlet_mean - expected)
    verification = NumericalVerificationResult(
        case_id="nv-inlet-boundary-response",
        category="boundary_condition_check",
        metric="inlet_velocity_error",
        value=error,
        tolerance=1.0e-12,
        passed=error <= 1.0e-12,
        evidence_status="synthetic_numerical_verification",
        limitations=_verification_limitations(
            "checks current inlet boundary assignment only"
        ),
    )
    verification.validate()
    return verification


def verify_long_run_scalar_drift(steps: int = 25) -> NumericalVerificationResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    mesh = _reference_mesh((6, 3, 2))
    flow = initialize_flow(mesh)
    scalar = ScalarField.uniform(mesh, name="constant_scalar", units="a.u.", value=1.0)
    initial_mass = float(scalar.values.sum() * mesh.dx * mesh.dy * mesh.dz)
    for _ in range(steps):
        scalar, result = step_scalar_transport(
            mesh,
            scalar,
            flow,
            ScalarTransportConfig(dt=0.01, diffusivity=0.0),
        )
        if not result.stable:
            break
    final_mass = float(scalar.values.sum() * mesh.dx * mesh.dy * mesh.dz)
    drift = abs(final_mass - initial_mass)
    verification = NumericalVerificationResult(
        case_id="nv-long-run-constant-scalar-drift",
        category="long_run_drift",
        metric="absolute_mass_drift",
        value=drift,
        tolerance=1.0e-12,
        passed=drift <= 1.0e-12,
        evidence_status="synthetic_numerical_verification",
        limitations=_verification_limitations(
            f"{steps} constant-field steps without flow, sources, or decay"
        ),
    )
    verification.validate()
    return verification


def build_reference_numerical_verification_suite() -> NumericalVerificationSuite:
    suite = NumericalVerificationSuite(
        suite_id="hs-30a-reference-numerical-verification",
        results=(
            verify_constant_scalar_conservation(),
            verify_flow_mass_residual_bound(),
            verify_mesh_refinement_sensitivity(),
            verify_boundary_condition_response(),
            verify_long_run_scalar_drift(),
        ),
        limitations=(
            "bounded synthetic numerical verification suite",
            "not real-plant validation, certification, or commissioning evidence",
        ),
    )
    suite.validate()
    return suite
