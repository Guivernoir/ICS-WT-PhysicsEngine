"""CFD performance envelope and runtime gate helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from .fields import ScalarField
from .mesh import BoundaryPatch, StructuredMesh, create_rectangular_mesh
from .scalar import ScalarTransportConfig, step_scalar_transport
from .solver import FlowSolverConfig, initialize_flow, solve_flow_step

BYTES_PER_FLOAT64 = 8
FLOW_ARRAY_COUNT = 4


@dataclass(frozen=True)
class CfdPerformancePreset:
    preset_id: str
    cells: tuple[int, int, int]
    extents: tuple[float, float, float]
    solver_dt: float
    scalar_count: int
    intended_use: str

    def validate(self) -> None:
        if not self.preset_id:
            raise ValueError("preset_id is required")
        if min(self.cells) <= 0:
            raise ValueError(f"{self.preset_id}: cells must be positive")
        if min(self.extents) <= 0.0:
            raise ValueError(f"{self.preset_id}: extents must be positive")
        if self.solver_dt <= 0.0:
            raise ValueError(f"{self.preset_id}: solver_dt must be positive")
        if self.scalar_count < 1:
            raise ValueError(f"{self.preset_id}: scalar_count must be positive")
        if not self.intended_use:
            raise ValueError(f"{self.preset_id}: intended_use is required")


@dataclass(frozen=True)
class CfdBenchmarkResult:
    preset_id: str
    cell_count: int
    estimated_field_memory_bytes: int
    iterations: int
    wall_time_seconds: float
    stable: bool
    max_cfl: float
    mass_residual: float

    def validate(self) -> None:
        if self.cell_count <= 0:
            raise ValueError("cell_count must be positive")
        if self.estimated_field_memory_bytes <= 0:
            raise ValueError("estimated memory must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.wall_time_seconds < 0.0:
            raise ValueError("wall time cannot be negative")


@dataclass(frozen=True)
class CfdRuntimePerformanceBudget:
    preset_id: str
    max_wall_time_seconds: float
    max_field_memory_bytes: int
    max_output_size_bytes: int
    max_mass_residual: float
    max_cfl: float
    long_run_steps: int
    evidence_status: str
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.preset_id:
            raise ValueError("budget preset_id is required")
        if self.max_wall_time_seconds <= 0.0:
            raise ValueError(f"{self.preset_id}: wall-time budget must be positive")
        if self.max_field_memory_bytes <= 0:
            raise ValueError(f"{self.preset_id}: memory budget must be positive")
        if self.max_output_size_bytes <= 0:
            raise ValueError(f"{self.preset_id}: output budget must be positive")
        if self.max_mass_residual <= 0.0:
            raise ValueError(f"{self.preset_id}: mass residual budget must be positive")
        if self.max_cfl <= 0.0:
            raise ValueError(f"{self.preset_id}: CFL budget must be positive")
        if self.long_run_steps <= 0:
            raise ValueError(f"{self.preset_id}: long-run steps must be positive")
        if self.evidence_status != "synthetic_runtime_performance_budget":
            raise ValueError(f"{self.preset_id}: unsupported budget evidence status")
        if "not hardware qualification" not in " ".join(self.limitations):
            raise ValueError(f"{self.preset_id}: missing hardware caveat")


@dataclass(frozen=True)
class CfdRuntimePerformanceGateRecord:
    preset_id: str
    cell_count: int
    solver_dt: float
    scalar_count: int
    estimated_field_memory_bytes: int
    output_size_bytes: int
    iterations: int
    wall_time_seconds: float
    max_wall_time_seconds: float
    max_cfl: float
    mass_residual: float
    long_run_drift: float
    stable: bool
    gate_passed: bool
    deterministic_signature: str
    evidence_status: str
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.preset_id:
            raise ValueError("gate record preset_id is required")
        if self.cell_count <= 0:
            raise ValueError(f"{self.preset_id}: cell_count must be positive")
        if self.solver_dt <= 0.0:
            raise ValueError(f"{self.preset_id}: solver_dt must be positive")
        if self.scalar_count <= 0:
            raise ValueError(f"{self.preset_id}: scalar_count must be positive")
        if self.estimated_field_memory_bytes <= 0:
            raise ValueError(f"{self.preset_id}: memory estimate must be positive")
        if self.output_size_bytes <= 0:
            raise ValueError(f"{self.preset_id}: output size must be positive")
        if self.iterations <= 0:
            raise ValueError(f"{self.preset_id}: iterations must be positive")
        if self.wall_time_seconds < 0.0:
            raise ValueError(f"{self.preset_id}: wall time cannot be negative")
        if self.max_wall_time_seconds <= 0.0:
            raise ValueError(f"{self.preset_id}: wall-time budget must be positive")
        if self.long_run_drift < 0.0:
            raise ValueError(f"{self.preset_id}: long-run drift cannot be negative")
        if self.evidence_status != "synthetic_runtime_performance_gate":
            raise ValueError(f"{self.preset_id}: unsupported gate evidence status")
        if "not hardware qualification" not in " ".join(self.limitations):
            raise ValueError(f"{self.preset_id}: missing hardware caveat")


def performance_presets() -> tuple[CfdPerformancePreset, ...]:
    return (
        CfdPerformancePreset(
            preset_id="tiny-grid",
            cells=(6, 3, 2),
            extents=(3.0, 1.0, 0.8),
            solver_dt=0.01,
            scalar_count=2,
            intended_use="unit tests and smoke checks",
        ),
        CfdPerformancePreset(
            preset_id="small-grid",
            cells=(10, 4, 3),
            extents=(5.0, 1.5, 1.2),
            solver_dt=0.01,
            scalar_count=3,
            intended_use="selected-area offline scenario checks",
        ),
        CfdPerformancePreset(
            preset_id="medium-grid",
            cells=(16, 6, 4),
            extents=(8.0, 3.0, 1.8),
            solver_dt=0.005,
            scalar_count=4,
            intended_use="pre-coupling benchmark ceiling on developer machines",
        ),
    )


def get_performance_preset(preset_id: str) -> CfdPerformancePreset:
    for preset in performance_presets():
        if preset.preset_id == preset_id:
            return preset
    raise ValueError(f"unknown performance preset: {preset_id}")


def performance_preset_ids() -> tuple[str, ...]:
    return tuple(preset.preset_id for preset in performance_presets())


def runtime_performance_budgets() -> tuple[CfdRuntimePerformanceBudget, ...]:
    return tuple(
        CfdRuntimePerformanceBudget(
            preset_id=preset.preset_id,
            max_wall_time_seconds=_wall_budget_for(preset.preset_id),
            max_field_memory_bytes=estimate_field_memory_bytes(
                mesh_for_preset(preset), scalar_count=preset.scalar_count
            )
            * 2,
            max_output_size_bytes=8192 + (preset.scalar_count * 1024),
            max_mass_residual=0.5,
            max_cfl=0.8,
            long_run_steps=8,
            evidence_status="synthetic_runtime_performance_budget",
            limitations=_performance_limitations(),
        )
        for preset in performance_presets()
    )


def get_runtime_performance_budget(
    preset_id: str,
) -> CfdRuntimePerformanceBudget:
    for budget in runtime_performance_budgets():
        if budget.preset_id == preset_id:
            return budget
    raise ValueError(f"unknown performance budget: {preset_id}")


def _wall_budget_for(preset_id: str) -> float:
    budgets = {
        "tiny-grid": 1.0,
        "small-grid": 2.0,
        "medium-grid": 4.0,
    }
    return budgets[preset_id]


def _performance_limitations() -> tuple[str, ...]:
    return (
        "synthetic local runtime performance evidence",
        "not hardware qualification, appliance sizing, or field performance evidence",
    )


def mesh_for_preset(preset: CfdPerformancePreset) -> StructuredMesh:
    preset.validate()
    return create_rectangular_mesh(
        cells=preset.cells,
        extents=preset.extents,
        boundaries=(
            BoundaryPatch("benchmark-inlet", "xmin", "inlet", 0.02),
            BoundaryPatch("benchmark-outlet", "xmax", "outlet", 0.0),
            BoundaryPatch("benchmark-floor", "zmin", "wall", 0.0),
            BoundaryPatch("benchmark-surface", "zmax", "symmetry", 0.0),
        ),
    )


def estimate_field_memory_bytes(
    mesh: StructuredMesh, *, scalar_count: int, include_mask: bool = True
) -> int:
    if scalar_count < 0:
        raise ValueError("scalar_count cannot be negative")
    float_arrays = FLOW_ARRAY_COUNT + scalar_count
    mask_bytes = mesh.cell_count if include_mask else 0
    return (mesh.cell_count * float_arrays * BYTES_PER_FLOAT64) + mask_bytes


def run_cfd_benchmark(
    preset: CfdPerformancePreset, *, iterations: int = 3
) -> CfdBenchmarkResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    mesh = mesh_for_preset(preset)
    flow = initialize_flow(mesh)
    scalar = ScalarField.uniform(mesh, name="benchmark_scalar", units="a.u.", value=1.0)
    flow_config = FlowSolverConfig(dt=preset.solver_dt, pressure_iterations=2)
    scalar_config = ScalarTransportConfig(dt=preset.solver_dt, diffusivity=1.0e-5)

    started = time.perf_counter()
    stable = True
    max_cfl = 0.0
    mass_residual = 0.0
    for _ in range(iterations):
        flow, flow_result = solve_flow_step(mesh, flow, flow_config)
        scalar, scalar_result = step_scalar_transport(
            mesh,
            scalar,
            flow,
            scalar_config,
            source_cells=((0, 0, 0, 0.01),),
        )
        stable = stable and flow_result.stable and scalar_result.stable
        max_cfl = max(max_cfl, flow_result.max_cfl)
        mass_residual = flow_result.mass_residual
    elapsed = time.perf_counter() - started

    result = CfdBenchmarkResult(
        preset_id=preset.preset_id,
        cell_count=mesh.cell_count,
        estimated_field_memory_bytes=estimate_field_memory_bytes(
            mesh, scalar_count=preset.scalar_count
        ),
        iterations=iterations,
        wall_time_seconds=elapsed,
        stable=stable,
        max_cfl=max_cfl,
        mass_residual=mass_residual,
    )
    result.validate()
    return result


def build_runtime_performance_gate(
    *, iterations: int = 2
) -> tuple[CfdRuntimePerformanceGateRecord, ...]:
    records = tuple(
        _build_gate_record(
            preset, get_runtime_performance_budget(preset.preset_id), iterations
        )
        for preset in performance_presets()
    )
    for record in records:
        record.validate()
    return records


def render_runtime_performance_gate_json(
    records: tuple[CfdRuntimePerformanceGateRecord, ...],
) -> str:
    """Render deterministic gate fields, excluding volatile wall-time values."""

    payload = {
        "artifact": "runtime_performance_gate",
        "evidence_status": "synthetic_runtime_performance_gate",
        "records": [
            {
                key: value
                for key, value in asdict(record).items()
                if key != "wall_time_seconds"
            }
            for record in records
        ],
        "volatile_fields": ("wall_time_seconds",),
        "limitations": list(_performance_limitations()),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _build_gate_record(
    preset: CfdPerformancePreset,
    budget: CfdRuntimePerformanceBudget,
    iterations: int,
) -> CfdRuntimePerformanceGateRecord:
    benchmark = run_cfd_benchmark(preset, iterations=iterations)
    mesh = mesh_for_preset(preset)
    output_size = _estimated_output_size(preset)
    drift = _long_run_drift(mesh, budget.long_run_steps, preset.solver_dt)
    gate_passed = (
        benchmark.stable
        and benchmark.wall_time_seconds <= budget.max_wall_time_seconds
        and benchmark.estimated_field_memory_bytes <= budget.max_field_memory_bytes
        and output_size <= budget.max_output_size_bytes
        and benchmark.mass_residual <= budget.max_mass_residual
        and benchmark.max_cfl <= budget.max_cfl
        and drift <= 1.0e-9
    )
    signature = (
        f"{preset.preset_id}:{mesh.cell_count}:{preset.solver_dt}:"
        f"{preset.scalar_count}:{benchmark.estimated_field_memory_bytes}:"
        f"{output_size}:{benchmark.max_cfl:.6g}:{benchmark.mass_residual:.6g}:"
        f"{drift:.6g}:{gate_passed}"
    )
    return CfdRuntimePerformanceGateRecord(
        preset_id=preset.preset_id,
        cell_count=mesh.cell_count,
        solver_dt=preset.solver_dt,
        scalar_count=preset.scalar_count,
        estimated_field_memory_bytes=benchmark.estimated_field_memory_bytes,
        output_size_bytes=output_size,
        iterations=benchmark.iterations,
        wall_time_seconds=benchmark.wall_time_seconds,
        max_wall_time_seconds=budget.max_wall_time_seconds,
        max_cfl=benchmark.max_cfl,
        mass_residual=benchmark.mass_residual,
        long_run_drift=drift,
        stable=benchmark.stable,
        gate_passed=gate_passed,
        deterministic_signature=signature,
        evidence_status="synthetic_runtime_performance_gate",
        limitations=_performance_limitations(),
    )


def _estimated_output_size(preset: CfdPerformancePreset) -> int:
    mesh = mesh_for_preset(preset)
    sample_rows = min(4, mesh.cell_count)
    scalar_row_bytes = 96 * preset.scalar_count * sample_rows
    flow_row_bytes = 128 * sample_rows
    metadata_bytes = 512
    return metadata_bytes + scalar_row_bytes + flow_row_bytes


def _long_run_drift(mesh: StructuredMesh, steps: int, dt: float) -> float:
    if steps <= 0:
        raise ValueError("long-run steps must be positive")
    flow = initialize_flow(mesh)
    scalar = ScalarField.uniform(mesh, name="drift_scalar", units="a.u.", value=1.0)
    initial_mass = float(scalar.values.sum() * mesh.dx * mesh.dy * mesh.dz)
    config = ScalarTransportConfig(dt=dt, diffusivity=0.0)
    for _ in range(steps):
        scalar, result = step_scalar_transport(mesh, scalar, flow, config)
        if not result.stable:
            return float("inf")
    final_mass = float(scalar.values.sum() * mesh.dx * mesh.dy * mesh.dz)
    return abs(final_mass - initial_mass)
