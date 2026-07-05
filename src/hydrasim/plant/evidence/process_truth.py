"""CFD-backed process evolution truth for staged HydraSim scenarios."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from hydrasim.hydraulics.cfd import (
    FlowSolverConfig,
    ScalarField,
    ScalarTransportConfig,
    initialize_flow,
    reference_area_model,
    solve_flow_step,
    step_scalar_transport,
)

from ..models import CfdProcessEvolution, HsScenario, HsTransaction


def _limitations() -> tuple[str, ...]:
    return (
        "synthetic CFD-backed process evolution",
        "not real-plant validation, commissioning, safety, or design evidence",
    )


def _area_scalar(area: str, transactions: Sequence[HsTransaction]) -> str:
    labels = " ".join(
        item
        for tx in transactions
        for item in (tx.scenario_label, tx.value_summary, tx.review_hint)
    ).lower()
    model = reference_area_model(area)
    scalars = set(model.scalar_names)
    if area == "dosing" and "ph" in labels and "ph_proxy" in scalars:
        return "ph_proxy"
    if area == "filtration" and "backwash" in labels and "headloss_proxy" in scalars:
        return "headloss_proxy"
    if area == "storage-pumping" and "level" in labels:
        return "chlorine" if "chlorine" in scalars else model.scalar_names[0]
    if area == "disinfection" and "chlorine" in scalars:
        return "chlorine"
    return model.scalar_names[0]


def _delta_for(transactions: Sequence[HsTransaction]) -> tuple[float, str]:
    labels = " ".join(
        item
        for tx in transactions
        for item in (tx.scenario_label, tx.value_summary, tx.review_hint, tx.response)
    ).lower()
    if "pump_failure" in labels:
        return -0.08, "decrease"
    if "backwash" in labels:
        return -0.05, "decrease"
    if "drift" in labels or "exception" in labels or "unknown" in labels:
        return 0.0, "review"
    if "chlorine_rate" in labels or "dose_rate" in labels or "corrective" in labels:
        return 0.06, "increase"
    if "process_deviation" in labels or "excursion" in labels:
        return 0.04, "review"
    if "maintenance" in labels or "manual" in labels or "failover" in labels:
        return 0.0, "review"
    return 0.0, "stable"


def _process_evolution_for_area(
    scenario: HsScenario,
    area: str,
    transactions: Sequence[HsTransaction],
) -> CfdProcessEvolution:
    model = reference_area_model(area)
    scalar_name = _area_scalar(area, transactions)
    scalar = ScalarField.uniform(
        model.mesh, name=scalar_name, units="process_units", value=1.0
    )
    flow, flow_result = solve_flow_step(
        model.mesh,
        initialize_flow(model.mesh),
        FlowSolverConfig(dt=0.005, pressure_iterations=2),
    )
    delta, trend = _delta_for(transactions)
    source_cell = (
        min(1, model.mesh.nx - 1),
        min(1, model.mesh.ny - 1),
        min(1, model.mesh.nz - 1),
    )
    source_cells: tuple[tuple[int, int, int, float], ...] = ()
    if delta != 0.0:
        source_cells = ((*source_cell, delta / 0.01),)
    evolved, _ = step_scalar_transport(
        model.mesh,
        scalar,
        flow,
        ScalarTransportConfig(dt=0.01, diffusivity=0.0),
        source_cells=source_cells,
    )
    i, j, k = source_cell
    evolution = CfdProcessEvolution(
        scenario_id=scenario.scenario_id,
        area=area,
        scalar_name=scalar_name,
        start_value=1.0,
        end_value=float(evolved.values[k, j, i]),
        trend=trend,
        cfd_basis=(
            f"area_model={model.area_id}; cells={model.mesh.cell_count}; "
            f"flow_mass_residual={flow_result.mass_residual:.6g}"
        ),
        evidence_status="synthetic_cfd_process_truth",
        limitations=_limitations(),
    )
    evolution.validate()
    return evolution


def build_process_evolution(
    scenario: HsScenario,
    transactions: Iterable[HsTransaction],
) -> tuple[CfdProcessEvolution, ...]:
    by_area: dict[str, list[HsTransaction]] = defaultdict(list)
    for tx in transactions:
        by_area[tx.area].append(tx)
    records = tuple(
        _process_evolution_for_area(scenario, area, tuple(area_txs))
        for area, area_txs in sorted(by_area.items())
        if area_txs
    )
    for record in records:
        record.validate()
    return records
