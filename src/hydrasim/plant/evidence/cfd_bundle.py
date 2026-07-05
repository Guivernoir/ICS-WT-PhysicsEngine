"""Compact CFD lab-bundle artifacts for staged HydraSim runtime output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from hydrasim.hydraulics.cfd import (
    AreaCfdModel,
    FlowField,
    FlowSolverConfig,
    FlowStepResult,
    ScalarField,
    ScalarTransportConfig,
    initialize_flow,
    reference_area_model,
    solve_flow_step,
    step_scalar_transport,
)

from ..models import CfdProcessEvolution, RuntimeArtifact


@dataclass(frozen=True)
class CfdSnapshot:
    record: CfdProcessEvolution
    model: AreaCfdModel
    flow: FlowField
    flow_result: FlowStepResult
    scalar: ScalarField
    evolved: ScalarField
    samples: tuple[tuple[int, int, int], ...]


def build_cfd_lab_artifacts(artifact: RuntimeArtifact) -> dict[str, bytes]:
    """Return deterministic compact CFD artifacts for a runtime bundle.

    HS-32 exports process-state evidence beside the network evidence. The
    outputs are intentionally compact: mesh/geometry metadata and sampled field
    values, not unrestricted full-array dumps.
    """

    snapshots = tuple(
        _snapshot_for_record(record) for record in artifact.process_evolution
    )
    return {
        "cfd-mesh-geometry.json": _mesh_geometry_json(snapshots).encode("utf-8"),
        "cfd-state-timeline.csv": _state_timeline_csv(snapshots).encode("utf-8"),
        "cfd-scalar-fields.csv": _scalar_fields_csv(snapshots).encode("utf-8"),
        "cfd-flow-snapshots.csv": _flow_snapshots_csv(snapshots).encode("utf-8"),
    }


def _snapshot_for_record(record: CfdProcessEvolution) -> CfdSnapshot:
    model = reference_area_model(record.area)
    flow, flow_result = solve_flow_step(
        model.mesh,
        initialize_flow(model.mesh),
        FlowSolverConfig(dt=0.005, pressure_iterations=2),
    )
    scalar = ScalarField.uniform(
        model.mesh,
        name=record.scalar_name,
        units="process_units",
        value=record.start_value,
    )
    source_cell = (
        min(1, model.mesh.nx - 1),
        min(1, model.mesh.ny - 1),
        min(1, model.mesh.nz - 1),
    )
    delta = record.end_value - record.start_value
    source_cells: tuple[tuple[int, int, int, float], ...] = ()
    if abs(delta) > 1.0e-12:
        source_cells = ((*source_cell, delta / 0.01),)
    evolved, _scalar_result = step_scalar_transport(
        model.mesh,
        scalar,
        flow,
        ScalarTransportConfig(dt=0.01, diffusivity=0.0),
        source_cells=source_cells,
    )
    samples = tuple(_sample_cells(model.mesh.nx, model.mesh.ny, model.mesh.nz))
    return CfdSnapshot(record, model, flow, flow_result, scalar, evolved, samples)


def _sample_cells(nx: int, ny: int, nz: int) -> tuple[tuple[int, int, int], ...]:
    raw = (
        (0, 0, 0),
        (min(1, nx - 1), min(1, ny - 1), min(1, nz - 1)),
        (nx // 2, ny // 2, nz // 2),
        (nx - 1, ny - 1, nz - 1),
    )
    seen: set[tuple[int, int, int]] = set()
    output: list[tuple[int, int, int]] = []
    for cell in raw:
        if cell not in seen:
            seen.add(cell)
            output.append(cell)
    return tuple(output)


def _mesh_geometry_json(snapshots: tuple[CfdSnapshot, ...]) -> str:
    areas = []
    for snapshot in snapshots:
        record = snapshot.record
        model = snapshot.model
        areas.append(
            {
                "scenario_id": record.scenario_id,
                "area_id": model.area_id,
                "mesh": {
                    "cells": [model.mesh.nx, model.mesh.ny, model.mesh.nz],
                    "extents_m": [model.mesh.lx, model.mesh.ly, model.mesh.lz],
                    "cell_count": model.mesh.cell_count,
                    "boundaries": [
                        asdict(boundary) for boundary in model.mesh.boundaries
                    ],
                    "obstacles": [
                        asdict(obstacle) for obstacle in model.mesh.obstacles
                    ],
                },
                "digital_twin": {
                    "status": model.twin_metadata.status.value,
                    "geometry_reference": model.twin_metadata.geometry_reference,
                    "equipment_reference": model.twin_metadata.equipment_reference,
                },
                "limitations": list(model.limitations) + list(record.limitations),
            }
        )
    payload = {
        "artifact": "cfd_mesh_geometry",
        "evidence_status": "synthetic_cfd_lab_bundle_v2",
        "array_policy": "compact metadata and sampled field values only",
        "areas": areas,
        "must_not_claim": [
            "real plant validation",
            "commissioning evidence",
            "certification",
            "safety-system protection",
            "full physical fidelity",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _state_timeline_csv(snapshots: tuple[CfdSnapshot, ...]) -> str:
    rows = [
        "scenario_id,area,timestamp_ms,scalar_name,start_value,end_value,trend,"
        "evidence_status,limitations"
    ]
    for snapshot in snapshots:
        record = snapshot.record
        rows.append(
            ",".join(
                (
                    record.scenario_id,
                    record.area,
                    "0",
                    record.scalar_name,
                    f"{record.start_value:.6g}",
                    f"{record.end_value:.6g}",
                    record.trend,
                    "synthetic_cfd_lab_bundle_v2",
                    _csv("; ".join(record.limitations)),
                )
            )
        )
    return "\n".join(rows) + "\n"


def _scalar_fields_csv(snapshots: tuple[CfdSnapshot, ...]) -> str:
    rows = [
        "scenario_id,area,scalar_name,cell_id,i,j,k,x_m,y_m,z_m,start_value,"
        "end_value,units,evidence_status"
    ]
    for snapshot in snapshots:
        record = snapshot.record
        model = snapshot.model
        scalar = snapshot.scalar
        evolved = snapshot.evolved
        for i, j, k in snapshot.samples:
            cell_id = model.mesh.cell_id(i, j, k)
            x, y, z = model.mesh.cell_center(i, j, k)
            rows.append(
                ",".join(
                    (
                        record.scenario_id,
                        record.area,
                        record.scalar_name,
                        str(cell_id),
                        str(i),
                        str(j),
                        str(k),
                        f"{x:.6g}",
                        f"{y:.6g}",
                        f"{z:.6g}",
                        f"{float(scalar.values[k, j, i]):.6g}",
                        f"{float(evolved.values[k, j, i]):.6g}",
                        scalar.units,
                        "synthetic_cfd_lab_bundle_v2",
                    )
                )
            )
    return "\n".join(rows) + "\n"


def _flow_snapshots_csv(snapshots: tuple[CfdSnapshot, ...]) -> str:
    rows = [
        "scenario_id,area,cell_id,i,j,k,u,v,w,pressure,mass_residual,max_cfl,"
        "stable,evidence_status"
    ]
    for snapshot in snapshots:
        record = snapshot.record
        model = snapshot.model
        flow = snapshot.flow
        result = snapshot.flow_result
        for i, j, k in snapshot.samples:
            cell_id = model.mesh.cell_id(i, j, k)
            rows.append(
                ",".join(
                    (
                        record.scenario_id,
                        record.area,
                        str(cell_id),
                        str(i),
                        str(j),
                        str(k),
                        f"{float(flow.u[k, j, i]):.6g}",
                        f"{float(flow.v[k, j, i]):.6g}",
                        f"{float(flow.w[k, j, i]):.6g}",
                        f"{float(flow.pressure[k, j, i]):.6g}",
                        f"{result.mass_residual:.6g}",
                        f"{result.max_cfl:.6g}",
                        str(result.stable).lower(),
                        "synthetic_cfd_lab_bundle_v2",
                    )
                )
            )
    return "\n".join(rows) + "\n"


def _csv(value: str) -> str:
    if "," in value or "\n" in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value
