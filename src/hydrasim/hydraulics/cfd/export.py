"""Deterministic CFD summary export."""

from __future__ import annotations

import json
from dataclasses import asdict

from .area_models import AreaCfdModel
from .fields import FlowField, ScalarField
from .solver import FlowStepResult


def export_cfd_summary(
    model: AreaCfdModel,
    flow: FlowField | None = None,
    scalar: ScalarField | None = None,
    flow_result: FlowStepResult | None = None,
) -> str:
    """Return stable JSON summary text without exporting bulky field arrays."""

    payload: dict[str, object] = {
        "area_id": model.area_id,
        "mesh": {
            "cells": [model.mesh.nx, model.mesh.ny, model.mesh.nz],
            "extents_m": [model.mesh.lx, model.mesh.ly, model.mesh.lz],
            "cell_count": model.mesh.cell_count,
            "boundaries": [asdict(boundary) for boundary in model.mesh.boundaries],
            "obstacles": [asdict(obstacle) for obstacle in model.mesh.obstacles],
        },
        "scalar_names": list(model.scalar_names),
        "digital_twin": {
            "name": model.twin_metadata.name,
            "status": model.twin_metadata.status.value,
            "geometry_reference": model.twin_metadata.geometry_reference,
            "equipment_reference": model.twin_metadata.equipment_reference,
            "uncertainty": [
                asdict(record) for record in model.twin_metadata.uncertainty
            ],
        },
        "limitations": list(model.limitations),
    }
    if flow is not None:
        flow.validate(model.mesh)
        payload["flow_summary"] = {
            "max_u": float(flow.u.max()),
            "max_v": float(flow.v.max()),
            "max_w": float(flow.w.max()),
            "pressure_min": float(flow.pressure.min()),
            "pressure_max": float(flow.pressure.max()),
        }
    if scalar is not None:
        scalar.validate(model.mesh)
        payload["scalar_summary"] = {
            "name": scalar.name,
            "units": scalar.units,
            "min": float(scalar.values.min()),
            "max": float(scalar.values.max()),
        }
    if flow_result is not None:
        payload["flow_diagnostics"] = asdict(flow_result)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
