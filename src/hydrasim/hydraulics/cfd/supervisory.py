"""Supervisory digital-twin records for CFD-backed plant context."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .area_models import AREA_IDS, AreaCfdModel, reference_area_model
from .fields import FlowField, ScalarField

SupervisoryRole = str
SupervisorySourceKind = str

VALID_SUPERVISORY_ROLES = {"hmi", "historian", "engineering_workstation"}
VALID_SUPERVISORY_SOURCE_KINDS = {
    "cfd_scalar_summary",
    "cfd_flow_summary",
    "digital_twin_metadata",
    "metadata_only",
}


@dataclass(frozen=True)
class SupervisoryProfile:
    """A synthetic supervisory consumer of CFD/digital-twin facts."""

    profile_id: str
    role: SupervisoryRole
    purpose: str
    area_scope: str
    polling_interval_seconds: float
    consumed_contexts: tuple[str, ...]
    evidence_status: str = "simulated_metadata"
    limitations: tuple[str, ...] = (
        "synthetic supervisory profile",
        "not vendor HMI, historian, or engineering-workstation emulation",
        "does not prove operational visibility or site identity",
    )

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if self.role not in VALID_SUPERVISORY_ROLES:
            raise ValueError(f"{self.profile_id}: unsupported supervisory role")
        if not self.purpose:
            raise ValueError(f"{self.profile_id}: purpose is required")
        if self.area_scope != "all" and self.area_scope not in AREA_IDS:
            raise ValueError(f"{self.profile_id}: unsupported area scope")
        if self.polling_interval_seconds <= 0.0:
            raise ValueError(f"{self.profile_id}: polling interval must be positive")
        if not self.consumed_contexts:
            raise ValueError(f"{self.profile_id}: consumed contexts are required")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.profile_id}: evidence status must be simulated")
        if not self.limitations:
            raise ValueError(f"{self.profile_id}: limitations are required")


@dataclass(frozen=True)
class SupervisoryTwinRecord:
    """A bounded supervisory view of process/digital-twin evidence."""

    record_id: str
    profile_id: str
    role: SupervisoryRole
    area_id: str
    variable: str
    value: float
    units: str
    source_kind: SupervisorySourceKind
    spatial_context: str
    twin_status: str
    site_identity: str = "unknown"
    evidence_status: str = "simulated_metadata"
    limitations: tuple[str, ...] = (
        "synthetic supervisory record",
        "not operational historian evidence",
        "not site identity or equipment identity evidence",
    )

    def validate(self) -> None:
        if not self.record_id:
            raise ValueError("record_id is required")
        if not self.profile_id:
            raise ValueError(f"{self.record_id}: profile_id is required")
        if self.role not in VALID_SUPERVISORY_ROLES:
            raise ValueError(f"{self.record_id}: unsupported role")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.record_id}: unsupported area")
        if not self.variable:
            raise ValueError(f"{self.record_id}: variable is required")
        if not math.isfinite(self.value):
            raise ValueError(f"{self.record_id}: value must be finite")
        if not self.units:
            raise ValueError(f"{self.record_id}: units are required")
        if self.source_kind not in VALID_SUPERVISORY_SOURCE_KINDS:
            raise ValueError(f"{self.record_id}: unsupported source kind")
        if not self.spatial_context:
            raise ValueError(f"{self.record_id}: spatial context is required")
        if not self.twin_status:
            raise ValueError(f"{self.record_id}: twin status is required")
        if self.site_identity != "unknown":
            raise ValueError(f"{self.record_id}: site identity must remain unknown")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.record_id}: evidence status must be simulated")
        if not self.limitations:
            raise ValueError(f"{self.record_id}: limitations are required")


def supervisory_profile_catalog() -> tuple[SupervisoryProfile, ...]:
    """Return deterministic synthetic supervisory consumer profiles."""

    return (
        SupervisoryProfile(
            profile_id="hmi-process-overview",
            role="hmi",
            purpose="Show selected-area CFD process context to a synthetic operator view.",
            area_scope="all",
            polling_interval_seconds=2.0,
            consumed_contexts=("scalar_summary", "flow_summary", "limitations"),
        ),
        SupervisoryProfile(
            profile_id="historian-process-trend",
            role="historian",
            purpose="Store deterministic process trend samples with spatial context.",
            area_scope="all",
            polling_interval_seconds=5.0,
            consumed_contexts=("scalar_summary", "twin_status", "spatial_context"),
        ),
        SupervisoryProfile(
            profile_id="engineering-model-context",
            role="engineering_workstation",
            purpose="Inspect synthetic mesh, geometry, and model-status metadata.",
            area_scope="all",
            polling_interval_seconds=30.0,
            consumed_contexts=("mesh_summary", "digital_twin_metadata", "limitations"),
        ),
    )


def supervisory_profile_ids() -> tuple[str, ...]:
    return tuple(profile.profile_id for profile in supervisory_profile_catalog())


def _scalar_records(
    model: AreaCfdModel,
    profile: SupervisoryProfile,
    scalar: ScalarField,
) -> tuple[SupervisoryTwinRecord, ...]:
    scalar.validate(model.mesh)
    values = (
        ("minimum", float(scalar.values.min())),
        ("mean", float(scalar.values.mean())),
        ("maximum", float(scalar.values.max())),
    )
    return tuple(
        SupervisoryTwinRecord(
            record_id=f"{profile.profile_id}:{model.area_id}:{scalar.name}:{name}",
            profile_id=profile.profile_id,
            role=profile.role,
            area_id=model.area_id,
            variable=f"{scalar.name}_{name}",
            value=value,
            units=scalar.units,
            source_kind="cfd_scalar_summary",
            spatial_context="area mesh summary",
            twin_status=model.twin_metadata.status.value,
        )
        for name, value in values
    )


def _flow_records(
    model: AreaCfdModel,
    profile: SupervisoryProfile,
    flow: FlowField,
) -> tuple[SupervisoryTwinRecord, ...]:
    flow.validate(model.mesh)
    values = (
        ("max_u", float(flow.u.max()), "m/s"),
        ("max_v", float(flow.v.max()), "m/s"),
        ("max_w", float(flow.w.max()), "m/s"),
        ("pressure_range", float(flow.pressure.max() - flow.pressure.min()), "Pa"),
    )
    return tuple(
        SupervisoryTwinRecord(
            record_id=f"{profile.profile_id}:{model.area_id}:{name}",
            profile_id=profile.profile_id,
            role=profile.role,
            area_id=model.area_id,
            variable=name,
            value=value,
            units=units,
            source_kind="cfd_flow_summary",
            spatial_context="area mesh summary",
            twin_status=model.twin_metadata.status.value,
        )
        for name, value, units in values
    )


def _metadata_records(
    model: AreaCfdModel,
    profile: SupervisoryProfile,
) -> tuple[SupervisoryTwinRecord, ...]:
    values = (
        ("mesh_cell_count", float(model.mesh.cell_count)),
        ("scalar_name_count", float(len(model.scalar_names))),
        ("uncertainty_record_count", float(len(model.twin_metadata.uncertainty))),
    )
    return tuple(
        SupervisoryTwinRecord(
            record_id=f"{profile.profile_id}:{model.area_id}:{name}",
            profile_id=profile.profile_id,
            role=profile.role,
            area_id=model.area_id,
            variable=name,
            value=value,
            units="count",
            source_kind="digital_twin_metadata",
            spatial_context=model.twin_metadata.geometry_reference,
            twin_status=model.twin_metadata.status.value,
        )
        for name, value in values
    )


def build_supervisory_records(
    model: AreaCfdModel,
    *,
    scalar: ScalarField | None = None,
    flow: FlowField | None = None,
    profiles: tuple[SupervisoryProfile, ...] | None = None,
) -> tuple[SupervisoryTwinRecord, ...]:
    """Build deterministic synthetic supervisory records for one CFD area."""

    model.validate()
    selected_profiles = (
        profiles if profiles is not None else supervisory_profile_catalog()
    )
    records: list[SupervisoryTwinRecord] = []
    for profile in selected_profiles:
        profile.validate()
        if profile.area_scope not in {"all", model.area_id}:
            continue
        if scalar is not None and "scalar_summary" in profile.consumed_contexts:
            records.extend(_scalar_records(model, profile, scalar))
        if flow is not None and "flow_summary" in profile.consumed_contexts:
            records.extend(_flow_records(model, profile, flow))
        if "digital_twin_metadata" in profile.consumed_contexts:
            records.extend(_metadata_records(model, profile))
    for record in records:
        record.validate()
    return tuple(records)


def build_reference_supervisory_records(
    area_id: str,
) -> tuple[SupervisoryTwinRecord, ...]:
    """Build metadata-only supervisory records for a reference CFD area."""

    return build_supervisory_records(reference_area_model(area_id))
