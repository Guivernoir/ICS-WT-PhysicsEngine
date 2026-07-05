"""Validation for reference water-plant profiles."""

from __future__ import annotations

from .catalog import (
    AREAS,
    CONTROL_SYSTEMS,
    MEDIA,
    NODE_STAGES,
    PRESETS,
    PROTOCOLS,
    TOPOLOGIES,
)
from .models import PlantProfile


def _has_duplicate(values) -> bool:
    seen = set()
    for value in values:
        if value in seen:
            return True
        seen.add(value)
    return False


def validate_profile(profile: PlantProfile) -> tuple[str, ...]:
    errors: list[str] = []
    if profile.preset not in PRESETS:
        errors.append(f"{profile.profile_id}: unknown preset {profile.preset}")
    if profile.control_system not in CONTROL_SYSTEMS:
        errors.append(f"{profile.profile_id}: unknown control system")
    if profile.topology not in TOPOLOGIES:
        errors.append(f"{profile.profile_id}: unknown topology")
    if profile.media not in MEDIA:
        errors.append(f"{profile.profile_id}: unknown media")
    if profile.protocol not in PROTOCOLS:
        errors.append(f"{profile.profile_id}: unknown protocol")
    if not profile.limitations:
        errors.append(f"{profile.profile_id}: limitations are required")

    for area in profile.areas:
        if area not in AREAS:
            errors.append(f"{profile.profile_id}: unknown area {area}")

    unit_ids = [unit.unit_id for unit in profile.units]
    node_ids = [node.node_id for node in profile.nodes]
    if _has_duplicate(unit_ids):
        errors.append(f"{profile.profile_id}: unit IDs must be unique")
    if _has_duplicate(node_ids):
        errors.append(f"{profile.profile_id}: node IDs must be unique")

    unit_lookup = {unit.unit_id: unit for unit in profile.units}
    for node in profile.nodes:
        if node.area not in AREAS:
            errors.append(f"{node.node_id}: unknown node area")
        if node.stage not in NODE_STAGES:
            errors.append(f"{node.node_id}: unknown node stage")
        if node.unit_id not in unit_lookup:
            errors.append(f"{node.node_id}: unit_id does not reference a unit")
        if node.modbus_port is not None and not 1 <= node.modbus_port <= 65535:
            errors.append(f"{node.node_id}: invalid Modbus port")
        if node.stage == "passive" and "passive" not in node.role.lower():
            errors.append(f"{node.node_id}: passive node role must be explicit")
    return tuple(errors)


def assert_valid_profile(profile: PlantProfile) -> PlantProfile:
    errors = validate_profile(profile)
    if errors:
        raise ValueError("; ".join(errors))
    return profile
