"""Runtime selection and filtering for staged plant profiles."""

from __future__ import annotations

from dataclasses import replace

from .catalog import AREA_CHOICES, STAGE_NODE_FILTERS, STAGE_TRANSACTION_FILTERS, STAGES
from .control import evaluate_controller_states
from .evidence import build_process_evolution, build_process_reviews
from .models import RuntimeArtifact
from .profiles import get_profile
from .scenarios import get_scenario


def build_runtime_artifact(
    profile_id: str,
    scenario_id: str,
    area: str,
    stage: str,
    control_system: str | None = None,
    topology: str | None = None,
    media: str | None = None,
) -> RuntimeArtifact:
    if area not in AREA_CHOICES:
        raise ValueError(f"unknown area {area!r}")
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if area == "all" and stage != "offline-export":
        raise ValueError("area 'all' is allowed only for offline-export")

    base_profile = get_profile(profile_id)
    profile = replace(
        base_profile,
        control_system=control_system or base_profile.control_system,
        topology=topology or base_profile.topology,
        media=media or base_profile.media,
    )
    scenario = get_scenario(scenario_id)
    selected_areas = tuple(profile.areas if area == "all" else (area,))
    missing = tuple(item for item in selected_areas if item not in profile.areas)
    if missing:
        raise ValueError(f"profile {profile_id!r} does not include area(s): {missing}")

    transaction_stages = STAGE_TRANSACTION_FILTERS[stage]
    transactions = tuple(
        tx
        for tx in scenario.transactions
        if tx.area in selected_areas and tx.stage in transaction_stages
    )
    referenced_nodes = {tx.actor_id for tx in transactions} | {
        tx.target_id for tx in transactions
    }
    allowed_node_stages = STAGE_NODE_FILTERS[stage]
    active_nodes = tuple(
        node
        for node in profile.nodes
        if (node.area in selected_areas and node.stage in allowed_node_stages)
        or node.node_id in referenced_nodes
    )
    if transactions:
        active_node_ids = {node.node_id for node in active_nodes}
        missing_nodes = sorted(referenced_nodes - active_node_ids)
        if missing_nodes:
            raise ValueError(
                f"scenario references nodes outside profile: {missing_nodes}"
            )

    active_units = tuple(unit for unit in profile.units if unit.area in selected_areas)
    controller_states = evaluate_controller_states(active_nodes, transactions)
    process_evolution = build_process_evolution(scenario, transactions)
    process_reviews = build_process_reviews(
        process_evolution,
        transactions,
        controller_states,
    )
    limitations = tuple(profile.limitations) + tuple(scenario.limitations)
    return RuntimeArtifact(
        profile,
        scenario,
        area,
        stage,
        active_units,
        active_nodes,
        transactions,
        controller_states,
        process_evolution,
        process_reviews,
        limitations,
    )
