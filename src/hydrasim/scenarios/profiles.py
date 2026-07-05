"""Built-in deterministic Modbus scenario profile index."""

from __future__ import annotations

from .models import ScenarioProfile
from .profile_expanded import EXPANDED_SCENARIOS
from .profile_primary import PRIMARY_SCENARIOS
from .profile_topology import TOPOLOGY_SCENARIOS
from .validator import assert_valid_scenario

SCENARIO_ALIASES: dict[str, str] = {
    "water-treatment-normal": "MVP-MB-HYDRA-002",
    "water-treatment-unknown-host": "MVP-MB-HYDRA-003",
    "water-treatment-smart-field": "MVP-MB-HYDRA-004",
    "water-treatment-maintenance-window": "MVP-MB-HYDRA-005",
    "water-treatment-process-fault": "MVP-MB-HYDRA-006",
    "water-treatment-misconfiguration": "MVP-MB-HYDRA-007",
    "water-treatment-noisy-network": "MVP-MB-HYDRA-008",
    "water-treatment-degraded-operations": "MVP-MB-HYDRA-009",
}

SCENARIOS: tuple[ScenarioProfile, ...] = (
    PRIMARY_SCENARIOS + TOPOLOGY_SCENARIOS + EXPANDED_SCENARIOS
)


def scenario_ids() -> tuple[str, ...]:
    return tuple(scenario.scenario_id for scenario in SCENARIOS)


def get_scenario(scenario_id: str) -> ScenarioProfile:
    scenario_id = SCENARIO_ALIASES.get(scenario_id, scenario_id)
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return assert_valid_scenario(scenario)
    known = ", ".join(scenario_ids() + tuple(SCENARIO_ALIASES))
    raise ValueError(f"unknown scenario {scenario_id!r}; expected one of: {known}")
