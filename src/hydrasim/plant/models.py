"""Reference water-plant architecture models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PlantUnit:
    unit_id: str
    label: str
    area: str
    kind: str
    description: str


@dataclass(frozen=True)
class PlantNode:
    node_id: str
    label: str
    area: str
    unit_id: str
    stage: str
    role: str
    mac: str
    ipv4: str
    protocol: str = "modbus-tcp"
    modbus_port: int | None = 502
    notes: str = ""


@dataclass(frozen=True)
class HsTransaction:
    ordinal: int
    timestamp_ms: int
    actor_id: str
    target_id: str
    stage: str
    area: str
    function_code: int
    operation: str
    address: int
    quantity: int
    value_summary: str
    response: str
    scenario_label: str
    review_hint: str = ""
    wire_values: Sequence[int] = ()


@dataclass(frozen=True)
class PlantProfile:
    profile_id: str
    name: str
    preset: str
    description: str
    areas: Sequence[str]
    control_system: str
    topology: str
    media: str
    protocol: str
    units: Sequence[PlantUnit]
    nodes: Sequence[PlantNode]
    limitations: Sequence[str]


@dataclass(frozen=True)
class HsScenario:
    scenario_id: str
    name: str
    purpose: str
    required_stages: Sequence[str]
    areas: Sequence[str]
    transactions: Sequence[HsTransaction]
    limitations: Sequence[str]


@dataclass(frozen=True)
class ControllerState:
    area: str
    controller_id: str
    mode: str
    routine: str
    alarm_state: str
    setpoint_summary: str
    effect_summary: str
    evidence_label: str


@dataclass(frozen=True)
class CfdProcessEvolution:
    scenario_id: str
    area: str
    scalar_name: str
    start_value: float
    end_value: float
    trend: str
    cfd_basis: str
    evidence_status: str
    limitations: Sequence[str]

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("process evolution scenario_id is required")
        if not self.area:
            raise ValueError("process evolution area is required")
        if not self.scalar_name:
            raise ValueError("process evolution scalar_name is required")
        if self.trend not in {"stable", "increase", "decrease", "review"}:
            raise ValueError(f"unsupported process trend: {self.trend}")
        if self.evidence_status != "synthetic_cfd_process_truth":
            raise ValueError("unsupported process evolution evidence status")
        if not self.limitations:
            raise ValueError("process evolution limitations are required")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError("process evolution must preserve validation caveat")


@dataclass(frozen=True)
class ScenarioProcessReview:
    scenario_id: str
    area: str
    process_truth: str
    observable_network_effects: str
    operator_historian_effects: str
    review_questions: Sequence[str]
    must_not_claim: Sequence[str]
    evidence_status: str

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario review scenario_id is required")
        if not self.area:
            raise ValueError("scenario review area is required")
        if not self.process_truth:
            raise ValueError("scenario review process truth is required")
        if not self.observable_network_effects:
            raise ValueError("scenario review network effects are required")
        if not self.operator_historian_effects:
            raise ValueError("scenario review operator/historian effects are required")
        if not self.review_questions:
            raise ValueError("scenario review questions are required")
        if not self.must_not_claim:
            raise ValueError("scenario review must-not-claim text is required")
        if self.evidence_status != "synthetic_scenario_process_review":
            raise ValueError("unsupported scenario review evidence status")
        blocked = " ".join(self.must_not_claim).lower()
        for phrase in ("real plant", "certification", "safety"):
            if phrase not in blocked:
                raise ValueError(f"scenario review must block {phrase!r} claims")


@dataclass(frozen=True)
class RuntimeArtifact:
    profile: PlantProfile
    scenario: HsScenario
    area: str
    stage: str
    active_units: Sequence[PlantUnit]
    active_nodes: Sequence[PlantNode]
    transactions: Sequence[HsTransaction]
    controller_states: Sequence[ControllerState]
    process_evolution: Sequence[CfdProcessEvolution]
    process_reviews: Sequence[ScenarioProcessReview]
    limitations: Sequence[str]
