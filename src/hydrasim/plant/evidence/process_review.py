"""Scenario process-truth review records for demo/training readiness."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from ..models import (
    CfdProcessEvolution,
    ControllerState,
    HsTransaction,
    ScenarioProcessReview,
)


def _network_effects(transactions: Sequence[HsTransaction]) -> str:
    reads = sum(1 for tx in transactions if tx.operation.startswith("read_"))
    writes = sum(1 for tx in transactions if tx.operation.startswith("write_"))
    exceptions = sum(1 for tx in transactions if tx.response.startswith("exception_"))
    review_hints = sorted({tx.review_hint for tx in transactions if tx.review_hint})
    parts = [f"reads={reads}", f"writes={writes}", f"exceptions={exceptions}"]
    if review_hints:
        parts.append("review_hints=" + "+".join(review_hints))
    return "; ".join(parts)


def _operator_historian_effects(
    transactions: Sequence[HsTransaction],
    controller_states: Sequence[ControllerState],
) -> str:
    supervisory = sum(1 for tx in transactions if tx.stage == "supervisory")
    historian = sum(1 for tx in transactions if tx.actor_id == "plant-historian")
    hmi = sum(1 for tx in transactions if tx.actor_id == "scada-hmi")
    engineering = sum(
        1 for tx in transactions if tx.actor_id == "engineering-workstation"
    )
    alarms = sorted({state.alarm_state for state in controller_states})
    return (
        f"supervisory_transactions={supervisory}; historian_reads={historian}; "
        f"hmi_reads={hmi}; engineering_actions={engineering}; "
        f"controller_alarm_states={'+'.join(alarms) if alarms else 'none'}"
    )


def _review_questions(transactions: Sequence[HsTransaction]) -> tuple[str, ...]:
    questions = [
        "Does the synthetic process trend match the scenario story?",
        "Do the observable Modbus transactions support the review prompt?",
        "Would an operator or engineer need more context before acting?",
    ]
    if any(tx.review_hint for tx in transactions):
        questions.append(
            "Are the review hints worded as prompts rather than conclusions?"
        )
    if any(tx.response.startswith("exception_") for tx in transactions):
        questions.append("Is the exception traffic handled as coverage evidence only?")
    return tuple(questions)


def _must_not_claim() -> tuple[str, ...]:
    return (
        "Do not claim real plant validation or operational evidence.",
        "Do not claim certification, commissioning authority, or compliance.",
        "Do not claim safety-system protection or safety impact.",
        "Do not claim equipment failure, attack, attribution, or malicious intent.",
        "Do not claim full physical fidelity or calibrated plant equivalence.",
    )


def build_process_reviews(
    process_evolution: Iterable[CfdProcessEvolution],
    transactions: Iterable[HsTransaction],
    controller_states: Iterable[ControllerState],
) -> tuple[ScenarioProcessReview, ...]:
    tx_by_area: dict[str, list[HsTransaction]] = defaultdict(list)
    for tx in transactions:
        tx_by_area[tx.area].append(tx)
    state_by_area: dict[str, list[ControllerState]] = defaultdict(list)
    for state in controller_states:
        state_by_area[state.area].append(state)

    reviews: list[ScenarioProcessReview] = []
    for record in process_evolution:
        area_transactions = tuple(tx_by_area.get(record.area, ()))
        area_states = tuple(state_by_area.get(record.area, ()))
        review = ScenarioProcessReview(
            scenario_id=record.scenario_id,
            area=record.area,
            process_truth=(
                f"{record.scalar_name} {record.trend}: "
                f"{record.start_value:.6g}->{record.end_value:.6g}; "
                f"{record.evidence_status}"
            ),
            observable_network_effects=_network_effects(area_transactions),
            operator_historian_effects=_operator_historian_effects(
                area_transactions,
                area_states,
            ),
            review_questions=_review_questions(area_transactions),
            must_not_claim=_must_not_claim(),
            evidence_status="synthetic_scenario_process_review",
        )
        review.validate()
        reviews.append(review)
    return tuple(reviews)
