"""Bounded controller-state evaluation for staged HydraSim artifacts."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .models import ControllerState, HsTransaction, PlantNode


def _routine_for(transactions: Sequence[HsTransaction]) -> str:
    labels = " ".join(tx.scenario_label for tx in transactions)
    hints = " ".join(tx.review_hint for tx in transactions)
    operations = {tx.operation for tx in transactions}
    if "maintenance" in labels or "manual_mode" in labels:
        return "manual-supervised"
    if "pump_failure" in labels or "backwash" in labels:
        return "interlock-like"
    if "dose_rate" in labels or "chlorine_rate" in labels:
        return "pid-lite-label"
    if any(op.startswith("write") for op in operations):
        return "threshold-adjustment"
    if "failover" in hints:
        return "failover-review"
    return "monitoring"


def _mode_for(transactions: Sequence[HsTransaction]) -> str:
    joined = " ".join(
        item for tx in transactions for item in (tx.scenario_label, tx.value_summary)
    )
    if "manual" in joined or "maintenance" in joined:
        return "manual"
    return "auto"


def _alarm_for(transactions: Sequence[HsTransaction]) -> str:
    joined = " ".join(
        item
        for tx in transactions
        for item in (tx.response, tx.review_hint, tx.scenario_label)
    )
    if "exception" in joined:
        return "exception-review"
    if "must_review" in joined or "process_deviation" in joined:
        return "review"
    return "normal"


def _setpoints(transactions: Sequence[HsTransaction]) -> str:
    values = tuple(tx.value_summary for tx in transactions if tx.value_summary)
    if not values:
        return "no write setpoint observed"
    return "; ".join(values)


def _effect_summary(transactions: Sequence[HsTransaction]) -> str:
    if any(tx.operation.startswith("write") for tx in transactions):
        return "synthetic controller write affects scenario truth only"
    if transactions:
        return "synthetic polling/monitoring context"
    return "no controller traffic selected"


def evaluate_controller_states(
    nodes: Iterable[PlantNode],
    transactions: Iterable[HsTransaction],
) -> tuple[ControllerState, ...]:
    by_area: dict[str, list[HsTransaction]] = defaultdict(list)
    txs = tuple(transactions)
    for tx in txs:
        if tx.stage in {"field-controllers", "supervisory"}:
            by_area[tx.area].append(tx)

    states: list[ControllerState] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.stage != "field-controller":
            continue
        area_txs = tuple(by_area.get(node.area, ()))
        evidence = area_txs[-1].scenario_label if area_txs else "selected_controller"
        states.append(
            ControllerState(
                node.area,
                node.node_id,
                _mode_for(area_txs),
                _routine_for(area_txs),
                _alarm_for(area_txs),
                _setpoints(area_txs),
                _effect_summary(area_txs),
                evidence,
            )
        )
    return tuple(states)
