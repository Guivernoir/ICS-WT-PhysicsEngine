"""Shared scenario profile models for HydraSim deterministic exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class NetworkNode:
    node_id: str
    label: str
    mac: str
    ipv4: str
    role: str
    modbus_port: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class FieldPoint:
    point_id: str
    label: str
    kind: Literal["sensor", "actuator"]
    register_space: str
    address: int
    width: int
    units: str
    endpoint_id: str
    notes: str


@dataclass(frozen=True)
class ModbusTransaction:
    ordinal: int
    timestamp_ms: int
    actor_id: str
    server_id: str
    function_code: int
    operation: str
    address: int
    quantity: int
    value_summary: str
    response: str
    scenario_label: str
    review_hint: str
    wire_values: Sequence[int] = ()


@dataclass(frozen=True)
class ScenarioProfile:
    scenario_id: str
    name: str
    purpose: str
    nodes: Sequence[NetworkNode]
    field_points: Sequence[FieldPoint]
    transactions: Sequence[ModbusTransaction]
    limitations: Sequence[str]
