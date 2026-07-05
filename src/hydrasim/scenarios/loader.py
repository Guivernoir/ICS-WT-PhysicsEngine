"""Load custom scenario profiles from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import FieldPoint, ModbusTransaction, NetworkNode, ScenarioProfile
from .validator import assert_valid_scenario


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required scenario field: {key}")
    return mapping[key]


def _str(mapping: dict[str, Any], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int(mapping: dict[str, Any], key: str) -> int:
    value = _required(mapping, key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _wire_values(mapping: dict[str, Any]) -> tuple[int, ...]:
    values = mapping.get("wire_values", [])
    if not isinstance(values, list):
        raise ValueError("wire_values must be a list of integers")
    output = []
    for value in values:
        if not isinstance(value, int) or not 0 <= value <= 65535:
            raise ValueError("wire_values entries must be uint16 integers")
        output.append(value)
    return tuple(output)


def _node(item: dict[str, Any]) -> NetworkNode:
    port = item.get("modbus_port")
    if port is not None and not isinstance(port, int):
        raise ValueError("modbus_port must be an integer when present")
    return NetworkNode(
        _str(item, "node_id"),
        _str(item, "label"),
        _str(item, "mac"),
        _str(item, "ipv4"),
        _str(item, "role"),
        port,
        str(item.get("notes", "")),
    )


def _field_point(item: dict[str, Any]) -> FieldPoint:
    kind = _str(item, "kind")
    if kind not in {"sensor", "actuator"}:
        raise ValueError("field point kind must be sensor or actuator")
    return FieldPoint(
        _str(item, "point_id"),
        _str(item, "label"),
        kind,  # type: ignore[arg-type]
        _str(item, "register_space"),
        _int(item, "address"),
        _int(item, "width"),
        _str(item, "units"),
        _str(item, "endpoint_id"),
        _str(item, "notes"),
    )


def _transaction(item: dict[str, Any]) -> ModbusTransaction:
    return ModbusTransaction(
        _int(item, "ordinal"),
        _int(item, "timestamp_ms"),
        _str(item, "actor_id"),
        _str(item, "server_id"),
        _int(item, "function_code"),
        _str(item, "operation"),
        _int(item, "address"),
        _int(item, "quantity"),
        str(item.get("value_summary", "")),
        _str(item, "response"),
        _str(item, "scenario_label"),
        str(item.get("review_hint", "")),
        _wire_values(item),
    )


def scenario_from_mapping(data: dict[str, Any]) -> ScenarioProfile:
    nodes = data.get("nodes")
    field_points = data.get("field_points", [])
    transactions = data.get("transactions")
    limitations = data.get("limitations", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("nodes must be a non-empty list")
    if not isinstance(field_points, list):
        raise ValueError("field_points must be a list")
    if not isinstance(transactions, list) or not transactions:
        raise ValueError("transactions must be a non-empty list")
    if not isinstance(limitations, list):
        raise ValueError("limitations must be a list")

    return assert_valid_scenario(
        ScenarioProfile(
            _str(data, "scenario_id"),
            _str(data, "name"),
            _str(data, "purpose"),
            tuple(_node(item) for item in nodes),
            tuple(_field_point(item) for item in field_points),
            tuple(_transaction(item) for item in transactions),
            tuple(str(item) for item in limitations),
        )
    )


def load_scenario_json(path: str | Path) -> ScenarioProfile:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("custom scenario JSON root must be an object")
    return scenario_from_mapping(data)
