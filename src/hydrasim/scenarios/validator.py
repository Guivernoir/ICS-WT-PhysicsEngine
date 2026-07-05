"""Scenario validation rules for HydraSim scenario profiles."""

from __future__ import annotations

from .models import ScenarioProfile

SUPPORTED_FUNCTION_CODES = frozenset((3, 4, 5, 6, 16))
EXPECTED_OPERATION_BY_FC = {
    3: "read_holding_registers",
    4: "read_input_registers",
    5: "write_single_coil",
    6: "write_single_register",
    16: "write_multiple_registers",
}


def _has_duplicate(values) -> bool:
    seen = set()
    for value in values:
        if value in seen:
            return True
        seen.add(value)
    return False


def validate_scenario(scenario: ScenarioProfile) -> tuple[str, ...]:
    errors: list[str] = []

    if not scenario.scenario_id:
        errors.append("scenario_id is required")
    if not scenario.name:
        errors.append("name is required")
    if not scenario.purpose:
        errors.append("purpose is required")
    if not scenario.nodes:
        errors.append("at least one node is required")
    if not scenario.transactions:
        errors.append("at least one transaction is required")
    if not scenario.limitations:
        errors.append("at least one limitation is required")

    node_ids = [node.node_id for node in scenario.nodes]
    if _has_duplicate(node_ids):
        errors.append("node_id values must be unique")
    node_lookup = {node.node_id: node for node in scenario.nodes}

    for node in scenario.nodes:
        if not node.node_id:
            errors.append("node_id values must be non-empty")
        if node.modbus_port is not None and not 1 <= node.modbus_port <= 65535:
            errors.append(f"{node.node_id}: modbus_port must be in 1..65535")

    passive_nodes = {
        node.node_id
        for node in scenario.nodes
        if "passive" in node.role.lower() or "observer" in node.role.lower()
    }

    field_point_ids = [point.point_id for point in scenario.field_points]
    if _has_duplicate(field_point_ids):
        errors.append("field point IDs must be unique")
    for point in scenario.field_points:
        if point.endpoint_id not in node_lookup:
            errors.append(f"{point.point_id}: endpoint_id does not reference a node")
        if point.kind not in {"sensor", "actuator"}:
            errors.append(f"{point.point_id}: kind must be sensor or actuator")
        if point.address < 0:
            errors.append(f"{point.point_id}: address must be non-negative")
        if point.width <= 0:
            errors.append(f"{point.point_id}: width must be positive")

    ordinals = [transaction.ordinal for transaction in scenario.transactions]
    if _has_duplicate(ordinals):
        errors.append("transaction ordinals must be unique")

    previous_timestamp = -1
    for transaction in scenario.transactions:
        prefix = f"transaction {transaction.ordinal}"
        if transaction.ordinal <= 0:
            errors.append(f"{prefix}: ordinal must be positive")
        if transaction.timestamp_ms < 0:
            errors.append(f"{prefix}: timestamp_ms must be non-negative")
        if transaction.timestamp_ms < previous_timestamp:
            errors.append(f"{prefix}: timestamps must be monotonic")
        previous_timestamp = transaction.timestamp_ms

        if transaction.actor_id not in node_lookup:
            errors.append(f"{prefix}: actor_id does not reference a node")
        if transaction.server_id not in node_lookup:
            errors.append(f"{prefix}: server_id does not reference a node")
        if transaction.actor_id == transaction.server_id:
            errors.append(f"{prefix}: actor and server must be different nodes")
        if (
            transaction.actor_id in passive_nodes
            or transaction.server_id in passive_nodes
        ):
            errors.append(f"{prefix}: passive observer nodes cannot transact")

        if transaction.function_code not in SUPPORTED_FUNCTION_CODES:
            errors.append(f"{prefix}: unsupported function code")
        expected = EXPECTED_OPERATION_BY_FC.get(transaction.function_code)
        if expected and transaction.operation != expected:
            errors.append(
                f"{prefix}: operation {transaction.operation!r} does not match "
                f"function code {transaction.function_code}"
            )
        if transaction.address < 0:
            errors.append(f"{prefix}: address must be non-negative")
        if transaction.quantity <= 0:
            errors.append(f"{prefix}: quantity must be positive")

        wire_len = len(transaction.wire_values)
        if transaction.function_code in (3, 4) and wire_len:
            errors.append(f"{prefix}: read operations must not include wire_values")
        if transaction.function_code == 5:
            if wire_len != 1:
                errors.append(f"{prefix}: function code 5 requires one wire value")
            elif transaction.wire_values[0] not in (0, 1):
                errors.append(f"{prefix}: coil wire value must be 0 or 1")
        if transaction.function_code == 6 and wire_len != 1:
            errors.append(f"{prefix}: function code 6 requires one wire value")
        if transaction.function_code == 16 and wire_len != transaction.quantity:
            errors.append(
                f"{prefix}: function code 16 requires quantity-matched wire_values"
            )
        for value in transaction.wire_values:
            if not 0 <= int(value) <= 65535:
                errors.append(f"{prefix}: wire_values must be uint16 values")

    return tuple(errors)


def assert_valid_scenario(scenario: ScenarioProfile) -> ScenarioProfile:
    errors = validate_scenario(scenario)
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"invalid scenario {scenario.scenario_id!r}: {joined}")
    return scenario
