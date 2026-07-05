"""Scenario execution primitives for Modbus command-center clients."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .models import ModbusTransaction, ScenarioProfile


@dataclass(frozen=True)
class CommandResult:
    ordinal: int
    actor_id: str
    function_code: int
    address: int
    status: str
    detail: str


def _response_status(response: Any) -> str:
    if response is None:
        return "ok"
    is_error = getattr(response, "isError", None)
    if callable(is_error) and is_error():
        return "error"
    return "ok"


def execute_transaction(
    client: Any, transaction: ModbusTransaction, unit_id: int = 1
) -> CommandResult:
    fc = transaction.function_code
    response = None

    if fc == 3:
        response = client.read_holding_registers(
            transaction.address, count=transaction.quantity, device_id=unit_id
        )
    elif fc == 4:
        response = client.read_input_registers(
            transaction.address, count=transaction.quantity, device_id=unit_id
        )
    elif fc == 5:
        value = bool(transaction.wire_values[0]) if transaction.wire_values else True
        response = client.write_coil(transaction.address, value, device_id=unit_id)
    elif fc == 6:
        if not transaction.wire_values:
            raise ValueError("function code 6 requires one wire value")
        response = client.write_register(
            transaction.address, transaction.wire_values[0], device_id=unit_id
        )
    elif fc == 16:
        if not transaction.wire_values:
            raise ValueError("function code 16 requires wire values")
        response = client.write_registers(
            transaction.address, list(transaction.wire_values), device_id=unit_id
        )
    else:
        raise ValueError(f"unsupported scenario function code: {fc}")

    return CommandResult(
        transaction.ordinal,
        transaction.actor_id,
        fc,
        transaction.address,
        _response_status(response),
        transaction.scenario_label,
    )


def run_scenario(
    client: Any,
    scenario: ScenarioProfile,
    unit_id: int = 1,
    time_scale: float = 0.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[CommandResult, ...]:
    results: list[CommandResult] = []
    previous_ms = 0
    for transaction in scenario.transactions:
        delay_ms = max(0, transaction.timestamp_ms - previous_ms)
        if time_scale > 0.0 and delay_ms:
            sleeper((delay_ms / 1000.0) / time_scale)
        results.append(execute_transaction(client, transaction, unit_id=unit_id))
        previous_ms = transaction.timestamp_ms
    return tuple(results)


def run_scenario_with_clients(
    clients_by_server_id: dict[str, Any],
    scenario: ScenarioProfile,
    unit_id: int = 1,
    time_scale: float = 0.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[CommandResult, ...]:
    results: list[CommandResult] = []
    previous_ms = 0
    for transaction in scenario.transactions:
        client = clients_by_server_id.get(transaction.server_id)
        if client is None:
            raise ValueError(f"missing client for server_id {transaction.server_id!r}")
        delay_ms = max(0, transaction.timestamp_ms - previous_ms)
        if time_scale > 0.0 and delay_ms:
            sleeper((delay_ms / 1000.0) / time_scale)
        results.append(execute_transaction(client, transaction, unit_id=unit_id))
        previous_ms = transaction.timestamp_ms
    return tuple(results)


def render_results_csv(results: Sequence[CommandResult]) -> str:
    rows = ["ordinal,actor_id,function_code,address,status,detail"]
    for result in results:
        rows.append(
            ",".join(
                (
                    str(result.ordinal),
                    result.actor_id,
                    str(result.function_code),
                    str(result.address),
                    result.status,
                    result.detail,
                )
            )
        )
    return "\n".join(rows) + "\n"
