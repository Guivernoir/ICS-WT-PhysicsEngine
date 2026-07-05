"""Shared profile data and constructors for built-in scenarios."""

from __future__ import annotations

import struct

from .models import FieldPoint, ModbusTransaction, NetworkNode

NODES: tuple[NetworkNode, ...] = (
    NetworkNode(
        "hmi",
        "Operator HMI client",
        "02:00:00:00:30:10",
        "198.51.100.10",
        "normal operator client",
    ),
    NetworkNode(
        "historian",
        "Historian-like poller",
        "02:00:00:00:30:11",
        "198.51.100.11",
        "read-only process historian style client",
    ),
    NetworkNode(
        "maintenance",
        "Engineering and maintenance workstation",
        "02:00:00:00:30:12",
        "198.51.100.12",
        "maintenance-window client",
    ),
    NetworkNode(
        "hydra_a",
        "HydraSim primary process endpoint",
        "02:00:00:00:30:20",
        "198.51.100.20",
        "Modbus TCP process endpoint",
        502,
        "Exposes the simulated water-treatment register map.",
    ),
    NetworkNode(
        "unknown",
        "Unknown client",
        "02:00:00:00:30:66",
        "198.51.100.66",
        "authorization unknown review candidate",
    ),
    NetworkNode(
        "observer",
        "Passive capture observer",
        "not-transmitting",
        "not-transmitting",
        "passive observer only",
        None,
        "The observer should not appear as a Modbus participant in normal traces.",
    ),
)

FIELD_POINTS: tuple[FieldPoint, ...] = (
    FieldPoint(
        "pH_inlet",
        "Inlet pH sensor",
        "sensor",
        "input_register",
        0,
        2,
        "pH",
        "hydra_a",
        "Supplied simulated field point behind the process endpoint.",
    ),
    FieldPoint(
        "pH_outlet",
        "Outlet pH sensor",
        "sensor",
        "input_register",
        4,
        2,
        "pH",
        "hydra_a",
        "Supplied simulated field point behind the process endpoint.",
    ),
    FieldPoint(
        "chlorine_outlet",
        "Outlet chlorine sensor",
        "sensor",
        "input_register",
        8,
        2,
        "mg/L",
        "hydra_a",
        "Supplied simulated field point behind the process endpoint.",
    ),
    FieldPoint(
        "flow_main",
        "Main flow sensor",
        "sensor",
        "input_register",
        10,
        2,
        "L/min",
        "hydra_a",
        "Supplied simulated field point behind the process endpoint.",
    ),
    FieldPoint(
        "acid_valve",
        "Acid dosing valve actuator",
        "actuator",
        "holding_register",
        0,
        2,
        "L/min",
        "hydra_a",
        "Writable simulated field point behind the process endpoint.",
    ),
    FieldPoint(
        "chlorine_pump",
        "Chlorine dosing pump actuator",
        "actuator",
        "holding_register",
        2,
        2,
        "L/min",
        "hydra_a",
        "Writable simulated field point behind the process endpoint.",
    ),
    FieldPoint(
        "inlet_valve",
        "Main inlet valve actuator",
        "actuator",
        "holding_register",
        4,
        2,
        "L/min",
        "hydra_a",
        "Writable simulated field point behind the process endpoint.",
    ),
)

COMMON_LIMITATIONS: tuple[str, ...] = (
    "HydraSim traffic is synthetic process simulation, not operational evidence.",
    "Passive observers are modeled as listeners and must not transmit traffic.",
    "Field points are supplied simulated metadata, not discovered network assets.",
    "Unknown-host activity deserves review but is not an incident or attack claim.",
)


def float32_registers(value: float) -> tuple[int, int]:
    packed = struct.pack(">f", value)
    return struct.unpack(">HH", packed)


def tx(
    ordinal: int,
    timestamp_ms: int,
    actor_id: str,
    function_code: int,
    operation: str,
    address: int,
    quantity: int,
    value_summary: str,
    response: str,
    scenario_label: str,
    review_hint: str = "",
    wire_values: tuple[int, ...] = (),
) -> ModbusTransaction:
    return tx_to(
        ordinal,
        timestamp_ms,
        actor_id,
        "hydra_a",
        function_code,
        operation,
        address,
        quantity,
        value_summary,
        response,
        scenario_label,
        review_hint,
        wire_values,
    )


def tx_to(
    ordinal: int,
    timestamp_ms: int,
    actor_id: str,
    server_id: str,
    function_code: int,
    operation: str,
    address: int,
    quantity: int,
    value_summary: str,
    response: str,
    scenario_label: str,
    review_hint: str = "",
    wire_values: tuple[int, ...] = (),
) -> ModbusTransaction:
    return ModbusTransaction(
        ordinal,
        timestamp_ms,
        actor_id,
        server_id,
        function_code,
        operation,
        address,
        quantity,
        value_summary,
        response,
        scenario_label,
        review_hint,
        wire_values,
    )
