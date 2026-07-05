"""Built-in multi-endpoint topology scenarios."""

from __future__ import annotations

from .models import FieldPoint, ModbusTransaction, NetworkNode, ScenarioProfile
from .profile_common import (
    COMMON_LIMITATIONS,
    FIELD_POINTS,
    NODES,
    float32_registers,
    tx_to,
)

SMART_FIELD_NODES: tuple[NetworkNode, ...] = (
    NetworkNode(
        "ph_probe_a",
        "Networked pH probe",
        "02:00:00:00:31:30",
        "198.51.100.30",
        "smart sensor Modbus TCP endpoint",
        502,
        "Optional modeled network identity for a smart field sensor.",
    ),
    NetworkNode(
        "chlorine_pump_a",
        "Networked chlorine dosing pump",
        "02:00:00:00:31:31",
        "198.51.100.31",
        "smart actuator Modbus TCP endpoint",
        502,
        "Optional modeled network identity for a smart field actuator.",
    ),
)

SMART_FIELD_POINTS: tuple[FieldPoint, ...] = (
    FieldPoint(
        "smart_pH_outlet",
        "Networked outlet pH probe reading",
        "sensor",
        "input_register",
        0,
        2,
        "pH",
        "ph_probe_a",
        "Supplied smart-field sensor with its own modeled network endpoint.",
    ),
    FieldPoint(
        "smart_chlorine_pump_setpoint",
        "Networked chlorine pump setpoint",
        "actuator",
        "holding_register",
        0,
        2,
        "L/min",
        "chlorine_pump_a",
        "Supplied smart-field actuator with its own modeled network endpoint.",
    ),
)

HYDRA_004_TRANSACTIONS: tuple[ModbusTransaction, ...] = (
    tx_to(
        1,
        0,
        "hmi",
        "hydra_a",
        4,
        "read_input_registers",
        0,
        12,
        "",
        "ok",
        "process_endpoint_poll",
    ),
    tx_to(
        2,
        250,
        "historian",
        "ph_probe_a",
        4,
        "read_input_registers",
        0,
        2,
        "",
        "ok",
        "smart_sensor_poll",
    ),
    tx_to(
        3,
        500,
        "hmi",
        "hydra_a",
        16,
        "write_multiple_registers",
        4,
        2,
        "inlet_flow=6.0",
        "ok",
        "operator_write",
        wire_values=float32_registers(6.0),
    ),
    tx_to(
        4,
        750,
        "maintenance",
        "chlorine_pump_a",
        3,
        "read_holding_registers",
        0,
        2,
        "",
        "ok",
        "smart_actuator_check",
    ),
    tx_to(
        5,
        1000,
        "maintenance",
        "chlorine_pump_a",
        16,
        "write_multiple_registers",
        0,
        2,
        "chlorine_flow=0.3",
        "ok",
        "smart_actuator_adjustment",
        wire_values=float32_registers(0.3),
    ),
    tx_to(
        6,
        1250,
        "historian",
        "hydra_a",
        4,
        "read_input_registers",
        0,
        16,
        "",
        "ok",
        "historian_poll",
    ),
)

TOPOLOGY_SCENARIOS: tuple[ScenarioProfile, ...] = (
    ScenarioProfile(
        "MVP-MB-HYDRA-004",
        "HydraSim smart-field multi-node topology",
        "Process endpoint plus explicitly modeled networked field sensor and actuator.",
        NODES + SMART_FIELD_NODES,
        FIELD_POINTS + SMART_FIELD_POINTS,
        HYDRA_004_TRANSACTIONS,
        COMMON_LIMITATIONS
        + (
            "Smart field endpoints are explicitly modeled synthetic identities.",
            "Live replay requires explicit endpoint targets for each server node.",
        ),
    ),
)
