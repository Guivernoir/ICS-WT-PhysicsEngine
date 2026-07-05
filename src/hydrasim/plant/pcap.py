"""PCAP rendering adapter for staged HydraSim artifacts."""

from __future__ import annotations

from hydrasim.scenarios.models import (
    ModbusTransaction,
    NetworkNode,
    ScenarioProfile,
)
from hydrasim.scenarios.pcap import render_pcap_bytes

from .models import RuntimeArtifact


def _network_node(node) -> NetworkNode:
    return NetworkNode(
        node.node_id,
        node.label,
        node.mac,
        node.ipv4,
        node.role,
        node.modbus_port,
        node.notes,
    )


def _transaction(tx) -> ModbusTransaction:
    return ModbusTransaction(
        tx.ordinal,
        tx.timestamp_ms,
        tx.actor_id,
        tx.target_id,
        tx.function_code,
        tx.operation,
        tx.address,
        tx.quantity,
        tx.value_summary,
        tx.response,
        tx.scenario_label,
        tx.review_hint,
        tx.wire_values,
    )


def render_hs_pcap_bytes(artifact: RuntimeArtifact) -> bytes:
    scenario = ScenarioProfile(
        artifact.scenario.scenario_id,
        artifact.scenario.name,
        artifact.scenario.purpose,
        tuple(_network_node(node) for node in artifact.active_nodes),
        (),
        tuple(_transaction(tx) for tx in artifact.transactions),
        artifact.limitations,
    )
    return render_pcap_bytes(scenario)
