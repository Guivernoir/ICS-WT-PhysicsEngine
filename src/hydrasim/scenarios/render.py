"""Render deterministic scenario profiles for external tooling."""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import ScenarioProfile


def _csv_cell(value: object) -> str:
    text = str(value)
    if "," in text or "\n" in text or '"' in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def render_transcript_csv(scenario: ScenarioProfile) -> str:
    header = (
        "ordinal,timestamp_ms,actor_id,server_id,function_code,operation,"
        "address,quantity,value_summary,response,scenario_label,review_hint,wire_values"
    )
    rows = [header]
    for tx in scenario.transactions:
        rows.append(
            ",".join(
                _csv_cell(value)
                for value in (
                    tx.ordinal,
                    tx.timestamp_ms,
                    tx.actor_id,
                    tx.server_id,
                    tx.function_code,
                    tx.operation,
                    tx.address,
                    tx.quantity,
                    tx.value_summary,
                    tx.response,
                    tx.scenario_label,
                    tx.review_hint,
                    " ".join(str(value) for value in tx.wire_values),
                )
            )
        )
    return "\n".join(rows) + "\n"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(output)


def render_markdown_summary(scenario: ScenarioProfile) -> str:
    node_rows = (
        (node.node_id, node.label, node.mac, node.ipv4, node.role)
        for node in scenario.nodes
    )
    field_rows = (
        (
            point.point_id,
            point.label,
            point.kind,
            point.register_space,
            point.address,
            point.units,
            point.endpoint_id,
        )
        for point in scenario.field_points
    )
    tx_rows = (
        (
            tx.ordinal,
            tx.timestamp_ms,
            tx.actor_id,
            tx.function_code,
            tx.operation,
            tx.address,
            tx.response,
            tx.review_hint or "normal_context",
            " ".join(str(value) for value in tx.wire_values) or "-",
        )
        for tx in scenario.transactions
    )
    parts = [
        f"# {scenario.scenario_id}: {scenario.name}",
        "",
        scenario.purpose,
        "",
        "## Network Nodes",
        "",
        _markdown_table(("ID", "Label", "MAC", "IPv4", "Role"), node_rows),
        "",
        "## Supplied Simulated Field Points",
        "",
        _markdown_table(
            ("ID", "Label", "Kind", "Space", "Address", "Units", "Endpoint"),
            field_rows,
        ),
        "",
        "## Transactions",
        "",
        _markdown_table(
            (
                "Ordinal",
                "Time ms",
                "Actor",
                "FC",
                "Operation",
                "Address",
                "Response",
                "Review hint",
                "Wire values",
            ),
            tx_rows,
        ),
        "",
        "## Limitations",
        "",
    ]
    parts.extend(f"- {limitation}" for limitation in scenario.limitations)
    return "\n".join(parts) + "\n"
