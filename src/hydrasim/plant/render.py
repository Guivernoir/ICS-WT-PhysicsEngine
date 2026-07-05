"""Render staged HydraSim runtime artifacts."""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import RuntimeArtifact


def _csv(value: object) -> str:
    text = str(value)
    if "," in text or "\n" in text or '"' in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(output)


def render_transcript_csv(artifact: RuntimeArtifact) -> str:
    rows = [
        "ordinal,timestamp_ms,actor_id,target_id,stage,area,function_code,"
        "operation,address,quantity,value_summary,response,scenario_label,"
        "review_hint,wire_values"
    ]
    for tx in artifact.transactions:
        rows.append(
            ",".join(
                _csv(value)
                for value in (
                    tx.ordinal,
                    tx.timestamp_ms,
                    tx.actor_id,
                    tx.target_id,
                    tx.stage,
                    tx.area,
                    tx.function_code,
                    tx.operation,
                    tx.address,
                    tx.quantity,
                    tx.value_summary,
                    tx.response,
                    tx.scenario_label,
                    tx.review_hint,
                    " ".join(str(item) for item in tx.wire_values),
                )
            )
        )
    return "\n".join(rows) + "\n"


def render_process_evolution_csv(artifact: RuntimeArtifact) -> str:
    rows = [
        "scenario_id,area,scalar_name,start_value,end_value,trend,cfd_basis,"
        "evidence_status,limitations"
    ]
    for record in artifact.process_evolution:
        rows.append(
            ",".join(
                _csv(value)
                for value in (
                    record.scenario_id,
                    record.area,
                    record.scalar_name,
                    f"{record.start_value:.6g}",
                    f"{record.end_value:.6g}",
                    record.trend,
                    record.cfd_basis,
                    record.evidence_status,
                    "; ".join(record.limitations),
                )
            )
        )
    return "\n".join(rows) + "\n"


def render_process_review_csv(artifact: RuntimeArtifact) -> str:
    rows = [
        "scenario_id,area,process_truth,observable_network_effects,"
        "operator_historian_effects,review_questions,must_not_claim,"
        "evidence_status"
    ]
    for record in artifact.process_reviews:
        rows.append(
            ",".join(
                _csv(value)
                for value in (
                    record.scenario_id,
                    record.area,
                    record.process_truth,
                    record.observable_network_effects,
                    record.operator_historian_effects,
                    "; ".join(record.review_questions),
                    "; ".join(record.must_not_claim),
                    record.evidence_status,
                )
            )
        )
    return "\n".join(rows) + "\n"


def render_summary_markdown(artifact: RuntimeArtifact) -> str:
    unit_rows = (
        (unit.unit_id, unit.area, unit.kind, unit.label)
        for unit in artifact.active_units
    )
    node_rows = (
        (
            node.node_id,
            node.area,
            node.stage,
            node.role,
            node.ipv4,
            node.modbus_port or "-",
        )
        for node in artifact.active_nodes
    )
    tx_rows = (
        (
            tx.ordinal,
            tx.timestamp_ms,
            tx.stage,
            tx.area,
            tx.actor_id,
            tx.target_id,
            tx.function_code,
            tx.operation,
            tx.response,
            tx.review_hint or "normal_context",
        )
        for tx in artifact.transactions
    )
    controller_rows = (
        (
            state.controller_id,
            state.area,
            state.mode,
            state.routine,
            state.alarm_state,
            state.setpoint_summary,
            state.evidence_label,
        )
        for state in artifact.controller_states
    )
    process_rows = (
        (
            record.area,
            record.scalar_name,
            f"{record.start_value:.4g}",
            f"{record.end_value:.4g}",
            record.trend,
            record.evidence_status,
        )
        for record in artifact.process_evolution
    )
    review_rows = (
        (
            record.area,
            record.process_truth,
            record.observable_network_effects,
            record.operator_historian_effects,
            record.evidence_status,
        )
        for record in artifact.process_reviews
    )
    parts = [
        f"# {artifact.profile.name}: {artifact.scenario.name}",
        "",
        artifact.scenario.purpose,
        "",
        "## Runtime Selection",
        "",
        f"- Profile: `{artifact.profile.profile_id}`",
        f"- Scenario: `{artifact.scenario.scenario_id}`",
        f"- Area: `{artifact.area}`",
        f"- Stage: `{artifact.stage}`",
        f"- Control system: `{artifact.profile.control_system}`",
        f"- Topology: `{artifact.profile.topology}`",
        f"- Media: `{artifact.profile.media}`",
        "",
        "## Active Units",
        "",
        _table(("Unit", "Area", "Kind", "Label"), unit_rows),
        "",
        "## Active Nodes",
        "",
        _table(("Node", "Area", "Stage", "Role", "IPv4", "Port"), node_rows),
        "",
        "## Transactions",
        "",
        _table(
            (
                "Ordinal",
                "Time ms",
                "Stage",
                "Area",
                "Actor",
                "Target",
                "FC",
                "Operation",
                "Response",
                "Review hint",
            ),
            tx_rows,
        ),
        "",
        "## Controller Logic State",
        "",
        _table(
            (
                "Controller",
                "Area",
                "Mode",
                "Routine",
                "Alarm state",
                "Setpoint summary",
                "Evidence label",
            ),
            controller_rows,
        ),
        "",
        "## CFD Process Evolution",
        "",
        _table(
            (
                "Area",
                "Scalar",
                "Start",
                "End",
                "Trend",
                "Evidence status",
            ),
            process_rows,
        ),
        "",
        "## Scenario Process-Truth Review",
        "",
        _table(
            (
                "Area",
                "Process truth",
                "Network effects",
                "Operator/historian effects",
                "Evidence status",
            ),
            review_rows,
        ),
        "",
        "## Limitations",
        "",
    ]
    parts.extend(f"- {item}" for item in artifact.limitations)
    return "\n".join(parts) + "\n"
