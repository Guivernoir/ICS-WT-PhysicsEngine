"""Lab bundle export for staged HydraSim runtime artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evidence.cfd_bundle import build_cfd_lab_artifacts
from .models import RuntimeArtifact
from .pcap import render_hs_pcap_bytes
from .render import (
    render_process_evolution_csv,
    render_process_review_csv,
    render_summary_markdown,
    render_transcript_csv,
)
from .runtime import build_runtime_artifact


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _topology_markdown(artifact: RuntimeArtifact) -> str:
    rows = [
        "| Node | Area | Stage | Role | IPv4 | Protocol |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for node in artifact.active_nodes:
        rows.append(
            f"| {node.node_id} | {node.area} | {node.stage} | "
            f"{node.role} | {node.ipv4} | {node.protocol} |"
        )
    return (
        f"# Topology: {artifact.profile.profile_id}\n\n"
        f"- Area selection: `{artifact.area}`\n"
        f"- Stage selection: `{artifact.stage}`\n"
        f"- Topology profile: `{artifact.profile.topology}`\n"
        f"- Media profile: `{artifact.profile.media}`\n\n" + "\n".join(rows) + "\n"
    )


def _capture_notes(artifact: RuntimeArtifact) -> str:
    return (
        f"# HydraSim Capture Notes: {artifact.profile.profile_id}\n\n"
        "This bundle is deterministic synthetic simulator output.\n\n"
        "## Boundaries\n\n"
        "- This is not an operational capture.\n"
        "- This does not prove field readiness or plant safety.\n"
        "- Topology and media are supplied simulator metadata.\n"
        "- Passive observers should not transmit traffic into the scenario.\n"
    )


def _controller_states_csv(artifact: RuntimeArtifact) -> str:
    rows = [
        "area,controller_id,mode,routine,alarm_state,setpoint_summary,"
        "effect_summary,evidence_label"
    ]
    for state in artifact.controller_states:
        rows.append(
            ",".join(
                (
                    state.area,
                    state.controller_id,
                    state.mode,
                    state.routine,
                    state.alarm_state,
                    state.setpoint_summary.replace(",", ";"),
                    state.effect_summary.replace(",", ";"),
                    state.evidence_label,
                )
            )
        )
    return "\n".join(rows) + "\n"


def export_hs_bundle(
    profile_id: str,
    scenario_id: str,
    area: str,
    stage: str,
    output_dir: str | Path,
    control_system: str | None = None,
    topology: str | None = None,
    media: str | None = None,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"bundle output already exists: {target}")
    target.mkdir(parents=True)

    artifact = build_runtime_artifact(
        profile_id,
        scenario_id,
        area,
        stage,
        control_system=control_system,
        topology=topology,
        media=media,
    )
    summary = render_summary_markdown(artifact).encode("utf-8")
    transcript = render_transcript_csv(artifact).encode("utf-8")
    topology_bytes = _topology_markdown(artifact).encode("utf-8")
    controller_states = _controller_states_csv(artifact).encode("utf-8")
    process_evolution = render_process_evolution_csv(artifact).encode("utf-8")
    process_reviews = render_process_review_csv(artifact).encode("utf-8")
    pcap = render_hs_pcap_bytes(artifact)
    notes = _capture_notes(artifact).encode("utf-8")
    cfd_artifacts = build_cfd_lab_artifacts(artifact)
    manifest = {
        "profile_id": artifact.profile.profile_id,
        "scenario_id": artifact.scenario.scenario_id,
        "area": artifact.area,
        "stage": artifact.stage,
        "source_class": "SyntheticReferencePlant",
        "control_system": artifact.profile.control_system,
        "topology": artifact.profile.topology,
        "media": artifact.profile.media,
        "protocol": artifact.profile.protocol,
        "unit_count": len(artifact.active_units),
        "node_count": len(artifact.active_nodes),
        "transaction_count": len(artifact.transactions),
        "controller_state_count": len(artifact.controller_states),
        "process_evolution_count": len(artifact.process_evolution),
        "process_review_count": len(artifact.process_reviews),
        "cfd_lab_bundle_version": "v2",
        "cfd_lab_artifact_count": len(cfd_artifacts),
        "limitations": list(artifact.limitations),
    }
    artifacts: dict[str, bytes] = {
        "summary.md": summary,
        "transcript.csv": transcript,
        "topology.md": topology_bytes,
        "controller-states.csv": controller_states,
        "process-evolution.csv": process_evolution,
        "process-review.csv": process_reviews,
        "scenario.pcap": pcap,
        "capture-notes.md": notes,
        "manifest.json": (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    }
    artifacts.update(cfd_artifacts)

    written: list[Path] = []
    checksum_lines: list[str] = []
    for name, data in artifacts.items():
        path = target / name
        path.write_bytes(data)
        written.append(path)
        checksum_lines.append(f"{_sha256(data)}  {name}")
    checksum_data = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    checksum_path = target / "checksums.sha256"
    checksum_path.write_bytes(checksum_data)
    written.append(checksum_path)
    return tuple(written)
