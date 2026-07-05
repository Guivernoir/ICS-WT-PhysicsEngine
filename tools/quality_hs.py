"""Reference Water Plant quality checks."""

from __future__ import annotations

from pathlib import Path


def check_hs_docs(root: Path, errors: list[str]) -> None:
    from hydrasim.plant import (
        build_reference_water_plant_cfd_release_candidate,
        build_runtime_artifact,
        get_profile,
        profile_ids,
        render_hs_pcap_bytes,
        render_process_evolution_csv,
        render_process_review_csv,
        render_release_candidate_json,
        render_transcript_csv,
        scenario_ids,
        validate_profile,
    )
    from hydrasim.plant.evidence.cfd_bundle import build_cfd_lab_artifacts

    public_docs = (root / "README.md").read_text("utf-8")
    for profile_id in profile_ids():
        problems = validate_profile(get_profile(profile_id))
        if problems:
            errors.append(f"{profile_id}: profile validation failed: {problems}")
        if profile_id not in public_docs:
            errors.append(f"{profile_id}: missing from public README")
    for scenario_id in scenario_ids():
        if scenario_id not in public_docs:
            errors.append(f"{scenario_id}: missing from public README")
        artifact = build_runtime_artifact(
            "reference-water-plant",
            scenario_id,
            "all",
            "offline-export",
        )
        _check_process_evidence(scenario_id, artifact, errors)

    artifact = build_runtime_artifact(
        "reference-water-plant",
        "HS-WTP-002",
        "all",
        "offline-export",
    )
    if render_transcript_csv(artifact) != render_transcript_csv(artifact):
        errors.append("reference-water-plant: transcript output is not deterministic")
    if render_hs_pcap_bytes(artifact) != render_hs_pcap_bytes(artifact):
        errors.append("reference-water-plant: PCAP output is not deterministic")
    if render_process_evolution_csv(artifact) != render_process_evolution_csv(artifact):
        errors.append("reference-water-plant: process truth is not deterministic")
    if render_process_review_csv(artifact) != render_process_review_csv(artifact):
        errors.append("reference-water-plant: process review is not deterministic")
    cfd_artifacts = build_cfd_lab_artifacts(artifact)
    if cfd_artifacts != build_cfd_lab_artifacts(artifact):
        errors.append("reference-water-plant: CFD lab artifacts are not deterministic")
    for name in (
        "cfd-mesh-geometry.json",
        "cfd-state-timeline.csv",
        "cfd-scalar-fields.csv",
        "cfd-flow-snapshots.csv",
    ):
        data = cfd_artifacts.get(name, b"").decode("utf-8")
        if "synthetic_cfd_lab_bundle_v2" not in data:
            errors.append(f"reference-water-plant: {name} missing HS-32 evidence")
    release = build_reference_water_plant_cfd_release_candidate()
    release.validate()
    if (
        release.evidence_status
        != "synthetic_reference_water_plant_cfd_release_candidate"
    ):
        errors.append("HS-35 release candidate: unsupported evidence status")
    if release.full_plant_offline.area != "all":
        errors.append("HS-35 release candidate: missing full-plant offline artifact")
    if release.selected_area_full_cell.stage != "full-cell":
        errors.append(
            "HS-35 release candidate: missing selected-area full-cell artifact"
        )
    if not release.selected_area_live_plan.nodes:
        errors.append("HS-35 release candidate: missing selected-area live plan")
    rendered_release = render_release_candidate_json(release)
    if rendered_release != render_release_candidate_json(release):
        errors.append("HS-35 release candidate: JSON output is not deterministic")
    if "synthetic_reference_water_plant_cfd_release_candidate" not in rendered_release:
        errors.append("HS-35 release candidate: missing evidence status")
    for phrase in _required_public_phrases():
        if phrase not in public_docs:
            errors.append(f"HydraSim public README: missing {phrase!r}")


def _check_process_evidence(scenario_id: str, artifact, errors: list[str]) -> None:
    from hydrasim.plant import render_process_evolution_csv, render_process_review_csv

    if not artifact.process_evolution:
        errors.append(f"{scenario_id}: missing CFD process evolution")
    for record in artifact.process_evolution:
        try:
            record.validate()
        except ValueError as exc:
            errors.append(f"{scenario_id}: invalid process evolution: {exc}")
    process_csv = render_process_evolution_csv(artifact)
    if "synthetic_cfd_process_truth" not in process_csv:
        errors.append(f"{scenario_id}: missing process truth evidence status")
    if not artifact.process_reviews:
        errors.append(f"{scenario_id}: missing scenario process review")
    for review in artifact.process_reviews:
        try:
            review.validate()
        except ValueError as exc:
            errors.append(f"{scenario_id}: invalid scenario process review: {exc}")
    review_csv = render_process_review_csv(artifact)
    if "synthetic_scenario_process_review" not in review_csv:
        errors.append(f"{scenario_id}: missing process review evidence status")
    if "Do not claim real plant validation" not in review_csv:
        errors.append(f"{scenario_id}: missing process review claim boundary")


def _required_public_phrases() -> tuple[str, ...]:
    return (
        "CFD Lab Bundle v2",
        "Digital-Twin Validation Gate",
        "External Review And Calibration Evidence Gate",
        "Reference Water Plant CFD release-candidate",
        "synthetic",
        "externally validated",
    )
