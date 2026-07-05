"""Reference Water Plant CFD release-candidate package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from hydrasim.hydraulics.cfd import (
    DigitalTwinValidationGate,
    ExternalReviewCalibrationGate,
    CfdRuntimePerformanceGateRecord,
    build_digital_twin_validation_gate,
    build_external_review_calibration_gate,
    build_runtime_performance_gate,
)

from .models import RuntimeArtifact
from .orchestration import LiveOrchestrationPlan, build_live_orchestration_plan
from .runtime import build_runtime_artifact

EXPECTED_BUNDLE_ARTIFACTS: tuple[str, ...] = (
    "summary.md",
    "transcript.csv",
    "topology.md",
    "controller-states.csv",
    "process-evolution.csv",
    "process-review.csv",
    "scenario.pcap",
    "capture-notes.md",
    "manifest.json",
    "cfd-mesh-geometry.json",
    "cfd-state-timeline.csv",
    "cfd-scalar-fields.csv",
    "cfd-flow-snapshots.csv",
    "checksums.sha256",
)


@dataclass(frozen=True)
class ReleaseCandidateCheck:
    check_id: str
    category: str
    status: str
    evidence: str
    limitations: Sequence[str]

    def validate(self) -> None:
        if not self.check_id:
            raise ValueError("release check_id is required")
        if not self.category:
            raise ValueError(f"{self.check_id}: category is required")
        if self.status not in {"passed", "pending"}:
            raise ValueError(f"{self.check_id}: unsupported status")
        if not self.evidence:
            raise ValueError(f"{self.check_id}: evidence is required")
        if not self.limitations:
            raise ValueError(f"{self.check_id}: limitations are required")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError(f"{self.check_id}: missing validation caveat")


@dataclass(frozen=True)
class ReferenceWaterPlantCfdReleaseCandidate:
    release_id: str
    profile_id: str
    scenario_id: str
    selected_area: str
    full_plant_offline: RuntimeArtifact
    selected_area_full_cell: RuntimeArtifact
    selected_area_live_plan: LiveOrchestrationPlan
    performance_records: Sequence[CfdRuntimePerformanceGateRecord]
    validation_gate: DigitalTwinValidationGate
    review_gate: ExternalReviewCalibrationGate
    expected_bundle_artifacts: Sequence[str]
    checks: Sequence[ReleaseCandidateCheck]
    evidence_status: str
    limitations: Sequence[str]

    def validate(self) -> None:
        if not self.release_id:
            raise ValueError("release_id is required")
        if self.profile_id != "reference-water-plant":
            raise ValueError("HS-35 release must target reference-water-plant")
        if self.full_plant_offline.area != "all":
            raise ValueError("release requires full-plant offline artifact")
        if self.full_plant_offline.stage != "offline-export":
            raise ValueError("full-plant artifact must be offline-export")
        if self.selected_area_full_cell.area != self.selected_area:
            raise ValueError("selected-area artifact does not match release area")
        if self.selected_area_full_cell.stage != "full-cell":
            raise ValueError("selected-area artifact must be full-cell")
        if self.selected_area_live_plan.area != self.selected_area:
            raise ValueError("selected-area live plan does not match release area")
        if not self.selected_area_live_plan.nodes:
            raise ValueError("selected-area live plan requires launchable nodes")
        for record in self.performance_records:
            record.validate()
            if not record.gate_passed:
                raise ValueError(f"{record.preset_id}: performance gate did not pass")
        self.validation_gate.validate()
        self.review_gate.validate()
        if tuple(self.expected_bundle_artifacts) != EXPECTED_BUNDLE_ARTIFACTS:
            raise ValueError("release expected bundle artifact list drifted")
        if not self.checks:
            raise ValueError("release checks are required")
        for check in self.checks:
            check.validate()
        if not all(check.status == "passed" for check in self.checks):
            raise ValueError("release candidate has unresolved checks")
        if (
            self.evidence_status
            != "synthetic_reference_water_plant_cfd_release_candidate"
        ):
            raise ValueError("unsupported release evidence status")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError("release candidate must preserve validation caveat")


def build_reference_water_plant_cfd_release_candidate(
    scenario_id: str = "HS-WTP-002",
    selected_area: str = "disinfection",
) -> ReferenceWaterPlantCfdReleaseCandidate:
    full_plant = build_runtime_artifact(
        "reference-water-plant", scenario_id, "all", "offline-export"
    )
    selected = build_runtime_artifact(
        "reference-water-plant", scenario_id, selected_area, "full-cell"
    )
    live_plan = build_live_orchestration_plan(
        "reference-water-plant",
        scenario_id,
        selected_area,
        "full-cell",
        duration_seconds=30.0,
        startup_delay_seconds=1.0,
    )
    performance = build_runtime_performance_gate(iterations=1)
    validation_gate = build_digital_twin_validation_gate()
    review_gate = build_external_review_calibration_gate()
    checks = _build_release_checks(
        full_plant,
        selected,
        live_plan,
        performance,
        validation_gate,
        review_gate,
    )
    release = ReferenceWaterPlantCfdReleaseCandidate(
        release_id="hs-35-reference-water-plant-cfd-release-candidate",
        profile_id="reference-water-plant",
        scenario_id=scenario_id,
        selected_area=selected_area,
        full_plant_offline=full_plant,
        selected_area_full_cell=selected,
        selected_area_live_plan=live_plan,
        performance_records=performance,
        validation_gate=validation_gate,
        review_gate=review_gate,
        expected_bundle_artifacts=EXPECTED_BUNDLE_ARTIFACTS,
        checks=checks,
        evidence_status="synthetic_reference_water_plant_cfd_release_candidate",
        limitations=(
            "release candidate uses synthetic reference plant evidence",
            "not real-plant validation, certification, commissioning, or safety evidence",
            "selected-area live launch uses common synthetic endpoint runtime",
        ),
    )
    release.validate()
    return release


def _release_limitations() -> tuple[str, ...]:
    return (
        "synthetic release-candidate check",
        "not real-plant validation, certification, commissioning, or safety evidence",
    )


def _build_release_checks(
    full_plant: RuntimeArtifact,
    selected: RuntimeArtifact,
    live_plan: LiveOrchestrationPlan,
    performance: Sequence[CfdRuntimePerformanceGateRecord],
    validation_gate: DigitalTwinValidationGate,
    review_gate: ExternalReviewCalibrationGate,
) -> tuple[ReleaseCandidateCheck, ...]:
    return (
        ReleaseCandidateCheck(
            "rc-profile",
            "profile",
            "passed",
            f"profile={full_plant.profile.profile_id}; areas={len(full_plant.profile.areas)}",
            _release_limitations(),
        ),
        ReleaseCandidateCheck(
            "rc-full-plant-offline",
            "offline_export",
            "passed",
            f"transactions={len(full_plant.transactions)}; "
            f"process_truth={len(full_plant.process_evolution)}",
            _release_limitations(),
        ),
        ReleaseCandidateCheck(
            "rc-selected-area-full-cell",
            "selected_area_runtime",
            "passed",
            f"area={selected.area}; nodes={len(selected.active_nodes)}; "
            f"transactions={len(selected.transactions)}",
            _release_limitations(),
        ),
        ReleaseCandidateCheck(
            "rc-selected-area-live-plan",
            "live_plan",
            "passed",
            f"area={live_plan.area}; launch_nodes={len(live_plan.nodes)}",
            _release_limitations(),
        ),
        ReleaseCandidateCheck(
            "rc-cfd-performance",
            "cfd_gate",
            "passed",
            f"records={len(performance)}; all_passed={all(item.gate_passed for item in performance)}",
            _release_limitations(),
        ),
        ReleaseCandidateCheck(
            "rc-digital-twin-validation",
            "cfd_gate",
            "passed",
            f"implementation_verified={validation_gate.implementation_verified}; "
            f"status={validation_gate.real_plant_validation_status}",
            _release_limitations(),
        ),
        ReleaseCandidateCheck(
            "rc-external-review-gate",
            "evidence_gate",
            "passed",
            f"disposition={review_gate.evidence_disposition}; "
            f"upgraded={review_gate.model_status_upgraded}",
            _release_limitations(),
        ),
    )


def render_release_candidate_json(
    release: ReferenceWaterPlantCfdReleaseCandidate,
) -> str:
    release.validate()
    payload = {
        "release_id": release.release_id,
        "evidence_status": release.evidence_status,
        "profile_id": release.profile_id,
        "scenario_id": release.scenario_id,
        "selected_area": release.selected_area,
        "full_plant_offline": {
            "area": release.full_plant_offline.area,
            "stage": release.full_plant_offline.stage,
            "transaction_count": len(release.full_plant_offline.transactions),
            "process_evolution_count": len(
                release.full_plant_offline.process_evolution
            ),
            "process_review_count": len(release.full_plant_offline.process_reviews),
        },
        "selected_area_full_cell": {
            "area": release.selected_area_full_cell.area,
            "stage": release.selected_area_full_cell.stage,
            "node_count": len(release.selected_area_full_cell.active_nodes),
            "transaction_count": len(release.selected_area_full_cell.transactions),
        },
        "selected_area_live_plan": {
            "area": release.selected_area_live_plan.area,
            "stage": release.selected_area_live_plan.stage,
            "launch_node_count": len(release.selected_area_live_plan.nodes),
        },
        "expected_bundle_artifacts": list(release.expected_bundle_artifacts),
        "performance_record_count": len(release.performance_records),
        "validation_status": release.validation_gate.real_plant_validation_status,
        "review_gate_disposition": release.review_gate.evidence_disposition,
        "checks": [
            {
                "check_id": check.check_id,
                "category": check.category,
                "status": check.status,
                "evidence": check.evidence,
            }
            for check in release.checks
        ],
        "limitations": list(release.limitations),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_release_candidate_markdown(
    release: ReferenceWaterPlantCfdReleaseCandidate,
) -> str:
    release.validate()
    rows = [
        "| Check | Category | Status | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for check in release.checks:
        rows.append(
            f"| `{check.check_id}` | `{check.category}` | `{check.status}` | {check.evidence} |"
        )
    artifacts = "\n".join(f"- `{item}`" for item in release.expected_bundle_artifacts)
    limitations = "\n".join(f"- {item}" for item in release.limitations)
    return (
        "# Reference Water Plant CFD Release Candidate\n\n"
        f"- Release ID: `{release.release_id}`\n"
        f"- Evidence status: `{release.evidence_status}`\n"
        f"- Profile: `{release.profile_id}`\n"
        f"- Scenario: `{release.scenario_id}`\n"
        f"- Selected live area: `{release.selected_area}`\n"
        f"- Validation status: `{release.validation_gate.real_plant_validation_status}`\n"
        f"- Review gate disposition: `{release.review_gate.evidence_disposition}`\n\n"
        "## Verification Checks\n\n"
        + "\n".join(rows)
        + "\n\n## Expected Bundle Artifacts\n\n"
        + artifacts
        + "\n\n## Limitations\n\n"
        + limitations
        + "\n"
    )
