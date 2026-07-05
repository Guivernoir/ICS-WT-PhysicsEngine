"""External review and calibration evidence gate records."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..digital_twin import CalibrationAssessment
from .validation import (
    DigitalTwinValidationGate,
    ExternalCfdComparisonRecord,
    build_digital_twin_validation_gate,
)


@dataclass(frozen=True)
class ExternalReviewEvidenceRecord:
    review_id: str
    reviewer_role: str
    evidence_reference: str
    review_scope: tuple[str, ...]
    observations: tuple[str, ...]
    accepted_recommendations: tuple[str, ...] = ()
    evidence_status: str = "review_recorded_not_validated"
    limitations: tuple[str, ...] = (
        "external review evidence is not real-plant validation by itself",
    )

    def validate(self) -> None:
        if not self.review_id:
            raise ValueError("review_id is required")
        if not self.reviewer_role:
            raise ValueError(f"{self.review_id}: reviewer_role is required")
        if not self.evidence_reference:
            raise ValueError(f"{self.review_id}: evidence_reference is required")
        if not self.review_scope:
            raise ValueError(f"{self.review_id}: review_scope is required")
        if not self.observations:
            raise ValueError(f"{self.review_id}: observations are required")
        if self.evidence_status != "review_recorded_not_validated":
            raise ValueError(f"{self.review_id}: unsupported evidence status")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError(f"{self.review_id}: missing validation caveat")


@dataclass(frozen=True)
class ExternalReviewCalibrationGate:
    gate_id: str
    validation_gate: DigitalTwinValidationGate
    review_records: tuple[ExternalReviewEvidenceRecord, ...]
    calibration_assessments: tuple[CalibrationAssessment, ...]
    evidence_status: str
    evidence_disposition: str
    model_status_upgraded: bool
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.gate_id:
            raise ValueError("review/calibration gate_id is required")
        self.validation_gate.validate()
        for record in self.review_records:
            record.validate()
        for assessment in self.calibration_assessments:
            assessment.validate()
        if self.evidence_status != "synthetic_external_review_calibration_gate":
            raise ValueError(f"{self.gate_id}: unsupported evidence status")
        expected_disposition = (
            "evidence_recorded_not_validated"
            if (
                self.review_records
                or self.validation_gate.external_comparisons
                or self.calibration_assessments
            )
            else "pending_external_review"
        )
        if self.evidence_disposition != expected_disposition:
            raise ValueError(f"{self.gate_id}: evidence disposition is inconsistent")
        if self.model_status_upgraded:
            raise ValueError(f"{self.gate_id}: gate must not auto-upgrade status")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError(f"{self.gate_id}: missing validation caveat")


def build_external_review_calibration_gate(
    *,
    review_records: tuple[ExternalReviewEvidenceRecord, ...] = (),
    external_comparisons: tuple[ExternalCfdComparisonRecord, ...] = (),
    calibration_assessments: tuple[CalibrationAssessment, ...] = (),
) -> ExternalReviewCalibrationGate:
    validation_gate = build_digital_twin_validation_gate(
        external_comparisons=external_comparisons
    )
    for record in review_records:
        record.validate()
    for assessment in calibration_assessments:
        assessment.validate()
    gate = ExternalReviewCalibrationGate(
        gate_id="hs-34a-external-review-calibration-evidence-gate",
        validation_gate=validation_gate,
        review_records=review_records,
        calibration_assessments=calibration_assessments,
        evidence_status="synthetic_external_review_calibration_gate",
        evidence_disposition=(
            "evidence_recorded_not_validated"
            if review_records or external_comparisons or calibration_assessments
            else "pending_external_review"
        ),
        model_status_upgraded=False,
        limitations=(
            "review, comparison, and calibration records inform future decisions",
            "not real-plant validation, certification, commissioning, or safety evidence",
            "model status upgrades require a later explicit validation decision",
        ),
    )
    gate.validate()
    return gate


def _review_payload(record: ExternalReviewEvidenceRecord) -> dict[str, object]:
    return {
        "review_id": record.review_id,
        "reviewer_role": record.reviewer_role,
        "evidence_reference": record.evidence_reference,
        "review_scope": list(record.review_scope),
        "observations": list(record.observations),
        "accepted_recommendations": list(record.accepted_recommendations),
        "evidence_status": record.evidence_status,
        "limitations": list(record.limitations),
    }


def _assessment_payload(assessment: CalibrationAssessment) -> dict[str, object]:
    return {
        "area_id": assessment.area_id,
        "calibration_status": assessment.calibration_status.value,
        "twin_status": assessment.twin_status.value,
        "confidence": assessment.confidence.value,
        "source_count": len(assessment.sources),
        "parameter_count": len(assessment.parameters),
        "residual_count": len(assessment.residuals),
        "calibration_record_count": len(assessment.calibration_records),
        "caveats": list(assessment.caveats),
    }


def render_external_review_calibration_gate_json(
    gate: ExternalReviewCalibrationGate,
) -> str:
    gate.validate()
    payload = {
        "gate_id": gate.gate_id,
        "evidence_status": gate.evidence_status,
        "evidence_disposition": gate.evidence_disposition,
        "model_status_upgraded": gate.model_status_upgraded,
        "validation_gate_id": gate.validation_gate.gate_id,
        "validation_status": gate.validation_gate.real_plant_validation_status,
        "review_records": [_review_payload(record) for record in gate.review_records],
        "external_comparison_count": len(gate.validation_gate.external_comparisons),
        "calibration_assessments": [
            _assessment_payload(assessment)
            for assessment in gate.calibration_assessments
        ],
        "limitations": list(gate.limitations),
    }
    return json.dumps(payload, indent=2, sort_keys=True)
