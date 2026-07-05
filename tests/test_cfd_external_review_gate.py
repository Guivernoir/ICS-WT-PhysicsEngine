"""External review and calibration evidence gate tests."""

from __future__ import annotations

import json
import unittest

from hydrasim.hydraulics.cfd import (
    EXTERNAL_COMPARISON_NOT_VALIDATION,
    CalibrationEvidenceClass,
    CalibrationEvidenceSource,
    CalibrationParameter,
    CalibrationRecord,
    CalibrationResidual,
    TwinStatus,
    ExternalCfdComparisonRecord,
    ExternalReviewEvidenceRecord,
    assess_calibration_evidence,
    build_external_review_calibration_gate,
    render_external_review_calibration_gate_json,
)


class CfdExternalReviewGateTests(unittest.TestCase):
    def test_default_external_review_gate_is_pending(self) -> None:
        gate = build_external_review_calibration_gate()

        gate.validate()
        self.assertEqual(gate.evidence_disposition, "pending_external_review")
        self.assertFalse(gate.model_status_upgraded)
        self.assertFalse(gate.review_records)
        self.assertFalse(gate.calibration_assessments)

    def test_review_record_requires_evidence_reference(self) -> None:
        with self.assertRaises(ValueError):
            ExternalReviewEvidenceRecord(
                review_id="missing-reference",
                reviewer_role="water-treatment-sme",
                evidence_reference="",
                review_scope=("dosing",),
                observations=("scope reviewed",),
            ).validate()

    def test_review_evidence_is_recorded_without_status_upgrade(self) -> None:
        review = ExternalReviewEvidenceRecord(
            review_id="sme-review-001",
            reviewer_role="water-treatment-sme",
            evidence_reference="review-note:synthetic-placeholder",
            review_scope=("dosing", "disinfection"),
            observations=("process labels are plausible for a synthetic lab",),
            accepted_recommendations=("add calibration evidence before validation",),
        )
        gate = build_external_review_calibration_gate(review_records=(review,))

        gate.validate()
        self.assertEqual(
            gate.evidence_status,
            "synthetic_external_review_calibration_gate",
        )
        self.assertEqual(gate.evidence_disposition, "evidence_recorded_not_validated")
        self.assertFalse(gate.model_status_upgraded)
        rendered = render_external_review_calibration_gate_json(gate)
        self.assertIn("review_recorded_not_validated", rendered)
        self.assertIn("not real-plant validation", rendered)

    def test_comparison_and_calibration_records_remain_non_validating(self) -> None:
        comparison = ExternalCfdComparisonRecord(
            comparison_id="comparison-001",
            area_id="disinfection",
            source_reference="external-cfd:placeholder",
            metric="mean_residual_proxy",
            hydrasim_value=0.5,
            external_value=0.505,
            tolerance=0.01,
            accepted=True,
        )
        source = CalibrationEvidenceSource(
            source_id="lab-calibration",
            evidence_class=CalibrationEvidenceClass.LAB_MEASUREMENT,
            description="synthetic lab calibration placeholder",
            source_reference="lab-record:placeholder",
            limitations=("not full plant validation",),
        )
        parameter = CalibrationParameter(
            parameter="diffusivity",
            value=1.0e-5,
            units="m2/s",
            source_id=source.source_id,
            accepted_min=0.0,
        )
        residual = CalibrationResidual.from_values(
            parameter="diffusivity",
            observed=1.01e-5,
            predicted=1.0e-5,
            tolerance=1.0e-6,
        )
        assessment = assess_calibration_evidence(
            area_id="disinfection",
            sources=(source,),
            parameters=(parameter,),
            residuals=(residual,),
            calibration_records=(
                CalibrationRecord(
                    parameter="diffusivity",
                    source=source.source_id,
                    value=1.0e-5,
                    units="m2/s",
                    status=TwinStatus.EVIDENCE_CALIBRATED,
                ),
            ),
        )

        gate = build_external_review_calibration_gate(
            external_comparisons=(comparison,),
            calibration_assessments=(assessment,),
        )

        gate.validate()
        self.assertFalse(gate.model_status_upgraded)
        self.assertEqual(
            gate.validation_gate.real_plant_validation_status,
            EXTERNAL_COMPARISON_NOT_VALIDATION,
        )
        rendered = render_external_review_calibration_gate_json(gate)
        self.assertEqual(rendered, render_external_review_calibration_gate_json(gate))
        payload = json.loads(rendered)
        self.assertEqual(payload["external_comparison_count"], 1)
        self.assertEqual(payload["calibration_assessments"][0]["source_count"], 1)
        self.assertFalse(payload["model_status_upgraded"])


if __name__ == "__main__":
    unittest.main()
