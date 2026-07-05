"""HS-30B calibration evidence model tests."""

from __future__ import annotations

import unittest

from hydrasim.hydraulics.cfd import (
    CalibrationDataPoint,
    CalibrationEvidenceClass,
    CalibrationEvidenceSource,
    CalibrationParameter,
    CalibrationResidual,
    CalibrationStatus,
    TwinStatus,
    assess_calibration_evidence,
    assess_calibration_fit,
    fit_calibration_parameter,
)


def _synthetic_source() -> CalibrationEvidenceSource:
    return CalibrationEvidenceSource(
        source_id="synthetic-calibration-run",
        evidence_class=CalibrationEvidenceClass.SYNTHETIC_REFERENCE,
        description="synthetic reference comparison",
        limitations=("synthetic source only",),
    )


def _field_source() -> CalibrationEvidenceSource:
    return CalibrationEvidenceSource(
        source_id="field-telemetry-window",
        evidence_class=CalibrationEvidenceClass.FIELD_TELEMETRY,
        description="sanitized field telemetry window",
        source_reference="recorded-source:demo-window",
        limitations=("sanitized external evidence", "not full plant validation"),
    )


class CfdCalibrationEvidenceTests(unittest.TestCase):
    def test_synthetic_fit_preserves_rejected_data_and_uncertainty(self) -> None:
        source = _synthetic_source()
        fit = fit_calibration_parameter(
            area_id="dosing",
            source=source,
            parameter="chlorine_offset",
            units="mg/L",
            samples=(
                CalibrationDataPoint(
                    "s1", source.source_id, "chlorine_offset", 1.2, 1.0
                ),
                CalibrationDataPoint(
                    "s2", source.source_id, "chlorine_offset", 1.4, 1.2
                ),
                CalibrationDataPoint(
                    "wrong", "other-source", "chlorine_offset", 0.0, 0.0
                ),
            ),
            residual_tolerance=0.001,
            accepted_min=0.0,
            accepted_max=1.0,
        )

        fit.validate()
        self.assertAlmostEqual(fit.fitted_parameter.value, 0.2)
        self.assertEqual(len(fit.residuals), 2)
        self.assertEqual(len(fit.rejected_data), 1)
        self.assertEqual(fit.calibration_record.status, TwinStatus.SYNTHETIC_VERIFIED)
        self.assertEqual(fit.validation_status, "fitted_not_validated")
        self.assertIn("not validation", " ".join(fit.caveats))
        self.assertLessEqual(fit.uncertainty_update.absolute_bound, 1.0e-12)

        assessment = assess_calibration_fit(fit)
        self.assertEqual(
            assessment.calibration_status, CalibrationStatus.SYNTHETIC_CALIBRATED
        )

    def test_field_fit_can_be_evidence_calibrated_with_explicit_record(self) -> None:
        source = _field_source()
        fit = fit_calibration_parameter(
            area_id="disinfection",
            source=source,
            parameter="chlorine_offset",
            units="mg/L",
            samples=(
                CalibrationDataPoint(
                    "f1", source.source_id, "chlorine_offset", 0.95, 0.9
                ),
                CalibrationDataPoint(
                    "f2", source.source_id, "chlorine_offset", 1.05, 1.0
                ),
            ),
            residual_tolerance=0.001,
            accepted_min=-0.5,
            accepted_max=0.5,
        )

        assessment = assess_calibration_fit(fit)

        self.assertEqual(fit.calibration_record.status, TwinStatus.EVIDENCE_CALIBRATED)
        self.assertEqual(
            assessment.calibration_status, CalibrationStatus.EVIDENCE_CALIBRATED
        )
        self.assertEqual(assessment.twin_status, TwinStatus.EVIDENCE_CALIBRATED)
        self.assertIn("not real-plant validation", " ".join(assessment.caveats))

    def test_direct_non_synthetic_assessment_needs_calibration_record(self) -> None:
        source = _field_source()
        parameter = CalibrationParameter(
            parameter="inlet_velocity",
            value=0.01,
            units="m/s",
            source_id=source.source_id,
        )
        residual = CalibrationResidual.from_values(
            parameter="inlet_velocity",
            observed=1.0,
            predicted=1.0,
            tolerance=0.01,
        )

        assessment = assess_calibration_evidence(
            area_id="intake",
            sources=(source,),
            parameters=(parameter,),
            residuals=(residual,),
        )

        self.assertEqual(
            assessment.calibration_status, CalibrationStatus.CALIBRATION_READY
        )
        self.assertNotEqual(
            assessment.calibration_status, CalibrationStatus.EVIDENCE_CALIBRATED
        )
        self.assertIn("explicit calibration record", " ".join(assessment.caveats))

    def test_residual_failure_rejects_fitted_assessment(self) -> None:
        source = _synthetic_source()
        fit = fit_calibration_parameter(
            area_id="filtration",
            source=source,
            parameter="headloss_offset",
            units="m",
            samples=(
                CalibrationDataPoint(
                    "h1", source.source_id, "headloss_offset", 1.0, 0.0
                ),
                CalibrationDataPoint(
                    "h2", source.source_id, "headloss_offset", 1.0, 0.2
                ),
            ),
            residual_tolerance=0.01,
        )

        assessment = assess_calibration_fit(fit)

        self.assertEqual(assessment.calibration_status, CalibrationStatus.REJECTED)
        self.assertTrue(any(not residual.accepted for residual in fit.residuals))

    def test_fit_rejects_out_of_range_parameter(self) -> None:
        source = _synthetic_source()
        with self.assertRaises(ValueError):
            fit_calibration_parameter(
                area_id="dosing",
                source=source,
                parameter="chlorine_offset",
                units="mg/L",
                samples=(
                    CalibrationDataPoint(
                        "s1", source.source_id, "chlorine_offset", 5.0, 1.0
                    ),
                ),
                residual_tolerance=0.1,
                accepted_min=-1.0,
                accepted_max=1.0,
            )


if __name__ == "__main__":
    unittest.main()
