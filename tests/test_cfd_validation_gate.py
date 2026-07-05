"""Digital-twin validation gate tests."""

from __future__ import annotations

import json
import unittest

from hydrasim.hydraulics.cfd import (
    AREA_IDS,
    EXTERNAL_COMPARISON_NOT_VALIDATION,
    REAL_PLANT_VALIDATION_BLOCKED,
    ExternalCfdComparisonRecord,
    build_digital_twin_validation_gate,
    build_reference_validation_area_records,
    render_digital_twin_validation_gate_json,
)


class CfdValidationGateTests(unittest.TestCase):
    def test_reference_validation_records_cover_all_areas(self) -> None:
        records = build_reference_validation_area_records()

        self.assertEqual(tuple(record.area_id for record in records), AREA_IDS)
        for record in records:
            with self.subTest(area_id=record.area_id):
                record.validate()
                self.assertTrue(record.reference_model_checked)
                self.assertEqual(
                    record.evidence_status,
                    "synthetic_digital_twin_validation_area",
                )
                self.assertIn("not real-plant validation", " ".join(record.limitations))

    def test_reference_validation_gate_verifies_implementation_only(self) -> None:
        gate = build_digital_twin_validation_gate()

        gate.validate()
        self.assertTrue(gate.implementation_verified)
        self.assertEqual(
            gate.real_plant_validation_status,
            REAL_PLANT_VALIDATION_BLOCKED,
        )
        self.assertEqual(
            gate.evidence_status,
            "synthetic_digital_twin_validation_gate",
        )
        self.assertFalse(gate.external_comparisons)

    def test_validation_gate_rendering_is_deterministic(self) -> None:
        gate = build_digital_twin_validation_gate()
        first = render_digital_twin_validation_gate_json(gate)
        second = render_digital_twin_validation_gate_json(gate)

        self.assertEqual(first, second)
        self.assertIn("synthetic_digital_twin_validation_gate", first)
        self.assertIn(REAL_PLANT_VALIDATION_BLOCKED, first)
        payload = json.loads(first)
        self.assertEqual(payload["gate_id"], "hs-34-digital-twin-validation-gate")
        self.assertEqual(len(payload["area_records"]), len(AREA_IDS))
        self.assertTrue(payload["numerical_suite"]["passed"])

    def test_external_comparison_requires_source_reference(self) -> None:
        with self.assertRaises(ValueError):
            ExternalCfdComparisonRecord(
                comparison_id="bad-comparison",
                area_id="dosing",
                source_reference="",
                metric="mean_velocity",
                hydrasim_value=0.02,
                external_value=0.02,
                tolerance=0.001,
                accepted=True,
            ).validate()

    def test_external_comparison_does_not_upgrade_real_plant_validation(self) -> None:
        comparison = ExternalCfdComparisonRecord(
            comparison_id="openfoam-dosing-reference",
            area_id="dosing",
            source_reference="external-cfd-record:synthetic-placeholder",
            metric="mean_velocity",
            hydrasim_value=0.02,
            external_value=0.0205,
            tolerance=0.001,
            accepted=True,
        )
        gate = build_digital_twin_validation_gate(external_comparisons=(comparison,))

        gate.validate()
        self.assertTrue(gate.implementation_verified)
        self.assertEqual(
            gate.real_plant_validation_status,
            EXTERNAL_COMPARISON_NOT_VALIDATION,
        )
        rendered = render_digital_twin_validation_gate_json(gate)
        self.assertIn("external_cfd_comparison_record", rendered)
        self.assertIn("not real-plant validation", rendered)


if __name__ == "__main__":
    unittest.main()
