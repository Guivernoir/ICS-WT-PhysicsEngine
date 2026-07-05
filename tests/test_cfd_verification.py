"""HS-30A numerical verification suite tests."""

from __future__ import annotations

import unittest

from hydrasim.hydraulics.cfd import (
    NumericalVerificationResult,
    build_reference_numerical_verification_suite,
    verify_boundary_condition_response,
    verify_constant_scalar_conservation,
    verify_flow_mass_residual_bound,
    verify_long_run_scalar_drift,
    verify_mesh_refinement_sensitivity,
)


class CfdVerificationTests(unittest.TestCase):
    def test_reference_numerical_verification_suite_passes(self) -> None:
        suite = build_reference_numerical_verification_suite()

        suite.validate()
        self.assertTrue(suite.passed)
        self.assertEqual(suite.suite_id, "hs-30a-reference-numerical-verification")
        categories = {result.category for result in suite.results}
        self.assertIn("manufactured_constant_solution", categories)
        self.assertIn("conservation_check", categories)
        self.assertIn("mesh_refinement_sensitivity", categories)
        self.assertIn("boundary_condition_check", categories)
        self.assertIn("long_run_drift", categories)
        for result in suite.results:
            self.assertLessEqual(result.value, result.tolerance)
            self.assertIn("not real-plant validation", " ".join(result.limitations))

    def test_individual_verification_cases_are_stable(self) -> None:
        cases = (
            verify_constant_scalar_conservation(),
            verify_flow_mass_residual_bound(),
            verify_mesh_refinement_sensitivity(),
            verify_boundary_condition_response(),
            verify_long_run_scalar_drift(steps=8),
        )

        for case in cases:
            with self.subTest(case_id=case.case_id):
                case.validate()
                self.assertTrue(case.passed)
                self.assertEqual(
                    case.evidence_status, "synthetic_numerical_verification"
                )

    def test_verification_result_requires_no_overclaim_caveat(self) -> None:
        result = NumericalVerificationResult(
            case_id="bad-case",
            category="bad",
            metric="bad",
            value=0.0,
            tolerance=1.0,
            passed=True,
            evidence_status="synthetic_numerical_verification",
            limitations=("synthetic only",),
        )

        with self.assertRaises(ValueError):
            result.validate()

    def test_long_run_verification_rejects_invalid_step_count(self) -> None:
        with self.assertRaises(ValueError):
            verify_long_run_scalar_drift(steps=0)


if __name__ == "__main__":
    unittest.main()
