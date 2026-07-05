"""HS-35 Reference Water Plant CFD release-candidate tests."""

from __future__ import annotations

import contextlib
import io
import json
import unittest

from hydrasim.plant import (  # noqa: E402
    build_reference_water_plant_cfd_release_candidate,
    render_release_candidate_json,
    render_release_candidate_markdown,
)
from hydrasim.plant.cli import main as hs_main  # noqa: E402
from hydrasim.plant.release import EXPECTED_BUNDLE_ARTIFACTS  # noqa: E402


class TestHsReleaseCandidate(unittest.TestCase):
    def test_release_candidate_combines_required_surfaces(self) -> None:
        release = build_reference_water_plant_cfd_release_candidate()

        release.validate()
        self.assertEqual(
            release.evidence_status,
            "synthetic_reference_water_plant_cfd_release_candidate",
        )
        self.assertEqual(release.full_plant_offline.area, "all")
        self.assertEqual(release.full_plant_offline.stage, "offline-export")
        self.assertEqual(release.selected_area_full_cell.area, "disinfection")
        self.assertEqual(release.selected_area_full_cell.stage, "full-cell")
        self.assertTrue(release.selected_area_live_plan.nodes)
        self.assertEqual(
            tuple(release.expected_bundle_artifacts), EXPECTED_BUNDLE_ARTIFACTS
        )
        self.assertTrue(all(check.status == "passed" for check in release.checks))
        self.assertFalse(release.review_gate.model_status_upgraded)
        self.assertIn(
            "blocked_missing_real_calibration_and_external_validation",
            release.validation_gate.real_plant_validation_status,
        )

    def test_release_candidate_renderers_are_deterministic(self) -> None:
        release = build_reference_water_plant_cfd_release_candidate(
            scenario_id="HS-WTP-004",
            selected_area="dosing",
        )

        first_json = render_release_candidate_json(release)
        second_json = render_release_candidate_json(release)
        self.assertEqual(first_json, second_json)
        payload = json.loads(first_json)
        self.assertEqual(payload["selected_area"], "dosing")
        self.assertEqual(payload["review_gate_disposition"], "pending_external_review")
        self.assertIn("checksums.sha256", payload["expected_bundle_artifacts"])

        first_md = render_release_candidate_markdown(release)
        second_md = render_release_candidate_markdown(release)
        self.assertEqual(first_md, second_md)
        self.assertIn("Reference Water Plant CFD Release Candidate", first_md)
        self.assertIn("not real-plant validation", first_md)

    def test_release_candidate_cli_outputs_markdown_and_json(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(hs_main(["release-candidate"]), 0)
        markdown = out.getvalue()
        self.assertIn("Reference Water Plant CFD Release Candidate", markdown)
        self.assertIn("synthetic_reference_water_plant_cfd_release_candidate", markdown)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(
                hs_main(
                    [
                        "release-candidate",
                        "--scenario",
                        "HS-WTP-004",
                        "--area",
                        "dosing",
                        "--format",
                        "json",
                    ]
                ),
                0,
            )
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["scenario_id"], "HS-WTP-004")
        self.assertEqual(payload["selected_area"], "dosing")


if __name__ == "__main__":
    unittest.main()
