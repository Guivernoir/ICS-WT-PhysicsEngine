import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from hydrasim.plant import (  # noqa: E402
    build_live_orchestration_plan,
    build_runtime_artifact,
    export_hs_bundle,
    get_profile,
    profile_ids,
    render_hs_pcap_bytes,
    render_live_plan,
    render_process_evolution_csv,
    render_process_review_csv,
    render_summary_markdown,
    render_transcript_csv,
    run_live_orchestration,
    scenario_ids,
    validate_profile,
)
from hydrasim.plant.cli import main as hs_main  # noqa: E402
from hydrasim.scenarios import get_scenario as get_legacy_scenario  # noqa: E402


class TestHsRuntime(unittest.TestCase):
    def test_expected_profiles_are_available_and_valid(self):
        self.assertEqual(
            profile_ids(),
            (
                "single-stage-legacy",
                "field-device-lab",
                "controller-cell",
                "supervisory-lab",
                "reference-water-plant",
            ),
        )
        for profile_id in profile_ids():
            self.assertEqual(validate_profile(get_profile(profile_id)), ())

    def test_expected_whole_plant_scenarios_are_available(self):
        self.assertEqual(
            scenario_ids(),
            (
                "HS-WTP-001",
                "HS-WTP-002",
                "HS-WTP-003",
                "HS-WTP-004",
                "HS-WTP-005",
                "HS-WTP-006",
                "HS-WTP-007",
                "HS-WTP-008",
                "HS-WTP-009",
                "HS-WTP-010",
                "HS-WTP-011",
                "HS-WTP-012",
            ),
        )

    def test_all_area_is_offline_export_only(self):
        with self.assertRaisesRegex(ValueError, "area 'all'"):
            build_runtime_artifact(
                "reference-water-plant",
                "HS-WTP-002",
                "all",
                "full-cell",
            )

    def test_selected_area_full_cell_includes_supervisory_references(self):
        artifact = build_runtime_artifact(
            "reference-water-plant",
            "HS-WTP-002",
            "disinfection",
            "full-cell",
        )
        node_ids = {node.node_id for node in artifact.active_nodes}
        self.assertIn("disinfection-plc", node_ids)
        self.assertIn("chlorine-analyzer-ai", node_ids)
        self.assertIn("plant-historian", node_ids)
        self.assertTrue(all(tx.area == "disinfection" for tx in artifact.transactions))
        states = {state.controller_id: state for state in artifact.controller_states}
        self.assertEqual(states["disinfection-plc"].routine, "monitoring")
        self.assertEqual(states["disinfection-plc"].mode, "auto")
        self.assertEqual(len(artifact.process_evolution), 1)
        process = artifact.process_evolution[0]
        self.assertEqual(process.area, "disinfection")
        self.assertEqual(process.evidence_status, "synthetic_cfd_process_truth")
        self.assertIn("not real-plant validation", " ".join(process.limitations))
        self.assertEqual(len(artifact.process_reviews), 1)
        review = artifact.process_reviews[0]
        self.assertEqual(review.area, "disinfection")
        self.assertEqual(
            review.evidence_status,
            "synthetic_scenario_process_review",
        )
        self.assertIn("writes=0", review.observable_network_effects)
        self.assertIn("real plant validation", " ".join(review.must_not_claim))

    def test_field_device_stage_keeps_only_activation_traffic(self):
        artifact = build_runtime_artifact(
            "reference-water-plant",
            "HS-WTP-001",
            "intake",
            "field-devices",
        )
        self.assertEqual([tx.stage for tx in artifact.transactions], ["field-devices"])
        self.assertIn("stage-driver", {node.node_id for node in artifact.active_nodes})

    def test_profile_overrides_are_reflected_in_summary(self):
        artifact = build_runtime_artifact(
            "reference-water-plant",
            "HS-WTP-002",
            "dosing",
            "full-cell",
            control_system="dcs-lite",
            topology="segmented-cell",
            media="mixed-lab",
        )
        rendered = render_summary_markdown(artifact)
        self.assertIn("Control system: `dcs-lite`", rendered)
        self.assertIn("Topology: `segmented-cell`", rendered)
        self.assertIn("Media: `mixed-lab`", rendered)
        self.assertIn("## Controller Logic State", rendered)
        self.assertIn("## CFD Process Evolution", rendered)
        self.assertIn("## Scenario Process-Truth Review", rendered)

    def test_transcript_and_pcap_are_deterministic(self):
        artifact = build_runtime_artifact(
            "reference-water-plant",
            "HS-WTP-002",
            "all",
            "offline-export",
        )
        self.assertEqual(
            render_transcript_csv(artifact), render_transcript_csv(artifact)
        )
        first = render_hs_pcap_bytes(artifact)
        second = render_hs_pcap_bytes(artifact)
        self.assertEqual(first, second)
        self.assertEqual(first[:4], b"\xd4\xc3\xb2\xa1")
        self.assertEqual(
            render_process_evolution_csv(artifact),
            render_process_evolution_csv(artifact),
        )
        self.assertIn(
            "synthetic_cfd_process_truth", render_process_evolution_csv(artifact)
        )
        self.assertEqual(
            render_process_review_csv(artifact),
            render_process_review_csv(artifact),
        )
        self.assertIn(
            "synthetic_scenario_process_review",
            render_process_review_csv(artifact),
        )

    def test_bundle_export_creates_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "hs-bundle"
            written = export_hs_bundle(
                "reference-water-plant",
                "HS-WTP-002",
                "all",
                "offline-export",
                target,
            )
            self.assertEqual(
                {path.name for path in written},
                {
                    "summary.md",
                    "transcript.csv",
                    "topology.md",
                    "controller-states.csv",
                    "process-evolution.csv",
                    "process-review.csv",
                    "scenario.pcap",
                    "capture-notes.md",
                    "manifest.json",
                    "cfd-flow-snapshots.csv",
                    "cfd-mesh-geometry.json",
                    "cfd-scalar-fields.csv",
                    "cfd-state-timeline.csv",
                    "checksums.sha256",
                },
            )
            manifest = json.loads((target / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["source_class"], "SyntheticReferencePlant")
            self.assertEqual(manifest["profile_id"], "reference-water-plant")
            self.assertGreater(manifest["transaction_count"], 0)
            self.assertGreaterEqual(manifest["controller_state_count"], 1)
            self.assertGreaterEqual(manifest["process_evolution_count"], 1)
            self.assertGreaterEqual(manifest["process_review_count"], 1)
            self.assertEqual(manifest["cfd_lab_bundle_version"], "v2")
            self.assertEqual(manifest["cfd_lab_artifact_count"], 4)
            process_csv = (target / "process-evolution.csv").read_text("utf-8")
            self.assertIn("synthetic_cfd_process_truth", process_csv)
            review_csv = (target / "process-review.csv").read_text("utf-8")
            self.assertIn("synthetic_scenario_process_review", review_csv)
            self.assertIn("Do not claim real plant validation", review_csv)
            mesh_json = (target / "cfd-mesh-geometry.json").read_text("utf-8")
            self.assertIn("synthetic_cfd_lab_bundle_v2", mesh_json)
            self.assertIn("array_policy", mesh_json)
            scalar_csv = (target / "cfd-scalar-fields.csv").read_text("utf-8")
            flow_csv = (target / "cfd-flow-snapshots.csv").read_text("utf-8")
            timeline_csv = (target / "cfd-state-timeline.csv").read_text("utf-8")
            self.assertIn("cell_id", scalar_csv)
            self.assertIn("mass_residual", flow_csv)
            self.assertIn("synthetic_cfd_lab_bundle_v2", timeline_csv)

    def test_bundle_export_uses_create_new_directory_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                export_hs_bundle(
                    "reference-water-plant",
                    "HS-WTP-002",
                    "all",
                    "offline-export",
                    target,
                )

    def test_cli_list_validate_and_run(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(hs_main(["list-profiles"]), 0)
        self.assertIn("reference-water-plant", out.getvalue())

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(hs_main(["validate-profile", "reference-water-plant"]), 0)
        self.assertIn("valid: reference-water-plant", out.getvalue())

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(
                hs_main(
                    [
                        "run",
                        "reference-water-plant",
                        "--scenario",
                        "HS-WTP-002",
                        "--area",
                        "disinfection",
                        "--stage",
                        "full-cell",
                        "--format",
                        "csv",
                    ]
                ),
                0,
            )
        self.assertIn("disinfection-plc", out.getvalue())

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(
                hs_main(
                    [
                        "launch-live",
                        "reference-water-plant",
                        "--scenario",
                        "HS-WTP-002",
                        "--area",
                        "disinfection",
                        "--stage",
                        "full-cell",
                        "--dry-run",
                    ]
                ),
                0,
            )
        self.assertIn("disinfection-plc", out.getvalue())

    def test_live_orchestration_plan_and_lifecycle(self):
        plan = build_live_orchestration_plan(
            "reference-water-plant",
            "HS-WTP-002",
            "disinfection",
            "full-cell",
            base_port=5600,
            duration_seconds=1.0,
            startup_delay_seconds=0.0,
        )
        rendered = render_live_plan(plan)
        self.assertIn("disinfection-plc", rendered)
        self.assertTrue(all(node.port >= 5600 for node in plan.nodes))

        launched = []

        class FakeProcess:
            def __init__(self, command, **_kwargs):
                self.command = command
                self.returncode = None
                launched.append(command)

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                if self.returncode is None:
                    self.returncode = 0
                return self.returncode

            def kill(self):
                self.returncode = -9

        results = run_live_orchestration(
            plan,
            process_factory=FakeProcess,
            sleeper=lambda _seconds: None,
        )
        self.assertEqual(len(results), len(plan.nodes))
        self.assertEqual(len(launched), len(plan.nodes))
        self.assertTrue(all("hydrasim" in command for command in launched))

    def test_legacy_mvp_scenarios_remain_available(self):
        self.assertEqual(
            get_legacy_scenario("water-treatment-normal").scenario_id,
            "MVP-MB-HYDRA-002",
        )


if __name__ == "__main__":
    unittest.main()
