"""CFD runtime performance gate tests."""

from __future__ import annotations

import json
import unittest

from hydrasim.hydraulics.cfd import (
    build_runtime_performance_gate,
    estimate_field_memory_bytes,
    get_performance_preset,
    get_runtime_performance_budget,
    mesh_for_preset,
    performance_preset_ids,
    performance_presets,
    render_runtime_performance_gate_json,
    run_cfd_benchmark,
    runtime_performance_budgets,
)


class CfdPerformanceGateTests(unittest.TestCase):
    def test_performance_presets_define_bounded_grid_envelope(self) -> None:
        self.assertEqual(
            performance_preset_ids(), ("tiny-grid", "small-grid", "medium-grid")
        )

        previous_cells = 0
        for preset in performance_presets():
            with self.subTest(preset_id=preset.preset_id):
                preset.validate()
                mesh = mesh_for_preset(preset)
                self.assertGreater(mesh.cell_count, previous_cells)
                previous_cells = mesh.cell_count
                estimated = estimate_field_memory_bytes(
                    mesh, scalar_count=preset.scalar_count
                )
                expected = mesh.cell_count * (4 + preset.scalar_count) * 8
                self.assertEqual(estimated, expected + mesh.cell_count)

    def test_tiny_cfd_benchmark_reports_stable_evidence(self) -> None:
        preset = get_performance_preset("tiny-grid")
        result = run_cfd_benchmark(preset, iterations=1)

        result.validate()
        self.assertEqual(result.preset_id, "tiny-grid")
        self.assertEqual(result.cell_count, 36)
        self.assertGreater(result.estimated_field_memory_bytes, 0)
        self.assertEqual(result.iterations, 1)
        self.assertGreaterEqual(result.wall_time_seconds, 0.0)
        self.assertTrue(result.stable)
        self.assertGreaterEqual(result.mass_residual, 0.0)

    def test_runtime_performance_gate_records_bounded_evidence(self) -> None:
        budgets = runtime_performance_budgets()
        self.assertEqual(
            tuple(budget.preset_id for budget in budgets),
            ("tiny-grid", "small-grid", "medium-grid"),
        )
        for budget in budgets:
            budget.validate()
            self.assertEqual(
                get_runtime_performance_budget(budget.preset_id),
                budget,
            )

        records = build_runtime_performance_gate(iterations=1)
        self.assertEqual(
            tuple(record.preset_id for record in records),
            ("tiny-grid", "small-grid", "medium-grid"),
        )
        for record in records:
            with self.subTest(preset_id=record.preset_id):
                record.validate()
                self.assertTrue(record.gate_passed)
                self.assertTrue(record.stable)
                self.assertGreater(record.estimated_field_memory_bytes, 0)
                self.assertGreater(record.output_size_bytes, 0)
                self.assertGreaterEqual(record.wall_time_seconds, 0.0)
                self.assertLessEqual(record.long_run_drift, 1.0e-9)
                self.assertIn(record.preset_id, record.deterministic_signature)
                self.assertEqual(
                    record.evidence_status,
                    "synthetic_runtime_performance_gate",
                )

        rendered_one = render_runtime_performance_gate_json(records)
        rendered_two = render_runtime_performance_gate_json(records)
        self.assertEqual(rendered_one, rendered_two)
        self.assertIn("synthetic_runtime_performance_gate", rendered_one)
        self.assertIn("volatile_fields", rendered_one)
        payload = json.loads(rendered_one)
        self.assertTrue(payload["records"])
        self.assertNotIn("wall_time_seconds", payload["records"][0])


if __name__ == "__main__":
    unittest.main()
