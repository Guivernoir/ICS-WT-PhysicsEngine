"""CFD and digital-twin foundation tests."""

from __future__ import annotations

import unittest

import numpy as np

from hydrasim.hydraulics.cfd import (
    AREA_IDS,
    BoundaryPatch,
    CalibrationConfidence,
    CalibrationEvidenceClass,
    CalibrationEvidenceSource,
    CalibrationParameter,
    CalibrationRecord,
    CalibrationResidual,
    CalibrationStatus,
    FlowSolverConfig,
    ScalarField,
    ScalarTransportConfig,
    TwinStatus,
    boundary_condition_catalog,
    boundary_condition_ids,
    conditions_for_geometry,
    create_rectangular_mesh,
    effective_diffusivity,
    assess_calibration_evidence,
    build_reference_calibration_assessment,
    evaluate_calibration_residual,
    export_cfd_summary,
    initialize_flow,
    reference_area_model,
    solve_flow_step,
    step_scalar_transport,
    unit_process_catalog,
    unit_process_ids,
    units_by_area,
)
from hydrasim.hydraulics.cfd.mesh import Obstacle
from hydrasim.hydraulics.cfd.turbulence import MixingModelConfig


class CfdDigitalTwinTests(unittest.TestCase):
    def test_mesh_has_deterministic_cell_ids(self) -> None:
        mesh = create_rectangular_mesh(
            cells=(4, 3, 2),
            extents=(2.0, 1.5, 1.0),
            boundaries=(BoundaryPatch("inlet", "xmin", "inlet", 0.02),),
        )

        self.assertEqual(mesh.cell_count, 24)
        self.assertEqual(mesh.cell_id(0, 0, 0), 0)
        self.assertEqual(mesh.cell_id(3, 2, 1), 23)
        self.assertEqual(mesh.indices_from_cell_id(17), (1, 1, 1))
        self.assertEqual(mesh.cell_center(0, 0, 0), (0.25, 0.25, 0.25))

    def test_invalid_mesh_and_obstacle_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_rectangular_mesh(cells=(0, 2, 2), extents=(1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            create_rectangular_mesh(
                cells=(3, 3, 2),
                extents=(1.0, 1.0, 1.0),
                obstacles=(Obstacle("bad", 2, 4, 0, 1, 0, 1),),
            )

    def test_flow_step_produces_stable_finite_diagnostics(self) -> None:
        model = reference_area_model("dosing")
        flow = initialize_flow(model.mesh)
        updated, result = solve_flow_step(
            model.mesh,
            flow,
            FlowSolverConfig(dt=0.01, inlet_velocity=0.02, pressure_iterations=4),
        )

        self.assertTrue(result.stable)
        self.assertGreater(result.active_cells, 0)
        self.assertTrue(np.isfinite(result.mass_residual))
        self.assertTrue(np.isfinite(updated.u).all())
        self.assertGreaterEqual(updated.u.max(), 0.0)

    def test_scalar_transport_is_nonnegative_and_deterministic(self) -> None:
        model = reference_area_model("disinfection")
        flow = initialize_flow(model.mesh)
        flow.u[:, :, :] = 0.01
        scalar = ScalarField.uniform(
            model.mesh, name="chlorine", units="mg/L", value=0.5
        )

        updated_one, result_one = step_scalar_transport(
            model.mesh,
            scalar,
            flow,
            ScalarTransportConfig(dt=0.01, diffusivity=1.0e-5, first_order_decay=0.01),
            source_cells=((1, 1, 1, 0.2),),
        )
        updated_two, result_two = step_scalar_transport(
            model.mesh,
            scalar,
            flow,
            ScalarTransportConfig(dt=0.01, diffusivity=1.0e-5, first_order_decay=0.01),
            source_cells=((1, 1, 1, 0.2),),
        )

        self.assertTrue(result_one.stable)
        self.assertGreaterEqual(result_one.min_value, 0.0)
        self.assertGreater(result_one.mass_after, 0.0)
        np.testing.assert_allclose(updated_one.values, updated_two.values)
        self.assertEqual(result_one, result_two)

    def test_mixing_model_extends_diffusivity(self) -> None:
        value = effective_diffusivity(
            1.0e-5,
            MixingModelConfig(
                turbulent_viscosity=2.0e-5, mixing_length=0.1, intensity=0.2
            ),
        )

        self.assertGreater(value, 1.0e-5)

    def test_reference_area_models_have_no_overclaim_limitations(self) -> None:
        for area_id in AREA_IDS:
            with self.subTest(area_id=area_id):
                model = reference_area_model(area_id)
                self.assertEqual(model.area_id, area_id)
                self.assertGreater(model.mesh.cell_count, 0)
                self.assertEqual(
                    model.twin_metadata.status, TwinStatus.SYNTHETIC_UNVALIDATED
                )
                text = " ".join(model.limitations)
                self.assertIn("uncalibrated", text)
                self.assertIn("not commissioning", text)

    def test_digital_twin_calibration_records_require_evidence_source(self) -> None:
        record = CalibrationRecord(
            parameter="inlet_velocity",
            source="synthetic benchmark",
            value=0.02,
            units="m/s",
            status=TwinStatus.SYNTHETIC_VERIFIED,
        )
        record.validate()

        with self.assertRaises(ValueError):
            CalibrationRecord(
                parameter="inlet_velocity",
                source="",
                value=0.02,
                units="m/s",
                status=TwinStatus.EVIDENCE_CALIBRATED,
            ).validate()

        residual, accepted = evaluate_calibration_residual(
            observed=1.1, predicted=1.0, tolerance=0.2
        )
        self.assertAlmostEqual(residual, 0.1)
        self.assertTrue(accepted)

    def test_reference_calibration_assessment_is_uncalibrated(self) -> None:
        for area_id in AREA_IDS:
            with self.subTest(area_id=area_id):
                model = reference_area_model(area_id)
                assessment = build_reference_calibration_assessment(
                    area_id,
                    uncertainty=model.twin_metadata.uncertainty,
                )

                assessment.validate()
                self.assertEqual(
                    assessment.calibration_status, CalibrationStatus.UNCALIBRATED
                )
                self.assertEqual(
                    assessment.twin_status, TwinStatus.SYNTHETIC_UNVALIDATED
                )
                self.assertEqual(assessment.confidence, CalibrationConfidence.UNKNOWN)
                self.assertIn("uncalibrated", " ".join(assessment.caveats))

    def test_calibration_assessment_classifies_synthetic_evidence(self) -> None:
        source = CalibrationEvidenceSource(
            source_id="synthetic-step-case",
            evidence_class=CalibrationEvidenceClass.SYNTHETIC_REFERENCE,
            description="deterministic reference step response",
            limitations=("synthetic calibration evidence only",),
        )
        parameter = CalibrationParameter(
            parameter="inlet_velocity",
            value=0.04,
            units="m/s",
            source_id=source.source_id,
            accepted_min=0.0,
            accepted_max=0.2,
        )
        residual = CalibrationResidual.from_values(
            parameter="inlet_velocity",
            observed=0.041,
            predicted=0.04,
            tolerance=0.005,
        )

        assessment = assess_calibration_evidence(
            area_id="dosing",
            sources=(source,),
            parameters=(parameter,),
            residuals=(residual,),
        )

        assessment.validate()
        self.assertEqual(
            assessment.calibration_status, CalibrationStatus.SYNTHETIC_CALIBRATED
        )
        self.assertEqual(assessment.twin_status, TwinStatus.SYNTHETIC_VERIFIED)
        self.assertEqual(assessment.confidence, CalibrationConfidence.MEDIUM)
        self.assertIn("not real-plant validation", " ".join(assessment.caveats))

    def test_calibration_assessment_requires_reference_for_real_evidence(self) -> None:
        with self.assertRaises(ValueError):
            CalibrationEvidenceSource(
                source_id="field-trend",
                evidence_class=CalibrationEvidenceClass.FIELD_TELEMETRY,
                description="field trend export",
                limitations=("external data not stored in HydraSim",),
            ).validate()

    def test_calibration_assessment_keeps_evidence_calibration_caveated(self) -> None:
        source = CalibrationEvidenceSource(
            source_id="lab-bench",
            evidence_class=CalibrationEvidenceClass.LAB_MEASUREMENT,
            description="bench comparison for reference contact basin",
            source_reference="lab-record:synthetic-placeholder",
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

        assessment.validate()
        self.assertEqual(
            assessment.calibration_status, CalibrationStatus.EVIDENCE_CALIBRATED
        )
        self.assertEqual(assessment.twin_status, TwinStatus.EVIDENCE_CALIBRATED)
        self.assertEqual(assessment.confidence, CalibrationConfidence.HIGH)
        self.assertIn("not real-plant validation", " ".join(assessment.caveats))

    def test_rejected_calibration_evidence_cannot_upgrade_status(self) -> None:
        source = CalibrationEvidenceSource(
            source_id="synthetic-step-case",
            evidence_class=CalibrationEvidenceClass.SYNTHETIC_REFERENCE,
            description="deterministic reference step response",
            limitations=("synthetic calibration evidence only",),
        )
        parameter = CalibrationParameter(
            parameter="inlet_velocity",
            value=0.04,
            units="m/s",
            source_id=source.source_id,
        )
        residual = CalibrationResidual.from_values(
            parameter="inlet_velocity",
            observed=0.09,
            predicted=0.04,
            tolerance=0.005,
        )

        assessment = assess_calibration_evidence(
            area_id="dosing",
            sources=(source,),
            parameters=(parameter,),
            residuals=(residual,),
        )

        assessment.validate()
        self.assertEqual(assessment.calibration_status, CalibrationStatus.REJECTED)
        self.assertEqual(assessment.twin_status, TwinStatus.SYNTHETIC_UNVALIDATED)
        self.assertEqual(assessment.confidence, CalibrationConfidence.LOW)

    def test_cfd_summary_export_is_stable(self) -> None:
        model = reference_area_model("intake")
        flow, diagnostics = solve_flow_step(
            model.mesh,
            initialize_flow(model.mesh),
            FlowSolverConfig(dt=0.01, pressure_iterations=2),
        )
        first = export_cfd_summary(model, flow=flow, flow_result=diagnostics)
        second = export_cfd_summary(model, flow=flow, flow_result=diagnostics)

        self.assertEqual(first, second)
        self.assertIn('"area_id": "intake"', first)
        self.assertIn('"status": "synthetic_unvalidated"', first)
        self.assertIn("not commissioning or plant-design evidence", first)

    def test_unit_process_catalog_covers_reference_water_processes(self) -> None:
        expected = (
            "intake-channel",
            "rapid-mix-dosing",
            "flocculation-placeholder",
            "clarifier",
            "filter-backwash",
            "contact-basin",
            "clearwell",
            "pumping-header",
            "chemical-feed-skid",
            "waste-backwash-handling",
        )

        self.assertEqual(unit_process_ids(), expected)
        for unit in unit_process_catalog():
            with self.subTest(unit_id=unit.unit_id):
                unit.validate()
                self.assertGreaterEqual(len(unit.boundaries), 2)
                directions = {boundary.direction for boundary in unit.boundaries}
                self.assertIn("input", directions)
                self.assertIn("output", directions)
                self.assertTrue(unit.process_variables)
                self.assertTrue(unit.instrumentation)
                self.assertIn(
                    "not a certified design model", " ".join(unit.limitations)
                )

    def test_units_by_area_links_process_contracts_to_area_models(self) -> None:
        for area_id in AREA_IDS:
            with self.subTest(area_id=area_id):
                model = reference_area_model(area_id)
                units = units_by_area(area_id)
                self.assertTrue(units)
                for unit in units:
                    self.assertEqual(unit.area_id, model.area_id)
                    unit_variables = set(unit.process_variables)
                    model_scalars = set(model.scalar_names)
                    self.assertTrue(unit_variables or model_scalars)

        dosing_units = unit_process_ids()
        self.assertIn("chemical-feed-skid", dosing_units)
        self.assertIn("flocculation-placeholder", dosing_units)

    def test_boundary_condition_catalog_covers_required_contracts(self) -> None:
        expected = (
            "bc-inlet-flow",
            "bc-outlet-flow",
            "bc-pump-discharge",
            "bc-valve-loss",
            "bc-dosing-injection",
            "bc-mechanical-mixer",
            "bc-baffle-wall",
            "bc-porous-filter-media",
            "bc-backwash-flow",
            "bc-drain-flow",
            "bc-recirculation-flow",
            "bc-free-surface-level",
        )

        self.assertEqual(boundary_condition_ids(), expected)
        for contract in boundary_condition_catalog():
            with self.subTest(condition_id=contract.condition_id):
                contract.validate()
                self.assertTrue(contract.required_variables)
                self.assertTrue(contract.effects)
                self.assertIn(
                    "not a commissioning or design-authority boundary model",
                    " ".join(contract.limitations),
                )

    def test_each_unit_process_geometry_has_boundary_conditions(self) -> None:
        for unit in unit_process_catalog():
            with self.subTest(unit_id=unit.unit_id):
                conditions = conditions_for_geometry(unit.geometry_class)
                self.assertTrue(conditions)
                condition_effects = {
                    effect for contract in conditions for effect in contract.effects
                }
                self.assertTrue(condition_effects)

        filter_conditions = boundary_condition_ids()
        self.assertIn("bc-porous-filter-media", filter_conditions)
        self.assertIn("bc-backwash-flow", filter_conditions)


if __name__ == "__main__":
    unittest.main()
