"""CFD device, controller, and supervisory coupling tests."""

from __future__ import annotations

import unittest

from hydrasim.hydraulics.cfd import (
    ControllerCouplingContract,
    ScalarField,
    SamplingRegion,
    SupervisoryTwinRecord,
    apply_controller_action_to_scalar,
    apply_scalar_source_region,
    build_reference_supervisory_records,
    build_supervisory_records,
    controller_coupling_catalog,
    device_coupling_catalog,
    evaluate_controller_action,
    initialize_flow,
    reference_area_model,
    sample_scalar_region,
    supervisory_profile_catalog,
    supervisory_profile_ids,
)


class CfdCouplingTests(unittest.TestCase):
    def test_sensor_sampling_region_reads_spatial_scalar_values(self) -> None:
        model = reference_area_model("intake")
        scalar = ScalarField.uniform(
            model.mesh, name="temperature", units="degC", value=10.0
        )
        scalar.values[:, :, -1] = 20.0

        inlet = sample_scalar_region(
            model.mesh, scalar, SamplingRegion("inlet", "inlet"), aggregation="mean"
        )
        outlet = sample_scalar_region(
            model.mesh, scalar, SamplingRegion("outlet", "outlet"), aggregation="mean"
        )

        self.assertAlmostEqual(inlet, 10.0)
        self.assertAlmostEqual(outlet, 20.0)

    def test_actuator_source_region_changes_scalar_without_mutating_original(
        self,
    ) -> None:
        model = reference_area_model("dosing")
        scalar = ScalarField.uniform(
            model.mesh, name="chlorine", units="mg/L", value=0.0
        )
        region = SamplingRegion("dose-region", "center")

        updated = apply_scalar_source_region(
            model.mesh, scalar, region, strength=2.0, dt=0.5
        )

        self.assertEqual(float(scalar.values.max()), 0.0)
        self.assertGreater(float(updated.values.max()), 0.0)
        self.assertAlmostEqual(
            sample_scalar_region(model.mesh, updated, region, aggregation="point"),
            1.0,
        )

    def test_device_coupling_catalog_is_simulated_and_valid(self) -> None:
        catalog = device_coupling_catalog()
        self.assertTrue(catalog)

        sensor_count = 0
        actuator_count = 0
        for contract in catalog:
            contract.validate()
            self.assertEqual(contract.evidence_status, "simulated_metadata")
            if hasattr(contract, "sensor_tag"):
                sensor_count += 1
            if hasattr(contract, "actuator_tag"):
                actuator_count += 1

        self.assertGreater(sensor_count, 0)
        self.assertGreater(actuator_count, 0)

    def test_controller_coupling_catalog_is_simulated_and_valid(self) -> None:
        catalog = controller_coupling_catalog()
        self.assertTrue(catalog)

        for contract in catalog:
            with self.subTest(controller_id=contract.controller_id):
                contract.validate()
                self.assertEqual(contract.evidence_status, "simulated_metadata")
                self.assertEqual(contract.routine, "pid-lite-label")
                self.assertGreaterEqual(
                    contract.maximum_output, contract.minimum_output
                )

    def test_controller_action_is_bounded_and_caveated(self) -> None:
        contract = ControllerCouplingContract(
            controller_id="CTL-DOSING-TEST",
            unit_id="rapid-mix-dosing",
            actuator_tag="ACT-RAPID-MIX-DOSING-CHEMICAL-INJECTION",
            routine="pid-lite-label",
            controlled_variable="chlorine",
            manipulated_variable="chlorine",
            setpoint=1.0,
            deadband=0.05,
            proportional_gain=3.0,
            minimum_output=0.0,
            maximum_output=2.0,
        )

        action = evaluate_controller_action(contract, process_value=0.0)
        action.validate()

        self.assertEqual(action.action_state, "increase-source")
        self.assertEqual(action.output, 2.0)
        self.assertEqual(action.evidence_status, "simulated_metadata")

        hold = evaluate_controller_action(contract, process_value=0.97)
        self.assertEqual(hold.action_state, "within-deadband")
        self.assertEqual(hold.output, 0.0)

        withhold = evaluate_controller_action(contract, process_value=1.5)
        self.assertEqual(withhold.action_state, "withhold-source")
        self.assertEqual(withhold.output, 0.0)

    def test_controller_action_changes_scalar_through_actuator_region(self) -> None:
        actuator = next(
            contract
            for contract in device_coupling_catalog()
            if getattr(contract, "actuator_tag", "")
            == "ACT-RAPID-MIX-DOSING-CHEMICAL-INJECTION"
        )
        controller = ControllerCouplingContract(
            controller_id="CTL-DOSING-TEST",
            unit_id=actuator.unit_id,
            actuator_tag=actuator.actuator_tag,
            routine="pid-lite-label",
            controlled_variable="chlorine",
            manipulated_variable=actuator.manipulated_variable,
            setpoint=1.0,
            deadband=0.01,
            proportional_gain=1.0,
            minimum_output=0.0,
            maximum_output=actuator.maximum_value,
        )
        model = reference_area_model("dosing")
        scalar = ScalarField.uniform(
            model.mesh, name="chlorine", units="mg/L", value=0.0
        )
        action = evaluate_controller_action(controller, process_value=0.2)

        updated = apply_controller_action_to_scalar(
            model.mesh, scalar, actuator, action, dt=0.5
        )

        self.assertEqual(float(scalar.values.max()), 0.0)
        self.assertGreater(float(updated.values.max()), 0.0)
        self.assertAlmostEqual(
            sample_scalar_region(
                model.mesh,
                updated,
                actuator.target_region,
                aggregation="point",
            ),
            0.4,
        )

        wrong_action = evaluate_controller_action(
            ControllerCouplingContract(
                controller_id="CTL-WRONG",
                unit_id=actuator.unit_id,
                actuator_tag="ACT-OTHER",
                routine="pid-lite-label",
                controlled_variable="chlorine",
                manipulated_variable=actuator.manipulated_variable,
                setpoint=1.0,
                deadband=0.01,
                proportional_gain=1.0,
                minimum_output=0.0,
                maximum_output=actuator.maximum_value,
            ),
            process_value=0.2,
        )
        with self.assertRaises(ValueError):
            apply_controller_action_to_scalar(
                model.mesh, scalar, actuator, wrong_action, dt=0.5
            )

    def test_supervisory_profiles_are_simulated_and_valid(self) -> None:
        self.assertEqual(
            supervisory_profile_ids(),
            (
                "hmi-process-overview",
                "historian-process-trend",
                "engineering-model-context",
            ),
        )

        for profile in supervisory_profile_catalog():
            with self.subTest(profile_id=profile.profile_id):
                profile.validate()
                self.assertEqual(profile.evidence_status, "simulated_metadata")
                self.assertIn(
                    "does not prove operational visibility or site identity",
                    " ".join(profile.limitations),
                )

    def test_supervisory_records_consume_cfd_state_without_site_identity(self) -> None:
        model = reference_area_model("disinfection")
        flow = initialize_flow(model.mesh)
        flow.u[:, :, :] = 0.02
        scalar = ScalarField.uniform(
            model.mesh, name="chlorine", units="mg/L", value=0.7
        )
        scalar.values[:, :, -1] = 1.1

        records = build_supervisory_records(model, scalar=scalar, flow=flow)

        self.assertTrue(records)
        variables = {record.variable for record in records}
        self.assertIn("chlorine_mean", variables)
        self.assertIn("max_u", variables)
        self.assertIn("mesh_cell_count", variables)
        roles = {record.role for record in records}
        self.assertEqual(roles, {"hmi", "historian", "engineering_workstation"})
        for record in records:
            record.validate()
            self.assertEqual(record.area_id, "disinfection")
            self.assertEqual(record.site_identity, "unknown")
            self.assertEqual(record.evidence_status, "simulated_metadata")
            self.assertNotIn("PLC", " ".join(record.limitations))

    def test_supervisory_records_reject_identity_overclaim(self) -> None:
        record = SupervisoryTwinRecord(
            record_id="bad-record",
            profile_id="historian-process-trend",
            role="historian",
            area_id="dosing",
            variable="chlorine_mean",
            value=1.0,
            units="mg/L",
            source_kind="cfd_scalar_summary",
            spatial_context="area mesh summary",
            twin_status="synthetic_unvalidated",
            site_identity="real-site-a",
        )

        with self.assertRaises(ValueError):
            record.validate()

    def test_reference_supervisory_records_are_metadata_only(self) -> None:
        records = build_reference_supervisory_records("intake")

        self.assertTrue(records)
        self.assertEqual(
            {record.source_kind for record in records}, {"digital_twin_metadata"}
        )
        for record in records:
            record.validate()
            self.assertEqual(record.area_id, "intake")
            self.assertEqual(record.site_identity, "unknown")


if __name__ == "__main__":
    unittest.main()
