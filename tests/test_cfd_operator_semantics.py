"""Operator, historian, and engineering semantics tests."""

from __future__ import annotations

import unittest

from hydrasim.hydraulics.cfd import (
    EngineeringWorkstationEvent,
    MaintenanceWindow,
    OperatorActionRecord,
    ScalarField,
    SupervisoryAlarmState,
    build_historian_trend_tags,
    build_reference_operator_historian_semantics,
    build_supervisory_records,
    derive_alarm_state,
    reference_area_model,
)


class CfdOperatorSemanticsTests(unittest.TestCase):
    def test_historian_trend_tags_are_derived_from_historian_records(self) -> None:
        model = reference_area_model("dosing")
        scalar = ScalarField.uniform(
            model.mesh, name="chlorine", units="mg/L", value=0.7
        )
        records = build_supervisory_records(model, scalar=scalar)

        tags = build_historian_trend_tags(records, sampling_interval_seconds=10.0)

        self.assertTrue(tags)
        for tag in tags:
            tag.validate()
            self.assertEqual(tag.area_id, "dosing")
            self.assertEqual(tag.sampling_interval_seconds, 10.0)
            self.assertEqual(tag.evidence_status, "simulated_metadata")
            self.assertIn(
                "not operational historian evidence", " ".join(tag.limitations)
            )

    def test_advisory_alarm_states_do_not_claim_incidents(self) -> None:
        record = next(
            item
            for item in build_supervisory_records(reference_area_model("dosing"))
            if item.variable == "mesh_cell_count"
        )

        high = derive_alarm_state(record, threshold_high=1.0)
        normal = derive_alarm_state(record, threshold_high=1000.0)

        self.assertEqual(high.state, "advisory_high")
        self.assertEqual(high.priority, "review")
        self.assertEqual(normal.state, "normal")
        self.assertEqual(normal.priority, "info")
        self.assertIn(
            "not incident, safety, or operational alarm evidence",
            " ".join(high.limitations),
        )

    def test_operator_action_and_maintenance_window_are_caveated(self) -> None:
        window = MaintenanceWindow(
            window_id="MW-DOSING-001",
            area_id="dosing",
            start_ms=0,
            end_ms=120000,
            purpose="synthetic dosing adjustment review",
        )
        action = OperatorActionRecord(
            action_id="OP-DOSING-001",
            area_id="dosing",
            action_kind="setpoint_change",
            variable="chlorine_setpoint",
            previous_value=0.7,
            requested_value=1.0,
            units="mg/L",
            operating_mode="maintenance",
            maintenance_window_id=window.window_id,
        )

        window.validate()
        action.validate()

        self.assertEqual(window.evidence_status, "simulated_metadata")
        self.assertEqual(action.evidence_status, "simulated_metadata")
        self.assertIn(
            "not a real operator action or authorization record",
            " ".join(action.limitations),
        )

        with self.assertRaises(ValueError):
            MaintenanceWindow(
                window_id="bad-window",
                area_id="dosing",
                start_ms=100,
                end_ms=100,
                purpose="bad",
            ).validate()

    def test_engineering_event_is_not_vendor_tool_evidence(self) -> None:
        event = EngineeringWorkstationEvent(
            event_id="ENG-DOSING-001",
            area_id="dosing",
            event_kind="profile_validation",
            target="synthetic model profile",
            operating_mode="maintenance",
            source_profile_id="engineering-model-context",
        )

        event.validate()

        self.assertEqual(event.evidence_status, "simulated_metadata")
        self.assertIn(
            "not a real configuration change or vendor engineering tool event",
            " ".join(event.limitations),
        )

        with self.assertRaises(ValueError):
            EngineeringWorkstationEvent(
                event_id="bad-event",
                area_id="dosing",
                event_kind="firmware_upload",
                target="controller",
                operating_mode="maintenance",
                source_profile_id="engineering-model-context",
            ).validate()

    def test_alarm_identity_overclaims_are_rejected(self) -> None:
        alarm = SupervisoryAlarmState(
            alarm_id="ALM-DOSING-001",
            area_id="dosing",
            variable="chlorine_mean",
            value=2.0,
            state="confirmed_incident",
            threshold_low=None,
            threshold_high=1.0,
            source_record_id="record-1",
        )

        with self.assertRaises(ValueError):
            alarm.validate()

    def test_reference_operator_historian_bundle_is_complete(self) -> None:
        bundle = build_reference_operator_historian_semantics("disinfection")

        bundle.validate()

        self.assertEqual(bundle.area_id, "disinfection")
        self.assertTrue(bundle.trend_tags)
        self.assertTrue(bundle.alarm_states)
        self.assertTrue(bundle.maintenance_windows)
        self.assertTrue(bundle.operator_actions)
        self.assertTrue(bundle.engineering_events)
        self.assertEqual(bundle.evidence_status, "simulated_metadata")


if __name__ == "__main__":
    unittest.main()
