import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from hydrasim.scenarios import (  # noqa: E402
    export_lab_bundle,
    get_scenario,
    load_scenario_json,
    render_markdown_summary,
    render_pcap_bytes,
    render_results_csv,
    render_transcript_csv,
    run_scenario,
    run_scenario_with_clients,
    scenario_ids,
    validate_scenario,
)


class TestMvpModbusScenarios(unittest.TestCase):
    def test_expected_scenario_ids_are_available(self):
        self.assertEqual(
            scenario_ids(),
            (
                "MVP-MB-HYDRA-002",
                "MVP-MB-HYDRA-003",
                "MVP-MB-HYDRA-004",
                "MVP-MB-HYDRA-005",
                "MVP-MB-HYDRA-006",
                "MVP-MB-HYDRA-007",
                "MVP-MB-HYDRA-008",
                "MVP-MB-HYDRA-009",
            ),
        )

    def test_built_in_scenarios_pass_validation(self):
        for scenario_id in scenario_ids():
            scenario = get_scenario(scenario_id)
            self.assertEqual(validate_scenario(scenario), ())

    def test_field_points_are_not_network_assets_by_default(self):
        scenario = get_scenario("MVP-MB-HYDRA-002")
        self.assertGreaterEqual(len(scenario.field_points), 5)

        endpoint_ids = {node.node_id for node in scenario.nodes}
        for point in scenario.field_points:
            self.assertIn(point.endpoint_id, endpoint_ids)
            self.assertNotIn("02:00", point.notes)
            self.assertIn("simulated field point", point.notes)

    def test_passive_observer_is_not_a_modbus_participant(self):
        scenario = get_scenario("MVP-MB-HYDRA-003")
        observers = [node for node in scenario.nodes if node.node_id == "observer"]
        self.assertEqual(len(observers), 1)
        self.assertEqual(observers[0].role, "passive observer only")

        actors = {transaction.actor_id for transaction in scenario.transactions}
        self.assertNotIn("observer", actors)

    def test_unknown_host_scenario_contains_review_candidates(self):
        scenario = get_scenario("MVP-MB-HYDRA-003")
        unknown_transactions = [
            tx for tx in scenario.transactions if tx.actor_id == "unknown"
        ]
        self.assertGreaterEqual(len(unknown_transactions), 3)
        self.assertTrue(
            any(tx.review_hint == "must_review" for tx in unknown_transactions)
        )
        self.assertTrue(
            any(tx.review_hint == "enumeration_like" for tx in unknown_transactions)
        )
        self.assertTrue(
            any(tx.review_hint == "exception_pattern" for tx in unknown_transactions)
        )

    def test_csv_export_is_deterministic_and_has_expected_columns(self):
        scenario = get_scenario("MVP-MB-HYDRA-002")
        first = render_transcript_csv(scenario)
        second = render_transcript_csv(scenario)

        self.assertEqual(first, second)
        self.assertTrue(
            first.startswith(
                "ordinal,timestamp_ms,actor_id,server_id,function_code,operation"
            )
        )
        self.assertIn("historian,hydra_a,4,read_input_registers", first)
        self.assertIn("maintenance,hydra_a,5,write_single_coil", first)
        self.assertIn("wire_values", first)

    def test_markdown_summary_explains_passive_and_simulated_limits(self):
        scenario = get_scenario("MVP-MB-HYDRA-003")
        rendered = render_markdown_summary(scenario)

        self.assertIn("Passive capture observer", rendered)
        self.assertIn("Supplied Simulated Field Points", rendered)
        self.assertIn("not an incident or attack claim", rendered)

    def test_aliases_load_canonical_scenarios(self):
        canonical = get_scenario("MVP-MB-HYDRA-002")
        alias = get_scenario("water-treatment-normal")
        self.assertEqual(alias.scenario_id, canonical.scenario_id)

        smart = get_scenario("water-treatment-smart-field")
        self.assertEqual(smart.scenario_id, "MVP-MB-HYDRA-004")

        maintenance = get_scenario("water-treatment-maintenance-window")
        self.assertEqual(maintenance.scenario_id, "MVP-MB-HYDRA-005")

        noisy = get_scenario("water-treatment-noisy-network")
        self.assertEqual(noisy.scenario_id, "MVP-MB-HYDRA-008")

        degraded = get_scenario("water-treatment-degraded-operations")
        self.assertEqual(degraded.scenario_id, "MVP-MB-HYDRA-009")

    def test_smart_field_scenario_models_multiple_server_endpoints(self):
        scenario = get_scenario("MVP-MB-HYDRA-004")
        server_ids = {tx.server_id for tx in scenario.transactions}
        self.assertEqual(server_ids, {"hydra_a", "ph_probe_a", "chlorine_pump_a"})

        field_endpoint_ids = {point.endpoint_id for point in scenario.field_points}
        self.assertIn("ph_probe_a", field_endpoint_ids)
        self.assertIn("chlorine_pump_a", field_endpoint_ids)

    def test_expanded_scenario_library_covers_hs6_families(self):
        families = {
            "MVP-MB-HYDRA-005": "maintenance_precheck",
            "MVP-MB-HYDRA-006": "process_deviation_context",
            "MVP-MB-HYDRA-007": "must_review",
            "MVP-MB-HYDRA-008": "exception_pattern",
            "MVP-MB-HYDRA-009": "process_deviation_context",
        }
        for scenario_id, expected_hint in families.items():
            scenario = get_scenario(scenario_id)
            hints = {tx.review_hint for tx in scenario.transactions}
            labels = {tx.scenario_label for tx in scenario.transactions}
            self.assertTrue(expected_hint in hints or expected_hint in labels)
            self.assertIn(
                "HydraSim traffic is synthetic process simulation, not operational evidence.",
                scenario.limitations,
            )

    def test_expanded_scenarios_have_deterministic_pcaps(self):
        for scenario_id in ("MVP-MB-HYDRA-005", "MVP-MB-HYDRA-006", "MVP-MB-HYDRA-007"):
            scenario = get_scenario(scenario_id)
            first = render_pcap_bytes(scenario)
            second = render_pcap_bytes(scenario)

            self.assertEqual(first, second)
            self.assertEqual(
                self._pcap_record_count(first), len(scenario.transactions) * 2
            )
            self.assertIn(bytes.fromhex("020000003020"), first)

    def test_custom_json_scenario_loader(self):
        payload = {
            "scenario_id": "CUSTOM-001",
            "name": "Custom read-only check",
            "purpose": "Exercise custom scenario loading.",
            "nodes": [
                {
                    "node_id": "client",
                    "label": "Custom client",
                    "mac": "02:00:00:00:40:10",
                    "ipv4": "203.0.113.10",
                    "role": "client",
                },
                {
                    "node_id": "hydra",
                    "label": "HydraSim endpoint",
                    "mac": "02:00:00:00:40:20",
                    "ipv4": "203.0.113.20",
                    "role": "Modbus TCP process endpoint",
                    "modbus_port": 502,
                },
            ],
            "field_points": [],
            "transactions": [
                {
                    "ordinal": 1,
                    "timestamp_ms": 0,
                    "actor_id": "client",
                    "server_id": "hydra",
                    "function_code": 4,
                    "operation": "read_input_registers",
                    "address": 0,
                    "quantity": 2,
                    "response": "ok",
                    "scenario_label": "custom_read",
                }
            ],
            "limitations": ["custom synthetic scenario"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            scenario = load_scenario_json(path)

        self.assertEqual(scenario.scenario_id, "CUSTOM-001")
        self.assertEqual(scenario.transactions[0].operation, "read_input_registers")

    def test_custom_json_rejects_passive_observer_transactions(self):
        payload = {
            "scenario_id": "BAD-001",
            "name": "Invalid passive actor",
            "purpose": "Reject passive observer as an actor.",
            "nodes": [
                {
                    "node_id": "observer",
                    "label": "Passive observer",
                    "mac": "not-transmitting",
                    "ipv4": "not-transmitting",
                    "role": "passive observer only",
                },
                {
                    "node_id": "hydra",
                    "label": "HydraSim endpoint",
                    "mac": "02:00:00:00:40:20",
                    "ipv4": "203.0.113.20",
                    "role": "Modbus TCP process endpoint",
                    "modbus_port": 502,
                },
            ],
            "field_points": [],
            "transactions": [
                {
                    "ordinal": 1,
                    "timestamp_ms": 0,
                    "actor_id": "observer",
                    "server_id": "hydra",
                    "function_code": 4,
                    "operation": "read_input_registers",
                    "address": 0,
                    "quantity": 2,
                    "response": "ok",
                    "scenario_label": "bad",
                }
            ],
            "limitations": ["bad synthetic scenario"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "passive observer"):
                load_scenario_json(path)

    def test_custom_json_rejects_incomplete_write_values(self):
        payload = {
            "scenario_id": "BAD-002",
            "name": "Invalid write",
            "purpose": "Reject missing deterministic wire values.",
            "nodes": [
                {
                    "node_id": "client",
                    "label": "Client",
                    "mac": "02:00:00:00:40:10",
                    "ipv4": "203.0.113.10",
                    "role": "client",
                },
                {
                    "node_id": "hydra",
                    "label": "HydraSim endpoint",
                    "mac": "02:00:00:00:40:20",
                    "ipv4": "203.0.113.20",
                    "role": "Modbus TCP process endpoint",
                    "modbus_port": 502,
                },
            ],
            "field_points": [],
            "transactions": [
                {
                    "ordinal": 1,
                    "timestamp_ms": 0,
                    "actor_id": "client",
                    "server_id": "hydra",
                    "function_code": 16,
                    "operation": "write_multiple_registers",
                    "address": 4,
                    "quantity": 2,
                    "response": "ok",
                    "scenario_label": "bad_write",
                    "wire_values": [1],
                }
            ],
            "limitations": ["bad synthetic scenario"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "quantity-matched wire_values"):
                load_scenario_json(path)

    def test_command_center_executor_maps_supported_modbus_actions(self):
        class Response:
            def isError(self):
                return False

        class FakeClient:
            def __init__(self):
                self.calls = []

            def read_input_registers(self, address, count, device_id):
                self.calls.append(("ir", address, count, device_id))
                return Response()

            def read_holding_registers(self, address, count, device_id):
                self.calls.append(("hr", address, count, device_id))
                return Response()

            def write_coil(self, address, value, device_id):
                self.calls.append(("coil", address, value, device_id))
                return Response()

            def write_register(self, address, value, device_id):
                self.calls.append(("wr", address, value, device_id))
                return Response()

            def write_registers(self, address, values, device_id):
                self.calls.append(("wrs", address, values, device_id))
                return Response()

        scenario = get_scenario("MVP-MB-HYDRA-002")
        client = FakeClient()
        results = run_scenario(client, scenario, time_scale=0.0)
        rendered = render_results_csv(results)

        self.assertEqual(len(results), len(scenario.transactions))
        self.assertIn(("wrs", 4, [16608, 0], 1), client.calls)
        self.assertIn(("coil", 10, True, 1), client.calls)
        self.assertIn("ordinal,actor_id,function_code,address,status,detail", rendered)

    def test_multi_endpoint_executor_routes_by_server_id(self):
        class Response:
            def isError(self):
                return False

        class FakeClient:
            def __init__(self, name):
                self.name = name
                self.calls = []

            def read_input_registers(self, address, count, device_id):
                self.calls.append((self.name, "ir", address, count, device_id))
                return Response()

            def read_holding_registers(self, address, count, device_id):
                self.calls.append((self.name, "hr", address, count, device_id))
                return Response()

            def write_registers(self, address, values, device_id):
                self.calls.append((self.name, "wrs", address, values, device_id))
                return Response()

        scenario = get_scenario("MVP-MB-HYDRA-004")
        clients = {
            "hydra_a": FakeClient("hydra"),
            "ph_probe_a": FakeClient("ph"),
            "chlorine_pump_a": FakeClient("pump"),
        }
        results = run_scenario_with_clients(clients, scenario, time_scale=0.0)

        self.assertEqual(len(results), len(scenario.transactions))
        self.assertIn(("ph", "ir", 0, 2, 1), clients["ph_probe_a"].calls)
        self.assertIn(
            ("pump", "wrs", 0, [16025, 39322], 1),
            clients["chlorine_pump_a"].calls,
        )

    def test_multi_endpoint_executor_rejects_missing_server_client(self):
        class Response:
            def isError(self):
                return False

        class FakeClient:
            def read_input_registers(self, address, count, device_id):
                return Response()

            def write_registers(self, address, values, device_id):
                return Response()

        scenario = get_scenario("MVP-MB-HYDRA-004")
        with self.assertRaisesRegex(ValueError, "ph_probe_a"):
            run_scenario_with_clients(
                {"hydra_a": FakeClient()},
                scenario,
                time_scale=0.0,
            )

    def test_pcap_export_is_deterministic_classic_ethernet(self):
        scenario = get_scenario("MVP-MB-HYDRA-002")
        first = render_pcap_bytes(scenario)
        second = render_pcap_bytes(scenario)

        self.assertEqual(first, second)
        self.assertEqual(first[:4], b"\xd4\xc3\xb2\xa1")
        self.assertEqual(self._pcap_record_count(first), len(scenario.transactions) * 2)
        self.assertIn(bytes.fromhex("020000003010"), first)
        self.assertIn(bytes.fromhex("020000003020"), first)

    def test_pcap_export_includes_smart_field_endpoint_macs(self):
        scenario = get_scenario("MVP-MB-HYDRA-004")
        exported = render_pcap_bytes(scenario)

        self.assertEqual(
            self._pcap_record_count(exported), len(scenario.transactions) * 2
        )
        self.assertIn(bytes.fromhex("020000003130"), exported)
        self.assertIn(bytes.fromhex("020000003131"), exported)

    def test_pcap_cli_requires_output_path(self):
        from hydrasim.scenarios.run import main

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["water-treatment-normal", "--format", "pcap"])

    def test_lab_bundle_export_creates_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bundle"
            written = export_lab_bundle("water-treatment-smart-field", target)

            names = {path.name for path in written}
            self.assertEqual(
                names,
                {
                    "transcript.csv",
                    "summary.md",
                    "scenario.pcap",
                    "capture-notes.md",
                    "manifest.json",
                    "checksums.sha256",
                },
            )
            manifest = json.loads((target / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["scenario_id"], "MVP-MB-HYDRA-004")
            self.assertEqual(manifest["source_class"], "SyntheticScenario")
            self.assertEqual(
                (target / "scenario.pcap").read_bytes()[:4],
                b"\xd4\xc3\xb2\xa1",
            )

            checksum_lines = (
                (target / "checksums.sha256").read_text("utf-8").splitlines()
            )
            checksum_names = {line.split("  ", 1)[1] for line in checksum_lines}
            self.assertIn("scenario.pcap", checksum_names)
            expected = hashlib.sha256(
                (target / "manifest.json").read_bytes()
            ).hexdigest()
            self.assertIn(f"{expected}  manifest.json", checksum_lines)

    def test_lab_bundle_export_uses_create_new_directory_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bundle"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                export_lab_bundle("water-treatment-normal", target)

    @staticmethod
    def _pcap_record_count(data: bytes) -> int:
        offset = 24
        count = 0
        while offset < len(data):
            incl_len = int.from_bytes(data[offset + 8 : offset + 12], "little")
            offset += 16 + incl_len
            count += 1
        return count


if __name__ == "__main__":
    unittest.main()
