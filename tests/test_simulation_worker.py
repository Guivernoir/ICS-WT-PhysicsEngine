from __future__ import annotations

import unittest
from unittest.mock import patch

from hydrasim import simulation_worker
from hydrasim.simulation_worker import SimulationCore, handle_request


class SimulationWorkerTests(unittest.TestCase):
    def test_worker_tick_returns_physical_snapshot(self) -> None:
        core = SimulationCore()
        response = handle_request(
            core,
            {
                "type": "tick",
                "dt": 1.0,
                "commands": {
                    "acid_flow": 0.0,
                    "chlorine_flow": 0.25,
                    "inlet_flow": 5.0,
                    "acid_concentration": 0.1,
                    "chlorine_concentration": 60.0,
                },
                "coils": {
                    "acid_pump_enable": True,
                    "chlorine_pump_enable": True,
                    "simulation_running": True,
                },
            },
        )

        self.assertIs(response["ok"], True)
        snapshot = response["snapshot"]
        self.assertEqual(snapshot["elapsed_seconds"], 1.0)
        self.assertGreaterEqual(snapshot["ph_outlet"], 0.0)
        self.assertLessEqual(snapshot["ph_outlet"], 14.0)
        self.assertGreaterEqual(snapshot["flow_rate"], 0.0)

    def test_worker_rejects_unknown_request_type(self) -> None:
        response = handle_request(SimulationCore(), {"type": "network"})

        self.assertIs(response["ok"], False)
        self.assertIn("unknown request type", response["error"])

    def test_main_handles_keyboard_interrupt_as_clean_shutdown(self) -> None:
        with patch.object(simulation_worker, "serve", side_effect=KeyboardInterrupt):
            self.assertEqual(simulation_worker.main(), 0)


if __name__ == "__main__":
    unittest.main()
