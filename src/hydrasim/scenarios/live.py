"""Live Modbus scenario replay helpers."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .executor import run_scenario, run_scenario_with_clients
from .loader import load_scenario_json
from .profiles import get_scenario


def run_live_scenario(
    scenario_id: str,
    host: str,
    port: int,
    unit_id: int = 1,
    time_scale: float = 1.0,
    custom_json: str | Path | None = None,
) -> None:
    try:
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError as exc:
        if exc.name == "pymodbus":
            raise RuntimeError("live scenario replay requires pymodbus")
        raise

    scenario = (
        load_scenario_json(custom_json) if custom_json else get_scenario(scenario_id)
    )
    client = ModbusTcpClient(host, port=port)
    if not client.connect():
        raise RuntimeError(f"could not connect to Modbus endpoint {host}:{port}")
    try:
        run_scenario(client, scenario, unit_id=unit_id, time_scale=time_scale)
    finally:
        client.close()


def run_live_scenario_targets(
    scenario_id: str,
    targets: dict[str, tuple[str, int]],
    unit_id: int = 1,
    time_scale: float = 1.0,
    custom_json: str | Path | None = None,
) -> None:
    try:
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError as exc:
        if exc.name == "pymodbus":
            raise RuntimeError("live scenario replay requires pymodbus")
        raise

    scenario = (
        load_scenario_json(custom_json) if custom_json else get_scenario(scenario_id)
    )
    server_ids = {transaction.server_id for transaction in scenario.transactions}
    missing = sorted(server_ids - set(targets))
    if missing:
        raise RuntimeError(f"missing live target mapping for: {', '.join(missing)}")

    clients = {}
    try:
        for server_id in sorted(server_ids):
            host, port = targets[server_id]
            client = ModbusTcpClient(host, port=port)
            if not client.connect():
                raise RuntimeError(f"could not connect {server_id} at {host}:{port}")
            clients[server_id] = client
        run_scenario_with_clients(
            clients,
            scenario,
            unit_id=unit_id,
            time_scale=time_scale,
        )
    finally:
        for client in clients.values():
            client.close()


def start_live_scenario_thread(
    scenario_id: str,
    host: str,
    port: int,
    unit_id: int = 1,
    time_scale: float = 1.0,
    startup_delay_s: float = 1.0,
    custom_json: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> threading.Thread:
    log = logger or logging.getLogger(__name__)

    def target() -> None:
        if startup_delay_s > 0.0:
            time.sleep(startup_delay_s)
        try:
            run_live_scenario(
                scenario_id,
                host,
                port,
                unit_id=unit_id,
                time_scale=time_scale,
                custom_json=custom_json,
            )
            log.info("Scenario replay completed: %s", scenario_id)
        except Exception as exc:
            log.error("Scenario replay failed: %s", exc)

    thread = threading.Thread(
        target=target,
        daemon=True,
        name=f"HydraSimScenario:{scenario_id}",
    )
    thread.start()
    return thread
