"""Stdio bridge that exposes HydraSim process physics without network ownership."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from .core import BoundaryConditions, IntegratedCSTR, ReactorConfiguration
from .runtime_loop import (
    apply_actuator_commands,
    initialize_actuators,
    initialize_sensors,
    read_all_sensors,
    step_actuators_into_boundary,
    validate_concentration,
    validate_flow_rate,
)
from .sensors import SensorFault, SensorReading


@dataclass
class WorkerCommands:
    acid_flow: float = 0.0
    chlorine_flow: float = 0.2
    inlet_flow: float = 5.0
    acid_concentration: float = 0.1
    chlorine_concentration: float = 60.0


@dataclass
class WorkerCoils:
    acid_pump_enable: bool = True
    chlorine_pump_enable: bool = True
    simulation_running: bool = True


class SimulationCore:
    """Owns physical simulation state for the Rust runtime bridge."""

    def __init__(self) -> None:
        self._initialize_state()

    def _initialize_state(self) -> None:
        self.config = ReactorConfiguration(
            volume=1000.0,
            n_zones=5,
            flow_rate=5.0,
            initial_pH=7.2,
            initial_chlorine=2.0,
            temperature=20.0,
        )
        self.reactor = IntegratedCSTR(self.config)
        self.boundary = BoundaryConditions(
            inlet_flow_rate=5.0,
            inlet_pH=self.config.inlet_pH,
            inlet_chlorine=self.config.inlet_chlorine,
            inlet_chloramine=self.config.inlet_chloramine,
            inlet_ammonia=self.config.inlet_ammonia,
            inlet_chlorine_demand=self.config.inlet_chlorine_demand,
            inlet_temperature=self.config.inlet_temperature,
            acid_flow_rate=0.0,
            acid_concentration=0.1,
            chlorine_flow_rate=0.0,
        )
        self.actuators = initialize_actuators(self.boundary)
        self.sensors = initialize_sensors(self.config, sim_start_time=0.0)
        self.state = self.reactor.state
        self.sim_time = 0.0
        self.readings = read_all_sensors(self.sensors, self.state, self.sim_time)
        self.commands = WorkerCommands()
        self.coils = WorkerCoils()

    def reset(self) -> dict[str, Any]:
        self._initialize_state()
        return self.snapshot()

    def tick(self, request: dict[str, Any]) -> dict[str, Any]:
        dt = validate_flow_rate(request.get("dt", 1.0), max_value=60.0)
        self.commands = parse_commands(request.get("commands"))
        self.coils = parse_coils(request.get("coils"))
        self.boundary.acid_concentration = self.commands.acid_concentration
        self.boundary.chlorine_concentration = self.commands.chlorine_concentration

        if self.coils.simulation_running:
            apply_actuator_commands(
                self.actuators,
                (
                    self.commands.acid_flow,
                    self.commands.chlorine_flow,
                    self.commands.inlet_flow,
                ),
                (
                    self.coils.acid_pump_enable,
                    self.coils.chlorine_pump_enable,
                    self.coils.simulation_running,
                ),
                self.boundary.inlet_flow_rate,
            )
            if dt > 0.0:
                step_actuators_into_boundary(self.actuators, self.boundary, dt)
                self.state = self.reactor.step(dt, boundary=self.boundary)
                self.sim_time += dt

        self.readings = read_all_sensors(self.sensors, self.state, self.sim_time)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        any_fault = any(
            reading.fault != SensorFault.NONE for reading in self.readings.values()
        )
        return {
            "elapsed_seconds": self.sim_time,
            "ph_inlet": reading_value(self.readings, "pH_inlet", self.state.pH[0]),
            "ph_middle": reading_value(
                self.readings, "pH_middle", self.state.pH[len(self.state.pH) // 2]
            ),
            "ph_outlet": reading_value(self.readings, "pH_outlet", self.state.pH[-1]),
            "chlorine_inlet": reading_value(
                self.readings, "chlorine_inlet", self.state.chlorine[0]
            ),
            "chlorine_outlet": reading_value(
                self.readings, "chlorine_outlet", self.state.chlorine[-1]
            ),
            "flow_rate": reading_value(
                self.readings, "flow_main", self.state.flow_rate
            ),
            "temperature_inlet": reading_value(
                self.readings, "temp_inlet", self.state.temperature[0]
            ),
            "temperature_outlet": reading_value(
                self.readings, "temp_outlet", self.state.temperature[-1]
            ),
            "system_status": 1 if any_fault else 0,
            "acid_flow": self.boundary.acid_flow_rate,
            "chlorine_flow": self.boundary.chlorine_flow_rate,
            "inlet_flow": self.boundary.inlet_flow_rate,
            "acid_concentration": self.boundary.acid_concentration,
            "chlorine_concentration": self.boundary.chlorine_concentration,
            "acid_enabled": self.coils.acid_pump_enable,
            "chlorine_enabled": self.coils.chlorine_pump_enable,
            "simulation_running": self.coils.simulation_running,
        }


def parse_commands(raw: Any) -> WorkerCommands:
    data = raw if isinstance(raw, dict) else {}
    return WorkerCommands(
        acid_flow=validate_flow_rate(data.get("acid_flow", 0.0), max_value=2.0),
        chlorine_flow=validate_flow_rate(data.get("chlorine_flow", 0.2), max_value=1.0),
        inlet_flow=validate_flow_rate(data.get("inlet_flow", 5.0), max_value=20.0),
        acid_concentration=validate_concentration(
            data.get("acid_concentration", 0.1), max_value=5.0
        ),
        chlorine_concentration=validate_concentration(
            data.get("chlorine_concentration", 60.0), max_value=200.0
        ),
    )


def parse_coils(raw: Any) -> WorkerCoils:
    data = raw if isinstance(raw, dict) else {}
    return WorkerCoils(
        acid_pump_enable=bool(data.get("acid_pump_enable", True)),
        chlorine_pump_enable=bool(data.get("chlorine_pump_enable", True)),
        simulation_running=bool(data.get("simulation_running", True)),
    )


def reading_value(
    readings: dict[str, SensorReading], name: str, fallback: float
) -> float:
    reading = readings.get(name)
    if reading is None:
        return float(fallback)
    value = float(reading.value)
    if value != value or value in {float("inf"), float("-inf")}:
        value = float(reading.raw_value)
    if value != value or value in {float("inf"), float("-inf")}:
        return float(fallback)
    return value


def handle_request(core: SimulationCore, request: dict[str, Any]) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type == "tick":
        return {"ok": True, "snapshot": core.tick(request)}
    if request_type == "snapshot":
        return {"ok": True, "snapshot": core.snapshot()}
    if request_type == "reset":
        return {"ok": True, "snapshot": core.reset()}
    return {"ok": False, "error": f"unknown request type {request_type!r}"}


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    core = SimulationCore()
    for line in stdin:
        try:
            request = json.loads(line)
            response = handle_request(core, request)
        except Exception as exc:  # pragma: no cover - process boundary fallback
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), file=stdout, flush=True)
    return 0


def main() -> int:
    try:
        return serve()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
