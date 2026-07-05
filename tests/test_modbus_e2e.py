import socket
import time
import importlib.util
import unittest

if importlib.util.find_spec("pymodbus") is None:
    raise RuntimeError(
        "pymodbus is required for the full HydraSim test gate; "
        'install with `pip install -e ".[modbus]"`'
    )

from pymodbus.client import ModbusTcpClient

from hydrasim.__main__ import (
    apply_actuator_commands,
    initialize_actuators,
    initialize_modbus_defaults,
    initialize_sensors,
    read_all_sensors,
    read_modbus_commands,
    read_modbus_dosing_concentrations,
    read_modbus_enable_bits,
    step_actuators_into_boundary,
    update_modbus_inputs,
)
from hydrasim.core import (
    BoundaryConditions,
    IntegratedCSTR,
    ReactorConfiguration,
)
from hydrasim.modbus import (
    ModbusDecoder,
    ModbusEncoder,
    ModbusRegisterMap,
    ModbusServerConfig,
    ModbusSlave,
)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class ModbusPlantHarness:
    def __init__(self):
        self.sim_time = 0.0
        self.sim_start = time.monotonic()

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
            inlet_pH=7.5,
            inlet_chlorine=0.0,
            inlet_temperature=20.0,
            acid_flow_rate=0.0,
            acid_concentration=0.1,
            chlorine_flow_rate=0.0,
        )
        self.actuators = initialize_actuators(self.boundary)
        self.sensors = initialize_sensors(self.config, self.sim_start, verbose=False)

        # Accelerate tests: bypass warmup so sensor values are immediately available.
        for sensor in self.sensors.values():
            sensor.power_on_time -= sensor.warmup_time_s + 5.0

        self.reg_map = ModbusRegisterMap()
        self.port = _free_port()
        self.slave = ModbusSlave(
            self.reg_map,
            ModbusServerConfig(host="127.0.0.1", port=self.port, unit_id=1),
        )
        self.slave.start(blocking=False)
        initialize_modbus_defaults(self.slave, self.boundary)

        self.client = ModbusTcpClient("127.0.0.1", port=self.port)
        deadline = time.time() + 3.0
        connected = False
        while time.time() < deadline:
            if self.client.connect():
                connected = True
                break
            time.sleep(0.05)
        if not connected:
            raise RuntimeError("Modbus client failed to connect to local server")

    def close(self):
        try:
            self.client.close()
        finally:
            self.slave.stop()

    def _write_float_hr(self, address: int, value: float):
        high, low = ModbusEncoder.float32_to_registers(value)
        response = self.client.write_registers(address, [high, low], device_id=1)
        if response.isError():
            raise AssertionError(
                f"Write holding register failed at {address}: {response}"
            )

    def _write_coil(self, address: int, value: bool):
        response = self.client.write_coil(address, value, device_id=1)
        if response.isError():
            raise AssertionError(f"Write coil failed at {address}: {response}")

    def read_float_input(self, address: int) -> float:
        response = self.client.read_input_registers(address, count=2, device_id=1)
        if response.isError():
            raise AssertionError(f"Read input register failed at {address}: {response}")
        return ModbusDecoder.registers_to_float32(
            response.registers[0], response.registers[1]
        )

    def step(self, dt: float = 1.0):
        commands = read_modbus_commands(self.slave)
        enable_bits = read_modbus_enable_bits(self.slave)
        acid_conc, chlorine_conc = read_modbus_dosing_concentrations(
            self.slave, self.boundary
        )
        self.boundary.acid_concentration = acid_conc
        self.boundary.chlorine_concentration = chlorine_conc

        if enable_bits[2]:
            apply_actuator_commands(
                self.actuators, commands, enable_bits, self.boundary.inlet_flow_rate
            )
            step_actuators_into_boundary(self.actuators, self.boundary, dt)
            state = self.reactor.step(dt, self.boundary)
        else:
            state = self.reactor.state

        current_time = self.sim_start + self.sim_time
        readings = read_all_sensors(self.sensors, state, current_time, verbose=False)
        update_modbus_inputs(self.slave, readings, self.sim_time)
        self.sim_time += dt
        return readings


class TestModbusPlantEndToEnd(unittest.TestCase):
    def setUp(self):
        self.harness = ModbusPlantHarness()

    def tearDown(self):
        self.harness.close()

    def test_holding_register_commands_drive_actuators(self):
        # Holding registers: acid=0, chlorine=2, inlet=4
        self.harness._write_float_hr(0, 1.2)
        self.harness._write_float_hr(2, 0.7)
        self.harness._write_float_hr(4, 7.0)

        for _ in range(8):
            self.harness.step(1.0)

        self.assertGreater(self.harness.boundary.acid_flow_rate, 0.2)
        self.assertGreater(self.harness.boundary.chlorine_flow_rate, 0.2)
        self.assertGreater(self.harness.boundary.inlet_flow_rate, 5.5)

    def test_coils_enable_disable_dosing(self):
        self.harness._write_float_hr(2, 0.8)  # chlorine command

        # Disable chlorine pump coil (address 1)
        self.harness._write_coil(1, False)
        for _ in range(5):
            self.harness.step(1.0)
        self.assertLess(self.harness.boundary.chlorine_flow_rate, 0.05)

        # Enable chlorine pump and ensure flow appears
        self.harness._write_coil(1, True)
        for _ in range(5):
            self.harness.step(1.0)
        self.assertGreater(self.harness.boundary.chlorine_flow_rate, 0.2)

    def test_simulation_running_coil_pauses_and_resumes_physics(self):
        t0 = self.harness.reactor.state.time

        # Pause simulation (coil address 2)
        self.harness._write_coil(2, False)
        for _ in range(3):
            self.harness.step(1.0)
        self.assertAlmostEqual(self.harness.reactor.state.time, t0, places=6)

        # Resume simulation
        self.harness._write_coil(2, True)
        for _ in range(3):
            self.harness.step(1.0)
        self.assertGreater(self.harness.reactor.state.time, t0)

    def test_sensor_values_are_published_to_modbus_input_registers(self):
        last_readings = None
        for _ in range(3):
            last_readings = self.harness.step(1.0)

        self.assertIsNotNone(last_readings)
        pH_in = self.harness.read_float_input(0)
        pH_mid = self.harness.read_float_input(2)
        pH_out = self.harness.read_float_input(4)
        cl_in = self.harness.read_float_input(6)
        temp_in = self.harness.read_float_input(12)

        self.assertAlmostEqual(pH_in, last_readings["pH_inlet"].value, places=3)
        self.assertAlmostEqual(pH_mid, last_readings["pH_middle"].value, places=3)
        self.assertAlmostEqual(pH_out, last_readings["pH_outlet"].value, places=3)
        self.assertAlmostEqual(cl_in, last_readings["chlorine_inlet"].value, places=3)
        self.assertAlmostEqual(temp_in, last_readings["temp_inlet"].value, places=3)

    def test_dosing_concentration_registers_affect_boundary(self):
        self.harness._write_float_hr(10, 0.8)  # acid concentration [mol/L]
        self.harness._write_float_hr(12, 90.0)  # chlorine concentration [mg/L]

        self.harness.step(1.0)

        self.assertAlmostEqual(self.harness.boundary.acid_concentration, 0.8, places=3)
        self.assertAlmostEqual(
            self.harness.boundary.chlorine_concentration, 90.0, places=3
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
