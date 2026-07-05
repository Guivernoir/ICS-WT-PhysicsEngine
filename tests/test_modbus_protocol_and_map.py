import unittest

from hydrasim.modbus import ModbusEncoder, ModbusDecoder, ModbusRegisterMap


class TestModbusProtocolAndMap(unittest.TestCase):
    def test_float_roundtrip(self):
        values = [0.0, 1.0, -1.0, 7.25, 3.14159, 123.456]
        for original in values:
            high, low = ModbusEncoder.float32_to_registers(original)
            decoded = ModbusDecoder.registers_to_float32(high, low)
            self.assertAlmostEqual(decoded, original, places=4)

    def test_int16_roundtrip(self):
        values = [-32768, -10, 0, 10, 32767]
        for original in values:
            reg = ModbusEncoder.int16_to_register(original)
            decoded = ModbusDecoder.register_to_int16(reg)
            self.assertEqual(decoded, original)

    def test_uint16_roundtrip(self):
        values = [0, 10, 32767, 65535]
        for original in values:
            reg = ModbusEncoder.uint16_to_register(original)
            decoded = ModbusDecoder.register_to_uint16(reg)
            self.assertEqual(decoded, original)

    def test_register_map_contains_required_registers(self):
        reg_map = ModbusRegisterMap()
        required = [
            "pH_inlet",
            "pH_middle",
            "pH_outlet",
            "chlorine_inlet",
            "chlorine_outlet",
            "flow_rate",
            "temperature_inlet",
            "temperature_outlet",
            "acid_flow_rate",
            "chlorine_flow_rate",
            "inlet_flow_rate",
            "acid_concentration",
            "chlorine_concentration",
            "acid_pump_enable",
            "chlorine_pump_enable",
            "simulation_running",
        ]
        for name in required:
            self.assertIsNotNone(reg_map.get_register_by_name(name), name)


if __name__ == "__main__":
    unittest.main()
