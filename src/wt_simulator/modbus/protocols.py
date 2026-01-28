"""
Modbus Protocol Encoding/Decoding
==================================

Data conversion utilities for Modbus register encoding.

This module handles ONLY data format conversion:
- Python floats ↔ Modbus register pairs (IEEE 754)
- Python ints ↔ Modbus registers
- Python bools ↔ Modbus coils

No protocol logic, no control, no validation beyond data type.

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import struct
import numpy as np
from typing import List, Tuple, Union


class ModbusEncoder:
    """
    Encoder for converting Python values to Modbus register format.

    Modbus uses 16-bit registers. Multi-word values (like float32)
    are stored in consecutive registers.

    Byte Order: Big-endian (network byte order) - Modbus standard
    """

    @staticmethod
    def float32_to_registers(value: float) -> Tuple[int, int]:
        """
        Convert Python float to two 16-bit Modbus registers.

        Uses IEEE 754 single-precision (32-bit) format.

        Args:
            value: Python float

        Returns:
            Tuple of two 16-bit register values (high word, low word)

        Example:
            >>> encoder = ModbusEncoder()
            >>> high, low = encoder.float32_to_registers(7.25)
            >>> # These register values encode 7.25 as IEEE 754
        """
        # Pack as big-endian IEEE 754 single precision
        packed = struct.pack(">f", value)

        # Unpack as two unsigned 16-bit integers
        high, low = struct.unpack(">HH", packed)

        return high, low

    @staticmethod
    def int16_to_register(value: int) -> int:
        """
        Convert Python signed int to 16-bit Modbus register.

        Args:
            value: Signed integer in range [-32768, 32767]

        Returns:
            16-bit unsigned register value

        Raises:
            ValueError: If value out of range
        """
        if not -32768 <= value <= 32767:
            raise ValueError(f"int16 value {value} out of range [-32768, 32767]")

        # Pack as signed 16-bit, unpack as unsigned
        packed = struct.pack(">h", value)
        (result,) = struct.unpack(">H", packed)

        return result

    @staticmethod
    def uint16_to_register(value: int) -> int:
        """
        Convert Python unsigned int to 16-bit Modbus register.

        Args:
            value: Unsigned integer in range [0, 65535]

        Returns:
            16-bit unsigned register value

        Raises:
            ValueError: If value out of range
        """
        if not 0 <= value <= 65535:
            raise ValueError(f"uint16 value {value} out of range [0, 65535]")

        return value

    @staticmethod
    def bool_to_coil(value: bool) -> int:
        """
        Convert Python bool to Modbus coil value.

        Args:
            value: Boolean value

        Returns:
            0 (False) or 1 (True)
        """
        return 1 if value else 0

    @staticmethod
    def array_to_registers(
        values: Union[List[float], np.ndarray], data_type: str = "float32"
    ) -> List[int]:
        """
        Convert array of values to Modbus registers.

        Args:
            values: List or array of values
            data_type: 'float32', 'int16', or 'uint16'

        Returns:
            List of 16-bit register values
        """
        registers = []

        for value in values:
            if data_type == "float32":
                high, low = ModbusEncoder.float32_to_registers(float(value))
                registers.extend([high, low])
            elif data_type == "int16":
                reg = ModbusEncoder.int16_to_register(int(value))
                registers.append(reg)
            elif data_type == "uint16":
                reg = ModbusEncoder.uint16_to_register(int(value))
                registers.append(reg)
            else:
                raise ValueError(f"Unknown data type: {data_type}")

        return registers


class ModbusDecoder:
    """
    Decoder for converting Modbus register format to Python values.

    Performs the inverse operations of ModbusEncoder.
    """

    @staticmethod
    def registers_to_float32(high: int, low: int) -> float:
        """
        Convert two 16-bit Modbus registers to Python float.

        Args:
            high: High 16-bit register
            low: Low 16-bit register

        Returns:
            Python float (IEEE 754 single precision)

        Example:
            >>> decoder = ModbusDecoder()
            >>> value = decoder.registers_to_float32(16480, 0)
            >>> # value ≈ 7.25
        """
        # Pack as two unsigned 16-bit integers
        packed = struct.pack(">HH", high, low)

        # Unpack as big-endian IEEE 754 single precision
        (result,) = struct.unpack(">f", packed)

        return result

    @staticmethod
    def register_to_int16(value: int) -> int:
        """
        Convert 16-bit Modbus register to Python signed int.

        Args:
            value: 16-bit unsigned register value

        Returns:
            Signed integer in range [-32768, 32767]
        """
        # Pack as unsigned 16-bit, unpack as signed
        packed = struct.pack(">H", value)
        (result,) = struct.unpack(">h", packed)

        return result

    @staticmethod
    def register_to_uint16(value: int) -> int:
        """
        Convert 16-bit Modbus register to Python unsigned int.

        Args:
            value: 16-bit unsigned register value

        Returns:
            Unsigned integer in range [0, 65535]
        """
        return value

    @staticmethod
    def coil_to_bool(value: int) -> bool:
        """
        Convert Modbus coil value to Python bool.

        Args:
            value: Coil value (0 or non-zero)

        Returns:
            True if non-zero, False if zero
        """
        return bool(value)

    @staticmethod
    def registers_to_array(
        registers: List[int], data_type: str = "float32", count: int = None
    ) -> Union[List[float], List[int]]:
        """
        Convert Modbus registers to array of values.

        Args:
            registers: List of 16-bit register values
            data_type: 'float32', 'int16', or 'uint16'
            count: Number of values to decode (None = auto-detect)

        Returns:
            List of decoded values
        """
        values = []

        if data_type == "float32":
            # Each float32 uses 2 registers
            n_values = len(registers) // 2 if count is None else count
            for i in range(n_values):
                high = registers[2 * i]
                low = registers[2 * i + 1]
                value = ModbusDecoder.registers_to_float32(high, low)
                values.append(value)

        elif data_type == "int16":
            n_values = len(registers) if count is None else count
            for i in range(n_values):
                value = ModbusDecoder.register_to_int16(registers[i])
                values.append(value)

        elif data_type == "uint16":
            n_values = len(registers) if count is None else count
            for i in range(n_values):
                value = ModbusDecoder.register_to_uint16(registers[i])
                values.append(value)

        else:
            raise ValueError(f"Unknown data type: {data_type}")

        return values


def validate_encoding():
    """Validate encode/decode round-trip."""
    encoder = ModbusEncoder()
    decoder = ModbusDecoder()

    # Test float32
    test_values = [0.0, 1.0, -1.0, 7.25, 3.14159, 100.5, -50.3]

    for original in test_values:
        high, low = encoder.float32_to_registers(original)
        decoded = decoder.registers_to_float32(high, low)

        # Allow small floating point error
        if abs(decoded - original) > 1e-6:
            raise AssertionError(
                f"Float32 encoding failed: {original} -> {decoded}, "
                f"registers: [{high}, {low}]"
            )

    # Test int16
    test_ints = [0, 1, -1, 1000, -1000, 32767, -32768]

    for original in test_ints:
        reg = encoder.int16_to_register(original)
        decoded = decoder.register_to_int16(reg)

        if decoded != original:
            raise AssertionError(
                f"int16 encoding failed: {original} -> {decoded}, " f"register: {reg}"
            )

    # Test uint16
    test_uints = [0, 1, 1000, 32767, 65535]

    for original in test_uints:
        reg = encoder.uint16_to_register(original)
        decoded = decoder.register_to_uint16(reg)

        if decoded != original:
            raise AssertionError(
                f"uint16 encoding failed: {original} -> {decoded}, " f"register: {reg}"
            )

    # Test bool
    for original in [True, False]:
        coil = encoder.bool_to_coil(original)
        decoded = decoder.coil_to_bool(coil)

        if decoded != original:
            raise AssertionError(
                f"bool encoding failed: {original} -> {decoded}, " f"coil: {coil}"
            )

    # Test array encoding
    float_array = [1.5, 2.5, 3.5]
    registers = encoder.array_to_registers(float_array, "float32")
    decoded_array = decoder.registers_to_array(registers, "float32")

    for orig, dec in zip(float_array, decoded_array):
        if abs(dec - orig) > 1e-6:
            raise AssertionError(
                f"Array encoding failed: {float_array} -> {decoded_array}"
            )

    print("✓ All encoding/decoding validations passed")


if __name__ == "__main__":
    """Test encoding/decoding."""
    validate_encoding()

    # Example usage
    print("\nExample Encoding:")
    print("-" * 60)

    encoder = ModbusEncoder()
    decoder = ModbusDecoder()

    # Encode pH value
    pH_value = 7.25
    high, low = encoder.float32_to_registers(pH_value)
    print(f"pH {pH_value} encodes to registers: [{high}, {low}]")

    # Decode back
    decoded = decoder.registers_to_float32(high, low)
    print(f"Decoding [{high}, {low}] gives: {decoded}")

    # Encode array
    sensor_values = [7.25, 2.1, 25.5]  # pH, Cl, temp
    registers = encoder.array_to_registers(sensor_values)
    print(f"\nSensor array {sensor_values}")
    print(f"Encodes to {len(registers)} registers: {registers}")

    # Decode back
    decoded_array = decoder.registers_to_array(registers)
    print(f"Decodes to: {decoded_array}")
