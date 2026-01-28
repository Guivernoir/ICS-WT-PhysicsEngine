"""
Modbus Interface Package
=========================

Modbus/TCP protocol adapter for exposing simulation data.

This package provides a pure protocol layer:
- Modbus TCP server
- Register mapping
- Data encoding/decoding

It does NOT:
- Implement control logic
- Validate actuator commands
- Enforce safety limits
- Know about physics or sensors

The Modbus layer is a communication protocol only.

Components:
- slave.py: Modbus TCP server
- register_map.py: Address space definition
- protocols.py: Data encoding/decoding

Usage Example:
>>> from modbus import ModbusSlave, ModbusRegisterMap
>>>
>>> # Create register map
>>> reg_map = ModbusRegisterMap()
>>>
>>> # Create Modbus server
>>> slave = ModbusSlave(reg_map)
>>>
>>> # Start server (non-blocking)
>>> slave.start(blocking=False)
>>>
>>> # Update sensor values (typically done by simulation loop)
>>> slave.update_input_register("pH_inlet", 7.25)
>>> slave.update_input_register("chlorine_inlet", 2.1)
>>>
>>> # Read actuator commands (typically done by simulation loop)
>>> acid_rate = slave.read_holding_register("acid_flow_rate")
>>> chlorine_rate = slave.read_holding_register("chlorine_flow_rate")

Architecture:

┌─────────────────┐
│   PLC / SCADA   │  External control system
└────────┬────────┘
         │ Modbus/TCP
┌────────▼────────┐
│  ModbusSlave    │  Protocol adapter (this package)
└────────┬────────┘
         │
┌────────▼────────┐
│   Simulation    │  Physics + Sensors
│     Loop        │
└─────────────────┘

Dependencies:
- pymodbus: Python Modbus library
  Install: pip install pymodbus

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Guilherme F. G. Santos"

from .register_map import ModbusRegisterMap, RegisterDefinition, RegisterType

from .protocols import ModbusEncoder, ModbusDecoder

from .slave import ModbusSlave, ModbusServerConfig

__all__ = [
    # Register mapping
    "ModbusRegisterMap",
    "RegisterDefinition",
    "RegisterType",
    # Encoding/decoding
    "ModbusEncoder",
    "ModbusDecoder",
    # Server
    "ModbusSlave",
    "ModbusServerConfig",
]


def print_package_info():
    """Print package information."""
    print("=" * 70)
    print("MODBUS INTERFACE PACKAGE")
    print("=" * 70)
    print()
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    print("Purpose:")
    print("  Pure Modbus/TCP protocol adapter")
    print("  Exposes simulation data to external systems")
    print()
    print("What it DOES:")
    print("  ✓ Serve Modbus/TCP on port 502")
    print("  ✓ Map sensor values to input registers")
    print("  ✓ Map actuator commands to holding registers")
    print("  ✓ Encode/decode data (IEEE 754 floats)")
    print()
    print("What it DOES NOT do:")
    print("  ✗ Implement control logic")
    print("  ✗ Validate actuator commands")
    print("  ✗ Enforce safety limits")
    print("  ✗ Implement physics")
    print("  ✗ Read sensors directly")
    print()
    print("Register Map:")
    print("  Input Registers (30001+):  Sensor values (read-only)")
    print("  Holding Registers (40001+): Actuator setpoints (read/write)")
    print("  Coils (00001+):            Binary outputs (read/write)")
    print("  Discrete Inputs (10001+):  Fault status (read-only)")
    print()
    print("Dependencies:")
    print("  - pymodbus (install: pip install pymodbus)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print_package_info()

    # Show register map
    print("\n")
    reg_map = ModbusRegisterMap()
    reg_map.print_register_map()
