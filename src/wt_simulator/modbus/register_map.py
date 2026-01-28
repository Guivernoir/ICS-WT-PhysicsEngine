"""
Modbus Register Map
===================

Defines the mapping between Modbus registers and data sources.

This module contains ONLY the register layout - it does not:
- Read sensors
- Control actuators
- Implement physics
- Enforce limits

Register Types:
- Input Registers (FC 04): Read-only sensor values
- Holding Registers (FC 03/06/16): Read/write actuator setpoints

Register Encoding:
- All floats use IEEE 754 single-precision (32-bit)
- Each float occupies 2 consecutive 16-bit registers
- Byte order: Big-endian (network byte order)

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import IntEnum


class RegisterType(IntEnum):
    """Modbus register types."""

    COIL = 0  # Discrete output (read/write)
    DISCRETE_INPUT = 1  # Discrete input (read-only)
    INPUT_REGISTER = 3  # Analog input (read-only)
    HOLDING_REGISTER = 4  # Analog output (read/write)


@dataclass
class RegisterDefinition:
    """
    Definition of a single Modbus register (or register pair for floats).

    Attributes:
        address: Starting register address (0-based)
        name: Human-readable identifier
        register_type: Coil, discrete input, input register, or holding register
        data_type: 'float32', 'int16', 'uint16', 'bool'
        units: Physical units (e.g., 'pH', 'mg/L', 'L/min')
        description: What this register represents
        read_only: Whether this register can be written
    """

    address: int
    name: str
    register_type: RegisterType
    data_type: str
    units: str
    description: str
    read_only: bool = True

    def validate(self):
        """Validate register definition."""
        if self.address < 0 or self.address > 65535:
            raise ValueError(f"Register address {self.address} out of range [0, 65535]")

        if self.data_type not in ["float32", "int16", "uint16", "bool"]:
            raise ValueError(f"Unknown data type: {self.data_type}")

        if self.register_type == RegisterType.HOLDING_REGISTER and self.read_only:
            raise ValueError(f"Holding register {self.name} marked as read-only")

        if self.register_type == RegisterType.INPUT_REGISTER and not self.read_only:
            raise ValueError(f"Input register {self.name} marked as writable")

    @property
    def size_words(self) -> int:
        """Number of 16-bit words this register occupies."""
        if self.data_type == "float32":
            return 2
        elif self.data_type in ["int16", "uint16"]:
            return 1
        elif self.data_type == "bool":
            return 1
        else:
            raise ValueError(f"Unknown data type: {self.data_type}")


class ModbusRegisterMap:
    """
    Complete Modbus register map for water treatment system.

    This class defines the register layout but does NOT:
    - Read sensor values (that's done by the caller)
    - Write actuator commands (that's done by the caller)
    - Implement control logic
    - Enforce limits

    It only defines WHERE data goes in the Modbus address space.
    """

    def __init__(self):
        """Initialize register map with standard layout."""
        self.input_registers: List[RegisterDefinition] = []
        self.holding_registers: List[RegisterDefinition] = []
        self.coils: List[RegisterDefinition] = []
        self.discrete_inputs: List[RegisterDefinition] = []

        self._define_input_registers()
        self._define_holding_registers()
        self._define_coils()
        self._define_discrete_inputs()

        # Validate all definitions
        self._validate_all()

    def _define_input_registers(self):
        """
        Define input registers (read-only sensor values).

        Address range: 30000-39999 (Modbus convention)
        Base address: 0 (internal addressing)
        """
        # pH sensors (3 locations for spatial measurement)
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=0,  # 30001-30002 in Modbus addressing
                    name="pH_inlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="pH",
                    description="pH at inlet (zone 0)",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=2,  # 30003-30004
                    name="pH_middle",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="pH",
                    description="pH at middle (zone n/2)",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=4,  # 30005-30006
                    name="pH_outlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="pH",
                    description="pH at outlet (zone -1)",
                    read_only=True,
                ),
            ]
        )

        # Chlorine sensors (2 locations)
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=6,  # 30007-30008
                    name="chlorine_inlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="mg/L",
                    description="Free chlorine at inlet",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=8,  # 30009-30010
                    name="chlorine_outlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="mg/L",
                    description="Free chlorine at outlet",
                    read_only=True,
                ),
            ]
        )

        # Flow sensor
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=10,  # 30011-30012
                    name="flow_rate",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Main flow rate",
                    read_only=True,
                ),
            ]
        )

        # Temperature sensors (2 locations)
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=12,  # 30013-30014
                    name="temperature_inlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="°C",
                    description="Water temperature at inlet",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=14,  # 30015-30016
                    name="temperature_outlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="°C",
                    description="Water temperature at outlet",
                    read_only=True,
                ),
            ]
        )

        # System status
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=100,  # 30101-30102
                    name="simulation_time",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="s",
                    description="Simulation elapsed time",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=102,  # 30103
                    name="system_status",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="uint16",
                    units="",
                    description="System status code (0=OK, >0=fault)",
                    read_only=True,
                ),
            ]
        )

    def _define_holding_registers(self):
        """
        Define holding registers (read/write actuator setpoints).

        Address range: 40000-49999 (Modbus convention)
        Base address: 0 (internal addressing)
        """
        # Actuator setpoints
        self.holding_registers.extend(
            [
                RegisterDefinition(
                    address=0,  # 40001-40002
                    name="acid_flow_rate",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Acid dosing pump flow rate setpoint",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=2,  # 40003-40004
                    name="chlorine_flow_rate",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Chlorine dosing pump flow rate setpoint",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=4,  # 40005-40006
                    name="inlet_flow_rate",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Main inlet flow rate setpoint",
                    read_only=False,
                ),
            ]
        )

        # Dosing concentrations (normally constant, but writable for config)
        self.holding_registers.extend(
            [
                RegisterDefinition(
                    address=10,  # 40011-40012
                    name="acid_concentration",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="mol/L",
                    description="Acid stock solution concentration",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=12,  # 40013-40014
                    name="chlorine_concentration",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="mg/L",
                    description="Chlorine stock solution concentration",
                    read_only=False,
                ),
            ]
        )

        # Simulation control
        self.holding_registers.extend(
            [
                RegisterDefinition(
                    address=100,  # 40101-40102
                    name="simulation_timestep",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="s",
                    description="Simulation time step",
                    read_only=False,
                ),
            ]
        )

    def _define_coils(self):
        """
        Define coils (read/write discrete outputs).

        Address range: 00000-09999 (Modbus convention)
        Base address: 0 (internal addressing)
        """
        self.coils.extend(
            [
                RegisterDefinition(
                    address=0,  # 00001
                    name="acid_pump_enable",
                    register_type=RegisterType.COIL,
                    data_type="bool",
                    units="",
                    description="Enable acid dosing pump (True=ON, False=OFF)",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=1,  # 00002
                    name="chlorine_pump_enable",
                    register_type=RegisterType.COIL,
                    data_type="bool",
                    units="",
                    description="Enable chlorine dosing pump (True=ON, False=OFF)",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=2,  # 00003
                    name="simulation_running",
                    register_type=RegisterType.COIL,
                    data_type="bool",
                    units="",
                    description="Simulation running (True=running, False=paused)",
                    read_only=False,
                ),
            ]
        )

    def _define_discrete_inputs(self):
        """
        Define discrete inputs (read-only discrete values).

        Address range: 10000-19999 (Modbus convention)
        Base address: 0 (internal addressing)
        """
        self.discrete_inputs.extend(
            [
                RegisterDefinition(
                    address=0,  # 10001
                    name="sensor_fault_pH_inlet",
                    register_type=RegisterType.DISCRETE_INPUT,
                    data_type="bool",
                    units="",
                    description="pH inlet sensor fault status",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=1,  # 10002
                    name="sensor_fault_pH_outlet",
                    register_type=RegisterType.DISCRETE_INPUT,
                    data_type="bool",
                    units="",
                    description="pH outlet sensor fault status",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=2,  # 10003
                    name="sensor_fault_chlorine",
                    register_type=RegisterType.DISCRETE_INPUT,
                    data_type="bool",
                    units="",
                    description="Chlorine sensor fault status",
                    read_only=True,
                ),
            ]
        )

    def _validate_all(self):
        """Validate all register definitions and check for conflicts."""
        # Validate each register
        all_registers = (
            self.input_registers
            + self.holding_registers
            + self.coils
            + self.discrete_inputs
        )

        for reg in all_registers:
            reg.validate()

        # Check for address conflicts within each type
        self._check_address_conflicts(self.input_registers, "Input registers")
        self._check_address_conflicts(self.holding_registers, "Holding registers")
        self._check_address_conflicts(self.coils, "Coils")
        self._check_address_conflicts(self.discrete_inputs, "Discrete inputs")

    def _check_address_conflicts(
        self, registers: List[RegisterDefinition], type_name: str
    ):
        """Check for overlapping register addresses."""
        address_ranges = []

        for reg in registers:
            start = reg.address
            end = reg.address + reg.size_words - 1
            address_ranges.append((start, end, reg.name))

        # Sort by start address
        address_ranges.sort(key=lambda x: x[0])

        # Check for overlaps
        for i in range(len(address_ranges) - 1):
            curr_start, curr_end, curr_name = address_ranges[i]
            next_start, next_end, next_name = address_ranges[i + 1]

            if curr_end >= next_start:
                raise ValueError(
                    f"{type_name} address conflict: {curr_name} "
                    f"[{curr_start}-{curr_end}] overlaps with {next_name} "
                    f"[{next_start}-{next_end}]"
                )

    def get_register_by_name(self, name: str) -> Optional[RegisterDefinition]:
        """
        Find register definition by name.

        Args:
            name: Register name

        Returns:
            RegisterDefinition if found, None otherwise
        """
        all_registers = (
            self.input_registers
            + self.holding_registers
            + self.coils
            + self.discrete_inputs
        )

        for reg in all_registers:
            if reg.name == name:
                return reg

        return None

    def get_register_by_address(
        self, address: int, register_type: RegisterType
    ) -> Optional[RegisterDefinition]:
        """
        Find register definition by address and type.

        Args:
            address: Register address
            register_type: Type of register

        Returns:
            RegisterDefinition if found, None otherwise
        """
        if register_type == RegisterType.INPUT_REGISTER:
            registers = self.input_registers
        elif register_type == RegisterType.HOLDING_REGISTER:
            registers = self.holding_registers
        elif register_type == RegisterType.COIL:
            registers = self.coils
        elif register_type == RegisterType.DISCRETE_INPUT:
            registers = self.discrete_inputs
        else:
            return None

        for reg in registers:
            if reg.address <= address < reg.address + reg.size_words:
                return reg

        return None

    def print_register_map(self):
        """Print complete register map for documentation."""
        print("=" * 80)
        print("MODBUS REGISTER MAP")
        print("=" * 80)

        print("\nINPUT REGISTERS (Read-Only Sensor Values)")
        print("-" * 80)
        print(
            f"{'Address':<10} {'Name':<25} {'Type':<10} {'Units':<10} {'Description':<30}"
        )
        print("-" * 80)
        for reg in self.input_registers:
            modbus_addr = 30001 + reg.address
            if reg.data_type == "float32":
                addr_str = f"{modbus_addr}-{modbus_addr+1}"
            else:
                addr_str = str(modbus_addr)
            print(
                f"{addr_str:<10} {reg.name:<25} {reg.data_type:<10} {reg.units:<10} {reg.description:<30}"
            )

        print("\nHOLDING REGISTERS (Read/Write Actuator Setpoints)")
        print("-" * 80)
        print(
            f"{'Address':<10} {'Name':<25} {'Type':<10} {'Units':<10} {'Description':<30}"
        )
        print("-" * 80)
        for reg in self.holding_registers:
            modbus_addr = 40001 + reg.address
            if reg.data_type == "float32":
                addr_str = f"{modbus_addr}-{modbus_addr+1}"
            else:
                addr_str = str(modbus_addr)
            print(
                f"{addr_str:<10} {reg.name:<25} {reg.data_type:<10} {reg.units:<10} {reg.description:<30}"
            )

        print("\nCOILS (Read/Write Discrete Outputs)")
        print("-" * 80)
        print(f"{'Address':<10} {'Name':<25} {'Description':<50}")
        print("-" * 80)
        for reg in self.coils:
            modbus_addr = 1 + reg.address
            print(f"{modbus_addr:<10} {reg.name:<25} {reg.description:<50}")

        print("\nDISCRETE INPUTS (Read-Only Status Bits)")
        print("-" * 80)
        print(f"{'Address':<10} {'Name':<25} {'Description':<50}")
        print("-" * 80)
        for reg in self.discrete_inputs:
            modbus_addr = 10001 + reg.address
            print(f"{modbus_addr:<10} {reg.name:<25} {reg.description:<50}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    """Test register map."""
    reg_map = ModbusRegisterMap()
    reg_map.print_register_map()

    # Test lookups
    print("\nTest Lookups:")
    print("-" * 80)

    reg = reg_map.get_register_by_name("pH_inlet")
    if reg:
        print(
            f"Found register: {reg.name} at address {reg.address} ({reg.description})"
        )

    reg = reg_map.get_register_by_address(0, RegisterType.INPUT_REGISTER)
    if reg:
        print(f"Register at IR:0 is {reg.name}")

    print("\n✓ Register map validation passed")
