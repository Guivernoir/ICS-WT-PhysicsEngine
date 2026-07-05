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

Maintenance register block  (added February 2026)
-------------------------------------------------
Holding registers (write to command a maintenance action):
  HR 200  maintenance_target_id    uint16   MaintenanceTarget enum value
  HR 201  maintenance_action_code  uint16   MaintenanceAction enum value
  HR 202  maintenance_param        float32  action parameter (HR 202-203)

Coils (write 1 to trigger; simulator auto-clears after execution):
  Coil 10  maintenance_trigger     bool

Input registers (written by simulator after each execution):
  IR 110  maintenance_status_code  uint16   MaintenanceStatus enum value
  IR 111  maintenance_last_target  uint16   echo of target_id
  IR 112  maintenance_last_action  uint16   echo of action_code

Workflow:
  1. Write target_id   → HR 200
  2. Write action_code → HR 201
  3. Write param       → HR 202-203 (float32, 0.0 if unused)
  4. Write True        → Coil 10 (trigger)
  5. Poll IR 110 until it is not PENDING (5)
  6. Read IR 110 for status, IR 111/112 for echo

Author: Guilherme F. G. Santos
Last updated: February 2026
License: MIT
"""

from typing import List, Optional

from .register_definitions import RegisterDefinitionMixin
from .register_types import RegisterDefinition, RegisterType


class ModbusRegisterMap(RegisterDefinitionMixin):
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_all(self):
        all_registers = (
            self.input_registers
            + self.holding_registers
            + self.coils
            + self.discrete_inputs
        )
        for reg in all_registers:
            reg.validate()

        self._check_address_conflicts(self.input_registers, "Input registers")
        self._check_address_conflicts(self.holding_registers, "Holding registers")
        self._check_address_conflicts(self.coils, "Coils")
        self._check_address_conflicts(self.discrete_inputs, "Discrete inputs")

    def _check_address_conflicts(
        self, registers: List[RegisterDefinition], type_name: str
    ):
        address_ranges = []
        for reg in registers:
            start = reg.address
            end = reg.address + reg.size_words - 1
            address_ranges.append((start, end, reg.name))

        address_ranges.sort(key=lambda x: x[0])

        for i in range(len(address_ranges) - 1):
            curr_start, curr_end, curr_name = address_ranges[i]
            next_start, next_end, next_name = address_ranges[i + 1]

            if curr_end >= next_start:
                raise ValueError(
                    f"{type_name} address conflict: {curr_name} "
                    f"[{curr_start}-{curr_end}] overlaps with {next_name} "
                    f"[{next_start}-{next_end}]"
                )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_register_by_name(self, name: str) -> Optional[RegisterDefinition]:
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

        print("\nINPUT REGISTERS (Read-Only)")
        print("-" * 80)
        print(
            f"{'Address':<12} {'Name':<30} {'Type':<10} {'Units':<10} {'Description'}"
        )
        print("-" * 80)
        for reg in self.input_registers:
            base = 30001 + reg.address
            addr = f"{base}-{base+1}" if reg.data_type == "float32" else str(base)
            print(
                f"{addr:<12} {reg.name:<30} {reg.data_type:<10} {reg.units:<10} {reg.description}"
            )

        print("\nHOLDING REGISTERS (Read/Write)")
        print("-" * 80)
        print(
            f"{'Address':<12} {'Name':<30} {'Type':<10} {'Units':<10} {'Description'}"
        )
        print("-" * 80)
        for reg in self.holding_registers:
            base = 40001 + reg.address
            addr = f"{base}-{base+1}" if reg.data_type == "float32" else str(base)
            print(
                f"{addr:<12} {reg.name:<30} {reg.data_type:<10} {reg.units:<10} {reg.description}"
            )

        print("\nCOILS (Read/Write)")
        print("-" * 80)
        print(f"{'Address':<12} {'Name':<30} {'Description'}")
        print("-" * 80)
        for reg in self.coils:
            print(f"{1+reg.address:<12} {reg.name:<30} {reg.description}")

        print("\nDISCRETE INPUTS (Read-Only)")
        print("-" * 80)
        print(f"{'Address':<12} {'Name':<30} {'Description'}")
        print("-" * 80)
        for reg in self.discrete_inputs:
            print(f"{10001+reg.address:<12} {reg.name:<30} {reg.description}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    reg_map = ModbusRegisterMap()
    reg_map.print_register_map()
