from dataclasses import dataclass
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
