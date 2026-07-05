"""Register definition builders for the HydraSim Modbus map."""

from __future__ import annotations

from .register_types import RegisterDefinition, RegisterType


class RegisterDefinitionMixin:
    input_registers: list[RegisterDefinition]
    holding_registers: list[RegisterDefinition]
    coils: list[RegisterDefinition]
    discrete_inputs: list[RegisterDefinition]

    # ------------------------------------------------------------------
    # Input registers (read-only)
    # ------------------------------------------------------------------

    def _define_input_registers(self) -> None:
        """
        Define input registers (read-only sensor values).

        Address range: 30000-39999 (Modbus convention)
        Base address: 0 (internal addressing)
        """
        # pH sensors
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=0,
                    name="pH_inlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="pH",
                    description="pH at inlet (zone 0)",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=2,
                    name="pH_middle",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="pH",
                    description="pH at middle (zone n/2)",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=4,
                    name="pH_outlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="pH",
                    description="pH at outlet (zone -1)",
                    read_only=True,
                ),
            ]
        )

        # Chlorine sensors
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=6,
                    name="chlorine_inlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="mg/L",
                    description="Free chlorine at inlet",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=8,
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
                    address=10,
                    name="flow_rate",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Main flow rate",
                    read_only=True,
                ),
            ]
        )

        # Temperature sensors
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=12,
                    name="temperature_inlet",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="°C",
                    description="Water temperature at inlet",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=14,
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
                    address=100,
                    name="simulation_time",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="float32",
                    units="s",
                    description="Simulation elapsed time",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=102,
                    name="system_status",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="uint16",
                    units="",
                    description="System status code (0=OK, >0=fault)",
                    read_only=True,
                ),
            ]
        )

        # ----------------------------------------------------------------
        # Maintenance feedback registers (read by external client to check
        # result after setting the trigger coil)
        # ----------------------------------------------------------------
        self.input_registers.extend(
            [
                RegisterDefinition(
                    address=110,
                    name="maintenance_status_code",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="uint16",
                    units="",
                    description=(
                        "Last maintenance result code "
                        "(0=SUCCESS 1=INVALID_TARGET 2=INVALID_ACTION "
                        "3=NOT_SUPPORTED 4=EXEC_ERROR 5=PENDING)"
                    ),
                    read_only=True,
                ),
                RegisterDefinition(
                    address=111,
                    name="maintenance_last_target",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="uint16",
                    units="",
                    description="Target ID echoed after last maintenance op",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=112,
                    name="maintenance_last_action",
                    register_type=RegisterType.INPUT_REGISTER,
                    data_type="uint16",
                    units="",
                    description="Action code echoed after last maintenance op",
                    read_only=True,
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Holding registers (read/write)
    # ------------------------------------------------------------------

    def _define_holding_registers(self) -> None:
        """
        Define holding registers (read/write actuator setpoints).

        Address range: 40000-49999 (Modbus convention)
        Base address: 0 (internal addressing)
        """
        # Process actuator setpoints
        self.holding_registers.extend(
            [
                RegisterDefinition(
                    address=0,
                    name="acid_flow_rate",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Acid dosing pump flow rate setpoint",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=2,
                    name="chlorine_flow_rate",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Chlorine dosing pump flow rate setpoint",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=4,
                    name="inlet_flow_rate",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="L/min",
                    description="Main inlet flow rate setpoint",
                    read_only=False,
                ),
            ]
        )

        # Dosing concentrations
        self.holding_registers.extend(
            [
                RegisterDefinition(
                    address=10,
                    name="acid_concentration",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="mol/L",
                    description="Acid stock solution concentration",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=12,
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
                    address=100,
                    name="simulation_timestep",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="s",
                    description="Simulation time step",
                    read_only=False,
                ),
            ]
        )

        # ----------------------------------------------------------------
        # Maintenance command registers
        # Write target, action, param → then pulse Coil 10 to execute.
        # ----------------------------------------------------------------
        self.holding_registers.extend(
            [
                RegisterDefinition(
                    address=200,
                    name="maintenance_target_id",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="uint16",
                    units="",
                    description=(
                        "Target device ID for maintenance action "
                        "(0=pH_inlet … 10=inlet_valve; see MaintenanceTarget enum)"
                    ),
                    read_only=False,
                ),
                RegisterDefinition(
                    address=201,
                    name="maintenance_action_code",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="uint16",
                    units="",
                    description=(
                        "Action code (0=CALIBRATE … 11=REPLACE_TUBE; "
                        "see MaintenanceAction enum)"
                    ),
                    read_only=False,
                ),
                RegisterDefinition(
                    address=202,
                    name="maintenance_param",
                    register_type=RegisterType.HOLDING_REGISTER,
                    data_type="float32",
                    units="",
                    description=(
                        "Action parameter (float32, HR 202-203). "
                        "For CALIBRATE: reference value (add 1000 to skip warmup). "
                        "Unused actions: write 0.0"
                    ),
                    read_only=False,
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Coils (read/write discrete)
    # ------------------------------------------------------------------

    def _define_coils(self) -> None:
        self.coils.extend(
            [
                RegisterDefinition(
                    address=0,
                    name="acid_pump_enable",
                    register_type=RegisterType.COIL,
                    data_type="bool",
                    units="",
                    description="Enable acid dosing pump (True=ON, False=OFF)",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=1,
                    name="chlorine_pump_enable",
                    register_type=RegisterType.COIL,
                    data_type="bool",
                    units="",
                    description="Enable chlorine dosing pump (True=ON, False=OFF)",
                    read_only=False,
                ),
                RegisterDefinition(
                    address=2,
                    name="simulation_running",
                    register_type=RegisterType.COIL,
                    data_type="bool",
                    units="",
                    description="Simulation running (True=running, False=paused)",
                    read_only=False,
                ),
                # ----------------------------------------------------------------
                # Maintenance trigger coil — write True to fire the action whose
                # parameters are in HR 200-203.  The simulator auto-clears this
                # coil once the action has completed (success or error).
                # ----------------------------------------------------------------
                RegisterDefinition(
                    address=10,
                    name="maintenance_trigger",
                    register_type=RegisterType.COIL,
                    data_type="bool",
                    units="",
                    description=(
                        "Write True to execute maintenance action defined in "
                        "HR 200-203. Simulator auto-clears to False after execution."
                    ),
                    read_only=False,
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Discrete inputs (read-only)
    # ------------------------------------------------------------------

    def _define_discrete_inputs(self) -> None:
        self.discrete_inputs.extend(
            [
                RegisterDefinition(
                    address=0,
                    name="sensor_fault_pH_inlet",
                    register_type=RegisterType.DISCRETE_INPUT,
                    data_type="bool",
                    units="",
                    description="pH inlet sensor fault status",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=1,
                    name="sensor_fault_pH_outlet",
                    register_type=RegisterType.DISCRETE_INPUT,
                    data_type="bool",
                    units="",
                    description="pH outlet sensor fault status",
                    read_only=True,
                ),
                RegisterDefinition(
                    address=2,
                    name="sensor_fault_chlorine",
                    register_type=RegisterType.DISCRETE_INPUT,
                    data_type="bool",
                    units="",
                    description="Chlorine sensor fault status",
                    read_only=True,
                ),
            ]
        )
