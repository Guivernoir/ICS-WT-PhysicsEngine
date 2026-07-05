"""Controller-to-CFD coupling contracts and helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .device_coupling import (
    ActuatorCouplingContract,
    apply_scalar_source_region,
    device_coupling_catalog,
)
from .fields import ScalarField
from .mesh import StructuredMesh

ControllerRoutine = str
ControllerActionState = str

VALID_CONTROLLER_ROUTINES = {
    "monitoring",
    "threshold-adjustment",
    "pid-lite-label",
    "manual-supervised",
    "interlock-like",
    "failover-review",
}
VALID_ACTION_STATES = {
    "monitoring-only",
    "within-deadband",
    "increase-source",
    "withhold-source",
}


@dataclass(frozen=True)
class ControllerCouplingContract:
    """A synthetic controller rule that can affect a CFD actuator boundary."""

    controller_id: str
    unit_id: str
    actuator_tag: str
    routine: ControllerRoutine
    controlled_variable: str
    manipulated_variable: str
    setpoint: float
    deadband: float
    proportional_gain: float
    minimum_output: float
    maximum_output: float
    evidence_status: str = "simulated_metadata"

    def validate(self) -> None:
        if not self.controller_id:
            raise ValueError("controller_id is required")
        if not self.unit_id:
            raise ValueError(f"{self.controller_id}: unit_id is required")
        if not self.actuator_tag:
            raise ValueError(f"{self.controller_id}: actuator_tag is required")
        if self.routine not in VALID_CONTROLLER_ROUTINES:
            raise ValueError(f"{self.controller_id}: unsupported routine")
        if not self.controlled_variable:
            raise ValueError(f"{self.controller_id}: controlled_variable is required")
        if not self.manipulated_variable:
            raise ValueError(f"{self.controller_id}: manipulated_variable is required")
        if not math.isfinite(self.setpoint):
            raise ValueError(f"{self.controller_id}: setpoint must be finite")
        if self.deadband < 0.0 or not math.isfinite(self.deadband):
            raise ValueError(f"{self.controller_id}: invalid deadband")
        if self.proportional_gain < 0.0 or not math.isfinite(self.proportional_gain):
            raise ValueError(f"{self.controller_id}: invalid proportional gain")
        if self.minimum_output < 0.0:
            raise ValueError(f"{self.controller_id}: minimum_output cannot be negative")
        if self.minimum_output > self.maximum_output:
            raise ValueError(f"{self.controller_id}: invalid output bounds")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.controller_id}: evidence status must be simulated")


@dataclass(frozen=True)
class ControllerCouplingAction:
    """A bounded synthetic controller action against a CFD actuator."""

    controller_id: str
    actuator_tag: str
    routine: ControllerRoutine
    controlled_variable: str
    process_value: float
    setpoint: float
    output: float
    action_state: ControllerActionState
    evidence_status: str = "simulated_metadata"

    def validate(self) -> None:
        if not self.controller_id:
            raise ValueError("controller_id is required")
        if not self.actuator_tag:
            raise ValueError(f"{self.controller_id}: actuator_tag is required")
        if self.routine not in VALID_CONTROLLER_ROUTINES:
            raise ValueError(f"{self.controller_id}: unsupported routine")
        if not self.controlled_variable:
            raise ValueError(f"{self.controller_id}: controlled_variable is required")
        if not math.isfinite(self.process_value):
            raise ValueError(f"{self.controller_id}: process value must be finite")
        if not math.isfinite(self.setpoint):
            raise ValueError(f"{self.controller_id}: setpoint must be finite")
        if self.output < 0.0 or not math.isfinite(self.output):
            raise ValueError(f"{self.controller_id}: invalid output")
        if self.action_state not in VALID_ACTION_STATES:
            raise ValueError(f"{self.controller_id}: unsupported action state")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.controller_id}: evidence status must be simulated")


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def evaluate_controller_action(
    contract: ControllerCouplingContract, *, process_value: float
) -> ControllerCouplingAction:
    """Evaluate a bounded synthetic controller action from a process value."""

    contract.validate()
    if not math.isfinite(process_value):
        raise ValueError("process value must be finite")
    if contract.routine == "monitoring":
        action = ControllerCouplingAction(
            contract.controller_id,
            contract.actuator_tag,
            contract.routine,
            contract.controlled_variable,
            process_value,
            contract.setpoint,
            0.0,
            "monitoring-only",
        )
        action.validate()
        return action

    error = contract.setpoint - process_value
    if abs(error) <= contract.deadband:
        output = 0.0
        state = "within-deadband"
    elif error > 0.0:
        output = _bounded(
            error * contract.proportional_gain,
            contract.minimum_output,
            contract.maximum_output,
        )
        state = "increase-source"
    else:
        output = 0.0
        state = "withhold-source"

    action = ControllerCouplingAction(
        contract.controller_id,
        contract.actuator_tag,
        contract.routine,
        contract.controlled_variable,
        process_value,
        contract.setpoint,
        output,
        state,
    )
    action.validate()
    return action


def apply_controller_action_to_scalar(
    mesh: StructuredMesh,
    scalar: ScalarField,
    actuator: ActuatorCouplingContract,
    action: ControllerCouplingAction,
    *,
    dt: float,
) -> ScalarField:
    """Apply a controller action through its declared actuator region."""

    actuator.validate()
    action.validate()
    if actuator.actuator_tag != action.actuator_tag:
        raise ValueError("controller action does not target this actuator")
    return apply_scalar_source_region(
        mesh,
        scalar,
        actuator.target_region,
        strength=action.output,
        dt=dt,
    )


def controller_contracts_for_actuators(
    actuators: tuple[ActuatorCouplingContract, ...],
) -> tuple[ControllerCouplingContract, ...]:
    """Create deterministic synthetic controller contracts for CFD actuators."""

    contracts: list[ControllerCouplingContract] = []
    for actuator in actuators:
        actuator.validate()
        contracts.append(
            ControllerCouplingContract(
                controller_id=f"CTL-{actuator.unit_id.upper()}",
                unit_id=actuator.unit_id,
                actuator_tag=actuator.actuator_tag,
                routine="pid-lite-label",
                controlled_variable=actuator.manipulated_variable,
                manipulated_variable=actuator.manipulated_variable,
                setpoint=1.0,
                deadband=0.05,
                proportional_gain=1.0,
                minimum_output=0.0,
                maximum_output=actuator.maximum_value,
            )
        )
    return tuple(contracts)


def controller_coupling_catalog() -> tuple[ControllerCouplingContract, ...]:
    """Return deterministic synthetic controller-to-CFD contracts."""

    actuators = tuple(
        contract
        for contract in device_coupling_catalog()
        if isinstance(contract, ActuatorCouplingContract)
    )
    return controller_contracts_for_actuators(actuators)
