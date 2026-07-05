"""Reusable CFD boundary-condition contracts for water-treatment units."""

from __future__ import annotations

from dataclasses import dataclass

BoundaryConditionKind = str
BoundaryVariable = str

VALID_BOUNDARY_CONDITION_KINDS = {
    "inlet_flow",
    "outlet_flow",
    "pump_discharge",
    "valve_loss",
    "dosing_injection",
    "mechanical_mixer",
    "baffle_wall",
    "porous_filter_media",
    "backwash_flow",
    "drain_flow",
    "recirculation_flow",
    "free_surface_level",
}
VALID_EFFECTS = {
    "velocity",
    "pressure",
    "scalar_source",
    "scalar_sink",
    "momentum_loss",
    "mixing",
    "porosity",
    "level",
}


@dataclass(frozen=True)
class BoundaryConditionContract:
    """A named boundary/source model that unit processes may reference."""

    condition_id: str
    kind: BoundaryConditionKind
    purpose: str
    required_variables: tuple[BoundaryVariable, ...]
    effects: tuple[str, ...]
    applies_to_geometry: tuple[str, ...]
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id is required")
        if self.kind not in VALID_BOUNDARY_CONDITION_KINDS:
            raise ValueError(f"{self.condition_id}: unsupported kind {self.kind}")
        if not self.purpose:
            raise ValueError(f"{self.condition_id}: purpose is required")
        if not self.required_variables:
            raise ValueError(f"{self.condition_id}: variables are required")
        if not self.effects:
            raise ValueError(f"{self.condition_id}: effects are required")
        unknown_effects = set(self.effects) - VALID_EFFECTS
        if unknown_effects:
            raise ValueError(f"{self.condition_id}: unknown effects {unknown_effects}")
        if not self.applies_to_geometry:
            raise ValueError(f"{self.condition_id}: geometry targets are required")
        if not self.limitations:
            raise ValueError(f"{self.condition_id}: limitations are required")


def _limitations(*extra: str) -> tuple[str, ...]:
    return (
        "synthetic boundary-condition contract",
        "requires numerical verification before plant-realism claims",
        "not a commissioning or design-authority boundary model",
        *extra,
    )


def boundary_condition_catalog() -> tuple[BoundaryConditionContract, ...]:
    """Return deterministic reusable boundary/source contracts."""

    return (
        BoundaryConditionContract(
            condition_id="bc-inlet-flow",
            kind="inlet_flow",
            purpose="Inject known upstream flow, temperature, and quality values.",
            required_variables=("flow_rate", "temperature", "scalar_profile"),
            effects=("velocity", "scalar_source"),
            applies_to_geometry=("open_channel", "rapid_mix_basin", "contact_basin"),
            limitations=_limitations("upstream turbulence profile is synthetic"),
        ),
        BoundaryConditionContract(
            condition_id="bc-outlet-flow",
            kind="outlet_flow",
            purpose="Represent downstream outflow without modeling the full network.",
            required_variables=("flow_rate", "hydraulic_head"),
            effects=("velocity", "pressure"),
            applies_to_geometry=("open_channel", "settling_basin", "storage_tank"),
            limitations=_limitations("downstream hydraulic network is out of scope"),
        ),
        BoundaryConditionContract(
            condition_id="bc-pump-discharge",
            kind="pump_discharge",
            purpose="Map pump setpoint/status to discharge flow and head context.",
            required_variables=("pump_status", "pump_speed", "hydraulic_head"),
            effects=("velocity", "pressure"),
            applies_to_geometry=("pump_header",),
            limitations=_limitations("pump curves are synthetic until calibrated"),
        ),
        BoundaryConditionContract(
            condition_id="bc-valve-loss",
            kind="valve_loss",
            purpose="Map valve position to a bounded hydraulic loss term.",
            required_variables=("valve_position", "loss_coefficient"),
            effects=("momentum_loss", "pressure"),
            applies_to_geometry=("pump_header", "open_channel"),
            limitations=_limitations("cavitation and detailed valve curves are absent"),
        ),
        BoundaryConditionContract(
            condition_id="bc-dosing-injection",
            kind="dosing_injection",
            purpose="Add chemical scalar source terms at a declared injection region.",
            required_variables=("chemical_flow", "concentration", "injection_region"),
            effects=("scalar_source", "mixing"),
            applies_to_geometry=(
                "rapid_mix_basin",
                "contact_basin",
                "chemical_feed_skid",
            ),
            limitations=_limitations(
                "chemical reaction detail is bounded by scalar model"
            ),
        ),
        BoundaryConditionContract(
            condition_id="bc-mechanical-mixer",
            kind="mechanical_mixer",
            purpose="Add bounded mixing energy to a local region.",
            required_variables=("mixer_status", "mixing_intensity", "region"),
            effects=("mixing",),
            applies_to_geometry=("rapid_mix_basin", "flocculation_placeholder"),
            limitations=_limitations("impeller geometry is not resolved"),
        ),
        BoundaryConditionContract(
            condition_id="bc-baffle-wall",
            kind="baffle_wall",
            purpose="Represent fixed baffle obstacles and directed flow pathing.",
            required_variables=("obstacle_cells", "porosity"),
            effects=("momentum_loss", "porosity"),
            applies_to_geometry=("settling_basin", "contact_basin"),
            limitations=_limitations("small-scale eddies near baffles are unresolved"),
        ),
        BoundaryConditionContract(
            condition_id="bc-porous-filter-media",
            kind="porous_filter_media",
            purpose="Represent filter-bed resistance and turbidity removal context.",
            required_variables=("media_resistance", "headloss_proxy", "turbidity"),
            effects=("momentum_loss", "scalar_sink", "porosity"),
            applies_to_geometry=("filter_bed",),
            limitations=_limitations(
                "media fouling and breakthrough need later validation"
            ),
        ),
        BoundaryConditionContract(
            condition_id="bc-backwash-flow",
            kind="backwash_flow",
            purpose="Reverse/flush flow through a filter during backwash scenarios.",
            required_variables=("backwash_status", "flow_rate", "duration"),
            effects=("velocity", "scalar_sink"),
            applies_to_geometry=("filter_bed", "waste_handling"),
            limitations=_limitations(
                "air scour and detailed solids transport are absent"
            ),
        ),
        BoundaryConditionContract(
            condition_id="bc-drain-flow",
            kind="drain_flow",
            purpose="Remove water or waste stream through a drain boundary.",
            required_variables=("drain_status", "flow_rate"),
            effects=("velocity", "scalar_sink"),
            applies_to_geometry=("settling_basin", "filter_bed", "waste_handling"),
            limitations=_limitations("waste treatment downstream is metadata only"),
        ),
        BoundaryConditionContract(
            condition_id="bc-recirculation-flow",
            kind="recirculation_flow",
            purpose="Route bounded recirculation between declared local regions.",
            required_variables=("source_region", "target_region", "flow_rate"),
            effects=("velocity", "scalar_source"),
            applies_to_geometry=("rapid_mix_basin", "contact_basin", "storage_tank"),
            limitations=_limitations("recirculation piping is not spatially resolved"),
        ),
        BoundaryConditionContract(
            condition_id="bc-free-surface-level",
            kind="free_surface_level",
            purpose="Track tank/channel level metadata for bounded grid presets.",
            required_variables=("water_level", "surface_area"),
            effects=("level", "pressure"),
            applies_to_geometry=("open_channel", "storage_tank", "settling_basin"),
            limitations=_limitations("free-surface deformation is not fully resolved"),
        ),
    )


def boundary_condition_ids() -> tuple[str, ...]:
    return tuple(contract.condition_id for contract in boundary_condition_catalog())


def get_boundary_condition(condition_id: str) -> BoundaryConditionContract:
    for contract in boundary_condition_catalog():
        if contract.condition_id == condition_id:
            return contract
    raise ValueError(f"unknown boundary condition: {condition_id}")


def conditions_for_geometry(
    geometry_class: str,
) -> tuple[BoundaryConditionContract, ...]:
    return tuple(
        contract
        for contract in boundary_condition_catalog()
        if geometry_class in contract.applies_to_geometry
    )
