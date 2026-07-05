"""Water-treatment unit-process contracts for CFD-backed simulation."""

from __future__ import annotations

from dataclasses import dataclass

from .area_models import AREA_IDS

BoundaryDirection = str
BoundaryKind = str
GeometryClass = str

VALID_BOUNDARY_DIRECTIONS = {"input", "output", "internal", "waste"}
VALID_BOUNDARY_KINDS = {
    "flow",
    "chemical_source",
    "recirculation",
    "drain",
    "level_surface",
    "media_interface",
}
VALID_GEOMETRY_CLASSES = {
    "open_channel",
    "rapid_mix_basin",
    "flocculation_placeholder",
    "settling_basin",
    "filter_bed",
    "contact_basin",
    "storage_tank",
    "pump_header",
    "chemical_feed_skid",
    "waste_handling",
}


@dataclass(frozen=True)
class ProcessBoundary:
    """A named unit-process boundary that later CFD boundary models can bind."""

    name: str
    direction: BoundaryDirection
    kind: BoundaryKind
    variables: tuple[str, ...]

    def validate(self) -> None:
        if not self.name:
            raise ValueError("boundary name is required")
        if self.direction not in VALID_BOUNDARY_DIRECTIONS:
            raise ValueError(f"unsupported boundary direction: {self.direction}")
        if self.kind not in VALID_BOUNDARY_KINDS:
            raise ValueError(f"unsupported boundary kind: {self.kind}")
        if not self.variables:
            raise ValueError(f"{self.name}: variables are required")


@dataclass(frozen=True)
class InstrumentationPoint:
    """A simulated measurement/control point tied to a process region."""

    tag: str
    kind: str
    measures: tuple[str, ...]
    sample_region: str
    evidence_status: str = "simulated_metadata"

    def validate(self) -> None:
        if not self.tag:
            raise ValueError("instrumentation tag is required")
        if not self.kind:
            raise ValueError(f"{self.tag}: kind is required")
        if not self.measures:
            raise ValueError(f"{self.tag}: measures are required")
        if not self.sample_region:
            raise ValueError(f"{self.tag}: sample_region is required")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.tag}: instrumentation must remain simulated")


@dataclass(frozen=True)
class UnitProcess:
    """A CFD-ready water-treatment unit-process contract."""

    unit_id: str
    area_id: str
    display_name: str
    purpose: str
    geometry_class: GeometryClass
    boundaries: tuple[ProcessBoundary, ...]
    process_variables: tuple[str, ...]
    instrumentation: tuple[InstrumentationPoint, ...]
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.unit_id:
            raise ValueError("unit_id is required")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.unit_id}: unsupported area {self.area_id}")
        if not self.display_name:
            raise ValueError(f"{self.unit_id}: display_name is required")
        if not self.purpose:
            raise ValueError(f"{self.unit_id}: purpose is required")
        if self.geometry_class not in VALID_GEOMETRY_CLASSES:
            raise ValueError(
                f"{self.unit_id}: unsupported geometry {self.geometry_class}"
            )
        if not self.boundaries:
            raise ValueError(f"{self.unit_id}: boundaries are required")
        if not self.process_variables:
            raise ValueError(f"{self.unit_id}: process_variables are required")
        if not self.instrumentation:
            raise ValueError(f"{self.unit_id}: instrumentation is required")
        if not self.limitations:
            raise ValueError(f"{self.unit_id}: limitations are required")
        directions = {boundary.direction for boundary in self.boundaries}
        if "input" not in directions or "output" not in directions:
            raise ValueError(f"{self.unit_id}: input and output boundaries required")
        for boundary in self.boundaries:
            boundary.validate()
        for point in self.instrumentation:
            point.validate()


def _flow_boundary(name: str, direction: BoundaryDirection) -> ProcessBoundary:
    return ProcessBoundary(
        name=name,
        direction=direction,
        kind="flow",
        variables=("flow_rate", "temperature", "hydraulic_head"),
    )


def _common_limitations(*extra: str) -> tuple[str, ...]:
    return (
        "synthetic unit-process contract",
        "not a certified design model",
        "requires CFD verification before process-realism claims",
        *extra,
    )


def unit_process_catalog() -> tuple[UnitProcess, ...]:
    """Return the deterministic reference water-treatment unit catalog."""

    return (
        UnitProcess(
            unit_id="intake-channel",
            area_id="intake",
            display_name="Raw-water intake channel",
            purpose="Introduce raw-water flow and upstream quality context.",
            geometry_class="open_channel",
            boundaries=(
                _flow_boundary("raw-water-inlet", "input"),
                _flow_boundary("to-rapid-mix", "output"),
            ),
            process_variables=("flow_rate", "temperature", "turbidity"),
            instrumentation=(
                InstrumentationPoint(
                    "FIT-INT-001", "flow_transmitter", ("flow_rate",), "inlet reach"
                ),
                InstrumentationPoint(
                    "AIT-INT-001", "quality_analyzer", ("turbidity",), "outlet reach"
                ),
            ),
            limitations=_common_limitations(
                "raw-water source variability is synthetic"
            ),
        ),
        UnitProcess(
            unit_id="rapid-mix-dosing",
            area_id="dosing",
            display_name="Rapid-mix and dosing basin",
            purpose="Mix incoming water with simulated chemical dosing sources.",
            geometry_class="rapid_mix_basin",
            boundaries=(
                _flow_boundary("from-intake", "input"),
                ProcessBoundary(
                    "chemical-injection",
                    "input",
                    "chemical_source",
                    ("chlorine", "ph_proxy", "demand_precursor"),
                ),
                _flow_boundary("to-clarification", "output"),
            ),
            process_variables=(
                "flow_rate",
                "chlorine",
                "ph_proxy",
                "demand_precursor",
            ),
            instrumentation=(
                InstrumentationPoint(
                    "AIT-DOS-001", "chlorine_analyzer", ("chlorine",), "mixed zone"
                ),
                InstrumentationPoint(
                    "AIT-DOS-002", "ph_analyzer", ("ph_proxy",), "mixed zone"
                ),
            ),
            limitations=_common_limitations(
                "mixing energy is represented by CFD terms"
            ),
        ),
        UnitProcess(
            unit_id="flocculation-placeholder",
            area_id="clarification",
            display_name="Flocculation-ready placeholder",
            purpose="Reserve a future flocculation unit without claiming chemistry.",
            geometry_class="flocculation_placeholder",
            boundaries=(
                _flow_boundary("from-rapid-mix", "input"),
                _flow_boundary("to-clarifier", "output"),
            ),
            process_variables=("flow_rate", "turbidity"),
            instrumentation=(
                InstrumentationPoint(
                    "AIT-FLOC-001", "quality_analyzer", ("turbidity",), "outlet zone"
                ),
            ),
            limitations=_common_limitations(
                "floc growth and settling are not implemented"
            ),
        ),
        UnitProcess(
            unit_id="clarifier",
            area_id="clarification",
            display_name="Clarifier and settling basin",
            purpose="Represent settling basin hydraulics and turbidity reduction context.",
            geometry_class="settling_basin",
            boundaries=(
                _flow_boundary("from-flocculation", "input"),
                _flow_boundary("to-filtration", "output"),
                ProcessBoundary("sludge-draw", "waste", "drain", ("solids_proxy",)),
            ),
            process_variables=("flow_rate", "turbidity", "solids_proxy"),
            instrumentation=(
                InstrumentationPoint(
                    "LIT-CLR-001", "level_transmitter", ("water_level",), "basin"
                ),
                InstrumentationPoint(
                    "AIT-CLR-001", "quality_analyzer", ("turbidity",), "effluent"
                ),
            ),
            limitations=_common_limitations(
                "settling behavior is CFD-supported context"
            ),
        ),
        UnitProcess(
            unit_id="filter-backwash",
            area_id="filtration",
            display_name="Filter bed and backwash interface",
            purpose="Represent filtration headloss context and backwash transitions.",
            geometry_class="filter_bed",
            boundaries=(
                _flow_boundary("from-clarifier", "input"),
                ProcessBoundary(
                    "filter-media",
                    "internal",
                    "media_interface",
                    ("headloss_proxy", "turbidity"),
                ),
                _flow_boundary("to-disinfection", "output"),
                ProcessBoundary("backwash-drain", "waste", "drain", ("flow_rate",)),
            ),
            process_variables=("flow_rate", "turbidity", "headloss_proxy"),
            instrumentation=(
                InstrumentationPoint(
                    "DPIT-FIL-001",
                    "differential_pressure",
                    ("headloss_proxy",),
                    "filter bed",
                ),
                InstrumentationPoint(
                    "AIT-FIL-001", "quality_analyzer", ("turbidity",), "filter outlet"
                ),
            ),
            limitations=_common_limitations("media fouling physics is not calibrated"),
        ),
        UnitProcess(
            unit_id="contact-basin",
            area_id="disinfection",
            display_name="Disinfection contact basin",
            purpose="Track spatial contact context for disinfectant residual.",
            geometry_class="contact_basin",
            boundaries=(
                _flow_boundary("from-filtration", "input"),
                ProcessBoundary(
                    "disinfectant-trim",
                    "input",
                    "chemical_source",
                    ("chlorine", "chloramine"),
                ),
                _flow_boundary("to-clearwell", "output"),
            ),
            process_variables=("flow_rate", "chlorine", "chloramine", "ammonia"),
            instrumentation=(
                InstrumentationPoint(
                    "AIT-DIS-001", "chlorine_analyzer", ("chlorine",), "basin outlet"
                ),
                InstrumentationPoint(
                    "AIT-DIS-002", "ammonia_analyzer", ("ammonia",), "basin outlet"
                ),
            ),
            limitations=_common_limitations("disinfection efficacy is not certified"),
        ),
        UnitProcess(
            unit_id="clearwell",
            area_id="storage-pumping",
            display_name="Clearwell and treated-water storage",
            purpose="Represent storage volume, residual decay context, and level state.",
            geometry_class="storage_tank",
            boundaries=(
                _flow_boundary("from-contact-basin", "input"),
                ProcessBoundary(
                    "free-surface", "internal", "level_surface", ("water_level",)
                ),
                _flow_boundary("to-pump-header", "output"),
            ),
            process_variables=("water_level", "chlorine", "temperature"),
            instrumentation=(
                InstrumentationPoint(
                    "LIT-CW-001", "level_transmitter", ("water_level",), "clearwell"
                ),
                InstrumentationPoint(
                    "AIT-CW-001", "chlorine_analyzer", ("chlorine",), "clearwell outlet"
                ),
            ),
            limitations=_common_limitations(
                "storage hydraulics use bounded grid presets"
            ),
        ),
        UnitProcess(
            unit_id="pumping-header",
            area_id="storage-pumping",
            display_name="Finished-water pumping header",
            purpose="Represent pump/valve boundary effects on downstream flow.",
            geometry_class="pump_header",
            boundaries=(
                _flow_boundary("from-clearwell", "input"),
                ProcessBoundary(
                    "pump-discharge",
                    "output",
                    "flow",
                    ("flow_rate", "hydraulic_head", "pump_status"),
                ),
            ),
            process_variables=("flow_rate", "hydraulic_head", "pump_status"),
            instrumentation=(
                InstrumentationPoint(
                    "FIT-PMP-001", "flow_transmitter", ("flow_rate",), "discharge"
                ),
                InstrumentationPoint(
                    "PIT-PMP-001", "pressure_transmitter", ("hydraulic_head",), "header"
                ),
            ),
            limitations=_common_limitations("pump curves are contract placeholders"),
        ),
        UnitProcess(
            unit_id="chemical-feed-skid",
            area_id="dosing",
            display_name="Chemical feed skid",
            purpose="Represent chemical source terms and dosing equipment context.",
            geometry_class="chemical_feed_skid",
            boundaries=(
                ProcessBoundary(
                    "chemical-storage", "input", "chemical_source", ("chemical_level",)
                ),
                ProcessBoundary(
                    "dose-output",
                    "output",
                    "chemical_source",
                    ("chemical_flow", "concentration"),
                ),
            ),
            process_variables=("chemical_level", "chemical_flow", "concentration"),
            instrumentation=(
                InstrumentationPoint(
                    "LIT-CHEM-001", "level_transmitter", ("chemical_level",), "tank"
                ),
                InstrumentationPoint(
                    "FIT-CHEM-001", "flow_transmitter", ("chemical_flow",), "dose line"
                ),
            ),
            limitations=_common_limitations("chemical storage hazards are not modeled"),
        ),
        UnitProcess(
            unit_id="waste-backwash-handling",
            area_id="filtration",
            display_name="Waste and backwash handling",
            purpose="Represent waste/backwash routing metadata for lab scenarios.",
            geometry_class="waste_handling",
            boundaries=(
                ProcessBoundary(
                    "filter-backwash-input", "input", "drain", ("flow_rate",)
                ),
                ProcessBoundary("waste-output", "output", "drain", ("flow_rate",)),
            ),
            process_variables=("flow_rate", "turbidity", "waste_status"),
            instrumentation=(
                InstrumentationPoint(
                    "FIT-WST-001", "flow_transmitter", ("flow_rate",), "waste line"
                ),
            ),
            limitations=_common_limitations("waste handling is metadata first"),
        ),
    )


def unit_process_ids() -> tuple[str, ...]:
    return tuple(unit.unit_id for unit in unit_process_catalog())


def get_unit_process(unit_id: str) -> UnitProcess:
    for unit in unit_process_catalog():
        if unit.unit_id == unit_id:
            return unit
    raise ValueError(f"unknown unit process: {unit_id}")


def units_by_area(area_id: str) -> tuple[UnitProcess, ...]:
    if area_id not in AREA_IDS:
        raise ValueError(f"unsupported area id: {area_id}")
    return tuple(unit for unit in unit_process_catalog() if unit.area_id == area_id)
