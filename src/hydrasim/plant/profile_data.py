"""Built-in reference water-plant profiles."""

from __future__ import annotations

from .catalog import AREAS
from .models import PlantNode, PlantProfile, PlantUnit

COMMON_LIMITATIONS = (
    "HydraSim plant profiles are synthetic reference configurations.",
    "Profile metadata does not identify a real plant, vendor deployment, or site.",
    "Architecture layers are for local simulation and traffic-generation labs.",
    "HydraSim does not claim safety, commissioning, certification, or field validation.",
)

AREA_LABELS = {
    "intake": "Raw-water intake",
    "dosing": "Chemical dosing",
    "clarification": "Clarification",
    "filtration": "Filtration",
    "disinfection": "Disinfection",
    "storage-pumping": "Storage and pumping",
    "distribution-edge": "Distribution edge placeholder",
}


def _mac(index: int) -> str:
    return f"02:00:00:00:50:{index:02x}"


def _ip(index: int) -> str:
    return f"198.51.100.{index}"


def _unit(area: str, kind: str, suffix: str, description: str) -> PlantUnit:
    return PlantUnit(
        f"{area}-{suffix}", f"{AREA_LABELS[area]} {kind}", area, kind, description
    )


def _node(
    node_id: str,
    label: str,
    area: str,
    unit_id: str,
    stage: str,
    role: str,
    index: int,
    port: int | None = 502,
    notes: str = "",
) -> PlantNode:
    return PlantNode(
        node_id,
        label,
        area,
        unit_id,
        stage,
        role,
        _mac(index),
        _ip(index),
        "modbus-tcp",
        port,
        notes,
    )


UNITS: tuple[PlantUnit, ...] = (
    _unit("intake", "intake channel", "channel", "Raw-water flow and quality context."),
    _unit("dosing", "dosing skid", "skid", "Chemical feed and setpoint context."),
    _unit(
        "clarification", "settling unit", "basin", "Clarification process placeholder."
    ),
    _unit("filtration", "filter unit", "filter", "Filter and backwash context."),
    _unit("disinfection", "contact basin", "basin", "Chlorine and pH process context."),
    _unit(
        "storage-pumping", "clearwell pump unit", "pump", "Storage and pumping context."
    ),
    _unit(
        "distribution-edge",
        "edge meter",
        "edge",
        "Optional downstream metadata context.",
    ),
    PlantUnit(
        "plant-supervisory",
        "Plant supervisory context",
        "intake",
        "supervisory",
        "Synthetic HMI, historian, engineering workstation, and observer context.",
    ),
)

FIELD_NODES: tuple[PlantNode, ...] = (
    _node(
        "intake-flow-ai",
        "Intake flow transmitter",
        "intake",
        "intake-channel",
        "field-device",
        "flow sensor",
        10,
    ),
    _node(
        "intake-valve-ao",
        "Intake control valve",
        "intake",
        "intake-channel",
        "field-device",
        "control valve",
        11,
    ),
    _node(
        "dosing-ph-ai",
        "Dosing pH analyzer",
        "dosing",
        "dosing-skid",
        "field-device",
        "pH analyzer",
        20,
    ),
    _node(
        "dosing-pump-ao",
        "Chemical dosing pump",
        "dosing",
        "dosing-skid",
        "field-device",
        "dosing pump",
        21,
    ),
    _node(
        "clarifier-level-ai",
        "Clarifier level transmitter",
        "clarification",
        "clarification-basin",
        "field-device",
        "level sensor",
        30,
    ),
    _node(
        "filter-dp-ai",
        "Filter differential pressure sensor",
        "filtration",
        "filtration-filter",
        "field-device",
        "pressure sensor",
        40,
    ),
    _node(
        "filter-backwash-ao",
        "Filter backwash valve",
        "filtration",
        "filtration-filter",
        "field-device",
        "backwash valve",
        41,
    ),
    _node(
        "chlorine-analyzer-ai",
        "Chlorine analyzer",
        "disinfection",
        "disinfection-basin",
        "field-device",
        "chlorine analyzer",
        50,
    ),
    _node(
        "chlorine-pump-ao",
        "Chlorine dosing pump",
        "disinfection",
        "disinfection-basin",
        "field-device",
        "dosing pump",
        51,
    ),
    _node(
        "clearwell-level-ai",
        "Clearwell level transmitter",
        "storage-pumping",
        "storage-pumping-pump",
        "field-device",
        "level sensor",
        60,
    ),
    _node(
        "finished-water-pump-ao",
        "Finished-water pump",
        "storage-pumping",
        "storage-pumping-pump",
        "field-device",
        "pump actuator",
        61,
    ),
    _node(
        "distribution-flow-ai",
        "Distribution edge flow meter",
        "distribution-edge",
        "distribution-edge-edge",
        "field-device",
        "edge meter",
        70,
    ),
)

CONTROLLER_NODES: tuple[PlantNode, ...] = (
    _node(
        "intake-plc",
        "Intake PLC persona",
        "intake",
        "intake-channel",
        "field-controller",
        "PLC-like controller",
        110,
    ),
    _node(
        "dosing-plc",
        "Dosing PLC persona",
        "dosing",
        "dosing-skid",
        "field-controller",
        "PLC-like controller",
        120,
    ),
    _node(
        "clarification-rtu",
        "Clarification RTU persona",
        "clarification",
        "clarification-basin",
        "field-controller",
        "RTU-like controller",
        130,
    ),
    _node(
        "filtration-plc",
        "Filtration PLC persona",
        "filtration",
        "filtration-filter",
        "field-controller",
        "PLC-like controller",
        140,
    ),
    _node(
        "disinfection-plc",
        "Disinfection PLC persona",
        "disinfection",
        "disinfection-basin",
        "field-controller",
        "PLC-like controller",
        150,
    ),
    _node(
        "pumping-plc",
        "Storage and pumping PLC persona",
        "storage-pumping",
        "storage-pumping-pump",
        "field-controller",
        "PLC-like controller",
        160,
    ),
)

SUPERVISORY_NODES: tuple[PlantNode, ...] = (
    _node(
        "stage-driver",
        "Synthetic stage activation driver",
        "intake",
        "plant-supervisory",
        "supervisory",
        "stage activation driver",
        199,
        None,
    ),
    _node(
        "scada-hmi",
        "SCADA HMI workstation",
        "intake",
        "plant-supervisory",
        "supervisory",
        "operator HMI",
        200,
        None,
    ),
    _node(
        "plant-historian",
        "Plant historian",
        "intake",
        "plant-supervisory",
        "supervisory",
        "historian",
        201,
        None,
    ),
    _node(
        "engineering-workstation",
        "Engineering workstation",
        "intake",
        "plant-supervisory",
        "supervisory",
        "engineering workstation",
        202,
        None,
    ),
    _node(
        "passive-observer",
        "Passive observer metadata node",
        "intake",
        "plant-supervisory",
        "passive",
        "passive observer only",
        250,
        None,
        "Must not transmit traffic.",
    ),
)

REFERENCE_PROFILE = PlantProfile(
    "reference-water-plant",
    "HydraSim Reference Water Plant",
    "reference-water-plant",
    "Full multi-area synthetic water-treatment plant profile.",
    AREAS,
    "scada-lite",
    "plant-zones",
    "ethernet",
    "modbus-tcp",
    UNITS,
    FIELD_NODES + CONTROLLER_NODES + SUPERVISORY_NODES,
    COMMON_LIMITATIONS,
)

PROFILES: tuple[PlantProfile, ...] = (
    PlantProfile(
        "single-stage-legacy",
        "Legacy single-stage process endpoint",
        "single-stage-legacy",
        "Compatibility profile representing the current monolithic Modbus endpoint.",
        ("disinfection",),
        "scada-lite",
        "flat-cell",
        "ethernet",
        "modbus-tcp",
        tuple(
            unit
            for unit in UNITS
            if unit.area == "disinfection" or unit.unit_id == "plant-supervisory"
        ),
        tuple(
            node
            for node in FIELD_NODES + SUPERVISORY_NODES
            if node.area == "disinfection"
            or node.node_id
            in {"stage-driver", "scada-hmi", "plant-historian", "passive-observer"}
        ),
        COMMON_LIMITATIONS
        + ("Legacy mode is preserved for regression compatibility.",),
    ),
    PlantProfile(
        "field-device-lab",
        "Field device lab",
        "field-device-lab",
        "Field-device-only profile for sensors, analyzers, valves, pumps, and meters.",
        AREAS,
        "scada-lite",
        "segmented-cell",
        "ethernet",
        "modbus-tcp",
        UNITS,
        FIELD_NODES + (SUPERVISORY_NODES[-1],),
        COMMON_LIMITATIONS
        + (
            "This profile does not include controller or supervisory runtime behavior.",
        ),
    ),
    PlantProfile(
        "controller-cell",
        "Controller cell",
        "controller-cell",
        "Field device and controller profile for PLC/RTU-like behavior.",
        AREAS[:-1],
        "pcs-minimal",
        "segmented-cell",
        "ethernet",
        "modbus-tcp",
        tuple(unit for unit in UNITS if unit.area != "distribution-edge"),
        tuple(node for node in FIELD_NODES if node.area != "distribution-edge")
        + CONTROLLER_NODES
        + (SUPERVISORY_NODES[-1],),
        COMMON_LIMITATIONS
        + (
            "Controller personas are bounded synthetic controllers, not real PLC firmware.",
        ),
    ),
    PlantProfile(
        "supervisory-lab",
        "Supervisory lab",
        "supervisory-lab",
        "HMI, historian, and engineering workstation profile with controller targets.",
        AREAS[:-1],
        "dcs-lite",
        "plant-zones",
        "ethernet",
        "modbus-tcp",
        tuple(unit for unit in UNITS if unit.area != "distribution-edge"),
        CONTROLLER_NODES + SUPERVISORY_NODES,
        COMMON_LIMITATIONS
        + ("Supervisory polling is synthetic and not a vendor system emulation.",),
    ),
    REFERENCE_PROFILE,
)
