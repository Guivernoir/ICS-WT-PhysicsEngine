"""HydraSim architecture constants for the reference water plant."""

AREAS = (
    "intake",
    "dosing",
    "clarification",
    "filtration",
    "disinfection",
    "storage-pumping",
    "distribution-edge",
)
AREA_CHOICES = AREAS + ("all",)
STAGES = (
    "field-devices",
    "field-controllers",
    "supervisory",
    "full-cell",
    "offline-export",
)
NODE_STAGES = ("field-device", "field-controller", "supervisory", "passive")
PRESETS = (
    "single-stage-legacy",
    "field-device-lab",
    "controller-cell",
    "supervisory-lab",
    "reference-water-plant",
)
CONTROL_SYSTEMS = ("scada-lite", "pcs-minimal", "dcs-lite")
TOPOLOGIES = ("flat-cell", "segmented-cell", "plant-zones")
MEDIA = ("ethernet", "serial-gateway-placeholder", "mixed-lab")
PROTOCOLS = ("modbus-tcp",)

STAGE_NODE_FILTERS = {
    "field-devices": ("field-device",),
    "field-controllers": ("field-device", "field-controller"),
    "supervisory": ("field-controller", "supervisory"),
    "full-cell": ("field-device", "field-controller", "supervisory", "passive"),
    "offline-export": ("field-device", "field-controller", "supervisory", "passive"),
}

STAGE_TRANSACTION_FILTERS = {
    "field-devices": ("field-devices",),
    "field-controllers": ("field-devices", "field-controllers"),
    "supervisory": ("supervisory",),
    "full-cell": ("field-devices", "field-controllers", "supervisory"),
    "offline-export": ("field-devices", "field-controllers", "supervisory"),
}
