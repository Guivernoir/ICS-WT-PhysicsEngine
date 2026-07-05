"""Maintenance command and health data models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------


class MaintenanceTarget(IntEnum):
    """Numeric IDs for every field device."""

    PH_INLET = 0
    PH_MIDDLE = 1
    PH_OUTLET = 2
    CHLORINE_INLET = 3
    CHLORINE_OUTLET = 4
    FLOW_MAIN = 5
    TEMP_INLET = 6
    TEMP_OUTLET = 7
    ACID_VALVE = 8
    CHLORINE_PUMP = 9
    INLET_VALVE = 10


class MaintenanceAction(IntEnum):
    """Numeric codes for every maintenance action."""

    # Universal sensor actions
    CALIBRATE = 0
    FULL_RESET = 1
    # pH-specific
    CLEAN_WATER = 2
    CLEAN_ACID = 3
    # Amperometric chlorine
    REPLACE_MEMBRANE = 4
    # DPD chlorine
    REPLACE_REAGENT = 5
    # Universal actuator actions
    RESET_FAULTS = 6
    CALIBRATE_ZERO = 7
    # ControlValve-specific
    RECALIBRATE_POSITIONER = 8
    # DosingPump-specific
    REPLACE_DIAPHRAGM = 9
    REPLACE_CHECK_VALVES = 10
    REPLACE_TUBE = 11


class MaintenanceStatus(IntEnum):
    """Result status codes returned after an action."""

    SUCCESS = 0
    INVALID_TARGET = 1
    INVALID_ACTION = 2
    ACTION_NOT_SUPPORTED = 3
    EXECUTION_ERROR = 4
    PENDING = 5


# ---------------------------------------------------------------------------
# Result and health dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MaintenanceResult:
    """Structured result of a maintenance operation."""

    status: MaintenanceStatus
    target_id: int
    action_id: int
    message: str
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def success(self) -> bool:
        return self.status == MaintenanceStatus.SUCCESS


@dataclass
class DeviceHealth:
    """Snapshot of a device's health for diagnostic purposes."""

    target_id: int
    target_name: str
    device_type: str  # "sensor" | "actuator"
    # Universal
    calibration_valid: bool
    cumulative_drift: float  # sensor units or %
    # Type-specific extras (populated per device)
    extras: Dict[str, Any] = field(default_factory=dict)
