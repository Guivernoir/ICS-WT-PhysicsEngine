"""
Maintenance Package
====================

Remote recalibration and repair of sensors and actuators.

This package is transport-agnostic: it exposes a pure-Python
``MaintenanceManager`` that callers (Modbus handler, HTTP endpoint,
test code) can drive directly.  The Modbus-facing integration lives in
``__main__.py``.

Usage
-----
    from hydrasim.maintenance import MaintenanceManager

    manager = MaintenanceManager(sensors, actuators)

    # Calibrate pH_inlet to 7.0, skipping warm-up delay
    result = manager.execute(
        target_id=0,           # MaintenanceTarget.PH_INLET
        action_code=0,         # MaintenanceAction.CALIBRATE
        param=1007.0,          # 1000 + reference_value → skip_warmup=True
    )
    print(result.message)

    # Get health snapshot for all devices
    health = manager.get_health_summary()
    for name, h in health.items():
        print(name, h.calibration_valid, h.cumulative_drift)

Register map additions (see modbus/register_map.py)
----------------------------------------------------
Holding registers
    HR 200  maintenance_target_id   uint16
    HR 201  maintenance_action_code uint16
    HR 202  maintenance_param       float32  (occupies HR 202-203)

Coils
    Coil 10  maintenance_trigger   bool  (write 1 to fire; auto-cleared after exec)

Input registers
    IR 110  maintenance_status_code   uint16
    IR 111  maintenance_last_target   uint16
    IR 112  maintenance_last_action   uint16

Author: Guilherme F. G. Santos
Date:   February 2026
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Guilherme F. G. Santos"

from .maintenance_manager import MaintenanceManager
from .models import (
    DeviceHealth,
    MaintenanceAction,
    MaintenanceResult,
    MaintenanceStatus,
    MaintenanceTarget,
)

__all__ = [
    "MaintenanceManager",
    "MaintenanceTarget",
    "MaintenanceAction",
    "MaintenanceStatus",
    "MaintenanceResult",
    "DeviceHealth",
]
