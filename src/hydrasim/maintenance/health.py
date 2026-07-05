"""Health reporting helpers for HydraSim maintenance management."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from .models import (
    DeviceHealth,
    MaintenanceAction,
    MaintenanceResult,
    MaintenanceStatus,
    MaintenanceTarget,
)

logger = logging.getLogger(__name__)


class MaintenanceHealthMixin:
    _SENSOR_KEYS: Dict[int, str]
    _ACTUATOR_KEYS: Dict[int, str]
    _sensors: Dict[str, Any]
    _actuators: Dict[str, Any]

    def get_health_summary(self) -> Dict[str, DeviceHealth]:
        """
        Return a health snapshot for every known device.

        Returns
        -------
        dict mapping device name → DeviceHealth
        """
        summary: Dict[str, DeviceHealth] = {}

        for tid, key in self._SENSOR_KEYS.items():
            sensor = self._sensors.get(key)
            if sensor is not None:
                summary[key] = self._sensor_health(tid, key, sensor)

        for tid, key in self._ACTUATOR_KEYS.items():
            actuator = self._actuators.get(key)
            if actuator is not None:
                summary[key] = self._actuator_health(tid, key, actuator)

        return summary

    def _sensor_health(self, tid: int, key: str, sensor: Any) -> DeviceHealth:
        """Build a DeviceHealth snapshot from a sensor object."""
        # Calibration validity (uses the most recent calibration record)
        cal_valid = False
        if hasattr(sensor, "calibration_history") and sensor.calibration_history:
            last_cal = sensor.calibration_history[-1]
            cal_valid = not last_cal.is_expired(time.monotonic())

        drift = getattr(sensor, "cumulative_drift", 0.0)

        extras: Dict[str, Any] = {}

        # pH sensor
        if hasattr(sensor, "slope_percentage"):
            extras["slope_pct"] = sensor.slope_percentage
            extras["membrane_fouling"] = sensor.membrane_fouling
            extras["glass_etching"] = sensor.glass_etching

        # Chlorine amperometric
        if hasattr(sensor, "membrane_fouling") and not hasattr(
            sensor, "slope_percentage"
        ):
            extras["membrane_fouling"] = sensor.membrane_fouling
            extras["membrane_age_days"] = sensor.membrane_age_days

        # Chlorine DPD
        if hasattr(sensor, "reagent_potency"):
            extras["reagent_potency"] = sensor.reagent_potency
            extras["reagent_age_days"] = sensor.reagent_age_days

        # Actuator-related status
        extras["status"] = getattr(sensor, "status", None)
        extras["fault"] = getattr(sensor, "fault", None)

        return DeviceHealth(
            target_id=tid,
            target_name=key,
            device_type="sensor",
            calibration_valid=cal_valid,
            cumulative_drift=drift,
            extras=extras,
        )

    def _actuator_health(self, tid: int, key: str, actuator: Any) -> DeviceHealth:
        """Build a DeviceHealth snapshot from an actuator object."""
        diag = actuator.diagnostics()

        extras: Dict[str, Any] = {
            "cycles_count": diag.cycles_count,
            "hours_runtime": diag.hours_runtime,
            "wear_factor": diag.wear_factor,
            "health_status": diag.health_status,
            "fault_code": diag.fault_code,
            "avg_response_time_s": diag.average_response_time,
        }

        # ControlValve-specific
        if hasattr(actuator, "_accumulated_drift"):
            extras["positioner_drift_pct"] = actuator._accumulated_drift

        # DosingPump-specific
        if hasattr(actuator, "_diaphragm_wear"):
            extras["diaphragm_wear"] = actuator._diaphragm_wear
            extras["check_valve_wear"] = actuator._check_valve_wear
            extras["total_strokes"] = actuator._total_strokes
        if hasattr(actuator, "_tube_wear"):
            extras["tube_wear"] = actuator._tube_wear

        return DeviceHealth(
            target_id=tid,
            target_name=key,
            device_type="actuator",
            calibration_valid=True,  # actuators don't expire calibration
            cumulative_drift=extras.get("positioner_drift_pct", 0.0),
            extras=extras,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_calibrate_param(param: float):
        """
        Decode the composite CALIBRATE parameter.

        Convention:
          param < 1000  → reference_value=param,        skip_warmup=False
          param ≥ 1000  → reference_value=param-1000,   skip_warmup=True
        """
        if param >= 1000.0:
            return param - 1000.0, True
        return param, False

    @staticmethod
    def _not_supported(
        target: MaintenanceTarget,
        action: MaintenanceAction,
        key: str,
        ts: float,
    ) -> MaintenanceResult:
        msg = f"{key}: action '{action.name}' not applicable to this device type"
        logger.warning(msg)
        return MaintenanceResult(
            MaintenanceStatus.ACTION_NOT_SUPPORTED, target, action, msg, ts
        )
