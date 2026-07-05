"""
Maintenance Manager
====================

Transport-agnostic core for remote recalibration and repair of sensors
and actuators in the water treatment simulator.

Callers (e.g. Modbus handler, HTTP endpoint, test code) translate their
wire format into a single call:

    result = manager.execute(target_id, action_code, param)

The manager returns a structured ``MaintenanceResult`` that the caller
can then encode back into whatever wire format it needs.

Design principles
-----------------
- Pure Python, zero I/O, no Modbus imports.
- Never raises; always returns MaintenanceResult with a status field.
- All side-effects confined to the passed-in sensor/actuator objects.
- Health summaries available for external monitoring/logging.

Target IDs
----------
Sensors  : pH_inlet=0, pH_middle=1, pH_outlet=2,
           chlorine_inlet=3, chlorine_outlet=4,
           flow_main=5, temp_inlet=6, temp_outlet=7
Actuators: acid_valve=8, chlorine_pump=9, inlet_valve=10

Action codes
------------
Universal (sensors) : CALIBRATE=0, FULL_RESET=1
pH only             : CLEAN_WATER=2, CLEAN_ACID=3
Amperometric Cl     : REPLACE_MEMBRANE=4
DPD Cl              : REPLACE_REAGENT=5
Universal (actuators): RESET_FAULTS=6, CALIBRATE_ZERO=7
ControlValve only   : RECALIBRATE_POSITIONER=8
DosingPump only     : REPLACE_DIAPHRAGM=9, REPLACE_CHECK_VALVES=10,
                      REPLACE_TUBE=11

CALIBRATE param encoding
------------------------
  param < 1000   →  reference_value = param,       skip_warmup = False
  param ≥ 1000   →  reference_value = param - 1000, skip_warmup = True
  (allows a single float register to carry both pieces of information)

Author: Guilherme F. G. Santos
Date:   February 2026
License: MIT
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .health import MaintenanceHealthMixin
from .models import (
    MaintenanceAction,
    MaintenanceResult,
    MaintenanceStatus,
    MaintenanceTarget,
)

logger = logging.getLogger(__name__)


class MaintenanceManager(MaintenanceHealthMixin):
    """
    Execute maintenance actions on sensors and actuators.

    Parameters
    ----------
    sensors : dict
        Mapping name → sensor object (as returned by
        ``create_realistic_sensor_suite``).
    actuators : dict
        Mapping name → actuator object (as returned by
        ``create_realistic_actuator_suite``).
    """

    # Maps target_id → sensor key in the sensors dict
    _SENSOR_KEYS: Dict[int, str] = {
        MaintenanceTarget.PH_INLET: "pH_inlet",
        MaintenanceTarget.PH_MIDDLE: "pH_middle",
        MaintenanceTarget.PH_OUTLET: "pH_outlet",
        MaintenanceTarget.CHLORINE_INLET: "chlorine_inlet",
        MaintenanceTarget.CHLORINE_OUTLET: "chlorine_outlet",
        MaintenanceTarget.FLOW_MAIN: "flow_main",
        MaintenanceTarget.TEMP_INLET: "temp_inlet",
        MaintenanceTarget.TEMP_OUTLET: "temp_outlet",
    }

    # Maps target_id → actuator key in the actuators dict
    _ACTUATOR_KEYS: Dict[int, str] = {
        MaintenanceTarget.ACID_VALVE: "acid_valve",
        MaintenanceTarget.CHLORINE_PUMP: "chlorine_pump",
        MaintenanceTarget.INLET_VALVE: "inlet_valve",
    }

    def __init__(
        self,
        sensors: Dict[str, Any],
        actuators: Dict[str, Any],
    ) -> None:
        self._sensors = sensors
        self._actuators = actuators

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(
        self,
        target_id: int,
        action_code: int,
        param: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> MaintenanceResult:
        """
        Execute a maintenance action and return its result.

        This method never raises; all errors are captured in the result.

        Parameters
        ----------
        target_id : int
            ``MaintenanceTarget`` value identifying the device.
        action_code : int
            ``MaintenanceAction`` value.
        param : float
            Optional parameter.  For CALIBRATE actions see the module
            docstring for the encoding convention.
        timestamp : float, optional
            Monotonic time to pass to sensor/actuator methods.
            Defaults to ``time.monotonic()``.

        Returns
        -------
        MaintenanceResult
        """
        ts = timestamp if timestamp is not None else time.monotonic()

        try:
            target = MaintenanceTarget(target_id)
        except ValueError:
            msg = f"Unknown target_id={target_id}"
            logger.warning(msg)
            return MaintenanceResult(
                MaintenanceStatus.INVALID_TARGET, target_id, action_code, msg, ts
            )

        try:
            action = MaintenanceAction(action_code)
        except ValueError:
            msg = f"Unknown action_code={action_code}"
            logger.warning(msg)
            return MaintenanceResult(
                MaintenanceStatus.INVALID_ACTION, target_id, action_code, msg, ts
            )

        logger.info(
            "Maintenance: target=%s action=%s param=%.4f",
            target.name,
            action.name,
            param,
        )

        if target_id in self._SENSOR_KEYS:
            return self._apply_sensor_action(target, action, param, ts)
        else:
            return self._apply_actuator_action(target, action, param, ts)

    # ------------------------------------------------------------------
    # Internal: sensor dispatch
    # ------------------------------------------------------------------

    def _apply_sensor_action(
        self,
        target: MaintenanceTarget,
        action: MaintenanceAction,
        param: float,
        ts: float,
    ) -> MaintenanceResult:
        key = self._SENSOR_KEYS[target]
        sensor = self._sensors.get(key)

        if sensor is None:
            msg = f"Sensor '{key}' not found in sensor suite"
            logger.error(msg)
            return MaintenanceResult(
                MaintenanceStatus.EXECUTION_ERROR, target, action, msg, ts
            )

        try:
            # ---- Universal sensor actions --------------------------------
            if action == MaintenanceAction.CALIBRATE:
                ref_val, skip_warmup = self._decode_calibrate_param(param)
                sensor.calibrate(ref_val, ts, skip_warmup=skip_warmup)
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: calibrated to {ref_val:.4f} (skip_warmup={skip_warmup})",
                    ts,
                )

            if action == MaintenanceAction.FULL_RESET:
                sensor.reset()
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: full factory reset applied",
                    ts,
                )

            # ---- pH-specific actions -------------------------------------
            if action == MaintenanceAction.CLEAN_WATER:
                if not hasattr(sensor, "clean_electrode"):
                    return self._not_supported(target, action, key, ts)
                sensor.clean_electrode("water_rinse", ts)
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: water rinse complete (~50% fouling removed)",
                    ts,
                )

            if action == MaintenanceAction.CLEAN_ACID:
                if not hasattr(sensor, "clean_electrode"):
                    return self._not_supported(target, action, key, ts)
                sensor.clean_electrode("acid_clean", ts)
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: acid clean complete (~90% mineral scale removed)",
                    ts,
                )

            # ---- Amperometric chlorine -----------------------------------
            if action == MaintenanceAction.REPLACE_MEMBRANE:
                if not hasattr(sensor, "replace_membrane"):
                    return self._not_supported(target, action, key, ts)
                sensor.replace_membrane(ts)
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: membrane replaced, fouling/polarisation reset",
                    ts,
                )

            # ---- DPD chlorine -------------------------------------------
            if action == MaintenanceAction.REPLACE_REAGENT:
                if not hasattr(sensor, "replace_reagent"):
                    return self._not_supported(target, action, key, ts)
                sensor.replace_reagent(ts)
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: reagent replaced, potency restored to 100%",
                    ts,
                )

            # Actuator codes on a sensor target are invalid
            return MaintenanceResult(
                MaintenanceStatus.ACTION_NOT_SUPPORTED,
                target,
                action,
                f"{key}: action '{action.name}' is not valid for sensors",
                ts,
            )

        except Exception as exc:
            msg = f"{key}: action '{action.name}' raised {type(exc).__name__}: {exc}"
            logger.exception(msg)
            return MaintenanceResult(
                MaintenanceStatus.EXECUTION_ERROR, target, action, msg, ts
            )

    # ------------------------------------------------------------------
    # Internal: actuator dispatch
    # ------------------------------------------------------------------

    def _apply_actuator_action(
        self,
        target: MaintenanceTarget,
        action: MaintenanceAction,
        param: float,
        ts: float,
    ) -> MaintenanceResult:
        key = self._ACTUATOR_KEYS[target]
        actuator = self._actuators.get(key)

        if actuator is None:
            msg = f"Actuator '{key}' not found in actuator suite"
            logger.error(msg)
            return MaintenanceResult(
                MaintenanceStatus.EXECUTION_ERROR, target, action, msg, ts
            )

        try:
            # ---- Universal actuator actions ------------------------------
            if action == MaintenanceAction.RESET_FAULTS:
                actuator.reset_faults()
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: fault codes cleared",
                    ts,
                )

            if action == MaintenanceAction.CALIBRATE_ZERO:
                actuator.calibrate_zero()
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: zero/span recalibration applied",
                    ts,
                )

            # ---- ControlValve-specific ----------------------------------
            if action == MaintenanceAction.RECALIBRATE_POSITIONER:
                if not hasattr(actuator, "recalibrate_positioner"):
                    return self._not_supported(target, action, key, ts)
                actuator.recalibrate_positioner()
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: positioner drift reset, CALIBRATION fault cleared",
                    ts,
                )

            # ---- DosingPump-specific ------------------------------------
            if action == MaintenanceAction.REPLACE_DIAPHRAGM:
                if not hasattr(actuator, "replace_diaphragm"):
                    return self._not_supported(target, action, key, ts)
                actuator.replace_diaphragm()
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: diaphragm replaced, stroke-wear counter reset",
                    ts,
                )

            if action == MaintenanceAction.REPLACE_CHECK_VALVES:
                if not hasattr(actuator, "replace_check_valves"):
                    return self._not_supported(target, action, key, ts)
                actuator.replace_check_valves()
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: check valves replaced, wear reset",
                    ts,
                )

            if action == MaintenanceAction.REPLACE_TUBE:
                if not hasattr(actuator, "replace_tube"):
                    return self._not_supported(target, action, key, ts)
                actuator.replace_tube()
                return MaintenanceResult(
                    MaintenanceStatus.SUCCESS,
                    target,
                    action,
                    f"{key}: peristaltic tube replaced",
                    ts,
                )

            # Sensor codes on an actuator target are invalid
            return MaintenanceResult(
                MaintenanceStatus.ACTION_NOT_SUPPORTED,
                target,
                action,
                f"{key}: action '{action.name}' is not valid for actuators",
                ts,
            )

        except Exception as exc:
            msg = f"{key}: action '{action.name}' raised {type(exc).__name__}: {exc}"
            logger.exception(msg)
            return MaintenanceResult(
                MaintenanceStatus.EXECUTION_ERROR, target, action, msg, ts
            )

    # ------------------------------------------------------------------
    # Internal: health snapshots
    # ------------------------------------------------------------------
