"""Maintenance-command bridge for the HydraSim Modbus slave."""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..maintenance import MaintenanceResult


class ModbusMaintenanceMixin:
    maintenance_manager: Any
    _lock: Any
    ir_block: Any

    if TYPE_CHECKING:

        def read_coil(self, name: str) -> bool: ...
        def read_holding_register(self, name: str) -> float: ...
        def write_coil(self, name: str, value: bool) -> None: ...

    def poll_maintenance(self) -> "Optional[MaintenanceResult]":
        """
        Check the maintenance trigger coil and, if set, dispatch to the
        MaintenanceManager, publish the result to IR 110-112, and
        auto-clear the coil.

        Call this once per simulation tick from your simulation loop.
        It is a no-op when no MaintenanceManager was provided or when
        the trigger coil is False.

        Returns
        -------
        MaintenanceResult | None
            The result of the maintenance action, or None if no action
            was triggered.
        """
        if self.maintenance_manager is None:
            return None

        if not self.read_coil("maintenance_trigger"):
            return None

        # --- Read command registers ----------------------------------------
        target_id = int(self.read_holding_register("maintenance_target_id"))
        action_code = int(self.read_holding_register("maintenance_action_code"))
        param = self.read_holding_register("maintenance_param")

        # --- Write PENDING status so clients can detect in-progress state ---
        self._write_maintenance_status(
            status_code=5,  # MaintenanceStatus.PENDING
            last_target=target_id,
            last_action=action_code,
        )

        # --- Dispatch to the manager ----------------------------------------
        result = self.maintenance_manager.execute(
            target_id=target_id,
            action_code=action_code,
            param=param,
        )

        # --- Publish result to IR 110-112 -----------------------------------
        self._write_maintenance_status(
            status_code=int(result.status),
            last_target=result.target_id,
            last_action=result.action_id,
        )

        # --- Auto-clear trigger coil ----------------------------------------
        self.write_coil("maintenance_trigger", False)

        logging.info(
            "Maintenance: target=%d action=%d status=%s — %s",
            result.target_id,
            result.action_id,
            result.status.name,
            result.message,
        )
        return result

    def _write_maintenance_status(
        self,
        status_code: int,
        last_target: int,
        last_action: int,
    ) -> None:
        """Write maintenance result fields to IR 110-112 (thread-safe)."""
        with self._lock:
            # IR addresses are 0-based internally; wire address = address + 1
            self.ir_block.setValues(111, [status_code])  # IR 110
            self.ir_block.setValues(112, [last_target])  # IR 111
            self.ir_block.setValues(113, [last_action])  # IR 112
