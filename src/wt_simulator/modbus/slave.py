"""
Modbus TCP Slave Server
=====================================================

Security-hardened Modbus/TCP server with correct async lifecycle.

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import asyncio
import threading
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from contextlib import suppress

# Modern pymodbus 3.x imports
from pymodbus import ModbusDeviceIdentification
from pymodbus.server import StartAsyncTcpServer, ServerAsyncStop
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusDeviceContext,
    ModbusServerContext,
)

from .register_map import ModbusRegisterMap, RegisterType
from .protocols import ModbusEncoder, ModbusDecoder


@dataclass
class ModbusServerConfig:
    """Configuration for Modbus TCP server."""

    host: str = "0.0.0.0"
    port: int = 5020
    unit_id: int = 1

    # Server identification
    vendor_name: str = "Water Treatment Simulator"
    product_code: str = "WTS-1000"
    vendor_url: str = "https://github.com/water-treatment-sim"
    product_name: str = "CSTR Physics Simulator"
    model_name: str = "Virtual PLC v1.0"
    version: str = "1.0.0"

    # Timeouts
    startup_timeout_sec: float = 5.0
    shutdown_timeout_sec: float = 3.0


class ModbusSlave:
    """
    Hardened Modbus TCP slave server with correct async patterns.

    CRITICAL: This version properly handles pymodbus 3.x async requirements
    by creating the server INSIDE the async context, not before.
    """

    def __init__(
        self,
        register_map: ModbusRegisterMap,
        config: Optional[ModbusServerConfig] = None,
    ):
        """Initialize Modbus slave server."""

        self.register_map = register_map
        self.config = config or ModbusServerConfig()

        # Encoder/decoder
        self.encoder = ModbusEncoder()
        self.decoder = ModbusDecoder()

        # Create data blocks
        self._create_data_blocks()

        # Create Modbus server context
        # Ensure slave_context uses the initialized blocks
        slave_context = ModbusDeviceContext(
            di=self.di_block, co=self.co_block, hr=self.hr_block, ir=self.ir_block
        )

        self.context = ModbusServerContext(
            devices={self.config.unit_id: slave_context}, single=False
        )

        self.identity = ModbusDeviceIdentification()
        self.identity.VendorName = self.config.vendor_name
        self.identity.ProductCode = self.config.product_code
        self.identity.VendorUrl = self.config.vendor_url
        self.identity.ProductName = self.config.product_name
        self.identity.ModelName = self.config.model_name
        self.identity.MajorMinorRevision = self.config.version

        # Lifecycle management
        self.server_task: Optional[asyncio.Task] = None
        self.server_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Synchronization
        self._lock = threading.RLock()
        self._running = threading.Event()
        self._server_ready = threading.Event()
        self._shutdown_requested = threading.Event()

        logging.info(
            f"Modbus slave initialized: {self.config.host}:{self.config.port}, "
            f"unit_id={self.config.unit_id}"
        )

    def _create_data_blocks(self):
        """Create Modbus data blocks."""
        max_ir_addr = max(
            (r.address + r.size_words for r in self.register_map.input_registers),
            default=0,
        )
        max_hr_addr = max(
            (r.address + r.size_words for r in self.register_map.holding_registers),
            default=0,
        )
        max_co_addr = max((r.address + 1 for r in self.register_map.coils), default=0)
        max_di_addr = max(
            (r.address + 1 for r in self.register_map.discrete_inputs), default=0
        )

        # Fixed-size blocks
        ir_size = max(max_ir_addr + 10, 200)
        hr_size = max(max_hr_addr + 10, 200)
        co_size = max(max_co_addr + 10, 100)
        di_size = max(max_di_addr + 10, 100)

        self.ir_block = ModbusSequentialDataBlock(0, [0] * ir_size)
        self.hr_block = ModbusSequentialDataBlock(0, [0] * hr_size)
        self.co_block = ModbusSequentialDataBlock(0, [0] * co_size)
        self.di_block = ModbusSequentialDataBlock(0, [0] * di_size)

    def update_input_register(self, name: str, value: float):
        """Update input register (thread-safe with validation)."""
        try:
            reg = self.register_map.get_register_by_name(name)
            if not reg or reg.register_type != RegisterType.INPUT_REGISTER:
                raise ValueError("Invalid register reference")

            # Validate range
            if not (-1e9 <= value <= 1e9):
                raise ValueError("Value out of range")

            with self._lock:
                if reg.data_type == "float32":
                    high, low = self.encoder.float32_to_registers(value)
                    self.ir_block.setValues(reg.address, [high, low])
                elif reg.data_type == "int16":
                    reg_val = self.encoder.int16_to_register(int(value))
                    self.ir_block.setValues(reg.address, [reg_val])
                elif reg.data_type == "uint16":
                    reg_val = self.encoder.uint16_to_register(int(value))
                    self.ir_block.setValues(reg.address, [reg_val])

        except ValueError:
            raise
        except Exception:
            raise ValueError("Register update failed")

    def update_discrete_input(self, name: str, value: bool):
        """Update discrete input (thread-safe)."""
        try:
            reg = self.register_map.get_register_by_name(name)
            if not reg or reg.register_type != RegisterType.DISCRETE_INPUT:
                raise ValueError("Invalid register reference")

            with self._lock:
                coil_val = self.encoder.bool_to_coil(value)
                self.di_block.setValues(reg.address, [coil_val])

        except ValueError:
            raise
        except Exception:
            raise ValueError("Discrete input update failed")

    def read_holding_register(self, name: str) -> float:
        """Read holding register (thread-safe)."""
        try:
            reg = self.register_map.get_register_by_name(name)
            if not reg or reg.register_type != RegisterType.HOLDING_REGISTER:
                raise ValueError("Invalid register reference")

            with self._lock:
                if reg.data_type == "float32":
                    values = self.hr_block.getValues(reg.address, 2)
                    return self.decoder.registers_to_float32(values[0], values[1])
                elif reg.data_type == "int16":
                    values = self.hr_block.getValues(reg.address, 1)
                    return self.decoder.register_to_int16(values[0])
                elif reg.data_type == "uint16":
                    values = self.hr_block.getValues(reg.address, 1)
                    return self.decoder.register_to_uint16(values[0])

        except ValueError:
            raise
        except Exception:
            raise ValueError("Register read failed")

    def read_coil(self, name: str) -> bool:
        """Read coil (thread-safe)."""
        try:
            reg = self.register_map.get_register_by_name(name)
            if not reg or reg.register_type != RegisterType.COIL:
                raise ValueError("Invalid register reference")

            with self._lock:
                values = self.co_block.getValues(reg.address, 1)
                return self.decoder.coil_to_bool(values[0])

        except ValueError:
            raise
        except Exception:
            raise ValueError("Coil read failed")

    def write_holding_register(self, name: str, value: float):
        """Write holding register (thread-safe with validation)."""
        try:
            reg = self.register_map.get_register_by_name(name)
            if not reg or reg.register_type != RegisterType.HOLDING_REGISTER:
                raise ValueError("Invalid register reference")

            if not (-1e9 <= value <= 1e9):
                raise ValueError("Value out of range")

            with self._lock:
                if reg.data_type == "float32":
                    high, low = self.encoder.float32_to_registers(value)
                    self.hr_block.setValues(reg.address, [high, low])
                elif reg.data_type == "int16":
                    reg_val = self.encoder.int16_to_register(int(value))
                    self.hr_block.setValues(reg.address, [reg_val])
                elif reg.data_type == "uint16":
                    reg_val = self.encoder.uint16_to_register(int(value))
                    self.hr_block.setValues(reg.address, [reg_val])

        except ValueError:
            raise
        except Exception:
            raise ValueError("Register write failed")

    def start(self, blocking: bool = True):
        """
        Start Modbus server (CRITICAL FIX: proper async initialization).

        Args:
            blocking: If True, block until server stops
                     If False, run in background thread
        """
        if self._running.is_set():
            logging.warning("Modbus server already running")
            return

        self._running.set()
        self._server_ready.clear()
        self._shutdown_requested.clear()

        if blocking:
            self._run_server()
        else:
            self.server_thread = threading.Thread(
                target=self._run_server, daemon=True, name="ModbusTCPServer"
            )
            self.server_thread.start()

            # Wait for server ready
            if not self._server_ready.wait(timeout=self.config.startup_timeout_sec):
                self._running.clear()
                raise RuntimeError("Server startup timeout")

            logging.info(
                f"Modbus server started on {self.config.host}:{self.config.port}"
            )

    def _run_server(self):
        """
        Run Modbus server (CRITICAL FIX: correct async pattern).

        Key change: Server is created INSIDE the async context using
        StartAsyncTcpServer, not before the event loop starts.
        """
        loop = None
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop

            # CRITICAL: Run the async server initialization inside the event loop
            loop.run_until_complete(self._async_run_server())

        except Exception as e:
            logging.error(f"Modbus server error: {type(e).__name__}")
            self._running.clear()

        finally:
            # Signal ready even on error (to unblock waiting threads)
            self._server_ready.set()

            # Cleanup
            if loop and not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                with suppress(Exception):
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

                loop.close()

            self._event_loop = None

    async def _async_run_server(self):
        """Async server runner for pymodbus 3.11.4."""
        try:
            self._server_ready.set()

            # Ensure the address is a tuple and parameters are explicitly named
            await StartAsyncTcpServer(
                context=self.context,
                identity=self.identity,
                address=(self.config.host, self.config.port),
            )

            while not self._shutdown_requested.is_set():
                await asyncio.sleep(0.1)

        except Exception as e:
            logging.error(f"Server runtime error: {e}")
            raise
        finally:
            await ServerAsyncStop()

    def stop(self):
        """Stop Modbus server (graceful shutdown)."""
        if not self._running.is_set():
            return

        self._shutdown_requested.set()
        self._running.clear()

        # Cancel all tasks in the event loop
        if self._event_loop and not self._event_loop.is_closed():
            try:
                # Get all tasks and cancel them
                def cancel_all_tasks():
                    tasks = [
                        t for t in asyncio.all_tasks(self._event_loop) if not t.done()
                    ]
                    for task in tasks:
                        task.cancel()

                self._event_loop.call_soon_threadsafe(cancel_all_tasks)

            except Exception as e:
                logging.warning(f"Server shutdown error: {type(e).__name__}")

        # Wait for thread to finish
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=self.config.shutdown_timeout_sec)

            if self.server_thread.is_alive():
                logging.warning("Server thread did not terminate cleanly")

        logging.info("Modbus server stopped")

    def get_all_holding_registers(self) -> Dict[str, float]:
        """Read all holding registers (thread-safe)."""
        values = {}
        for reg in self.register_map.holding_registers:
            try:
                values[reg.name] = self.read_holding_register(reg.name)
            except Exception:
                values[reg.name] = 0.0
        return values

    def get_all_coils(self) -> Dict[str, bool]:
        """Read all coils (thread-safe)."""
        values = {}
        for reg in self.register_map.coils:
            try:
                values[reg.name] = self.read_coil(reg.name)
            except Exception:
                values[reg.name] = False
        return values

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running.is_set()


def example_usage():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)

    reg_map = ModbusRegisterMap()
    config = ModbusServerConfig(host="127.0.0.1", port=5020)

    slave = ModbusSlave(reg_map, config)

    # Initialize values
    slave.write_holding_register("acid_flow_rate", 0.5)
    slave.update_input_register("pH_inlet", 7.25)
    slave.update_discrete_input("sensor_fault_pH_inlet", False)

    print("Starting Modbus server on 127.0.0.1:5020")
    print("Press Ctrl+C to stop")

    try:
        slave.start(blocking=False)
    except RuntimeError as e:
        print(f"Failed to start: {e}")
        return

    try:
        import math

        t = 0
        while slave.is_running:
            time.sleep(1.0)
            t += 1
            pH = 7.2 + 0.1 * math.sin(t * 0.1)
            slave.update_input_register("pH_inlet", pH)
            acid_rate = slave.read_holding_register("acid_flow_rate")
            print(f"t={t}s: pH={pH:.2f}, acid={acid_rate:.2f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        slave.stop()


if __name__ == "__main__":
    example_usage()
