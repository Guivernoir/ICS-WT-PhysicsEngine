# Modbus Interface Module

**Version:** 1.0.0  
**Author:** Guilherme F. G. Santos  
**License:** MIT  

## Table of Contents

- [Overview](#overview)
- [What This Module Does](#what-this-module-does)
- [What This Module Does NOT Do](#what-this-module-does-not-do)
- [Architecture](#architecture)
- [Capabilities](#capabilities)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Maintenance Subsystem](#maintenance-subsystem)
- [Edge Cases & Limitations](#edge-cases--limitations)
- [Validation](#validation)
- [Integration with Simulation](#integration-with-simulation)
- [References](#references)

## Overview

This module implements a **pure Modbus/TCP protocol adapter** for the water treatment reactor simulation. It exposes sensor readings as read-only input registers and accepts actuator setpoints as writable holding registers, enabling external SCADA systems, HMIs, PLCs, or control software to interact with the simulator as if it were real hardware.

The design is deliberately narrow: **protocol and data formatting only**. It does not understand the meaning of the data, validate values, enforce limits, or implement any control logic. This mirrors real industrial Modbus devices, where the protocol layer is dumb and responsibility for data interpretation lies with the application.

## What This Module Does

✅ **Modbus/TCP Server**  
- Listens on TCP port 502 (or custom)  
- Supports standard function codes (FC 01–06, 15–16, etc.)  
- Handles multiple client connections  

✅ **Register Mapping**  
- Defines a complete, water-treatment-specific register map  
- Input Registers (30001+): Sensor values (pH, chlorine, flow, temperature, faults)  
- Holding Registers (40001+): Actuator setpoints (acid flow rate, chlorine dosing, etc.)  
- Coils (00001+): Discrete outputs (pumps on/off, valves)  
- Discrete Inputs (10001+): Status bits (sensor faults, calibration expired)  

✅ **Data Encoding/Decoding**  
- IEEE 754 single-precision float32 over two consecutive registers (big-endian)  
- Signed/unsigned 16-bit integers  
- Boolean coils  
- Round-trip safe conversion utilities  

✅ **Dynamic Updates**  
- API to push sensor values (`update_input_register()`)  
- API to read actuator commands (`read_holding_register()`)  
- Non-blocking server start  

✅ **Device Identification**  
- Vendor name, product code, model, version strings visible to clients  

## What This Module Does NOT Do

❌ **Data Interpretation or Validation**  
- No range checking, no safety limits, no command validation  
- Accepts any writable value (realistic Modbus behavior)  

❌ **Control Logic**  
- No PID, MPC, sequencing, alarms, or interlocks  
- No physics, chemistry, or sensor simulation  

❌ **Security Features**  
- No authentication, TLS, or access control (Modbus TCP is plaintext)  

❌ **Advanced Modbus Features**  
- No diagnostics counters (bad CRCs, timeouts)  
- No custom function codes  
- No Modbus RTU/serial support  

❌ **Simulation Coupling**  
- Does not read sensors or drive actuators directly  
- Caller must update registers in the simulation loop  

## Architecture

```
┌─────────────────────┐
│   External Client   │  (SCADA, HMI, PLC, OPC UA gateway, etc.)
└─────────┬───────────┘
          │ Modbus/TCP (port 502)
┌─────────▼───────────┐
│   ModbusSlave       │  ← pymodbus TCP server
│   (this module)     │
└─────────┬───────────┘
          │ Register read/write
┌─────────▼───────────┐
│   Simulation Loop   │  ← Physics engine + Sensors
│   (your code)       │  → Update inputs, read holding registers
└─────────────────────┘
```

- **ModbusSlave**: pymodbus-based TCP server  
- **ModbusRegisterMap**: Defines address → name mapping  
- **ModbusEncoder/Decoder**: Float/int/bool conversions  
- **ModbusServerConfig**: Host, port, device info  

## Capabilities

- **Realistic register layout** for water treatment: pH inlet/outlet, chlorine, flow, temperature, dosing rates, fault bits  
- **Big-endian IEEE 754 float32** (standard in modern devices)  
- **Non-blocking operation** — integrate with simulation timestep loop  
- **Easy value updates** via name-based API (no manual address math)  
- **Device identification** string for client discovery  
- **Validation tests** for encoding round-trips  

## Installation

```bash
# Requires pymodbus
pip install pymodbus

# Copy modbus/ directory into your project
# or install from GitHub
git clone https://github.com/your-repo/hydrasimulation.git
cd hydrasimulation/modbus
pip install -e .
```

## Quick Start

```python
from modbus import ModbusSlave, ModbusRegisterMap, ModbusServerConfig

# Create register map
reg_map = ModbusRegisterMap()

# Configure server (use 5020 for testing to avoid root)
config = ModbusServerConfig(
    host="127.0.0.1",
    port=5020,
    unit_id=1
)

# Create slave
slave = ModbusSlave(reg_map, config)

# Start server (non-blocking)
slave.start(blocking=False)

# In simulation loop: update sensors
slave.update_input_register("pH_inlet", 7.25)
slave.update_input_register("chlorine_outlet", 1.8)

# Read actuator commands
acid_rate = slave.read_holding_register("acid_flow_rate")
print(f"Acid dosing command: {acid_rate} L/min")
```

## Usage Examples

See `slave.py` example_usage() for a full demo with simulated pH oscillation.

```python
# Update multiple values at once
slave.update_input_registers({
    "pH_inlet": 7.25,
    "pH_outlet": 7.18,
    "flow_rate": 50.0,
    "temperature_inlet": 20.5
})

# Read all holding registers
setpoints = slave.get_all_holding_registers()
print(setpoints)
```

## Maintenance Subsystem

The `hydrasim.maintenance` package provides remote recalibration and repair of sensors and actuators over Modbus. The `ModbusSlave` acts as the bridge: it polls a dedicated trigger coil, reads the command registers, dispatches to `MaintenanceManager`, writes the result back to status registers, and auto-clears the trigger — all in one call.

### Register Layout

| Space | Address | Name | Type | Description |
|---|---|---|---|---|
| Holding | HR 200 | `maintenance_target_id` | uint16 | `MaintenanceTarget` enum value (0–10) |
| Holding | HR 201 | `maintenance_action_code` | uint16 | `MaintenanceAction` enum value (0–11) |
| Holding | HR 202–203 | `maintenance_param` | float32 | Action parameter; for CALIBRATE add 1000 to skip warm-up |
| Coil | Coil 10 | `maintenance_trigger` | bool | Write `True` to fire; auto-cleared after execution |
| Input | IR 110 | `maintenance_status_code` | uint16 | `MaintenanceStatus` result (0=SUCCESS … 5=PENDING) |
| Input | IR 111 | `maintenance_last_target` | uint16 | Echo of `target_id` |
| Input | IR 112 | `maintenance_last_action` | uint16 | Echo of `action_code` |

### Workflow (client side)

1. Write `target_id`   → HR 200  
2. Write `action_code` → HR 201  
3. Write `param`       → HR 202–203 (float32; write `0.0` if unused)  
4. Write `True`        → Coil 10 (trigger)  
5. Poll IR 110 until value ≠ 5 (PENDING)  
6. Read IR 110 for final status, IR 111/112 for echo  

### Integration (server side)

Pass a `MaintenanceManager` instance when constructing `ModbusSlave`, then call `slave.poll_maintenance()` once per simulation tick:

```python
from hydrasim.maintenance import MaintenanceManager
from modbus import ModbusSlave, ModbusRegisterMap, ModbusServerConfig

manager = MaintenanceManager(sensors, actuators)
slave   = ModbusSlave(ModbusRegisterMap(), ModbusServerConfig(), maintenance_manager=manager)
slave.start(blocking=False)

while simulation_running:
    state    = reactor.step(dt, boundary)
    readings = sensors.read_all(state)

    slave.update_input_register("pH_inlet", readings["pH_inlet"].value)
    # ... other sensors

    # Dispatch any pending maintenance command
    result = slave.poll_maintenance()
    if result and not result.success:
        logging.warning("Maintenance failed: %s", result.message)

    # Read actuator setpoints
    boundary.acid_flow_rate      = slave.read_holding_register("acid_flow_rate")
    boundary.chlorine_flow_rate  = slave.read_holding_register("chlorine_flow_rate")

    time.sleep(dt)
```

`poll_maintenance()` is a no-op (returns `None`) when no `MaintenanceManager` is attached or when the trigger coil is `False`, so it is safe to call unconditionally.

### Target IDs and Action Codes

| ID | Target | Valid actions |
|---|---|---|
| 0–2 | pH_inlet / pH_middle / pH_outlet | CALIBRATE, FULL_RESET, CLEAN_WATER, CLEAN_ACID |
| 3–4 | chlorine_inlet / chlorine_outlet | CALIBRATE, FULL_RESET, REPLACE_MEMBRANE (amperometric), REPLACE_REAGENT (DPD) |
| 5 | flow_main | CALIBRATE, FULL_RESET |
| 6–7 | temp_inlet / temp_outlet | CALIBRATE, FULL_RESET |
| 8, 10 | acid_valve / inlet_valve (ControlValve) | RESET_FAULTS, CALIBRATE_ZERO, RECALIBRATE_POSITIONER |
| 9 | chlorine_pump (DosingPump) | RESET_FAULTS, CALIBRATE_ZERO, REPLACE_DIAPHRAGM, REPLACE_CHECK_VALVES, REPLACE_TUBE |

## Edge Cases & Limitations

- **Port 502** may require root/admin on some systems (use >1024 for testing)  
- **Concurrent writes** to same register — last write wins (pymodbus behavior)  
- **No atomic multi-register writes** across boundaries  
- **Float precision** limited to single-precision (IEEE 754)  
- **No response delay** simulation (real devices have 1–50 ms)  
- **Plaintext** — no encryption (real Modbus TCP issue)  
- **Single unit ID** — no multi-slave routing  

## Validation

Run encoding/decoding tests:

```python
from modbus.protocols import validate_encoding
validate_encoding()  # ✓ All encoding/decoding validations passed
```

Register map consistency checked on init.

## Integration with Simulation

Typical pattern:

```python
while simulation_running:
    # Physics + sensors step
    state = reactor.step(dt, boundary)
    readings = sensors.read_all(state)

    # Update Modbus inputs
    slave.update_input_register("pH_inlet", readings["pH_inlet"].value)
    # ... other sensors

    # Read commands
    acid_rate = slave.read_holding_register("acid_flow_rate")
    chlorine_rate = slave.read_holding_register("chlorine_flow_rate")

    # Apply to next physics step
    boundary.acid_flow_rate = acid_rate
    boundary.chlorine_flow_rate = chlorine_rate

    time.sleep(dt)
```

## References

- Modbus Organization: "Modbus Application Protocol Specification V1.1b3" (2006)  
- pymodbus documentation: https://pymodbus.readthedocs.io/  
- Modbus TCP/IP Specification  
- Common water treatment SCADA register maps (AWWA, EPA guidelines)  

For questions, contributions, or bugs:  
**Email:** strukturaenterprise@gmail.com  
**GitHub:** [https://github.com/Guivernoir]

Last Updated: February 20, 2026