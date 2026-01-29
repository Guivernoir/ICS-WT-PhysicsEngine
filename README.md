# Water Treatment Simulator - Physics & Instrumentation Module

**Version:** 1.0.0  
**Status:** Production-Ready  
**Module Type:** Plant Simulation Subsystem  
**Author:** Guilherme F. G. Santos  
**License:** MIT  
**Date:** January 2026

---

## Module Overview

This module provides **complete plant-side simulation** for a water treatment CSTR (Continuous Stirred-Tank Reactor). It simulates everything a real water treatment system does: chemistry, physics, sensors, and industrial communication protocols.

**Purpose:** Provide a high-fidelity plant model for control system development, operator training, and system integration testing.

**Scope of This Module:**
- Rigorous chemical kinetics and transport phenomena (validated against literature)
- Industrial-grade sensor simulation (noise, drift, fouling, faults)
- Modbus/TCP server for external control system integration
- Zero control logic (this is the plant, not the controller)

**What This Module Does NOT Include:**
- Control algorithms (PID, MPC, etc.) - that's a separate module
- Actuator command generation - that's the control layer's job
- HMI/SCADA screens - that's the visualization layer
- Historical data logging - that's the infrastructure layer

This is the **process simulator**. It responds to control commands and returns sensor readings, exactly like a real plant.

---

## System Architecture

### This Module's Role in the Larger System

```
┌──────────────────────────────────────────────────────────────┐
│                    OVERALL SYSTEM                            │
│                                                              │
│  ┌─────────────────────────────────────────────────┐         │
│  │  OPERATOR INTERFACE (Separate Module)           │         │
│  │  - HMI screens                                  │         │
│  │  - Trending and alarms                          │         │
│  │  - Historical data visualization                │         │
│  └────────────┬────────────────────────────────────┘         │
│               │                                              │
│  ┌────────────▼────────────────────────────────────┐         │
│  │  CONTROL LAYER (Separate Module)                │         │
│  │  - PID/MPC controllers                          │         │
│  │  - Setpoint management                          │         │
│  │  - Alarm logic and interlocks                   │         │
│  └────────────┬────────────────────────────────────┘         │
│               │ Commands (dosing rates, flows)               │
│               │                                              │
│  ╔════════════▼════════════════════════════════════╗         │
│  ║  THIS MODULE: PLANT SIMULATION                  ║         │
│  ║                                                 ║         │
│  ║  ┌────────────────────────────────────────────┐ ║         │
│  ║  │  Modbus/TCP Interface                      │ ║         │
│  ║  │  - Register mapping                        │ ║         │
│  ║  │  - Protocol handling                       │ ║         │
│  ║  └───────────┬────────────────────────────────┘ ║         │
│  ║              │                                  ║         │
│  ║  ┌───────────▼────────────────────────────────┐ ║         │
│  ║  │  Sensor Suite (High Fidelity)              │ ║         │
│  ║  │  - Noise, drift, fouling, faults           │ ║         │
│  ║  │  - Sample line delays                      │ ║         │
│  ║  │  - Calibration dynamics                    │ ║         │
│  ║  └───────────┬────────────────────────────────┘ ║         │
│  ║              │                                  ║         │
│  ║  ┌───────────▼───────────────────────────────┐  ║         │
│  ║  │  Physics Engine (Validated)               │  ║         │
│  ║  │  - Multi-zone CSTR                        │  ║         │
│  ║  │  - Chemical kinetics                      │  ║         │
│  ║  │  - Transport phenomena                    │  ║         │
│  ║  │  - Conservation laws                      │  ║         │
│  ║  └───────────────────────────────────────────┘  ║         │
│  ╚═════════════════════════════════════════════════╝         │
│                                                              │
│  ┌─────────────────────────────────────────────────┐         │
│  │  DATA INFRASTRUCTURE (Separate Module)          │         │
│  │  - Time-series database                         │         │
│  │  - Historical logging                           │         │
│  │  - Backup and recovery                          │         │
│  └─────────────────────────────────────────────────┘         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Key Design Principle:** This module simulates the **physical plant only**. It accepts commands via Modbus (like a real plant accepts valve positions) and returns sensor readings (like a real plant's field instruments). All intelligence lives elsewhere.

---

## What This MVP Does (And Doesn't Do)

### ✅ Implemented & Validated

**Physics Engine (`wt_simulator.core`)**
- Multi-zone CSTR (2-20 zones, configurable)
- Rigorous thermodynamics (Arrhenius kinetics, Q₁₀ ~ 1.8)
- Aqueous chemistry (Henderson-Hasselbalch, Newton-Raphson pH solver)
- Turbulent transport (Corrsin correlation, Peclet numbers)
- Spatial stratification (Richardson numbers, buoyancy effects)
- Temperature range: 0-100°C (liquid water at standard pressure)
- pH range: 0-14 (converges for most buffer systems)
- Chlorine: 0-10 mg/L with pH-dependent decay

**Sensor Models (`wt_simulator.sensors`)**
- **pH Sensor:** Glass electrode with Nernst equation, membrane fouling, junction drift
- **Chlorine Sensor:** Amperometric and DPD colorimetric, cross-sensitivities (O₃, H₂O₂, ClO₂)
- **Flow Sensor:** Turbine and magnetic, air bubble detection, vibration effects
- **Temperature Sensor:** RTD and thermocouple, self-heating, cold junction errors
- **Critical Realism Feature:** Sample line transport delays (10-60s typical)

**Modbus/TCP Interface (`wt_simulator.modbus`)**
- Standards-compliant Modbus TCP server
- Complete register map for water treatment
- Input Registers (30001+): Sensor values (read-only)
- Holding Registers (40001+): Actuator setpoints (read/write)
- Coils and Discrete Inputs: Binary I/O
- IEEE 754 float32 encoding (big-endian)
- Thread-safe, non-blocking operation

**Security & Robustness**
- Bounded memory (deque with maxlen, no unbounded growth)
- Thread-safe sensor operations (RLock protection)
- Input validation (zero-trust on all external data)
- Monotonic time enforcement
- Graceful degradation on sensor faults

### ❌ Not Implemented (Yet)

**Control Systems (Phase 2)**
- PID controllers for pH, chlorine, flow
- Model Predictive Control (MPC)
- Feedforward compensation
- Cascade control loops
- Anti-windup logic
- Setpoint ramping

**Actuator Models (Phase 2)**
- Control valve dynamics (positioner, hysteresis)
- Dosing pump dynamics (pulsation, cavitation)
- Variable frequency drives (VFDs)
- On/off valves with dead time

**Advanced Features (Phase 3)**
- Operator interface (HMI)
- Historical data logging (Parquet/TimescaleDB)
- Fault injection framework
- Multi-reactor networks
- Biological processes (nitrification, biofilms)

---

## Quick Start

### Prerequisites

```bash
python >= 3.10
numpy >= 1.24
scipy >= 1.11
pymodbus >= 3.11
```

**Expected output:**
```
Running Physics Engine Validation Suite
======================================================================

1. Thermodynamics...
✓ All thermodynamic validations passed

2. Chemistry...
✓ All chemistry validations passed

3. Transport...
✓ All transport validations passed

4. Spatial...
✓ All spatial validations passed

5. Integrated Reactor...
✓ All integrated reactor validations passed

======================================================================
ALL VALIDATIONS PASSED ✓
Physics engine verified for correctness.
======================================================================
```

### Running the Simulator

```bash
# Start simulation with Modbus server
python -m wt_simulator --port 5020 --dt 1.0

# Run without Modbus (testing mode)
python -m wt_simulator --no-modbus --dt 1.0 --duration 300
```

**Command-line options:**
- `--port`: Modbus TCP port (default: 5020)
- `--host`: Bind address (default: 127.0.0.1)
- `--dt`: Simulation timestep in seconds (default: 1.0)
- `--duration`: Total runtime in seconds (default: infinite)
- `--verbose`: Enable detailed sensor warnings
- `--no-modbus`: Run without Modbus server

---

## Connecting a SCADA System

### Modbus Register Map

**Input Registers (Read-Only Sensors):**
```
30001-30002: pH_inlet        (float32)
30003-30004: pH_middle       (float32)
30005-30006: pH_outlet       (float32)
30007-30008: chlorine_inlet  (float32, mg/L)
30009-30010: chlorine_outlet (float32, mg/L)
30011-30012: flow_rate       (float32, L/min)
30013-30014: temp_inlet      (float32, °C)
30015-30016: temp_outlet     (float32, °C)
30101-30102: simulation_time (float32, seconds)
30103:       system_status   (uint16, 0=OK, >0=fault)
```

**Holding Registers (Read/Write Actuators):**
```
40001-40002: acid_flow_rate      (float32, L/min)
40003-40004: chlorine_flow_rate  (float32, L/min)
40005-40006: inlet_flow_rate     (float32, L/min)
40011-40012: acid_concentration  (float32, mol/L)
40013-40014: chlorine_concentration (float32, mg/L)
```

**Discrete Inputs (Read-Only Status Bits):**
```
10001: sensor_fault_pH_inlet
10002: sensor_fault_pH_outlet
10003: sensor_fault_chlorine
```

### Example: Python Modbus Client

```python
from pymodbus.client import ModbusTcpClient

client = ModbusTcpClient('127.0.0.1', port=5020)
client.connect()

# Read pH outlet (registers 30005-30006)
result = client.read_input_registers(4, 2, slave=1)
if not result.isError():
    # Decode IEEE 754 float32 (big-endian)
    import struct
    raw = struct.pack('>HH', result.registers[0], result.registers[1])
    pH_outlet = struct.unpack('>f', raw)[0]
    print(f"pH outlet: {pH_outlet:.2f}")

# Write acid dosing command (registers 40001-40002)
acid_rate = 0.5  # L/min
raw = struct.pack('>f', acid_rate)
high, low = struct.unpack('>HH', raw)
client.write_registers(0, [high, low], slave=1)

client.close()
```

---

## Physics Engine Deep Dive

### Validation Status

All physics modules pass comprehensive validation against:
- **Thermodynamics:** Arrhenius equation, Q₁₀ coefficients (1.5-2.5 range verified)
- **Chemistry:** Henderson-Hasselbalch equation, charge balance convergence
- **Transport:** Mass conservation (row sums ≈ 0 within 1e-12), Peclet numbers
- **Spatial:** Density anomaly at 4°C (correctly modeled)
- **Reactor:** Energy balance closure, chlorine mass balance over time

**Reference Validation:**
```
Chlorine decay at 20°C: k = 0.0001 s⁻¹ (EPA literature value)
Chlorine decay at 30°C: k = 0.00018 s⁻¹ (Q₁₀ = 1.8, verified)
Water ionization at 25°C: Kw = 1.0e-14 (CRC Handbook)
Carbonate pKa1 at 25°C: 6.35 (Stumm & Morgan)
Neutral pH at 0°C: 7.47 (verified against Van't Hoff)
```

### Key Assumptions & Limitations

**Valid Operating Envelope:**
- Temperature: 0-100°C (liquid water, standard pressure)
- pH: 0-14 (Newton-Raphson converges for most buffers)
- Flow rate: 0-1000 L/min (batch mode supported with Q=0)
- Chlorine: 0-10 mg/L (typical water treatment range)
- No biological processes (no nitrification, no biofilms)
- No dissolved gases (no O₂, CO₂ exchange with atmosphere)
- No particle settling or filtration

**Numerical Methods:**
- ODE integration: scipy.integrate.solve_ivp (Radau method, stiff-stable)
- pH calculation: Newton-Raphson with charge balance (converges in <100 iterations)
- Mixing time: Corrsin correlation (validated for Re > 4000)

**Edge Cases Handled:**
- Batch mode (flow_rate = 0): Mixing still occurs via impeller
- Extreme pH (<2, >12): Reduced tolerance to 1e-8 for convergence
- High stratification (Ri > 10): Mixing suppression up to 50%
- Very low flow (<0.1 L/min): Treated as batch mode with diffusion

---

## Sensor Realism: What Makes This Different

### Why Sample Line Delays Matter

**Problem in typical simulators:**
```
┌────────────┐     ┌────────┐
│  Physics   │ --> │ Sensor │ --> Instantaneous reading
│  (pH=7.0)  │     │        │     pH_measured = 7.0
└────────────┘     └────────┘
```

**Reality in water treatment plants:**
```
┌────────────┐     ┌─────────────┐     ┌────────┐
│  Physics   │ --> │ Sample Line │ --> │ Sensor │
│  (pH=7.0   │     │ (10-60s     │     │        │
│   at t=0)  │     │  transport) │     │        │
└────────────┘     └─────────────┘     └────────┘
                          ↓
                    pH_measured = 7.0 at t=30s
                    (30 second delay)
```

**Impact on control:**
- PID tuning completely different with delays
- Phase lag causes instability
- Dead time compensation required
- Smith predictor becomes necessary

**This simulator models sample line delays correctly.** Every sensor can have a `SampleLine` object with configurable volume and flow rate. The delay is physically calculated, not guessed.

### Sensor Fault Modes

Real sensors fail in specific ways. This simulator models:

**pH Sensor:**
- Open circuit (wire disconnected) → NaN reading
- Glass membrane fouling (biofilm) → Slow response, offset
- Junction contamination → Drift
- Calibration expiration → Warning flag

**Chlorine Sensor:**
- Membrane fouling (amperometric) → Reduced sensitivity
- Reagent degradation (DPD) → Low readings
- Cross-sensitivity to O₃, H₂O₂, ClO₂ → False high readings
- Air bubbles → Intermittent dropouts

**Flow Sensor:**
- Air bubbles → Zero reading (most common fault)
- Low conductivity (magnetic) → Cannot measure
- Bearing wear (turbine) → Increased friction, threshold
- Vibration → Mechanical noise

**Temperature Sensor:**
- Self-heating (RTD) → Reads high
- Cold junction drift (thermocouple) → Offset
- Stem conduction → Error proportional to ambient difference

**Why this matters:** Control algorithms must handle sensor faults gracefully. This simulator lets you test fault tolerance without breaking real hardware.

---

## Modbus Integration Best Practices

### Zero-Trust Input Validation

The simulator validates **all** Modbus writes before applying them to physics:

```python
def validate_flow_rate(value: float, max_value: float = 20.0) -> float:
    """Validate and clamp flow rate within safe bounds."""
    if not isinstance(value, (int, float)):
        return 0.0
    if value != value:  # Check for NaN
        return 0.0
    return max(0.0, min(float(value), max_value))
```

**What this prevents:**
- NaN injection attacks
- Integer overflow
- Negative flows (physically impossible)
- Flows exceeding equipment limits

**Defense-in-depth:** Validation occurs at three layers:
1. Modbus protocol layer (pymodbus)
2. Simulator input layer (validate_flow_rate)
3. Physics engine (bounds checking in ODE solver)

### Thread Safety

All Modbus register operations are thread-safe:
- `update_input_register()`: Protected by RLock
- `read_holding_register()`: Protected by RLock
- Data blocks: pymodbus internal locking

**Why this matters:** Real SCADA systems hammer Modbus servers with concurrent requests. This simulator won't deadlock or corrupt state.

---

## Development Roadmap

### Phase 1: MVP (Current - Completed)

- [x] Physics engine with full validation
- [x] Sensor suite with realistic faults
- [x] Modbus/TCP server
- [x] Security hardening (bounded memory, thread-safe)
- [x] Documentation (this README, module READMEs)

### Phase 2: Control Systems (Q1 2026)

- [ ] PID controller module (`wt_simulator.control.pid`)
- [ ] MPC controller module (`wt_simulator.control.mpc`)
- [ ] Feedforward compensation
- [ ] Actuator dynamics (valves, pumps)
- [ ] Control tuning utilities (Ziegler-Nichols, Lambda)
- [ ] Integration tests (closed-loop stability)

### Phase 3: Advanced Features (Q2 2026)

- [ ] Web-based HMI (React + Flask backend)
- [ ] Historical data logging (Parquet format)
- [ ] Trending and visualization (Plotly/Grafana)
- [ ] Fault injection framework (scripted scenarios)
- [ ] Multi-reactor networks (distributed systems)
- [ ] Advanced chemistry (nitrification, denitrification)

### Phase 4: Production Hardening (Q3 2026)

- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Load testing (1000+ Modbus clients)
- [ ] Security audit (OWASP Top 10)
- [ ] OPC UA server (in addition to Modbus)

---

## Contributing

### Code Style

**This project follows strict engineering discipline:**
- Type hints everywhere (mypy --strict passing)
- Docstrings (Google style)
- No `assert` statements in production code (use explicit validation)
- Bounded memory (no unbounded lists/dicts)
- Thread-safe by default (locks where needed)

**Before submitting PR:**
```bash
# Run all validations
python -m wt_simulator.core
python -m wt_simulator.sensors

# Type checking
mypy src/wt_simulator --strict

# Code formatting
black src/wt_simulator
isort src/wt_simulator
```

### Physics Changes Require Validation

If you modify physics code, you **must**:
1. Update validation functions in the same file
2. Show validation output in PR description
3. Reference literature values (textbook, paper, handbook)
4. Explain physical reasoning (not just "it works")

**Example good PR description:**
```
Modified chlorine decay Arrhenius parameters based on EPA Water Treatment
Handbook (2016 edition, page 342). Changed activation energy from 40 kJ/mol
to 45 kJ/mol to match observed Q₁₀ = 1.8 at 20-30°C range.

Validation output:
✓ k(20°C) = 0.0001 s⁻¹ (EPA reference value)
✓ k(30°C) = 0.00018 s⁻¹ (Q₁₀ = 1.8, within ±5%)
✓ All other thermodynamic validations passed
```

### Sensor Changes Require Realism Check

If you add/modify sensor models:
1. Cite manufacturer datasheet or instrumentation reference
2. Show typical fault modes (not just ideal operation)
3. Include installation effects (flow, vibration, temperature)
4. Demonstrate sample line delays if applicable

---

## Known Issues & Limitations

### Current Limitations

**Physics Engine:**
- No biological processes (nitrification, biofilms)
- No gas exchange (O₂, CO₂ dissolution)
- No particle dynamics (settling, filtration)
- Temperature limited to 0-100°C (liquid water at standard pressure)
- Single-phase only (no ice, no vapor)

**Sensor Models:**
- No electromagnetic interference (EMI) modeling
- No cable capacitance effects
- No ground loop simulation
- Sample line heat transfer simplified (exponential model)

**Modbus:**
- Single unit ID only (no multi-slave routing)
- No Modbus RTU/serial support
- No authentication or encryption
- No diagnostics counters (bad CRCs, timeouts)

**Performance:**
- Physics timestep must be ≥0.1s for numerical stability
- Maximum 20 zones before performance degradation
- No GPU acceleration (CPU-only integration)

### Reporting Bugs

**Good bug report:**
```
Title: pH calculation diverges at high alkalinity

Description:
Newton-Raphson pH solver fails to converge when alkalinity > 500 mg/L as CaCO₃.
Exceeds MAX_ITERATIONS (100) and raises RuntimeError.

Reproduction:
buffer = BufferSystem(alkalinity=600, total_carbonate=10.0, temperature=20)
chem = AqueousChemistry(buffer)
chem.calculate_pH()  # Raises RuntimeError

Expected: pH calculation should converge or raise specific error
Actual: Generic RuntimeError after 100 iterations

Environment:
Python 3.11.5
numpy 1.24.3
scipy 1.11.2
Ubuntu 22.04
```

**Bad bug report:**
```
"It doesn't work"
"pH is wrong"
"Crashed when I ran it"
```

---

## References

### Textbooks

1. **Fogler, H. S.** (2016). *Elements of Chemical Reaction Engineering* (5th ed.). Prentice Hall.
2. **Levenspiel, O.** (1999). *Chemical Reaction Engineering* (3rd ed.). Wiley.
3. **Stumm, W., & Morgan, J. J.** (1996). *Aquatic Chemistry* (3rd ed.). Wiley-Interscience.
4. **Benjamin, M. M.** (2014). *Water Chemistry* (2nd ed.). Waveland Press.
5. **Bird, R. B., Stewart, W. E., & Lightfoot, E. N.** (2007). *Transport Phenomena* (2nd ed.). Wiley.

### Standards & Handbooks

6. **EPA.** (2006). *Hydraulic Analysis of Water Treatment Tanks*. EPA/600/R-06/070.
7. **AWWA.** (2016). *Water Chlorination/Chloramination Practices and Principles*. Manual M20.
8. **CRC Handbook of Chemistry and Physics.** (2022). 103rd Edition. CRC Press.
9. **ISA RP60.6.** *Installation, Operation, and Maintenance of pH Sensors*.

### Modbus Protocol

10. **Modbus Organization.** (2006). *Modbus Application Protocol Specification V1.1b3*.
11. **pymodbus Documentation.** https://pymodbus.readthedocs.io/

---

## License

MIT License

Copyright (c) 2026 Guilherme F. G. Santos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contact

**Author:** Guilherme F. G. Santos  
**Email:** strukturaenterprise@gmail.com  
**GitHub:** https://github.com/Guivernoir

For bug reports, feature requests, or collaboration inquiries, please open an issue on GitHub or contact via email.

---

**Last Updated:** January 28, 2026  
**Document Version:** 1.0.0
