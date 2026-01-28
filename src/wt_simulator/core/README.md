# Water Treatment Reactor Physics Engine

**Version:** 1.0.0  
**Author:** Guilherme F. G. Santos  
**License:** MIT  
**Date:** January 2026

## Table of Contents

- [Overview](#overview)
- [What This Module Does](#what-this-module-does)
- [What This Module Does NOT Do](#what-this-module-does-not-do)
- [Architecture](#architecture)
- [Capabilities](#capabilities)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Edge Cases & Limitations](#edge-cases--limitations)
- [Validation](#validation)
- [Integration with Control Systems](#integration-with-control-systems)
- [References](#references)

---

## Overview

This is a **pure physics simulation engine** for water treatment reactors, implementing rigorous transport phenomena, chemical equilibria, and reaction kinetics. It models a Continuous Stirred-Tank Reactor (CSTR) with:

- **Multi-zone spatial discretization** (vertical stratification)
- **Temperature-dependent kinetics** (Arrhenius equations)
- **pH buffering and aqueous chemistry** (carbonate system)
- **Chlorine speciation and decay** (disinfection kinetics)
- **Turbulent mixing and transport** (diffusion, advection)
- **Density stratification** (thermal and solutal effects)

### Design Philosophy

**Separation of Concerns:** This module contains ONLY physics. No sensors, no controllers, no setpoints. It accepts **physical boundary conditions** (flows, concentrations, temperatures) and returns **physical state** (pH, chlorine, temperature by zone).

Think of it as: *"Given these actual physical flows entering the tank, what happens inside?"*

---

## What This Module Does

✅ **Physical Transport Phenomena**
- Turbulent mixing between zones (impeller-driven)
- Molecular and turbulent diffusion
- Advective transport with bulk flow
- Density-driven stratification

✅ **Chemical Equilibria**
- pH calculation via charge balance (Newton-Raphson)
- Carbonate buffer system (H₂CO₃/HCO₃⁻/CO₃²⁻)
- Water ionization (temperature-dependent Kw)
- Chlorine speciation (HOCl ⇌ H⁺ + OCl⁻)

✅ **Reaction Kinetics**
- Temperature-dependent chlorine decay (Arrhenius)
- pH-dependent decay rates (HOCl vs OCl⁻)
- First-order disinfection kinetics

✅ **Conservation Laws**
- Mass balance (water, chlorine, alkalinity)
- Energy balance (heat transfer, mixing)
- Charge balance (electroneutrality)

✅ **State Information**
- Query any parameter at any zone at any time
- Access derived quantities (buffering capacity, density, etc.)
- Time-resolved simulation with adaptive ODE integration

---

## What This Module Does NOT Do

❌ **Sensor Models**
- No measurement noise, drift, or calibration errors
- No sensor dynamics (response time, dead bands)
- Pure physics assumes perfect, instantaneous measurements

❌ **Control Systems**
- No PID controllers, MPC, or fuzzy logic
- No setpoints or target tracking
- No control valve dynamics or actuator models

❌ **Optimization**
- No cost functions or objective optimization
- No model predictive control (MPC)
- No reinforcement learning or decision-making

❌ **Fault Detection**
- No anomaly detection or diagnostics
- No fault-tolerant control strategies
- Assumes all equipment functions as specified

❌ **Advanced Phenomena** (unless extended)
- No biological processes (bacteria, biofilms)
- No particle settling or filtration
- No multi-component adsorption
- No dissolved gases (O₂, CO₂ exchange with atmosphere)
- No phase changes (ice, vapor)

---

## Architecture

### Module Structure

```
physics_engine/
├── __init__.py            # Package interface and validation runner
├── thermodynamics.py      # Temperature-dependent kinetics (Arrhenius, Kw, pKa)
├── chemistry.py           # Aqueous chemistry (pH, buffers, speciation)
├── transport.py           # Mixing, diffusion, advection
├── spatial.py             # Stratification, density effects
└── reactor.py             # Integrated CSTR model (ODE system)
```

### Data Flow

```
BoundaryConditions (inputs)
    ↓
IntegratedCSTR.step(dt, boundary)
    ↓
ODE Integration (scipy.integrate.solve_ivp)
    ↓
ReactorState (outputs)
    ↓
User queries: get_state_at_location(zone, param)
```

### Key Classes

**`ReactorConfiguration`**
- Geometry (volume, height, diameter, n_zones)
- Flow (flow_rate, impeller_speed, turbulent_intensity)
- Chemistry (alkalinity, total_carbonate, initial_pH)
- Initial conditions (temperature, chlorine)

**`BoundaryConditions`**
- Inlet stream (flow_rate, pH, chlorine, temperature)
- Dosing streams (acid_flow_rate, chlorine_flow_rate)
- Environmental forcing (ambient_temperature, heat_loss)

**`ReactorState`**
- Primary variables (pH, chlorine, temperature) per zone
- Derived quantities (H⁺ concentration, density, decay rates)

---

## Capabilities

### Spatial Resolution

- **Multi-zone discretization:** 2 to 20+ vertical zones
- **Default:** 5 zones (reasonable for most applications)
- **Trade-off:** More zones = higher accuracy, slower computation

### Temperature Range

- **Liquid water:** 0°C to 100°C (standard pressure)
- **Enforced rigorously:** Exceeding bounds raises ValueError
- **Reason:** Physics equations valid for liquid phase only

### pH Range

- **Typical:** 2 to 12 (water treatment range)
- **Newton-Raphson robust:** Converges for most buffer systems
- **Extreme pH (< 2, > 12):** May require tighter tolerances or activity coefficient corrections

### Flow Modes

- **Continuous:** flow_rate > 0, residence time = V/Q
- **Batch:** flow_rate = 0, residence_time = None (infinite)
- **Semi-batch:** Supported via time-varying BoundaryConditions

### Time Scales

- **Fast processes (ms-s):** Mixing, turbulent diffusion
- **Medium processes (min-hr):** pH equilibration, chlorine decay
- **Slow processes (hr-days):** Temperature equilibration (if heat loss enabled)

### Chlorine Decay

- **Temperature-dependent:** Q₁₀ ≈ 1.8 (realistic for water treatment)
- **pH-dependent:** HOCl decays ~50x faster than OCl⁻
- **First-order kinetics:** d[Cl]/dt = -k(T, pH) · [Cl]

---

## Installation

### Requirements

```bash
numpy >= 1.20
scipy >= 1.7
dataclasses (Python 3.7+)
```

### Setup

```bash
# Clone or copy the physics_engine directory
cd /path/to/your/project

# Import in your code
from physics_engine import IntegratedCSTR, ReactorConfiguration, BoundaryConditions
```

---

## Quick Start

### Minimal Example

```python
from physics_engine import IntegratedCSTR, ReactorConfiguration, BoundaryConditions

# 1. Configure reactor
config = ReactorConfiguration(
    volume=1000.0,        # L
    height=2.0,           # m
    diameter=0.798,       # m (calculated to match volume)
    n_zones=5,            # Vertical discretization
    flow_rate=5.0,        # L/min
    initial_pH=7.0,
    initial_chlorine=2.0, # mg/L
    temperature=20.0      # °C
)

# 2. Create reactor
reactor = IntegratedCSTR(config)

# 3. Define boundary conditions (no dosing, just inlet flow)
boundary = BoundaryConditions(
    inlet_flow_rate=5.0,
    inlet_pH=7.5,
    inlet_chlorine=0.0,
    inlet_temperature=20.0
)

# 4. Simulate for 5 minutes (300 seconds)
dt = 1.0  # Time step [s]
for step in range(300):
    state = reactor.step(dt, boundary)
    
    if step % 60 == 0:  # Print every minute
        outlet_pH = state.pH[-1]  # Last zone = outlet
        outlet_Cl = state.chlorine[-1]
        print(f"t={step}s: pH_outlet={outlet_pH:.3f}, Cl_outlet={outlet_Cl:.3f} mg/L")
```

---

## Usage Examples

### Example 1: Acid Dosing for pH Control

```python
# Simulate acid addition to reduce pH from 8.0 to 7.0
config = ReactorConfiguration(
    volume=1000.0,
    height=2.0,
    n_zones=5,
    flow_rate=5.0,
    initial_pH=8.0,  # Start alkaline
    temperature=20.0
)

reactor = IntegratedCSTR(config)

# Add acid at 0.5 L/min of 0.1 M HCl
boundary = BoundaryConditions(
    inlet_flow_rate=5.0,
    inlet_pH=8.0,
    acid_flow_rate=0.5,        # L/min
    acid_concentration=0.1     # mol/L
)

# Simulate until pH stabilizes
for step in range(600):  # 10 minutes
    state = reactor.step(dt=1.0, boundary=boundary)
    pH_inlet = state.pH[0]
    print(f"t={step}s: pH_inlet={pH_inlet:.3f}")
```

### Example 2: Chlorination with Decay

```python
# Monitor chlorine concentration with temperature-dependent decay
config = ReactorConfiguration(
    volume=1000.0,
    height=2.0,
    n_zones=5,
    flow_rate=5.0,
    initial_chlorine=0.0,  # Start with no chlorine
    temperature=25.0       # Warmer = faster decay
)

reactor = IntegratedCSTR(config)

# Dose chlorine at inlet
boundary = BoundaryConditions(
    inlet_flow_rate=5.0,
    inlet_chlorine=5.0,        # mg/L in feed
    chlorine_flow_rate=0.0     # Or dose separately
)

# Simulate and track chlorine profile
for step in range(1800):  # 30 minutes
    state = reactor.step(dt=1.0, boundary=boundary)
    
    if step % 300 == 0:
        print(f"t={step/60:.1f} min:")
        for i, Cl in enumerate(state.chlorine):
            print(f"  Zone {i}: {Cl:.3f} mg/L")
```

### Example 3: Batch Mode (No Flow)

```python
# Simulate batch chlorine decay
config = ReactorConfiguration(
    volume=1000.0,
    height=2.0,
    n_zones=5,
    flow_rate=0.0,             # BATCH MODE
    initial_chlorine=3.0,
    temperature=20.0
)

reactor = IntegratedCSTR(config)

boundary = BoundaryConditions(
    inlet_flow_rate=0.0        # No flow
)

# Monitor decay over 24 hours
Cl_initial = config.initial_chlorine
for hour in range(24):
    for _ in range(3600):  # 1 hour = 3600 seconds
        state = reactor.step(dt=1.0, boundary=boundary)
    
    Cl_avg = state.chlorine.mean()
    decay_fraction = (Cl_initial - Cl_avg) / Cl_initial
    print(f"Hour {hour+1}: Avg Cl = {Cl_avg:.3f} mg/L ({decay_fraction*100:.1f}% decayed)")
```

### Example 4: Thermal Stratification

```python
# Simulate warm inlet creating stratification
config = ReactorConfiguration(
    volume=1000.0,
    height=2.0,
    n_zones=5,
    flow_rate=5.0,
    temperature=15.0,           # Tank initially cold
    enable_thermal_stratification=True
)

reactor = IntegratedCSTR(config)

# Inlet is 10°C warmer
boundary = BoundaryConditions(
    inlet_flow_rate=5.0,
    inlet_temperature=25.0      # Warm inlet
)

# Monitor temperature profile
for step in range(3600):  # 1 hour
    state = reactor.step(dt=1.0, boundary=boundary)
    
    if step % 600 == 0:
        print(f"t={step/60:.1f} min:")
        for i, T in enumerate(state.temperature):
            height = (i + 0.5) * (config.height / config.n_zones)
            print(f"  z={height:.2f}m: T={T:.2f}°C")
        
        # Calculate Richardson number (stratification strength)
        delta_rho = reactor.spatial.densities[0] - reactor.spatial.densities[-1]
        print(f"  Δρ = {delta_rho:.3f} kg/m³")
```

---

## Edge Cases & Limitations

### 1. Batch Mode (flow_rate = 0)

**Behavior:**
- Residence time is `None` (not infinity)
- Mixing still occurs via impeller
- Outlet zone has no special treatment

**Use Cases:**
- Chlorine contact tanks
- Disinfection chambers
- Chemical mixing vessels

**Limitations:**
- No volume change (assumes perfectly mixed, no evaporation)
- Impeller must be running for mixing

---

### 2. Extreme pH (< 2 or > 12)

**Behavior:**
- Newton-Raphson may require more iterations
- Strong acid/base assumption may break down
- Activity coefficients not accounted for

**Recommendations:**
- Reduce `PH_TOLERANCE` to 1e-8 for tight convergence
- For ionic strength > 0.1 M, consider Debye-Hückel corrections
- Validate chemistry module against analytical solutions

**Example:**
```python
chem = AqueousChemistry(buffer_system)
try:
    pH = chem.calculate_pH(initial_guess=2.0, tolerance=1e-8, max_iter=200)
except RuntimeError as e:
    print(f"pH calculation failed: {e}")
    # Handle convergence failure
```

---

### 3. High Stratification (Richardson Ri > 10)

**Behavior:**
- Mixing suppression may plateau
- Dead zones can form
- Impeller-based correlation may not apply

**Indicators:**
- Temperature gradient > 5°C from bottom to top
- Density difference > 2 kg/m³
- Mixing time >> residence time

**Recommendations:**
- Increase impeller speed or recirculation ratio
- Add more zones (n_zones > 10) for better resolution
- Consider implementing dead zone model

---

### 4. Fast Reactions (Stiff ODEs)

**Behavior:**
- Acid-base equilibration is near-instantaneous (ms timescale)
- Chlorine decay is slow (hours timescale)
- System may be stiff (wide range of time constants)

**Solution:**
- scipy's `solve_ivp` automatically detects stiffness
- Use `method='BDF'` for stiff systems if needed
- Reduce absolute tolerance `atol` to 1e-9 or lower

**Example:**
```python
# In reactor.py, modify step() method:
solution = solve_ivp(
    fun=lambda t, y: self.derivatives(t, y, boundary),
    t_span=(0, dt),
    y0=y_initial,
    method='BDF',        # Backward Differentiation Formula (stiff-stable)
    atol=1e-9,           # Tighter tolerance
    rtol=1e-6
)
```

---

### 5. Temperature Extremes

**0°C (Freezing Point):**
- Ice formation not modeled
- Density maximum at 4°C handled correctly
- Supercooling not supported

**100°C (Boiling Point):**
- Vapor pressure effects not modeled
- No phase change to steam
- For pressurized systems, extend temperature bounds

**Recommendation:** For applications outside [0, 100]°C:
```python
# In thermodynamics.py, modify bounds:
T_MIN_C = -5.0   # Supercooled water
T_MAX_C = 150.0  # Pressurized system

# And implement appropriate density/viscosity correlations
```

---

### 6. Very Low Flow Rates (< 0.1 L/min)

**Behavior:**
- Residence time > 10,000 minutes
- Advection negligible compared to diffusion
- Approaches batch mode

**Numerical Issues:**
- Small Q/V terms may cause roundoff errors
- Peclet number Pe << 1

**Solution:**
- Treat as batch mode if flow_rate < threshold
- Use higher numerical precision if needed

---

### 7. Large Number of Zones (n_zones > 20)

**Behavior:**
- Computational cost scales as O(n²) for mixing matrix
- More accurate spatial resolution
- Finer capture of stratification

**Trade-offs:**
- n_zones = 5: Fast, suitable for well-mixed tanks
- n_zones = 10: Good balance for moderately stratified
- n_zones = 20+: High resolution, slower (consider sparse matrices)

---

### 8. Multi-Component Systems

**Current Limitation:**
- Only chlorine species tracked explicitly
- Other dissolved species affect density but not explicitly tracked

**Extension Strategy:**
```python
# To add tracer or contaminant:
# 1. Add to ReactorState:
#    tracer: np.ndarray = field(default_factory=lambda: np.zeros(5))
# 2. Add to BoundaryConditions:
#    inlet_tracer: float = 0.0
# 3. Add ODE in reactor.derivatives():
#    dTracer_dt = Q/V * (C_inlet - C_zone) + K_matrix @ C_zones
```

---

## Validation

All modules include comprehensive validation functions that verify:

1. **Thermodynamics (`validate_thermodynamics()`)**
   - Arrhenius equation at reference temperature
   - Q₁₀ temperature coefficient (1.5-2.5 range)
   - Water ionization constant Kw(T)
   - Neutral pH shifts with temperature
   - Carbonate pKa temperature dependence

2. **Chemistry (`validate_chemistry()`)**
   - Charge balance convergence (Newton-Raphson)
   - Alpha values sum to 1 (carbonate speciation)
   - pH response to acid/base addition
   - Buffering capacity maximum near pKa
   - Chlorine speciation balance

3. **Transport (`validate_transport()`)**
   - Mass conservation (row sums of K_matrix)
   - Mixing time vs Reynolds number correlation
   - Peclet number calculation
   - Tracer response (dispersion validation)

4. **Spatial (`validate_spatial()`)**
   - Density anomaly at 4°C
   - Temperature continuity at boundaries
   - Richardson number calculation
   - Stratification suppression factors

5. **Reactor (`validate_integrated_reactor()`)**
   - Energy balance closure
   - Chlorine mass balance over time
   - pH dynamics with acid dosing
   - Spatial profiles consistency

### Running Validation

```python
from physics_engine import run_all_validations

# Run all tests
run_all_validations()

# Or run individual modules
from physics_engine import validate_thermodynamics, validate_chemistry
validate_thermodynamics()
validate_chemistry()
```

**Expected Output:**
```
Running Physics Engine Validation Suite
======================================================================

1. Thermodynamics...
✓ All thermodynamic validations passed
  - Kinetics tolerance: 1e-10
  - Equilibrium tolerance: 1e-06
  - pH tolerance: 0.0001

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

---

## Integration with Control Systems

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTROL SYSTEM LAYER                     │
│  - Sensors (with noise, drift, calibration)                 │
│  - Controllers (PID, MPC, fuzzy logic)                      │
│  - Setpoints and objectives                                 │
│  - Actuator dynamics (valves, pumps)                        │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               ▼ (read state)         ▼ (send commands)
┌──────────────────────────────────────────────────────────────┐
│                    INTERFACE LAYER                           │
│  - Convert measurements to true state                        │
│  - Convert control commands to BoundaryConditions            │
└──────────────┬──────────────────────┬────────────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    PHYSICS ENGINE                            │
│  - Pure transport phenomena                                  │
│  - Chemical equilibria                                       │
│  - Reaction kinetics                                         │
│  - Conservation laws                                         │
└──────────────────────────────────────────────────────────────┘
```

## References

### Textbooks

1. **Fogler, H. S.** (2016). *Elements of Chemical Reaction Engineering* (5th ed.). Prentice Hall.
   - Chapter 3: Rate Laws and Stoichiometry
   - Chapter 13: Distributions of Residence Times for Reactors

2. **Levenspiel, O.** (1999). *Chemical Reaction Engineering* (3rd ed.). Wiley.
   - Chapter 5: Ideal Reactors for a Single Reaction
   - Chapter 9: Temperature and Pressure Effects

3. **Stumm, W., & Morgan, J. J.** (1996). *Aquatic Chemistry* (3rd ed.). Wiley-Interscience.
   - Chapter 3: Acid-Base Chemistry
   - Chapter 4: Dissolved Carbon Dioxide

4. **Benjamin, M. M.** (2014). *Water Chemistry* (2nd ed.). Waveland Press.
   - Chapter 3: Acid-Base Equilibria
   - Chapter 9: Disinfection

5. **Bird, R. B., Stewart, W. E., & Lightfoot, E. N.** (2007). *Transport Phenomena* (2nd ed.). Wiley.
   - Chapter 2: Momentum Transport
   - Chapter 17: Diffusion

### Standards & Handbooks

6. **EPA.** (2006). *Hydraulic Analysis of Water Treatment Tanks*. EPA/600/R-06/070.

7. **AWWA.** (2016). *Water Chlorination/Chloramination Practices and Principles*. Manual M20.

8. **CRC Handbook of Chemistry and Physics.** (2022). 103rd Edition. CRC Press.

### Journal Articles

9. **Corrsin, S.** (1957). "Simple theory of an idealized turbulent mixer." *AIChE Journal*, 3(3), 329-330.

10. **Fischer, H. B., et al.** (1979). *Mixing in Inland and Coastal Waters*. Academic Press.

---

## Contact

For questions, bug reports, or contributions:
- **Author:** Guilherme F. G. Santos
- **Email:** [strukturaenterprise@gmail.com]
- **GitHub:** [https://github.com/Guivernoir]

---

**Last Updated:** January 28, 2026