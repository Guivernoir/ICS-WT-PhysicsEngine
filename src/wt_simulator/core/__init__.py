"""
Physics Engine Core Package
===========================

Water treatment reactor physics simulation.

This package provides rigorous implementations of:
- Thermodynamics: Temperature-dependent kinetics and equilibria
- Chemistry: pH buffering, acid-base reactions, chlorine speciation
- Transport: Turbulent mixing, diffusion, advection
- Spatial: Stratification, multi-zone modeling
- Reactor: Integrated CSTR with complete physics

USAGE EXAMPLE
============

```python
from physics_engine.core import IntegratedCSTR, ReactorConfiguration, BoundaryConditions

# Configure reactor
config = ReactorConfiguration(
    volume=1000,  # L
    height=2.0,   # m
    n_zones=5,
    flow_rate=5.0,  # L/min
    initial_pH=7.5,
    initial_chlorine=2.0,
    temperature=20.0
)

# Create reactor
reactor = IntegratedCSTR(config)

# Define boundary conditions (physical inputs)
boundary = BoundaryConditions(
    inlet_flow_rate=5.0,  # L/min
    inlet_pH=7.5,
    inlet_chlorine=0.0,
    inlet_temperature=20.0,
    acid_flow_rate=0.5,  # L/min acid dosing
    acid_concentration=0.1,  # mol/L
    chlorine_flow_rate=0.0
)

# Simulate
for _ in range(300):
    state = reactor.step(dt=1.0, boundary=boundary)

# Access state at any location
pH_outlet = reactor.get_state_at_location(zone_idx=4, parameter='pH')
```

PHYSICS VALIDATION
=================

All modules include comprehensive validation:
- Conservation laws (mass, energy, charge)
- Physical bounds enforcement
- Literature value verification
- Numerical stability checks

PURE PHYSICS ARCHITECTURE
=========================

This is a PURE PHYSICS simulation with NO control system logic.
It models the physical behavior of water treatment processes only.

WHAT THIS MODULE DOES:
- Models physical transport phenomena (mixing, diffusion, advection)
- Calculates chemical equilibria and reaction kinetics
- Enforces conservation laws (mass, energy, charge)
- Provides state information at any spatial location and time
- Accepts boundary conditions (physical flows) as inputs

WHAT THIS MODULE DOES NOT DO:
- NO sensor models (measurement noise, drift, calibration)
- NO control algorithms (PID, MPC, fuzzy logic)
- NO setpoints or target values
- NO actuator dynamics (valve response, pump curves)
- NO fault detection or diagnosis
- NO optimization or decision-making

CONTROL SYSTEM INTERFACE
========================

For control systems, sensors, and actuators, use a SEPARATE LAYER that:

1. **Reads State from Physics Engine:**
   ```python
   true_pH = reactor.get_state_at_location(zone_idx=0, parameter='pH')
   true_chlorine = reactor.get_state_at_location(zone_idx=0, parameter='chlorine')
   ```

2. **Applies Sensor Models (your responsibility):**
   ```python
   measured_pH = true_pH + sensor_noise + sensor_bias
   measured_chlorine = true_chlorine * calibration_factor + drift
   ```

3. **Implements Control Logic (your responsibility):**
   ```python
   pH_error = pH_setpoint - measured_pH
   acid_flow_command = pid_controller.update(pH_error, dt)
   ```

4. **Sends Physical Boundary Conditions Back to Physics Engine:**
   ```python
   boundary = BoundaryConditions(
       inlet_flow_rate=5.0,
       inlet_pH=7.5,
       acid_flow_rate=acid_flow_command,  # From your controller
       acid_concentration=0.1,
       chlorine_flow_rate=chlorine_flow_command  # From your controller
   )
   reactor.step(dt=1.0, boundary=boundary)
   ```

SEPARATION OF CONCERNS
======================

Physics Engine (this module):
- Input: BoundaryConditions (physical flows)
- Output: ReactorState (pH, chlorine, temperature by zone)
- Responsibility: Accurate physical modeling

Control System (your code):
- Input: ReactorState (possibly with sensor models applied)
- Output: BoundaryConditions (control actions)
- Responsibility: Achieving operational objectives

This separation ensures:
1. Physics model is reusable across different control strategies
2. Control system can be tested independently
3. Sensor models can be added/modified without changing physics
4. Clear interfaces between components

NUMERICAL INTEGRATION
=====================

The reactor uses scipy.integrate.solve_ivp for ODE integration:
- Default method: RK45 (Runge-Kutta 4th/5th order, adaptive step)
- Can handle stiff systems with BDF method if needed
- Automatically adjusts time step for numerical stability
- Enforces conservation laws at every step

Typical tolerances (can be adjusted):
- rtol = 1e-6 (relative tolerance)
- atol = 1e-9 (absolute tolerance)

TEMPERATURE BOUNDS
=================

All modules enforce liquid water temperature range [0, 100]°C at standard pressure.
If your application requires:
- Supercooled water: Extend temperature bounds in thermodynamics.py
- Pressurized systems: Implement pressure-dependent phase equilibria

EDGE CASES & LIMITATIONS
========================

1. **Batch Mode (flow_rate = 0):**
   - Supported: Set flow_rate=0 in configuration
   - Residence time is None (not infinity)
   - Mixing still occurs via impeller

2. **Extreme pH (< 2 or > 12):**
   - Newton-Raphson may converge slowly
   - Strong acid/base assumption may break down
   - Consider activity coefficients for high ionic strength

3. **Very High Stratification (Ri > 10):**
   - Mixing suppression may be underestimated
   - Consider adding dead zone modeling
   - Impeller-based correlation may not apply

4. **Fast Reactions:**
   - System may be stiff (use BDF integrator)
   - Reduce integration tolerances if needed
   - Verify mass balance after each step

5. **Temperature Extremes:**
   - 0°C: Ice formation not modeled
   - 100°C: Vapor pressure effects not modeled
   - Supercooling/superheating not supported

VALIDATION STATUS
================

All modules pass comprehensive validation tests:
✓ Thermodynamics: Arrhenius kinetics, Q10 coefficients, Kw temperature dependence
✓ Chemistry: Charge balance convergence, Henderson-Hasselbalch equation
✓ Transport: Mass conservation, Peclet numbers, mixing time correlations
✓ Spatial: Density stratification, Richardson numbers
✓ Reactor: Energy balance, chlorine mass balance, pH dynamics

Run validation: `python -m physics_engine.core` or call `run_all_validations()`

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

# Version
__version__ = "1.0.0"
__author__ = "Guilherme F. G. Santos"

# Core thermodynamics
from .thermodynamics import (
    TemperatureDependentKinetics,
    ArrheniusParameters,
    validate_thermodynamics,
)

# Aqueous chemistry
from .chemistry import AqueousChemistry, BufferSystem, validate_chemistry

# Transport phenomena
from .transport import (
    TransportModel,
    GeometryParameters,
    FlowParameters,
    validate_transport,
)

# Spatial modeling
from .spatial import SpatialModel, StratificationParameters, validate_spatial

# Integrated reactor
from .reactor import (
    IntegratedCSTR,
    ReactorConfiguration,
    ReactorState,
    BoundaryConditions,
    validate_integrated_reactor,
)

# Convenience imports
__all__ = [
    # Main reactor class
    "IntegratedCSTR",
    "ReactorConfiguration",
    "ReactorState",
    "BoundaryConditions",
    # Thermodynamics
    "TemperatureDependentKinetics",
    "ArrheniusParameters",
    # Chemistry
    "AqueousChemistry",
    "BufferSystem",
    # Transport
    "TransportModel",
    "GeometryParameters",
    "FlowParameters",
    # Spatial
    "SpatialModel",
    "StratificationParameters",
    # Validation functions
    "validate_thermodynamics",
    "validate_chemistry",
    "validate_transport",
    "validate_spatial",
    "validate_integrated_reactor",
]


def run_all_validations():
    """
    Run all physics validation tests.

    This should be run after any code changes to ensure
    physics correctness is maintained.
    """
    print("Running Physics Engine Validation Suite")
    print("=" * 70)

    print("\n1. Thermodynamics...")
    validate_thermodynamics()

    print("\n2. Chemistry...")
    validate_chemistry()

    print("\n3. Transport...")
    validate_transport()

    print("\n4. Spatial...")
    validate_spatial()

    print("\n5. Integrated Reactor...")
    validate_integrated_reactor()

    print("\n" + "=" * 70)
    print("ALL VALIDATIONS PASSED ✓")
    print("Physics engine verified for correctness.")
    print("=" * 70)


if __name__ == "__main__":
    """Run all validations when package is executed."""
    run_all_validations()
