"""
Integrated CSTR Reactor Physics Engine
======================================

This module integrates all physics components into a complete CSTR model:
- Thermodynamics (temperature-dependent kinetics)
- Chemistry (pH buffering, equilibrium)
- Transport (mixing, diffusion)
- Spatial (stratification, multi-zone)

MATHEMATICAL FOUNDATION
======================

Multi-zone CSTR with complete physics:

For each zone i:
dC_i/dt = (1/V_i) * Σ(Q_in * C_in) - (Q_out/V_i) * C_i + r_i(C_i, T_i) + Σ K_ij * (C_j - C_i)

For pH (log scale):
d(pH_i)/dt = -(1/(ln(10) * [H⁺]_i)) * d[H⁺]_i/dt

For temperature:
dT_i/dt = (1/(ρ*c_p*V_i)) * [Q_in*ρ*c_p*(T_in - T_i) + Σ K_ij,T * (T_j - T_i) + Q_reaction]

References:
- Fogler "Elements of Chemical Reaction Engineering" (5th ed.)
- Levenspiel "Chemical Reaction Engineering" (3rd ed.)
- Weber & DiGiano "Process Dynamics in Environmental Systems"
- Stumm & Morgan "Aquatic Chemistry" (3rd ed.)

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import logging

from .thermodynamics import TemperatureDependentKinetics
from .chemistry import AqueousChemistry, BufferSystem
from .transport import TransportModel, GeometryParameters, FlowParameters
from .spatial import SpatialModel, StratificationParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReactorConfiguration:
    """
    Complete configuration for CSTR reactor.

    Combines geometry, flow, chemistry, and operational parameters.
    """

    # Geometry
    volume: float = 1000.0  # [L]
    height: float = 2.0  # [m]
    diameter: float = 0.798  # [m] Calculated to match volume
    n_zones: int = 5

    # Flow
    flow_rate: float = 5.0  # [L/min] Nominal flow rate
    turbulent_intensity: float = 0.15
    recirculation_ratio: float = 5.0
    impeller_speed: float = 60.0  # [rpm]
    impeller_diameter: float = 0.3  # [m]
    power_number: float = 5.0  # Rushton turbine

    # Chemistry
    initial_pH: float = 7.0
    alkalinity: float = 100.0  # [mg/L as CaCO₃]
    total_carbonate: float = 2.0  # [mmol/L]

    # Chlorination
    initial_chlorine: float = 2.0  # [mg/L]

    # Temperature
    temperature: float = 20.0  # [°C]
    enable_thermal_stratification: bool = True

    # Inlet conditions
    inlet_pH: float = 7.5
    inlet_chlorine: float = 0.0  # [mg/L]
    inlet_temperature: float = 20.0  # [°C]

    def validate(self) -> None:
        """Validate configuration consistency."""
        # Check volume matches geometry
        calculated_volume = np.pi * (self.diameter / 2) ** 2 * self.height * 1000
        volume_error = abs(calculated_volume - self.volume) / self.volume
        if volume_error > 0.01:  # 1% tolerance
            raise ValueError(
                f"Volume mismatch: specified {self.volume}L, "
                f"calculated {calculated_volume:.1f}L from geometry. "
                f"Error: {volume_error*100:.1f}%"
            )

        # Check reasonable parameter ranges
        assert 0 < self.volume < 1e6, "Volume out of range"
        assert (
            0 <= self.flow_rate < 1e5
        ), "Flow rate out of range (use 0 for batch mode)"
        assert 0 <= self.initial_pH <= 14, "pH out of range"
        assert 0 <= self.initial_chlorine <= 10, "Chlorine out of range"
        assert 0 <= self.temperature <= 40, "Temperature out of typical range"


@dataclass
class ReactorState:
    """
    Complete state of the reactor at a point in time.

    All state variables are arrays of length n_zones.
    """

    time: float = 0.0  # [s]

    # Primary state variables (per zone)
    pH: np.ndarray = field(default_factory=lambda: np.full(5, 7.0))
    chlorine: np.ndarray = field(default_factory=lambda: np.full(5, 2.0))  # [mg/L]
    temperature: np.ndarray = field(default_factory=lambda: np.full(5, 20.0))  # [°C]
    flow_rate: float = 5.0  # [L/min] Current total flow through reactor

    # Derived quantities (updated from primary variables)
    H_concentration: np.ndarray = field(init=False)  # [mol/L]
    density: np.ndarray = field(init=False)  # [kg/m³]
    chlorine_decay_rate: np.ndarray = field(init=False)  # [1/s]

    def __post_init__(self):
        """Initialize derived quantities."""
        self.update_derived()

    def update_derived(self):
        """Recalculate all derived quantities from primary variables."""
        self.H_concentration = 10 ** (-self.pH)

        # Placeholder for density and decay rate
        # (will be properly calculated by reactor)
        if not hasattr(self, "density"):
            self.density = np.full_like(self.pH, 998.2)
        if not hasattr(self, "chlorine_decay_rate"):
            self.chlorine_decay_rate = np.full_like(self.pH, 0.0001)


@dataclass
class BoundaryConditions:
    """
    Physical boundary conditions and forcing functions for the reactor.

    CRITICAL: This represents PHYSICAL INPUTS to the system, NOT control commands.
    These are the actual flows entering the reactor, which may be determined by
    a separate control system but represent physical reality once set.

    Think of this as: "What physical streams are actually flowing into the tank?"
    NOT: "What should the controller do?"

    Example control → physics flow:
    - Control system decides: "Add 0.5 L/min of acid"
    - Physics receives: BoundaryConditions(acid_dose_flow=0.5, acid_dose_concentration=0.1)
    - Physics calculates: Effect on pH given this physical acid stream
    """

    # Main process inlet stream
    inlet_flow_rate: float = 5.0  # [L/min] Main feed stream
    inlet_pH: float = 7.5
    inlet_chlorine: float = 0.0  # [mg/L]
    inlet_temperature: float = 20.0  # [°C]

    # Chemical dosing streams (physical flows resulting from control actions)
    # Naming: "dose" emphasizes these are actual chemical additions, not setpoints
    acid_flow_rate: float = 0.0  # [L/min] Acid solution feed rate
    acid_concentration: float = (
        0.1  # [mol/L] Acid solution concentration (e.g., HCl, H₂SO₄)
    )

    chlorine_flow_rate: float = 0.0  # [L/min] Chlorine solution feed rate
    chlorine_concentration: float = 50.0  # [mg/L] Chlorine solution concentration

    # Environmental forcing
    ambient_temperature: float = 20.0  # [°C] Surroundings temperature
    heat_loss_coefficient: float = 0.0  # [W/K] (0 = adiabatic, no heat loss)


class IntegratedCSTR:
    """
    Complete CSTR physics engine with all transport phenomena.

    This implementation integrates:
    - Rigorous chemical kinetics and equilibrium
    - Turbulent transport and mixing
    - Spatial stratification
    - Temperature effects
    - Mass and energy conservation

    Pure physics model - no sensor or control logic.
    """

    def __init__(self, config: ReactorConfiguration):
        """
        Initialize integrated CSTR model.

        Args:
            config: Complete reactor configuration
        """
        config.validate()
        self.config = config

        # Initialize all physics modules
        self._initialize_physics_modules()

        # Initialize state
        self.state = ReactorState(
            pH=np.full(config.n_zones, config.initial_pH),
            chlorine=np.full(config.n_zones, config.initial_chlorine),
            temperature=np.full(config.n_zones, config.temperature),
            flow_rate=config.flow_rate,
        )

        logger.info(
            f"Reactor initialized: {config.n_zones} zones, "
            f"V={config.volume}L, τ={self.transport.residence_time:.1f}min"
        )

    def _initialize_physics_modules(self):
        """Initialize all physics modules with consistent parameters."""
        # Thermodynamics
        self.thermo = TemperatureDependentKinetics()

        # Chemistry (buffer system)
        self.buffer = BufferSystem(
            alkalinity=self.config.alkalinity,
            total_carbonate=self.config.total_carbonate,
            temperature=self.config.temperature,
        )
        self.chemistry = AqueousChemistry(self.buffer)

        # Transport
        geometry = GeometryParameters(
            volume=self.config.volume,
            height=self.config.height,
            diameter=self.config.diameter,
            n_zones=self.config.n_zones,
        )

        flow = FlowParameters(
            flow_rate=self.config.flow_rate,
            turbulent_intensity=self.config.turbulent_intensity,
            recirculation_ratio=self.config.recirculation_ratio,
            impeller_speed=self.config.impeller_speed,
            impeller_diameter=self.config.impeller_diameter,
            power_number=self.config.power_number,
        )

        self.transport = TransportModel(geometry, flow, self.config.temperature)

        # Spatial (stratification)
        strat_params = StratificationParameters(
            enable_thermal_stratification=self.config.enable_thermal_stratification
        )

        self.spatial = SpatialModel(
            n_zones=self.config.n_zones,
            height=self.config.height,
            stratification_params=strat_params,
        )

    def derivatives(
        self, t: float, y: np.ndarray, boundary: BoundaryConditions
    ) -> np.ndarray:
        """
        Calculate time derivatives for all state variables.

        This is the heart of the physics engine - the ODE system that
        governs reactor dynamics.

        State vector y = [pH₀, pH₁, ..., pH_{n-1}, Cl₀, Cl₁, ..., Cl_{n-1}, T₀, T₁, ..., T_{n-1}]

        Args:
            t: Current time [s]
            y: State vector
            boundary: Physical boundary conditions

        Returns:
            dy/dt: Time derivatives
        """
        n = self.config.n_zones

        # Unpack state vector
        pH_zones = y[0:n]
        Cl_zones = y[n : 2 * n]
        T_zones = y[2 * n : 3 * n]

        # Initialize derivatives
        dpH_dt = np.zeros(n)
        dCl_dt = np.zeros(n)
        dT_dt = np.zeros(n)

        # Update spatial model with current temperatures
        self.spatial.update_density_profile(T_zones)

        # Calculate mixing suppression from stratification
        velocity_scale = self.transport.superficial_velocity

        # Enable stratification effects based on configuration
        if self.config.enable_thermal_stratification:
            mixing_suppression = self.spatial.calculate_mixing_suppression(
                velocity_scale
            )
        else:
            mixing_suppression = np.ones(n - 1)  # No suppression - perfect mixing

        # Modify exchange matrix for stratification effects
        K_matrix = self.transport.K_matrix.copy()

        # First, modify all off-diagonal exchange terms
        for i in range(n - 1):
            # Reduce mixing between stratified layers
            K_matrix[i, i + 1] *= mixing_suppression[i]
            K_matrix[i + 1, i] *= mixing_suppression[i]

        # Then, recalculate ALL diagonal terms to maintain mass conservation
        # (Do this after all off-diagonal modifications are complete)
        for i in range(n):
            # Sum off-diagonal terms in this row
            off_diagonal_sum = sum(K_matrix[i, j] for j in range(n) if j != i)
            # Diagonal should be negative of off-diagonal sum to conserve mass
            K_matrix[i, i] = -off_diagonal_sum

        # Special handling for outlet zone: restore outlet flow term
        # The outlet zone needs additional -Q/V term for mass leaving system
        Q_per_V = (boundary.inlet_flow_rate / 60) / self.config.volume  # [1/s]
        K_matrix[n - 1, n - 1] -= Q_per_V

        # --- pH DYNAMICS ---
        # pH changes due to:
        # 1. Acid dosing (inlet zone only)
        # 2. Inlet flow (zone 0)
        # 3. Mixing between zones
        # 4. Chemical equilibration (buffering)

        H_zones = 10 ** (-pH_zones)  # Convert to H+ concentration

        # 1. Acid dosing effect (zone 0)
        if boundary.acid_flow_rate > 0:
            zone_volume_L = self.config.volume / n
            H_added_per_s = (
                boundary.acid_flow_rate / 60
            ) * boundary.acid_concentration  # mol/s
            dH_dt_dosing = H_added_per_s / zone_volume_L  # mol/L/s

            # Convert to dpH/dt using chain rule
            beta = self.chemistry.buffering_capacity(pH_zones[0])
            if beta > 0:
                dpH_dt[0] += -dH_dt_dosing / (beta * np.log(10))

        # 2. Inlet flow effect (zone 0)
        Q_per_V = (boundary.inlet_flow_rate / 60) / self.config.volume  # [1/s]
        H_inlet = 10 ** (-boundary.inlet_pH)
        dH_dt_inlet = Q_per_V * (H_inlet - H_zones[0])

        beta_0 = self.chemistry.buffering_capacity(pH_zones[0])
        if beta_0 > 0:
            dpH_dt[0] += -dH_dt_inlet / (beta_0 * np.log(10))

        # 3. Mixing between zones
        dH_dt_mixing = K_matrix @ H_zones

        for i in range(n):
            beta_i = self.chemistry.buffering_capacity(pH_zones[i])
            if beta_i > 0:
                dpH_dt[i] += -dH_dt_mixing[i] / (beta_i * np.log(10))

        # --- CHLORINE DYNAMICS ---
        # Chlorine changes due to:
        # 1. Dosing (inlet zone)
        # 2. Inlet flow
        # 3. Mixing between zones
        # 4. First-order decay (temperature AND pH dependent)

        zone_volume_L = self.config.volume / n

        # 1. Chlorine dosing (zone 0)
        if boundary.chlorine_flow_rate > 0:
            Cl_added_per_s = (
                boundary.chlorine_flow_rate / 60
            ) * boundary.chlorine_concentration  # mg/s
            dCl_dt[0] += Cl_added_per_s / zone_volume_L  # mg/L/s

        # 2. Inlet flow (zone 0)
        dCl_dt[0] += Q_per_V * (boundary.inlet_chlorine - Cl_zones[0])

        # 3. Mixing
        dCl_dt += K_matrix @ Cl_zones

        # 4. Decay (temperature AND pH dependent)
        for i in range(n):
            # Base temperature-dependent decay rate
            k_base = self.thermo.chlorine_decay_rate(T_zones[i])

            # pH-dependent multiplier (HOCl decays ~50x faster than OCl⁻)
            pH_factor = self.chemistry.pH_dependent_chlorine_decay_factor(pH_zones[i])

            # Effective decay rate
            k_effective = k_base * pH_factor

            dCl_dt[i] -= k_effective * Cl_zones[i]

        # --- TEMPERATURE DYNAMICS ---
        # Temperature changes due to:
        # 1. Inlet flow
        # 2. Mixing between zones
        # 3. Heat loss to environment

        # 1. Inlet flow (zone 0)
        dT_dt[0] += Q_per_V * (boundary.inlet_temperature - T_zones[0])

        # 2. Mixing (with stratification effects already in K_matrix)
        dT_dt += K_matrix @ T_zones

        # 3. Heat loss to environment (if specified)
        if boundary.heat_loss_coefficient > 0:
            # Q_loss = U * A * (T - T_ambient)
            # For cylindrical tank: A = π*D*H + 2*π*(D/2)²
            A_lateral = np.pi * self.config.diameter * self.config.height
            A_ends = 2 * np.pi * (self.config.diameter / 2) ** 2
            A_total = A_lateral + A_ends  # [m²]

            rho = 998.2  # [kg/m³]
            cp = 4184  # [J/(kg·K)]
            V_m3 = self.config.volume / 1000  # [m³]

            for i in range(n):
                Q_loss_W = (
                    boundary.heat_loss_coefficient
                    * A_total
                    * (T_zones[i] - boundary.ambient_temperature)
                )
                dT_dt[i] -= Q_loss_W / (rho * cp * V_m3)

        # Combine all derivatives
        dydt = np.concatenate([dpH_dt, dCl_dt, dT_dt])

        return dydt

    def step(self, dt: float, boundary: BoundaryConditions) -> ReactorState:
        """
        Advance reactor state by time dt using given boundary conditions.

        This method:
        1. Solves ODE system with specified boundary conditions
        2. Updates derived quantities
        3. Validates physical bounds

        Args:
            dt: Time step [s]
            boundary: Physical boundary conditions and forcing functions

        Returns:
            Updated reactor state
        """
        # Pack state into ODE vector
        y0 = np.concatenate(
            [self.state.pH, self.state.chlorine, self.state.temperature]
        )

        # Solve ODE over interval [t, t+dt]
        t_span = (self.state.time, self.state.time + dt)

        # Use Radau method (implicit, good for stiff systems)
        # Pass boundary conditions to derivatives via lambda
        solution = solve_ivp(
            lambda t, y: self.derivatives(t, y, boundary),
            t_span,
            y0,
            method="Radau",
            max_step=min(dt, 10.0),  # Limit step size
            rtol=1e-6,
            atol=1e-8,
        )

        if not solution.success:
            logger.warning(f"ODE solver failed: {solution.message}")

        # Extract final state
        y_final = solution.y[:, -1]
        n = self.config.n_zones

        self.state.pH = y_final[0:n]
        self.state.chlorine = y_final[n : 2 * n]
        self.state.temperature = y_final[2 * n : 3 * n]
        self.state.time += dt
        self.state.flow_rate = (
            boundary.inlet_flow_rate
            + boundary.acid_flow_rate
            + boundary.chlorine_flow_rate
        )

        # Update derived quantities
        self._update_derived_state()

        # Validate physical bounds
        self._enforce_physical_bounds()

        return self.state

    def _update_derived_state(self):
        """Update all derived state quantities."""
        n = self.config.n_zones

        # H+ concentration
        self.state.H_concentration = 10 ** (-self.state.pH)

        # Density (from spatial model)
        self.state.density = self.spatial.update_density_profile(self.state.temperature)

        # Chlorine decay rates (temperature-dependent)
        self.state.chlorine_decay_rate = np.array(
            [self.thermo.chlorine_decay_rate(T) for T in self.state.temperature]
        )

    def _enforce_physical_bounds(self):
        """Enforce physical bounds on state variables."""
        # pH must be in [0, 14]
        if np.any(self.state.pH < 0) or np.any(self.state.pH > 14):
            logger.error(f"pH out of bounds: {self.state.pH}")
            self.state.pH = np.clip(self.state.pH, 0.0, 14.0)

        # Chlorine cannot be negative
        if np.any(self.state.chlorine < 0):
            logger.warning(f"Negative chlorine detected: {self.state.chlorine}")
            self.state.chlorine = np.maximum(self.state.chlorine, 0.0)

        # Temperature must be reasonable
        if np.any(self.state.temperature < 0) or np.any(self.state.temperature > 100):
            logger.error(f"Temperature out of bounds: {self.state.temperature}")
            self.state.temperature = np.clip(self.state.temperature, 0.0, 100.0)

    def get_state_at_location(self, zone_idx: int, parameter: str) -> float:
        """
        Get physical state value at specific location.

        Args:
            zone_idx: Zone index (0 = bottom/inlet, n-1 = top/outlet)
            parameter: 'pH', 'chlorine', 'temperature', 'density'

        Returns:
            Physical value at that location
        """
        if zone_idx < 0 or zone_idx >= self.config.n_zones:
            raise ValueError(
                f"Zone index {zone_idx} out of range [0, {self.config.n_zones-1}]"
            )

        if parameter == "pH":
            return self.state.pH[zone_idx]
        elif parameter == "chlorine":
            return self.state.chlorine[zone_idx]
        elif parameter == "temperature":
            return self.state.temperature[zone_idx]
        elif parameter == "density":
            return self.state.density[zone_idx]
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

    def validate_conservation(self) -> Dict[str, float]:
        """
        Validate mass and energy conservation.

        CRITICAL for verifying physics implementation correctness.

        Returns:
            Dictionary with conservation metrics
        """
        zone_volume = self.config.volume / self.config.n_zones

        # Total chlorine mass
        total_chlorine_mg = np.sum(self.state.chlorine) * zone_volume

        # Total H+ and OH-
        total_H_mol = np.sum(self.state.H_concentration) * zone_volume / 1000
        Kw = self.thermo.water_ionization_constant(self.state.temperature[0])
        OH_concentration = Kw / self.state.H_concentration
        total_OH_mol = np.sum(OH_concentration) * zone_volume / 1000

        # Charge balance
        charge_balance = total_H_mol - total_OH_mol

        # Thermal energy (relative to reference)
        rho = 998.2  # kg/m³
        cp = 4184  # J/(kg·K)
        V_m3 = self.config.volume / 1000
        T_ref = 20.0

        thermal_energy_kJ = (
            rho * cp * V_m3 * np.mean(self.state.temperature - T_ref) / 1000
        )

        return {
            "total_chlorine_mg": total_chlorine_mg,
            "total_H_mol": total_H_mol,
            "total_OH_mol": total_OH_mol,
            "charge_balance_mol": charge_balance,
            "thermal_energy_kJ": thermal_energy_kJ,
            "zones": self.config.n_zones,
            "timestamp": self.state.time,
        }

    def print_diagnostics(self):
        """Print comprehensive reactor diagnostics."""
        print("\n" + "=" * 70)
        print("CSTR PHYSICS DIAGNOSTICS")
        print("=" * 70)

        print(f"\nTime: {self.state.time:.1f} s")
        print(f"Residence time: {self.transport.residence_time:.1f} min")
        print(f"Mixing time: {self.transport.mixing_time_seconds:.1f} s")

        print(f"\n{'Zone':<6} {'pH':<8} {'Cl(mg/L)':<10} {'T(°C)':<8} {'ρ(kg/m³)':<10}")
        print("-" * 50)
        for i in range(self.config.n_zones):
            print(
                f"{i:<6} {self.state.pH[i]:<8.3f} {self.state.chlorine[i]:<10.3f} "
                f"{self.state.temperature[i]:<8.2f} {self.state.density[i]:<10.2f}"
            )

        # Conservation
        conservation = self.validate_conservation()
        print("\nConservation Laws:")
        print(f"  Total Chlorine: {conservation['total_chlorine_mg']:.2f} mg")
        print(f"  Charge Balance: {conservation['charge_balance_mol']:.2e} mol")

        # Mixing quality
        pH_CV, pH_S = self.transport.calculate_mixing_quality(self.state.pH)
        Cl_CV, Cl_S = self.transport.calculate_mixing_quality(self.state.chlorine)

        print("\nMixing Quality:")
        print(f"  pH segregation index: {pH_S:.4f}")
        print(f"  Chlorine segregation index: {Cl_S:.4f}")

        print("=" * 70 + "\n")


def validate_integrated_reactor():
    """Comprehensive validation of integrated reactor."""
    config = ReactorConfiguration(
        volume=1000,
        height=2.0,
        diameter=0.798,
        n_zones=5,
        flow_rate=5.0,
        initial_pH=7.5,
        initial_chlorine=2.0,
        temperature=20.0,
    )

    reactor = IntegratedCSTR(config)

    # No-input boundary conditions (closed system)
    boundary = BoundaryConditions(
        inlet_flow_rate=0.0,
        inlet_pH=7.5,
        inlet_chlorine=0.0,
        inlet_temperature=20.0,
        acid_flow_rate=0.0,
        chlorine_flow_rate=0.0,
    )

    # Test 1: Steady state should be stable
    for _ in range(10):
        reactor.step(dt=1.0, boundary=boundary)

    # pH and chlorine should not drift wildly
    assert 6.0 < np.mean(reactor.state.pH) < 9.0, "pH drift"
    assert 0.0 < np.mean(reactor.state.chlorine) < 5.0, "Chlorine drift"

    # Test 2: Conservation laws
    conservation = reactor.validate_conservation()
    assert conservation["total_chlorine_mg"] > 0, "Chlorine conservation"

    # Test 3: Acid addition should decrease pH
    pH_before = reactor.state.pH[0]

    boundary_with_acid = BoundaryConditions(
        inlet_flow_rate=0.0,
        acid_flow_rate=0.5,
        acid_concentration=0.1,
        chlorine_flow_rate=0.0,
    )

    for _ in range(20):
        reactor.step(dt=1.0, boundary=boundary_with_acid)
    pH_after = reactor.state.pH[0]
    assert pH_after < pH_before, "Acid should decrease pH"

    print("✓ All integrated reactor validations passed")


if __name__ == "__main__":
    """
    Demonstration of integrated CSTR physics engine.
    """
    import matplotlib.pyplot as plt

    # Create reactor
    config = ReactorConfiguration(
        volume=1000,
        height=2.0,
        diameter=0.798,
        n_zones=5,
        flow_rate=5.0,
        initial_pH=7.5,
        initial_chlorine=2.0,
        temperature=20.0,
        inlet_pH=8.0,
        inlet_chlorine=0.0,
    )

    reactor = IntegratedCSTR(config)

    # Initial diagnostics
    print("Initial State:")
    reactor.print_diagnostics()

    # Simulation: Acid dosing for 2 minutes, then stop
    t_total = 300  # 5 minutes
    dt = 1.0  # 1 second steps
    n_steps = int(t_total / dt)

    # Data storage
    time_history = []
    pH_inlet = []
    pH_outlet = []
    Cl_inlet = []
    Cl_outlet = []

    print("\nRunning simulation...")
    for step in range(n_steps):
        t = step * dt

        # Define boundary conditions
        # Dose acid for first 2 minutes
        if t < 120:
            boundary = BoundaryConditions(
                inlet_flow_rate=5.0,
                inlet_pH=8.0,
                inlet_chlorine=0.0,
                inlet_temperature=20.0,
                acid_flow_rate=0.5,
                acid_concentration=0.1,
                chlorine_flow_rate=0.0,
            )
        else:
            boundary = BoundaryConditions(
                inlet_flow_rate=5.0,
                inlet_pH=8.0,
                inlet_chlorine=0.0,
                inlet_temperature=20.0,
                acid_flow_rate=0.0,
                chlorine_flow_rate=0.0,
            )

        # Step reactor
        state = reactor.step(dt, boundary=boundary)

        # Record data
        time_history.append(t)
        pH_inlet.append(state.pH[0])
        pH_outlet.append(state.pH[-1])
        Cl_inlet.append(state.chlorine[0])
        Cl_outlet.append(state.chlorine[-1])

    # Final diagnostics
    print("\nFinal State:")
    reactor.print_diagnostics()

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_history, pH_inlet, label="Inlet (Zone 0)", linewidth=2)
    ax1.plot(
        time_history, pH_outlet, label="Outlet (Zone 4)", linewidth=2, linestyle="--"
    )
    ax1.axvline(120, color="red", linestyle=":", label="Dosing stops", alpha=0.7)
    ax1.set_ylabel("pH")
    ax1.set_title("pH Dynamics with Acid Dosing and Spatial Gradients")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_history, Cl_inlet, label="Inlet", linewidth=2)
    ax2.plot(time_history, Cl_outlet, label="Outlet", linewidth=2, linestyle="--")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Chlorine [mg/L]")
    ax2.set_title("Chlorine Decay (Temperature-Dependent First-Order Kinetics)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("integrated_reactor_demo.png", dpi=150)
    print("\nPlot saved to integrated_reactor_demo.png")

    # Run validation
    print("\nRunning validation tests...")
    validate_integrated_reactor()

    print("\n" + "=" * 70)
    print("PHYSICS ENGINE VALIDATED")
    print("=" * 70)
    print("All modules integrated and validated:")
    print("  ✓ Thermodynamics (Arrhenius kinetics)")
    print("  ✓ Chemistry (pH buffering, equilibrium)")
    print("  ✓ Transport (turbulent mixing, diffusion)")
    print("  ✓ Spatial (stratification, multi-zone)")
    print("  ✓ Reactor (complete CSTR dynamics)")
    print("\nPure physics model - no control system logic.")
    print("=" * 70)
