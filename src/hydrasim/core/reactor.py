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
Date: February 2026
License: MIT
"""

import numpy as np
from typing import Dict
from scipy.integrate import solve_ivp
import logging

from .thermodynamics import TemperatureDependentKinetics
from .chemistry import AqueousChemistry, BufferSystem
from .transport import TransportModel, GeometryParameters, FlowParameters
from .spatial import SpatialModel, StratificationParameters
from .reactor_models import BoundaryConditions, ReactorConfiguration, ReactorState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedCSTR:
    """
    Complete CSTR physics engine with all transport phenomena.

    This implementation integrates:
    - Rigorous chemical kinetics and equilibrium
    - Turbulent transport and mixing
    - Spatial stratification
    - Temperature effects
    - Mass and energy conservation

    Physics model only; sensor and control logic are handled elsewhere.
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
            chloramine=np.full(config.n_zones, config.initial_chloramine),
            ammonia=np.full(config.n_zones, config.initial_ammonia),
            chlorine_demand=np.full(config.n_zones, config.initial_chlorine_demand),
            temperature=np.full(config.n_zones, config.temperature),
            flow_rate=config.flow_rate,
        )

        residence = (
            f"{self.transport.residence_time:.1f}min"
            if self.transport.residence_time is not None
            else "batch"
        )
        logger.info(
            f"Reactor initialized: {config.n_zones} zones, "
            f"V={config.volume}L, τ={residence}"
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
        from .reactor_dynamics import calculate_derivatives

        return calculate_derivatives(self, t, y, boundary)

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
            [
                self.state.pH,
                self.state.chlorine,
                self.state.temperature,
                self.state.ammonia,
                self.state.chloramine,
                self.state.chlorine_demand,
            ]
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
        self.state.ammonia = y_final[3 * n : 4 * n]
        self.state.chloramine = y_final[4 * n : 5 * n]
        self.state.chlorine_demand = y_final[5 * n : 6 * n]
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

        # Combined chlorine cannot be negative
        if np.any(self.state.chloramine < 0):
            logger.warning(f"Negative chloramine detected: {self.state.chloramine}")
            self.state.chloramine = np.maximum(self.state.chloramine, 0.0)

        # Ammonia cannot be negative
        if np.any(self.state.ammonia < 0):
            logger.warning(f"Negative ammonia detected: {self.state.ammonia}")
            self.state.ammonia = np.maximum(self.state.ammonia, 0.0)

        # Chlorine demand precursor cannot be negative
        if np.any(self.state.chlorine_demand < 0):
            logger.warning(
                f"Negative chlorine demand detected: {self.state.chlorine_demand}"
            )
            self.state.chlorine_demand = np.maximum(self.state.chlorine_demand, 0.0)

        # Temperature must be reasonable
        if np.any(self.state.temperature < 0) or np.any(self.state.temperature > 100):
            logger.error(f"Temperature out of bounds: {self.state.temperature}")
            self.state.temperature = np.clip(self.state.temperature, 0.0, 100.0)

    def get_state_at_location(self, zone_idx: int, parameter: str) -> float:
        """
        Get physical state value at specific location.

        Args:
            zone_idx: Zone index (0 = bottom/inlet, n-1 = top/outlet)
            parameter: 'pH', 'chlorine', 'chloramine', 'ammonia', 'chlorine_demand', 'temperature', 'density'

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
        elif parameter == "chloramine":
            return self.state.chloramine[zone_idx]
        elif parameter == "ammonia":
            return self.state.ammonia[zone_idx]
        elif parameter == "chlorine_demand":
            return self.state.chlorine_demand[zone_idx]
        elif parameter == "temperature":
            return self.state.temperature[zone_idx]
        elif parameter == "density":
            return self.state.density[zone_idx]
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

    def validate_conservation(self) -> Dict[str, float]:
        """
        Validate mass and energy conservation.

        Useful for checking mass and energy consistency during runs.

        Returns:
            Dictionary with conservation metrics
        """
        zone_volume = self.config.volume / self.config.n_zones

        # Total chlorine mass
        total_chlorine_mg = np.sum(self.state.chlorine) * zone_volume
        total_chloramine_mg = np.sum(self.state.chloramine) * zone_volume
        total_chlorine_equivalent_mg = total_chlorine_mg + total_chloramine_mg

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
            "total_chloramine_mg": total_chloramine_mg,
            "total_chlorine_equivalent_mg": total_chlorine_equivalent_mg,
            "total_H_mol": total_H_mol,
            "total_OH_mol": total_OH_mol,
            "charge_balance_mol": charge_balance,
            "thermal_energy_kJ": thermal_energy_kJ,
            "zones": self.config.n_zones,
            "timestamp": self.state.time,
        }

    def print_diagnostics(self) -> None:
        from .reactor_diagnostics import print_diagnostics

        print_diagnostics(self)


def validate_integrated_reactor() -> None:
    from .reactor_validation import validate_integrated_reactor as run_validation

    run_validation()


if __name__ == "__main__":
    from .reactor_validation import demo_reactor

    demo_reactor()
