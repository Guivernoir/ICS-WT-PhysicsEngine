"""
Spatial Modeling Module for Water Treatment Reactor
===================================================

This module implements spatial discretization and stratification effects
in water treatment tanks, including:
- Vertical stratification (density-driven)
- Inlet jet penetration
- Dead zones and short-circuiting
- Temperature and density gradients

THEORETICAL FOUNDATION
=====================

1. Density Stratification:
   ρ(T, C_dissolved) = ρ₀ · [1 - β_T·ΔT + β_C·ΔC]

   Where:
   - β_T: Thermal expansion coefficient ≈ 2.1e-4 /°C for water
   - β_C: Solutal expansion coefficient (varies with solute)

2. Richardson Number (buoyancy vs inertia):
   Ri = (g·Δρ·H) / (ρ₀·u²)

   Ri > 1: Stratification stable
   Ri < 0.1: Mixing dominates

3. Froude Number (inertial vs gravitational):
   Fr = u / sqrt(g·H)

   Fr > 1: Supercritical flow (inertia dominates)
   Fr < 1: Subcritical flow (gravity important)

4. Jet Penetration Depth:
   z_jet/d = 6.2 · Fr_jet

   Where d is jet diameter and Fr_jet is jet Froude number

References:
- Turner "Buoyancy Effects in Fluids" (1973)
- Fischer et al "Mixing in Inland and Coastal Waters" (1979)
- Imteaz & Asaeda "Modeling Stratified Flows" (2000)

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from scipy.interpolate import interp1d

from .thermodynamics import TemperatureDependentKinetics


@dataclass
class StratificationParameters:
    """
    Parameters controlling stratification behavior.

    Attributes:
        enable_thermal_stratification: Include temperature effects
        enable_density_stratification: Include dissolved species effects
        critical_richardson: Ri above which stratification is stable
        mixing_suppression_factor: Reduction in K_exchange when stratified
    """

    enable_thermal_stratification: bool = True
    enable_density_stratification: bool = True
    critical_richardson: float = 0.25  # Typical value
    mixing_suppression_factor: float = 0.5  # 50% reduction when stratified


class SpatialModel:
    """
    Multi-zone spatial model with stratification effects.

    This class handles:
    - Vertical zone discretization
    - Density-driven stratification
    - Inlet jet dynamics
    - Dead zone identification
    - Spatial gradient analysis
    """

    # Physical constants
    G_GRAVITY = 9.81  # [m/s²]
    WATER_DENSITY_20C = 998.2  # [kg/m³]
    THERMAL_EXPANSION_COEFF = 2.1e-4  # [1/°C]

    # Density anomaly coefficient for parabolic correction near 4°C
    # Fitted to CRC Handbook data for maximum density at 4°C
    # α = 0.008 kg/(m³·°C²) provides accurate fit over [0, 8]°C range
    # Reference: CRC Handbook of Chemistry and Physics, 103rd Edition (2022)
    DENSITY_ANOMALY_COEFF = 0.008  # [kg/(m³·°C²)]

    # Dissolved species expansion coefficients [m³/kg]
    SOLUTAL_EXPANSION = {
        "NaCl": 7.0e-4,  # Salt
        "CaCO3": 2.0e-4,  # Hardness
        "Chlorine": 1.0e-5,  # Negligible
    }

    def __init__(
        self,
        n_zones: int,
        height: float,
        stratification_params: Optional[StratificationParameters] = None,
    ):
        """
        Initialize spatial model.

        Args:
            n_zones: Number of vertical zones
            height: Total tank height [m]
            stratification_params: Stratification control parameters
        """
        if n_zones < 2:
            raise ValueError(f"Need at least 2 zones, got {n_zones}")

        self.n_zones = n_zones
        self.height = height
        self.zone_height = height / n_zones

        if stratification_params is None:
            stratification_params = StratificationParameters()
        self.strat_params = stratification_params

        self.thermo = TemperatureDependentKinetics()

        # Zone center elevations (from bottom)
        self.zone_centers = np.array(
            [(i + 0.5) * self.zone_height for i in range(n_zones)]
        )

        # Initialize state arrays
        self.temperatures = np.zeros(n_zones)
        self.densities = np.zeros(n_zones)
        self.mixing_suppression = np.ones(n_zones - 1)  # Between adjacent zones

    def calculate_water_density(
        self, temperature: float, salinity_g_L: float = 0.0
    ) -> float:
        """
        Calculate water density as function of temperature and salinity.

        Uses empirical correlation with parabolic correction near 4°C for
        maximum density anomaly. For T outside [0, 8]°C, uses linear thermal
        expansion model.

        ρ(T, S) = ρ_max + α·(T - 4)² for 0 ≤ T ≤ 8°C
        ρ(T, S) = ρ₀ - β_T·ρ₀·(T - T₀) for T > 8°C

        Plus salinity correction: +0.7 kg/m³ per g/L TDS

        Args:
            temperature: Water temperature [°C]
            salinity_g_L: Total dissolved solids [g/L]

        Returns:
            Water density [kg/m³]

        Example:
            >>> spatial = SpatialModel(5, 2.0)
            >>> rho_4 = spatial.calculate_water_density(4.0, 0.0)
            >>> abs(rho_4 - 999.97) < 0.5  # Maximum density at 4°C
            True
            >>> rho_cold = spatial.calculate_water_density(4.0, 0.0)
            >>> rho_warm = spatial.calculate_water_density(20.0, 0.0)
            >>> rho_cold > rho_warm  # Cold water denser
            True
        """
        # Maximum density at 4°C
        rho_max = 999.97  # kg/m³

        if temperature <= 8.0:
            # Parabolic correction near density maximum
            # Uses class constant DENSITY_ANOMALY_COEFF for accurate fit to CRC data
            # This branch handles 0 ≤ T ≤ 8°C with continuous derivative at T=8
            delta_rho_T = -self.DENSITY_ANOMALY_COEFF * (temperature - 4.0) ** 2
            rho = rho_max + delta_rho_T
        else:
            # Linear thermal expansion for higher temperatures (T > 8°C)
            # Ensures smooth transition from parabolic region
            rho_0 = self.WATER_DENSITY_20C
            T_0 = 20.0
            delta_rho_T = -self.THERMAL_EXPANSION_COEFF * rho_0 * (temperature - T_0)
            rho = rho_0 + delta_rho_T

        # Salinity effect (dissolved species increase density)
        # Approximate: 1 g/L salinity increases density by ~0.7 kg/m³
        delta_rho_S = 0.7 * salinity_g_L

        rho += delta_rho_S

        return rho

    def update_density_profile(
        self,
        temperatures: np.ndarray,
        concentrations: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Update density profile based on temperature and composition.

        Args:
            temperatures: Temperature in each zone [°C]
            concentrations: Dictionary of species concentrations [g/L]

        Returns:
            Density in each zone [kg/m³]
        """
        if len(temperatures) != self.n_zones:
            raise ValueError(
                f"Expected {self.n_zones} temperatures, got {len(temperatures)}"
            )

        self.temperatures = temperatures.copy()

        # Calculate density for each zone
        densities = np.zeros(self.n_zones)

        for i in range(self.n_zones):
            # Total dissolved solids (approximate from concentrations)
            TDS = 0.0
            if concentrations:
                TDS = sum(
                    concentrations.get(species, np.zeros(self.n_zones))[i]
                    for species in concentrations
                )

            densities[i] = self.calculate_water_density(temperatures[i], TDS)

        self.densities = densities

        return densities

    def calculate_richardson_number(
        self, zone_idx: int, velocity_scale: float
    ) -> float:
        """
        Calculate Richardson number between adjacent zones.

        Ri = (g·Δρ·Δz) / (ρ₀·u²)

        Args:
            zone_idx: Index of interface (between zone_idx and zone_idx+1)
            velocity_scale: Characteristic velocity [m/s]

        Returns:
            Richardson number (dimensionless)

        Example:
            >>> spatial = SpatialModel(5, 2.0)
            >>> temps = np.array([25, 24, 22, 20, 19])  # Hot on top
            >>> spatial.update_density_profile(temps)
            >>> Ri = spatial.calculate_richardson_number(0, 0.01)
            >>> Ri > 0  # Stable stratification (hot on top)
            True
        """
        if zone_idx < 0 or zone_idx >= self.n_zones - 1:
            raise ValueError(f"Invalid zone index for interface: {zone_idx}")

        # Density difference across interface
        delta_rho = self.densities[zone_idx + 1] - self.densities[zone_idx]
        rho_avg = 0.5 * (self.densities[zone_idx] + self.densities[zone_idx + 1])

        # Richardson number
        if velocity_scale > 1e-6:
            Ri = (self.G_GRAVITY * delta_rho * self.zone_height) / (
                rho_avg * velocity_scale**2
            )
        else:
            Ri = float("inf")  # No flow → infinite Ri

        return Ri

    def is_stratification_stable(self, zone_idx: int, velocity_scale: float) -> bool:
        """
        Determine if stratification is stable at interface.

        Args:
            zone_idx: Interface index
            velocity_scale: Characteristic velocity [m/s]

        Returns:
            True if stratification stable
        """
        Ri = self.calculate_richardson_number(zone_idx, velocity_scale)

        # Stratification stable if Ri > critical value
        return Ri > self.strat_params.critical_richardson

    def calculate_mixing_suppression(self, velocity_scale: float) -> np.ndarray:
        """
        Calculate reduction in inter-zone mixing due to stratification.

        When stratification is stable (Ri > Ri_crit), mixing is suppressed.

        Args:
            velocity_scale: Characteristic velocity [m/s]

        Returns:
            Suppression factors for each interface [0-1]
            1.0 = full mixing, 0.0 = no mixing
        """
        suppression = np.ones(self.n_zones - 1)

        if not self.strat_params.enable_thermal_stratification:
            return suppression

        for i in range(self.n_zones - 1):
            if self.is_stratification_stable(i, velocity_scale):
                # Reduce mixing when stratified
                suppression[i] = self.strat_params.mixing_suppression_factor

        self.mixing_suppression = suppression

        return suppression

    def calculate_brunt_vaisala_frequency(self, zone_idx: int) -> float:
        """
        Calculate Brunt-Väisälä (buoyancy) frequency.

        N² = -(g/ρ₀) · (dρ/dz)

        This characterizes the strength of stratification.
        N > 0: Stable stratification
        N² < 0: Unstable stratification (convection)

        Args:
            zone_idx: Zone index for gradient calculation

        Returns:
            N² [1/s²]
        """
        if zone_idx < 0 or zone_idx >= self.n_zones - 1:
            return 0.0

        # Density gradient
        drho_dz = (
            self.densities[zone_idx + 1] - self.densities[zone_idx]
        ) / self.zone_height

        rho_avg = 0.5 * (self.densities[zone_idx] + self.densities[zone_idx + 1])

        N_squared = -(self.G_GRAVITY / rho_avg) * drho_dz

        return N_squared

    def identify_thermocline(self) -> Optional[float]:
        """
        Identify thermocline depth (maximum temperature gradient).

        Returns:
            Depth of thermocline [m] from top, or None if no thermocline
        """
        if not self.strat_params.enable_thermal_stratification:
            return None

        # Calculate temperature gradients
        temp_gradients = np.zeros(self.n_zones - 1)
        for i in range(self.n_zones - 1):
            temp_gradients[i] = (
                abs(self.temperatures[i + 1] - self.temperatures[i]) / self.zone_height
            )

        # Find maximum gradient
        max_grad_idx = np.argmax(temp_gradients)
        max_grad = temp_gradients[max_grad_idx]

        # Only report if gradient is significant (>0.5°C/m)
        if max_grad > 0.5:
            # Depth from top of tank
            thermocline_depth = self.height - self.zone_centers[max_grad_idx]
            return thermocline_depth
        else:
            return None

    def calculate_inlet_jet_penetration(
        self, inlet_velocity: float, inlet_diameter: float, inlet_zone: int = 0
    ) -> float:
        """
        Calculate jet penetration depth from inlet.

        Uses empirical correlation:
        z_jet = 6.2 · d · Fr_jet

        Where Fr_jet = u / sqrt(g·d)

        Args:
            inlet_velocity: Jet velocity [m/s]
            inlet_diameter: Jet diameter [m]
            inlet_zone: Zone where inlet enters (default: 0 = bottom)

        Returns:
            Penetration depth [m]
        """
        # Jet Froude number
        Fr_jet = inlet_velocity / np.sqrt(self.G_GRAVITY * inlet_diameter)

        # Penetration depth
        z_jet = 6.2 * inlet_diameter * Fr_jet

        # Limit to tank height
        z_jet = min(z_jet, self.height)

        return z_jet

    def estimate_dead_zones(
        self,
        velocity_field: Optional[np.ndarray] = None,
        threshold_velocity: float = 0.001,
    ) -> List[int]:
        """
        Identify potential dead zones (low velocity regions).

        Dead zones have poor mixing and accumulate contaminants.

        Args:
            velocity_field: Velocity magnitude in each zone [m/s]
            threshold_velocity: Velocity below which zone is "dead" [m/s]

        Returns:
            List of zone indices that are potential dead zones
        """
        if velocity_field is None:
            # Can't identify without velocity information
            return []

        dead_zones = []

        for i, vel in enumerate(velocity_field):
            if vel < threshold_velocity:
                dead_zones.append(i)

        return dead_zones

    def calculate_spatial_gradients(
        self, parameter: np.ndarray, parameter_name: str = "parameter"
    ) -> Dict[str, float]:
        """
        Calculate spatial gradient statistics for a parameter.

        Useful for detecting:
        - Poor mixing (large gradients)
        - Sensor location bias
        - Process abnormalities

        Args:
            parameter: Value in each zone
            parameter_name: Name for diagnostics

        Returns:
            Dictionary with gradient statistics
        """
        if len(parameter) != self.n_zones:
            raise ValueError(f"Expected {self.n_zones} values, got {len(parameter)}")

        # Calculate gradients between adjacent zones
        gradients = np.diff(parameter) / self.zone_height

        stats = {
            "mean_value": np.mean(parameter),
            "std_value": np.std(parameter),
            "max_value": np.max(parameter),
            "min_value": np.min(parameter),
            "range": np.max(parameter) - np.min(parameter),
            "max_gradient": np.max(np.abs(gradients)),
            "mean_gradient": np.mean(np.abs(gradients)),
            "gradient_location": int(
                np.argmax(np.abs(gradients))
            ),  # Zone with max gradient
        }

        return stats

    def interpolate_to_depth(
        self, parameter: np.ndarray, depth_from_top: float
    ) -> float:
        """
        Interpolate parameter value at arbitrary depth.

        Useful for sensor placement analysis.

        Args:
            parameter: Value in each zone
            depth_from_top: Depth from top of tank [m]

        Returns:
            Interpolated value at specified depth
        """
        if len(parameter) != self.n_zones:
            raise ValueError(f"Expected {self.n_zones} values, got {len(parameter)}")

        if depth_from_top < 0 or depth_from_top > self.height:
            raise ValueError(f"Depth {depth_from_top}m outside tank [0, {self.height}]")

        # Convert depth from top to elevation from bottom
        elevation = self.height - depth_from_top

        # Interpolate
        interpolator = interp1d(
            self.zone_centers, parameter, kind="linear", fill_value="extrapolate"
        )

        return float(interpolator(elevation))

    def print_spatial_diagnostics(self) -> None:
        """Print detailed spatial diagnostics."""
        print("Spatial Model Diagnostics")
        print("=" * 60)
        print(f"Number of zones: {self.n_zones}")
        print(f"Tank height: {self.height:.2f} m")
        print(f"Zone height: {self.zone_height:.3f} m")
        print()

        print("Temperature Profile:")
        print(
            f"{'Zone':<8} {'Elevation(m)':<15} {'Temp(°C)':<12} {'Density(kg/m³)':<15}"
        )
        print("-" * 60)
        for i in range(self.n_zones):
            print(
                f"{i:<8} {self.zone_centers[i]:<15.3f} {self.temperatures[i]:<12.2f} {self.densities[i]:<15.2f}"
            )

        print()
        print("Stratification Analysis:")
        thermocline = self.identify_thermocline()
        if thermocline:
            print(f"Thermocline depth: {thermocline:.2f} m from top")
        else:
            print("No significant thermocline detected")

        print()
        print("Inter-zone Mixing:")
        print(f"{'Interface':<12} {'N²(1/s²)':<15} {'Mixing Factor':<15}")
        print("-" * 60)
        for i in range(self.n_zones - 1):
            N_sq = self.calculate_brunt_vaisala_frequency(i)
            print(f"{i}-{i+1:<9} {N_sq:<15.6f} {self.mixing_suppression[i]:<15.3f}")

        print("=" * 60)


def validate_spatial() -> None:
    """
    Comprehensive validation of spatial model.

    Tests:
    1. Density calculations (including 4°C anomaly)
    2. Richardson number
    3. Stratification stability
    4. Gradient calculations
    5. Interpolation
    """
    spatial = SpatialModel(n_zones=5, height=2.0)

    # Test 1: Density at 4°C should be maximum (~999.97 kg/m³)
    rho_4 = spatial.calculate_water_density(4.0)
    assert (
        abs(rho_4 - 999.97) < 0.5
    ), f"Density at 4°C should be ~999.97 kg/m³, got {rho_4}"

    # Test 2: Density increases with decreasing temperature (above 4°C)
    rho_cold = spatial.calculate_water_density(5.0)
    rho_warm = spatial.calculate_water_density(20.0)
    assert rho_cold > rho_warm, "Water at 5°C should be denser than at 20°C"

    # Test 3: Density decreases with decreasing temperature (below 4°C - anomalous)
    rho_3 = spatial.calculate_water_density(3.0)
    rho_4_ref = spatial.calculate_water_density(4.0)
    assert rho_3 < rho_4_ref, "Water at 3°C should be less dense than at 4°C (anomaly)"

    # Test 4: Stable stratification (hot on top)
    temps_stable = np.array([25, 23, 21, 19, 17])  # Decreasing with depth
    spatial.update_density_profile(temps_stable)

    Ri = spatial.calculate_richardson_number(0, 0.01)
    assert Ri > 0, "Hot water on top should give positive Ri"

    # Test 5: Unstable stratification (cold on top)
    temps_unstable = np.array([17, 19, 21, 23, 25])  # Increasing with depth
    spatial.update_density_profile(temps_unstable)

    Ri_unstable = spatial.calculate_richardson_number(0, 0.01)
    assert Ri_unstable < 0, "Cold water on top should give negative Ri"

    # Test 6: Gradient calculation
    param = np.array([7.0, 7.1, 7.2, 7.1, 7.0])
    stats = spatial.calculate_spatial_gradients(param, "pH")
    assert abs(stats["mean_value"] - 7.08) < 0.01, "Mean calculation error"

    # Test 7: Interpolation
    value_at_mid = spatial.interpolate_to_depth(param, 1.0)
    assert 7.0 <= value_at_mid <= 7.2, "Interpolated value should be in range"

    print("✓ All spatial validations passed")


if __name__ == "__main__":
    """
    Demonstration of spatial modeling and stratification effects.
    """
    spatial = SpatialModel(
        n_zones=10,
        height=2.0,
        stratification_params=StratificationParameters(
            enable_thermal_stratification=True, critical_richardson=0.25
        ),
    )

    # Simulate temperature profile with thermocline
    # Hot inlet at top, cold bottom
    elevations = spatial.zone_centers
    temperatures = 20.0 + 5.0 * np.tanh((elevations - 1.0) / 0.3)

    spatial.update_density_profile(temperatures)

    # Calculate mixing suppression
    velocity_scale = 0.01  # m/s
    suppression = spatial.calculate_mixing_suppression(velocity_scale)

    # Print diagnostics
    spatial.print_spatial_diagnostics()

    print("\nSpatial Gradient Analysis (pH example):")
    print("-" * 60)
    pH_profile = np.array([7.0, 7.05, 7.1, 7.15, 7.2, 7.25, 7.2, 7.15, 7.1, 7.05])
    pH_stats = spatial.calculate_spatial_gradients(pH_profile, "pH")

    print(f"Mean pH: {pH_stats['mean_value']:.3f}")
    print(f"pH range: {pH_stats['range']:.3f}")
    print(
        f"Max gradient: {pH_stats['max_gradient']:.3f} pH/m at zone {pH_stats['gradient_location']}"
    )
    print()

    # Run validation
    validate_spatial()
