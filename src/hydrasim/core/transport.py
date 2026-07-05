"""
Transport Phenomena Module for Water Treatment Reactor
======================================================

This module implements mass transport mechanisms in a CSTR:
- Turbulent mixing and inter-zone exchange
- Molecular and turbulent diffusion
- Advective transport with flow
- Dispersion modeling

THEORETICAL FOUNDATION
=====================

1. General Transport Equation:
   ∂C/∂t + u·∇C = D·∇²C + r(C)

   Where:
   - C: Concentration
   - u: Velocity field
   - D: Diffusion coefficient (molecular + turbulent)
   - r(C): Reaction term

2. Turbulent Mixing (Compartmental Model):
   J_ij = K_ij · (C_i - C_j)

   Where K_ij is the inter-compartment exchange coefficient

3. Mixing Time (95% homogeneity):
   t_mix = V/(Q_rec · N_passes)

   Typically 60-180 seconds for water treatment tanks

4. Peclet Number (convection vs diffusion):
   Pe = L·u/D

   High Pe (>100): Advection-dominated
   Low Pe (<1): Diffusion-dominated

References:
- Bird, Stewart & Lightfoot "Transport Phenomena" (2nd ed.)
- Levenspiel "Chemical Reaction Engineering" (3rd ed.)
- EPA "Hydraulic Analysis of Water Treatment Tanks"
- Fogler "Elements of Chemical Reaction Engineering" (5th ed.)

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from typing import Tuple

from .thermodynamics import TemperatureDependentKinetics
from .transport_models import FlowParameters, GeometryParameters


class TransportModel:
    """
    Multi-zone transport model for CSTR with turbulent mixing.

    Implements:
    - Advective transport (bulk flow)
    - Turbulent diffusion (inter-zone exchange)
    - Molecular diffusion (minor contribution)
    - Tracer response validation
    """

    # Physical constants
    WATER_VISCOSITY = 1e-6  # [m²/s] Kinematic viscosity at 20°C

    # Mixing correlation constant for Corrsin equation
    # C_mixing ≈ 10-15 for 95% homogeneity in stirred tanks
    # Value of 12.0 is conservative estimate for water treatment applications
    # Reference: Paul, Atiemo-Obeng & Kresta "Handbook of Industrial Mixing" (2004)
    C_MIXING = 12.0  # Dimensionless constant for mixing time correlation

    def __init__(
        self,
        geometry: GeometryParameters,
        flow: FlowParameters,
        temperature: float = 20.0,
    ):
        """
        Initialize transport model.

        Args:
            geometry: Tank geometry parameters
            flow: Flow characteristics
            temperature: Operating temperature [°C]
        """
        geometry.validate()
        flow.validate()

        self.geometry = geometry
        self.flow = flow
        self.temperature = temperature
        self.residence_time: float | None = None
        self.superficial_velocity = 0.0
        self.impeller_tip_speed = 0.0

        # Batch mode flag
        self.is_batch_mode = self.flow.flow_rate == 0.0

        self.thermo = TemperatureDependentKinetics()

        # Calculate transport parameters
        self._calculate_transport_coefficients()

        # Build exchange matrix
        self.K_matrix = self._build_exchange_matrix()

    def _calculate_transport_coefficients(self) -> None:
        """
        Calculate all transport coefficients from geometry and flow.

        Includes:
        - Reynolds number (impeller-based for stirred tanks)
        - Turbulent diffusivity
        - Mixing time (impeller-based correlation)
        - Residence time
        """
        # Hydraulic residence time
        # For batch mode (flow_rate = 0), residence_time is undefined/infinite
        # We set to None to make batch mode explicit
        if self.flow.flow_rate > 0:
            self.residence_time = self.geometry.volume / self.flow.flow_rate  # [min]
        else:
            self.residence_time = None  # Batch mode - no continuous flow

        # Superficial velocity (for reference only)
        Q_m3_s = self.flow.flow_rate / 60000.0  # Convert L/min to m³/s
        self.superficial_velocity = Q_m3_s / self.geometry.cross_sectional_area  # [m/s]

        # Impeller tip speed (actual mixing velocity)
        N_rps = self.flow.impeller_speed / 60.0  # Convert rpm to rps
        D_imp = self.flow.impeller_diameter
        self.impeller_tip_speed = (
            np.pi * D_imp * self.flow.impeller_speed / 60.0
        )  # [m/s]

        # Reynolds number (impeller-based for stirred tanks)
        self.Re = (self.flow.impeller_speed / 60.0) * D_imp**2 / self.WATER_VISCOSITY

        # Turbulent diffusivity from impeller energy dissipation
        # D_turb = C * N * D_imp² where C ≈ 0.1 for Rushton turbines
        self.D_turbulent = 0.1 * N_rps * D_imp**2

        # Molecular diffusion (temperature-dependent)
        self.D_molecular = self.thermo.diffusion_coefficient(self.temperature)

        # Effective diffusivity (turbulent >> molecular in stirred tanks)
        self.D_effective = self.D_turbulent + self.D_molecular

        # Mixing time using Corrsin correlation for stirred tanks
        # t_mix = C * (H/D_imp) / (N * Np^(1/3))
        # where C is the mixing correlation constant (see class constant C_MIXING)
        Np = self.flow.power_number
        self.mixing_time_seconds = (
            self.C_MIXING * (self.geometry.height / D_imp) / (N_rps * Np ** (1.0 / 3.0))
        )  # [s]
        self.mixing_time = self.mixing_time_seconds / 60.0  # [min]

        # Peclet number (advection vs diffusion)
        self.Pe = self.geometry.height * self.superficial_velocity / self.D_effective

    def _build_exchange_matrix(self) -> np.ndarray:
        """
        Build inter-zone exchange coefficient matrix.

        For vertical zones, exchange occurs primarily between adjacent zones.
        Exchange coefficient: K_ij = D_eff * A / Δz

        Additional considerations:
        - Inlet zone receives inflow
        - Outlet zone has outflow
        - Stratification reduces exchange (buoyancy effects)

        Returns:
            K_matrix: [n_zones × n_zones] exchange coefficients [1/s]

        Matrix structure (for n=5):
        [[-k₁  k₁   0    0    0 ]
         [ k₁ -2k₁  k₁   0    0 ]
         [ 0   k₁ -2k₁  k₁   0 ]
         [ 0   0   k₁ -2k₁  k₁]
         [ 0   0   0   k₁  -k₁]]
        """
        n = self.geometry.n_zones

        # Exchange coefficient between adjacent zones
        # K = D_eff * A / Δz [m³/s] / [m³] = [1/s]
        K_exchange = (
            self.D_effective
            * self.geometry.cross_sectional_area
            / self.geometry.zone_height
        )

        # Convert to 1/s by dividing by zone volume (m³)
        zone_volume_m3 = self.geometry.zone_volume / 1000.0
        K_exchange_per_s = K_exchange / zone_volume_m3

        # Build tridiagonal matrix
        K_matrix = np.zeros((n, n))

        for i in range(n):
            # Exchange with zone below (i-1)
            if i > 0:
                K_matrix[i, i - 1] = K_exchange_per_s

            # Exchange with zone above (i+1)
            if i < n - 1:
                K_matrix[i, i + 1] = K_exchange_per_s

        # Enforce strict mass conservation: diagonal = -sum(off-diagonal)
        for i in range(n):
            K_matrix[i, i] = -np.sum(K_matrix[i, :]) + K_matrix[i, i]

        # Add advective transport (flow from zone to zone)
        # Q/V term for advection
        Q_per_V = (self.flow.flow_rate / 60.0) / self.geometry.volume  # [1/s]

        # Inlet enters zone 0
        # No change to K_matrix for inlet (handled separately in reactor)

        # Outlet from zone n-1
        K_matrix[n - 1, n - 1] -= Q_per_V

        # Verify conservation (should be nearly zero, allowing for outlet flow)
        # All rows except outlet should have zero sum
        row_sums = K_matrix.sum(axis=1)
        for i in range(n - 1):  # Check all except outlet zone
            if abs(row_sums[i]) > 1e-12:
                raise ValueError(
                    f"Mass conservation violated in zone {i}: row sum = {row_sums[i]:.2e} "
                    f"(should be < 1e-12)"
                )

        # Outlet zone has negative sum equal to -Q/V (mass leaves system)
        expected_outlet_sum = -Q_per_V
        if abs(row_sums[n - 1] - expected_outlet_sum) > 1e-12:
            raise ValueError(
                f"Outlet mass balance wrong: got {row_sums[n-1]:.2e}, "
                f"expected {expected_outlet_sum:.2e}"
            )

        return K_matrix

    def calculate_mixing_quality(
        self, concentrations: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate mixing quality metrics.

        1. Coefficient of Variation (CV):
           CV = σ / μ

        2. Segregation Index (0 = perfect mixing, 1 = no mixing):
           S = (σ² - σ²_perfect) / (σ²_complete_segregation - σ²_perfect)

        Args:
            concentrations: Concentration in each zone

        Returns:
            (CV, segregation_index)

        Example:
            >>> geom = GeometryParameters(1000, 2.0, 0.9, 5)
            >>> flow = FlowParameters(5.0)
            >>> transport = TransportModel(geom, flow)
            >>> C = np.array([2.0, 1.95, 2.01, 1.98, 2.02])  # Well mixed
            >>> CV, S = transport.calculate_mixing_quality(C)
            >>> CV < 0.05  # Good mixing
            True
        """
        mean_C = np.mean(concentrations)
        std_C = np.std(concentrations)

        # Coefficient of variation
        CV = std_C / mean_C if mean_C > 0 else 0.0

        # Segregation index (simple version)
        # σ²_perfect = 0 (perfect mixing)
        # σ²_complete_seg ≈ mean_C² (all in one zone)
        variance = std_C**2
        variance_perfect = 0.0
        variance_segregated = mean_C**2

        if variance_segregated > variance_perfect:
            S = (variance - variance_perfect) / (variance_segregated - variance_perfect)
            S = np.clip(S, 0.0, 1.0)  # Bound to [0, 1]
        else:
            S = 0.0

        return CV, S

    def tracer_response(
        self, time_points: np.ndarray, tracer_input_mode: str = "pulse"
    ) -> np.ndarray:
        """
        Calculate theoretical tracer response for validation.

        Tracer studies are used to validate mixing models experimentally.

        For ideal CSTR with n tanks in series:
        E(t) = (n/τ)ⁿ · tⁿ⁻¹ / (n-1)! · exp(-n·t/τ)

        Where:
        - τ: Mean residence time
        - n: Number of tanks (= n_zones for our model)

        Args:
            time_points: Time values for response [s]
            tracer_input_mode: 'pulse' or 'step'

        Returns:
            Tracer concentration vs time (normalized)

        Example:
            >>> geom = GeometryParameters(1000, 2.0, 0.9, 5)
            >>> flow = FlowParameters(5.0)
            >>> transport = TransportModel(geom, flow)
            >>> t = np.linspace(0, 600, 100)
            >>> E_t = transport.tracer_response(t, 'pulse')
            >>> np.trapz(E_t, t) - 1.0 < 0.01  # Should integrate to 1
            True
        """
        if self.residence_time is None:
            raise ValueError("tracer response requires continuous-flow mode")
        tau = self.residence_time * 60  # Convert min to s
        n = self.geometry.n_zones

        if tracer_input_mode == "pulse":
            # Tanks-in-series model for pulse input
            # E(t) = (n/τ)ⁿ · tⁿ⁻¹ / (n-1)! · exp(-n·t/τ)

            from scipy.special import factorial

            E_t = np.zeros_like(time_points)
            valid = time_points > 0  # Avoid t=0 issues

            E_t[valid] = (
                (n / tau) ** n
                * time_points[valid] ** (n - 1)
                / factorial(n - 1)
                * np.exp(-n * time_points[valid] / tau)
            )

            return E_t

        elif tracer_input_mode == "step":
            # Step response (F-curve)
            # F(t) = 1 - exp(-n·t/τ) · Σ[(n·t/τ)ⁱ / i!] for i=0 to n-1

            from scipy.special import gammainc

            F_t = 1.0 - gammainc(n, n * time_points / tau)

            return F_t

        else:
            raise ValueError(f"Unknown tracer input mode: {tracer_input_mode}")

    def dispersion_number(self) -> float:
        """
        Calculate dispersion number (D/uL).

        Characterizes deviation from plug flow:
        - D/uL = 0: Plug flow
        - D/uL = ∞: Perfect mixing (CSTR)
        - D/uL ~ 0.01-0.1: Typical for large tanks

        Returns:
            Dispersion number (dimensionless)
        """
        if self.superficial_velocity <= 0.0:
            return float("inf")
        D_over_uL = self.D_effective / (
            self.superficial_velocity * self.geometry.height
        )
        return D_over_uL

    def tanks_in_series_equivalent(self) -> float:
        """
        Calculate equivalent number of CSTRs in series.

        Related to dispersion number by:
        n = 1 / (2 * D/uL) for large n

        Returns:
            Equivalent number of ideal CSTRs
        """
        D_over_uL = self.dispersion_number()

        if D_over_uL > 0:
            n_equiv = 1.0 / (2.0 * D_over_uL)
        else:
            n_equiv = float("inf")  # Plug flow

        return n_equiv

    def print_diagnostics(self) -> None:
        """Print detailed transport diagnostics."""
        print("Transport Model Diagnostics")
        print("=" * 60)
        print(
            f"Reynolds number: {self.Re:.0f} "
            + (
                "(Turbulent)"
                if self.Re > 4000
                else "(Transitional)" if self.Re > 2000 else "(Laminar)"
            )
        )
        residence = (
            f"{self.residence_time:.1f} min"
            if self.residence_time is not None
            else "batch"
        )
        print(f"Residence time: {residence}")
        print(f"Mixing time (95%): {self.mixing_time_seconds:.1f} s")
        print(f"Superficial velocity: {self.superficial_velocity:.4f} m/s")
        print()
        print(f"Molecular diffusivity: {self.D_molecular:.2e} m²/s")
        print(f"Turbulent diffusivity: {self.D_turbulent:.2e} m²/s")
        print(f"Effective diffusivity: {self.D_effective:.2e} m²/s")
        print()
        print(f"Peclet number: {self.Pe:.1f} (Advection/Diffusion)")
        print(f"Dispersion number: {self.dispersion_number():.4f}")
        print(f"Tanks-in-series equivalent: {self.tanks_in_series_equivalent():.1f}")
        print("=" * 60)


def validate_transport() -> None:
    from .transport_validation import validate_transport as run_validation

    run_validation()


if __name__ == "__main__":
    from .transport_validation import demo_transport

    demo_transport()
