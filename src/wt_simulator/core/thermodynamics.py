"""
Thermodynamics Module for Water Treatment Reactor
==================================================

This module implements temperature-dependent reaction kinetics and
thermodynamic properties critical for accurate water treatment simulation.

THEORETICAL FOUNDATION
=====================

1. Arrhenius Equation for Temperature Dependence:
   k(T) = k₀ * exp(-Eₐ/(R*T))

   Where:
   - k(T): Rate constant at temperature T [K]
   - k₀: Pre-exponential factor [units depend on reaction order]
   - Eₐ: Activation energy [J/mol]
   - R: Universal gas constant = 8.314 J/(mol·K)

2. Van't Hoff Equation for Equilibrium:
   d(ln K)/dT = ΔH°/(R*T²)

3. Temperature Correction for pH:
   Kw(T) = Kw(25°C) * exp[ΔH_w/R * (1/298.15 - 1/T)]

LITERATURE VALUES
================

Chlorine Decay Kinetics (from EPA Water Treatment Handbook):
- k₀ = 2.3e10 [1/s] (pre-exponential factor)
- Eₐ = 45,000 J/mol (activation energy)
- Reference: k(20°C) ≈ 0.0001 s⁻¹

Water Ionization:
- Kw(25°C) = 1.0e-14 [mol²/L²]
- ΔH_w = 55,900 J/mol (heat of ionization)

Carbonate Buffer System:
- pKa1(25°C) = 6.35 (H₂CO₃ ⇌ H⁺ + HCO₃⁻)
- pKa2(25°C) = 10.33 (HCO₃⁻ ⇌ H⁺ + CO₃²⁻)
- Temperature dependence: ΔpKa/ΔT ≈ -0.008 pH/°C

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


# Universal Constants
R_GAS = 8.314  # [J/(mol·K)] Universal gas constant
T_REFERENCE_K = 293.15  # [K] Reference temperature (20°C)
T_REFERENCE_C = 20.0  # [°C] Reference temperature


@dataclass
class ArrheniusParameters:
    """
    Parameters for Arrhenius temperature dependence.

    Attributes:
        k_ref: Rate constant at reference temperature [units vary]
        E_a: Activation energy [J/mol]
        T_ref: Reference temperature [K]
    """

    k_ref: float
    E_a: float
    T_ref: float = T_REFERENCE_K

    def validate(self) -> None:
        """Validate physical consistency of parameters."""
        if self.k_ref <= 0:
            raise ValueError(f"Rate constant must be positive: k_ref={self.k_ref}")
        if self.E_a < 0:
            raise ValueError(f"Activation energy must be non-negative: E_a={self.E_a}")
        if self.T_ref < 273.15 or self.T_ref > 373.15:
            raise ValueError(
                f"Reference temperature out of water range: T_ref={self.T_ref}K"
            )


class TemperatureDependentKinetics:
    """
    Handles all temperature-dependent reaction kinetics and equilibria.

    This class implements rigorous thermodynamic corrections for:
    - Chlorine decay rates (disinfection kinetics)
    - Water ionization constant (pH calculations)
    - Carbonate buffer equilibria
    - Diffusion coefficients (Stokes-Einstein relation)
    """

    # Chlorine Decay Parameters (EPA literature, validated)
    CHLORINE_DECAY = ArrheniusParameters(
        k_ref=0.0001, E_a=45000.0, T_ref=T_REFERENCE_K  # [s⁻¹] at 20°C  # [J/mol]
    )

    # Water Ionization Enthalpy
    DELTA_H_WATER = 55900.0  # [J/mol]
    KW_25C = 1.0e-14  # [mol²/L²]

    # Carbonate System Parameters
    PKA1_25C = 6.35  # H₂CO₃ first dissociation
    PKA2_25C = 10.33  # HCO₃⁻ second dissociation
    DPKA_DT = -0.008  # [pH/°C] Temperature coefficient

    # Diffusion Constants
    # Typical molecular diffusion coefficient for small molecules (Cl₂, HOCl) in water at 20°C
    # Reference: Cussler "Diffusion: Mass Transfer in Fluid Systems" (3rd ed.)
    D_MOLECULAR_REF = 1.0e-9  # [m²/s] at 20°C

    # Temperature bounds for liquid water physics
    T_MIN_C = 0.0  # [°C] Freezing point (standard pressure)
    T_MAX_C = 100.0  # [°C] Boiling point (standard pressure)

    # Validation tolerances
    TOLERANCE_KINETICS = 1e-10  # Absolute tolerance for rate constants
    TOLERANCE_EQUILIBRIUM = 1e-6  # Absolute tolerance for equilibrium constants
    TOLERANCE_PH = 1e-4  # Absolute tolerance for pH values (±0.0001 pH units)

    def __init__(self):
        """Initialize thermodynamic calculator with validated parameters."""
        self.CHLORINE_DECAY.validate()

    @staticmethod
    def celsius_to_kelvin(temp_c: float) -> float:
        """
        Convert Celsius to Kelvin with bounds checking.

        Enforces liquid water temperature range [0, 100]°C at standard pressure.
        For supercooled or pressurized systems, use validate_temperature_extended().

        Args:
            temp_c: Temperature in Celsius

        Returns:
            Temperature in Kelvin

        Raises:
            ValueError: If temperature outside liquid water range [0, 100]°C
        """
        if (
            temp_c < TemperatureDependentKinetics.T_MIN_C
            or temp_c > TemperatureDependentKinetics.T_MAX_C
        ):
            raise ValueError(
                f"Temperature {temp_c}°C outside liquid water range "
                f"[{TemperatureDependentKinetics.T_MIN_C}, {TemperatureDependentKinetics.T_MAX_C}]°C. "
                f"This indicates either:\n"
                f"  1. Invalid input data\n"
                f"  2. Numerical instability in ODE integration (reduce tolerances)\n"
                f"  3. System requires pressurized/supercooled water model"
            )
        return temp_c + 273.15

    def arrhenius_rate(
        self, temp_c: float, params: Optional[ArrheniusParameters] = None
    ) -> float:
        """
        Calculate temperature-corrected rate constant using Arrhenius equation.

        k(T) = k_ref * exp[-E_a/R * (1/T - 1/T_ref)]

        This formulation is numerically stable and physically accurate over
        the typical water treatment temperature range (0-40°C).

        Args:
            temp_c: Operating temperature [°C]
            params: Arrhenius parameters (defaults to chlorine decay)

        Returns:
            Temperature-corrected rate constant [same units as k_ref]

        Example:
            >>> thermo = TemperatureDependentKinetics()
            >>> k_20 = thermo.arrhenius_rate(20.0)  # Returns 0.0001 s⁻¹
            >>> k_30 = thermo.arrhenius_rate(30.0)  # Returns ~0.00018 s⁻¹
            >>> assert k_30 > k_20  # Higher temp → faster decay
        """
        if params is None:
            params = self.CHLORINE_DECAY

        T_K = self.celsius_to_kelvin(temp_c)

        # Arrhenius equation in stable form
        exponent = -(params.E_a / R_GAS) * (1.0 / T_K - 1.0 / params.T_ref)
        k_T = params.k_ref * np.exp(exponent)

        return k_T

    def water_ionization_constant(self, temp_c: float) -> float:
        """
        Calculate temperature-corrected water ionization constant Kw.

        Kw(T) = Kw(25°C) * exp[ΔH_w/R * (1/298.15 - 1/T)]

        This affects pH calculations at non-standard temperatures, as:
        pH + pOH = pKw(T)

        Args:
            temp_c: Operating temperature [°C]

        Returns:
            Kw(T) [mol²/L²]

        Example:
            >>> thermo = TemperatureDependentKinetics()
            >>> Kw_25 = thermo.water_ionization_constant(25.0)
            >>> abs(Kw_25 - 1e-14) < 1e-16  # Should equal reference
            True
            >>> Kw_0 = thermo.water_ionization_constant(0.0)
            >>> Kw_0 < Kw_25  # Colder water has lower Kw
            True
        """
        T_K = self.celsius_to_kelvin(temp_c)
        T_ref_K = 298.15  # 25°C in Kelvin

        # Van't Hoff equation for equilibrium constant
        exponent = (self.DELTA_H_WATER / R_GAS) * (1.0 / T_ref_K - 1.0 / T_K)
        Kw_T = self.KW_25C * np.exp(exponent)

        return Kw_T

    def neutral_pH(self, temp_c: float) -> float:
        """
        Calculate neutral pH at given temperature.

        At neutrality: [H⁺] = [OH⁻] = sqrt(Kw)
        Therefore: pH = -log10(sqrt(Kw)) = 0.5 * pKw

        Args:
            temp_c: Operating temperature [°C]

        Returns:
            Neutral pH at this temperature

        Example:
            >>> thermo = TemperatureDependentKinetics()
            >>> pH_25 = thermo.neutral_pH(25.0)
            >>> abs(pH_25 - 7.0) < 0.01  # ~7.0 at 25°C
            True
            >>> pH_0 = thermo.neutral_pH(0.0)
            >>> pH_0 > pH_25  # Neutral pH increases at low temp
            True
        """
        Kw = self.water_ionization_constant(temp_c)
        pKw = -np.log10(Kw)
        return 0.5 * pKw

    def carbonate_pKa(self, temp_c: float, dissociation: int = 1) -> float:
        """
        Calculate temperature-corrected carbonate system pKa values.

        The carbonate buffer system is crucial for pH stability in water:
        H₂CO₃ ⇌ H⁺ + HCO₃⁻  (pKa1)
        HCO₃⁻ ⇌ H⁺ + CO₃²⁻  (pKa2)

        Temperature dependence is approximately linear over 0-40°C range.

        Args:
            temp_c: Operating temperature [°C]
            dissociation: 1 for first pKa, 2 for second pKa

        Returns:
            pKa value at specified temperature

        Raises:
            ValueError: If dissociation not 1 or 2

        Example:
            >>> thermo = TemperatureDependentKinetics()
            >>> pKa1_25 = thermo.carbonate_pKa(25.0, dissociation=1)
            >>> abs(pKa1_25 - 6.35) < 0.01
            True
        """
        if dissociation not in [1, 2]:
            raise ValueError(f"Dissociation must be 1 or 2, got {dissociation}")

        # Reference values at 25°C
        pKa_ref = self.PKA1_25C if dissociation == 1 else self.PKA2_25C

        # Linear temperature correction (valid for 0-40°C)
        delta_T = temp_c - 25.0
        pKa_T = pKa_ref + self.DPKA_DT * delta_T

        return pKa_T

    def diffusion_coefficient(
        self, temp_c: float, viscosity_ratio: float = 1.0
    ) -> float:
        """
        Calculate temperature-corrected diffusion coefficient.

        Uses Stokes-Einstein relation:
        D(T) = D_ref * (T/T_ref) * (μ_ref/μ(T))

        Where μ is dynamic viscosity, which decreases with temperature.

        For water, viscosity approximately follows:
        μ(T) ≈ μ_ref * exp[1800*(1/T - 1/T_ref)]

        Args:
            temp_c: Operating temperature [°C]
            viscosity_ratio: μ_ref/μ(T) if known (otherwise calculated)

        Returns:
            Diffusion coefficient [m²/s]

        Example:
            >>> thermo = TemperatureDependentKinetics()
            >>> D_20 = thermo.diffusion_coefficient(20.0)
            >>> D_30 = thermo.diffusion_coefficient(30.0)
            >>> D_30 > D_20  # Higher temp → faster diffusion
            True
        """
        T_K = self.celsius_to_kelvin(temp_c)

        if viscosity_ratio == 1.0:
            # Calculate viscosity ratio from temperature
            # Water viscosity model (simplified but accurate for 0-40°C)
            exponent = 1800.0 * (1.0 / T_K - 1.0 / T_REFERENCE_K)
            viscosity_ratio = np.exp(-exponent)  # Note: inverse for D calculation

        # Stokes-Einstein relation
        D_T = self.D_MOLECULAR_REF * (T_K / T_REFERENCE_K) * viscosity_ratio

        return D_T

    def chlorine_decay_rate(self, temp_c: float) -> float:
        """
        Calculate chlorine decay rate at specified temperature.

        Chlorine decay follows first-order kinetics:
        d[Cl]/dt = -k(T) * [Cl]

        This is the primary disinfectant decay mechanism in water treatment.

        Args:
            temp_c: Operating temperature [°C]

        Returns:
            First-order decay rate constant [s⁻¹]

        Example:
            >>> thermo = TemperatureDependentKinetics()
            >>> k_20 = thermo.chlorine_decay_rate(20.0)
            >>> abs(k_20 - 0.0001) < 1e-6  # Should match reference
            True
            >>> k_30 = thermo.chlorine_decay_rate(30.0)
            >>> k_30 / k_20  # Q10 ≈ 1.8 for this activation energy
            1.7...
        """
        return self.arrhenius_rate(temp_c, self.CHLORINE_DECAY)

    def temperature_compensation_factor(
        self, temp_c: float, ref_temp_c: float = T_REFERENCE_C
    ) -> float:
        """
        Calculate multiplicative factor for temperature compensation.

        Useful for converting measurements or predictions between temperatures.

        Args:
            temp_c: Operating temperature [°C]
            ref_temp_c: Reference temperature [°C]

        Returns:
            k(temp_c) / k(ref_temp_c) ratio

        Example:
            >>> thermo = TemperatureDependentKinetics()
            >>> factor = thermo.temperature_compensation_factor(30.0, 20.0)
            >>> factor > 1.0  # Decay faster at higher temperature
            True
        """
        k_operating = self.chlorine_decay_rate(temp_c)
        k_reference = self.chlorine_decay_rate(ref_temp_c)

        return k_operating / k_reference


def validate_thermodynamics() -> None:
    """
    Comprehensive validation of thermodynamic calculations.

    Tests:
    1. Arrhenius equation at reference temperature
    2. Kw temperature dependence
    3. Neutral pH shifts with temperature
    4. Carbonate pKa values
    5. Physical bounds on all parameters
    6. Q10 temperature coefficient
    """
    thermo = TemperatureDependentKinetics()
    tol_kinetics = thermo.TOLERANCE_KINETICS
    tol_eq = thermo.TOLERANCE_EQUILIBRIUM
    tol_ph = thermo.TOLERANCE_PH

    # Test 1: Arrhenius at reference should return k_ref
    k_ref = thermo.chlorine_decay_rate(T_REFERENCE_C)
    assert abs(k_ref - 0.0001) < tol_kinetics, f"k_ref mismatch: {k_ref}"

    # Test 2: Kw at 25°C should be 1e-14
    Kw_25 = thermo.water_ionization_constant(25.0)
    assert abs(Kw_25 - 1e-14) < tol_eq * 1e-14, f"Kw(25°C) mismatch: {Kw_25}"

    # Test 3: Neutral pH at 25°C should be 7.0
    pH_neutral_25 = thermo.neutral_pH(25.0)
    assert abs(pH_neutral_25 - 7.0) < tol_ph, f"pH(25°C) mismatch: {pH_neutral_25}"

    # Test 4: pKa1 at 25°C should be 6.35
    pKa1_25 = thermo.carbonate_pKa(25.0, 1)
    assert abs(pKa1_25 - 6.35) < tol_ph, f"pKa1(25°C) mismatch: {pKa1_25}"

    # Test 5: Temperature monotonicity
    temps = [0, 10, 20, 30, 40]
    k_values = [thermo.chlorine_decay_rate(T) for T in temps]
    assert all(
        k_values[i] < k_values[i + 1] for i in range(len(k_values) - 1)
    ), "Decay rate should increase with temperature"

    # Test 6: Q10 coefficient should be in reasonable range (1.5-2.5)
    k_20 = thermo.chlorine_decay_rate(20.0)
    k_30 = thermo.chlorine_decay_rate(30.0)
    Q10 = k_30 / k_20
    assert (
        1.5 < Q10 < 2.5
    ), f"Q10 = {Q10:.3f} outside expected range [1.5, 2.5] for activation energy {thermo.CHLORINE_DECAY.E_a} J/mol"

    # Test 7: Temperature bounds enforcement
    try:
        thermo.celsius_to_kelvin(-10.0)
        assert False, "Should have raised ValueError for T < 0°C"
    except ValueError:
        pass  # Expected

    try:
        thermo.celsius_to_kelvin(110.0)
        assert False, "Should have raised ValueError for T > 100°C"
    except ValueError:
        pass  # Expected

    print("✓ All thermodynamic validations passed")
    print(f"  - Kinetics tolerance: {tol_kinetics}")
    print(f"  - Equilibrium tolerance: {tol_eq}")
    print(f"  - pH tolerance: {tol_ph}")


if __name__ == "__main__":
    """
    Demonstration of temperature effects on water treatment kinetics.
    """
    thermo = TemperatureDependentKinetics()

    print("Water Treatment Thermodynamics")
    print("=" * 60)
    print()

    # Temperature range for analysis
    temperatures = [0, 5, 10, 15, 20, 25, 30, 35, 40]

    print(f"{'T(°C)':<8} {'k_Cl(s⁻¹)':<12} {'Kw':<12} {'pH_neutral':<12} {'pKa1':<8}")
    print("-" * 60)

    for T in temperatures:
        k_cl = thermo.chlorine_decay_rate(T)
        Kw = thermo.water_ionization_constant(T)
        pH_n = thermo.neutral_pH(T)
        pKa1 = thermo.carbonate_pKa(T, 1)

        print(f"{T:<8.1f} {k_cl:<12.6f} {Kw:<12.2e} {pH_n:<12.3f} {pKa1:<8.3f}")

    print()
    print("Key Observations:")
    print(
        f"  • Chlorine decay rate increases {thermo.chlorine_decay_rate(30)/thermo.chlorine_decay_rate(10):.2f}x from 10°C to 30°C"
    )
    print(
        f"  • Neutral pH shifts from {thermo.neutral_pH(0):.2f} at 0°C to {thermo.neutral_pH(40):.2f} at 40°C"
    )
    print("  • Carbonate buffering capacity varies with temperature")
    print()

    # Run validation
    validate_thermodynamics()
