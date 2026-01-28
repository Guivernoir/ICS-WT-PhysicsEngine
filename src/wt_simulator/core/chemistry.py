"""
Chemistry Module for Water Treatment Reactor
============================================

This module implements reaction kinetics and chemical equilibrium for
water treatment processes with emphasis on pH buffering and chlorination.

THEORETICAL FOUNDATION
=====================

1. Carbonate Buffer System (Henderson-Hasselbalch):
   pH = pKa + log([A⁻]/[HA])

   For carbonate system:
   H₂CO₃ ⇌ H⁺ + HCO₃⁻    (pKa1 = 6.35 at 25°C)
   HCO₃⁻ ⇌ H⁺ + CO₃²⁻    (pKa2 = 10.33 at 25°C)

2. Charge Balance:
   [H⁺] + Σ[cations] = [OH⁻] + Σ[anions]

3. Mass Balance:
   C_T = [H₂CO₃] + [HCO₃⁻] + [CO₃²⁻]

4. Chlorine Speciation:
   HOCl ⇌ H⁺ + OCl⁻    (pKa = 7.5 at 25°C)
   Free chlorine = [HOCl] + [OCl⁻]

NUMERICAL METHODS
================

pH calculations use Newton-Raphson iteration on charge balance:
f(pH) = [H⁺] - [OH⁻] + [HCO₃⁻] + 2[CO₃²⁻] - alkalinity = 0

Convergence criterion: |Δ pH| < 1e-6

References:
- Stumm & Morgan "Aquatic Chemistry" (3rd ed.)
- Benjamin "Water Chemistry" (2nd ed.)
- Snoeyink & Jenkins "Water Chemistry" (1980)

Author: Guilherme F. G. Santos
Date: January 2026
License: MIT
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
import warnings

from .thermodynamics import TemperatureDependentKinetics


@dataclass
class BufferSystem:
    """
    Parameters for a buffer system in water.

    Attributes:
        alkalinity: Total alkalinity [mg/L as CaCO₃]
        total_carbonate: Total carbonate species [mmol/L]
        temperature: Operating temperature [°C]
    """

    alkalinity: float  # [mg/L as CaCO₃]
    total_carbonate: float  # [mmol/L]
    temperature: float = 20.0  # [°C]

    def validate(self) -> None:
        """Validate buffer system parameters."""
        if self.alkalinity < 0:
            raise ValueError(f"Alkalinity cannot be negative: {self.alkalinity}")
        if self.total_carbonate < 0:
            raise ValueError(
                f"Total carbonate cannot be negative: {self.total_carbonate}"
            )
        if self.temperature < 0 or self.temperature > 40:
            warnings.warn(
                f"Temperature {self.temperature}°C outside typical range [0, 40]"
            )


class AqueousChemistry:
    """
    Rigorous aqueous chemistry calculations for water treatment.

    This class handles:
    - pH calculations with carbonate buffering
    - Acid/base addition effects
    - Chlorine speciation
    - Buffering capacity (β = dC_base/d(pH))
    """

    # Physical constants
    CACO3_MW = 100.09  # [g/mol] Calcium carbonate molecular weight
    PROTON_CHARGE = 1  # Elementary charge

    # Convergence parameters for iterative pH calculation
    PH_TOLERANCE = 1e-6  # Convergence criterion
    MAX_ITERATIONS = 100  # Maximum Newton-Raphson iterations

    def __init__(self, buffer_system: BufferSystem):
        """
        Initialize chemistry calculator with buffer system.

        Args:
            buffer_system: Buffer parameters for the water
        """
        buffer_system.validate()
        self.buffer = buffer_system
        self.thermo = TemperatureDependentKinetics()

        # Cache temperature-dependent constants
        self._update_temperature_constants()

    def _update_temperature_constants(self) -> None:
        """Update all temperature-dependent equilibrium constants."""
        T = self.buffer.temperature

        self.Kw = self.thermo.water_ionization_constant(T)
        self.pKw = -np.log10(self.Kw)

        self.pKa1 = self.thermo.carbonate_pKa(T, dissociation=1)
        self.Ka1 = 10 ** (-self.pKa1)

        self.pKa2 = self.thermo.carbonate_pKa(T, dissociation=2)
        self.Ka2 = 10 ** (-self.pKa2)

        # Chlorine dissociation (HOCl ⇌ H⁺ + OCl⁻)
        # pKa varies slightly with T: pKa(T) ≈ 7.5 + 0.01*(T-25)
        self.pKa_HOCl = 7.5 + 0.01 * (T - 25.0)
        self.Ka_HOCl = 10 ** (-self.pKa_HOCl)

    def H_from_pH(self, pH: float) -> float:
        """
        Convert pH to hydrogen ion concentration.

        Args:
            pH: pH value

        Returns:
            [H⁺] in mol/L
        """
        return 10 ** (-pH)

    def pH_from_H(self, H: float) -> float:
        """
        Convert hydrogen ion concentration to pH.

        Args:
            H: [H⁺] in mol/L

        Returns:
            pH value
        """
        return -np.log10(H)

    def alpha_carbonate(self, pH: float) -> Tuple[float, float, float]:
        """
        Calculate carbonate species distribution (alpha values).

        For total carbonate C_T:
        α₀ = [H₂CO₃]/C_T
        α₁ = [HCO₃⁻]/C_T
        α₂ = [CO₃²⁻]/C_T

        Where: α₀ + α₁ + α₂ = 1

        Args:
            pH: pH value

        Returns:
            (α₀, α₁, α₂) - fractions of each carbonate species

        Example:
            >>> chem = AqueousChemistry(BufferSystem(100, 2.0, 20))
            >>> a0, a1, a2 = chem.alpha_carbonate(7.0)
            >>> abs(a0 + a1 + a2 - 1.0) < 1e-10  # Must sum to 1
            True
        """
        H = self.H_from_pH(pH)

        # Denominator for all alphas
        D = H**2 + self.Ka1 * H + self.Ka1 * self.Ka2

        # Distribution of species
        alpha_0 = H**2 / D  # H₂CO₃
        alpha_1 = (self.Ka1 * H) / D  # HCO₃⁻
        alpha_2 = (self.Ka1 * self.Ka2) / D  # CO₃²⁻

        return alpha_0, alpha_1, alpha_2

    def charge_balance_error(self, pH: float) -> float:
        """
        Calculate charge balance error for given pH.

        Charge balance equation:
        [H⁺] + Σ[cations] = [OH⁻] + [HCO₃⁻] + 2[CO₃²⁻] + Σ[other anions]

        Rearranged:
        f(pH) = [H⁺] - [OH⁻] + [HCO₃⁻] + 2[CO₃²⁻] - (alkalinity/50,000)

        Where alkalinity is in mg/L as CaCO₃ and 50,000 converts to eq/L.

        Args:
            pH: Trial pH value

        Returns:
            Charge balance error [eq/L]
        """
        H = self.H_from_pH(pH)
        OH = self.Kw / H

        # Carbonate species concentrations
        C_T_mol = self.buffer.total_carbonate / 1000.0  # Convert mmol/L to mol/L
        alpha_0, alpha_1, alpha_2 = self.alpha_carbonate(pH)

        HCO3 = alpha_1 * C_T_mol
        CO3 = alpha_2 * C_T_mol

        # Alkalinity in equivalents per liter
        # Alkalinity [mg/L as CaCO₃] / 50,000 = [eq/L]
        alk_eq = self.buffer.alkalinity / 50000.0

        # Charge balance error
        error = H - OH + HCO3 + 2 * CO3 - alk_eq

        return error

    def charge_balance_derivative(self, pH: float) -> float:
        """
        Calculate derivative of charge balance with respect to pH.

        d(f)/d(pH) = d(f)/d[H⁺] * d[H⁺]/d(pH)

        Used for Newton-Raphson iteration: pH_new = pH_old - f/f'

        Args:
            pH: Current pH value

        Returns:
            df/d(pH)
        """
        H = self.H_from_pH(pH)

        # d[H⁺]/d(pH) = -ln(10) * [H⁺]
        dH_dpH = -np.log(10) * H

        # d[OH⁻]/d(pH) = d(Kw/[H⁺])/d(pH) = -Kw/[H⁺]² * d[H⁺]/d(pH)
        dOH_dpH = -(self.Kw / H**2) * dH_dpH

        # Carbonate derivatives (complex but necessary for accuracy)
        C_T_mol = self.buffer.total_carbonate / 1000.0
        alpha_0, alpha_1, alpha_2 = self.alpha_carbonate(pH)

        # These derivatives come from differentiating the alpha expressions
        D = H**2 + self.Ka1 * H + self.Ka1 * self.Ka2
        dD_dH = 2 * H + self.Ka1

        dalpha1_dH = self.Ka1 * (D - H * dD_dH) / D**2
        dalpha2_dH = -self.Ka1 * self.Ka2 * dD_dH / D**2

        dHCO3_dpH = C_T_mol * dalpha1_dH * dH_dpH
        dCO3_dpH = C_T_mol * dalpha2_dH * dH_dpH

        # Total derivative
        df_dpH = dH_dpH - dOH_dpH + dHCO3_dpH + 2 * dCO3_dpH

        return df_dpH

    def calculate_pH(
        self,
        initial_guess: float = 7.0,
        tolerance: float = PH_TOLERANCE,
        max_iter: int = MAX_ITERATIONS,
    ) -> float:
        """
        Calculate equilibrium pH using Newton-Raphson iteration.

        Solves charge balance equation iteratively:
        pH_new = pH_old - f(pH_old) / f'(pH_old)

        Args:
            initial_guess: Starting pH for iteration
            tolerance: Convergence criterion |ΔpH|
            max_iter: Maximum iterations before failure

        Returns:
            Equilibrium pH

        Raises:
            RuntimeError: If convergence not achieved

        Example:
            >>> buffer = BufferSystem(alkalinity=100, total_carbonate=2.0, temperature=20)
            >>> chem = AqueousChemistry(buffer)
            >>> pH = chem.calculate_pH()
            >>> 6.0 < pH < 9.0  # Should be reasonable for this buffer
            True
        """
        pH = initial_guess

        for iteration in range(max_iter):
            # Evaluate charge balance and derivative
            f = self.charge_balance_error(pH)
            df_dpH = self.charge_balance_derivative(pH)

            # Newton-Raphson update
            if abs(df_dpH) < 1e-15:
                raise RuntimeError(
                    f"Derivative too small at pH={pH:.3f}, cannot continue"
                )

            delta_pH = -f / df_dpH
            pH_new = pH + delta_pH

            # Ensure pH stays in physical bounds
            pH_new = np.clip(pH_new, 0.0, 14.0)

            # Check convergence
            if abs(delta_pH) < tolerance:
                return pH_new

            pH = pH_new

        # If we get here, convergence failed
        raise RuntimeError(
            f"pH calculation did not converge after {max_iter} iterations. "
            f"Final pH={pH:.3f}, error={f:.2e}"
        )

    def add_acid(self, volume_L: float, acid_mol: float, current_pH: float) -> float:
        """
        Calculate new pH after adding strong acid.

        Strong acid completely dissociates: HA → H⁺ + A⁻

        This affects the charge balance and shifts the carbonate equilibrium.

        Args:
            volume_L: Total volume of solution [L]
            acid_mol: Moles of acid added [mol]
            current_pH: pH before acid addition

        Returns:
            New pH after acid addition

        Example:
            >>> buffer = BufferSystem(alkalinity=100, total_carbonate=2.0, temperature=20)
            >>> chem = AqueousChemistry(buffer)
            >>> pH_initial = chem.calculate_pH()
            >>> pH_after = chem.add_acid(volume_L=1000, acid_mol=0.001, current_pH=pH_initial)
            >>> pH_after < pH_initial  # pH should decrease
            True
        """
        # Acid adds H⁺, which reduces alkalinity
        # Δ alkalinity = -acid_mol / volume * 50000 [mg/L as CaCO₃]
        delta_alk = -(acid_mol / volume_L) * 50000.0

        # Create temporary buffer with adjusted alkalinity
        new_buffer = BufferSystem(
            alkalinity=self.buffer.alkalinity + delta_alk,
            total_carbonate=self.buffer.total_carbonate,
            temperature=self.buffer.temperature,
        )

        temp_chem = AqueousChemistry(new_buffer)
        new_pH = temp_chem.calculate_pH(initial_guess=current_pH)

        return new_pH

    def add_base(self, volume_L: float, base_mol: float, current_pH: float) -> float:
        """
        Calculate new pH after adding strong base.

        Strong base completely dissociates: BOH → B⁺ + OH⁻

        Args:
            volume_L: Total volume of solution [L]
            base_mol: Moles of base added [mol]
            current_pH: pH before base addition

        Returns:
            New pH after base addition
        """
        # Base adds OH⁻, which increases alkalinity
        delta_alk = (base_mol / volume_L) * 50000.0

        new_buffer = BufferSystem(
            alkalinity=self.buffer.alkalinity + delta_alk,
            total_carbonate=self.buffer.total_carbonate,
            temperature=self.buffer.temperature,
        )

        temp_chem = AqueousChemistry(new_buffer)
        new_pH = temp_chem.calculate_pH(initial_guess=current_pH)

        return new_pH

    def buffering_capacity(self, pH: float) -> float:
        """
        Calculate buffering capacity β at given pH.

        β = dC_base/d(pH)

        Buffering capacity is maximum near pKa values (6.35 and 10.33 for carbonate).

        Args:
            pH: pH value

        Returns:
            Buffering capacity [mol/L per pH unit]

        Example:
            >>> buffer = BufferSystem(alkalinity=100, total_carbonate=2.0, temperature=20)
            >>> chem = AqueousChemistry(buffer)
            >>> beta_6 = chem.buffering_capacity(6.35)  # Near pKa1
            >>> beta_8 = chem.buffering_capacity(8.0)   # Between pKa values
            >>> beta_6 > beta_8  # Buffering stronger near pKa
            True
        """
        H = self.H_from_pH(pH)

        # Water contribution to buffering
        beta_water = 2.303 * (H + self.Kw / H)

        # Carbonate system contribution
        C_T_mol = self.buffer.total_carbonate / 1000.0
        alpha_0, alpha_1, alpha_2 = self.alpha_carbonate(pH)

        beta_carbonate = (
            2.303
            * C_T_mol
            * (alpha_0 * alpha_1 + 4 * alpha_1 * alpha_2 + alpha_0 * alpha_2)
        )

        return beta_water + beta_carbonate

    def chlorine_speciation(
        self, total_chlorine_mg_L: float, pH: float
    ) -> Dict[str, float]:
        """
        Calculate chlorine speciation between HOCl and OCl⁻.

        HOCl ⇌ H⁺ + OCl⁻

        HOCl is a stronger disinfectant than OCl⁻, so speciation matters
        for disinfection effectiveness.

        Args:
            total_chlorine_mg_L: Total free chlorine [mg/L as Cl₂]
            pH: pH value

        Returns:
            Dictionary with 'HOCl', 'OCl', 'HOCl_fraction', 'OCl_fraction'

        Example:
            >>> buffer = BufferSystem(alkalinity=100, total_carbonate=2.0, temperature=20)
            >>> chem = AqueousChemistry(buffer)
            >>> spec = chem.chlorine_speciation(2.0, pH=7.0)
            >>> spec['HOCl'] + spec['OCl'] - 2.0 < 1e-10  # Must sum to total
            True
        """
        H = self.H_from_pH(pH)

        # Fraction as HOCl
        alpha_HOCl = H / (H + self.Ka_HOCl)

        # Fraction as OCl⁻
        alpha_OCl = self.Ka_HOCl / (H + self.Ka_HOCl)

        HOCl_mg_L = alpha_HOCl * total_chlorine_mg_L
        OCl_mg_L = alpha_OCl * total_chlorine_mg_L

        return {
            "HOCl": HOCl_mg_L,
            "OCl": OCl_mg_L,
            "HOCl_fraction": alpha_HOCl,
            "OCl_fraction": alpha_OCl,
            "effective_disinfection": alpha_HOCl,  # HOCl is ~80x more effective
        }

    def pH_dependent_chlorine_decay_factor(self, pH: float) -> float:
        """
        Calculate pH-dependent chlorine decay rate multiplier.

        HOCl (hypochlorous acid) decays much faster than OCl⁻ (hypochlorite ion).
        Literature shows HOCl decay rate is 50-100x faster than OCl⁻.

        k_effective(pH) = k_base * [α_HOCl * k_HOCl_rel + α_OCl * k_OCl_rel]

        Where:
        - k_HOCl_rel = 1.0 (reference species)
        - k_OCl_rel = 0.02 (50x slower than HOCl)

        Args:
            pH: pH value

        Returns:
            Decay rate multiplier relative to base rate [dimensionless]

        Example:
            >>> buffer = BufferSystem(alkalinity=100, total_carbonate=2.0, temperature=20)
            >>> chem = AqueousChemistry(buffer)
            >>> factor_6 = chem.pH_dependent_chlorine_decay_factor(6.0)  # Acidic, mostly HOCl
            >>> factor_9 = chem.pH_dependent_chlorine_decay_factor(9.0)  # Basic, mostly OCl⁻
            >>> factor_6 > factor_9  # HOCl decays faster
            True
        """
        H = self.H_from_pH(pH)

        # Fraction as HOCl and OCl⁻
        alpha_HOCl = H / (H + self.Ka_HOCl)
        alpha_OCl = self.Ka_HOCl / (H + self.Ka_HOCl)

        # Relative decay rates (HOCl as reference = 1.0)
        k_HOCl_relative = 1.0
        k_OCl_relative = 0.02  # OCl⁻ decays 50x slower

        # Weighted average decay factor
        decay_factor = alpha_HOCl * k_HOCl_relative + alpha_OCl * k_OCl_relative

        return decay_factor


def validate_chemistry() -> None:
    """
    Comprehensive validation of chemistry calculations.

    Tests:
    1. Charge balance convergence
    2. Alpha values sum to 1
    3. pH changes with acid/base addition
    4. Buffering capacity maximum near pKa
    5. Chlorine speciation
    """
    buffer = BufferSystem(alkalinity=100, total_carbonate=2.0, temperature=20)
    chem = AqueousChemistry(buffer)

    # Test 1: pH calculation converges
    pH = chem.calculate_pH()
    assert 6.0 < pH < 9.0, f"pH {pH} outside expected range"

    # Test 2: Alpha values sum to 1
    a0, a1, a2 = chem.alpha_carbonate(pH)
    assert abs(a0 + a1 + a2 - 1.0) < 1e-10, "Alpha values don't sum to 1"

    # Test 3: Acid addition decreases pH
    pH_after_acid = chem.add_acid(1000, 0.001, pH)
    assert pH_after_acid < pH, "Acid should decrease pH"

    # Test 4: Base addition increases pH
    pH_after_base = chem.add_base(1000, 0.001, pH)
    assert pH_after_base > pH, "Base should increase pH"

    # Test 5: Buffering capacity maximum near pKa
    beta_6_35 = chem.buffering_capacity(6.35)
    beta_8_0 = chem.buffering_capacity(8.0)
    assert beta_6_35 > beta_8_0, "Buffering should be stronger near pKa"

    # Test 6: Chlorine speciation
    spec = chem.chlorine_speciation(2.0, 7.0)
    assert abs(spec["HOCl"] + spec["OCl"] - 2.0) < 1e-10, "Chlorine doesn't balance"

    print("✓ All chemistry validations passed")


if __name__ == "__main__":
    """
    Demonstration of aqueous chemistry calculations.
    """
    # Create buffer system representative of typical drinking water
    buffer = BufferSystem(
        alkalinity=100,  # mg/L as CaCO₃ (moderate hardness)
        total_carbonate=2.0,  # mmol/L
        temperature=20,  # °C
    )

    chem = AqueousChemistry(buffer)

    print("Water Chemistry Demonstration")
    print("=" * 60)
    print(f"Alkalinity: {buffer.alkalinity} mg/L as CaCO₃")
    print(f"Total Carbonate: {buffer.total_carbonate} mmol/L")
    print(f"Temperature: {buffer.temperature}°C")
    print()

    # Calculate equilibrium pH
    pH_eq = chem.calculate_pH()
    print(f"Equilibrium pH: {pH_eq:.3f}")
    print()

    # Analyze buffering capacity across pH range
    print("Buffering Capacity Analysis:")
    print(f"{'pH':<8} {'β (mmol/L/pH)':<20} {'[HOCl]/[Cl_total]':<20}")
    print("-" * 60)

    for pH in np.arange(6.0, 9.5, 0.5):
        beta = chem.buffering_capacity(pH) * 1000  # Convert to mmol/L
        spec = chem.chlorine_speciation(2.0, pH)
        print(f"{pH:<8.1f} {beta:<20.2f} {spec['HOCl_fraction']:<20.3f}")

    print()
    print("Key Observations:")
    print(f"  • Maximum buffering near pKa1 = {chem.pKa1:.2f}")
    print(f"  • HOCl dominant below pH {chem.pKa_HOCl:.1f} (better disinfection)")
    print("  • Typical drinking water pH: 6.5-8.5")
    print()

    # Run validation
    validate_chemistry()
