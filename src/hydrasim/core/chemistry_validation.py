"""Validation and demonstration helpers for aqueous chemistry."""

import numpy as np

from .chemistry import AqueousChemistry
from .chemistry_models import BufferSystem


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


def demo_chemistry() -> None:
    """Demonstrate aqueous chemistry calculations."""
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


if __name__ == "__main__":
    demo_chemistry()
