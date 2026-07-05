"""Integrated reactor diagnostics helpers."""

from typing import Any


def print_diagnostics(self: Any) -> None:
    """Print comprehensive reactor diagnostics."""
    print("\n" + "=" * 70)
    print("CSTR PHYSICS DIAGNOSTICS")
    print("=" * 70)

    print(f"\nTime: {self.state.time:.1f} s")
    if self.transport.residence_time is None:
        print("Residence time: batch mode")
    else:
        print(f"Residence time: {self.transport.residence_time:.1f} min")
    print(f"Mixing time: {self.transport.mixing_time_seconds:.1f} s")

    print(
        f"\n{'Zone':<6} {'pH':<8} {'FreeCl':<10} {'ClNH2':<10} {'NH3-N':<10} {'T(°C)':<8}"
    )
    print("-" * 65)
    for i in range(self.config.n_zones):
        print(
            f"{i:<6} {self.state.pH[i]:<8.3f} {self.state.chlorine[i]:<10.3f} "
            f"{self.state.chloramine[i]:<10.3f} {self.state.ammonia[i]:<10.3f} "
            f"{self.state.temperature[i]:<8.2f}"
        )

    # Conservation
    conservation = self.validate_conservation()
    print("\nConservation Laws:")
    print(f"  Total Chlorine: {conservation['total_chlorine_mg']:.2f} mg")
    print(f"  Total Chloramine: {conservation['total_chloramine_mg']:.2f} mg")
    print(f"  Charge Balance: {conservation['charge_balance_mol']:.2e} mol")

    # Mixing quality
    pH_CV, pH_S = self.transport.calculate_mixing_quality(self.state.pH)
    Cl_CV, Cl_S = self.transport.calculate_mixing_quality(self.state.chlorine)

    print("\nMixing Quality:")
    print(f"  pH segregation index: {pH_S:.4f}")
    print(f"  Chlorine segregation index: {Cl_S:.4f}")

    print("=" * 70 + "\n")
