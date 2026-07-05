"""Spatial diagnostics helpers."""

from typing import Any


def print_spatial_diagnostics(self: Any) -> None:
    """Print detailed spatial diagnostics."""
    print("Spatial Model Diagnostics")
    print("=" * 60)
    print(f"Number of zones: {self.n_zones}")
    print(f"Tank height: {self.height:.2f} m")
    print(f"Zone height: {self.zone_height:.3f} m")
    print()

    print("Temperature Profile:")
    print(f"{'Zone':<8} {'Elevation(m)':<15} {'Temp(°C)':<12} {'Density(kg/m³)':<15}")
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
