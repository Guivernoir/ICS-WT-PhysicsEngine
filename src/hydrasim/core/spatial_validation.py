"""Validation and demonstration helpers for spatial models."""

import numpy as np

from .spatial import SpatialModel
from .spatial_models import StratificationParameters


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


def demo_spatial() -> None:
    """Demonstrate spatial modeling and stratification effects."""
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
    spatial.calculate_mixing_suppression(velocity_scale)

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


if __name__ == "__main__":
    demo_spatial()
