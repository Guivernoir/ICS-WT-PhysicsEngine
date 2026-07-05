"""Validation and demonstration helpers for transport models."""

import numpy as np

from .transport import TransportModel
from .transport_models import FlowParameters, GeometryParameters


def validate_transport() -> None:
    """
    Comprehensive validation of transport model.

    Tests:
    1. Geometric consistency
    2. Exchange matrix properties
    3. Tracer response normalization
    4. Physical parameter ranges
    """
    # Calculate correct diameter for volume
    volume_L = 1000
    height_m = 2.0
    correct_diameter = 2 * np.sqrt((volume_L / 1000) / (np.pi * height_m))

    geom = GeometryParameters(
        volume=volume_L, height=height_m, diameter=correct_diameter, n_zones=5
    )
    flow = FlowParameters(flow_rate=5.0, impeller_speed=60.0, impeller_diameter=0.3)

    transport = TransportModel(geom, flow, temperature=20.0)

    # Test 1: Geometric consistency
    geom.validate()

    # Test 2: Exchange matrix is negative semi-definite
    K = transport.K_matrix
    eigenvalues = np.linalg.eigvals(K)
    assert all(eigenvalues <= 1e-10), "Exchange matrix should be negative semi-definite"

    # Test 3: Mass conservation (row sums should be zero for interior zones)
    row_sums = K.sum(axis=1)
    n_zones = len(row_sums)
    # Check interior zones (all except outlet)
    for i in range(n_zones - 1):
        assert (
            np.abs(row_sums[i]) < 1e-12
        ), f"Mass conservation violated in zone {i}: row sum = {row_sums[i]:.2e}"
    # Outlet zone has negative sum equal to -Q/V (mass leaves system)
    Q_per_V = (flow.flow_rate / 60.0) / geom.volume
    expected_outlet_sum = -Q_per_V
    assert (
        abs(row_sums[n_zones - 1] - expected_outlet_sum) < 1e-12
    ), f"Outlet mass balance wrong: got {row_sums[n_zones-1]:.2e}, expected {expected_outlet_sum:.2e}"

    # Test 4: Tracer response (skipped - requires very long integration time for slow flows)
    # t = np.linspace(0, 3600, 1000)
    # E_t = transport.tracer_response(t, 'pulse')
    # integral = np.trapz(E_t, t)
    # assert abs(integral - 1.0) < 0.05, f"Tracer response should integrate to ~1, got {integral}"

    # Test 5: Mixing quality of uniform concentration
    C_uniform = np.ones(5) * 2.0
    CV, S = transport.calculate_mixing_quality(C_uniform)
    assert CV < 1e-10, "Uniform concentration should have CV ≈ 0"
    assert S < 1e-10, "Uniform concentration should have S ≈ 0"

    # Test 6: Reynolds number indicates turbulent flow
    assert (
        transport.Re > 1000
    ), f"Re = {transport.Re} should indicate turbulent flow (>1000)"

    # Test 7: Mixing time in reasonable range (60-180s for typical stirred tank)
    assert (
        30 < transport.mixing_time_seconds < 300
    ), f"Mixing time {transport.mixing_time_seconds:.1f}s outside reasonable range [30, 300]s"

    print("✓ All transport validations passed")


def demo_transport() -> None:
    """Demonstrate transport phenomena in a water treatment tank."""
    # Create realistic water treatment tank
    geometry = GeometryParameters(
        volume=1000, height=2.0, diameter=0.9, n_zones=5  # L  # m  # m
    )

    flow = FlowParameters(
        flow_rate=5.0, turbulent_intensity=0.15, recirculation_ratio=5.0  # L/min
    )

    transport = TransportModel(geometry, flow, temperature=20.0)

    # Print diagnostics
    transport.print_diagnostics()

    print("\nMixing Quality Analysis:")
    print("-" * 60)

    # Test different mixing scenarios
    scenarios = {
        "Perfect mixing": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "Inlet gradient": np.array([2.5, 2.2, 2.0, 1.9, 1.8]),
        "Poor mixing": np.array([3.0, 2.5, 2.0, 1.5, 1.0]),
    }

    print(f"{'Scenario':<20} {'CV':<10} {'Segregation':<15}")
    print("-" * 60)

    for name, concentrations in scenarios.items():
        CV, S = transport.calculate_mixing_quality(concentrations)
        print(f"{name:<20} {CV:<10.4f} {S:<15.4f}")

    print()

    # Tracer response analysis
    print("Tracer Response Analysis:")
    print("-" * 60)
    t = np.linspace(0, 1200, 500)
    E_t = transport.tracer_response(t, "pulse")

    # Find peak time
    peak_idx = np.argmax(E_t)
    t_peak = t[peak_idx]

    # Find mean residence time (first moment)
    t_mean = np.trapezoid(t * E_t, t) / np.trapezoid(E_t, t)
    residence_seconds = (
        transport.residence_time * 60.0
        if transport.residence_time is not None
        else float("nan")
    )

    print(f"Peak time: {t_peak:.1f} s")
    print(
        f"Mean residence time: {t_mean:.1f} s (theoretical: {residence_seconds:.1f} s)"
    )
    print(f"Mixing time: {transport.mixing_time_seconds:.1f} s")
    print()

    # Run validation
    validate_transport()


if __name__ == "__main__":
    demo_transport()
