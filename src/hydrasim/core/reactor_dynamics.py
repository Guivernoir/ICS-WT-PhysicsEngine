"""ODE dynamics for the integrated reactor model."""

from typing import Any

import numpy as np

from .reactor_models import BoundaryConditions


def calculate_derivatives(
    self: Any, t: float, y: np.ndarray, boundary: BoundaryConditions
) -> np.ndarray:
    """
    Calculate time derivatives for all state variables.

    This is the heart of the physics engine - the ODE system that
    governs reactor dynamics.

    State vector:
    y = [pH₀..pHₙ, Cl_free₀..Cl_freeₙ, T₀..Tₙ, NH3₀..NH3ₙ, Cl_combined₀..Cl_combinedₙ, Demand₀..Demandₙ]

    Args:
        t: Current time [s]
        y: State vector
        boundary: Physical boundary conditions

    Returns:
        dy/dt: Time derivatives
    """
    n = self.config.n_zones

    # Unpack state vector
    pH_zones = y[0:n]
    Cl_zones = y[n : 2 * n]
    T_zones = y[2 * n : 3 * n]
    NH3_zones = y[3 * n : 4 * n]
    Cl_combined_zones = y[4 * n : 5 * n]
    demand_zones = y[5 * n : 6 * n]

    # Initialize derivatives
    dpH_dt = np.zeros(n)
    dCl_dt = np.zeros(n)
    dT_dt = np.zeros(n)
    dNH3_dt = np.zeros(n)
    dCl_combined_dt = np.zeros(n)
    dDemand_dt = np.zeros(n)

    # Update spatial model with current temperatures
    self.spatial.update_density_profile(T_zones)

    # Calculate mixing suppression from stratification
    velocity_scale = self.transport.superficial_velocity

    # Enable stratification effects based on configuration
    if self.config.enable_thermal_stratification:
        mixing_suppression = self.spatial.calculate_mixing_suppression(velocity_scale)
    else:
        mixing_suppression = np.ones(n - 1)  # No suppression - perfect mixing

    # Modify exchange matrix for stratification effects
    K_matrix = self.transport.K_matrix.copy()

    # First, modify all off-diagonal exchange terms
    for i in range(n - 1):
        # Reduce mixing between stratified layers
        K_matrix[i, i + 1] *= mixing_suppression[i]
        K_matrix[i + 1, i] *= mixing_suppression[i]

    # Then, recalculate ALL diagonal terms to maintain mass conservation
    # (Do this after all off-diagonal modifications are complete)
    for i in range(n):
        # Sum off-diagonal terms in this row
        off_diagonal_sum = sum(K_matrix[i, j] for j in range(n) if j != i)
        # Diagonal should be negative of off-diagonal sum to conserve mass
        K_matrix[i, i] = -off_diagonal_sum

    # Total flow controls hydraulic residence. Dosing streams are real hydraulic inflows.
    Q_main = max(0.0, boundary.inlet_flow_rate)
    Q_acid = max(0.0, boundary.acid_flow_rate)
    Q_chlorine = max(0.0, boundary.chlorine_flow_rate)
    Q_total = Q_main + Q_acid + Q_chlorine

    # Through-flow intensity [1/s] for compartment advection.
    Q_per_V = (Q_total / 60.0) / self.config.volume if Q_total > 0 else 0.0  # [1/s]

    # Mixed influent quality from all hydraulic inflows.
    if Q_total > 0:
        H_main = 10 ** (-boundary.inlet_pH)
        H_acid = max(boundary.acid_concentration, 1e-12)  # strong-acid approximation
        H_chlorine = 10 ** (-boundary.chlorine_solution_pH)

        H_in_mixed = (
            Q_main * H_main + Q_acid * H_acid + Q_chlorine * H_chlorine
        ) / Q_total
        Cl_in_mixed = (
            Q_main * boundary.inlet_chlorine
            + Q_chlorine * boundary.chlorine_concentration
        ) / Q_total
        NH3_in_mixed = (Q_main * boundary.inlet_ammonia) / Q_total
        Cl_combined_in_mixed = (Q_main * boundary.inlet_chloramine) / Q_total
        demand_in_mixed = (Q_main * boundary.inlet_chlorine_demand) / Q_total
        T_in_mixed = (
            Q_main * boundary.inlet_temperature
            + Q_acid * boundary.acid_temperature
            + Q_chlorine * boundary.chlorine_temperature
        ) / Q_total
    else:
        H_in_mixed = 10 ** (-pH_zones[0])
        Cl_in_mixed = Cl_zones[0]
        NH3_in_mixed = NH3_zones[0]
        Cl_combined_in_mixed = Cl_combined_zones[0]
        demand_in_mixed = demand_zones[0]
        T_in_mixed = T_zones[0]

    # --- pH DYNAMICS ---
    # pH changes due to:
    # 1. Mixed influent H+ concentration
    # 2. Inter-zone mixing
    # 3. Carbonate buffering
    H_zones = 10 ** (-pH_zones)  # [mol/L]
    dH_dt = K_matrix @ H_zones

    # Through-tank advection: each zone receives upstream concentration and
    # discharges downstream concentration at the same hydraulic rate.
    if Q_per_V > 0:
        dH_dt[1:] += Q_per_V * (H_zones[:-1] - H_zones[1:])

    # Inlet boundary enters zone 0.
    dH_dt[0] += Q_per_V * (H_in_mixed - H_zones[0])

    for i in range(n):
        beta_i = self.chemistry.buffering_capacity(pH_zones[i])
        if beta_i > 0:
            dpH_dt[i] += -dH_dt[i] / (beta_i * np.log(10))

    # --- FREE CHLORINE / AMMONIA / CHLORAMINE / DEMAND DYNAMICS ---
    # Advective boundary exchange in inlet zone
    dCl_dt[0] += Q_per_V * (Cl_in_mixed - Cl_zones[0])
    dNH3_dt[0] += Q_per_V * (NH3_in_mixed - NH3_zones[0])
    dCl_combined_dt[0] += Q_per_V * (Cl_combined_in_mixed - Cl_combined_zones[0])
    dDemand_dt[0] += Q_per_V * (demand_in_mixed - demand_zones[0])

    # Inter-zone exchange
    dCl_dt += K_matrix @ Cl_zones
    dNH3_dt += K_matrix @ NH3_zones
    dCl_combined_dt += K_matrix @ Cl_combined_zones
    dDemand_dt += K_matrix @ demand_zones

    # Through-tank advection (zone i-1 -> zone i).
    if Q_per_V > 0:
        dCl_dt[1:] += Q_per_V * (Cl_zones[:-1] - Cl_zones[1:])
        dNH3_dt[1:] += Q_per_V * (NH3_zones[:-1] - NH3_zones[1:])
        dCl_combined_dt[1:] += Q_per_V * (
            Cl_combined_zones[:-1] - Cl_combined_zones[1:]
        )
        dDemand_dt[1:] += Q_per_V * (demand_zones[:-1] - demand_zones[1:])

    # Reaction kinetics (lumped but physically grounded):
    # 1) Free chlorine self-decay
    # 2) NH3 + HOCl -> chloramines (consumes free chlorine and ammonia)
    # 3) Free chlorine demand from bulk reducing agents/organics
    CL2_PER_N_MASS = 70.906 / 14.007  # stoichiometric mg Cl2 per mg N
    K_CHLORAMINE_REF = 1.2e-3  # [L/(mg*s)] at 20°C for NH3-N basis
    K_DEMAND_REF = 3.0e-4  # [L/(mg*s)] at 20°C
    K_CHLORAMINE_DECAY_REF = 1.5e-5  # [1/s] at 20°C

    for i in range(n):
        T = T_zones[i]
        pH = pH_zones[i]

        # 1) Free chlorine self-decay
        k_base = self.thermo.chlorine_decay_rate(T)
        pH_factor = self.chemistry.pH_dependent_chlorine_decay_factor(pH)
        free_decay = k_base * pH_factor * Cl_zones[i]

        # Hypochlorous acid fraction controls chlorine reactivity.
        # HOCl is much more reactive than OCl- for most reactions.
        pKa_hocl = 7.5 + 0.01 * (T - 25.0)
        hocl_fraction = 1.0 / (1.0 + 10 ** (pH - pKa_hocl))

        # 2) Chloramine formation from ammonia (via NH3 free-base fraction)
        pKa_nh4 = 9.25 - 0.03 * (T - 25.0)
        nh3_fraction = 1.0 / (1.0 + 10 ** (pKa_nh4 - pH))
        k_temp_chloramine = 1.6 ** ((T - 20.0) / 10.0)
        k_chloramine = (
            K_CHLORAMINE_REF * k_temp_chloramine * nh3_fraction * hocl_fraction
        )
        nh3_consumption = k_chloramine * Cl_zones[i] * NH3_zones[i]  # [mgN/L/s]
        free_from_nh3 = CL2_PER_N_MASS * nh3_consumption  # [mgCl2/L/s]

        # 3) Bulk chlorine demand (organics/reducing agents lumped)
        k_temp_demand = 1.4 ** ((T - 20.0) / 10.0)
        demand_consumption = (
            K_DEMAND_REF
            * k_temp_demand
            * Cl_zones[i]
            * demand_zones[i]
            * (0.2 + 0.8 * hocl_fraction)
        )  # [mgCl2/L/s]

        # Combined chlorine decay (much slower than free chlorine)
        k_chloramine_decay = K_CHLORAMINE_DECAY_REF * k_temp_demand
        chloramine_decay = k_chloramine_decay * Cl_combined_zones[i]

        dCl_dt[i] -= free_decay + free_from_nh3 + demand_consumption
        dNH3_dt[i] -= nh3_consumption
        dDemand_dt[i] -= demand_consumption
        dCl_combined_dt[i] += free_from_nh3 - chloramine_decay

    # --- TEMPERATURE DYNAMICS ---
    # Temperature changes due to:
    # 1. Mixed influent stream
    # 2. Inter-zone exchange + through-tank advection
    # 3. Heat loss to environment
    dT_dt[0] += Q_per_V * (T_in_mixed - T_zones[0])
    dT_dt += K_matrix @ T_zones
    if Q_per_V > 0:
        dT_dt[1:] += Q_per_V * (T_zones[:-1] - T_zones[1:])

    # 3. Heat loss to environment (if specified)
    if boundary.heat_loss_coefficient > 0:
        # Q_loss = U * A * (T - T_ambient)
        # For cylindrical tank: A = π*D*H + 2*π*(D/2)²
        A_lateral = np.pi * self.config.diameter * self.config.height
        A_ends = 2 * np.pi * (self.config.diameter / 2) ** 2
        A_total = A_lateral + A_ends  # [m²]

        rho = 998.2  # [kg/m³]
        cp = 4184  # [J/(kg·K)]
        V_m3 = self.config.volume / 1000  # [m³]

        for i in range(n):
            Q_loss_W = (
                boundary.heat_loss_coefficient
                * A_total
                * (T_zones[i] - boundary.ambient_temperature)
            )
            dT_dt[i] -= Q_loss_W / (rho * cp * V_m3)

    # Combine all derivatives
    dydt = np.concatenate([dpH_dt, dCl_dt, dT_dt, dNH3_dt, dCl_combined_dt, dDemand_dt])

    return dydt
