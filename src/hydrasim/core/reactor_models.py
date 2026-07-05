"""Integrated reactor data models."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ReactorConfiguration:
    """
    Complete configuration for CSTR reactor.

    Combines geometry, flow, chemistry, and operational parameters.
    """

    # Geometry
    volume: float = 1000.0  # [L]
    height: float = 2.0  # [m]
    diameter: float = 0.798  # [m] Calculated to match volume
    n_zones: int = 5

    # Flow
    flow_rate: float = 5.0  # [L/min] Nominal flow rate
    turbulent_intensity: float = 0.15
    recirculation_ratio: float = 5.0
    impeller_speed: float = 60.0  # [rpm]
    impeller_diameter: float = 0.3  # [m]
    power_number: float = 5.0  # Rushton turbine

    # Chemistry
    initial_pH: float = 7.0
    alkalinity: float = 100.0  # [mg/L as CaCO₃]
    total_carbonate: float = 2.0  # [mmol/L]

    # Chlorination
    initial_chlorine: float = 2.0  # [mg/L]
    initial_chloramine: float = 0.0  # [mg/L as Cl2]
    initial_ammonia: float = 0.15  # [mg/L as N]
    initial_chlorine_demand: float = 0.4  # [mg/L as Cl2 equivalent]

    # Temperature
    temperature: float = 20.0  # [°C]
    enable_thermal_stratification: bool = True

    # Inlet conditions
    inlet_pH: float = 7.5
    inlet_chlorine: float = 0.0  # [mg/L]
    inlet_chloramine: float = 0.0  # [mg/L as Cl2]
    inlet_ammonia: float = 0.2  # [mg/L as N]
    inlet_chlorine_demand: float = 0.6  # [mg/L as Cl2 equivalent]
    inlet_temperature: float = 20.0  # [°C]

    def validate(self) -> None:
        """Validate configuration consistency."""
        # Check volume matches geometry
        calculated_volume = np.pi * (self.diameter / 2) ** 2 * self.height * 1000
        volume_error = abs(calculated_volume - self.volume) / self.volume
        if volume_error > 0.01:  # 1% tolerance
            raise ValueError(
                f"Volume mismatch: specified {self.volume}L, "
                f"calculated {calculated_volume:.1f}L from geometry. "
                f"Error: {volume_error*100:.1f}%"
            )

        # Check reasonable parameter ranges
        assert 0 < self.volume < 1e6, "Volume out of range"
        assert (
            0 <= self.flow_rate < 1e5
        ), "Flow rate out of range (use 0 for batch mode)"
        assert 0 <= self.initial_pH <= 14, "pH out of range"
        assert 0 <= self.initial_chlorine <= 10, "Chlorine out of range"
        assert 0 <= self.initial_chloramine <= 20, "Chloramine out of range"
        assert 0 <= self.initial_ammonia <= 20, "Ammonia out of range"
        assert 0 <= self.initial_chlorine_demand <= 20, "Chlorine demand out of range"
        assert 0 <= self.temperature <= 40, "Temperature out of typical range"


@dataclass
class ReactorState:
    """
    Complete state of the reactor at a point in time.

    All state variables are arrays of length n_zones.
    """

    time: float = 0.0  # [s]

    # Primary state variables (per zone)
    pH: np.ndarray = field(default_factory=lambda: np.full(5, 7.0))
    chlorine: np.ndarray = field(
        default_factory=lambda: np.full(5, 2.0)
    )  # [mg/L as Cl2]
    chloramine: np.ndarray = field(
        default_factory=lambda: np.full(5, 0.0)
    )  # [mg/L as Cl2]
    ammonia: np.ndarray = field(default_factory=lambda: np.full(5, 0.15))  # [mg/L as N]
    chlorine_demand: np.ndarray = field(
        default_factory=lambda: np.full(5, 0.4)
    )  # [mg/L as Cl2 equivalent]
    temperature: np.ndarray = field(default_factory=lambda: np.full(5, 20.0))  # [°C]
    flow_rate: float = 5.0  # [L/min] Current total flow through reactor

    # Derived quantities (updated from primary variables)
    H_concentration: np.ndarray = field(init=False)  # [mol/L]
    density: np.ndarray = field(init=False)  # [kg/m³]
    chlorine_decay_rate: np.ndarray = field(init=False)  # [1/s]

    def __post_init__(self):
        """Initialize derived quantities."""
        self.update_derived()

    def update_derived(self):
        """Recalculate all derived quantities from primary variables."""
        self.H_concentration = 10 ** (-self.pH)

        # Placeholder for density and decay rate
        # (will be properly calculated by reactor)
        if not hasattr(self, "density"):
            self.density = np.full_like(self.pH, 998.2)
        if not hasattr(self, "chlorine_decay_rate"):
            self.chlorine_decay_rate = np.full_like(self.pH, 0.0001)


@dataclass
class BoundaryConditions:
    """
    Physical boundary conditions and forcing functions for the reactor.

    These fields represent physical inlet streams, not controller intent.
    They are the flows and concentrations applied to the reactor model.

    Think of this as: "What physical streams are actually flowing into the tank?"
    NOT: "What should the controller do?"

    Example control → physics flow:
    - Control system decides: "Add 0.5 L/min of acid"
    - Physics receives: BoundaryConditions(acid_dose_flow=0.5, acid_dose_concentration=0.1)
    - Physics calculates: Effect on pH given this physical acid stream
    """

    # Main process inlet stream
    inlet_flow_rate: float = 5.0  # [L/min] Main feed stream
    inlet_pH: float = 7.5
    inlet_chlorine: float = 0.0  # [mg/L]
    inlet_chloramine: float = 0.0  # [mg/L as Cl2]
    inlet_ammonia: float = 0.2  # [mg/L as N]
    inlet_chlorine_demand: float = 0.6  # [mg/L as Cl2 equivalent]
    inlet_temperature: float = 20.0  # [°C]

    # Chemical dosing streams (physical flows resulting from control actions)
    # Naming: "dose" emphasizes these are actual chemical additions, not setpoints
    acid_flow_rate: float = 0.0  # [L/min] Acid solution feed rate
    acid_concentration: float = (
        0.1  # [mol/L] Acid solution concentration (e.g., HCl, H₂SO₄)
    )
    acid_temperature: float = 20.0  # [°C]

    chlorine_flow_rate: float = 0.0  # [L/min] Chlorine solution feed rate
    chlorine_concentration: float = 50.0  # [mg/L] Chlorine solution concentration
    chlorine_solution_pH: float = (
        11.5  # [-] sodium hypochlorite stock is strongly basic
    )
    chlorine_temperature: float = 20.0  # [°C]

    # Environmental forcing
    ambient_temperature: float = 20.0  # [°C] Surroundings temperature
    heat_loss_coefficient: float = 0.0  # [W/K] (0 = adiabatic, no heat loss)
