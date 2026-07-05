"""Device-to-CFD coupling contracts and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import ScalarField
from .mesh import StructuredMesh
from .unit_processes import UnitProcess, unit_process_catalog

RegionSelector = str
Aggregation = str

VALID_REGION_SELECTORS = {"all", "inlet", "outlet", "center", "explicit_cells"}
VALID_AGGREGATIONS = {"mean", "minimum", "maximum", "point"}


@dataclass(frozen=True)
class SamplingRegion:
    """A spatial sampling region for a simulated sensor."""

    region_id: str
    selector: RegionSelector
    cell_ids: tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.region_id:
            raise ValueError("region_id is required")
        if self.selector not in VALID_REGION_SELECTORS:
            raise ValueError(f"{self.region_id}: unsupported selector {self.selector}")
        if self.selector == "explicit_cells" and not self.cell_ids:
            raise ValueError(f"{self.region_id}: explicit cells are required")

    def resolve(self, mesh: StructuredMesh) -> tuple[tuple[int, int, int], ...]:
        self.validate()
        if self.selector == "all":
            return tuple(
                (i, j, k)
                for k in range(mesh.nz)
                for j in range(mesh.ny)
                for i in range(mesh.nx)
            )
        if self.selector == "inlet":
            return tuple((0, j, k) for k in range(mesh.nz) for j in range(mesh.ny))
        if self.selector == "outlet":
            return tuple(
                (mesh.nx - 1, j, k) for k in range(mesh.nz) for j in range(mesh.ny)
            )
        if self.selector == "center":
            return ((mesh.nx // 2, mesh.ny // 2, mesh.nz // 2),)
        return tuple(mesh.indices_from_cell_id(cell_id) for cell_id in self.cell_ids)


@dataclass(frozen=True)
class SensorCouplingContract:
    sensor_tag: str
    unit_id: str
    variable: str
    units: str
    sampling_region: SamplingRegion
    aggregation: Aggregation = "mean"
    sample_line_delay_seconds: float = 0.0
    evidence_status: str = "simulated_metadata"

    def validate(self) -> None:
        if not self.sensor_tag:
            raise ValueError("sensor_tag is required")
        if not self.unit_id:
            raise ValueError(f"{self.sensor_tag}: unit_id is required")
        if not self.variable:
            raise ValueError(f"{self.sensor_tag}: variable is required")
        if not self.units:
            raise ValueError(f"{self.sensor_tag}: units are required")
        if self.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(f"{self.sensor_tag}: unsupported aggregation")
        if self.sample_line_delay_seconds < 0.0:
            raise ValueError(f"{self.sensor_tag}: delay cannot be negative")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.sensor_tag}: evidence status must be simulated")
        self.sampling_region.validate()


@dataclass(frozen=True)
class ActuatorCouplingContract:
    actuator_tag: str
    unit_id: str
    target_condition_id: str
    manipulated_variable: str
    target_region: SamplingRegion
    minimum_value: float
    maximum_value: float
    units: str
    evidence_status: str = "simulated_metadata"

    def validate(self) -> None:
        if not self.actuator_tag:
            raise ValueError("actuator_tag is required")
        if not self.unit_id:
            raise ValueError(f"{self.actuator_tag}: unit_id is required")
        if not self.target_condition_id:
            raise ValueError(f"{self.actuator_tag}: target condition is required")
        if not self.manipulated_variable:
            raise ValueError(f"{self.actuator_tag}: manipulated variable is required")
        if self.minimum_value > self.maximum_value:
            raise ValueError(f"{self.actuator_tag}: invalid value bounds")
        if not self.units:
            raise ValueError(f"{self.actuator_tag}: units are required")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.actuator_tag}: evidence status must be simulated")
        self.target_region.validate()


def sample_scalar_region(
    mesh: StructuredMesh,
    scalar: ScalarField,
    region: SamplingRegion,
    *,
    aggregation: Aggregation = "mean",
) -> float:
    """Sample a scalar field over a declared spatial region."""

    scalar.validate(mesh)
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    indices = region.resolve(mesh)
    values = np.array([scalar.values[k, j, i] for i, j, k in indices], dtype=np.float64)
    if aggregation == "mean":
        return float(values.mean())
    if aggregation == "minimum":
        return float(values.min())
    if aggregation == "maximum":
        return float(values.max())
    return float(values[0])


def apply_scalar_source_region(
    mesh: StructuredMesh,
    scalar: ScalarField,
    region: SamplingRegion,
    *,
    strength: float,
    dt: float,
) -> ScalarField:
    """Apply a bounded source term to a scalar over a region."""

    if dt < 0.0:
        raise ValueError("dt cannot be negative")
    scalar.validate(mesh)
    values = np.array(scalar.values, copy=True)
    indices = region.resolve(mesh)
    increment = strength * dt / max(len(indices), 1)
    for i, j, k in indices:
        values[k, j, i] += increment
    if scalar.nonnegative:
        values = np.maximum(values, 0.0)
    return ScalarField(
        name=scalar.name,
        units=scalar.units,
        values=values,
        nonnegative=scalar.nonnegative,
    )


def sensor_contracts_for_unit(unit: UnitProcess) -> tuple[SensorCouplingContract, ...]:
    contracts: list[SensorCouplingContract] = []
    for point in unit.instrumentation:
        for variable in point.measures:
            selector = "outlet" if "outlet" in point.sample_region else "center"
            contracts.append(
                SensorCouplingContract(
                    sensor_tag=point.tag,
                    unit_id=unit.unit_id,
                    variable=variable,
                    units="process_units",
                    sampling_region=SamplingRegion(point.sample_region, selector),
                )
            )
    return tuple(contracts)


def actuator_contracts_for_unit(
    unit: UnitProcess,
) -> tuple[ActuatorCouplingContract, ...]:
    contracts: list[ActuatorCouplingContract] = []
    for boundary in unit.boundaries:
        if boundary.kind != "chemical_source":
            continue
        contracts.append(
            ActuatorCouplingContract(
                actuator_tag=f"ACT-{unit.unit_id.upper()}-{boundary.name.upper()}",
                unit_id=unit.unit_id,
                target_condition_id="bc-dosing-injection",
                manipulated_variable=boundary.variables[0],
                target_region=SamplingRegion(boundary.name, "center"),
                minimum_value=0.0,
                maximum_value=100.0,
                units="process_units_per_second",
            )
        )
    return tuple(contracts)


def device_coupling_catalog() -> (
    tuple[SensorCouplingContract | ActuatorCouplingContract, ...]
):
    contracts: list[SensorCouplingContract | ActuatorCouplingContract] = []
    for unit in unit_process_catalog():
        contracts.extend(sensor_contracts_for_unit(unit))
        contracts.extend(actuator_contracts_for_unit(unit))
    return tuple(contracts)
