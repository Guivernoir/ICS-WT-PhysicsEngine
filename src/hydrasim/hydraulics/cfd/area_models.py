"""Reference water-treatment CFD area models."""

from __future__ import annotations

from dataclasses import dataclass

from .digital_twin import DigitalTwinMetadata, TwinStatus, UncertaintyRecord
from .mesh import BoundaryPatch, Obstacle, StructuredMesh, create_rectangular_mesh

AREA_IDS = (
    "intake",
    "dosing",
    "clarification",
    "filtration",
    "disinfection",
    "storage-pumping",
)


@dataclass(frozen=True)
class AreaCfdModel:
    area_id: str
    mesh: StructuredMesh
    scalar_names: tuple[str, ...]
    twin_metadata: DigitalTwinMetadata
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if self.area_id not in AREA_IDS:
            raise ValueError(f"unsupported area id: {self.area_id}")
        if not self.scalar_names:
            raise ValueError(f"{self.area_id}: scalar_names cannot be empty")
        if not self.limitations:
            raise ValueError(f"{self.area_id}: limitations are required")
        self.twin_metadata.validate()


def _base_boundaries(inlet_velocity: float) -> tuple[BoundaryPatch, ...]:
    return (
        BoundaryPatch("inlet", "xmin", "inlet", inlet_velocity),
        BoundaryPatch("outlet", "xmax", "outlet", 0.0),
        BoundaryPatch("floor", "zmin", "wall", 0.0),
        BoundaryPatch("surface", "zmax", "symmetry", 0.0),
        BoundaryPatch("left-wall", "ymin", "wall", 0.0),
        BoundaryPatch("right-wall", "ymax", "wall", 0.0),
    )


def _metadata(area_id: str) -> DigitalTwinMetadata:
    return DigitalTwinMetadata(
        name=f"reference-water-{area_id}",
        geometry_reference=f"synthetic-profile:{area_id}",
        equipment_reference=f"simulated-equipment:{area_id}",
        status=TwinStatus.SYNTHETIC_UNVALIDATED,
        uncertainty=(
            UncertaintyRecord(
                quantity="hydraulic_fidelity",
                absolute_bound=1.0,
                basis="synthetic geometry without site calibration",
            ),
        ),
    )


def reference_area_model(area_id: str) -> AreaCfdModel:
    """Return a small deterministic CFD area model for the reference plant."""

    if area_id not in AREA_IDS:
        raise ValueError(f"unsupported area id: {area_id}")

    area_shapes = {
        "intake": ((12, 4, 3), (6.0, 2.0, 1.5), (), ("temperature", "turbidity")),
        "dosing": (
            (10, 4, 3),
            (5.0, 1.5, 1.2),
            (),
            ("chlorine", "ph_proxy", "demand_precursor"),
        ),
        "clarification": (
            (14, 6, 3),
            (8.0, 3.0, 1.8),
            (Obstacle("settling-baffle", 6, 7, 1, 5, 0, 2),),
            ("turbidity", "temperature"),
        ),
        "filtration": (
            (10, 5, 4),
            (4.0, 2.0, 1.8),
            (Obstacle("filter-media", 3, 8, 1, 4, 0, 2),),
            ("headloss_proxy", "turbidity"),
        ),
        "disinfection": (
            (16, 4, 3),
            (10.0, 2.0, 1.5),
            (Obstacle("contact-baffle", 7, 8, 0, 3, 0, 2),),
            ("chlorine", "chloramine", "ammonia", "ph_proxy"),
        ),
        "storage-pumping": (
            (12, 5, 3),
            (7.0, 3.0, 2.0),
            (),
            ("chlorine", "temperature"),
        ),
    }
    cells, extents, obstacles, scalars = area_shapes[area_id]
    mesh = create_rectangular_mesh(
        cells=cells,
        extents=extents,
        boundaries=_base_boundaries(inlet_velocity=0.04),
        obstacles=obstacles,
    )
    model = AreaCfdModel(
        area_id=area_id,
        mesh=mesh,
        scalar_names=scalars,
        twin_metadata=_metadata(area_id),
        limitations=(
            "synthetic CFD geometry",
            "uncalibrated digital-twin parameters",
            "not commissioning or plant-design evidence",
        ),
    )
    model.validate()
    return model
