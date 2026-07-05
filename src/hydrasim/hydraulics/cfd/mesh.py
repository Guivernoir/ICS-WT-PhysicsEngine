"""Structured finite-volume mesh primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FaceName = str
BoundaryKind = str

VALID_FACES = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
VALID_BOUNDARY_KINDS = {"wall", "inlet", "outlet", "symmetry", "source"}


@dataclass(frozen=True)
class BoundaryPatch:
    """A named boundary on one rectangular mesh face."""

    name: str
    face: FaceName
    kind: BoundaryKind
    value: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("boundary name is required")
        if self.face not in VALID_FACES:
            raise ValueError(f"unsupported boundary face: {self.face}")
        if self.kind not in VALID_BOUNDARY_KINDS:
            raise ValueError(f"unsupported boundary kind: {self.kind}")


@dataclass(frozen=True)
class Obstacle:
    """Inclusive-exclusive blocked cell index range."""

    name: str
    i_min: int
    i_max: int
    j_min: int
    j_max: int
    k_min: int
    k_max: int

    def validate(self, mesh: "StructuredMesh") -> None:
        if not self.name:
            raise ValueError("obstacle name is required")
        checks = (
            0 <= self.i_min < self.i_max <= mesh.nx,
            0 <= self.j_min < self.j_max <= mesh.ny,
            0 <= self.k_min < self.k_max <= mesh.nz,
        )
        if not all(checks):
            raise ValueError(f"obstacle {self.name!r} is outside mesh bounds")


@dataclass(frozen=True)
class StructuredMesh:
    """A deterministic Cartesian cell-centered finite-volume mesh."""

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    boundaries: tuple[BoundaryPatch, ...] = field(default_factory=tuple)
    obstacles: tuple[Obstacle, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) <= 0:
            raise ValueError("mesh dimensions must be positive")
        if min(self.lx, self.ly, self.lz) <= 0.0:
            raise ValueError("mesh extents must be positive")
        for boundary in self.boundaries:
            boundary.validate()
        for obstacle in self.obstacles:
            obstacle.validate(self)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz

    def cell_id(self, i: int, j: int, k: int) -> int:
        if not (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            raise ValueError("cell index is outside mesh bounds")
        return (k * self.ny * self.nx) + (j * self.nx) + i

    def indices_from_cell_id(self, cell_id: int) -> tuple[int, int, int]:
        if not (0 <= cell_id < self.cell_count):
            raise ValueError("cell id is outside mesh bounds")
        k, rem = divmod(cell_id, self.nx * self.ny)
        j, i = divmod(rem, self.nx)
        return (i, j, k)

    def cell_center(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        self.cell_id(i, j, k)
        return (
            (i + 0.5) * self.dx,
            (j + 0.5) * self.dy,
            (k + 0.5) * self.dz,
        )

    def active_mask(self) -> np.ndarray:
        mask = np.ones(self.shape, dtype=bool)
        for obstacle in self.obstacles:
            mask[
                obstacle.k_min : obstacle.k_max,
                obstacle.j_min : obstacle.j_max,
                obstacle.i_min : obstacle.i_max,
            ] = False
        return mask


def create_rectangular_mesh(
    *,
    cells: tuple[int, int, int],
    extents: tuple[float, float, float],
    boundaries: tuple[BoundaryPatch, ...] = (),
    obstacles: tuple[Obstacle, ...] = (),
) -> StructuredMesh:
    """Create a rectangular mesh using `(nx, ny, nz)` and `(lx, ly, lz)`."""

    nx, ny, nz = cells
    lx, ly, lz = extents
    return StructuredMesh(
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
        boundaries=boundaries,
        obstacles=obstacles,
    )
