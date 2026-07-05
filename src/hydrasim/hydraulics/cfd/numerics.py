"""Small deterministic numerical helpers for CFD primitives."""

from __future__ import annotations

import numpy as np


def laplacian(values: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    center = padded[1:-1, 1:-1, 1:-1]
    x_term = padded[1:-1, 1:-1, 2:] - 2.0 * center + padded[1:-1, 1:-1, :-2]
    y_term = padded[1:-1, 2:, 1:-1] - 2.0 * center + padded[1:-1, :-2, 1:-1]
    z_term = padded[2:, 1:-1, 1:-1] - 2.0 * center + padded[:-2, 1:-1, 1:-1]
    return (x_term / (dx * dx)) + (y_term / (dy * dy)) + (z_term / (dz * dz))


def divergence(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    dudx = np.gradient(u, dx, axis=2, edge_order=1)
    dvdy = np.gradient(v, dy, axis=1, edge_order=1)
    dwdz = np.gradient(w, dz, axis=0, edge_order=1)
    return dudx + dvdy + dwdz


def gradient(
    values: np.ndarray, dx: float, dy: float, dz: float
) -> tuple[np.ndarray, ...]:
    dz_grad, dy_grad, dx_grad = np.gradient(values, dz, dy, dx, edge_order=1)
    return dx_grad, dy_grad, dz_grad
