"""Subcell quadrature for approximate solid volume and face-area fractions."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.operations import mask


@dataclass(frozen=True)
class CellFractions:
    """Sampled solid fractions in [0, 1]; fluid fractions are one minus these.

    ``volume`` has one entry per cell. ``faces[axis]`` has one extra entry along
    that axis, including both periodic boundary copies. Fraction times the cell
    volume/face area gives the corresponding solid measure. This is midpoint
    quadrature, not a reconstructed interface; sub-sample features can be missed.
    """

    volume: np.ndarray
    faces: tuple[np.ndarray, ...]
    samples_per_axis: int


@sp.profile("simgeometry.cell_fractions", tags=_PROFILE_TAGS)
def cell_fractions(geometry, grid, *, samples_per_axis=4):
    """Estimate cell and face occupancy using a regular subcell sample lattice."""
    if geometry.ndim != grid.ndim:
        raise ValueError("Geometry and grid dimensions must match")
    if int(samples_per_axis) != samples_per_axis or samples_per_axis < 1:
        raise ValueError("samples_per_axis must be a positive integer")
    n = int(samples_per_axis)
    axes = [np.asarray(a, float) for a in grid.get_grid_points()]
    periodic = getattr(grid, "periodic", None) or (False,) * grid.ndim
    h = np.asarray(grid.grid_spacing)
    lower = np.asarray(grid.lower_bound)
    upper = np.asarray(grid.upper_bound)
    cell_counts = tuple(
        len(a) if flag else len(a) - 1 for a, flag in zip(axes, periodic)
    )
    starts = [a if flag else a[:-1] for a, flag in zip(axes, periodic)]
    offsets = (np.arange(n) + 0.5) / n

    def sample(base_axes, fixed=None):
        result = np.zeros(tuple(len(a) for a in base_axes))
        base = np.stack([a.ravel() for a in np.meshgrid(*base_axes, indexing="ij")])
        combinations = list(
            product(*([0.0] if axis == fixed else offsets for axis in range(grid.ndim)))
        )
        for delta in combinations:
            points = base + h[:, None] * np.asarray(delta)[:, None]
            for axis, flag in enumerate(periodic):
                if flag:
                    points[axis] = lower[axis] + (points[axis] - lower[axis]) % (
                        upper[axis] - lower[axis]
                    )
            result += mask(geometry, points).reshape(result.shape)
        return result / len(combinations)

    volume = sample(starts).reshape(cell_counts)
    faces = []
    for axis in range(grid.ndim):
        face_axes = list(starts)
        face_axes[axis] = (
            np.r_[axes[axis], upper[axis]] if periodic[axis] else axes[axis]
        )
        faces.append(sample(face_axes, fixed=axis))
    return CellFractions(volume, tuple(faces), n)
