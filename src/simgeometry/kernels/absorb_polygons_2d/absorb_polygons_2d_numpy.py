"""The NumPy version of the ``absorb_polygons_2d`` kernel, run when Pyccel is unavailable.

Same arguments and result as ``absorb_polygons_2d_pyccel.py``, vectorized over
the markers (loops over the objects and their edges).
"""

from __future__ import annotations

import numpy as np

from simgeometry.kernels.polygon_hits import first_hits


def absorb_polygons_2d(
    positions: np.ndarray,
    start: np.ndarray,
    rows: np.ndarray,
    use_start: int,
    sweep_length: float,
    vertices: np.ndarray,
    polygon_offsets: np.ndarray,
    object_first: np.ndarray,
    object_last: np.ndarray,
    object_inverted: np.ndarray,
    object_lower: np.ndarray,
    object_upper: np.ndarray,
    object_boxed: np.ndarray,
    alive: np.ndarray,
    hit: np.ndarray,
) -> None:
    """Absorb the alive markers ``rows`` that hit an object; ``hit[i]`` is its index."""
    x1, y1 = positions[0, rows], positions[1, rows]
    if use_start == 1:
        x0, y0 = start[0, rows], start[1, rows]
        sweep = (sweep_length > 0.0) & (
            (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) <= sweep_length * sweep_length
        )
    else:
        x0, y0 = x1, y1
        sweep = np.zeros(rows.shape[0], dtype=bool)
    live = alive[rows]
    tested = np.empty((object_first.shape[0], rows.shape[0]), dtype=bool)
    for obj in range(object_first.shape[0]):
        tested[obj] = live
        if object_boxed[obj] == 1:
            tested[obj] &= ~(
                (np.maximum(x0, x1) < object_lower[obj, 0])
                | (np.minimum(x0, x1) > object_upper[obj, 0])
                | (np.maximum(y0, y1) < object_lower[obj, 1])
                | (np.minimum(y0, y1) > object_upper[obj, 1])
            )
    hit[...] = first_hits(
        vertices,
        polygon_offsets,
        object_first,
        object_last,
        object_inverted,
        tested,
        x0,
        y0,
        x1,
        y1,
        sweep,
    )
    alive[rows[hit >= 0]] = False
